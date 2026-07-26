// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL emission for tensor-level dispatch steps.
//!
//! Extracted from `codegen_msl_tensor` to keep both files under the 500-line
//! limit. The parent module owns types (`DispatchStep`, `TensorMSLCodegenError`)
//! and dispatch planning (`build_dispatch_plan`); this module owns MSL string
//! generation (`emit_reduce_kernel`, `emit_tensor_msl`, etc.).

use crate::codegen_msl::{self, MSL_PRELUDE};
use crate::codegen_msl_structural;
use crate::codegen_msl_tensor::{TensorMSLCodegenError, REDUCE_THREADGROUP_SIZE};
use crate::ir::ScalarType;
use crate::precision::{PrecisionContract, PrecisionTier};
use crate::tensor_ir::{BroadcastAlignment, ReduceOp, TensorKernelDef};

// Per-op MSL emission helpers (binary, activation, padding).
#[path = "codegen_msl_tensor_emit_ops.rs"]
mod ops;

// Complex MSL emission (linear, matmul, softmax, embedding).
#[path = "codegen_msl_tensor_emit_complex.rs"]
mod complex;

// Fused GEMM + activation MSL emission (Part of #2256).
#[path = "codegen_msl_tensor_emit_gemm.rs"]
mod gemm;
pub use gemm::{emit_linear_activation_kernel, gemm_activation_msl_var};

// Index-based MSL emission (IndexSelect, Gather, f32→u32 conversion). Part of #2278.
#[path = "codegen_msl_tensor_emit_index.rs"]
mod index;

// Simdgroup-tiled GEMM MSL emission (Part of #2275).
#[path = "codegen_msl_tensor_emit_simdgroup.rs"]
mod simdgroup;
pub use simdgroup::emit_simdgroup_linear_activation_kernel;
pub use simdgroup::emit_simdgroup_linear_standalone_kernel;

// Tiled shared-memory GEMM MSL emission (Part of #3230 Gap 1).
#[path = "codegen_msl_tensor_emit_tiled.rs"]
pub(super) mod tiled;

// Per-step dispatch: matches a DispatchStep and returns its MSL source.
#[path = "codegen_msl_tensor_emit_step.rs"]
mod step;

/// Emit MSL source for a threadgroup-parallel reduction kernel.
///
/// Generates a Metal compute kernel that:
/// 1. Each thread accumulates a partial sum/mean over its stride
/// 2. Tree reduction in shared memory with threadgroup barriers
/// 3. Thread 0 writes the final result (divided by reduce_dim for Mean)
///
/// This kernel assumes the reduced axis is the last (contiguous) dimension in
/// row-major layout. `build_dispatch_plan`/`emit_tensor_msl` enforce this.
///
/// Buffer layout:
/// - `buffer(0)`: input data
/// - `buffer(1)`: output data
/// - `buffer(2)`: reduce_dim (uint)
/// - `buffer(3)`: outer_size (uint)
#[must_use]
pub(crate) fn emit_reduce_kernel(
    name: &str,
    op: ReduceOp,
    dtype: ScalarType,
    contract: PrecisionContract,
) -> String {
    let t = codegen_msl::msl_type(dtype);
    // Accumulate in f32 for f16/bf16 to avoid catastrophic precision loss (#1352).
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let needs_cast = t != acc;

    // Op-specific: identity element and phase-3 divisor.
    let (identity, mean_divisor) = match op {
        ReduceOp::Sum => (format!("{acc}(0)"), String::new()),
        ReduceOp::Mean => (format!("{acc}(0)"), format!(" / {acc}(reduce_dim)")),
        ReduceOp::Max => ("-INFINITY".to_string(), String::new()),
        ReduceOp::Min => ("INFINITY".to_string(), String::new()),
    };

    let is_extremum = matches!(op, ReduceOp::Max | ReduceOp::Min);

    // Load from input: cast to accumulator type if needed.
    let load_expr = if needs_cast {
        format!("{acc}(input[gid * reduce_dim + i])")
    } else {
        "input[gid * reduce_dim + i]".to_string()
    };

    // For Sum/Mean: Strict uses Kahan compensated summation for near-f64 precision (#1814).
    // Normal uses named intermediates to prevent FMA contraction.
    // Relaxed: raw `+=` (fast-math enabled, contraction desirable).
    // For Max/Min: always use fmax/fmin (no FMA concern).
    let strict_sum = matches!(contract.tier, PrecisionTier::Strict) && !is_extremum;
    let relaxed = matches!(contract.tier, PrecisionTier::Relaxed);
    let (accum_phase1, accum_phase2) = if is_extremum {
        let fn_name = if matches!(op, ReduceOp::Max) {
            "fmax"
        } else {
            "fmin"
        };
        (
            format!("        partial = {fn_name}(partial, {load_expr});"),
            format!("            shared[lid] = {fn_name}(shared[lid], shared[lid + stride]);"),
        )
    } else if strict_sum {
        // Kahan compensated summation: tracks running error in `comp` variable.
        // Provides ~2x mantissa bits of precision from f32 arithmetic (#1814).
        (
            format!(
                "        {acc} y = {load_expr} - comp;\n        \
                 {acc} t = partial + y;\n        \
                 comp = (t - partial) - y;\n        \
                 partial = t;"
            ),
            format!(
                "            {acc} a = shared[lid];\n            \
                 {acc} b = shared[lid + stride];\n            \
                 {acc} ca = shared_comp[lid];\n            \
                 {acc} cb = shared_comp[lid + stride];\n            \
                 {acc} y = b - (ca + cb);\n            \
                 {acc} t = a + y;\n            \
                 shared_comp[lid] = (t - a) - y;\n            \
                 shared[lid] = t;"
            ),
        )
    } else if relaxed {
        (
            format!("        partial += {load_expr};"),
            "            shared[lid] += shared[lid + stride];".to_string(),
        )
    } else {
        // Normal tier: named intermediates prevent FMA contraction.
        (
            format!("        {acc} val = {load_expr};\n        partial = partial + val;"),
            format!("            {acc} a = shared[lid];\n            {acc} b = shared[lid + stride];\n            shared[lid] = a + b;"),
        )
    };

    let phase1_comment = if is_extremum {
        "// Phase 1: Each thread computes partial extremum over its stride"
    } else {
        "// Phase 1: Each thread accumulates a partial sum over its stride"
    };

    // Cast accumulator result back to storage type on final write.
    let store_expr = if needs_cast {
        format!("{t}(shared[0]{mean_divisor})")
    } else {
        format!("shared[0]{mean_divisor}")
    };

    // Kahan-compensated reductions need a second shared memory array and
    // a per-thread compensation variable (#1814).
    let shared_comp_decl = if strict_sum {
        format!(
            "\n    threadgroup {acc} shared_comp[{REDUCE_THREADGROUP_SIZE}];"
        )
    } else {
        String::new()
    };
    let comp_init = if strict_sum {
        format!("\n    {acc} comp = {acc}(0);")
    } else {
        String::new()
    };
    let comp_store = if strict_sum {
        "\n    shared_comp[lid] = comp;"
    } else {
        ""
    };

    format!(
        r#"// precision: {tier}
[[kernel]] void {name}(
    device const {t}* input   [[buffer(0)]],
    device {t}* output        [[buffer(1)]],
    constant uint& reduce_dim [[buffer(2)]],
    constant uint& outer_size [[buffer(3)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]]
) {{
    if (gid >= outer_size) return;

    threadgroup {acc} shared[{tg_sz}];{shared_comp_decl}

    {phase1_comment}{comp_init}
    {acc} partial = {identity};
    for (uint i = lid; i < reduce_dim; i += tg_sz) {{
{accum_phase1}
    }}
    shared[lid] = partial;{comp_store}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Tree reduction in shared memory
    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {{
        if (lid < stride) {{
{accum_phase2}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Phase 3: Write result
    if (lid == 0) {{
        output[gid] = {store_expr};
    }}
}}"#,
        tier = contract.tier.as_str(),
        name = name,
        t = t,
        acc = acc,
        tg_sz = REDUCE_THREADGROUP_SIZE,
        identity = identity,
        phase1_comment = phase1_comment,
        comp_init = comp_init,
        comp_store = comp_store,
        shared_comp_decl = shared_comp_decl,
        accum_phase1 = accum_phase1,
        accum_phase2 = accum_phase2,
        store_expr = store_expr,
    )
}

/// Emit MSL source for a broadcast kernel (delegates to `codegen_msl_structural`).
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn emit_broadcast_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    output_shape: &[usize],
    alignment: BroadcastAlignment,
) -> Result<String, TensorMSLCodegenError> {
    codegen_msl_structural::emit_broadcast_kernel(name, dtype, input_shape, output_shape, alignment)
}

/// Emit all MSL source for a tensor kernel (Normal precision by default).
///
/// See [`emit_tensor_msl_with_contract`] for explicit precision control.
#[must_use = "returns a Result that may contain an error"]
pub fn emit_tensor_msl(
    kernel: &TensorKernelDef,
    dtype: ScalarType,
) -> Result<String, TensorMSLCodegenError> {
    emit_tensor_msl_with_contract(
        kernel,
        dtype,
        PrecisionContract::bootstrap(PrecisionTier::Normal, dtype),
    )
}

/// Emit all MSL source with an explicit precision contract.
#[must_use = "returns a Result that may contain an error"]
pub fn emit_tensor_msl_with_contract(
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    contract: PrecisionContract,
) -> Result<String, TensorMSLCodegenError> {
    let (plan, _effective_output, expanded_kernel) =
        crate::codegen_msl_tensor::build_dispatch_plan_full(kernel, dtype)?;
    emit_tensor_msl_with_plan(&plan, &expanded_kernel, contract)
}

/// Emit all MSL source from a pre-computed dispatch plan.
///
/// Use this when you already have the plan from [`build_dispatch_plan_full`] to
/// avoid computing it twice. [`emit_tensor_msl_with_contract`] calls
/// `build_dispatch_plan_full` internally — if the caller also needs the plan
/// (e.g., for GPU dispatch), use `build_dispatch_plan_full` once and pass the
/// result here.
#[must_use = "returns a Result that may contain an error"]
pub fn emit_tensor_msl_with_plan(
    plan: &[crate::codegen_msl_tensor::DispatchStep],
    expanded_kernel: &TensorKernelDef,
    contract: PrecisionContract,
) -> Result<String, TensorMSLCodegenError> {
    use crate::codegen_msl_tensor::DispatchStep;

    let mut msl = String::from(MSL_PRELUDE);

    // Emit `#include <metal_simdgroup_matrix>` once at plan level when any
    // step uses simdgroup matmul. Per-step emission was removed to prevent
    // redefinition errors in concatenated MSL (#2894).
    let has_simdgroup = plan.iter().any(|s| {
        matches!(
            s,
            DispatchStep::SimdgroupLinear(..) | DispatchStep::SimdgroupMatMul(..)
        )
    });
    if has_simdgroup {
        msl.push_str("#include <metal_simdgroup_matrix>\n\n");
    }

    for s in plan {
        if let Some(src) = step::emit_step_msl(s, expanded_kernel, contract)? {
            msl.push_str(&src);
            msl.push('\n');
            msl.push('\n');
        }
    }

    Ok(msl)
}
