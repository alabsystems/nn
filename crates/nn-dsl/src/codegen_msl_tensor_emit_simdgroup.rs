// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL emission for simdgroup-tiled GEMM kernels in the compiled pipeline.
//!
//! Uses `simdgroup_matrix<T, 8, 8>` hardware cooperative multiply-accumulate
//! with 32×32 output tiles, 128 threads per threadgroup (4 SIMD groups of 32).
//! Dimensions M, N, K are embedded as compile-time MSL constants (known at
//! dispatch plan generation time), eliminating runtime constant buffer binding.
//!
//! Part of #2275.

use crate::codegen_msl;
use crate::codegen_msl_structural;
use crate::codegen_msl_tensor::TensorMSLCodegenError;
use crate::ir::ScalarType;
use crate::trace_compile::GemmActivation;

/// Emit a simdgroup-tiled Linear kernel.
///
/// Weight is `[out_features, in_features]` (row-major), read transposed by
/// the kernel. Buffer layout: input(0), weight(1), [bias(2),] output(2 or 3).
pub(super) fn emit_simdgroup_linear_kernel(
    name: &str,
    dtype: ScalarType,
    in_features: usize,
    out_features: usize,
    batch_size: usize,
    has_bias: bool,
) -> Result<String, TensorMSLCodegenError> {
    // Linear: A=[batch, in_feat], B=[out_feat, in_feat]^T => C=[batch, out_feat]
    // M=batch_size, K=in_features, N=out_features, transpose_b=true
    emit_simdgroup_gemm_kernel(
        name,
        dtype,
        batch_size,
        in_features,
        out_features,
        1, // batch_count=1 (batch flattened into M)
        true,
        false,
        has_bias,
        None,
        None,
    )
}

/// Emit a simdgroup-tiled Linear+Activation kernel (standalone MSL).
///
/// Identical to [`emit_simdgroup_linear_kernel`] but applies the given
/// activation in the write-back epilogue. Includes the full MSL prelude
/// (`#include <metal_stdlib>`, `#include <metal_simdgroup_matrix>`,
/// `using namespace metal;`) so it can be compiled standalone by
/// `KernelPipeline::from_msl` without the plan-level prelude.
///
/// Part of #2256 D4.
pub fn emit_simdgroup_linear_activation_kernel(
    name: &str,
    dtype: ScalarType,
    in_features: usize,
    out_features: usize,
    batch_size: usize,
    has_bias: bool,
    activation: &GemmActivation,
) -> Result<String, TensorMSLCodegenError> {
    let body = emit_simdgroup_gemm_kernel(
        name,
        dtype,
        batch_size,
        in_features,
        out_features,
        1,
        true,
        false,
        has_bias,
        None,
        Some(activation),
    )?;
    Ok(format!(
        "#include <metal_stdlib>\n\
         #include <metal_simdgroup_matrix>\n\
         using namespace metal;\n\n\
         {body}"
    ))
}

/// Emit a simdgroup-tiled Linear kernel (standalone MSL, no activation).
///
/// Includes the full MSL prelude so it can be compiled standalone by
/// `KernelPipeline::from_msl`. Used by NormLinear two-dispatch path
/// (Phase 3 GEMM after separate norm dispatch).
///
/// Buffer layout: input(0), weight(1), [bias(2),] output(2 or 3).
/// Weight is `[out_features, in_features]` (row-major), read transposed.
///
/// Part of #3292.
pub fn emit_simdgroup_linear_standalone_kernel(
    name: &str,
    dtype: ScalarType,
    in_features: usize,
    out_features: usize,
    batch_size: usize,
    has_bias: bool,
) -> Result<String, TensorMSLCodegenError> {
    let body = emit_simdgroup_gemm_kernel(
        name,
        dtype,
        batch_size,
        in_features,
        out_features,
        1,
        true,
        false,
        has_bias,
        None,
        None,
    )?;
    Ok(format!(
        "#include <metal_stdlib>\n\
         #include <metal_simdgroup_matrix>\n\
         using namespace metal;\n\n\
         {body}"
    ))
}

/// Emit a simdgroup-tiled MatMul kernel.
///
/// Buffer layout: left(0), right(1), output(2).
pub(super) fn emit_simdgroup_matmul_kernel(
    name: &str,
    dtype: ScalarType,
    m: usize,
    k: usize,
    n: usize,
    batch_size: usize,
    transpose_right: bool,
    broadcast_right: bool,
    scale: Option<f32>,
) -> Result<String, TensorMSLCodegenError> {
    emit_simdgroup_gemm_kernel(
        name,
        dtype,
        m,
        k,
        n,
        batch_size,
        transpose_right,
        broadcast_right,
        false,
        scale,
        None,
    )
}

/// Core simdgroup GEMM MSL emitter, parameterized for both Linear and MatMul.
///
/// Generates a complete MSL kernel function using `simdgroup_matrix<T, 8, 8>`
/// with 32×32 output tiles. All dimensions are compile-time MSL constants.
#[allow(clippy::too_many_arguments)]
fn emit_simdgroup_gemm_kernel(
    name: &str,
    dtype: ScalarType,
    m: usize,
    k: usize,
    n: usize,
    batch_count: usize,
    transpose_b: bool,
    broadcast_b: bool,
    has_bias: bool,
    scale: Option<f32>,
    activation: Option<&GemmActivation>,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let is_half = t == "half";
    // Shared memory element type matches buffer type; accumulators are always float.
    let shared_t = t;
    let operand_t = t;
    let zero = if is_half { "half(0.0h)" } else { "0.0f" };

    let m_val = codegen_msl_structural::safe_msl_uint(m)?;
    let k_val = codegen_msl_structural::safe_msl_uint(k)?;
    let n_val = codegen_msl_structural::safe_msl_uint(n)?;
    let batch_val = codegen_msl_structural::safe_msl_uint(batch_count)?;

    // Buffer binding indices depend on bias presence.
    let (bias_param, out_buf_idx) = if has_bias {
        (
            format!("    device const {t}* bias   [[buffer(2)]],\n"),
            "3",
        )
    } else {
        (String::new(), "2")
    };

    // B offset: broadcast=0, else batch_idx * K * N.
    let b_batch_stride = codegen_msl_structural::safe_msl_uint(k * n)?;
    let b_offset_expr = if broadcast_b {
        "0".to_string()
    } else {
        format!("batch_idx * ({b_batch_stride})")
    };

    // B tile load expression: normal or transposed.
    // Constants are prefixed with kernel name to avoid redefinition when multiple
    // simdgroup kernels are concatenated in a single MSL compilation unit (#2894).
    let b_load = if transpose_b {
        // B stored as [N, K], read B[gc, gr] where gc=n-col, gr=k-row.
        format!("(gr < {name}_K_DIM && gc < {name}_N_DIM) ? B[b_offset + gc * {name}_K_DIM + gr] : {zero}")
    } else {
        // B stored as [K, N], read B[gr, gc] where gr=k-row, gc=n-col.
        format!("(gr < {name}_K_DIM && gc < {name}_N_DIM) ? B[b_offset + gr * {name}_N_DIM + gc] : {zero}")
    };

    // Output write: optional bias add and scale multiply.
    let bias_add = if has_bias {
        if is_half {
            "            val += float(bias[gc]);\n".to_string()
        } else {
            "            val += bias[gc];\n".to_string()
        }
    } else {
        String::new()
    };
    let scale_mul = match scale {
        Some(s) => format!("            val *= float({s});\n"),
        None => String::new(),
    };
    // Optional activation applied to accumulator (val is always float).
    let activation_apply = match activation {
        Some(act) => super::gemm::gemm_activation_msl_var(act, "float", "val", "            "),
        None => String::new(),
    };

    // Final store: cast float accumulator back to storage type if half.
    let store_expr = if is_half {
        "half(val)".to_string()
    } else {
        "val".to_string()
    };

    // Per-kernel prefixed constants avoid redefinition when multiple simdgroup
    // kernels are concatenated in one MSL compilation unit (#2894).
    // The `#include <metal_simdgroup_matrix>` is emitted once at plan level
    // by `emit_tensor_msl_with_plan()`.
    Ok(format!(
        r#"constant uint {name}_TILE = 32;
constant uint {name}_SIMD_SIZE = 32;
constant uint {name}_PADDED = {name}_TILE + 1;
constant uint {name}_M_DIM = {m_val};
constant uint {name}_K_DIM = {k_val};
constant uint {name}_N_DIM = {n_val};
constant uint {name}_BATCH_COUNT = {batch_val};

kernel void {name}(
    device const {t}* A       [[buffer(0)]],
    device const {t}* B       [[buffer(1)]],
{bias_param}    device {t}* C             [[buffer({out_buf_idx})]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]]
) {{
    uint batch_idx = tgid.z;
    if (batch_idx >= {name}_BATCH_COUNT) return;

    uint a_offset = batch_idx * {name}_M_DIM * {name}_K_DIM;
    uint b_offset = {b_offset_expr};
    uint c_offset = batch_idx * {name}_M_DIM * {name}_N_DIM;

    uint tile_row = tgid.y * {name}_TILE;
    uint tile_col = tgid.x * {name}_TILE;

    threadgroup {shared_t} As[{name}_TILE * {name}_PADDED];
    threadgroup {shared_t} Bs[{name}_TILE * {name}_PADDED];
    threadgroup float tile_out[{name}_TILE * {name}_PADDED];

    uint sg_col_start = sg_id * 8;

    simdgroup_matrix<float, 8, 8> acc[4];
    for (uint i = 0; i < 4; i++) {{
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);
    }}

    uint tid_linear = sg_id * {name}_SIMD_SIZE + lane_id;
    uint num_k_tiles = ({name}_K_DIM + {name}_TILE - 1) / {name}_TILE;

    for (uint kt = 0; kt < num_k_tiles; kt++) {{
        uint k_start = kt * {name}_TILE;

        // Cooperative load A tile [TILE x TILE] into shared memory.
        for (uint idx = tid_linear; idx < {name}_TILE * {name}_TILE; idx += 128) {{
            uint row = idx / {name}_TILE;
            uint col = idx % {name}_TILE;
            uint gr = tile_row + row;
            uint gc = k_start + col;
            {shared_t} val = (gr < {name}_M_DIM && gc < {name}_K_DIM) ? A[a_offset + gr * {name}_K_DIM + gc] : {zero};
            As[row * {name}_PADDED + col] = val;
        }}

        // Cooperative load B tile [TILE x TILE] into shared memory.
        for (uint idx = tid_linear; idx < {name}_TILE * {name}_TILE; idx += 128) {{
            uint row = idx / {name}_TILE;
            uint col = idx % {name}_TILE;
            uint gr = k_start + row;
            uint gc = tile_col + col;
            {shared_t} bval = {b_load};
            Bs[row * {name}_PADDED + col] = bval;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // SIMD multiply-accumulate over K in 8-wide steps.
        for (uint kk = 0; kk < {name}_TILE; kk += 8) {{
            simdgroup_matrix<{operand_t}, 8, 8> Bmat;
            simdgroup_load(Bmat, &Bs[kk * {name}_PADDED + sg_col_start], {name}_PADDED);
            for (uint ri = 0; ri < 4; ri++) {{
                simdgroup_matrix<{operand_t}, 8, 8> Amat;
                simdgroup_load(Amat, &As[(ri * 8) * {name}_PADDED + kk], {name}_PADDED);
                simdgroup_multiply_accumulate(acc[ri], Amat, Bmat, acc[ri]);
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Store accumulators to shared tile_out.
    for (uint ri = 0; ri < 4; ri++) {{
        simdgroup_store(acc[ri], &tile_out[(ri * 8) * {name}_PADDED + sg_col_start], {name}_PADDED);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperative write to global memory.
    for (uint idx = tid_linear; idx < {name}_TILE * {name}_TILE; idx += 128) {{
        uint r = idx / {name}_TILE;
        uint c = idx % {name}_TILE;
        uint gr = tile_row + r;
        uint gc = tile_col + c;
        if (gr < {name}_M_DIM && gc < {name}_N_DIM) {{
            float val = tile_out[r * {name}_PADDED + c];
{bias_add}{scale_mul}{activation_apply}            C[c_offset + gr * {name}_N_DIM + gc] = {store_expr};
        }}
    }}
}}"#,
    ))
}
