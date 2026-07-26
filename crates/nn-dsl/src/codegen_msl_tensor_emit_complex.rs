// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL emission for complex tensor ops: linear, matmul, softmax, embedding.
//!
//! Extracted from `codegen_msl_tensor_emit_ops.rs` to keep that file under
//! the 500-line limit as new ops (Conv2d, GroupNorm, STFT) are added.
//! All functions are `pub(super)` — called from `codegen_msl_tensor_emit_step.rs`.

use crate::codegen_msl;
use crate::codegen_msl_structural;
use crate::codegen_msl_tensor::TensorMSLCodegenError;
use crate::ir::ScalarType;

/// Emit MSL source for a linear (fully-connected) layer kernel (no activation).
///
/// Thin wrapper over [`super::gemm::emit_linear_activation_kernel`] with no
/// activation and no MSL prelude (caller provides the prelude).
pub(super) fn emit_linear_kernel(
    name: &str,
    dtype: ScalarType,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
) -> Result<String, TensorMSLCodegenError> {
    super::gemm::emit_linear_activation_kernel(
        name,
        dtype,
        in_features,
        out_features,
        has_bias,
        None,  // no activation
        false, // no prelude (caller adds it)
    )
}

/// Emit a naive batched GEMM kernel for MatMul.
///
/// Both inputs are runtime buffers (bounded variables). Output element `[b,i,j]` is:
/// `sum_k(left[b,i,k] * right[b,k,j]) * scale` (or `right[b,j,k]` if `transpose_right`).
pub(super) fn emit_matmul_kernel(
    name: &str,
    dtype: ScalarType,
    m: usize,
    k: usize,
    n: usize,
    transpose_right: bool,
    broadcast_right: bool,
    scale: Option<f32>,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    // Accumulate dot product in f32 for f16/bf16 precision (#1352).
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let needs_cast = t != acc;
    let m_val = codegen_msl_structural::safe_msl_uint(m)?;
    let k_val = codegen_msl_structural::safe_msl_uint(k)?;
    let n_val = codegen_msl_structural::safe_msl_uint(n)?;

    let right_index = if transpose_right {
        "batch_offset_r + j * K + kk"
    } else {
        "batch_offset_r + kk * N + j"
    };

    let scale_line = match scale {
        Some(s) => format!("    sum *= {acc}({s});\n"),
        None => String::new(),
    };

    let mk = codegen_msl_structural::safe_msl_uint(m * k)?;
    let mn = codegen_msl_structural::safe_msl_uint(m * n)?;

    // When broadcast_right is true, right has no batch dim — all batches share
    // the same right matrix. Offset is always 0, preventing OOB reads.
    let batch_offset_r_line = if broadcast_right {
        "    uint batch_offset_r = 0;".to_string()
    } else {
        let right_batch_stride = if transpose_right {
            codegen_msl_structural::safe_msl_uint(n * k)?
        } else {
            codegen_msl_structural::safe_msl_uint(k * n)?
        };
        format!("    uint batch_offset_r = batch_idx * ({right_batch_stride});")
    };

    // Cast loads to accumulator type for half-precision inputs.
    let load_left = if needs_cast {
        format!("{acc}(left[batch_offset_l + i * K + kk])")
    } else {
        "left[batch_offset_l + i * K + kk]".to_string()
    };
    let load_right = if needs_cast {
        format!("{acc}(right[{right_index}])")
    } else {
        format!("right[{right_index}]")
    };
    // Cast accumulator result back to storage type on final write.
    let store_expr = if needs_cast {
        format!("{t}(sum)")
    } else {
        "sum".to_string()
    };

    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* left   [[buffer(0)]],
    device const {t}* right  [[buffer(1)]],
    device {t}* output       [[buffer(2)]],
    constant uint& total     [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total) return;
    const uint M = {m_val};
    const uint K = {k_val};
    const uint N = {n_val};

    uint batch_idx = tid / ({mn});
    uint within = tid % ({mn});
    uint i = within / N;
    uint j = within % N;

    uint batch_offset_l = batch_idx * ({mk});
{batch_offset_r_line}

    {acc} sum = 0;
    for (uint kk = 0; kk < K; kk++) {{
        sum += {load_left} * {load_right};
    }}
{scale_line}    output[tid] = {store_expr};
}}"#,
    ))
}

/// Emit MSL source for a threadgroup-parallel softmax kernel.
///
/// Three-phase algorithm (numerically stable):
/// 1. Find max along the axis (shared memory tree reduction)
/// 2. Compute `exp(x - max)` and sum (shared memory tree reduction)
/// 3. Normalize: `output[i] = exp(input[i] - max) / sum`
///
/// Each threadgroup handles one independent softmax slice (one row/outer element).
/// Uses `metal::precise::exp` for numerical stability.
///
/// Buffer layout:
/// - `buffer(0)`: input data
/// - `buffer(1)`: output data
/// - `buffer(2)`: axis_size (uint) — size of the dimension being softmax'd
/// - `buffer(3)`: outer_size (uint) — number of independent slices
pub(super) fn emit_softmax_kernel(name: &str, dtype: ScalarType) -> String {
    let t = codegen_msl::msl_type(dtype);
    // Accumulate in f32 for f16/bf16 to avoid catastrophic precision loss (#1352).
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let needs_cast = t != acc;
    let tg_sz = crate::codegen_msl_tensor::REDUCE_THREADGROUP_SIZE;

    // Load from input: cast to accumulator type if needed.
    let load = |idx: &str| -> String {
        if needs_cast {
            format!("{acc}(input[{idx}])")
        } else {
            format!("input[{idx}]")
        }
    };
    let load_i = load("base + i");

    // Cast accumulator result back to storage type for output write.
    let store = |expr: &str| -> String {
        if needs_cast {
            format!("{t}({expr})")
        } else {
            expr.to_string()
        }
    };

    format!(
        r#"[[kernel]] void {name}(
    device const {t}* input   [[buffer(0)]],
    device {t}* output        [[buffer(1)]],
    constant uint& axis_size  [[buffer(2)]],
    constant uint& outer_size [[buffer(3)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]]
) {{
    if (gid >= outer_size) return;
    threadgroup {acc} shared_max[{tg_sz}];
    threadgroup {acc} shared_sum[{tg_sz}];
    uint base = gid * axis_size;
    // Phase 1: Find max along the axis (for numerical stability)
    {acc} local_max = -HUGE_VALF;
    for (uint i = lid; i < axis_size; i += tg_sz) {{
        {acc} val = {load_i};
        if (val > local_max) local_max = val;
    }}
    shared_max[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {{
        if (lid < stride) {{
            {acc} a = shared_max[lid];
            {acc} b = shared_max[lid + stride];
            shared_max[lid] = (a > b) ? a : b;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    {acc} max_val = shared_max[0];
    // Guard: all-neg-inf lane → zero output (#1326).
    if (max_val == -HUGE_VALF) {{
        for (uint i = lid; i < axis_size; i += tg_sz) {{
            output[base + i] = {t}(0);
        }}
        return;
    }}
    // Guard: +inf in lane → uniform over +inf positions, 0 elsewhere (#1339).
    if (max_val == HUGE_VALF) {{
        threadgroup uint shared_count[{tg_sz}];
        uint local_count = 0;
        for (uint i = lid; i < axis_size; i += tg_sz) {{
            if ({load_i} == HUGE_VALF) local_count++;
        }}
        shared_count[lid] = local_count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {{
            if (lid < stride) {{
                shared_count[lid] += shared_count[lid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        {acc} prob = {acc}(1) / {acc}(shared_count[0]);
        for (uint i = lid; i < axis_size; i += tg_sz) {{
            output[base + i] = ({load_i} == HUGE_VALF) ? {store_prob} : {t}(0);
        }}
        return;
    }}
    // Phase 2: Compute sum of exp(x - max)
    {acc} local_sum = {acc}(0);
    for (uint i = lid; i < axis_size; i += tg_sz) {{
        local_sum += metal::precise::exp({load_i} - max_val);
    }}
    shared_sum[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {{
        if (lid < stride) {{
            shared_sum[lid] += shared_sum[lid + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    {acc} sum_val = shared_sum[0];
    // Phase 3: Normalize
    for (uint i = lid; i < axis_size; i += tg_sz) {{
        output[base + i] = {store_norm};
    }}
}}"#,
        name = name,
        t = t,
        acc = acc,
        tg_sz = tg_sz,
        load_i = load_i,
        store_prob = store("prob"),
        store_norm = store(&format!(
            "metal::precise::exp({load_i} - max_val) / sum_val"
        )),
    )
}

/// Emit MSL source for an embedding table lookup kernel.
///
/// `output[tid] = weight[uint(indices[tid / D]) * D + tid % D]`
/// where `D = embedding_dim`. Indices are stored as f32 (cast to uint in MSL).
///
/// Note: this kernel is used by the `execute_tensor_dispatch` f32 pipeline
/// (contract tests, verification). The `DynTensor::index_select` GPU path uses
/// a separate raw MSL kernel with native `uint*` indices to preserve precision
/// for indices > 2^24.
pub(super) fn emit_embedding_kernel(
    name: &str,
    dtype: ScalarType,
    embedding_dim: usize,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let ed = codegen_msl_structural::safe_msl_uint(embedding_dim)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* indices [[buffer(0)]],
    device const {t}* weight  [[buffer(1)]],
    device {t}* output        [[buffer(2)]],
    constant uint& total      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total) return;
    const uint EMBEDDING_DIM = {ed};
    uint seq_idx = tid / EMBEDDING_DIM;
    uint dim_idx = tid % EMBEDDING_DIM;
    uint row = uint(indices[seq_idx]);
    output[tid] = weight[row * EMBEDDING_DIM + dim_idx];
}}"#
    ))
}
