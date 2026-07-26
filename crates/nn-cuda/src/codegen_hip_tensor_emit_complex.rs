// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP C++ emission for complex tensor ops: linear, matmul, softmax, embedding.
//!
//! Parallel to `nn-dsl::codegen_msl_tensor_emit_complex` — each function
//! generates a complete HIP `__global__` kernel.

use crate::codegen_hip::{hip_accumulator_type, hip_type, safe_hip_uint, REDUCE_BLOCK_SIZE};
use crate::HipCodegenError;
use nn_dsl::ScalarType;

/// Emit HIP kernel for a naive linear (fully-connected) layer.
///
/// `out[row, col] = dot(input[row, :], weight[col, :]) + bias[col]`
/// Each thread produces one output element. Accumulation in f32 for f16/bf16.
pub fn emit_linear_kernel(
    name: &str,
    dtype: ScalarType,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let acc = hip_accumulator_type(dtype);
    let needs_cast = t != acc;
    let in_feat = safe_hip_uint(in_features)?;
    let out_feat = safe_hip_uint(out_features)?;

    let bias_param = if has_bias {
        format!("    const {t}* __restrict__ bias,\n")
    } else {
        String::new()
    };

    let bias_line = if has_bias {
        if needs_cast {
            format!("    sum += ({acc})bias[col];\n")
        } else {
            "    sum += bias[col];\n".to_string()
        }
    } else {
        String::new()
    };

    let load_input = if needs_cast {
        format!("({acc})input[row * IN_FEATURES + k]")
    } else {
        "input[row * IN_FEATURES + k]".to_string()
    };
    let load_weight = if needs_cast {
        format!("({acc})weight[col * IN_FEATURES + k]")
    } else {
        "weight[col * IN_FEATURES + k]".to_string()
    };
    let store_expr = if needs_cast {
        format!("({t})sum")
    } else {
        "sum".to_string()
    };

    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    const {t}* __restrict__ weight,
{bias_param}    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const unsigned int IN_FEATURES = {in_feat};
    const unsigned int OUT_FEATURES = {out_feat};

    unsigned int row = tid / OUT_FEATURES;
    unsigned int col = tid % OUT_FEATURES;

    {acc} sum = 0;
    for (unsigned int k = 0; k < IN_FEATURES; k++) {{
        sum += {load_input} * {load_weight};
    }}
{bias_line}    output[tid] = {store_expr};
}}"#
    ))
}

/// Emit a naive batched GEMM kernel for MatMul.
///
/// Output element `[b,i,j]` = `sum_k(left[b,i,k] * right[b,k,j]) * scale`.
/// Supports `transpose_right` and `broadcast_right`.
pub fn emit_matmul_kernel(
    name: &str,
    dtype: ScalarType,
    m: usize,
    k: usize,
    n: usize,
    transpose_right: bool,
    broadcast_right: bool,
    scale: Option<f32>,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let acc = hip_accumulator_type(dtype);
    let needs_cast = t != acc;
    let m_val = safe_hip_uint(m)?;
    let k_val = safe_hip_uint(k)?;
    let n_val = safe_hip_uint(n)?;
    let mk = safe_hip_uint(m * k)?;
    let mn = safe_hip_uint(m * n)?;

    let right_index = if transpose_right {
        "batch_offset_r + j * K + kk"
    } else {
        "batch_offset_r + kk * N + j"
    };

    let scale_line = match scale {
        Some(s) => format!("    sum *= ({acc}){s:.8}f;\n"),
        None => String::new(),
    };

    let batch_offset_r_line = if broadcast_right {
        "    unsigned int batch_offset_r = 0;".to_string()
    } else {
        let right_batch_stride = if transpose_right {
            safe_hip_uint(n * k)?
        } else {
            safe_hip_uint(k * n)?
        };
        format!("    unsigned int batch_offset_r = batch_idx * ({right_batch_stride});")
    };

    let load_left = if needs_cast {
        format!("({acc})left[batch_offset_l + i * K + kk]")
    } else {
        "left[batch_offset_l + i * K + kk]".to_string()
    };
    let load_right = if needs_cast {
        format!("({acc})right[{right_index}]")
    } else {
        format!("right[{right_index}]")
    };
    let store_expr = if needs_cast {
        format!("({t})sum")
    } else {
        "sum".to_string()
    };

    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ left,
    const {t}* __restrict__ right,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const unsigned int M = {m_val};
    const unsigned int K = {k_val};
    const unsigned int N = {n_val};

    unsigned int batch_idx = tid / ({mn});
    unsigned int within = tid % ({mn});
    unsigned int i = within / N;
    unsigned int j = within % N;

    unsigned int batch_offset_l = batch_idx * ({mk});
{batch_offset_r_line}

    {acc} sum = 0;
    for (unsigned int kk = 0; kk < K; kk++) {{
        sum += {load_left} * {load_right};
    }}
{scale_line}    output[tid] = {store_expr};
}}"#
    ))
}

/// Emit HIP kernel for threadgroup-parallel softmax.
///
/// Three-phase algorithm matching the MSL implementation:
/// 1. Find max along axis (shared memory tree reduction)
/// 2. Compute exp(x - max) and sum
/// 3. Normalize: output[i] = exp(input[i] - max) / sum
pub fn emit_softmax_kernel(name: &str, dtype: ScalarType) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let acc = hip_accumulator_type(dtype);
    let needs_cast = t != acc;
    let tg_sz = REDUCE_BLOCK_SIZE;

    let load = |idx: &str| -> String {
        if needs_cast {
            format!("({acc})input[{idx}]")
        } else {
            format!("input[{idx}]")
        }
    };
    let load_i = load("base + i");

    let store = |expr: &str| -> String {
        if needs_cast {
            format!("({t})({expr})")
        } else {
            expr.to_string()
        }
    };

    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    {t}* __restrict__ output,
    const unsigned int axis_size,
    const unsigned int outer_size
) {{
    unsigned int gid = blockIdx.x;
    unsigned int lid = threadIdx.x;
    unsigned int tg_sz = blockDim.x;
    if (gid >= outer_size) return;

    __shared__ {acc} shared_max[{tg_sz}];
    __shared__ {acc} shared_sum[{tg_sz}];
    unsigned int base = gid * axis_size;

    // Phase 1: Find max along the axis
    {acc} local_max = -HUGE_VALF;
    for (unsigned int i = lid; i < axis_size; i += tg_sz) {{
        {acc} val = {load_i};
        if (val > local_max) local_max = val;
    }}
    shared_max[lid] = local_max;
    __syncthreads();
    for (unsigned int stride = tg_sz / 2; stride > 0; stride >>= 1) {{
        if (lid < stride) {{
            {acc} a = shared_max[lid];
            {acc} b = shared_max[lid + stride];
            shared_max[lid] = (a > b) ? a : b;
        }}
        __syncthreads();
    }}
    {acc} max_val = shared_max[0];

    // Guard: all-neg-inf → zero output (#1326)
    if (max_val == -HUGE_VALF) {{
        for (unsigned int i = lid; i < axis_size; i += tg_sz) {{
            output[base + i] = ({t})0;
        }}
        return;
    }}

    // Phase 2: Compute sum of exp(x - max)
    {acc} local_sum = ({acc})0;
    for (unsigned int i = lid; i < axis_size; i += tg_sz) {{
        local_sum += expf({load_i} - max_val);
    }}
    shared_sum[lid] = local_sum;
    __syncthreads();
    for (unsigned int stride = tg_sz / 2; stride > 0; stride >>= 1) {{
        if (lid < stride) {{
            shared_sum[lid] += shared_sum[lid + stride];
        }}
        __syncthreads();
    }}
    {acc} sum_val = shared_sum[0];

    // Phase 3: Normalize
    for (unsigned int i = lid; i < axis_size; i += tg_sz) {{
        output[base + i] = {store_norm};
    }}
}}"#,
        name = name,
        t = t,
        acc = acc,
        tg_sz = tg_sz,
        load_i = load_i,
        store_norm = store(&format!("expf({load_i} - max_val) / sum_val")),
    ))
}

/// Emit HIP kernel for embedding table lookup.
///
/// `output[tid] = weight[uint(indices[tid / D]) * D + tid % D]`
pub fn emit_embedding_kernel(
    name: &str,
    dtype: ScalarType,
    embedding_dim: usize,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let ed = safe_hip_uint(embedding_dim)?;
    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ indices,
    const {t}* __restrict__ weight,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const unsigned int EMBEDDING_DIM = {ed};
    unsigned int seq_idx = tid / EMBEDDING_DIM;
    unsigned int dim_idx = tid % EMBEDDING_DIM;
    unsigned int row = (unsigned int)indices[seq_idx];
    output[tid] = weight[row * EMBEDDING_DIM + dim_idx];
}}"#
    ))
}

#[cfg(test)]
#[path = "codegen_hip_tensor_tests_complex.rs"]
mod tests;
