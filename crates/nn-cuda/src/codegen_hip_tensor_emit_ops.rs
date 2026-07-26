// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP C++ emission for elementwise tensor ops.
//!
//! Parallel to `nn-dsl::codegen_msl_tensor_emit_ops` — each function generates
//! a HIP `__global__` kernel string for a single elementwise operation.

use crate::codegen_hip::{hip_type, safe_hip_uint};
use crate::HipCodegenError;
use nn_dsl::ScalarType;

/// Emit HIP kernel for binary addition: `out[tid] = left[tid] + right[tid]`.
pub fn emit_binary_add_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let n = safe_hip_uint(total_elements)?;
    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ left,
    const {t}* __restrict__ right,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n}u) return;
    output[tid] = left[tid] + right[tid];
}}
"#
    ))
}

/// Emit HIP kernel for binary multiplication: `out[tid] = left[tid] * right[tid]`.
pub fn emit_binary_mul_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let n = safe_hip_uint(total_elements)?;
    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ left,
    const {t}* __restrict__ right,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n}u) return;
    output[tid] = left[tid] * right[tid];
}}
"#
    ))
}

/// Emit HIP kernel for sigmoid: `out[tid] = 1 / (1 + exp(-in[tid]))`.
pub fn emit_sigmoid_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let n = safe_hip_uint(total_elements)?;
    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n}u) return;
    float x = (float)input[tid];
    output[tid] = ({t})(1.0f / (1.0f + expf(-x)));
}}
"#
    ))
}

/// Emit HIP kernel for GELU (tanh approximation via exp).
///
/// Matches the scalar reference in `gelu.rs` — exp-based form for
/// NY verification compatibility.
pub fn emit_gelu_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let n = safe_hip_uint(total_elements)?;
    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n}u) return;
    float x = (float)input[tid];
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    float e2 = expf(2.0f * inner);
    output[tid] = ({t})(0.5f * x * (2.0f - 2.0f / (e2 + 1.0f)));
}}
"#
    ))
}

/// Emit HIP kernel for GELU (exact erf via Abramowitz & Stegun 7.1.26).
pub fn emit_gelu_erf_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let n = safe_hip_uint(total_elements)?;
    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n}u) return;
    float x = (float)input[tid];
    // erf(x / sqrt(2)) via Abramowitz & Stegun 7.1.26
    float u = x * 0.7071067811865476f;
    float ax = fabsf(u);
    float et = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = ((((1.0614054f * et + (-1.453152f)) * et + 1.4214138f) * et + (-0.28449674f)) * et + 0.2548296f) * et;
    float sign_u = (u >= 0.0f) ? 1.0f : -1.0f;
    float erf_val = sign_u * (1.0f - poly * expf(-(u * u)));
    output[tid] = ({t})(0.5f * x * (1.0f + erf_val));
}}
"#
    ))
}

/// Emit HIP kernel for ReLU: `out[tid] = max(in[tid], 0)`.
pub fn emit_relu_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let n = safe_hip_uint(total_elements)?;
    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n}u) return;
    {t} x = input[tid];
    output[tid] = (x > ({t})0) ? x : ({t})0;
}}
"#
    ))
}

/// Emit HIP kernel for tanh: `out[tid] = tanh(in[tid])`.
pub fn emit_tanh_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let n = safe_hip_uint(total_elements)?;
    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n}u) return;
    output[tid] = ({t})tanhf((float)input[tid]);
}}
"#
    ))
}

#[cfg(test)]
#[path = "codegen_hip_tensor_tests_ops.rs"]
mod tests;
