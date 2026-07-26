// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP C++ emission for index-based and padding ops: ZeroPad1d, IndexSelect,
//! Gather, and f32→u32 conversion.
//!
//! Extracted from `codegen_hip_tensor_emit_elementwise.rs` to keep files under
//! the 500-line limit. Mirrors `nn-dsl::codegen_msl_tensor_emit_index`.
//!
//! Part of #2241 (HIP codegen Phase 4).

use crate::codegen_hip::{hip_type, safe_hip_uint};
use crate::HipCodegenError;
use nn_dsl::ScalarType;

/// Emit HIP kernel for ZeroPad1d.
///
/// Pads a `[channels, length]` tensor with zeros on the left/right sides.
/// Output element `[c, t]` reads `input[c * in_length + (t - pad_left)]`
/// when `t` falls in the input range, otherwise writes 0.
pub fn emit_zero_pad_1d_hip(
    name: &str,
    dtype: ScalarType,
    channels: usize,
    in_length: usize,
    pad_left: usize,
    out_length: usize,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let total =
        channels
            .checked_mul(out_length)
            .ok_or_else(|| HipCodegenError::ShapeProductOverflow {
                shape: vec![channels, out_length],
            })?;
    let n = safe_hip_uint(total)?;
    let ol = safe_hip_uint(out_length)?;
    let il = safe_hip_uint(in_length)?;
    let pl = safe_hip_uint(pad_left)?;
    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n}u) return;
    unsigned int c = tid / {ol}u;
    unsigned int ot = tid % {ol}u;
    if (ot >= {pl}u && ot < {pl}u + {il}u) {{
        output[tid] = input[c * {il}u + (ot - {pl}u)];
    }} else {{
        output[tid] = ({t})0;
    }}
}}
"#
    ))
}

/// Emit HIP kernel for IndexSelect.
///
/// Generalizes embedding to arbitrary `dim`. Output shape is the input shape
/// with dimension `dim` replaced by `num_indices`. Uses the same 3-way
/// decomposition as MSL: `tid = outer * NUM_INDICES * INNER + idx_pos * INNER + inner_idx`.
///
/// Indices are native `unsigned int*` with OOB clamping.
pub fn emit_index_select_hip(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    dim: usize,
) -> Result<String, HipCodegenError> {
    if dim >= input_shape.len() {
        return Err(HipCodegenError::AxisOutOfBounds {
            axis: dim,
            rank: input_shape.len(),
        });
    }
    let dim_size = input_shape[dim];
    if dim_size == 0 {
        return Err(HipCodegenError::InvalidParameter(format!(
            "dimension {dim} has size 0"
        )));
    }
    let t = hip_type(dtype)?;
    let inner: usize = input_shape[dim + 1..].iter().product::<usize>().max(1);
    let inner_s = safe_hip_uint(inner)?;
    let dim_size_s = safe_hip_uint(dim_size)?;
    let outer: usize = input_shape[..dim].iter().product::<usize>().max(1);
    let outer_s = safe_hip_uint(outer)?;
    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    const unsigned int* __restrict__ indices,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const unsigned int INNER = {inner_s};
    const unsigned int DIM_SIZE = {dim_size_s};
    const unsigned int OUTER = {outer_s};
    unsigned int num_indices = total / (OUTER * INNER);
    unsigned int inner_idx = tid % INNER;
    unsigned int remaining = tid / INNER;
    unsigned int idx_pos = remaining % num_indices;
    unsigned int outer_idx = remaining / num_indices;
    unsigned int src_row = indices[idx_pos];
    if (src_row >= DIM_SIZE) src_row = DIM_SIZE - 1;
    unsigned int src_idx = outer_idx * (DIM_SIZE * INNER) + src_row * INNER + inner_idx;
    output[tid] = input[src_idx];
}}"#
    ))
}

/// Emit HIP kernel for Gather.
///
/// `output[tid]` reads from `input` with dimension `dim` coordinate replaced
/// by `indices[tid]`. Index tensor has the same rank as input. Uses per-element
/// index lookup (unlike IndexSelect which uses per-row).
///
/// Indices are native `unsigned int*` with OOB clamping.
pub fn emit_gather_hip(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    dim: usize,
) -> Result<String, HipCodegenError> {
    if dim >= input_shape.len() {
        return Err(HipCodegenError::AxisOutOfBounds {
            axis: dim,
            rank: input_shape.len(),
        });
    }
    let dim_size = input_shape[dim];
    if dim_size == 0 {
        return Err(HipCodegenError::InvalidParameter(format!(
            "dimension {dim} has size 0"
        )));
    }
    let t = hip_type(dtype)?;
    let inner: usize = input_shape[dim + 1..].iter().product::<usize>().max(1);
    let inner_s = safe_hip_uint(inner)?;
    let dim_size_s = safe_hip_uint(dim_size)?;
    let outer: usize = input_shape[..dim].iter().product::<usize>().max(1);
    let outer_s = safe_hip_uint(outer)?;
    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    const unsigned int* __restrict__ indices,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const unsigned int INNER = {inner_s};
    const unsigned int DIM_SIZE = {dim_size_s};
    const unsigned int OUTER = {outer_s};
    unsigned int out_dim_size = total / (OUTER * INNER);
    unsigned int inner_idx = tid % INNER;
    unsigned int remaining = tid / INNER;
    unsigned int outer_idx = remaining / out_dim_size;
    unsigned int src_dim = indices[tid];
    if (src_dim >= DIM_SIZE) src_dim = DIM_SIZE - 1;
    unsigned int src_idx = outer_idx * (DIM_SIZE * INNER) + src_dim * INNER + inner_idx;
    output[tid] = input[src_idx];
}}"#
    ))
}

/// Emit a small HIP kernel that converts f32 indices to u32.
///
/// IndexSelect/Gather kernels read `unsigned int*` indices for precision.
/// This conversion kernel bridges f32 storage to u32.
pub fn emit_f32_to_u32_hip(name: &str) -> (String, String) {
    let conv_name = format!("{name}_f32_to_u32");
    let hip = format!(
        r#"extern "C" __global__ void {conv_name}(
    const float* __restrict__ input,
    unsigned int* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    float v = input[tid];
    output[tid] = (v < 0.0f) ? 0u : (unsigned int)(v);
}}"#
    );
    (conv_name, hip)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ZeroPad1d tests ---

    #[test]
    fn test_zero_pad_1d_basic() {
        let src = emit_zero_pad_1d_hip("zp_test", ScalarType::F32, 4, 10, 3, 16).unwrap();
        assert!(src.contains("extern \"C\" __global__ void zp_test"));
        assert!(src.contains("16u"));
        assert!(src.contains("10u"));
        assert!(src.contains("3u"));
    }

    #[test]
    fn test_zero_pad_1d_f16() {
        let src = emit_zero_pad_1d_hip("zp_f16", ScalarType::F16, 2, 8, 1, 10).unwrap();
        assert!(src.contains("half"));
        assert!(src.contains("(half)0"));
    }

    // --- IndexSelect tests ---

    #[test]
    fn test_index_select_2d_dim0() {
        let src = emit_index_select_hip("isel", ScalarType::F32, &[10, 8], 0).unwrap();
        assert!(src.contains("extern \"C\" __global__ void isel"));
        assert!(src.contains("DIM_SIZE = 10"));
        assert!(src.contains("INNER = 8"));
        assert!(src.contains("src_row = indices[idx_pos]"));
    }

    #[test]
    fn test_index_select_3d_dim1() {
        let src = emit_index_select_hip("isel3d", ScalarType::F32, &[2, 5, 3], 1).unwrap();
        assert!(src.contains("DIM_SIZE = 5"));
        assert!(src.contains("INNER = 3"));
        assert!(src.contains("OUTER = 2"));
    }

    #[test]
    fn test_index_select_dim_oob() {
        let result = emit_index_select_hip("bad", ScalarType::F32, &[4, 8], 2);
        assert!(result.is_err());
    }

    // --- Gather tests ---

    #[test]
    fn test_gather_2d_dim0() {
        let src = emit_gather_hip("gath", ScalarType::F32, &[10, 8], 0).unwrap();
        assert!(src.contains("extern \"C\" __global__ void gath"));
        assert!(src.contains("DIM_SIZE = 10"));
        assert!(src.contains("src_dim = indices[tid]"));
    }

    #[test]
    fn test_gather_dim_oob() {
        let result = emit_gather_hip("bad", ScalarType::F32, &[4], 1);
        assert!(result.is_err());
    }

    // --- f32_to_u32 conversion ---

    #[test]
    fn test_f32_to_u32_kernel() {
        let (conv_name, src) = emit_f32_to_u32_hip("lookup");
        assert_eq!(conv_name, "lookup_f32_to_u32");
        assert!(src.contains("extern \"C\" __global__ void lookup_f32_to_u32"));
        assert!(src.contains("(unsigned int)(v)"));
    }
}
