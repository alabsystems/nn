// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL emission for index-based tensor ops: IndexSelect, Gather, f32→u32 conversion.
//!
//! Extracted from `codegen_msl_tensor_emit_complex.rs` to keep files under
//! the 500-line limit. All kernel-emit functions are `pub(super)` — called
//! from `codegen_msl_tensor_emit_step.rs`.
//!
//! Part of #2278.

use crate::codegen_msl;
use crate::codegen_msl_structural;
use crate::codegen_msl_tensor::TensorMSLCodegenError;
use crate::ir::ScalarType;

/// Emit MSL source for an index-select kernel.
///
/// Generalizes embedding to arbitrary `dim`. Output shape is the input shape
/// with dimension `dim` replaced by `num_indices` (length of 1-D index tensor).
///
/// 3-way decomposition of flat output `tid`:
///   `tid = outer_idx * NUM_INDICES * INNER + idx_pos * INNER + inner_idx`
///
/// Buffer layout: buffer(0) = input, buffer(1) = indices (native uint),
/// buffer(2) = output, buffer(3) = total element count.
///
/// Indices are native `uint*` to preserve precision for indices > 2^24
/// (f32 has only 24-bit mantissa). The dispatch encoder converts f32
/// indices to u32 before binding. OOB indices are clamped to the last
/// valid row (defense-in-depth, matching runtime DynTensor behavior).
/// Part of #2278.
pub(super) fn emit_index_select_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    dim: usize,
) -> Result<String, TensorMSLCodegenError> {
    if dim >= input_shape.len() {
        return Err(TensorMSLCodegenError::InvalidDim {
            dim,
            rank: input_shape.len(),
        });
    }
    let dim_size = input_shape[dim];
    if dim_size == 0 {
        return Err(TensorMSLCodegenError::EmptyDim { dim });
    }
    let t = codegen_msl::msl_type(dtype);
    let inner: usize = input_shape[dim + 1..].iter().product::<usize>().max(1);
    let inner_s = codegen_msl_structural::safe_msl_uint(inner)?;
    let dim_size_s = codegen_msl_structural::safe_msl_uint(dim_size)?;
    // NUM_INDICES is derived at runtime: total / (outer * INNER).
    // Embed outer as a constant so the kernel can compute K.
    let outer: usize = input_shape[..dim].iter().product::<usize>().max(1);
    let outer_s = codegen_msl_structural::safe_msl_uint(outer)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input    [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device {t}* output         [[buffer(2)]],
    constant uint& total       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total) return;
    const uint INNER = {inner_s};
    const uint DIM_SIZE = {dim_size_s};
    const uint OUTER = {outer_s};
    uint num_indices = total / (OUTER * INNER);
    uint inner_idx = tid % INNER;
    uint remaining = tid / INNER;
    uint idx_pos = remaining % num_indices;
    uint outer_idx = remaining / num_indices;
    uint src_row = indices[idx_pos];
    if (src_row >= DIM_SIZE) src_row = DIM_SIZE - 1;
    uint src_idx = outer_idx * (DIM_SIZE * INNER) + src_row * INNER + inner_idx;
    output[tid] = input[src_idx];
}}"#
    ))
}

/// Emit MSL source for a gather kernel.
///
/// `output[tid]` reads from `input` with dimension `dim` coordinate replaced
/// by `indices[tid]`. Index tensor has the same rank as input.
///
/// For dim `d`: input_shape[i] == output_shape[i] for all i != d.
/// Uses the same 3-way decomposition as index_select, but the lookup index
/// is per-element (`indices[tid]`) instead of per-row (`indices[idx_pos]`).
///
/// Indices are native `uint*` with OOB clamping. Part of #2278.
pub(super) fn emit_gather_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    dim: usize,
) -> Result<String, TensorMSLCodegenError> {
    if dim >= input_shape.len() {
        return Err(TensorMSLCodegenError::InvalidDim {
            dim,
            rank: input_shape.len(),
        });
    }
    let dim_size = input_shape[dim];
    if dim_size == 0 {
        return Err(TensorMSLCodegenError::EmptyDim { dim });
    }
    let t = codegen_msl::msl_type(dtype);
    let inner: usize = input_shape[dim + 1..].iter().product::<usize>().max(1);
    let inner_s = codegen_msl_structural::safe_msl_uint(inner)?;
    let dim_size_s = codegen_msl_structural::safe_msl_uint(dim_size)?;
    let outer: usize = input_shape[..dim].iter().product::<usize>().max(1);
    let outer_s = codegen_msl_structural::safe_msl_uint(outer)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input    [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device {t}* output         [[buffer(2)]],
    constant uint& total       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total) return;
    const uint INNER = {inner_s};
    const uint DIM_SIZE = {dim_size_s};
    const uint OUTER = {outer_s};
    uint out_dim_size = total / (OUTER * INNER);
    uint inner_idx = tid % INNER;
    uint remaining = tid / INNER;
    uint outer_idx = remaining / out_dim_size;
    uint src_dim = indices[tid];
    if (src_dim >= DIM_SIZE) src_dim = DIM_SIZE - 1;
    uint src_idx = outer_idx * (DIM_SIZE * INNER) + src_dim * INNER + inner_idx;
    output[tid] = input[src_idx];
}}"#
    ))
}

/// Emit a small MSL kernel that converts f32 indices to u32.
///
/// The compiled pipeline stores all tensors as f32. IndexSelect/Gather
/// kernels read `uint*` indices for precision. This conversion kernel
/// bridges the two: `output[tid] = uint(input[tid])`.
///
/// Returns the kernel name and MSL source.
/// Part of #2278.
pub(super) fn emit_f32_to_u32_kernel(name: &str) -> (String, String) {
    let conv_name = format!("{name}_f32_to_u32");
    let msl = format!(
        r#"[[kernel]] void {conv_name}(
    device const float* input [[buffer(0)]],
    device uint* output       [[buffer(1)]],
    constant uint& total      [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total) return;
    float v = input[tid];
    output[tid] = (v < 0.0f) ? 0u : uint(v);
}}"#
    );
    (conv_name, msl)
}
