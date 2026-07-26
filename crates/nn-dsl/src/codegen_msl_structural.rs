// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL code generation for structural tensor operations.
//!
//! AxisSelect, Stack, Broadcast, Narrow, Transpose, Concat.
//! Reshape is a zero-copy buffer alias and needs no MSL kernel.
//!
//! Part of #19 (K2-K8 kernel ports): enables K6 RoPE MSL codegen.

use crate::codegen_msl;
use crate::codegen_msl_tensor::TensorMSLCodegenError;
use crate::ir::ScalarType;

#[path = "codegen_msl_structural_concat.rs"]
mod concat;
pub(crate) use concat::emit_concat_kernel;

#[path = "codegen_msl_structural_spatial.rs"]
mod spatial;
pub(crate) use spatial::{
    build_broadcast_index_body, emit_broadcast_kernel, emit_tiled_transpose_2d_kernel,
    emit_transpose_kernel,
};

/// Format a `usize` value as an MSL `uint` literal, rejecting values > `u32::MAX`.
///
/// MSL `uint` is 32-bit unsigned. On 64-bit hosts, `usize` is 64-bit, so stride
/// values computed from large tensor shapes could exceed `u32::MAX` without this guard.
pub(crate) fn safe_msl_uint(val: usize) -> Result<String, TensorMSLCodegenError> {
    if val > u32::MAX as usize {
        return Err(TensorMSLCodegenError::StrideExceedsU32 {
            value: val,
            max: u32::MAX,
        });
    }
    Ok(val.to_string())
}

/// Emit MSL for an AxisSelect kernel.
///
/// Selects index `select_index` along `axis` from input shape `input_shape`.
/// Output shape is `input_shape` with the `axis` dimension removed.
///
/// Each output thread computes its multi-dimensional coordinate in the output
/// shape, inserts `select_index` at the `axis` position to form the input
/// coordinate, and copies one element.
///
/// Buffer layout:
/// - `buffer(0)`: input data
/// - `buffer(1)`: output data
/// - `buffer(2)`: total output elements (uint)
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn emit_axis_select_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    axis: usize,
    select_index: usize,
) -> Result<String, TensorMSLCodegenError> {
    if axis >= input_shape.len() {
        return Err(TensorMSLCodegenError::AxisOutOfBounds {
            axis,
            rank: input_shape.len(),
        });
    }
    let t = codegen_msl::msl_type(dtype);

    // Output shape: input_shape with axis dimension removed.
    let out_shape: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &d)| d)
        .collect();
    let out_rank = out_shape.len();

    // Output strides (row-major).
    let out_strides = row_major_strides(&out_shape)?;

    // Input strides (row-major).
    let in_strides = row_major_strides(input_shape)?;

    // Build the index mapping: decompose tid into output coords,
    // then compute input linear index by inserting select_index at axis.
    let mut body = String::new();
    body.push_str("    uint in_idx = 0;\n");
    body.push_str("    uint remainder = tid;\n");

    let mut out_dim = 0;
    for (in_dim, &in_stride) in in_strides.iter().enumerate() {
        let in_s = safe_msl_uint(in_stride)?;
        if in_dim == axis {
            // Fixed dimension: always use select_index.
            let sel = safe_msl_uint(select_index)?;
            body.push_str(&format!("    in_idx += {sel} * {in_s};\n"));
        } else {
            // Variable dimension: extract coordinate from output tid.
            let s = safe_msl_uint(out_strides[out_dim])?;
            body.push_str(&format!("    uint c{out_dim} = remainder / {s};\n"));
            body.push_str(&format!("    remainder = remainder % {s};\n"));
            body.push_str(&format!("    in_idx += c{out_dim} * {in_s};\n"));
            out_dim += 1;
        }
    }
    // Validate index decomposition completeness: out_dim increments once per
    // non-axis input dimension, so must equal out_rank. Failure here indicates
    // a logic bug in the loop above. Upgraded from debug_assert_eq per #892.
    if out_dim != out_rank {
        return Err(TensorMSLCodegenError::InvalidParameter(format!(
            "emit_axis_select_kernel: index decomposition produced {out_dim} output dims, expected {out_rank}"
        )));
    }

    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    constant uint& total    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total) return;
{body}    output[tid] = input[in_idx];
}}"#,
    ))
}

#[path = "codegen_msl_structural_stack.rs"]
mod stack;
pub(crate) use stack::emit_stack_kernel;

/// Emit MSL for a Narrow/Slice kernel.
///
/// Extracts `[start, start+length)` along `axis` from `input_shape`,
/// preserving the axis dimension in the output. Each output thread
/// decomposes its tid into output coordinates, adds `start` to the
/// axis coordinate, and copies one element.
///
/// Buffer layout:
/// - `buffer(0)`: input data
/// - `buffer(1)`: output data
/// - `buffer(2)`: total output elements (uint)
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn emit_narrow_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    axis: usize,
    start: usize,
    length: usize,
) -> Result<String, TensorMSLCodegenError> {
    if axis >= input_shape.len() {
        return Err(TensorMSLCodegenError::AxisOutOfBounds {
            axis,
            rank: input_shape.len(),
        });
    }
    let t = codegen_msl::msl_type(dtype);

    // Output shape: input_shape with dim[axis] replaced by length.
    let mut out_shape = input_shape.to_vec();
    out_shape[axis] = length;

    let out_strides = row_major_strides(&out_shape)?;
    let in_strides = row_major_strides(input_shape)?;

    let start_str = safe_msl_uint(start)?;
    let mut body = String::new();
    body.push_str("    uint in_idx = 0;\n");
    body.push_str("    uint remainder = tid;\n");

    for (dim, &out_stride) in out_strides.iter().enumerate() {
        let os = safe_msl_uint(out_stride)?;
        let is = safe_msl_uint(in_strides[dim])?;
        body.push_str(&format!("    uint c{dim} = remainder / {os};\n"));
        body.push_str(&format!("    remainder = remainder % {os};\n"));
        if dim == axis {
            body.push_str(&format!("    in_idx += (c{dim} + {start_str}) * {is};\n"));
        } else {
            body.push_str(&format!("    in_idx += c{dim} * {is};\n"));
        }
    }

    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    constant uint& total    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total) return;
{body}    output[tid] = input[in_idx];
}}"#,
    ))
}

/// Compute row-major strides for a shape.
///
/// Returns `Err` if the stride product overflows `usize`.
pub(super) fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>, TensorMSLCodegenError> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].checked_mul(shape[i + 1]).ok_or_else(|| {
            TensorMSLCodegenError::ShapeProductOverflow {
                shape: shape.to_vec(),
            }
        })?;
    }
    Ok(strides)
}

#[cfg(test)]
#[path = "codegen_msl_structural_tests.rs"]
mod tests;
