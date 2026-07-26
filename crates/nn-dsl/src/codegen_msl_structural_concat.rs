// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL code generation for the Concat kernel.
//!
//! Extracted from `codegen_msl_structural.rs` to keep that file under
//! the 500-line limit. Part of #810.

use crate::codegen_msl;
use crate::codegen_msl_tensor::TensorMSLCodegenError;
use crate::ir::ScalarType;

use super::safe_msl_uint;

/// Emit MSL for a Concat kernel.
///
/// Concatenates `n_inputs` tensors along `axis`, where each input may have a
/// different size along `axis` but must match on all other dimensions. The
/// output axis size is the sum of all input axis sizes.
///
/// Strategy: decompose each output thread's tid into `(outer, axis_coord, inner)`:
/// - `inner` = product of dims after axis (identical for all inputs)
/// - `axis_coord` = position along the concatenated axis
/// - `outer` = product of dims before axis (identical for all inputs)
///
/// The `axis_coord` determines which input to read from. Per-input index is:
/// `outer * (input_i_axis_size * inner_stride) + local_axis * inner_stride + inner`
///
/// Buffer layout:
/// - `buffer(0)` .. `buffer(n_inputs - 1)`: input buffers
/// - `buffer(n_inputs)`: output buffer
/// - `buffer(n_inputs + 1)`: total output elements (uint)
///
/// Part of #810.
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn emit_concat_kernel(
    name: &str,
    dtype: ScalarType,
    first_input_shape: &[usize],
    input_axis_sizes: &[usize],
    axis: usize,
) -> Result<String, TensorMSLCodegenError> {
    let n_inputs = input_axis_sizes.len();
    if n_inputs == 0 {
        return Err(TensorMSLCodegenError::EmptyStack);
    }
    if axis >= first_input_shape.len() {
        return Err(TensorMSLCodegenError::AxisOutOfBounds {
            axis,
            rank: first_input_shape.len(),
        });
    }
    // When n_inputs exceeds direct binding capacity, switch to packed kernel.
    if n_inputs > codegen_msl::MAX_DIRECT_BINDING_INPUTS {
        return emit_concat_kernel_packed(name, dtype, first_input_shape, input_axis_sizes, axis);
    }
    let highest_index = n_inputs + 1;
    if highest_index > codegen_msl::MAX_METAL_BUFFER_INDEX {
        return Err(TensorMSLCodegenError::BufferLimitExceeded {
            required: n_inputs + 2,
            max: codegen_msl::MAX_METAL_BUFFER_INDEX + 1,
            max_index: codegen_msl::MAX_METAL_BUFFER_INDEX,
        });
    }
    let t = codegen_msl::msl_type(dtype);

    // inner_stride = product of all dims after axis (same for all inputs).
    let inner_stride: usize = first_input_shape[axis + 1..]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| TensorMSLCodegenError::ShapeProductOverflow {
            shape: first_input_shape.to_vec(),
        })?;
    let total_axis: usize = input_axis_sizes.iter().sum();
    // axis_inner_stride = total_axis * inner_stride (for output decomposition).
    let axis_inner = total_axis.checked_mul(inner_stride).ok_or_else(|| {
        TensorMSLCodegenError::ShapeProductOverflow {
            shape: first_input_shape.to_vec(),
        }
    })?;

    // Build buffer parameter declarations.
    let mut params = String::new();
    for i in 0..n_inputs {
        params.push_str(&format!(
            "    device const {t}* input{i} [[buffer({i})]],\n"
        ));
    }
    let total_buf = n_inputs + 1;
    params.push_str(&format!(
        "    device {t}* output      [[buffer({n_inputs})]],\n"
    ));
    params.push_str(&format!(
        "    constant uint& total    [[buffer({total_buf})]],\n"
    ));
    params.push_str("    uint tid [[thread_position_in_grid]]");

    let inner_s = safe_msl_uint(inner_stride)?;
    let axis_inner_s = safe_msl_uint(axis_inner)?;
    let total_axis_s = safe_msl_uint(total_axis)?;

    let mut body = String::new();
    // Decompose tid into (outer, axis_coord, inner).
    body.push_str(&format!("    uint outer = tid / {axis_inner_s};\n"));
    body.push_str(&format!(
        "    uint axis_coord = (tid / {inner_s}) % {total_axis_s};\n"
    ));
    body.push_str(&format!("    uint inner = tid % {inner_s};\n"));

    // Determine which input buffer and local axis offset.
    body.push_str("    uint which_input = 0;\n");
    body.push_str("    uint local_axis = axis_coord;\n");
    let mut cumsum = 0usize;
    for (i, &sz) in input_axis_sizes.iter().enumerate() {
        cumsum += sz;
        if i < n_inputs - 1 {
            let cs = safe_msl_uint(cumsum)?;
            body.push_str(&format!(
                "    if (axis_coord >= {cs}) {{ which_input = {}; local_axis = axis_coord - {cs}; }}\n",
                i + 1
            ));
        }
    }

    // Compute per-input flat index:
    //   in_idx = outer * (input_i_axis_size * inner_stride) + local_axis * inner_stride + inner
    // The per-input axis stride (input_i_axis_size * inner_stride) varies by input.
    let mut idx_select = String::new();
    #[allow(clippy::needless_range_loop)] // Index used for both conditional logic and array access
    for i in 0..n_inputs {
        let per_input_stride = safe_msl_uint(
            input_axis_sizes[i]
                .checked_mul(inner_stride)
                .ok_or_else(|| TensorMSLCodegenError::ShapeProductOverflow {
                    shape: first_input_shape.to_vec(),
                })?,
        )?;
        if n_inputs == 1 {
            idx_select = format!(
                "    uint in_idx = outer * {per_input_stride} + local_axis * {inner_s} + inner;\n"
            );
            break;
        }
        if i == 0 {
            idx_select.push_str(&format!(
                "    uint in_idx = (which_input == 0) ? (outer * {per_input_stride} + local_axis * {inner_s} + inner)"
            ));
        } else if i == n_inputs - 1 {
            idx_select.push_str(&format!(
                " : (outer * {per_input_stride} + local_axis * {inner_s} + inner);\n"
            ));
        } else {
            idx_select.push_str(&format!(
                " : (which_input == {i}) ? (outer * {per_input_stride} + local_axis * {inner_s} + inner)"
            ));
        }
    }

    // Select value from the correct input buffer.
    let mut val_select = String::new();
    if n_inputs == 1 {
        val_select = format!("    {t} val = input0[in_idx];\n");
    } else {
        for i in 0..n_inputs {
            if i == 0 {
                val_select.push_str(&format!(
                    "    {t} val = (which_input == 0) ? input0[in_idx]"
                ));
            } else if i == n_inputs - 1 {
                val_select.push_str(&format!(" : input{i}[in_idx];\n"));
            } else {
                val_select.push_str(&format!(" : (which_input == {i}) ? input{i}[in_idx]"));
            }
        }
    }

    Ok(format!(
        r#"[[kernel]] void {name}(
{params}
) {{
    if (tid >= total) return;
{body}{idx_select}{val_select}    output[tid] = val;
}}"#,
    ))
}

/// Emit MSL for a packed Concat kernel.
///
/// Used when `n_inputs > MAX_DIRECT_BINDING_INPUTS`. All input buffers are
/// packed into one contiguous buffer with element offsets. An additional
/// `input_strides` buffer carries per-input `axis_size * inner_stride`
/// values needed for index computation.
///
/// Buffer layout:
/// - `buffer(0)`: packed input buffer (all inputs concatenated element-wise)
/// - `buffer(1)`: offsets array (`constant uint*`, element offset per input)
/// - `buffer(2)`: input_strides array (`constant uint*`, `axis_size * inner_stride` per input)
/// - `buffer(3)`: output buffer
/// - `buffer(4)`: total output elements (uint)
fn emit_concat_kernel_packed(
    name: &str,
    dtype: ScalarType,
    first_input_shape: &[usize],
    input_axis_sizes: &[usize],
    axis: usize,
) -> Result<String, TensorMSLCodegenError> {
    let n_inputs = input_axis_sizes.len();
    let t = codegen_msl::msl_type(dtype);

    // inner_stride = product of all dims after axis (same for all inputs).
    let inner_stride: usize = first_input_shape[axis + 1..]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| TensorMSLCodegenError::ShapeProductOverflow {
            shape: first_input_shape.to_vec(),
        })?;
    let total_axis: usize = input_axis_sizes.iter().sum();
    let axis_inner = total_axis.checked_mul(inner_stride).ok_or_else(|| {
        TensorMSLCodegenError::ShapeProductOverflow {
            shape: first_input_shape.to_vec(),
        }
    })?;

    let inner_s = safe_msl_uint(inner_stride)?;
    let axis_inner_s = safe_msl_uint(axis_inner)?;
    let total_axis_s = safe_msl_uint(total_axis)?;

    let mut body = String::new();
    // Decompose tid into (outer, axis_coord, inner).
    body.push_str(&format!("    uint outer = tid / {axis_inner_s};\n"));
    body.push_str(&format!(
        "    uint axis_coord = (tid / {inner_s}) % {total_axis_s};\n"
    ));
    body.push_str(&format!("    uint inner = tid % {inner_s};\n"));

    // Determine which input and local axis offset via cumulative axis sizes.
    // Emit a cascading if-chain (same logic as direct kernel).
    body.push_str("    uint which_input = 0;\n");
    body.push_str("    uint local_axis = axis_coord;\n");
    let mut cumsum = 0usize;
    for (i, &sz) in input_axis_sizes.iter().enumerate() {
        cumsum += sz;
        if i < n_inputs - 1 {
            let cs = safe_msl_uint(cumsum)?;
            body.push_str(&format!(
                "    if (axis_coord >= {cs}) {{ which_input = {}; local_axis = axis_coord - {cs}; }}\n",
                i + 1
            ));
        }
    }

    // Compute index using per-input stride from the input_strides buffer,
    // then read from packed buffer at the computed offset.
    body.push_str(&format!(
        "    uint in_idx = outer * input_strides[which_input] + local_axis * {inner_s} + inner;\n"
    ));
    body.push_str("    uint base = offsets[which_input];\n");
    body.push_str(&format!("    {t} val = packed_inputs[base + in_idx];\n"));

    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* packed_inputs  [[buffer(0)]],
    constant uint* offsets           [[buffer(1)]],
    constant uint* input_strides     [[buffer(2)]],
    device {t}* output               [[buffer(3)]],
    constant uint& total             [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total) return;
{body}    output[tid] = val;
}}"#,
    ))
}
