// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL code generation for Stack tensor operations (direct + packed).
//!
//! Extracted from `codegen_msl_structural.rs` (Part of #1970 D6).

use crate::codegen_msl;
use crate::codegen_msl_tensor::TensorMSLCodegenError;
use crate::ir::ScalarType;

use super::{row_major_strides, safe_msl_uint};

/// Emit MSL for a Stack kernel.
///
/// Stacks `n_inputs` tensors of shape `input_shape` along `axis`, producing
/// an output with a new dimension of size `n_inputs` inserted at `axis`.
///
/// Each output thread computes its multi-dimensional coordinate, extracts the
/// `axis` coordinate to determine which input buffer to read from, and
/// computes the source index within that input.
///
/// Buffer layout:
/// - `buffer(0)` .. `buffer(n_inputs - 1)`: input buffers
/// - `buffer(n_inputs)`: output buffer
/// - `buffer(n_inputs + 1)`: total output elements (uint)
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn emit_stack_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    n_inputs: usize,
    axis: usize,
) -> Result<String, TensorMSLCodegenError> {
    if n_inputs == 0 {
        return Err(TensorMSLCodegenError::EmptyStack);
    }
    // axis can be 0..=input_shape.len() (insert before any dim, or after last).
    if axis > input_shape.len() {
        return Err(TensorMSLCodegenError::AxisOutOfBounds {
            axis,
            rank: input_shape.len(),
        });
    }
    // When n_inputs exceeds direct binding capacity, switch to packed kernel.
    if n_inputs > codegen_msl::MAX_DIRECT_BINDING_INPUTS {
        return emit_stack_kernel_packed(name, dtype, input_shape, n_inputs, axis);
    }
    // Stack kernel uses n_inputs + 2 buffers: inputs + output + total.
    let highest_index = n_inputs + 1;
    if highest_index > codegen_msl::MAX_METAL_BUFFER_INDEX {
        return Err(TensorMSLCodegenError::BufferLimitExceeded {
            required: n_inputs + 2,
            max: codegen_msl::MAX_METAL_BUFFER_INDEX + 1,
            max_index: codegen_msl::MAX_METAL_BUFFER_INDEX,
        });
    }
    let t = codegen_msl::msl_type(dtype);

    // Output shape: input_shape with n_inputs inserted at axis.
    let mut out_shape = Vec::with_capacity(input_shape.len() + 1);
    out_shape.extend_from_slice(&input_shape[..axis]);
    out_shape.push(n_inputs);
    out_shape.extend_from_slice(&input_shape[axis..]);

    let out_strides = row_major_strides(&out_shape)?;
    let in_strides = row_major_strides(input_shape)?;

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

    // Decompose tid into output coords, extract the stack axis coord,
    // and compute the input linear index from the remaining coords.
    let mut body = String::new();
    body.push_str("    uint in_idx = 0;\n");
    body.push_str("    uint which_input = 0;\n");
    body.push_str("    uint remainder = tid;\n");

    let mut in_dim = 0;
    for (out_dim, &stride) in out_strides.iter().enumerate() {
        let s = safe_msl_uint(stride)?;
        body.push_str(&format!("    uint c{out_dim} = remainder / {s};\n"));
        body.push_str(&format!("    remainder = remainder % {s};\n"));

        if out_dim == axis {
            // This is the stacked dimension — determines which input buffer.
            body.push_str(&format!("    which_input = c{out_dim};\n"));
        } else {
            // Map to input coordinate.
            let in_s = safe_msl_uint(in_strides[in_dim])?;
            body.push_str(&format!("    in_idx += c{out_dim} * {in_s};\n"));
            in_dim += 1;
        }
    }

    // Select the correct input buffer based on which_input.
    let mut select = String::new();
    for i in 0..n_inputs {
        if i == 0 {
            select.push_str(&format!(
                "    {t} val = (which_input == 0) ? input0[in_idx]"
            ));
        } else if i == n_inputs - 1 {
            select.push_str(&format!(" : input{i}[in_idx];\n"));
        } else {
            select.push_str(&format!(" : (which_input == {i}) ? input{i}[in_idx]"));
        }
    }
    // Handle edge case of exactly 1 input (unlikely but safe).
    if n_inputs == 1 {
        select = format!("    {t} val = input0[in_idx];\n");
    }

    Ok(format!(
        r#"[[kernel]] void {name}(
{params}
) {{
    if (tid >= total) return;
{body}{select}    output[tid] = val;
}}"#,
    ))
}

/// Emit MSL for a packed Stack kernel.
///
/// Used when `n_inputs > MAX_DIRECT_BINDING_INPUTS`. All input buffers are
/// packed into one contiguous buffer with element offsets, using only 4
/// buffer slots regardless of input count.
///
/// Buffer layout:
/// - `buffer(0)`: packed input buffer (all inputs concatenated element-wise)
/// - `buffer(1)`: offsets array (`constant uint*`, one per input = element offset)
/// - `buffer(2)`: output buffer
/// - `buffer(3)`: total output elements (uint)
fn emit_stack_kernel_packed(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    n_inputs: usize,
    axis: usize,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);

    // Output shape: input_shape with n_inputs inserted at axis.
    let mut out_shape = Vec::with_capacity(input_shape.len() + 1);
    out_shape.extend_from_slice(&input_shape[..axis]);
    out_shape.push(n_inputs);
    out_shape.extend_from_slice(&input_shape[axis..]);

    let out_strides = row_major_strides(&out_shape)?;
    let in_strides = row_major_strides(input_shape)?;

    // Decompose tid into output coords, extract the stack axis coord,
    // and compute the input linear index from the remaining coords.
    let mut body = String::new();
    body.push_str("    uint in_idx = 0;\n");
    body.push_str("    uint which_input = 0;\n");
    body.push_str("    uint remainder = tid;\n");

    let mut in_dim = 0;
    for (out_dim, &stride) in out_strides.iter().enumerate() {
        let s = safe_msl_uint(stride)?;
        body.push_str(&format!("    uint c{out_dim} = remainder / {s};\n"));
        body.push_str(&format!("    remainder = remainder % {s};\n"));

        if out_dim == axis {
            body.push_str(&format!("    which_input = c{out_dim};\n"));
        } else {
            let in_s = safe_msl_uint(in_strides[in_dim])?;
            body.push_str(&format!("    in_idx += c{out_dim} * {in_s};\n"));
            in_dim += 1;
        }
    }

    // Read from packed buffer using offset: packed_inputs[offsets[which_input] + in_idx]
    body.push_str("    uint base = offsets[which_input];\n");
    body.push_str(&format!("    {t} val = packed_inputs[base + in_idx];\n"));

    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* packed_inputs [[buffer(0)]],
    constant uint* offsets          [[buffer(1)]],
    device {t}* output              [[buffer(2)]],
    constant uint& total            [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total) return;
{body}    output[tid] = val;
}}"#,
    ))
}
