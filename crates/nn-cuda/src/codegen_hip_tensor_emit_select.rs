// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP C++ emission for selection/rearrangement ops: AxisSelect, Stack.
//!
//! Extracted from `codegen_hip_tensor_emit_structural` to keep files under
//! the 500-line limit. AxisSelect removes a dimension; Stack inserts one
//! from multiple same-shape inputs.

use crate::codegen_hip::{hip_type, safe_hip_uint};
use crate::HipCodegenError;
use nn_dsl::ScalarType;

/// Compute row-major strides for a shape (duplicated from structural for
/// module independence — both files need this, extraction to a shared helper
/// is a follow-up).
fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>, HipCodegenError> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].checked_mul(shape[i + 1]).ok_or_else(|| {
            HipCodegenError::ShapeProductOverflow {
                shape: shape.to_vec(),
            }
        })?;
    }
    Ok(strides)
}

/// Emit HIP kernel for axis selection (removing one dimension by fixing its index).
///
/// Output shape is input shape with the `axis` dimension removed.
/// Each output element maps to `input[..., select_index, ...]`.
pub fn emit_axis_select_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    axis: usize,
    select_index: usize,
) -> Result<String, HipCodegenError> {
    if axis >= input_shape.len() {
        return Err(HipCodegenError::AxisOutOfBounds {
            axis,
            rank: input_shape.len(),
        });
    }
    let t = hip_type(dtype)?;

    let mut out_shape: Vec<usize> = input_shape.to_vec();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let out_strides = row_major_strides(&out_shape)?;
    let in_strides = row_major_strides(input_shape)?;
    let select_s = safe_hip_uint(select_index)?;

    let mut body = String::new();
    body.push_str(&format!(
        "    unsigned int in_idx = {select_s} * {};\n",
        safe_hip_uint(in_strides[axis])?
    ));
    body.push_str("    unsigned int remainder = tid;\n");

    let mut out_dim = 0;
    for (in_dim, &in_stride) in in_strides.iter().enumerate() {
        if in_dim == axis {
            continue;
        }
        let os = safe_hip_uint(out_strides[out_dim])?;
        let is = safe_hip_uint(in_stride)?;
        body.push_str(&format!(
            "    unsigned int c{out_dim} = remainder / {os};\n"
        ));
        body.push_str(&format!("    remainder = remainder % {os};\n"));
        body.push_str(&format!("    in_idx += c{out_dim} * {is};\n"));
        out_dim += 1;
    }

    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
{body}    output[tid] = input[in_idx];
}}"#,
    ))
}

/// Emit HIP kernel for stack (insert new axis from multiple same-shape inputs).
///
/// Output shape is `input_shape` with `n_inputs` inserted at `axis`.
/// HIP has no Metal buffer-index limit, so all inputs are passed as
/// direct pointer parameters (no packed-buffer fallback needed).
pub fn emit_stack_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    n_inputs: usize,
    axis: usize,
) -> Result<String, HipCodegenError> {
    if n_inputs == 0 {
        return Err(HipCodegenError::EmptyStack);
    }
    if axis > input_shape.len() {
        return Err(HipCodegenError::AxisOutOfBounds {
            axis,
            rank: input_shape.len() + 1,
        });
    }
    let t = hip_type(dtype)?;

    // Output shape: insert n_inputs at axis.
    let mut out_shape = input_shape.to_vec();
    out_shape.insert(axis, n_inputs);

    let out_strides = row_major_strides(&out_shape)?;
    let in_strides = row_major_strides(input_shape)?;

    // Build parameter list: n_inputs input pointers, then output, then total.
    let mut params = String::new();
    for i in 0..n_inputs {
        params.push_str(&format!("    const {t}* __restrict__ input{i},\n"));
    }
    params.push_str(&format!("    {t}* __restrict__ output,\n"));
    params.push_str("    const unsigned int total");

    // Build index decomposition body.
    let mut body = String::new();
    body.push_str("    unsigned int in_idx = 0;\n");
    body.push_str("    unsigned int which_input = 0;\n");
    body.push_str("    unsigned int remainder = tid;\n");

    let mut in_dim = 0;
    for (out_dim, &out_stride) in out_strides.iter().enumerate() {
        let os = safe_hip_uint(out_stride)?;
        body.push_str(&format!(
            "    unsigned int c{out_dim} = remainder / {os};\n"
        ));
        body.push_str(&format!("    remainder = remainder % {os};\n"));
        if out_dim == axis {
            body.push_str(&format!("    which_input = c{out_dim};\n"));
        } else {
            if in_dim < in_strides.len() {
                let is = safe_hip_uint(in_strides[in_dim])?;
                body.push_str(&format!("    in_idx += c{out_dim} * {is};\n"));
            }
            in_dim += 1;
        }
    }

    // Value selection via ternary chain.
    let mut val_select = String::new();
    if n_inputs == 1 {
        val_select.push_str(&format!("    {t} val = input0[in_idx];\n"));
    } else {
        val_select.push_str(&format!("    {t} val = "));
        for i in 0..n_inputs {
            if i == n_inputs - 1 {
                val_select.push_str(&format!("input{i}[in_idx];\n"));
            } else {
                val_select.push_str(&format!("(which_input == {i}) ? input{i}[in_idx] : "));
            }
        }
    }

    Ok(format!(
        r#"extern "C" __global__ void {name}(
{params}
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
{body}{val_select}    output[tid] = val;
}}"#,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- AxisSelect tests ---

    #[test]
    fn test_axis_select_2d_axis0() {
        let src = emit_axis_select_kernel("asel_0", ScalarType::F32, &[4, 8], 0, 2).unwrap();
        assert!(src.contains("extern \"C\" __global__ void asel_0"));
        assert!(src.contains("in_idx = 2 * 8"));
        assert!(src.contains("output[tid] = input[in_idx]"));
    }

    #[test]
    fn test_axis_select_3d_axis1() {
        let src = emit_axis_select_kernel("asel_1", ScalarType::F32, &[2, 5, 3], 1, 3).unwrap();
        assert!(src.contains("extern \"C\" __global__ void asel_1"));
        assert!(src.contains("in_idx = 3 * 3"));
    }

    #[test]
    fn test_axis_select_axis_oob() {
        let result = emit_axis_select_kernel("bad", ScalarType::F32, &[4, 8], 2, 0);
        assert!(result.is_err());
    }

    // --- Stack tests ---

    #[test]
    fn test_stack_two_inputs_axis0() {
        let src = emit_stack_kernel("stack_a0", ScalarType::F32, &[4, 8], 2, 0).unwrap();
        assert!(src.contains("extern \"C\" __global__ void stack_a0"));
        assert!(src.contains("input0"));
        assert!(src.contains("input1"));
        assert!(src.contains("which_input"));
    }

    #[test]
    fn test_stack_three_inputs_axis1() {
        let src = emit_stack_kernel("stack_a1", ScalarType::F32, &[4, 8], 3, 1).unwrap();
        assert!(src.contains("input0"));
        assert!(src.contains("input1"));
        assert!(src.contains("input2"));
        assert!(src.contains("which_input == 1"));
    }

    #[test]
    fn test_stack_single_input() {
        let src = emit_stack_kernel("stack_one", ScalarType::F32, &[4, 8], 1, 0).unwrap();
        assert!(src.contains("input0[in_idx]"));
    }

    #[test]
    fn test_stack_empty_error() {
        let result = emit_stack_kernel("bad", ScalarType::F32, &[4, 8], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_stack_axis_out_of_bounds() {
        let result = emit_stack_kernel("bad", ScalarType::F32, &[4, 8], 2, 3);
        assert!(result.is_err());
    }
}
