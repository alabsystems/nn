// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP C++ emission for structural tensor ops: Reduce, Broadcast, Narrow,
//! Transpose, and Concat.
//!
//! Parallel to `nn-dsl::codegen_msl_structural*` — each function generates
//! a HIP `__global__` kernel string. Uses shared-memory tree reduction for
//! Reduce, and modular index decomposition for Broadcast/Narrow/Transpose/Concat.

use crate::codegen_hip::{hip_accumulator_type, hip_type, safe_hip_uint, REDUCE_BLOCK_SIZE};
use crate::HipCodegenError;
use nn_dsl::{BroadcastAlignment, ReduceOp, ScalarType};

/// Compute row-major strides for a shape.
///
/// `strides[i] = product(shape[i+1..])`. Last stride is 1.
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

/// Emit HIP kernel for shared-memory tree reduction (Sum, Mean, Max, Min).
///
/// Parallel to `nn-dsl::codegen_msl_tensor_emit::emit_reduce_kernel`.
/// Uses `__shared__` memory and `__syncthreads()` for cooperative reduction.
/// Normal precision tier (named intermediates, no Kahan for HIP PoC).
pub fn emit_reduce_kernel(
    name: &str,
    op: ReduceOp,
    dtype: ScalarType,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let acc = hip_accumulator_type(dtype);
    let needs_cast = t != acc;
    let tg_sz = REDUCE_BLOCK_SIZE;

    let (identity, mean_divisor) = match op {
        ReduceOp::Sum => (format!("({acc})0"), String::new()),
        ReduceOp::Mean => (format!("({acc})0"), format!(" / ({acc})reduce_dim")),
        ReduceOp::Max => ("-HUGE_VALF".to_string(), String::new()),
        ReduceOp::Min => ("HUGE_VALF".to_string(), String::new()),
        _ => {
            return Err(HipCodegenError::UnsupportedIRVariant {
                variant_desc: "ReduceOp",
            })
        }
    };

    let is_extremum = matches!(op, ReduceOp::Max | ReduceOp::Min);

    let load_expr = if needs_cast {
        format!("({acc})input[gid * reduce_dim + i]")
    } else {
        "input[gid * reduce_dim + i]".to_string()
    };

    let (accum_phase1, accum_phase2) = if is_extremum {
        let fn_name = if matches!(op, ReduceOp::Max) {
            "fmaxf"
        } else {
            "fminf"
        };
        (
            format!("        partial = {fn_name}(partial, {load_expr});"),
            format!("            shared[lid] = {fn_name}(shared[lid], shared[lid + stride]);"),
        )
    } else {
        // Normal tier: named intermediates prevent FMA contraction.
        (
            format!("        {acc} val = {load_expr};\n        partial = partial + val;"),
            format!("            {acc} a = shared[lid];\n            {acc} b = shared[lid + stride];\n            shared[lid] = a + b;"),
        )
    };

    let store_expr = if needs_cast {
        format!("({t})(shared[0]{mean_divisor})")
    } else {
        format!("shared[0]{mean_divisor}")
    };

    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    {t}* __restrict__ output,
    const unsigned int reduce_dim,
    const unsigned int outer_size
) {{
    unsigned int gid = blockIdx.x;
    unsigned int lid = threadIdx.x;
    unsigned int tg_sz = blockDim.x;
    if (gid >= outer_size) return;

    __shared__ {acc} shared[{tg_sz}];

    // Phase 1: Each thread accumulates a partial result over its stride
    {acc} partial = {identity};
    for (unsigned int i = lid; i < reduce_dim; i += tg_sz) {{
{accum_phase1}
    }}
    shared[lid] = partial;
    __syncthreads();

    // Phase 2: Tree reduction in shared memory
    for (unsigned int stride = tg_sz / 2; stride > 0; stride >>= 1) {{
        if (lid < stride) {{
{accum_phase2}
        }}
        __syncthreads();
    }}

    // Phase 3: Write result
    if (lid == 0) {{
        output[gid] = {store_expr};
    }}
}}"#,
    ))
}

/// Emit HIP kernel for broadcast via modular index decomposition.
///
/// Maps each output element back to the corresponding input element.
/// Handles both Left-aligned and Right-aligned broadcast patterns.
pub fn emit_broadcast_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    output_shape: &[usize],
    alignment: BroadcastAlignment,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let rank = output_shape.len();
    let input_rank = input_shape.len();

    let mut index_body = String::new();
    index_body.push_str("    unsigned int in_idx = 0;\n");
    index_body.push_str("    unsigned int remainder = tid;\n");

    let out_strides = row_major_strides(output_shape)?;

    let offset = match alignment {
        BroadcastAlignment::Left => 0,
        BroadcastAlignment::Right => rank.saturating_sub(input_rank),
        _ => {
            return Err(HipCodegenError::UnsupportedIRVariant {
                variant_desc: "BroadcastAlignment",
            })
        }
    };

    let in_strides = row_major_strides(input_shape)?;

    for (i, &stride) in out_strides.iter().enumerate() {
        let s = safe_hip_uint(stride)?;
        index_body.push_str(&format!(
            "    unsigned int coord_{i} = remainder / {s};\n    remainder = remainder % {s};\n",
        ));
        let input_idx = if i >= offset { i - offset } else { continue };
        if input_idx < input_rank && input_shape[input_idx] > 1 {
            let in_s = safe_hip_uint(in_strides[input_idx])?;
            index_body.push_str(&format!("    in_idx += coord_{i} * {in_s};\n"));
        }
    }

    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
{index_body}    output[tid] = input[in_idx];
}}"#,
    ))
}

/// Emit HIP kernel for narrow (contiguous slice along one axis).
///
/// Each output element maps to `input[..., coord_axis + start, ...]`.
pub fn emit_narrow_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    axis: usize,
    start: usize,
    length: usize,
) -> Result<String, HipCodegenError> {
    if axis >= input_shape.len() {
        return Err(HipCodegenError::AxisOutOfBounds {
            axis,
            rank: input_shape.len(),
        });
    }
    let t = hip_type(dtype)?;

    let mut out_shape = input_shape.to_vec();
    out_shape[axis] = length;

    let out_strides = row_major_strides(&out_shape)?;
    let in_strides = row_major_strides(input_shape)?;

    let start_str = safe_hip_uint(start)?;
    let mut body = String::new();
    body.push_str("    unsigned int in_idx = 0;\n");
    body.push_str("    unsigned int remainder = tid;\n");

    for (dim, &out_stride) in out_strides.iter().enumerate() {
        let os = safe_hip_uint(out_stride)?;
        let is = safe_hip_uint(in_strides[dim])?;
        body.push_str(&format!("    unsigned int c{dim} = remainder / {os};\n"));
        body.push_str(&format!("    remainder = remainder % {os};\n"));
        if dim == axis {
            body.push_str(&format!("    in_idx += (c{dim} + {start_str}) * {is};\n"));
        } else {
            body.push_str(&format!("    in_idx += c{dim} * {is};\n"));
        }
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

/// Emit HIP kernel for transpose (axis permutation).
///
/// Decomposes output tid into coordinates, maps to input coordinates via
/// the permutation, and reads from the corresponding input flat index.
pub fn emit_transpose_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    axes: &[usize],
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;

    let out_shape: Vec<usize> = axes.iter().map(|&a| input_shape[a]).collect();

    let out_strides = row_major_strides(&out_shape)?;
    let in_strides = row_major_strides(input_shape)?;

    let mut body = String::new();
    body.push_str("    unsigned int in_idx = 0;\n");
    body.push_str("    unsigned int remainder = tid;\n");

    for (out_dim, &out_stride) in out_strides.iter().enumerate() {
        let os = safe_hip_uint(out_stride)?;
        let in_axis = axes[out_dim];
        let is = safe_hip_uint(in_strides[in_axis])?;
        body.push_str(&format!(
            "    unsigned int c{out_dim} = remainder / {os};\n"
        ));
        body.push_str(&format!("    remainder = remainder % {os};\n"));
        body.push_str(&format!("    in_idx += c{out_dim} * {is};\n"));
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

/// Emit HIP kernel for concat along an existing axis.
///
/// Each output thread decomposes tid into (outer, axis_coord, inner),
/// determines which input buffer contributes, computes the per-input
/// flat index, and copies the value.
///
/// HIP has no Metal buffer-index limit, so all inputs are passed as
/// direct pointer parameters (no packed-buffer fallback needed).
pub fn emit_concat_kernel(
    name: &str,
    dtype: ScalarType,
    first_input_shape: &[usize],
    input_axis_sizes: &[usize],
    axis: usize,
) -> Result<String, HipCodegenError> {
    let n_inputs = input_axis_sizes.len();
    if n_inputs == 0 {
        return Err(HipCodegenError::EmptyStack);
    }
    if axis >= first_input_shape.len() {
        return Err(HipCodegenError::AxisOutOfBounds {
            axis,
            rank: first_input_shape.len(),
        });
    }
    let t = hip_type(dtype)?;

    let inner_stride: usize = first_input_shape[axis + 1..]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| HipCodegenError::ShapeProductOverflow {
            shape: first_input_shape.to_vec(),
        })?;
    let total_axis: usize = input_axis_sizes.iter().sum();
    let axis_inner = total_axis.checked_mul(inner_stride).ok_or_else(|| {
        HipCodegenError::ShapeProductOverflow {
            shape: first_input_shape.to_vec(),
        }
    })?;

    // Build parameter list.
    let mut params = String::new();
    for i in 0..n_inputs {
        params.push_str(&format!("    const {t}* __restrict__ input{i},\n"));
    }
    params.push_str(&format!("    {t}* __restrict__ output,\n"));
    params.push_str("    const unsigned int total");

    let inner_s = safe_hip_uint(inner_stride)?;
    let axis_inner_s = safe_hip_uint(axis_inner)?;
    let total_axis_s = safe_hip_uint(total_axis)?;

    let mut body = String::new();
    body.push_str(&format!("    unsigned int outer = tid / {axis_inner_s};\n"));
    body.push_str(&format!(
        "    unsigned int axis_coord = (tid / {inner_s}) % {total_axis_s};\n"
    ));
    body.push_str(&format!("    unsigned int inner = tid % {inner_s};\n"));

    // Determine which input buffer (skip for single-input case).
    if n_inputs > 1 {
        body.push_str("    unsigned int which_input = 0;\n");
    }
    body.push_str("    unsigned int local_axis = axis_coord;\n");
    let mut cumsum = 0usize;
    for (i, &sz) in input_axis_sizes.iter().enumerate() {
        cumsum += sz;
        if i < n_inputs - 1 {
            let cs = safe_hip_uint(cumsum)?;
            body.push_str(&format!(
                "    if (axis_coord >= {cs}) {{ which_input = {}; local_axis = axis_coord - {cs}; }}\n",
                i + 1
            ));
        }
    }

    // Compute per-input flat index.
    let mut idx_select = String::new();
    for (i, &axis_size) in input_axis_sizes.iter().enumerate() {
        let per_input_stride =
            safe_hip_uint(axis_size.checked_mul(inner_stride).ok_or_else(|| {
                HipCodegenError::ShapeProductOverflow {
                    shape: first_input_shape.to_vec(),
                }
            })?)?;
        if n_inputs == 1 {
            idx_select = format!(
                "    unsigned int in_idx = outer * {per_input_stride} + local_axis * {inner_s} + inner;\n"
            );
            break;
        }
        if i == 0 {
            idx_select.push_str(&format!(
                "    unsigned int in_idx = (which_input == 0) ? (outer * {per_input_stride} + local_axis * {inner_s} + inner)"
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

    // Select value from correct input buffer.
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
        r#"extern "C" __global__ void {name}(
{params}
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
{body}{idx_select}{val_select}    output[tid] = val;
}}"#,
    ))
}

#[cfg(test)]
#[path = "codegen_hip_tensor_tests_structural.rs"]
mod tests;
