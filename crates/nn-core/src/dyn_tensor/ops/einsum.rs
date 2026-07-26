// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Einstein summation (einsum) for [`DynTensor`].
//!
//! Supports arbitrary contractions, batch dimensions, traces, transposes,
//! and implicit output mode. Notation follows NumPy/PyTorch convention:
//! `"ij,jk->ik"` (matmul), `"ii->"` (trace), `"ij->ji"` (transpose).

use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

/// Parsed einsum notation with input and output subscripts.
#[derive(Debug, Clone)]
pub struct EinsumNotation {
    /// One entry per input tensor, each a list of axis labels.
    pub input_subscripts: Vec<Vec<char>>,
    /// Output axis labels. Empty means scalar output.
    pub output_subscripts: Vec<char>,
}

impl EinsumNotation {
    /// Parse an einsum notation string.
    ///
    /// Accepted forms:
    /// - Explicit: `"ij,jk->ik"`
    /// - Implicit: `"ij,jk"` (output = sorted unique non-contracted indices)
    pub fn parse(notation: &str) -> Result<Self> {
        let notation = notation.replace(' ', "");
        if notation.is_empty() {
            return Err(TensorError::Unsupported(
                "einsum: empty notation string".into(),
            ));
        }

        let (inputs_str, output_str) = if let Some((lhs, rhs)) = notation.split_once("->") {
            (lhs, Some(rhs))
        } else {
            (notation.as_str(), None)
        };

        let input_subscripts: Vec<Vec<char>> =
            inputs_str.split(',').map(|s| s.chars().collect()).collect();

        // Validate: all characters must be lowercase ASCII letters.
        for (i, subs) in input_subscripts.iter().enumerate() {
            for &c in subs {
                if !c.is_ascii_lowercase() {
                    return Err(TensorError::Unsupported(format!(
                        "einsum: invalid character '{c}' in input subscript {i} \
                         (only lowercase a-z allowed)"
                    )));
                }
            }
        }

        let output_subscripts = if let Some(out) = output_str {
            let chars: Vec<char> = out.chars().collect();
            for &c in &chars {
                if !c.is_ascii_lowercase() {
                    return Err(TensorError::Unsupported(format!(
                        "einsum: invalid character '{c}' in output subscript \
                         (only lowercase a-z allowed)"
                    )));
                }
            }
            chars
        } else {
            // Implicit mode: output = sorted unique indices that appear exactly
            // once across ALL inputs (non-contracted).
            let mut counts: HashMap<char, usize> = HashMap::new();
            for subs in &input_subscripts {
                for &c in subs {
                    *counts.entry(c).or_insert(0) += 1;
                }
            }
            let mut out: Vec<char> = counts
                .into_iter()
                .filter(|&(_, count)| count == 1)
                .map(|(c, _)| c)
                .collect();
            out.sort_unstable();
            out
        };

        // Validate output indices appear in at least one input.
        let all_input_chars: std::collections::HashSet<char> = input_subscripts
            .iter()
            .flat_map(|s| s.iter().copied())
            .collect();
        for &c in &output_subscripts {
            if !all_input_chars.contains(&c) {
                return Err(TensorError::Unsupported(format!(
                    "einsum: output index '{c}' does not appear in any input subscript"
                )));
            }
        }

        Ok(Self {
            input_subscripts,
            output_subscripts,
        })
    }
}

/// Einstein summation over one or more tensors.
///
/// Supports:
/// - Contraction: `"ij,jk->ik"` (matmul)
/// - Batched contraction: `"bij,bjk->bik"` (batched matmul)
/// - Trace: `"ii->"` (sum of diagonal)
/// - Batch trace: `"bii->b"`
/// - Diagonal extraction: `"ii->i"`
/// - Batch diagonal: `"bii->bi"`
/// - Outer product: `"i,j->ij"`
/// - Batch outer product: `"bi,bj->bij"`
/// - Hadamard (element-wise) multiply: `"ij,ij->ij"`
/// - Matrix-vector: `"ij,j->i"`
/// - Batch matrix-vector: `"bij,bj->bi"`
/// - Transpose: `"ij->ji"`
/// - Row/column sums: `"ij->i"`, `"ij->j"`
/// - Full sum: `"ij->"`
/// - Implicit output: `"ij,jk"` = `"ij,jk->ik"`
///
/// Common patterns are dispatched to optimized fast paths that use direct
/// ndarray indexing instead of the generic O(prod(all_dims)) loop.
///
/// All computation is performed in f32 for numerical precision; the result
/// dtype follows the first input tensor (matching the matmul convention).
///
/// # Errors
///
/// Returns an error if:
/// - The notation is malformed
/// - The number of tensors does not match the number of input subscripts
/// - Dimension sizes are inconsistent for the same index label
/// - Tensors have the wrong rank for their subscript
pub fn einsum(notation: &str, tensors: &[&DynTensor]) -> Result<DynTensor> {
    let parsed = EinsumNotation::parse(notation)?;

    if tensors.len() != parsed.input_subscripts.len() {
        return Err(TensorError::Unsupported(format!(
            "einsum: notation has {} input(s) but {} tensor(s) provided",
            parsed.input_subscripts.len(),
            tensors.len()
        )));
    }

    // Check that each tensor's rank matches its subscript length.
    for (tensor, subs) in tensors.iter().zip(&parsed.input_subscripts) {
        if tensor.rank() != subs.len() {
            return Err(TensorError::RankMismatch {
                expected: subs.len(),
                actual: tensor.rank(),
            });
        }
    }

    // Build a map from index label -> dimension size and validate consistency.
    let mut index_sizes: HashMap<char, usize> = HashMap::new();
    for (tensor, subs) in tensors.iter().zip(&parsed.input_subscripts) {
        for (&label, &dim_size) in subs.iter().zip(tensor.dims().iter()) {
            if let Some(&existing) = index_sizes.get(&label) {
                if existing != dim_size {
                    return Err(TensorError::InvalidShape(format!(
                        "einsum: index '{label}' has inconsistent sizes: \
                         {existing} vs {dim_size}"
                    )));
                }
            } else {
                index_sizes.insert(label, dim_size);
            }
        }
    }

    // Prepare f32 arrays for all inputs.
    let arrays: Vec<ArrayD<f32>> = tensors
        .iter()
        .map(|t| t.to_f32_array())
        .collect::<Result<Vec<_>>>()?;

    let target_dtype = tensors[0].dtype();

    // Try optimized fast paths before falling back to generic loop.
    if let Some(result) = try_fast_path(&parsed, &arrays, &index_sizes) {
        return DynTensor::from_f32_result(result?, target_dtype);
    }

    // --- Generic fallback path ---

    // Collect all unique indices across inputs.
    let output_set: std::collections::HashSet<char> =
        parsed.output_subscripts.iter().copied().collect();

    // Summation (contracted) indices: appear in inputs but not in output.
    let all_indices: Vec<char> = {
        let mut seen = std::collections::HashSet::new();
        let mut ordered = Vec::new();
        for subs in &parsed.input_subscripts {
            for &c in subs {
                if seen.insert(c) {
                    ordered.push(c);
                }
            }
        }
        ordered
    };
    let sum_indices: Vec<char> = all_indices
        .iter()
        .filter(|c| !output_set.contains(c))
        .copied()
        .collect();

    // Compute output shape.
    let output_shape: Vec<usize> = parsed
        .output_subscripts
        .iter()
        .map(|c| index_sizes[c])
        .collect();

    // Total output elements.
    let output_numel: usize = output_shape.iter().product();

    // Build ranges for summation indices.
    let sum_ranges: Vec<usize> = sum_indices.iter().map(|c| index_sizes[c]).collect();

    // Allocate output.
    let mut output_data = vec![0.0f32; output_numel];

    // For each output element, iterate over all summation index combinations,
    // multiply the corresponding input elements, and accumulate.
    //
    // This is a general-purpose O(prod(all_dims)) implementation. Not optimized
    // for specific patterns like matmul (use DynTensor::matmul for that).

    // Precompute: for each output position, what are the output index values?
    // And for each summation combination, what are the full index assignments?
    let output_rank = parsed.output_subscripts.len();

    // Strides for output indices.
    let output_strides: Vec<usize> = {
        let mut strides = vec![1usize; output_rank];
        for i in (0..output_rank).rev() {
            if i + 1 < output_rank {
                strides[i] = strides[i + 1] * output_shape[i + 1];
            }
        }
        strides
    };

    // Total summation combinations.
    let sum_total: usize = sum_ranges.iter().product();

    for out_idx in 0..output_numel {
        // Decode output flat index into per-dimension indices.
        let mut index_values: HashMap<char, usize> = HashMap::new();
        let mut remaining = out_idx;
        for (dim_pos, &label) in parsed.output_subscripts.iter().enumerate() {
            let stride = if dim_pos < output_strides.len() {
                output_strides[dim_pos]
            } else {
                1
            };
            let idx = remaining / stride;
            remaining %= stride;
            index_values.insert(label, idx);
        }

        let mut acc = 0.0f32;

        // Iterate over all summation index combinations.
        for sum_flat in 0..sum_total {
            // Decode summation flat index.
            let mut s_rem = sum_flat;
            for (s_pos, &label) in sum_indices.iter().enumerate() {
                let stride: usize = sum_ranges[s_pos + 1..].iter().product();
                let idx = s_rem / stride.max(1);
                s_rem %= stride.max(1);
                index_values.insert(label, idx);
            }

            // Multiply elements from each input tensor.
            let mut product = 1.0f32;
            for (tensor_idx, subs) in parsed.input_subscripts.iter().enumerate() {
                let arr = &arrays[tensor_idx];
                let nd_index: Vec<usize> = subs.iter().map(|c| index_values[c]).collect();
                product *= arr[IxDyn(&nd_index)];
            }
            acc += product;
        }

        output_data[out_idx] = acc;
    }

    // Build output tensor.
    let result_arr = if output_shape.is_empty() {
        // Scalar output: shape []
        ArrayD::from_shape_vec(IxDyn(&[]), output_data)?
    } else {
        ArrayD::from_shape_vec(IxDyn(&output_shape), output_data)?
    };

    DynTensor::from_f32_result(result_arr, target_dtype)
}

// ---------------------------------------------------------------------------
// Optimized fast paths for common einsum patterns.
//
// Each returns `Some(Ok(ArrayD))` on a match, `Some(Err)` on a matched
// pattern with a runtime error, or `None` to fall through to the generic loop.
// ---------------------------------------------------------------------------

/// Classify the einsum pattern and dispatch to an optimized implementation
/// when one exists. Returns `None` to fall through to the generic path.
fn try_fast_path(
    parsed: &EinsumNotation,
    arrays: &[ArrayD<f32>],
    index_sizes: &HashMap<char, usize>,
) -> Option<Result<ArrayD<f32>>> {
    let ins = &parsed.input_subscripts;
    let out = &parsed.output_subscripts;
    let n_inputs = ins.len();

    // --- Single-input patterns ---
    if n_inputs == 1 {
        let subs = &ins[0];
        let a = &arrays[0];

        // Trace: "ii->" — sum of diagonal.
        if subs.len() == 2 && subs[0] == subs[1] && out.is_empty() {
            return Some(fast_trace(a, index_sizes[&subs[0]]));
        }

        // Diagonal extraction: "ii->i"
        if subs.len() == 2 && subs[0] == subs[1] && out.len() == 1 && out[0] == subs[0] {
            return Some(fast_diagonal(a, index_sizes[&subs[0]]));
        }

        // Batch trace: "bii->b"
        if subs.len() == 3
            && subs[1] == subs[2]
            && subs[0] != subs[1]
            && out.len() == 1
            && out[0] == subs[0]
        {
            let b = index_sizes[&subs[0]];
            let n = index_sizes[&subs[1]];
            return Some(fast_batch_trace(a, b, n));
        }

        // Batch diagonal: "bii->bi"
        if subs.len() == 3
            && subs[1] == subs[2]
            && subs[0] != subs[1]
            && out.len() == 2
            && out[0] == subs[0]
            && out[1] == subs[1]
        {
            let b = index_sizes[&subs[0]];
            let n = index_sizes[&subs[1]];
            return Some(fast_batch_diagonal(a, b, n));
        }

        // No single-input fast path matched.
        return None;
    }

    // --- Two-input patterns ---
    if n_inputs == 2 {
        let s0 = &ins[0];
        let s1 = &ins[1];
        let a = &arrays[0];
        let b = &arrays[1];

        // Outer product: "i,j->ij"
        if s0.len() == 1
            && s1.len() == 1
            && s0[0] != s1[0]
            && out.len() == 2
            && out[0] == s0[0]
            && out[1] == s1[0]
        {
            let m = index_sizes[&s0[0]];
            let n = index_sizes[&s1[0]];
            return Some(fast_outer_product(a, b, m, n));
        }

        // Batch outer product: "bi,bj->bij"
        if s0.len() == 2
            && s1.len() == 2
            && s0[0] == s1[0]
            && s0[1] != s1[1]
            && out.len() == 3
            && out[0] == s0[0]
            && out[1] == s0[1]
            && out[2] == s1[1]
        {
            let batch = index_sizes[&s0[0]];
            let m = index_sizes[&s0[1]];
            let n = index_sizes[&s1[1]];
            return Some(fast_batch_outer_product(a, b, batch, m, n));
        }

        // Hadamard (element-wise) multiply: subscripts identical, output identical.
        // Generalised: works for any rank where both inputs and output share the
        // exact same subscripts (e.g., "ij,ij->ij", "ijk,ijk->ijk").
        if s0 == s1 && out == s0.as_slice() {
            return Some(fast_hadamard(a, b));
        }

        // Matrix-vector: "ij,j->i"
        if s0.len() == 2 && s1.len() == 1 && s0[1] == s1[0] && out.len() == 1 && out[0] == s0[0] {
            let m = index_sizes[&s0[0]];
            let k = index_sizes[&s0[1]];
            return Some(fast_matvec(a, b, m, k));
        }

        // Batch matrix-vector: "bij,bj->bi"
        if s0.len() == 3
            && s1.len() == 2
            && s0[0] == s1[0]
            && s0[2] == s1[1]
            && s0[1] != s0[2]
            && out.len() == 2
            && out[0] == s0[0]
            && out[1] == s0[1]
        {
            let batch = index_sizes[&s0[0]];
            let m = index_sizes[&s0[1]];
            let k = index_sizes[&s0[2]];
            return Some(fast_batch_matvec(a, b, batch, m, k));
        }

        // Dot product: "i,i->"
        if s0.len() == 1 && s1.len() == 1 && s0[0] == s1[0] && out.is_empty() {
            return Some(fast_dot(a, b, index_sizes[&s0[0]]));
        }

        // No two-input fast path matched.
        return None;
    }

    // More than 2 inputs: no fast path.
    None
}

// -- Fast path implementations ------------------------------------------------

/// Trace of a square matrix: `ii->` => sum of diagonal.
fn fast_trace(a: &ArrayD<f32>, n: usize) -> Result<ArrayD<f32>> {
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[IxDyn(&[i, i])];
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&[]), vec![sum])?)
}

/// Extract diagonal of a square matrix: `ii->i`.
fn fast_diagonal(a: &ArrayD<f32>, n: usize) -> Result<ArrayD<f32>> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(a[IxDyn(&[i, i])]);
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&[n]), data)?)
}

/// Batch trace: `bii->b` => per-batch sum of diagonal.
fn fast_batch_trace(a: &ArrayD<f32>, batch: usize, n: usize) -> Result<ArrayD<f32>> {
    let mut data = Vec::with_capacity(batch);
    for bi in 0..batch {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += a[IxDyn(&[bi, i, i])];
        }
        data.push(sum);
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&[batch]), data)?)
}

/// Batch diagonal: `bii->bi` => per-batch diagonal extraction.
fn fast_batch_diagonal(a: &ArrayD<f32>, batch: usize, n: usize) -> Result<ArrayD<f32>> {
    let mut data = Vec::with_capacity(batch * n);
    for bi in 0..batch {
        for i in 0..n {
            data.push(a[IxDyn(&[bi, i, i])]);
        }
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&[batch, n]), data)?)
}

/// Outer product: `i,j->ij`.
fn fast_outer_product(a: &ArrayD<f32>, b: &ArrayD<f32>, m: usize, n: usize) -> Result<ArrayD<f32>> {
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        let ai = a[IxDyn(&[i])];
        for j in 0..n {
            data.push(ai * b[IxDyn(&[j])]);
        }
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&[m, n]), data)?)
}

/// Batch outer product: `bi,bj->bij`.
fn fast_batch_outer_product(
    a: &ArrayD<f32>,
    b: &ArrayD<f32>,
    batch: usize,
    m: usize,
    n: usize,
) -> Result<ArrayD<f32>> {
    let mut data = Vec::with_capacity(batch * m * n);
    for bi in 0..batch {
        for i in 0..m {
            let ai = a[IxDyn(&[bi, i])];
            for j in 0..n {
                data.push(ai * b[IxDyn(&[bi, j])]);
            }
        }
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&[batch, m, n]), data)?)
}

/// Hadamard (element-wise) multiply with matching subscripts.
fn fast_hadamard(a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
    Ok(a * b)
}

/// Matrix-vector multiply: `ij,j->i`.
fn fast_matvec(a: &ArrayD<f32>, b: &ArrayD<f32>, m: usize, k: usize) -> Result<ArrayD<f32>> {
    let mut data = Vec::with_capacity(m);
    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += a[IxDyn(&[i, j])] * b[IxDyn(&[j])];
        }
        data.push(sum);
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&[m]), data)?)
}

/// Batch matrix-vector multiply: `bij,bj->bi`.
fn fast_batch_matvec(
    a: &ArrayD<f32>,
    b: &ArrayD<f32>,
    batch: usize,
    m: usize,
    k: usize,
) -> Result<ArrayD<f32>> {
    let mut data = Vec::with_capacity(batch * m);
    for bi in 0..batch {
        for i in 0..m {
            let mut sum = 0.0f32;
            for j in 0..k {
                sum += a[IxDyn(&[bi, i, j])] * b[IxDyn(&[bi, j])];
            }
            data.push(sum);
        }
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&[batch, m]), data)?)
}

/// Dot product: `i,i->`.
fn fast_dot(a: &ArrayD<f32>, b: &ArrayD<f32>, n: usize) -> Result<ArrayD<f32>> {
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[IxDyn(&[i])] * b[IxDyn(&[i])];
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&[]), vec![sum])?)
}

#[cfg(test)]
#[path = "tests_einsum.rs"]
mod tests;
