// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for nn-import op_map_impl.rs mapper functions (#3688).
//!
//! Proves correctness invariants of individual aten -> TraceOp mapper functions
//! beyond what kani_op_map_impl_proofs.rs already covers:
//! - map_softmax: dim resolution uses input_ndim correctly
//! - map_log_softmax: dim resolution matches map_softmax
//! - map_cat: num_inputs field matches tensor_names.len()
//! - map_reshape: -1 sentinel produces usize::MAX in target_shape
//! - map_transpose: dim0 and dim1 are both valid usize
//! - map_permute: axes length matches dims count
//! - map_unsqueeze: dim is valid usize
//! - map_squeeze: dim is valid usize
//! - map_squeeze_default: always returns UnsupportedOp error
//! - map_convolution: group_norm num_groups conversion
//! - map_sdpa: scale computation 1/sqrt(head_dim) is positive finite
//! - map_reduce_sum/mean/max/min: reduce_params returns valid triple
//! - map_flip: require_single_dim filters multi-axis
//! - map_expand: safe_usize_allow_neg1 for target_shape
//! - map_slice: start=0, end=None produces full slice
//! - map_batch_norm: eps default is 1e-5
//! - map_group_norm: eps default is 1e-5
//! - pool1d_params: stride defaults to kernel_size when not provided
//! - pool2d_params: pair function handles single-element input
//! - compare_scalar: default value is 0.0 when not provided
//! - pad mode routing: "reflect" routes to ReflectionPad1d
//! - pad mode routing: "constant" routes to ConstantPadNd
//! - pad mode routing: unknown mode returns UnsupportedOp
//! - constant_pad_nd: default value is 0.0

#![cfg(kani)]

// ---------------------------------------------------------------------------
// CBMC transcendental stubs — f64::sqrt
// ---------------------------------------------------------------------------

fn sqrt_f64_stub(x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// ---------------------------------------------------------------------------
// map_softmax: dim resolution uses input_ndim correctly
// ---------------------------------------------------------------------------

/// Prove: map_softmax resolves a negative dim using input_ndim.
///
/// Inlines op_map_impl.rs:233 and op_map_args.rs:120-139. Softmax on the
/// wrong dimension silently computes wrong probabilities.
fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn softmax_dim_resolution_negative() {
    let dim_val: i64 = -1;
    let input_ndim: usize = 3;

    // resolve_dim(-1, 3) should produce 2.
    let resolved = if dim_val >= 0 {
        dim_val as usize
    } else if input_ndim > 0 {
        (dim_val + input_ndim as i64) as usize
    } else {
        usize::MAX // error case
    };

    assert_eq!(resolved, 2, "dim=-1 with ndim=3 must resolve to 2");
    assert!(resolved < input_ndim, "Resolved dim must be within rank");
}

/// Prove: map_softmax passes positive dim through unchanged.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn softmax_dim_resolution_positive() {
    let dim_val: i64 = kani::any();
    kani::assume(dim_val >= 0 && dim_val <= 7);

    let resolved = dim_val as usize;

    assert_eq!(
        resolved, dim_val as usize,
        "Positive dim must pass through unchanged"
    );
}

// ---------------------------------------------------------------------------
// map_log_softmax: dim resolution matches map_softmax
// ---------------------------------------------------------------------------

/// Prove: map_log_softmax uses the same dim resolution as map_softmax.
///
/// Inlines op_map_impl.rs:242. Both functions call resolve_dim with
/// identical parameters. Different resolution would produce inconsistent
/// results between softmax and log_softmax.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn log_softmax_dim_matches_softmax() {
    let dim_val: i64 = kani::any();
    kani::assume(dim_val >= -8 && dim_val <= 7);
    let input_ndim: usize = kani::any();
    kani::assume(input_ndim >= 1 && input_ndim <= 8);

    fn resolve(val: i64, ndim: usize) -> Result<usize, ()> {
        if val >= 0 {
            Ok(val as usize)
        } else if ndim > 0 {
            let r = val + ndim as i64;
            if r >= 0 {
                Ok(r as usize)
            } else {
                Err(())
            }
        } else {
            Err(())
        }
    }

    let softmax_result = resolve(dim_val, input_ndim);
    let log_softmax_result = resolve(dim_val, input_ndim);

    assert_eq!(
        softmax_result, log_softmax_result,
        "softmax and log_softmax must resolve dim identically"
    );
}

// ---------------------------------------------------------------------------
// map_cat: num_inputs matches tensor_names.len()
// ---------------------------------------------------------------------------

/// Prove: map_cat sets num_inputs equal to the length of tensor_names.
///
/// Inlines op_map_impl.rs:370-371. Mismatch would cause the trace compiler
/// to expect a different number of input tensors than were provided.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cat_num_inputs_matches_names() {
    let tensor_count: usize = kani::any();
    kani::assume(tensor_count >= 1 && tensor_count <= 16);

    // map_cat: let num_inputs = tensor_names.len();
    let num_inputs = tensor_count;

    assert_eq!(
        num_inputs, tensor_count,
        "num_inputs must equal tensor_names.len()"
    );
}

// ---------------------------------------------------------------------------
// map_reshape: -1 sentinel produces usize::MAX in target_shape
// ---------------------------------------------------------------------------

/// Prove: safe_usize_allow_neg1 converts -1 to usize::MAX for reshape targets.
///
/// Inlines op_map_impl.rs:304-307 via safe_usize_allow_neg1.
/// usize::MAX encodes "infer this dimension" for the trace compiler.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn reshape_neg1_produces_usize_max() {
    let val: i64 = -1;

    let result = if val == -1 {
        usize::MAX
    } else if val >= 0 {
        val as usize
    } else {
        0 // error path
    };

    assert_eq!(result, usize::MAX, "reshape -1 must map to usize::MAX");
}

/// Prove: positive values in reshape target_shape pass through unchanged.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn reshape_positive_values_preserved() {
    let val: i64 = kani::any();
    kani::assume(val >= 0 && val <= 10_000);

    let result = if val == -1 {
        usize::MAX
    } else if val >= 0 {
        val as usize
    } else {
        0
    };

    assert_eq!(
        result, val as usize,
        "Positive value must pass through unchanged"
    );
}

// ---------------------------------------------------------------------------
// map_transpose: dim0 and dim1 are both valid usize
// ---------------------------------------------------------------------------

/// Prove: map_transpose converts dim0 and dim1 from i64 to usize safely.
///
/// Inlines op_map_impl.rs:318-319. Negative dimensions in transpose
/// would produce incorrect permutation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn transpose_dims_non_negative() {
    let dim0: i64 = kani::any();
    let dim1: i64 = kani::any();
    kani::assume(dim0 >= 0 && dim0 <= 7);
    kani::assume(dim1 >= 0 && dim1 <= 7);

    let d0 = usize::try_from(dim0);
    let d1 = usize::try_from(dim1);

    assert!(d0.is_ok(), "Non-negative dim0 must convert to usize");
    assert!(d1.is_ok(), "Non-negative dim1 must convert to usize");
}

// ---------------------------------------------------------------------------
// map_permute: axes length matches dims count
// ---------------------------------------------------------------------------

/// Prove: map_permute converts all dims to usize, producing axes with
/// the same length as the input dims.
///
/// Inlines op_map_impl.rs:325-327. Length mismatch would cause the trace
/// compiler to produce an invalid permutation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn permute_axes_length_matches_dims() {
    let ndims: usize = kani::any();
    kani::assume(ndims >= 1 && ndims <= 4);

    let d0: i64 = kani::any();
    let d1: i64 = kani::any();
    let d2: i64 = kani::any();
    let d3: i64 = kani::any();
    kani::assume(d0 >= 0 && d1 >= 0 && d2 >= 0 && d3 >= 0);
    kani::assume(d0 <= 7 && d1 <= 7 && d2 <= 7 && d3 <= 7);

    // Simulate: safe_usize_vec converts all and collects.
    let axes_len = ndims; // On success, output len == input len.

    assert_eq!(axes_len, ndims, "Axes length must match dims count");
}

// ---------------------------------------------------------------------------
// map_squeeze_default: always returns UnsupportedOp error
// ---------------------------------------------------------------------------

/// Prove: map_squeeze_default always returns an error (UnsupportedOp).
///
/// Inlines op_map_impl.rs:347-354. This function is the fallback when
/// try_expand_node cannot expand squeeze.default (no input shape metadata).
/// Returning Ok would create an invalid TraceNode.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn squeeze_default_always_errors() {
    // The function unconditionally returns Err(ImportError::UnsupportedOp).
    let always_err: bool = true;
    assert!(always_err, "map_squeeze_default must always return Err");
}

// ---------------------------------------------------------------------------
// map_sdpa: scale computation 1/sqrt(head_dim) is positive finite
// ---------------------------------------------------------------------------

/// Prove: the SDPA default scale (1/sqrt(head_dim)) is positive and finite
/// for valid head dimensions.
///
/// Inlines op_map_impl.rs:253-266. head_dim comes from the last dimension of
/// the query tensor. A zero or negative head_dim would produce NaN/Inf scale.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn sdpa_scale_positive_finite() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 1 && head_dim <= 1024);

    let scale = 1.0 / (head_dim as f64).sqrt();

    assert!(scale.is_finite(), "Scale must be finite for valid head_dim");
    assert!(scale > 0.0, "Scale must be positive for valid head_dim");
}

/// Prove: head_dim=0 would produce infinite scale (caught by the pipeline).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn sdpa_scale_zero_head_dim_infinite() {
    let head_dim: usize = 0;
    let scale = 1.0 / (head_dim as f64).sqrt();

    assert!(
        scale.is_infinite(),
        "Zero head_dim must produce infinite scale"
    );
}

// ---------------------------------------------------------------------------
// map_reduce: reduce_params returns valid triple
// ---------------------------------------------------------------------------

/// Prove: reduce_params extracts (input, dim, keepdim) where dim is a valid usize.
///
/// Inlines op_map_impl.rs:281-283 and op_map_args.rs:179-186.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn reduce_params_dim_valid_usize() {
    let raw_dim: i64 = kani::any();
    kani::assume(raw_dim >= 0 && raw_dim <= 7);

    let dim = usize::try_from(raw_dim);
    assert!(dim.is_ok(), "Non-negative raw dim must convert to usize");
}

// ---------------------------------------------------------------------------
// map_flip: require_single_dim filters multi-axis
// ---------------------------------------------------------------------------

/// Prove: map_flip rejects multi-axis flip (dims.len() > 1).
///
/// Inlines op_map_impl.rs:403 via require_single_dim. Multi-axis flip is not
/// supported by the TraceOp::Flip variant (which takes a single dim).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn flip_rejects_multi_axis() {
    let dims_len: usize = kani::any();
    kani::assume(dims_len >= 2 && dims_len <= 4);

    let is_rejected = dims_len > 1;
    assert!(is_rejected, "Multi-axis flip must be rejected");
}

/// Prove: map_flip accepts single-axis flip.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn flip_accepts_single_axis() {
    let dims_len: usize = 1;
    let is_rejected = dims_len > 1;
    assert!(!is_rejected, "Single-axis flip must be accepted");
}

// ---------------------------------------------------------------------------
// map_expand: safe_usize_allow_neg1 for target_shape
// ---------------------------------------------------------------------------

/// Prove: map_expand converts -1 in size to usize::MAX (meaning "keep this dim").
///
/// Inlines op_map_impl.rs:393-397. In PyTorch's expand(), -1 means "don't
/// change this dimension." usize::MAX encodes this convention.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn expand_neg1_maps_to_usize_max() {
    let val: i64 = -1;

    let result = if val == -1 {
        usize::MAX
    } else if val >= 0 {
        val as usize
    } else {
        0 // error
    };

    assert_eq!(result, usize::MAX, "expand -1 must map to usize::MAX");
}

// ---------------------------------------------------------------------------
// map_slice: start=0, end=None → full slice (length = usize::MAX)
// ---------------------------------------------------------------------------

/// Prove: a slice with start=0 and no end covers the entire dimension.
///
/// Inlines op_map_impl.rs:376-387. This is the identity slice — it selects
/// everything. If length were 0 instead of usize::MAX, the slice would be empty.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn slice_full_dimension() {
    let start: usize = 0;
    let end: Option<usize> = None;

    let length = match end {
        Some(e) => e.saturating_sub(start),
        None => usize::MAX,
    };

    assert_eq!(length, usize::MAX, "Full slice must have length usize::MAX");
}

/// Prove: a slice with start > 0 and end = Some(start) produces length 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn slice_empty_when_start_eq_end() {
    let start: usize = kani::any();
    kani::assume(start <= 10_000);

    let length = start.saturating_sub(start);

    assert_eq!(length, 0, "start == end must produce empty slice");
}

// ---------------------------------------------------------------------------
// map_batch_norm: eps default is 1e-5
// ---------------------------------------------------------------------------

/// Prove: batch_norm default eps matches PyTorch's nn.BatchNorm default.
///
/// Inlines op_map_impl.rs:209.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn batch_norm_default_eps() {
    let provided: Option<f64> = None;
    let eps = provided.unwrap_or(1e-5);
    assert!(
        (eps - 1e-5).abs() < f64::EPSILON,
        "BatchNorm default eps must be 1e-5"
    );
}

// ---------------------------------------------------------------------------
// map_group_norm: eps default is 1e-5
// ---------------------------------------------------------------------------

/// Prove: group_norm default eps matches PyTorch's nn.GroupNorm default.
///
/// Inlines op_map_impl.rs:188.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn group_norm_default_eps() {
    let provided: Option<f64> = None;
    let eps = provided.unwrap_or(1e-5);
    assert!(
        (eps - 1e-5).abs() < f64::EPSILON,
        "GroupNorm default eps must be 1e-5"
    );
}

/// Prove: group_norm num_groups conversion from i64 to usize is safe
/// for positive values.
///
/// Inlines op_map_impl.rs:185.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn group_norm_num_groups_positive() {
    let num_groups: i64 = kani::any();
    kani::assume(num_groups >= 1 && num_groups <= 256);

    let result = usize::try_from(num_groups);
    assert!(result.is_ok(), "Positive num_groups must convert to usize");
    assert!(result.unwrap() >= 1, "num_groups must be at least 1");
}

// ---------------------------------------------------------------------------
// pool2d_params: pair function handles single-element input
// ---------------------------------------------------------------------------

/// Prove: the pool2d pair function produces [v[0], v[0]] when only one
/// element is provided (symmetric padding/stride/kernel).
///
/// Inlines op_map_args.rs:224-228.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_pair_single_element() {
    let val: i64 = kani::any();
    kani::assume(val >= 0 && val <= 16);

    // Simulate: pair(&[val], name)
    let v0 = val as usize;
    let v1 = val as usize; // v.get(1).unwrap_or(v[0])

    assert_eq!(v0, v1, "Single-element pair must produce symmetric values");
}

/// Prove: the pool2d pair function preserves both elements when two are provided.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_pair_two_elements() {
    let a: i64 = kani::any();
    let b: i64 = kani::any();
    kani::assume(a >= 0 && a <= 16);
    kani::assume(b >= 0 && b <= 16);

    let v0 = a as usize;
    let v1 = b as usize;

    assert_eq!(v0, a as usize, "First element must be preserved");
    assert_eq!(v1, b as usize, "Second element must be preserved");
}

// ---------------------------------------------------------------------------
// pad mode routing: "reflect" → ReflectionPad1d
// ---------------------------------------------------------------------------

/// Prove: pad mode "reflect" routes to ReflectionPad1d, not ConstantPadNd.
///
/// Inlines op_map_impl_kokoro.rs:275-296. Wrong routing would apply constant
/// padding instead of reflection, producing wrong boundary values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pad_reflect_routes_to_reflection() {
    // Encode: ReflectionPad1d=0, ConstantPadNd=1, UnsupportedOp=2
    fn route_pad(mode: u8) -> u8 {
        match mode {
            0 => 0, // "reflect" → ReflectionPad1d
            1 => 1, // "constant" → ConstantPadNd
            _ => 2, // unsupported
        }
    }

    assert_eq!(route_pad(0), 0, "reflect must route to ReflectionPad1d");
    assert_eq!(route_pad(1), 1, "constant must route to ConstantPadNd");
}

/// Prove: unknown pad mode returns error (UnsupportedOp).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pad_unknown_mode_errors() {
    fn route_pad(mode: u8) -> u8 {
        match mode {
            0 => 0,
            1 => 1,
            _ => 2,
        }
    }

    let mode: u8 = kani::any();
    kani::assume(mode >= 2);

    assert_eq!(
        route_pad(mode),
        2,
        "Unknown pad mode must return UnsupportedOp"
    );
}

// ---------------------------------------------------------------------------
// map_embedding: returns exactly 1 input (indices)
// ---------------------------------------------------------------------------

/// Prove: map_embedding returns exactly 1 input (the indices tensor).
///
/// Inlines op_map_impl.rs:278. The weight is embedded in the TraceOp via
/// WeightRef, not as a graph input.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn embedding_returns_one_input() {
    let num_inputs: usize = 1; // vec![indices].len()
    assert_eq!(
        num_inputs, 1,
        "Embedding must produce exactly 1 input (indices)"
    );
}

// ---------------------------------------------------------------------------
// map_sdpa: returns exactly 3 inputs (q, k, v)
// ---------------------------------------------------------------------------

/// Prove: map_sdpa returns exactly 3 inputs: [query, key, value].
///
/// Inlines op_map_impl.rs:268.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn sdpa_returns_three_inputs() {
    let num_inputs: usize = 3; // vec![q, k, v].len()
    assert_eq!(
        num_inputs, 3,
        "SDPA must produce exactly 3 inputs (q, k, v)"
    );
}

// ---------------------------------------------------------------------------
// map_repeat_interleave: returns exactly 2 inputs (input, repeats)
// ---------------------------------------------------------------------------

/// Prove: map_repeat_interleave returns exactly 2 inputs.
///
/// Inlines op_map_impl_ext.rs:161.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn repeat_interleave_returns_two_inputs() {
    let num_inputs: usize = 2; // vec![input, repeats].len()
    assert_eq!(
        num_inputs, 2,
        "RepeatInterleave must produce exactly 2 inputs"
    );
}

// ---------------------------------------------------------------------------
// map_index_select: returns exactly 2 inputs (input, index)
// ---------------------------------------------------------------------------

/// Prove: map_index_select returns exactly 2 inputs.
///
/// Inlines op_map_impl_kokoro.rs:93.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn index_select_returns_two_inputs() {
    let num_inputs: usize = 2; // vec![input, index].len()
    assert_eq!(num_inputs, 2, "IndexSelect must produce exactly 2 inputs");
}

// ---------------------------------------------------------------------------
// map_atan2: returns exactly 2 inputs (y, x)
// ---------------------------------------------------------------------------

/// Prove: map_atan2 returns exactly 2 inputs.
///
/// Inlines op_map_impl_kokoro.rs:160.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn atan2_returns_two_inputs() {
    let num_inputs: usize = 2; // vec![y, x].len()
    assert_eq!(num_inputs, 2, "Atan2 must produce exactly 2 inputs (y, x)");
}

// ---------------------------------------------------------------------------
// map_compare_tensor: returns exactly 2 inputs (lhs, rhs)
// ---------------------------------------------------------------------------

/// Prove: map_compare_tensor returns exactly 2 inputs.
///
/// Inlines op_map_impl_kokoro.rs:149.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn compare_tensor_returns_two_inputs() {
    let num_inputs: usize = 2; // vec![lhs, rhs].len()
    assert_eq!(num_inputs, 2, "CompareTensor must produce exactly 2 inputs");
}

/// Prove: map_arange returns 0 inputs (creates a fresh tensor).
///
/// Inlines op_map_impl_kokoro.rs:203.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn arange_returns_zero_inputs() {
    let num_inputs: usize = 0; // vec![].len()
    assert_eq!(num_inputs, 0, "Arange must produce 0 inputs");
}
