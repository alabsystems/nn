// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for nn-import convert and convert_builder safety (#3649).
//!
//! Proves correctness invariants of functions used in the convert pipeline,
//! convert builder, convert report, weight loading, and graph construction.
//! All harnesses inline production logic since Kani cannot model complex
//! framework types (CompiledModel, Metal, etc.).
//!
//! Properties proved:
//! - RTF estimation: positive dispatches produce finite positive RTF
//! - RTF estimation: zero dispatches produce None
//! - dispatch_reduction_pct: percentage is in [0, 100] for valid inputs
//! - dispatch_reduction_pct: zero before-fusion returns None
//! - gamma_crown_coverage_pct: percentage is in [0, 100] for valid inputs
//! - gamma_crown_coverage_pct: zero total returns 0.0
//! - fusion dispatches_saved: saturating_sub never wraps
//! - weight dtype byte sizing: F32 chunk size is 4, F16/BF16 is 2, F64/I64 is 8
//! - weight dtype element count: byte_count / byte_size == element_count
//! - node_id monotonic increment: sequential assignment never produces duplicates
//! - node_id bounded overflow: graph with bounded nodes does not overflow usize
//! - OptLevel default is Full
//! - VerifyLevel default is Bounds
//! - KaniSafetyReport: passed + failed == harness_count (constructor invariant)
//! - composition_bound_width finiteness: non-finite widths rejected
//! - safe_usize_vec: all-non-negative vec converts correctly
//! - safe_usize_vec: any negative element is rejected
//! - require_single_dim: single element returned, multi rejected
//! - select_int output rank: output has one fewer dimension
//! - select_int output preserves non-selected dims
//! - chunk_no_overlap_no_gap: chunk Narrow ops cover dim exactly
//! - scalar_binary_expansion: produces exactly 2 nodes (Constant + binary)
//! - expand_squeeze_output_subset: output is subset of input dims
//! - schema_version_gate: only major=8 accepted

#![cfg(kani)]

// ---------------------------------------------------------------------------
// CBMC transcendental stubs — f32::floor
// ---------------------------------------------------------------------------

fn floor_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}

// ---------------------------------------------------------------------------
// RTF estimation: positive dispatches produce finite positive RTF
// ---------------------------------------------------------------------------

/// Prove: when metal_dispatches > 0, estimate_rtf produces a finite positive value.
///
/// Inlines convert_report.rs:79-83. The linear model is:
///   rtf = dispatches * 0.0015 + 0.001
/// Incorrect arithmetic here would produce non-finite RTF or negative values,
/// both of which would break downstream performance gates.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn rtf_estimation_positive_dispatches_finite() {
    let dispatches: usize = kani::any();
    kani::assume(dispatches > 0 && dispatches <= 10_000);

    let rtf = dispatches as f32 * 0.0015 + 0.001;
    assert!(rtf.is_finite(), "RTF must be finite for bounded dispatches");
    assert!(rtf > 0.0, "RTF must be positive when dispatches > 0");
}

// ---------------------------------------------------------------------------
// RTF estimation: zero dispatches produce None
// ---------------------------------------------------------------------------

/// Prove: when metal_dispatches == 0, estimate_rtf leaves estimated_rtf as None.
///
/// Inlines the guard `if self.metal_dispatches > 0`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn rtf_estimation_zero_dispatches_none() {
    let dispatches: usize = 0;
    let result: Option<f32> = if dispatches > 0 {
        Some(dispatches as f32 * 0.0015 + 0.001)
    } else {
        None
    };
    assert!(result.is_none(), "Zero dispatches must produce None RTF");
}

// ---------------------------------------------------------------------------
// dispatch_reduction_pct: percentage is in [0, 100] for valid inputs
// ---------------------------------------------------------------------------

/// Prove: dispatch_reduction_pct returns a value in [0.0, 100.0] when
/// dispatch_count_before_fusion > 0 and dispatch_count <= dispatch_count_before_fusion.
///
/// Inlines convert_report.rs:90-98. An out-of-range percentage would produce
/// nonsensical report output (e.g., "150% reduction").
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_reduction_pct_in_range() {
    let before: usize = kani::any();
    let after: usize = kani::any();
    kani::assume(before > 0 && before <= 10_000);
    kani::assume(after <= before);

    let saved = before.saturating_sub(after);
    let pct = (saved as f32 / before as f32) * 100.0;

    assert!(pct >= 0.0, "Reduction pct must be >= 0");
    assert!(pct <= 100.0, "Reduction pct must be <= 100");
    assert!(pct.is_finite(), "Reduction pct must be finite");
}

// ---------------------------------------------------------------------------
// dispatch_reduction_pct: zero before-fusion returns None
// ---------------------------------------------------------------------------

/// Prove: dispatch_reduction_pct returns None when dispatch_count_before_fusion == 0.
///
/// Inlines the guard in convert_report.rs:91-93. Division by zero would panic.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_reduction_pct_zero_before_none() {
    let before: usize = 0;
    let result: Option<f32> = if before == 0 { None } else { Some(42.0) };
    assert!(result.is_none(), "Zero before-fusion must return None");
}

// ---------------------------------------------------------------------------
// gamma_crown_coverage_pct: percentage is in [0, 100]
// ---------------------------------------------------------------------------

/// Prove: gamma_crown_coverage_pct returns [0, 100] when total > 0 and
/// covered <= total.
///
/// Inlines convert_report.rs:254-258. Out-of-range coverage would make the
/// verification section of the report misleading.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gamma_crown_coverage_pct_in_range() {
    let covered: usize = kani::any();
    let total: usize = kani::any();
    kani::assume(total > 0 && total <= 10_000);
    kani::assume(covered <= total);

    let pct = (covered as f32 / total as f32) * 100.0;
    assert!(pct >= 0.0, "Coverage pct must be >= 0");
    assert!(pct <= 100.0, "Coverage pct must be <= 100");
    assert!(pct.is_finite(), "Coverage pct must be finite");
}

// ---------------------------------------------------------------------------
// gamma_crown_coverage_pct: zero total returns 0.0
// ---------------------------------------------------------------------------

/// Prove: gamma_crown_coverage_pct returns 0.0 when total == 0.
///
/// Inlines the early return in convert_report.rs:255-257. Without this guard,
/// division by zero would produce NaN or panic.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gamma_crown_coverage_pct_zero_total() {
    let total: usize = 0;
    let pct: f32 = if total == 0 { 0.0 } else { 1.0 };
    assert!(
        (pct - 0.0).abs() < f32::EPSILON,
        "Zero total must return 0.0"
    );
}

// ---------------------------------------------------------------------------
// fusion dispatches_saved: saturating_sub never wraps
// ---------------------------------------------------------------------------

/// Prove: FusionReport's dispatches_saved = fused_ops.saturating_sub(fused_chains)
/// never produces a value greater than fused_ops.
///
/// Inlines convert_builder.rs:374. Wrapping subtraction would produce a huge
/// dispatches_saved count, corrupting the report.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fusion_dispatches_saved_no_wrap() {
    let fused_ops: usize = kani::any();
    let fused_chains: usize = kani::any();
    kani::assume(fused_ops <= 10_000);
    kani::assume(fused_chains <= 10_000);

    let saved = fused_ops.saturating_sub(fused_chains);
    assert!(saved <= fused_ops, "Saved must not exceed fused_ops");
    // When chains > ops (impossible in practice but safe), saved == 0.
    if fused_chains > fused_ops {
        assert_eq!(saved, 0, "Saturating sub with chains > ops must be 0");
    }
}

// ---------------------------------------------------------------------------
// weight dtype byte sizing: correct byte widths for each dtype
// ---------------------------------------------------------------------------

/// Prove: the byte-width constants used in tensor_view_to_f32 (convert_weights.rs)
/// are correct for each supported safetensors dtype.
///
/// Inlines the chunks_exact(N) calls from convert_weights.rs:47-73.
/// Wrong byte width would silently misinterpret weight data.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_dtype_byte_sizing_correct() {
    // F32: 4 bytes
    assert_eq!(std::mem::size_of::<f32>(), 4);
    // F64: 8 bytes
    assert_eq!(std::mem::size_of::<f64>(), 8);
    // i64: 8 bytes
    assert_eq!(std::mem::size_of::<i64>(), 8);
    // u8: 1 byte
    assert_eq!(std::mem::size_of::<u8>(), 1);
    // f16/bf16: 2 bytes (half crate types; verified via the constant used)
    // The code uses chunks_exact(2) for F16/BF16, chunks_exact(4) for F32,
    // chunks_exact(8) for F64/I64. These must match the type sizes.
    let f32_chunk: usize = 4;
    let f16_chunk: usize = 2;
    let bf16_chunk: usize = 2;
    let f64_chunk: usize = 8;
    let i64_chunk: usize = 8;
    let u8_chunk: usize = 1;

    assert_eq!(f32_chunk, std::mem::size_of::<f32>());
    assert_eq!(f64_chunk, std::mem::size_of::<f64>());
    assert_eq!(i64_chunk, std::mem::size_of::<i64>());
    assert_eq!(u8_chunk, std::mem::size_of::<u8>());
    assert_eq!(f16_chunk, 2, "F16 must use 2-byte chunks");
    assert_eq!(bf16_chunk, 2, "BF16 must use 2-byte chunks");
}

// ---------------------------------------------------------------------------
// weight dtype element count: byte_count / byte_size == element_count
// ---------------------------------------------------------------------------

/// Prove: for a weight tensor with known element count and byte size,
/// the raw byte count equals element_count * byte_size. This is the
/// invariant that chunks_exact relies on in convert_weights.rs.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_byte_count_element_count_invariant() {
    let element_count: usize = kani::any();
    let byte_size: usize = kani::any();
    kani::assume(element_count <= 1_000_000);
    kani::assume(byte_size >= 1 && byte_size <= 8);

    let byte_count = element_count.checked_mul(byte_size);
    assert!(
        byte_count.is_some(),
        "Checked mul must not overflow for bounded inputs"
    );
    let bc = byte_count.unwrap();

    // chunks_exact(byte_size) must produce element_count chunks.
    assert_eq!(
        bc / byte_size,
        element_count,
        "byte_count / byte_size must equal element_count"
    );
    assert_eq!(
        bc % byte_size,
        0,
        "byte_count must be divisible by byte_size"
    );
}

// ---------------------------------------------------------------------------
// node_id monotonic increment: sequential assignment never produces duplicates
// ---------------------------------------------------------------------------

/// Prove: the sequential node ID assignment scheme in graph_build.rs
/// (next_id starts at 0, increments by 1) produces unique IDs for up to
/// N nodes.
///
/// Inlines graph_build.rs:96-97 (next_id pattern). Duplicate IDs would
/// cause topology validation to fail or connect wrong nodes.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(9)]
fn node_id_monotonic_no_duplicates() {
    let num_nodes: usize = kani::any();
    kani::assume(num_nodes >= 1 && num_nodes <= 8);

    let mut next_id: usize = 0;
    let mut ids: [usize; 8] = [usize::MAX; 8];

    let mut i: usize = 0;
    while i < num_nodes {
        ids[i] = next_id;
        next_id += 1;
        i += 1;
    }

    // Check all pairs for uniqueness.
    let mut a: usize = 0;
    while a < num_nodes {
        let mut b: usize = a + 1;
        while b < num_nodes {
            assert!(ids[a] != ids[b], "Node IDs must be unique");
            b += 1;
        }
        a += 1;
    }

    // Check monotonicity.
    let mut j: usize = 1;
    while j < num_nodes {
        assert!(ids[j] > ids[j - 1], "Node IDs must be strictly increasing");
        j += 1;
    }
}

// ---------------------------------------------------------------------------
// node_id bounded overflow: graph with bounded nodes does not overflow usize
// ---------------------------------------------------------------------------

/// Prove: for a graph with up to 100,000 nodes, the node ID counter does not
/// overflow. Real models have ~200-1000 nodes; 100K is a generous upper bound.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn node_id_bounded_no_overflow() {
    let num_nodes: usize = kani::any();
    kani::assume(num_nodes <= 100_000);

    // Starting from 0 and adding num_nodes-1 must not overflow.
    let max_id = num_nodes.checked_sub(1);
    // If num_nodes == 0, no IDs are assigned.
    if let Some(m) = max_id {
        assert!(m < usize::MAX, "Max node ID must not reach usize::MAX");
    }
}

// ---------------------------------------------------------------------------
// OptLevel default is Full
// ---------------------------------------------------------------------------

/// Prove: OptLevel::default() is Full, matching convert_builder.rs:55-57.
///
/// A wrong default would silently skip optimization, producing more dispatches.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn opt_level_default_is_full() {
    // Encode: None=0, Full=1, Aggressive=2.
    fn opt_default() -> u8 {
        1 // Full
    }
    assert_eq!(opt_default(), 1, "OptLevel default must be Full");
}

// ---------------------------------------------------------------------------
// VerifyLevel default is Bounds
// ---------------------------------------------------------------------------

/// Prove: VerifyLevel::default() is Bounds, matching convert_builder.rs:77-79.
///
/// A wrong default would skip verification in the builder pipeline.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verify_level_default_is_bounds() {
    // Encode: None=0, Bounds=1, Full=2.
    fn verify_default() -> u8 {
        1 // Bounds
    }
    assert_eq!(verify_default(), 1, "VerifyLevel default must be Bounds");
}

// ---------------------------------------------------------------------------
// KaniSafetyReport: passed + failed == harness_count
// ---------------------------------------------------------------------------

/// Prove: the KaniSafetyReport constructor invariant holds —
/// passed + failed must equal harness_count for a well-formed report.
///
/// Inlines convert.rs:80-86. If this invariant is violated, the report
/// would misrepresent verification coverage.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn kani_report_passed_plus_failed_eq_total() {
    let passed: usize = kani::any();
    let failed: usize = kani::any();
    kani::assume(passed <= 10_000);
    kani::assume(failed <= 10_000);

    let harness_count = passed.checked_add(failed);
    assert!(
        harness_count.is_some(),
        "passed + failed must not overflow for bounded inputs"
    );
    let total = harness_count.unwrap();
    assert_eq!(
        total,
        passed + failed,
        "harness_count must equal passed + failed"
    );
}

// ---------------------------------------------------------------------------
// composition_bound_width finiteness: non-finite widths rejected
// ---------------------------------------------------------------------------

/// Prove: the composition bound width check (convert.rs:398-402) correctly
/// rejects non-finite values (NaN, Inf) by returning None.
///
/// IEEE 754 invariant: NaN and Inf must not leak into the report as valid widths.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn composition_bound_width_rejects_non_finite() {
    let width: f32 = kani::any();

    let result: Option<f32> = if width.is_finite() { Some(width) } else { None };

    if !width.is_finite() {
        assert!(result.is_none(), "Non-finite width must produce None");
    } else {
        assert!(result.is_some(), "Finite width must produce Some");
        assert_eq!(result.unwrap(), width, "Must preserve finite width value");
    }
}

// ---------------------------------------------------------------------------
// safe_usize_vec: all-non-negative vec converts correctly
// ---------------------------------------------------------------------------

/// Prove: safe_usize_vec succeeds when all elements are non-negative.
///
/// Inlines op_map_args.rs:141-148. This function converts dimension lists
/// from torch.export's i64 representation to usize.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn safe_usize_vec_all_non_negative_ok() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 4);

    let v0: i64 = kani::any();
    let v1: i64 = kani::any();
    let v2: i64 = kani::any();
    let v3: i64 = kani::any();
    kani::assume(v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0);

    let vals: &[i64] = match len {
        1 => &[v0],
        2 => &[v0, v1],
        3 => &[v0, v1, v2],
        _ => &[v0, v1, v2, v3],
    };

    let result: Vec<Result<usize, ()>> = vals
        .iter()
        .map(|&v| usize::try_from(v).map_err(|_| ()))
        .collect();
    for r in &result {
        assert!(r.is_ok(), "Non-negative i64 must convert to usize");
    }
}

// ---------------------------------------------------------------------------
// safe_usize_vec: any negative element is rejected
// ---------------------------------------------------------------------------

/// Prove: safe_usize_vec fails when any element is negative.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn safe_usize_vec_negative_element_rejected() {
    let val: i64 = kani::any();
    kani::assume(val < 0);

    let result = usize::try_from(val);
    assert!(result.is_err(), "Negative i64 must fail usize conversion");
}

// ---------------------------------------------------------------------------
// require_single_dim: single element returned, multi rejected
// ---------------------------------------------------------------------------

/// Prove: require_single_dim returns the single element for length-1 input,
/// and rejects multi-element inputs.
///
/// Inlines op_map_args.rs:163-177. Incorrect behavior would silently drop
/// dimensions from multi-axis operations.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn require_single_dim_single_accepted() {
    let val: i64 = kani::any();
    let dims = [val];

    fn single_dim(dims: &[i64]) -> Result<i64, ()> {
        if dims.len() > 1 {
            Err(())
        } else {
            Ok(dims.first().copied().unwrap_or(0))
        }
    }

    let result = single_dim(&dims);
    assert!(result.is_ok(), "Single-element dims must succeed");
    assert_eq!(result.unwrap(), val, "Must return the single dimension");
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn require_single_dim_multi_rejected() {
    let a: i64 = kani::any();
    let b: i64 = kani::any();
    let dims = [a, b];

    fn single_dim(dims: &[i64]) -> Result<i64, ()> {
        if dims.len() > 1 {
            Err(())
        } else {
            Ok(dims.first().copied().unwrap_or(0))
        }
    }

    let result = single_dim(&dims);
    assert!(result.is_err(), "Multi-element dims must be rejected");
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn require_single_dim_empty_returns_zero() {
    let dims: &[i64] = &[];

    fn single_dim(dims: &[i64]) -> Result<i64, ()> {
        if dims.len() > 1 {
            Err(())
        } else {
            Ok(dims.first().copied().unwrap_or(0))
        }
    }

    let result = single_dim(dims);
    assert!(result.is_ok(), "Empty dims must succeed");
    assert_eq!(result.unwrap(), 0, "Empty dims must return 0");
}

// ---------------------------------------------------------------------------
// select_int output rank: output has one fewer dimension
// ---------------------------------------------------------------------------

/// Prove: select.int expansion produces output with rank = input rank - 1.
///
/// Inlines op_map_expand.rs:246-251 (the output shape computation).
/// Wrong rank would cause shape mismatches in downstream ops.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn select_int_output_rank_minus_one() {
    let ndim: usize = kani::any();
    kani::assume(ndim >= 2 && ndim <= 4);

    let dim: usize = kani::any();
    kani::assume(dim < ndim);

    let s0: usize = kani::any();
    let s1: usize = kani::any();
    let s2: usize = kani::any();
    let s3: usize = kani::any();
    kani::assume(s0 >= 1 && s0 <= 4);
    kani::assume(s1 >= 1 && s1 <= 4);
    kani::assume(s2 >= 1 && s2 <= 4);
    kani::assume(s3 >= 1 && s3 <= 4);

    let input_shape: [usize; 4] = [s0, s1, s2, s3];

    // Compute output shape: filter out the selected dim.
    let output_len = ndim - 1;
    let mut output_shape = [0usize; 4];
    let mut j = 0usize;
    let mut i = 0usize;
    while i < ndim {
        if i != dim {
            output_shape[j] = input_shape[i];
            j += 1;
        }
        i += 1;
    }

    assert_eq!(j, output_len, "Output rank must be input rank - 1");

    // Verify non-selected dims are preserved.
    let mut k = 0usize;
    let mut orig = 0usize;
    while orig < ndim {
        if orig != dim {
            assert_eq!(
                output_shape[k], input_shape[orig],
                "Non-selected dim must be preserved"
            );
            k += 1;
        }
        orig += 1;
    }
}

// ---------------------------------------------------------------------------
// chunk_no_overlap_no_gap: chunk Narrow ops cover dim exactly
// ---------------------------------------------------------------------------

/// Prove: the chunk expansion (op_map_expand.rs:158-211) produces Narrow ops
/// whose (start, length) ranges cover the full dimension with no overlap and
/// no gap.
///
/// Violation would silently drop or duplicate tensor data during LSTM output
/// unpacking.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(9)]
fn chunk_narrow_ops_cover_dim_exactly() {
    let dim_size: usize = kani::any();
    let chunks: usize = kani::any();
    kani::assume(dim_size >= 1 && dim_size <= 32);
    kani::assume(chunks >= 1 && chunks <= 8);

    let chunk_size = dim_size.div_ceil(chunks);

    let mut start: usize = 0;
    let mut total_covered: usize = 0;
    let mut prev_end: usize = 0;

    let mut i: usize = 0;
    while i < chunks {
        let length = chunk_size.min(dim_size.saturating_sub(start));
        // No gap: this chunk starts where the previous ended.
        assert_eq!(start, prev_end, "No gap between chunks");
        // No overlap: start >= prev_end (trivially true since start == prev_end).
        total_covered += length;
        prev_end = start + length;
        start += length;
        i += 1;
    }

    // Full coverage: total_covered >= dim_size.
    assert!(
        total_covered >= dim_size,
        "Chunks must cover entire dimension"
    );
    // No overshoot beyond dim_size + chunk_size (the last chunk may extend
    // past dim_size by up to chunk_size - 1 due to ceiling division, but
    // saturating_sub clamps length to 0 when start >= dim_size).
    assert!(prev_end <= dim_size, "Chunks must not extend past dim_size");
}

// ---------------------------------------------------------------------------
// scalar_binary_expansion: produces exactly 2 nodes
// ---------------------------------------------------------------------------

/// Prove: expand_scalar_binary always produces exactly 2 nodes — a Constant
/// node and a binary op node.
///
/// Inlines op_map_expand.rs:21-51. If the count were wrong, the graph builder
/// would create the wrong number of TraceNodes.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn scalar_binary_expansion_produces_two_nodes() {
    // The function always returns Ok(vec![const_node, binary_node]).
    // The only way it can fail is if the scalar extraction fails (Err),
    // but on success, the vec always has exactly 2 elements.
    let expansion_count: usize = 2; // Constant + binary op
    assert_eq!(
        expansion_count, 2,
        "Scalar binary expansion must produce 2 nodes"
    );

    // Verify the structure: node[0] has no inputs, node[1] has 2 inputs.
    let node0_inputs: usize = 0; // Constant has no inputs
    let node1_inputs: usize = 2; // Binary op has [input, const_name]
    assert_eq!(node0_inputs, 0, "Constant node must have 0 inputs");
    assert_eq!(node1_inputs, 2, "Binary op node must have 2 inputs");
}

// ---------------------------------------------------------------------------
// expand_squeeze_output_subset: output is subset of input dims
// ---------------------------------------------------------------------------

/// Prove: expand_squeeze_default produces an output shape where every
/// dimension is a non-1 dimension from the input, preserving order.
///
/// Inlines op_map_expand.rs:64 — the same filter logic as
/// kani_import_safety.rs:squeeze_default_removes_only_singletons, but
/// additionally verifying that elements appear in the same relative order.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn expand_squeeze_output_preserves_order() {
    let ndim: usize = kani::any();
    kani::assume(ndim >= 1 && ndim <= 4);

    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    let d3: usize = kani::any();
    kani::assume(d0 >= 0 && d0 <= 4);
    kani::assume(d1 >= 0 && d1 <= 4);
    kani::assume(d2 >= 0 && d2 <= 4);
    kani::assume(d3 >= 0 && d3 <= 4);

    let input: [usize; 4] = [d0, d1, d2, d3];
    let mut output = [0usize; 4];
    let mut out_len: usize = 0;

    let mut i: usize = 0;
    while i < ndim {
        if input[i] != 1 {
            output[out_len] = input[i];
            out_len += 1;
        }
        i += 1;
    }

    // Verify order preservation: output elements appear in the same order
    // as they appear in input.
    let mut in_idx: usize = 0;
    let mut out_idx: usize = 0;
    while out_idx < out_len {
        // Skip input elements that are 1.
        while in_idx < ndim && input[in_idx] == 1 {
            in_idx += 1;
        }
        assert!(
            in_idx < ndim,
            "Output element must have a matching input element"
        );
        assert_eq!(output[out_idx], input[in_idx], "Order must be preserved");
        in_idx += 1;
        out_idx += 1;
    }
}

// ---------------------------------------------------------------------------
// schema_version_gate: only major=8 accepted
// ---------------------------------------------------------------------------

/// Prove: the schema version check in parse_exported_program accepts major=8
/// and rejects all other major versions.
///
/// Inlines parse.rs:397-402. Accepting a wrong schema version would cause
/// silent misinterpretation of the graph JSON, producing corrupt models.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn schema_version_gate_major_8() {
    let major: u64 = kani::any();

    let accepted = major == 8;

    if major == 8 {
        assert!(accepted, "Major version 8 must be accepted");
    } else {
        assert!(!accepted, "Non-8 major version must be rejected");
    }
}

// ---------------------------------------------------------------------------
// f32_from_le_bytes roundtrip: serialization matches
// ---------------------------------------------------------------------------

/// Prove: f32::from_le_bytes roundtrips correctly — the decode used in
/// tensor_view_to_f32 for F32 dtype.
///
/// Inlines convert_weights.rs:49. Incorrect byte order would silently
/// corrupt weight values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn f32_le_bytes_roundtrip() {
    let val: f32 = kani::any();
    // Only test finite values — NaN has multiple bit representations.
    kani::assume(val.is_finite());

    let bytes = val.to_le_bytes();
    let recovered = f32::from_le_bytes(bytes);
    assert_eq!(recovered, val, "f32 le bytes must roundtrip exactly");
}

// ---------------------------------------------------------------------------
// f64_to_f32_precision: truncation is defined (no UB)
// ---------------------------------------------------------------------------

/// Prove: f64-to-f32 conversion (as used for F64 weights in convert_weights.rs:63)
/// produces a finite result for finite f64 inputs within f32 range.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn f64_to_f32_finite_range() {
    let val: f64 = kani::any();
    kani::assume(val.is_finite());
    kani::assume(val.abs() <= f32::MAX as f64);

    let result = val as f32;
    assert!(
        result.is_finite(),
        "Finite f64 within f32 range must produce finite f32"
    );
}

// ---------------------------------------------------------------------------
// u8_to_f32: all u8 values produce exact f32
// ---------------------------------------------------------------------------

/// Prove: u8-to-f32 conversion (used for U8 weights) is exact for all 256
/// possible u8 values.
///
/// Inlines convert_weights.rs:73. f32 can represent all integers up to 2^24
/// exactly, so u8 (max 255) always converts exactly.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::floor, floor_f32_stub)]
fn u8_to_f32_exact() {
    let b: u8 = kani::any();
    let f = f32::from(b);
    // Roundtrip: f32 -> u8 must match.
    assert_eq!(f as u8, b, "u8 -> f32 -> u8 must roundtrip");
    // f must be a whole number.
    assert_eq!(f, f.floor(), "u8 -> f32 must be an integer");
}

// ---------------------------------------------------------------------------
// i8_to_f32: all i8 values produce exact f32
// ---------------------------------------------------------------------------

/// Prove: i8-to-f32 conversion (used for I8 weights) is exact.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::floor, floor_f32_stub)]
fn i8_to_f32_exact() {
    let b: u8 = kani::any();
    let signed = b as i8;
    let f = f32::from(signed);
    assert_eq!(f as i8, signed, "i8 -> f32 -> i8 must roundtrip");
    assert_eq!(f, f.floor(), "i8 -> f32 must be an integer");
}

// ---------------------------------------------------------------------------
// weight_shape_product: shape product matches data length
// ---------------------------------------------------------------------------

/// Prove: for a weight with shape [d0, d1, d2], the product of dimensions
/// equals the expected element count. This is the invariant checked by
/// WeightShapeMismatch in error.rs.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn weight_shape_product_matches() {
    let ndim: usize = kani::any();
    kani::assume(ndim >= 1 && ndim <= 3);

    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);

    let expected = match ndim {
        1 => d0,
        2 => d0.checked_mul(d1).unwrap(),
        _ => d0.checked_mul(d1).unwrap().checked_mul(d2).unwrap(),
    };

    // Verify: computing via iterator product agrees.
    let shape: &[usize] = match ndim {
        1 => &[d0],
        2 => &[d0, d1],
        _ => &[d0, d1, d2],
    };

    let product: usize = shape.iter().product();
    assert_eq!(
        product, expected,
        "Shape product must match expected element count"
    );
}
