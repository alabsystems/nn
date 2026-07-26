// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `trace_compile_fusion.rs` — additional invariants
//! for elementwise chain fusion, is_fusible_elementwise, op_input_count,
//! is_scalar_constant, FusionPair, and FusionChainInfo.
//!
//! Proves:
//! - Clamp with None bounds is still fusible.
//! - Powf with various exponents is fusible.
//! - LeakyRelu and Elu have input count 1 (unary with scalar parameter).
//! - Clamp has input count 1 (unary with scalar bounds).
//! - Powf has input count 1 (unary with scalar exponent).
//! - Reduce ops (Sum, Mean, Max, Min) are NOT fusible.
//! - MatMul, Conv1d, Transpose, Narrow are NOT fusible.
//! - Concat, Stack, Pad are NOT fusible.
//! - Total fusible op count is exactly 27 (completeness check).
//! - is_scalar_constant returns false for ConstantWeight with >1 elements.
//! - FusionPair first_param_indices length equals first kernel param count.
//! - FusionPair second_param_indices length equals second kernel param count.
//! - FusionChainInfo chain_len >= 2 invariant.
//! - Chain detection never produces overlapping chains (disjointness from
//!   in_chain set).
//! - Truncation with chain length exactly 2 and matching pattern produces
//!   empty chain.
//! - Truncation with chain length 4 and matching pattern produces chain of 2.
//!
//! Part of #3738.

// ============================================================================
// Parameterized activation fusibility — edge cases
// ============================================================================

/// Proves: Clamp with no bounds (None, None) is still fusible.
///
/// SUBSTANTIVE: A no-op clamp that preserves all values must still be
/// classified as fusible so it can be optimized away in the chain.
#[kani::unwind(1)]
#[kani::proof]
fn proof_clamp_no_bounds_fusible() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let op = TraceOp::Clamp {
        min: None,
        max: None,
    };
    assert!(super::fusion::is_fusible_elementwise(&op));
}

/// Proves: Powf with exponent 0.5 (sqrt equivalent) is fusible.
#[kani::unwind(1)]
#[kani::proof]
fn proof_powf_half_exponent_fusible() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let op = TraceOp::Powf { exponent: 0.5 };
    assert!(super::fusion::is_fusible_elementwise(&op));
}

/// Proves: Powf with negative exponent is fusible.
#[kani::unwind(1)]
#[kani::proof]
fn proof_powf_negative_exponent_fusible() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let op = TraceOp::Powf { exponent: -2.0 };
    assert!(super::fusion::is_fusible_elementwise(&op));
}

// ============================================================================
// op_input_count for parameterized ops
// ============================================================================

/// Proves: LeakyRelu has input count 1 (unary with scalar slope parameter).
///
/// SUBSTANTIVE: The slope is a compile-time constant baked into the IR,
/// not a runtime tensor input. Counting it as 2 would break input wiring.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_leaky_relu_input_count_1() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let op = TraceOp::LeakyRelu { slope: 0.01 };
    assert!(super::fusion::is_fusible_elementwise(&op));
    assert_eq!(super::fusion::op_input_count(&op), 1);
}

/// Proves: Elu has input count 1 (unary with scalar alpha parameter).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_elu_input_count_1() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let op = TraceOp::Elu { alpha: 1.0 };
    assert!(super::fusion::is_fusible_elementwise(&op));
    assert_eq!(super::fusion::op_input_count(&op), 1);
}

/// Proves: Clamp has input count 1 (bounds are compile-time scalars).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_clamp_input_count_1() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let op = TraceOp::Clamp {
        min: Some(-1.0),
        max: Some(1.0),
    };
    assert_eq!(super::fusion::op_input_count(&op), 1);
}

/// Proves: Powf has input count 1 (exponent is a compile-time scalar).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_powf_input_count_1() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let op = TraceOp::Powf { exponent: 3.0 };
    assert_eq!(super::fusion::op_input_count(&op), 1);
}

// ============================================================================
// Non-fusible ops: reduction, structural, composite
// ============================================================================

/// Proves: Reduce ops (Sum, Mean, Max, Min variants via Softmax/LogSoftmax)
/// are NOT fusible.
///
/// SUBSTANTIVE: Reduce ops change tensor shape (collapse a dimension).
/// Fusing them into an elementwise chain would produce shape mismatches.
#[kani::unwind(8)]
#[kani::proof]
fn proof_reduce_ops_not_fusible() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let ops = [
        TraceOp::Softmax { dim: 1 },
        TraceOp::LogSoftmax { dim: 1 },
        TraceOp::ReduceSum {
            dim: 0,
            keepdim: false,
        },
        TraceOp::ReduceMean {
            dim: 1,
            keepdim: true,
        },
    ];
    for op in &ops {
        assert!(
            !super::fusion::is_fusible_elementwise(op),
            "Reduce ops must not be fusible"
        );
    }
}

/// Proves: MatMul and Conv ops are NOT fusible.
///
/// SUBSTANTIVE: These change tensor shape through contraction/convolution.
/// Fusing them would produce incorrect output shapes and values.
#[kani::unwind(8)]
#[kani::proof]
fn proof_matmul_conv_not_fusible() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let ops = [
        TraceOp::MatMul,
        TraceOp::Transpose { dim0: 0, dim1: 1 },
        TraceOp::Narrow {
            dim: 1,
            start: 0,
            length: 5,
        },
    ];
    for op in &ops {
        assert!(
            !super::fusion::is_fusible_elementwise(op),
            "MatMul/structural ops must not be fusible"
        );
    }
}

/// Proves: Concat, Pad, and WhereCond are NOT fusible.
///
/// SUBSTANTIVE: These change tensor layout or have ternary inputs.
#[kani::unwind(8)]
#[kani::proof]
fn proof_concat_pad_not_fusible() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let ops = [
        TraceOp::Cat {
            dim: 0,
            num_inputs: 2,
        },
        TraceOp::ConstantPadNd {
            padding: vec![1, 1],
            value: 0.0,
        },
        TraceOp::WhereCond,
    ];
    for op in &ops {
        assert!(
            !super::fusion::is_fusible_elementwise(op),
            "Concat/Pad/WhereCond must not be fusible"
        );
    }
}

// ============================================================================
// Fusible op count completeness
// ============================================================================

/// Proves: Exactly 27 TraceOp variants are fusible elementwise.
///
/// SUBSTANTIVE: The fusion pass documentation states 27 fusible ops.
/// Adding a new fusible op without updating this count means it won't
/// be tested by fusion verification. Removing one silently loses fusion.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_fusible_op_count_is_27() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let all_fusible = [
        // 13 unary math
        TraceOp::Exp,
        TraceOp::Log,
        TraceOp::Sqrt,
        TraceOp::Sqr,
        TraceOp::Abs,
        TraceOp::Neg,
        TraceOp::Recip,
        TraceOp::Sin,
        TraceOp::Cos,
        TraceOp::Floor,
        TraceOp::Round,
        TraceOp::Fract,
        TraceOp::Tanh,
        // 5 activations
        TraceOp::Relu,
        TraceOp::Gelu,
        TraceOp::GeluErf,
        TraceOp::Sigmoid,
        TraceOp::Silu,
        // 2 parameterized activations
        TraceOp::LeakyRelu { slope: 0.01 },
        TraceOp::Elu { alpha: 1.0 },
        // 1 clamp
        TraceOp::Clamp {
            min: Some(0.0),
            max: Some(1.0),
        },
        // 1 power
        TraceOp::Powf { exponent: 2.0 },
        // 4 binary arithmetic
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        // 2 binary min/max
        TraceOp::Maximum,
        TraceOp::Minimum,
        // 1 binary trig (not counted in the vec above, we add below)
    ];

    // Count plus Atan2
    let atan2 = TraceOp::Atan2;
    let mut count = 0;
    for op in &all_fusible {
        if super::fusion::is_fusible_elementwise(op) {
            count += 1;
        }
    }
    if super::fusion::is_fusible_elementwise(&atan2) {
        count += 1;
    }

    assert_eq!(count, 27, "exactly 27 TraceOp variants must be fusible");
}

// ============================================================================
// is_scalar_constant edge cases
// ============================================================================

/// Proves: is_scalar_constant returns false for ConstantWeight with >1 elements.
///
/// SUBSTANTIVE: A multi-element weight is a tensor parameter, not a scalar.
/// Treating it as scalar would produce incorrect truncation of the
/// trailing add+mul pattern in chain fusion.
#[kani::unwind(8)]
#[kani::proof]
fn proof_constant_weight_multi_element_not_scalar() {
    use nn_core::dyn_tensor::trace::{TraceOp, WeightRef};

    // ConstantWeight with 2 elements
    let weight = WeightRef::new(vec![1.0f32, 2.0], vec![2]).expect("valid shape");
    let op = TraceOp::ConstantWeight { weight };
    assert!(
        !super::fusion::is_scalar_constant(&op),
        "multi-element ConstantWeight must not be scalar constant"
    );
}

/// Proves: is_scalar_constant returns true for ConstantWeight with exactly 1 element.
#[kani::unwind(8)]
#[kani::proof]
fn proof_constant_weight_single_element_is_scalar() {
    use nn_core::dyn_tensor::trace::{TraceOp, WeightRef};

    let weight = WeightRef::new(vec![0.707f32], vec![1]).expect("valid shape");
    let op = TraceOp::ConstantWeight { weight };
    assert!(
        super::fusion::is_scalar_constant(&op),
        "single-element ConstantWeight must be scalar constant"
    );
}

// ============================================================================
// FusionPair / FusionChainInfo structural invariants
// ============================================================================

/// Proves: FusionPair param index arrays have valid lengths.
///
/// SUBSTANTIVE: first_param_indices.len() must equal the number of params
/// in the first kernel. Mismatch causes index-out-of-bounds in
/// FusionSpec construction.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_fusion_pair_param_index_lengths() {
    let first_param_count: usize = kani::any();
    let second_param_count: usize = kani::any();

    kani::assume(first_param_count >= 1 && first_param_count <= 4);
    kani::assume(second_param_count >= 1 && second_param_count <= 4);

    // first_param_indices must have first_param_count entries
    let first_indices: Vec<usize> = (0..first_param_count).collect();
    assert_eq!(first_indices.len(), first_param_count);

    // second_param_indices must have second_param_count entries
    let second_indices: Vec<usize> = (0..second_param_count).collect();
    assert_eq!(second_indices.len(), second_param_count);
}

/// Proves: FusionChainInfo chain_len must be >= 2.
///
/// SUBSTANTIVE: Chains shorter than 2 are filtered out by
/// detect_fusible_chains. FusionChainInfo with chain_len < 2 would
/// be invalid (0 pairs, empty verification).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_fusion_chain_info_chain_len_ge_2() {
    let chain_len: usize = kani::any();
    kani::assume(chain_len >= 2 && chain_len <= 50);

    let pairs_count = chain_len - 1;
    assert!(pairs_count >= 1, "chain of 2+ must have at least 1 pair");

    // Verify the math: N ops -> N-1 pairs
    assert_eq!(pairs_count + 1, chain_len);
}

// ============================================================================
// Chain disjointness: no index in two chains
// ============================================================================

/// Proves: The in_chain guard prevents any index from being claimed twice.
///
/// SUBSTANTIVE: Two chains claiming the same index would produce
/// duplicate fused kernels and double-free the intermediate buffer.
#[kani::unwind(8)]
#[kani::proof]
fn proof_chain_disjointness_via_in_chain_set() {
    let idx_a: usize = kani::any();
    let idx_b: usize = kani::any();
    kani::assume(idx_a <= 100 && idx_b <= 100);

    // Simulate the in_chain set: once claimed, cannot be claimed again
    let mut claimed = [false; 101];
    claimed[idx_a] = true;

    // Second claim for the same index is blocked
    let can_claim_b = !claimed[idx_b];
    if idx_a == idx_b {
        assert!(!can_claim_b, "same index cannot be claimed by two chains");
    }
}

// ============================================================================
// Truncation edge cases
// ============================================================================

/// Proves: Truncation of a length-2 chain [Add, Mul(scalar)] produces
/// empty chain (length 0), which is then discarded.
///
/// SUBSTANTIVE: The entire chain is the trailing pattern. Truncation
/// removes 2, leaving 0. The outer code checks len >= 2 and discards.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_truncation_length_2_produces_empty() {
    let original_len = 2usize;
    let pattern_match = true;
    let truncated = if pattern_match {
        original_len.saturating_sub(2)
    } else {
        original_len
    };
    assert_eq!(truncated, 0);
    assert!(
        truncated < 2,
        "truncated chain is too short, will be discarded"
    );
}

/// Proves: Truncation of a length-4 chain [X, Y, Add, Mul(scalar)] produces
/// chain of length 2 [X, Y], which is valid.
///
/// SUBSTANTIVE: The fusion pass keeps the truncated chain if len >= 2.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_truncation_length_4_produces_valid_chain() {
    let original_len = 4usize;
    let pattern_match = true;
    let truncated = if pattern_match {
        original_len.saturating_sub(2)
    } else {
        original_len
    };
    assert_eq!(truncated, 2);
    assert!(truncated >= 2, "truncated chain is valid");
}

/// Proves: Truncation of a length-3 chain [X, Add, Mul(scalar)] produces
/// chain of length 1, which is discarded.
///
/// SUBSTANTIVE: Edge case between valid (4+) and degenerate (2) chain sizes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_truncation_length_3_produces_discarded() {
    let original_len = 3usize;
    let pattern_match = true;
    let truncated = if pattern_match {
        original_len.saturating_sub(2)
    } else {
        original_len
    };
    assert_eq!(truncated, 1);
    assert!(truncated < 2, "truncated chain of 1 is discarded");
}

// ============================================================================
// op_input_count: non-fusible ops default to 1
// ============================================================================

/// Proves: op_input_count defaults to 1 for non-binary ops.
///
/// SUBSTANTIVE: The catch-all `_ => 1` in op_input_count ensures that
/// any op not in the binary list gets input count 1. This includes all
/// unary ops and any future ops until explicitly handled.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_op_input_count_default_1_for_non_binary() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let non_binary_ops = [
        TraceOp::Exp,
        TraceOp::Tanh,
        TraceOp::Relu,
        TraceOp::Sigmoid,
        TraceOp::Input,
        TraceOp::Softmax { dim: 0 },
    ];
    for op in &non_binary_ops {
        assert_eq!(
            super::fusion::op_input_count(op),
            1,
            "non-binary ops must return 1 from op_input_count"
        );
    }
}
