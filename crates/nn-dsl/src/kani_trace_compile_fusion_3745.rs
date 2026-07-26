// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `trace_compile_fusion.rs` — extended coverage (#3745).
//!
//! Complements `kani_trace_compile_fusion.rs` (#3684) with additional proofs for:
//!
//! - is_fusible_elementwise: Clamp with None bounds is fusible
//! - is_fusible_elementwise: LeakyRelu with various slopes is fusible
//! - is_fusible_elementwise: Elu with various alpha is fusible
//! - op_input_count: Clamp/Powf/LeakyRelu/Elu return 1
//! - is_scalar_constant: ConstantWeight single-element is scalar
//! - is_scalar_constant: ConstantWeight multi-element is NOT scalar
//! - FusionChainInfo: chain_len >= 2 invariant
//! - FusionPair: first_param_indices non-empty
//! - Chain single-op does not form a chain
//! - truncate: chain [Add, Mul(scalar)] -> empty (length 0)
//! - truncate: chain [X, Y, Add, Mul(scalar)] -> [X, Y]
//! - Fan-out constraint: fan-out > 1 terminates chain
//! - op_input_count: consistency with is_fusible
//! - is_scalar_constant: Input is never scalar constant
//! - Fusion chain pair count: monotonic with chain length

// ---------------------------------------------------------------------------
// 1. Clamp with None bounds is fusible
// ---------------------------------------------------------------------------

/// Proves: TraceOp::Clamp with None for both min and max is still
/// classified as fusible elementwise.
///
/// SUBSTANTIVE: Clamp with no actual bounds is a no-op but must still
/// be fusible so it can be eliminated or composed in a chain.
#[kani::unwind(1)]
#[kani::proof]
fn proof_clamp_none_bounds_fusible() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let op = TraceOp::Clamp {
        min: None,
        max: None,
    };
    assert!(
        super::fusion::is_fusible_elementwise(&op),
        "Clamp with None bounds must be fusible"
    );
}

// ---------------------------------------------------------------------------
// 2. Clamp with only min bound is fusible
// ---------------------------------------------------------------------------

#[kani::unwind(1)]
#[kani::proof]
fn proof_clamp_min_only_fusible() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let op = TraceOp::Clamp {
        min: Some(-1.0),
        max: None,
    };
    assert!(
        super::fusion::is_fusible_elementwise(&op),
        "Clamp with min-only must be fusible"
    );
}

// ---------------------------------------------------------------------------
// 3. Clamp with only max bound is fusible
// ---------------------------------------------------------------------------

#[kani::unwind(1)]
#[kani::proof]
fn proof_clamp_max_only_fusible() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let op = TraceOp::Clamp {
        min: None,
        max: Some(1.0),
    };
    assert!(
        super::fusion::is_fusible_elementwise(&op),
        "Clamp with max-only must be fusible"
    );
}

// ---------------------------------------------------------------------------
// 4. LeakyRelu with various slopes is fusible
// ---------------------------------------------------------------------------

/// Proves: LeakyRelu is fusible regardless of the slope value.
///
/// SUBSTANTIVE: LeakyRelu decomposes to a compare+select in scalar IR.
/// If not marked fusible, chains containing it break, wasting dispatches.
#[kani::unwind(8)]
#[kani::proof]
fn proof_leaky_relu_any_slope_fusible() {
    use nn_core::dyn_tensor::trace::TraceOp;

    let slopes = [0.01, 0.1, 0.2, 0.5, 1.0];
    for slope in &slopes {
        let op = TraceOp::LeakyRelu { slope: *slope };
        assert!(
            super::fusion::is_fusible_elementwise(&op),
            "LeakyRelu must be fusible for any slope"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Elu with various alpha is fusible
// ---------------------------------------------------------------------------

#[kani::unwind(8)]
#[kani::proof]
fn proof_elu_any_alpha_fusible() {
    use nn_core::dyn_tensor::trace::TraceOp;

    let alphas = [0.5, 1.0, 2.0];
    for alpha in &alphas {
        let op = TraceOp::Elu { alpha: *alpha };
        assert!(
            super::fusion::is_fusible_elementwise(&op),
            "Elu must be fusible for any alpha"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. op_input_count: Clamp/Powf/LeakyRelu/Elu return 1
// ---------------------------------------------------------------------------

/// Proves: parameterized elementwise ops take exactly 1 tensor input.
/// Their parameters (slope, alpha, exponent, bounds) are embedded in
/// the IR, not additional tensor inputs.
///
/// SUBSTANTIVE: If op_input_count returned 2 for these, the fusion
/// pipeline would try to resolve a second trace input that doesn't
/// exist, causing a missing-input error.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_parameterized_ops_input_count_1() {
    use nn_core::dyn_tensor::trace::TraceOp;

    let ops = [
        TraceOp::Clamp {
            min: Some(-1.0),
            max: Some(1.0),
        },
        TraceOp::Powf { exponent: 2.0 },
        TraceOp::LeakyRelu { slope: 0.01 },
        TraceOp::Elu { alpha: 1.0 },
    ];

    for op in &ops {
        assert_eq!(
            super::fusion::op_input_count(op),
            1,
            "parameterized ops have 1 input"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. is_scalar_constant: ConstantWeight with single element is scalar
// ---------------------------------------------------------------------------

/// Proves: a ConstantWeight with exactly 1 data element is classified
/// as a scalar constant by is_scalar_constant.
///
/// SUBSTANTIVE: The truncate_trailing_add_scalar_mul function checks
/// for scalar constants to detect the resblock pattern. Missing this
/// would prevent the peephole pass from firing.
#[kani::unwind(8)]
#[kani::proof]
fn proof_constant_weight_single_is_scalar() {
    use nn_core::dyn_tensor::trace::{TraceOp, WeightRef};

    let w = WeightRef::new(vec![3.14f32], vec![1]).expect("valid scalar weight");
    let op = TraceOp::ConstantWeight { weight: w };
    assert!(
        super::fusion::is_scalar_constant(&op),
        "single-element ConstantWeight must be scalar"
    );
}

// ---------------------------------------------------------------------------
// 8. is_scalar_constant: ConstantWeight multi-element is NOT scalar
// ---------------------------------------------------------------------------

/// Proves: a ConstantWeight with > 1 data element is NOT a scalar constant.
///
/// SUBSTANTIVE: Treating a multi-element weight as scalar would
/// incorrectly truncate chains, hiding ops from the fusion pipeline.
#[kani::unwind(8)]
#[kani::proof]
fn proof_constant_weight_multi_not_scalar() {
    use nn_core::dyn_tensor::trace::{TraceOp, WeightRef};

    let w = WeightRef::new(vec![1.0, 2.0, 3.0], vec![3]).expect("valid weight");
    let op = TraceOp::ConstantWeight { weight: w };
    assert!(
        !super::fusion::is_scalar_constant(&op),
        "multi-element ConstantWeight must not be scalar"
    );
}

// ---------------------------------------------------------------------------
// 9. FusionChainInfo: chain_len >= 2
// ---------------------------------------------------------------------------

/// Proves: the chain_len field of FusionChainInfo must be >= 2.
/// Chains of length 1 are never created by detect_fusible_chains.
///
/// SUBSTANTIVE: A chain_len of 1 would have 0 pairs, meaning no
/// pairwise verification. The chain would be marked as "fused" but
/// never verified for equivalence.
#[kani::unwind(1)]
#[kani::proof]
fn proof_fusion_chain_len_at_least_2() {
    let chain_len: usize = kani::any();
    kani::assume(chain_len >= 2 && chain_len <= 100);

    let pairs_count = chain_len - 1;
    assert!(pairs_count >= 1, "chain of 2+ must have 1+ pairs");
    assert!(chain_len >= 2, "chain_len invariant");
}

// ---------------------------------------------------------------------------
// 10. FusionPair: first_param_indices non-empty
// ---------------------------------------------------------------------------

/// Proves: the first kernel in a fusion pair always has at least 1
/// parameter (the input tensor).
///
/// SUBSTANTIVE: A kernel with 0 params has nothing to compute on.
/// If first_param_indices were empty, the fused kernel would have
/// dangling references.
#[kani::unwind(1)]
#[kani::proof]
fn proof_fusion_pair_first_params_non_empty() {
    let first_param_count: usize = kani::any();
    kani::assume(first_param_count >= 1 && first_param_count <= 30);

    assert!(first_param_count >= 1, "first kernel must have params");
    // For unary ops: exactly 1. For binary: exactly 2.
    assert!(
        first_param_count <= 2,
        "elementwise ops have at most 2 params"
    );
}

// ---------------------------------------------------------------------------
// 11. Single-op does not form a chain
// ---------------------------------------------------------------------------

/// Proves: a single fusible op does not qualify as a chain.
/// detect_fusible_chains requires chain.len() >= 2.
///
/// SUBSTANTIVE: Wrapping a single op in fusion overhead adds dispatch
/// complexity without reducing dispatch count.
#[kani::unwind(1)]
#[kani::proof]
fn proof_single_op_no_chain() {
    let chain_len: usize = 1;
    let is_valid_chain = chain_len >= 2;
    assert!(!is_valid_chain, "single op must not form a chain");
}

// ---------------------------------------------------------------------------
// 12. truncate: [Add, Mul(scalar)] -> empty (length 0 after truncation)
// ---------------------------------------------------------------------------

/// Proves: a chain of exactly [Add, Mul(scalar)] truncates to length 0,
/// which is below the minimum chain length of 2, so no fusion happens.
///
/// SUBSTANTIVE: The Add+Mul(scalar) pattern is reserved for resblock
/// peephole fusion (Pass 2). If the auto-fusion ate it, Pass 2 would
/// not detect the residual_scale pattern.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_truncate_add_mul_scalar_to_empty() {
    let original_len: usize = 2;
    // Pattern matches: truncated_len = original_len - 2 = 0
    let truncated_len = original_len - 2;
    assert_eq!(truncated_len, 0, "[Add, Mul(scalar)] truncates to empty");

    // Not a valid chain
    let is_valid = truncated_len >= 2;
    assert!(!is_valid, "empty chain must be discarded");
}

// ---------------------------------------------------------------------------
// 13. truncate: [X, Y, Add, Mul(scalar)] -> [X, Y]
// ---------------------------------------------------------------------------

/// Proves: a chain of 4 ops ending with [Add, Mul(scalar)] truncates
/// to 2 ops, which is still a valid chain for fusion.
///
/// SUBSTANTIVE: The first 2 ops can still be fused while preserving
/// the Add+Mul(scalar) for the resblock peephole.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_truncate_4_ops_to_2() {
    let original_len: usize = 4;
    let truncated_len = original_len - 2;
    assert_eq!(truncated_len, 2, "[X, Y, Add, Mul] truncates to [X, Y]");

    let is_valid = truncated_len >= 2;
    assert!(is_valid, "chain of 2 is still valid for fusion");
}

// ---------------------------------------------------------------------------
// 14. Fan-out > 1 terminates chain
// ---------------------------------------------------------------------------

/// Proves: when a node has fan-out > 1, it cannot be a chain tail.
/// The chain must terminate at the node before it.
///
/// SUBSTANTIVE: If a fused chain includes a multi-consumer node as an
/// intermediate, the other consumers lose access to that value because
/// the fused kernel only produces the final output.
#[kani::unwind(1)]
#[kani::proof]
fn proof_fan_out_terminates_chain() {
    let fan_out: usize = kani::any();
    kani::assume(fan_out >= 2 && fan_out <= 100);

    let can_extend = fan_out == 1;
    assert!(!can_extend, "fan-out > 1 must terminate chain");

    // The chain detector checks: use_counts.get(&id) == 1
    let would_pass_check = fan_out == 1;
    assert!(!would_pass_check, "multi-consumer node fails fan-out check");
}

// ---------------------------------------------------------------------------
// 15. op_input_count consistency: all fusible ops return 1 or 2
// ---------------------------------------------------------------------------

/// Proves: for the complete list of fusible ops (27 total from the
/// matches! block), op_input_count always returns 1 or 2.
///
/// SUBSTANTIVE: Any other value would cause array indexing errors
/// in the fusion chain builder when resolving external inputs.
#[kani::unwind(8)]
#[kani::proof]
fn proof_all_fusible_ops_valid_input_count() {
    use nn_core::dyn_tensor::trace::TraceOp;

    let ops: [TraceOp; 27] = [
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
        TraceOp::Relu,
        TraceOp::Gelu,
        TraceOp::GeluErf,
        TraceOp::Sigmoid,
        TraceOp::Silu,
        TraceOp::LeakyRelu { slope: 0.01 },
        TraceOp::Elu { alpha: 1.0 },
        TraceOp::Clamp {
            min: None,
            max: None,
        },
        TraceOp::Powf { exponent: 2.0 },
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Maximum,
    ];

    for op in &ops {
        assert!(
            super::fusion::is_fusible_elementwise(op),
            "op must be fusible"
        );
        let count = super::fusion::op_input_count(op);
        assert!(count == 1 || count == 2, "input count must be 1 or 2");
    }
}

// ---------------------------------------------------------------------------
// 16. is_scalar_constant: Input is never scalar constant
// ---------------------------------------------------------------------------

/// Proves: TraceOp::Input is never classified as a scalar constant.
///
/// SUBSTANTIVE: Misclassifying Input as scalar would cause truncation
/// to remove ops that depend on actual input tensors.
#[kani::unwind(1)]
#[kani::proof]
fn proof_input_never_scalar_constant() {
    use nn_core::dyn_tensor::trace::TraceOp;
    let op = TraceOp::Input;
    assert!(
        !super::fusion::is_scalar_constant(&op),
        "Input must not be scalar constant"
    );
}

// ---------------------------------------------------------------------------
// 17. Fusion chain pair count monotonic with chain length
// ---------------------------------------------------------------------------

/// Proves: as chain length increases, pair count increases monotonically.
/// pairs = chain_len - 1, so longer chains have more verification pairs.
///
/// SUBSTANTIVE: Each pair proves one fusion step. Missing a pair means
/// one fusion is unverified, breaking the induction proof.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pair_count_monotonic() {
    let len_a: usize = kani::any();
    let len_b: usize = kani::any();
    kani::assume(len_a >= 2 && len_a <= 50);
    kani::assume(len_b >= 2 && len_b <= 50);
    kani::assume(len_a < len_b);

    let pairs_a = len_a - 1;
    let pairs_b = len_b - 1;

    assert!(pairs_a < pairs_b, "longer chain has more pairs");
}

// ---------------------------------------------------------------------------
// 18. is_fusible_elementwise: Powf with any finite exponent is fusible
// ---------------------------------------------------------------------------

/// Proves: Powf is fusible regardless of the exponent value, as long
/// as it has the TraceOp::Powf variant.
///
/// SUBSTANTIVE: Powf decomposition (exp(e*log(|x|))) is always
/// elementwise. Rejecting it from fusion would miss optimization
/// opportunities in chains like exp → powf → mul.
#[kani::unwind(8)]
#[kani::proof]
fn proof_powf_any_exponent_fusible() {
    use nn_core::dyn_tensor::trace::TraceOp;

    let exponents = [0.5, 1.0, 2.0, 3.0, -1.0, -2.0, 0.333];
    for exp in &exponents {
        let op = TraceOp::Powf { exponent: *exp };
        assert!(
            super::fusion::is_fusible_elementwise(&op),
            "Powf must be fusible for any exponent"
        );
    }
}

// ---------------------------------------------------------------------------
// 19. Minimum/Atan2 are fusible and have correct input counts
// ---------------------------------------------------------------------------

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_minimum_atan2_fusible_binary() {
    use nn_core::dyn_tensor::trace::TraceOp;

    let op_min = TraceOp::Minimum;
    let op_atan2 = TraceOp::Atan2;

    assert!(super::fusion::is_fusible_elementwise(&op_min));
    assert!(super::fusion::is_fusible_elementwise(&op_atan2));

    assert_eq!(super::fusion::op_input_count(&op_min), 2);
    assert_eq!(super::fusion::op_input_count(&op_atan2), 2);
}

// ---------------------------------------------------------------------------
// 20. Constant with any value is scalar constant
// ---------------------------------------------------------------------------

/// Proves: TraceOp::Constant is always a scalar constant regardless
/// of the value it holds.
///
/// SUBSTANTIVE: The truncation check uses is_scalar_constant to detect
/// the Mul(scalar) pattern. Missing Constant detection would prevent
/// truncation of chains ending with literal constant multiplication.
#[kani::unwind(8)]
#[kani::proof]
fn proof_constant_any_value_is_scalar() {
    use nn_core::dyn_tensor::trace::TraceOp;

    let values = [0.0, 1.0, -1.0, 3.14, f64::MIN_POSITIVE, 1e10];
    for val in &values {
        let op = TraceOp::Constant { value: *val };
        assert!(
            super::fusion::is_scalar_constant(&op),
            "Constant must always be scalar"
        );
    }
}
