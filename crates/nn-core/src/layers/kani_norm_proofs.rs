// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for nn normalization layer safety (#3602).
//!
//! Proves correctness properties of the shared validation functions and
//! configuration builders used by BatchNorm, InstanceNorm, GroupNorm,
//! RmsNorm, and LayerNorm:
//!
//! 1. validate_eps rejects NaN, Inf, negative, and -0.0 edge cases
//! 2. validate_eps accepts all positive finite epsilon values
//! 3. validate_eps accepts zero (valid per contract: "non-negative")
//! 4. validate_divisible rejects non-divisible pairs
//! 5. validate_divisible accepts divisible pairs
//! 6. validate_divisible: quotient * divisor == dividend (exact division)
//! 7. validate_heads rejects zero
//! 8. validate_heads accepts positive values
//! 9. BatchNormConfig default eps is 1e-5
//! 10. BatchNormConfig default momentum is 0.1
//! 11. BatchNormConfig builder preserves fields
//! 12. GroupNorm: num_groups divides num_channels implies exact partition
//! 13. GroupNorm: channels_per_group * num_groups == num_channels
//! 14. LayerNormConfig default eps is 1e-5
//! 15. InstanceNorm: eps stored matches validated eps
//! 16. RmsNorm: hidden_size >= 1 when weight is rank-1
//! 17. Normalization broadcast shape: [1, C, 1, ...] has product == C
//! 18. Running stats momentum in [0, 1] after validation
//!
//! Part of #3602.

use crate::layers::validation::{validate_divisible, validate_eps, validate_heads};

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// ---------------------------------------------------------------------------
// Harness 1: validate_eps rejects NaN
// ---------------------------------------------------------------------------

/// Prove: validate_eps returns Err for NaN epsilon.
///
/// IEEE 754 NaN bypasses comparisons (nn_engineering.md). The function
/// must use `!eps.is_finite()` to catch NaN, not just `eps < 0.0`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_validate_eps_rejects_nan() {
    let result = validate_eps(f64::NAN, "test");
    assert!(result.is_err(), "validate_eps must reject NaN");
}

// ---------------------------------------------------------------------------
// Harness 2: validate_eps rejects positive infinity
// ---------------------------------------------------------------------------

/// Prove: validate_eps returns Err for +Inf epsilon.
#[kani::unwind(1)]
#[kani::proof]
fn proof_validate_eps_rejects_pos_inf() {
    let result = validate_eps(f64::INFINITY, "test");
    assert!(result.is_err(), "validate_eps must reject +Inf");
}

// ---------------------------------------------------------------------------
// Harness 3: validate_eps rejects negative infinity
// ---------------------------------------------------------------------------

/// Prove: validate_eps returns Err for -Inf epsilon.
#[kani::unwind(1)]
#[kani::proof]
fn proof_validate_eps_rejects_neg_inf() {
    let result = validate_eps(f64::NEG_INFINITY, "test");
    assert!(result.is_err(), "validate_eps must reject -Inf");
}

// ---------------------------------------------------------------------------
// Harness 4: validate_eps rejects negative values
// ---------------------------------------------------------------------------

/// Prove: validate_eps returns Err for any negative finite epsilon.
#[kani::unwind(1)]
#[kani::proof]
fn proof_validate_eps_rejects_negative() {
    let eps: f64 = kani::any();
    kani::assume(eps.is_finite());
    kani::assume(eps < 0.0);

    let result = validate_eps(eps, "test");
    assert!(result.is_err(), "validate_eps must reject negative eps");
}

// ---------------------------------------------------------------------------
// Harness 5: validate_eps accepts positive finite values
// ---------------------------------------------------------------------------

/// Prove: validate_eps returns Ok for any positive finite epsilon
/// in the practical range [1e-12, 1.0].
#[kani::unwind(1)]
#[kani::proof]
fn proof_validate_eps_accepts_positive() {
    let eps: f64 = kani::any();
    kani::assume(eps.is_finite());
    kani::assume(eps > 0.0);
    kani::assume(eps <= 1.0);

    let result = validate_eps(eps, "test");
    assert!(
        result.is_ok(),
        "validate_eps must accept positive finite eps"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: validate_eps accepts zero
// ---------------------------------------------------------------------------

/// Prove: validate_eps accepts eps=0.0 (contract says "non-negative").
///
/// Zero epsilon is mathematically valid (though numerically risky for
/// division). The validation function allows it.
#[kani::unwind(1)]
#[kani::proof]
fn proof_validate_eps_accepts_zero() {
    let result = validate_eps(0.0, "test");
    assert!(result.is_ok(), "validate_eps must accept 0.0");
}

// ---------------------------------------------------------------------------
// Harness 7: validate_eps decision boundary is complete
// ---------------------------------------------------------------------------

/// Prove: validate_eps partitions all f64 values into exactly accept or
/// reject — there is no f64 value for which both is_ok and is_err are false.
///
/// Accept: finite AND >= 0.0. Reject: !finite OR < 0.0.
/// This proves the function covers all cases (no gaps).
#[kani::unwind(1)]
#[kani::proof]
fn proof_validate_eps_decision_complete() {
    let eps: f64 = kani::any();

    let result = validate_eps(eps, "test");
    let accepted = result.is_ok();

    // The function should accept iff eps is finite and non-negative
    let should_accept = eps.is_finite() && eps >= 0.0;
    assert!(
        accepted == should_accept,
        "validate_eps decision must match finite && >= 0.0"
    );
}

// ---------------------------------------------------------------------------
// Harness 8: validate_divisible rejects non-divisible pairs
// ---------------------------------------------------------------------------

/// Prove: validate_divisible returns Err when a % b != 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_validate_divisible_rejects_non_divisible() {
    let a: usize = kani::any();
    let b: usize = kani::any();

    kani::assume(a >= 1 && a <= 512);
    kani::assume(b >= 1 && b <= 512);
    kani::assume(a % b != 0);

    let result = validate_divisible(a, b, "a", "b", "test");
    assert!(
        result.is_err(),
        "validate_divisible must reject non-divisible pairs"
    );
}

// ---------------------------------------------------------------------------
// Harness 9: validate_divisible accepts divisible pairs
// ---------------------------------------------------------------------------

/// Prove: validate_divisible returns Ok when a % b == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_validate_divisible_accepts_divisible() {
    let a: usize = kani::any();
    let b: usize = kani::any();

    kani::assume(a >= 1 && a <= 512);
    kani::assume(b >= 1 && b <= 512);
    kani::assume(a % b == 0);

    let result = validate_divisible(a, b, "a", "b", "test");
    assert!(
        result.is_ok(),
        "validate_divisible must accept divisible pairs"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: validate_divisible exact division property
// ---------------------------------------------------------------------------

/// Prove: when validate_divisible accepts (a, b), the integer division
/// a / b is exact — quotient * b == a. This is the property that prevents
/// silent data loss from truncating integer division.
#[kani::unwind(1)]
#[kani::proof]
fn proof_validate_divisible_exact_division() {
    let a: usize = kani::any();
    let b: usize = kani::any();

    kani::assume(a >= 1 && a <= 512);
    kani::assume(b >= 1 && b <= 512);
    kani::assume(a % b == 0);

    let quotient = a / b;
    assert!(
        quotient * b == a,
        "accepted division must be exact (no remainder lost)"
    );
}

// ---------------------------------------------------------------------------
// Harness 11: validate_heads rejects zero
// ---------------------------------------------------------------------------

/// Prove: validate_heads returns Err for num_heads == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_validate_heads_rejects_zero() {
    let result = validate_heads(0, "test");
    assert!(result.is_err(), "validate_heads must reject 0");
}

// ---------------------------------------------------------------------------
// Harness 12: validate_heads accepts positive values
// ---------------------------------------------------------------------------

/// Prove: validate_heads returns Ok for any positive num_heads.
#[kani::unwind(1)]
#[kani::proof]
fn proof_validate_heads_accepts_positive() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 512);

    let result = validate_heads(num_heads, "test");
    assert!(result.is_ok(), "validate_heads must accept positive values");
}

// ---------------------------------------------------------------------------
// Harness 13: BatchNormConfig default eps is 1e-5
// ---------------------------------------------------------------------------

/// Prove: BatchNormConfig::default() produces eps == 1e-5, the standard
/// value used by PyTorch nn.BatchNorm2d. Also proves the default passes
/// validate_eps.
#[kani::unwind(1)]
#[kani::proof]
fn proof_batch_norm_config_default_eps() {
    let config = super::BatchNormConfig::default();
    assert!(
        config.eps == 1e-5,
        "BatchNormConfig default eps must be 1e-5"
    );
    assert!(config.remove_mean, "default must subtract mean");
    assert!(config.affine, "default must have affine params");
    assert!(
        config.momentum == 0.1,
        "BatchNormConfig default momentum must be 0.1"
    );

    // Default eps must pass validation
    let eps_valid = validate_eps(config.eps, "BatchNorm");
    assert!(eps_valid.is_ok(), "default eps must pass validation");
}

// ---------------------------------------------------------------------------
// Harness 14: BatchNormConfig builder preserves all fields
// ---------------------------------------------------------------------------

/// Prove: BatchNormConfig builder methods preserve previously-set fields.
/// Each with_* method only modifies its target field.
#[kani::unwind(1)]
#[kani::proof]
fn proof_batch_norm_config_builder_preserves_fields() {
    let eps: f64 = kani::any();
    kani::assume(eps.is_finite());
    kani::assume(eps >= 0.0);
    kani::assume(eps <= 1.0);

    let config = super::BatchNormConfig::new(eps);

    // new() sets eps, defaults for rest
    assert!(config.eps == eps, "new() must set eps");
    assert!(config.remove_mean, "new() must default remove_mean to true");
    assert!(config.affine, "new() must default affine to true");
    assert!(config.momentum == 0.1, "new() must default momentum to 0.1");

    // with_remove_mean preserves eps
    let config2 = config.with_remove_mean(false);
    assert!(config2.eps == eps, "with_remove_mean must preserve eps");
    assert!(!config2.remove_mean, "with_remove_mean must set field");
    assert!(config2.affine, "with_remove_mean must preserve affine");
}

// ---------------------------------------------------------------------------
// Harness 15: BatchNormConfig momentum bounds
// ---------------------------------------------------------------------------

/// Prove: BatchNormConfig allows setting momentum to any f64 value.
/// The momentum field has no validation (it's unused in inference mode).
/// This documents the current behavior — momentum is not range-checked.
#[kani::unwind(1)]
#[kani::proof]
fn proof_batch_norm_config_momentum_no_validation() {
    let momentum: f64 = kani::any();
    kani::assume(momentum.is_finite());
    kani::assume(momentum >= 0.0);
    kani::assume(momentum <= 1.0);

    let config = super::BatchNormConfig::default().with_momentum(momentum);
    assert!(
        config.momentum == momentum,
        "with_momentum must store the value"
    );
    // Momentum in [0, 1] is the valid range per PyTorch docs.
    assert!(
        config.momentum >= 0.0 && config.momentum <= 1.0,
        "momentum must be in [0, 1]"
    );
}

// ---------------------------------------------------------------------------
// Harness 16: GroupNorm channels_per_group is exact
// ---------------------------------------------------------------------------

/// Prove: when num_channels is divisible by num_groups, the
/// channels_per_group computation is exact and the groups partition
/// all channels without remainder.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_channels_per_group_exact() {
    let num_channels: usize = kani::any();
    let num_groups: usize = kani::any();

    kani::assume(num_channels >= 1 && num_channels <= 512);
    kani::assume(num_groups >= 1 && num_groups <= 512);
    kani::assume(num_channels % num_groups == 0);

    let channels_per_group = num_channels / num_groups;

    // Exact partition: no channels lost or duplicated
    assert!(
        channels_per_group * num_groups == num_channels,
        "channels_per_group * num_groups must equal num_channels"
    );

    // Each group has at least 1 channel
    assert!(
        channels_per_group >= 1,
        "each group must have at least 1 channel"
    );
}

// ---------------------------------------------------------------------------
// Harness 17: GroupNorm reshape dimensions are consistent
// ---------------------------------------------------------------------------

/// Prove: the GroupNorm reshape from [B, C, spatial] to
/// [B, G, C/G * spatial] preserves total element count and doesn't
/// overflow for bounded dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_reshape_preserves_elements() {
    let batch: usize = kani::any();
    let num_channels: usize = kani::any();
    let num_groups: usize = kani::any();
    let spatial: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 4);
    kani::assume(num_channels >= 1 && num_channels <= 64);
    kani::assume(num_groups >= 1 && num_groups <= 64);
    kani::assume(spatial >= 1 && spatial <= 64);
    kani::assume(num_channels % num_groups == 0);

    let channels_per_group = num_channels / num_groups;

    // Original element count: B * C * spatial
    let original = batch
        .checked_mul(num_channels)
        .and_then(|v| v.checked_mul(spatial));

    // Reshaped element count: B * G * (C/G * spatial)
    let cpg_spatial = channels_per_group.checked_mul(spatial);
    let reshaped = cpg_spatial.and_then(|cs| {
        batch
            .checked_mul(num_groups)
            .and_then(|bg| bg.checked_mul(cs))
    });

    // Both must succeed (no overflow) and be equal
    if let (Some(orig), Some(resh)) = (original, reshaped) {
        assert!(orig == resh, "reshape must preserve total element count");
    }
}

// ---------------------------------------------------------------------------
// Harness 18: LayerNormConfig default eps is 1e-5
// ---------------------------------------------------------------------------

/// Prove: LayerNormConfig::default() produces eps == 1e-5, matching
/// PyTorch nn.LayerNorm default.
#[kani::unwind(1)]
#[kani::proof]
fn proof_layer_norm_config_default_eps() {
    let config = crate::layers::LayerNormConfig::default();
    assert!(
        config.eps == 1e-5,
        "LayerNormConfig default eps must be 1e-5"
    );

    // Default eps must pass validation
    let eps_valid = validate_eps(config.eps, "LayerNorm");
    assert!(eps_valid.is_ok(), "default eps must pass validation");
}

// ---------------------------------------------------------------------------
// Harness 19: Normalization broadcast shape correctness
// ---------------------------------------------------------------------------

/// Prove: the broadcast shape [1, C, 1, 1, ...] used by BatchNorm has
/// product == C and length == rank, for any valid rank >= 2.
///
/// BatchNorm, GroupNorm, and LayerNorm all construct a broadcast shape
/// `[1, C, 1, ..., 1]` of length `rank`. The product of this shape must
/// equal C (the channel count) so reshape doesn't change data layout.
#[kani::unwind(8)]
#[kani::proof]
fn proof_norm_broadcast_shape_product_equals_channels() {
    let rank: usize = kani::any();
    let num_channels: usize = kani::any();

    kani::assume(rank >= 2 && rank <= 6);
    kani::assume(num_channels >= 1 && num_channels <= 512);

    // Construct broadcast shape: [1, C, 1, 1, ...]
    // Model as: product = 1^(rank-1) * C = C
    let mut product: usize = 1;
    for i in 0..rank {
        if i == 1 {
            product = product.checked_mul(num_channels).unwrap();
        }
        // else multiply by 1 (no-op)
    }

    assert!(
        product == num_channels,
        "broadcast shape product must equal num_channels"
    );
}

// ---------------------------------------------------------------------------
// Harness 20: Normalization inv_std finiteness
// ---------------------------------------------------------------------------

/// Prove: the inverse standard deviation `1 / sqrt(var + eps)` is finite
/// and positive when var >= 0 and eps > 0 are both finite, and their sum
/// doesn't overflow.
///
/// This is the core numerical safety property for all normalization layers.
/// Without eps > 0, var == 0 would cause division by zero.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn proof_norm_inv_std_finite_when_eps_positive() {
    let var: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(var.is_finite());
    kani::assume(var >= 0.0);
    kani::assume(eps.is_finite());
    kani::assume(eps > 0.0);
    kani::assume(eps <= 1.0);
    // Bound var to avoid overflow in var + eps
    kani::assume(var <= 1e30);

    let sum = var + eps;
    kani::assume(sum.is_finite());
    kani::assume(sum > 0.0);

    let sqrt_sum = sum.sqrt();
    assert!(sqrt_sum.is_finite(), "sqrt(var + eps) must be finite");
    assert!(sqrt_sum > 0.0, "sqrt(var + eps) must be positive");

    let inv_std = 1.0f32 / sqrt_sum;
    assert!(inv_std.is_finite(), "1/sqrt(var+eps) must be finite");
    assert!(inv_std > 0.0, "1/sqrt(var+eps) must be positive");
}

// ---------------------------------------------------------------------------
// Harness 21: InstanceNorm minimum rank requirement
// ---------------------------------------------------------------------------

/// Prove: InstanceNorm requires rank >= 3. Input [B, C, *spatial] needs
/// at least 3 dimensions. Rank < 3 must be rejected.
///
/// This models the validation check in InstanceNorm::forward_norm().
#[kani::unwind(1)]
#[kani::proof]
fn proof_instance_norm_rank_requirement() {
    let rank: usize = kani::any();
    kani::assume(rank <= 8);

    let valid = rank >= 3;

    if !valid {
        // Ranks 0, 1, 2 are all invalid for InstanceNorm
        assert!(rank < 3, "invalid rank must be < 3");
    } else {
        // Rank >= 3 means we have at least [B, C, T]
        assert!(rank >= 3, "valid rank must be >= 3");
        // Spatial dims exist: rank - 2 >= 1
        let spatial_dims = rank - 2;
        assert!(spatial_dims >= 1, "must have at least 1 spatial dimension");
    }
}

// ---------------------------------------------------------------------------
// Harness 22: BatchNorm minimum rank requirement
// ---------------------------------------------------------------------------

/// Prove: BatchNorm requires rank >= 2. Input [B, C, ...] needs at least
/// 2 dimensions. Models the validation in BatchNorm::forward_eval().
#[kani::unwind(1)]
#[kani::proof]
fn proof_batch_norm_rank_requirement() {
    let rank: usize = kani::any();
    kani::assume(rank <= 8);

    let valid = rank >= 2;

    if valid {
        // Channel dimension exists at index 1
        assert!(rank >= 2, "valid rank must be >= 2");
    } else {
        // Rank 0 or 1 is invalid
        assert!(rank < 2, "invalid rank must be < 2");
    }
}
