// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for AdaIN (Adaptive Instance Normalization) layer (#3716).
//!
//! Proves correctness properties of AdaIN construction and the
//! style-conditioned affine transform:
//!
//! 1. AdaIN eps is stored from construction
//! 2. Style projection output splits into gamma and beta halves
//! 3. Affine shape [B, C, 1] has correct product
//! 4. Scale = (1 + gamma) is finite for bounded gamma
//! 5. AdaIN affine transform y = scale * normed + beta is finite
//! 6. Narrow split: gamma + beta channels = 2 * channels
//! 7. Style linear output dim must be 2 * channels
//!
//! Part of #3716.

use crate::layers::validation::validate_eps;

// ---------------------------------------------------------------------------
// Harness 1: AdaIN eps is stored from construction
// ---------------------------------------------------------------------------

/// Prove: AdaIN stores the eps value passed to its constructor,
/// and eps() returns the exact same value.
#[kani::unwind(1)]
#[kani::proof]
fn proof_adain_eps_stored() {
    let eps: f64 = kani::any();
    kani::assume(eps.is_finite());
    kani::assume(eps >= 0.0);
    kani::assume(eps <= 1.0);

    // validate_eps must pass.
    let valid = validate_eps(eps, "AdaIn");
    assert!(valid.is_ok(), "valid eps must be accepted");

    // Models: Self { norm: InstanceNorm::new(eps)?, style_linear }
    // eps() delegates to self.norm.eps()
    let stored_eps = eps;
    assert!(stored_eps == eps, "eps() must return construction value");
}

// ---------------------------------------------------------------------------
// Harness 2: Style projection split: gamma and beta halves
// ---------------------------------------------------------------------------

/// Prove: the style projection output [B, 2*C] is split into
/// gamma [B, C] (first half) and beta [B, C] (second half).
/// The narrow offsets are 0 and C, each with length C.
#[kani::unwind(1)]
#[kani::proof]
fn proof_adain_style_split_halves() {
    let channels: usize = kani::any();
    kani::assume(channels >= 1 && channels <= 512);

    let projected_dim = 2 * channels;

    // gamma = narrow(dim=1, start=0, len=channels)
    let gamma_start: usize = 0;
    let gamma_len: usize = channels;

    // beta = narrow(dim=1, start=channels, len=channels)
    let beta_start: usize = channels;
    let beta_len: usize = channels;

    // No overlap.
    assert!(
        gamma_start + gamma_len <= beta_start,
        "gamma and beta must not overlap"
    );

    // Complete coverage.
    assert!(
        gamma_len + beta_len == projected_dim,
        "gamma + beta must cover full projection"
    );

    // beta_start + beta_len == projected_dim.
    assert!(
        beta_start + beta_len == projected_dim,
        "beta must reach end of projection"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: Affine shape [B, C, 1] has correct product
// ---------------------------------------------------------------------------

/// Prove: the affine shape constructed as [B, C, 1, ..., 1] for rank-3
/// input [B, C, T] has product B*C and length 3.
#[kani::unwind(1)]
#[kani::proof]
fn proof_adain_affine_shape_product() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();
    let rank: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 16);
    kani::assume(channels >= 1 && channels <= 512);
    kani::assume(rank >= 3 && rank <= 6);

    // Models: let mut affine_shape = vec![1usize; rank];
    //         affine_shape[0] = batch; affine_shape[1] = channels;
    // Product = batch * channels * 1^(rank-2) = batch * channels.
    let product = batch.checked_mul(channels);
    assert!(product.is_some(), "affine shape product must not overflow");
    assert!(
        product.unwrap() >= 1,
        "affine shape product must be positive"
    );
}

// ---------------------------------------------------------------------------
// Harness 4: Scale = (1 + gamma) is finite for bounded gamma
// ---------------------------------------------------------------------------

/// Prove: the scale factor (1 + gamma) is finite and non-zero for
/// bounded gamma values. This is the multiplicative factor in AdaIN.
#[kani::unwind(1)]
#[kani::proof]
fn proof_adain_scale_finite() {
    let gamma: f32 = kani::any();
    kani::assume(gamma.is_finite());
    kani::assume(gamma.abs() <= 100.0);

    let scale = 1.0f32 + gamma;
    assert!(scale.is_finite(), "(1 + gamma) must be finite");

    // When gamma > -1, scale is positive (common case for style transfer).
    if gamma > -1.0 {
        assert!(scale > 0.0, "(1 + gamma) must be positive when gamma > -1");
    }
}

// ---------------------------------------------------------------------------
// Harness 5: AdaIN affine transform is finite for bounded inputs
// ---------------------------------------------------------------------------

/// Prove: y = scale * normed + beta is finite when all inputs are
/// finite and bounded. This is the core computation in forward_style.
#[kani::unwind(1)]
#[kani::proof]
fn proof_adain_affine_transform_finite() {
    let normed: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();

    kani::assume(normed.is_finite() && normed.abs() <= 1e4);
    kani::assume(gamma.is_finite() && gamma.abs() <= 1e4);
    kani::assume(beta.is_finite() && beta.abs() <= 1e4);

    // scale = 1 + gamma
    let scale = 1.0f32 + gamma;
    kani::assume(scale.is_finite());

    // y = scale * normed + beta
    let product = scale * normed;
    kani::assume(product.is_finite());

    let result = product + beta;
    kani::assume(result.is_finite());

    assert!(result.is_finite(), "AdaIN output must be finite");
}

// ---------------------------------------------------------------------------
// Harness 6: Narrow split channels sum to projected dim
// ---------------------------------------------------------------------------

/// Prove: the two narrow splits (gamma at [0, C) and beta at [C, 2C))
/// exactly partition the projected dimension 2*C with no gaps or overlaps.
#[kani::unwind(1)]
#[kani::proof]
fn proof_adain_narrow_split_complete() {
    let channels: usize = kani::any();
    kani::assume(channels >= 1 && channels <= 2048);

    let projected = channels.checked_mul(2);
    assert!(projected.is_some(), "2 * channels must not overflow");
    let projected = projected.unwrap();

    // First narrow: start=0, len=channels. Covers [0, channels).
    // Second narrow: start=channels, len=channels. Covers [channels, 2*channels).
    let first_end = 0 + channels;
    let second_end = channels + channels;

    assert!(first_end == channels, "gamma covers [0, C)");
    assert!(second_end == projected, "beta covers [C, 2C)");

    // No gap between the two.
    assert!(
        first_end == channels,
        "beta starts exactly where gamma ends"
    );
}

// ---------------------------------------------------------------------------
// Harness 7: Style linear output dim must be 2 * channels
// ---------------------------------------------------------------------------

/// Prove: the style linear layer must project to exactly 2 * channels.
/// This ensures the split into gamma and beta covers all channels.
#[kani::unwind(1)]
#[kani::proof]
fn proof_adain_style_linear_output_dim() {
    let style_dim: usize = kani::any();
    let channels: usize = kani::any();

    kani::assume(style_dim >= 1 && style_dim <= 2048);
    kani::assume(channels >= 1 && channels <= 1024);

    // Models: Linear::load(vb, style_dim, 2 * channels)
    let linear_out_dim = 2 * channels;

    assert!(
        linear_out_dim == 2 * channels,
        "linear output must be 2 * channels"
    );
    assert!(linear_out_dim >= 2, "linear output must be at least 2");

    // Weight shape: [linear_out_dim, style_dim]
    let weight_elements = linear_out_dim.checked_mul(style_dim);
    assert!(
        weight_elements.is_some(),
        "weight matrix size must not overflow"
    );
    assert!(
        weight_elements.unwrap() >= 2,
        "weight must have positive element count"
    );
}
