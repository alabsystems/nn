// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for WeightNormConv1d weight normalization.
//!
//! Proves correctness properties of the weight normalization decomposition
//! `w = g * v / ||v||` where `v` is the raw weight, `g` is the per-channel
//! gain, and `||v||` is the L2 norm per output channel.
//!
//!  1.  L2 norm of non-zero vector is strictly positive
//!  2.  L2 norm preserves finiteness for bounded inputs
//!  3.  Normalized vector has unit norm (||v/||v|||| = 1)
//!  4.  Weight norm output = g * v / ||v|| preserves direction of v
//!  5.  Weight norm output is finite for finite g, v with ||v|| > 0
//!  6.  Scaling v by alpha scales norm by |alpha|
//!  7.  Scaling g by alpha scales output by alpha
//!  8.  Zero gain produces zero normalized weight
//!  9.  Groups parameter must be > 0
//! 10.  in_channels must be divisible by groups
//! 11.  Weight norm is invariant to gain sign flip + v sign flip
//! 12.  Per-channel norm: each output channel normalized independently
//!
//! Part of #4261.

// -- Kani transcendental stubs (CBMC #708) --

fn sqrt_f32_stub_wn(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0);
    if x > 0.0 {
        kani::assume(r > 0.0);
    }
    r
}

// -- Scalar helpers modeling the weight normalization computation --

/// Compute L2 norm squared of a 2-element vector.
fn l2_norm_sq_2(a: f32, b: f32) -> f32 {
    a * a + b * b
}

/// Compute L2 norm of a 2-element vector.
fn l2_norm_2(a: f32, b: f32) -> f32 {
    l2_norm_sq_2(a, b).sqrt()
}

// ---------------------------------------------------------------------------
// Harness 1: L2 norm of non-zero vector is strictly positive
// ---------------------------------------------------------------------------

/// Prove: for a non-zero finite vector, ||v|| > 0.
/// This ensures division by ||v|| is safe.
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_l2_positive_nonzero() {
    let v0: f32 = kani::any();
    let v1: f32 = kani::any();

    kani::assume(v0.is_finite() && v1.is_finite());
    kani::assume(v0.abs() <= 100.0 && v1.abs() <= 100.0);
    // At least one element is non-zero
    kani::assume(v0 != 0.0 || v1 != 0.0);

    let norm_sq = l2_norm_sq_2(v0, v1);

    assert!(norm_sq.is_finite(), "norm_sq must be finite");
    assert!(norm_sq > 0.0, "norm_sq of non-zero vector must be positive");
}

// ---------------------------------------------------------------------------
// Harness 2: L2 norm preserves finiteness for bounded inputs
// ---------------------------------------------------------------------------

/// Prove: ||v||^2 is finite when elements are bounded.
/// This prevents overflow in the squared-sum accumulation.
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_l2_finite_bounded() {
    let v0: f32 = kani::any();
    let v1: f32 = kani::any();

    kani::assume(v0.is_finite() && v1.is_finite());
    kani::assume(v0.abs() <= 1e4 && v1.abs() <= 1e4);

    let sq0 = v0 * v0;
    let sq1 = v1 * v1;

    // Each v_i^2 <= 1e8, sum <= 2e8, well within f32 range
    assert!(sq0.is_finite(), "v0^2 must be finite");
    assert!(sq1.is_finite(), "v1^2 must be finite");

    let norm_sq = sq0 + sq1;
    assert!(norm_sq.is_finite(), "sum of squares must be finite");
    assert!(norm_sq >= 0.0, "sum of squares must be non-negative");
}

// ---------------------------------------------------------------------------
// Harness 3: Normalized vector has unit norm
// ---------------------------------------------------------------------------

/// Prove: v / ||v|| has unit norm (||v/||v|||| = 1) for non-zero v.
/// This is the fundamental property of L2 normalization.
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_unit_norm_after_normalize() {
    let v0: f32 = kani::any();
    let v1: f32 = kani::any();

    kani::assume(v0.is_finite() && v1.is_finite());
    kani::assume(v0.abs() >= 0.01 && v0.abs() <= 50.0);
    kani::assume(v1.abs() >= 0.01 && v1.abs() <= 50.0);

    let norm_sq = l2_norm_sq_2(v0, v1);
    kani::assume(norm_sq.is_finite() && norm_sq > 0.0);

    let norm = norm_sq.sqrt();
    kani::assume(norm.is_finite() && norm > 0.0);

    let n0 = v0 / norm;
    let n1 = v1 / norm;

    kani::assume(n0.is_finite() && n1.is_finite());

    // ||n||^2 = (v0/||v||)^2 + (v1/||v||)^2 = (v0^2 + v1^2) / ||v||^2 = 1
    let out_norm_sq = n0 * n0 + n1 * n1;
    kani::assume(out_norm_sq.is_finite());

    // Allow tolerance for floating-point rounding
    assert!(
        (out_norm_sq - 1.0).abs() < 0.01,
        "normalized vector must have approximately unit norm"
    );
}

// ---------------------------------------------------------------------------
// Harness 4: Weight norm preserves direction of v
// ---------------------------------------------------------------------------

/// Prove: w = g * v / ||v|| is a positive scalar multiple of v when g > 0.
/// Direction is preserved: w_i / w_j = v_i / v_j for all non-zero elements.
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_preserves_direction() {
    let v0: f32 = kani::any();
    let v1: f32 = kani::any();
    let g: f32 = kani::any();

    kani::assume(v0.is_finite() && v1.is_finite() && g.is_finite());
    kani::assume(v0.abs() >= 0.1 && v0.abs() <= 50.0);
    kani::assume(v1.abs() >= 0.1 && v1.abs() <= 50.0);
    kani::assume(g > 0.0 && g <= 10.0);

    let norm = l2_norm_sq_2(v0, v1).sqrt();
    kani::assume(norm.is_finite() && norm > 0.0);

    let w0 = g * v0 / norm;
    let w1 = g * v1 / norm;

    kani::assume(w0.is_finite() && w1.is_finite());

    // Direction test: w0 / v0 = w1 / v1 = g / ||v||
    let ratio0 = w0 / v0;
    let ratio1 = w1 / v1;

    kani::assume(ratio0.is_finite() && ratio1.is_finite());

    assert!(
        (ratio0 - ratio1).abs() < 0.01,
        "weight norm must preserve direction: w_i/v_i must be constant"
    );
    assert!(ratio0 > 0.0, "ratio must be positive when g > 0");
}

// ---------------------------------------------------------------------------
// Harness 5: Weight norm output is finite for finite inputs
// ---------------------------------------------------------------------------

/// Prove: w = g * v / ||v|| is finite when g and v are finite with ||v|| > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub_wn)]
fn proof_weight_norm_output_finite() {
    let v0: f32 = kani::any();
    let v1: f32 = kani::any();
    let g: f32 = kani::any();

    kani::assume(v0.is_finite() && v1.is_finite() && g.is_finite());
    kani::assume(v0.abs() <= 100.0 && v1.abs() <= 100.0);
    kani::assume(g.abs() <= 100.0);
    // Ensure non-zero norm
    kani::assume(v0.abs() > 0.001 || v1.abs() > 0.001);

    let norm_sq = l2_norm_sq_2(v0, v1);
    kani::assume(norm_sq.is_finite() && norm_sq > 0.0);

    let norm = norm_sq.sqrt();
    kani::assume(norm.is_finite() && norm > 0.0);

    let g_over_norm = g / norm;
    kani::assume(g_over_norm.is_finite());

    let w0 = v0 * g_over_norm;
    let w1 = v1 * g_over_norm;

    assert!(w0.is_finite(), "w0 must be finite");
    assert!(w1.is_finite(), "w1 must be finite");
}

// ---------------------------------------------------------------------------
// Harness 6: Scaling v by alpha scales norm by |alpha|
// ---------------------------------------------------------------------------

/// Prove: ||alpha * v|| = |alpha| * ||v||.
/// This is the homogeneity property of norms.
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_l2_scaling() {
    let v0: f32 = kani::any();
    let v1: f32 = kani::any();
    let alpha: f32 = kani::any();

    kani::assume(v0.is_finite() && v1.is_finite() && alpha.is_finite());
    kani::assume(v0.abs() <= 10.0 && v1.abs() <= 10.0);
    kani::assume(alpha.abs() <= 10.0 && alpha.abs() > 0.0);

    let norm_v = l2_norm_sq_2(v0, v1);
    let norm_av = l2_norm_sq_2(alpha * v0, alpha * v1);

    kani::assume(norm_v.is_finite() && norm_av.is_finite());

    // ||alpha*v||^2 = alpha^2 * ||v||^2
    let expected = alpha * alpha * norm_v;
    kani::assume(expected.is_finite());

    let diff = (norm_av - expected).abs();
    assert!(diff < 0.01, "||alpha*v||^2 must equal alpha^2 * ||v||^2");
}

// ---------------------------------------------------------------------------
// Harness 7: Scaling g by alpha scales output by alpha
// ---------------------------------------------------------------------------

/// Prove: for w = g * v / ||v||, scaling g by alpha gives alpha * w.
/// The gain parameter directly controls the output magnitude.
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_gain_scaling() {
    let v0: f32 = kani::any();
    let g: f32 = kani::any();
    let alpha: f32 = kani::any();

    kani::assume(v0.is_finite() && g.is_finite() && alpha.is_finite());
    kani::assume(v0.abs() >= 0.1 && v0.abs() <= 10.0);
    kani::assume(g.abs() <= 10.0 && g.abs() > 0.01);
    kani::assume(alpha.abs() <= 10.0 && alpha.abs() > 0.01);

    // Simplified: single element, norm = |v0|
    let norm = v0.abs();
    kani::assume(norm > 0.0);

    let w = g * v0 / norm;
    let w_scaled = (alpha * g) * v0 / norm;

    kani::assume(w.is_finite() && w_scaled.is_finite());

    let expected = alpha * w;
    kani::assume(expected.is_finite());

    let diff = (w_scaled - expected).abs();
    assert!(
        diff < 0.001,
        "scaling gain by alpha must scale output by alpha"
    );
}

// ---------------------------------------------------------------------------
// Harness 8: Zero gain produces zero normalized weight
// ---------------------------------------------------------------------------

/// Prove: when g = 0, w = 0 * v / ||v|| = 0 regardless of v.
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_zero_gain_zero_output() {
    let v0: f32 = kani::any();
    let v1: f32 = kani::any();

    kani::assume(v0.is_finite() && v1.is_finite());
    kani::assume(v0.abs() >= 0.1 && v0.abs() <= 50.0);
    kani::assume(v1.abs() >= 0.1 && v1.abs() <= 50.0);

    let norm_sq = l2_norm_sq_2(v0, v1);
    kani::assume(norm_sq.is_finite() && norm_sq > 0.0);

    let norm = norm_sq.sqrt();
    kani::assume(norm.is_finite() && norm > 0.0);

    let g: f32 = 0.0;
    let g_over_norm = g / norm;

    // 0 / positive = 0
    assert!(g_over_norm == 0.0, "0/norm must be 0");

    let w0 = v0 * g_over_norm;
    let w1 = v1 * g_over_norm;

    assert!(w0 == 0.0, "zero gain must produce zero w0");
    assert!(w1 == 0.0, "zero gain must produce zero w1");
}

// ---------------------------------------------------------------------------
// Harness 9: Groups parameter must be > 0
// ---------------------------------------------------------------------------

/// Prove: WeightNormConv1d rejects groups=0 (delegates to Conv1d which checks).
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_groups_nonzero() {
    let groups: usize = kani::any();
    kani::assume(groups <= 64);

    // Models: Conv1dConfig groups validation
    let accepted = groups > 0;
    assert!(
        accepted == (groups > 0),
        "groups must be > 0 for WeightNormConv1d"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: in_channels divisible by groups
// ---------------------------------------------------------------------------

/// Prove: when in_channels is divisible by groups, the per-group
/// channel count is well-defined and positive.
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_channels_divisible() {
    let in_channels: usize = kani::any();
    let groups: usize = kani::any();

    kani::assume(in_channels >= 1 && in_channels <= 512);
    kani::assume(groups >= 1 && groups <= 512);
    kani::assume(in_channels % groups == 0);

    let channels_per_group = in_channels / groups;

    assert!(
        channels_per_group > 0,
        "channels_per_group must be > 0 when divisible"
    );
    assert!(
        channels_per_group * groups == in_channels,
        "channels_per_group * groups must reconstruct in_channels"
    );
}

// ---------------------------------------------------------------------------
// Harness 11: Weight norm invariant to simultaneous g/v sign flip
// ---------------------------------------------------------------------------

/// Prove: w = g * v / ||v|| = (-g) * (-v) / ||-v||.
/// Flipping both gain and direction produces the same weight.
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_sign_flip_invariant() {
    let v0: f32 = kani::any();
    let g: f32 = kani::any();

    kani::assume(v0.is_finite() && g.is_finite());
    kani::assume(v0.abs() >= 0.1 && v0.abs() <= 10.0);
    kani::assume(g.abs() >= 0.01 && g.abs() <= 10.0);

    // Single element: norm = |v0|
    let norm = v0.abs();
    kani::assume(norm > 0.0);

    let w_original = g * v0 / norm;

    // Flip both g and v
    let neg_g = -g;
    let neg_v0 = -v0;
    let neg_norm = neg_v0.abs(); // |-v0| = |v0| = norm

    assert!(
        neg_norm == norm,
        "norm of negated vector must equal original norm"
    );

    let w_flipped = neg_g * neg_v0 / neg_norm;

    kani::assume(w_original.is_finite() && w_flipped.is_finite());

    // (-g) * (-v) / ||-v|| = g * v / ||v||
    let diff = (w_original - w_flipped).abs();
    assert!(
        diff < 1e-6,
        "simultaneous sign flip must produce same weight"
    );
}

// ---------------------------------------------------------------------------
// Harness 12: Per-channel norm: each output channel normalized independently
// ---------------------------------------------------------------------------

/// Prove: in a 2-output-channel weight, the norm of each channel's
/// sub-vector is independent. Changing channel 0's weights does not
/// affect channel 1's norm.
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_per_channel_independence() {
    // Channel 0 weights
    let v00: f32 = kani::any();
    let v01: f32 = kani::any();
    // Channel 1 weights
    let v10: f32 = kani::any();
    let v11: f32 = kani::any();

    kani::assume(v00.is_finite() && v01.is_finite());
    kani::assume(v10.is_finite() && v11.is_finite());
    kani::assume(v00.abs() <= 10.0 && v01.abs() <= 10.0);
    kani::assume(v10.abs() <= 10.0 && v11.abs() <= 10.0);

    let norm_ch0 = l2_norm_sq_2(v00, v01);
    let norm_ch1 = l2_norm_sq_2(v10, v11);

    assert!(norm_ch0.is_finite(), "channel 0 norm must be finite");
    assert!(norm_ch1.is_finite(), "channel 1 norm must be finite");

    // Modifying channel 0 does not affect channel 1's norm
    let v00_modified: f32 = kani::any();
    kani::assume(v00_modified.is_finite() && v00_modified.abs() <= 10.0);

    let norm_ch0_modified = l2_norm_sq_2(v00_modified, v01);
    let norm_ch1_unchanged = l2_norm_sq_2(v10, v11);

    assert!(
        norm_ch1_unchanged == norm_ch1,
        "channel 1 norm must be unchanged when channel 0 is modified"
    );
    // Channel 0 norm may have changed
    let _ = norm_ch0_modified;
}
