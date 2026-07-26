// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification for FusedResBlock peephole transform equivalence proofs.
//!
//! Exercises the three peephole transform equivalence proofs from #4311:
//!
//! 1. **FusedResBlock wiring** (Pass 2): residual + shortcut + scale
//! 2. **Style projection absorption** (Pass 3): Linear absorption into ResBlock
//! 3. **Batched style projection** (Pass 4): per-block → batched matmul
//!
//! All three are algebraic identities (affine operations), so IBP is exact
//! and the diff bounds are 0.0.
//!
//! These tests wrap the production verification functions in
//! `nn_verify::resblock_equivalence` and exercise them with Kokoro-realistic
//! dimensions.
//!
//! Part of #4311: Verification gaps for Milestone 1 Kokoro certifying compiler.

use nn_verify::PropMethod;
use nn_verify::{TransformPass, TransformProofBundle, TransformProofEntry};

/// Tolerance for "the difference network proves an exact affine identity"
/// through CROWN's *graph merge* path.
///
/// The two graphs are bit-identical, so `f(x) - g(x)` is the zero function and
/// the certified diff interval is symmetric `[-d, +d]` and always encloses 0
/// (verified by the soundness assertion on the scaled tests) — i.e. the bounds
/// stay SOUND. The residual `d` is the floor of certified f32/f64 rounding error
/// that CROWN now carries through every DAG merge node (the difference network's
/// final `Sub`, plus the residual `Add`).
///
/// Why this is no longer ~1e-30: ny commit 5de589a6 ("fix(soundness): close
/// 15 audited false-proof bugs across CROWN/IBP backward paths") made the
/// `CrownMergeAccumulator` *carry* the per-coefficient merge roundoff
/// (`u·|merged_coeff|`, u = 2^-24 for the f32 path / 2^-53 for the f64
/// accumulate) into the certified coefficient-error matrices instead of
/// silently dropping it; `linear_f64.rs::concretize` then applies it OUTWARD
/// (`sum_l -= Σ_j max(|x_l|,|x_u|)·err`, `sum_u += …`). Previously this error
/// was discarded, so an affine identity cancelled to a denormal and the
/// bound *under-counted* the true reachable f32 roundoff — exactly the
/// false-proof class the audit closed. The new (correct) bound is the
/// honest certified roundoff floor. For the pure-linear wiring/absorption/
/// batched-matmul graphs (Linear / Conv / Add only) that floor is ~1e-13..1e-10
/// here, still 4+ orders tighter than `within_epsilon`'s 1e-6, so it still
/// distinguishes a true zero-diff identity from a merely within-tolerance one.
///
/// Mirrors the canonical `AFFINE_ZERO_TOL` in `nn_verify::resblock_equivalence`'s
/// own unit tests (which were already fixed this way and pass 13/13).
const AFFINE_ZERO_TOL: f32 = 1e-8;

/// Tolerance for the SCALED residual variants `(x + f(x)) · (1/√2)`, which
/// add an elementwise `Mul`-by-constant on top of the affine wiring.
///
/// The `Mul` backward uses the McCormick bilinear envelope (the scale is a
/// graph input with the degenerate interval `[s, s]`, so the envelope is the
/// exact affine `s·x` — no relaxation gap, the diff still encloses 0). But
/// McCormick emits a *non-zero coefficient on the scale input* equal to the
/// pre-scale value `x`/`f(x)`, and because the scalar scale broadcasts to all
/// `C·T` outputs that coefficient is `+=`-accumulated across every output
/// position onto the single scale element. The merge-roundoff term
/// (`u·|merged_coeff|`, ny 5de589a6) is taken on that *broadcast-accumulated*
/// coefficient and then multiplied at concretize by `max(|s_l|,|s_u|)`, so
/// the certified roundoff floor here is ~3e-5 — orders of magnitude above the
/// pure-linear floor, but still a SOUND symmetric band that encloses the true
/// zero diff. The per-test `epsilon` for the scaled cases is widened to 1e-4
/// to match this honest floor (was 1e-6, calibrated against the old behavior
/// that dropped the merge error). Mirrors `AFFINE_SCALED_TOL` in the lib tests.
const AFFINE_SCALED_TOL: f32 = 1e-4;

// ===========================================================================
// Task 1: FusedResBlock wiring equivalence compose tests
// ===========================================================================

/// Kokoro Generator ResBlock: same channels, no shortcut, no scale.
#[test]
fn test_compose_resblock_wiring_generator() {
    // Generator ResBlocks: 512 → 512 (use 32 for test speed).
    let result = nn_verify::resblock_equivalence::verify_resblock_wiring_equivalence(
        32,    // channels_in
        32,    // channels_out
        4,     // time_len
        false, // no shortcut
        None,  // no scale
        0.01,  // weight_mag
        1e-6,  // epsilon
    )
    .expect("generator resblock wiring equivalence");

    // SOUND: the certified diff band must enclose the true diff (0).
    assert!(
        result.diff_lower <= 0.0 && result.diff_upper >= 0.0,
        "diff band must enclose 0 (soundness), got [{}, {}]",
        result.diff_lower,
        result.diff_upper
    );
    assert!(
        result.within_epsilon,
        "generator resblock wiring must be within epsilon"
    );
    assert!(
        result.max_abs_diff < AFFINE_ZERO_TOL,
        "identical wiring topology → diff ~0.0, got {}",
        result.max_abs_diff
    );
    assert!(
        result.diff_lower <= 0.0 && result.diff_upper >= 0.0,
        "diff band must enclose 0 (soundness), got [{}, {}]",
        result.diff_lower, result.diff_upper,
    );
}

/// Kokoro F0 ResBlock (block 0): same channels, no upsample, 1/sqrt(2) scale.
///
/// SCALED variant: the `(x + f(x)) · (1/√2)` McCormick scale path lifts the
/// certified roundoff floor to ~1e-5 (see `AFFINE_SCALED_TOL`), so this test
/// uses the scaled tolerance and asserts the enclose-0 soundness property.
#[test]
fn test_compose_resblock_wiring_f0_block0() {
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    let result = nn_verify::resblock_equivalence::verify_resblock_wiring_equivalence(
        16,
        16,
        4,
        false,
        Some(inv_sqrt2 as f32),
        0.01,
        AFFINE_SCALED_TOL, // McCormick scale path: certified roundoff floor ~3e-5
    )
    .expect("f0 block 0 resblock wiring equivalence");

    // SOUND: the certified diff band must still enclose the true diff (0).
    assert!(
        result.diff_lower <= 0.0 && result.diff_upper >= 0.0,
        "diff band must enclose 0 (soundness), got [{}, {}]",
        result.diff_lower,
        result.diff_upper
    );
    assert!(result.within_epsilon);
    assert!(
        result.max_abs_diff < AFFINE_SCALED_TOL,
        "scaled affine identity diff should be at the certified roundoff floor, got {}",
        result.max_abs_diff
    );
}

/// Kokoro F0 ResBlock (block 1): dim change + shortcut + 1/sqrt(2) scale.
///
/// SCALED variant: same `(. ) · (1/√2)` McCormick scale path as block 0 (plus a
/// conv1x1 shortcut), so it uses the scaled tolerance and asserts enclose-0.
#[test]
fn test_compose_resblock_wiring_f0_block1() {
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    let result = nn_verify::resblock_equivalence::verify_resblock_wiring_equivalence(
        16,
        8,
        4,
        true, // shortcut conv1x1 for dim change
        Some(inv_sqrt2 as f32),
        0.01,
        AFFINE_SCALED_TOL, // McCormick scale path: certified roundoff floor ~3e-5
    )
    .expect("f0 block 1 resblock wiring equivalence");

    // SOUND: the certified diff band must still enclose the true diff (0).
    assert!(
        result.diff_lower <= 0.0 && result.diff_upper >= 0.0,
        "diff band must enclose 0 (soundness), got [{}, {}]",
        result.diff_lower,
        result.diff_upper
    );
    assert!(result.within_epsilon);
    assert!(
        result.max_abs_diff < AFFINE_SCALED_TOL,
        "scaled affine identity diff should be at the certified roundoff floor, got {}",
        result.max_abs_diff
    );
}

// ===========================================================================
// Task 2: Style projection absorption equivalence compose tests
// ===========================================================================

/// Style absorption with Kokoro-production dimensions.
#[test]
fn test_compose_style_absorption_kokoro_dims() {
    let result = nn_verify::resblock_equivalence::verify_style_absorption_equivalence(
        128, // Kokoro style_dim
        256, // channels
        0.01, 1e-6,
    )
    .expect("style absorption kokoro dims");

    // SOUND: the certified diff band must enclose the true diff (0).
    assert!(
        result.diff_lower <= 0.0 && result.diff_upper >= 0.0,
        "diff band must enclose 0 (soundness), got [{}, {}]",
        result.diff_lower,
        result.diff_upper
    );
    assert!(result.within_epsilon);
    assert!(
        result.max_abs_diff < AFFINE_ZERO_TOL,
        "affine identity → diff ~0.0, got {}",
        result.max_abs_diff
    );
    assert!(
        result.diff_lower <= 0.0 && result.diff_upper >= 0.0,
        "diff band must enclose 0 (soundness), got [{}, {}]",
        result.diff_lower, result.diff_upper,
    );
}

/// Style absorption with small dimensions for fast regression.
#[test]
fn test_compose_style_absorption_small() {
    let result =
        nn_verify::resblock_equivalence::verify_style_absorption_equivalence(8, 4, 0.1, 1e-6)
            .expect("style absorption small dims");

    // SOUND: the certified diff band must enclose the true diff (0).
    assert!(
        result.diff_lower <= 0.0 && result.diff_upper >= 0.0,
        "diff band must enclose 0 (soundness), got [{}, {}]",
        result.diff_lower,
        result.diff_upper
    );
    assert!(result.within_epsilon);
    assert!(
        result.max_abs_diff < AFFINE_ZERO_TOL,
        "style absorption (small) diff ~0.0, got {}",
        result.max_abs_diff
    );
    assert!(
        result.diff_lower <= 0.0 && result.diff_upper >= 0.0,
        "diff band must enclose 0 (soundness), got [{}, {}]",
        result.diff_lower, result.diff_upper,
    );
}

// ===========================================================================
// Task 3: Batched style projection equivalence compose tests
// ===========================================================================

/// Batched style with Kokoro generator channel pattern.
#[test]
fn test_compose_batched_style_kokoro_generator() {
    let result = nn_verify::resblock_equivalence::verify_batched_style_equivalence(
        128,
        &[512, 256, 128, 64],
        0.01,
        1e-6,
    )
    .expect("batched style kokoro generator");

    // SOUND: the certified diff band must enclose the true diff (0).
    assert!(
        result.diff_lower <= 0.0 && result.diff_upper >= 0.0,
        "diff band must enclose 0 (soundness), got [{}, {}]",
        result.diff_lower,
        result.diff_upper
    );
    assert!(result.within_epsilon);
    assert!(
        result.max_abs_diff < AFFINE_ZERO_TOL,
        "block-diagonal matmul identity → diff ~0.0, got {}",
        result.max_abs_diff
    );
    assert!(
        result.diff_lower <= 0.0 && result.diff_upper >= 0.0,
        "diff band must enclose 0 (soundness), got [{}, {}]",
        result.diff_lower, result.diff_upper,
    );
}

/// Batched style with F0 channel pattern.
#[test]
fn test_compose_batched_style_f0() {
    let result = nn_verify::resblock_equivalence::verify_batched_style_equivalence(
        128,
        &[256, 256, 128],
        0.01,
        1e-6,
    )
    .expect("batched style f0");

    // SOUND: the certified diff band must enclose the true diff (0).
    assert!(
        result.diff_lower <= 0.0 && result.diff_upper >= 0.0,
        "diff band must enclose 0 (soundness), got [{}, {}]",
        result.diff_lower,
        result.diff_upper
    );
    assert!(result.within_epsilon);
    assert!(
        result.max_abs_diff < AFFINE_ZERO_TOL,
        "batched style (f0) diff ~0.0, got {}",
        result.max_abs_diff
    );
    assert!(
        result.diff_lower <= 0.0 && result.diff_upper >= 0.0,
        "diff band must enclose 0 (soundness), got [{}, {}]",
        result.diff_lower, result.diff_upper,
    );
}

// ===========================================================================
// Task 4: TransformProofBundle integration tests
// ===========================================================================

/// Construct a TransformProofBundle from all three peephole pass proofs.
///
/// This is the integration test for the certificate chain: each verified
/// transform produces a `TransformProofEntry`, and the bundle collects them.
#[test]
fn test_compose_transform_proof_bundle_kokoro() {
    let mut bundle = TransformProofBundle::new("kokoro");
    bundle.set_total_transforms(3);

    // Pass 2: FusedResBlock wiring
    let wiring_result = nn_verify::resblock_equivalence::verify_resblock_wiring_equivalence(
        8, 8, 4, false, None, 0.01, 1e-6,
    )
    .expect("wiring proof");

    bundle.push(TransformProofEntry::new(
        "FusedResBlock wiring (generator)",
        TransformPass::FusedResBlockWiring,
        wiring_result.diff_lower,
        wiring_result.diff_upper,
        1e-6,
        PropMethod::Ibp, // IBP is exact for affine
    ));

    // Pass 3: Style absorption
    let style_result =
        nn_verify::resblock_equivalence::verify_style_absorption_equivalence(128, 256, 0.01, 1e-6)
            .expect("style absorption proof");

    bundle.push(TransformProofEntry::new(
        "Style projection absorption",
        TransformPass::StyleProjectionAbsorption,
        style_result.diff_lower,
        style_result.diff_upper,
        1e-6,
        PropMethod::Ibp,
    ));

    // Pass 4: Batched style
    let batch_result = nn_verify::resblock_equivalence::verify_batched_style_equivalence(
        128,
        &[512, 256, 128, 64],
        0.01,
        1e-6,
    )
    .expect("batched style proof");

    bundle.push(TransformProofEntry::new(
        "Batched style projection",
        TransformPass::BatchedStyleProjection,
        batch_result.diff_lower,
        batch_result.diff_upper,
        1e-6,
        PropMethod::Ibp,
    ));

    // All entries should be proved.
    assert_eq!(bundle.proved_count(), 3, "all 3 transforms must be proved");
    assert!(bundle.all_verified(), "all transforms verified");
    assert_eq!(bundle.unverified_count(), 0, "no unverified transforms");

    // Serialize and deserialize roundtrip.
    let json = bundle.to_json().expect("serialize bundle");
    let deserialized = TransformProofBundle::from_json(&json).expect("deserialize bundle");
    assert_eq!(deserialized.proved_count(), 3);
    assert!(deserialized.all_verified());
    assert_eq!(deserialized.model_name, "kokoro");
}

/// TransformProofEntry with Lean4 proof term attachment.
#[test]
fn test_compose_transform_proof_with_lean4() {
    let entry = TransformProofEntry::new(
        "FusedResBlock wiring",
        TransformPass::FusedResBlockWiring,
        0.0,
        0.0,
        1e-6,
        PropMethod::Ibp,
    )
    .with_lean4_proof(
        "-- Lean4 proof\ntheorem resblock_wiring_equiv : True := trivial".to_string(),
    );

    assert!(entry.is_proved());
    assert!(entry.has_lean4_proof());
    assert_eq!(entry.pass_id, TransformPass::FusedResBlockWiring);
}

// ===========================================================================
// Task 5: generate_kokoro_transform_bundle integration (#4311)
// ===========================================================================

/// End-to-end: generate a Kokoro transform bundle using the convenience API.
#[test]
fn test_compose_generate_kokoro_transform_bundle() {
    use nn_verify::resblock_equivalence::{
        generate_kokoro_transform_bundle, KokoroTransformConfig,
    };

    let config = KokoroTransformConfig::default();
    let bundle =
        generate_kokoro_transform_bundle(&config).expect("generate_kokoro_transform_bundle");

    assert_eq!(bundle.proved_count(), 3);
    assert!(bundle.all_verified());
    assert_eq!(bundle.unverified_count(), 0);

    // Verify specific pass IDs are present.
    let pass_ids: Vec<_> = bundle.entries.iter().map(|e| e.pass_id).collect();
    assert!(pass_ids.contains(&TransformPass::FusedResBlockWiring));
    assert!(pass_ids.contains(&TransformPass::StyleProjectionAbsorption));
    assert!(pass_ids.contains(&TransformPass::BatchedStyleProjection));

    // All three bundle passes are pure-linear (the default-config wiring uses no
    // scale), so each certified diff band must enclose the true diff (0) and its
    // magnitude must sit at the honest pure-linear roundoff floor (< 1e-8).
    for entry in &bundle.entries {
        // SOUND: the certified diff band must enclose the true diff (0).
        assert!(
            entry.diff_lower <= 0.0 && entry.diff_upper >= 0.0,
            "{}: diff band must enclose 0 (soundness), got [{}, {}]",
            entry.transform_name,
            entry.diff_lower,
            entry.diff_upper
        );
        assert!(
            entry.max_abs_diff < AFFINE_ZERO_TOL,
            "{}: expected ~0.0 diff for affine identity, got {}",
            entry.transform_name,
            entry.max_abs_diff
        );
        assert!(
            entry.diff_lower <= 0.0 && entry.diff_upper >= 0.0,
            "diff band must enclose 0 (soundness), got [{}, {}]",
            entry.diff_lower, entry.diff_upper,
        );
    }
}

/// CertifyConfig accepts a TransformProofBundle.
#[test]
fn test_compose_certify_config_with_transform_proofs() {
    use nn_verify::certify::CertifyConfig;
    use nn_verify::resblock_equivalence::{
        generate_kokoro_transform_bundle, KokoroTransformConfig,
    };

    let bundle =
        generate_kokoro_transform_bundle(&KokoroTransformConfig::default()).expect("bundle");
    let config = CertifyConfig::new("kokoro").with_transform_proofs(bundle);

    assert!(config.transform_proofs.is_some());
    let tp = config.transform_proofs.unwrap();
    assert_eq!(tp.proved_count(), 3);
    assert!(tp.all_verified());
}
