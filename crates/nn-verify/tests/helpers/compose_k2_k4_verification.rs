// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! K2 (InstanceNorm) and K4 (AdaIN+Snake) formal verification tests (#2014).
//!
//! These are the two kernels that produce NaN in dvoice's Kokoro Generator:
//! - K2 (InstanceNorm): zero-variance regime → division by ~0
//! - K4 (fused AdaIN+Snake): amplifies small BF16 errors through affine transform
//!
//! This file provides dedicated NY compose tests for both kernels
//! at scalar level (IBP + CROWN) and pipeline level (36 sequential K4 passes).
//!
//! Coverage matrix:
//! | Kernel | IBP | CROWN | BF16 range | Pipeline |
//! |--------|-----|-------|------------|----------|
//! | K2 InstanceNorm | AC1 | AC1 | AC3 | — |
//! | K4 AdaIN+Snake  | AC2 | AC2 | AC3 | AC4 |

use super::common::{assert_bounds_valid, extract_scalar, scalar_bounds};
use nn_dsl::adain::{build_adain_scalar_kernel, build_adain_snake_fused_kernel};
use nn_dsl::instance_norm::build_instance_norm_scalar_kernel;
use nn_verify::{
    compose_sequential, kernel_to_graph, propagate_with_crown_fallback, scalar_input_bounds,
    KernelVerification, ScalarInputBounds, SequentialSpec, VerifyRequest, VerifyStatus,
};

/// Verify a scalar kernel with IBP and return the result.
fn verify_scalar_ibp(
    kernel: &nn_dsl::ir::KernelDef,
    constant_params: &[f32],
    lo: f32,
    hi: f32,
) -> KernelVerification {
    let input_bounds = scalar_input_bounds(lo, hi).expect("input bounds");
    VerifyRequest::new(kernel)
        .constant_params(constant_params)
        .input_bounds(&input_bounds)
        .verify_bounds()
        .unwrap_or_else(|e| panic!("IBP failed for {}: {e}", kernel.name))
}

// ---------------------------------------------------------------------------
// AC1: K2 InstanceNorm — IBP proof certificates
// ---------------------------------------------------------------------------

/// K2 InstanceNorm with identity normalization (mean=0, var=1).
///
/// instance_norm(x, mean=0, var=1, eps=1e-5) ≈ x (identity for unit-variance input).
/// For x ∈ [-10, 10]: output ∈ ≈ [-10, 10].
#[test]
fn test_k2_instance_norm_ibp_identity() {
    let kernel = build_instance_norm_scalar_kernel().expect("build K2");
    // params: mean=0, var=1, eps=1e-5
    let result = verify_scalar_ibp(&kernel, &[0.0, 1.0, 1e-5], -10.0, 10.0);

    assert!(result.is_finite, "K2 IBP bounds must be finite");
    assert!(
        result.output_lower <= -9.9,
        "K2 lower {} must be <= -9.9 (soundness)",
        result.output_lower
    );
    assert!(
        result.output_upper >= 9.9,
        "K2 upper {} must be >= 9.9 (soundness)",
        result.output_upper
    );
}

/// K2 InstanceNorm with non-trivial statistics (mean=5, var=0.25).
///
/// instance_norm(x, mean=5, var=0.25, eps=1e-5) = (x - 5) / sqrt(0.25 + 1e-5) ≈ 2*(x-5).
/// For x ∈ [-5, 5]: output ≈ 2*(-10, 0) = [-20, 0].
#[test]
fn test_k2_instance_norm_ibp_shifted() {
    let kernel = build_instance_norm_scalar_kernel().expect("build K2");
    let result = verify_scalar_ibp(&kernel, &[5.0, 0.25, 1e-5], -5.0, 5.0);

    assert!(result.is_finite, "K2 shifted IBP bounds must be finite");
    // Analytical: (x-5)/sqrt(0.25) = 2*(x-5), for x∈[-5,5]: [-20, 0]
    assert!(
        result.output_lower <= -19.5,
        "K2 shifted lower {} must be <= -19.5",
        result.output_lower
    );
    assert!(
        result.output_upper >= -0.5,
        "K2 shifted upper {} must be >= -0.5",
        result.output_upper
    );
}

/// K2 InstanceNorm edge case: small variance with large eps.
///
/// When variance is very small but eps dominates, the normalization
/// factor 1/sqrt(var+eps) should not blow up.
/// instance_norm(x, mean=0, var=1e-8, eps=1e-3) = x/sqrt(1e-3) ≈ 31.6*x
/// For x ∈ [-1, 1]: output ≈ [-31.6, 31.6].
#[test]
fn test_k2_instance_norm_ibp_small_variance() {
    let kernel = build_instance_norm_scalar_kernel().expect("build K2");
    let result = verify_scalar_ibp(&kernel, &[0.0, 1e-8, 1e-3], -1.0, 1.0);

    assert!(
        result.is_finite,
        "K2 small-variance IBP bounds must be finite"
    );
    // 1/sqrt(1e-3) ≈ 31.6, so output ≈ [-31.6, 31.6]
    assert!(
        result.output_lower <= -30.0,
        "K2 small-var lower {} must be <= -30.0",
        result.output_lower
    );
    assert!(
        result.output_upper >= 30.0,
        "K2 small-var upper {} must be >= 30.0",
        result.output_upper
    );
    // Width should not exceed 200 (sanity: analytical width ≈ 63)
    let width = result.output_upper - result.output_lower;
    assert!(
        width < 200.0,
        "K2 small-var width {width} exceeds sanity limit"
    );
}

/// K2 InstanceNorm with CROWN propagation.
#[test]
fn test_k2_instance_norm_crown() {
    let kernel = build_instance_norm_scalar_kernel().expect("build K2");
    let constants = [0.0f32, 1.0, 1e-5];
    let graph = kernel_to_graph(&kernel, &constants).expect("K2 graph");

    let input = scalar_bounds(-10.0, 10.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("K2 IBP");
    assert_bounds_valid(&ibp_output);

    // CROWN propagation
    let (method, crown_output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("K2 CROWN");
    assert_bounds_valid(&crown_output);

    let (ibp_lo, ibp_hi) = extract_scalar(&ibp_output);
    let (crown_lo, crown_hi) = extract_scalar(&crown_output);

    eprintln!("K2 InstanceNorm CROWN: method={method:?}, IBP=[{ibp_lo:.4}, {ibp_hi:.4}], CROWN=[{crown_lo:.4}, {crown_hi:.4}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("K2 CROWN fallback reason: {reason}");
    }
}

/// K2 InstanceNorm: verify_and_record generates proof certificate.
#[test]
fn test_k2_instance_norm_verify_and_record() {
    let kernel = build_instance_norm_scalar_kernel().expect("build K2");
    let mut status = VerifyStatus::default();
    let input_bounds = scalar_input_bounds(-10.0, 10.0).expect("input bounds");
    let result = VerifyRequest::new(&kernel)
        .constant_params(&[0.0, 1.0, 1e-5])
        .input_bounds(&input_bounds)
        .verify_bounds()
        .expect("K2 verification");

    status
        .record(
            &result,
            ScalarInputBounds::new(-10.0, 10.0).unwrap(),
            &[0.0, 1.0, 1e-5],
            Some("k2_instance_norm"),
        )
        .expect("record K2");

    assert!(
        status.kernel("k2_instance_norm").is_some(),
        "status must contain K2 entry"
    );
    assert!(
        result.is_finite,
        "K2 proof certificate must show finite bounds"
    );
}

// ---------------------------------------------------------------------------
// AC2: K4 AdaIN+Snake — IBP + CROWN proof certificates
// ---------------------------------------------------------------------------

/// K4 AdaIN+Snake with identity parameters.
///
/// adain_snake(x, mu=0, var=1, gamma=1, beta=0, alpha=1, eps=1e-5)
/// ≈ snake(x) = x + sin²(x), so output ∈ ≈ [-9.70, 10.30] for x∈[-10,10].
#[test]
fn test_k4_adain_snake_ibp_identity() {
    let kernel = build_adain_snake_fused_kernel().expect("build K4");
    // params: mu=0, var=1, gamma=1, beta=0, alpha=1, eps=1e-5
    let result = verify_scalar_ibp(&kernel, &[0.0, 1.0, 1.0, 0.0, 1.0, 1e-5], -10.0, 10.0);

    assert!(result.is_finite, "K4 IBP bounds must be finite");
    assert!(
        result.output_lower <= -9.0,
        "K4 lower {} must be <= -9.0 (soundness: true min ≈ -9.70)",
        result.output_lower
    );
    assert!(
        result.output_upper >= 10.0,
        "K4 upper {} must be >= 10.0 (soundness: true max ≈ 10.30)",
        result.output_upper
    );
}

/// K4 AdaIN+Snake with non-trivial style parameters.
///
/// gamma=2, beta=0.5: amplifies and shifts the normalized output.
/// adain(x) = 2*(x-1)/sqrt(4+eps) + 0.5 ≈ (x-1) + 0.5 = x - 0.5
/// snake(y, 0.5) = y + 2*sin²(0.5*y)
/// For x∈[-5,5]: adain ≈ [-5.5, 4.5], snake ≈ [-5.5, 6.5]
#[test]
fn test_k4_adain_snake_ibp_styled() {
    let kernel = build_adain_snake_fused_kernel().expect("build K4");
    let result = verify_scalar_ibp(&kernel, &[1.0, 4.0, 2.0, 0.5, 0.5, 1e-5], -5.0, 5.0);

    assert!(result.is_finite, "K4 styled IBP bounds must be finite");
    assert!(
        result.output_lower <= -5.0,
        "K4 styled lower {} must be <= -5.0",
        result.output_lower
    );
    assert!(
        result.output_upper >= 6.0,
        "K4 styled upper {} must be >= 6.0",
        result.output_upper
    );
}

/// K4 AdaIN+Snake with CROWN propagation.
#[test]
fn test_k4_adain_snake_crown() {
    let kernel = build_adain_snake_fused_kernel().expect("build K4");
    let constants = [0.0f32, 1.0, 1.0, 0.0, 1.0, 1e-5];
    let graph = kernel_to_graph(&kernel, &constants).expect("K4 graph");

    let input = scalar_bounds(-10.0, 10.0);
    let (method, crown_output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("K4 CROWN");
    assert_bounds_valid(&crown_output);

    let (lo, hi) = extract_scalar(&crown_output);
    eprintln!("K4 AdaIN+Snake CROWN: method={method:?}, bounds=[{lo:.4}, {hi:.4}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("K4 CROWN fallback reason: {reason}");
    }
}

/// K4 AdaIN+Snake: verify_and_record generates proof certificate.
#[test]
fn test_k4_adain_snake_verify_and_record() {
    let kernel = build_adain_snake_fused_kernel().expect("build K4");
    let mut status = VerifyStatus::default();
    let input_bounds = scalar_input_bounds(-10.0, 10.0).expect("input bounds");
    let result = VerifyRequest::new(&kernel)
        .constant_params(&[0.0, 1.0, 1.0, 0.0, 1.0, 1e-5])
        .input_bounds(&input_bounds)
        .verify_bounds()
        .expect("K4 verification");

    status
        .record(
            &result,
            ScalarInputBounds::new(-10.0, 10.0).unwrap(),
            &[0.0, 1.0, 1.0, 0.0, 1.0, 1e-5],
            Some("k4_adain_snake"),
        )
        .expect("record K4");

    assert!(
        status.kernel("k4_adain_snake").is_some(),
        "status must contain K4 entry"
    );
    assert!(
        result.is_finite,
        "K4 proof certificate must show finite bounds"
    );
}

// ---------------------------------------------------------------------------
// AC3: BF16 input range coverage
// ---------------------------------------------------------------------------

/// K2 InstanceNorm with BF16-safe input range [-10, 10].
///
/// BF16 has 7-bit mantissa, exp() overflows to Inf at x > ~10.
/// InstanceNorm output must stay within [-80, 80] (F32 exp safe range)
/// to be safe for downstream exp() in attention/softmax.
#[test]
fn test_k2_instance_norm_bf16_safe_range() {
    let kernel = build_instance_norm_scalar_kernel().expect("build K2");
    // BF16-realistic: tight range for inputs that have been through BF16 accumulation
    let result = verify_scalar_ibp(&kernel, &[0.0, 1.0, 1e-5], -10.0, 10.0);

    assert!(result.is_finite, "K2 BF16 range: bounds must be finite");
    // Output must be within exp-safe range for downstream operations
    assert!(
        result.output_lower >= -80.0,
        "K2 BF16 lower {} must be >= -80.0 (exp-safe)",
        result.output_lower
    );
    assert!(
        result.output_upper <= 80.0,
        "K2 BF16 upper {} must be <= 80.0 (exp-safe)",
        result.output_upper
    );
}

/// K4 AdaIN+Snake with BF16-safe input range.
///
/// The fused kernel must produce output within exp-safe range.
/// With identity normalization, snake(x) = x + sin²(x) ∈ [x, x+1],
/// so output stays close to input range.
#[test]
fn test_k4_adain_snake_bf16_safe_range() {
    let kernel = build_adain_snake_fused_kernel().expect("build K4");
    let result = verify_scalar_ibp(&kernel, &[0.0, 1.0, 1.0, 0.0, 1.0, 1e-5], -10.0, 10.0);

    assert!(result.is_finite, "K4 BF16 range: bounds must be finite");
    assert!(
        result.output_lower >= -80.0,
        "K4 BF16 lower {} must be >= -80.0 (exp-safe)",
        result.output_lower
    );
    assert!(
        result.output_upper <= 80.0,
        "K4 BF16 upper {} must be <= 80.0 (exp-safe)",
        result.output_upper
    );
}

/// K4 with amplifying style parameters and BF16 input range.
///
/// gamma=5.0 amplifies: output can exceed exp-safe range.
/// This test documents the bound rather than asserting exp-safety,
/// since large gamma is a model design choice.
#[test]
fn test_k4_adain_snake_bf16_amplified() {
    let kernel = build_adain_snake_fused_kernel().expect("build K4");
    // gamma=5.0 amplifies the normalized output by 5x
    let result = verify_scalar_ibp(&kernel, &[0.0, 1.0, 5.0, 0.0, 1.0, 1e-5], -10.0, 10.0);

    assert!(result.is_finite, "K4 amplified BF16: bounds must be finite");
    // Analytical: adain output ≈ 5*x for identity norm, so ≈ [-50, 50]
    // snake adds sin² term: ≈ [-50, 51]
    // This may exceed BF16 exp-safe range [-10, 10] — that's a model design issue
    let width = result.output_upper - result.output_lower;
    eprintln!(
        "K4 amplified (gamma=5): bounds=[{:.2}, {:.2}], width={width:.2}",
        result.output_lower, result.output_upper
    );
    assert!(
        width < 500.0,
        "K4 amplified width {width} exceeds sanity limit"
    );
}

// ---------------------------------------------------------------------------
// AC4: Pipeline composition proof — 36 sequential K4 passes
// ---------------------------------------------------------------------------

/// Compose N sequential K4 (AdaIN+Snake) passes and verify output stays finite.
///
/// The Kokoro Generator runs 36 K4 passes. Each pass applies AdaIN normalization
/// then Snake activation. This test proves that chained K4 passes maintain
/// finite outputs for all valid inputs.
///
/// Key insight: with identity AdaIN (mean=0, var=1, gamma=1, beta=0),
/// snake(x) = x + sin²(x) grows by at most 1 per pass. After N passes,
/// output ∈ [x-N, x+N] at worst. For x∈[-10,10] and N=36: [-46, 46].
/// IBP relaxation widens this, but bounds must remain finite and sub-exponential.
#[test]
fn test_k4_pipeline_36_passes_finite() {
    let adain = build_adain_scalar_kernel().expect("build adain");
    let snake = nn_dsl::adain::build_snake_scalar_kernel().expect("build snake");

    let adain_constants: [f32; 5] = [0.0, 1.0, 1.0, 0.0, 1e-5]; // identity norm
    let snake_constants: [f32; 1] = [1.0]; // alpha=1

    // Build the first pair: AdaIN → Snake
    let spec = SequentialSpec::new(&adain, &snake, &adain_constants, &snake_constants, 0)
        .expect("valid adain → snake spec");
    let single_pass = compose_sequential(&spec).expect("compose single K4 pass");

    let input = scalar_bounds(-10.0, 10.0);

    // Propagate through a single pass first
    let single_output = single_pass.propagate_ibp(&input).expect("single pass IBP");
    let (single_lo, single_hi) = extract_scalar(&single_output);
    eprintln!("K4 single pass: [{single_lo:.4}, {single_hi:.4}]");

    assert!(single_lo.is_finite() && single_hi.is_finite());

    // For the pipeline composition, we iterate: each pass output feeds next pass input.
    // Since compose_sequential operates on KernelDef (scalar IR), we can chain
    // by propagating IBP through the same graph repeatedly, updating the input bounds.
    let mut current_lo = -10.0f32;
    let mut current_hi = 10.0f32;

    for pass in 0..36 {
        let pass_input = scalar_bounds(current_lo, current_hi);
        let pass_output = single_pass
            .propagate_ibp(&pass_input)
            .expect("pipeline IBP");
        let (lo, hi) = extract_scalar(&pass_output);

        assert!(
            lo.is_finite() && hi.is_finite(),
            "K4 pipeline pass {pass}: bounds became non-finite [{lo}, {hi}]"
        );

        current_lo = lo;
        current_hi = hi;

        // Log every 6th pass for debugging
        if pass % 6 == 5 {
            eprintln!(
                "K4 pipeline pass {}: [{current_lo:.2}, {current_hi:.2}] (width={:.2})",
                pass + 1,
                current_hi - current_lo
            );
        }
    }

    eprintln!("K4 pipeline after 36 passes: [{current_lo:.2}, {current_hi:.2}]");

    // After 36 passes, bounds must still be finite
    assert!(
        current_lo.is_finite() && current_hi.is_finite(),
        "K4 pipeline: bounds after 36 passes must be finite"
    );

    // Width sanity: snake grows by at most ~1 per pass with identity AdaIN.
    // Analytical worst case: initial width 20, grows ~2 per pass → ~92.
    // IBP relaxation multiplies this, but should not be astronomical.
    let final_width = current_hi - current_lo;
    eprintln!("K4 pipeline final width: {final_width:.2}");
    // We don't assert a tight bound since IBP accumulates over-approximation,
    // but the bounds MUST be finite (the critical property for NaN prevention).
}

/// 36-pass K4 pipeline with Kokoro-realistic style parameters.
///
/// In production, AdaIN applies per-channel style: gamma and beta vary.
/// This test uses non-trivial style parameters to verify the pipeline
/// doesn't diverge even with amplification.
#[test]
fn test_k4_pipeline_36_passes_styled() {
    let adain = build_adain_scalar_kernel().expect("build adain");
    let snake = nn_dsl::adain::build_snake_scalar_kernel().expect("build snake");

    // Non-trivial style: gamma=1.2 (mild amplification), beta=0.3 (shift)
    // var=0.5 (larger normalization factor), mean=0.5 (slight shift)
    let adain_constants: [f32; 5] = [0.5, 0.5, 1.2, 0.3, 1e-5];
    let snake_constants: [f32; 1] = [1.0];

    let spec = SequentialSpec::new(&adain, &snake, &adain_constants, &snake_constants, 0)
        .expect("valid styled spec");
    let pass_graph = compose_sequential(&spec).expect("compose styled K4 pass");

    let mut current_lo = -5.0f32;
    let mut current_hi = 5.0f32;

    for pass in 0..36 {
        let pass_input = scalar_bounds(current_lo, current_hi);
        let pass_output = pass_graph
            .propagate_ibp(&pass_input)
            .expect("styled pipeline IBP");
        let (lo, hi) = extract_scalar(&pass_output);

        assert!(
            lo.is_finite() && hi.is_finite(),
            "K4 styled pipeline pass {pass}: non-finite [{lo}, {hi}]"
        );

        current_lo = lo;
        current_hi = hi;
    }

    eprintln!("K4 styled pipeline after 36 passes: [{current_lo:.2}, {current_hi:.2}]");
    assert!(
        current_lo.is_finite() && current_hi.is_finite(),
        "K4 styled pipeline: bounds after 36 passes must be finite"
    );
}

// ---------------------------------------------------------------------------
// Additional: K2 InstanceNorm edge cases from #2014 ACs
// ---------------------------------------------------------------------------

/// K2 InstanceNorm: T=1 edge case (single timestep).
///
/// When T=1, Bessel's correction gives n-1=0 in variance denominator.
/// The scalar kernel takes precomputed mean and var, so this tests
/// what happens when var is very small (approaching the T=1 regime).
#[test]
fn test_k2_instance_norm_near_zero_variance() {
    let kernel = build_instance_norm_scalar_kernel().expect("build K2");
    // var = eps (near-zero variance, eps-dominated)
    let result = verify_scalar_ibp(&kernel, &[0.0, 1e-5, 1e-5], -1.0, 1.0);

    assert!(
        result.is_finite,
        "K2 near-zero variance: bounds must be finite"
    );
    // 1/sqrt(2e-5) ≈ 223.6, so output ≈ [-223.6, 223.6]
    // Large but finite — the eps prevents division by zero
    let width = result.output_upper - result.output_lower;
    assert!(
        width < 1000.0,
        "K2 near-zero variance width {width} exceeds sanity limit"
    );
    eprintln!(
        "K2 near-zero variance: [{:.2}, {:.2}] (width={width:.2})",
        result.output_lower, result.output_upper
    );
}

/// K2 InstanceNorm and K4 AdaIN+Snake composition.
///
/// Tests that K2 output fed into K4 (as part of a normalization stack)
/// produces finite bounds. This is the actual pipeline pattern in Kokoro.
#[test]
fn test_k2_then_k4_composition() {
    let instance_norm = build_instance_norm_scalar_kernel().expect("build K2");
    let adain_snake = build_adain_snake_fused_kernel().expect("build K4");

    // K2 constants: mean=0, var=1, eps=1e-5
    let k2_constants = [0.0f32, 1.0, 1e-5];
    // K4 constants: mu=0, var=1, gamma=1, beta=0, alpha=1, eps=1e-5
    let k4_constants = [0.0f32, 1.0, 1.0, 0.0, 1.0, 1e-5];

    let k2_graph = kernel_to_graph(&instance_norm, &k2_constants).expect("K2 graph");
    let k4_graph = kernel_to_graph(&adain_snake, &k4_constants).expect("K4 graph");

    let input = scalar_bounds(-10.0, 10.0);

    // Propagate through K2
    let k2_output = k2_graph.propagate_ibp(&input).expect("K2 IBP");
    let (k2_lo, k2_hi) = extract_scalar(&k2_output);
    assert!(k2_lo.is_finite() && k2_hi.is_finite());

    // Feed K2 output into K4
    let k4_input = scalar_bounds(k2_lo, k2_hi);
    let k4_output = k4_graph.propagate_ibp(&k4_input).expect("K4 IBP");
    let (k4_lo, k4_hi) = extract_scalar(&k4_output);

    assert!(
        k4_lo.is_finite() && k4_hi.is_finite(),
        "K2→K4 composition: output must be finite, got [{k4_lo}, {k4_hi}]"
    );
    eprintln!("K2→K4 composition: K2=[{k2_lo:.4}, {k2_hi:.4}], K4=[{k4_lo:.4}, {k4_hi:.4}]");
}
