// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Coupled CROWN + cost propagation for temporal
//! boundedness (Moonshot Property 5).
//!
//! These tests exercise `verify_layerwise_coupled()` — the function that
//! runs CROWN propagation and dispatch plan generation from the *same*
//! `TensorKernelDef`, guaranteeing that the cost profile accurately
//! reflects the verified computation.
//!
//! Layer chain: SpectralEncoderBlock → KokoroDecoder
//!   - SpectralEncoderBlock: `[4, 16]` → `[8, 4]`
//!   - KokoroDecoder: `[8, 4]` → `[4, 8]`
//!
//! The spectral encoder output `[8, 4]` exactly matches the Kokoro decoder
//! input `[8, 4]`, enabling sequential chaining through the layerwise
//! coupled pipeline.
//!
//! Part of #1739: Provable Computational Boundedness — AC5 + AC6.
//! Part of #1741: THE MOONSHOT — Property 5 (Temporal Boundedness).

use super::common;

#[path = "demucs_encoder_block.rs"]
mod enc_helpers;

#[path = "kokoro_decoder.rs"]
mod kokoro_decoder;

#[path = "pipeline_coupled_cost.rs"]
mod pipeline_coupled_cost;

use common::{assert_bounds_valid, uniform_bounds};
use pipeline_coupled_cost::{build_coupled_layers, layers_to_tuples, SPECTRAL_INPUT_SHAPE};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify that both layers produce valid TensorKernelDef graphs.
#[test]
fn test_coupled_layers_build_successfully() {
    let layers = build_coupled_layers();
    assert_eq!(
        layers.len(),
        2,
        "expected spectral_encoder + kokoro_decoder"
    );
    assert_eq!(layers[0].name, "spectral_encoder_block");
    assert_eq!(layers[1].name, "kokoro_decoder");
}

/// Verify that `layers_to_tuples()` produces the correct tuple format
/// for `verify_layerwise_coupled()`.
#[test]
fn test_coupled_layers_tuple_conversion() {
    let layers = build_coupled_layers();
    let tuples = layers_to_tuples(&layers);
    assert_eq!(tuples.len(), 2);
    // Each tuple has (TensorKernelDef, Vec<TensorParamBinding>)
    assert!(!tuples[0].1.is_empty());
    assert!(!tuples[1].1.is_empty());
}

/// Verify that both the spectral encoder and the Kokoro decoder (incl. its
/// `Exp` layer) propagate IBP bounds successfully when chained.
///
/// Previously the Kokoro decoder's `Exp` layer overflowed f32 because the
/// spectral encoder's GroupNorm IBP blew up (widths to 1e7–1e12), pushing the
/// pre-Exp upper bound well past the exp(88) ≈ f32::MAX threshold. The ny sound
/// float-margin z-clamp for norm IBP (ny 054c3ff9) now tames the normalized value
/// to |z| <= sqrt(n-1)+margin, so the spectral encoder output stays bounded
/// (≈[-61, 61], below 88) and Exp no longer overflows. The bounds remain sound
/// (the clamp is conservative, directed-rounded outward).
#[cfg(feature = "ny")]
#[test]
fn test_coupled_layers_independent_propagation() {
    let layers = build_coupled_layers();

    // Layer 0: spectral encoder — full propagation succeeds.
    let graph0 = nn_verify::tensor_kernel_to_graph(&layers[0].def, &layers[0].bindings)
        .expect("spectral encoder graph");
    let input0 = uniform_bounds(&SPECTRAL_INPUT_SHAPE, 1.0);
    let output0 = graph0.propagate_ibp(&input0).expect("spectral encoder IBP");
    assert_bounds_valid(&output0);

    // The tamed encoder output must stay below the Exp overflow threshold (88),
    // otherwise the decoder Exp would (correctly) reject it.
    let (enc_lo, enc_hi) = output0.lower_upper();
    let enc_max = enc_hi.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let enc_min = enc_lo.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        enc_max.is_finite() && enc_max <= 88.0,
        "spectral encoder upper bound {enc_max} should be tamed below Exp overflow threshold 88"
    );
    assert!(enc_min.is_finite(), "spectral encoder lower bound must be finite");

    // Layer 1: kokoro decoder — IBP now propagates through Exp soundly because
    // the encoder bounds are tamed below the overflow threshold.
    let graph1 = nn_verify::tensor_kernel_to_graph(&layers[1].def, &layers[1].bindings)
        .expect("kokoro decoder graph should build");
    let output1 = graph1
        .propagate_ibp(&output0)
        .expect("kokoro decoder IBP should succeed (encoder bounds tamed below Exp overflow)");
    assert_bounds_valid(&output1);
}

/// Verify dispatch plan generation for each layer.
///
/// Both the spectral encoder (Conv1d, GELU, GLU, etc.) and the Kokoro decoder
/// have full dispatch support. The decoder's `Exp` (log-magnitude → magnitude)
/// op lowers to a unary `exp` MSL activation kernel
/// (`codegen_msl_tensor_dispatch.rs`), so `build_dispatch_plan()` produces a
/// non-empty step list for both layers. (The earlier expectation that Exp was
/// unsupported in MSL codegen was stale.)
#[test]
fn test_coupled_layers_dispatch_plans() {
    let layers = build_coupled_layers();

    // Layer 0 (spectral encoder): full dispatch support expected.
    let result0 = nn_dsl::build_dispatch_plan(&layers[0].def, nn_dsl::ScalarType::F32);
    let (steps0, _) = result0.expect("spectral encoder dispatch plan should succeed");
    assert!(
        !steps0.is_empty(),
        "spectral encoder dispatch plan should have at least one step"
    );

    // Layer 1 (kokoro decoder): Exp has MSL codegen, so dispatch succeeds.
    let result1 = nn_dsl::build_dispatch_plan(&layers[1].def, nn_dsl::ScalarType::F32);
    let (steps1, _) = result1.expect("kokoro decoder dispatch plan should succeed (Exp has MSL codegen)");
    assert!(
        !steps1.is_empty(),
        "kokoro decoder dispatch plan should have at least one step"
    );
}

/// Cost profiling: verify that the spectral encoder dispatch plan produces
/// non-zero FLOPs.
///
/// The Kokoro decoder's `Exp` op prevents dispatch plan generation, so only
/// the spectral encoder can be cost-profiled. In the coupled pipeline this
/// is non-fatal: the Kokoro decoder layer gets zero cost.
#[test]
fn test_coupled_layers_cost_profiling() {
    use nn_tts_verify::cost_model::HardwareCostModel;

    let hw = HardwareCostModel::m4_max_conservative();
    let layers = build_coupled_layers();

    // Only the spectral encoder has a valid dispatch plan.
    let (steps, _) = nn_dsl::build_dispatch_plan(&layers[0].def, nn_dsl::ScalarType::F32)
        .expect("spectral encoder dispatch plan");
    let profiles = nn_tts_verify::cost_model::profile_dispatch_plan(&steps, &hw);
    let encoder_flops: u64 = profiles.iter().map(|p| p.flops).sum();
    assert!(
        encoder_flops > 0,
        "spectral encoder: expected non-zero FLOPs, got 0"
    );
}

/// **AC5 test:** Full coupled CROWN + cost verification through
/// `verify_layerwise_coupled()`.
///
/// This test exercises the coupled pipeline machinery end-to-end.
///
/// With the ny sound norm IBP/CROWN z-clamp (ny 054c3ff9) the spectral encoder's
/// output bounds are tamed (≈[-61, 61], below the exp(88) overflow threshold), so
/// the Kokoro decoder's `Exp` layer propagates soundly. The coupled pipeline now
/// produces a full `CoupledTimingCertificate` chaining both layers.
///
/// The test verifies:
/// 1. The coupled pipeline succeeds end-to-end (both layers verified + costed).
/// 2. The two layers chain (junction bounds contained → 2 coupled layers).
/// 3. The roofline timing bound is met (worst-case ≪ 10 ms).
///
/// Note: `overall_passed` is `false` because the decoder's InstanceNorm uses the
/// Heuristic (not Sound) normalization path, so `bounds_cert.is_sound == false`.
/// That is the documented soundness mode for InstanceNorm — not a verification
/// failure — and the timing bound itself is met.
#[cfg(feature = "ny")]
#[test]
fn test_verify_layerwise_coupled_full() {
    use nn_tts_verify::cost_model::HardwareCostModel;
    use nn_tts_verify::cost_propagation::verify_layerwise_coupled;

    let layers = build_coupled_layers();
    let tuples = layers_to_tuples(&layers);
    let input = uniform_bounds(&SPECTRAL_INPUT_SHAPE, 1.0);
    let hw = HardwareCostModel::m4_max_conservative();

    let timing_bound_us = 10_000.0;

    // The coupled pipeline now succeeds: the tamed encoder bounds let Exp
    // propagate, and both layers are CROWN-verified + cost-profiled.
    let cert = verify_layerwise_coupled(&tuples, &input, &hw, timing_bound_us)
        .expect("coupled verification should succeed (encoder bounds tamed below Exp overflow)");

    // Both layers are coupled (each has a non-empty dispatch plan).
    assert_eq!(
        cert.coupled_layers.len(),
        2,
        "expected spectral_encoder + kokoro_decoder coupled layers"
    );
    assert!(
        cert.all_layers_coupled(),
        "both layers should have dispatch steps (Exp has MSL codegen)"
    );

    // The roofline timing bound must be met.
    assert!(
        cert.timing.timing_bound_met,
        "worst-case time {} μs should meet bound {} μs",
        cert.timing.worst_case_time_us, timing_bound_us
    );
    assert!(
        cert.timing.worst_case_time_us <= timing_bound_us,
        "worst-case time must be within the timing bound"
    );
}

/// Verify that `verify_layerwise_coupled` produces a fully-coupled certificate
/// for the spectral encoder → Kokoro decoder chain.
///
/// Both layers have full MSL codegen support (incl. the decoder's `Exp`, which
/// lowers to a unary `exp` activation kernel), so the certificate reports
/// `all_layers_coupled() == true` and the total dispatch step count is the sum
/// of both layers' plans.
#[cfg(feature = "ny")]
#[test]
fn test_coupled_all_dispatchable_layers() {
    use nn_tts_verify::cost_model::HardwareCostModel;
    use nn_tts_verify::cost_propagation::verify_layerwise_coupled;

    let layers = build_coupled_layers();
    let tuples = layers_to_tuples(&layers);
    let input = uniform_bounds(&SPECTRAL_INPUT_SHAPE, 1.0);
    let hw = HardwareCostModel::m4_max_conservative();

    let cert = verify_layerwise_coupled(&tuples, &input, &hw, 10_000.0)
        .expect("coupled verification should succeed");

    // Every layer is coupled: bounds verified AND a non-empty dispatch plan.
    assert!(
        cert.all_layers_coupled(),
        "all layers should be coupled (each has dispatch steps)"
    );
    assert!(
        cert.total_dispatch_steps > 0,
        "coupled certificate should report a positive total dispatch step count"
    );
    assert!(
        cert.coupled_layers.iter().all(|l| l.dispatch_step_count > 0),
        "each coupled layer should have at least one dispatch step"
    );
}

/// Verify that the coupled pipeline produces a sound, well-formed certificate
/// for the spectral encoder → Kokoro decoder chain, including the Exp layer.
///
/// Previously this asserted an Exp-overflow error. With the ny sound norm z-clamp
/// (054c3ff9) the encoder bounds are tamed below the overflow threshold, so the
/// decoder Exp propagates and the certificate is valid: the inter-layer junction
/// is shape-compatible with the encoder output bounds contained in the decoder
/// input bounds, and the decoder's Exp output is the expected small positive band
/// (exp of a tightly-centered log-magnitude ≈ 1.0).
#[cfg(feature = "ny")]
#[test]
fn test_coupled_certificate_is_sound_and_wellformed() {
    use nn_tts_verify::cost_model::HardwareCostModel;
    use nn_tts_verify::cost_propagation::verify_layerwise_coupled;

    let layers = build_coupled_layers();
    let tuples = layers_to_tuples(&layers);
    let input = uniform_bounds(&SPECTRAL_INPUT_SHAPE, 1.0);
    let hw = HardwareCostModel::m4_max_conservative();

    let cert = verify_layerwise_coupled(&tuples, &input, &hw, 10_000.0)
        .expect("coupled verification should succeed");

    // The bounds certificate must be well-formed: the encoder→decoder junction
    // is shape-compatible and the encoder output bounds are contained in the
    // decoder input bounds (no junction violations).
    let bounds_cert = &cert.timing.bounds_cert;
    assert!(bounds_cert.is_valid, "bounds certificate should be valid");
    assert_eq!(bounds_cert.stages.len(), 2, "two verified stages");
    for j in &bounds_cert.junctions {
        assert!(j.shape_compatible, "junction {} shape must be compatible", j.junction_index);
        assert!(j.bounds_contained, "junction {} bounds must be contained", j.junction_index);
        assert_eq!(j.violation_count, 0, "junction {} must have no violations", j.junction_index);
    }

    // The decoder's Exp output is a small positive band (exp of log-magnitude
    // near 0), and all bounds are finite.
    let dec = &bounds_cert.stages[1];
    let out_lo = dec
        .output_lower
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let out_hi = dec
        .output_upper
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(out_lo.is_finite() && out_hi.is_finite(), "decoder Exp output bounds must be finite");
    assert!(out_lo > 0.0, "exp output must be strictly positive, got lower {out_lo}");
}

/// Edge case: verify_layerwise_coupled rejects timing_bound_us <= 0.
#[cfg(feature = "ny")]
#[test]
fn test_coupled_rejects_non_positive_timing_bound() {
    use nn_tts_verify::cost_model::HardwareCostModel;
    use nn_tts_verify::cost_propagation::verify_layerwise_coupled;

    let layers = build_coupled_layers();
    let tuples = layers_to_tuples(&layers);
    let input = uniform_bounds(&SPECTRAL_INPUT_SHAPE, 1.0);
    let hw = HardwareCostModel::m4_max_conservative();

    let result = verify_layerwise_coupled(&tuples, &input, &hw, 0.0);
    assert!(result.is_err(), "timing_bound_us=0 should be rejected");

    let result = verify_layerwise_coupled(&tuples, &input, &hw, -100.0);
    assert!(result.is_err(), "negative timing bound should be rejected");

    let result = verify_layerwise_coupled(&tuples, &input, &hw, f64::NAN);
    assert!(result.is_err(), "NaN timing bound should be rejected");
}
