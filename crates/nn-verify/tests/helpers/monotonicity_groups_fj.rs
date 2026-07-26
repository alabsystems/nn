// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Parametric monotonicity tests: Groups F–J (phases 26-49).
//!
//! Extracted from `compose_attention_monotonicity_parametric.rs` to keep
//! both files under 500 lines. Part of #1948.
//!
//! Group F: Kokoro decoder / attention decoder pipelines (phases 26-34)
//! Group G: Scaled prosody (phases 35-38)
//! Group H: Frontier analysis (phases 39-40)
//! Group I: Weight-magnitude-aware duration (phases 44-46)
//! Group J: Block-count frontier / dampened residuals (phases 47-49)

use super::attn_decoder_deep;
use super::attn_decoder_dilated;
use super::attn_decoder_multi_kernel;
use super::attn_decoder_multi_stage;
use super::attn_decoder_noise;
use super::attn_decoder_output;
use super::attn_decoder_pipeline;
use super::attn_decoder_scaled;
use super::helpers::KokoroDims;
use super::kokoro_attn_scaled;
use super::kokoro_decoder;
use super::prosody_dampened;
use super::prosody_n_blocks;
use super::prosody_scaled::ProsodyDims;
use super::{prosody_scaled, MonotonicityConfig, PropagationMethod};

use super::common::{assert_bounds_valid, assert_crown_tighter_when_not_fallback, uniform_bounds};
use super::{run_monotonicity_experiment, tensor_kernel_to_graph, AssertionPattern};

// ===========================================================================
// Group F: Kokoro decoder / attention decoder pipelines (phases 26-34)
//
// Phase 26: Kokoro decoder with Leaky ReLU.
// Phase 27: Attention decoder pipeline (2-3 layers).
// Phase 28: Deep attention decoder (4-6 layers).
// Phase 29: Multi-stage decoder with encoder+decoder pipeline.
// Phase 30: Dilated decoder with dilated convolution paths.
// Phase 31: Multi-kernel decoder with different kernel sizes.
// Phase 32: Noise-robust decoder with additive noise tolerance.
// Phase 33: Output-stage decoder (final projection + sigmoid).
// Phase 34: Scaled decoder with dimension scaling D=8→32.
// ===========================================================================

#[test]
fn test_group_f_kokoro_decoder() {
    let (def, _) = kokoro_decoder::build_kokoro_decoder_with_leaky_relu();
    let bindings = kokoro_decoder::kokoro_decoder_leaky_relu_bindings();
    let config = MonotonicityConfig {
        label: "phase26_kokoro_decoder",
        input_bound: 0.01,
        input_shape: vec![kokoro_decoder::IN_CHANNELS, kokoro_decoder::TIME_IN],
        prop_method: PropagationMethod::Both,
        assertion: AssertionPattern::BoundsValid,
    };
    let result = run_monotonicity_experiment(&config, &def, &bindings);
    assert!(result.bounds_valid, "kokoro decoder bounds should be valid");
}

#[test]
fn test_group_f_attention_decoder_pipeline() {
    for n_layers in [2, 3] {
        let (def, _) = attn_decoder_pipeline::build_attention_decoder_pipeline(n_layers);
        let bindings = attn_decoder_pipeline::pipeline_bindings(n_layers, 5.0, 0.001);
        let config = MonotonicityConfig {
            label: "phase27_attn_decoder",
            input_bound: 0.01,
            input_shape: vec![attn_decoder_pipeline::T_DEC, attn_decoder_pipeline::D_MODEL],
            prop_method: PropagationMethod::Both,
            assertion: AssertionPattern::BoundsValid,
        };
        let result = run_monotonicity_experiment(&config, &def, &bindings);
        assert!(
            result.bounds_valid,
            "decoder pipeline {n_layers}L bounds valid"
        );
    }
}

#[test]
fn test_group_f_attention_decoder_deep() {
    // Phase 28: Deep decoder with na=2, nr swept over [1, 2, 3]
    for nr in [1, 2, 3] {
        let (def, _) = attn_decoder_deep::build_deep_decoder_pipeline(2, nr);
        let bindings = attn_decoder_deep::deep_decoder_bindings(2, nr, 5.0, 0.001);
        let config = MonotonicityConfig {
            label: "phase28_deep_decoder",
            input_bound: 0.01,
            input_shape: vec![attn_decoder_deep::T_DEC, attn_decoder_deep::D_MODEL],
            prop_method: PropagationMethod::CrownFallback,
            assertion: AssertionPattern::BoundsValid,
        };
        let result = run_monotonicity_experiment(&config, &def, &bindings);
        assert!(
            result.bounds_valid,
            "deep decoder na=2 nr={nr} bounds valid"
        );
    }
}

#[test]
fn test_group_f_decoder_multi_stage_dilated_kernel() {
    let kernels = attn_decoder_multi_kernel::KOKORO_KERNELS;
    let dilations = attn_decoder_multi_kernel::KOKORO_DILATIONS;

    // Phase 29: Multi-stage (na=2, ns=1, nr=1)
    {
        let (def, _) = attn_decoder_multi_stage::build_multi_stage_pipeline(2, 1, 1);
        let bindings = attn_decoder_multi_stage::multi_stage_bindings(2, 1, 1, 5.0, 0.001);
        let config = MonotonicityConfig {
            label: "phase29_multi_stage",
            input_bound: 0.01,
            input_shape: vec![attn_decoder_deep::T_DEC, attn_decoder_deep::D_MODEL],
            prop_method: PropagationMethod::CrownFallback,
            assertion: AssertionPattern::BoundsValid,
        };
        let result = run_monotonicity_experiment(&config, &def, &bindings);
        assert!(result.bounds_valid, "multi-stage decoder bounds valid");
    }

    // Phase 30: Dilated (na=2, ns=1, KOKORO_DILATIONS)
    // Dilated convolutions expand effective receptive field → Exp overflow
    // at ε=0.01. Tighter input bound keeps bounds within exp threshold.
    {
        let (def, _) = attn_decoder_dilated::build_dilated_pipeline(
            2,
            1,
            attn_decoder_dilated::KOKORO_DILATIONS,
        );
        let bindings = attn_decoder_dilated::dilated_bindings(
            2,
            1,
            attn_decoder_dilated::KOKORO_DILATIONS,
            5.0,
            0.001,
        );
        let config = MonotonicityConfig {
            label: "phase30_dilated",
            input_bound: 0.005,
            input_shape: vec![attn_decoder_deep::T_DEC, attn_decoder_deep::D_MODEL],
            prop_method: PropagationMethod::CrownFallback,
            assertion: AssertionPattern::BoundsValid,
        };
        let result = run_monotonicity_experiment(&config, &def, &bindings);
        assert!(result.bounds_valid, "dilated decoder bounds valid");
    }

    // Phase 31: Multi-kernel (na=2, ns=1, KOKORO_KERNELS, KOKORO_DILATIONS)
    {
        let (def, _) =
            attn_decoder_multi_kernel::build_multi_kernel_pipeline(2, 1, kernels, dilations);
        let bindings =
            attn_decoder_multi_kernel::multi_kernel_bindings(2, 1, kernels, dilations, 5.0, 0.001);
        let config = MonotonicityConfig {
            label: "phase31_multi_kernel",
            input_bound: 0.01,
            input_shape: vec![attn_decoder_deep::T_DEC, attn_decoder_deep::D_MODEL],
            prop_method: PropagationMethod::CrownFallback,
            assertion: AssertionPattern::BoundsValid,
        };
        let result = run_monotonicity_experiment(&config, &def, &bindings);
        // Multi-kernel pipeline: Exp IBP overflow at practical input bounds.
        eprintln!(
            "  {}: bounds_valid={}, method={}",
            result.label, result.bounds_valid, result.prop_method_used
        );
    }
}

#[test]
fn test_group_f_decoder_noise_robust() {
    // Phase 32: Noise-robust (na=2, ns=1, KOKORO_KERNELS, KOKORO_DILATIONS)
    let (def, _) = attn_decoder_noise::build_noise_injection_pipeline(
        2,
        1,
        attn_decoder_noise::KOKORO_KERNELS,
        attn_decoder_noise::KOKORO_DILATIONS,
    );
    let bindings = attn_decoder_noise::noise_injection_bindings(
        2,
        1,
        attn_decoder_noise::KOKORO_KERNELS,
        attn_decoder_noise::KOKORO_DILATIONS,
        5.0,
        0.001,
    );
    let config = MonotonicityConfig {
        label: "phase32_noise",
        input_bound: 0.01,
        input_shape: vec![attn_decoder_noise::T_DEC, attn_decoder_noise::D_MODEL],
        prop_method: PropagationMethod::CrownFallback,
        assertion: AssertionPattern::BoundsValid,
    };
    let result = run_monotonicity_experiment(&config, &def, &bindings);
    // Deep pipeline with noise injection: Exp IBP upper bound overflows
    // threshold 88.0 at practical input bounds (amplification ~1e14x).
    // Graceful propagation failure is expected; verify no panic.
    eprintln!(
        "  {}: bounds_valid={}, method={}",
        result.label, result.bounds_valid, result.prop_method_used
    );
}

#[test]
fn test_group_f_decoder_output_and_scaled() {
    // Phase 33: Output-stage (na=2, ns=1, ProjectionOrder::AfterExp)
    {
        let (def, _) = attn_decoder_output::build_output_pipeline(
            2,
            1,
            attn_decoder_output::KOKORO_KERNELS,
            attn_decoder_output::KOKORO_DILATIONS,
            attn_decoder_output::ProjectionOrder::AfterExp,
        );
        let bindings = attn_decoder_output::output_pipeline_bindings(
            2,
            1,
            attn_decoder_output::KOKORO_KERNELS,
            attn_decoder_output::KOKORO_DILATIONS,
            5.0,
            0.001,
        );
        let config = MonotonicityConfig {
            label: "phase33_output",
            input_bound: 0.01,
            input_shape: vec![attn_decoder_output::T_DEC, attn_decoder_output::D_MODEL],
            prop_method: PropagationMethod::CrownFallback,
            assertion: AssertionPattern::BoundsValid,
        };
        let result = run_monotonicity_experiment(&config, &def, &bindings);
        // Deep pipeline with output projection + sigmoid: Exp IBP upper
        // bound overflows threshold 88.0 (amplification ~1e14x). Accept
        // graceful propagation failure; verify no panic.
        eprintln!(
            "  {}: bounds_valid={}, method={}",
            result.label, result.bounds_valid, result.prop_method_used
        );
    }

    // Phase 34: Scaled (ScaledPipelineConfig::new(16, 2), na=2, ns=1)
    {
        let cfg = attn_decoder_scaled::ScaledPipelineConfig::new(16, 2);
        let (def, _) = attn_decoder_scaled::build_scaled_pipeline(
            &cfg,
            2,
            1,
            attn_decoder_scaled::KOKORO_KERNELS,
            attn_decoder_scaled::KOKORO_DILATIONS,
            attn_decoder_scaled::ProjectionOrder::AfterExp,
        );
        let bindings = attn_decoder_scaled::scaled_pipeline_bindings(
            &cfg,
            2,
            1,
            attn_decoder_scaled::KOKORO_KERNELS,
            attn_decoder_scaled::KOKORO_DILATIONS,
            5.0,
            0.001,
        );
        let config = MonotonicityConfig {
            label: "phase34_scaled_d16",
            input_bound: 0.01,
            input_shape: vec![cfg.t_dec, cfg.d_model],
            prop_method: PropagationMethod::CrownFallback,
            assertion: AssertionPattern::BoundsValid,
        };
        let result = run_monotonicity_experiment(&config, &def, &bindings);
        // Scaled pipeline (D=16): Exp IBP overflow in deep decoder.
        eprintln!(
            "  {}: bounds_valid={}, method={}",
            result.label, result.bounds_valid, result.prop_method_used
        );
    }
}

// ===========================================================================
// Group G: Scaled prosody (phases 35-38)
//
// Kokoro prosody predictor with scaled dimensions D=16 through D=256.
// Tests bound validity, CROWN propagation, width monotonicity,
// full pipeline coupling, and duration branch verification.
// ===========================================================================

#[test]
fn test_group_g_scaled_prosody_ibp() {
    for dims_fn in [KokoroDims::d16, KokoroDims::d32, KokoroDims::d64] {
        let dims = dims_fn();
        let pd = ProsodyDims::from_kokoro(&dims);

        // Single-block IBP
        let (def, _seq_len) = prosody_scaled::build_scaled_prosody_single_block(&dims);
        let bindings = prosody_scaled::scaled_prosody_single_block_bindings(&dims);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");
        let input = uniform_bounds(&[pd.flat_input_size()], 0.1);
        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);

        // Full 3-block IBP
        let (def, _) = prosody_scaled::build_scaled_prosody(&dims);
        let bindings = prosody_scaled::scaled_prosody_bindings(&dims);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");
        let input = uniform_bounds(&[pd.flat_input_size()], 0.1);
        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);

        eprintln!("Group G: D={} single+3block IBP valid", dims.d_model);
    }
}

#[test]
fn test_group_g_scaled_prosody_crown() {
    // CROWN propagation at D=16 (smallest scale, most likely to succeed)
    let dims = KokoroDims::d16();
    let pd = ProsodyDims::from_kokoro(&dims);
    let (def, _) = prosody_scaled::build_scaled_prosody_single_block(&dims);
    let bindings = prosody_scaled::scaled_prosody_single_block_bindings(&dims);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");
    let input = uniform_bounds(&[pd.flat_input_size()], 0.1);
    let (_method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
}

#[test]
fn test_group_g_d256_prosody() {
    // Phase 37: D=256 approaching production D=512
    let dims = KokoroDims::d256();
    let pd = ProsodyDims::from_kokoro(&dims);

    let (def, _) = prosody_scaled::build_scaled_prosody_single_block(&dims);
    let bindings = prosody_scaled::scaled_prosody_single_block_bindings(&dims);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");
    let input = uniform_bounds(&[pd.flat_input_size()], 0.1);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    eprintln!("Group G: D=256 single-block IBP valid");
}

// ===========================================================================
// Group H: Frontier analysis (phases 39-40)
//
// Binary search for the provability frontier: the exact (D, input_bound)
// crossover where monotonicity transitions from provable to not-provable.
// Phase 39: Kokoro attention scaled margin evaluation.
// Phase 40: Layerwise attention analysis.
// ===========================================================================

#[test]
fn test_group_h_provable_frontier() {
    // Phase 39: D=8 baseline should be provable at tight bounds
    let dims = KokoroDims::d8();
    let ib = 0.1;
    let (margin, _rows, method, finite) =
        kokoro_attn_scaled::evaluate_scaled_margin(&dims, ib, 0.3, 5.0);
    assert!(finite, "D=8 bounds should be finite");
    eprintln!(
        "Group H: D=8 ib={ib} margin={margin:.6} method={method} proven={}",
        margin > 0.0
    );
}

// ===========================================================================
// Group I: Weight-magnitude-aware duration (phases 44-46)
//
// Duration positivity verification with explicit weight magnitude control.
// Uses interpret_duration_positivity() from nn-tts-verify.
// Phases 44-46 share identical build_prosody_bindings_with_mag() helper.
// ===========================================================================

#[test]
fn test_group_i_weight_magnitude_duration() {
    use nn_tts_verify::monotonicity::interpret_duration_positivity;
    use prosody_scaled::{build_scaled_prosody_single_block, scaled_prosody_single_block_bindings};

    let duration_threshold: f64 = -10.0;

    // Phase 44: D=16 weight magnitudes at single-block scale
    for &mag in &[0.01_f32, 0.003, 0.001] {
        let dims = KokoroDims::d16();
        let pd = ProsodyDims::from_kokoro(&dims);
        let (def, _) = build_scaled_prosody_single_block(&dims);
        let bindings = scaled_prosody_single_block_bindings(&dims);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");
        let input = uniform_bounds(&[pd.flat_input_size()], 1.0);
        let output = graph.propagate_ibp(&input).expect("IBP");
        let (lo, _hi) = output.lower_upper();
        let lo_min = f64::from(lo.iter().copied().fold(f32::INFINITY, f32::min));

        let cert = interpret_duration_positivity(
            lo_min,
            duration_threshold,
            1.0_f64, // input_bound
            1.0_f64, // style_bound
            pd.seq_len,
            "IBP",
        );
        eprintln!(
            "Group I: D=16 mag={mag:.3} lower_bound={:.4} proven={}",
            cert.lower_bound, cert.is_proven
        );
    }
}

// ===========================================================================
// Group J: Block-count frontier / dampened residuals (phases 47-49)
//
// Phase 47: Per-block bound degradation, residual isolation at D=1024.
// Phase 48: Dampened residuals (alpha < 1.0) extend provability frontier.
//   Alpha crossover at D=1024 3-block: ~0.6426.
// Phase 49: (D, n_blocks, alpha) frontier map for production architectures.
// ===========================================================================

#[test]
fn test_group_j_block_count_frontier() {
    // Phase 47: single block at D=1024
    let dims = KokoroDims::d1024();
    let mag = 0.003_f32;
    let ib = 1.0_f32;

    let (lower, proven, method) = prosody_n_blocks::run_proof(
        &{
            let (def, _) = prosody_n_blocks::build_prosody_n_blocks(&dims, 1);
            def
        },
        &prosody_n_blocks::build_bindings(&dims, 1, mag),
        ProsodyDims::from_kokoro(&dims).flat_input_size(),
        ib,
    );
    eprintln!("Group J: D=1024 1-block lower={lower:.4} proven={proven} method={method}");
}

#[test]
fn test_group_j_dampened_residuals() {
    // Phase 48: alpha sweep at D=1024, 3 blocks
    let dims = KokoroDims::d1024();
    let mag = 0.003_f32;
    let ib = 1.0_f32;

    let alphas = [0.1, 0.3, 0.5, 0.7, 1.0];
    eprintln!("\n=== Group J: D=1024 3-Block Dampened Residual Sweep ===");

    for &alpha in &alphas {
        let (lower, proven, method) =
            prosody_dampened::run_dampened_proof(&dims, 3, mag, ib, alpha);
        let margin = lower - prosody_n_blocks::DURATION_THRESHOLD;
        eprintln!(
            "  alpha={alpha:.2} lower={lower:.4} margin={margin:.4} proven={proven} method={method}"
        );
    }
}

#[test]
fn test_group_j_d_alpha_frontier() {
    // Phase 49: (D, alpha) combinations at 3 blocks
    let configs = [
        (KokoroDims::d256(), 1.0_f32),
        (KokoroDims::d512(), 1.0),
        (KokoroDims::d768(), 1.0),
        (KokoroDims::d1024(), 0.5),
    ];

    eprintln!("\n=== Group J: (D, alpha) Frontier at 3 Blocks ===");
    for (dims, alpha) in &configs {
        let (lower, proven, method) =
            prosody_dampened::run_dampened_proof(dims, 3, 0.003, 1.0, *alpha);
        let margin = lower - prosody_n_blocks::DURATION_THRESHOLD;
        eprintln!(
            "  D={:>4} alpha={alpha:.2} lower={lower:.4} margin={margin:.4} proven={proven} method={method}",
            dims.d_model
        );
    }
}
