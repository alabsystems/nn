// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Parametric attention monotonicity tests.
//!
//! Consolidates 30 `compose_attention_monotonicity_phase*.rs` files (12,314 LOC)
//! into a single parametric test binary. Each experimental configuration from
//! the original phase files is expressed as a `MonotonicityConfig` dispatched
//! through the shared `run_monotonicity_experiment()` harness.
//!
//! # Builder groups
//!
//! The 30 phases use 9+ distinct builder patterns, organized below:
//!
//! - **Group A (phases 3, 5):** Simple attention scores via `build_attention_scores_simple/positional`
//! - **Group B (phases 7-10):** Kokoro ProsodyPredictor-style encoder via `phase7/8/10_builders`
//! - **Group C (phase 6):** Layered composition (inline builders)
//! - **Group D (phase 11):** N-block stacking via `phase11_builders`
//! - **Group E (phase 25):** Deep attention stack via `deep_attention_stack`
//! - **Group F (phases 26-34):** Kokoro decoder / attention decoder pipelines
//! - **Group G (phases 35-38):** Scaled prosody via `kokoro_scaled_pipeline` + `kokoro_prosody_scaled`
//! - **Group H (phases 39-40):** Frontier analysis via `kokoro_attn_scaled`
//! - **Group I (phases 44-46):** Weight-magnitude-aware duration
//! - **Group J (phases 47-49):** Block-count frontier / dampened residuals
//!
//! Design: `designs/archive/2026-03-11-monotonicity-test-parametrization.md`
//! Part of #1916.

// --- Helper module imports (each phase file used a subset of these) ---
// Note: modules referenced by monotonicity_groups_fj.rs via `super::`
// must be `pub(crate)` for child module visibility.

#[allow(dead_code, unreachable_pub)]
#[path = "attention_monotonicity.rs"]
mod attn_helpers;

#[allow(dead_code, unreachable_pub)]
#[path = "phase7_builders.rs"]
mod phase7_builders;

#[allow(dead_code, unreachable_pub)]
#[path = "phase8_builders.rs"]
mod phase8_builders;

#[allow(dead_code, unreachable_pub)]
#[path = "phase10_builders.rs"]
mod phase10_builders;

#[allow(dead_code, unreachable_pub)]
#[path = "phase11_builders.rs"]
mod phase11_builders;

#[allow(dead_code, unreachable_pub)]
#[path = "deep_attention_stack.rs"]
mod deep_stack;

#[allow(dead_code, unreachable_pub)]
#[path = "kokoro_decoder.rs"]
pub(crate) mod kokoro_decoder;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_decoder_pipeline.rs"]
pub(crate) mod attn_decoder_pipeline;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_decoder_deep.rs"]
pub(crate) mod attn_decoder_deep;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_decoder_multi_stage.rs"]
pub(crate) mod attn_decoder_multi_stage;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_decoder_dilated.rs"]
pub(crate) mod attn_decoder_dilated;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_decoder_multi_kernel.rs"]
pub(crate) mod attn_decoder_multi_kernel;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_decoder_noise.rs"]
pub(crate) mod attn_decoder_noise;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_decoder_output.rs"]
pub(crate) mod attn_decoder_output;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_decoder_scaled.rs"]
pub(crate) mod attn_decoder_scaled;

#[allow(dead_code, unreachable_pub)]
#[path = "kokoro_scaled_pipeline.rs"]
pub(crate) mod helpers;

#[allow(dead_code, unreachable_pub)]
#[path = "kokoro_prosody_scaled.rs"]
pub(crate) mod prosody_scaled;

#[allow(dead_code, unreachable_pub)]
#[path = "kokoro_attn_scaled.rs"]
pub(crate) mod kokoro_attn_scaled;

#[allow(dead_code, unreachable_pub)]
#[path = "kokoro_attn_layerwise.rs"]
mod kokoro_attn_layerwise;

#[allow(dead_code, unreachable_pub)]
#[path = "kokoro_prosody_n_blocks.rs"]
pub(crate) mod prosody_n_blocks;

#[allow(dead_code, unreachable_pub)]
#[path = "kokoro_prosody_dampened.rs"]
pub(crate) mod prosody_dampened;

#[allow(dead_code, unreachable_pub)]
#[path = "kokoro_full_pipeline.rs"]
mod kokoro_full;

#[allow(dead_code, unreachable_pub)]
#[path = "monotonicity_groups_fj.rs"]
mod groups_fj;

pub(crate) use super::common;

pub(crate) use super::common::monotonicity::{
    run_experiment_batch, run_monotonicity_experiment, AssertionPattern, MonotonicityConfig,
    PropagationMethod,
};
use super::common::{assert_bounds_valid, uniform_bounds};
pub(crate) use nn_verify::tensor_kernel_to_graph;

// ===========================================================================
// Group A: Simple attention scores (phases 3, 5)
//
// Phase 3: Unrestricted Variable Q — k-scaling does not help because bounds
// on diagonal and off-diagonal scores scale identically.
// Phase 5: Position-aware attention (Q = hidden + PE, K = PE) — PE diagonal
// dominance overcomes Variable perturbation when input_bound is small.
// ===========================================================================

#[test]
fn test_group_a_simple_attention_scores() {
    use attn_helpers::*;

    // Phase 3 configs: input_bound sweep with K-scale=1.0
    let input_bounds = [1.0, 0.5, 0.3, 0.1, 0.05];
    let mut experiments = Vec::new();

    for &ib in &input_bounds {
        let (def, _) = build_attention_scores_simple();
        let bindings = attention_scores_simple_bindings_scaled(1.0);
        experiments.push((
            MonotonicityConfig {
                label: if (ib - 0.1_f32).abs() < 1e-6 {
                    "phase3_ib0.1"
                } else if (ib - 1.0_f32).abs() < 1e-6 {
                    "phase3_ib1.0"
                } else {
                    "phase3_ib_other"
                },
                input_bound: ib,
                input_shape: vec![SEQ_LEN, D_MODEL],
                prop_method: PropagationMethod::CrownFallback,
                assertion: AssertionPattern::Monotonicity {
                    seq_len: SEQ_LEN,
                    enc_len: SEQ_LEN,
                    expect_proven: false,
                    min_margin_floor: None,
                },
            },
            def,
            bindings,
        ));
    }

    run_experiment_batch("Group A: Phase 3 — input bound sweep", &experiments);
}

#[test]
fn test_group_a_positional_attention() {
    use attn_helpers::*;

    // Phase 5: position-aware (Q = hidden + PE, K = PE)
    let pe_scale = 5.0;
    let input_bounds = [1.0, 0.5, 0.3, 0.1, 0.05];
    let mut experiments = Vec::new();

    for &ib in &input_bounds {
        let (def, _) = build_attention_scores_positional();
        let bindings = attention_scores_positional_bindings_scaled(pe_scale);
        experiments.push((
            MonotonicityConfig {
                label: "phase5_positional",
                input_bound: ib,
                input_shape: vec![SEQ_LEN, D_MODEL],
                prop_method: PropagationMethod::CrownFallback,
                assertion: AssertionPattern::Monotonicity {
                    seq_len: SEQ_LEN,
                    enc_len: SEQ_LEN,
                    expect_proven: false,
                    min_margin_floor: None,
                },
            },
            def,
            bindings,
        ));
    }

    run_experiment_batch("Group A: Phase 5 — positional attention", &experiments);
}

// ===========================================================================
// Group B: Kokoro ProsodyPredictor encoder (phases 7-10)
//
// Phase 7: ProsodyPredictor-style encoder (Linear→ReLU→LayerNorm→Linear→ReLU).
// Phase 8: Two architecture variants (A and B) with Conv1d encoder.
// Phase 9: Detailed margin analysis with tighter bounds.
// Phase 10: Two-block stacking analysis.
// ===========================================================================

#[test]
fn test_group_b_prosody_encoder() {
    let mut experiments = Vec::new();

    // Phase 7 Arch A: ProsodyPredictor-inspired (Linear→ReLU→LayerNorm→Linear→ReLU)
    {
        let (def, _) = phase7_builders::build_prosody_inspired_attention_scores();
        let bindings = phase7_builders::prosody_inspired_bindings(0.3, 5.0);
        experiments.push((
            MonotonicityConfig {
                label: "phase7_arch_a",
                input_bound: 1.0,
                input_shape: vec![attn_helpers::SEQ_LEN, attn_helpers::D_MODEL],
                prop_method: PropagationMethod::CrownFallback,
                assertion: AssertionPattern::BoundsValid,
            },
            def,
            bindings,
        ));
    }

    // Phase 7 Arch B: Conv1d encoder
    {
        let (def, _) = phase7_builders::build_conv_block_attention_scores();
        let bindings = phase7_builders::conv_block_bindings(0.3, 5.0);
        experiments.push((
            MonotonicityConfig {
                label: "phase7_arch_b",
                input_bound: 1.0,
                input_shape: vec![attn_helpers::SEQ_LEN, attn_helpers::D_MODEL],
                prop_method: PropagationMethod::CrownFallback,
                assertion: AssertionPattern::BoundsValid,
            },
            def,
            bindings,
        ));
    }

    run_experiment_batch("Group B: Phase 7 — prosody encoder", &experiments);
}

#[test]
fn test_group_b_two_block_stacking() {
    let (def, _) = phase10_builders::build_two_block_prosody_predictor();
    let bindings = phase10_builders::two_block_bindings(0.3, 5.0);
    let config = MonotonicityConfig {
        label: "phase10_two_block",
        input_bound: 1.0,
        input_shape: vec![attn_helpers::SEQ_LEN, attn_helpers::D_MODEL],
        prop_method: PropagationMethod::CrownFallback,
        assertion: AssertionPattern::BoundsValid,
    };
    let result = run_monotonicity_experiment(&config, &def, &bindings);
    assert!(result.bounds_valid, "two-block bounds should be valid");
}

// ===========================================================================
// Group C: Layered composition (phase 6)
//
// Composes an encoder prefix (Linear→ReLU) before attention. ReLU clamps
// negative values to zero, producing tighter CROWN bounds than symmetric
// [-B, B] input. Combined with PE diagonal dominance from Phase 5.
// ===========================================================================

#[test]
fn test_group_c_layered_attention() {
    use attn_helpers::*;
    use nn_dsl::tensor_block_builder::TensorBlockBuilder;
    use ndarray::{ArrayD, IxDyn};

    fn build_layered_attention_scores() -> nn_dsl::tensor_ir::TensorKernelDef {
        let mut b = TensorBlockBuilder::new("attn_scores_layered");
        let raw_input = b.add_input("raw_input", &[SEQ_LEN, D_MODEL]);
        let w_enc = b.add_input("w_enc", &[D_MODEL, D_MODEL]);
        let pe = b.add_input("pe", &[SEQ_LEN, D_MODEL]);
        let k = b.add_input("key", &[SEQ_LEN, D_MODEL]);
        let linear_out = b.add_matmul(raw_input, w_enc, false, None, &[SEQ_LEN, D_MODEL]);
        let hidden = b.add_relu(linear_out, &[SEQ_LEN, D_MODEL]);
        let q = b.add_binary_add(hidden, pe, &[SEQ_LEN, D_MODEL]);
        let scale = 1.0 / (D_MODEL as f32).sqrt();
        let scores = b.add_matmul(q, k, true, Some(scale), &[SEQ_LEN, SEQ_LEN]);
        b.build(scores)
            .expect("valid layered attention scores graph")
    }

    fn build_encoder_weight(d: usize, scale: f32) -> ArrayD<f32> {
        let mut data = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                data[i * d + j] = if i == j { scale } else { scale * 0.1 };
            }
        }
        ArrayD::from_shape_vec(IxDyn(&[d, d]), data).expect("valid shape")
    }

    let enc_scale = 0.3;
    let pe_scale = 5.0;

    let def = build_layered_attention_scores();
    let w_enc = build_encoder_weight(D_MODEL, enc_scale);
    let mut pe = build_sinusoidal_pe(SEQ_LEN, D_MODEL);
    pe.mapv_inplace(|v| v * pe_scale);
    let bindings = vec![
        nn_verify::TensorParamBinding::Variable,
        nn_verify::TensorParamBinding::ConstantTensor(w_enc),
        nn_verify::TensorParamBinding::ConstantTensor(pe.clone()),
        nn_verify::TensorParamBinding::ConstantTensor(pe),
    ];

    let config = MonotonicityConfig {
        label: "phase6_layered",
        input_bound: 1.0,
        input_shape: vec![SEQ_LEN, D_MODEL],
        prop_method: PropagationMethod::CrownFallback,
        assertion: AssertionPattern::Monotonicity {
            seq_len: SEQ_LEN,
            enc_len: SEQ_LEN,
            expect_proven: false,
            min_margin_floor: None,
        },
    };
    let result = run_monotonicity_experiment(&config, &def, &bindings);
    assert!(
        result.bounds_valid,
        "layered attention bounds should be valid"
    );
}

// ===========================================================================
// Group D: N-block stacking (phase 11)
//
// Real Kokoro ProsodyPredictor uses 3 stacked blocks at d_model=512.
// Crossover bound scales with D_MODEL (8→12→16). Depth penalty is additive.
// ===========================================================================

#[test]
fn test_group_d_n_block_stacking() {
    use phase11_builders::*;

    let configs = [
        (
            ProsodyConfig {
                n_blocks: 3,
                seq_len: 4,
                d_model: 8,
            },
            "phase11_3b_d8",
        ),
        (
            ProsodyConfig {
                n_blocks: 3,
                seq_len: 4,
                d_model: 12,
            },
            "phase11_3b_d12",
        ),
        (
            ProsodyConfig {
                n_blocks: 3,
                seq_len: 4,
                d_model: 16,
            },
            "phase11_3b_d16",
        ),
    ];

    for (cfg, label) in &configs {
        let (def, _) = build_n_block_prosody_predictor(cfg);
        let bindings = n_block_bindings(cfg, 0.3, 5.0);
        let config = MonotonicityConfig {
            label,
            input_bound: 0.15,
            input_shape: vec![cfg.seq_len, cfg.d_model],
            prop_method: PropagationMethod::CrownFallback,
            assertion: AssertionPattern::Monotonicity {
                seq_len: cfg.seq_len,
                enc_len: cfg.seq_len,
                expect_proven: false,
                min_margin_floor: None,
            },
        };
        let result = run_monotonicity_experiment(&config, &def, &bindings);
        assert!(result.bounds_valid, "{label}: bounds should be valid");
    }
}

// ===========================================================================
// Group E: Deep attention stack (phase 25)
//
// N-layer attention + FFN stacking measuring margin degradation with depth.
// Graceful degradation → provable for real models with tight input bounds.
// ===========================================================================

#[test]
fn test_group_e_deep_attention_stack() {
    let t_dec = 8;
    let t_enc = 4;
    let d_model = 8;
    let num_heads = 2;
    let ffn_dim = 16;

    for n_layers in [2, 3, 4] {
        let def = deep_stack::build_deep_attention_stack(
            n_layers, t_dec, t_enc, d_model, num_heads, ffn_dim,
        );
        let bindings = deep_stack::deep_stack_bindings(
            n_layers, t_dec, t_enc, d_model, num_heads, ffn_dim, 5.0, 0.001,
        );
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = uniform_bounds(&[t_dec, d_model], 0.3);
        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        eprintln!(
            "Group E: {n_layers}-layer stack, {} nodes, bounds valid",
            graph.num_nodes()
        );
    }
}

// Groups F–J (phases 26-49) extracted to helpers/monotonicity_groups_fj.rs
// via #[path] submodule above. Part of #1948.
