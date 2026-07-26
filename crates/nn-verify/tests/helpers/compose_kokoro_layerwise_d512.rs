// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-layer composition of the Kokoro pipeline at D=512 (production scale).
//!
//! **Status (2026-03-25):** D=512 full-pipeline layerwise CROWN is intractable
//! because Conv1d [512,512,3] = 786K weight elements in the TextEncoder makes
//! CROWN backward propagation exceed 20+ minutes per layer. The solution is a
//! **hybrid IBP/CROWN strategy**: IBP for layers with large Conv1d weights,
//! CROWN for layers with tractable weights.
//!
//! Scaling wall (CROWN only):
//! - D=256/S=2: **works** (~9min total, 5 layers)
//! - D=512/S=2: **intractable** (>20min per layer for TextEncoder Conv1d [512,512,3])
//!
//! **Hybrid strategy (IBP + CROWN):**
//! - Layers 0-2 (TextEncoder, VocoderPre, Upsample): **IBP** — Conv1d [512,512,3]
//!   = 786K and Conv1d [256,512,3] = 393K are too large for CROWN backward pass.
//! - Layer 3 (ResBlock): **CROWN** — Conv1d [256,256,3] = 196K is CROWN-tractable
//!   (confirmed by `compose_kokoro_generator_d512_crown::test_d512_resblock_crown_feasibility_probe`).
//! - Layer 4 (VocoderOutput): **CROWN** — Conv1d [128,256,3] = 98K is CROWN-tractable.
//!
//! The hybrid approach uses `verify_layerwise_mixed` from `nn-tts-verify` with
//! per-group `GroupVerifyMode` to apply IBP where CROWN is intractable and CROWN
//! where it provides tighter bounds. This makes D=512 verification **complete**
//! with wider-but-sound IBP bounds for the large Conv1d layers and tighter CROWN
//! bounds for the smaller layers.
//!
//! Architecture decomposition (5 layers):
//! ```text
//!   Layer 0: TextEncoder — Conv1d[512,512,3] + ReLU + Linear  [512, 2] → [512, 2]  (IBP)
//!   Layer 1: VocoderPre — Conv1d[256,512,3] + LeakyReLU       [512, 2] → [256, 2]  (IBP)
//!   Layer 2: VocoderUpsample — ConvTranspose1d[256,256,4]     [256, 2] → [256, 4]  (IBP)
//!   Layer 3: VocoderResBlock — InstNorm+Snake+Conv1d[256,256,3] [256, 4] → [256, 4] (CROWN)
//!   Layer 4: VocoderOutput — LeakyReLU+Conv1d[128,256,3]+Exp  [256, 4] → [128, 4]  (CROWN)
//! ```
//!
//! Part of #1741: THE MOONSHOT — production-scale D=512 verification.
//! Part of #2576: D=512 CROWN intractability — hybrid IBP/CROWN solution.

#[path = "kokoro_scaled_pipeline.rs"]
mod d512_scaled_helpers;
// Alias needed: kokoro_scaled_layerwise.rs references `super::helpers::KokoroDims`.
use d512_scaled_helpers as helpers;

#[path = "kokoro_scaled_layerwise.rs"]
mod layerwise_helpers;

use super::common::kokoro_weights::uniform_bt;
use super::common::{assert_bounds_valid, bounds_min_max, uniform_bounds};
use d512_scaled_helpers::KokoroDims;
use layerwise_helpers::build_kokoro_layerwise;

// ===========================================================================
// D=512 graph construction — validates that the pipeline CAN be built.
// ===========================================================================

/// D=512 layerwise: graph construction succeeds at production scale.
///
/// This validates that the pipeline decomposition, weight dimensions, and
/// junction compatibility work at D=512.
#[test]
fn test_kokoro_layerwise_d512_graph_construction() {
    let dims = KokoroDims::d512();
    let layers = build_kokoro_layerwise(&dims);

    assert_eq!(layers.len(), 5, "D=512 should decompose into 5 layers");

    let e2e_output_len = dims.out_channels * dims.time_up();
    eprintln!(
        "D=512 graph construction: 5 layers, {} output dims ({}x{})",
        e2e_output_len,
        dims.out_channels,
        dims.time_up()
    );

    // Validate that initial bounds can be created at D=512 scale
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let (lo, hi) = initial.lower_upper();
    assert_eq!(
        lo.len(),
        dims.d_model * dims.seq_len,
        "initial bounds shape mismatch"
    );
    assert_eq!(
        hi.len(),
        dims.d_model * dims.seq_len,
        "initial bounds shape mismatch"
    );
    eprintln!(
        "D=512 graph: initial bounds created at [{}, {}]",
        dims.d_model, dims.seq_len
    );
}

// ===========================================================================
// D=512 hybrid IBP/CROWN verification — makes D=512 tractable (#2576).
//
// Strategy: IBP for layers with intractable Conv1d weights, CROWN for the rest.
// Uses `verify_layerwise_mixed` with per-group `GroupVerifyMode`.
// ===========================================================================

/// D=512 hybrid verification via `verify_layerwise_mixed`.
///
/// Groups:
/// - Group 0 (layers 0,1,2): IBP — TextEncoder Conv1d [512,512,3] = 786K intractable
/// - Group 1 (layer 3): CROWN — ResBlock Conv1d [256,256,3] = 196K tractable
/// - Group 2 (layer 4): CROWN — Output Conv1d [128,256,3] = 98K tractable
///
/// This proves D=512 verification is tractable with hybrid mode. The bounds
/// will be wider than pure CROWN (due to IBP on layers 0-2) but still sound.
///
/// Part of #2576: D=512 CROWN verification intractable — hybrid solution.
#[test]
fn test_kokoro_layerwise_d512_hybrid_ibp_crown() {
    use nn_tts_verify::{verify_layerwise_mixed, GroupVerifyMode, LayerwiseGrouping};

    let dims = KokoroDims::d512();
    let layers = build_kokoro_layerwise(&dims);
    assert_eq!(layers.len(), 5);

    let grouping = LayerwiseGrouping {
        groups: vec![
            vec![0, 1, 2], // pre-norm: IBP (Conv1d [512,512,3] intractable)
            vec![3],       // ResBlock: CROWN (Conv1d [256,256,3] tractable)
            vec![4],       // output: CROWN (Conv1d [128,256,3] tractable)
        ],
    };

    let modes = vec![
        GroupVerifyMode::Ibp,   // layers 0-2: intractable for CROWN
        GroupVerifyMode::Crown, // layer 3: ResBlock CROWN-tractable
        GroupVerifyMode::Crown, // layer 4: output CROWN-tractable
    ];

    let initial = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    eprintln!("D=512 hybrid verification (IBP + CROWN + CROWN):");
    let cert = verify_layerwise_mixed(&layers, &initial, &grouping, &modes)
        .expect("D=512 hybrid layerwise verification");

    assert!(cert.is_valid, "D=512 hybrid pipeline must be valid");
    assert_eq!(cert.stages.len(), 3, "should have 3 stages");

    // Report per-stage results
    for (i, stage) in cert.stages.iter().enumerate() {
        let group_name = match i {
            0 => "pre-norm (IBP)",
            1 => "resblock (CROWN)",
            2 => "output (CROWN)",
            _ => "unknown",
        };
        eprintln!("  Stage {i} ({group_name}): method={}", stage.method);
    }

    // Stage 0 must be IBP (intractable for CROWN)
    assert!(
        cert.stages[0].method.contains("IBP") || cert.stages[0].method.contains("ibp"),
        "pre-norm stage must be IBP, got: {}",
        cert.stages[0].method
    );

    // End-to-end bounds must be finite
    let e2e_lo = cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let e2e_hi = cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        e2e_lo.is_finite() && e2e_hi.is_finite(),
        "D=512 hybrid e2e output bounds must be finite: [{e2e_lo}, {e2e_hi}]"
    );
    let e2e_width = e2e_hi - e2e_lo;
    eprintln!(
        "D=512 hybrid: e2e output [{e2e_lo:.4e}, {e2e_hi:.4e}] width={e2e_width:.4e}, valid={}",
        cert.is_valid
    );
}

/// D=512 hybrid per-layer IBP propagation — verifies each layer individually.
///
/// Runs IBP through all 5 layers sequentially, chaining output bounds as input
/// bounds for the next layer. This is the IBP-only baseline that the hybrid test
/// improves upon (layers 3-4 get CROWN tightening in the hybrid test).
///
/// Part of #2576.
#[test]
fn test_kokoro_layerwise_d512_per_layer_ibp() {
    let dims = KokoroDims::d512();
    let layers = build_kokoro_layerwise(&dims);
    let initial = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    let layer_names = [
        "TextEncoder",
        "VocoderPre",
        "VocoderUpsample",
        "VocoderResBlock",
        "VocoderOutput",
    ];

    eprintln!("D=512 per-layer IBP propagation:");
    let mut current_bounds = initial;

    for (i, (def, bindings)) in layers.iter().enumerate() {
        let graph = nn_verify::tensor_kernel_to_graph(def, bindings)
            .unwrap_or_else(|e| panic!("layer {i} ({}) graph: {e}", layer_names[i]));
        let output = graph
            .propagate_ibp(&current_bounds)
            .unwrap_or_else(|e| panic!("layer {i} ({}) IBP: {e}", layer_names[i]));

        assert_bounds_valid(&output);
        let (lo, hi) = bounds_min_max(&output);
        let width = hi - lo;
        eprintln!(
            "  Layer {i} ({}): [{lo:.4e}, {hi:.4e}] width={width:.4e}",
            layer_names[i]
        );

        current_bounds = output;
    }

    let (final_lo, final_hi) = bounds_min_max(&current_bounds);
    assert!(
        final_lo.is_finite() && final_hi.is_finite(),
        "D=512 IBP final output must be finite: [{final_lo}, {final_hi}]"
    );
    eprintln!("D=512 per-layer IBP complete: final=[{final_lo:.4e}, {final_hi:.4e}]");
}

/// D=512 hybrid vs IBP-only comparison — measures CROWN tightening at D=512.
///
/// Compares the e2e bounds from:
/// 1. IBP-only: all 5 layers use IBP
/// 2. Hybrid: layers 0-2 IBP, layers 3-4 CROWN
///
/// The hybrid approach should produce bounds at least as tight as IBP-only
/// (CROWN bounds are provably tighter-or-equal to IBP bounds).
///
/// Part of #2576.
#[test]
fn test_kokoro_layerwise_d512_hybrid_vs_ibp_comparison() {
    use nn_tts_verify::{verify_layerwise_mixed, GroupVerifyMode, LayerwiseGrouping};

    let dims = KokoroDims::d512();
    let layers = build_kokoro_layerwise(&dims);
    let initial = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    let grouping = LayerwiseGrouping {
        groups: vec![vec![0, 1, 2], vec![3], vec![4]],
    };

    // IBP-only baseline
    let ibp_modes = vec![
        GroupVerifyMode::Ibp,
        GroupVerifyMode::Ibp,
        GroupVerifyMode::Ibp,
    ];
    let ibp_cert = verify_layerwise_mixed(&layers, &initial, &grouping, &ibp_modes)
        .expect("IBP-only verification");

    let ibp_lo = ibp_cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let ibp_hi = ibp_cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let ibp_width = ibp_hi - ibp_lo;

    // Hybrid: IBP + CROWN + CROWN
    let hybrid_modes = vec![
        GroupVerifyMode::Ibp,
        GroupVerifyMode::Crown,
        GroupVerifyMode::Crown,
    ];
    let hybrid_cert = verify_layerwise_mixed(&layers, &initial, &grouping, &hybrid_modes)
        .expect("hybrid verification");

    let hybrid_lo = hybrid_cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let hybrid_hi = hybrid_cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let hybrid_width = hybrid_hi - hybrid_lo;

    eprintln!("D=512 IBP-only:  e2e=[{ibp_lo:.4e}, {ibp_hi:.4e}] width={ibp_width:.4e}");
    eprintln!("D=512 hybrid:    e2e=[{hybrid_lo:.4e}, {hybrid_hi:.4e}] width={hybrid_width:.4e}");

    // Both must be finite
    assert!(
        ibp_lo.is_finite() && ibp_hi.is_finite(),
        "IBP e2e must be finite"
    );
    assert!(
        hybrid_lo.is_finite() && hybrid_hi.is_finite(),
        "hybrid e2e must be finite"
    );

    // Hybrid bounds must not be wider than IBP (CROWN is at least as tight as IBP)
    let tightening = if hybrid_width > 1e-20 && ibp_width > 1e-20 {
        ibp_width / hybrid_width
    } else {
        1.0
    };
    eprintln!("D=512 tightening ratio (IBP/hybrid): {tightening:.4}x");

    assert!(
        tightening >= 0.99,
        "hybrid must not produce wider bounds than IBP: ratio={tightening:.4}"
    );

    // Report per-stage methods
    for (i, stage) in hybrid_cert.stages.iter().enumerate() {
        eprintln!("  hybrid stage {i}: method={}", stage.method);
    }
}
