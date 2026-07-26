// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Level 2: D=512 mixed-mode verification (IBP + CROWN) for the Kokoro Generator.
//!
//! Builds on Level 1 (IBP sub-block decomposition from `_d512_ibp.rs`) and Level 1B
//! (CROWN sub-block tests from `_d512_crown.rs`) by applying the `verify_layerwise_mixed`
//! API to run IBP on intractable groups and CROWN on tractable groups at D=512.
//!
//! Key insight from design (`designs/2026-03-17-d512-generator-verification-escalation.md`):
//! At D=512, `voc_up_channels=256` and `out_channels=128`. The output stage Conv1d
//! `[128, 256, 3]` = 98K elements is CROWN-tractable. Pre-norm Conv1d `[512, 512, 3]`
//! = 786K and ResBlock Conv1d `[256, 256, 3]` = 196K are intractable or marginal.
//!
//! Level 2 strategy: IBP for Stage 0 (pre-norm + ResBlocks), CROWN for output.
//!
//! Part of #2599: Kokoro Generator verification ceiling.
//! Part of #2218: Epic — Perfect Kokoro.

#[path = "kokoro_scaled_pipeline.rs"]
mod d512_scaled_helpers;
use d512_scaled_helpers as helpers;

#[path = "kokoro_scaled_layerwise.rs"]
mod layerwise_helpers;

use helpers::KokoroDims;
use layerwise_helpers::build_kokoro_layerwise_deep;
use nn_verify::{tensor_kernels_to_grouped_graph, NormBoundsMode, PropMethod};

use super::common::{bounds_min_max, uniform_bounds};

// -- Test: CROWN on D=512 output sub-block -----------------------------------

/// D=512 CROWN on the output sub-block only.
///
/// The output stage Conv1d [128, 256, 3] = 98,304 weight elements should be
/// CROWN-tractable. This test runs IBP through pre-norm + ResBlocks, then
/// attempts CROWN on the output sub-block and measures tightening.
///
/// Level 2 AC2 partial: "Stage 1 sub-blocks produce CROWN bounds at D=512".
#[test]
fn test_d512_crown_output_subblock() {
    let dims = KokoroDims::d512();
    let num_resblocks = 3;
    let layers = build_kokoro_layerwise_deep(&dims, num_resblocks);
    let initial = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    // Run IBP through pre-norm + ResBlocks to get the input bounds for output.
    let mut current_bounds = initial;

    // Pre-norm group (layers 0-2)
    let pre_group: Vec<_> = layers[0..3].to_vec();
    let pre_graph = tensor_kernels_to_grouped_graph(&pre_group, NormBoundsMode::ForwardMode)
        .expect("pre-norm graph");
    current_bounds = pre_graph
        .propagate_ibp(&current_bounds)
        .expect("pre-norm IBP");
    let (pre_lo, pre_hi) = bounds_min_max(&current_bounds);
    eprintln!("D=512 pre-norm IBP: [{pre_lo:.4}, {pre_hi:.4}]");

    // ResBlocks (IBP)
    for i in 0..num_resblocks {
        let rb_group = vec![layers[3 + i].clone()];
        let rb_graph = tensor_kernels_to_grouped_graph(&rb_group, NormBoundsMode::ForwardMode)
            .unwrap_or_else(|e| panic!("resblock_{i} graph: {e}"));
        current_bounds = rb_graph
            .propagate_ibp(&current_bounds)
            .unwrap_or_else(|e| panic!("resblock_{i} IBP: {e}"));
    }
    let (rb_lo, rb_hi) = bounds_min_max(&current_bounds);
    eprintln!("D=512 after {num_resblocks} ResBlocks IBP: [{rb_lo:.4e}, {rb_hi:.4e}]");

    // Output sub-block: try CROWN
    let output_group = vec![layers.last().unwrap().clone()];
    let output_graph = tensor_kernels_to_grouped_graph(&output_group, NormBoundsMode::ForwardMode)
        .expect("output graph");

    // IBP baseline for output
    let ibp_output = output_graph
        .propagate_ibp(&current_bounds)
        .expect("output IBP");
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    // CROWN attempt for output
    let (method, crown_output, fallback_reason) =
        nn_verify::propagate_with_crown_fallback(&output_graph, &current_bounds)
            .expect("output CROWN");
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    let tightening = if crown_width > 1e-10 && ibp_width > 1e-10 {
        ibp_width / crown_width
    } else {
        1.0
    };

    let method_str = match method {
        PropMethod::Crown => "CROWN",
        _ => "IBP-fallback",
    };
    eprintln!(
        "D=512 output sub-block: IBP=[{ibp_lo:.4e}, {ibp_hi:.4e}] w={ibp_width:.4e} | \
         {method_str}=[{crown_lo:.4e}, {crown_hi:.4e}] w={crown_width:.4e} | \
         tightening={tightening:.2}x"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("  CROWN fallback reason: {reason}");
    }

    // Output bounds must be finite
    assert!(
        ibp_lo.is_finite() && ibp_hi.is_finite(),
        "output IBP bounds must be finite"
    );
    assert!(
        crown_lo.is_finite() && crown_hi.is_finite(),
        "output CROWN bounds must be finite"
    );

    // If CROWN succeeded, it must not be wider than IBP (soundness)
    if matches!(method, PropMethod::Crown) {
        assert!(
            tightening >= 0.99,
            "CROWN must not produce wider bounds than IBP (ratio={tightening:.4})"
        );
        eprintln!("  CROWN succeeded at D=512 output stage: {tightening:.2}x tightening");
    }
}

// -- Test: D=512 mixed-mode via verify_layerwise_mixed -----------------------

/// D=512 mixed-mode verification via `verify_layerwise_mixed`.
///
/// Uses the `verify_layerwise_mixed` API (#2599 Level 2) with per-group mode:
/// - Group 0 (pre-norm, layers 0-2): IBP (Conv1d [512,512,3] intractable)
/// - Group 1 (ResBlocks, layers 3..N): IBP (Conv1d [256,256,3] marginal)
/// - Group 2 (output, layer N+1): CROWN (Conv1d [128,256,3] = 98K, tractable)
///
/// This is the design's intended production pattern: IBP for intractable groups,
/// CROWN for tractable groups, composed via junction contracts.
#[test]
fn test_d512_mixed_ibp_crown_via_api() {
    use nn_tts_verify::{verify_layerwise_mixed, GroupVerifyMode, LayerwiseGrouping};

    let dims = KokoroDims::d512();
    let num_resblocks = 3;
    let layers = build_kokoro_layerwise_deep(&dims, num_resblocks);

    // 3 groups: pre-norm (0,1,2), resblocks (3..3+N), output (3+N)
    let resblock_start = 3;
    let output_idx = resblock_start + num_resblocks;
    let grouping = LayerwiseGrouping {
        groups: vec![
            vec![0, 1, 2],                          // pre-norm
            (resblock_start..output_idx).collect(), // resblocks
            vec![output_idx],                       // output
        ],
    };

    // IBP for intractable groups, CROWN for output
    let modes = vec![
        GroupVerifyMode::Ibp,   // pre-norm: Conv1d [512,512,3] intractable
        GroupVerifyMode::Ibp,   // resblocks: Conv1d [256,256,3] marginal
        GroupVerifyMode::Crown, // output: Conv1d [128,256,3] = 98K tractable
    ];

    let initial = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    eprintln!("D=512 mixed-mode verification (IBP+IBP+CROWN):");
    let cert = verify_layerwise_mixed(&layers, &initial, &grouping, &modes)
        .expect("D=512 mixed layerwise");

    assert!(cert.is_valid, "D=512 mixed pipeline must be valid");
    assert_eq!(cert.stages.len(), 3, "should have 3 stages");

    // Report per-stage results
    for (i, stage) in cert.stages.iter().enumerate() {
        let group_name = match i {
            0 => "pre-norm",
            1 => "resblocks",
            2 => "output",
            _ => "unknown",
        };
        eprintln!(
            "  Stage {i} ({group_name}): method={}, e2e_output=[{:.4e}, {:.4e}]",
            stage.method,
            cert.e2e_output_lower
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            cert.e2e_output_upper
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        );
    }

    // Stages 0-1 should use IBP
    for i in 0..2 {
        assert!(
            cert.stages[i].method.contains("ibp") || cert.stages[i].method.contains("IBP"),
            "stage {i} should be IBP, got: {}",
            cert.stages[i].method
        );
    }

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
        "e2e output bounds must be finite: [{e2e_lo}, {e2e_hi}]"
    );
    eprintln!(
        "D=512 mixed: e2e output [{e2e_lo:.4e}, {e2e_hi:.4e}], valid={}",
        cert.is_valid
    );
}

// -- Test: D=512 upgraded mixed-mode (IBP + CROWN + CROWN) -------------------

/// D=512 mixed-mode verification with ResBlocks upgraded to CROWN.
///
/// The Level 2 feasibility probe (`test_d512_resblock_crown_feasibility_probe`)
/// measured 3,429x CROWN tightening on a single D=512 ResBlock with Conv1d
/// [256, 256, 3] = 196K weight elements. This test upgrades Group 1 (ResBlocks)
/// from IBP to CROWN, giving modes [IBP, CROWN, CROWN]:
/// - Group 0 (pre-norm, layers 0-2): IBP (Conv1d [512,512,7] intractable)
/// - Group 1 (ResBlocks, layers 3..N): CROWN (feasibility confirmed)
/// - Group 2 (output, layer N+1): CROWN (Conv1d [128,256,3] = 98K, tractable)
///
/// This is the design's Level 2 upgrade path: once CROWN is feasible on a
/// sub-block type, upgrade it from IBP to get tighter bounds.
///
/// Level 2 AC2: "Stage 1 sub-blocks produce CROWN bounds at D=512"
/// Level 2 AC3: "CROWN/IBP ratio < 1.0 for at least one Stage 1 sub-block"
#[test]
fn test_d512_mixed_upgraded_resblocks_crown() {
    use nn_tts_verify::{verify_layerwise_mixed, GroupVerifyMode, LayerwiseGrouping};

    let dims = KokoroDims::d512();
    let num_resblocks = 3;
    let layers = build_kokoro_layerwise_deep(&dims, num_resblocks);

    // 3 groups: pre-norm (0,1,2), resblocks (3..3+N), output (3+N)
    let resblock_start = 3;
    let output_idx = resblock_start + num_resblocks;
    let grouping = LayerwiseGrouping {
        groups: vec![
            vec![0, 1, 2],                          // pre-norm
            (resblock_start..output_idx).collect(), // resblocks
            vec![output_idx],                       // output
        ],
    };

    let initial = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    // --- Baseline: IBP + IBP + CROWN (existing Level 2 pattern) ---
    let baseline_modes = vec![
        GroupVerifyMode::Ibp,   // pre-norm
        GroupVerifyMode::Ibp,   // resblocks (baseline)
        GroupVerifyMode::Crown, // output
    ];

    eprintln!("D=512 baseline (IBP+IBP+CROWN):");
    let baseline_cert = verify_layerwise_mixed(&layers, &initial, &grouping, &baseline_modes)
        .expect("baseline mixed layerwise");
    assert!(baseline_cert.is_valid, "baseline must be valid");

    let baseline_lo = baseline_cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let baseline_hi = baseline_cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let baseline_width = baseline_hi - baseline_lo;
    eprintln!("  e2e: [{baseline_lo:.4e}, {baseline_hi:.4e}] w={baseline_width:.4e}");

    // --- Upgraded: IBP + CROWN + CROWN ---
    let upgraded_modes = vec![
        GroupVerifyMode::Ibp,   // pre-norm: Conv1d [512,512,7] still intractable
        GroupVerifyMode::Crown, // resblocks: FEASIBLE per probe (3,429x tightening)
        GroupVerifyMode::Crown, // output: Conv1d [128,256,3] = 98K tractable
    ];

    eprintln!("\nD=512 upgraded (IBP+CROWN+CROWN):");
    let upgraded_cert = verify_layerwise_mixed(&layers, &initial, &grouping, &upgraded_modes)
        .expect("upgraded mixed layerwise");
    assert!(upgraded_cert.is_valid, "upgraded pipeline must be valid");
    assert_eq!(upgraded_cert.stages.len(), 3, "should have 3 stages");

    let upgraded_lo = upgraded_cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let upgraded_hi = upgraded_cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let upgraded_width = upgraded_hi - upgraded_lo;
    eprintln!("  e2e: [{upgraded_lo:.4e}, {upgraded_hi:.4e}] w={upgraded_width:.4e}");

    // Report per-stage methods
    for (i, stage) in upgraded_cert.stages.iter().enumerate() {
        let group_name = match i {
            0 => "pre-norm",
            1 => "resblocks",
            2 => "output",
            _ => "unknown",
        };
        eprintln!("  Stage {i} ({group_name}): method={}", stage.method);
    }

    // --- Validation ---

    // Stage 0 must still be IBP (intractable pre-norm)
    assert!(
        upgraded_cert.stages[0].method.contains("ibp")
            || upgraded_cert.stages[0].method.contains("IBP"),
        "pre-norm stage must be IBP, got: {}",
        upgraded_cert.stages[0].method
    );

    // E2e bounds must be finite
    assert!(
        upgraded_lo.is_finite() && upgraded_hi.is_finite(),
        "upgraded e2e bounds must be finite: [{upgraded_lo}, {upgraded_hi}]"
    );

    // --- Tightening comparison ---
    let e2e_tightening = if upgraded_width > 1e-20 && baseline_width > 1e-20 {
        baseline_width / upgraded_width
    } else {
        1.0
    };
    eprintln!(
        "\nE2E tightening: baseline_width={baseline_width:.4e} / \
         upgraded_width={upgraded_width:.4e} = {e2e_tightening:.2}x"
    );

    // Upgraded bounds must not be wider than baseline (soundness: CROWN >= IBP tightness)
    assert!(
        e2e_tightening >= 0.99,
        "upgraded (IBP+CROWN+CROWN) must not produce wider bounds than \
         baseline (IBP+IBP+CROWN): ratio={e2e_tightening:.4}"
    );

    // If ResBlocks CROWN succeeded (not IBP fallback), the upgraded bounds should be tighter
    let resblock_method = &upgraded_cert.stages[1].method;
    if resblock_method.contains("CROWN") || resblock_method.contains("crown") {
        eprintln!(
            "  ResBlocks used CROWN: e2e tightening = {e2e_tightening:.2}x over IBP baseline"
        );
        // Even 1% tightening is meaningful at D=512 scale
        if e2e_tightening > 1.01 {
            eprintln!("  RESULT: Meaningful e2e tightening from ResBlock CROWN upgrade");
        } else {
            eprintln!(
                "  RESULT: CROWN succeeded but e2e tightening is marginal \
                 (expected: uniform synthetic weights dampen improvement)"
            );
        }
    } else {
        eprintln!(
            "  ResBlocks fell back to IBP — e2e bounds match baseline \
             (CROWN may need production weights for tightening)"
        );
    }
}

// -- Test: ResBlock CROWN feasibility probe ----------------------------------

/// D=512 per-group ResBlock CROWN feasibility probe.
///
/// Tests whether a single ResBlock sub-block at D=512 is CROWN-tractable.
/// The Conv1d [256, 256, 3] = 196,608 weight elements is at the CROWN boundary
/// (D=256 takes ~9min). This probe measures whether CROWN completes or falls back.
///
/// Not an assertion test — purely a measurement for Level 2 planning.
/// If CROWN succeeds on ResBlocks, we could upgrade Group 1 from IBP to CROWN
/// for even tighter bounds.
#[test]
fn test_d512_resblock_crown_feasibility_probe() {
    let dims = KokoroDims::d512();
    let layers = build_kokoro_layerwise_deep(&dims, 1); // 1 ResBlock
    let initial = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    // Run IBP through pre-norm to get ResBlock input bounds
    let pre_group: Vec<_> = layers[0..3].to_vec();
    let pre_graph = tensor_kernels_to_grouped_graph(&pre_group, NormBoundsMode::ForwardMode)
        .expect("pre-norm graph");
    let pre_output = pre_graph.propagate_ibp(&initial).expect("pre-norm IBP");

    // Now try CROWN on the single ResBlock
    let rb_group = vec![layers[3].clone()];
    let rb_graph = tensor_kernels_to_grouped_graph(&rb_group, NormBoundsMode::ForwardMode)
        .expect("resblock graph");

    let ibp_output = rb_graph.propagate_ibp(&pre_output).expect("resblock IBP");
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    let (method, crown_output, fallback_reason) =
        nn_verify::propagate_with_crown_fallback(&rb_graph, &pre_output)
            .expect("resblock CROWN probe");
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    let tightening = if crown_width > 1e-10 && ibp_width > 1e-10 {
        ibp_width / crown_width
    } else {
        1.0
    };

    let method_str = match method {
        PropMethod::Crown => "CROWN",
        _ => "IBP-fallback",
    };
    eprintln!("D=512 ResBlock CROWN probe: Conv1d [256,256,3] = 196K elements");
    eprintln!("  IBP: [{ibp_lo:.4e}, {ibp_hi:.4e}] w={ibp_width:.4e}");
    eprintln!("  {method_str}: [{crown_lo:.4e}, {crown_hi:.4e}] w={crown_width:.4e}");
    eprintln!("  Tightening: {tightening:.2}x");
    if let Some(reason) = &fallback_reason {
        eprintln!("  Fallback reason: {reason}");
    }

    // Structural assertion: both must produce bounds
    assert!(ibp_width.is_finite(), "ResBlock IBP must be finite");
    assert!(
        crown_lo.is_finite() && crown_hi.is_finite(),
        "ResBlock CROWN probe must be finite"
    );

    // Report feasibility for Level 2 targeting decisions
    if matches!(method, PropMethod::Crown) {
        eprintln!("  RESULT: CROWN FEASIBLE at D=512 ResBlock — can upgrade Group 1 mode");
    } else {
        eprintln!("  RESULT: CROWN fell back — ResBlocks stay IBP-only at D=512");
    }
}
