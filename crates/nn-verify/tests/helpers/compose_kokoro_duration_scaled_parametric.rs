// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Parametric Kokoro duration positivity tests — Groups C+D+E.
//!
//! - **Group C (Phase 42):** Scaled D=8..D=128 frontier
//! - **Group D (Phase 43):** Production-scale D=256..D=512
//! - **Group E:** Sensitivity analysis (weight magnitude + input bound)
//!
//! Groups A+B (fixed D=8 and T=4 LSTM) are in
//! `compose_kokoro_duration_parametric.rs`.
//!
//! Design: `designs/archive/2026-03-11-monotonicity-test-parametrization.md`
//! Part of #1981.

// Test helper functions may appear unused when not all groups are active.
#![allow(dead_code)]

#[path = "kokoro_prosody.rs"]
mod kokoro_prosody;

#[path = "kokoro_scaled_pipeline.rs"]
mod duration_scaled_helpers;
// Alias needed: kokoro_prosody_scaled.rs references `super::helpers::KokoroDims`.
use duration_scaled_helpers as helpers;

#[path = "kokoro_prosody_scaled.rs"]
mod prosody_scaled;

#[path = "kokoro_duration_helpers.rs"]
mod duration_helpers;

use duration_helpers::{
    crossover_sweep, print_certificates, run_scaled_proof, run_scaled_proof_single_block,
    run_sensitivity_single_block, run_sensitivity_three_block,
};
use duration_scaled_helpers::KokoroDims;

// ===========================================================================
// Group C: Scaled D=8..D=128 frontier (Phase 42)
//
// Duration positivity across increasing model dimensions. Demonstrates
// CROWN/IBP scaling behavior with the Kokoro ProsodyPredictor architecture.
// ===========================================================================

#[test]
fn test_group_c_d8_baseline() {
    let (lo_min, method, is_proven, _) = run_scaled_proof(&KokoroDims::d8(), 1.0);
    assert!(
        is_proven,
        "D=8 baseline should be proven: lo_min={lo_min:.6} ({method})"
    );
    assert!(
        lo_min > -1.0,
        "D=8 lower_bound {lo_min:.6} should be > -1.0"
    );
}

#[test]
fn test_group_c_dimension_sweep() {
    let dimensions = [
        (KokoroDims::d16(), "d16"),
        (KokoroDims::d32(), "d32"),
        (KokoroDims::d64(), "d64"),
        (KokoroDims::d128(), "d128"),
    ];

    eprintln!("\n=== Group C: Dimension sweep D=16..128 ===");
    for (dims, label) in &dimensions {
        for &ib in &[0.1, 0.5, 1.0] {
            let (lo_min, method, is_proven, is_finite) = run_scaled_proof(dims, ib);
            assert!(
                is_finite,
                "{label} ib={ib}: bounds should be finite, got {lo_min}"
            );
            // Bounds tightness: lo_min > -1e6 prevents vacuously wide IBP (#2594).
            assert!(
                lo_min > -1e6,
                "{label} ib={ib}: lower bound {lo_min} below -1e6 (vacuously wide)"
            );
            eprintln!(
                "  {label:>4} ib={ib:.1}: lo_min={lo_min:>12.6} method={method:>5} proven={is_proven}"
            );
        }
    }
}

#[test]
fn test_group_c_multi_scale_frontier() {
    // D=8 must be provable; larger D may not be.
    let (_, _, d8_proven, _) = run_scaled_proof(&KokoroDims::d8(), 1.0);
    assert!(d8_proven, "D=8 must be proven at ib=1.0");

    // Frontier: collect provability at ib=1.0 across scales.
    let scales: Vec<(KokoroDims, &str)> = vec![
        (KokoroDims::d8(), "d8"),
        (KokoroDims::d16(), "d16"),
        (KokoroDims::d32(), "d32"),
        (KokoroDims::d64(), "d64"),
        (KokoroDims::d128(), "d128"),
    ];
    eprintln!("\n=== Group C: Multi-scale frontier (ib=1.0) ===");
    for (dims, label) in &scales {
        let (lo_min, method, is_proven, is_finite) = run_scaled_proof(dims, 1.0);
        assert!(is_finite, "{label}: bounds not finite");
        assert!(
            lo_min > -1e6,
            "{label}: lower bound {lo_min} below -1e6 (vacuously wide)"
        );
        eprintln!("  {label}: lo_min={lo_min:.6} ({method}) proven={is_proven}");
    }
}

#[test]
fn test_group_c_block_depth_vs_scale() {
    // Compare 1-block vs 3-block across scales.
    eprintln!("\n=== Group C: Block depth vs scale ===");
    for dims_fn in [
        KokoroDims::d8,
        KokoroDims::d16,
        KokoroDims::d32,
        KokoroDims::d64,
    ] {
        let dims = dims_fn();
        let (lo_1b, m1, proven_1b, fin_1b) = run_scaled_proof_single_block(&dims, 1.0);
        let (lo_3b, m3, proven_3b, fin_3b) = run_scaled_proof(&dims, 1.0);
        assert!(fin_1b, "d={}: 1-block bounds not finite", dims.d_model);
        assert!(fin_3b, "d={}: 3-block bounds not finite", dims.d_model);
        eprintln!(
            "  d={:>3}: 1-block lo={lo_1b:>10.6} ({m1}) proven={proven_1b}, \
             3-block lo={lo_3b:>10.6} ({m3}) proven={proven_3b}",
            dims.d_model
        );
    }
}

#[test]
fn test_group_c_d32_crossover_sweep() {
    let (lo, hi) = crossover_sweep(&KokoroDims::d32(), 5.0, 6);
    eprintln!("Group C: D=32 crossover ib ~= [{lo:.4}, {hi:.4}]");
    assert!(lo > 0.0, "crossover lower bound should be positive");
}

#[test]
fn test_group_c_certificates() {
    let dims_list = [
        (KokoroDims::d8(), "d8"),
        (KokoroDims::d16(), "d16"),
        (KokoroDims::d32(), "d32"),
    ];
    print_certificates(&dims_list, "Group C: Duration positivity certificates");
    // d8 at ib=1.0 must be provable (baseline).
    let (_, _, d8_proven, _) = run_scaled_proof(&KokoroDims::d8(), 1.0);
    assert!(d8_proven, "Group C certificates: D=8 must be proven");
}

#[test]
fn test_group_c_input_bound_sensitivity() {
    // 2D grid: dimension x input_bound.
    let dims_list = [
        (KokoroDims::d8(), "d8"),
        (KokoroDims::d16(), "d16"),
        (KokoroDims::d32(), "d32"),
    ];
    let input_bounds = [0.1, 0.3, 0.5, 1.0, 2.0];

    eprintln!("\n=== Group C: Input bound sensitivity ===");
    for (dims, label) in &dims_list {
        for &ib in &input_bounds {
            let (lo_min, method, is_proven, is_finite) = run_scaled_proof(dims, ib);
            assert!(
                is_finite,
                "{label} ib={ib}: bounds not finite, got {lo_min}"
            );
            assert!(
                lo_min > -1e6,
                "{label} ib={ib}: lower bound {lo_min} below -1e6 (vacuously wide)"
            );
            eprintln!("  {label} ib={ib:.1}: lo_min={lo_min:.6} ({method}) proven={is_proven}");
        }
    }
}

// ===========================================================================
// Group D: Production-scale D=256..D=512 (Phase 43)
//
// Extends the frontier to production Kokoro decoder dimensions.
// ===========================================================================

#[test]
fn test_group_d_d256_positivity() {
    let dims = KokoroDims::d256();
    eprintln!("\n=== Group D: D=256 duration positivity ===");
    for &ib in &[0.1, 0.5, 1.0] {
        let (lo_min, method, is_proven, is_finite) = run_scaled_proof(&dims, ib);
        assert!(
            is_finite,
            "D=256 ib={ib}: bounds should be finite, got {lo_min}"
        );
        // Bounds tightness: lo_min > -1e6 prevents vacuously wide IBP (#2594).
        assert!(
            lo_min > -1e6,
            "D=256 ib={ib}: lower bound {lo_min} below -1e6 (vacuously wide)"
        );
        eprintln!("  ib={ib:.1}: lo_min={lo_min:>12.6} ({method}) proven={is_proven}");
    }
    // At ib=0.5, D=256 should be proven.
    let (_, _, proven_05, _) = run_scaled_proof(&dims, 0.5);
    assert!(proven_05, "D=256 should be proven at ib=0.5");
}

#[test]
fn test_group_d_d512_positivity() {
    let dims = KokoroDims::d512();
    eprintln!("\n=== Group D: D=512 duration positivity ===");
    for &ib in &[0.1, 0.5, 1.0] {
        let (lo_min, method, _is_proven, is_finite) = run_scaled_proof(&dims, ib);
        assert!(
            is_finite,
            "D=512 ib={ib}: bounds should be finite, got {lo_min}"
        );
        // Bounds tightness: lo_min > -1e6 prevents vacuously wide IBP (#2594).
        assert!(
            lo_min > -1e6,
            "D=512 ib={ib}: lower bound {lo_min} below -1e6 (vacuously wide)"
        );
        eprintln!("  ib={ib:.1}: lo_min={lo_min:>12.6} ({method})");
    }
}

#[test]
fn test_group_d_full_frontier() {
    // Full frontier D=8..D=512 at ib=1.0. D=8 must prove.
    // Expect monotonic decrease in lo_min with dimension (within 5% tolerance).
    let dims_list: Vec<(KokoroDims, &str)> = vec![
        (KokoroDims::d8(), "d8"),
        (KokoroDims::d16(), "d16"),
        (KokoroDims::d32(), "d32"),
        (KokoroDims::d64(), "d64"),
        (KokoroDims::d128(), "d128"),
        (KokoroDims::d256(), "d256"),
        (KokoroDims::d512(), "d512"),
    ];

    let mut prev_lo: Option<f64> = None;
    eprintln!("\n=== Group D: Full frontier D=8..D=512 ===");
    for (dims, label) in &dims_list {
        let (lo_min, method, is_proven, _) = run_scaled_proof(dims, 1.0);
        eprintln!("  {label}: lo_min={lo_min:.6} ({method}) proven={is_proven}");

        if *label == "d8" {
            assert!(is_proven, "D=8 must be proven at ib=1.0");
        }

        // Check monotonic decrease with 5% tolerance.
        if let Some(prev) = prev_lo {
            assert!(
                lo_min <= prev * 1.05,
                "{label}: lo_min {lo_min:.6} exceeds previous {prev:.6} by more than 5%"
            );
        }
        prev_lo = Some(lo_min);
    }
}

#[test]
fn test_group_d_d256_crossover_sweep() {
    let (lo, hi) = crossover_sweep(&KokoroDims::d256(), 10.0, 6);
    eprintln!("Group D: D=256 crossover ib ~= [{lo:.4}, {hi:.4}]");
    assert!(lo > 0.0, "crossover lower bound should be positive");
    assert!(lo < hi, "crossover range should be valid: {lo} < {hi}");
}

#[test]
fn test_group_d_d512_crossover_sweep() {
    let (lo, hi) = crossover_sweep(&KokoroDims::d512(), 10.0, 6);
    eprintln!("Group D: D=512 crossover ib ~= [{lo:.4}, {hi:.4}]");
    assert!(lo > 0.0, "crossover lower bound should be positive");
    assert!(lo < hi, "crossover range should be valid: {lo} < {hi}");
}

#[test]
fn test_group_d_block_depth_production() {
    // 1-block vs 3-block at production dimensions.
    eprintln!("\n=== Group D: Block depth at production scale ===");
    for dims_fn in [KokoroDims::d128, KokoroDims::d256] {
        let dims = dims_fn();
        let (lo_1b, m1, proven_1b, fin_1b) = run_scaled_proof_single_block(&dims, 1.0);
        let (lo_3b, m3, proven_3b, fin_3b) = run_scaled_proof(&dims, 1.0);
        assert!(fin_1b, "d={}: 1-block bounds not finite", dims.d_model);
        assert!(fin_3b, "d={}: 3-block bounds not finite", dims.d_model);
        eprintln!(
            "  d={:>3}: 1-block lo={lo_1b:>10.6} ({m1}) proven={proven_1b}, \
             3-block lo={lo_3b:>10.6} ({m3}) proven={proven_3b}",
            dims.d_model
        );
    }
}

#[test]
fn test_group_d_certificates_production() {
    let dims_list = [
        (KokoroDims::d128(), "d128"),
        (KokoroDims::d256(), "d256"),
        (KokoroDims::d512(), "d512"),
    ];
    print_certificates(&dims_list, "Group D: Production certificates");
    // All production dimensions must have finite bounds at ib=1.0.
    for (dims, label) in &dims_list {
        let (_, _, _, is_finite) = run_scaled_proof(dims, 1.0);
        assert!(is_finite, "{label}: production bounds not finite");
    }
}

#[test]
fn test_group_d_scaling_law() {
    // Measure lower / D and lower / D^2 scaling.
    let dims_list: Vec<(KokoroDims, &str)> = vec![
        (KokoroDims::d8(), "d8"),
        (KokoroDims::d32(), "d32"),
        (KokoroDims::d128(), "d128"),
        (KokoroDims::d256(), "d256"),
        (KokoroDims::d512(), "d512"),
    ];

    eprintln!("\n=== Group D: Scaling law ===");
    for (dims, label) in &dims_list {
        let (lo_min, method, _, is_finite) = run_scaled_proof(dims, 1.0);
        assert!(is_finite, "{label}: bounds not finite");
        assert!(
            lo_min > -1e6,
            "{label}: lower bound {lo_min} below -1e6 (vacuously wide)"
        );
        let d = dims.d_model as f64;
        eprintln!(
            "  {label}: lo/D={:.6e}, lo/D^2={:.6e} ({method})",
            lo_min / d,
            lo_min / (d * d)
        );
    }
}

// ===========================================================================
// Group E: Sensitivity analysis (ICLR publication)
//
// Weight magnitude and input bound sensitivity for the Kokoro ProsodyPredictor.
// ===========================================================================

#[test]
fn test_group_e_weight_sensitivity_single_block() {
    let weight_mags = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5];
    let ib = 1.0;

    eprintln!("\n=== Group E: Weight sensitivity (single-block, ib={ib}) ===");
    let mut prev_lo: Option<f64> = None;
    for &wm in &weight_mags {
        let (lo_min, method, is_proven) = run_sensitivity_single_block(wm, ib);
        eprintln!("  wm={wm:.3}: lo_min={lo_min:.6} ({method}) proven={is_proven}");
        // Lower bound should decrease with increasing weight magnitude.
        if let Some(prev) = prev_lo {
            assert!(
                lo_min <= prev + 1e-4,
                "wm={wm}: lo_min {lo_min:.6} should not exceed prev {prev:.6}"
            );
        }
        prev_lo = Some(lo_min);
    }
    // Small weights should be proven.
    let (_, _, proven_001) = run_sensitivity_single_block(0.001, ib);
    assert!(proven_001, "wm=0.001 should be proven");
}

#[test]
fn test_group_e_weight_sensitivity_three_blocks() {
    let weight_mags = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5];
    let ib = 1.0;

    eprintln!("\n=== Group E: Weight sensitivity (three-block, ib={ib}) ===");
    let mut prev_lo: Option<f64> = None;
    for &wm in &weight_mags {
        let (lo_min, method, is_proven) = run_sensitivity_three_block(wm, ib);
        eprintln!("  wm={wm:.3}: lo_min={lo_min:.6} ({method}) proven={is_proven}");
        // Lower bound should decrease with increasing weight magnitude.
        if let Some(prev) = prev_lo {
            assert!(
                lo_min <= prev + 1e-4,
                "wm={wm}: lo_min {lo_min:.6} should not exceed prev {prev:.6}"
            );
        }
        prev_lo = Some(lo_min);
    }
    // Smallest weight should be proven.
    let (_, _, proven_001) = run_sensitivity_three_block(0.001, ib);
    assert!(proven_001, "wm=0.001 three-block should be proven");
}

#[test]
fn test_group_e_block_depth_vs_weight() {
    // 1-block vs 3-block at different weight magnitudes.
    let weight_mags = [0.01, 0.05, 0.1];
    let ib = 1.0;

    eprintln!("\n=== Group E: Block depth vs weight interaction ===");
    for &wm in &weight_mags {
        let (lo_1b, m1, p1) = run_sensitivity_single_block(wm, ib);
        let (lo_3b, m3, p3) = run_sensitivity_three_block(wm, ib);
        assert!(lo_1b.is_finite(), "wm={wm}: 1-block bounds not finite");
        assert!(lo_3b.is_finite(), "wm={wm}: 3-block bounds not finite");
        eprintln!(
            "  wm={wm:.2}: 1b lo={lo_1b:.6} ({m1}) proven={p1}, \
             3b lo={lo_3b:.6} ({m3}) proven={p3}"
        );
    }
}

#[test]
fn test_group_e_input_bound_sensitivity_single_block() {
    let input_bounds = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0];
    let wm = 0.01;

    eprintln!("\n=== Group E: Input bound sensitivity (1-block, wm={wm}) ===");
    for &ib in &input_bounds {
        let (lo_min, method, is_proven) = run_sensitivity_single_block(wm, ib);
        eprintln!("  ib={ib:.1}: lo_min={lo_min:.6} ({method}) proven={is_proven}");
    }
    // Small input bound should be proven.
    let (_, _, proven_01) = run_sensitivity_single_block(wm, 0.1);
    assert!(proven_01, "ib=0.1 wm=0.01 should be proven");
}

#[test]
fn test_group_e_input_bound_sensitivity_three_blocks() {
    let input_bounds = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0];
    let wm = 0.01;

    eprintln!("\n=== Group E: Input bound sensitivity (3-block, wm={wm}) ===");
    for &ib in &input_bounds {
        let (lo_min, method, is_proven) = run_sensitivity_three_block(wm, ib);
        assert!(lo_min.is_finite(), "ib={ib}: three-block bounds not finite");
        eprintln!("  ib={ib:.1}: lo_min={lo_min:.6} ({method}) proven={is_proven}");
    }
    // Small input bound should be proven.
    let (_, _, proven_01) = run_sensitivity_three_block(wm, 0.1);
    assert!(proven_01, "ib=0.1 wm=0.01 three-block should be proven");
}

#[test]
fn test_group_e_combined_sensitivity_surface() {
    // 2D grid: weight_mag x input_bound. Count proven configurations.
    let weight_mags = [0.01, 0.05, 0.1];
    let input_bounds = [0.1, 0.5, 1.0, 2.0];
    let mut proven_count = 0;
    let total = weight_mags.len() * input_bounds.len();

    eprintln!("\n=== Group E: Combined sensitivity surface ===");
    for &wm in &weight_mags {
        for &ib in &input_bounds {
            let (lo_min, method, is_proven) = run_sensitivity_single_block(wm, ib);
            if is_proven {
                proven_count += 1;
            }
            eprintln!("  wm={wm:.2} ib={ib:.1}: lo_min={lo_min:.6} ({method}) proven={is_proven}");
        }
    }
    eprintln!("  Proven: {proven_count}/{total}");
    assert!(
        proven_count > 0,
        "at least one configuration should be provable"
    );
}
