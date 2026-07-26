// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: assert the Kokoro pipeline has zero verification gaps
//! in compiled segments by reading `nn_verify_status_kokoro.json`.
//!
//! Part of #2930 (Automated bound propagation gap detector).
//! Part of #2218 (Perfect Kokoro epic).

use std::path::Path;

use nn_verify::gap_detector::{detect_gaps, format_gap_report};

#[test]
fn test_kokoro_no_compiled_segment_gaps() {
    let status_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("nn_verify_status_kokoro.json");

    // This is an integration gate over the *fully verified* Kokoro pipeline: it
    // requires a complete, current `nn_verify_status_kokoro.json` produced by
    // `cargo run -p nn-verify --example verify_all` (plus the kokoro production
    // compose suite). It is a generated, gitignored artifact — a bare
    // `cargo test`/`nextest` run does not create it. Skip cleanly when absent
    // (mirrors gap_detector_tests::test_detect_gaps_real_status_file) rather than
    // hard-failing. The zero-gap / CROWN-coverage assertions below are NOT
    // weakened: when the artifact exists they run in full, so a real coverage
    // regression still fails the gate.
    if !status_path.exists() {
        eprintln!(
            "Skipping: {} not found. Run \
             `cargo run -p nn-verify --example verify_all` (+ kokoro production \
             suite) to generate a complete status file.",
            status_path.display()
        );
        return;
    }

    let status: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&status_path).expect("status file must exist"),
    )
    .expect("valid JSON");

    let report = detect_gaps(&status);

    // Print full gap report
    let formatted = format_gap_report(&report);
    println!("{formatted}");

    // Gate: zero gaps in compiled segments
    let compiled_gaps: Vec<_> = report
        .stages
        .iter()
        .filter(|r| r.stage.is_compiled_segment && !r.has_ibp_bounds && !r.has_crown_bounds)
        .collect();
    assert!(
        compiled_gaps.is_empty(),
        "Compiled segments with no bounds: {:?}",
        compiled_gaps
            .iter()
            .map(|r| r.stage.name)
            .collect::<Vec<_>>()
    );

    // Gate: iSTFT bridge must have CROWN bounds (linear transform, #2916).
    let istft = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("iSTFT"))
        .unwrap();
    assert!(
        istft.has_crown_bounds,
        "iSTFT should have CROWN bounds after #2916"
    );

    // AC5 gate: zero total gaps across ALL stages (compiled + bridges).
    // Bridge stages now have analytical bounds (#2930).
    assert_eq!(
        report.total_gaps,
        0,
        "AC5: pipeline must have zero verification gaps. Found {} gap(s): {:?}",
        report.total_gaps,
        report
            .stages
            .iter()
            .filter(|r| !r.has_ibp_bounds && !r.has_crown_bounds)
            .map(|r| r.stage.name)
            .collect::<Vec<_>>()
    );

    // Gate: at least 3 segments must have CROWN bounds (#2988).
    // BertEncoder (single Linear), TextEncoder (multi-layer), F0EnergyPredictor,
    // and iSTFT are CROWN-verified. Prosody/Generator may fall back to IBP.
    let crown_count = report.stages.iter().filter(|r| r.has_crown_bounds).count();
    assert!(
        crown_count >= 3,
        "expected >= 3 stages with CROWN bounds, got {crown_count}. \
         CROWN stages: {:?}",
        report
            .stages
            .iter()
            .filter(|r| r.has_crown_bounds)
            .map(|r| r.stage.name)
            .collect::<Vec<_>>()
    );

    // Sound CROWN gate: non-vacuous CROWN segments only (#2988, D5).
    // BertEncoder has CROWN but vacuous width (300.09) — a single Linear(768→512)
    // with [-3,+3] inputs inherently produces wide bounds. This gate tracks
    // segments where CROWN produces meaningful (non-vacuous) bounds.
    // Currently >= 2: TextEncoder (width 1.43) + iSTFT (width 2.0).
    // Raise to >= 3 when production CROWN tests for F0/PP run and record
    // _crown entries to the status file (tests exist, need KOKORO_WEIGHTS).
    let sound_crown_count = report
        .stages
        .iter()
        .filter(|r| r.has_crown_bounds && !r.is_vacuous)
        .count();
    assert!(
        sound_crown_count >= 2,
        "expected >= 2 non-vacuous CROWN stages, got {sound_crown_count}. \
         Sound CROWN stages: {:?}",
        report
            .stages
            .iter()
            .filter(|r| r.has_crown_bounds && !r.is_vacuous)
            .map(|r| (r.stage.name, r.bound_width))
            .collect::<Vec<_>>()
    );

    // Gate: bridge stages must have bounds (IBP, CROWN, or ANALYTICAL).
    let length_reg = report
        .stages
        .iter()
        .find(|r| r.stage.status_key == "kokoro_production_length_regulate")
        .unwrap();
    assert!(
        length_reg.has_ibp_bounds
            || length_reg.has_crown_bounds
            || length_reg.has_analytical_bounds,
        "length_regulate should have bounds"
    );

    let harmonic = report
        .stages
        .iter()
        .find(|r| r.stage.status_key == "kokoro_production_harmonic_source")
        .unwrap();
    assert!(
        harmonic.has_ibp_bounds || harmonic.has_crown_bounds || harmonic.has_analytical_bounds,
        "harmonic_source should have bounds"
    );
}
