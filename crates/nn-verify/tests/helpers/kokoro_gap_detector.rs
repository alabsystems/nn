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
            .filter(|r| !r.has_any_bounds())
            .map(|r| r.stage.name)
            .collect::<Vec<_>>()
    );

    // Gate: bridge stages must have bounds (analytical, IBP, or CROWN).
    let length_reg = report
        .stages
        .iter()
        .find(|r| r.stage.status_key == "kokoro_production_length_regulate")
        .unwrap();
    assert!(
        length_reg.has_any_bounds(),
        "length_regulate should have bounds"
    );

    let harmonic = report
        .stages
        .iter()
        .find(|r| r.stage.status_key == "kokoro_production_harmonic_source")
        .unwrap();
    assert!(
        harmonic.has_any_bounds(),
        "harmonic_source should have bounds"
    );
}
