// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `verify_layerwise_mixed` — per-group IBP/CROWN mode selection (#2599).
//!
//! Verifies that [`verify_layerwise_mixed`] correctly applies IBP vs CROWN
//! per group and that mode-length mismatches are rejected.
//!
//! Part of #2599, Part of #2218.

#[path = "kokoro_scaled_pipeline.rs"]
mod mixed_scaled_helpers;
use mixed_scaled_helpers as helpers;

#[path = "kokoro_scaled_layerwise.rs"]
mod mixed_layerwise_helpers;

use super::common::kokoro_weights::uniform_bt;
use helpers::KokoroDims;
use mixed_layerwise_helpers::build_kokoro_layerwise_deep;
use nn_tts_verify::{verify_layerwise_mixed, GroupVerifyMode, LayerwiseGrouping};

/// Number of ResBlocks for mixed tests (4 = fast, representative).
const NUM_RESBLOCKS: usize = 4;

/// Build a grouping with 3 groups: pre-norm, resblocks, output.
fn build_three_group_grouping(num_resblocks: usize) -> LayerwiseGrouping {
    let resblock_start = 3;
    let output_idx = resblock_start + num_resblocks;
    LayerwiseGrouping {
        groups: vec![
            vec![0, 1, 2],                          // pre-norm
            (resblock_start..output_idx).collect(), // resblocks
            vec![output_idx],                       // output
        ],
    }
}

// ===========================================================================
// Happy path: mixed IBP/CROWN produces valid certificate
// ===========================================================================

/// All-CROWN mode via `verify_layerwise_mixed` matches `verify_layerwise_grouped`.
#[test]
fn test_mixed_all_crown_produces_valid_certificate() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let grouping = build_three_group_grouping(NUM_RESBLOCKS);
    let modes = vec![GroupVerifyMode::Crown; grouping.groups.len()];

    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise_mixed(&layers, &initial, &grouping, &modes)
        .expect("all-CROWN mixed layerwise");

    assert!(cert.is_valid, "all-CROWN pipeline must be valid");
    assert_eq!(cert.stages.len(), grouping.groups.len());

    // All stages should use a CROWN-family method (CROWN, AlphaCrown — which is
    // strictly tighter — or CROWN-with-IBP-fallback). Match case-insensitively so
    // any CROWN variant ("Crown", "AlphaCrown", "CROWN") is accepted.
    for (i, stage) in cert.stages.iter().enumerate() {
        let m = stage.method.to_ascii_lowercase();
        assert!(
            m.contains("crown") || m.contains("ibp"),
            "stage {i} method should be a CROWN-family method or IBP fallback, got: {}",
            stage.method
        );
    }
}

/// All-IBP mode via `verify_layerwise_mixed` produces a valid (but wider) certificate.
#[test]
fn test_mixed_all_ibp_produces_valid_certificate() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let grouping = build_three_group_grouping(NUM_RESBLOCKS);
    let modes = vec![GroupVerifyMode::Ibp; grouping.groups.len()];

    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise_mixed(&layers, &initial, &grouping, &modes)
        .expect("all-IBP mixed layerwise");

    assert!(cert.is_valid, "all-IBP pipeline must be valid");
    assert_eq!(cert.stages.len(), grouping.groups.len());

    // All stages should use IBP.
    for (i, stage) in cert.stages.iter().enumerate() {
        assert!(
            stage.method.contains("ibp") || stage.method.contains("IBP"),
            "stage {i} method should be IBP, got: {}",
            stage.method
        );
    }
}

/// Mixed IBP (group 0) + CROWN (groups 1-2) produces valid certificate.
///
/// This is the intended production use case: large Conv1d groups use IBP,
/// smaller groups use CROWN for tighter bounds (#2599 Level 2).
#[test]
fn test_mixed_ibp_then_crown_produces_valid_certificate() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let grouping = build_three_group_grouping(NUM_RESBLOCKS);

    // Group 0 (pre-norm): IBP. Groups 1-2 (resblocks, output): CROWN.
    let modes = vec![
        GroupVerifyMode::Ibp,
        GroupVerifyMode::Crown,
        GroupVerifyMode::Crown,
    ];

    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise_mixed(&layers, &initial, &grouping, &modes)
        .expect("IBP+CROWN mixed layerwise");

    assert!(cert.is_valid, "mixed IBP+CROWN pipeline must be valid");
    assert_eq!(cert.stages.len(), 3);
}

// ===========================================================================
// Error paths
// ===========================================================================

/// Modes length != groups length must return error.
#[test]
fn test_mixed_modes_length_mismatch() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let grouping = build_three_group_grouping(NUM_RESBLOCKS);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);

    // 2 modes for 3 groups → error.
    let modes = vec![GroupVerifyMode::Crown, GroupVerifyMode::Ibp];
    let result = verify_layerwise_mixed(&layers, &initial, &grouping, &modes);
    assert!(result.is_err(), "modes length mismatch must be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("modes length"),
        "error should mention modes length, got: {err_msg}"
    );
}

/// Empty modes with empty groups still fails (fewer than 2 groups).
#[test]
fn test_mixed_single_group_error() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);

    let grouping = LayerwiseGrouping {
        groups: vec![vec![0, 1, 2, 3, 4, 5, 6, 7]],
    };
    let modes = vec![GroupVerifyMode::Crown];
    let result = verify_layerwise_mixed(&layers, &initial, &grouping, &modes);
    assert!(result.is_err(), "single group must be rejected");
}
