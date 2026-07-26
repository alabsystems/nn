// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: operational_state Kokoro verification counts must match
//! the live `nn_verify_status_kokoro.json` status file.

use std::path::Path;

use nn_verify::status::{compute_proof_strength, ProofStrength};
use nn_verify::VerifyStatus;

fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn test_kokoro_operational_state_counts_match_status_file() {
    let ws = workspace_root();
    let status_path = ws.join("nn_verify_status_kokoro.json");
    let state_path = ws.join("operational_state.json");

    // Both artifacts are generated, gitignored outputs of the verification run
    // (`cargo run --example verify_all` + operational-state writer); a bare
    // `cargo test`/`nextest` run does not produce them. Skip cleanly when either
    // is absent — mirrors the convention in
    // gap_detector_tests::test_detect_gaps_real_status_file — rather than
    // hard-failing on a missing artifact. The cross-consistency assertions below
    // are unchanged and still run whenever both artifacts exist.
    if !status_path.exists() || !state_path.exists() {
        eprintln!(
            "Skipping: missing {} and/or {}. Run \
             `cargo run -p nn-verify --example verify_all` to generate them.",
            status_path.display(),
            state_path.display()
        );
        return;
    }

    let status = VerifyStatus::load(&status_path).expect("load Kokoro status file");
    let state: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&state_path).expect("read operational state"),
    )
    .expect("parse operational state");

    let soundness = &state["verification_counts"]["kokoro_soundness"];
    let (sound_count, heuristic_count) = status.soundness_counts();
    assert_eq!(
        soundness["sound"].as_u64(),
        Some(sound_count as u64),
        "operational_state kokoro_soundness.sound must match non-stale status entries",
    );
    assert_eq!(
        soundness["heuristic"].as_u64(),
        Some(heuristic_count as u64),
        "operational_state kokoro_soundness.heuristic must match non-stale status entries",
    );
    assert_eq!(
        soundness["total"].as_u64(),
        Some((sound_count + heuristic_count) as u64),
        "operational_state kokoro_soundness.total must match non-stale status entries",
    );

    let mut sound_crown = 0usize;
    let mut sound_ibp = 0usize;
    let mut sound_mixed = 0usize;
    let mut heuristic_non_vacuous = 0usize;
    let mut vacuous = 0usize;

    for entry in status.kernels().values() {
        if entry.stale {
            continue;
        }
        let strength = entry.proof_strength.unwrap_or_else(|| {
            compute_proof_strength(entry.soundness_mode, entry.method, entry.output_width)
        });
        match strength {
            ProofStrength::SoundCrown => sound_crown += 1,
            ProofStrength::SoundIbp => sound_ibp += 1,
            ProofStrength::SoundMixed => sound_mixed += 1,
            ProofStrength::Heuristic => heuristic_non_vacuous += 1,
            ProofStrength::Vacuous => vacuous += 1,
            _ => panic!("unexpected proof_strength variant in Kokoro status file"),
        }
    }

    let proof_strength = &state["verification_counts"]["kokoro_proof_strength"];
    assert_eq!(
        proof_strength["sound_crown"].as_u64(),
        Some(sound_crown as u64),
        "operational_state kokoro_proof_strength.sound_crown must match status file",
    );
    assert_eq!(
        proof_strength["sound_ibp"].as_u64(),
        Some(sound_ibp as u64),
        "operational_state kokoro_proof_strength.sound_ibp must match status file",
    );
    assert_eq!(
        proof_strength["sound_mixed"].as_u64(),
        Some(sound_mixed as u64),
        "operational_state kokoro_proof_strength.sound_mixed must match status file",
    );
    assert_eq!(
        proof_strength["sound_total"].as_u64(),
        Some((sound_crown + sound_ibp + sound_mixed) as u64),
        "operational_state kokoro_proof_strength.sound_total must match status file",
    );
    assert_eq!(
        proof_strength["heuristic_non_vacuous"].as_u64(),
        Some(heuristic_non_vacuous as u64),
        "operational_state kokoro_proof_strength.heuristic_non_vacuous must match status file",
    );
    assert_eq!(
        proof_strength["vacuous"].as_u64(),
        Some(vacuous as u64),
        "operational_state kokoro_proof_strength.vacuous must match status file",
    );
    assert_eq!(
        proof_strength["total"].as_u64(),
        Some((sound_crown + sound_ibp + sound_mixed + heuristic_non_vacuous + vacuous) as u64),
        "operational_state kokoro_proof_strength.total must match status file",
    );
}
