// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended verification status file management and soundness tracking tests.
//!
//! Covers: JSON parsing, SoundnessMode/ProofStrength enums, entry CRUD via
//! record_pipeline, stale filtering, cross-model status, status report
//! generation, JSON round-trip fidelity, and concurrent load_locked access.
//!
//! Part of #4186.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use nn_verify::status::{compute_proof_strength, ProofStrength};
use nn_verify::status_report::{GapSummary, ModelSummary, StatusReport, VerificationBreakdown};
use nn_verify::{
    model_for_kernel, model_status_path, InputBoundsRecord, KernelStatus, OutputBoundsRecord,
    ParamInputRecord, PropMethod, VerificationSoundnessMode, VerifyOutcome, VerifyStatus,
    MODEL_CATEGORIES,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn temp_dir_unique(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "nn_status_ext_{}_{}_{}",
        prefix,
        std::process::id(),
        nanos
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn temp_file_unique(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("nn_status_ext_{prefix}_{nanos}.json"))
}

/// Build a KernelStatus via `KernelStatus::new()` using constructors for
/// non-exhaustive types (cross-crate struct literal is forbidden).
fn make_entry(
    soundness: VerificationSoundnessMode,
    method: PropMethod,
    output_width: f32,
    stale: bool,
) -> KernelStatus {
    let input_bounds = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let output_bounds = OutputBoundsRecord::new(-output_width / 2.0, output_width / 2.0);
    let mut ks = KernelStatus::new(
        VerifyOutcome::Verified,
        method,
        input_bounds,
        output_bounds,
        output_width,
        soundness,
    );
    ks.stale = stale;
    ks
}

/// Create a `VerifyStatus` from a JSON object using serde.
fn status_from_json(json: serde_json::Value) -> VerifyStatus {
    serde_json::from_value(json).expect("deserialize VerifyStatus from JSON")
}

/// Build a VerifyStatus using `record_pipeline` for a given set of entries.
fn build_status_via_pipeline(
    entries: &[(&str, PropMethod, f32, f32, VerificationSoundnessMode)],
) -> VerifyStatus {
    let mut status = VerifyStatus::default();
    for &(name, method, out_lo, out_hi, soundness) in entries {
        status
            .record_pipeline(
                name,
                method,
                -1.0,
                1.0,
                out_lo,
                out_hi,
                &[1],
                soundness,
                None,
            )
            .expect("record_pipeline");
    }
    status
}

// ===========================================================================
// 1. VerifyStatus JSON parsing
// ===========================================================================

#[test]
fn parse_empty_status_json() {
    let status: VerifyStatus = serde_json::from_str(r#"{"kernels":{}}"#).expect("parse empty");
    assert_eq!(status.kernel_count(), 0);
}

#[test]
fn parse_status_with_single_kernel() {
    let entry = make_entry(
        VerificationSoundnessMode::Sound,
        PropMethod::Ibp,
        2.0,
        false,
    );
    let json = serde_json::json!({
        "kernels": {
            "snake_alpha_1": serde_json::to_value(&entry).unwrap()
        }
    });
    let status = status_from_json(json);
    assert_eq!(status.kernel_count(), 1);
    assert!(status.has_kernel("snake_alpha_1"));
    let k = status.kernel("snake_alpha_1").expect("kernel exists");
    assert_eq!(k.status, VerifyOutcome::Verified);
    assert_eq!(k.soundness_mode, VerificationSoundnessMode::Sound);
}

#[test]
fn parse_status_with_multiple_kernels_and_history() {
    let e1 = make_entry(
        VerificationSoundnessMode::Sound,
        PropMethod::Crown,
        1.5,
        false,
    );
    let e2 = make_entry(
        VerificationSoundnessMode::Heuristic,
        PropMethod::Ibp,
        50.0,
        false,
    );
    let json = serde_json::json!({
        "kernels": {
            "kokoro_decoder": serde_json::to_value(&e1).unwrap(),
            "demucs_encoder": serde_json::to_value(&e2).unwrap(),
        },
        "history": {
            "kokoro_decoder": [serde_json::to_value(&e1).unwrap()]
        }
    });
    let status = status_from_json(json);
    assert_eq!(status.kernel_count(), 2);
    assert!(status.has_kernel("kokoro_decoder"));
    assert!(status.has_kernel("demucs_encoder"));
    let hist = status.history_for("kokoro_decoder");
    assert!(hist.is_some());
    assert_eq!(hist.unwrap().len(), 1);
}

#[test]
fn parse_legacy_json_without_soundness_mode_defaults_to_heuristic() {
    // Legacy JSON missing `soundness_mode` should default to Heuristic (fail-closed).
    let json_str = r#"{
        "kernels": {
            "legacy_kernel": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {
                    "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
                    "constant_params": [],
                    "input_shape": [1],
                    "input_range": [-1.0, 1.0]
                },
                "output_bounds": {"lower": -2.0, "upper": 2.0},
                "output_width": 4.0
            }
        }
    }"#;
    let status: VerifyStatus = serde_json::from_str(json_str).expect("parse legacy");
    let k = status.kernel("legacy_kernel").expect("kernel");
    // Default soundness_mode is Heuristic per soundness_compat::default_soundness_mode
    assert_eq!(k.soundness_mode, VerificationSoundnessMode::Heuristic);
}

// ===========================================================================
// 2. SoundnessMode enum coverage
// ===========================================================================

#[test]
fn soundness_mode_sound_serialization() {
    let json = serde_json::to_string(&VerificationSoundnessMode::Sound).expect("serialize");
    assert_eq!(json, r#""sound""#);
    let rt: VerificationSoundnessMode = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(rt, VerificationSoundnessMode::Sound);
}

#[test]
fn soundness_mode_heuristic_serialization() {
    let json = serde_json::to_string(&VerificationSoundnessMode::Heuristic).expect("serialize");
    assert_eq!(json, r#""heuristic""#);
    let rt: VerificationSoundnessMode = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(rt, VerificationSoundnessMode::Heuristic);
}

#[test]
fn soundness_mode_equality() {
    assert_eq!(
        VerificationSoundnessMode::Sound,
        VerificationSoundnessMode::Sound
    );
    assert_ne!(
        VerificationSoundnessMode::Sound,
        VerificationSoundnessMode::Heuristic
    );
}

// ===========================================================================
// 3. Entry CRUD via record_pipeline (add, update, mark stale)
// ===========================================================================

#[test]
fn add_entry_via_record_pipeline() {
    let mut status = VerifyStatus::default();
    status
        .record_pipeline(
            "test_kernel",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -5.0,
            5.0,
            &[1, 64],
            VerificationSoundnessMode::Sound,
            Some(&[1, 3, 16000]),
        )
        .expect("record_pipeline");

    assert_eq!(status.kernel_count(), 1);
    let k = status.kernel("test_kernel").expect("kernel exists");
    assert_eq!(k.status, VerifyOutcome::Verified);
    assert_eq!(k.method, PropMethod::Ibp);
    assert_eq!(k.soundness_mode, VerificationSoundnessMode::Sound);
    assert!((k.output_width - 10.0).abs() < 1e-6);
}

#[test]
fn update_entry_overwrites_latest() {
    let mut status = VerifyStatus::default();
    status
        .record_pipeline(
            "test_kernel",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -5.0,
            5.0,
            &[1],
            VerificationSoundnessMode::Heuristic,
            None,
        )
        .expect("first record");

    // Overwrite with a CROWN result.
    status
        .record_pipeline(
            "test_kernel",
            PropMethod::Crown,
            -1.0,
            1.0,
            -2.0,
            2.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("second record");

    assert_eq!(status.kernel_count(), 1);
    let k = status.kernel("test_kernel").expect("kernel");
    assert_eq!(k.method, PropMethod::Crown);
    assert_eq!(k.soundness_mode, VerificationSoundnessMode::Sound);
    // History should have 2 entries.
    let hist = status.history_for("test_kernel").expect("history");
    assert_eq!(hist.len(), 2);
}

#[test]
fn mark_stale_excludes_from_counts() {
    let mut status = VerifyStatus::default();
    status
        .record_pipeline(
            "kokoro_decoder",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -3.0,
            3.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record");
    status
        .record_pipeline(
            "demucs_encoder",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -3.0,
            3.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record");

    assert_eq!(status.soundness_counts(), (2, 0));

    status
        .mark_stale("kokoro_decoder", "superseded by new architecture")
        .expect("mark_stale");

    let (sound, heuristic) = status.soundness_counts();
    assert_eq!(sound, 1);
    assert_eq!(heuristic, 0);

    // The stale entry still exists in kernels.
    assert!(status.has_kernel("kokoro_decoder"));
    let k = status.kernel("kokoro_decoder").expect("entry");
    assert!(k.stale);
    assert_eq!(
        k.stale_reason.as_deref(),
        Some("superseded by new architecture")
    );
}

#[test]
fn mark_stale_nonexistent_kernel_returns_error() {
    let mut status = VerifyStatus::default();
    let result = status.mark_stale("nonexistent", "reason");
    assert!(result.is_err());
}

#[test]
fn record_pipeline_rejects_non_finite_bounds() {
    let mut status = VerifyStatus::default();
    let result = status.record_pipeline(
        "bad",
        PropMethod::Ibp,
        f32::NAN,
        1.0,
        -1.0,
        1.0,
        &[1],
        VerificationSoundnessMode::Sound,
        None,
    );
    assert!(result.is_err());
}

// ===========================================================================
// 4. Sound count aggregation
// ===========================================================================

#[test]
fn soundness_counts_empty_status() {
    let status = VerifyStatus::default();
    assert_eq!(status.soundness_counts(), (0, 0));
}

#[test]
fn soundness_counts_mixed_entries() {
    let status = build_status_via_pipeline(&[
        (
            "k1",
            PropMethod::Ibp,
            -1.0,
            1.0,
            VerificationSoundnessMode::Sound,
        ),
        (
            "k2",
            PropMethod::Crown,
            -2.0,
            2.0,
            VerificationSoundnessMode::Sound,
        ),
        (
            "k3",
            PropMethod::Ibp,
            -3.0,
            3.0,
            VerificationSoundnessMode::Heuristic,
        ),
    ]);
    let (sound, heuristic) = status.soundness_counts();
    assert_eq!(sound, 2);
    assert_eq!(heuristic, 1);
}

#[test]
fn soundness_counts_all_heuristic() {
    let status = build_status_via_pipeline(&[
        (
            "k1",
            PropMethod::Ibp,
            -1.0,
            1.0,
            VerificationSoundnessMode::Heuristic,
        ),
        (
            "k2",
            PropMethod::Ibp,
            -2.0,
            2.0,
            VerificationSoundnessMode::Heuristic,
        ),
    ]);
    assert_eq!(status.soundness_counts(), (0, 2));
}

// ===========================================================================
// 5. ProofStrength classification
// ===========================================================================

#[test]
fn proof_strength_sound_crown() {
    let strength = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Crown, 5.0);
    assert_eq!(strength, ProofStrength::SoundCrown);
}

#[test]
fn proof_strength_sound_alpha_crown() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::AlphaCrown,
        5.0,
    );
    assert_eq!(strength, ProofStrength::SoundCrown);
}

#[test]
fn proof_strength_sound_beta_crown() {
    let strength =
        compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::BetaCrown, 5.0);
    assert_eq!(strength, ProofStrength::SoundCrown);
}

#[test]
fn proof_strength_sound_analytical() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::Analytical,
        5.0,
    );
    assert_eq!(strength, ProofStrength::SoundCrown);
}

#[test]
fn proof_strength_sound_ibp() {
    let strength = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Ibp, 5.0);
    assert_eq!(strength, ProofStrength::SoundIbp);
}

#[test]
fn proof_strength_sound_mixed() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::MixedIbpCrown,
        5.0,
    );
    assert_eq!(strength, ProofStrength::SoundMixed);
}

#[test]
fn proof_strength_heuristic() {
    let strength =
        compute_proof_strength(VerificationSoundnessMode::Heuristic, PropMethod::Crown, 5.0);
    assert_eq!(strength, ProofStrength::Heuristic);
}

#[test]
fn proof_strength_vacuous_overrides_sound() {
    // Width > VACUOUS_WIDTH_THRESHOLD (100.0) => Vacuous regardless of soundness.
    let strength =
        compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Crown, 200.0);
    assert_eq!(strength, ProofStrength::Vacuous);
}

#[test]
fn proof_strength_vacuous_overrides_heuristic() {
    let strength =
        compute_proof_strength(VerificationSoundnessMode::Heuristic, PropMethod::Ibp, 150.0);
    assert_eq!(strength, ProofStrength::Vacuous);
}

#[test]
fn proof_strength_at_threshold_boundary() {
    // Exactly at threshold (100.0) should NOT be vacuous.
    let at = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Ibp, 100.0);
    assert_eq!(at, ProofStrength::SoundIbp);

    // Just above threshold.
    let above = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Ibp, 100.01);
    assert_eq!(above, ProofStrength::Vacuous);
}

#[test]
fn proof_strength_counts_on_status() {
    let status = build_status_via_pipeline(&[
        (
            "k_crown",
            PropMethod::Crown,
            -1.0,
            1.0,
            VerificationSoundnessMode::Sound,
        ),
        (
            "k_ibp",
            PropMethod::Ibp,
            -1.0,
            1.0,
            VerificationSoundnessMode::Sound,
        ),
        (
            "k_heur",
            PropMethod::Ibp,
            -10.0,
            10.0,
            VerificationSoundnessMode::Heuristic,
        ),
        (
            "k_vacuous",
            PropMethod::Ibp,
            -200.0,
            200.0,
            VerificationSoundnessMode::Heuristic,
        ),
    ]);
    let (sc, si, h, v) = status.proof_strength_counts();
    assert_eq!(sc, 1, "sound_crown");
    assert_eq!(si, 1, "sound_ibp");
    assert_eq!(h, 1, "heuristic");
    assert_eq!(v, 1, "vacuous");
}

// ===========================================================================
// 6. Stale filtering in aggregates
// ===========================================================================

#[test]
fn stale_entries_excluded_from_soundness_counts() {
    let mut status = build_status_via_pipeline(&[
        (
            "k1",
            PropMethod::Ibp,
            -1.0,
            1.0,
            VerificationSoundnessMode::Sound,
        ),
        (
            "k2",
            PropMethod::Ibp,
            -1.0,
            1.0,
            VerificationSoundnessMode::Sound,
        ),
        (
            "k3",
            PropMethod::Ibp,
            -1.0,
            1.0,
            VerificationSoundnessMode::Heuristic,
        ),
    ]);
    assert_eq!(status.soundness_counts(), (2, 1));

    status.mark_stale("k1", "obsolete").expect("mark_stale");
    assert_eq!(status.soundness_counts(), (1, 1));
}

#[test]
fn stale_entries_excluded_from_proof_strength_counts() {
    let mut status = build_status_via_pipeline(&[
        (
            "k_crown",
            PropMethod::Crown,
            -1.0,
            1.0,
            VerificationSoundnessMode::Sound,
        ),
        (
            "k_ibp",
            PropMethod::Ibp,
            -1.0,
            1.0,
            VerificationSoundnessMode::Sound,
        ),
    ]);
    let (sc, si, _, _) = status.proof_strength_counts();
    assert_eq!(sc, 1);
    assert_eq!(si, 1);

    status
        .mark_stale("k_crown", "obsolete")
        .expect("mark_stale");
    let (sc2, si2, _, _) = status.proof_strength_counts();
    assert_eq!(sc2, 0, "stale SoundCrown should be excluded");
    assert_eq!(si2, 1);
}

#[test]
fn stale_entries_counted_in_breakdown_stale_field() {
    let e1 = make_entry(VerificationSoundnessMode::Sound, PropMethod::Ibp, 2.0, true);
    let e2 = make_entry(
        VerificationSoundnessMode::Sound,
        PropMethod::Ibp,
        2.0,
        false,
    );
    let entries: Vec<&KernelStatus> = vec![&e1, &e2];
    let breakdown = VerificationBreakdown::from_entries(&entries);
    assert_eq!(breakdown.stale, 1);
    assert_eq!(breakdown.total, 1);
    assert_eq!(breakdown.sound, 1);
}

// ===========================================================================
// 7. Cross-model status (kernel classification)
// ===========================================================================

#[test]
fn model_classification_covers_all_known_models() {
    let test_cases = [
        ("kokoro_decoder", "kokoro"),
        ("kokoro_full_pipeline", "kokoro"),
        ("demucs_spectral_decoder", "demucs"),
        ("htdemucs_full", "demucs"),
        ("silero_vad_full", "silero"),
        ("whisper_encoder_block", "whisper"),
        ("qwen3_attention_head", "qwen3"),
        ("glm5_decoder_block", "glm5"),
        ("glm_self_attention", "glm"),
        ("gptoss_embed_norm_attn", "gptoss"),
        ("gpt_oss_decoder", "gptoss"),
        ("moe_dispatch_softmax", "gptoss"),
        ("snake_alpha_1", "shared"),
        ("gelu", "shared"),
        ("relu", "shared"),
        ("adain", "shared"),
    ];
    for &(kernel_name, expected_model) in &test_cases {
        assert_eq!(
            model_for_kernel(kernel_name),
            expected_model,
            "model_for_kernel({kernel_name}) mismatch"
        );
    }
}

#[test]
fn model_status_path_format() {
    for &model in MODEL_CATEGORIES {
        let path = model_status_path(Path::new("/workspace"), model);
        let expected = format!("/workspace/nn_verify_status_{model}.json");
        assert_eq!(path.to_str().unwrap(), expected);
    }
}

#[test]
fn split_by_model_round_trip() {
    let mut status = VerifyStatus::default();
    status
        .record_pipeline(
            "kokoro_dec",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -3.0,
            3.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record");
    status
        .record_pipeline(
            "demucs_enc",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -3.0,
            3.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record");
    status
        .record_pipeline(
            "gelu",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -3.0,
            3.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record");

    let models = status.split_by_model();
    assert!(models.contains_key("kokoro"));
    assert!(models.contains_key("demucs"));
    assert!(models.contains_key("shared"));
    assert_eq!(models["kokoro"].kernel_count(), 1);
    assert_eq!(models["demucs"].kernel_count(), 1);
    assert_eq!(models["shared"].kernel_count(), 1);
}

#[test]
fn save_and_load_per_model_files() {
    let dir = temp_dir_unique("per_model");
    let mut status = VerifyStatus::default();
    status
        .record_pipeline(
            "kokoro_a",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -2.0,
            2.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record");
    status
        .record_pipeline(
            "whisper_b",
            PropMethod::Crown,
            -1.0,
            1.0,
            -1.0,
            1.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record");

    status.save_per_model(&dir).expect("save_per_model");

    // Verify files exist for kokoro and whisper.
    assert!(model_status_path(&dir, "kokoro").exists());
    assert!(model_status_path(&dir, "whisper").exists());
    // Shared should NOT exist (no shared kernels).
    assert!(!model_status_path(&dir, "shared").exists());

    // Load merged should reconstruct all entries.
    let merged = VerifyStatus::load_merged(&dir).expect("load_merged");
    assert_eq!(merged.kernel_count(), 2);
    assert!(merged.has_kernel("kokoro_a"));
    assert!(merged.has_kernel("whisper_b"));

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// 8. Status report generation
// ===========================================================================

#[test]
fn report_from_single_status() {
    let status = build_status_via_pipeline(&[
        (
            "k1",
            PropMethod::Ibp,
            -1.0,
            1.0,
            VerificationSoundnessMode::Sound,
        ),
        (
            "k2",
            PropMethod::Crown,
            -2.0,
            2.0,
            VerificationSoundnessMode::Sound,
        ),
        (
            "k3",
            PropMethod::Ibp,
            -10.0,
            10.0,
            VerificationSoundnessMode::Heuristic,
        ),
    ]);
    let report = StatusReport::from_verify_status("test_model", &status);
    assert_eq!(report.total_entries(), 3);
    assert_eq!(report.summary.sound, 2);
    assert_eq!(report.summary.heuristic, 1);
    assert_eq!(report.models.len(), 1);
    assert_eq!(report.models[0].model, "test_model");
}

#[test]
fn report_from_status_files_in_temp_dir() {
    let dir = temp_dir_unique("report_gen");

    let mut kokoro_status = VerifyStatus::default();
    kokoro_status
        .record_pipeline(
            "kokoro_dec",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -3.0,
            3.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record");
    kokoro_status
        .save(&model_status_path(&dir, "kokoro"))
        .expect("save");

    let mut whisper_status = VerifyStatus::default();
    whisper_status
        .record_pipeline(
            "whisper_enc",
            PropMethod::Crown,
            -1.0,
            1.0,
            -1.0,
            1.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record");
    whisper_status
        .save(&model_status_path(&dir, "whisper"))
        .expect("save");

    let report = StatusReport::from_status_files(&dir).expect("report");
    assert_eq!(report.total_entries(), 2);
    assert_eq!(report.summary.sound, 2);
    assert_eq!(report.models.len(), 2);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn report_text_contains_expected_sections() {
    let status = build_status_via_pipeline(&[(
        "k1",
        PropMethod::Ibp,
        -1.0,
        1.0,
        VerificationSoundnessMode::Sound,
    )]);
    let report = StatusReport::from_verify_status("kokoro", &status)
        .with_kani_count(10000)
        .with_gap_summary(GapSummary {
            stages_checked: 10,
            gaps: 0,
            vacuous: 2,
        });

    let text = report.to_text();
    assert!(
        text.contains("nn Verification Status Report"),
        "missing header"
    );
    assert!(text.contains("Total entries: 1"), "missing total");
    assert!(text.contains("sound"), "missing soundness");
    assert!(text.contains("Kani harnesses: 10000"), "missing kani count");
    assert!(text.contains("Gaps: 0"), "missing gap info");
    assert!(text.contains("Vacuous: 2"), "missing vacuous info");
    assert!(text.contains("kokoro"), "missing model name");
}

#[test]
fn report_display_trait_produces_same_as_to_text() {
    let status = build_status_via_pipeline(&[(
        "k1",
        PropMethod::Ibp,
        -1.0,
        1.0,
        VerificationSoundnessMode::Sound,
    )]);
    let report = StatusReport::from_verify_status("test", &status);
    let display_output = format!("{report}");
    let text_output = report.to_text();
    assert_eq!(display_output, text_output);
}

#[test]
fn report_empty_models_produces_zero_entries() {
    let report =
        StatusReport::from_status_files(Path::new("/nonexistent/dir/xyz")).expect("empty report");
    assert_eq!(report.total_entries(), 0);
    assert!(report.models.is_empty());
    assert_eq!(report.kani_count, None);
    assert_eq!(report.gap_summary, None);
    assert_eq!(report.trend, None);
}

// ===========================================================================
// 9. JSON round-trip fidelity
// ===========================================================================

#[test]
fn full_kernel_status_roundtrip() {
    let input_bounds = InputBoundsRecord::new(
        &[
            ParamInputRecord::new(0, -10.0, 10.0),
            ParamInputRecord::new(1, 0.0, 1.0),
        ],
        &[0.5, 2.0],
    );
    let output_bounds = OutputBoundsRecord::with_shape(-5.0, 5.0, vec![2]);
    let entry = KernelStatus::new(
        VerifyOutcome::IbpFallback,
        PropMethod::AlphaCrown,
        input_bounds,
        output_bounds,
        10.0,
        VerificationSoundnessMode::Sound,
    );

    let json = serde_json::to_string_pretty(&entry).expect("serialize");
    let rt: KernelStatus = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(rt.status, entry.status);
    assert_eq!(rt.method, entry.method);
    assert_eq!(rt.output_width, entry.output_width);
    assert_eq!(rt.soundness_mode, entry.soundness_mode);
    assert_eq!(rt.proof_strength, entry.proof_strength);
    assert_eq!(rt.input_bounds, entry.input_bounds);
    assert_eq!(rt.output_bounds, entry.output_bounds);
    assert_eq!(rt.crown_error, None);
    assert_eq!(rt.smt, None);
    assert!(!rt.stale);
}

#[test]
fn verify_status_save_load_roundtrip_via_file() {
    let path = temp_file_unique("roundtrip");
    let mut status = VerifyStatus::default();
    status
        .record_pipeline(
            "test_k1",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -3.0,
            3.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record");
    status
        .record_pipeline(
            "test_k2",
            PropMethod::Crown,
            -2.0,
            2.0,
            -5.0,
            5.0,
            &[1, 64],
            VerificationSoundnessMode::Heuristic,
            Some(&[1, 3, 16000]),
        )
        .expect("record");

    status.save(&path).expect("save");
    let loaded = VerifyStatus::load(&path).expect("load");

    assert_eq!(loaded.kernel_count(), 2);
    assert!(loaded.has_kernel("test_k1"));
    assert!(loaded.has_kernel("test_k2"));

    let k1 = loaded.kernel("test_k1").expect("k1");
    assert_eq!(k1.method, PropMethod::Ibp);
    assert_eq!(k1.soundness_mode, VerificationSoundnessMode::Sound);

    let k2 = loaded.kernel("test_k2").expect("k2");
    assert_eq!(k2.method, PropMethod::Crown);
    assert_eq!(k2.soundness_mode, VerificationSoundnessMode::Heuristic);

    // History should be present too.
    assert!(loaded.history_for("test_k1").is_some());
    assert!(loaded.history_for("test_k2").is_some());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn stale_fields_survive_roundtrip() {
    let path = temp_file_unique("stale_rt");
    let mut status = VerifyStatus::default();
    status
        .record_pipeline(
            "k1",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -3.0,
            3.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record");
    status
        .mark_stale("k1", "old architecture")
        .expect("mark_stale");

    status.save(&path).expect("save");
    let loaded = VerifyStatus::load(&path).expect("load");
    let k = loaded.kernel("k1").expect("k1");
    assert!(k.stale);
    assert_eq!(k.stale_reason.as_deref(), Some("old architecture"));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn proof_strength_serialization_roundtrip() {
    let variants = [
        ProofStrength::SoundCrown,
        ProofStrength::SoundIbp,
        ProofStrength::SoundMixed,
        ProofStrength::Heuristic,
        ProofStrength::Vacuous,
    ];
    for variant in &variants {
        let json = serde_json::to_string(variant).expect("serialize");
        let rt: ProofStrength = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&rt, variant, "ProofStrength roundtrip failed for {json}");
    }
}

#[test]
fn verify_outcome_serialization_roundtrip() {
    let variants = [
        VerifyOutcome::Verified,
        VerifyOutcome::BoundsComputed,
        VerifyOutcome::IbpFallback,
        VerifyOutcome::Failed,
        VerifyOutcome::SmtContradiction,
    ];
    for variant in &variants {
        let json = serde_json::to_string(variant).expect("serialize");
        let rt: VerifyOutcome = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&rt, variant, "VerifyOutcome roundtrip failed for {json}");
    }
}

#[test]
fn prop_method_serialization_roundtrip() {
    let variants = [
        PropMethod::Ibp,
        PropMethod::Crown,
        PropMethod::AlphaCrown,
        PropMethod::BetaCrown,
        PropMethod::Analytical,
        PropMethod::MixedIbpCrown,
    ];
    for variant in &variants {
        let json = serde_json::to_string(variant).expect("serialize");
        let rt: PropMethod = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&rt, variant, "PropMethod roundtrip failed for {json}");
    }
}

#[test]
fn model_summary_roundtrip() {
    let summary = ModelSummary {
        model: "kokoro".to_string(),
        breakdown: VerificationBreakdown {
            total: 53,
            sound: 53,
            heuristic: 0,
            stale: 10,
            sound_crown: 7,
            sound_ibp: 45,
            sound_mixed: 1,
            heuristic_non_vacuous: 0,
            vacuous: 0,
        },
    };
    let json = serde_json::to_string(&summary).expect("serialize");
    let rt: ModelSummary = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(rt, summary);
}

// ===========================================================================
// 10. Concurrent access via load_locked
// ===========================================================================

#[test]
fn load_locked_save_produces_consistent_file() {
    let path = temp_file_unique("locked");

    // Create initial status.
    let mut status = VerifyStatus::default();
    status
        .record_pipeline(
            "k1",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -3.0,
            3.0,
            &[1],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record");
    status.save(&path).expect("initial save");

    // Use load_locked to modify and save.
    {
        let mut locked = VerifyStatus::load_locked(&path).expect("load_locked");
        locked
            .status
            .record_pipeline(
                "k2",
                PropMethod::Crown,
                -1.0,
                1.0,
                -2.0,
                2.0,
                &[1],
                VerificationSoundnessMode::Sound,
                None,
            )
            .expect("record k2 under lock");
        locked.save().expect("locked save");
    } // lock released here

    // Verify both entries persisted.
    let reloaded = VerifyStatus::load(&path).expect("reload");
    assert_eq!(reloaded.kernel_count(), 2);
    assert!(reloaded.has_kernel("k1"));
    assert!(reloaded.has_kernel("k2"));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_locked_from_nonexistent_file_creates_default() {
    let path = temp_file_unique("locked_new");
    // Ensure the file does not exist.
    let _ = std::fs::remove_file(&path);

    let locked = VerifyStatus::load_locked(&path).expect("load_locked on missing file");
    assert_eq!(locked.status.kernel_count(), 0);

    // Lock file may have been created as a side-effect.
    let lock_path = path.with_extension("json.lock");
    let _ = std::fs::remove_file(&lock_path);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_locked_prevents_concurrent_mutation() {
    // This test verifies the locking mechanism by acquiring a lock and
    // verifying the status file is coherent after sequential locked operations.
    let path = temp_file_unique("concurrent");
    let status = VerifyStatus::default();
    status.save(&path).expect("initial save");

    // First locked session: add entry.
    {
        let mut locked = VerifyStatus::load_locked(&path).expect("lock 1");
        locked
            .status
            .record_pipeline(
                "k_first",
                PropMethod::Ibp,
                -1.0,
                1.0,
                -3.0,
                3.0,
                &[1],
                VerificationSoundnessMode::Sound,
                None,
            )
            .expect("record");
        locked.save().expect("save 1");
    }

    // Second locked session: add another entry.
    {
        let mut locked = VerifyStatus::load_locked(&path).expect("lock 2");
        assert!(
            locked.status.has_kernel("k_first"),
            "first entry should be visible"
        );
        locked
            .status
            .record_pipeline(
                "k_second",
                PropMethod::Crown,
                -1.0,
                1.0,
                -2.0,
                2.0,
                &[1],
                VerificationSoundnessMode::Sound,
                None,
            )
            .expect("record");
        locked.save().expect("save 2");
    }

    // Verify final state.
    let final_status = VerifyStatus::load(&path).expect("final load");
    assert_eq!(final_status.kernel_count(), 2);
    assert!(final_status.has_kernel("k_first"));
    assert!(final_status.has_kernel("k_second"));

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Bonus: VerificationBreakdown.sound_fraction edge cases
// ===========================================================================

#[test]
fn sound_fraction_zero_when_no_entries() {
    let b = VerificationBreakdown::default();
    assert!((b.sound_fraction() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn sound_fraction_one_when_all_sound() {
    let e1 = make_entry(
        VerificationSoundnessMode::Sound,
        PropMethod::Ibp,
        2.0,
        false,
    );
    let e2 = make_entry(
        VerificationSoundnessMode::Sound,
        PropMethod::Crown,
        1.0,
        false,
    );
    let entries: Vec<&KernelStatus> = vec![&e1, &e2];
    let b = VerificationBreakdown::from_entries(&entries);
    assert!((b.sound_fraction() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn sound_fraction_only_counts_non_stale() {
    let e_sound = make_entry(
        VerificationSoundnessMode::Sound,
        PropMethod::Ibp,
        2.0,
        false,
    );
    let e_stale = make_entry(VerificationSoundnessMode::Sound, PropMethod::Ibp, 2.0, true);
    let entries: Vec<&KernelStatus> = vec![&e_sound, &e_stale];
    let b = VerificationBreakdown::from_entries(&entries);
    // Only 1 non-stale entry, which is sound. Fraction = 1.0.
    assert!((b.sound_fraction() - 1.0).abs() < f64::EPSILON);
    assert_eq!(b.total, 1);
    assert_eq!(b.stale, 1);
}
