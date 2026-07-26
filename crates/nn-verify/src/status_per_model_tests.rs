// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for per-model status file splitting (#2577).

use std::path::{Path, PathBuf};

use super::*;
use crate::soundness_compat::VerificationSoundnessMode;
use crate::verify_types::PropMethod;
use crate::{InputBoundsRecord, KernelStatus, OutputBoundsRecord, ParamInputRecord, VerifyOutcome};

/// Build a minimal `KernelStatus` for testing. Stale fields are parameterized.
fn make_test_entry(stale: bool, stale_hint: Option<&str>) -> KernelStatus {
    KernelStatus {
        status: VerifyOutcome::Verified,
        method: PropMethod::Ibp,
        input_bounds: InputBoundsRecord {
            variable_inputs: vec![ParamInputRecord {
                param_index: 0,
                lower: -1.0,
                upper: 1.0,
            }],
            constant_params: vec![],
            input_shape: Some(vec![1]),
            input_range: Some((-1.0, 1.0)),
        },
        output_bounds: OutputBoundsRecord {
            lower: -1.0,
            upper: 1.0,
            tensor_lower: None,
            tensor_upper: None,
            shape: None,
            is_infeasible: false,
        },
        output_width: 2.0,
        crown_error: None,
        soundness_mode: VerificationSoundnessMode::Heuristic,
        smt: None,
        crown_coverage: None,
        ibp_comparison_width: None,
        crown_ibp_ratio: None,
        weight_artifact: None,
        soundness_justification: None,
        stale,
        stale_reason: stale_hint.map(String::from),
        proof_strength: None,
    }
}

#[test]
fn test_model_for_kernel_classification() {
    assert_eq!(model_for_kernel("kokoro_decoder"), "kokoro");
    assert_eq!(model_for_kernel("kokoro_full_pipeline"), "kokoro");
    assert_eq!(model_for_kernel("demucs_spectral_decoder_dconv"), "demucs");
    assert_eq!(model_for_kernel("htdemucs_full"), "demucs");
    assert_eq!(model_for_kernel("silero_vad_full"), "silero");
    assert_eq!(model_for_kernel("whisper_full"), "whisper");
    assert_eq!(model_for_kernel("qwen3_full"), "qwen3");
    assert_eq!(model_for_kernel("glm5_self_attention"), "glm5");
    assert_eq!(model_for_kernel("glm5_decoder_block"), "glm5");
    assert_eq!(model_for_kernel("gptoss_embed_norm_attn"), "gptoss");
    assert_eq!(model_for_kernel("gpt_oss_decoder"), "gptoss");
    assert_eq!(model_for_kernel("moe_dispatch_softmax"), "gptoss");
    assert_eq!(model_for_kernel("adain"), "shared");
    assert_eq!(model_for_kernel("snake_alpha_1"), "shared");
    assert_eq!(model_for_kernel("gelu"), "shared");
    assert_eq!(model_for_kernel("relu"), "shared");
}

#[test]
fn test_model_status_path() {
    let root = Path::new("/workspace");
    assert_eq!(
        model_status_path(root, "kokoro"),
        PathBuf::from("/workspace/nn_verify_status_kokoro.json")
    );
    assert_eq!(
        model_status_path(root, "shared"),
        PathBuf::from("/workspace/nn_verify_status_shared.json")
    );
}

#[test]
fn test_split_by_model_empty() {
    let status = VerifyStatus::default();
    let models = status.split_by_model();
    assert!(models.is_empty());
}

/// Regression test: `split_by_model` must preserve `stale` and `stale_reason`.
/// Data loss during #2577 migration lost 13 stale flags.
#[test]
fn test_split_by_model_preserves_stale_flags() {
    let stale_entry = make_test_entry(true, Some("test stale reason"));
    let non_stale_entry = make_test_entry(false, None);

    let mut status = VerifyStatus::default();
    status
        .kernels
        .insert("kokoro_decoder".to_string(), stale_entry.clone());
    status
        .kernels
        .insert("adain".to_string(), non_stale_entry.clone());
    status
        .history
        .insert("kokoro_decoder".to_string(), vec![stale_entry]);
    status
        .history
        .insert("adain".to_string(), vec![non_stale_entry]);

    let models = status.split_by_model();

    let kokoro = &models["kokoro"];
    let k_entry = &kokoro.kernels["kokoro_decoder"];
    assert!(k_entry.stale, "stale flag lost in kernels split");
    assert_eq!(k_entry.stale_reason.as_deref(), Some("test stale reason"));
    let k_hist = &kokoro.history["kokoro_decoder"][0];
    assert!(k_hist.stale, "stale flag lost in history split");
    assert_eq!(k_hist.stale_reason.as_deref(), Some("test stale reason"));

    let shared = &models["shared"];
    assert!(!shared.kernels["adain"].stale);
    assert!(shared.kernels["adain"].stale_reason.is_none());
}

/// Regression test: `save_per_model → load_merged` filesystem round-trip
/// must preserve all data including stale flags. #2577 migration data loss.
#[test]
fn test_save_per_model_load_merged_roundtrip() {
    let mut status = VerifyStatus::default();
    status.kernels.insert(
        "kokoro_dec".to_string(),
        make_test_entry(true, Some("stale")),
    );
    status
        .kernels
        .insert("demucs_enc".to_string(), make_test_entry(false, None));
    status
        .kernels
        .insert("silero_vad".to_string(), make_test_entry(false, None));
    status.kernels.insert(
        "whisper_enc".to_string(),
        make_test_entry(true, Some("stale")),
    );
    status
        .kernels
        .insert("qwen3_attn".to_string(), make_test_entry(false, None));
    status
        .kernels
        .insert("relu".to_string(), make_test_entry(false, None));
    status.history.insert(
        "kokoro_dec".to_string(),
        vec![make_test_entry(true, Some("h"))],
    );

    let tmp = std::env::temp_dir().join(format!("nn_per_model_roundtrip_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create temp dir");

    status.save_per_model(&tmp).expect("save_per_model");
    let loaded = VerifyStatus::load_merged(&tmp).expect("load_merged");

    assert_eq!(status.kernel_count(), loaded.kernel_count(), "kernel count");
    for (name, orig) in status.kernels() {
        let got = loaded
            .kernel(name)
            .unwrap_or_else(|| panic!("missing: {name}"));
        assert_eq!(orig.stale, got.stale, "stale mismatch: {name}");
        assert_eq!(
            orig.stale_reason, got.stale_reason,
            "reason mismatch: {name}"
        );
        assert_eq!(orig.status, got.status, "status mismatch: {name}");
    }

    assert_eq!(
        status.history().len(),
        loaded.history().len(),
        "history count"
    );
    let h = loaded.history_for("kokoro_dec").expect("kokoro history");
    assert_eq!(h.len(), 1);
    assert!(h[0].stale, "stale flag lost in history round-trip");

    let _ = std::fs::remove_dir_all(&tmp);
}
