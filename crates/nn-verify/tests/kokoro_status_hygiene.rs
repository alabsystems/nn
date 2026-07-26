// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use nn_verify::{model_status_path, VerifyStatus};

fn temp_status_copy(src: &Path) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    path.push(format!(
        "nn_verify_status_kokoro_hygiene_{}_{}.json",
        std::process::id(),
        nanos
    ));
    std::fs::copy(src, &path).expect("copy status fixture");
    path
}

#[test]
fn test_obsolete_kokoro_entries_marked_stale() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let temp_status_path = temp_status_copy(&model_status_path(workspace_root, "kokoro"));
    let mut locked = VerifyStatus::load_locked(&temp_status_path).expect("load_locked");

    let stale_entries = [
        (
            "kokoro_chained_norm_crown_n10",
            "Superseded by kokoro_chained_norm_kokoro_n10; legacy CROWN-through-normalization diagnostic kept for history only.",
        ),
        (
            "kokoro_chained_norm_crown_n58",
            "Superseded by kokoro_chained_norm_kokoro_n58; legacy CROWN-through-normalization diagnostic kept for history only.",
        ),
        (
            "kokoro_chained_norm_pure_n2",
            "Degenerate pure InstanceNorm stress test retained for audit history only; not an active Kokoro production proof target.",
        ),
        (
            "kokoro_production_moonshot_2stage",
            "Superseded by later 3-stage/4-stage production moonshot pipelines; the 2-stage intermediate remains as historical diagnostic only.",
        ),
        (
            "kokoro_production_bert_encoder",
            "Standalone PlBert+bert_encoder uses an all-vocabulary embedding hull that is much looser than production phoneme inputs; downstream text/pipeline proofs are the active coverage target.",
        ),
        (
            "kokoro_production_bert_encoder_crown",
            "Standalone PlBert+bert_encoder uses an all-vocabulary embedding hull that is much looser than production phoneme inputs; downstream text/pipeline proofs are the active coverage target.",
        ),
        (
            "kokoro_production_prosody_predictor",
            "Standalone ProsodyPredictor uses synthetic text-feature/style bounds; superseded by composed text-to-prosody and moonshot pipeline proofs over real upstream TextEncoder ranges.",
        ),
        (
            "kokoro_production_prosody_predictor_crown",
            "Standalone ProsodyPredictor uses synthetic text-feature/style bounds; superseded by composed text-to-prosody and moonshot pipeline proofs over real upstream TextEncoder ranges.",
        ),
        (
            "kokoro_production_f0_predictor",
            "Standalone F0EnergyPredictor uses synthetic aligned/style bounds; superseded by downstream pipeline proofs. The corrected standalone F0 diagnostic remains as history only.",
        ),
        (
            "kokoro_production_f0_predictor_crown",
            "Standalone F0EnergyPredictor uses synthetic aligned/style bounds; superseded by downstream pipeline proofs. The corrected standalone F0 diagnostic remains as history only.",
        ),
    ];

    for (key, reason) in stale_entries {
        if locked.status.has_kernel(key) {
            locked.status.mark_stale(key, reason).expect("mark_stale");
            let entry = locked.status.kernel(key).expect("stale entry");
            assert!(entry.stale, "{key} must be stale");
            assert_eq!(entry.stale_reason.as_deref(), Some(reason));
        }
    }

    locked.save().expect("save status");
    drop(locked);

    let validation = VerifyStatus::load(&temp_status_path).expect("reload status");
    for (key, reason) in stale_entries {
        if let Some(entry) = validation.kernel(key) {
            assert!(entry.stale, "{key} must remain stale after reload");
            assert_eq!(entry.stale_reason.as_deref(), Some(reason));
        }
    }

    let _ = std::fs::remove_file(&temp_status_path);
}
