// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-model status file split for concurrent Worker safety.
//!
//! Splits the monolithic `nn_verify_status.json` into per-model files to
//! prevent concurrent modification races (#2577). Each model gets its own
//! file with its own advisory lock, so Workers verifying different models
//! never contend on the same file.
//!
//! Model categories: `kokoro`, `demucs`, `silero`, `whisper`, `qwen3`, `glm5`, `glm`, `shared`.
//!
//! Part of #2577, #2218.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::VerifyStatus;
use crate::error::VerifyError;

/// All known model categories for status file splitting.
pub const MODEL_CATEGORIES: &[&str] = &[
    "kokoro", "demucs", "silero", "whisper", "qwen3", "glm5", "glm", "gptoss", "shared",
];

/// Classify a kernel name into its model category.
///
/// Uses prefix-based matching. Kernels that don't match any model prefix
/// are classified as `"shared"` (generic/reusable kernels like `adain`,
/// `gelu`, `snake`, `relu`, etc.).
#[must_use]
pub fn model_for_kernel(name: &str) -> &'static str {
    if name.starts_with("kokoro") {
        "kokoro"
    } else if name.starts_with("demucs") || name.starts_with("htdemucs") {
        "demucs"
    } else if name.starts_with("silero") {
        "silero"
    } else if name.starts_with("whisper") {
        "whisper"
    } else if name.starts_with("qwen3") {
        "qwen3"
    } else if name.starts_with("glm5") {
        "glm5"
    } else if name.starts_with("glm") {
        "glm"
    } else if name.starts_with("gptoss")
        || name.starts_with("gpt_oss")
        || name.starts_with("moe_dispatch")
    {
        "gptoss"
    } else {
        "shared"
    }
}

/// Compute the per-model status file path.
///
/// Given the workspace root directory, returns the path for a specific model's
/// status file. Example: `workspace_root/nn_verify_status_kokoro.json`.
#[must_use]
pub fn model_status_path(workspace_root: &Path, model: &str) -> PathBuf {
    workspace_root.join(format!("nn_verify_status_{model}.json"))
}

impl VerifyStatus {
    /// Split this status into per-model `VerifyStatus` instances.
    ///
    /// Each returned entry contains only the kernels and history belonging
    /// to that model category. The map key is the model category name.
    #[must_use]
    pub fn split_by_model(&self) -> BTreeMap<String, Self> {
        let mut models: BTreeMap<String, Self> = BTreeMap::new();

        for (name, status) in &self.kernels {
            let model = model_for_kernel(name).to_string();
            models
                .entry(model)
                .or_default()
                .kernels
                .insert(name.clone(), status.clone());
        }

        for (name, history_entries) in &self.history {
            let model = model_for_kernel(name).to_string();
            models
                .entry(model)
                .or_default()
                .history
                .insert(name.clone(), history_entries.clone());
        }

        models
    }

    /// Load and merge all per-model status files from a directory.
    ///
    /// Reads each `nn_verify_status_{model}.json` file that exists and merges
    /// all kernels and history into a single `VerifyStatus`. Files that don't
    /// exist are silently skipped.
    pub fn load_merged(workspace_root: &Path) -> Result<Self, VerifyError> {
        let mut merged = Self::default();

        for &model in MODEL_CATEGORIES {
            let path = model_status_path(workspace_root, model);
            if path.exists() {
                let model_status = Self::load(&path)?;
                for (name, status) in model_status.kernels {
                    merged.kernels.insert(name, status);
                }
                for (name, history_entries) in model_status.history {
                    merged.history.insert(name, history_entries);
                }
            }
        }

        Ok(merged)
    }

    /// Save this status split across per-model files.
    ///
    /// Splits the status by model category and writes each to its own file
    /// using atomic write semantics. Empty model categories are not written
    /// (and existing empty files are removed).
    pub fn save_per_model(&self, workspace_root: &Path) -> Result<(), VerifyError> {
        let models = self.split_by_model();

        for &model in MODEL_CATEGORIES {
            let path = model_status_path(workspace_root, model);
            match models.get(model) {
                Some(status) if !status.kernels.is_empty() => {
                    status.save(&path)?;
                }
                _ => {
                    // Remove empty model files to avoid clutter.
                    let _ = std::fs::remove_file(&path);
                }
            }
        }

        Ok(())
    }

    /// Migrate from monolithic file to per-model files.
    ///
    /// If the monolithic `nn_verify_status.json` exists:
    /// 1. Load it
    /// 2. Split and save per-model files
    /// 3. Rename the monolithic file to `.bak`
    ///
    /// If per-model files already exist, this is a no-op.
    pub fn migrate_to_per_model(workspace_root: &Path) -> Result<(), VerifyError> {
        let monolithic = workspace_root.join("nn_verify_status.json");

        // Check if already migrated: at least one per-model file exists.
        let any_per_model = MODEL_CATEGORIES
            .iter()
            .any(|model| model_status_path(workspace_root, model).exists());

        if any_per_model {
            return Ok(());
        }

        if !monolithic.exists() {
            return Ok(());
        }

        let status = Self::load(&monolithic)?;
        status.save_per_model(workspace_root)?;

        // Keep the monolithic file as backup.
        let backup = workspace_root.join("nn_verify_status.json.bak");
        std::fs::rename(&monolithic, &backup).map_err(VerifyError::Io)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soundness_compat::VerificationSoundnessMode;
    use crate::verify_types::PropMethod;
    use crate::{
        InputBoundsRecord, KernelStatus, OutputBoundsRecord, ParamInputRecord, VerifyOutcome,
    };

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
        assert_eq!(model_for_kernel("glm_self_attention"), "glm");
        assert_eq!(model_for_kernel("glm_decoder_block"), "glm");
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

    #[test]
    fn test_mark_stale_updates_latest_and_history() {
        let mut status = VerifyStatus::default();
        status
            .kernels
            .insert("kokoro_dec".to_string(), make_test_entry(false, None));
        status
            .history
            .insert("kokoro_dec".to_string(), vec![make_test_entry(false, None)]);

        status
            .mark_stale("kokoro_dec", "superseded by traced production segment")
            .expect("mark_stale");

        let latest = status.kernel("kokoro_dec").expect("latest entry");
        assert!(latest.stale);
        assert_eq!(
            latest.stale_reason.as_deref(),
            Some("superseded by traced production segment")
        );

        let hist = status.history_for("kokoro_dec").expect("history");
        assert!(hist[0].stale);
        assert_eq!(
            hist[0].stale_reason.as_deref(),
            Some("superseded by traced production segment")
        );
    }
}
