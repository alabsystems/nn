// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Training checkpoint persistence for optimizer state.
//!
//! [`OptimizerCheckpoint`] allows saving and restoring optimizer state
//! (moment estimates, step counter) across process restarts.
//! [`TrainingCheckpoint`] bundles model weights + optimizer state + metadata
//! into a checkpoint directory.

use std::collections::HashMap;
use std::path::Path;

use nn_core::dyn_tensor::DynTensor;
use serde::{Deserialize, Serialize};

use crate::error::{OptimError, Result};

/// Snapshot of optimizer state for serialization.
#[derive(Debug)]
#[non_exhaustive]
pub struct OptimizerSnapshot {
    /// Named moment/velocity tensors (saved as safetensors).
    pub tensors: HashMap<String, DynTensor>,
    /// Scalar state: step count, LR, config, etc. (saved as JSON).
    pub metadata: serde_json::Value,
}

/// Trait for optimizers that can save and restore their state.
pub trait OptimizerCheckpoint {
    /// Export optimizer state as named tensors + metadata.
    fn save_checkpoint(&self) -> Result<OptimizerSnapshot>;

    /// Restore optimizer state from a previous snapshot.
    fn load_checkpoint(&mut self, snapshot: &OptimizerSnapshot) -> Result<()>;
}

/// Bundle for complete training state (model + optimizer + metadata).
pub struct TrainingCheckpoint;

/// Metadata for the training state JSON sidecar.
#[derive(Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TrainingMetadata {
    /// Global training step.
    pub step: usize,
    /// Current learning rate.
    pub lr: f64,
    /// Optional grad scaler state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grad_scaler: Option<GradScalerState>,
    /// Any extra user-provided metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

/// Serializable snapshot of GradScaler state.
#[derive(Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GradScalerState {
    pub scale: f64,
    pub growth_tracker: usize,
}

impl TrainingCheckpoint {
    /// Save complete training state to a directory.
    ///
    /// Creates:
    /// - `model.safetensors` — VarMap named tensors
    /// - `optimizer.safetensors` — optimizer moment/velocity tensors
    /// - `training_state.json` — step count, LR, grad scaler, extra metadata
    pub fn save(
        dir: impl AsRef<Path>,
        var_map: &nn_autodiff::VarMap,
        optimizer: &impl OptimizerCheckpoint,
        metadata: &TrainingMetadata,
    ) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        // Save model weights
        var_map.save_safetensors(dir.join("model.safetensors"))?;

        // Save optimizer state
        let snapshot = optimizer.save_checkpoint()?;
        if !snapshot.tensors.is_empty() {
            nn_core::dyn_tensor::save_safetensors(
                &snapshot.tensors,
                dir.join("optimizer.safetensors"),
            )?;
        }

        // Save metadata (step, lr, grad_scaler, optimizer metadata)
        let mut full_meta =
            serde_json::to_value(metadata).map_err(|e| OptimError::CheckpointSerde {
                reason: format!("failed to serialize metadata: {e}"),
            })?;
        // Merge optimizer-specific metadata
        if let serde_json::Value::Object(ref mut map) = full_meta {
            map.insert("optimizer".to_string(), snapshot.metadata);
        }
        let json =
            serde_json::to_string_pretty(&full_meta).map_err(|e| OptimError::CheckpointSerde {
                reason: format!("failed to serialize JSON: {e}"),
            })?;
        std::fs::write(dir.join("training_state.json"), json)?;

        Ok(())
    }

    /// Load complete training state from a directory.
    ///
    /// Returns the training metadata (including any extra fields).
    pub fn load(
        dir: impl AsRef<Path>,
        var_map: &mut nn_autodiff::VarMap,
        optimizer: &mut impl OptimizerCheckpoint,
    ) -> Result<TrainingMetadata> {
        let dir = dir.as_ref();

        // Load model weights
        var_map.load_safetensors(dir.join("model.safetensors"))?;

        // Load metadata
        let json_bytes = std::fs::read(dir.join("training_state.json"))?;
        let full_meta: serde_json::Value =
            serde_json::from_slice(&json_bytes).map_err(|e| OptimError::CheckpointSerde {
                reason: format!("failed to parse training_state.json: {e}"),
            })?;

        let metadata: TrainingMetadata =
            serde_json::from_value(full_meta.clone()).map_err(|e| OptimError::CheckpointSerde {
                reason: format!("invalid training metadata: {e}"),
            })?;

        // Load optimizer state
        let opt_tensors = if dir.join("optimizer.safetensors").exists() {
            nn_core::dyn_tensor::load_safetensors(dir.join("optimizer.safetensors"))?
        } else {
            HashMap::new()
        };
        let opt_metadata = full_meta
            .get("optimizer")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let snapshot = OptimizerSnapshot {
            tensors: opt_tensors,
            metadata: opt_metadata,
        };
        optimizer.load_checkpoint(&snapshot)?;

        Ok(metadata)
    }
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
