// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint save/load implementation for [`super::AdamW`].
//!
//! Extracted from `adam.rs` to keep both files under 500 lines,
//! matching the `adafactor_checkpoint.rs` pattern.

use std::collections::HashMap;

use crate::checkpoint::{OptimizerCheckpoint, OptimizerSnapshot};
use crate::error::{OptimError, Result};

use super::{AdamConfig, AdamW};

/// Restore AdamW config fields from checkpoint metadata and validate.
pub(super) fn restore_adam_config(
    config: &mut AdamConfig,
    metadata: &serde_json::Value,
) -> Result<()> {
    if let Some(v) = metadata.get("lr").and_then(serde_json::Value::as_f64) {
        config.lr = v;
    }
    if let Some(v) = metadata.get("beta1").and_then(serde_json::Value::as_f64) {
        config.beta1 = v;
    }
    if let Some(v) = metadata.get("beta2").and_then(serde_json::Value::as_f64) {
        config.beta2 = v;
    }
    if let Some(v) = metadata.get("eps").and_then(serde_json::Value::as_f64) {
        config.eps = v;
    }
    if let Some(v) = metadata
        .get("weight_decay")
        .and_then(serde_json::Value::as_f64)
    {
        config.weight_decay = v;
    }
    // Validate restored config — same invariants as new()
    crate::error::validate_lr(config.lr)?;
    if !(0.0..1.0).contains(&config.beta1) {
        return Err(OptimError::InvalidParam {
            param: "beta1",
            reason: format!("must be in [0, 1), got {}", config.beta1),
        });
    }
    if !(0.0..1.0).contains(&config.beta2) {
        return Err(OptimError::InvalidParam {
            param: "beta2",
            reason: format!("must be in [0, 1), got {}", config.beta2),
        });
    }
    if config.eps <= 0.0 || !config.eps.is_finite() {
        return Err(OptimError::InvalidParam {
            param: "eps",
            reason: format!("must be finite and positive, got {}", config.eps),
        });
    }
    crate::error::validate_weight_decay(config.weight_decay)?;
    Ok(())
}

impl OptimizerCheckpoint for AdamW {
    fn save_checkpoint(&self) -> Result<OptimizerSnapshot> {
        let mut tensors = HashMap::new();
        for (i, state) in self.states.iter().enumerate() {
            tensors.insert(format!("adam_{i}_m"), state.first_moment.clone());
            tensors.insert(format!("adam_{i}_v"), state.second_moment.clone());
        }
        let metadata = serde_json::json!({
            "type": "AdamW",
            "step": self.step_t,
            "lr": self.config.lr,
            "beta1": self.config.beta1,
            "beta2": self.config.beta2,
            "eps": self.config.eps,
            "weight_decay": self.config.weight_decay,
        });
        Ok(OptimizerSnapshot { tensors, metadata })
    }

    fn load_checkpoint(&mut self, snapshot: &OptimizerSnapshot) -> Result<()> {
        // Restore step counter
        if let Some(step) = snapshot
            .metadata
            .get("step")
            .and_then(serde_json::Value::as_u64)
        {
            self.step_t = usize::try_from(step)
                .map_err(|_| OptimError::CheckpointStepOverflow { step: step as i64 })?;
        }
        restore_adam_config(&mut self.config, &snapshot.metadata)?;
        // Restore moment tensors
        for (i, state) in self.states.iter_mut().enumerate() {
            if let Some(m) = snapshot.tensors.get(&format!("adam_{i}_m")) {
                if m.dims() != state.first_moment.dims() {
                    return Err(OptimError::CheckpointShapeMismatch {
                        key: format!("adam_{i}_m"),
                        expected: state.first_moment.dims().to_vec(),
                        got: m.dims().to_vec(),
                    });
                }
                crate::error::validate_checkpoint_tensor(m, &format!("adam_{i}_m"))?;
                state.first_moment = m.clone();
            }
            if let Some(v) = snapshot.tensors.get(&format!("adam_{i}_v")) {
                if v.dims() != state.second_moment.dims() {
                    return Err(OptimError::CheckpointShapeMismatch {
                        key: format!("adam_{i}_v"),
                        expected: state.second_moment.dims().to_vec(),
                        got: v.dims().to_vec(),
                    });
                }
                crate::error::validate_checkpoint_tensor(v, &format!("adam_{i}_v"))?;
                state.second_moment = v.clone();
            }
        }
        Ok(())
    }
}
