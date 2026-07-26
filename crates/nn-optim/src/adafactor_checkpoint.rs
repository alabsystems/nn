// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint save/load implementation for [`super::AdaFactor`].

use std::collections::HashMap;

use nn_core::dyn_tensor::DynTensor;

use crate::checkpoint::{OptimizerCheckpoint, OptimizerSnapshot};
use crate::error::{OptimError, Result};

use super::{AdaFactor, AdaFactorConfig};

/// Restore an optional tensor from a snapshot, validating shape and finiteness.
///
/// When `slot` is `None` (state not yet initialized), validates against
/// `expected_dims` from the variable shape. This prevents loading a
/// checkpoint with mismatched shapes into uninitialized optimizer state.
pub(super) fn restore_tensor(
    tensors: &HashMap<String, DynTensor>,
    key: &str,
    slot: &mut Option<DynTensor>,
    expected_dims: &[usize],
) -> Result<()> {
    if let Some(loaded) = tensors.get(key) {
        let dims_to_check = if let Some(ref existing) = slot {
            existing.dims().to_vec()
        } else {
            expected_dims.to_vec()
        };
        if loaded.dims() != dims_to_check {
            return Err(OptimError::CheckpointShapeMismatch {
                key: key.to_string(),
                expected: dims_to_check,
                got: loaded.dims().to_vec(),
            });
        }
        crate::error::validate_checkpoint_tensor(loaded, key)?;
        *slot = Some(loaded.clone());
    }
    Ok(())
}

/// Restore AdaFactor config fields from checkpoint metadata and validate.
pub(super) fn restore_config(
    config: &mut AdaFactorConfig,
    metadata: &serde_json::Value,
) -> Result<()> {
    if let Some(v) = metadata.get("lr").and_then(serde_json::Value::as_f64) {
        config.lr = v;
    }
    if let Some(v) = metadata
        .get("relative_step")
        .and_then(serde_json::Value::as_bool)
    {
        config.relative_step = v;
    }
    if let Some(v) = metadata.get("eps_rms").and_then(serde_json::Value::as_f64) {
        config.eps_rms = v;
    }
    if let Some(v) = metadata
        .get("eps_denom")
        .and_then(serde_json::Value::as_f64)
    {
        config.eps_denom = v;
    }
    if let Some(v) = metadata
        .get("decay_rate")
        .and_then(serde_json::Value::as_f64)
    {
        config.decay_rate = v;
    }
    if let Some(json_val) = metadata.get("beta1") {
        config.beta1 = json_val.as_f64(); // null → None, number → Some
    }
    if let Some(v) = metadata
        .get("weight_decay")
        .and_then(serde_json::Value::as_f64)
    {
        config.weight_decay = v;
    }
    // Validate restored config — same invariants as new()
    crate::error::validate_lr(config.lr)?;
    if let Some(b1) = config.beta1 {
        if !(0.0..1.0).contains(&b1) {
            return Err(OptimError::InvalidParam {
                param: "beta1",
                reason: format!("must be in [0, 1), got {b1}"),
            });
        }
    }
    if config.eps_denom <= 0.0 || !config.eps_denom.is_finite() {
        return Err(OptimError::InvalidParam {
            param: "eps_denom",
            reason: format!("must be positive and finite, got {}", config.eps_denom),
        });
    }
    if config.eps_rms <= 0.0 || !config.eps_rms.is_finite() {
        return Err(OptimError::InvalidParam {
            param: "eps_rms",
            reason: format!("must be positive and finite, got {}", config.eps_rms),
        });
    }
    if !config.decay_rate.is_finite() {
        return Err(OptimError::InvalidParam {
            param: "decay_rate",
            reason: format!("must be finite, got {}", config.decay_rate),
        });
    }
    crate::error::validate_weight_decay(config.weight_decay)?;
    Ok(())
}

impl OptimizerCheckpoint for AdaFactor {
    fn save_checkpoint(&self) -> Result<OptimizerSnapshot> {
        let mut tensors = HashMap::new();
        for (i, state) in self.states.iter().enumerate() {
            if let Some(ref m) = state.first_moment {
                tensors.insert(format!("adafactor_{i}_m"), m.clone());
            }
            if let Some(ref v) = state.second_moment_full {
                tensors.insert(format!("adafactor_{i}_v_full"), v.clone());
            }
            if let Some(ref r) = state.row_factor {
                tensors.insert(format!("adafactor_{i}_row"), r.clone());
            }
            if let Some(ref c) = state.col_factor {
                tensors.insert(format!("adafactor_{i}_col"), c.clone());
            }
        }
        let metadata = serde_json::json!({
            "type": "AdaFactor",
            "step": self.step_t,
            "lr": self.config.lr,
            "relative_step": self.config.relative_step,
            "eps_rms": self.config.eps_rms,
            "eps_denom": self.config.eps_denom,
            "decay_rate": self.config.decay_rate,
            "beta1": self.config.beta1,
            "weight_decay": self.config.weight_decay,
        });
        Ok(OptimizerSnapshot { tensors, metadata })
    }

    fn load_checkpoint(&mut self, snapshot: &OptimizerSnapshot) -> Result<()> {
        let step = snapshot
            .metadata
            .get("step")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        self.step_t = usize::try_from(step)
            .map_err(|_| OptimError::CheckpointStepOverflow { step: step as i64 })?;
        restore_config(&mut self.config, &snapshot.metadata)?;
        for (i, state) in self.states.iter_mut().enumerate() {
            let var_dims = state.var.dims()?;
            // first_moment and second_moment_full have same shape as var
            restore_tensor(
                &snapshot.tensors,
                &format!("adafactor_{i}_m"),
                &mut state.first_moment,
                &var_dims,
            )?;
            restore_tensor(
                &snapshot.tensors,
                &format!("adafactor_{i}_v_full"),
                &mut state.second_moment_full,
                &var_dims,
            )?;
            // row_factor: var dims with last dim = 1
            let mut row_dims = var_dims.clone();
            if row_dims.len() >= 2 {
                let last = row_dims.len() - 1;
                row_dims[last] = 1;
            }
            restore_tensor(
                &snapshot.tensors,
                &format!("adafactor_{i}_row"),
                &mut state.row_factor,
                &row_dims,
            )?;
            // col_factor: var dims with second-to-last dim = 1
            let mut col_dims = var_dims;
            if col_dims.len() >= 2 {
                let second_last = col_dims.len() - 2;
                col_dims[second_last] = 1;
            }
            restore_tensor(
                &snapshot.tensors,
                &format!("adafactor_{i}_col"),
                &mut state.col_factor,
                &col_dims,
            )?;
        }
        Ok(())
    }
}
