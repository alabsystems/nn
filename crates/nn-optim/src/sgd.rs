// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SGD optimizer with optional momentum and weight decay.
//!
//! Update rule (with momentum):
//! ```text
//! v_t = momentum * v_{t-1} + grad
//! theta = theta - lr * v_t
//! ```
//!
//! With weight decay (L2 regularization applied to gradient):
//! ```text
//! grad = grad + weight_decay * theta
//! ```

use std::collections::HashMap;

use crate::checkpoint::{OptimizerCheckpoint, OptimizerSnapshot};
use crate::error::{OptimError, Result};
use crate::optimizer::Optimizer;
use nn_autodiff::{GradStore, Var};
use nn_core::dyn_tensor::DynTensor;

/// Configuration for the SGD optimizer.
///
/// Defaults: lr=1e-2, momentum=0.0, weight_decay=0.0.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SgdConfig {
    /// Learning rate.
    pub lr: f64,
    /// Momentum coefficient (0.0 = no momentum).
    pub momentum: f64,
    /// Weight decay (L2 regularization) coefficient.
    pub weight_decay: f64,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            momentum: 0.0,
            weight_decay: 0.0,
        }
    }
}

/// SGD optimizer with optional momentum and weight decay.
#[derive(Debug)]
pub struct Sgd {
    vars: Vec<Var>,
    velocities: Vec<Option<DynTensor>>,
    config: SgdConfig,
}

impl Sgd {
    /// Create a new SGD optimizer for the given trainable variables.
    ///
    /// # Errors
    ///
    /// Returns `InvalidParam` if `lr` is negative or not finite, `momentum`
    /// is negative or not finite, or `weight_decay` is negative or not finite.
    #[must_use = "optimizer must be stored to perform training steps"]
    pub fn new(vars: Vec<Var>, config: SgdConfig) -> Result<Self> {
        crate::error::validate_lr(config.lr)?;
        if !config.momentum.is_finite() || config.momentum < 0.0 {
            return Err(OptimError::InvalidParam {
                param: "momentum",
                reason: format!("must be non-negative and finite, got {}", config.momentum),
            });
        }
        crate::error::validate_weight_decay(config.weight_decay)?;
        let n = vars.len();
        Ok(Self {
            vars,
            velocities: vec![None; n],
            config,
        })
    }

    /// Momentum coefficient.
    #[must_use]
    pub fn momentum(&self) -> f64 {
        self.config.momentum
    }

    /// Weight decay coefficient.
    #[must_use]
    pub fn weight_decay(&self) -> f64 {
        self.config.weight_decay
    }

    /// Configuration reference.
    #[must_use]
    pub fn config(&self) -> &SgdConfig {
        &self.config
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        let lr = self.config.lr;
        let momentum = self.config.momentum;
        let weight_decay = self.config.weight_decay;

        for (i, var) in self.vars.iter().enumerate() {
            let grad = match grads.get(var) {
                Some(g) => g.clone(),
                None => continue,
            };
            crate::error::validate_gradient(&grad)?;

            // Apply weight decay to gradient: grad += wd * theta
            let grad = if weight_decay > 0.0 {
                grad.add(&var.data()?.mul_scalar(weight_decay)?)?
            } else {
                grad
            };

            // Apply momentum
            let update = if momentum > 0.0 {
                let v = match &self.velocities[i] {
                    Some(prev) => prev.mul_scalar(momentum)?.add(&grad)?,
                    None => grad.clone(),
                };
                self.velocities[i] = Some(v.clone());
                v
            } else {
                grad
            };

            // theta = theta - lr * update
            let new_data = var.data()?.sub(&update.mul_scalar(lr)?)?;
            crate::error::validate_update(&new_data)?;
            var.set(&new_data)?;
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) -> Result<()> {
        crate::error::validate_lr(lr)?;
        self.config.lr = lr;
        Ok(())
    }
}

impl OptimizerCheckpoint for Sgd {
    fn save_checkpoint(&self) -> Result<OptimizerSnapshot> {
        let mut tensors = HashMap::new();
        for (i, vel) in self.velocities.iter().enumerate() {
            if let Some(v) = vel {
                tensors.insert(format!("sgd_{i}_velocity"), v.clone());
            }
        }
        let metadata = serde_json::json!({
            "type": "Sgd",
            "lr": self.config.lr,
            "momentum": self.config.momentum,
            "weight_decay": self.config.weight_decay,
        });
        Ok(OptimizerSnapshot { tensors, metadata })
    }

    fn load_checkpoint(&mut self, snapshot: &OptimizerSnapshot) -> Result<()> {
        // Restore config from metadata
        if let Some(v) = snapshot
            .metadata
            .get("lr")
            .and_then(serde_json::Value::as_f64)
        {
            self.config.lr = v;
        }
        if let Some(v) = snapshot
            .metadata
            .get("momentum")
            .and_then(serde_json::Value::as_f64)
        {
            self.config.momentum = v;
        }
        if let Some(v) = snapshot
            .metadata
            .get("weight_decay")
            .and_then(serde_json::Value::as_f64)
        {
            self.config.weight_decay = v;
        }
        // Validate restored config — same invariants as new()
        crate::error::validate_lr(self.config.lr)?;
        if !self.config.momentum.is_finite() || self.config.momentum < 0.0 {
            return Err(OptimError::InvalidParam {
                param: "momentum",
                reason: format!(
                    "must be non-negative and finite, got {}",
                    self.config.momentum
                ),
            });
        }
        crate::error::validate_weight_decay(self.config.weight_decay)?;
        // Restore velocity tensors
        for (i, vel) in self.velocities.iter_mut().enumerate() {
            let key = format!("sgd_{i}_velocity");
            if let Some(loaded) = snapshot.tensors.get(&key) {
                let expected_dims = if let Some(existing) = vel.as_ref() {
                    existing.dims().to_vec()
                } else {
                    self.vars[i].dims()?
                };
                if loaded.dims() != expected_dims {
                    return Err(OptimError::CheckpointShapeMismatch {
                        key: key.clone(),
                        expected: expected_dims,
                        got: loaded.dims().to_vec(),
                    });
                }
                crate::error::validate_checkpoint_tensor(loaded, &key)?;
                *vel = Some(loaded.clone());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "sgd_tests.rs"]
mod tests;
