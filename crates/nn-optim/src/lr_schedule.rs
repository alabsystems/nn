// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Learning rate schedulers.
//!
//! Provides [`LrSchedule`] trait and common schedules:
//! - [`WarmupSchedule`]: linear warmup then constant
//! - [`CosineSchedule`]: cosine annealing with optional warmup

use crate::error::{OptimError, Result};
use crate::optimizer::Optimizer;
use nn_autodiff::GradStore;

/// Learning rate schedule that adjusts LR over training steps.
pub trait LrSchedule {
    /// Get the learning rate for the given step.
    fn lr_at_step(&self, step: usize) -> f64;
}

/// Linear warmup from 0 to `base_lr`, then constant.
///
/// For step < warmup_steps: lr = base_lr * step / warmup_steps
/// For step >= warmup_steps: lr = base_lr
#[derive(Debug, Clone, PartialEq)]
pub struct WarmupSchedule {
    base_lr: f64,
    warmup_steps: usize,
}

impl WarmupSchedule {
    /// Create a warmup schedule.
    ///
    /// - `base_lr`: target learning rate after warmup
    /// - `warmup_steps`: number of steps for linear warmup
    ///
    /// # Errors
    ///
    /// Returns `InvalidParam` if `base_lr` is negative or not finite.
    pub fn new(base_lr: f64, warmup_steps: usize) -> Result<Self> {
        if !base_lr.is_finite() || base_lr < 0.0 {
            return Err(OptimError::InvalidParam {
                param: "base_lr",
                reason: format!("must be non-negative and finite, got {base_lr}"),
            });
        }
        Ok(Self {
            base_lr,
            warmup_steps,
        })
    }

    /// Base learning rate (target after warmup).
    #[must_use]
    pub fn base_lr(&self) -> f64 {
        self.base_lr
    }

    /// Number of warmup steps.
    #[must_use]
    pub fn warmup_steps(&self) -> usize {
        self.warmup_steps
    }
}

impl LrSchedule for WarmupSchedule {
    fn lr_at_step(&self, step: usize) -> f64 {
        if self.warmup_steps == 0 {
            return self.base_lr;
        }
        if step < self.warmup_steps {
            self.base_lr * (step as f64 / self.warmup_steps as f64)
        } else {
            self.base_lr
        }
    }
}

/// Cosine annealing with optional linear warmup.
///
/// For step < warmup_steps: linear warmup from 0 to base_lr
/// For step >= warmup_steps: cosine decay from base_lr to min_lr
///
/// lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
/// where progress = (step - warmup) / (total - warmup)
#[derive(Debug, Clone, PartialEq)]
pub struct CosineSchedule {
    base_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
}

impl CosineSchedule {
    /// Create a cosine schedule.
    ///
    /// - `base_lr`: peak learning rate
    /// - `min_lr`: minimum learning rate at end of schedule
    /// - `warmup_steps`: linear warmup steps (0 for no warmup)
    /// - `total_steps`: total training steps (must be > warmup_steps)
    ///
    /// # Errors
    ///
    /// Returns `InvalidParam` if:
    /// - `base_lr` or `min_lr` is negative or not finite
    /// - `total_steps` is zero
    /// - `warmup_steps >= total_steps` (no room for cosine decay)
    /// - `min_lr > base_lr` (inverted schedule)
    pub fn new(base_lr: f64, min_lr: f64, warmup_steps: usize, total_steps: usize) -> Result<Self> {
        if !base_lr.is_finite() || base_lr < 0.0 {
            return Err(OptimError::InvalidParam {
                param: "base_lr",
                reason: format!("must be non-negative and finite, got {base_lr}"),
            });
        }
        if !min_lr.is_finite() || min_lr < 0.0 {
            return Err(OptimError::InvalidParam {
                param: "min_lr",
                reason: format!("must be non-negative and finite, got {min_lr}"),
            });
        }
        if total_steps == 0 {
            return Err(OptimError::InvalidParam {
                param: "total_steps",
                reason: "must be > 0".to_string(),
            });
        }
        if warmup_steps >= total_steps {
            return Err(OptimError::InvalidParam {
                param: "warmup_steps",
                reason: format!("must be < total_steps ({total_steps}), got {warmup_steps}"),
            });
        }
        if min_lr > base_lr {
            return Err(OptimError::InvalidParam {
                param: "min_lr",
                reason: format!("must be <= base_lr ({base_lr}), got {min_lr}"),
            });
        }
        Ok(Self {
            base_lr,
            min_lr,
            warmup_steps,
            total_steps,
        })
    }

    /// Peak learning rate.
    #[must_use]
    pub fn base_lr(&self) -> f64 {
        self.base_lr
    }

    /// Minimum learning rate.
    #[must_use]
    pub fn min_lr(&self) -> f64 {
        self.min_lr
    }

    /// Total training steps.
    #[must_use]
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
}

impl LrSchedule for CosineSchedule {
    fn lr_at_step(&self, step: usize) -> f64 {
        // Warmup phase
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.base_lr * (step as f64 / self.warmup_steps as f64);
        }

        // Past total steps: clamp to min_lr
        if step >= self.total_steps {
            return self.min_lr;
        }

        // Cosine annealing phase
        // warmup_steps < total_steps is enforced by new(), so this is always > 0.
        let decay_steps = self.total_steps - self.warmup_steps;
        let progress = (step - self.warmup_steps) as f64 / decay_steps as f64;
        self.min_lr
            + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (progress * std::f64::consts::PI).cos())
    }
}

/// Apply a learning rate schedule to an optimizer step.
///
/// Sets the optimizer's learning rate according to the schedule, then
/// performs the optimization step.
pub fn step_with_schedule<O: Optimizer>(
    optimizer: &mut O,
    grads: &GradStore,
    schedule: &dyn LrSchedule,
    step: usize,
) -> Result<()> {
    optimizer.set_learning_rate(schedule.lr_at_step(step))?;
    optimizer.step(grads)
}

#[cfg(test)]
#[path = "lr_schedule_tests.rs"]
mod tests;
