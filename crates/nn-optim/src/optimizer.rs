// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Optimizer trait for updating trainable parameters.
//!
//! All optimizers take a [`GradStore`] from `backward()` and update [`Var`]
//! parameters in-place. The trait provides a convenience [`backward_step()`]
//! that combines backward + step.

use crate::error::Result;
use nn_autodiff::{GradStore, TrackedTensor};
use std::sync::Arc;

/// Optimizer that updates [`Var`] parameters using gradients.
///
/// Implementors: [`Sgd`](crate::Sgd), [`AdamW`](crate::AdamW), [`AdaFactor`](crate::AdaFactor).
pub trait Optimizer {
    /// Perform one optimization step given gradients.
    fn step(&mut self, grads: &GradStore) -> Result<()>;

    /// Convenience: backward + step in one call.
    ///
    /// Computes gradients of `loss` via reverse-mode AD, then updates all
    /// parameters tracked by this optimizer.
    fn backward_step(&mut self, loss: &Arc<TrackedTensor>) -> Result<()> {
        let grads = nn_autodiff::backward(loss)?;
        self.step(&grads)
    }

    /// Current learning rate.
    fn learning_rate(&self) -> f64;

    /// Update learning rate.
    ///
    /// # Errors
    ///
    /// Returns `InvalidParam` if `lr` is negative or not finite.
    fn set_learning_rate(&mut self, lr: f64) -> Result<()>;
}
