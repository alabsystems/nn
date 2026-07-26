// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Convenience integration between [`TrainableModule`] and [`run_training_loop`].
//!
//! Bridges trainable modules (e.g., `TrainableLoraLinear` from nn-optim) with
//! the verification-guided training loop, reducing boilerplate for the common
//! case of fine-tuning a module with audio losses.
//!
//! # Architecture
//!
//! nn-autodiff does not depend on nn-optim or nn-tts-verify. This module
//! provides generic helpers that callers (who import both crates) use to
//! assemble the training pipeline. The caller supplies:
//!
//! - A `&dyn TrainableModule` (from nn-autodiff or nn-optim)
//! - A loss function (`FnMut` producing scalar loss from module output)
//! - An evaluation function (from nn-tts-verify)
//! - An optimizer step function (from nn-optim)
//!
//! # Example
//!
//! ```rust,ignore
//! use nn_autodiff::train_module::train_with_module;
//! use nn_autodiff::train_loop::{TrainLoopConfig, SampleScore};
//!
//! // Module: TrainableLoraLinear from nn-optim
//! let module: &dyn TrainableModule = &lora_layer;
//!
//! // Run verification-guided fine-tuning
//! let summary = train_with_module(
//!     &config,
//!     &module,
//!     corpus.len(),
//!     |epoch| { /* evaluate with nn-tts-verify */ vec![] },
//!     |sample_idx| {
//!         let input = get_input(sample_idx);
//!         let reference = get_reference(sample_idx);
//!         (input, reference)
//!     },
//!     |output, reference| {
//!         multi_res_stft_loss(&output, &reference, &[512, 1024, 2048])
//!     },
//!     |loss| { adam.backward_step(loss).map_err(Into::into) },
//! ).expect("training");
//! ```
//!
//! References:
//! - Design: nn#1726 (Self-Improving TTS via Verification-Guided Fine-Tuning)

use crate::error::{AutodiffError, Result};
use crate::train_loop::{run_training_loop, SampleScore, TrainLoopConfig, TrainingSummary};
use crate::trainable::TrainableModule;
use crate::TrackedTensor;
use std::sync::Arc;

/// Run verification-guided fine-tuning on a [`TrainableModule`].
///
/// This is a higher-level wrapper around [`run_training_loop`] that eliminates
/// the boilerplate of feeding sample inputs through a module and computing loss.
///
/// # Arguments
///
/// * `config` — Training loop configuration (epochs, curriculum fraction, etc.).
/// * `module` — The trainable module to fine-tune.
/// * `n_samples` — Total number of samples in the corpus.
/// * `evaluate` — Scores all samples using non-differentiable metrics.
/// * `get_sample` — Returns `(input, reference)` tensors for a sample index.
/// * `compute_loss` — Computes scalar loss from module output and reference.
/// * `optimizer_step` — Performs backward pass + optimizer step on the loss.
///
/// # Training loop per epoch
///
/// 1. **Evaluate**: Score all samples (non-differentiable)
/// 2. **Select**: Choose worst-performing samples as curriculum
/// 3. **For each curriculum sample**:
///    a. `get_sample(idx)` → `(input, reference)`
///    b. `module.forward(&input)` → `output`
///    c. `compute_loss(&output, &reference)` → scalar loss
///    d. `optimizer_step(&loss)` → backward + update
/// 4. **Report**: Log epoch metrics
pub fn train_with_module<E, S, L, O>(
    config: &TrainLoopConfig,
    module: &dyn TrainableModule,
    n_samples: usize,
    evaluate: E,
    mut get_sample: S,
    mut compute_loss: L,
    optimizer_step: O,
) -> Result<TrainingSummary>
where
    E: FnMut(usize) -> Vec<SampleScore>,
    S: FnMut(usize) -> (Arc<TrackedTensor>, Arc<TrackedTensor>),
    L: FnMut(&Arc<TrackedTensor>, &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>>,
    O: FnMut(&Arc<TrackedTensor>) -> Result<()>,
{
    run_training_loop(
        config,
        n_samples,
        evaluate,
        |sample_idx| {
            let (input, reference) = get_sample(sample_idx);
            let output = module.forward(&input)?;
            compute_loss(&output, &reference)
        },
        optimizer_step,
    )
}

/// Collect all trainable [`Var`]s from a module, cloned for optimizer registration.
///
/// Convenience for the common pattern of creating an optimizer from a module's
/// parameters. Returns owned `Var` clones suitable for `AdamW::new(vars, config)`.
///
/// # Example
///
/// ```rust,ignore
/// let vars = collect_vars(&lora_layer);
/// let mut adam = AdamW::new(vars, adam_config).expect("optimizer");
/// ```
pub fn collect_vars(module: &dyn TrainableModule) -> Vec<crate::var::Var> {
    module.vars().into_iter().cloned().collect()
}

/// Count the total number of trainable parameters in a module.
///
/// Useful for logging and verifying that LoRA rank settings produce
/// the expected parameter count.
pub fn count_parameters(module: &dyn TrainableModule) -> Result<usize> {
    let mut total = 0;
    for var in module.vars() {
        let dims = var.dims()?;
        total += dims.iter().product::<usize>();
    }
    Ok(total)
}

/// Verify that a module produces finite output for a given input.
///
/// Useful as a pre-training sanity check. Returns `Ok(())` if the output
/// is finite, or `Err` with details about non-finite values.
pub fn verify_forward_finite(
    module: &dyn TrainableModule,
    input: &Arc<TrackedTensor>,
) -> Result<()> {
    let output = module.forward(input)?;
    let tensor = output.tensor();
    let cpu_tensor =
        tensor
            .to_device(&nn_core::Device::Cpu)
            .map_err(|e| AutodiffError::InvalidConfig {
                op: "verify_forward_finite",
                reason: format!("failed to transfer output to CPU: {e}"),
            })?;
    let arr = cpu_tensor
        .to_f32_array()
        .map_err(|e| AutodiffError::InvalidConfig {
            op: "verify_forward_finite",
            reason: format!("failed to read output: {e}"),
        })?;
    let non_finite = arr.iter().filter(|v| !v.is_finite()).count();
    if non_finite > 0 {
        return Err(AutodiffError::InvalidConfig {
            op: "verify_forward_finite",
            reason: format!("{non_finite} non-finite values in module output"),
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "train_module_tests.rs"]
mod tests;
