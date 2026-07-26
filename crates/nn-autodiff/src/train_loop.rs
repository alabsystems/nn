// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verification-guided training loop for TTS fine-tuning.
//!
//! Provides a generic training loop that integrates differentiable audio losses
//! with non-differentiable evaluation metrics. The caller supplies:
//!
//! - A **model forward function** (`FnMut` producing audio from input features)
//! - A **loss function** combining differentiable surrogates from [`crate::audio_losses`]
//! - An **evaluation function** scoring samples via non-differentiable metrics
//! - A **curriculum selector** choosing which samples to train on next
//!
//! This design keeps nn-autodiff independent of nn-tts-verify and nn-optim.
//! The caller (who has both crates) wires them together through the trait.
//!
//! # Training loop phases (per epoch)
//!
//! 1. **Evaluate**: Score all samples with non-differentiable metrics
//! 2. **Select**: Choose worst-performing samples as curriculum
//! 3. **Train**: Run differentiable loss + backward + optimizer step on curriculum
//! 4. **Report**: Log metrics for the epoch
//!
//! References:
//! - Design: nn#1726 (Self-Improving TTS via Verification-Guided Fine-Tuning)
//! - Kong et al. 2020, "HiFi-GAN" — multi-resolution STFT training
//! - Kumar et al. 2019, "MelGAN" — feature matching in GAN training

use crate::error::{AutodiffError, Result};
use crate::grad::backward;
use crate::TrackedTensor;
use std::sync::Arc;

/// Configuration for the verification-guided training loop.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TrainLoopConfig {
    /// Maximum number of epochs (outer loop iterations).
    pub max_epochs: usize,
    /// Fraction of corpus to select as curriculum each epoch.
    /// Must be in (0.0, 1.0].
    pub curriculum_fraction: f64,
    /// Stop early if evaluation score exceeds this threshold.
    /// `None` means no early stopping.
    pub target_score: Option<f64>,
    /// Log metrics every N epochs. 0 means never log.
    pub log_interval: usize,
}

impl Default for TrainLoopConfig {
    fn default() -> Self {
        Self {
            max_epochs: 10,
            curriculum_fraction: 0.1,
            target_score: None,
            log_interval: 1,
        }
    }
}

/// Metrics collected after each epoch.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Epoch number (0-indexed).
    pub epoch: usize,
    /// Mean training loss over curriculum samples this epoch.
    pub mean_loss: f64,
    /// Mean evaluation score over all samples (from non-differentiable metrics).
    pub mean_eval_score: f64,
    /// Number of samples selected as curriculum this epoch.
    pub curriculum_size: usize,
    /// Number of training steps (gradient updates) this epoch.
    pub train_steps: usize,
}

/// Summary of a completed training run.
#[derive(Debug, Clone)]
pub struct TrainingSummary {
    /// Per-epoch metrics.
    pub epoch_metrics: Vec<EpochMetrics>,
    /// Total number of gradient update steps across all epochs.
    pub total_steps: usize,
    /// Whether early stopping was triggered.
    pub early_stopped: bool,
    /// Final evaluation score (last epoch's mean_eval_score).
    pub final_score: f64,
}

/// Per-sample evaluation result from non-differentiable metrics.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SampleScore {
    /// Index into the sample corpus.
    pub index: usize,
    /// Quality score in [0, 1] where 1 is perfect.
    pub score: f64,
}

impl SampleScore {
    /// Create a new sample evaluation result.
    #[must_use]
    pub fn new(index: usize, score: f64) -> Self {
        Self { index, score }
    }
}

/// Run a verification-guided training loop.
///
/// This is the core integration point connecting differentiable training
/// (nn-autodiff audio losses) with non-differentiable evaluation
/// (nn-tts-verify metrics).
///
/// # Arguments
///
/// * `config` — Training loop configuration.
/// * `n_samples` — Total number of samples in the corpus.
/// * `evaluate` — Scores all samples using non-differentiable metrics.
///   Returns a `Vec<SampleScore>` with one entry per sample.
/// * `compute_loss` — Computes differentiable loss for a curriculum sample.
///   Takes a sample index, returns the scalar loss as a `TrackedTensor`.
/// * `optimizer_step` — Performs backward pass + optimizer step on the loss.
///   Takes the loss tensor, should call `backward()` and optimizer internally.
///
/// # Returns
///
/// [`TrainingSummary`] with per-epoch metrics and final evaluation score.
///
/// # Example
///
/// ```rust,ignore
/// use nn_autodiff::train_loop::{run_training_loop, TrainLoopConfig, SampleScore};
///
/// let config = TrainLoopConfig { max_epochs: 5, ..Default::default() };
///
/// let summary = run_training_loop(
///     &config,
///     corpus.len(),
///     |epoch| {
///         // Evaluate all samples with nn-tts-verify
///         corpus.iter().enumerate().map(|(i, sample)| {
///             let cert = verifier.verify(&sample.audio).expect("verify");
///             SampleScore { index: i, score: cert.pass_rate() }
///         }).collect()
///     },
///     |sample_idx| {
///         // Compute differentiable loss for one sample
///         let cand = synthesize(sample_idx);
///         let refr = get_reference(sample_idx);
///         multi_res_stft_loss(&cand, &refr, &[512, 1024, 2048])
///     },
///     |loss| {
///         // backward + optimizer step
///         adam.backward_step(&loss)
///     },
/// )?;
/// ```
pub fn run_training_loop<E, L, O>(
    config: &TrainLoopConfig,
    n_samples: usize,
    mut evaluate: E,
    mut compute_loss: L,
    mut optimizer_step: O,
) -> Result<TrainingSummary>
where
    E: FnMut(usize) -> Vec<SampleScore>,
    L: FnMut(usize) -> Result<Arc<TrackedTensor>>,
    O: FnMut(&Arc<TrackedTensor>) -> Result<()>,
{
    validate_config(config)?;

    if n_samples == 0 {
        return Err(AutodiffError::InvalidConfig {
            op: "run_training_loop",
            reason: "n_samples must be > 0".to_string(),
        });
    }

    let mut epoch_metrics = Vec::with_capacity(config.max_epochs);
    let mut total_steps = 0;
    let mut early_stopped = false;

    for epoch in 0..config.max_epochs {
        // Phase 1: Evaluate all samples
        let mut scores = evaluate(epoch);

        // Validate scores
        if scores.is_empty() {
            return Err(AutodiffError::InvalidConfig {
                op: "run_training_loop",
                reason: "evaluate returned empty scores".to_string(),
            });
        }

        let mean_eval = mean_score(&scores);

        // Check early stopping
        if let Some(target) = config.target_score {
            if mean_eval >= target {
                epoch_metrics.push(EpochMetrics {
                    epoch,
                    mean_loss: 0.0,
                    mean_eval_score: mean_eval,
                    curriculum_size: 0,
                    train_steps: 0,
                });
                early_stopped = true;
                break;
            }
        }

        // Phase 2: Select curriculum (worst-performing samples)
        let curriculum = select_curriculum(&mut scores, config.curriculum_fraction, n_samples);
        let curriculum_size = curriculum.len();

        // Phase 3: Train on curriculum samples
        let mut epoch_loss_sum = 0.0;
        let mut epoch_steps = 0;

        for &sample_idx in &curriculum {
            let loss = compute_loss(sample_idx)?;

            // Extract scalar loss value for logging
            let loss_scalar =
                loss.tensor()
                    .to_scalar::<f32>()
                    .map_err(|e| AutodiffError::InvalidConfig {
                        op: "run_training_loop",
                        reason: format!("failed to read loss value: {e}"),
                    })?;
            if loss_scalar.is_finite() {
                epoch_loss_sum += f64::from(loss_scalar);
            }

            // Backward + optimizer step
            optimizer_step(&loss)?;
            epoch_steps += 1;
        }

        total_steps += epoch_steps;

        let mean_loss = if epoch_steps > 0 {
            epoch_loss_sum / epoch_steps as f64
        } else {
            0.0
        };

        epoch_metrics.push(EpochMetrics {
            epoch,
            mean_loss,
            mean_eval_score: mean_eval,
            curriculum_size,
            train_steps: epoch_steps,
        });
    }

    let final_score = epoch_metrics
        .last()
        .map(|m| m.mean_eval_score)
        .unwrap_or(0.0);

    Ok(TrainingSummary {
        epoch_metrics,
        total_steps,
        early_stopped,
        final_score,
    })
}

/// Compute differentiable loss and return gradients (convenience for callers).
///
/// Wraps `backward()` for the common case where the caller just needs
/// the gradient store from a loss tensor.
pub fn compute_gradients(loss: &Arc<TrackedTensor>) -> Result<crate::GradStore> {
    backward(loss)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_config(config: &TrainLoopConfig) -> Result<()> {
    if config.max_epochs == 0 {
        return Err(AutodiffError::InvalidConfig {
            op: "run_training_loop",
            reason: "max_epochs must be > 0".to_string(),
        });
    }
    if config.curriculum_fraction <= 0.0 || config.curriculum_fraction > 1.0 {
        return Err(AutodiffError::InvalidConfig {
            op: "run_training_loop",
            reason: format!(
                "curriculum_fraction must be in (0.0, 1.0], got {}",
                config.curriculum_fraction
            ),
        });
    }
    if !config.curriculum_fraction.is_finite() {
        return Err(AutodiffError::InvalidConfig {
            op: "run_training_loop",
            reason: "curriculum_fraction must be finite".to_string(),
        });
    }
    if let Some(target) = config.target_score {
        if !target.is_finite() {
            return Err(AutodiffError::InvalidConfig {
                op: "run_training_loop",
                reason: "target_score must be finite".to_string(),
            });
        }
    }
    Ok(())
}

/// Select the worst-performing samples as curriculum.
///
/// Sorts scores ascending (worst first) and takes at least 1 sample
/// and at most `fraction * n_samples` samples.
fn select_curriculum(scores: &mut [SampleScore], fraction: f64, n_samples: usize) -> Vec<usize> {
    // Sort by score ascending (worst first)
    scores.sort_by(|a, b| a.score.total_cmp(&b.score));

    let count = ((n_samples as f64 * fraction).ceil() as usize)
        .max(1)
        .min(scores.len());
    scores[..count].iter().map(|s| s.index).collect()
}

/// Mean of sample scores, handling edge cases.
fn mean_score(scores: &[SampleScore]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let sum: f64 = scores.iter().map(|s| s.score).sum();
    sum / scores.len() as f64
}

#[cfg(test)]
#[path = "train_loop_tests.rs"]
mod tests;
