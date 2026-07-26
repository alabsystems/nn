// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verified optimizers and training utilities for nn.
//!
//! Provides:
//! - [`Optimizer`] trait with `step()`, `backward_step()`, `learning_rate()`
//! - [`Sgd`] — SGD with momentum and weight decay
//! - [`SgdConfig`] — Configuration for SGD
//! - [`AdamW`] — AdamW with bias correction and decoupled weight decay
//! - [`AdamConfig`] — Configuration with defaults matching candle/PyTorch
//! - [`LrSchedule`] trait + [`WarmupSchedule`] + [`CosineSchedule`]
//! - [`step_with_schedule()`] — convenience for schedule-driven training
//! - [`LoraLinear`] — Low-Rank Adaptation for parameter-efficient fine-tuning (inference)
//! - [`TrainableLoraLinear`] — LoRA with gradient tracking for training
//! - [`LoraConfig`] — Configuration for LoRA injection targets
//! - [`GradScaler`] — Gradient scaling for mixed-precision training
//! - [`GradScalerConfig`] — Configuration for gradient scaler
//! - [`AdaFactor`] — AdaFactor with factored second moments (Shazeer & Stern, 2018)
//! - [`AdaFactorConfig`] — Configuration for AdaFactor
//! - [`clip_grad_norm`] — Clip total gradient L2 norm (matches `torch.nn.utils.clip_grad_norm_`)
//! - [`clip_grad_value`] — Clamp gradient elements to symmetric range (matches `torch.nn.utils.clip_grad_value_`)

pub mod adafactor;
pub mod adam;
pub mod checkpoint;
pub mod error;
pub mod grad_clip;
pub mod grad_scaler;
pub mod lora;
pub mod lr_schedule;
pub mod optimizer;
pub mod sgd;

#[cfg(kani)]
#[path = "kani_optim_proofs.rs"]
mod kani_optim_proofs;

#[cfg(kani)]
#[path = "kani_optim_proofs_adam.rs"]
mod kani_optim_proofs_adam;

#[cfg(kani)]
#[path = "kani_grad_scaler_proofs.rs"]
mod kani_grad_scaler_proofs;

#[cfg(kani)]
#[path = "kani_lr_schedule_proofs.rs"]
mod kani_lr_schedule_proofs;

#[cfg(kani)]
#[path = "kani_lora_proofs.rs"]
mod kani_lora_proofs;

#[cfg(kani)]
#[path = "kani_validate_gradient_proofs.rs"]
mod kani_validate_gradient_proofs;

#[cfg(kani)]
#[path = "kani_grad_clip_proofs.rs"]
mod kani_grad_clip_proofs;

#[cfg(kani)]
#[path = "kani_optim_proofs_advanced.rs"]
mod kani_optim_proofs_advanced;

#[cfg(kani)]
#[path = "kani_grad_scaler_proofs2.rs"]
mod kani_grad_scaler_proofs2;

#[cfg(kani)]
#[path = "kani_lora_proofs2.rs"]
mod kani_lora_proofs2;

#[cfg(kani)]
#[path = "kani_sgd_proofs2.rs"]
mod kani_sgd_proofs2;

#[cfg(kani)]
#[path = "kani_lr_schedule_proofs2.rs"]
mod kani_lr_schedule_proofs2;

#[cfg(kani)]
#[path = "kani_adafactor_adam_proofs.rs"]
mod kani_adafactor_adam_proofs;

#[cfg(kani)]
mod kani_adafactor;

#[cfg(kani)]
#[path = "kani_optim_proofs3.rs"]
mod kani_optim_proofs3;

#[cfg(kani)]
mod kani_adam;

#[cfg(kani)]
mod kani_grad_scaler;

#[cfg(kani)]
mod kani_lora;

#[cfg(kani)]
#[path = "kani_optim_wave11.rs"]
mod kani_optim_wave11;

#[cfg(kani)]
#[path = "kani_optimizer_update_safety.rs"]
mod kani_optimizer_update_safety;

pub use adafactor::{AdaFactor, AdaFactorConfig};
pub use adam::{AdamConfig, AdamW};
pub use checkpoint::{
    GradScalerState, OptimizerCheckpoint, OptimizerSnapshot, TrainingCheckpoint, TrainingMetadata,
};
pub use error::{OptimError, Result};
pub use grad_clip::{clip_grad_norm, clip_grad_value};
pub use grad_scaler::{GradScaler, GradScalerConfig};
pub use lora::{LoraConfig, LoraLinear, TrainableLoraLinear};
pub use lr_schedule::{step_with_schedule, CosineSchedule, LrSchedule, WarmupSchedule};
pub use optimizer::Optimizer;
pub use sgd::{Sgd, SgdConfig};

#[cfg(test)]
#[path = "optim_extra_tests.rs"]
mod optim_extra_tests;

#[cfg(test)]
#[path = "optim_expanded_tests.rs"]
mod optim_expanded_tests;

#[cfg(test)]
#[path = "sgd_momentum_tests.rs"]
mod sgd_momentum_tests;

#[cfg(test)]
#[path = "optimizer_tests.rs"]
mod optimizer_tests;

#[cfg(test)]
#[path = "schedule_scaler_tests.rs"]
mod schedule_scaler_tests;

#[cfg(test)]
#[path = "adafactor_convergence_tests.rs"]
mod adafactor_convergence_tests;

#[cfg(test)]
#[path = "lora_expanded_tests.rs"]
mod lora_expanded_tests;

#[cfg(test)]
#[path = "optim_checkpoint_extended_tests.rs"]
mod optim_checkpoint_extended_tests;

#[cfg(test)]
#[path = "optimizer_convergence_tests.rs"]
mod optimizer_convergence_tests;

#[cfg(test)]
#[path = "lr_schedule_extended_tests.rs"]
mod lr_schedule_extended_tests;

#[cfg(test)]
#[path = "checkpoint_convergence_tests.rs"]
mod checkpoint_convergence_tests;

#[cfg(test)]
#[path = "optimizer_extended_tests.rs"]
mod optimizer_extended_tests;

#[cfg(test)]
#[path = "scheduler_extended_tests.rs"]
mod scheduler_extended_tests;

#[cfg(test)]
#[path = "optim_config_extended_tests.rs"]
mod optim_config_extended_tests;
