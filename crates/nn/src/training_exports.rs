// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Training re-exports for `nn::training`.
//!
//! Extracted from `lib.rs` for 450-line compliance.

// Automatic differentiation
pub use nn_autodiff::{
    backward, AutodiffError, Fan, GradStore, Init, NodeId, Op, TrackedLstmState, TrackedTensor,
    TrainableBatchNorm, TrainableConv1d, TrainableConv2d, TrainableConvTranspose1d,
    TrainableEmbedding, TrainableGroupNorm, TrainableInstanceNorm, TrainableLayerNorm,
    TrainableLinear, TrainableLstm, TrainableModule, TrainableMultiHeadAttention, TrainableRmsNorm,
    TrainableSwiGlu, Var, VarId, VarMap,
};

// Optimizers
pub use nn_optim::{
    AdaFactor, AdaFactorConfig, AdamConfig, AdamW, OptimError, Optimizer, Sgd, SgdConfig,
};

// Learning rate scheduling
pub use nn_optim::{step_with_schedule, CosineSchedule, LrSchedule, WarmupSchedule};

// LoRA (Low-Rank Adaptation)
pub use nn_optim::{LoraConfig, LoraLinear, TrainableLoraLinear};

// Gradient scaling for mixed-precision training
pub use nn_optim::{GradScaler, GradScalerConfig};

// Gradient clipping utilities
pub use nn_optim::{clip_grad_norm, clip_grad_value};

// Kokoro LoRA-wrapped decoder for singing fine-tuning (#4318)
pub use nn_autodiff::{
    KokoroLoraConfig, LoraConv1d, MergedKokoroWeights, TrainableGenerator, TrainableKokoroDecoder,
    TrainableStage1ResBlk,
};

// Audio losses for TTS training (mel spectrogram, multi-resolution STFT, feature matching)
pub use nn_autodiff::{
    feature_matching_loss, mel_spectrogram_loss, multi_res_stft_loss, stft_loss,
};

// Verification-guided training loop
pub use nn_autodiff::{
    compute_gradients, run_training_loop, EpochMetrics, SampleScore, TrainLoopConfig,
    TrainingSummary,
};

// TrainableModule integration bridge (generic training with closures)
pub use nn_autodiff::{collect_vars, count_parameters, train_with_module, verify_forward_finite};

// Causal tracing — activation capture and gradient-based attribution
pub use nn_autodiff::{
    backward_for_vars, ActivationCapture, HookHandle, HookableModule, HookedModule,
};

// Training checkpoint persistence
pub use nn_optim::{
    GradScalerState, OptimizerCheckpoint, OptimizerSnapshot, TrainingCheckpoint, TrainingMetadata,
};

// Result type aliases for training code
pub use nn_autodiff::Result as AutodiffResult;
pub use nn_optim::Result as OptimResult;
