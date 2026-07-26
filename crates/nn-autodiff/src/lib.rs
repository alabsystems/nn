// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reverse-mode automatic differentiation for nn.
//!
//! Provides gradient tape recording and backward pass execution in pure Rust,
//! matching candle's `Var`/`GradStore`/`backward()` architecture.
//!
//! # Architecture
//!
//! - **`Var`**: Trainable variable with interior mutability (optimizer updates).
//! - **`TrackedTensor`**: DynTensor + Op tracking for the computation graph.
//! - **`Op`**: Enum of all differentiable operations.
//! - **`GradStore`** + **`backward()`**: Reverse-mode gradient accumulation (D2).
//!
//! # Usage
//!
//! ```no_run
//! use nn_autodiff::{Var, TrackedTensor};
//! use nn_core::{DType, Device};
//! use std::sync::Arc;
//!
//! let var = Var::zeros(&[3, 4], DType::F32, &Device::Cpu).expect("alloc");
//! let x = Arc::new(TrackedTensor::from_var(&var).expect("lock"));
//! let y = x.sqr().expect("sqr");  // y = x^2, records Op::Sqr
//! ```

pub mod audio_losses;
pub(crate) mod backward_rules;
pub mod backward_rules_conv_attn;
pub mod error;
pub mod grad;
pub mod graph_viz;
pub mod hooked;
pub mod hooks;
pub mod loss_scaling;
pub mod op;
pub mod tracked;
pub mod train_loop;
pub mod train_module;
pub mod trainable;
pub mod trainable_kokoro_decoder;
pub mod var;
pub mod var_init;
pub mod var_map;

#[cfg(test)]
#[path = "backward_rules_tests.rs"]
mod backward_rules_tests;

#[cfg(test)]
#[path = "backward_integration_tests.rs"]
mod backward_integration_tests;

#[cfg(test)]
#[path = "var_tracked_tests.rs"]
mod var_tracked_tests;

#[cfg(test)]
#[path = "higher_order_tests.rs"]
mod higher_order_tests;

#[cfg(test)]
#[path = "higher_order_compose_tests.rs"]
mod higher_order_compose_tests;

#[cfg(test)]
#[path = "backward_gradient_check_tests.rs"]
mod backward_gradient_check_tests;

#[cfg(test)]
#[path = "tape_graph_tests.rs"]
mod tape_graph_tests;

#[cfg(test)]
#[path = "backward_rules_matmul_tests.rs"]
mod backward_rules_matmul_tests;

#[cfg(test)]
#[path = "backward_rules_nn_tests.rs"]
mod backward_rules_nn_tests;

#[cfg(test)]
#[path = "tape_mechanics_tests.rs"]
mod tape_mechanics_tests;

#[cfg(test)]
#[path = "backward_rules_extended_tests.rs"]
mod backward_rules_extended_tests;

#[cfg(test)]
#[path = "autodiff_tape_extended_tests.rs"]
mod autodiff_tape_extended_tests;

#[cfg(test)]
#[path = "tape_graph_extended_tests.rs"]
mod tape_graph_extended_tests;

#[cfg(test)]
#[path = "loss_extended_tests.rs"]
mod loss_extended_tests;

#[cfg(test)]
#[path = "autodiff_chain_rule_tests.rs"]
mod autodiff_chain_rule_tests;

#[cfg(test)]
#[path = "autodiff_loss_extended_tests.rs"]
mod autodiff_loss_extended_tests;

#[cfg(test)]
#[path = "autodiff_gradient_extended_tests.rs"]
mod autodiff_gradient_extended_tests;

#[cfg(kani)]
mod kani_backward_proofs;

#[cfg(kani)]
#[path = "kani_audio_losses.rs"]
mod kani_audio_losses;

#[cfg(kani)]
#[path = "kani_audio_losses_bounds.rs"]
mod kani_audio_losses_bounds;

#[cfg(kani)]
#[path = "kani_tracked_composite_ops.rs"]
mod kani_tracked_composite_ops;

#[cfg(kani)]
#[path = "kani_tracked_composite_ops_graph.rs"]
mod kani_tracked_composite_ops_graph;

#[cfg(kani)]
#[path = "kani_backward_rules.rs"]
mod kani_backward_rules;

#[cfg(kani)]
#[path = "kani_backward_rules_chain.rs"]
mod kani_backward_rules_chain;

#[cfg(kani)]
#[path = "kani_backward_rules_norm.rs"]
mod kani_backward_rules_norm;

#[cfg(kani)]
#[path = "kani_grad.rs"]
mod kani_grad;

#[cfg(kani)]
#[path = "kani_backward_rules_special.rs"]
mod kani_backward_rules_special;

#[cfg(kani)]
#[path = "kani_trainable_extra.rs"]
mod kani_trainable_extra;

#[cfg(kani)]
#[path = "kani_op.rs"]
mod kani_op;

#[cfg(kani)]
#[path = "kani_train_loop.rs"]
mod kani_train_loop;

#[cfg(kani)]
#[path = "kani_op_extended.rs"]
mod kani_op_extended;

#[cfg(kani)]
#[path = "kani_backward_rules_norm_extended.rs"]
mod kani_backward_rules_norm_extended;

#[cfg(kani)]
#[path = "kani_train_loop_extended.rs"]
mod kani_train_loop_extended;

#[cfg(kani)]
#[path = "kani_grad_extended.rs"]
mod kani_grad_extended;

#[cfg(kani)]
#[path = "kani_error_init_hooks.rs"]
mod kani_error_init_hooks;

#[cfg(kani)]
#[path = "kani_training_safety.rs"]
mod kani_training_safety;

#[cfg(kani)]
#[path = "kani_gradient_safety_extended.rs"]
mod kani_gradient_safety_extended;

pub use audio_losses::{
    feature_matching_loss, mel_spectrogram_loss, multi_res_stft_loss, stft_loss,
};
pub use backward_rules_conv_attn::{conv1d_backward, scaled_dot_product_attention_backward};
pub use error::{AutodiffError, Result};
pub use grad::{backward, backward_for_vars, GradStore};
pub use hooked::HookedModule;
pub use hooks::{ActivationCapture, HookHandle, HookableModule};
pub use loss_scaling::{cast_grad_to_f32, DynamicLossScaler, MixedPrecisionConfig};
pub use op::Op;
pub use tracked::{NodeId, TrackedTensor};
pub use train_loop::{
    compute_gradients, run_training_loop, EpochMetrics, SampleScore, TrainLoopConfig,
    TrainingSummary,
};
pub use train_module::{collect_vars, count_parameters, train_with_module, verify_forward_finite};
pub use trainable::{
    TrackedLstmState, TrainableBatchNorm, TrainableConv1d, TrainableConv2d,
    TrainableConvTranspose1d, TrainableEmbedding, TrainableGroupNorm, TrainableInstanceNorm,
    TrainableLayerNorm, TrainableLinear, TrainableLstm, TrainableModule,
    TrainableMultiHeadAttention, TrainableRmsNorm, TrainableSwiGlu,
};
pub use trainable_kokoro_decoder::{
    KokoroLoraConfig, LoraConv1d, MergedKokoroWeights, TrainableGenerator, TrainableKokoroDecoder,
    TrainableStage1ResBlk,
};
pub use var::{Var, VarId};
pub use var_init::{Fan, Init};
pub use var_map::VarMap;
