// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SMT verification backend using ay.
//!
//! Translates `KernelDef` IR to ay `AYProgram` using Real arithmetic,
//! with uninterpreted function (UF) approximations for transcendental
//! operations (sin, cos, exp, etc.) in Phase A.
//!
//! # Phase A (current)
//!
//! Transcendental functions are encoded as uninterpreted functions with
//! axiomatic range constraints (e.g. `-1 <= sin_approx(x) <= 1`). Results
//! are recorded with `encoding = UfApprox` to distinguish from exact proofs.
//!
//! # Phase B (engineering task)
//!
//! `ay-bindings` now exposes IEEE 754 floating-point sorts via `ay-theories/fp`
//! (40+ methods in `expr/fp.rs`). Switching the Snake path to exact FP encoding
//! is an nn-verify engineering task, no longer blocked on ay upstream.
//! Until that migration, transcendental-kernel SMT proofs remain marked as
//! approximation.

mod error;
mod prove;
mod prove_multi;
mod snake_uf;
mod translate;
mod translate_linearity;
mod translate_node;
mod translate_real;
mod translate_uf;
mod translated_kernel;
pub(crate) mod ay_activation_properties;
pub(crate) mod ay_attention_mechanism_properties;
pub(crate) mod ay_broadcast_reduction_properties;
pub(crate) mod ay_convolution_properties;
pub(crate) mod ay_data_pipeline_properties;
pub(crate) mod ay_embedding_properties;
pub(crate) mod ay_fp_snake;
pub(crate) mod ay_gradient_proofs;
pub(crate) mod ay_linear_algebra_properties;
pub(crate) mod ay_loss_function_properties;
pub(crate) mod ay_matrix_decomposition_vlm;
pub(crate) mod ay_normalization_properties;
pub(crate) mod ay_optimizer_properties;
pub(crate) mod ay_pooling_properties;
pub(crate) mod ay_quantization_error;
pub(crate) mod ay_regularization_properties;
pub(crate) mod ay_reshape_view_properties;
pub(crate) mod ay_rope_properties;
pub(crate) mod ay_sequence_model_properties;
pub(crate) mod ay_sparse_attention_properties;
pub(crate) mod ay_tensor_decomposition_properties;
pub(crate) mod ay_training_loop_properties;
pub(crate) mod ay_weight_init_properties;

pub(crate) use error::SmtError;
pub use prove::{
    kernel_to_smt2, kernel_to_smt2_with_bounds, verify_kernel_smt, verify_kernel_smt_with_bounds,
};
pub use prove_multi::verify_kernel_smt_multi;
pub use translated_kernel::TranslatedKernel;

#[cfg(kani)]
#[path = "ay_encoding_kani.rs"]
mod ay_encoding_kani;

#[cfg(kani)]
#[path = "kani_ay_fp_snake.rs"]
mod kani_ay_fp_snake;
