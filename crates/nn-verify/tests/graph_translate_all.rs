// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated graph translation tests — all categories in one binary.
//!
//! Merges 53 individual `graph_translate_*.rs` test files into a single test
//! binary to eliminate 51 redundant NY link steps (53 → 1).
//!
//! Categories (53 modules total):
//!   Scalar / Base (1): base
//!   Conv (7): conv1d, conv1d_crown, conv1d_dilated, conv2d, conv2d_crown,
//!     conv_transpose_1d, conv_transpose_1d_crown
//!   Elementwise (13): adain, adain_fused, binary_add, binary_mul, gelu, glu,
//!     native, ops, ops_extended, piecewise, sigmoid, silu_mul, softmax
//!   Norm (7): batch_norm, group_norm, group_norm_numerical, layer_norm,
//!     layer_norm_monolithic, native_norm, rms_norm
//!   Tensor (8): tensor, tensor_multi, tensor_multi_var, tensor_norm,
//!     tensor_norm_affine, tensor_norm_constfold, tensor_norm_forward_mode,
//!     tensor_norm_forward_mode_pipeline
//!   Composite (11): attention, catchall, catchall_unary, concat, embedding,
//!     linear, lstm, lstm_validation, matmul, narrow, transpose
//!   Structural (5): structural_ops, structural_ops_concat,
//!     structural_ops_ibp, structural_ops_ibp_reshape_concat,
//!     structural_ops_ibp_stack
//!
//! Part of #1982.

// Shared helpers (e.g. tensor_norm_forward_mode_pipeline.rs) are included
// both directly and transitively. Inherent to the #[path] pattern.
#![allow(clippy::duplicate_mod)]

mod common;

// ── Scalar / Base ─────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_base.rs"]
mod base;

// ── Conv ────────────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_conv1d.rs"]
mod conv1d;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_conv1d_crown.rs"]
mod conv1d_crown;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_conv1d_dilated.rs"]
mod conv1d_dilated;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_conv2d.rs"]
mod conv2d;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_conv2d_crown.rs"]
mod conv2d_crown;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_conv_transpose_1d.rs"]
mod conv_transpose_1d;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_conv_transpose_1d_crown.rs"]
mod conv_transpose_1d_crown;

// ── Elementwise / Activation ────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_adain.rs"]
mod adain;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_adain_fused.rs"]
mod adain_fused;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_binary_add.rs"]
mod binary_add;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_binary_mul.rs"]
mod binary_mul;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_gelu.rs"]
mod gelu;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_glu.rs"]
mod glu;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_native.rs"]
mod native;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_ops.rs"]
mod ops;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_ops_extended.rs"]
mod ops_extended;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_piecewise.rs"]
mod piecewise;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_sigmoid.rs"]
mod sigmoid;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_silu_mul.rs"]
mod silu_mul;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_softmax.rs"]
mod softmax;

// ── Norm ────────────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_batch_norm.rs"]
mod batch_norm;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_group_norm.rs"]
mod group_norm;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_group_norm_numerical.rs"]
mod group_norm_numerical;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_layer_norm.rs"]
mod layer_norm;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_layer_norm_monolithic.rs"]
mod layer_norm_monolithic;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_native_norm.rs"]
mod native_norm;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_rms_norm.rs"]
mod rms_norm;

// ── Tensor ──────────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_tensor.rs"]
mod tensor;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_tensor_multi.rs"]
mod tensor_multi;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_tensor_multi_var.rs"]
mod tensor_multi_var;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_tensor_norm.rs"]
mod tensor_norm;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_tensor_norm_affine.rs"]
mod tensor_norm_affine;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_tensor_norm_constfold.rs"]
mod tensor_norm_constfold;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_tensor_norm_forward_mode.rs"]
mod tensor_norm_forward_mode;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_tensor_norm_forward_mode_pipeline.rs"]
mod tensor_norm_forward_mode_pipeline;

// ── Composite ───────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_attention.rs"]
mod attention;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_catchall.rs"]
mod catchall;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_catchall_unary.rs"]
mod catchall_unary;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_concat.rs"]
mod concat;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_embedding.rs"]
mod embedding;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_linear.rs"]
mod linear;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_lstm.rs"]
mod lstm;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_lstm_validation.rs"]
mod lstm_validation;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_matmul.rs"]
mod matmul;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_narrow.rs"]
mod narrow;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_transpose.rs"]
mod transpose;

// ── Structural ──────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_structural_ops.rs"]
mod structural_ops;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_structural_ops_concat.rs"]
mod structural_ops_concat;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_structural_ops_ibp.rs"]
mod structural_ops_ibp;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_structural_ops_ibp_reshape_concat.rs"]
mod structural_ops_ibp_reshape_concat;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_translate_structural_ops_ibp_stack.rs"]
mod structural_ops_ibp_stack;

// ── Native elementwise ───────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/graph_tensor_native_elementwise.rs"]
mod native_elementwise;
