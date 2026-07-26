// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated miscellaneous composition tests.
//!
//! Combines 13 miscellaneous composition test files into a single test binary
//! to reduce compilation overhead (13 NY link steps → 1).
//!
//! - Sequential (2): compose_sequential, compose_sequential_dvoice
//! - Tensor chain (4): compose_tensor_chain, compose_tensor_chain_two_layer, compose_tensor_chain_simple, compose_model_to_graph
//! - Adversarial (2): compose_adversarial_phoneme_stability, compose_adversarial_robustness_verify
//! - Single: compose_residual_connection, compose_softmax_attention_phase23,
//!   compose_softplus_exp, compose_k2_k4_verification,
//!   compose_phoneme_certificate_phase16, compose_unicode_perturbation_crown
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

// ── Sequential ────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_sequential.rs"]
mod sequential;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_sequential_dvoice.rs"]
mod sequential_dvoice;

// ── Tensor chain ──────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_tensor_chain.rs"]
mod tensor_chain;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_tensor_chain_two_layer.rs"]
mod tensor_chain_two_layer;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_tensor_chain_simple.rs"]
mod tensor_chain_simple;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_model_to_graph.rs"]
mod model_to_graph;

// ── Adversarial ───────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_adversarial_phoneme_stability.rs"]
mod adversarial_phoneme;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_adversarial_robustness_verify.rs"]
mod adversarial_robustness;

// ── Other ─────────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_residual_connection.rs"]
mod residual_connection;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_softmax_attention_phase23.rs"]
mod softmax_attention;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_softplus_exp.rs"]
mod softplus_exp;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_k2_k4_verification.rs"]
mod k2_k4_verification;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_phoneme_certificate_phase16.rs"]
mod phoneme_certificate;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_unicode_perturbation_crown.rs"]
mod unicode_perturbation;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_upsample1d.rs"]
mod upsample1d;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_singing_pitch_bounds.rs"]
mod singing_pitch_bounds;

// ── Chained normalization (#2702) ────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_chained_norm.rs"]
mod chained_norm;

// ── AC3 counterfactual: Kahan vs naive (#2738) ──────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_counterfactual_kahan.rs"]
mod counterfactual_kahan;

// ── Normalization layers: BatchNorm + InstanceNorm + AdaIn (#3565) ──────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_normalization_layers.rs"]
mod normalization_layers;
