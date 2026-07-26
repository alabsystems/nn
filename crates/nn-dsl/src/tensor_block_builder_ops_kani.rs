// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harness module declarations for composite tensor op builders.
//!
//! Groups MatMul, Attention, Embedding, LSTM, Transpose, and Gated DeltaNet
//! builder Kani harnesses. Part of #729 dvoice epic.
//!
//! Extracted from `tensor_block_builder_ops.rs` to keep files under 500 lines.

#[path = "matmul_kani_builder_tests.rs"]
mod matmul_kani_builder;

#[path = "attention_kani_builder_tests.rs"]
mod attention_kani_builder;

#[path = "embedding_kani_builder_tests.rs"]
mod embedding_kani_builder;

#[path = "lstm_kani_builder_tests.rs"]
mod lstm_kani_builder;

#[path = "transpose_kani_builder_tests.rs"]
mod transpose_kani_builder;

#[path = "gated_delta_net_kani_monolithic_tests.rs"]
mod gated_delta_net_kani_monolithic;
