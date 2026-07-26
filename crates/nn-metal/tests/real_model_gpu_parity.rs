// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![allow(dead_code, unreachable_pub)]

//! Real-model GPU parity tests: CPU vs Metal with production-scale shapes.
//!
//! Tests key operations at shapes drawn from real models (Silero VAD, Kokoro,
//! HTDemucs, Whisper, Qwen3) to catch numerical divergence that only manifests
//! at production tensor dimensions. Includes:
//!
//! - **Full model forward:** Silero VAD with real safetensors weights
//! - **MatMul:** Real model shapes (transformer projections, attention)
//! - **Conv1d:** Encoder shapes from Kokoro/HTDemucs/Silero VAD
//! - **Softmax:** Attention-sized tensors from real models
//! - **LayerNorm:** Real model hidden dimensions (768, 1024, 2048)
//! - **LSTM:** Silero VAD LSTM cell dimensions
//! - **Attention block:** Multi-head attention at transformer scale
//!
//! Tests gated on `SILERO_VAD_WEIGHTS` / `KOKORO_WEIGHTS` env vars skip
//! gracefully when weights are unavailable.

mod test_utils;

#[path = "real_model_gpu_parity/attention_real_shapes.rs"]
mod attention_real_shapes;
#[path = "real_model_gpu_parity/conv1d_real_shapes.rs"]
mod conv1d_real_shapes;
#[path = "real_model_gpu_parity/large_vocab_softmax.rs"]
mod large_vocab_softmax;
#[path = "real_model_gpu_parity/layer_norm_real_shapes.rs"]
mod layer_norm_real_shapes;
#[path = "real_model_gpu_parity/lstm_real_shapes.rs"]
mod lstm_real_shapes;
#[path = "real_model_gpu_parity/matmul_real_shapes.rs"]
mod matmul_real_shapes;
#[path = "real_model_gpu_parity/silero_vad_full.rs"]
mod silero_vad_full;
#[path = "real_model_gpu_parity/softmax_real_shapes.rs"]
mod softmax_real_shapes;
