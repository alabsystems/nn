// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![allow(dead_code, unreachable_pub)]

//! Consolidated model forward pass tests: GPU forward for GLM5, Qwen3,
//! WeSpeaker, Whisper, Silero VAD, BF16, and LSTM production parity.

mod test_utils;

#[path = "model_forward/bf16_model_integration.rs"]
mod bf16_model_integration;
#[path = "model_forward/glm5_gpu_forward.rs"]
mod glm5_gpu_forward;
#[path = "model_forward/lstm_production_parity.rs"]
mod lstm_production_parity;
#[path = "model_forward/model_gpu_parity.rs"]
mod model_gpu_parity;
#[path = "model_forward/qwen3_gpu_forward.rs"]
mod qwen3_gpu_forward;
#[path = "model_forward/silero_vad_e2e.rs"]
mod silero_vad_e2e;
#[path = "model_forward/silero_vad_e2e_contract.rs"]
mod silero_vad_e2e_contract;
#[path = "model_forward/wespeaker_gpu_forward.rs"]
mod wespeaker_gpu_forward;
#[path = "model_forward/whisper_gpu_forward.rs"]
mod whisper_gpu_forward;
#[path = "model_forward/whisper_metal_parity.rs"]
mod whisper_metal_parity;
