#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated nn-whisper integration tests (5 → 1 binary).

#[allow(dead_code, unreachable_pub)]
#[path = "whisper/audio_integration.rs"]
mod audio_integration;

#[allow(dead_code, unreachable_pub)]
#[path = "whisper/block_attention_forward.rs"]
mod block_attention_forward;

#[allow(dead_code, unreachable_pub)]
#[path = "whisper/decode_integration.rs"]
mod decode_integration;

#[allow(dead_code, unreachable_pub)]
#[path = "whisper/safetensors_load.rs"]
mod safetensors_load;

#[allow(dead_code, unreachable_pub)]
#[path = "whisper/whisper_e2e.rs"]
mod whisper_e2e;

#[allow(dead_code, unreachable_pub)]
#[path = "whisper/encoder_decoder_integration.rs"]
mod encoder_decoder_integration;
