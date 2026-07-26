// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Whisper encoder/decoder/cross-attention/kv-cache/full/deep/mel/pipeline composition tests.
//!
//! Combines 9 Whisper test files into a single test binary to reduce
//! compilation overhead (9 NY link steps -> 1).
//!
//! Part of #1982, #3536, #3572, #3576, #4186, #4276, #4314.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_whisper_encoder.rs"]
mod whisper_encoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_whisper_decoder.rs"]
mod whisper_decoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_whisper_cross_attention.rs"]
mod whisper_cross_attention;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_whisper_kv_cache.rs"]
mod whisper_kv_cache;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_whisper_full.rs"]
mod whisper_full;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_whisper_deep.rs"]
mod whisper_deep;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_whisper_mel_spectrogram.rs"]
mod whisper_mel_spectrogram;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_whisper_pipeline.rs"]
mod whisper_pipeline;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_whisper_encoder_bounds.rs"]
mod whisper_encoder_bounds;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_whisper_gelu_reverify.rs"]
mod whisper_gelu_reverify;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_whisper_gelu_reverify_deep.rs"]
mod whisper_gelu_reverify_deep;
