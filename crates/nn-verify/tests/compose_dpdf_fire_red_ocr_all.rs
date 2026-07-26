// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Standalone test binary for FireRed-OCR vision-language pipeline compose
//! verification tests.
//!
//! Verifies NY IBP and CROWN bound propagation through the FireRed-OCR
//! vision-language pipeline: vision encoder (patch embedding + Conv-BN-ReLU +
//! projection), language decoder (embedding + RMSNorm + causal self-attention +
//! SwiGLU FFN), cross-attention, CTC output head, and end-to-end composition.
//!
//! 30 tests across two modules:
//!
//! **fire_red_ocr** (14 tests):
//! - Vision encoder: patch embedding, Conv-BN-ReLU block, 2-block depth, projection
//! - Language decoder: embedding + position, self-attention, cross-attention, 2-layer stack
//! - Full pipeline: vision -> projection -> decoder, CTC head, end-to-end (IBP + CROWN)
//! - Verify-and-record: decoder block + end-to-end pipeline
//!
//! **firered_vision_lang** (16 tests):
//! - ViT visual encoder feature extraction (IBP + CROWN)
//! - Visual token projection to language space (IBP + CROWN)
//! - Language decoder self-attention per layer (IBP + CROWN)
//! - Cross-attention between visual and text tokens (IBP)
//! - RoPE position encoding for text tokens (IBP)
//! - SwiGLU FFN bounds per decoder layer (IBP + CROWN)
//! - Layer norm (RMSNorm sandwich) bounds (IBP + CROWN)
//! - LM head token prediction bounds (IBP)
//! - OCR character-level prediction bounds (IBP)
//! - Layout-aware position encoding bounds (IBP)
//! - Multi-resolution visual feature bounds (IBP)
//! - Full vision-to-OCR pipeline composition (IBP + CROWN)
//! - Confidence score per character bounds (IBP)
//! - Reading order prediction bounds (IBP)
//! - Verify-and-record: full pipeline + decoder layer
//!
//! Part of #4240: FireRed-OCR vision-language pipeline compose tests.

mod common;

/// FireRed-OCR vision-language pipeline compose tests (14 tests).
///
/// Tests vision encoder -> projection -> decoder -> CTC head bounds
/// propagation through the full FireRed-OCR Qwen3-VL-2B architecture.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_fire_red_ocr.rs"]
mod fire_red_ocr;

/// FireRed-OCR vision-language pipeline compose tests (16 tests).
///
/// Tests ViT encoder, visual token projection, decoder self-attention,
/// cross-attention, RoPE, SwiGLU FFN, RMSNorm sandwich, LM head,
/// OCR character prediction, layout position encoding, multi-resolution
/// features, full pipeline (IBP + CROWN), confidence scores, reading
/// order prediction, and verify-and-record pipelines.
///
/// Part of #4240: FireRed-OCR vision-language pipeline compose tests.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_firered_vision_lang.rs"]
mod firered_vision_lang;
