// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compile-time static assertions for OCR model architectures.
//!
//! These const assertions verify architectural invariants of PaddleOCR-VL and
//! FireRed-OCR (Qwen3-VL-2B based) at compile time. Dimension mismatches,
//! invalid GQA ratios, and incompatible FFN sizes are caught before any
//! runtime code executes.

// =============================================================================
// PaddleOCR-VL configuration constants
// =============================================================================
//
// Reference: PaddleOCR-VL architecture
// Vision encoder: ViT-B/16 style (768-dim, 12 heads, 12 layers)
// Decoder: 6-layer transformer decoder, vocab_size=6625 (Chinese + English + symbols)

const PADDLE_VISION_HIDDEN_DIM: usize = 768;
const PADDLE_VISION_NUM_HEADS: usize = 12;
const PADDLE_VISION_LAYERS: usize = 12;
const PADDLE_DECODER_HIDDEN_DIM: usize = 768;
const PADDLE_DECODER_LAYERS: usize = 6;
const PADDLE_VOCAB_SIZE: usize = 6625;
const PADDLE_VISION_HEAD_DIM: usize = PADDLE_VISION_HIDDEN_DIM / PADDLE_VISION_NUM_HEADS;

// -- PaddleOCR-VL const assertions --

// Vision encoder: hidden_dim must be divisible by num_heads
const _: () = assert!(
    PADDLE_VISION_HIDDEN_DIM.is_multiple_of(PADDLE_VISION_NUM_HEADS),
    "PaddleOCR-VL: vision hidden_dim must be divisible by num_heads"
);

// Vision encoder: head_dim is 64 (768 / 12)
const _: () = assert!(
    PADDLE_VISION_HEAD_DIM == 64,
    "PaddleOCR-VL: expected vision head_dim = 64"
);

// Vision encoder: layer count must be positive
const _: () = assert!(
    PADDLE_VISION_LAYERS > 0,
    "PaddleOCR-VL: vision layers must be > 0"
);

// Decoder: layer count must be positive
const _: () = assert!(
    PADDLE_DECODER_LAYERS > 0,
    "PaddleOCR-VL: decoder layers must be > 0"
);

// Decoder: hidden_dim must be positive
const _: () = assert!(
    PADDLE_DECODER_HIDDEN_DIM > 0,
    "PaddleOCR-VL: decoder hidden_dim must be > 0"
);

// Vocab size must be positive
const _: () = assert!(
    PADDLE_VOCAB_SIZE > 0,
    "PaddleOCR-VL: vocab_size must be > 0"
);

// Vision-decoder compatibility: vision output dim matches decoder input dim
// (PaddleOCR-VL uses same dim for both, no projection layer needed)
const _: () = assert!(
    PADDLE_VISION_HIDDEN_DIM == PADDLE_DECODER_HIDDEN_DIM,
    "PaddleOCR-VL: vision output dim must match decoder input dim"
);

// =============================================================================
// FireRed-OCR (Qwen3-VL-2B) configuration constants
// =============================================================================
//
// Reference: FireRed-OCR uses Qwen3-VL-2B as backbone
// Language model: 28-layer decoder-only transformer with GQA (12 heads, 2 KV heads)
// Vision encoder: ViT with 1280-dim, 16 heads

const FIRE_RED_HIDDEN_DIM: usize = 1536;
const FIRE_RED_NUM_HEADS: usize = 12;
const FIRE_RED_NUM_KV_HEADS: usize = 2;
const FIRE_RED_NUM_LAYERS: usize = 28;
const FIRE_RED_VISION_HIDDEN_DIM: usize = 1280;
const FIRE_RED_VISION_NUM_HEADS: usize = 16;
const FIRE_RED_VOCAB_SIZE: usize = 151_936;
const FIRE_RED_INTERMEDIATE_SIZE: usize = 8960;
const FIRE_RED_HEAD_DIM: usize = FIRE_RED_HIDDEN_DIM / FIRE_RED_NUM_HEADS;
const FIRE_RED_GQA_RATIO: usize = FIRE_RED_NUM_HEADS / FIRE_RED_NUM_KV_HEADS;
const FIRE_RED_VISION_HEAD_DIM: usize = FIRE_RED_VISION_HIDDEN_DIM / FIRE_RED_VISION_NUM_HEADS;

// -- FireRed-OCR language model const assertions --

// LM: hidden_dim must be divisible by num_heads
const _: () = assert!(
    FIRE_RED_HIDDEN_DIM.is_multiple_of(FIRE_RED_NUM_HEADS),
    "FireRed-OCR: hidden_dim must be divisible by num_heads"
);

// LM: head_dim is 128 (1536 / 12)
const _: () = assert!(
    FIRE_RED_HEAD_DIM == 128,
    "FireRed-OCR: expected head_dim = 128 (Qwen3 constant)"
);

// LM: GQA ratio must be an integer (num_heads % num_kv_heads == 0)
const _: () = assert!(
    FIRE_RED_NUM_HEADS.is_multiple_of(FIRE_RED_NUM_KV_HEADS),
    "FireRed-OCR: num_heads must be divisible by num_kv_heads (GQA)"
);

// LM: GQA ratio is 6 (12 heads / 2 KV heads)
const _: () = assert!(
    FIRE_RED_GQA_RATIO == 6,
    "FireRed-OCR: expected GQA ratio = 6"
);

// LM: num_kv_heads must be positive
const _: () = assert!(
    FIRE_RED_NUM_KV_HEADS > 0,
    "FireRed-OCR: num_kv_heads must be > 0"
);

// LM: layer count must be positive
const _: () = assert!(
    FIRE_RED_NUM_LAYERS > 0,
    "FireRed-OCR: num_layers must be > 0"
);

// LM: vocab_size must be positive
const _: () = assert!(
    FIRE_RED_VOCAB_SIZE > 0,
    "FireRed-OCR: vocab_size must be > 0"
);

// LM: intermediate_size must be larger than hidden_dim (FFN expansion)
const _: () = assert!(
    FIRE_RED_INTERMEDIATE_SIZE > FIRE_RED_HIDDEN_DIM,
    "FireRed-OCR: intermediate_size must be > hidden_dim (FFN expansion)"
);

// -- FireRed-OCR vision encoder const assertions --

// Vision: hidden_dim must be divisible by num_heads
const _: () = assert!(
    FIRE_RED_VISION_HIDDEN_DIM.is_multiple_of(FIRE_RED_VISION_NUM_HEADS),
    "FireRed-OCR: vision hidden_dim must be divisible by vision num_heads"
);

// Vision: head_dim is 80 (1280 / 16)
const _: () = assert!(
    FIRE_RED_VISION_HEAD_DIM == 80,
    "FireRed-OCR: expected vision head_dim = 80"
);

// Vision: layer count implicit via Qwen2VL ViT (typically 32 layers for
// 1280-dim, but the assertion here focuses on dimension compatibility)

// Vision-LM compatibility: vision_hidden_dim != hidden_dim, so a projection
// layer is required. Assert they are both positive (projection handles the gap).
const _: () = assert!(
    FIRE_RED_VISION_HIDDEN_DIM > 0,
    "FireRed-OCR: vision hidden_dim must be > 0"
);
const _: () = assert!(
    FIRE_RED_HIDDEN_DIM > 0,
    "FireRed-OCR: LM hidden_dim must be > 0"
);

// Vision-LM projection: vision_dim (1280) != lm_dim (1536), confirming a
// projection layer is architecturally necessary.
const _: () = assert!(
    FIRE_RED_VISION_HIDDEN_DIM != FIRE_RED_HIDDEN_DIM,
    "FireRed-OCR: vision_dim != lm_dim means projection layer is required"
);

// =============================================================================
// Runtime tests for architectural consistency
// =============================================================================

#[test]
fn test_paddleocr_vl_vision_config_consistency() {
    // Verify vision encoder dimensions are self-consistent
    assert_eq!(PADDLE_VISION_HIDDEN_DIM / PADDLE_VISION_NUM_HEADS, 64);
    assert_eq!(PADDLE_VISION_LAYERS, 12);
    assert_eq!(PADDLE_DECODER_LAYERS, 6);
    assert_eq!(PADDLE_VOCAB_SIZE, 6625);
    // Vision output directly feeds decoder (same dim)
    assert_eq!(PADDLE_VISION_HIDDEN_DIM, PADDLE_DECODER_HIDDEN_DIM);
}

#[test]
fn test_firered_ocr_qwen3_vl_2b_config_consistency() {
    // Language model dimensions
    assert_eq!(FIRE_RED_HIDDEN_DIM, 1536);
    assert_eq!(FIRE_RED_NUM_HEADS, 12);
    assert_eq!(FIRE_RED_NUM_KV_HEADS, 2);
    assert_eq!(FIRE_RED_NUM_LAYERS, 28);
    assert_eq!(FIRE_RED_VOCAB_SIZE, 151_936);
    assert_eq!(FIRE_RED_INTERMEDIATE_SIZE, 8960);
    // GQA: 12 heads / 2 kv_heads = 6 groups
    assert_eq!(FIRE_RED_NUM_HEADS / FIRE_RED_NUM_KV_HEADS, 6);
    // Head dim: 1536 / 12 = 128 (Qwen3 constant)
    assert_eq!(FIRE_RED_HIDDEN_DIM / FIRE_RED_NUM_HEADS, 128);
}

#[test]
fn test_firered_ocr_vision_encoder_config_consistency() {
    // Vision encoder dimensions
    assert_eq!(FIRE_RED_VISION_HIDDEN_DIM, 1280);
    assert_eq!(FIRE_RED_VISION_NUM_HEADS, 16);
    // Vision head dim: 1280 / 16 = 80
    assert_eq!(FIRE_RED_VISION_HIDDEN_DIM / FIRE_RED_VISION_NUM_HEADS, 80);
    // Vision dim != LM dim: projection layer needed
    assert_ne!(FIRE_RED_VISION_HIDDEN_DIM, FIRE_RED_HIDDEN_DIM);
}

#[test]
fn test_firered_ocr_ffn_expansion() {
    // FFN intermediate size should be substantially larger than hidden dim
    // Qwen3-VL-2B uses ~5.8x expansion (8960 / 1536)
    let expansion_ratio = FIRE_RED_INTERMEDIATE_SIZE / FIRE_RED_HIDDEN_DIM;
    assert!(
        expansion_ratio >= 4,
        "FFN expansion ratio should be >= 4x, got {expansion_ratio}x"
    );
    assert!(
        expansion_ratio <= 8,
        "FFN expansion ratio should be <= 8x, got {expansion_ratio}x"
    );
}
