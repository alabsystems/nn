// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compile-time static tests for Granite-Docling-258M model configuration.
//!
//! Granite-Docling-258M = SigLIP2-base-patch16 (vision) + Granite-165M (language).
//! These const assertions catch configuration mismatches at compile time rather
//! than at runtime weight-loading time.
//!
//! Reference: `ibm-granite/granite-vision-docling-258m-preview`

// ---- SigLIP2 Vision Encoder (base-patch16-384) ----

const SIGLIP2_IMAGE_SIZE: usize = 384;
const SIGLIP2_PATCH_SIZE: usize = 16;
const SIGLIP2_HIDDEN_DIM: usize = 768;
const SIGLIP2_NUM_HEADS: usize = 12;
const SIGLIP2_NUM_LAYERS: usize = 12;
const SIGLIP2_INTERMEDIATE_SIZE: usize = 3072;
const SIGLIP2_NUM_CHANNELS: usize = 3;

// ---- Granite-165M Language Model ----

const GRANITE_HIDDEN_DIM: usize = 768;
const GRANITE_NUM_HEADS: usize = 12;
const GRANITE_VOCAB_SIZE: usize = 49152;
const GRANITE_NUM_LAYERS: usize = 12;

// ---- Derived Constants ----

const SIGLIP2_GRID_SIZE: usize = SIGLIP2_IMAGE_SIZE / SIGLIP2_PATCH_SIZE;
const SIGLIP2_NUM_PATCHES: usize = SIGLIP2_GRID_SIZE * SIGLIP2_GRID_SIZE;
const SIGLIP2_HEAD_DIM: usize = SIGLIP2_HIDDEN_DIM / SIGLIP2_NUM_HEADS;

// ---- SigLIP2 Const Assertions ----

// Patches must tile the image evenly (no partial patches).
const _: () = assert!(
    SIGLIP2_IMAGE_SIZE.is_multiple_of(SIGLIP2_PATCH_SIZE),
    "SigLIP2: image_size must be divisible by patch_size"
);

// Hidden dimension must split evenly across attention heads.
const _: () = assert!(
    SIGLIP2_HIDDEN_DIM.is_multiple_of(SIGLIP2_NUM_HEADS),
    "SigLIP2: hidden_dim must be divisible by num_heads"
);

// Verify derived patch count: (384/16)^2 = 24^2 = 576.
const _: () = assert!(
    SIGLIP2_NUM_PATCHES == 576,
    "SigLIP2: num_patches must be (image_size/patch_size)^2 = 576"
);

// Grid size sanity: 384/16 = 24.
const _: () = assert!(
    SIGLIP2_GRID_SIZE == 24,
    "SigLIP2: grid_size must be image_size/patch_size = 24"
);

// Head dimension: 768/12 = 64.
const _: () = assert!(
    SIGLIP2_HEAD_DIM == 64,
    "SigLIP2: head_dim must be hidden_dim/num_heads = 64"
);

// Layer count must be positive.
const _: () = assert!(SIGLIP2_NUM_LAYERS > 0, "SigLIP2: num_layers must be > 0");

// MLP intermediate size must be positive.
const _: () = assert!(
    SIGLIP2_INTERMEDIATE_SIZE > 0,
    "SigLIP2: intermediate_size must be > 0"
);

// Standard RGB input.
const _: () = assert!(
    SIGLIP2_NUM_CHANNELS == 3,
    "SigLIP2: num_channels must be 3 (RGB)"
);

// ---- Granite-165M Const Assertions ----

// Hidden dimension must split evenly across attention heads.
const _: () = assert!(
    GRANITE_HIDDEN_DIM.is_multiple_of(GRANITE_NUM_HEADS),
    "Granite: hidden_dim must be divisible by num_heads"
);

// Vocabulary size must be positive.
const _: () = assert!(GRANITE_VOCAB_SIZE > 0, "Granite: vocab_size must be > 0");

// Layer count must be positive.
const _: () = assert!(GRANITE_NUM_LAYERS > 0, "Granite: num_layers must be > 0");

// ---- Cross-Model Compatibility Assertions ----

// SigLIP2 output dim must match Granite cross-attention input dim.
// This is the critical interface: the vision encoder output feeds into the
// language model's cross-attention layers. Dimension mismatch = runtime crash.
const _: () = assert!(
    SIGLIP2_HIDDEN_DIM == GRANITE_HIDDEN_DIM,
    "Granite-Docling: SigLIP2 output dim must match Granite cross-attention input dim"
);

// ---- Runtime Tests ----

#[test]
fn test_granite_docling_siglip2_config_matches_constants() {
    // Verify our const values match what SigLip2Config::base_patch16 produces.
    let config = super::SigLip2Config::base_patch16(SIGLIP2_IMAGE_SIZE)
        .expect("SigLip2Config::base_patch16 should succeed");
    assert_eq!(config.hidden_size, SIGLIP2_HIDDEN_DIM);
    assert_eq!(config.num_heads, SIGLIP2_NUM_HEADS);
    assert_eq!(config.num_layers, SIGLIP2_NUM_LAYERS);
    assert_eq!(config.patch_size, SIGLIP2_PATCH_SIZE);
    assert_eq!(config.image_size, SIGLIP2_IMAGE_SIZE);
    assert_eq!(config.intermediate_size, SIGLIP2_INTERMEDIATE_SIZE);
    assert_eq!(config.num_channels, SIGLIP2_NUM_CHANNELS);
}

#[test]
fn test_granite_docling_vit_config_num_patches() {
    let config = super::SigLip2Config::base_patch16(SIGLIP2_IMAGE_SIZE)
        .expect("SigLip2Config::base_patch16 should succeed");
    let vit = config
        .to_vit_config()
        .expect("to_vit_config should succeed");
    assert_eq!(vit.num_patches(), SIGLIP2_NUM_PATCHES);
    // SigLIP2 has no CLS token, so seq_len == num_patches.
    assert_eq!(vit.seq_len(), SIGLIP2_NUM_PATCHES);
}

#[test]
fn test_granite_docling_cross_model_dim_alignment() {
    // The vision encoder hidden dim must equal the language model hidden dim
    // for the cross-attention projection to work without an adapter layer.
    assert_eq!(
        SIGLIP2_HIDDEN_DIM, GRANITE_HIDDEN_DIM,
        "Vision and language hidden dims must match for Granite-Docling-258M"
    );
}
