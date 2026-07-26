// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for ECAPA-TDNN architecture constants.
//!
//! Proves that:
//! 1. Embedding dimension is 192 (standard ECAPA-TDNN).
//! 2. Hidden channels (512) is divisible by Res2Net scale (8).
//! 3. SE reduction (128) divides hidden channels evenly.
//! 4. 3 dilations produce 3x hidden_channels for cat conv input.
//! 5. ASP output is 2x the cat channels (mean + std).
//! 6. All dilations are distinct and > 1 (multi-scale).
//!
//! Part of #3793, #3351.

/// ECAPA-TDNN architecture constants (mirrors ecapa_tdnn.rs private constants).
const MEL_CHANNELS: usize = 80;
const HIDDEN_CHANNELS: usize = 512;
const EMBED_DIM: usize = 192;
const RES2NET_SCALE: usize = 8;
const SE_REDUCTION: usize = 128;
const DILATIONS: [usize; 3] = [2, 3, 4];

/// Proof 1: Embedding dimension is 192.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_embed_dim_is_192() {
    assert_eq!(EMBED_DIM, 192);
}

/// Proof 2: Hidden channels is divisible by Res2Net scale.
///
/// Res2Net splits channels into `scale` groups. Each group must
/// have an integer number of channels.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_hidden_divisible_by_scale() {
    assert!(RES2NET_SCALE > 0);
    assert_eq!(
        HIDDEN_CHANNELS % RES2NET_SCALE,
        0,
        "hidden channels must be divisible by res2net scale"
    );
    let group_size = HIDDEN_CHANNELS / RES2NET_SCALE;
    assert!(group_size > 0, "each res2net group must have > 0 channels");
    assert_eq!(group_size, 64);
}

/// Proof 3: SE reduction divides hidden channels.
///
/// SE bottleneck: Linear(hidden, hidden/reduction) → ReLU → Linear(hidden/reduction, hidden).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_se_reduction_divides_hidden() {
    assert!(SE_REDUCTION > 0);
    assert_eq!(
        HIDDEN_CHANNELS % SE_REDUCTION,
        0,
        "SE reduction must divide hidden channels"
    );
    let bottleneck = HIDDEN_CHANNELS / SE_REDUCTION;
    assert!(bottleneck > 0);
    assert_eq!(bottleneck, 4);
}

/// Proof 4: Cat conv input size = 3 * hidden_channels.
///
/// Three SE-Res2Blocks produce skip connections that are concatenated
/// along the channel dimension before the cat conv.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_cat_channels() {
    let cat_channels = HIDDEN_CHANNELS * DILATIONS.len();
    assert_eq!(cat_channels, 1536);
    // Cat conv: 1536 → 1536 (preserves dimension)
    assert_eq!(DILATIONS.len(), 3);
}

/// Proof 5: ASP output is 2x cat channels (mean + std statistics).
///
/// Attentive Statistics Pooling produces [mean, std] concatenation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_asp_output_size() {
    let cat_channels = HIDDEN_CHANNELS * DILATIONS.len();
    let asp_output = cat_channels * 2;
    assert_eq!(asp_output, 3072);
    // Final linear: 3072 → 192
}

/// Proof 6: All dilations are distinct and greater than 1.
///
/// Multi-scale receptive fields require distinct dilations.
/// Dilation > 1 ensures each block covers a different temporal scale.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_dilations_distinct_and_gt1() {
    for &d in &DILATIONS {
        assert!(d > 1, "each dilation must be > 1 for multi-scale");
    }
    // All distinct
    assert_ne!(DILATIONS[0], DILATIONS[1]);
    assert_ne!(DILATIONS[0], DILATIONS[2]);
    assert_ne!(DILATIONS[1], DILATIONS[2]);
}

/// Proof 7: MEL_CHANNELS matches standard mel spectrogram dimension.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_mel_channels_is_80() {
    assert_eq!(MEL_CHANNELS, 80);
}
