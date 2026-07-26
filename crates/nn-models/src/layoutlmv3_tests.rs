// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the LayoutLMv3 model builder.

use super::*;

#[test]
fn test_layoutlmv3_config_preset_valid() {
    let config = LayoutLMv3Config::preset(7);
    config.validate().expect("preset should be valid");
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_layers, 12);
    assert_eq!(config.num_heads, 12);
    assert_eq!(config.vocab_size, 50_265);
    assert_eq!(config.num_labels, 7);
    assert_eq!(config.max_2d_pos, 1024);
}

#[test]
fn test_layoutlmv3_config_patch_dim() {
    let config = LayoutLMv3Config::preset(7);
    // 3 * 16 * 16 = 768
    assert_eq!(config.patch_dim(), 768);
}

#[test]
fn test_layoutlmv3_config_visual_seq_len() {
    let config = LayoutLMv3Config::preset(7);
    // (224 / 16)^2 = 14^2 = 196
    assert_eq!(config.visual_seq_len(), 196);
}

#[test]
fn test_layoutlmv3_config_zero_labels_rejected() {
    let config = LayoutLMv3Config::preset(0);
    assert!(config.validate().is_err());
}

#[test]
fn test_layoutlmv3_config_divisibility_rejected() {
    let config = LayoutLMv3Config {
        hidden_size: 100,
        num_heads: 12,
        ..LayoutLMv3Config::preset(7)
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_layoutlmv3_config_image_not_divisible_rejected() {
    let config = LayoutLMv3Config {
        image_size: 225,
        patch_size: 16,
        ..LayoutLMv3Config::preset(7)
    };
    assert!(config.validate().is_err());
}
