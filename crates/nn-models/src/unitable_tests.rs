// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the UniTable model builder.

use super::*;

#[test]
fn test_unitable_config_preset_valid() {
    let config = UniTableConfig::preset();
    config.validate().expect("preset should be valid");
    assert_eq!(config.hidden_dim, 768);
    assert_eq!(config.num_layers, 6);
    assert_eq!(config.num_heads, 12);
    assert_eq!(config.vocab_size, 200);
    assert_eq!(config.patch_size, 16);
    assert_eq!(config.image_size, 448);
}

#[test]
fn test_unitable_config_patch_dim() {
    let config = UniTableConfig::preset();
    // 3 channels * 16 * 16 = 768
    assert_eq!(config.patch_dim(), 768);
}

#[test]
fn test_unitable_config_image_seq_len() {
    let config = UniTableConfig::preset();
    // (448 / 16)^2 = 28^2 = 784
    assert_eq!(config.image_seq_len(), 784);
}

#[test]
fn test_unitable_config_zero_hidden_dim_rejected() {
    let config = UniTableConfig {
        hidden_dim: 0,
        ..UniTableConfig::preset()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_unitable_config_divisibility_rejected() {
    let config = UniTableConfig {
        hidden_dim: 100,
        num_heads: 12,
        ..UniTableConfig::preset()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_unitable_config_image_not_divisible_rejected() {
    let config = UniTableConfig {
        image_size: 450,
        patch_size: 16,
        ..UniTableConfig::preset()
    };
    assert!(config.validate().is_err());
}
