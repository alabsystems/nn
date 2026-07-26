// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Qwen2VLVitConfig and Qwen3VLVitConfig.

use super::{Qwen2VLVitConfig, Qwen3VLVitConfig};

#[test]
fn test_qwen25_vl_7b_defaults() {
    let config = Qwen2VLVitConfig::qwen25_vl_7b().unwrap();
    assert_eq!(config.hidden_size, 1280);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_heads, 16);
    assert_eq!(config.intermediate_size, 5120);
    assert_eq!(config.window_size, 14);
    assert_eq!(config.head_dim(), 80);
}

#[test]
fn test_window_layer_default_odd() {
    let config = Qwen2VLVitConfig::qwen25_vl_7b().unwrap();
    // Even layers -> global, odd layers -> window
    assert!(!config.is_window_layer(0));
    assert!(config.is_window_layer(1));
    assert!(!config.is_window_layer(2));
    assert!(config.is_window_layer(3));
    assert!(config.is_window_layer(31));
}

#[test]
fn test_window_layer_explicit() {
    let config = Qwen2VLVitConfig::new(3, 128, 8, 4, 256, 14, 2, 1e-6, 7, vec![2, 5, 7]).unwrap();
    assert!(!config.is_window_layer(0));
    assert!(!config.is_window_layer(1));
    assert!(config.is_window_layer(2));
    assert!(!config.is_window_layer(3));
    assert!(config.is_window_layer(5));
    assert!(config.is_window_layer(7));
}

#[test]
fn test_validate_zero_window_size() {
    let err = Qwen2VLVitConfig::new(3, 128, 4, 4, 256, 14, 2, 1e-6, 0, Vec::new());
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("window_size"), "msg: {msg}");
}

#[test]
fn test_validate_window_layer_out_of_range() {
    let err = Qwen2VLVitConfig::new(3, 128, 4, 4, 256, 14, 2, 1e-6, 7, vec![0, 5]);
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("window_layers"), "msg: {msg}");
}

#[test]
fn test_validate_hidden_size_not_divisible_by_heads() {
    let err = Qwen2VLVitConfig::new(3, 100, 4, 3, 256, 14, 2, 1e-6, 7, Vec::new());
    assert!(err.is_err());
}

#[test]
fn test_validate_zero_num_layers() {
    let err = Qwen2VLVitConfig::new(3, 128, 0, 4, 256, 14, 2, 1e-6, 7, Vec::new());
    assert!(err.is_err());
}

// -- Qwen3VLVitConfig tests ---------------------------------------------------

#[test]
fn test_qwen3_vl_2b_defaults() {
    let config = Qwen3VLVitConfig::qwen3_vl_2b().unwrap();
    assert_eq!(config.hidden_size, 1280);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_heads, 16);
    assert_eq!(config.intermediate_size, 5120);
    assert_eq!(config.window_size, 14);
    assert_eq!(config.global_every_n, 4);
    assert_eq!(config.head_dim(), 80);
    assert_eq!(config.deepstack_layers, vec![7, 15, 23, 31]);
    assert_eq!(config.deepstack_output_size, 1536);
}

#[test]
fn test_qwen3_vl_7b_defaults() {
    let config = Qwen3VLVitConfig::qwen3_vl_7b().unwrap();
    assert_eq!(config.hidden_size, 3584);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_heads, 28);
    assert_eq!(config.intermediate_size, 18944);
    assert_eq!(config.head_dim(), 128);
    assert_eq!(config.deepstack_output_size, 3584);
}

#[test]
fn test_qwen3_vl_72b_defaults() {
    let config = Qwen3VLVitConfig::qwen3_vl_72b().unwrap();
    assert_eq!(config.hidden_size, 3584);
    assert_eq!(config.num_layers, 80);
    assert_eq!(config.num_heads, 28);
    assert_eq!(config.deepstack_layers, vec![19, 39, 59, 79]);
    assert_eq!(config.deepstack_output_size, 8192);
}

#[test]
fn test_qwen3_vl_global_layer_pattern() {
    let config = Qwen3VLVitConfig::qwen3_vl_2b().unwrap();
    // With global_every_n=4: layers 3, 7, 11, 15, 19, 23, 27, 31 are global
    assert!(config.is_window_layer(0));
    assert!(config.is_window_layer(1));
    assert!(config.is_window_layer(2));
    assert!(!config.is_window_layer(3)); // global
    assert!(config.is_window_layer(4));
    assert!(config.is_window_layer(5));
    assert!(config.is_window_layer(6));
    assert!(!config.is_window_layer(7)); // global
    assert!(!config.is_window_layer(31)); // global (layer 32 is last, 32 % 4 == 0)
}

#[test]
fn test_qwen3_vl_window_pattern_generation() {
    let config = Qwen3VLVitConfig::qwen3_vl_2b().unwrap();
    let pattern = config.window_pattern();
    assert_eq!(pattern.len(), 32);
    // Every 4th layer (3, 7, 11, ...) is global (false), rest are window (true)
    for (i, &is_window) in pattern.iter().enumerate() {
        let expected = (i + 1) % 4 != 0;
        assert_eq!(
            is_window, expected,
            "Layer {i}: expected window={expected}, got {is_window}"
        );
    }
    // Count: 32 layers, 8 global, 24 window
    let window_count = pattern.iter().filter(|&&v| v).count();
    let global_count = pattern.iter().filter(|&&v| !v).count();
    assert_eq!(window_count, 24);
    assert_eq!(global_count, 8);
}

#[test]
fn test_qwen3_vl_all_window_when_global_every_n_zero() {
    // global_every_n = 0 means no global layers at all
    let config =
        Qwen3VLVitConfig::new(3, 128, 8, 4, 256, 14, 2, 1e-6, 7, 0, Vec::new(), 0).unwrap();
    for i in 0..8 {
        assert!(config.is_window_layer(i), "Layer {i} should be window");
        assert!(!config.is_global_layer(i), "Layer {i} should not be global");
    }
}

#[test]
fn test_qwen3_vl_validate_zero_window_size() {
    let err = Qwen3VLVitConfig::new(3, 128, 4, 4, 256, 14, 2, 1e-6, 0, 4, Vec::new(), 0);
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("window_size"), "msg: {msg}");
}

#[test]
fn test_qwen3_vl_validate_deepstack_out_of_range() {
    let err = Qwen3VLVitConfig::new(3, 128, 4, 4, 256, 14, 2, 1e-6, 7, 4, vec![0, 5], 128);
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("deepstack_layers"), "msg: {msg}");
}

#[test]
fn test_qwen3_vl_validate_deepstack_output_size_zero_with_layers() {
    let err = Qwen3VLVitConfig::new(3, 128, 8, 4, 256, 14, 2, 1e-6, 7, 4, vec![3, 7], 0);
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("deepstack_output_size"), "msg: {msg}");
}

#[test]
fn test_qwen3_vl_head_dim_consistency() {
    // Qwen3-VL-2B: 1280 / 16 = 80 (same vision encoder as Qwen2.5-VL)
    let config_2b = Qwen3VLVitConfig::qwen3_vl_2b().unwrap();
    assert_eq!(config_2b.head_dim(), 80);

    // Qwen3-VL-7B: 3584 / 28 = 128
    let config_7b = Qwen3VLVitConfig::qwen3_vl_7b().unwrap();
    assert_eq!(config_7b.head_dim(), 128);

    // Qwen3-VL-72B: same vision encoder as 7B
    let config_72b = Qwen3VLVitConfig::qwen3_vl_72b().unwrap();
    assert_eq!(config_72b.head_dim(), 128);
}
