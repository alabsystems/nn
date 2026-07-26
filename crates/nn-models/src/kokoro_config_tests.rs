// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::kokoro_config`].

use super::*;

// -- Default config ----------------------------------------------------------

#[test]
fn test_default_config_d_en() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.d_en, 512);
}

#[test]
fn test_default_config_style_dim() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.style_dim, 128);
}

#[test]
fn test_default_upsample_rates_product() {
    let cfg = KokoroConfig::default();
    let product: usize = cfg.upsample_rates.iter().product();
    // upsample_rates [10, 6] → product 60 (hop_length for iSTFT)
    assert_eq!(product, 60);
}

#[test]
fn test_default_upsample_kernel_sizes_match_rates() {
    let cfg = KokoroConfig::default();
    // kernel_size == 2 * rate for each upsample stage
    for (ks, rate) in cfg
        .upsample_kernel_sizes
        .iter()
        .zip(cfg.upsample_rates.iter())
    {
        assert_eq!(*ks, 2 * rate, "kernel_size should be 2× rate");
    }
}

#[test]
fn test_default_resblock_dilations_len_matches_kernels() {
    let cfg = KokoroConfig::default();
    assert_eq!(
        cfg.resblock_dilations.len(),
        cfg.resblock_kernel_sizes.len()
    );
}

#[test]
fn test_default_plbert_config() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.plbert.vocab_size, 178);
    assert_eq!(cfg.plbert.hidden_size, 768);
}

// -- Validate ----------------------------------------------------------------

#[test]
fn test_default_config_validates() {
    KokoroConfig::default().validate().unwrap();
}

#[test]
fn test_validate_rejects_zero_d_en() {
    let cfg = KokoroConfig {
        d_en: 0,
        ..Default::default()
    };
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("d_en"));
}

#[test]
fn test_validate_rejects_zero_style_dim() {
    let cfg = KokoroConfig {
        style_dim: 0,
        ..Default::default()
    };
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("style_dim"));
}

#[test]
fn test_validate_rejects_n_fft_not_divisible_by_4() {
    let cfg = KokoroConfig {
        n_fft: 6,
        ..Default::default()
    };
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("n_fft"));
}

#[test]
fn test_validate_rejects_empty_upsample_rates() {
    let cfg = KokoroConfig {
        upsample_rates: vec![],
        ..Default::default()
    };
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("upsample_rates"));
}

#[test]
fn test_validate_rejects_zero_max_dur() {
    let cfg = KokoroConfig {
        max_dur: 0,
        ..Default::default()
    };
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("max_dur"));
}
