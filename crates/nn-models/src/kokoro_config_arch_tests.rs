// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Architecture validation tests for Kokoro-82M TTS model (#3942).
//!
//! Validates structural invariants: config defaults, dimension relationships,
//! PlBert sub-config consistency, generator architecture properties, and
//! builder pattern correctness.

use super::*;

// ---------------------------------------------------------------------------
// Default config architectural properties
// ---------------------------------------------------------------------------

#[test]
fn test_default_gen_initial_channels() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.gen_initial_channels, 512);
}

#[test]
fn test_default_f0_bilstm_hidden() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.f0_bilstm_hidden, 256);
}

#[test]
fn test_default_max_dur() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.max_dur, 50);
}

#[test]
fn test_default_n_prosody_layers() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.n_prosody_layers, 3);
}

// ---------------------------------------------------------------------------
// Generator upsample architecture invariants
// ---------------------------------------------------------------------------

#[test]
fn test_upsample_rates_and_kernel_sizes_same_length() {
    let cfg = KokoroConfig::default();
    assert_eq!(
        cfg.upsample_rates.len(),
        cfg.upsample_kernel_sizes.len(),
        "upsample_rates and upsample_kernel_sizes must have same length"
    );
}

#[test]
fn test_upsample_kernel_is_2x_rate() {
    // This is an ISTFTNet architecture invariant: kernel_size = 2 * stride
    let cfg = KokoroConfig::default();
    for (i, (rate, kernel)) in cfg
        .upsample_rates
        .iter()
        .zip(cfg.upsample_kernel_sizes.iter())
        .enumerate()
    {
        assert_eq!(
            *kernel,
            2 * rate,
            "upsample stage {i}: kernel_size ({kernel}) should be 2 * rate ({rate})"
        );
    }
}

#[test]
fn test_upsample_total_stride_is_hop_length() {
    // Total upsample product = hop_length for iSTFT reconstruction
    let cfg = KokoroConfig::default();
    let hop_length: usize = cfg.upsample_rates.iter().product();
    assert_eq!(
        hop_length, 60,
        "total upsample stride (hop_length) should be 60"
    );
}

// ---------------------------------------------------------------------------
// Resblock architecture invariants
// ---------------------------------------------------------------------------

#[test]
fn test_resblock_dilations_match_kernel_count() {
    let cfg = KokoroConfig::default();
    assert_eq!(
        cfg.resblock_dilations.len(),
        cfg.resblock_kernel_sizes.len(),
        "resblock_dilations length must match resblock_kernel_sizes length"
    );
}

#[test]
fn test_resblock_kernel_sizes_are_odd() {
    // Conv1d with same-padding requires odd kernel sizes for symmetry
    let cfg = KokoroConfig::default();
    for (i, &ks) in cfg.resblock_kernel_sizes.iter().enumerate() {
        assert!(
            ks % 2 == 1,
            "resblock kernel {i} ({ks}) should be odd for symmetric padding"
        );
    }
}

#[test]
fn test_resblock_dilations_are_positive() {
    let cfg = KokoroConfig::default();
    for (i, dilations) in cfg.resblock_dilations.iter().enumerate() {
        for (j, &d) in dilations.iter().enumerate() {
            assert!(d > 0, "resblock {i}, dilation {j} ({d}) must be positive");
        }
    }
}

// ---------------------------------------------------------------------------
// n_fft architecture constraint
// ---------------------------------------------------------------------------

#[test]
fn test_n_fft_divisible_by_4() {
    let cfg = KokoroConfig::default();
    assert!(
        cfg.n_fft.is_multiple_of(4),
        "n_fft ({}) must be divisible by 4",
        cfg.n_fft
    );
}

#[test]
fn test_n_fft_default_value() {
    // n_fft = 20 means the iSTFT produces 20+2=22 frequency bins (n_fft/2 + 1 = 11)
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.n_fft, 20);
}

// ---------------------------------------------------------------------------
// PlBert sub-config consistency
// ---------------------------------------------------------------------------

#[test]
fn test_plbert_hidden_divisible_by_heads() {
    let cfg = KokoroConfig::default();
    assert!(
        cfg.plbert
            .hidden_size
            .is_multiple_of(cfg.plbert.num_attention_heads),
        "PlBert hidden_size ({}) must be divisible by num_attention_heads ({})",
        cfg.plbert.hidden_size,
        cfg.plbert.num_attention_heads
    );
}

#[test]
fn test_plbert_head_dim() {
    let cfg = KokoroConfig::default();
    let head_dim = cfg.plbert.hidden_size / cfg.plbert.num_attention_heads;
    assert_eq!(head_dim, 64, "PlBert head_dim = 768/12 = 64");
}

#[test]
fn test_plbert_factorized_embedding_dim() {
    // ALBERT-style: embedding_dim (128) is smaller than hidden_size (768)
    let cfg = KokoroConfig::default();
    assert!(
        cfg.plbert.embedding_dim < cfg.plbert.hidden_size,
        "PlBert uses factorized embeddings: embedding_dim < hidden_size"
    );
}

#[test]
fn test_plbert_vocab_matches_kokoro_phonemes() {
    let cfg = KokoroConfig::default();
    assert_eq!(
        cfg.plbert.vocab_size, 178,
        "PlBert vocab = 178 (Kokoro phonemes)"
    );
}

// ---------------------------------------------------------------------------
// KokoroConfig::new() constructor
// ---------------------------------------------------------------------------

#[test]
fn test_new_constructor_equals_default() {
    let from_new = KokoroConfig::new();
    let from_default = KokoroConfig::default();
    assert_eq!(from_new.d_en, from_default.d_en);
    assert_eq!(from_new.style_dim, from_default.style_dim);
    assert_eq!(from_new.n_fft, from_default.n_fft);
    assert_eq!(from_new.max_dur, from_default.max_dur);
    assert_eq!(
        from_new.gen_initial_channels,
        from_default.gen_initial_channels
    );
}

// ---------------------------------------------------------------------------
// Config clone independence
// ---------------------------------------------------------------------------

#[test]
fn test_config_clone_independence() {
    let c1 = KokoroConfig::default();
    let mut c2 = c1.clone();
    c2.d_en = 9999;
    assert_eq!(
        c1.d_en, 512,
        "original should be unchanged after clone mutation"
    );
    assert_eq!(c2.d_en, 9999);
}

// ---------------------------------------------------------------------------
// Config Debug format includes key fields
// ---------------------------------------------------------------------------

#[test]
fn test_config_debug_contains_key_fields() {
    let c = KokoroConfig::default();
    let debug = format!("{c:?}");
    assert!(debug.contains("d_en"), "Debug should contain d_en");
    assert!(
        debug.contains("style_dim"),
        "Debug should contain style_dim"
    );
    assert!(debug.contains("n_fft"), "Debug should contain n_fft");
    assert!(debug.contains("plbert"), "Debug should contain plbert");
}

// ---------------------------------------------------------------------------
// d_en = gen_initial_channels (architectural constraint)
// ---------------------------------------------------------------------------

#[test]
fn test_d_en_equals_gen_initial_channels() {
    let cfg = KokoroConfig::default();
    assert_eq!(
        cfg.d_en, cfg.gen_initial_channels,
        "d_en and gen_initial_channels should both be 512"
    );
}

// ---------------------------------------------------------------------------
// Channel progression through upsample stages
// ---------------------------------------------------------------------------

#[test]
fn test_channel_halving_through_upsample() {
    // ISTFTNet halves channels at each upsample stage:
    // initial_channels=512 -> 256 -> 128
    let cfg = KokoroConfig::default();
    let mut channels = cfg.gen_initial_channels;
    for _ in &cfg.upsample_rates {
        channels /= 2;
        assert!(channels > 0, "channels should not reach zero");
    }
    // After 2 upsample stages: 512 -> 256 -> 128
    assert_eq!(channels, 128);
}
