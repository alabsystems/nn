// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for KokoroConfig validation invariants.
//!
//! Proves that:
//! 1. Default config passes validation.
//! 2. Zero d_en fails validation.
//! 3. Zero style_dim fails validation.
//! 4. Zero max_dur fails validation.
//! 5. n_fft=0 fails validation.
//! 6. n_fft not divisible by 4 fails validation.
//! 7. Empty upsample_rates fails validation.
//! 8. Default config field relationships are consistent.
//!
//! Part of #3793, #3351.

use crate::kokoro_tts::KokoroConfig;

/// Proof 1: Default KokoroConfig passes validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kokoro_config_default_validates() {
    let config = KokoroConfig::default();
    let result = config.validate();
    assert!(result.is_ok(), "default KokoroConfig must pass validation");
}

/// Proof 2: KokoroConfig with d_en=0 fails validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kokoro_config_zero_d_en_fails() {
    let mut config = KokoroConfig::default();
    config.d_en = 0;
    let result = config.validate();
    assert!(result.is_err(), "d_en=0 must fail validation");
}

/// Proof 3: KokoroConfig with style_dim=0 fails validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kokoro_config_zero_style_dim_fails() {
    let mut config = KokoroConfig::default();
    config.style_dim = 0;
    let result = config.validate();
    assert!(result.is_err(), "style_dim=0 must fail validation");
}

/// Proof 4: KokoroConfig with max_dur=0 fails validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kokoro_config_zero_max_dur_fails() {
    let mut config = KokoroConfig::default();
    config.max_dur = 0;
    let result = config.validate();
    assert!(result.is_err(), "max_dur=0 must fail validation");
}

/// Proof 5: KokoroConfig with n_fft=0 fails validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kokoro_config_zero_n_fft_fails() {
    let mut config = KokoroConfig::default();
    config.n_fft = 0;
    let result = config.validate();
    assert!(result.is_err(), "n_fft=0 must fail validation");
}

/// Proof 6: n_fft not divisible by 4 fails validation.
///
/// Tests all residues 1,2,3 mod 4 in range [1,15].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kokoro_config_n_fft_not_div4_fails() {
    let val: u8 = kani::any();
    kani::assume(val >= 1 && val <= 15);
    kani::assume(val % 4 != 0);
    let mut config = KokoroConfig::default();
    config.n_fft = val as usize;
    let result = config.validate();
    assert!(
        result.is_err(),
        "n_fft={} (not divisible by 4) must fail validation",
        val
    );
}

/// Proof 7: Empty upsample_rates fails validation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kokoro_config_empty_upsample_rates_fails() {
    let mut config = KokoroConfig::default();
    config.upsample_rates = vec![];
    let result = config.validate();
    assert!(result.is_err(), "empty upsample_rates must fail validation");
}

/// Proof 8: Default config field consistency.
///
/// Verifies that the default KokoroConfig has self-consistent
/// field values matching the Kokoro-82M architecture specification.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kokoro_config_default_field_consistency() {
    let config = KokoroConfig::default();

    // d_en must equal gen_initial_channels (both 512 in Kokoro-82M)
    assert_eq!(config.d_en, config.gen_initial_channels);

    // upsample_rates and upsample_kernel_sizes must have same length
    assert_eq!(
        config.upsample_rates.len(),
        config.upsample_kernel_sizes.len()
    );

    // resblock_dilations and resblock_kernel_sizes must have same length
    assert_eq!(
        config.resblock_dilations.len(),
        config.resblock_kernel_sizes.len()
    );

    // n_fft must be divisible by 4 (iSTFT constraint)
    assert_eq!(config.n_fft % 4, 0);

    // style_dim * 2 = 256 (full voice embedding splits into decoder + prosody halves)
    assert_eq!(config.style_dim * 2, 256);
}
