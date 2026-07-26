// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for the Multi-Token Prediction head config.
//!
//! Proves properties of `MtpHeadConfig::validate`:
//! - Rejects zero dimensions
//! - Rejects non-finite norm_eps when per_head_norm is enabled
//! - Accepts valid configurations
//! - Default config is always valid

use super::*;

/// Prove `MtpHeadConfig::validate` rejects num_predict_tokens == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_mtp_config_rejects_zero_predict_tokens() {
    let config = MtpHeadConfig {
        num_predict_tokens: 0,
        hidden_size: 256,
        vocab_size: 1000,
        shared_trunk: false,
        per_head_norm: false,
        norm_eps: 1e-5,
    };
    assert!(
        config.validate().is_err(),
        "num_predict_tokens=0 must be rejected"
    );
}

/// Prove `MtpHeadConfig::validate` rejects hidden_size == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_mtp_config_rejects_zero_hidden_size() {
    let config = MtpHeadConfig {
        num_predict_tokens: 4,
        hidden_size: 0,
        vocab_size: 1000,
        shared_trunk: false,
        per_head_norm: false,
        norm_eps: 1e-5,
    };
    assert!(config.validate().is_err(), "hidden_size=0 must be rejected");
}

/// Prove `MtpHeadConfig::validate` rejects vocab_size == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_mtp_config_rejects_zero_vocab_size() {
    let config = MtpHeadConfig {
        num_predict_tokens: 4,
        hidden_size: 256,
        vocab_size: 0,
        shared_trunk: false,
        per_head_norm: false,
        norm_eps: 1e-5,
    };
    assert!(config.validate().is_err(), "vocab_size=0 must be rejected");
}

/// Prove `MtpHeadConfig::validate` rejects NaN norm_eps when per_head_norm is enabled.
#[kani::unwind(1)]
#[kani::proof]
fn proof_mtp_config_rejects_nan_norm_eps() {
    let config = MtpHeadConfig {
        num_predict_tokens: 4,
        hidden_size: 256,
        vocab_size: 1000,
        shared_trunk: false,
        per_head_norm: true,
        norm_eps: f64::NAN,
    };
    assert!(
        config.validate().is_err(),
        "NaN norm_eps with per_head_norm must be rejected"
    );
}

/// Prove `MtpHeadConfig::validate` rejects Inf norm_eps when per_head_norm is enabled.
#[kani::unwind(1)]
#[kani::proof]
fn proof_mtp_config_rejects_inf_norm_eps() {
    let config = MtpHeadConfig {
        num_predict_tokens: 4,
        hidden_size: 256,
        vocab_size: 1000,
        shared_trunk: false,
        per_head_norm: true,
        norm_eps: f64::INFINITY,
    };
    assert!(
        config.validate().is_err(),
        "Inf norm_eps with per_head_norm must be rejected"
    );
}

/// Prove `MtpHeadConfig::validate` accepts valid configs with per_head_norm disabled.
/// Non-finite norm_eps is allowed when per_head_norm is false (norm_eps is unused).
#[kani::unwind(1)]
#[kani::proof]
fn proof_mtp_config_accepts_valid_no_norm() {
    let num_predict: usize = kani::any();
    kani::assume(num_predict >= 1 && num_predict <= 16);
    let hidden: usize = kani::any();
    kani::assume(hidden >= 1 && hidden <= 4096);
    let vocab: usize = kani::any();
    kani::assume(vocab >= 1 && vocab <= 200000);

    let config = MtpHeadConfig {
        num_predict_tokens: num_predict,
        hidden_size: hidden,
        vocab_size: vocab,
        shared_trunk: false,
        per_head_norm: false,
        norm_eps: 1e-5,
    };
    assert!(
        config.validate().is_ok(),
        "valid config without norm must pass"
    );
}

/// Prove `MtpHeadConfig::validate` accepts valid configs with per_head_norm enabled.
#[kani::unwind(1)]
#[kani::proof]
fn proof_mtp_config_accepts_valid_with_norm() {
    let num_predict: usize = kani::any();
    kani::assume(num_predict >= 1 && num_predict <= 16);
    let hidden: usize = kani::any();
    kani::assume(hidden >= 1 && hidden <= 4096);
    let vocab: usize = kani::any();
    kani::assume(vocab >= 1 && vocab <= 200000);
    let eps: f64 = kani::any();
    kani::assume(eps > 0.0 && eps.is_finite() && eps < 1.0);

    let config = MtpHeadConfig {
        num_predict_tokens: num_predict,
        hidden_size: hidden,
        vocab_size: vocab,
        shared_trunk: false,
        per_head_norm: true,
        norm_eps: eps,
    };
    assert!(
        config.validate().is_ok(),
        "valid config with norm must pass"
    );
}

/// Prove the default `MtpHeadConfig` is always valid.
#[kani::unwind(1)]
#[kani::proof]
fn proof_mtp_config_default_is_valid() {
    let config = MtpHeadConfig::default();
    assert!(
        config.validate().is_ok(),
        "default MtpHeadConfig must be valid"
    );
}

/// Prove non-finite norm_eps is accepted when per_head_norm is disabled.
/// This is important: the field exists but is not checked unless the feature is on.
#[kani::unwind(1)]
#[kani::proof]
fn proof_mtp_config_allows_nonfinite_eps_when_norm_off() {
    let config_nan = MtpHeadConfig {
        num_predict_tokens: 4,
        hidden_size: 256,
        vocab_size: 1000,
        shared_trunk: false,
        per_head_norm: false,
        norm_eps: f64::NAN,
    };
    assert!(
        config_nan.validate().is_ok(),
        "NaN norm_eps with per_head_norm=false must be accepted"
    );

    let config_inf = MtpHeadConfig {
        num_predict_tokens: 4,
        hidden_size: 256,
        vocab_size: 1000,
        shared_trunk: false,
        per_head_norm: false,
        norm_eps: f64::INFINITY,
    };
    assert!(
        config_inf.validate().is_ok(),
        "Inf norm_eps with per_head_norm=false must be accepted"
    );
}
