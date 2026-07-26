// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::kokoro_error`].

use super::*;

#[test]
fn test_invalid_speed_display() {
    let err = KokoroError::InvalidSpeed { value: -1.0 };
    let msg = format!("{err}");
    assert!(msg.contains("-1"), "should show the invalid value");
    assert!(
        msg.contains("speed"),
        "should mention speed (case-insensitive)"
    );
}

#[test]
fn test_non_finite_intermediate_display() {
    let err = KokoroError::NonFiniteIntermediate {
        stage: "encoder",
        count: 42,
    };
    let msg = format!("{err}");
    assert!(msg.contains("encoder"));
    assert!(msg.contains("42"));
}

#[test]
fn test_istft_failed_display() {
    let err = KokoroError::IstftFailed(KokoroIstftError::NonFiniteInput);
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite"),
        "should contain inner error message: {msg}"
    );
}

#[test]
fn test_istft_bin_mismatch_display() {
    let err = KokoroError::IstftBinMismatch {
        actual: 9,
        expected: 11,
        n_fft: 20,
    };
    let msg = format!("{err}");
    assert!(msg.contains("9"), "should show actual bins: {msg}");
    assert!(msg.contains("11"), "should show expected bins: {msg}");
    assert!(msg.contains("20"), "should show n_fft: {msg}");
}

#[test]
fn test_istft_array_layout_display() {
    let err = KokoroError::IstftArrayLayout;
    let msg = format!("{err}");
    assert!(
        msg.contains("standard-layout") || msg.contains("contiguous"),
        "should mention layout issue: {msg}"
    );
}

#[test]
fn test_missing_source_module_display() {
    let err = KokoroError::MissingSourceModule;
    let msg = format!("{err}");
    assert!(
        msg.contains("SourceModule"),
        "should mention SourceModule: {msg}"
    );
    assert!(
        msg.contains("STFT") || msg.contains("excitation"),
        "should mention frequency-domain context: {msg}"
    );
}

#[test]
fn test_missing_source_module_into_tensor_error() {
    let err = KokoroError::MissingSourceModule;
    let te: nn_core::TensorError = err.into_tensor_error();
    match te {
        nn_core::TensorError::Unsupported(msg) => {
            assert!(msg.contains("SourceModule"), "wrapped message: {msg}");
        }
        other => panic!("expected Unsupported, got: {other:?}"),
    }
}

#[test]
fn test_log_mag_clamp_max_prevents_overflow() {
    // exp(LOG_MAG_CLAMP_MAX) should be < f32::MAX
    let val = (LOG_MAG_CLAMP_MAX as f32).exp();
    assert!(
        val.is_finite(),
        "exp({LOG_MAG_CLAMP_MAX}) should be finite f32"
    );
}

#[test]
fn test_generator_config_mismatch_display() {
    let err = KokoroError::GeneratorConfigMismatch {
        field: "upsample_kernel_sizes",
        reference_field: "upsample_rates",
        expected: 2,
        actual: 3,
    };
    let msg = format!("{err}");
    assert!(msg.contains("upsample_kernel_sizes"), "{msg}");
    assert!(msg.contains("upsample_rates"), "{msg}");
    assert!(msg.contains("2"), "expected length: {msg}");
    assert!(msg.contains("3"), "actual length: {msg}");
}

// -- validate_speed tests (#2218 F6 deduplication) ---------------------------

#[test]
fn test_validate_speed_positive_finite_ok() {
    assert!(validate_speed(1.0).is_ok());
    assert!(validate_speed(0.5).is_ok());
    assert!(validate_speed(100.0).is_ok());
    assert!(validate_speed(f32::MIN_POSITIVE).is_ok());
    assert!(validate_speed(f32::MAX).is_ok());
}

#[test]
fn test_validate_speed_zero_rejected() {
    let err = validate_speed(0.0).unwrap_err();
    assert!(matches!(err, KokoroError::InvalidSpeed { value } if value == 0.0));
}

#[test]
fn test_validate_speed_negative_zero_rejected() {
    // IEEE 754: -0.0 <= 0.0 is true, so -0.0 must be rejected.
    let err = validate_speed(-0.0).unwrap_err();
    assert!(matches!(err, KokoroError::InvalidSpeed { .. }));
}

#[test]
fn test_validate_speed_negative_rejected() {
    let err = validate_speed(-1.0).unwrap_err();
    assert!(matches!(err, KokoroError::InvalidSpeed { value } if value == -1.0));
}

#[test]
fn test_validate_speed_nan_rejected() {
    let err = validate_speed(f32::NAN).unwrap_err();
    match err {
        KokoroError::InvalidSpeed { value } => assert!(value.is_nan()),
        other => panic!("expected InvalidSpeed, got: {other:?}"),
    }
}

#[test]
fn test_validate_speed_infinity_rejected() {
    let err = validate_speed(f32::INFINITY).unwrap_err();
    assert!(matches!(err, KokoroError::InvalidSpeed { value } if value == f32::INFINITY));
}

#[test]
fn test_validate_speed_neg_infinity_rejected() {
    let err = validate_speed(f32::NEG_INFINITY).unwrap_err();
    assert!(matches!(err, KokoroError::InvalidSpeed { value } if value == f32::NEG_INFINITY));
}

// -- validate_generator_config tests -----------------------------------------

#[test]
fn test_validate_generator_config_default_ok() {
    use crate::kokoro_tts::KokoroConfig;
    let config = KokoroConfig::default();
    assert!(validate_generator_config(&config).is_ok());
}

#[test]
fn test_validate_generator_config_upsample_mismatch() {
    use crate::kokoro_tts::KokoroConfig;
    let config = KokoroConfig {
        upsample_kernel_sizes: vec![20], // len 1, but upsample_rates has len 2
        ..KokoroConfig::default()
    };
    let err = validate_generator_config(&config).unwrap_err();
    match err {
        KokoroError::GeneratorConfigMismatch {
            field,
            expected,
            actual,
            ..
        } => {
            assert_eq!(field, "upsample_kernel_sizes");
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        }
        other => panic!("expected GeneratorConfigMismatch, got: {other:?}"),
    }
}

#[test]
fn test_validate_generator_config_resblock_mismatch() {
    use crate::kokoro_tts::KokoroConfig;
    let config = KokoroConfig {
        resblock_dilations: vec![vec![1, 3, 5]], // len 1, but resblock_kernel_sizes has len 3
        ..KokoroConfig::default()
    };
    let err = validate_generator_config(&config).unwrap_err();
    match err {
        KokoroError::GeneratorConfigMismatch {
            field,
            expected,
            actual,
            ..
        } => {
            assert_eq!(field, "resblock_dilations");
            assert_eq!(expected, 3);
            assert_eq!(actual, 1);
        }
        other => panic!("expected GeneratorConfigMismatch, got: {other:?}"),
    }
}

// -- From<KokoroError> for TensorError (implicit conversion via ?) ---------

#[test]
fn test_kokoro_error_from_tensor_variant_extracts_inner() {
    // KokoroError::Tensor(te) → te (unwrap inner)
    let inner = nn_core::TensorError::RankMismatch {
        expected: 3,
        actual: 2,
    };
    let kokoro_err = KokoroError::Tensor(inner);
    let te: nn_core::TensorError = kokoro_err.into();
    assert!(
        matches!(
            te,
            nn_core::TensorError::RankMismatch {
                expected: 3,
                actual: 2
            }
        ),
        "should extract inner TensorError, got: {te:?}"
    );
}

#[test]
fn test_kokoro_error_from_non_tensor_wraps_as_unsupported() {
    // Non-Tensor variants wrap as TensorError::Unsupported(display_string).
    let err = KokoroError::InvalidSpeed { value: -5.0 };
    let te: nn_core::TensorError = err.into();
    match te {
        nn_core::TensorError::Unsupported(msg) => {
            assert!(msg.contains("-5"), "should contain the value: {msg}");
            assert!(msg.contains("speed"), "should mention speed: {msg}");
        }
        other => panic!("expected Unsupported, got: {other:?}"),
    }
}

#[test]
fn test_kokoro_error_from_istft_wraps_as_unsupported() {
    let err = KokoroError::IstftFailed(KokoroIstftError::ZeroHop);
    let te: nn_core::TensorError = err.into();
    match te {
        nn_core::TensorError::Unsupported(msg) => {
            assert!(
                msg.contains("hop_length"),
                "should contain inner error message: {msg}"
            );
        }
        other => panic!("expected Unsupported, got: {other:?}"),
    }
}
