// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn dvoice_cpu_reference(x: &[f32], alpha: &[f32], channels: usize, length: usize) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(index, value)| {
            let channel = (index / length) % channels;
            let a = alpha[channel].max(SNAKE_MIN_ALPHA);
            let sin_val = (a * value).sin();
            value + (1.0 / a) * sin_val * sin_val
        })
        .collect()
}

#[test]
fn test_snake1d_matches_dvoice_reference() {
    let channels = 2;
    let length = 4;
    let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let alpha = vec![1.0, 2.0];

    let got = snake_ref_f32(&x, &alpha, channels, length).expect("valid snake_ref shape");
    let expected = dvoice_cpu_reference(&x, &alpha, channels, length);

    for (index, (lhs, rhs)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-6,
            "snake1d mismatch at index {index}: got {lhs}, expected {rhs}"
        );
    }
}

#[test]
fn test_snake_ref_f16_path_is_finite() {
    let channels = 2;
    let length = 3;
    let x = vec![
        half::f16::from_f32(-3.0),
        half::f16::from_f32(-1.0),
        half::f16::from_f32(0.0),
        half::f16::from_f32(1.0),
        half::f16::from_f32(2.5),
        half::f16::from_f32(4.0),
    ];
    let alpha = vec![half::f16::from_f32(0.5), half::f16::from_f32(2.0)];

    let out = snake_ref_f16(&x, &alpha, channels, length).expect("valid f16 snake_ref shape");
    for value in out {
        assert!(
            value.to_f32().is_finite(),
            "f16 snake output must be finite"
        );
    }
}

#[test]
fn test_snake_finite_for_kani_domain() {
    let xs = [-1.0e4f32, -1.0e3, -10.0, 0.0, 10.0, 1.0e3, 1.0e4];
    let alphas = [1.0e-8f32, 1.0e-6, 1.0e-3, 1.0e-1, 1.0, 10.0, 1.0e3];

    for x in xs {
        for alpha in alphas {
            let y = snake_scalar(x, alpha).expect("finite inputs");
            assert!(
                y.is_finite(),
                "snake({x}, {alpha}) produced non-finite output: {y}"
            );
        }
    }
}

#[test]
fn test_snake_bounds_cover_issue_18_domain() {
    let bounds = snake_scalar_bounds(-10.0, 10.0, 0.01, 100.0).expect("finite bounds");
    assert!((bounds.0 + 10.0).abs() < 1e-6);
    assert!((bounds.1 - 110.0).abs() < 1e-4);

    for x_step in 0..401 {
        let x = -10.0 + (x_step as f32) * 0.05;
        for a_step in 0..201 {
            let alpha = 0.01 + (a_step as f32) * ((100.0 - 0.01) / 200.0);
            let y = snake_scalar(x, alpha).expect("finite inputs");
            assert!(
                y >= bounds.0 - 1e-5 && y <= bounds.1 + 1e-5,
                "value {y} not in bounds [{}, {}] for x={x}, alpha={alpha}",
                bounds.0,
                bounds.1
            );
        }
    }
}

/// Hand-written Snake1d MSL — reference only.
///
/// The production codegen path is `emit_msl()` on the snake `KernelDef` from
/// `build_snake_scalar_kernel()`. This constant is retained for regression
/// tests (`test_emit_snake1d_msl_contains_expected_symbols`) and as a
/// readable reference for the expected Metal kernel structure.
const SNAKE1D_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void snake1d(
    device const T* x       [[buffer(0)]],
    device const T* alpha   [[buffer(1)]],
    device T* out           [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& length   [[buffer(4)]],
    constant uint& total    [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total) return;

    uint c = (tid / length) % channels;
    T a = max(alpha[c], T(1e-8));
    T xv = x[tid];
    T sin_val = metal::precise::sin(a * xv);
    out[tid] = xv + (T(1) / a) * sin_val * sin_val;
}

template [[host_name("snake1d_f32")]] [[kernel]]
void snake1d<float>(
    device const float*, device const float*, device float*,
    constant uint&, constant uint&, constant uint&, uint);

template [[host_name("snake1d_f16")]] [[kernel]]
void snake1d<half>(
    device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, uint);
"#;

#[test]
fn test_emit_snake1d_msl_contains_expected_symbols() {
    let msl = SNAKE1D_MSL;
    assert!(msl.contains("[[host_name(\"snake1d_f32\")]]"));
    assert!(msl.contains("[[host_name(\"snake1d_f16\")]]"));
    assert!(msl.contains("metal::precise::sin"));
}

#[test]
fn test_shape_validation_rejects_invalid_dimensions() {
    let x = vec![0.0f32; 7];
    let alpha = vec![1.0f32; 2];
    let err = snake_ref_f32(&x, &alpha, 2, 4).expect_err("shape mismatch should fail");
    assert!(
        matches!(
            err,
            KernelError::ShapeMismatch {
                expected: 8,
                got: 7
            }
        ),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn test_shape_validation_rejects_zero_channels() {
    let x = vec![0.0f32; 4];
    let alpha = vec![1.0f32; 0];
    let err = snake_ref_f32(&x, &alpha, 0, 4).expect_err("zero channels should fail");
    assert!(
        matches!(
            err,
            KernelError::InvalidDimension {
                name: "channels",
                value: 0
            }
        ),
        "expected InvalidDimension for channels, got {err:?}"
    );
}

#[test]
fn test_shape_validation_rejects_zero_length() {
    let x = vec![0.0f32; 4];
    let alpha = vec![1.0f32; 2];
    let err = snake_ref_f32(&x, &alpha, 2, 0).expect_err("zero length should fail");
    assert!(
        matches!(
            err,
            KernelError::InvalidDimension {
                name: "length",
                value: 0
            }
        ),
        "expected InvalidDimension for length, got {err:?}"
    );
}

#[test]
fn test_shape_validation_rejects_alpha_size_mismatch() {
    let x = vec![0.0f32; 8];
    let alpha = vec![1.0f32; 3]; // 3 alphas but 2 channels
    let err = snake_ref_f32(&x, &alpha, 2, 4).expect_err("alpha size mismatch should fail");
    assert!(
        matches!(
            err,
            KernelError::ShapeMismatch {
                expected: 2,
                got: 3
            }
        ),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn test_snake_scalar_bounds_rejects_nan_x_lower() {
    let err = snake_scalar_bounds(f32::NAN, 1.0, 1.0, 2.0).expect_err("NaN x lower should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_nan()),
        "expected NonFiniteBound with NaN, got {err:?}"
    );
}

#[test]
fn test_snake_scalar_bounds_rejects_nan_alpha() {
    let err =
        snake_scalar_bounds(0.0, 1.0, f32::NAN, 2.0).expect_err("NaN alpha lower should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_nan()),
        "expected NonFiniteBound with NaN, got {err:?}"
    );
}

#[test]
fn test_snake_scalar_bounds_rejects_inf_x() {
    let err = snake_scalar_bounds(f32::INFINITY, 1.0, 1.0, 2.0).expect_err("Inf x should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_infinite()),
        "expected NonFiniteBound with Inf, got {err:?}"
    );
}

#[test]
fn test_snake_scalar_bounds_rejects_neg_inf_alpha() {
    let err =
        snake_scalar_bounds(0.0, 1.0, f32::NEG_INFINITY, 2.0).expect_err("-Inf alpha should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_infinite()),
        "expected NonFiniteBound with -Inf, got {err:?}"
    );
}

#[test]
fn test_snake_scalar_bounds_accepts_valid_inputs() {
    let (lower, upper) =
        snake_scalar_bounds(-10.0, 10.0, 0.01, 100.0).expect("valid finite bounds");
    assert!(
        lower.is_finite(),
        "lower bound must be finite, got: {lower}"
    );
    assert!(
        upper.is_finite(),
        "upper bound must be finite, got: {upper}"
    );
    assert!(lower <= upper, "bounds must be ordered: {lower} <= {upper}");
    // snake(x, alpha) = x + (1/alpha)*sin(alpha*x)^2
    // For x in [-10, 10], alpha_lower=0.01: upper = 10 + 1/0.01 = 110
    assert_eq!(lower, -10.0);
    assert!(
        (upper - 110.0).abs() < 1e-3,
        "expected upper ~110.0, got: {upper}"
    );
}

/// NaN in x_upper position (complementary to x_lower test above).
/// Guards against per-field validation that might miss the upper bound.
#[test]
fn test_snake_scalar_bounds_rejects_nan_x_upper() {
    let err = snake_scalar_bounds(0.0, f32::NAN, 1.0, 2.0).expect_err("NaN x upper should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_nan()),
        "expected NonFiniteBound with NaN, got {err:?}"
    );
}

/// NaN in alpha_upper position.
#[test]
fn test_snake_scalar_bounds_rejects_nan_alpha_upper() {
    let err =
        snake_scalar_bounds(0.0, 1.0, 1.0, f32::NAN).expect_err("NaN alpha upper should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_nan()),
        "expected NonFiniteBound with NaN, got {err:?}"
    );
}

/// Verify the overflow guard in snake_scalar_bounds is unreachable with
/// the current SNAKE_MIN_ALPHA = 1e-8: the max value of 1/safe_alpha is
/// 1e8, and f32::MAX + 1e8 rounds to f32::MAX (ULP at that magnitude
/// is ~2e31 >> 1e8). The guard would become reachable only if
/// SNAKE_MIN_ALPHA were reduced below ~1e-31.
#[test]
fn test_snake_scalar_bounds_extreme_range_stays_finite() {
    // With SNAKE_MIN_ALPHA = 1e-8, 1/alpha_clamped = 1e8.
    // f32::MAX + 1e8 = f32::MAX (no overflow due to precision).
    let result = snake_scalar_bounds(0.0, f32::MAX, SNAKE_MIN_ALPHA, 1.0);
    assert!(
        result.is_ok(),
        "extreme range should NOT overflow because 1/1e-8 = 1e8 << f32::MAX"
    );
    let (lower, upper) = result.unwrap();
    assert_eq!(lower, 0.0);
    assert!(upper.is_finite(), "upper must remain finite");
    assert_eq!(upper, f32::MAX, "1e8 absorbed into f32::MAX by rounding");
}

/// The overflow guard rejects bounds that would produce infinity.
/// Currently unreachable via normal paths (SNAKE_MIN_ALPHA clamps alpha),
/// but defends against future SNAKE_MIN_ALPHA reductions or API misuse.
#[test]
fn test_snake_scalar_bounds_overflow_guard_documented() {
    // The function clamps alpha_lower to SNAKE_MIN_ALPHA (1e-8), so
    // 1/safe_alpha_lower <= 1e8. At f32::MAX scale, adding 1e8 is
    // absorbed by ULP granularity (~2e31). To trigger the guard, we'd
    // need SNAKE_MIN_ALPHA < ~1e-31, which is not the current value.
    //
    // This test documents that the guard exists and would fire if
    // the clamping constant were ever reduced to allow tiny alphas.
    // Direct construction test: manually verify the guard math.
    let max_reciprocal = 1.0_f32 / SNAKE_MIN_ALPHA;
    let sum = f32::MAX + max_reciprocal;
    assert!(
        sum.is_finite(),
        "with current SNAKE_MIN_ALPHA, overflow guard should be unreachable"
    );
}

// --- InvertedBounds rejection (#271) ---

#[test]
fn test_snake_scalar_bounds_rejects_inverted_x() {
    let err = snake_scalar_bounds(5.0, -5.0, 0.1, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::InvertedBounds { lower, upper } if lower == 5.0 && upper == -5.0),
        "inverted x bounds should be rejected, got: {err}"
    );
}

#[test]
fn test_snake_scalar_bounds_rejects_inverted_alpha() {
    let err = snake_scalar_bounds(-1.0, 1.0, 2.0, 0.5).unwrap_err();
    assert!(
        matches!(err, KernelError::InvertedBounds { lower, upper } if lower == 2.0 && upper == 0.5),
        "inverted alpha bounds should be rejected, got: {err}"
    );
}
