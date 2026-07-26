// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD dtype casting operations.

use super::*;

// ---------------------------------------------------------------------------
// bf16 roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_f32_to_bf16_roundtrip() {
    let values = [1.0f32, 2.0, 0.5, -3.5, 100.0, -0.125, 42.0, 0.001];
    let mut bf16_buf = [0u16; 8];
    let mut roundtrip = [0.0f32; 8];

    f32_to_bf16(&values, &mut bf16_buf);
    bf16_to_f32(&bf16_buf, &mut roundtrip);

    for (i, (&orig, &rt)) in values.iter().zip(roundtrip.iter()).enumerate() {
        // bf16 has ~7 bits of mantissa, so relative error is about 1/128 ~ 0.008
        let tol = orig.abs() * 0.01 + 1e-6;
        assert!(
            (orig - rt).abs() < tol,
            "bf16 roundtrip index {i}: orig={orig}, roundtrip={rt}, diff={}",
            (orig - rt).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// f16 roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_f32_to_f16_roundtrip() {
    let values = [1.0f32, 2.0, 0.5, -3.5, 100.0, -0.125, 42.0, 0.001];
    let mut f16_buf = [0u16; 8];
    let mut roundtrip = [0.0f32; 8];

    f32_to_f16(&values, &mut f16_buf);
    f16_to_f32(&f16_buf, &mut roundtrip);

    for (i, (&orig, &rt)) in values.iter().zip(roundtrip.iter()).enumerate() {
        // f16 has 10-bit mantissa, relative error about 1/1024 ~ 0.001
        let tol = orig.abs() * 0.002 + 1e-4;
        assert!(
            (orig - rt).abs() < tol,
            "f16 roundtrip index {i}: orig={orig}, roundtrip={rt}, diff={}",
            (orig - rt).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// bf16 zero
// ---------------------------------------------------------------------------

#[test]
fn test_bf16_zero() {
    let input = [0.0f32];
    let mut bf16_buf = [0xFFFFu16; 1];
    let mut output = [99.0f32; 1];

    f32_to_bf16(&input, &mut bf16_buf);
    bf16_to_f32(&bf16_buf, &mut output);

    assert_eq!(
        output[0], 0.0,
        "bf16 zero roundtrip failed: got {}",
        output[0]
    );
    assert!(
        output[0].is_sign_positive(),
        "bf16 zero should be positive zero"
    );
}

// ---------------------------------------------------------------------------
// bf16 negative
// ---------------------------------------------------------------------------

#[test]
fn test_bf16_negative() {
    let input = [-1.0f32, -0.5, -100.0, -0.001];
    let mut bf16_buf = [0u16; 4];
    let mut output = [0.0f32; 4];

    f32_to_bf16(&input, &mut bf16_buf);
    bf16_to_f32(&bf16_buf, &mut output);

    for (i, (&orig, &rt)) in input.iter().zip(output.iter()).enumerate() {
        assert!(
            rt < 0.0,
            "bf16 negative index {i}: expected negative, got {rt}"
        );
        let tol = orig.abs() * 0.01 + 1e-6;
        assert!(
            (orig - rt).abs() < tol,
            "bf16 negative index {i}: orig={orig}, roundtrip={rt}, diff={}",
            (orig - rt).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// Various values
// ---------------------------------------------------------------------------

#[test]
fn test_cast_various_values() {
    let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, -100.0];

    // bf16 roundtrip
    {
        let mut bf16_buf = [0u16; 6];
        let mut output = [0.0f32; 6];
        f32_to_bf16(&values, &mut bf16_buf);
        bf16_to_f32(&bf16_buf, &mut output);

        for (i, (&orig, &rt)) in values.iter().zip(output.iter()).enumerate() {
            let tol = orig.abs() * 0.01 + 1e-6;
            assert!(
                (orig - rt).abs() < tol,
                "bf16 various index {i}: orig={orig}, roundtrip={rt}"
            );
        }
    }

    // f16 roundtrip
    {
        let mut f16_buf = [0u16; 6];
        let mut output = [0.0f32; 6];
        f32_to_f16(&values, &mut f16_buf);
        f16_to_f32(&f16_buf, &mut output);

        for (i, (&orig, &rt)) in values.iter().zip(output.iter()).enumerate() {
            let tol = orig.abs() * 0.002 + 1e-4;
            assert!(
                (orig - rt).abs() < tol,
                "f16 various index {i}: orig={orig}, roundtrip={rt}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar vs dispatch consistency
// ---------------------------------------------------------------------------

#[test]
fn test_f32_to_bf16_dispatch_matches_scalar() {
    let input: Vec<f32> = (0..33).map(|i| (i as f32 - 16.0) * 0.3).collect();
    let mut out_dispatch = vec![0u16; 33];
    let mut out_scalar = vec![0u16; 33];
    f32_to_bf16(&input, &mut out_dispatch);
    f32_to_bf16_scalar(&input, &mut out_scalar);
    assert_eq!(out_dispatch, out_scalar);
}

#[test]
fn test_bf16_to_f32_dispatch_matches_scalar() {
    // Create bf16 values from known f32 values
    let f32_input: Vec<f32> = (0..33).map(|i| (i as f32 - 16.0) * 0.3).collect();
    let mut bf16_buf = vec![0u16; 33];
    f32_to_bf16_scalar(&f32_input, &mut bf16_buf);

    let mut out_dispatch = vec![0.0f32; 33];
    let mut out_scalar = vec![0.0f32; 33];
    bf16_to_f32(&bf16_buf, &mut out_dispatch);
    bf16_to_f32_scalar(&bf16_buf, &mut out_scalar);

    for (i, (&d, &s)) in out_dispatch.iter().zip(out_scalar.iter()).enumerate() {
        assert!(
            (d - s).abs() < 1e-10,
            "bf16_to_f32 mismatch at index {i}: dispatch={d}, scalar={s}"
        );
    }
}

#[test]
fn test_f32_to_f16_dispatch_matches_scalar() {
    let input: Vec<f32> = (0..33).map(|i| (i as f32 - 16.0) * 0.3).collect();
    let mut out_dispatch = vec![0u16; 33];
    let mut out_scalar = vec![0u16; 33];
    f32_to_f16(&input, &mut out_dispatch);
    f32_to_f16_scalar(&input, &mut out_scalar);
    assert_eq!(out_dispatch, out_scalar);
}

#[test]
fn test_f16_to_f32_dispatch_matches_scalar() {
    let f32_input: Vec<f32> = (0..33).map(|i| (i as f32 - 16.0) * 0.3).collect();
    let mut f16_buf = vec![0u16; 33];
    f32_to_f16_scalar(&f32_input, &mut f16_buf);

    let mut out_dispatch = vec![0.0f32; 33];
    let mut out_scalar = vec![0.0f32; 33];
    f16_to_f32(&f16_buf, &mut out_dispatch);
    f16_to_f32_scalar(&f16_buf, &mut out_scalar);

    for (i, (&d, &s)) in out_dispatch.iter().zip(out_scalar.iter()).enumerate() {
        assert!(
            (d - s).abs() < 1e-10,
            "f16_to_f32 mismatch at index {i}: dispatch={d}, scalar={s}"
        );
    }
}

// ---------------------------------------------------------------------------
// Empty input
// ---------------------------------------------------------------------------

#[test]
fn test_cast_empty_input() {
    let empty_f32: &[f32] = &[];
    let empty_u16: &[u16] = &[];
    let mut out_u16: Vec<u16> = vec![];
    let mut out_f32: Vec<f32> = vec![];

    f32_to_f16(empty_f32, &mut out_u16);
    f16_to_f32(empty_u16, &mut out_f32);
    f32_to_bf16(empty_f32, &mut out_u16);
    bf16_to_f32(empty_u16, &mut out_f32);

    assert!(out_u16.is_empty());
    assert!(out_f32.is_empty());
}

// ---------------------------------------------------------------------------
// NEON-specific tests (aarch64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[test]
fn test_f32_to_bf16_neon_matches_scalar() {
    let input: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 1.5).collect();
    let mut out_neon = vec![0u16; 17];
    let mut out_scalar = vec![0u16; 17];
    f32_to_bf16_neon(&input, &mut out_neon);
    f32_to_bf16_scalar(&input, &mut out_scalar);
    assert_eq!(out_neon, out_scalar);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_bf16_to_f32_neon_matches_scalar() {
    let f32_input: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 1.5).collect();
    let mut bf16_buf = vec![0u16; 17];
    f32_to_bf16_scalar(&f32_input, &mut bf16_buf);

    let mut out_neon = vec![0.0f32; 17];
    let mut out_scalar = vec![0.0f32; 17];
    bf16_to_f32_neon(&bf16_buf, &mut out_neon);
    bf16_to_f32_scalar(&bf16_buf, &mut out_scalar);

    for (i, (&n, &s)) in out_neon.iter().zip(out_scalar.iter()).enumerate() {
        assert!(
            (n - s).abs() < 1e-10,
            "bf16_to_f32 NEON mismatch at index {i}: neon={n}, scalar={s}"
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_f32_to_f16_neon_matches_scalar() {
    let input: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 1.5).collect();
    let mut out_neon = vec![0u16; 17];
    let mut out_scalar = vec![0u16; 17];
    f32_to_f16_neon(&input, &mut out_neon);
    f32_to_f16_scalar(&input, &mut out_scalar);
    // NEON vcvt_f16_f32 may differ slightly in rounding from scalar; compare via roundtrip
    let mut rt_neon = vec![0.0f32; 17];
    let mut rt_scalar = vec![0.0f32; 17];
    f16_to_f32_scalar(&out_neon, &mut rt_neon);
    f16_to_f32_scalar(&out_scalar, &mut rt_scalar);
    for (i, (&n, &s)) in rt_neon.iter().zip(rt_scalar.iter()).enumerate() {
        assert!(
            (n - s).abs() < 1e-3,
            "f32_to_f16 NEON roundtrip mismatch at index {i}: neon={n}, scalar={s}"
        );
    }
}
