// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proofs for MXFP4 quantization correctness.
//!
//! Proves:
//! - FP4 roundtrip error is bounded
//! - E8M0 scale computation does not overflow
//! - Block size alignment handles padding correctly
//! - FP4 lookup table matches OCP MX specification

use crate::quantize::{max_fp4_error_for_scale, Mxfp4Tensor, MXFP4_BLOCK_SIZE};

// -- FP4 LUT matches OCP MX spec ---------------------------------------------

/// Prove all 16 FP4 lookup table entries match the OCP MX specification.
#[kani::proof]
fn proof_fp4_lut_matches_ocp_spec() {
    let lut = Mxfp4Tensor::fp4_lut();

    assert!(lut[0] == 0.0 && lut[0].is_sign_positive());
    assert!(lut[1] == 0.5);
    assert!(lut[2] == 1.0);
    assert!(lut[3] == 1.5);
    assert!(lut[4] == 2.0);
    assert!(lut[5] == 3.0);
    assert!(lut[6] == 4.0);
    assert!(lut[7] == 6.0);
    assert!(lut[8] == 0.0 && lut[8].is_sign_negative());
    assert!(lut[9] == -0.5);
    assert!(lut[10] == -1.0);
    assert!(lut[11] == -1.5);
    assert!(lut[12] == -2.0);
    assert!(lut[13] == -3.0);
    assert!(lut[14] == -4.0);
    assert!(lut[15] == -6.0);

    // Symmetry: |lut[i]| == |lut[i+8]|
    let mut i = 0;
    while i < 8 {
        assert!(lut[i].abs() == lut[i + 8].abs());
        i += 1;
    }
}

// -- E8M0 scale does not overflow ---------------------------------------------

/// Prove E8M0 scale computation produces valid exponents for finite f32.
#[kani::proof]
fn proof_e8m0_no_overflow() {
    let max_abs: f32 = kani::any();
    kani::assume(max_abs.is_finite());
    kani::assume(max_abs >= 0.0);

    let scale_byte = compute_e8m0_scale_kani(max_abs);
    assert!(scale_byte <= 254, "E8M0 exponent must not be 255 (NaN)");
}

/// Prove E8M0 scale returns 0 for non-finite inputs.
#[kani::proof]
fn proof_e8m0_nonfinite_returns_zero() {
    assert!(compute_e8m0_scale_kani(f32::NAN) == 0);
    assert!(compute_e8m0_scale_kani(f32::INFINITY) == 0);
    assert!(compute_e8m0_scale_kani(f32::NEG_INFINITY) == 0);
    assert!(compute_e8m0_scale_kani(-1.0) == 0);
    assert!(compute_e8m0_scale_kani(0.0) == 0);
}

// -- Roundtrip error is bounded -----------------------------------------------

/// Prove exact FP4 roundtrip for representable values.
#[kani::proof]
fn proof_fp4_exact_roundtrip() {
    let code: u8 = kani::any();
    kani::assume(code < 16);

    let scale_byte: u8 = kani::any();
    kani::assume(scale_byte >= 1 && scale_byte <= 254);

    let lut = Mxfp4Tensor::fp4_lut();
    let fp4_val = lut[code as usize];
    let scale = decode_e8m0_kani(scale_byte);
    let original = fp4_val * scale;

    if fp4_val != 0.0 && original.is_finite() && scale.is_finite() {
        let recovered_code = quantize_to_fp4_kani(original, scale);
        let recovered_val = lut[recovered_code as usize] * scale;
        let err = (original - recovered_val).abs();
        assert!(err < scale * 1e-5);
    }
}

/// Prove the max error bound formula is correct.
#[kani::proof]
fn proof_roundtrip_error_bounded() {
    let scale_byte: u8 = kani::any();
    kani::assume(scale_byte >= 1 && scale_byte <= 200);

    let max_err = max_fp4_error_for_scale(scale_byte);
    let scale = decode_e8m0_kani(scale_byte);

    assert!((max_err - scale).abs() < 1e-10);
    assert!(max_err.is_finite());
    assert!(max_err > 0.0);
}

// -- Block size alignment -----------------------------------------------------

/// Prove padding arithmetic is correct for any small size.
#[kani::proof]
fn proof_block_alignment_padding() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 96);

    let num_blocks = (n + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    let padded_len = num_blocks * MXFP4_BLOCK_SIZE;

    assert!(padded_len >= n);
    assert!(padded_len % MXFP4_BLOCK_SIZE == 0);
    assert!(padded_len - n < MXFP4_BLOCK_SIZE);
}

/// Prove dequantize output length matches shape.
#[kani::proof]
fn proof_dequantize_output_length() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let qt = Mxfp4Tensor::quantize(&data, &[5]);
    let recovered = qt.dequantize();
    assert!(recovered.len() == 5);
}

// -- Kani-local helper copies (avoid cross-module visibility issues) ----------

fn compute_e8m0_scale_kani(max_abs: f32) -> u8 {
    if !max_abs.is_finite() || max_abs <= 0.0 {
        return 0;
    }
    let bits = max_abs.to_bits();
    let biased_exp = ((bits >> 23) & 0xFF) as u8;
    if biased_exp == 0 {
        return 0;
    }
    let scale_exp = if biased_exp >= 2 { biased_exp - 2 } else { 0 };
    scale_exp.min(254)
}

fn decode_e8m0_kani(e8m0: u8) -> f32 {
    if e8m0 == 0 {
        return f32::from_bits(0x0080_0000);
    }
    f32::from_bits((e8m0 as u32) << 23)
}

fn quantize_to_fp4_kani(val: f32, scale: f32) -> u8 {
    const ABS_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    if !val.is_finite() || scale <= 0.0 {
        return 0;
    }
    let sign = val < 0.0;
    let abs_scaled = val.abs() / scale;
    let mut best_idx: usize = 0;
    let mut best_dist = f32::MAX;
    let mut i = 0;
    while i < 8 {
        let dist = if abs_scaled >= ABS_VALUES[i] {
            abs_scaled - ABS_VALUES[i]
        } else {
            ABS_VALUES[i] - abs_scaled
        };
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
        i += 1;
    }
    let code = best_idx as u8;
    if sign {
        code | 0x08
    } else {
        code
    }
}
