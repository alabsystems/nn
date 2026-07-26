// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

// -- Zero handling ------------------------------------------------------------

#[test]
fn test_mxfp4_quantize_dequantize_zeros() {
    let values = [0.0_f32; MXFP4_BLOCK_SIZE];
    let block = quantize_block(&values).unwrap();
    let output = dequantize_block(&block);

    for (i, &v) in output.iter().enumerate() {
        assert_eq!(v, 0.0, "element {i} should be zero, got {v}");
    }
}

// -- Roundtrip accuracy -------------------------------------------------------

#[test]
fn test_mxfp4_roundtrip_small_values() {
    // Small values that should be representable with reasonable accuracy.
    let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
    for i in 0..MXFP4_BLOCK_SIZE {
        values[i] = (i as f32 / 31.0) * 2.0 - 1.0; // range [-1, 1]
    }

    let block = quantize_block(&values).unwrap();
    let output = dequantize_block(&block);

    // MXFP4 is very low precision (4-bit), so expect coarse quantization.
    // The maximum representable magnitude in E1M2 is 6.0, and we have 8 levels.
    // For values in [-1, 1], the relative error can be significant.
    for &v in &output {
        assert!(v.is_finite(), "all output values must be finite");
    }

    // The max error should be bounded by the quantization step size.
    let max_err = values
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // With 8 magnitude levels and shared exponent, error should be < 1.0 for
    // inputs in [-1, 1] range.
    assert!(
        max_err < 1.0,
        "roundtrip max error for [-1,1] range: {max_err}"
    );
}

#[test]
fn test_mxfp4_roundtrip_large_values() {
    let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
    for i in 0..MXFP4_BLOCK_SIZE {
        values[i] = (i as f32 - 16.0) * 100.0; // range [-1600, 1500]
    }

    let block = quantize_block(&values).unwrap();
    let output = dequantize_block(&block);

    for &v in &output {
        assert!(v.is_finite(), "all output values must be finite");
    }
}

#[test]
fn test_mxfp4_roundtrip_identical_values() {
    // All identical nonzero values — tests constant-block path.
    let values = [3.14_f32; MXFP4_BLOCK_SIZE];

    let block = quantize_block(&values).unwrap();
    let output = dequantize_block(&block);

    // All outputs should be the same (since all inputs are the same).
    let first = output[0];
    for (i, &v) in output.iter().enumerate() {
        assert_eq!(
            v, first,
            "element {i} should match element 0: {v} != {first}"
        );
    }
    assert!(first.is_finite());
    // The dequantized value should be close to 3.14.
    assert!(
        (first - 3.14).abs() < 2.0,
        "constant 3.14 should quantize close: got {first}"
    );
}

#[test]
fn test_mxfp4_roundtrip_negative_values() {
    let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
    for i in 0..MXFP4_BLOCK_SIZE {
        values[i] = -(i as f32 + 1.0) * 0.1; // [-0.1, -3.2]
    }

    let block = quantize_block(&values).unwrap();
    let output = dequantize_block(&block);

    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "element {i} must be finite");
        // All nonzero inputs are negative, outputs should be non-positive.
        assert!(v <= 0.0, "element {i} should be non-positive: {v}");
    }
}

// -- Overflow clamping --------------------------------------------------------

#[test]
fn test_mxfp4_overflow_clamping() {
    // Values with extreme dynamic range within one block.
    // The shared exponent must accommodate the largest value,
    // so small values will be quantized to zero.
    let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
    values[0] = 1e10;
    values[1] = 1e-10; // should clamp to 0 due to shared exponent

    let block = quantize_block(&values).unwrap();
    let output = dequantize_block(&block);

    assert!(output[0].is_finite());
    assert!(output[1].is_finite());

    // The tiny value should be quantized to 0 (below representable range).
    assert_eq!(
        output[1], 0.0,
        "tiny value should be quantized to 0: got {}",
        output[1]
    );
}

// -- Non-finite rejection -----------------------------------------------------

#[test]
fn test_mxfp4_nan_rejected() {
    let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
    values[5] = f32::NAN;

    let result = quantize_block(&values);
    assert!(result.is_err(), "NaN input should be rejected");
}

#[test]
fn test_mxfp4_inf_rejected() {
    let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
    values[10] = f32::INFINITY;

    let result = quantize_block(&values);
    assert!(result.is_err(), "Inf input should be rejected");
}

#[test]
fn test_mxfp4_neg_inf_rejected() {
    let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
    values[0] = f32::NEG_INFINITY;

    let result = quantize_block(&values);
    assert!(result.is_err(), "Neg Inf input should be rejected");
}

// -- Tensor-level operations --------------------------------------------------

#[test]
fn test_mxfp4_tensor_exact_block_boundary() {
    let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    assert_eq!(data.len() % MXFP4_BLOCK_SIZE, 0);

    let tensor = quantize_tensor(&data).unwrap();
    assert_eq!(tensor.num_blocks(), 2);
    assert_eq!(tensor.original_len(), 64);

    let output = dequantize_tensor(&tensor);
    assert_eq!(output.len(), 64);

    for &v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_mxfp4_tensor_non_aligned_length() {
    // 50 elements = 1 full block (32) + 1 partial block (18 + 14 padding zeros).
    let data: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) * 0.1).collect();

    let tensor = quantize_tensor(&data).unwrap();
    assert_eq!(tensor.num_blocks(), 2);
    assert_eq!(tensor.original_len(), 50);

    let output = dequantize_tensor(&tensor);
    assert_eq!(
        output.len(),
        50,
        "output should be truncated to original length"
    );

    for &v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_mxfp4_tensor_single_element() {
    let data = vec![42.0_f32];
    let tensor = quantize_tensor(&data).unwrap();
    assert_eq!(tensor.num_blocks(), 1);
    assert_eq!(tensor.original_len(), 1);

    let output = dequantize_tensor(&tensor);
    assert_eq!(output.len(), 1);
    assert!(output[0].is_finite());
}

#[test]
fn test_mxfp4_tensor_empty() {
    let data: Vec<f32> = vec![];
    let tensor = quantize_tensor(&data).unwrap();
    assert_eq!(tensor.num_blocks(), 0);
    assert_eq!(tensor.original_len(), 0);

    let output = dequantize_tensor(&tensor);
    assert!(output.is_empty());
}

#[test]
fn test_mxfp4_tensor_nan_in_data() {
    let mut data = vec![0.0_f32; 64];
    data[33] = f32::NAN;

    let result = quantize_tensor(&data);
    assert!(result.is_err(), "NaN in tensor data should be rejected");
}

// -- Compression ratio --------------------------------------------------------

#[test]
fn test_mxfp4_compression_ratio() {
    let data: Vec<f32> = vec![1.0; 1024];
    let tensor = quantize_tensor(&data).unwrap();

    assert_eq!(tensor.num_blocks(), 32); // 1024 / 32
    assert_eq!(tensor.storage_bytes(), 32 * 17); // 32 blocks * 17 bytes/block = 544
    assert_eq!(tensor.f32_storage_bytes(), 1024 * 4); // 4096

    let ratio = tensor.compression_ratio();
    // 4096 / 544 = 7.53x — much better than INT8 (4x) due to 4-bit storage.
    assert!(
        ratio > 7.0,
        "compression ratio should be ~7.5x, got {ratio:.2}"
    );
}

// -- E1M2 encoding details ----------------------------------------------------

#[test]
fn test_e1m2_magnitudes_exact_roundtrip() {
    // The 8 canonical E1M2 magnitudes should roundtrip exactly when
    // block_scale = 1.0 (shared_exp = 127).
    let block_scale = 1.0_f32;

    for (idx, &expected) in E1M2_MAGNITUDES.iter().enumerate() {
        let code = encode_e1m2(expected, block_scale);
        let decoded = decode_e1m2(code, block_scale);
        assert_eq!(
            decoded, expected,
            "E1M2 magnitude {idx} ({expected}) should roundtrip exactly, got {decoded}"
        );

        // Also test negative
        if expected != 0.0 {
            let code_neg = encode_e1m2(-expected, block_scale);
            let decoded_neg = decode_e1m2(code_neg, block_scale);
            assert_eq!(
                decoded_neg, -expected,
                "negative E1M2 magnitude {idx} ({}) should roundtrip exactly, got {decoded_neg}",
                -expected
            );
        }
    }
}

#[test]
fn test_e1m2_zero_block_scale() {
    // block_scale = 0 (shared_exp = 0) should encode everything as 0.
    let code = encode_e1m2(42.0, 0.0);
    assert_eq!(code, 0, "zero block_scale should encode to 0");
}

#[test]
fn test_block_scale_from_exp_values() {
    // shared_exp = 127 => scale = 2^0 = 1.0
    assert_eq!(block_scale_from_exp(127), 1.0);
    // shared_exp = 128 => scale = 2^1 = 2.0
    assert_eq!(block_scale_from_exp(128), 2.0);
    // shared_exp = 126 => scale = 2^-1 = 0.5
    assert_eq!(block_scale_from_exp(126), 0.5);
    // shared_exp = 0 => scale = 2^-127 (very small but positive)
    let scale_0 = block_scale_from_exp(0);
    assert!(scale_0 > 0.0, "scale for exp=0 must be positive");
    assert!(scale_0.is_finite(), "scale for exp=0 must be finite");
}

// -- Sign preservation ---------------------------------------------------------

#[test]
fn test_mxfp4_sign_preservation() {
    let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
    // Alternating positive and negative values.
    for i in 0..MXFP4_BLOCK_SIZE {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        values[i] = sign * (i as f32 + 1.0) * 0.5;
    }

    let block = quantize_block(&values).unwrap();
    let output = dequantize_block(&block);

    for i in 0..MXFP4_BLOCK_SIZE {
        if values[i] > 0.0 {
            assert!(
                output[i] >= 0.0,
                "element {i}: positive input {}, negative output {}",
                values[i],
                output[i]
            );
        } else if values[i] < 0.0 {
            assert!(
                output[i] <= 0.0,
                "element {i}: negative input {}, positive output {}",
                values[i],
                output[i]
            );
        }
    }
}

// -- Shared exponent computation ----------------------------------------------

#[test]
fn test_shared_exponent_all_zeros() {
    let values = [0.0_f32; MXFP4_BLOCK_SIZE];
    let exp = compute_shared_exponent(&values);
    assert_eq!(exp, 0, "all-zero block should have shared_exp = 0");
}

#[test]
fn test_shared_exponent_unit_range() {
    // Values in [-1, 1]; max_abs = 1.0.
    // Need: 1.0 <= 6.0 * 2^(exp - 127)
    // => 2^(exp - 127) >= 1/6
    // => exp - 127 >= log2(1/6) ~ -2.585
    // => exp >= 124.4 => exp = 125 (ceil)
    let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
    values[0] = 1.0;
    values[1] = -1.0;

    let exp = compute_shared_exponent(&values);
    // The exponent should be chosen such that max_abs fits within E1M2 range.
    let block_scale = block_scale_from_exp(exp);
    let max_representable = E1M2_MAX_MAGNITUDE * block_scale;
    assert!(
        max_representable >= 1.0,
        "block scale should accommodate max_abs=1.0: max_representable={max_representable}"
    );
}
