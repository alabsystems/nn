// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `DynTensor::resize_bilinear` — bilinear interpolation resize (#3535).

use crate::dyn_tensor::test_helpers::tnd;

/// Helper: assert all values are approximately equal with given tolerance.
fn assert_approx(actual: &[f32], expected: &[f32], tol: f32, ctx: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{ctx}: length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{ctx}[{i}]: got {a}, expected {e}, diff={}",
            (a - e).abs()
        );
    }
}

// -- Identity resize (same size -> same values) --------------------------------

#[test]
fn test_resize_bilinear_identity_3d() {
    // [C, H, W] = [1, 2, 3]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = tnd(&data, &[1, 2, 3]);
    let y = x.resize_bilinear(2, 3).unwrap();
    assert_eq!(y.dims(), &[1, 2, 3]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_resize_bilinear_identity_4d() {
    // [N, C, H, W] = [1, 1, 2, 3]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = tnd(&data, &[1, 1, 2, 3]);
    let y = x.resize_bilinear(2, 3).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 3]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), data);
}

// -- 2x upscale ---------------------------------------------------------------

#[test]
fn test_resize_bilinear_2x_upscale() {
    // [C, H, W] = [1, 2, 2], upscale to 4x4
    // Input:
    //   1.0  2.0
    //   3.0  4.0
    let x = tnd(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let y = x.resize_bilinear(4, 4).unwrap();
    assert_eq!(y.dims(), &[1, 4, 4]);

    let out = y.to_flat_vec::<f32>().unwrap();
    // With half-pixel center mapping, corner values are interpolated inward.
    // All output values should be within [1.0, 4.0] (convex combination).
    for &v in &out {
        assert!((1.0..=4.0).contains(&v), "value {v} out of [1, 4] range");
    }
    // Center-ish pixels should be close to the average (2.5).
    let center_avg = (out[5] + out[6] + out[9] + out[10]) / 4.0;
    assert!(
        (center_avg - 2.5).abs() < 0.5,
        "center avg {center_avg} should be near 2.5"
    );
}

// -- 2x downscale -------------------------------------------------------------

#[test]
fn test_resize_bilinear_2x_downscale() {
    // [C, H, W] = [1, 4, 4], downscale to 2x2
    // Uniform gradient: values 0..15
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let x = tnd(&data, &[1, 4, 4]);
    let y = x.resize_bilinear(2, 2).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2]);

    let out = y.to_flat_vec::<f32>().unwrap();
    // All values should be within [0, 15] (convex combination of source pixels).
    for &v in &out {
        assert!(
            (0.0..=15.0).contains(&v),
            "value {v} out of source range [0, 15]"
        );
    }
    // For a uniform gradient, the output should be close to the mean of
    // each 2x2 quadrant. Quadrant means: 2.5, 4.5, 10.5, 12.5.
    // Half-pixel mapping doesn't produce exact quadrant means but should be close.
    assert_approx(&out, &[2.5, 4.5, 10.5, 12.5], 1.0, "2x_downscale");
}

// -- Non-square resize --------------------------------------------------------

#[test]
fn test_resize_bilinear_non_square() {
    // [C, H, W] = [1, 2, 4], resize to 4x2 (stretch H, shrink W)
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let x = tnd(&data, &[1, 2, 4]);
    let y = x.resize_bilinear(4, 2).unwrap();
    assert_eq!(y.dims(), &[1, 4, 2]);

    let out = y.to_flat_vec::<f32>().unwrap();
    // All 8 output values must be within source range [0, 7].
    for &v in &out {
        assert!((0.0..=7.0).contains(&v), "value {v} out of range");
    }
}

// -- Both [C,H,W] and [N,C,H,W] inputs ---------------------------------------

#[test]
fn test_resize_bilinear_rank3_and_rank4_agree() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    // Rank 3: [C=1, H=3, W=4]
    let x3 = tnd(&data, &[1, 3, 4]);
    let y3 = x3.resize_bilinear(6, 8).unwrap();
    assert_eq!(y3.dims(), &[1, 6, 8]);

    // Rank 4: [N=1, C=1, H=3, W=4]
    let x4 = tnd(&data, &[1, 1, 3, 4]);
    let y4 = x4.resize_bilinear(6, 8).unwrap();
    assert_eq!(y4.dims(), &[1, 1, 6, 8]);

    // Flat data must be identical.
    let v3 = y3.to_flat_vec::<f32>().unwrap();
    let v4 = y4.to_flat_vec::<f32>().unwrap();
    assert_approx(&v3, &v4, 1e-6, "rank3_vs_rank4");
}

// -- Multi-channel batched ----------------------------------------------------

#[test]
fn test_resize_bilinear_batched_multi_channel() {
    // [N=2, C=3, H=2, W=2] -> 4x4
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let x = tnd(&data, &[2, 3, 2, 2]);
    let y = x.resize_bilinear(4, 4).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4, 4]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap().len(), 2 * 3 * 4 * 4);
}

// -- Error cases --------------------------------------------------------------

#[test]
fn test_resize_bilinear_rank2_rejected() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(x.resize_bilinear(4, 4).is_err());
}

#[test]
fn test_resize_bilinear_zero_target_rejected() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    assert!(x.resize_bilinear(0, 4).is_err());
    assert!(x.resize_bilinear(4, 0).is_err());
}

// -- 1x1 input upscale --------------------------------------------------------

#[test]
fn test_resize_bilinear_1x1_upscale() {
    // Single pixel input: all outputs must equal the input value.
    let x = tnd(&[7.5], &[1, 1, 1]);
    let y = x.resize_bilinear(3, 3).unwrap();
    assert_eq!(y.dims(), &[1, 3, 3]);
    let out = y.to_flat_vec::<f32>().unwrap();
    for &v in &out {
        assert!(
            (v - 7.5).abs() < 1e-6,
            "1x1 upscale: all pixels should be 7.5, got {v}"
        );
    }
}

// -- Known-value bilinear check -----------------------------------------------

#[test]
fn test_resize_bilinear_known_values_3x3_to_5x5() {
    // [C=1, H=3, W=3]:
    //   0  1  2
    //   3  4  5
    //   6  7  8
    let data: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let x = tnd(&data, &[1, 3, 3]);
    let y = x.resize_bilinear(5, 5).unwrap();
    assert_eq!(y.dims(), &[1, 5, 5]);

    let out = y.to_flat_vec::<f32>().unwrap();
    // Corners should be close to input corners (0, 2, 6, 8) but pulled
    // inward by half-pixel center mapping.
    // Top-left output pixel: src_y = (0+0.5)*(3/5)-0.5 = -0.2 -> clamped to 0.0
    // src_x = same -> 0.0. So output[0] ~ input[0,0] = 0.0
    assert!((out[0] - 0.0).abs() < 0.01, "top-left: got {}", out[0]);
    // Bottom-right: src_y = (4+0.5)*(3/5)-0.5 = 2.2 -> clamped to 2.0
    // src_x = same -> 2.0. So output[24] ~ input[2,2] = 8.0
    assert!(
        (out[24] - 8.0).abs() < 0.01,
        "bottom-right: got {}",
        out[24]
    );
    // Center output pixel (2,2): src_y = (2+0.5)*(3/5)-0.5 = 1.0
    // src_x = same -> 1.0. So output[12] ~ input[1,1] = 4.0
    assert!((out[12] - 4.0).abs() < 0.01, "center: got {}", out[12]);
}
