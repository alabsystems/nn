// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for window attention utilities (#2439).
//!
//! Verifies end-to-end properties for the Qwen2.5-VL use case:
//! - Window partition -> per-window operation -> unpartition roundtrip
//! - Correctness with non-trivial data (not just shapes)
//! - Large grid sizes matching real ViT configurations
//! - Interaction with batch processing

use super::{window_partition, window_unpartition};
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

fn det_data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.017).sin() * 0.5)
        .collect()
}

// -- Roundtrip with per-window transformation ---------------------------------

/// Simulate the Qwen2.5-VL pattern: partition -> add bias per-window -> unpartition.
/// Verify that adding a per-window bias and then unpartitioning produces the
/// expected spatial result.
#[test]
fn test_window_partition_transform_unpartition() {
    let (h, w, d, ws) = (4, 4, 8, 2);
    let b = 1;
    let data = det_data(b * h * w * d, 42.0);
    let x = DynTensor::from_vec(data.clone(), &[b, h * w, d], &Device::Cpu).unwrap();

    let (windowed, ph, pw) = window_partition(&x, h, w, ws).unwrap();
    // 4x4 grid, ws=2 -> 4 windows of 4 tokens each
    assert_eq!(windowed.dims(), &[4, 4, d]);

    // Add 1.0 to all values in windowed form
    let ones = DynTensor::ones(&[4, 4, d], DType::F32, &Device::Cpu).unwrap();
    let windowed_plus = windowed.broadcast_add(&ones).unwrap();

    let recovered = window_unpartition(&windowed_plus, h, w, ph, pw, ws, b).unwrap();
    assert_eq!(recovered.dims(), &[b, h * w, d]);

    // Every value should be original + 1.0
    let orig = data;
    let rec = recovered.to_flat_vec::<f32>().unwrap();
    for (i, (o, r)) in orig.iter().zip(rec.iter()).enumerate() {
        assert!(
            (r - (o + 1.0)).abs() < 1e-6,
            "Mismatch at {i}: expected {}, got {r}",
            o + 1.0
        );
    }
}

// -- Padding roundtrip with non-trivial data ----------------------------------

/// Verify that padding and unpadding preserves data values exactly.
/// Uses a grid size that requires padding (5x3, ws=4).
#[test]
fn test_window_roundtrip_padded_values_preserved() {
    let (h, w, d, ws) = (5, 3, 4, 4);
    let b = 1;
    let n = b * h * w * d;
    let data = det_data(n, 77.0);
    let x = DynTensor::from_vec(data.clone(), &[b, h * w, d], &Device::Cpu).unwrap();

    let (windowed, ph, pw) = window_partition(&x, h, w, ws).unwrap();
    assert_eq!(ph, 8); // 5 -> pad to 8
    assert_eq!(pw, 4); // 3 -> pad to 4

    let recovered = window_unpartition(&windowed, h, w, ph, pw, ws, b).unwrap();
    assert_eq!(recovered.dims(), &[b, h * w, d]);

    let rec = recovered.to_flat_vec::<f32>().unwrap();
    assert_eq!(data.len(), rec.len());
    for (i, (orig, recov)) in data.iter().zip(rec.iter()).enumerate() {
        assert!(
            (orig - recov).abs() < 1e-7,
            "Value mismatch at {i}: orig={orig}, recovered={recov}"
        );
    }
}

// -- Large grid (Qwen2.5-VL scale) -------------------------------------------

/// Test with a 14x14 grid (standard ViT-B/16 with 224x224 images) and
/// window_size=7 (Qwen2.5-VL default). Exactly divisible.
#[test]
fn test_window_14x14_grid_ws7() {
    let (h, w, d, ws) = (14, 14, 32, 7);
    let b = 2;
    let n = b * h * w * d;
    let data = det_data(n, 42.0);
    let x = DynTensor::from_vec(data.clone(), &[b, h * w, d], &Device::Cpu).unwrap();

    let (windowed, ph, pw) = window_partition(&x, h, w, ws).unwrap();
    // 14/7=2 windows per dim, 2*2=4 windows per batch, 2 batches -> 8 window-batches
    assert_eq!(windowed.dims(), &[b * 4, ws * ws, d]);
    assert_eq!(ph, 14);
    assert_eq!(pw, 14);

    let recovered = window_unpartition(&windowed, h, w, ph, pw, ws, b).unwrap();
    assert_eq!(recovered.dims(), &[b, h * w, d]);

    let rec = recovered.to_flat_vec::<f32>().unwrap();
    for (i, (orig, recov)) in data.iter().zip(rec.iter()).enumerate() {
        assert!(
            (orig - recov).abs() < 1e-7,
            "14x14 roundtrip mismatch at {i}"
        );
    }
}

/// Test with a 32x32 grid (ViT with 512x512 images, patch=16) and
/// window_size=8. Exactly divisible.
#[test]
fn test_window_32x32_grid_ws8() {
    let (h, w, d, ws) = (32, 32, 16, 8);
    let b = 1;
    let n = b * h * w * d;
    let data = det_data(n, 99.0);
    let x = DynTensor::from_vec(data.clone(), &[b, h * w, d], &Device::Cpu).unwrap();

    let (windowed, ph, pw) = window_partition(&x, h, w, ws).unwrap();
    // 32/8=4 windows per dim, 4*4=16 windows
    assert_eq!(windowed.dims(), &[16, 64, d]);
    assert_eq!(ph, 32);
    assert_eq!(pw, 32);

    let recovered = window_unpartition(&windowed, h, w, ph, pw, ws, b).unwrap();
    let rec = recovered.to_flat_vec::<f32>().unwrap();
    for (i, (orig, recov)) in data.iter().zip(rec.iter()).enumerate() {
        assert!(
            (orig - recov).abs() < 1e-7,
            "32x32 roundtrip mismatch at {i}"
        );
    }
}

// -- Window content correctness -----------------------------------------------

/// Verify that window_partition places the correct spatial tokens into each
/// window. Uses a 4x4 grid with identifiable values to check spatial mapping.
#[test]
fn test_window_partition_spatial_mapping() {
    let (h, w, d, ws) = (4, 4, 1, 2);
    let b = 1;
    // Each token has a unique value = its spatial index
    let data: Vec<f32> = (0..h * w).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[b, h * w, d], &Device::Cpu).unwrap();

    let (windowed, _ph, _pw) = window_partition(&x, h, w, ws).unwrap();
    let flat = windowed.to_flat_vec::<f32>().unwrap();

    // 4x4 grid with ws=2 creates 4 windows of 4 tokens:
    // Window 0: rows 0-1, cols 0-1 -> positions (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 4, 5
    // Window 1: rows 0-1, cols 2-3 -> positions (0,2), (0,3), (1,2), (1,3) -> indices 2, 3, 6, 7
    // Window 2: rows 2-3, cols 0-1 -> positions (2,0), (2,1), (3,0), (3,1) -> indices 8, 9, 12, 13
    // Window 3: rows 2-3, cols 2-3 -> positions (2,2), (2,3), (3,2), (3,3) -> indices 10, 11, 14, 15
    let expected = vec![
        0.0, 1.0, 4.0, 5.0, // window 0
        2.0, 3.0, 6.0, 7.0, // window 1
        8.0, 9.0, 12.0, 13.0, // window 2
        10.0, 11.0, 14.0, 15.0, // window 3
    ];
    assert_eq!(
        flat.len(),
        expected.len(),
        "flat={flat:?}, expected={expected:?}"
    );
    for (i, (a, e)) in flat.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "Window content mismatch at {i}: got {a}, expected {e}"
        );
    }
}

// -- Padded window content correctness ----------------------------------------

/// Verify that padded regions contain zeros (not garbage data).
#[test]
fn test_window_partition_padded_regions_are_zeros() {
    let (h, w, d, ws) = (3, 3, 2, 4);
    let b = 1;
    // All tokens have value 1.0 so padding (0.0) is distinguishable
    let x = DynTensor::ones(&[b, h * w, d], DType::F32, &Device::Cpu).unwrap();

    let (windowed, ph, pw) = window_partition(&x, h, w, ws).unwrap();
    assert_eq!(ph, 4); // 3 -> 4
    assert_eq!(pw, 4); // 3 -> 4
                       // 1 window of 4x4=16 tokens, d=2
    assert_eq!(windowed.dims(), &[1, 16, 2]);

    let flat = windowed.to_flat_vec::<f32>().unwrap();
    // In the 4x4 padded grid, positions (row, col) where row >= 3 or col >= 3
    // should be zeros. The window reshapes [4, 4] -> flat [16], so:
    // Position (row, col) in the window = index (row * 4 + col).
    // Padded positions: (0,3), (1,3), (2,3), (3,0), (3,1), (3,2), (3,3)
    let padded_indices: Vec<usize> = vec![3, 7, 11, 12, 13, 14, 15];
    for &pidx in &padded_indices {
        for di in 0..d {
            let flat_idx = pidx * d + di;
            assert!(
                flat[flat_idx].abs() < 1e-8,
                "Padded position ({pidx}, d={di}) should be 0.0, got {}",
                flat[flat_idx]
            );
        }
    }

    // Non-padded positions should be 1.0
    let nonpadded_indices: Vec<usize> = vec![0, 1, 2, 4, 5, 6, 8, 9, 10];
    for &nidx in &nonpadded_indices {
        for di in 0..d {
            let flat_idx = nidx * d + di;
            assert!(
                (flat[flat_idx] - 1.0).abs() < 1e-8,
                "Non-padded position ({nidx}, d={di}) should be 1.0, got {}",
                flat[flat_idx]
            );
        }
    }
}
