// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for window attention utilities.

use super::{window_partition, window_unpartition};
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

#[test]
fn test_window_partition_exact_divisible() {
    // 4x4 grid, window_size=2 -> 4 windows of 2x2=4 tokens
    let x = DynTensor::ones(&[1, 16, 8], DType::F32, &Device::Cpu).unwrap();
    let (w, ph, pw) = window_partition(&x, 4, 4, 2).unwrap();
    assert_eq!(w.dims(), &[4, 4, 8]); // 1*4 windows, 4 tokens each, dim 8
    assert_eq!(ph, 4);
    assert_eq!(pw, 4);
}

#[test]
fn test_window_partition_needs_padding() {
    // 3x3 grid, window_size=2 -> pads to 4x4 -> 4 windows
    let x = DynTensor::ones(&[1, 9, 8], DType::F32, &Device::Cpu).unwrap();
    let (w, ph, pw) = window_partition(&x, 3, 3, 2).unwrap();
    assert_eq!(ph, 4);
    assert_eq!(pw, 4);
    assert_eq!(w.dims(), &[4, 4, 8]);
}

#[test]
fn test_window_roundtrip_exact() {
    // Partition then unpartition should be identity when exactly divisible.
    let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let x = DynTensor::new(&data, &[1, 4, 8], &Device::Cpu).unwrap();
    let (w, ph, pw) = window_partition(&x, 2, 2, 2).unwrap();
    let recovered = window_unpartition(&w, 2, 2, ph, pw, 2, 1).unwrap();
    assert_eq!(recovered.dims(), &[1, 4, 8]);
    let out = recovered.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, out);
}

#[test]
fn test_window_roundtrip_with_padding() {
    // 3x5 grid, window_size=4 -> pads to 4x8 -> 2 windows
    let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
    let x = DynTensor::new(&data, &[1, 15, 4], &Device::Cpu).unwrap();
    let (w, ph, pw) = window_partition(&x, 3, 5, 4).unwrap();
    assert_eq!(ph, 4);
    assert_eq!(pw, 8);
    let recovered = window_unpartition(&w, 3, 5, ph, pw, 4, 1).unwrap();
    assert_eq!(recovered.dims(), &[1, 15, 4]);
    let out = recovered.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, out);
}

#[test]
fn test_window_partition_batch() {
    // batch=2, 4x4 grid, window_size=2
    let x = DynTensor::ones(&[2, 16, 8], DType::F32, &Device::Cpu).unwrap();
    let (w, ph, pw) = window_partition(&x, 4, 4, 2).unwrap();
    // 2 batches * 4 windows = 8
    assert_eq!(w.dims(), &[8, 4, 8]);
    assert_eq!(ph, 4);
    assert_eq!(pw, 4);
}

#[test]
fn test_window_zero_size_error() {
    let x = DynTensor::ones(&[1, 4, 8], DType::F32, &Device::Cpu).unwrap();
    let err = window_partition(&x, 2, 2, 0).unwrap_err();
    assert!(format!("{err:?}").contains("window_size must be > 0"));
}

#[test]
fn test_window_seq_len_mismatch() {
    let x = DynTensor::ones(&[1, 5, 8], DType::F32, &Device::Cpu).unwrap();
    let err = window_partition(&x, 2, 3, 2).unwrap_err();
    assert!(format!("{err:?}").contains("ShapeMismatch"));
}

#[test]
fn test_window_single_window() {
    // window_size equals grid size -> 1 window
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let x = DynTensor::new(&data, &[1, 4, 3], &Device::Cpu).unwrap();
    let (w, ph, pw) = window_partition(&x, 2, 2, 2).unwrap();
    assert_eq!(w.dims(), &[1, 4, 3]);
    let recovered = window_unpartition(&w, 2, 2, ph, pw, 2, 1).unwrap();
    assert_eq!(recovered.to_flat_vec::<f32>().unwrap(), data);
}
