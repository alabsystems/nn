// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for pooling, embedding, and upsampling layers in nn-core.
//! Covers output shape contracts, value correctness, config defaults,
//! and round-trip properties for PixelShuffle/PixelUnshuffle.

use crate::dyn_tensor::DynTensor;
use crate::layers::vision::{PixelShuffle, PixelUnshuffle, Upsample2d, UpsampleMode};
use crate::layers::{
    AdaptiveAvgPool2d, AvgPool2d, Embedding, MaxPool1d, MaxPool2d, Module, Pool1dConfig,
    Pool2dConfig,
};
use crate::{DType, Device};

// =============================================================================
// Pooling Layers
// =============================================================================

// -- MaxPool2d output shape ---------------------------------------------------

#[test]
fn test_max_pool2d_output_shape() {
    // [N, C, H, W] = [2, 3, 8, 8], kernel=2, stride=2 -> [2, 3, 4, 4]
    let x = DynTensor::full(&[2, 3, 8, 8], 1.0, DType::F32, &Device::Cpu).unwrap();
    let layer = MaxPool2d::new(Pool2dConfig::new(2)).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4, 4], "H/stride x W/stride shape");
}

#[test]
fn test_max_pool2d_output_shape_non_square() {
    // [1, 1, 6, 10], kernel=3, stride=3 -> [1, 1, 2, 3] (floor division)
    let x = DynTensor::full(&[1, 1, 6, 10], 0.5, DType::F32, &Device::Cpu).unwrap();
    let layer = MaxPool2d::new(Pool2dConfig::new(3)).unwrap();
    let y = layer.forward(&x).unwrap();
    // out_h = (6 - 3)/3 + 1 = 2, out_w = (10 - 3)/3 + 1 = 3 (floor truncation at 10/3)
    assert_eq!(y.dims(), &[1, 1, 2, 3]);
}

// -- AvgPool2d output shape ---------------------------------------------------

#[test]
fn test_avg_pool2d_output_shape() {
    // [2, 3, 8, 8], kernel=2, stride=2 -> [2, 3, 4, 4]
    let x = DynTensor::full(&[2, 3, 8, 8], 2.0, DType::F32, &Device::Cpu).unwrap();
    let layer = AvgPool2d::new(Pool2dConfig::new(2)).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4, 4], "H/stride x W/stride shape");
}

#[test]
fn test_avg_pool2d_value_correctness() {
    // All-constant input: avg pool of constant is the constant itself
    let x = DynTensor::full(&[1, 1, 4, 4], 3.5, DType::F32, &Device::Cpu).unwrap();
    let layer = AvgPool2d::new(Pool2dConfig::new(2)).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(
            (v - 3.5).abs() < 1e-6,
            "constant-input avg pool should yield constant output, got {v}"
        );
    }
}

// -- MaxPool1d basic ----------------------------------------------------------

#[test]
fn test_max_pool1d_basic() {
    // max pool picks maximum values from each kernel window
    let data = vec![1.0, 5.0, 3.0, 7.0, 2.0, 9.0];
    let x = DynTensor::from_vec(data, &[1, 1, 6], &Device::Cpu).unwrap();
    let layer = MaxPool1d::new(Pool1dConfig::new(3).with_stride(3)).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Window 1: max(1,5,3) = 5, Window 2: max(7,2,9) = 9
    assert_eq!(vals, vec![5.0, 9.0]);
}

#[test]
fn test_max_pool1d_stride_less_than_kernel() {
    // Overlapping windows: kernel=3, stride=1
    let data = vec![1.0, 3.0, 2.0, 5.0, 4.0];
    let x = DynTensor::from_vec(data, &[1, 1, 5], &Device::Cpu).unwrap();
    let layer = MaxPool1d::new(Pool1dConfig::new(3).with_stride(1)).unwrap();
    let y = layer.forward(&x).unwrap();
    // out = (5 - 3)/1 + 1 = 3
    assert_eq!(y.dims(), &[1, 1, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // max(1,3,2)=3, max(3,2,5)=5, max(2,5,4)=5
    assert_eq!(vals, vec![3.0, 5.0, 5.0]);
}

// -- AvgPool1d basic (using DynTensor::avg_pool2d on 1D-shaped data) ----------
// Note: there is no AvgPool1d nn layer; we test avg_pool2d value correctness.

#[test]
fn test_avg_pool2d_computes_mean() {
    // 2x2 kernel on known data to verify mean computation
    #[rustfmt::skip]
    let data = vec![
        2.0, 4.0,
        6.0, 8.0,
    ];
    let x = DynTensor::from_vec(data, &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let layer = AvgPool2d::new(Pool2dConfig::new(2)).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 1, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // mean(2,4,6,8) = 5.0
    assert!(
        (vals[0] - 5.0).abs() < 1e-6,
        "avg should be 5.0, got {}",
        vals[0]
    );
}

// -- AdaptiveAvgPool2d to 1x1 (global average pooling) ------------------------

#[test]
fn test_adaptive_avg_pool2d_to_1x1() {
    // Global average pooling: [1, 2, 4, 4] -> [1, 2, 1, 1]
    // Channel 0: all 1.0, Channel 1: all 2.0
    let mut data = vec![1.0f32; 16];
    data.extend(vec![2.0f32; 16]);
    let x = DynTensor::from_vec(data, &[1, 2, 4, 4], &Device::Cpu).unwrap();
    let layer = AdaptiveAvgPool2d::new(1, 1).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 1, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-6, "channel 0 avg should be 1.0");
    assert!((vals[1] - 2.0).abs() < 1e-6, "channel 1 avg should be 2.0");
}

#[test]
fn test_adaptive_avg_pool2d_preserves_batch_channel() {
    // [3, 5, 7, 7] -> [3, 5, 2, 3]
    let x = DynTensor::full(&[3, 5, 7, 7], 1.0, DType::F32, &Device::Cpu).unwrap();
    let layer = AdaptiveAvgPool2d::new(2, 3).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[3, 5, 2, 3]);
}

// -- Pool2dConfig defaults ----------------------------------------------------

#[test]
fn test_pool_config_defaults() {
    let cfg = Pool2dConfig::new(5);
    assert_eq!(cfg.kernel_size, 5);
    assert_eq!(cfg.stride, 5, "stride should default to kernel_size");
    assert_eq!(cfg.padding, 0, "padding should default to 0");
}

#[test]
fn test_pool1d_config_defaults() {
    let cfg = Pool1dConfig::new(4);
    assert_eq!(cfg.kernel_size, 4);
    assert_eq!(cfg.stride, 4, "stride should default to kernel_size");
    assert_eq!(cfg.padding, 0, "padding should default to 0");
}

#[test]
fn test_pool2d_config_builder_methods() {
    let cfg = Pool2dConfig::new(3).with_stride(2).with_padding(1);
    assert_eq!(cfg.kernel_size, 3);
    assert_eq!(cfg.stride, 2);
    assert_eq!(cfg.padding, 1);
}

#[test]
fn test_pool1d_config_builder_methods() {
    let cfg = Pool1dConfig::new(5).with_stride(2).with_padding(1);
    assert_eq!(cfg.kernel_size, 5);
    assert_eq!(cfg.stride, 2);
    assert_eq!(cfg.padding, 1);
}

// =============================================================================
// Embedding Layers
// =============================================================================

// -- Embedding output shape ---------------------------------------------------

#[test]
fn test_embedding_output_shape() {
    // [seq_len] -> [seq_len, embed_dim]
    let weight = DynTensor::full(&[10, 32], 0.1, DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 3, 7], &[3], &Device::Cpu).unwrap();
    let y = emb.forward(&ids).unwrap();
    assert_eq!(y.dims(), &[3, 32], "seq_len=3, embed_dim=32");
}

#[test]
fn test_embedding_output_shape_batched() {
    // [B, S] -> [B, S, D]
    let weight = DynTensor::full(&[100, 64], 0.1, DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 1, 2, 3, 4, 5], &[2, 3], &Device::Cpu).unwrap();
    let y = emb.forward(&ids).unwrap();
    assert_eq!(y.dims(), &[2, 3, 64]);
}

// -- Embedding lookup correctness ---------------------------------------------

#[test]
fn test_embedding_lookup_specific_rows() {
    // Verify that specific indices retrieve the correct rows
    let weight = DynTensor::new(
        &[
            1.0, 2.0, 3.0, // id 0
            4.0, 5.0, 6.0, // id 1
            7.0, 8.0, 9.0, // id 2
            10.0, 11.0, 12.0, // id 3
        ],
        &[4, 3],
        &Device::Cpu,
    )
    .unwrap();
    let emb = Embedding::new(weight).unwrap();
    let result = emb.forward_ids(&[3, 1]).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // id 3 -> [10, 11, 12]
    assert_eq!(&vals[0..3], &[10.0, 11.0, 12.0]);
    // id 1 -> [4, 5, 6]
    assert_eq!(&vals[3..6], &[4.0, 5.0, 6.0]);
}

#[test]
fn test_embedding_lookup_repeated_index() {
    let weight = DynTensor::new(&[100.0, 200.0, 300.0, 400.0], &[2, 2], &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let result = emb.forward_ids(&[0, 0, 0]).unwrap();
    assert_eq!(result.dims(), &[3, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // All three lookups should return row 0
    for chunk in vals.chunks(2) {
        assert_eq!(chunk, &[100.0, 200.0]);
    }
}

// -- Embedding vocab size validation ------------------------------------------

#[test]
fn test_embedding_vocab_size_out_of_range() {
    let weight = DynTensor::full(&[5, 8], 1.0, DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    // Index 5 is out of range for vocab_size=5 (valid: 0..4)
    let err = emb.forward_ids(&[5]);
    assert!(err.is_err(), "index == vocab_size should be out of range");
}

#[test]
fn test_embedding_vocab_size_boundary() {
    let weight = DynTensor::full(&[5, 8], 1.0, DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    // Index 4 is the last valid index for vocab_size=5
    let result = emb.forward_ids(&[4]);
    assert!(result.is_ok(), "index vocab_size-1 should be valid");
    assert_eq!(result.unwrap().dims(), &[1, 8]);
}

// -- Embedding zero init pattern ----------------------------------------------

#[test]
fn test_embedding_zero_init() {
    // Verify that a zeros-initialized embedding returns zeros for any index
    let weight = DynTensor::zeros(&[10, 16], DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let result = emb.forward_ids(&[0, 5, 9]).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(
            v.abs() < 1e-10,
            "zero-init embedding should return zeros, got {v}"
        );
    }
}

#[test]
fn test_embedding_ones_init() {
    // All-ones weight: every lookup should return all-ones vector
    let weight = DynTensor::ones(&[4, 3], DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let result = emb.forward_ids(&[0, 1, 2, 3]).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "ones-init embedding should return 1.0, got {v}"
        );
    }
}

// -- Embedding embeddings() alias ---------------------------------------------

#[test]
fn test_embedding_embeddings_alias() {
    let weight = DynTensor::full(&[5, 3], 0.5, DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    // embeddings() and weight() should return the same tensor
    assert_eq!(emb.embeddings().dims(), emb.weight().dims());
}

// =============================================================================
// Upsampling
// =============================================================================

// -- Upsample nearest 2x -----------------------------------------------------

#[test]
fn test_upsample_nearest_2x() {
    // [1, 1, 2, 2] -> [1, 1, 4, 4] doubles spatial dimensions
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let layer = Upsample2d::new(2.0, 2.0, UpsampleMode::Nearest).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4], "spatial dims should double");
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Each original pixel is replicated in a 2x2 block
    // Row 0: [1, 1, 2, 2]
    assert_eq!(&vals[0..4], &[1.0, 1.0, 2.0, 2.0]);
    // Row 1: [1, 1, 2, 2] (repeat)
    assert_eq!(&vals[4..8], &[1.0, 1.0, 2.0, 2.0]);
    // Row 2: [3, 3, 4, 4]
    assert_eq!(&vals[8..12], &[3.0, 3.0, 4.0, 4.0]);
    // Row 3: [3, 3, 4, 4] (repeat)
    assert_eq!(&vals[12..16], &[3.0, 3.0, 4.0, 4.0]);
}

#[test]
fn test_upsample_nearest_preserves_batch_channel() {
    // [2, 3, 4, 4] with 2x upsample -> [2, 3, 8, 8]
    let x = DynTensor::full(&[2, 3, 4, 4], 1.0, DType::F32, &Device::Cpu).unwrap();
    let layer = Upsample2d::new(2.0, 2.0, UpsampleMode::Nearest).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 8, 8]);
}

// -- PixelShuffle shape -------------------------------------------------------

#[test]
fn test_pixel_shuffle_shape() {
    // [N, C*r^2, H, W] -> [N, C, H*r, W*r]
    // [1, 12, 2, 2] with r=2 -> [1, 3, 4, 4] (12 = 3 * 2^2)
    let data: Vec<f32> = (1..=48).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 12, 2, 2], &Device::Cpu).unwrap();
    let layer = PixelShuffle::new(2).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 3, 4, 4], "C*r^2=12 -> C=3, H*r=4, W*r=4");
}

#[test]
fn test_pixel_shuffle_shape_r3() {
    // [1, 9, 1, 1] with r=3 -> [1, 1, 3, 3]
    let data: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 9, 1, 1], &Device::Cpu).unwrap();
    let layer = PixelShuffle::new(3).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
}

// -- PixelUnshuffle shape (inverse of PixelShuffle) ---------------------------

#[test]
fn test_pixel_unshuffle_shape() {
    // [N, C, H*r, W*r] -> [N, C*r^2, H, W]
    // [1, 3, 4, 4] with r=2 -> [1, 12, 2, 2]
    let data: Vec<f32> = (1..=48).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 3, 4, 4], &Device::Cpu).unwrap();
    let layer = PixelUnshuffle::new(2).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(
        y.dims(),
        &[1, 12, 2, 2],
        "C=3 -> C*r^2=12, H*r=4 -> H=2, W*r=4 -> W=2"
    );
}

// -- PixelShuffle/PixelUnshuffle round-trip via nn layer -----------------------

#[test]
fn test_pixel_shuffle_unshuffle_roundtrip_via_layer() {
    let data: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 8, 2, 2], &Device::Cpu).unwrap();
    let shuffle = PixelShuffle::new(2).unwrap();
    let unshuffle = PixelUnshuffle::new(2).unwrap();
    let shuffled = shuffle.forward(&x).unwrap();
    assert_eq!(shuffled.dims(), &[1, 2, 4, 4]);
    let restored = unshuffle.forward(&shuffled).unwrap();
    assert_eq!(restored.dims(), &[1, 8, 2, 2]);
    assert_eq!(restored.to_flat_vec::<f32>().unwrap(), data);
}

// -- Additional pooling stride/padding combos ---------------------------------

#[test]
fn test_max_pool2d_with_padding() {
    // [1, 1, 3, 3], kernel=3, stride=1, padding=1 -> [1, 1, 3, 3]
    #[rustfmt::skip]
    let data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let x = DynTensor::from_vec(data, &[1, 1, 3, 3], &Device::Cpu).unwrap();
    let cfg = Pool2dConfig::new(3).with_stride(1).with_padding(1);
    let layer = MaxPool2d::new(cfg).unwrap();
    let y = layer.forward(&x).unwrap();
    // out = (3 + 2*1 - 3)/1 + 1 = 3
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Center element: max of all 9 values = 9.0
    assert!(
        (vals[4] - 9.0).abs() < 1e-6,
        "center should be 9.0, got {}",
        vals[4]
    );
}

#[test]
fn test_avg_pool2d_with_stride_1() {
    // Overlapping avg pool: [1, 1, 4, 4], kernel=2, stride=1 -> [1, 1, 3, 3]
    let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cfg = Pool2dConfig::new(2).with_stride(1);
    let layer = AvgPool2d::new(cfg).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Top-left 2x2: mean(1,2,5,6) = 3.5
    assert!(
        (vals[0] - 3.5).abs() < 1e-6,
        "expected 3.5, got {}",
        vals[0]
    );
}
