#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Embedding layer and its Module::forward input validation.

use super::*;
use crate::{DType, Device, DynTensor, TensorError};

// -- Embedding ---------------------------------------------------------------

#[test]
fn test_embedding_lookup() {
    let weight = DynTensor::new(
        &[
            10.0, 11.0, // id 0
            20.0, 21.0, // id 1
            30.0, 31.0, // id 2
        ],
        &[3, 2],
        &Device::Cpu,
    )
    .unwrap();
    let emb = Embedding::new(weight).unwrap();
    let result = emb.forward_ids(&[2, 0, 1]).unwrap();
    assert_eq!(result.dims(), &[3, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 30.0).abs() < 1e-6); // id 2
    assert!((vals[1] - 31.0).abs() < 1e-6);
    assert!((vals[2] - 10.0).abs() < 1e-6); // id 0
    assert!((vals[3] - 11.0).abs() < 1e-6);
    assert!((vals[4] - 20.0).abs() < 1e-6); // id 1
    assert!((vals[5] - 21.0).abs() < 1e-6);
}

#[test]
fn test_embedding_out_of_range() {
    let weight = DynTensor::ones(&[3, 2], DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let err = emb.forward_ids(&[5]).unwrap_err();
    assert!(
        matches!(err, TensorError::EmbeddingIndexOutOfRange { .. }),
        "expected EmbeddingIndexOutOfRange, got: {err:?}"
    );
}

#[test]
fn test_embedding_via_module() {
    let weight = DynTensor::new(&[100.0, 200.0, 300.0, 400.0], &[2, 2], &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::new(&[1.0, 0.0], &[2], &Device::Cpu).unwrap();
    let result = ids.apply(&emb).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 300.0).abs() < 1e-6); // id 1
    assert!((vals[1] - 400.0).abs() < 1e-6);
    assert!((vals[2] - 100.0).abs() < 1e-6); // id 0
    assert!((vals[3] - 200.0).abs() < 1e-6);
}

#[test]
fn test_embedding_accessors() {
    let weight = DynTensor::ones(&[5, 3], DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    assert_eq!(emb.weight().dims(), &[5, 3]);
}

// -- Embedding Module::forward input validation (P10 strategic audit) ---------

#[test]
fn test_embedding_forward_rejects_nan_index() {
    let weight = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let x = DynTensor::new(&[f32::NAN], &[1], &Device::Cpu).unwrap();
    let err = emb.forward(&x).unwrap_err();
    assert!(
        err.to_string().contains("non-negative finite integer"),
        "NaN index should be rejected, got: {err}"
    );
}

#[test]
fn test_embedding_forward_rejects_negative_index() {
    let weight = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let x = DynTensor::new(&[-1.0], &[1], &Device::Cpu).unwrap();
    let err = emb.forward(&x).unwrap_err();
    assert!(
        err.to_string().contains("non-negative finite integer"),
        "negative index should be rejected, got: {err}"
    );
}

#[test]
fn test_embedding_forward_rejects_infinity_index() {
    let weight = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let x = DynTensor::new(&[f32::INFINITY], &[1], &Device::Cpu).unwrap();
    let err = emb.forward(&x).unwrap_err();
    assert!(
        err.to_string().contains("non-negative finite integer"),
        "infinity index should be rejected, got: {err}"
    );
}

#[test]
fn test_embedding_forward_rejects_fractional_index() {
    let weight = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let x = DynTensor::new(&[0.5], &[1], &Device::Cpu).unwrap();
    let err = emb.forward(&x).unwrap_err();
    assert!(
        err.to_string().contains("non-negative finite integer"),
        "fractional index should be rejected, got: {err}"
    );
}

#[test]
fn test_embedding_forward_accepts_valid_f32_indices() {
    let weight = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let x = DynTensor::new(&[0.0, 1.0], &[2], &Device::Cpu).unwrap();
    let y = emb.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
}

// -- Embedding::forward with U32 indices (AC2 — argmax returns U32) -----------

#[test]
fn test_embedding_forward_u32_indices() {
    let weight = DynTensor::new(&[100.0, 200.0, 300.0, 400.0], &[2, 2], &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    // U32 index tensor — matches argmax/topk return type
    let ids = DynTensor::from_vec_u32(vec![1, 0], &[2], &Device::Cpu).unwrap();
    assert_eq!(ids.dtype(), DType::U32);
    let result = emb.forward(&ids).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 300.0).abs() < 1e-6); // id 1
    assert!((vals[1] - 400.0).abs() < 1e-6);
    assert!((vals[2] - 100.0).abs() < 1e-6); // id 0
    assert!((vals[3] - 200.0).abs() < 1e-6);
}

#[test]
fn test_embedding_forward_u32_out_of_range() {
    let weight = DynTensor::ones(&[3, 2], DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_u32(vec![5], &[1], &Device::Cpu).unwrap();
    let err = emb.forward(&ids).unwrap_err();
    // U32 path goes through index_select which returns InvalidShape
    assert!(
        matches!(err, TensorError::InvalidShape(ref msg) if msg.contains("out of bounds"))
            || matches!(err, TensorError::EmbeddingIndexOutOfRange { .. }),
        "U32 out-of-range index should be rejected, got: {err:?}"
    );
}

// -- Embedding::new() constructor validation ----------------------------------

#[test]
fn test_embedding_new_rejects_1d_weight() {
    let w = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let err = Embedding::new(w).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { expected: 2, .. }),
        "expected RankMismatch for 1D weight, got: {err:?}"
    );
}

#[test]
fn test_embedding_new_rejects_3d_weight() {
    let w = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1], &Device::Cpu).unwrap();
    let err = Embedding::new(w).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { expected: 2, .. }),
        "expected RankMismatch for 3D weight, got: {err:?}"
    );
}

// -- Embedding::forward with I64 indices (candle default for token IDs) --------

#[test]
fn test_embedding_forward_i64_indices() {
    let weight = DynTensor::new(&[100.0, 200.0, 300.0, 400.0], &[2, 2], &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_i64(vec![1, 0], &[2], &Device::Cpu).unwrap();
    assert_eq!(ids.dtype(), DType::I64);
    let result = emb.forward(&ids).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 300.0).abs() < 1e-6); // id 1
    assert!((vals[1] - 400.0).abs() < 1e-6);
    assert!((vals[2] - 100.0).abs() < 1e-6); // id 0
    assert!((vals[3] - 200.0).abs() < 1e-6);
}

#[test]
fn test_embedding_forward_i64_out_of_range() {
    let weight = DynTensor::ones(&[3, 2], DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_i64(vec![5], &[1], &Device::Cpu).unwrap();
    let err = emb.forward(&ids).unwrap_err();
    // I64 path goes through index_select which returns InvalidShape
    assert!(
        matches!(err, TensorError::InvalidShape(ref msg) if msg.contains("out of bounds"))
            || matches!(err, TensorError::EmbeddingIndexOutOfRange { .. }),
        "I64 out-of-range index should be rejected, got: {err:?}"
    );
}

#[test]
fn test_embedding_forward_i64_negative_rejects() {
    let weight = DynTensor::ones(&[3, 2], DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_i64(vec![-1], &[1], &Device::Cpu).unwrap();
    let err = emb.forward(&ids).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-negative")
            || msg.contains("out of u32 range")
            || msg.contains("out of bounds"),
        "negative I64 index should be rejected, got: {msg}"
    );
}

// -- Embedding::forward multi-dimensional input (shape preservation) -----------

#[test]
fn test_embedding_forward_2d_u32_shape() {
    // [B, S] = [2, 3] input, embed_dim = 4 → output should be [2, 3, 4]
    let weight = DynTensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, // id 0
            5.0, 6.0, 7.0, 8.0, // id 1
            9.0, 10.0, 11.0, 12.0, // id 2
        ],
        &[3, 4],
        &Device::Cpu,
    )
    .unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 1, 2, 2, 1, 0], &[2, 3], &Device::Cpu).unwrap();
    let result = emb.forward(&ids).unwrap();
    assert_eq!(
        result.dims(),
        &[2, 3, 4],
        "output shape should be [B, S, D]"
    );
    // Verify first batch, first token (id 0)
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(&vals[0..4], &[1.0, 2.0, 3.0, 4.0]);
    // Verify second batch, last token (id 0)
    assert_eq!(&vals[20..24], &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_embedding_forward_2d_i64_shape() {
    // [B, S] = [1, 2] input, embed_dim = 3 → output should be [1, 2, 3]
    let weight =
        DynTensor::new(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_i64(vec![1, 0], &[1, 2], &Device::Cpu).unwrap();
    let result = emb.forward(&ids).unwrap();
    assert_eq!(
        result.dims(),
        &[1, 2, 3],
        "output shape should be [1, S, D]"
    );
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(&vals[0..3], &[40.0, 50.0, 60.0]); // id 1
    assert_eq!(&vals[3..6], &[10.0, 20.0, 30.0]); // id 0
}

#[test]
fn test_embedding_forward_2d_f32_shape() {
    // Legacy F32 path with 2D input
    let weight = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::new(&[0.0, 1.0, 1.0, 0.0], &[2, 2], &Device::Cpu).unwrap();
    let result = emb.forward(&ids).unwrap();
    assert_eq!(
        result.dims(),
        &[2, 2, 2],
        "output shape should be [B, S, D]"
    );
}

#[test]
fn test_embedding_forward_3d_shape() {
    // [B, H, S] = [2, 1, 2] input, embed_dim = 3 → output should be [2, 1, 2, 3]
    let weight = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        &Device::Cpu,
    )
    .unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 1, 2, 0], &[2, 1, 2], &Device::Cpu).unwrap();
    let result = emb.forward(&ids).unwrap();
    assert_eq!(
        result.dims(),
        &[2, 1, 2, 3],
        "output shape should preserve all input dims + embed_dim"
    );
}

#[test]
fn test_embedding_forward_1d_shape_unchanged() {
    // 1D input [N] → output [N, D] (existing behavior must not regress)
    let weight = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 1], &[2], &Device::Cpu).unwrap();
    let result = emb.forward(&ids).unwrap();
    assert_eq!(
        result.dims(),
        &[2, 2],
        "1D input shape must still produce [N, D]"
    );
}
