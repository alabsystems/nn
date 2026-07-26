// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tensor operation compatibility tests for the candle→nn migration.
//!
//! Core model/VarBuilder/LSTM/KvCache migration tests are in `candle_migration.rs`.
//! These tests verify tensor operations (cat, stack, broadcast, maximum/minimum,
//! selection ops, narrow, transpose, error handling) match candle's API.
//!
//! Run: `cargo test -p nn --test candle_migration_ops`

use nn::{DType, Device, DynTensor, Result, Shape, TensorError};

/// Verify cat/stack accept owned tensor slices (candle convention).
///
/// candle: `Tensor::cat(&[a, b], 0)` where a, b are owned `Tensor` values
/// nn:    same — `DynTensor::cat(&[a, b], 0)` works with owned or borrowed
#[test]
fn test_cat_stack_owned_slice() -> Result<()> {
    let a = DynTensor::ones(&[2, 3], DType::F32, &Device::Cpu)?;
    let b = DynTensor::ones(&[2, 3], DType::F32, &Device::Cpu)?;

    // Owned slice (candle convention): &[Tensor]
    let c = DynTensor::cat(&[a.clone(), b.clone()], 0)?;
    assert_eq!(c.dims(), &[4, 3]);

    let s = DynTensor::stack(&[a, b], 0)?;
    assert_eq!(s.dims(), &[2, 2, 3]);
    Ok(())
}

/// Verify broadcast_as accepts both owned and borrowed Shape (#1265 AC3).
///
/// candle: `tensor.broadcast_as(other.shape())?`   (owned Shape)
///         `tensor.broadcast_as(&shape)?`           (borrowed Shape)
/// nn:    same — broadcast_as accepts impl AsRef<[usize]>
#[test]
fn test_broadcast_as_owned_and_borrowed_shape() -> Result<()> {
    let eps = DynTensor::new(&[1e-12], &[1, 1], &Device::Cpu)?;
    let x = DynTensor::zeros(&[4, 8], DType::F32, &Device::Cpu)?;

    // Owned Shape (candle pattern: eps.broadcast_as(x.shape()))
    let result = eps.broadcast_as(x.shape())?;
    assert_eq!(result.dims(), &[4, 8]);

    // Borrowed Shape reference
    let shape = Shape::from_dims(&[4, 8]);
    let result2 = eps.broadcast_as(&shape)?;
    assert_eq!(result2.dims(), &[4, 8]);

    // Raw slice (bonus: works because of AsRef<[usize]>)
    let result3 = eps.broadcast_as([4, 8])?;
    assert_eq!(result3.dims(), &[4, 8]);

    Ok(())
}

/// Verify maximum() and minimum() element-wise ops (#1265 AC4).
///
/// candle: `a.maximum(&b)?`
/// nn:    same
#[test]
fn test_maximum_minimum() -> Result<()> {
    let a = DynTensor::new(&[1.0, 5.0, 3.0], &[3], &Device::Cpu)?;
    let b = DynTensor::new(&[4.0, 2.0, 3.0], &[3], &Device::Cpu)?;

    let max_result = a.maximum(&b)?;
    let max_vals = max_result.to_flat_vec::<f32>()?;
    assert_eq!(max_vals, vec![4.0, 5.0, 3.0]);

    let min_result = a.minimum(&b)?;
    let min_vals = min_result.to_flat_vec::<f32>()?;
    assert_eq!(min_vals, vec![1.0, 2.0, 3.0]);

    Ok(())
}

/// Verify selection ops work with non-f32 dtypes (dvoice migration pattern).
///
/// dvoice uses:
/// - U32 token IDs for embedding lookup + gather
/// - I64 indices for CTC decoding
/// - U8 attention masks with where_cond
/// - expand for broadcasting masks to match batch dimensions
///
/// These ops must work across all dtypes, not just f32 (#1264).
#[test]
fn test_selection_ops_multi_dtype() -> Result<()> {
    // index_select with U32 IDs (Whisper token lookup pattern)
    let vocab = DynTensor::ones(&[100, 64], DType::F32, &Device::Cpu)?;
    let ids = DynTensor::from_vec_u32(vec![0, 5, 99], &[3], &Device::Cpu)?;
    let selected = vocab.index_select(&ids, 0)?;
    assert_eq!(selected.dims(), &[3, 64]);

    // gather with U32 indices (attention score gathering)
    let scores = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu)?;
    let idx = DynTensor::from_vec_u32(vec![2, 0, 1, 2], &[2, 2], &Device::Cpu)?;
    let gathered = scores.gather(&idx, 1)?;
    assert_eq!(gathered.dims(), &[2, 2]);
    let vals = gathered.to_flat_vec::<f32>()?;
    assert_eq!(vals, vec![3.0, 1.0, 5.0, 6.0]);

    // where_cond with U8 mask (attention masking pattern)
    let mask = DynTensor::from_vec_u8(vec![1, 0, 1], &[3], &Device::Cpu)?;
    let on_true = DynTensor::new(&[10.0, 20.0, 30.0], &[3], &Device::Cpu)?;
    let on_false = DynTensor::new(&[0.0, 0.0, 0.0], &[3], &Device::Cpu)?;
    let result = mask.where_cond(&on_true, &on_false)?;
    let vals = result.to_flat_vec::<f32>()?;
    assert_eq!(vals, vec![10.0, 0.0, 30.0]);

    // expand for broadcasting (mask expansion to batch dims)
    let mask_1d = DynTensor::new(&[1.0, 0.0, 1.0], &[1, 3], &Device::Cpu)?;
    let expanded = mask_1d.expand([4, 3])?;
    assert_eq!(expanded.dims(), &[4, 3]);

    Ok(())
}

/// Verify narrow/slice operation (used in LSTM state splitting, GLU decomposition).
///
/// candle: `tensor.narrow(dim, start, len)?`
/// nn:    same API
#[test]
fn test_narrow_for_lstm_state_split() -> Result<()> {
    // LSTM hidden state split pattern: h_c [batch, 2*hidden] → h, c
    let hidden_size = 32;
    let h_c = DynTensor::ones(&[1, 2 * hidden_size], DType::F32, &Device::Cpu)?;

    let h = h_c.narrow(1, 0, hidden_size)?;
    let c = h_c.narrow(1, hidden_size, hidden_size)?;

    assert_eq!(h.dims(), &[1, hidden_size]);
    assert_eq!(c.dims(), &[1, hidden_size]);

    Ok(())
}

/// Verify transpose and contiguous (frequent in dvoice attention code).
///
/// candle: `tensor.transpose(dim0, dim1)?.contiguous()?`
/// nn:    same API
#[test]
fn test_transpose_contiguous() -> Result<()> {
    // Attention pattern: [batch, seq, heads, head_dim] → [batch, heads, seq, head_dim]
    let x = DynTensor::ones(&[1, 10, 4, 32], DType::F32, &Device::Cpu)?;
    let x = x.transpose(1, 2)?;
    assert_eq!(x.dims(), &[1, 4, 10, 32]);

    let x = x.contiguous()?;
    assert_eq!(x.dims(), &[1, 4, 10, 32]);

    Ok(())
}

/// Verify error types match candle's pattern.
///
/// candle: `Result<T>` with `candle_core::Error`
/// nn:    `Result<T>` with `nn::TensorError`
#[test]
fn test_error_handling_pattern() {
    // Shape mismatch produces TensorError, matching candle's pattern
    let a = DynTensor::ones(&[2, 3], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::ones(&[4, 5], DType::F32, &Device::Cpu).unwrap();

    let result = a.add(&b);
    assert!(result.is_err());

    // Error is TensorError (nn's equivalent of candle_core::Error)
    let err = result.unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "Expected ShapeMismatch, got: {err}"
    );
}
