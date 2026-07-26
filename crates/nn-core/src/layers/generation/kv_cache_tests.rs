#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for KV cache (dynamic doubling-buffer implementation).
//!
//! Autoregressive generation and integration tests: `kv_cache_tests_gen.rs`
//! Performance and regression tests: `kv_cache_tests_perf.rs`

use crate::dyn_tensor::DynTensor;
use crate::layers::kv_cache::{KvCache, KvCacheLayer};
use crate::{DType, Device};

#[path = "kv_cache_tests_gen.rs"]
mod autoregressive;

#[path = "kv_cache_tests_buffer.rs"]
mod buffer;

// ---------------------------------------------------------------------------
// KvCacheLayer basic tests
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_layer_empty() {
    let layer = KvCacheLayer::empty();
    assert!(layer.is_empty());
    assert_eq!(layer.seq_len(), 0);
    assert!(layer.key().unwrap().is_none());
    assert!(layer.value().unwrap().is_none());
}

#[test]
fn test_kv_cache_layer_new_compat() {
    // candle-nn compatibility: KvCache::new(2, 4096)
    let layer = KvCacheLayer::new(2, 4096).unwrap();
    assert!(layer.is_empty());
    assert_eq!(layer.seq_len(), 0);
    assert!(layer.key().unwrap().is_none());
    assert!(layer.value().unwrap().is_none());
}

#[test]
fn test_kv_cache_layer_new_rejects_wrong_dim() {
    // dim must be 2 (sequence dimension)
    assert!(KvCacheLayer::new(0, 4096).is_err());
    assert!(KvCacheLayer::new(1, 4096).is_err());
    assert!(KvCacheLayer::new(3, 4096).is_err());
}

#[test]
fn test_kv_cache_layer_new_append_reset_cycle() {
    // Verifies the candle-compatible constructor works through full lifecycle
    let mut layer = KvCacheLayer::new(2, 4096).unwrap();

    let k = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::full(&[1, 2, 3, 4], 2.0, DType::F32, &Device::Cpu).unwrap();
    let (full_k, full_v) = layer.append(&k, &v).unwrap();
    assert_eq!(full_k.dims(), &[1, 2, 3, 4]);
    assert_eq!(full_v.dims(), &[1, 2, 3, 4]);
    assert_eq!(layer.seq_len(), 3);

    layer.reset();
    assert!(layer.is_empty());
    assert_eq!(layer.seq_len(), 0);
}

#[test]
fn test_kv_cache_layer_append_once() {
    let mut layer = KvCacheLayer::empty();
    // [batch=1, heads=2, seq=3, dim=4]
    let k = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::full(&[1, 2, 3, 4], 2.0, DType::F32, &Device::Cpu).unwrap();

    let (full_k, full_v) = layer.append(&k, &v).unwrap();
    assert_eq!(full_k.dims(), &[1, 2, 3, 4]);
    assert_eq!(full_v.dims(), &[1, 2, 3, 4]);
    assert_eq!(layer.seq_len(), 3);
    assert!(!layer.is_empty());
}

#[test]
fn test_kv_cache_layer_append_grows_sequence() {
    let mut layer = KvCacheLayer::empty();
    // First append: seq=3
    let k1 = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let v1 = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    layer.append(&k1, &v1).unwrap();
    assert_eq!(layer.seq_len(), 3);

    // Second append: seq=2
    let k2 = DynTensor::full(&[1, 2, 2, 4], 3.0, DType::F32, &Device::Cpu).unwrap();
    let v2 = DynTensor::full(&[1, 2, 2, 4], 4.0, DType::F32, &Device::Cpu).unwrap();
    let (full_k, full_v) = layer.append(&k2, &v2).unwrap();

    assert_eq!(full_k.dims(), &[1, 2, 5, 4]);
    assert_eq!(full_v.dims(), &[1, 2, 5, 4]);
    assert_eq!(layer.seq_len(), 5);

    // Verify data: first 3 positions should be 1.0, next 2 should be 3.0 (key)
    let k_data = full_k.to_flat_vec::<f32>().unwrap();
    // Batch 0, head 0, first element of position 0
    assert!((k_data[0] - 1.0).abs() < 1e-6);
    // Batch 0, head 0, first element of position 3 (after cat)
    assert!((k_data[3 * 4] - 3.0).abs() < 1e-6);
}

#[test]
fn test_kv_cache_layer_reset() {
    let mut layer = KvCacheLayer::empty();
    let k = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    layer.append(&k, &v).unwrap();
    assert_eq!(layer.seq_len(), 3);

    layer.reset();
    assert!(layer.is_empty());
    assert_eq!(layer.seq_len(), 0);
}

#[test]
fn test_kv_cache_layer_rejects_mismatched_kv_shapes() {
    let mut layer = KvCacheLayer::empty();
    let k = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 2, 3, 8], DType::F32, &Device::Cpu).unwrap(); // wrong head_dim
    let result = layer.append(&k, &v);
    assert!(result.is_err());
}

#[test]
fn test_kv_cache_layer_rejects_rank_too_low() {
    let mut layer = KvCacheLayer::empty();
    let k = DynTensor::ones(&[4, 4], DType::F32, &Device::Cpu).unwrap(); // rank 2
    let v = DynTensor::ones(&[4, 4], DType::F32, &Device::Cpu).unwrap();
    let result = layer.append(&k, &v);
    assert!(result.is_err());
}

#[test]
fn test_kv_cache_layer_rejects_dim_mismatch_on_second_append() {
    let mut layer = KvCacheLayer::empty();
    let k1 = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let v1 = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    layer.append(&k1, &v1).unwrap();

    // Different num_heads
    let k2 = DynTensor::ones(&[1, 4, 1, 4], DType::F32, &Device::Cpu).unwrap();
    let v2 = DynTensor::ones(&[1, 4, 1, 4], DType::F32, &Device::Cpu).unwrap();
    let result = layer.append(&k2, &v2);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// candle-nn compatibility accessors (#1218)
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_layer_dim_returns_seq_dim() {
    let layer = KvCacheLayer::empty();
    assert_eq!(layer.dim(), 2, "dim() should always return SEQ_DIM=2");
}

#[test]
fn test_kv_cache_layer_k_v_aliases() {
    let mut layer = KvCacheLayer::empty();
    // Empty cache: k()/v() return None
    assert!(layer.k().unwrap().is_none());
    assert!(layer.v().unwrap().is_none());

    // After append: k()/v() return the same as key()/value()
    let k = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::full(&[1, 2, 3, 4], 2.0, DType::F32, &Device::Cpu).unwrap();
    layer.append(&k, &v).unwrap();

    let k_result = layer.k().unwrap().unwrap();
    let v_result = layer.v().unwrap().unwrap();
    let key_result = layer.key().unwrap().unwrap();
    let value_result = layer.value().unwrap().unwrap();
    assert_eq!(k_result.dims(), key_result.dims());
    assert_eq!(v_result.dims(), value_result.dims());
}

// ---------------------------------------------------------------------------
// KvCache (multi-layer) tests
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_new() {
    let cache = KvCache::new(4);
    assert_eq!(cache.num_layers(), 4);
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_kv_cache_layer_access() {
    let mut cache = KvCache::new(3);
    assert!(cache.layer(0).is_ok());
    assert!(cache.layer(2).is_ok());
    assert!(cache.layer(3).is_err()); // out of range

    let k = DynTensor::ones(&[1, 2, 5, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 2, 5, 4], DType::F32, &Device::Cpu).unwrap();
    cache.layer_mut(1).unwrap().append(&k, &v).unwrap();

    assert_eq!(cache.seq_len(), 5); // first non-empty layer
    assert!(!cache.is_empty());
}

#[test]
fn test_kv_cache_reset() {
    let mut cache = KvCache::new(2);
    let k = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    cache.layer_mut(0).unwrap().append(&k, &v).unwrap();
    cache.layer_mut(1).unwrap().append(&k, &v).unwrap();

    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
}

// ---------------------------------------------------------------------------
// KvCacheBackend trait tests (#1223)
// ---------------------------------------------------------------------------

use crate::layers::kv_cache::{KvCacheBackend, KvCacheLayerBackend};

/// Helper: exercise the KvCacheLayerBackend trait on any implementor.
fn exercise_layer_backend(layer: &mut dyn KvCacheLayerBackend) {
    assert!(layer.is_empty());
    assert_eq!(layer.seq_len(), 0);

    let k = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::full(&[1, 2, 3, 4], 2.0, DType::F32, &Device::Cpu).unwrap();
    let (full_k, full_v) = layer.append(&k, &v).unwrap();
    assert_eq!(full_k.dims(), &[1, 2, 3, 4]);
    assert_eq!(full_v.dims(), &[1, 2, 3, 4]);
    assert_eq!(layer.seq_len(), 3);
    assert!(!layer.is_empty());

    // Append one more token.
    let k2 = DynTensor::full(&[1, 2, 1, 4], 3.0, DType::F32, &Device::Cpu).unwrap();
    let v2 = DynTensor::full(&[1, 2, 1, 4], 4.0, DType::F32, &Device::Cpu).unwrap();
    let (full_k2, full_v2) = layer.append(&k2, &v2).unwrap();
    assert_eq!(full_k2.dims(), &[1, 2, 4, 4]);
    assert_eq!(full_v2.dims(), &[1, 2, 4, 4]);
    assert_eq!(layer.seq_len(), 4);

    layer.reset();
    assert!(layer.is_empty());
    assert_eq!(layer.seq_len(), 0);
}

#[test]
fn test_layer_backend_trait_kv_cache_layer() {
    let mut layer = KvCacheLayer::empty();
    exercise_layer_backend(&mut layer);
}

/// Helper: exercise the KvCacheBackend trait on any implementor.
fn exercise_cache_backend(cache: &mut dyn KvCacheBackend) {
    assert_eq!(cache.num_layers(), 2);
    assert_eq!(cache.seq_len(), 0);

    // Append via trait.
    let k = DynTensor::ones(&[1, 2, 5, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 2, 5, 4], DType::F32, &Device::Cpu).unwrap();
    cache.layer_backend_mut(0).unwrap().append(&k, &v).unwrap();
    assert_eq!(cache.seq_len(), 5);

    // Out-of-range layer.
    assert!(cache.layer_backend_mut(99).is_err());

    cache.reset();
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_cache_backend_trait_kv_cache() {
    let mut cache = KvCache::new(2);
    exercise_cache_backend(&mut cache);
}

// generate(), performance, and regression tests extracted to kv_cache_tests_perf.rs
#[path = "kv_cache_tests_perf.rs"]
mod perf;
