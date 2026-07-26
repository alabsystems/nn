// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for pre-allocated KV cache (GPU-resident compiled decoder inference).

use crate::dyn_tensor::DynTensor;
use crate::layers::generation::prealloc_kv_cache::PreallocKvCacheLayer;
use crate::layers::generation::prealloc_kv_cache_multi::PreallocKvCache;
use crate::layers::generation::{KvCacheBackend, KvCacheLayerBackend};
use crate::{DType, Device};

// ---------------------------------------------------------------------------
// PreallocKvCacheLayer basic tests
// ---------------------------------------------------------------------------

#[test]
fn test_prealloc_layer_new() {
    let layer = PreallocKvCacheLayer::new(128).unwrap();
    assert!(layer.is_empty());
    assert_eq!(layer.seq_len(), 0);
    assert_eq!(layer.max_seq_len(), 128);
    assert_eq!(layer.remaining_capacity(), 128);
    assert!(!layer.is_allocated());
    assert!(layer.key().unwrap().is_none());
    assert!(layer.value().unwrap().is_none());
}

#[test]
fn test_prealloc_layer_rejects_zero_max_seq() {
    assert!(PreallocKvCacheLayer::new(0).is_err());
}

#[test]
fn test_prealloc_layer_append_once() {
    let mut layer = PreallocKvCacheLayer::new(128).unwrap();
    // [batch=1, heads=4, seq=3, head_dim=64]
    let k = DynTensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::full(&[1, 4, 3, 64], 2.0, DType::F32, &Device::Cpu).unwrap();

    let (full_k, full_v) = layer.append(&k, &v).unwrap();
    assert_eq!(full_k.dims(), &[1, 4, 3, 64]);
    assert_eq!(full_v.dims(), &[1, 4, 3, 64]);
    assert_eq!(layer.seq_len(), 3);
    assert_eq!(layer.remaining_capacity(), 125);
    assert!(layer.is_allocated());
    assert!(!layer.is_empty());
}

#[test]
fn test_prealloc_layer_append_grows_sequence() {
    let mut layer = PreallocKvCacheLayer::new(128).unwrap();
    // First append: seq=3
    let k1 = DynTensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
    let v1 = DynTensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
    layer.append(&k1, &v1).unwrap();
    assert_eq!(layer.seq_len(), 3);

    // Second append: seq=2 (single-token-like decode step)
    let k2 = DynTensor::full(&[1, 4, 2, 64], 3.0, DType::F32, &Device::Cpu).unwrap();
    let v2 = DynTensor::full(&[1, 4, 2, 64], 4.0, DType::F32, &Device::Cpu).unwrap();
    let (full_k, full_v) = layer.append(&k2, &v2).unwrap();

    assert_eq!(full_k.dims(), &[1, 4, 5, 64]);
    assert_eq!(full_v.dims(), &[1, 4, 5, 64]);
    assert_eq!(layer.seq_len(), 5);

    // Verify data: first 3 positions should be 1.0, next 2 should be 3.0 (key)
    let k_data = full_k.to_flat_vec::<f32>().unwrap();
    // Batch 0, head 0, position 0, first element
    assert!((k_data[0] - 1.0).abs() < 1e-6);
    // Batch 0, head 0, position 3 (after first append), first element
    assert!((k_data[3 * 64] - 3.0).abs() < 1e-6);
}

#[test]
fn test_prealloc_layer_single_token_decode_loop() {
    // Simulate autoregressive decoding: 4 heads, 64 head_dim, max_seq=128
    let mut layer = PreallocKvCacheLayer::new(128).unwrap();
    let batch = 1;
    let num_heads = 4;
    let head_dim = 64;

    for step in 0..50 {
        let val = (step + 1) as f64;
        let k = DynTensor::full(
            &[batch, num_heads, 1, head_dim],
            val,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let v = DynTensor::full(
            &[batch, num_heads, 1, head_dim],
            val * 10.0,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let (full_k, full_v) = layer.append(&k, &v).unwrap();
        assert_eq!(full_k.dim(2).unwrap(), step + 1);
        assert_eq!(full_v.dim(2).unwrap(), step + 1);

        // Verify latest position has correct value.
        let k_data = full_k.to_flat_vec::<f32>().unwrap();
        let v_data = full_v.to_flat_vec::<f32>().unwrap();
        // Head 0, position step, first element:
        let k_offset = step * head_dim;
        assert!(
            (k_data[k_offset] - val as f32).abs() < 1e-5,
            "step {step}: key expected {val}, got {}",
            k_data[k_offset]
        );
        let v_offset = step * head_dim;
        assert!(
            (v_data[v_offset] - (val * 10.0) as f32).abs() < 1e-4,
            "step {step}: value expected {}, got {}",
            val * 10.0,
            v_data[v_offset]
        );
    }

    assert_eq!(layer.seq_len(), 50);
    assert_eq!(layer.remaining_capacity(), 78);
}

#[test]
fn test_prealloc_layer_rejects_overflow() {
    let mut layer = PreallocKvCacheLayer::new(4).unwrap();
    let k = DynTensor::ones(&[1, 2, 3, 8], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 2, 3, 8], DType::F32, &Device::Cpu).unwrap();
    layer.append(&k, &v).unwrap();
    assert_eq!(layer.seq_len(), 3);

    // Try to append 2 more (would need 5, but max is 4).
    let k2 = DynTensor::ones(&[1, 2, 2, 8], DType::F32, &Device::Cpu).unwrap();
    let v2 = DynTensor::ones(&[1, 2, 2, 8], DType::F32, &Device::Cpu).unwrap();
    let result = layer.append(&k2, &v2);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("max_seq_len"),
        "error should mention max_seq_len: {msg}"
    );

    // seq_len should be unchanged after failed append.
    assert_eq!(layer.seq_len(), 3);
}

#[test]
fn test_prealloc_layer_rejects_mismatched_kv() {
    let mut layer = PreallocKvCacheLayer::new(128).unwrap();
    let k = DynTensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 4, 3, 32], DType::F32, &Device::Cpu).unwrap(); // wrong dim
    assert!(layer.append(&k, &v).is_err());
}

#[test]
fn test_prealloc_layer_rejects_rank_too_low() {
    let mut layer = PreallocKvCacheLayer::new(128).unwrap();
    let k = DynTensor::ones(&[4, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[4, 4], DType::F32, &Device::Cpu).unwrap();
    assert!(layer.append(&k, &v).is_err());
}

#[test]
fn test_prealloc_layer_rejects_dim_mismatch_on_second_append() {
    let mut layer = PreallocKvCacheLayer::new(128).unwrap();
    let k1 = DynTensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
    let v1 = DynTensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
    layer.append(&k1, &v1).unwrap();

    // Different num_heads on second append.
    let k2 = DynTensor::ones(&[1, 2, 1, 64], DType::F32, &Device::Cpu).unwrap();
    let v2 = DynTensor::ones(&[1, 2, 1, 64], DType::F32, &Device::Cpu).unwrap();
    assert!(layer.append(&k2, &v2).is_err());
}

#[test]
fn test_prealloc_layer_reset() {
    let mut layer = PreallocKvCacheLayer::new(128).unwrap();
    let k = DynTensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
    layer.append(&k, &v).unwrap();
    assert!(layer.is_allocated());

    layer.reset();
    assert!(layer.is_empty());
    assert_eq!(layer.seq_len(), 0);
    assert!(!layer.is_allocated());
    assert!(layer.key().unwrap().is_none());
    assert_eq!(layer.remaining_capacity(), 128);
}

#[test]
fn test_prealloc_layer_clear_preserves_allocation() {
    let mut layer = PreallocKvCacheLayer::new(128).unwrap();
    let k = DynTensor::ones(&[1, 4, 10, 64], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 4, 10, 64], DType::F32, &Device::Cpu).unwrap();
    layer.append(&k, &v).unwrap();
    assert_eq!(layer.seq_len(), 10);
    assert!(layer.is_allocated());

    layer.clear();
    assert_eq!(layer.seq_len(), 0);
    assert!(layer.is_empty());
    assert!(layer.is_allocated()); // buffers preserved
    assert_eq!(layer.remaining_capacity(), 128);

    // Re-append should work without re-allocation.
    let k2 = DynTensor::full(&[1, 4, 5, 64], 9.0, DType::F32, &Device::Cpu).unwrap();
    let v2 = DynTensor::full(&[1, 4, 5, 64], 8.0, DType::F32, &Device::Cpu).unwrap();
    let (full_k, _) = layer.append(&k2, &v2).unwrap();
    assert_eq!(full_k.dims(), &[1, 4, 5, 64]);
    let data = full_k.to_flat_vec::<f32>().unwrap();
    assert!((data[0] - 9.0).abs() < 1e-6);
}

#[test]
fn test_prealloc_layer_data_integrity_across_appends() {
    let mut layer = PreallocKvCacheLayer::new(128).unwrap();
    let head_dim = 4;

    for step in 0..20 {
        let val = (step + 1) as f64;
        let k = DynTensor::full(&[1, 1, 1, head_dim], val, DType::F32, &Device::Cpu).unwrap();
        let v =
            DynTensor::full(&[1, 1, 1, head_dim], val * 10.0, DType::F32, &Device::Cpu).unwrap();
        let (full_k, full_v) = layer.append(&k, &v).unwrap();

        // Verify all previous positions are correct.
        let k_data = full_k.to_flat_vec::<f32>().unwrap();
        let v_data = full_v.to_flat_vec::<f32>().unwrap();
        for j in 0..=step {
            let expected_k = (j + 1) as f32;
            let expected_v = expected_k * 10.0;
            assert!(
                (k_data[j * head_dim] - expected_k).abs() < 1e-6,
                "key[{j}] expected {expected_k}, got {}",
                k_data[j * head_dim]
            );
            assert!(
                (v_data[j * head_dim] - expected_v).abs() < 1e-5,
                "value[{j}] expected {expected_v}, got {}",
                v_data[j * head_dim]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// PreallocKvCache (multi-layer) tests
// ---------------------------------------------------------------------------

#[test]
fn test_prealloc_cache_new() {
    let cache = PreallocKvCache::new(2, 128).unwrap();
    assert_eq!(cache.num_layers(), 2);
    assert_eq!(cache.max_seq_len(), 128);
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.remaining_capacity(), 128);
}

#[test]
fn test_prealloc_cache_rejects_zero_max_seq() {
    assert!(PreallocKvCache::new(2, 0).is_err());
}

#[test]
fn test_prealloc_cache_layer_access() {
    let mut cache = PreallocKvCache::new(3, 128).unwrap();
    assert!(cache.layer(0).is_ok());
    assert!(cache.layer(2).is_ok());
    assert!(cache.layer(3).is_err());

    let k = DynTensor::ones(&[1, 4, 5, 64], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 4, 5, 64], DType::F32, &Device::Cpu).unwrap();
    cache.layer_mut(1).unwrap().append(&k, &v).unwrap();

    assert_eq!(cache.seq_len(), 5);
    assert!(!cache.is_empty());
    assert_eq!(cache.remaining_capacity(), 123);
}

#[test]
fn test_prealloc_cache_reset() {
    let mut cache = PreallocKvCache::new(2, 128).unwrap();
    let k = DynTensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
    cache.layer_mut(0).unwrap().append(&k, &v).unwrap();
    cache.layer_mut(1).unwrap().append(&k, &v).unwrap();

    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_prealloc_cache_clear_preserves_allocations() {
    let mut cache = PreallocKvCache::new(2, 128).unwrap();
    let k = DynTensor::ones(&[1, 4, 10, 64], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 4, 10, 64], DType::F32, &Device::Cpu).unwrap();
    cache.layer_mut(0).unwrap().append(&k, &v).unwrap();
    cache.layer_mut(1).unwrap().append(&k, &v).unwrap();

    assert!(cache.layer(0).unwrap().is_allocated());
    assert!(cache.layer(1).unwrap().is_allocated());

    cache.clear();
    assert!(cache.is_empty());
    assert!(cache.layer(0).unwrap().is_allocated());
    assert!(cache.layer(1).unwrap().is_allocated());
}

// ---------------------------------------------------------------------------
// KvCacheLayerBackend trait tests
// ---------------------------------------------------------------------------

#[test]
fn test_prealloc_layer_backend_trait() {
    let mut layer = PreallocKvCacheLayer::new(128).unwrap();
    let backend: &mut dyn KvCacheLayerBackend = &mut layer;

    assert!(backend.is_empty());
    assert_eq!(backend.seq_len(), 0);

    let k = DynTensor::ones(&[1, 4, 3, 64], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::full(&[1, 4, 3, 64], 2.0, DType::F32, &Device::Cpu).unwrap();
    let (full_k, full_v) = backend.append(&k, &v).unwrap();
    assert_eq!(full_k.dims(), &[1, 4, 3, 64]);
    assert_eq!(full_v.dims(), &[1, 4, 3, 64]);
    assert_eq!(backend.seq_len(), 3);
    assert!(!backend.is_empty());

    // Append one more token.
    let k2 = DynTensor::full(&[1, 4, 1, 64], 3.0, DType::F32, &Device::Cpu).unwrap();
    let v2 = DynTensor::full(&[1, 4, 1, 64], 4.0, DType::F32, &Device::Cpu).unwrap();
    let (full_k2, full_v2) = backend.append(&k2, &v2).unwrap();
    assert_eq!(full_k2.dims(), &[1, 4, 4, 64]);
    assert_eq!(full_v2.dims(), &[1, 4, 4, 64]);
    assert_eq!(backend.seq_len(), 4);

    backend.reset();
    assert!(backend.is_empty());
    assert_eq!(backend.seq_len(), 0);
}

// ---------------------------------------------------------------------------
// KvCacheBackend trait tests
// ---------------------------------------------------------------------------

#[test]
fn test_prealloc_cache_backend_trait() {
    let mut cache = PreallocKvCache::new(2, 128).unwrap();
    let backend: &mut dyn KvCacheBackend = &mut cache;

    assert_eq!(backend.num_layers(), 2);
    assert_eq!(backend.seq_len(), 0);

    // Append via trait.
    let k = DynTensor::ones(&[1, 4, 5, 64], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 4, 5, 64], DType::F32, &Device::Cpu).unwrap();
    backend
        .layer_backend_mut(0)
        .unwrap()
        .append(&k, &v)
        .unwrap();
    assert_eq!(backend.seq_len(), 5);

    // Out-of-range layer.
    assert!(backend.layer_backend_mut(99).is_err());

    backend.reset();
    assert_eq!(backend.seq_len(), 0);
}

// ---------------------------------------------------------------------------
// Full decode simulation: 2 layers, 4 heads, 64 head_dim, max_seq=128
// ---------------------------------------------------------------------------

#[test]
fn test_prealloc_cache_full_decode_simulation() {
    let num_layers = 2;
    let num_heads = 4;
    let head_dim = 64;
    let max_seq = 128;
    let batch = 1;
    let prompt_len = 8;
    let decode_steps = 20;

    let mut cache = PreallocKvCache::new(num_layers, max_seq).unwrap();

    // Prefill: append prompt K/V for all layers.
    for layer_idx in 0..num_layers {
        let k = DynTensor::full(
            &[batch, num_heads, prompt_len, head_dim],
            1.0,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let v = DynTensor::full(
            &[batch, num_heads, prompt_len, head_dim],
            2.0,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let (full_k, full_v) = cache.layer_mut(layer_idx).unwrap().append(&k, &v).unwrap();
        assert_eq!(full_k.dim(2).unwrap(), prompt_len);
        assert_eq!(full_v.dim(2).unwrap(), prompt_len);
    }
    assert_eq!(cache.seq_len(), prompt_len);

    // Decode: single-token appends.
    for step in 0..decode_steps {
        for layer_idx in 0..num_layers {
            let val = (step + 10) as f64;
            let k = DynTensor::full(
                &[batch, num_heads, 1, head_dim],
                val,
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
            let v = DynTensor::full(
                &[batch, num_heads, 1, head_dim],
                val * 10.0,
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
            let (full_k, full_v) = cache.layer_mut(layer_idx).unwrap().append(&k, &v).unwrap();
            let expected_seq = prompt_len + step + 1;
            assert_eq!(full_k.dim(2).unwrap(), expected_seq);
            assert_eq!(full_v.dim(2).unwrap(), expected_seq);
        }
    }

    let final_seq = prompt_len + decode_steps;
    assert_eq!(cache.seq_len(), final_seq);
    assert_eq!(cache.remaining_capacity(), max_seq - final_seq);

    // Verify data at layer 0: prompt positions should be 1.0, decode should be 10..29.
    let layer0_k = cache.layer(0).unwrap().key().unwrap().unwrap();
    let k_data = layer0_k.to_flat_vec::<f32>().unwrap();
    // Head 0, position 0 (prompt): should be 1.0
    assert!(
        (k_data[0] - 1.0).abs() < 1e-6,
        "prompt position 0 should be 1.0, got {}",
        k_data[0]
    );
    // Head 0, position prompt_len (first decode step): should be 10.0
    let decode_start_idx = prompt_len * head_dim;
    assert!(
        (k_data[decode_start_idx] - 10.0).abs() < 1e-6,
        "first decode position should be 10.0, got {}",
        k_data[decode_start_idx]
    );
}
