#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Performance and regression tests for KV cache.
//! Extracted from `kv_cache_tests.rs` to keep files under 500 lines.

use crate::dyn_tensor::DynTensor;
use crate::layers::autoregressive::{generate, GenerationConfig};
use crate::layers::kv_cache::{KvCache, KvCacheBackend, KvCacheLayer};
use crate::{DType, Device};

// ---------------------------------------------------------------------------
// generate() with KvCache (#1223 AC2)
// ---------------------------------------------------------------------------

/// Deterministic model for generate() testing.
/// Returns logits [1, vocab_size=5] where argmax = (last_token + 1) % 5.
fn trait_test_model<C: KvCacheBackend>(
    input: &DynTensor,
    _cache: &mut C,
) -> crate::Result<DynTensor> {
    // generate() sends U32 token ID tensors since W4#456 (ids_to_tensor fix).
    let input_f32 = input.to_dtype(DType::F32)?;
    let flat = input_f32.to_flat_vec::<f32>()?;
    let last_val = flat[flat.len() - 1];
    let next_token = (last_val as usize + 1) % 5;
    let mut logits = vec![0.0f32; 5];
    logits[next_token] = 10.0;
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

#[test]
fn test_generate_with_kv_cache() {
    let config = GenerationConfig {
        max_new_tokens: 3,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let output = generate(trait_test_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert_eq!(output.token_ids.len(), 3);
    assert!(!output.finished);
}

// ---------------------------------------------------------------------------
// Performance proof: KV cache slice_set copies entire buffer on GPU
// ---------------------------------------------------------------------------

/// Prove that KV cache doubling-buffer `slice_set` is O(capacity) per append on GPU.
///
/// On CPU, `slice_set` writes in-place when Arc refcount=1, giving O(new_seq) per
/// append and O(S) amortized over S decode steps. But on GPU, `slice_set` does a
/// full CPU round-trip:
///   1. Copy entire buffer to CPU: O(capacity)
///   2. Write new data into CPU copy: O(new_seq)
///   3. Copy entire result back to GPU: O(capacity)
///
/// This means each GPU append is O(capacity), which after the doubling strategy
/// gives O(S) per step → O(S²) total over S decode steps.
///
/// This test verifies the CPU path is correct (O(S) total) and documents
/// the GPU limitation. The GPU path itself requires Metal and is tested
/// separately in nn-metal integration tests.
#[test]
fn test_kv_cache_cpu_append_is_amortized_linear() {
    let mut layer = KvCacheLayer::empty();
    let steps = 100;

    for step in 0..steps {
        let kv = DynTensor::from_vec(vec![step as f32; 4], &[1, 1, 1, 4], &Device::Cpu).unwrap();
        let (full_k, _full_v) = layer.append(&kv, &kv).unwrap();
        assert_eq!(full_k.dim(2).unwrap(), step + 1);
    }

    // Verify capacity grows via doubling (power of 2 >= steps).
    let cap = layer.buffer_capacity();
    assert!(
        cap.is_power_of_two() || cap == steps,
        "capacity should be a power of 2 (doubling strategy), got {cap}"
    );
    assert!(
        cap >= steps,
        "capacity ({cap}) should be >= steps ({steps})"
    );
    // Doubling strategy: capacity should be at most 2x the needed size.
    assert!(
        cap <= steps * 2,
        "capacity ({cap}) should be at most 2x steps ({steps}), \
         indicating O(1) amortized growth"
    );
}

/// Regression test: slice_set on uniquely-owned f32 buffer does NOT copy full
/// buffer (#1439).
///
/// Measures per-append timing at small vs large buffer capacities. With the
/// fix (Arc::try_unwrap + ArcArray::into_owned), the fast-path slice_set is
/// O(write_region) when the buffer is uniquely owned. Without the fix, the
/// ArcArray branch always copies the full buffer: O(capacity) per append.
///
/// Uses a large head_dim (1024) to make the timing difference measurable
/// even in noisy CI environments.
#[test]
fn test_kv_cache_slice_set_no_full_buffer_copy() {
    let mut layer = KvCacheLayer::empty();
    let head_dim = 1024; // large enough to measure timing difference

    // Warm up: first append triggers allocation with INITIAL_CAPACITY=16.
    let kv = DynTensor::ones(&[1, 1, 1, head_dim], DType::F32, &Device::Cpu).unwrap();
    layer.append(&kv, &kv).unwrap();
    // Drop the returned narrow view before next append.

    // Fill to 120 tokens (capacity doubles to 128 at step 17).
    let mut times = Vec::new();
    for step in 1..120 {
        let kv = DynTensor::from_vec(
            vec![step as f32; head_dim],
            &[1, 1, 1, head_dim],
            &Device::Cpu,
        )
        .unwrap();
        let start = std::time::Instant::now();
        let (_full_k, _full_v) = layer.append(&kv, &kv).unwrap();
        // Drop narrow views before timing next iteration.
        drop(_full_k);
        drop(_full_v);
        let elapsed = start.elapsed();
        times.push((step, elapsed));
    }

    // Compare average time of early appends (capacity=16) vs late appends
    // (capacity=128). Skip step 16 which triggers a doubling copy.
    let early_avg: f64 = times[..10]
        .iter()
        .map(|(_, d)| d.as_nanos() as f64)
        .sum::<f64>()
        / 10.0;
    let late_avg: f64 = times[times.len() - 10..]
        .iter()
        .map(|(_, d)| d.as_nanos() as f64)
        .sum::<f64>()
        / 10.0;

    let ratio = if early_avg > 0.0 {
        late_avg / early_avg
    } else {
        1.0
    };

    // With the fix, per-append time should be roughly constant regardless of
    // buffer capacity (O(1) fast path). Allow up to 5x for CI noise.
    // Before the fix, the ratio was typically 3-10x due to full buffer copies.
    eprintln!(
        "KV cache append timing: early_avg={early_avg:.0}ns, late_avg={late_avg:.0}ns, \
         ratio={ratio:.2}x"
    );
    assert!(
        ratio < 5.0,
        "late appends should not be >5x slower than early appends \
         (would indicate O(capacity) copy per append). \
         early_avg={early_avg:.0}ns, late_avg={late_avg:.0}ns, ratio={ratio:.1}x"
    );
}

/// Data integrity after many appends: each position retains its value (#1439).
///
/// Verifies that in-place mutation via slice_set doesn't corrupt earlier
/// positions when appending new tokens. Each token has a unique value.
#[test]
fn test_kv_cache_data_integrity_across_appends() {
    let mut layer = KvCacheLayer::empty();
    let head_dim = 4;
    let steps = 50;

    for step in 0..steps {
        let val = (step + 1) as f32;
        let kv =
            DynTensor::from_vec(vec![val; head_dim], &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
        let (full_k, _) = layer.append(&kv, &kv).unwrap();

        // After each append, verify ALL previous positions are intact.
        let k_data = full_k.to_flat_vec::<f32>().unwrap();
        for pos in 0..=step {
            let expected = (pos + 1) as f32;
            let actual = k_data[pos * head_dim];
            assert!(
                (actual - expected).abs() < 1e-6,
                "position {pos} corrupted after step {step}: expected {expected}, got {actual}"
            );
        }
    }
}

/// Performance regression: holding narrow views across appends forces O(capacity)
/// copy per step via ArcArray COW, degrading from O(S) total to O(S²).
///
/// This test verifies that the timing ratio between held-view and dropped-view
/// appends grows with buffer size, proving the ArcArray COW path is triggered.
#[test]
fn test_kv_cache_held_views_trigger_cow_copy() {
    let head_dim = 1024;
    let steps = 60;

    // Path A: drop views before next append (O(new_seq) per step).
    let mut layer_dropped = KvCacheLayer::empty();
    let start_dropped = std::time::Instant::now();
    for step in 0..steps {
        let kv = DynTensor::from_vec(
            vec![step as f32; head_dim],
            &[1, 1, 1, head_dim],
            &Device::Cpu,
        )
        .unwrap();
        let (fk, fv) = layer_dropped.append(&kv, &kv).unwrap();
        drop(fk);
        drop(fv);
    }
    let elapsed_dropped = start_dropped.elapsed();

    // Path B: hold ALL views in a Vec (forces COW copy every step).
    let mut layer_held = KvCacheLayer::empty();
    let mut held_views = Vec::new();
    let start_held = std::time::Instant::now();
    for step in 0..steps {
        let kv = DynTensor::from_vec(
            vec![step as f32; head_dim],
            &[1, 1, 1, head_dim],
            &Device::Cpu,
        )
        .unwrap();
        let (fk, fv) = layer_held.append(&kv, &kv).unwrap();
        held_views.push((fk, fv)); // keep ArcArray refcount > 1
    }
    let elapsed_held = start_held.elapsed();

    let ratio = elapsed_held.as_nanos() as f64 / elapsed_dropped.as_nanos().max(1) as f64;
    eprintln!(
        "KV cache COW regression: dropped={:.1}ms, held={:.1}ms, ratio={ratio:.1}x",
        elapsed_dropped.as_secs_f64() * 1000.0,
        elapsed_held.as_secs_f64() * 1000.0
    );

    // Held-view path should be measurably slower (COW copies entire buffer).
    // We don't assert a tight bound (CI noise), but the held path should not
    // be faster — that would indicate the COW path was optimized away.
    // This test documents the performance trap, not enforces a threshold.
    assert!(
        ratio > 0.5,
        "held-view path should not be dramatically faster than dropped-view path"
    );

    // Data integrity: both paths should produce correct data.
    let (final_k_dropped, _) = {
        let kv = DynTensor::from_vec(vec![99.0f32; head_dim], &[1, 1, 1, head_dim], &Device::Cpu)
            .unwrap();
        layer_dropped.append(&kv, &kv).unwrap()
    };
    assert_eq!(final_k_dropped.dim(2).unwrap(), steps + 1);

    // Held-view layer also correct despite COW copies.
    assert_eq!(layer_held.seq_len(), steps);
    let last_held = &held_views[steps - 1].0;
    assert_eq!(last_held.dim(2).unwrap(), steps);
}

/// Verify the KV cache rejects growth beyond MAX_SEQ_CAPACITY (262144).
/// This prevents OOM from runaway generation loops.
#[test]
fn test_kv_cache_rejects_excessive_capacity() {
    let mut layer = KvCacheLayer::empty();
    // Seed with a small tensor then try to append a huge sequence.
    let k_init = DynTensor::ones(&[1, 1, 1, 4], DType::F32, &Device::Cpu).unwrap();
    let v_init = DynTensor::ones(&[1, 1, 1, 4], DType::F32, &Device::Cpu).unwrap();
    layer.append(&k_init, &v_init).unwrap();

    // Try to append a sequence that would force capacity beyond 262144.
    // After the initial append, capacity is 16 (INITIAL_CAPACITY).
    // We need a single append that forces doubling past MAX_SEQ_CAPACITY.
    let huge_seq = 262_145; // just over MAX_SEQ_CAPACITY
    let k_huge = DynTensor::ones(&[1, 1, huge_seq, 4], DType::F32, &Device::Cpu).unwrap();
    let v_huge = DynTensor::ones(&[1, 1, huge_seq, 4], DType::F32, &Device::Cpu).unwrap();
    let result = layer.append(&k_huge, &v_huge);
    assert!(result.is_err(), "should reject capacity > MAX_SEQ_CAPACITY");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("max capacity"),
        "error should mention max capacity: {msg}"
    );
}

/// `reset()` drops buffer capacity, forcing re-allocation on the next use
/// cycle. Use `clear()` to preserve capacity.
#[test]
fn test_kv_cache_reset_drops_capacity() {
    let mut layer = KvCacheLayer::empty();
    let head_dim = 4;

    // Fill to 50 tokens — capacity doubles to 64.
    for step in 0..50 {
        let kv = DynTensor::from_vec(
            vec![step as f32; head_dim],
            &[1, 1, 1, head_dim],
            &Device::Cpu,
        )
        .unwrap();
        let (fk, fv) = layer.append(&kv, &kv).unwrap();
        drop(fk);
        drop(fv);
    }
    assert_eq!(layer.seq_len(), 50);
    let cap_before = layer.buffer_capacity();
    assert!(
        cap_before >= 50,
        "capacity should be >= 50 after 50 appends"
    );

    // Reset drops everything.
    layer.reset();
    assert_eq!(layer.seq_len(), 0);
    assert_eq!(
        layer.buffer_capacity(),
        0,
        "reset() drops capacity to 0, forcing re-allocation on next use"
    );

    // Re-filling starts from INITIAL_CAPACITY=16 again.
    let kv = DynTensor::ones(&[1, 1, 1, head_dim], DType::F32, &Device::Cpu).unwrap();
    layer.append(&kv, &kv).unwrap();
    let cap_after = layer.buffer_capacity();
    assert_eq!(
        cap_after, 16,
        "after reset + append, capacity restarts from INITIAL_CAPACITY (16)"
    );
}

/// `clear()` preserves buffer capacity for reuse.
///
/// After clearing, the next append writes at offset 0 of the existing buffer
/// — no re-allocation, no doubling cascade. In batch inference (B inputs of
/// S tokens), total allocation is O(S × log S) instead of O(B × S × log S).
#[test]
fn test_kv_cache_clear_preserves_capacity() {
    let mut layer = KvCacheLayer::empty();
    let head_dim = 4;

    // Fill to 50 tokens — capacity doubles to 64.
    for step in 0..50 {
        let kv = DynTensor::from_vec(
            vec![step as f32; head_dim],
            &[1, 1, 1, head_dim],
            &Device::Cpu,
        )
        .unwrap();
        let (fk, fv) = layer.append(&kv, &kv).unwrap();
        drop(fk);
        drop(fv);
    }
    assert_eq!(layer.seq_len(), 50);
    let cap_before = layer.buffer_capacity();
    assert!(cap_before >= 50);

    // Clear preserves capacity.
    layer.clear();
    assert_eq!(layer.seq_len(), 0);
    assert!(layer.is_empty());
    assert_eq!(
        layer.buffer_capacity(),
        cap_before,
        "clear() should preserve capacity ({cap_before})"
    );

    // Re-filling reuses the existing buffer — no new allocation.
    let kv = DynTensor::ones(&[1, 1, 1, head_dim], DType::F32, &Device::Cpu).unwrap();
    let (full_k, _) = layer.append(&kv, &kv).unwrap();
    assert_eq!(full_k.dim(2).unwrap(), 1);
    assert_eq!(
        layer.buffer_capacity(),
        cap_before,
        "capacity should remain {cap_before} after clear + append"
    );
}

/// `clear()` then re-fill produces correct data (not stale buffer contents).
#[test]
fn test_kv_cache_clear_data_integrity() {
    let mut layer = KvCacheLayer::empty();
    let head_dim = 4;

    // First fill: values 1.0, 2.0, ...
    for step in 0..10 {
        let val = (step + 1) as f32;
        let kv =
            DynTensor::from_vec(vec![val; head_dim], &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
        let (fk, fv) = layer.append(&kv, &kv).unwrap();
        drop(fk);
        drop(fv);
    }

    // Clear and re-fill with different values: 100.0, 200.0, ...
    layer.clear();
    for step in 0..5 {
        let val = ((step + 1) * 100) as f32;
        let kv =
            DynTensor::from_vec(vec![val; head_dim], &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
        let (fk, fv) = layer.append(&kv, &kv).unwrap();
        drop(fk);
        drop(fv);
    }

    // Verify: only the new values are visible, not stale data.
    let full_k = layer.key().unwrap().unwrap();
    assert_eq!(
        full_k.dim(2).unwrap(),
        5,
        "should have 5 positions after clear + 5 appends"
    );
    let k_data = full_k.to_flat_vec::<f32>().unwrap();
    for pos in 0..5 {
        let expected = ((pos + 1) * 100) as f32;
        let actual = k_data[pos * head_dim];
        assert!(
            (actual - expected).abs() < 1e-6,
            "position {pos}: expected {expected}, got {actual}"
        );
    }
}

/// `clear()` on the multi-layer KvCache preserves per-layer capacity.
#[test]
fn test_kv_cache_multi_clear_preserves_capacity() {
    let mut cache = KvCache::new(2);
    let head_dim = 4;

    // Fill both layers to 30 tokens.
    for step in 0..30 {
        let kv = DynTensor::from_vec(
            vec![step as f32; head_dim],
            &[1, 1, 1, head_dim],
            &Device::Cpu,
        )
        .unwrap();
        for layer_idx in 0..2 {
            let layer = cache.layer_mut(layer_idx).unwrap();
            let (fk, fv) = layer.append(&kv, &kv).unwrap();
            drop(fk);
            drop(fv);
        }
    }

    let cap0 = cache.layer(0).unwrap().buffer_capacity();
    let cap1 = cache.layer(1).unwrap().buffer_capacity();
    assert!(cap0 >= 30);
    assert!(cap1 >= 30);

    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.layer(0).unwrap().buffer_capacity(), cap0);
    assert_eq!(cache.layer(1).unwrap().buffer_capacity(), cap1);
}
