// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`LiveEditApply`].

use super::*;
use crate::context::MetalContext;
use crate::weight_edit::{WeightEditError, WeightEditSpec};
use nn_core::layers::KvCacheLayer;

/// Helper: create a MetalBuffer with the given f32 data.
fn make_buffer(ctx: &MetalContext, data: &[f32]) -> MetalBuffer {
    ctx.create_buffer(data).expect("create buffer")
}

#[test]
fn test_apply_basic_no_kv_cache() {
    let ctx = MetalContext::new().expect("Metal device");
    let original = [1.0_f32, 2.0, 3.0, 4.0];
    let mut buf = make_buffer(&ctx, &original);

    let new_data = [10.0_f32, 20.0, 30.0, 40.0];
    let spec = WeightEditSpec {
        layer_name: "test.weight",
        new_data: &new_data,
    };

    let receipt = LiveEditApply::apply(&mut buf, &spec, None).expect("apply");
    assert_eq!(receipt.elements_written, 4);
    assert!(!receipt.kv_invalidated);
    assert_eq!(receipt.kv_generation_before, 0);
    assert_eq!(receipt.kv_generation_after, 0);

    // Verify buffer contents.
    let contents: &[f32] = buf.contents().expect("readback");
    assert_eq!(contents, &[10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_apply_with_kv_cache_invalidation() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut buf = make_buffer(&ctx, &[0.0_f32; 8]);
    let mut cache = KvCacheLayer::empty();

    // Pre-check: generation starts at 0.
    assert_eq!(cache.weight_generation(), 0);

    let spec = WeightEditSpec {
        layer_name: "encoder.linear.weight",
        new_data: &[1.0_f32; 8],
    };

    let receipt = LiveEditApply::apply(&mut buf, &spec, Some(&mut cache)).expect("apply");
    assert_eq!(receipt.elements_written, 8);
    assert!(receipt.kv_invalidated);
    assert_eq!(receipt.kv_generation_before, 0);
    assert_eq!(receipt.kv_generation_after, 1);
    assert_eq!(cache.weight_generation(), 1);
}

#[test]
fn test_apply_multiple_edits_bump_generation() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut buf = make_buffer(&ctx, &[0.0_f32; 4]);
    let mut cache = KvCacheLayer::empty();

    for i in 0..5u64 {
        let data = vec![i as f32; 4];
        let spec = WeightEditSpec {
            layer_name: "layer.weight",
            new_data: &data,
        };
        let receipt = LiveEditApply::apply(&mut buf, &spec, Some(&mut cache)).expect("apply");
        assert_eq!(receipt.kv_generation_before, i);
        assert_eq!(receipt.kv_generation_after, i + 1);
    }
    assert_eq!(cache.weight_generation(), 5);
}

#[test]
fn test_apply_rejects_nan_data() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut buf = make_buffer(&ctx, &[0.0_f32; 4]);

    let bad_data = [1.0_f32, f32::NAN, 3.0, 4.0];
    let spec = WeightEditSpec {
        layer_name: "bad.weight",
        new_data: &bad_data,
    };

    let err = LiveEditApply::apply(&mut buf, &spec, None).unwrap_err();
    assert!(matches!(err, LiveEditError::WeightEdit(_)));
}

#[test]
fn test_apply_rejects_inf_data() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut buf = make_buffer(&ctx, &[0.0_f32; 4]);

    let bad_data = [1.0_f32, 2.0, f32::INFINITY, 4.0];
    let spec = WeightEditSpec {
        layer_name: "bad.weight",
        new_data: &bad_data,
    };

    let err = LiveEditApply::apply(&mut buf, &spec, None).unwrap_err();
    assert!(matches!(err, LiveEditError::WeightEdit(_)));
}

#[test]
fn test_apply_rejects_empty_data() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut buf = make_buffer(&ctx, &[0.0_f32; 4]);

    let spec = WeightEditSpec {
        layer_name: "empty.weight",
        new_data: &[],
    };

    let err = LiveEditApply::apply(&mut buf, &spec, None).unwrap_err();
    assert!(matches!(err, LiveEditError::WeightEdit(_)));
}

#[test]
fn test_apply_kv_cache_cleared_on_invalidate() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut buf = make_buffer(&ctx, &[0.0_f32; 4]);

    // Build a cache with some entries by appending dummy data.
    let mut cache = KvCacheLayer::empty();
    let arr_k = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[1, 1, 2, 4]));
    let k = DynTensor::from_cpu_f32(arr_k).expect("key tensor");
    let arr_v = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[1, 1, 2, 4]));
    let v = DynTensor::from_cpu_f32(arr_v).expect("value tensor");
    let _ = cache.append(&k, &v).expect("append");
    assert_eq!(cache.seq_len(), 2);

    // Now apply a weight edit — should clear the cache.
    let spec = WeightEditSpec {
        layer_name: "encoder.weight",
        new_data: &[1.0_f32; 4],
    };
    let receipt = LiveEditApply::apply(&mut buf, &spec, Some(&mut cache)).expect("apply");
    assert!(receipt.kv_invalidated);
    assert_eq!(
        cache.seq_len(),
        0,
        "cache should be cleared after invalidate"
    );
    assert_eq!(cache.weight_generation(), 1);
}

#[test]
fn test_error_display() {
    let err = LiveEditError::WeightEdit(WeightEditError::EmptyData {
        layer_name: "test".to_string(),
    });
    let msg = format!("{err}");
    assert!(
        msg.contains("empty"),
        "error message should mention empty: {msg}"
    );
}

// ── Delta-apply tests ──────────────────────────────────────────────

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::KvCache;

/// Helper: create a DynTensor from a flat f32 slice.
fn make_delta(data: &[f32]) -> DynTensor {
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[data.len()]), data.to_vec())
        .expect("delta array");
    DynTensor::from_cpu_f32(arr).expect("delta tensor")
}

#[test]
fn test_apply_delta_basic_no_kv_cache() {
    let ctx = MetalContext::new().expect("Metal device");
    let original = [1.0_f32, 2.0, 3.0, 4.0];
    let mut buf = make_buffer(&ctx, &original);

    let delta = make_delta(&[0.5, -0.5, 1.0, -1.0]);

    let receipt = LiveEditApply::apply_delta(&mut buf, &delta, None).expect("apply_delta");
    assert_eq!(receipt.elements_written, 4);
    assert_eq!(receipt.layers_invalidated, 0);

    // Verify: W_new = W_old + ΔW.
    let contents: &[f32] = buf.contents().expect("readback");
    assert_eq!(contents, &[1.5, 1.5, 4.0, 3.0]);
}

#[test]
fn test_apply_delta_with_multi_layer_kv_cache() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut buf = make_buffer(&ctx, &[0.0_f32; 4]);
    let mut cache = KvCache::new(3);

    // Verify all layers start at generation 0.
    for i in 0..3 {
        assert_eq!(cache.layer(i).unwrap().weight_generation(), 0);
    }

    let delta = make_delta(&[1.0, 2.0, 3.0, 4.0]);
    let receipt =
        LiveEditApply::apply_delta(&mut buf, &delta, Some(&mut cache)).expect("apply_delta");

    assert_eq!(receipt.elements_written, 4);
    assert_eq!(receipt.layers_invalidated, 3);

    // All layers should have generation bumped to 1.
    for i in 0..3 {
        assert_eq!(cache.layer(i).unwrap().weight_generation(), 1);
    }

    // Buffer should contain the delta values (0 + delta = delta).
    let contents: &[f32] = buf.contents().expect("readback");
    assert_eq!(contents, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_apply_delta_accumulates_over_multiple_edits() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut buf = make_buffer(&ctx, &[10.0_f32; 4]);

    // Apply three successive deltas.
    for _ in 0..3 {
        let delta = make_delta(&[1.0; 4]);
        LiveEditApply::apply_delta(&mut buf, &delta, None).expect("apply_delta");
    }

    // 10.0 + 3 * 1.0 = 13.0
    let contents: &[f32] = buf.contents().expect("readback");
    assert_eq!(contents, &[13.0, 13.0, 13.0, 13.0]);
}

#[test]
fn test_apply_delta_rejects_size_mismatch() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut buf = make_buffer(&ctx, &[0.0_f32; 4]);

    // Delta has 3 elements, buffer has 4.
    let delta = make_delta(&[1.0, 2.0, 3.0]);

    let err = LiveEditApply::apply_delta(&mut buf, &delta, None).unwrap_err();
    assert!(
        matches!(
            err,
            LiveEditError::DeltaSizeMismatch {
                buffer_len: 4,
                delta_len: 3
            }
        ),
        "expected DeltaSizeMismatch, got: {err:?}"
    );
}

#[test]
fn test_apply_delta_rejects_nan_in_delta() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut buf = make_buffer(&ctx, &[1.0_f32; 4]);

    // Delta contains NaN — result will be NaN → NonFiniteResult.
    let delta = make_delta(&[0.0, f32::NAN, 0.0, 0.0]);

    let err = LiveEditApply::apply_delta(&mut buf, &delta, None).unwrap_err();
    assert!(
        matches!(err, LiveEditError::NonFiniteResult { count: 1 }),
        "expected NonFiniteResult, got: {err:?}"
    );
}

#[test]
fn test_apply_delta_rejects_inf_result() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut buf = make_buffer(&ctx, &[f32::MAX; 4]);

    // Adding MAX to MAX overflows to Inf.
    let delta = make_delta(&[f32::MAX; 4]);

    let err = LiveEditApply::apply_delta(&mut buf, &delta, None).unwrap_err();
    assert!(
        matches!(err, LiveEditError::NonFiniteResult { count: 4 }),
        "expected NonFiniteResult with count=4, got: {err:?}"
    );
}

#[test]
fn test_apply_delta_preserves_buffer_on_error() {
    let ctx = MetalContext::new().expect("Metal device");
    let original = [1.0_f32, 2.0, 3.0, 4.0];
    let mut buf = make_buffer(&ctx, &original);

    // Try a mismatched delta — should fail without modifying the buffer.
    let delta = make_delta(&[1.0, 2.0]);
    let _ = LiveEditApply::apply_delta(&mut buf, &delta, None);

    // Buffer should still contain original data.
    let contents: &[f32] = buf.contents().expect("readback");
    assert_eq!(contents, &original);
}

#[test]
fn test_apply_delta_error_display() {
    let err = LiveEditError::DeltaSizeMismatch {
        buffer_len: 512,
        delta_len: 256,
    };
    let msg = format!("{err}");
    assert!(msg.contains("512"), "should mention buffer_len: {msg}");
    assert!(msg.contains("256"), "should mention delta_len: {msg}");

    let err = LiveEditError::NonFiniteResult { count: 3 };
    let msg = format!("{err}");
    assert!(msg.contains("3"), "should mention count: {msg}");
    assert!(
        msg.contains("non-finite"),
        "should mention non-finite: {msg}"
    );
}
