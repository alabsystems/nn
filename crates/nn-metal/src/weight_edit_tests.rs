// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{MetalBuffer, MetalContext};

fn create_test_buffer(ctx: &MetalContext, len: usize) -> MetalBuffer {
    let data = vec![0.0f32; len];
    ctx.create_buffer(&data).expect("create test buffer")
}

#[test]
fn test_apply_weight_edit_basic() {
    let ctx = MetalContext::new().expect("Metal context");
    let mut buf = create_test_buffer(&ctx, 4);
    let new_data = [1.0f32, 2.0, 3.0, 4.0];
    let spec = WeightEditSpec {
        layer_name: "test.weight",
        new_data: &new_data,
    };

    let count = apply_weight_edit(&mut buf, &spec).expect("apply");
    assert_eq!(count, 4);

    // Verify data was written
    let readback = buf.contents::<f32>().expect("readback");
    assert_eq!(readback, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_apply_weight_edit_empty_data() {
    let ctx = MetalContext::new().expect("Metal context");
    let mut buf = create_test_buffer(&ctx, 4);
    let spec = WeightEditSpec {
        layer_name: "test.weight",
        new_data: &[],
    };

    let err = apply_weight_edit(&mut buf, &spec).unwrap_err();
    assert!(
        matches!(err, WeightEditError::EmptyData { .. }),
        "expected EmptyData, got {err:?}"
    );
}

#[test]
fn test_apply_weight_edit_nan_rejected() {
    let ctx = MetalContext::new().expect("Metal context");
    let mut buf = create_test_buffer(&ctx, 4);
    let new_data = [1.0f32, f32::NAN, 3.0, 4.0];
    let spec = WeightEditSpec {
        layer_name: "test.weight",
        new_data: &new_data,
    };

    let err = apply_weight_edit(&mut buf, &spec).unwrap_err();
    match err {
        WeightEditError::NonFiniteData { count, .. } => assert_eq!(count, 1),
        other => panic!("expected NonFiniteData, got {other:?}"),
    }
}

#[test]
fn test_apply_weight_edit_inf_rejected() {
    let ctx = MetalContext::new().expect("Metal context");
    let mut buf = create_test_buffer(&ctx, 4);
    let new_data = [f32::INFINITY, f32::NEG_INFINITY, 3.0, 4.0];
    let spec = WeightEditSpec {
        layer_name: "test.weight",
        new_data: &new_data,
    };

    let err = apply_weight_edit(&mut buf, &spec).unwrap_err();
    match err {
        WeightEditError::NonFiniteData { count, .. } => assert_eq!(count, 2),
        other => panic!("expected NonFiniteData, got {other:?}"),
    }
}

#[test]
fn test_apply_weight_edit_with_generation() {
    let ctx = MetalContext::new().expect("Metal context");
    let mut buf = create_test_buffer(&ctx, 4);
    let cache: crate::GpuWeightCache<Vec<f32>> = crate::GpuWeightCache::new();

    // Initialize the cache with some data
    let _ = cache.get_or_init_with(|| Ok(vec![0.0; 4]), |e: String| e);
    assert_eq!(cache.generation(), 0);

    let new_data = [1.0f32, 2.0, 3.0, 4.0];
    let spec = WeightEditSpec {
        layer_name: "test.weight",
        new_data: &new_data,
    };

    let result = apply_weight_edit_with_generation(&mut buf, &spec, &cache).expect("apply");
    assert_eq!(result.previous_generation, 0);
    assert_eq!(result.new_generation, 1);
    assert_eq!(result.elements_written, 4);
}

#[test]
fn test_apply_weight_edit_multiple_generations() {
    let ctx = MetalContext::new().expect("Metal context");
    let mut buf = create_test_buffer(&ctx, 4);
    let cache: crate::GpuWeightCache<Vec<f32>> = crate::GpuWeightCache::new();

    for i in 0..3u64 {
        let new_data = [i as f32; 4];
        let spec = WeightEditSpec {
            layer_name: "test.weight",
            new_data: &new_data,
        };
        let result = apply_weight_edit_with_generation(&mut buf, &spec, &cache).expect("apply");
        assert_eq!(result.previous_generation, i);
        assert_eq!(result.new_generation, i + 1);
    }
    assert_eq!(cache.generation(), 3);
}

#[test]
fn test_weight_edit_error_display() {
    let err = WeightEditError::EmptyData {
        layer_name: "encoder.weight".to_string(),
    };
    assert!(err.to_string().contains("empty data"));

    let err = WeightEditError::NonFiniteData {
        layer_name: "decoder.weight".to_string(),
        count: 5,
    };
    assert!(err.to_string().contains("5 non-finite"));
}

#[test]
fn test_partial_write_smaller_than_buffer() {
    let ctx = MetalContext::new().expect("Metal context");
    let mut buf = create_test_buffer(&ctx, 8);
    // Write only 4 elements into an 8-element buffer
    let new_data = [1.0f32, 2.0, 3.0, 4.0];
    let spec = WeightEditSpec {
        layer_name: "test.weight",
        new_data: &new_data,
    };

    let count = apply_weight_edit(&mut buf, &spec).expect("apply");
    assert_eq!(count, 4);

    // First 4 elements should be updated
    let readback = buf.contents::<f32>().expect("readback");
    assert_eq!(&readback[..4], &[1.0, 2.0, 3.0, 4.0]);
}
