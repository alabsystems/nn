#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `nn` module: Module trait, Linear, and NaN/Inf propagation.
//!
//! Norm tests (LayerNorm, GroupNorm, BatchNorm): `nn_tests_norm.rs`
//! Embedding tests: `nn_tests_embedding.rs`

use super::*;
use crate::{DType, Device, DynTensor, Result, TensorError};

#[path = "tests_norm.rs"]
mod norm;

#[path = "tests_embedding.rs"]
mod embedding;

// -- Module trait + apply ----------------------------------------------------

#[test]
fn test_apply_uses_module_forward() {
    struct Identity;
    impl Module for Identity {
        fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
            Ok(x.clone())
        }
    }
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let y = x.apply(&Identity).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

// -- Linear ------------------------------------------------------------------

#[test]
fn test_linear_no_bias() {
    let w = DynTensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3], &Device::Cpu).unwrap();
    let linear = Linear::new(w, None).unwrap();
    let x = DynTensor::new(&[3.0, 5.0, 7.0], &[1, 3], &Device::Cpu).unwrap();
    let y = linear.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 3.0).abs() < 1e-6);
    assert!((vals[1] - 5.0).abs() < 1e-6);
}

#[test]
fn test_linear_with_bias() {
    let w = DynTensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let b = DynTensor::new(&[10.0, 20.0], &[2], &Device::Cpu).unwrap();
    let linear = Linear::new(w, Some(b)).unwrap();
    let x = DynTensor::new(&[1.0, 2.0], &[1, 2], &Device::Cpu).unwrap();
    let y = linear.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 11.0).abs() < 1e-6);
    assert!((vals[1] - 22.0).abs() < 1e-6);
}

#[test]
fn test_linear_batch() {
    let w = DynTensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        &[4, 3],
        &Device::Cpu,
    )
    .unwrap();
    let linear = Linear::new(w, None).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap();
    let y = linear.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-6);
    assert!((vals[3] - 6.0).abs() < 1e-6);
    assert!((vals[4] - 4.0).abs() < 1e-6);
    assert!((vals[7] - 15.0).abs() < 1e-6);
}

#[test]
fn test_linear_accessors() {
    let w = DynTensor::new(&[1.0, 2.0], &[1, 2], &Device::Cpu).unwrap();
    let b = DynTensor::new(&[3.0], &[1], &Device::Cpu).unwrap();
    let linear = Linear::new(w, Some(b)).unwrap();
    assert_eq!(linear.weight().dims(), &[1, 2]);
    assert!(linear.bias().is_some());
    assert_eq!(linear.bias().unwrap().dims(), &[1]);
}

// -- Linear rank validation --------------------------------------------------

#[test]
fn test_linear_rejects_1d_weight() {
    let w = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let err = Linear::new(w, None).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { expected: 2, .. }),
        "Expected rank error, got: {err}"
    );
}

#[test]
fn test_linear_rejects_3d_weight() {
    let w = DynTensor::new(&[1.0; 24], &[2, 3, 4], &Device::Cpu).unwrap();
    let err = Linear::new(w, None).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { expected: 2, .. }),
        "Expected rank error, got: {err}"
    );
}

#[test]
fn test_linear_1x1_weight() {
    // Degenerate case: 1x1 weight where transpose is a no-op.
    let w = DynTensor::new(&[3.0], &[1, 1], &Device::Cpu).unwrap();
    let linear = Linear::new(w, None).unwrap();
    let x = DynTensor::new(&[2.0], &[1, 1], &Device::Cpu).unwrap();
    let y = linear.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        (vals[0] - 6.0).abs() < 1e-6,
        "1x1 Linear: 2*3=6, got {}",
        vals[0]
    );
}

#[test]
fn test_linear_out_features_in_features() {
    let w = DynTensor::new(&[1.0; 12], &[3, 4], &Device::Cpu).unwrap();
    let linear = Linear::new(w, None).unwrap();
    assert_eq!(linear.out_features(), 3);
    assert_eq!(linear.in_features(), 4);
}

// -- check_output_finite (#1320) -----------------------------------------------

#[test]
fn test_check_output_finite_cpu_passes_for_valid_data() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    check_output_finite(&x, "test").unwrap();
}

#[test]
fn test_check_output_finite_cpu_detects_nan() {
    let x = DynTensor::new(&[1.0, f32::NAN, 3.0], &[3], &Device::Cpu).unwrap();
    let err = check_output_finite(&x, "test_layer").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("test_layer"),
        "error should name the layer: {msg}"
    );
    assert!(msg.contains("1"), "error should report count: {msg}");
}

#[test]
fn test_check_output_finite_cpu_detects_inf() {
    let x = DynTensor::new(&[f32::INFINITY, f32::NEG_INFINITY], &[2], &Device::Cpu).unwrap();
    let err = check_output_finite(&x, "inf_layer").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("inf_layer"),
        "error should name the layer: {msg}"
    );
    assert!(msg.contains("2"), "error should report count=2: {msg}");
}

#[test]
fn test_check_output_finite_cpu_mixed_nan_inf() {
    let x = DynTensor::new(
        &[1.0, f32::NAN, f32::INFINITY, 4.0, f32::NEG_INFINITY],
        &[5],
        &Device::Cpu,
    )
    .unwrap();
    let err = check_output_finite(&x, "mixed").unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("3"), "expected 3 non-finite, got: {msg}");
}

// -- NaN / Inf propagation ---------------------------------------------------
// IEEE 754 behavior: NaN propagation through arithmetic layers (Linear, LayerNorm,
// GroupNorm, BatchNorm) is mathematically correct -- these are element-wise/reduction
// ops where NaN in -> NaN out is the expected IEEE 754 semantic.
//
// Defense-in-depth NaN checks belong at model forward boundaries (e.g., SileroVad::forward,
// HTDemucs::forward) per design doc #929/#941, not at individual layer level.

#[test]
fn test_nan_propagation() {
    let w = DynTensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let linear = Linear::new(w, None).unwrap();
    let x = DynTensor::new(&[f32::NAN, 1.0], &[1, 2], &Device::Cpu).unwrap();
    let y = linear.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(vals[0].is_nan(), "NaN should propagate through Linear");
}

#[test]
fn test_inf_propagation() {
    let w = DynTensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let linear = Linear::new(w, None).unwrap();
    let x = DynTensor::new(&[f32::INFINITY, 1.0], &[1, 2], &Device::Cpu).unwrap();
    let y = linear.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(vals[0].is_infinite(), "Inf should propagate through Linear");
}

#[test]
fn test_linear_4d_input() {
    // Linear with 4D input: [B, H, S, in] × [in, out]^T → [B, H, S, out]
    // weight: [out=2, in=3], so weight^T = [3, 2]
    let w = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap();
    let linear = Linear::new(w, None).unwrap();
    // input: [1, 2, 1, 3] — batch=1, heads=2, seq=1, in_features=3
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 1, 3], &Device::Cpu).unwrap();
    let y = linear.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 1, 2]);
    let flat = y.to_flat_vec::<f32>().unwrap();
    // Head 0: [1,2,3] × [[1,4],[2,5],[3,6]] = [14, 32]
    assert_eq!(&flat[0..2], &[14.0, 32.0]);
    // Head 1: [4,5,6] × [[1,4],[2,5],[3,6]] = [32, 77]
    assert_eq!(&flat[2..4], &[32.0, 77.0]);
}

// -- Constructor shape validation tests (#1316) --------------------------------

#[test]
fn test_linear_rejects_bias_shape_mismatch() {
    // weight [3, 2] → out_features=3, bias must be [3]
    let w = DynTensor::ones(&[3, 2], DType::F32, &Device::Cpu).unwrap();
    let bad_bias = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let err = Linear::new(w, Some(bad_bias)).unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "expected shape mismatch error, got: {err}"
    );
}

#[test]
fn test_linear_accepts_correct_bias() {
    let w = DynTensor::ones(&[3, 2], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::ones(&[3], DType::F32, &Device::Cpu).unwrap();
    assert!(Linear::new(w, Some(bias)).is_ok());
}

#[test]
fn test_conv1d_rejects_bias_shape_mismatch() {
    // weight [4, 2, 3] → out_channels=4, bias must be [4]
    let w = DynTensor::ones(&[4, 2, 3], DType::F32, &Device::Cpu).unwrap();
    let bad_bias = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let err = Conv1d::new(w, Some(bad_bias), Conv1dConfig::default()).unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "expected shape mismatch error, got: {err}"
    );
}

#[test]
fn test_conv2d_rejects_bias_shape_mismatch() {
    // weight [8, 3, 3, 3] → out_channels=8, bias must be [8]
    let w = DynTensor::ones(&[8, 3, 3, 3], DType::F32, &Device::Cpu).unwrap();
    let bad_bias = DynTensor::ones(&[3], DType::F32, &Device::Cpu).unwrap();
    let err = Conv2d::new(w, Some(bad_bias), Conv2dConfig::default()).unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "expected shape mismatch error, got: {err}"
    );
}

#[test]
fn test_layer_norm_rejects_weight_bias_shape_mismatch() {
    let weight = DynTensor::ones(&[10], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::ones(&[5], DType::F32, &Device::Cpu).unwrap();
    let err = LayerNorm::new(weight, bias, 1e-5).unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "expected shape mismatch error, got: {err}"
    );
}

#[test]
fn test_rms_norm_rejects_non_1d_weight() {
    let weight = DynTensor::ones(&[2, 3], DType::F32, &Device::Cpu).unwrap();
    let err = RmsNorm::new(weight, 1e-5).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { expected: 1, .. }),
        "expected rank mismatch error, got: {err}"
    );
}

#[test]
fn test_group_norm_rejects_weight_shape_mismatch() {
    // num_channels=4, but weight is [3] (wrong)
    let weight = DynTensor::ones(&[3], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let err = GroupNorm::new(2, 4, weight, bias, 1e-5).unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "expected shape mismatch error, got: {err}"
    );
}

#[test]
fn test_group_norm_rejects_bias_shape_mismatch() {
    // num_channels=4, weight is [4] (correct), but bias is [2] (wrong)
    let weight = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let err = GroupNorm::new(2, 4, weight, bias, 1e-5).unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "expected shape mismatch error, got: {err}"
    );
}

// -- NanCheckPolicy (#1915) ---------------------------------------------------

#[test]
fn test_nan_check_policy_default_is_always() {
    assert_eq!(nan_check_policy(), NanCheckPolicy::Always);
}

#[test]
fn test_nan_check_policy_always_catches_nan() {
    let x = DynTensor::new(&[1.0, f32::NAN], &[2], &Device::Cpu).unwrap();
    with_nan_check_policy(NanCheckPolicy::Always, || {
        let err = check_output_finite(&x, "test").unwrap_err();
        assert!(matches!(err, TensorError::NonFiniteData { .. }));
    });
}

#[test]
fn test_nan_check_policy_skip_allows_nan_through() {
    let x = DynTensor::new(&[1.0, f32::NAN, f32::INFINITY], &[3], &Device::Cpu).unwrap();
    with_nan_check_policy(NanCheckPolicy::Skip, || {
        // Should return Ok despite NaN and Inf in the tensor.
        check_output_finite(&x, "skipped_layer").unwrap();
    });
}

#[test]
fn test_nan_check_policy_scope_restores_prior() {
    // Start with default (Always).
    assert_eq!(nan_check_policy(), NanCheckPolicy::Always);
    with_nan_check_policy(NanCheckPolicy::Skip, || {
        assert_eq!(nan_check_policy(), NanCheckPolicy::Skip);
    });
    // Restored to Always after scope exits.
    assert_eq!(nan_check_policy(), NanCheckPolicy::Always);
}

#[test]
fn test_nan_check_policy_nested_scopes() {
    assert_eq!(nan_check_policy(), NanCheckPolicy::Always);
    with_nan_check_policy(NanCheckPolicy::Skip, || {
        assert_eq!(nan_check_policy(), NanCheckPolicy::Skip);
        // Inner scope overrides to Always.
        with_nan_check_policy(NanCheckPolicy::Always, || {
            assert_eq!(nan_check_policy(), NanCheckPolicy::Always);
            // NaN check catches in inner scope.
            let x = DynTensor::new(&[f32::NAN], &[1], &Device::Cpu).unwrap();
            assert!(check_output_finite(&x, "inner").is_err());
        });
        // Restored to Skip in outer scope.
        assert_eq!(nan_check_policy(), NanCheckPolicy::Skip);
        let x = DynTensor::new(&[f32::NAN], &[1], &Device::Cpu).unwrap();
        assert!(check_output_finite(&x, "outer").is_ok());
    });
    assert_eq!(nan_check_policy(), NanCheckPolicy::Always);
}

#[test]
fn test_nan_check_policy_skip_valid_data_still_ok() {
    // Skip mode should return Ok for valid data too.
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    with_nan_check_policy(NanCheckPolicy::Skip, || {
        check_output_finite(&x, "valid").unwrap();
    });
}

#[test]
fn test_nan_check_policy_restored_after_panic() {
    // RAII guard must restore the prior policy even when `f` panics.
    assert_eq!(nan_check_policy(), NanCheckPolicy::Always);
    let result = std::panic::catch_unwind(|| {
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            assert_eq!(nan_check_policy(), NanCheckPolicy::Skip);
            panic!("intentional panic inside NaN-skip scope");
        });
    });
    assert!(result.is_err(), "closure should have panicked");
    // Policy must be restored to Always despite the panic.
    assert_eq!(
        nan_check_policy(),
        NanCheckPolicy::Always,
        "RAII guard must restore prior policy on panic unwind"
    );
}
