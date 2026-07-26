#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended DynTensor tests: arange edge cases, arange_step, repeat, dtype variants.
//! Extracted from tests.rs to stay under the 500-line limit.

use super::*;
use crate::dyn_tensor::test_helpers::cpu;

// -- arange edge cases (P10 strategic audit) ----------------------------------

#[test]
fn test_arange_negative_range_returns_empty() {
    let t = DynTensor::arange(5.0, 0.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[0]);
}

#[test]
fn test_arange_equal_bounds_returns_empty() {
    let t = DynTensor::arange(3.0, 3.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[0]);
}

#[test]
fn test_arange_nan_start_returns_error() {
    let err = DynTensor::arange(f64::NAN, 5.0, &cpu()).unwrap_err();
    assert!(err.to_string().contains("finite"), "got: {err}");
}

#[test]
fn test_arange_infinity_end_returns_error() {
    let err = DynTensor::arange(0.0, f64::INFINITY, &cpu()).unwrap_err();
    assert!(err.to_string().contains("finite"), "got: {err}");
}

#[test]
fn test_arange_step_finite_inputs_overflow_returns_error() {
    // All three inputs are finite, but (1e308 - 0) / 1e-308 = Inf.
    // Without the length guard, `Inf as usize` saturates to usize::MAX,
    // attempting a multi-exabyte allocation.
    let err = DynTensor::arange_step(0.0, 1e308, 1e-308, &cpu()).unwrap_err();
    assert!(
        err.to_string().contains("exceeds maximum"),
        "expected length overflow error, got: {err}"
    );
}

#[test]
fn test_arange_step_subtraction_overflow_returns_error() {
    // start=-MAX, end=MAX: (end - start) overflows to Infinity.
    let err = DynTensor::arange_step(-f64::MAX, f64::MAX, 1.0, &cpu()).unwrap_err();
    assert!(
        err.to_string().contains("exceeds maximum"),
        "expected length overflow error, got: {err}"
    );
}

#[test]
fn test_arange_step_element_overflow_f32_returns_error() {
    // start=3.4e38, step=1e30: individual elements overflow f32 range.
    // Without the per-element guard, elements silently become f32::INFINITY.
    let err = DynTensor::arange_step(3.4e38, 3.5e38, 1e30, &cpu()).unwrap_err();
    assert!(
        err.to_string().contains("overflow"),
        "expected f32 overflow error, got: {err}"
    );
}

#[test]
fn test_arange_step_negative_element_overflow_f32_returns_error() {
    // Negative direction: elements underflow past -f32::MAX.
    let err = DynTensor::arange_step(-3.4e38, -3.5e38, -1e30, &cpu()).unwrap_err();
    assert!(
        err.to_string().contains("overflow"),
        "expected f32 overflow error, got: {err}"
    );
}

// -- NonFiniteData error variant tests ----------------------------------------

#[test]
fn test_non_finite_data_error_format() {
    let err = TensorError::NonFiniteData {
        name: "encoder.weight".to_string(),
        count: 3,
    };
    assert!(err.to_string().contains("encoder.weight"));
    assert!(err.to_string().contains("3"));
}

#[test]
fn test_new_allows_nan_intentionally() {
    // DynTensor::new() does NOT validate finiteness (by design — test code needs NaN).
    // Finiteness validation happens at weight-loading boundaries (SafeTensorsBackend).
    let t = DynTensor::new(&[1.0, f32::NAN, 3.0], &[3], &cpu()).unwrap();
    assert_eq!(t.dims(), &[3]);
}

// -- arange_step tests --------------------------------------------------------

#[test]
fn test_arange_step_basic() {
    let t = DynTensor::arange_step(0.0, 10.0, 2.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[5]);
    assert_eq!(
        t.to_flat_vec::<f32>().unwrap(),
        vec![0.0, 2.0, 4.0, 6.0, 8.0]
    );
}

#[test]
fn test_arange_step_fractional() {
    let t = DynTensor::arange_step(0.0, 1.0, 0.25, &cpu()).unwrap();
    assert_eq!(t.dims(), &[4]);
    let vals = t.to_flat_vec::<f32>().unwrap();
    for (a, b) in vals.iter().zip([0.0, 0.25, 0.5, 0.75].iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_arange_step_negative() {
    let t = DynTensor::arange_step(5.0, 0.0, -1.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[5]);
    assert_eq!(
        t.to_flat_vec::<f32>().unwrap(),
        vec![5.0, 4.0, 3.0, 2.0, 1.0]
    );
}

#[test]
fn test_arange_step_zero_returns_error() {
    assert!(DynTensor::arange_step(0.0, 10.0, 0.0, &cpu()).is_err());
}

#[test]
fn test_arange_step_wrong_direction_returns_empty() {
    let t = DynTensor::arange_step(0.0, 10.0, -1.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[0]);
}

#[test]
fn test_arange_delegates_to_arange_step() {
    // Verify arange(a,b) == arange_step(a,b,1.0)
    let a = DynTensor::arange(0.0, 5.0, &cpu()).unwrap();
    let b = DynTensor::arange_step(0.0, 5.0, 1.0, &cpu()).unwrap();
    assert_eq!(
        a.to_flat_vec::<f32>().unwrap(),
        b.to_flat_vec::<f32>().unwrap()
    );
}

// -- repeat (tile) tests ------------------------------------------------------

#[test]
fn test_repeat_1d() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let r = t.repeat([3]).unwrap();
    assert_eq!(r.dims(), &[9]);
    assert_eq!(
        r.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn test_repeat_2d() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let r = t.repeat([2, 3]).unwrap();
    assert_eq!(r.dims(), &[4, 6]);
    let flat = r.to_flat_vec::<f32>().unwrap();
    // Row 0 of original [1,2] repeated 3x → [1,2,1,2,1,2]
    assert_eq!(&flat[0..6], &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    // Row 1 of original [3,4] repeated 3x → [3,4,3,4,3,4]
    assert_eq!(&flat[6..12], &[3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);
}

#[test]
fn test_repeat_noop() {
    let t = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    let r = t.repeat([1]).unwrap();
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn test_repeat_rank_mismatch() {
    let t = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(t.repeat([2, 2]).is_err());
}

// -- zeros/ones with non-F32 dtypes (#1199, updated for #1646 native bf16/f16) -

#[test]
fn test_zeros_bf16_succeeds() {
    let t = DynTensor::zeros(&[3], DType::BF16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16); // native bf16 storage (#1646)
    let flat = t.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|&v| v == 0.0));
}

#[test]
fn test_ones_f16_succeeds() {
    let t = DynTensor::ones(&[3], DType::F16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F16); // native f16 storage (#1646)
    let flat = t.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|&v| v == 1.0));
}

#[test]
fn test_zeros_like_bf16_labeled() {
    let t = DynTensor::full(&[2, 3], 1.0, DType::BF16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16); // native bf16 storage (#1646)
    let z = t.zeros_like().unwrap();
    assert_eq!(z.dims(), &[2, 3]);
    assert_eq!(z.dtype(), DType::BF16);
    let flat = z.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|&v| v == 0.0));
}

// -- AC2 regression: from_vec_i64 on CPU device produces CPU tensor --

#[test]
fn test_from_vec_i64_cpu_device() {
    let t = DynTensor::from_vec_i64(vec![10, 20, 30], &[3], &cpu()).unwrap();
    assert_eq!(t.device(), Device::Cpu);
    assert_eq!(t.dims(), &[3]);
    assert_eq!(t.as_cpu_i64().unwrap().as_slice().unwrap(), &[10, 20, 30]);
}

// -- AC3 regression: flatten_all uses checked_numel --

#[test]
fn test_flatten_all_normal() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let flat = t.flatten_all().unwrap();
    assert_eq!(flat.dims(), &[4]);
}
