#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DynTensor core type and CPU shape operations.

use super::*;
use crate::dyn_tensor::test_helpers::cpu;

// -- Constructor tests --------------------------------------------------------

#[test]
fn test_new_from_slice() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 2]);
    assert_eq!(t.rank(), 2);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.device(), Device::Cpu);
    assert_eq!(t.numel(), 4);
}

#[test]
fn test_new_data_length_mismatch() {
    let result = DynTensor::new(&[1.0, 2.0, 3.0], &[2, 2], &cpu());
    assert!(result.is_err());
}

#[test]
fn test_zeros() {
    let t = DynTensor::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[3, 4]);
    assert_eq!(t.numel(), 12);
    let flat = t.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|&v| v == 0.0));
}

#[test]
fn test_ones() {
    let t = DynTensor::ones(&[2, 3], DType::F32, &cpu()).unwrap();
    let flat = t.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|&v| v == 1.0));
}

#[test]
fn test_from_vec() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert_eq!(t.dims(), &[3]);
    let v = t.to_vec1::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_full() {
    let t = DynTensor::full(&[2, 2], 2.75, DType::F32, &cpu()).unwrap();
    let flat = t.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|&v| (v - 2.75_f32).abs() < 1e-6));
}

// -- full() integer dtype tests (#1399 AC1) -----------------------------------

#[test]
fn test_full_u32() {
    let t = DynTensor::full(&[3], 42.0, DType::U32, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U32);
    assert_eq!(t.as_cpu_u32().unwrap().as_slice().unwrap(), &[42, 42, 42]);
}

#[test]
fn test_full_u8() {
    let t = DynTensor::full(&[2], 255.0, DType::U8, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U8);
    assert_eq!(t.as_cpu_u8().unwrap().as_slice().unwrap(), &[255, 255]);
}

#[test]
fn test_full_i64() {
    let t = DynTensor::full(&[2], -100.0, DType::I64, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::I64);
    assert_eq!(t.as_cpu_i64().unwrap().as_slice().unwrap(), &[-100, -100]);
}

#[test]
fn test_full_u32_negative_rejects() {
    let err = DynTensor::full(&[1], -1.0, DType::U32, &cpu()).unwrap_err();
    assert!(
        err.to_string().contains("cannot be represented"),
        "got: {err}"
    );
}

#[test]
fn test_full_u8_overflow_rejects() {
    let err = DynTensor::full(&[1], 256.0, DType::U8, &cpu()).unwrap_err();
    assert!(
        err.to_string().contains("cannot be represented"),
        "got: {err}"
    );
}

#[test]
fn test_full_i64_fractional_rejects() {
    let err = DynTensor::full(&[1], 1.5, DType::I64, &cpu()).unwrap_err();
    assert!(
        err.to_string().contains("cannot be represented"),
        "got: {err}"
    );
}

#[test]
fn test_full_nan_u32_returns_error() {
    let err = DynTensor::full(&[2], f64::NAN, DType::U32, &cpu()).unwrap_err();
    assert!(
        err.to_string().contains("cannot be represented"),
        "got: {err}"
    );
}

#[test]
fn test_full_inf_u8_returns_error() {
    let err = DynTensor::full(&[2], f64::INFINITY, DType::U8, &cpu()).unwrap_err();
    assert!(
        err.to_string().contains("cannot be represented"),
        "got: {err}"
    );
}

#[test]
fn test_full_nan_i64_returns_error() {
    let err = DynTensor::full(&[2], f64::NAN, DType::I64, &cpu()).unwrap_err();
    assert!(
        err.to_string().contains("cannot be represented"),
        "got: {err}"
    );
}

#[test]
fn test_full_neg_inf_i64_returns_error() {
    let err = DynTensor::full(&[2], f64::NEG_INFINITY, DType::I64, &cpu()).unwrap_err();
    assert!(
        err.to_string().contains("cannot be represented"),
        "got: {err}"
    );
}

#[test]
fn test_arange() {
    let t = DynTensor::arange(0.0, 5.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[5]);
    let v = t.to_vec1::<f32>().unwrap();
    assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

// -- Shape query tests --------------------------------------------------------

#[test]
fn test_dims1() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert_eq!(t.dims1().unwrap(), 3);
}

#[test]
fn test_dims1_wrong_rank() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    assert!(t.dims1().is_err());
}

#[test]
fn test_dims2() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims2().unwrap(), (2, 3));
}

#[test]
fn test_dims3() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims3().unwrap(), (2, 3, 4));
}

#[test]
fn test_dims4() {
    let t = DynTensor::zeros(&[2, 3, 4, 5], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims4().unwrap(), (2, 3, 4, 5));
}

#[test]
fn test_dim() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dim(0).unwrap(), 2);
    assert_eq!(t.dim(1).unwrap(), 3);
    assert_eq!(t.dim(2).unwrap(), 4);
    assert!(t.dim(3).is_err());
}

// -- Data extraction tests ----------------------------------------------------

#[test]
fn test_to_scalar() {
    let t = DynTensor::full(&[], 42.0, DType::F32, &cpu()).unwrap();
    assert_eq!(t.to_scalar::<f32>().unwrap(), 42.0);
}

#[test]
fn test_to_scalar_non_scalar_fails() {
    let t = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(t.to_scalar::<f32>().is_err());
}

#[test]
fn test_flatten_all() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let flat = t.flatten_all().unwrap();
    assert_eq!(flat.dims(), &[6]);
    assert_eq!(
        flat.to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

// -- Shape manipulation tests -------------------------------------------------

#[test]
fn test_reshape() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], &cpu()).unwrap();
    let r = t.reshape([2, 3]).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    assert_eq!(r.numel(), 6);
}

#[test]
fn test_reshape_numel_mismatch() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], &cpu()).unwrap();
    assert!(t.reshape([3, 2]).is_err());
}

#[test]
fn test_narrow() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5], &cpu()).unwrap();
    let n = t.narrow(0, 1, 3).unwrap();
    assert_eq!(n.dims(), &[3]);
    assert_eq!(n.to_vec1::<f32>().unwrap(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_narrow_out_of_bounds() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert!(t.narrow(0, 2, 3).is_err());
}

#[test]
fn test_narrow_usize_overflow() {
    // start + len wraps usize::MAX — checked_add must catch this (#981).
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let err = t.narrow(0, usize::MAX - 1, 3).unwrap_err();
    assert!(err.to_string().contains("overflows"), "got: {err}");
}

#[test]
fn test_unsqueeze() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let u = t.unsqueeze(0).unwrap();
    assert_eq!(u.dims(), &[1, 3]);
    let u2 = t.unsqueeze(1).unwrap();
    assert_eq!(u2.dims(), &[3, 1]);
}

#[test]
fn test_squeeze() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let s = t.squeeze(0).unwrap();
    assert_eq!(s.dims(), &[3]);
}

#[test]
fn test_squeeze_non_unit_fails() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    assert!(t.squeeze(0).is_err());
}

#[test]
fn test_transpose() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let tr = t.transpose(0, 1).unwrap();
    assert_eq!(tr.dims(), &[3, 2]);
    // [1,2,3; 4,5,6] transposed = [1,4; 2,5; 3,6]
    let flat = tr.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_permute() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let p = t.permute([2, 0, 1]).unwrap();
    assert_eq!(p.dims(), &[4, 2, 3]);
}

#[test]
fn test_permute_invalid() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    assert!(t.permute([0, 0]).is_err()); // duplicate
    assert!(t.permute([0]).is_err()); // wrong length
}

#[test]
fn test_contiguous() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let c = t.contiguous().unwrap();
    assert_eq!(c.dims(), &[2, 3]);
}

#[test]
fn test_chunk() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5], &cpu()).unwrap();
    let chunks = t.chunk(2, 0).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].dims(), &[3]);
    assert_eq!(chunks[1].dims(), &[2]);
}

// -- Cat and Stack tests ------------------------------------------------------

#[test]
fn test_cat() {
    let a = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![3.0, 4.0, 5.0], &[3], &cpu()).unwrap();
    let c = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[5]);
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_stack() {
    let a = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![3.0, 4.0], &[2], &cpu()).unwrap();
    let s = DynTensor::stack(&[&a, &b], 0).unwrap();
    assert_eq!(s.dims(), &[2, 2]);
    let flat = s.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
}

// -- Cat/Stack with owned tensors (candle compat, #1257) ----------------------

#[test]
fn test_cat_owned_slice() {
    let a = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![3.0, 4.0, 5.0], &[3], &cpu()).unwrap();
    // candle convention: &[Tensor] (owned values in slice)
    let c = DynTensor::cat(&[a, b], 0).unwrap();
    assert_eq!(c.dims(), &[5]);
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_stack_owned_slice() {
    let a = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![3.0, 4.0], &[2], &cpu()).unwrap();
    // candle convention: &[Tensor] (owned values in slice)
    let s = DynTensor::stack(&[a, b], 0).unwrap();
    assert_eq!(s.dims(), &[2, 2]);
    let flat = s.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
}

// -- Debug impl ---------------------------------------------------------------

#[test]
fn test_debug() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let dbg = format!("{t:?}");
    assert!(dbg.contains("DynTensor"));
    assert!(dbg.contains("[2, 3]"));
}

// -- Overflow protection ------------------------------------------------------

#[test]
fn test_dimension_overflow() {
    let result = DynTensor::zeros(&[usize::MAX, 2], DType::F32, &cpu());
    assert!(result.is_err());
}

// Extended tests (arange edge cases, arange_step, repeat, dtype variants)
// extracted to tests_extended.rs to stay under the 500-line limit.
#[path = "tests_extended.rs"]
mod extended;
