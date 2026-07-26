#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for generic `to_vec1/2/3` and `to_scalar` via [`WithDType`].

use crate::dyn_tensor::test_helpers::cpu;
use crate::dyn_tensor::DynTensor;

// -- to_vec1 generic ----------------------------------------------------------

#[test]
fn test_to_vec1_f32_matches_typed() {
    let t = DynTensor::new(&[1.0_f32, 2.0, 3.0], &[3], &cpu()).unwrap();
    let generic: Vec<f32> = t.to_vec1().unwrap();
    let typed = t.to_vec1::<f32>().unwrap();
    assert_eq!(generic, typed);
}

#[test]
fn test_to_vec1_u32() {
    let t = DynTensor::from_vec_u32(vec![10, 20, 30], &[3], &cpu()).unwrap();
    let v: Vec<u32> = t.to_vec1().unwrap();
    assert_eq!(v, vec![10, 20, 30]);
}

#[test]
fn test_to_vec1_u8() {
    let t = DynTensor::from_vec_u8(vec![0, 1, 1, 0], &[4], &cpu()).unwrap();
    let v: Vec<u8> = t.to_vec1().unwrap();
    assert_eq!(v, vec![0, 1, 1, 0]);
}

#[test]
fn test_to_vec1_rank_mismatch() {
    let t = DynTensor::new(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let err = t.to_vec1::<f32>().unwrap_err();
    assert!(
        format!("{err}").contains("rank") || format!("{err:?}").contains("RankMismatch"),
        "expected rank mismatch error, got: {err:?}"
    );
}

#[test]
fn test_to_vec1_dtype_mismatch() {
    let t = DynTensor::new(&[1.0_f32, 2.0], &[2], &cpu()).unwrap();
    let err = t.to_vec1::<u32>().unwrap_err();
    assert!(
        format!("{err:?}").contains("DTypeMismatch"),
        "expected dtype mismatch error, got: {err:?}"
    );
}

// -- to_vec2 generic ----------------------------------------------------------

#[test]
fn test_to_vec2_f32_matches_typed() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let generic: Vec<Vec<f32>> = t.to_vec2().unwrap();
    let typed: Vec<Vec<f32>> = t.to_vec2().unwrap();
    assert_eq!(generic, typed);
}

#[test]
fn test_to_vec2_u32() {
    let t = DynTensor::from_vec_u32(vec![1, 2, 3, 4], &[2, 2], &cpu()).unwrap();
    let v: Vec<Vec<u32>> = t.to_vec2().unwrap();
    assert_eq!(v, vec![vec![1, 2], vec![3, 4]]);
}

#[test]
fn test_to_vec2_rank_mismatch() {
    let t = DynTensor::new(&[1.0_f32, 2.0, 3.0], &[3], &cpu()).unwrap();
    let err = t.to_vec2::<f32>().unwrap_err();
    assert!(
        format!("{err:?}").contains("RankMismatch"),
        "expected RankMismatch error, got: {err:?}"
    );
}

// -- to_vec3 generic ----------------------------------------------------------

#[test]
fn test_to_vec3_f32_matches_typed() {
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &cpu()).unwrap();
    let generic: Vec<Vec<Vec<f32>>> = t.to_vec3().unwrap();
    let typed: Vec<Vec<Vec<f32>>> = t.to_vec3().unwrap();
    assert_eq!(generic, typed);
}

#[test]
fn test_to_vec3_rank_mismatch() {
    let t = DynTensor::new(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let err = t.to_vec3::<f32>().unwrap_err();
    assert!(
        format!("{err:?}").contains("RankMismatch"),
        "expected RankMismatch error, got: {err:?}"
    );
}

// -- to_scalar generic --------------------------------------------------------

#[test]
fn test_to_scalar_f32_matches_typed() {
    let t = DynTensor::new(&[42.0_f32], &[1], &cpu()).unwrap();
    let generic: f32 = t.to_scalar().unwrap();
    let typed = t.to_scalar::<f32>().unwrap();
    assert_eq!(generic, typed);
}

#[test]
fn test_to_scalar_u32() {
    let t = DynTensor::from_vec_u32(vec![7], &[1], &cpu()).unwrap();
    let v: u32 = t.to_scalar().unwrap();
    assert_eq!(v, 7);
}

#[test]
fn test_to_scalar_not_single_element() {
    let t = DynTensor::new(&[1.0_f32, 2.0], &[2], &cpu()).unwrap();
    let err = t.to_scalar::<f32>().unwrap_err();
    assert!(
        format!("{err}").contains("1 element"),
        "expected single-element error, got: {err:?}"
    );
}

// -- 0-D scalar tensor (candle compat) ----------------------------------------

#[test]
fn test_to_scalar_0d_tensor() {
    let t = DynTensor::new(&[99.5_f32], &[], &cpu()).unwrap();
    let v: f32 = t.to_scalar().unwrap();
    assert_eq!(v, 99.5);
}

// -- I64 WithDType tests (#1207) ----------------------------------------------

#[test]
fn test_to_vec1_i64() {
    let t = DynTensor::from_vec_i64(vec![10, -20, 30], &[3], &cpu()).unwrap();
    let v: Vec<i64> = t.to_vec1().unwrap();
    assert_eq!(v, vec![10, -20, 30]);
}

#[test]
fn test_to_scalar_i64() {
    let t = DynTensor::from_vec_i64(vec![42], &[1], &cpu()).unwrap();
    let v: i64 = t.to_scalar().unwrap();
    assert_eq!(v, 42);
}

#[test]
fn test_to_vec2_i64() {
    let t = DynTensor::from_vec_i64(vec![1, 2, 3, 4], &[2, 2], &cpu()).unwrap();
    let v: Vec<Vec<i64>> = t.to_vec2().unwrap();
    assert_eq!(v, vec![vec![1, 2], vec![3, 4]]);
}

#[test]
fn test_i64_dtype_mismatch() {
    // An f32 tensor should reject i64 extraction
    let t = DynTensor::new(&[1.0_f32, 2.0], &[2], &cpu()).unwrap();
    let err = t.to_vec1::<i64>().unwrap_err();
    assert!(
        format!("{err:?}").contains("DTypeMismatch"),
        "expected dtype mismatch error, got: {err:?}"
    );
}

// -- f16 WithDType tests (#1749 AC1) ------------------------------------------

#[test]
fn test_to_vec1_f16() {
    use half::f16;
    let data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
    let t = DynTensor::from_vec_f16(data.clone(), &[3], &cpu()).unwrap();
    let v: Vec<f16> = t.to_vec1().unwrap();
    assert_eq!(v, data);
}

#[test]
fn test_to_scalar_f16() {
    use half::f16;
    let t = DynTensor::from_vec_f16(vec![f16::from_f32(42.0)], &[1], &cpu()).unwrap();
    let v: f16 = t.to_scalar().unwrap();
    assert_eq!(v, f16::from_f32(42.0));
}

// -- bf16 WithDType tests (#1749 AC1) -----------------------------------------

#[test]
fn test_to_vec1_bf16() {
    use half::bf16;
    let data = vec![
        bf16::from_f32(1.0),
        bf16::from_f32(2.5),
        bf16::from_f32(3.0),
    ];
    let t = DynTensor::from_vec_bf16(data.clone(), &[3], &cpu()).unwrap();
    let v: Vec<bf16> = t.to_vec1().unwrap();
    assert_eq!(v, data);
}

#[test]
fn test_to_vec2_bf16() {
    use half::bf16;
    let data = vec![
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
    ];
    let t = DynTensor::from_vec_bf16(data, &[2, 2], &cpu()).unwrap();
    let v: Vec<Vec<bf16>> = t.to_vec2().unwrap();
    assert_eq!(
        v,
        vec![
            vec![bf16::from_f32(1.0), bf16::from_f32(2.0)],
            vec![bf16::from_f32(3.0), bf16::from_f32(4.0)],
        ]
    );
}

#[test]
fn test_to_scalar_bf16() {
    use half::bf16;
    let t = DynTensor::from_vec_bf16(vec![bf16::from_f32(7.5)], &[1], &cpu()).unwrap();
    let v: bf16 = t.to_scalar().unwrap();
    assert_eq!(v, bf16::from_f32(7.5));
}

// -- from_vec_f16/bf16 tests (#1749 AC2) --------------------------------------

#[test]
fn test_from_vec_f16_shape_mismatch() {
    use half::f16;
    let data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
    let err = DynTensor::from_vec_f16(data, &[3], &cpu()).unwrap_err();
    assert!(
        format!("{err:?}").contains("DataLengthMismatch"),
        "expected DataLengthMismatch, got: {err:?}"
    );
}

#[test]
fn test_from_vec_bf16_shape_mismatch() {
    use half::bf16;
    let data = vec![bf16::from_f32(1.0)];
    let err = DynTensor::from_vec_bf16(data, &[2, 2], &cpu()).unwrap_err();
    assert!(
        format!("{err:?}").contains("DataLengthMismatch"),
        "expected DataLengthMismatch, got: {err:?}"
    );
}

#[test]
fn test_from_vec_f16_preserves_dtype() {
    use crate::DType;
    use half::f16;
    let t = DynTensor::from_vec_f16(vec![f16::from_f32(1.0), f16::from_f32(2.0)], &[2], &cpu())
        .unwrap();
    assert_eq!(t.dtype(), DType::F16);
}

#[test]
fn test_from_vec_bf16_preserves_dtype() {
    use crate::DType;
    use half::bf16;
    let t = DynTensor::from_vec_bf16(vec![bf16::from_f32(1.0), bf16::from_f32(2.0)], &[2], &cpu())
        .unwrap();
    assert_eq!(t.dtype(), DType::BF16);
}

// -- arange_i64 tests (#1749 AC3) ---------------------------------------------

#[test]
fn test_arange_i64_basic() {
    let t = DynTensor::arange_i64(0, 5, &cpu()).unwrap();
    let v: Vec<i64> = t.to_vec1().unwrap();
    assert_eq!(v, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_arange_i64_negative_range() {
    let t = DynTensor::arange_i64(-3, 2, &cpu()).unwrap();
    let v: Vec<i64> = t.to_vec1().unwrap();
    assert_eq!(v, vec![-3, -2, -1, 0, 1]);
}

#[test]
fn test_arange_i64_empty() {
    let t = DynTensor::arange_i64(5, 3, &cpu()).unwrap();
    assert_eq!(t.dims(), &[0]);
}

#[test]
fn test_arange_i64_single_element() {
    let t = DynTensor::arange_i64(10, 11, &cpu()).unwrap();
    let v: Vec<i64> = t.to_vec1().unwrap();
    assert_eq!(v, vec![10]);
}

// -- from_typed_vec generic constructor (#2471 Finding 2) ---------------------

#[test]
fn test_from_typed_vec_f32() {
    let t = DynTensor::from_typed_vec::<f32>(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert_eq!(t.dtype(), crate::DType::F32);
    let v: Vec<f32> = t.to_vec1().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_from_typed_vec_u32() {
    let t = DynTensor::from_typed_vec::<u32>(vec![10, 20, 30], &[3], &cpu()).unwrap();
    assert_eq!(t.dtype(), crate::DType::U32);
    let v: Vec<u32> = t.to_vec1().unwrap();
    assert_eq!(v, vec![10, 20, 30]);
}

#[test]
fn test_from_typed_vec_i64() {
    let t = DynTensor::from_typed_vec::<i64>(vec![-5, 0, 5], &[3], &cpu()).unwrap();
    assert_eq!(t.dtype(), crate::DType::I64);
    let v: Vec<i64> = t.to_vec1().unwrap();
    assert_eq!(v, vec![-5, 0, 5]);
}

#[test]
fn test_from_typed_vec_u8() {
    let t = DynTensor::from_typed_vec::<u8>(vec![0, 128, 255], &[3], &cpu()).unwrap();
    assert_eq!(t.dtype(), crate::DType::U8);
    let v: Vec<u8> = t.to_vec1().unwrap();
    assert_eq!(v, vec![0, 128, 255]);
}

#[test]
fn test_from_typed_vec_f16() {
    use half::f16;
    let data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
    let t = DynTensor::from_typed_vec::<f16>(data.clone(), &[2], &cpu()).unwrap();
    assert_eq!(t.dtype(), crate::DType::F16);
    let v: Vec<f16> = t.to_vec1().unwrap();
    assert_eq!(v, data);
}

#[test]
fn test_from_typed_vec_bf16() {
    use half::bf16;
    let data = vec![bf16::from_f32(3.0), bf16::from_f32(4.0)];
    let t = DynTensor::from_typed_vec::<bf16>(data.clone(), &[2], &cpu()).unwrap();
    assert_eq!(t.dtype(), crate::DType::BF16);
    let v: Vec<bf16> = t.to_vec1().unwrap();
    assert_eq!(v, data);
}

#[test]
fn test_from_typed_vec_2d() {
    let t = DynTensor::from_typed_vec::<f32>(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 2]);
    let v: Vec<Vec<f32>> = t.to_vec2().unwrap();
    assert_eq!(v, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
}

#[test]
fn test_from_typed_vec_shape_mismatch() {
    let err = DynTensor::from_typed_vec::<f32>(vec![1.0, 2.0], &[3], &cpu()).unwrap_err();
    assert!(
        format!("{err:?}").contains("DataLengthMismatch"),
        "expected DataLengthMismatch, got: {err:?}"
    );
}
