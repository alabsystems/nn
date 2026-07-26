#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for candle-compatible DynTensor APIs (#1094).

use super::*;
use crate::dyn_tensor::test_helpers::cpu;
use crate::DType;

// -- D enum tests -------------------------------------------------------------

#[test]
fn test_d_minus1_resolves_to_last_dim() {
    assert_eq!(D::Minus1.resolve(3).unwrap(), 2);
    assert_eq!(D::Minus1.resolve(1).unwrap(), 0);
    assert_eq!(D::Minus1.resolve(5).unwrap(), 4);
}

#[test]
fn test_d_minus2_resolves_to_second_to_last() {
    assert_eq!(D::Minus2.resolve(3).unwrap(), 1);
    assert_eq!(D::Minus2.resolve(2).unwrap(), 0);
    assert_eq!(D::Minus2.resolve(5).unwrap(), 3);
}

#[test]
fn test_d_minus1_rank_zero_error() {
    let result = D::Minus1.resolve(0);
    assert!(result.is_err());
}

#[test]
fn test_d_minus2_rank_one_error() {
    let result = D::Minus2.resolve(1);
    assert!(result.is_err());
}

#[test]
fn test_d_minus2_rank_zero_error() {
    let result = D::Minus2.resolve(0);
    assert!(result.is_err());
}

// -- zeros_like / ones_like tests --------------------------------------------

#[test]
fn test_zeros_like_shape_and_values() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let z = t.zeros_like().unwrap();
    assert_eq!(z.dims(), &[2, 2]);
    assert_eq!(z.dtype(), DType::F32);
    let vals = z.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| v == 0.0));
}

#[test]
fn test_ones_like_shape_and_values() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let o = t.ones_like().unwrap();
    assert_eq!(o.dims(), &[2, 3]);
    assert_eq!(o.dtype(), DType::F32);
    let vals = o.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| v == 1.0));
}

#[test]
fn test_zeros_like_preserves_high_rank() {
    let t = DynTensor::zeros(&[2, 3, 4, 5], DType::F32, &cpu()).unwrap();
    let z = t.zeros_like().unwrap();
    assert_eq!(z.dims(), &[2, 3, 4, 5]);
    let vals = z.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 2 * 3 * 4 * 5);
    assert!(vals.iter().all(|&v| v == 0.0), "all values should be zero");
}

// -- t() transpose tests -----------------------------------------------------

#[test]
fn test_t_transposes_2d() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3], &cpu()).unwrap();
    let tt = t.t().unwrap();
    assert_eq!(tt.dims(), &[3, 2]);
    // Original [0,1,2; 3,4,5] -> transposed [0,3; 1,4; 2,5]
    let vals: Vec<Vec<f32>> = tt.to_vec2().unwrap();
    assert_eq!(vals[0], &[0.0, 3.0]);
    assert_eq!(vals[1], &[1.0, 4.0]);
    assert_eq!(vals[2], &[2.0, 5.0]);
}

#[test]
fn test_t_transposes_3d_last_two_dims() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &cpu()).unwrap();
    let tt = t.t().unwrap();
    assert_eq!(tt.dims(), &[2, 4, 3]);
}

#[test]
fn test_t_rank_1_error() {
    let t = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(t.t().is_err());
}

// -- to_vec2_f32 / to_vec3_f32 tests ----------------------------------------

#[test]
fn test_to_vec2_f32_correct_structure() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3], &cpu()).unwrap();
    let v: Vec<Vec<f32>> = t.to_vec2().unwrap();
    assert_eq!(v.len(), 2);
    assert_eq!(v[0], &[0.0, 1.0, 2.0]);
    assert_eq!(v[1], &[3.0, 4.0, 5.0]);
}

#[test]
fn test_to_vec2_f32_rank_mismatch() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert!(t.to_vec2::<f32>().is_err());
}

#[test]
fn test_to_vec3_f32_correct_structure() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &cpu()).unwrap();
    let v: Vec<Vec<Vec<f32>>> = t.to_vec3().unwrap();
    assert_eq!(v.len(), 2);
    assert_eq!(v[0].len(), 3);
    assert_eq!(v[0][0], &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(v[0][1], &[4.0, 5.0, 6.0, 7.0]);
    assert_eq!(v[1][2], &[20.0, 21.0, 22.0, 23.0]);
}

#[test]
fn test_to_vec3_f32_rank_mismatch() {
    let t = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(t.to_vec3::<f32>().is_err());
}

// -- IndexOp `.i()` tests extracted to candle_compat_indexing_tests.rs (#1565) --
#[path = "candle_compat_indexing_tests.rs"]
mod indexing;

// -- Tuple syntax constructor tests (#1288) -----------------------------------

#[test]
fn test_zeros_tuple_2d() {
    let t = DynTensor::zeros((2, 3), DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert!(t.to_flat_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
}

#[test]
fn test_zeros_tuple_3d() {
    let t = DynTensor::zeros((2, 3, 4), DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3, 4]);
}

#[test]
fn test_zeros_tuple_4d() {
    let t = DynTensor::zeros((2, 3, 4, 5), DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3, 4, 5]);
}

#[test]
fn test_ones_tuple_2d() {
    let t = DynTensor::ones((3, 4), DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[3, 4]);
    assert!(t.to_flat_vec::<f32>().unwrap().iter().all(|&v| v == 1.0));
}

#[test]
fn test_new_tuple_2d() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, (2, 3), &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_from_vec_tuple_3d() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::from_vec(data.clone(), (2, 3, 4), &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3, 4]);
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_full_tuple_2d() {
    let t = DynTensor::full((3, 4), 7.0, DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[3, 4]);
    assert!(t.to_flat_vec::<f32>().unwrap().iter().all(|&v| v == 7.0));
}

#[test]
fn test_zeros_vec_dims() {
    let dims = vec![2, 3, 4];
    let t = DynTensor::zeros(dims, DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3, 4]);
}

#[test]
fn test_zeros_scalar_dim() {
    let t = DynTensor::zeros(5usize, DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[5]);
}

// -- #1650 API completeness tests ---------------------------------------------

#[test]
fn test_flatten_to_vec1_u8() {
    let t = DynTensor::from_vec_u8(vec![1, 0, 1, 0, 1, 1], &[2, 3], &cpu()).unwrap();
    let v: Vec<u8> = t.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(v, vec![1, 0, 1, 0, 1, 1]);
}

#[test]
fn test_flatten_to_vec1_i64() {
    let t = DynTensor::from_vec_i64(vec![-1, 0, 42, 100], &[2, 2], &cpu()).unwrap();
    let v: Vec<i64> = t.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(v, vec![-1, 0, 42, 100]);
}

#[test]
fn test_full_like() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let f = t.full_like(7.0).unwrap();
    assert_eq!(f.dims(), &[2, 3]);
    assert_eq!(f.dtype(), t.dtype());
    let v = f.to_flat_vec::<f32>().unwrap();
    assert!(v.iter().all(|&x| (x - 7.0).abs() < 1e-6));
}

#[test]
fn test_reshape_as() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let target = DynTensor::new(&[0.0; 6], &[3, 2], &cpu()).unwrap();
    let reshaped = a.reshape_as(&target).unwrap();
    assert_eq!(reshaped.dims(), &[3, 2]);
    assert_eq!(
        reshaped.to_flat_vec::<f32>().unwrap(),
        a.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_expand_as() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let target = DynTensor::new(&[0.0; 6], &[2, 3], &cpu()).unwrap();
    let expanded = a.expand_as(&target).unwrap();
    assert_eq!(expanded.dims(), &[2, 3]);
    let v = expanded.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

// -- 4-tuple/5-tuple IndexOp, Shape type, broadcast, conversions --------------
// Extracted to candle_compat_indexop_tests.rs for file-size compliance (#1227).
#[path = "candle_compat_indexop_tests.rs"]
mod indexop_tests;
