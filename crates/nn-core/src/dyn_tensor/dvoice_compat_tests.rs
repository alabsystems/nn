#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for dvoice-required DynTensor methods: gelu_erf, floor,
//! var_keepdim, from_slice, and Embedding::embeddings(). Part of #1163.

use crate::dyn_tensor::test_helpers::{approx_eq, cpu, t1d};
use crate::DynTensor;

// -- gelu_erf -----------------------------------------------------------------

#[test]
fn test_gelu_erf_zero() {
    let t = t1d(&[0.0]);
    let result = t.gelu_erf().unwrap();
    let vals = result.to_vec1::<f32>().unwrap();
    assert!(approx_eq(vals[0], 0.0, 1e-6), "gelu_erf(0) should be 0");
}

#[test]
fn test_gelu_erf_positive() {
    // GELU(1.0) ≈ 0.8413 (exact erf-based)
    let t = t1d(&[1.0]);
    let result = t.gelu_erf().unwrap();
    let vals = result.to_vec1::<f32>().unwrap();
    assert!(
        approx_eq(vals[0], 0.8413, 1e-3),
        "gelu_erf(1.0) ≈ 0.8413, got {}",
        vals[0]
    );
}

#[test]
fn test_gelu_erf_negative() {
    // GELU(-1.0) ≈ -0.1587
    let t = t1d(&[-1.0]);
    let result = t.gelu_erf().unwrap();
    let vals = result.to_vec1::<f32>().unwrap();
    assert!(
        approx_eq(vals[0], -0.1587, 1e-3),
        "gelu_erf(-1.0) ≈ -0.1587, got {}",
        vals[0]
    );
}

#[test]
fn test_gelu_erf_vs_gelu_tanh() {
    // erf-based and tanh-approximation should be close but not identical
    let t = t1d(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let erf_result = t.gelu_erf().unwrap().to_vec1::<f32>().unwrap();
    let tanh_result = t.gelu().unwrap().to_vec1::<f32>().unwrap();
    for (e, a) in erf_result.iter().zip(tanh_result.iter()) {
        assert!(
            approx_eq(*e, *a, 0.02),
            "gelu_erf and gelu(tanh) should be close: erf={e}, tanh={a}"
        );
    }
}

// -- floor --------------------------------------------------------------------

#[test]
fn test_floor_integers() {
    let t = t1d(&[1.0, 2.0, -3.0]);
    let result = t.floor().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![1.0, 2.0, -3.0]);
}

#[test]
fn test_floor_fractional() {
    let t = t1d(&[1.7, -1.3, 0.5, -0.5]);
    let result = t.floor().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![1.0, -2.0, 0.0, -1.0]);
}

// -- var_keepdim --------------------------------------------------------------

#[test]
fn test_var_keepdim_1d() {
    // [1, 2, 3] → mean=2, var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
    let t = t1d(&[1.0, 2.0, 3.0]);
    let result = t.var_keepdim(0).unwrap();
    assert_eq!(result.dims(), &[1]);
    let vals = result.to_vec1::<f32>().unwrap();
    assert!(
        approx_eq(vals[0], 2.0 / 3.0, 1e-5),
        "var([1,2,3]) ≈ 0.6667, got {}",
        vals[0]
    );
}

#[test]
fn test_var_keepdim_2d() {
    // [[1, 2], [3, 4]] → var along dim 1:
    // row 0: mean=1.5, var=0.25; row 1: mean=3.5, var=0.25
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let result = t.var_keepdim(1).unwrap();
    assert_eq!(result.dims(), &[2, 1]);
    let flat = result.to_flat_vec::<f32>().unwrap();
    assert!(approx_eq(flat[0], 0.25, 1e-5), "got {}", flat[0]);
    assert!(approx_eq(flat[1], 0.25, 1e-5), "got {}", flat[1]);
}

#[test]
fn test_var_keepdim_constant() {
    // Constant tensor → variance = 0
    let t = t1d(&[5.0, 5.0, 5.0, 5.0]);
    let result = t.var_keepdim(0).unwrap();
    let vals = result.to_vec1::<f32>().unwrap();
    assert!(approx_eq(vals[0], 0.0, 1e-6), "var of constant = 0");
}

// -- from_slice ---------------------------------------------------------------

#[test]
fn test_from_slice_basic() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = DynTensor::from_slice(&data, &[2, 3], &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), data.to_vec());
}

#[test]
fn test_from_slice_matches_new() {
    let data = [10.0f32, 20.0, 30.0];
    let t1 = DynTensor::from_slice(&data, &[3], &cpu()).unwrap();
    let t2 = DynTensor::new(&data, &[3], &cpu()).unwrap();
    assert_eq!(
        t1.to_flat_vec::<f32>().unwrap(),
        t2.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_from_slice_shape_mismatch() {
    let data = [1.0f32, 2.0, 3.0];
    let result = DynTensor::from_slice(&data, &[2, 2], &cpu());
    assert!(result.is_err(), "shape [2,2] doesn't match 3 elements");
}

// -- Embedding::embeddings() --------------------------------------------------

#[test]
fn test_embedding_embeddings_accessor() {
    use crate::layers::Embedding;
    let weight = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu()).unwrap();
    let emb = Embedding::new(weight).unwrap();
    // embeddings() should return same reference as weight()
    assert_eq!(emb.embeddings().dims(), &[3, 2]);
    assert_eq!(
        emb.embeddings().to_flat_vec::<f32>().unwrap(),
        emb.weight().to_flat_vec::<f32>().unwrap()
    );
}
