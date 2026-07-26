// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended GPU validation tests.
//!
//! Covers BF16 dtype acceptance through dispatch_def (#1659), error paths in
//! gpu_where_cond (mask dtype rejection), gpu_cat, and gpu_compare_tensor.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::test_common::init;

// -- Helper: BF16 GPU tensor (same as dyn_tensor_metal_tests_validation.rs) ----
//
// Uses the production cpu_to_gpu() path which creates 2-byte f16 Metal buffers
// for bf16 tensors (#1646 D7). MSL kernels read `half*` and accumulate in `float`.

fn make_bf16_gpu_tensor(shape: &[usize]) -> DynTensor {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = vec![0.0f32; numel];
    let cpu = DynTensor::new(&data, shape, &Device::Cpu).unwrap();
    let bf16_cpu = cpu.to_dtype(DType::BF16).unwrap();
    bf16_cpu.to_device(&Device::metal()).unwrap()
}

fn make_u32_gpu_tensor(shape: &[usize]) -> DynTensor {
    let numel: usize = shape.iter().product();
    let data: Vec<u32> = vec![0; numel];
    let cpu = DynTensor::from_vec_u32(data, shape, &Device::Cpu).unwrap();
    cpu.to_device(&Device::metal()).unwrap()
}

// -- BF16 acceptance: gpu_silu (#1659) -----------------------------------------

#[test]
fn test_bf16_silu_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[4]);
    let result = a.silu().expect("bf16 silu should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16, "output preserves bf16 dtype");
    assert_eq!(result.dims(), &[4]);
}

// -- BF16 acceptance: gpu_compare (ge/gt/lt/le/eq/ne) -------------------------

#[test]
fn test_bf16_compare_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[4]);
    let result = a.ge(1.0).expect("bf16 ge should succeed via dispatch_def");
    assert_eq!(result.dims(), &[4]);
}

// -- BF16 acceptance: gpu_compare_tensor --------------------------------------

#[test]
fn test_bf16_compare_tensor_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[4]);
    let b = make_bf16_gpu_tensor(&[4]);
    let result = a
        .broadcast_ge(&b)
        .expect("bf16 compare_tensor should succeed via dispatch_def");
    assert_eq!(result.dims(), &[4]);
}

// -- gpu_compare_tensor shape mismatch ----------------------------------------

#[test]
fn test_gpu_compare_tensor_shape_mismatch() {
    init();
    // Shapes [3] and [2] are not broadcast-compatible, so this should error.
    let a = DynTensor::new(&[1.0f32, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let b = DynTensor::new(&[1.0f32, 2.0], &[2], &Device::metal()).unwrap();
    let err = a.broadcast_ge(&b);
    assert!(
        err.is_err(),
        "compare_tensor with incompatible shapes should error"
    );
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("mismatch") || msg.contains("shape") || msg.contains("broadcast"),
        "error should mention shape/broadcast issue: {msg}"
    );
}

// -- gpu_where_cond: I64 mask dtype rejected ----------------------------------

#[test]
fn test_gpu_where_cond_u32_mask_rejected() {
    init();
    // U32 mask is not U8 or F32 — should be rejected by the dtype check
    // before GPU dispatch is attempted.
    let mask = make_u32_gpu_tensor(&[3]);
    let on_true = DynTensor::new(&[1.0f32, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let on_false = DynTensor::new(&[4.0f32, 5.0, 6.0], &[3], &Device::metal()).unwrap();
    let err = mask.where_cond(&on_true, &on_false);
    assert!(err.is_err(), "where_cond with U32 mask should be rejected");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("mask") || msg.contains("U8 or F32") || msg.contains("unsupported"),
        "error should mention mask dtype: {msg}"
    );
}

// -- gpu_where_cond: BF16 as mask rejected ------------------------------------
// BF16 is used as `self` (mask position) in the where_cond call below.
// Mask dtype must be U8 or F32 — BF16 is rejected by the mask dtype check
// in gpu_where_cond, not by dispatch_def.

#[test]
fn test_gpu_where_cond_bf16_mask_rejected() {
    init();
    let mask_cpu = DynTensor::from_vec_u8(vec![1, 0, 1], &[3], &Device::Cpu).unwrap();
    let mask = mask_cpu.to_device(&Device::metal()).unwrap();
    let on_true = make_bf16_gpu_tensor(&[3]);
    let on_false = DynTensor::new(&[4.0f32, 5.0, 6.0], &[3], &Device::metal()).unwrap();
    // on_true.where_cond(a, b) → self=on_true is the mask, a=mask is on_true, b=on_false
    let err = on_true.where_cond(&mask, &on_false);
    assert!(
        err.is_err(),
        "where_cond with bf16 as mask should be rejected"
    );
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("mismatch")
            || msg.contains("Unsupported")
            || msg.contains("BF16")
            || msg.contains("where_cond"),
        "error should mention dtype or where_cond issue: {msg}"
    );
}

// -- BF16 acceptance: gpu_softmax (#1659) --------------------------------------

#[test]
fn test_bf16_softmax_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[2, 3]);
    let result = a
        .softmax(1)
        .expect("bf16 softmax should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16, "output preserves bf16 dtype");
    assert_eq!(result.dims(), &[2, 3]);
}

// -- BF16 acceptance: gpu_narrow (#1659) ---------------------------------------

#[test]
fn test_bf16_narrow_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[4]);
    let result = a
        .narrow(0, 1, 2)
        .expect("bf16 narrow should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16, "output preserves bf16 dtype");
    assert_eq!(result.dims(), &[2]);
}

// -- BF16 acceptance: gpu_transpose (#1659) -----------------------------------

#[test]
fn test_bf16_transpose_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[2, 3]);
    let result = a
        .transpose(0, 1)
        .expect("bf16 transpose should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16, "output preserves bf16 dtype");
    assert_eq!(result.dims(), &[3, 2]);
}

// -- BF16 acceptance: gpu_cat (#1659) -----------------------------------------

#[test]
fn test_bf16_cat_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[3]);
    let b = make_bf16_gpu_tensor(&[3]);
    let result = DynTensor::cat(&[&a, &b], 0).expect("bf16 cat should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16, "output preserves bf16 dtype");
    assert_eq!(result.dims(), &[6]);
}

// -- BF16 acceptance: gpu_log_softmax (#1659) ---------------------------------

#[test]
fn test_bf16_log_softmax_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[2, 3]);
    let result = a
        .log_softmax(1)
        .expect("bf16 log_softmax should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16, "output preserves bf16 dtype");
    assert_eq!(result.dims(), &[2, 3]);
}

// -- BF16 acceptance: gpu_permute (#1659) -------------------------------------

#[test]
fn test_bf16_permute_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[2, 3, 4]);
    let result = a
        .permute([2, 0, 1])
        .expect("bf16 permute should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16, "output preserves bf16 dtype");
    assert_eq!(result.dims(), &[4, 2, 3]);
}

// -- BF16 acceptance: gpu_expand (#1659) --------------------------------------

#[test]
fn test_bf16_expand_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[1, 3]);
    let result = a
        .expand([4, 3])
        .expect("bf16 expand should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16, "output preserves bf16 dtype");
    assert_eq!(result.dims(), &[4, 3]);
}

// -- gpu_cat: empty tensor list -----------------------------------------------

#[test]
fn test_gpu_cat_empty_list() {
    init();
    let empty: Vec<&DynTensor> = vec![];
    let err = DynTensor::cat(&empty, 0);
    assert!(err.is_err(), "cat with empty list should error");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("empty") || msg.contains("at least"),
        "cat empty error should mention empty list: {msg}"
    );
}
