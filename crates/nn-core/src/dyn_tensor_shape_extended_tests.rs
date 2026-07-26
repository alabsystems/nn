// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended DynTensor shape and broadcasting tests.
//!
//! Covers broadcasting rules, shape manipulation, narrow/slice, concat/stack,
//! repeat/tile, flatten, expand, contiguous, and edge cases (scalar, high-rank).

use crate::dyn_tensor::ops::broadcast_output_shape;
use crate::dyn_tensor::test_helpers::{approx_eq, cpu, t1d, t2d, tnd};
use crate::{DType, DynTensor};

// =============================================================================
// 1. Broadcasting rules — NumPy-style right-aligned semantics
// =============================================================================

#[test]
fn test_broadcast_3x1_plus_1x4() {
    // [3,1] + [1,4] -> [3,4]
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[1, 4], &cpu()).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[3, 4]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    // Row 0: 1 + [10,20,30,40] = [11,21,31,41]
    // Row 1: 2 + [10,20,30,40] = [12,22,32,42]
    // Row 2: 3 + [10,20,30,40] = [13,23,33,43]
    assert_eq!(
        vals,
        vec![11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,]
    );
}

#[test]
fn test_broadcast_2x3x4_plus_4() {
    // [2,3,4] + [4] -> [2,3,4]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let a = tnd(&data, &[2, 3, 4]);
    let b = t1d(&[100.0, 200.0, 300.0, 400.0]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3, 4]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    // First row: [0,1,2,3] + [100,200,300,400] = [100,201,302,403]
    assert_eq!(&vals[0..4], &[100.0, 201.0, 302.0, 403.0]);
    // Second row: [4,5,6,7] + [100,200,300,400] = [104,205,306,407]
    assert_eq!(&vals[4..8], &[104.0, 205.0, 306.0, 407.0]);
}

#[test]
fn test_broadcast_shape_3x1_and_1x4() {
    let out = broadcast_output_shape(&[3, 1], &[1, 4]).unwrap();
    assert_eq!(out, vec![3, 4]);
}

#[test]
fn test_broadcast_shape_2x3x4_and_4() {
    let out = broadcast_output_shape(&[2, 3, 4], &[4]).unwrap();
    assert_eq!(out, vec![2, 3, 4]);
}

#[test]
fn test_broadcast_shape_1x1_and_5x5() {
    let out = broadcast_output_shape(&[1, 1], &[5, 5]).unwrap();
    assert_eq!(out, vec![5, 5]);
}

#[test]
fn test_broadcast_incompatible_3x4_plus_3() {
    // [3,4] + [3] -> error: trailing dim 4 != 3
    let result = broadcast_output_shape(&[3, 4], &[3]);
    assert!(result.is_err());
}

#[test]
fn test_broadcast_values_mul_3x1_by_1x4() {
    // [3,1] * [1,4] -> [3,4], outer-product-like
    let a = DynTensor::from_vec(vec![2.0, 3.0, 5.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0, 100.0, 1000.0, 10000.0], &[1, 4], &cpu()).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dims(), &[3, 4]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals,
        vec![
            20.0, 200.0, 2000.0, 20000.0, 30.0, 300.0, 3000.0, 30000.0, 50.0, 500.0, 5000.0,
            50000.0,
        ]
    );
}

#[test]
fn test_broadcast_sub_3d_with_1d() {
    // [2,2,3] - [3] -> [2,2,3]
    let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let a = tnd(&data, &[2, 2, 3]);
    let b = t1d(&[0.0, 10.0, 100.0]);
    let c = a.sub(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2, 3]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    // First slice: [1,2,3] - [0,10,100] = [1,-8,-97]
    assert!(approx_eq(vals[0], 1.0, 1e-6));
    assert!(approx_eq(vals[1], -8.0, 1e-6));
    assert!(approx_eq(vals[2], -97.0, 1e-6));
}

// =============================================================================
// 2. Shape manipulation — reshape, permute, transpose, squeeze, unsqueeze,
//    expand, contiguous
// =============================================================================

#[test]
fn test_reshape_1d_to_3d() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::from_vec(data.clone(), &[24], &cpu()).unwrap();
    let r = t.reshape([2, 3, 4]).unwrap();
    assert_eq!(r.dims(), &[2, 3, 4]);
    assert_eq!(r.numel(), 24);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_reshape_preserves_data_order() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.reshape([3, 2]).unwrap();
    assert_eq!(
        r.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_reshape_to_scalar() {
    let t = DynTensor::from_vec(vec![42.0], &[1], &cpu()).unwrap();
    let r = t.reshape([]).unwrap();
    assert_eq!(r.dims(), &[] as &[usize]);
    assert_eq!(r.to_scalar::<f32>().unwrap(), 42.0);
}

#[test]
fn test_reshape_from_scalar() {
    let t = DynTensor::full(&[], 7.0, DType::F32, &cpu()).unwrap();
    let r = t.reshape([1, 1, 1]).unwrap();
    assert_eq!(r.dims(), &[1, 1, 1]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![7.0]);
}

#[test]
fn test_reshape_numel_mismatch_error() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let err = t.reshape([2, 2]);
    assert!(err.is_err());
}

#[test]
fn test_permute_4d() {
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4, 5]);
    // NCHW -> NHWC
    let p = t.permute([0, 2, 3, 1]).unwrap();
    assert_eq!(p.dims(), &[2, 4, 5, 3]);
    assert_eq!(p.numel(), 120);
}

#[test]
fn test_permute_identity() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let p = t.permute([0, 1, 2]).unwrap();
    assert_eq!(p.dims(), &[2, 3, 4]);
    assert_eq!(p.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_permute_invalid_axis_error() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    // Axis out of range
    assert!(t.permute([0, 3]).is_err());
}

#[test]
fn test_transpose_is_permute_2d() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let tr = t.transpose(0, 1).unwrap();
    let perm = t.permute([1, 0]).unwrap();
    assert_eq!(tr.dims(), perm.dims());
    assert_eq!(
        tr.to_flat_vec::<f32>().unwrap(),
        perm.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_transpose_3d_inner() {
    // Transpose last two dims of a 3D tensor
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let tr = t.transpose(1, 2).unwrap();
    assert_eq!(tr.dims(), &[2, 4, 3]);
}

#[test]
fn test_transpose_self_noop() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let tr = t.transpose(0, 0).unwrap();
    assert_eq!(tr.dims(), &[2, 2]);
    assert_eq!(
        tr.to_flat_vec::<f32>().unwrap(),
        t.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_squeeze_multiple_unit_dims() {
    let t = DynTensor::zeros(&[1, 3, 1, 4, 1], DType::F32, &cpu()).unwrap();
    let s0 = t.squeeze(0).unwrap();
    assert_eq!(s0.dims(), &[3, 1, 4, 1]);
    let s2 = s0.squeeze(1).unwrap();
    assert_eq!(s2.dims(), &[3, 4, 1]);
    let s4 = s2.squeeze(2).unwrap();
    assert_eq!(s4.dims(), &[3, 4]);
}

#[test]
fn test_unsqueeze_at_end() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let u = t.unsqueeze(1).unwrap();
    assert_eq!(u.dims(), &[3, 1]);
}

#[test]
fn test_unsqueeze_at_beginning() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let u = t.unsqueeze(0).unwrap();
    assert_eq!(u.dims(), &[1, 2, 2]);
}

#[test]
fn test_unsqueeze_at_middle() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let u = t.unsqueeze(1).unwrap();
    assert_eq!(u.dims(), &[2, 1, 3]);
    assert_eq!(u.numel(), 6);
}

#[test]
fn test_expand_basic() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let e = t.expand([4, 3]).unwrap();
    assert_eq!(e.dims(), &[4, 3]);
    let vals = e.to_flat_vec::<f32>().unwrap();
    // Each row should be [1,2,3]
    for row in vals.chunks(3) {
        assert_eq!(row, &[1.0, 2.0, 3.0]);
    }
}

#[test]
fn test_expand_3d() {
    let t = DynTensor::from_vec(vec![5.0], &[1, 1, 1], &cpu()).unwrap();
    let e = t.expand([2, 3, 4]).unwrap();
    assert_eq!(e.dims(), &[2, 3, 4]);
    let vals = e.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| approx_eq(v, 5.0, 1e-6)));
    assert_eq!(vals.len(), 24);
}

#[test]
fn test_expand_non_unit_dim_same_size() {
    // expand where the dim is already the target size (no-op for that dim)
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let e = t.expand([3, 5]).unwrap();
    assert_eq!(e.dims(), &[3, 5]);
}

#[test]
fn test_expand_non_unit_dim_error() {
    // dim is 3, cannot expand to 5
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let err = t.expand([5]);
    assert!(err.is_err());
}

#[test]
fn test_expand_rank_mismatch_error() {
    let t = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    let err = t.expand([2, 3]);
    assert!(err.is_err());
}

#[test]
fn test_contiguous_returns_same_data() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let c = t.contiguous().unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert!(c.is_contiguous());
}

#[test]
fn test_contiguous_after_transpose() {
    // Transposing can produce non-contiguous layout internally;
    // contiguous() should produce a contiguous copy.
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let tr = t.transpose(0, 1).unwrap();
    let c = tr.contiguous().unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    assert!(c.is_contiguous());
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

// =============================================================================
// 3. Narrow/slice — verify correct sub-tensor extraction
// =============================================================================

#[test]
fn test_narrow_dim0_2d() {
    // [4,3] narrow along dim 0: start=1, len=2 -> [2,3]
    let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let t = tnd(&data, &[4, 3]);
    let n = t.narrow(0, 1, 2).unwrap();
    assert_eq!(n.dims(), &[2, 3]);
    assert_eq!(
        n.to_flat_vec::<f32>().unwrap(),
        vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn test_narrow_dim1_2d() {
    // [3,5] narrow along dim 1: start=1, len=3 -> [3,3]
    let data: Vec<f32> = (0..15).map(|i| i as f32).collect();
    let t = tnd(&data, &[3, 5]);
    let n = t.narrow(1, 1, 3).unwrap();
    assert_eq!(n.dims(), &[3, 3]);
    let vals = n.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0]);
}

#[test]
fn test_narrow_3d_middle_dim() {
    // [2,4,3] narrow along dim 1: start=1, len=2 -> [2,2,3]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 4, 3]);
    let n = t.narrow(1, 1, 2).unwrap();
    assert_eq!(n.dims(), &[2, 2, 3]);
    let vals = n.to_flat_vec::<f32>().unwrap();
    // Batch 0: rows 1-2 of [4,3]: [3,4,5, 6,7,8]
    assert_eq!(&vals[0..6], &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    // Batch 1: rows 1-2 of [4,3]: [15,16,17, 18,19,20]
    assert_eq!(&vals[6..12], &[15.0, 16.0, 17.0, 18.0, 19.0, 20.0]);
}

#[test]
fn test_narrow_full_range_is_identity() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = tnd(&data, &[3, 4]);
    let n = t.narrow(0, 0, 3).unwrap();
    assert_eq!(n.dims(), &[3, 4]);
    assert_eq!(n.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_narrow_single_element() {
    let t = t1d(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let n = t.narrow(0, 2, 1).unwrap();
    assert_eq!(n.dims(), &[1]);
    assert_eq!(n.to_vec1::<f32>().unwrap(), vec![30.0]);
}

#[test]
fn test_narrow_out_of_bounds_error() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    assert!(t.narrow(0, 1, 4).is_err());
}

#[test]
fn test_narrow_last_dim() {
    // [2,5] narrow dim 1: start=3, len=2 -> [2,2]
    let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 5]);
    let n = t.narrow(1, 3, 2).unwrap();
    assert_eq!(n.dims(), &[2, 2]);
    assert_eq!(n.to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0, 8.0, 9.0]);
}

// =============================================================================
// 4. Concat/stack — verify along various dimensions
// =============================================================================

#[test]
fn test_cat_dim0() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[4, 2]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn test_cat_dim1() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 2, 3);
    let c = DynTensor::cat(&[&a, &b], 1).unwrap();
    assert_eq!(c.dims(), &[2, 5]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals,
        vec![1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]
    );
}

#[test]
fn test_cat_3d_dim2() {
    let a = tnd(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let b = tnd(&[5.0, 6.0, 7.0, 8.0], &[1, 2, 2]);
    let c = DynTensor::cat(&[&a, &b], 2).unwrap();
    assert_eq!(c.dims(), &[1, 2, 4]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
    );
}

#[test]
fn test_cat_multiple_tensors() {
    let a = t1d(&[1.0]);
    let b = t1d(&[2.0, 3.0]);
    let c = t1d(&[4.0, 5.0, 6.0]);
    let r = DynTensor::cat(&[&a, &b, &c], 0).unwrap();
    assert_eq!(r.dims(), &[6]);
    assert_eq!(
        r.to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_cat_empty_list_error() {
    let r: Result<DynTensor, _> = DynTensor::cat(&[] as &[&DynTensor], 0);
    assert!(r.is_err());
}

#[test]
fn test_cat_shape_mismatch_error() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    // cat along dim 0 requires dim 1 to match
    assert!(DynTensor::cat(&[&a, &b], 0).is_err());
}

#[test]
fn test_stack_creates_new_dim() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let c = t1d(&[7.0, 8.0, 9.0]);
    let r = DynTensor::stack(&[&a, &b, &c], 0).unwrap();
    assert_eq!(r.dims(), &[3, 3]);
    assert_eq!(
        r.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn test_stack_dim1() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0]);
    let r = DynTensor::stack(&[&a, &b], 1).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    // a = [1,2], b = [3,4], stacked along dim 1:
    // [[1,3],[2,4]]
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_stack_2d_tensors() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let r = DynTensor::stack(&[&a, &b], 0).unwrap();
    assert_eq!(r.dims(), &[2, 2, 2]);
}

#[test]
fn test_stack_empty_error() {
    let r: Result<DynTensor, _> = DynTensor::stack(&[] as &[&DynTensor], 0);
    assert!(r.is_err());
}

#[test]
fn test_stack_shape_mismatch_error() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0, 5.0]);
    assert!(DynTensor::stack(&[&a, &b], 0).is_err());
}

// =============================================================================
// 5. Repeat/tile — verify tensor repetition
// =============================================================================

#[test]
fn test_repeat_basic_2d() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let r = t.repeat([2, 3]).unwrap();
    assert_eq!(r.dims(), &[4, 6]);
}

#[test]
fn test_repeat_values_1d() {
    let t = t1d(&[1.0, 2.0]);
    let r = t.repeat([3]).unwrap();
    assert_eq!(r.dims(), &[6]);
    assert_eq!(
        r.to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn test_repeat_no_op() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let r = t.repeat([1, 1]).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_repeat_zero_produces_empty() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let r = t.repeat([0]).unwrap();
    assert_eq!(r.dims(), &[0]);
    assert_eq!(r.numel(), 0);
}

#[test]
fn test_repeat_rank_mismatch_error() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(t.repeat([2]).is_err());
}

#[test]
fn test_tile_is_repeat() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let r = t.repeat([4]).unwrap();
    let ti = t.tile([4]).unwrap();
    assert_eq!(r.dims(), ti.dims());
    assert_eq!(
        r.to_flat_vec::<f32>().unwrap(),
        ti.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_repeat_2d_values() {
    // [[1,2],[3,4]].repeat([1,2]) -> [[1,2,1,2],[3,4,3,4]]
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let r = t.repeat([1, 2]).unwrap();
    assert_eq!(r.dims(), &[2, 4]);
    assert_eq!(
        r.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
    );
}

// =============================================================================
// 6. View/reshape — verify compatible shapes, error on incompatible
// =============================================================================

#[test]
fn test_reshape_various_compatible() {
    let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
    let t = DynTensor::from_vec(data, &[60], &cpu()).unwrap();

    let r1 = t.reshape([2, 30]).unwrap();
    assert_eq!(r1.dims(), &[2, 30]);

    let r2 = t.reshape([3, 4, 5]).unwrap();
    assert_eq!(r2.dims(), &[3, 4, 5]);

    let r3 = t.reshape([5, 3, 2, 2]).unwrap();
    assert_eq!(r3.dims(), &[5, 3, 2, 2]);

    let r4 = t.reshape([1, 60]).unwrap();
    assert_eq!(r4.dims(), &[1, 60]);
}

#[test]
fn test_reshape_incompatible_sizes() {
    let t = DynTensor::from_vec(vec![1.0; 12], &[12], &cpu()).unwrap();
    assert!(t.reshape([5, 3]).is_err()); // 15 != 12
    assert!(t.reshape([7]).is_err()); // 7 != 12
    assert!(t.reshape([2, 2, 2]).is_err()); // 8 != 12
}

#[test]
fn test_reshape_roundtrip() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let flat = t.reshape([24]).unwrap();
    let back = flat.reshape([2, 3, 4]).unwrap();
    assert_eq!(back.dims(), &[2, 3, 4]);
    assert_eq!(back.to_flat_vec::<f32>().unwrap(), data);
}

// =============================================================================
// 7. Flatten — verify across dimension ranges
// =============================================================================

#[test]
fn test_flatten_all_dims() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let f = t.flatten_all().unwrap();
    assert_eq!(f.dims(), &[24]);
    assert_eq!(f.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_flatten_range() {
    let t = DynTensor::zeros(&[2, 3, 4, 5], DType::F32, &cpu()).unwrap();
    let f = t.flatten(1, 2).unwrap();
    assert_eq!(f.dims(), &[2, 12, 5]);
}

#[test]
fn test_flatten_first_two() {
    let t = DynTensor::zeros(&[3, 4, 5], DType::F32, &cpu()).unwrap();
    let f = t.flatten(0, 1).unwrap();
    assert_eq!(f.dims(), &[12, 5]);
}

#[test]
fn test_flatten_last_two() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let f = t.flatten(1, 2).unwrap();
    assert_eq!(f.dims(), &[2, 12]);
}

#[test]
fn test_flatten_same_dim_noop() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let f = t.flatten(1, 1).unwrap();
    assert_eq!(f.dims(), &[2, 3, 4]);
}

#[test]
fn test_flatten_invalid_range_error() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    // start > end
    assert!(t.flatten(2, 1).is_err());
}

#[test]
fn test_flatten_all_of_4d() {
    let t = DynTensor::zeros(&[2, 3, 4, 5], DType::F32, &cpu()).unwrap();
    let f = t.flatten(0, 3).unwrap();
    assert_eq!(f.dims(), &[120]);
}

// =============================================================================
// 8. Shape error messages — verify helpful diagnostics
// =============================================================================

#[test]
fn test_reshape_error_contains_sizes() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let err = t.reshape([2, 2]).unwrap_err();
    let msg = err.to_string();
    // Should mention the mismatch (3 vs 4)
    assert!(
        msg.contains('3') || msg.contains('4'),
        "reshape error should mention sizes: {msg}"
    );
}

#[test]
fn test_squeeze_non_unit_error_message() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let err = t.squeeze(0).unwrap_err();
    let msg = err.to_string();
    // Should indicate that dimension 0 is not size 1
    assert!(
        msg.contains('2') || msg.contains("squeeze") || msg.contains("size 1"),
        "squeeze error should be descriptive: {msg}"
    );
}

#[test]
fn test_narrow_dim_out_of_range_error() {
    let t = DynTensor::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    let err = t.narrow(5, 0, 1).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("out of range")
            || msg.contains("dim")
            || msg.contains('5')
            || msg.contains('2'),
        "narrow dim error should be descriptive: {msg}"
    );
}

#[test]
fn test_expand_incompatible_error_message() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let err = t.expand([5]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("expand") || msg.contains('3') || msg.contains('5'),
        "expand error should be descriptive: {msg}"
    );
}

#[test]
fn test_cat_rank_mismatch_error() {
    let a = t1d(&[1.0, 2.0]);
    let b = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let err = DynTensor::cat(&[&a, &b], 0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("ank") || msg.contains('1') || msg.contains('2'),
        "cat rank error should be descriptive: {msg}"
    );
}

#[test]
fn test_broadcast_incompatible_error_message() {
    let result = broadcast_output_shape(&[3, 4], &[5]);
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("broadcast") || msg.contains('4') || msg.contains('5'),
        "broadcast error should be descriptive: {msg}"
    );
}

// =============================================================================
// 9. Edge cases — scalar, 1-element, high-rank tensors
// =============================================================================

#[test]
fn test_scalar_tensor_creation_and_access() {
    let t = DynTensor::full(&[], 3.14, DType::F32, &cpu()).unwrap();
    assert_eq!(t.rank(), 0);
    assert_eq!(t.dims(), &[] as &[usize]);
    assert_eq!(t.numel(), 1);
    assert!(approx_eq(t.to_scalar::<f32>().unwrap(), 3.14, 1e-5));
}

#[test]
fn test_scalar_tensor_broadcast_with_nd() {
    let scalar = DynTensor::full(&[], 10.0, DType::F32, &cpu()).unwrap();
    let tensor = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let result = tensor.add(&scalar).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(
        result.to_flat_vec::<f32>().unwrap(),
        vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    );
}

#[test]
fn test_scalar_tensor_reshape_to_1d() {
    let scalar = DynTensor::full(&[], 42.0, DType::F32, &cpu()).unwrap();
    let r = scalar.reshape([1]).unwrap();
    assert_eq!(r.dims(), &[1]);
    assert_eq!(r.to_vec1::<f32>().unwrap(), vec![42.0]);
}

#[test]
fn test_single_element_tensor_various_shapes() {
    // 1 element in different shapes should all hold the same value
    let t = DynTensor::full(&[1], 7.0, DType::F32, &cpu()).unwrap();
    let r1 = t.reshape([1, 1]).unwrap();
    assert_eq!(r1.dims(), &[1, 1]);
    let r2 = t.reshape([1, 1, 1]).unwrap();
    assert_eq!(r2.dims(), &[1, 1, 1]);
    let r3 = t.reshape([]).unwrap();
    assert_eq!(r3.rank(), 0);
    assert!(approx_eq(r3.to_scalar::<f32>().unwrap(), 7.0, 1e-6));
}

#[test]
fn test_high_rank_tensor_creation() {
    // 8-dimensional tensor — use vec! because Shape only impl From for arrays up to [usize; 7]
    let dims: Vec<usize> = vec![1, 2, 1, 2, 1, 2, 1, 2];
    let t = DynTensor::zeros(&dims, DType::F32, &cpu()).unwrap();
    assert_eq!(t.rank(), 8);
    assert_eq!(t.numel(), 16);
}

#[test]
fn test_high_rank_reshape() {
    let t = DynTensor::zeros(&[1, 2, 1, 3, 1, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.numel(), 24);
    let r = t.reshape([2, 3, 4]).unwrap();
    assert_eq!(r.dims(), &[2, 3, 4]);
}

#[test]
fn test_high_rank_broadcast() {
    // [1,1,1,1,3] + [2,2,2,2,1] -> [2,2,2,2,3]
    let a = tnd(&[1.0, 2.0, 3.0], &[1, 1, 1, 1, 3]);
    let b_data: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
    let b = tnd(&b_data, &[2, 2, 2, 2, 1]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2, 2, 2, 3]);
    assert_eq!(c.numel(), 48);
}

#[test]
fn test_high_rank_6d_broadcast_shape() {
    let out = broadcast_output_shape(&[1, 2, 1, 4, 1, 6], &[3, 1, 5, 1, 7, 1]).unwrap();
    assert_eq!(out, vec![3, 2, 5, 4, 7, 6]);
}

#[test]
fn test_empty_tensor_operations() {
    // [0] tensor
    let t = DynTensor::from_vec(Vec::<f32>::new(), &[0], &cpu()).unwrap();
    assert_eq!(t.numel(), 0);
    assert_eq!(t.rank(), 1);

    // reshape empty -> [0, 3]
    let r = t.reshape([0, 3]).unwrap();
    assert_eq!(r.dims(), &[0, 3]);
    assert_eq!(r.numel(), 0);
}

#[test]
fn test_empty_tensor_cat() {
    let a = DynTensor::from_vec(Vec::<f32>::new(), &[0, 3], &cpu()).unwrap();
    let b = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let c = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_squeeze_unsqueeze_roundtrip_3d() {
    let t = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    let squeezed = t.squeeze(2).unwrap();
    assert_eq!(squeezed.dims(), &[2, 3]);
    let unsqueezed = squeezed.unsqueeze(2).unwrap();
    assert_eq!(unsqueezed.dims(), &[2, 3, 1]);
    assert_eq!(
        unsqueezed.to_flat_vec::<f32>().unwrap(),
        t.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_chunk_basic() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let chunks = t.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(chunks[1].to_vec1::<f32>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(chunks[2].to_vec1::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_chunk_uneven() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let chunks = t.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(chunks[1].to_vec1::<f32>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(chunks[2].to_vec1::<f32>().unwrap(), vec![5.0]);
}

#[test]
fn test_chunk_zero_error() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    assert!(t.chunk(0, 0).is_err());
}

#[test]
fn test_split_basic() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let parts = t.split([2, 3], 0).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(parts[1].to_vec1::<f32>().unwrap(), vec![3.0, 4.0, 5.0]);
}

#[test]
fn test_split_mismatch_error() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0]);
    // sizes sum to 5, but dim is 4
    assert!(t.split([2, 3], 0).is_err());
}

#[test]
fn test_broadcast_left_1d_to_3d() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let expanded = t.broadcast_left((2usize, 4usize)).unwrap();
    assert_eq!(expanded.dims(), &[2, 4, 3]);
    // Every [3]-slice should be [1,2,3]
    let vals = expanded.to_flat_vec::<f32>().unwrap();
    for chunk in vals.chunks(3) {
        assert_eq!(chunk, &[1.0, 2.0, 3.0]);
    }
}

#[test]
fn test_get_selects_and_removes_dim0() {
    let t = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let row0 = t.get(0).unwrap();
    assert_eq!(row0.dims(), &[2]);
    assert_eq!(row0.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    let row2 = t.get(2).unwrap();
    assert_eq!(row2.to_vec1::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_flip_1d() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let f = t.flip(0).unwrap();
    assert_eq!(f.to_vec1::<f32>().unwrap(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_flip_2d_dim0() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let f = t.flip(0).unwrap();
    assert_eq!(f.dims(), &[3, 2]);
    assert_eq!(
        f.to_flat_vec::<f32>().unwrap(),
        vec![5.0, 6.0, 3.0, 4.0, 1.0, 2.0]
    );
}

#[test]
fn test_roll_basic() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let r = t.roll(&[1], &[0]).unwrap();
    assert_eq!(r.to_vec1::<f32>().unwrap(), vec![4.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_expand_as_convenience() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let target = DynTensor::zeros(&[4, 3], DType::F32, &cpu()).unwrap();
    let e = t.expand_as(&target).unwrap();
    assert_eq!(e.dims(), &[4, 3]);
}

#[test]
fn test_reshape_as_convenience() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], &cpu()).unwrap();
    let target = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let r = t.reshape_as(&target).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
}
