// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for tensor-vs-tensor comparison convenience methods (`eq_tensor`,
//! `ne_tensor`, `lt_tensor`, `le_tensor`, `gt_tensor`, `ge_tensor`),
//! `where_cond`, `clamp`/`clamp_min`/`clamp_max`, scalar comparisons,
//! and edge cases (broadcasting, edge values, multi-dtype).

use crate::dyn_tensor::test_helpers::{cpu, t1d, t2d};
use crate::{DType, DynTensor};

// ============================================================================
// eq_tensor
// ============================================================================

#[test]
fn test_eq_tensor_same_shape() {
    let a = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let b = t1d(&[1.0, 9.0, 3.0, 0.0]);
    let mask = a.eq_tensor(&b).unwrap();
    assert_eq!(mask.dtype(), DType::U8);
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 1, 0]);
}

#[test]
fn test_eq_tensor_broadcast() {
    // [2, 3] vs [1, 3] -> [2, 3]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = DynTensor::from_vec(vec![1.0, 5.0, 3.0], &[1, 3], &cpu()).unwrap();
    let mask = a.eq_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[2, 3]);
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[1, 0, 1, 0, 1, 0]
    );
}

#[test]
fn test_eq_tensor_broadcast_column() {
    // [3, 2] vs [3, 1] -> [3, 2]: broadcast along the column dimension
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let b = DynTensor::from_vec(vec![1.0, 3.0, 5.0], &[3, 1], &cpu()).unwrap();
    let mask = a.eq_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[3, 2]);
    // row 0: [1==1, 2==1] -> [1, 0]
    // row 1: [3==3, 4==3] -> [1, 0]
    // row 2: [5==5, 6==5] -> [1, 0]
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[1, 0, 1, 0, 1, 0]
    );
}

#[test]
fn test_eq_tensor_scalar_rank0() {
    // rank-0 scalar vs [3] -> [3]
    let a = t1d(&[5.0, 3.0, 5.0]);
    let b = DynTensor::full(&[], 5.0, DType::F32, &cpu()).unwrap();
    let mask = a.eq_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[3]);
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 1]);
}

#[test]
fn test_eq_tensor_both_broadcast() {
    // [3, 1] vs [1, 4] -> [3, 4]: both sides broadcast
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();
    let mask = a.eq_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[3, 4]);
    // row 0 (val=1): [1==1, 1==2, 1==3, 1==4] -> [1, 0, 0, 0]
    // row 1 (val=2): [2==1, 2==2, 2==3, 2==4] -> [0, 1, 0, 0]
    // row 2 (val=3): [3==1, 3==2, 3==3, 3==4] -> [0, 0, 1, 0]
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    );
}

// ============================================================================
// ne_tensor
// ============================================================================

#[test]
fn test_ne_tensor_same_shape() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[1.0, 9.0, 3.0]);
    let mask = a.ne_tensor(&b).unwrap();
    assert_eq!(mask.dtype(), DType::U8);
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 1, 0]);
}

#[test]
fn test_ne_tensor_broadcast() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let mask = a.ne_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[2, 2]);
    // row 0: [1==1, 2==2] -> [0, 0]; row 1: [3!=1, 4!=2] -> [1, 1]
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 0, 1, 1]);
}

#[test]
fn test_ne_tensor_scalar_rank0() {
    let a = t1d(&[1.0, 2.0, 3.0, 2.0]);
    let b = DynTensor::full(&[], 2.0, DType::F32, &cpu()).unwrap();
    let mask = a.ne_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[4]);
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 1, 0]);
}

#[test]
fn test_ne_tensor_both_broadcast() {
    // [3, 1] vs [1, 4] -> [3, 4]
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();
    let mask = a.ne_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[3, 4]);
    // Complement of eq_tensor: 0 where eq is 1, 1 where eq is 0
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    );
}

// ============================================================================
// lt_tensor
// ============================================================================

#[test]
fn test_lt_tensor_same_shape() {
    let a = t1d(&[1.0, 5.0, 3.0]);
    let b = t1d(&[2.0, 4.0, 3.0]);
    let mask = a.lt_tensor(&b).unwrap();
    assert_eq!(mask.dtype(), DType::U8);
    // 1<2=1, 5<4=0, 3<3=0
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 0]);
}

#[test]
fn test_lt_tensor_broadcast() {
    // [3] vs [1] -> [3]
    let a = t1d(&[1.0, 5.0, 3.0]);
    let b = DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap();
    let mask = a.lt_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[3]);
    // 1<3=1, 5<3=0, 3<3=0
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 0]);
}

#[test]
fn test_lt_tensor_scalar_rank0() {
    let a = t1d(&[1.0, 5.0, 3.0, 7.0]);
    let b = DynTensor::full(&[], 4.0, DType::F32, &cpu()).unwrap();
    let mask = a.lt_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[4]);
    // 1<4=1, 5<4=0, 3<4=1, 7<4=0
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 1, 0]);
}

#[test]
fn test_lt_tensor_both_broadcast() {
    // [3, 1] vs [1, 3] -> [3, 3]
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let mask = a.lt_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[3, 3]);
    // row 0 (a=1): [1<1, 1<2, 1<3] -> [0, 1, 1]
    // row 1 (a=2): [2<1, 2<2, 2<3] -> [0, 0, 1]
    // row 2 (a=3): [3<1, 3<2, 3<3] -> [0, 0, 0]
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 1, 1, 0, 0, 1, 0, 0, 0]
    );
}

// ============================================================================
// le_tensor
// ============================================================================

#[test]
fn test_le_tensor_same_shape() {
    let a = t1d(&[1.0, 5.0, 3.0]);
    let b = t1d(&[2.0, 4.0, 3.0]);
    let mask = a.le_tensor(&b).unwrap();
    assert_eq!(mask.dtype(), DType::U8);
    // 1<=2=1, 5<=4=0, 3<=3=1
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 1]);
}

#[test]
fn test_le_tensor_broadcast_scalar() {
    // scalar []: broadcast against [4]
    let a = t1d(&[1.0, 3.0, 5.0, 7.0]);
    let b = DynTensor::full(&[], 3.0, DType::F32, &cpu()).unwrap();
    let mask = a.le_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[4]);
    // 1<=3=1, 3<=3=1, 5<=3=0, 7<=3=0
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 1, 0, 0]);
}

#[test]
fn test_le_tensor_both_broadcast() {
    // [3, 1] vs [1, 3] -> [3, 3]
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let mask = a.le_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[3, 3]);
    // row 0 (a=1): [1<=1, 1<=2, 1<=3] -> [1, 1, 1]
    // row 1 (a=2): [2<=1, 2<=2, 2<=3] -> [0, 1, 1]
    // row 2 (a=3): [3<=1, 3<=2, 3<=3] -> [0, 0, 1]
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[1, 1, 1, 0, 1, 1, 0, 0, 1]
    );
}

// ============================================================================
// gt_tensor
// ============================================================================

#[test]
fn test_gt_tensor_same_shape() {
    let a = t1d(&[1.0, 5.0, 3.0]);
    let b = t1d(&[2.0, 4.0, 3.0]);
    let mask = a.gt_tensor(&b).unwrap();
    assert_eq!(mask.dtype(), DType::U8);
    // 1>2=0, 5>4=1, 3>3=0
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 1, 0]);
}

#[test]
fn test_gt_tensor_broadcast_2d() {
    // [2, 3] vs [3] -> [2, 3]
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let b = t1d(&[2.0, 4.0, 3.0]);
    let mask = a.gt_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[2, 3]);
    // row 0: [1>2=0, 5>4=1, 3>3=0]; row 1: [4>2=1, 2>4=0, 6>3=1]
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 1, 0, 1, 0, 1]
    );
}

#[test]
fn test_gt_tensor_scalar_rank0() {
    let a = t1d(&[10.0, 2.0, 7.0]);
    let b = DynTensor::full(&[], 5.0, DType::F32, &cpu()).unwrap();
    let mask = a.gt_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[3]);
    // 10>5=1, 2>5=0, 7>5=1
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 1]);
}

#[test]
fn test_gt_tensor_both_broadcast() {
    // [3, 1] vs [1, 3] -> [3, 3]
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let mask = a.gt_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[3, 3]);
    // row 0 (a=1): [1>1, 1>2, 1>3] -> [0, 0, 0]
    // row 1 (a=2): [2>1, 2>2, 2>3] -> [1, 0, 0]
    // row 2 (a=3): [3>1, 3>2, 3>3] -> [1, 1, 0]
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 0, 0, 1, 0, 0, 1, 1, 0]
    );
}

// ============================================================================
// ge_tensor
// ============================================================================

#[test]
fn test_ge_tensor_same_shape() {
    let a = t1d(&[1.0, 5.0, 3.0]);
    let b = t1d(&[2.0, 4.0, 3.0]);
    let mask = a.ge_tensor(&b).unwrap();
    assert_eq!(mask.dtype(), DType::U8);
    // 1>=2=0, 5>=4=1, 3>=3=1
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 1, 1]);
}

#[test]
fn test_ge_tensor_broadcast_row() {
    // [2, 3] vs [1, 3] -> [2, 3]
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let b = DynTensor::from_vec(vec![3.0, 3.0, 3.0], &[1, 3], &cpu()).unwrap();
    let mask = a.ge_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[2, 3]);
    // row 0: [1>=3=0, 5>=3=1, 3>=3=1]; row 1: [4>=3=1, 2>=3=0, 6>=3=1]
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 1, 1, 1, 0, 1]
    );
}

#[test]
fn test_ge_tensor_scalar_rank0() {
    let a = t1d(&[1.0, 3.0, 5.0]);
    let b = DynTensor::full(&[], 3.0, DType::F32, &cpu()).unwrap();
    let mask = a.ge_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[3]);
    // 1>=3=0, 3>=3=1, 5>=3=1
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 1, 1]);
}

#[test]
fn test_ge_tensor_both_broadcast() {
    // [3, 1] vs [1, 3] -> [3, 3]
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let mask = a.ge_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[3, 3]);
    // row 0 (a=1): [1>=1, 1>=2, 1>=3] -> [1, 0, 0]
    // row 1 (a=2): [2>=1, 2>=2, 2>=3] -> [1, 1, 0]
    // row 2 (a=3): [3>=1, 3>=2, 3>=3] -> [1, 1, 1]
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[1, 0, 0, 1, 1, 0, 1, 1, 1]
    );
}

// ============================================================================
// Edge values: 0.0, -0.0, very large, very small, near-epsilon differences
// ============================================================================

#[test]
fn test_compare_zero_and_negative_zero() {
    // IEEE 754: 0.0 == -0.0 is true
    let a = t1d(&[0.0, -0.0, 0.0, -0.0]);
    let b = t1d(&[-0.0, 0.0, 0.0, -0.0]);

    let eq = a.eq_tensor(&b).unwrap();
    assert_eq!(
        eq.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[1, 1, 1, 1],
        "0.0 == -0.0 should be true per IEEE 754"
    );

    let ne = a.ne_tensor(&b).unwrap();
    assert_eq!(
        ne.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 0, 0, 0],
        "0.0 != -0.0 should be false per IEEE 754"
    );

    // lt: 0.0 < -0.0 => false, -0.0 < 0.0 => false
    let lt = a.lt_tensor(&b).unwrap();
    assert_eq!(lt.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 0, 0, 0]);
}

#[test]
fn test_compare_very_large_values() {
    let large = f32::MAX;
    let a = t1d(&[large, -large, large, 0.0]);
    let b = t1d(&[large, large, -large, large]);

    let eq = a.eq_tensor(&b).unwrap();
    assert_eq!(eq.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 0, 0]);

    let gt = a.gt_tensor(&b).unwrap();
    // large > large = 0, -large > large = 0, large > -large = 1, 0 > large = 0
    assert_eq!(gt.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 0, 1, 0]);
}

#[test]
fn test_compare_very_small_values() {
    let tiny = f32::MIN_POSITIVE; // smallest positive normal
    let a = t1d(&[tiny, -tiny, tiny, 0.0]);
    let b = t1d(&[0.0, 0.0, tiny, tiny]);

    let gt = a.gt_tensor(&b).unwrap();
    // tiny > 0 = 1, -tiny > 0 = 0, tiny > tiny = 0, 0 > tiny = 0
    assert_eq!(gt.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 0, 0]);

    let lt = a.lt_tensor(&b).unwrap();
    // tiny < 0 = 0, -tiny < 0 = 1, tiny < tiny = 0, 0 < tiny = 1
    assert_eq!(lt.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 1, 0, 1]);
}

#[test]
fn test_compare_near_epsilon_differences() {
    let base = 1.0f32;
    let eps = f32::EPSILON;
    let a = t1d(&[base, base, base + eps, base - eps]);
    let b = t1d(&[base, base + eps, base, base]);

    let eq = a.eq_tensor(&b).unwrap();
    // 1.0 == 1.0 = 1, 1.0 == 1.0+eps = 0, 1.0+eps == 1.0 = 0, 1.0-eps == 1.0 = 0
    assert_eq!(eq.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 0, 0]);

    let lt = a.lt_tensor(&b).unwrap();
    // 1.0 < 1.0 = 0, 1.0 < 1.0+eps = 1, 1.0+eps < 1.0 = 0, 1.0-eps < 1.0 = 1
    assert_eq!(lt.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 1, 0, 1]);
}

#[test]
fn test_compare_subnormal_values() {
    // Subnormals: smaller than MIN_POSITIVE but greater than 0
    let subnormal = f32::MIN_POSITIVE / 2.0;
    let a = t1d(&[subnormal, 0.0, subnormal]);
    let b = t1d(&[0.0, subnormal, subnormal]);

    let gt = a.gt_tensor(&b).unwrap();
    // subnormal > 0 = 1, 0 > subnormal = 0, subnormal > subnormal = 0
    assert_eq!(gt.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 0]);

    let eq = a.eq_tensor(&b).unwrap();
    assert_eq!(eq.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 0, 1]);
}

// ============================================================================
// Comparison op consistency: for identical tensors
// ============================================================================

#[test]
fn test_compare_tensor_all_ops_consistent() {
    // For identical tensors: eq=all 1, ne=all 0, lt=all 0, le=all 1, gt=all 0, ge=all 1
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);

    let eq = a.eq_tensor(&b).unwrap();
    let ne = a.ne_tensor(&b).unwrap();
    let lt = a.lt_tensor(&b).unwrap();
    let le = a.le_tensor(&b).unwrap();
    let gt = a.gt_tensor(&b).unwrap();
    let ge = a.ge_tensor(&b).unwrap();

    assert_eq!(eq.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 1, 1]);
    assert_eq!(ne.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 0, 0]);
    assert_eq!(lt.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 0, 0]);
    assert_eq!(le.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 1, 1]);
    assert_eq!(gt.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 0, 0]);
    assert_eq!(ge.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 1, 1]);
}

#[test]
fn test_compare_strict_ordering_consistency() {
    // For a < b strictly: lt=1, le=1, gt=0, ge=0, eq=0, ne=1
    let a = t1d(&[1.0, 1.0, 1.0]);
    let b = t1d(&[2.0, 2.0, 2.0]);

    assert_eq!(
        a.lt_tensor(&b)
            .unwrap()
            .as_cpu_u8()
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1, 1, 1]
    );
    assert_eq!(
        a.le_tensor(&b)
            .unwrap()
            .as_cpu_u8()
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1, 1, 1]
    );
    assert_eq!(
        a.gt_tensor(&b)
            .unwrap()
            .as_cpu_u8()
            .unwrap()
            .as_slice()
            .unwrap(),
        &[0, 0, 0]
    );
    assert_eq!(
        a.ge_tensor(&b)
            .unwrap()
            .as_cpu_u8()
            .unwrap()
            .as_slice()
            .unwrap(),
        &[0, 0, 0]
    );
    assert_eq!(
        a.eq_tensor(&b)
            .unwrap()
            .as_cpu_u8()
            .unwrap()
            .as_slice()
            .unwrap(),
        &[0, 0, 0]
    );
    assert_eq!(
        a.ne_tensor(&b)
            .unwrap()
            .as_cpu_u8()
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1, 1, 1]
    );
}

#[test]
fn test_compare_eq_ne_complement() {
    // eq and ne should be complements for all elements
    let a = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = t1d(&[5.0, 2.0, 1.0, 4.0, 3.0]);

    let eq = a.eq_tensor(&b).unwrap();
    let ne = a.ne_tensor(&b).unwrap();
    let eq_vals = eq.as_cpu_u8().unwrap();
    let ne_vals = ne.as_cpu_u8().unwrap();

    for (e, n) in eq_vals.iter().zip(ne_vals.iter()) {
        assert_eq!(
            *e + *n,
            1,
            "eq + ne should be 1 for every element, got eq={e}, ne={n}"
        );
    }
}

#[test]
fn test_compare_lt_ge_complement() {
    // lt and ge should be complements (for non-NaN values)
    let a = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = t1d(&[5.0, 2.0, 1.0, 4.0, 3.0]);

    let lt = a.lt_tensor(&b).unwrap();
    let ge = a.ge_tensor(&b).unwrap();
    let lt_vals = lt.as_cpu_u8().unwrap();
    let ge_vals = ge.as_cpu_u8().unwrap();

    for (l, g) in lt_vals.iter().zip(ge_vals.iter()) {
        assert_eq!(
            *l + *g,
            1,
            "lt + ge should be 1 for every element, got lt={l}, ge={g}"
        );
    }
}

#[test]
fn test_compare_gt_le_complement() {
    // gt and le should be complements (for non-NaN values)
    let a = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = t1d(&[5.0, 2.0, 1.0, 4.0, 3.0]);

    let gt = a.gt_tensor(&b).unwrap();
    let le = a.le_tensor(&b).unwrap();
    let gt_vals = gt.as_cpu_u8().unwrap();
    let le_vals = le.as_cpu_u8().unwrap();

    for (g, l) in gt_vals.iter().zip(le_vals.iter()) {
        assert_eq!(
            *g + *l,
            1,
            "gt + le should be 1 for every element, got gt={g}, le={l}"
        );
    }
}

// ============================================================================
// Error cases
// ============================================================================

#[test]
fn test_compare_tensor_incompatible_shapes_errors() {
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(a.eq_tensor(&b).is_err());
    assert!(a.ne_tensor(&b).is_err());
    assert!(a.lt_tensor(&b).is_err());
    assert!(a.le_tensor(&b).is_err());
    assert!(a.gt_tensor(&b).is_err());
    assert!(a.ge_tensor(&b).is_err());
}

#[test]
fn test_compare_tensor_empty_tensors() {
    let a = DynTensor::from_vec(Vec::<f32>::new(), &[0], &cpu()).unwrap();
    let b = DynTensor::from_vec(Vec::<f32>::new(), &[0], &cpu()).unwrap();
    let mask = a.eq_tensor(&b).unwrap();
    assert_eq!(mask.dims(), &[0]);
    assert_eq!(mask.dtype(), DType::U8);
}

// ============================================================================
// where_cond
// ============================================================================

#[test]
fn test_where_cond_with_eq_tensor() {
    let a = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let b = t1d(&[1.0, 9.0, 3.0, 0.0]);
    let mask = a.eq_tensor(&b).unwrap();

    let on_true = t1d(&[10.0, 20.0, 30.0, 40.0]);
    let on_false = t1d(&[100.0, 200.0, 300.0, 400.0]);
    let result = mask.where_cond(&on_true, &on_false).unwrap();

    let vals = result.to_f32_array().unwrap();
    // eq at [0, 2] -> on_true; ne at [1, 3] -> on_false
    assert_eq!(vals.as_slice().unwrap(), &[10.0, 200.0, 30.0, 400.0]);
}

#[test]
fn test_where_cond_with_gt_tensor_broadcast() {
    // mask from gt_tensor with broadcasting, then where_cond
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let threshold = t1d(&[3.0, 3.0, 3.0]);
    let mask = a.gt_tensor(&threshold).unwrap();
    assert_eq!(mask.dims(), &[2, 3]);

    let high = DynTensor::full(&[2, 3], 1.0, DType::F32, &cpu()).unwrap();
    let low = DynTensor::full(&[2, 3], 0.0, DType::F32, &cpu()).unwrap();
    let result = mask.where_cond(&high, &low).unwrap();

    let vals = result.to_f32_array().unwrap();
    // row 0: [1>3=0, 5>3=1, 3>3=0] -> [0, 1, 0]
    // row 1: [4>3=1, 2>3=0, 6>3=1] -> [1, 0, 1]
    assert_eq!(vals.as_slice().unwrap(), &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn test_where_cond_broadcast_mask() {
    // mask [1, 3], on_true [2, 3], on_false [2, 3] -> [2, 3]
    let mask_data =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3]), vec![1u8, 0, 1]).unwrap();
    let mask = DynTensor::from_cpu_u8(mask_data).unwrap();

    let on_true = t2d(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3);
    let on_false = t2d(&[100.0, 200.0, 300.0, 400.0, 500.0, 600.0], 2, 3);

    let result = mask.where_cond(&on_true, &on_false).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let vals = result.to_f32_array().unwrap();
    // mask [1, 0, 1] broadcasts to both rows
    assert_eq!(
        vals.as_slice().unwrap(),
        &[10.0, 200.0, 30.0, 40.0, 500.0, 60.0]
    );
}

#[test]
fn test_where_cond_all_true() {
    let mask_data =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![1u8, 1, 1, 1]).unwrap();
    let mask = DynTensor::from_cpu_u8(mask_data).unwrap();

    let on_true = t1d(&[10.0, 20.0, 30.0, 40.0]);
    let on_false = t1d(&[100.0, 200.0, 300.0, 400.0]);
    let result = mask.where_cond(&on_true, &on_false).unwrap();

    let vals = result.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_where_cond_all_false() {
    let mask_data =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![0u8, 0, 0, 0]).unwrap();
    let mask = DynTensor::from_cpu_u8(mask_data).unwrap();

    let on_true = t1d(&[10.0, 20.0, 30.0, 40.0]);
    let on_false = t1d(&[100.0, 200.0, 300.0, 400.0]);
    let result = mask.where_cond(&on_true, &on_false).unwrap();

    let vals = result.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[100.0, 200.0, 300.0, 400.0]);
}

#[test]
fn test_where_cond_broadcast_true_false_tensors() {
    // mask [1, 3], on_true [3, 1], on_false [1, 3] -> all broadcast to [3, 3]
    // All three have rank 2 (expand requires same rank).
    let mask_data =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3]), vec![1u8, 0, 1]).unwrap();
    let mask = DynTensor::from_cpu_u8(mask_data).unwrap();

    let on_true = DynTensor::from_vec(vec![10.0, 20.0, 30.0], &[3, 1], &cpu()).unwrap();
    let on_false = DynTensor::from_vec(vec![100.0, 200.0, 300.0], &[1, 3], &cpu()).unwrap();

    let result = mask.where_cond(&on_true, &on_false).unwrap();
    assert_eq!(result.dims(), &[3, 3]);
    let vals = result.to_f32_array().unwrap();
    // Broadcast shapes:
    //   mask [1, 3] -> [3, 3]: [[1,0,1],[1,0,1],[1,0,1]]
    //   on_true [3, 1] -> [3, 3]: [[10,10,10],[20,20,20],[30,30,30]]
    //   on_false [1, 3] -> [3, 3]: [[100,200,300],[100,200,300],[100,200,300]]
    // Result: row 0: [10, 200, 10], row 1: [20, 200, 20], row 2: [30, 200, 30]
    assert_eq!(
        vals.as_slice().unwrap(),
        &[10.0, 200.0, 10.0, 20.0, 200.0, 20.0, 30.0, 200.0, 30.0]
    );
}

#[test]
fn test_where_cond_cross_rank_broadcast_errors() {
    // expand requires same rank: rank-0 mask cannot broadcast to rank-1 tensors
    // because broadcast_output_shape produces a different rank than the mask.
    let mask_data = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[]), vec![1u8]).unwrap();
    let mask = DynTensor::from_cpu_u8(mask_data).unwrap();

    let on_true = t1d(&[10.0, 20.0, 30.0]);
    let on_false = t1d(&[100.0, 200.0, 300.0]);
    // Cross-rank broadcast not supported by expand
    assert!(mask.where_cond(&on_true, &on_false).is_err());
}

#[test]
fn test_where_cond_same_rank_scalar_broadcast() {
    // Same-rank broadcast: mask [1], on_true [3], on_false [3] -> [3]
    let mask_data = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![1u8]).unwrap();
    let mask = DynTensor::from_cpu_u8(mask_data).unwrap();

    let on_true = t1d(&[10.0, 20.0, 30.0]);
    let on_false = t1d(&[100.0, 200.0, 300.0]);
    let result = mask.where_cond(&on_true, &on_false).unwrap();

    let vals = result.to_f32_array().unwrap();
    // All true: select on_true for everything
    assert_eq!(vals.as_slice().unwrap(), &[10.0, 20.0, 30.0]);
}

#[test]
fn test_where_cond_preserves_negative_values() {
    // Verify where_cond works correctly with negative values
    let mask_data =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![1u8, 0, 1, 0]).unwrap();
    let mask = DynTensor::from_cpu_u8(mask_data).unwrap();

    let on_true = t1d(&[-1.0, -2.0, -3.0, -4.0]);
    let on_false = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let result = mask.where_cond(&on_true, &on_false).unwrap();

    let vals = result.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn test_where_cond_from_comparison_chain() {
    // Realistic pattern: compare -> where_cond -> further compute
    let x = t1d(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let zeros = t1d(&[0.0, 0.0, 0.0, 0.0, 0.0]);
    let mask = x.ge_tensor(&zeros).unwrap();
    // mask: [0, 0, 1, 1, 1] (>= 0)

    let result = mask.where_cond(&x, &zeros).unwrap();
    let vals = result.to_f32_array().unwrap();
    // relu-like: keep positive, zero out negative
    assert_eq!(vals.as_slice().unwrap(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
}

// ============================================================================
// clamp
// ============================================================================

#[test]
fn test_clamp_both_bounds() {
    let x = t1d(&[-5.0, -1.0, 0.0, 1.0, 5.0, 10.0]);
    let result = x.clamp(0.0, 5.0).unwrap();
    let vals = result.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[0.0, 0.0, 0.0, 1.0, 5.0, 5.0]);
}

#[test]
fn test_clamp_min_only() {
    let x = t1d(&[-5.0, -1.0, 0.0, 1.0, 5.0]);
    let result = x.clamp_min(0.0).unwrap();
    let vals = result.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[0.0, 0.0, 0.0, 1.0, 5.0]);
}

#[test]
fn test_clamp_max_only() {
    let x = t1d(&[-5.0, -1.0, 0.0, 1.0, 5.0]);
    let result = x.clamp_max(1.0).unwrap();
    let vals = result.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[-5.0, -1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_clamp_values_already_in_range() {
    let x = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let result = x.clamp(0.0, 5.0).unwrap();
    let vals = result.to_f32_array().unwrap();
    // All values already in [0, 5], so no change
    assert_eq!(vals.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_clamp_negative_range() {
    let x = t1d(&[-10.0, -5.0, -3.0, 0.0, 3.0]);
    let result = x.clamp(-5.0, -1.0).unwrap();
    let vals = result.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[-5.0, -5.0, -3.0, -1.0, -1.0]);
}

#[test]
fn test_clamp_single_point() {
    // min == max: all values collapse to that single point
    let x = t1d(&[-2.0, 0.0, 2.0, 5.0]);
    let result = x.clamp(3.0, 3.0).unwrap();
    let vals = result.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn test_clamp_min_negative_values() {
    let x = t1d(&[-10.0, -5.0, -1.0, 0.0, 1.0]);
    let result = x.clamp_min(-3.0).unwrap();
    let vals = result.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[-3.0, -3.0, -1.0, 0.0, 1.0]);
}

#[test]
fn test_clamp_max_large_values() {
    let x = t1d(&[100.0, 1000.0, 1e6, -1.0]);
    let result = x.clamp_max(500.0).unwrap();
    let vals = result.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[100.0, 500.0, 500.0, -1.0]);
}

#[test]
fn test_clamp_2d_tensor() {
    let x = t2d(&[-3.0, 0.0, 3.0, 6.0, -1.0, 10.0], 2, 3);
    let result = x.clamp(0.0, 5.0).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let vals = result.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[0.0, 0.0, 3.0, 5.0, 0.0, 5.0]);
}

#[test]
fn test_clamp_preserves_dtype_f32() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let result = x.clamp(0.0, 5.0).unwrap();
    assert_eq!(result.dtype(), DType::F32);
}

#[test]
fn test_clamp_min_preserves_dtype_f32() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let result = x.clamp_min(0.0).unwrap();
    assert_eq!(result.dtype(), DType::F32);
}

#[test]
fn test_clamp_max_preserves_dtype_f32() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let result = x.clamp_max(5.0).unwrap();
    assert_eq!(result.dtype(), DType::F32);
}

// ============================================================================
// Scalar comparison (eq/ne/lt/le/gt/ge with f64 scalar)
// ============================================================================

#[test]
fn test_scalar_eq() {
    let x = t1d(&[1.0, 2.0, 3.0, 2.0, 1.0]);
    let mask = x.eq(2.0).unwrap();
    assert_eq!(mask.dtype(), DType::U8);
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 1, 0, 1, 0]
    );
}

#[test]
fn test_scalar_ne() {
    let x = t1d(&[1.0, 2.0, 3.0, 2.0, 1.0]);
    let mask = x.ne(2.0).unwrap();
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[1, 0, 1, 0, 1]
    );
}

#[test]
fn test_scalar_lt() {
    let x = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let mask = x.lt(3.0).unwrap();
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[1, 1, 0, 0, 0]
    );
}

#[test]
fn test_scalar_le() {
    let x = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let mask = x.le(3.0).unwrap();
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[1, 1, 1, 0, 0]
    );
}

#[test]
fn test_scalar_gt() {
    let x = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let mask = x.gt(3.0).unwrap();
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 0, 0, 1, 1]
    );
}

#[test]
fn test_scalar_ge() {
    let x = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let mask = x.ge(3.0).unwrap();
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 0, 1, 1, 1]
    );
}

#[test]
fn test_scalar_compare_zero() {
    let x = t1d(&[-1.0, -0.0, 0.0, 1.0]);

    let eq = x.eq(0.0).unwrap();
    // IEEE 754: -0.0 == 0.0 is true
    assert_eq!(eq.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 1, 1, 0]);

    let gt = x.gt(0.0).unwrap();
    // -1 > 0 = 0, -0 > 0 = 0, 0 > 0 = 0, 1 > 0 = 1
    assert_eq!(gt.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 0, 0, 1]);
}

#[test]
fn test_scalar_compare_2d() {
    let x = t2d(&[1.0, 5.0, 3.0, 7.0], 2, 2);
    let mask = x.gt(3.0).unwrap();
    assert_eq!(mask.dims(), &[2, 2]);
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 1, 0, 1]);
}

// ============================================================================
// Multi-dtype: BF16 comparisons
// ============================================================================

#[test]
fn test_compare_bf16_tensors() {
    let a_f32 = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let b_f32 = t1d(&[1.0, 3.0, 3.0, 2.0]);
    let a = a_f32.to_dtype(DType::BF16).unwrap();
    let b = b_f32.to_dtype(DType::BF16).unwrap();

    let eq = a.eq_tensor(&b).unwrap();
    assert_eq!(eq.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 1, 0]);

    let lt = a.lt_tensor(&b).unwrap();
    // 1<1=0, 2<3=1, 3<3=0, 4<2=0
    assert_eq!(lt.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 1, 0, 0]);

    let gt = a.gt_tensor(&b).unwrap();
    // 1>1=0, 2>3=0, 3>3=0, 4>2=1
    assert_eq!(gt.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 0, 0, 1]);
}

#[test]
fn test_scalar_compare_bf16() {
    let x_f32 = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = x_f32.to_dtype(DType::BF16).unwrap();

    let mask = x.gt(3.0).unwrap();
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 0, 0, 1, 1]
    );
}

#[test]
fn test_clamp_bf16() {
    let x_f32 = t1d(&[-5.0, -1.0, 0.0, 1.0, 5.0, 10.0]);
    let x = x_f32.to_dtype(DType::BF16).unwrap();
    let result = x.clamp(0.0, 5.0).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    // Convert back to f32 for value check
    let result_f32 = result.to_dtype(DType::F32).unwrap();
    let vals = result_f32.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[0.0, 0.0, 0.0, 1.0, 5.0, 5.0]);
}

#[test]
fn test_where_cond_bf16_tensors() {
    let mask_data =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![1u8, 0, 1, 0]).unwrap();
    let mask = DynTensor::from_cpu_u8(mask_data).unwrap();

    let on_true = t1d(&[10.0, 20.0, 30.0, 40.0])
        .to_dtype(DType::BF16)
        .unwrap();
    let on_false = t1d(&[100.0, 200.0, 300.0, 400.0])
        .to_dtype(DType::BF16)
        .unwrap();
    let result = mask.where_cond(&on_true, &on_false).unwrap();
    assert_eq!(result.dtype(), DType::BF16);

    let result_f32 = result.to_dtype(DType::F32).unwrap();
    let vals = result_f32.to_f32_array().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[10.0, 200.0, 30.0, 400.0]);
}

// ============================================================================
// Multi-dtype: F16 comparisons
// ============================================================================

#[test]
fn test_compare_f16_tensors() {
    let a_f32 = t1d(&[1.0, 2.0, 3.0]);
    let b_f32 = t1d(&[2.0, 2.0, 1.0]);
    let a = a_f32.to_dtype(DType::F16).unwrap();
    let b = b_f32.to_dtype(DType::F16).unwrap();

    let eq = a.eq_tensor(&b).unwrap();
    assert_eq!(eq.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 1, 0]);

    let le = a.le_tensor(&b).unwrap();
    // 1<=2=1, 2<=2=1, 3<=1=0
    assert_eq!(le.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 1, 0]);
}

// ============================================================================
// 3D broadcast patterns
// ============================================================================

#[test]
fn test_compare_3d_broadcast() {
    // [2, 1, 3] vs [1, 2, 3] -> [2, 2, 3]
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[1, 2, 3], &cpu()).unwrap();

    let eq = a.eq_tensor(&b).unwrap();
    assert_eq!(eq.dims(), &[2, 2, 3]);
    let vals = eq.as_cpu_u8().unwrap();
    let slice = vals.as_slice().unwrap();
    // a[0,0,:] = [1,2,3], b[0,0,:] = [1,5,3] -> [1,0,1]
    // a[0,0,:] = [1,2,3], b[0,1,:] = [4,2,6] -> [0,1,0]
    // a[1,0,:] = [4,5,6], b[0,0,:] = [1,5,3] -> [0,1,0]
    // a[1,0,:] = [4,5,6], b[0,1,:] = [4,2,6] -> [1,0,1]
    assert_eq!(slice, &[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]);
}

// ============================================================================
// Comparison + where_cond end-to-end patterns
// ============================================================================

#[test]
fn test_threshold_gating_pattern() {
    // Common ML pattern: threshold gating (like attention mask)
    let logits = t1d(&[-2.0, -0.5, 0.0, 0.5, 2.0]);
    let threshold = DynTensor::full(&[], 0.0, DType::F32, &cpu()).unwrap();
    let mask = logits.ge_tensor(&threshold).unwrap();

    let neg_inf = DynTensor::full(&[5], f64::from(f32::NEG_INFINITY), DType::F32, &cpu()).unwrap();
    let result = mask.where_cond(&logits, &neg_inf).unwrap();

    let vals = result.to_f32_array().unwrap();
    assert_eq!(
        vals.as_slice().unwrap(),
        &[f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, 0.5, 2.0]
    );
}

#[test]
fn test_clamp_then_compare() {
    // Clamp then compare: values outside range should all be at boundary
    let x = t1d(&[-10.0, -1.0, 0.0, 1.0, 10.0]);
    let clamped = x.clamp(-2.0, 2.0).unwrap();

    // After clamping, -10 -> -2, 10 -> 2; values in range unchanged
    let ge_zero = clamped.ge(0.0).unwrap();
    assert_eq!(
        ge_zero.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 0, 1, 1, 1]
    );
}

#[test]
fn test_compare_single_element_tensors() {
    let a = t1d(&[42.0]);
    let b = t1d(&[42.0]);
    let eq = a.eq_tensor(&b).unwrap();
    assert_eq!(eq.as_cpu_u8().unwrap().as_slice().unwrap(), &[1]);

    let c = t1d(&[43.0]);
    let ne = a.ne_tensor(&c).unwrap();
    assert_eq!(ne.as_cpu_u8().unwrap().as_slice().unwrap(), &[1]);
}
