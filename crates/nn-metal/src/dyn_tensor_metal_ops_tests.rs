#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native math/activation op tests — validates round-trip elimination (#1147).
//!
//! These ops previously forced GPU→CPU→GPU transfers. After the round-trip
//! elimination changes, they stay on GPU via decomposition into existing
//! GPU-native primitives (relu, exp, log, neg, add_scalar, mul_scalar).
//!
//! Reduction tests (sum_all, mean_all, reduce_sum/max/min) are in
//! `dyn_tensor_metal_ops_tests_reductions.rs`.
//!
//! Maximum/minimum tests (BinaryOp::Maximum, BinaryOp::Minimum, NaN, Inf) are in
//! `dyn_tensor_metal_ops_tests_maxmin.rs`.

#[path = "dyn_tensor_metal_ops_tests_reductions.rs"]
mod reductions;

#[path = "dyn_tensor_metal_ops_tests_maxmin.rs"]
mod maxmin;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::test_common::{assert_gpu_vals, init};

// -- elu ----------------------------------------------------------------------

#[test]
fn test_gpu_elu_positive_unchanged() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let r = t.elu(1.0).unwrap();
    assert_gpu_vals(&r, &[1.0, 2.0, 3.0], 1e-4, "elu(positive)");
}

#[test]
fn test_gpu_elu_negative() {
    init();
    // elu(-1, alpha=1) = 1*(exp(-1)-1) ≈ -0.6321
    let t = DynTensor::new(&[-1.0, -2.0, 0.0], &[3], &Device::metal()).unwrap();
    let r = t.elu(1.0).unwrap();
    let expected = [(-1.0_f32).exp_m1(), (-2.0_f32).exp_m1(), 0.0];
    assert_gpu_vals(&r, &expected, 1e-4, "elu(negative)");
}

// -- clamp_min / clamp_max / clamp -------------------------------------------

#[test]
fn test_gpu_clamp_min_stays_on_gpu() {
    init();
    let t = DynTensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5], &Device::metal()).unwrap();
    let r = t.clamp_min(0.0).unwrap();
    assert_gpu_vals(&r, &[0.0, 0.0, 0.0, 1.0, 2.0], 1e-6, "clamp_min(0)");
}

#[test]
fn test_gpu_clamp_max_stays_on_gpu() {
    init();
    let t = DynTensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5], &Device::metal()).unwrap();
    let r = t.clamp_max(0.5).unwrap();
    assert_gpu_vals(&r, &[-2.0, -1.0, 0.0, 0.5, 0.5], 1e-6, "clamp_max(0.5)");
}

#[test]
fn test_gpu_clamp_stays_on_gpu() {
    init();
    let t = DynTensor::new(&[-3.0, -1.0, 0.5, 1.5, 3.0], &[5], &Device::metal()).unwrap();
    let r = t.clamp(-1.0, 1.0).unwrap();
    assert_gpu_vals(&r, &[-1.0, -1.0, 0.5, 1.0, 1.0], 1e-6, "clamp(-1,1)");
}

// Regression test for dvoice #641 / nn #1652: clamp with nonzero bounds
// returning all zeros on Metal. Tests the exact reproduction case from the report.
#[test]
fn test_gpu_clamp_nonzero_bounds() {
    init();
    let t = DynTensor::new(&[0.5, 1.0, 2.0], &[3], &Device::metal()).unwrap();
    let r = t.clamp(0.1, 10.0).unwrap();
    // All values within [0.1, 10.0] — should be unchanged.
    assert_gpu_vals(&r, &[0.5, 1.0, 2.0], 1e-6, "clamp(0.1, 10.0)");
}

// Clamp where min/max are fractional (not 0 or integer boundaries).
#[test]
fn test_gpu_clamp_fractional_bounds() {
    init();
    let t = DynTensor::new(&[0.01, 0.05, 0.5, 5.0, 50.0], &[5], &Device::metal()).unwrap();
    let r = t.clamp(0.1, 10.0).unwrap();
    assert_gpu_vals(
        &r,
        &[0.1, 0.1, 0.5, 5.0, 10.0],
        1e-6,
        "clamp(0.1, 10.0) fractional",
    );
}

// -- powf ---------------------------------------------------------------------

#[test]
fn test_gpu_powf_stays_on_gpu() {
    init();
    // powf(2) = x^2, GPU path uses exp(2*log(x))
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::metal()).unwrap();
    let r = t.powf(2.0).unwrap();
    assert_gpu_vals(&r, &[1.0, 4.0, 9.0, 16.0], 1e-3, "powf(2)");
}

#[test]
fn test_gpu_powf_sqrt() {
    init();
    // powf(0.5) = sqrt(x)
    let t = DynTensor::new(&[4.0, 9.0, 16.0, 25.0], &[4], &Device::metal()).unwrap();
    let r = t.powf(0.5).unwrap();
    assert_gpu_vals(&r, &[2.0, 3.0, 4.0, 5.0], 1e-3, "powf(0.5)");
}

#[test]
fn test_gpu_powf_negative_base_even_exponent() {
    init();
    // (-x)^2 = x^2 — even integer exponent, negative base should work.
    // Fixes #1171: previously produced NaN via exp(2*log(-2)) = exp(NaN).
    let t = DynTensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5], &Device::metal()).unwrap();
    let r = t.powf(2.0).unwrap();
    assert_gpu_vals(&r, &[4.0, 1.0, 0.0, 1.0, 4.0], 1e-3, "powf(2) neg base");
}

#[test]
fn test_gpu_powf_negative_base_odd_exponent() {
    init();
    // (-x)^3 = -x^3 — odd integer exponent, negative base should negate.
    let t = DynTensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5], &Device::metal()).unwrap();
    let r = t.powf(3.0).unwrap();
    assert_gpu_vals(&r, &[-8.0, -1.0, 0.0, 1.0, 8.0], 1e-3, "powf(3) neg base");
}

#[test]
fn test_gpu_powf_negative_base_non_integer_produces_nan() {
    init();
    // (-x)^0.5 = NaN for negative base — IEEE 754 semantics.
    let t = DynTensor::new(&[-4.0, 4.0], &[2], &Device::metal()).unwrap();
    let r = t.powf(0.5).unwrap();
    let vals = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(vals[0].is_nan(), "(-4)^0.5 should be NaN, got {}", vals[0]);
    assert!(
        (vals[1] - 2.0).abs() < 1e-3,
        "4^0.5 should be 2.0, got {}",
        vals[1]
    );
}

#[test]
fn test_gpu_powf_matches_cpu_negative_inputs() {
    init();
    // GPU and CPU should produce identical results for negative inputs.
    let data = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let cpu = DynTensor::new(&data, &[7], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[7], &Device::metal()).unwrap();
    for e in [2.0, 3.0, 4.0, -1.0] {
        let cpu_r = cpu.powf(e).unwrap().to_flat_vec::<f32>().unwrap();
        let gpu_r = gpu
            .powf(e)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();
        for (i, (c, g)) in cpu_r.iter().zip(gpu_r.iter()).enumerate() {
            if c.is_nan() {
                assert!(g.is_nan(), "powf({e})[{i}]: CPU=NaN but GPU={g}");
            } else if c.is_infinite() {
                assert!(
                    g.is_infinite() && c.signum() == g.signum(),
                    "powf({e})[{i}]: CPU={c} but GPU={g}"
                );
            } else {
                let tol = c.abs() * 1e-3 + 1e-6;
                assert!(
                    (c - g).abs() < tol,
                    "powf({e})[{i}]: CPU={c} vs GPU={g}, diff={}",
                    (c - g).abs()
                );
            }
        }
    }
}

// -- GPU division-by-zero behavior (#1180) ------------------------------------

#[test]
fn test_gpu_div_by_zero_produces_inf() {
    init();
    // Documents #1180: GPU division by zero silently produces Inf.
    // W4-45 added check_gpu_div_result_finite() but it was removed (#1147)
    // because GPU→CPU transfer on every division was too expensive.
    // GPU now follows IEEE 754 semantics; model-level NaN guards (#941, #958)
    // catch non-finite values at stage boundaries instead.
    let a = DynTensor::new(&[1.0, 2.0], &[2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[0.0, 1.0], &[2], &Device::metal()).unwrap();
    let result = a.broadcast_div(&b);
    // GPU path succeeds silently — IEEE 754 semantics.
    let r = result.expect("GPU div by zero should succeed (IEEE 754)");
    let vals = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        vals[0].is_infinite(),
        "GPU 1/0 should be Inf, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 2.0).abs() < 1e-6,
        "GPU 2/1 should be 2.0, got {}",
        vals[1]
    );
}

// -- max_all / min_all GPU dispatch (W3-41 reduce_all_impl) -------------------

#[test]
fn test_gpu_max_all_2d() {
    init();
    // max_all on GPU 2D tensor: reduce_all_impl reduces dims on GPU then folds.
    let t = DynTensor::new(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let m = t.max_all().unwrap();
    assert_eq!(
        m.device(),
        Device::metal(),
        "max_all must preserve GPU device"
    );
    assert_eq!(m.dims(), &[] as &[usize]);
    assert_gpu_vals(&m, &[6.0], 1e-4, "max_all 2x3");
}

#[test]
fn test_gpu_min_all_2d() {
    init();
    let t = DynTensor::new(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let m = t.min_all().unwrap();
    assert_eq!(
        m.device(),
        Device::metal(),
        "min_all must preserve GPU device"
    );
    assert_eq!(m.dims(), &[] as &[usize]);
    assert_gpu_vals(&m, &[1.0], 1e-4, "min_all 2x3");
}

#[test]
fn test_gpu_max_all_rank1() {
    init();
    // Rank-1: while loop exits immediately, full [N] transferred to CPU for fold.
    let t = DynTensor::new(&[-5.0, -3.0, -1.0, -4.0], &[4], &Device::metal()).unwrap();
    let m = t.max_all().unwrap();
    assert_eq!(
        m.device(),
        Device::metal(),
        "rank-1 max_all must preserve GPU device"
    );
    assert_eq!(m.dims(), &[] as &[usize], "max_all rank-1 should be scalar");
    assert_gpu_vals(&m, &[-1.0], 1e-4, "max_all rank-1 negative");
}

#[test]
fn test_gpu_min_all_rank1() {
    init();
    let t = DynTensor::new(&[10.0, 2.0, 7.0, 5.0], &[4], &Device::metal()).unwrap();
    let m = t.min_all().unwrap();
    assert_eq!(
        m.device(),
        Device::metal(),
        "rank-1 min_all must preserve GPU device"
    );
    assert_eq!(m.dims(), &[] as &[usize], "min_all rank-1 should be scalar");
    assert_gpu_vals(&m, &[2.0], 1e-4, "min_all rank-1");
}

#[test]
fn test_gpu_max_all_3d() {
    init();
    // 3D: reduce_all_impl loops twice (rank 3 → 2 → 1) then folds.
    let t = DynTensor::new(
        &[1.0, 9.0, 3.0, 4.0, 5.0, 2.0, 7.0, 8.0],
        &[2, 2, 2],
        &Device::metal(),
    )
    .unwrap();
    let m = t.max_all().unwrap();
    assert_eq!(m.device(), Device::metal());
    assert_gpu_vals(&m, &[9.0], 1e-4, "max_all 2x2x2");
}

// -- Floor GPU test -----------------------------------------------------------

#[test]
fn test_gpu_floor() {
    init();
    let t = DynTensor::new(&[1.7, -0.3, 2.0, 3.9, -2.1], &[5], &Device::metal()).unwrap();
    let r = t.floor().unwrap();
    assert_eq!(r.device(), Device::metal(), "floor must stay on GPU");
    assert_gpu_vals(&r, &[1.0, -1.0, 2.0, 3.0, -3.0], 1e-6, "floor");
}

// -- CompareOp::Eq and CompareOp::Ne GPU tests --------------------------------

/// Helper: convert GPU comparison result to Vec<u8> via CPU round-trip.
/// GPU compare returns F32 (0.0/1.0) for where_cond fast-path (#1323).
fn mask_to_u8_vec(mask: &DynTensor) -> Vec<u8> {
    let cpu = mask.to_device(&Device::Cpu).unwrap();
    cpu.to_flat_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(|v| v as u8)
        .collect()
}

#[test]
fn test_gpu_compare_eq() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 2.0], &[4], &Device::metal()).unwrap();
    let r = t.eq(2.0).unwrap();
    assert_eq!(r.device(), Device::metal(), "eq must stay on GPU");
    // GPU compare returns F32 (0.0/1.0) for where_cond fast-path (#1323).
    assert_eq!(r.dtype(), DType::F32);
    assert_eq!(mask_to_u8_vec(&r), vec![0, 1, 0, 1]);
}

#[test]
fn test_gpu_compare_ne() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 2.0], &[4], &Device::metal()).unwrap();
    let r = t.ne(2.0).unwrap();
    assert_eq!(r.device(), Device::metal(), "ne must stay on GPU");
    // GPU compare returns F32 (0.0/1.0) for where_cond fast-path (#1323).
    assert_eq!(r.dtype(), DType::F32);
    assert_eq!(mask_to_u8_vec(&r), vec![1, 0, 1, 0]);
}

// -- Tensor-vs-tensor comparison GPU tests (#1368 AC1) -------------------------

#[test]
fn test_gpu_compare_tensor_ge() {
    init();
    let a = DynTensor::new(&[1.0, 3.0, 2.0, 5.0], &[4], &Device::metal()).unwrap();
    let b = DynTensor::new(&[2.0, 3.0, 4.0, 1.0], &[4], &Device::metal()).unwrap();
    let r = a.broadcast_ge(&b).unwrap();
    assert_eq!(r.device(), Device::metal(), "ge must stay on GPU");
    // 1>=2=F, 3>=3=T, 2>=4=F, 5>=1=T
    assert_eq!(mask_to_u8_vec(&r), vec![0, 1, 0, 1]);
}

#[test]
fn test_gpu_compare_tensor_gt() {
    init();
    let a = DynTensor::new(&[1.0, 3.0, 2.0, 5.0], &[4], &Device::metal()).unwrap();
    let b = DynTensor::new(&[2.0, 3.0, 4.0, 1.0], &[4], &Device::metal()).unwrap();
    let r = a.broadcast_gt(&b).unwrap();
    assert_eq!(r.device(), Device::metal(), "gt must stay on GPU");
    // 1>2=F, 3>3=F, 2>4=F, 5>1=T
    assert_eq!(mask_to_u8_vec(&r), vec![0, 0, 0, 1]);
}

#[test]
fn test_gpu_compare_tensor_lt() {
    init();
    let a = DynTensor::new(&[1.0, 3.0, 2.0, 5.0], &[4], &Device::metal()).unwrap();
    let b = DynTensor::new(&[2.0, 3.0, 4.0, 1.0], &[4], &Device::metal()).unwrap();
    let r = a.broadcast_lt(&b).unwrap();
    assert_eq!(r.device(), Device::metal(), "lt must stay on GPU");
    // 1<2=T, 3<3=F, 2<4=T, 5<1=F
    assert_eq!(mask_to_u8_vec(&r), vec![1, 0, 1, 0]);
}

#[test]
fn test_gpu_compare_tensor_le() {
    init();
    let a = DynTensor::new(&[1.0, 3.0, 2.0, 5.0], &[4], &Device::metal()).unwrap();
    let b = DynTensor::new(&[2.0, 3.0, 4.0, 1.0], &[4], &Device::metal()).unwrap();
    let r = a.broadcast_le(&b).unwrap();
    assert_eq!(r.device(), Device::metal(), "le must stay on GPU");
    // 1<=2=T, 3<=3=T, 2<=4=T, 5<=1=F
    assert_eq!(mask_to_u8_vec(&r), vec![1, 1, 1, 0]);
}

#[test]
fn test_gpu_compare_tensor_eq() {
    init();
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::metal()).unwrap();
    let b = DynTensor::new(&[1.0, 9.0, 3.0, 0.0], &[4], &Device::metal()).unwrap();
    let r = a.broadcast_eq(&b).unwrap();
    assert_eq!(r.device(), Device::metal(), "tensor eq must stay on GPU");
    assert_eq!(mask_to_u8_vec(&r), vec![1, 0, 1, 0]);
}

#[test]
fn test_gpu_compare_tensor_ne() {
    init();
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::metal()).unwrap();
    let b = DynTensor::new(&[1.0, 9.0, 3.0, 0.0], &[4], &Device::metal()).unwrap();
    let r = a.broadcast_ne(&b).unwrap();
    assert_eq!(r.device(), Device::metal(), "tensor ne must stay on GPU");
    assert_eq!(mask_to_u8_vec(&r), vec![0, 1, 0, 1]);
}

// Maximum/minimum tests in `dyn_tensor_metal_ops_tests_maxmin.rs`.

// Triu/tril tests loaded via dyn_tensor_metal.rs (top-level #[path] declaration).
