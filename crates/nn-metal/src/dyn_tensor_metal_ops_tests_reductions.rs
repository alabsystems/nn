#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU reduction op tests — sum_all, mean_all, reduce_sum, reduce_max, reduce_min.
//!
//! Extracted from `dyn_tensor_metal_ops_tests.rs` for 500-line compliance (#1306).

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_gpu_vals, init};

// -- sum_all / mean_all -------------------------------------------------------

#[test]
fn test_gpu_sum_all_reduces_on_gpu() {
    init();
    // sum_all reduces on GPU, result stays on GPU device (#1172).
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let s = t.sum_all().unwrap();
    assert_eq!(
        s.device(),
        Device::metal(),
        "sum_all must preserve GPU device"
    );
    assert_eq!(s.dims(), &[] as &[usize]);
    assert_gpu_vals(&s, &[21.0], 1e-4, "sum_all 2x3");
}

#[test]
fn test_gpu_mean_all_reduces_on_gpu() {
    init();
    // mean_all reduces on GPU, result stays on GPU device (#1172).
    let t = DynTensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2], &Device::metal()).unwrap();
    let m = t.mean_all().unwrap();
    assert_eq!(
        m.device(),
        Device::metal(),
        "mean_all must preserve GPU device"
    );
    assert_eq!(m.dims(), &[] as &[usize]);
    assert_gpu_vals(&m, &[5.0], 1e-4, "mean_all 2x2");
}

#[test]
fn test_gpu_sum_all_result_usable_in_gpu_ops() {
    init();
    // The primary use case from #1172: sum_all result can be used directly
    // in follow-up GPU operations without "mixed device" errors.
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let s = t.sum_all().unwrap(); // scalar []: should be 6.0 on GPU
    assert_eq!(s.device(), Device::metal());
    // broadcast_div: [3] / [] should work without mixed-device error
    let normed = t.broadcast_div(&s).unwrap();
    assert_eq!(
        normed.device(),
        Device::metal(),
        "broadcast_div result must stay on GPU"
    );
    assert_gpu_vals(
        &normed,
        &[1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0],
        1e-4,
        "sum_all→div",
    );
}

#[test]
fn test_gpu_mean_all_result_usable_in_gpu_ops() {
    init();
    // mean_all result can be used in follow-up GPU ops (#1172).
    let t = DynTensor::new(&[2.0, 4.0, 6.0, 8.0], &[4], &Device::metal()).unwrap();
    let m = t.mean_all().unwrap(); // scalar []: should be 5.0 on GPU
    assert_eq!(m.device(), Device::metal());
    // broadcast_sub: [4] - [] should work without mixed-device error
    let centered = t.broadcast_sub(&m).unwrap();
    assert_eq!(
        centered.device(),
        Device::metal(),
        "broadcast_sub result must stay on GPU"
    );
    assert_gpu_vals(&centered, &[-3.0, -1.0, 1.0, 3.0], 1e-4, "mean_all→sub");
}

// -- rank-0 reduce guard (rank-1 input → scalar) ----------------------------

#[test]
fn test_gpu_sum_all_rank1_input() {
    init();
    // Rank-1 GPU tensor: sum_all reduces to rank 0 via CPU final sum,
    // then transfers back to GPU. Exercises `while t.rank() > 1` early-exit.
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let s = t.sum_all().unwrap();
    assert_eq!(
        s.device(),
        Device::metal(),
        "rank-1 sum_all must preserve GPU device"
    );
    assert_eq!(s.dims(), &[] as &[usize], "sum_all rank-1 should be scalar");
    assert_gpu_vals(&s, &[6.0], 1e-4, "sum_all rank-1");
}

#[test]
fn test_gpu_mean_all_rank1_input() {
    init();
    let t = DynTensor::new(&[2.0, 4.0, 6.0], &[3], &Device::metal()).unwrap();
    let m = t.mean_all().unwrap();
    assert_eq!(
        m.device(),
        Device::metal(),
        "rank-1 mean_all must preserve GPU device"
    );
    assert_eq!(
        m.dims(),
        &[] as &[usize],
        "mean_all rank-1 should be scalar"
    );
    assert_gpu_vals(&m, &[4.0], 1e-4, "mean_all rank-1");
}

#[test]
fn test_gpu_reduce_sum_rank1_keepdim_false() {
    init();
    // Direct reduce_impl on rank-1 with keepdim=false. This exercises the
    // `if reduce_shape.is_empty()` guard in gpu_reduce that produces [1] output.
    let t = DynTensor::new(&[10.0, 20.0, 30.0], &[3], &Device::metal()).unwrap();
    let r = t.sum(0).unwrap();
    let val = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        (val[0] - 60.0).abs() < 1e-4,
        "sum(dim=0) should be 60.0, got {}",
        val[0]
    );
}

#[test]
fn test_gpu_reduce_max_rank1() {
    init();
    let t = DynTensor::new(&[3.0, 1.0, 5.0, 2.0], &[4], &Device::metal()).unwrap();
    let r = t.max_keepdim(0).unwrap();
    let val = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        (val[0] - 5.0).abs() < 1e-6,
        "max should be 5.0, got {}",
        val[0]
    );
    assert_eq!(r.dims(), &[1], "max_keepdim(0) on rank-1 should be [1]");
}

#[test]
fn test_gpu_reduce_min_rank1() {
    init();
    let t = DynTensor::new(&[3.0, 1.0, 5.0, 2.0], &[4], &Device::metal()).unwrap();
    let r = t.min_keepdim(0).unwrap();
    let val = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        (val[0] - 1.0).abs() < 1e-6,
        "min should be 1.0, got {}",
        val[0]
    );
    assert_eq!(r.dims(), &[1], "min_keepdim(0) on rank-1 should be [1]");
}
