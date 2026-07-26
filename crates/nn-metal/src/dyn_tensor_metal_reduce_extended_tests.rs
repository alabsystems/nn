#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended GPU reduce tests — last-axis, 4D, single-element, var_keepdim (#2069).
//!
//! Split from `dyn_tensor_metal_reduce_tests.rs` for 500-line compliance.
//! These tests exercise paths NOT covered by the non-last-axis and zero-length
//! tests in the base reduce_tests module.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_gpu_vals, init};

// -- last-axis reduce (direct gpu_reduce path) --------------------------------
// These exercise the native MSL reduce kernel directly (no transpose).

#[test]
fn test_gpu_sum_last_dim_2d() {
    init();
    // [2,3] sum(dim=1) → [2] — last-axis, hits gpu_reduce directly
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let r = t.sum(1).unwrap();
    assert_eq!(
        r.device(),
        Device::metal(),
        "last-axis reduce must stay on GPU"
    );
    assert_eq!(r.dims(), &[2]);
    assert_gpu_vals(&r, &[6.0, 15.0], 1e-4, "sum last_dim 2d");
}

#[test]
fn test_gpu_mean_last_dim_3d() {
    init();
    // [2,2,3] mean(dim=2) → [2,2]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 2, 3], &Device::metal()).unwrap();
    let r = t.mean(2).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[2, 2]);
    // mean of [1,2,3]=2, [4,5,6]=5, [7,8,9]=8, [10,11,12]=11
    assert_gpu_vals(&r, &[2.0, 5.0, 8.0, 11.0], 1e-4, "mean last_dim 3d");
}

#[test]
fn test_gpu_max_last_dim_2d() {
    init();
    // [2,4] max(dim=1) → [2]
    let t = DynTensor::new(
        &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
        &[2, 4],
        &Device::metal(),
    )
    .unwrap();
    let r = t.max(1).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[2]);
    assert_gpu_vals(&r, &[4.0, 9.0], 1e-4, "max last_dim 2d");
}

#[test]
fn test_gpu_min_last_dim_2d() {
    init();
    // [2,4] min(dim=1) → [2]
    let t = DynTensor::new(
        &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
        &[2, 4],
        &Device::metal(),
    )
    .unwrap();
    let r = t.min(1).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[2]);
    assert_gpu_vals(&r, &[1.0, 2.0], 1e-4, "min last_dim 2d");
}

#[test]
fn test_gpu_sum_keepdim_last_dim_3d() {
    init();
    // [2,2,3] sum_keepdim(dim=2) → [2,2,1]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 2, 3], &Device::metal()).unwrap();
    let r = t.sum_keepdim(2).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[2, 2, 1]);
    assert_gpu_vals(
        &r,
        &[6.0, 15.0, 24.0, 33.0],
        1e-4,
        "sum_keepdim last_dim 3d",
    );
}

// -- 4D tensor reduces --------------------------------------------------------

#[test]
fn test_gpu_sum_4d_last_dim() {
    init();
    // [2,2,2,3] sum(dim=3) → [2,2,2]
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 2, 2, 3], &Device::metal()).unwrap();
    let r = t.sum(3).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[2, 2, 2]);
    // sums of consecutive triples: 6, 15, 24, 33, 42, 51, 60, 69
    assert_gpu_vals(
        &r,
        &[6.0, 15.0, 24.0, 33.0, 42.0, 51.0, 60.0, 69.0],
        1e-4,
        "sum 4d last_dim",
    );
}

#[test]
fn test_gpu_mean_4d_dim1() {
    init();
    // [2,3,2,2] mean(dim=1) → [2,2,2] — non-last-axis on 4D
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 2, 2], &Device::metal()).unwrap();
    let gpu_r = t.mean(1).unwrap();
    assert_eq!(gpu_r.device(), Device::metal());
    assert_eq!(gpu_r.dims(), &[2, 2, 2]);

    // Verify against CPU
    let cpu_t = DynTensor::new(&data, &[2, 3, 2, 2], &Device::Cpu).unwrap();
    let cpu_r = cpu_t.mean(1).unwrap();
    let gpu_vals = gpu_r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_r.to_flat_vec::<f32>().unwrap();
    for (i, (g, c)) in gpu_vals.iter().zip(&cpu_vals).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "mean 4d dim1 parity[{i}]: gpu={g}, cpu={c}"
        );
    }
}

#[test]
fn test_gpu_max_4d_dim0() {
    init();
    // [2,2,2,2] max(dim=0) → [2,2,2] — non-last-axis on 4D
    let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 2, 2, 2], &Device::metal()).unwrap();
    let gpu_r = t.max(0).unwrap();
    assert_eq!(gpu_r.device(), Device::metal());
    assert_eq!(gpu_r.dims(), &[2, 2, 2]);
    // max of pairs: (1,9)→9, (2,10)→10, ..., (8,16)→16
    assert_gpu_vals(
        &gpu_r,
        &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        1e-4,
        "max 4d dim0",
    );
}

// -- single-element reduce ----------------------------------------------------

#[test]
fn test_gpu_sum_single_element() {
    init();
    // [1] sum(dim=0) → [] — single-element tensor reduce
    let t = DynTensor::new(&[42.0], &[1], &Device::metal()).unwrap();
    let r = t.sum(0).unwrap();
    assert_eq!(r.device(), Device::metal());
    let vals = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        (vals[0] - 42.0).abs() < 1e-6,
        "single element sum = {}",
        vals[0]
    );
}

#[test]
fn test_gpu_mean_single_element_2d() {
    init();
    // [1,1] mean(dim=0) → [1] — single-element 2D reduce
    let t = DynTensor::new(&[7.5], &[1, 1], &Device::metal()).unwrap();
    let r = t.mean(0).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[1]);
    assert_gpu_vals(&r, &[7.5], 1e-6, "mean single_element 2d");
}

// -- var_keepdim on GPU (decomposed: mean→sub→sqr→mean) -----------------------

#[test]
fn test_gpu_var_keepdim_2d() {
    init();
    // [2,4] var_keepdim(dim=1) → [2,1]
    // Row 0: [1,2,3,4] mean=2.5, var=1.25
    // Row 1: [5,5,5,5] mean=5.0, var=0.0
    let t = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0],
        &[2, 4],
        &Device::metal(),
    )
    .unwrap();
    let r = t.var_keepdim(1).unwrap();
    assert_eq!(r.device(), Device::metal(), "var_keepdim must stay on GPU");
    assert_eq!(r.dims(), &[2, 1]);
    assert_gpu_vals(&r, &[1.25, 0.0], 1e-4, "var_keepdim last_dim 2d");
}

#[test]
fn test_gpu_var_keepdim_non_last_dim() {
    init();
    // [3,2] var_keepdim(dim=0) → [1,2] — non-last-axis variance
    // Col 0: [1,2,3] mean=2, var=((1-2)^2+(2-2)^2+(3-2)^2)/3 = 2/3
    // Col 1: [4,4,4] mean=4, var=0
    let t = DynTensor::new(&[1.0, 4.0, 2.0, 4.0, 3.0, 4.0], &[3, 2], &Device::metal()).unwrap();
    let r = t.var_keepdim(0).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[1, 2]);
    assert_gpu_vals(&r, &[2.0 / 3.0, 0.0], 1e-4, "var_keepdim dim0 2d");
}

#[test]
fn test_gpu_var_keepdim_parity_with_cpu() {
    init();
    // [2,3,4] var_keepdim(dim=2) — CPU/GPU parity
    let data: Vec<f32> = (0..24).map(|x| x as f32 * 0.3 + 1.0).collect();
    let gpu = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();
    let cpu = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();
    let gpu_var = gpu.var_keepdim(2).unwrap();
    let cpu_var = cpu.var_keepdim(2).unwrap();
    assert_eq!(gpu_var.dims(), cpu_var.dims());
    assert_eq!(gpu_var.device(), Device::metal());
    let gpu_vals = gpu_var
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_var.to_flat_vec::<f32>().unwrap();
    for (i, (g, c)) in gpu_vals.iter().zip(&cpu_vals).enumerate() {
        assert!(
            (g - c).abs() < 1e-3,
            "var_keepdim parity[{i}]: gpu={g}, cpu={c}"
        );
    }
}

// -- max/min keepdim non-last-axis (transpose+keepdim+transpose-back) ---------

#[test]
fn test_gpu_max_keepdim_dim0_3d() {
    init();
    // [2,3,4] max_keepdim(dim=0) → [1,3,4]
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let gpu = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();
    let cpu = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();
    let gpu_r = gpu.max_keepdim(0).unwrap();
    let cpu_r = cpu.max_keepdim(0).unwrap();
    assert_eq!(gpu_r.device(), Device::metal());
    assert_eq!(gpu_r.dims(), &[1, 3, 4]);
    assert_eq!(gpu_r.dims(), cpu_r.dims());
    let gv = gpu_r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_r.to_flat_vec::<f32>().unwrap();
    for (i, (g, c)) in gv.iter().zip(&cv).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "max_keepdim dim0 3d parity[{i}]: gpu={g}, cpu={c}"
        );
    }
}

#[test]
fn test_gpu_min_keepdim_dim0_3d() {
    init();
    // [2,3,4] min_keepdim(dim=0) → [1,3,4]
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let gpu = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();
    let cpu = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();
    let gpu_r = gpu.min_keepdim(0).unwrap();
    let cpu_r = cpu.min_keepdim(0).unwrap();
    assert_eq!(gpu_r.device(), Device::metal());
    assert_eq!(gpu_r.dims(), &[1, 3, 4]);
    assert_eq!(gpu_r.dims(), cpu_r.dims());
    let gv = gpu_r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_r.to_flat_vec::<f32>().unwrap();
    for (i, (g, c)) in gv.iter().zip(&cv).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "min_keepdim dim0 3d parity[{i}]: gpu={g}, cpu={c}"
        );
    }
}

#[test]
fn test_gpu_max_keepdim_dim1_3d() {
    init();
    // [2,3,4] max_keepdim(dim=1) → [2,1,4] — mid-axis keepdim
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let gpu = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();
    let cpu = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();
    let gpu_r = gpu.max_keepdim(1).unwrap();
    let cpu_r = cpu.max_keepdim(1).unwrap();
    assert_eq!(gpu_r.device(), Device::metal());
    assert_eq!(gpu_r.dims(), &[2, 1, 4]);
    assert_eq!(gpu_r.dims(), cpu_r.dims());
    let gv = gpu_r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_r.to_flat_vec::<f32>().unwrap();
    for (i, (g, c)) in gv.iter().zip(&cv).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "max_keepdim dim1 3d parity[{i}]: gpu={g}, cpu={c}"
        );
    }
}

#[test]
fn test_gpu_min_keepdim_dim1_3d() {
    init();
    // [2,3,4] min_keepdim(dim=1) → [2,1,4] — mid-axis keepdim
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let gpu = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();
    let cpu = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();
    let gpu_r = gpu.min_keepdim(1).unwrap();
    let cpu_r = cpu.min_keepdim(1).unwrap();
    assert_eq!(gpu_r.device(), Device::metal());
    assert_eq!(gpu_r.dims(), &[2, 1, 4]);
    assert_eq!(gpu_r.dims(), cpu_r.dims());
    let gv = gpu_r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_r.to_flat_vec::<f32>().unwrap();
    for (i, (g, c)) in gv.iter().zip(&cv).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "min_keepdim dim1 3d parity[{i}]: gpu={g}, cpu={c}"
        );
    }
}

// -- last-axis reduce parity with CPU -----------------------------------------

#[test]
fn test_gpu_reduce_last_dim_parity_with_cpu() {
    init();
    // All four reduce ops on last dim must match CPU.
    let data: Vec<f32> = (0..24).map(|x| x as f32 * 0.7 - 5.0).collect();
    let gpu = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();
    let cpu = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();

    for (op_name, gpu_r, cpu_r) in [
        ("sum", gpu.sum(2).unwrap(), cpu.sum(2).unwrap()),
        ("mean", gpu.mean(2).unwrap(), cpu.mean(2).unwrap()),
        ("max", gpu.max(2).unwrap(), cpu.max(2).unwrap()),
        ("min", gpu.min(2).unwrap(), cpu.min(2).unwrap()),
    ] {
        assert_eq!(gpu_r.dims(), cpu_r.dims(), "{op_name}: shape mismatch");
        assert_eq!(
            gpu_r.device(),
            Device::metal(),
            "{op_name}: must stay on GPU"
        );
        let gv = gpu_r
            .to_device(&Device::Cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();
        let cv = cpu_r.to_flat_vec::<f32>().unwrap();
        for (i, (g, c)) in gv.iter().zip(&cv).enumerate() {
            assert!(
                (g - c).abs() < 1e-4,
                "{op_name} last_dim parity[{i}]: gpu={g}, cpu={c}"
            );
        }
    }
}
