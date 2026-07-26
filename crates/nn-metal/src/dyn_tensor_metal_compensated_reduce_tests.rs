#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Differential tests for Kahan compensated GPU reductions (#1814).
//!
//! Verifies that `sum_compensated` / `mean_compensated` produce results
//! closer to the f64 reference than the naive `sum` / `mean` on GPU.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::init;

/// Compute f64-precision reference sum.
fn f64_sum(data: &[f32]) -> f64 {
    data.iter().map(|&v| f64::from(v)).sum::<f64>()
}

/// Compute f64-precision reference mean.
fn f64_mean(data: &[f32]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = data.len() as f64;
    f64_sum(data) / n
}

/// Build a pathological dataset: large constant + small perturbations.
///
/// Naive summation of `1e6 + 1e-2` repeated N times loses ~8 digits of
/// precision on the perturbation component. Kahan compensated summation
/// recovers most of this.
fn pathological_data(n: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if i % 2 == 0 {
            data.push(1e6);
        } else {
            data.push(1e-2);
        }
    }
    data
}

/// Extract scalar f32 from a GPU tensor.
fn scalar_f32(t: &DynTensor) -> f32 {
    t.to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()[0]
}

// -- Sum tests ----------------------------------------------------------------

#[test]
fn test_compensated_sum_last_axis_closer_to_f64() {
    init();
    let n = 4096;
    let data = pathological_data(n);
    let ref_sum = f64_sum(&data);

    let t = DynTensor::new(&data, &[n], &Device::metal()).unwrap();

    let naive = t.sum(0).unwrap();
    let kahan = t.sum_compensated(0).unwrap();

    let naive_val = f64::from(scalar_f32(&naive));
    let kahan_val = f64::from(scalar_f32(&kahan));

    let naive_err = (naive_val - ref_sum).abs();
    let kahan_err = (kahan_val - ref_sum).abs();

    assert!(
        kahan_err <= naive_err,
        "Kahan should be at least as precise as naive: \
         kahan_err={kahan_err:.6e}, naive_err={naive_err:.6e}, ref={ref_sum:.6e}"
    );
}

#[test]
fn test_compensated_sum_keepdim() {
    init();
    let n = 2048;
    let data = pathological_data(n);
    let ref_sum = f64_sum(&data);

    let t = DynTensor::new(&data, &[n], &Device::metal()).unwrap();

    let kahan = t.sum_compensated_keepdim(0).unwrap();
    assert_eq!(kahan.dims(), &[1], "keepdim should produce [1]");

    let kahan_val = f64::from(scalar_f32(&kahan));
    let kahan_err = (kahan_val - ref_sum).abs();
    let rel_err = kahan_err / ref_sum.abs();

    assert!(
        rel_err < 1e-5,
        "Kahan sum keepdim relative error too large: {rel_err:.6e}"
    );
}

#[test]
fn test_compensated_mean_last_axis() {
    init();
    let n = 4096;
    let data = pathological_data(n);
    let ref_mean = f64_mean(&data);

    let t = DynTensor::new(&data, &[n], &Device::metal()).unwrap();

    let naive = t.mean(0).unwrap();
    let kahan = t.mean_compensated(0).unwrap();

    let naive_val = f64::from(scalar_f32(&naive));
    let kahan_val = f64::from(scalar_f32(&kahan));

    let naive_err = (naive_val - ref_mean).abs();
    let kahan_err = (kahan_val - ref_mean).abs();

    assert!(
        kahan_err <= naive_err,
        "Kahan mean should be at least as precise as naive: \
         kahan_err={kahan_err:.6e}, naive_err={naive_err:.6e}"
    );
}

#[test]
fn test_compensated_mean_keepdim() {
    init();
    let n = 2048;
    let data = pathological_data(n);
    let ref_mean = f64_mean(&data);

    let t = DynTensor::new(&data, &[n], &Device::metal()).unwrap();

    let kahan = t.mean_compensated_keepdim(0).unwrap();
    assert_eq!(kahan.dims(), &[1]);

    let kahan_val = f64::from(scalar_f32(&kahan));
    let kahan_err = (kahan_val - ref_mean).abs();
    let rel_err = kahan_err / ref_mean.abs();

    assert!(
        rel_err < 1e-5,
        "Kahan mean keepdim relative error too large: {rel_err:.6e}"
    );
}

// -- Non-last-axis tests (transpose path) ------------------------------------

#[test]
fn test_compensated_sum_dim0_2d() {
    init();
    let rows = 1024;
    let cols = 4;
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            if r % 2 == 0 {
                data.push(1e6 + (c as f32) * 0.1);
            } else {
                data.push(1e-2 + (c as f32) * 0.001);
            }
        }
    }

    // f64 reference: sum along dim 0 → [cols]
    let mut ref_sums = vec![0.0f64; cols];
    for r in 0..rows {
        for c in 0..cols {
            ref_sums[c] += f64::from(data[r * cols + c]);
        }
    }

    let t = DynTensor::new(&data, &[rows, cols], &Device::metal()).unwrap();
    let kahan = t.sum_compensated(0).unwrap();

    assert_eq!(kahan.dims(), &[cols]);

    let kahan_vals = kahan
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    for c in 0..cols {
        let err = (f64::from(kahan_vals[c]) - ref_sums[c]).abs();
        let rel = err / ref_sums[c].abs();
        assert!(
            rel < 1e-5,
            "col {c}: Kahan sum dim0 rel_err={rel:.6e} (val={}, ref={})",
            kahan_vals[c],
            ref_sums[c]
        );
    }
}

#[test]
fn test_compensated_mean_dim0_2d() {
    init();
    let rows = 1024;
    let cols = 4;
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            if r % 2 == 0 {
                data.push(1e5 + (c as f32));
            } else {
                data.push(1e-3);
            }
        }
    }

    // f64 reference: mean along dim 0 → [cols]
    let mut ref_sums = vec![0.0f64; cols];
    for r in 0..rows {
        for c in 0..cols {
            ref_sums[c] += f64::from(data[r * cols + c]);
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let ref_means: Vec<f64> = ref_sums.iter().map(|s| s / rows as f64).collect();

    let t = DynTensor::new(&data, &[rows, cols], &Device::metal()).unwrap();
    let kahan = t.mean_compensated(0).unwrap();

    assert_eq!(kahan.dims(), &[cols]);

    let kahan_vals = kahan
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    for c in 0..cols {
        let err = (f64::from(kahan_vals[c]) - ref_means[c]).abs();
        let rel = err / ref_means[c].abs();
        assert!(rel < 1e-5, "col {c}: Kahan mean dim0 rel_err={rel:.6e}");
    }
}

// -- GPU stays on device test -------------------------------------------------

#[test]
fn test_compensated_sum_stays_on_gpu() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::metal()).unwrap();
    let r = t.sum_compensated(0).unwrap();
    assert_eq!(r.device(), Device::metal(), "result must stay on GPU");
    assert_eq!(r.dims(), &[] as &[usize]);
    let val = scalar_f32(&r);
    assert!((val - 10.0).abs() < 1e-4, "sum should be 10.0, got {val}");
}

#[test]
fn test_compensated_mean_stays_on_gpu() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::metal()).unwrap();
    let r = t.mean_compensated(0).unwrap();
    assert_eq!(r.device(), Device::metal(), "result must stay on GPU");
    assert_eq!(r.dims(), &[] as &[usize]);
    let val = scalar_f32(&r);
    assert!((val - 2.5).abs() < 1e-4, "mean should be 2.5, got {val}");
}

// -- 3D non-last-axis (exercises permutation path in gpu_reduce_compensated) --

#[test]
fn test_compensated_sum_dim0_3d() {
    init();
    // [2,3,4] sum_compensated(dim=0) → [3,4]
    // Exercises the permutation path (perm.remove(dim); perm.push(dim))
    // in gpu_reduce_compensated for dim < new_rank.
    let data: Vec<f32> = (1..=24).map(|x| x as f32 * 100.0 + 0.01).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();

    let gpu_r = t.sum_compensated(0).unwrap();
    assert_eq!(gpu_r.device(), Device::metal());
    assert_eq!(gpu_r.dims(), &[3, 4]);

    // Verify against CPU sum
    let cpu_t = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();
    let cpu_r = cpu_t.sum(0).unwrap();
    let gv = gpu_r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_r.to_flat_vec::<f32>().unwrap();
    for (i, (g, c)) in gv.iter().zip(&cv).enumerate() {
        assert!(
            (g - c).abs() < 1e-2,
            "compensated sum dim0 3d parity[{i}]: gpu={g}, cpu={c}"
        );
    }
}

#[test]
fn test_compensated_mean_dim1_3d() {
    init();
    // [2,3,4] mean_compensated(dim=1) → [2,4]
    // Mid-axis exercises the non-last-axis transpose+reduce+permute path.
    let data: Vec<f32> = (1..=24).map(|x| x as f32 * 10.0 + 0.001).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();

    let gpu_r = t.mean_compensated(1).unwrap();
    assert_eq!(gpu_r.device(), Device::metal());
    assert_eq!(gpu_r.dims(), &[2, 4]);

    // Verify against CPU mean
    let cpu_t = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();
    let cpu_r = cpu_t.mean(1).unwrap();
    let gv = gpu_r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_r.to_flat_vec::<f32>().unwrap();
    for (i, (g, c)) in gv.iter().zip(&cv).enumerate() {
        assert!(
            (g - c).abs() < 1e-2,
            "compensated mean dim1 3d parity[{i}]: gpu={g}, cpu={c}"
        );
    }
}

// -- Keepdim on multi-dimensional tensors ------------------------------------

#[test]
fn test_compensated_sum_keepdim_2d() {
    init();
    // [4,3] sum_compensated_keepdim(dim=1) → [4,1]
    let data: Vec<f32> = (1..=12).map(|x| x as f32 * 1e5 + 0.01).collect();
    let t = DynTensor::new(&data, &[4, 3], &Device::metal()).unwrap();

    let gpu_r = t.sum_compensated_keepdim(1).unwrap();
    assert_eq!(gpu_r.device(), Device::metal());
    assert_eq!(gpu_r.dims(), &[4, 1]);

    // Verify against CPU sum_keepdim
    let cpu_t = DynTensor::new(&data, &[4, 3], &Device::Cpu).unwrap();
    let cpu_r = cpu_t.sum_keepdim(1).unwrap();
    let gv = gpu_r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_r.to_flat_vec::<f32>().unwrap();
    for (i, (g, c)) in gv.iter().zip(&cv).enumerate() {
        assert!(
            (g - c).abs() < 1.0,
            "compensated sum_keepdim 2d parity[{i}]: gpu={g}, cpu={c}"
        );
    }
}

#[test]
fn test_compensated_mean_keepdim_dim0_2d() {
    init();
    // [4,3] mean_compensated_keepdim(dim=0) → [1,3] — non-last-axis keepdim
    let data: Vec<f32> = (1..=12).map(|x| x as f32 * 1e4 + 0.001).collect();
    let t = DynTensor::new(&data, &[4, 3], &Device::metal()).unwrap();

    let gpu_r = t.mean_compensated_keepdim(0).unwrap();
    assert_eq!(gpu_r.device(), Device::metal());
    assert_eq!(gpu_r.dims(), &[1, 3]);

    // Verify against CPU mean_keepdim
    let cpu_t = DynTensor::new(&data, &[4, 3], &Device::Cpu).unwrap();
    let cpu_r = cpu_t.mean_keepdim(0).unwrap();
    let gv = gpu_r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_r.to_flat_vec::<f32>().unwrap();
    for (i, (g, c)) in gv.iter().zip(&cv).enumerate() {
        assert!(
            (g - c).abs() < 1.0,
            "compensated mean_keepdim dim0 2d parity[{i}]: gpu={g}, cpu={c}"
        );
    }
}

// -- Zero-length axis tests (defense-in-depth) --------------------------------

#[test]
fn test_compensated_mean_zero_length_returns_error() {
    init();
    // Mean over zero-length axis is undefined (0/0) — must return error.
    let t = DynTensor::new(&[], &[2, 0, 3], &Device::Cpu).unwrap();
    let err = t.mean_compensated(1).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("ero") && msg.contains("ength"),
        "expected ZeroLengthDimension error, got: {msg}"
    );
}

#[test]
fn test_compensated_sum_zero_length_returns_error() {
    init();
    // Compensated sum over zero-length axis returns error (Kahan loop cannot
    // execute on 0 elements). Unlike standard `sum` which returns 0.0 via
    // ndarray, the compensated path rejects empty axes for both Sum and Mean.
    let t = DynTensor::new(&[], &[2, 0, 3], &Device::Cpu).unwrap();
    let err = t.sum_compensated(1).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("ero") && msg.contains("ength"),
        "expected ZeroLengthDimension error, got: {msg}"
    );
}
