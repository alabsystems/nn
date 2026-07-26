// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! BF16 native Metal storage integration tests.
//!
//! Exercises the consumer-facing bf16 pipeline end-to-end through the `nn`
//! crate API — the exact path dvoice uses:
//!
//! 1. Create bf16 CPU tensors via `to_dtype(DType::BF16)`
//! 2. Transfer to Metal GPU via `to_device(&Device::metal())`
//! 3. Run ops (matmul, conv1d, layernorm, linear) on GPU
//! 4. Read results back to CPU and verify correctness
//!
//! This file satisfies #1705 AC3: `cargo test -p nn --test metal_bf16_tests`
//!
//! Run: `cargo test -p nn --test metal_bf16_tests`

use nn::{Conv1dConfig, DType, Device, DynTensor, Module};

/// Initialize Metal GPU backend. Returns true if available.
fn init_gpu() -> bool {
    match nn_metal::MetalBackend::init() {
        Ok(_) => {
            nn_metal::register_metal_dyn_backend();
            true
        }
        Err(_) => false,
    }
}

/// Create a bf16 GPU tensor from f32 data.
fn bf16_gpu(data: &[f32], shape: &[usize]) -> DynTensor {
    let cpu = DynTensor::new(data, shape, &Device::Cpu).unwrap();
    let bf16 = cpu.to_dtype(DType::BF16).unwrap();
    bf16.to_device(&Device::metal()).unwrap()
}

/// Read GPU tensor back to CPU f32 for assertions.
fn to_f32_vec(t: &DynTensor) -> Vec<f32> {
    let cpu = t.to_device(&Device::Cpu).unwrap();
    let f32_cpu = cpu.to_dtype(DType::F32).unwrap();
    f32_cpu.to_flat_vec::<f32>().unwrap()
}

// ---------------------------------------------------------------------------
// AC3 requirement: matmul
// ---------------------------------------------------------------------------

#[test]
fn test_bf16_matmul_gpu() {
    if !init_gpu() {
        return;
    }

    // [2, 3] × [3, 2] -> [2, 2]
    let a = bf16_gpu(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = bf16_gpu(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);

    assert_eq!(a.dtype(), DType::BF16);
    assert_eq!(b.dtype(), DType::BF16);

    let c = a.matmul(&b).expect("bf16 matmul should succeed on GPU");
    assert_eq!(c.dtype(), DType::BF16);
    assert_eq!(c.dims(), &[2, 2]);

    let vals = to_f32_vec(&c);
    // Row 0: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    // Row 1: [4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11]
    assert!(
        (vals[0] - 4.0).abs() < 0.1,
        "c[0,0] = {}, expected 4.0",
        vals[0]
    );
    assert!(
        (vals[1] - 5.0).abs() < 0.1,
        "c[0,1] = {}, expected 5.0",
        vals[1]
    );
    assert!(
        (vals[2] - 10.0).abs() < 0.1,
        "c[1,0] = {}, expected 10.0",
        vals[2]
    );
    assert!(
        (vals[3] - 11.0).abs() < 0.1,
        "c[1,1] = {}, expected 11.0",
        vals[3]
    );
}

#[test]
fn test_bf16_matmul_large_simdgroup() {
    if !init_gpu() {
        return;
    }

    // Large enough to trigger simdgroup GEMM (dims % 8 == 0, M*N >= 16384, K >= 128)
    let m = 128;
    let k = 128;
    let n = 128;
    let a_data: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.1).collect();

    let a = bf16_gpu(&a_data, &[m, k]);
    let b = bf16_gpu(&b_data, &[n, k]);

    // Also compute on CPU for reference
    let a_cpu = DynTensor::new(&a_data, &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[n, k], &Device::Cpu).unwrap();
    let b_cpu_t = b_cpu.t().unwrap();

    let b_t = b.t().unwrap();
    let c_gpu = a.matmul(&b_t).expect("bf16 simdgroup matmul");
    let c_cpu = a_cpu.matmul(&b_cpu_t).unwrap();

    let gpu_vals = to_f32_vec(&c_gpu);
    let cpu_vals = c_cpu.to_flat_vec::<f32>().unwrap();

    // Allow bf16 precision tolerance (bf16 has ~3 decimal digits)
    let max_err: f32 = gpu_vals
        .iter()
        .zip(cpu_vals.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 5.0,
        "bf16 simdgroup matmul max error = {max_err}, expected < 5.0"
    );
}

// ---------------------------------------------------------------------------
// AC3 requirement: conv1d
// ---------------------------------------------------------------------------

#[test]
fn test_bf16_conv1d_gpu() {
    if !init_gpu() {
        return;
    }

    // [batch=1, channels=2, length=8]
    let input_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let input = bf16_gpu(&input_data, &[1, 2, 8]);

    // Build Conv1d with bf16 weights on GPU
    let weight_data: Vec<f32> = vec![0.1; 2 * 2 * 3]; // [out_ch=2, in_ch=2, kernel=3]
    let bias_data: Vec<f32> = vec![0.0; 2];

    let weight = DynTensor::new(&weight_data, &[2, 2, 3], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let bias = DynTensor::new(&bias_data, &[2], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();

    let conv =
        nn::Conv1d::new(weight, Some(bias), Conv1dConfig::default()).expect("conv1d construction");
    let out = conv.forward(&input).expect("bf16 conv1d forward");

    assert_eq!(out.dtype(), DType::BF16);
    // conv1d default: stride=1, padding=0 -> output length = 8 - 3 + 1 = 6
    assert_eq!(out.dims(), &[1, 2, 6]);

    let vals = to_f32_vec(&out);
    assert!(vals.iter().all(|v| v.is_finite()), "all outputs finite");
}

// ---------------------------------------------------------------------------
// AC3 requirement: layernorm
// ---------------------------------------------------------------------------

#[test]
fn test_bf16_layernorm_gpu() {
    if !init_gpu() {
        return;
    }

    // [batch=2, features=4]
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = bf16_gpu(&input_data, &[2, 4]);

    let weight = DynTensor::new(&[1.0f32; 4], &[4], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let bias = DynTensor::new(&[0.0f32; 4], &[4], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();

    let ln = nn::LayerNorm::new(weight, bias, 1e-5).expect("layernorm construction");
    let out = ln.forward(&input).expect("bf16 layernorm forward");

    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.dims(), &[2, 4]);

    let vals = to_f32_vec(&out);
    assert!(vals.iter().all(|v| v.is_finite()), "all outputs finite");

    // LayerNorm with unit weight and zero bias normalizes each row to mean≈0, std≈1
    // Check first row mean is near 0
    let row0_mean: f32 = vals[..4].iter().sum::<f32>() / 4.0;
    assert!(
        row0_mean.abs() < 0.1,
        "layernorm row mean = {row0_mean}, expected ≈0"
    );
}

// ---------------------------------------------------------------------------
// AC3 requirement: linear
// ---------------------------------------------------------------------------

#[test]
fn test_bf16_linear_gpu() {
    if !init_gpu() {
        return;
    }

    // [batch=2, in_features=4] -> [batch=2, out_features=3]
    let input_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let input = bf16_gpu(&input_data, &[2, 4]);

    // Identity-like weight: [out=3, in=4], first 3 columns are identity
    let weight_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, // row 1
        0.0, 0.0, 1.0, 0.0, // row 2
    ];
    let weight = DynTensor::new(&weight_data, &[3, 4], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();

    let linear = nn::Linear::new(weight, None).expect("linear construction");
    let out = linear.forward(&input).expect("bf16 linear forward");

    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.dims(), &[2, 3]);

    let vals = to_f32_vec(&out);
    // Input [1,0,0,0] through identity-like weight -> [1,0,0]
    assert!(
        (vals[0] - 1.0).abs() < 0.01,
        "linear out[0,0] = {}, expected 1.0",
        vals[0]
    );
    assert!(
        vals[1].abs() < 0.01,
        "linear out[0,1] = {}, expected 0.0",
        vals[1]
    );
    // Input [0,1,0,0] through identity-like weight -> [0,1,0]
    assert!(
        vals[3].abs() < 0.01,
        "linear out[1,0] = {}, expected 0.0",
        vals[3]
    );
    assert!(
        (vals[4] - 1.0).abs() < 0.01,
        "linear out[1,1] = {}, expected 1.0",
        vals[4]
    );
}

// ---------------------------------------------------------------------------
// Additional coverage: bf16 GPU round-trip preserves dtype
// ---------------------------------------------------------------------------

#[test]
fn test_bf16_dtype_preserved_through_pipeline() {
    if !init_gpu() {
        return;
    }

    // Create bf16 CPU tensor
    let cpu = DynTensor::new(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let bf16_cpu = cpu.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16_cpu.dtype(), DType::BF16);

    // Transfer to GPU
    let gpu = bf16_cpu.to_device(&Device::metal()).unwrap();
    assert_eq!(gpu.dtype(), DType::BF16, "GPU tensor preserves bf16 dtype");

    // Run ops chain: add -> relu -> sum
    let result = gpu
        .add(&gpu)
        .unwrap()
        .relu()
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    assert_eq!(
        result.dtype(),
        DType::BF16,
        "dtype preserved through op chain"
    );

    // Transfer back to CPU
    let back_cpu = result.to_device(&Device::Cpu).unwrap();
    assert_eq!(back_cpu.dtype(), DType::BF16, "dtype preserved on readback");
}

#[test]
fn test_bf16_softmax_gpu() {
    if !init_gpu() {
        return;
    }

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = bf16_gpu(&data, &[2, 3]);

    let out = nn::softmax_last_dim(&input).expect("bf16 softmax");
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.dims(), &[2, 3]);

    let vals = to_f32_vec(&out);
    // Each row should sum to ~1.0
    let row0_sum: f32 = vals[..3].iter().sum();
    let row1_sum: f32 = vals[3..].iter().sum();
    assert!(
        (row0_sum - 1.0).abs() < 0.05,
        "softmax row0 sum = {row0_sum}"
    );
    assert!(
        (row1_sum - 1.0).abs() < 0.05,
        "softmax row1 sum = {row1_sum}"
    );
}

#[test]
fn test_bf16_rmsnorm_gpu() {
    if !init_gpu() {
        return;
    }

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = bf16_gpu(&data, &[2, 4]);

    let weight = DynTensor::new(&[1.0f32; 4], &[4], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();

    let rms = nn::RmsNorm::new(weight, 1e-5).expect("rmsnorm construction");
    let out = rms.forward(&input).expect("bf16 rmsnorm forward");

    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.dims(), &[2, 4]);

    let vals = to_f32_vec(&out);
    assert!(vals.iter().all(|v| v.is_finite()), "all outputs finite");
}
