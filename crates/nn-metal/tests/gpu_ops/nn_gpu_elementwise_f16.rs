// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU elementwise op tests for BF16/F16 dtypes (#3230 retype_kernel).
//!
//! Verifies that unary (abs, neg, sqr, floor, round) and binary
//! (sub, div, maximum, minimum) Elementwise ops dispatch correctly for
//! BF16/F16 tensors via the `retype_kernel` pattern, ensuring MSL buffer
//! declarations match the actual `half*` GPU data.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

fn init() {
    gpu_init();
}

fn assert_close(gpu: &DynTensor, cpu: &DynTensor, label: &str) {
    // BF16: 7-bit mantissa, eps ≈ 0.0078. F16: 10-bit mantissa, eps ≈ 9.8e-4.
    // Transcendental ops (log, fract, sin) amplify rounding at BF16 boundaries.
    let tol = if gpu.dtype() == DType::BF16 {
        1e-2
    } else {
        2e-3
    };
    assert_gpu_cpu_close(gpu, cpu, tol, label);
}

fn gpu_tensor(data: &[f32], shape: &[usize], dtype: DType) -> DynTensor {
    let t = DynTensor::new(data, shape, &Device::metal()).unwrap();
    t.to_dtype(dtype).unwrap()
}

fn cpu_tensor(data: &[f32], shape: &[usize], dtype: DType) -> DynTensor {
    let t = DynTensor::new(data, shape, &Device::Cpu).unwrap();
    t.to_dtype(dtype).unwrap()
}

// -- Unary: abs BF16 ---------------------------------------------------------

#[test]
fn test_abs_bf16() {
    init();
    let data = vec![-3.0, -1.5, 0.0, 1.5, 3.0];
    let y_cpu = cpu_tensor(&data, &[5], DType::BF16).abs().unwrap();
    let y_gpu = gpu_tensor(&data, &[5], DType::BF16).abs().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "abs_bf16");
}

// -- Unary: neg BF16 ---------------------------------------------------------

#[test]
fn test_neg_bf16() {
    init();
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let y_cpu = cpu_tensor(&data, &[6], DType::BF16).neg().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::BF16).neg().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "neg_bf16");
}

// -- Unary: sqr BF16 ---------------------------------------------------------

#[test]
fn test_sqr_bf16() {
    init();
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let y_cpu = cpu_tensor(&data, &[6], DType::BF16).sqr().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::BF16).sqr().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "sqr_bf16");
}

// -- Unary: floor BF16 -------------------------------------------------------

#[test]
fn test_floor_bf16() {
    init();
    let data = vec![-2.7, -1.3, 0.0, 0.9, 2.5, 3.1];
    let y_cpu = cpu_tensor(&data, &[6], DType::BF16).floor().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::BF16).floor().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "floor_bf16");
}

// -- Unary: round BF16 -------------------------------------------------------

#[test]
fn test_round_bf16() {
    init();
    let data = vec![-2.7, -1.3, 0.0, 0.5, 2.5, 3.7];
    let y_cpu = cpu_tensor(&data, &[6], DType::BF16).round().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::BF16).round().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "round_bf16");
}

// -- Binary: sub BF16 --------------------------------------------------------

#[test]
fn test_sub_bf16() {
    init();
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![0.5, 1.0, 1.5, 2.0];
    let y_cpu = cpu_tensor(&a, &[4], DType::BF16)
        .sub(&cpu_tensor(&b, &[4], DType::BF16))
        .unwrap();
    let y_gpu = gpu_tensor(&a, &[4], DType::BF16)
        .sub(&gpu_tensor(&b, &[4], DType::BF16))
        .unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "sub_bf16");
}

// -- Binary: div BF16 --------------------------------------------------------

#[test]
fn test_div_bf16() {
    init();
    let a = vec![2.0, 4.0, 6.0, 8.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let y_cpu = cpu_tensor(&a, &[4], DType::BF16)
        .div(&cpu_tensor(&b, &[4], DType::BF16))
        .unwrap();
    let y_gpu = gpu_tensor(&a, &[4], DType::BF16)
        .div(&gpu_tensor(&b, &[4], DType::BF16))
        .unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "div_bf16");
}

// -- Binary: maximum BF16 ----------------------------------------------------

#[test]
fn test_maximum_bf16() {
    init();
    let a = vec![-1.0, 2.0, 0.5, 4.0];
    let b = vec![1.0, -2.0, 0.5, 3.0];
    let y_cpu = cpu_tensor(&a, &[4], DType::BF16)
        .maximum(&cpu_tensor(&b, &[4], DType::BF16))
        .unwrap();
    let y_gpu = gpu_tensor(&a, &[4], DType::BF16)
        .maximum(&gpu_tensor(&b, &[4], DType::BF16))
        .unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "maximum_bf16");
}

// -- Binary: minimum BF16 ----------------------------------------------------

#[test]
fn test_minimum_bf16() {
    init();
    let a = vec![-1.0, 2.0, 0.5, 4.0];
    let b = vec![1.0, -2.0, 0.5, 3.0];
    let y_cpu = cpu_tensor(&a, &[4], DType::BF16)
        .minimum(&cpu_tensor(&b, &[4], DType::BF16))
        .unwrap();
    let y_gpu = gpu_tensor(&a, &[4], DType::BF16)
        .minimum(&gpu_tensor(&b, &[4], DType::BF16))
        .unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "minimum_bf16");
}

// -- F16 variants ------------------------------------------------------------

#[test]
fn test_abs_f16() {
    init();
    let data = vec![-3.0, -1.5, 0.0, 1.5, 3.0];
    let y_cpu = cpu_tensor(&data, &[5], DType::F16).abs().unwrap();
    let y_gpu = gpu_tensor(&data, &[5], DType::F16).abs().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "abs_f16");
}

#[test]
fn test_neg_f16() {
    init();
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let y_cpu = cpu_tensor(&data, &[6], DType::F16).neg().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::F16).neg().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "neg_f16");
}

#[test]
fn test_sqr_f16() {
    init();
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let y_cpu = cpu_tensor(&data, &[6], DType::F16).sqr().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::F16).sqr().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "sqr_f16");
}

#[test]
fn test_sub_f16() {
    init();
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![0.5, 1.0, 1.5, 2.0];
    let y_cpu = cpu_tensor(&a, &[4], DType::F16)
        .sub(&cpu_tensor(&b, &[4], DType::F16))
        .unwrap();
    let y_gpu = gpu_tensor(&a, &[4], DType::F16)
        .sub(&gpu_tensor(&b, &[4], DType::F16))
        .unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "sub_f16");
}

#[test]
fn test_maximum_f16() {
    init();
    let a = vec![-1.0, 2.0, 0.5, 4.0];
    let b = vec![1.0, -2.0, 0.5, 3.0];
    let y_cpu = cpu_tensor(&a, &[4], DType::F16)
        .maximum(&cpu_tensor(&b, &[4], DType::F16))
        .unwrap();
    let y_gpu = gpu_tensor(&a, &[4], DType::F16)
        .maximum(&gpu_tensor(&b, &[4], DType::F16))
        .unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "maximum_f16");
}

// -- Transcendental unary: exp BF16 ------------------------------------------

#[test]
fn test_exp_bf16() {
    init();
    let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let y_cpu = cpu_tensor(&data, &[6], DType::BF16).exp().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::BF16).exp().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "exp_bf16");
}

#[test]
fn test_sqrt_bf16() {
    init();
    let data = vec![0.0, 0.25, 1.0, 4.0, 9.0, 16.0];
    let y_cpu = cpu_tensor(&data, &[6], DType::BF16).sqrt().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::BF16).sqrt().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "sqrt_bf16");
}

#[test]
fn test_sin_bf16() {
    init();
    let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let y_cpu = cpu_tensor(&data, &[6], DType::BF16).sin().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::BF16).sin().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "sin_bf16");
}

#[test]
fn test_cos_bf16() {
    init();
    let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let y_cpu = cpu_tensor(&data, &[6], DType::BF16).cos().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::BF16).cos().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "cos_bf16");
}

#[test]
fn test_log_bf16() {
    init();
    let data = vec![0.1, 0.5, 1.0, 2.0, 10.0];
    let y_cpu = cpu_tensor(&data, &[5], DType::BF16).log().unwrap();
    let y_gpu = gpu_tensor(&data, &[5], DType::BF16).log().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "log_bf16");
}

#[test]
fn test_recip_bf16() {
    init();
    let data = vec![0.5, 1.0, 2.0, 4.0, 10.0];
    let y_cpu = cpu_tensor(&data, &[5], DType::BF16).recip().unwrap();
    let y_gpu = gpu_tensor(&data, &[5], DType::BF16).recip().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "recip_bf16");
}

#[test]
fn test_fract_bf16() {
    init();
    let data = vec![-2.7, -1.3, 0.0, 0.9, 2.5, 3.1];
    let y_cpu = cpu_tensor(&data, &[6], DType::BF16).fract().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::BF16).fract().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "fract_bf16");
}

// -- Binary: atan2 BF16 ------------------------------------------------------

#[test]
fn test_atan2_bf16() {
    init();
    let a = vec![-1.0, 0.0, 1.0, 2.0];
    let b = vec![1.0, 1.0, -1.0, 0.5];
    let y_cpu = cpu_tensor(&a, &[4], DType::BF16)
        .atan2(&cpu_tensor(&b, &[4], DType::BF16))
        .unwrap();
    let y_gpu = gpu_tensor(&a, &[4], DType::BF16)
        .atan2(&gpu_tensor(&b, &[4], DType::BF16))
        .unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "atan2_bf16");
}

// -- Binary: div F16 ---------------------------------------------------------

#[test]
fn test_div_f16() {
    init();
    let a = vec![2.0, 4.0, 6.0, 8.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let y_cpu = cpu_tensor(&a, &[4], DType::F16)
        .div(&cpu_tensor(&b, &[4], DType::F16))
        .unwrap();
    let y_gpu = gpu_tensor(&a, &[4], DType::F16)
        .div(&gpu_tensor(&b, &[4], DType::F16))
        .unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "div_f16");
}

// -- Binary: minimum F16 ----------------------------------------------------

#[test]
fn test_minimum_f16() {
    init();
    let a = vec![-1.0, 2.0, 0.5, 4.0];
    let b = vec![1.0, -2.0, 0.5, 3.0];
    let y_cpu = cpu_tensor(&a, &[4], DType::F16)
        .minimum(&cpu_tensor(&b, &[4], DType::F16))
        .unwrap();
    let y_gpu = gpu_tensor(&a, &[4], DType::F16)
        .minimum(&gpu_tensor(&b, &[4], DType::F16))
        .unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "minimum_f16");
}

// -- F16 transcendental unary ------------------------------------------------

#[test]
fn test_exp_f16() {
    init();
    let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let y_cpu = cpu_tensor(&data, &[6], DType::F16).exp().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::F16).exp().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "exp_f16");
}

#[test]
fn test_sqrt_f16() {
    init();
    let data = vec![0.0, 0.25, 1.0, 4.0, 9.0, 16.0];
    let y_cpu = cpu_tensor(&data, &[6], DType::F16).sqrt().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::F16).sqrt().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "sqrt_f16");
}

#[test]
fn test_sin_f16() {
    init();
    let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let y_cpu = cpu_tensor(&data, &[6], DType::F16).sin().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::F16).sin().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "sin_f16");
}

#[test]
fn test_cos_f16() {
    init();
    let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let y_cpu = cpu_tensor(&data, &[6], DType::F16).cos().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::F16).cos().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "cos_f16");
}

#[test]
fn test_log_f16() {
    init();
    let data = vec![0.1, 0.5, 1.0, 2.0, 10.0];
    let y_cpu = cpu_tensor(&data, &[5], DType::F16).log().unwrap();
    let y_gpu = gpu_tensor(&data, &[5], DType::F16).log().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "log_f16");
}

#[test]
fn test_recip_f16() {
    init();
    let data = vec![0.5, 1.0, 2.0, 4.0, 10.0];
    let y_cpu = cpu_tensor(&data, &[5], DType::F16).recip().unwrap();
    let y_gpu = gpu_tensor(&data, &[5], DType::F16).recip().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "recip_f16");
}

#[test]
fn test_fract_f16() {
    init();
    let data = vec![-2.7, -1.3, 0.0, 0.9, 2.5, 3.1];
    let y_cpu = cpu_tensor(&data, &[6], DType::F16).fract().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::F16).fract().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "fract_f16");
}

#[test]
fn test_floor_f16() {
    init();
    let data = vec![-2.7, -1.3, 0.0, 0.9, 2.5, 3.1];
    let y_cpu = cpu_tensor(&data, &[6], DType::F16).floor().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::F16).floor().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "floor_f16");
}

#[test]
fn test_round_f16() {
    init();
    let data = vec![-2.7, -1.3, 0.0, 0.5, 2.5, 3.7];
    let y_cpu = cpu_tensor(&data, &[6], DType::F16).round().unwrap();
    let y_gpu = gpu_tensor(&data, &[6], DType::F16).round().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "round_f16");
}

// -- Binary: atan2 F16 -------------------------------------------------------

#[test]
fn test_atan2_f16() {
    init();
    let a = vec![-1.0, 0.0, 1.0, 2.0];
    let b = vec![1.0, 1.0, -1.0, 0.5];
    let y_cpu = cpu_tensor(&a, &[4], DType::F16)
        .atan2(&cpu_tensor(&b, &[4], DType::F16))
        .unwrap();
    let y_gpu = gpu_tensor(&a, &[4], DType::F16)
        .atan2(&gpu_tensor(&b, &[4], DType::F16))
        .unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "atan2_f16");
}

// -- Unary: gelu_erf BF16 ---------------------------------------------------

#[test]
fn test_gelu_erf_bf16() {
    init();
    let data = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let y_cpu = cpu_tensor(&data, &[7], DType::BF16).gelu_erf().unwrap();
    let y_gpu = gpu_tensor(&data, &[7], DType::BF16).gelu_erf().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::BF16);
    assert_close(&y_gpu, &y_cpu, "gelu_erf_bf16");
}

#[test]
fn test_gelu_erf_f16() {
    init();
    let data = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let y_cpu = cpu_tensor(&data, &[7], DType::F16).gelu_erf().unwrap();
    let y_gpu = gpu_tensor(&data, &[7], DType::F16).gelu_erf().unwrap();
    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dtype(), DType::F16);
    assert_close(&y_gpu, &y_cpu, "gelu_erf_f16");
}
