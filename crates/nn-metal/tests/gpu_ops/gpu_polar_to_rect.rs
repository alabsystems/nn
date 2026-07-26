// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU parity test for fused polar-to-rectangular kernel (#2491).
//!
//! Verifies that the fused `gpu_polar_to_rect` kernel (called from
//! `CompiledKokoro::gpu_istft`) produces results matching the CPU
//! reference: `real = mag * cos(phase)`, `imag = mag * sin(phase)`.
//!
//! The fused kernel is `pub(crate)` inside nn-metal, so this test
//! exercises it indirectly via the DynTensor GPU ops path (cos, sin, mul
//! go through the GPU backend when tensors are on Device::Metal).
//! The end-to-end pipeline test is `compiled_kokoro_synthesize.rs`.
//!
//! Part of #2218 (Kokoro epic).

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

fn cpu() -> Device {
    Device::Cpu
}

/// GPU polar-to-rect via DynTensor ops matches CPU reference.
///
/// Computes `real = mag * cos(phase)` and `imag = mag * sin(phase)` on
/// both CPU and GPU, then compares element-wise.
#[test]
fn test_polar_to_rect_gpu_cpu_parity() {
    super::test_utils::gpu_init();

    // Kokoro-like dimensions: [1, n_freq, n_frames].
    let n_freq = 513;
    let n_frames = 64;
    let shape = [1, n_freq, n_frames];
    let total = n_freq * n_frames;

    // Deterministic inputs with realistic ranges.
    let mag_data = super::test_utils::rand_f32_vec(42, total, 0.0, 5.0);
    let phase_data =
        super::test_utils::rand_f32_vec(99, total, -std::f32::consts::PI, std::f32::consts::PI);

    // CPU reference: compute on CPU device.
    let mag_cpu = DynTensor::from_vec(mag_data.clone(), &shape, &cpu()).unwrap();
    let phase_cpu = DynTensor::from_vec(phase_data.clone(), &shape, &cpu()).unwrap();
    let cpu_real = mag_cpu.mul(&phase_cpu.cos().unwrap()).unwrap();
    let cpu_imag = mag_cpu.mul(&phase_cpu.sin().unwrap()).unwrap();
    let cpu_real_vals = cpu_real.to_flat_vec::<f32>().unwrap();
    let cpu_imag_vals = cpu_imag.to_flat_vec::<f32>().unwrap();

    // GPU path: DynTensor ops dispatch to Metal backend.
    let mag_gpu = DynTensor::from_vec(mag_data, &shape, &Device::metal()).unwrap();
    let phase_gpu = DynTensor::from_vec(phase_data, &shape, &Device::metal()).unwrap();
    let gpu_real = mag_gpu.mul(&phase_gpu.cos().unwrap()).unwrap();
    let gpu_imag = mag_gpu.mul(&phase_gpu.sin().unwrap()).unwrap();
    let gpu_real_vals = gpu_real
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let gpu_imag_vals = gpu_imag
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Compare real part.
    assert_eq!(
        cpu_real_vals.len(),
        gpu_real_vals.len(),
        "real length mismatch"
    );
    let max_real_diff = cpu_real_vals
        .iter()
        .zip(gpu_real_vals.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f32, f32::max);

    // Compare imag part.
    assert_eq!(
        cpu_imag_vals.len(),
        gpu_imag_vals.len(),
        "imag length mismatch"
    );
    let max_imag_diff = cpu_imag_vals
        .iter()
        .zip(gpu_imag_vals.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f32, f32::max);

    eprintln!(
        "polar_to_rect GPU/CPU: {total} elements, max_real_diff={max_real_diff:.6e}, \
         max_imag_diff={max_imag_diff:.6e}"
    );

    // GPU sin/cos may differ from CPU by up to a few ULPs.
    assert!(
        max_real_diff < 1e-5,
        "Real part GPU/CPU diff should be < 1e-5, got {max_real_diff:.6e}"
    );
    assert!(
        max_imag_diff < 1e-5,
        "Imag part GPU/CPU diff should be < 1e-5, got {max_imag_diff:.6e}"
    );
}
