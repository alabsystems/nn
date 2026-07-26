// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! WeSpeaker ResNet34 GPU forward pass tests.
//!
//! Verifies the complete WeSpeaker pipeline runs on Metal GPU:
//! Conv2d stem → ResNet34 body → TSTP pooling → Linear head.
//!
//! Tests:
//! 1. Zero-weight forward: validates shape and device placement.
//! 2. CPU vs GPU parity: same zero weights, compare element-wise.
//! 3. Pretrained forward: real weights on GPU (gated on file existence).
//!
//! Issue: #2294

use std::path::Path;

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::{load_safetensors, DynTensor};
use nn_core::{DType, Device, VarBuilder};
use nn_metal::{register_metal_dyn_backend, MetalBackend};
use nn_models::WeSpeakerResNet34;

fn init() {
    let _ = MetalBackend::init();
    register_metal_dyn_backend();
}

/// Workspace root for locating model weight files.
fn repo_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent() // crates/
        .unwrap()
        .parent() // workspace root
        .unwrap()
        .to_path_buf()
}

// -- Zero-weight forward on GPU -----------------------------------------------

#[test]
fn test_wespeaker_gpu_forward_zeros() {
    init();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let model = WeSpeakerResNet34::load(&vb).expect("GPU model load");

    // Input: [1, 1, 200, 80] fbank features on GPU.
    let input = DynTensor::zeros(&[1, 1, 200, 80], DType::F32, &Device::metal())
        .expect("input tensor creation");

    let result = model.forward(&input);
    assert!(result.is_ok(), "forward on GPU should succeed: {result:?}");

    let out = result.unwrap();
    assert_eq!(out.dims(), &[1, 256], "output shape should be [B, 256]");
    assert_eq!(out.device(), Device::metal(), "output should stay on GPU");
}

// -- CPU vs GPU parity (zero weights) -----------------------------------------

#[test]
fn test_wespeaker_cpu_gpu_match_zeros() {
    gpu_init();

    // CPU reference.
    let vb_cpu = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model_cpu = WeSpeakerResNet34::load(&vb_cpu).expect("CPU model load");
    let input_cpu =
        DynTensor::zeros(&[1, 1, 200, 80], DType::F32, &Device::Cpu).expect("CPU input");
    let out_cpu = model_cpu.forward(&input_cpu).expect("CPU forward");

    // GPU.
    let vb_gpu = VarBuilder::zeros(DType::F32, &Device::metal());
    let model_gpu = WeSpeakerResNet34::load(&vb_gpu).expect("GPU model load");
    let input_gpu =
        DynTensor::zeros(&[1, 1, 200, 80], DType::F32, &Device::metal()).expect("GPU input");
    let out_gpu = model_gpu.forward(&input_gpu).expect("GPU forward");

    // Conv2d + BatchNorm + 36 layers: tolerance 1e-3 for f32 accumulation.
    assert_gpu_cpu_close(&out_gpu, &out_cpu, 1e-3, "wespeaker_zeros");
}

// -- Pretrained weights on GPU (gated on file existence) ----------------------

#[test]
fn test_wespeaker_gpu_forward_pretrained() {
    init();

    let weights_path = repo_root().join("models/wespeaker/weights.safetensors");
    if !weights_path.exists() {
        eprintln!(
            "Skipping: {} not found. Run: python3 scripts/export_wespeaker.py",
            weights_path.display()
        );
        return;
    }

    let tensors = load_safetensors(&weights_path).expect("load weights");
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::metal());
    let model = WeSpeakerResNet34::load(&vb).expect("GPU model load with pretrained weights");

    // Same input shape as parity test.
    let input =
        DynTensor::zeros(&[1, 1, 300, 80], DType::F32, &Device::metal()).expect("input tensor");
    let out = model
        .forward(&input)
        .expect("GPU forward with pretrained weights");

    assert_eq!(out.dims(), &[1, 256]);
    assert_eq!(out.device(), Device::metal());

    // With pretrained weights, output should be finite and nonzero.
    let cpu_out = out.to_device(&Device::Cpu).unwrap();
    let data = cpu_out.to_flat_vec::<f32>().expect("to vec");
    assert!(
        data.iter().all(|v| v.is_finite()),
        "output has non-finite values"
    );
    assert!(
        data.iter().any(|v| *v != 0.0),
        "output is all zeros with pretrained weights"
    );
}

// -- Pretrained GPU vs CPU parity ---------------------------------------------

#[test]
fn test_wespeaker_gpu_cpu_parity_pretrained() {
    gpu_init();

    let weights_path = repo_root().join("models/wespeaker/weights.safetensors");
    let ref_path = repo_root().join("models/wespeaker/reference.safetensors");
    if !weights_path.exists() || !ref_path.exists() {
        eprintln!(
            "Skipping: weight/reference files not found. Run: python3 scripts/export_wespeaker.py"
        );
        return;
    }

    let reference = load_safetensors(&ref_path).expect("load reference");
    let ref_input = reference
        .get("input_fbank")
        .expect("reference missing input_fbank");
    let ref_output = reference.get("output").expect("reference missing output");

    // GPU model.
    let tensors = load_safetensors(&weights_path).expect("load weights");
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::metal());
    let model = WeSpeakerResNet34::load(&vb).expect("GPU model load");

    // Move input to GPU.
    let gpu_input = ref_input.to_device(&Device::metal()).expect("input to GPU");
    let gpu_output = model.forward(&gpu_input).expect("GPU forward");
    let gpu_cpu = gpu_output.to_device(&Device::Cpu).unwrap();
    let gpu_data = gpu_cpu.to_flat_vec::<f32>().expect("gpu to vec");
    let ref_data = ref_output.to_flat_vec::<f32>().expect("ref to vec");

    // Compute metrics.
    assert_eq!(gpu_data.len(), ref_data.len());
    let max_diff: f32 = gpu_data
        .iter()
        .zip(ref_data.iter())
        .map(|(g, r)| (g - r).abs())
        .fold(0.0f32, f32::max);
    let dot: f32 = gpu_data
        .iter()
        .zip(ref_data.iter())
        .map(|(g, r)| g * r)
        .sum();
    let norm_g: f32 = gpu_data.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_r: f32 = ref_data.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cosine = if norm_g > 0.0 && norm_r > 0.0 {
        dot / (norm_g * norm_r)
    } else {
        0.0
    };

    eprintln!("WeSpeaker GPU parity: max_abs_diff={max_diff:.6e}, cosine={cosine:.6}");

    // GPU adds additional numerical error from Metal dispatch.
    // Tolerance is wider than CPU-only parity (1e-3 vs 1e-3).
    assert!(
        max_diff < 5e-3,
        "GPU vs PyTorch max abs diff too large: {max_diff:.6e} (want < 5e-3)"
    );
    assert!(
        cosine > 0.999,
        "GPU vs PyTorch cosine too low: {cosine:.6} (want > 0.999)"
    );
}
