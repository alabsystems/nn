#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Kokoro TTS F0/energy prediction (`kokoro_f0.rs`).

use super::*;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{
    AdaIn, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, InstanceNormPrecision,
    Linear,
};
use nn_core::DType;

/// Create a small AdainResBlk1d manually for testing (no weight loading).
fn make_test_adain_resblk(
    dim_in: usize,
    dim_out: usize,
    style_dim: usize,
    upsample: bool,
) -> AdainResBlk1d {
    let device = nn_core::Device::Cpu;
    let n1_w = DynTensor::zeros(&[2 * dim_in, style_dim], DType::F32, &device).unwrap();
    let n1_b = DynTensor::zeros(&[2 * dim_in], DType::F32, &device).unwrap();
    let n1 = AdaIn::new_with_precision(
        Linear::new(n1_w, Some(n1_b)).unwrap(),
        1e-5,
        InstanceNormPrecision::MatchPyTorchCpu,
    )
    .unwrap();

    let n2_w = DynTensor::zeros(&[2 * dim_out, style_dim], DType::F32, &device).unwrap();
    let n2_b = DynTensor::zeros(&[2 * dim_out], DType::F32, &device).unwrap();
    let n2 = AdaIn::new_with_precision(
        Linear::new(n2_w, Some(n2_b)).unwrap(),
        1e-5,
        InstanceNormPrecision::MatchPyTorchCpu,
    )
    .unwrap();

    let c1_w = DynTensor::zeros(&[dim_out, dim_in, 3], DType::F32, &device).unwrap();
    let c1_b = DynTensor::zeros(&[dim_out], DType::F32, &device).unwrap();
    let c1 = Conv1d::new(c1_w, Some(c1_b), Conv1dConfig::default().with_padding(1)).unwrap();

    let c2_w = DynTensor::zeros(&[dim_out, dim_out, 3], DType::F32, &device).unwrap();
    let c2_b = DynTensor::zeros(&[dim_out], DType::F32, &device).unwrap();
    let c2 = Conv1d::new(c2_w, Some(c2_b), Conv1dConfig::default().with_padding(1)).unwrap();

    let skip_conv = if dim_in != dim_out {
        let sw = DynTensor::zeros(&[dim_out, dim_in, 1], DType::F32, &device).unwrap();
        let sb = DynTensor::zeros(&[dim_out], DType::F32, &device).unwrap();
        Some(Conv1d::new(sw, Some(sb), Conv1dConfig::default()).unwrap())
    } else {
        None
    };

    let pool = if upsample {
        let pw = DynTensor::zeros(&[dim_in, 1, 3], DType::F32, &device).unwrap();
        let pb = DynTensor::zeros(&[dim_in], DType::F32, &device).unwrap();
        Some(
            ConvTranspose1d::new(
                pw,
                Some(pb),
                ConvTranspose1dConfig::default()
                    .with_stride(2)
                    .with_padding(1)
                    .with_output_padding(1)
                    .with_groups(dim_in),
            )
            .expect("valid ConvTranspose1d config"),
        )
    } else {
        None
    };

    AdainResBlk1d {
        n1,
        n2,
        c1,
        c2,
        skip_conv,
        pool,
        upsample,
    }
}

#[test]
fn test_adain_resblk_same_dim_no_upsample() {
    let block = make_test_adain_resblk(8, 8, 4, false);
    let x = DynTensor::zeros(&[1, 8, 10], DType::F32, &nn_core::Device::Cpu).unwrap();
    let style = DynTensor::zeros(&[1, 4], DType::F32, &nn_core::Device::Cpu).unwrap();
    let out = block.forward(&x, &style).unwrap();
    assert_eq!(out.dims(), &[1, 8, 10]);
}

#[test]
fn test_adain_resblk_dim_change_no_upsample() {
    let block = make_test_adain_resblk(8, 4, 4, false);
    let x = DynTensor::zeros(&[1, 8, 10], DType::F32, &nn_core::Device::Cpu).unwrap();
    let style = DynTensor::zeros(&[1, 4], DType::F32, &nn_core::Device::Cpu).unwrap();
    let out = block.forward(&x, &style).unwrap();
    assert_eq!(out.dims(), &[1, 4, 10]);
}

#[test]
fn test_adain_resblk_with_upsample() {
    let block = make_test_adain_resblk(8, 4, 4, true);
    let x = DynTensor::zeros(&[1, 8, 10], DType::F32, &nn_core::Device::Cpu).unwrap();
    let style = DynTensor::zeros(&[1, 4], DType::F32, &nn_core::Device::Cpu).unwrap();
    let out = block.forward(&x, &style).unwrap();
    // Upsampled: T=10 → T=20
    assert_eq!(out.dims(), &[1, 4, 20]);
}

#[test]
fn test_adain_resblk_nonzero_weights() {
    // Verify block produces non-trivial output with small nonzero weights.
    let device = nn_core::Device::Cpu;
    let dim = 4;
    let style_dim = 2;

    let n1_w = DynTensor::full(&[2 * dim, style_dim], 0.1, DType::F32, &device).unwrap();
    let n1_b = DynTensor::zeros(&[2 * dim], DType::F32, &device).unwrap();
    let n1 = AdaIn::new_with_precision(
        Linear::new(n1_w, Some(n1_b)).unwrap(),
        1e-5,
        InstanceNormPrecision::MatchPyTorchCpu,
    )
    .unwrap();

    let n2_w = DynTensor::full(&[2 * dim, style_dim], 0.1, DType::F32, &device).unwrap();
    let n2_b = DynTensor::zeros(&[2 * dim], DType::F32, &device).unwrap();
    let n2 = AdaIn::new_with_precision(
        Linear::new(n2_w, Some(n2_b)).unwrap(),
        1e-5,
        InstanceNormPrecision::MatchPyTorchCpu,
    )
    .unwrap();

    let c1_w = DynTensor::full(&[dim, dim, 3], 0.01, DType::F32, &device).unwrap();
    let c1_b = DynTensor::zeros(&[dim], DType::F32, &device).unwrap();
    let c1 = Conv1d::new(c1_w, Some(c1_b), Conv1dConfig::default().with_padding(1)).unwrap();

    let c2_w = DynTensor::full(&[dim, dim, 3], 0.01, DType::F32, &device).unwrap();
    let c2_b = DynTensor::zeros(&[dim], DType::F32, &device).unwrap();
    let c2 = Conv1d::new(c2_w, Some(c2_b), Conv1dConfig::default().with_padding(1)).unwrap();

    let block = AdainResBlk1d {
        n1,
        n2,
        c1,
        c2,
        skip_conv: None,
        pool: None,
        upsample: false,
    };

    let x = DynTensor::full(&[1, dim, 6], 1.0, DType::F32, &device).unwrap();
    let style = DynTensor::full(&[1, style_dim], 0.5, DType::F32, &device).unwrap();
    let out = block.forward(&x, &style).unwrap();
    assert_eq!(out.dims(), &[1, dim, 6]);
    // With nonzero input and weights, output should not be all zeros
    let flat = out.to_flat_vec::<f32>().unwrap();
    let max_val = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_val.abs() > 1e-6,
        "expected non-trivial output, got max={max_val}"
    );
}
