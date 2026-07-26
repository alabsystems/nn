// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Stage1ResBlk and FullDecoder.

use super::*;
use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

fn ones_tensor(shape: &[usize]) -> DynTensor {
    DynTensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
}

/// Mini config for shape-only tests. Production dims (d_en=512, gen=512,
/// upsample=[10,6]) cause a 39s CPU forward pass that only checks shapes.
/// This config exercises the same code paths with ~500,000x less computation.
fn mini_config() -> KokoroConfig {
    KokoroConfig {
        d_en: 16,
        n_prosody_layers: 3,
        style_dim: 4,
        upsample_rates: vec![2, 2],
        upsample_kernel_sizes: vec![4, 4],
        resblock_kernel_sizes: vec![3],
        resblock_dilations: vec![vec![1]],
        gen_initial_channels: 16,
        n_fft: 4,
        f0_bilstm_hidden: 4,
        max_dur: 5,
        plbert: crate::plbert::PlbertConfig::default(),
    }
}

#[test]
fn test_stage1_resblk_no_upsample() {
    let dim_in = 16;
    let dim_out = 32;
    let style_dim = 4;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let block =
        Stage1ResBlk::load(&vb, dim_in, dim_out, style_dim, false).expect("load Stage1ResBlk");

    let x = ones_tensor(&[1, dim_in, 10]);
    let style = ones_tensor(&[1, style_dim]);
    let out = block.forward(&x, &style).expect("forward");

    // Output shape: [1, dim_out, T] — same time dimension
    assert_eq!(out.dims(), &[1, dim_out, 10]);
}

#[test]
fn test_stage1_resblk_same_channels() {
    let dim = 16;
    let style_dim = 4;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let block =
        Stage1ResBlk::load(&vb, dim, dim, style_dim, false).expect("load Stage1ResBlk same dims");

    let x = ones_tensor(&[1, dim, 10]);
    let style = ones_tensor(&[1, style_dim]);
    let out = block.forward(&x, &style).expect("forward");

    // No conv1x1 when dim_in == dim_out
    assert_eq!(out.dims(), &[1, dim, 10]);
}

#[test]
fn test_stage1_resblk_upsample_2x() {
    let dim_in = 16;
    let dim_out = 8;
    let style_dim = 4;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let block =
        Stage1ResBlk::load(&vb, dim_in, dim_out, style_dim, true).expect("load Stage1ResBlk up");

    let x = ones_tensor(&[1, dim_in, 10]);
    let style = ones_tensor(&[1, style_dim]);
    let out = block.forward(&x, &style).expect("forward upsample");

    // Output shape: [1, dim_out, 2*T]
    assert_eq!(out.dims(), &[1, dim_out, 20]);
}

#[test]
fn test_nearest_upsample_2x() {
    let x = DynTensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[1, 1, 3], &Device::Cpu).unwrap();
    let up = nearest_upsample_2x(&x).unwrap();
    assert_eq!(up.dims(), &[1, 1, 6]);
    let vals = up.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

#[test]
fn test_full_decoder_loads() {
    // Verify that FullDecoder can be constructed with zero weights
    // at production Kokoro dimensions.
    let config = KokoroConfig::default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let decoder = FullDecoder::load(vb.pp("decoder"), &config);
    assert!(
        decoder.is_ok(),
        "FullDecoder should load: {}",
        decoder
            .err()
            .map_or_else(|| "ok".to_string(), |e| e.to_string())
    );
}

#[test]
fn test_full_decoder_forward_shape() {
    // Use mini config: shape-only test doesn't need production dims (39s -> <1s).
    let config = mini_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let decoder = FullDecoder::load(vb.pp("decoder"), &config).expect("load");

    let d_en = config.d_en;
    let style_dim = config.style_dim;
    let t_mel = 4;
    let asr = ones_tensor(&[1, d_en, t_mel]);
    let f0_curve = ones_tensor(&[1, 1, 2 * t_mel]);
    let n_curve = ones_tensor(&[1, 1, 2 * t_mel]);
    let style = ones_tensor(&[1, style_dim]);

    let n_bins = config.n_fft / 2 + 1;
    let upsample_factor: usize = config.upsample_rates.iter().product();
    let t_full = 2 * t_mel * upsample_factor;
    let har_source = ones_tensor(&[1, 2 * n_bins, t_full]);

    let (mag, phase) = decoder
        .forward(&asr, &f0_curve, &n_curve, &style, &har_source)
        .expect("forward");

    // Output: [B, n_bins, T_out]
    assert_eq!(mag.dims()[0], 1);
    assert_eq!(mag.dims()[1], n_bins);
    assert_eq!(phase.dims()[0], 1);
    assert_eq!(phase.dims()[1], n_bins);
    assert_eq!(mag.dims()[2], phase.dims()[2]);
}

#[test]
fn test_stage1_resblk_encode_dims() {
    // Test the exact dimensions used by the encode block: 514→1024
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let block =
        Stage1ResBlk::load(&vb, 514, 1024, 128, false).expect("load encode block (514→1024)");

    let x = ones_tensor(&[1, 514, 8]);
    let style = ones_tensor(&[1, 128]);
    let out = block.forward(&x, &style).expect("forward");
    assert_eq!(out.dims(), &[1, 1024, 8]);
}

#[test]
fn test_stage1_resblk_decode_dims() {
    // Test the exact dimensions used by decode blocks: 1090→1024 (no upsample)
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let block =
        Stage1ResBlk::load(&vb, 1090, 1024, 128, false).expect("load decode block (1090→1024)");

    let x = ones_tensor(&[1, 1090, 8]);
    let style = ones_tensor(&[1, 128]);
    let out = block.forward(&x, &style).expect("forward");
    assert_eq!(out.dims(), &[1, 1024, 8]);
}

#[test]
fn test_stage1_resblk_final_decode_dims() {
    // Test the exact dimensions used by the final decode block: 1090→512, upsample=2×
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let block = Stage1ResBlk::load(&vb, 1090, 512, 128, true)
        .expect("load final decode block (1090→512, up)");

    let x = ones_tensor(&[1, 1090, 8]);
    let style = ones_tensor(&[1, 128]);
    let out = block.forward(&x, &style).expect("forward upsample");
    assert_eq!(out.dims(), &[1, 512, 16]);
}
