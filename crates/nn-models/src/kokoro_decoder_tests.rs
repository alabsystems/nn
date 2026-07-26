#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Kokoro Generator (ISTFTNet).

use super::*;
use crate::kokoro_tts::KokoroConfig;
use crate::plbert::PlbertConfig;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::var_builder::VarBuilder;
use nn_core::DType;
use std::collections::HashMap;

/// Small test config for Generator tests (matches make_generator_weights).
fn test_generator_config() -> KokoroConfig {
    KokoroConfig {
        upsample_rates: vec![2],
        upsample_kernel_sizes: vec![4],
        resblock_kernel_sizes: vec![3],
        resblock_dilations: vec![vec![1, 2]],
        gen_initial_channels: 8,
        style_dim: 4,
        n_fft: 4,
        d_en: 512,
        n_prosody_layers: 3,
        f0_bilstm_hidden: 256,
        max_dur: 50,
        plbert: PlbertConfig::default(),
    }
}

/// Insert Conv1d weights into tensor map.
fn insert_conv1d(
    tensors: &mut HashMap<String, DynTensor>,
    prefix: &str,
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
) {
    tensors.insert(
        format!("{prefix}.weight"),
        DynTensor::zeros(&[out_ch, in_ch, kernel], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        format!("{prefix}.bias"),
        DynTensor::zeros(&[out_ch], DType::F32, &cpu()).unwrap(),
    );
}

/// Insert ConvTranspose1d weights (shape: [in_ch, out_ch, kernel]).
fn insert_conv_transpose(
    tensors: &mut HashMap<String, DynTensor>,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
) {
    tensors.insert(
        format!("{prefix}.weight"),
        DynTensor::zeros(&[in_ch, out_ch, kernel], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        format!("{prefix}.bias"),
        DynTensor::zeros(&[out_ch], DType::F32, &cpu()).unwrap(),
    );
}

/// Insert AdaIN weights: Linear projecting [style_dim] -> [2*channels].
fn insert_adain(
    tensors: &mut HashMap<String, DynTensor>,
    prefix: &str,
    channels: usize,
    style_dim: usize,
) {
    tensors.insert(
        format!("{prefix}.fc.weight"),
        DynTensor::zeros(&[2 * channels, style_dim], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        format!("{prefix}.fc.bias"),
        DynTensor::zeros(&[2 * channels], DType::F32, &cpu()).unwrap(),
    );
}

/// Insert ResBlock weights for a given block prefix.
fn insert_resblock(
    tensors: &mut HashMap<String, DynTensor>,
    prefix: &str,
    channels: usize,
    kernel_size: usize,
    num_dilations: usize,
    style_dim: usize,
) {
    for i in 0..num_dilations {
        insert_conv1d(
            tensors,
            &format!("{prefix}.convs1.{i}"),
            channels,
            channels,
            kernel_size,
        );
        insert_conv1d(
            tensors,
            &format!("{prefix}.convs2.{i}"),
            channels,
            channels,
            kernel_size,
        );
        insert_adain(
            tensors,
            &format!("{prefix}.adain1.{i}"),
            channels,
            style_dim,
        );
        insert_adain(
            tensors,
            &format!("{prefix}.adain2.{i}"),
            channels,
            style_dim,
        );
        tensors.insert(
            format!("{prefix}.alpha1.{i}"),
            DynTensor::full(&[1, channels, 1], 1.0, DType::F32, &cpu()).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.alpha2.{i}"),
            DynTensor::full(&[1, channels, 1], 1.0, DType::F32, &cpu()).unwrap(),
        );
    }
}

/// Insert ResBlock weights with a specified fill value (non-zero for behavioral tests).
fn insert_resblock_filled(
    tensors: &mut HashMap<String, DynTensor>,
    prefix: &str,
    ch: usize,
    ks: usize,
    nd: usize,
    sd: usize,
    fill: f64,
) {
    for i in 0..nd {
        for t in ["convs1", "convs2"] {
            tensors.insert(
                format!("{prefix}.{t}.{i}.weight"),
                DynTensor::full(&[ch, ch, ks], fill, DType::F32, &cpu()).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.{t}.{i}.bias"),
                DynTensor::zeros(&[ch], DType::F32, &cpu()).unwrap(),
            );
        }
        for t in ["adain1", "adain2"] {
            tensors.insert(
                format!("{prefix}.{t}.{i}.fc.weight"),
                DynTensor::full(&[2 * ch, sd], fill * 10.0, DType::F32, &cpu()).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.{t}.{i}.fc.bias"),
                DynTensor::zeros(&[2 * ch], DType::F32, &cpu()).unwrap(),
            );
        }
        for t in ["alpha1", "alpha2"] {
            tensors.insert(
                format!("{prefix}.{t}.{i}"),
                DynTensor::full(&[1, ch, 1], 1.0, DType::F32, &cpu()).unwrap(),
            );
        }
    }
}

/// Build a minimal Generator weight map for testing.
///
/// Uses: initial_channels=8, style_dim=4, n_fft=4, upsample_rates=[2],
/// upsample_kernel_sizes=[4], resblock_kernel_sizes=[3], resblock_dilations=[[1,2]].
fn make_generator_weights() -> HashMap<String, DynTensor> {
    let ch = 8;
    let next_ch = 4;
    let style_dim = 4;
    let n_fft = 4;
    let n_bins = n_fft / 2 + 1; // 3

    let mut tensors = HashMap::new();
    insert_conv1d(&mut tensors, "conv_pre", ch, ch, 7);
    insert_conv_transpose(&mut tensors, "ups.0", ch, next_ch, 4);
    insert_conv1d(&mut tensors, "noise_convs.0", next_ch, 2 * n_bins, 1);
    // noise_res: kernel=11 (last stage), dilations=[1,3,5] (hardcoded per PyTorch).
    insert_resblock(&mut tensors, "noise_res.0", next_ch, 11, 3, style_dim);
    insert_resblock(&mut tensors, "resblocks.0", next_ch, 3, 2, style_dim);
    insert_conv1d(&mut tensors, "conv_post", 2 * n_bins, next_ch, 7);
    tensors
}

#[test]
fn test_resblock_load() {
    let ch = 4;
    let style_dim = 4;
    let mut tensors = HashMap::new();
    insert_resblock(&mut tensors, "rb", ch, 3, 2, style_dim);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let rb = ResBlock::load(vb.pp("rb"), ch, 3, &[1, 2], style_dim);
    assert!(rb.is_ok(), "ResBlock::load failed: {:?}", rb.err());
}

#[test]
fn test_resblock_forward() {
    let ch = 4;
    let style_dim = 4;
    let mut tensors = HashMap::new();
    insert_resblock(&mut tensors, "rb", ch, 3, 2, style_dim);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let rb = ResBlock::load(vb.pp("rb"), ch, 3, &[1, 2], style_dim).unwrap();
    let x = DynTensor::zeros(&[1, ch, 8], DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, style_dim], DType::F32, &cpu()).unwrap();
    let result = rb.forward(&x, &style);
    assert!(
        result.is_ok(),
        "ResBlock::forward failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().dims(), &[1, ch, 8]);
}

/// Verify ResBlock with non-zero weights transforms input (not just residual identity).
#[test]
fn test_resblock_forward_non_trivial() {
    let (ch, sd) = (4, 4);
    let mut tensors = HashMap::new();
    insert_resblock_filled(&mut tensors, "rb", ch, 3, 2, sd, 0.01);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let rb = ResBlock::load(vb.pp("rb"), ch, 3, &[1, 2], sd).unwrap();

    let x = DynTensor::full(&[1, ch, 8], 1.0, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&[1, sd], 0.5, DType::F32, &cpu()).unwrap();
    let out = rb.forward(&x, &style).unwrap();
    assert_eq!(out.dims(), &[1, ch, 8]);

    let in_vals = x.to_flat_vec::<f32>().unwrap();
    let out_vals = out.to_flat_vec::<f32>().unwrap();
    let diff: f32 = in_vals
        .iter()
        .zip(&out_vals)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "non-zero weights must transform input, diff={diff}"
    );
    assert!(
        out_vals.iter().all(|v| v.is_finite()),
        "output must be finite"
    );
}

#[test]
fn test_generator_load() {
    let tensors = make_generator_weights();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let generator = Generator::load(&vb, &test_generator_config());
    assert!(
        generator.is_ok(),
        "Generator::load failed: {:?}",
        generator.err()
    );
}

#[test]
fn test_generator_forward_shape() {
    let tensors = make_generator_weights();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let generator = Generator::load(&vb, &test_generator_config()).unwrap();
    let n_bins = 4 / 2 + 1;
    let x = DynTensor::zeros(&[1, 8, 4], DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 4], DType::F32, &cpu()).unwrap();
    let har_source = DynTensor::zeros(&[1, 2 * n_bins, 8], DType::F32, &cpu()).unwrap();
    let (mag, phase) = generator.forward(&x, &style, &har_source).unwrap();
    // Output length = 8 (from ups) + 1 (from reflection_pad1d on last stage) = 9
    assert_eq!(mag.dims(), &[1, n_bins, 9]);
    assert_eq!(phase.dims(), &[1, n_bins, 9]);
}

#[test]
fn test_generator_output_finiteness() {
    let tensors = make_generator_weights();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let generator = Generator::load(&vb, &test_generator_config()).unwrap();
    let n_bins = 4 / 2 + 1;
    let x = DynTensor::full(&[1, 8, 4], 0.5, DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 4], DType::F32, &cpu()).unwrap();
    let har_source = DynTensor::zeros(&[1, 2 * n_bins, 8], DType::F32, &cpu()).unwrap();
    let (mag, phase) = generator.forward(&x, &style, &har_source).unwrap();
    assert!(mag
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
    assert!(phase
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn test_generator_two_stages() {
    let ch = 16;
    let mid_ch = 8;
    let final_ch = 4;
    let style_dim = 4;
    let n_fft = 6;
    let n_bins = n_fft / 2 + 1;
    let mut tensors = HashMap::new();
    insert_conv1d(&mut tensors, "conv_pre", ch, ch, 7);
    insert_conv_transpose(&mut tensors, "ups.0", ch, mid_ch, 4);
    // Stage 0 (not last): cumulative_stride=2, kernel=2*2=4
    insert_conv1d(&mut tensors, "noise_convs.0", mid_ch, 2 * n_bins, 4);
    // noise_res: kernel=7 (non-last), dilations=[1,3,5] (hardcoded per PyTorch).
    insert_resblock(&mut tensors, "noise_res.0", mid_ch, 7, 3, style_dim);
    insert_resblock(&mut tensors, "resblocks.0", mid_ch, 3, 2, style_dim);
    insert_conv_transpose(&mut tensors, "ups.1", mid_ch, final_ch, 4);
    // Stage 1 (last): kernel=1
    insert_conv1d(&mut tensors, "noise_convs.1", final_ch, 2 * n_bins, 1);
    // noise_res: kernel=11 (last), dilations=[1,3,5] (hardcoded per PyTorch).
    insert_resblock(&mut tensors, "noise_res.1", final_ch, 11, 3, style_dim);
    insert_resblock(&mut tensors, "resblocks.1", final_ch, 3, 2, style_dim);
    insert_conv1d(&mut tensors, "conv_post", 2 * n_bins, final_ch, 7);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let config = KokoroConfig {
        upsample_rates: vec![2, 2],
        upsample_kernel_sizes: vec![4, 4],
        resblock_kernel_sizes: vec![3],
        resblock_dilations: vec![vec![1, 2]],
        gen_initial_channels: ch,
        style_dim,
        n_fft,
        ..test_generator_config()
    };
    let generator = Generator::load(&vb, &config).unwrap();
    let x = DynTensor::zeros(&[1, ch, 4], DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, style_dim], DType::F32, &cpu()).unwrap();
    let har_source = DynTensor::zeros(&[1, 2 * n_bins, 16], DType::F32, &cpu()).unwrap();
    let (mag, phase) = generator.forward(&x, &style, &har_source).unwrap();
    // Output length = 16 (from 2 ups stages) + 1 (from reflection_pad1d on last) = 17
    assert_eq!(mag.dims(), &[1, n_bins, 17]);
    assert_eq!(phase.dims(), &[1, n_bins, 17]);
}

/// Verify noise injection runs without error for both zero and non-zero har_source.
#[test]
fn test_generator_noise_injection_finiteness() {
    let tensors = make_generator_weights();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let generator = Generator::load(&vb, &test_generator_config()).unwrap();
    let n_bins = 4 / 2 + 1;
    let x = DynTensor::full(&[1, 8, 4], 0.5, DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 4], DType::F32, &cpu()).unwrap();
    for fill in [0.0, 1.0] {
        let har = DynTensor::full(&[1, 2 * n_bins, 8], fill, DType::F32, &cpu()).unwrap();
        let (mag, phase) = generator.forward(&x, &style, &har).unwrap();
        assert!(mag
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
        assert!(phase
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }
}

/// Verify Generator output properties: magnitude > 0 (exp), phase in [-1, 1] (sin).
#[test]
fn test_generator_output_properties() {
    let tensors = make_generator_weights();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let generator = Generator::load(&vb, &test_generator_config()).unwrap();
    let n_bins = 4 / 2 + 1;
    let x = DynTensor::full(&[1, 8, 4], 0.5, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&[1, 4], 0.3, DType::F32, &cpu()).unwrap();
    let har = DynTensor::zeros(&[1, 2 * n_bins, 8], DType::F32, &cpu()).unwrap();
    let (mag, phase) = generator.forward(&x, &style, &har).unwrap();
    let mv = mag.to_flat_vec::<f32>().unwrap();
    let pv = phase.to_flat_vec::<f32>().unwrap();
    assert!(
        mv.iter().all(|v| v.is_finite() && *v > 0.0),
        "mag = exp(x) must be > 0"
    );
    assert!(
        pv.iter().all(|v| *v >= -1.0 && *v <= 1.0),
        "phase = sin(x) in [-1, 1]"
    );
}

/// Verify `snake(x, alpha)` ≈ `snake_tensor(x, uniform_alpha)` within tolerance.
/// AC2 of #1049: scalar and tensor paths agree for alpha in clamped range [1e-6, 1e6].
#[test]
fn test_snake_scalar_tensor_parity() {
    let x = DynTensor::new(&[0.0f32, 1.0, -1.0, 0.5, -0.5, 2.7], &[1, 2, 3], &cpu()).unwrap();
    for alpha in [1e-6, 0.001, 0.1, 1.0, 2.5, 10.0, 100.0, 1e6] {
        let scalar_out = x.snake(alpha).unwrap();
        let alpha_tensor = DynTensor::full(&[1, 2, 1], alpha, DType::F32, &cpu()).unwrap();
        let tensor_out = x.snake_tensor(&alpha_tensor).unwrap();
        let s = scalar_out.to_flat_vec::<f32>().unwrap();
        let t = tensor_out.to_flat_vec::<f32>().unwrap();
        assert_eq!(s.len(), t.len());
        for (i, (&sv, &tv)) in s.iter().zip(t.iter()).enumerate() {
            assert!(
                (sv - tv).abs() < 1e-4,
                "alpha={alpha}, element {i}: scalar={sv}, tensor={tv}, diff={}",
                (sv - tv).abs()
            );
        }
    }
}

/// Verify `snake()` clamps alpha to [1e-8, 1e6] matching GPU/DSL `SNAKE_MIN_ALPHA`.
#[test]
fn test_snake_clamps_alpha() {
    let x = DynTensor::new(&[1.0f32, 2.0], &[1, 1, 2], &cpu()).unwrap();
    // Alpha below 1e-8 should behave like alpha=1e-8 (SNAKE_MIN_ALPHA)
    let v_tiny = x.snake(1e-12).unwrap().to_flat_vec::<f32>().unwrap();
    let v_min = x.snake(1e-8).unwrap().to_flat_vec::<f32>().unwrap();
    for (a, b) in v_tiny.iter().zip(v_min.iter()) {
        assert!((a - b).abs() < 1e-6, "below 1e-8 should clamp: {a} vs {b}");
    }
    // Alpha above 1e6 should behave like alpha=1e6
    let v_huge = x.snake(1e10).unwrap().to_flat_vec::<f32>().unwrap();
    let v_max = x.snake(1e6).unwrap().to_flat_vec::<f32>().unwrap();
    for (a, b) in v_huge.iter().zip(v_max.iter()) {
        assert!((a - b).abs() < 1e-6, "above 1e6 should clamp: {a} vs {b}");
    }
}

/// Verify snake CPU scalar vs decomposed parity for alpha in [1e-8, 1e-6).
///
/// After W1 commit 8ac5715 changed SNAKE_MIN_ALPHA from 1e-6 to 1e-8,
/// we verify that the decomposed `DynTensor::snake()` path produces the
/// same result as the analytical `snake_scalar` formula for these small alphas.
#[test]
fn test_snake_parity_near_clamp_boundary() {
    let alphas = [1e-8_f64, 5e-8, 1e-7, 5e-7, 1e-6, 1e-5, 0.1, 1.0];
    let x_vals = [0.0_f32, 1.0, -1.0, 100.0, -100.0];
    for &alpha in &alphas {
        let a = alpha.max(1e-8) as f32;
        for &x in &x_vals {
            let expected = x + (1.0 / a) * (a * x).sin().powi(2);
            let t = DynTensor::new(&[x], &[1, 1, 1], &cpu()).unwrap();
            let result = t.snake(alpha).unwrap().to_flat_vec::<f32>().unwrap()[0];
            assert!(
                (result - expected).abs() < 1e-4,
                "snake parity: alpha={alpha}, x={x}: got {result}, expected {expected}"
            );
        }
    }
}

/// Verify per-channel snake activation produces different outputs per channel.
#[test]
fn test_snake_tensor_per_channel() {
    let alpha = DynTensor::new(&[1.0f32, 5.0], &[1, 2, 1], &cpu()).unwrap();
    let x = DynTensor::full(&[1, 2, 4], 1.0, DType::F32, &cpu()).unwrap();
    let out = x.snake_tensor(&alpha).unwrap();
    let flat = out.to_flat_vec::<f32>().unwrap();
    // Channel 0: snake(1.0, alpha=1.0) = 1.0 + sin(1.0)^2 ~ 1.708
    let expected_ch0 = 1.0 + (1.0_f32).sin().powi(2);
    // Channel 1: snake(1.0, alpha=5.0) = 1.0 + (1/5.0) * sin(5.0)^2 ~ 1.184
    let a: f32 = 5.0;
    let expected_ch1 = 1.0 + (1.0 / f64::from(a)) as f32 * (a * 1.0_f32).sin().powi(2);
    for &v in &flat[..4] {
        assert!(
            (v - expected_ch0).abs() < 1e-5,
            "ch0: expected {expected_ch0}, got {v}"
        );
    }
    for &v in &flat[4..8] {
        assert!(
            (v - expected_ch1).abs() < 1e-5,
            "ch1: expected {expected_ch1}, got {v}"
        );
    }
    assert!(
        (flat[0] - flat[4]).abs() > 0.1,
        "per-channel alphas must differ"
    );
}
