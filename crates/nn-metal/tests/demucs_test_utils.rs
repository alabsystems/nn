// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for HTDemucs parity tests.
//!
//! Extracted from `demucs_dconv_debug.rs` to bring that file under 500 lines.
//! Used by `demucs_dconv_debug.rs` and potentially other Demucs test files.
//!
//! Part of #887 — test code-health.

// Shared module included by multiple test binaries via `mod demucs_test_utils;`.
// Each binary uses a subset, so unused function warnings are expected.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::ScalarType;
use nn_metal::{HTDemucsWeights, MetalBackend, PipelineCache, WeightMap};

// ---------------------------------------------------------------------------
// Path and data loading helpers
// ---------------------------------------------------------------------------

pub(crate) fn project_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("project root")
        .to_path_buf()
}

pub(crate) fn load_f32_bin(path: &Path) -> Option<Vec<f32>> {
    if !path.exists() {
        return None;
    }
    let bytes = std::fs::read(path).expect("read binary file");
    assert_eq!(bytes.len() % 4, 0, "file size not aligned to f32");
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Load test weights and audio, returning None if files absent.
pub(crate) fn load_test_weights() -> Option<(HTDemucsWeights, PipelineCache, Vec<f32>)> {
    let root = project_root();
    let weights_path = root.join("models/demucs/htdemucs_temporal.safetensors");
    let audio_path = root.join("models/demucs/parity_audio_1024.bin");
    if !weights_path.exists() || !audio_path.exists() {
        eprintln!("SKIP: weights or parity audio not found");
        return None;
    }
    let backend = MetalBackend::init().expect("Metal backend");
    let ctx = backend.context().clone();
    // SAFETY: Weight file is a valid safetensors file and is not modified during the test.
    let wm = unsafe { WeightMap::load(&weights_path, &ctx).expect("load") };
    let weights = HTDemucsWeights::from_weight_map(&wm).expect("weights");
    let cache = PipelineCache::new(ctx);
    let audio = load_f32_bin(&audio_path).expect("audio");
    Some((weights, cache, audio))
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

/// Compare first N elements and report mismatches.
pub(crate) fn compare_first_n(
    label: &str,
    rust: &[f32],
    python: &[f32],
    n: usize,
    tol: f32,
) -> usize {
    let n = n.min(rust.len()).min(python.len());
    let mut violations = 0;
    for i in 0..n {
        let err = (rust[i] - python[i]).abs();
        if err > tol {
            eprintln!(
                "  {label}[{i}]: rust={:.8}, py={:.8}, err={:.8}",
                rust[i], python[i], err
            );
            violations += 1;
        }
    }
    if violations == 0 {
        eprintln!("{label}: first {n} match within {tol}");
    } else {
        eprintln!("{label}: {violations}/{n} exceed {tol}");
    }
    violations
}

/// Max absolute error between two slices.
pub(crate) fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Compute CPU-expected GroupNorm G=1 output (no affine).
pub(crate) fn cpu_expected_gn(input: &[f32]) -> Vec<f32> {
    let n = input.len() as f64;
    let mean: f64 = input.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var: f64 = input
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n;
    let std_val = (var + 1e-5).sqrt();
    input
        .iter()
        .map(|&v| ((f64::from(v) - mean) / std_val) as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// Audio normalization
// ---------------------------------------------------------------------------

/// Normalize stereo audio using mono mean/std (matching Rust htdemucs.rs).
pub(crate) fn normalize_audio(audio: &[f32], audio_t: usize) -> Vec<f32> {
    let mut mono_sum = 0.0f64;
    for i in 0..audio_t {
        let s = f64::midpoint(f64::from(audio[i]), f64::from(audio[audio_t + i]));
        mono_sum += s;
    }
    let mean = (mono_sum / audio_t as f64) as f32;
    let mut var_sum = 0.0f64;
    for i in 0..audio_t {
        let s = f64::midpoint(f64::from(audio[i]), f64::from(audio[audio_t + i])) as f32;
        let diff = f64::from(s - mean);
        var_sum += diff * diff;
    }
    let std_val = ((var_sum / audio_t as f64).sqrt() as f32).max(1e-8);
    audio.iter().map(|&v| (v - mean) / std_val).collect()
}

// ---------------------------------------------------------------------------
// GPU dispatch helpers
// ---------------------------------------------------------------------------

/// Dispatch GroupNorm G=1 without affine on GPU and return result.
pub(crate) fn dispatch_gn_noaffine(
    cache: &PipelineCache,
    input: &[f32],
    channels: usize,
    t_len: usize,
) -> Vec<f32> {
    let eps_val = [1e-5f32];
    let mut b = TensorBlockBuilder::new("gn_noaff");
    let inp = b.add_input("data", &[channels, t_len]);
    let eps = b.add_input("eps", &[1]);
    let out = b.add_group_norm_g1(inp, eps, None, None, channels, t_len);
    let def = b.build(out).expect("build");
    let mut m: HashMap<&str, &[f32]> = HashMap::new();
    m.insert("data", input);
    m.insert("eps", &eps_val);
    nn_metal::execute_tensor_dispatch(cache, &def, ScalarType::F32, &m).expect("gn dispatch")
}

/// Dispatch a Conv1d + GELU kernel and return result.
pub(crate) fn dispatch_conv_gelu(
    cache: &PipelineCache,
    normalized: &[f32],
    weights: &HTDemucsWeights,
    audio_t: usize,
) -> Vec<f32> {
    let in_ch = 2;
    let out_ch = 48;
    let kernel_size = 8;
    let stride = 4;
    let padding = 2;
    let conv_t_out = (audio_t + 2 * padding - kernel_size) / stride + 1;

    let mut b = TensorBlockBuilder::new("conv0_gelu");
    let d = b.add_input("data", &[in_ch, audio_t]);
    let w = b.add_input("conv_weight", &[out_ch, in_ch, kernel_size]);
    let bi = b.add_input("conv_bias", &[out_ch]);
    let c = b.add_conv1d(d, w, Some(bi), stride, padding, &[out_ch, conv_t_out]);
    let g = b.add_gelu(c, &[out_ch, conv_t_out]);
    let def = b.build(g).expect("build");

    let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
    inputs.insert("data", normalized);
    inputs.insert("conv_weight", &weights.encoder.blocks[0].conv_weight);
    inputs.insert("conv_bias", &weights.encoder.blocks[0].conv_bias);

    nn_metal::execute_tensor_dispatch(cache, &def, ScalarType::F32, &inputs)
        .expect("conv+gelu dispatch")
}

/// Build and dispatch a kernel for: ZeroPad1d + Conv1d (compress).
pub(crate) fn dispatch_compress_conv(
    cache: &PipelineCache,
    gelu_out: &[f32],
    weights: &HTDemucsWeights,
    channels: usize,
    t_len: usize,
) -> Vec<f32> {
    let compressed = channels / 8;
    let kernel_size = 3;
    let dilation = 1;
    let pad_left = (kernel_size - 1) * dilation;
    let padded_t = t_len + pad_left;

    let mut b = TensorBlockBuilder::new("compress_conv");
    let inp = b.add_input("data", &[channels, t_len]);
    let w = b.add_input("weight", &[compressed, channels, kernel_size]);
    let bi = b.add_input("bias", &[compressed]);
    let padded = b.add_zero_pad_1d(inp, pad_left, 0, &[channels, padded_t]);
    let c = b.add_conv1d_full(padded, w, Some(bi), 1, 0, dilation, 1, &[compressed, t_len]);
    let def = b.build(c).expect("build");

    let dc = &weights.encoder.blocks[0].dconv[0];
    let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
    inputs.insert("data", gelu_out);
    inputs.insert("weight", &dc.conv_compress_weight);
    inputs.insert("bias", &dc.conv_compress_bias);

    nn_metal::execute_tensor_dispatch(cache, &def, ScalarType::F32, &inputs)
        .expect("compress conv dispatch")
}

/// Build and dispatch GroupNorm(G=1) with affine on the compressed output.
pub(crate) fn dispatch_group_norm_g1(
    cache: &PipelineCache,
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    t_len: usize,
) -> Vec<f32> {
    let eps_val = [1e-5f32];
    let mut b = TensorBlockBuilder::new("gn1");
    let inp = b.add_input("data", &[channels, t_len]);
    let eps = b.add_input("eps", &[1]);
    let g = b.add_input("gamma", &[channels]);
    let bt = b.add_input("beta", &[channels]);
    let out = b.add_group_norm_g1(inp, eps, Some(g), Some(bt), channels, t_len);
    let def = b.build(out).expect("build");

    let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
    inputs.insert("data", input);
    inputs.insert("eps", &eps_val);
    inputs.insert("gamma", gamma);
    inputs.insert("beta", beta);

    nn_metal::execute_tensor_dispatch(cache, &def, ScalarType::F32, &inputs)
        .expect("group_norm dispatch")
}

/// Build full DConv sublayer 0 graph and return output node id.
pub(crate) fn build_dconv0_graph(
    b: &mut TensorBlockBuilder,
    channels: usize,
    t_len: usize,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let compressed = channels / 8;
    let doubled = channels * 2;
    let dconv_kernel = 3;
    let dilation: usize = 1;
    let pad_left = (dconv_kernel - 1) * dilation;
    let padded_t = t_len + pad_left;

    let inp = b.add_input("data", &[channels, t_len]);
    let cw = b.add_input("cw", &[compressed, channels, dconv_kernel]);
    let cb = b.add_input("cb", &[compressed]);
    let padded = b.add_zero_pad_1d(inp, pad_left, 0, &[channels, padded_t]);
    let c1 = b.add_conv1d_full(
        padded,
        cw,
        Some(cb),
        1,
        0,
        dilation,
        1,
        &[compressed, t_len],
    );

    let eps1 = b.add_input("eps1", &[1]);
    let ng = b.add_input("ng", &[compressed]);
    let nb = b.add_input("nb", &[compressed]);
    let n1 = b.add_group_norm_g1(c1, eps1, Some(ng), Some(nb), compressed, t_len);
    let g1 = b.add_gelu(n1, &[compressed, t_len]);

    let ew = b.add_input("ew", &[doubled, compressed, 1]);
    let eb = b.add_input("eb", &[doubled]);
    let c2 = b.add_conv1d(g1, ew, Some(eb), 1, 0, &[doubled, t_len]);

    let eps2 = b.add_input("eps2", &[1]);
    let eng = b.add_input("eng", &[doubled]);
    let enb = b.add_input("enb", &[doubled]);
    let n2 = b.add_group_norm_g1(c2, eps2, Some(eng), Some(enb), doubled, t_len);

    let glu = b.add_glu(n2, 0, &[doubled, t_len]).expect("glu");
    let ls_w = b.add_input("ls", &[channels]);
    let ls = b.add_layer_scale(glu, ls_w, &[channels, t_len]);
    b.add_binary_add(inp, ls, &[channels, t_len])
}
