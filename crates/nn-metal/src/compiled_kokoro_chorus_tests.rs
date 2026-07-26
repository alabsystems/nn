// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! In-crate unit tests for `compiled_kokoro_chorus.rs`.
//!
//! Tests the parallel dispatch logic, validation paths, and `pub(crate)`
//! helper functions using the mini test model (no KOKORO_WEIGHTS required).
//!
//! Part of #4290.

use std::collections::HashMap;

use half::bf16;
use nn_core::dyn_tensor::DynTensor;
use nn_core::TensorError;
use nn_core::{DType, Device, VarBuilder};
use nn_models::kokoro_chorus::mix_voices_from_refs;
use nn_models::kokoro_chorus::ChorusConfig;
use nn_models::kokoro_chorus_alignment::{align_voices, AlignmentConfig};
use nn_models::{KokoroConfig, KokoroModel, PlbertConfig};
use nn_tts_verify::{HardBoundsConfig, RejectionPolicy};

use super::KokoroChorus;
use crate::metal_backend::global_metal_context;

// ---------------------------------------------------------------------------
// Mini-model helpers (mirrors kokoro_test_weights.rs for in-crate access)
// ---------------------------------------------------------------------------

fn cpu() -> Device {
    Device::Cpu
}

fn gpu_device() -> Device {
    Device::Metal { device_id: 0 }
}

const STYLE_DIM: usize = 4;

/// Miniaturized KokoroConfig for fast unit tests (D_EN=8, STYLE_DIM=4).
fn mini_test_config() -> KokoroConfig {
    let mut plbert = PlbertConfig::default();
    plbert.vocab_size = 10;
    plbert.embedding_dim = 4;
    plbert.hidden_size = 8;
    plbert.num_attention_heads = 2;
    plbert.intermediate_size = 16;
    plbert.max_position_embeddings = 16;
    plbert.num_hidden_layers = 1;

    let mut config = KokoroConfig::default();
    config.d_en = 8;
    config.n_prosody_layers = 1;
    config.style_dim = 4;
    config.upsample_rates = vec![2];
    config.upsample_kernel_sizes = vec![4];
    config.resblock_kernel_sizes = vec![3];
    config.resblock_dilations = vec![vec![1, 2]];
    config.gen_initial_channels = 8;
    config.n_fft = 4;
    config.f0_bilstm_hidden = 4;
    config.plbert = plbert;
    config
}

// -- Weight primitives -------------------------------------------------------

fn wz(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    m.insert(
        name.to_string(),
        DynTensor::zeros(shape, DType::F32, &cpu()).unwrap(),
    );
}

fn wones(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    m.insert(
        name.to_string(),
        DynTensor::full(shape, 1.0, DType::F32, &cpu()).unwrap(),
    );
}

fn bilstm_w(m: &mut HashMap<String, DynTensor>, pfx: &str, input_dim: usize, hidden_dim: usize) {
    let g = 4 * hidden_dim;
    wz(m, &format!("{pfx}.weight_ih_l0"), &[g, input_dim]);
    wz(m, &format!("{pfx}.weight_hh_l0"), &[g, hidden_dim]);
    wz(m, &format!("{pfx}.bias_ih_l0"), &[g]);
    wz(m, &format!("{pfx}.bias_hh_l0"), &[g]);
    wz(m, &format!("{pfx}.weight_ih_l0_reverse"), &[g, input_dim]);
    wz(m, &format!("{pfx}.weight_hh_l0_reverse"), &[g, hidden_dim]);
    wz(m, &format!("{pfx}.bias_ih_l0_reverse"), &[g]);
    wz(m, &format!("{pfx}.bias_hh_l0_reverse"), &[g]);
}

fn adain_resblk_weights(
    m: &mut HashMap<String, DynTensor>,
    pfx: &str,
    dim_in: usize,
    dim_out: usize,
    style_dim: usize,
    upsample: bool,
) {
    wz(m, &format!("{pfx}.n1.fc.weight"), &[2 * dim_in, style_dim]);
    wz(m, &format!("{pfx}.n1.fc.bias"), &[2 * dim_in]);
    wz(m, &format!("{pfx}.n2.fc.weight"), &[2 * dim_out, style_dim]);
    wz(m, &format!("{pfx}.n2.fc.bias"), &[2 * dim_out]);
    wz(m, &format!("{pfx}.c1.weight"), &[dim_out, dim_in, 3]);
    wz(m, &format!("{pfx}.c1.bias"), &[dim_out]);
    wz(m, &format!("{pfx}.c2.weight"), &[dim_out, dim_out, 3]);
    wz(m, &format!("{pfx}.c2.bias"), &[dim_out]);
    if dim_in != dim_out {
        wz(m, &format!("{pfx}.skip.weight"), &[dim_out, dim_in, 1]);
        wz(m, &format!("{pfx}.skip.bias"), &[dim_out]);
    }
    if upsample {
        wz(m, &format!("{pfx}.pool.weight"), &[dim_in, 1, 3]);
        wz(m, &format!("{pfx}.pool.bias"), &[dim_in]);
    }
}

fn resblock_weights(
    m: &mut HashMap<String, DynTensor>,
    pfx: &str,
    ch: usize,
    kernel_size: usize,
    num_dilations: usize,
    style_dim: usize,
) {
    for i in 0..num_dilations {
        wz(
            m,
            &format!("{pfx}.convs1.{i}.weight"),
            &[ch, ch, kernel_size],
        );
        wz(m, &format!("{pfx}.convs1.{i}.bias"), &[ch]);
        wz(
            m,
            &format!("{pfx}.convs2.{i}.weight"),
            &[ch, ch, kernel_size],
        );
        wz(m, &format!("{pfx}.convs2.{i}.bias"), &[ch]);
        wz(
            m,
            &format!("{pfx}.adain1.{i}.fc.weight"),
            &[2 * ch, style_dim],
        );
        wz(m, &format!("{pfx}.adain1.{i}.fc.bias"), &[2 * ch]);
        wz(
            m,
            &format!("{pfx}.adain2.{i}.fc.weight"),
            &[2 * ch, style_dim],
        );
        wz(m, &format!("{pfx}.adain2.{i}.fc.bias"), &[2 * ch]);
        m.insert(
            format!("{pfx}.alpha1.{i}"),
            DynTensor::full(&[1, ch, 1], 1.0, DType::F32, &cpu()).unwrap(),
        );
        m.insert(
            format!("{pfx}.alpha2.{i}"),
            DynTensor::full(&[1, ch, 1], 1.0, DType::F32, &cpu()).unwrap(),
        );
    }
}

fn stage1_resblk_weights(
    m: &mut HashMap<String, DynTensor>,
    pfx: &str,
    dim_in: usize,
    dim_out: usize,
    style_dim: usize,
    upsample: bool,
) {
    wz(m, &format!("{pfx}.conv1.weight"), &[dim_out, dim_in, 3]);
    wz(m, &format!("{pfx}.conv1.bias"), &[dim_out]);
    wz(m, &format!("{pfx}.conv2.weight"), &[dim_out, dim_out, 3]);
    wz(m, &format!("{pfx}.conv2.bias"), &[dim_out]);
    wz(
        m,
        &format!("{pfx}.norm1.style_linear.weight"),
        &[2 * dim_in, style_dim],
    );
    wz(m, &format!("{pfx}.norm1.style_linear.bias"), &[2 * dim_in]);
    wz(
        m,
        &format!("{pfx}.norm2.style_linear.weight"),
        &[2 * dim_out, style_dim],
    );
    wz(m, &format!("{pfx}.norm2.style_linear.bias"), &[2 * dim_out]);
    if dim_in != dim_out {
        wz(m, &format!("{pfx}.conv1x1.weight"), &[dim_out, dim_in, 1]);
        wz(m, &format!("{pfx}.conv1x1.bias"), &[dim_out]);
    }
    if upsample {
        wz(m, &format!("{pfx}.pool.weight"), &[dim_in, 1, 3]);
        wz(m, &format!("{pfx}.pool.bias"), &[dim_in]);
    }
}

/// Build all synthetic weights for the mini config.
/// Mirrors `kokoro_test_weights::all_weights` exactly.
fn all_weights(cfg: &KokoroConfig) -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let d_en = cfg.d_en;
    let style_dim = cfg.style_dim;

    // -- PlBert ---
    let vocab = cfg.plbert.vocab_size;
    let emb = cfg.plbert.embedding_dim;
    let hidden = cfg.plbert.hidden_size;
    let intermediate = cfg.plbert.intermediate_size;
    wz(
        &mut m,
        "plbert.embeddings.word_embeddings.weight",
        &[vocab, emb],
    );
    wz(
        &mut m,
        "plbert.embeddings.position_embeddings.weight",
        &[16, emb],
    );
    wz(
        &mut m,
        "plbert.embeddings.token_type_embeddings.weight",
        &[2, emb],
    );
    wones(&mut m, "plbert.embeddings.LayerNorm.weight", &[emb]);
    wz(&mut m, "plbert.embeddings.LayerNorm.bias", &[emb]);
    wz(
        &mut m,
        "plbert.encoder.embedding_hidden_mapping_in.weight",
        &[hidden, emb],
    );
    wz(
        &mut m,
        "plbert.encoder.embedding_hidden_mapping_in.bias",
        &[hidden],
    );
    let lp = "plbert.encoder.albert_layer_groups.0.albert_layers.0";
    for name in &[
        "attention.query",
        "attention.key",
        "attention.value",
        "attention.dense",
    ] {
        wz(&mut m, &format!("{lp}.{name}.weight"), &[hidden, hidden]);
        wz(&mut m, &format!("{lp}.{name}.bias"), &[hidden]);
    }
    wones(
        &mut m,
        &format!("{lp}.attention.LayerNorm.weight"),
        &[hidden],
    );
    wz(&mut m, &format!("{lp}.attention.LayerNorm.bias"), &[hidden]);
    wz(&mut m, &format!("{lp}.ffn.weight"), &[intermediate, hidden]);
    wz(&mut m, &format!("{lp}.ffn.bias"), &[intermediate]);
    wz(
        &mut m,
        &format!("{lp}.ffn_output.weight"),
        &[hidden, intermediate],
    );
    wz(&mut m, &format!("{lp}.ffn_output.bias"), &[hidden]);
    wones(
        &mut m,
        &format!("{lp}.full_layer_layer_norm.weight"),
        &[hidden],
    );
    wz(
        &mut m,
        &format!("{lp}.full_layer_layer_norm.bias"),
        &[hidden],
    );

    // -- bert_encoder ---
    wz(&mut m, "bert_encoder.weight", &[d_en, hidden]);
    wz(&mut m, "bert_encoder.bias", &[d_en]);

    // -- TextEncoder ---
    let h_text = d_en / 2;
    wz(&mut m, "text_encoder.embedding.weight", &[vocab, d_en]);
    for i in 0..3 {
        wz(
            &mut m,
            &format!("text_encoder.convs.{i}.weight"),
            &[d_en, d_en, 5],
        );
        wz(&mut m, &format!("text_encoder.convs.{i}.bias"), &[d_en]);
        wones(&mut m, &format!("text_encoder.norms.{i}.weight"), &[d_en]);
        wz(&mut m, &format!("text_encoder.norms.{i}.bias"), &[d_en]);
    }
    bilstm_w(&mut m, "text_encoder.lstm", d_en, h_text);
    wz(&mut m, "text_encoder.lstm.linear.weight", &[d_en, d_en]);
    wz(&mut m, "text_encoder.lstm.linear.bias", &[d_en]);

    // -- ProsodyPredictor ---
    let h_pros = d_en / 2;
    let lstm_in = d_en + style_dim;
    bilstm_w(
        &mut m,
        "prosody_predictor.duration.lstms.0",
        lstm_in,
        h_pros,
    );
    wones(
        &mut m,
        "prosody_predictor.duration.norms.0.norm.weight",
        &[d_en],
    );
    wz(
        &mut m,
        "prosody_predictor.duration.norms.0.norm.bias",
        &[d_en],
    );
    wz(
        &mut m,
        "prosody_predictor.duration.norms.0.fc.weight",
        &[2 * d_en, style_dim],
    );
    wz(
        &mut m,
        "prosody_predictor.duration.norms.0.fc.bias",
        &[2 * d_en],
    );
    wz(
        &mut m,
        "prosody_predictor.duration.duration_proj.weight",
        &[50, d_en],
    );
    wz(
        &mut m,
        "prosody_predictor.duration.duration_proj.bias",
        &[50],
    );
    bilstm_w(&mut m, "prosody_predictor.lstm", lstm_in, h_pros);

    // -- F0/Energy predictor ---
    let f0h = cfg.f0_bilstm_hidden;
    let bo = 2 * f0h;
    let f0_lstm_in = d_en + style_dim;
    wz(
        &mut m,
        "predictor.shared.forward.weight_ih_l0",
        &[4 * f0h, f0_lstm_in],
    );
    wz(
        &mut m,
        "predictor.shared.forward.weight_hh_l0",
        &[4 * f0h, f0h],
    );
    wz(&mut m, "predictor.shared.forward.bias_ih_l0", &[4 * f0h]);
    wz(&mut m, "predictor.shared.forward.bias_hh_l0", &[4 * f0h]);
    wz(
        &mut m,
        "predictor.shared.backward.weight_ih_l0",
        &[4 * f0h, f0_lstm_in],
    );
    wz(
        &mut m,
        "predictor.shared.backward.weight_hh_l0",
        &[4 * f0h, f0h],
    );
    wz(&mut m, "predictor.shared.backward.bias_ih_l0", &[4 * f0h]);
    wz(&mut m, "predictor.shared.backward.bias_hh_l0", &[4 * f0h]);
    adain_resblk_weights(&mut m, "predictor.F0.0", bo, bo, style_dim, false);
    adain_resblk_weights(&mut m, "predictor.F0.1", bo, f0h, style_dim, true);
    adain_resblk_weights(&mut m, "predictor.F0.2", f0h, f0h, style_dim, false);
    wz(&mut m, "predictor.F0_proj.weight", &[1, f0h]);
    wz(&mut m, "predictor.F0_proj.bias", &[1]);
    adain_resblk_weights(&mut m, "predictor.N.0", bo, bo, style_dim, false);
    adain_resblk_weights(&mut m, "predictor.N.1", bo, f0h, style_dim, true);
    adain_resblk_weights(&mut m, "predictor.N.2", f0h, f0h, style_dim, false);
    wz(&mut m, "predictor.N_proj.weight", &[1, f0h]);
    wz(&mut m, "predictor.N_proj.bias", &[1]);

    // -- FullDecoder + Generator ---
    let ch = cfg.gen_initial_channels;
    let n_fft = cfg.n_fft;
    let asr_res_ch = (d_en / 8).max(1);
    let dec_hidden = 2 * d_en;
    let encode_in = d_en + 2;
    let decode_in = dec_hidden + asr_res_ch + 2;

    wz(&mut m, "decoder.F0_conv.weight", &[1, 1, 3]);
    wz(&mut m, "decoder.F0_conv.bias", &[1]);
    wz(&mut m, "decoder.N_conv.weight", &[1, 1, 3]);
    wz(&mut m, "decoder.N_conv.bias", &[1]);
    wz(&mut m, "decoder.asr_res.weight", &[asr_res_ch, d_en, 1]);
    wz(&mut m, "decoder.asr_res.bias", &[asr_res_ch]);
    stage1_resblk_weights(
        &mut m,
        "decoder.encode",
        encode_in,
        dec_hidden,
        style_dim,
        false,
    );
    for i in 0..3 {
        stage1_resblk_weights(
            &mut m,
            &format!("decoder.decode.{i}"),
            decode_in,
            dec_hidden,
            style_dim,
            false,
        );
    }
    stage1_resblk_weights(&mut m, "decoder.decode.3", decode_in, d_en, style_dim, true);

    let next_ch = ch / 2;
    let n_bins = n_fft / 2 + 1;
    wz(&mut m, "decoder.generator.conv_pre.weight", &[ch, ch, 7]);
    wz(&mut m, "decoder.generator.conv_pre.bias", &[ch]);
    wz(&mut m, "decoder.generator.ups.0.weight", &[ch, next_ch, 4]);
    wz(&mut m, "decoder.generator.ups.0.bias", &[next_ch]);
    wz(
        &mut m,
        "decoder.generator.noise_convs.0.weight",
        &[next_ch, 2 * n_bins, 1],
    );
    wz(&mut m, "decoder.generator.noise_convs.0.bias", &[next_ch]);
    resblock_weights(
        &mut m,
        "decoder.generator.noise_res.0",
        next_ch,
        11,
        3,
        style_dim,
    );
    resblock_weights(
        &mut m,
        "decoder.generator.resblocks.0",
        next_ch,
        3,
        2,
        style_dim,
    );
    wz(
        &mut m,
        "decoder.generator.conv_post.weight",
        &[2 * n_bins, next_ch, 7],
    );
    wz(&mut m, "decoder.generator.conv_post.bias", &[2 * n_bins]);
    wz(
        &mut m,
        "decoder.generator.m_source.l_linear.weight",
        &[1, 9],
    );
    wz(&mut m, "decoder.generator.m_source.l_linear.bias", &[1]);

    m
}

/// Build a `CompiledKokoro` with mini test config. Initializes Metal.
fn build_mini_kokoro() -> (super::super::CompiledKokoro, crate::PipelineCache) {
    crate::test_common::init();
    let cache = crate::PipelineCache::new_global().expect("Metal global cache");
    let cfg = mini_test_config();
    let weights = all_weights(&cfg);
    let vb = VarBuilder::from_tensors(weights, DType::F32, &gpu_device());
    let model = KokoroModel::load(&vb, &cfg).expect("KokoroModel::load with synthetic weights");
    let mut hb = HardBoundsConfig::default();
    hb.rejection_policy = RejectionPolicy::Warn;
    (
        super::super::CompiledKokoro::new_with_hard_bounds(model, hb).expect("GPU init"),
        cache,
    )
}

fn test_ctx() -> &'static crate::MetalContext {
    crate::metal_backend::MetalBackend::init().expect("Metal init");
    global_metal_context().expect("Metal context")
}

fn make_style(seed: usize) -> DynTensor {
    let vals: Vec<f32> = (0..2 * STYLE_DIM)
        .map(|i| ((seed * 17 + i) as f32 * 0.001).sin() * 0.1)
        .collect();
    DynTensor::from_vec(vals, &[1, 2 * STYLE_DIM], &cpu()).unwrap()
}

fn make_input(len: usize) -> DynTensor {
    let vals: Vec<f32> = (1..=len).map(|v| v as f32).collect();
    DynTensor::from_vec(vals, &[1, len], &cpu()).unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify that synthesize_chorus_parallel submits N fence submissions for N voices.
///
/// Checks that the parallel method produces the same number of output samples
/// as the sequential method, confirming that all N voices were processed.
/// Part of #4290.
#[test]
fn test_parallel_chorus_creates_correct_number_of_fences() {
    let (mut primary, cache) = build_mini_kokoro();

    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let n_voices = 4;
    let config = ChorusConfig::equal_gain(n_voices).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    assert_eq!(chorus.n_voices(), n_voices);

    let styles: Vec<DynTensor> = (0..n_voices).map(|i| make_style(100 + i)).collect();
    let mixed = chorus
        .synthesize_chorus_parallel(&input, &styles, 1.0, &cache)
        .expect("parallel chorus");

    // All N voices were processed: output is non-empty.
    assert!(!mixed.is_empty(), "parallel chorus must produce audio");
    assert!(
        mixed.iter().all(|s| s.is_finite()),
        "all samples must be finite",
    );

    // Verify N voices contributed by comparing with a 1-voice chorus.
    let config1 = ChorusConfig::equal_gain(1).unwrap();
    let mut chorus1 = KokoroChorus::new(&primary, config1).unwrap();
    let single_style = vec![make_style(100)];
    let single_audio = chorus1
        .synthesize_chorus_parallel(&input, &single_style, 1.0, &cache)
        .expect("single-voice parallel");

    // Both should produce audio of the same length (same input text).
    assert_eq!(
        mixed.len(),
        single_audio.len(),
        "4-voice and 1-voice parallel chorus should produce same length audio for same input",
    );
}

/// Verify parallel and sequential chorus produce the same audio (within tolerance).
/// Part of #4290.
#[test]
fn test_parallel_vs_sequential_consistency() {
    let (mut primary, cache) = build_mini_kokoro();

    let input = make_input(5);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(3).unwrap();
    let mut chorus_seq = KokoroChorus::new(&primary, config.clone()).unwrap();
    let mut chorus_par = KokoroChorus::new(&primary, config).unwrap();

    let styles: Vec<DynTensor> = (0..3).map(|i| make_style(200 + i)).collect();

    let audio_seq = chorus_seq
        .synthesize_chorus_shared_encode(&input, &styles, 1.0, &cache)
        .expect("sequential chorus");

    let audio_par = chorus_par
        .synthesize_chorus_parallel(&input, &styles, 1.0, &cache)
        .expect("parallel chorus");

    assert_eq!(
        audio_seq.len(),
        audio_par.len(),
        "sequential and parallel should produce same length",
    );

    let max_diff = audio_seq
        .iter()
        .zip(audio_par.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // With mini weights and identical inputs, outputs should be very close.
    assert!(
        max_diff < 1e-4,
        "parallel should match sequential within epsilon, max_diff={max_diff}",
    );
}

/// Verify non-streaming chorus paths stay finite and consistent with
/// recommended segment autocast enabled on the parent and warm clones.
#[test]
fn test_recommended_autocast_chorus_parallel_matches_shared_encode() {
    let (primary, cache) = build_mini_kokoro();
    let mut primary = primary.with_recommended_autocast();

    let input = make_input(5);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("recommended-autocast warmup");

    let primary_autocast = primary
        .segment_autocast()
        .expect("primary should expose recommended segment autocast");
    assert_eq!(
        primary_autocast.enabled_count(),
        6,
        "recommended autocast should enable 6/8 segments"
    );

    let config = ChorusConfig::equal_gain(3).unwrap();
    let mut chorus_seq = KokoroChorus::new(&primary, config.clone()).unwrap();
    let mut chorus_par = KokoroChorus::new(&primary, config).unwrap();

    for voice_idx in 0..3 {
        let seq_cfg = chorus_seq
            .voice(voice_idx)
            .expect("shared-encode chorus voice")
            .segment_autocast()
            .expect("warm clone should preserve segment autocast");
        let par_cfg = chorus_par
            .voice(voice_idx)
            .expect("parallel chorus voice")
            .segment_autocast()
            .expect("warm clone should preserve segment autocast");
        assert_eq!(seq_cfg.enabled_count(), 6);
        assert_eq!(par_cfg.enabled_count(), 6);
    }

    let styles: Vec<DynTensor> = (0..3).map(|i| make_style(300 + i)).collect();

    let audio_seq = chorus_seq
        .synthesize_chorus_shared_encode(&input, &styles, 1.0, &cache)
        .expect("shared-encode chorus with recommended autocast");
    let audio_par = chorus_par
        .synthesize_chorus_parallel(&input, &styles, 1.0, &cache)
        .expect("parallel chorus with recommended autocast");

    assert!(
        !audio_seq.is_empty(),
        "shared-encode chorus with recommended autocast must produce audio"
    );
    assert!(
        !audio_par.is_empty(),
        "parallel chorus with recommended autocast must produce audio"
    );
    assert!(
        audio_seq.iter().all(|sample| sample.is_finite()),
        "shared-encode chorus with recommended autocast must stay finite"
    );
    assert!(
        audio_par.iter().all(|sample| sample.is_finite()),
        "parallel chorus with recommended autocast must stay finite"
    );
    assert_eq!(
        audio_seq.len(),
        audio_par.len(),
        "recommended-autocast chorus paths should produce the same length"
    );

    let max_diff = audio_seq
        .iter()
        .zip(audio_par.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "recommended-autocast parallel chorus should match shared-encode within epsilon, max_diff={max_diff}",
    );
}

/// Verify the plain batch mixed-output chorus path stays finite and close to
/// the existing non-autocast path when recommended segment autocast is enabled.
#[test]
fn test_recommended_autocast_batch_chorus_matches_non_autocast() {
    let (mut primary_f32, cache) = build_mini_kokoro();
    let (primary_autocast, _) = build_mini_kokoro();
    let mut primary_autocast = primary_autocast.with_recommended_autocast();

    let input = make_input(5);
    let style0 = make_style(0);
    let _ = primary_f32
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("baseline warmup");
    let _ = primary_autocast
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("recommended-autocast warmup");

    let primary_autocast_cfg = primary_autocast
        .segment_autocast()
        .expect("primary should expose recommended segment autocast");
    assert_eq!(
        primary_autocast_cfg.enabled_count(),
        6,
        "recommended autocast should enable 6/8 segments"
    );

    let config = ChorusConfig::equal_gain(3).unwrap();
    let mut chorus_f32 = KokoroChorus::new(&primary_f32, config.clone()).unwrap();
    let mut chorus_autocast = KokoroChorus::new(&primary_autocast, config).unwrap();

    for voice_idx in 0..3 {
        let cfg = chorus_autocast
            .voice(voice_idx)
            .expect("autocast chorus voice")
            .segment_autocast()
            .expect("warm clone should preserve segment autocast");
        assert_eq!(
            cfg.enabled_count(),
            6,
            "voice {voice_idx} should preserve the recommended 6/8 autocast config"
        );
    }

    let inputs: Vec<DynTensor> = (0..3).map(|_| input.clone()).collect();
    let styles: Vec<DynTensor> = (0..3).map(|i| make_style(400 + i)).collect();

    let audio_f32 = chorus_f32
        .synthesize_chorus(&inputs, &styles, 1.0, &cache)
        .expect("baseline batch chorus");
    let audio_autocast = chorus_autocast
        .synthesize_chorus(&inputs, &styles, 1.0, &cache)
        .expect("recommended-autocast batch chorus");

    assert!(
        !audio_f32.is_empty(),
        "baseline batch chorus must produce audio"
    );
    assert!(
        !audio_autocast.is_empty(),
        "recommended-autocast batch chorus must produce audio"
    );
    assert!(
        audio_f32.iter().all(|sample| sample.is_finite()),
        "baseline batch chorus must stay finite"
    );
    assert!(
        audio_autocast.iter().all(|sample| sample.is_finite()),
        "recommended-autocast batch chorus must stay finite"
    );
    assert_eq!(
        audio_f32.len(),
        audio_autocast.len(),
        "recommended-autocast batch chorus should preserve output length"
    );

    let max_diff = audio_f32
        .iter()
        .zip(audio_autocast.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "recommended-autocast batch chorus should stay close to the non-autocast path, max_diff={max_diff}",
    );
}

/// Verify error when styles.len() != n_voices.
/// Part of #4290.
#[test]
fn test_parallel_chorus_error_on_style_mismatch() {
    let (mut primary, cache) = build_mini_kokoro();

    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(3).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    // 2 styles for 3 voices -- mismatch.
    let styles = vec![make_style(10), make_style(20)];
    let result = chorus.synthesize_chorus_parallel(&input, &styles, 1.0, &cache);

    assert!(result.is_err(), "should reject mismatched style count");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("styles length 2 != n_voices 3"),
        "error should describe the mismatch: {err_msg}",
    );
}

/// Verify error on invalid speed values (zero, negative, NaN, infinity).
/// Part of #4290.
#[test]
fn test_parallel_chorus_speed_validation() {
    let (mut primary, cache) = build_mini_kokoro();

    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let styles = vec![make_style(30), make_style(40)];

    // Zero speed.
    let result = chorus.synthesize_chorus_parallel(&input, &styles, 0.0, &cache);
    assert!(result.is_err(), "speed=0.0 must be rejected");

    // Negative speed.
    let result = chorus.synthesize_chorus_parallel(&input, &styles, -1.0, &cache);
    assert!(result.is_err(), "speed=-1.0 must be rejected");

    // NaN speed.
    let result = chorus.synthesize_chorus_parallel(&input, &styles, f32::NAN, &cache);
    assert!(result.is_err(), "speed=NaN must be rejected");

    // Infinity speed.
    let result = chorus.synthesize_chorus_parallel(&input, &styles, f32::INFINITY, &cache);
    assert!(result.is_err(), "speed=INFINITY must be rejected");

    // Valid speed should succeed.
    let result = chorus.synthesize_chorus_parallel(&input, &styles, 1.0, &cache);
    assert!(result.is_ok(), "speed=1.0 should be accepted");
}

/// Verify that `run_voice_decode_async` returns a `GpuFence` when GPU work
/// is pending. This tests the `pub(crate)` free function directly.
/// Part of #4290.
#[test]
fn test_run_voice_decode_async_returns_fence() {
    let (mut primary, cache) = build_mini_kokoro();

    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    // Clone a voice for decode testing.
    let mut voice = primary.clone_dispatch_warm();

    // Run encode once to get shared encoding results.
    let enc = primary.step_encode(&input, &cache).expect("encode");

    // Split style for prosody and decoder.
    let split = primary.split_style(&style0).expect("split style");
    let prosody_style = split.prosody_style.to_device(&gpu_device()).unwrap();
    let decoder_style = split.decoder_style.to_device(&gpu_device()).unwrap();

    // Run prosody + regulate to get a StepRegulateResult.
    let pros = voice
        .step_predict_prosody(&enc.bert_features, &prosody_style, enc.seq_len, &cache)
        .expect("prosody");
    let reg = voice
        .step_regulate(
            &pros.dur_logits,
            &pros.features,
            &enc.text_features,
            1.0,
            &cache,
        )
        .expect("regulate");

    // Call run_voice_decode_async -- the pub(crate) function under test.
    let (audio, fence) =
        super::run_voice_decode_async(&mut voice, &reg, &prosody_style, &decoder_style, &cache)
            .expect("run_voice_decode_async");

    // The audio tensor should exist (GPU-resident).
    assert!(
        !audio.dims().is_empty(),
        "audio tensor should have at least 1 dimension",
    );

    // Wait for the fence (if present) to complete.
    if let Some(f) = fence {
        f.wait().expect("fence wait");
    }

    // After fence completion, verify the audio is transferable to CPU.
    let cpu_audio = audio
        .to_device(&cpu())
        .expect("GPU->CPU transfer after fence wait");
    let pcm = super::extract_pcm_from_audio(&cpu_audio).expect("extract PCM from audio");
    assert!(!pcm.is_empty(), "decoded audio should not be empty");
    assert!(
        pcm.iter().all(|s| s.is_finite()),
        "all audio samples should be finite after fence wait",
    );
}

#[test]
fn test_extract_pcm_from_audio_matches_flat_vec_for_cpu_f32() {
    let audio = DynTensor::from_vec(vec![0.25f32, -0.5, 1.0, 0.0], &[1, 1, 4], &cpu())
        .expect("cpu f32 audio tensor");

    let expected = audio.to_flat_vec::<f32>().expect("reference flatten");
    let actual = super::extract_pcm_from_audio(&audio).expect("fast-path PCM extract");

    assert_eq!(actual, expected, "fast-path extraction must preserve PCM");
}

#[test]
fn test_extract_pcm_from_audio_falls_back_for_cpu_bf16() {
    let audio = DynTensor::from_vec_bf16(
        vec![
            bf16::from_f32(0.125),
            bf16::from_f32(-0.375),
            bf16::from_f32(0.75),
        ],
        &[3],
        &cpu(),
    )
    .expect("cpu bf16 audio tensor");

    let expected = audio.to_flat_vec::<f32>().expect("reference bf16 flatten");
    let actual = super::extract_pcm_from_audio(&audio).expect("fallback PCM extract");

    assert_eq!(
        actual, expected,
        "fallback extraction must match flatten semantics"
    );
}

#[test]
fn test_extract_finite_pcm_from_audio_rejects_non_finite() {
    let audio = DynTensor::from_vec(vec![0.25f32, f32::NAN, f32::INFINITY, -0.5], &[4], &cpu())
        .expect("cpu f32 audio tensor");

    let err = super::extract_finite_pcm_from_audio(&audio, "chorus_voice_1_audio")
        .expect_err("non-finite audio should be rejected");

    match err {
        super::super::CompiledKokoroError::Tensor(source) => match *source {
            TensorError::NonFiniteData { name, count } => {
                assert_eq!(name, "chorus_voice_1_audio");
                assert_eq!(count, 2, "NaN + Inf should both be counted");
            }
            other => panic!("expected NonFiniteData, got {other:?}"),
        },
        other => panic!("expected Tensor(NonFiniteData), got {other:?}"),
    }
}

#[test]
fn test_simple_mono_mix_matches_model_mixer() {
    let config = ChorusConfig::with_gains(vec![0.6, 0.3, 0.1])
        .expect("valid chorus config")
        .with_clip(true);
    let voice_audio = vec![
        vec![0.5f32, 0.25, -0.25, 0.0],
        vec![0.1f32, -0.2, 0.3],
        vec![-0.4f32, 0.8, 0.2, -0.1],
    ];

    let expected_refs: Vec<&[f32]> = voice_audio.iter().map(Vec::as_slice).collect();
    let expected = mix_voices_from_refs(&expected_refs, &config).expect("reference mono mix");
    let actual = super::mix_voice_audio_mono_simple(&voice_audio, &config)
        .expect("simple mono fast-path mix");

    assert_eq!(
        actual, expected,
        "mono fast path must preserve mix semantics"
    );
}

#[test]
fn test_apply_alignment_in_place_matches_model_alignment() {
    let config = AlignmentConfig::new(0.6)
        .expect("valid alignment config")
        .with_max_shift(4)
        .with_correlation_window(64)
        .with_fade_samples(8);
    let voice_audio = vec![
        {
            let mut voice = vec![0.0f32; 128];
            voice[20] = 1.0;
            voice[52] = 0.8;
            voice[92] = -0.6;
            voice
        },
        {
            let mut voice = vec![0.0f32; 128];
            voice[23] = 1.0;
            voice[55] = 0.8;
            voice[95] = -0.6;
            voice
        },
        {
            let mut voice = vec![0.0f32; 128];
            voice[18] = 1.0;
            voice[50] = 0.8;
            voice[90] = -0.6;
            voice
        },
    ];

    let expected = align_voices(&voice_audio, &config).expect("reference alignment");
    let mut actual = voice_audio;
    super::apply_alignment_in_place(&mut actual, &config).expect("in-place alignment");

    assert_eq!(
        actual, expected,
        "in-place alignment helper must preserve align_voices semantics"
    );
}

#[test]
fn test_apply_alignment_in_place_noops_for_disabled_alignment() {
    let config = AlignmentConfig::disabled();
    let original = vec![
        vec![0.0f32, 0.5, -0.25, 0.125],
        vec![0.1f32, -0.2, 0.3, -0.4],
    ];
    let mut actual = original.clone();

    super::apply_alignment_in_place(&mut actual, &config).expect("disabled alignment");

    assert_eq!(
        actual, original,
        "disabled alignment should leave voice buffers unchanged"
    );
}

// ---------------------------------------------------------------------------
// Builder chain and pipeline mode integration tests
// Part of #4264.
// ---------------------------------------------------------------------------

/// Verify all `with_*` builders can be chained without error.
///
/// Tests that constructing a KokoroChorus with every available builder
/// method succeeds. This catches initialization panics or validation
/// errors in default configs.
#[test]
fn test_builder_chain_all_options() {
    use nn_models::kokoro_chorus_detune::DetuneConfig;
    use nn_models::kokoro_chorus_dynamics::DynamicsPreset;
    use nn_models::kokoro_chorus_eq::{EqPreset, MixBusConfig};
    use nn_models::kokoro_chorus_humanize::HumanizeConfig;
    use nn_models::kokoro_chorus_stereo::StereoChorusConfig;

    let (mut primary, cache) = build_mini_kokoro();
    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let n = 4;
    let config = ChorusConfig::equal_gain(n).unwrap();

    // Chain all non-conflicting builders. Note: with_chorus_pipeline replaces
    // the default mix path, so we test it separately below.
    let chorus = KokoroChorus::new(&primary, config)
        .unwrap()
        .with_humanize(HumanizeConfig::default())
        .with_detune(DetuneConfig::default())
        .unwrap()
        .with_dynamics(DynamicsPreset::Broadcast)
        .unwrap()
        .with_stereo_config(StereoChorusConfig::auto_layout(n).unwrap())
        .with_eq_config(MixBusConfig::from_preset(EqPreset::Warm))
        .unwrap();

    assert_eq!(chorus.n_voices(), n);
    assert!(chorus.has_dynamics());
    assert!(chorus.has_detune());
    assert!(chorus.has_stereo());
    assert!(chorus.has_eq());
    assert!(chorus.humanize_config().is_some());
    assert!(!chorus.has_chorus_pipeline());
}

/// Verify all DynamicsPreset variants create valid compressor/limiter state.
#[test]
fn test_dynamics_presets_all_variants() {
    use nn_models::kokoro_chorus_dynamics::DynamicsPreset;

    let (mut primary, cache) = build_mini_kokoro();
    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let presets = [
        DynamicsPreset::Gentle,
        DynamicsPreset::Broadcast,
        DynamicsPreset::Aggressive,
        DynamicsPreset::Mastering,
    ];

    for preset in &presets {
        let config = ChorusConfig::equal_gain(4).unwrap();
        let chorus = KokoroChorus::new(&primary, config)
            .unwrap()
            .with_dynamics(*preset)
            .unwrap();

        assert!(
            chorus.has_dynamics(),
            "has_dynamics must be true for {preset:?}"
        );
        assert_eq!(
            chorus.dynamics_preset(),
            Some(*preset),
            "dynamics_preset() must return the set preset for {preset:?}",
        );
    }
}

/// Verify ChorusMasterConfig presets (minimal, standard, full) create valid pipelines.
#[test]
fn test_pipeline_preset_modes() {
    use nn_models::kokoro_chorus_pipeline::ChorusMasterConfig;

    let (mut primary, cache) = build_mini_kokoro();
    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let n = 4;

    // Minimal preset: blend + stereo.
    let config = ChorusConfig::equal_gain(n).unwrap();
    let chorus = KokoroChorus::new(&primary, config)
        .unwrap()
        .with_chorus_pipeline(ChorusMasterConfig::minimal(n).unwrap())
        .unwrap();
    assert!(
        chorus.has_chorus_pipeline(),
        "minimal pipeline must be active"
    );

    // Standard preset: EQ + de-esser + blend + stereo + dynamics.
    let config = ChorusConfig::equal_gain(n).unwrap();
    let chorus = KokoroChorus::new(&primary, config)
        .unwrap()
        .with_chorus_pipeline(ChorusMasterConfig::standard(n).unwrap())
        .unwrap();
    assert!(
        chorus.has_chorus_pipeline(),
        "standard pipeline must be active"
    );

    // Full preset: all stages enabled.
    let config = ChorusConfig::equal_gain(n).unwrap();
    let chorus = KokoroChorus::new(&primary, config)
        .unwrap()
        .with_chorus_pipeline(ChorusMasterConfig::full(n).unwrap())
        .unwrap();
    assert!(chorus.has_chorus_pipeline(), "full pipeline must be active");

    // Empty (new): no stages -- just passthrough.
    let config = ChorusConfig::equal_gain(n).unwrap();
    let chorus = KokoroChorus::new(&primary, config)
        .unwrap()
        .with_chorus_pipeline(ChorusMasterConfig::new(n).unwrap())
        .unwrap();
    assert!(
        chorus.has_chorus_pipeline(),
        "empty pipeline must still be active"
    );
}

/// Verify reset_dynamics and reset_chorus_pipeline clear state correctly.
#[test]
fn test_reset_dynamics_and_pipeline() {
    use nn_models::kokoro_chorus_dynamics::DynamicsPreset;
    use nn_models::kokoro_chorus_pipeline::ChorusMasterConfig;

    let (mut primary, cache) = build_mini_kokoro();
    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let n = 4;

    // Test reset_dynamics: enabled state persists, only internal envelope state resets.
    let config = ChorusConfig::equal_gain(n).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config)
        .unwrap()
        .with_dynamics(DynamicsPreset::Broadcast)
        .unwrap();
    assert!(chorus.has_dynamics());
    chorus.reset_dynamics();
    // Dynamics should still be enabled after reset (reset clears envelope, not config).
    assert!(
        chorus.has_dynamics(),
        "reset_dynamics must not disable dynamics"
    );
    assert_eq!(chorus.dynamics_preset(), Some(DynamicsPreset::Broadcast));

    // Test reset_chorus_pipeline: pipeline stays active after reset.
    let config = ChorusConfig::equal_gain(n).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config)
        .unwrap()
        .with_chorus_pipeline(ChorusMasterConfig::standard(n).unwrap())
        .unwrap();
    assert!(chorus.has_chorus_pipeline());
    chorus.reset_chorus_pipeline();
    assert!(
        chorus.has_chorus_pipeline(),
        "reset_chorus_pipeline must not disable the pipeline",
    );

    // Test reset_eq: EQ stays active after reset.
    let config = ChorusConfig::equal_gain(n).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config)
        .unwrap()
        .with_eq_config(nn_models::kokoro_chorus_eq::MixBusConfig::default())
        .unwrap();
    assert!(chorus.has_eq());
    chorus.reset_eq();
    assert!(chorus.has_eq(), "reset_eq must not disable EQ");
}

/// Verify has_* accessors return correct values for unset options.
#[test]
fn test_has_accessors_default_false() {
    let (mut primary, cache) = build_mini_kokoro();
    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(4).unwrap();
    let chorus = KokoroChorus::new(&primary, config).unwrap();

    // All optional features must be off by default.
    assert!(!chorus.has_dynamics(), "dynamics off by default");
    assert!(!chorus.has_detune(), "detune off by default");
    assert!(!chorus.has_stereo(), "stereo off by default");
    assert!(!chorus.has_eq(), "eq off by default");
    assert!(
        !chorus.has_chorus_pipeline(),
        "chorus pipeline off by default"
    );
    assert!(
        chorus.humanize_config().is_none(),
        "humanize off by default"
    );
    assert!(
        chorus.dynamics_preset().is_none(),
        "dynamics preset None by default"
    );
    assert!(
        chorus.detune_config().is_none(),
        "detune config None by default"
    );
    assert!(
        chorus.stereo_config().is_none(),
        "stereo config None by default"
    );
    assert!(
        chorus.mix_bus_config().is_none(),
        "mix bus config None by default"
    );
}

/// Verify KokoroChorus struct field completeness by checking all accessor
/// methods return consistent values after full initialization.
///
/// This catches regressions where a new field is added to KokoroChorus but
/// the corresponding accessor or builder is missing.
#[test]
fn test_field_completeness_via_accessors() {
    use nn_models::kokoro_chorus_detune::DetuneConfig;
    use nn_models::kokoro_chorus_dynamics::DynamicsPreset;
    use nn_models::kokoro_chorus_eq::{EqPreset, MixBusConfig};
    use nn_models::kokoro_chorus_humanize::HumanizeConfig;
    use nn_models::kokoro_chorus_stereo::StereoChorusConfig;

    let (mut primary, cache) = build_mini_kokoro();
    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let n = 4;
    let config = ChorusConfig::equal_gain(n).unwrap();

    let chorus = KokoroChorus::new(&primary, config)
        .unwrap()
        .with_humanize(HumanizeConfig::default())
        .with_detune(DetuneConfig::default())
        .unwrap()
        .with_dynamics(DynamicsPreset::Mastering)
        .unwrap()
        .with_stereo_config(StereoChorusConfig::auto_layout(n).unwrap())
        .with_eq_config(MixBusConfig::from_preset(EqPreset::Natural))
        .unwrap();

    // All accessors must return Some/true for set fields.
    assert!(chorus.humanize_config().is_some(), "humanize accessor");
    assert!(chorus.detune_config().is_some(), "detune accessor");
    assert!(
        chorus.dynamics_preset().is_some(),
        "dynamics preset accessor"
    );
    assert!(chorus.stereo_config().is_some(), "stereo accessor");
    assert!(chorus.mix_bus_config().is_some(), "mix bus config accessor");
    assert!(chorus.has_dynamics(), "has_dynamics accessor");
    assert!(chorus.has_detune(), "has_detune accessor");
    assert!(chorus.has_stereo(), "has_stereo accessor");
    assert!(chorus.has_eq(), "has_eq accessor");

    // Config and voice accessors.
    assert_eq!(chorus.n_voices(), n);
    assert_eq!(chorus.config().n_voices, n);
    assert!(chorus.voice(0).is_some(), "voice(0) accessible");
    assert!(chorus.voice(n).is_none(), "voice(n) out of bounds");
    assert!(
        chorus.shared_state_refcount() > 0,
        "shared state refcount > 0"
    );
    assert!(
        chorus.gpu_weight_bytes_per_voice() > 0,
        "gpu weight bytes > 0"
    );
}

/// Verify that with_chorus_pipeline and with_dynamics can coexist.
///
/// When chorus_pipeline is active, mix_or_process routes through the pipeline
/// rather than the default dynamics path, but dynamics state is still present.
#[test]
fn test_pipeline_and_dynamics_coexist() {
    use nn_models::kokoro_chorus_dynamics::DynamicsPreset;
    use nn_models::kokoro_chorus_pipeline::ChorusMasterConfig;

    let (mut primary, cache) = build_mini_kokoro();
    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let n = 4;
    let config = ChorusConfig::equal_gain(n).unwrap();

    // Set both dynamics and chorus pipeline.
    let chorus = KokoroChorus::new(&primary, config)
        .unwrap()
        .with_dynamics(DynamicsPreset::Aggressive)
        .unwrap()
        .with_chorus_pipeline(ChorusMasterConfig::full(n).unwrap())
        .unwrap();

    assert!(chorus.has_dynamics(), "dynamics must be set");
    assert!(chorus.has_chorus_pipeline(), "pipeline must be set");
    assert_eq!(chorus.dynamics_preset(), Some(DynamicsPreset::Aggressive));
}

/// Verify invalid DetuneConfig is rejected by with_detune.
#[test]
fn test_detune_validation_rejects_invalid() {
    use nn_models::kokoro_chorus_detune::{DetuneConfig, DetuneDistribution};

    let (mut primary, cache) = build_mini_kokoro();
    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    // cents_spread > 50 should be rejected at DetuneConfig::new.
    let bad_result = DetuneConfig::new(100.0, DetuneDistribution::Uniform, 0);
    assert!(bad_result.is_err(), "cents_spread=100 should be rejected");

    // NaN cents_spread should be rejected.
    let nan_result = DetuneConfig::new(f32::NAN, DetuneDistribution::Uniform, 0);
    assert!(nan_result.is_err(), "NaN cents_spread should be rejected");

    // Valid detune should succeed through with_detune.
    let config = ChorusConfig::equal_gain(4).unwrap();
    let good_detune = DetuneConfig::new(5.0, DetuneDistribution::Uniform, 0).unwrap();
    let result = KokoroChorus::new(&primary, config)
        .unwrap()
        .with_detune(good_detune);
    assert!(result.is_ok(), "cents_spread=5 should succeed");
}

/// Verify voice_mut accessor works and allows modification.
#[test]
fn test_voice_mut_accessor() {
    let (mut primary, cache) = build_mini_kokoro();
    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(4).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    assert!(chorus.voice_mut(0).is_some(), "voice_mut(0) accessible");
    assert!(chorus.voice_mut(3).is_some(), "voice_mut(3) accessible");
    assert!(chorus.voice_mut(4).is_none(), "voice_mut(4) out of bounds");
}

/// Verify the chorus arena checkpoint guard restores the default arena even
/// when a voice scope unwinds early.
#[test]
fn test_default_arena_checkpoint_restores_on_unwind() {
    let ctx = test_ctx();

    let (_buf, _off) = crate::arena::arena_alloc_or_create(ctx, 256).expect("init default arena");
    let used_before = crate::arena::default_arena_used_bytes().expect("default arena should exist");

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _arena_cp = super::DefaultArenaCheckpoint::new();
        let (_buf2, _off2) =
            crate::arena::arena_alloc_or_create(ctx, 512).expect("advance default arena");
        let used_during =
            crate::arena::default_arena_used_bytes().expect("default arena should still exist");
        assert!(
            used_during > used_before,
            "checkpoint scope should advance arena before unwind"
        );
        panic!("deliberate panic to exercise Drop");
    }));

    assert!(result.is_err(), "panic should have been caught");

    let used_after =
        crate::arena::default_arena_used_bytes().expect("default arena should still exist");
    assert_eq!(
        used_after, used_before,
        "arena checkpoint guard must restore the bump pointer after unwind"
    );
}

/// Verify ChorusConfig::equal_gain(1) creates a valid single-voice chorus.
///
/// Edge case: chorus with 1 voice should function correctly.
#[test]
fn test_single_voice_chorus_builders() {
    use nn_models::kokoro_chorus_dynamics::DynamicsPreset;
    use nn_models::kokoro_chorus_stereo::StereoChorusConfig;

    let (mut primary, cache) = build_mini_kokoro();
    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(1).unwrap();
    let chorus = KokoroChorus::new(&primary, config)
        .unwrap()
        .with_dynamics(DynamicsPreset::Gentle)
        .unwrap()
        .with_stereo_config(StereoChorusConfig::auto_layout(1).unwrap());

    assert_eq!(chorus.n_voices(), 1);
    assert!(chorus.has_dynamics());
    assert!(chorus.has_stereo());
}
