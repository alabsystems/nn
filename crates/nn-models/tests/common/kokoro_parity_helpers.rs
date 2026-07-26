// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for Kokoro TTS parity tests.
//! Extracted from kokoro_parity.rs for 500-line limit.

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;
use nn_models::kokoro_tts::KokoroConfig;
use nn_models::plbert::PlbertConfig;
use std::collections::HashMap;

// -- Helper: build a VarBuilder with explicit tensors for TextEncoder ----------

/// Default test vocab size for TextEncoder (matches plbert.vocab_size in test configs).
pub(super) const TEST_VOCAB_SIZE: usize = 10;

pub(super) fn text_encoder_tensors(d_en: usize) -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    insert_text_encoder_weights_standalone(&mut m, d_en);
    m
}

/// Insert all TextEncoder weights including embedding + conv + norm + LSTM + projection.
fn insert_text_encoder_weights_standalone(m: &mut HashMap<String, DynTensor>, d_en: usize) {
    let hidden = d_en / 2;
    let four_h = 4 * hidden;
    let p = "text_encoder";
    // Embedding(vocab_size, d_en)
    zw(
        m,
        &format!("{p}.embedding.weight"),
        &[TEST_VOCAB_SIZE, d_en],
    );
    // 3× Conv1d(d_en, d_en, k=5) + LayerNorm(d_en)
    for i in 0..3 {
        zw(m, &format!("{p}.convs.{i}.weight"), &[d_en, d_en, 5]);
        zw(m, &format!("{p}.convs.{i}.bias"), &[d_en]);
        m.insert(
            format!("{p}.norms.{i}.weight"),
            DynTensor::full(&[d_en], 1.0, DType::F32, &cpu()).unwrap(),
        );
        zw(m, &format!("{p}.norms.{i}.bias"), &[d_en]);
    }
    // BiLSTM
    zw(m, &format!("{p}.lstm.weight_ih_l0"), &[four_h, d_en]);
    zw(m, &format!("{p}.lstm.weight_hh_l0"), &[four_h, hidden]);
    zw(m, &format!("{p}.lstm.bias_ih_l0"), &[four_h]);
    zw(m, &format!("{p}.lstm.bias_hh_l0"), &[four_h]);
    zw(
        m,
        &format!("{p}.lstm.weight_ih_l0_reverse"),
        &[four_h, d_en],
    );
    zw(
        m,
        &format!("{p}.lstm.weight_hh_l0_reverse"),
        &[four_h, hidden],
    );
    zw(m, &format!("{p}.lstm.bias_ih_l0_reverse"), &[four_h]);
    zw(m, &format!("{p}.lstm.bias_hh_l0_reverse"), &[four_h]);
    zw(m, &format!("{p}.lstm.linear.weight"), &[d_en, d_en]);
    zw(m, &format!("{p}.lstm.linear.bias"), &[d_en]);
}

// -- Helper: build a KokoroConfig matching test weights -----------------------

pub(super) fn parity_plbert_config(plbert_hidden: usize) -> PlbertConfig {
    let mut c = PlbertConfig::default();
    c.vocab_size = 10;
    c.embedding_dim = 4;
    c.hidden_size = plbert_hidden;
    c.num_attention_heads = 2;
    c.intermediate_size = 16;
    c.max_position_embeddings = 16;
    c.num_hidden_layers = 1;
    c.layer_norm_eps = 1e-12;
    c
}

#[allow(clippy::field_reassign_with_default)] // KokoroConfig is #[non_exhaustive]
pub(super) fn parity_kokoro_config(d_en: usize, n_prosody_layers: usize) -> KokoroConfig {
    let mut c = KokoroConfig::default();
    c.d_en = d_en;
    c.n_prosody_layers = n_prosody_layers;
    c.style_dim = 4;
    c.upsample_rates = vec![2];
    c.upsample_kernel_sizes = vec![4];
    c.resblock_kernel_sizes = vec![3];
    c.resblock_dilations = vec![vec![1, 2]];
    c.gen_initial_channels = d_en;
    c.n_fft = 4;
    c.f0_bilstm_hidden = d_en / 2;
    c.plbert = parity_plbert_config(d_en);
    c
}

// -- Helper: build a VarBuilder with tensors for full KokoroModel -------------

fn zw(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    m.insert(
        name.to_string(),
        DynTensor::zeros(shape, DType::F32, &cpu()).unwrap(),
    );
}

fn ow(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    m.insert(
        name.to_string(),
        DynTensor::full(shape, 1.0, DType::F32, &cpu()).unwrap(),
    );
}

fn insert_plbert_weights(m: &mut HashMap<String, DynTensor>, config: &PlbertConfig) {
    let p = "plbert";
    let h = config.hidden_size;
    let e = config.embedding_dim;
    zw(
        m,
        &format!("{p}.embeddings.word_embeddings.weight"),
        &[config.vocab_size, e],
    );
    zw(
        m,
        &format!("{p}.embeddings.position_embeddings.weight"),
        &[config.max_position_embeddings, e],
    );
    zw(
        m,
        &format!("{p}.embeddings.token_type_embeddings.weight"),
        &[2, e],
    );
    ow(m, &format!("{p}.embeddings.LayerNorm.weight"), &[e]);
    zw(m, &format!("{p}.embeddings.LayerNorm.bias"), &[e]);
    zw(
        m,
        &format!("{p}.encoder.embedding_hidden_mapping_in.weight"),
        &[h, e],
    );
    zw(
        m,
        &format!("{p}.encoder.embedding_hidden_mapping_in.bias"),
        &[h],
    );
    let lp = format!("{p}.encoder.albert_layer_groups.0.albert_layers.0");
    for name in &[
        "attention.query",
        "attention.key",
        "attention.value",
        "attention.dense",
    ] {
        zw(m, &format!("{lp}.{name}.weight"), &[h, h]);
        zw(m, &format!("{lp}.{name}.bias"), &[h]);
    }
    ow(m, &format!("{lp}.attention.LayerNorm.weight"), &[h]);
    zw(m, &format!("{lp}.attention.LayerNorm.bias"), &[h]);
    zw(
        m,
        &format!("{lp}.ffn.weight"),
        &[config.intermediate_size, h],
    );
    zw(m, &format!("{lp}.ffn.bias"), &[config.intermediate_size]);
    zw(
        m,
        &format!("{lp}.ffn_output.weight"),
        &[h, config.intermediate_size],
    );
    zw(m, &format!("{lp}.ffn_output.bias"), &[h]);
    ow(m, &format!("{lp}.full_layer_layer_norm.weight"), &[h]);
    zw(m, &format!("{lp}.full_layer_layer_norm.bias"), &[h]);
}

fn insert_text_encoder_weights(m: &mut HashMap<String, DynTensor>, d_en: usize) {
    let hidden = d_en / 2;
    let four_h = 4 * hidden;
    let p = "text_encoder";
    let vocab_size = TEST_VOCAB_SIZE;
    // Embedding(vocab_size, d_en)
    zw(m, &format!("{p}.embedding.weight"), &[vocab_size, d_en]);
    // 3× Conv1d(d_en, d_en, k=5) + LayerNorm(d_en)
    for i in 0..3 {
        zw(m, &format!("{p}.convs.{i}.weight"), &[d_en, d_en, 5]);
        zw(m, &format!("{p}.convs.{i}.bias"), &[d_en]);
        ow(m, &format!("{p}.norms.{i}.weight"), &[d_en]);
        zw(m, &format!("{p}.norms.{i}.bias"), &[d_en]);
    }
    // BiLSTM
    zw(m, &format!("{p}.lstm.weight_ih_l0"), &[four_h, d_en]);
    zw(m, &format!("{p}.lstm.weight_hh_l0"), &[four_h, hidden]);
    zw(m, &format!("{p}.lstm.bias_ih_l0"), &[four_h]);
    zw(m, &format!("{p}.lstm.bias_hh_l0"), &[four_h]);
    zw(
        m,
        &format!("{p}.lstm.weight_ih_l0_reverse"),
        &[four_h, d_en],
    );
    zw(
        m,
        &format!("{p}.lstm.weight_hh_l0_reverse"),
        &[four_h, hidden],
    );
    zw(m, &format!("{p}.lstm.bias_ih_l0_reverse"), &[four_h]);
    zw(m, &format!("{p}.lstm.bias_hh_l0_reverse"), &[four_h]);
    zw(m, &format!("{p}.lstm.linear.weight"), &[d_en, d_en]);
    zw(m, &format!("{p}.lstm.linear.bias"), &[d_en]);
}

fn insert_prosody_weights(m: &mut HashMap<String, DynTensor>, config: &KokoroConfig) {
    let d_en = config.d_en;
    let style_dim = config.style_dim;
    let hidden = d_en / 2;
    let four_h = 4 * hidden;
    let lstm_input = d_en + style_dim;
    // DurationEncoder: BiLSTM + AdaLayerNorm blocks.
    for i in 0..config.n_prosody_layers {
        // BiLSTM weights under duration.lstms.{i}.
        let lp = format!("prosody_predictor.duration.lstms.{i}");
        zw(m, &format!("{lp}.weight_ih_l0"), &[four_h, lstm_input]);
        zw(m, &format!("{lp}.weight_hh_l0"), &[four_h, hidden]);
        zw(m, &format!("{lp}.bias_ih_l0"), &[four_h]);
        zw(m, &format!("{lp}.bias_hh_l0"), &[four_h]);
        zw(
            m,
            &format!("{lp}.weight_ih_l0_reverse"),
            &[four_h, lstm_input],
        );
        zw(m, &format!("{lp}.weight_hh_l0_reverse"), &[four_h, hidden]);
        zw(m, &format!("{lp}.bias_ih_l0_reverse"), &[four_h]);
        zw(m, &format!("{lp}.bias_hh_l0_reverse"), &[four_h]);
        // AdaLayerNorm under duration.norms.{i}.
        let np = format!("prosody_predictor.duration.norms.{i}");
        ow(m, &format!("{np}.norm.weight"), &[d_en]);
        zw(m, &format!("{np}.norm.bias"), &[d_en]);
        zw(m, &format!("{np}.fc.weight"), &[2 * d_en, style_dim]);
        zw(m, &format!("{np}.fc.bias"), &[2 * d_en]);
    }
    // Duration projection under duration.
    zw(
        m,
        "prosody_predictor.duration.duration_proj.weight",
        &[config.max_dur, d_en],
    );
    zw(
        m,
        "prosody_predictor.duration.duration_proj.bias",
        &[config.max_dur],
    );
    // Final duration BiLSTM under lstm.
    let dl = "prosody_predictor.lstm";
    zw(m, &format!("{dl}.weight_ih_l0"), &[four_h, lstm_input]);
    zw(m, &format!("{dl}.weight_hh_l0"), &[four_h, hidden]);
    zw(m, &format!("{dl}.bias_ih_l0"), &[four_h]);
    zw(m, &format!("{dl}.bias_hh_l0"), &[four_h]);
    zw(
        m,
        &format!("{dl}.weight_ih_l0_reverse"),
        &[four_h, lstm_input],
    );
    zw(m, &format!("{dl}.weight_hh_l0_reverse"), &[four_h, hidden]);
    zw(m, &format!("{dl}.bias_ih_l0_reverse"), &[four_h]);
    zw(m, &format!("{dl}.bias_hh_l0_reverse"), &[four_h]);
}

fn insert_resblock_weights(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    channels: usize,
    style_dim: usize,
    kernel_size: usize,
    num_dilations: usize,
) {
    for i in 0..num_dilations {
        zw(
            m,
            &format!("{prefix}.convs1.{i}.weight"),
            &[channels, channels, kernel_size],
        );
        zw(m, &format!("{prefix}.convs1.{i}.bias"), &[channels]);
        zw(
            m,
            &format!("{prefix}.convs2.{i}.weight"),
            &[channels, channels, kernel_size],
        );
        zw(m, &format!("{prefix}.convs2.{i}.bias"), &[channels]);
        zw(
            m,
            &format!("{prefix}.adain1.{i}.fc.weight"),
            &[2 * channels, style_dim],
        );
        zw(m, &format!("{prefix}.adain1.{i}.fc.bias"), &[2 * channels]);
        zw(
            m,
            &format!("{prefix}.adain2.{i}.fc.weight"),
            &[2 * channels, style_dim],
        );
        zw(m, &format!("{prefix}.adain2.{i}.fc.bias"), &[2 * channels]);
        m.insert(
            format!("{prefix}.alpha1.{i}"),
            DynTensor::full(&[1, channels, 1], 1.0, DType::F32, &cpu()).unwrap(),
        );
        m.insert(
            format!("{prefix}.alpha2.{i}"),
            DynTensor::full(&[1, channels, 1], 1.0, DType::F32, &cpu()).unwrap(),
        );
    }
}

fn insert_stage1_resblk_weights(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    dim_in: usize,
    dim_out: usize,
    style_dim: usize,
    upsample: bool,
) {
    zw(m, &format!("{prefix}.conv1.weight"), &[dim_out, dim_in, 3]);
    zw(m, &format!("{prefix}.conv1.bias"), &[dim_out]);
    zw(m, &format!("{prefix}.conv2.weight"), &[dim_out, dim_out, 3]);
    zw(m, &format!("{prefix}.conv2.bias"), &[dim_out]);
    zw(
        m,
        &format!("{prefix}.norm1.style_linear.weight"),
        &[2 * dim_in, style_dim],
    );
    zw(
        m,
        &format!("{prefix}.norm1.style_linear.bias"),
        &[2 * dim_in],
    );
    zw(
        m,
        &format!("{prefix}.norm2.style_linear.weight"),
        &[2 * dim_out, style_dim],
    );
    zw(
        m,
        &format!("{prefix}.norm2.style_linear.bias"),
        &[2 * dim_out],
    );
    if dim_in != dim_out {
        zw(
            m,
            &format!("{prefix}.conv1x1.weight"),
            &[dim_out, dim_in, 1],
        );
        zw(m, &format!("{prefix}.conv1x1.bias"), &[dim_out]);
    }
    if upsample {
        zw(m, &format!("{prefix}.pool.weight"), &[dim_in, 1, 3]);
        zw(m, &format!("{prefix}.pool.bias"), &[dim_in]);
    }
}

fn insert_decoder_weights(m: &mut HashMap<String, DynTensor>, config: &KokoroConfig) {
    let dp = "decoder";
    let d_en = config.d_en;
    let style_dim = config.style_dim;
    let asr_res_ch = (d_en / 8).max(1);
    let hidden = 2 * d_en;
    let encode_in = d_en + 2;
    let decode_in = hidden + asr_res_ch + 2;
    // FullDecoder: F0/N downsampling, compressed skip, encode/decode blocks.
    zw(m, &format!("{dp}.F0_conv.weight"), &[1, 1, 3]);
    zw(m, &format!("{dp}.F0_conv.bias"), &[1]);
    zw(m, &format!("{dp}.N_conv.weight"), &[1, 1, 3]);
    zw(m, &format!("{dp}.N_conv.bias"), &[1]);
    zw(m, &format!("{dp}.asr_res.weight"), &[asr_res_ch, d_en, 1]);
    zw(m, &format!("{dp}.asr_res.bias"), &[asr_res_ch]);
    insert_stage1_resblk_weights(
        m,
        &format!("{dp}.encode"),
        encode_in,
        hidden,
        style_dim,
        false,
    );
    for i in 0..3 {
        insert_stage1_resblk_weights(
            m,
            &format!("{dp}.decode.{i}"),
            decode_in,
            hidden,
            style_dim,
            false,
        );
    }
    insert_stage1_resblk_weights(
        m,
        &format!("{dp}.decode.3"),
        decode_in,
        d_en,
        style_dim,
        true,
    );
    // Generator weights under decoder.generator.*
    let gp = format!("{dp}.generator");
    let gen_ch = config.gen_initial_channels;
    let next_ch = gen_ch / 2;
    let n_bins = config.n_fft / 2 + 1;
    zw(m, &format!("{gp}.conv_pre.weight"), &[gen_ch, gen_ch, 7]);
    zw(m, &format!("{gp}.conv_pre.bias"), &[gen_ch]);
    zw(m, &format!("{gp}.ups.0.weight"), &[gen_ch, next_ch, 4]);
    zw(m, &format!("{gp}.ups.0.bias"), &[next_ch]);
    zw(
        m,
        &format!("{gp}.noise_convs.0.weight"),
        &[next_ch, 2 * n_bins, 1],
    );
    zw(m, &format!("{gp}.noise_convs.0.bias"), &[next_ch]);
    // PyTorch reference: noise_res uses kernel=11 (last stage), dilations=[1,3,5].
    insert_resblock_weights(
        m,
        &format!("{gp}.noise_res.0"),
        next_ch,
        config.style_dim,
        11,
        3,
    );
    insert_resblock_weights(
        m,
        &format!("{gp}.resblocks.0"),
        next_ch,
        config.style_dim,
        3,
        2,
    );
    zw(
        m,
        &format!("{gp}.conv_post.weight"),
        &[2 * n_bins, next_ch, 7],
    );
    zw(m, &format!("{gp}.conv_post.bias"), &[2 * n_bins]);
}

fn insert_adain_resblk_weights(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    dim_in: usize,
    dim_out: usize,
    style_dim: usize,
    upsample: bool,
) {
    zw(
        m,
        &format!("{prefix}.n1.fc.weight"),
        &[2 * dim_in, style_dim],
    );
    zw(m, &format!("{prefix}.n1.fc.bias"), &[2 * dim_in]);
    zw(
        m,
        &format!("{prefix}.n2.fc.weight"),
        &[2 * dim_out, style_dim],
    );
    zw(m, &format!("{prefix}.n2.fc.bias"), &[2 * dim_out]);
    zw(m, &format!("{prefix}.c1.weight"), &[dim_out, dim_in, 3]);
    zw(m, &format!("{prefix}.c1.bias"), &[dim_out]);
    zw(m, &format!("{prefix}.c2.weight"), &[dim_out, dim_out, 3]);
    zw(m, &format!("{prefix}.c2.bias"), &[dim_out]);
    if dim_in != dim_out {
        zw(m, &format!("{prefix}.skip.weight"), &[dim_out, dim_in, 1]);
        zw(m, &format!("{prefix}.skip.bias"), &[dim_out]);
    }
    if upsample {
        zw(m, &format!("{prefix}.pool.weight"), &[dim_in, 1, 3]);
        zw(m, &format!("{prefix}.pool.bias"), &[dim_in]);
    }
}

fn insert_f0_predictor_weights(m: &mut HashMap<String, DynTensor>, config: &KokoroConfig) {
    let d_en = config.d_en;
    let style_dim = config.style_dim;
    let h = config.f0_bilstm_hidden;
    let bilstm_out = 2 * h;
    let four_h = 4 * h;
    let p = "predictor";
    // Shared BiLSTM: input dim = d_en + style_dim (cat of dur_features + style)
    let bilstm_input = d_en + style_dim;
    // Decomposed BiLSTM format matching real Kokoro weights (Part of #2691).
    zw(
        m,
        &format!("{p}.shared.forward.weight_ih.weight"),
        &[four_h, bilstm_input],
    );
    zw(
        m,
        &format!("{p}.shared.forward.weight_hh.weight"),
        &[four_h, h],
    );
    zw(m, &format!("{p}.shared.forward.weight_ih.bias"), &[four_h]);
    zw(m, &format!("{p}.shared.forward.weight_hh.bias"), &[four_h]);
    zw(
        m,
        &format!("{p}.shared.backward.weight_ih.weight"),
        &[four_h, bilstm_input],
    );
    zw(
        m,
        &format!("{p}.shared.backward.weight_hh.weight"),
        &[four_h, h],
    );
    zw(m, &format!("{p}.shared.backward.weight_ih.bias"), &[four_h]);
    zw(m, &format!("{p}.shared.backward.weight_hh.bias"), &[four_h]);
    // F0 blocks: 0: bilstm_out→bilstm_out, 1: bilstm_out→h (upsample), 2: h→h
    insert_adain_resblk_weights(
        m,
        &format!("{p}.F0.0"),
        bilstm_out,
        bilstm_out,
        style_dim,
        false,
    );
    insert_adain_resblk_weights(m, &format!("{p}.F0.1"), bilstm_out, h, style_dim, true);
    insert_adain_resblk_weights(m, &format!("{p}.F0.2"), h, h, style_dim, false);
    zw(m, &format!("{p}.F0_proj.weight"), &[1, h]);
    zw(m, &format!("{p}.F0_proj.bias"), &[1]);
    // Energy (N) blocks: same architecture
    insert_adain_resblk_weights(
        m,
        &format!("{p}.N.0"),
        bilstm_out,
        bilstm_out,
        style_dim,
        false,
    );
    insert_adain_resblk_weights(m, &format!("{p}.N.1"), bilstm_out, h, style_dim, true);
    insert_adain_resblk_weights(m, &format!("{p}.N.2"), h, h, style_dim, false);
    zw(m, &format!("{p}.N_proj.weight"), &[1, h]);
    zw(m, &format!("{p}.N_proj.bias"), &[1]);
}

pub(super) fn kokoro_model_tensors(config: &KokoroConfig) -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    insert_plbert_weights(&mut m, &config.plbert);
    zw(
        &mut m,
        "bert_encoder.weight",
        &[config.d_en, config.plbert.hidden_size],
    );
    zw(&mut m, "bert_encoder.bias", &[config.d_en]);
    insert_text_encoder_weights(&mut m, config.d_en);
    insert_prosody_weights(&mut m, config);
    insert_f0_predictor_weights(&mut m, config);
    insert_decoder_weights(&mut m, config);
    m
}
