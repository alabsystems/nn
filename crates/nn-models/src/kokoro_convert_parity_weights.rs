// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight builder helpers for Kokoro auto-converter parity tests.
//! Extracted for 500-line compliance. Mirrors kokoro_tts_tests_model_weights.rs
//! but is self-contained so the parity test module does not depend on the
//! kokoro_tts test module's internal types.

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;
use std::collections::HashMap;

use super::{T_D_EN, T_EMB, T_F0_HIDDEN, T_GEN_CH, T_HIDDEN, T_N_FFT, T_STYLE, T_VOCAB};

fn z(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    m.insert(
        name.to_string(),
        DynTensor::zeros(shape, DType::F32, &cpu()).unwrap(),
    );
}

fn ones(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    m.insert(
        name.to_string(),
        DynTensor::full(shape, 1.0, DType::F32, &cpu()).unwrap(),
    );
}

pub(super) fn make_kokoro_weights() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    insert_plbert(&mut m);
    z(&mut m, "bert_encoder.weight", &[T_D_EN, T_HIDDEN]);
    z(&mut m, "bert_encoder.bias", &[T_D_EN]);
    insert_text_encoder(&mut m);
    insert_prosody(&mut m);
    insert_f0_predictor(&mut m);
    insert_decoder(&mut m);
    insert_source_module(&mut m);
    m
}

fn insert_plbert(m: &mut HashMap<String, DynTensor>) {
    let p = "plbert";
    z(
        m,
        &format!("{p}.embeddings.word_embeddings.weight"),
        &[T_VOCAB, T_EMB],
    );
    z(
        m,
        &format!("{p}.embeddings.position_embeddings.weight"),
        &[16, T_EMB],
    );
    z(
        m,
        &format!("{p}.embeddings.token_type_embeddings.weight"),
        &[2, T_EMB],
    );
    ones(m, &format!("{p}.embeddings.LayerNorm.weight"), &[T_EMB]);
    z(m, &format!("{p}.embeddings.LayerNorm.bias"), &[T_EMB]);
    z(
        m,
        &format!("{p}.encoder.embedding_hidden_mapping_in.weight"),
        &[T_HIDDEN, T_EMB],
    );
    z(
        m,
        &format!("{p}.encoder.embedding_hidden_mapping_in.bias"),
        &[T_HIDDEN],
    );
    let lp = format!("{p}.encoder.albert_layer_groups.0.albert_layers.0");
    for name in &[
        "attention.query",
        "attention.key",
        "attention.value",
        "attention.dense",
    ] {
        z(m, &format!("{lp}.{name}.weight"), &[T_HIDDEN, T_HIDDEN]);
        z(m, &format!("{lp}.{name}.bias"), &[T_HIDDEN]);
    }
    ones(m, &format!("{lp}.attention.LayerNorm.weight"), &[T_HIDDEN]);
    z(m, &format!("{lp}.attention.LayerNorm.bias"), &[T_HIDDEN]);
    z(m, &format!("{lp}.ffn.weight"), &[16, T_HIDDEN]);
    z(m, &format!("{lp}.ffn.bias"), &[16]);
    z(m, &format!("{lp}.ffn_output.weight"), &[T_HIDDEN, 16]);
    z(m, &format!("{lp}.ffn_output.bias"), &[T_HIDDEN]);
    ones(
        m,
        &format!("{lp}.full_layer_layer_norm.weight"),
        &[T_HIDDEN],
    );
    z(m, &format!("{lp}.full_layer_layer_norm.bias"), &[T_HIDDEN]);
}

fn insert_text_encoder(m: &mut HashMap<String, DynTensor>) {
    let hidden = T_D_EN / 2;
    let p = "text_encoder";
    z(m, &format!("{p}.embedding.weight"), &[T_VOCAB, T_D_EN]);
    for i in 0..3 {
        z(m, &format!("{p}.convs.{i}.weight"), &[T_D_EN, T_D_EN, 5]);
        z(m, &format!("{p}.convs.{i}.bias"), &[T_D_EN]);
        ones(m, &format!("{p}.norms.{i}.weight"), &[T_D_EN]);
        z(m, &format!("{p}.norms.{i}.bias"), &[T_D_EN]);
    }
    z(m, &format!("{p}.lstm.weight_ih_l0"), &[4 * hidden, T_D_EN]);
    z(m, &format!("{p}.lstm.weight_hh_l0"), &[4 * hidden, hidden]);
    z(m, &format!("{p}.lstm.bias_ih_l0"), &[4 * hidden]);
    z(m, &format!("{p}.lstm.bias_hh_l0"), &[4 * hidden]);
    z(
        m,
        &format!("{p}.lstm.weight_ih_l0_reverse"),
        &[4 * hidden, T_D_EN],
    );
    z(
        m,
        &format!("{p}.lstm.weight_hh_l0_reverse"),
        &[4 * hidden, hidden],
    );
    z(m, &format!("{p}.lstm.bias_ih_l0_reverse"), &[4 * hidden]);
    z(m, &format!("{p}.lstm.bias_hh_l0_reverse"), &[4 * hidden]);
    z(m, &format!("{p}.lstm.linear.weight"), &[T_D_EN, T_D_EN]);
    z(m, &format!("{p}.lstm.linear.bias"), &[T_D_EN]);
}

fn insert_prosody(m: &mut HashMap<String, DynTensor>) {
    let p = "prosody_predictor";
    let hidden = T_D_EN / 2;
    let four_h = 4 * hidden;
    let lstm_input = T_D_EN + T_STYLE;
    let l = format!("{p}.duration.lstms.0");
    z(m, &format!("{l}.weight_ih_l0"), &[four_h, lstm_input]);
    z(m, &format!("{l}.weight_hh_l0"), &[four_h, hidden]);
    z(m, &format!("{l}.bias_ih_l0"), &[four_h]);
    z(m, &format!("{l}.bias_hh_l0"), &[four_h]);
    z(
        m,
        &format!("{l}.weight_ih_l0_reverse"),
        &[four_h, lstm_input],
    );
    z(m, &format!("{l}.weight_hh_l0_reverse"), &[four_h, hidden]);
    z(m, &format!("{l}.bias_ih_l0_reverse"), &[four_h]);
    z(m, &format!("{l}.bias_hh_l0_reverse"), &[four_h]);
    let n = format!("{p}.duration.norms.0");
    ones(m, &format!("{n}.norm.weight"), &[T_D_EN]);
    z(m, &format!("{n}.norm.bias"), &[T_D_EN]);
    z(m, &format!("{n}.fc.weight"), &[2 * T_D_EN, T_STYLE]);
    z(m, &format!("{n}.fc.bias"), &[2 * T_D_EN]);
    z(
        m,
        &format!("{p}.duration.duration_proj.weight"),
        &[50, T_D_EN],
    );
    z(m, &format!("{p}.duration.duration_proj.bias"), &[50]);
    let dl = format!("{p}.lstm");
    z(m, &format!("{dl}.weight_ih_l0"), &[four_h, lstm_input]);
    z(m, &format!("{dl}.weight_hh_l0"), &[four_h, hidden]);
    z(m, &format!("{dl}.bias_ih_l0"), &[four_h]);
    z(m, &format!("{dl}.bias_hh_l0"), &[four_h]);
    z(
        m,
        &format!("{dl}.weight_ih_l0_reverse"),
        &[four_h, lstm_input],
    );
    z(m, &format!("{dl}.weight_hh_l0_reverse"), &[four_h, hidden]);
    z(m, &format!("{dl}.bias_ih_l0_reverse"), &[four_h]);
    z(m, &format!("{dl}.bias_hh_l0_reverse"), &[four_h]);
}

fn insert_resblock(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    ch: usize,
    kernel: usize,
    ndil: usize,
) {
    for i in 0..ndil {
        z(m, &format!("{prefix}.convs1.{i}.weight"), &[ch, ch, kernel]);
        z(m, &format!("{prefix}.convs1.{i}.bias"), &[ch]);
        z(m, &format!("{prefix}.convs2.{i}.weight"), &[ch, ch, kernel]);
        z(m, &format!("{prefix}.convs2.{i}.bias"), &[ch]);
        z(
            m,
            &format!("{prefix}.adain1.{i}.fc.weight"),
            &[2 * ch, T_STYLE],
        );
        z(m, &format!("{prefix}.adain1.{i}.fc.bias"), &[2 * ch]);
        z(
            m,
            &format!("{prefix}.adain2.{i}.fc.weight"),
            &[2 * ch, T_STYLE],
        );
        z(m, &format!("{prefix}.adain2.{i}.fc.bias"), &[2 * ch]);
        m.insert(
            format!("{prefix}.alpha1.{i}"),
            DynTensor::full(&[1, ch, 1], 1.0, DType::F32, &cpu()).unwrap(),
        );
        m.insert(
            format!("{prefix}.alpha2.{i}"),
            DynTensor::full(&[1, ch, 1], 1.0, DType::F32, &cpu()).unwrap(),
        );
    }
}

fn insert_stage1_resblk(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    din: usize,
    dout: usize,
    sdim: usize,
    up: bool,
) {
    z(m, &format!("{prefix}.conv1.weight"), &[dout, din, 3]);
    z(m, &format!("{prefix}.conv1.bias"), &[dout]);
    z(m, &format!("{prefix}.conv2.weight"), &[dout, dout, 3]);
    z(m, &format!("{prefix}.conv2.bias"), &[dout]);
    z(
        m,
        &format!("{prefix}.norm1.style_linear.weight"),
        &[2 * din, sdim],
    );
    z(m, &format!("{prefix}.norm1.style_linear.bias"), &[2 * din]);
    z(
        m,
        &format!("{prefix}.norm2.style_linear.weight"),
        &[2 * dout, sdim],
    );
    z(m, &format!("{prefix}.norm2.style_linear.bias"), &[2 * dout]);
    if din != dout {
        z(m, &format!("{prefix}.conv1x1.weight"), &[dout, din, 1]);
        z(m, &format!("{prefix}.conv1x1.bias"), &[dout]);
    }
    if up {
        z(m, &format!("{prefix}.pool.weight"), &[din, 1, 3]);
        z(m, &format!("{prefix}.pool.bias"), &[din]);
    }
}

fn insert_adain_resblk(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    din: usize,
    dout: usize,
    sdim: usize,
    up: bool,
) {
    z(m, &format!("{prefix}.n1.fc.weight"), &[2 * din, sdim]);
    z(m, &format!("{prefix}.n1.fc.bias"), &[2 * din]);
    z(m, &format!("{prefix}.n2.fc.weight"), &[2 * dout, sdim]);
    z(m, &format!("{prefix}.n2.fc.bias"), &[2 * dout]);
    z(m, &format!("{prefix}.c1.weight"), &[dout, din, 3]);
    z(m, &format!("{prefix}.c1.bias"), &[dout]);
    z(m, &format!("{prefix}.c2.weight"), &[dout, dout, 3]);
    z(m, &format!("{prefix}.c2.bias"), &[dout]);
    if din != dout {
        z(m, &format!("{prefix}.skip.weight"), &[dout, din, 1]);
        z(m, &format!("{prefix}.skip.bias"), &[dout]);
    }
    if up {
        z(m, &format!("{prefix}.pool.weight"), &[din, 1, 3]);
        z(m, &format!("{prefix}.pool.bias"), &[din]);
    }
}

fn insert_f0_predictor(m: &mut HashMap<String, DynTensor>) {
    let p = "predictor";
    let h = T_F0_HIDDEN;
    let bilstm_out = 2 * h;
    let bilstm_input = T_D_EN + T_STYLE;
    z(
        m,
        &format!("{p}.shared.weight_ih_l0"),
        &[4 * h, bilstm_input],
    );
    z(m, &format!("{p}.shared.weight_hh_l0"), &[4 * h, h]);
    z(m, &format!("{p}.shared.bias_ih_l0"), &[4 * h]);
    z(m, &format!("{p}.shared.bias_hh_l0"), &[4 * h]);
    z(
        m,
        &format!("{p}.shared.weight_ih_l0_reverse"),
        &[4 * h, bilstm_input],
    );
    z(m, &format!("{p}.shared.weight_hh_l0_reverse"), &[4 * h, h]);
    z(m, &format!("{p}.shared.bias_ih_l0_reverse"), &[4 * h]);
    z(m, &format!("{p}.shared.bias_hh_l0_reverse"), &[4 * h]);
    insert_adain_resblk(
        m,
        &format!("{p}.F0.0"),
        bilstm_out,
        bilstm_out,
        T_STYLE,
        false,
    );
    insert_adain_resblk(m, &format!("{p}.F0.1"), bilstm_out, h, T_STYLE, true);
    insert_adain_resblk(m, &format!("{p}.F0.2"), h, h, T_STYLE, false);
    z(m, &format!("{p}.F0_proj.weight"), &[1, h]);
    z(m, &format!("{p}.F0_proj.bias"), &[1]);
    insert_adain_resblk(
        m,
        &format!("{p}.N.0"),
        bilstm_out,
        bilstm_out,
        T_STYLE,
        false,
    );
    insert_adain_resblk(m, &format!("{p}.N.1"), bilstm_out, h, T_STYLE, true);
    insert_adain_resblk(m, &format!("{p}.N.2"), h, h, T_STYLE, false);
    z(m, &format!("{p}.N_proj.weight"), &[1, h]);
    z(m, &format!("{p}.N_proj.bias"), &[1]);
}

fn insert_decoder(m: &mut HashMap<String, DynTensor>) {
    let p = "decoder";
    let d_en = T_D_EN;
    let sdim = T_STYLE;
    let asr_res_ch = (d_en / 8).max(1);
    let hidden = 2 * d_en;
    let encode_in = d_en + 2;
    let decode_in = hidden + asr_res_ch + 2;
    z(m, &format!("{p}.F0_conv.weight"), &[1, 1, 3]);
    z(m, &format!("{p}.F0_conv.bias"), &[1]);
    z(m, &format!("{p}.N_conv.weight"), &[1, 1, 3]);
    z(m, &format!("{p}.N_conv.bias"), &[1]);
    z(m, &format!("{p}.asr_res.weight"), &[asr_res_ch, d_en, 1]);
    z(m, &format!("{p}.asr_res.bias"), &[asr_res_ch]);
    insert_stage1_resblk(m, &format!("{p}.encode"), encode_in, hidden, sdim, false);
    for i in 0..3 {
        insert_stage1_resblk(
            m,
            &format!("{p}.decode.{i}"),
            decode_in,
            hidden,
            sdim,
            false,
        );
    }
    insert_stage1_resblk(m, &format!("{p}.decode.3"), decode_in, d_en, sdim, true);
    let gp = format!("{p}.generator");
    let ch = T_GEN_CH;
    let next_ch = ch / 2;
    let n_bins = T_N_FFT / 2 + 1;
    z(m, &format!("{gp}.conv_pre.weight"), &[ch, ch, 7]);
    z(m, &format!("{gp}.conv_pre.bias"), &[ch]);
    z(m, &format!("{gp}.ups.0.weight"), &[ch, next_ch, 4]);
    z(m, &format!("{gp}.ups.0.bias"), &[next_ch]);
    z(
        m,
        &format!("{gp}.noise_convs.0.weight"),
        &[next_ch, 2 * n_bins, 1],
    );
    z(m, &format!("{gp}.noise_convs.0.bias"), &[next_ch]);
    insert_resblock(m, &format!("{gp}.noise_res.0"), next_ch, 11, 3);
    insert_resblock(m, &format!("{gp}.resblocks.0"), next_ch, 3, 2);
    z(
        m,
        &format!("{gp}.conv_post.weight"),
        &[2 * n_bins, next_ch, 7],
    );
    z(m, &format!("{gp}.conv_post.bias"), &[2 * n_bins]);
}

fn insert_source_module(m: &mut HashMap<String, DynTensor>) {
    let p = "decoder.generator.m_source";
    z(m, &format!("{p}.l_linear.weight"), &[1, 9]);
    z(m, &format!("{p}.l_linear.bias"), &[1]);
}
