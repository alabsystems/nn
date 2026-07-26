// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared synthetic weight construction for Kokoro compiled pipeline tests.
//!
//! Eliminates ~1,600 lines of duplicated helper functions across 8 test files.
//! All weight functions are parameterized by [`KokoroConfig`] so they work
//! with both miniaturized (D=8) and production-scale (D=512) dimensions.
//!
//! # Usage
//!
//! ```rust,ignore
//! use super::kokoro_test_weights as kw;
//! let config = kw::mini_test_config();
//! let weights = kw::all_weights(&config);
//! let kokoro = kw::build_kokoro_with_config(&config);
//! ```

use std::collections::HashMap;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_metal::compiled_kokoro::CompiledKokoro;
use nn_models::{KokoroConfig, PlbertConfig};
use nn_tts_verify::{HardBoundsConfig, RejectionPolicy};

fn cpu() -> Device {
    Device::Cpu
}

fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

// -- Primitive helpers --------------------------------------------------------

/// Insert a zero tensor into the weight map.
pub fn z(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    m.insert(
        name.to_string(),
        DynTensor::zeros(shape, DType::F32, &cpu()).unwrap(),
    );
}

/// Insert a ones tensor into the weight map.
pub fn ones(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    m.insert(
        name.to_string(),
        DynTensor::full(shape, 1.0, DType::F32, &cpu()).unwrap(),
    );
}

/// Insert a Conv1d weight+bias pair.
pub fn conv1d_w(
    m: &mut HashMap<String, DynTensor>,
    pfx: &str,
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
) {
    z(m, &format!("{pfx}.weight"), &[out_ch, in_ch, kernel]);
    z(m, &format!("{pfx}.bias"), &[out_ch]);
}

/// Insert BiLSTM weights for a single layer (forward + reverse).
pub fn bilstm_w(
    m: &mut HashMap<String, DynTensor>,
    pfx: &str,
    input_dim: usize,
    hidden_dim: usize,
) {
    let gate_dim = 4 * hidden_dim;
    z(m, &format!("{pfx}.weight_ih_l0"), &[gate_dim, input_dim]);
    z(m, &format!("{pfx}.weight_hh_l0"), &[gate_dim, hidden_dim]);
    z(m, &format!("{pfx}.bias_ih_l0"), &[gate_dim]);
    z(m, &format!("{pfx}.bias_hh_l0"), &[gate_dim]);
    z(
        m,
        &format!("{pfx}.weight_ih_l0_reverse"),
        &[gate_dim, input_dim],
    );
    z(
        m,
        &format!("{pfx}.weight_hh_l0_reverse"),
        &[gate_dim, hidden_dim],
    );
    z(m, &format!("{pfx}.bias_ih_l0_reverse"), &[gate_dim]);
    z(m, &format!("{pfx}.bias_hh_l0_reverse"), &[gate_dim]);
}

// -- Config -------------------------------------------------------------------

/// Miniaturized KokoroConfig for fast unit tests (D_EN=8, STYLE_DIM=4).
///
/// Matches the dimensions used by `kokoro_tts_tests_model.rs` and the majority
/// of Kokoro test files. Use this as the default for new tests.
pub fn mini_test_config() -> KokoroConfig {
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

// -- Segment weight builders --------------------------------------------------

/// PlBert (ALBERT) weights.
pub fn plbert_weights(m: &mut HashMap<String, DynTensor>, cfg: &KokoroConfig) {
    let vocab = cfg.plbert.vocab_size;
    let emb = cfg.plbert.embedding_dim;
    let hidden = cfg.plbert.hidden_size;
    let intermediate = cfg.plbert.intermediate_size;
    let p = "plbert";

    z(
        m,
        &format!("{p}.embeddings.word_embeddings.weight"),
        &[vocab, emb],
    );
    z(
        m,
        &format!("{p}.embeddings.position_embeddings.weight"),
        &[16, emb],
    );
    z(
        m,
        &format!("{p}.embeddings.token_type_embeddings.weight"),
        &[2, emb],
    );
    ones(m, &format!("{p}.embeddings.LayerNorm.weight"), &[emb]);
    z(m, &format!("{p}.embeddings.LayerNorm.bias"), &[emb]);
    z(
        m,
        &format!("{p}.encoder.embedding_hidden_mapping_in.weight"),
        &[hidden, emb],
    );
    z(
        m,
        &format!("{p}.encoder.embedding_hidden_mapping_in.bias"),
        &[hidden],
    );

    let lp = format!("{p}.encoder.albert_layer_groups.0.albert_layers.0");
    for name in &[
        "attention.query",
        "attention.key",
        "attention.value",
        "attention.dense",
    ] {
        z(m, &format!("{lp}.{name}.weight"), &[hidden, hidden]);
        z(m, &format!("{lp}.{name}.bias"), &[hidden]);
    }
    ones(m, &format!("{lp}.attention.LayerNorm.weight"), &[hidden]);
    z(m, &format!("{lp}.attention.LayerNorm.bias"), &[hidden]);
    z(m, &format!("{lp}.ffn.weight"), &[intermediate, hidden]);
    z(m, &format!("{lp}.ffn.bias"), &[intermediate]);
    z(
        m,
        &format!("{lp}.ffn_output.weight"),
        &[hidden, intermediate],
    );
    z(m, &format!("{lp}.ffn_output.bias"), &[hidden]);
    ones(m, &format!("{lp}.full_layer_layer_norm.weight"), &[hidden]);
    z(m, &format!("{lp}.full_layer_layer_norm.bias"), &[hidden]);
}

/// TextEncoder weights (convs + BiLSTM + linear projection).
pub fn text_encoder_weights(m: &mut HashMap<String, DynTensor>, cfg: &KokoroConfig) {
    let d_en = cfg.d_en;
    let vocab = cfg.plbert.vocab_size;
    let h = d_en / 2;
    let p = "text_encoder";

    z(m, &format!("{p}.embedding.weight"), &[vocab, d_en]);
    for i in 0..3 {
        z(m, &format!("{p}.convs.{i}.weight"), &[d_en, d_en, 5]);
        z(m, &format!("{p}.convs.{i}.bias"), &[d_en]);
        ones(m, &format!("{p}.norms.{i}.weight"), &[d_en]);
        z(m, &format!("{p}.norms.{i}.bias"), &[d_en]);
    }
    bilstm_w(m, &format!("{p}.lstm"), d_en, h);
    z(m, &format!("{p}.lstm.linear.weight"), &[d_en, d_en]);
    z(m, &format!("{p}.lstm.linear.bias"), &[d_en]);
}

/// ProsodyPredictor weights (duration encoder + final BiLSTM).
pub fn prosody_weights(m: &mut HashMap<String, DynTensor>, cfg: &KokoroConfig) {
    let d_en = cfg.d_en;
    let style_dim = cfg.style_dim;
    let h = d_en / 2;
    let lstm_input = d_en + style_dim;
    let p = "prosody_predictor";

    // DurationEncoder BiLSTM layer 0
    bilstm_w(m, &format!("{p}.duration.lstms.0"), lstm_input, h);
    // DurationEncoder AdaLayerNorm
    let n = format!("{p}.duration.norms.0");
    ones(m, &format!("{n}.norm.weight"), &[d_en]);
    z(m, &format!("{n}.norm.bias"), &[d_en]);
    z(m, &format!("{n}.fc.weight"), &[2 * d_en, style_dim]);
    z(m, &format!("{n}.fc.bias"), &[2 * d_en]);
    // DurationEncoder projection (max_dur=50)
    z(
        m,
        &format!("{p}.duration.duration_proj.weight"),
        &[50, d_en],
    );
    z(m, &format!("{p}.duration.duration_proj.bias"), &[50]);
    // Final ProsodyPredictor BiLSTM
    bilstm_w(m, &format!("{p}.lstm"), lstm_input, h);
}

/// AdaIN ResBlock weights (used in F0/Energy predictor).
pub fn adain_resblk_weights(
    m: &mut HashMap<String, DynTensor>,
    pfx: &str,
    dim_in: usize,
    dim_out: usize,
    style_dim: usize,
    upsample: bool,
) {
    z(m, &format!("{pfx}.n1.fc.weight"), &[2 * dim_in, style_dim]);
    z(m, &format!("{pfx}.n1.fc.bias"), &[2 * dim_in]);
    z(m, &format!("{pfx}.n2.fc.weight"), &[2 * dim_out, style_dim]);
    z(m, &format!("{pfx}.n2.fc.bias"), &[2 * dim_out]);
    z(m, &format!("{pfx}.c1.weight"), &[dim_out, dim_in, 3]);
    z(m, &format!("{pfx}.c1.bias"), &[dim_out]);
    z(m, &format!("{pfx}.c2.weight"), &[dim_out, dim_out, 3]);
    z(m, &format!("{pfx}.c2.bias"), &[dim_out]);
    if dim_in != dim_out {
        z(m, &format!("{pfx}.skip.weight"), &[dim_out, dim_in, 1]);
        z(m, &format!("{pfx}.skip.bias"), &[dim_out]);
    }
    if upsample {
        z(m, &format!("{pfx}.pool.weight"), &[dim_in, 1, 3]);
        z(m, &format!("{pfx}.pool.bias"), &[dim_in]);
    }
}

/// F0/Energy predictor weights.
pub fn f0_predictor_weights(m: &mut HashMap<String, DynTensor>, cfg: &KokoroConfig) {
    let d_en = cfg.d_en;
    let style_dim = cfg.style_dim;
    let h = cfg.f0_bilstm_hidden;
    let bo = 2 * h;
    let bilstm_input = d_en + style_dim;
    let p = "predictor";

    // Shared BiLSTM
    z(
        m,
        &format!("{p}.shared.forward.weight_ih_l0"),
        &[4 * h, bilstm_input],
    );
    z(m, &format!("{p}.shared.forward.weight_hh_l0"), &[4 * h, h]);
    z(m, &format!("{p}.shared.forward.bias_ih_l0"), &[4 * h]);
    z(m, &format!("{p}.shared.forward.bias_hh_l0"), &[4 * h]);
    z(
        m,
        &format!("{p}.shared.backward.weight_ih_l0"),
        &[4 * h, bilstm_input],
    );
    z(m, &format!("{p}.shared.backward.weight_hh_l0"), &[4 * h, h]);
    z(m, &format!("{p}.shared.backward.bias_ih_l0"), &[4 * h]);
    z(m, &format!("{p}.shared.backward.bias_hh_l0"), &[4 * h]);

    // F0 blocks: bo→bo, bo→h (upsample), h→h
    adain_resblk_weights(m, &format!("{p}.F0.0"), bo, bo, style_dim, false);
    adain_resblk_weights(m, &format!("{p}.F0.1"), bo, h, style_dim, true);
    adain_resblk_weights(m, &format!("{p}.F0.2"), h, h, style_dim, false);
    z(m, &format!("{p}.F0_proj.weight"), &[1, h]);
    z(m, &format!("{p}.F0_proj.bias"), &[1]);

    // Energy (N) blocks: same architecture
    adain_resblk_weights(m, &format!("{p}.N.0"), bo, bo, style_dim, false);
    adain_resblk_weights(m, &format!("{p}.N.1"), bo, h, style_dim, true);
    adain_resblk_weights(m, &format!("{p}.N.2"), h, h, style_dim, false);
    z(m, &format!("{p}.N_proj.weight"), &[1, h]);
    z(m, &format!("{p}.N_proj.bias"), &[1]);
}

/// Generator ResBlock weights (Snake activation + AdaIN).
pub fn resblock_weights(
    m: &mut HashMap<String, DynTensor>,
    pfx: &str,
    ch: usize,
    kernel_size: usize,
    num_dilations: usize,
    style_dim: usize,
) {
    for i in 0..num_dilations {
        z(
            m,
            &format!("{pfx}.convs1.{i}.weight"),
            &[ch, ch, kernel_size],
        );
        z(m, &format!("{pfx}.convs1.{i}.bias"), &[ch]);
        z(
            m,
            &format!("{pfx}.convs2.{i}.weight"),
            &[ch, ch, kernel_size],
        );
        z(m, &format!("{pfx}.convs2.{i}.bias"), &[ch]);
        z(
            m,
            &format!("{pfx}.adain1.{i}.fc.weight"),
            &[2 * ch, style_dim],
        );
        z(m, &format!("{pfx}.adain1.{i}.fc.bias"), &[2 * ch]);
        z(
            m,
            &format!("{pfx}.adain2.{i}.fc.weight"),
            &[2 * ch, style_dim],
        );
        z(m, &format!("{pfx}.adain2.{i}.fc.bias"), &[2 * ch]);
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

/// Stage1 ResBlock weights (FullDecoder encode/decode blocks).
pub fn stage1_resblk_weights(
    m: &mut HashMap<String, DynTensor>,
    pfx: &str,
    dim_in: usize,
    dim_out: usize,
    style_dim: usize,
    upsample: bool,
) {
    z(m, &format!("{pfx}.conv1.weight"), &[dim_out, dim_in, 3]);
    z(m, &format!("{pfx}.conv1.bias"), &[dim_out]);
    z(m, &format!("{pfx}.conv2.weight"), &[dim_out, dim_out, 3]);
    z(m, &format!("{pfx}.conv2.bias"), &[dim_out]);
    z(
        m,
        &format!("{pfx}.norm1.style_linear.weight"),
        &[2 * dim_in, style_dim],
    );
    z(m, &format!("{pfx}.norm1.style_linear.bias"), &[2 * dim_in]);
    z(
        m,
        &format!("{pfx}.norm2.style_linear.weight"),
        &[2 * dim_out, style_dim],
    );
    z(m, &format!("{pfx}.norm2.style_linear.bias"), &[2 * dim_out]);
    if dim_in != dim_out {
        z(m, &format!("{pfx}.conv1x1.weight"), &[dim_out, dim_in, 1]);
        z(m, &format!("{pfx}.conv1x1.bias"), &[dim_out]);
    }
    if upsample {
        z(m, &format!("{pfx}.pool.weight"), &[dim_in, 1, 3]);
        z(m, &format!("{pfx}.pool.bias"), &[dim_in]);
    }
}

/// FullDecoder + Generator weights.
pub fn decoder_weights(m: &mut HashMap<String, DynTensor>, cfg: &KokoroConfig) {
    let d_en = cfg.d_en;
    let style_dim = cfg.style_dim;
    let ch = cfg.gen_initial_channels;
    let n_fft = cfg.n_fft;
    let p = "decoder";

    let asr_res_ch = (d_en / 8).max(1);
    let hidden = 2 * d_en;
    let encode_in = d_en + 2;
    let decode_in = hidden + asr_res_ch + 2;

    // FullDecoder Stage1
    z(m, &format!("{p}.F0_conv.weight"), &[1, 1, 3]);
    z(m, &format!("{p}.F0_conv.bias"), &[1]);
    z(m, &format!("{p}.N_conv.weight"), &[1, 1, 3]);
    z(m, &format!("{p}.N_conv.bias"), &[1]);
    z(m, &format!("{p}.asr_res.weight"), &[asr_res_ch, d_en, 1]);
    z(m, &format!("{p}.asr_res.bias"), &[asr_res_ch]);
    stage1_resblk_weights(
        m,
        &format!("{p}.encode"),
        encode_in,
        hidden,
        style_dim,
        false,
    );
    for i in 0..3 {
        stage1_resblk_weights(
            m,
            &format!("{p}.decode.{i}"),
            decode_in,
            hidden,
            style_dim,
            false,
        );
    }
    stage1_resblk_weights(
        m,
        &format!("{p}.decode.3"),
        decode_in,
        d_en,
        style_dim,
        true,
    );

    // Generator weights under decoder.generator.*
    let gp = format!("{p}.generator");
    let next_ch = ch / 2;
    let n_bins = n_fft / 2 + 1;
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
    resblock_weights(m, &format!("{gp}.noise_res.0"), next_ch, 11, 3, style_dim);
    resblock_weights(m, &format!("{gp}.resblocks.0"), next_ch, 3, 2, style_dim);
    z(
        m,
        &format!("{gp}.conv_post.weight"),
        &[2 * n_bins, next_ch, 7],
    );
    z(m, &format!("{gp}.conv_post.bias"), &[2 * n_bins]);

    // SourceModule: SineGen + Linear(9, 1) + tanh
    let n_harmonics = 9;
    z(
        m,
        &format!("{gp}.m_source.l_linear.weight"),
        &[1, n_harmonics],
    );
    z(m, &format!("{gp}.m_source.l_linear.bias"), &[1]);
}

// -- Aggregate builders -------------------------------------------------------

/// Construct the full set of synthetic weights for all Kokoro segments.
///
/// Returns a `HashMap<String, DynTensor>` suitable for `VarBuilder::from_tensors`.
pub fn all_weights(cfg: &KokoroConfig) -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    plbert_weights(&mut m, cfg);
    z(
        &mut m,
        "bert_encoder.weight",
        &[cfg.d_en, cfg.plbert.hidden_size],
    );
    z(&mut m, "bert_encoder.bias", &[cfg.d_en]);
    text_encoder_weights(&mut m, cfg);
    prosody_weights(&mut m, cfg);
    f0_predictor_weights(&mut m, cfg);
    decoder_weights(&mut m, cfg);
    m
}

/// Build a `CompiledKokoro` from synthetic weights with the given config.
///
/// Calls `gpu_init()` and `metal_setup()` internally. Returns both the
/// compiled model and the pipeline cache for tests that need it.
pub fn build_kokoro_with_config(cfg: &KokoroConfig) -> (CompiledKokoro, nn_metal::PipelineCache) {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let weights = all_weights(cfg);
    // Load weights on GPU — CompiledKokoro::new() only transfers SourceModule
    // via ensure_source_device, leaving other sub-modules on VarBuilder's
    // device. CPU weights cause "gpu_data called on CPU tensor" (#3097).
    let vb = VarBuilder::from_tensors(weights, DType::F32, &gpu());
    let model =
        nn_models::KokoroModel::load(&vb, cfg).expect("KokoroModel::load with synthetic weights");
    // Use Warn policy: miniaturized zero-weight model produces near-silent audio
    // (RMS ~5e-9) which fails the non_silence hard bound (threshold 0.01).
    // With Reject policy (default since #3781), synthesize() returns Err.
    // Warn policy records failures in the certificate but returns Ok so tests
    // can inspect structural bounds and gate metrics. Matches the pattern in
    // compiled_kokoro_hard_bounds.rs.
    let mut hb = HardBoundsConfig::default();
    hb.rejection_policy = RejectionPolicy::Warn;
    (
        CompiledKokoro::new_with_hard_bounds(model, hb).expect("GPU init"),
        cache,
    )
}

/// Build a `CompiledKokoro` with [`mini_test_config`] defaults.
pub fn build_kokoro_mini() -> (CompiledKokoro, nn_metal::PipelineCache) {
    build_kokoro_with_config(&mini_test_config())
}
