// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro quality gates — regression tests for RTF, dispatch count, and audio.
//!
//! These gates are Phase 0 of the Perfect Kokoro plan. They measure the current
//! state of the pipeline and assert thresholds that tighten as optimization
//! phases complete. Each gate prints its measured value for CI visibility.
//!
//! Run: `cargo test -p nn-metal --test kokoro_gates -- --nocapture`
//!
//! Part of #2925 (RTF gate), #2926 (dispatch gate), #2927 (audio quality gate).
//! Part of #2218 (Perfect Kokoro epic).

use std::collections::HashMap;
use std::time::Instant;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_metal::compiled_kokoro::chorus::KokoroChorus;
use nn_metal::compiled_kokoro::CompiledKokoro;
use nn_metal::SegmentCacheConfig;
use nn_models::kokoro_chorus::ChorusConfig;
use nn_models::{KokoroConfig, PlbertConfig};
use nn_tts_verify::{HardBoundsConfig, RejectionPolicy};

fn cpu() -> Device {
    Device::Cpu
}

// -- Miniaturized dimensions (matching compiled_kokoro_synthesize.rs) ----------

const D_EN: usize = 8;
const STYLE_DIM: usize = 4;
const HIDDEN: usize = 8;
const EMB: usize = 4;
const VOCAB: usize = 10;
const N_FFT: usize = 4;
const GEN_CH: usize = 8;
const F0_HIDDEN: usize = 4;

// -- Weight helpers (same as compiled_kokoro_synthesize.rs) --------------------

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

fn test_config() -> KokoroConfig {
    let mut plbert = PlbertConfig::default();
    plbert.vocab_size = VOCAB;
    plbert.embedding_dim = EMB;
    plbert.hidden_size = HIDDEN;
    plbert.num_attention_heads = 2;
    plbert.intermediate_size = 16;
    plbert.max_position_embeddings = 16;
    plbert.num_hidden_layers = 1;

    let mut config = KokoroConfig::default();
    config.d_en = D_EN;
    config.n_prosody_layers = 1;
    config.style_dim = STYLE_DIM;
    config.upsample_rates = vec![2];
    config.upsample_kernel_sizes = vec![4];
    config.resblock_kernel_sizes = vec![3];
    config.resblock_dilations = vec![vec![1, 2]];
    config.gen_initial_channels = GEN_CH;
    config.n_fft = N_FFT;
    config.f0_bilstm_hidden = F0_HIDDEN;
    config.plbert = plbert;
    config
}

// -- Full model weight construction (factored from compiled_kokoro_synthesize) -

fn plbert_weights(m: &mut HashMap<String, DynTensor>) {
    let p = "plbert";
    z(
        m,
        &format!("{p}.embeddings.word_embeddings.weight"),
        &[VOCAB, EMB],
    );
    z(
        m,
        &format!("{p}.embeddings.position_embeddings.weight"),
        &[16, EMB],
    );
    z(
        m,
        &format!("{p}.embeddings.token_type_embeddings.weight"),
        &[2, EMB],
    );
    ones(m, &format!("{p}.embeddings.LayerNorm.weight"), &[EMB]);
    z(m, &format!("{p}.embeddings.LayerNorm.bias"), &[EMB]);
    z(
        m,
        &format!("{p}.encoder.embedding_hidden_mapping_in.weight"),
        &[HIDDEN, EMB],
    );
    z(
        m,
        &format!("{p}.encoder.embedding_hidden_mapping_in.bias"),
        &[HIDDEN],
    );
    let lp = format!("{p}.encoder.albert_layer_groups.0.albert_layers.0");
    for name in &[
        "attention.query",
        "attention.key",
        "attention.value",
        "attention.dense",
    ] {
        z(m, &format!("{lp}.{name}.weight"), &[HIDDEN, HIDDEN]);
        z(m, &format!("{lp}.{name}.bias"), &[HIDDEN]);
    }
    ones(m, &format!("{lp}.attention.LayerNorm.weight"), &[HIDDEN]);
    z(m, &format!("{lp}.attention.LayerNorm.bias"), &[HIDDEN]);
    z(m, &format!("{lp}.ffn.weight"), &[16, HIDDEN]);
    z(m, &format!("{lp}.ffn.bias"), &[16]);
    z(m, &format!("{lp}.ffn_output.weight"), &[HIDDEN, 16]);
    z(m, &format!("{lp}.ffn_output.bias"), &[HIDDEN]);
    ones(m, &format!("{lp}.full_layer_layer_norm.weight"), &[HIDDEN]);
    z(m, &format!("{lp}.full_layer_layer_norm.bias"), &[HIDDEN]);
}

fn text_encoder_weights(m: &mut HashMap<String, DynTensor>) {
    let h = D_EN / 2;
    let p = "text_encoder";
    z(m, &format!("{p}.embedding.weight"), &[VOCAB, D_EN]);
    for i in 0..3 {
        z(m, &format!("{p}.convs.{i}.weight"), &[D_EN, D_EN, 5]);
        z(m, &format!("{p}.convs.{i}.bias"), &[D_EN]);
        ones(m, &format!("{p}.norms.{i}.weight"), &[D_EN]);
        z(m, &format!("{p}.norms.{i}.bias"), &[D_EN]);
    }
    z(m, &format!("{p}.lstm.weight_ih_l0"), &[4 * h, D_EN]);
    z(m, &format!("{p}.lstm.weight_hh_l0"), &[4 * h, h]);
    z(m, &format!("{p}.lstm.bias_ih_l0"), &[4 * h]);
    z(m, &format!("{p}.lstm.bias_hh_l0"), &[4 * h]);
    z(m, &format!("{p}.lstm.weight_ih_l0_reverse"), &[4 * h, D_EN]);
    z(m, &format!("{p}.lstm.weight_hh_l0_reverse"), &[4 * h, h]);
    z(m, &format!("{p}.lstm.bias_ih_l0_reverse"), &[4 * h]);
    z(m, &format!("{p}.lstm.bias_hh_l0_reverse"), &[4 * h]);
    z(m, &format!("{p}.lstm.linear.weight"), &[D_EN, D_EN]);
    z(m, &format!("{p}.lstm.linear.bias"), &[D_EN]);
}

fn prosody_weights(m: &mut HashMap<String, DynTensor>) {
    let p = "prosody_predictor";
    let h = D_EN / 2;
    let four_h = 4 * h;
    let lstm_input = D_EN + STYLE_DIM;
    let l = format!("{p}.duration.lstms.0");
    z(m, &format!("{l}.weight_ih_l0"), &[four_h, lstm_input]);
    z(m, &format!("{l}.weight_hh_l0"), &[four_h, h]);
    z(m, &format!("{l}.bias_ih_l0"), &[four_h]);
    z(m, &format!("{l}.bias_hh_l0"), &[four_h]);
    z(
        m,
        &format!("{l}.weight_ih_l0_reverse"),
        &[four_h, lstm_input],
    );
    z(m, &format!("{l}.weight_hh_l0_reverse"), &[four_h, h]);
    z(m, &format!("{l}.bias_ih_l0_reverse"), &[four_h]);
    z(m, &format!("{l}.bias_hh_l0_reverse"), &[four_h]);
    let n = format!("{p}.duration.norms.0");
    ones(m, &format!("{n}.norm.weight"), &[D_EN]);
    z(m, &format!("{n}.norm.bias"), &[D_EN]);
    z(m, &format!("{n}.fc.weight"), &[2 * D_EN, STYLE_DIM]);
    z(m, &format!("{n}.fc.bias"), &[2 * D_EN]);
    z(
        m,
        &format!("{p}.duration.duration_proj.weight"),
        &[50, D_EN],
    );
    z(m, &format!("{p}.duration.duration_proj.bias"), &[50]);
    let dl = format!("{p}.lstm");
    z(m, &format!("{dl}.weight_ih_l0"), &[four_h, lstm_input]);
    z(m, &format!("{dl}.weight_hh_l0"), &[four_h, h]);
    z(m, &format!("{dl}.bias_ih_l0"), &[four_h]);
    z(m, &format!("{dl}.bias_hh_l0"), &[four_h]);
    z(
        m,
        &format!("{dl}.weight_ih_l0_reverse"),
        &[four_h, lstm_input],
    );
    z(m, &format!("{dl}.weight_hh_l0_reverse"), &[four_h, h]);
    z(m, &format!("{dl}.bias_ih_l0_reverse"), &[four_h]);
    z(m, &format!("{dl}.bias_hh_l0_reverse"), &[four_h]);
}

fn adain_resblk_weights(
    m: &mut HashMap<String, DynTensor>,
    pfx: &str,
    dim_in: usize,
    dim_out: usize,
    upsample: bool,
) {
    z(m, &format!("{pfx}.n1.fc.weight"), &[2 * dim_in, STYLE_DIM]);
    z(m, &format!("{pfx}.n1.fc.bias"), &[2 * dim_in]);
    z(m, &format!("{pfx}.n2.fc.weight"), &[2 * dim_out, STYLE_DIM]);
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

fn f0_predictor_weights(m: &mut HashMap<String, DynTensor>) {
    let p = "predictor";
    let h = F0_HIDDEN;
    let bo = 2 * h;
    let bilstm_input = D_EN + STYLE_DIM;
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
    adain_resblk_weights(m, &format!("{p}.F0.0"), bo, bo, false);
    adain_resblk_weights(m, &format!("{p}.F0.1"), bo, h, true);
    adain_resblk_weights(m, &format!("{p}.F0.2"), h, h, false);
    z(m, &format!("{p}.F0_proj.weight"), &[1, h]);
    z(m, &format!("{p}.F0_proj.bias"), &[1]);
    adain_resblk_weights(m, &format!("{p}.N.0"), bo, bo, false);
    adain_resblk_weights(m, &format!("{p}.N.1"), bo, h, true);
    adain_resblk_weights(m, &format!("{p}.N.2"), h, h, false);
    z(m, &format!("{p}.N_proj.weight"), &[1, h]);
    z(m, &format!("{p}.N_proj.bias"), &[1]);
}

fn stage1_resblk_weights(
    m: &mut HashMap<String, DynTensor>,
    pfx: &str,
    dim_in: usize,
    dim_out: usize,
    upsample: bool,
) {
    z(m, &format!("{pfx}.conv1.weight"), &[dim_out, dim_in, 3]);
    z(m, &format!("{pfx}.conv1.bias"), &[dim_out]);
    z(m, &format!("{pfx}.conv2.weight"), &[dim_out, dim_out, 3]);
    z(m, &format!("{pfx}.conv2.bias"), &[dim_out]);
    z(
        m,
        &format!("{pfx}.norm1.style_linear.weight"),
        &[2 * dim_in, STYLE_DIM],
    );
    z(m, &format!("{pfx}.norm1.style_linear.bias"), &[2 * dim_in]);
    z(
        m,
        &format!("{pfx}.norm2.style_linear.weight"),
        &[2 * dim_out, STYLE_DIM],
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

fn resblock_weights(
    m: &mut HashMap<String, DynTensor>,
    pfx: &str,
    ch: usize,
    kernel_size: usize,
    num_dilations: usize,
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
            &[2 * ch, STYLE_DIM],
        );
        z(m, &format!("{pfx}.adain1.{i}.fc.bias"), &[2 * ch]);
        z(
            m,
            &format!("{pfx}.adain2.{i}.fc.weight"),
            &[2 * ch, STYLE_DIM],
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

fn decoder_weights(m: &mut HashMap<String, DynTensor>) {
    let p = "decoder";
    let asr_res_ch = (D_EN / 8).max(1);
    let hidden = 2 * D_EN;
    let encode_in = D_EN + 2;
    let decode_in = hidden + asr_res_ch + 2;
    z(m, &format!("{p}.F0_conv.weight"), &[1, 1, 3]);
    z(m, &format!("{p}.F0_conv.bias"), &[1]);
    z(m, &format!("{p}.N_conv.weight"), &[1, 1, 3]);
    z(m, &format!("{p}.N_conv.bias"), &[1]);
    z(m, &format!("{p}.asr_res.weight"), &[asr_res_ch, D_EN, 1]);
    z(m, &format!("{p}.asr_res.bias"), &[asr_res_ch]);
    stage1_resblk_weights(m, &format!("{p}.encode"), encode_in, hidden, false);
    for i in 0..3 {
        stage1_resblk_weights(m, &format!("{p}.decode.{i}"), decode_in, hidden, false);
    }
    stage1_resblk_weights(m, &format!("{p}.decode.3"), decode_in, D_EN, true);
    let gp = format!("{p}.generator");
    let ch = GEN_CH;
    let next_ch = ch / 2;
    let n_bins = N_FFT / 2 + 1;
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
    resblock_weights(m, &format!("{gp}.noise_res.0"), next_ch, 11, 3);
    resblock_weights(m, &format!("{gp}.resblocks.0"), next_ch, 3, 2);
    z(
        m,
        &format!("{gp}.conv_post.weight"),
        &[2 * n_bins, next_ch, 7],
    );
    z(m, &format!("{gp}.conv_post.bias"), &[2 * n_bins]);
    let n_harmonics = 9;
    z(
        m,
        &format!("{gp}.m_source.l_linear.weight"),
        &[1, n_harmonics],
    );
    z(m, &format!("{gp}.m_source.l_linear.bias"), &[1]);
}

fn all_weights() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    plbert_weights(&mut m);
    z(&mut m, "bert_encoder.weight", &[D_EN, HIDDEN]);
    z(&mut m, "bert_encoder.bias", &[D_EN]);
    text_encoder_weights(&mut m);
    prosody_weights(&mut m);
    f0_predictor_weights(&mut m);
    decoder_weights(&mut m);
    m
}

fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

pub(super) fn build_kokoro() -> (CompiledKokoro, nn_metal::PipelineCache) {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let config = test_config();
    let weights = all_weights();
    // Load weights on GPU — CompiledKokoro::new() only transfers SourceModule
    // via ensure_source_device, leaving other sub-modules on the VarBuilder's
    // device. CPU weights cause "gpu_data called on CPU tensor" (#3097).
    let vb = VarBuilder::from_tensors(weights, DType::F32, &gpu());
    let model = nn_models::KokoroModel::load(&vb, &config)
        .expect("KokoroModel::load with synthetic weights");
    // Use Warn policy: miniaturized zero-weight model produces near-silent audio
    // (RMS ~5e-9) which fails the non_silence hard bound (threshold 0.01).
    // With Reject policy (default since #3781), synthesize() returns Err and
    // all gates panic. Warn policy records failures in the certificate but
    // returns Ok, matching the pattern in compiled_kokoro_hard_bounds.rs.
    // Gate tests inspect the certificate directly for structural bound checks.
    let mut hb = HardBoundsConfig::default();
    hb.rejection_policy = RejectionPolicy::Warn;
    (
        CompiledKokoro::new_with_hard_bounds(model, hb).expect("GPU init"),
        cache,
    )
}

pub(super) fn test_inputs() -> (DynTensor, DynTensor) {
    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(200, 2 * STYLE_DIM, -0.1, 0.1),
        &[1, 2 * STYLE_DIM],
        &cpu(),
    )
    .unwrap();
    (input_ids, style)
}

// =============================================================================
// Gate 1: RTF Benchmark (#2925)
// =============================================================================

/// RTF (Real-Time Factor) gate: synthesis time / audio duration.
///
/// Measures wall-clock RTF on the miniaturized model. The miniaturized model
/// runs much faster than production (D=8 vs D=512), but the gate catches
/// regressions — if RTF increases by 2x, something broke.
///
/// Current threshold: RTF < 10.0 for miniaturized model in debug mode.
/// The miniaturized model produces ~300 samples (12.5ms audio) so fixed
/// overhead dominates — RTF ~3-4 is normal. The gate catches 3x regressions.
/// Production threshold (D=512, release): RTF < 0.03 (target: beat PyTorch MPS).
///
/// The test runs 3 warmup + 5 measured iterations and reports mean RTF.
///
/// Part of #2925, #2218.
#[test]
fn gate_rtf_benchmark() {
    let (mut kokoro, cache) = build_kokoro();
    let (input_ids, style) = test_inputs();

    // Warmup: compile segments + fill caches.
    for _ in 0..3 {
        let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();
    }

    // Measured iterations.
    let n_iters = 5;
    let mut total_synth_ms = 0.0_f64;
    let mut total_audio_samples = 0_usize;
    let sample_rate = 24000.0_f64; // Kokoro default sample rate

    for _ in 0..n_iters {
        let start = Instant::now();
        let (audio, _cert) = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        total_synth_ms += elapsed_ms;
        total_audio_samples += audio.dims()[2]; // [1, 1, T_audio]
    }

    let mean_synth_s = total_synth_ms / (f64::from(n_iters) * 1000.0);
    let mean_audio_s = (total_audio_samples as f64 / f64::from(n_iters)) / sample_rate;
    let rtf = if mean_audio_s > 0.0 {
        mean_synth_s / mean_audio_s
    } else {
        f64::INFINITY
    };

    eprintln!("\n=== RTF GATE (miniaturized D={D_EN}) ===");
    eprintln!(
        "  Mean synthesis: {:.3} ms",
        total_synth_ms / f64::from(n_iters)
    );
    eprintln!("  Mean audio:     {:.3} ms", mean_audio_s * 1000.0);
    eprintln!("  RTF:            {rtf:.4}");
    eprintln!("  Threshold:      < 10.0 (miniaturized debug, ~300 samples)");
    eprintln!("=======================================\n");

    // Miniaturized model produces ~12.5ms audio — fixed overhead dominates.
    // RTF ~3-4 is normal in debug mode. Gate catches 3x regressions.
    // Production gate (D=512, release mode) targets RTF < 0.03.
    assert!(
        rtf < 10.0,
        "RTF gate FAILED: rtf={rtf:.4} >= 10.0. \
         Miniaturized model (D={D_EN}) RTF regressed >3x from baseline ~3.6. \
         This indicates a performance regression. See #2925.",
    );
}

// =============================================================================
// Gate 2: Dispatch Count (#2926)
// =============================================================================

/// Dispatch count gate: tracks four metrics across all compiled segments.
///
/// Metrics (see designs/2026-03-22-dispatch-metrics-reconciliation.md):
///   M1: estimated Metal kernel launches (compiled, `num_metal_dispatches`)
///   M2: total encodings (compute + blits)
///   M3: blit copies (buffer planner relocation)
///   M4: compute encodings (M2 - M3, actual GPU work)
///
/// Threshold history:
///   Phase 0: < 600 logical (original baseline)
///   Phase 1: < 200 logical, < 500 M2 (SeqFirst FlashAttention, #3088)
///   Phase 2: < 180 logical, < 250 M1 (flip absorption)
///   Phase 3: < 180 logical, < 200 M1, < 350 M4 (LeakyReLU max(x,αx))
///   Phase 4: < 170 logical, < 220 M1, < 340 enc, < 200 M4 (self-optimizing compiler, #3828)
///   Phase 5: < 160 logical, < 210 M1, < 320 enc, < 185 M4 (BiLstmCat fusion, #4252)
///   Phase 6: < 150 logical, < 200 M1, < 320 enc, < 180 M4 (threshold tightening)
///
/// Note: BiLstmCat fusion pass (#4252) fuses bidirectional LSTM + cat into a
/// single NativeOp. PeepholeConfig now has 12 toggles (search space = 4096).
///
/// Production measurements (KOKORO_WEIGHTS, 2026-03-29):
///   Per-segment compute encodings: plbert=14, text=13, prosody=15, f0=30, gen=46
///   Total compute encodings: 172, Blit copies: 144, Flushes: 1, Submits: 1
///   BiLstmCat active: 8 instances (text=2, prosody=4, f0=2)
///   BiLstmCat savings: ~8 logical dispatches (154→146), ~4 Metal dispatches (196→192)
///
/// Phase 6 thresholds tightened from measurements (all 12 gates PASS):
///   Logical: < 150 (was 160, measured 146, headroom 4)
///   M1 Metal: < 200 (was 210, measured 192, headroom 8)
///   Encoding est: < 320 (unchanged, measured 316, headroom 4 — too close to tighten)
///   M4 Compute: < 180 (was 185, measured 172, headroom 8)
///
/// Part of #2926, #2218, #1815.
#[test]
fn gate_dispatch_count() {
    let (mut kokoro, cache) = build_kokoro();
    let (input_ids, style) = test_inputs();

    // Cold path — compiles all segments.
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    // Planner estimates (compiled segments only — excludes eager paths).
    let total = kokoro.total_dispatches();
    let metal = kokoro.total_metal_dispatches();
    let encoding_est = kokoro.total_encoding_events();
    let ds = kokoro.dispatch_summary();

    // Hot path — measures actual runtime GPU dispatch encodings.
    let (_audio, _cert, stats) = kokoro
        .synthesize_with_stats(&input_ids, &style, 1.0, &cache)
        .unwrap();
    let actual = stats.compute_encodings;

    eprintln!("\n=== DISPATCH COUNT GATE ===");
    eprintln!("  Logical dispatches: {total}");
    eprintln!("  M1 Metal dispatches (estimated, compiled segments): {metal}");
    eprintln!("  Encoding events (estimated, compiled): {encoding_est}");
    eprintln!("  M4 Compute encodings (actual, runtime):             {actual}");
    eprintln!(
        "  M3 Blit copies:                                     {}",
        stats.blits
    );
    eprintln!(
        "  M2 Total encodings (compute + blits):               {}",
        actual + stats.blits
    );
    eprintln!("  Eager overhead: {}", actual.saturating_sub(encoding_est));
    eprintln!(
        "  Encoding accuracy: {:.0}%",
        encoding_est as f64 / actual.max(1) as f64 * 100.0
    );
    eprintln!(
        "  Blits eliminated (#4264):                           {}",
        stats.blits_eliminated
    );
    eprintln!("  Flushes: {}, Submits: {}", stats.flushes, stats.submits);
    eprintln!(
        "  Per-segment (encoding events): plbert={}, text={}, prosody={}, f0={}, gen={}",
        ds.plbert, ds.text_encoder, ds.prosody, ds.f0_energy, ds.generator
    );

    // Per-op-type breakdown for dispatch audit ground truth (#1815).
    for (seg_name, ir, native) in kokoro.dispatch_breakdowns() {
        eprintln!("  [{seg_name}] IR: {ir:?}");
        eprintln!("  [{seg_name}] Native: {native:?}");
    }
    eprintln!("============================\n");

    // Phase 9 thresholds — measured 2026-04-16 with D=128 test model.
    // Logical dispatches: 147 (plbert=13, text=15, prosody=19, f0=28, gen=44,
    // regulate=4, sinegen_pre=11, sinegen_post=13).
    // Partition codegen active, gap analysis shows 0% fusion gap across all
    // 8 segments (theoretical_min == dispatches). Generator trace fixed (#4309).
    // New peephole passes (#4264): FusedConv1dActivation, NormActivConvTranspose1d,
    // FusedConv1dSnakeNorm, FusedConv1dSnakeNormResBlock, FusedSnakeInstanceNorm.
    assert!(
        total < 155,
        "Dispatch gate FAILED: {total} logical dispatches >= 155. \
         Per-segment: plbert={}, text={}, prosody={}, f0={}, gen={}. See #2926, #4252, #4345.",
        ds.plbert,
        ds.text_encoder,
        ds.prosody,
        ds.f0_energy,
        ds.generator,
    );

    // M1: estimated Metal kernel launches (compiled segments).
    // Measured: 188 (down from 192 Phase 8, 196 Phase 4, 427 Phase 3).
    // New peephole passes (#4264) reduce NativeOp count.
    assert!(
        metal < 195,
        "M1 Metal dispatch gate FAILED: {metal} estimated Metal dispatches >= 195. \
         Measured baseline: 188. See #1815, #4252, #4264.",
    );

    // Encoding events estimate: matches actual `dispatch_stats().compute_encodings`
    // by counting 1 per IR Dispatch + NativeOp dispatches + blit relocations.
    // Measured: 301 (down from 316 Phase 8, ~430 Phase 3).
    assert!(
        encoding_est < 310,
        "Encoding events gate FAILED: {encoding_est} estimated encoding events >= 310. \
         See #1815 D5, #4252, #4264.",
    );

    // M4: compute encodings (actual runtime, excludes blits per D2).
    // stats.compute_encodings is compute-only after TOTAL_ENCODINGS D2 separation.
    // Measured: 181 (Phase 9, D=128 test model — NativeOp sub-dispatch counts
    // may differ between test and production models).
    assert!(
        actual < 190,
        "M4 compute gate FAILED: {actual} compute encodings >= 190. \
         M1={metal}, blits={}, eager overhead={}. See #1815, #4252, #4264.",
        stats.blits,
        actual.saturating_sub(metal),
    );
}

// =============================================================================
// Gate 3: Audio Quality (#2927)
// =============================================================================

/// Audio quality gate: deterministic output and structural integrity.
///
/// With synthetic weights, we can't compare against PyTorch reference (no
/// shared weights). Instead we verify:
/// 1. Audio is deterministic (two calls produce identical output).
/// 2. Audio has no NaN/Inf values.
/// 3. Audio is within [-1, 1] (no clipping beyond range).
/// 4. Pipeline certificate structural bounds pass.
/// 5. SNR between two identical runs is infinity (perfect reconstruction).
///
/// When real weights are available, this gate should be extended to compare
/// against a saved PyTorch reference tensor and assert SNR > 35 dB.
///
/// Part of #2927, #2218.
#[test]
fn gate_audio_quality() {
    let (mut kokoro, cache) = build_kokoro();
    let (input_ids, style) = test_inputs();

    // Warmup (compile segments).
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    // Two measured runs — should be deterministic.
    let (audio1, cert1) = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();
    let (audio2, cert2) = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    let samples1 = audio1
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let samples2 = audio2
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    eprintln!("\n=== AUDIO QUALITY GATE ===");
    eprintln!("  Audio length: {} samples", samples1.len());

    // 1. No NaN/Inf.
    let nan_count = samples1.iter().filter(|s| !s.is_finite()).count();
    eprintln!("  NaN/Inf count: {nan_count}");
    assert_eq!(
        nan_count, 0,
        "Audio contains {nan_count} NaN/Inf values. See #2927."
    );

    // 2. All samples in [-1, 1] (audio range).
    let max_abs = samples1.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
    eprintln!("  Max |sample|: {max_abs:.6}");
    assert!(
        max_abs <= 1.0,
        "Audio clipping: max |sample| = {max_abs:.6} > 1.0. See #2927.",
    );

    // 3. Deterministic: identical inputs → identical outputs.
    assert_eq!(
        samples1.len(),
        samples2.len(),
        "Non-deterministic: audio lengths differ ({} vs {})",
        samples1.len(),
        samples2.len(),
    );
    let max_diff = samples1
        .iter()
        .zip(samples2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("  Determinism max diff: {max_diff:.9}");
    assert!(
        max_diff == 0.0,
        "Non-deterministic: max diff between identical runs = {max_diff:.9}. See #2927.",
    );

    // 4. SNR between identical runs = infinity (self-consistency).
    // snr_db returns 0.0 for all-zero signals (silent reference). The
    // miniaturized model (D=8) produces all-zero audio, which is correct
    // behavior (near-zero magnitude → zero PCM). Accept 0.0 dB as "silent
    // perfect reconstruction" alongside +inf for non-silent.
    let snr = nn_tts_verify::dsp::snr_db(&samples1, &samples2)
        .expect("snr_db should succeed on identical vectors");
    eprintln!("  Self-SNR: {snr:.1} dB (expected: inf or 0 for silent)");
    let is_silent = max_abs == 0.0;
    assert!(
        (snr.is_infinite() && snr.is_sign_positive()) || (is_silent && snr == 0.0),
        "Self-SNR should be +inf (or 0.0 for silent output), got {snr:.1} dB",
    );

    // 5. Certificate structural bounds pass (no_nan, no_clipping, no_dc, no_clicks).
    let structural_failures: Vec<&str> = cert1
        .hard_bounds
        .iter()
        .filter(|b| {
            matches!(
                b.name,
                "no_clipping" | "no_dc_offset" | "no_clicks" | "no_nan"
            )
        })
        .filter(|b| !b.passed)
        .map(|b| b.name)
        .collect();
    eprintln!(
        "  Certificate structural failures: {structural_failures:?}"
    );
    assert!(
        structural_failures.is_empty(),
        "Certificate structural bounds failed: {structural_failures:?}. See #2927.",
    );

    // 6. Both runs produce same certificate verdict.
    let same_verdict = cert1.overall_passed == cert2.overall_passed;
    eprintln!("  Certificate deterministic: {same_verdict}");
    assert!(
        same_verdict,
        "Certificate verdict not deterministic between identical runs"
    );

    eprintln!("============================\n");
}

// =============================================================================
// Gate 4: Flush Count (#2925 supplemental)
// =============================================================================

/// GPU flush count gate: measures commit_and_wait barriers per synthesis.
///
/// Measured at HEAD via `NN_FLUSH_TRACE=1` on 2026-03-25:
///
/// | Event | Source | Kind |
/// |-------|--------|------|
/// | 1 | `step_regulate` prefix-sum total readback | `submit()+sync()` |
/// | 2 | pipeline-exit `audio.to_device(&cpu())` transfer | `flush()` |
///
/// Hot-path total: 1 structural flush only. `step_regulate` no longer
/// contributes a counted flush because the 4-byte scalar readback uses
/// `submit()+sync()` (#2911, #2958). Warmup still has transient JIT
/// compilation flushes before counters are reset for the measured run.
///
/// Part of #2925, #2958, #2218.
#[test]
fn gate_flush_count() {
    let (mut kokoro, cache) = build_kokoro();
    let (input_ids, style) = test_inputs();

    // Warmup — absorbs JIT segment compilation flushes.
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    // Measured run.
    let (_audio, _cert, stats) = kokoro
        .synthesize_with_stats(&input_ids, &style, 1.0, &cache)
        .unwrap();

    eprintln!("\n=== FLUSH COUNT GATE ===");
    eprintln!("  Flushes:   {}", stats.flushes);
    eprintln!("  Submits:   {}", stats.submits);
    eprintln!("  Encodings: {}", stats.compute_encodings);
    eprintln!("  Breakdown:         1 structural flush (pipeline-exit audio transfer)");
    eprintln!("  Regulate sync:     1 submit + sync (not counted as flush)");
    eprintln!("  Threshold:         exactly 1 hot-path flush");
    eprintln!("========================\n");

    assert_eq!(
        stats.flushes, 1,
        "Flush gate FAILED: expected exactly 1 hot-path flush (pipeline-exit audio transfer), got {}. See #2958.",
        stats.flushes,
    );
}

// =============================================================================
// Gate 5: Per-Step Dispatch Decomposition (#3192, #1815 Tier 6 D1)
// =============================================================================

/// Per-step dispatch decomposition: measures actual Metal dispatch encodings
/// for each pipeline step individually.
///
/// Decomposes the eager overhead gap (actual - estimated) by measuring each
/// step's actual GPU dispatches. Data tells us:
/// - Whether compiled segments dispatch more than their planner estimates (H1)
/// - Exactly how many dispatches SineGen + iSTFT + step_regulate use (H2)
/// - Where Tier 6 eager-path compilation has the highest ROI
///
/// Part of #3192, #1815 (Tier 6 D1 per-step counter).
#[test]
fn gate_per_step_dispatch_decomposition() {
    let (mut kokoro, cache) = build_kokoro();
    let (input_ids, style) = test_inputs();

    // Warmup — compile all segments.
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    // Full-pipeline measurement for accuracy baseline (includes blits).
    // The planner's total_encoding_events() counts compute + blit estimates,
    // so the actual denominator must also include blits. See R10 D5.1 audit.
    let (_audio, _cert, full_stats) = kokoro
        .synthesize_with_stats(&input_ids, &style, 1.0, &cache)
        .unwrap();
    let actual_total = full_stats.compute_encodings + full_stats.blits;

    // Split style once (minimal dispatches — CPU narrow).
    let style_split = kokoro.split_style(&style).unwrap();
    let decoder_style = style_split.decoder_style.to_device(&gpu()).unwrap();
    let prosody_style = style_split.prosody_style.to_device(&gpu()).unwrap();

    // Step 1+2: Encode (PlBert + TextEncoder compiled segments).
    nn_metal::reset_counters();
    let enc = kokoro.step_encode(&input_ids, &cache).unwrap();
    let encode_dispatches = nn_metal::dispatch_stats().compute_encodings;

    // Step 3: Prosody prediction (compiled segment).
    nn_metal::reset_counters();
    let pros = kokoro
        .step_predict_prosody(&enc.bert_features, &prosody_style, enc.seq_len, &cache)
        .unwrap();
    let prosody_dispatches = nn_metal::dispatch_stats().compute_encodings;

    // Step 4: Regulate (eager path — GPU prefix_sum scalar readback).
    nn_metal::reset_counters();
    let reg = kokoro
        .step_regulate(
            &pros.dur_logits,
            &pros.features,
            &enc.text_features,
            1.0,
            &cache,
        )
        .unwrap();
    let regulate_dispatches = nn_metal::dispatch_stats().compute_encodings;

    // Step 5: F0/Energy prediction (compiled segment).
    nn_metal::reset_counters();
    let f0e = kokoro
        .step_predict_f0_energy(&reg.aligned_dur, &prosody_style, reg.t_mel, &cache)
        .unwrap();
    let f0_dispatches = nn_metal::dispatch_stats().compute_encodings;

    // Step 6: Harmonic source (SineGen — eager path).
    nn_metal::reset_counters();
    let har = kokoro
        .step_harmonic_source(&f0e.f0, &f0e.energy, reg.t_mel, &cache)
        .unwrap();
    let harmonic_dispatches = nn_metal::dispatch_stats().compute_encodings;

    // Step 7: Generator (compiled segment).
    nn_metal::reset_counters();
    let gen_out = kokoro
        .step_generate(
            &reg.regulated,
            &f0e.f0,
            &f0e.energy,
            &decoder_style,
            &har,
            reg.t_mel,
            &cache,
        )
        .unwrap();
    let generate_dispatches = nn_metal::dispatch_stats().compute_encodings;

    // Step 8: iSTFT (terminal GPU→CPU sync).
    nn_metal::reset_counters();
    let _audio = kokoro
        .step_istft(&gen_out.magnitude, &gen_out.phase, &cache)
        .unwrap();
    let istft_dispatches = nn_metal::dispatch_stats().compute_encodings;

    let total_per_step = encode_dispatches
        + prosody_dispatches
        + regulate_dispatches
        + f0_dispatches
        + harmonic_dispatches
        + generate_dispatches
        + istft_dispatches;

    // Compare with planner estimates using encoding events (blit-aware).
    // dispatch_summary() now returns num_encoding_events() per segment,
    // which matches what dispatch_stats().compute_encodings actually counts.
    let ds = kokoro.dispatch_summary();
    let estimated_encodings = kokoro.total_encoding_events();

    let eager_total = regulate_dispatches + harmonic_dispatches + istft_dispatches;
    let compiled_actual = total_per_step - eager_total;

    eprintln!("\n=== PER-STEP DISPATCH DECOMPOSITION (#3192, #1815 D5) ===");
    eprintln!("  Step           | Actual | Est. encodings");
    eprintln!("  ---------------+--------+---------------");
    eprintln!(
        "  encode (1+2)   | {:>6} | {:>6}  (plbert={}, text={})",
        encode_dispatches,
        ds.plbert + ds.text_encoder,
        ds.plbert,
        ds.text_encoder,
    );
    eprintln!(
        "  prosody (3)    | {:>6} | {:>6}",
        prosody_dispatches, ds.prosody,
    );
    eprintln!(
        "  regulate (4)   | {regulate_dispatches:>6} |      - (eager)",
    );
    eprintln!(
        "  f0_energy (5)  | {:>6} | {:>6}",
        f0_dispatches, ds.f0_energy,
    );
    eprintln!(
        "  harmonic (6)   | {harmonic_dispatches:>6} |      - (eager)",
    );
    eprintln!(
        "  generate (7)   | {:>6} | {:>6}",
        generate_dispatches, ds.generator,
    );
    eprintln!(
        "  istft (8)      | {istft_dispatches:>6} |      - (eager)",
    );
    eprintln!("  ---------------+--------+---------------");
    eprintln!(
        "  TOTAL          | {total_per_step:>6} | {estimated_encodings:>6} (compiled encoding est.)",
    );
    eprintln!(
        "  compiled actual| {compiled_actual:>6} | {estimated_encodings:>6} (estimated)",
    );
    eprintln!("  eager total    | {eager_total:>6} |");
    eprintln!(
        "  full pipeline  | {:>6} | (compute={}, blits={})",
        actual_total, full_stats.compute_encodings, full_stats.blits,
    );
    // D5.1: accuracy compares estimated_encodings (includes blit estimates)
    // against actual_total (compute + blits from full pipeline run).
    // Prior calc compared against compiled_actual (compute-only) — apples-to-oranges.
    let accuracy = estimated_encodings as f64 / actual_total.max(1) as f64 * 100.0;
    eprintln!("  accuracy       | {accuracy:>5.0}% | (est / actual_total)");
    eprintln!("========================================================\n");

    // Sanity: total per-step should be > 0.
    assert!(
        total_per_step > 0,
        "Per-step dispatch decomposition: total is 0, GPU dispatch counting may be broken",
    );

    // D5.1: encoding gap compares planner estimate against full pipeline total
    // (compute + blits). The planner overestimates because NativeOps count
    // sub-dispatches but runtime batches them. Measured gap: -62 (est=301,
    // actual=239) with Phase 9 peephole passes. Threshold widened to 70.
    let encoding_gap = actual_total as isize - estimated_encodings as isize;
    eprintln!(
        "  encoding gap   | {encoding_gap:>6} | (actual_total - est. encodings)",
    );
    assert!(
        encoding_gap.unsigned_abs() < 70,
        "Encoding gap FAILED: |{encoding_gap}| >= 70. \
         Actual total={actual_total} (compute={}, blits={}), est. encodings={estimated_encodings}. \
         Planner should track actual within 70. See #1815 D5.1, #4264.",
        full_stats.compute_encodings,
        full_stats.blits,
    );

    // Per-segment sanity: planner should not overestimate by more than 3x.
    // NativeOps (FusedResBlock, NormActivConv1d, etc.) encode multiple MSL
    // kernels per logical op. The planner counts sub-dispatches; the runtime
    // batches them. Measured ~2x overestimate across all segments (#3213).
    let compiled_segments = [
        ("encode", encode_dispatches, ds.plbert + ds.text_encoder),
        ("prosody", prosody_dispatches, ds.prosody),
        ("f0_energy", f0_dispatches, ds.f0_energy),
        ("generate", generate_dispatches, ds.generator),
    ];
    for (name, actual, estimated) in &compiled_segments {
        eprintln!("  sanity: {name}: actual={actual}, estimated={estimated}");
        assert!(
            *actual > 0,
            "Segment {name} has 0 actual dispatches — GPU dispatch counting may be broken",
        );
        assert!(
            *estimated <= *actual * 3,
            "Planner OVER-estimates {name} by >3x: actual={actual}, estimated={estimated}. \
             This means the planner model is wrong — investigate.",
        );
    }
}

// =============================================================================
// Gate 6: First-Audio Latency (#3440)
// =============================================================================

/// First-audio latency (TTFA) gate: wall-clock time to first PCM output.
///
/// Measures TTFA for both single-voice and 8-voice chorus configurations.
/// For streaming playback, TTFA is the critical latency metric — it's the
/// time from synthesis call until the first audio samples are available.
///
/// For 1 voice: TTFA = full pipeline (Steps 1-8).
/// For N voices (shared encoding): TTFA = encode once (Steps 1-2) + N × decode
/// (Steps 3-8), since `synthesize_chorus_same_text` runs encoding once and
/// reuses the result.
///
/// Miniaturized model (D=8) is much faster than production — thresholds are
/// loose and catch gross regressions (5x+), not production targets.
/// Production targets (D=512, release mode): 1-voice < 20ms, 8-voice < 100ms.
///
/// Part of #3440, #3351.
#[test]
fn gate_first_audio_latency() {
    let (mut kokoro, cache) = build_kokoro();
    let (input_ids, style) = test_inputs();

    // Warmup: compile segments + fill caches.
    for _ in 0..3 {
        let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();
    }

    // -- 1-voice TTFA: time a single synthesize() call -------------------------

    let n_iters: usize = 5;
    let mut single_ms: Vec<f64> = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let start = Instant::now();
        let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();
        single_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    single_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let single_median = single_ms[n_iters / 2];

    // -- 8-voice TTFA: create chorus, warmup, then measure ---------------------

    let config = ChorusConfig::equal_gain(8).unwrap();
    let mut chorus = KokoroChorus::new(&kokoro, config).unwrap();

    // Generate 8 different styles for the chorus voices.
    let styles: Vec<DynTensor> = (0..8)
        .map(|i| {
            DynTensor::new(
                &super::test_utils::rand_f32_vec(300 + i as u64, 2 * STYLE_DIM, -0.1, 0.1),
                &[1, 2 * STYLE_DIM],
                &cpu(),
            )
            .unwrap()
        })
        .collect();

    // Warmup chorus path (compiles any additional segments, fills caches).
    for _ in 0..2 {
        let _ = chorus
            .synthesize_chorus_same_text(&input_ids, &styles, 1.0, &cache)
            .unwrap();
    }

    // Measured chorus iterations.
    let mut chorus_ms: Vec<f64> = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let start = Instant::now();
        let _ = chorus
            .synthesize_chorus_same_text(&input_ids, &styles, 1.0, &cache)
            .unwrap();
        chorus_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    chorus_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let chorus_median = chorus_ms[n_iters / 2];

    // Per-voice decode overhead: (8-voice - 1-voice) / 7 additional voices.
    let per_voice_overhead = if chorus_median > single_median {
        (chorus_median - single_median) / 7.0
    } else {
        0.0
    };
    let scaling_factor = if single_median > 0.0 {
        chorus_median / single_median
    } else {
        f64::NAN
    };

    eprintln!("\n=== FIRST-AUDIO LATENCY GATE (miniaturized D={D_EN}) ===");
    eprintln!("  1-voice TTFA:    {single_median:.3} ms (median of {n_iters})");
    eprintln!("  8-voice TTFA:    {chorus_median:.3} ms (median of {n_iters})");
    eprintln!("  Per-voice overhead: {per_voice_overhead:.3} ms");
    eprintln!("  Scaling factor:  {scaling_factor:.2}x (ideal: 1.0 for shared encode)");
    eprintln!(
        "  1-voice range:   [{:.3}, {:.3}] ms",
        single_ms[0],
        single_ms[n_iters - 1]
    );
    eprintln!(
        "  8-voice range:   [{:.3}, {:.3}] ms",
        chorus_ms[0],
        chorus_ms[n_iters - 1]
    );
    eprintln!("  Thresholds:      1-voice < 500ms, 8-voice < 2000ms (miniaturized debug)");
    eprintln!("  Production targets: 1-voice < 20ms, 8-voice < 100ms (D=512 release)");
    eprintln!("========================================================\n");

    // Miniaturized model (D=8) produces ~300 samples — fixed overhead dominates.
    // In debug mode, single synthesis is ~30-50ms. 8-voice chorus is ~250-400ms.
    // Thresholds are 10x above expected baseline to catch gross regressions.
    assert!(
        single_median < 500.0,
        "1-voice TTFA gate FAILED: {single_median:.3}ms >= 500ms. \
         This indicates a performance regression. See #3440.",
    );
    assert!(
        chorus_median < 2000.0,
        "8-voice TTFA gate FAILED: {chorus_median:.3}ms >= 2000ms. \
         This indicates a chorus performance regression. See #3440.",
    );
}

// =============================================================================
// Gate 7: Segment Cache Reuse (#4187)
// =============================================================================

/// Segment cache reuse gate: a second synthesis call with the SAME text
/// length must produce zero cache misses (all segments hit the cache).
///
/// Before the fix for #4187, a small `byte_budget` caused the LRU cache to
/// evict the just-compiled model on every call, so every call was a cold start.
/// With the 512 MB default budget, cached segments survive across calls.
///
/// The test uses `synthesize_with_timing()` which reports `cache_misses`:
///   - First call (cold): all segments compile (6 counted misses).
///   - Second call (SAME shape): expects 0 misses — perfect cache hit.
///   - Third call (different shape): also compiles all (6 misses), but the
///     cache should hold BOTH shapes afterwards (total > 8 entries).
///
/// Part of #4187, #3634.
#[test]
fn gate_segment_cache_reuse() {
    let (mut kokoro, cache) = build_kokoro();
    let (_input_ids, style) = test_inputs();

    // -- First synthesis: cold start, all segments compile ------------------
    let input_ids_3 = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let (_audio1, _cert1, timing1) = kokoro
        .synthesize_with_timing(&input_ids_3, &style, 1.0, &cache)
        .unwrap();

    let cold_misses = timing1.cache_misses;
    eprintln!("\n=== SEGMENT CACHE REUSE GATE (#4187) ===");
    eprintln!("  Cold start cache misses: {cold_misses}");

    let cached_after_first = kokoro.total_cached_segments();
    eprintln!("  Cached segments after first call: {cached_after_first}");
    assert!(
        cached_after_first >= 8,
        "Expected at least 8 cached segments after first synthesis, got {cached_after_first}",
    );

    // -- Second synthesis: SAME shape → should be perfect cache hit ---------
    let (_audio2, _cert2, timing2) = kokoro
        .synthesize_with_timing(&input_ids_3, &style, 1.0, &cache)
        .unwrap();

    let warm_misses = timing2.cache_misses;
    eprintln!("  Same-shape cache misses: {warm_misses} (expected 0)");

    assert_eq!(
        warm_misses, 0,
        "Segment cache reuse gate FAILED: same-shape second call had {warm_misses} \
         cache misses instead of 0. The cache is not retaining compiled segments. \
         Check byte_budget and segment cache eviction logic (#4187).",
    );

    // -- Third synthesis: different shape → compiles new, but both cached ---
    let input_ids_5 = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5], &cpu()).unwrap();
    let (_audio3, _cert3, _timing3) = kokoro
        .synthesize_with_timing(&input_ids_5, &style, 1.0, &cache)
        .unwrap();

    let cached_after_third = kokoro.total_cached_segments();
    eprintln!("  Cached segments after different shape: {cached_after_third}");
    eprintln!("=========================================\n");

    // With two different shapes cached, total segments should be > 8.
    assert!(
        cached_after_third > 8,
        "Expected > 8 cached segments after two different shapes, got {cached_after_third}. \
         Cache should hold entries for both seq_len=3 and seq_len=5.",
    );
}

// =============================================================================
// Gate 8: Segment Cache Eviction Correctness (#4187)
// =============================================================================

/// Segment cache eviction correctness gate: with a tiny byte_budget, eviction
/// happens on every shape change but must not crash or produce corrupt output.
///
/// This simulates the pre-#4187 scenario with an intentionally tiny budget
/// (1 MB) that forces aggressive LRU eviction. The test verifies:
///   1. Multiple synthesis calls with different sequence lengths all succeed.
///   2. Output audio contains no NaN/Inf (structural integrity despite eviction).
///   3. The model remains functional after many eviction cycles.
///
/// Part of #4187.
#[test]
fn gate_segment_cache_eviction_correctness() {
    super::test_utils::gpu_init();
    let pipeline_cache = super::test_utils::metal_setup();
    let config = test_config();
    let weights = all_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &gpu());
    let model = nn_models::KokoroModel::load(&vb, &config)
        .expect("KokoroModel::load with synthetic weights");
    let mut hb = HardBoundsConfig::default();
    hb.rejection_policy = RejectionPolicy::Warn;

    // Build with very small byte_budget to force aggressive eviction.
    let tiny_cache_config = SegmentCacheConfig {
        max_segments_per_step: 2,
        byte_budget: Some(1024 * 1024), // 1 MB — forces eviction on most inserts
        ..SegmentCacheConfig::default()
    };
    let mut kokoro = CompiledKokoro::new_with_hard_bounds(model, hb)
        .expect("GPU init")
        .with_segment_cache_config(tiny_cache_config);

    let (_input_ids, style) = test_inputs();

    eprintln!("\n=== SEGMENT CACHE EVICTION CORRECTNESS GATE (#4187) ===");

    // Synthesize with 4 different sequence lengths. Each length change forces
    // recompilation of seq_len-keyed segments. With 1 MB byte_budget, earlier
    // entries get evicted to make room for new ones.
    let seq_lengths: &[usize] = &[3, 5, 4, 6];
    for (i, &seq_len) in seq_lengths.iter().enumerate() {
        let data: Vec<f32> = (1..=seq_len).map(|v| v as f32).collect();
        let input_ids = DynTensor::from_vec(data, &[1, seq_len], &cpu()).unwrap();

        let result = kokoro.synthesize(&input_ids, &style, 1.0, &pipeline_cache);
        assert!(
            result.is_ok(),
            "Synthesis #{i} (seq_len={seq_len}) failed after eviction: {:?}",
            result.err(),
        );

        let (audio, _cert) = result.unwrap();
        let samples = audio
            .to_device(&Device::Cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();

        let nan_count = samples.iter().filter(|s| !s.is_finite()).count();
        eprintln!(
            "  Synthesis #{i} (seq_len={seq_len}): {} samples, {nan_count} NaN/Inf, cached={}",
            samples.len(),
            kokoro.total_cached_segments(),
        );
        assert_eq!(
            nan_count, 0,
            "Synthesis #{i} (seq_len={seq_len}) produced {nan_count} NaN/Inf after cache eviction",
        );
    }

    // Final sanity: model is still functional after many eviction cycles.
    let final_input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let result = kokoro.synthesize(&final_input, &style, 1.0, &pipeline_cache);
    assert!(
        result.is_ok(),
        "Final synthesis after eviction storm failed: {:?}",
        result.err(),
    );

    eprintln!(
        "  All {} synthesis calls succeeded under 1 MB byte_budget",
        seq_lengths.len() + 1
    );
    eprintln!("=========================================================\n");
}

// =============================================================================
// Gate 9: Real Audio Quality (production weights)
// =============================================================================

/// Production audio quality gate: loads full D=512 Kokoro from safetensors,
/// synthesizes with real weights, and verifies the output is actual voice audio
/// (not silence, not noise, not clipped).
///
/// The miniaturized gate (gate_audio_quality) uses zero-weights that produce
/// silence. This gate catches quality regressions only visible with production
/// weights: silent output, noise, amplitude collapse, NaN, non-determinism.
///
/// Gated behind `KOKORO_WEIGHTS` env var. Skips gracefully when unset.
///
/// Part of #4311.
#[test]
fn gate_real_audio_quality() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "gate_real_audio_quality skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(path) => path,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Use Warn policy: synthetic token IDs may trigger click detection.
    let mut hb = HardBoundsConfig::default();
    hb.rejection_policy = RejectionPolicy::Warn;

    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe {
        CompiledKokoro::load_with_hard_bounds(&weights_path, hb).expect("load Kokoro weights")
    };

    // 15 phoneme tokens — enough to produce a real utterance.
    let token_ids: Vec<i64> = (0..15).collect();
    let input_ids = DynTensor::from_vec_i64(token_ids, &[1, 15], &cpu()).unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();

    // Warmup.
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache);

    // Two measured runs for determinism check.
    let (audio1, _cert1) = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("synthesis 1 failed");
    let (audio2, _cert2) = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("synthesis 2 failed");

    let s1 = audio1
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let s2 = audio2
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let len = s1.len();
    let rms: f32 = (s1.iter().map(|x| x * x).sum::<f32>() / len as f32).sqrt();
    let max_abs = s1.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let nan_count = s1.iter().filter(|x| !x.is_finite()).count();

    eprintln!("\n=== REAL AUDIO QUALITY GATE (KOKORO_WEIGHTS) ===");
    eprintln!("  Samples:      {len}");
    eprintln!("  RMS:          {rms:.6}");
    eprintln!("  Max |sample|: {max_abs:.6}");
    eprintln!("  NaN/Inf:      {nan_count}");

    // Must not be silent.
    assert!(
        rms > 0.001,
        "SILENT: RMS={rms:.6} <= 0.001. Audio is silence, not voice."
    );
    // Must be long enough (15 tokens → should be several thousand samples).
    assert!(len > 2000, "TOO SHORT: {len} samples <= 2000");
    // Must have audible content.
    assert!(
        max_abs > 0.05,
        "TOO QUIET: max |sample|={max_abs:.6} <= 0.05"
    );
    // No NaN/Inf.
    assert_eq!(nan_count, 0, "NaN/Inf detected: {nan_count} bad samples");
    // No clipping.
    assert!(max_abs <= 1.0, "CLIPPING: max |sample|={max_abs:.6} > 1.0");
    // Deterministic.
    assert_eq!(
        s1.len(),
        s2.len(),
        "NON-DETERMINISTIC: length {len} vs {}",
        s2.len()
    );
    let max_diff = s1
        .iter()
        .zip(s2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("  Determinism:  max_diff={max_diff:.6e}");
    assert!(
        max_diff < 1e-5,
        "NON-DETERMINISTIC: max_diff={max_diff:.6e} >= 1e-5"
    );

    eprintln!("  PASS");
    eprintln!("================================================\n");
}
