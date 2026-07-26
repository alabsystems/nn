// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro benchmark suite — dispatch count, latency, and memory metrics.
//!
//! Measures compiled vs. eager execution for each Kokoro pipeline segment:
//! 1. Text pipeline (bert_encoder + TextEncoder BiLSTM)
//! 2. ProsodyPredictor (Conv1d + AdaLayerNorm + LSTM)
//! 3. F0EnergyPredictor (BiLSTM + AdainResBlk1d)
//! 4. Generator (vocoder: Conv1d, ConvTranspose1d, Snake, AdaIN)
//!
//! Reports per-segment and total:
//! - GPU dispatch count (compiled steps vs. eager ops)
//! - Wall-clock latency (compiled vs. eager, warmup + N iterations)
//! - Buffer plan memory (total_bytes vs. naive_total, reuse ratio)
//!
//! Run: `cargo test -p nn-metal --test kokoro_benchmark -- --nocapture`
//!
//! Part of #2231 (Kokoro benchmark suite).
//! Part of #2218 (Kokoro epic).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, TensorError, VarBuilder};
use nn_metal::compiled_model::CompiledModel;
use nn_models::kokoro_decoder::Generator;
use nn_models::kokoro_error::KokoroError;
use nn_models::kokoro_f0::F0EnergyPredictor;
use nn_models::kokoro_tts::{ProsodyPredictor, TextEncoder};
use nn_models::KokoroConfig;

fn cpu() -> Device {
    Device::Cpu
}

fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

// -- Dimensions ---------------------------------------------------------------

const GEN_CH: usize = 8;
const GEN_NEXT_CH: usize = 4;
const GEN_STYLE_DIM: usize = 4;
const GEN_N_FFT: usize = 4;
const GEN_N_BINS: usize = GEN_N_FFT / 2 + 1;
const GEN_KERNEL: usize = 3;
const D_EN: usize = 8;
const VOCAB_SIZE: usize = 16;
const STYLE_DIM: usize = 4;
const BILSTM_HIDDEN: usize = D_EN / 2;
const PROSODY_N_LAYERS: usize = 2;

const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

// -- Weight helpers -----------------------------------------------------------

fn z(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    m.insert(
        name.to_string(),
        DynTensor::full(shape, 0.01, DType::F32, &cpu()).unwrap(),
    );
}

fn conv1d_w(m: &mut HashMap<String, DynTensor>, pfx: &str, o: usize, i: usize, k: usize) {
    z(m, &format!("{pfx}.weight"), &[o, i, k]);
    z(m, &format!("{pfx}.bias"), &[o]);
}

fn bilstm_w(m: &mut HashMap<String, DynTensor>, pfx: &str, input_dim: usize, hidden_dim: usize) {
    z(
        m,
        &format!("{pfx}.weight_ih_l0"),
        &[4 * hidden_dim, input_dim],
    );
    z(
        m,
        &format!("{pfx}.weight_hh_l0"),
        &[4 * hidden_dim, hidden_dim],
    );
    z(m, &format!("{pfx}.bias_ih_l0"), &[4 * hidden_dim]);
    z(m, &format!("{pfx}.bias_hh_l0"), &[4 * hidden_dim]);
    z(
        m,
        &format!("{pfx}.weight_ih_l0_reverse"),
        &[4 * hidden_dim, input_dim],
    );
    z(
        m,
        &format!("{pfx}.weight_hh_l0_reverse"),
        &[4 * hidden_dim, hidden_dim],
    );
    z(m, &format!("{pfx}.bias_ih_l0_reverse"), &[4 * hidden_dim]);
    z(m, &format!("{pfx}.bias_hh_l0_reverse"), &[4 * hidden_dim]);
}

fn adain_resblk_w(
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
    conv1d_w(m, &format!("{pfx}.c1"), dim_out, dim_in, 3);
    conv1d_w(m, &format!("{pfx}.c2"), dim_out, dim_out, 3);
    if dim_in != dim_out {
        z(m, &format!("{pfx}.skip.weight"), &[dim_out, dim_in, 1]);
    }
    if upsample {
        z(m, &format!("{pfx}.pool.weight"), &[dim_in, 1, 3]);
        z(m, &format!("{pfx}.pool.bias"), &[dim_in]);
    }
}

fn resblock_w(
    m: &mut HashMap<String, DynTensor>,
    pfx: &str,
    ch: usize,
    k: usize,
    num_dilations: usize,
) {
    for i in 0..num_dilations {
        conv1d_w(m, &format!("{pfx}.convs1.{i}"), ch, ch, k);
        conv1d_w(m, &format!("{pfx}.convs2.{i}"), ch, ch, k);
        z(
            m,
            &format!("{pfx}.adain1.{i}.fc.weight"),
            &[2 * ch, GEN_STYLE_DIM],
        );
        z(m, &format!("{pfx}.adain1.{i}.fc.bias"), &[2 * ch]);
        z(
            m,
            &format!("{pfx}.adain2.{i}.fc.weight"),
            &[2 * ch, GEN_STYLE_DIM],
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

// -- Model builders -----------------------------------------------------------

fn build_generator() -> Generator {
    let mut m = HashMap::new();
    conv1d_w(&mut m, "conv_pre", GEN_CH, GEN_CH, 7);
    z(&mut m, "ups.0.weight", &[GEN_CH, GEN_NEXT_CH, 4]);
    z(&mut m, "ups.0.bias", &[GEN_NEXT_CH]);
    conv1d_w(&mut m, "noise_convs.0", GEN_NEXT_CH, 2 * GEN_N_BINS, 1);
    resblock_w(&mut m, "noise_res.0", GEN_NEXT_CH, 11, 3);
    resblock_w(&mut m, "resblocks.0", GEN_NEXT_CH, GEN_KERNEL, 1);
    conv1d_w(&mut m, "conv_post", 2 * GEN_N_BINS, GEN_NEXT_CH, 7);
    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    let mut config = KokoroConfig::default();
    config.upsample_rates = vec![2];
    config.upsample_kernel_sizes = vec![4];
    config.resblock_kernel_sizes = vec![GEN_KERNEL];
    config.resblock_dilations = vec![vec![1]];
    config.gen_initial_channels = GEN_CH;
    config.style_dim = GEN_STYLE_DIM;
    config.n_fft = GEN_N_FFT;
    Generator::load(&vb, &config).expect("Generator::load")
}

fn build_text_encoder() -> TextEncoder {
    let mut m = HashMap::new();
    let h = D_EN / 2;

    // Embedding
    z(&mut m, "embedding.weight", &[VOCAB_SIZE, D_EN]);

    // 3x Conv1d(d_en, d_en, k=5) + LayerNorm(d_en)
    for i in 0..3 {
        z(&mut m, &format!("convs.{i}.weight"), &[D_EN, D_EN, 5]);
        z(&mut m, &format!("convs.{i}.bias"), &[D_EN]);
        m.insert(
            format!("norms.{i}.weight"),
            DynTensor::full(&[D_EN], 1.0, DType::F32, &cpu()).unwrap(),
        );
        z(&mut m, &format!("norms.{i}.bias"), &[D_EN]);
    }

    // BiLSTM
    let p = "lstm";
    z(&mut m, &format!("{p}.weight_ih_l0"), &[4 * h, D_EN]);
    z(&mut m, &format!("{p}.weight_hh_l0"), &[4 * h, h]);
    z(&mut m, &format!("{p}.bias_ih_l0"), &[4 * h]);
    z(&mut m, &format!("{p}.bias_hh_l0"), &[4 * h]);
    z(&mut m, &format!("{p}.weight_ih_l0_reverse"), &[4 * h, D_EN]);
    z(&mut m, &format!("{p}.weight_hh_l0_reverse"), &[4 * h, h]);
    z(&mut m, &format!("{p}.bias_ih_l0_reverse"), &[4 * h]);
    z(&mut m, &format!("{p}.bias_hh_l0_reverse"), &[4 * h]);
    z(&mut m, &format!("{p}.linear.weight"), &[D_EN, D_EN]);
    z(&mut m, &format!("{p}.linear.bias"), &[D_EN]);

    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    TextEncoder::load(&vb, VOCAB_SIZE, D_EN).expect("TextEncoder")
}

fn build_prosody() -> ProsodyPredictor {
    let mut m = HashMap::new();
    let d = D_EN;
    let h = D_EN / 2;
    let s = STYLE_DIM;
    let lstm_input = d + s; // DurationEncoder concatenates features + style

    // DurationEncoder: n_layers BiLSTMs + AdaLayerNorms
    for i in 0..PROSODY_N_LAYERS {
        bilstm_w(&mut m, &format!("duration.lstms.{i}"), lstm_input, h);
        // AdaLayerNorm: fc (style -> gamma+beta) + optional norm weights
        z(
            &mut m,
            &format!("duration.norms.{i}.fc.weight"),
            &[2 * d, s],
        );
        z(&mut m, &format!("duration.norms.{i}.fc.bias"), &[2 * d]);
        m.insert(
            format!("duration.norms.{i}.norm.weight"),
            DynTensor::full(&[d], 1.0, DType::F32, &cpu()).unwrap(),
        );
        m.insert(
            format!("duration.norms.{i}.norm.bias"),
            DynTensor::full(&[d], 0.0, DType::F32, &cpu()).unwrap(),
        );
    }
    // Duration projection: d_model -> max_dur
    z(&mut m, "duration.duration_proj.weight", &[50, d]);
    z(&mut m, "duration.duration_proj.bias", &[50]);

    // Final duration BiLSTM
    bilstm_w(&mut m, "lstm", lstm_input, h);

    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    ProsodyPredictor::load(&vb, D_EN, STYLE_DIM, PROSODY_N_LAYERS, 50).expect("ProsodyPredictor")
}

fn build_f0_pred() -> F0EnergyPredictor {
    let mut m = HashMap::new();
    let s = STYLE_DIM;
    let bilstm_out = 2 * BILSTM_HIDDEN;
    let bilstm_input = D_EN + STYLE_DIM;

    // Shared BiLSTM: input = d_model + style_dim
    bilstm_w(&mut m, "shared", bilstm_input, BILSTM_HIDDEN);

    // F0 and Energy (N) heads: 3 AdainResBlk1d blocks each
    // Block dims: [bilstm_out -> bilstm_out], [bilstm_out -> bilstm_hidden (upsample)],
    //             [bilstm_hidden -> bilstm_hidden]
    for head in ["F0", "N"] {
        adain_resblk_w(
            &mut m,
            &format!("{head}.0"),
            bilstm_out,
            bilstm_out,
            s,
            false,
        );
        adain_resblk_w(
            &mut m,
            &format!("{head}.1"),
            bilstm_out,
            BILSTM_HIDDEN,
            s,
            true,
        );
        adain_resblk_w(
            &mut m,
            &format!("{head}.2"),
            BILSTM_HIDDEN,
            BILSTM_HIDDEN,
            s,
            false,
        );
        z(&mut m, &format!("{head}_proj.weight"), &[1, BILSTM_HIDDEN]);
        z(&mut m, &format!("{head}_proj.bias"), &[1]);
    }

    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    F0EnergyPredictor::load(&vb, D_EN, STYLE_DIM, BILSTM_HIDDEN).expect("F0EnergyPredictor")
}

// -- Metrics types + report ---------------------------------------------------

struct SegmentMetrics {
    name: &'static str,
    compiled_dispatches: usize,
    compiled_steps: usize,
    buffer_total_bytes: usize,
    buffer_naive_bytes: usize,
    compiled_latency_us: f64,
    eager_latency_us: f64,
}

fn report(metrics: &[SegmentMetrics]) {
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("  KOKORO BENCHMARK REPORT");
    eprintln!("  Warmup: {WARMUP_ITERS} iters, Bench: {BENCH_ITERS} iters");
    eprintln!("{}", "=".repeat(80));
    eprintln!(
        "  {:<20} {:>10} {:>10} {:>12} {:>12} {:>8}",
        "Segment", "Dispatches", "Steps", "Compiled us", "Eager us", "Speedup"
    );
    eprintln!("{}", "-".repeat(80));
    let (mut td, mut tc, mut te, mut tb, mut tn) = (0, 0.0, 0.0, 0, 0);
    for m in metrics {
        let speedup = m.eager_latency_us / m.compiled_latency_us.max(1.0);
        let reuse =
            100.0 * (1.0 - m.buffer_total_bytes as f64 / m.buffer_naive_bytes.max(1) as f64);
        eprintln!(
            "  {:<20} {:>10} {:>10} {:>12.1} {:>12.1} {:>7.2}x",
            m.name,
            m.compiled_dispatches,
            m.compiled_steps,
            m.compiled_latency_us,
            m.eager_latency_us,
            speedup
        );
        eprintln!(
            "  {:<20} buffer: {} B (naive: {} B, reuse: {:.1}%)",
            "", m.buffer_total_bytes, m.buffer_naive_bytes, reuse
        );
        td += m.compiled_dispatches;
        tc += m.compiled_latency_us;
        te += m.eager_latency_us;
        tb += m.buffer_total_bytes;
        tn += m.buffer_naive_bytes;
    }
    eprintln!("{}", "-".repeat(80));
    let ts = te / tc.max(1.0);
    let tr = 100.0 * (1.0 - tb as f64 / tn.max(1) as f64);
    eprintln!(
        "  {:<20} {:>10} {:>10} {:>12.1} {:>12.1} {:>7.2}x",
        "TOTAL", td, "-", tc, te, ts
    );
    eprintln!(
        "  {:<20} buffer: {} B (naive: {} B, reuse: {:.1}%)",
        "", tb, tn, tr
    );
    eprintln!("{}\n", "=".repeat(80));
}

// -- Timing helper -----------------------------------------------------------

fn bench_latency<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP_ITERS {
        f();
    }
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        f();
    }
    start.elapsed().as_micros() as f64 / BENCH_ITERS as f64
}

// -- Per-segment benchmark functions ------------------------------------------

fn bench_text_pipeline(cache: &nn_metal::PipelineCache) -> SegmentMetrics {
    let te = build_text_encoder();
    let batch = 1;
    let seq_len = 16;
    // Token IDs [B, T] — values in [0, VOCAB_SIZE)
    let token_ids: Vec<i64> = (0..batch * seq_len)
        .map(|i| (i % VOCAB_SIZE) as i64)
        .collect();
    let tokens = DynTensor::from_vec_i64(token_ids, &[batch, seq_len], &cpu()).unwrap();

    // Compiled path uses F32 tokens: Metal GPU rejects I64 transfer (#1697),
    // and the compiled Embedding kernel reads indices as float* → uint.
    // Eager path keeps I64 (Embedding::forward handles I64 natively via index_select).
    let tokens_f32 = tokens.to_dtype(DType::F32).unwrap();

    let (out, mut graph) = trace_graph(|| {
        let mut inp = tokens_f32.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        te.forward(&inp).map_err(KokoroError::into_tensor_error)
    })
    .unwrap();
    if let Some(id) = out.trace_id() {
        let _ = graph.set_primary_output(id);
    }
    let compiled = CompiledModel::builder(&graph, cache).build().unwrap();
    let bp = compiled.buffer_plan();
    let tokens_gpu = tokens_f32.to_device(&gpu()).unwrap();

    let compiled_us = bench_latency(|| {
        let _ = compiled.execute_dyn(cache, &[&tokens_gpu]).unwrap();
    });
    let eager_us = bench_latency(|| {
        let _ = te.forward(&tokens).unwrap();
    });

    SegmentMetrics {
        name: "TextPipeline",
        compiled_dispatches: compiled.num_dispatches(),
        compiled_steps: compiled.num_steps(),
        buffer_total_bytes: bp.total_bytes,
        buffer_naive_bytes: bp.naive_total,
        compiled_latency_us: compiled_us,
        eager_latency_us: eager_us,
    }
}

fn bench_prosody(cache: &nn_metal::PipelineCache) -> SegmentMetrics {
    let prosody = build_prosody();
    let (batch, t) = (1, 16);
    let text_feat = DynTensor::new(
        &super::test_utils::rand_f32_vec(50, batch * D_EN * t, -0.5, 0.5),
        &[batch, D_EN, t],
        &cpu(),
    )
    .unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(51, batch * STYLE_DIM, -0.5, 0.5),
        &[batch, STYLE_DIM],
        &cpu(),
    )
    .unwrap();

    let (dur_out, mut graph) = trace_graph(|| {
        let mut inp = text_feat.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let (d, _f) = prosody
            .forward(&inp, &sty)
            .map_err(KokoroError::into_tensor_error)?;
        Ok(d)
    })
    .unwrap();
    if let Some(id) = dur_out.trace_id() {
        let _ = graph.set_primary_output(id);
    }
    let compiled = CompiledModel::builder(&graph, cache).build().unwrap();
    let bp = compiled.buffer_plan();
    let text_gpu = text_feat.to_device(&gpu()).unwrap();
    let style_gpu = style.to_device(&gpu()).unwrap();

    let compiled_us = bench_latency(|| {
        let _ = compiled
            .execute_dyn(cache, &[&text_gpu, &style_gpu])
            .unwrap();
    });
    let eager_us = bench_latency(|| {
        let _ = prosody.forward(&text_feat, &style).unwrap();
    });

    SegmentMetrics {
        name: "ProsodyPredictor",
        compiled_dispatches: compiled.num_dispatches(),
        compiled_steps: compiled.num_steps(),
        buffer_total_bytes: bp.total_bytes,
        buffer_naive_bytes: bp.naive_total,
        compiled_latency_us: compiled_us,
        eager_latency_us: eager_us,
    }
}

fn bench_f0_energy(cache: &nn_metal::PipelineCache) -> SegmentMetrics {
    let f0_pred = build_f0_pred();
    let (batch, t_mel) = (1, 16);
    // F0EnergyPredictor expects aligned features with d_model+style_dim channels
    // (output of ProsodyPredictor already includes style).
    let aligned_dim = D_EN + STYLE_DIM;
    let aligned = DynTensor::new(
        &super::test_utils::rand_f32_vec(60, batch * aligned_dim * t_mel, -0.5, 0.5),
        &[batch, aligned_dim, t_mel],
        &cpu(),
    )
    .unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(61, batch * STYLE_DIM, -0.5, 0.5),
        &[batch, STYLE_DIM],
        &cpu(),
    )
    .unwrap();

    let (f0_out, mut graph) = trace_graph(|| {
        let mut inp = aligned.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let (f, _e) = f0_pred
            .forward(&inp, &sty)
            .map_err(KokoroError::into_tensor_error)?;
        Ok(f)
    })
    .unwrap();
    if let Some(id) = f0_out.trace_id() {
        let _ = graph.set_primary_output(id);
    }
    let compiled = CompiledModel::builder(&graph, cache).build().unwrap();
    let bp = compiled.buffer_plan();
    let aligned_gpu = aligned.to_device(&gpu()).unwrap();
    let style_gpu = style.to_device(&gpu()).unwrap();

    let compiled_us = bench_latency(|| {
        let _ = compiled
            .execute_dyn(cache, &[&aligned_gpu, &style_gpu])
            .unwrap();
    });
    let eager_us = bench_latency(|| {
        let _ = f0_pred.forward(&aligned, &style).unwrap();
    });

    SegmentMetrics {
        name: "F0EnergyPredictor",
        compiled_dispatches: compiled.num_dispatches(),
        compiled_steps: compiled.num_steps(),
        buffer_total_bytes: bp.total_bytes,
        buffer_naive_bytes: bp.naive_total,
        compiled_latency_us: compiled_us,
        eager_latency_us: eager_us,
    }
}

fn bench_generator(cache: &nn_metal::PipelineCache) -> SegmentMetrics {
    let generator = build_generator();
    let (batch, t_in, t_full) = (1, 16, 32);
    let x = DynTensor::new(
        &super::test_utils::rand_f32_vec(70, batch * GEN_CH * t_in, -0.5, 0.5),
        &[batch, GEN_CH, t_in],
        &cpu(),
    )
    .unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(71, batch * GEN_STYLE_DIM, -0.5, 0.5),
        &[batch, GEN_STYLE_DIM],
        &cpu(),
    )
    .unwrap();
    let har = DynTensor::new(
        &super::test_utils::rand_f32_vec(72, batch * 2 * GEN_N_BINS * t_full, -0.5, 0.5),
        &[batch, 2 * GEN_N_BINS, t_full],
        &cpu(),
    )
    .unwrap();

    let (out, mut graph) = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let mut h = har.clone();
        h.set_trace_id(record_input(h.dims(), DType::F32).unwrap());
        let (mag, _) = generator
            .forward(&inp, &sty, &h)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        Ok(mag)
    })
    .unwrap();
    if let Some(id) = out.trace_id() {
        let _ = graph.set_primary_output(id);
    }
    let compiled = CompiledModel::builder(&graph, cache).build().unwrap();
    let bp = compiled.buffer_plan();
    let x_gpu = x.to_device(&gpu()).unwrap();
    let style_gpu = style.to_device(&gpu()).unwrap();
    let har_gpu = har.to_device(&gpu()).unwrap();

    let compiled_us = bench_latency(|| {
        let _ = compiled
            .execute_dyn(cache, &[&x_gpu, &style_gpu, &har_gpu])
            .unwrap();
    });
    let eager_us = bench_latency(|| {
        let _ = generator.forward(&x, &style, &har).unwrap();
    });

    SegmentMetrics {
        name: "Generator",
        compiled_dispatches: compiled.num_dispatches(),
        compiled_steps: compiled.num_steps(),
        buffer_total_bytes: bp.total_bytes,
        buffer_naive_bytes: bp.naive_total,
        compiled_latency_us: compiled_us,
        eager_latency_us: eager_us,
    }
}

// -- Main benchmark test ------------------------------------------------------

/// Benchmark all 4 Kokoro segments: compiled vs. eager dispatch count + latency.
#[test]
fn bench_kokoro_segments() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let metrics = vec![
        bench_text_pipeline(&cache),
        bench_prosody(&cache),
        bench_f0_energy(&cache),
        bench_generator(&cache),
    ];
    report(&metrics);
}

// -- Dispatch count gate ------------------------------------------------------

/// Gate: total compiled dispatches across all 5 Kokoro segments must stay under
/// threshold. Catches unfused or decomposed ops that inflate GPU kernel launches.
///
/// Uses real CompiledKokoro pipeline to measure actual production dispatch counts.
/// Requires `KOKORO_WEIGHTS` env var pointing to kokoro_v1_0.safetensors.
///
/// Run:
///   KOKORO_WEIGHTS=path/to/kokoro_v1_0.safetensors \
///   cargo test -p nn-metal --test kokoro_benchmark kokoro_dispatch_count_gate -- --nocapture
///
/// Part of #2926.
#[test]
fn kokoro_dispatch_count_gate() {
    let weights_path =
        match super::kokoro_test_env::require_kokoro_weights("dispatch gate not enforced.") {
            Some(path) => path,
            None => return,
        };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Use Warn policy: benchmark tests measure dispatch counts, not audio quality.
    // Synthetic test tokens [0..7] may produce click artifacts that fail the
    // no_clicks hard bound with production weights. Part of #4262.
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("failed to load Kokoro weights")
    };

    // Synthesize one utterance to populate all segment caches.
    let input_ids =
        DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();
    let speed = 1.0;
    kokoro
        .synthesize(&input_ids, &style, speed, &cache)
        .expect("synthesis must succeed to populate segment caches");

    // Read per-segment dispatch counts.
    let summary = kokoro.dispatch_summary();
    let total = kokoro.total_dispatches();
    let metal_total = kokoro.total_metal_dispatches();

    eprintln!("\n  Dispatch count gate (Part of #2926):");
    eprintln!("    {:<20} {:>4} dispatches", "PlBert", summary.plbert);
    eprintln!(
        "    {:<20} {:>4} dispatches",
        "TextEncoder", summary.text_encoder
    );
    eprintln!("    {:<20} {:>4} dispatches", "Prosody", summary.prosody);
    eprintln!("    {:<20} {:>4} dispatches", "F0Energy", summary.f0_energy);
    eprintln!(
        "    {:<20} {:>4} dispatches",
        "Generator", summary.generator
    );
    eprintln!("    {:<20} {:>4} logical dispatches", "TOTAL", total);
    eprintln!(
        "    {:<20} {:>4} Metal kernel launches",
        "TOTAL", metal_total
    );

    // Sanity: all 5 segments must have been compiled.
    assert!(total > 0, "No dispatches — segment caches not populated");
    assert!(summary.plbert > 0, "PlBert segment not compiled");
    assert!(summary.generator > 0, "Generator segment not compiled");

    // Gate: ratchet down as fused kernels land.
    // Known baseline (production D=512): ~476 logical dispatches, ~674 Metal launches.
    // Miniaturized (D=8) baseline is ~146, but this gate uses production weights.
    const DISPATCH_THRESHOLD: usize = 500;
    assert!(
        total <= DISPATCH_THRESHOLD,
        "Kokoro total dispatches {total} exceeds gate {DISPATCH_THRESHOLD}. \
         Check for unfused kernels or decomposed ops."
    );
}

// -- RTF (real-time factor) benchmark gate ------------------------------------

/// Gate: Kokoro RTF must stay below threshold to prevent performance regressions.
///
/// RTF = wall_clock_synthesis_time / audio_duration.
/// RTF < 1.0 means faster than real-time. Lower is better.
///
/// Requires `KOKORO_WEIGHTS` env var pointing to kokoro_v1_0.safetensors.
/// Requires Metal GPU hardware (macOS only).
///
/// Current baseline: RTF ~0.159 on M4 Max (2026-03-19).
/// Gate threshold: 0.200 (25% headroom above baseline, catches regressions).
/// Ultimate target: 0.03 (within 25% of PyTorch MPS 0.024).
/// Tighten threshold as optimizations land (SourceModule GPU migration,
/// FusedAdainResBlock NativeOp, dispatch reduction).
///
/// Run:
///   KOKORO_WEIGHTS=path/to/kokoro_v1_0.safetensors \
///   cargo test -p nn-metal --test kokoro_benchmark kokoro_rtf_gate -- --nocapture
///
/// Part of #2925.
#[test]
fn kokoro_rtf_gate() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "RTF gate not enforced. Set KOKORO_WEIGHTS to enable this regression gate.",
    ) {
        Some(path) => path,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Use Warn policy: RTF benchmark measures timing, not audio quality.
    // Synthetic test tokens [0..7] may produce click artifacts that fail the
    // no_clicks hard bound with production weights. Part of #4262.
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("failed to load Kokoro weights")
    };

    // Standard test utterance: 8 phoneme tokens at speed 1.0.
    let input_ids =
        DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();
    let speed = 1.0;

    // Warmup: populate segment caches + JIT compile.
    const RTF_WARMUP: usize = 3;
    const RTF_BENCH_ITERS: usize = 10;
    for _ in 0..RTF_WARMUP {
        kokoro
            .synthesize(&input_ids, &style, speed, &cache)
            .expect("warmup synthesis failed");
    }

    // Timed runs: measure wall-clock, audio duration, and per-stage breakdown.
    let mut total_wall_secs = 0.0_f64;
    let mut total_audio_secs = 0.0_f64;
    const SAMPLE_RATE: f64 = 24_000.0;

    // Per-stage accumulators.
    let mut stage_encode = Duration::ZERO;
    let mut stage_prosody = Duration::ZERO;
    let mut stage_regulate = Duration::ZERO;
    let mut stage_f0_energy = Duration::ZERO;
    let mut stage_harmonic = Duration::ZERO;
    let mut stage_generate = Duration::ZERO;
    let mut stage_istft = Duration::ZERO;
    let mut stage_verify = Duration::ZERO;

    for _ in 0..RTF_BENCH_ITERS {
        let (audio, _cert, timing) = kokoro
            .synthesize_with_timing(&input_ids, &style, speed, &cache)
            .expect("benchmark synthesis failed");

        // Audio shape: [1, 1, T_audio] or [1, T_audio].
        let num_samples = *audio.dims().last().expect("audio must have at least 1 dim");
        assert!(num_samples > 0, "Synthesis produced 0 audio samples");
        let audio_duration_secs = num_samples as f64 / SAMPLE_RATE;

        total_wall_secs += timing.total.as_secs_f64();
        total_audio_secs += audio_duration_secs;

        stage_encode += timing.encode;
        stage_prosody += timing.prosody;
        stage_regulate += timing.regulate;
        stage_f0_energy += timing.f0_energy;
        stage_harmonic += timing.harmonic;
        stage_generate += timing.generate;
        stage_istft += timing.istft;
        stage_verify += timing.verify;
    }

    let avg_wall_secs = total_wall_secs / RTF_BENCH_ITERS as f64;
    let avg_audio_secs = total_audio_secs / RTF_BENCH_ITERS as f64;
    let rtf = avg_wall_secs / avg_audio_secs;
    let total_stage_ms = total_wall_secs * 1000.0;

    eprintln!("\n  RTF gate (Part of #2925):");
    eprintln!("    Warmup iters:  {RTF_WARMUP}");
    eprintln!("    Bench iters:   {RTF_BENCH_ITERS}");
    eprintln!("    Avg wall time: {:.3} ms", avg_wall_secs * 1000.0);
    eprintln!("    Avg audio dur: {:.3} ms", avg_audio_secs * 1000.0);
    eprintln!("    RTF:           {rtf:.4}");

    // Per-stage breakdown: identifies which pipeline stage dominates latency.
    let n = RTF_BENCH_ITERS as f64;
    let stages: [(&str, Duration); 8] = [
        ("encode", stage_encode),
        ("prosody", stage_prosody),
        ("regulate", stage_regulate),
        ("f0_energy", stage_f0_energy),
        ("harmonic", stage_harmonic),
        ("generate", stage_generate),
        ("istft", stage_istft),
        ("verify", stage_verify),
    ];
    eprintln!("\n    RTF stage breakdown (avg over {RTF_BENCH_ITERS} iters):");
    for (name, dur) in &stages {
        let avg_ms = dur.as_secs_f64() * 1000.0 / n;
        let pct = dur.as_secs_f64() * 1000.0 / total_stage_ms * 100.0;
        eprintln!("      {name:<14} {avg_ms:>8.2} ms  ({pct:>5.1}%)");
    }
    eprintln!("      -------------------------");
    eprintln!(
        "      {:<14} {:>8.2} ms  (RTF: {rtf:.4})",
        "total",
        avg_wall_secs * 1000.0
    );

    // Gate: RTF must be below threshold.
    // Baseline RTF ~0.159 on M4 Max (2026-03-19). Gate = 0.200 (25% headroom).
    // PyTorch MPS target: 0.03. Tighten as optimizations land.
    const RTF_THRESHOLD: f64 = 0.200;
    assert!(
        rtf < RTF_THRESHOLD,
        "Kokoro RTF {rtf:.4} exceeds gate {RTF_THRESHOLD} \
         (baseline ~0.159, PyTorch MPS = 0.024). \
         Performance regression detected."
    );

    eprintln!("    PASS: RTF {rtf:.4} < {RTF_THRESHOLD} gate");
}

// -- F16 vs F32 RTF comparison ------------------------------------------------

/// RTF comparison: F16 mixed-precision vs F32 baseline.
///
/// Runs both pipelines on identical input and reports:
/// - F32 RTF, F16 RTF, and the speedup ratio
/// - Per-stage breakdown for both pipelines
///
/// This is the primary measurement for #2981 acceptance criteria:
/// "RTF measurement showing improvement over 0.159 baseline."
///
/// Requires `KOKORO_WEIGHTS` env var.
///
/// Run:
///   KOKORO_WEIGHTS=path/to/kokoro_v1_0.safetensors \
///   cargo test -p nn-metal --test kokoro_all kokoro_f16_rtf_comparison -- --nocapture
///
/// Part of #2981.
#[test]
fn kokoro_f16_rtf_comparison() {
    let weights_path =
        match super::kokoro_test_env::require_kokoro_weights("F16 RTF comparison not run.") {
            Some(path) => path,
            None => return,
        };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let input_ids =
        DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();
    let speed = 1.0;

    const F16_WARMUP: usize = 3;
    const F16_BENCH_ITERS: usize = 10;
    const SAMPLE_RATE: f64 = 24_000.0;

    // Use Warn policy: benchmark tests measure timing, not audio quality.
    // Synthetic test tokens [0..7] may produce click artifacts that fail the
    // no_clicks hard bound with production weights. Part of #4262.
    let mut hb_f16 = nn_tts_verify::HardBoundsConfig::default();
    hb_f16.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // --- F32 baseline ---
    // SAFETY: safetensors file not modified while alive.
    let mut f32_kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(
            &weights_path,
            hb_f16,
        )
        .expect("load CompiledKokoro F32")
    };

    for _ in 0..F16_WARMUP {
        f32_kokoro
            .synthesize(&input_ids, &style, speed, &cache)
            .expect("F32 warmup failed");
    }

    let mut f32_wall = 0.0_f64;
    let mut f32_audio = 0.0_f64;
    let mut f32_stages = [Duration::ZERO; 8];

    for _ in 0..F16_BENCH_ITERS {
        let (audio, _cert, timing) = f32_kokoro
            .synthesize_with_timing(&input_ids, &style, speed, &cache)
            .expect("F32 benchmark failed");
        let n = *audio.dims().last().unwrap() as f64;
        f32_wall += timing.total.as_secs_f64();
        f32_audio += n / SAMPLE_RATE;
        f32_stages[0] += timing.encode;
        f32_stages[1] += timing.prosody;
        f32_stages[2] += timing.regulate;
        f32_stages[3] += timing.f0_energy;
        f32_stages[4] += timing.harmonic;
        f32_stages[5] += timing.generate;
        f32_stages[6] += timing.istft;
        f32_stages[7] += timing.verify;
    }

    let f32_rtf = (f32_wall / F16_BENCH_ITERS as f64) / (f32_audio / F16_BENCH_ITERS as f64);

    // --- F16 mixed-precision ---
    let mut hb_f16b = nn_tts_verify::HardBoundsConfig::default();
    hb_f16b.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;
    // SAFETY: safetensors file not modified while alive.
    let mut f16_kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb_f16b)
            .expect("load CompiledKokoro F16")
    }
    .with_autocast();

    for _ in 0..F16_WARMUP {
        f16_kokoro
            .synthesize(&input_ids, &style, speed, &cache)
            .expect("F16 warmup failed");
    }

    let mut f16_wall = 0.0_f64;
    let mut f16_audio = 0.0_f64;
    let mut f16_stages = [Duration::ZERO; 8];

    for _ in 0..F16_BENCH_ITERS {
        let (audio, _cert, timing) = f16_kokoro
            .synthesize_with_timing(&input_ids, &style, speed, &cache)
            .expect("F16 benchmark failed");
        let n = *audio.dims().last().unwrap() as f64;
        f16_wall += timing.total.as_secs_f64();
        f16_audio += n / SAMPLE_RATE;
        f16_stages[0] += timing.encode;
        f16_stages[1] += timing.prosody;
        f16_stages[2] += timing.regulate;
        f16_stages[3] += timing.f0_energy;
        f16_stages[4] += timing.harmonic;
        f16_stages[5] += timing.generate;
        f16_stages[6] += timing.istft;
        f16_stages[7] += timing.verify;
    }

    let f16_rtf = (f16_wall / F16_BENCH_ITERS as f64) / (f16_audio / F16_BENCH_ITERS as f64);
    let speedup = f32_rtf / f16_rtf;

    // --- Report ---
    let stage_names = [
        "encode",
        "prosody",
        "regulate",
        "f0_energy",
        "harmonic",
        "generate",
        "istft",
        "verify",
    ];
    let n = F16_BENCH_ITERS as f64;

    eprintln!("\n  F16 vs F32 RTF comparison (Part of #2981):");
    eprintln!("    Warmup: {F16_WARMUP}, Bench: {F16_BENCH_ITERS} iters\n");
    eprintln!(
        "    {:<14} {:>10} {:>10} {:>10}",
        "Stage", "F32 (ms)", "F16 (ms)", "Speedup"
    );
    eprintln!("    {}", "-".repeat(48));

    for (i, name) in stage_names.iter().enumerate() {
        let f32_ms = f32_stages[i].as_secs_f64() * 1000.0 / n;
        let f16_ms = f16_stages[i].as_secs_f64() * 1000.0 / n;
        let stage_speedup = f32_ms / f16_ms.max(0.001);
        eprintln!(
            "    {name:<14} {f32_ms:>9.2} {f16_ms:>9.2} {stage_speedup:>9.2}x"
        );
    }

    let f32_avg_ms = f32_wall * 1000.0 / n;
    let f16_avg_ms = f16_wall * 1000.0 / n;
    eprintln!("    {}", "-".repeat(48));
    eprintln!(
        "    {:<14} {:>9.2} {:>9.2} {:>9.2}x",
        "TOTAL", f32_avg_ms, f16_avg_ms, speedup
    );
    eprintln!("\n    F32 RTF: {f32_rtf:.4}");
    eprintln!("    F16 RTF: {f16_rtf:.4}");
    eprintln!("    Speedup: {speedup:.2}x");
    eprintln!(
        "    Savings: {:.1} ms per synthesis",
        f32_avg_ms - f16_avg_ms
    );

    // F16 should be faster than F32 due to 2x ALU throughput.
    // Assert improvement exists (even if small on first runs).
    assert!(
        f16_rtf < f32_rtf,
        "F16 RTF ({f16_rtf:.4}) not faster than F32 RTF ({f32_rtf:.4}). \
         Mixed-precision pipeline should be faster due to 2x Metal ALU throughput."
    );

    eprintln!("\n    PASS: F16 RTF {f16_rtf:.4} < F32 RTF {f32_rtf:.4} ({speedup:.2}x speedup)");
}

// -- Autocast vs F32 RTF comparison -------------------------------------------

/// RTF comparison: per-op autocast vs F32 baseline.
///
/// Unlike `with_mixed_precision()` (which stores intermediates as F16 and can
/// NaN), `with_autocast()` keeps all intermediate buffers F32 and only uses
/// F16 for compute-dominant kernels (Linear, Conv, FlashAttention).
///
/// Requires `KOKORO_WEIGHTS` env var.
///
/// Run:
///   KOKORO_WEIGHTS=path/to/kokoro_v1_0.safetensors \
///   cargo test -p nn-metal --test kokoro_all kokoro_autocast_rtf_comparison -- --nocapture
///
/// Part of #2981, #3085.
#[test]
fn kokoro_autocast_rtf_comparison() {
    let weights_path =
        match super::kokoro_test_env::require_kokoro_weights("autocast RTF comparison not run.") {
            Some(path) => path,
            None => return,
        };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let input_ids =
        DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();
    let speed = 1.0;

    const AC_WARMUP: usize = 3;
    const AC_BENCH_ITERS: usize = 10;
    const SAMPLE_RATE: f64 = 24_000.0;

    // Use Warn policy: autocast benchmark measures timing, not audio quality.
    // Synthetic test tokens [0..7] may produce click artifacts that fail the
    // no_clicks hard bound with production weights. Part of #4262.

    // --- F32 baseline ---
    let mut hb_ac_f32 = nn_tts_verify::HardBoundsConfig::default();
    hb_ac_f32.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;
    let mut f32_kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb_ac_f32)
            .expect("load CompiledKokoro F32")
    };

    for _ in 0..AC_WARMUP {
        f32_kokoro
            .synthesize(&input_ids, &style, speed, &cache)
            .expect("F32 warmup failed");
    }

    let mut f32_wall = 0.0_f64;
    let mut f32_audio = 0.0_f64;
    let mut f32_stages = [Duration::ZERO; 8];

    for _ in 0..AC_BENCH_ITERS {
        let (audio, _cert, timing) = f32_kokoro
            .synthesize_with_timing(&input_ids, &style, speed, &cache)
            .expect("F32 benchmark failed");
        let n = *audio.dims().last().unwrap() as f64;
        f32_wall += timing.total.as_secs_f64();
        f32_audio += n / SAMPLE_RATE;
        f32_stages[0] += timing.encode;
        f32_stages[1] += timing.prosody;
        f32_stages[2] += timing.regulate;
        f32_stages[3] += timing.f0_energy;
        f32_stages[4] += timing.harmonic;
        f32_stages[5] += timing.generate;
        f32_stages[6] += timing.istft;
        f32_stages[7] += timing.verify;
    }

    let f32_rtf = (f32_wall / AC_BENCH_ITERS as f64) / (f32_audio / AC_BENCH_ITERS as f64);

    // --- Autocast (per-op F16) ---
    let mut hb_ac = nn_tts_verify::HardBoundsConfig::default();
    hb_ac.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;
    let mut ac_kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb_ac)
            .expect("load CompiledKokoro autocast")
    }
    .with_autocast();

    for _ in 0..AC_WARMUP {
        ac_kokoro
            .synthesize(&input_ids, &style, speed, &cache)
            .expect("autocast warmup failed");
    }

    let mut ac_wall = 0.0_f64;
    let mut ac_audio = 0.0_f64;
    let mut ac_stages = [Duration::ZERO; 8];

    for _ in 0..AC_BENCH_ITERS {
        let (audio, _cert, timing) = ac_kokoro
            .synthesize_with_timing(&input_ids, &style, speed, &cache)
            .expect("autocast benchmark failed");
        let n = *audio.dims().last().unwrap() as f64;
        ac_wall += timing.total.as_secs_f64();
        ac_audio += n / SAMPLE_RATE;
        ac_stages[0] += timing.encode;
        ac_stages[1] += timing.prosody;
        ac_stages[2] += timing.regulate;
        ac_stages[3] += timing.f0_energy;
        ac_stages[4] += timing.harmonic;
        ac_stages[5] += timing.generate;
        ac_stages[6] += timing.istft;
        ac_stages[7] += timing.verify;
    }

    let ac_rtf = (ac_wall / AC_BENCH_ITERS as f64) / (ac_audio / AC_BENCH_ITERS as f64);
    let speedup = f32_rtf / ac_rtf;

    // --- Report ---
    let stage_names = [
        "encode",
        "prosody",
        "regulate",
        "f0_energy",
        "harmonic",
        "generate",
        "istft",
        "verify",
    ];
    let n = AC_BENCH_ITERS as f64;

    eprintln!("\n  Autocast vs F32 RTF comparison (Part of #2981, #3085):");
    eprintln!("    Warmup: {AC_WARMUP}, Bench: {AC_BENCH_ITERS} iters\n");
    eprintln!(
        "    {:<14} {:>10} {:>10} {:>10}",
        "Stage", "F32 (ms)", "AC (ms)", "Speedup"
    );
    eprintln!("    {}", "-".repeat(48));

    for (i, name) in stage_names.iter().enumerate() {
        let f32_ms = f32_stages[i].as_secs_f64() * 1000.0 / n;
        let ac_ms = ac_stages[i].as_secs_f64() * 1000.0 / n;
        let stage_speedup = f32_ms / ac_ms.max(0.001);
        eprintln!(
            "    {name:<14} {f32_ms:>9.2} {ac_ms:>9.2} {stage_speedup:>9.2}x"
        );
    }

    let f32_avg_ms = f32_wall * 1000.0 / n;
    let ac_avg_ms = ac_wall * 1000.0 / n;
    eprintln!("    {}", "-".repeat(48));
    eprintln!(
        "    {:<14} {:>9.2} {:>9.2} {:>9.2}x",
        "TOTAL", f32_avg_ms, ac_avg_ms, speedup
    );
    eprintln!("\n    F32 RTF:      {f32_rtf:.4}");
    eprintln!("    Autocast RTF: {ac_rtf:.4}");
    eprintln!("    Speedup:      {speedup:.2}x");
    eprintln!(
        "    Savings:      {:.1} ms per synthesis",
        f32_avg_ms - ac_avg_ms
    );
}
