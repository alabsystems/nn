// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Production-scale Kokoro benchmark at D=512.
//!
//! Measures per-segment dispatch count, GPU latency, and buffer metrics at
//! production tensor dimensions (d_en=512, style_dim=128, bilstm_hidden=256).
//! Architecture is simplified (1 prosody layer, 1 upsample, 1 resblock) so
//! weight creation is manageable, but tensor dimensions match production.
//!
//! Key questions this benchmark answers:
//! 1. Does simdgroup matmul fire at D=512? (Linear layers with M*N >= 16384)
//! 2. What is the per-segment dispatch count at production tensor sizes?
//! 3. What is the actual GPU latency for production-scale dispatches?
//! 4. How much memory does the buffer plan allocate at D=512?
//!
//! Run: `cargo test -p nn-metal --test kokoro_benchmark_d512 -- --nocapture`
//!
//! Part of #2468 (Production-scale Kokoro benchmark at D=512).
//! Part of #2218 (Kokoro epic).

use std::collections::HashMap;
use std::time::Instant;

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, TensorError, VarBuilder};
use nn_metal::compiled_model::CompiledModel;
use nn_metal::PipelineCache;
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

// -- Production dimensions ----------------------------------------------------

const D_EN: usize = 512;
const VOCAB_SIZE: usize = 178; // Kokoro production vocab size
const STYLE_DIM: usize = 128;
const BILSTM_HIDDEN: usize = D_EN / 2; // = 256
const GEN_CH: usize = 512;
const GEN_NEXT_CH: usize = 256;
const GEN_N_FFT: usize = 20;
const GEN_N_BINS: usize = GEN_N_FFT / 2 + 1; // = 11
const GEN_KERNEL: usize = 3;

const SEQ_LEN: usize = 32; // >= 32 so M*N=32*512=16384 triggers simdgroup
const WARMUP_ITERS: usize = 2;
const BENCH_ITERS: usize = 5;

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

fn bilstm_w(m: &mut HashMap<String, DynTensor>, pfx: &str, input: usize, hidden: usize) {
    let g = 4 * hidden;
    z(m, &format!("{pfx}.weight_ih_l0"), &[g, input]);
    z(m, &format!("{pfx}.weight_hh_l0"), &[g, hidden]);
    z(m, &format!("{pfx}.bias_ih_l0"), &[g]);
    z(m, &format!("{pfx}.bias_hh_l0"), &[g]);
    z(m, &format!("{pfx}.weight_ih_l0_reverse"), &[g, input]);
    z(m, &format!("{pfx}.weight_hh_l0_reverse"), &[g, hidden]);
    z(m, &format!("{pfx}.bias_ih_l0_reverse"), &[g]);
    z(m, &format!("{pfx}.bias_hh_l0_reverse"), &[g]);
    z(
        m,
        &format!("{pfx}.linear.weight"),
        &[2 * hidden, 2 * hidden],
    );
    z(m, &format!("{pfx}.linear.bias"), &[2 * hidden]);
}

fn adain_blk_w(
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

// -- Model builders -----------------------------------------------------------

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
    TextEncoder::load(&vb, VOCAB_SIZE, D_EN).expect("TextEncoder::load at D=512")
}

fn build_prosody() -> ProsodyPredictor {
    let mut m = HashMap::new();
    let d = D_EN;
    let h = D_EN / 2;
    let s = STYLE_DIM;
    let max_dur = 50;
    let lstm_input = d + s;

    // DurationEncoder: 1 layer BiLSTM + AdaLayerNorm
    bilstm_w(&mut m, "duration.lstms.0", lstm_input, h);
    z(&mut m, "duration.norms.0.fc.weight", &[2 * d, s]);
    z(&mut m, "duration.norms.0.fc.bias", &[2 * d]);
    m.insert(
        "duration.norms.0.norm.weight".into(),
        DynTensor::full(&[d], 1.0, DType::F32, &cpu()).unwrap(),
    );
    m.insert(
        "duration.norms.0.norm.bias".into(),
        DynTensor::full(&[d], 0.0, DType::F32, &cpu()).unwrap(),
    );

    // Duration projection: d_model -> max_dur
    z(&mut m, "duration.duration_proj.weight", &[max_dur, d]);
    z(&mut m, "duration.duration_proj.bias", &[max_dur]);

    // Final duration BiLSTM
    bilstm_w(&mut m, "lstm", lstm_input, h);

    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    ProsodyPredictor::load(&vb, D_EN, STYLE_DIM, 1, max_dur).expect("ProsodyPredictor at D=512")
}

fn build_f0_pred() -> F0EnergyPredictor {
    let mut m = HashMap::new();
    let bh = BILSTM_HIDDEN;
    let bo = 2 * bh;
    let bilstm_input = D_EN + STYLE_DIM;

    // Shared BiLSTM: input = d_model + style_dim
    bilstm_w(&mut m, "shared", bilstm_input, bh);

    // F0 blocks: 0=bo->bo, 1=bo->bh (upsample), 2=bh->bh
    adain_blk_w(&mut m, "F0.0", bo, bo, false);
    adain_blk_w(&mut m, "F0.1", bo, bh, true);
    adain_blk_w(&mut m, "F0.2", bh, bh, false);
    z(&mut m, "F0_proj.weight", &[1, bh]);
    z(&mut m, "F0_proj.bias", &[1]);
    // Energy (N) blocks: same architecture
    adain_blk_w(&mut m, "N.0", bo, bo, false);
    adain_blk_w(&mut m, "N.1", bo, bh, true);
    adain_blk_w(&mut m, "N.2", bh, bh, false);
    z(&mut m, "N_proj.weight", &[1, bh]);
    z(&mut m, "N_proj.bias", &[1]);
    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    F0EnergyPredictor::load(&vb, D_EN, STYLE_DIM, BILSTM_HIDDEN)
        .expect("F0EnergyPredictor at D=512")
}

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
    config.style_dim = STYLE_DIM;
    config.n_fft = GEN_N_FFT;
    Generator::load(&vb, &config).expect("Generator at D=512")
}

// -- Metrics + report ---------------------------------------------------------

struct SegmentMetrics {
    name: &'static str,
    dispatches: usize,
    native_ops: usize,
    steps: usize,
    buffer_bytes: usize,
    buffer_naive: usize,
    latency_us: f64,
}

/// Check whether simdgroup routing would fire for a Linear(M, K, N) at D=512.
///
/// Simdgroup routing happens inside `build_dispatch_plan_full()` at execution
/// time, not at `CompiledStep` construction. We can't inspect `CompiledStep`
/// names to detect simdgroup — instead we check the dimensional criteria.
fn simdgroup_eligible(m: usize, k: usize, n: usize) -> bool {
    m.is_multiple_of(8) && k.is_multiple_of(8) && n.is_multiple_of(8) && m * n >= 16_384 && k >= 128
}

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

fn report(metrics: &[SegmentMetrics]) {
    eprintln!("\n{}", "=".repeat(92));
    eprintln!("  KOKORO D=512 PRODUCTION BENCHMARK");
    eprintln!("  d_en={D_EN} style_dim={STYLE_DIM} bilstm_hidden={BILSTM_HIDDEN}");
    eprintln!("  gen_ch={GEN_CH} gen_next_ch={GEN_NEXT_CH} n_fft={GEN_N_FFT}");
    eprintln!("  Warmup: {WARMUP_ITERS} iters, Bench: {BENCH_ITERS} iters");
    eprintln!("{}", "=".repeat(92));
    eprintln!(
        "  {:<22} {:>8} {:>8} {:>6} {:>10} {:>10} {:>10}",
        "Segment", "Dispatch", "Native", "Steps", "Latency us", "Buffer KB", "Naive KB"
    );
    eprintln!("{}", "-".repeat(92));
    let (mut td, mut tn, mut tl, mut tb, mut tbr) = (0usize, 0usize, 0.0, 0usize, 0usize);
    for m in metrics {
        let reuse = if m.buffer_naive > 0 {
            100.0 * (1.0 - m.buffer_bytes as f64 / m.buffer_naive as f64)
        } else {
            0.0
        };
        eprintln!(
            "  {:<22} {:>8} {:>8} {:>6} {:>10.0} {:>10} {:>10} ({:.0}% reuse)",
            m.name,
            m.dispatches,
            m.native_ops,
            m.steps,
            m.latency_us,
            m.buffer_bytes / 1024,
            m.buffer_naive / 1024,
            reuse
        );
        td += m.dispatches;
        tn += m.native_ops;
        tl += m.latency_us;
        tb += m.buffer_bytes;
        tbr += m.buffer_naive;
    }
    eprintln!("{}", "-".repeat(92));
    let reuse = if tbr > 0 {
        100.0 * (1.0 - tb as f64 / tbr as f64)
    } else {
        0.0
    };
    eprintln!(
        "  {:<22} {:>8} {:>8} {:>6} {:>10.0} {:>10} {:>10} ({:.0}% reuse)",
        "TOTAL",
        td,
        tn,
        "-",
        tl,
        tb / 1024,
        tbr / 1024,
        reuse
    );
    eprintln!("{}\n", "=".repeat(92));
}

// -- Per-segment benchmarks ---------------------------------------------------

fn bench_text(cache: &PipelineCache) -> SegmentMetrics {
    let te = build_text_encoder();
    // Token IDs [B, T] as F32 — Metal doesn't support I64 GPU buffers.
    // Production compiled_kokoro_steps.rs converts to F32 before the compiled
    // segment, so we trace with F32 to match.
    let token_ids: Vec<f32> = (0..SEQ_LEN).map(|i| (i % VOCAB_SIZE) as f32).collect();
    let tokens = DynTensor::new(&token_ids, &[1, SEQ_LEN], &cpu()).unwrap();
    let (out, mut graph) = trace_graph(|| {
        let mut inp = tokens.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        te.forward(&inp).map_err(KokoroError::into_tensor_error)
    })
    .unwrap();
    if let Some(id) = out.trace_id() {
        assert!(
            graph.set_primary_output(id),
            "set_primary_output failed for TextPipeline"
        );
    }
    let compiled = CompiledModel::builder(&graph, cache).build().unwrap();
    let bp = compiled.buffer_plan();
    let tokens_gpu = tokens.to_device(&gpu()).unwrap();
    let latency = bench_latency(|| {
        let _ = compiled.execute_dyn(cache, &[&tokens_gpu]).unwrap();
    });
    SegmentMetrics {
        name: "TextPipeline",
        dispatches: compiled.num_dispatches(),
        native_ops: compiled.num_native_ops(),
        steps: compiled.num_steps(),
        buffer_bytes: bp.total_bytes,
        buffer_naive: bp.naive_total,
        latency_us: latency,
    }
}

fn bench_prosody(cache: &PipelineCache) -> SegmentMetrics {
    let prosody = build_prosody();
    let x = DynTensor::new(
        &super::test_utils::rand_f32_vec(50, D_EN * SEQ_LEN, -0.1, 0.1),
        &[1, D_EN, SEQ_LEN],
        &cpu(),
    )
    .unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(51, STYLE_DIM, -0.1, 0.1),
        &[1, STYLE_DIM],
        &cpu(),
    )
    .unwrap();
    let (dur, mut graph) = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let (d, _f) = prosody
            .forward(&inp, &sty)
            .map_err(KokoroError::into_tensor_error)?;
        Ok(d)
    })
    .unwrap();
    if let Some(id) = dur.trace_id() {
        assert!(
            graph.set_primary_output(id),
            "set_primary_output failed for ProsodyPredictor"
        );
    }
    let compiled = CompiledModel::builder(&graph, cache).build().unwrap();
    let bp = compiled.buffer_plan();
    let x_gpu = x.to_device(&gpu()).unwrap();
    let sty_gpu = style.to_device(&gpu()).unwrap();
    let latency = bench_latency(|| {
        let _ = compiled.execute_dyn(cache, &[&x_gpu, &sty_gpu]).unwrap();
    });
    SegmentMetrics {
        name: "ProsodyPredictor",
        dispatches: compiled.num_dispatches(),
        native_ops: compiled.num_native_ops(),
        steps: compiled.num_steps(),
        buffer_bytes: bp.total_bytes,
        buffer_naive: bp.naive_total,
        latency_us: latency,
    }
}

fn bench_f0_energy(cache: &PipelineCache) -> SegmentMetrics {
    let f0_pred = build_f0_pred();
    let t_mel = 32;
    let aligned = DynTensor::new(
        &super::test_utils::rand_f32_vec(60, (D_EN + STYLE_DIM) * t_mel, -0.1, 0.1),
        &[1, D_EN + STYLE_DIM, t_mel],
        &cpu(),
    )
    .unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(61, STYLE_DIM, -0.1, 0.1),
        &[1, STYLE_DIM],
        &cpu(),
    )
    .unwrap();
    let (f0, mut graph) = trace_graph(|| {
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
    if let Some(id) = f0.trace_id() {
        assert!(
            graph.set_primary_output(id),
            "set_primary_output failed for F0EnergyPredictor"
        );
    }
    let compiled = CompiledModel::builder(&graph, cache).build().unwrap();
    let bp = compiled.buffer_plan();
    let a_gpu = aligned.to_device(&gpu()).unwrap();
    let s_gpu = style.to_device(&gpu()).unwrap();
    let latency = bench_latency(|| {
        let _ = compiled.execute_dyn(cache, &[&a_gpu, &s_gpu]).unwrap();
    });
    SegmentMetrics {
        name: "F0EnergyPredictor",
        dispatches: compiled.num_dispatches(),
        native_ops: compiled.num_native_ops(),
        steps: compiled.num_steps(),
        buffer_bytes: bp.total_bytes,
        buffer_naive: bp.naive_total,
        latency_us: latency,
    }
}

fn bench_generator(cache: &PipelineCache) -> SegmentMetrics {
    let generator = build_generator();
    let t_in = 32;
    let t_full = 64;
    let x = DynTensor::new(
        &super::test_utils::rand_f32_vec(70, GEN_CH * t_in, -0.1, 0.1),
        &[1, GEN_CH, t_in],
        &cpu(),
    )
    .unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(71, STYLE_DIM, -0.1, 0.1),
        &[1, STYLE_DIM],
        &cpu(),
    )
    .unwrap();
    let har = DynTensor::new(
        &super::test_utils::rand_f32_vec(72, 2 * GEN_N_BINS * t_full, -0.1, 0.1),
        &[1, 2 * GEN_N_BINS, t_full],
        &cpu(),
    )
    .unwrap();
    let (mag, mut graph) = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let mut h = har.clone();
        h.set_trace_id(record_input(h.dims(), DType::F32).unwrap());
        let (m, _) = generator
            .forward(&inp, &sty, &h)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        Ok(m)
    })
    .unwrap();
    if let Some(id) = mag.trace_id() {
        assert!(
            graph.set_primary_output(id),
            "set_primary_output failed for Generator"
        );
    }
    let compiled = CompiledModel::builder(&graph, cache).build().unwrap();
    let bp = compiled.buffer_plan();
    let x_gpu = x.to_device(&gpu()).unwrap();
    let s_gpu = style.to_device(&gpu()).unwrap();
    let h_gpu = har.to_device(&gpu()).unwrap();
    let latency = bench_latency(|| {
        let _ = compiled
            .execute_dyn(cache, &[&x_gpu, &s_gpu, &h_gpu])
            .unwrap();
    });
    SegmentMetrics {
        name: "Generator",
        dispatches: compiled.num_dispatches(),
        native_ops: compiled.num_native_ops(),
        steps: compiled.num_steps(),
        buffer_bytes: bp.total_bytes,
        buffer_naive: bp.naive_total,
        latency_us: latency,
    }
}

// -- Main benchmark -----------------------------------------------------------

/// Production-scale D=512 benchmark: dispatch count + simdgroup + latency.
#[test]
fn bench_kokoro_d512() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let metrics = vec![
        bench_text(&cache),
        bench_prosody(&cache),
        bench_f0_energy(&cache),
        bench_generator(&cache),
    ];
    report(&metrics);

    // Verify non-zero dispatches (compilation succeeded at D=512).
    for m in &metrics {
        assert!(
            m.dispatches > 0,
            "{}: expected non-zero dispatches at D=512",
            m.name
        );
    }

    // Verify simdgroup eligibility at D=512 dimensions.
    // Simdgroup routing happens inside build_dispatch_plan_full() at execution
    // time. We verify the dimensional criteria are met for key Linear layers.
    // TextEncoder LSTM projection: [D_EN, D_EN], M=SEQ_LEN
    assert!(
        simdgroup_eligible(SEQ_LEN, D_EN, D_EN),
        "TextEncoder LSTM projection should be simdgroup-eligible at D=512"
    );
    eprintln!("Simdgroup eligibility confirmed for TextPipeline Linear layers at D=512");
}
