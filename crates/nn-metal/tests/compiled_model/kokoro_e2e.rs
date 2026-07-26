// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CompiledModel Kokoro e2e tests — trace + compile + execute on Metal GPU.
//!
//! Tests the largest compilable Kokoro subsections through the full
//! CompiledModel pipeline: trace_graph() → builder().build() → execute_dyn().
//!
//! The full KokoroModel cannot be traced as one graph because `length_regulate`
//! does CPU readback mid-forward (dynamic repeat based on predicted durations).
//! Instead we trace the two largest subsections independently:
//!
//! 1. **Generator** (vocoder): Conv1d, ConvTranspose1d, Snake, AdaIN,
//!    InstanceNorm, LeakyRelu, Sin, Exp, Clamp — the most compute-heavy path.
//! 2. **Text pipeline**: Linear, BiLSTM, Transpose, Flip, Cat — encoder path.
//!
//! Part of #2229 (CompiledModel Kokoro e2e test).
//! Part of #2218 (Kokoro epic).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::{DType, Device, VarBuilder};
use nn_metal::compiled_model::CompiledModel;
use nn_metal::PipelineCache;
use nn_models::kokoro_decoder::Generator;
use nn_models::kokoro_error::KokoroError;
use nn_models::kokoro_f0::F0EnergyPredictor;
use nn_models::kokoro_tts::{ProsodyPredictor, TextEncoder};
use nn_models::{KokoroConfig, PlBert, PlbertConfig};

fn cpu() -> Device {
    Device::Cpu
}

fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

// -- Shared dimensions -------------------------------------------------------

const GEN_CH: usize = 8;
const GEN_NEXT_CH: usize = 4;
const GEN_STYLE_DIM: usize = 4;
const GEN_N_FFT: usize = 4;
const GEN_N_BINS: usize = GEN_N_FFT / 2 + 1;
const GEN_KERNEL: usize = 3;
const D_EN: usize = 8;
const VOCAB_SIZE: usize = 16;

// -- Weight construction helpers ----------------------------------------------

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

fn generator_weights() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    conv1d_w(&mut m, "conv_pre", GEN_CH, GEN_CH, 7);
    z(&mut m, "ups.0.weight", &[GEN_CH, GEN_NEXT_CH, 4]);
    z(&mut m, "ups.0.bias", &[GEN_NEXT_CH]);
    conv1d_w(&mut m, "noise_convs.0", GEN_NEXT_CH, 2 * GEN_N_BINS, 1);
    resblock_w(&mut m, "noise_res.0", GEN_NEXT_CH, 11, 3);
    resblock_w(&mut m, "resblocks.0", GEN_NEXT_CH, GEN_KERNEL, 1);
    conv1d_w(&mut m, "conv_post", 2 * GEN_N_BINS, GEN_NEXT_CH, 7);
    m
}

const STYLE_DIM: usize = 4;
const BILSTM_HIDDEN: usize = D_EN / 2;
const PROSODY_N_LAYERS: usize = 2;

fn prosody_predictor_weights() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let d = D_EN;
    let h = D_EN / 2;
    let s = STYLE_DIM;
    let bilstm_in = d + s;
    let max_dur = 50;
    // DurationEncoder: 3× BiLSTM + AdaLayerNorm blocks under "duration.*"
    for i in 0..PROSODY_N_LAYERS {
        // BiLstm under "duration.lstms.{i}": forward + reverse directions
        let bp = format!("duration.lstms.{i}");
        z(&mut m, &format!("{bp}.weight_ih_l0"), &[4 * h, bilstm_in]);
        z(&mut m, &format!("{bp}.weight_hh_l0"), &[4 * h, h]);
        z(&mut m, &format!("{bp}.bias_ih_l0"), &[4 * h]);
        z(&mut m, &format!("{bp}.bias_hh_l0"), &[4 * h]);
        z(
            &mut m,
            &format!("{bp}.weight_ih_l0_reverse"),
            &[4 * h, bilstm_in],
        );
        z(&mut m, &format!("{bp}.weight_hh_l0_reverse"), &[4 * h, h]);
        z(&mut m, &format!("{bp}.bias_ih_l0_reverse"), &[4 * h]);
        z(&mut m, &format!("{bp}.bias_hh_l0_reverse"), &[4 * h]);
        // AdaLayerNorm under "duration.norms.{i}": LayerNorm + style Linear
        let np = format!("duration.norms.{i}");
        m.insert(
            format!("{np}.norm.weight"),
            DynTensor::full(&[d], 1.0, DType::F32, &cpu()).unwrap(),
        );
        z(&mut m, &format!("{np}.norm.bias"), &[d]);
        z(&mut m, &format!("{np}.fc.weight"), &[2 * d, s]);
        z(&mut m, &format!("{np}.fc.bias"), &[2 * d]);
    }
    // Duration projection under "duration.duration_proj"
    z(&mut m, "duration.duration_proj.weight", &[max_dur, d]);
    z(&mut m, "duration.duration_proj.bias", &[max_dur]);
    // Final duration BiLSTM under "lstm"
    z(&mut m, "lstm.weight_ih_l0", &[4 * h, bilstm_in]);
    z(&mut m, "lstm.weight_hh_l0", &[4 * h, h]);
    z(&mut m, "lstm.bias_ih_l0", &[4 * h]);
    z(&mut m, "lstm.bias_hh_l0", &[4 * h]);
    z(&mut m, "lstm.weight_ih_l0_reverse", &[4 * h, bilstm_in]);
    z(&mut m, "lstm.weight_hh_l0_reverse", &[4 * h, h]);
    z(&mut m, "lstm.bias_ih_l0_reverse", &[4 * h]);
    z(&mut m, "lstm.bias_hh_l0_reverse", &[4 * h]);
    m
}

fn f0_energy_predictor_weights() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let d = D_EN;
    let bh = BILSTM_HIDDEN;
    let bo = 2 * bh; // bilstm output = 2 * hidden = D_EN
    let s = STYLE_DIM;

    // Shared BiLSTM: input = d_model + style_dim (cat(features, style))
    let bilstm_in = d + s;
    z(&mut m, "shared.forward.weight_ih_l0", &[4 * bh, bilstm_in]);
    z(&mut m, "shared.forward.weight_hh_l0", &[4 * bh, bh]);
    z(&mut m, "shared.forward.bias_ih_l0", &[4 * bh]);
    z(&mut m, "shared.forward.bias_hh_l0", &[4 * bh]);
    z(&mut m, "shared.backward.weight_ih_l0", &[4 * bh, bilstm_in]);
    z(&mut m, "shared.backward.weight_hh_l0", &[4 * bh, bh]);
    z(&mut m, "shared.backward.bias_ih_l0", &[4 * bh]);
    z(&mut m, "shared.backward.bias_hh_l0", &[4 * bh]);

    // Helper for AdainResBlk1d weights
    let adain_blk = |m: &mut HashMap<String, DynTensor>,
                     pfx: &str,
                     dim_in: usize,
                     dim_out: usize,
                     upsample: bool| {
        z(m, &format!("{pfx}.n1.fc.weight"), &[2 * dim_in, s]);
        z(m, &format!("{pfx}.n1.fc.bias"), &[2 * dim_in]);
        z(m, &format!("{pfx}.n2.fc.weight"), &[2 * dim_out, s]);
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
    };

    // F0 blocks: 0=bo→bo, 1=bo→bh (upsample), 2=bh→bh
    adain_blk(&mut m, "F0.0", bo, bo, false);
    adain_blk(&mut m, "F0.1", bo, bh, true);
    adain_blk(&mut m, "F0.2", bh, bh, false);
    z(&mut m, "F0_proj.weight", &[1, bh]);
    z(&mut m, "F0_proj.bias", &[1]);

    // Energy (N) blocks: same architecture
    adain_blk(&mut m, "N.0", bo, bo, false);
    adain_blk(&mut m, "N.1", bo, bh, true);
    adain_blk(&mut m, "N.2", bh, bh, false);
    z(&mut m, "N_proj.weight", &[1, bh]);
    z(&mut m, "N_proj.bias", &[1]);

    m
}

fn text_encoder_weights() -> HashMap<String, DynTensor> {
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
    m
}

// -- PlBert dimensions --------------------------------------------------------

const PLBERT_EMB_DIM: usize = 4;
const PLBERT_HIDDEN: usize = 8; // same as D_EN for simplicity
const PLBERT_INTERMEDIATE: usize = 16;
const PLBERT_NUM_HEADS: usize = 2;
const PLBERT_MAX_POS: usize = 16;
const PLBERT_NUM_LAYERS: usize = 2;

fn plbert_weights() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let ed = PLBERT_EMB_DIM;
    let h = PLBERT_HIDDEN;
    let im = PLBERT_INTERMEDIATE;

    // Embeddings
    z(
        &mut m,
        "embeddings.word_embeddings.weight",
        &[VOCAB_SIZE, ed],
    );
    z(
        &mut m,
        "embeddings.position_embeddings.weight",
        &[PLBERT_MAX_POS, ed],
    );
    z(&mut m, "embeddings.token_type_embeddings.weight", &[2, ed]);
    m.insert(
        "embeddings.LayerNorm.weight".into(),
        DynTensor::full(&[ed], 1.0, DType::F32, &cpu()).unwrap(),
    );
    z(&mut m, "embeddings.LayerNorm.bias", &[ed]);

    // Factorized projection: embedding_dim -> hidden_size
    z(
        &mut m,
        "encoder.embedding_hidden_mapping_in.weight",
        &[h, ed],
    );
    z(&mut m, "encoder.embedding_hidden_mapping_in.bias", &[h]);

    // Shared ALBERT layer
    let lp = "encoder.albert_layer_groups.0.albert_layers.0";
    for name in &[
        "attention.query",
        "attention.key",
        "attention.value",
        "attention.dense",
    ] {
        z(&mut m, &format!("{lp}.{name}.weight"), &[h, h]);
        z(&mut m, &format!("{lp}.{name}.bias"), &[h]);
    }
    m.insert(
        format!("{lp}.attention.LayerNorm.weight"),
        DynTensor::full(&[h], 1.0, DType::F32, &cpu()).unwrap(),
    );
    z(&mut m, &format!("{lp}.attention.LayerNorm.bias"), &[h]);
    z(&mut m, &format!("{lp}.ffn.weight"), &[im, h]);
    z(&mut m, &format!("{lp}.ffn.bias"), &[im]);
    z(&mut m, &format!("{lp}.ffn_output.weight"), &[h, im]);
    z(&mut m, &format!("{lp}.ffn_output.bias"), &[h]);
    m.insert(
        format!("{lp}.full_layer_layer_norm.weight"),
        DynTensor::full(&[h], 1.0, DType::F32, &cpu()).unwrap(),
    );
    z(&mut m, &format!("{lp}.full_layer_layer_norm.bias"), &[h]);

    m
}

// -- Compile + execute + verify helper ----------------------------------------

fn compile_execute_verify(
    cache: &PipelineCache,
    graph: &ComputationGraph,
    inputs_gpu: &[&DynTensor],
    ref_vals: &[f32],
    tol: f32,
    label: &str,
) {
    // Use lower-level API for better error diagnostics on compilation failure.
    let plan = nn_dsl::trace_compile::compile_trace_to_plan_with_fusion(graph)
        .unwrap_or_else(|e| panic!("{label}: trace compilation failed: {e:?}"));
    let compiled = CompiledModel::from_plan(&plan, graph, cache)
        .unwrap_or_else(|e| panic!("{label}: from_plan failed: {e:?}"));
    eprintln!(
        "{label} compiled: {} steps, {} dispatches",
        compiled.num_steps(),
        compiled.num_dispatches()
    );
    let result = compiled
        .execute_dyn(cache, inputs_gpu)
        .unwrap_or_else(|e| panic!("{label}: execute_dyn failed: {e}"));
    let gpu_vals = result
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(
        gpu_vals.len(),
        ref_vals.len(),
        "{label}: output length mismatch"
    );
    let mut max_diff: f32 = 0.0;
    for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
        let diff = (g - r).abs();
        max_diff = max_diff.max(diff);
        assert!(diff <= tol, "{label}[{i}]: gpu={g}, ref={r}, diff={diff}");
    }
    eprintln!(
        "{label} max diff: {max_diff:.2e}, dispatches: {}",
        compiled.num_dispatches()
    );
}

// -- Test: Generator (vocoder) e2e -------------------------------------------

fn build_test_generator() -> Generator {
    let weights = generator_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
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

/// Kokoro Generator e2e: trace → compile → execute on GPU → verify vs eager.
///
/// Exercises Conv1d, ConvTranspose1d, Snake, AdaIN, InstanceNorm, LeakyRelu,
/// Sin, Exp, Clamp, Narrow, Add, MulScalar. Three inputs.
#[test]
fn test_compiled_kokoro_generator() {
    use nn_core::dyn_tensor::trace::{record_input, trace_graph};
    use nn_core::TensorError;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let generator = build_test_generator();

    let batch = 1;
    let t_in = 8;
    let t_full = 16;
    let x = DynTensor::new(
        &super::test_utils::rand_f32_vec(42, batch * GEN_CH * t_in, -0.5, 0.5),
        &[batch, GEN_CH, t_in],
        &cpu(),
    )
    .unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(43, batch * GEN_STYLE_DIM, -0.5, 0.5),
        &[batch, GEN_STYLE_DIM],
        &cpu(),
    )
    .unwrap();
    let har = DynTensor::new(
        &super::test_utils::rand_f32_vec(44, batch * 2 * GEN_N_BINS * t_full, -0.5, 0.5),
        &[batch, 2 * GEN_N_BINS, t_full],
        &cpu(),
    )
    .unwrap();

    let (ref_mag, _) = generator.forward(&x, &style, &har).expect("eager forward");
    let ref_vals = ref_mag.to_flat_vec::<f32>().unwrap();

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
    .expect("trace_graph for Generator");
    // Default output_nodes points to last traced op (sin for phase).
    // Replace with magnitude (the returned tensor) as the actual output.
    if let Some(id) = out.trace_id() {
        assert!(
            graph.set_primary_output(id),
            "magnitude trace_id not found in graph"
        );
    }
    eprintln!("Generator trace: {} nodes", graph.nodes().len());

    let x_gpu = x.to_device(&gpu()).unwrap();
    let style_gpu = style.to_device(&gpu()).unwrap();
    let har_gpu = har.to_device(&gpu()).unwrap();
    // Measured max diff: ~0.17–0.25 (non-deterministic across runs). Root cause:
    // GPU parallel reductions in InstanceNorm (via AdaIN in ResBlock) introduce
    // ~O(1e-3) per-layer drift, amplified through Snake activations and the
    // exp(clamp(log_mag)) output. exp() amplifies additive pre-exp drift.
    // Metal threadgroup scheduling variability causes run-to-run fluctuation.
    // Same mechanism as F0EnergyPredictor (#2449). Tolerance 0.5 covers observed
    // variance with margin.
    compile_execute_verify(
        &cache,
        &graph,
        &[&x_gpu, &style_gpu, &har_gpu],
        &ref_vals,
        0.5,
        "Generator",
    );
}

/// Kokoro Generator multi-output e2e: validate both magnitude and phase heads.
///
/// The magnitude-only test above cannot catch bugs isolated to the phase
/// branch (`narrow(offset>0) -> sin`). This regression test exercises the
/// exact multi-output marking path used by `CompiledKokoro::step_generate`.
#[test]
fn test_compiled_kokoro_generator_multi_output() {
    use std::cell::Cell;

    use nn_core::dyn_tensor::trace::{record_input, trace_graph};
    use nn_core::TensorError;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let generator = build_test_generator();

    let batch = 1;
    let t_in = 8;
    let t_full = 16;
    let x = DynTensor::new(
        &super::test_utils::rand_f32_vec(142, batch * GEN_CH * t_in, -0.5, 0.5),
        &[batch, GEN_CH, t_in],
        &cpu(),
    )
    .unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(143, batch * GEN_STYLE_DIM, -0.5, 0.5),
        &[batch, GEN_STYLE_DIM],
        &cpu(),
    )
    .unwrap();
    let har = DynTensor::new(
        &super::test_utils::rand_f32_vec(144, batch * 2 * GEN_N_BINS * t_full, -0.5, 0.5),
        &[batch, 2 * GEN_N_BINS, t_full],
        &cpu(),
    )
    .unwrap();

    let (ref_mag, ref_phase) = generator.forward(&x, &style, &har).expect("eager forward");
    let ref_mag_vals = ref_mag.to_flat_vec::<f32>().unwrap();
    let ref_phase_vals = ref_phase.to_flat_vec::<f32>().unwrap();

    let phase_id: Cell<Option<u64>> = Cell::new(None);
    let (mag_out, mut graph) = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let mut h = har.clone();
        h.set_trace_id(record_input(h.dims(), DType::F32).unwrap());
        let (mag, phase) = generator
            .forward(&inp, &sty, &h)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        phase_id.set(phase.trace_id());
        Ok(mag)
    })
    .expect("trace_graph for Generator multi-output");
    if let Some(id) = mag_out.trace_id() {
        assert!(
            graph.set_primary_output(id),
            "magnitude trace_id not found in graph"
        );
    }
    if let Some(id) = phase_id.get() {
        assert!(graph.mark_output(id), "phase trace_id not found in graph");
    }

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile generator multi-output");
    assert_eq!(compiled.num_outputs(), 2, "magnitude + phase = 2 outputs");

    let x_gpu = x.to_device(&gpu()).unwrap();
    let style_gpu = style.to_device(&gpu()).unwrap();
    let har_gpu = har.to_device(&gpu()).unwrap();
    let outputs = compiled
        .execute_dyn_outputs(&cache, &[&x_gpu, &style_gpu, &har_gpu])
        .expect("execute generator multi-output");
    assert_eq!(outputs.len(), 2, "expected 2 outputs");

    let mag_vals = outputs[0]
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let phase_vals = outputs[1]
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(
        mag_vals.len(),
        ref_mag_vals.len(),
        "magnitude length mismatch"
    );
    assert_eq!(
        phase_vals.len(),
        ref_phase_vals.len(),
        "phase length mismatch"
    );
    assert!(
        phase_vals.iter().all(|v| v.is_finite()),
        "compiled generator phase contains non-finite values"
    );

    let mut max_mag_diff = 0.0f32;
    for (i, (g, r)) in mag_vals.iter().zip(ref_mag_vals.iter()).enumerate() {
        let diff = (g - r).abs();
        max_mag_diff = max_mag_diff.max(diff);
        assert!(diff <= 0.5, "mag[{i}]: gpu={g}, ref={r}, diff={diff}");
    }

    let mut max_phase_diff = 0.0f32;
    for (i, (g, r)) in phase_vals.iter().zip(ref_phase_vals.iter()).enumerate() {
        let diff = (g - r).abs();
        max_phase_diff = max_phase_diff.max(diff);
        assert!(diff <= 0.5, "phase[{i}]: gpu={g}, ref={r}, diff={diff}");
    }

    eprintln!(
        "Generator multi-output max diff: mag={max_mag_diff:.2e}, phase={max_phase_diff:.2e}, dispatches={}",
        compiled.num_dispatches()
    );
}

/// Same as `test_compiled_kokoro_generator_multi_output`, but uses the
/// `_no_fence` execution path and relies on CPU readback to flush the lazy
/// batch. This matches `CompiledKokoro::step_generate`.
#[test]
fn test_compiled_kokoro_generator_multi_output_no_fence() {
    use std::cell::Cell;

    use nn_core::dyn_tensor::trace::{record_input, trace_graph};
    use nn_core::TensorError;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let generator = build_test_generator();

    let batch = 1;
    let t_in = 8;
    let t_full = 16;
    let x = DynTensor::new(
        &super::test_utils::rand_f32_vec(242, batch * GEN_CH * t_in, -0.5, 0.5),
        &[batch, GEN_CH, t_in],
        &cpu(),
    )
    .unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(243, batch * GEN_STYLE_DIM, -0.5, 0.5),
        &[batch, GEN_STYLE_DIM],
        &cpu(),
    )
    .unwrap();
    let har = DynTensor::new(
        &super::test_utils::rand_f32_vec(244, batch * 2 * GEN_N_BINS * t_full, -0.5, 0.5),
        &[batch, 2 * GEN_N_BINS, t_full],
        &cpu(),
    )
    .unwrap();

    let (ref_mag, ref_phase) = generator.forward(&x, &style, &har).expect("eager forward");
    let ref_mag_vals = ref_mag.to_flat_vec::<f32>().unwrap();
    let ref_phase_vals = ref_phase.to_flat_vec::<f32>().unwrap();

    let phase_id: Cell<Option<u64>> = Cell::new(None);
    let (mag_out, mut graph) = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let mut h = har.clone();
        h.set_trace_id(record_input(h.dims(), DType::F32).unwrap());
        let (mag, phase) = generator
            .forward(&inp, &sty, &h)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        phase_id.set(phase.trace_id());
        Ok(mag)
    })
    .expect("trace_graph for Generator multi-output no-fence");
    if let Some(id) = mag_out.trace_id() {
        assert!(
            graph.set_primary_output(id),
            "magnitude trace_id not found in graph"
        );
    }
    if let Some(id) = phase_id.get() {
        assert!(graph.mark_output(id), "phase trace_id not found in graph");
    }

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile generator multi-output no-fence");

    let x_gpu = x.to_device(&gpu()).unwrap();
    let style_gpu = style.to_device(&gpu()).unwrap();
    let har_gpu = har.to_device(&gpu()).unwrap();
    let outputs = compiled
        .execute_dyn_outputs_no_fence(&cache, &[&x_gpu, &style_gpu, &har_gpu])
        .expect("execute generator multi-output no-fence");
    assert_eq!(outputs.len(), 2, "expected 2 outputs");

    let mag_vals = outputs[0]
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let phase_vals = outputs[1]
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert!(
        phase_vals.iter().all(|v| v.is_finite()),
        "no-fence compiled generator phase contains non-finite values"
    );

    let mut max_mag_diff = 0.0f32;
    for (g, r) in mag_vals.iter().zip(ref_mag_vals.iter()) {
        max_mag_diff = max_mag_diff.max((g - r).abs());
    }

    let mut max_phase_diff = 0.0f32;
    for (g, r) in phase_vals.iter().zip(ref_phase_vals.iter()) {
        max_phase_diff = max_phase_diff.max((g - r).abs());
    }

    assert!(
        max_mag_diff <= 0.5,
        "no-fence magnitude max diff {max_mag_diff} exceeds tolerance"
    );
    assert!(
        max_phase_diff <= 0.5,
        "no-fence phase max diff {max_phase_diff} exceeds tolerance"
    );
}

// -- Test: Text pipeline e2e -------------------------------------------------

/// Kokoro text pipeline post-embedding: Conv1d + LayerNorm + BiLSTM.
///
/// Traces the post-embedding portion of TextEncoder (float inputs only)
/// to avoid I64→GPU transfer limitation. Embedding runs eagerly on CPU,
/// then the float output is the traced variable input.
///
/// Exercises Conv1d, LayerNorm, LeakyReLU, Transpose, LSTM (2x via NativeOp),
/// Flip (2x), Cat through the full CompiledModel pipeline.
///
/// Part of #2236 (LSTM sequence fusion in compiled plan).
#[test]
fn test_compiled_kokoro_text_pipeline() {
    use nn_core::dyn_tensor::trace::{record_input, trace_graph};

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let te_weights = text_encoder_weights();
    let vb_te = VarBuilder::from_tensors(te_weights, DType::F32, &cpu());
    let text_encoder = TextEncoder::load(&vb_te, VOCAB_SIZE, D_EN).unwrap();

    let seq_len = 3;
    let token_ids: Vec<i64> = (0..seq_len).map(|i| (i % VOCAB_SIZE) as i64).collect();
    let tokens = DynTensor::from_vec_i64(token_ids, &[1, seq_len], &cpu()).unwrap();

    // Full eager reference on CPU.
    let ref_output = text_encoder.forward(&tokens).unwrap();
    let ref_vals = ref_output.to_flat_vec::<f32>().unwrap();

    // Embedding on CPU (I64 → F32), then trace the float-only portion.
    let embedded = text_encoder.embed_to_channels_first(&tokens).unwrap();

    let (out, mut graph) = trace_graph(|| {
        let mut x = embedded.clone();
        x.set_trace_id(record_input(x.dims(), DType::F32).unwrap());
        text_encoder
            .forward_post_embedding(&x)
            .map_err(KokoroError::into_tensor_error)
    })
    .expect("trace_graph for text pipeline (post-embedding)");
    eprintln!("Text pipeline trace: {} nodes", graph.nodes().len());

    if let Some(id) = out.trace_id() {
        assert!(
            graph.set_primary_output(id),
            "text pipeline trace_id not found in graph"
        );
    }

    // GPU execution with float input.
    let embedded_gpu = embedded.to_device(&gpu()).unwrap();
    compile_execute_verify(
        &cache,
        &graph,
        &[&embedded_gpu],
        &ref_vals,
        1e-3,
        "TextPipeline",
    );
}

// -- Test: ProsodyPredictor (segment 2) e2e -----------------------------------

/// Kokoro ProsodyPredictor e2e: trace → compile → execute on GPU → verify vs eager.
///
/// Exercises Conv1d, AdaLayerNorm (LayerNorm + Linear + Narrow + broadcast ops),
/// Cat, Expand, LSTM (via NativeOp::LstmSequence), Transpose, Linear, Squeeze.
/// Two inputs: text features + style embedding.
///
/// Part of #2430 (CompiledModel Kokoro middle segments).
/// Part of #2218 (Kokoro epic).
#[test]
fn test_compiled_kokoro_prosody_predictor() {
    use nn_core::dyn_tensor::trace::{record_input, trace_graph};

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let weights = prosody_predictor_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let prosody = ProsodyPredictor::load(&vb, D_EN, STYLE_DIM, PROSODY_N_LAYERS, 50)
        .expect("ProsodyPredictor load");

    let batch = 1;
    let seq_len = 4;
    let x_data = super::test_utils::rand_f32_vec(60, batch * D_EN * seq_len, -0.5, 0.5);
    let x = DynTensor::new(&x_data, &[batch, D_EN, seq_len], &cpu()).unwrap();
    let style_data = super::test_utils::rand_f32_vec(61, batch * STYLE_DIM, -0.5, 0.5);
    let style = DynTensor::new(&style_data, &[batch, STYLE_DIM], &cpu()).unwrap();

    // Eager CPU reference
    let (ref_dur, _) = prosody.forward(&x, &style).expect("eager forward");
    let ref_vals = ref_dur.to_flat_vec::<f32>().unwrap();

    let (out, mut graph) = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let (dur_logits, _) = prosody
            .forward(&inp, &sty)
            .map_err(KokoroError::into_tensor_error)?;
        Ok(dur_logits)
    })
    .expect("trace_graph for ProsodyPredictor");
    eprintln!("ProsodyPredictor trace: {} nodes", graph.nodes().len());

    if let Some(id) = out.trace_id() {
        assert!(
            graph.set_primary_output(id),
            "dur_logits trace_id not found"
        );
    }

    let x_gpu = x.to_device(&gpu()).unwrap();
    let style_gpu = style.to_device(&gpu()).unwrap();
    compile_execute_verify(
        &cache,
        &graph,
        &[&x_gpu, &style_gpu],
        &ref_vals,
        1e-3,
        "ProsodyPredictor",
    );
}

// -- Test: F0EnergyPredictor (segment 3) e2e ----------------------------------

/// Kokoro F0EnergyPredictor e2e: trace → compile → execute on GPU → verify vs eager.
///
/// Exercises BiLSTM (Flip + LSTM + Cat), AdainResBlk1d (AdaIN + LeakyReLU +
/// Conv1d + ConvTranspose1d + upsample_nearest_1d), Linear, Transpose.
/// Two inputs: aligned features + style embedding.
///
/// Uses `t_mel = 16` to give InstanceNorm enough spatial elements for stable
/// GPU reduction (same as Generator test). With `t_mel = 4`, InstanceNorm
/// reduces over only 4 values — GPU parallel vs CPU sequential reduction
/// rounding diverges, amplified ~100-300x by normalization. See #2449.
///
/// Part of #2430 (CompiledModel Kokoro middle segments).
/// Part of #2218 (Kokoro epic).
#[test]
fn test_compiled_kokoro_f0_energy_predictor() {
    use nn_core::dyn_tensor::trace::{record_input, trace_graph};

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let weights = f0_energy_predictor_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let f0_pred =
        F0EnergyPredictor::load(&vb, D_EN, STYLE_DIM, BILSTM_HIDDEN).expect("F0EnergyPredictor");

    let batch = 1;
    // Use t_mel = 16 (not 4) so InstanceNorm in AdaIN has enough spatial
    // elements for stable GPU parallel reduction. See #2449 for analysis.
    let t_mel = 16;
    // F0EnergyPredictor::forward expects [B, d_model+style_dim, T_mel] because
    // the DurationEncoder output already includes style. (#2683)
    let aligned_dim = D_EN + STYLE_DIM;
    let aligned_data = super::test_utils::rand_f32_vec(70, batch * aligned_dim * t_mel, -0.5, 0.5);
    let aligned = DynTensor::new(&aligned_data, &[batch, aligned_dim, t_mel], &cpu()).unwrap();
    let style_data = super::test_utils::rand_f32_vec(71, batch * STYLE_DIM, -0.5, 0.5);
    let style = DynTensor::new(&style_data, &[batch, STYLE_DIM], &cpu()).unwrap();

    // Eager CPU reference — test F0 output
    let (ref_f0, _) = f0_pred.forward(&aligned, &style).expect("eager forward");
    let ref_vals = ref_f0.to_flat_vec::<f32>().unwrap();

    let (out, mut graph) = trace_graph(|| {
        let mut inp = aligned.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let (f0_out, _) = f0_pred
            .forward(&inp, &sty)
            .map_err(KokoroError::into_tensor_error)?;
        Ok(f0_out)
    })
    .expect("trace_graph for F0EnergyPredictor");
    eprintln!("F0EnergyPredictor trace: {} nodes", graph.nodes().len());

    if let Some(id) = out.trace_id() {
        assert!(graph.set_primary_output(id), "f0 trace_id not found");
    }

    let aligned_gpu = aligned.to_device(&gpu()).unwrap();
    let style_gpu = style.to_device(&gpu()).unwrap();
    // Measured max diff: ~7.4e-3. Root cause (confirmed in #2449 diagnostic):
    // BiLSTM fused GPU kernel vs CPU sequential LSTM introduces ~O(1e-3) drift,
    // amplified ~10x by 12 chained InstanceNorm normalization passes (6 blocks
    // × 2 AdaIN each). PrecisionTier::Strict (Kahan reductions) has zero
    // effect — drift is NOT from reduction order. GPU rsqrt vs CPU sqrt+recip
    // also contributes. Tolerance 1e-2 is the achievable bound for this
    // model architecture. See #2449 diagnostic at
    // compiled_model_kokoro_f0_precision.rs.
    compile_execute_verify(
        &cache,
        &graph,
        &[&aligned_gpu, &style_gpu],
        &ref_vals,
        1e-2,
        "F0EnergyPredictor",
    );
}

// -- Test: PlBert + bert_encoder (segment 0) e2e ------------------------------

/// Kokoro PlBert e2e: trace → compile → execute on GPU → verify vs eager.
///
/// Exercises Embedding lookup, broadcast_add, LayerNorm, Linear (factorized
/// projection), self-attention (Q/K/V matmul, softmax, dense), FFN (GELU),
/// shared ALBERT layer (2 iterations). Three inputs: token IDs (F32),
/// pre-computed position embeddings, pre-computed token-type embeddings.
///
/// The trace structure matches `compile_seg_plbert` in `compiled_kokoro_segments.rs`:
/// pre-compute position/token-type embeddings outside the trace scope, then
/// trace word_embedding + broadcast_add + forward_core + bert_encoder + transpose.
///
/// Part of #2744 (compile PlBert as segment 0).
/// Part of #2218 (Kokoro epic).
#[test]
fn test_compiled_kokoro_plbert() {
    use nn_core::dyn_tensor::trace::{record_input, trace_graph};

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let mut config = PlbertConfig::default();
    config.vocab_size = VOCAB_SIZE;
    config.embedding_dim = PLBERT_EMB_DIM;
    config.hidden_size = PLBERT_HIDDEN;
    config.num_attention_heads = PLBERT_NUM_HEADS;
    config.intermediate_size = PLBERT_INTERMEDIATE;
    config.max_position_embeddings = PLBERT_MAX_POS;
    config.num_hidden_layers = PLBERT_NUM_LAYERS;
    config.layer_norm_eps = 1e-12;

    let weights = plbert_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let plbert = PlBert::load(&vb, &config).expect("PlBert::load");

    // bert_encoder: Linear(hidden_size → d_en). In test: [D_EN, PLBERT_HIDDEN].
    let be_w = DynTensor::full(&[D_EN, PLBERT_HIDDEN], 0.01, DType::F32, &cpu()).unwrap();
    let be_b = DynTensor::full(&[D_EN], 0.01, DType::F32, &cpu()).unwrap();
    let bert_encoder = Linear::new(be_w, Some(be_b)).unwrap();

    let batch = 1;
    let seq_len = 4;
    // Token IDs as F32 (compiled pipeline converts U32 → F32).
    let input_ids = DynTensor::from_vec(
        (0..seq_len)
            .map(|i| (i % VOCAB_SIZE) as f32)
            .collect::<Vec<_>>(),
        &[batch, seq_len],
        &cpu(),
    )
    .unwrap();

    // Eager CPU reference: PlBert full forward + bert_encoder + transpose.
    let plbert_out = plbert.forward(&input_ids).expect("PlBert eager forward");
    let ref_output = bert_encoder
        .forward(&plbert_out)
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let ref_vals = ref_output.to_flat_vec::<f32>().unwrap();

    // Pre-compute position and token-type embeddings (matching compile_seg_plbert).
    let position_ids = DynTensor::arange_u32(0, seq_len as u32, &cpu()).unwrap();
    let pos_emb = plbert
        .position_embeddings()
        .forward(&position_ids)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let token_type_ids = DynTensor::zeros(&[seq_len], DType::U32, &cpu()).unwrap();
    let type_emb = plbert
        .token_type_embeddings()
        .forward(&token_type_ids)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    // Trace the compiled path (3 inputs: token IDs, pos_emb, type_emb).
    let (out, mut graph) = trace_graph(|| {
        let mut ids = input_ids.clone();
        ids.set_trace_id(record_input(ids.dims(), DType::F32).unwrap());
        let mut pe = pos_emb.clone();
        pe.set_trace_id(record_input(pe.dims(), DType::F32).unwrap());
        let mut te = type_emb.clone();
        te.set_trace_id(record_input(te.dims(), DType::F32).unwrap());

        let word_emb = plbert.word_embeddings().forward(&ids)?;
        let combined = word_emb.broadcast_add(&pe)?.broadcast_add(&te)?;
        let bert_output = plbert.forward_core(&combined)?;
        let bert_features = bert_encoder.forward(&bert_output)?.transpose(1, 2)?;
        Ok(bert_features)
    })
    .expect("trace_graph for PlBert");
    eprintln!("PlBert trace: {} nodes", graph.nodes().len());

    if let Some(id) = out.trace_id() {
        assert!(
            graph.set_primary_output(id),
            "PlBert trace_id not found in graph"
        );
    }

    let ids_gpu = input_ids.to_device(&gpu()).unwrap();
    let pos_gpu = pos_emb.to_device(&gpu()).unwrap();
    let type_gpu = type_emb.to_device(&gpu()).unwrap();
    // PlBert has 2 ALBERT layers each with self-attention (softmax) and 2
    // LayerNorms, plus a pre-LayerNorm and factorized projection. GPU parallel
    // reductions in LayerNorm and softmax introduce drift that accumulates.
    // Expected tolerance similar to F0EnergyPredictor (1e-2).
    compile_execute_verify(
        &cache,
        &graph,
        &[&ids_gpu, &pos_gpu, &type_gpu],
        &ref_vals,
        1e-2,
        "PlBert",
    );
}
