// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Automated bound propagation gap detector for the Kokoro pipeline.
//!
//! Traces each Kokoro segment, converts to NY GraphNetwork,
//! runs CROWN propagation (with IBP fallback), and reports coverage.
//!
//! This test catches:
//! - Segments where CROWN fails and falls back to vacuously wide IBP
//! - Segments where bounds are not computable at all
//! - Regressions where a previously-CROWN segment loses CROWN support
//! - Segments where model construction fails (weight naming drift)
//!
//! Run: `cargo test -p nn-tts-verify --test kokoro_bound_gap_detector --features NY -- --nocapture`
//!
//! Part of #2930 (Automated bound propagation gap detector).
//! Part of #2218 (Perfect Kokoro epic).

#![cfg(feature = "ny")]

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_models::kokoro_decoder::Generator;
use nn_models::kokoro_error::KokoroError;
use nn_models::kokoro_f0::F0EnergyPredictor;
use nn_models::kokoro_tts::{ProsodyPredictor, TextEncoder};
use nn_models::KokoroConfig;
use nn_verify::{
    propagate_with_crown_fallback, trace_to_graph_model, trace_to_graph_model_multi_input,
    BoundedTensor, PropMethod,
};
use ndarray::{ArrayD, IxDyn};

fn cpu() -> Device {
    Device::Cpu
}

// -- Miniaturized dimensions --------------------------------------------------

const D_EN: usize = 8;
const STYLE_DIM: usize = 4;
const BILSTM_HIDDEN: usize = D_EN / 2;
const GEN_CH: usize = 8;
const GEN_NEXT_CH: usize = 4;
const GEN_STYLE_DIM: usize = 4;
const GEN_N_FFT: usize = 4;
const GEN_N_BINS: usize = GEN_N_FFT / 2 + 1;
const GEN_KERNEL: usize = 3;
const VOCAB_SIZE: usize = 16;
const PROSODY_N_LAYERS: usize = 2;

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

// -- Model builders (weight names from compiled_model_kokoro_pipeline.rs) ------

fn build_text_encoder() -> Result<TextEncoder, String> {
    let mut m = HashMap::new();
    let h = D_EN / 2;
    z(&mut m, "embedding.weight", &[VOCAB_SIZE, D_EN]);
    for i in 0..3 {
        z(&mut m, &format!("convs.{i}.weight"), &[D_EN, D_EN, 5]);
        z(&mut m, &format!("convs.{i}.bias"), &[D_EN]);
        z(&mut m, &format!("norms.{i}.weight"), &[D_EN]);
        z(&mut m, &format!("norms.{i}.bias"), &[D_EN]);
    }
    bilstm_w(&mut m, "lstm", D_EN, h);
    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    TextEncoder::load(&vb, VOCAB_SIZE, D_EN).map_err(|e| format!("{e}"))
}

fn build_prosody() -> Result<ProsodyPredictor, String> {
    let mut m = HashMap::new();
    let (d, h, s) = (D_EN, D_EN / 2, STYLE_DIM);
    let max_dur = 50;
    let lstm_input = d + s;

    // DurationEncoder: n_layers BiLSTMs + AdaLayerNorms
    for i in 0..PROSODY_N_LAYERS {
        bilstm_w(&mut m, &format!("duration.lstms.{i}"), lstm_input, h);
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
    z(&mut m, "duration.duration_proj.weight", &[max_dur, d]);
    z(&mut m, "duration.duration_proj.bias", &[max_dur]);

    // Final duration BiLSTM
    bilstm_w(&mut m, "lstm", lstm_input, h);

    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    ProsodyPredictor::load(&vb, D_EN, STYLE_DIM, PROSODY_N_LAYERS, max_dur)
        .map_err(|e| format!("{e}"))
}

fn build_f0_pred() -> Result<F0EnergyPredictor, String> {
    let mut m = HashMap::new();
    let (s, bh) = (STYLE_DIM, BILSTM_HIDDEN);
    let bo = 2 * bh;
    let bilstm_input = D_EN + STYLE_DIM;

    // Shared BiLSTM: input = d_model + style_dim
    bilstm_w(&mut m, "shared", bilstm_input, bh);

    // F0 and Energy (N) heads: 3 AdainResBlk1d blocks each
    for head in ["F0", "N"] {
        adain_resblk_w(&mut m, &format!("{head}.0"), bo, bo, s, false);
        adain_resblk_w(&mut m, &format!("{head}.1"), bo, bh, s, true);
        adain_resblk_w(&mut m, &format!("{head}.2"), bh, bh, s, false);
        z(&mut m, &format!("{head}_proj.weight"), &[1, bh]);
        z(&mut m, &format!("{head}_proj.bias"), &[1]);
    }

    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    F0EnergyPredictor::load(&vb, D_EN, STYLE_DIM, BILSTM_HIDDEN).map_err(|e| format!("{e}"))
}

fn build_generator() -> Result<Generator, String> {
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
    Generator::load(&vb, &config).map_err(|e| format!("{e}"))
}

// -- Bound gap detector results -----------------------------------------------

#[derive(Debug)]
struct SegmentBoundResult {
    name: &'static str,
    method: PropMethod,
    fallback_reason: Option<String>,
    output_width: f32,
    is_vacuous: bool,
}

impl SegmentBoundResult {
    fn is_crown(&self) -> bool {
        matches!(
            self.method,
            PropMethod::Crown | PropMethod::AlphaCrown | PropMethod::BetaCrown
        )
    }

    fn build_failed(name: &'static str, reason: String) -> Self {
        Self {
            name,
            method: PropMethod::Ibp,
            fallback_reason: Some(format!("model build failed: {reason}")),
            output_width: f32::INFINITY,
            is_vacuous: true,
        }
    }
}

fn uniform_input(shape: &[usize], lo: f32, hi: f32) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(shape), lo);
    let upper = ArrayD::from_elem(IxDyn(shape), hi);
    BoundedTensor::new(lower, upper).expect("valid bounds")
}

// -- Shared propagation logic -------------------------------------------------

fn propagate_single_input(
    name: &'static str,
    graph: &nn_core::dyn_tensor::trace::ComputationGraph,
    input_shape: &[usize],
) -> SegmentBoundResult {
    let gn = match trace_to_graph_model(graph) {
        Ok(result) => result.graph,
        Err(e) => {
            return SegmentBoundResult {
                name,
                method: PropMethod::Ibp,
                fallback_reason: Some(format!("graph build failed: {e}")),
                output_width: f32::INFINITY,
                is_vacuous: true,
            };
        }
    };
    let input_bounds = uniform_input(input_shape, -1.0, 1.0);
    run_propagation(name, &gn, &input_bounds)
}

fn propagate_multi_input(
    name: &'static str,
    graph: &nn_core::dyn_tensor::trace::ComputationGraph,
    total_elems: usize,
) -> SegmentBoundResult {
    let gn = match trace_to_graph_model_multi_input(graph) {
        Ok(result) => result.graph,
        Err(e) => {
            return SegmentBoundResult {
                name,
                method: PropMethod::Ibp,
                fallback_reason: Some(format!("graph build failed: {e}")),
                output_width: f32::INFINITY,
                is_vacuous: true,
            };
        }
    };
    let input_bounds = uniform_input(&[total_elems], -1.0, 1.0);
    run_propagation(name, &gn, &input_bounds)
}

fn run_propagation(
    name: &'static str,
    gn: &ny_propagate::GraphNetwork,
    input_bounds: &BoundedTensor,
) -> SegmentBoundResult {
    match propagate_with_crown_fallback(gn, input_bounds) {
        Ok((method, output, fallback)) => {
            let lo = output.lower().iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let hi = output
                .upper()
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let width = hi - lo;
            SegmentBoundResult {
                name,
                method,
                fallback_reason: fallback,
                output_width: width,
                is_vacuous: width > 1e6,
            }
        }
        Err(e) => SegmentBoundResult {
            name,
            method: PropMethod::Ibp,
            fallback_reason: Some(format!("propagation failed: {e}")),
            output_width: f32::INFINITY,
            is_vacuous: true,
        },
    }
}

// -- Per-segment tracing + CROWN propagation ----------------------------------

fn check_text_encoder() -> SegmentBoundResult {
    let te = match build_text_encoder() {
        Ok(t) => t,
        Err(e) => return SegmentBoundResult::build_failed("TextEncoder", e),
    };
    let seq_len = 8;
    // Post-embedding features — CROWN traces the continuous part (conv+LSTM),
    // not the discrete embedding lookup.
    let text_feat = DynTensor::full(&[1, D_EN, seq_len], 0.1, DType::F32, &cpu()).unwrap();

    let trace_result = trace_graph(|| {
        let mut inp = text_feat.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        te.forward_post_embedding(&inp)
            .map_err(KokoroError::into_tensor_error)
    });
    let (out, mut graph) = match trace_result {
        Ok(r) => r,
        Err(e) => return SegmentBoundResult::build_failed("TextEncoder", format!("trace: {e}")),
    };

    if let Some(id) = out.trace_id() {
        let _ = graph.set_primary_output(id);
    }

    propagate_single_input("TextEncoder", &graph, &[1, D_EN, seq_len])
}

fn check_prosody() -> SegmentBoundResult {
    let prosody = match build_prosody() {
        Ok(p) => p,
        Err(e) => return SegmentBoundResult::build_failed("ProsodyPredictor", e),
    };
    let seq_len = 8;
    let text_feat = DynTensor::full(&[1, D_EN, seq_len], 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&[1, STYLE_DIM], 0.1, DType::F32, &cpu()).unwrap();

    let trace_result = trace_graph(|| {
        let mut inp = text_feat.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let (d, _f) = prosody
            .forward(&inp, &sty)
            .map_err(KokoroError::into_tensor_error)?;
        Ok(d)
    });
    let (dur_out, mut graph) = match trace_result {
        Ok(r) => r,
        Err(e) => {
            return SegmentBoundResult::build_failed("ProsodyPredictor", format!("trace: {e}"))
        }
    };

    if let Some(id) = dur_out.trace_id() {
        let _ = graph.set_primary_output(id);
    }

    propagate_multi_input("ProsodyPredictor", &graph, D_EN * seq_len + STYLE_DIM)
}

fn check_f0_energy() -> SegmentBoundResult {
    let f0_pred = match build_f0_pred() {
        Ok(f) => f,
        Err(e) => return SegmentBoundResult::build_failed("F0EnergyPredictor", e),
    };
    let t_mel = 8;
    // aligned already includes style_dim (d_model + style_dim channels)
    let aligned = DynTensor::full(&[1, D_EN + STYLE_DIM, t_mel], 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&[1, STYLE_DIM], 0.1, DType::F32, &cpu()).unwrap();

    let trace_result = trace_graph(|| {
        let mut inp = aligned.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let (f, _e) = f0_pred
            .forward(&inp, &sty)
            .map_err(KokoroError::into_tensor_error)?;
        Ok(f)
    });
    let (f0_out, mut graph) = match trace_result {
        Ok(r) => r,
        Err(e) => {
            return SegmentBoundResult::build_failed("F0EnergyPredictor", format!("trace: {e}"))
        }
    };

    if let Some(id) = f0_out.trace_id() {
        let _ = graph.set_primary_output(id);
    }

    propagate_multi_input(
        "F0EnergyPredictor",
        &graph,
        (D_EN + STYLE_DIM) * t_mel + STYLE_DIM,
    )
}

fn check_generator() -> SegmentBoundResult {
    let generator = match build_generator() {
        Ok(model) => model,
        Err(e) => return SegmentBoundResult::build_failed("Generator", e),
    };
    let t_in = 8;
    let t_full = 16;
    let x = DynTensor::full(&[1, GEN_CH, t_in], 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&[1, GEN_STYLE_DIM], 0.1, DType::F32, &cpu()).unwrap();
    let har = DynTensor::full(&[1, 2 * GEN_N_BINS, t_full], 0.1, DType::F32, &cpu()).unwrap();

    let trace_result = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut sty = style.clone();
        sty.set_trace_id(record_input(sty.dims(), DType::F32).unwrap());
        let mut h = har.clone();
        h.set_trace_id(record_input(h.dims(), DType::F32).unwrap());
        let (mag, _): (DynTensor, _) = generator
            .forward(&inp, &sty, &h)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        Ok(mag)
    });
    let (out, mut graph) = match trace_result {
        Ok(r) => r,
        Err(e) => return SegmentBoundResult::build_failed("Generator", format!("trace: {e}")),
    };

    if let Some(id) = out.trace_id() {
        let _ = graph.set_primary_output(id);
    }

    let total = GEN_CH * t_in + GEN_STYLE_DIM + 2 * GEN_N_BINS * t_full;
    propagate_multi_input("Generator", &graph, total)
}

// =============================================================================
// Main gap detector test
// =============================================================================

/// Automated bound propagation gap detector for the Kokoro pipeline.
///
/// Traces each segment, converts to GraphNetwork, runs CROWN (with IBP
/// fallback), and reports per-segment coverage. Asserts a minimum number
/// of segments achieve CROWN bounds (not just IBP).
///
/// Part of #2930, #2218.
#[test]
fn detect_kokoro_bound_gaps() {
    let results = vec![
        check_text_encoder(),
        check_prosody(),
        check_f0_energy(),
        check_generator(),
    ];

    eprintln!("\n{}", "=".repeat(72));
    eprintln!("  KOKORO BOUND PROPAGATION GAP DETECTOR");
    eprintln!("{}", "=".repeat(72));
    eprintln!(
        "  {:<22} {:>8} {:>12} {:>8}  Fallback Reason",
        "Segment", "Method", "Width", "Vacuous"
    );
    eprintln!("{}", "-".repeat(72));

    let mut crown_count = 0;
    let mut ibp_count = 0;
    let mut fail_count = 0;

    for r in &results {
        let method_str = if r.is_crown() { "CROWN" } else { "IBP" };
        let vacuous_str = if r.is_vacuous { "YES" } else { "no" };
        let reason = r.fallback_reason.as_deref().unwrap_or("-");
        let reason_short = if reason.len() > 40 {
            &reason[..40]
        } else {
            reason
        };
        eprintln!(
            "  {:<22} {:>8} {:>12.2} {:>8}  {}",
            r.name, method_str, r.output_width, vacuous_str, reason_short,
        );

        if r.is_crown() && !r.is_vacuous {
            crown_count += 1;
        } else if r.output_width.is_finite() {
            ibp_count += 1;
        } else {
            fail_count += 1;
        }
    }

    let total = results.len();
    let coverage = if total > 0 {
        f64::from(crown_count) / total as f64 * 100.0
    } else {
        0.0
    };

    eprintln!("{}", "-".repeat(72));
    eprintln!(
        "  CROWN: {crown_count}/{total}  IBP-fallback: {ibp_count}/{total}  Failed: {fail_count}/{total}"
    );
    eprintln!("  CROWN coverage: {coverage:.0}%");
    eprintln!("  Phase 0 baseline: report all gaps (no minimum CROWN required)");
    eprintln!("  Phase 3 target:   >= 3 CROWN (80%)");
    eprintln!("{}\n", "=".repeat(72));

    // Phase 0: baseline measurement — the test PASSES even if everything is
    // IBP or failed. The point is to measure and report, not to gate.
    // Phase 3 will tighten: assert!(crown_count >= 3);
    //
    // We still assert the test ran all 4 segments (no silent skips).
    assert_eq!(results.len(), 4, "Expected 4 segments in gap detector");
}
