// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace-based per-layer CROWN composition of the Kokoro pipeline.
//!
//! Each layer is traced via `trace_graph()` on real DynTensor operations,
//! converted to GraphNetwork via `trace_to_graph_model()`, then verified
//! with per-layer CROWN propagation via `verify_layerwise_from_graphs()`.
//!
//! Architecture decomposition (5 layers, S = seq_len, T = time_up):
//! ```text
//!   Layer 0: TextEncoder — Conv1d+ReLU+Linear       [1,D,S] → [1,E,S]
//!   Layer 1: VocoderPre — Conv1d+LeakyReLU           [1,E,S] → [1,V,S]
//!   Layer 2: VocoderUpsample — ConvTranspose1d       [1,V,S] → [1,U,T]
//!   Layer 3: VocoderResBlock — LeakyReLU+Conv1d+res  [1,U,T] → [1,U,T]
//!   Layer 4: VocoderOutput — LeakyReLU+Conv1d+Clamp+Exp [1,U,T] → [1,O,T]
//! ```
//!
//! Part of #2593: Migrate Kokoro TensorBlockBuilder specs to trace-based.
//! Part of #2218: Epic — Perfect Kokoro.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module};
use nn_core::test_utils::cpu;
use nn_core::DType;
use nn_tts_verify::verify_layerwise_from_graphs;
use nn_verify::{trace_to_graph_model, GraphNetwork, PropMethod, VerifyStatus};

use super::common::kokoro_recording::{
    pipeline_crown_coverage, pipeline_tight_stage_count, record_pipeline_certificate,
};
use super::common::kokoro_weights::uniform_bt;

// -- Dimension configuration --------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Dims {
    d_model: usize,
    enc_dim: usize,
    voc_ch: usize,
    voc_up_ch: usize,
    out_ch: usize,
    seq_len: usize,
    up_stride: usize,
    up_kernel: usize,
}

impl Dims {
    fn d64() -> Self {
        Self {
            d_model: 64,
            enc_dim: 64,
            voc_ch: 32,
            voc_up_ch: 32,
            out_ch: 16,
            seq_len: 8,
            up_stride: 2,
            up_kernel: 4,
        }
    }
    fn d128() -> Self {
        Self {
            d_model: 128,
            enc_dim: 128,
            voc_ch: 64,
            voc_up_ch: 64,
            out_ch: 32,
            seq_len: 8,
            up_stride: 2,
            up_kernel: 4,
        }
    }
    fn d256() -> Self {
        Self {
            d_model: 256,
            enc_dim: 256,
            voc_ch: 128,
            voc_up_ch: 128,
            out_ch: 64,
            seq_len: 2,
            up_stride: 2,
            up_kernel: 4,
        }
    }
    fn up_padding(&self) -> usize {
        (self.up_kernel - self.up_stride) / 2
    }
    fn time_up(&self) -> usize {
        (self.seq_len - 1) * self.up_stride + self.up_kernel - 2 * self.up_padding()
    }
}

const WEIGHT_MAG: f64 = 0.001;

// -- Per-layer trace functions ------------------------------------------------

fn w(shape: &[usize]) -> DynTensor {
    DynTensor::full(shape, WEIGHT_MAG, DType::F32, &cpu()).unwrap()
}

fn z(shape: &[usize]) -> DynTensor {
    DynTensor::zeros(shape, DType::F32, &cpu()).unwrap()
}

/// Trace a closure as a single-input GraphNetwork.
fn trace_single(
    input_shape: &[usize],
    f: impl FnOnce(DynTensor) -> nn_core::Result<DynTensor>,
) -> GraphNetwork {
    let input = DynTensor::zeros(input_shape, DType::F32, &cpu()).unwrap();
    let (_, graph) = trace_graph(|| {
        let mut x = input.clone();
        let id = record_input(x.dims(), DType::F32).unwrap();
        x.set_trace_id(id);
        f(x)
    })
    .expect("trace");
    trace_to_graph_model(&graph).expect("graph").graph
}

/// Layer 0: TextEncoder — Conv1d(d→d,k=3,p=1) + ReLU + Linear(d→enc).
fn trace_text_encoder(dims: &Dims) -> GraphNetwork {
    let d = dims.d_model;
    let enc = dims.enc_dim;
    let conv = Conv1d::new(w(&[d, d, 3]), None, Conv1dConfig::default().with_padding(1)).unwrap();
    let proj = Linear::new(w(&[enc, d]), Some(z(&[enc]))).unwrap();
    trace_single(&[1, d, dims.seq_len], move |x| {
        let x = conv.forward(&x)?;
        let x = x.relu()?;
        let x = x.transpose(1, 2)?; // [1,d,s] → [1,s,d]
        let x = proj.forward(&x)?; // [1,s,enc]
        x.transpose(1, 2) // [1,enc,s]
    })
}

/// Layer 1: VocoderPre — Conv1d(enc→voc_ch,k=3,p=1) + LeakyReLU(0.1).
fn trace_vocoder_pre(dims: &Dims) -> GraphNetwork {
    let conv = Conv1d::new(
        w(&[dims.voc_ch, dims.enc_dim, 3]),
        None,
        Conv1dConfig::default().with_padding(1),
    )
    .unwrap();
    trace_single(&[1, dims.enc_dim, dims.seq_len], move |x| {
        let x = conv.forward(&x)?;
        x.leaky_relu(0.1)
    })
}

/// Layer 2: VocoderUpsample — ConvTranspose1d(voc_ch→voc_up_ch).
fn trace_vocoder_upsample(dims: &Dims) -> GraphNetwork {
    let up = ConvTranspose1d::new(
        w(&[dims.voc_ch, dims.voc_up_ch, dims.up_kernel]),
        None,
        ConvTranspose1dConfig::default()
            .with_stride(dims.up_stride)
            .with_padding(dims.up_padding()),
    )
    .unwrap();
    trace_single(&[1, dims.voc_ch, dims.seq_len], move |x| up.forward(&x))
}

/// Layer 3: VocoderResBlock — LeakyReLU(0.1) + Conv1d + residual.
fn trace_vocoder_resblock(dims: &Dims) -> GraphNetwork {
    let conv = Conv1d::new(
        w(&[dims.voc_up_ch, dims.voc_up_ch, 3]),
        None,
        Conv1dConfig::default().with_padding(1),
    )
    .unwrap();
    trace_single(&[1, dims.voc_up_ch, dims.time_up()], move |x| {
        let residual = x.clone();
        let h = x.leaky_relu(0.1)?;
        let h = conv.forward(&h)?;
        residual.add(&h)
    })
}

/// Layer 4: VocoderOutput — LeakyReLU(0.01) + Conv1d + Clamp + Exp.
///
/// The clamp to [-88, 88] matches production `kokoro_decoder.rs:279` where
/// `log_mag_clamped = log_mag.clamp(-LOG_MAG_CLAMP_MAX, LOG_MAG_CLAMP_MAX)`.
/// Without this, ForwardMode bounds through deep ResBlock chains can exceed the
/// Exp overflow threshold, causing verification failures (#2625).
fn trace_vocoder_output(dims: &Dims) -> GraphNetwork {
    let conv = Conv1d::new(
        w(&[dims.out_ch, dims.voc_up_ch, 3]),
        None,
        Conv1dConfig::default().with_padding(1),
    )
    .unwrap();
    trace_single(&[1, dims.voc_up_ch, dims.time_up()], move |x| {
        let x = x.leaky_relu(0.01)?;
        let x = conv.forward(&x)?;
        let x = x.clamp(-88.0, 88.0)?;
        x.exp()
    })
}

fn trace_kokoro_layerwise(dims: &Dims) -> Vec<GraphNetwork> {
    vec![
        trace_text_encoder(dims),
        trace_vocoder_pre(dims),
        trace_vocoder_upsample(dims),
        trace_vocoder_resblock(dims),
        trace_vocoder_output(dims),
    ]
}

// -- Bounds helpers -----------------------------------------------------------

fn f64_min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}
fn f64_max(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn assert_p1_p2(cert: &nn_tts_verify::PipelineCertificate, label: &str) {
    let lo_min = f64_min(&cert.e2e_output_lower);
    let hi_max = f64_max(&cert.e2e_output_upper);
    assert!(
        lo_min > 0.0,
        "{label} P1: exp output positive, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "{label} P2: output bounded, got {hi_max}"
    );
    eprintln!("{label}: [{lo_min:.8}, {hi_max:.6}] — P1 ✓ P2 ✓");
}

fn assert_non_vacuous(cert: &nn_tts_verify::PipelineCertificate, label: &str) {
    let total = cert.stages.len();
    let crown_stages = pipeline_tight_stage_count(cert);
    let coverage = pipeline_crown_coverage(cert);
    let output_width = (f64_max(&cert.e2e_output_upper) - f64_min(&cert.e2e_output_lower)) as f32;
    assert!(
        coverage >= 0.5 && output_width < 10.0,
        "{label}: vacuous — crown_coverage={coverage:.2} ({crown_stages}/{total}), \
         output_width={output_width:.4}"
    );
    eprintln!("{label}: non-vacuous ✓ coverage={coverage:.2} width={output_width:.4}");
}

// -- Status recording ---------------------------------------------------------

fn record_pipeline_to_status(
    status: &mut VerifyStatus,
    cert: &nn_tts_verify::PipelineCertificate,
    key: &str,
    output_shape: &[usize],
) -> PropMethod {
    record_pipeline_certificate(status, key, cert, output_shape, None)
}

/// Assert no single layer expands IBP bounds by more than `max_factor` (#2594 AC2).
///
/// Propagates IBP through each layer sequentially, tracking the max element-wise
/// width at each stage. Layers that shrink or maintain bounds are fine; layers
/// that expand beyond `max_factor` indicate bound explosion.
fn assert_per_layer_expansion(
    graphs: &[GraphNetwork],
    initial: &nn_verify::BoundedTensor,
    max_factor: f32,
    label: &str,
) {
    let layer_names = [
        "TextEncoder",
        "VocoderPre",
        "VocoderUpsample",
        "VocoderResBlock",
        "VocoderOutput",
    ];
    let mut current = initial.clone();
    let mut prev_width = current.max_width();
    for (i, graph) in graphs.iter().enumerate() {
        current = graph
            .propagate_ibp(&current)
            .unwrap_or_else(|e| panic!("IBP layer {i}: {e}"));
        let cur_width = current.max_width();
        let name = layer_names.get(i).unwrap_or(&"unknown");
        if prev_width > 1e-10 {
            let expansion = cur_width / prev_width;
            eprintln!(
                "  {label} layer {i} ({name}): width {prev_width:.4} → {cur_width:.4} ({expansion:.1}x)"
            );
            assert!(
                expansion < max_factor,
                "{label} layer {i} ({name}): expansion {expansion:.1}x exceeds {max_factor}x"
            );
        }
        prev_width = cur_width;
    }
}

fn compute_ibp_width(graphs: &[GraphNetwork], initial: &nn_verify::BoundedTensor) -> f32 {
    let mut current = initial.clone();
    for (i, graph) in graphs.iter().enumerate() {
        current = graph
            .propagate_ibp(&current)
            .unwrap_or_else(|e| panic!("IBP layer {i}: {e}"));
    }
    let lo = current
        .lower()
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let hi = current
        .upper()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    hi - lo
}

fn verify_and_record(
    dims: &Dims,
    status: &mut VerifyStatus,
    key: &str,
) -> (nn_tts_verify::PipelineCertificate, PropMethod) {
    let graphs = trace_kokoro_layerwise(dims);
    let initial = uniform_bt(&[1, dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert =
        verify_layerwise_from_graphs(&graphs, &initial).unwrap_or_else(|e| panic!("{key}: {e}"));
    let out_shape = cert.stages.last().expect("stages").output_shape.clone();
    let method = record_pipeline_to_status(status, &cert, key, &out_shape);
    let ibp_width = compute_ibp_width(&graphs, &initial);
    if method.is_tight() {
        status
            .record_crown_comparison(key, ibp_width)
            .unwrap_or_else(|e| panic!("{key} IBP: {e}"));
    } else {
        eprintln!("{key}: skipping CROWN/IBP comparison because method={method:?}");
    }
    let crown_width = (f64_max(&cert.e2e_output_upper) - f64_min(&cert.e2e_output_lower)) as f32;
    let ratio = crown_width / ibp_width.max(1e-10);
    eprintln!(
        "{key}: method={method:?}, crown_w={crown_width:.6}, ibp_w={ibp_width:.6}, ratio={ratio:.4}"
    );
    (cert, method)
}

fn status_file_path() -> std::path::PathBuf {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("root");
    nn_verify::model_status_path(workspace_root, "kokoro")
}

// -- Tests --------------------------------------------------------------------

#[test]
fn test_trace_layerwise_d64_all_properties() {
    let dims = Dims::d64();
    let graphs = trace_kokoro_layerwise(&dims);
    let initial = uniform_bt(&[1, dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise_from_graphs(&graphs, &initial).expect("D=64 layerwise");
    assert!(cert.is_valid, "D=64 layerwise must be valid");
    assert_p1_p2(&cert, "D=64");

    // Per-layer expansion factor check (#2594 AC2): no single layer should
    // explode bounds by more than 1000x. Calibrated from D=64 test runs where
    // individual layers expand by 1-50x depending on architecture.
    assert_per_layer_expansion(&graphs, &initial, 1000.0, "D=64");
}

#[test]
fn test_trace_layerwise_d128_all_properties() {
    let dims = Dims::d128();
    let graphs = trace_kokoro_layerwise(&dims);
    let initial = uniform_bt(&[1, dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise_from_graphs(&graphs, &initial).expect("D=128 layerwise");
    assert!(cert.is_valid, "D=128 layerwise must be valid");
    assert_p1_p2(&cert, "D=128");
    assert_non_vacuous(&cert, "D=128");
}

#[test]
fn test_trace_layerwise_d256_all_properties() {
    let dims = Dims::d256();
    let graphs = trace_kokoro_layerwise(&dims);
    let initial = uniform_bt(&[1, dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise_from_graphs(&graphs, &initial).expect("D=256 layerwise");
    assert!(cert.is_valid, "D=256 layerwise must be valid");
    assert_p1_p2(&cert, "D=256");
}

/// Persist D=64/D=128/D=256 CROWN results, overwriting stale builder entries.
#[test]
fn test_trace_layerwise_persist_crown_status() {
    let status_path = status_file_path();
    let mut locked = VerifyStatus::load_locked(&status_path).expect("load_locked");
    let (_, d64_method) = verify_and_record(
        &Dims::d64(),
        &mut locked.status,
        "kokoro_layerwise_d64_crown",
    );
    let (_, d128_method) = verify_and_record(
        &Dims::d128(),
        &mut locked.status,
        "kokoro_layerwise_d128_crown",
    );
    let (_, d256_method) = verify_and_record(
        &Dims::d256(),
        &mut locked.status,
        "kokoro_layerwise_d256_crown",
    );
    locked.save().expect("save status");
    drop(locked);

    let v = VerifyStatus::load_locked(&status_path).expect("validation");
    for (key, expected_method) in [
        ("kokoro_layerwise_d64_crown", d64_method),
        ("kokoro_layerwise_d128_crown", d128_method),
        ("kokoro_layerwise_d256_crown", d256_method),
    ] {
        let entry = v.status.kernel(key).unwrap_or_else(|| panic!("{key}"));
        assert_eq!(
            entry.method, expected_method,
            "{key} status entry must reflect actual per-stage method mix"
        );
        assert!(
            expected_method.is_tight(),
            "{key} silently degraded from CROWN-family propagation to {expected_method:?}"
        );
        assert!(
            entry.crown_ibp_ratio.is_some(),
            "{key} must keep CROWN/IBP comparison data when recorded as {expected_method:?}"
        );
        assert!(
            !entry.stale,
            "{key} must not be stale after trace-based refresh"
        );
    }
}
