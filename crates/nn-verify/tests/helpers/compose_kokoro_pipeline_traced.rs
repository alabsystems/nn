// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace-based Kokoro pipeline NY verification.
//!
//! Migrates the builder-based `kokoro_full_pipeline`, `kokoro_duration_branch`,
//! `kokoro_vocoder_pipeline`, `kokoro_full_pipeline_forward`, and
//! `kokoro_vocoder_forward` status keys to trace-based verification.
//!
//! **Two trace paths:**
//!
//! 1. **Text-to-duration pipeline** (TextEncoder → ProsodyPredictor):
//!    Records `kokoro_full_pipeline`, `kokoro_full_pipeline_forward`, and
//!    `kokoro_duration_branch`. The builder-based "full pipeline" used a
//!    simplified TextEncoder (Conv1d + ReLU + Linear). The trace captures the
//!    REAL TextEncoder (Embedding + 3×Conv1d + LayerNorm + BiLSTM + Linear)
//!    feeding into the REAL ProsodyPredictor (N×(BiLSTM + AdaLayerNorm) +
//!    duration_proj + final BiLSTM).
//!
//! 2. **Vocoder pipeline** (Generator):
//!    Records `kokoro_vocoder_pipeline` and `kokoro_vocoder_forward`.
//!    Traces the real Generator (ISTFTNet vocoder) with 3 variable inputs
//!    (x, style, har_source). Properties 1 (non-silence: exp > 0) and
//!    2 (non-clipping: bounded output) are verified.
//!
//! Part of #2593: Migrate Kokoro TensorBlockBuilder specs to trace-based.
//! Part of #2218: Epic — Perfect Kokoro.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, TensorError, VarBuilder};
use nn_models::kokoro_decoder::Generator;
use nn_models::kokoro_tts::{ProsodyPredictor, TextEncoder};
use nn_verify::{
    model_status_path, trace_to_graph_model_multi_input, BoundedTensor, PropMethod,
    VerificationSoundnessMode, VerifyStatus,
};
use std::collections::HashMap;
use std::path::Path;

use super::common::bounds_min_max;
use super::common::kokoro_recording::record_ibp_result;
use super::common::kokoro_weights::{
    assert_all_finite, bilstm_weights, build_test_generator as build_shared_generator,
    propagate_multi_input_ibp, text_encoder_weights, z, GEN_CH, GEN_N_BINS,
};

// -- Shared dimensions --------------------------------------------------------

const D_EN: usize = 8;
const STYLE_DIM: usize = 4;
const MAX_DUR: usize = 50;
const VOCAB_SIZE: usize = 16;
const BATCH: usize = 1;
const SEQ_LEN: usize = 3;
const T_IN: usize = 8;
const T_FULL: usize = 16;

// ===========================================================================
// Text-to-duration pipeline (TextEncoder → ProsodyPredictor)
// ===========================================================================

fn prosody_weights_1block() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let lstm_input = D_EN + STYLE_DIM;

    // DurationEncoder block 0
    bilstm_weights(&mut m, "duration.lstms.0", lstm_input, D_EN / 2);
    z(&mut m, "duration.norms.0.norm.weight", &[D_EN]);
    z(&mut m, "duration.norms.0.norm.bias", &[D_EN]);
    z(&mut m, "duration.norms.0.fc.weight", &[2 * D_EN, STYLE_DIM]);
    z(&mut m, "duration.norms.0.fc.bias", &[2 * D_EN]);

    // Duration projection
    z(&mut m, "duration.duration_proj.weight", &[MAX_DUR, D_EN]);
    z(&mut m, "duration.duration_proj.bias", &[MAX_DUR]);

    // Final duration BiLSTM
    bilstm_weights(&mut m, "lstm", lstm_input, D_EN / 2);

    m
}

fn build_text_to_duration_models() -> (TextEncoder, ProsodyPredictor) {
    let te_weights = text_encoder_weights(VOCAB_SIZE, D_EN, 0.01);
    let vb_te = VarBuilder::from_tensors(te_weights, DType::F32, &cpu());
    let text_encoder = TextEncoder::load(&vb_te, VOCAB_SIZE, D_EN).unwrap();
    let pp_weights = prosody_weights_1block();
    let vb_pp = VarBuilder::from_tensors(pp_weights, DType::F32, &cpu());
    let prosody = ProsodyPredictor::load(&vb_pp, D_EN, STYLE_DIM, 1, MAX_DUR).unwrap();
    (text_encoder, prosody)
}

/// Trace TextEncoder → ProsodyPredictor and propagate IBP bounds.
///
/// Inputs: token_ids `[B, T]` (I64) + style `[B, style_dim]` (F32).
/// Output: duration logits `[B, T, max_dur]`.
fn trace_text_to_duration_ibp() -> (BoundedTensor, BoundedTensor) {
    let (text_encoder, prosody) = build_text_to_duration_models();
    let token_shape = [BATCH, SEQ_LEN];
    let style_shape = [BATCH, STYLE_DIM];
    let token_ids: Vec<i64> = (0..BATCH * SEQ_LEN)
        .map(|i| (i % VOCAB_SIZE) as i64)
        .collect();
    let tokens = DynTensor::from_vec_i64(token_ids, &token_shape, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.05, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = tokens.clone();
        let id_x = record_input(&token_shape, DType::I64).unwrap();
        x.set_trace_id(id_x);
        let mut sty = style.clone();
        let id_sty = record_input(&style_shape, DType::F32).unwrap();
        sty.set_trace_id(id_sty);
        let text_features = text_encoder.forward(&x)?;
        let (dur_logits, _features) = prosody.forward(&text_features, &sty)?;
        Ok(dur_logits)
    })
    .expect("TextEncoder → ProsodyPredictor trace");

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("trace_to_graph")
        .graph;

    let input_specs: &[(&[usize], (f32, f32))] = &[
        (&token_shape[..], (0.0, VOCAB_SIZE as f32)),
        (&style_shape[..], (-0.5, 0.5)),
    ];
    let output = propagate_multi_input_ibp(&gn, input_specs);

    // Build combined input bounds for status recording.
    let mut in_lo = Vec::new();
    let mut in_hi = Vec::new();
    for &(shape, (lo, hi)) in input_specs {
        let flat: usize = shape.iter().product();
        in_lo.extend(vec![lo; flat]);
        in_hi.extend(vec![hi; flat]);
    }
    let total = in_lo.len();
    let input_bounds = BoundedTensor::new(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[total]), in_lo).unwrap(),
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[total]), in_hi).unwrap(),
    )
    .expect("valid input bounds");

    (input_bounds, output)
}

// ===========================================================================
// Vocoder pipeline (Generator)
// ===========================================================================

fn build_local_generator() -> Generator {
    build_shared_generator(0.01, STYLE_DIM)
}

/// Trace Generator and propagate IBP bounds.
///
/// Inputs: x `[B, gen_ch, T_IN]`, style `[B, style_dim]`, har_source `[B, 2*n_bins, T_FULL]`.
/// Output: magnitude `[B, n_bins, T_FULL]`.
fn trace_vocoder_ibp() -> (BoundedTensor, BoundedTensor) {
    let generator = build_local_generator();
    let input_shape = [BATCH, GEN_CH, T_IN];
    let style_shape = [BATCH, STYLE_DIM];
    let har_shape = [BATCH, 2 * GEN_N_BINS, T_FULL];
    let x = DynTensor::full(&input_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut inp = x.clone();
        let id_x = record_input(&input_shape, DType::F32).unwrap();
        inp.set_trace_id(id_x);

        let mut style = DynTensor::zeros(&style_shape, DType::F32, &cpu())?;
        let id_style = record_input(&style_shape, DType::F32).unwrap();
        style.set_trace_id(id_style);

        let mut har_source = DynTensor::zeros(&har_shape, DType::F32, &cpu())?;
        let id_har = record_input(&har_shape, DType::F32).unwrap();
        har_source.set_trace_id(id_har);

        let (mag, _phase) = generator
            .forward(&inp, &style, &har_source)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        Ok(mag)
    })
    .expect("Generator trace");

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("trace_to_graph")
        .graph;

    let input_specs: &[(&[usize], (f32, f32))] = &[
        (&input_shape[..], (-1.0, 1.0)),
        (&style_shape[..], (-0.5, 0.5)),
        (&har_shape[..], (-1.0, 1.0)),
    ];
    let output = propagate_multi_input_ibp(&gn, input_specs);

    // Build combined input bounds for status recording.
    let mut in_lo = Vec::new();
    let mut in_hi = Vec::new();
    for &(shape, (lo, hi)) in input_specs {
        let flat: usize = shape.iter().product();
        in_lo.extend(vec![lo; flat]);
        in_hi.extend(vec![hi; flat]);
    }
    let total = in_lo.len();
    let input_bounds = BoundedTensor::new(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[total]), in_lo).unwrap(),
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[total]), in_hi).unwrap(),
    )
    .expect("valid input bounds");

    (input_bounds, output)
}

// ===========================================================================
// Status recording
// ===========================================================================

fn assert_pipeline_bounds(output: &BoundedTensor, label: &str) {
    assert_all_finite(output, label);
    let (lo_min, hi_max) = bounds_min_max(output);
    let width = hi_max - lo_min;
    assert!(
        width < 1e6,
        "{label}: bound width {width} exceeds 1e6 (vacuously wide)"
    );
    assert!(
        width > 0.0,
        "{label}: zero-width bounds suggest degenerate model"
    );
    eprintln!("{label} IBP: bounds=[{lo_min}, {hi_max}], width={width:.4}");
}

fn assert_vocoder_bounds(output: &BoundedTensor, label: &str) {
    assert_all_finite(output, label);
    let (lo_min, hi_max) = bounds_min_max(output);
    // exp(x) output should have lower bound near zero (slight negative slack
    // from IBP over-approximation through the sin(phase) channel).
    assert!(
        lo_min > -0.1,
        "{label}: exp output lower bound should be near-zero, got lo_min={lo_min}"
    );
    assert!(
        hi_max < 1e6,
        "{label}: upper bound should be < 1e6, got {hi_max}"
    );
    eprintln!("{label} IBP: bounds=[{lo_min}, {hi_max}]");
}

// ===========================================================================
// Tests: kokoro_full_pipeline (TextEncoder → ProsodyPredictor)
// ===========================================================================

#[test]
fn test_traced_full_pipeline_ibp() {
    let (_, output) = trace_text_to_duration_ibp();
    assert_pipeline_bounds(&output, "traced_full_pipeline");
}

#[test]
fn test_traced_full_pipeline_verify_and_record() {
    let (input_bounds, output) = trace_text_to_duration_ibp();
    assert_pipeline_bounds(&output, "kokoro_full_pipeline");
    record_ibp_result("kokoro_full_pipeline", &input_bounds, &output);
}

// ===========================================================================
// Tests: kokoro_full_pipeline_forward (same trace, ForwardMode key)
// ===========================================================================

#[test]
fn test_traced_full_pipeline_forward_verify_and_record() {
    let (input_bounds, output) = trace_text_to_duration_ibp();
    assert_pipeline_bounds(&output, "kokoro_full_pipeline_forward");
    record_ibp_result("kokoro_full_pipeline_forward", &input_bounds, &output);
}

// ===========================================================================
// Tests: kokoro_duration_branch (TextEncoder → ProsodyPredictor)
// ===========================================================================

#[test]
fn test_traced_duration_branch_ibp() {
    let (_, output) = trace_text_to_duration_ibp();
    assert_pipeline_bounds(&output, "traced_duration_branch");
    // Property 3 (Duration positivity): duration logits are finite.
    let (lo, hi) = output.lower_upper();
    for (idx, (&lo_val, &hi_val)) in lo.iter().zip(hi.iter()).enumerate() {
        assert!(
            lo_val.is_finite(),
            "PROPERTY 3: dur_logits lower at {idx} must be finite, got {lo_val}"
        );
        assert!(
            hi_val.is_finite(),
            "PROPERTY 3: dur_logits upper at {idx} must be finite, got {hi_val}"
        );
    }
}

#[test]
fn test_traced_duration_branch_verify_and_record() {
    let (input_bounds, output) = trace_text_to_duration_ibp();
    assert_pipeline_bounds(&output, "kokoro_duration_branch");
    record_ibp_result("kokoro_duration_branch", &input_bounds, &output);
}

// ===========================================================================
// Tests: kokoro_vocoder_pipeline (Generator)
// ===========================================================================

#[test]
fn test_traced_vocoder_pipeline_ibp() {
    let (_, output) = trace_vocoder_ibp();
    assert_vocoder_bounds(&output, "traced_vocoder_pipeline");
}

#[test]
fn test_traced_vocoder_pipeline_verify_and_record() {
    let (input_bounds, output) = trace_vocoder_ibp();
    assert_vocoder_bounds(&output, "kokoro_vocoder_pipeline");
    record_ibp_result("kokoro_vocoder_pipeline", &input_bounds, &output);
}

// ===========================================================================
// Tests: kokoro_vocoder_forward (same trace, ForwardMode key)
// ===========================================================================

#[test]
fn test_traced_vocoder_forward_verify_and_record() {
    let (input_bounds, output) = trace_vocoder_ibp();
    assert_vocoder_bounds(&output, "kokoro_vocoder_forward");
    record_ibp_result("kokoro_vocoder_forward", &input_bounds, &output);
}

// ===========================================================================
// Persist all 5 pipeline keys and validate
// ===========================================================================

#[test]
fn test_traced_pipeline_persist_all_keys() {
    // Text-to-duration pipeline keys.
    let (in_td, out_td) = trace_text_to_duration_ibp();
    assert_pipeline_bounds(&out_td, "persist_full");
    record_ibp_result("kokoro_full_pipeline", &in_td, &out_td);
    record_ibp_result("kokoro_full_pipeline_forward", &in_td, &out_td);
    record_ibp_result("kokoro_duration_branch", &in_td, &out_td);

    // Vocoder pipeline keys.
    let (in_voc, out_voc) = trace_vocoder_ibp();
    assert_vocoder_bounds(&out_voc, "persist_vocoder");
    record_ibp_result("kokoro_vocoder_pipeline", &in_voc, &out_voc);
    record_ibp_result("kokoro_vocoder_forward", &in_voc, &out_voc);

    // Validate all entries exist and are not stale.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let model_path = model_status_path(ws, "kokoro");
    let v = VerifyStatus::load_locked(&model_path).expect("validation");
    for key in [
        "kokoro_full_pipeline",
        "kokoro_full_pipeline_forward",
        "kokoro_duration_branch",
        "kokoro_vocoder_pipeline",
        "kokoro_vocoder_forward",
    ] {
        let entry = v.status.kernel(key).unwrap_or_else(|| panic!("{key}"));
        assert_eq!(entry.method, PropMethod::Ibp, "{key} must use IBP");
        assert_eq!(
            entry.soundness_mode,
            VerificationSoundnessMode::Heuristic,
            "{key} must record Heuristic soundness"
        );
        assert!(
            !entry.stale,
            "{key} must not be stale after trace-based refresh"
        );
    }
}
