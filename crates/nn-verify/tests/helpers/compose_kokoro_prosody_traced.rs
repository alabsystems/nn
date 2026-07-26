// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace-based Kokoro ProsodyPredictor NY verification.
//!
//! Migrates the builder-based `kokoro_prosody_single_block`,
//! `kokoro_prosody_three_blocks`, and `kokoro_prosody_t4` status keys to
//! trace-based verification by tracing the real `ProsodyPredictor` module.
//!
//! Architecture: N × (BiLSTM + AdaLayerNorm) DurationEncoder blocks
//! → duration projection → final BiLSTM.
//!
//! Two inputs: text_features `[B, d_model, T]`, style `[B, style_dim]`.
//!
//! Part of #2593: Migrate Kokoro TensorBlockBuilder specs to trace-based.
//! Part of #2218: Epic — Perfect Kokoro.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_models::kokoro_tts::ProsodyPredictor;
use nn_verify::{
    model_status_path, trace_to_graph_model_multi_input, BoundedTensor, PropMethod,
    VerificationSoundnessMode, VerifyStatus,
};
use std::collections::HashMap;
use std::path::Path;

use super::common::bounds_min_max;
use super::common::kokoro_recording::record_ibp_result;
use super::common::kokoro_weights::{
    assert_all_finite, bilstm_weights, propagate_multi_input_ibp, z,
};

// -- Test dimensions ----------------------------------------------------------

const D_EN: usize = 8;
const STYLE_DIM: usize = 4;
const MAX_DUR: usize = 50;
const BATCH: usize = 1;

/// Build minimal ProsodyPredictor weights for `n_layers` DurationEncoder blocks.
fn prosody_weights(n_layers: usize) -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let lstm_input = D_EN + STYLE_DIM;

    for i in 0..n_layers {
        bilstm_weights(&mut m, &format!("duration.lstms.{i}"), lstm_input, D_EN / 2);
        // AdaLayerNorm (norm + style projection)
        z(&mut m, &format!("duration.norms.{i}.norm.weight"), &[D_EN]);
        z(&mut m, &format!("duration.norms.{i}.norm.bias"), &[D_EN]);
        z(
            &mut m,
            &format!("duration.norms.{i}.fc.weight"),
            &[2 * D_EN, STYLE_DIM],
        );
        z(&mut m, &format!("duration.norms.{i}.fc.bias"), &[2 * D_EN]);
    }

    // Duration projection
    z(&mut m, "duration.duration_proj.weight", &[MAX_DUR, D_EN]);
    z(&mut m, "duration.duration_proj.bias", &[MAX_DUR]);

    // Final duration BiLSTM
    bilstm_weights(&mut m, "lstm", lstm_input, D_EN / 2);

    m
}

// -- Trace + IBP helpers ------------------------------------------------------

/// Trace ProsodyPredictor and propagate IBP bounds.
///
/// Returns `(input_bounds, output_bounds)` for status recording.
fn trace_prosody_ibp(n_layers: usize, seq_len: usize) -> (BoundedTensor, BoundedTensor) {
    let weights = prosody_weights(n_layers);
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let prosody = ProsodyPredictor::load(&vb, D_EN, STYLE_DIM, n_layers, MAX_DUR)
        .expect("ProsodyPredictor::load");

    let text_shape = [BATCH, D_EN, seq_len];
    let style_shape = [BATCH, STYLE_DIM];
    let text_features = DynTensor::full(&text_shape, 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.05, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut text = text_features.clone();
        let id_text = record_input(&text_shape, DType::F32).unwrap();
        text.set_trace_id(id_text);
        let mut sty = style.clone();
        let id_style = record_input(&style_shape, DType::F32).unwrap();
        sty.set_trace_id(id_style);
        let (dur_logits, _features) = prosody.forward(&text, &sty)?;
        Ok(dur_logits)
    })
    .expect("ProsodyPredictor trace");

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("trace_to_graph")
        .graph;

    let input_specs: &[(&[usize], (f32, f32))] = &[
        (&text_shape[..], (-1.0, 1.0)),
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

// -- Status recording (delegates to shared helper, Part of #2623) -------------

fn assert_prosody_bounds(output: &BoundedTensor, label: &str) {
    assert_all_finite(output, label);
    let (lo_min, hi_max) = bounds_min_max(output);
    let max_width = hi_max - lo_min;
    assert!(
        max_width < 1e6,
        "{label}: bound width {max_width} exceeds 1e6 (vacuously wide)"
    );
    assert!(
        max_width > 0.0,
        "{label}: zero-width bounds suggest degenerate model"
    );
    eprintln!("{label} IBP: bounds=[{lo_min}, {hi_max}], width={max_width:.4}");
}

// -- Tests: kokoro_prosody_single_block (1 DurationEncoder block, T=1) --------

#[test]
fn test_traced_prosody_single_block_ibp() {
    let (_, output) = trace_prosody_ibp(1, 1);
    assert_prosody_bounds(&output, "prosody_single_block");
}

#[test]
fn test_traced_prosody_single_block_verify_and_record() {
    let (input_bounds, output) = trace_prosody_ibp(1, 1);
    assert_prosody_bounds(&output, "kokoro_prosody_single_block");
    record_ibp_result("kokoro_prosody_single_block", &input_bounds, &output);
}

// -- Tests: kokoro_prosody_three_blocks (3 DurationEncoder blocks, T=1) -------

#[test]
fn test_traced_prosody_three_blocks_ibp() {
    let (_, output) = trace_prosody_ibp(3, 1);
    assert_prosody_bounds(&output, "prosody_three_blocks");
}

#[test]
fn test_traced_prosody_three_blocks_verify_and_record() {
    let (input_bounds, output) = trace_prosody_ibp(3, 1);
    assert_prosody_bounds(&output, "kokoro_prosody_three_blocks");
    record_ibp_result("kokoro_prosody_three_blocks", &input_bounds, &output);
}

// -- Tests: kokoro_prosody_t4 (3 blocks, T=4 LSTM unrolling) -----------------

#[test]
fn test_traced_prosody_t4_ibp() {
    let (_, output) = trace_prosody_ibp(3, 4);
    assert_prosody_bounds(&output, "prosody_t4");
}

#[test]
fn test_traced_prosody_t4_verify_and_record() {
    let (input_bounds, output) = trace_prosody_ibp(3, 4);
    assert_prosody_bounds(&output, "kokoro_prosody_t4");
    record_ibp_result("kokoro_prosody_t4", &input_bounds, &output);
}

/// Persist all 3 prosody keys and validate they are no longer stale.
#[test]
fn test_traced_prosody_persist_all_keys() {
    // Single block (T=1)
    let (in_1b, out_1b) = trace_prosody_ibp(1, 1);
    assert_prosody_bounds(&out_1b, "persist_single");
    record_ibp_result("kokoro_prosody_single_block", &in_1b, &out_1b);

    // Three blocks (T=1)
    let (in_3b, out_3b) = trace_prosody_ibp(3, 1);
    assert_prosody_bounds(&out_3b, "persist_three");
    record_ibp_result("kokoro_prosody_three_blocks", &in_3b, &out_3b);

    // Three blocks, T=4
    let (in_t4, out_t4) = trace_prosody_ibp(3, 4);
    assert_prosody_bounds(&out_t4, "persist_t4");
    record_ibp_result("kokoro_prosody_t4", &in_t4, &out_t4);

    // Validate all entries exist and are not stale.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let model_path = model_status_path(ws, "kokoro");
    let v = VerifyStatus::load_locked(&model_path).expect("validation");
    for key in [
        "kokoro_prosody_single_block",
        "kokoro_prosody_three_blocks",
        "kokoro_prosody_t4",
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
