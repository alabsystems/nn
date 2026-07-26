// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace-based Kokoro decoder (ISTFTNet Generator) NY verification.
//!
//! Migrates the builder-based `kokoro_decoder` and `kokoro_decoder_leaky_relu`
//! status keys to trace-based verification by tracing the real `Generator`
//! module. The traced graph captures the corrected architecture automatically,
//! eliminating topology drift when the model changes.
//!
//! Architecture: Conv1d(conv_pre) → LeakyReLU → ConvTranspose1d (upsample)
//! → ResBlock(AdaIN + Snake + Conv1d) + noise + residual → LeakyReLU
//! → Conv1d(conv_post) → split → exp(log_mag).
//!
//! Part of #2593: Migrate Kokoro TensorBlockBuilder specs to trace-based.
//! Part of #2218: Epic — Perfect Kokoro.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, TensorError};
use nn_models::kokoro_decoder::Generator;
use nn_verify::{
    model_status_path, trace_to_graph_model_multi_input, BoundedTensor, PropMethod,
    VerificationSoundnessMode, VerifyStatus,
};
use std::path::Path;

use super::common::bounds_min_max;
use super::common::kokoro_recording::record_ibp_result;
use super::common::kokoro_weights::{
    assert_all_finite, build_test_generator as build_shared_generator, propagate_multi_input_ibp,
    GEN_CH, GEN_N_BINS,
};

// -- Test dimensions ----------------------------------------------------------

const STYLE_DIM: usize = 4;
const T_IN: usize = 8;
const T_FULL: usize = 16;
const BATCH: usize = 1;

// -- Weight/model builders ----------------------------------------------------

fn build_local_generator() -> Generator {
    build_shared_generator(0.01, STYLE_DIM)
}

fn trace_generator_ibp(generator: &Generator) -> (BoundedTensor, BoundedTensor) {
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

// -- Status recording ---------------------------------------------------------

fn assert_decoder_bounds(output: &BoundedTensor, label: &str) {
    assert_all_finite(output, label);
    let (lo_min, hi_max) = bounds_min_max(output);
    // Mathematically exp(x) > 0, but IBP through narrow+exp adds slight slack
    // from the sin(phase) channel sharing conv_post output before the split.
    // Allow small negative slack from IBP over-approximation.
    assert!(
        lo_min > -0.1,
        "{label}: exp output lower bound should be near-zero, got lo_min={lo_min}"
    );
    assert!(
        hi_max < 1e6,
        "{label}: upper bound magnitude should be < 1e6, got {hi_max}"
    );
    eprintln!("{label} IBP: bounds=[{lo_min}, {hi_max}]");
}

// -- Tests: kokoro_decoder ----------------------------------------------------
//
// The trace-based "kokoro_decoder" test traces the full Generator (which
// inherently includes LeakyReLU activations in the real architecture).

#[test]
fn test_traced_decoder_ibp_propagates() {
    let generator = build_local_generator();
    let (_, output) = trace_generator_ibp(&generator);
    assert_decoder_bounds(&output, "traced_decoder");
}

#[test]
fn test_traced_decoder_verify_and_record() {
    let generator = build_local_generator();
    let (input_bounds, output) = trace_generator_ibp(&generator);
    assert_decoder_bounds(&output, "kokoro_decoder");
    record_ibp_result("kokoro_decoder", &input_bounds, &output);
}

// -- Tests: kokoro_decoder_leaky_relu -----------------------------------------
//
// In the builder-based approach, `kokoro_decoder_leaky_relu` was a separate
// graph that added LeakyReLU activations to the decoder. In the real model,
// the Generator ALWAYS has LeakyReLU (it's part of the architecture).
// The trace-based approach captures this correctly — both keys record the
// same traced graph (the real Generator with LeakyReLU).

#[test]
fn test_traced_decoder_leaky_relu_verify_and_record() {
    let generator = build_local_generator();
    let (input_bounds, output) = trace_generator_ibp(&generator);
    assert_decoder_bounds(&output, "kokoro_decoder_leaky_relu");
    record_ibp_result("kokoro_decoder_leaky_relu", &input_bounds, &output);
}

/// Persist both decoder keys and validate they are no longer stale.
#[test]
fn test_traced_decoder_persist_both_keys() {
    let generator = build_local_generator();
    let (input_bounds, output) = trace_generator_ibp(&generator);
    assert_decoder_bounds(&output, "decoder_persist");

    record_ibp_result("kokoro_decoder", &input_bounds, &output);
    record_ibp_result("kokoro_decoder_leaky_relu", &input_bounds, &output);

    // Validate both entries exist and are not stale.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let model_path = model_status_path(ws, "kokoro");
    let v = VerifyStatus::load_locked(&model_path).expect("validation");
    for key in ["kokoro_decoder", "kokoro_decoder_leaky_relu"] {
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
