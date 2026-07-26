// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace-based scaled Kokoro pipeline NY verification.
//!
//! Migrates the builder-based `kokoro_scaled_d=16`, `kokoro_scaled_d=32`,
//! and `kokoro_scaled_d=64` status keys to trace-based verification by
//! tracing the real `Generator` module at each scale.
//!
//! The builder-based version uses a simplified architecture (Conv1d + ReLU +
//! Linear for text encoder, simplified vocoder). The trace-based version uses
//! the REAL Generator architecture (Conv1d(conv_pre) → LeakyReLU →
//! ConvTranspose1d → ResBlock(AdaIN + Snake + Conv1d) + noise → conv_post →
//! split → exp(log_mag)), which is the part of the pipeline most likely to
//! diverge from the builder approximation.
//!
//! Part of #2593: Migrate Kokoro TensorBlockBuilder specs to trace-based.
//! Part of #2218: Epic — Perfect Kokoro.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, TensorError, VarBuilder};
use nn_models::kokoro_decoder::Generator;
use nn_models::KokoroConfig;
use nn_verify::{
    model_status_path, trace_to_graph_model_multi_input, BoundedTensor, PropMethod,
    VerificationSoundnessMode, VerifyStatus,
};
use std::path::Path;

use super::common::bounds_min_max;
use super::common::kokoro_recording::record_ibp_result;
use super::common::kokoro_weights::{
    assert_all_finite, generator_weights_scaled, propagate_multi_input_ibp,
};

// -- Scaled dimension configurations ------------------------------------------

/// Dimensions for a scaled Generator test at a given D value.
struct ScaledDims {
    /// Generator initial channels (= D).
    gen_ch: usize,
    /// n_fft for iSTFT output.
    n_fft: usize,
    /// Resblock kernel size.
    resblock_kernel: usize,
    /// Style embedding dimension.
    style_dim: usize,
    /// Input time steps.
    t_in: usize,
    /// Full time (after upsample by 2x).
    t_full: usize,
}

impl ScaledDims {
    fn d16() -> Self {
        Self {
            gen_ch: 16,
            n_fft: 4,
            resblock_kernel: 3,
            style_dim: 4,
            t_in: 4,
            t_full: 8,
        }
    }

    fn d32() -> Self {
        Self {
            gen_ch: 32,
            n_fft: 4,
            resblock_kernel: 3,
            style_dim: 8,
            t_in: 4,
            t_full: 8,
        }
    }

    fn d64() -> Self {
        Self {
            gen_ch: 64,
            n_fft: 4,
            resblock_kernel: 3,
            style_dim: 8,
            t_in: 4,
            t_full: 8,
        }
    }

    fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

// -- Model builder + trace ----------------------------------------------------

fn build_scaled_generator(dims: &ScaledDims) -> Generator {
    let weights = generator_weights_scaled(
        dims.gen_ch,
        dims.n_fft,
        dims.resblock_kernel,
        0.01,
        dims.style_dim,
    );
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let mut config = KokoroConfig::default();
    config.upsample_rates = vec![2];
    config.upsample_kernel_sizes = vec![4];
    config.resblock_kernel_sizes = vec![dims.resblock_kernel];
    config.resblock_dilations = vec![vec![1]];
    config.gen_initial_channels = dims.gen_ch;
    config.style_dim = dims.style_dim;
    config.n_fft = dims.n_fft;
    Generator::load(&vb, &config).expect("Generator::load")
}

fn trace_scaled_ibp(dims: &ScaledDims) -> (BoundedTensor, BoundedTensor) {
    let generator = build_scaled_generator(dims);
    let batch = 1;
    let input_shape = [batch, dims.gen_ch, dims.t_in];
    let style_shape = [batch, dims.style_dim];
    let har_shape = [batch, 2 * dims.n_bins(), dims.t_full];
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

fn assert_scaled_bounds(output: &BoundedTensor, label: &str) {
    assert_all_finite(output, label);
    let (lo_min, hi_max) = bounds_min_max(output);
    assert!(
        hi_max - lo_min < 1e6,
        "{label}: bound width {} exceeds 1e6 (vacuously wide)",
        hi_max - lo_min
    );
    assert!(
        hi_max - lo_min > 0.0,
        "{label}: zero-width bounds suggest degenerate model"
    );
    eprintln!(
        "{label} IBP: bounds=[{lo_min}, {hi_max}], width={:.4}",
        hi_max - lo_min
    );
}

// -- Tests: D=16 (first scaling step) -----------------------------------------

#[test]
fn test_traced_scaled_d16_ibp() {
    let (_, output) = trace_scaled_ibp(&ScaledDims::d16());
    assert_scaled_bounds(&output, "traced_d16");
}

#[test]
fn test_traced_scaled_d16_verify_and_record() {
    let (input_bounds, output) = trace_scaled_ibp(&ScaledDims::d16());
    assert_scaled_bounds(&output, "kokoro_scaled_d=16");
    record_ibp_result("kokoro_scaled_d=16", &input_bounds, &output);
}

// -- Tests: D=32 (meaningful step toward production) --------------------------

#[test]
fn test_traced_scaled_d32_ibp() {
    let (_, output) = trace_scaled_ibp(&ScaledDims::d32());
    assert_scaled_bounds(&output, "traced_d32");
}

#[test]
fn test_traced_scaled_d32_verify_and_record() {
    let (input_bounds, output) = trace_scaled_ibp(&ScaledDims::d32());
    assert_scaled_bounds(&output, "kokoro_scaled_d=32");
    record_ibp_result("kokoro_scaled_d=32", &input_bounds, &output);
}

// -- Tests: D=64 (requires per-layer CROWN for tight bounds) ------------------

#[test]
fn test_traced_scaled_d64_ibp() {
    let (_, output) = trace_scaled_ibp(&ScaledDims::d64());
    assert_scaled_bounds(&output, "traced_d64");
}

#[test]
fn test_traced_scaled_d64_verify_and_record() {
    let (input_bounds, output) = trace_scaled_ibp(&ScaledDims::d64());
    assert_scaled_bounds(&output, "kokoro_scaled_d=64");
    record_ibp_result("kokoro_scaled_d=64", &input_bounds, &output);
}

// -- CROWN on output stage (Level 3 pipeline validation) ----------------------

/// Trace only the output stage of the Generator and run CROWN + IBP.
///
/// This tests the CROWN pipeline on the tractable output sub-block
/// (LeakyReLU → conv_post → split → clamp → exp/sin) without the intractable
/// upsample stages. Input bounds are conservative [-1, 1].
///
/// Part of #2599: Level 3 — CROWN on production weights pipeline validation.
fn trace_output_stage_crown(
    dims: &ScaledDims,
) -> (nn_verify::GraphNetwork, BoundedTensor, BoundedTensor) {
    let generator = build_scaled_generator(dims);
    let batch = 1;
    let ch_final = dims.gen_ch / 2;
    let t_out = dims.t_in * 2 + 1;

    let h_shape = [batch, ch_final, t_out];
    let h = DynTensor::full(&h_shape, 0.1, DType::F32, &cpu()).unwrap();
    let (_result, graph) = trace_graph(|| {
        let mut h_t = h.clone();
        h_t.set_trace_id(record_input(&h_shape, DType::F32).expect("trace active"));
        let (mag, _phase) = generator
            .forward_output_stage(&h_t)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        Ok(mag)
    })
    .expect("output_stage trace");
    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("output trace_to_graph")
        .graph;

    let input_bounds = BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&h_shape), -1.0f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&h_shape), 1.0f32),
    )
    .expect("valid bounds");

    let ibp_output = gn.propagate_ibp(&input_bounds).expect("output IBP");
    assert_all_finite(&ibp_output, "output_stage IBP");

    (gn, input_bounds, ibp_output)
}

#[test]
fn test_traced_d16_output_stage_crown() {
    let (gn, input_bounds, ibp_output) = trace_output_stage_crown(&ScaledDims::d16());
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);

    let (method, crown_output, _note) =
        nn_verify::propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN");
    assert_all_finite(&crown_output, "output_stage CROWN d16");
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);

    let ibp_width = ibp_hi - ibp_lo;
    let crown_width = crown_hi - crown_lo;
    let ratio = if crown_width > 0.0 {
        ibp_width / crown_width
    } else {
        1.0
    };

    eprintln!(
        "D=16 output stage ({method:?}): IBP=[{ibp_lo:.6}, {ibp_hi:.6}], \
         CROWN=[{crown_lo:.6}, {crown_hi:.6}], tightening={ratio:.2}x"
    );
    assert!(ratio >= 1.0, "CROWN must be at least as tight as IBP");
}

#[test]
fn test_traced_d32_output_stage_crown() {
    let (gn, input_bounds, ibp_output) = trace_output_stage_crown(&ScaledDims::d32());
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);

    let (method, crown_output, _note) =
        nn_verify::propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN");
    assert_all_finite(&crown_output, "output_stage CROWN d32");
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);

    let ibp_width = ibp_hi - ibp_lo;
    let crown_width = crown_hi - crown_lo;
    let ratio = if crown_width > 0.0 {
        ibp_width / crown_width
    } else {
        1.0
    };

    eprintln!(
        "D=32 output stage ({method:?}): IBP=[{ibp_lo:.6}, {ibp_hi:.6}], \
         CROWN=[{crown_lo:.6}, {crown_hi:.6}], tightening={ratio:.2}x"
    );
    assert!(ratio >= 1.0, "CROWN must be at least as tight as IBP");
}

// -- Persist all 3 scaled keys and validate -----------------------------------

#[test]
fn test_traced_scaled_persist_all_keys() {
    let dims_configs: &[(&str, ScaledDims)] = &[
        ("kokoro_scaled_d=16", ScaledDims::d16()),
        ("kokoro_scaled_d=32", ScaledDims::d32()),
        ("kokoro_scaled_d=64", ScaledDims::d64()),
    ];

    for (key, dims) in dims_configs {
        let (input_bounds, output) = trace_scaled_ibp(dims);
        assert_scaled_bounds(&output, key);
        record_ibp_result(key, &input_bounds, &output);
    }

    // Validate all entries exist and are not stale.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let model_path = model_status_path(ws, "kokoro");
    let v = VerifyStatus::load_locked(&model_path).expect("validation");
    for key in [
        "kokoro_scaled_d=16",
        "kokoro_scaled_d=32",
        "kokoro_scaled_d=64",
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
