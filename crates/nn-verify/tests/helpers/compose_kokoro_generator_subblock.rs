// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sub-block segmented verification for the Kokoro Generator.
//!
//! Breaks the Generator into independently-verifiable sub-blocks:
//! - Block 0: conv_pre (single Conv1d)
//! - Block 1..N: upsample stages (LeakyReLU + ConvTranspose1d + noise + ResBlocks)
//! - Block N+1: output stage (LeakyReLU + conv_post + clamp + exp/sin)
//!
//! Each sub-block is traced independently with its own variable inputs and IBP
//! propagation. Junction contracts verify that output bounds of block k are
//! contained within the assumed input bounds of block k+1.
//!
//! This resolves the [-inf, inf] bound explosion from monolithic Generator IBP
//! by proving finite bounds per sub-block and composing the results.
//!
//! Part of #2597: Generator [-inf, inf] bounds make pipeline certificate unsound.
//! Part of #2218: Epic — Perfect Kokoro.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, TensorError};
use nn_models::kokoro_decoder::Generator;
use nn_verify::{
    model_for_kernel, model_status_path, trace_to_graph_model, trace_to_graph_model_multi_input,
    BoundedTensor, PropMethod, VerificationSoundnessMode, VerifyStatus,
};
use std::path::Path;

use super::common::bounds_min_max;
use super::common::kokoro_weights::{
    assert_all_finite, build_test_generator as build_shared_generator, propagate_multi_input_ibp,
    GEN_CH, GEN_N_BINS,
};

// -- Constants ----------------------------------------------------------------

const STYLE_DIM: usize = 4;
const GEN_NEXT_CH: usize = GEN_CH / 2; // ch after one upsample stage
const BATCH: usize = 1;
const T_IN: usize = 8;
const UPSAMPLE_RATE: usize = 2;
const T_AFTER_UP: usize = T_IN * UPSAMPLE_RATE; // 16
const T_FULL: usize = T_AFTER_UP; // har_source temporal dim

// -- Builder ------------------------------------------------------------------

pub(super) fn build_test_generator() -> Generator {
    build_shared_generator(0.01, STYLE_DIM)
}

// -- Sub-block tracing helpers ------------------------------------------------

/// Trace conv_pre sub-block and return IBP output bounds.
pub(super) fn trace_conv_pre(generator: &Generator) -> BoundedTensor {
    let x_shape = [BATCH, GEN_CH, T_IN];
    let x = DynTensor::full(&x_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(&x_shape, DType::F32).unwrap());
        let out = generator
            .forward_conv_pre(&inp)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        Ok(out)
    })
    .expect("conv_pre trace");

    let gn = trace_to_graph_model(&graph)
        .expect("conv_pre trace_to_graph")
        .graph;
    let input_bounds = super::common::kokoro_weights::asymmetric_bounds(&x_shape, -1.0, 1.0);
    gn.propagate_ibp(&input_bounds).expect("conv_pre IBP")
}

/// Trace one upsample stage and return IBP output bounds.
///
/// Takes 3 variable inputs: hidden state h, style embedding, harmonic source.
pub(super) fn trace_upsample_stage(
    generator: &Generator,
    stage: usize,
    h_bounds: (f32, f32),
) -> BoundedTensor {
    let h_shape = [BATCH, GEN_CH, T_IN];
    let style_shape = [BATCH, STYLE_DIM];
    let har_shape = [BATCH, 2 * GEN_N_BINS, T_FULL];
    let h = DynTensor::full(&h_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut h_t = h.clone();
        h_t.set_trace_id(record_input(&h_shape, DType::F32).unwrap());

        let mut style = DynTensor::zeros(&style_shape, DType::F32, &cpu())?;
        style.set_trace_id(record_input(&style_shape, DType::F32).unwrap());

        let mut har = DynTensor::zeros(&har_shape, DType::F32, &cpu())?;
        har.set_trace_id(record_input(&har_shape, DType::F32).unwrap());

        let out = generator
            .forward_upsample_stage(stage, &h_t, &style, &har)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        Ok(out)
    })
    .expect("upsample_stage trace");

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("upsample trace_to_graph")
        .graph;
    propagate_multi_input_ibp(
        &gn,
        &[
            (&h_shape[..], h_bounds),
            (&style_shape[..], (-0.5, 0.5)),
            (&har_shape[..], (-1.0, 1.0)),
        ],
    )
}

/// Trace output stage and return IBP output bounds.
pub(super) fn trace_output_stage(generator: &Generator, h_bounds: (f32, f32)) -> BoundedTensor {
    // After upsample: channels = GEN_NEXT_CH, temporal = T_AFTER_UP (+1 for reflection pad)
    let t_out = T_AFTER_UP + 1; // reflection_pad1d adds 1 on the left for last stage
    let h_shape = [BATCH, GEN_NEXT_CH, t_out];
    let h = DynTensor::full(&h_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut h_t = h.clone();
        h_t.set_trace_id(record_input(&h_shape, DType::F32).unwrap());
        let (mag, _phase) = generator
            .forward_output_stage(&h_t)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        Ok(mag)
    })
    .expect("output_stage trace");

    let gn = trace_to_graph_model(&graph)
        .expect("output trace_to_graph")
        .graph;
    let input_bounds =
        super::common::kokoro_weights::asymmetric_bounds(&h_shape, h_bounds.0, h_bounds.1);
    gn.propagate_ibp(&input_bounds).expect("output IBP")
}

/// Trace output conv_post (pre-activation) and return IBP bounds for raw log_mag.
///
/// Returns the raw conv_post log_magnitude bounds BEFORE clamp/exp.
/// Used to check J3_MAGNITUDE junction contract.
pub(super) fn trace_output_conv_post_log_mag(
    generator: &Generator,
    h_bounds: (f32, f32),
) -> BoundedTensor {
    let t_out = T_AFTER_UP + 1;
    let h_shape = [BATCH, GEN_NEXT_CH, t_out];
    let h = DynTensor::full(&h_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut h_t = h.clone();
        h_t.set_trace_id(record_input(&h_shape, DType::F32).unwrap());
        let (log_mag, _phase_raw) = generator
            .forward_output_conv_post(&h_t)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        Ok(log_mag)
    })
    .expect("output_conv_post log_mag trace");

    let gn = trace_to_graph_model(&graph)
        .expect("output_conv_post trace_to_graph")
        .graph;
    let input_bounds =
        super::common::kokoro_weights::asymmetric_bounds(&h_shape, h_bounds.0, h_bounds.1);
    gn.propagate_ibp(&input_bounds)
        .expect("output_conv_post IBP")
}

/// Trace output conv_post (pre-activation) and return IBP bounds for raw phase.
///
/// Returns the raw conv_post phase_raw bounds BEFORE sin().
/// Used to check J3B_PHASE junction contract.
pub(super) fn trace_output_conv_post_phase(
    generator: &Generator,
    h_bounds: (f32, f32),
) -> BoundedTensor {
    let t_out = T_AFTER_UP + 1;
    let h_shape = [BATCH, GEN_NEXT_CH, t_out];
    let h = DynTensor::full(&h_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut h_t = h.clone();
        h_t.set_trace_id(record_input(&h_shape, DType::F32).unwrap());
        let (_log_mag, phase_raw) = generator
            .forward_output_conv_post(&h_t)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        Ok(phase_raw)
    })
    .expect("output_conv_post phase trace");

    let gn = trace_to_graph_model(&graph)
        .expect("output_conv_post trace_to_graph")
        .graph;
    let input_bounds =
        super::common::kokoro_weights::asymmetric_bounds(&h_shape, h_bounds.0, h_bounds.1);
    gn.propagate_ibp(&input_bounds)
        .expect("output_conv_post IBP")
}

// -- Status recording ---------------------------------------------------------

pub(super) fn record_subblock(status_key: &str, output: &BoundedTensor) {
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let model = model_for_kernel(status_key);
    let model_path = model_status_path(ws, model);
    let locked = match VerifyStatus::load_locked(&model_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("WARN: could not acquire lock for {status_key}: {e} — skipping recording");
            return;
        }
    };
    let mut locked = locked;

    let (lo, hi) = bounds_min_max(output);
    let (lo_arr, _) = output.lower_upper();
    let out_shape = [lo_arr.len()];

    locked
        .status
        .record_pipeline(
            status_key,
            PropMethod::Ibp,
            lo,
            hi,
            lo,
            hi,
            &out_shape,
            VerificationSoundnessMode::Sound,
            Some(output.shape()), // output shape as proxy — no separate input_bounds
        )
        .expect("record_pipeline");
    locked.save().expect("save status");
    eprintln!("Recorded {status_key}: bounds=[{lo}, {hi}]");
}

// -- Tests --------------------------------------------------------------------

/// AC1: conv_pre sub-block produces finite IBP bounds.
#[test]
fn test_generator_subblock_conv_pre() {
    let generator = build_test_generator();
    let output = trace_conv_pre(&generator);
    assert_all_finite(&output, "conv_pre");
    // Conv1d sub-block: width ~1.1 at test scale (D=8, fill=0.01).
    super::common::assert_bounds_width(&output, 100.0, "conv_pre");
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("conv_pre IBP bounds: [{lo}, {hi}]");
}

/// AC2: upsample_stage[0] sub-block produces finite IBP bounds.
#[test]
fn test_generator_subblock_upsample_stage_0() {
    let generator = build_test_generator();
    let conv_pre_out = trace_conv_pre(&generator);
    let (lo, hi) = bounds_min_max(&conv_pre_out);
    eprintln!("conv_pre output bounds for junction: [{lo}, {hi}]");

    let output = trace_upsample_stage(&generator, 0, (lo, hi));
    assert_all_finite(&output, "upsample_stage_0");
    // Upsample stage: width ~13.6 at test scale. Allow up to 1000 for margin.
    super::common::assert_bounds_width(&output, 1000.0, "upsample_stage_0");
    let (lo2, hi2) = bounds_min_max(&output);
    eprintln!("upsample_stage_0 IBP bounds: [{lo2}, {hi2}]");
}

/// AC3: output_stage sub-block produces finite, bounded IBP bounds.
/// sin(x) ∈ [-1, 1] and exp(clamp(x, -88, 88)) ∈ [exp(-88), exp(88)].
#[test]
fn test_generator_subblock_output_stage() {
    let generator = build_test_generator();
    let conv_pre_out = trace_conv_pre(&generator);
    let (lo1, hi1) = bounds_min_max(&conv_pre_out);
    let upsample_out = trace_upsample_stage(&generator, 0, (lo1, hi1));
    let (lo2, hi2) = bounds_min_max(&upsample_out);
    eprintln!("upsample output bounds for junction: [{lo2}, {hi2}]");

    let output = trace_output_stage(&generator, (lo2, hi2));
    assert_all_finite(&output, "output_stage");
    // Output stage: width ~1.0 at test scale (exp+sin through clamp).
    super::common::assert_bounds_width(&output, 100.0, "output_stage");
    let (lo3, hi3) = bounds_min_max(&output);
    eprintln!("output_stage IBP bounds: [{lo3}, {hi3}]");

    // Bounds must be tight — IBP through narrow+clamp+exp may have slight slack
    // from the sin(phase) channel sharing the conv_post output before narrow.
    assert!(
        lo3 > -0.1,
        "output lower bound should be near-zero, got lo={lo3}"
    );
    assert!(hi3 < 1e20, "output upper bound should be < 1e20, got {hi3}");
}

/// AC4: Full segmented pipeline — all sub-blocks produce finite bounds and
/// junction contracts hold (output of block k feeds block k+1).
#[test]
fn test_generator_subblock_full_pipeline() {
    let generator = build_test_generator();

    // Block 0: conv_pre
    let conv_pre_out = trace_conv_pre(&generator);
    assert_all_finite(&conv_pre_out, "pipeline/conv_pre");
    let (lo0, hi0) = bounds_min_max(&conv_pre_out);
    eprintln!("Pipeline block 0 (conv_pre): [{lo0}, {hi0}]");

    // Block 1..N: upsample stages — with per-layer expansion factor tracking (#2594 AC2).
    let mut prev_bounds = (lo0, hi0);
    let mut prev_width = hi0 - lo0;
    for stage in 0..generator.num_stages() {
        let stage_out = trace_upsample_stage(&generator, stage, prev_bounds);
        assert_all_finite(&stage_out, &format!("pipeline/upsample_{stage}"));
        let (lo, hi) = bounds_min_max(&stage_out);
        let width = hi - lo;
        let expansion = if prev_width > 1e-10 {
            width / prev_width
        } else {
            1.0
        };
        eprintln!(
            "Pipeline block {} (upsample_{stage}): [{lo}, {hi}] width={width:.3} expansion={expansion:.2}x",
            stage + 1
        );
        // Per-layer expansion should not exceed 100x — detects bound explosion early.
        assert!(
            expansion < 100.0,
            "upsample_{stage} expansion {expansion:.2}x exceeds 100x threshold"
        );
        prev_bounds = (lo, hi);
        prev_width = width;
    }

    // Block N+1: output stage
    let output = trace_output_stage(&generator, prev_bounds);
    assert_all_finite(&output, "pipeline/output");
    // Full pipeline: width must be bounded (was [-inf, inf] monolithically).
    super::common::assert_bounds_width(&output, 1e6, "pipeline/output");
    let (lo_final, hi_final) = bounds_min_max(&output);
    eprintln!("Pipeline block final (output): [{lo_final}, {hi_final}]");
    // Output should be near-zero lower (exp > 0 mathematically; IBP may add slack)
    assert!(
        lo_final > -0.1,
        "output lower bound should be near-zero, got {lo_final}"
    );

    // Record the segmented pipeline result
    record_subblock("kokoro_generator_subblock_pipeline", &output);
    eprintln!(
        "Generator sub-block pipeline: all {} blocks have finite bounds. Final: [{lo_final}, {hi_final}]",
        generator.num_stages() + 2
    );
}

/// Compare: monolithic vs segmented Generator IBP to demonstrate tightening.
#[test]
fn test_generator_subblock_vs_monolithic() {
    let generator = build_test_generator();

    // Monolithic: trace full Generator with 3 inputs
    let x_shape = [BATCH, GEN_CH, T_IN];
    let style_shape = [BATCH, STYLE_DIM];
    let har_shape = [BATCH, 2 * GEN_N_BINS, T_FULL];
    let x = DynTensor::full(&x_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(&x_shape, DType::F32).unwrap());
        let mut style = DynTensor::zeros(&style_shape, DType::F32, &cpu())?;
        style.set_trace_id(record_input(&style_shape, DType::F32).unwrap());
        let mut har = DynTensor::zeros(&har_shape, DType::F32, &cpu())?;
        har.set_trace_id(record_input(&har_shape, DType::F32).unwrap());
        let (mag, _phase) = generator
            .forward(&inp, &style, &har)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        Ok(mag)
    })
    .expect("monolithic trace");

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("monolithic trace_to_graph")
        .graph;
    let mono_out = propagate_multi_input_ibp(
        &gn,
        &[
            (&x_shape[..], (-1.0, 1.0)),
            (&style_shape[..], (-0.5, 0.5)),
            (&har_shape[..], (-1.0, 1.0)),
        ],
    );
    let (mono_lo, mono_hi) = bounds_min_max(&mono_out);
    eprintln!("Monolithic IBP: [{mono_lo}, {mono_hi}]");

    // Segmented pipeline
    let conv_pre_out = trace_conv_pre(&generator);
    let (lo0, hi0) = bounds_min_max(&conv_pre_out);
    let up_out = trace_upsample_stage(&generator, 0, (lo0, hi0));
    let (lo1, hi1) = bounds_min_max(&up_out);
    let seg_out = trace_output_stage(&generator, (lo1, hi1));
    let (seg_lo, seg_hi) = bounds_min_max(&seg_out);
    eprintln!("Segmented IBP: [{seg_lo}, {seg_hi}]");

    // At minimum, segmented must be finite when monolithic might not be
    assert!(
        seg_lo.is_finite() && seg_hi.is_finite(),
        "segmented bounds must be finite"
    );

    // Log comparison
    let mono_width = if mono_lo.is_finite() && mono_hi.is_finite() {
        mono_hi - mono_lo
    } else {
        f32::INFINITY
    };
    let seg_width = seg_hi - seg_lo;
    eprintln!("Bound width comparison: monolithic={mono_width}, segmented={seg_width}");
    if mono_width.is_finite() {
        let ratio = mono_width / seg_width;
        eprintln!("Monolithic/Segmented width ratio: {ratio:.2}x");
    } else {
        eprintln!("Monolithic bounds are infinite — segmented resolves this");
    }
}
