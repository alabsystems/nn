// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace-based NY compose verification for SineGen pre/post segments.
//!
//! SineGen is the harmonic source generator in Kokoro TTS. It converts an F0
//! pitch contour into sine-wave excitation signals. The pipeline is split into
//! two traceable segments separated by an eager cumsum step:
//!
//! **sinegen_pre** (segment 5a): F0 pitch → fractional phase at frame rate.
//!   Input: `f0 [B, T_frames, 1]`
//!   Operations: unsqueeze → expand → reshape (upsample) → broadcast_mul (harmonics)
//!     → mul_scalar (1/sr) → fract → interp_downsample_gpu
//!   Output: `rad_frames [B, T_frames, n_ch]`
//!
//! **sinegen_post** (segment 5b): cumulative phase + F0 → excitation signal.
//!   Inputs: `cum [B, T_frames, n_ch]`, `f0 [B, T_frames, 1]`
//!   Operations: voiced mask (expand + gt + to_dtype) → phase scaling (mul_scalar 2*pi*upp)
//!     → interp_upsample_gpu → sin → mul_scalar (sine_amp) → broadcast_mul (voiced)
//!     → linear → tanh → transpose
//!   Output: `excitation [B, 1, T_audio]`
//!
//! Part of #4186: Add compose tests for SineGen segments.
//! Part of #2218: Epic — Perfect Kokoro.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::test_utils::cpu;
use nn_core::DType;
use nn_models::{interp_downsample_gpu, interp_upsample_gpu};
use nn_verify::{trace_to_graph_model, trace_to_graph_model_multi_input, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

use super::common::bounds_min_max;
use super::common::kokoro_recording::record_ibp_result;
use super::common::kokoro_weights::assert_all_finite;

// -- Test dimensions (small for fast tests) -----------------------------------

/// Batch size.
const BATCH: usize = 1;
/// Number of time frames in the F0 pitch contour.
const T_FRAMES: usize = 4;
/// Upsample factor (product of upsample rates). Production = 60 (10*6).
/// Using 6 for test speed while exercising the upsample/downsample logic.
const UPP: usize = 6;
/// Audio-rate samples = T_FRAMES * UPP.
const T_AUDIO: usize = T_FRAMES * UPP;
/// Sampling rate in Hz.
const SR: f32 = 24000.0;
/// Number of harmonic channels (fundamental + 8 overtones).
const N_CH: usize = 9;
/// Sine amplitude scaling factor.
const SINE_AMP: f32 = 0.1;
/// F0 voiced/unvoiced threshold in Hz.
const VOICED_THRESHOLD: f64 = 10.0;

// -- Trace helpers ------------------------------------------------------------

/// Trace sinegen_pre: F0 → fractional phase at frame rate.
///
/// Replicates `compiled_kokoro_trace_fns::trace_seg_sinegen_pre` using public
/// APIs accessible from nn-verify tests.
fn trace_sinegen_pre_ibp() -> (BoundedTensor, BoundedTensor) {
    let f0_shape = [BATCH, T_FRAMES, 1];
    let f0 = DynTensor::full(&f0_shape, 200.0, DType::F32, &cpu()).unwrap();
    let device = cpu();

    let (_result, graph) = trace_graph(|| {
        let mut f0_in = f0.clone();
        let id = record_input(&f0_shape, DType::F32).unwrap();
        f0_in.set_trace_id(id);

        // Steps 1-2: upsample F0 to audio rate, expand to harmonics.
        let f0_audio = f0_in
            .unsqueeze(2)?
            .expand([BATCH, T_FRAMES, UPP, 1])?
            .reshape([BATCH, T_AUDIO, 1])?;
        let harmonics_data: Vec<f32> = (1..=N_CH).map(|h| h as f32).collect();
        let harmonics = DynTensor::from_vec(harmonics_data, &[1, 1, N_CH], &device)?;
        let freq = f0_audio.broadcast_mul(&harmonics)?;

        // Step 3: normalize and fract.
        let rad_audio = freq.mul_scalar(1.0 / f64::from(SR))?.fract()?;

        // Step 4: downsample to frame rate.
        let rad_frames = interp_downsample_gpu(&rad_audio, T_FRAMES)?;

        Ok(rad_frames)
    })
    .expect("sinegen_pre trace");

    let gn = trace_to_graph_model(&graph).expect("trace_to_graph").graph;

    // F0 input bounds: typical speech F0 range [50, 500] Hz.
    // Single-input mode expects BoundedTensor in the original tensor shape
    // (not flattened), because the trace-to-graph translator uses unbatched
    // convention (is_batched=false) and shape ops like Unsqueeze/Expand
    // operate on the full tensor dimensions.
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&f0_shape), 50.0f32),
        ArrayD::from_elem(IxDyn(&f0_shape), 500.0f32),
    )
    .expect("valid input bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    (input_bounds, output)
}

/// Trace sinegen_post: cumulative phase + F0 → excitation.
///
/// Replicates `compiled_kokoro_trace_fns::trace_seg_sinegen_post` using public
/// APIs accessible from nn-verify tests.
fn trace_sinegen_post_ibp() -> (BoundedTensor, BoundedTensor) {
    let cum_shape = [BATCH, T_FRAMES, N_CH];
    let f0_shape = [BATCH, T_FRAMES, 1];

    let cum_gpu = DynTensor::full(&cum_shape, 0.5, DType::F32, &cpu()).unwrap();
    let f0_gpu = DynTensor::full(&f0_shape, 200.0, DType::F32, &cpu()).unwrap();

    // Build a small Linear [1, N_CH] with bias [1] for SourceModule's l_linear.
    let weight = DynTensor::full(&[1, N_CH], 0.01, DType::F32, &cpu()).unwrap();
    let bias = DynTensor::full(&[1], 0.01, DType::F32, &cpu()).unwrap();
    let l_linear = Linear::new(weight, Some(bias)).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut cum_in = cum_gpu.clone();
        let id_cum = record_input(&cum_shape, DType::F32).unwrap();
        cum_in.set_trace_id(id_cum);

        let mut f0_in = f0_gpu.clone();
        let id_f0 = record_input(&f0_shape, DType::F32).unwrap();
        f0_in.set_trace_id(id_f0);

        // Voiced mask: f0 → expand to audio rate → gt(threshold) → to_dtype(F32).
        let voiced = f0_in
            .unsqueeze(2)?
            .expand([BATCH, T_FRAMES, UPP, 1])?
            .reshape([BATCH, T_AUDIO, 1])?
            .gt(VOICED_THRESHOLD)?
            .to_dtype(DType::F32)?;

        // Step 6: scale by 2*pi*upp.
        let phase_frames = cum_in.mul_scalar(std::f64::consts::TAU * UPP as f64)?;

        // Step 7: upsample phase to audio rate.
        let phase_audio = interp_upsample_gpu(&phase_frames, T_AUDIO)?;

        // Step 8: sin(phase) * sine_amp.
        let sines = phase_audio.sin()?.mul_scalar(f64::from(SINE_AMP))?;

        // SourceModule: sines * voiced → linear → tanh → transpose.
        let sine_wavs = sines.broadcast_mul(&voiced)?;
        let projected = l_linear.forward(&sine_wavs)?;
        projected.tanh()?.transpose(1, 2)
    })
    .expect("sinegen_post trace");

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("trace_to_graph")
        .graph;

    // Multi-input bounds: cum_phase in [0, 1] (fractional), F0 in [50, 500] Hz.
    let input_specs: &[(&[usize], (f32, f32))] =
        &[(&cum_shape[..], (0.0, 1.0)), (&f0_shape[..], (50.0, 500.0))];

    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for &(shape, (lo, hi)) in input_specs {
        let flat: usize = shape.iter().product();
        lower.extend(vec![lo; flat]);
        upper.extend(vec![hi; flat]);
    }
    let total = lower.len();
    let flat_input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper.clone()).unwrap(),
    )
    .expect("valid flat input bounds");

    let output = gn.propagate_ibp(&flat_input).expect("IBP propagation");

    // Build combined input bounds for status recording.
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper).unwrap(),
    )
    .expect("valid input bounds");

    (input_bounds, output)
}

// -- Bounds assertion helpers -------------------------------------------------

fn assert_sinegen_pre_bounds(output: &BoundedTensor, label: &str) {
    assert_all_finite(output, label);
    let (lo_min, hi_max) = bounds_min_max(output);
    let width = hi_max - lo_min;

    // rad_frames = fract(freq / sr), so output should be in [0, 1).
    // IBP over-approximation may widen slightly beyond [0, 1].
    assert!(
        width < 100.0,
        "{label}: bound width {width} exceeds 100.0 (vacuously wide)"
    );
    assert!(
        width > 0.0,
        "{label}: zero-width bounds suggest degenerate trace"
    );
    eprintln!("{label} IBP: bounds=[{lo_min}, {hi_max}], width={width:.4}");
}

fn assert_sinegen_post_bounds(output: &BoundedTensor, label: &str) {
    assert_all_finite(output, label);
    let (lo_min, hi_max) = bounds_min_max(output);
    let width = hi_max - lo_min;

    // Output passes through tanh, so true output is in (-1, 1).
    // IBP over-approximation may be wider, but should not be vacuous.
    assert!(
        lo_min >= -1.0 - 1e-4,
        "{label}: tanh output lower bound {lo_min} below -1.0 (tanh range violation)"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "{label}: tanh output upper bound {hi_max} above 1.0 (tanh range violation)"
    );
    assert!(
        width > 0.0,
        "{label}: zero-width bounds suggest degenerate trace"
    );
    eprintln!("{label} IBP: bounds=[{lo_min}, {hi_max}], width={width:.4}");
}

// ===========================================================================
// Tests: sinegen_pre — F0 to fractional phase
// ===========================================================================

/// IBP through sinegen_pre produces finite, non-vacuous bounds.
#[test]
fn test_traced_sinegen_pre_ibp() {
    let (_, output) = trace_sinegen_pre_ibp();
    assert_sinegen_pre_bounds(&output, "sinegen_pre");
}

/// Record sinegen_pre IBP result to the Kokoro verification status file.
#[test]
fn test_traced_sinegen_pre_verify_and_record() {
    let (input_bounds, output) = trace_sinegen_pre_ibp();
    assert_sinegen_pre_bounds(&output, "kokoro_sinegen_pre");
    record_ibp_result("kokoro_sinegen_pre", &input_bounds, &output);
}

// ===========================================================================
// Tests: sinegen_post — cumulative phase + F0 to excitation
// ===========================================================================

/// IBP through sinegen_post produces finite, non-vacuous bounds.
#[test]
fn test_traced_sinegen_post_ibp() {
    let (_, output) = trace_sinegen_post_ibp();
    assert_sinegen_post_bounds(&output, "sinegen_post");
}

/// Record sinegen_post IBP result to the Kokoro verification status file.
#[test]
fn test_traced_sinegen_post_verify_and_record() {
    let (input_bounds, output) = trace_sinegen_post_ibp();
    assert_sinegen_post_bounds(&output, "kokoro_sinegen_post");
    record_ibp_result("kokoro_sinegen_post", &input_bounds, &output);
}

// ===========================================================================
// Property tests: SineGen-specific invariants
// ===========================================================================

/// sinegen_post output passes through tanh, so bounds must be within [-1, 1].
///
/// This is a domain-specific property: the SourceModule applies tanh as its
/// final nonlinearity, bounding the excitation signal. IBP through tanh
/// should produce bounds within [-1, 1] (exact for tanh).
#[test]
fn test_sinegen_post_tanh_bound_property() {
    let (_, output) = trace_sinegen_post_ibp();
    let (lo_min, hi_max) = bounds_min_max(&output);
    assert!(
        lo_min >= -1.0 - 1e-4 && hi_max <= 1.0 + 1e-4,
        "P2 (non-clipping) for sinegen_post: output [{lo_min}, {hi_max}] must be in [-1, 1]"
    );
    eprintln!("P2 for sinegen_post: excitation bounds [{lo_min:.6}, {hi_max:.6}] within [-1, 1]");
}

/// sinegen_pre output represents fractional phase — bounds should be moderate.
///
/// The `fract()` operation produces values in [0, 1), but IBP over-approximation
/// through the preceding multiply + fract chain may widen this. We assert the
/// bounds are non-vacuous (width < 10, as the true range is at most 1.0).
#[test]
fn test_sinegen_pre_phase_range_property() {
    let (_, output) = trace_sinegen_pre_ibp();
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("sinegen_pre phase range: [{lo_min:.6}, {hi_max:.6}], width={width:.4}");
    // fract output is in [0, 1) but IBP may over-approximate through
    // the interp_downsample step. Width should still be moderate.
    assert!(
        width < 10.0,
        "sinegen_pre phase width {width} is unexpectedly wide (expected < 10.0)"
    );
}

/// Persist both sinegen keys and validate they are not stale.
#[test]
fn test_traced_sinegen_persist_all_keys() {
    // sinegen_pre
    let (in_pre, out_pre) = trace_sinegen_pre_ibp();
    assert_sinegen_pre_bounds(&out_pre, "persist_sinegen_pre");
    record_ibp_result("kokoro_sinegen_pre", &in_pre, &out_pre);

    // sinegen_post
    let (in_post, out_post) = trace_sinegen_post_ibp();
    assert_sinegen_post_bounds(&out_post, "persist_sinegen_post");
    record_ibp_result("kokoro_sinegen_post", &in_post, &out_post);

    // Validate entries exist and are not stale.
    let ws = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let model_path = nn_verify::model_status_path(ws, "kokoro");
    let v = nn_verify::VerifyStatus::load_locked(&model_path).expect("validation");
    for key in ["kokoro_sinegen_pre", "kokoro_sinegen_post"] {
        let entry = v.status.kernel(key).unwrap_or_else(|| panic!("{key}"));
        assert_eq!(
            entry.method,
            nn_verify::PropMethod::Ibp,
            "{key} must use IBP"
        );
        assert_eq!(
            entry.soundness_mode,
            nn_verify::VerificationSoundnessMode::Heuristic,
            "{key} must record Heuristic soundness"
        );
        assert!(
            !entry.stale,
            "{key} must not be stale after trace-based recording"
        );
    }
}
