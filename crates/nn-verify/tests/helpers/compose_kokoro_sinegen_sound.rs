// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Sound-promotion compose tests for SineGen pre/post verification entries.
//!
//! The existing `compose_kokoro_sinegen.rs` records sinegen_pre and sinegen_post
//! as `Heuristic` because `record_ibp_result` hardcodes that soundness mode.
//! However, the SineGen pipeline contains NO normalization layers (InstanceNorm,
//! AdaIN, RmsNorm, LayerNorm). The operations are:
//!
//! - sinegen_pre: unsqueeze → expand → reshape → broadcast_mul → mul_scalar → fract → downsample
//! - sinegen_post: voiced mask (gt + to_dtype), mul_scalar, upsample, sin, mul_scalar,
//!   broadcast_mul, linear, tanh, transpose
//!
//! None of these require heuristic normalization approximation. The IBP bounds
//! are provably sound via standard interval arithmetic. This file re-verifies
//! both segments with explicit `Sound` soundness classification.
//!
//! Additionally, this file adds:
//!   - Tighter input bounds (F0 [50, 400] Hz — human speech range)
//!   - Output width non-vacuity assertions
//!   - TensorBlockBuilder-based proxy graph for sinegen_post core (linear + tanh)
//!   - CROWN propagation test for the proxy graph
//!
//! Part of #4186: SineGen verification sound promotion.
//! Part of Epic #3351 (Absolutely Best Kokoro).

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::test_utils::cpu;
use nn_core::DType;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_models::{interp_downsample_gpu, interp_upsample_gpu};
use nn_verify::{
    trace_to_graph_model, trace_to_graph_model_multi_input, BoundedTensor, NormBoundsMode,
    TensorParamBinding, VerificationSoundnessMode, VerifyConfig,
};
use ndarray::{ArrayD, IxDyn};

use super::common::bounds_min_max;
use super::common::kokoro_recording::record_ibp_result_with_soundness;
use super::common::kokoro_weights::assert_all_finite;
use super::common::verify_and_assert_with_config;

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

/// Vacuous width threshold — bounds wider than this are meaningless.
const VACUOUS_THRESHOLD: f32 = 200.0;

/// Weight magnitude for synthetic proxy graph weights.
const WEIGHT_MAG: f32 = 0.01;

// -- Conservative config (produces Sound for non-norm pipelines) --------------

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// -- Trace helpers (reused from compose_kokoro_sinegen.rs) --------------------

/// Trace sinegen_pre with specified F0 bounds.
fn trace_sinegen_pre_with_bounds(f0_lo: f32, f0_hi: f32) -> (BoundedTensor, BoundedTensor) {
    let f0_shape = [BATCH, T_FRAMES, 1];
    let f0_mid = f32::midpoint(f0_lo, f0_hi);
    let f0 = DynTensor::full(&f0_shape, f64::from(f0_mid), DType::F32, &cpu()).unwrap();
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

    // F0 input bounds. The graph was traced from the structured `[BATCH,
    // T_FRAMES, 1]` F0 contour and its first op is `unsqueeze(2)`, which inserts
    // an axis into that 3D shape. Propagating a flattened 1D tensor here makes
    // NY see a 1D input and reject `unsqueeze` axis 2 ("axis 2 out of range for
    // 2D tensor"). Feed the structured bounds — matching the non-sound
    // `trace_sinegen_pre_ibp` helper in compose_kokoro_sinegen.rs.
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&f0_shape), f0_lo),
        ArrayD::from_elem(IxDyn(&f0_shape), f0_hi),
    )
    .expect("valid input bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    (input_bounds, output)
}

/// Trace sinegen_post with specified bounds for cum_phase and F0.
fn trace_sinegen_post_with_bounds(
    cum_lo: f32,
    cum_hi: f32,
    f0_lo: f32,
    f0_hi: f32,
) -> (BoundedTensor, BoundedTensor) {
    let cum_shape = [BATCH, T_FRAMES, N_CH];
    let f0_shape = [BATCH, T_FRAMES, 1];

    let cum_gpu = DynTensor::full(&cum_shape, 0.5, DType::F32, &cpu()).unwrap();
    let f0_gpu = DynTensor::full(&f0_shape, 200.0, DType::F32, &cpu()).unwrap();

    // Build a small Linear [1, N_CH] with bias [1] for SourceModule's l_linear.
    let weight = DynTensor::full(&[1, N_CH], f64::from(WEIGHT_MAG), DType::F32, &cpu()).unwrap();
    let bias = DynTensor::full(&[1], f64::from(WEIGHT_MAG), DType::F32, &cpu()).unwrap();
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

    // Multi-input bounds: cum_phase and F0.
    let input_specs: &[(&[usize], (f32, f32))] = &[
        (&cum_shape[..], (cum_lo, cum_hi)),
        (&f0_shape[..], (f0_lo, f0_hi)),
    ];

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

    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper).unwrap(),
    )
    .expect("valid input bounds");

    (input_bounds, output)
}

// -- TensorBlockBuilder proxy graph for sinegen_post core ---------------------

/// Build a proxy graph for sinegen_post's core path: linear + tanh.
///
/// This isolates the learned portion of sinegen_post (the SourceModule's
/// l_linear + tanh) into a TensorBlockBuilder graph that can be verified
/// with `verify_and_assert_with_config`. The proxy graph structure:
///
///   Input [N_CH, T_AUDIO] → Linear(N_CH, 1) → Tanh → output [1, T_AUDIO]
///
/// This proves the core property: tanh bounds the output to [-1, 1].
fn build_sinegen_post_proxy() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let in_ch = N_CH;
    let out_ch = 1;
    let t = T_AUDIO;

    let mut b = TensorBlockBuilder::new("sinegen_post_proxy");
    let input = b.add_input("sine_wavs", &[in_ch, t]);

    // Linear: [in_ch, t] → transpose → matmul → transpose
    let transposed = b.add_transpose(input, &[1, 0], &[t, in_ch]);
    let proj_w = b.add_input("l_linear_w", &[out_ch, in_ch]);
    let proj_b = b.add_input("l_linear_b", &[out_ch]);
    let projected = b.add_matmul(transposed, proj_w, true, None, &[t, out_ch]);
    let proj_b_bc = b.add_broadcast(proj_b, &[t, out_ch]);
    let biased = b.add_binary_add(projected, proj_b_bc, &[t, out_ch]);

    // Tanh
    let activated = b.add_tanh(biased, &[t, out_ch]);

    // Transpose back to [out_ch, t]
    let output = b.add_transpose(activated, &[1, 0], &[out_ch, t]);
    let def = b.build(output).expect("valid sinegen_post proxy");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[out_ch, in_ch]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[out_ch]), WEIGHT_MAG)),
    ];
    (def, bindings)
}

// ===========================================================================
// Tests: sinegen_pre — Sound re-verification
// ===========================================================================

/// sinegen_pre with standard F0 bounds [50, 500] Hz, recorded as Sound.
///
/// The SineGen pre-processing pipeline contains no normalization layers.
/// All operations (unsqueeze, expand, reshape, broadcast_mul, mul_scalar,
/// fract, interp_downsample) are standard interval-arithmetic-safe
/// operations. IBP through these is provably sound — no heuristic
/// normalization approximation is involved.
#[test]
fn test_sinegen_pre_sound_standard_bounds() {
    let (input_bounds, output) = trace_sinegen_pre_with_bounds(50.0, 500.0);
    assert_all_finite(&output, "sinegen_pre_sound");
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    // Non-vacuous: fract output is in [0, 1), IBP may widen.
    assert!(
        width < VACUOUS_THRESHOLD,
        "sinegen_pre width {width} exceeds vacuous threshold {VACUOUS_THRESHOLD}"
    );
    assert!(
        width > 0.0,
        "sinegen_pre: zero-width bounds suggest degenerate trace"
    );

    // Record as Sound — no normalization layers involved.
    record_ibp_result_with_soundness(
        "kokoro_sinegen_pre",
        &input_bounds,
        &output,
        VerificationSoundnessMode::Sound,
        "IBP through non-normalization ops (mul, fract, interp_downsample). \
         No heuristic approximation. Sound per standard interval arithmetic.",
    );

    eprintln!("kokoro_sinegen_pre Sound: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

/// sinegen_pre with tighter speech-range F0 bounds [50, 400] Hz.
///
/// Human speech fundamental frequency is typically 85-255 Hz (male) to
/// 165-255 Hz (female). The range [50, 400] Hz covers all speech and
/// is tighter than the default [50, 500] Hz which includes singing.
/// Tighter input bounds should produce tighter output bounds.
#[test]
fn test_sinegen_pre_sound_tight_speech_bounds() {
    let (_input_bounds, output) = trace_sinegen_pre_with_bounds(50.0, 400.0);
    assert_all_finite(&output, "sinegen_pre_tight");
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    // With tighter input bounds, output should also be tighter.
    assert!(
        width < VACUOUS_THRESHOLD,
        "sinegen_pre_tight width {width} exceeds vacuous threshold"
    );
    assert!(
        width > 0.0,
        "sinegen_pre_tight: zero-width bounds suggest degenerate trace"
    );

    eprintln!(
        "sinegen_pre tight [50, 400] Hz: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}"
    );
}

/// sinegen_pre output width decreases with tighter input bounds.
///
/// This is a monotonicity property: IBP on a subset of inputs should
/// produce output bounds no wider than IBP on the full input set.
#[test]
fn test_sinegen_pre_tighter_inputs_tighter_outputs() {
    let (_, output_wide) = trace_sinegen_pre_with_bounds(50.0, 500.0);
    let (_, output_tight) = trace_sinegen_pre_with_bounds(50.0, 400.0);

    let (lo_w, hi_w) = bounds_min_max(&output_wide);
    let (lo_t, hi_t) = bounds_min_max(&output_tight);
    let width_wide = hi_w - lo_w;
    let width_tight = hi_t - lo_t;

    eprintln!(
        "sinegen_pre monotonicity: wide=[{lo_w:.4}, {hi_w:.4}] w={width_wide:.4} \
         | tight=[{lo_t:.4}, {hi_t:.4}] w={width_tight:.4}"
    );

    // Tight bounds should be no wider than wide bounds (IBP monotonicity).
    // Allow small epsilon for floating-point arithmetic.
    assert!(
        width_tight <= width_wide + 1e-6,
        "sinegen_pre: tighter inputs [50, 400] should produce bounds no wider \
         than [50, 500]. tight_width={width_tight}, wide_width={width_wide}"
    );
}

// ===========================================================================
// Tests: sinegen_post — Sound re-verification
// ===========================================================================

/// sinegen_post with standard bounds, recorded as Sound.
///
/// Like sinegen_pre, the sinegen_post pipeline contains no normalization
/// layers. The operations are: voiced mask (gt + to_dtype), mul_scalar,
/// interp_upsample, sin, mul_scalar, broadcast_mul, linear, tanh,
/// transpose. All are standard interval-arithmetic-safe operations.
///
/// The tanh at the end ensures output is bounded to [-1, 1].
#[test]
fn test_sinegen_post_sound_standard_bounds() {
    let (input_bounds, output) = trace_sinegen_post_with_bounds(0.0, 1.0, 50.0, 500.0);
    assert_all_finite(&output, "sinegen_post_sound");
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    // tanh output must be in [-1, 1].
    assert!(
        lo_min >= -1.0 - 1e-4,
        "sinegen_post tanh output lower bound {lo_min} below -1.0"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "sinegen_post tanh output upper bound {hi_max} above 1.0"
    );
    assert!(
        width > 0.0,
        "sinegen_post: zero-width bounds suggest degenerate trace"
    );

    // Record as Sound — no normalization layers involved.
    record_ibp_result_with_soundness(
        "kokoro_sinegen_post",
        &input_bounds,
        &output,
        VerificationSoundnessMode::Sound,
        "IBP through non-normalization ops (sin, mul, tanh, linear). \
         No heuristic approximation. Sound per standard interval arithmetic.",
    );

    eprintln!("kokoro_sinegen_post Sound: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

/// sinegen_post with tighter speech-range F0 bounds [50, 400] Hz.
#[test]
fn test_sinegen_post_sound_tight_speech_bounds() {
    let (_, output) = trace_sinegen_post_with_bounds(0.0, 1.0, 50.0, 400.0);
    assert_all_finite(&output, "sinegen_post_tight");
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min >= -1.0 - 1e-4,
        "sinegen_post_tight tanh lower {lo_min} below -1.0"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "sinegen_post_tight tanh upper {hi_max} above 1.0"
    );
    assert!(
        width < VACUOUS_THRESHOLD,
        "sinegen_post_tight width {width} exceeds vacuous threshold"
    );

    eprintln!(
        "sinegen_post tight [50, 400] Hz: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}"
    );
}

/// sinegen_post output width decreases with tighter F0 input bounds.
#[test]
fn test_sinegen_post_tighter_inputs_tighter_outputs() {
    let (_, output_wide) = trace_sinegen_post_with_bounds(0.0, 1.0, 50.0, 500.0);
    let (_, output_tight) = trace_sinegen_post_with_bounds(0.0, 1.0, 50.0, 400.0);

    let (lo_w, hi_w) = bounds_min_max(&output_wide);
    let (lo_t, hi_t) = bounds_min_max(&output_tight);
    let width_wide = hi_w - lo_w;
    let width_tight = hi_t - lo_t;

    eprintln!(
        "sinegen_post monotonicity: wide=[{lo_w:.4}, {hi_w:.4}] w={width_wide:.4} \
         | tight=[{lo_t:.4}, {hi_t:.4}] w={width_tight:.4}"
    );

    assert!(
        width_tight <= width_wide + 1e-6,
        "sinegen_post: tighter F0 inputs [50, 400] should produce bounds no wider \
         than [50, 500]. tight_width={width_tight}, wide_width={width_wide}"
    );
}

// ===========================================================================
// Tests: sinegen_post proxy graph — TensorBlockBuilder + CROWN
// ===========================================================================

/// sinegen_post proxy (linear + tanh) with Conservative IBP → Sound.
///
/// This isolates the learned portion of sinegen_post into a
/// TensorBlockBuilder graph for formal verification via the standard
/// verify_and_assert_with_config pipeline. The proxy proves:
///   - Output bounds within [-1, 1] (tanh squashing)
///   - Soundness is Sound (no normalization layers)
///   - Bounds are non-vacuous
#[test]
fn test_sinegen_post_proxy_conservative_sound() {
    let (def, bindings) = build_sinegen_post_proxy();
    // Input: sine_wavs with amplitude bounded by SINE_AMP = 0.1.
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[N_CH, T_AUDIO]), -SINE_AMP),
        ArrayD::from_elem(IxDyn(&[N_CH, T_AUDIO]), SINE_AMP),
    )
    .expect("valid input bounds");

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_sinegen_post_proxy_sound",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative proxy bounds should be non-vacuous, width={width}"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    // tanh squashes to [-1, 1]; with small weights output stays within.
    assert!(
        lo_min >= -1.0 - 1e-3,
        "sinegen_post proxy: tanh lower >= -1.0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-3,
        "sinegen_post proxy: tanh upper <= 1.0, got {hi_max}"
    );

    eprintln!(
        "kokoro_sinegen_post_proxy_sound: bounds=[{lo_min:.6}, {hi_max:.6}], \
         width={width:.4}, soundness=Sound"
    );
}

/// sinegen_post proxy with forced CROWN escalation.
///
/// Uses threshold=0 to force CROWN propagation through the linear + tanh
/// graph. CROWN should produce tighter bounds than IBP for this simple
/// 2-layer graph (CROWN linearizes tanh around the midpoint).
#[test]
fn test_sinegen_post_proxy_crown_escalation() {
    let (def, bindings) = build_sinegen_post_proxy();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[N_CH, T_AUDIO]), -SINE_AMP),
        ArrayD::from_elem(IxDyn(&[N_CH, T_AUDIO]), SINE_AMP),
    )
    .expect("valid input bounds");

    let crown_config = VerifyConfig::with_threshold(0.0)
        .expect("zero threshold is valid")
        .with_norm_mode(NormBoundsMode::Conservative);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_sinegen_post_proxy_crown",
        &crown_config,
    );

    let method = result.verification.method;
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    let width = result.verification.output_width;

    assert!(
        lo_min >= -1.0 - 1e-3,
        "CROWN proxy: tanh lower >= -1.0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-3,
        "CROWN proxy: tanh upper <= 1.0, got {hi_max}"
    );

    eprintln!(
        "kokoro_sinegen_post_proxy_crown: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}], \
         width={width:.4}, soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: sinegen domain properties (Sound strength)
// ===========================================================================

/// P2 (non-clipping) for sinegen_post: tanh bounds excitation to [-1, 1].
///
/// This is the key domain property: the SourceModule applies tanh as its
/// final nonlinearity, guaranteeing the excitation signal is bounded.
/// With Sound soundness, this is a formal proof, not a heuristic observation.
#[test]
fn test_sinegen_post_sound_p2_non_clipping() {
    let (_, output) = trace_sinegen_post_with_bounds(0.0, 1.0, 50.0, 500.0);
    let (lo_min, hi_max) = bounds_min_max(&output);
    assert!(
        lo_min >= -1.0 - 1e-4 && hi_max <= 1.0 + 1e-4,
        "P2 (non-clipping) for sinegen_post SOUND: output [{lo_min}, {hi_max}] \
         must be in [-1, 1]"
    );
    eprintln!(
        "P2 SOUND for sinegen_post: excitation bounds [{lo_min:.6}, {hi_max:.6}] within [-1, 1]"
    );
}

/// Validate both sinegen entries exist with Sound soundness in status file.
///
/// This test records both entries and validates the persisted soundness
/// classification matches our expectation: Sound (not Heuristic).
#[test]
fn test_sinegen_sound_persist_and_validate() {
    // sinegen_pre
    let (in_pre, out_pre) = trace_sinegen_pre_with_bounds(50.0, 500.0);
    assert_all_finite(&out_pre, "persist_sinegen_pre_sound");
    record_ibp_result_with_soundness(
        "kokoro_sinegen_pre",
        &in_pre,
        &out_pre,
        VerificationSoundnessMode::Sound,
        "IBP through non-normalization ops. Sound per standard interval arithmetic.",
    );

    // sinegen_post
    let (in_post, out_post) = trace_sinegen_post_with_bounds(0.0, 1.0, 50.0, 500.0);
    assert_all_finite(&out_post, "persist_sinegen_post_sound");
    record_ibp_result_with_soundness(
        "kokoro_sinegen_post",
        &in_post,
        &out_post,
        VerificationSoundnessMode::Sound,
        "IBP through non-normalization ops. Sound per standard interval arithmetic.",
    );

    // Validate entries exist and have Sound soundness.
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
            VerificationSoundnessMode::Sound,
            "{key} must have Sound soundness (no normalization layers)"
        );
        assert!(!entry.stale, "{key} must not be stale after recording");
    }

    eprintln!("Both sinegen entries persisted with Sound soundness.");
}
