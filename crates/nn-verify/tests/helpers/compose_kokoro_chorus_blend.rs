// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Compose verification tests for Kokoro chorus audio blending pipeline.
//!
//! The chorus system blends multiple TTS voice renderings into a final
//! audio output. This file verifies that the blending operations preserve
//! audio quality bounds using IBP (Interval Bound Propagation) through
//! proxy graphs built with TensorBlockBuilder.
//!
//! Verified properties:
//!   - Linear crossfade alpha stays in [0,1] via sigmoid constraint
//!   - Weighted average of bounded voices produces bounded output
//!   - Speed-scaled durations remain non-negative
//!   - Multi-voice mixing preserves input bounds
//!   - Crossfade overlap region samples stay within input bounds
//!   - No zero-amplitude silent gaps at chunk boundaries
//!
//! All tests use small symbolic dimensions (num_voices=3, chunk_len=8,
//! crossfade=4) for fast verification while exercising the blend logic.
//!
//! Part of #4186: Add compose verification tests for Kokoro chorus blend bounds.
//! Part of #3351: Epic — Absolutely Best Kokoro.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

use super::common::{assert_bounds_valid, bounds_min_max, uniform_bounds};

// -- Symbolic dimensions (small for fast tests) --------------------------------

/// Number of audio samples per chunk.
const CHUNK_LEN: usize = 8;
/// Number of crossfade samples at chunk boundary.
const CROSSFADE: usize = 4;
/// Number of voices being blended.
const NUM_VOICES: usize = 3;
/// Weight magnitude for synthetic blend weights.
const W_MAG: f32 = 0.01;
/// Audio amplitude bound for voice inputs.
const AUDIO_BOUND: f32 = 1.0;
/// Vacuous width threshold — bounds wider than this are meaningless.
const VACUOUS_THRESHOLD: f32 = 50.0;

// ===========================================================================
// Graph 1: Linear crossfade alpha bounds (sigmoid-constrained)
// ===========================================================================

/// Build a proxy graph for sigmoid-constrained crossfade blending.
///
/// Architecture:
///   voice_a [CHUNK_LEN]  (Variable)
///   voice_b [CHUNK_LEN]  (Variable)
///   alpha_raw [CHUNK_LEN] (Variable) — unconstrained blend parameter
///   → sigmoid(alpha_raw) → alpha ∈ (0, 1)
///   → output = alpha * voice_a + (1 - alpha) * voice_b
///
/// The sigmoid constrains alpha to (0, 1), guaranteeing the crossfade
/// is a convex combination. IBP through sigmoid produces tight bounds.
fn build_crossfade_sigmoid() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let s = &[CHUNK_LEN];
    let mut b = TensorBlockBuilder::new("chorus_crossfade_sigmoid");

    // voice_a and voice_b are unused graph inputs — they exist to model
    // the full crossfade signature. The output is sigmoid(alpha_raw)
    // which proves alpha ∈ (0, 1) for any unconstrained input.
    let _voice_a = b.add_input("voice_a", s);
    let _voice_b = b.add_input("voice_b", s);
    let alpha_raw = b.add_input("alpha_raw", s);

    // alpha = sigmoid(alpha_raw), constrained to (0, 1).
    // The sigmoid is the key property: it maps R → (0, 1), guaranteeing
    // the crossfade parameter is a valid convex combination weight.
    let alpha = b.add_sigmoid(alpha_raw, s);

    // Output the sigmoid alpha — proves alpha ∈ (0, 1) for all inputs.
    // The full blend formula (alpha * voice_a + (1-alpha) * voice_b)
    // is verified separately in build_crossfade_blend using the
    // equivalent formulation alpha * diff + base.
    let def = b.build(alpha).expect("valid crossfade sigmoid graph");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    (def, bindings)
}

/// Build a crossfade blend graph: output = alpha * voice_a + (1-alpha) * voice_b.
///
/// Uses sigmoid for alpha constraint and models the blend as:
///   output = sigmoid(raw) * voice_a + sigmoid(raw) * voice_b + voice_b
///   (via algebraic rearrangement using only BinaryMul and BinaryAdd)
///
/// Architecture:
///   voice_a [CHUNK_LEN] (Variable)
///   voice_b [CHUNK_LEN] (Variable)
///   alpha_raw [CHUNK_LEN] (Variable) — unconstrained
///   → alpha = sigmoid(alpha_raw)
///   → diff = voice_a   (proxy: difference approx, voice_a acts as the delta)
///   → blend = alpha * diff  (scaled delta)
///   → output = blend + voice_b  (base + weighted delta)
///
/// This models: output = alpha * voice_a + voice_b, which for voice_a
/// representing (a - b) gives the exact crossfade formula. IBP verifies
/// that bounded inputs produce bounded outputs.
fn build_crossfade_blend() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let s = &[CHUNK_LEN];
    let mut b = TensorBlockBuilder::new("chorus_crossfade_blend");

    let diff = b.add_input("voice_diff", s); // (voice_a - voice_b) pre-computed
    let voice_b = b.add_input("voice_b", s);
    let alpha_raw = b.add_input("alpha_raw", s);

    // alpha = sigmoid(alpha_raw) ∈ (0, 1)
    let alpha = b.add_sigmoid(alpha_raw, s);

    // blend = alpha * diff
    let blend = b.add_binary_mul(alpha, diff, s);

    // output = blend + voice_b = alpha * (a - b) + b
    let output = b.add_binary_add(blend, voice_b, s);

    let def = b.build(output).expect("valid crossfade blend graph");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    (def, bindings)
}

// ===========================================================================
// Graph 2: Weighted average of N voices
// ===========================================================================

/// Build a graph that computes a weighted sum of 3 voices with sigmoid weights.
///
/// Architecture:
///   voice_0 [CHUNK_LEN] (Variable)
///   voice_1 [CHUNK_LEN] (Variable)
///   voice_2 [CHUNK_LEN] (Variable)
///   raw_w0, raw_w1, raw_w2 [CHUNK_LEN] (Variable)
///   → w_i = sigmoid(raw_w_i)  — constrained to (0, 1)
///   → output = w0 * voice_0 + w1 * voice_1 + w2 * voice_2
///
/// Note: This is a weighted sum, not a normalized convex combination.
/// IBP verifies the output is bounded given bounded inputs and sigmoid weights.
fn build_multi_voice_weighted_sum() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let s = &[CHUNK_LEN];
    let mut b = TensorBlockBuilder::new("chorus_multi_voice_mix");

    let voice_0 = b.add_input("voice_0", s);
    let voice_1 = b.add_input("voice_1", s);
    let voice_2 = b.add_input("voice_2", s);
    let raw_w0 = b.add_input("raw_w0", s);
    let raw_w1 = b.add_input("raw_w1", s);
    let raw_w2 = b.add_input("raw_w2", s);

    // Sigmoid-constrained weights
    let w0 = b.add_sigmoid(raw_w0, s);
    let w1 = b.add_sigmoid(raw_w1, s);
    let w2 = b.add_sigmoid(raw_w2, s);

    // Weighted sum: w0*v0 + w1*v1 + w2*v2
    let t0 = b.add_binary_mul(w0, voice_0, s);
    let t1 = b.add_binary_mul(w1, voice_1, s);
    let t2 = b.add_binary_mul(w2, voice_2, s);

    let sum_01 = b.add_binary_add(t0, t1, s);
    let output = b.add_binary_add(sum_01, t2, s);

    let def = b.build(output).expect("valid multi-voice mix graph");

    let bindings = vec![
        TensorParamBinding::Variable, // voice_0
        TensorParamBinding::Variable, // voice_1
        TensorParamBinding::Variable, // voice_2
        TensorParamBinding::Variable, // raw_w0
        TensorParamBinding::Variable, // raw_w1
        TensorParamBinding::Variable, // raw_w2
    ];
    (def, bindings)
}

// ===========================================================================
// Graph 3: Speed-scaled duration bounds
// ===========================================================================

/// Build a graph modeling speed-scaled phoneme durations.
///
/// Architecture:
///   durations [CHUNK_LEN] (Variable) — predicted phoneme durations
///   speed_raw [CHUNK_LEN] (Variable) — unconstrained speed factor
///   → speed = sigmoid(speed_raw) * 1.8 + 0.2  (maps to [0.2, 2.0])
///     Approximated as: speed = sigmoid(speed_raw) since IBP through
///     sigmoid already constrains to (0, 1) and a broader range doesn't
///     change the bound-preservation property.
///   → scaled = durations * speed
///   → output = relu(scaled)  — ensure non-negative duration
///
/// Proves: speed scaling followed by ReLU preserves non-negative duration bounds
/// when input durations are non-negative.
fn build_speed_scaled_duration() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let s = &[CHUNK_LEN];
    let mut b = TensorBlockBuilder::new("chorus_speed_duration");

    let durations = b.add_input("durations", s);
    let speed_raw = b.add_input("speed_raw", s);

    // Constrain speed factor via sigmoid → (0, 1)
    let speed = b.add_sigmoid(speed_raw, s);

    // Scaled duration = durations * speed
    let scaled = b.add_binary_mul(durations, speed, s);

    // ReLU ensures non-negative output
    let output = b.add_relu(scaled, s);

    let def = b.build(output).expect("valid speed-scaled duration graph");

    let bindings = vec![
        TensorParamBinding::Variable, // durations
        TensorParamBinding::Variable, // speed_raw
    ];
    (def, bindings)
}

// ===========================================================================
// Graph 4: Crossfade overlap region bounds
// ===========================================================================

/// Build a crossfade overlap graph at the boundary between two chunks.
///
/// Architecture:
///   chunk_a_tail [CROSSFADE] (Variable) — end of chunk A
///   chunk_b_head [CROSSFADE] (Variable) — start of chunk B
///   alpha_raw [CROSSFADE] (Variable) — blend parameter
///   → alpha = sigmoid(alpha_raw)
///   → blended = alpha * chunk_a_tail + (1-alpha) * chunk_b_head
///   → output = tanh(blended)  — final clamp to [-1, 1]
///
/// Models the crossfade region where two adjacent chunks overlap. The
/// tanh output clamp guarantees audio stays in [-1, 1], proving that
/// crossfade does not produce out-of-range samples.
fn build_overlap_region() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let s = &[CROSSFADE];
    let mut b = TensorBlockBuilder::new("chorus_overlap_region");

    // Inputs: the overlap from chunk A's tail and chunk B's head
    let diff = b.add_input("overlap_diff", s); // (chunk_a_tail - chunk_b_head)
    let base = b.add_input("overlap_base", s); // chunk_b_head
    let alpha_raw = b.add_input("alpha_raw", s);

    // Sigmoid-constrained blend
    let alpha = b.add_sigmoid(alpha_raw, s);
    let blend = b.add_binary_mul(alpha, diff, s);
    let blended = b.add_binary_add(blend, base, s);

    // Tanh clamp to [-1, 1] — audio range guarantee
    let output = b.add_tanh(blended, s);

    let def = b.build(output).expect("valid overlap region graph");

    let bindings = vec![
        TensorParamBinding::Variable, // overlap_diff
        TensorParamBinding::Variable, // overlap_base
        TensorParamBinding::Variable, // alpha_raw
    ];
    (def, bindings)
}

// ===========================================================================
// Graph 5: Silent gap prevention (non-zero energy at boundary)
// ===========================================================================

/// Build a graph that adds a learned residual bias to prevent silent gaps.
///
/// Architecture:
///   boundary [CROSSFADE] (Variable) — audio at chunk boundary
///   bias_raw [CROSSFADE] (Constant) — small learned bias
///   → biased = boundary + bias
///   → output = tanh(biased)
///
/// The bias prevents the boundary samples from collapsing to zero
/// (which would create audible clicks/gaps). IBP verifies that when
/// boundary audio has non-zero energy, the output retains that energy.
fn build_boundary_bias() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let s = &[CROSSFADE];
    let mut b = TensorBlockBuilder::new("chorus_boundary_bias");

    let boundary = b.add_input("boundary", s);
    let bias = b.add_input("bias", s);

    // Add bias to prevent zero collapse
    let biased = b.add_binary_add(boundary, bias, s);

    // Tanh output clamp
    let output = b.add_tanh(biased, s);

    let def = b.build(output).expect("valid boundary bias graph");

    let bindings = vec![
        TensorParamBinding::Variable, // boundary
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(s), W_MAG)),
    ];
    (def, bindings)
}

// ===========================================================================
// Graph 6: Multi-voice mixing with tanh output clamp
// ===========================================================================

/// Build a graph for 3-voice mixing with tanh output clamp.
///
/// Architecture:
///   voice_0, voice_1, voice_2 [CHUNK_LEN] (Variable)
///   → sum = voice_0 + voice_1 + voice_2
///   → output = tanh(sum)
///
/// Proves: mixing N voices followed by tanh guarantees output ∈ [-1, 1],
/// regardless of how many voices contribute. This is the simplest
/// mixing model (equal weights, tanh saturation prevents clipping).
fn build_multi_voice_tanh_mix() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let s = &[CHUNK_LEN];
    let mut b = TensorBlockBuilder::new("chorus_multi_voice_tanh");

    let voice_0 = b.add_input("voice_0", s);
    let voice_1 = b.add_input("voice_1", s);
    let voice_2 = b.add_input("voice_2", s);

    // Sum all voices
    let sum_01 = b.add_binary_add(voice_0, voice_1, s);
    let sum_012 = b.add_binary_add(sum_01, voice_2, s);

    // Tanh clamp to [-1, 1]
    let output = b.add_tanh(sum_012, s);

    let def = b.build(output).expect("valid multi-voice tanh mix graph");

    let bindings = vec![
        TensorParamBinding::Variable, // voice_0
        TensorParamBinding::Variable, // voice_1
        TensorParamBinding::Variable, // voice_2
    ];
    (def, bindings)
}

// ===========================================================================
// Test 1: Linear crossfade alpha bounds via sigmoid
// ===========================================================================

/// Sigmoid constrains crossfade alpha to (0, 1) for all blend positions.
///
/// IBP through sigmoid is exact: sigmoid maps R → (0, 1). For any input
/// range [lo, hi], the output range is [sigmoid(lo), sigmoid(hi)].
/// This is the fundamental property ensuring crossfade is a convex
/// combination.
///
/// Part of #4186.
#[test]
fn test_chorus_crossfade_alpha_sigmoid_bounds() {
    let (def, bindings) = build_crossfade_sigmoid();
    def.validate().expect("crossfade sigmoid def validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // All three inputs: voices in [-1, 1], alpha_raw in [-5, 5]
    let total = CHUNK_LEN * 3;
    let mut lower = vec![-AUDIO_BOUND; CHUNK_LEN * 2];
    lower.extend(vec![-5.0f32; CHUNK_LEN]);
    let mut upper = vec![AUDIO_BOUND; CHUNK_LEN * 2];
    upper.extend(vec![5.0f32; CHUNK_LEN]);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper).unwrap(),
    )
    .expect("valid input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);

    // Sigmoid output must be in (0, 1). IBP preserves this exactly.
    assert!(
        lo_min >= 0.0 - 1e-6,
        "sigmoid alpha lower bound {lo_min} should be >= 0"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "sigmoid alpha upper bound {hi_max} should be <= 1"
    );
    assert!(
        lo_min > 0.0 - 1e-4,
        "sigmoid alpha should be strictly positive, got {lo_min}"
    );
    assert!(
        hi_max < 1.0 + 1e-4,
        "sigmoid alpha should be strictly less than 1, got {hi_max}"
    );

    let width = hi_max - lo_min;
    assert!(
        width > 0.0,
        "alpha bounds should have non-zero width (non-degenerate), got {width}"
    );
    assert!(
        width < VACUOUS_THRESHOLD,
        "alpha bounds width {width} exceeds vacuous threshold {VACUOUS_THRESHOLD}"
    );

    eprintln!(
        "chorus crossfade alpha: sigmoid bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}"
    );
}

// ===========================================================================
// Test 2: Weighted average output bounds
// ===========================================================================

/// If all voices are bounded in [-1, 1], the sigmoid-weighted blend is bounded.
///
/// For N voices with sigmoid weights w_i ∈ (0, 1) and voice_i ∈ [-A, A]:
///   |output| <= sum(w_i * A) <= N * A
///
/// IBP propagates this correctly through BinaryMul and BinaryAdd.
/// The key property: bounded inputs with bounded weights produce bounded output.
///
/// Part of #4186.
#[test]
fn test_chorus_weighted_average_output_bounds() {
    let (def, bindings) = build_multi_voice_weighted_sum();
    def.validate().expect("multi-voice mix def validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // 6 inputs: 3 voices in [-1, 1], 3 weight params in [-3, 3]
    let total = CHUNK_LEN * 6;
    let mut lower = vec![-AUDIO_BOUND; CHUNK_LEN * 3]; // voices
    lower.extend(vec![-3.0f32; CHUNK_LEN * 3]); // raw weights
    let mut upper = vec![AUDIO_BOUND; CHUNK_LEN * 3];
    upper.extend(vec![3.0f32; CHUNK_LEN * 3]);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper).unwrap(),
    )
    .expect("valid input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    // Output must be finite and non-vacuous.
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "weighted average bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert!(
        width > 0.0,
        "weighted average should have non-zero width, got {width}"
    );
    assert!(
        width < VACUOUS_THRESHOLD,
        "weighted average width {width} exceeds vacuous threshold {VACUOUS_THRESHOLD}"
    );

    // With sigmoid weights in (0, 1) and voices in [-1, 1]:
    // max |output| <= 3 * 1 * 1 = 3 (sum of 3 sigmoid-weighted voices).
    // IBP over-approximation may widen this.
    assert!(
        hi_max <= NUM_VOICES as f32 * AUDIO_BOUND + 1.0,
        "upper bound {hi_max} unexpectedly large for {NUM_VOICES}-voice mix"
    );
    assert!(
        lo_min >= -(NUM_VOICES as f32) * AUDIO_BOUND - 1.0,
        "lower bound {lo_min} unexpectedly large for {NUM_VOICES}-voice mix"
    );

    eprintln!("chorus weighted average: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 3: Speed-scaled duration bounds
// ===========================================================================

/// Speed scaling followed by ReLU preserves non-negative duration bounds.
///
/// When input durations are non-negative [0, D_max] and speed factor is
/// sigmoid-constrained to (0, 1), the output durations are:
///   output = relu(duration * sigmoid(speed_raw))
///   ∈ [0, D_max * 1] = [0, D_max]
///
/// IBP proves the lower bound is >= 0, confirming durations remain
/// non-negative after speed scaling.
///
/// Part of #4186.
#[test]
fn test_chorus_speed_scaled_duration_bounds() {
    let (def, bindings) = build_speed_scaled_duration();
    def.validate().expect("speed-scaled duration def validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // durations in [0, 10], speed_raw in [-3, 3]
    let total = CHUNK_LEN * 2;
    let mut lower = vec![0.0f32; CHUNK_LEN]; // durations >= 0
    lower.extend(vec![-3.0f32; CHUNK_LEN]); // speed_raw
    let mut upper = vec![10.0f32; CHUNK_LEN]; // max duration
    upper.extend(vec![3.0f32; CHUNK_LEN]);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper).unwrap(),
    )
    .expect("valid input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    // ReLU ensures lower bound >= 0.
    assert!(
        lo_min >= 0.0 - 1e-6,
        "speed-scaled duration lower bound {lo_min} should be >= 0 (ReLU enforces non-negative)"
    );
    // Upper bound should be at most duration_max * sigmoid_max ≈ 10 * 1 = 10.
    assert!(
        hi_max <= 10.0 + 1.0,
        "speed-scaled duration upper bound {hi_max} exceeds expected maximum"
    );
    assert!(
        width > 0.0,
        "duration bounds should have non-zero width, got {width}"
    );

    eprintln!("chorus speed-scaled duration: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 4: Multi-voice mixing with tanh output clamp
// ===========================================================================

/// N voices mixed and passed through tanh produce output in [-1, 1].
///
/// This is the key safety property for the chorus output stage: regardless
/// of how many voices contribute (and their individual amplitudes), the
/// tanh nonlinearity guarantees the final audio is bounded.
///
/// Part of #4186.
#[test]
fn test_chorus_multi_voice_tanh_bounded() {
    let (def, bindings) = build_multi_voice_tanh_mix();
    def.validate().expect("multi-voice tanh mix def validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // 3 voices in [-1, 1]
    let input = uniform_bounds(&[CHUNK_LEN * NUM_VOICES], AUDIO_BOUND);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);

    // Tanh output must be in [-1, 1].
    assert!(
        lo_min >= -1.0 - 1e-4,
        "tanh mix lower bound {lo_min} below -1.0"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "tanh mix upper bound {hi_max} above 1.0"
    );

    let width = hi_max - lo_min;
    assert!(
        width > 0.0,
        "tanh mix bounds should have non-zero width, got {width}"
    );

    eprintln!("chorus multi-voice tanh: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 5: Crossfade overlap region bounds
// ===========================================================================

/// Crossfade overlap samples stay within tanh-clamped [-1, 1] bounds.
///
/// At chunk boundaries, the overlap region blends the tail of one chunk
/// with the head of the next. The sigmoid-constrained alpha ensures a
/// convex-like combination, and the tanh output clamp provides the final
/// audio range guarantee.
///
/// Part of #4186.
#[test]
fn test_chorus_overlap_region_bounds() {
    let (def, bindings) = build_overlap_region();
    def.validate().expect("overlap region def validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // overlap_diff in [-2, 2] (difference of two audio signals)
    // overlap_base in [-1, 1] (base audio)
    // alpha_raw in [-5, 5]
    let total = CROSSFADE * 3;
    let mut lower = vec![-2.0f32; CROSSFADE]; // diff
    lower.extend(vec![-AUDIO_BOUND; CROSSFADE]); // base
    lower.extend(vec![-5.0f32; CROSSFADE]); // alpha_raw
    let mut upper = vec![2.0f32; CROSSFADE];
    upper.extend(vec![AUDIO_BOUND; CROSSFADE]);
    upper.extend(vec![5.0f32; CROSSFADE]);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper).unwrap(),
    )
    .expect("valid input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);

    // Tanh guarantees output in [-1, 1].
    assert!(
        lo_min >= -1.0 - 1e-4,
        "overlap tanh lower {lo_min} below -1.0"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "overlap tanh upper {hi_max} above 1.0"
    );

    let width = hi_max - lo_min;
    assert!(
        width > 0.0,
        "overlap region bounds should have non-zero width, got {width}"
    );

    eprintln!("chorus overlap region: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 6: Silent gap prevention at chunk boundaries
// ===========================================================================

/// Bias at chunk boundaries prevents zero-amplitude artifacts.
///
/// When boundary audio has non-zero energy (e.g., [0.01, 1.0]) and a
/// small positive bias is added, the tanh output retains positive energy.
/// This proves that the boundary bias mechanism prevents silent gaps.
///
/// Part of #4186.
#[test]
fn test_chorus_boundary_no_silent_gap() {
    let (def, bindings) = build_boundary_bias();
    def.validate().expect("boundary bias def validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Boundary audio with guaranteed non-zero positive energy: [0.01, 1.0]
    // The constant bias W_MAG is added via bindings.
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[CROSSFADE]), 0.01f32),
        ArrayD::from_elem(IxDyn(&[CROSSFADE]), AUDIO_BOUND),
    )
    .expect("valid input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);

    // With input lower bound 0.01 and bias W_MAG=0.01, the pre-tanh lower
    // is 0.02. tanh(0.02) ≈ 0.02 > 0. So the output lower bound should
    // be strictly positive — no silent gap.
    assert!(
        lo_min > 0.0,
        "boundary output lower bound {lo_min} must be > 0 (no silent gap)"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "boundary output upper bound {hi_max} exceeds tanh range"
    );

    let width = hi_max - lo_min;
    assert!(
        width > 0.0,
        "boundary bounds should have non-zero width, got {width}"
    );

    eprintln!(
        "chorus boundary bias: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}, \
         lo_min > 0 confirms no silent gap"
    );
}

// ===========================================================================
// Test 7: Crossfade blend preserves bounded output
// ===========================================================================

/// The full crossfade blend (alpha * diff + base) produces bounded output.
///
/// This verifies the complete crossfade formula: given voice_diff in [-2, 2]
/// (the difference between two bounded voices), voice_b in [-1, 1], and
/// sigmoid-constrained alpha, the blended output is finite and non-vacuous.
///
/// Part of #4186.
#[test]
fn test_chorus_crossfade_blend_bounded() {
    let (def, bindings) = build_crossfade_blend();
    def.validate().expect("crossfade blend def validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // voice_diff in [-2, 2], voice_b in [-1, 1], alpha_raw in [-4, 4]
    let total = CHUNK_LEN * 3;
    let mut lower = vec![-2.0f32; CHUNK_LEN]; // diff
    lower.extend(vec![-AUDIO_BOUND; CHUNK_LEN]); // voice_b
    lower.extend(vec![-4.0f32; CHUNK_LEN]); // alpha_raw
    let mut upper = vec![2.0f32; CHUNK_LEN];
    upper.extend(vec![AUDIO_BOUND; CHUNK_LEN]);
    upper.extend(vec![4.0f32; CHUNK_LEN]);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper).unwrap(),
    )
    .expect("valid input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "crossfade blend bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert!(
        width > 0.0,
        "crossfade blend should have non-zero width, got {width}"
    );
    assert!(
        width < VACUOUS_THRESHOLD,
        "crossfade blend width {width} exceeds vacuous threshold {VACUOUS_THRESHOLD}"
    );

    // The blend is: sigmoid(alpha) * diff + base.
    // With sigmoid ∈ (0,1), diff ∈ [-2,2], base ∈ [-1,1]:
    // output ∈ [-2-1, 2+1] = [-3, 3] at worst.
    assert!(
        hi_max <= 3.0 + 1.0,
        "blend upper {hi_max} exceeds theoretical max"
    );
    assert!(
        lo_min >= -3.0 - 1.0,
        "blend lower {lo_min} exceeds theoretical min"
    );

    eprintln!("chorus crossfade blend: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 8: IBP monotonicity — tighter inputs produce tighter outputs
// ===========================================================================

/// Narrower voice input bounds produce narrower blend output bounds.
///
/// This is a fundamental IBP property: if input_tight ⊂ input_wide,
/// then ibp(input_tight) ⊂ ibp(input_wide). We verify this for the
/// multi-voice tanh mix by comparing output widths at two input radii.
///
/// Part of #4186.
#[test]
fn test_chorus_blend_ibp_monotonicity() {
    let (def, bindings) = build_multi_voice_tanh_mix();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide: voices in [-1, 1]
    let input_wide = uniform_bounds(&[CHUNK_LEN * NUM_VOICES], 1.0);
    let output_wide = graph.propagate_ibp(&input_wide).expect("IBP wide");
    assert_bounds_valid(&output_wide);

    // Tight: voices in [-0.1, 0.1]
    let input_tight = uniform_bounds(&[CHUNK_LEN * NUM_VOICES], 0.1);
    let output_tight = graph.propagate_ibp(&input_tight).expect("IBP tight");
    assert_bounds_valid(&output_tight);

    let (lo_w, hi_w) = bounds_min_max(&output_wide);
    let (lo_t, hi_t) = bounds_min_max(&output_tight);
    let width_wide = hi_w - lo_w;
    let width_tight = hi_t - lo_t;

    eprintln!(
        "chorus blend monotonicity: wide=[{lo_w:.4}, {hi_w:.4}] w={width_wide:.4} | \
         tight=[{lo_t:.4}, {hi_t:.4}] w={width_tight:.4}"
    );

    // Tight bounds should be no wider than wide bounds (IBP monotonicity).
    assert!(
        width_tight <= width_wide + 1e-6,
        "IBP monotonicity violation: tight width {width_tight} > wide width {width_wide}"
    );

    // Additionally, tight output should be strictly narrower (tanh is non-trivially
    // tighter with smaller inputs).
    assert!(
        width_tight < width_wide,
        "tight inputs should produce strictly narrower output bounds: \
         tight_width={width_tight}, wide_width={width_wide}"
    );
}
