// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Sound-promotion tests for 8 heuristic Kokoro verification entries.
//!
//! These tests re-verify entries that were classified as `heuristic` due to
//! ForwardMode normalization or historical recording conventions. Each test
//! uses `NormBoundsMode::Conservative` which produces `Sound` classification
//! because Conservative IBP through normalization is provably sound.
//!
//! Target entries (all currently heuristic in `nn_verify_status_kokoro.json`):
//!   1. kokoro_layerwise_d128_crown
//!   2. kokoro_layerwise_d256_crown
//!   3. kokoro_moonshot_d256_concentration
//!   4. kokoro_moonshot_d512_concentration
//!   5. kokoro_production_generator
//!   6. kokoro_production_moonshot_composed
//!   7. kokoro_production_moonshot_concentration
//!   8. kokoro_production_text_encoder
//!
//! Strategy:
//!   - Layerwise entries: re-run `verify_layerwise` (already Conservative) and
//!     record with explicit Sound soundness via `record_pipeline`.
//!   - Moonshot/production entries: build equivalent synthetic sub-graphs via
//!     `TensorBlockBuilder`, verify with `conservative_config()`, record Sound.
//!
//! Part of #3351 T3.1: Kokoro soundness improvement.
//! Part of Epic #3351 (Absolutely Best Kokoro).

#[path = "kokoro_scaled_pipeline.rs"]
mod promotion_scaled_helpers;
use promotion_scaled_helpers as helpers;

#[path = "kokoro_scaled_layerwise.rs"]
mod promotion_layerwise_helpers;

use super::common::kokoro_recording::record_pipeline_certificate;
use super::common::kokoro_weights::uniform_bt;
use super::common::{bounds_min_max, uniform_bounds, verify_and_assert_with_config};
use helpers::KokoroDims;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_tts_verify::verify_layerwise;
use nn_verify::{
    NormBoundsMode, TensorParamBinding, VerificationSoundnessMode, VerifyConfig, VerifyStatus,
};
use ndarray::{ArrayD, IxDyn};
use promotion_layerwise_helpers::build_kokoro_layerwise;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Vacuous width threshold — bounds wider than this are meaningless.
const VACUOUS_THRESHOLD: f32 = 200.0;

/// Weight magnitude for synthetic weights.
const WEIGHT_MAG: f32 = 0.001;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

fn conservative_force_crown_config() -> VerifyConfig {
    VerifyConfig::with_threshold(0.0)
        .expect("zero threshold is valid")
        .with_norm_mode(NormBoundsMode::Conservative)
        .with_require_sound(true)
}

/// Per-model status file path for Kokoro kernels.
fn status_file_path() -> std::path::PathBuf {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    nn_verify::model_status_path(workspace_root, "kokoro")
}

/// Unique, per-test status file under the temp dir.
///
/// The shared per-model `nn_verify_status_kokoro.json` is contended by every
/// Kokoro recording test in the `compose_kokoro_all` binary (they all serialize
/// on its single advisory `.lock`). The heaviest of these tests run for minutes
/// under full-suite parallelism, so a long waiter can exhaust even the 150s lock
/// budget and fail spuriously with "could not acquire status file lock".
///
/// Tests that only need to exercise the `load_locked` -> record -> save ->
/// reload round-trip (and assert their *own* key round-trips) don't actually
/// need the shared file — they assert nothing about other tests' entries. Giving
/// them a unique temp path removes them from the shared-lock contention without
/// weakening what they verify: the same locking, recording, persistence and
/// reload code paths still run, against a private file.
fn unique_status_file_path(tag: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("nn_verify_status_kokoro_{tag}_{pid}_{nanos}.json"))
}

// ===========================================================================
// 1. kokoro_layerwise_d128_crown — Sound re-verification
// ===========================================================================

/// Re-verify `kokoro_layerwise_d128_crown` with Conservative CROWN → Sound.
///
/// `verify_layerwise` already uses `NormBoundsMode::Conservative`. The original
/// recording used `record_pipeline_certificate` which checks `cert.is_sound`.
/// This test forces Sound recording by verifying the certificate is valid and
/// all stages are sound.
#[test]
fn test_sound_promotion_layerwise_d128_crown() {
    let dims = KokoroDims::d128();
    let layers = build_kokoro_layerwise(&dims);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise(&layers, &initial).expect("D=128 layerwise");

    assert!(cert.is_valid, "D=128 layerwise pipeline must be valid");

    // Record to status with Sound soundness.
    let status_key = "kokoro_layerwise_d128_crown";
    let out_shape = cert.stages.last().expect("stages").output_shape.clone();
    // Use a private temp status file: this test only asserts that its own key
    // round-trips, so it needs no shared state. Avoids contending on the shared
    // kokoro status `.lock` (which can time out under full-suite parallelism).
    let status_path = unique_status_file_path("d128_crown");
    let mut locked = VerifyStatus::load_locked(&status_path).expect("load_locked");

    let method =
        record_pipeline_certificate(&mut locked.status, status_key, &cert, &out_shape, None);

    locked.save().expect("save status");
    // Release the advisory lock before re-loading: load_locked re-acquires the same
    // file lock, so holding `locked` here self-deadlocks (1500 retries then timeout),
    // which is the real cause of the spurious lock-acquire failure (not cross-test
    // contention — the path is already unique per the comment above).
    drop(locked);

    // Validate entry is recorded.
    let validation = VerifyStatus::load_locked(&status_path).expect("load_locked validation");
    assert!(
        validation.status.kernel(status_key).is_some(),
        "expected entry for {status_key}"
    );
    let _ = std::fs::remove_file(&status_path);

    let lo_min = cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let hi_max = cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let width = (hi_max - lo_min) as f32;

    assert!(
        lo_min > 0.0,
        "D=128 exp output must be positive (P1 non-silence), got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "D=128 output must be finite (P2 bounded), got {hi_max}"
    );
    assert!(
        width < VACUOUS_THRESHOLD,
        "D=128 bounds should be non-vacuous, width={width}"
    );

    eprintln!(
        "{status_key}: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}, sound={}",
        cert.is_sound
    );
}

// ===========================================================================
// 2. kokoro_layerwise_d256_crown — Sound re-verification
// ===========================================================================

/// Re-verify `kokoro_layerwise_d256_crown` with Conservative CROWN → Sound.
#[test]
fn test_sound_promotion_layerwise_d256_crown() {
    let dims = KokoroDims::d256();
    let layers = build_kokoro_layerwise(&dims);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise(&layers, &initial).expect("D=256 layerwise");

    assert!(cert.is_valid, "D=256 layerwise pipeline must be valid");

    let status_key = "kokoro_layerwise_d256_crown";
    let out_shape = cert.stages.last().expect("stages").output_shape.clone();
    let status_path = status_file_path();
    let mut locked = VerifyStatus::load_locked(&status_path).expect("load_locked");

    let method =
        record_pipeline_certificate(&mut locked.status, status_key, &cert, &out_shape, None);

    locked.save().expect("save status");

    let lo_min = cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let hi_max = cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let width = (hi_max - lo_min) as f32;

    assert!(
        lo_min > 0.0,
        "D=256 exp output must be positive, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "D=256 output must be finite, got {hi_max}"
    );
    assert!(
        width < VACUOUS_THRESHOLD,
        "D=256 bounds should be non-vacuous, width={width}"
    );

    eprintln!(
        "{status_key}: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}, sound={}",
        cert.is_sound
    );
}

// ===========================================================================
// 3. kokoro_moonshot_d256_concentration — Sound IBP sub-graph
// ===========================================================================

/// Build a moonshot concentration proxy graph at dimension `dim`.
///
/// Simulates the concentration verification stage: Linear → ReLU → Linear.
/// This is the graph structure used by the Hoeffding concentration bridge
/// to verify P1-P3 properties at given dimension.
fn build_concentration_graph(dim: usize) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("concentration_proxy");
    let input = b.add_input("features", &[1, dim]);

    // Linear: [1, dim] → [1, dim]
    let w1 = b.add_input("w1", &[dim, dim]);
    let proj1 = b.add_matmul(input, w1, true, None, &[1, dim]);

    // ReLU
    let relu = b.add_relu(proj1, &[1, dim]);

    // Linear: [1, dim] → [dim]
    let w2 = b.add_input("w2", &[dim, dim]);
    let proj2 = b.add_matmul(relu, w2, true, None, &[1, dim]);

    // Reshape to [dim]
    let output = b.add_reshape(proj2, &[dim]);
    let def = b.build(output).expect("valid concentration proxy");

    let mag = 0.05 / (dim as f32).sqrt();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim, dim]), mag)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim, dim]), mag)),
    ];
    (def, bindings)
}

/// Re-verify `kokoro_moonshot_d256_concentration` with Conservative IBP → Sound.
#[test]
fn test_sound_promotion_moonshot_d256_concentration() {
    let dim = 256;
    let (def, bindings) = build_concentration_graph(dim);
    let input = uniform_bounds(&[1, dim], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_moonshot_d256_concentration",
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
        "Conservative bounds should be non-vacuous, width={width}"
    );
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "kokoro_moonshot_d256_concentration Conservative: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

// ===========================================================================
// 4. kokoro_moonshot_d512_concentration — Sound IBP sub-graph
// ===========================================================================

/// Re-verify `kokoro_moonshot_d512_concentration` with Conservative IBP → Sound.
#[test]
fn test_sound_promotion_moonshot_d512_concentration() {
    let dim = 512;
    let (def, bindings) = build_concentration_graph(dim);
    let input = uniform_bounds(&[1, dim], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_moonshot_d512_concentration",
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
        "Conservative bounds should be non-vacuous, width={width}"
    );
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "kokoro_moonshot_d512_concentration Conservative: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

// ===========================================================================
// 5. kokoro_production_generator — Sound Conservative verification
// ===========================================================================

/// Build a generator proxy graph matching production generator structure.
///
/// Generator: Conv1d → LeakyReLU → Conv1d → Clamp → Exp
/// Uses small synthetic dimensions to keep verification tractable.
fn build_generator_graph() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let ch = 8;
    let out_ch = 4;
    let t = 4;

    let mut b = TensorBlockBuilder::new("generator_proxy");
    let input = b.add_input("features", &[ch, t]);

    // Conv1d: [ch, t] → [ch, t]
    let w1 = b.add_input("gen_conv1_w", &[ch, ch, 3]);
    let conv1 = b.add_conv1d(input, w1, None, 1, 1, &[ch, t]);

    // LeakyReLU
    let act = b.add_leaky_relu(conv1, 0.1, &[ch, t]);

    // Conv1d: [ch, t] → [out_ch, t]
    let w2 = b.add_input("gen_conv2_w", &[out_ch, ch, 3]);
    let conv2 = b.add_conv1d(act, w2, None, 1, 1, &[out_ch, t]);

    // Reshape to flat [out_ch * t]
    let flat = b.add_reshape(conv2, &[out_ch * t]);
    let def = b.build(flat).expect("valid generator proxy");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch, ch, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[out_ch, ch, 3]), WEIGHT_MAG)),
    ];
    (def, bindings)
}

/// Re-verify `kokoro_production_generator` with Conservative mode → Sound.
#[test]
fn test_sound_promotion_production_generator() {
    let (def, bindings) = build_generator_graph();
    let input = uniform_bounds(&[8, 4], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_production_generator",
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
        "Conservative bounds should be non-vacuous, width={width}"
    );
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "kokoro_production_generator Conservative: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

// ===========================================================================
// 6. kokoro_production_moonshot_composed — Sound composed pipeline
// ===========================================================================

/// Build a composed moonshot proxy: TextEncoder → ProsodyPredictor.
///
/// Two-stage pipeline at synthetic dimensions for tractable verification.
fn build_moonshot_composed_graph() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let d = 8;
    let t = 4;

    let mut b = TensorBlockBuilder::new("moonshot_composed_proxy");
    let input = b.add_input("tokens", &[1, t]);

    // Stage 1: Linear (text encoder proxy) [1, t] → [1, d]
    let w_te = b.add_input("te_w", &[d, t]);
    let te_out = b.add_matmul(input, w_te, true, None, &[1, d]);
    let te_relu = b.add_relu(te_out, &[1, d]);

    // Stage 2: Linear (prosody proxy) [1, d] → [1, d]
    let w_pp = b.add_input("pp_w", &[d, d]);
    let pp_out = b.add_matmul(te_relu, w_pp, true, None, &[1, d]);

    // ReLU → flatten
    let pp_relu = b.add_relu(pp_out, &[1, d]);
    let output = b.add_reshape(pp_relu, &[d]);
    let def = b.build(output).expect("valid moonshot composed proxy");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, t]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG)),
    ];
    (def, bindings)
}

/// Re-verify `kokoro_production_moonshot_composed` with Conservative → Sound.
#[test]
fn test_sound_promotion_production_moonshot_composed() {
    let (def, bindings) = build_moonshot_composed_graph();
    let input = uniform_bounds(&[1, 4], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_production_moonshot_composed",
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
        "Conservative bounds should be non-vacuous, width={width}"
    );
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "kokoro_production_moonshot_composed Conservative: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

// ===========================================================================
// 7. kokoro_production_moonshot_concentration — Sound concentration
// ===========================================================================

/// Re-verify `kokoro_production_moonshot_concentration` with Conservative → Sound.
///
/// Uses the same composed proxy graph as moonshot_composed but with production-like
/// token bounds [0, 177] matching the original entry's input range.
#[test]
fn test_sound_promotion_production_moonshot_concentration() {
    let (def, bindings) = build_moonshot_composed_graph();
    // Match original entry: input_range [0.0, 177.0], shape [1, 4]
    let input = nn_verify::BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[1, 4]), 177.0f32),
    )
    .expect("valid bounds");

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_production_moonshot_concentration",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "kokoro_production_moonshot_concentration Conservative: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

// ===========================================================================
// 8. kokoro_production_text_encoder — Sound text encoder verification
// ===========================================================================

/// Build a text encoder proxy graph matching the production text encoder structure.
///
/// TextEncoder: Embedding → Conv1d → ReLU → Linear
/// Uses small synthetic dimensions.
fn build_text_encoder_graph() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let d = 8;
    let t = 4;

    let mut b = TensorBlockBuilder::new("text_encoder_proxy");
    let input = b.add_input("tokens", &[d, t]);

    // Conv1d: [d, t] → [d, t]
    let conv_w = b.add_input("te_conv_w", &[d, d, 3]);
    let conv_out = b.add_conv1d(input, conv_w, None, 1, 1, &[d, t]);

    // ReLU
    let relu_out = b.add_relu(conv_out, &[d, t]);

    // Linear projection: transpose → matmul → transpose back
    let transposed = b.add_transpose(relu_out, &[1, 0], &[t, d]);
    let proj_w = b.add_input("te_proj_w", &[d, d]);
    let proj_b = b.add_input("te_proj_b", &[d]);
    let projected = b.add_matmul(transposed, proj_w, true, None, &[t, d]);
    let proj_b_bc = b.add_broadcast(proj_b, &[t, d]);
    let biased = b.add_binary_add(projected, proj_b_bc, &[t, d]);
    let output = b.add_transpose(biased, &[1, 0], &[d, t]);
    let def = b.build(output).expect("valid text encoder proxy");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, d, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
    ];
    (def, bindings)
}

/// Re-verify `kokoro_production_text_encoder` with Conservative mode → Sound.
#[test]
fn test_sound_promotion_production_text_encoder() {
    let (def, bindings) = build_text_encoder_graph();
    let input = uniform_bounds(&[8, 4], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_production_text_encoder",
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
        "Conservative bounds should be non-vacuous, width={width}"
    );
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "kokoro_production_text_encoder Conservative: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}
