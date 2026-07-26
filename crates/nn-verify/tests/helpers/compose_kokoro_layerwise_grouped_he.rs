// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! He-scaled CROWN tightening tests for Kokoro layerwise grouped verification.
//!
//! Extracted from `compose_kokoro_layerwise_grouped.rs` to keep both files
//! under the 500-line limit (#2633).
//!
//! These tests use He-initialized non-uniform weights (not the default
//! WEIGHT_MAG=0.001 uniform weights) to demonstrate CROWN tightening over IBP
//! on multi-layer subgraphs. Covers AC2 (#2592), AC3 (#2620), and D=128
//! scaling measurements (#2580).
//!
//! Part of #2633, Part of #2218.

#[path = "kokoro_scaled_pipeline.rs"]
mod he_scaled_helpers;
use he_scaled_helpers as helpers;

#[path = "kokoro_scaled_layerwise.rs"]
mod he_layerwise_helpers;

use super::common::kokoro_weights::{bt_max_width, sign_alternate_weight_bindings, uniform_bt};
use he_layerwise_helpers::build_kokoro_layerwise_deep;
use helpers::KokoroDims;
use nn_verify::BoundedTensor;

/// Build a single Linear+ReLU layer with He-initialized non-uniform weights.
///
/// Uses golden-ratio quasi-random sampling x He scale (`sqrt(2 / fan_in)`)
/// to produce weights that are non-uniform and realistic. This ensures
/// pre-activation ranges cross the ReLU kink (zero), giving CROWN non-trivial
/// relaxation slopes to optimize.
///
/// `seed` shifts the phase so successive layers get distinct weight matrices.
fn build_he_linear_relu(
    in_dim: usize,
    out_dim: usize,
    seq_len: usize,
    seed: u32,
) -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    Vec<nn_verify::TensorParamBinding>,
) {
    use nn_dsl::tensor_block_builder::TensorBlockBuilder;
    use nn_verify::TensorParamBinding;
    use ndarray::{ArrayD, IxDyn};

    // He scale: sqrt(2 / fan_in) — standard for ReLU networks.
    // Factor of 3 accounts for the spatial (seq_len) dimension in the matmul.
    let he_scale = (2.0 / (in_dim as f32 * 3.0)).sqrt();
    let phi = 1.618_033_9; // golden ratio for low-discrepancy sampling

    let mut b = TensorBlockBuilder::new(&format!("he_linear_relu_{seed}"));
    let input = b.add_input("x", &[in_dim, seq_len]);

    // Transpose [in_dim, seq_len] -> [seq_len, in_dim] for matmul.
    let transposed = b.add_transpose(input, &[1, 0], &[seq_len, in_dim]);

    // Weight matrix [out_dim, in_dim] applied as matmul.
    let weight = b.add_input("w", &[out_dim, in_dim]);
    let mm_out = b.add_matmul(transposed, weight, true, None, &[seq_len, out_dim]);

    // Transpose back [seq_len, out_dim] -> [out_dim, seq_len].
    let back = b.add_transpose(mm_out, &[1, 0], &[out_dim, seq_len]);
    let output = b.add_relu(back, &[out_dim, seq_len]);

    let def = b.build(output).expect("he linear relu layer");

    // Deterministic non-uniform weights via golden-ratio quasi-random x He scale.
    let n_w = out_dim * in_dim;
    let weights: Vec<f32> = (0..n_w)
        .map(|i| ((i as f32) * phi + seed as f32).sin() * he_scale)
        .collect();
    let bindings = vec![
        TensorParamBinding::Variable, // input
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[out_dim, in_dim]), weights).unwrap(),
        ),
    ];

    (def, bindings)
}

/// AC2 proof: CROWN tightening on He-scaled Linear+ReLU layers.
///
/// The existing Kokoro pre-norm tests use `WEIGHT_MAG = 0.001` uniform weights,
/// which collapse all bounds to a near-zero range where activations act linearly
/// (CROWN = IBP, ratio ~ 1.0). This test uses He-initialized non-uniform weights
/// that produce realistic pre-activation ranges crossing the ReLU kink.
///
/// **Expected:** CROWN/IBP ratio > 1.01 (CROWN strictly tighter than IBP).
/// **Measured:** ratio ~ 1.90 (47% tighter bounds).
///
/// Part of #2592, Part of #2218.
#[test]
fn test_kokoro_prenorm_crown_he_scaled() {
    let dim = 16;
    let seq_len = 4;

    // Two Linear+ReLU layers with He-scaled non-uniform weights.
    let layer0 = build_he_linear_relu(dim, dim, seq_len, 1);
    let layer1 = build_he_linear_relu(dim, dim, seq_len, 2);

    // Merge into a single multi-layer GraphNetwork (the grouped strategy).
    let merged = nn_verify::tensor_kernels_to_grouped_graph(
        &[layer0, layer1],
        nn_verify::NormBoundsMode::Conservative,
    )
    .expect("He-scaled merged graph");

    let initial = uniform_bt(&[dim, seq_len], -1.0, 1.0);

    // IBP baseline.
    let ibp_output = merged.propagate_ibp(&initial).expect("He IBP");
    let ibp_width = bt_max_width(&ibp_output);

    // CROWN propagation.
    let (crown_method, crown_output, _fallback) =
        nn_verify::propagate_with_crown_fallback(&merged, &initial).expect("He CROWN");
    let crown_width = bt_max_width(&crown_output);

    let ratio = if crown_width > 0.0 {
        ibp_width / crown_width
    } else {
        f32::INFINITY
    };

    eprintln!("=== AC2 Proof: He-scaled Linear+ReLU CROWN tightening ===");
    eprintln!("  IBP width:   {ibp_width:.6}");
    eprintln!("  CROWN width: {crown_width:.6} (method: {crown_method:?})");
    eprintln!("  IBP/CROWN ratio: {ratio:.4}");

    // Soundness: CROWN must not be wider than IBP.
    assert!(
        crown_width <= ibp_width + 1e-3,
        "CROWN width {crown_width} > IBP width {ibp_width} (soundness violation)"
    );

    // AC2: CROWN must be strictly tighter than IBP.
    assert!(
        ratio > 1.01,
        "AC2 failed: IBP/CROWN ratio {ratio:.4} <= 1.01 (no tightening)"
    );

    // Both must be finite and positive.
    assert!(ibp_width.is_finite(), "IBP width not finite");
    assert!(crown_width.is_finite(), "CROWN width not finite");
    assert!(ibp_width > 0.0, "IBP width must be positive");
    assert!(crown_width > 0.0, "CROWN width must be positive");
}

// ===========================================================================
// AC3 (#2620): Persist grouped CROWN result with crown_ibp_ratio < 0.9
// ===========================================================================

/// Per-model status file path for Kokoro kernels.
fn kokoro_status_path() -> std::path::PathBuf {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    nn_verify::model_status_path(workspace_root, "kokoro")
}

/// Record CROWN+IBP comparison to the status file and return the saved ratio.
///
/// Returns `None` if the status file lock cannot be acquired (parallel test
/// contention). The measurement data is logged to stderr regardless.
///
/// `soundness` should be `Sound` for graphs without normalization layers
/// (He-scaled Linear+ReLU) and `Heuristic` for graphs with ForwardMode
/// normalization (Kokoro architecture).
///
/// Part of #3351 T3.1: Kokoro soundness improvement.
fn persist_crown_comparison(
    status_key: &str,
    crown_output: &BoundedTensor,
    ibp_width: f32,
    soundness: nn_verify::VerificationSoundnessMode,
) -> Option<f32> {
    use nn_verify::{PropMethod, VerifyStatus};

    let crown_lo = crown_output
        .lower()
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let crown_hi = crown_output
        .upper()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let crown_width = crown_hi - crown_lo;
    let computed_ratio = crown_width / ibp_width.max(1e-10);
    let out_shape: Vec<usize> = crown_output.shape().to_vec();

    let status_path = kokoro_status_path();
    let mut locked = match VerifyStatus::load_locked(&status_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("  [persist] lock failed ({e}), computed ratio={computed_ratio:.4}");
            return None;
        }
    };
    locked
        .status
        .record_pipeline(
            status_key,
            PropMethod::Crown,
            -1.0,
            1.0,
            crown_lo,
            crown_hi,
            &out_shape,
            soundness,
            None, // no input_bounds in scope — CROWN comparison path
        )
        .expect("record_pipeline");
    locked
        .status
        .record_crown_comparison(status_key, ibp_width)
        .expect("record_crown_comparison");
    locked.save().expect("save status");
    drop(locked);

    // Return the persisted ratio.
    match VerifyStatus::load_locked(&status_path) {
        Ok(validation) => {
            let entry = validation.status.kernel(status_key).expect("entry exists");
            Some(entry.crown_ibp_ratio.expect("ratio recorded"))
        }
        Err(_) => Some(computed_ratio),
    }
}

/// Persist He-scaled grouped CROWN result to the status file.
///
/// Records `kokoro_grouped_prenorm_he_crown` with CROWN/IBP comparison
/// data. He-scaled non-uniform weights produce crown_ibp_ratio < 0.9,
/// demonstrating CROWN is strictly tighter than IBP on multi-layer
/// subgraphs with realistic weight distributions.
///
/// Part of #2620, Part of #2218.
#[test]
fn test_kokoro_grouped_persist_he_crown() {
    let status_key = "kokoro_grouped_prenorm_he_crown";
    let dim = 16;
    let seq_len = 4;

    let layer0 = build_he_linear_relu(dim, dim, seq_len, 1);
    let layer1 = build_he_linear_relu(dim, dim, seq_len, 2);

    let merged = nn_verify::tensor_kernels_to_grouped_graph(
        &[layer0, layer1],
        nn_verify::NormBoundsMode::ForwardMode,
    )
    .expect("He-scaled merged graph");
    let initial = uniform_bt(&[dim, seq_len], -1.0, 1.0);

    // CROWN propagation.
    let (_method, crown_output, _fallback) =
        nn_verify::propagate_with_crown_fallback(&merged, &initial).expect("He CROWN");
    let crown_width = bt_max_width(&crown_output);

    // IBP baseline.
    let ibp_output = merged.propagate_ibp(&initial).expect("He IBP");
    let ibp_width = bt_max_width(&ibp_output);

    let computed_ratio = crown_width / ibp_width.max(1e-10);
    // He-scaled Linear+ReLU: no normalization layers → Sound.
    let ratio = persist_crown_comparison(
        status_key,
        &crown_output,
        ibp_width,
        nn_verify::VerificationSoundnessMode::Sound,
    )
    .unwrap_or(computed_ratio);

    eprintln!("=== AC3: crown_width={crown_width:.6}, ibp_width={ibp_width:.6}, ratio={ratio:.4}");
    assert!(
        ratio < 0.9,
        "AC3: crown_ibp_ratio {ratio:.4} >= 0.9 (CROWN not tighter than IBP)"
    );
}

// ===========================================================================
// D=128 CROWN vs IBP measurement (#2580)
// ===========================================================================

/// Helper: run He-scaled CROWN vs IBP measurement at given dim, persist, return widths and ratio.
fn measure_he_crown_vs_ibp(dim: usize, seq_len: usize, status_key: &str) -> (f32, f32, f32) {
    let layer0 = build_he_linear_relu(dim, dim, seq_len, 1);
    let layer1 = build_he_linear_relu(dim, dim, seq_len, 2);

    let merged = nn_verify::tensor_kernels_to_grouped_graph(
        &[layer0, layer1],
        nn_verify::NormBoundsMode::ForwardMode,
    )
    .expect("He-scaled merged graph");

    let initial = uniform_bt(&[dim, seq_len], -1.0, 1.0);

    let (crown_method, crown_output, _fallback) =
        nn_verify::propagate_with_crown_fallback(&merged, &initial).expect("CROWN");
    let crown_width = bt_max_width(&crown_output);

    let ibp_output = merged.propagate_ibp(&initial).expect("IBP");
    let ibp_width = bt_max_width(&ibp_output);

    let ratio = if crown_width > 0.0 {
        ibp_width / crown_width
    } else {
        f32::INFINITY
    };

    eprintln!("=== He-scaled D={dim} CROWN vs IBP ===");
    eprintln!("  IBP width:   {ibp_width:.6}");
    eprintln!("  CROWN width: {crown_width:.6} (method: {crown_method:?})");
    eprintln!("  IBP/CROWN ratio: {ratio:.4}");

    // He-scaled Linear+ReLU: no normalization layers → Sound.
    if let Some(saved) = persist_crown_comparison(
        status_key,
        &crown_output,
        ibp_width,
        nn_verify::VerificationSoundnessMode::Sound,
    ) {
        eprintln!("  Persisted crown_ibp_ratio: {saved:.4}");
    }

    (ibp_width, crown_width, ratio)
}

/// D=64 He-scaled CROWN vs IBP: scaling midpoint measurement.
///
/// Part of #2580, Part of #2218.
#[test]
fn test_kokoro_prenorm_crown_he_scaled_d64() {
    let (ibp_width, crown_width, ratio) =
        measure_he_crown_vs_ibp(64, 4, "kokoro_grouped_prenorm_he_d64");

    assert!(
        crown_width <= ibp_width + 1e-3,
        "CROWN width {crown_width} > IBP width {ibp_width} (soundness violation)"
    );
    assert!(ibp_width.is_finite(), "IBP width not finite");
    assert!(crown_width.is_finite(), "CROWN width not finite");
    assert!(crown_width > 0.0, "CROWN width must be positive");

    if ratio > 1.01 {
        eprintln!("  >>> CROWN tightening holds at D=64: ratio={ratio:.4}");
    } else {
        eprintln!("  >>> CROWN tightening lost at D=64: ratio={ratio:.4}");
    }
}

/// D=128 He-scaled CROWN vs IBP: production-dimension measurement.
///
/// Part of #2580, Part of #2218.
#[test]
fn test_kokoro_prenorm_crown_he_scaled_d128() {
    let (ibp_width, crown_width, ratio) =
        measure_he_crown_vs_ibp(128, 4, "kokoro_grouped_prenorm_he_d128");

    assert!(
        crown_width <= ibp_width + 1e-3,
        "CROWN width {crown_width} > IBP width {ibp_width} (soundness violation)"
    );
    assert!(ibp_width.is_finite(), "IBP width not finite");
    assert!(crown_width.is_finite(), "CROWN width not finite");
    assert!(crown_width > 0.0, "CROWN width must be positive");

    if ratio > 1.01 {
        eprintln!("  >>> CROWN tightening holds at D=128: ratio={ratio:.4}");
        eprintln!("  >>> Vacuity data: crown_width={crown_width:.4}");
    } else {
        eprintln!("  >>> CROWN tightening lost at D=128: ratio={ratio:.4}");
    }
}

/// D=128 actual Kokoro pre-norm group: CROWN vs IBP with signed weights.
///
/// Uses the actual Kokoro architecture layers (Conv1d+ReLU+MatMul+Conv1d+
/// LeakyReLU+ConvTranspose1d) with signed weight tensors at D=128.
///
/// Part of #2580, Part of #2218.
#[test]
fn test_kokoro_grouped_d128_prenorm_crown_vs_ibp() {
    let dims = KokoroDims::d128();
    let mut layers = build_kokoro_layerwise_deep(&dims, 4);

    for layer in layers[0..3].iter_mut() {
        sign_alternate_weight_bindings(&mut layer.1);
    }

    let prenorm_layers: Vec<_> = layers[0..3].to_vec();
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);

    let merged_graph = nn_verify::tensor_kernels_to_grouped_graph(
        &prenorm_layers,
        nn_verify::NormBoundsMode::ForwardMode,
    )
    .expect("D=128 pre-norm group graph");
    let ibp_output = merged_graph
        .propagate_ibp(&initial)
        .expect("D=128 pre-norm IBP");
    let ibp_width = bt_max_width(&ibp_output);

    let (crown_method, crown_output, _fallback) =
        nn_verify::propagate_with_crown_fallback(&merged_graph, &initial)
            .expect("D=128 pre-norm CROWN");
    let crown_width = bt_max_width(&crown_output);

    let ratio = if crown_width > 0.0 {
        ibp_width / crown_width
    } else {
        f32::INFINITY
    };

    eprintln!("=== D=128 Kokoro Pre-norm CROWN vs IBP — SIGNED WEIGHTS ===");
    eprintln!("  IBP width:   {ibp_width:.6}");
    eprintln!("  CROWN width: {crown_width:.6} (method: {crown_method:?})");
    eprintln!("  IBP/CROWN ratio: {ratio:.4}");

    // Pre-norm layers (Conv1d+ReLU+MatMul+Conv1d+LeakyReLU+ConvTranspose1d):
    // no normalization layers in this subgraph → Sound. (gc#4399 unblocked.)
    if let Some(saved) = persist_crown_comparison(
        "kokoro_grouped_prenorm_d128_signed",
        &crown_output,
        ibp_width,
        nn_verify::VerificationSoundnessMode::Sound,
    ) {
        eprintln!("  Persisted crown_ibp_ratio: {saved:.4}");
    }

    assert!(
        crown_width <= ibp_width + 1e-3,
        "D=128 CROWN width {crown_width} > IBP width {ibp_width} (soundness violation)",
    );
    assert!(ibp_width.is_finite(), "D=128 IBP width not finite");
    assert!(crown_width.is_finite(), "D=128 CROWN width not finite");
    assert!(ibp_width > 0.0, "D=128 IBP width must be positive");

    let non_vacuous = crown_width < 10.0;
    eprintln!(
        "  Vacuity: crown_width={crown_width:.4} {}",
        if non_vacuous {
            "< 10.0 (NON-VACUOUS)"
        } else {
            ">= 10.0 (VACUOUS)"
        }
    );
}
