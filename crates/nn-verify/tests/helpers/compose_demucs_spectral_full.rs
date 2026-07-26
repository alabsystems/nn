// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Demucs spectral full decoder block composition.
//!
//! Composes all 3 spectral decoder sub-defs into a single NY
//! `GraphNetwork`:
//!
//! ```text
//! Encoder:   data [IN_CH, F*T] → Conv1d(stride) → GELU
//! Rewrite:   skip_add → Reshape[C,F,T] → Conv2d(3×3) → Reshape[2C,F*T] → GLU
//! DConv:     Conv1d(dilated) → GN(G=1) → GELU → Conv1d(1×1) → GN(G=1) → GLU → LS → residual (×1)
//! ConvTr:    ConvTranspose1d → Narrow(trim) → GELU
//! ```
//!
//! The spectral decoder in Demucs operates on [C, F*T] flattened data.
//! The rewrite stage reshapes to [C, F, T] for Conv2d, then back to [2C, F*T]
//! for GLU. DConv operates on the flattened representation. ConvTranspose
//! upsamples along the trailing dimension (modeling frequency upsampling).
//!
//! Single-depth, small dims (ENC_CH=8, F=4, T=4) for NY tractability.
//! DConv depth=1, weight_mag=0.001 (same as temporal full pipeline).
//!
//! Part of #779 Phase B — spectral decoder full block composition.
//!
//! Builder infrastructure extracted to `helpers/spectral_full.rs`
//! for 500-line compliance.

use super::common;

#[path = "spectral_full.rs"]
mod helpers;

use common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use helpers::{build_spectral_full, spectral_full_bindings, FT, IN_CH};
use nn_verify::{tensor_kernel_to_graph, VerificationSoundnessMode};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full spectral decoder pipeline TensorKernelDef validates.
#[test]
fn test_spectral_full_def_validates() {
    let (def, _) = build_spectral_full();
    def.validate()
        .expect("spectral full pipeline def should validate");
}

/// Full spectral decoder pipeline translates to NY GraphNetwork.
#[test]
fn test_spectral_full_graph_builds() {
    let (def, target_len) = build_spectral_full();
    assert!(target_len > 0, "output length should be > 0");

    let bindings = spectral_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("spectral full pipeline graph should translate");

    // Encoder + Rewrite(Conv2d+GLU) + DConv(deep) + ConvTranspose + GELU → many nodes
    assert!(
        graph.num_nodes() >= 20,
        "spectral full graph should have >= 20 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full spectral decoder pipeline.
///
/// Exercises: encoder → Conv2d rewrite → GLU → DConv → ConvTranspose → trim → GELU.
/// IBP bounds expected to be wide through decomposed GroupNorm in DConv but finite.
#[test]
fn test_spectral_full_ibp_propagates() {
    let (def, target_len) = build_spectral_full();
    let bindings = spectral_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, FT], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through spectral full pipeline");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[IN_CH, target_len],
        "output shape mismatch: expected [{}, {}], got {:?}",
        IN_CH,
        target_len,
        output.lower_upper().0.shape()
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Spectral full pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    // Bounds-magnitude sanity check: spectral pipeline with small weights (0.001)
    // and [-1, 1] input should produce output bounds well within ±1.0. Observed: ~±3.9e-6.
    assert!(
        lo_min.abs() < 1.0,
        "IBP lower bound magnitude should be < 1.0, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1.0,
        "IBP upper bound magnitude should be < 1.0, got {hi_max}"
    );
}

/// CROWN propagation through the full spectral decoder pipeline.
///
/// CROWN may fall back to IBP on Conv2d multiplicative interactions or DConv's
/// decomposed GroupNorm. Uses `assert_crown_tighter_when_not_fallback` to
/// verify CROWN produces tighter bounds than IBP when CROWN succeeds.
#[test]
fn test_spectral_full_crown_propagation() {
    let (def, target_len) = build_spectral_full();
    let bindings = spectral_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, FT], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[IN_CH, target_len],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Spectral full pipeline: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    // Magnitude assertions matching IBP counterpart: small weights (0.001) and
    // [-1, 1] input should produce output bounds well within ±1.0.
    assert!(
        lo_min.abs() < 1.0,
        "CROWN: lower bound magnitude should be < 1.0, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1.0,
        "CROWN: upper bound magnitude should be < 1.0, got {hi_max}"
    );
}

/// Full spectral pipeline verify and record under "demucs_spectral_full" key.
#[test]
fn test_spectral_full_verify_and_record() {
    let (def, target_len) = build_spectral_full();
    let bindings = spectral_full_bindings();
    let input = uniform_bounds(&[IN_CH, FT], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "demucs_spectral_full");
    assert_eq!(result.num_variables, 1, "single Variable input (data)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[IN_CH, target_len]);

    // Soundness provenance must be set (#1984).
    // Sound or Heuristic depending on whether normalization heuristics were used.
    assert!(
        matches!(
            result.verification.soundness_mode,
            VerificationSoundnessMode::Sound | VerificationSoundnessMode::Heuristic
        ),
        "soundness mode should be Sound or Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
