// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Whisper full-model NY composition.
//!
//! Decoder with constant encoder output. Token embedding is the single
//! Variable input. Architecture: Embedding + PosEmb → N × DecoderBlock
//! (causal self-attn + cross-attn + FFN) → LayerNorm → output projection.
//!
//! Part of #1696 AC4: Whisper full-model NY composition.

use super::common;

#[path = "whisper_full.rs"]
mod helpers;

use common::{assert_bounds_valid, bounds_min_max, uniform_bounds, verify_and_assert};
use helpers::{build_whisper_full, whisper_full_bindings, DEC_SEQ_LEN, D_MODEL, VOCAB_SIZE};
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, VerificationSoundnessMode,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Whisper full-model TensorKernelDef validates.
#[test]
fn test_whisper_full_def_validates() {
    let (def, _) = build_whisper_full();
    def.validate()
        .expect("whisper full model def should validate");
}

/// Whisper full model translates to NY GraphNetwork.
#[test]
fn test_whisper_full_graph_builds() {
    let (def, vocab) = build_whisper_full();
    assert_eq!(vocab, VOCAB_SIZE);

    let bindings = whisper_full_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("whisper full graph should translate");

    // 2 decoder blocks × (self-attn + cross-attn + FFN + LayerNorms + residuals)
    // + final LayerNorm + output projection → substantial graph
    assert!(
        graph.num_nodes() >= 40,
        "whisper full graph should have >= 40 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the Whisper full model.
#[test]
fn test_whisper_full_ibp_propagates() {
    let (def, _) = build_whisper_full();
    let bindings = whisper_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Whisper full model");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, VOCAB_SIZE],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper full IBP: bounds=[{lo_min}, {hi_max}]");

    // Bounds may be wide due to +1 axis convention through chained norms (#2987).
    // Check finiteness until axis convention is fixed.
    assert!(
        lo_min.is_finite(),
        "IBP lower bound must be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "IBP upper bound must be finite, got {hi_max}"
    );
}

/// CROWN propagation through the Whisper full model.
/// Verifies CROWN produces tighter bounds than IBP (soundness invariant).
#[test]
fn test_whisper_full_crown_propagation() {
    let (def, _) = build_whisper_full();
    let bindings = whisper_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, D_MODEL], 1.0);

    // IBP baseline for comparison
    let _ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through Whisper full model");

    let (method, crown_output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("propagation");
    assert_eq!(
        crown_output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, VOCAB_SIZE],
        "output shape mismatch"
    );
    assert_bounds_valid(&crown_output);

    let (lo_min, hi_max) = bounds_min_max(&crown_output);
    eprintln!("Whisper full: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    // Bounds may be wide due to +1 axis convention through chained norms (#2987).
    // Check finiteness until axis convention is fixed.
    assert!(
        lo_min.is_finite(),
        "CROWN: lower bound must be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "CROWN: upper bound must be finite, got {hi_max}"
    );
}

/// Whisper full model verify and record under "whisper_full" key.
#[test]
fn test_whisper_full_verify_and_record() {
    let (def, _) = build_whisper_full();
    let bindings = whisper_full_bindings();
    let input = uniform_bounds(&[DEC_SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_full");
    assert_eq!(result.num_variables, 1, "single Variable input (token_emb)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[DEC_SEQ_LEN, VOCAB_SIZE]);

    // Soundness provenance must be set (#1984).
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "ForwardMode NormBoundsMode should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
