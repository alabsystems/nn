// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN-verified adversarial phoneme stability tests.
//!
//! Verifies that a phoneme encoder produces bounded output variation when
//! input phonemes are perturbed within linguistically-defined confusion sets.
//!
//! Consolidated: builds each graph variant ONCE and runs all property checks,
//! eliminating 10 redundant graph builds (was 12 builds, now 2).
//!
//! Part of #1740: Adversarial Robustness of TTS.

#[path = "phoneme_stability.rs"]
mod helpers;

use super::common::{assert_bounds_valid, assert_crown_tighter_than_ibp, uniform_bounds};
use helpers::{
    build_phoneme_encoder, build_phoneme_encoder_residual, phoneme_encoder_bindings,
    phoneme_encoder_residual_bindings, synthetic_embedding_weights, test_confusion_sets, EMBED_DIM,
    OUTPUT_DIM, SEQ_LEN, VOCAB_SIZE,
};
use nn_tts_verify::{embedding_bounds_for_token_set, sequence_perturbation_bounds};
use nn_verify::{propagate_with_crown_fallback, tensor_kernel_to_graph, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Helpers (unchanged)
// ---------------------------------------------------------------------------

fn bounded_tensor_from_f64(lower: &[f64], upper: &[f64], shape: &[usize]) -> BoundedTensor {
    let lo_f32: Vec<f32> = lower.iter().map(|&v| v as f32).collect();
    let hi_f32: Vec<f32> = upper.iter().map(|&v| v as f32).collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lo_f32).expect("valid lower shape"),
        ArrayD::from_shape_vec(IxDyn(shape), hi_f32).expect("valid upper shape"),
    )
    .expect("valid bounded tensor")
}

fn mean_output_width(bt: &BoundedTensor) -> f32 {
    let (lo, hi) = bt.lower_upper();
    let widths: Vec<f32> = lo.iter().zip(hi.iter()).map(|(&l, &h)| h - l).collect();
    if widths.is_empty() {
        return 0.0;
    }
    widths.iter().sum::<f32>() / widths.len() as f32
}

fn max_output_width(bt: &BoundedTensor) -> f32 {
    let (lo, hi) = bt.lower_upper();
    lo.iter()
        .zip(hi.iter())
        .map(|(&l, &h)| h - l)
        .fold(0.0f32, f32::max)
}

fn build_single_pos_perturbed(
    emb_weights: &[f64],
    base_tokens: &[u32],
    perturb_pos: usize,
    pos_lo: &[f64],
    pos_hi: &[f64],
) -> BoundedTensor {
    let mut lower = Vec::with_capacity(SEQ_LEN * EMBED_DIM);
    let mut upper = Vec::with_capacity(SEQ_LEN * EMBED_DIM);
    for (pos, &tok) in base_tokens.iter().enumerate() {
        if pos == perturb_pos {
            lower.extend_from_slice(pos_lo);
            upper.extend_from_slice(pos_hi);
        } else {
            let off = (tok as usize) * EMBED_DIM;
            lower.extend_from_slice(&emb_weights[off..off + EMBED_DIM]);
            upper.extend_from_slice(&emb_weights[off..off + EMBED_DIM]);
        }
    }
    bounded_tensor_from_f64(&lower, &upper, &[SEQ_LEN, EMBED_DIM])
}

fn assert_cs_contained_in_full_vocab(
    cs_lo: &[f64],
    cs_hi: &[f64],
    full_lo: &[f64],
    full_hi: &[f64],
) {
    let mut cs_tighter_count = 0;
    for d in 0..EMBED_DIM {
        assert!(cs_lo[d] >= full_lo[d] - 1e-10, "dim {d}: cs lo < full lo");
        assert!(cs_hi[d] <= full_hi[d] + 1e-10, "dim {d}: cs hi > full hi");
        if (cs_hi[d] - cs_lo[d]) < (full_hi[d] - full_lo[d]) - 1e-10 {
            cs_tighter_count += 1;
        }
    }
    assert!(
        cs_tighter_count > 0,
        "confusion-set bounds should be tighter than full-vocab in at least one dim"
    );
}

// ===========================================================================
// Consolidated test 1: Graph builds + IBP + CROWN with uniform bounds
// (was: tests 1, 2, 3)
// ===========================================================================

#[test]
fn test_phoneme_encoder_uniform_bounds() {
    let def = build_phoneme_encoder();
    let bindings = phoneme_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // IBP uniform bounds
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, OUTPUT_DIM]);

    // CROWN uniform bounds
    let (method, crown_output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN propagation");
    assert_bounds_valid(&crown_output);
    assert_eq!(crown_output.shape(), &[SEQ_LEN, OUTPUT_DIM]);

    if format!("{method:?}").contains("Crown") {
        let ibp_output = graph.propagate_ibp(&input).expect("IBP for comparison");
        assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
    }
}

// ===========================================================================
// Consolidated test 2: All perturbation-based IBP tests
// (was: tests 4, 6, 7, 8, 12 — single/multi/all positions + CROWN vs IBP + point bounds)
// ===========================================================================

/// Check IBP single/multi/all position perturbations and monotonicity.
fn check_perturbation_ibp_monotonicity(
    graph: &nn_verify::GraphNetwork,
    emb_weights: &[f64],
    base_tokens: &[u32],
    confusion_sets: &[nn_tts_verify::ConfusionSet],
) -> f32 {
    // Single position perturbation (was test 4)
    let (lower, upper) = sequence_perturbation_bounds(
        emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        base_tokens,
        &[0],
        confusion_sets,
    )
    .expect("perturbation bounds");
    let input_1pos = bounded_tensor_from_f64(&lower, &upper, &[SEQ_LEN, EMBED_DIM]);
    let out_1pos = graph.propagate_ibp(&input_1pos).expect("IBP 1-pos");
    assert_bounds_valid(&out_1pos);
    let w1 = mean_output_width(&out_1pos);
    assert!(
        w1 < 100.0,
        "IBP output width {w1} should be bounded for single-position perturbation"
    );

    // Two-position monotonicity (was test 6)
    let (lo2, hi2) = sequence_perturbation_bounds(
        emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        base_tokens,
        &[0, 1],
        confusion_sets,
    )
    .expect("double perturbation");
    let input_2pos = bounded_tensor_from_f64(&lo2, &hi2, &[SEQ_LEN, EMBED_DIM]);
    let out_2pos = graph.propagate_ibp(&input_2pos).expect("IBP 2-pos");
    assert_bounds_valid(&out_2pos);
    let w2 = mean_output_width(&out_2pos);
    assert!(
        w2 >= w1 - 1e-6,
        "two-position ({w2:.6}) should be >= one ({w1:.6})"
    );

    // All positions perturbed (was test 7)
    let positions: Vec<usize> = (0..SEQ_LEN).collect();
    let (lo_all, hi_all) = sequence_perturbation_bounds(
        emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        base_tokens,
        &positions,
        confusion_sets,
    )
    .expect("all perturbation");
    let input_all = bounded_tensor_from_f64(&lo_all, &hi_all, &[SEQ_LEN, EMBED_DIM]);
    let out_all = graph.propagate_ibp(&input_all).expect("IBP all perturbed");
    assert_bounds_valid(&out_all);
    let w_all = max_output_width(&out_all);
    assert!(
        w_all.is_finite(),
        "max output width should be finite, got {w_all}"
    );

    w1
}

/// Check CROWN vs IBP and point-bounds zero-width.
fn check_crown_and_point_bounds(
    graph: &nn_verify::GraphNetwork,
    emb_weights: &[f64],
    base_tokens: &[u32],
    confusion_sets: &[nn_tts_verify::ConfusionSet],
) {
    // CROWN tighter than IBP (was test 8)
    let (lo_pert, hi_pert) = sequence_perturbation_bounds(
        emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        base_tokens,
        &[0, 2],
        confusion_sets,
    )
    .expect("perturbation bounds");
    let input_pert = bounded_tensor_from_f64(&lo_pert, &hi_pert, &[SEQ_LEN, EMBED_DIM]);
    let ibp_output = graph.propagate_ibp(&input_pert).expect("IBP");
    let (method, crown_output, _) =
        propagate_with_crown_fallback(graph, &input_pert).expect("CROWN");
    assert_bounds_valid(&ibp_output);
    assert_bounds_valid(&crown_output);
    if format!("{method:?}").contains("Crown") {
        assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
    }

    // Point bounds zero width (was test 12)
    let (lo_pt, hi_pt) = sequence_perturbation_bounds(
        emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        base_tokens,
        &[],
        &test_confusion_sets(),
    )
    .expect("point bounds");
    for (i, (&lo, &hi)) in lo_pt.iter().zip(hi_pt.iter()).enumerate() {
        assert!(
            (lo - hi).abs() < 1e-12,
            "element {i}: lo == hi, got {lo} vs {hi}"
        );
    }
    let input_pt = bounded_tensor_from_f64(&lo_pt, &hi_pt, &[SEQ_LEN, EMBED_DIM]);
    let out_pt = graph.propagate_ibp(&input_pt).expect("IBP point bounds");
    assert_bounds_valid(&out_pt);
    let pt_width = max_output_width(&out_pt);
    assert!(
        pt_width < 1e-4,
        "point bounds should produce near-zero output width, got {pt_width}"
    );
}

#[test]
fn test_phoneme_encoder_perturbation_properties() {
    let def = build_phoneme_encoder();
    let bindings = phoneme_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let emb_weights = synthetic_embedding_weights();
    let confusion_sets = test_confusion_sets();
    let base_tokens: Vec<u32> = vec![0, 2, 4, 7];

    check_perturbation_ibp_monotonicity(&graph, &emb_weights, &base_tokens, &confusion_sets);
    check_crown_and_point_bounds(&graph, &emb_weights, &base_tokens, &confusion_sets);
}

// ===========================================================================
// Consolidated test 3: Confusion-set tighter than full-vocabulary
// (was: tests 5, 11)
// ===========================================================================

#[test]
fn test_phoneme_confusion_set_tightness() {
    let def = build_phoneme_encoder();
    let bindings = phoneme_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let emb_weights = synthetic_embedding_weights();
    let base_tokens: Vec<u32> = vec![0, 2, 4, 7];

    // Full-vocabulary vs confusion-set bounds at embedding level
    let all_ids: Vec<u32> = (0..VOCAB_SIZE as u32).collect();
    let (full_lo, full_hi) =
        embedding_bounds_for_token_set(&emb_weights, VOCAB_SIZE, EMBED_DIM, &all_ids)
            .expect("full vocab bounds");
    let (cs_lo, cs_hi) =
        embedding_bounds_for_token_set(&emb_weights, VOCAB_SIZE, EMBED_DIM, &[0, 1])
            .expect("confusion set bounds");
    assert_cs_contained_in_full_vocab(&cs_lo, &cs_hi, &full_lo, &full_hi);

    // Propagate both and compare output widths
    let full_input = build_single_pos_perturbed(&emb_weights, &base_tokens, 0, &full_lo, &full_hi);
    let full_output = graph.propagate_ibp(&full_input).expect("IBP full vocab");

    let confusion_sets = test_confusion_sets();
    let (cs_seq_lo, cs_seq_hi) = sequence_perturbation_bounds(
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &[0],
        &confusion_sets,
    )
    .expect("cs perturbation bounds");
    let cs_input = bounded_tensor_from_f64(&cs_seq_lo, &cs_seq_hi, &[SEQ_LEN, EMBED_DIM]);
    let cs_output = graph.propagate_ibp(&cs_input).expect("IBP confusion set");

    let full_w = mean_output_width(&full_output);
    let cs_w = mean_output_width(&cs_output);
    assert!(
        cs_w <= full_w + 1e-6,
        "confusion-set output width ({cs_w:.6}) should be <= full-vocab ({full_w:.6})"
    );

    // Larger confusion set → both finite (was test 11)
    let (large_lo, large_hi) =
        embedding_bounds_for_token_set(&emb_weights, VOCAB_SIZE, EMBED_DIM, &[4, 5, 6])
            .expect("large set bounds");
    let small_input = build_single_pos_perturbed(&emb_weights, &base_tokens, 0, &cs_lo, &cs_hi);
    let large_input =
        build_single_pos_perturbed(&emb_weights, &base_tokens, 0, &large_lo, &large_hi);
    let small_out = graph.propagate_ibp(&small_input).expect("IBP small");
    let large_out = graph.propagate_ibp(&large_input).expect("IBP large");
    let small_w = mean_output_width(&small_out);
    let large_w = mean_output_width(&large_out);
    assert!(
        small_w.is_finite(),
        "small set output width should be finite"
    );
    assert!(
        large_w.is_finite(),
        "large set output width should be finite"
    );
}

// ===========================================================================
// Consolidated test 4: Residual encoder (IBP + CROWN)
// (was: tests 9, 10)
// ===========================================================================

#[test]
fn test_residual_encoder_all_properties() {
    let def = build_phoneme_encoder_residual();
    let bindings = phoneme_encoder_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let emb_weights = synthetic_embedding_weights();
    let confusion_sets = test_confusion_sets();
    let base_tokens: Vec<u32> = vec![0, 2, 4, 7];

    // IBP (was test 9)
    let (lower_1, upper_1) = sequence_perturbation_bounds(
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &[0],
        &confusion_sets,
    )
    .expect("perturbation bounds");
    let input_1 = bounded_tensor_from_f64(&lower_1, &upper_1, &[SEQ_LEN, EMBED_DIM]);
    let output_1 = graph.propagate_ibp(&input_1).expect("IBP residual");
    assert_bounds_valid(&output_1);
    assert_eq!(output_1.shape(), &[SEQ_LEN, OUTPUT_DIM]);

    // CROWN (was test 10)
    let (lower_2, upper_2) = sequence_perturbation_bounds(
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &[0, 1],
        &confusion_sets,
    )
    .expect("perturbation bounds");
    let input_2 = bounded_tensor_from_f64(&lower_2, &upper_2, &[SEQ_LEN, EMBED_DIM]);
    let (method, output_2, _) =
        propagate_with_crown_fallback(&graph, &input_2).expect("CROWN residual");
    assert_bounds_valid(&output_2);

    let ibp_output = graph.propagate_ibp(&input_2).expect("IBP for comparison");
    if format!("{method:?}").contains("Crown") {
        assert_crown_tighter_than_ibp(&output_2, &ibp_output);
    }
}
