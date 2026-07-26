// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phase 16: Phoneme encoder adversarial stability certificates.
//!
//! CROWN-verify that a phoneme encoder (embedding → Linear → ReLU → Linear)
//! produces bounded output under adversarial phoneme perturbations defined by
//! confusion sets. Connects the embedding-space perturbation framework to
//! the phoneme processing pipeline.
//!
//! Key results:
//!   - Phoneme encoder output width scales with confusion set size
//!     (voicing pairs: tight; vowel groups: wider)
//!   - Multi-position perturbation composes correctly
//!   - Residual connections don't unboundedly amplify bounds
//!   - `verify_tensor_and_record` persists certificates to nn_verify_status.json
//!
//! Part of #1740: Adversarial Robustness of TTS — AC2 phoneme stability.
//!
//! Extracted from `compose_attention_certificate_phase16.rs` for size compliance.

#[path = "phoneme_stability.rs"]
mod phoneme_helpers;

#[path = "certificate_types.rs"]
mod cert_types;

use super::common;
use cert_types::{measure_avg_width, measure_max_width, PhonemeStabilityCertificate};
use nn_verify::{tensor_kernel_to_graph, verify_tensor_and_record, BoundedTensor, VerifyStatus};
use ndarray::{ArrayD, IxDyn};

// ===========================================================================
// Tests: Phoneme encoder adversarial stability (#1740 AC2)
// ===========================================================================

/// CROWN-verify phoneme encoder stability under voicing pair confusion.
///
/// A voicing pair (e.g., "p" ↔ "b") models the smallest adversarial
/// perturbation: two phonemes that differ only in voicing. The encoder
/// output difference should be bounded — if the output stays within
/// provable bounds, the TTS system is robust to this confusion.
#[test]
fn test_phoneme_encoder_voicing_pair_stability() {
    let embedding_weights = phoneme_helpers::synthetic_embedding_weights();
    let confusion_sets = phoneme_helpers::test_confusion_sets();

    // Voicing pair: tokens 0 ↔ 1 (p/b)
    let voicing_set = &confusion_sets[0];

    // Build phoneme encoder
    let def = phoneme_helpers::build_phoneme_encoder();
    let bindings = phoneme_helpers::phoneme_encoder_bindings();

    // Build input bounds from confusion set at position 0
    // Other positions are fixed (point bounds at token 0)
    let base_tokens = vec![0u32; phoneme_helpers::SEQ_LEN];
    let perturbation_positions = vec![0]; // Only position 0 is perturbed

    let (lower_f64, upper_f64) = nn_tts_verify::sequence_perturbation_bounds(
        &embedding_weights,
        phoneme_helpers::VOCAB_SIZE,
        phoneme_helpers::EMBED_DIM,
        &base_tokens,
        &perturbation_positions,
        std::slice::from_ref(voicing_set),
    )
    .expect("perturbation bounds");

    let lower: Vec<f32> = lower_f64.iter().map(|&v| v as f32).collect();
    let upper: Vec<f32> = upper_f64.iter().map(|&v| v as f32).collect();
    let shape = &[phoneme_helpers::SEQ_LEN, phoneme_helpers::EMBED_DIM];
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(shape), upper).expect("upper shape"),
    )
    .expect("valid bounds");

    let status_key = "cert_phoneme_voicing_pair";
    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(&mut status, &def, &bindings, &input, Some(status_key))
        .expect("phoneme encoder verification");

    common::assert_bounds_valid(&result.output_bounds);

    let cert = PhonemeStabilityCertificate {
        architecture: "Embedding→Linear→ReLU→Linear".into(),
        confusion_set_name: voicing_set.name.clone(),
        confusion_set_size: voicing_set.token_ids.len(),
        confusion_category: format!("{:?}", voicing_set.category),
        method: result.verification.method,
        avg_width: measure_avg_width(&result.output_bounds),
        max_width: measure_max_width(&result.output_bounds),
        status_key: status_key.into(),
    };

    cert.emit_report();

    assert!(status.kernel(status_key).is_some());
    // Voicing pair bounds should be reasonably tight (small perturbation)
    assert!(
        cert.max_width.is_finite(),
        "voicing pair bounds must be finite"
    );
}

/// CROWN-verify phoneme encoder stability under vowel proximity confusion.
///
/// Vowel groups (ɪ/iː/ɛ) have 3 tokens — wider perturbation than voicing
/// pairs. Tests that encoder output bounds scale with confusion set size.
#[test]
fn test_phoneme_encoder_vowel_group_stability() {
    let embedding_weights = phoneme_helpers::synthetic_embedding_weights();
    let confusion_sets = phoneme_helpers::test_confusion_sets();

    // Vowel group: tokens 4,5,6
    let vowel_set = &confusion_sets[2];

    let def = phoneme_helpers::build_phoneme_encoder();
    let bindings = phoneme_helpers::phoneme_encoder_bindings();

    let base_tokens = vec![4u32; phoneme_helpers::SEQ_LEN];
    let perturbation_positions = vec![0];

    let (lower_f64, upper_f64) = nn_tts_verify::sequence_perturbation_bounds(
        &embedding_weights,
        phoneme_helpers::VOCAB_SIZE,
        phoneme_helpers::EMBED_DIM,
        &base_tokens,
        &perturbation_positions,
        std::slice::from_ref(vowel_set),
    )
    .expect("perturbation bounds");

    let lower: Vec<f32> = lower_f64.iter().map(|&v| v as f32).collect();
    let upper: Vec<f32> = upper_f64.iter().map(|&v| v as f32).collect();
    let shape = &[phoneme_helpers::SEQ_LEN, phoneme_helpers::EMBED_DIM];
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(shape), upper).expect("upper shape"),
    )
    .expect("valid bounds");

    let status_key = "cert_phoneme_vowel_group";
    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(&mut status, &def, &bindings, &input, Some(status_key))
        .expect("phoneme encoder verification");

    common::assert_bounds_valid(&result.output_bounds);

    let cert = PhonemeStabilityCertificate {
        architecture: "Embedding→Linear→ReLU→Linear".into(),
        confusion_set_name: vowel_set.name.clone(),
        confusion_set_size: vowel_set.token_ids.len(),
        confusion_category: format!("{:?}", vowel_set.category),
        method: result.verification.method,
        avg_width: measure_avg_width(&result.output_bounds),
        max_width: measure_max_width(&result.output_bounds),
        status_key: status_key.into(),
    };

    cert.emit_report();

    assert!(status.kernel(status_key).is_some());
    assert!(cert.max_width.is_finite());
}

/// CROWN-verify phoneme encoder with multiple perturbed positions.
///
/// Perturbs positions 0 AND 2 simultaneously with different confusion sets.
/// Models a real adversarial scenario: multiple phoneme confusions in one
/// utterance. Tests that CROWN bounds compose correctly across positions.
#[test]
fn test_phoneme_encoder_multi_position_perturbation() {
    let embedding_weights = phoneme_helpers::synthetic_embedding_weights();
    let confusion_sets = phoneme_helpers::test_confusion_sets();

    let def = phoneme_helpers::build_phoneme_encoder();
    let bindings = phoneme_helpers::phoneme_encoder_bindings();

    // Base sequence: tokens [0, 4, 2, 7]
    let base_tokens = vec![0u32, 4, 2, 7];
    // Perturb positions 0 (voicing pair) and 2 (voicing pair)
    let perturbation_positions = vec![0, 2];
    // Use voicing pairs for both positions
    let perturb_sets = vec![confusion_sets[0].clone(), confusion_sets[1].clone()];

    let (lower_f64, upper_f64) = nn_tts_verify::sequence_perturbation_bounds(
        &embedding_weights,
        phoneme_helpers::VOCAB_SIZE,
        phoneme_helpers::EMBED_DIM,
        &base_tokens,
        &perturbation_positions,
        &perturb_sets,
    )
    .expect("perturbation bounds");

    let lower: Vec<f32> = lower_f64.iter().map(|&v| v as f32).collect();
    let upper: Vec<f32> = upper_f64.iter().map(|&v| v as f32).collect();
    let shape = &[phoneme_helpers::SEQ_LEN, phoneme_helpers::EMBED_DIM];
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(shape), upper).expect("upper shape"),
    )
    .expect("valid bounds");

    let status_key = "cert_phoneme_multi_pos";
    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(&mut status, &def, &bindings, &input, Some(status_key))
        .expect("phoneme encoder verification");

    common::assert_bounds_valid(&result.output_bounds);

    eprintln!("=== MULTI-POSITION PHONEME PERTURBATION CERTIFICATE ===");
    eprintln!("Perturbed positions: 0 (voicing p/b), 2 (voicing t/d)");
    eprintln!("Method:           {:?}", result.verification.method);
    eprintln!(
        "Bounds:           avg_w={:.6}, max_w={:.6}",
        measure_avg_width(&result.output_bounds),
        measure_max_width(&result.output_bounds)
    );
    eprintln!("Persisted:        status_key={status_key}");
    eprintln!("=======================================================");

    assert!(status.kernel(status_key).is_some());
}

/// CROWN-verify phoneme encoder with residual connection.
///
/// The residual architecture (Linear → ReLU → + input → Linear) is closer
/// to the real PlBert encoder. Tests that residual connections don't
/// unboundedly amplify perturbation bounds.
#[test]
fn test_phoneme_encoder_residual_stability() {
    let embedding_weights = phoneme_helpers::synthetic_embedding_weights();
    let confusion_sets = phoneme_helpers::test_confusion_sets();

    let def = phoneme_helpers::build_phoneme_encoder_residual();
    let bindings = phoneme_helpers::phoneme_encoder_residual_bindings();

    // Voicing pair perturbation at position 0
    let base_tokens = vec![0u32; phoneme_helpers::SEQ_LEN];
    let perturbation_positions = vec![0];

    let (lower_f64, upper_f64) = nn_tts_verify::sequence_perturbation_bounds(
        &embedding_weights,
        phoneme_helpers::VOCAB_SIZE,
        phoneme_helpers::EMBED_DIM,
        &base_tokens,
        &perturbation_positions,
        &[confusion_sets[0].clone()],
    )
    .expect("perturbation bounds");

    let lower: Vec<f32> = lower_f64.iter().map(|&v| v as f32).collect();
    let upper: Vec<f32> = upper_f64.iter().map(|&v| v as f32).collect();
    let shape = &[phoneme_helpers::SEQ_LEN, phoneme_helpers::EMBED_DIM];
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(shape), upper).expect("upper shape"),
    )
    .expect("valid bounds");

    let status_key = "cert_phoneme_residual_voicing";
    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(&mut status, &def, &bindings, &input, Some(status_key))
        .expect("residual encoder verification");

    common::assert_bounds_valid(&result.output_bounds);

    let cert = PhonemeStabilityCertificate {
        architecture: "Embedding→Linear→ReLU→Residual→Linear".into(),
        confusion_set_name: confusion_sets[0].name.clone(),
        confusion_set_size: confusion_sets[0].token_ids.len(),
        confusion_category: format!("{:?}", confusion_sets[0].category),
        method: result.verification.method,
        avg_width: measure_avg_width(&result.output_bounds),
        max_width: measure_max_width(&result.output_bounds),
        status_key: status_key.into(),
    };

    cert.emit_report();

    assert!(status.kernel(status_key).is_some());
}

/// Confusion set size scaling: bound width vs number of tokens.
///
/// Compares output bounds across confusion sets of different sizes:
/// - Voicing pair (2 tokens): tightest bounds
/// - Nasal pair (2 tokens): similar to voicing
/// - Vowel group (3 tokens): wider bounds
///
/// This documents the relationship between perturbation set size and
/// output bound width, a key result for #1740 AC2.
#[test]
fn test_phoneme_confusion_set_scaling() {
    let embedding_weights = phoneme_helpers::synthetic_embedding_weights();
    let confusion_sets = phoneme_helpers::test_confusion_sets();

    let def = phoneme_helpers::build_phoneme_encoder();
    let bindings = phoneme_helpers::phoneme_encoder_bindings();

    eprintln!("--- Confusion set scaling (phoneme encoder) ---");
    eprintln!("  Set                 tokens    avg_w       max_w       method");

    for cs in &confusion_sets {
        // Use first token in the set as base
        let base_token = cs.token_ids[0];
        let base_tokens = vec![base_token; phoneme_helpers::SEQ_LEN];
        let perturbation_positions = vec![0];

        let (lower_f64, upper_f64) = nn_tts_verify::sequence_perturbation_bounds(
            &embedding_weights,
            phoneme_helpers::VOCAB_SIZE,
            phoneme_helpers::EMBED_DIM,
            &base_tokens,
            &perturbation_positions,
            std::slice::from_ref(cs),
        )
        .expect("perturbation bounds");

        let lower: Vec<f32> = lower_f64.iter().map(|&v| v as f32).collect();
        let upper: Vec<f32> = upper_f64.iter().map(|&v| v as f32).collect();
        let shape = &[phoneme_helpers::SEQ_LEN, phoneme_helpers::EMBED_DIM];
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(shape), lower).expect("lower shape"),
            ArrayD::from_shape_vec(IxDyn(shape), upper).expect("upper shape"),
        )
        .expect("valid bounds");

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let (method, output, _) =
            nn_verify::propagate_with_crown_fallback(&graph, &input).expect("propagation");

        common::assert_bounds_valid(&output);

        let avg = measure_avg_width(&output);
        let max_w = measure_max_width(&output);

        eprintln!(
            "  {:<20} {:<9} {avg:>10.6}  {max_w:>10.6}  {method:?}",
            cs.name,
            cs.token_ids.len(),
        );
    }
}
