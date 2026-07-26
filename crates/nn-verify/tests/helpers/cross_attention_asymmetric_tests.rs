// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Asymmetric cross-attention monotonicity (Q_SEQ != KV_SEQ).
//!
//! Phase 20 of #1729: extends Phase 19's square attention proofs to the
//! asymmetric case where the decoder has more steps than the encoder has
//! positions. This is the realistic TTS scenario: a codec produces T_dec
//! audio token steps attending to T_enc text embeddings where T_dec > T_enc.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 20.

#![allow(dead_code)]

use super::asym;
use super::common;
use nn_verify::tensor_kernel_to_graph;

// ---------------------------------------------------------------------------
// Constants — asymmetric TTS parameters
// ---------------------------------------------------------------------------

/// Decoder sequence length (codec token steps) — longer than encoder.
const T_DEC: usize = 10;

/// Encoder sequence length (text/phoneme positions) — shorter.
const T_ENC: usize = 6;

/// Model dimension (embedding size).
const D_MODEL: usize = 8;

/// Number of attention heads.
const NUM_HEADS: usize = 2;

/// Weight scale for projections.
const W_SCALE: f32 = 0.05;

// ---------------------------------------------------------------------------
// Test 1: Simple asymmetric graph builds correctly
// ---------------------------------------------------------------------------

#[test]
fn test_asymmetric_simple_graph_builds() {
    let def = asym::build_asymmetric_scores_simple(T_DEC, T_ENC, D_MODEL);
    let bindings = asym::simple_bindings(T_ENC, D_MODEL);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    assert_eq!(
        graph.num_nodes(),
        1,
        "simple asymmetric graph should be 1 fused node, got {}",
        graph.num_nodes()
    );
}

// ---------------------------------------------------------------------------
// Test 2: Simple asymmetric IBP produces [T_dec, T_enc] output
// ---------------------------------------------------------------------------

#[test]
fn test_asymmetric_simple_ibp_output_shape() {
    let def = asym::build_asymmetric_scores_simple(T_DEC, T_ENC, D_MODEL);
    let bindings = asym::simple_bindings(T_ENC, D_MODEL);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 1.0);
    let output = asym::graph_propagate(&def, &bindings, &input);
    let (lo, _hi) = output.lower_upper();

    assert_eq!(
        lo.shape(),
        &[T_DEC, T_ENC],
        "output shape must be [T_dec={T_DEC}, T_enc={T_ENC}]"
    );
    common::assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Test 3: Simple asymmetric certificate — correct dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_asymmetric_simple_certificate_dimensions() {
    let def = asym::build_asymmetric_scores_simple(T_DEC, T_ENC, D_MODEL);
    let bindings = asym::simple_bindings(T_ENC, D_MODEL);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 1.0);
    let output = asym::graph_propagate(&def, &bindings, &input);

    let cert = asym::extract_certificate(&output, T_DEC, T_ENC, 1.0, "IBP");

    assert_eq!(cert.decoder_steps, T_DEC);
    assert_eq!(cert.encoder_positions, T_ENC);
    assert_eq!(cert.row_margins.len(), T_ENC);
    for (row, margin) in cert.row_margins.iter().enumerate() {
        assert!(margin.is_finite(), "row {row} margin must be finite");
    }
}

// ---------------------------------------------------------------------------
// Test 4: Projected multi-head asymmetric — graph builds and propagates
// ---------------------------------------------------------------------------

#[test]
fn test_asymmetric_projected_ibp() {
    let def = asym::build_asymmetric_scores_projected(T_DEC, T_ENC, D_MODEL, NUM_HEADS);
    let bindings = asym::projected_bindings(T_ENC, D_MODEL, W_SCALE);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, _hi) = output.lower_upper();

    assert_eq!(
        lo.shape(),
        &[NUM_HEADS, T_DEC, T_ENC],
        "projected output shape must be [H={NUM_HEADS}, T_dec={T_DEC}, T_enc={T_ENC}]"
    );
    common::assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Test 5: Projected multi-head — per-head certificates
// ---------------------------------------------------------------------------

#[test]
fn test_asymmetric_projected_per_head_certificate() {
    let def = asym::build_asymmetric_scores_projected(T_DEC, T_ENC, D_MODEL, NUM_HEADS);
    let bindings = asym::projected_bindings(T_ENC, D_MODEL, W_SCALE);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 1.0);
    let output = asym::graph_propagate(&def, &bindings, &input);
    let (lo, hi) = output.lower_upper();

    for head in 0..NUM_HEADS {
        let score_lower: Vec<f32> = (0..T_DEC * T_ENC)
            .map(|i| lo[[head, i / T_ENC, i % T_ENC]])
            .collect();
        let score_upper: Vec<f32> = (0..T_DEC * T_ENC)
            .map(|i| hi[[head, i / T_ENC, i % T_ENC]])
            .collect();

        let cert = nn_tts_verify::monotonicity::interpret_attention_monotonicity(
            &score_lower,
            &score_upper,
            T_DEC,
            T_ENC,
            1.0,
            "IBP",
        )
        .expect("valid cert");

        eprintln!(
            "Head {head}: min_margin={:.4}, proven={}, dec={}, enc={}",
            cert.min_margin, cert.is_proven, cert.decoder_steps, cert.encoder_positions
        );

        assert_eq!(cert.decoder_steps, T_DEC);
        assert_eq!(cert.encoder_positions, T_ENC);
        assert_eq!(cert.row_margins.len(), T_ENC);
    }
}

// ---------------------------------------------------------------------------
// Test 6: CROWN propagation on asymmetric simple scores
// ---------------------------------------------------------------------------

#[test]
fn test_asymmetric_simple_crown() {
    let def = asym::build_asymmetric_scores_simple(T_DEC, T_ENC, D_MODEL);
    let bindings = asym::simple_bindings(T_ENC, D_MODEL);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 1.0);

    let (method, output, fallback) = common::assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("Asymmetric simple CROWN: method={method:?}, fallback={fallback:?}");
    common::assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[T_DEC, T_ENC]);
}

// ---------------------------------------------------------------------------
// Test 7: PE-aware asymmetric — proven monotonicity with tight bounds
// ---------------------------------------------------------------------------

#[test]
fn test_asymmetric_pe_aware_proven_monotonicity() {
    let pe_scale = 3.0;
    let def = asym::build_asymmetric_scores_pe_aware(T_DEC, T_ENC, D_MODEL);
    let bindings = asym::pe_aware_bindings(T_DEC, T_ENC, D_MODEL, pe_scale);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 0.01); // tight

    let output = asym::graph_propagate(&def, &bindings, &input);
    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[T_DEC, T_ENC]);

    let cert = asym::extract_certificate(&output, T_DEC, T_ENC, 0.01, "IBP");

    eprintln!(
        "PE-aware asymmetric: min_margin={:.6}, proven={}, margins={:?}",
        cert.min_margin, cert.is_proven, cert.row_margins
    );

    assert_eq!(cert.decoder_steps, T_DEC);
    assert_eq!(cert.encoder_positions, T_ENC);
    assert_eq!(cert.row_margins.len(), T_ENC);

    assert!(
        cert.is_proven,
        "diagonal dominance should be provable with PE-aware asymmetric Q: \
         min_margin={}, margins={:?}",
        cert.min_margin, cert.row_margins
    );
    assert!(cert.min_margin > 0.0);
}

// ---------------------------------------------------------------------------
// Test 8: Multiple aspect ratios — verify certificate correctness
// ---------------------------------------------------------------------------

#[test]
fn test_asymmetric_multiple_aspect_ratios() {
    let d = 8;
    let pe_scale = 3.0;
    let input_bound = 0.01f32;

    let ratios: &[(usize, usize)] = &[
        (4, 4),  // 1:1 (square, baseline)
        (6, 4),  // 1.5:1
        (8, 4),  // 2:1 (typical for short utterances)
        (12, 4), // 3:1 (typical for medium utterances)
    ];

    for &(t_dec, t_enc) in ratios {
        let def = asym::build_asymmetric_scores_pe_aware(t_dec, t_enc, d);
        let bindings = asym::pe_aware_bindings(t_dec, t_enc, d, pe_scale);
        let input = common::uniform_bounds(&[t_dec, d], input_bound);
        let output = asym::graph_propagate(&def, &bindings, &input);

        let cert = asym::extract_certificate(&output, t_dec, t_enc, f64::from(input_bound), "IBP");

        eprintln!(
            "Ratio {t_dec}:{t_enc}: min_margin={:.6}, proven={}, diag_rows={}",
            cert.min_margin,
            cert.is_proven,
            cert.row_margins.len()
        );

        assert_eq!(cert.decoder_steps, t_dec);
        assert_eq!(cert.encoder_positions, t_enc);
        assert_eq!(cert.row_margins.len(), t_enc.min(t_dec));

        assert!(
            cert.is_proven,
            "ratio {t_dec}:{t_enc} should be provable: min_margin={}",
            cert.min_margin
        );
    }
}

// ---------------------------------------------------------------------------
// Test 9: Extreme aspect ratio (T_dec >> T_enc) — stress test
// ---------------------------------------------------------------------------

#[test]
fn test_asymmetric_extreme_ratio() {
    let t_dec = 20;
    let t_enc = 4;
    let d = 8;
    let pe_scale = 4.0;
    let input_bound = 0.01f32;

    let def = asym::build_asymmetric_scores_pe_aware(t_dec, t_enc, d);
    let bindings = asym::pe_aware_bindings(t_dec, t_enc, d, pe_scale);
    let input = common::uniform_bounds(&[t_dec, d], input_bound);
    let output = asym::graph_propagate(&def, &bindings, &input);
    let (lo, _) = output.lower_upper();

    assert_eq!(lo.shape(), &[t_dec, t_enc]);

    let cert = asym::extract_certificate(&output, t_dec, t_enc, f64::from(input_bound), "IBP");

    eprintln!(
        "Extreme {t_dec}:{t_enc}: min_margin={:.6}, proven={}, diag_rows={}",
        cert.min_margin,
        cert.is_proven,
        cert.row_margins.len()
    );

    assert_eq!(cert.row_margins.len(), t_enc);
    assert!(
        cert.is_proven,
        "extreme ratio should be provable with pe_scale={pe_scale}: \
         min_margin={}",
        cert.min_margin
    );
}
