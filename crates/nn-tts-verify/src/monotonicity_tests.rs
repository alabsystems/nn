// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

// -- Duration positivity tests -------------------------------------------

#[test]
fn test_proven_certificate() {
    let cert = interpret_duration_positivity(-3.5, -10.0, 1.0, 1.0, 1, "CROWN");
    assert!(cert.is_proven);
    assert_eq!(cert.lower_bound, -3.5);
    assert_eq!(cert.threshold, -10.0);
    assert_eq!(cert.sequence_length, 1);
    assert_eq!(cert.propagation_mode, "CROWN");
}

#[test]
fn test_not_proven_certificate() {
    let cert = interpret_duration_positivity(-15.0, -10.0, 1.0, 1.0, 1, "IBP");
    assert!(!cert.is_proven);
    assert_eq!(cert.lower_bound, -15.0);
}

#[test]
fn test_boundary_case_not_proven() {
    let cert = interpret_duration_positivity(-10.0, -10.0, 1.0, 1.0, 1, "CROWN");
    assert!(!cert.is_proven);
}

#[test]
fn test_positive_lower_bound() {
    let cert = interpret_duration_positivity(0.5, -10.0, 1.0, 1.0, 1, "CROWN");
    assert!(cert.is_proven);
    assert!(cert.lower_bound > 0.0);
}

#[test]
fn test_duration_propagation_mode_preserves_provenance_and_normalizes_crown_family() {
    let cert = interpret_duration_positivity(-3.5, -10.0, 1.0, 1.0, 1, "alpha-CROWN");
    assert_eq!(cert.propagation_mode, "alpha-CROWN");
    assert!(cert.is_sound_crown_family());
}

#[test]
fn test_sound_crown_family_classifier_accepts_alpha_beta_variants_only() {
    for mode in [
        "CROWN",
        "crown",
        "AlphaCrown",
        "alpha-CROWN",
        "beta_crown",
        "BetaCrown",
    ] {
        assert!(
            propagation_mode_is_sound_crown_family(mode),
            "expected {mode} to classify as sound CROWN-family"
        );
    }

    for mode in ["IBP", "mixed_IBP_CROWN", "Analytical", "unknown"] {
        assert!(
            !propagation_mode_is_sound_crown_family(mode),
            "expected {mode} to stay outside the sound CROWN-family"
        );
    }
}

// -- Attention monotonicity tests ----------------------------------------

#[test]
fn test_attention_diagonal_dominant_proven() {
    // 3×3 score matrix where diagonal lower > off-diagonal upper.
    // Diagonal lower bounds: [5.0, 6.0, 7.0]
    // Off-diagonal upper bounds: all 2.0
    #[rustfmt::skip]
    let lower = [5.0f32, 1.0, 1.0,
                  1.0, 6.0, 1.0,
                  1.0, 1.0, 7.0];
    #[rustfmt::skip]
    let upper = [6.0f32, 2.0, 2.0,
                  2.0, 7.0, 2.0,
                  2.0, 2.0, 8.0];
    let cert = interpret_attention_monotonicity(&lower, &upper, 3, 3, 1.0, "CROWN").unwrap();
    assert!(cert.is_proven);
    assert!(cert.min_margin > 0.0);
    // Row 0: lower(S[0,0])=5.0, max_offdiag_upper=2.0 → margin=3.0
    assert!((cert.row_margins[0] - 3.0).abs() < 1e-10);
    // Row 1: lower(S[1,1])=6.0, max_offdiag_upper=2.0 → margin=4.0
    assert!((cert.row_margins[1] - 4.0).abs() < 1e-10);
    // Row 2: lower(S[2,2])=7.0, max_offdiag_upper=2.0 → margin=5.0
    assert!((cert.row_margins[2] - 5.0).abs() < 1e-10);
    assert!((cert.min_margin - 3.0).abs() < 1e-10);
}

#[test]
fn test_attention_not_diagonal_dominant() {
    // Off-diagonal upper exceeds diagonal lower in row 1.
    #[rustfmt::skip]
    let lower = [5.0f32, 1.0, 1.0,
                  1.0, 2.0, 1.0,  // diag lower=2.0
                  1.0, 1.0, 7.0];
    #[rustfmt::skip]
    let upper = [6.0f32, 2.0, 2.0,
                  4.0, 3.0, 4.0,  // off-diag upper=4.0 > diag lower=2.0
                  2.0, 2.0, 8.0];
    let cert = interpret_attention_monotonicity(&lower, &upper, 3, 3, 1.0, "IBP").unwrap();
    assert!(!cert.is_proven);
    // Row 1: lower(S[1,1])=2.0, max_offdiag_upper=4.0 → margin=-2.0
    assert!((cert.row_margins[1] - (-2.0)).abs() < 1e-10);
    assert!(cert.min_margin < 0.0);
}

#[test]
fn test_attention_single_position_trivial() {
    // 1×1 attention: trivially monotonic (no off-diagonal elements).
    let lower = [1.0f32];
    let upper = [2.0f32];
    let cert = interpret_attention_monotonicity(&lower, &upper, 1, 1, 1.0, "CROWN").unwrap();
    assert!(cert.is_proven);
    assert!(cert.min_margin.is_infinite());
}

#[test]
fn test_attention_2x2_boundary() {
    // Exactly at boundary: diagonal lower == off-diagonal upper → not proven.
    let lower = [3.0f32, 1.0, 1.0, 3.0];
    let upper = [4.0f32, 3.0, 3.0, 4.0]; // off-diag upper=3.0 == diag lower=3.0
    let cert = interpret_attention_monotonicity(&lower, &upper, 2, 2, 1.0, "CROWN").unwrap();
    assert!(!cert.is_proven);
    assert!((cert.min_margin - 0.0).abs() < 1e-10);
}

#[test]
fn test_attention_rectangular() {
    // 2×3 attention matrix (more encoder positions than decoder steps).
    // Only check rows 0 and 1 (diagonal exists for min(2,3)=2 rows).
    #[rustfmt::skip]
    let lower = [5.0f32, 1.0, 1.0,
                  1.0, 6.0, 1.0];
    #[rustfmt::skip]
    let upper = [6.0f32, 2.0, 2.0,
                  2.0, 7.0, 2.0];
    let cert = interpret_attention_monotonicity(&lower, &upper, 2, 3, 1.0, "CROWN").unwrap();
    assert!(cert.is_proven);
    assert_eq!(cert.row_margins.len(), 2);
    assert!((cert.row_margins[0] - 3.0).abs() < 1e-10);
    assert!((cert.row_margins[1] - 4.0).abs() < 1e-10);
}

#[test]
fn test_attention_propagation_mode_preserves_beta_crown_provenance() {
    #[rustfmt::skip]
    let lower = [5.0f32, 1.0, 1.0,
                  1.0, 6.0, 1.0,
                  1.0, 1.0, 7.0];
    #[rustfmt::skip]
    let upper = [6.0f32, 2.0, 2.0,
                  2.0, 7.0, 2.0,
                  2.0, 2.0, 8.0];
    let cert = interpret_attention_monotonicity(&lower, &upper, 3, 3, 1.0, "BetaCrown").unwrap();
    assert_eq!(cert.propagation_mode, "BetaCrown");
    assert!(cert.is_sound_crown_family());
}

// -- Multi-head weight margin aggregation tests ---------------------------

#[test]
fn test_multi_head_all_proven() {
    // 2 heads, 3 decoder steps, all margins positive.
    let head0 = vec![0.5, 0.3, 0.7];
    let head1 = vec![0.2, 0.6, 0.4];
    let cert = from_multi_head_weight_margins(&[head0, head1], 3, 3, 1.0, "IBP").unwrap();
    assert!(cert.is_proven);
    // Per-step minimum: t0=min(0.5,0.2)=0.2, t1=min(0.3,0.6)=0.3, t2=min(0.7,0.4)=0.4
    assert!((cert.row_margins[0] - 0.2).abs() < 1e-10);
    assert!((cert.row_margins[1] - 0.3).abs() < 1e-10);
    assert!((cert.row_margins[2] - 0.4).abs() < 1e-10);
    assert!((cert.min_margin - 0.2).abs() < 1e-10);
}

#[test]
fn test_multi_head_one_head_negative() {
    // Head 1 has a negative margin at step 1 → not proven.
    let head0 = vec![0.5, 0.3];
    let head1 = vec![0.2, -0.1];
    let cert = from_multi_head_weight_margins(&[head0, head1], 2, 2, 1.0, "CROWN").unwrap();
    assert!(!cert.is_proven);
    assert!((cert.row_margins[1] - (-0.1)).abs() < 1e-10);
    assert!(cert.min_margin < 0.0);
}

#[test]
fn test_multi_head_single_head_passthrough() {
    // Single head: should be identical to the head's margins.
    let head0 = vec![0.8, 0.4, 0.6];
    let cert = from_multi_head_weight_margins(&[head0], 3, 3, 0.5, "IBP").unwrap();
    assert!(cert.is_proven);
    assert!((cert.min_margin - 0.4).abs() < 1e-10);
    assert_eq!(cert.row_margins.len(), 3);
}

// -- Weight magnitude validation tests (Phase 30) --------------------------

#[test]
fn test_weight_magnitude_all_within_bound() {
    // 3 layers with weights well below the magnitude bound.
    let w0 = vec![0.001f32, -0.002, 0.003, -0.001];
    let w1 = vec![0.002f32, 0.001, -0.003, 0.002];
    let w2 = vec![-0.001f32, 0.002, 0.001, -0.002];
    let weights: Vec<&[f32]> = vec![&w0, &w1, &w2];
    let names = vec!["q_proj", "k_proj", "v_proj"];
    let fan_ins = vec![192, 192, 192];

    let cert = validate_weight_magnitudes(&weights, &names, &fan_ins, 192, 0.005).unwrap();
    assert!(cert.all_within_bound);
    assert_eq!(cert.violating_layers, 0);
    assert!((cert.per_layer_max_abs[0] - 0.003).abs() < 1e-10);
    assert!((cert.per_layer_max_abs[1] - 0.003).abs() < 1e-10);
    assert!((cert.per_layer_max_abs[2] - 0.002).abs() < 1e-10);
}

#[test]
fn test_weight_magnitude_one_violating_layer() {
    // Layer 1 has a weight exceeding the magnitude bound.
    let w0 = vec![0.001f32, -0.002];
    let w1 = vec![0.010f32, 0.001]; // max_abs = 0.01 > bound 0.005
    let weights: Vec<&[f32]> = vec![&w0, &w1];
    let names = vec!["q_proj", "k_proj"];
    let fan_ins = vec![64, 64];

    let cert = validate_weight_magnitudes(&weights, &names, &fan_ins, 64, 0.005).unwrap();
    assert!(!cert.all_within_bound);
    assert_eq!(cert.violating_layers, 1);
    assert!((cert.per_layer_max_abs[1] - 0.01).abs() < 1e-6);
}

#[test]
fn test_weight_magnitude_xavier_normalized() {
    // With fan_in=100, weight mag=0.1, normalized = 0.1 * sqrt(100) = 1.0.
    let w = vec![0.1f32, -0.05, 0.08];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["linear"];
    let fan_ins = vec![100];

    let cert = validate_weight_magnitudes(&weights, &names, &fan_ins, 100, 1.0).unwrap();
    // max_abs = f64::from(0.1f32) ≈ 0.10000000149, normalized ≈ 1.0000000149
    // Use f32-appropriate tolerance (1e-6) since weight data originates as f32.
    assert!((cert.max_normalized_magnitude - 1.0).abs() < 1e-6);
}

#[test]
fn test_weight_magnitude_dimension_mismatch_names() {
    let w = vec![0.1f32];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["a", "b"]; // 2 names for 1 weight
    let fan_ins = vec![10];

    let err = validate_weight_magnitudes(&weights, &names, &fan_ins, 10, 0.1);
    assert!(err.is_err());
}

#[test]
fn test_weight_magnitude_dimension_mismatch_fan_ins() {
    let w = vec![0.1f32];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["a"];
    let fan_ins = vec![10, 20]; // 2 fan_ins for 1 weight

    let err = validate_weight_magnitudes(&weights, &names, &fan_ins, 10, 0.1);
    assert!(err.is_err());
}

#[test]
fn test_weight_magnitude_empty_weights() {
    let weights: Vec<&[f32]> = vec![];
    let names: Vec<&str> = vec![];
    let fan_ins: Vec<usize> = vec![];

    let cert = validate_weight_magnitudes(&weights, &names, &fan_ins, 192, 0.005).unwrap();
    assert!(cert.all_within_bound);
    assert_eq!(cert.violating_layers, 0);
    assert_eq!(cert.per_layer_max_abs.len(), 0);
    // No layers → max_normalized_magnitude stays at initial 0.0.
    assert!((cert.max_normalized_magnitude - 0.0).abs() < 1e-10);
}

// -- max_provable_input_bound tests ----------------------------------------

#[test]
fn test_max_provable_input_bound_basic() {
    // D=192, max_mag=0.003, pe_margin=1.0
    // max_ib = 1.0 / (192 * 0.003) = 1.0 / 0.576 ≈ 1.7361
    let w = vec![0.003f32, -0.001, 0.002];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["q_proj"];
    let fan_ins = vec![192];

    let cert = validate_weight_magnitudes(&weights, &names, &fan_ins, 192, 0.005).unwrap();
    let max_ib = max_provable_input_bound(&cert, 1.0);
    let expected = 1.0 / (192.0 * 0.003);
    assert!((max_ib - expected).abs() < 1e-6);
}

#[test]
fn test_max_provable_input_bound_zero_weights() {
    // All-zero weights → provable at any input bound → INFINITY.
    let w = vec![0.0f32, 0.0, 0.0];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["q_proj"];
    let fan_ins = vec![192];

    let cert = validate_weight_magnitudes(&weights, &names, &fan_ins, 192, 0.005).unwrap();
    let max_ib = max_provable_input_bound(&cert, 1.0);
    assert!(max_ib.is_infinite());
}

#[test]
fn test_max_provable_input_bound_large_weights() {
    // D=256, max_mag=0.1, pe_margin=1.0 → max_ib = 1.0 / (256 * 0.1) = 0.0390625
    let w = vec![0.1f32, -0.05, 0.08];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["q_proj"];
    let fan_ins = vec![256];

    let cert = validate_weight_magnitudes(&weights, &names, &fan_ins, 256, 0.2).unwrap();
    let max_ib = max_provable_input_bound(&cert, 1.0);
    let expected = 1.0 / (256.0 * 0.1);
    assert!((max_ib - expected).abs() < 1e-6);
}

#[test]
fn test_max_provable_input_bound_uses_worst_layer() {
    // Two layers: max_mag across layers should use the larger one.
    let w0 = vec![0.001f32, -0.002];
    let w1 = vec![0.005f32, 0.001]; // This layer has the larger max_abs.
    let weights: Vec<&[f32]> = vec![&w0, &w1];
    let names = vec!["q_proj", "k_proj"];
    let fan_ins = vec![128, 128];

    let cert = validate_weight_magnitudes(&weights, &names, &fan_ins, 128, 0.01).unwrap();
    let max_ib = max_provable_input_bound(&cert, 1.0);
    // Should use max_mag = 0.005 (from w1), not 0.002 (from w0).
    let expected = 1.0 / (128.0 * 0.005);
    assert!((max_ib - expected).abs() < 1e-6);
}

// -- NaN/Inf defense-in-depth tests (P1-256 algorithm audit) ---------------

#[test]
fn test_weight_magnitude_nan_weight_rejected() {
    // NaN in weight data must be caught, not silently pass validation.
    // f64::max swallows NaN (IEEE 754-2008 maxNum: max(x, NaN) = x), so fold
    // would silently skip NaN elements. The guard checks elements directly.
    let w = vec![0.001f32, f32::NAN, 0.002];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["q_proj"];
    let fan_ins = vec![192];

    let result = validate_weight_magnitudes(&weights, &names, &fan_ins, 192, 0.005);
    assert!(result.is_err(), "NaN weights must be rejected");
}

#[test]
fn test_weight_magnitude_inf_weight_rejected() {
    // Inf weight produces Inf max_abs, which must be caught.
    let w = vec![0.001f32, f32::INFINITY, 0.002];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["q_proj"];
    let fan_ins = vec![192];

    let result = validate_weight_magnitudes(&weights, &names, &fan_ins, 192, 0.005);
    assert!(result.is_err(), "Inf weights must be rejected");
}

#[test]
fn test_weight_magnitude_neg_inf_weight_rejected() {
    let w = vec![f32::NEG_INFINITY, 0.001, 0.002];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["q_proj"];
    let fan_ins = vec![192];

    let result = validate_weight_magnitudes(&weights, &names, &fan_ins, 192, 0.005);
    assert!(result.is_err(), "Neg-Inf weights must be rejected");
}

#[test]
fn test_max_provable_input_bound_nan_pe_margin() {
    // NaN pe_margin must not produce NaN result — return 0.0 (not provable).
    let w = vec![0.003f32, -0.001];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["q_proj"];
    let fan_ins = vec![192];

    let cert = validate_weight_magnitudes(&weights, &names, &fan_ins, 192, 0.005).unwrap();
    let max_ib = max_provable_input_bound(&cert, f64::NAN);
    assert_eq!(max_ib, 0.0, "NaN pe_margin must return 0.0");
}

#[test]
fn test_max_provable_input_bound_neg_pe_margin() {
    let w = vec![0.003f32, -0.001];
    let weights: Vec<&[f32]> = vec![&w];
    let names = vec!["q_proj"];
    let fan_ins = vec![192];

    let cert = validate_weight_magnitudes(&weights, &names, &fan_ins, 192, 0.005).unwrap();
    let max_ib = max_provable_input_bound(&cert, -1.0);
    assert_eq!(max_ib, 0.0, "negative pe_margin must return 0.0");
}

// -- NaN defense: attention fold patterns ------------------------------------

/// NaN in score_lower is now rejected at entry (defense-in-depth).
///
/// Previously, IEEE 754 minNum skipped NaN in `fold(INFINITY, f64::min)`,
/// producing false-positive `is_proven=true` despite corrupted data.
/// Now the input guard catches NaN before margin computation.
#[test]
fn test_attention_nan_diagonal_lower_rejected() {
    // 3×3 score matrix. Row 1 diagonal has NaN in lower bounds.
    #[rustfmt::skip]
    let lower = [5.0f32, 1.0, 1.0,
                  1.0, f32::NAN, 1.0,  // NaN at diagonal [1,1]
                  1.0, 1.0, 7.0];
    #[rustfmt::skip]
    let upper = [6.0f32, 2.0, 2.0,
                  2.0, 7.0, 2.0,
                  2.0, 2.0, 8.0];
    let result = interpret_attention_monotonicity(&lower, &upper, 3, 3, 1.0, "CROWN");
    assert!(result.is_err(), "NaN in score_lower should be rejected");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("must be finite"),
        "error should mention finiteness, got: {err_msg}"
    );
}

/// NaN in score_upper off-diagonal is now rejected at entry.
#[test]
fn test_attention_nan_offdiag_upper_rejected() {
    #[rustfmt::skip]
    let lower = [5.0f32, 1.0, 1.0,
                  1.0, 6.0, 1.0,
                  1.0, 1.0, 7.0];
    #[rustfmt::skip]
    let upper = [6.0f32, f32::NAN, 2.0,   // NaN at off-diag [0,1]
                  2.0, 7.0, 2.0,
                  2.0, 2.0, 8.0];
    let result = interpret_attention_monotonicity(&lower, &upper, 3, 3, 1.0, "CROWN");
    assert!(result.is_err(), "NaN in score_upper should be rejected");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("must be finite"),
        "error should mention finiteness, got: {err_msg}"
    );
}

/// Inf in score_lower is now rejected at entry (defense-in-depth).
///
/// While Inf diagonal lower is mathematically coherent (infinite dominance),
/// non-finite input data indicates upstream corruption and should not
/// produce a certificate. The caller should sanitize bounds before calling.
#[test]
fn test_attention_inf_diagonal_lower_rejected() {
    #[rustfmt::skip]
    let lower = [5.0f32, 1.0, 1.0,
                  1.0, f32::INFINITY, 1.0,  // Inf at diagonal [1,1]
                  1.0, 1.0, 7.0];
    #[rustfmt::skip]
    let upper = [6.0f32, 2.0, 2.0,
                  2.0, 7.0, 2.0,
                  2.0, 2.0, 8.0];
    let result = interpret_attention_monotonicity(&lower, &upper, 3, 3, 1.0, "CROWN");
    assert!(result.is_err(), "Inf in score_lower should be rejected");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("must be finite"),
        "error should mention finiteness, got: {err_msg}"
    );
}

/// Truncated head margin vectors must be rejected (#1994 regression test).
///
/// Previously, heads with margin vectors shorter than `diag_count` were
/// silently skipped, leaving `f64::INFINITY` as the minimum — a fail-open
/// soundness gap. Now `from_multi_head_weight_margins` validates upfront.
#[test]
fn test_multi_head_truncated_margins_rejected() {
    // Head 0: 3 margins (full). Head 1: 2 margins (truncated, missing step 2).
    let head0 = vec![0.5, 0.3, 0.7];
    let head1 = vec![0.2, 0.6]; // Only 2 entries, diag_count=3
    let result = from_multi_head_weight_margins(&[head0, head1], 3, 3, 1.0, "IBP");
    assert!(result.is_err(), "truncated head margins must be rejected");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("per_head_row_margins")
            && err_msg.contains("expected 3")
            && err_msg.contains("got 2"),
        "error should identify the dimension mismatch, got: {err_msg}",
    );
}

/// NaN in multi-head margins is now rejected at entry (defense-in-depth).
#[test]
fn test_multi_head_nan_margin_rejected() {
    // Head 0: all margins positive. Head 1: margin 0 has NaN.
    let head0 = vec![3.0, 4.0, 5.0];
    let head1 = vec![f64::NAN, 2.0, 3.0];
    let per_head = vec![head0, head1];
    let result = from_multi_head_weight_margins(&per_head, 3, 3, 1.0, "IBP");
    // NaN in per_head_row_margins[1] should produce an error.
    assert!(result.is_err(), "NaN margins should be rejected");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("must be finite"),
        "error should mention finiteness, got: {err_msg}",
    );
}
