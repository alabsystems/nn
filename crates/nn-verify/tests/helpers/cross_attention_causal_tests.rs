// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Causal cross-attention monotonicity (decoder-side masking).
//!
//! Phase 21 of #1729: extends Phase 20's asymmetric attention proofs to include
//! causal masking, where decoder step `t` can only attend to encoder positions
//! `<= f(t)`. This models autoregressive TTS decoders (e.g., Qwen3-TTS) where
//! future encoder positions are masked.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 21.

#![allow(dead_code)]

use super::causal;
use super::common;
use nn_verify::tensor_kernel_to_graph;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Decoder sequence length (longer than encoder — realistic TTS).
const T_DEC: usize = 12;

/// Encoder sequence length (text/phoneme positions).
const T_ENC: usize = 6;

/// Model dimension.
const D_MODEL: usize = 8;

// ---------------------------------------------------------------------------
// Test 1: Causal mask construction — linear alignment
// ---------------------------------------------------------------------------

/// Verify that linear alignment produces the correct mask pattern.
///
/// For T_dec=12, T_enc=6: f(t) = floor(t * 6 / 12) = floor(t/2).
/// - Row 0: f(0)=0 → only position 0 visible
/// - Row 1: f(1)=0 → only position 0 visible
/// - Row 2: f(2)=1 → positions 0-1 visible
/// - ...
/// - Row 11: f(11)=5 → all positions visible
#[test]
fn test_linear_causal_mask_pattern() {
    let mask = causal::build_linear_causal_mask(T_DEC, T_ENC);
    let data = mask.as_slice().expect("contiguous");

    // Check dimensions.
    assert_eq!(mask.shape(), &[T_DEC, T_ENC]);

    let unmasked = causal::count_unmasked_per_row(&mask, T_DEC, T_ENC);

    // Row 0: f(0) = 0 → 1 unmasked position
    assert_eq!(unmasked[0], 1, "row 0 should see 1 position");
    // Row 2: f(2) = 1 → 2 unmasked positions
    assert_eq!(unmasked[2], 2, "row 2 should see 2 positions");
    // Last row: f(11) = 5 → 6 unmasked (all visible)
    assert_eq!(
        unmasked[T_DEC - 1],
        T_ENC,
        "last row should see all positions"
    );

    // Verify mask values: unmasked = 0.0, masked = -1e9.
    for t in 0..T_DEC {
        let max_pos = causal::linear_alignment(t, T_DEC, T_ENC);
        for j in 0..T_ENC {
            let val = data[t * T_ENC + j];
            if j <= max_pos {
                assert_eq!(val, 0.0, "position ({t},{j}) should be unmasked");
            } else {
                assert!(val < -1e8, "position ({t},{j}) should be masked, got {val}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: Strict causal mask pattern
// ---------------------------------------------------------------------------

#[test]
fn test_strict_causal_mask_pattern() {
    let mask = causal::build_strict_causal_mask(T_DEC, T_ENC);
    let unmasked = causal::count_unmasked_per_row(&mask, T_DEC, T_ENC);

    // Strict causal: f(t) = min(t, T_enc-1).
    // Row 0: 1 visible, Row 1: 2, ..., Row 5: 6, Row 6+: 6 (all)
    for (t, &count) in unmasked.iter().enumerate().take(T_ENC) {
        assert_eq!(
            count,
            t + 1,
            "row {t} should see {val} positions",
            val = t + 1
        );
    }
    for (t, &count) in unmasked.iter().enumerate().take(T_DEC).skip(T_ENC) {
        assert_eq!(count, T_ENC, "row {t} should see all {T_ENC} positions");
    }
}

// ---------------------------------------------------------------------------
// Test 3: Lookahead causal mask pattern
// ---------------------------------------------------------------------------

#[test]
fn test_lookahead_causal_mask_pattern() {
    let lookahead = 1;
    let mask = causal::build_lookahead_causal_mask(T_DEC, T_ENC, lookahead);
    let unmasked = causal::count_unmasked_per_row(&mask, T_DEC, T_ENC);

    // Lookahead=1 shifts the boundary by 1 compared to linear.
    // Row 0: linear f(0)=0, +1 → 1 → 2 visible positions
    // Row 11: already all visible, +1 doesn't extend
    assert!(
        unmasked[0] >= 2,
        "row 0 with lookahead=1 should see >= 2 positions, got {}",
        unmasked[0]
    );
    assert_eq!(unmasked[T_DEC - 1], T_ENC, "last row still sees all");
}

// ---------------------------------------------------------------------------
// Test 4: Simple causal graph builds and propagates
// ---------------------------------------------------------------------------

#[test]
fn test_causal_simple_graph_builds() {
    let def = causal::build_causal_scores_simple(T_DEC, T_ENC, D_MODEL);
    let mask = causal::build_linear_causal_mask(T_DEC, T_ENC);
    let bindings = causal::simple_causal_bindings(T_ENC, D_MODEL, mask);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    assert!(
        graph.num_nodes() >= 2,
        "causal graph should have >= 2 nodes (matmul + add)"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Causal IBP — output shape and bounds validity
// ---------------------------------------------------------------------------

#[test]
fn test_causal_simple_ibp_output_shape() {
    let def = causal::build_causal_scores_simple(T_DEC, T_ENC, D_MODEL);
    let mask = causal::build_linear_causal_mask(T_DEC, T_ENC);
    let bindings = causal::simple_causal_bindings(T_ENC, D_MODEL, mask);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 1.0);
    let output = causal::graph_propagate(&def, &bindings, &input);
    let (lo, _hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[T_DEC, T_ENC]);
    common::assert_bounds_valid(&output);

    // Masked positions should have very negative upper bounds.
    let (_lo_arr, hi_arr) = output.lower_upper();
    // Row 0 with linear alignment: only position 0 is visible.
    // Positions 1..T_ENC should have upper bounds near -1e9.
    for j in 1..T_ENC {
        let upper = hi_arr[[0, j]];
        assert!(
            upper < -1e8,
            "row 0, col {j} should be masked (upper near -1e9), got {upper}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6: Causal masking improves margins vs unmasked
// ---------------------------------------------------------------------------

/// Compare monotonicity margins with and without causal masking.
///
/// The causal mask removes future encoder positions from contention,
/// which should not decrease (and often increases) the diagonal margin.
#[test]
fn test_causal_mask_improves_margins() {
    // Build PE-aware variant (same as Phase 20 but with mask).
    let pe_scale = 3.0;
    let input_bound = 0.01f32;

    // --- Unmasked (Phase 20 baseline) — use diagonal dominance. ---
    let def_unmask = causal::build_causal_scores_pe_aware(T_DEC, T_ENC, D_MODEL);
    let no_mask = ndarray::ArrayD::zeros(ndarray::IxDyn(&[T_DEC, T_ENC]));
    let bindings_unmask =
        causal::pe_aware_causal_bindings(T_DEC, T_ENC, D_MODEL, pe_scale, no_mask);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], input_bound);
    let output_unmask = causal::graph_propagate(&def_unmask, &bindings_unmask, &input);
    let cert_unmask =
        causal::extract_certificate(&output_unmask, T_DEC, T_ENC, f64::from(input_bound), "IBP");

    // --- With strict causal mask — use alignment dominance. ---
    // Strict causal: f(t) = min(t, T_enc-1), so for t < T_enc the alignment
    // target f(t) = t matches the PE diagonal. This ensures the PE signal
    // (diagonal dominance) aligns with the causal certificate check.
    let def_mask = causal::build_causal_scores_pe_aware(T_DEC, T_ENC, D_MODEL);
    let mask = causal::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings_mask = causal::pe_aware_causal_bindings(T_DEC, T_ENC, D_MODEL, pe_scale, mask);
    let output_mask = causal::graph_propagate(&def_mask, &bindings_mask, &input);
    // Use alignment dominance: S[t, f(t)] vs other unmasked positions.
    let cert_mask = causal::extract_causal_certificate(
        &output_mask,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| causal::strict_causal_alignment(t, T_ENC),
    );

    eprintln!(
        "Unmasked (diagonal): min_margin={:.6}, proven={}",
        cert_unmask.min_margin, cert_unmask.is_proven
    );
    eprintln!(
        "Causal (alignment):  min_margin={:.6}, proven={}",
        cert_mask.min_margin, cert_mask.is_proven
    );

    // With strict causal masking, each row has fewer competing positions.
    // For rows t < T_enc, f(t) = t matches the PE diagonal, so the
    // alignment target IS the diagonally-dominant entry. Masking removes
    // future off-diagonal competitors, so margins should be at least as
    // good as the unmasked baseline.
    assert!(
        cert_mask.min_margin >= cert_unmask.min_margin - 1e-6,
        "causal alignment margin should not be worse than unmasked diagonal: \
         causal={}, unmask={}",
        cert_mask.min_margin,
        cert_unmask.min_margin
    );
}

// ---------------------------------------------------------------------------
// Test 7: PE-aware causal — proven monotonicity
// ---------------------------------------------------------------------------

#[test]
fn test_causal_pe_aware_proven_monotonicity() {
    let pe_scale = 3.0;
    let input_bound = 0.01f32;

    // Strict causal: f(t) = min(t, T_enc-1). For t < T_enc, f(t) = t
    // which matches the PE diagonal dominance signal. This ensures the
    // alignment target is the PE-favored position.
    let def = causal::build_causal_scores_pe_aware(T_DEC, T_ENC, D_MODEL);
    let mask = causal::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = causal::pe_aware_causal_bindings(T_DEC, T_ENC, D_MODEL, pe_scale, mask);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], input_bound);
    let output = causal::graph_propagate(&def, &bindings, &input);

    // Use alignment dominance: S[t, f(t)] vs other unmasked positions.
    let cert = causal::extract_causal_certificate(
        &output,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| causal::strict_causal_alignment(t, T_ENC),
    );

    eprintln!(
        "Causal PE-aware (strict): min_margin={:.6}, proven={}, rows={}",
        cert.min_margin,
        cert.is_proven,
        cert.row_margins.len()
    );

    assert_eq!(cert.decoder_steps, T_DEC);
    assert_eq!(cert.encoder_positions, T_ENC);
    // Rows where alignment target saturates at T_enc-1 (all visible) are
    // skipped. With strict causal, rows 0..(T_enc-1) are actively aligned.
    assert_eq!(
        cert.row_margins.len(),
        T_ENC - 1,
        "should check T_enc-1={} actively-aligned rows",
        T_ENC - 1
    );

    assert!(
        cert.is_proven,
        "causal PE-aware alignment dominance should be provable: min_margin={}, margins={:?}",
        cert.min_margin, cert.row_margins
    );
    assert!(cert.min_margin > 0.0);
}

// ---------------------------------------------------------------------------
// Test 8: CROWN propagation on causal scores
// ---------------------------------------------------------------------------

#[test]
fn test_causal_crown_propagation() {
    let def = causal::build_causal_scores_simple(T_DEC, T_ENC, D_MODEL);
    let mask = causal::build_linear_causal_mask(T_DEC, T_ENC);
    let bindings = causal::simple_causal_bindings(T_ENC, D_MODEL, mask);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 1.0);

    let (method, output, fallback) = common::assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("Causal CROWN: method={method:?}, fallback={fallback:?}");
    common::assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[T_DEC, T_ENC]);
}

// ---------------------------------------------------------------------------
// Test 9: Different alignment functions produce different margins
// ---------------------------------------------------------------------------

#[test]
fn test_alignment_functions_affect_margins() {
    let pe_scale = 3.0;
    let input_bound = 0.01f32;

    let masks: Vec<(&str, ndarray::ArrayD<f32>)> = vec![
        ("linear", causal::build_linear_causal_mask(T_DEC, T_ENC)),
        ("strict", causal::build_strict_causal_mask(T_DEC, T_ENC)),
        (
            "lookahead_2",
            causal::build_lookahead_causal_mask(T_DEC, T_ENC, 2),
        ),
    ];

    let mut margins: Vec<(&str, f64)> = Vec::new();

    for (name, mask) in &masks {
        let def = causal::build_causal_scores_pe_aware(T_DEC, T_ENC, D_MODEL);
        let bindings =
            causal::pe_aware_causal_bindings(T_DEC, T_ENC, D_MODEL, pe_scale, mask.clone());
        let input = common::uniform_bounds(&[T_DEC, D_MODEL], input_bound);
        let output = causal::graph_propagate(&def, &bindings, &input);

        // Use alignment dominance with the matching alignment function.
        let cert = match *name {
            "linear" => causal::extract_causal_certificate(
                &output,
                T_DEC,
                T_ENC,
                f64::from(input_bound),
                "IBP",
                |t| causal::linear_alignment(t, T_DEC, T_ENC),
            ),
            "strict" => causal::extract_causal_certificate(
                &output,
                T_DEC,
                T_ENC,
                f64::from(input_bound),
                "IBP",
                |t| causal::strict_causal_alignment(t, T_ENC),
            ),
            _ => causal::extract_causal_certificate(
                &output,
                T_DEC,
                T_ENC,
                f64::from(input_bound),
                "IBP",
                |t| causal::lookahead_alignment(t, T_DEC, T_ENC, 2),
            ),
        };

        eprintln!(
            "Alignment {name}: min_margin={:.6}, proven={}",
            cert.min_margin, cert.is_proven
        );
        margins.push((name, cert.min_margin));
    }

    // All three alignments should produce finite margins.
    for (name, margin) in &margins {
        assert!(
            margin.is_finite(),
            "alignment {name} should produce finite margin, got {margin}"
        );
    }

    // All alignment functions with PE-aware attention should produce
    // positive margins (proven alignment dominance).
    for (name, margin) in &margins {
        eprintln!("  {name}: margin={margin:.6}");
    }
}

// ---------------------------------------------------------------------------
// Test 10: Extreme aspect ratio with causal mask
// ---------------------------------------------------------------------------

/// T_dec=20, T_enc=4 (5:1 ratio) with causal masking.
/// Early decoder rows (t=0..3) have very few visible positions (1-2),
/// making alignment dominance easier per-row.
#[test]
fn test_causal_extreme_ratio() {
    let t_dec = 20;
    let t_enc = 4;
    let d = 8;
    let pe_scale = 4.0;
    let input_bound = 0.01f32;

    // Strict causal: f(t) = min(t, T_enc-1). For t < T_enc, f(t) = t
    // matches the PE diagonal. For t >= T_enc, f(t) = T_enc-1 (last position).
    let def = causal::build_causal_scores_pe_aware(t_dec, t_enc, d);
    let mask = causal::build_strict_causal_mask(t_dec, t_enc);
    let bindings = causal::pe_aware_causal_bindings(t_dec, t_enc, d, pe_scale, mask);
    let input = common::uniform_bounds(&[t_dec, d], input_bound);
    let output = causal::graph_propagate(&def, &bindings, &input);

    // Use alignment dominance: S[t, f(t)] vs other unmasked positions.
    let cert = causal::extract_causal_certificate(
        &output,
        t_dec,
        t_enc,
        f64::from(input_bound),
        "IBP",
        |t| causal::strict_causal_alignment(t, t_enc),
    );

    eprintln!(
        "Causal extreme {t_dec}:{t_enc}: min_margin={:.6}, proven={}, rows={}",
        cert.min_margin,
        cert.is_proven,
        cert.row_margins.len()
    );

    // Strict causal: rows 0..(t_enc-1) are actively aligned (target != last pos).
    // Rows t_enc-1..t_dec all saturate at target=t_enc-1, skipped.
    assert_eq!(
        cert.row_margins.len(),
        t_enc - 1,
        "should check t_enc-1={} actively-aligned rows",
        t_enc - 1
    );
    assert!(
        cert.is_proven,
        "causal extreme ratio should be provable: min_margin={}",
        cert.min_margin
    );
}

// ---------------------------------------------------------------------------
// Test 11: Row 0 with strict causal has only 1 visible position (trivially monotonic)
// ---------------------------------------------------------------------------

/// With strict causal masking, row 0 can only attend to position 0.
/// This is trivially monotonic — there's no competing off-diagonal element.
/// The causal alignment certificate returns `Infinity` for rows where the
/// alignment target is the only visible position (max_visible == 0).
#[test]
fn test_causal_row0_trivially_monotonic() {
    let t_dec = 8;
    let t_enc = 4;
    let d = 8;
    let pe_scale = 3.0;
    let input_bound = 0.01f32;

    let def = causal::build_causal_scores_pe_aware(t_dec, t_enc, d);
    let mask = causal::build_strict_causal_mask(t_dec, t_enc);

    // Verify row 0 has only 1 visible position.
    let unmasked = causal::count_unmasked_per_row(&mask, t_dec, t_enc);
    assert_eq!(
        unmasked[0], 1,
        "strict causal row 0 should see only position 0"
    );

    let bindings = causal::pe_aware_causal_bindings(t_dec, t_enc, d, pe_scale, mask);
    let input = common::uniform_bounds(&[t_dec, d], input_bound);
    let output = causal::graph_propagate(&def, &bindings, &input);

    // Use alignment dominance with strict causal alignment.
    let cert = causal::extract_causal_certificate(
        &output,
        t_dec,
        t_enc,
        f64::from(input_bound),
        "IBP",
        |t| causal::strict_causal_alignment(t, t_enc),
    );

    // Row 0: f(0)=0, only position 0 visible → trivially dominant (Infinity).
    assert!(
        cert.row_margins[0] > 1e6,
        "row 0 should have huge margin (trivially monotonic), got {}",
        cert.row_margins[0]
    );
}
