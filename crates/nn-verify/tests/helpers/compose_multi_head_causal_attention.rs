// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Multi-head causal cross-attention monotonicity.
//!
//! Phase 22 of #1729: extends Phase 21's causal attention to multi-head
//! attention with independent per-head alignment dominance checks.
//!
//! Key property: when each attention head independently preserves causal
//! alignment dominance, the multi-head attention as a whole is monotonic.
//! Near-identity W_q/W_k preserve the PE structure per-head, so each
//! head's score matrix inherits the diagonal dominance from PE.
//!
//! The certificate reports per-head margins and the worst-case head's
//! margin determines whether the overall proof succeeds.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 22.

#[path = "multi_head_causal.rs"]
mod mh_causal;

use super::common;
use nn_verify::tensor_kernel_to_graph;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Decoder sequence length.
const T_DEC: usize = 8;

/// Encoder sequence length.
const T_ENC: usize = 4;

/// Model dimension (must be divisible by NUM_HEADS).
const D_MODEL: usize = 8;

/// Number of attention heads.
const NUM_HEADS: usize = 2;

// ---------------------------------------------------------------------------
// Test 1: Simple multi-head causal graph builds and propagates
// ---------------------------------------------------------------------------

#[test]
fn test_mh_causal_simple_graph_builds() {
    let def = mh_causal::build_mh_causal_simple(T_DEC, T_ENC, D_MODEL, NUM_HEADS);
    let mask = mh_causal::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = mh_causal::mh_simple_bindings(T_ENC, D_MODEL, mask);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    assert!(
        graph.num_nodes() >= 4,
        "multi-head graph should have >= 4 nodes (reshape+transpose+matmul+add), got {}",
        graph.num_nodes()
    );
}

// ---------------------------------------------------------------------------
// Test 2: Output shape is [H, T_dec, T_enc]
// ---------------------------------------------------------------------------

#[test]
fn test_mh_causal_output_shape() {
    let def = mh_causal::build_mh_causal_simple(T_DEC, T_ENC, D_MODEL, NUM_HEADS);
    let mask = mh_causal::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = mh_causal::mh_simple_bindings(T_ENC, D_MODEL, mask);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 1.0);
    let output = mh_causal::graph_propagate(&def, &bindings, &input);
    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_HEADS, T_DEC, T_ENC],
        "output shape should be [H, T_dec, T_enc]"
    );
    common::assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Test 3: Masked positions have very negative upper bounds (all heads)
// ---------------------------------------------------------------------------

#[test]
fn test_mh_causal_masked_positions_all_heads() {
    let def = mh_causal::build_mh_causal_simple(T_DEC, T_ENC, D_MODEL, NUM_HEADS);
    let mask = mh_causal::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = mh_causal::mh_simple_bindings(T_ENC, D_MODEL, mask);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 0.1);
    let output = mh_causal::graph_propagate(&def, &bindings, &input);
    let (_lo, hi) = output.lower_upper();

    // With strict causal: row 0 sees only position 0 → positions 1..T_ENC masked.
    // Check all heads have masked positions with very negative upper bounds.
    for h in 0..NUM_HEADS {
        for j in 1..T_ENC {
            let upper = hi[[h, 0, j]];
            assert!(
                upper < -1e8,
                "head {h}, row 0, col {j} should be masked (upper near -1e9), got {upper}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4: Per-head certificate — all heads proven with PE-aware projected
// ---------------------------------------------------------------------------

#[test]
fn test_mh_causal_projected_all_heads_proven() {
    // Use D=16 so each head (d_k=8) gets 4 frequency pairs spanning the spectrum.
    // With D=8 and H=2, head 1 gets only high-frequency-base dims (omega ~0.01)
    // where positions are nearly indistinguishable for small T_enc.
    let d_model = 16;
    let num_heads = 2; // d_k=8, 4 freq pairs per head
    let pe_scale = 5.0;
    let w_perturbation = 0.001;
    let input_bound = 0.01f32;

    let def = mh_causal::build_mh_causal_projected(T_DEC, T_ENC, d_model, num_heads);
    let mask = mh_causal::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = mh_causal::mh_projected_bindings(
        T_DEC,
        T_ENC,
        d_model,
        num_heads,
        pe_scale,
        w_perturbation,
        mask,
    );
    let input = common::uniform_bounds(&[T_DEC, d_model], input_bound);
    let output = mh_causal::graph_propagate(&def, &bindings, &input);

    let cert = mh_causal::extract_mh_causal_certificate(
        &output,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| mh_causal::strict_causal_alignment(t, T_ENC),
    );

    eprintln!(
        "Multi-head projected: min_margin={:.6}, proven={}, heads_proven={}/{}",
        cert.min_margin, cert.is_proven, cert.proven_heads, cert.num_heads
    );
    for (h, margin) in cert.per_head_min_margin.iter().enumerate() {
        eprintln!("  Head {h}: min_margin={margin:.6}");
    }

    assert_eq!(cert.num_heads, num_heads);
    assert_eq!(cert.decoder_steps, T_DEC);
    assert_eq!(cert.encoder_positions, T_ENC);

    // All heads should be individually proven.
    assert_eq!(
        cert.proven_heads, num_heads,
        "all {} heads should have proven alignment dominance, only {} proven",
        num_heads, cert.proven_heads
    );
    assert!(
        cert.is_proven,
        "overall certificate should be proven: min_margin={}",
        cert.min_margin
    );
    assert!(cert.min_margin > 0.0);
}

// ---------------------------------------------------------------------------
// Test 5: Per-head margins differ but all positive
// ---------------------------------------------------------------------------

/// With near-identity W_q/W_k, heads see different PE frequency bands.
/// Low-frequency heads (head 0) have stronger margins than high-frequency heads.
/// With sufficient d_k (enough freq pairs per head), all heads are proven.
#[test]
fn test_mh_causal_per_head_margins_differ() {
    // Use D=16 so each head has enough frequency diversity.
    let d_model = 16;
    let num_heads = 2;
    let pe_scale = 5.0;
    let w_perturbation = 0.001;
    let input_bound = 0.01f32;

    let def = mh_causal::build_mh_causal_projected(T_DEC, T_ENC, d_model, num_heads);
    let mask = mh_causal::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = mh_causal::mh_projected_bindings(
        T_DEC,
        T_ENC,
        d_model,
        num_heads,
        pe_scale,
        w_perturbation,
        mask,
    );
    let input = common::uniform_bounds(&[T_DEC, d_model], input_bound);
    let output = mh_causal::graph_propagate(&def, &bindings, &input);

    let cert = mh_causal::extract_mh_causal_certificate(
        &output,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| mh_causal::strict_causal_alignment(t, T_ENC),
    );

    // Each head should have a finite positive minimum margin.
    for (h, margin) in cert.per_head_min_margin.iter().enumerate() {
        assert!(
            margin.is_finite() && *margin > 0.0,
            "head {h} should have positive finite margin, got {margin}"
        );
    }

    // Per-head margins should differ — head 0 (low-freq PE) typically stronger.
    assert!(
        (cert.per_head_min_margin[0] - cert.per_head_min_margin[1]).abs() > 1e-6,
        "head margins should differ due to frequency distribution"
    );

    // Per-head row margins should have the same count (same alignment function).
    for h in 0..num_heads {
        assert_eq!(
            cert.per_head_row_margins[h].len(),
            T_ENC - 1,
            "head {h} should check T_ENC-1={} actively-aligned rows",
            T_ENC - 1
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6: 4-head attention with larger dimension
// ---------------------------------------------------------------------------

#[test]
fn test_mh_causal_4_heads() {
    let t_dec = 8;
    let t_enc = 4;
    // D=32 gives d_k=8 per head (4 freq pairs each), ensuring each head has
    // enough frequency diversity for position-distinctive PE vectors.
    let d_model = 32;
    let num_heads = 4;
    let pe_scale = 6.0;
    let w_perturbation = 0.001;
    let input_bound = 0.01f32;

    let def = mh_causal::build_mh_causal_projected(t_dec, t_enc, d_model, num_heads);
    let mask = mh_causal::build_strict_causal_mask(t_dec, t_enc);
    let bindings = mh_causal::mh_projected_bindings(
        t_dec,
        t_enc,
        d_model,
        num_heads,
        pe_scale,
        w_perturbation,
        mask,
    );
    let input = common::uniform_bounds(&[t_dec, d_model], input_bound);
    let output = mh_causal::graph_propagate(&def, &bindings, &input);

    let cert = mh_causal::extract_mh_causal_certificate(
        &output,
        num_heads,
        t_dec,
        t_enc,
        f64::from(input_bound),
        "IBP",
        |t| mh_causal::strict_causal_alignment(t, t_enc),
    );

    eprintln!(
        "4-head: min_margin={:.6}, proven={}, heads_proven={}/{}",
        cert.min_margin, cert.is_proven, cert.proven_heads, cert.num_heads
    );
    for (h, margin) in cert.per_head_min_margin.iter().enumerate() {
        eprintln!("  Head {h}: min_margin={margin:.6}");
    }

    assert_eq!(cert.num_heads, num_heads);
    // All heads should be proven with sufficient d_k and PE scale.
    assert_eq!(
        cert.proven_heads, num_heads,
        "all 4 heads should be proven, got {}",
        cert.proven_heads
    );
    assert!(cert.is_proven);
}

// ---------------------------------------------------------------------------
// Test 7: Row 0 trivially monotonic in all heads
// ---------------------------------------------------------------------------

/// With strict causal, row 0 sees only position 0 → trivially dominant.
/// All heads should report infinite margin for row 0.
#[test]
fn test_mh_causal_row0_trivially_monotonic_all_heads() {
    let d_model = 16;
    let num_heads = 2;
    let pe_scale = 5.0;
    let w_perturbation = 0.001;
    let input_bound = 0.01f32;

    let def = mh_causal::build_mh_causal_projected(T_DEC, T_ENC, d_model, num_heads);
    let mask = mh_causal::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = mh_causal::mh_projected_bindings(
        T_DEC,
        T_ENC,
        d_model,
        num_heads,
        pe_scale,
        w_perturbation,
        mask,
    );
    let input = common::uniform_bounds(&[T_DEC, d_model], input_bound);
    let output = mh_causal::graph_propagate(&def, &bindings, &input);

    let cert = mh_causal::extract_mh_causal_certificate(
        &output,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| mh_causal::strict_causal_alignment(t, T_ENC),
    );

    // Row 0 is the first row_margin in each head.
    for h in 0..num_heads {
        assert!(
            cert.per_head_row_margins[h][0] > 1e6,
            "head {h}: row 0 should have huge margin (trivially monotonic), got {}",
            cert.per_head_row_margins[h][0]
        );
    }
}

// ---------------------------------------------------------------------------
// Test 8: CROWN propagation on multi-head causal
// ---------------------------------------------------------------------------

#[test]
fn test_mh_causal_crown_propagation() {
    let def = mh_causal::build_mh_causal_simple(T_DEC, T_ENC, D_MODEL, NUM_HEADS);
    let mask = mh_causal::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = mh_causal::mh_simple_bindings(T_ENC, D_MODEL, mask);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 1.0);

    let (method, output, fallback) = common::assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("MH causal CROWN: method={method:?}, fallback={fallback:?}");
    common::assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[NUM_HEADS, T_DEC, T_ENC]);
}

// ---------------------------------------------------------------------------
// Test 9: Multi-head vs single-head margin comparison
// ---------------------------------------------------------------------------

/// More heads means smaller d_k per head, which means fewer PE frequency pairs
/// and potentially weaker diagonal dominance. With sufficient D, both H=2 and
/// H=4 prove, but H=2 should have stronger worst-case margin than H=4.
#[test]
fn test_mh_2_heads_vs_4_heads_margins() {
    let d_model = 32; // d_k=16 for H=2, d_k=8 for H=4
    let pe_scale = 6.0;
    let input_bound = 0.01f32;

    // 2-head: d_k=16 (8 freq pairs per head)
    let def_2h = mh_causal::build_mh_causal_projected(T_DEC, T_ENC, d_model, 2);
    let mask = mh_causal::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings_2h =
        mh_causal::mh_projected_bindings(T_DEC, T_ENC, d_model, 2, pe_scale, 0.001, mask.clone());
    let input_2h = common::uniform_bounds(&[T_DEC, d_model], input_bound);
    let output_2h = mh_causal::graph_propagate(&def_2h, &bindings_2h, &input_2h);

    let cert_2h = mh_causal::extract_mh_causal_certificate(
        &output_2h,
        2,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| mh_causal::strict_causal_alignment(t, T_ENC),
    );

    // 4-head: d_k=8 (4 freq pairs per head)
    let def_4h = mh_causal::build_mh_causal_projected(T_DEC, T_ENC, d_model, 4);
    let bindings_4h =
        mh_causal::mh_projected_bindings(T_DEC, T_ENC, d_model, 4, pe_scale, 0.001, mask);
    let input_4h = common::uniform_bounds(&[T_DEC, d_model], input_bound);
    let output_4h = mh_causal::graph_propagate(&def_4h, &bindings_4h, &input_4h);

    let cert_4h = mh_causal::extract_mh_causal_certificate(
        &output_4h,
        4,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| mh_causal::strict_causal_alignment(t, T_ENC),
    );

    eprintln!(
        "H=2: min_margin={:.6}, proven={}, heads_proven={}/2",
        cert_2h.min_margin, cert_2h.is_proven, cert_2h.proven_heads
    );
    eprintln!(
        "H=4: min_margin={:.6}, proven={}, heads_proven={}/4",
        cert_4h.min_margin, cert_4h.is_proven, cert_4h.proven_heads
    );

    // Both should prove with sufficient D.
    assert!(cert_2h.is_proven, "2-head should be proven");
    assert!(cert_4h.is_proven, "4-head should be proven");

    // Both should have finite positive margins.
    assert!(cert_2h.min_margin > 0.0 && cert_2h.min_margin.is_finite());
    assert!(cert_4h.min_margin > 0.0 && cert_4h.min_margin.is_finite());

    // H=2 should have stronger worst-case margin (more freq pairs per head).
    assert!(
        cert_2h.min_margin > cert_4h.min_margin,
        "2-head margin ({:.6}) should exceed 4-head margin ({:.6})",
        cert_2h.min_margin,
        cert_4h.min_margin
    );
}

// ---------------------------------------------------------------------------
// Test 10: Linear alignment with multi-head
// ---------------------------------------------------------------------------

#[test]
fn test_mh_causal_linear_alignment() {
    let d_model = 16;
    let num_heads = 2;
    let pe_scale = 5.0;
    let w_perturbation = 0.001;
    let input_bound = 0.01f32;

    let def = mh_causal::build_mh_causal_projected(T_DEC, T_ENC, d_model, num_heads);
    let mask = mh_causal::build_linear_causal_mask(T_DEC, T_ENC);
    let bindings = mh_causal::mh_projected_bindings(
        T_DEC,
        T_ENC,
        d_model,
        num_heads,
        pe_scale,
        w_perturbation,
        mask,
    );
    let input = common::uniform_bounds(&[T_DEC, d_model], input_bound);
    let output = mh_causal::graph_propagate(&def, &bindings, &input);

    let cert = mh_causal::extract_mh_causal_certificate(
        &output,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| mh_causal::linear_alignment(t, T_DEC, T_ENC),
    );

    eprintln!(
        "MH linear alignment: min_margin={:.6}, proven={}, heads_proven={}/{}",
        cert.min_margin, cert.is_proven, cert.proven_heads, cert.num_heads
    );

    // All heads should have finite margins.
    for (h, margin) in cert.per_head_min_margin.iter().enumerate() {
        assert!(
            margin.is_finite(),
            "head {h} should have finite margin, got {margin}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 11: Extreme aspect ratio (20:4) with 2 heads
// ---------------------------------------------------------------------------

#[test]
fn test_mh_causal_extreme_ratio() {
    let t_dec = 20;
    let t_enc = 4;
    // D=16 gives d_k=8 per head (4 freq pairs), sufficient for position
    // distinctiveness even with high decoder/encoder ratio.
    let d_model = 16;
    let num_heads = 2;
    let pe_scale = 5.0;
    let w_perturbation = 0.001;
    let input_bound = 0.01f32;

    let def = mh_causal::build_mh_causal_projected(t_dec, t_enc, d_model, num_heads);
    let mask = mh_causal::build_strict_causal_mask(t_dec, t_enc);
    let bindings = mh_causal::mh_projected_bindings(
        t_dec,
        t_enc,
        d_model,
        num_heads,
        pe_scale,
        w_perturbation,
        mask,
    );
    let input = common::uniform_bounds(&[t_dec, d_model], input_bound);
    let output = mh_causal::graph_propagate(&def, &bindings, &input);

    let cert = mh_causal::extract_mh_causal_certificate(
        &output,
        num_heads,
        t_dec,
        t_enc,
        f64::from(input_bound),
        "IBP",
        |t| mh_causal::strict_causal_alignment(t, t_enc),
    );

    eprintln!(
        "MH extreme {t_dec}:{t_enc}: min_margin={:.6}, proven={}, heads_proven={}/{}",
        cert.min_margin, cert.is_proven, cert.proven_heads, cert.num_heads
    );

    // Each head should check t_enc-1 actively-aligned rows.
    for h in 0..num_heads {
        assert_eq!(
            cert.per_head_row_margins[h].len(),
            t_enc - 1,
            "head {h}: should check t_enc-1={} rows",
            t_enc - 1
        );
    }

    assert!(
        cert.is_proven,
        "extreme ratio should be provable: min_margin={}",
        cert.min_margin
    );
}
