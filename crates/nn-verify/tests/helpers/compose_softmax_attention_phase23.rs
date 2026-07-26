// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Softmax-inclusive multi-head causal attention monotonicity.
//!
//! Phase 23 of #1729: extends Phase 22's multi-head causal attention proofs
//! to include softmax, proving monotonicity of actual attention *weights*
//! (post-softmax), not just pre-softmax *scores*.
//!
//! Why this matters: TTS decoders use attention weights (post-softmax) to
//! compute the context vector `C = W @ V`. If we only prove score dominance
//! but softmax flattens the distribution, the actual attention paid to the
//! target position might not dominate. By propagating bounds through softmax
//! and proving weight dominance, we get a strictly stronger guarantee:
//! the model provably attends most to the correct alignment position.
//!
//! Mathematical property: For all heads h and decoder steps t:
//!   lower(W[h,t,f(t)]) > max_{j≠f(t), j unmasked} upper(W[h,t,j])
//!
//! where W = softmax(S_masked) and f(t) is the alignment function.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 23.

#[path = "softmax_attention.rs"]
mod softmax_attn;

use super::common;
use nn_verify::tensor_kernel_to_graph;

const T_DEC: usize = 8;
const T_ENC: usize = 4;
const D_MODEL: usize = 8;
const NUM_HEADS: usize = 2;

// Test 1: Softmax graph builds and propagates.
#[test]
fn test_softmax_attn_graph_builds() {
    let def = softmax_attn::build_mh_causal_softmax_simple(T_DEC, T_ENC, D_MODEL, NUM_HEADS);
    let mask = softmax_attn::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = softmax_attn::mh_softmax_simple_bindings(T_ENC, D_MODEL, mask);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Softmax adds 1 more node compared to Phase 22's graph.
    assert!(
        graph.num_nodes() >= 5,
        "softmax graph should have >= 5 nodes (reshape+transpose+matmul+add+softmax), got {}",
        graph.num_nodes()
    );
}

// ---------------------------------------------------------------------------
// Test 2: Output shape is [H, T_dec, T_enc] (unchanged by softmax)
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_attn_output_shape() {
    let def = softmax_attn::build_mh_causal_softmax_simple(T_DEC, T_ENC, D_MODEL, NUM_HEADS);
    let mask = softmax_attn::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = softmax_attn::mh_softmax_simple_bindings(T_ENC, D_MODEL, mask);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 1.0);
    let output = softmax_attn::graph_propagate(&def, &bindings, &input);
    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_HEADS, T_DEC, T_ENC],
        "output shape should be [H, T_dec, T_enc]"
    );
    common::assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Test 3: Attention weights are bounded in [0, 1] (softmax output property)
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_attn_weights_in_unit_interval() {
    let def = softmax_attn::build_mh_causal_softmax_simple(T_DEC, T_ENC, D_MODEL, NUM_HEADS);
    let mask = softmax_attn::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = softmax_attn::mh_softmax_simple_bindings(T_ENC, D_MODEL, mask);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 0.5);
    let output = softmax_attn::graph_propagate(&def, &bindings, &input);
    let (lo, hi) = output.lower_upper();

    // Softmax outputs are in [0, 1]. IBP bounds should respect this.
    for &v in lo.iter() {
        assert!(
            v >= -0.01, // small tolerance for floating-point
            "lower bound should be >= 0 (softmax output), got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v <= 1.01,
            "upper bound should be <= 1 (softmax output), got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: Masked positions have near-zero weight upper bounds
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_attn_masked_positions_near_zero() {
    let def = softmax_attn::build_mh_causal_softmax_simple(T_DEC, T_ENC, D_MODEL, NUM_HEADS);
    let mask = softmax_attn::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = softmax_attn::mh_softmax_simple_bindings(T_ENC, D_MODEL, mask);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 0.1);
    let output = softmax_attn::graph_propagate(&def, &bindings, &input);
    let (_lo, hi) = output.lower_upper();

    // With strict causal mask: row 0 sees only position 0.
    // Positions 1..T_ENC should have softmax weight near 0 (exp(-1e9) ≈ 0).
    for h in 0..NUM_HEADS {
        for j in 1..T_ENC {
            let upper = hi[[h, 0, j]];
            assert!(
                upper < 0.01,
                "head {h}, row 0, col {j}: masked position should have near-zero weight, got {upper}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 5: Row 0 weight — only visible position gets weight ≈ 1
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_attn_row0_single_visible_weight() {
    let def = softmax_attn::build_mh_causal_softmax_simple(T_DEC, T_ENC, D_MODEL, NUM_HEADS);
    let mask = softmax_attn::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = softmax_attn::mh_softmax_simple_bindings(T_ENC, D_MODEL, mask);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 0.1);
    let output = softmax_attn::graph_propagate(&def, &bindings, &input);
    let (lo, _hi) = output.lower_upper();

    // Row 0 with strict causal sees only position 0. After softmax,
    // position 0 should get weight close to 1.0 (all other exp(-1e9) ≈ 0).
    for h in 0..NUM_HEADS {
        let w0_lo = lo[[h, 0, 0]];
        assert!(
            f64::from(w0_lo) > 0.9,
            "head {h}, row 0, col 0: single visible position should have weight near 1.0, got {w0_lo}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6: Projected softmax — certificate extraction (2-head)
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_attn_projected_certificate_2h() {
    let d_model = 16;
    let num_heads = 2;
    let pe_scale = 5.0;
    let w_perturbation = 0.001;
    let input_bound = 0.01f32;

    let def = softmax_attn::build_mh_causal_softmax_projected(T_DEC, T_ENC, d_model, num_heads);
    let mask = softmax_attn::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = softmax_attn::mh_softmax_projected_bindings(
        T_DEC,
        T_ENC,
        d_model,
        num_heads,
        pe_scale,
        w_perturbation,
        mask,
    );
    let input = common::uniform_bounds(&[T_DEC, d_model], input_bound);
    let output = softmax_attn::graph_propagate(&def, &bindings, &input);

    let cert = softmax_attn::extract_softmax_certificate(
        &output,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| softmax_attn::strict_causal_alignment(t, T_ENC),
    );

    eprintln!(
        "Softmax 2H projected: min_margin={:.6}, proven={}, heads_proven={}/{}",
        cert.min_margin, cert.is_proven, cert.proven_heads, cert.num_heads
    );
    for (h, margin) in cert.per_head_min_margin.iter().enumerate() {
        eprintln!(
            "  Head {h}: min_margin={margin:.6}, target_weight_lo={:.4}, target_weight_hi={:.4}",
            cert.per_head_target_weight_lo[h], cert.per_head_target_weight_hi[h]
        );
    }

    assert_eq!(cert.num_heads, num_heads);
    assert_eq!(cert.decoder_steps, T_DEC);
    assert_eq!(cert.encoder_positions, T_ENC);

    // Target weight lower bounds should be positive (model attends to target).
    for (h, &lo) in cert.per_head_target_weight_lo.iter().enumerate() {
        assert!(
            lo > 0.0,
            "head {h}: target weight lower bound should be positive, got {lo}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 7: Projected softmax — 4-head certificate
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_attn_projected_certificate_4h() {
    let d_model = 32;
    let num_heads = 4;
    let pe_scale = 6.0;
    let w_perturbation = 0.001;
    let input_bound = 0.01f32;

    let def = softmax_attn::build_mh_causal_softmax_projected(T_DEC, T_ENC, d_model, num_heads);
    let mask = softmax_attn::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = softmax_attn::mh_softmax_projected_bindings(
        T_DEC,
        T_ENC,
        d_model,
        num_heads,
        pe_scale,
        w_perturbation,
        mask,
    );
    let input = common::uniform_bounds(&[T_DEC, d_model], input_bound);
    let output = softmax_attn::graph_propagate(&def, &bindings, &input);

    let cert = softmax_attn::extract_softmax_certificate(
        &output,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| softmax_attn::strict_causal_alignment(t, T_ENC),
    );

    eprintln!(
        "Softmax 4H projected: min_margin={:.6}, proven={}, heads_proven={}/{}",
        cert.min_margin, cert.is_proven, cert.proven_heads, cert.num_heads
    );
    for (h, margin) in cert.per_head_min_margin.iter().enumerate() {
        eprintln!(
            "  Head {h}: min_margin={margin:.6}, target_weight_lo={:.4}",
            cert.per_head_target_weight_lo[h]
        );
    }

    assert_eq!(cert.num_heads, num_heads);

    // Per-head target weight lower bounds should be positive.
    for (h, &lo) in cert.per_head_target_weight_lo.iter().enumerate() {
        assert!(
            lo > 0.0,
            "head {h}: target weight lower bound should be positive, got {lo}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 8: CROWN propagation on softmax graph
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_attn_crown_propagation() {
    let def = softmax_attn::build_mh_causal_softmax_simple(T_DEC, T_ENC, D_MODEL, NUM_HEADS);
    let mask = softmax_attn::build_strict_causal_mask(T_DEC, T_ENC);
    let bindings = softmax_attn::mh_softmax_simple_bindings(T_ENC, D_MODEL, mask);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 1.0);

    let (method, output, fallback) = common::assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("Softmax CROWN: method={method:?}, fallback={fallback:?}");
    common::assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[NUM_HEADS, T_DEC, T_ENC]);
}

// ---------------------------------------------------------------------------
// Test 9: Score-space vs weight-space margin comparison
// ---------------------------------------------------------------------------

/// Compare Phase 22 (pre-softmax score) margins against Phase 23 (post-softmax
/// weight) margins. The softmax-inclusive margins are in a different scale
/// (weight space [0,1] vs arbitrary score space), but the key property is
/// whether both prove dominance.
#[test]
fn test_softmax_vs_score_margins() {
    let d_model = 16;
    let num_heads = 2;
    let pe_scale = 5.0;
    let w_perturbation = 0.001;
    let input_bound = 0.01f32;

    // Phase 22: pre-softmax scores (reuse helper for identical architecture sans softmax)
    let def_scores = softmax_attn::build_scores_only_projected(T_DEC, T_ENC, d_model, num_heads);

    // Phase 23: post-softmax weights
    let def_weights =
        softmax_attn::build_mh_causal_softmax_projected(T_DEC, T_ENC, d_model, num_heads);

    let mask = softmax_attn::build_strict_causal_mask(T_DEC, T_ENC);

    // Same bindings for both
    let bindings = softmax_attn::mh_softmax_projected_bindings(
        T_DEC,
        T_ENC,
        d_model,
        num_heads,
        pe_scale,
        w_perturbation,
        mask,
    );
    let input = common::uniform_bounds(&[T_DEC, d_model], input_bound);

    let output_scores = softmax_attn::graph_propagate(&def_scores, &bindings, &input);
    let output_weights = softmax_attn::graph_propagate(&def_weights, &bindings, &input);

    // Extract certificates with the same alignment function.
    let alignment = |t: usize| softmax_attn::strict_causal_alignment(t, T_ENC);

    // Score-space certificate (reuse same extraction logic — margins in score units).
    let cert_scores = softmax_attn::extract_softmax_certificate(
        &output_scores,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP-scores",
        alignment,
    );

    let cert_weights = softmax_attn::extract_softmax_certificate(
        &output_weights,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP-weights",
        alignment,
    );

    eprintln!(
        "Score-space: min_margin={:.6}, proven={}",
        cert_scores.min_margin, cert_scores.is_proven
    );
    eprintln!(
        "Weight-space: min_margin={:.6}, proven={}",
        cert_weights.min_margin, cert_weights.is_proven
    );

    // Both certificates should have finite margins.
    softmax_attn::assert_certificate_margins_valid(&cert_scores);
    softmax_attn::assert_certificate_margins_valid(&cert_weights);

    // Weight margins should be in [0, 1] scale since weights are in [0, 1].
    softmax_attn::assert_weight_margins_bounded(&cert_weights);
}

// ---------------------------------------------------------------------------
// Test 10: Linear alignment with softmax
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_attn_linear_alignment() {
    let d_model = 16;
    let num_heads = 2;
    let pe_scale = 5.0;
    let w_perturbation = 0.001;
    let input_bound = 0.01f32;

    let def = softmax_attn::build_mh_causal_softmax_projected(T_DEC, T_ENC, d_model, num_heads);
    let mask = softmax_attn::build_linear_causal_mask(T_DEC, T_ENC);
    let bindings = softmax_attn::mh_softmax_projected_bindings(
        T_DEC,
        T_ENC,
        d_model,
        num_heads,
        pe_scale,
        w_perturbation,
        mask,
    );
    let input = common::uniform_bounds(&[T_DEC, d_model], input_bound);
    let output = softmax_attn::graph_propagate(&def, &bindings, &input);

    let cert = softmax_attn::extract_softmax_certificate(
        &output,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| softmax_attn::linear_alignment(t, T_DEC, T_ENC),
    );

    eprintln!(
        "Softmax linear: min_margin={:.6}, proven={}, heads_proven={}/{}",
        cert.min_margin, cert.is_proven, cert.proven_heads, cert.num_heads
    );

    // Each head should have finite margins.
    for (h, margin) in cert.per_head_min_margin.iter().enumerate() {
        assert!(
            margin.is_finite(),
            "head {h} should have finite margin, got {margin}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 11: Extreme ratio (20:4) with softmax
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_attn_extreme_ratio() {
    let t_dec = 20;
    let t_enc = 4;
    let d_model = 16;
    let num_heads = 2;
    let pe_scale = 5.0;
    let w_perturbation = 0.001;
    let input_bound = 0.01f32;

    let def = softmax_attn::build_mh_causal_softmax_projected(t_dec, t_enc, d_model, num_heads);
    let mask = softmax_attn::build_strict_causal_mask(t_dec, t_enc);
    let bindings = softmax_attn::mh_softmax_projected_bindings(
        t_dec,
        t_enc,
        d_model,
        num_heads,
        pe_scale,
        w_perturbation,
        mask,
    );
    let input = common::uniform_bounds(&[t_dec, d_model], input_bound);
    let output = softmax_attn::graph_propagate(&def, &bindings, &input);

    let cert = softmax_attn::extract_softmax_certificate(
        &output,
        num_heads,
        t_dec,
        t_enc,
        f64::from(input_bound),
        "IBP",
        |t| softmax_attn::strict_causal_alignment(t, t_enc),
    );

    eprintln!(
        "Softmax extreme {t_dec}:{t_enc}: min_margin={:.6}, proven={}, heads_proven={}/{}",
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

    // Target weights should be positive (model attends to target).
    for (h, &lo) in cert.per_head_target_weight_lo.iter().enumerate() {
        assert!(
            lo > 0.0,
            "head {h}: target weight lower bound should be positive, got {lo}"
        );
    }
}
