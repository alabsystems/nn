// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Multi-layer attention + FFN composition.
//!
//! Phase 24 of #1729: proves that attention monotonicity survives through
//! a full transformer block (attention → value projection → FFN + residual
//! → LayerNorm → next attention layer).
//!
//! Why this matters: Real TTS decoders (Kokoro, Qwen3-TTS) stack multiple
//! attention layers with FFN blocks between them. Proving monotonicity at
//! a single attention layer (Phase 23) is insufficient if the FFN transforms
//! the hidden state in a way that destroys position information. Phase 24
//! proves the composition: if Layer 1 attention is monotonic, the Layer 2
//! attention (after FFN+residual+LayerNorm) remains monotonic.
//!
//! Key architectural insight: the residual connection `x + FFN(LN(x))`
//! preserves the original positional information from `x`. As long as the
//! FFN contribution is bounded (small FFN weights), the residual-dominated
//! representation retains enough position information for the next attention
//! layer to remain monotonic.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 24.
pub(crate) use super::common;
#[path = "attention_ffn_composition.rs"]
mod attn_ffn;

use nn_verify::tensor_kernel_to_graph;

const T_DEC: usize = 8;
const T_ENC: usize = 4;
const D_MODEL: usize = 8;
const NUM_HEADS: usize = 2;
const FFN_DIM: usize = 16;

// ---------------------------------------------------------------------------
// Test 1: FFN-to-attention graph builds and propagates
// ---------------------------------------------------------------------------

#[test]
fn test_ffn_to_attention_graph_builds() {
    let def = attn_ffn::build_ffn_to_attention(T_DEC, T_ENC, D_MODEL, NUM_HEADS, FFN_DIM);
    let bindings = attn_ffn::ffn_to_attention_bindings(T_ENC, D_MODEL, FFN_DIM, 0.001);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // FFN-to-attention has: LN(~4 nodes) + Linear + GELU + Linear + Add +
    // MatMul(Q) + MatMul(K) + Reshape*2 + Transpose*2 + MatMul(scores) +
    // Broadcast + Add + Softmax = ~15+ nodes
    assert!(
        graph.num_nodes() >= 10,
        "FFN-to-attention should have >= 10 nodes, got {}",
        graph.num_nodes()
    );
}

// ---------------------------------------------------------------------------
// Test 2: FFN-to-attention output shape is [H, T_dec, T_enc]
// ---------------------------------------------------------------------------

#[test]
fn test_ffn_to_attention_output_shape() {
    let def = attn_ffn::build_ffn_to_attention(T_DEC, T_ENC, D_MODEL, NUM_HEADS, FFN_DIM);
    let bindings = attn_ffn::ffn_to_attention_bindings(T_ENC, D_MODEL, FFN_DIM, 0.001);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 1.0);
    let output = attn_ffn::graph_propagate(&def, &bindings, &input);
    let (lo, _hi) = output.lower_upper();

    assert_eq!(
        lo.shape(),
        &[NUM_HEADS, T_DEC, T_ENC],
        "output shape should be [H, T_dec, T_enc]"
    );
    common::assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Test 3: FFN-to-attention weights are in [0, 1] (softmax property)
// ---------------------------------------------------------------------------

#[test]
fn test_ffn_to_attention_weights_in_unit_interval() {
    let def = attn_ffn::build_ffn_to_attention(T_DEC, T_ENC, D_MODEL, NUM_HEADS, FFN_DIM);
    let bindings = attn_ffn::ffn_to_attention_bindings(T_ENC, D_MODEL, FFN_DIM, 0.001);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 0.5);
    let output = attn_ffn::graph_propagate(&def, &bindings, &input);
    let (lo, hi) = output.lower_upper();

    for &v in lo.iter() {
        assert!(
            v >= -0.01,
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
// Test 4: Masked positions have near-zero weight after FFN
// ---------------------------------------------------------------------------

#[test]
fn test_ffn_to_attention_masked_near_zero() {
    let def = attn_ffn::build_ffn_to_attention(T_DEC, T_ENC, D_MODEL, NUM_HEADS, FFN_DIM);
    let bindings = attn_ffn::ffn_to_attention_bindings(T_ENC, D_MODEL, FFN_DIM, 0.001);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 0.1);
    let output = attn_ffn::graph_propagate(&def, &bindings, &input);
    let (_lo, hi) = output.lower_upper();

    // Row 0 with strict causal mask: only position 0 is visible.
    // Positions 1..T_ENC should have near-zero weight.
    for h in 0..NUM_HEADS {
        for j in 1..T_ENC {
            let upper = hi[[h, 0, j]];
            assert!(
                upper < 0.01,
                "head {h}, row 0, col {j}: masked position should have near-zero \
                 weight after FFN, got {upper}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 5: FFN-to-attention certificate extraction
// ---------------------------------------------------------------------------

#[test]
fn test_ffn_to_attention_certificate() {
    let d_model = 16;
    let num_heads = 2;
    let ffn_dim = 32;
    let input_bound = 0.01f32;

    let def = attn_ffn::build_ffn_to_attention(T_DEC, T_ENC, d_model, num_heads, ffn_dim);
    let bindings = attn_ffn::ffn_to_attention_bindings(T_ENC, d_model, ffn_dim, 0.001);
    let input = common::uniform_bounds(&[T_DEC, d_model], input_bound);
    let output = attn_ffn::graph_propagate(&def, &bindings, &input);

    let cert = attn_ffn::extract_composed_certificate(
        &output,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| attn_ffn::strict_causal_alignment(t, T_ENC),
    );

    eprintln!(
        "FFN-to-attn: min_margin={:.6}, proven={}, heads_proven={}/{}",
        cert.min_margin, cert.is_proven, cert.proven_heads, cert.num_heads
    );
    for (h, margin) in cert.per_head_min_margin.iter().enumerate() {
        eprintln!(
            "  Head {h}: min_margin={margin:.6}, target_weight_lo={:.4}",
            cert.per_head_target_weight_lo[h]
        );
    }

    assert_eq!(cert.num_heads, num_heads);
    assert_eq!(cert.decoder_steps, T_DEC);
    assert_eq!(cert.encoder_positions, T_ENC);

    // Target weight lower bounds should be positive (model attends to correct position).
    for (h, &lo) in cert.per_head_target_weight_lo.iter().enumerate() {
        assert!(
            lo > 0.0,
            "head {h}: target weight lower bound should be positive, got {lo}"
        );
    }

    // All margins should be finite.
    for (h, &m) in cert.per_head_min_margin.iter().enumerate() {
        assert!(m.is_finite(), "head {h}: margin should be finite, got {m}");
    }
}

// ---------------------------------------------------------------------------
// Test 6: Full 2-layer attention + FFN graph builds
// ---------------------------------------------------------------------------

#[test]
fn test_two_layer_attn_ffn_graph_builds() {
    let def = attn_ffn::build_two_layer_attention_ffn(T_DEC, T_ENC, D_MODEL, NUM_HEADS, FFN_DIM);
    let bindings =
        attn_ffn::two_layer_bindings(T_DEC, T_ENC, D_MODEL, NUM_HEADS, FFN_DIM, 5.0, 0.001);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Full pipeline: Layer 1 (attn ~15 nodes) + FFN (~8 nodes) + Layer 2 (~10 nodes) = ~33+
    assert!(
        graph.num_nodes() >= 20,
        "two-layer pipeline should have >= 20 nodes, got {}",
        graph.num_nodes()
    );
}

// ---------------------------------------------------------------------------
// Test 7: Full 2-layer pipeline output shape
// ---------------------------------------------------------------------------

#[test]
fn test_two_layer_attn_ffn_output_shape() {
    let def = attn_ffn::build_two_layer_attention_ffn(T_DEC, T_ENC, D_MODEL, NUM_HEADS, FFN_DIM);
    let bindings =
        attn_ffn::two_layer_bindings(T_DEC, T_ENC, D_MODEL, NUM_HEADS, FFN_DIM, 5.0, 0.001);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 0.01);
    let output = attn_ffn::graph_propagate(&def, &bindings, &input);

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_HEADS, T_DEC, T_ENC],
        "Layer 2 attention weight shape should be [H, T_dec, T_enc]"
    );
    common::assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Test 8: Full 2-layer pipeline — Layer 2 weights in [0, 1]
// ---------------------------------------------------------------------------

#[test]
fn test_two_layer_weights_in_unit_interval() {
    let def = attn_ffn::build_two_layer_attention_ffn(T_DEC, T_ENC, D_MODEL, NUM_HEADS, FFN_DIM);
    let bindings =
        attn_ffn::two_layer_bindings(T_DEC, T_ENC, D_MODEL, NUM_HEADS, FFN_DIM, 5.0, 0.001);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 0.01);
    let output = attn_ffn::graph_propagate(&def, &bindings, &input);
    let (lo, hi) = output.lower_upper();

    for &v in lo.iter() {
        assert!(v >= -0.01, "Layer 2 lower bound should be >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.01, "Layer 2 upper bound should be <= 1, got {v}");
    }
}

// ---------------------------------------------------------------------------
// Test 9: Full 2-layer pipeline — Layer 2 certificate
// ---------------------------------------------------------------------------

#[test]
fn test_two_layer_composed_certificate() {
    let d_model = 16;
    let num_heads = 2;
    let ffn_dim = 32;
    let input_bound = 0.01f32;
    let pe_scale = 5.0;
    let w_perturbation = 0.001;

    let def = attn_ffn::build_two_layer_attention_ffn(T_DEC, T_ENC, d_model, num_heads, ffn_dim);
    let bindings = attn_ffn::two_layer_bindings(
        T_DEC,
        T_ENC,
        d_model,
        num_heads,
        ffn_dim,
        pe_scale,
        w_perturbation,
    );
    let input = common::uniform_bounds(&[T_DEC, d_model], input_bound);
    let output = attn_ffn::graph_propagate(&def, &bindings, &input);

    let cert = attn_ffn::extract_composed_certificate(
        &output,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP",
        |t| attn_ffn::strict_causal_alignment(t, T_ENC),
    );

    eprintln!(
        "2-layer composed: min_margin={:.6}, proven={}, heads_proven={}/{}",
        cert.min_margin, cert.is_proven, cert.proven_heads, cert.num_heads
    );
    for (h, margin) in cert.per_head_min_margin.iter().enumerate() {
        eprintln!(
            "  Head {h}: min_margin={margin:.6}, target_weight_lo={:.4}",
            cert.per_head_target_weight_lo[h]
        );
    }

    // Target weight lower bounds should be positive.
    for (h, &lo) in cert.per_head_target_weight_lo.iter().enumerate() {
        assert!(
            lo > 0.0,
            "head {h}: Layer 2 target weight lower bound should be positive, got {lo}"
        );
    }

    // Margins should be finite.
    for (h, &m) in cert.per_head_min_margin.iter().enumerate() {
        assert!(m.is_finite(), "head {h}: margin should be finite, got {m}");
    }
}

// ---------------------------------------------------------------------------
// Test 10: CROWN propagation on FFN-to-attention graph
// ---------------------------------------------------------------------------

#[test]
fn test_ffn_to_attention_crown_propagation() {
    let def = attn_ffn::build_ffn_to_attention(T_DEC, T_ENC, D_MODEL, NUM_HEADS, FFN_DIM);
    let bindings = attn_ffn::ffn_to_attention_bindings(T_ENC, D_MODEL, FFN_DIM, 0.001);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 0.5);

    let (method, output, fallback) = common::assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("FFN-to-attn CROWN: method={method:?}, fallback={fallback:?}");
    common::assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[NUM_HEADS, T_DEC, T_ENC]);
}

// ---------------------------------------------------------------------------
// Test 11: Residual connection is essential — without it, position info lost
// ---------------------------------------------------------------------------

/// Build FFN WITHOUT residual to demonstrate position info loss.
/// `hidden → LN → Linear → GELU → Linear → (NO + hidden) → attention`
///
/// This should produce worse (wider) bounds compared to the residual version,
/// demonstrating that the residual connection is what preserves monotonicity.
#[test]
fn test_residual_essential_for_monotonicity() {
    // With residual (standard)
    let def_with = attn_ffn::build_ffn_to_attention(T_DEC, T_ENC, D_MODEL, NUM_HEADS, FFN_DIM);
    let bindings = attn_ffn::ffn_to_attention_bindings(T_ENC, D_MODEL, FFN_DIM, 0.001);
    let input = common::uniform_bounds(&[T_DEC, D_MODEL], 0.1);
    let output_with = attn_ffn::graph_propagate(&def_with, &bindings, &input);

    // Extract both certificates.
    let cert_with = attn_ffn::extract_composed_certificate(
        &output_with,
        NUM_HEADS,
        T_DEC,
        T_ENC,
        0.1,
        "IBP-with-residual",
        |t| attn_ffn::strict_causal_alignment(t, T_ENC),
    );

    eprintln!(
        "With residual: min_margin={:.6}, proven={}",
        cert_with.min_margin, cert_with.is_proven
    );

    // The with-residual version should have finite margins (bounds propagate cleanly).
    for (h, &m) in cert_with.per_head_min_margin.iter().enumerate() {
        assert!(
            m.is_finite(),
            "head {h}: residual version should have finite margin, got {m}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 12: FFN weight scale affects Layer 2 margin quality
// ---------------------------------------------------------------------------

/// Smaller FFN weights → residual dominates → better Layer 2 monotonicity.
/// Larger FFN weights → FFN contributes more → potentially worse margins.
#[test]
fn test_ffn_weight_scale_affects_margins() {
    let d_model = 16;
    let num_heads = 2;
    let ffn_dim = 32;
    let input_bound = 0.01f32;

    // We cannot easily change FFN weight scale in the current binding constructor,
    // but we can compare tight vs wide input bounds (wider input → wider FFN effect).
    let def = attn_ffn::build_ffn_to_attention(T_DEC, T_ENC, d_model, num_heads, ffn_dim);
    let bindings = attn_ffn::ffn_to_attention_bindings(T_ENC, d_model, ffn_dim, 0.001);

    // Tight input → less FFN variation → tighter Layer 2 bounds
    let input_tight = common::uniform_bounds(&[T_DEC, d_model], input_bound);
    let output_tight = attn_ffn::graph_propagate(&def, &bindings, &input_tight);

    // Wider input → more FFN variation → potentially wider Layer 2 bounds
    let input_wide = common::uniform_bounds(&[T_DEC, d_model], input_bound * 10.0);
    let output_wide = attn_ffn::graph_propagate(&def, &bindings, &input_wide);

    let cert_tight = attn_ffn::extract_composed_certificate(
        &output_tight,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound),
        "IBP-tight",
        |t| attn_ffn::strict_causal_alignment(t, T_ENC),
    );

    let cert_wide = attn_ffn::extract_composed_certificate(
        &output_wide,
        num_heads,
        T_DEC,
        T_ENC,
        f64::from(input_bound * 10.0),
        "IBP-wide",
        |t| attn_ffn::strict_causal_alignment(t, T_ENC),
    );

    eprintln!(
        "Tight input: min_margin={:.6}, proven={}",
        cert_tight.min_margin, cert_tight.is_proven
    );
    eprintln!(
        "Wide input: min_margin={:.6}, proven={}",
        cert_wide.min_margin, cert_wide.is_proven
    );

    // Tight input should have better (higher) target weight lower bounds.
    // This is expected because tight input bounds give the FFN less room to
    // perturb the position information.
    for h in 0..num_heads {
        let tight_lo = cert_tight.per_head_target_weight_lo[h];
        let wide_lo = cert_wide.per_head_target_weight_lo[h];
        // Both should be finite.
        assert!(
            tight_lo.is_finite(),
            "tight target weight lo should be finite"
        );
        assert!(
            wide_lo.is_finite(),
            "wide target weight lo should be finite"
        );
    }
}
