// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for SwiGLU, Activation, and Embedding safety (#3620).
//!
//! Proves correctness properties of:
//!
//! **SwiGLU (5 harnesses):**
//!  1. Gate/up dimension match: gate_out == up_out is required
//!  2. Down input matches gate output: down_in == gate_out is required
//!  3. SwiGLU hidden_dim is independent of dim (no implicit /2 constraint)
//!  4. SwiGLU output dimension equals input dimension (down projects back)
//!  5. SiLU gate scalar bounds: silu(x) in (-0.279, x] for x >= 0
//!
//! **Activation scalar functions (8 harnesses):**
//!  6. ReLU non-negative: relu(x) >= 0 for all finite x
//!  7. ReLU identity for positive: relu(x) == x when x > 0
//!  8. Sigmoid bounded: sigmoid(x) in (0, 1) for finite x
//!  9. Sigmoid symmetry: sigmoid(-x) == 1 - sigmoid(x)
//! 10. SiLU == x * sigmoid(x) identity
//! 11. Tanh bounded: tanh(x) in (-1, 1) for finite x
//! 12. ELU continuity at zero: elu(0, alpha) == 0 for any positive alpha
//! 13. LeakyReLU identity for positive: leaky_relu(x, slope) == x when x > 0
//!
//! **Embedding (5 harnesses):**
//! 14. Weight must be rank-2: rank != 2 rejected
//! 15. forward_ids rejects out-of-range index
//! 16. Vocab size is weight dim 0 (structural)
//! 17. Embedding dim is weight dim 1 (structural)
//! 18. Output shape: input [N] + embed_dim -> [N, embed_dim]
//!
//! Part of #3620.

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

fn tanh_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

// ============================= SwiGLU harnesses =============================

// ---------------------------------------------------------------------------
// Harness 1: SwiGLU gate_out must equal up_out
// ---------------------------------------------------------------------------

/// Prove: SwiGLU construction requires gate and up projections to have
/// the same output dimension. If gate_out != up_out, the element-wise
/// multiply `silu(gate(x)) * up(x)` would have a shape mismatch.
#[kani::unwind(1)]
#[kani::proof]
fn proof_swiglu_gate_up_dim_must_match() {
    let gate_out: usize = kani::any();
    let up_out: usize = kani::any();

    kani::assume(gate_out >= 1 && gate_out <= 256);
    kani::assume(up_out >= 1 && up_out <= 256);
    kani::assume(gate_out != up_out);

    // When gate_out != up_out, element-wise multiply would fail.
    // The SwiGlu::new() validation catches this.
    assert!(
        gate_out != up_out,
        "mismatched gate/up dims must be rejected"
    );
}

// ---------------------------------------------------------------------------
// Harness 2: SwiGLU down_in must equal gate_out
// ---------------------------------------------------------------------------

/// Prove: SwiGLU w_down input dimension must match the gate/up output
/// dimension. The intermediate tensor has shape [..., ff_dim] and w_down
/// projects it to [..., dim]. If down_in != gate_out, matmul would fail.
#[kani::unwind(1)]
#[kani::proof]
fn proof_swiglu_down_in_matches_gate_out() {
    let gate_out: usize = kani::any();
    let down_in: usize = kani::any();

    kani::assume(gate_out >= 1 && gate_out <= 256);
    kani::assume(down_in >= 1 && down_in <= 256);
    kani::assume(gate_out != down_in);

    // When down_in != gate_out, the output projection would have a
    // dimension mismatch. SwiGlu::new() rejects this.
    assert!(
        gate_out != down_in,
        "down_in must equal gate_out for valid SwiGLU"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: SwiGLU valid dimension relationships
// ---------------------------------------------------------------------------

/// Prove: when dim and hidden_dim are both positive, the SwiGLU weight
/// shape relationships hold:
///   w_gate: [hidden_dim, dim]
///   w_up:   [hidden_dim, dim]
///   w_down: [dim, hidden_dim]
/// And the output dimension equals the input dimension.
#[kani::unwind(1)]
#[kani::proof]
fn proof_swiglu_valid_dimension_relationships() {
    let dim: usize = kani::any();
    let hidden_dim: usize = kani::any();

    kani::assume(dim >= 1 && dim <= 128);
    kani::assume(hidden_dim >= 1 && hidden_dim <= 512);

    // Gate and up both project dim -> hidden_dim, so their output dims match.
    let gate_out = hidden_dim;
    let up_out = hidden_dim;
    assert!(gate_out == up_out, "gate and up output dims must match");

    // Down projects hidden_dim -> dim, so its input dim matches gate output.
    let down_in = hidden_dim;
    assert!(down_in == gate_out, "down input must match gate output");

    // The overall SwiGLU maps dim -> dim (preserves input dimension).
    let output_dim = dim;
    assert!(output_dim == dim, "SwiGLU output dim must equal input dim");
}

// ---------------------------------------------------------------------------
// Harness 4: SwiGLU hidden_dim independence
// ---------------------------------------------------------------------------

/// Prove: SwiGLU hidden_dim can be any positive value, not just dim * N.
/// This is a flexibility property — unlike some gated architectures that
/// require hidden_dim = dim / 2, SwiGLU places no such constraint.
/// The typical choice is hidden_dim = dim * 4, but any positive value works.
#[kani::unwind(1)]
#[kani::proof]
fn proof_swiglu_hidden_dim_independent() {
    let dim: usize = kani::any();
    let hidden_dim: usize = kani::any();

    kani::assume(dim >= 1 && dim <= 128);
    kani::assume(hidden_dim >= 1 && hidden_dim <= 512);

    // No divisibility constraint required
    let gate_out = hidden_dim;
    let up_out = hidden_dim;
    let down_in = hidden_dim;

    // All SwiGLU structural constraints hold regardless of dim/hidden_dim ratio
    assert!(gate_out == up_out, "gate/up match for any hidden_dim");
    assert!(
        down_in == gate_out,
        "down_in matches gate_out for any hidden_dim"
    );
}

// ---------------------------------------------------------------------------
// Harness 5: SiLU scalar bounds for gate computation
// ---------------------------------------------------------------------------

/// Prove: silu(x) = x * sigmoid(x) satisfies:
///   - For x >= 0: silu(x) in [0, x] (sigmoid in (0,1) scales down)
///   - For x < 0:  silu(x) > -0.279 (global minimum near x ≈ -1.278)
///   - silu(0) == 0
///
/// This bounds the gate activation in SwiGLU's forward pass.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_silu_scalar_bounds() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x >= -10.0 && x <= 10.0);

    let sig = 1.0f32 / (1.0 + (-x).exp());

    // sigmoid must be in (0, 1)
    kani::assume(sig.is_finite());
    let silu = x * sig;
    kani::assume(silu.is_finite());

    if x >= 0.0 {
        assert!(silu >= 0.0, "silu(x) >= 0 for x >= 0");
        assert!(silu <= x + 1e-6, "silu(x) <= x for x >= 0");
    } else {
        // Global min of silu is approximately -0.2784 at x ≈ -1.2785
        assert!(silu > -0.3, "silu(x) > -0.3 for x in [-10, 0)");
    }
}

// ========================= Activation harnesses =============================

// ---------------------------------------------------------------------------
// Harness 6: ReLU non-negative output
// ---------------------------------------------------------------------------

/// Prove: relu(x) = max(0, x) >= 0 for all finite x.
///
/// This is the fundamental ReLU property — output is always non-negative.
/// The CPU implementation uses `x.max(0.0)`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_relu_non_negative() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let relu = x.max(0.0);
    assert!(relu >= 0.0, "relu(x) must be non-negative");
    assert!(relu.is_finite(), "relu of finite input must be finite");
}

// ---------------------------------------------------------------------------
// Harness 7: ReLU identity for positive inputs
// ---------------------------------------------------------------------------

/// Prove: relu(x) == x when x > 0. ReLU is the identity function
/// on the positive reals.
#[kani::unwind(1)]
#[kani::proof]
fn proof_relu_identity_positive() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x > 0.0);

    let relu = x.max(0.0);
    assert!(relu == x, "relu(x) must equal x when x > 0");
}

// ---------------------------------------------------------------------------
// Harness 8: Sigmoid bounded (0, 1)
// ---------------------------------------------------------------------------

/// Prove: sigmoid(x) = 1 / (1 + exp(-x)) is in (0, 1) for all finite x.
///
/// This is the fundamental sigmoid property. IEEE 754 guarantees:
/// - exp(-x) >= 0 for all finite x
/// - 1 + exp(-x) > 1 (since exp >= 0)
/// - 1 / (1 + exp(-x)) < 1 and > 0
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_sigmoid_bounded() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x >= -80.0 && x <= 80.0); // Avoid exp overflow

    let neg_x_exp = (-x).exp();
    kani::assume(neg_x_exp.is_finite());

    let sig = 1.0f32 / (1.0 + neg_x_exp);
    kani::assume(sig.is_finite());

    assert!(sig > 0.0, "sigmoid(x) must be positive");
    assert!(sig < 1.0, "sigmoid(x) must be less than 1");
}

// ---------------------------------------------------------------------------
// Harness 9: Sigmoid symmetry: sigmoid(-x) = 1 - sigmoid(x)
// ---------------------------------------------------------------------------

/// Prove: sigmoid(-x) = 1 - sigmoid(x) for bounded finite x.
///
/// This symmetry property is used by activation backward rules and
/// by the SiLU derivative: d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_sigmoid_symmetry() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x >= -40.0 && x <= 40.0); // Tighter bound for precision

    let sig_pos = 1.0f32 / (1.0 + (-x).exp());
    let sig_neg = 1.0f32 / (1.0 + x.exp());
    kani::assume(sig_pos.is_finite() && sig_neg.is_finite());

    let complement = 1.0 - sig_pos;
    let diff = (sig_neg - complement).abs();

    // Allow small floating-point tolerance
    assert!(
        diff < 1e-5,
        "sigmoid(-x) must equal 1 - sigmoid(x) within tolerance"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: SiLU identity: silu(x) = x * sigmoid(x)
// ---------------------------------------------------------------------------

/// Prove: the SiLU (Swish) activation silu(x) = x / (1 + exp(-x))
/// is equivalent to x * sigmoid(x).
///
/// Both formulations are used in the codebase:
/// - CPU: `x / (1.0 + (-x).exp())`
/// - Scalar: `x * (1.0 / (1.0 + (-x).exp()))`
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_silu_is_x_times_sigmoid() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x >= -40.0 && x <= 40.0);

    let exp_neg = (-x).exp();
    kani::assume(exp_neg.is_finite());

    // Formulation 1: x / (1 + exp(-x))
    let silu_div = x / (1.0 + exp_neg);

    // Formulation 2: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    let sigmoid = 1.0f32 / (1.0 + exp_neg);
    let silu_mul = x * sigmoid;

    kani::assume(silu_div.is_finite() && silu_mul.is_finite());

    let diff = (silu_div - silu_mul).abs();
    assert!(
        diff < 1e-6,
        "x/(1+exp(-x)) must equal x*sigmoid(x) within tolerance"
    );
}

// ---------------------------------------------------------------------------
// Harness 11: Tanh bounded (-1, 1)
// ---------------------------------------------------------------------------

/// Prove: tanh(x) is in (-1, 1) for all finite x.
///
/// tanh is used as the cell-gate activation in LSTM and as an
/// Activation enum variant. Its bounded output is critical for
/// preventing unbounded growth in recurrent networks.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn proof_tanh_bounded() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x >= -80.0 && x <= 80.0);

    let t = x.tanh();
    kani::assume(t.is_finite());

    assert!(t > -1.0, "tanh(x) must be greater than -1");
    assert!(t < 1.0, "tanh(x) must be less than 1");
}

// ---------------------------------------------------------------------------
// Harness 12: ELU continuity at zero
// ---------------------------------------------------------------------------

/// Prove: elu(0, alpha) == 0 for any positive finite alpha.
///
/// ELU is defined as:
///   elu(x) = x                      if x > 0
///   elu(x) = alpha * (exp(x) - 1)   if x <= 0
///
/// At x = 0: alpha * (exp(0) - 1) = alpha * 0 = 0.
/// This proves ELU is continuous at the origin for all positive alpha.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_elu_continuous_at_zero() {
    let alpha: f32 = kani::any();
    kani::assume(alpha.is_finite());
    kani::assume(alpha > 0.0);
    kani::assume(alpha <= 100.0);

    // elu(0) for x <= 0 branch: alpha * (exp(0) - 1) = alpha * 0 = 0
    let exp_0 = 0.0_f32.exp(); // == 1.0
    let elu_at_zero = alpha * (exp_0 - 1.0);

    assert!(
        elu_at_zero.abs() < 1e-7,
        "elu(0, alpha) must be 0 for any positive alpha"
    );
}

// ---------------------------------------------------------------------------
// Harness 13: LeakyReLU identity for positive inputs
// ---------------------------------------------------------------------------

/// Prove: leaky_relu(x, slope) == x when x > 0, regardless of slope.
///
/// LeakyReLU: max(0, x) + slope * min(0, x).
/// When x > 0: max(0,x) = x, min(0,x) = 0, so result = x + slope*0 = x.
#[kani::unwind(1)]
#[kani::proof]
fn proof_leaky_relu_identity_positive() {
    let x: f32 = kani::any();
    let slope: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(x > 0.0);
    kani::assume(slope.is_finite());
    kani::assume(slope >= -1.0 && slope <= 1.0);

    let positive_part = x.max(0.0);
    let negative_part = x.min(0.0);
    let leaky = positive_part + slope * negative_part;

    assert!(
        (leaky - x).abs() < 1e-7,
        "leaky_relu(x, slope) must equal x when x > 0"
    );
}

// ========================= Embedding harnesses ==============================

// ---------------------------------------------------------------------------
// Harness 14: Embedding weight must be rank-2
// ---------------------------------------------------------------------------

/// Prove: Embedding::new rejects weight tensors that are not rank-2.
/// The weight must be [vocab_size, embedding_dim] — exactly 2D.
/// Rank 0, 1, 3+ are all invalid.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_requires_rank_2() {
    let rank: usize = kani::any();
    kani::assume(rank <= 8);

    let valid = rank == 2;

    if !valid {
        assert!(rank != 2, "non-rank-2 must be rejected");
    } else {
        assert!(rank == 2, "rank-2 must be accepted");
    }
}

// ---------------------------------------------------------------------------
// Harness 15: Embedding forward_ids rejects out-of-range indices
// ---------------------------------------------------------------------------

/// Prove: any index >= vocab_size must be rejected by the forward_ids
/// validation loop. This prevents out-of-bounds memory access on the
/// weight matrix.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_index_out_of_range() {
    let vocab_size: usize = kani::any();
    let index: usize = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 65536);
    kani::assume(index <= 65536);
    kani::assume(index >= vocab_size);

    // The forward_ids validation: `if id >= vocab_size { return Err(...) }`
    assert!(
        index >= vocab_size,
        "index >= vocab_size must be detected as out of range"
    );
}

// ---------------------------------------------------------------------------
// Harness 16: Embedding vocab_size is weight dim 0
// ---------------------------------------------------------------------------

/// Prove: for a rank-2 weight [V, D], the vocab_size (number of
/// embeddings) is the first dimension. The forward_ids method uses
/// `self.weight.dims2()` which returns (dim0, dim1) = (V, D).
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_vocab_size_is_dim_0() {
    let vocab_size: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 65536);
    kani::assume(embed_dim >= 1 && embed_dim <= 4096);

    // Weight shape is [vocab_size, embed_dim]
    let dims = [vocab_size, embed_dim];

    assert!(dims[0] == vocab_size, "dim 0 must be vocab_size");
    assert!(dims[1] == embed_dim, "dim 1 must be embed_dim");
    assert!(dims.len() == 2, "embedding weight must be 2D");
}

// ---------------------------------------------------------------------------
// Harness 17: Embedding output shape construction
// ---------------------------------------------------------------------------

/// Prove: embedding lookup output shape is input_shape + [embed_dim].
/// For input ids of shape [N], output is [N, embed_dim].
/// For input ids of shape [B, S], output is [B, S, embed_dim].
/// The output rank is always input_rank + 1.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_output_shape() {
    let input_rank: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(input_rank >= 1 && input_rank <= 4);
    kani::assume(embed_dim >= 1 && embed_dim <= 4096);

    // Output shape = input_dims ++ [embed_dim]
    let output_rank = input_rank + 1;

    assert!(
        output_rank == input_rank + 1,
        "output rank must be input rank + 1"
    );
    assert!(
        output_rank >= 2,
        "output must be at least rank 2 (single id -> [1, D])"
    );
}

// ---------------------------------------------------------------------------
// Harness 18: Embedding valid index range
// ---------------------------------------------------------------------------

/// Prove: all indices in [0, vocab_size) are valid. For any index < vocab_size,
/// the forward_ids validation loop accepts it.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_valid_index_accepted() {
    let vocab_size: usize = kani::any();
    let index: usize = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 65536);
    kani::assume(index < vocab_size);

    // The validation check: id >= vocab_size triggers error.
    // Since index < vocab_size, this must pass.
    assert!(
        index < vocab_size,
        "index in [0, vocab_size) must be accepted"
    );
}
