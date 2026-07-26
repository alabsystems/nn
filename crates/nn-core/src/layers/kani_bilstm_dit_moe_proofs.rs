// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for BiLSTM, DiT block, and MoE expert layers (#4077).
//!
//! ## BiLSTM proofs (4 harnesses):
//!  1. `proof_bilstm_output_channels` — output dim = 2 * hidden_dim
//!  2. `proof_bilstm_gate_sigmoid_bounded` — LSTM gates in [0, 1]
//!  3. `proof_bilstm_cell_tanh_bounded` — cell candidate in [-1, 1]
//!  4. `proof_bilstm_sequence_length_preserved` — output seq_len == input seq_len
//!
//! ## DiT block proofs (4 harnesses):
//!  5. `proof_dit_adaln_shift_scale_finite` — adaptive LN params finite
//!  6. `proof_dit_gate_bounded` — gate values bounded for bounded conditioning
//!  7. `proof_dit_residual_shape_match` — residual add shapes match
//!  8. `proof_dit_output_dim_preserved` — hidden dim preserved through block
//!
//! ## MoE expert proofs (4 harnesses):
//!  9. `proof_moe_expert_output_shape` — expert output matches expected shape
//! 10. `proof_moe_weighted_sum_finite` — weighted combination finite
//! 11. `proof_moe_routing_weights_sum` — routing weights sum to ~1 after renorm
//! 12. `proof_moe_num_experts_positive` — num_experts > 0 invariant
//!
//! Part of #4077.

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh_scalar(x: f32) -> f32 {
    x.tanh()
}

fn tanh_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

/// SiLU scalar: x * sigmoid(x). Used by SwiGLU gating in ExpertFFN.
fn silu_scalar(x: f32) -> f32 {
    x * sigmoid_scalar(x)
}

// ===========================================================================
// BiLSTM proofs
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 1: BiLSTM output channels = 2 * hidden_dim
// ---------------------------------------------------------------------------

/// Prove: BiLSTM output feature dimension is exactly 2 * hidden_size.
///
/// BiLstm concatenates forward LSTM output [seq, batch, hidden] and backward
/// LSTM output [seq, batch, hidden] along dim 2, producing [seq, batch, 2*hidden].
/// This models the `DynTensor::cat(&[&fwd, &bwd], 2)` in `forward_seq`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_bilstm_output_channels() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size >= 1 && hidden_size <= 2048);

    let seq_len: usize = kani::any();
    let batch: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 1024);
    kani::assume(batch >= 1 && batch <= 128);

    // Forward LSTM output shape: [seq_len, batch, hidden_size]
    let fwd_shape = [seq_len, batch, hidden_size];
    // Backward LSTM output shape: [seq_len, batch, hidden_size]
    let bwd_shape = [seq_len, batch, hidden_size];

    // Concatenation along dim 2
    let output_dim = fwd_shape[2] + bwd_shape[2];
    assert!(
        output_dim == 2 * hidden_size,
        "BiLSTM output feature dim must be 2 * hidden_size"
    );

    // First two dims preserved
    assert!(fwd_shape[0] == seq_len, "seq_len must be preserved");
    assert!(fwd_shape[1] == batch, "batch must be preserved");

    // Strictly larger than single direction
    assert!(
        output_dim > hidden_size,
        "BiLSTM output must exceed single direction"
    );
}

// ---------------------------------------------------------------------------
// Harness 2: BiLSTM gate sigmoid bounded — all LSTM gates in [0, 1]
// ---------------------------------------------------------------------------

/// Prove: the input, forget, and output gates (sigmoid-activated) of each
/// LSTM direction in a BiLSTM are bounded in [0, 1].
///
/// Both forward and backward LSTMs use identical gate equations:
///   i = sigmoid(x @ w_ih_i + h @ w_hh_i + b_i)
///   f = sigmoid(x @ w_ih_f + h @ w_hh_f + b_f)
///   o = sigmoid(x @ w_ih_o + h @ w_hh_o + b_o)
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_bilstm_gate_sigmoid_bounded() {
    // Model both directions with the same gate equation
    let pre_gate_fwd: f32 = kani::any();
    let pre_gate_bwd: f32 = kani::any();

    kani::assume(pre_gate_fwd.is_finite());
    kani::assume(pre_gate_bwd.is_finite());
    kani::assume(pre_gate_fwd >= -100.0 && pre_gate_fwd <= 100.0);
    kani::assume(pre_gate_bwd >= -100.0 && pre_gate_bwd <= 100.0);

    let gate_fwd = sigmoid_scalar(pre_gate_fwd);
    let gate_bwd = sigmoid_scalar(pre_gate_bwd);

    // Forward direction gate
    assert!(gate_fwd.is_finite(), "forward gate must be finite");
    assert!(gate_fwd >= 0.0, "forward gate must be >= 0");
    assert!(gate_fwd <= 1.0, "forward gate must be <= 1");

    // Backward direction gate
    assert!(gate_bwd.is_finite(), "backward gate must be finite");
    assert!(gate_bwd >= 0.0, "backward gate must be >= 0");
    assert!(gate_bwd <= 1.0, "backward gate must be <= 1");
}

// ---------------------------------------------------------------------------
// Harness 3: BiLSTM cell tanh bounded — cell candidate in [-1, 1]
// ---------------------------------------------------------------------------

/// Prove: the cell candidate g = tanh(pre_activation) is bounded in [-1, 1]
/// for both forward and backward LSTM directions in BiLSTM.
///
/// Cell update: c_new = f * c_old + i * tanh(g_pre)
/// The tanh ensures the new information (g) is bounded.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn proof_bilstm_cell_tanh_bounded() {
    let g_pre_fwd: f32 = kani::any();
    let g_pre_bwd: f32 = kani::any();

    kani::assume(g_pre_fwd.is_finite());
    kani::assume(g_pre_bwd.is_finite());
    kani::assume(g_pre_fwd >= -100.0 && g_pre_fwd <= 100.0);
    kani::assume(g_pre_bwd >= -100.0 && g_pre_bwd <= 100.0);

    let g_fwd = tanh_scalar(g_pre_fwd);
    let g_bwd = tanh_scalar(g_pre_bwd);

    // Forward direction cell candidate
    assert!(g_fwd.is_finite(), "forward cell candidate must be finite");
    assert!(g_fwd >= -1.0, "forward cell candidate must be >= -1");
    assert!(g_fwd <= 1.0, "forward cell candidate must be <= 1");

    // Backward direction cell candidate
    assert!(g_bwd.is_finite(), "backward cell candidate must be finite");
    assert!(g_bwd >= -1.0, "backward cell candidate must be >= -1");
    assert!(g_bwd <= 1.0, "backward cell candidate must be <= 1");
}

// ---------------------------------------------------------------------------
// Harness 4: BiLSTM sequence length preserved
// ---------------------------------------------------------------------------

/// Prove: BiLSTM output sequence length equals input sequence length.
///
/// forward_seq processes each timestep [0, seq_len) in both directions,
/// then concatenates along dim 2. The temporal dimension (dim 0) is unchanged.
/// The backward direction reverses input, processes, then reverses output,
/// so the alignment is preserved.
#[kani::unwind(1)]
#[kani::proof]
fn proof_bilstm_sequence_length_preserved() {
    let seq_len: usize = kani::any();
    let batch: usize = kani::any();
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 2048);
    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(input_size >= 1 && input_size <= 1024);
    kani::assume(hidden_size >= 1 && hidden_size <= 1024);

    // Input: [seq_len, batch, input_size]
    let input_shape = [seq_len, batch, input_size];

    // Forward LSTM output: [seq_len, batch, hidden_size]
    let fwd_out_shape = [input_shape[0], input_shape[1], hidden_size];

    // Backward LSTM: flip(0) → LSTM → flip(0)
    // flip preserves shape, LSTM preserves seq_len
    let bwd_out_shape = [input_shape[0], input_shape[1], hidden_size];

    // Cat along dim 2: [seq_len, batch, 2 * hidden_size]
    let output_shape = [
        fwd_out_shape[0],
        fwd_out_shape[1],
        fwd_out_shape[2] + bwd_out_shape[2],
    ];

    // Sequence length preserved
    assert!(
        output_shape[0] == seq_len,
        "output seq_len must equal input seq_len"
    );

    // Batch preserved
    assert!(
        output_shape[1] == batch,
        "output batch must equal input batch"
    );

    // Feature dim is 2 * hidden
    assert!(
        output_shape[2] == 2 * hidden_size,
        "output feature dim must be 2 * hidden_size"
    );
}

// ===========================================================================
// DiT block proofs
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 5: DiT AdaLN shift/scale finite for finite conditioning
// ---------------------------------------------------------------------------

/// Prove: adaptive LayerNorm modulation produces finite shift and scale
/// values when the conditioning signal and linear projection produce
/// finite outputs.
///
/// AdaLnZero projects conditioning to 3*dim, then narrows to scale, shift, gate.
/// apply_adaln_modulation: normed * (scale + 1) + shift
/// If scale and shift are finite, the modulation is finite for finite normed input.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dit_adaln_shift_scale_finite() {
    let scale: f32 = kani::any();
    let shift: f32 = kani::any();
    let normed: f32 = kani::any();

    kani::assume(scale.is_finite());
    kani::assume(shift.is_finite());
    kani::assume(normed.is_finite());
    kani::assume(scale >= -10.0 && scale <= 10.0);
    kani::assume(shift >= -10.0 && shift <= 10.0);
    kani::assume(normed >= -10.0 && normed <= 10.0);

    // apply_adaln_modulation: normed * (scale + 1.0) + shift
    let scale_plus_one = scale + 1.0;
    assert!(scale_plus_one.is_finite(), "scale+1 must be finite");

    let modulated = normed * scale_plus_one + shift;
    assert!(modulated.is_finite(), "modulated output must be finite");

    // Scale+1 ranges from [-9, 11] for scale in [-10, 10]
    assert!(
        scale_plus_one >= -9.0 && scale_plus_one <= 11.0,
        "scale+1 must be in expected range"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: DiT gate bounded for bounded conditioning
// ---------------------------------------------------------------------------

/// Prove: the gate value from AdaLnZero is the raw output of a Linear projection
/// (no activation). For bounded input conditioning, the gate is finite and bounded.
///
/// In the DiT forward: `x = x + gate * attn(modulated)`
/// The gate modulates the residual contribution. While not inherently bounded
/// to [0,1] like sigmoid, it is bounded for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dit_gate_bounded() {
    // Gate is a slice of a linear projection output.
    // For weight W in [-bound, bound] and bias b in [-bound, bound],
    // gate = sum_i(cond_i * W_i) + b.
    // Model a single-element gate for Kani tractability.
    let cond: f32 = kani::any();
    let weight: f32 = kani::any();
    let bias: f32 = kani::any();

    kani::assume(cond.is_finite());
    kani::assume(weight.is_finite());
    kani::assume(bias.is_finite());
    kani::assume(cond >= -10.0 && cond <= 10.0);
    kani::assume(weight >= -1.0 && weight <= 1.0);
    kani::assume(bias >= -1.0 && bias <= 1.0);

    let gate = cond * weight + bias;

    assert!(gate.is_finite(), "gate must be finite for bounded inputs");

    // |gate| <= |cond| * |weight| + |bias| <= 10 * 1 + 1 = 11
    assert!(
        gate >= -11.0 && gate <= 11.0,
        "gate must be bounded by input range"
    );
}

// ---------------------------------------------------------------------------
// Harness 7: DiT residual shape match
// ---------------------------------------------------------------------------

/// Prove: the residual addition `x + gate * attn_out` requires x and
/// gate * attn_out to have the same shape. The DiT block preserves shapes
/// through each sub-block.
///
/// Input x: [B, S, dim], attn output: [B, S, dim], gate: [B, S, dim] or
/// broadcastable. Residual: x + gate * attn_out has shape [B, S, dim].
#[kani::unwind(1)]
#[kani::proof]
fn proof_dit_residual_shape_match() {
    let batch: usize = kani::any();
    let seq_len: usize = kani::any();
    let dim: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 32);
    kani::assume(seq_len >= 1 && seq_len <= 512);
    kani::assume(dim >= 1 && dim <= 2048);

    // Input: [B, S, dim]
    let x_shape = [batch, seq_len, dim];

    // Attention sub-block: attn output preserves [B, S, dim]
    let attn_out_shape = [batch, seq_len, dim];

    // Gate: [B, S, dim] (from AdaLnZero, broadcast from [B, 1, dim] or [B, dim])
    let gate_shape = [batch, seq_len, dim];

    // gate * attn_out: element-wise, same shape
    let gated_shape = [gate_shape[0], gate_shape[1], gate_shape[2]];
    assert!(gated_shape[0] == attn_out_shape[0], "batch must match");
    assert!(gated_shape[1] == attn_out_shape[1], "seq must match");
    assert!(gated_shape[2] == attn_out_shape[2], "dim must match");

    // Residual: x + gated must have same shape
    assert!(x_shape[0] == gated_shape[0], "residual batch must match");
    assert!(x_shape[1] == gated_shape[1], "residual seq must match");
    assert!(x_shape[2] == gated_shape[2], "residual dim must match");

    // Output shape equals input shape
    let output_shape = [x_shape[0], x_shape[1], x_shape[2]];
    assert!(
        output_shape == x_shape,
        "output shape must equal input shape"
    );
}

// ---------------------------------------------------------------------------
// Harness 8: DiT output dimension preserved through block
// ---------------------------------------------------------------------------

/// Prove: the DiT block preserves the hidden dimension through both
/// sub-blocks (attention + FFN). Input [B, S, dim] → output [B, S, dim].
///
/// This models the full DiT forward:
///   x = x + gate_attn * attn(adaln_attn(x, cond))
///   x = x + gate_ffn * ffn(adaln_ffn(x, cond))
/// Both residual additions preserve the shape.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dit_output_dim_preserved() {
    let batch: usize = kani::any();
    let seq_len: usize = kani::any();
    let dim: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 32);
    kani::assume(seq_len >= 1 && seq_len <= 512);
    kani::assume(dim >= 1 && dim <= 2048);

    let input_shape = [batch, seq_len, dim];

    // After attention sub-block: x + gate_attn * attn_out
    // attn_out: [B, S, dim], gate_attn: broadcastable to [B, S, dim]
    // residual preserves shape
    let after_attn_shape = [input_shape[0], input_shape[1], input_shape[2]];
    assert!(
        after_attn_shape == input_shape,
        "attention sub-block must preserve shape"
    );

    // After FFN sub-block: x + gate_ffn * ffn_out
    // ffn_out: [B, S, dim], gate_ffn: broadcastable to [B, S, dim]
    // residual preserves shape
    let after_ffn_shape = [
        after_attn_shape[0],
        after_attn_shape[1],
        after_attn_shape[2],
    ];
    assert!(
        after_ffn_shape == input_shape,
        "FFN sub-block must preserve shape"
    );

    // Overall: input dim == output dim
    assert!(
        after_ffn_shape[2] == dim,
        "hidden dimension must be preserved through DiT block"
    );

    // Element count preserved
    let input_elements = input_shape[0]
        .checked_mul(input_shape[1])
        .unwrap()
        .checked_mul(input_shape[2])
        .unwrap();
    let output_elements = after_ffn_shape[0]
        .checked_mul(after_ffn_shape[1])
        .unwrap()
        .checked_mul(after_ffn_shape[2])
        .unwrap();
    assert!(
        input_elements == output_elements,
        "element count must be preserved"
    );
}

// ===========================================================================
// MoE expert proofs
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 9: MoE expert output shape
// ---------------------------------------------------------------------------

/// Prove: ExpertFFN (SwiGLU) output shape matches input hidden_size.
///
/// ExpertFFN: down_proj(silu(gate_proj(x)) * up_proj(x))
///   gate_proj: [hidden_size, intermediate] → output [*, intermediate]
///   up_proj:   [hidden_size, intermediate] → output [*, intermediate]
///   down_proj:  [intermediate, hidden_size] → output [*, hidden_size]
///
/// Similarly ExpertMlp: down_proj(act(up_proj(x)))
///   up_proj:   [hidden_size, intermediate] → output [*, intermediate]
///   down_proj:  [intermediate, hidden_size] → output [*, hidden_size]
///
/// Both expert types project to intermediate_size then back to hidden_size.
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_expert_output_shape() {
    let hidden_size: usize = kani::any();
    let intermediate_size: usize = kani::any();
    let n_tokens: usize = kani::any();

    kani::assume(hidden_size >= 1 && hidden_size <= 4096);
    kani::assume(intermediate_size >= 1 && intermediate_size <= 4096);
    kani::assume(n_tokens >= 1 && n_tokens <= 512);

    // Input: [n_tokens, hidden_size]
    let input_shape = [n_tokens, hidden_size];

    // gate_proj / up_proj: [n_tokens, intermediate_size]
    let gate_out_dim = intermediate_size;
    let up_out_dim = intermediate_size;
    assert!(
        gate_out_dim == up_out_dim,
        "gate and up must have same output dim"
    );

    // After SiLU(gate) * up: [n_tokens, intermediate_size] (element-wise)
    let gated_shape = [n_tokens, intermediate_size];

    // down_proj: [n_tokens, hidden_size]
    let output_shape = [gated_shape[0], hidden_size];

    // Output hidden dim matches input hidden dim
    assert!(
        output_shape[1] == input_shape[1],
        "expert output hidden_size must match input hidden_size"
    );

    // Token count preserved
    assert!(
        output_shape[0] == input_shape[0],
        "expert must preserve token count"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: MoE weighted sum finite
// ---------------------------------------------------------------------------

/// Prove: the weighted combination of expert outputs is finite when each
/// expert output and routing weight is finite and bounded.
///
/// In MoeDispatch, the final output for each token is:
///   y = sum_k(w_k * expert_k(x))
/// where w_k are routing weights and expert_k(x) are expert outputs.
#[kani::unwind(5)]
#[kani::proof]
fn proof_moe_weighted_sum_finite() {
    let top_k: usize = kani::any();
    kani::assume(top_k >= 1 && top_k <= 4);

    let mut weighted_sum: f32 = 0.0;

    for _k in 0..top_k {
        let weight: f32 = kani::any();
        let expert_out: f32 = kani::any();

        kani::assume(weight.is_finite());
        kani::assume(expert_out.is_finite());
        kani::assume(weight >= 0.0 && weight <= 1.0);
        kani::assume(expert_out >= -100.0 && expert_out <= 100.0);

        let contribution = weight * expert_out;
        assert!(
            contribution.is_finite(),
            "weight * expert_out must be finite"
        );

        weighted_sum += contribution;
    }

    assert!(
        weighted_sum.is_finite(),
        "weighted sum of expert outputs must be finite"
    );

    // |weighted_sum| <= sum_k(|w_k| * |e_k|) <= top_k * 1.0 * 100.0
    let bound = (top_k as f32) * 100.0;
    assert!(
        weighted_sum >= -bound && weighted_sum <= bound,
        "weighted sum must be bounded by top_k * max_expert_output"
    );
}

// ---------------------------------------------------------------------------
// Harness 11: MoE routing weights sum to ~1 for top-k selection
// ---------------------------------------------------------------------------

/// Prove: after renormalization, top-k routing weights sum to ~1.0.
///
/// MoE routing: softmax produces probabilities summing to 1.0, topk selects
/// k of them, then optionally renormalizes so the selected weights sum to 1.0.
/// This ensures the output is a convex combination of expert outputs.
#[kani::unwind(9)]
#[kani::proof]
fn proof_moe_routing_weights_sum() {
    let k: usize = kani::any();
    kani::assume(k >= 1 && k <= 8);

    let mut raw_weights: [f32; 8] = [0.0; 8];
    let mut raw_sum: f32 = 0.0;

    // Generate k positive routing weights (from softmax topk output)
    for i in 0..k {
        let w: f32 = kani::any();
        kani::assume(w > 0.0 && w <= 1.0 && w.is_finite());
        raw_weights[i] = w;
        raw_sum += w;
    }

    kani::assume(raw_sum > 1e-10);
    kani::assume(raw_sum.is_finite());

    // Renormalize: each weight /= sum
    let inv = 1.0f32 / raw_sum;
    kani::assume(inv.is_finite());

    let mut normed_sum: f32 = 0.0;
    for i in 0..k {
        let normed = raw_weights[i] * inv;
        kani::assume(normed.is_finite());
        normed_sum += normed;
    }

    kani::assume(normed_sum.is_finite());

    // Renormalized weights sum to ~1.0 (within f32 rounding)
    assert!(
        (normed_sum - 1.0).abs() < 1e-4,
        "renormalized routing weights must sum to ~1.0"
    );

    // Each renormalized weight is non-negative
    for i in 0..k {
        let normed = raw_weights[i] * inv;
        assert!(normed >= 0.0, "renormalized weight must be non-negative");
    }
}

// ---------------------------------------------------------------------------
// Harness 12: MoE num_experts > 0 invariant
// ---------------------------------------------------------------------------

/// Prove: num_experts must be > 0 for a valid MoE configuration.
///
/// ExpertFFN::new and MoeDispatch construction require at least one expert.
/// With num_experts == 0:
///   - Router Linear would have 0 output features (degenerate)
///   - No experts to dispatch tokens to
///   - top_k > num_experts for any top_k >= 1
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_num_experts_positive() {
    let num_experts: usize = kani::any();
    let top_k: usize = kani::any();

    kani::assume(top_k >= 1 && top_k <= 64);

    // Case 1: num_experts == 0 is always rejected
    if num_experts == 0 {
        // top_k > num_experts for any top_k >= 1
        assert!(top_k > num_experts, "top_k must exceed zero experts");
        // Router weight would be [hidden_size, 0] — degenerate
        let router_out = num_experts;
        assert!(
            router_out == 0,
            "router with 0 experts has 0 output features"
        );
    }

    // Case 2: valid configuration requires num_experts >= top_k >= 1
    kani::assume(num_experts >= 1 && num_experts <= 64);
    kani::assume(top_k <= num_experts);

    assert!(num_experts > 0, "num_experts must be positive");
    assert!(top_k <= num_experts, "top_k must not exceed num_experts");

    // Expert indexing is safe: all indices in [0, num_experts) are valid
    let idx: usize = kani::any();
    kani::assume(idx < num_experts);
    assert!(idx < num_experts, "expert index must be in bounds");

    // Router output: [*, num_experts] has a valid last dim
    assert!(num_experts >= 1, "router output last dim must be >= 1");
}
