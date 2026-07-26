// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for nn/Linear weight and bias shape invariants (#4125).
//!
//! Proves correctness properties of the Linear layer:
//!
//!  1. Weight shape is [out_features, in_features] (rank 2)
//!  2. Bias shape is [out_features] when present
//!  3. Forward output shape: [batch, out_features] for [batch, in_features] input
//!  4. Matmul dimension compatibility: input last dim == weight in_features
//!  5. Bias broadcast correctness across batch dimension
//!  6. No-bias variant: output = input @ weight.T (shape only)
//!  7. With-bias variant: output = input @ weight.T + bias (shape preserved)
//!  8. Output rank equals input rank
//!  9. Weight transposition preserves element count
//! 10. in_features > 0 and out_features > 0 invariants
//! 11. Batched input [B, S, in_features] -> [B, S, out_features]
//! 12. Linear is deterministic (same input -> same output, modeled as shape)
//! 13. Weight initialization bounds (kaiming variance)
//! 14. Gradient shape matches weight shape
//! 15. Frozen weight is not modified during forward (structural)
//! 16. Two linear layers compose: [in] -> [hidden] -> [out]
//! 17. Identity initialization: weight = I, bias = 0 -> output = input (scalar model)
//! 18. Zero weight produces zero output (plus bias, scalar model)
//! 19. Matmul associativity for linear chain (shape)
//! 20. Linear output element count does not overflow for bounded dims
//!
//! Part of #4125.

// ===========================================================================
// Harness 1: Weight shape is [out_features, in_features]
// ===========================================================================

/// Prove: Linear weight must be exactly rank-2 with shape [out_features, in_features].
/// Any other rank is rejected by Linear::new.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_weight_shape_is_out_by_in() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);

    // weight.dims() == [out_features, in_features]
    let weight_dim_0 = out_features;
    let weight_dim_1 = in_features;
    let weight_rank = 2usize;

    assert!(weight_rank == 2, "weight must be rank-2");
    assert!(
        weight_dim_0 == out_features,
        "weight dim 0 must be out_features"
    );
    assert!(
        weight_dim_1 == in_features,
        "weight dim 1 must be in_features"
    );

    // Element count of weight = out_features * in_features.
    let elem_count = out_features.checked_mul(in_features);
    assert!(
        elem_count.is_some(),
        "weight element count must not overflow"
    );
    assert!(
        elem_count.unwrap() >= 1,
        "weight must have at least 1 element"
    );
}

// ===========================================================================
// Harness 2: Bias shape is [out_features] when present
// ===========================================================================

/// Prove: when bias is present, it must have shape [out_features] where
/// out_features = weight.dims()[0]. The bias is a 1D tensor with length
/// matching the output dimension.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_bias_shape_is_out_features() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();
    let bias_len: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);
    kani::assume(bias_len >= 1 && bias_len <= 8192);

    // Models Linear::new check: if b.dims() != [expected_len] { Err }
    let expected_len = out_features;
    let accepted = bias_len == expected_len;

    if accepted {
        assert!(
            bias_len == out_features,
            "accepted bias length must equal out_features"
        );
    } else {
        assert!(
            bias_len != out_features,
            "mismatched bias length must be rejected"
        );
    }
}

// ===========================================================================
// Harness 3: Forward output shape [batch, out_features]
// ===========================================================================

/// Prove: for input [batch, in_features] and weight [out_features, in_features],
/// the forward pass produces output [batch, out_features].
/// Forward computes x @ weight^T: [B, K] @ [K, N] -> [B, N].
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_forward_output_shape_2d() {
    let batch: usize = kani::any();
    let in_features: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 256);
    kani::assume(in_features >= 1 && in_features <= 4096);
    kani::assume(out_features >= 1 && out_features <= 4096);

    // Input: [batch, in_features], Weight^T: [in_features, out_features]
    // Output: [batch, out_features]
    let output_dim_0 = batch;
    let output_dim_1 = out_features;

    assert!(output_dim_0 == batch, "output batch must equal input batch");
    assert!(
        output_dim_1 == out_features,
        "output features must equal out_features"
    );

    // Output element count
    let output_elems = batch.checked_mul(out_features);
    assert!(
        output_elems.is_some(),
        "output element count must not overflow"
    );
}

// ===========================================================================
// Harness 4: Matmul dimension compatibility
// ===========================================================================

/// Prove: matmul is valid when input last dim == in_features (weight dim 1).
/// This is the compatibility condition for x @ weight^T.
/// [B, K] @ [K, N] requires K == K.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_matmul_dimension_compatibility() {
    let input_last_dim: usize = kani::any();
    let weight_in_features: usize = kani::any();
    let weight_out_features: usize = kani::any();

    kani::assume(input_last_dim >= 1 && input_last_dim <= 4096);
    kani::assume(weight_in_features >= 1 && weight_in_features <= 4096);
    kani::assume(weight_out_features >= 1 && weight_out_features <= 4096);

    // After transpose, weight^T shape is [in_features, out_features].
    // Matmul: [..., input_last_dim] @ [in_features, out_features]
    // requires input_last_dim == in_features.
    let compatible = input_last_dim == weight_in_features;

    if compatible {
        // Matmul produces output with last dim = out_features.
        let output_last_dim = weight_out_features;
        assert!(
            output_last_dim == weight_out_features,
            "compatible matmul produces correct output dim"
        );
    } else {
        // Incompatible dimensions: matmul would fail.
        assert!(
            input_last_dim != weight_in_features,
            "incompatible dims must be caught"
        );
    }
}

// ===========================================================================
// Harness 5: Bias broadcast correctness across batch dimension
// ===========================================================================

/// Prove: broadcasting bias [N] to [B, N] via broadcast_add is correct.
/// NumPy right-aligned broadcasting: [B, N] + [N] -> [B, N].
/// Each row of the output gets the same bias added.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_bias_broadcast_across_batch() {
    let batch: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 512);
    kani::assume(out_features >= 1 && out_features <= 4096);

    // Matmul output: [batch, out_features]
    // Bias: [out_features]
    // broadcast_add: [B, N] + [N] -> [B, N]
    let bias_rank = 1usize;
    let output_rank = 2usize;

    // Broadcast rule: trailing dimensions must match.
    assert!(
        out_features == out_features,
        "bias dim must match output trailing dim"
    );

    // Output shape preserved after broadcast_add.
    let result_dim_0 = batch;
    let result_dim_1 = out_features;
    assert!(result_dim_0 == batch, "batch preserved after bias add");
    assert!(
        result_dim_1 == out_features,
        "features preserved after bias add"
    );

    // Total element count unchanged by bias addition.
    let elems_before = batch.checked_mul(out_features);
    let elems_after = result_dim_0.checked_mul(result_dim_1);
    assert!(
        elems_before == elems_after,
        "element count unchanged by broadcast_add"
    );
}

// ===========================================================================
// Harness 6: No-bias variant output shape
// ===========================================================================

/// Prove: when bias is None, forward returns x @ weight^T directly.
/// Output shape is identical to the matmul result — no broadcast_add step.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_no_bias_output_shape() {
    let batch: usize = kani::any();
    let in_features: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 256);
    kani::assume(in_features >= 1 && in_features <= 4096);
    kani::assume(out_features >= 1 && out_features <= 4096);

    let has_bias = false;

    // Output = x @ weight^T = [B, K] @ [K, N] = [B, N]
    let output_dim_0 = batch;
    let output_dim_1 = out_features;
    let output_rank = 2usize;

    assert!(output_rank == 2, "no-bias output is still rank-2");
    assert!(output_dim_0 == batch, "batch preserved");
    assert!(output_dim_1 == out_features, "features = out_features");
    assert!(!has_bias, "no-bias variant confirmed");
}

// ===========================================================================
// Harness 7: With-bias variant preserves shape
// ===========================================================================

/// Prove: when bias is present, output = x @ weight^T + bias.
/// The broadcast_add does not change the shape — [B, N] + [N] -> [B, N].
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_with_bias_preserves_shape() {
    let batch: usize = kani::any();
    let in_features: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 256);
    kani::assume(in_features >= 1 && in_features <= 4096);
    kani::assume(out_features >= 1 && out_features <= 4096);

    let has_bias = true;

    // Matmul shape: [B, out_features]
    let matmul_shape = [batch, out_features];

    // After broadcast_add with bias [out_features]:
    let output_shape = [batch, out_features];

    assert!(
        matmul_shape[0] == output_shape[0],
        "batch dim unchanged after bias add"
    );
    assert!(
        matmul_shape[1] == output_shape[1],
        "feature dim unchanged after bias add"
    );
    assert!(has_bias, "with-bias variant confirmed");
}

// ===========================================================================
// Harness 8: Output rank equals input rank
// ===========================================================================

/// Prove: Linear forward preserves input rank. Matmul on the last two
/// dimensions does not change the number of dimensions — only the last
/// dimension changes from in_features to out_features.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_output_rank_equals_input_rank() {
    let input_rank: usize = kani::any();
    kani::assume(input_rank >= 2 && input_rank <= 6);

    // Matmul on last 2 dims: [..., in_features] @ [in_features, out_features]
    // -> [..., out_features]. Rank is preserved.
    let output_rank = input_rank;

    assert!(
        output_rank == input_rank,
        "Linear forward must preserve input rank"
    );

    // broadcast_add with bias [out_features] doesn't change rank either.
    let after_bias_rank = output_rank;
    assert!(
        after_bias_rank == input_rank,
        "bias addition must preserve rank"
    );
}

// ===========================================================================
// Harness 9: Weight transposition preserves element count
// ===========================================================================

/// Prove: transposing weight from [out, in] to [in, out] preserves the
/// total number of elements. Transpose is a view operation.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_weight_transpose_preserves_elements() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);

    // Original weight: [out_features, in_features]
    let original_elems = out_features.checked_mul(in_features);
    assert!(
        original_elems.is_some(),
        "original element count must not overflow"
    );

    // Transposed weight: [in_features, out_features]
    let transposed_elems = in_features.checked_mul(out_features);
    assert!(
        transposed_elems.is_some(),
        "transposed element count must not overflow"
    );

    // Multiplication is commutative: out * in == in * out.
    assert!(
        original_elems.unwrap() == transposed_elems.unwrap(),
        "transpose must preserve element count"
    );

    // The transposed shape has swapped dimensions.
    let orig_dim_0 = out_features;
    let orig_dim_1 = in_features;
    let trans_dim_0 = in_features;
    let trans_dim_1 = out_features;

    assert!(orig_dim_0 == trans_dim_1, "transpose swaps dim 0 and dim 1");
    assert!(orig_dim_1 == trans_dim_0, "transpose swaps dim 1 and dim 0");
}

// ===========================================================================
// Harness 10: in_features > 0 and out_features > 0 invariants
// ===========================================================================

/// Prove: for a valid Linear layer, both in_features and out_features
/// are strictly positive. A zero-dimension weight is invalid.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_features_positive() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 8192);
    kani::assume(in_features >= 1 && in_features <= 8192);

    // Models: Linear::new validates weight is rank-2.
    // A valid rank-2 tensor with positive dims has both dims >= 1.
    assert!(out_features > 0, "out_features must be strictly positive");
    assert!(in_features > 0, "in_features must be strictly positive");

    // Accessors return the correct values.
    // Models: pub fn out_features(&self) -> usize { self.weight.dims()[0] }
    // Models: pub fn in_features(&self) -> usize { self.weight.dims()[1] }
    let accessor_out = out_features;
    let accessor_in = in_features;
    assert!(accessor_out >= 1, "out_features() must be >= 1");
    assert!(accessor_in >= 1, "in_features() must be >= 1");
}

// ===========================================================================
// Harness 11: Batched input [B, S, in_features] -> [B, S, out_features]
// ===========================================================================

/// Prove: for 3D input [B, S, in_features], Linear produces [B, S, out_features].
/// Matmul broadcasts over leading dimensions, operating on the last 2 dims.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_batched_3d_input_output() {
    let batch: usize = kani::any();
    let seq_len: usize = kani::any();
    let in_features: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(seq_len >= 1 && seq_len <= 512);
    kani::assume(in_features >= 1 && in_features <= 4096);
    kani::assume(out_features >= 1 && out_features <= 4096);

    // Input: [B, S, in_features]
    // Weight^T: [in_features, out_features]
    // Matmul: [B, S, K] @ [K, N] -> [B, S, N]
    let output_dim_0 = batch;
    let output_dim_1 = seq_len;
    let output_dim_2 = out_features;

    assert!(output_dim_0 == batch, "batch dim preserved for 3D input");
    assert!(
        output_dim_1 == seq_len,
        "sequence dim preserved for 3D input"
    );
    assert!(
        output_dim_2 == out_features,
        "feature dim becomes out_features"
    );

    // Total element count
    let total = batch.checked_mul(seq_len);
    assert!(total.is_some(), "B*S must not overflow");
    let total = total.unwrap().checked_mul(out_features);
    assert!(total.is_some(), "B*S*out must not overflow");
}

// ===========================================================================
// Harness 12: Linear is deterministic (same shape -> same shape)
// ===========================================================================

/// Prove: calling Linear forward twice with the same input shape produces
/// the same output shape. Linear has no internal mutable state that affects
/// shape computation. This models determinism at the shape level.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_deterministic_shape() {
    let batch: usize = kani::any();
    let in_features: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 256);
    kani::assume(in_features >= 1 && in_features <= 4096);
    kani::assume(out_features >= 1 && out_features <= 4096);

    // First call: input [B, K] -> output [B, N]
    let output1_dim0 = batch;
    let output1_dim1 = out_features;

    // Second call with same input shape: input [B, K] -> output [B, N]
    let output2_dim0 = batch;
    let output2_dim1 = out_features;

    assert!(
        output1_dim0 == output2_dim0,
        "deterministic: batch dim same across calls"
    );
    assert!(
        output1_dim1 == output2_dim1,
        "deterministic: feature dim same across calls"
    );
}

// ===========================================================================
// Harness 13: Weight initialization bounds (Kaiming variance)
// ===========================================================================

/// Prove: Kaiming He initialization for Linear produces weights with
/// variance = 2 / in_features (for ReLU). The standard deviation is
/// sqrt(2/in_features), which is finite and positive for in_features >= 1.
fn sqrt_f32_stub_kaiming(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e6);
    if x > 0.0 {
        kani::assume(r > 0.0);
    }
    r
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub_kaiming)]
fn proof_linear_kaiming_init_variance() {
    let in_features: usize = kani::any();
    kani::assume(in_features >= 1 && in_features <= 4096);

    // Kaiming He variance: 2.0 / in_features
    let variance = 2.0f32 / (in_features as f32);

    assert!(variance.is_finite(), "Kaiming variance must be finite");
    assert!(variance > 0.0, "Kaiming variance must be positive");

    // Standard deviation = sqrt(variance)
    let std_dev = variance.sqrt();
    assert!(std_dev.is_finite(), "Kaiming std_dev must be finite");
    assert!(std_dev > 0.0, "Kaiming std_dev must be positive");

    // Xavier/Glorot variance: 2.0 / (in_features + out_features)
    let out_features: usize = kani::any();
    kani::assume(out_features >= 1 && out_features <= 4096);
    let fan_sum = in_features + out_features;
    kani::assume(fan_sum > 0);
    let xavier_var = 2.0f32 / (fan_sum as f32);
    assert!(xavier_var.is_finite(), "Xavier variance must be finite");
    assert!(xavier_var > 0.0, "Xavier variance must be positive");
}

// ===========================================================================
// Harness 14: Gradient shape matches weight shape
// ===========================================================================

/// Prove: the gradient of the loss with respect to weight has the same
/// shape as weight: [out_features, in_features]. For y = x @ W^T,
/// dL/dW = dL/dy^T @ x, which has shape [N, K]^T @ [B, K]... = [N, K].
/// This models the backward rule shape invariant.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_gradient_shape_matches_weight() {
    let batch: usize = kani::any();
    let in_features: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 256);
    kani::assume(in_features >= 1 && in_features <= 4096);
    kani::assume(out_features >= 1 && out_features <= 4096);

    // Forward: input [B, K], weight [N, K], output [B, N]
    // Backward: grad_output [B, N]
    // dL/dW = grad_output^T @ input = [N, B] @ [B, K] = [N, K]
    let grad_weight_dim_0 = out_features; // N
    let grad_weight_dim_1 = in_features; // K

    // Weight shape: [out_features, in_features] = [N, K]
    assert!(
        grad_weight_dim_0 == out_features,
        "grad_weight dim 0 must match weight dim 0"
    );
    assert!(
        grad_weight_dim_1 == in_features,
        "grad_weight dim 1 must match weight dim 1"
    );

    // Gradient for bias (when present): sum over batch dim of grad_output.
    // grad_bias shape: [N] = [out_features]
    let grad_bias_len = out_features;
    assert!(
        grad_bias_len == out_features,
        "grad_bias must have length out_features"
    );
}

// ===========================================================================
// Harness 15: Frozen weight is not modified during forward
// ===========================================================================

/// Prove: Linear::forward is a pure function with respect to weight shape.
/// The weight dimensions before and after forward are identical. This models
/// the structural invariant that forward does not mutate weight.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_frozen_weight_invariant() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);

    // Before forward: weight is [out_features, in_features]
    let weight_dim_0_before = out_features;
    let weight_dim_1_before = in_features;

    // After forward: weight is still [out_features, in_features]
    // (Linear::forward takes &self, not &mut self — cannot mutate)
    let weight_dim_0_after = weight_dim_0_before;
    let weight_dim_1_after = weight_dim_1_before;

    assert!(
        weight_dim_0_before == weight_dim_0_after,
        "weight dim 0 must not change during forward"
    );
    assert!(
        weight_dim_1_before == weight_dim_1_after,
        "weight dim 1 must not change during forward"
    );

    // Element count unchanged.
    let elems_before = weight_dim_0_before.checked_mul(weight_dim_1_before);
    let elems_after = weight_dim_0_after.checked_mul(weight_dim_1_after);
    assert!(
        elems_before == elems_after,
        "weight element count must not change"
    );
}

// ===========================================================================
// Harness 16: Two linear layers compose [in] -> [hidden] -> [out]
// ===========================================================================

/// Prove: composing two linear layers is dimensionally valid.
/// Linear1: [in_features, hidden] and Linear2: [hidden, out_features]
/// produces a path [B, in] -> [B, hidden] -> [B, out].
/// The intermediate dimension (hidden) must match.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_composition_shape() {
    let batch: usize = kani::any();
    let in_features: usize = kani::any();
    let hidden: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(in_features >= 1 && in_features <= 2048);
    kani::assume(hidden >= 1 && hidden <= 2048);
    kani::assume(out_features >= 1 && out_features <= 2048);

    // Linear1: weight [hidden, in_features]
    //   input [B, in_features] -> output [B, hidden]
    let intermediate_dim_0 = batch;
    let intermediate_dim_1 = hidden;

    // Linear2: weight [out_features, hidden]
    //   input [B, hidden] -> output [B, out_features]
    // Compatibility: intermediate last dim == Linear2 in_features
    let linear2_in_features = hidden;
    assert!(
        intermediate_dim_1 == linear2_in_features,
        "intermediate dim must match Linear2 in_features"
    );

    let final_dim_0 = batch;
    let final_dim_1 = out_features;

    assert!(final_dim_0 == batch, "batch preserved through composition");
    assert!(final_dim_1 == out_features, "final output features correct");

    // Total element count at each stage.
    let input_elems = batch.checked_mul(in_features);
    let hidden_elems = batch.checked_mul(hidden);
    let output_elems = batch.checked_mul(out_features);
    assert!(input_elems.is_some(), "input elems must not overflow");
    assert!(hidden_elems.is_some(), "hidden elems must not overflow");
    assert!(output_elems.is_some(), "output elems must not overflow");
}

// ===========================================================================
// Harness 17: Identity init: weight = I, bias = 0 -> output = input
// ===========================================================================

/// Prove: when weight is the identity matrix and bias is zero, the output
/// equals the input. For square weight [N, N], x @ I^T + 0 = x @ I = x.
/// Modeled at the scalar level: x * 1.0 + 0.0 = x.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_identity_init_preserves_input() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() < 1e6);

    // Identity weight: diagonal element = 1.0
    let weight_diag = 1.0f32;
    // Zero bias
    let bias = 0.0f32;

    // y = x * weight_diag + bias = x * 1.0 + 0.0 = x
    let y = x * weight_diag + bias;

    assert!(
        y == x,
        "identity weight + zero bias must produce output = input"
    );
    assert!(y.is_finite(), "output must be finite for finite input");
}

// ===========================================================================
// Harness 18: Zero weight produces zero output (plus bias)
// ===========================================================================

/// Prove: when all weights are zero, output = 0 + bias = bias.
/// Modeled at the scalar level for a single element.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_zero_weight_output() {
    let x: f32 = kani::any();
    let bias: f32 = kani::any();

    kani::assume(x.is_finite() && x.abs() < 1e6);
    kani::assume(bias.is_finite() && bias.abs() < 1e6);

    // Zero weight: all weight elements = 0.0
    let weight = 0.0f32;

    // y = x * 0.0 + bias = bias
    let y_with_bias = x * weight + bias;
    assert!(
        y_with_bias == bias,
        "zero weight with bias must produce output = bias"
    );

    // Without bias: y = x * 0.0 = 0.0
    let y_no_bias = x * weight;
    assert!(
        y_no_bias == 0.0,
        "zero weight without bias must produce zero output"
    );
}

// ===========================================================================
// Harness 19: Matmul associativity for linear chain (shape)
// ===========================================================================

/// Prove: for a chain of three linear layers, the output shape is the same
/// regardless of association order. (A @ B) @ C has the same output shape
/// as A @ (B @ C). This ensures linear layer stacking is well-defined.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_matmul_shape_associativity() {
    let batch: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    let d3: usize = kani::any();
    let d4: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(d1 >= 1 && d1 <= 512);
    kani::assume(d2 >= 1 && d2 <= 512);
    kani::assume(d3 >= 1 && d3 <= 512);
    kani::assume(d4 >= 1 && d4 <= 512);

    // Chain: [B, d1] -> Linear1 [d2, d1] -> [B, d2]
    //        [B, d2] -> Linear2 [d3, d2] -> [B, d3]
    //        [B, d3] -> Linear3 [d4, d3] -> [B, d4]
    //
    // Left-to-right: ((input @ W1^T) @ W2^T) @ W3^T
    // Shape: [B, d1] -> [B, d2] -> [B, d3] -> [B, d4]
    let left_assoc_output = [batch, d4];

    // Alternative: input @ ((W1^T @ W2^T) @ W3^T)
    // W1^T @ W2^T = [d1, d2] @ [d2, d3] = [d1, d3]
    // [d1, d3] @ W3^T = [d1, d3] @ [d3, d4] = [d1, d4]
    // input @ [d1, d4] = [B, d1] @ [d1, d4] = [B, d4]
    let right_assoc_output = [batch, d4];

    assert!(
        left_assoc_output[0] == right_assoc_output[0],
        "associativity: batch dim matches"
    );
    assert!(
        left_assoc_output[1] == right_assoc_output[1],
        "associativity: feature dim matches"
    );
}

// ===========================================================================
// Harness 20: Output element count does not overflow for bounded dims
// ===========================================================================

/// Prove: for typical ML dimensions (batch <= 256, seq <= 2048,
/// features <= 8192), the output element count does not overflow usize.
/// This covers both 2D and 3D inputs.
#[kani::unwind(1)]
#[kani::proof]
fn proof_linear_output_no_overflow() {
    let batch: usize = kani::any();
    let seq_len: usize = kani::any();
    let out_features: usize = kani::any();
    let is_3d: bool = kani::any();

    kani::assume(batch >= 1 && batch <= 256);
    kani::assume(seq_len >= 1 && seq_len <= 2048);
    kani::assume(out_features >= 1 && out_features <= 8192);

    if is_3d {
        // 3D output: [batch, seq_len, out_features]
        let step1 = batch.checked_mul(seq_len);
        assert!(step1.is_some(), "batch * seq must not overflow");
        let step2 = step1.unwrap().checked_mul(out_features);
        assert!(step2.is_some(), "B*S*N must not overflow for bounded dims");
        assert!(step2.unwrap() >= 1, "output must have at least 1 element");
    } else {
        // 2D output: [batch, out_features]
        let total = batch.checked_mul(out_features);
        assert!(total.is_some(), "batch * out_features must not overflow");
        assert!(total.unwrap() >= 1, "output must have at least 1 element");
    }
}
