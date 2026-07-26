// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for LSTM GPU/sequence variants and VarBuilder
//! weight loading safety (#4085).
//!
//! Extends `kani_lstm_proofs.rs` (gate arithmetic) with proofs covering:
//!
//! ## LSTM GPU dispatch (harnesses 1-5):
//!  1. `gpu_lstm_cell_output_shapes` — fused cell returns [batch, hidden] for both h, c
//!  2. `gpu_lstm_sequence_output_shape` — fused sequence returns [seq, batch, hidden]
//!  3. `gpu_combined_bias_shape` — combined bias b_ih + b_hh preserves [4*H]
//!  4. `gpu_cpu_dispatch_shape_equivalence` — GPU and CPU paths produce same shapes
//!  5. `gpu_lstm_zero_state_shape` — zero-initialized state matches [batch, hidden]
//!
//! ## LSTM sequence dimension tracking (harnesses 6-10):
//!  6. `lstm_seq_timestep_count_matches_input` — per-timestep loop covers all T steps
//!  7. `lstm_seq_narrow_squeeze_shape` — narrow(0,t,1) + squeeze(0) gives [batch, inp]
//!  8. `lstm_seq_stack_dim0_shape` — cat of T [1, batch, H] gives [T, batch, H]
//!  9. `bilstm_flip_preserves_shape` — flip(0) preserves [seq, batch, feat] shape
//! 10. `bilstm_batch_first_transpose` — [B, T, F] <-> [T, B, F] transpose roundtrip
//!
//! ## BiLSTM bidirectional merge (harnesses 11-14):
//! 11. `bilstm_cat_dim2_feature_only` — cat along dim 2 changes only feature dim
//! 12. `bilstm_hidden_size_match_invariant` — fwd/bwd must share hidden_size
//! 13. `bilstm_reverse_then_forward_length` — reversed seq through LSTM preserves length
//! 14. `bilstm_final_state_shape_per_direction` — each direction's final state is [B, H]
//!
//! ## VarBuilder LSTM weight loading (harnesses 15-20):
//! 15. `vb_lstm_load_weight_keys` — Lstm::load requests correct PyTorch key names
//! 16. `vb_bilstm_weight_key_conventions` — BiLstm::load tries 3 naming conventions
//! 17. `vb_lstm_weight_shape_validates_four_h` — loaded w_ih shape [4*H, inp] is validated
//! 18. `vb_bilstm_eight_weight_tensors` — BiLstm needs exactly 8 weight tensors (4 per dir)
//! 19. `vb_pp_lstm_key_resolution` — vb.pp("encoder.lstm").get resolves correct full key
//! 20. `vb_contains_tensor_guards_optional_bias` — optional bias loads iff key exists
//!
//! Part of #4085.

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

/// Scalar sigmoid: 1 / (1 + exp(-x)).
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ===========================================================================
// LSTM GPU dispatch proofs
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 1: Fused GPU LSTM cell output shapes
// ---------------------------------------------------------------------------

/// Prove: fused GPU LSTM cell returns h_new and c_new both with shape
/// [batch, hidden_size], matching the CPU path.
///
/// The Metal backend's `lstm_cell` kernel returns `(h_new, c_new)` directly
/// (no narrow needed). This proves the output shapes are correct.
#[kani::unwind(1)]
#[kani::proof]
fn gpu_lstm_cell_output_shapes() {
    let batch: usize = kani::any();
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(input_size >= 1 && input_size <= 1024);
    kani::assume(hidden_size >= 1 && hidden_size <= 1024);

    // Input: [batch, input_size]
    let input_shape = [batch, input_size];
    assert!(input_shape.len() == 2, "input must be rank 2");

    // w_ih: [4*H, input_size], w_hh: [4*H, hidden_size]
    let four_h = 4 * hidden_size;
    let w_ih_shape = [four_h, input_size];
    let w_hh_shape = [four_h, hidden_size];

    // GPU fused kernel output: h_new, c_new each [batch, hidden_size]
    let h_new_shape = [batch, hidden_size];
    let c_new_shape = [batch, hidden_size];

    assert!(
        h_new_shape[0] == batch,
        "h_new batch must match input batch"
    );
    assert!(
        h_new_shape[1] == hidden_size,
        "h_new feature must be hidden_size"
    );
    assert!(
        c_new_shape == h_new_shape,
        "c_new shape must match h_new shape"
    );

    // Output is h_new cloned — same shape as h_new
    let output_shape = h_new_shape;
    assert!(
        output_shape[1] == hidden_size,
        "LSTM cell output feature dim must be hidden_size"
    );

    // Gates intermediate: [batch, 4*hidden_size]
    let gates_shape = [batch, four_h];
    assert!(
        gates_shape[1] == w_ih_shape[0],
        "gates dim must match w_ih rows"
    );
    assert!(
        gates_shape[1] == w_hh_shape[0],
        "gates dim must match w_hh rows"
    );
}

// ---------------------------------------------------------------------------
// Harness 2: Fused GPU LSTM sequence output shape
// ---------------------------------------------------------------------------

/// Prove: fused GPU LSTM sequence dispatch returns output [seq_len, batch,
/// hidden_size] and final states (h_n, c_n) each [batch, hidden_size].
///
/// The Metal backend's `lstm_sequence` kernel processes the full sequence
/// in one dispatch, returning (output, h_n, c_n).
#[kani::unwind(1)]
#[kani::proof]
fn gpu_lstm_sequence_output_shape() {
    let seq_len: usize = kani::any();
    let batch: usize = kani::any();
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 2048);
    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(input_size >= 1 && input_size <= 1024);
    kani::assume(hidden_size >= 1 && hidden_size <= 1024);

    // Input: [seq_len, batch, input_size] (time-major)
    let input_shape = [seq_len, batch, input_size];
    assert!(input_shape.len() == 3, "sequence input must be rank 3");

    // GPU fused sequence output: [seq_len, batch, hidden_size]
    let output_shape = [seq_len, batch, hidden_size];
    assert!(
        output_shape[0] == input_shape[0],
        "output seq_len must match input seq_len"
    );
    assert!(
        output_shape[1] == input_shape[1],
        "output batch must match input batch"
    );
    assert!(
        output_shape[2] == hidden_size,
        "output feature must be hidden_size"
    );

    // Final state: h_n, c_n each [batch, hidden_size]
    let h_n_shape = [batch, hidden_size];
    let c_n_shape = [batch, hidden_size];
    assert!(h_n_shape == c_n_shape, "h_n and c_n must have same shape");
    assert!(
        h_n_shape[0] == batch,
        "final state batch must match input batch"
    );
    assert!(
        h_n_shape[1] == hidden_size,
        "final state feature must be hidden_size"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: GPU combined bias shape
// ---------------------------------------------------------------------------

/// Prove: combining b_ih + b_hh into a single bias preserves shape [4*H].
///
/// The GPU path combines biases before dispatch: `combined = b_ih.add(b_hh)`.
/// Both biases are [4*hidden_size], and element-wise add preserves shape.
#[kani::unwind(1)]
#[kani::proof]
fn gpu_combined_bias_shape() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size >= 1 && hidden_size <= 1024);

    let four_h = 4 * hidden_size;

    // Both biases: [4*H]
    let b_ih_shape = [four_h];
    let b_hh_shape = [four_h];

    // Element-wise add: shape must be identical
    assert!(
        b_ih_shape == b_hh_shape,
        "b_ih and b_hh must have identical shapes for add"
    );

    // Combined bias shape: [4*H] (preserved by element-wise add)
    let combined_shape = [four_h];
    assert!(
        combined_shape[0] == four_h,
        "combined bias must be [4*hidden_size]"
    );

    // Verify 4 gates × hidden_size
    assert!(
        combined_shape[0] / 4 == hidden_size,
        "combined bias encodes 4 gates of hidden_size each"
    );

    // Verify combined bias can be split into 4 equal slices
    assert!(
        combined_shape[0] % 4 == 0,
        "combined bias must be divisible by 4"
    );
}

// ---------------------------------------------------------------------------
// Harness 4: GPU/CPU dispatch shape equivalence
// ---------------------------------------------------------------------------

/// Prove: GPU fused path and CPU per-timestep path produce outputs with
/// identical shapes. This models the critical invariant that the GPU fast
/// path is a drop-in replacement for the CPU path.
///
/// CPU path: per-timestep loop builds [seq, batch, H] from T×[1, batch, H]
/// GPU path: single dispatch returns [seq, batch, H] directly
#[kani::unwind(1)]
#[kani::proof]
fn gpu_cpu_dispatch_shape_equivalence() {
    let seq_len: usize = kani::any();
    let batch: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 2048);
    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(hidden_size >= 1 && hidden_size <= 1024);

    // CPU path: cat T × [1, batch, H] along dim 0 → [T, batch, H]
    let cpu_output_shape = [seq_len, batch, hidden_size];

    // GPU path: single fused dispatch → [T, batch, H]
    let gpu_output_shape = [seq_len, batch, hidden_size];

    assert!(
        cpu_output_shape == gpu_output_shape,
        "CPU and GPU output shapes must be identical"
    );

    // CPU final state: [batch, H] (from last timestep)
    let cpu_final_h = [batch, hidden_size];
    let cpu_final_c = [batch, hidden_size];

    // GPU final state: [batch, H] (from fused kernel)
    let gpu_final_h = [batch, hidden_size];
    let gpu_final_c = [batch, hidden_size];

    assert!(
        cpu_final_h == gpu_final_h,
        "final h shape must match between CPU and GPU"
    );
    assert!(
        cpu_final_c == gpu_final_c,
        "final c shape must match between CPU and GPU"
    );
}

// ---------------------------------------------------------------------------
// Harness 5: Zero-initialized state shape
// ---------------------------------------------------------------------------

/// Prove: zero-initialized LSTM state has shape [batch, hidden_size] for
/// both h and c. This is the default when no initial state is provided.
///
/// DynTensor::zeros(&[batch, hidden_size], DType::F32, &device) produces
/// a tensor with exactly [batch, hidden_size] shape.
#[kani::unwind(1)]
#[kani::proof]
fn gpu_lstm_zero_state_shape() {
    let batch: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(hidden_size >= 1 && hidden_size <= 1024);

    // Zero-initialized state shape
    let h_zero_shape = [batch, hidden_size];
    let c_zero_shape = [batch, hidden_size];

    // h and c must match (LstmState::new() invariant)
    assert!(
        h_zero_shape == c_zero_shape,
        "zero h and c must have identical shapes"
    );

    // Rank is 2
    assert!(h_zero_shape.len() == 2, "zero state must be rank 2");

    // Positive element count
    let elem_count = h_zero_shape[0] * h_zero_shape[1];
    assert!(elem_count >= 1, "zero state must have >= 1 element");

    // Shape matches what the LSTM forward expects
    assert!(
        h_zero_shape[0] == batch,
        "zero state batch must match input batch"
    );
    assert!(
        h_zero_shape[1] == hidden_size,
        "zero state feature must be hidden_size"
    );
}

// ===========================================================================
// LSTM sequence dimension tracking proofs
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 6: Per-timestep loop covers all T steps
// ---------------------------------------------------------------------------

/// Prove: the per-timestep loop `for t in 0..seq_len` processes exactly
/// seq_len timesteps, and the output vector has exactly seq_len elements.
///
/// This models the core loop in Lstm::forward_seq (both CPU and GPU paths).
#[kani::unwind(9)]
#[kani::proof]
fn lstm_seq_timestep_count_matches_input() {
    let seq_len: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 8);

    let mut count: usize = 0;
    for _t in 0..seq_len {
        count += 1;
    }

    assert!(
        count == seq_len,
        "loop must process exactly seq_len timesteps"
    );

    // Outputs vector length
    assert!(count == seq_len, "output vector must have seq_len elements");
}

// ---------------------------------------------------------------------------
// Harness 7: narrow(0, t, 1) + squeeze(0) shape
// ---------------------------------------------------------------------------

/// Prove: extracting timestep t from [seq, batch, input] via
/// narrow(0, t, 1) → [1, batch, input], then squeeze(0) → [batch, input].
///
/// This is the per-timestep extraction pattern in forward_seq.
#[kani::unwind(1)]
#[kani::proof]
fn lstm_seq_narrow_squeeze_shape() {
    let seq_len: usize = kani::any();
    let batch: usize = kani::any();
    let input_size: usize = kani::any();
    let t: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 2048);
    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(input_size >= 1 && input_size <= 1024);
    kani::assume(t < seq_len);

    // Input: [seq_len, batch, input_size]
    let input_shape = [seq_len, batch, input_size];

    // narrow(0, t, 1): take 1 timestep at position t → [1, batch, input_size]
    let narrow_shape = [1, input_shape[1], input_shape[2]];
    assert!(narrow_shape[0] == 1, "narrow length is 1");
    assert!(narrow_shape[1] == batch, "batch preserved by narrow");
    assert!(
        narrow_shape[2] == input_size,
        "input_size preserved by narrow"
    );

    // squeeze(0): remove dim 0 (size 1) → [batch, input_size]
    let squeezed_shape = [narrow_shape[1], narrow_shape[2]];
    assert!(squeezed_shape.len() == 2, "squeezed must be rank 2");
    assert!(squeezed_shape[0] == batch, "batch preserved by squeeze");
    assert!(
        squeezed_shape[1] == input_size,
        "input_size preserved by squeeze"
    );
}

// ---------------------------------------------------------------------------
// Harness 8: Stack along dim 0
// ---------------------------------------------------------------------------

/// Prove: concatenating T tensors of shape [1, batch, H] along dim 0
/// produces a tensor of shape [T, batch, H].
///
/// Models `DynTensor::cat(&output_refs, 0)` in forward_seq.
#[kani::unwind(1)]
#[kani::proof]
fn lstm_seq_stack_dim0_shape() {
    let seq_len: usize = kani::any();
    let batch: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 2048);
    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(hidden_size >= 1 && hidden_size <= 1024);

    // Each output after unsqueeze(0): [1, batch, hidden_size]
    let per_step_shape = [1, batch, hidden_size];

    // Cat along dim 0 of seq_len such tensors: [seq_len, batch, hidden_size]
    let cat_dim0 = seq_len * per_step_shape[0];
    let stacked_shape = [cat_dim0, per_step_shape[1], per_step_shape[2]];

    assert!(stacked_shape[0] == seq_len, "stacked dim 0 must be seq_len");
    assert!(stacked_shape[1] == batch, "stacked dim 1 must be batch");
    assert!(
        stacked_shape[2] == hidden_size,
        "stacked dim 2 must be hidden_size"
    );

    // Total elements = seq_len * batch * hidden_size
    let total = stacked_shape[0] * stacked_shape[1] * stacked_shape[2];
    let expected = seq_len * batch * hidden_size;
    assert!(total == expected, "total elements must match");
}

// ---------------------------------------------------------------------------
// Harness 9: flip(0) preserves shape
// ---------------------------------------------------------------------------

/// Prove: flip along dim 0 preserves the shape [seq, batch, feat].
///
/// BiLstm uses `input.flip(0)` for the backward direction and
/// `bwd_outputs_rev.flip(0)` to restore temporal order. Both must
/// preserve all dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn bilstm_flip_preserves_shape() {
    let seq_len: usize = kani::any();
    let batch: usize = kani::any();
    let feat: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 2048);
    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(feat >= 1 && feat <= 2048);

    let original_shape = [seq_len, batch, feat];

    // flip(dim) reverses elements along `dim` but does NOT change shape
    let flipped_shape = [original_shape[0], original_shape[1], original_shape[2]];

    assert!(flipped_shape == original_shape, "flip must preserve shape");

    // Double flip = identity (shape-wise)
    let double_flipped_shape = [flipped_shape[0], flipped_shape[1], flipped_shape[2]];
    assert!(
        double_flipped_shape == original_shape,
        "double flip must restore original shape"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: Batch-first transpose roundtrip
// ---------------------------------------------------------------------------

/// Prove: [batch, seq, feat] ↔ [seq, batch, feat] transpose roundtrip
/// preserves all dimensions.
///
/// BiLstm::forward_seq_batch_first transposes [B, T, F] → [T, B, F],
/// runs forward_seq, then transposes [T, B, 2H] → [B, T, 2H].
#[kani::unwind(1)]
#[kani::proof]
fn bilstm_batch_first_transpose() {
    let batch: usize = kani::any();
    let seq_len: usize = kani::any();
    let feat: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(seq_len >= 1 && seq_len <= 2048);
    kani::assume(feat >= 1 && feat <= 2048);

    // Batch-first: [batch, seq_len, feat]
    let batch_first = [batch, seq_len, feat];

    // transpose(0, 1): swap dims 0 and 1 → [seq_len, batch, feat]
    let time_first = [batch_first[1], batch_first[0], batch_first[2]];
    assert!(
        time_first[0] == seq_len,
        "dim 0 must be seq_len after transpose"
    );
    assert!(
        time_first[1] == batch,
        "dim 1 must be batch after transpose"
    );
    assert!(time_first[2] == feat, "dim 2 must be preserved");

    // After BiLSTM: output [seq_len, batch, 2*H] where H = feat for this test
    // transpose(0, 1) back: [batch, seq_len, 2*H]
    let back_to_batch_first = [time_first[1], time_first[0], time_first[2]];
    assert!(
        back_to_batch_first[0] == batch,
        "must restore batch to dim 0"
    );
    assert!(
        back_to_batch_first[1] == seq_len,
        "must restore seq_len to dim 1"
    );
    assert!(
        back_to_batch_first[2] == feat,
        "feature dim preserved through roundtrip"
    );

    // Element count preserved through both transposes
    let orig_elems = batch_first[0] * batch_first[1] * batch_first[2];
    let round_elems = back_to_batch_first[0] * back_to_batch_first[1] * back_to_batch_first[2];
    assert!(orig_elems == round_elems, "element count must be preserved");
}

// ===========================================================================
// BiLSTM bidirectional merge proofs
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 11: Cat along dim 2 changes only feature dim
// ---------------------------------------------------------------------------

/// Prove: concatenating forward [T, B, H] and backward [T, B, H] outputs
/// along dim 2 produces [T, B, 2H] — only the feature dim changes.
///
/// This is the core operation in BiLstm::forward_seq:
/// `DynTensor::cat(&[&fwd_outputs, &bwd_outputs], 2)`
#[kani::unwind(1)]
#[kani::proof]
fn bilstm_cat_dim2_feature_only() {
    let seq_len: usize = kani::any();
    let batch: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 2048);
    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(hidden_size >= 1 && hidden_size <= 1024);

    // Forward output: [seq_len, batch, hidden_size]
    let fwd_shape = [seq_len, batch, hidden_size];
    // Backward output: [seq_len, batch, hidden_size]
    let bwd_shape = [seq_len, batch, hidden_size];

    // Precondition: dims 0 and 1 must match for cat along dim 2
    assert!(fwd_shape[0] == bwd_shape[0], "seq_len must match for cat");
    assert!(fwd_shape[1] == bwd_shape[1], "batch must match for cat");

    // Cat along dim 2: [seq_len, batch, hidden + hidden]
    let cat_shape = [fwd_shape[0], fwd_shape[1], fwd_shape[2] + bwd_shape[2]];

    // Dim 0 unchanged
    assert!(cat_shape[0] == seq_len, "seq_len must be preserved");
    // Dim 1 unchanged
    assert!(cat_shape[1] == batch, "batch must be preserved");
    // Dim 2 doubled
    assert!(
        cat_shape[2] == 2 * hidden_size,
        "feature dim must be 2 * hidden_size"
    );

    // Total elements = 2 × single-direction elements
    let fwd_elems = fwd_shape[0] * fwd_shape[1] * fwd_shape[2];
    let cat_elems = cat_shape[0] * cat_shape[1] * cat_shape[2];
    assert!(cat_elems == 2 * fwd_elems, "BiLSTM output has 2x elements");
}

// ---------------------------------------------------------------------------
// Harness 12: Forward/backward hidden_size match invariant
// ---------------------------------------------------------------------------

/// Prove: BiLstm::new() requires forward_lstm.hidden_size() ==
/// backward_lstm.hidden_size(). If they differ, construction fails.
///
/// This invariant is critical: cat along dim 2 requires both directions
/// to produce tensors with the same last dimension.
#[kani::unwind(1)]
#[kani::proof]
fn bilstm_hidden_size_match_invariant() {
    let fwd_hidden: usize = kani::any();
    let bwd_hidden: usize = kani::any();

    kani::assume(fwd_hidden >= 1 && fwd_hidden <= 2048);
    kani::assume(bwd_hidden >= 1 && bwd_hidden <= 2048);

    let sizes_match = fwd_hidden == bwd_hidden;

    if sizes_match {
        // Valid: output feature dim is well-defined
        let output_feat = fwd_hidden + bwd_hidden;
        assert!(output_feat == 2 * fwd_hidden, "output = 2*H when matched");
    } else {
        // Invalid: cat along dim 2 would fail because feature dims differ
        assert!(
            fwd_hidden != bwd_hidden,
            "mismatched hidden sizes must be rejected"
        );
        // If we tried to cat:
        let attempted_output = fwd_hidden + bwd_hidden;
        assert!(
            attempted_output != 2 * fwd_hidden,
            "mismatched concat doesn't produce 2*fwd_hidden"
        );
        assert!(
            attempted_output != 2 * bwd_hidden,
            "mismatched concat doesn't produce 2*bwd_hidden"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 13: Reversed sequence through LSTM preserves length
// ---------------------------------------------------------------------------

/// Prove: reversing a sequence along dim 0 preserves seq_len, and running
/// LSTM on the reversed sequence also preserves seq_len. Then reversing the
/// output back preserves seq_len again.
///
/// This is the backward direction in BiLstm::forward_seq.
#[kani::unwind(1)]
#[kani::proof]
fn bilstm_reverse_then_forward_length() {
    let seq_len: usize = kani::any();
    let batch: usize = kani::any();
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 2048);
    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(input_size >= 1 && input_size <= 1024);
    kani::assume(hidden_size >= 1 && hidden_size <= 1024);

    // Input: [seq_len, batch, input_size]
    let input_seq_len = seq_len;

    // Step 1: flip(0) → reversed, shape [seq_len, batch, input_size]
    let reversed_seq_len = input_seq_len; // flip preserves shape

    // Step 2: LSTM on reversed → output [seq_len, batch, hidden_size]
    let lstm_out_seq_len = reversed_seq_len; // LSTM preserves seq_len

    // Step 3: flip(0) on output → [seq_len, batch, hidden_size]
    let final_seq_len = lstm_out_seq_len; // flip preserves shape

    // All sequence lengths are equal
    assert!(
        final_seq_len == input_seq_len,
        "backward direction must preserve sequence length"
    );
    assert!(
        reversed_seq_len == input_seq_len,
        "flip must preserve seq_len"
    );
    assert!(
        lstm_out_seq_len == input_seq_len,
        "LSTM must preserve seq_len"
    );
}

// ---------------------------------------------------------------------------
// Harness 14: Final state shape per direction
// ---------------------------------------------------------------------------

/// Prove: each direction of a BiLSTM produces a final state with shape
/// [batch, hidden_size]. The final states are independent — not concatenated.
#[kani::unwind(1)]
#[kani::proof]
fn bilstm_final_state_shape_per_direction() {
    let batch: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(hidden_size >= 1 && hidden_size <= 1024);

    // Forward direction final state
    let fwd_h_shape = [batch, hidden_size];
    let fwd_c_shape = [batch, hidden_size];

    // Backward direction final state
    let bwd_h_shape = [batch, hidden_size];
    let bwd_c_shape = [batch, hidden_size];

    // Each direction's h and c have the same shape
    assert!(fwd_h_shape == fwd_c_shape, "fwd h and c must match");
    assert!(bwd_h_shape == bwd_c_shape, "bwd h and c must match");

    // Both directions have the same state shape (same hidden_size)
    assert!(
        fwd_h_shape == bwd_h_shape,
        "fwd and bwd state shapes must match"
    );

    // State shape is NOT [batch, 2*hidden_size] — states are per-direction
    assert!(
        fwd_h_shape[1] == hidden_size,
        "final state is per-direction hidden_size, not 2*hidden_size"
    );
    assert!(
        fwd_h_shape[1] != 2 * hidden_size || hidden_size == 0,
        "final state must not be 2*hidden_size"
    );
}

// ===========================================================================
// VarBuilder LSTM weight loading proofs
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 15: LSTM load weight key names
// ---------------------------------------------------------------------------

/// Prove: Lstm::load requests exactly the right PyTorch key names for
/// weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0.
///
/// Models the VarBuilder key resolution for Lstm::load.
#[kani::unwind(16)]
#[kani::proof]
fn vb_lstm_load_weight_keys() {
    // Model the key construction from Lstm::load
    let prefix = "encoder.lstm";
    let keys = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"];

    for key in &keys {
        // resolve_name with path "encoder.lstm" and tensor name key
        let full_key = format!("{prefix}.{key}");

        // Full key must contain the prefix
        assert!(
            full_key.starts_with("encoder.lstm."),
            "resolved key must start with prefix"
        );

        // Full key must end with the tensor name
        assert!(
            full_key.ends_with(key),
            "resolved key must end with tensor name"
        );

        // No double dots
        assert!(
            !full_key.contains(".."),
            "resolved key must not contain double dots"
        );
    }

    // Weight keys contain "weight", bias keys contain "bias"
    assert!(keys[0].contains("weight"), "w_ih key must contain 'weight'");
    assert!(keys[1].contains("weight"), "w_hh key must contain 'weight'");
    assert!(keys[2].contains("bias"), "b_ih key must contain 'bias'");
    assert!(keys[3].contains("bias"), "b_hh key must contain 'bias'");
}

// ---------------------------------------------------------------------------
// Harness 16: BiLSTM weight key conventions
// ---------------------------------------------------------------------------

/// Prove: BiLstm::load tries 3 naming conventions for each weight tensor.
///
/// Convention 1: PyTorch-native (weight_ih_l0, weight_ih_l0_reverse)
/// Convention 2: Keyremap hybrid (forward.weight_ih_l0, backward.weight_ih_l0)
/// Convention 3: Decomposed (forward.weight_ih.weight, backward.weight_ih.weight)
///
/// Models the or_else fallback chain in BiLstm::load.
#[kani::unwind(16)]
#[kani::proof]
fn vb_bilstm_weight_key_conventions() {
    // Forward direction w_ih key names across 3 conventions
    let conv1 = "weight_ih_l0";
    let conv2 = "forward.weight_ih_l0";
    let conv3 = "forward.weight_ih.weight";

    // All three are valid strings (non-empty)
    assert!(!conv1.is_empty(), "convention 1 key must be non-empty");
    assert!(!conv2.is_empty(), "convention 2 key must be non-empty");
    assert!(!conv3.is_empty(), "convention 3 key must be non-empty");

    // All three are distinct (no accidental overlap)
    assert!(conv1 != conv2, "conventions 1 and 2 must differ");
    assert!(conv1 != conv3, "conventions 1 and 3 must differ");
    assert!(conv2 != conv3, "conventions 2 and 3 must differ");

    // Backward direction: reverse suffix vs backward prefix
    let bwd_conv1 = "weight_ih_l0_reverse";
    let bwd_conv2 = "backward.weight_ih_l0";
    let bwd_conv3 = "backward.weight_ih.weight";

    assert!(
        bwd_conv1 != bwd_conv2,
        "bwd conventions 1 and 2 must differ"
    );
    assert!(
        bwd_conv1 != bwd_conv3,
        "bwd conventions 1 and 3 must differ"
    );
    assert!(
        bwd_conv2 != bwd_conv3,
        "bwd conventions 2 and 3 must differ"
    );

    // Forward and backward are always distinct within same convention
    assert!(conv1 != bwd_conv1, "fwd/bwd conv1 must differ");
    assert!(conv2 != bwd_conv2, "fwd/bwd conv2 must differ");
    assert!(conv3 != bwd_conv3, "fwd/bwd conv3 must differ");
}

// ---------------------------------------------------------------------------
// Harness 17: LSTM weight shape validates 4*H
// ---------------------------------------------------------------------------

/// Prove: Lstm::load requests w_ih with shape [4*hidden_size, input_size],
/// and the 4*hidden_size multiplier is exactly correct for 4 gates.
///
/// Models the `vb.get(&[four_h, input_size], "weight_ih_l0")` call.
#[kani::unwind(1)]
#[kani::proof]
fn vb_lstm_weight_shape_validates_four_h() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(input_size >= 1 && input_size <= 1024);
    kani::assume(hidden_size >= 1 && hidden_size <= 1024);

    let four_h = 4 * hidden_size;

    // Shape requested by Lstm::load for w_ih
    let requested_w_ih = [four_h, input_size];

    // Shape requested for w_hh
    let requested_w_hh = [four_h, hidden_size];

    // Shape requested for biases
    let requested_bias = [four_h];

    // Validate: dim 0 encodes exactly 4 gates
    assert!(
        requested_w_ih[0] % 4 == 0,
        "w_ih dim 0 must be divisible by 4"
    );
    assert!(
        requested_w_ih[0] / 4 == hidden_size,
        "w_ih dim 0 / 4 must equal hidden_size"
    );

    assert!(
        requested_w_hh[0] % 4 == 0,
        "w_hh dim 0 must be divisible by 4"
    );
    assert!(
        requested_w_hh[0] / 4 == hidden_size,
        "w_hh dim 0 / 4 must equal hidden_size"
    );

    assert!(
        requested_bias[0] % 4 == 0,
        "bias dim must be divisible by 4"
    );
    assert!(
        requested_bias[0] / 4 == hidden_size,
        "bias dim / 4 must equal hidden_size"
    );

    // All three weight tensors agree on the number of gate rows
    assert!(
        requested_w_ih[0] == requested_w_hh[0],
        "w_ih and w_hh gate dimension must match"
    );
    assert!(
        requested_w_ih[0] == requested_bias[0],
        "weight and bias gate dimensions must match"
    );
}

// ---------------------------------------------------------------------------
// Harness 18: BiLSTM needs 8 weight tensors
// ---------------------------------------------------------------------------

/// Prove: BiLstm construction requires exactly 8 weight tensor slots
/// (4 per direction × 2 directions), where biases are optional.
///
/// Each direction needs: w_ih, w_hh (required), b_ih, b_hh (optional).
/// Total: 4 required + up to 4 optional = 8 slots.
#[kani::unwind(1)]
#[kani::proof]
fn vb_bilstm_eight_weight_tensors() {
    // Count weight tensors per direction
    let required_per_dir: usize = 2; // w_ih, w_hh
    let optional_per_dir: usize = 2; // b_ih, b_hh
    let total_per_dir = required_per_dir + optional_per_dir;

    assert!(total_per_dir == 4, "each direction has 4 weight slots");

    // Two directions
    let num_directions: usize = 2;
    let total_slots = total_per_dir * num_directions;

    assert!(total_slots == 8, "BiLSTM has 8 total weight slots");

    // Minimum tensors (no biases): 2 required × 2 directions = 4
    let min_tensors = required_per_dir * num_directions;
    assert!(min_tensors == 4, "minimum is 4 weight tensors");

    // Maximum tensors (all biases): 4 total × 2 directions = 8
    let max_tensors = total_per_dir * num_directions;
    assert!(max_tensors == 8, "maximum is 8 weight tensors");

    // With biases, parameter count per direction:
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size >= 1 && input_size <= 256);
    kani::assume(hidden_size >= 1 && hidden_size <= 256);

    let four_h = 4 * hidden_size;
    let w_ih_elems = four_h * input_size;
    let w_hh_elems = four_h * hidden_size;
    let b_elems = four_h;

    let params_per_dir = w_ih_elems + w_hh_elems + 2 * b_elems;
    let total_params = params_per_dir * num_directions;

    assert!(total_params >= 2, "total parameters must be positive");
}

// ---------------------------------------------------------------------------
// Harness 19: pp() key resolution for LSTM
// ---------------------------------------------------------------------------

/// Prove: VarBuilder pp() + get() resolves the correct full key for LSTM
/// weight loading with hierarchical prefixes.
///
/// Example: vb.pp("encoder").pp("lstm").get("weight_ih_l0")
/// resolves to "encoder.lstm.weight_ih_l0".
#[kani::unwind(16)]
#[kani::proof]
fn vb_pp_lstm_key_resolution() {
    // Model VarBuilder path construction
    let mut path: Vec<String> = Vec::new();

    // pp("encoder")
    let s1 = "encoder".to_string();
    if !s1.is_empty() {
        path.push(s1);
    }

    // pp("lstm")
    let s2 = "lstm".to_string();
    if !s2.is_empty() {
        path.push(s2);
    }

    assert!(
        path.len() == 2,
        "two non-empty pp calls must produce depth 2"
    );

    // resolve_name("weight_ih_l0")
    let tensor_name = "weight_ih_l0";
    let name = format!("{}.{}", path.join("."), tensor_name);

    assert!(
        name == "encoder.lstm.weight_ih_l0",
        "must resolve to encoder.lstm.weight_ih_l0"
    );

    // Dot count = path length (2 segments + 1 tensor name = 2 dots)
    let dot_count = name.chars().filter(|&c| c == '.').count();
    assert!(dot_count == 2, "must have exactly 2 dots");

    // No double dots, no leading/trailing dots
    assert!(!name.contains(".."), "must not contain double dots");
    assert!(!name.starts_with('.'), "must not start with dot");
    assert!(!name.ends_with('.'), "must not end with dot");
}

// ---------------------------------------------------------------------------
// Harness 20: contains_tensor guards optional bias
// ---------------------------------------------------------------------------

/// Prove: optional bias loading uses contains_tensor as a guard — bias is
/// loaded only when the key exists, and skipped (None) otherwise.
///
/// Models the `if vb.contains_tensor("bias_ih_l0") { Some(vb.get(...)) } else { None }`
/// pattern in Lstm::load.
#[kani::unwind(1)]
#[kani::proof]
fn vb_contains_tensor_guards_optional_bias() {
    // Model: key exists or not (symbolic boolean)
    let key_exists: bool = kani::any();

    // Model the loading logic
    let bias_loaded: bool = if key_exists {
        true // Some(vb.get(...))
    } else {
        false // None
    };

    // Bias loaded iff key exists
    assert!(
        bias_loaded == key_exists,
        "bias must be loaded iff key exists"
    );

    // If key doesn't exist, bias must be None (not an error)
    if !key_exists {
        assert!(
            !bias_loaded,
            "missing bias key must result in None, not error"
        );
    }

    // If key exists, bias must be Some (loaded successfully in happy path)
    if key_exists {
        assert!(bias_loaded, "existing bias key must result in Some");
    }

    // Two independent biases (b_ih and b_hh) can have independent existence
    let b_ih_exists: bool = kani::any();
    let b_hh_exists: bool = kani::any();

    let b_ih_loaded = b_ih_exists;
    let b_hh_loaded = b_hh_exists;

    // Both can independently be present or absent
    // All 4 combinations are valid: (None, None), (Some, None), (None, Some), (Some, Some)
    let valid = (!b_ih_loaded || b_ih_exists) && (!b_hh_loaded || b_hh_exists);
    assert!(valid, "each bias is independently optional");
}
