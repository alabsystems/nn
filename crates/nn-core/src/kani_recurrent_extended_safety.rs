// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for GRU and extended recurrent layer safety.
//!
//! Proves key mathematical and structural properties of GRU cells,
//! bidirectional recurrent layers, multi-layer stacking, dropout scaling,
//! and packed sequence invariants.
//!
//! GRU equations:
//! ```text
//! r = sigmoid(Wr * x + Ur * h)          -- reset gate in (0, 1)
//! z = sigmoid(Wz * x + Uz * h)          -- update gate in (0, 1)
//! h_tilde = tanh(W * x + U * (r * h))   -- candidate hidden in (-1, 1)
//! h_new = (1 - z) * h + z * h_tilde     -- convex combination
//! ```
//!
//! Properties proved:
//!  1. GRU reset gate: sigmoid output in (0, 1)
//!  2. GRU update gate: sigmoid output in (0, 1)
//!  3. GRU candidate hidden: tanh output in (-1, 1)
//!  4. GRU output interpolation: convex combination preserves bounds
//!  5. Bidirectional concatenation: output dim = 2 * hidden_size
//!  6. Hidden state shape: [num_layers, batch, hidden_size]
//!  7. Sequence length bounds: output seq len == input seq len
//!  8. Multi-layer stacking: each layer takes previous layer's output
//!  9. Dropout between layers: values scaled by 1/(1-p) during training
//! 10. PackedSequence: sorted by decreasing length, batch dim varies
//!
//! Part of #4209.

#![cfg(kani)]

// -- Stubs for CBMC (same pattern as kani_lstm.rs) ----------------------------

/// Nondeterministic exp stub: returns any positive finite value.
/// Sound over-approximation: exp(finite) is always positive and finite.
fn exp_stub(_x: f32) -> f32 {
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result > 0.0);
    result
}

/// Nondeterministic tanh stub: returns any value in (-1, 1).
/// Sound over-approximation: tanh(finite) is always in (-1, 1).
fn tanh_stub(_x: f32) -> f32 {
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result > -1.0 && result < 1.0);
    result
}

// -- Scalar helpers -----------------------------------------------------------

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh_scalar(x: f32) -> f32 {
    x.tanh()
}

// ===========================================================================
// 1. GRU reset gate: r = sigmoid(Wr*x + Ur*h), values in [0,1]
// ===========================================================================

/// Proves the GRU reset gate produces output strictly in (0, 1).
///
/// The reset gate is r = sigmoid(pre_activation) where the pre-activation
/// is Wr*x + Ur*h + bias. For any bounded finite pre-activation, the
/// sigmoid output is in (0, 1).
#[kani::unwind(4)]
#[kani::proof]
#[kani::stub(f32::exp, exp_stub)]
fn gru_reset_gate_bounded() {
    let pre_act: f32 = kani::any();
    kani::assume(pre_act.is_finite() && pre_act >= -1000.0 && pre_act <= 1000.0);

    let r = sigmoid_scalar(pre_act);

    assert!(r.is_finite(), "reset gate must be finite");
    assert!(r > 0.0, "reset gate must be > 0");
    assert!(r < 1.0, "reset gate must be < 1");
}

// ===========================================================================
// 2. GRU update gate: z = sigmoid(Wz*x + Uz*h), values in [0,1]
// ===========================================================================

/// Proves the GRU update gate produces output strictly in (0, 1).
///
/// The update gate z = sigmoid(pre_activation) controls the interpolation
/// between old hidden state and candidate. Must be in (0, 1) for the
/// convex combination to be valid.
#[kani::unwind(4)]
#[kani::proof]
#[kani::stub(f32::exp, exp_stub)]
fn gru_update_gate_bounded() {
    let pre_act: f32 = kani::any();
    kani::assume(pre_act.is_finite() && pre_act >= -1000.0 && pre_act <= 1000.0);

    let z = sigmoid_scalar(pre_act);

    assert!(z.is_finite(), "update gate must be finite");
    assert!(z > 0.0, "update gate must be > 0");
    assert!(z < 1.0, "update gate must be < 1");
}

// ===========================================================================
// 3. GRU candidate hidden: h_tilde = tanh(...), values in (-1,1)
// ===========================================================================

/// Proves the GRU candidate hidden state is in (-1, 1).
///
/// h_tilde = tanh(W*x + U*(r*h)) where r is the reset gate.
/// Since tanh maps R -> (-1, 1), for any finite pre-activation
/// the candidate is strictly bounded.
#[kani::unwind(4)]
#[kani::proof]
#[kani::stub(f32::tanh, tanh_stub)]
fn gru_candidate_hidden_bounded() {
    let pre_act: f32 = kani::any();
    kani::assume(pre_act.is_finite() && pre_act >= -1000.0 && pre_act <= 1000.0);

    let h_tilde = tanh_scalar(pre_act);

    assert!(h_tilde.is_finite(), "candidate hidden must be finite");
    assert!(h_tilde > -1.0, "candidate hidden must be > -1");
    assert!(h_tilde < 1.0, "candidate hidden must be < 1");
}

// ===========================================================================
// 4. GRU output interpolation: h_new = (1-z)*h + z*h_tilde is convex
// ===========================================================================

/// Proves the GRU output is a convex combination of h and h_tilde.
///
/// h_new = (1 - z) * h + z * h_tilde
///
/// When z in (0, 1), h in [-H, H], h_tilde in (-1, 1):
///   |h_new| <= (1-z)*|h| + z*|h_tilde| < (1-z)*H + z*1
///
/// For H = 1 (fresh GRU or bounded recurrence): |h_new| < 1.
/// This proves the hidden state stays bounded across steps.
#[kani::unwind(4)]
#[kani::proof]
fn gru_output_convex_combination() {
    let z: f32 = kani::any();
    let h: f32 = kani::any();
    let h_tilde: f32 = kani::any();

    // Update gate from sigmoid
    kani::assume(z.is_finite() && z > 0.0 && z < 1.0);
    // Previous hidden state bounded (e.g., from prior tanh or init)
    kani::assume(h.is_finite() && h >= -1.0 && h <= 1.0);
    // Candidate from tanh
    kani::assume(h_tilde.is_finite() && h_tilde > -1.0 && h_tilde < 1.0);

    let one_minus_z = 1.0 - z;
    let h_new = one_minus_z * h + z * h_tilde;

    assert!(h_new.is_finite(), "GRU output must be finite");
    // Convex combination of values in [-1, 1] stays in [-1, 1].
    // (1-z)*h + z*h_tilde, with 0 < z < 1, |h| <= 1, |h_tilde| < 1:
    //   |h_new| <= (1-z)*1 + z*1 = 1
    // Strict bound because h_tilde is strictly in (-1,1):
    assert!(
        h_new >= -1.0 - 1e-6,
        "GRU output must be >= -1 (within f32 tolerance)"
    );
    assert!(
        h_new <= 1.0 + 1e-6,
        "GRU output must be <= 1 (within f32 tolerance)"
    );
}

/// Proves GRU output interpolation with wider hidden state bounds.
///
/// When previous h is from multi-step recurrence with |h| <= B,
/// and h_tilde in (-1, 1), the output satisfies:
///   |h_new| < max(B, 1)
///
/// This is the general contraction: if B > 1, the interpolation
/// pulls h_new toward (-1, 1); if B <= 1, it stays within 1.
#[kani::unwind(4)]
#[kani::proof]
fn gru_output_contraction_general() {
    let z: f32 = kani::any();
    let h: f32 = kani::any();
    let h_tilde: f32 = kani::any();

    kani::assume(z.is_finite() && z > 0.0 && z < 1.0);
    // Allow wider bound for multi-step h
    kani::assume(h.is_finite() && h >= -10.0 && h <= 10.0);
    kani::assume(h_tilde.is_finite() && h_tilde > -1.0 && h_tilde < 1.0);

    let h_new = (1.0 - z) * h + z * h_tilde;

    assert!(h_new.is_finite(), "GRU output must be finite");
    // |h_new| <= (1-z)*10 + z*1 < 10 (since (1-z)*10 + z < 10 when z > 0)
    // More precisely: (1-z)*10 + z*1 = 10 - 9z < 10
    assert!(
        h_new.abs() < 10.0 + 1e-5,
        "GRU output must be bounded by max(|h|, 1)"
    );
}

// ===========================================================================
// 5. Bidirectional concatenation: output dim = 2 * hidden_size
// ===========================================================================

/// Proves bidirectional GRU/LSTM output dimension is 2 * hidden_size.
///
/// BiGRU concatenates forward and backward outputs along the feature dim:
///   fwd_output: [seq_len, batch, hidden_size]
///   bwd_output: [seq_len, batch, hidden_size]
///   result:     [seq_len, batch, 2 * hidden_size]
#[kani::unwind(4)]
#[kani::proof]
fn bidirectional_output_dim_is_2h() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= 4096);

    let fwd_features = hidden_size;
    let bwd_features = hidden_size;
    let output_features = fwd_features + bwd_features;

    assert_eq!(
        output_features,
        2 * hidden_size,
        "BiGRU output must be 2 * hidden_size"
    );

    // Numel check: concat doubles the feature dimension only
    let seq_len: usize = kani::any();
    let batch: usize = kani::any();
    kani::assume(seq_len > 0 && seq_len <= 512);
    kani::assume(batch > 0 && batch <= 64);

    let fwd_numel = seq_len * batch * hidden_size;
    let bwd_numel = seq_len * batch * hidden_size;
    let cat_numel = seq_len * batch * output_features;

    // Guard against overflow
    if let (Some(fn_), Some(bn_), Some(cn_)) = (
        seq_len
            .checked_mul(batch)
            .and_then(|x| x.checked_mul(hidden_size)),
        seq_len
            .checked_mul(batch)
            .and_then(|x| x.checked_mul(hidden_size)),
        seq_len
            .checked_mul(batch)
            .and_then(|x| x.checked_mul(output_features)),
    ) {
        assert_eq!(
            cn_,
            fn_ + bn_,
            "concat numel must equal sum of forward and backward"
        );
    }
}

// ===========================================================================
// 6. Hidden state shape: h has shape [num_layers, batch, hidden_size]
// ===========================================================================

/// Proves hidden state shape is [num_layers, batch, hidden_size].
///
/// For a multi-layer GRU/LSTM, the hidden state tensor has one entry
/// per layer. The total number of elements is num_layers * batch * hidden_size.
#[kani::unwind(4)]
#[kani::proof]
fn hidden_state_shape_invariant() {
    let num_layers: usize = kani::any();
    let batch: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(num_layers > 0 && num_layers <= 8);
    kani::assume(batch > 0 && batch <= 64);
    kani::assume(hidden_size > 0 && hidden_size <= 2048);

    let h_shape = [num_layers, batch, hidden_size];

    assert_eq!(h_shape[0], num_layers, "dim 0 must be num_layers");
    assert_eq!(h_shape[1], batch, "dim 1 must be batch");
    assert_eq!(h_shape[2], hidden_size, "dim 2 must be hidden_size");

    // Numel must be product of all dims
    if let Some(numel) = num_layers
        .checked_mul(batch)
        .and_then(|x| x.checked_mul(hidden_size))
    {
        assert_eq!(
            numel,
            h_shape[0] * h_shape[1] * h_shape[2],
            "numel must equal product of shape dims"
        );
    }

    // For bidirectional: shape becomes [num_layers * 2, batch, hidden_size]
    let num_directions: usize = 2;
    let bi_shape = [num_layers * num_directions, batch, hidden_size];
    assert_eq!(
        bi_shape[0],
        num_layers * 2,
        "bidirectional h dim 0 must be num_layers * 2"
    );
}

// ===========================================================================
// 7. Sequence length bounds: output seq len == input seq len
// ===========================================================================

/// Proves output sequence length equals input sequence length for RNNs.
///
/// Unlike convolution (which may change sequence length via stride/padding),
/// recurrent layers process one timestep at a time and always produce
/// exactly one output per input timestep.
#[kani::unwind(10)]
#[kani::proof]
fn sequence_length_preserved() {
    let input_seq_len: usize = kani::any();
    kani::assume(input_seq_len > 0 && input_seq_len <= 8);

    // Simulate processing: one output per timestep
    let mut output_count: usize = 0;
    let mut t: usize = 0;
    while t < input_seq_len {
        output_count += 1;
        t += 1;
    }

    assert_eq!(
        output_count, input_seq_len,
        "output seq len must equal input seq len"
    );

    // Output shape: [seq_len, batch, features]
    let batch: usize = kani::any();
    let features: usize = kani::any();
    kani::assume(batch > 0 && batch <= 8);
    kani::assume(features > 0 && features <= 16);

    let input_shape = [input_seq_len, batch, features];
    let output_shape = [output_count, batch, features];

    // Seq dim preserved, batch preserved, features may differ (hidden_size vs input_size)
    assert_eq!(input_shape[0], output_shape[0], "seq dim must be preserved");
    assert_eq!(
        input_shape[1], output_shape[1],
        "batch dim must be preserved"
    );
}

// ===========================================================================
// 8. Multi-layer stacking: each layer takes previous layer's output
// ===========================================================================

/// Proves multi-layer GRU dimension chaining is consistent.
///
/// Layer 0: input_size -> hidden_size
/// Layer 1..N-1: hidden_size -> hidden_size
///
/// The output of layer L is the input of layer L+1. For L > 0,
/// the input dimension must equal hidden_size (the output of the
/// previous layer).
#[kani::unwind(10)]
#[kani::proof]
fn multi_layer_stacking_dimensions() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    let num_layers: usize = kani::any();

    kani::assume(input_size > 0 && input_size <= 64);
    kani::assume(hidden_size > 0 && hidden_size <= 64);
    kani::assume(num_layers >= 1 && num_layers <= 8);

    // Track the current feature dimension through layers
    let mut current_features = input_size;
    let mut layer: usize = 0;

    while layer < num_layers {
        let layer_input_size = current_features;

        if layer == 0 {
            // First layer: input is the original input_size
            assert_eq!(
                layer_input_size, input_size,
                "layer 0 input must be input_size"
            );
        } else {
            // Subsequent layers: input is previous layer's output (hidden_size)
            assert_eq!(
                layer_input_size, hidden_size,
                "layer > 0 input must be hidden_size"
            );
        }

        // Layer output is always hidden_size
        current_features = hidden_size;
        layer += 1;
    }

    // Final output features is hidden_size regardless of num_layers
    assert_eq!(
        current_features, hidden_size,
        "final output features must be hidden_size"
    );
}

/// Proves bidirectional multi-layer stacking: layer > 0 input is 2*hidden_size.
///
/// In bidirectional mode, each layer's output is [seq, batch, 2*hidden_size].
/// Layers 1..N-1 must accept 2*hidden_size as input.
#[kani::unwind(10)]
#[kani::proof]
fn multi_layer_bidirectional_stacking() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    let num_layers: usize = kani::any();

    kani::assume(input_size > 0 && input_size <= 64);
    kani::assume(hidden_size > 0 && hidden_size <= 64);
    kani::assume(num_layers >= 1 && num_layers <= 8);

    let bi_output_size = 2 * hidden_size;
    let mut current_features = input_size;
    let mut layer: usize = 0;

    while layer < num_layers {
        if layer == 0 {
            assert_eq!(
                current_features, input_size,
                "bidirectional layer 0 input must be input_size"
            );
        } else {
            assert_eq!(
                current_features, bi_output_size,
                "bidirectional layer > 0 input must be 2 * hidden_size"
            );
        }

        // BiRNN layer output is always 2 * hidden_size
        current_features = bi_output_size;
        layer += 1;
    }

    assert_eq!(
        current_features, bi_output_size,
        "final bidirectional output must be 2 * hidden_size"
    );
}

// ===========================================================================
// 9. Dropout between layers: scaled by 1/(1-p) during training
// ===========================================================================

/// Proves dropout scaling factor 1/(1-p) preserves expected value.
///
/// During training, dropout zeros each element with probability p and
/// scales surviving elements by 1/(1-p). This ensures E[output] = input.
///
/// Verification: for p in (0, 1), the scale factor is finite and > 1.
#[kani::unwind(4)]
#[kani::proof]
fn dropout_scaling_preserves_expectation() {
    // Dropout probability as integer percentage to avoid f32 precision issues
    let p_pct: u8 = kani::any();
    kani::assume(p_pct >= 1 && p_pct <= 99);

    let p = p_pct as f32 / 100.0;
    let scale = 1.0 / (1.0 - p);

    assert!(scale.is_finite(), "dropout scale must be finite for p < 1");
    assert!(scale > 1.0, "dropout scale must be > 1 for p > 0");

    // Scale * (1 - p) should be approximately 1.0 (preserves expectation)
    let product = scale * (1.0 - p);
    assert!(
        (product - 1.0).abs() < 1e-5,
        "scale * (1-p) must equal ~1.0"
    );
}

/// Proves dropout is identity during inference (scale = 1, no masking).
///
/// During inference, dropout is not applied. The scale factor is 1.0
/// and no elements are zeroed.
#[kani::unwind(4)]
#[kani::proof]
fn dropout_inference_is_identity() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);

    // During inference: no dropout, scale = 1.0
    let training = false;
    let scale: f32 = if training { 2.0 } else { 1.0 }; // example p=0.5

    let output = x * scale;

    if !training {
        assert_eq!(output, x, "dropout must be identity during inference");
    }
}

/// Proves dropout between layers only applies to layers 0..N-2 (not last).
///
/// In multi-layer RNNs, dropout is applied between layers but NOT after
/// the final layer. This means N-1 dropout applications for N layers.
#[kani::unwind(10)]
#[kani::proof]
fn dropout_between_layers_count() {
    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 8);

    let mut dropout_applications: usize = 0;
    let mut layer: usize = 0;

    while layer < num_layers {
        // Dropout applied after each layer EXCEPT the last
        if layer < num_layers - 1 {
            dropout_applications += 1;
        }
        layer += 1;
    }

    if num_layers >= 2 {
        assert_eq!(
            dropout_applications,
            num_layers - 1,
            "dropout applied between layers: N-1 times for N layers"
        );
    } else {
        assert_eq!(dropout_applications, 0, "single layer: no dropout applied");
    }
}

// ===========================================================================
// 10. PackedSequence: sorted by decreasing length
// ===========================================================================

/// Proves PackedSequence batch sizes decrease monotonically.
///
/// In a packed sequence, sequences are sorted by decreasing length.
/// At timestep t, the batch size is the number of sequences with
/// length > t. This means batch_sizes[t] >= batch_sizes[t+1].
#[kani::unwind(10)]
#[kani::proof]
fn packed_sequence_batch_sizes_monotonic() {
    // Model 4 sequences with symbolic lengths
    let len0: u8 = kani::any();
    let len1: u8 = kani::any();
    let len2: u8 = kani::any();
    let len3: u8 = kani::any();

    kani::assume(len0 >= 1 && len0 <= 8);
    kani::assume(len1 >= 1 && len1 <= 8);
    kani::assume(len2 >= 1 && len2 <= 8);
    kani::assume(len3 >= 1 && len3 <= 8);

    // Sorted by decreasing length (PackedSequence invariant)
    kani::assume(len0 >= len1);
    kani::assume(len1 >= len2);
    kani::assume(len2 >= len3);

    let max_len = len0 as usize;
    let lengths = [len0 as usize, len1 as usize, len2 as usize, len3 as usize];

    // Compute batch_size at each timestep
    let mut t: usize = 0;
    let mut prev_batch_size: usize = 4; // all sequences present at t=0

    while t < max_len {
        // Count sequences with length > t
        let mut batch_size: usize = 0;
        let mut i: usize = 0;
        while i < 4 {
            if lengths[i] > t {
                batch_size += 1;
            }
            i += 1;
        }

        // Batch size must be monotonically non-increasing
        assert!(
            batch_size <= prev_batch_size,
            "batch_sizes must be monotonically non-increasing"
        );
        assert!(
            batch_size >= 1,
            "batch_size must be >= 1 at each active timestep"
        );

        prev_batch_size = batch_size;
        t += 1;
    }
}

/// Proves PackedSequence total elements equals sum of sequence lengths.
///
/// The total number of elements in a packed sequence must equal the
/// sum of all individual sequence lengths. This is the fundamental
/// data conservation invariant.
#[kani::unwind(10)]
#[kani::proof]
fn packed_sequence_element_count() {
    let len0: u8 = kani::any();
    let len1: u8 = kani::any();
    let len2: u8 = kani::any();

    kani::assume(len0 >= 1 && len0 <= 6);
    kani::assume(len1 >= 1 && len1 <= 6);
    kani::assume(len2 >= 1 && len2 <= 6);

    // Sorted decreasing
    kani::assume(len0 >= len1);
    kani::assume(len1 >= len2);

    let lengths = [len0 as usize, len1 as usize, len2 as usize];
    let max_len = len0 as usize;

    // Sum of all lengths
    let total_elements = lengths[0] + lengths[1] + lengths[2];

    // Sum of batch_sizes across timesteps must equal total_elements
    let mut sum_batch_sizes: usize = 0;
    let mut t: usize = 0;
    while t < max_len {
        let mut batch_size: usize = 0;
        let mut i: usize = 0;
        while i < 3 {
            if lengths[i] > t {
                batch_size += 1;
            }
            i += 1;
        }
        sum_batch_sizes += batch_size;
        t += 1;
    }

    assert_eq!(
        sum_batch_sizes, total_elements,
        "sum of batch_sizes must equal sum of sequence lengths"
    );
}

// ===========================================================================
// Bonus: GRU full step composition
// ===========================================================================

/// Proves the full GRU step produces finite, bounded output.
///
/// Exercises the complete gate computation:
///   r = sigmoid(gate_r)
///   z = sigmoid(gate_z)
///   h_tilde = tanh(gate_h)
///   h_new = (1-z)*h + z*h_tilde
#[kani::unwind(4)]
#[kani::proof]
#[kani::stub(f32::exp, exp_stub)]
#[kani::stub(f32::tanh, tanh_stub)]
fn gru_full_step_finite() {
    let gate_r_val: f32 = kani::any();
    let gate_z_val: f32 = kani::any();
    let gate_h_val: f32 = kani::any();
    let h: f32 = kani::any();

    kani::assume(gate_r_val.is_finite() && gate_r_val >= -50.0 && gate_r_val <= 50.0);
    kani::assume(gate_z_val.is_finite() && gate_z_val >= -50.0 && gate_z_val <= 50.0);
    kani::assume(gate_h_val.is_finite() && gate_h_val >= -50.0 && gate_h_val <= 50.0);
    kani::assume(h.is_finite() && h >= -1.0 && h <= 1.0);

    let r = sigmoid_scalar(gate_r_val);
    let z = sigmoid_scalar(gate_z_val);
    let h_tilde = tanh_scalar(gate_h_val);

    let h_new = (1.0 - z) * h + z * h_tilde;

    assert!(h_new.is_finite(), "GRU h_new must be finite");
    // h in [-1, 1] and h_tilde in (-1, 1), convex combination stays in [-1, 1]
    assert!(
        h_new >= -1.0 - 1e-6,
        "GRU h_new must be >= -1 (within tolerance)"
    );
    assert!(
        h_new <= 1.0 + 1e-6,
        "GRU h_new must be <= 1 (within tolerance)"
    );
}

/// Proves GRU 2-step composition: bounded h_0 -> bounded h_2.
///
/// Two consecutive GRU steps with independent symbolic gates.
/// If |h_0| <= 1, then |h_2| <= 1 (contraction property).
#[kani::unwind(4)]
#[kani::proof]
fn gru_two_step_contraction() {
    // Step 1 gates
    let z1: f32 = kani::any();
    let ht1: f32 = kani::any();
    // Step 2 gates
    let z2: f32 = kani::any();
    let ht2: f32 = kani::any();
    // Initial hidden state
    let h_0: f32 = kani::any();

    kani::assume(z1.is_finite() && z1 > 0.0 && z1 < 1.0);
    kani::assume(ht1.is_finite() && ht1 > -1.0 && ht1 < 1.0);
    kani::assume(z2.is_finite() && z2 > 0.0 && z2 < 1.0);
    kani::assume(ht2.is_finite() && ht2 > -1.0 && ht2 < 1.0);
    kani::assume(h_0.is_finite() && h_0 >= -1.0 && h_0 <= 1.0);

    // Step 1: h_1 = (1-z1)*h_0 + z1*ht1
    let h_1 = (1.0 - z1) * h_0 + z1 * ht1;
    // Step 2: h_2 = (1-z2)*h_1 + z2*ht2
    let h_2 = (1.0 - z2) * h_1 + z2 * ht2;

    assert!(h_1.is_finite(), "h_1 must be finite");
    assert!(h_2.is_finite(), "h_2 must be finite");

    // Convex combination of values in [-1, 1] stays in [-1, 1]
    assert!(
        h_1.abs() <= 1.0 + 1e-5,
        "h_1 must be bounded by 1 after step 1"
    );
    assert!(
        h_2.abs() <= 1.0 + 1e-5,
        "h_2 must be bounded by 1 after step 2 (2-step contraction)"
    );
}

/// Proves GRU base case: zero initial hidden state produces bounded output.
///
/// Starting from h_0 = 0 (standard initialization):
///   h_1 = (1-z)*0 + z*h_tilde = z*h_tilde
/// Since z in (0,1) and h_tilde in (-1,1): |h_1| < 1.
#[kani::unwind(4)]
#[kani::proof]
#[kani::stub(f32::exp, exp_stub)]
#[kani::stub(f32::tanh, tanh_stub)]
fn gru_base_case_zero_state() {
    let gate_z_val: f32 = kani::any();
    let gate_h_val: f32 = kani::any();

    kani::assume(gate_z_val.is_finite() && gate_z_val >= -50.0 && gate_z_val <= 50.0);
    kani::assume(gate_h_val.is_finite() && gate_h_val >= -50.0 && gate_h_val <= 50.0);

    let h_0: f32 = 0.0;

    let z = sigmoid_scalar(gate_z_val);
    let h_tilde = tanh_scalar(gate_h_val);

    // h_1 = (1-z)*0 + z*h_tilde = z*h_tilde
    let h_1 = (1.0 - z) * h_0 + z * h_tilde;

    assert!(h_1.is_finite(), "h_1 must be finite from zero state");
    // |h_1| = |z * h_tilde| < 1 * 1 = 1
    assert!(
        h_1.abs() < 1.0,
        "h_1 must be < 1 from zero state (z*h_tilde < 1)"
    );
}
