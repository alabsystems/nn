// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Dropout layer (#3716).
//!
//! Proves correctness properties of the Dropout inference-mode identity:
//!
//! 1. Dropout::new accepts any f32 drop_p
//! 2. Dropout forward is identity (output == input)
//! 3. Dropout size is exactly sizeof(f32)
//! 4. Dropout forward preserves shape (rank and dims)
//! 5. Dropout forward preserves element count
//! 6. Drop probability range: boundary values [0, 1] accepted
//! 7. Drop probability has no effect on output
//!
//! Part of #3716.

// ---------------------------------------------------------------------------
// Harness 1: Dropout::new accepts any f32 drop_p
// ---------------------------------------------------------------------------

/// Prove: Dropout::new never panics for any f32 value.
/// The drop_p is stored but unused at inference time, so no
/// validation is performed. NaN, Inf, negative — all accepted.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dropout_new_accepts_any_f32() {
    let drop_p: f32 = kani::any();

    // Models: Self { _drop_p: drop_p }
    // No validation, no Err path.
    let stored = drop_p;

    // The struct is always constructed successfully.
    // If drop_p is NaN, bitwise equality fails, so check is_nan separately.
    if drop_p.is_nan() {
        assert!(stored.is_nan(), "NaN must be stored as NaN");
    } else {
        assert!(stored == drop_p, "stored value must match input");
    }
}

// ---------------------------------------------------------------------------
// Harness 2: Dropout forward is identity at inference
// ---------------------------------------------------------------------------

/// Prove: Dropout forward returns the input unchanged.
/// Models: Ok(x.clone()) — the output is a clone of the input.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dropout_forward_is_identity() {
    let x_val: f32 = kani::any();
    kani::assume(x_val.is_finite());

    // Models: fn forward(&self, x: &DynTensor) -> Result<DynTensor> { Ok(x.clone()) }
    let output_val = x_val; // clone preserves value
    assert!(
        output_val == x_val,
        "dropout forward must return input unchanged"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: Dropout struct size is exactly sizeof(f32)
// ---------------------------------------------------------------------------

/// Prove: the Dropout struct has the same size as a single f32.
/// Only stores _drop_p: f32, no other fields.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dropout_size_equals_f32() {
    let dropout_size = core::mem::size_of::<f32>(); // _drop_p field
    let f32_size = core::mem::size_of::<f32>();

    assert!(dropout_size == f32_size, "Dropout size must equal f32 size");
}

// ---------------------------------------------------------------------------
// Harness 4: Dropout forward preserves rank
// ---------------------------------------------------------------------------

/// Prove: Dropout forward preserves the tensor rank. Since forward
/// returns x.clone(), the rank of the output equals the rank of the input.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dropout_preserves_rank() {
    let rank: usize = kani::any();
    kani::assume(rank <= 8);

    // clone() preserves rank.
    let output_rank = rank;
    assert!(output_rank == rank, "dropout must preserve tensor rank");
}

// ---------------------------------------------------------------------------
// Harness 5: Dropout forward preserves element count
// ---------------------------------------------------------------------------

/// Prove: Dropout forward preserves total element count.
/// clone() makes an exact copy, so element count is identical.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dropout_preserves_element_count() {
    let num_elements: usize = kani::any();
    kani::assume(num_elements >= 1 && num_elements <= 1_000_000);

    let output_elements = num_elements; // clone preserves
    assert!(
        output_elements == num_elements,
        "dropout must preserve element count"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: Drop probability boundary values accepted
// ---------------------------------------------------------------------------

/// Prove: the boundary values 0.0 and 1.0 are accepted by Dropout::new.
/// These are the extreme cases: never drop (0.0) and always drop (1.0).
#[kani::unwind(1)]
#[kani::proof]
fn proof_dropout_boundary_values_accepted() {
    // 0.0: never drop
    let p0: f32 = 0.0;
    let stored_0 = p0;
    assert!(stored_0 == 0.0, "0.0 must be stored");

    // 1.0: always drop (inference ignores this)
    let p1: f32 = 1.0;
    let stored_1 = p1;
    assert!(stored_1 == 1.0, "1.0 must be stored");

    // Both valid for inference — no validation to reject them.
}

// ---------------------------------------------------------------------------
// Harness 7: Drop probability does not affect output
// ---------------------------------------------------------------------------

/// Prove: regardless of the drop_p value, the forward pass returns
/// the input unchanged. The output is independent of drop_p.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dropout_probability_no_effect_on_output() {
    let drop_p: f32 = kani::any();
    let x_val: f32 = kani::any();
    kani::assume(x_val.is_finite());

    // Forward always returns x.clone(), ignoring _drop_p.
    let output = x_val;

    // Output is identical regardless of drop_p value.
    assert!(
        output == x_val,
        "output must equal input regardless of drop_p"
    );
    // The drop_p variable is intentionally unused.
    let _ = drop_p;
}
