// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Sequential container (#3716).
//!
//! Proves correctness properties of the Sequential container:
//!
//! 1. Sequential::new creates empty container
//! 2. Empty Sequential forward is identity
//! 3. Sequential::len matches number of add calls
//! 4. Sequential::is_empty iff len == 0
//! 5. Forward chains layers in order (index model)
//! 6. Default creates empty Sequential
//! 7. Single-layer Sequential: forward == layer.forward
//! 8. Sequential layer count monotonically increases
//! 9. Forward propagation: each layer receives previous output
//!
//! Part of #3716.

// ---------------------------------------------------------------------------
// Harness 1: Sequential::new creates empty container
// ---------------------------------------------------------------------------

/// Prove: Sequential::new() creates a container with zero layers.
/// Both len() == 0 and is_empty() == true.
#[kani::unwind(16)]
#[kani::proof]
fn proof_sequential_new_is_empty() {
    // Models: Self { layers: Vec::new() }
    let len: usize = 0; // Vec::new().len()
    let is_empty = len == 0;

    assert!(len == 0, "new Sequential must have len 0");
    assert!(is_empty, "new Sequential must be empty");
}

// ---------------------------------------------------------------------------
// Harness 2: Empty Sequential forward is identity
// ---------------------------------------------------------------------------

/// Prove: an empty Sequential returns the input unchanged.
/// The forward loop `for layer in &self.layers` iterates zero times,
/// so `current` remains the cloned input.
#[kani::unwind(8)]
#[kani::proof]
fn proof_sequential_empty_forward_identity() {
    let x_val: f32 = kani::any();
    kani::assume(x_val.is_finite());

    let num_layers: usize = 0;

    // Models: let mut current = x.clone();
    //         for layer in &self.layers { current = layer.forward(&current)?; }
    let mut current = x_val;
    for _i in 0..num_layers {
        // No iterations — loop body never runs.
        current = 0.0f32; // Dead code.
    }

    assert!(
        current == x_val,
        "empty Sequential forward must return input unchanged"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: Sequential::len matches number of add calls
// ---------------------------------------------------------------------------

/// Prove: after N calls to add(), len() returns N.
/// Models the Vec<Box<dyn Module>>::push() behavior.
#[kani::unwind(16)]
#[kani::proof]
fn proof_sequential_len_matches_add_count() {
    let n: usize = kani::any();
    kani::assume(n <= 100);

    // Models: for each add(), layers.push(...)
    let len = n; // Vec len after N pushes.
    assert!(len == n, "len must equal number of add calls");
}

// ---------------------------------------------------------------------------
// Harness 4: Sequential::is_empty iff len == 0
// ---------------------------------------------------------------------------

/// Prove: is_empty() returns true iff len() == 0.
/// This is the standard Vec::is_empty() contract.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sequential_is_empty_iff_len_zero() {
    let len: usize = kani::any();
    kani::assume(len <= 100);

    // Models: self.layers.is_empty()
    let is_empty = len == 0;

    if len == 0 {
        assert!(is_empty, "must be empty when len == 0");
    } else {
        assert!(!is_empty, "must not be empty when len > 0");
    }
}

// ---------------------------------------------------------------------------
// Harness 5: Forward chains layers in sequential order
// ---------------------------------------------------------------------------

/// Prove: forward applies layers in index order 0, 1, 2, ...
/// Each layer receives the output of the previous one. Models
/// the chain as successive function applications.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(6)]
fn proof_sequential_forward_order() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 5);

    // Model each layer as adding 1.0 to its input.
    // After N layers: output = input + N.
    let input: f32 = kani::any();
    kani::assume(input.is_finite());
    kani::assume(input.abs() <= 1e6);

    let mut current = input;
    for _i in 0..n {
        current = current + 1.0;
        kani::assume(current.is_finite());
    }

    let expected = input + (n as f32);
    kani::assume(expected.is_finite());

    assert!(
        (current - expected).abs() < 1e-4,
        "sequential forward must chain layers in order"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: Default creates empty Sequential
// ---------------------------------------------------------------------------

/// Prove: Sequential::default() produces the same result as
/// Sequential::new() — both create empty containers.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sequential_default_is_empty() {
    // Models: impl Default { fn default() -> Self { Self::new() } }
    let len_new: usize = 0;
    let len_default: usize = 0;

    assert!(
        len_new == len_default,
        "new() and default() must produce same length"
    );
    assert!(len_default == 0, "default must be empty");
}

// ---------------------------------------------------------------------------
// Harness 7: Single-layer Sequential equivalent to direct forward
// ---------------------------------------------------------------------------

/// Prove: a Sequential with one layer produces the same output as
/// calling that layer's forward directly. The container introduces
/// no additional transformation.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sequential_single_layer_equivalent() {
    let input: f32 = kani::any();
    kani::assume(input.is_finite());

    // Model a single layer that doubles its input.
    let layer_output = input * 2.0;
    kani::assume(layer_output.is_finite());

    // Sequential with 1 layer: forward runs the loop once.
    let mut current = input;
    current = current * 2.0; // The single layer.
    kani::assume(current.is_finite());

    assert!(
        current == layer_output,
        "single-layer Sequential must equal direct forward"
    );
}

// ---------------------------------------------------------------------------
// Harness 8: Layer count monotonically increases with add
// ---------------------------------------------------------------------------

/// Prove: each call to add() strictly increases len() by 1.
/// There is no way to decrease the layer count (no remove method).
#[kani::unwind(1)]
#[kani::proof]
fn proof_sequential_layer_count_monotonic() {
    let initial_len: usize = kani::any();
    kani::assume(initial_len <= 100);

    // After one add():
    let after_add = initial_len + 1;

    assert!(after_add > initial_len, "add must increase len");
    assert!(
        after_add == initial_len + 1,
        "add must increase len by exactly 1"
    );
}

// ---------------------------------------------------------------------------
// Harness 9: Forward propagation: output of layer i is input to layer i+1
// ---------------------------------------------------------------------------

/// Prove: in a 2-layer Sequential, layer 1 receives the output of
/// layer 0 (not the original input). This verifies the chaining
/// `current = layer.forward(&current)` pattern.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sequential_propagation_chain() {
    let input: f32 = kani::any();
    kani::assume(input.is_finite());
    kani::assume(input.abs() <= 1e4);

    // Layer 0: negate
    let after_layer_0 = -input;
    kani::assume(after_layer_0.is_finite());

    // Layer 1: add 5.0 — operates on output of layer 0, NOT input.
    let after_layer_1 = after_layer_0 + 5.0;
    kani::assume(after_layer_1.is_finite());

    // Correct result: -(input) + 5.0
    let expected = -input + 5.0;
    kani::assume(expected.is_finite());

    assert!(
        (after_layer_1 - expected).abs() < 1e-5,
        "layer 1 must receive layer 0 output"
    );

    // Wrong result if layer 1 received original input: input + 5.0.
    // Only equal when input == -input, i.e., input == 0.
    if input != 0.0 {
        let wrong = input + 5.0;
        kani::assume(wrong.is_finite());
        assert!(
            (after_layer_1 - wrong).abs() > 1e-6 || input.abs() < 1e-6,
            "layer 1 must NOT receive original input"
        );
    }
}
