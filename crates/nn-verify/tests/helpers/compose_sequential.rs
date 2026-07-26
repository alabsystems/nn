// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: sequential kernel composition through NY.
//!
//! Validates that NY can propagate bounds through a multi-layer
//! composed graph (kernel A → kernel B), demonstrating the composition path
//! required for model-level verification (#534).

use super::common;
use super::common::{assert_crown_tighter_when_not_fallback, extract_scalar, scalar_bounds};
use nn_verify::{compose_sequential, SequentialSpec};

// --- Composition tests ---

#[test]
fn test_snake_then_scale_composition() {
    // Snake: f(x, alpha) = x + (1/alpha) * sin(alpha*x)^2  (alpha=1.0)
    // Scale: g(x) = 2*x + 1
    // Composition: g(f(x)) = 2*(x + sin(x)^2) + 1
    //
    // For x in [-5, 5]:
    //   snake(x, 1.0) range ≈ [-4.16, 6.16]
    //   scale(snake(x)) ≈ [-7.32, 13.32]
    let snake = common::snake_kernel();
    let scale = common::parse_kernel("fn scale(x: f32) -> f32 { x * 2.0 + 1.0 }");

    let spec = SequentialSpec::new(&snake, &scale, &[1.0], &[], 0).expect("valid spec");
    let graph = compose_sequential(&spec).expect("compose snake → scale");

    let input = scalar_bounds(-5.0, 5.0);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = extract_scalar(&output);

    // Bounds should be finite.
    assert!(lo.is_finite(), "lower bound should be finite, got {lo}");
    assert!(hi.is_finite(), "upper bound should be finite, got {hi}");

    // Sound IBP bounds must contain the true output range.
    // Corrected empirical: scale(snake(-5, 1)) ≈ -7.16, scale(snake(5, 1)) ≈ 12.84.
    // (Previous comment of -7.32/13.32 was incorrect.)
    // IBP must be at least this wide; use slightly inside to allow for IBP relaxation.
    assert!(
        lo < -7.0,
        "lower bound should be < -7.0 (true min ≈ -7.16), got {lo}"
    );
    assert!(
        hi > 12.5,
        "upper bound should be > 12.5 (true max ≈ 12.84), got {hi}"
    );

    // Width guard: analytical range ≈ 20, IBP should not exceed 10x.
    let width = hi - lo;
    assert!(
        width < 200.0,
        "IBP width {width} exceeds 10x analytical range (~20); likely computation error"
    );
}

#[test]
fn test_snake_then_scale_crown_tighter_than_ibp() {
    let snake = common::snake_kernel();
    let scale = common::parse_kernel("fn scale(x: f32) -> f32 { x * 2.0 + 1.0 }");

    let spec = SequentialSpec::new(&snake, &scale, &[1.0], &[], 0).expect("valid spec");
    let graph = compose_sequential(&spec).expect("compose snake → scale");

    let input = scalar_bounds(-5.0, 5.0);
    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("Sequential: method={method:?}, fallback={fallback_reason:?}");
    common::assert_bounds_valid(&output);
}

#[test]
fn test_scale_then_clamp_composition() {
    // Scale: f(x) = 2*x + 1
    // Clamp: g(x) = clamp(x, -1, 1)
    // Composition: clamp(2*x + 1, -1, 1)
    //
    // For x in [-5, 5]: scale output is [-9, 11], then clamped to [-1, 1].
    // NY should prove output bounds are within [-1, 1].
    let scale = common::parse_kernel("fn scale(x: f32) -> f32 { x * 2.0 + 1.0 }");
    let clamp = common::parse_kernel("fn clamped(x: f32) -> f32 { x.clamp(-1.0, 1.0) }");

    let spec = SequentialSpec::new(&scale, &clamp, &[], &[], 0).expect("valid spec");
    let graph = compose_sequential(&spec).expect("compose scale → clamp");

    let input = scalar_bounds(-5.0, 5.0);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = extract_scalar(&output);

    assert!(lo.is_finite(), "lower bound should be finite, got {lo}");
    assert!(hi.is_finite(), "upper bound should be finite, got {hi}");

    // Clamp guarantees output in [-1, 1].
    assert!(lo >= -1.0 - 1e-6, "lower bound should be >= -1.0, got {lo}");
    assert!(hi <= 1.0 + 1e-6, "upper bound should be <= 1.0, got {hi}");
}

#[test]
fn test_composed_and_naive_both_produce_sound_bounds() {
    // Compare two approaches to multi-kernel verification:
    //   Naive: verify snake alone → use its output bounds as input to scale
    //   Composed: snake → scale as a single NY graph
    //
    // Both must produce finite, ordered bounds. IBP accumulates width through
    // layers, so the composed graph may be wider or tighter depending on
    // NY's per-layer relaxation. The key property is that both are
    // sound (contain the true output range).
    let snake = common::snake_kernel();
    let scale = common::parse_kernel("fn scale(x: f32) -> f32 { x * 2.0 + 1.0 }");

    let x_lo = -5.0f32;
    let x_hi = 5.0f32;

    // Naive: snake bounds → scale bounds
    let snake_graph = nn_verify::kernel_to_graph(&snake, &[1.0]).expect("build snake graph");
    let snake_output = snake_graph
        .propagate_ibp(&scalar_bounds(x_lo, x_hi))
        .expect("snake IBP");
    let (snake_lo, snake_hi) = extract_scalar(&snake_output);

    let scale_graph = nn_verify::kernel_to_graph(&scale, &[]).expect("build scale graph");
    let naive_output = scale_graph
        .propagate_ibp(&scalar_bounds(snake_lo, snake_hi))
        .expect("naive scale IBP");
    let (naive_lo, naive_hi) = extract_scalar(&naive_output);

    // Composed: single graph
    let spec = SequentialSpec::new(&snake, &scale, &[1.0], &[], 0).expect("valid spec");
    let composed = compose_sequential(&spec).expect("compose snake → scale");
    let composed_output = composed
        .propagate_ibp(&scalar_bounds(x_lo, x_hi))
        .expect("composed IBP");
    let (composed_lo, composed_hi) = extract_scalar(&composed_output);

    // Both should be finite.
    assert!(naive_lo.is_finite() && naive_hi.is_finite());
    assert!(composed_lo.is_finite() && composed_hi.is_finite());

    // Both must contain the true output range.
    // Corrected: scale(snake(-5, 1)) ≈ -7.16, scale(snake(5, 1)) ≈ 12.84.
    assert!(
        naive_lo < -7.0,
        "naive lower should be < -7.0 (true min ≈ -7.16), got {naive_lo}"
    );
    assert!(
        naive_hi > 12.5,
        "naive upper should be > 12.5 (true max ≈ 12.84), got {naive_hi}"
    );
    assert!(
        composed_lo < -7.0,
        "composed lower should be < -7.0, got {composed_lo}"
    );
    assert!(
        composed_hi > 12.5,
        "composed upper should be > 12.5, got {composed_hi}"
    );

    // Width guards: analytical range ≈ 20, IBP should not exceed 10x.
    let naive_width = naive_hi - naive_lo;
    let composed_width = composed_hi - composed_lo;
    assert!(
        naive_width < 200.0,
        "naive IBP width {naive_width} exceeds 10x analytical range"
    );
    assert!(
        composed_width < 200.0,
        "composed IBP width {composed_width} exceeds 10x analytical range"
    );
}

#[test]
fn test_composition_param_count_validation() {
    let snake = common::snake_kernel();
    let scale = common::parse_kernel("fn scale(x: f32) -> f32 { x * 2.0 + 1.0 }");

    // Wrong number of first_constants: snake has 2 params (x, alpha), so needs 1 constant.
    // Validation now happens at SequentialSpec::new(), not compose_sequential().
    let result = SequentialSpec::new(&snake, &scale, &[], &[], 0);
    assert!(result.is_err(), "should reject wrong first_constants count");

    // chain_param out of bounds: scale has 1 param.
    let result = SequentialSpec::new(&snake, &scale, &[1.0], &[], 5);
    assert!(result.is_err(), "should reject out-of-bounds chain_param");

    // Boundary chain_param: scale has 1 param (index 0), so chain_param=1 is exactly out of bounds.
    let result = SequentialSpec::new(&snake, &scale, &[1.0], &[], 1);
    assert!(
        result.is_err(),
        "should reject chain_param at boundary (== params.len())"
    );

    // Wrong second_constants count: snake has 2 params (y, alpha), chain_param=0
    // occupies one slot, so second needs 1 constant. Provide 0.
    let result = SequentialSpec::new(&scale, &snake, &[], &[], 0);
    assert!(
        result.is_err(),
        "should reject wrong second_constants count"
    );
}

#[test]
fn test_composition_rejects_non_finite_constants() {
    let snake = common::snake_kernel();
    let scale = common::parse_kernel("fn scale(x: f32) -> f32 { x * 2.0 + 1.0 }");

    // NaN in first_constants.
    let result = SequentialSpec::new(&snake, &scale, &[f32::NAN], &[], 0);
    assert!(result.is_err(), "should reject NaN in first_constants");

    // Infinity in first_constants.
    let result = SequentialSpec::new(&snake, &scale, &[f32::INFINITY], &[], 0);
    assert!(result.is_err(), "should reject Infinity in first_constants");

    // NEG_INFINITY in first_constants.
    let result = SequentialSpec::new(&snake, &scale, &[f32::NEG_INFINITY], &[], 0);
    assert!(
        result.is_err(),
        "should reject -Infinity in first_constants"
    );

    // NaN in second_constants: compose scale → snake, alpha=NaN.
    let result = SequentialSpec::new(&scale, &snake, &[], &[f32::NAN], 0);
    assert!(result.is_err(), "should reject NaN in second_constants");

    // Infinity in second_constants.
    let result = SequentialSpec::new(&scale, &snake, &[], &[f32::INFINITY], 0);
    assert!(
        result.is_err(),
        "should reject Infinity in second_constants"
    );
}

// Dvoice-realistic composition tests extracted to separate file (#1669).
#[path = "compose_sequential_dvoice.rs"]
mod dvoice;
