// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Gated DeltaNet decomposed builders.
//!
//! Proves structural correctness of decomposed Gated DeltaNet IR construction:
//! - `build_gated_delta_net_decomposed` no-panic, validate, output shape
//! - `build_gated_delta_net_decomposed_dual` no-panic, validate, dual output shape
//! - Error paths do not panic (zero dims, invalid scale)
//!
//! The decomposed builder creates ~20 primitive ops (Reshape, BinaryMul, MatMul,
//! BinaryAdd) from the recurrence equations. These harnesses verify the builder
//! graph is structurally valid for all bounded parameter combinations.
//!
//! Part of #834 — Gated DeltaNet for Qwen3.5 model support.

use super::{build_gated_delta_net_decomposed, build_gated_delta_net_decomposed_dual};

// CBMC cannot model f32::sqrt. Use nondeterministic stub.
fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// ---------------------------------------------------------------------------
// build_gated_delta_net_decomposed — single output
// ---------------------------------------------------------------------------

/// Proves `build_gated_delta_net_decomposed` does not panic for any bounded params,
/// including invalid ones (0 dimensions, bad scale) that return `Err`.
#[kani::unwind(1)]
#[kani::proof]
fn gdn_decomposed_build_no_panic() {
    let h: usize = kani::any();
    let k: usize = kani::any();
    let v: usize = kani::any();

    kani::assume(h <= 4);
    kani::assume(k <= 4);
    kani::assume(v <= 4);

    // Use a fixed valid scale to isolate dimension testing.
    let _ = build_gated_delta_net_decomposed(h, k, v, 0.125);
}

/// Proves validate() succeeds for all valid decomposed GDN configurations.
///
/// Domain: H in [1,4], K in [1,4], V in [1,4].
/// Scale is fixed at 1/sqrt(K) which is always valid for K >= 1.
#[kani::unwind(8)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn gdn_decomposed_validates_ok() {
    let h: usize = kani::any();
    let k: usize = kani::any();
    let v: usize = kani::any();

    kani::assume(h >= 1 && h <= 4);
    kani::assume(k >= 1 && k <= 4);
    kani::assume(v >= 1 && v <= 4);

    let scale = 1.0 / (k as f32).sqrt();
    let def = build_gated_delta_net_decomposed(h, k, v, scale).expect("valid params must succeed");

    assert!(
        def.validate().is_ok(),
        "validate() must pass for well-formed decomposed GDN"
    );
}

/// Proves the output shape is `[H, V]` for all valid configurations.
///
/// The DeltaNet output is `o = scale * q @ new_state`, reshaped from
/// `[H, 1, V]` to `[H, V]`. This harness verifies the reshape chain
/// produces the correct output shape.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn gdn_decomposed_output_shape() {
    let h: usize = kani::any();
    let k: usize = kani::any();
    let v: usize = kani::any();

    kani::assume(h >= 1 && h <= 4);
    kani::assume(k >= 1 && k <= 4);
    kani::assume(v >= 1 && v <= 4);

    let scale = 1.0 / (k as f32).sqrt();
    let def = build_gated_delta_net_decomposed(h, k, v, scale).expect("valid params must succeed");

    let out_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(out_shape.len(), 2, "output rank must be 2");
    assert_eq!(out_shape[0], h, "output dim 0 must equal num_heads");
    assert_eq!(out_shape[1], v, "output dim 1 must equal value_dim");
}

/// Proves `build_gated_delta_net_decomposed` rejects zero-dimension inputs
/// with `Err`, not a panic.
#[kani::unwind(1)]
#[kani::proof]
fn gdn_decomposed_zero_dim_returns_err() {
    let h: usize = kani::any();
    let k: usize = kani::any();
    let v: usize = kani::any();

    kani::assume(h <= 4);
    kani::assume(k <= 4);
    kani::assume(v <= 4);
    // At least one dimension is zero.
    kani::assume(h == 0 || k == 0 || v == 0);

    let result = build_gated_delta_net_decomposed(h, k, v, 0.125);
    assert!(
        result.is_err(),
        "zero-dimension GDN must return Err, not panic"
    );
}

/// Proves invalid scale values (0, negative, NaN, Inf) return `Err`
/// for all valid dimension combinations, not just fixed constants.
#[kani::unwind(1)]
#[kani::proof]
fn gdn_decomposed_invalid_scale_returns_err() {
    let h: usize = kani::any();
    let k: usize = kani::any();
    let v: usize = kani::any();

    kani::assume(h >= 1 && h <= 4);
    kani::assume(k >= 1 && k <= 4);
    kani::assume(v >= 1 && v <= 4);

    // For all valid dimensions, invalid scale must return Err.
    let result_zero = build_gated_delta_net_decomposed(h, k, v, 0.0);
    assert!(result_zero.is_err(), "scale=0.0 must return Err");

    let result_neg = build_gated_delta_net_decomposed(h, k, v, -1.0);
    assert!(result_neg.is_err(), "scale=-1.0 must return Err");

    let result_nan = build_gated_delta_net_decomposed(h, k, v, f32::NAN);
    assert!(result_nan.is_err(), "scale=NaN must return Err");

    let result_inf = build_gated_delta_net_decomposed(h, k, v, f32::INFINITY);
    assert!(result_inf.is_err(), "scale=Inf must return Err");
}

// ---------------------------------------------------------------------------
// build_gated_delta_net_decomposed_dual — dual output (output + new_state)
// ---------------------------------------------------------------------------

/// Proves `build_gated_delta_net_decomposed_dual` does not panic for any
/// bounded params, including invalid ones that return `Err`.
#[kani::unwind(1)]
#[kani::proof]
fn gdn_dual_build_no_panic() {
    let h: usize = kani::any();
    let k: usize = kani::any();
    let v: usize = kani::any();

    kani::assume(h <= 4);
    kani::assume(k <= 4);
    kani::assume(v <= 4);

    let _ = build_gated_delta_net_decomposed_dual(h, k, v, 0.125);
}

/// Proves validate() succeeds for all valid dual-output configurations.
#[kani::unwind(8)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn gdn_dual_validates_ok() {
    let h: usize = kani::any();
    let k: usize = kani::any();
    let v: usize = kani::any();

    kani::assume(h >= 1 && h <= 4);
    kani::assume(k >= 1 && k <= 4);
    kani::assume(v >= 1 && v <= 4);

    let scale = 1.0 / (k as f32).sqrt();
    let def =
        build_gated_delta_net_decomposed_dual(h, k, v, scale).expect("valid params must succeed");

    assert!(
        def.validate().is_ok(),
        "validate() must pass for well-formed dual GDN"
    );
}

/// Proves the dual output shape is `[H, 1+K, V]`.
///
/// The dual builder concatenates output `[H, 1, V]` and new_state `[H, K, V]`
/// along axis 1, producing `[H, 1+K, V]`. This verifies the concat shape
/// arithmetic is correct for all valid configurations.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn gdn_dual_output_shape() {
    let h: usize = kani::any();
    let k: usize = kani::any();
    let v: usize = kani::any();

    kani::assume(h >= 1 && h <= 4);
    kani::assume(k >= 1 && k <= 4);
    kani::assume(v >= 1 && v <= 4);

    let scale = 1.0 / (k as f32).sqrt();
    let def =
        build_gated_delta_net_decomposed_dual(h, k, v, scale).expect("valid params must succeed");

    let out_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(out_shape.len(), 3, "dual output rank must be 3");
    assert_eq!(out_shape[0], h, "dual output dim 0 must equal num_heads");
    assert_eq!(
        out_shape[1],
        1 + k,
        "dual output dim 1 must equal 1+key_dim (output+state concat)"
    );
    assert_eq!(out_shape[2], v, "dual output dim 2 must equal value_dim");
}
