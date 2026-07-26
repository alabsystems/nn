// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for monolithic `add_gated_delta_net` builder.
//!
//! Proves structural correctness of the monolithic GatedDeltaNet tensor IR node:
//! - validate() succeeds for all valid bounded parameters
//! - Output shape is `[H, V]`
//! - 6 inputs + 1 GatedDeltaNet op = 7 nodes
//!
//! The monolithic builder creates a single `TensorOpKind::GatedDeltaNet` node,
//! which is translated by `translate_gated_delta_net` in nn-verify to 9
//! NY nodes. These harnesses verify the IR-level construction.
//!
//! Part of #834 — Gated DeltaNet for Qwen3.5 model support.

use crate::tensor_block_builder::TensorBlockBuilder;

fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
    }
    r
}

/// Proves monolithic `add_gated_delta_net` + `build` + `validate()` succeeds
/// for all valid bounded parameter combinations.
///
/// Domain: H in [1,4], K in [1,4], V in [1,4].
/// Scale = 1/sqrt(K), which is always finite and positive for K >= 1.
#[kani::unwind(8)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn gdn_monolithic_validates_ok() {
    let h: usize = kani::any();
    let k: usize = kani::any();
    let v: usize = kani::any();

    kani::assume(h >= 1 && h <= 4);
    kani::assume(k >= 1 && k <= 4);
    kani::assume(v >= 1 && v <= 4);

    let scale = 1.0 / (k as f32).sqrt();

    let mut b = TensorBlockBuilder::new("kani_gdn");
    let q = b.add_input("q", &[h, k]);
    let ki = b.add_input("k", &[h, k]);
    let vi = b.add_input("v", &[h, v]);
    let state = b.add_input("state", &[h, k, v]);
    let gate = b.add_input("gate", &[h, 1, 1]);
    let beta = b.add_input("beta", &[h, 1]);

    let out = b.add_gated_delta_net(q, ki, vi, state, gate, beta, scale, &[h, v]);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_ok(),
        "validate() must pass for well-formed monolithic GDN"
    );
}

/// Proves the monolithic GDN output shape is `[H, V]`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn gdn_monolithic_output_shape() {
    let h: usize = kani::any();
    let k: usize = kani::any();
    let v: usize = kani::any();

    kani::assume(h >= 1 && h <= 4);
    kani::assume(k >= 1 && k <= 4);
    kani::assume(v >= 1 && v <= 4);

    let scale = 1.0 / (k as f32).sqrt();

    let mut b = TensorBlockBuilder::new("kani_gdn_shape");
    let q = b.add_input("q", &[h, k]);
    let ki = b.add_input("k", &[h, k]);
    let vi = b.add_input("v", &[h, v]);
    let state = b.add_input("state", &[h, k, v]);
    let gate = b.add_input("gate", &[h, 1, 1]);
    let beta = b.add_input("beta", &[h, 1]);

    let out = b.add_gated_delta_net(q, ki, vi, state, gate, beta, scale, &[h, v]);
    let def = b.build(out).expect("valid graph");

    let out_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(out_shape.len(), 2, "output rank must be 2");
    assert_eq!(out_shape[0], h, "output dim 0 must equal num_heads");
    assert_eq!(out_shape[1], v, "output dim 1 must equal value_dim");
}

/// Proves the monolithic GDN has exactly 7 nodes: 6 inputs + 1 op.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn gdn_monolithic_node_count() {
    let h: usize = kani::any();
    let k: usize = kani::any();
    let v: usize = kani::any();

    kani::assume(h >= 1 && h <= 4);
    kani::assume(k >= 1 && k <= 4);
    kani::assume(v >= 1 && v <= 4);

    let scale = 1.0 / (k as f32).sqrt();

    let mut b = TensorBlockBuilder::new("kani_gdn_count");
    let q = b.add_input("q", &[h, k]);
    let ki = b.add_input("k", &[h, k]);
    let vi = b.add_input("v", &[h, v]);
    let state = b.add_input("state", &[h, k, v]);
    let gate = b.add_input("gate", &[h, 1, 1]);
    let beta = b.add_input("beta", &[h, 1]);

    let out = b.add_gated_delta_net(q, ki, vi, state, gate, beta, scale, &[h, v]);
    let def = b.build(out).expect("valid graph");

    assert_eq!(
        def.nodes.len(),
        7,
        "monolithic GDN must have exactly 7 nodes (6 inputs + 1 GatedDeltaNet op)"
    );
}
