#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MoE value-correctness tests verifying actual output values against expected results.
//!
//! These tests exercise the core dispatch+accumulate logic with known expected outputs,
//! catching accumulation order bugs, scale errors, and off-by-one in dispatch_expert.

use crate::dyn_tensor::DynTensor;
use crate::layers::{Linear, Module};
use crate::Device;

use super::{make_router, MoeLayer, SwiGluExpert};

/// Create a SwiGluExpert with a specific scalar weight for all projections.
/// Different scalar values produce different expert outputs, enabling
/// value-correctness tests that distinguish between experts.
fn make_scaled_expert(dim: usize, ff_dim: usize, scale: f32) -> SwiGluExpert {
    let gate_w =
        DynTensor::from_vec(vec![scale; ff_dim * dim], &[ff_dim, dim], &Device::Cpu).unwrap();
    let up_w =
        DynTensor::from_vec(vec![scale; ff_dim * dim], &[ff_dim, dim], &Device::Cpu).unwrap();
    let down_w =
        DynTensor::from_vec(vec![scale; dim * ff_dim], &[dim, ff_dim], &Device::Cpu).unwrap();
    SwiGluExpert::new(
        Linear::new(gate_w, None).unwrap(),
        Linear::new(up_w, None).unwrap(),
        Linear::new(down_w, None).unwrap(),
    )
    .unwrap()
}

/// Verify MoE output matches direct expert output for a deterministic single-expert route.
///
/// With top_k=1 and an identity gate, input [1.0, 0.0] routes entirely to expert 0
/// (renormalized weight = 1.0), so MoE output == expert_0.forward(input).
/// Uses experts with different weights so routing to the wrong expert would fail.
#[test]
fn test_moe_value_correctness_single_route() {
    let model_dim = 2;
    let ff_dim = 4;
    let num_experts = 2;
    let top_k = 1;

    // Two experts with DIFFERENT weights — ensures routing matters
    let expert_0 = make_scaled_expert(model_dim, ff_dim, 0.1);
    let expert_1 = make_scaled_expert(model_dim, ff_dim, 0.5);

    let x = DynTensor::from_vec(vec![1.0, 0.0], &[1, model_dim], &Device::Cpu).unwrap();

    // Compute expert_0's direct output for the test input
    let expected_output = expert_0.forward(&x).unwrap();
    let expected_vals = expected_output.to_flat_vec::<f32>().unwrap();

    // Sanity: experts produce different outputs
    let other_output = expert_1.forward(&x).unwrap();
    let other_vals = other_output.to_flat_vec::<f32>().unwrap();
    assert!(
        expected_vals
            .iter()
            .zip(other_vals.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "experts must produce different outputs for the test to be meaningful"
    );

    // Build MoE layer with same expert instances (moved in).
    // Identity gate: input [1.0, 0.0] -> expert 0 logit=1.0, expert 1 logit=0.0
    // softmax -> [0.73, 0.27], topk=1 selects expert 0, renormalized weight = 1.0
    let router = make_router(num_experts, top_k, model_dim);
    let moe = MoeLayer::new(router, vec![expert_0, expert_1], None).unwrap();
    let moe_output = moe.forward(&x).unwrap();
    let moe_vals = moe_output.to_flat_vec::<f32>().unwrap();

    assert_eq!(moe_vals.len(), expected_vals.len());
    for i in 0..moe_vals.len() {
        let err = (moe_vals[i] - expected_vals[i]).abs();
        assert!(
            err < 1e-6,
            "MoE value mismatch at [{i}]: moe={:.8}, expected={:.8}, err={:.8}",
            moe_vals[i],
            expected_vals[i],
            err,
        );
    }
}

/// Verify MoE correctly weights two experts when top_k=2 and both are active.
///
/// Uses experts with different weight scales (0.1 vs 0.3) so the weighted sum
/// is non-trivial. Clones experts to compute reference outputs from the same
/// instances that are moved into MoE.
#[test]
fn test_moe_value_correctness_weighted_sum() {
    let model_dim = 2;
    let ff_dim = 4;
    let num_experts = 2;
    let top_k = 2; // both experts active

    // Two experts with DIFFERENT weights
    let expert_0 = make_scaled_expert(model_dim, ff_dim, 0.1);
    let expert_1 = make_scaled_expert(model_dim, ff_dim, 0.3);

    let x = DynTensor::from_vec(vec![1.0, 0.0], &[1, model_dim], &Device::Cpu).unwrap();

    // Compute reference outputs before moving experts into MoE
    // forward() takes &self, so no clone needed
    let out_0 = expert_0.forward(&x).unwrap();
    let out_1 = expert_1.forward(&x).unwrap();
    let vals_0 = out_0.to_flat_vec::<f32>().unwrap();
    let vals_1 = out_1.to_flat_vec::<f32>().unwrap();

    // Experts must produce different outputs for this test to be meaningful
    assert!(
        vals_0
            .iter()
            .zip(vals_1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "experts must produce different outputs, got: {vals_0:?} vs {vals_1:?}"
    );

    // Get routing weights and indices from a separate router instance
    let router_ref = make_router(num_experts, top_k, model_dim);
    let routing = router_ref.forward(&x).unwrap();
    let routing_weights = routing.weights.to_flat_vec::<f32>().unwrap();
    let routing_indices = routing.indices.as_cpu_u32().unwrap();

    // Map weights to expert indices (topk returns in descending-weight order)
    let expert_outputs = [&vals_0, &vals_1];
    let expected: Vec<f32> = (0..model_dim)
        .map(|d| {
            (0..top_k)
                .map(|k| {
                    let w = routing_weights[k];
                    let expert_idx = routing_indices[[0, k]] as usize;
                    w * expert_outputs[expert_idx][d]
                })
                .sum::<f32>()
        })
        .collect();

    // Build MoE with the SAME expert instances (not recreated copies)
    let router = make_router(num_experts, top_k, model_dim);
    let moe = MoeLayer::new(router, vec![expert_0, expert_1], None).unwrap();
    let moe_output = moe.forward(&x).unwrap();
    let moe_vals = moe_output.to_flat_vec::<f32>().unwrap();

    assert_eq!(moe_vals.len(), expected.len());
    for i in 0..moe_vals.len() {
        let err = (moe_vals[i] - expected[i]).abs();
        assert!(
            err < 1e-5,
            "MoE weighted sum mismatch at [{i}]: moe={:.8}, expected={:.8}, weights={routing_weights:?}, err={:.8}",
            moe_vals[i],
            expected[i],
            err,
        );
    }
}

/// Verify MoE with shared expert adds shared output to routed output.
///
/// Shared expert output is unconditionally added to every token's output.
/// Result = routed_output + shared_expert(x).
#[test]
fn test_moe_value_correctness_with_shared_expert() {
    let model_dim = 2;
    let ff_dim = 4;
    let num_experts = 2;
    let top_k = 1;

    let expert_0 = make_scaled_expert(model_dim, ff_dim, 0.1);
    let expert_1 = make_scaled_expert(model_dim, ff_dim, 0.5);
    let shared = make_scaled_expert(model_dim, ff_dim, 0.2);

    let x = DynTensor::from_vec(vec![1.0, 0.0], &[1, model_dim], &Device::Cpu).unwrap();

    // Reference: expert_0 output (routed, topk=1 selects expert 0) + shared output
    let routed_ref = expert_0.forward(&x).unwrap();
    let shared_ref = shared.forward(&x).unwrap();
    let routed_vals = routed_ref.to_flat_vec::<f32>().unwrap();
    let shared_vals = shared_ref.to_flat_vec::<f32>().unwrap();
    let expected_vals: Vec<f32> = routed_vals
        .iter()
        .zip(shared_vals.iter())
        .map(|(r, s)| r + s)
        .collect();

    // Build MoE with shared expert using same instances
    let router = make_router(num_experts, top_k, model_dim);
    let moe = MoeLayer::new(router, vec![expert_0, expert_1], Some(shared)).unwrap();
    let moe_output = moe.forward(&x).unwrap();
    let moe_vals = moe_output.to_flat_vec::<f32>().unwrap();

    assert_eq!(moe_vals.len(), expected_vals.len());
    for i in 0..moe_vals.len() {
        let err = (moe_vals[i] - expected_vals[i]).abs();
        assert!(
            err < 1e-6,
            "MoE+shared mismatch at [{i}]: moe={:.8}, expected={:.8}, err={:.8}",
            moe_vals[i],
            expected_vals[i],
            err,
        );
    }
}
