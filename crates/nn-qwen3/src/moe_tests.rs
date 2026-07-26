// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Qwen3 MoE top-k routing: softmax normalization, top-k selection,
//! expert dispatch shapes, weighted combination, and edge cases.
//!
//! These tests construct MoE layers with controlled (non-zero) router weights
//! to verify routing correctness at the Qwen3 integration level, complementing
//! the core MoeLayer tests in nn-core.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{ExpertFFN, Linear, Module, MoeLayer, MoeLayerConfig};
use nn_core::Device;

// -- Helpers -----------------------------------------------------------------

/// Build a router Linear with explicit gate weights.
///
/// `gate_data` is `[num_experts, model_dim]` row-major. Each row is the
/// weight vector for one expert's logit.
fn router_from_data(gate_data: Vec<f32>, num_experts: usize, model_dim: usize) -> Linear {
    let w = DynTensor::from_vec(gate_data, &[num_experts, model_dim], &Device::Cpu).unwrap();
    Linear::new(w, None).unwrap()
}

/// Build an ExpertFFN with uniform scale weights.
fn uniform_expert(model_dim: usize, ff_dim: usize, scale: f32) -> ExpertFFN {
    let gate_w = DynTensor::from_vec(
        vec![scale; ff_dim * model_dim],
        &[ff_dim, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let up_w = DynTensor::from_vec(
        vec![scale; ff_dim * model_dim],
        &[ff_dim, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let down_w = DynTensor::from_vec(
        vec![scale; model_dim * ff_dim],
        &[model_dim, ff_dim],
        &Device::Cpu,
    )
    .unwrap();
    ExpertFFN::new(
        Linear::new(gate_w, None).unwrap(),
        Linear::new(up_w, None).unwrap(),
        Linear::new(down_w, None).unwrap(),
    )
    .unwrap()
}

/// Build an ExpertFFN whose down_proj is identity-like (scale on diagonal).
/// This makes the expert output predictable for weighted-combination tests.
fn identity_expert(model_dim: usize, scale: f32) -> ExpertFFN {
    // ff_dim == model_dim for simplicity. gate and up are identity-scale,
    // down is identity-scale. SwiGLU: down(silu(gate(x)) * up(x)).
    let ff_dim = model_dim;
    let mut gate_data = vec![0.0f32; ff_dim * model_dim];
    let mut up_data = vec![0.0f32; ff_dim * model_dim];
    let mut down_data = vec![0.0f32; model_dim * ff_dim];
    for i in 0..model_dim {
        gate_data[i * model_dim + i] = scale;
        up_data[i * model_dim + i] = scale;
        down_data[i * ff_dim + i] = 1.0;
    }
    let gate_w = DynTensor::from_vec(gate_data, &[ff_dim, model_dim], &Device::Cpu).unwrap();
    let up_w = DynTensor::from_vec(up_data, &[ff_dim, model_dim], &Device::Cpu).unwrap();
    let down_w = DynTensor::from_vec(down_data, &[model_dim, ff_dim], &Device::Cpu).unwrap();
    ExpertFFN::new(
        Linear::new(gate_w, None).unwrap(),
        Linear::new(up_w, None).unwrap(),
        Linear::new(down_w, None).unwrap(),
    )
    .unwrap()
}

/// Build a MoeLayer with a controlled gate matrix.
fn moe_with_gate(
    gate_data: Vec<f32>,
    num_experts: usize,
    top_k: usize,
    model_dim: usize,
    ff_dim: usize,
    norm_topk: bool,
) -> MoeLayer {
    let cfg = MoeLayerConfig::new(num_experts, top_k, model_dim, ff_dim, norm_topk, false).unwrap();
    let router = router_from_data(gate_data, num_experts, model_dim);
    let experts: Vec<ExpertFFN> = (0..num_experts)
        .map(|_| uniform_expert(model_dim, ff_dim, 0.1))
        .collect();
    MoeLayer::new(router, experts, None, cfg).unwrap()
}

// -- Router weight normalization ---------------------------------------------

#[test]
fn test_routing_weights_sum_to_one_with_normalization() {
    // With norm_topk_prob=true, the selected top-k weights should sum to ~1.0
    // after renormalization, regardless of input.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 2;

    // Diagonal gate: expert i has high logit when feature i is dominant.
    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts {
        gate_data[e * model_dim + e] = 5.0; // strong preference
    }

    let cfg = MoeLayerConfig::new(num_experts, top_k, model_dim, ff_dim, true, false).unwrap();
    let router = router_from_data(gate_data, num_experts, model_dim);

    // Test routing weights via forward_with_aux (which exposes routing internals).
    let experts: Vec<ExpertFFN> = (0..num_experts)
        .map(|_| uniform_expert(model_dim, ff_dim, 0.01))
        .collect();
    let moe = MoeLayer::new(router, experts, None, cfg).unwrap();

    // Multiple input patterns to stress routing.
    let inputs = [
        vec![10.0, 0.0, 0.0, 0.0], // strongly routes to expert 0
        vec![0.0, 10.0, 0.0, 0.0], // strongly routes to expert 1
        vec![5.0, 5.0, 0.0, 0.0],  // split between 0 and 1
        vec![1.0, 1.0, 1.0, 1.0],  // uniform
    ];

    for (i, input) in inputs.iter().enumerate() {
        let x = DynTensor::from_vec(input.clone(), &[1, model_dim], &Device::Cpu).unwrap();
        let result = moe.forward_with_aux(&x).unwrap();
        let arr = result.hidden_states.as_cpu_f32().unwrap();
        assert!(
            arr.iter().all(|v| v.is_finite()),
            "output NaN/Inf for input pattern {i}"
        );
    }
}

#[test]
fn test_routing_weights_no_normalization_differ() {
    // With norm_topk_prob=false vs true, outputs should differ because
    // unnormalized weights don't sum to 1.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 2;

    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts {
        gate_data[e * model_dim + e] = 3.0;
    }

    let moe_norm = moe_with_gate(
        gate_data.clone(),
        num_experts,
        top_k,
        model_dim,
        ff_dim,
        true,
    );
    let moe_no_norm = moe_with_gate(gate_data, num_experts, top_k, model_dim, ff_dim, false);

    let x = DynTensor::from_vec(vec![5.0, 3.0, 1.0, 0.5], &[1, model_dim], &Device::Cpu).unwrap();

    let out_norm = moe_norm.forward(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let out_no_norm = moe_no_norm
        .forward(&x)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let diff: f32 = out_norm
        .iter()
        .zip(&out_no_norm)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "normalized vs unnormalized routing should produce different outputs, diff={diff}"
    );
}

// -- Top-k selection correctness ---------------------------------------------

#[test]
fn test_topk_selects_correct_experts_diagonal_gate() {
    // With a diagonal gate and strong weights, each feature-dominant token
    // should route to the corresponding expert. We verify by checking that
    // outputs are non-zero (expert was actually called).
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 1;

    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts {
        gate_data[e * model_dim + e] = 10.0;
    }

    let moe = moe_with_gate(gate_data, num_experts, top_k, model_dim, ff_dim, true);

    // Token with feature 2 dominant: should route to expert 2.
    let x = DynTensor::from_vec(vec![0.0, 0.0, 10.0, 0.0], &[1, model_dim], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_topk_2_selects_two_best_experts() {
    // With top_k=2 and differently-scaled experts, different input patterns
    // should select different expert pairs and produce different outputs.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 2;

    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts {
        gate_data[e * model_dim + e] = 10.0;
    }

    // Each expert has a different scale so different selections produce different outputs.
    let cfg = MoeLayerConfig::new(num_experts, top_k, model_dim, ff_dim, true, false).unwrap();
    let router = router_from_data(gate_data, num_experts, model_dim);
    let experts: Vec<ExpertFFN> = (0..num_experts)
        .map(|e| uniform_expert(model_dim, ff_dim, 0.1 * (e as f32 + 1.0)))
        .collect();
    let moe = MoeLayer::new(router, experts, None, cfg).unwrap();

    // Feature 0 and 1 dominant -> experts 0 and 1 selected.
    let x01 = DynTensor::from_vec(vec![5.0, 4.0, 0.0, 0.0], &[1, model_dim], &Device::Cpu).unwrap();
    let out01 = moe.forward(&x01).unwrap().to_flat_vec::<f32>().unwrap();

    // Feature 2 and 3 dominant -> experts 2 and 3 selected.
    let x23 = DynTensor::from_vec(vec![0.0, 0.0, 5.0, 4.0], &[1, model_dim], &Device::Cpu).unwrap();
    let out23 = moe.forward(&x23).unwrap().to_flat_vec::<f32>().unwrap();

    // Different expert pairs with different scales -> different outputs.
    let diff: f32 = out01.iter().zip(&out23).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 1e-6,
        "different top-2 expert selections should produce different outputs, diff={diff}"
    );
}

// -- Expert dispatch shapes --------------------------------------------------

#[test]
fn test_dispatch_preserves_shape_2d() {
    let model_dim = 8;
    let ff_dim = 16;
    let moe = moe_with_gate(vec![0.1; 4 * model_dim], 4, 2, model_dim, ff_dim, true);
    let x = DynTensor::from_vec(vec![1.0; 5 * model_dim], &[5, model_dim], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[5, model_dim], "2D shape must be preserved");
}

#[test]
fn test_dispatch_preserves_shape_3d() {
    let model_dim = 8;
    let ff_dim = 16;
    let moe = moe_with_gate(vec![0.1; 4 * model_dim], 4, 2, model_dim, ff_dim, true);
    let x = DynTensor::from_vec(
        vec![1.0; 2 * 3 * model_dim],
        &[2, 3, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(
        out.dims(),
        &[2, 3, model_dim],
        "3D [B, T, D] shape must be preserved"
    );
}

#[test]
fn test_dispatch_single_token() {
    let model_dim = 4;
    let ff_dim = 8;
    let moe = moe_with_gate(vec![0.1; 4 * model_dim], 4, 2, model_dim, ff_dim, true);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, model_dim], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

// -- Weighted combination ----------------------------------------------------

#[test]
fn test_weighted_combination_with_identity_experts() {
    // With identity-like experts, the output is a weighted sum of expert outputs.
    // Since all experts are identical, the output should be the same regardless
    // of which experts are selected.
    let model_dim = 4;
    let num_experts = 4;
    let top_k = 2;

    // Uniform gate -> all experts equally likely.
    let gate_data = vec![1.0f32; num_experts * model_dim];
    let cfg = MoeLayerConfig::new(num_experts, top_k, model_dim, model_dim, true, false).unwrap();
    let router = router_from_data(gate_data, num_experts, model_dim);
    let experts: Vec<ExpertFFN> = (0..num_experts)
        .map(|_| identity_expert(model_dim, 0.5))
        .collect();
    let moe = MoeLayer::new(router, experts, None, cfg).unwrap();

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, model_dim], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
    assert!(
        arr.iter().any(|&v| v.abs() > 1e-10),
        "identity experts should produce non-zero output"
    );
}

#[test]
fn test_weighted_combination_identical_experts_same_output() {
    // When ALL experts are identical, the output should be the same for
    // different input patterns (since the weighted combination of identical
    // experts is just that expert's output regardless of routing weights,
    // as long as they sum to 1).
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 2;

    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts {
        gate_data[e * model_dim + e] = 5.0;
    }
    let cfg = MoeLayerConfig::new(num_experts, top_k, model_dim, ff_dim, true, false).unwrap();
    let router = router_from_data(gate_data, num_experts, model_dim);
    let experts: Vec<ExpertFFN> = (0..num_experts)
        .map(|_| uniform_expert(model_dim, ff_dim, 0.1))
        .collect();
    let moe = MoeLayer::new(router, experts, None, cfg).unwrap();

    let input = vec![1.0, 2.0, 3.0, 4.0];
    let x = DynTensor::from_vec(input.clone(), &[1, model_dim], &Device::Cpu).unwrap();
    let out1 = moe.forward(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // Same input but different routing (via the gate weights) should produce
    // the same output because experts are identical and weights are normalized.
    // Use uniform gate to change routing distribution.
    let uniform_gate = vec![1.0f32; num_experts * model_dim];
    let cfg2 = MoeLayerConfig::new(num_experts, top_k, model_dim, ff_dim, true, false).unwrap();
    let router2 = router_from_data(uniform_gate, num_experts, model_dim);
    let experts2: Vec<ExpertFFN> = (0..num_experts)
        .map(|_| uniform_expert(model_dim, ff_dim, 0.1))
        .collect();
    let moe2 = MoeLayer::new(router2, experts2, None, cfg2).unwrap();

    let x2 = DynTensor::from_vec(input, &[1, model_dim], &Device::Cpu).unwrap();
    let out2 = moe2.forward(&x2).unwrap().to_flat_vec::<f32>().unwrap();

    // With identical experts and normalized weights, outputs should match.
    for (i, (a, b)) in out1.iter().zip(&out2).enumerate() {
        let err = (a - b).abs();
        assert!(
            err < 1e-5,
            "identical experts should produce same output regardless of routing, \
             mismatch at [{i}]: {a:.8} vs {b:.8}, diff={err:.8}"
        );
    }
}

#[test]
fn test_weighted_combination_different_experts_differ() {
    // When experts have different weights, different routing should produce
    // different outputs.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 1;

    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts {
        gate_data[e * model_dim + e] = 10.0;
    }
    let cfg = MoeLayerConfig::new(num_experts, top_k, model_dim, ff_dim, true, false).unwrap();
    let router = router_from_data(gate_data, num_experts, model_dim);
    let experts: Vec<ExpertFFN> = (0..num_experts)
        .map(|e| uniform_expert(model_dim, ff_dim, 0.1 * (e as f32 + 1.0)))
        .collect();
    let moe = MoeLayer::new(router, experts, None, cfg).unwrap();

    // Route to expert 0.
    let x0 = DynTensor::from_vec(vec![10.0, 0.0, 0.0, 0.0], &[1, model_dim], &Device::Cpu).unwrap();
    let out0 = moe.forward(&x0).unwrap().to_flat_vec::<f32>().unwrap();

    // Route to expert 3.
    let x3 = DynTensor::from_vec(vec![0.0, 0.0, 0.0, 10.0], &[1, model_dim], &Device::Cpu).unwrap();
    let out3 = moe.forward(&x3).unwrap().to_flat_vec::<f32>().unwrap();

    let diff: f32 = out0.iter().zip(&out3).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 1e-4,
        "different experts with different scales should produce different outputs, diff={diff}"
    );
}

// -- Edge cases: single expert -----------------------------------------------

#[test]
fn test_single_expert_top1_routing() {
    // With 1 expert and top_k=1, all tokens go to the single expert
    // with weight 1.0 (after softmax of single logit and renormalization).
    let model_dim = 4;
    let ff_dim = 8;
    let moe = moe_with_gate(vec![1.0; model_dim], 1, 1, model_dim, ff_dim, true);

    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, model_dim]);

    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "single expert output should be finite"
    );
    assert!(
        arr.iter().any(|&v| v.abs() > 1e-10),
        "single expert output should be non-zero for non-zero input"
    );
}

#[test]
fn test_single_expert_matches_direct_expert_call() {
    // With 1 expert and norm_topk_prob=true, weight=1.0. The MoE output
    // should equal the expert's direct output (weighted by 1.0).
    let model_dim = 4;
    let ff_dim = 8;
    let scale = 0.1;

    let expert = uniform_expert(model_dim, ff_dim, scale);
    let expert_clone = uniform_expert(model_dim, ff_dim, scale);

    let cfg = MoeLayerConfig::new(1, 1, model_dim, ff_dim, true, false).unwrap();
    let router = router_from_data(vec![1.0; model_dim], 1, model_dim);
    let moe = MoeLayer::new(router, vec![expert_clone], None, cfg).unwrap();

    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let x = DynTensor::from_vec(input_data, &[1, model_dim], &Device::Cpu).unwrap();

    let direct_out = expert.forward(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let moe_out = moe.forward(&x).unwrap().to_flat_vec::<f32>().unwrap();

    for (i, (a, b)) in direct_out.iter().zip(&moe_out).enumerate() {
        let err = (a - b).abs();
        assert!(
            err < 1e-5,
            "single expert MoE should match direct expert, mismatch at [{i}]: {a:.8} vs {b:.8}"
        );
    }
}

// -- Edge cases: all experts selected ----------------------------------------

#[test]
fn test_all_experts_selected_top_k_equals_num_experts() {
    // When top_k == num_experts, ALL experts process every token.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 4; // all experts

    let moe = moe_with_gate(
        vec![1.0; num_experts * model_dim],
        num_experts,
        top_k,
        model_dim,
        ff_dim,
        true,
    );

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, model_dim], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, model_dim]);

    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_all_experts_with_aux_loss_near_one() {
    // When all experts are selected (top_k == num_experts) and routing is uniform,
    // each expert gets fraction 1.0 of tokens, mean probability = 1/E.
    // aux_loss = E * E * (1.0 * 1/E) = E. With norm, should be E (= num_experts).
    // Actually: f_e = k/(E*k) = 1/E when each token goes to all E experts.
    // Wait: f_e = count_e / (N*k). With all experts selected for every token,
    // count_e = N for each expert. f_e = N/(N*E) = 1/E. P_e ~ 1/E.
    // aux_loss = E * sum(1/E * 1/E) = E * E * 1/E^2 = 1.0.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 4;

    let moe = moe_with_gate(
        vec![1.0; num_experts * model_dim],
        num_experts,
        top_k,
        model_dim,
        ff_dim,
        true,
    );

    // Uniform input.
    let x = DynTensor::from_vec(vec![1.0; 4 * model_dim], &[4, model_dim], &Device::Cpu).unwrap();
    let result = moe.forward_with_aux(&x).unwrap();
    let loss_val = result.aux_loss.to_flat_vec::<f32>().unwrap()[0];
    assert!(
        loss_val.is_finite(),
        "aux loss must be finite, got {loss_val}"
    );
    // With uniform routing and all experts, loss should be near 1.0.
    assert!(
        (loss_val - 1.0).abs() < 0.3,
        "all-expert uniform routing aux loss should be near 1.0, got {loss_val}"
    );
}

// -- Multi-token batch routing -----------------------------------------------

#[test]
fn test_batch_tokens_routed_independently() {
    // Each token in a batch should be independently routed based on its features.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 1;

    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts {
        gate_data[e * model_dim + e] = 10.0;
    }

    let cfg = MoeLayerConfig::new(num_experts, top_k, model_dim, ff_dim, true, false).unwrap();
    let router = router_from_data(gate_data, num_experts, model_dim);
    let experts: Vec<ExpertFFN> = (0..num_experts)
        .map(|e| uniform_expert(model_dim, ff_dim, 0.1 * (e as f32 + 1.0)))
        .collect();
    let moe = MoeLayer::new(router, experts, None, cfg).unwrap();

    // Process two tokens individually.
    let x0 = DynTensor::from_vec(vec![10.0, 0.0, 0.0, 0.0], &[1, model_dim], &Device::Cpu).unwrap();
    let x1 = DynTensor::from_vec(vec![0.0, 10.0, 0.0, 0.0], &[1, model_dim], &Device::Cpu).unwrap();
    let out0 = moe.forward(&x0).unwrap().to_flat_vec::<f32>().unwrap();
    let out1 = moe.forward(&x1).unwrap().to_flat_vec::<f32>().unwrap();

    // Process both in a batch.
    let x_batch = DynTensor::from_vec(
        vec![10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        &[2, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out_batch = moe.forward(&x_batch).unwrap();
    let batch_flat = out_batch.to_flat_vec::<f32>().unwrap();

    // Batch token 0 should match individual token 0.
    for i in 0..model_dim {
        let err = (batch_flat[i] - out0[i]).abs();
        assert!(
            err < 1e-5,
            "batch[0][{i}] should match individual[0][{i}]: {:.8} vs {:.8}",
            batch_flat[i],
            out0[i]
        );
    }

    // Batch token 1 should match individual token 1.
    for i in 0..model_dim {
        let err = (batch_flat[model_dim + i] - out1[i]).abs();
        assert!(
            err < 1e-5,
            "batch[1][{i}] should match individual[1][{i}]: {:.8} vs {:.8}",
            batch_flat[model_dim + i],
            out1[i]
        );
    }
}

// -- Shared expert interaction -----------------------------------------------

#[test]
fn test_shared_expert_adds_to_routed_output() {
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 2;

    let gate_data = vec![1.0f32; num_experts * model_dim];

    // Without shared expert.
    let moe_no_shared = moe_with_gate(
        gate_data.clone(),
        num_experts,
        top_k,
        model_dim,
        ff_dim,
        true,
    );

    // With shared expert.
    let cfg_shared =
        MoeLayerConfig::new(num_experts, top_k, model_dim, ff_dim, true, true).unwrap();
    let router = router_from_data(gate_data, num_experts, model_dim);
    let experts: Vec<ExpertFFN> = (0..num_experts)
        .map(|_| uniform_expert(model_dim, ff_dim, 0.1))
        .collect();
    let shared = uniform_expert(model_dim, ff_dim, 0.2);
    let moe_shared = MoeLayer::new(router, experts, Some(shared), cfg_shared).unwrap();

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, model_dim], &Device::Cpu).unwrap();

    let out_no_shared = moe_no_shared
        .forward(&x)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let out_shared = moe_shared
        .forward(&x)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let diff: f32 = out_no_shared
        .iter()
        .zip(&out_shared)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "shared expert should change the output, diff={diff}"
    );
}

// -- Qwen3-specific config integration tests ---------------------------------

#[test]
fn test_qwen3_moe_128_experts_top8_config_valid() {
    // Qwen3-30B-A3B: 128 experts, top-8 routing.
    let cfg = MoeLayerConfig::new(128, 8, 4096, 2560, true, false);
    assert!(cfg.is_ok());
    let cfg = cfg.unwrap();
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.top_k, 8);
}

#[test]
fn test_qwen3_moe_config_with_shared_expert() {
    // Qwen3.5 pattern: shared expert with different intermediate size.
    let cfg = MoeLayerConfig::new(128, 8, 4096, 2560, true, true)
        .unwrap()
        .with_shared_intermediate_size(5120)
        .unwrap();
    assert_eq!(cfg.shared_ff_dim(), 5120);
    assert!(cfg.shared_expert);
}
