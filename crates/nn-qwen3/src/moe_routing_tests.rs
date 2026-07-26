// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MoE routing and dispatch tests for Qwen3: load balancing, capacity stress,
//! aux loss as balance signal, edge cases (all-same-expert, many-experts-few-tokens).
//!
//! Complements `moe_tests.rs` (router weights, top-k selection, shapes, weighted
//! combination) with tests specifically targeting routing correctness under
//! adversarial and stress conditions.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{ExpertFFN, Linear, Module, MoeLayer, MoeLayerConfig};
use nn_core::Device;

// -- Helpers ------------------------------------------------------------------

/// Build a router Linear with explicit gate weights.
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

/// Build a MoeLayer where each expert has a distinct scale, allowing us to
/// detect which expert(s) processed each token.
fn moe_with_distinct_experts(
    gate_data: Vec<f32>,
    num_experts: usize,
    top_k: usize,
    model_dim: usize,
    ff_dim: usize,
) -> MoeLayer {
    let cfg = MoeLayerConfig::new(num_experts, top_k, model_dim, ff_dim, true, false).unwrap();
    let router = router_from_data(gate_data, num_experts, model_dim);
    let experts: Vec<ExpertFFN> = (0..num_experts)
        .map(|e| uniform_expert(model_dim, ff_dim, 0.1 * (e as f32 + 1.0)))
        .collect();
    MoeLayer::new(router, experts, None, cfg).unwrap()
}

// -- Load balancing via aux loss -----------------------------------------------

#[test]
fn test_aux_loss_minimal_with_uniform_routing() {
    // With a uniform gate (all experts equally likely) and uniform input,
    // the aux loss should be near 1.0 (perfectly balanced). The formula is
    // num_experts * sum_e(f_e * P_e). With uniform: f_e = 1/E, P_e = 1/E,
    // aux_loss = E * E * (1/E)^2 = 1.0.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 2;

    // Uniform gate: all entries equal.
    let gate_data = vec![1.0f32; num_experts * model_dim];
    let moe = moe_with_gate(gate_data, num_experts, top_k, model_dim, ff_dim, true);

    // Many uniform tokens for statistical stability.
    let n_tokens = 32;
    let x = DynTensor::from_vec(
        vec![1.0f32; n_tokens * model_dim],
        &[n_tokens, model_dim],
        &Device::Cpu,
    )
    .unwrap();

    let result = moe.forward_with_aux(&x).unwrap();
    let loss_val = result.aux_loss.to_flat_vec::<f32>().unwrap()[0];
    assert!(
        loss_val.is_finite(),
        "aux loss must be finite, got {loss_val}"
    );
    // With uniform routing, aux loss converges to 1.0.
    assert!(
        (loss_val - 1.0).abs() < 0.2,
        "uniform routing aux loss should be near 1.0, got {loss_val}"
    );
}

#[test]
fn test_aux_loss_higher_with_skewed_routing() {
    // When routing is heavily skewed (one expert gets most tokens), aux loss
    // should be higher than with uniform routing.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 1;

    // Skewed gate: expert 0 has very high logit for all features.
    let mut skewed_gate = vec![0.0f32; num_experts * model_dim];
    for d in 0..model_dim {
        skewed_gate[0 * model_dim + d] = 100.0; // expert 0 dominates
    }
    let moe_skewed = moe_with_gate(skewed_gate, num_experts, top_k, model_dim, ff_dim, true);

    // Uniform gate for comparison.
    let uniform_gate = vec![1.0f32; num_experts * model_dim];
    let moe_uniform = moe_with_gate(uniform_gate, num_experts, top_k, model_dim, ff_dim, true);

    let n_tokens = 16;
    let x = DynTensor::from_vec(
        vec![1.0f32; n_tokens * model_dim],
        &[n_tokens, model_dim],
        &Device::Cpu,
    )
    .unwrap();

    let skewed_loss = moe_skewed
        .forward_with_aux(&x)
        .unwrap()
        .aux_loss
        .to_flat_vec::<f32>()
        .unwrap()[0];
    let uniform_loss = moe_uniform
        .forward_with_aux(&x)
        .unwrap()
        .aux_loss
        .to_flat_vec::<f32>()
        .unwrap()[0];

    assert!(
        skewed_loss > uniform_loss,
        "skewed routing ({skewed_loss:.6}) should have higher aux loss than uniform ({uniform_loss:.6})"
    );
}

// -- Capacity / overflow: many tokens, few experts ----------------------------

#[test]
fn test_many_tokens_few_experts_no_overflow() {
    // Stress test: 128 tokens with only 2 experts, top-1. Every token must be
    // processed and the output shape must be preserved.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 2;
    let top_k = 1;

    let gate_data = vec![1.0f32; num_experts * model_dim];
    let moe = moe_with_gate(gate_data, num_experts, top_k, model_dim, ff_dim, true);

    let n_tokens = 128;
    let x = DynTensor::from_vec(
        vec![1.0f32; n_tokens * model_dim],
        &[n_tokens, model_dim],
        &Device::Cpu,
    )
    .unwrap();

    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[n_tokens, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "all outputs must be finite with 128 tokens / 2 experts"
    );
}

#[test]
fn test_many_tokens_many_experts_top_k() {
    // 64 tokens, 16 experts, top-4. Tests that scatter-gather handles
    // high fan-out correctly (each token dispatched to 4 experts).
    let model_dim = 8;
    let ff_dim = 16;
    let num_experts = 16;
    let top_k = 4;

    let gate_data = vec![0.1f32; num_experts * model_dim];
    let moe = moe_with_gate(gate_data, num_experts, top_k, model_dim, ff_dim, true);

    let n_tokens = 64;
    let x = DynTensor::from_vec(
        vec![1.0f32; n_tokens * model_dim],
        &[n_tokens, model_dim],
        &Device::Cpu,
    )
    .unwrap();

    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[n_tokens, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

// -- Edge case: all tokens route to same expert --------------------------------

#[test]
fn test_all_tokens_same_expert_output_finite() {
    // Extreme skew: gate forces ALL tokens to expert 0. Output must still be
    // finite and have correct shape. Other experts get zero tokens.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 1;

    // Expert 0 has massive logit for every feature.
    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for d in 0..model_dim {
        gate_data[0 * model_dim + d] = 100.0;
    }

    let moe = moe_with_gate(gate_data, num_experts, top_k, model_dim, ff_dim, true);

    let n_tokens = 8;
    let x = DynTensor::from_vec(
        vec![1.0f32; n_tokens * model_dim],
        &[n_tokens, model_dim],
        &Device::Cpu,
    )
    .unwrap();

    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[n_tokens, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "all-same-expert routing must produce finite output"
    );
    assert!(
        arr.iter().any(|&v| v.abs() > 1e-10),
        "all-same-expert should produce non-zero output for non-zero input"
    );
}

#[test]
fn test_all_tokens_same_expert_with_distinct_experts() {
    // When all tokens route to expert 0 (via extreme gate bias) with top-1,
    // the output should differ from when they all route to expert 3.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 1;

    // Gate biased to expert 0.
    let mut gate_e0 = vec![0.0f32; num_experts * model_dim];
    for d in 0..model_dim {
        gate_e0[0 * model_dim + d] = 100.0;
    }
    let moe_e0 = moe_with_distinct_experts(gate_e0, num_experts, top_k, model_dim, ff_dim);

    // Gate biased to expert 3.
    let mut gate_e3 = vec![0.0f32; num_experts * model_dim];
    for d in 0..model_dim {
        gate_e3[3 * model_dim + d] = 100.0;
    }
    let moe_e3 = moe_with_distinct_experts(gate_e3, num_experts, top_k, model_dim, ff_dim);

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, model_dim], &Device::Cpu).unwrap();

    let out_e0 = moe_e0.forward(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let out_e3 = moe_e3.forward(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let diff: f32 = out_e0.iter().zip(&out_e3).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 1e-4,
        "routing to expert 0 vs expert 3 (different scales) must differ, diff={diff}"
    );
}

// -- Edge case: many experts, single token ------------------------------------

#[test]
fn test_single_token_many_experts_top1() {
    // 1 token, 16 experts, top-1. Only one expert should process it.
    let model_dim = 8;
    let ff_dim = 16;
    let num_experts = 16;
    let top_k = 1;

    let gate_data = vec![0.1f32; num_experts * model_dim];
    let moe = moe_with_gate(gate_data, num_experts, top_k, model_dim, ff_dim, true);

    let x = DynTensor::from_vec(vec![1.0f32; model_dim], &[1, model_dim], &Device::Cpu).unwrap();

    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_single_token_many_experts_top_half() {
    // 1 token, 8 experts, top-4. Token is dispatched to 4 of 8 experts.
    let model_dim = 8;
    let ff_dim = 16;
    let num_experts = 8;
    let top_k = 4;

    let gate_data = vec![0.1f32; num_experts * model_dim];
    let moe = moe_with_gate(gate_data, num_experts, top_k, model_dim, ff_dim, true);

    let x = DynTensor::from_vec(vec![1.0f32; model_dim], &[1, model_dim], &Device::Cpu).unwrap();

    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
    assert!(
        arr.iter().any(|&v| v.abs() > 1e-10),
        "top-4 of 8 experts should produce non-zero output"
    );
}

// -- Expert dispatch correctness: verify correct expert applied ----------------

#[test]
fn test_expert_dispatch_deterministic_same_input() {
    // Same input must always produce exactly the same output (deterministic routing).
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 2;

    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts {
        gate_data[e * model_dim + e] = 5.0;
    }

    let moe = moe_with_distinct_experts(gate_data, num_experts, top_k, model_dim, ff_dim);

    let input = vec![3.0, 1.0, 2.0, 0.5];
    let x = DynTensor::from_vec(input.clone(), &[1, model_dim], &Device::Cpu).unwrap();

    let out1 = moe.forward(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let x2 = DynTensor::from_vec(input, &[1, model_dim], &Device::Cpu).unwrap();
    let out2 = moe.forward(&x2).unwrap().to_flat_vec::<f32>().unwrap();

    assert_eq!(
        out1, out2,
        "MoE routing must be deterministic for same input"
    );
}

#[test]
fn test_expert_dispatch_per_token_independence() {
    // Changing one token in a batch must not affect the output of other tokens.
    // (Within the scatter-gather dispatch, tokens are independent.)
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 1;

    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts {
        gate_data[e * model_dim + e] = 10.0;
    }
    let moe = moe_with_distinct_experts(gate_data, num_experts, top_k, model_dim, ff_dim);

    // Batch A: [token_fixed, token_a]
    let batch_a = DynTensor::from_vec(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        &[2, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out_a = moe.forward(&batch_a).unwrap().to_flat_vec::<f32>().unwrap();

    // Batch B: [token_fixed, token_b] (different second token)
    let batch_b = DynTensor::from_vec(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
        &[2, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out_b = moe.forward(&batch_b).unwrap().to_flat_vec::<f32>().unwrap();

    // First token's output should be identical in both batches.
    for i in 0..model_dim {
        let err = (out_a[i] - out_b[i]).abs();
        assert!(
            err < 1e-6,
            "changing token 1 should not affect token 0: out_a[{i}]={:.8} vs out_b[{i}]={:.8}, diff={err:.8}",
            out_a[i], out_b[i]
        );
    }

    // Second token's output should differ (different routing).
    let diff: f32 = (0..model_dim)
        .map(|i| (out_a[model_dim + i] - out_b[model_dim + i]).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "different second tokens should produce different outputs, diff={diff}"
    );
}

// -- 3D batch shape through full MoE model ------------------------------------

#[test]
fn test_moe_model_3d_batch_shape_preserved() {
    // The Qwen3MoeModel forward path reshapes [B, T, D] -> [B*T, D] for MoE
    // dispatch and back. Verify the final output shape is correct.
    use crate::moe::{Qwen3MoeConfig, Qwen3MoeModel};
    use crate::Qwen3Config;
    use nn_core::var_builder::VarBuilder;
    use nn_core::DType;

    let cfg = Qwen3MoeConfig {
        base: Qwen3Config {
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            vocab_size: 50,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 32,
            tie_word_embeddings: true,
            rope_scaling: None,
        },
        num_experts: 4,
        num_experts_per_tok: 2,
        shared_expert: false,
        shared_expert_intermediate_size: None,
    };

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    // Forward with 3 tokens.
    let logits = model.forward(&[1, 2, 3], &[0, 1, 2]).unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.base.vocab_size]);
    let arr = logits.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

// -- Load balance: expert frequency with diagonal gate -------------------------

#[test]
fn test_load_balance_diagonal_gate_diverse_tokens() {
    // With a diagonal gate and diverse token features, the aux loss should
    // indicate reasonably balanced routing (each token feature activates
    // a different expert).
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 1;

    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts {
        gate_data[e * model_dim + e] = 10.0;
    }

    let moe = moe_with_gate(gate_data, num_experts, top_k, model_dim, ff_dim, true);

    // 4 tokens, each with a different dominant feature -> routes to 4 different experts.
    let x = DynTensor::from_vec(
        vec![
            10.0, 0.0, 0.0, 0.0, // -> expert 0
            0.0, 10.0, 0.0, 0.0, // -> expert 1
            0.0, 0.0, 10.0, 0.0, // -> expert 2
            0.0, 0.0, 0.0, 10.0, // -> expert 3
        ],
        &[4, model_dim],
        &Device::Cpu,
    )
    .unwrap();

    let result = moe.forward_with_aux(&x).unwrap();
    let loss = result.aux_loss.to_flat_vec::<f32>().unwrap()[0];

    // With perfect 1/4 distribution, aux_loss = E * E * (1/E * 1/E) = 1.0.
    assert!(loss.is_finite(), "aux loss must be finite");
    assert!(
        (loss - 1.0).abs() < 0.3,
        "perfectly balanced routing should have aux loss near 1.0, got {loss}"
    );
}

// -- Edge case: zero-input tokens through MoE ----------------------------------

#[test]
fn test_zero_input_tokens_produce_zero_output() {
    // When all input features are zero, the output should be zero
    // (linear transforms of zero are zero, SiLU(0)=0, etc.).
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 2;

    let gate_data = vec![1.0f32; num_experts * model_dim];
    let moe = moe_with_gate(gate_data, num_experts, top_k, model_dim, ff_dim, true);

    let x = DynTensor::zeros(&[3, model_dim], nn_core::DType::F32, &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|&v| v.abs() < 1e-10),
        "zero input should produce zero output through MoE"
    );
}

// -- Shared expert with skewed routing ----------------------------------------

#[test]
fn test_shared_expert_always_contributes_regardless_of_routing() {
    // Even when routing is heavily skewed, the shared expert processes ALL
    // tokens, so its contribution is always present.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 1;

    // Gate forces everything to expert 0.
    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for d in 0..model_dim {
        gate_data[0 * model_dim + d] = 100.0;
    }

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
    let shared = uniform_expert(model_dim, ff_dim, 0.3);
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
        diff > 1e-5,
        "shared expert should contribute even with skewed routing, diff={diff}"
    );
}

// -- Large top-k stress -------------------------------------------------------

#[test]
fn test_top_k_equals_num_experts_large_batch() {
    // top_k == num_experts (all experts active) with a large batch.
    let model_dim = 8;
    let ff_dim = 16;
    let num_experts = 8;
    let top_k = 8;

    let gate_data = vec![1.0f32; num_experts * model_dim];
    let moe = moe_with_gate(gate_data, num_experts, top_k, model_dim, ff_dim, true);

    let n_tokens = 32;
    let x = DynTensor::from_vec(
        vec![1.0f32; n_tokens * model_dim],
        &[n_tokens, model_dim],
        &Device::Cpu,
    )
    .unwrap();

    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[n_tokens, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "all-experts-active large batch must be finite"
    );
}

// -- Config edge cases --------------------------------------------------------

#[test]
fn test_moe_config_top_k_equals_num_experts_valid() {
    let cfg = MoeLayerConfig::new(4, 4, 256, 512, true, false);
    assert!(cfg.is_ok());
}

#[test]
fn test_moe_config_top_k_one_valid() {
    let cfg = MoeLayerConfig::new(128, 1, 4096, 2560, true, false);
    assert!(cfg.is_ok());
}

#[test]
fn test_moe_config_single_expert_top_one_valid() {
    let cfg = MoeLayerConfig::new(1, 1, 256, 512, true, false);
    assert!(cfg.is_ok());
}
