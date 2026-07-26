// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for autoregressive cost bound model (#1739 D5, #1741 Phase 18).
//!
//! Validates worst-case timing bounds for variable-length decode loops
//! in autoregressive TTS models (Qwen3-TTS pattern).

use super::*;
use nn_dsl::ir::ScalarType;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_dsl::DispatchStep;

fn node(id: usize) -> TensorNodeId {
    TensorNodeId::new(id)
}

/// Build a minimal single-step decode plan: Linear (attention) + Linear (FFN).
///
/// This models a transformer decoder step at worst-case KV length:
/// - Attention: Q×K^T + softmax + attn×V → approximated as 2 Linear ops
/// - FFN: up-project + activation + down-project → approximated as 2 Linear ops
fn qwen3_decode_step(model_dim: usize, ffn_dim: usize, kv_length: usize) -> Vec<DispatchStep> {
    vec![
        // Q×K^T attention: [1, model_dim] × [model_dim, kv_length]
        DispatchStep::Linear {
            kernel_name: "attn_qk".to_string(),
            dtype: ScalarType::F32,
            input: node(0),
            weight: node(1),
            bias: None,
            output: node(2),
            in_features: model_dim,
            out_features: kv_length,
            batch_size: 1,
            total_elements: kv_length,
        },
        // attn×V: [1, kv_length] × [kv_length, model_dim]
        DispatchStep::Linear {
            kernel_name: "attn_v".to_string(),
            dtype: ScalarType::F32,
            input: node(2),
            weight: node(3),
            bias: None,
            output: node(4),
            in_features: kv_length,
            out_features: model_dim,
            batch_size: 1,
            total_elements: model_dim,
        },
        // FFN up-project: [1, model_dim] × [model_dim, ffn_dim]
        DispatchStep::Linear {
            kernel_name: "ffn_up".to_string(),
            dtype: ScalarType::F32,
            input: node(4),
            weight: node(5),
            bias: None,
            output: node(6),
            in_features: model_dim,
            out_features: ffn_dim,
            batch_size: 1,
            total_elements: ffn_dim,
        },
        // FFN down-project: [1, ffn_dim] × [ffn_dim, model_dim]
        DispatchStep::Linear {
            kernel_name: "ffn_down".to_string(),
            dtype: ScalarType::F32,
            input: node(6),
            weight: node(7),
            bias: None,
            output: node(8),
            in_features: ffn_dim,
            out_features: model_dim,
            batch_size: 1,
            total_elements: model_dim,
        },
    ]
}

// --- AutoregressiveCostBound tests ---

#[test]
fn test_autoregressive_bound_basic() {
    let model_dim = 512;
    let ffn_dim = 2048;
    let max_steps = 100;
    let kv_length = max_steps; // worst-case KV length = max_steps

    let step_plan = qwen3_decode_step(model_dim, ffn_dim, kv_length);
    let hw = HardwareCostModel::m4_max_conservative();

    let bound = bound_autoregressive_inference(&step_plan, max_steps, &hw).expect("valid bound");

    // max_steps should be preserved
    assert_eq!(bound.max_steps, 100);
    // 4 dispatch steps per decode step
    assert_eq!(bound.per_step_plan.len(), 4);
    assert_eq!(bound.per_step_profiles.len(), 4);

    // Total = max_steps × per_step
    let per_step = bound.per_step_time_us();
    assert!(per_step > 0.0, "per-step time must be positive");
    assert!(
        (bound.worst_case_total_us - per_step * max_steps as f64).abs() < 0.01,
        "total must equal max_steps × per_step"
    );
}

#[test]
fn test_autoregressive_bound_zero_steps_errors() {
    let step_plan = qwen3_decode_step(512, 2048, 100);
    let hw = HardwareCostModel::m4_max_conservative();

    let result = bound_autoregressive_inference(&step_plan, 0, &hw);
    assert!(result.is_err(), "max_steps=0 must error");
}

#[test]
fn test_autoregressive_bound_single_step() {
    let step_plan = qwen3_decode_step(256, 1024, 1);
    let hw = HardwareCostModel::m4_max_conservative();

    let bound = bound_autoregressive_inference(&step_plan, 1, &hw).expect("valid bound");

    assert_eq!(bound.max_steps, 1);
    // For 1 step, total equals per-step
    assert!(
        (bound.worst_case_total_us - bound.per_step_time_us()).abs() < 0.001,
        "1-step total must equal per-step"
    );
}

#[test]
fn test_autoregressive_bound_within_timing() {
    let model_dim = 512;
    let ffn_dim = 2048;
    let max_steps = 50;
    let kv_length = max_steps;

    let step_plan = qwen3_decode_step(model_dim, ffn_dim, kv_length);
    let hw = HardwareCostModel::m4_max_conservative();

    let bound = bound_autoregressive_inference(&step_plan, max_steps, &hw).expect("valid bound");

    // Set a generous timing bound (1 second = 1_000_000 μs)
    assert!(
        bound.within_bound(1_000_000.0),
        "50-step decode for small model should be under 1 second"
    );

    // Set an impossibly tight bound (1 μs)
    assert!(
        !bound.within_bound(1.0),
        "50-step decode cannot complete in 1 μs"
    );
}

#[test]
fn test_autoregressive_bound_qwen3_100_token_generation() {
    // Qwen3-8B scale: d_model=4096, ffn_dim=11008, max_tokens=100
    let model_dim = 4096;
    let ffn_dim = 11008;
    let max_steps = 100;
    let kv_length = max_steps;

    let step_plan = qwen3_decode_step(model_dim, ffn_dim, kv_length);
    let hw = HardwareCostModel::m4_max_conservative();

    let bound = bound_autoregressive_inference(&step_plan, max_steps, &hw).expect("valid bound");

    // For Qwen3-8B at 100 tokens, worst-case is meaningful but conservative:
    // Per-step FLOPs: 2 * (4096*100 + 100*4096 + 4096*11008 + 11008*4096) ≈ 180M FLOPs
    // Per-step overhead: 4 dispatches × 10 μs = 40 μs
    // Per-step compute: ~180M / 2.84e12 = ~63 μs
    // Per-step memory: weights dominate → additional bandwidth time
    // At conservative model: ~1.86ms per step → ~186ms total for 100 steps
    // This is well within 500ms for TTS (real-time is ~1s for 100 tokens)
    assert!(
        bound.worst_case_total_us < 500_000.0, // 500ms
        "Qwen3-8B 100-token decode should be under 500ms, got {:.1}ms",
        bound.worst_case_total_us / 1000.0,
    );

    // Total FLOPs should be reasonable (100M-100B range for 100 steps)
    assert!(
        bound.worst_case_total_flops > 100_000_000,
        "total FLOPs should be > 100M"
    );
    assert!(
        bound.worst_case_total_flops < 100_000_000_000,
        "total FLOPs should be < 100B"
    );
}

#[test]
fn test_autoregressive_bound_scaling_with_steps() {
    let step_plan = qwen3_decode_step(512, 2048, 200);
    let hw = HardwareCostModel::m4_max_conservative();

    let bound_50 = bound_autoregressive_inference(&step_plan, 50, &hw).expect("valid");
    let bound_100 = bound_autoregressive_inference(&step_plan, 100, &hw).expect("valid");
    let bound_200 = bound_autoregressive_inference(&step_plan, 200, &hw).expect("valid");

    // Total time should scale linearly with max_steps (same per-step cost)
    let ratio_100_50 = bound_100.worst_case_total_us / bound_50.worst_case_total_us;
    let ratio_200_100 = bound_200.worst_case_total_us / bound_100.worst_case_total_us;

    assert!(
        (ratio_100_50 - 2.0).abs() < 0.001,
        "100/50 ratio should be 2.0, got {ratio_100_50:.4}"
    );
    assert!(
        (ratio_200_100 - 2.0).abs() < 0.001,
        "200/100 ratio should be 2.0, got {ratio_200_100:.4}"
    );
}

#[test]
fn test_autoregressive_bound_kv_length_affects_cost() {
    let hw = HardwareCostModel::m4_max_conservative();

    // Short KV cache (10 tokens decoded so far)
    let step_short = qwen3_decode_step(512, 2048, 10);
    let bound_short = bound_autoregressive_inference(&step_short, 50, &hw).expect("valid");

    // Long KV cache (500 tokens decoded so far)
    let step_long = qwen3_decode_step(512, 2048, 500);
    let bound_long = bound_autoregressive_inference(&step_long, 50, &hw).expect("valid");

    // Longer KV cache → more attention FLOPs → higher per-step cost
    assert!(
        bound_long.per_step_time_us() > bound_short.per_step_time_us(),
        "longer KV cache must increase per-step cost: short={:.1}μs, long={:.1}μs",
        bound_short.per_step_time_us(),
        bound_long.per_step_time_us(),
    );
}

#[test]
fn test_autoregressive_bound_report_format() {
    let step_plan = qwen3_decode_step(512, 2048, 100);
    let hw = HardwareCostModel::m4_max_conservative();

    let bound = bound_autoregressive_inference(&step_plan, 100, &hw).expect("valid");

    let report = bound.report();

    assert!(
        report.contains("Autoregressive Cost Bound"),
        "missing header"
    );
    assert!(
        report.contains("Max decode steps: 100"),
        "missing max steps"
    );
    assert!(report.contains("Per-step time:"), "missing per-step time");
    assert!(report.contains("Worst-case total:"), "missing total time");
    assert!(report.contains("Worst-case FLOPs:"), "missing FLOPs");
    assert!(report.contains("Worst-case memory:"), "missing memory");
    assert!(report.contains("Hardware:"), "missing hardware");
    assert!(
        report.contains("Dispatch steps per decode step: 4"),
        "missing step count"
    );
}

#[test]
fn test_autoregressive_bound_empty_step_plan() {
    let hw = HardwareCostModel::m4_max_conservative();

    // Empty dispatch plan (degenerate case — identity decoder)
    let bound = bound_autoregressive_inference(&[], 100, &hw).expect("valid");

    // With no dispatch steps, per-step time = 0, total = 0
    assert_eq!(bound.per_step_profiles.len(), 0);
    assert!(
        bound.worst_case_total_us.abs() < f64::EPSILON,
        "empty plan should have 0 total time"
    );
    assert!(bound.within_bound(1.0), "empty plan within any bound");
}

#[test]
fn test_autoregressive_flops_saturating_mul() {
    // Very large FLOPs × very large steps should not overflow (saturating)
    let step_plan = qwen3_decode_step(16384, 65536, 4096);
    let hw = HardwareCostModel::m4_max_conservative();

    // max_steps large enough that FLOPs could overflow u64
    let bound = bound_autoregressive_inference(&step_plan, 100_000, &hw).expect("valid");

    // Should not panic from overflow — saturating_mul handles it
    assert!(bound.worst_case_total_flops > 0, "FLOPs must be positive");
}
