// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Step-type distribution and profiling integration tests for the Kokoro
//! encoder dispatch plan.
//!
//! Extracted from `kokoro_encoder_dispatch_tests.rs` to keep files under
//! the 500-line limit.

use super::*;
use crate::cost_model::{
    estimate_peak_memory, profile_dispatch_plan, total_estimated_time_us, total_flops,
    total_memory_bytes, HardwareCostModel,
};
use nn_dsl::DispatchStep;

// --- Step type distribution ---

#[test]
fn test_step_type_distribution() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();

    let n_embedding = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Embedding { .. }))
        .count();
    let n_linear = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Linear { .. }))
        .count();
    let n_matmul = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::MatMul { .. }))
        .count();
    let n_softmax = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Softmax { .. }))
        .count();
    let n_sigmoid = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Sigmoid { .. }))
        .count();
    let n_gelu = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Gelu { .. }))
        .count();
    let n_tanh = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Tanh { .. }))
        .count();
    let n_add = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::BinaryAdd { .. }))
        .count();
    let n_mul = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::BinaryMul { .. }))
        .count();
    let n_conv1d = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Conv1d(_)))
        .count();
    let n_conv_t = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::ConvTranspose1d(_)))
        .count();

    // 3 Embeddings (word, pos, token_type)
    assert_eq!(n_embedding, 3, "3 Embedding lookups");

    // MatMul: 2 per ALBERT layer (Q×K, attn×V) × 12 = 24
    assert_eq!(n_matmul, 24, "24 MatMuls (12 layers × 2)");

    // Softmax: 1 per ALBERT layer × 12 = 12
    assert_eq!(n_softmax, 12, "12 Softmax (12 layers × 1)");

    // GELU: 1 per ALBERT layer × 12 = 12
    assert_eq!(n_gelu, 12, "12 GELU (12 layers × 1)");

    // Verify all types sum to total
    let sum = n_embedding
        + n_linear
        + n_matmul
        + n_softmax
        + n_sigmoid
        + n_gelu
        + n_tanh
        + n_add
        + n_mul
        + n_conv1d
        + n_conv_t;
    assert_eq!(
        sum, TOTAL_EXPECTED_STEPS,
        "all types sum to {TOTAL_EXPECTED_STEPS}"
    );
}

// --- Profiling integration ---

#[test]
fn test_profile_encoder_plan() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    let model = HardwareCostModel::m4_max();
    let profiles = profile_dispatch_plan(&plan, &model);
    assert_eq!(profiles.len(), plan.len());
    let total_time = total_estimated_time_us(&profiles);
    let total_f = total_flops(&profiles);
    let total_mem = total_memory_bytes(&profiles);
    assert!(total_f > 0, "total FLOPs should be positive");
    assert!(total_mem > 0, "total memory should be positive");
    // Encoder should be < 200ms on M4 Max for 100 tokens
    assert!(
        total_time < 200_000.0,
        "encoder < 200ms on M4 Max, got {total_time:.1} μs"
    );
}

#[test]
fn test_peak_memory_encoder() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    let mem = estimate_peak_memory(&plan);
    // Encoder at 100 tokens — peak memory should be under 500 MB
    assert!(
        mem.peak_total_mb() < 500.0,
        "peak memory {:.3} MB should be < 500 MB",
        mem.peak_total_mb()
    );
    assert_eq!(mem.per_step_output_bytes.len(), plan.len());
}
