// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PeepholeConfig search space: exhaustive enumeration, bitmask
//! roundtrip, budget timeout, optimization invariants, config persistence,
//! and `is_default_config` correctness.
//!
//! Part of #4186.

use std::time::Duration;

use nn_core::dyn_tensor::trace::ComputationGraph;

use super::*;
use crate::cost_model::CostModel;
use crate::trace_compile::{CompiledPlan, PeepholeConfig};

// ---------------------------------------------------------------------------
// Section 1: optimize_plan returns valid config
// ---------------------------------------------------------------------------

#[test]
fn test_optimize_plan_returns_valid_config_empty_graph() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let result = optimize_plan(&graph, Duration::from_secs(5))
        .expect("optimize_plan should succeed on empty graph");

    // Result invariants.
    assert_eq!(result.dispatch_count, 0);
    assert_eq!(result.baseline_dispatch_count, 0);
    assert!(result.configs_explored >= 1);
    assert!(result.best_cost_ns >= 0.0);
    assert!(result.baseline_cost_ns >= 0.0);
}

#[test]
fn test_optimize_plan_best_leq_baseline() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let result = optimize_plan(&graph, Duration::from_millis(200)).expect("should succeed");
    assert!(
        result.dispatch_count <= result.baseline_dispatch_count,
        "best ({}) must be <= baseline ({})",
        result.dispatch_count,
        result.baseline_dispatch_count,
    );
}

// ---------------------------------------------------------------------------
// Section 2: Exhaustive search covers all 2^N configs
// ---------------------------------------------------------------------------

#[test]
fn test_enumerate_covers_all_2n_configs() {
    // Count lazily (O(1) memory) rather than materializing 2^28 configs.
    assert_eq!(
        enumerate_peephole_configs().count(),
        1usize << PEEPHOLE_FIELD_COUNT
    );
}

#[test]
fn test_enumerate_first_is_all_off() {
    // First config = bitmask 0 (O(1), no full enumeration).
    let bits = config_to_bitmask_test(&config_from_bitmask(0));
    assert_eq!(bits, 0, "first config should be bitmask 0 (all off)");
}

#[test]
fn test_enumerate_last_is_all_on() {
    // Last config = all-bits-set bitmask (O(1)).
    let expected = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    let last = config_from_bitmask(expected);
    let bits = config_to_bitmask_test(&last);
    assert_eq!(bits, expected, "last config should be all-on bitmask");
}

// ---------------------------------------------------------------------------
// Section 3: Budget timeout is respected
// ---------------------------------------------------------------------------

#[test]
fn test_zero_budget_explores_only_baseline() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let result = optimize_plan(&graph, Duration::ZERO).expect("should succeed with zero budget");
    assert_eq!(result.configs_explored, 1, "zero budget = baseline only");
}

#[test]
fn test_tiny_budget_limits_exploration() {
    let graph = ComputationGraph::from_nodes(vec![]);
    // 1 nanosecond budget: should explore very few configs.
    let result =
        optimize_plan(&graph, Duration::from_nanos(1)).expect("should succeed with 1ns budget");
    assert!(
        result.configs_explored <= 3,
        "1ns budget should explore at most a few configs, got {}",
        result.configs_explored,
    );
}

#[test]
fn test_generous_budget_explores_more() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let result_short =
        optimize_plan(&graph, Duration::from_nanos(1)).expect("short budget should succeed");
    let result_long =
        optimize_plan(&graph, Duration::from_secs(30)).expect("long budget should succeed");
    assert!(
        result_long.configs_explored >= result_short.configs_explored,
        "longer budget should explore >= configs than shorter budget"
    );
}

// ---------------------------------------------------------------------------
// Section 4: Optimized config produces <= default dispatch count
// ---------------------------------------------------------------------------

#[test]
fn test_optimized_dispatch_count_never_worse() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let result = optimize_plan(&graph, Duration::from_secs(1)).expect("should succeed");
    assert!(result.dispatch_count <= result.baseline_dispatch_count);
}

#[test]
fn test_optimize_with_cost_never_worse() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let cost_model = CostModel::apple_m4();
    let result = optimize_plan_with_cost(&graph, &cost_model, Duration::from_secs(1))
        .expect("should succeed");
    assert!(result.dispatch_count <= result.baseline_dispatch_count);
    assert!(
        result.best_cost_ns <= result.baseline_cost_ns,
        "best cost ({:.0}) should be <= baseline cost ({:.0})",
        result.best_cost_ns,
        result.baseline_cost_ns,
    );
}

// ---------------------------------------------------------------------------
// Section 5: is_default_config correctness
// ---------------------------------------------------------------------------

#[test]
fn test_default_is_default() {
    assert!(is_default_config(&PeepholeConfig::default()));
}

#[test]
fn test_all_false_is_not_default() {
    let cfg = config_from_bitmask(0);
    assert!(!is_default_config(&cfg));
}

#[test]
fn test_single_field_off_is_not_default() {
    for bit in 0..PEEPHOLE_FIELD_COUNT {
        let mask = ((1u32 << PEEPHOLE_FIELD_COUNT) - 1) ^ (1u32 << bit);
        let cfg = config_from_bitmask(mask);
        assert!(
            !is_default_config(&cfg),
            "turning off bit {bit} should make config non-default"
        );
    }
}

#[test]
fn test_exactly_one_default_in_enumeration() {
    // Filter lazily (O(1) memory) rather than materializing 2^28 configs.
    let default_count = enumerate_peephole_configs()
        .filter(is_default_config)
        .count();
    assert_eq!(default_count, 1, "exactly one config should be default");
}

// ---------------------------------------------------------------------------
// Section 6: Bitmask <-> config roundtrip
// ---------------------------------------------------------------------------

/// Convert a PeepholeConfig back to bitmask for test verification.
fn config_to_bitmask_test(cfg: &PeepholeConfig) -> u32 {
    let fields: [bool; 28] = [
        cfg.norm_activ_conv1d,
        cfg.fused_resblock,
        cfg.linear_activation,
        cfg.add_layer_norm,
        cfg.norm_linear,
        cfg.attention_transpose,
        cfg.flip_lstm,
        cfg.batched_linear_projection,
        cfg.channels_first_layer_norm,
        cfg.silu_mul,
        cfg.auto_fuse_elementwise,
        cfg.bilstm_cat,
        cfg.add_norm_linear,
        cfg.fuse_adain_snake,
        cfg.fuse_upsample_conv1d,
        cfg.fuse_instance_norm_mul_add,
        cfg.fuse_conv1d_activation,
        cfg.fuse_snake_instance_norm,
        cfg.fuse_conv1d_snake_norm,
        cfg.fuse_conv1d_snake_norm_resblock,
        cfg.fuse_add_instance_norm_conv1x1,
        cfg.fuse_conv_transpose1d_activation,
        cfg.norm_activ_conv_transpose1d,
        cfg.fuse_instance_norm_conv1d,
        cfg.fuse_conv1d_instance_norm,
        cfg.fuse_linear_layer_norm,
        cfg.fuse_resblock_chain,
        cfg.fuse_activation_conv1d,
    ];
    let mut mask = 0u32;
    for (i, &val) in fields.iter().enumerate() {
        if val {
            mask |= 1 << i;
        }
    }
    mask
}

#[test]
fn test_bitmask_roundtrip_boundary_values() {
    // Test the boundaries: 0, 1, max-1, max.
    let total = 1u32 << PEEPHOLE_FIELD_COUNT;
    for mask in [0, 1, total / 2, total - 2, total - 1] {
        let cfg = config_from_bitmask(mask);
        let recovered = config_to_bitmask_test(&cfg);
        assert_eq!(
            recovered, mask,
            "roundtrip failed for mask {mask:#06x}: got {recovered:#06x}"
        );
    }
}

#[test]
fn test_bitmask_roundtrip_all_single_bits() {
    for bit in 0..PEEPHOLE_FIELD_COUNT {
        let mask = 1u32 << bit;
        let cfg = config_from_bitmask(mask);
        let recovered = config_to_bitmask_test(&cfg);
        assert_eq!(recovered, mask, "single-bit roundtrip failed for bit {bit}");
    }
}

#[test]
fn test_bitmask_roundtrip_all_configs() {
    let total = 1u32 << PEEPHOLE_FIELD_COUNT;
    for mask in 0..total {
        let cfg = config_from_bitmask(mask);
        let recovered = config_to_bitmask_test(&cfg);
        assert_eq!(recovered, mask);
    }
}

// ---------------------------------------------------------------------------
// Section 7: PassImpactEntry and analyze_pass_impact
// ---------------------------------------------------------------------------

#[test]
fn test_analyze_pass_impact_returns_all_passes() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let entries = analyze_pass_impact(&graph).expect("should succeed on empty graph");
    assert_eq!(entries.len(), PEEPHOLE_FIELD_COUNT as usize);
}

#[test]
fn test_analyze_pass_impact_sorted_descending() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let entries = analyze_pass_impact(&graph).expect("should succeed");
    for window in entries.windows(2) {
        assert!(
            window[0].impact >= window[1].impact,
            "entries should be sorted descending: {} ({}) before {} ({})",
            window[0].pass_name,
            window[0].impact,
            window[1].pass_name,
            window[1].impact,
        );
    }
}

#[test]
fn test_analyze_pass_impact_all_names_unique() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let entries = analyze_pass_impact(&graph).expect("should succeed");
    let mut names: Vec<&str> = entries.iter().map(|e| e.pass_name.as_str()).collect();
    let original_len = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), original_len, "all pass names should be unique");
}

// ---------------------------------------------------------------------------
// Section 8: optimize_segments
// ---------------------------------------------------------------------------

#[test]
fn test_optimize_segments_empty_input() {
    let cost_model = CostModel::apple_m4();
    let results = optimize_segments(&[], &cost_model, Duration::from_secs(1));
    assert!(results.is_empty());
}

#[test]
fn test_optimize_segments_single_segment() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let cost_model = CostModel::apple_m4();
    let segments: Vec<(&str, &ComputationGraph)> = vec![("decoder", &graph)];
    let results = optimize_segments(&segments, &cost_model, Duration::from_secs(1));
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].segment_name, "decoder");
    assert_eq!(results[0].result.dispatch_count, 0);
}

#[test]
fn test_optimize_segments_multiple_segments() {
    let graph_a = ComputationGraph::from_nodes(vec![]);
    let graph_b = ComputationGraph::from_nodes(vec![]);
    let cost_model = CostModel::apple_m4();
    let segments: Vec<(&str, &ComputationGraph)> =
        vec![("encoder", &graph_a), ("decoder", &graph_b)];
    let results = optimize_segments(&segments, &cost_model, Duration::from_secs(1));
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].segment_name, "encoder");
    assert_eq!(results[1].segment_name, "decoder");
}

// ---------------------------------------------------------------------------
// Section 9: Summarize edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_summarize_with_zero_baseline_cost() {
    let result = OptimizationResult {
        plan: CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        },
        config: PeepholeConfig::default(),
        dispatch_count: 0,
        configs_explored: 1,
        baseline_dispatch_count: 0,
        best_cost_ns: 0.0,
        baseline_cost_ns: 0.0,
    };
    let summary = result.summarize();
    assert!(!summary.is_empty());
    assert!(summary.contains("baseline is 0 dispatches"));
}

#[test]
fn test_summarize_configs_explored_count() {
    let result = OptimizationResult {
        plan: CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        },
        config: PeepholeConfig::default(),
        dispatch_count: 50,
        configs_explored: 65536,
        baseline_dispatch_count: 100,
        best_cost_ns: 5000.0,
        baseline_cost_ns: 10000.0,
    };
    let summary = result.summarize();
    assert!(summary.contains("65536"), "should show configs explored");
    assert!(summary.contains("50.0%"), "should show 50% reduction");
}
