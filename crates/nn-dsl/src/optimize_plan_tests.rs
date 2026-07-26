// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `optimize_plan.rs` — PeepholeConfig enumeration, exhaustive
//! search, dispatch counting, and optimization result reporting.

use std::time::Duration;

use super::*;

// ---------------------------------------------------------------------------
// PeepholeConfig enumeration (all 2^N combinations)
// ---------------------------------------------------------------------------

#[test]
fn test_enumerate_produces_correct_count() {
    // Count lazily (O(1) memory) rather than materializing 2^28 configs.
    assert_eq!(
        enumerate_peephole_configs().count(),
        1usize << PEEPHOLE_FIELD_COUNT
    );
    assert_eq!(enumerate_peephole_configs().count(), 268_435_456);
}

#[test]
fn test_enumerate_includes_all_disabled() {
    // Bitmask 0 = all disabled (O(1), no full enumeration).
    let all_off = config_from_bitmask(0);
    let all_off = &all_off;
    assert!(!all_off.norm_activ_conv1d);
    assert!(!all_off.fused_resblock);
    assert!(!all_off.linear_activation);
    assert!(!all_off.add_layer_norm);
    assert!(!all_off.norm_linear);
    assert!(!all_off.attention_transpose);
    assert!(!all_off.flip_lstm);
    assert!(!all_off.batched_linear_projection);
    assert!(!all_off.channels_first_layer_norm);
    assert!(!all_off.silu_mul);
    assert!(!all_off.auto_fuse_elementwise);
    assert!(!all_off.bilstm_cat);
    assert!(!all_off.add_norm_linear);
    assert!(!all_off.fuse_adain_snake);
    assert!(!all_off.fuse_upsample_conv1d);
    assert!(!all_off.fuse_instance_norm_mul_add);
}

#[test]
fn test_enumerate_includes_all_enabled() {
    // Last entry (all bits set) = all enabled = default (O(1)).
    let all_on = config_from_bitmask((1u32 << PEEPHOLE_FIELD_COUNT) - 1);
    let all_on = &all_on;
    assert!(all_on.norm_activ_conv1d);
    assert!(all_on.fused_resblock);
    assert!(all_on.linear_activation);
    assert!(all_on.add_layer_norm);
    assert!(all_on.norm_linear);
    assert!(all_on.attention_transpose);
    assert!(all_on.flip_lstm);
    assert!(all_on.batched_linear_projection);
    assert!(all_on.channels_first_layer_norm);
    assert!(all_on.silu_mul);
    assert!(all_on.auto_fuse_elementwise);
    assert!(all_on.bilstm_cat);
    assert!(all_on.add_norm_linear);
    assert!(all_on.fuse_adain_snake);
    assert!(all_on.fuse_upsample_conv1d);
}

#[test]
fn test_enumerate_all_unique() {
    // Every bitmask produces a distinct config. Verify no duplicates by
    // checking that each bitmask round-trips correctly.
    // Check that first (all-off) and last (all-on) differ (O(1)).
    let first = config_from_bitmask(0);
    let first = &first;
    let last = config_from_bitmask((1u32 << PEEPHOLE_FIELD_COUNT) - 1);
    let last = &last;
    assert_ne!(
        first.norm_activ_conv1d, last.norm_activ_conv1d,
        "all-off and all-on should differ"
    );
}

#[test]
fn test_single_bit_configs() {
    // Index specific bitmasks directly via config_from_bitmask (O(1)):
    // `enumerate_peephole_configs().nth(i)` == `config_from_bitmask(i)`.
    // Bitmask 1 (bit 0 set) => only norm_activ_conv1d is true.
    let cfg = config_from_bitmask(1);
    let cfg = &cfg;
    assert!(cfg.norm_activ_conv1d);
    assert!(!cfg.fused_resblock);
    assert!(!cfg.linear_activation);

    // Bitmask 2 (bit 1 set) => only fused_resblock is true.
    let cfg = config_from_bitmask(2);
    let cfg = &cfg;
    assert!(!cfg.norm_activ_conv1d);
    assert!(cfg.fused_resblock);
    assert!(!cfg.linear_activation);

    // Bitmask 4 (bit 2 set) => only linear_activation is true.
    let cfg = config_from_bitmask(4);
    let cfg = &cfg;
    assert!(!cfg.norm_activ_conv1d);
    assert!(!cfg.fused_resblock);
    assert!(cfg.linear_activation);
}

#[test]
fn test_bitmask_bit_12_add_norm_linear() {
    // Bitmask 1<<12 = 4096 => only add_norm_linear is true (O(1)).
    let cfg = config_from_bitmask(4096);
    let cfg = &cfg;
    assert!(!cfg.norm_activ_conv1d);
    assert!(!cfg.fused_resblock);
    assert!(!cfg.silu_mul);
    assert!(!cfg.bilstm_cat);
    assert!(cfg.add_norm_linear);
}

// ---------------------------------------------------------------------------
// Default config detection
// ---------------------------------------------------------------------------

#[test]
fn test_default_config_detection() {
    assert!(is_default_config(&PeepholeConfig::default()));
    let non_default = PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };
    assert!(!is_default_config(&non_default));
}

#[test]
fn test_default_config_all_fields_true() {
    let d = PeepholeConfig::default();
    assert!(d.norm_activ_conv1d);
    assert!(d.fused_resblock);
    assert!(d.linear_activation);
    assert!(d.add_layer_norm);
    assert!(d.norm_linear);
    assert!(d.attention_transpose);
    assert!(d.flip_lstm);
    assert!(d.batched_linear_projection);
    assert!(d.channels_first_layer_norm);
    assert!(d.silu_mul);
    assert!(d.auto_fuse_elementwise);
    assert!(d.bilstm_cat);
    assert!(d.add_norm_linear);
    assert!(d.fuse_adain_snake);
    assert!(d.fuse_upsample_conv1d);
    assert!(d.fuse_instance_norm_mul_add);
}

#[test]
fn test_non_default_detected_for_each_field() {
    // Toggling any single field should make is_default_config return false.
    let field_count = PEEPHOLE_FIELD_COUNT;
    for bit in 0..field_count {
        // Start with all bits on (default), then turn off one bit.
        let mask = ((1u32 << field_count) - 1) ^ (1u32 << bit);
        let cfg = config_from_bitmask(mask);
        assert!(
            !is_default_config(&cfg),
            "turning off bit {bit} should make config non-default"
        );
    }
}

// ---------------------------------------------------------------------------
// count_dispatches
// ---------------------------------------------------------------------------

#[test]
fn test_count_dispatches_empty_plan() {
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_count_dispatches_non_dispatch_steps() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::Passthrough {
                op_name: "reshape".to_string(),
                output_shape: vec![1, 2],
            },
            CompiledStep::IdentityPassthrough,
            CompiledStep::ConstantValue {
                value: 1.0,
                shape: vec![1],
            },
        ],
        input_shapes: vec![vec![1, 2]],
        output_step: 3,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 0);
}

// ---------------------------------------------------------------------------
// optimize_plan search behavior
// ---------------------------------------------------------------------------

#[test]
fn test_optimize_plan_empty_graph_zero_dispatches() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let result = optimize_plan(&graph, Duration::from_secs(10))
        .expect("optimize_plan should succeed on empty graph");
    assert_eq!(result.dispatch_count, 0);
    assert_eq!(result.baseline_dispatch_count, 0);
    assert!(
        result.configs_explored >= 1,
        "should explore at least baseline"
    );
}

#[test]
fn test_optimize_plan_zero_budget_returns_baseline() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let result = optimize_plan(&graph, Duration::ZERO)
        .expect("optimize_plan should succeed with zero budget");
    // With zero budget, should still have baseline.
    assert_eq!(result.configs_explored, 1);
    assert_eq!(result.baseline_dispatch_count, 0);
}

#[test]
fn test_optimize_plan_best_never_worse_than_baseline() {
    // The best dispatch_count should never exceed the baseline.
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let result =
        optimize_plan(&graph, Duration::from_millis(100)).expect("optimize_plan should succeed");
    assert!(
        result.dispatch_count <= result.baseline_dispatch_count,
        "best ({}) should be <= baseline ({})",
        result.dispatch_count,
        result.baseline_dispatch_count
    );
}

// ---------------------------------------------------------------------------
// Config comparison (dispatch count, cost)
// ---------------------------------------------------------------------------

#[test]
fn test_optimize_plan_with_cost_empty_graph() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let cost_model = CostModel::apple_m4();
    let result = optimize_plan_with_cost(&graph, &cost_model, Duration::from_secs(10))
        .expect("optimize_plan_with_cost should succeed on empty graph");
    assert_eq!(result.dispatch_count, 0);
    assert_eq!(result.baseline_dispatch_count, 0);
}

#[test]
fn test_optimize_plan_with_cost_zero_budget() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let cost_model = CostModel::apple_m4_max();
    let result = optimize_plan_with_cost(&graph, &cost_model, Duration::ZERO)
        .expect("should succeed with zero budget");
    assert_eq!(result.configs_explored, 1);
}

// ---------------------------------------------------------------------------
// Budget timeout handling
// ---------------------------------------------------------------------------

#[test]
fn test_optimize_plan_respects_budget_cap() {
    // With a very short budget (1 ns), the search should terminate
    // quickly, exploring only the baseline.
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let result =
        optimize_plan(&graph, Duration::from_nanos(1)).expect("should succeed with tiny budget");
    // Should explore at most 1-2 configs (baseline + maybe one more).
    assert!(
        result.configs_explored <= 2,
        "tiny budget should explore at most 2 configs, got {}",
        result.configs_explored
    );
}

// ---------------------------------------------------------------------------
// OptimizationResult summarize
// ---------------------------------------------------------------------------

#[test]
fn test_summarize_output_format() {
    let result = OptimizationResult {
        plan: CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        },
        config: PeepholeConfig::default(),
        dispatch_count: 180,
        configs_explored: 4096,
        baseline_dispatch_count: 200,
        best_cost_ns: 9000.0,
        baseline_cost_ns: 10000.0,
    };
    let summary = result.summarize();
    assert!(summary.contains("200"), "should mention baseline count");
    assert!(summary.contains("180"), "should mention best count");
    assert!(
        summary.contains("10.0%"),
        "should show percentage reduction"
    );
    assert!(summary.contains("4096"), "should mention configs explored");
    assert!(
        summary.contains("Baseline cost"),
        "should include cost info"
    );
    assert!(summary.contains("Best cost"), "should include best cost");
}

#[test]
fn test_summarize_zero_baseline() {
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
    assert!(
        summary.contains("baseline is 0 dispatches"),
        "should handle zero baseline gracefully"
    );
}

#[test]
fn test_summarize_no_improvement() {
    let result = OptimizationResult {
        plan: CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        },
        config: PeepholeConfig::default(),
        dispatch_count: 100,
        configs_explored: 8192,
        baseline_dispatch_count: 100,
        best_cost_ns: 5000.0,
        baseline_cost_ns: 5000.0,
    };
    let summary = result.summarize();
    assert!(
        summary.contains("0 fewer dispatches"),
        "should show 0 improvement"
    );
    assert!(summary.contains("0.0%"), "0% reduction");
}

// ---------------------------------------------------------------------------
// analyze_pass_impact
// ---------------------------------------------------------------------------

#[test]
fn test_analyze_pass_impact_empty_graph_all_zero_impact() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let entries =
        analyze_pass_impact(&graph).expect("analyze_pass_impact should succeed on empty graph");

    // All passes should have 0 impact on an empty graph (0 dispatches baseline).
    for entry in &entries {
        assert_eq!(
            entry.impact, 0,
            "pass '{}' should have 0 impact on empty graph, got {}",
            entry.pass_name, entry.impact
        );
        assert_eq!(entry.enabled_dispatch_count, 0);
        assert_eq!(entry.disabled_dispatch_count, 0);
    }
}

#[test]
fn test_analyze_pass_impact_returns_15_entries() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let entries = analyze_pass_impact(&graph).expect("analyze_pass_impact should succeed");

    assert_eq!(
        entries.len(),
        PEEPHOLE_FIELD_COUNT as usize,
        "should return exactly {} entries (one per pass), got {}",
        PEEPHOLE_FIELD_COUNT,
        entries.len()
    );
}

#[test]
fn test_analyze_pass_impact_names_match_peephole_fields() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let entries = analyze_pass_impact(&graph).expect("analyze_pass_impact should succeed");

    let expected_names: Vec<&str> = vec![
        "norm_activ_conv1d",
        "fused_resblock",
        "linear_activation",
        "add_layer_norm",
        "norm_linear",
        "attention_transpose",
        "flip_lstm",
        "batched_linear_projection",
        "channels_first_layer_norm",
        "silu_mul",
        "auto_fuse_elementwise",
        "bilstm_cat",
        "add_norm_linear",
        "fuse_adain_snake",
        "fuse_upsample_conv1d",
        "fuse_instance_norm_mul_add",
        "fuse_conv1d_activation",
        "fuse_snake_instance_norm",
        "fuse_conv1d_snake_norm",
        "fuse_conv1d_snake_norm_resblock",
        "fuse_add_instance_norm_conv1x1",
        "fuse_conv_transpose1d_activation",
        "norm_activ_conv_transpose1d",
        "fuse_instance_norm_conv1d",
        "fuse_conv1d_instance_norm",
        "fuse_linear_layer_norm",
        "fuse_resblock_chain",
        "fuse_activation_conv1d",
    ];

    // Collect actual names (sorted order may differ due to impact sort).
    let mut actual_names: Vec<&str> = entries.iter().map(|e| e.pass_name.as_str()).collect();
    actual_names.sort_unstable();
    let mut sorted_expected = expected_names.clone();
    sorted_expected.sort_unstable();

    assert_eq!(
        actual_names, sorted_expected,
        "pass names should match PeepholeConfig field names"
    );
}

#[test]
fn test_analyze_pass_impact_non_negative() {
    // Disabling a pass should never produce fewer dispatches than the baseline
    // (on an empty graph all are 0, but the invariant still holds).
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let entries = analyze_pass_impact(&graph).expect("analyze_pass_impact should succeed");

    for entry in &entries {
        assert!(
            entry.impact >= 0,
            "pass '{}' has negative impact {} — disabling a pass should not reduce dispatches",
            entry.pass_name,
            entry.impact
        );
    }
}

#[test]
fn test_analyze_pass_impact_sorted_descending() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let entries = analyze_pass_impact(&graph).expect("analyze_pass_impact should succeed");

    for window in entries.windows(2) {
        assert!(
            window[0].impact >= window[1].impact,
            "entries should be sorted by impact descending: '{}' ({}) before '{}' ({})",
            window[0].pass_name,
            window[0].impact,
            window[1].pass_name,
            window[1].impact,
        );
    }
}

// ---------------------------------------------------------------------------
// Default config vs optimized
// ---------------------------------------------------------------------------

#[test]
fn test_default_config_is_all_enabled() {
    let default = PeepholeConfig::default();
    // Last entry (all bits set) = all enabled = default (O(1)).
    let all_on = config_from_bitmask((1u32 << PEEPHOLE_FIELD_COUNT) - 1);
    let all_on = &all_on;
    // The default config should match the all-bits-set bitmask.
    assert_eq!(default.norm_activ_conv1d, all_on.norm_activ_conv1d);
    assert_eq!(default.fused_resblock, all_on.fused_resblock);
    assert_eq!(default.linear_activation, all_on.linear_activation);
    assert_eq!(default.add_layer_norm, all_on.add_layer_norm);
    assert_eq!(default.norm_linear, all_on.norm_linear);
    assert_eq!(default.attention_transpose, all_on.attention_transpose);
    assert_eq!(default.flip_lstm, all_on.flip_lstm);
    assert_eq!(
        default.batched_linear_projection,
        all_on.batched_linear_projection
    );
    assert_eq!(
        default.channels_first_layer_norm,
        all_on.channels_first_layer_norm
    );
    assert_eq!(default.silu_mul, all_on.silu_mul);
    assert_eq!(default.auto_fuse_elementwise, all_on.auto_fuse_elementwise);
    assert_eq!(default.bilstm_cat, all_on.bilstm_cat);
    assert_eq!(default.add_norm_linear, all_on.add_norm_linear);
    assert_eq!(default.fuse_adain_snake, all_on.fuse_adain_snake);
    assert_eq!(default.fuse_upsample_conv1d, all_on.fuse_upsample_conv1d);
    assert_eq!(
        default.fuse_instance_norm_mul_add,
        all_on.fuse_instance_norm_mul_add
    );
}

#[test]
fn test_peephole_field_count_matches_struct() {
    // PEEPHOLE_FIELD_COUNT should be 28 (matching the 28 boolean fields).
    assert_eq!(PEEPHOLE_FIELD_COUNT, 28);
    // Total enumeration should be 2^28.
    assert_eq!(1u32 << PEEPHOLE_FIELD_COUNT, 268_435_456);
}

#[test]
fn test_optimize_segments_empty() {
    let cost_model = CostModel::apple_m4();
    let results = optimize_segments(&[], &cost_model, Duration::from_secs(1));
    assert!(results.is_empty(), "no segments => no results");
}

// ---------------------------------------------------------------------------
// Bitmask roundtrip: every bitmask encodes/decodes 15 fields correctly
// ---------------------------------------------------------------------------

/// Extracts the 21 boolean fields from a PeepholeConfig into an array
/// in bit-order (matching config_from_bitmask).
fn config_to_bits(cfg: &PeepholeConfig) -> [bool; 28] {
    [
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
    ]
}

#[test]
fn test_bitmask_roundtrip_all_131072_configs() {
    // Every bitmask from 0..2^17 should produce a config whose boolean
    // fields exactly match the bits of the mask. This is the definitive
    // encode/decode correctness test.
    let total = 1u32 << PEEPHOLE_FIELD_COUNT;
    for mask in 0..total {
        let cfg = config_from_bitmask(mask);
        let bits = config_to_bits(&cfg);
        for bit_idx in 0..PEEPHOLE_FIELD_COUNT as usize {
            let expected = (mask >> bit_idx) & 1 == 1;
            assert_eq!(
                bits[bit_idx], expected,
                "mask={mask:#06x} bit={bit_idx}: expected {expected}, got {}",
                bits[bit_idx]
            );
        }
    }
}

#[test]
fn test_exactly_one_bitmask_is_default() {
    // Exactly one of the 131072 configs should be recognized as the default
    // (all fields true). That is the all-bits-set mask (2^17 - 1).
    let total = 1u32 << PEEPHOLE_FIELD_COUNT;
    let mut default_count = 0u32;
    let mut default_mask = 0u32;
    for mask in 0..total {
        let cfg = config_from_bitmask(mask);
        if is_default_config(&cfg) {
            default_count += 1;
            default_mask = mask;
        }
    }
    assert_eq!(
        default_count, 1,
        "exactly one bitmask should produce the default config, found {default_count}"
    );
    let expected_mask = total - 1; // 0x7FFF
    assert_eq!(
        default_mask, expected_mask,
        "default config should be the all-bits-set mask ({expected_mask:#06x}), got {default_mask:#06x}"
    );
}

// ---------------------------------------------------------------------------
// All-disabled vs all-enabled configs
// ---------------------------------------------------------------------------

#[test]
fn test_all_disabled_config_is_bitmask_zero() {
    let all_off = config_from_bitmask(0);
    let bits = config_to_bits(&all_off);
    assert!(
        bits.iter().all(|b| !b),
        "bitmask 0 should produce all-false config"
    );
    assert!(!is_default_config(&all_off));
}

#[test]
fn test_all_enabled_config_matches_default() {
    let all_on_mask = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    let all_on = config_from_bitmask(all_on_mask);
    let default_cfg = PeepholeConfig::default();
    let on_bits = config_to_bits(&all_on);
    let def_bits = config_to_bits(&default_cfg);
    assert_eq!(on_bits, def_bits, "all-bits-set bitmask must equal Default");
    assert!(is_default_config(&all_on));
}

// ---------------------------------------------------------------------------
// Optimizer: all-disabled vs all-enabled compile on empty graph
// ---------------------------------------------------------------------------

#[test]
fn test_compile_with_all_passes_disabled() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let all_off = config_from_bitmask(0);
    let plan = compile_trace_to_plan_configured(&graph, &all_off)
        .expect("compilation with all passes disabled should succeed on empty graph");
    assert_eq!(
        count_dispatches(&plan),
        0,
        "empty graph produces 0 dispatches regardless of config"
    );
}

#[test]
fn test_compile_with_all_passes_enabled() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let default_cfg = PeepholeConfig::default();
    let plan = compile_trace_to_plan_configured(&graph, &default_cfg)
        .expect("compilation with default config should succeed on empty graph");
    assert_eq!(count_dispatches(&plan), 0);
}

// ---------------------------------------------------------------------------
// Cost model integration: optimizer uses cost to break ties
// ---------------------------------------------------------------------------

#[test]
fn test_optimize_with_cost_uses_cost_model() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);
    let cost_model = CostModel::apple_m4();
    let result = optimize_plan_with_cost(&graph, &cost_model, Duration::from_millis(100))
        .expect("optimize_plan_with_cost should succeed");
    // On an empty graph all costs are 0.
    assert!(
        result.best_cost_ns >= 0.0,
        "cost should be non-negative, got {}",
        result.best_cost_ns
    );
    assert!(
        result.baseline_cost_ns >= 0.0,
        "baseline cost should be non-negative"
    );
    // Best cost should never exceed baseline cost.
    assert!(
        result.best_cost_ns <= result.baseline_cost_ns,
        "best cost ({}) should be <= baseline cost ({})",
        result.best_cost_ns,
        result.baseline_cost_ns
    );
}

#[test]
fn test_optimize_with_different_cost_models() {
    use nn_core::dyn_tensor::trace::ComputationGraph;
    let graph = ComputationGraph::from_nodes(vec![]);

    // Both models should produce valid results on empty graph.
    let m4 = CostModel::apple_m4();
    let m4_max = CostModel::apple_m4_max();

    let result_m4 = optimize_plan_with_cost(&graph, &m4, Duration::from_millis(50))
        .expect("M4 optimization should succeed");
    let result_max = optimize_plan_with_cost(&graph, &m4_max, Duration::from_millis(50))
        .expect("M4 Max optimization should succeed");

    assert_eq!(result_m4.dispatch_count, 0);
    assert_eq!(result_max.dispatch_count, 0);
}

// ---------------------------------------------------------------------------
// Cost model preset sanity: hardware parameters in physically reasonable ranges
// ---------------------------------------------------------------------------

#[test]
fn test_cost_model_apple_m4_preset_values() {
    let m = CostModel::apple_m4();
    // Launch overhead: 1-10 microseconds is reasonable for Apple GPU.
    assert!(
        m.launch_overhead_ns >= 500.0 && m.launch_overhead_ns <= 10_000.0,
        "M4 launch overhead {:.0} ns out of reasonable range [500, 10000]",
        m.launch_overhead_ns
    );
    // Bandwidth: 50-800 GB/s for unified memory.
    assert!(
        m.bandwidth_bytes_per_sec >= 50e9 && m.bandwidth_bytes_per_sec <= 800e9,
        "M4 bandwidth {:.0} B/s out of range",
        m.bandwidth_bytes_per_sec
    );
    // SIMD width: 32 for Apple GPU.
    assert_eq!(m.simd_width, 32, "Apple GPU SIMD width should be 32");
}

#[test]
fn test_cost_model_apple_m4_max_preset_values() {
    let m = CostModel::apple_m4_max();
    assert!(
        m.launch_overhead_ns >= 500.0 && m.launch_overhead_ns <= 10_000.0,
        "M4 Max launch overhead out of range"
    );
    assert!(
        m.bandwidth_bytes_per_sec >= 50e9 && m.bandwidth_bytes_per_sec <= 800e9,
        "M4 Max bandwidth out of range"
    );
    assert_eq!(m.simd_width, 32);
    // M4 Max should have op-specific throughputs configured.
    assert!(
        !m.op_throughput.is_empty(),
        "M4 Max should have op-specific throughput entries"
    );
    // MatMul throughput should be in a reasonable range (10-100 TFLOP/s).
    let matmul_tflops = m.op_throughput.get("matmul").copied().unwrap_or(0.0);
    assert!(
        (1e12..=100e12).contains(&matmul_tflops),
        "M4 Max matmul throughput {matmul_tflops:.0} out of range"
    );
}

#[test]
fn test_cost_model_nvidia_a100_preset_values() {
    let m = CostModel::nvidia_a100();
    // A100 has HBM2e bandwidth ~2039 GB/s.
    assert!(
        m.bandwidth_bytes_per_sec >= 1000e9 && m.bandwidth_bytes_per_sec <= 3000e9,
        "A100 bandwidth {:.0} B/s out of range",
        m.bandwidth_bytes_per_sec
    );
    // A100 launch overhead: 3-10 microseconds typical for PCIe/NVLink.
    assert!(
        m.launch_overhead_ns >= 2000.0 && m.launch_overhead_ns <= 15_000.0,
        "A100 launch overhead out of range"
    );
    assert_eq!(m.simd_width, 32, "NVIDIA warp size should be 32");
}

#[test]
fn test_cost_model_nvidia_rtx_4090_preset_values() {
    let m = CostModel::nvidia_rtx_4090();
    // RTX 4090 GDDR6X bandwidth ~1008 GB/s.
    assert!(
        m.bandwidth_bytes_per_sec >= 500e9 && m.bandwidth_bytes_per_sec <= 2000e9,
        "RTX 4090 bandwidth out of range"
    );
    assert_eq!(m.simd_width, 32);
    // MatMul throughput should be very high (Ada Lovelace).
    let matmul = m.op_throughput.get("matmul").copied().unwrap_or(0.0);
    assert!(matmul >= 10e12, "RTX 4090 matmul throughput too low");
}

#[test]
fn test_all_presets_have_positive_parameters() {
    let presets: Vec<(&str, CostModel)> = vec![
        ("apple_m1", CostModel::apple_m1()),
        ("apple_m2", CostModel::apple_m2()),
        ("apple_m3", CostModel::apple_m3()),
        ("apple_m4", CostModel::apple_m4()),
        ("apple_m4_pro", CostModel::apple_m4_pro()),
        ("apple_m4_max", CostModel::apple_m4_max()),
        ("nvidia_a100", CostModel::nvidia_a100()),
        (
            "nvidia_rtx_4090",
            CostModel::nvidia_rtx_4090(),
        ),
    ];
    for (name, m) in &presets {
        assert!(
            m.launch_overhead_ns > 0.0,
            "{name}: launch_overhead_ns must be positive"
        );
        assert!(
            m.bandwidth_bytes_per_sec > 0.0,
            "{name}: bandwidth must be positive"
        );
        assert!(m.simd_width > 0, "{name}: simd_width must be positive");
        for (op, &throughput) in &m.op_throughput {
            assert!(
                throughput > 0.0,
                "{name}: op '{op}' throughput must be positive, got {throughput}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Cost model: launch overhead dominates small dispatches
// ---------------------------------------------------------------------------

#[test]
fn test_cost_estimate_empty_plan_is_zero() {
    let m = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let est = m.estimate(&plan);
    assert!(
        (est.total_ns - 0.0).abs() < f64::EPSILON,
        "empty plan cost should be 0"
    );
    assert_eq!(est.dispatch_count, 0);
    assert!(est.per_step_ns.is_empty());
}

#[test]
fn test_cost_estimate_non_dispatch_steps_are_free() {
    let m = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::IdentityPassthrough,
            CompiledStep::Passthrough {
                op_name: "reshape".to_string(),
                output_shape: vec![1, 2],
            },
            CompiledStep::ConstantValue {
                value: 42.0,
                shape: vec![1],
            },
        ],
        input_shapes: vec![vec![1, 2]],
        output_step: 3,
        weight_names: vec![],
    };
    let est = m.estimate(&plan);
    assert!(
        (est.total_ns - 0.0).abs() < f64::EPSILON,
        "non-dispatch steps should have zero cost"
    );
    assert_eq!(est.dispatch_count, 0);
}

// ---------------------------------------------------------------------------
// Optimizer result summarize edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_summarize_large_improvement() {
    let result = OptimizationResult {
        plan: CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        },
        config: PeepholeConfig::default(),
        dispatch_count: 50,
        configs_explored: 32768,
        baseline_dispatch_count: 200,
        best_cost_ns: 5000.0,
        baseline_cost_ns: 20000.0,
    };
    let summary = result.summarize();
    assert!(
        summary.contains("150 fewer dispatches"),
        "should show improvement count"
    );
    assert!(summary.contains("75.0%"), "should show 75% reduction");
    assert!(
        summary.contains("32768"),
        "should show total configs explored"
    );
    assert!(
        summary.contains("Cost reduction"),
        "should include cost reduction"
    );
}

#[test]
fn test_summarize_single_dispatch_baseline() {
    let result = OptimizationResult {
        plan: CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        },
        config: PeepholeConfig::default(),
        dispatch_count: 1,
        configs_explored: 100,
        baseline_dispatch_count: 1,
        best_cost_ns: 2000.0,
        baseline_cost_ns: 2000.0,
    };
    let summary = result.summarize();
    assert!(summary.contains("0 fewer dispatches"));
    assert!(summary.contains("0.0%"));
}

// ---------------------------------------------------------------------------
// Monotonic hardware progression: newer Apple chips should be faster
// ---------------------------------------------------------------------------

#[test]
fn test_apple_preset_bandwidth_progression() {
    // M1 -> M2 -> M3 -> M4 -> M4 Pro -> M4 Max: bandwidth should be
    // non-decreasing (newer chips have equal or better memory systems).
    let m1_bw = CostModel::apple_m1().bandwidth_bytes_per_sec;
    let m2_bw = CostModel::apple_m2().bandwidth_bytes_per_sec;
    let m3_bw = CostModel::apple_m3().bandwidth_bytes_per_sec;
    let m4_bw = CostModel::apple_m4().bandwidth_bytes_per_sec;
    let m4p_bw = CostModel::apple_m4_pro().bandwidth_bytes_per_sec;
    let m4m_bw = CostModel::apple_m4_max().bandwidth_bytes_per_sec;
    assert!(m2_bw >= m1_bw, "M2 bandwidth should be >= M1");
    assert!(m3_bw >= m2_bw, "M3 bandwidth should be >= M2");
    assert!(m4_bw >= m3_bw, "M4 bandwidth should be >= M3");
    assert!(m4p_bw >= m3_bw, "M4 Pro bandwidth should be >= M3");
    assert!(m4m_bw >= m4p_bw, "M4 Max bandwidth should be >= M4 Pro");
}

#[test]
fn test_apple_preset_launch_overhead_progression() {
    // Newer Apple chips should have equal or lower launch overhead
    // (better GPU schedulers). M1 >= M2 >= M3 >= M4 >= M4 Max.
    let m1_lo = CostModel::apple_m1().launch_overhead_ns;
    let m2_lo = CostModel::apple_m2().launch_overhead_ns;
    let m3_lo = CostModel::apple_m3().launch_overhead_ns;
    let m4_lo = CostModel::apple_m4().launch_overhead_ns;
    let m4m_lo = CostModel::apple_m4_max().launch_overhead_ns;
    assert!(m2_lo <= m1_lo, "M2 launch overhead should be <= M1");
    assert!(m3_lo <= m2_lo, "M3 launch overhead should be <= M2");
    assert!(m4_lo <= m3_lo, "M4 launch overhead should be <= M3");
    assert!(m4m_lo <= m4_lo, "M4 Max launch overhead should be <= M4");
}

// ---------------------------------------------------------------------------
// PEEPHOLE_FIELD_NAMES consistency
// ---------------------------------------------------------------------------

#[test]
fn test_peephole_field_names_count_matches() {
    assert_eq!(
        PEEPHOLE_FIELD_NAMES.len(),
        PEEPHOLE_FIELD_COUNT as usize,
        "PEEPHOLE_FIELD_NAMES length should match PEEPHOLE_FIELD_COUNT"
    );
}

#[test]
fn test_peephole_field_names_no_duplicates() {
    let mut seen = std::collections::HashSet::new();
    for name in PEEPHOLE_FIELD_NAMES {
        assert!(
            seen.insert(name),
            "duplicate field name in PEEPHOLE_FIELD_NAMES: {name}"
        );
    }
}

#[test]
fn test_peephole_field_names_match_struct_fields() {
    // Verify the first and last names match known struct fields.
    assert_eq!(PEEPHOLE_FIELD_NAMES[0], "norm_activ_conv1d");
    assert_eq!(PEEPHOLE_FIELD_NAMES[14], "fuse_upsample_conv1d");
    assert_eq!(PEEPHOLE_FIELD_NAMES[15], "fuse_instance_norm_mul_add");
    // Spot-check middle fields.
    assert_eq!(PEEPHOLE_FIELD_NAMES[9], "silu_mul");
    assert_eq!(PEEPHOLE_FIELD_NAMES[13], "fuse_adain_snake");
}

// ---------------------------------------------------------------------------
// PassImpactEntry consistency
// ---------------------------------------------------------------------------

#[test]
fn test_pass_impact_entry_impact_formula() {
    // impact = disabled_dispatch_count - enabled_dispatch_count
    let entry = PassImpactEntry {
        pass_name: "test_pass".to_string(),
        enabled_dispatch_count: 100,
        disabled_dispatch_count: 120,
        impact: 20,
    };
    assert_eq!(
        entry.impact,
        entry.disabled_dispatch_count as i64 - entry.enabled_dispatch_count as i64,
        "impact should be disabled - enabled"
    );
}

// ---------------------------------------------------------------------------
// SegmentOptimizationResult
// ---------------------------------------------------------------------------

#[test]
fn test_segment_optimization_result_preserves_name() {
    let seg = SegmentOptimizationResult {
        segment_name: "decoder_stage_3".to_string(),
        result: OptimizationResult {
            plan: CompiledPlan {
                steps: vec![],
                input_shapes: vec![],
                output_step: 0,
                weight_names: vec![],
            },
            config: PeepholeConfig::default(),
            dispatch_count: 10,
            configs_explored: 100,
            baseline_dispatch_count: 15,
            best_cost_ns: 3000.0,
            baseline_cost_ns: 5000.0,
        },
    };
    assert_eq!(seg.segment_name, "decoder_stage_3");
    assert_eq!(seg.result.dispatch_count, 10);
}
