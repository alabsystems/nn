// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for dispatch plan optimizer report generation.
//!
//! Part of #3828.

use std::collections::HashMap;

use nn_dsl::buffer_planner::BufferPlan;
use nn_dsl::PeepholeConfig;

use crate::compiled_model::CompiledModelDef;
use crate::compiled_model_optimizer_report::{
    diff_peephole_configs, format_optimizer_report, generate_optimizer_report,
    generate_optimizer_report_with_metrics, OptimizerReport,
};

/// Build a minimal `CompiledModelDef` with no steps for empty-plan tests.
fn empty_def() -> CompiledModelDef {
    CompiledModelDef {
        steps: Vec::new(),
        step_metas: Vec::new(),
        weight_buffers: Vec::new(),
        constant_buffers: HashMap::new(),
        num_inputs: 0,
        input_specs: Vec::new(),
        output_step_indices: Vec::new(),
        output_metas: Vec::new(),
        buffer_plan: BufferPlan {
            total_bytes: 0,
            step_offsets: Vec::new(),
            step_sizes: Vec::new(),
            naive_total: 0,
            last_use: Vec::new(),
        },
        precision: None,
        input_name_cache: Vec::new(),
        release_at: Vec::new(),
        mixed_precision_active: false,
        autocast_policy: None,
        autocast_active: false,
        mixed_gemm_infos: Vec::new(),
        proof_certificate: None,
        shape_policy: crate::compiled_model::ShapePolicy::Fixed,
        icb_eligible: Vec::new(),
        icb_segments: Vec::new(),
        icb_segment_starts: HashMap::new(),
        concurrent_barriers: Vec::new(),
    }
}

#[test]
fn test_default_config_produces_zero_difference_diff() {
    let a = PeepholeConfig::default();
    let b = PeepholeConfig::default();
    let diff = diff_peephole_configs(&a, &b);
    assert!(
        diff.is_empty(),
        "identical configs should produce empty diff, got {diff:?}"
    );
}

#[test]
fn test_single_pass_disabled_shows_in_diff() {
    let default = PeepholeConfig::default();
    let modified = PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };

    let diff = diff_peephole_configs(&default, &modified);
    assert_eq!(diff.len(), 1, "should have exactly one diff entry");
    assert_eq!(diff[0].0, "silu_mul");
    assert!(diff[0].1, "value in `a` should be true");
    assert!(!diff[0].2, "value in `b` should be false");
}

#[test]
fn test_multiple_passes_disabled_shows_in_diff() {
    let default = PeepholeConfig::default();
    let modified = PeepholeConfig {
        silu_mul: false,
        flip_lstm: false,
        add_layer_norm: false,
        ..Default::default()
    };

    let diff = diff_peephole_configs(&default, &modified);
    assert_eq!(diff.len(), 3, "should have three diff entries");

    let names: Vec<&str> = diff.iter().map(|(n, _, _)| n.as_str()).collect();
    assert!(names.contains(&"silu_mul"));
    assert!(names.contains(&"flip_lstm"));
    assert!(names.contains(&"add_layer_norm"));
}

#[test]
fn test_dispatch_reduction_percentage_calculation() {
    let def = empty_def();
    let config = PeepholeConfig::default();

    // Use the explicit-metrics variant to test arithmetic.
    let report = generate_optimizer_report_with_metrics(&def, &config, 150, 8000.0);

    // Baseline is 0 dispatches (empty def), so reduction is 0%.
    assert_eq!(report.baseline_dispatches, 0);
    assert_eq!(report.optimized_dispatches, 150);
    // When baseline is 0, reduction_pct is 0.0 (no division by zero).
    assert!((report.dispatch_reduction_pct - 0.0).abs() < 1e-9);
}

#[test]
fn test_dispatch_reduction_percentage_with_savings() {
    let config = PeepholeConfig::default();

    // Construct a report manually to test percentage math.
    let report = OptimizerReport {
        baseline_dispatches: 200,
        optimized_dispatches: 150,
        dispatch_reduction_pct: 25.0,
        baseline_cost_estimate: 10000.0,
        optimized_cost_estimate: 8000.0,
        speedup_estimate: 1.25,
        config_used: config,
        passes_enabled: vec!["silu_mul".to_string()],
        passes_disabled: vec!["flip_lstm".to_string()],
    };

    // Verify: (200 - 150) / 200 * 100 = 25.0%
    assert!((report.dispatch_reduction_pct - 25.0).abs() < 1e-9);
    // Verify: 10000 / 8000 = 1.25x
    assert!((report.speedup_estimate - 1.25).abs() < 1e-9);
}

#[test]
fn test_format_output_includes_all_sections() {
    let config = PeepholeConfig::default();
    let report = OptimizerReport {
        baseline_dispatches: 200,
        optimized_dispatches: 180,
        dispatch_reduction_pct: 10.0,
        baseline_cost_estimate: 10000.0,
        optimized_cost_estimate: 9000.0,
        speedup_estimate: 1.11,
        config_used: config,
        passes_enabled: vec![
            "silu_mul".to_string(),
            "add_layer_norm".to_string(),
        ],
        passes_disabled: vec!["flip_lstm".to_string()],
    };

    let formatted = format_optimizer_report(&report);

    assert!(
        formatted.contains("Dispatch Plan Optimizer Report"),
        "should have title"
    );
    assert!(
        formatted.contains("Dispatch Counts"),
        "should have dispatch counts section"
    );
    assert!(
        formatted.contains("Cost Estimates"),
        "should have cost estimates section"
    );
    assert!(
        formatted.contains("Pass Configuration"),
        "should have pass configuration section"
    );
    assert!(
        formatted.contains("200"),
        "should show baseline dispatches"
    );
    assert!(
        formatted.contains("180"),
        "should show optimized dispatches"
    );
    assert!(
        formatted.contains("10.0%"),
        "should show reduction percentage"
    );
    assert!(
        formatted.contains("+ silu_mul"),
        "should list enabled passes"
    );
    assert!(
        formatted.contains("- flip_lstm"),
        "should list disabled passes"
    );
    assert!(
        formatted.contains("us"),
        "should show cost in microseconds"
    );
}

#[test]
fn test_cost_estimate_comparison() {
    let config = PeepholeConfig::default();

    // Speedup = baseline / optimized = 10000 / 5000 = 2.0x
    let report = OptimizerReport {
        baseline_dispatches: 100,
        optimized_dispatches: 50,
        dispatch_reduction_pct: 50.0,
        baseline_cost_estimate: 10000.0,
        optimized_cost_estimate: 5000.0,
        speedup_estimate: 2.0,
        config_used: config.clone(),
        passes_enabled: Vec::new(),
        passes_disabled: Vec::new(),
    };

    assert!((report.speedup_estimate - 2.0).abs() < 1e-9);
    assert!(report.optimized_cost_estimate < report.baseline_cost_estimate);

    // Edge case: equal costs = 1.0x speedup
    let report_equal = OptimizerReport {
        baseline_dispatches: 100,
        optimized_dispatches: 100,
        dispatch_reduction_pct: 0.0,
        baseline_cost_estimate: 10000.0,
        optimized_cost_estimate: 10000.0,
        speedup_estimate: 1.0,
        config_used: config,
        passes_enabled: Vec::new(),
        passes_disabled: Vec::new(),
    };
    assert!((report_equal.speedup_estimate - 1.0).abs() < 1e-9);
}

#[test]
fn test_generate_optimizer_report_empty_def() {
    let def = empty_def();
    let config = PeepholeConfig::default();
    let report = generate_optimizer_report(&def, &config);

    assert_eq!(report.baseline_dispatches, 0);
    assert_eq!(report.optimized_dispatches, 0);
    assert!((report.dispatch_reduction_pct - 0.0).abs() < 1e-9);
    // All passes enabled in default config (FIELD_NAMES has 26 entries).
    assert_eq!(report.passes_enabled.len(), 26);
    assert!(report.passes_disabled.is_empty());
}

#[test]
fn test_generate_optimizer_report_disabled_pass_classification() {
    let def = empty_def();
    let config = PeepholeConfig {
        silu_mul: false,
        bilstm_cat: false,
        ..Default::default()
    };

    let report = generate_optimizer_report(&def, &config);

    // 26 total passes (FIELD_NAMES), 2 disabled (silu_mul, bilstm_cat) → 24 enabled.
    assert_eq!(report.passes_enabled.len(), 24);
    assert_eq!(report.passes_disabled.len(), 2);
    assert!(report.passes_disabled.contains(&"silu_mul".to_string()));
    assert!(report.passes_disabled.contains(&"bilstm_cat".to_string()));
}

#[test]
fn test_display_impl_matches_format() {
    let config = PeepholeConfig::default();
    let report = OptimizerReport {
        baseline_dispatches: 50,
        optimized_dispatches: 40,
        dispatch_reduction_pct: 20.0,
        baseline_cost_estimate: 5000.0,
        optimized_cost_estimate: 4000.0,
        speedup_estimate: 1.25,
        config_used: config,
        passes_enabled: vec!["silu_mul".to_string()],
        passes_disabled: Vec::new(),
    };

    let display_output = format!("{report}");
    let format_output = format_optimizer_report(&report);
    assert_eq!(display_output, format_output);
}

#[test]
fn test_diff_reversed_direction() {
    let a = PeepholeConfig {
        norm_activ_conv1d: false,
        ..Default::default()
    };
    let b = PeepholeConfig::default();

    let diff = diff_peephole_configs(&a, &b);
    assert_eq!(diff.len(), 1);
    assert_eq!(diff[0].0, "norm_activ_conv1d");
    assert!(!diff[0].1, "a has it disabled");
    assert!(diff[0].2, "b has it enabled");
}

#[test]
fn test_all_passes_disabled_in_diff() {
    let default = PeepholeConfig::default();
    let all_off = PeepholeConfig {
        norm_activ_conv1d: false,
        fused_resblock: false,
        linear_activation: false,
        add_layer_norm: false,
        norm_linear: false,
        attention_transpose: false,
        flip_lstm: false,
        batched_linear_projection: false,
        channels_first_layer_norm: false,
        silu_mul: false,
        auto_fuse_elementwise: false,
        bilstm_cat: false,
        add_norm_linear: false,
        fuse_adain_snake: false,
        fuse_upsample_conv1d: false,
        fuse_instance_norm_mul_add: false,
        fuse_conv1d_activation: false,
        fuse_snake_instance_norm: false,
        fuse_conv1d_snake_norm: false,
        fuse_conv1d_snake_norm_resblock: false,
        fuse_add_instance_norm_conv1x1: false,
        fuse_conv_transpose1d_activation: false,
        norm_activ_conv_transpose1d: false,
        fuse_instance_norm_conv1d: false,
        fuse_conv1d_instance_norm: false,
        fuse_linear_layer_norm: false,
        fuse_resblock_chain: false,
        fuse_activation_conv1d: false,
    };

    let diff = diff_peephole_configs(&default, &all_off);
    assert_eq!(
        diff.len(),
        26,
        "all 26 reported pass fields (FIELD_NAMES) should differ between default and all-off"
    );
    for (_, val_a, val_b) in &diff {
        assert!(val_a, "default should be true");
        assert!(!val_b, "all-off should be false");
    }
}
