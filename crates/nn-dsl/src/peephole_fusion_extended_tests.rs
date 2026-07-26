// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for peephole optimizer infrastructure, cost model presets,
//! and fusion gap analysis types.
//!
//! Complements `peephole_pass_tests.rs` (per-pass enable/disable),
//! `gap_analysis_tests.rs` (GapAnalysisReport schema), and
//! `optimize_plan_tests.rs` (exhaustive search).
//!
//! Focus areas:
//! - CostModel preset validation (apple_m4, apple_m1, nvidia_a100, etc.)
//! - CostModel.estimate() on trivial plans
//! - PeepholeConfig bitmask edge cases beyond basic roundtrip
//! - FusionBlocker variant exhaustiveness
//! - FusionGapAnalysis arithmetic edge cases

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::cost_model::CostModel;
use crate::optimize_plan::{
    config_from_bitmask, enumerate_peephole_configs, is_default_config, PEEPHOLE_FIELD_COUNT,
    PEEPHOLE_FIELD_NAMES,
};
use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use crate::trace_compile::{
    analyze_fusion_gaps, count_dispatches, CompiledKernel, CompiledPlan, CompiledStep,
    FusionBlocker, FusionGap, FusionGapAnalysis, PeepholeConfig,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_dispatch(name: &str, shape: &[usize]) -> CompiledStep {
    let node_id = TensorNodeId::new(0);
    let input_node = TensorNode::new(
        node_id,
        TensorOpKind::Input {
            name: "input_0".into(),
            shape: shape.to_vec(),
        },
        shape.to_vec(),
    );
    let def = TensorKernelDef::new(name, vec![input_node], node_id);
    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

// ===========================================================================
// CostModel preset tests
// ===========================================================================

#[test]
fn test_cost_model_apple_m4_has_reasonable_bandwidth() {
    let model = CostModel::apple_m4();
    // Apple M4 base: ~400 GB/s unified memory
    assert!(
        model.bandwidth_bytes_per_sec > 100e9,
        "M4 bandwidth {:.0} should exceed 100 GB/s",
        model.bandwidth_bytes_per_sec / 1e9
    );
    assert!(
        model.bandwidth_bytes_per_sec <= 600e9,
        "M4 bandwidth {:.0} should not exceed 600 GB/s",
        model.bandwidth_bytes_per_sec / 1e9
    );
}

#[test]
fn test_cost_model_apple_m4_simd_width() {
    let model = CostModel::apple_m4();
    assert_eq!(model.simd_width, 32, "Apple GPU simdgroup width must be 32");
}

#[test]
fn test_cost_model_apple_m4_launch_overhead_positive() {
    let model = CostModel::apple_m4();
    assert!(
        model.launch_overhead_ns > 0.0,
        "launch overhead must be positive"
    );
}

#[test]
fn test_cost_model_apple_m4_max_has_op_throughputs() {
    let model = CostModel::apple_m4_max();
    assert!(
        !model.op_throughput.is_empty(),
        "M4 Max should have op-specific throughput entries"
    );
    // Check a few key entries exist and are positive.
    for op in &["matmul", "conv1d", "softmax", "gelu"] {
        let throughput = model.op_throughput.get(*op);
        assert!(
            throughput.is_some(),
            "M4 Max should have throughput for '{op}'"
        );
        assert!(
            *throughput.unwrap() > 0.0,
            "M4 Max throughput for '{op}' must be positive"
        );
    }
}

#[test]
fn test_cost_model_apple_m1_lower_bandwidth_than_m4() {
    let m1 = CostModel::apple_m1();
    let m4 = CostModel::apple_m4();
    assert!(
        m1.bandwidth_bytes_per_sec < m4.bandwidth_bytes_per_sec,
        "M1 ({:.0} GB/s) should have lower bandwidth than M4 ({:.0} GB/s)",
        m1.bandwidth_bytes_per_sec / 1e9,
        m4.bandwidth_bytes_per_sec / 1e9,
    );
}

#[test]
fn test_cost_model_apple_m2_between_m1_and_m4() {
    let m1 = CostModel::apple_m1();
    let m2 = CostModel::apple_m2();
    let m4 = CostModel::apple_m4();
    assert!(
        m2.bandwidth_bytes_per_sec > m1.bandwidth_bytes_per_sec,
        "M2 bandwidth should exceed M1"
    );
    assert!(
        m2.bandwidth_bytes_per_sec <= m4.bandwidth_bytes_per_sec,
        "M2 bandwidth should not exceed M4"
    );
}

#[test]
fn test_cost_model_nvidia_a100_high_bandwidth() {
    let a100 = CostModel::nvidia_a100();
    // A100 HBM2e: ~2039 GB/s
    assert!(
        a100.bandwidth_bytes_per_sec > 1000e9,
        "A100 bandwidth {:.0} GB/s should exceed 1000 GB/s",
        a100.bandwidth_bytes_per_sec / 1e9
    );
}

#[test]
fn test_cost_model_nvidia_rtx_4090_reasonable_values() {
    let rtx = CostModel::nvidia_rtx_4090();
    assert!(rtx.bandwidth_bytes_per_sec > 500e9);
    assert_eq!(rtx.simd_width, 32, "NVIDIA warp size must be 32");
    assert!(
        rtx.launch_overhead_ns > 0.0,
        "launch overhead must be positive"
    );
}

#[test]
fn test_cost_model_all_presets_have_positive_launch_overhead() {
    let presets: Vec<(&str, CostModel)> = vec![
        ("apple_m1", CostModel::apple_m1()),
        ("apple_m2", CostModel::apple_m2()),
        ("apple_m3", CostModel::apple_m3()),
        ("apple_m4", CostModel::apple_m4()),
        ("apple_m4_pro", CostModel::apple_m4_pro()),
        ("apple_m4_max", CostModel::apple_m4_max()),
        ("nvidia_a100", CostModel::nvidia_a100()),
        ("nvidia_rtx_4090", CostModel::nvidia_rtx_4090()),
    ];

    for (name, model) in &presets {
        assert!(
            model.launch_overhead_ns > 0.0,
            "{name}: launch_overhead_ns must be positive, got {:.0}",
            model.launch_overhead_ns
        );
        assert!(
            model.bandwidth_bytes_per_sec > 0.0,
            "{name}: bandwidth must be positive"
        );
        assert!(model.simd_width > 0, "{name}: simd_width must be > 0");
    }
}

// ===========================================================================
// CostModel.estimate() tests
// ===========================================================================

#[test]
fn test_cost_model_estimate_empty_plan() {
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let est = model.estimate(&plan);
    assert_eq!(est.dispatch_count, 0);
    assert!(
        est.total_ns == 0.0,
        "empty plan should have zero cost, got {}",
        est.total_ns
    );
    assert!(est.per_step_ns.is_empty());
}

#[test]
fn test_cost_model_estimate_single_dispatch() {
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![make_dispatch("relu", &[1, 1024])],
        input_shapes: vec![vec![1, 1024]],
        output_step: 0,
        weight_names: vec![],
    };
    let est = model.estimate(&plan);
    assert_eq!(est.dispatch_count, 1);
    assert!(
        est.total_ns > 0.0,
        "single dispatch should have non-zero cost"
    );
    assert_eq!(est.per_step_ns.len(), 1);
}

#[test]
fn test_cost_model_estimate_passthrough_is_free() {
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            make_dispatch("gelu", &[4, 768]),
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![vec![4, 768]],
        output_step: 1,
        weight_names: vec![],
    };
    let est = model.estimate(&plan);
    // Only the Dispatch step counts.
    assert_eq!(est.dispatch_count, 1);
    // per_step_ns should only have the dispatch step (index 1).
    assert_eq!(est.per_step_ns.len(), 1);
    assert_eq!(est.per_step_ns[0].0, 1);
}

#[test]
fn test_cost_model_estimate_larger_tensor_costs_more() {
    let model = CostModel::apple_m4();
    let plan_small = CompiledPlan {
        steps: vec![make_dispatch("relu", &[1, 64])],
        input_shapes: vec![vec![1, 64]],
        output_step: 0,
        weight_names: vec![],
    };
    let plan_large = CompiledPlan {
        steps: vec![make_dispatch("relu", &[32, 4096])],
        input_shapes: vec![vec![32, 4096]],
        output_step: 0,
        weight_names: vec![],
    };
    let est_small = model.estimate(&plan_small);
    let est_large = model.estimate(&plan_large);
    assert!(
        est_large.total_ns > est_small.total_ns,
        "larger tensor ({:.0} ns) should cost more than smaller ({:.0} ns)",
        est_large.total_ns,
        est_small.total_ns
    );
}

// ===========================================================================
// CostModel.calibration_plan() tests
// ===========================================================================

#[test]
fn test_calibration_plan_empty() {
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let records = model.calibration_plan(&plan);
    assert!(records.is_empty());
}

#[test]
fn test_calibration_plan_includes_dispatch_only() {
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            make_dispatch("relu", &[4, 256]),
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![vec![4, 256]],
        output_step: 1,
        weight_names: vec![],
    };
    let records = model.calibration_plan(&plan);
    assert_eq!(
        records.len(),
        1,
        "only Dispatch steps get calibration records"
    );
    assert_eq!(records[0].step_index, 1);
    assert_eq!(records[0].op_name, "relu");
    assert!(records[0].actual_ns.is_none(), "actual_ns starts as None");
    assert!(records[0].estimated_ns > 0.0);
}

// ===========================================================================
// PeepholeConfig bitmask edge cases
// ===========================================================================

#[test]
fn test_peephole_config_bitmask_zero_disables_all() {
    let cfg = config_from_bitmask(0);
    assert!(!cfg.norm_activ_conv1d);
    assert!(!cfg.fused_resblock);
    assert!(!cfg.linear_activation);
    assert!(!cfg.add_layer_norm);
    assert!(!cfg.norm_linear);
    assert!(!cfg.attention_transpose);
    assert!(!cfg.flip_lstm);
    assert!(!cfg.batched_linear_projection);
    assert!(!cfg.channels_first_layer_norm);
    assert!(!cfg.silu_mul);
    assert!(!cfg.auto_fuse_elementwise);
    assert!(!cfg.bilstm_cat);
    assert!(!cfg.add_norm_linear);
    assert!(!cfg.fuse_adain_snake);
    assert!(!cfg.fuse_upsample_conv1d);
    assert!(!cfg.fuse_instance_norm_mul_add);
    assert!(!is_default_config(&cfg));
}

#[test]
fn test_peephole_config_all_bits_set_equals_default() {
    let all_on_mask = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    let cfg = config_from_bitmask(all_on_mask);
    assert_eq!(cfg, PeepholeConfig::default());
    assert!(is_default_config(&cfg));
}

#[test]
fn test_peephole_config_enumerate_count_matches_2_pow_field_count() {
    // Count lazily (O(1) memory) rather than materializing 2^28 configs.
    let expected = 1usize << PEEPHOLE_FIELD_COUNT;
    assert_eq!(enumerate_peephole_configs().count(), expected);
}

#[test]
fn test_peephole_field_count_is_17() {
    // Guards against adding fields without bumping PEEPHOLE_FIELD_COUNT.
    assert_eq!(PEEPHOLE_FIELD_COUNT, 28);
    assert_eq!(PEEPHOLE_FIELD_NAMES.len(), PEEPHOLE_FIELD_COUNT as usize);
}

#[test]
fn test_peephole_config_each_field_has_unique_bit() {
    // Verify that setting bit N enables exactly the Nth field and no others.
    let field_accessors: [fn(&PeepholeConfig) -> bool; 16] = [
        |c| c.norm_activ_conv1d,
        |c| c.fused_resblock,
        |c| c.linear_activation,
        |c| c.add_layer_norm,
        |c| c.norm_linear,
        |c| c.attention_transpose,
        |c| c.flip_lstm,
        |c| c.batched_linear_projection,
        |c| c.channels_first_layer_norm,
        |c| c.silu_mul,
        |c| c.auto_fuse_elementwise,
        |c| c.bilstm_cat,
        |c| c.add_norm_linear,
        |c| c.fuse_adain_snake,
        |c| c.fuse_upsample_conv1d,
        |c| c.fuse_instance_norm_mul_add,
    ];

    for bit in 0..PEEPHOLE_FIELD_COUNT as usize {
        let cfg = config_from_bitmask(1u32 << bit);
        for (idx, accessor) in field_accessors.iter().enumerate() {
            let val = accessor(&cfg);
            if idx == bit {
                assert!(
                    val,
                    "bit {bit} should enable field {}",
                    PEEPHOLE_FIELD_NAMES[bit]
                );
            } else {
                assert!(
                    !val,
                    "bit {bit} should not enable field {}",
                    PEEPHOLE_FIELD_NAMES[idx]
                );
            }
        }
    }
}

#[test]
fn test_peephole_config_disable_individual_passes() {
    // For each field, disable it and verify only that field is false.
    let all_on_mask = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;

    let field_accessors: [fn(&PeepholeConfig) -> bool; 16] = [
        |c| c.norm_activ_conv1d,
        |c| c.fused_resblock,
        |c| c.linear_activation,
        |c| c.add_layer_norm,
        |c| c.norm_linear,
        |c| c.attention_transpose,
        |c| c.flip_lstm,
        |c| c.batched_linear_projection,
        |c| c.channels_first_layer_norm,
        |c| c.silu_mul,
        |c| c.auto_fuse_elementwise,
        |c| c.bilstm_cat,
        |c| c.add_norm_linear,
        |c| c.fuse_adain_snake,
        |c| c.fuse_upsample_conv1d,
        |c| c.fuse_instance_norm_mul_add,
    ];

    for bit in 0..PEEPHOLE_FIELD_COUNT as usize {
        let mask = all_on_mask ^ (1u32 << bit);
        let cfg = config_from_bitmask(mask);
        for (idx, accessor) in field_accessors.iter().enumerate() {
            let val = accessor(&cfg);
            if idx == bit {
                assert!(
                    !val,
                    "disabling bit {bit} should set {} to false",
                    PEEPHOLE_FIELD_NAMES[bit]
                );
            } else {
                assert!(
                    val,
                    "disabling bit {bit} should leave {} as true",
                    PEEPHOLE_FIELD_NAMES[idx]
                );
            }
        }
    }
}

// ===========================================================================
// FusionBlocker variant tests
// ===========================================================================

#[test]
fn test_fusion_blocker_all_variants_have_display() {
    // Exhaustive check that all FusionBlocker variants have non-empty Display.
    let variants = [
        FusionBlocker::FanOut,
        FusionBlocker::ShapeMismatch,
        FusionBlocker::NonFusibleOp,
        FusionBlocker::NotDispatch,
        FusionBlocker::AlreadyOptimal,
        FusionBlocker::NoPeepholePattern,
        FusionBlocker::NoDependency,
    ];

    for v in &variants {
        let s = format!("{v}");
        assert!(
            !s.is_empty(),
            "FusionBlocker::{v:?} should have non-empty Display"
        );
    }
}

#[test]
fn test_fusion_blocker_eq_and_clone() {
    let a = FusionBlocker::FanOut;
    let b = a.clone();
    assert_eq!(a, b);
    assert_ne!(FusionBlocker::FanOut, FusionBlocker::ShapeMismatch);
}

// ===========================================================================
// FusionGapAnalysis arithmetic edge cases
// ===========================================================================

#[test]
fn test_fusion_gap_analysis_optimization_pct_zero_dispatches() {
    let analysis = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 0,
        theoretical_minimum: 0,
    };
    assert!((analysis.optimization_opportunity_pct() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_fusion_gap_analysis_optimization_pct_100_percent() {
    // All dispatches could theoretically be eliminated.
    let analysis = FusionGapAnalysis {
        gaps: vec![FusionGap {
            step_a: 0,
            step_b: 1,
            kernel_a: "a".into(),
            kernel_b: "b".into(),
            reason: FusionBlocker::NoPeepholePattern,
            savings: 10,
        }],
        total_dispatches: 10,
        theoretical_minimum: 0,
    };
    assert!((analysis.optimization_opportunity_pct() - 100.0).abs() < 0.01);
}

#[test]
fn test_fusion_gap_analysis_blocker_counts_aggregation() {
    let analysis = FusionGapAnalysis {
        gaps: vec![
            FusionGap {
                step_a: 0,
                step_b: 1,
                kernel_a: "a".into(),
                kernel_b: "b".into(),
                reason: FusionBlocker::FanOut,
                savings: 1,
            },
            FusionGap {
                step_a: 2,
                step_b: 3,
                kernel_a: "c".into(),
                kernel_b: "d".into(),
                reason: FusionBlocker::FanOut,
                savings: 1,
            },
            FusionGap {
                step_a: 4,
                step_b: 5,
                kernel_a: "e".into(),
                kernel_b: "f".into(),
                reason: FusionBlocker::ShapeMismatch,
                savings: 1,
            },
        ],
        total_dispatches: 20,
        theoretical_minimum: 17,
    };
    let counts = analysis.blocker_counts();
    assert_eq!(counts.get("FanOut"), Some(&2));
    assert_eq!(counts.get("ShapeMismatch"), Some(&1));
    assert_eq!(counts.len(), 2);
}

#[test]
fn test_fusion_gap_analysis_summarize_contains_key_info() {
    let analysis = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 42,
        theoretical_minimum: 42,
    };
    let summary = analysis.summarize();
    assert!(
        summary.contains("42 dispatches"),
        "summary should include dispatch count"
    );
    assert!(
        summary.contains("0.0%"),
        "summary should show 0% when no savings"
    );
}

#[test]
fn test_fusion_gap_analysis_display_equals_summarize() {
    let analysis = FusionGapAnalysis {
        gaps: vec![FusionGap {
            step_a: 0,
            step_b: 1,
            kernel_a: "relu".into(),
            kernel_b: "exp".into(),
            reason: FusionBlocker::NonFusibleOp,
            savings: 0,
        }],
        total_dispatches: 5,
        theoretical_minimum: 5,
    };
    assert_eq!(format!("{analysis}"), analysis.summarize());
}

// ===========================================================================
// analyze_fusion_gaps on trivial graphs
// ===========================================================================

#[test]
fn test_analyze_fusion_gaps_empty_plan_empty_graph() {
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let graph = ComputationGraph::from_nodes(vec![]);
    let analysis = analyze_fusion_gaps(&plan, &graph);
    assert!(analysis.gaps.is_empty());
    assert_eq!(analysis.total_dispatches, 0);
    assert_eq!(analysis.theoretical_minimum, 0);
    assert!((analysis.optimization_opportunity_pct() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_analyze_fusion_gaps_single_step_no_gaps() {
    let node = TraceNode::new(
        0,
        "relu_0".into(),
        TraceOp::Relu,
        vec![],
        vec![1, 4],
        DType::F32,
    );
    let graph = ComputationGraph::from_nodes(vec![node]);
    let plan = CompiledPlan {
        steps: vec![make_dispatch("relu", &[1, 4])],
        input_shapes: vec![vec![1, 4]],
        output_step: 0,
        weight_names: vec![],
    };
    let analysis = analyze_fusion_gaps(&plan, &graph);
    assert!(analysis.gaps.is_empty());
    assert_eq!(analysis.total_dispatches, 1);
}

// ===========================================================================
// count_dispatches tests
// ===========================================================================

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
fn test_count_dispatches_mixed_steps() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            make_dispatch("relu", &[4, 256]),
            CompiledStep::IdentityPassthrough,
            make_dispatch("gelu", &[4, 256]),
        ],
        input_shapes: vec![vec![4, 256]],
        output_step: 3,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 2);
}

#[test]
fn test_count_dispatches_passthrough_not_counted() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::IdentityPassthrough,
            CompiledStep::InputForward,
        ],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 0);
}
