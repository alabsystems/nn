// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `cost_model.rs` — roofline cost model, hardware presets,
//! calibration, and CostEstimate methods.

use super::*;
use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use crate::trace_compile::{CompiledKernel, CompiledPlan, CompiledStep};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal `CompiledPlan` with no steps.
fn empty_plan() -> CompiledPlan {
    CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    }
}

/// Build a `CompiledKernel` with a given name and output shape.
fn make_kernel(name: &str, output_shape: &[usize]) -> CompiledKernel {
    let output_id = TensorNodeId::new(1);
    let node = TensorNode::new(
        output_id,
        TensorOpKind::Input {
            name: "x".to_string(),
            shape: output_shape.to_vec(),
        },
        output_shape.to_vec(),
    );
    let def = TensorKernelDef {
        name: name.to_string(),
        nodes: vec![node],
        output: output_id,
    };
    CompiledKernel::new(def)
}

/// Build a single-dispatch plan for the given op name and shape.
fn single_dispatch_plan(op_name: &str, shape: &[usize]) -> CompiledPlan {
    let kernel = make_kernel(op_name, shape);
    CompiledPlan {
        steps: vec![CompiledStep::Dispatch {
            kernel,
            weight_data: Default::default(),
            external_node_ids: None,
        }],
        input_shapes: vec![shape.to_vec()],
        output_step: 0,
        weight_names: vec![],
    }
}

// ---------------------------------------------------------------------------
// CostModel construction
// ---------------------------------------------------------------------------

#[test]
fn test_apple_m4_construction() {
    let model = CostModel::apple_m4();
    assert_eq!(model.launch_overhead_ns, 2000.0);
    assert_eq!(model.bandwidth_bytes_per_sec, 400e9);
    assert_eq!(model.simd_width, 32);
    assert!(
        model.op_throughput.is_empty(),
        "base M4 has no op-specific throughputs"
    );
}

#[test]
fn test_apple_m4_max_construction() {
    let model = CostModel::apple_m4_max();
    assert_eq!(model.launch_overhead_ns, 1500.0);
    assert_eq!(model.bandwidth_bytes_per_sec, 400e9);
    assert_eq!(model.simd_width, 32);
    assert!(
        !model.op_throughput.is_empty(),
        "M4 Max should have op-specific throughputs"
    );
    assert!(model.op_throughput.contains_key("matmul"));
    assert!(model.op_throughput.contains_key("softmax"));
}

#[test]
fn test_custom_cost_model_construction() {
    let mut op_throughput = HashMap::new();
    op_throughput.insert("custom_op".to_string(), 5e12);
    let model = CostModel {
        launch_overhead_ns: 1000.0,
        op_throughput,
        bandwidth_bytes_per_sec: 200e9,
        simd_width: 64,
    };
    assert_eq!(model.launch_overhead_ns, 1000.0);
    assert_eq!(model.bandwidth_bytes_per_sec, 200e9);
    assert_eq!(model.simd_width, 64);
    assert_eq!(model.op_throughput["custom_op"], 5e12);
}

// ---------------------------------------------------------------------------
// Dispatch cost estimation
// ---------------------------------------------------------------------------

#[test]
fn test_empty_plan_zero_cost() {
    let model = CostModel::apple_m4();
    let est = model.estimate(&empty_plan());
    assert_eq!(est.total_ns, 0.0);
    assert_eq!(est.dispatch_count, 0);
    assert!(est.per_step_ns.is_empty());
}

#[test]
fn test_single_dispatch_includes_launch_overhead() {
    let model = CostModel::apple_m4();
    let plan = single_dispatch_plan("gelu", &[1, 256]);
    let est = model.estimate(&plan);
    assert_eq!(est.dispatch_count, 1);
    assert!(
        est.total_ns >= model.launch_overhead_ns,
        "total_ns ({}) should be >= launch_overhead_ns ({})",
        est.total_ns,
        model.launch_overhead_ns
    );
    assert_eq!(est.per_step_ns.len(), 1);
}

#[test]
fn test_passthrough_steps_are_free() {
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::Passthrough {
                op_name: "reshape".to_string(),
                output_shape: vec![1, 256],
            },
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![vec![1, 256]],
        output_step: 1,
        weight_names: vec![],
    };
    let est = model.estimate(&plan);
    assert_eq!(est.total_ns, 0.0);
    assert_eq!(est.dispatch_count, 0);
}

#[test]
fn test_constant_value_step_is_free() {
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![CompiledStep::ConstantValue {
            value: 1.0,
            shape: vec![4],
        }],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let est = model.estimate(&plan);
    assert_eq!(est.total_ns, 0.0);
    assert_eq!(est.dispatch_count, 0);
}

#[test]
fn test_multiple_dispatches_cost_is_additive() {
    let model = CostModel::apple_m4();
    let kernel = make_kernel("relu", &[1, 1024]);
    let plan_1 = CompiledPlan {
        steps: vec![CompiledStep::Dispatch {
            kernel: kernel.clone(),
            weight_data: Default::default(),
            external_node_ids: None,
        }],
        input_shapes: vec![vec![1, 1024]],
        output_step: 0,
        weight_names: vec![],
    };
    let plan_2 = CompiledPlan {
        steps: vec![
            CompiledStep::Dispatch {
                kernel: kernel.clone(),
                weight_data: Default::default(),
                external_node_ids: None,
            },
            CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            },
        ],
        input_shapes: vec![vec![1, 1024]],
        output_step: 1,
        weight_names: vec![],
    };
    let est_1 = model.estimate(&plan_1);
    let est_2 = model.estimate(&plan_2);
    // Two dispatches should cost approximately 2x a single dispatch.
    assert!(
        (est_2.total_ns - 2.0 * est_1.total_ns).abs() < 1e-6,
        "two identical dispatches should cost exactly 2x single: {} vs 2*{}",
        est_2.total_ns,
        est_1.total_ns
    );
    assert_eq!(est_2.dispatch_count, 2);
}

// ---------------------------------------------------------------------------
// Memory bandwidth modeling
// ---------------------------------------------------------------------------

#[test]
fn test_m4_produces_reasonable_estimates() {
    let model = CostModel::apple_m4();
    let plan = single_dispatch_plan("matmul", &[1024, 1024]);
    let est = model.estimate(&plan);
    // 1M elements * 4 bytes * 2 (read+write) = 8 MB.
    // At 400 GB/s: ~20 ns memory time + 2000 ns launch.
    assert!(est.total_ns > 2000.0, "should include launch overhead");
    assert!(
        est.total_ns < 100_000.0,
        "1M elements should cost < 100 us on M4"
    );
}

#[test]
fn test_memory_bound_classification_elementwise() {
    // With base M4 (default 1 TFLOP/s, 400 GB/s):
    // arithmetic intensity = 1 FLOP / 8 bytes = 0.125 FLOP/byte
    // machine balance = 1e12 / 400e9 = 2.5 FLOP/byte
    // 0.125 < 2.5 => memory-bound.
    let model = CostModel::apple_m4();
    let records = model.calibration_plan(&single_dispatch_plan("relu", &[1024]));
    assert_eq!(records.len(), 1);
    assert!(
        records[0].is_memory_bound,
        "small elementwise op should be memory-bound on base M4"
    );
}

#[test]
fn test_higher_bandwidth_reduces_memory_cost() {
    // A100 has ~5x the bandwidth of M4 Max. For memory-bound ops,
    // A100 should produce lower cost estimates.
    let m4_max = CostModel::apple_m4_max();
    let a100 = CostModel::nvidia_a100();
    let plan = single_dispatch_plan("relu", &[1024, 1024]);
    let est_m4 = m4_max.estimate(&plan);
    let est_a100 = a100.estimate(&plan);
    assert!(
        est_a100.total_ns < est_m4.total_ns,
        "A100 ({:.1} ns) should be faster for memory-bound relu than M4 Max ({:.1} ns)",
        est_a100.total_ns,
        est_m4.total_ns
    );
}

// ---------------------------------------------------------------------------
// Compute throughput calculations
// ---------------------------------------------------------------------------

#[test]
fn test_known_op_uses_specific_throughput() {
    let m4_max = CostModel::apple_m4_max();
    let m4 = CostModel::apple_m4();
    // M4 Max has matmul throughput of 30 TFLOP/s; base M4 uses default 1 TFLOP/s.
    // For the same element count, M4 Max should be faster.
    let plan = single_dispatch_plan("matmul", &[1024, 1024]);
    let est_max = m4_max.estimate(&plan);
    let est_base = m4.estimate(&plan);
    assert!(
        est_max.total_ns < est_base.total_ns,
        "M4 Max ({:.1} ns) should be faster than base M4 ({:.1} ns) for matmul",
        est_max.total_ns,
        est_base.total_ns
    );
}

#[test]
fn test_unknown_op_uses_default_throughput() {
    let m4 = CostModel::apple_m4();
    let m4_max = CostModel::apple_m4_max();
    let plan = single_dispatch_plan("exotic_custom_op", &[1024]);
    let est_m4 = m4.estimate(&plan);
    let est_m4_max = m4_max.estimate(&plan);
    // Both use default throughput (1 TFLOP/s), same bandwidth.
    // Only difference is launch_overhead_ns (2000 vs 1500).
    let overhead_diff = m4.launch_overhead_ns - m4_max.launch_overhead_ns;
    let cost_diff = est_m4.total_ns - est_m4_max.total_ns;
    assert!(
        (cost_diff - overhead_diff).abs() < 1e-6,
        "cost difference ({cost_diff:.6}) should equal launch overhead difference ({overhead_diff:.6})"
    );
}

#[test]
fn test_lower_throughput_op_costs_more() {
    // Softmax (8 TFLOP/s) should cost more than matmul (30 TFLOP/s)
    // for the same element count, since max(compute, memory) will be
    // larger when compute_ns is larger.
    let model = CostModel::apple_m4_max();
    let elements = &[10000, 10000]; // 100M elements
    let est_matmul = model.estimate(&single_dispatch_plan("matmul", elements));
    let est_softmax = model.estimate(&single_dispatch_plan("softmax", elements));
    assert!(est_matmul.total_ns > 0.0);
    assert!(est_softmax.total_ns > 0.0);
    assert!(
        est_softmax.total_ns >= est_matmul.total_ns,
        "lower-throughput op (softmax: {:.1} ns) should cost >= \
         higher-throughput op (matmul: {:.1} ns)",
        est_softmax.total_ns,
        est_matmul.total_ns
    );
}

// ---------------------------------------------------------------------------
// Roofline model properties (compute-bound vs memory-bound)
// ---------------------------------------------------------------------------

#[test]
fn test_roofline_manual_check() {
    // Verify the roofline model against manual calculation for M4 Max.
    //   elements = 1024 * 1024 = 1_048_576
    //   matmul throughput = 30 TFLOP/s
    //   compute_ns = 1_048_576 / 30e12 * 1e9
    //   memory_ns  = 1_048_576 * 8 / 400e9 * 1e9
    //   occupancy  = 1.0 (1_048_576 is a multiple of 32)
    //   total = 1500 + max(compute_ns, memory_ns)
    let model = CostModel::apple_m4_max();
    let plan = single_dispatch_plan("matmul", &[1024, 1024]);
    let est = model.estimate(&plan);
    let elements = 1024.0 * 1024.0;
    let compute_ns = (elements / 30e12) * 1e9;
    let memory_ns = (elements * 4.0 * 2.0 / 400e9) * 1e9;
    let expected = 1500.0 + f64::max(compute_ns, memory_ns);
    assert!(
        (est.total_ns - expected).abs() < 1e-3,
        "roofline estimate ({:.6} ns) should match manual calculation ({:.6} ns)",
        est.total_ns,
        expected
    );
}

#[test]
fn test_occupancy_penalty_non_aligned() {
    let model = CostModel::apple_m4();
    let plan_aligned = single_dispatch_plan("relu", &[32]);
    let plan_unaligned = single_dispatch_plan("relu", &[33]);
    let est_aligned = model.estimate(&plan_aligned);
    let est_unaligned = model.estimate(&plan_unaligned);
    // Unaligned should have higher cost due to occupancy penalty.
    assert!(
        est_unaligned.total_ns > est_aligned.total_ns,
        "unaligned ({}) should cost more than aligned ({})",
        est_unaligned.total_ns,
        est_aligned.total_ns
    );
}

#[test]
fn test_occupancy_penalty_exactly_one_remainder() {
    // 33 elements: 32 full + 1 remainder. Occupancy = max(0.1, 1/32).
    let model = CostModel::apple_m4();
    let plan = single_dispatch_plan("relu", &[33]);
    let est = model.estimate(&plan);
    // Cost should be launch_overhead + max(compute, memory) / occupancy.
    // With occupancy = 1/32 = 0.03125, clamped to 0.1, the penalty is 10x.
    assert!(
        est.total_ns > model.launch_overhead_ns,
        "cost with occupancy penalty should exceed launch overhead"
    );
}

#[test]
fn test_cost_monotonic_with_dispatch_count() {
    let model = CostModel::apple_m4_max();
    let kernel = make_kernel("gelu", &[1, 1024]);
    let mut costs = Vec::new();
    for n in 1..=5 {
        let steps: Vec<CompiledStep> = (0..n)
            .map(|_| CompiledStep::Dispatch {
                kernel: kernel.clone(),
                weight_data: Default::default(),
                external_node_ids: None,
            })
            .collect();
        let plan = CompiledPlan {
            steps,
            input_shapes: vec![vec![1, 1024]],
            output_step: 0,
            weight_names: vec![],
        };
        costs.push(model.estimate(&plan).total_ns);
    }
    for i in 1..costs.len() {
        assert!(
            costs[i] > costs[i - 1],
            "cost should increase with dispatch count: {} dispatches ({:.1} ns) \
             <= {} dispatches ({:.1} ns)",
            i + 1,
            costs[i],
            i,
            costs[i - 1]
        );
    }
}

#[test]
fn test_larger_matrices_cost_more() {
    let model = CostModel::apple_m4_max();
    let sizes: &[&[usize]] = &[
        &[32, 32],     // 1K elements
        &[128, 128],   // 16K elements
        &[512, 512],   // 256K elements
        &[1024, 1024], // 1M elements
    ];
    let mut costs = Vec::new();
    for shape in sizes {
        costs.push(
            model
                .estimate(&single_dispatch_plan("matmul", shape))
                .total_ns,
        );
    }
    for i in 1..costs.len() {
        assert!(
            costs[i] > costs[i - 1],
            "larger matrix should cost more: shape {:?} ({:.1} ns) \
             <= shape {:?} ({:.1} ns)",
            sizes[i],
            costs[i],
            sizes[i - 1],
            costs[i - 1]
        );
    }
}

#[test]
fn test_zero_elements_returns_launch_overhead_only() {
    // A kernel with shape [0] has 0 elements. step_cost should return
    // launch_overhead_ns.
    let model = CostModel::apple_m4();
    let kernel = make_kernel("relu", &[0]);
    let plan = CompiledPlan {
        steps: vec![CompiledStep::Dispatch {
            kernel,
            weight_data: Default::default(),
            external_node_ids: None,
        }],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let est = model.estimate(&plan);
    assert!(
        (est.total_ns - model.launch_overhead_ns).abs() < 1e-6,
        "zero-element dispatch should cost exactly launch_overhead_ns: {} vs {}",
        est.total_ns,
        model.launch_overhead_ns
    );
}

// ---------------------------------------------------------------------------
// CostEstimate methods
// ---------------------------------------------------------------------------

#[test]
fn test_cost_estimate_summarize_not_empty() {
    let model = CostModel::apple_m4();
    let est = model.estimate(&single_dispatch_plan("softmax", &[1, 512]));
    let summary = est.summarize();
    assert!(summary.contains("CostEstimate:"));
    assert!(summary.contains("1 dispatches"));
}

#[test]
fn test_cost_estimate_display_matches_summarize() {
    let est = CostEstimate {
        total_ns: 5000.0,
        per_step_ns: vec![(0, 5000.0)],
        dispatch_count: 1,
    };
    assert_eq!(format!("{est}"), est.summarize());
}

#[test]
fn test_top_expensive_steps_sorted_descending() {
    let est = CostEstimate {
        total_ns: 10000.0,
        per_step_ns: vec![(0, 1000.0), (1, 5000.0), (2, 3000.0), (3, 1000.0)],
        dispatch_count: 4,
    };
    let top2 = est.top_expensive_steps(2);
    assert_eq!(top2.len(), 2);
    assert_eq!(top2[0], (1, 5000.0));
    assert_eq!(top2[1], (2, 3000.0));
}

#[test]
fn test_top_expensive_steps_n_exceeds_count() {
    let est = CostEstimate {
        total_ns: 2000.0,
        per_step_ns: vec![(0, 2000.0)],
        dispatch_count: 1,
    };
    let top5 = est.top_expensive_steps(5);
    assert_eq!(top5.len(), 1, "should return all when n > available");
}

#[test]
fn test_top_expensive_steps_empty() {
    let est = CostEstimate {
        total_ns: 0.0,
        per_step_ns: vec![],
        dispatch_count: 0,
    };
    assert!(est.top_expensive_steps(3).is_empty());
}

// ---------------------------------------------------------------------------
// Hardware presets
// ---------------------------------------------------------------------------

/// Helper: verify common constraints across all presets with op_throughput.
fn assert_preset_reasonable(model: &CostModel, name: &str) {
    assert!(
        model.launch_overhead_ns > 0.0,
        "{name}: launch_overhead_ns must be positive"
    );
    assert!(
        model.bandwidth_bytes_per_sec > 0.0,
        "{name}: bandwidth_bytes_per_sec must be positive"
    );
    assert_eq!(
        model.simd_width, 32,
        "{name}: simd_width should be 32 (GPU warp/simdgroup size)"
    );
    assert!(
        !model.op_throughput.is_empty(),
        "{name}: should have op-specific throughputs"
    );
    for (op, &tflops) in &model.op_throughput {
        assert!(
            tflops > 0.0,
            "{name}: throughput for '{op}' must be positive, got {tflops}"
        );
    }
}

#[test]
fn test_apple_m1_preset_fields() {
    let model = CostModel::apple_m1();
    assert_preset_reasonable(&model, "apple_m1");
    assert_eq!(model.launch_overhead_ns, 3000.0);
    assert_eq!(model.bandwidth_bytes_per_sec, 68.25e9);
}

#[test]
fn test_apple_m2_preset_fields() {
    let model = CostModel::apple_m2();
    assert_preset_reasonable(&model, "apple_m2");
    assert_eq!(model.launch_overhead_ns, 2500.0);
    assert_eq!(model.bandwidth_bytes_per_sec, 100e9);
}

#[test]
fn test_apple_m3_preset_fields() {
    let model = CostModel::apple_m3();
    assert_preset_reasonable(&model, "apple_m3");
    assert_eq!(model.launch_overhead_ns, 2000.0);
    assert_eq!(model.bandwidth_bytes_per_sec, 100e9);
}

#[test]
fn test_apple_m4_pro_preset_fields() {
    let model = CostModel::apple_m4_pro();
    assert_preset_reasonable(&model, "apple_m4_pro");
    assert_eq!(model.launch_overhead_ns, 1800.0);
    assert_eq!(model.bandwidth_bytes_per_sec, 273e9);
}

#[test]
fn test_nvidia_a100_preset_fields() {
    let model = CostModel::nvidia_a100();
    assert_preset_reasonable(&model, "nvidia_a100");
    assert_eq!(model.launch_overhead_ns, 5000.0);
    assert_eq!(model.bandwidth_bytes_per_sec, 2039e9);
}

#[test]
fn test_nvidia_rtx_4090_preset_fields() {
    let model = CostModel::nvidia_rtx_4090();
    assert_preset_reasonable(&model, "nvidia_rtx_4090");
    assert_eq!(model.launch_overhead_ns, 7000.0);
    assert_eq!(model.bandwidth_bytes_per_sec, 1008e9);
}

#[test]
fn test_apple_silicon_generation_ordering() {
    // M1 < M2 < M3 < M4 Pro < M4 Max in matmul throughput.
    let m1 = CostModel::apple_m1();
    let m2 = CostModel::apple_m2();
    let m3 = CostModel::apple_m3();
    let m4_pro = CostModel::apple_m4_pro();
    let m4_max = CostModel::apple_m4_max();

    let t_m1 = m1.op_throughput["matmul"];
    let t_m2 = m2.op_throughput["matmul"];
    let t_m3 = m3.op_throughput["matmul"];
    let t_m4_pro = m4_pro.op_throughput["matmul"];
    let t_m4_max = m4_max.op_throughput["matmul"];

    assert!(t_m1 < t_m2, "M1 matmul < M2 matmul");
    assert!(t_m2 < t_m3, "M2 matmul < M3 matmul");
    assert!(t_m3 < t_m4_pro, "M3 matmul < M4 Pro matmul");
    assert!(t_m4_pro < t_m4_max, "M4 Pro matmul < M4 Max matmul");

    // Base M4 has no op_throughput entries — it uses the default.
    let m4 = CostModel::apple_m4();
    assert!(
        m4.op_throughput.is_empty(),
        "base M4 should have no op-specific throughputs"
    );
}

#[test]
fn test_apple_silicon_bandwidth_ordering() {
    let m1 = CostModel::apple_m1();
    let m2 = CostModel::apple_m2();
    let m4_pro = CostModel::apple_m4_pro();
    let m4_max = CostModel::apple_m4_max();

    assert!(m1.bandwidth_bytes_per_sec < m2.bandwidth_bytes_per_sec);
    assert!(m2.bandwidth_bytes_per_sec < m4_pro.bandwidth_bytes_per_sec);
    assert!(m4_pro.bandwidth_bytes_per_sec < m4_max.bandwidth_bytes_per_sec);
}

#[test]
fn test_nvidia_higher_bandwidth_than_apple() {
    let m4_max = CostModel::apple_m4_max();
    let a100 = CostModel::nvidia_a100();
    let rtx4090 = CostModel::nvidia_rtx_4090();
    assert!(a100.bandwidth_bytes_per_sec > m4_max.bandwidth_bytes_per_sec);
    assert!(rtx4090.bandwidth_bytes_per_sec > m4_max.bandwidth_bytes_per_sec);
}

#[test]
fn test_nvidia_higher_dispatch_overhead_than_apple() {
    let m4_max = CostModel::apple_m4_max();
    let a100 = CostModel::nvidia_a100();
    let rtx4090 = CostModel::nvidia_rtx_4090();
    assert!(a100.launch_overhead_ns > m4_max.launch_overhead_ns);
    assert!(rtx4090.launch_overhead_ns > m4_max.launch_overhead_ns);
}

#[test]
fn test_dispatch_overhead_ranges() {
    // Apple Silicon: 1-3 us.
    let apple_models: Vec<(&str, CostModel)> = vec![
        ("M1", CostModel::apple_m1()),
        ("M2", CostModel::apple_m2()),
        ("M3", CostModel::apple_m3()),
        ("M4", CostModel::apple_m4()),
        ("M4 Pro", CostModel::apple_m4_pro()),
        ("M4 Max", CostModel::apple_m4_max()),
    ];
    for (name, model) in &apple_models {
        assert!(
            (1000.0..=3000.0).contains(&model.launch_overhead_ns),
            "{name}: Apple launch overhead ({:.0} ns) should be 1-3 us",
            model.launch_overhead_ns
        );
    }
    // NVIDIA: 5-10 us.
    let nvidia_models: Vec<(&str, CostModel)> = vec![
        ("A100", CostModel::nvidia_a100()),
        ("RTX 4090", CostModel::nvidia_rtx_4090()),
    ];
    for (name, model) in &nvidia_models {
        assert!(
            (5000.0..=10000.0).contains(&model.launch_overhead_ns),
            "{name}: NVIDIA launch overhead ({:.0} ns) should be 5-10 us",
            model.launch_overhead_ns
        );
    }
}

#[test]
fn test_all_presets_produce_positive_estimates() {
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
    let plan = single_dispatch_plan("matmul", &[512, 512]);
    for (name, model) in &presets {
        let est = model.estimate(&plan);
        assert!(
            est.total_ns > 0.0,
            "{name}: total_ns should be positive, got {:.6}",
            est.total_ns
        );
        assert_eq!(est.dispatch_count, 1, "{name}: should have 1 dispatch");
    }
}

#[test]
fn test_newer_apple_silicon_faster_on_matmul() {
    let m1 = CostModel::apple_m1();
    let m2 = CostModel::apple_m2();
    let m3 = CostModel::apple_m3();
    let plan = single_dispatch_plan("matmul", &[1024, 1024]);
    let est_m1 = m1.estimate(&plan);
    let est_m2 = m2.estimate(&plan);
    let est_m3 = m3.estimate(&plan);
    assert!(
        est_m2.total_ns < est_m1.total_ns,
        "M2 should be faster than M1"
    );
    assert!(
        est_m3.total_ns < est_m2.total_ns,
        "M3 should be faster than M2"
    );
}

#[test]
fn test_rtx_4090_higher_matmul_throughput_than_a100() {
    let a100 = CostModel::nvidia_a100();
    let rtx4090 = CostModel::nvidia_rtx_4090();
    assert!(
        rtx4090.op_throughput["matmul"] > a100.op_throughput["matmul"],
        "RTX 4090 matmul throughput should exceed A100"
    );
}

#[test]
fn test_a100_faster_than_rtx_4090_for_memory_bound() {
    // A100 HBM2e bandwidth > RTX 4090 GDDR6X bandwidth.
    let a100 = CostModel::nvidia_a100();
    let rtx4090 = CostModel::nvidia_rtx_4090();
    let plan = single_dispatch_plan("relu", &[1024, 1024]);
    let est_a100 = a100.estimate(&plan);
    let est_rtx4090 = rtx4090.estimate(&plan);
    assert!(
        est_a100.total_ns < est_rtx4090.total_ns,
        "A100 should be faster on memory-bound workload due to higher bandwidth"
    );
}

// ---------------------------------------------------------------------------
// Calibration plan & report
// ---------------------------------------------------------------------------

#[test]
fn test_calibration_plan_empty() {
    let model = CostModel::apple_m4();
    let records = model.calibration_plan(&empty_plan());
    assert!(records.is_empty());
}

#[test]
fn test_calibration_plan_matches_dispatch_count() {
    let model = CostModel::apple_m4_max();
    let k1 = make_kernel("matmul", &[512, 512]);
    let k2 = make_kernel("gelu", &[1, 1024]);
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::Dispatch {
                kernel: k1,
                weight_data: Default::default(),
                external_node_ids: None,
            },
            CompiledStep::Passthrough {
                op_name: "reshape".to_string(),
                output_shape: vec![1, 512, 512],
            },
            CompiledStep::Dispatch {
                kernel: k2,
                weight_data: Default::default(),
                external_node_ids: None,
            },
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![vec![512, 512]],
        output_step: 3,
        weight_names: vec![],
    };
    let records = model.calibration_plan(&plan);
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].step_index, 1);
    assert_eq!(records[0].op_name, "matmul");
    assert_eq!(records[1].step_index, 3);
    assert_eq!(records[1].op_name, "gelu");
    for r in &records {
        assert!(r.actual_ns.is_none());
        assert!(r.estimated_ns > 0.0);
    }
}

#[test]
fn test_calibration_estimated_matches_estimate() {
    let model = CostModel::apple_m4_max();
    let plan = single_dispatch_plan("softmax", &[1, 2048]);
    let est = model.estimate(&plan);
    let records = model.calibration_plan(&plan);
    assert_eq!(records.len(), 1);
    assert!(
        (records[0].estimated_ns - est.total_ns).abs() < 1e-6,
        "calibration estimated_ns ({:.6}) should match estimate total_ns ({:.6})",
        records[0].estimated_ns,
        est.total_ns
    );
}

#[test]
fn test_calibration_report_no_actuals() {
    let records = vec![CalibrationRecord {
        step_index: 0,
        estimated_ns: 5000.0,
        actual_ns: None,
        op_name: "matmul".to_string(),
        is_memory_bound: false,
    }];
    let report = CalibrationReport::from_records(&records);
    assert_eq!(report.mean_absolute_error_ns, 0.0);
    assert!(report.correlation.is_nan());
}

#[test]
fn test_calibration_report_empty_records() {
    let report = CalibrationReport::from_records(&[]);
    assert_eq!(report.mean_absolute_error_ns, 0.0);
    assert!(report.correlation.is_nan());
}

#[test]
fn test_calibration_report_perfect_prediction() {
    let records = vec![
        CalibrationRecord {
            step_index: 0,
            estimated_ns: 1000.0,
            actual_ns: Some(1000.0),
            op_name: "a".to_string(),
            is_memory_bound: true,
        },
        CalibrationRecord {
            step_index: 1,
            estimated_ns: 3000.0,
            actual_ns: Some(3000.0),
            op_name: "b".to_string(),
            is_memory_bound: false,
        },
        CalibrationRecord {
            step_index: 2,
            estimated_ns: 5000.0,
            actual_ns: Some(5000.0),
            op_name: "c".to_string(),
            is_memory_bound: false,
        },
    ];
    let report = CalibrationReport::from_records(&records);
    assert!(
        report.mean_absolute_error_ns.abs() < 1e-10,
        "MAE should be 0"
    );
    assert!(report.max_overestimate_ns.abs() < 1e-10);
    assert!(report.max_underestimate_ns.abs() < 1e-10);
    assert!(
        (report.correlation - 1.0).abs() < 1e-10,
        "correlation should be 1.0, got {}",
        report.correlation
    );
}

#[test]
fn test_calibration_report_overestimate_and_underestimate() {
    let records = vec![
        CalibrationRecord {
            step_index: 0,
            estimated_ns: 5000.0,
            actual_ns: Some(3000.0), // overestimate by 2000
            op_name: "a".to_string(),
            is_memory_bound: true,
        },
        CalibrationRecord {
            step_index: 1,
            estimated_ns: 1000.0,
            actual_ns: Some(4000.0), // underestimate by 3000
            op_name: "b".to_string(),
            is_memory_bound: false,
        },
    ];
    let report = CalibrationReport::from_records(&records);
    assert!(
        (report.mean_absolute_error_ns - 2500.0).abs() < 1e-6,
        "MAE should be 2500"
    );
    assert!(
        (report.max_overestimate_ns - 2000.0).abs() < 1e-6,
        "max overestimate should be 2000"
    );
    assert!(
        (report.max_underestimate_ns - 3000.0).abs() < 1e-6,
        "max underestimate should be 3000"
    );
    assert!(report.correlation < 0.0, "should be negative correlation");
}

#[test]
fn test_calibration_report_single_actual_nan_correlation() {
    let records = vec![CalibrationRecord {
        step_index: 0,
        estimated_ns: 2000.0,
        actual_ns: Some(2500.0),
        op_name: "a".to_string(),
        is_memory_bound: true,
    }];
    let report = CalibrationReport::from_records(&records);
    assert!((report.mean_absolute_error_ns - 500.0).abs() < 1e-6);
    assert!(report.correlation.is_nan());
}

#[test]
fn test_calibration_report_mixed_profiled_and_unprofiled() {
    let records = vec![
        CalibrationRecord {
            step_index: 0,
            estimated_ns: 1000.0,
            actual_ns: Some(1200.0),
            op_name: "a".to_string(),
            is_memory_bound: true,
        },
        CalibrationRecord {
            step_index: 1,
            estimated_ns: 5000.0,
            actual_ns: None,
            op_name: "b".to_string(),
            is_memory_bound: false,
        },
        CalibrationRecord {
            step_index: 2,
            estimated_ns: 3000.0,
            actual_ns: Some(2800.0),
            op_name: "c".to_string(),
            is_memory_bound: false,
        },
    ];
    let report = CalibrationReport::from_records(&records);
    assert!(
        (report.mean_absolute_error_ns - 200.0).abs() < 1e-6,
        "MAE should be 200 (only profiled records)"
    );
}

// ---------------------------------------------------------------------------
// Calibration plausibility tests
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_overhead_dominates_many_small_dispatches() {
    // With many tiny dispatches (1-element kernels), launch overhead should
    // dominate the total cost. The compute and memory time for 1 element is
    // negligible compared to the ~2000 ns launch overhead.
    let model = CostModel::apple_m4();
    let kernel = make_kernel("relu", &[1]);
    let n = 100;
    let steps: Vec<CompiledStep> = (0..n)
        .map(|_| CompiledStep::Dispatch {
            kernel: kernel.clone(),
            weight_data: Default::default(),
            external_node_ids: None,
        })
        .collect();
    let plan = CompiledPlan {
        steps,
        input_shapes: vec![vec![1]],
        output_step: 0,
        weight_names: vec![],
    };
    let est = model.estimate(&plan);
    let pure_overhead = n as f64 * model.launch_overhead_ns;
    // Launch overhead should account for at least 90% of total cost.
    let overhead_fraction = pure_overhead / est.total_ns;
    assert!(
        overhead_fraction > 0.90,
        "launch overhead ({:.1} ns) should be >90% of total cost ({:.1} ns), \
         fraction = {:.4}",
        pure_overhead,
        est.total_ns,
        overhead_fraction
    );
}

#[test]
fn test_m4_cost_geq_m4_max_cost_all_ops() {
    // apple_m4() should produce cost estimates >= apple_m4_max() for every
    // known op. M4 Max has higher throughput and lower launch overhead.
    let m4 = CostModel::apple_m4();
    let m4_max = CostModel::apple_m4_max();
    let ops = [
        "matmul",
        "conv1d",
        "conv2d",
        "softmax",
        "gelu",
        "relu",
        "snake",
        "silu",
        "layer_norm",
        "instance_norm",
    ];
    for op in &ops {
        let plan = single_dispatch_plan(op, &[1024, 1024]);
        let est_m4 = m4.estimate(&plan);
        let est_max = m4_max.estimate(&plan);
        assert!(
            est_m4.total_ns >= est_max.total_ns,
            "base M4 cost ({:.1} ns) should be >= M4 Max cost ({:.1} ns) for op '{op}'",
            est_m4.total_ns,
            est_max.total_ns
        );
    }
}

#[test]
fn test_increasing_launch_overhead_increases_cost() {
    // Increasing the launch_overhead_ns parameter should increase total cost
    // for any non-empty plan.
    let plan = single_dispatch_plan("relu", &[256, 256]);
    let overheads = [500.0, 1000.0, 2000.0, 5000.0, 10000.0];
    let mut prev_cost = 0.0;
    for &overhead in &overheads {
        let model = CostModel {
            launch_overhead_ns: overhead,
            op_throughput: HashMap::new(),
            bandwidth_bytes_per_sec: 400e9,
            simd_width: 32,
        };
        let est = model.estimate(&plan);
        assert!(
            est.total_ns > prev_cost,
            "higher launch overhead ({:.0} ns) should produce higher cost \
             ({:.1} ns) than previous ({:.1} ns)",
            overhead,
            est.total_ns,
            prev_cost
        );
        prev_cost = est.total_ns;
    }
}

#[test]
fn test_cost_estimate_serde_roundtrip() {
    // CostEstimate derives Serialize + Deserialize. Verify roundtrip.
    let original = CostEstimate {
        total_ns: 12345.678,
        per_step_ns: vec![(0, 5000.0), (2, 4000.0), (5, 3345.678)],
        dispatch_count: 3,
    };
    let json = serde_json::to_string(&original).expect("serialize CostEstimate");
    let restored: CostEstimate = serde_json::from_str(&json).expect("deserialize CostEstimate");
    assert!(
        (restored.total_ns - original.total_ns).abs() < 1e-10,
        "total_ns roundtrip mismatch: {} vs {}",
        restored.total_ns,
        original.total_ns
    );
    assert_eq!(restored.per_step_ns.len(), original.per_step_ns.len());
    for (i, ((orig_idx, orig_ns), (rest_idx, rest_ns))) in original
        .per_step_ns
        .iter()
        .zip(restored.per_step_ns.iter())
        .enumerate()
    {
        assert_eq!(orig_idx, rest_idx, "step index mismatch at position {i}");
        assert!(
            (orig_ns - rest_ns).abs() < 1e-10,
            "step ns mismatch at position {i}: {orig_ns} vs {rest_ns}"
        );
    }
    assert_eq!(restored.dispatch_count, original.dispatch_count);
}

#[test]
fn test_cost_estimate_serde_roundtrip_empty() {
    // Empty CostEstimate should also roundtrip cleanly.
    let original = CostEstimate {
        total_ns: 0.0,
        per_step_ns: vec![],
        dispatch_count: 0,
    };
    let json = serde_json::to_string(&original).expect("serialize empty CostEstimate");
    let restored: CostEstimate =
        serde_json::from_str(&json).expect("deserialize empty CostEstimate");
    assert_eq!(restored.total_ns, 0.0);
    assert!(restored.per_step_ns.is_empty());
    assert_eq!(restored.dispatch_count, 0);
}

#[test]
fn test_n_dispatches_higher_cost_than_one_dispatch() {
    // A plan with N dispatches must have strictly higher cost than a plan
    // with 1 dispatch (same kernel).
    let model = CostModel::apple_m4_max();
    let kernel = make_kernel("softmax", &[1, 512]);
    let plan_1 = CompiledPlan {
        steps: vec![CompiledStep::Dispatch {
            kernel: kernel.clone(),
            weight_data: Default::default(),
            external_node_ids: None,
        }],
        input_shapes: vec![vec![1, 512]],
        output_step: 0,
        weight_names: vec![],
    };
    for n in [2, 5, 10, 50] {
        let steps: Vec<CompiledStep> = (0..n)
            .map(|_| CompiledStep::Dispatch {
                kernel: kernel.clone(),
                weight_data: Default::default(),
                external_node_ids: None,
            })
            .collect();
        let plan_n = CompiledPlan {
            steps,
            input_shapes: vec![vec![1, 512]],
            output_step: 0,
            weight_names: vec![],
        };
        let est_1 = model.estimate(&plan_1);
        let est_n = model.estimate(&plan_n);
        assert!(
            est_n.total_ns > est_1.total_ns,
            "{n}-dispatch plan ({:.1} ns) should cost more than 1-dispatch plan ({:.1} ns)",
            est_n.total_ns,
            est_1.total_ns
        );
    }
}

// ---------------------------------------------------------------------------
// CostModel::calibrate and CalibrationReport extensions
// ---------------------------------------------------------------------------

#[test]
fn test_calibrate_perfect_calibration() {
    // When predicted == actual for all entries, mean_error_ratio = 1.0.
    let predictions = vec![
        ("matmul".to_string(), 5000.0),
        ("softmax".to_string(), 3000.0),
        ("relu".to_string(), 1000.0),
    ];
    let actuals = vec![
        ("matmul".to_string(), 5000.0),
        ("softmax".to_string(), 3000.0),
        ("relu".to_string(), 1000.0),
    ];
    let report =
        CostModel::calibrate(&predictions, &actuals).expect("perfect calibration should succeed");

    assert!(
        (report.mean_error_ratio - 1.0).abs() < 1e-10,
        "mean_error_ratio should be 1.0 for perfect calibration, got {}",
        report.mean_error_ratio
    );
    assert!(
        (report.max_error_ratio - 1.0).abs() < 1e-10,
        "max_error_ratio should be 1.0 for perfect calibration, got {}",
        report.max_error_ratio
    );
    assert!(
        report.mean_absolute_error_ns.abs() < 1e-10,
        "MAE should be 0 for perfect calibration"
    );
    assert_eq!(report.entries.len(), 3);
}

#[test]
fn test_calibrate_2x_overestimate() {
    // Predicted = 2 * actual for all entries.
    let predictions = vec![
        ("matmul".to_string(), 10000.0),
        ("softmax".to_string(), 6000.0),
    ];
    let actuals = vec![
        ("matmul".to_string(), 5000.0),
        ("softmax".to_string(), 3000.0),
    ];
    let report = CostModel::calibrate(&predictions, &actuals)
        .expect("2x overestimate calibration should succeed");

    assert!(
        (report.mean_error_ratio - 2.0).abs() < 1e-10,
        "mean_error_ratio should be 2.0, got {}",
        report.mean_error_ratio
    );
    assert!(
        (report.max_error_ratio - 2.0).abs() < 1e-10,
        "max_error_ratio should be 2.0, got {}",
        report.max_error_ratio
    );
    // MAE = (5000 + 3000) / 2 = 4000
    assert!(
        (report.mean_absolute_error_ns - 4000.0).abs() < 1e-6,
        "MAE should be 4000, got {}",
        report.mean_absolute_error_ns
    );
}

#[test]
fn test_calibrate_mixed_accuracy() {
    // matmul: predicted 2x actual (ratio=2.0)
    // softmax: predicted == actual (ratio=1.0)
    // relu: predicted 0.5x actual (ratio=0.5)
    let predictions = vec![
        ("matmul".to_string(), 10000.0),
        ("softmax".to_string(), 3000.0),
        ("relu".to_string(), 500.0),
    ];
    let actuals = vec![
        ("matmul".to_string(), 5000.0),
        ("softmax".to_string(), 3000.0),
        ("relu".to_string(), 1000.0),
    ];
    let report = CostModel::calibrate(&predictions, &actuals)
        .expect("mixed accuracy calibration should succeed");

    // mean_error_ratio = (2.0 + 1.0 + 0.5) / 3 = 3.5 / 3 ~= 1.1667
    let expected_mean = (2.0 + 1.0 + 0.5) / 3.0;
    assert!(
        (report.mean_error_ratio - expected_mean).abs() < 1e-10,
        "mean_error_ratio should be {expected_mean:.4}, got {:.4}",
        report.mean_error_ratio
    );
    // max_error_ratio = 2.0 (matmul)
    assert!(
        (report.max_error_ratio - 2.0).abs() < 1e-10,
        "max_error_ratio should be 2.0, got {}",
        report.max_error_ratio
    );
    assert_eq!(report.entries.len(), 3);

    // Verify adjustment_factors: actual/predicted for each.
    let factors = report.adjustment_factors();
    assert_eq!(factors.len(), 3);
    assert!(
        (factors["matmul"] - 0.5).abs() < 1e-10,
        "matmul factor should be 0.5"
    );
    assert!(
        (factors["softmax"] - 1.0).abs() < 1e-10,
        "softmax factor should be 1.0"
    );
    assert!(
        (factors["relu"] - 2.0).abs() < 1e-10,
        "relu factor should be 2.0"
    );
}

#[test]
fn test_calibrate_empty_predictions_error() {
    let predictions: Vec<(String, f64)> = vec![];
    let actuals = vec![("matmul".to_string(), 5000.0)];
    let result = CostModel::calibrate(&predictions, &actuals);
    assert!(result.is_err(), "empty predictions should produce error");
    assert!(
        matches!(result.unwrap_err(), CalibrationError::NoMatchingSteps),
        "should be NoMatchingSteps"
    );
}

#[test]
fn test_calibrate_empty_actuals_error() {
    let predictions = vec![("matmul".to_string(), 5000.0)];
    let actuals: Vec<(String, f64)> = vec![];
    let result = CostModel::calibrate(&predictions, &actuals);
    assert!(result.is_err(), "empty actuals should produce error");
    assert!(
        matches!(result.unwrap_err(), CalibrationError::NoMatchingSteps),
        "should be NoMatchingSteps"
    );
}

#[test]
fn test_calibrate_no_matching_names_error() {
    let predictions = vec![("matmul".to_string(), 5000.0)];
    let actuals = vec![("conv1d".to_string(), 3000.0)];
    let result = CostModel::calibrate(&predictions, &actuals);
    assert!(result.is_err(), "no matching names should produce error");
    assert!(
        matches!(result.unwrap_err(), CalibrationError::NoMatchingSteps),
        "should be NoMatchingSteps"
    );
}

#[test]
fn test_calibrate_non_positive_actual_error() {
    let predictions = vec![("matmul".to_string(), 5000.0)];
    let actuals = vec![("matmul".to_string(), 0.0)];
    let result = CostModel::calibrate(&predictions, &actuals);
    assert!(result.is_err(), "zero actual_ns should produce error");
    match result.unwrap_err() {
        CalibrationError::NonPositiveActual { name, actual_ns } => {
            assert_eq!(name, "matmul");
            assert_eq!(actual_ns, 0.0);
        }
        other => panic!("expected NonPositiveActual, got {other:?}"),
    }
}

#[test]
fn test_calibrate_negative_actual_error() {
    let predictions = vec![("softmax".to_string(), 3000.0)];
    let actuals = vec![("softmax".to_string(), -100.0)];
    let result = CostModel::calibrate(&predictions, &actuals);
    assert!(result.is_err(), "negative actual_ns should produce error");
    assert!(matches!(
        result.unwrap_err(),
        CalibrationError::NonPositiveActual { .. }
    ));
}

#[test]
fn test_calibrate_partial_name_overlap() {
    // Only 'matmul' is in both; 'softmax' and 'conv1d' are disjoint.
    let predictions = vec![
        ("matmul".to_string(), 5000.0),
        ("softmax".to_string(), 3000.0),
    ];
    let actuals = vec![
        ("matmul".to_string(), 4000.0),
        ("conv1d".to_string(), 2000.0),
    ];
    let report =
        CostModel::calibrate(&predictions, &actuals).expect("partial overlap should succeed");
    assert_eq!(report.entries.len(), 1, "only matmul should match");
    assert_eq!(report.entries[0].step_name, "matmul");
    // ratio = 5000 / 4000 = 1.25
    assert!(
        (report.mean_error_ratio - 1.25).abs() < 1e-10,
        "mean_error_ratio should be 1.25, got {}",
        report.mean_error_ratio
    );
}

#[test]
fn test_calibration_report_summary_contains_fields() {
    let predictions = vec![("matmul".to_string(), 5000.0), ("relu".to_string(), 1000.0)];
    let actuals = vec![("matmul".to_string(), 4000.0), ("relu".to_string(), 1200.0)];
    let report = CostModel::calibrate(&predictions, &actuals).expect("calibration should succeed");
    let summary = report.summary();
    assert!(
        summary.contains("Calibration Report"),
        "summary should contain header"
    );
    assert!(
        summary.contains("Mean absolute error"),
        "summary should contain MAE"
    );
    assert!(
        summary.contains("Mean error ratio"),
        "summary should contain mean error ratio"
    );
    assert!(
        summary.contains("Max error ratio"),
        "summary should contain max error ratio"
    );
    assert!(
        summary.contains("matmul"),
        "summary should list matmul entry"
    );
    assert!(summary.contains("relu"), "summary should list relu entry");
    assert!(
        summary.contains("Entries: 2"),
        "summary should show 2 entries"
    );
}

#[test]
fn test_calibration_report_adjustment_factors_empty_entries() {
    // from_records produces empty entries, so adjustment_factors is empty.
    let records = vec![CalibrationRecord {
        step_index: 0,
        estimated_ns: 5000.0,
        actual_ns: Some(4000.0),
        op_name: "matmul".to_string(),
        is_memory_bound: false,
    }];
    let report = CalibrationReport::from_records(&records);
    let factors = report.adjustment_factors();
    assert!(
        factors.is_empty(),
        "from_records should produce empty adjustment_factors"
    );
}

#[test]
fn test_calibration_report_adjustment_factors_sorted() {
    // BTreeMap should be sorted by key.
    let predictions = vec![
        ("relu".to_string(), 1000.0),
        ("matmul".to_string(), 5000.0),
        ("conv1d".to_string(), 3000.0),
    ];
    let actuals = vec![
        ("relu".to_string(), 800.0),
        ("matmul".to_string(), 4000.0),
        ("conv1d".to_string(), 2500.0),
    ];
    let report = CostModel::calibrate(&predictions, &actuals).expect("calibration should succeed");
    let factors = report.adjustment_factors();
    let keys: Vec<&String> = factors.keys().collect();
    assert_eq!(keys, vec!["conv1d", "matmul", "relu"]);
}

#[test]
fn test_from_records_populates_error_ratios() {
    // Verify that from_records now also populates mean_error_ratio
    // and max_error_ratio for records with positive actuals.
    let records = vec![
        CalibrationRecord {
            step_index: 0,
            estimated_ns: 10000.0,
            actual_ns: Some(5000.0), // ratio = 2.0
            op_name: "a".to_string(),
            is_memory_bound: false,
        },
        CalibrationRecord {
            step_index: 1,
            estimated_ns: 3000.0,
            actual_ns: Some(3000.0), // ratio = 1.0
            op_name: "b".to_string(),
            is_memory_bound: false,
        },
    ];
    let report = CalibrationReport::from_records(&records);
    // mean_error_ratio = (2.0 + 1.0) / 2 = 1.5
    assert!(
        (report.mean_error_ratio - 1.5).abs() < 1e-10,
        "mean_error_ratio should be 1.5, got {}",
        report.mean_error_ratio
    );
    // max_error_ratio = 2.0
    assert!(
        (report.max_error_ratio - 2.0).abs() < 1e-10,
        "max_error_ratio should be 2.0, got {}",
        report.max_error_ratio
    );
}

#[test]
fn test_from_records_no_actuals_nan_ratios() {
    let records = vec![CalibrationRecord {
        step_index: 0,
        estimated_ns: 5000.0,
        actual_ns: None,
        op_name: "matmul".to_string(),
        is_memory_bound: false,
    }];
    let report = CalibrationReport::from_records(&records);
    assert!(
        report.mean_error_ratio.is_nan(),
        "mean_error_ratio should be NaN with no actuals"
    );
    assert!(
        report.max_error_ratio.is_nan(),
        "max_error_ratio should be NaN with no actuals"
    );
}

#[test]
fn test_calibrate_single_entry_nan_correlation() {
    // With only 1 matched entry, correlation should be NaN.
    let predictions = vec![("matmul".to_string(), 5000.0)];
    let actuals = vec![("matmul".to_string(), 4000.0)];
    let report = CostModel::calibrate(&predictions, &actuals).expect("single entry should succeed");
    assert!(
        report.correlation.is_nan(),
        "correlation should be NaN with 1 entry"
    );
    // But ratio should be defined: 5000/4000 = 1.25
    assert!(
        (report.mean_error_ratio - 1.25).abs() < 1e-10,
        "mean_error_ratio should be 1.25"
    );
}

#[test]
fn test_calibrate_summary_no_entries_from_records() {
    // from_records produces a report with empty entries; summary should
    // still work and not contain "Entries:" line.
    let report = CalibrationReport::from_records(&[]);
    let summary = report.summary();
    assert!(summary.contains("Calibration Report"));
    assert!(!summary.contains("Entries:"));
}
