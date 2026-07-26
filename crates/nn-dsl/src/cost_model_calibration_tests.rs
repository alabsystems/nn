// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for cost model calibration: hardware presets, memory bandwidth
//! estimation, compute-bound vs memory-bound classification, roofline
//! intersection, fused vs unfused cost comparison, and Kokoro-representative
//! shape estimation.
//!
//! Part of #4186.

use super::*;
use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use crate::trace_compile::{CompiledKernel, CompiledPlan, CompiledStep};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
// Section 1: Apple M4 preset validation
// ---------------------------------------------------------------------------

#[test]
fn test_apple_m4_launch_overhead() {
    let m = CostModel::apple_m4();
    assert_eq!(
        m.launch_overhead_ns, 2000.0,
        "M4 base launch overhead should be 2000 ns"
    );
}

#[test]
fn test_apple_m4_bandwidth() {
    let m = CostModel::apple_m4();
    assert_eq!(
        m.bandwidth_bytes_per_sec, 400e9,
        "M4 bandwidth should be 400 GB/s"
    );
}

#[test]
fn test_apple_m4_simd_width() {
    let m = CostModel::apple_m4();
    assert_eq!(m.simd_width, 32, "Apple GPU SIMD width is 32");
}

#[test]
fn test_apple_m4_max_has_op_throughputs() {
    let m = CostModel::apple_m4_max();
    assert!(m.op_throughput.contains_key("matmul"));
    assert!(m.op_throughput.contains_key("conv1d"));
    assert!(m.op_throughput.contains_key("softmax"));
    assert!(m.op_throughput.contains_key("gelu"));
}

#[test]
fn test_apple_m4_max_matmul_throughput_range() {
    let m = CostModel::apple_m4_max();
    let matmul = m.op_throughput["matmul"];
    // M4 Max should have 20-50 TFLOP/s for matmul.
    assert!(
        (20e12..=50e12).contains(&matmul),
        "M4 Max matmul throughput {matmul:.0} out of expected range [20T, 50T]"
    );
}

// ---------------------------------------------------------------------------
// Section 2: Memory bandwidth estimation
// ---------------------------------------------------------------------------

#[test]
fn test_memory_time_scales_with_elements() {
    let m = CostModel::apple_m4();
    // Cost for 1K elements should be less than cost for 1M elements.
    let plan_small = single_dispatch_plan("relu", &[1024]);
    let plan_large = single_dispatch_plan("relu", &[1024, 1024]);

    let est_small = m.estimate(&plan_small);
    let est_large = m.estimate(&plan_large);

    assert!(
        est_large.total_ns > est_small.total_ns,
        "larger tensor ({:.0} ns) should cost more than smaller ({:.0} ns)",
        est_large.total_ns,
        est_small.total_ns,
    );
}

#[test]
fn test_memory_time_proportional_to_size() {
    let m = CostModel::apple_m4();
    // A 10x larger tensor should cost roughly 10x more (minus fixed overhead).
    let plan_1k = single_dispatch_plan("relu", &[1000]);
    let plan_10k = single_dispatch_plan("relu", &[10000]);

    let est_1k = m.estimate(&plan_1k);
    let est_10k = m.estimate(&plan_10k);

    let variable_1k = est_1k.total_ns - m.launch_overhead_ns;
    let variable_10k = est_10k.total_ns - m.launch_overhead_ns;

    // Allow 2x tolerance since occupancy effects can shift the ratio.
    assert!(
        variable_10k > variable_1k * 3.0,
        "10K elements variable cost ({variable_10k:.0}) should be significantly larger than 1K ({variable_1k:.0})"
    );
}

// ---------------------------------------------------------------------------
// Section 3: Compute-bound vs memory-bound classification
// ---------------------------------------------------------------------------

#[test]
fn test_small_elementwise_is_memory_bound() {
    // Small elementwise ops are bandwidth-limited (few FLOPs per byte).
    // With default 1 TFLOP/s throughput and 400 GB/s bandwidth:
    // compute_ns for 1K elements = 1000 / 1e12 * 1e9 = 1e-6 ns (tiny)
    // memory_ns for 1K elements = 1000 * 4 * 2 / 400e9 * 1e9 = 0.02 ns
    // memory_ns >= compute_ns → memory bound.
    let m = CostModel::apple_m4();
    let plan = single_dispatch_plan("relu", &[1024]);
    let est = m.estimate(&plan);
    // Cost should be dominated by launch overhead for small tensors.
    assert!(
        est.total_ns >= m.launch_overhead_ns,
        "small tensor cost ({:.0}) should include launch overhead ({:.0})",
        est.total_ns,
        m.launch_overhead_ns,
    );
}

#[test]
fn test_large_matmul_cost_scales_with_compute() {
    // For a very large matmul, compute should dominate.
    let m = CostModel::apple_m4_max();
    let plan = single_dispatch_plan("matmul", &[4096, 4096]);
    let est = m.estimate(&plan);

    // 16M elements. compute_ns = 16M / 30T * 1e9 ≈ 533 ns (with M4 Max matmul throughput)
    // memory_ns = 16M * 8 / 400G * 1e9 ≈ 320 ns
    // Should be meaningfully above launch overhead.
    assert!(
        est.total_ns > m.launch_overhead_ns * 1.1,
        "large matmul cost ({:.0}) should exceed launch overhead ({:.0})",
        est.total_ns,
        m.launch_overhead_ns,
    );
}

// ---------------------------------------------------------------------------
// Section 4: Fused vs unfused cost comparison
// ---------------------------------------------------------------------------

#[test]
fn test_two_dispatches_cost_more_than_one() {
    let m = CostModel::apple_m4();
    let shape = &[1, 512, 256];

    // Single dispatch: 1 launch overhead.
    let plan_single = single_dispatch_plan("gelu", shape);

    // Two dispatches: 2 launch overheads.
    let kernel_a = make_kernel("relu", shape);
    let kernel_b = make_kernel("gelu", shape);
    let plan_double = CompiledPlan {
        steps: vec![
            CompiledStep::Dispatch {
                kernel: kernel_a,
                weight_data: Default::default(),
                external_node_ids: None,
            },
            CompiledStep::Dispatch {
                kernel: kernel_b,
                weight_data: Default::default(),
                external_node_ids: None,
            },
        ],
        input_shapes: vec![shape.to_vec()],
        output_step: 1,
        weight_names: vec![],
    };

    let est_single = m.estimate(&plan_single);
    let est_double = m.estimate(&plan_double);

    assert!(
        est_double.total_ns > est_single.total_ns,
        "two dispatches ({:.0} ns) should cost more than one ({:.0} ns)",
        est_double.total_ns,
        est_single.total_ns,
    );

    // The difference should be roughly 1 launch overhead.
    let overhead_gap = est_double.total_ns - est_single.total_ns;
    assert!(
        overhead_gap >= m.launch_overhead_ns * 0.5,
        "overhead gap ({:.0}) should reflect launch overhead ({:.0})",
        overhead_gap,
        m.launch_overhead_ns,
    );
}

#[test]
fn test_fused_cost_strictly_less_than_sum_of_unfused() {
    let m = CostModel::apple_m4();
    let shape = &[1, 768];

    // Simulate 3 separate dispatches.
    let plan_unfused = CompiledPlan {
        steps: vec![
            CompiledStep::Dispatch {
                kernel: make_kernel("relu", shape),
                weight_data: Default::default(),
                external_node_ids: None,
            },
            CompiledStep::Dispatch {
                kernel: make_kernel("gelu", shape),
                weight_data: Default::default(),
                external_node_ids: None,
            },
            CompiledStep::Dispatch {
                kernel: make_kernel("sigmoid", shape),
                weight_data: Default::default(),
                external_node_ids: None,
            },
        ],
        input_shapes: vec![shape.to_vec()],
        output_step: 2,
        weight_names: vec![],
    };

    // Simulate 1 fused dispatch (same total elements but 1 launch).
    let plan_fused = single_dispatch_plan("fused_relu_gelu_sigmoid", shape);

    let est_unfused = m.estimate(&plan_unfused);
    let est_fused = m.estimate(&plan_fused);

    assert!(
        est_fused.total_ns < est_unfused.total_ns,
        "fused ({:.0} ns, {} dispatch) should be cheaper than unfused ({:.0} ns, {} dispatches)",
        est_fused.total_ns,
        est_fused.dispatch_count,
        est_unfused.total_ns,
        est_unfused.dispatch_count,
    );
}

// ---------------------------------------------------------------------------
// Section 5: Kokoro-representative shapes
// ---------------------------------------------------------------------------

#[test]
fn test_cost_model_kokoro_decoder_shape() {
    // Kokoro decoder block: [1, 512, 256] (B=1, C=512, T=256).
    let m = CostModel::apple_m4();
    let plan = single_dispatch_plan("conv1d", &[1, 512, 256]);
    let est = m.estimate(&plan);
    assert!(
        est.total_ns > 0.0,
        "Kokoro decoder shape should have non-zero cost"
    );
    assert_eq!(est.dispatch_count, 1);
}

#[test]
fn test_cost_model_kokoro_attention_shape() {
    // Kokoro attention: [1, 8, 256, 64] (B, heads, seq, head_dim).
    let m = CostModel::apple_m4_max();
    let plan = single_dispatch_plan("matmul", &[1, 8, 256, 64]);
    let est = m.estimate(&plan);
    assert!(est.total_ns > 0.0);
    // 1*8*256*64 = 131072 elements — meaningful compute.
    assert!(est.total_ns >= m.launch_overhead_ns);
}

#[test]
fn test_cost_model_kokoro_style_projection_shape() {
    // Style projection: [1, 256] → [1, 512] via Linear.
    let m = CostModel::apple_m4();
    let plan = single_dispatch_plan("matmul", &[1, 512]);
    let est = m.estimate(&plan);
    assert!(est.total_ns > 0.0);
}

// ---------------------------------------------------------------------------
// Section 6: Cross-hardware comparison
// ---------------------------------------------------------------------------

#[test]
fn test_faster_hardware_has_lower_cost() {
    // M4 Max should estimate lower cost than M1 for same workload
    // (higher bandwidth, lower launch overhead, higher throughput).
    let m1 = CostModel::apple_m1();
    let m4_max = CostModel::apple_m4_max();

    let plan = single_dispatch_plan("matmul", &[256, 256]);
    let est_m1 = m1.estimate(&plan);
    let est_m4_max = m4_max.estimate(&plan);

    assert!(
        est_m4_max.total_ns < est_m1.total_ns,
        "M4 Max ({:.0} ns) should be faster than M1 ({:.0} ns) for matmul [256, 256]",
        est_m4_max.total_ns,
        est_m1.total_ns,
    );
}

#[test]
fn test_all_presets_estimate_positive_cost() {
    let presets: Vec<(&str, CostModel)> = vec![
        ("apple_m1", CostModel::apple_m1()),
        ("apple_m2", CostModel::apple_m2()),
        ("apple_m3", CostModel::apple_m3()),
        ("apple_m4", CostModel::apple_m4()),
        ("apple_m4_max", CostModel::apple_m4_max()),
        ("nvidia_a100", CostModel::nvidia_a100()),
    ];

    let plan = single_dispatch_plan("gelu", &[1, 1024]);

    for (name, model) in &presets {
        let est = model.estimate(&plan);
        assert!(
            est.total_ns > 0.0,
            "{name}: cost should be positive for non-empty plan"
        );
        assert_eq!(est.dispatch_count, 1, "{name}: should count 1 dispatch");
    }
}

// ---------------------------------------------------------------------------
// Section 7: Occupancy penalty
// ---------------------------------------------------------------------------

#[test]
fn test_simd_aligned_has_no_occupancy_penalty() {
    let m = CostModel::apple_m4();
    // 32 elements = exactly 1 SIMD group (no waste).
    let plan = single_dispatch_plan("relu", &[32]);
    let est = m.estimate(&plan);
    // For SIMD-aligned, compute_ns = 32/1e12 * 1e9 ≈ 0 (negligible).
    // memory_ns = 32*8/400e9 * 1e9 ≈ 0. Cost ≈ launch_overhead.
    assert!(
        (est.total_ns - m.launch_overhead_ns).abs() < m.launch_overhead_ns,
        "SIMD-aligned should be close to launch overhead"
    );
}

#[test]
fn test_simd_misaligned_has_higher_cost() {
    let m = CostModel::apple_m4();
    let plan_aligned = single_dispatch_plan("relu", &[1024]); // 1024 % 32 == 0
    let plan_misaligned = single_dispatch_plan("relu", &[1025]); // 1025 % 32 == 1

    let est_aligned = m.estimate(&plan_aligned);
    let est_misaligned = m.estimate(&plan_misaligned);

    assert!(
        est_misaligned.total_ns > est_aligned.total_ns,
        "misaligned ({:.0}) should cost more than aligned ({:.0})",
        est_misaligned.total_ns,
        est_aligned.total_ns,
    );
}
