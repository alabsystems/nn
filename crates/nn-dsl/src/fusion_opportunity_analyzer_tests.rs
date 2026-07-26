// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the fusion opportunity analyzer.

use std::collections::HashMap;

use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use crate::trace_compile::{CompiledKernel, CompiledPlan, CompiledStep, NativeOpKind};

use super::{
    analyze_fusion_opportunities, classify_kernel_name, is_fusible_pair, OpCategory,
    DISPATCH_OVERHEAD_NS,
};

/// Build a minimal `CompiledStep::Dispatch` with a given kernel name.
fn make_dispatch(name: &str) -> CompiledStep {
    let node_id = TensorNodeId::new(0);
    let input_node = TensorNode::new(
        node_id,
        TensorOpKind::Input {
            name: "input_0".into(),
            shape: vec![1, 4],
        },
        vec![1, 4],
    );
    let def = TensorKernelDef::new(name, vec![input_node], node_id);
    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

/// Build a minimal `CompiledStep::NativeOp` with a given variant.
fn make_native_op(op: NativeOpKind) -> CompiledStep {
    CompiledStep::NativeOp {
        op,
        weight_data: HashMap::new(),
    }
}

// -- classify_kernel_name tests -----------------------------------------------

#[test]
fn test_classify_kernel_name_norm_ops() {
    assert_eq!(classify_kernel_name("layer_norm"), OpCategory::Norm);
    assert_eq!(classify_kernel_name("instance_norm"), OpCategory::Norm);
    assert_eq!(classify_kernel_name("rms_norm"), OpCategory::Norm);
    assert_eq!(classify_kernel_name("group_norm"), OpCategory::Norm);
}

#[test]
fn test_classify_kernel_name_activation_ops() {
    assert_eq!(classify_kernel_name("relu"), OpCategory::Activation);
    assert_eq!(classify_kernel_name("gelu"), OpCategory::Activation);
    assert_eq!(classify_kernel_name("silu"), OpCategory::Activation);
    assert_eq!(classify_kernel_name("snake"), OpCategory::Activation);
    assert_eq!(classify_kernel_name("leaky_relu"), OpCategory::Activation);
    assert_eq!(classify_kernel_name("tanh"), OpCategory::Activation);
    assert_eq!(classify_kernel_name("sigmoid"), OpCategory::Activation);
}

#[test]
fn test_classify_kernel_name_conv_ops() {
    assert_eq!(classify_kernel_name("conv1d"), OpCategory::Conv);
    assert_eq!(classify_kernel_name("conv2d"), OpCategory::Conv);
    assert_eq!(classify_kernel_name("conv_transpose_1d"), OpCategory::Conv);
}

#[test]
fn test_classify_kernel_name_linear_ops() {
    assert_eq!(classify_kernel_name("linear"), OpCategory::Linear);
    assert_eq!(classify_kernel_name("matmul"), OpCategory::Linear);
}

#[test]
fn test_classify_kernel_name_elementwise_ops() {
    assert_eq!(classify_kernel_name("add"), OpCategory::Elementwise);
    assert_eq!(classify_kernel_name("mul"), OpCategory::Elementwise);
    assert_eq!(classify_kernel_name("exp"), OpCategory::Elementwise);
    assert_eq!(classify_kernel_name("neg"), OpCategory::Elementwise);
}

#[test]
fn test_classify_kernel_name_reduction_ops() {
    assert_eq!(classify_kernel_name("softmax"), OpCategory::Reduction);
    assert_eq!(classify_kernel_name("log_softmax"), OpCategory::Reduction);
}

#[test]
fn test_classify_kernel_name_unknown() {
    assert_eq!(classify_kernel_name("weird_custom_op"), OpCategory::Other);
}

#[test]
fn test_classify_kernel_name_fused_prefix_stripped() {
    // "fused_layer_norm_..." should still classify as Norm.
    assert_eq!(
        classify_kernel_name("fused_layer_norm_relu_x2"),
        OpCategory::Norm
    );
}

// -- is_fusible_pair tests ----------------------------------------------------

#[test]
fn test_fusible_pair_norm_activation() {
    assert!(is_fusible_pair(OpCategory::Norm, OpCategory::Activation));
}

#[test]
fn test_fusible_pair_linear_activation() {
    assert!(is_fusible_pair(OpCategory::Linear, OpCategory::Activation));
}

#[test]
fn test_fusible_pair_activation_conv() {
    assert!(is_fusible_pair(OpCategory::Activation, OpCategory::Conv));
}

#[test]
fn test_non_fusible_pair() {
    assert!(!is_fusible_pair(OpCategory::Other, OpCategory::Norm));
    assert!(!is_fusible_pair(
        OpCategory::NativeOp,
        OpCategory::Elementwise
    ));
}

// -- analyze_fusion_opportunities tests ---------------------------------------

#[test]
fn test_empty_plan_no_opportunities() {
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let result = analyze_fusion_opportunities(&plan);
    assert!(result.is_empty());
}

#[test]
fn test_single_step_no_opportunities() {
    let plan = CompiledPlan {
        steps: vec![make_dispatch("relu")],
        input_shapes: vec![vec![1, 4]],
        output_step: 0,
        weight_names: vec![],
    };
    let result = analyze_fusion_opportunities(&plan);
    assert!(result.is_empty());
}

#[test]
fn test_no_fusible_consecutive_ops() {
    // softmax → linear: Reduction+Linear is NOT in FUSIBLE_PAIRS.
    let plan = CompiledPlan {
        steps: vec![make_dispatch("softmax"), make_dispatch("linear")],
        input_shapes: vec![],
        output_step: 1,
        weight_names: vec![],
    };
    let result = analyze_fusion_opportunities(&plan);
    assert!(result.is_empty());
}

#[test]
fn test_detects_norm_activation_pair() {
    // instance_norm → snake: Norm+Activation is fusible.
    let plan = CompiledPlan {
        steps: vec![make_dispatch("instance_norm"), make_dispatch("snake")],
        input_shapes: vec![],
        output_step: 1,
        weight_names: vec![],
    };
    let result = analyze_fusion_opportunities(&plan);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].sequence, vec!["instance_norm", "snake"]);
    assert_eq!(result[0].count, 1);
    assert!((result[0].total_cost_ns - DISPATCH_OVERHEAD_NS).abs() < 1.0);
}

#[test]
fn test_detects_linear_activation_pair() {
    // linear → gelu: Linear+Activation is fusible.
    let plan = CompiledPlan {
        steps: vec![make_dispatch("linear"), make_dispatch("gelu")],
        input_shapes: vec![],
        output_step: 1,
        weight_names: vec![],
    };
    let result = analyze_fusion_opportunities(&plan);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].sequence, vec!["linear", "gelu"]);
    assert_eq!(result[0].count, 1);
}

#[test]
fn test_repeated_pair_counted() {
    // Two occurrences of linear → relu.
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch("linear"),
            make_dispatch("relu"),
            CompiledStep::InputForward, // break continuity
            make_dispatch("linear"),
            make_dispatch("relu"),
        ],
        input_shapes: vec![],
        output_step: 4,
        weight_names: vec![],
    };
    let result = analyze_fusion_opportunities(&plan);
    let linear_relu: Vec<_> = result
        .iter()
        .filter(|o| o.sequence == vec!["linear", "relu"])
        .collect();
    assert_eq!(linear_relu.len(), 1);
    assert_eq!(linear_relu[0].count, 2);
    assert!((linear_relu[0].total_cost_ns - 2.0 * DISPATCH_OVERHEAD_NS).abs() < 1.0);
}

#[test]
fn test_triple_detected() {
    // instance_norm → snake → conv1d:
    // Norm+Activation (fusible) AND Activation+Conv (fusible) → triple.
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch("instance_norm"),
            make_dispatch("snake"),
            make_dispatch("conv1d"),
        ],
        input_shapes: vec![],
        output_step: 2,
        weight_names: vec![],
    };
    let result = analyze_fusion_opportunities(&plan);
    // Should find: pair(instance_norm, snake), pair(snake, conv1d),
    // AND triple(instance_norm, snake, conv1d).
    let triples: Vec<_> = result.iter().filter(|o| o.sequence.len() == 3).collect();
    assert_eq!(triples.len(), 1);
    assert_eq!(
        triples[0].sequence,
        vec!["instance_norm", "snake", "conv1d"]
    );
    assert_eq!(triples[0].count, 1);
    // Triple saves 2 dispatches per occurrence.
    assert!((triples[0].total_cost_ns - 2.0 * DISPATCH_OVERHEAD_NS).abs() < 1.0);
}

#[test]
fn test_sorted_by_cost_descending() {
    // Create a plan where patterns appear with different frequencies.
    // Use non-adjacent groups so filtering doesn't create new adjacencies.
    // norm+activation appears 3x, linear+activation appears 1x.
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch("layer_norm"),
            make_dispatch("relu"),
            make_dispatch("softmax"), // breaks fusibility chain
            make_dispatch("layer_norm"),
            make_dispatch("relu"),
            make_dispatch("softmax"),
            make_dispatch("layer_norm"),
            make_dispatch("relu"),
            make_dispatch("softmax"),
            make_dispatch("linear"),
            make_dispatch("gelu"),
        ],
        input_shapes: vec![],
        output_step: 10,
        weight_names: vec![],
    };
    let result = analyze_fusion_opportunities(&plan);
    assert!(result.len() >= 2);
    // First result should have the highest cost.
    // layer_norm+relu appears 3 times = 3*2000ns = 6000ns.
    // linear+gelu appears 1 time = 1*2000ns = 2000ns.
    assert_eq!(result[0].sequence, vec!["layer_norm", "relu"]);
    assert_eq!(result[0].count, 3);
    // Verify descending sort.
    for window in result.windows(2) {
        assert!(window[0].total_cost_ns >= window[1].total_cost_ns);
    }
}

#[test]
fn test_native_ops_skipped() {
    // NativeOp steps are already fused and should not appear in opportunities.
    let plan = CompiledPlan {
        steps: vec![
            make_native_op(NativeOpKind::InstanceNorm {
                eps: 1e-5,
                input_shape: vec![1, 64, 128],
            }),
            make_dispatch("snake"),
        ],
        input_shapes: vec![],
        output_step: 1,
        weight_names: vec![],
    };
    let result = analyze_fusion_opportunities(&plan);
    // InstanceNorm NativeOp is skipped (already fused), so no pair detected.
    assert!(result.is_empty());
}

#[test]
fn test_passthrough_steps_skipped() {
    // Passthrough steps should not appear in opportunities.
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch("relu"),
            CompiledStep::Passthrough {
                op_name: "reshape".into(),
                output_shape: vec![1, 4],
            },
            make_dispatch("exp"),
        ],
        input_shapes: vec![],
        output_step: 2,
        weight_names: vec![],
    };
    let result = analyze_fusion_opportunities(&plan);
    // relu and exp are not adjacent in the dispatch-only sequence, and
    // Elementwise+Elementwise IS fusible, so they should form a pair.
    let relu_exp: Vec<_> = result
        .iter()
        .filter(|o| o.sequence == vec!["relu", "exp"])
        .collect();
    assert_eq!(relu_exp.len(), 1);
}

#[test]
fn test_adain_snake_unfused_detected() {
    // Simulates the unfused pattern that FusedAdainSnake replaces:
    // instance_norm → mul → add → snake.
    // This should detect multiple fusible pairs and a triple.
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch("instance_norm"),
            make_dispatch("mul"),
            make_dispatch("add"),
            make_dispatch("snake"),
        ],
        input_shapes: vec![],
        output_step: 3,
        weight_names: vec![],
    };
    let result = analyze_fusion_opportunities(&plan);
    // Should detect at least:
    // - (instance_norm, mul) = Norm+Elementwise
    // - (mul, add) = Elementwise+Elementwise
    // - (add, snake) = Elementwise+Activation
    assert!(!result.is_empty());

    let has_norm_elem = result
        .iter()
        .any(|o| o.sequence == vec!["instance_norm", "mul"]);
    let has_elem_elem = result.iter().any(|o| o.sequence == vec!["mul", "add"]);
    let has_elem_activ = result.iter().any(|o| o.sequence == vec!["add", "snake"]);
    assert!(has_norm_elem, "should detect instance_norm+mul pair");
    assert!(has_elem_elem, "should detect mul+add pair");
    assert!(has_elem_activ, "should detect add+snake pair");
}

// -- Kokoro-representative fusion opportunity analysis -----------------------

/// Build a dispatch plan that mimics Kokoro's real dispatch patterns across
/// all segments, using actual kernel names from the compiled pipeline.
///
/// Kokoro segments (from compiled_kokoro_segments.rs):
///   0. PlBert: 12 transformer layers (LayerNorm → Linear → attention → Add → LayerNorm → Linear → activation → Linear → Add)
///   1. TextEncoder: embedding + duration predictor
///   2. ProsodyPredictor: AdaLayerNorm blocks
///   3. F0EnergyPredictor: AdainLeakyRelu + Conv1d blocks, upsample+conv pairs
///   4. Generator: 12 ResBlocks (InstanceNorm → mul → add → Snake → Conv1d), upsample+conv
///   5. iSTFT: elementwise signal processing
///
/// The plan represents unfused dispatch steps that would appear BEFORE
/// peephole passes. This reveals which fusion opportunities have the
/// highest dispatch-reduction potential.
///
/// Part of #4264 (RTF optimization).
fn build_kokoro_representative_plan() -> CompiledPlan {
    let mut steps = Vec::new();

    // -- Segment 0: PlBert (12 transformer layers) --
    // Each transformer layer: LayerNorm → Q/K/V Linear → matmul → softmax →
    //   matmul → Linear → Add → LayerNorm → Linear → gelu → Linear → Add
    for _ in 0..12 {
        // Self-attention sub-layer
        steps.push(make_dispatch("layer_norm"));
        steps.push(make_dispatch("linear")); // Q projection
        steps.push(make_dispatch("linear")); // K projection
        steps.push(make_dispatch("linear")); // V projection
        steps.push(make_dispatch("matmul")); // Q @ K^T
        steps.push(make_dispatch("softmax"));
        steps.push(make_dispatch("matmul")); // attn @ V
        steps.push(make_dispatch("linear")); // output projection
        steps.push(make_dispatch("add")); // residual
                                          // FFN sub-layer
        steps.push(make_dispatch("layer_norm"));
        steps.push(make_dispatch("linear")); // FFN up
        steps.push(make_dispatch("gelu"));
        steps.push(make_dispatch("linear")); // FFN down
        steps.push(make_dispatch("add")); // residual
    }

    // -- Segment 3: F0 energy predictor (6 AdainResBlk1d blocks) --
    // Each block: InstanceNorm → mul(gamma) → add(beta) → leaky_relu → conv1d ×2 + residual
    for _ in 0..6 {
        // Phase 1
        steps.push(make_dispatch("instance_norm"));
        steps.push(make_dispatch("mul"));
        steps.push(make_dispatch("add"));
        steps.push(make_dispatch("leaky_relu"));
        steps.push(make_dispatch("conv1d"));
        // Phase 2
        steps.push(make_dispatch("instance_norm"));
        steps.push(make_dispatch("mul"));
        steps.push(make_dispatch("add"));
        steps.push(make_dispatch("leaky_relu"));
        steps.push(make_dispatch("conv1d"));
        steps.push(make_dispatch("add")); // residual
    }
    // F0 upsample chain (6 pairs)
    for _ in 0..6 {
        steps.push(make_dispatch("conv_transpose_1d")); // upsample
        steps.push(make_dispatch("conv1d"));
    }

    // -- Segment 4: Generator (12 ResBlocks with Snake activation) --
    // Each ResBlock: InstanceNorm → mul(gamma) → add(beta) → snake → conv1d ×2 + residual
    for _ in 0..12 {
        // Phase 1
        steps.push(make_dispatch("instance_norm"));
        steps.push(make_dispatch("mul"));
        steps.push(make_dispatch("add"));
        steps.push(make_dispatch("snake"));
        steps.push(make_dispatch("conv1d"));
        // Phase 2
        steps.push(make_dispatch("instance_norm"));
        steps.push(make_dispatch("mul"));
        steps.push(make_dispatch("add"));
        steps.push(make_dispatch("snake"));
        steps.push(make_dispatch("conv1d"));
        steps.push(make_dispatch("add")); // residual
    }
    // Generator upsample chain (4 stages)
    for _ in 0..4 {
        steps.push(make_dispatch("conv_transpose_1d"));
        steps.push(make_dispatch("leaky_relu"));
    }

    // -- Segment 5: iSTFT (signal processing) --
    steps.push(make_dispatch("exp"));
    steps.push(make_dispatch("mul")); // magnitude * cos/sin
    steps.push(make_dispatch("mul"));
    steps.push(make_dispatch("add"));

    let n = steps.len();
    CompiledPlan {
        steps,
        input_shapes: vec![vec![1, 512]],
        output_step: n - 1,
        weight_names: vec![],
    }
}

#[test]
fn test_kokoro_representative_fusion_opportunities() {
    let plan = build_kokoro_representative_plan();
    let total_dispatch_steps = plan
        .steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count();

    let opportunities = analyze_fusion_opportunities(&plan);

    // The plan should have substantial dispatch steps and fusion opportunities.
    assert!(
        total_dispatch_steps > 300,
        "Expected >300 dispatch steps in Kokoro-representative plan, got {total_dispatch_steps}"
    );
    assert!(
        !opportunities.is_empty(),
        "Should detect fusion opportunities in Kokoro dispatch plan"
    );

    // Print a summary report (visible with --nocapture).
    eprintln!("\n=== Kokoro Fusion Opportunity Report ===");
    eprintln!("Total unfused dispatch steps: {total_dispatch_steps}");
    eprintln!("Fusion opportunities found: {}", opportunities.len());

    let total_saveable_dispatches: usize = opportunities
        .iter()
        .map(|o| o.count * (o.sequence.len() - 1))
        .sum();
    eprintln!(
        "Total dispatch savings potential: {total_saveable_dispatches} dispatches ({:.1}% reduction)",
        100.0 * total_saveable_dispatches as f64 / total_dispatch_steps as f64
    );

    eprintln!("\nTop 15 fusion opportunities (by estimated savings):");
    eprintln!(
        "{:<50} {:>6} {:>8} {:>12}",
        "Pattern", "Count", "Saves", "Cost (us)"
    );
    eprintln!("{:-<50} {:-<6} {:-<8} {:-<12}", "", "", "", "");
    for opp in opportunities.iter().take(15) {
        let pattern = opp.sequence.join(" -> ");
        let saves = opp.count * (opp.sequence.len() - 1);
        eprintln!(
            "{:<50} {:>6} {:>8} {:>12.1}",
            pattern,
            opp.count,
            saves,
            opp.total_cost_ns / 1000.0,
        );
    }

    // -- Structural assertions --
    // The most frequent patterns in Kokoro should be instance_norm→mul
    // (from unfused AdaIN in Generator and F0 blocks).
    let instance_norm_mul: Vec<_> = opportunities
        .iter()
        .filter(|o| o.sequence == vec!["instance_norm", "mul"])
        .collect();
    assert!(
        !instance_norm_mul.is_empty(),
        "Should detect instance_norm -> mul fusion opportunity (AdaIN pattern)"
    );
    // 12 Generator ResBlocks × 2 phases + 6 F0 blocks × 2 phases = 36 occurrences.
    assert_eq!(
        instance_norm_mul[0].count, 36,
        "instance_norm -> mul should appear 36 times (12 generator + 6 f0 blocks, 2 phases each)"
    );

    // Layer_norm → linear should appear from PlBert (12 layers × 2 sub-layers = 24).
    let layernorm_linear: Vec<_> = opportunities
        .iter()
        .filter(|o| o.sequence == vec!["layer_norm", "linear"])
        .collect();
    assert!(
        !layernorm_linear.is_empty(),
        "Should detect layer_norm -> linear fusion opportunity (PlBert pattern)"
    );
    assert_eq!(
        layernorm_linear[0].count, 24,
        "layer_norm -> linear should appear 24 times (12 PlBert layers × 2 sub-layers)"
    );

    // Mul → add (style affine) should appear 36 times.
    let mul_add: Vec<_> = opportunities
        .iter()
        .filter(|o| o.sequence == vec!["mul", "add"])
        .collect();
    assert!(
        !mul_add.is_empty(),
        "Should detect mul -> add fusion opportunity"
    );

    // Triple: instance_norm → mul → add should be detected.
    let norm_mul_add: Vec<_> = opportunities
        .iter()
        .filter(|o| o.sequence == vec!["instance_norm", "mul", "add"])
        .collect();
    assert!(
        !norm_mul_add.is_empty(),
        "Should detect instance_norm -> mul -> add triple (AdaIN affine pattern)"
    );

    // Sorted descending by cost.
    for window in opportunities.windows(2) {
        assert!(
            window[0].total_cost_ns >= window[1].total_cost_ns,
            "Results should be sorted descending by total_cost_ns"
        );
    }
}
