// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for individual peephole optimization passes and PeepholeConfig infrastructure.
//!
//! Validates:
//! - Each boolean field in PeepholeConfig enables/disables its corresponding pass
//! - PeepholeConfig::default() has all 16 passes enabled
//! - Disabling a pass leaves the graph unchanged for patterns that pass would match
//! - Enabling a pass fuses the expected patterns
//! - Bitmask encoding/decoding roundtrip correctness
//! - PEEPHOLE_FIELD_COUNT matches the actual struct field count

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use crate::optimize_plan::{
    config_from_bitmask, enumerate_peephole_configs, is_default_config, PEEPHOLE_FIELD_COUNT,
    PEEPHOLE_FIELD_NAMES,
};
use crate::trace_compile::{
    compile_trace_to_plan_configured, count_dispatches, CompiledKernel, CompiledStep,
    GemmActivation, NativeOpKind, PeepholeConfig,
};

// ---------------------------------------------------------------------------
// Helper constructors (shared with trace_compile_peephole_tests.rs)
// ---------------------------------------------------------------------------

fn test_node(id: u64, name: &str, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        TraceOp::Relu,
        inputs,
        shape,
        DType::F32,
    )
}

fn test_input_node(id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape,
        DType::F32,
    )
}

fn make_adain_leaky_relu(input_shape: &[usize]) -> CompiledStep {
    CompiledStep::NativeOp {
        op: NativeOpKind::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.2,
            input_shape: input_shape.to_vec(),
            external_node_ids: None,
        },
        weight_data: HashMap::new(),
    }
}

fn make_conv1d_dispatch(
    input_shape: &[usize],
    output_shape: &[usize],
    weight_shape: &[usize],
    padding: usize,
    dilation: usize,
) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let bias_shape = [weight_shape[0]];
    let mut b = TensorBlockBuilder::new("conv1d");
    let input = b.add_input("input_0", input_shape);
    let w = b.add_input("weight", weight_shape);
    let bi = b.add_input("bias", &bias_shape);
    let output = b.add_conv1d_full(input, w, Some(bi), 1, padding, dilation, 1, output_shape);
    let def = b.build(output).expect("valid conv1d IR");

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(
            vec![0.0f32; weight_shape.iter().product()],
            weight_shape.to_vec(),
        )
        .expect("valid weight"),
    );
    weight_data.insert(
        "bias".to_string(),
        WeightRef::new(vec![0.0f32; bias_shape[0]], bias_shape.to_vec()).expect("valid bias"),
    );

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: None,
    }
}

fn make_linear_dispatch(
    input_shape: &[usize],
    out_features: usize,
    has_bias: bool,
) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let in_features = *input_shape.last().unwrap();
    let mut output_shape = input_shape.to_vec();
    *output_shape.last_mut().unwrap() = out_features;
    let weight_shape = [out_features, in_features];

    let mut b = TensorBlockBuilder::new("linear");
    let input = b.add_input("input_0", input_shape);
    let w = b.add_input("weight", &weight_shape);
    let bi = if has_bias {
        Some(b.add_input("bias", &[out_features]))
    } else {
        None
    };
    let output = b.add_linear(input, w, bi, &output_shape);
    let def = b.build(output).expect("valid linear IR");

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(
            vec![0.0f32; out_features * in_features],
            weight_shape.to_vec(),
        )
        .expect("valid weight"),
    );
    if has_bias {
        weight_data.insert(
            "bias".to_string(),
            WeightRef::new(vec![0.0f32; out_features], vec![out_features]).expect("valid bias"),
        );
    }

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: None,
    }
}

fn make_activation_dispatch(name: &str, shape: &[usize]) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("input_0", shape);
    let output = match name {
        "relu" => b.add_relu(input, shape),
        "gelu" => b.add_gelu(input, shape),
        "sigmoid" => b.add_sigmoid(input, shape),
        _ => panic!("unsupported test activation: {name}"),
    };
    let def = b.build(output).expect("valid activation IR");

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

fn make_layernorm_native(input_shape: &[usize], hidden_dim: usize, eps: f32) -> CompiledStep {
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(vec![1.0f32; hidden_dim], vec![hidden_dim]).expect("valid weight"),
    );
    weight_data.insert(
        "bias".to_string(),
        WeightRef::new(vec![0.0f32; hidden_dim], vec![hidden_dim]).expect("valid bias"),
    );
    CompiledStep::NativeOp {
        op: NativeOpKind::LayerNorm {
            eps,
            input_shape: input_shape.to_vec(),
            hidden_dim,
        },
        weight_data,
    }
}

// ---------------------------------------------------------------------------
// PeepholeConfig::default() — all 16 passes enabled
// ---------------------------------------------------------------------------

#[test]
fn test_peephole_config_default_all_enabled() {
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
fn test_peephole_config_default_equals_all_enabled_bitmask() {
    let default = PeepholeConfig::default();
    let all_on_mask = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    let all_on = config_from_bitmask(all_on_mask);
    assert_eq!(
        default, all_on,
        "Default config must equal all-bits-set bitmask"
    );
}

// ---------------------------------------------------------------------------
// PEEPHOLE_FIELD_COUNT matches actual struct fields (17)
// ---------------------------------------------------------------------------

#[test]
fn test_field_count_is_20() {
    assert_eq!(PEEPHOLE_FIELD_COUNT, 28);
    assert_eq!(PEEPHOLE_FIELD_NAMES.len(), 28);
}

#[test]
fn test_enumeration_total_is_2_pow_17() {
    // Count lazily (O(1) memory) rather than materializing 2^28 configs.
    assert_eq!(enumerate_peephole_configs().count(), 268_435_456);
}

// ---------------------------------------------------------------------------
// Bitmask encoding/decoding roundtrip
// ---------------------------------------------------------------------------

/// Extract the 16 boolean fields from PeepholeConfig in bit order.
fn config_to_bits(cfg: &PeepholeConfig) -> [bool; 21] {
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
    ]
}

#[test]
fn test_bitmask_roundtrip_spot_checks() {
    // All-off
    let all_off = config_from_bitmask(0);
    let bits_off = config_to_bits(&all_off);
    assert!(bits_off.iter().all(|b| !b), "bitmask 0 = all false");

    // All-on
    let all_on_mask = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    let all_on = config_from_bitmask(all_on_mask);
    let bits_on = config_to_bits(&all_on);
    assert!(bits_on.iter().all(|b| *b), "all-bits-set = all true");

    // Single-bit: bit 9 = silu_mul
    let silu_only = config_from_bitmask(1 << 9);
    assert!(silu_only.silu_mul);
    assert!(!silu_only.norm_activ_conv1d);
    assert!(!silu_only.fused_resblock);
    assert!(!silu_only.fuse_instance_norm_mul_add);

    // Single-bit: bit 15 = fuse_instance_norm_mul_add
    let last_only = config_from_bitmask(1 << 15);
    assert!(last_only.fuse_instance_norm_mul_add);
    assert!(!last_only.norm_activ_conv1d);
    assert!(!last_only.silu_mul);
}

#[test]
fn test_bitmask_roundtrip_exhaustive_sample() {
    // Check every single-bit mask: only the corresponding field should be true.
    for bit in 0..PEEPHOLE_FIELD_COUNT {
        let mask = 1u32 << bit;
        let cfg = config_from_bitmask(mask);
        let bits = config_to_bits(&cfg);
        for (idx, &val) in bits.iter().enumerate() {
            if idx == bit as usize {
                assert!(val, "bit {bit} should be true for mask {mask:#x}");
            } else {
                assert!(
                    !val,
                    "bit {idx} should be false for mask {mask:#x} (only bit {bit} set)"
                );
            }
        }
    }
}

#[test]
fn test_bitmask_roundtrip_complementary_pairs() {
    // For each pair of complementary masks (mask, ~mask & all_on), their fields
    // should be exactly inverted.
    let all_on_mask = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    for mask in [0u32, 1, 0b1010_1010_1010_1010, 0xFF00, all_on_mask] {
        let cfg = config_from_bitmask(mask);
        let complement = config_from_bitmask((!mask) & all_on_mask);
        let bits_a = config_to_bits(&cfg);
        let bits_b = config_to_bits(&complement);
        for idx in 0..16 {
            assert_ne!(
                bits_a[idx], bits_b[idx],
                "bit {idx}: mask {mask:#x} and complement should be inverted"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// is_default_config: exactly one mask matches
// ---------------------------------------------------------------------------

#[test]
fn test_is_default_detects_default() {
    assert!(is_default_config(&PeepholeConfig::default()));
}

#[test]
fn test_is_default_rejects_each_single_field_toggle() {
    let all_on_mask = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    for bit in 0..PEEPHOLE_FIELD_COUNT {
        let mask = all_on_mask ^ (1u32 << bit);
        let cfg = config_from_bitmask(mask);
        assert!(
            !is_default_config(&cfg),
            "toggling off bit {bit} ({}) should make config non-default",
            PEEPHOLE_FIELD_NAMES[bit as usize]
        );
    }
}

// ---------------------------------------------------------------------------
// Disabling norm_activ_conv1d prevents NormActivConv1d fusion
// ---------------------------------------------------------------------------

#[test]
fn test_disabling_norm_activ_conv1d_prevents_fusion() {
    let input_shape = [1, 512, 100];
    let output_shape = [1, 512, 100];
    let weight_shape = [512, 512, 3];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        make_adain_leaky_relu(&input_shape),
        make_conv1d_dispatch(&input_shape, &output_shape, &weight_shape, 1, 1),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_input_node(1, vec![1, 512, 1]),
        test_input_node(2, vec![1, 512, 1]),
        test_node(3, "adain", vec![0, 1, 2], input_shape.to_vec()),
        test_node(4, "conv1d", vec![3], output_shape.to_vec()),
    ]);

    // Disable ONLY norm_activ_conv1d.
    let config = PeepholeConfig {
        norm_activ_conv1d: false,
        ..Default::default()
    };

    crate::trace_compile::peephole::apply_peephole_with_config(&mut steps, &graph, &config);

    // Step 3 should remain AdainLeakyRelu (not fused to NormActivConv1d).
    assert!(
        matches!(
            &steps[3],
            CompiledStep::NativeOp {
                op: NativeOpKind::AdainLeakyRelu { .. },
                ..
            }
        ),
        "AdainLeakyRelu should remain unfused when norm_activ_conv1d is disabled"
    );

    // Step 4 should remain a Dispatch (conv1d), not IdentityPassthrough.
    assert!(
        matches!(&steps[4], CompiledStep::Dispatch { .. }),
        "conv1d dispatch should remain when norm_activ_conv1d is disabled"
    );
}

#[test]
fn test_enabling_norm_activ_conv1d_fuses() {
    let input_shape = [1, 512, 100];
    let output_shape = [1, 512, 100];
    let weight_shape = [512, 512, 3];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        make_adain_leaky_relu(&input_shape),
        make_conv1d_dispatch(&input_shape, &output_shape, &weight_shape, 1, 1),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_input_node(1, vec![1, 512, 1]),
        test_input_node(2, vec![1, 512, 1]),
        test_node(3, "adain", vec![0, 1, 2], input_shape.to_vec()),
        test_node(4, "conv1d", vec![3], output_shape.to_vec()),
    ]);

    // Use default config (norm_activ_conv1d = true).
    crate::trace_compile::peephole::apply_peephole_with_config(
        &mut steps,
        &graph,
        &PeepholeConfig::default(),
    );

    // Step 3 should be NormActivConv1d.
    assert!(
        matches!(
            &steps[3],
            CompiledStep::NativeOp {
                op: NativeOpKind::NormActivConv1d { .. },
                ..
            }
        ),
        "AdainLeakyRelu + Conv1d should fuse to NormActivConv1d with default config"
    );

    assert!(
        matches!(&steps[4], CompiledStep::IdentityPassthrough),
        "conv1d position should become IdentityPassthrough after fusion"
    );
}

// ---------------------------------------------------------------------------
// Disabling linear_activation prevents Linear+Activation fusion
// ---------------------------------------------------------------------------

#[test]
fn test_disabling_linear_activation_prevents_fusion() {
    let input_shape = [4, 768];
    let out_features = 3072;
    let output_shape = [4, 3072];

    let mut steps = vec![
        CompiledStep::InputForward,
        make_linear_dispatch(&input_shape, out_features, true),
        make_activation_dispatch("relu", &output_shape),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_node(1, "linear", vec![0], output_shape.to_vec()),
        test_node(2, "relu", vec![1], output_shape.to_vec()),
    ]);

    let config = PeepholeConfig {
        linear_activation: false,
        ..Default::default()
    };

    crate::trace_compile::peephole::apply_peephole_with_config(&mut steps, &graph, &config);

    // Step 1 should remain a Dispatch (not fused to LinearActivation).
    assert!(
        matches!(&steps[1], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "linear"),
        "linear should remain unfused when linear_activation is disabled"
    );
}

#[test]
fn test_enabling_linear_activation_fuses() {
    let input_shape = [4, 768];
    let out_features = 3072;
    let output_shape = [4, 3072];

    let mut steps = vec![
        CompiledStep::InputForward,
        make_linear_dispatch(&input_shape, out_features, true),
        make_activation_dispatch("relu", &output_shape),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_node(1, "linear", vec![0], output_shape.to_vec()),
        test_node(2, "relu", vec![1], output_shape.to_vec()),
    ]);

    crate::trace_compile::peephole::apply_peephole_with_config(
        &mut steps,
        &graph,
        &PeepholeConfig::default(),
    );

    match &steps[1] {
        CompiledStep::NativeOp {
            op: NativeOpKind::LinearActivation { activation, .. },
            ..
        } => {
            assert_eq!(*activation, GemmActivation::Relu);
        }
        other => panic!(
            "expected LinearActivation at step 1, got: {:?}",
            std::mem::discriminant(other)
        ),
    }
}

// ---------------------------------------------------------------------------
// Disabling norm_linear prevents NormLinear / FusedLayerNormLinear fusion
// ---------------------------------------------------------------------------

#[test]
fn test_disabling_norm_linear_prevents_fusion() {
    let shape = [4, 768];
    let hidden_dim = 768;
    let out_features = 3072;

    let mut steps = vec![
        CompiledStep::InputForward,
        make_layernorm_native(&shape, hidden_dim, 1e-5),
        make_linear_dispatch(&shape, out_features, true),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.to_vec()),
        test_node(1, "layernorm", vec![0], shape.to_vec()),
        test_node(2, "linear", vec![1], vec![4, out_features]),
    ]);

    let config = PeepholeConfig {
        norm_linear: false,
        ..Default::default()
    };

    crate::trace_compile::peephole::apply_peephole_with_config(&mut steps, &graph, &config);

    // Step 1 should remain LayerNorm (not fused).
    assert!(
        matches!(
            &steps[1],
            CompiledStep::NativeOp {
                op: NativeOpKind::LayerNorm { .. },
                ..
            }
        ),
        "LayerNorm should remain unfused when norm_linear is disabled"
    );
}

// ---------------------------------------------------------------------------
// All-disabled config: no peephole fusion happens
// ---------------------------------------------------------------------------

#[test]
fn test_all_disabled_config_no_fusion() {
    let input_shape = [1, 512, 100];
    let output_shape = [1, 512, 100];
    let weight_shape = [512, 512, 3];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        make_adain_leaky_relu(&input_shape),
        make_conv1d_dispatch(&input_shape, &output_shape, &weight_shape, 1, 1),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_input_node(1, vec![1, 512, 1]),
        test_input_node(2, vec![1, 512, 1]),
        test_node(3, "adain", vec![0, 1, 2], input_shape.to_vec()),
        test_node(4, "conv1d", vec![3], output_shape.to_vec()),
    ]);

    // All passes disabled.
    let config = config_from_bitmask(0);

    crate::trace_compile::peephole::apply_peephole_with_config(&mut steps, &graph, &config);

    // Nothing should be fused.
    assert!(
        matches!(
            &steps[3],
            CompiledStep::NativeOp {
                op: NativeOpKind::AdainLeakyRelu { .. },
                ..
            }
        ),
        "with all passes disabled, AdainLeakyRelu should remain"
    );
    assert!(
        matches!(&steps[4], CompiledStep::Dispatch { .. }),
        "with all passes disabled, conv1d dispatch should remain"
    );
}

// ---------------------------------------------------------------------------
// compile_trace_to_plan_configured: all-disabled vs default on empty graph
// ---------------------------------------------------------------------------

#[test]
fn test_configured_compile_empty_graph_all_disabled() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let all_off = config_from_bitmask(0);
    let plan = compile_trace_to_plan_configured(&graph, &all_off)
        .expect("compile with all passes disabled on empty graph");
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_configured_compile_empty_graph_default() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let plan = compile_trace_to_plan_configured(&graph, &PeepholeConfig::default())
        .expect("compile with default config on empty graph");
    assert_eq!(count_dispatches(&plan), 0);
}

// ---------------------------------------------------------------------------
// PeepholeConfig PartialEq
// ---------------------------------------------------------------------------

#[test]
fn test_peephole_config_partial_eq() {
    let a = PeepholeConfig::default();
    let b = PeepholeConfig::default();
    assert_eq!(a, b);

    let c = PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };
    assert_ne!(a, c);
}

#[test]
fn test_peephole_config_clone() {
    let original = PeepholeConfig::default();
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

// ---------------------------------------------------------------------------
// PEEPHOLE_FIELD_NAMES corresponds to struct fields in bit order
// ---------------------------------------------------------------------------

#[test]
fn test_field_names_match_struct_fields() {
    assert_eq!(PEEPHOLE_FIELD_NAMES[0], "norm_activ_conv1d");
    assert_eq!(PEEPHOLE_FIELD_NAMES[1], "fused_resblock");
    assert_eq!(PEEPHOLE_FIELD_NAMES[2], "linear_activation");
    assert_eq!(PEEPHOLE_FIELD_NAMES[3], "add_layer_norm");
    assert_eq!(PEEPHOLE_FIELD_NAMES[4], "norm_linear");
    assert_eq!(PEEPHOLE_FIELD_NAMES[5], "attention_transpose");
    assert_eq!(PEEPHOLE_FIELD_NAMES[6], "flip_lstm");
    assert_eq!(PEEPHOLE_FIELD_NAMES[7], "batched_linear_projection");
    assert_eq!(PEEPHOLE_FIELD_NAMES[8], "channels_first_layer_norm");
    assert_eq!(PEEPHOLE_FIELD_NAMES[9], "silu_mul");
    assert_eq!(PEEPHOLE_FIELD_NAMES[10], "auto_fuse_elementwise");
    assert_eq!(PEEPHOLE_FIELD_NAMES[11], "bilstm_cat");
    assert_eq!(PEEPHOLE_FIELD_NAMES[12], "add_norm_linear");
    assert_eq!(PEEPHOLE_FIELD_NAMES[13], "fuse_adain_snake");
    assert_eq!(PEEPHOLE_FIELD_NAMES[14], "fuse_upsample_conv1d");
    assert_eq!(PEEPHOLE_FIELD_NAMES[15], "fuse_instance_norm_mul_add");
}

// ---------------------------------------------------------------------------
// PeepholeConfig explicit construction with all fields
// ---------------------------------------------------------------------------

#[test]
fn test_explicit_struct_construction_all_false() {
    let config = PeepholeConfig {
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
    assert!(!is_default_config(&config));
    let bits = config_to_bits(&config);
    assert!(bits.iter().all(|b| !b));
}

#[test]
fn test_explicit_struct_construction_all_true() {
    let config = PeepholeConfig {
        norm_activ_conv1d: true,
        fused_resblock: true,
        linear_activation: true,
        add_layer_norm: true,
        norm_linear: true,
        attention_transpose: true,
        flip_lstm: true,
        batched_linear_projection: true,
        channels_first_layer_norm: true,
        silu_mul: true,
        auto_fuse_elementwise: true,
        bilstm_cat: true,
        add_norm_linear: true,
        fuse_adain_snake: true,
        fuse_upsample_conv1d: true,
        fuse_instance_norm_mul_add: true,
        fuse_conv1d_activation: true,
        fuse_snake_instance_norm: true,
        fuse_conv1d_snake_norm: true,
        fuse_conv1d_snake_norm_resblock: true,
        fuse_add_instance_norm_conv1x1: true,
        fuse_conv_transpose1d_activation: true,
        norm_activ_conv_transpose1d: true,
        fuse_instance_norm_conv1d: true,
        fuse_conv1d_instance_norm: true,
        fuse_linear_layer_norm: true,
        fuse_resblock_chain: true,
        fuse_activation_conv1d: true,
    };
    assert!(is_default_config(&config));
    assert_eq!(config, PeepholeConfig::default());
}
