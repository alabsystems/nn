// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests: F16 autocast passthrough.
//!
//! Validates that passthrough-safe activations (ReLU, LeakyRelu, ELU, etc.)
//! between F16 compute ops inherit the predecessor's dtype instead of forcing
//! F16→F32→F16 cast roundtrips. Matches PyTorch's "implicit" autocast category.
//!
//! Part of #2981.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::helpers::{assert_close, create_input_buffer, input_node};

fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
    WeightRef::new(data, shape).expect("weight")
}

// -- Conv1d → LeakyRelu → Conv1d passthrough (#2981) --------------------------

/// Autocast passthrough: LeakyRelu between two Conv1d layers should stay F16
/// instead of forcing F16→F32→F16 cast roundtrip. Validates the forward
/// propagation pass in `compiled_model_builder.rs`. Part of #2981.
#[test]
fn test_autocast_conv_activation_passthrough() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;
    use nn_metal::compiled_model::CompiledModel;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_ch, mid_ch, out_ch, ks, in_len, pad) = (4, 8, 6, 3, 16, 1);
    let mid_len = (in_len + 2 * pad - ks) + 1; // 16
    let out_len = (mid_len + 2 * pad - ks) + 1; // 16

    let w1 = super::test_utils::rand_f32_vec(0xFA55_0001, mid_ch * in_ch * ks, -0.5, 0.5);
    let b1 = super::test_utils::rand_f32_vec(0xFA55_0002, mid_ch, -0.1, 0.1);
    let w2 = super::test_utils::rand_f32_vec(0xFA55_0003, out_ch * mid_ch * ks, -0.5, 0.5);
    let b2 = super::test_utils::rand_f32_vec(0xFA55_0004, out_ch, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xFA55_0005, in_ch * in_len, -1.0, 1.0);

    // Graph: input → Conv1d → LeakyRelu → Conv1d
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[in_ch, in_len]),
        TraceNode::new(
            1,
            "conv1d_a".into(),
            TraceOp::Conv1d {
                weight: weight(w1, vec![mid_ch, in_ch, ks]),
                bias: Some(weight(b1, vec![mid_ch])),
                padding: pad,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vec![0],
            vec![mid_ch, mid_len],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "leaky_relu_0".into(),
            TraceOp::LeakyRelu { slope: 0.01 },
            vec![1],
            vec![mid_ch, mid_len],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "conv1d_b".into(),
            TraceOp::Conv1d {
                weight: weight(w2, vec![out_ch, mid_ch, ks]),
                bias: Some(weight(b2, vec![out_ch])),
                padding: pad,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vec![2],
            vec![out_ch, out_len],
            DType::F32,
        ),
    ]);

    let input_buf = create_input_buffer(&cache, &input_data);
    let numel = out_ch * out_len;

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_buf = f32_model.execute(&cache, &[&input_buf]).expect("f32 exec");
    let f32_result = super::helpers::read_output_n(&f32_buf, numel);

    // Autocast with passthrough.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast(), "model should be autocast");

    // Key assertion: with passthrough, LeakyRelu (step 2) + both Conv1d steps
    // should all be F16 → at least 3 F16 steps (conv_a, leaky_relu, conv_b).
    let f16_count = ac_model.num_autocast_f16_steps();
    assert!(
        f16_count >= 3,
        "expected >= 3 F16 steps (2 conv + 1 leaky_relu passthrough), got {f16_count}"
    );

    let ac_buf = ac_model
        .execute(&cache, &[&input_buf])
        .expect("autocast exec");
    let ac_result = super::helpers::read_output_n(&ac_buf, numel);

    // Two conv layers in F16 → slightly larger tolerance.
    assert_close("conv_activation_passthrough", &ac_result, &f32_result, 5e-2);
}

// -- Conv1d → Sigmoid → Conv1d passthrough (#3275) ----------------------------

/// Autocast passthrough: Sigmoid/Tanh/Gelu between Conv1d layers should stay
/// F16 instead of forcing casts. Validates the 4 ops added in #3275.
#[test]
fn test_autocast_sigmoid_tanh_gelu_passthrough() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;
    use nn_metal::compiled_model::CompiledModel;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_ch, mid_ch, out_ch, ks, in_len, pad) = (4, 8, 6, 3, 16, 1);
    let mid_len = (in_len + 2 * pad - ks) + 1; // 16
    let out_len = (mid_len + 2 * pad - ks) + 1; // 16

    let w1 = super::test_utils::rand_f32_vec(0x3275_0001, mid_ch * in_ch * ks, -0.3, 0.3);
    let b1 = super::test_utils::rand_f32_vec(0x3275_0002, mid_ch, -0.1, 0.1);
    let w2 = super::test_utils::rand_f32_vec(0x3275_0003, out_ch * mid_ch * ks, -0.3, 0.3);
    let b2 = super::test_utils::rand_f32_vec(0x3275_0004, out_ch, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0x3275_0005, in_ch * in_len, -1.0, 1.0);

    // Test each of the 4 newly-added passthrough ops.
    for (name, op) in [
        ("sigmoid", TraceOp::Sigmoid),
        ("tanh", TraceOp::Tanh),
        ("gelu", TraceOp::Gelu),
        ("gelu_erf", TraceOp::GeluErf),
    ] {
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[in_ch, in_len]),
            TraceNode::new(
                1,
                "conv1d_a".into(),
                TraceOp::Conv1d {
                    weight: weight(w1.clone(), vec![mid_ch, in_ch, ks]),
                    bias: Some(weight(b1.clone(), vec![mid_ch])),
                    padding: pad,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                },
                vec![0],
                vec![mid_ch, mid_len],
                DType::F32,
            ),
            TraceNode::new(
                2,
                format!("{name}_0"),
                op.clone(),
                vec![1],
                vec![mid_ch, mid_len],
                DType::F32,
            ),
            TraceNode::new(
                3,
                "conv1d_b".into(),
                TraceOp::Conv1d {
                    weight: weight(w2.clone(), vec![out_ch, mid_ch, ks]),
                    bias: Some(weight(b2.clone(), vec![out_ch])),
                    padding: pad,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                },
                vec![2],
                vec![out_ch, out_len],
                DType::F32,
            ),
        ]);

        let input_buf = create_input_buffer(&cache, &input_data);
        let numel = out_ch * out_len;

        // F32 baseline.
        let f32_model = CompiledModel::builder(&graph, &cache)
            .build()
            .expect("compile f32");
        let f32_result = super::helpers::read_output_n(
            &f32_model.execute(&cache, &[&input_buf]).expect("f32 exec"),
            numel,
        );

        // Autocast with passthrough.
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let ac_model = CompiledModel::builder(&graph, &cache)
            .autocast(policy)
            .build()
            .unwrap_or_else(|e| panic!("compile autocast {name}: {e}"));
        assert!(ac_model.is_autocast(), "{name}: model should be autocast");

        // At least 3 F16 steps: conv_a + activation + conv_b.
        let f16_count = ac_model.num_autocast_f16_steps();
        assert!(
            f16_count >= 3,
            "{name}: expected >= 3 F16 steps (2 conv + 1 {name} passthrough), got {f16_count}"
        );

        let ac_result = super::helpers::read_output_n(
            &ac_model
                .execute(&cache, &[&input_buf])
                .expect("autocast exec"),
            numel,
        );
        assert_close(
            &format!("conv_{name}_passthrough"),
            &ac_result,
            &f32_result,
            5e-2,
        );
    }
}
