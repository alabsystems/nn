// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for verifiability classification.

use nn_core::dyn_tensor::trace::TraceOp;

use super::{classify_op, VerifiabilityClass};

#[test]
fn test_relu_is_verifiable() {
    assert_eq!(classify_op(&TraceOp::Relu), VerifiabilityClass::Verifiable);
}

#[test]
fn test_add_is_verifiable() {
    assert_eq!(classify_op(&TraceOp::Add), VerifiabilityClass::Verifiable);
}

#[test]
fn test_linear_is_verifiable() {
    let op = TraceOp::Linear {
        weight: nn_core::dyn_tensor::trace::WeightRef::new(vec![0.0; 16], vec![4, 4]).unwrap(),
        bias: None,
    };
    assert_eq!(classify_op(&op), VerifiabilityClass::Verifiable);
}

#[test]
fn test_reshape_is_shape_only() {
    let op = TraceOp::Reshape {
        target_shape: vec![2, 3],
    };
    assert_eq!(classify_op(&op), VerifiabilityClass::ShapeOnly);
}

#[test]
fn test_instance_norm_is_bounded() {
    let op = TraceOp::InstanceNorm { eps: 1e-5 };
    match classify_op(&op) {
        VerifiabilityClass::VerifiableBounded { max_dim } => {
            assert_eq!(max_dim, 512);
        }
        other => panic!("expected VerifiableBounded, got {other:?}"),
    }
}

#[test]
fn test_atan2_is_verifiable() {
    assert_eq!(classify_op(&TraceOp::Atan2), VerifiabilityClass::Verifiable);
}

#[test]
fn test_sdpa_is_verifiable_bounded() {
    let op = TraceOp::Sdpa { scale: 1.0 };
    match classify_op(&op) {
        VerifiabilityClass::VerifiableBounded { max_dim } => assert_eq!(max_dim, 512),
        other => panic!("expected VerifiableBounded, got {other:?}"),
    }
}

#[test]
fn test_sdpa_causal_is_verifiable_bounded() {
    let op = TraceOp::SdpaCausal { scale: 1.0 };
    match classify_op(&op) {
        VerifiabilityClass::VerifiableBounded { max_dim } => assert_eq!(max_dim, 512),
        other => panic!("expected VerifiableBounded, got {other:?}"),
    }
}

#[test]
fn test_dropout_is_passthrough() {
    assert_eq!(
        classify_op(&TraceOp::Dropout),
        VerifiabilityClass::Passthrough
    );
}

#[test]
fn test_powf_special_cases() {
    // x^1 = identity
    assert_eq!(
        classify_op(&TraceOp::Powf { exponent: 1.0 }),
        VerifiabilityClass::Verifiable
    );
    // x^2 = sqr
    assert_eq!(
        classify_op(&TraceOp::Powf { exponent: 2.0 }),
        VerifiabilityClass::Verifiable
    );
    // x^0.5 = sqrt
    assert_eq!(
        classify_op(&TraceOp::Powf { exponent: 0.5 }),
        VerifiabilityClass::Verifiable
    );
    // x^3 = not verifiable
    assert_eq!(
        classify_op(&TraceOp::Powf { exponent: 3.0 }),
        VerifiabilityClass::UnverifiableLearned
    );
}

#[test]
fn test_allows_compilation() {
    assert!(VerifiabilityClass::Verifiable.allows_compilation());
    assert!(VerifiabilityClass::ShapeOnly.allows_compilation());
    assert!(VerifiabilityClass::UnverifiableSafe.allows_compilation());
    assert!(!VerifiabilityClass::UnverifiableLearned.allows_compilation());
}

#[test]
fn test_needs_decomposition() {
    let bounded = VerifiabilityClass::VerifiableBounded { max_dim: 512 };
    assert!(!bounded.needs_decomposition(256));
    assert!(!bounded.needs_decomposition(512));
    assert!(bounded.needs_decomposition(1024));

    assert!(!VerifiabilityClass::Verifiable.needs_decomposition(10000));
}

#[test]
fn test_custom_op_is_unverifiable() {
    let op = TraceOp::Custom {
        name: "nn_op".to_string(),
    };
    assert_eq!(classify_op(&op), VerifiabilityClass::UnverifiableLearned);
}

#[test]
fn test_clamp_is_verifiable() {
    let op = TraceOp::Clamp {
        min: Some(-1.0),
        max: Some(1.0),
    };
    assert_eq!(classify_op(&op), VerifiabilityClass::Verifiable);
}

#[test]
fn test_embedding_is_verifiable() {
    let op = TraceOp::Embedding {
        weight: nn_core::dyn_tensor::trace::WeightRef::new(vec![0.0; 6400], vec![100, 64])
            .unwrap(),
    };
    assert_eq!(classify_op(&op), VerifiabilityClass::Verifiable);
}

#[test]
fn test_index_select_is_verifiable() {
    assert_eq!(
        classify_op(&TraceOp::IndexSelect { dim: 0 }),
        VerifiabilityClass::Verifiable
    );
}

#[test]
fn test_gather_is_verifiable() {
    assert_eq!(
        classify_op(&TraceOp::Gather { dim: 0 }),
        VerifiabilityClass::Verifiable
    );
}

#[test]
fn test_rotary_embedding_is_verifiable() {
    let wr = nn_core::dyn_tensor::trace::WeightRef::new(vec![0.0; 8], vec![2, 4]).unwrap();
    let op = TraceOp::RotaryEmbedding {
        head_dim: 8,
        offset: 0,
        cos_cache: wr.clone(),
        sin_cache: wr,
    };
    assert_eq!(classify_op(&op), VerifiabilityClass::Verifiable);
}

// -- classify_callee_name tests --

#[test]
fn test_classify_callee_name_verifiable_ops() {
    use super::classify_callee_name;
    let verifiable = [
        "relu",
        "gelu",
        "sigmoid",
        "tanh",
        "silu",
        "snake",
        "elu",
        "leaky_relu",
        "linear",
        "conv1d",
        "conv2d",
        "matmul",
        "embedding",
        "softmax",
        "avg_pool2d",
        "max_pool2d",
        "clamp",
        "index_select",
        "rope",
    ];
    for name in &verifiable {
        assert_eq!(
            classify_callee_name(name),
            VerifiabilityClass::Verifiable,
            "{name} should be Verifiable"
        );
    }
}

#[test]
fn test_classify_callee_name_bounded_ops() {
    use super::classify_callee_name;
    let bounded = [
        "layer_norm",
        "rms_norm",
        "instance_norm",
        "group_norm",
        "batch_norm",
    ];
    for name in &bounded {
        assert!(
            matches!(
                classify_callee_name(name),
                VerifiabilityClass::VerifiableBounded { .. }
            ),
            "{name} should be VerifiableBounded"
        );
    }
}

#[test]
fn test_classify_callee_name_shape_only() {
    use super::classify_callee_name;
    let shape = [
        "reshape",
        "transpose",
        "cat",
        "stack",
        "squeeze",
        "unsqueeze",
    ];
    for name in &shape {
        assert_eq!(
            classify_callee_name(name),
            VerifiabilityClass::ShapeOnly,
            "{name} should be ShapeOnly"
        );
    }
}

#[test]
fn test_classify_callee_name_unknown_is_unverifiable() {
    use super::classify_callee_name;
    assert_eq!(
        classify_callee_name("custom_attention_v2"),
        VerifiabilityClass::UnverifiableLearned,
    );
}

#[test]
fn test_is_verifiable() {
    assert!(VerifiabilityClass::Verifiable.is_verifiable());
    assert!(VerifiabilityClass::VerifiableBounded { max_dim: 512 }.is_verifiable());
    assert!(VerifiabilityClass::ShapeOnly.is_verifiable());
    assert!(VerifiabilityClass::Passthrough.is_verifiable());
    assert!(VerifiabilityClass::UnverifiableSafe.is_verifiable());
    assert!(!VerifiabilityClass::UnverifiableLearned.is_verifiable());
}

/// D4: Consistency test — every op that the trace-to-graph translator supports
/// must NOT be classified as `UnverifiableLearned`. This catches drift when
/// someone adds translator support but forgets to update `classify_op()`.
#[test]
fn test_translator_supported_ops_are_not_unverifiable_learned() {
    let wr = nn_core::dyn_tensor::trace::WeightRef::new(vec![0.0; 16], vec![4, 4]).unwrap();
    let translator_supported: Vec<TraceOp> = vec![
        // Unary activations
        TraceOp::Relu,
        TraceOp::Gelu,
        TraceOp::GeluErf,
        TraceOp::Sigmoid,
        TraceOp::Tanh,
        TraceOp::Exp,
        TraceOp::Log,
        TraceOp::Sqrt,
        TraceOp::Sqr,
        TraceOp::Abs,
        TraceOp::Neg,
        TraceOp::Recip,
        TraceOp::Sin,
        TraceOp::Cos,
        TraceOp::Silu,
        TraceOp::Floor,
        TraceOp::Round,
        // Fract removed: no trace_to_graph translator (#3226)
        TraceOp::Atan2,
        // Binary
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Maximum,
        TraceOp::Minimum,
        // Softmax
        TraceOp::Softmax { dim: 0 },
        TraceOp::LogSoftmax { dim: 0 },
        // Normalization
        TraceOp::LayerNorm {
            eps: 1e-5,
            weight: wr.clone(),
            bias: wr.clone(),
        },
        TraceOp::RmsNorm {
            eps: 1e-5,
            weight: wr.clone(),
        },
        TraceOp::InstanceNorm { eps: 1e-5 },
        TraceOp::GroupNorm {
            num_groups: 1,
            eps: 1e-5,
            weight: wr.clone(),
            bias: wr.clone(),
        },
        TraceOp::BatchNorm {
            eps: 1e-5,
            weight: wr.clone(),
            bias: wr.clone(),
            running_mean: wr.clone(),
            running_var: wr.clone(),
        },
        // Linear/Conv
        TraceOp::Linear {
            weight: wr.clone(),
            bias: None,
        },
        TraceOp::Conv1d {
            weight: wr.clone(),
            bias: None,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        },
        TraceOp::MatMul,
        // Attention
        TraceOp::Sdpa { scale: 1.0 },
        TraceOp::SdpaCausal { scale: 1.0 },
        TraceOp::RotaryEmbedding {
            head_dim: 8,
            offset: 0,
            cos_cache: wr.clone(),
            sin_cache: wr.clone(),
        },
        // Gather
        TraceOp::IndexSelect { dim: 0 },
        TraceOp::Gather { dim: 0 },
        // Clamp
        TraceOp::Clamp {
            min: Some(0.0),
            max: Some(1.0),
        },
        // Embedding
        TraceOp::Embedding { weight: wr },
        // Reductions
        TraceOp::ReduceSum {
            dim: 0,
            keepdim: false,
        },
        TraceOp::ReduceMean {
            dim: 0,
            keepdim: false,
        },
        // Pooling
        TraceOp::AvgPool2d {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
        },
        // Activations
        TraceOp::LeakyRelu { slope: 0.01 },
        TraceOp::Elu { alpha: 1.0 },
    ];
    for op in &translator_supported {
        let class = classify_op(op);
        assert!(
            class.allows_compilation(),
            "{} is classified as UnverifiableLearned but has translator support",
            op.canonical_name(),
        );
    }
}

/// #3226: Fract reclassified from Verifiable to UnverifiableSafe —
/// no trace_to_graph translator exists.
#[test]
fn test_fract_is_unverifiable_safe() {
    assert_eq!(
        classify_op(&TraceOp::Fract),
        VerifiabilityClass::UnverifiableSafe,
    );
}

/// #3226: Arange reclassified from Verifiable to UnverifiableSafe —
/// no trace_to_graph translator exists.
#[test]
fn test_arange_is_unverifiable_safe() {
    let op = TraceOp::Arange {
        start: 0.0,
        end: 10.0,
        step: 1.0,
    };
    assert_eq!(classify_op(&op), VerifiabilityClass::UnverifiableSafe);
}

/// AC2: Ops classified as Verifiable must have trace_to_graph support.
/// This is the inverse of `test_translator_supported_ops_are_not_unverifiable_learned`.
/// Fract and Arange were caught by this gap — classified Verifiable but no translator.
#[test]
fn test_verifiable_ops_not_in_translator_list_are_flagged() {
    // Ops that previously claimed Verifiable without translator support.
    // If any of these regress back to Verifiable, this test catches it.
    let no_translator = vec![
        TraceOp::Fract,
        TraceOp::Arange {
            start: 0.0,
            end: 1.0,
            step: 0.1,
        },
    ];
    for op in &no_translator {
        let class = classify_op(op);
        assert_ne!(
            class,
            VerifiabilityClass::Verifiable,
            "{} must NOT be Verifiable without a trace_to_graph translator",
            op.canonical_name(),
        );
    }
}
