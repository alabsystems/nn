// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for TraceOp construction/classification, DType properties, and nn layer configs.
//! Issue: #3816

use nn_core::dyn_tensor::trace::{TraceNode, TraceOp, TraceOpClass, WeightRef};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{
    Activation, BatchNormConfig, Conv1dConfig, Conv2dConfig, ConvTranspose1dConfig, Dropout,
    Embedding, LayerNorm, LayerNormConfig, Linear, Module, Pool1dConfig, RmsNorm, Sequential,
};
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// TraceOp construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_traceop_input_classification() {
    let op = TraceOp::Input;
    assert_eq!(op.classification(), TraceOpClass::Input);
    assert_eq!(op.expected_arity(), Some(0));
}

#[test]
fn test_traceop_binary_elementwise_classification() {
    for op in [
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Maximum,
        TraceOp::Minimum,
    ] {
        assert_eq!(
            op.classification(),
            TraceOpClass::BinaryElementwise,
            "expected BinaryElementwise for {op:?}"
        );
        assert_eq!(op.expected_arity(), Some(2), "expected arity 2 for {op:?}");
    }
}

#[test]
fn test_traceop_unary_elementwise_classification() {
    let ops = [
        TraceOp::Relu,
        TraceOp::Gelu,
        TraceOp::Silu,
        TraceOp::Tanh,
        TraceOp::Sigmoid,
        TraceOp::Exp,
        TraceOp::Log,
        TraceOp::Sqrt,
        TraceOp::Sqr,
        TraceOp::Abs,
        TraceOp::Neg,
        TraceOp::Recip,
        TraceOp::Sin,
        TraceOp::Cos,
        TraceOp::Tan,
        TraceOp::Floor,
        TraceOp::Ceil,
        TraceOp::Round,
        TraceOp::Sign,
        TraceOp::Fract,
    ];
    for op in ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::UnaryElementwise,
            "expected UnaryElementwise for {op:?}"
        );
        assert_eq!(op.expected_arity(), Some(1), "expected arity 1 for {op:?}");
    }
}

#[test]
fn test_traceop_reduction_classification() {
    let ops = [
        TraceOp::ReduceSum {
            dim: 0,
            keepdim: false,
        },
        TraceOp::ReduceMean {
            dim: 1,
            keepdim: true,
        },
        TraceOp::ReduceMax {
            dim: 2,
            keepdim: false,
        },
        TraceOp::ReduceMin {
            dim: 0,
            keepdim: true,
        },
    ];
    for op in ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::Reduction,
            "expected Reduction for {op:?}"
        );
        assert_eq!(op.expected_arity(), Some(1));
    }
}

#[test]
fn test_traceop_shape_only_classification() {
    let ops = [
        TraceOp::Reshape {
            target_shape: vec![2, 3],
        },
        TraceOp::Unsqueeze { dim: 0 },
        TraceOp::Squeeze { dim: 1 },
    ];
    for op in ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::ShapeOnly,
            "expected ShapeOnly for {op:?}"
        );
        assert_eq!(op.expected_arity(), Some(1));
    }
}

#[test]
fn test_traceop_shape_data_move_classification() {
    let ops = [
        TraceOp::Transpose { dim0: 0, dim1: 1 },
        TraceOp::Permute {
            axes: vec![2, 0, 1],
        },
        TraceOp::Narrow {
            dim: 0,
            start: 0,
            length: 5,
        },
        TraceOp::Cat {
            dim: 0,
            num_inputs: 3,
        },
        TraceOp::Expand {
            target_shape: vec![4, 4],
        },
    ];
    for op in &ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::ShapeDataMove,
            "expected ShapeDataMove for {op:?}"
        );
    }
}

#[test]
fn test_traceop_cat_variable_arity() {
    let cat2 = TraceOp::Cat {
        dim: 0,
        num_inputs: 2,
    };
    assert_eq!(cat2.expected_arity(), Some(2));

    let cat5 = TraceOp::Cat {
        dim: 1,
        num_inputs: 5,
    };
    assert_eq!(cat5.expected_arity(), Some(5));
}

#[test]
fn test_traceop_normalization_classification() {
    let w = WeightRef::from_shape(&[4]);
    let b = WeightRef::from_shape(&[4]);
    let ops = [
        TraceOp::LayerNorm {
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone(),
        },
        TraceOp::RmsNorm {
            eps: 1e-5,
            weight: w.clone(),
        },
        TraceOp::GroupNorm {
            num_groups: 2,
            eps: 1e-5,
            weight: w,
            bias: b,
        },
        TraceOp::InstanceNorm { eps: 1e-5 },
    ];
    for op in &ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::Normalization,
            "expected Normalization for {op:?}"
        );
        assert_eq!(op.expected_arity(), Some(1));
    }
}

#[test]
fn test_traceop_attention_classification() {
    let ops = [
        TraceOp::Softmax { dim: 1 },
        TraceOp::LogSoftmax { dim: 1 },
        TraceOp::SdpaCausal { scale: 0.125 },
    ];
    for op in &ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::Attention,
            "expected Attention for {op:?}"
        );
    }
    // Softmax and LogSoftmax are unary
    assert_eq!(TraceOp::Softmax { dim: 0 }.expected_arity(), Some(1));
    assert_eq!(TraceOp::LogSoftmax { dim: 0 }.expected_arity(), Some(1));
    // SdpaCausal takes Q, K, V = 3
    assert_eq!(
        TraceOp::SdpaCausal { scale: 0.125 }.expected_arity(),
        Some(3)
    );
}

#[test]
fn test_traceop_named_activation_classification() {
    let ops = [
        TraceOp::Elu { alpha: 1.0 },
        TraceOp::LeakyRelu { slope: 0.01 },
        TraceOp::Softplus,
        TraceOp::Selu,
        TraceOp::Celu { alpha: 1.0 },
        TraceOp::Mish,
        TraceOp::HardSigmoid,
        TraceOp::HardSwish,
        TraceOp::Softsign,
    ];
    for op in &ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::NamedActivation,
            "expected NamedActivation for {op:?}"
        );
        assert_eq!(op.expected_arity(), Some(1));
    }
}

#[test]
fn test_traceop_pooling_classification() {
    let ops = [
        TraceOp::MaxPool1d {
            kernel_size: 2,
            stride: 2,
            padding: 0,
        },
        TraceOp::AvgPool2d {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
        },
        TraceOp::MaxPool2d {
            kernel_size: [3, 3],
            stride: [1, 1],
            padding: [1, 1],
        },
        TraceOp::AdaptiveAvgPool2d {
            output_size: [1, 1],
        },
    ];
    for op in &ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::Pooling,
            "expected Pooling for {op:?}"
        );
        assert_eq!(op.expected_arity(), Some(1));
    }
}

#[test]
fn test_traceop_custom_classification() {
    let op = TraceOp::Custom {
        name: "nn_custom_op".into(),
    };
    assert_eq!(op.classification(), TraceOpClass::Custom);
    assert_eq!(op.expected_arity(), Some(1));
}

#[test]
fn test_traceop_dropout_is_identity() {
    let op = TraceOp::Dropout;
    assert_eq!(op.classification(), TraceOpClass::Identity);
    assert_eq!(op.expected_arity(), Some(1));
}

#[test]
fn test_traceop_matmul_classification() {
    let op = TraceOp::MatMul;
    assert_eq!(op.classification(), TraceOpClass::MatMul);
    assert_eq!(op.expected_arity(), Some(2));
}

#[test]
fn test_traceop_weighted_linear_classification() {
    let w = WeightRef::from_shape(&[4, 3]);
    let op = TraceOp::Linear {
        weight: w,
        bias: None,
    };
    assert_eq!(op.classification(), TraceOpClass::WeightedLinear);
    assert_eq!(op.expected_arity(), Some(1));
}

#[test]
fn test_traceop_conv1d_with_bias() {
    let w = WeightRef::from_shape(&[8, 4, 3]);
    let b = WeightRef::from_shape(&[8]);
    let op = TraceOp::Conv1d {
        weight: w,
        bias: Some(b),
        padding: 1,
        stride: 1,
        dilation: 1,
        groups: 1,
    };
    assert_eq!(op.classification(), TraceOpClass::WeightedLinear);
    assert_eq!(op.expected_arity(), Some(1));
}

// ---------------------------------------------------------------------------
// TraceNode construction test
// ---------------------------------------------------------------------------

#[test]
fn test_tracenode_construction_and_accessors() {
    let node = TraceNode::new(
        42,
        "relu_0".into(),
        TraceOp::Relu,
        vec![1],
        vec![2, 3],
        DType::F32,
    );
    assert_eq!(node.id(), 42);
    assert_eq!(node.name(), "relu_0");
    assert!(matches!(node.op(), TraceOp::Relu));
    assert_eq!(node.inputs(), &[1]);
    assert_eq!(node.output_shape(), &[2, 3]);
    assert_eq!(node.output_dtype(), DType::F32);
}

// ---------------------------------------------------------------------------
// WeightRef construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_weightref_new_valid() {
    let wr = WeightRef::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    assert_eq!(wr.data().len(), 6);
    assert_eq!(wr.shape(), &[2, 3]);
}

#[test]
fn test_weightref_new_mismatch_errors() {
    let result = WeightRef::new(vec![1.0, 2.0, 3.0], vec![2, 3]);
    assert!(result.is_err(), "data length 3 != shape product 6");
}

#[test]
fn test_weightref_from_shape_has_empty_data() {
    let wr = WeightRef::from_shape(&[4, 8]);
    assert!(wr.data().is_empty());
    assert_eq!(wr.shape(), &[4, 8]);
}

// ---------------------------------------------------------------------------
// DType tests (supplementary to existing inline tests)
// ---------------------------------------------------------------------------

#[test]
fn test_dtype_float_int_partition() {
    // Every variant is either float, int, or Bool (neither)
    let all = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    for dt in all {
        let count = [dt.is_float(), dt.is_int(), dt == DType::Bool]
            .iter()
            .filter(|&&b| b)
            .count();
        assert_eq!(count, 1, "{dt} should be exactly one of float/int/bool");
    }
}

#[test]
fn test_dtype_size_bytes_power_of_two() {
    // All dtype sizes are powers of two (1, 2, 4, 8)
    let all = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    for dt in all {
        let sz = dt.size_bytes();
        assert!(sz.is_power_of_two(), "{dt} size {sz} is not power of two");
    }
}

#[test]
fn test_dtype_display_roundtrip_lowercase() {
    // Display is lowercase, no spaces
    let all = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    for dt in all {
        let s = format!("{dt}");
        assert_eq!(s, s.to_lowercase(), "{dt} display should be lowercase");
        assert!(!s.contains(' '), "{dt} display should have no spaces");
    }
}

#[test]
fn test_dtype_clone_eq() {
    let a = DType::BF16;
    let b = a;
    assert_eq!(a, b);
    assert_eq!(a.size_bytes(), b.size_bytes());
}

// ---------------------------------------------------------------------------
// nn layer config tests
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_config_default() {
    let cfg = Conv1dConfig::default();
    assert_eq!(cfg.padding, 0);
    assert_eq!(cfg.stride, 1);
    assert_eq!(cfg.dilation, 1);
    assert_eq!(cfg.groups, 1);
}

#[test]
fn test_conv1d_config_builder_chain() {
    let cfg = Conv1dConfig::new(2, 3, 1).with_groups(4);
    assert_eq!(cfg.padding, 2);
    assert_eq!(cfg.stride, 3);
    assert_eq!(cfg.dilation, 1);
    assert_eq!(cfg.groups, 4);
}

#[test]
fn test_conv2d_config_default() {
    let cfg = Conv2dConfig::default();
    assert_eq!(cfg.padding, 0);
    assert_eq!(cfg.stride, 1);
    assert_eq!(cfg.dilation, 1);
    assert_eq!(cfg.groups, 1);
}

#[test]
fn test_conv_transpose1d_config_default() {
    let cfg = ConvTranspose1dConfig::default();
    assert_eq!(cfg.padding, 0);
    assert_eq!(cfg.output_padding, 0);
    assert_eq!(cfg.stride, 1);
    assert_eq!(cfg.dilation, 1);
    assert_eq!(cfg.groups, 1);
}

#[test]
fn test_batch_norm_config_default() {
    let cfg = BatchNormConfig::default();
    assert!((cfg.eps - 1e-5).abs() < 1e-10);
    assert!(cfg.remove_mean);
    assert!(cfg.affine);
    assert!((cfg.momentum - 0.1).abs() < 1e-10);
}

#[test]
fn test_batch_norm_config_builder_chain() {
    let cfg = BatchNormConfig::new(1e-3)
        .with_remove_mean(false)
        .with_affine(false)
        .with_momentum(0.01);
    assert!((cfg.eps - 1e-3).abs() < 1e-10);
    assert!(!cfg.remove_mean);
    assert!(!cfg.affine);
    assert!((cfg.momentum - 0.01).abs() < 1e-10);
}

#[test]
fn test_layer_norm_config_default() {
    let cfg = LayerNormConfig::default();
    assert!((cfg.eps - 1e-5).abs() < 1e-10);
}

#[test]
fn test_layer_norm_config_custom_eps() {
    let cfg = LayerNormConfig::new(1e-6);
    assert!((cfg.eps - 1e-6).abs() < 1e-12);
}

#[test]
fn test_pool1d_config_stride_defaults_to_kernel_size() {
    let cfg = Pool1dConfig::new(4);
    assert_eq!(cfg.kernel_size, 4);
    assert_eq!(cfg.stride, 4); // defaults to kernel_size
    assert_eq!(cfg.padding, 0);
}

#[test]
fn test_pool1d_config_builder() {
    let cfg = Pool1dConfig::new(3).with_stride(2).with_padding(1);
    assert_eq!(cfg.kernel_size, 3);
    assert_eq!(cfg.stride, 2);
    assert_eq!(cfg.padding, 1);
}

// ---------------------------------------------------------------------------
// nn layer construction validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_linear_rejects_non_2d_weight() {
    let w = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let result = Linear::new(w, None);
    assert!(result.is_err());
}

#[test]
fn test_linear_rejects_mismatched_bias() {
    let w = DynTensor::ones(&[4, 3], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::ones(&[3], DType::F32, &Device::Cpu).unwrap(); // should be [4]
    let result = Linear::new(w, Some(b));
    assert!(result.is_err());
}

#[test]
fn test_linear_accepts_matching_bias() {
    let w = DynTensor::ones(&[4, 3], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let lin = Linear::new(w, Some(b)).unwrap();
    assert_eq!(lin.out_features(), 4);
}

#[test]
fn test_embedding_rejects_non_2d_weight() {
    let w = DynTensor::ones(&[10], DType::F32, &Device::Cpu).unwrap();
    let result = Embedding::new(w);
    assert!(result.is_err());
}

#[test]
fn test_embedding_accepts_2d_weight() {
    let w = DynTensor::ones(&[100, 32], DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(w).unwrap();
    assert_eq!(emb.weight().dims(), &[100, 32]);
}

#[test]
fn test_rms_norm_rejects_non_1d_weight() {
    let w = DynTensor::ones(&[4, 4], DType::F32, &Device::Cpu).unwrap();
    let result = RmsNorm::new(w, 1e-5);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_rejects_mismatched_weight_bias() {
    let w = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::ones(&[8], DType::F32, &Device::Cpu).unwrap();
    let result = LayerNorm::new(w, b, 1e-5);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_accepts_matching_shapes() {
    let w = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(w, b, 1e-5).unwrap();
    assert_eq!(ln.weight().dims(), &[4]);
}

#[test]
fn test_sequential_len_and_forward() {
    let mut seq = Sequential::new();
    assert!(seq.is_empty());
    assert_eq!(seq.len(), 0);

    seq.add(Activation::Relu);
    seq.add(Activation::Sigmoid);
    assert_eq!(seq.len(), 2);
    assert!(!seq.is_empty());

    let x = DynTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], &Device::Cpu).unwrap();
    let y = seq.forward(&x).unwrap();
    assert_eq!(y.dims(), &[3]);
    // relu(-1,0,1) = (0,0,1), sigmoid(0,0,1) = (0.5, 0.5, ~0.731)
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 0.5).abs() < 1e-4);
    assert!((vals[1] - 0.5).abs() < 1e-4);
}

#[test]
fn test_dropout_is_identity_at_inference() {
    let d = Dropout::new(0.9);
    let x = DynTensor::from_vec(vec![42.0, -7.5], &[2], &Device::Cpu).unwrap();
    let y = d.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![42.0, -7.5]);
}
