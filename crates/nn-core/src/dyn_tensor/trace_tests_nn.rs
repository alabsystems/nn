// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! nn Module tracing tests for DynTensor computation graph (D2/D4/D5).
//!
//! Tests that each nn layer's `forward()` records the correct composite
//! `TraceOp` node in the trace graph.

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

fn t1d(data: &[f32]) -> DynTensor {
    DynTensor::new(data, &[data.len()], &cpu()).unwrap()
}

fn t2d(data: &[f32], rows: usize, cols: usize) -> DynTensor {
    DynTensor::new(data, &[rows, cols], &cpu()).unwrap()
}

// -- nn Module tracing (D2/D4) ------------------------------------------------

#[test]
fn test_trace_linear() {
    use crate::layers::{Linear, Module};

    let weight = t2d(&[1.0, 0.0, 0.0, 1.0], 2, 2);
    let bias = t1d(&[0.5, -0.5]);
    let linear = Linear::new(weight, Some(bias)).unwrap();

    let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);

    let (result, graph): (DynTensor, _) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = linear.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // Graph has input + intermediate primitive ops + composite Linear node
    assert!(graph.len() >= 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Linear { weight, bias } => {
            assert_eq!(weight.shape(), &[2, 2]);
            assert!(bias.is_some());
        }
        other => panic!("expected Linear, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[2, 2]);
    assert_eq!(result.dims(), &[2, 2]);
}

#[test]
fn test_trace_layer_norm() {
    use crate::layers::{LayerNorm, Module};

    let weight = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let bias = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let (_, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = ln.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // Graph has input + intermediate primitive ops + composite LayerNorm node
    assert!(graph.len() >= 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::LayerNorm { eps, weight, bias } => {
            assert!((*eps - 1e-5).abs() < 1e-12);
            assert_eq!(weight.shape(), &[4]);
            assert_eq!(bias.shape(), &[4]);
        }
        other => panic!("expected LayerNorm, got {other:?}"),
    }
}

#[test]
fn test_trace_group_norm() {
    use crate::layers::{GroupNorm, Module};

    let weight = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let bias = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    let gn = GroupNorm::new(2, 4, weight, bias, 1e-5).unwrap();

    // [batch=1, channels=4, spatial=3]
    let x = DynTensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[1, 4, 3],
        &cpu(),
    )
    .unwrap();

    let (_, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = gn.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // Graph has input + intermediate primitive ops + composite GroupNorm node
    assert!(graph.len() >= 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::GroupNorm {
            num_groups,
            eps,
            weight,
            bias,
        } => {
            assert_eq!(*num_groups, 2);
            assert!((*eps - 1e-5).abs() < 1e-12);
            assert_eq!(weight.shape(), &[4]);
            assert_eq!(bias.shape(), &[4]);
        }
        other => panic!("expected GroupNorm, got {other:?}"),
    }
}

#[test]
fn test_trace_rms_norm() {
    use crate::layers::{Module, RmsNorm};

    let weight = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let rn = RmsNorm::new(weight, 1e-5).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let (_, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = rn.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // Graph has input + intermediate primitive ops + composite RmsNorm node
    assert!(graph.len() >= 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::RmsNorm { eps, weight } => {
            assert!((*eps - 1e-5).abs() < 1e-12);
            assert_eq!(weight.shape(), &[4]);
        }
        other => panic!("expected RmsNorm, got {other:?}"),
    }
}

#[test]
fn test_trace_batch_norm() {
    use crate::layers::{BatchNorm, Module};

    let mean = DynTensor::zeros(&[2], DType::F32, &cpu()).unwrap();
    let var = DynTensor::ones(&[2], DType::F32, &cpu()).unwrap();
    let weight = DynTensor::ones(&[2], DType::F32, &cpu()).unwrap();
    let bias = DynTensor::zeros(&[2], DType::F32, &cpu()).unwrap();
    let bn = BatchNorm::new(mean, var, Some(weight), Some(bias), 1e-5).unwrap();

    // [batch=1, channels=2, length=3]
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &cpu()).unwrap();

    let (_, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = bn.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // Graph has input + intermediate primitive ops + composite BatchNorm node
    assert!(graph.len() >= 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::BatchNorm {
            eps,
            weight,
            bias,
            running_mean,
            running_var,
        } => {
            assert!((*eps - 1e-5).abs() < 1e-12);
            assert_eq!(weight.shape(), &[2]);
            assert_eq!(bias.shape(), &[2]);
            assert_eq!(running_mean.shape(), &[2]);
            assert_eq!(running_var.shape(), &[2]);
        }
        other => panic!("expected BatchNorm, got {other:?}"),
    }
}

#[test]
fn test_trace_instance_norm() {
    use crate::layers::{InstanceNorm, Module};

    let inorm = InstanceNorm::new(1e-5).unwrap();

    // [batch=1, channels=2, spatial=4]
    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 2, 4],
        &cpu(),
    )
    .unwrap();

    let (_, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = inorm.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // input + intermediate primitive ops + InstanceNorm (composite ops decompose)
    assert!(graph.len() >= 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::InstanceNorm { eps } => {
            assert!((*eps - 1e-5).abs() < 1e-12);
        }
        other => panic!("expected InstanceNorm, got {other:?}"),
    }
}

// -- Embedding tracing (D5) ---------------------------------------------------

#[test]
fn test_trace_embedding() {
    use crate::layers::{Embedding, Module};

    // vocab_size=3, embed_dim=2
    let embed_weight = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu()).unwrap();
    let emb = Embedding::new(embed_weight).unwrap();

    // token ids [0, 2, 1]
    let ids = DynTensor::from_vec_u32(vec![0, 2, 1], &[3], &cpu()).unwrap();

    let (result, graph): (DynTensor, _) = trace_graph(|| {
        let mut ids = ids.clone();
        let id = record_input(&[3], DType::U32).unwrap();
        ids.set_trace_id(id);
        let y = emb.forward(&ids)?;
        Ok(y)
    })
    .unwrap();

    // input + intermediate ops (reshape, etc.) + Embedding (composite ops decompose)
    assert!(graph.len() >= 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Embedding { weight } => {
            assert_eq!(weight.shape(), &[3, 2]);
        }
        other => panic!("expected Embedding, got {other:?}"),
    }
    assert_eq!(result.dims(), &[3, 2]);
}

// -- Conv tracing (D5) --------------------------------------------------------

#[test]
fn test_trace_conv1d() {
    // input: [batch=1, channels=1, length=5]
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    // kernel: [out_ch=1, in_ch=1, kernel_size=3]
    let k = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let mut k = k.clone();
        let id_x = record_input(&[1, 1, 5], DType::F32).unwrap();
        x.set_trace_id(id_x);
        let id_k = record_input(&[1, 1, 3], DType::F32).unwrap();
        k.set_trace_id(id_k);
        let y = x.conv1d(&k, 0, 1, 1, 1)?;
        Ok(y)
    })
    .unwrap();

    // 2 inputs + 1 conv1d = 3 nodes
    assert_eq!(graph.len(), 3);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Conv1d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => {
            assert_eq!(weight.shape(), &[1, 1, 3]);
            assert!(bias.is_none());
            assert_eq!(*padding, 0);
            assert_eq!(*stride, 1);
            assert_eq!(*dilation, 1);
            assert_eq!(*groups, 1);
        }
        other => panic!("expected Conv1d, got {other:?}"),
    }
    // conv1d with padding=0, stride=1: output length = 5 - 3 + 1 = 3
    assert_eq!(output.output_shape(), &[1, 1, 3]);
    assert_eq!(output.inputs().len(), 2);

    // Verify computation: [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![6.0, 9.0, 12.0]);
}

#[test]
fn test_trace_conv2d() {
    // input: [batch=1, channels=1, height=3, width=3]
    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
        &cpu(),
    )
    .unwrap();
    // kernel: [out_ch=1, in_ch=1, kH=2, kW=2]
    let k = DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let mut k = k.clone();
        let id_x = record_input(&[1, 1, 3, 3], DType::F32).unwrap();
        x.set_trace_id(id_x);
        let id_k = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        k.set_trace_id(id_k);
        let y = x.conv2d(&k, 0, 1, 1, 1)?;
        Ok(y)
    })
    .unwrap();

    // 2 inputs + 1 conv2d = 3 nodes
    assert_eq!(graph.len(), 3);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Conv2d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => {
            assert_eq!(weight.shape(), &[1, 1, 2, 2]);
            assert!(bias.is_none());
            assert_eq!(*padding, [0, 0]);
            assert_eq!(*stride, [1, 1]);
            assert_eq!(*dilation, [1, 1]);
            assert_eq!(*groups, 1);
        }
        other => panic!("expected Conv2d, got {other:?}"),
    }
    // conv2d with padding=0, stride=1: output = [1, 1, 2, 2]
    assert_eq!(output.output_shape(), &[1, 1, 2, 2]);
    assert_eq!(output.inputs().len(), 2);

    // Verify computation: 2x2 windows summed
    // [1+2+4+5, 2+3+5+6, 4+5+7+8, 5+6+8+9] = [12, 16, 24, 28]
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![12.0, 16.0, 24.0, 28.0]);
}

// -- WeightRef data capture tests ---------------------------------------------

#[test]
fn test_to_weight_ref_captures_data() {
    // CPU f32 tensor should have data captured, not just shape.
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = DynTensor::new(&data, &[2, 3], &cpu()).unwrap();
    let wref = t.to_weight_ref().unwrap();

    assert_eq!(wref.shape(), &[2, 3]);
    assert!(
        !wref.data().is_empty(),
        "WeightRef data should not be empty for CPU f32 tensor"
    );
    assert_eq!(wref.data().len(), 6);
    assert_eq!(wref.data(), data.as_slice());
}

#[test]
fn test_to_weight_ref_linear_has_weight_data() {
    use crate::layers::{Linear, Module};

    let weight_data = vec![1.0f32, 0.0, 0.0, 1.0];
    let weight = t2d(&weight_data, 2, 2);
    let bias_data = vec![0.5f32, -0.5];
    let bias = t1d(&bias_data);
    let linear = Linear::new(weight, Some(bias)).unwrap();

    let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);

    let (_, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = linear.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Linear { weight, bias } => {
            // Weight data should be non-empty (actual data captured)
            assert!(
                !weight.data().is_empty(),
                "Linear weight data should be captured"
            );
            assert_eq!(weight.data().len(), 4);
            assert_eq!(weight.data(), weight_data.as_slice());
            assert_eq!(weight.shape(), &[2, 2]);

            // Bias data should also be non-empty
            let bias = bias.as_ref().unwrap();
            assert!(
                !bias.data().is_empty(),
                "Linear bias data should be captured"
            );
            assert_eq!(bias.data().len(), 2);
            assert_eq!(bias.data(), bias_data.as_slice());
        }
        other => panic!("expected Linear, got {other:?}"),
    }
}

#[test]
fn test_to_weight_ref_layer_norm_has_weight_data() {
    use crate::layers::{LayerNorm, Module};

    let weight = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let bias = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let (_, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = ln.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::LayerNorm { weight, bias, .. } => {
            assert!(
                !weight.data().is_empty(),
                "LayerNorm weight data should be captured"
            );
            assert_eq!(weight.data(), &[1.0f32; 4]);
            assert!(
                !bias.data().is_empty(),
                "LayerNorm bias data should be captured"
            );
            assert_eq!(bias.data(), &[0.0f32; 4]);
        }
        other => panic!("expected LayerNorm, got {other:?}"),
    }
}
