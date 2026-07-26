// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Attention module tracing tests for DynTensor computation graph (#2147).
//!
//! Tests that SDPA, RoPE, and MultiHeadAttention forward passes record
//! the correct composite `TraceOp` nodes in the trace graph.

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

#[test]
fn test_trace_sdpa() {
    use crate::layers::attention::sdpa;

    // [B=1, H=2, S=3, head_dim=4]
    let q = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    let k = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    let v = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    let scale = 1.0 / (4.0f64).sqrt();

    let (result, graph) = trace_graph(|| {
        let mut q = q.clone();
        let mut k = k.clone();
        let mut v = v.clone();
        let id_q = record_input(&[1, 2, 3, 4], DType::F32).unwrap();
        q.set_trace_id(id_q);
        let id_k = record_input(&[1, 2, 3, 4], DType::F32).unwrap();
        k.set_trace_id(id_k);
        let id_v = record_input(&[1, 2, 3, 4], DType::F32).unwrap();
        v.set_trace_id(id_v);
        sdpa(&q, &k, &v, None, scale)
    })
    .unwrap();

    // 3 inputs + 1 composite Sdpa = 4 nodes (decomposed ops suppressed)
    assert_eq!(graph.len(), 4);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Sdpa { scale: s } => {
            assert!((s - scale).abs() < 1e-12);
        }
        other => panic!("expected Sdpa, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[1, 2, 3, 4]);
    assert_eq!(output.inputs().len(), 3);
    assert_eq!(result.dims(), &[1, 2, 3, 4]);
}

#[test]
fn test_trace_sdpa_causal() {
    use crate::layers::attention::sdpa_causal;

    // [B=1, H=2, S=3, head_dim=4]
    let q = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    let k = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    let v = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    let scale = 1.0 / (4.0f64).sqrt();

    let (result, graph) = trace_graph(|| {
        let mut q = q.clone();
        let mut k = k.clone();
        let mut v = v.clone();
        let id_q = record_input(&[1, 2, 3, 4], DType::F32).unwrap();
        q.set_trace_id(id_q);
        let id_k = record_input(&[1, 2, 3, 4], DType::F32).unwrap();
        k.set_trace_id(id_k);
        let id_v = record_input(&[1, 2, 3, 4], DType::F32).unwrap();
        v.set_trace_id(id_v);
        sdpa_causal(&q, &k, &v, scale)
    })
    .unwrap();

    // 3 inputs + 1 composite SdpaCausal = 4 nodes (decomposed ops suppressed)
    assert_eq!(graph.len(), 4);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::SdpaCausal { scale: s } => {
            assert!((s - scale).abs() < 1e-12);
        }
        other => panic!("expected SdpaCausal, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[1, 2, 3, 4]);
    // SdpaCausal always has 3 inputs (no mask)
    assert_eq!(output.inputs().len(), 3);
    assert_eq!(result.dims(), &[1, 2, 3, 4]);
}

#[test]
fn test_trace_rope() {
    use crate::layers::RotaryEmbedding;

    let rope = RotaryEmbedding::new(4, 16, 10000.0, &cpu()).unwrap();

    // [B=1, H=2, S=3, head_dim=4]
    let x = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 3, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        rope.apply(&x, 0)
    })
    .unwrap();

    // 1 input + 1 composite RotaryEmbedding = 2 nodes
    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::RotaryEmbedding {
            head_dim,
            offset,
            cos_cache,
            sin_cache,
        } => {
            assert_eq!(*head_dim, 4);
            assert_eq!(*offset, 0);
            // cos/sin narrowed from [16, 2] to [3, 2] (seq_len=3, half_dim=2)
            assert_eq!(cos_cache.shape(), &[3, 2]);
            assert_eq!(sin_cache.shape(), &[3, 2]);
            assert!(!cos_cache.data().is_empty(), "cos_cache should have data");
            assert!(!sin_cache.data().is_empty(), "sin_cache should have data");
        }
        other => panic!("expected RotaryEmbedding, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[1, 2, 3, 4]);
    assert_eq!(output.inputs().len(), 1);
    assert_eq!(result.dims(), &[1, 2, 3, 4]);
}

#[test]
fn test_trace_multi_head_attention() {
    use crate::layers::{Linear, Module};

    let dim = 8;
    let num_heads = 2;
    let head_dim = dim / num_heads; // 4

    // Create projection weights: [out_features, in_features]
    let make_proj = |seed: u64| -> Linear {
        let w_data: Vec<f32> = (0..dim * dim)
            .map(|i| ((i as f64 + seed as f64) * 0.01) as f32)
            .collect();
        let w = DynTensor::new(&w_data, &[dim, dim], &cpu()).unwrap();
        Linear::new(w, None).unwrap()
    };

    let mha = crate::layers::attention::MultiHeadAttention::new(
        make_proj(1), // q_proj
        make_proj(2), // k_proj
        make_proj(3), // v_proj
        make_proj(4), // out_proj
        num_heads,
        num_heads, // num_kv_heads = num_heads (standard MHA)
    )
    .unwrap();

    // [B=1, S=3, D=8]
    let x_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let x = DynTensor::new(&x_data, &[1, 3, 8], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 3, 8], DType::F32).unwrap();
        x.set_trace_id(id);
        Module::forward(&mha, &x)
    })
    .unwrap();

    // 1 input + 1 composite MultiHeadAttention = 2 nodes
    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::MultiHeadAttention {
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
        } => {
            assert_eq!(*nh, num_heads);
            assert_eq!(*nkv, num_heads);
            assert_eq!(*hd, head_dim);
        }
        other => panic!("expected MultiHeadAttention, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[1, 3, 8]);
    assert_eq!(output.inputs().len(), 1); // self-attention: single input
    assert_eq!(result.dims(), &[1, 3, 8]);
}
