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
            scale,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            out_weight,
            out_bias,
            rope,
            cross_attention,
            has_mask,
        } => {
            assert_eq!(*nh, num_heads);
            assert_eq!(*nkv, num_heads);
            assert_eq!(*hd, head_dim);
            // scale is the recorded 1/sqrt(head_dim) the kernel applied.
            assert!((*scale - 1.0 / (head_dim as f64).sqrt()).abs() < 1e-12);
            // All four projections captured with REAL data (not placeholders).
            for (w, seed) in [(q_weight, 1u64), (k_weight, 2), (v_weight, 3), (out_weight, 4)] {
                assert_eq!(w.shape(), &[dim, dim]);
                assert!(!w.is_placeholder());
                // Spot-check the captured data matches the seeded weights.
                assert!((w.data()[0] - (seed as f64 * 0.01) as f32).abs() < 1e-6);
            }
            assert!(q_bias.is_none());
            assert!(k_bias.is_none());
            assert!(v_bias.is_none());
            assert!(out_bias.is_none());
            // Module::forward = self-attention, no mask, no RoPE.
            assert!(rope.is_none());
            assert!(!*cross_attention);
            assert!(!*has_mask);
        }
        other => panic!("expected MultiHeadAttention, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[1, 3, 8]);
    assert_eq!(output.inputs().len(), 1); // self-attention: single input
    assert_eq!(result.dims(), &[1, 3, 8]);
}

/// Cross-attention + mask + RoPE: the capture that used to be silently
/// dropped. The recorded node must carry the mask as a REAL dataflow edge
/// (auto-registered `ConstantWeight`, last input), `cross_attention` /
/// `has_mask` set, and both RoPE window pairs narrowed to the lengths the
/// compute body actually rotated (S_q = 3 vs S_kv = 5 differ here).
#[test]
fn test_trace_mha_cross_attention_mask_rope_capture() {
    use crate::layers::{Linear, RotaryEmbedding};

    let dim = 8;
    let num_heads = 2;
    let head_dim = dim / num_heads; // 4

    let make_proj = |seed: u64| -> Linear {
        let w_data: Vec<f32> = (0..dim * dim)
            .map(|i| ((i as f64 + seed as f64) * 0.01) as f32)
            .collect();
        let w = DynTensor::new(&w_data, &[dim, dim], &cpu()).unwrap();
        Linear::new(w, None).unwrap()
    };

    let mha = crate::layers::attention::MultiHeadAttention::new(
        make_proj(1),
        make_proj(2),
        make_proj(3),
        make_proj(4),
        num_heads,
        num_heads,
    )
    .unwrap();
    let rope = RotaryEmbedding::new(head_dim, 16, 10000.0, &cpu()).unwrap();
    let rope_offset = 2;

    // Query [B=1, S_q=3, D=8], encoder K/V source [B=1, S_kv=5, D=8],
    // additive mask broadcastable to [B, H, S_q, S_kv] — created OUTSIDE the
    // trace scope so it has no trace id.
    let x = DynTensor::ones(&[1, 3, dim], DType::F32, &cpu()).unwrap();
    let enc = DynTensor::ones(&[1, 5, dim], DType::F32, &cpu()).unwrap();
    let mask = DynTensor::zeros(&[1, 1, 3, 5], DType::F32, &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id_x = record_input(&[1, 3, dim], DType::F32).unwrap();
        x.set_trace_id(id_x);
        let mut enc = enc.clone();
        let id_enc = record_input(&[1, 5, dim], DType::F32).unwrap();
        enc.set_trace_id(id_enc);
        mha.forward(&x, Some(&enc), Some(&mask), Some(&rope), rope_offset)
    })
    .unwrap();

    // 2 inputs + 1 auto-registered ConstantWeight (mask) + 1 MHA = 4 nodes.
    assert_eq!(graph.len(), 4);
    let output = graph.output_node().unwrap();
    // Inputs in fixed order: x, kv_input, mask (last).
    assert_eq!(output.inputs().len(), 3);
    let mask_node_id = output.inputs()[2];
    let mask_node = graph
        .nodes()
        .iter()
        .find(|n| n.id() == mask_node_id)
        .expect("mask input node present");
    match mask_node.op() {
        TraceOp::ConstantWeight { weight } => {
            assert_eq!(weight.shape(), &[1, 1, 3, 5]);
        }
        other => panic!("expected auto-registered ConstantWeight mask, got {other:?}"),
    }
    match output.op() {
        TraceOp::MultiHeadAttention {
            rope,
            cross_attention,
            has_mask,
            ..
        } => {
            assert!(*cross_attention, "kv_input given: cross_attention");
            assert!(*has_mask, "mask given: has_mask");
            let rope = rope.as_deref().expect("rope applied: capture present");
            assert_eq!(rope.head_dim, head_dim);
            assert_eq!(rope.offset, rope_offset);
            // Q rotated over S_q=3, K over S_kv=5, both at offset 2.
            assert_eq!(rope.q_cos_cache.shape(), &[3, head_dim / 2]);
            assert_eq!(rope.q_sin_cache.shape(), &[3, head_dim / 2]);
            assert_eq!(rope.k_cos_cache.shape(), &[5, head_dim / 2]);
            assert_eq!(rope.k_sin_cache.shape(), &[5, head_dim / 2]);
            for w in [
                &rope.q_cos_cache,
                &rope.q_sin_cache,
                &rope.k_cos_cache,
                &rope.k_sin_cache,
            ] {
                assert!(!w.data().is_empty(), "rope windows carry real data");
            }
        }
        other => panic!("expected MultiHeadAttention, got {other:?}"),
    }
    assert_eq!(result.dims(), &[1, 3, dim]);
}
