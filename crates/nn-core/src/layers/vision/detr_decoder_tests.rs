// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::vision::detr_decoder::{DetrDecoder, DetrDecoderLayer};
use crate::layers::{LayerNorm, Linear, MultiHeadAttention};
use crate::{DType, Device};

fn make_linear(in_f: usize, out_f: usize) -> Linear {
    let weight = DynTensor::full(&[out_f, in_f], 0.01, DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[out_f], DType::F32, &Device::Cpu).unwrap();
    Linear::new(weight, Some(bias)).unwrap()
}

fn make_layer_norm(dim: usize) -> LayerNorm {
    let weight = DynTensor::ones(&[dim], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[dim], DType::F32, &Device::Cpu).unwrap();
    LayerNorm::new(weight, bias, 1e-5).unwrap()
}

fn make_mha(dim: usize, num_heads: usize) -> MultiHeadAttention {
    let q_proj = make_linear(dim, dim);
    let k_proj = make_linear(dim, dim);
    let v_proj = make_linear(dim, dim);
    let out_proj = make_linear(dim, dim);
    MultiHeadAttention::new(q_proj, k_proj, v_proj, out_proj, num_heads, num_heads).unwrap()
}

fn make_decoder_layer(dim: usize, num_heads: usize, ffn_dim: usize) -> DetrDecoderLayer {
    DetrDecoderLayer::new(
        make_mha(dim, num_heads),
        make_mha(dim, num_heads),
        make_layer_norm(dim),
        make_layer_norm(dim),
        make_layer_norm(dim),
        make_linear(dim, ffn_dim),
        make_linear(ffn_dim, dim),
    )
}

fn make_decoder(
    dim: usize,
    num_heads: usize,
    ffn_dim: usize,
    num_layers: usize,
    num_queries: usize,
    num_classes: usize,
) -> DetrDecoder {
    let query_embed = DynTensor::full(&[num_queries, dim], 0.01, DType::F32, &Device::Cpu).unwrap();
    let layers = (0..num_layers)
        .map(|_| make_decoder_layer(dim, num_heads, ffn_dim))
        .collect();
    let final_norm = make_layer_norm(dim);
    let class_head = make_linear(dim, num_classes + 1);
    let bbox_head = make_linear(dim, 4);
    DetrDecoder::new(query_embed, layers, final_norm, class_head, bbox_head).unwrap()
}

#[test]
fn test_detr_decoder_output_shapes() {
    let dim = 64;
    let num_queries = 10;
    let num_classes = 6;
    let decoder = make_decoder(dim, 4, 128, 2, num_queries, num_classes);

    // Simulate encoder output: [B, H*W, D]
    let memory = DynTensor::full(&[1, 25, 64], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = decoder.forward_decode(&memory, None).unwrap();

    assert_eq!(output.class_logits.dims(), &[1, 10, 7]); // num_classes + 1
    assert_eq!(output.bbox_preds.dims(), &[1, 10, 4]);
}

#[test]
fn test_detr_decoder_batch() {
    let decoder = make_decoder(32, 4, 64, 1, 5, 3);
    let memory = DynTensor::full(&[4, 16, 32], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = decoder.forward_decode(&memory, None).unwrap();

    assert_eq!(output.class_logits.dims(), &[4, 5, 4]); // 3 classes + 1
    assert_eq!(output.bbox_preds.dims(), &[4, 5, 4]);
}

#[test]
fn test_detr_bbox_sigmoid_range() {
    let decoder = make_decoder(32, 4, 64, 1, 5, 3);
    let memory = DynTensor::full(&[1, 9, 32], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = decoder.forward_decode(&memory, None).unwrap();

    // After sigmoid, bbox values should be in (0, 1)
    let vals = output.bbox_preds.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(v > 0.0 && v < 1.0, "bbox value {v} not in (0,1)");
    }
}

#[test]
fn test_detr_decoder_layer_forward() {
    let dim = 32;
    let layer = make_decoder_layer(dim, 4, 64);
    let tgt = DynTensor::full(&[1, 5, 32], 0.5, DType::F32, &Device::Cpu).unwrap();
    let memory = DynTensor::full(&[1, 16, 32], 0.5, DType::F32, &Device::Cpu).unwrap();
    let out = layer.forward_layer(&tgt, &memory, None).unwrap();
    assert_eq!(out.dims(), &[1, 5, 32]);
}
