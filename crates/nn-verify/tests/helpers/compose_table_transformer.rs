// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep NY compose tests for Table Transformer (DETR) subgraphs.
//!
//! Verifies bounds propagation through intermediate-depth compositions bridging
//! the sub-block tests in `compose_dpdf_table_transformer.rs` and full e2e:
//!
//! 1. **Decoder self-attention** — object queries self-attend (IBP + CROWN)
//! 2. **Full DETR decoder layer** — self-attn + cross-attn + FFN (CROWN)
//! 3. **Two-layer encoder stack** — depth composition widening (IBP + CROWN)
//! 4. **Input projection + PE** — Conv2d(1x1) + sinusoidal PE (IBP)
//! 5. **Encoder-to-decoder pipeline** — encoder norm -> cross-attn (CROWN)
//! 6. **Full pipeline** — encoder -> decoder -> cls + box heads (IBP)
//!
//! Dimensions: HIDDEN_DIM=16, SEQ_LEN=4, NUM_HEADS=4 (small for fast verify).
//! All tests use IbpValidated soundness mode per nn engineering rules.
//!
//! Part of #3883: deep NY compose tests for Table Transformer.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

const HIDDEN_DIM: usize = 16;
const FFN_DIM: usize = 64;
const NUM_HEADS: usize = 4;
const NUM_QUERIES: usize = 4;
const ENC_SEQ_LEN: usize = 4;
const NUM_CLASSES: usize = 6;
const BACKBONE_CH: usize = 32;
const FEAT_SIZE: usize = 2;
const WEIGHT_MAG: f32 = 0.02;

fn w(shape: &[usize]) -> ArrayD<f32> { ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG) }
fn ones(shape: &[usize]) -> ArrayD<f32> { ArrayD::from_elem(IxDyn(shape), 1.0f32) }
fn zeros(shape: &[usize]) -> ArrayD<f32> { ArrayD::from_elem(IxDyn(shape), 0.0f32) }

/// Standard LayerNorm + attention + residual + LayerNorm + FFN(ReLU) + residual.
fn add_encoder_layer(b: &mut TensorBlockBuilder, input: TensorNodeId, pfx: &str) -> TensorNodeId {
    let d = HIDDEN_DIM;
    let seq_shape = [ENC_SEQ_LEN, d];
    let ffn_shape = [ENC_SEQ_LEN, FFN_DIM];

    let ln1_eps = b.add_input(&format!("{pfx}_ln1_eps"), &[1]);
    let ln1_w = b.add_input(&format!("{pfx}_ln1_w"), &[d]);
    let ln1_b = b.add_input(&format!("{pfx}_ln1_b"), &[d]);
    let qw = b.add_input(&format!("{pfx}_qw"), &[d, d]);
    let kw = b.add_input(&format!("{pfx}_kw"), &[d, d]);
    let vw = b.add_input(&format!("{pfx}_vw"), &[d, d]);
    let ow = b.add_input(&format!("{pfx}_ow"), &[d, d]);
    let ln2_eps = b.add_input(&format!("{pfx}_ln2_eps"), &[1]);
    let ln2_w = b.add_input(&format!("{pfx}_ln2_w"), &[d]);
    let ln2_b = b.add_input(&format!("{pfx}_ln2_b"), &[d]);
    let ffn_up = b.add_input(&format!("{pfx}_ffn_up"), &[FFN_DIM, d]);
    let ffn_dn = b.add_input(&format!("{pfx}_ffn_dn"), &[d, FFN_DIM]);

    let n1 = b.add_layer_norm(input, ln1_eps, 1, ln1_w, ln1_b, &seq_shape);
    let attn = b.add_multi_head_attention(
        n1, qw, kw, vw, ow, NUM_HEADS, AttentionMask::Standard, &seq_shape,
    ).expect("self-attention");
    let r1 = b.add_binary_add(input, attn, &seq_shape);
    let n2 = b.add_layer_norm(r1, ln2_eps, 1, ln2_w, ln2_b, &seq_shape);
    let h = b.add_linear(n2, ffn_up, None, &ffn_shape);
    let h = b.add_relu(h, &ffn_shape);
    let h = b.add_linear(h, ffn_dn, None, &seq_shape);
    b.add_binary_add(r1, h, &seq_shape)
}

/// Bindings for one encoder layer (12 constants).
fn encoder_layer_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    vec![
        TensorParamBinding::ConstantScalar(1e-5),              // ln1_eps
        TensorParamBinding::ConstantTensor(ones(&[d])),        // ln1_w
        TensorParamBinding::ConstantTensor(zeros(&[d])),       // ln1_b
        TensorParamBinding::ConstantTensor(w(&[d, d])),        // qw
        TensorParamBinding::ConstantTensor(w(&[d, d])),        // kw
        TensorParamBinding::ConstantTensor(w(&[d, d])),        // vw
        TensorParamBinding::ConstantTensor(w(&[d, d])),        // ow
        TensorParamBinding::ConstantScalar(1e-5),              // ln2_eps
        TensorParamBinding::ConstantTensor(ones(&[d])),        // ln2_w
        TensorParamBinding::ConstantTensor(zeros(&[d])),       // ln2_b
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, d])),  // ffn_up
        TensorParamBinding::ConstantTensor(w(&[d, FFN_DIM])),  // ffn_dn
    ]
}

// 1. Decoder self-attention: LN -> self-attn -> residual
fn build_decoder_self_attn() -> TensorKernelDef {
    let d = HIDDEN_DIM;
    let s = [NUM_QUERIES, d];
    let mut b = TensorBlockBuilder::new("tt_deep_decoder_self_attn");
    let inp = b.add_input("queries", &s);
    let eps = b.add_input("ln_eps", &[1]);
    let lw = b.add_input("ln_w", &[d]);
    let lb = b.add_input("ln_b", &[d]);
    let qw = b.add_input("qw", &[d, d]);
    let kw = b.add_input("kw", &[d, d]);
    let vw = b.add_input("vw", &[d, d]);
    let ow = b.add_input("ow", &[d, d]);
    let n = b.add_layer_norm(inp, eps, 1, lw, lb, &s);
    let a = b.add_multi_head_attention(
        n, qw, kw, vw, ow, NUM_HEADS, AttentionMask::Standard, &s,
    ).expect("self-attn");
    let out = b.add_binary_add(inp, a, &s);
    b.build(out).expect("decoder self-attn kernel")
}

fn decoder_self_attn_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[d])),
        TensorParamBinding::ConstantTensor(zeros(&[d])),
        TensorParamBinding::ConstantTensor(w(&[d, d])),
        TensorParamBinding::ConstantTensor(w(&[d, d])),
        TensorParamBinding::ConstantTensor(w(&[d, d])),
        TensorParamBinding::ConstantTensor(w(&[d, d])),
    ]
}

#[test]
fn test_decoder_self_attn_ibp() {
    let g = tensor_kernel_to_graph(&build_decoder_self_attn(), &decoder_self_attn_bindings())
        .expect("graph");
    let out = g.propagate_ibp(&uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0)).expect("IBP");
    assert_eq!(out.lower_upper().0.shape(), &[NUM_QUERIES, HIDDEN_DIM]);
    assert_bounds_valid(&out);
    let (lo, hi) = bounds_min_max(&out);
    eprintln!("TT deep decoder self-attn IBP: [{lo}, {hi}]");
}

#[test]
fn test_decoder_self_attn_crown() {
    let g = tensor_kernel_to_graph(&build_decoder_self_attn(), &decoder_self_attn_bindings())
        .expect("graph");
    let inp = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);
    let (m, out, fb) = assert_crown_tighter_when_not_fallback(&g, &inp);
    let (lo, hi) = bounds_min_max(&out);
    eprintln!("TT deep decoder self-attn CROWN: method={m:?}, [{lo}, {hi}]");
    if let Some(r) = &fb { eprintln!("Fallback: {r}"); }
}

#[test]
fn test_decoder_self_attn_verify() {
    let r = verify_and_assert(
        &build_decoder_self_attn(), &decoder_self_attn_bindings(),
        &uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0), "tt_deep_decoder_self_attn",
    );
    assert_eq!(r.num_variables, 1);
}

// 2. Full DETR decoder layer: self-attn + cross-attn + FFN
fn build_full_decoder_layer() -> TensorKernelDef {
    let d = HIDDEN_DIM;
    let qs = [NUM_QUERIES, d];
    let fs = [NUM_QUERIES, FFN_DIM];
    let mut b = TensorBlockBuilder::new("tt_deep_full_decoder_layer");

    let inp = b.add_input("queries", &qs);
    let mem = b.add_input("memory", &[ENC_SEQ_LEN, d]);

    // Self-attention
    let e1 = b.add_input("l1e", &[1]);
    let w1 = b.add_input("l1w", &[d]); let b1 = b.add_input("l1b", &[d]);
    let sqw = b.add_input("sqw", &[d, d]); let skw = b.add_input("skw", &[d, d]);
    let svw = b.add_input("svw", &[d, d]); let sow = b.add_input("sow", &[d, d]);
    let n1 = b.add_layer_norm(inp, e1, 1, w1, b1, &qs);
    let sa = b.add_multi_head_attention(
        n1, sqw, skw, svw, sow, NUM_HEADS, AttentionMask::Standard, &qs,
    ).expect("self-attn");
    let r1 = b.add_binary_add(inp, sa, &qs);

    // Cross-attention
    let e2 = b.add_input("l2e", &[1]);
    let w2 = b.add_input("l2w", &[d]); let b2 = b.add_input("l2b", &[d]);
    let cqw = b.add_input("cqw", &[d, d]); let ckw = b.add_input("ckw", &[d, d]);
    let cvw = b.add_input("cvw", &[d, d]); let cow = b.add_input("cow", &[d, d]);
    let n2 = b.add_layer_norm(r1, e2, 1, w2, b2, &qs);
    let ca = b.add_multi_head_cross_attention(
        n2, mem, cqw, ckw, cvw, cow, NUM_HEADS, AttentionMask::Standard, &qs,
    ).expect("cross-attn");
    let r2 = b.add_binary_add(r1, ca, &qs);

    // FFN
    let e3 = b.add_input("l3e", &[1]);
    let w3 = b.add_input("l3w", &[d]); let b3 = b.add_input("l3b", &[d]);
    let fu = b.add_input("fu", &[FFN_DIM, d]);
    let fd = b.add_input("fd", &[d, FFN_DIM]);
    let n3 = b.add_layer_norm(r2, e3, 1, w3, b3, &qs);
    let h = b.add_linear(n3, fu, None, &fs);
    let h = b.add_relu(h, &fs);
    let h = b.add_linear(h, fd, None, &qs);
    let out = b.add_binary_add(r2, h, &qs);
    b.build(out).expect("full decoder layer")
}

fn full_decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let mem = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, d]), 0.1f32);
    let mut v = vec![TensorParamBinding::Variable, TensorParamBinding::ConstantTensor(mem)];
    // 3 sub-blocks x (eps + ln_w + ln_b + 4 attn weights or 2 ffn weights)
    for _ in 0..3 {
        v.push(TensorParamBinding::ConstantScalar(1e-5));
        v.push(TensorParamBinding::ConstantTensor(ones(&[d])));
        v.push(TensorParamBinding::ConstantTensor(zeros(&[d])));
    }
    // Self-attn: 4 weights
    for _ in 0..4 { v.insert(5, TensorParamBinding::ConstantTensor(w(&[d, d]))); }
    // Cross-attn: 4 weights
    for _ in 0..4 { v.insert(12, TensorParamBinding::ConstantTensor(w(&[d, d]))); }
    // FFN: up + down
    v.push(TensorParamBinding::ConstantTensor(w(&[FFN_DIM, d])));
    v.push(TensorParamBinding::ConstantTensor(w(&[d, FFN_DIM])));
    v
}

#[test]
fn test_full_decoder_layer_crown() {
    let g = tensor_kernel_to_graph(&build_full_decoder_layer(), &full_decoder_layer_bindings())
        .expect("graph");
    let inp = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);
    let (m, out, fb) = assert_crown_tighter_when_not_fallback(&g, &inp);
    assert_eq!(out.lower_upper().0.shape(), &[NUM_QUERIES, HIDDEN_DIM]);
    let (lo, hi) = bounds_min_max(&out);
    eprintln!("TT deep full decoder layer: method={m:?}, [{lo}, {hi}]");
    if let Some(r) = &fb { eprintln!("Fallback: {r}"); }
}

#[test]
fn test_full_decoder_layer_verify() {
    let r = verify_and_assert(
        &build_full_decoder_layer(), &full_decoder_layer_bindings(),
        &uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0), "tt_deep_full_decoder_layer",
    );
    assert_eq!(r.num_variables, 1);
}

// 3. Two-layer encoder stack
fn build_two_layer_encoder() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("tt_deep_two_layer_encoder");
    let inp = b.add_input("features", &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let l1 = add_encoder_layer(&mut b, inp, "e0");
    let l2 = add_encoder_layer(&mut b, l1, "e1");
    b.build(l2).expect("two-layer encoder")
}

fn two_layer_encoder_bindings() -> Vec<TensorParamBinding> {
    let mut v = vec![TensorParamBinding::Variable];
    v.extend(encoder_layer_bindings());
    v.extend(encoder_layer_bindings());
    v
}

#[test]
fn test_two_layer_encoder_ibp() {
    let g = tensor_kernel_to_graph(&build_two_layer_encoder(), &two_layer_encoder_bindings())
        .expect("graph");
    let out = g.propagate_ibp(&uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0)).expect("IBP");
    assert_eq!(out.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&out);
    let (lo, hi) = bounds_min_max(&out);
    eprintln!("TT deep two-layer encoder IBP: [{lo}, {hi}]");
}

#[test]
fn test_two_layer_encoder_crown() {
    let g = tensor_kernel_to_graph(&build_two_layer_encoder(), &two_layer_encoder_bindings())
        .expect("graph");
    let inp = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);
    let (m, _, fb) = assert_crown_tighter_when_not_fallback(&g, &inp);
    eprintln!("TT deep two-layer encoder CROWN: method={m:?}");
    if let Some(r) = &fb { eprintln!("Fallback: {r}"); }
}

#[test]
fn test_two_layer_encoder_verify() {
    let r = verify_and_assert(
        &build_two_layer_encoder(), &two_layer_encoder_bindings(),
        &uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0), "tt_deep_two_layer_encoder",
    );
    assert_eq!(r.num_variables, 1);
}

// 4. Input projection + positional encoding
fn build_input_proj_pe() -> TensorKernelDef {
    let d = HIDDEN_DIM;
    let s = FEAT_SIZE;
    let seq = s * s;
    let mut b = TensorBlockBuilder::new("tt_deep_input_proj_pe");
    let inp = b.add_input("backbone", &[BACKBONE_CH, s, s]);
    let pw = b.add_input("pw", &[d, BACKBONE_CH, 1, 1]);
    let pb = b.add_input("pb", &[d]);
    let proj = b.add_conv2d(inp, pw, Some(pb), 1, 1, 0, 0, &[d, s, s]);
    let flat = b.add_reshape(proj, &[d, seq]);
    let tr = b.add_transpose(flat, &[1, 0], &[seq, d]);
    let pe = b.add_input("pe", &[seq, d]);
    let out = b.add_binary_add(tr, pe, &[seq, d]);
    b.build(out).expect("input proj + PE")
}

fn sinusoidal_pe() -> ArrayD<f32> {
    let (seq, d) = (ENC_SEQ_LEN, HIDDEN_DIM);
    let mut data = vec![0.0f32; seq * d];
    for t in 0..seq {
        for i in 0..d / 2 {
            let f = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / d as f64);
            data[t * d + 2 * i] = f.sin() as f32;
            data[t * d + 2 * i + 1] = f.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq, d]), data).expect("PE")
}

fn input_proj_pe_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, BACKBONE_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(sinusoidal_pe()),
    ]
}

#[test]
fn test_input_proj_pe_ibp() {
    let g = tensor_kernel_to_graph(&build_input_proj_pe(), &input_proj_pe_bindings())
        .expect("graph");
    let out = g.propagate_ibp(&uniform_bounds(&[BACKBONE_CH, FEAT_SIZE, FEAT_SIZE], 2.0))
        .expect("IBP");
    assert_eq!(out.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&out);
    let (lo, hi) = bounds_min_max(&out);
    eprintln!("TT deep input proj+PE IBP: [{lo}, {hi}]");
}

#[test]
fn test_input_proj_pe_verify() {
    let r = verify_and_assert(
        &build_input_proj_pe(), &input_proj_pe_bindings(),
        &uniform_bounds(&[BACKBONE_CH, FEAT_SIZE, FEAT_SIZE], 2.0), "tt_deep_input_proj_pe",
    );
    assert_eq!(r.num_variables, 1);
}

// 5. Encoder-to-decoder: encoder norm -> cross-attention

fn build_enc_to_dec() -> TensorKernelDef {
    let d = HIDDEN_DIM;
    let es = [ENC_SEQ_LEN, d];
    let qs = [NUM_QUERIES, d];
    let mut b = TensorBlockBuilder::new("tt_deep_encoder_to_decoder");
    let inp = b.add_input("enc_out", &es);
    let eps = b.add_input("eps", &[1]);
    let lw = b.add_input("lw", &[d]);
    let lb = b.add_input("lb", &[d]);
    let mem = b.add_layer_norm(inp, eps, 1, lw, lb, &es);
    let q = b.add_input("queries", &qs);
    let cqw = b.add_input("cqw", &[d, d]); let ckw = b.add_input("ckw", &[d, d]);
    let cvw = b.add_input("cvw", &[d, d]); let cow = b.add_input("cow", &[d, d]);
    let out = b.add_multi_head_cross_attention(
        q, mem, cqw, ckw, cvw, cow, NUM_HEADS, AttentionMask::Standard, &qs,
    ).expect("cross-attn");
    b.build(out).expect("enc-to-dec")
}

fn enc_to_dec_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let qi = ArrayD::from_elem(IxDyn(&[NUM_QUERIES, d]), 0.01f32);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[d])),
        TensorParamBinding::ConstantTensor(zeros(&[d])),
        TensorParamBinding::ConstantTensor(qi),
        TensorParamBinding::ConstantTensor(w(&[d, d])),
        TensorParamBinding::ConstantTensor(w(&[d, d])),
        TensorParamBinding::ConstantTensor(w(&[d, d])),
        TensorParamBinding::ConstantTensor(w(&[d, d])),
    ]
}

#[test]
fn test_enc_to_dec_crown() {
    let g = tensor_kernel_to_graph(&build_enc_to_dec(), &enc_to_dec_bindings()).expect("graph");
    let inp = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);
    let (m, out, fb) = assert_crown_tighter_when_not_fallback(&g, &inp);
    assert_eq!(out.lower_upper().0.shape(), &[NUM_QUERIES, HIDDEN_DIM]);
    let (lo, hi) = bounds_min_max(&out);
    eprintln!("TT deep enc-to-dec CROWN: method={m:?}, [{lo}, {hi}]");
    if let Some(r) = &fb { eprintln!("Fallback: {r}"); }
}

#[test]
fn test_enc_to_dec_verify() {
    let r = verify_and_assert(
        &build_enc_to_dec(), &enc_to_dec_bindings(),
        &uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0), "tt_deep_encoder_to_decoder",
    );
    assert_eq!(r.num_variables, 1);
}

// 6. Full pipeline: encoder -> decoder -> cls + box heads

fn build_full_pipeline() -> TensorKernelDef {
    let d = HIDDEN_DIM;
    let es = [ENC_SEQ_LEN, d];
    let qs = [NUM_QUERIES, d];
    let total = NUM_CLASSES + 4;
    let mut b = TensorBlockBuilder::new("tt_deep_full_pipeline");

    let inp = b.add_input("features", &es);
    let enc = add_encoder_layer(&mut b, inp, "enc");

    // Encoder norm
    let ne = b.add_input("ne", &[1]);
    let nw = b.add_input("nw", &[d]);
    let nb = b.add_input("nb", &[d]);
    let mem = b.add_layer_norm(enc, ne, 1, nw, nb, &es);

    // Cross-attention
    let q = b.add_input("q", &qs);
    let cqw = b.add_input("cqw", &[d, d]); let ckw = b.add_input("ckw", &[d, d]);
    let cvw = b.add_input("cvw", &[d, d]); let cow = b.add_input("cow", &[d, d]);
    let dec = b.add_multi_head_cross_attention(
        q, mem, cqw, ckw, cvw, cow, NUM_HEADS, AttentionMask::Standard, &qs,
    ).expect("cross-attn");

    // Cls head: Linear -> sigmoid
    let clw = b.add_input("clw", &[NUM_CLASSES, d]);
    let clb = b.add_input("clb", &[NUM_CLASSES]);
    let cl = b.add_linear(dec, clw, Some(clb), &[NUM_QUERIES, NUM_CLASSES]);
    let cls = b.add_sigmoid(cl, &[NUM_QUERIES, NUM_CLASSES]);
    // Box head: Linear -> sigmoid
    let bxw = b.add_input("bxw", &[4, d]);
    let bxb = b.add_input("bxb", &[4]);
    let bl = b.add_linear(dec, bxw, Some(bxb), &[NUM_QUERIES, 4]);
    let bx = b.add_sigmoid(bl, &[NUM_QUERIES, 4]);

    let out = b.add_concat(&[cls, bx], 1, &[NUM_QUERIES, total]);
    b.build(out).expect("full pipeline")
}

fn full_pipeline_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let qi = ArrayD::from_elem(IxDyn(&[NUM_QUERIES, d]), 0.01f32);
    let mut v = vec![TensorParamBinding::Variable];
    v.extend(encoder_layer_bindings());
    // Encoder norm
    v.extend([
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[d])),
        TensorParamBinding::ConstantTensor(zeros(&[d])),
    ]);
    // Cross-attention
    v.push(TensorParamBinding::ConstantTensor(qi));
    for _ in 0..4 { v.push(TensorParamBinding::ConstantTensor(w(&[d, d]))); }
    // Cls head
    v.push(TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, d])));
    v.push(TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])));
    // Box head
    v.push(TensorParamBinding::ConstantTensor(w(&[4, d])));
    v.push(TensorParamBinding::ConstantTensor(zeros(&[4])));
    v
}

/// All outputs go through sigmoid => bounded in [0, 1].
#[test]
fn test_full_pipeline_ibp() {
    let g = tensor_kernel_to_graph(&build_full_pipeline(), &full_pipeline_bindings())
        .expect("graph");
    let out = g.propagate_ibp(&uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0)).expect("IBP");
    let total = NUM_CLASSES + 4;
    assert_eq!(out.lower_upper().0.shape(), &[NUM_QUERIES, total]);
    assert_bounds_valid(&out);
    let (lo, hi) = bounds_min_max(&out);
    eprintln!("TT deep full pipeline IBP: [{lo}, {hi}]");
    let eps = 1e-6;
    assert!(lo >= 0.0 - eps, "sigmoid lo >= 0, got {lo}");
    assert!(hi <= 1.0 + eps, "sigmoid hi <= 1, got {hi}");
}

#[test]
fn test_full_pipeline_verify() {
    let r = verify_and_assert(
        &build_full_pipeline(), &full_pipeline_bindings(),
        &uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0), "tt_deep_full_pipeline",
    );
    assert_eq!(r.num_variables, 1);
    let (lo, _) = r.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, NUM_CLASSES + 4]);
}
