// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for training loop pipeline bound propagation.
//!
//! Verifies NY IBP and CROWN bound propagation through the key stages
//! of a training loop: forward pass, loss computation, gradient flow, optimizer
//! step, weight update preservation, multi-step stability, mixed precision,
//! batch normalization, and learning rate schedules.
//!
//! ## Tests (14 tests)
//!
//!  1. Forward pass linear chain (IBP)
//!  2. MSE loss bounds (IBP)
//!  3. Cross-entropy loss bounds (IBP)
//!  4. Gradient flow through linear layers (IBP + CROWN)
//!  5. Optimizer step LR scaling (IBP)
//!  6. Weight update bound preservation (IBP)
//!  7. Multi-step training stability (IBP + CROWN)
//!  8. Mixed precision training bounds (IBP)
//!  9. Batch normalization running stats (IBP)
//! 10. LR warmup schedule (IBP)
//! 11. LR cosine decay (IBP)
//! 12. Gradient clipping (IBP)
//! 13. Full training step E2E (IBP + CROWN)
//! 14. Verify-and-record pipeline (IBP)
//!
//! Part of #4219: Compose tests for training loop pipeline bound verification.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

const HIDDEN: usize = 8;
const FFN: usize = 16;
const SEQ: usize = 4;
const CLS: usize = 4;
const W_MAG: f32 = 0.02;
const LR: f32 = 1e-3;

fn w(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), W_MAG))
}
fn zeros(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}
fn const_tensor(shape: &[usize], val: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), val))
}

// 1. Forward pass linear chain with activations (IBP)

#[test]
fn test_training_forward_pass_linear_chain_ibp() {
    let mut b = TensorBlockBuilder::new("training_forward_chain");
    let input = b.add_input("input", &[SEQ, HIDDEN]);
    let w1 = b.add_input("w1", &[FFN, HIDDEN]);
    let b1 = b.add_input("b1", &[FFN]);
    let h = b.add_linear(input, w1, Some(b1), &[SEQ, FFN]);
    let h_act = b.add_relu(h, &[SEQ, FFN]);
    let w2 = b.add_input("w2", &[CLS, FFN]);
    let b2 = b.add_input("b2", &[CLS]);
    let out = b.add_linear(h_act, w2, Some(b2), &[SEQ, CLS]);
    let def = b.build(out).expect("valid forward chain");

    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[FFN, HIDDEN]),
        zeros(&[FFN]),
        w(&[CLS, FFN]),
        zeros(&[CLS]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph
        .propagate_ibp(&uniform_bounds(&[SEQ, HIDDEN], 1.0))
        .expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, CLS]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Training forward chain IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// 2. MSE loss computation bounds (IBP)

#[test]
fn test_training_mse_loss_computation_ibp() {
    let shape = [SEQ, CLS];
    let mut b = TensorBlockBuilder::new("training_mse_loss");
    let pred = b.add_input("pred", &shape);
    let target = b.add_input("target", &shape);
    let neg_ones_node = b.add_input("neg", &shape);
    let neg_target = b.add_binary_mul(target, neg_ones_node, &shape);
    let diff = b.add_binary_add(pred, neg_target, &shape);
    let out = b.add_binary_mul(diff, diff, &shape);
    let def = b.build(out).expect("valid MSE loss");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&shape, 0.25),
        const_tensor(&shape, -1.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph
        .propagate_ibp(&uniform_bounds(&shape, 1.0))
        .expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Training MSE loss IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
    assert!(hi < 1e6, "MSE upper bound reasonable, got {hi}");
}

// 3. Cross-entropy style loss bounds (IBP)

#[test]
fn test_training_cross_entropy_loss_ibp() {
    let shape = [SEQ, CLS];
    let mut b = TensorBlockBuilder::new("training_ce_loss");
    let logits = b.add_input("logits", &shape);
    let log_probs = b.add_log_softmax(logits, 1, &shape);
    let target = b.add_input("target", &shape);
    let weighted = b.add_binary_mul(log_probs, target, &shape);
    let neg = b.add_input("neg", &shape);
    let out = b.add_binary_mul(weighted, neg, &shape);
    let def = b.build(out).expect("valid CE loss");

    let mut target_data = vec![0.0f32; SEQ * CLS];
    for s in 0..SEQ {
        target_data[s * CLS + (s % CLS)] = 1.0;
    }
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&shape), target_data).unwrap(),
        ),
        const_tensor(&shape, -1.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph
        .propagate_ibp(&uniform_bounds(&shape, 2.0))
        .expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Training CE loss IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// 4. Gradient flow through linear layers (IBP + CROWN)

#[test]
fn test_training_gradient_flow_linear_ibp_crown() {
    let mut b = TensorBlockBuilder::new("training_grad_flow");
    let grad_out = b.add_input("grad_out", &[SEQ, CLS]);
    let wt1 = b.add_input("wt1", &[HIDDEN, CLS]);
    let g1 = b.add_linear(grad_out, wt1, None, &[SEQ, HIDDEN]);
    let wt2 = b.add_input("wt2", &[FFN, HIDDEN]);
    let out = b.add_linear(g1, wt2, None, &[SEQ, FFN]);
    let def = b.build(out).expect("valid grad flow");

    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[HIDDEN, CLS]),
        w(&[FFN, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, CLS], 1.0);

    let ibp = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp);
    let (lo, hi) = bounds_min_max(&ibp);
    eprintln!("Training grad flow IBP: [{lo:.6}, {hi:.6}]");

    let (method, crown, fb) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown);
    eprintln!("Training grad flow CROWN: method={method:?}, [{clo:.6}, {chi:.6}]");
    if let Some(r) = &fb {
        eprintln!("Fallback: {r}");
    }
}

// 5. Optimizer step learning rate scaling (IBP)

#[test]
fn test_training_optimizer_lr_scaling_ibp() {
    let shape = [HIDDEN, FFN];
    let mut b = TensorBlockBuilder::new("training_opt_lr");
    let weight_p = b.add_input("weight", &shape);
    let gradient = b.add_input("gradient", &shape);
    let lr_s = b.add_input("neg_lr", &shape);
    let scaled = b.add_binary_mul(gradient, lr_s, &shape);
    let out = b.add_binary_add(weight_p, scaled, &shape);
    let def = b.build(out).expect("valid opt step");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&shape, 0.5),
        const_tensor(&shape, -LR),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph
        .propagate_ibp(&uniform_bounds(&shape, 0.1))
        .expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Training opt LR step IBP: [{lo:.6}, {hi:.6}]");
    assert!(hi - lo < 2.0, "opt step width bounded, got {}", hi - lo);
}

// 6. Weight update bound preservation (IBP)

#[test]
fn test_training_weight_update_preservation_ibp() {
    let shape = [HIDDEN, FFN];
    let mut b = TensorBlockBuilder::new("training_weight_update");
    let weight_p = b.add_input("weight", &shape);
    let gradient = b.add_input("gradient", &shape);
    let lr_s = b.add_input("neg_lr", &shape);
    let scaled = b.add_binary_mul(gradient, lr_s, &shape);
    let updated = b.add_binary_add(weight_p, scaled, &shape);
    let out = b.add_sigmoid(updated, &shape);
    let def = b.build(out).expect("valid weight update");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&shape, 0.1),
        const_tensor(&shape, -LR),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph
        .propagate_ibp(&uniform_bounds(&shape, 0.5))
        .expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Training weight update preservation IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo >= -1e-4, "sigmoid lo >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-4, "sigmoid hi <= 1, got {hi}");
}

// 7. Multi-step training stability (IBP + CROWN)

#[test]
fn test_training_multi_step_stability_ibp_crown() {
    let shape = [SEQ, HIDDEN];
    let mut b = TensorBlockBuilder::new("training_multi_step");
    let input = b.add_input("input", &shape);

    // Step 1: forward + backward proxy
    let w1 = b.add_input("s1w", &[HIDDEN, HIDDEN]);
    let f1 = b.add_linear(input, w1, None, &shape);
    let a1 = b.add_relu(f1, &shape);
    let w1t = b.add_input("s1wt", &[HIDDEN, HIDDEN]);
    let g1 = b.add_linear(a1, w1t, None, &shape);

    // Step 2: forward + backward proxy
    let w2 = b.add_input("s2w", &[HIDDEN, HIDDEN]);
    let f2 = b.add_linear(g1, w2, None, &shape);
    let a2 = b.add_relu(f2, &shape);
    let w2t = b.add_input("s2wt", &[HIDDEN, HIDDEN]);
    let out = b.add_linear(a2, w2t, None, &shape);
    let def = b.build(out).expect("valid multi-step");

    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[HIDDEN, HIDDEN]),
        w(&[HIDDEN, HIDDEN]),
        w(&[HIDDEN, HIDDEN]),
        w(&[HIDDEN, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let ib = uniform_bounds(&shape, 1.0);

    let ibp = graph.propagate_ibp(&ib).expect("IBP");
    assert_bounds_valid(&ibp);
    let (lo, hi) = bounds_min_max(&ibp);
    eprintln!("Training multi-step IBP: [{lo:.6}, {hi:.6}]");

    let (method, crown, fb) = assert_crown_tighter_when_not_fallback(&graph, &ib);
    let (clo, chi) = bounds_min_max(&crown);
    eprintln!("Training multi-step CROWN: method={method:?}, [{clo:.6}, {chi:.6}]");
    if let Some(r) = &fb {
        eprintln!("Fallback: {r}");
    }
}

// 8. Mixed precision training bounds (IBP)

#[test]
fn test_training_mixed_precision_bounds_ibp() {
    let shape = [SEQ, HIDDEN];
    let bf16_mag: f32 = 0.01;
    let mut b = TensorBlockBuilder::new("training_mixed_prec");
    let input = b.add_input("input", &shape);
    let w1 = b.add_input("bf16_w1", &[FFN, HIDDEN]);
    let b1 = b.add_input("bf16_b1", &[FFN]);
    let h = b.add_linear(input, w1, Some(b1), &[SEQ, FFN]);
    let h_act = b.add_relu(h, &[SEQ, FFN]);
    let w2 = b.add_input("fp32_w2", &[HIDDEN, FFN]);
    let b2 = b.add_input("fp32_b2", &[HIDDEN]);
    let out = b.add_linear(h_act, w2, Some(b2), &shape);
    let def = b.build(out).expect("valid mixed prec");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[FFN, HIDDEN], bf16_mag),
        zeros(&[FFN]),
        w(&[HIDDEN, FFN]),
        zeros(&[HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph
        .propagate_ibp(&uniform_bounds(&shape, 1.0))
        .expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Training mixed prec IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// 9. Batch normalization running stats update (IBP)

#[test]
fn test_training_batchnorm_running_stats_update_ibp() {
    let ch = HIDDEN;
    let shape = [ch, SEQ, SEQ];
    let mut b = TensorBlockBuilder::new("training_bn_stats");
    let input = b.add_input("features", &shape);
    let bn_mean = b.add_input("bn_mean", &[ch]);
    let bn_var = b.add_input("bn_var", &[ch]);
    let bn_w = b.add_input("bn_w", &[ch]);
    let bn_b = b.add_input("bn_b", &[ch]);
    let eps = b.add_input("eps", &[1]);
    let normed = b.add_batch_norm(input, bn_mean, bn_var, bn_w, bn_b, eps, &shape);
    let out = b.add_relu(normed, &shape);
    let def = b.build(out).expect("valid BN stats");

    let bindings = vec![
        TensorParamBinding::Variable,
        zeros(&[ch]),
        ones(&[ch]),
        ones(&[ch]),
        zeros(&[ch]),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph
        .propagate_ibp(&uniform_bounds(&shape, 1.0))
        .expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Training BN stats IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo >= -1e-4, "ReLU lo >= 0, got {lo}");
}

// 10. Learning rate warmup schedule bounds (IBP)

#[test]
fn test_training_lr_warmup_schedule_ibp() {
    let shape = [HIDDEN, HIDDEN];
    let warmup_lr: f32 = LR * 0.1;
    let mut b = TensorBlockBuilder::new("training_lr_warmup");
    let weight_p = b.add_input("weight", &shape);
    let gradient = b.add_input("gradient", &shape);
    let ws = b.add_input("warmup_lr", &shape);
    let scaled = b.add_binary_mul(gradient, ws, &shape);
    let out = b.add_binary_add(weight_p, scaled, &shape);
    let def = b.build(out).expect("valid warmup");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&shape, 0.5),
        const_tensor(&shape, -warmup_lr),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph
        .propagate_ibp(&uniform_bounds(&shape, 0.1))
        .expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Training LR warmup IBP: [{lo:.6}, {hi:.6}]");
    assert!(hi - lo < 1.0, "warmup width small, got {}", hi - lo);
}

// 11. Learning rate cosine decay bounds (IBP)

#[test]
fn test_training_lr_cosine_decay_ibp() {
    let shape = [HIDDEN, HIDDEN];
    let mid_lr = (1e-3_f32 + 1e-5) / 2.0;
    let mut b = TensorBlockBuilder::new("training_lr_cosine");
    let weight_p = b.add_input("weight", &shape);
    let gradient = b.add_input("gradient", &shape);
    let cs = b.add_input("cos_lr", &shape);
    let scaled = b.add_binary_mul(gradient, cs, &shape);
    let out = b.add_binary_add(weight_p, scaled, &shape);
    let def = b.build(out).expect("valid cosine decay");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&shape, 0.3),
        const_tensor(&shape, -mid_lr),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph
        .propagate_ibp(&uniform_bounds(&shape, 0.1))
        .expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Training LR cosine decay IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// 12. Gradient clipping bound preservation (IBP)

#[test]
fn test_training_gradient_clipping_ibp() {
    let shape = [HIDDEN, FFN];
    let mut b = TensorBlockBuilder::new("training_grad_clip");
    let weight_p = b.add_input("weight", &shape);
    let gradient = b.add_input("gradient", &shape);
    let clipped = b.add_tanh(gradient, &shape);
    let lr_s = b.add_input("neg_lr", &shape);
    let scaled = b.add_binary_mul(clipped, lr_s, &shape);
    let out = b.add_binary_add(weight_p, scaled, &shape);
    let def = b.build(out).expect("valid grad clip");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&shape, 1.0),
        const_tensor(&shape, -LR),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph
        .propagate_ibp(&uniform_bounds(&shape, 0.5))
        .expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Training grad clip IBP: [{lo:.6}, {hi:.6}]");
    assert!(hi - lo < 2.0, "clipped width bounded, got {}", hi - lo);
}

// 13. Full training step end-to-end (IBP + CROWN)

#[test]
fn test_training_full_step_end_to_end_ibp_crown() {
    let shape = [SEQ, HIDDEN];
    let cls_shape = [SEQ, CLS];
    let mut b = TensorBlockBuilder::new("training_full_step");
    let input = b.add_input("input", &shape);

    // Forward: Linear -> ReLU -> Linear -> softmax
    let w1 = b.add_input("w1", &[FFN, HIDDEN]);
    let f1 = b.add_linear(input, w1, None, &[SEQ, FFN]);
    let a1 = b.add_relu(f1, &[SEQ, FFN]);
    let w2 = b.add_input("w2", &[CLS, FFN]);
    let logits = b.add_linear(a1, w2, None, &cls_shape);
    let probs = b.add_softmax(logits, 1, &cls_shape);

    // Backward proxy: project back through transposed weights
    let w2t = b.add_input("w2t", &[FFN, CLS]);
    let g1 = b.add_linear(probs, w2t, None, &[SEQ, FFN]);
    let w1t = b.add_input("w1t", &[HIDDEN, FFN]);
    let out = b.add_linear(g1, w1t, None, &shape);
    let def = b.build(out).expect("valid full step");

    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[FFN, HIDDEN]),
        w(&[CLS, FFN]),
        w(&[FFN, CLS]),
        w(&[HIDDEN, FFN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let ib = uniform_bounds(&shape, 1.0);

    let ibp = graph.propagate_ibp(&ib).expect("IBP");
    assert_bounds_valid(&ibp);
    let (lo, hi) = bounds_min_max(&ibp);
    eprintln!("Training full step IBP: [{lo:.6}, {hi:.6}]");

    let (method, crown, fb) = assert_crown_tighter_when_not_fallback(&graph, &ib);
    let (clo, chi) = bounds_min_max(&crown);
    eprintln!("Training full step CROWN: method={method:?}, [{clo:.6}, {chi:.6}]");
    if let Some(r) = &fb {
        eprintln!("Fallback: {r}");
    }
}

// 14. Verify-and-record training pipeline (IBP)

#[test]
fn test_training_pipeline_verify_and_record() {
    let mut b = TensorBlockBuilder::new("training_pipeline_record");
    let input = b.add_input("input", &[SEQ, HIDDEN]);
    let w1 = b.add_input("w1", &[FFN, HIDDEN]);
    let b1 = b.add_input("b1", &[FFN]);
    let h = b.add_linear(input, w1, Some(b1), &[SEQ, FFN]);
    let h_act = b.add_relu(h, &[SEQ, FFN]);
    let w2 = b.add_input("w2", &[CLS, FFN]);
    let b2 = b.add_input("b2", &[CLS]);
    let logits = b.add_linear(h_act, w2, Some(b2), &[SEQ, CLS]);
    let out = b.add_softmax(logits, 1, &[SEQ, CLS]);
    let def = b.build(out).expect("valid recording pipeline");

    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[FFN, HIDDEN]),
        zeros(&[FFN]),
        w(&[CLS, FFN]),
        zeros(&[CLS]),
    ];
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input_bounds, "training_pipeline_record");
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Training verify-and-record: [{lo:.6}, {hi:.6}], mode={:?}",
        result.verification.soundness_mode
    );
    assert!(lo >= -1e-4, "softmax lo >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-4, "softmax hi <= 1, got {hi}");
}
