// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for gradient tape, computation graph structure, training
//! loop patterns, and mixed-precision training integration.
//!
//! Covers:
//! - Gradient tape: chain rule, product rule, quotient rule, accumulation, detach
//! - Graph structure: DAG shape, fan-out, fan-in, memory release
//! - Training loop patterns: SGD, Adam, grad clipping, LR scheduling, loss fns
//! - Mixed precision: F32 grads from BF16, loss scaling, dtype consistency

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;

use crate::grad::backward;
use crate::loss_scaling::{cast_grad_to_f32, DynamicLossScaler, MixedPrecisionConfig};
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

fn vec_var(data: Vec<f32>) -> Var {
    let n = data.len();
    Var::new(DynTensor::from_vec(data, &[n], &cpu()).unwrap())
}

fn mat_var(data: Vec<f32>, rows: usize, cols: usize) -> Var {
    Var::new(DynTensor::from_vec(data, &[rows, cols], &cpu()).unwrap())
}

fn get_grad_vec(grads: &crate::GradStore, var: &Var) -> Vec<f32> {
    grads
        .get(var)
        .expect("gradient should exist")
        .to_flat_vec::<f32>()
        .unwrap()
}

fn assert_approx(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "index {i}: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

// ===========================================================================
// 1. Gradient Tape Tests
// ===========================================================================

#[test]
fn test_tape_chain_rule_x_squared() {
    // d/dx (x^2) = 2x, at x=3 => grad=6
    let x = scalar_var(3.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.sqr().unwrap();
    let grads = backward(&y).unwrap();
    assert_approx(&get_grad_vec(&grads, &x), &[6.0], 1e-5);
}

#[test]
fn test_tape_product_rule() {
    // d/dx (x * y) at x=3, y=4 => d/dx=y=4, d/dy=x=3
    let x = scalar_var(3.0);
    let y = scalar_var(4.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let z = tx.mul(&ty).unwrap();
    let grads = backward(&z).unwrap();
    assert_approx(&get_grad_vec(&grads, &x), &[4.0], 1e-5);
    assert_approx(&get_grad_vec(&grads, &y), &[3.0], 1e-5);
}

#[test]
fn test_tape_quotient_rule() {
    // d/dx (x / y) at x=6, y=3 => d/dx=1/y=1/3, d/dy=-x/y^2=-6/9=-2/3
    let x = scalar_var(6.0);
    let y = scalar_var(3.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let z = tx.div(&ty).unwrap();
    let grads = backward(&z).unwrap();
    assert_approx(&get_grad_vec(&grads, &x), &[1.0 / 3.0], 1e-5);
    assert_approx(&get_grad_vec(&grads, &y), &[-6.0 / 9.0], 1e-5);
}

#[test]
fn test_tape_chain_through_multiple_ops() {
    // d/dx sin(x^2) = cos(x^2) * 2x, at x=1.0
    // sin(1) = 0.8414..., cos(1) = 0.5403...
    // grad = cos(1.0) * 2.0 = 1.0806...
    let x = scalar_var(1.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let x_sq = tx.sqr().unwrap();
    let y = x_sq.sin().unwrap();
    let grads = backward(&y).unwrap();
    let expected = 1.0_f32.cos() * 2.0; // cos(x^2)*2x at x=1
    assert_approx(&get_grad_vec(&grads, &x), &[expected], 1e-4);
}

#[test]
fn test_tape_gradient_accumulation_fan_out() {
    // z = x + x = 2x, dz/dx = 2
    let x = scalar_var(5.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let z = tx.add(&tx).unwrap();
    let grads = backward(&z).unwrap();
    assert_approx(&get_grad_vec(&grads, &x), &[2.0], 1e-5);
}

#[test]
fn test_tape_gradient_accumulation_triple_use() {
    // z = x * x * x (via tracked mul), d/dx(x^3) = 3x^2
    // Built as: t1 = x*x, t2 = t1*x
    // d(t2)/d(t1) = x, d(t1)/dx = 2x => chain: x * 2x = 2x^2
    // d(t2)/dx (direct use in t2) = t1 = x^2
    // Total = 2x^2 + x^2 = 3x^2
    let x = scalar_var(2.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let t1 = tx.mul(&tx).unwrap(); // x^2
    let t2 = t1.mul(&tx).unwrap(); // x^3
    let grads = backward(&t2).unwrap();
    let expected = 3.0 * 4.0; // 3 * x^2 = 12
    assert_approx(&get_grad_vec(&grads, &x), &[expected], 1e-4);
}

#[test]
fn test_tape_detach_stops_gradient() {
    // z = x * detach(x), grad should be x (only one path)
    let x = scalar_var(4.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tx_detached = tx.detach();
    let z = tx.mul(&tx_detached).unwrap();
    let grads = backward(&z).unwrap();
    // d/dx (x * c) where c=detach(x)=4 => grad = c = 4
    assert_approx(&get_grad_vec(&grads, &x), &[4.0], 1e-5);
}

#[test]
fn test_tape_nested_computation_exp_sqr() {
    // d/dx exp(x^2) = exp(x^2) * 2x, at x=1.0
    let x = scalar_var(1.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let x_sq = tx.sqr().unwrap();
    let y = x_sq.exp().unwrap();
    let grads = backward(&y).unwrap();
    let expected = (1.0_f32).exp() * 2.0; // e^1 * 2
    assert_approx(&get_grad_vec(&grads, &x), &[expected], 1e-4);
}

#[test]
fn test_tape_chain_sigmoid_sqr() {
    // d/dx sigmoid(x)^2 = 2*sigmoid(x)*sigmoid'(x) = 2*s*(s*(1-s))
    // at x=0: s=0.5, s'=0.25, result=2*0.5*0.25=0.25
    let x = scalar_var(0.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let s = tx.sigmoid().unwrap();
    let y = s.sqr().unwrap();
    let grads = backward(&y).unwrap();
    // d(s^2)/dx = 2*s * ds/dx = 2*0.5 * 0.25 = 0.25
    assert_approx(&get_grad_vec(&grads, &x), &[0.25], 1e-5);
}

#[test]
fn test_tape_neg_chain() {
    // d/dx (-x^2) = -2x, at x=3 => -6
    let x = scalar_var(3.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.sqr().unwrap().neg().unwrap();
    let grads = backward(&y).unwrap();
    assert_approx(&get_grad_vec(&grads, &x), &[-6.0], 1e-5);
}

#[test]
fn test_tape_add_scalar_chain() {
    // d/dx ((x + 5)^2) = 2*(x+5), at x=3 => 2*8 = 16
    let x = scalar_var(3.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let shifted = tx.add_scalar(5.0).unwrap();
    let y = shifted.sqr().unwrap();
    let grads = backward(&y).unwrap();
    assert_approx(&get_grad_vec(&grads, &x), &[16.0], 1e-4);
}

#[test]
fn test_tape_mul_scalar_chain() {
    // d/dx (3*x)^2 = d/dx 9*x^2 = 18x, at x=2 => 36
    let x = scalar_var(2.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let scaled = tx.mul_scalar(3.0).unwrap();
    let y = scaled.sqr().unwrap();
    let grads = backward(&y).unwrap();
    assert_approx(&get_grad_vec(&grads, &x), &[36.0], 1e-4);
}

#[test]
fn test_tape_powf_gradient() {
    // d/dx x^3 = 3x^2 at x=2 => 12
    let x = scalar_var(2.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.powf(3.0).unwrap();
    let grads = backward(&y).unwrap();
    assert_approx(&get_grad_vec(&grads, &x), &[12.0], 1e-3);
}

// ===========================================================================
// 2. Graph Structure Tests
// ===========================================================================

#[test]
fn test_graph_dag_no_cycles_leaf_no_op() {
    // Leaf nodes have op() == None
    let x = scalar_var(1.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    assert!(tx.op().is_none(), "leaf should have no op");
    assert!(tx.is_var(), "leaf from var should be var");
}

#[test]
fn test_graph_intermediate_has_op() {
    let x = scalar_var(1.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.sqr().unwrap();
    assert!(y.op().is_some(), "intermediate should have an op");
    assert!(!y.is_var(), "intermediate should not be var");
}

#[test]
fn test_graph_fan_out_one_input_multiple_ops() {
    // x used in both branches: z = x^2 + x
    let x = scalar_var(3.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let x_sq = tx.sqr().unwrap();
    let z = x_sq.add(&tx).unwrap();
    let grads = backward(&z).unwrap();
    // d/dx(x^2 + x) = 2x + 1 = 7
    assert_approx(&get_grad_vec(&grads, &x), &[7.0], 1e-4);
}

#[test]
fn test_graph_fan_in_multiple_inputs_merged() {
    // z = a * b + c, three separate vars converge
    let a = scalar_var(2.0);
    let b = scalar_var(3.0);
    let c = scalar_var(4.0);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let tc = Arc::new(TrackedTensor::from_var(&c).unwrap());
    let ab = ta.mul(&tb).unwrap();
    let z = ab.add(&tc).unwrap();
    let grads = backward(&z).unwrap();
    // dz/da = b = 3, dz/db = a = 2, dz/dc = 1
    assert_approx(&get_grad_vec(&grads, &a), &[3.0], 1e-5);
    assert_approx(&get_grad_vec(&grads, &b), &[2.0], 1e-5);
    assert_approx(&get_grad_vec(&grads, &c), &[1.0], 1e-5);
}

#[test]
fn test_graph_diamond_structure() {
    // Diamond: x -> a, x -> b, a+b -> z
    // a = x^2, b = 2*x, z = a + b = x^2 + 2x
    // dz/dx = 2x + 2 at x=3 => 8
    let x = scalar_var(3.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let a = tx.sqr().unwrap();
    let b = tx.mul_scalar(2.0).unwrap();
    let z = a.add(&b).unwrap();
    let grads = backward(&z).unwrap();
    assert_approx(&get_grad_vec(&grads, &x), &[8.0], 1e-4);
}

#[test]
fn test_graph_constant_leaf_no_gradient() {
    // Constant tensors should not receive gradients
    let x = scalar_var(3.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let c_data = DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap();
    let tc = Arc::new(TrackedTensor::from_tensor(c_data));
    let z = tx.mul(&tc).unwrap();
    let grads = backward(&z).unwrap();
    assert!(grads.get(&x).is_some(), "var should have gradient");
    // constant leaf has no VarId, so GradStore won't have it
    assert_eq!(grads.var_count(), 1, "only one var in the graph");
}

#[test]
fn test_graph_unique_node_ids() {
    let x = scalar_var(1.0);
    let y = scalar_var(2.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let z = tx.add(&ty).unwrap();
    // All nodes should have unique IDs
    assert_ne!(tx.node_id(), ty.node_id());
    assert_ne!(tx.node_id(), z.node_id());
    assert_ne!(ty.node_id(), z.node_id());
}

#[test]
fn test_graph_memory_release_into_tensor() {
    // into_tensor() drops the op graph, releasing Arc references
    let x = scalar_var(2.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.sqr().unwrap();
    // Extract the data, dropping the graph
    let data = Arc::try_unwrap(y).unwrap().into_tensor().unwrap();
    let vals = data.to_flat_vec::<f32>().unwrap();
    assert_approx(&vals, &[4.0], 1e-5);
    // tx should now have only one reference (ours)
    assert_eq!(Arc::strong_count(&tx), 1);
}

// ===========================================================================
// 3. Training Loop Pattern Tests
// ===========================================================================

#[test]
fn test_training_sgd_step_manual() {
    // Manual SGD: w -= lr * grad
    let w = scalar_var(5.0);
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let loss = tw.sqr().unwrap(); // loss = w^2, grad = 2w = 10
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&w).unwrap();

    let lr = 0.1_f64;
    let w_data = w.data().unwrap();
    let scaled_grad = grad.mul_scalar(lr).unwrap();
    let new_w = w_data.sub(&scaled_grad).unwrap();
    w.set(&new_w).unwrap();

    let w_val = w.data().unwrap().to_flat_vec::<f32>().unwrap();
    // 5.0 - 0.1 * 10.0 = 4.0
    assert_approx(&w_val, &[4.0], 1e-5);
}

#[test]
fn test_training_adam_moment_update_pattern() {
    // Verify Adam moment update equations manually
    let beta1: f64 = 0.9;
    let beta2: f64 = 0.999;
    let lr: f64 = 0.001;
    let eps: f64 = 1e-8;

    let w = scalar_var(1.0);
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let loss = tw.sqr().unwrap(); // grad = 2.0
    let grads = backward(&loss).unwrap();
    let grad_val = f64::from(grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap()[0]);

    // Step 1 moment update
    let m1 = beta1 * 0.0 + (1.0 - beta1) * grad_val; // 0.2
    let v1 = beta2 * 0.0 + (1.0 - beta2) * grad_val * grad_val; // 0.004
    let m_hat = m1 / (1.0 - beta1); // 2.0
    let v_hat = v1 / (1.0 - beta2); // 4.0
    let update = lr * m_hat / (v_hat.sqrt() + eps); // ~0.001

    let new_w = 1.0 - update;
    assert!((new_w - 0.999).abs() < 0.01, "adam update should be ~0.001");
}

#[test]
fn test_training_gradient_clipping_pattern() {
    // Simulate gradient clipping: if norm > max_norm, scale down
    let w = vec_var(vec![3.0, 4.0]); // grad norm will be 5.0
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    // loss = sum(w) => grad = [1, 1] -- but we want large grads
    // loss = sum(w^2) => grad = [6, 8], norm = 10
    let sq = tw.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&w).unwrap();
    let grad_vals = grad.to_flat_vec::<f32>().unwrap();
    // grads should be [6.0, 8.0]
    assert_approx(&grad_vals, &[6.0, 8.0], 1e-4);

    // Clip to max_norm = 5.0
    let max_norm: f64 = 5.0;
    let norm: f64 = grad_vals
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 10.0).abs() < 0.01);

    if norm > max_norm {
        let scale = max_norm / norm;
        let clipped: Vec<f32> = grad_vals.iter().map(|&v| v * scale as f32).collect();
        let clipped_norm: f64 = clipped
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            (clipped_norm - max_norm).abs() < 0.01,
            "clipped norm should be max_norm"
        );
    }
}

#[test]
fn test_training_lr_schedule_warmup_pattern() {
    // Simulate linear warmup: lr = base_lr * step / warmup_steps
    let base_lr: f64 = 0.01;
    let warmup_steps: usize = 10;

    for step in 0..=warmup_steps {
        let lr = if step < warmup_steps {
            base_lr * (step as f64 / warmup_steps as f64)
        } else {
            base_lr
        };
        if step == 0 {
            assert_eq!(lr, 0.0);
        } else if step == warmup_steps {
            assert_eq!(lr, base_lr);
        } else {
            assert!(lr > 0.0 && lr < base_lr);
        }
    }
}

#[test]
fn test_training_mse_loss_computation() {
    // MSE = mean((pred - target)^2)
    let pred = vec_var(vec![1.0, 2.0, 3.0]);
    let target_data = DynTensor::from_vec(vec![1.5, 2.5, 2.5], &[3], &cpu()).unwrap();
    let target = Arc::new(TrackedTensor::from_tensor(target_data));

    let tp = Arc::new(TrackedTensor::from_var(&pred).unwrap());
    let diff = tp.sub(&target).unwrap();
    let sq = diff.sqr().unwrap();
    let loss = sq.mean_keepdim(0).unwrap();

    let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    // MSE = ((-.5)^2 + (-.5)^2 + (.5)^2) / 3 = 0.75 / 3 = 0.25
    assert!((loss_val - 0.25).abs() < 1e-5);

    let grads = backward(&loss).unwrap();
    let grad = get_grad_vec(&grads, &pred);
    // d(MSE)/d(pred_i) = 2*(pred_i - target_i) / N
    // = [2*(-0.5)/3, 2*(-0.5)/3, 2*(0.5)/3] = [-1/3, -1/3, 1/3]
    assert_approx(&grad, &[-1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0], 1e-4);
}

#[test]
fn test_training_l1_loss_pattern() {
    // L1 loss = mean(|pred - target|)
    let pred = vec_var(vec![1.0, 3.0]);
    let target_data = DynTensor::from_vec(vec![2.0, 1.0], &[2], &cpu()).unwrap();
    let target = Arc::new(TrackedTensor::from_tensor(target_data));

    let tp = Arc::new(TrackedTensor::from_var(&pred).unwrap());
    let diff = tp.sub(&target).unwrap();
    let absd = diff.abs().unwrap();
    let loss = absd.mean_keepdim(0).unwrap();

    let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    // (|1-2| + |3-1|) / 2 = (1 + 2) / 2 = 1.5
    assert!((loss_val - 1.5).abs() < 1e-5);
}

#[test]
fn test_training_batch_simulation() {
    // Simulate batch training: accumulate gradients over mini-batch
    let w = scalar_var(2.0);
    let batch_inputs = vec![1.0_f32, 2.0, 3.0];
    let _lr = 0.01;

    let mut total_grad = 0.0_f64;
    for &input in &batch_inputs {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let input_t = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![input], &[1], &cpu()).unwrap(),
        ));
        let pred = tw.mul(&input_t).unwrap(); // pred = w * input
        let loss = pred.sqr().unwrap(); // loss = (w*input)^2
        let grads = backward(&loss).unwrap();
        let g = get_grad_vec(&grads, &w)[0];
        total_grad += f64::from(g);
    }
    let avg_grad = total_grad / batch_inputs.len() as f64;
    // d/dw (w*x)^2 = 2*w*x^2, averaged over batch
    // = (2*2*1 + 2*2*4 + 2*2*9) / 3 = (4 + 16 + 36) / 3 = 56/3
    let expected = (2.0 * 2.0 * 1.0 + 2.0 * 2.0 * 4.0 + 2.0 * 2.0 * 9.0) / 3.0;
    assert!((avg_grad - expected).abs() < 0.1);
}

#[test]
fn test_training_parameter_groups_different_lr() {
    // Two parameter groups with different learning rates
    let w1 = scalar_var(5.0);
    let w2 = scalar_var(3.0);

    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());

    let z1 = tw1.sqr().unwrap(); // loss1 = w1^2
    let z2 = tw2.sqr().unwrap(); // loss2 = w2^2
    let loss = z1.add(&z2).unwrap();
    let grads = backward(&loss).unwrap();

    // Apply different LR per group
    let lr1 = 0.1;
    let lr2 = 0.01;

    let g1 = grads.get(&w1).unwrap();
    let g2 = grads.get(&w2).unwrap();

    let new_w1 = w1
        .data()
        .unwrap()
        .sub(&g1.mul_scalar(lr1).unwrap())
        .unwrap();
    let new_w2 = w2
        .data()
        .unwrap()
        .sub(&g2.mul_scalar(lr2).unwrap())
        .unwrap();

    w1.set(&new_w1).unwrap();
    w2.set(&new_w2).unwrap();

    // w1: 5 - 0.1*10 = 4.0
    // w2: 3 - 0.01*6 = 2.94
    assert_approx(
        &w1.data().unwrap().to_flat_vec::<f32>().unwrap(),
        &[4.0],
        1e-5,
    );
    assert_approx(
        &w2.data().unwrap().to_flat_vec::<f32>().unwrap(),
        &[2.94],
        1e-5,
    );
}

#[test]
fn test_training_multi_step_convergence() {
    // Run 10 SGD steps on loss = w^2, verify w approaches 0
    let w = scalar_var(5.0);
    let lr = 0.1_f64;

    for _ in 0..10 {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let loss = tw.sqr().unwrap();
        let grads = backward(&loss).unwrap();
        let grad = grads.get(&w).unwrap();
        let scaled_grad = grad.mul_scalar(lr).unwrap();
        let new_w = w.data().unwrap().sub(&scaled_grad).unwrap();
        w.set(&new_w).unwrap();
    }

    let w_val = w.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    // After 10 steps with lr=0.1: w * (1-2*0.1)^10 = 5 * 0.8^10 ~ 0.537
    assert!(w_val.abs() < 1.0, "w should converge toward 0, got {w_val}");
}

#[test]
fn test_training_weight_decay_pattern() {
    // Simulate weight decay: grad = grad + wd * w
    let w = scalar_var(4.0);
    let wd = 0.01;
    let lr = 0.1;

    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let loss = tw.sqr().unwrap(); // grad = 2w = 8
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&w).unwrap();

    // Effective grad = grad + wd * w = 8 + 0.01*4 = 8.04
    let w_data = w.data().unwrap();
    let decay_term = w_data.mul_scalar(wd).unwrap();
    let effective_grad = grad.add(&decay_term).unwrap();
    let new_w = w_data.sub(&effective_grad.mul_scalar(lr).unwrap()).unwrap();
    w.set(&new_w).unwrap();

    let w_val = w.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    // 4.0 - 0.1 * 8.04 = 4.0 - 0.804 = 3.196
    assert_approx(&[w_val], &[3.196], 1e-3);
}

// ===========================================================================
// 4. Mixed Precision Training Tests
// ===========================================================================

#[test]
fn test_mixed_precision_f32_gradients_from_bf16_data() {
    // BF16 data → F32 computation → F32 gradients
    let bf16_data = DynTensor::from_vec(vec![2.0_f32], &[1], &cpu())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    // Cast to F32 for computation (as the framework does internally)
    let f32_data = bf16_data.to_dtype(DType::F32).unwrap();
    let w = Var::new(f32_data);
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let loss = tw.sqr().unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&w).unwrap();
    assert_eq!(grad.dtype(), DType::F32, "gradient should be F32");
    let grad_val = grad.to_flat_vec::<f32>().unwrap();
    assert_approx(&grad_val, &[4.0], 0.1); // 2x = 4, with BF16 precision
}

#[test]
fn test_mixed_precision_loss_scaling_roundtrip() {
    let scaler = DynamicLossScaler::default();
    let loss = DynTensor::from_vec(vec![0.5_f32], &[1], &cpu()).unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();
    let scaled_val = scaled.to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (scaled_val - 0.5 * 65536.0).abs() < 1.0,
        "scaled = loss * scale_factor"
    );

    // Unscale
    let mut grads = vec![scaled];
    scaler.unscale_gradients(&mut grads).unwrap();
    let unscaled_val = grads[0].to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (unscaled_val - 0.5).abs() < 1e-3,
        "unscaled should match original"
    );
}

#[test]
fn test_mixed_precision_scaler_backoff_on_inf() {
    let mut scaler = DynamicLossScaler::default();
    let initial_scale = scaler.scale_factor();

    // Simulate an inf gradient
    scaler.update(true);
    assert!(
        scaler.scale_factor() < initial_scale,
        "scale should decrease on inf"
    );
    assert_eq!(scaler.consecutive_good_steps(), 0);
}

#[test]
fn test_mixed_precision_scaler_growth_on_good_steps() {
    let config = MixedPrecisionConfig {
        loss_scale: 100.0,
        grad_dtype: DType::F32,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 3, // grow after 3 good steps
    };
    let mut scaler = DynamicLossScaler::new(config).unwrap();

    for _ in 0..3 {
        scaler.update(false);
    }
    assert_eq!(
        scaler.scale_factor(),
        200.0,
        "scale should double after growth_interval good steps"
    );
}

#[test]
fn test_mixed_precision_cast_grad_to_f32() {
    // F32 → passthrough
    let f32_grad = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let result = cast_grad_to_f32(&f32_grad).unwrap();
    assert_eq!(result.dtype(), DType::F32);

    // BF16 → upcast to F32
    let bf16_grad = f32_grad.to_dtype(DType::BF16).unwrap();
    let result = cast_grad_to_f32(&bf16_grad).unwrap();
    assert_eq!(result.dtype(), DType::F32);

    // F16 → upcast to F32
    let f16_grad = f32_grad.to_dtype(DType::F16).unwrap();
    let result = cast_grad_to_f32(&f16_grad).unwrap();
    assert_eq!(result.dtype(), DType::F32);
}

#[test]
fn test_mixed_precision_gradient_dtype_consistency() {
    // All gradients from backward should have the same dtype as the loss
    let w1 = vec_var(vec![1.0, 2.0]);
    let w2 = vec_var(vec![3.0, 4.0]);
    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let z = tw1.add(&tw2).unwrap();
    let loss = z.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let g1 = grads.get(&w1).unwrap();
    let g2 = grads.get(&w2).unwrap();
    assert_eq!(g1.dtype(), DType::F32, "grad dtype should match loss dtype");
    assert_eq!(g2.dtype(), DType::F32, "grad dtype should match loss dtype");
}

// ===========================================================================
// 5. Additional Edge Case & Integration Tests
// ===========================================================================

#[test]
fn test_backward_for_vars_selective_gradients() {
    // backward_for_vars should only return gradients for specified vars
    let w1 = scalar_var(2.0);
    let w2 = scalar_var(3.0);
    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let loss = tw1.mul(&tw2).unwrap();
    let grads = crate::backward_for_vars(&loss, &[&w1]).unwrap();
    assert!(grads.get(&w1).is_some(), "w1 should have gradient");
    assert!(grads.get(&w2).is_none(), "w2 should be filtered out");
}

#[test]
fn test_backward_non_scalar_loss_error() {
    let w = vec_var(vec![1.0, 2.0, 3.0]);
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let result = backward(&tw);
    assert!(result.is_err(), "backward should reject non-scalar loss");
}

#[test]
fn test_vector_gradient_sum_reduction() {
    // Vector var, reduce to scalar via sum, check grad is all-ones
    let w = vec_var(vec![1.0, 2.0, 3.0, 4.0]);
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let loss = tw.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = get_grad_vec(&grads, &w);
    assert_approx(&grad, &[1.0, 1.0, 1.0, 1.0], 1e-5);
}

#[test]
fn test_vector_gradient_mean_reduction() {
    // Vector var, reduce to scalar via mean, check grad is 1/N
    let w = vec_var(vec![1.0, 2.0, 3.0, 4.0]);
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let loss = tw.mean_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = get_grad_vec(&grads, &w);
    assert_approx(&grad, &[0.25, 0.25, 0.25, 0.25], 1e-5);
}

#[test]
fn test_matmul_gradient_2d() {
    // z = x @ w^T, loss = sum(z)
    // dL/dw = x^T
    let x_data = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let x = Arc::new(TrackedTensor::from_tensor(x_data));

    let w = mat_var(vec![3.0, 4.0, 5.0, 6.0], 2, 2);
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let wt = tw.transpose(0, 1).unwrap();
    let z = x.matmul(&wt).unwrap(); // [1,2] @ [2,2] = [1,2]
    let loss = z.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    assert!(grads.get(&w).is_some(), "weight should have gradient");
    let grad_shape = grads.get(&w).unwrap().dims().to_vec();
    assert_eq!(
        grad_shape,
        vec![2, 2],
        "gradient shape should match weight shape"
    );
}

#[test]
fn test_training_loop_loss_decreasing() {
    // Verify that loss decreases over multiple training steps
    let w = scalar_var(10.0);
    let lr = 0.05;
    let mut prev_loss = f32::INFINITY;

    for step in 0..5 {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let loss = tw.sqr().unwrap(); // loss = w^2
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];

        assert!(
            loss_val < prev_loss,
            "step {step}: loss {loss_val} should be less than prev {prev_loss}"
        );
        prev_loss = loss_val;

        let grads = backward(&loss).unwrap();
        let grad = grads.get(&w).unwrap();
        let new_w = w
            .data()
            .unwrap()
            .sub(&grad.mul_scalar(lr).unwrap())
            .unwrap();
        w.set(&new_w).unwrap();
    }
}
