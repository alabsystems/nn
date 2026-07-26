// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for standalone conv1d and SDPA backward functions.
//!
//! Validates `conv1d_backward` and `scaled_dot_product_attention_backward`
//! against numerical finite-difference gradients.

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use super::{conv1d_backward, scaled_dot_product_attention_backward};

/// Sum all elements as f64 for loss computation.
fn sum_f64(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

/// Finite-difference gradient check: compares analytical vs numerical gradients.
fn check_fd_grad(analytical: &[f32], data: &[f32], eps: f32, fwd: impl Fn(Vec<f32>) -> f64) {
    let tol = 1e-2;
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        let numerical = (fwd(plus) - fwd(minus)) / (2.0 * f64::from(eps));
        let analytical_f64 = f64::from(analytical[i]);
        let err = (analytical_f64 - numerical).abs();
        assert!(
            err < tol,
            "grad[{i}]: analytical={analytical_f64}, numerical={numerical}, err={err}, tol={tol}",
        );
    }
}

// ── Conv1d backward tests ──────────────────────────────────────────────

/// Forward conv1d loss (DynTensor only, no tape).
fn conv1d_ref_loss(
    x_data: &[f32],
    w_data: &[f32],
    x_shape: &[usize],
    w_shape: &[usize],
    stride: usize,
    padding: usize,
) -> f64 {
    let x = DynTensor::from_vec(x_data.to_vec(), x_shape, &cpu()).unwrap();
    let w = DynTensor::from_vec(w_data.to_vec(), w_shape, &cpu()).unwrap();
    sum_f64(&x.conv1d(&w, padding, stride, 1, 1).unwrap())
}

/// FD test: conv1d grad_input (no padding, stride=1).
#[test]
fn test_conv1d_backward_grad_input_fd() {
    let in_ch = 2;
    let in_len = 6;
    let out_ch = 3;
    let k_size = 3;

    let x_data: Vec<f32> = (0..in_ch * in_len).map(|v| v as f32 * 0.1 - 0.3).collect();
    let w_data: Vec<f32> = (0..out_ch * in_ch * k_size)
        .map(|v| v as f32 * 0.08 - 0.2)
        .collect();

    let x_shape = [1, in_ch, in_len];
    let w_shape = [out_ch, in_ch, k_size];

    // Compute forward to get grad_output shape.
    let x = DynTensor::from_vec(x_data.clone(), &x_shape, &cpu()).unwrap();
    let w = DynTensor::from_vec(w_data.clone(), &w_shape, &cpu()).unwrap();
    let y = x.conv1d(&w, 0, 1, 1, 1).unwrap();

    // grad_output = ones (sum loss).
    let grad_output = DynTensor::ones(y.dims(), y.dtype(), &y.device()).unwrap();

    let (grad_input, _grad_weight) = conv1d_backward(&grad_output, &x, &w, 1, 0).unwrap();

    // Verify shape.
    assert_eq!(grad_input.dims(), x.dims());

    // FD check.
    let analytical = grad_input.to_flat_vec::<f32>().unwrap();
    let eps = 1e-3_f32;
    check_fd_grad(&analytical, &x_data, eps, |d| {
        conv1d_ref_loss(&d, &w_data, &x_shape, &w_shape, 1, 0)
    });
}

/// FD test: conv1d grad_weight (no padding, stride=1).
#[test]
fn test_conv1d_backward_grad_weight_fd() {
    let in_ch = 2;
    let in_len = 6;
    let out_ch = 3;
    let k_size = 3;

    let x_data: Vec<f32> = (0..in_ch * in_len).map(|v| v as f32 * 0.1 - 0.3).collect();
    let w_data: Vec<f32> = (0..out_ch * in_ch * k_size)
        .map(|v| v as f32 * 0.08 - 0.2)
        .collect();

    let x_shape = [1, in_ch, in_len];
    let w_shape = [out_ch, in_ch, k_size];

    let x = DynTensor::from_vec(x_data.clone(), &x_shape, &cpu()).unwrap();
    let w = DynTensor::from_vec(w_data.clone(), &w_shape, &cpu()).unwrap();
    let y = x.conv1d(&w, 0, 1, 1, 1).unwrap();

    let grad_output = DynTensor::ones(y.dims(), y.dtype(), &y.device()).unwrap();
    let (_grad_input, grad_weight) = conv1d_backward(&grad_output, &x, &w, 1, 0).unwrap();

    // Verify shape.
    assert_eq!(grad_weight.dims(), w.dims());

    // FD check.
    let analytical = grad_weight.to_flat_vec::<f32>().unwrap();
    let eps = 1e-3_f32;
    check_fd_grad(&analytical, &w_data, eps, |d| {
        conv1d_ref_loss(&x_data, &d, &x_shape, &w_shape, 1, 0)
    });
}

/// FD test: conv1d grad_input with padding=1, stride=2.
#[test]
fn test_conv1d_backward_grad_input_padded_strided_fd() {
    let in_ch = 2;
    let in_len = 8;
    let out_ch = 2;
    let k_size = 3;
    let stride = 2;
    let padding = 1;

    let x_data: Vec<f32> = (0..in_ch * in_len).map(|v| v as f32 * 0.05 - 0.4).collect();
    let w_data: Vec<f32> = (0..out_ch * in_ch * k_size)
        .map(|v| v as f32 * 0.1 - 0.3)
        .collect();

    let x_shape = [1, in_ch, in_len];
    let w_shape = [out_ch, in_ch, k_size];

    let x = DynTensor::from_vec(x_data.clone(), &x_shape, &cpu()).unwrap();
    let w = DynTensor::from_vec(w_data.clone(), &w_shape, &cpu()).unwrap();
    let y = x.conv1d(&w, padding, stride, 1, 1).unwrap();

    let grad_output = DynTensor::ones(y.dims(), y.dtype(), &y.device()).unwrap();
    let (grad_input, _) = conv1d_backward(&grad_output, &x, &w, stride, padding).unwrap();

    assert_eq!(grad_input.dims(), x.dims());

    let analytical = grad_input.to_flat_vec::<f32>().unwrap();
    let eps = 1e-3_f32;
    check_fd_grad(&analytical, &x_data, eps, |d| {
        conv1d_ref_loss(&d, &w_data, &x_shape, &w_shape, stride, padding)
    });
}

/// FD test: conv1d grad_weight with padding=1, stride=2.
#[test]
fn test_conv1d_backward_grad_weight_padded_strided_fd() {
    let in_ch = 2;
    let in_len = 8;
    let out_ch = 2;
    let k_size = 3;
    let stride = 2;
    let padding = 1;

    let x_data: Vec<f32> = (0..in_ch * in_len).map(|v| v as f32 * 0.05 - 0.4).collect();
    let w_data: Vec<f32> = (0..out_ch * in_ch * k_size)
        .map(|v| v as f32 * 0.1 - 0.3)
        .collect();

    let x_shape = [1, in_ch, in_len];
    let w_shape = [out_ch, in_ch, k_size];

    let x = DynTensor::from_vec(x_data.clone(), &x_shape, &cpu()).unwrap();
    let w = DynTensor::from_vec(w_data.clone(), &w_shape, &cpu()).unwrap();
    let y = x.conv1d(&w, padding, stride, 1, 1).unwrap();

    let grad_output = DynTensor::ones(y.dims(), y.dtype(), &y.device()).unwrap();
    let (_, grad_weight) = conv1d_backward(&grad_output, &x, &w, stride, padding).unwrap();

    assert_eq!(grad_weight.dims(), w.dims());

    let analytical = grad_weight.to_flat_vec::<f32>().unwrap();
    let eps = 1e-3_f32;
    check_fd_grad(&analytical, &w_data, eps, |d| {
        conv1d_ref_loss(&x_data, &d, &x_shape, &w_shape, stride, padding)
    });
}

/// Shape correctness: conv1d_backward output shapes match input/weight shapes.
#[test]
fn test_conv1d_backward_shapes() {
    let configs = [
        // (in_ch, in_len, out_ch, k_size, stride, padding)
        (1, 5, 1, 3, 1, 0),
        (2, 8, 4, 3, 1, 1),
        (3, 10, 6, 5, 2, 2),
        (4, 16, 8, 3, 2, 1),
    ];

    for (in_ch, in_len, out_ch, k_size, stride, padding) in configs {
        let x = DynTensor::ones(&[1, in_ch, in_len], nn_core::DType::F32, &cpu()).unwrap();
        let w = DynTensor::ones(&[out_ch, in_ch, k_size], nn_core::DType::F32, &cpu()).unwrap();
        let y = x.conv1d(&w, padding, stride, 1, 1).unwrap();
        let grad_output = DynTensor::ones(y.dims(), y.dtype(), &y.device()).unwrap();

        let (gi, gw) = conv1d_backward(&grad_output, &x, &w, stride, padding).unwrap();
        assert_eq!(
            gi.dims(),
            x.dims(),
            "grad_input shape mismatch for config ({in_ch}, {in_len}, {out_ch}, {k_size}, s={stride}, p={padding})"
        );
        assert_eq!(
            gw.dims(),
            w.dims(),
            "grad_weight shape mismatch for config ({in_ch}, {in_len}, {out_ch}, {k_size}, s={stride}, p={padding})"
        );
    }
}

/// Rank validation: conv1d_backward rejects non-3D inputs.
#[test]
fn test_conv1d_backward_rank_validation() {
    let x_2d = DynTensor::ones(&[2, 3], nn_core::DType::F32, &cpu()).unwrap();
    let w_3d = DynTensor::ones(&[1, 2, 3], nn_core::DType::F32, &cpu()).unwrap();
    let g_3d = DynTensor::ones(&[1, 1, 1], nn_core::DType::F32, &cpu()).unwrap();

    let result = conv1d_backward(&g_3d, &x_2d, &w_3d, 1, 0);
    assert!(result.is_err());
}

// ── Scaled dot-product attention backward tests ────────────────────────

/// Forward SDPA loss (DynTensor only, no tape).
fn sdpa_ref_loss(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    shape: &[usize], // [B, H, S, d_k]
) -> f64 {
    let q = DynTensor::from_vec(q_data.to_vec(), shape, &cpu()).unwrap();
    let k = DynTensor::from_vec(k_data.to_vec(), shape, &cpu()).unwrap();
    let v = DynTensor::from_vec(v_data.to_vec(), shape, &cpu()).unwrap();

    let d_k = shape[3];
    let scale = 1.0 / (d_k as f64).sqrt();

    // scores = Q @ K^T / sqrt(d_k)
    let k_t = k.transpose(2, 3).unwrap();
    let scores = q.matmul(&k_t).unwrap().mul_scalar(scale).unwrap();
    let attn = scores.softmax(3).unwrap();
    let out = attn.matmul(&v).unwrap();
    sum_f64(&out)
}

/// FD test: SDPA grad_q.
#[test]
fn test_sdpa_backward_grad_q_fd() {
    let shape = [1, 2, 3, 4]; // B=1, H=2, S=3, d_k=4
    let n = shape.iter().product::<usize>();

    let q_data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.05).collect();
    let k_data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 3.0) * 0.04).collect();
    let v_data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 4.0) * 0.06).collect();

    let q = DynTensor::from_vec(q_data.clone(), &shape, &cpu()).unwrap();
    let k = DynTensor::from_vec(k_data.clone(), &shape, &cpu()).unwrap();
    let v = DynTensor::from_vec(v_data.clone(), &shape, &cpu()).unwrap();

    let d_k = shape[3];
    let scale = 1.0 / (d_k as f64).sqrt();
    let k_t = k.transpose(2, 3).unwrap();
    let scores = q.matmul(&k_t).unwrap().mul_scalar(scale).unwrap();
    let attn = scores.softmax(3).unwrap();
    let out = attn.matmul(&v).unwrap();

    let grad_output = DynTensor::ones(out.dims(), out.dtype(), &out.device()).unwrap();
    let (grad_q, _, _) = scaled_dot_product_attention_backward(&grad_output, &q, &k, &v).unwrap();

    assert_eq!(grad_q.dims(), q.dims());

    let analytical = grad_q.to_flat_vec::<f32>().unwrap();
    let eps = 1e-3_f32;
    check_fd_grad(&analytical, &q_data, eps, |d| {
        sdpa_ref_loss(&d, &k_data, &v_data, &shape)
    });
}

/// FD test: SDPA grad_k.
#[test]
fn test_sdpa_backward_grad_k_fd() {
    let shape = [1, 2, 3, 4];
    let n = shape.iter().product::<usize>();

    let q_data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.05).collect();
    let k_data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 3.0) * 0.04).collect();
    let v_data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 4.0) * 0.06).collect();

    let q = DynTensor::from_vec(q_data.clone(), &shape, &cpu()).unwrap();
    let k = DynTensor::from_vec(k_data.clone(), &shape, &cpu()).unwrap();
    let v = DynTensor::from_vec(v_data.clone(), &shape, &cpu()).unwrap();

    let d_k = shape[3];
    let scale = 1.0 / (d_k as f64).sqrt();
    let k_t = k.transpose(2, 3).unwrap();
    let scores = q.matmul(&k_t).unwrap().mul_scalar(scale).unwrap();
    let attn = scores.softmax(3).unwrap();
    let out = attn.matmul(&v).unwrap();

    let grad_output = DynTensor::ones(out.dims(), out.dtype(), &out.device()).unwrap();
    let (_, grad_k, _) = scaled_dot_product_attention_backward(&grad_output, &q, &k, &v).unwrap();

    assert_eq!(grad_k.dims(), k.dims());

    let analytical = grad_k.to_flat_vec::<f32>().unwrap();
    let eps = 1e-3_f32;
    check_fd_grad(&analytical, &k_data, eps, |d| {
        sdpa_ref_loss(&q_data, &d, &v_data, &shape)
    });
}

/// FD test: SDPA grad_v.
#[test]
fn test_sdpa_backward_grad_v_fd() {
    let shape = [1, 2, 3, 4];
    let n = shape.iter().product::<usize>();

    let q_data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.05).collect();
    let k_data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 3.0) * 0.04).collect();
    let v_data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 4.0) * 0.06).collect();

    let q = DynTensor::from_vec(q_data.clone(), &shape, &cpu()).unwrap();
    let k = DynTensor::from_vec(k_data.clone(), &shape, &cpu()).unwrap();
    let v = DynTensor::from_vec(v_data.clone(), &shape, &cpu()).unwrap();

    let d_k = shape[3];
    let scale = 1.0 / (d_k as f64).sqrt();
    let k_t = k.transpose(2, 3).unwrap();
    let scores = q.matmul(&k_t).unwrap().mul_scalar(scale).unwrap();
    let attn = scores.softmax(3).unwrap();
    let out = attn.matmul(&v).unwrap();

    let grad_output = DynTensor::ones(out.dims(), out.dtype(), &out.device()).unwrap();
    let (_, _, grad_v) = scaled_dot_product_attention_backward(&grad_output, &q, &k, &v).unwrap();

    assert_eq!(grad_v.dims(), v.dims());

    let analytical = grad_v.to_flat_vec::<f32>().unwrap();
    let eps = 1e-3_f32;
    check_fd_grad(&analytical, &v_data, eps, |d| {
        sdpa_ref_loss(&q_data, &k_data, &d, &shape)
    });
}

/// Shape correctness: SDPA backward output shapes match input shapes.
#[test]
fn test_sdpa_backward_shapes() {
    let configs: &[[usize; 4]] = &[
        [1, 1, 2, 4],  // minimal
        [1, 2, 3, 4],  // multi-head
        [2, 4, 8, 16], // larger
    ];

    for shape in configs {
        let n = shape.iter().product::<usize>();
        let q =
            DynTensor::from_vec((0..n).map(|i| i as f32 * 0.01).collect(), shape, &cpu()).unwrap();
        let k = DynTensor::from_vec(
            (0..n).map(|i| i as f32 * 0.01 + 0.1).collect(),
            shape,
            &cpu(),
        )
        .unwrap();
        let v = DynTensor::from_vec(
            (0..n).map(|i| i as f32 * 0.01 - 0.1).collect(),
            shape,
            &cpu(),
        )
        .unwrap();

        let grad_out = DynTensor::ones(shape, nn_core::DType::F32, &cpu()).unwrap();
        let (gq, gk, gv) = scaled_dot_product_attention_backward(&grad_out, &q, &k, &v).unwrap();

        assert_eq!(gq.dims(), shape.as_slice(), "grad_q shape for {shape:?}");
        assert_eq!(gk.dims(), shape.as_slice(), "grad_k shape for {shape:?}");
        assert_eq!(gv.dims(), shape.as_slice(), "grad_v shape for {shape:?}");
    }
}

/// Rank validation: SDPA backward rejects non-4D inputs.
#[test]
fn test_sdpa_backward_rank_validation() {
    let t3d = DynTensor::ones(&[1, 2, 3], nn_core::DType::F32, &cpu()).unwrap();
    let t4d = DynTensor::ones(&[1, 1, 2, 3], nn_core::DType::F32, &cpu()).unwrap();

    // 3D query should fail.
    let result = scaled_dot_product_attention_backward(&t4d, &t3d, &t4d, &t4d);
    assert!(result.is_err());
}
