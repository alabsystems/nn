#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::Linear;
use crate::{DType, Device, Result, TensorError};

/// Helper: create a Linear layer with given weights (no bias).
fn linear_no_bias(weight: &[f32], out_f: usize, in_f: usize) -> Linear {
    let w =
        DynTensor::from_vec(weight.to_vec(), &[out_f, in_f], &Device::Cpu).expect("linear weight");
    Linear::new(w, None).unwrap()
}

/// Helper: create a Linear layer with given weights and bias.
fn linear_with_bias(weight: &[f32], bias: &[f32], out_f: usize, in_f: usize) -> Linear {
    let w =
        DynTensor::from_vec(weight.to_vec(), &[out_f, in_f], &Device::Cpu).expect("linear weight");
    let b = DynTensor::from_vec(bias.to_vec(), &[out_f], &Device::Cpu).expect("linear bias");
    Linear::new(w, Some(b)).unwrap()
}

#[test]
fn test_swiglu_forward_shape() -> Result<()> {
    let dim = 4;
    let ff_dim = 16;

    let w_gate = linear_no_bias(&vec![0.1; ff_dim * dim], ff_dim, dim);
    let w_up = linear_no_bias(&vec![0.1; ff_dim * dim], ff_dim, dim);
    let w_down = linear_no_bias(&vec![0.1; dim * ff_dim], dim, ff_dim);
    let swiglu = SwiGlu::new(w_gate, w_up, w_down).unwrap();

    let x = DynTensor::ones(&[2, 3, dim], DType::F32, &Device::Cpu)?;
    let out = swiglu.forward(&x)?;

    assert_eq!(out.dims(), &[2, 3, dim]);
    Ok(())
}

#[test]
fn test_swiglu_gating_range() -> Result<()> {
    // SiLU saturates to ~x for large positive, ~0 for large negative.
    // With identity-like gate weights and zero up weights, output should be near zero.
    let dim = 2;
    let ff_dim = 2;

    // Gate: identity mapping
    let w_gate = linear_no_bias(&[1.0, 0.0, 0.0, 1.0], ff_dim, dim);
    // Up: zeros -> output should be zero regardless of gate
    let w_up = linear_no_bias(&vec![0.0; ff_dim * dim], ff_dim, dim);
    let w_down = linear_no_bias(&[1.0, 0.0, 0.0, 1.0], dim, ff_dim);
    let swiglu = SwiGlu::new(w_gate, w_up, w_down).unwrap();

    let x = DynTensor::from_vec(vec![5.0, -5.0], &[1, dim], &Device::Cpu)?;
    let out = swiglu.forward(&x)?;
    let vals = out.to_flat_vec::<f32>()?;

    // silu(gate(x)) * 0 = 0
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.abs() < 1e-6, "element {i} should be ~0, got {v}");
    }
    Ok(())
}

#[test]
fn test_swiglu_known_values() -> Result<()> {
    // Manual computation: dim=2, ff_dim=2
    // gate_w = identity, up_w = identity, down_w = identity
    // x = [1.0, 2.0]
    // gate(x) = silu([1.0, 2.0]) = [0.7310586, 1.7615942]
    // up(x) = [1.0, 2.0]
    // h = gate * up = [0.7310586, 3.5231884]
    // out = down(h) = [0.7310586, 3.5231884]
    let dim = 2;
    let ff_dim = 2;

    let identity = [1.0, 0.0, 0.0, 1.0];
    let w_gate = linear_no_bias(&identity, ff_dim, dim);
    let w_up = linear_no_bias(&identity, ff_dim, dim);
    let w_down = linear_no_bias(&identity, dim, ff_dim);
    let swiglu = SwiGlu::new(w_gate, w_up, w_down).unwrap();

    let x = DynTensor::from_vec(vec![1.0, 2.0], &[1, dim], &Device::Cpu)?;
    let out = swiglu.forward(&x)?;
    let vals = out.to_flat_vec::<f32>()?;

    // silu(1.0) = 1.0 * sigmoid(1.0) = 0.7310586
    // silu(2.0) = 2.0 * sigmoid(2.0) = 1.7615942
    let expected_0 = 0.7310586_f32 * 1.0; // gate * up
    let expected_1 = 1.7615942_f32 * 2.0;

    assert!(
        (vals[0] - expected_0).abs() < 1e-4,
        "expected {expected_0}, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - expected_1).abs() < 1e-4,
        "expected {expected_1}, got {}",
        vals[1]
    );
    Ok(())
}

#[test]
fn test_swiglu_with_bias() -> Result<()> {
    // Verify SwiGlu works with biased Linear layers (CosyVoice3 pattern).
    let dim = 2;
    let ff_dim = 2;

    let identity = [1.0, 0.0, 0.0, 1.0];
    let bias = [0.5, -0.5];
    let w_gate = linear_with_bias(&identity, &bias, ff_dim, dim);
    let w_up = linear_no_bias(&identity, ff_dim, dim);
    let w_down = linear_no_bias(&identity, dim, ff_dim);
    let swiglu = SwiGlu::new(w_gate, w_up, w_down).unwrap();

    let x = DynTensor::from_vec(vec![1.0, 1.0], &[1, dim], &Device::Cpu)?;
    let out = swiglu.forward(&x)?;
    let vals = out.to_flat_vec::<f32>()?;

    // gate_input = [1.0+0.5, 1.0-0.5] = [1.5, 0.5]
    // silu(1.5) ≈ 1.5 * sigmoid(1.5) ≈ 1.5 * 0.8175745 ≈ 1.2263618
    // silu(0.5) ≈ 0.5 * sigmoid(0.5) ≈ 0.5 * 0.6224593 ≈ 0.3112297
    // up = [1.0, 1.0]
    // h = [1.2263618, 0.3112297]
    // out = h (identity down)
    let silu_1_5 = 1.5 * (1.0 / (1.0 + (-1.5_f32).exp()));
    let silu_0_5 = 0.5 * (1.0 / (1.0 + (-0.5_f32).exp()));

    assert!(
        (vals[0] - silu_1_5).abs() < 1e-4,
        "expected {silu_1_5}, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - silu_0_5).abs() < 1e-4,
        "expected {silu_0_5}, got {}",
        vals[1]
    );
    Ok(())
}

#[test]
fn test_swiglu_output_finite() -> Result<()> {
    let dim = 4;
    let ff_dim = 8;

    let w_gate = linear_no_bias(&vec![0.1; ff_dim * dim], ff_dim, dim);
    let w_up = linear_no_bias(&vec![0.1; ff_dim * dim], ff_dim, dim);
    let w_down = linear_no_bias(&vec![0.1; dim * ff_dim], dim, ff_dim);
    let swiglu = SwiGlu::new(w_gate, w_up, w_down).unwrap();

    // Use large-ish values to test finiteness
    let x = DynTensor::from_vec(vec![50.0, -50.0, 25.0, -25.0], &[1, dim], &Device::Cpu)?;
    let out = swiglu.forward(&x)?;
    let vals = out.to_flat_vec::<f32>()?;

    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "element {i} not finite: {v}");
    }
    Ok(())
}

#[test]
fn test_swiglu_accessors() {
    let dim = 4;
    let ff_dim = 8;
    let w_gate = linear_no_bias(&vec![0.1; ff_dim * dim], ff_dim, dim);
    let w_up = linear_no_bias(&vec![0.1; ff_dim * dim], ff_dim, dim);
    let w_down = linear_no_bias(&vec![0.1; dim * ff_dim], dim, ff_dim);
    let swiglu = SwiGlu::new(w_gate, w_up, w_down).unwrap();

    // Verify accessors return the right shapes
    assert_eq!(swiglu.w_gate().weight().dims(), &[ff_dim, dim]);
    assert_eq!(swiglu.w_up().weight().dims(), &[ff_dim, dim]);
    assert_eq!(swiglu.w_down().weight().dims(), &[dim, ff_dim]);
}

#[test]
fn test_swiglu_batch_dims() -> Result<()> {
    // SwiGlu should work with 2D input [B, dim] and 3D [B, S, dim]
    let dim = 4;
    let ff_dim = 8;
    let w_gate = linear_no_bias(&vec![0.1; ff_dim * dim], ff_dim, dim);
    let w_up = linear_no_bias(&vec![0.1; ff_dim * dim], ff_dim, dim);
    let w_down = linear_no_bias(&vec![0.1; dim * ff_dim], dim, ff_dim);
    let swiglu = SwiGlu::new(w_gate, w_up, w_down).unwrap();

    // 2D input
    let x2d = DynTensor::ones(&[3, dim], DType::F32, &Device::Cpu)?;
    let out2d = swiglu.forward(&x2d)?;
    assert_eq!(out2d.dims(), &[3, dim]);

    // 3D input
    let x3d = DynTensor::ones(&[2, 5, dim], DType::F32, &Device::Cpu)?;
    let out3d = swiglu.forward(&x3d)?;
    assert_eq!(out3d.dims(), &[2, 5, dim]);
    Ok(())
}

#[test]
fn test_swiglu_nan_input_returns_error() {
    // NaN input propagates through Linear → silu → mul → Linear → NaN output.
    // check_output_finite should catch it and return NonFiniteData.
    let dim = 2;
    let ff_dim = 2;
    let identity = [1.0, 0.0, 0.0, 1.0];
    let w_gate = linear_no_bias(&identity, ff_dim, dim);
    let w_up = linear_no_bias(&identity, ff_dim, dim);
    let w_down = linear_no_bias(&identity, dim, ff_dim);
    let swiglu = SwiGlu::new(w_gate, w_up, w_down).unwrap();

    let x = DynTensor::from_vec(vec![f32::NAN, 1.0], &[1, dim], &Device::Cpu).unwrap();
    let result = swiglu.forward(&x);
    assert!(result.is_err());
    match result.unwrap_err() {
        TensorError::NonFiniteData { name, count } => {
            assert!(
                name.contains("SwiGlu"),
                "expected SwiGlu in name, got {name}"
            );
            assert!(count > 0, "expected non-zero count");
        }
        other => panic!("expected NonFiniteData, got {other:?}"),
    }
}
