#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::{Linear, RmsNorm};
use crate::{DType, Device, Result};

/// Helper: create a Linear layer with given weights (no bias).
fn linear_no_bias(weight: &[f32], out_f: usize, in_f: usize) -> Linear {
    let w =
        DynTensor::from_vec(weight.to_vec(), &[out_f, in_f], &Device::Cpu).expect("linear weight");
    Linear::new(w, None).unwrap()
}

/// Helper: identity RmsNorm.
fn identity_rms_norm(dim: usize) -> RmsNorm {
    let w = DynTensor::ones(&[dim], DType::F32, &Device::Cpu).expect("ones");
    RmsNorm::new(w, 1e-6).expect("valid RmsNorm")
}

/// Helper: identity Module (passthrough).
fn identity_module() -> Box<dyn Module + Send + Sync> {
    Box::new(|x: &DynTensor| -> Result<DynTensor> { Ok(x.clone()) })
}

// -- DiTBlock tests -----------------------------------------------------------

#[test]
fn test_dit_block_forward_shape() -> Result<()> {
    let dim = 4;
    let cond_dim = 4;

    // AdaLN: cond_dim -> 3*dim = 12
    let proj1 = linear_no_bias(&vec![0.1; 12 * cond_dim], 12, cond_dim);
    let proj2 = linear_no_bias(&vec![0.1; 12 * cond_dim], 12, cond_dim);
    let adaln_attn = AdaLnZero::new(proj1, Box::new(identity_rms_norm(dim)), dim)?;
    let adaln_ffn = AdaLnZero::new(proj2, Box::new(identity_rms_norm(dim)), dim)?;

    let block = DiTBlock::new(adaln_attn, identity_module(), adaln_ffn, identity_module())?;

    let x = DynTensor::ones(&[2, 3, dim], DType::F32, &Device::Cpu)?;
    let cond = DynTensor::ones(&[2, 3, cond_dim], DType::F32, &Device::Cpu)?;

    let out = block.forward(&x, &cond)?;
    assert_eq!(out.dims(), &[2, 3, dim]);
    Ok(())
}

#[test]
fn test_dit_block_zero_gate_passthrough() -> Result<()> {
    // With zero projection weights: gate=0, so gated residual = x + 0*anything = x
    let dim = 4;
    let cond_dim = 2;

    // Zero projections -> scale=0, shift=0, gate=0
    let proj1 = linear_no_bias(&vec![0.0; 12 * cond_dim], 12, cond_dim);
    let proj2 = linear_no_bias(&vec![0.0; 12 * cond_dim], 12, cond_dim);
    let adaln_attn = AdaLnZero::new(proj1, Box::new(identity_rms_norm(dim)), dim)?;
    let adaln_ffn = AdaLnZero::new(proj2, Box::new(identity_rms_norm(dim)), dim)?;

    // Attention and FFN are identity — doesn't matter since gate=0
    let block = DiTBlock::new(adaln_attn, identity_module(), adaln_ffn, identity_module())?;

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, dim], &Device::Cpu)?;
    let cond = DynTensor::ones(&[1, cond_dim], DType::F32, &Device::Cpu)?;

    let out = block.forward(&x, &cond)?;
    let out_vals = out.to_flat_vec::<f32>()?;
    let x_vals = x.to_flat_vec::<f32>()?;

    // With zero gate, output should equal input
    for (i, (o, x_v)) in out_vals.iter().zip(x_vals.iter()).enumerate() {
        assert!(
            (o - x_v).abs() < 1e-5,
            "passthrough failed at {i}: expected {x_v}, got {o}"
        );
    }
    Ok(())
}

#[test]
fn test_dit_block_output_finite() -> Result<()> {
    let dim = 4;
    let cond_dim = 4;

    let proj1 = linear_no_bias(&vec![0.01; 12 * cond_dim], 12, cond_dim);
    let proj2 = linear_no_bias(&vec![0.01; 12 * cond_dim], 12, cond_dim);
    let adaln_attn = AdaLnZero::new(proj1, Box::new(identity_rms_norm(dim)), dim)?;
    let adaln_ffn = AdaLnZero::new(proj2, Box::new(identity_rms_norm(dim)), dim)?;

    let block = DiTBlock::new(adaln_attn, identity_module(), adaln_ffn, identity_module())?;

    let x = DynTensor::from_vec(vec![10.0, -10.0, 5.0, -5.0], &[1, dim], &Device::Cpu)?;
    let cond = DynTensor::from_vec(vec![50.0, -50.0, 25.0, -25.0], &[1, cond_dim], &Device::Cpu)?;

    let out = block.forward(&x, &cond)?;
    for (i, &v) in out.to_flat_vec::<f32>()?.iter().enumerate() {
        assert!(v.is_finite(), "element {i} not finite: {v}");
    }
    Ok(())
}

// -- DiTBlockDual tests -------------------------------------------------------

#[test]
fn test_dit_block_dual_forward_shape() -> Result<()> {
    let dim = 4;

    // Modulation: dim -> 6*dim = 24
    let modulation = linear_no_bias(&vec![0.1; 24 * dim], 24, dim);
    let adaln = AdaLnZeroDual::new(modulation, dim)?;

    let block = DiTBlockDual::new(
        adaln,
        identity_module(), // norm_attn
        identity_module(), // attn
        identity_module(), // norm_ffn
        identity_module(), // ffn
    )?;

    let x = DynTensor::ones(&[2, 3, dim], DType::F32, &Device::Cpu)?;
    let t_emb = DynTensor::ones(&[2, dim], DType::F32, &Device::Cpu)?;

    let out = block.forward(&x, &t_emb)?;
    assert_eq!(out.dims(), &[2, 3, dim]);
    Ok(())
}

#[test]
fn test_dit_block_dual_zero_gate_passthrough() -> Result<()> {
    let dim = 4;

    // Zero modulation -> all 6 params = 0, gate1 = gate2 = 0 -> passthrough
    let modulation = linear_no_bias(&vec![0.0; 24 * dim], 24, dim);
    let adaln = AdaLnZeroDual::new(modulation, dim)?;

    let block = DiTBlockDual::new(
        adaln,
        identity_module(),
        identity_module(),
        identity_module(),
        identity_module(),
    )?;

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, dim], &Device::Cpu)?;
    // Zero t_emb -> silu(0) = 0 -> projection = 0
    let t_emb = DynTensor::zeros(&[1, dim], DType::F32, &Device::Cpu)?;

    let out = block.forward(&x, &t_emb)?;
    let out_vals = out.to_flat_vec::<f32>()?;
    let x_vals = x.to_flat_vec::<f32>()?;

    for (i, (o, x_v)) in out_vals.iter().zip(x_vals.iter()).enumerate() {
        assert!(
            (o - x_v).abs() < 1e-5,
            "dual passthrough failed at {i}: expected {x_v}, got {o}"
        );
    }
    Ok(())
}

// -- NaN input tests (finiteness check, #1209) --------------------------------

#[test]
fn test_dit_block_nan_input_returns_error() -> Result<()> {
    // NaN in x is caught by either RmsNorm (inner layer) or check_output_finite.
    // Either way, the forward call returns an error — NaN does not silently pass.
    let dim = 4;
    let cond_dim = 4;

    let proj1 = linear_no_bias(&vec![0.1; 12 * cond_dim], 12, cond_dim);
    let proj2 = linear_no_bias(&vec![0.1; 12 * cond_dim], 12, cond_dim);
    let adaln_attn = AdaLnZero::new(proj1, Box::new(identity_rms_norm(dim)), dim)?;
    let adaln_ffn = AdaLnZero::new(proj2, Box::new(identity_rms_norm(dim)), dim)?;

    let block = DiTBlock::new(adaln_attn, identity_module(), adaln_ffn, identity_module())?;

    let x = DynTensor::from_vec(vec![f32::NAN, 1.0, 2.0, 3.0], &[1, dim], &Device::Cpu)?;
    let cond = DynTensor::ones(&[1, cond_dim], DType::F32, &Device::Cpu)?;

    let result = block.forward(&x, &cond);
    assert!(result.is_err(), "DiTBlock should reject NaN input, got Ok");
    Ok(())
}

#[test]
fn test_dit_block_dual_nan_input_returns_error() -> Result<()> {
    // NaN in x propagates through identity modules → gated residual → NaN output.
    // check_output_finite catches it at the DiTBlockDual boundary.
    let dim = 4;

    let modulation = linear_no_bias(&vec![0.1; 24 * dim], 24, dim);
    let adaln = AdaLnZeroDual::new(modulation, dim)?;

    let block = DiTBlockDual::new(
        adaln,
        identity_module(),
        identity_module(),
        identity_module(),
        identity_module(),
    )?;

    let x = DynTensor::from_vec(vec![f32::NAN, 1.0, 2.0, 3.0], &[1, dim], &Device::Cpu)?;
    let t_emb = DynTensor::ones(&[1, dim], DType::F32, &Device::Cpu)?;

    let result = block.forward(&x, &t_emb);
    assert!(
        result.is_err(),
        "DiTBlockDual should reject NaN input, got Ok"
    );
    // Verify it's a finiteness-related error
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(
        err_str.contains("finite") || err_str.contains("NaN") || err_str.contains("DiTBlockDual"),
        "expected finiteness error, got: {err_str}"
    );
    Ok(())
}
