#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::{Linear, RmsNorm};
use crate::{Device, Result, TensorError};

/// Helper: create a Linear layer with given weights (no bias).
fn linear_no_bias(weight: &[f32], out_f: usize, in_f: usize) -> Linear {
    let w =
        DynTensor::from_vec(weight.to_vec(), &[out_f, in_f], &Device::Cpu).expect("linear weight");
    Linear::new(w, None).unwrap()
}

/// Helper: create an identity RmsNorm (weight = ones, small eps).
fn identity_rms_norm(dim: usize) -> RmsNorm {
    let w = DynTensor::ones(&[dim], crate::DType::F32, &Device::Cpu).expect("ones");
    RmsNorm::new(w, 1e-6).expect("valid RmsNorm")
}

// -- AdaLnZero tests ----------------------------------------------------------

#[test]
fn test_adaln_zero_forward_shape() -> Result<()> {
    let dim = 4;
    let cond_dim = 8;

    // Projection: cond_dim -> 3 * dim = 12
    let proj = linear_no_bias(&vec![0.1; 12 * cond_dim], 12, cond_dim);
    let norm = Box::new(identity_rms_norm(dim));
    let adaln = AdaLnZero::new(proj, norm, dim)?;

    let x = DynTensor::ones(&[2, 3, dim], crate::DType::F32, &Device::Cpu)?;
    let cond = DynTensor::ones(&[2, 3, cond_dim], crate::DType::F32, &Device::Cpu)?;

    let (modulated, gate) = adaln.forward(&x, &cond)?;

    // Output shapes should match input x
    assert_eq!(modulated.dims(), &[2, 3, dim]);
    assert_eq!(gate.dims(), &[2, 3, dim]);
    Ok(())
}

#[test]
fn test_adaln_zero_scale_identity() -> Result<()> {
    // When scale=0 and shift=0, modulated should equal norm(x)
    let dim = 4;
    let cond_dim = 2;

    // Zero projection weights -> scale=0, shift=0, gate=0
    let proj = linear_no_bias(&vec![0.0; 12 * cond_dim], 12, cond_dim);
    let norm = Box::new(identity_rms_norm(dim));
    let adaln = AdaLnZero::new(proj, norm, dim)?;

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, dim], &Device::Cpu)?;
    let cond = DynTensor::ones(&[1, cond_dim], crate::DType::F32, &Device::Cpu)?;

    let (modulated, gate) = adaln.forward(&x, &cond)?;

    // With zero projection: scale=0, so (1+0)*norm(x) + 0 = norm(x)
    // RmsNorm(x) = x / rms(x) * weight, weight=ones
    // rms = sqrt(mean(x^2) + eps) = sqrt((1+4+9+16)/4 + 1e-6) = sqrt(7.5)
    let rms = (30.0_f32 / 4.0).sqrt();
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0].iter().map(|&v| v / rms).collect();

    let modulated_vals = modulated.to_flat_vec::<f32>()?;
    for (actual, exp) in modulated_vals.iter().zip(expected.iter()) {
        assert!(
            (actual - exp).abs() < 1e-5,
            "identity modulation: expected {exp}, got {actual}"
        );
    }

    // Gate should be all zeros
    let gate_vals = gate.to_flat_vec::<f32>()?;
    for v in &gate_vals {
        assert!(v.abs() < 1e-6, "zero gate expected, got {v}");
    }
    Ok(())
}

#[test]
fn test_adaln_zero_modulation_values() -> Result<()> {
    // Verify modulation: normed * (1 + scale) + shift
    let dim = 2;
    let cond_dim = 1;

    // Build projection that maps [1.0] -> [scale0, scale1, shift0, shift1, gate0, gate1]
    // We want scale=[0.5, 0.5], shift=[1.0, 1.0], gate=[0.0, 0.0]
    // proj weights: [6, 1], proj([1]) = weights column
    let proj_weights = vec![0.5, 0.5, 1.0, 1.0, 0.0, 0.0];
    let proj = linear_no_bias(&proj_weights, 6, cond_dim);

    // Use a no-op norm (just passes through). Build as closure.
    let norm: Box<dyn Module + Send + Sync> =
        Box::new(|x: &DynTensor| -> Result<DynTensor> { Ok(x.clone()) });
    let adaln = AdaLnZero::new(proj, norm, dim)?;

    let x = DynTensor::from_vec(vec![2.0, 4.0], &[1, dim], &Device::Cpu)?;
    let cond = DynTensor::from_vec(vec![1.0], &[1, cond_dim], &Device::Cpu)?;

    let (modulated, gate) = adaln.forward(&x, &cond)?;
    let mod_vals = modulated.to_flat_vec::<f32>()?;

    // modulated = x * (1 + 0.5) + 1.0 = x * 1.5 + 1.0
    // [2.0 * 1.5 + 1.0, 4.0 * 1.5 + 1.0] = [4.0, 7.0]
    assert!((mod_vals[0] - 4.0).abs() < 1e-5, "got {}", mod_vals[0]);
    assert!((mod_vals[1] - 7.0).abs() < 1e-5, "got {}", mod_vals[1]);

    let gate_vals = gate.to_flat_vec::<f32>()?;
    assert!(gate_vals[0].abs() < 1e-6);
    assert!(gate_vals[1].abs() < 1e-6);
    Ok(())
}

// -- AdaLnZeroDual tests ------------------------------------------------------

#[test]
fn test_adaln_dual_forward_shape() -> Result<()> {
    let dim = 4;

    // Modulation: dim -> 6 * dim = 24
    let modulation = linear_no_bias(&vec![0.1; 24 * dim], 24, dim);
    let dual = AdaLnZeroDual::new(modulation, dim)?;

    let t_emb = DynTensor::ones(&[2, dim], crate::DType::F32, &Device::Cpu)?;
    let params = dual.forward(&t_emb)?;

    // All 6 params should have shape [2, dim]
    assert_eq!(params.shift1.dims(), &[2, dim]);
    assert_eq!(params.scale1.dims(), &[2, dim]);
    assert_eq!(params.gate1.dims(), &[2, dim]);
    assert_eq!(params.shift2.dims(), &[2, dim]);
    assert_eq!(params.scale2.dims(), &[2, dim]);
    assert_eq!(params.gate2.dims(), &[2, dim]);
    Ok(())
}

#[test]
fn test_adaln_dual_six_chunks_independent() -> Result<()> {
    let dim = 2;

    // Identity-ish modulation: weight is identity-block diagonal
    // silu(ones) ≈ [0.731, 0.731] for each element
    // We use a weight that produces distinct values per chunk
    let mut weights = vec![0.0f32; 12 * dim];
    // Split order: scale, shift, gate per stream
    // Row 0 (scale1[0]): col 0 = 1.0
    weights[0] = 1.0;
    // Row 1 (scale1[1]): col 1 = 1.0
    weights[dim + 1] = 1.0;
    // Row 2 (shift1[0]): col 0 = 2.0
    weights[2 * dim] = 2.0;
    // Row 3 (shift1[1]): col 1 = 2.0
    weights[3 * dim + 1] = 2.0;
    // Row 4 (gate1[0]): col 0 = 3.0
    weights[4 * dim] = 3.0;
    // Row 5 (gate1[1]): col 1 = 3.0
    weights[5 * dim + 1] = 3.0;
    // Row 6 (scale2[0]): col 0 = 4.0
    weights[6 * dim] = 4.0;
    // Row 7 (scale2[1]): col 1 = 4.0
    weights[7 * dim + 1] = 4.0;
    // Row 8 (shift2[0]): col 0 = 5.0
    weights[8 * dim] = 5.0;
    // Row 9 (shift2[1]): col 1 = 5.0
    weights[9 * dim + 1] = 5.0;
    // Row 10 (gate2[0]): col 0 = 6.0
    weights[10 * dim] = 6.0;
    // Row 11 (gate2[1]): col 1 = 6.0
    weights[11 * dim + 1] = 6.0;

    let modulation = linear_no_bias(&weights, 12, dim);
    let dual = AdaLnZeroDual::new(modulation, dim)?;

    let t_emb = DynTensor::ones(&[1, dim], crate::DType::F32, &Device::Cpu)?;
    let params = dual.forward(&t_emb)?;

    // silu(1.0) ≈ 0.7310586
    let s = 0.7310586_f32;

    let check = |name: &str, tensor: &DynTensor, expected_factor: f32| -> Result<()> {
        let vals = tensor.to_flat_vec::<f32>()?;
        let expected = s * expected_factor;
        assert!(
            (vals[0] - expected).abs() < 1e-4,
            "{name}[0]: expected {expected}, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - expected).abs() < 1e-4,
            "{name}[1]: expected {expected}, got {}",
            vals[1]
        );
        Ok(())
    };

    // Split order: scale, shift, gate per stream (offsets 0, d, 2d, 3d, 4d, 5d)
    check("scale1", &params.scale1, 1.0)?;
    check("shift1", &params.shift1, 2.0)?;
    check("gate1", &params.gate1, 3.0)?;
    check("scale2", &params.scale2, 4.0)?;
    check("shift2", &params.shift2, 5.0)?;
    check("gate2", &params.gate2, 6.0)?;
    Ok(())
}

// -- LowRankAdaLn tests -------------------------------------------------------

#[test]
fn test_low_rank_adaln_forward_shape() -> Result<()> {
    let dim = 4;
    let cond_dim = 8;
    let rank = 2;

    let down = linear_no_bias(&vec![0.1; rank * cond_dim], rank, cond_dim);
    let up = linear_no_bias(&vec![0.1; (3 * dim) * rank], 3 * dim, rank);
    let norm = Box::new(identity_rms_norm(dim));
    let adaln = LowRankAdaLn::new(down, up, norm, dim)?;

    let x = DynTensor::ones(&[2, 3, dim], crate::DType::F32, &Device::Cpu)?;
    let cond = DynTensor::ones(&[2, 3, cond_dim], crate::DType::F32, &Device::Cpu)?;

    let (modulated, gate) = adaln.forward(&x, &cond)?;
    assert_eq!(modulated.dims(), &[2, 3, dim]);
    assert_eq!(gate.dims(), &[2, 3, dim]);
    Ok(())
}

#[test]
fn test_low_rank_adaln_bottleneck_identity() -> Result<()> {
    // With zero weights in down/up projections, should get identity modulation
    let dim = 4;
    let cond_dim = 4;
    let rank = 2;

    let down = linear_no_bias(&vec![0.0; rank * cond_dim], rank, cond_dim);
    let up = linear_no_bias(&vec![0.0; (3 * dim) * rank], 3 * dim, rank);
    let norm = Box::new(identity_rms_norm(dim));
    let adaln = LowRankAdaLn::new(down, up, norm, dim)?;

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, dim], &Device::Cpu)?;
    let cond = DynTensor::ones(&[1, cond_dim], crate::DType::F32, &Device::Cpu)?;

    let (modulated, gate) = adaln.forward(&x, &cond)?;

    // Zero projection -> scale=0, shift=0 -> modulated = norm(x)
    let rms = (30.0_f32 / 4.0).sqrt();
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0].iter().map(|&v| v / rms).collect();
    let mod_vals = modulated.to_flat_vec::<f32>()?;
    for (actual, exp) in mod_vals.iter().zip(expected.iter()) {
        assert!(
            (actual - exp).abs() < 1e-5,
            "low-rank identity: expected {exp}, got {actual}"
        );
    }

    let gate_vals = gate.to_flat_vec::<f32>()?;
    for v in &gate_vals {
        assert!(v.abs() < 1e-6, "zero gate expected, got {v}");
    }
    Ok(())
}

// -- apply_adaln_modulation tests ---------------------------------------------

#[test]
fn test_apply_adaln_modulation_basic() -> Result<()> {
    let normed = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &Device::Cpu)?;
    let scale = DynTensor::from_vec(vec![0.0, 1.0, -0.5], &[1, 3], &Device::Cpu)?;
    let shift = DynTensor::from_vec(vec![0.0, 0.0, 10.0], &[1, 3], &Device::Cpu)?;

    let result = apply_adaln_modulation(&normed, &scale, &shift)?;
    let vals = result.to_flat_vec::<f32>()?;

    // normed * (1 + scale) + shift
    // [1*(1+0)+0, 2*(1+1)+0, 3*(1-0.5)+10] = [1.0, 4.0, 11.5]
    assert!((vals[0] - 1.0).abs() < 1e-5);
    assert!((vals[1] - 4.0).abs() < 1e-5);
    assert!((vals[2] - 11.5).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_adaln_dual_zero_init_produces_identity() -> Result<()> {
    // With zero-initialized projection weights, SiLU(0)=0, proj(0)=0
    // All 6 params should be zero: scale=0 means (1+0)=1 (identity),
    // shift=0 means no offset, gate=0 means no sub-block contribution.
    let dim = 4;
    let modulation = linear_no_bias(&vec![0.0; 24 * dim], 24, dim);
    let dual = AdaLnZeroDual::new(modulation, dim)?;

    // Zero input -> silu(0) = 0 -> projection = 0
    let t_emb = DynTensor::zeros(&[1, dim], crate::DType::F32, &Device::Cpu)?;
    let params = dual.forward(&t_emb)?;

    let check_zero = |name: &str, t: &DynTensor| -> Result<()> {
        let vals = t.to_flat_vec::<f32>()?;
        for (i, v) in vals.iter().enumerate() {
            assert!(v.abs() < 1e-6, "{name}[{i}] should be 0, got {v}");
        }
        Ok(())
    };

    check_zero("shift1", &params.shift1)?;
    check_zero("scale1", &params.scale1)?;
    check_zero("gate1", &params.gate1)?;
    check_zero("shift2", &params.shift2)?;
    check_zero("scale2", &params.scale2)?;
    check_zero("gate2", &params.gate2)?;
    Ok(())
}

// -- Error path tests (dim=0 rejection) ---------------------------------------

#[test]
fn test_adaln_zero_dim_zero_returns_err() {
    let proj = linear_no_bias(&[0.0; 6], 3, 2);
    let norm: Box<dyn Module + Send + Sync> =
        Box::new(|x: &DynTensor| -> Result<DynTensor> { Ok(x.clone()) });
    let err = match AdaLnZero::new(proj, norm, 0) {
        Err(e) => e,
        Ok(_) => panic!("expected Err for dim=0"),
    };
    assert!(
        matches!(err, TensorError::InvalidShape(ref msg) if msg.contains("dim must be > 0")),
        "unexpected error: {err}"
    );
}

#[test]
fn test_adaln_dual_dim_zero_returns_err() {
    let modulation = linear_no_bias(&[0.0; 12], 6, 2);
    let err = match AdaLnZeroDual::new(modulation, 0) {
        Err(e) => e,
        Ok(_) => panic!("expected Err for dim=0"),
    };
    assert!(
        matches!(err, TensorError::InvalidShape(ref msg) if msg.contains("dim must be > 0")),
        "unexpected error: {err}"
    );
}

#[test]
fn test_low_rank_adaln_dim_zero_returns_err() {
    let down = linear_no_bias(&[0.0; 4], 2, 2);
    let up = linear_no_bias(&[0.0; 6], 3, 2);
    let norm: Box<dyn Module + Send + Sync> =
        Box::new(|x: &DynTensor| -> Result<DynTensor> { Ok(x.clone()) });
    let err = match LowRankAdaLn::new(down, up, norm, 0) {
        Err(e) => e,
        Ok(_) => panic!("expected Err for dim=0"),
    };
    assert!(
        matches!(err, TensorError::InvalidShape(ref msg) if msg.contains("dim must be > 0")),
        "unexpected error: {err}"
    );
}

#[test]
fn test_apply_adaln_modulation_output_finite() -> Result<()> {
    let normed = DynTensor::from_vec(vec![100.0, -100.0, 50.0], &[1, 3], &Device::Cpu)?;
    let scale = DynTensor::from_vec(vec![10.0, -0.99, 0.0], &[1, 3], &Device::Cpu)?;
    let shift = DynTensor::from_vec(vec![-500.0, 500.0, 0.0], &[1, 3], &Device::Cpu)?;
    let result = apply_adaln_modulation(&normed, &scale, &shift)?;
    let vals = result.to_flat_vec::<f32>()?;
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "element {i} not finite: {v}");
    }
    // normed * (1 + scale) + shift = [100*11-500, -100*0.01+500, 50*1+0]
    assert!((vals[0] - 600.0).abs() < 1e-3, "got {}", vals[0]);
    assert!((vals[1] - 499.0).abs() < 1e-3, "got {}", vals[1]);
    assert!((vals[2] - 50.0).abs() < 1e-3, "got {}", vals[2]);
    Ok(())
}

#[test]
fn test_adaln_dual_output_all_finite() -> Result<()> {
    let dim = 4;
    let modulation = linear_no_bias(&vec![0.1; 24 * dim], 24, dim);
    let dual = AdaLnZeroDual::new(modulation, dim)?;
    let t_emb = DynTensor::from_vec(vec![50.0, -50.0, 25.0, -25.0], &[1, dim], &Device::Cpu)?;
    let params = dual.forward(&t_emb)?;
    for (name, t) in [
        ("scale1", &params.scale1),
        ("shift1", &params.shift1),
        ("gate1", &params.gate1),
        ("scale2", &params.scale2),
        ("shift2", &params.shift2),
        ("gate2", &params.gate2),
    ] {
        for (i, &v) in t.to_flat_vec::<f32>()?.iter().enumerate() {
            assert!(v.is_finite(), "{name}[{i}] not finite: {v}");
        }
    }
    Ok(())
}

#[test]
fn test_adaln_zero_accessor() -> Result<()> {
    let proj = linear_no_bias(&vec![0.1; 12 * 8], 12, 8);
    let norm = Box::new(identity_rms_norm(4));
    let adaln = AdaLnZero::new(proj, norm, 4)?;
    assert_eq!(adaln.dim(), 4);
    Ok(())
}

#[test]
fn test_adaln_dual_accessor() -> Result<()> {
    let modulation = linear_no_bias(&vec![0.1; 24 * 4], 24, 4);
    let dual = AdaLnZeroDual::new(modulation, 4)?;
    assert_eq!(dual.dim(), 4);
    Ok(())
}

#[test]
fn test_low_rank_adaln_accessor() -> Result<()> {
    let down = linear_no_bias(&[0.1; 2 * 8], 2, 8);
    let up = linear_no_bias(&[0.1; 12 * 2], 12, 2);
    let norm = Box::new(identity_rms_norm(4));
    let adaln = LowRankAdaLn::new(down, up, norm, 4)?;
    assert_eq!(adaln.dim(), 4);
    Ok(())
}
