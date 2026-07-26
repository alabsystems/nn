#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Mathematical property and finiteness tests for [`JointAttention`].
//!
//! Extracted from `joint_attention_tests.rs` — tests convex combination bounds,
//! dominant key concentration, scale factor effects, uniform score averaging,
//! large input finiteness, and NaN input rejection.

use crate::layers::{JointAttention, Linear};
use crate::{Device, DynTensor, Result};

/// Helper: create an identity Linear(dim, dim) with no bias.
fn identity_linear(dim: usize) -> Linear {
    let mut data = vec![0.0f32; dim * dim];
    for i in 0..dim {
        data[i * dim + i] = 1.0;
    }
    let w = DynTensor::new(&data, &[dim, dim], &Device::Cpu).unwrap();
    Linear::new(w, None).unwrap()
}

// -- Test 7: Output bounded by value range (convex combination property). -----
// Attention output at each position is a convex combination of V vectors,
// so each output element lies in [min(V column), max(V column)].
#[test]
fn test_attention_output_bounded_by_values() -> Result<()> {
    let dim = 4;
    let attn = JointAttention::new(
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        1,
        dim,
    )?;
    let x = DynTensor::new(
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        &[1, 2, 4],
        &Device::Cpu,
    )?;
    let ctx = DynTensor::new(
        &[0.2, 0.5, 0.1, 0.8, 0.9, 0.1, 0.7, 0.3, 0.4, 0.8, 0.3, 0.0],
        &[1, 3, 4],
        &Device::Cpu,
    )?;
    let out = attn.forward_single_ctx(&x, &ctx)?;
    let out_vals = out.to_flat_vec::<f32>()?;
    // KV = [x, ctx] = 5 tokens
    let all_kv: &[f32] = &[
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.5, 0.1, 0.8, 0.9, 0.1, 0.7, 0.3, 0.4, 0.8,
        0.3, 0.0,
    ];
    for q in 0..2 {
        for d in 0..dim {
            let out_val = out_vals[q * dim + d];
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for t in 0..5 {
                lo = lo.min(all_kv[t * dim + d]);
                hi = hi.max(all_kv[t * dim + d]);
            }
            assert!(
                out_val >= lo - 1e-5 && out_val <= hi + 1e-5,
                "output[q={q}, d={d}] = {out_val} outside V range [{lo}, {hi}]"
            );
        }
    }
    Ok(())
}

// -- Test 8: Dominant key match concentrates attention on matching token. ------
#[test]
fn test_attention_dominant_key_concentrates_output() -> Result<()> {
    let dim = 4;
    let attn = JointAttention::new(
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        1,
        dim,
    )?;
    // Q=[5,0,0,0]: strong signal in dim 0
    let x = DynTensor::new(&[5.0, 0.0, 0.0, 0.0], &[1, 1, 4], &Device::Cpu)?;
    let ctx = DynTensor::new(
        &[5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0], // matching + orthogonal
        &[1, 2, 4],
        &Device::Cpu,
    )?;
    let out = attn.forward_single_ctx(&x, &ctx)?;
    let vals = out.to_flat_vec::<f32>()?;
    // KV = [x, ctx0, ctx1]. x and ctx0 match Q; ctx1 is orthogonal.
    // scores = [12.5, 12.5, 0] → softmax ≈ [0.5, 0.5, ~0]
    // Output ≈ [5, ~0, 0, 0]
    assert!(vals[0] > 4.0, "dominant dim preserved: got {}", vals[0]);
    assert!(vals[1] < 0.5, "orthogonal dim suppressed: got {}", vals[1]);
    Ok(())
}

// -- Test 9: Scale factor effect — higher 1/sqrt(d) produces peakier attention.
#[test]
fn test_attention_scale_factor_effect() -> Result<()> {
    // head_dim=2: scale=1/sqrt(2)≈0.707 vs head_dim=4: scale=0.5
    let attn_a = JointAttention::new(
        identity_linear(2),
        identity_linear(2),
        identity_linear(2),
        identity_linear(2),
        1,
        2,
    )?;
    let attn_b = JointAttention::new(
        identity_linear(4),
        identity_linear(4),
        identity_linear(4),
        identity_linear(4),
        1,
        4,
    )?;
    let x_a = DynTensor::new(&[1.0, 0.0], &[1, 1, 2], &Device::Cpu)?;
    let ctx_a = DynTensor::new(&[0.0, 1.0], &[1, 1, 2], &Device::Cpu)?;
    let out_a = attn_a
        .forward_single_ctx(&x_a, &ctx_a)?
        .to_flat_vec::<f32>()?;

    let x_b = DynTensor::new(&[1.0, 0.0, 0.0, 0.0], &[1, 1, 4], &Device::Cpu)?;
    let ctx_b = DynTensor::new(&[0.0, 1.0, 0.0, 0.0], &[1, 1, 4], &Device::Cpu)?;
    let out_b = attn_b
        .forward_single_ctx(&x_b, &ctx_b)?
        .to_flat_vec::<f32>()?;
    // Higher scale → more peaked → self-token gets more weight → out[0] closer to 1
    assert!(
        out_a[0] > out_b[0],
        "dim2={} should > dim4={}",
        out_a[0],
        out_b[0]
    );
    Ok(())
}

// -- Test 10: Output finiteness for large input magnitudes. -------------------
#[test]
fn test_attention_output_finite_for_large_inputs() -> Result<()> {
    let dim = 4;
    let attn = JointAttention::new(
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        2,
        dim,
    )?;
    let x = DynTensor::new(&[100.0, -100.0, 50.0, -50.0], &[1, 1, 4], &Device::Cpu)?;
    let ctx = DynTensor::new(
        &[80.0, -80.0, 40.0, -40.0, 0.01, -0.01, 0.005, -0.005],
        &[1, 2, 4],
        &Device::Cpu,
    )?;
    let out = attn.forward_single_ctx(&x, &ctx)?;
    for (i, &v) in out.to_flat_vec::<f32>()?.iter().enumerate() {
        assert!(v.is_finite(), "element {i} not finite: {v}");
    }
    Ok(())
}

// -- Test 11: Uniform scores → output = mean of V (softmax 1/N property). ----
#[test]
fn test_attention_uniform_scores_yield_mean_output() -> Result<()> {
    let dim = 4;
    let attn = JointAttention::new(
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        1,
        dim,
    )?;
    // All tokens identical [1,1,1,1] → all scores equal → uniform weights
    let x = DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 4], &Device::Cpu)?;
    let ctx = DynTensor::new(
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[1, 2, 4],
        &Device::Cpu,
    )?;
    let out = attn.forward_single_ctx(&x, &ctx)?;
    for (i, &v) in out.to_flat_vec::<f32>()?.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-4,
            "element {i}: expected ~1.0, got {v}"
        );
    }
    Ok(())
}

// -- Finiteness validation (#1202) --------------------------------------------

#[test]
fn test_joint_attention_nan_input_returns_error() -> Result<()> {
    let dim = 4;
    let attn = JointAttention::new(
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        2,
        dim,
    )?;
    let mut data = vec![0.5f32; 4];
    data[0] = f32::NAN;
    let x = DynTensor::new(&data, &[1, 1, 4], &Device::Cpu)?;
    let ctx = DynTensor::new(&[1.0; 4], &[1, 1, 4], &Device::Cpu)?;
    let result = attn.forward_single_ctx(&x, &ctx);
    assert!(result.is_err(), "NaN input should produce an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Non-finite") || msg.contains("NaN"),
        "error should mention non-finite: {msg}"
    );
    Ok(())
}
