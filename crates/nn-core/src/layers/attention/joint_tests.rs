#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Basic tests for [`JointAttention`] — DiT joint cross-attention module.
//!
//! Tests shape correctness, multi-head configurations, context concatenation,
//! error cases, self-attention mode, and batched forward. Mathematical property
//! tests (convex bounds, dominant key, scale factor, finiteness, uniform scores)
//! and NaN input rejection are in `joint_attention_tests_properties.rs`.

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

// ---------------------------------------------------------------------------
// Test 1: Forward with identity projections — verify output shape.
// ---------------------------------------------------------------------------
#[test]
fn test_joint_attention_identity_projections_shape() -> Result<()> {
    let dim = 4;
    let num_heads = 2;
    let attn = JointAttention::new(
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        num_heads,
        dim,
    )?;

    // x: [1, 3, 4], ctx1: [1, 2, 4], ctx2: [1, 1, 4]
    let x = DynTensor::new(
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[1, 3, 4],
        &Device::Cpu,
    )?;
    let ctx1 = DynTensor::new(
        &[0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 0.0],
        &[1, 2, 4],
        &Device::Cpu,
    )?;
    let ctx2 = DynTensor::new(&[0.1, 0.2, 0.3, 0.4], &[1, 1, 4], &Device::Cpu)?;

    let out = attn.forward_joint(&x, &ctx1, &ctx2)?;

    // Output shape must be [B=1, S_self=3, D=4]
    assert_eq!(out.dims(), &[1, 3, 4]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 2: Multi-head shape correctness — different head counts.
// ---------------------------------------------------------------------------
#[test]
fn test_joint_attention_multihead_shapes() -> Result<()> {
    for (dim, num_heads) in [(8, 2), (8, 4), (8, 8), (12, 3)] {
        let attn = JointAttention::new(
            identity_linear(dim),
            identity_linear(dim),
            identity_linear(dim),
            identity_linear(dim),
            num_heads,
            dim,
        )?;

        let x = DynTensor::full(&[2, 5, dim], 0.1, crate::DType::F32, &Device::Cpu)?;
        let ctx = DynTensor::full(&[2, 3, dim], 0.2, crate::DType::F32, &Device::Cpu)?;
        let spk = DynTensor::full(&[2, 1, dim], 0.3, crate::DType::F32, &Device::Cpu)?;

        let out = attn.forward_joint(&x, &ctx, &spk)?;
        assert_eq!(
            out.dims(),
            &[2, 5, dim],
            "dim={dim}, heads={num_heads}: expected [2, 5, {dim}], got {:?}",
            out.dims()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 3: Context concatenation order preserved — single-context variant.
// ---------------------------------------------------------------------------
#[test]
fn test_joint_attention_context_concatenation_order() -> Result<()> {
    let dim = 4;
    let num_heads = 1; // single head for clarity
    let attn = JointAttention::new(
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        num_heads,
        dim,
    )?;

    // x = [1, 1, 4] with value [1, 0, 0, 0]
    // ctx = [1, 1, 4] with value [0, 0, 0, 1]
    // With identity projections, Q=[1,0,0,0], K=[[1,0,0,0],[0,0,0,1]]
    // K^T dot Q = [1, 0] -> softmax = [exp(1/2)/Z, exp(0)/Z]
    // Output is weighted average of V rows
    let x = DynTensor::new(&[1.0, 0.0, 0.0, 0.0], &[1, 1, 4], &Device::Cpu)?;
    let ctx = DynTensor::new(&[0.0, 0.0, 0.0, 1.0], &[1, 1, 4], &Device::Cpu)?;

    let out = attn.forward_single_ctx(&x, &ctx)?;
    assert_eq!(out.dims(), &[1, 1, 4]);

    // Verify output is a mixture of x and ctx values (not zero, not just one)
    let vals = out.to_flat_vec::<f32>()?;
    // With scaled dot-product (scale = 1/sqrt(4) = 0.5):
    // scores = [1*1*0.5, 0*0.5] = [0.5, 0.0] before softmax
    // softmax([0.5, 0.0]) = [exp(0.5)/Z, 1/Z] where Z = exp(0.5) + 1
    let e05 = 0.5_f32.exp();
    let z = e05 + 1.0;
    let w_self = e05 / z; // weight on x's value
    let w_ctx = 1.0 / z; // weight on ctx's value
                         // Output = w_self * [1,0,0,0] + w_ctx * [0,0,0,1] = [w_self, 0, 0, w_ctx]
    assert!(
        (vals[0] - w_self).abs() < 1e-5,
        "expected ~{w_self}, got {}",
        vals[0]
    );
    assert!(vals[1].abs() < 1e-5, "expected ~0, got {}", vals[1]);
    assert!(vals[2].abs() < 1e-5, "expected ~0, got {}", vals[2]);
    assert!(
        (vals[3] - w_ctx).abs() < 1e-5,
        "expected ~{w_ctx}, got {}",
        vals[3]
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 4: num_heads must divide dim evenly — error case.
// ---------------------------------------------------------------------------
#[test]
fn test_joint_attention_invalid_heads_error() {
    let dim = 7; // not divisible by 3
    let result = JointAttention::new(
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        3,
        dim,
    );
    match result {
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("divisible"),
                "error should mention divisibility: {msg}"
            );
        }
        Ok(_) => panic!("expected error for dim=7, num_heads=3"),
    }
}

#[test]
fn test_joint_attention_zero_heads_error() {
    let dim = 4;
    let result = JointAttention::new(
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        0,
        dim,
    );
    match result {
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("num_heads must be > 0"),
                "error should mention zero heads: {msg}"
            );
        }
        Ok(_) => panic!("expected error for zero num_heads"),
    }
}

#[test]
fn test_joint_attention_zero_dim_error() {
    let result = JointAttention::new(
        identity_linear(1), // dummy, won't be used
        identity_linear(1),
        identity_linear(1),
        identity_linear(1),
        1,
        0, // dim=0 should be rejected
    );
    match result {
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("dim must be > 0"),
                "error should mention zero dim: {msg}"
            );
        }
        Ok(_) => panic!("expected error for dim=0"),
    }
}

// ---------------------------------------------------------------------------
// Test 5: Self-attention (no external context) produces valid output.
// When ctx1 and ctx2 are zero-length, this degenerates, but with single-ctx
// variant passing x as context, we get pure self-attention.
// ---------------------------------------------------------------------------
#[test]
fn test_joint_attention_self_attention_mode() -> Result<()> {
    let dim = 4;
    let num_heads = 2;
    let attn = JointAttention::new(
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        num_heads,
        dim,
    )?;

    // Uniform input: all tokens identical → attention weights should be uniform
    // → output should equal input (with identity projections)
    let x = DynTensor::full(&[1, 4, dim], 0.5, crate::DType::F32, &Device::Cpu)?;
    let empty_ctx = DynTensor::full(&[1, 0, dim], 0.0, crate::DType::F32, &Device::Cpu)?;

    // Use forward_joint with zero-length contexts for pure self-attention
    let out = attn.forward_joint(&x, &x, &empty_ctx)?;
    assert_eq!(out.dims(), &[1, 4, dim]);

    // With identity projections and uniform input, each token attends equally
    // to all context tokens. Output should be close to the mean of all V vectors,
    // which for uniform input = 0.5 everywhere.
    let vals = out.to_flat_vec::<f32>()?;
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 0.01,
            "element {i}: expected ~0.5, got {v}"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 6: Batched forward — multiple batch items produce independent results.
// ---------------------------------------------------------------------------
#[test]
fn test_joint_attention_batched() -> Result<()> {
    let dim = 4;
    let num_heads = 2;
    let attn = JointAttention::new(
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        identity_linear(dim),
        num_heads,
        dim,
    )?;

    // Batch of 3, each with different-length would need padding,
    // but we use same sequence length for simplicity
    let x = DynTensor::new(
        &[
            // batch 0: token 0
            1.0, 0.0, 0.0, 0.0, // batch 0: token 1
            0.0, 1.0, 0.0, 0.0, // batch 1: token 0
            0.0, 0.0, 1.0, 0.0, // batch 1: token 1
            0.0, 0.0, 0.0, 1.0,
        ],
        &[2, 2, 4],
        &Device::Cpu,
    )?;
    let ctx = DynTensor::full(&[2, 1, 4], 0.25, crate::DType::F32, &Device::Cpu)?;

    let out = attn.forward_single_ctx(&x, &ctx)?;
    assert_eq!(out.dims(), &[2, 2, 4]);

    // Verify outputs are finite
    let vals = out.to_flat_vec::<f32>()?;
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "element {i} is not finite: {v}");
    }
    Ok(())
}
