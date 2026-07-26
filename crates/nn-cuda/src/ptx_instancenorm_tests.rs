// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX InstanceNorm kernel generation.
//!
//! Covers PTX structural checks, reference computation verification,
//! edge cases, and typical dimensions.

use super::*;

// =========================================================================
// PTX structural validation
// =========================================================================

#[test]
fn test_instancenorm_ptx_valid_spirv() {
    // Verify PTX string contains key instructions for instance normalization
    let ptx = generate_instancenorm_ptx(64, 32, 32, 1e-5);

    // PTX version and target
    assert!(ptx.contains(".version 7.0"), "must target PTX 7.0");
    assert!(ptx.contains(".target sm_70"), "must target sm_70");
    assert!(
        ptx.contains(".address_size 64"),
        "must use 64-bit addressing"
    );

    // Kernel entry point
    assert!(
        ptx.contains(".visible .entry instancenorm_f32"),
        "must have entry point"
    );

    // Key instructions for normalization
    assert!(
        ptx.contains("rsqrt.approx.f32"),
        "must use rsqrt for 1/sqrt(var+eps)"
    );
    assert!(ptx.contains("sub.f32"), "must subtract mean (sub.f32)");
    assert!(
        ptx.contains("fma.rn.f32"),
        "must use fma for gamma*norm+beta"
    );

    // Warp-level reduction for computing mean/variance
    assert!(
        ptx.contains("shfl.down.sync"),
        "must use warp shuffle for reduction"
    );

    // Memory operations
    assert!(
        ptx.contains("ld.global.f32"),
        "must load from global memory"
    );
    assert!(ptx.contains("st.global.f32"), "must store to global memory");

    // Parameters
    assert!(ptx.contains("param_input"), "must have input param");
    assert!(ptx.contains("param_output"), "must have output param");
    assert!(ptx.contains("param_gamma"), "must have gamma param");
    assert!(ptx.contains("param_beta"), "must have beta param");
    assert!(ptx.contains("param_spatial"), "must have spatial param");

    // Block size
    assert!(ptx.contains(".reqntid 256"), "must request 256 threads");
}

// =========================================================================
// Reference: identity transform (gamma=1, beta=0)
// =========================================================================

#[test]
fn test_instancenorm_reference_identity() {
    // gamma=1, beta=0 should normalize to approximately mean=0, var=1
    let n = 1;
    let c = 2;
    let h = 4;
    let w = 4;
    let spatial = h * w;

    // Create input with known distribution per channel
    let mut input = vec![0.0f32; n * c * spatial];
    for ch in 0..c {
        for i in 0..spatial {
            input[ch * spatial + i] = (i as f32) * 0.1 + ch as f32;
        }
    }

    let gamma = vec![1.0f32; c];
    let beta = vec![0.0f32; c];
    let eps = 1e-5;

    let output = instancenorm_reference(&input, &gamma, &beta, n, c, h, w, eps);

    // For each channel, output should have mean ~0 and variance ~1
    for ch in 0..c {
        let base = ch * spatial;
        let channel_out = &output[base..base + spatial];

        let mean: f32 = channel_out.iter().sum::<f32>() / spatial as f32;
        let var: f32 = channel_out
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f32>()
            / spatial as f32;

        assert!(
            mean.abs() < 1e-4,
            "channel {ch}: mean should be ~0, got {mean}"
        );
        assert!(
            (var - 1.0).abs() < 0.01,
            "channel {ch}: var should be ~1, got {var}"
        );
    }
}

// =========================================================================
// Reference: scale and shift (gamma=2, beta=1)
// =========================================================================

#[test]
fn test_instancenorm_reference_scale_shift() {
    let n = 1;
    let c = 1;
    let h = 2;
    let w = 2;
    let spatial = h * w;

    let input = vec![1.0, 2.0, 3.0, 4.0]; // mean=2.5, var=1.25
    let gamma = vec![2.0f32];
    let beta = vec![1.0f32];
    let eps = 1e-5;

    let output = instancenorm_reference(&input, &gamma, &beta, n, c, h, w, eps);

    // Check that gamma=2 and beta=1 are applied
    let mean: f32 = output.iter().sum::<f32>() / spatial as f32;
    // Output mean should be beta = 1.0 (since normalized mean is 0, gamma*0 + beta = beta)
    assert!(
        (mean - 1.0).abs() < 1e-4,
        "output mean should be ~beta=1.0, got {mean}"
    );

    // Output variance should be gamma^2 = 4.0 (since normalized var ~1)
    let var: f32 = output.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / spatial as f32;
    assert!(
        (var - 4.0).abs() < 0.1,
        "output var should be ~gamma^2=4.0, got {var}"
    );
}

// =========================================================================
// Reference: single spatial element (H=W=1)
// =========================================================================

#[test]
fn test_instancenorm_reference_single_spatial() {
    // When H=W=1, spatial=1, so mean=x, var=0, normalized=0 -> output=beta
    let n = 2;
    let c = 3;
    let h = 1;
    let w = 1;

    let input = vec![5.0, 10.0, -3.0, 7.0, 2.0, 8.0]; // N=2, C=3
    let gamma = vec![2.0, 3.0, 0.5];
    let beta = vec![1.0, -1.0, 0.0];
    let eps = 1e-5;

    let output = instancenorm_reference(&input, &gamma, &beta, n, c, h, w, eps);

    // With spatial=1: mean=x, var=0, (x-mean)/sqrt(0+eps) ~= 0
    // output ~= gamma*0 + beta = beta for each channel
    for batch in 0..n {
        for ch in 0..c {
            let idx = batch * c + ch;
            assert!(
                (output[idx] - beta[ch]).abs() < 1e-2,
                "batch={batch}, ch={ch}: expected ~{}, got {}",
                beta[ch],
                output[idx]
            );
        }
    }
}

// =========================================================================
// Reference: manual computation check
// =========================================================================

#[test]
fn test_instancenorm_reference_matches_formula() {
    let n = 1;
    let c = 1;
    let h = 1;
    let w = 4;

    let input = vec![2.0, 4.0, 6.0, 8.0];
    let gamma = vec![1.0];
    let beta = vec![0.0];
    let eps = 0.0;

    let output = instancenorm_reference(&input, &gamma, &beta, n, c, h, w, eps);

    // Manual: mean = (2+4+6+8)/4 = 5.0
    let mean = 5.0f32;
    // var = ((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2) / 4 = (9+1+1+9)/4 = 5.0
    let var = 5.0f32;
    let inv_std = 1.0 / var.sqrt(); // 1/sqrt(5)

    let expected: Vec<f32> = input.iter().map(|x| (x - mean) * inv_std).collect();

    for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "element {i}: got {got}, expected {exp}"
        );
    }
}

// =========================================================================
// Additional tests
// =========================================================================

#[test]
fn test_instancenorm_ptx_different_dims_produce_different_ptx() {
    let ptx_a = generate_instancenorm_ptx(32, 8, 8, 1e-5);
    let ptx_b = generate_instancenorm_ptx(64, 16, 16, 1e-5);
    assert_ne!(ptx_a, ptx_b, "different dims should produce different PTX");
}

#[test]
fn test_instancenorm_ptx_different_eps_produce_different_ptx() {
    let ptx_a = generate_instancenorm_ptx(32, 8, 8, 1e-5);
    let ptx_b = generate_instancenorm_ptx(32, 8, 8, 1e-3);
    assert_ne!(ptx_a, ptx_b, "different eps should produce different PTX");
}

#[test]
fn test_instancenorm_ptx_reasonable_size() {
    let ptx = generate_instancenorm_ptx(64, 32, 32, 1e-5);
    assert!(
        ptx.len() > 1000,
        "PTX should be substantial, got {} bytes",
        ptx.len()
    );
    assert!(ptx.len() < 100_000, "PTX too large: {} bytes", ptx.len());
}

#[test]
fn test_instancenorm_reference_multi_batch_multi_channel() {
    let n = 2;
    let c = 3;
    let h = 4;
    let w = 4;
    let spatial = h * w;
    let total = n * c * spatial;

    let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let gamma = vec![1.0; c];
    let beta = vec![0.0; c];
    let eps = 1e-5;

    let output = instancenorm_reference(&input, &gamma, &beta, n, c, h, w, eps);

    assert_eq!(output.len(), total);

    // All outputs should be finite
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} is not finite");
    }

    // Each (batch, channel) pair should have mean ~0
    for batch in 0..n {
        for ch in 0..c {
            let base = batch * c * spatial + ch * spatial;
            let slice = &output[base..base + spatial];
            let mean: f32 = slice.iter().sum::<f32>() / spatial as f32;
            assert!(
                mean.abs() < 1e-4,
                "batch={batch}, ch={ch}: mean should be ~0, got {mean}"
            );
        }
    }
}

#[test]
fn test_instancenorm_reference_zero_input() {
    let n = 1;
    let c = 2;
    let h = 3;
    let w = 3;
    let total = n * c * h * w;

    let input = vec![0.0f32; total];
    let gamma = vec![1.0; c];
    let beta = vec![0.0; c];
    let eps = 1e-5;

    let output = instancenorm_reference(&input, &gamma, &beta, n, c, h, w, eps);

    // All zeros input: mean=0, var=0, (x-0)/sqrt(0+eps) ~= 0 -> output ~= beta = 0
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.abs() < 1e-3,
            "output[{i}] should be ~0 for zero input, got {v}"
        );
    }
}

#[test]
fn test_instancenorm_block_size_constant() {
    assert_eq!(INSTANCENORM_BLOCK_SIZE, 256);
}

#[test]
fn test_instancenorm_ptx_has_shared_memory() {
    // With block_size=256 (8 warps), cross-warp reduction uses shared memory
    let ptx = generate_instancenorm_ptx(32, 8, 8, 1e-5);
    assert!(
        ptx.contains(".shared"),
        "InstanceNorm with 256 threads should use shared memory for cross-warp reduction"
    );
}

#[test]
fn test_instancenorm_ptx_ends_with_closing_brace() {
    let ptx = generate_instancenorm_ptx(32, 8, 8, 1e-5);
    let trimmed = ptx.trim_end();
    assert!(trimmed.ends_with('}'));
}
