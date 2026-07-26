// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for PTX depthwise conv2d kernel generation.
//!
//! These tests cover additional kernel sizes, identity-kernel semantics,
//! stride-2 output length halving, multi-channel independence, and PTX
//! syntactic validity.

use crate::ptx_depthwise_conv::{
    depthwise_conv2d_output_size, depthwise_conv2d_reference, generate_depthwise_conv2d_ptx,
    PtxDepthwiseConv2dConfig,
};

// ---------------------------------------------------------------------------
// Different kernel sizes: k=1, 3, 5, 7
// ---------------------------------------------------------------------------

#[test]
fn test_depthwise_conv_ptx_different_kernel_sizes() {
    for k in [1, 3, 5, 7] {
        let config = PtxDepthwiseConv2dConfig::new(&format!("dw_k{k}"), 16, k, k);
        let ptx = generate_depthwise_conv2d_ptx(&config).unwrap_or_else(|_| panic!("PTX generation should succeed for kernel size {k}x{k}"));
        // Must contain the entry point
        assert!(
            ptx.contains(&format!(".visible .entry dw_k{k}")),
            "kernel {k}x{k}: missing entry point"
        );
        // Must contain the kernel size in the header comment
        assert!(
            ptx.contains(&format!("kernel={k}x{k}")),
            "kernel {k}x{k}: missing kernel size in header"
        );
        // Must contain standard PTX structure
        assert!(ptx.contains(".version"), "kernel {k}x{k}: missing .version");
        assert!(ptx.contains("ret;"), "kernel {k}x{k}: missing ret");
    }
}

// ---------------------------------------------------------------------------
// Identity kernel: [0, 1, 0] with zero padding acts as identity
// ---------------------------------------------------------------------------

#[test]
fn test_depthwise_conv_reference_identity_kernel() {
    // 1D depthwise conv with kernel [0, 1, 0] and padding=1 should be identity.
    // We model this as a 1-channel, 1xN 2D conv with kernel 1x3, padding 0x1.
    let h_in = 1;
    let w_in = 8;
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    // Weight: [C=1, kH=1, kW=3] = [0.0, 1.0, 0.0]
    let weight = vec![0.0, 1.0, 0.0];
    let config = PtxDepthwiseConv2dConfig::new("dw_id", 1, 1, 3).with_padding(0, 1);
    let out = depthwise_conv2d_reference(&input, &weight, None, &config, 1, h_in, w_in);

    // Output size: (8 + 2*1 - 3) / 1 + 1 = 8
    assert_eq!(
        depthwise_conv2d_output_size(w_in, 3, 1, 1),
        Some(w_in),
        "same-padding identity should preserve width"
    );
    assert_eq!(out.len(), 8);

    // Each output element should equal the corresponding input element
    for i in 0..8 {
        assert!(
            (out[i] - input[i]).abs() < 1e-6,
            "identity kernel mismatch at index {i}: expected {}, got {}",
            input[i],
            out[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Stride 2 halves output length
// ---------------------------------------------------------------------------

#[test]
fn test_depthwise_conv_reference_stride2() {
    // 1 batch, 1 channel, 1x8 input, 1x1 kernel (pointwise), stride_w=2
    // Output width = (8 + 0 - 1) / 2 + 1 = 4
    let h_in = 1;
    let w_in = 8;
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let weight = vec![1.0]; // 1x1 kernel, weight = 1.0

    let config = PtxDepthwiseConv2dConfig::new("dw_s2", 1, 1, 1).with_stride(1, 2);
    let out = depthwise_conv2d_reference(&input, &weight, None, &config, 1, h_in, w_in);

    let expected_len = depthwise_conv2d_output_size(w_in, 1, 2, 0).unwrap();
    assert_eq!(expected_len, 4, "stride 2 should halve the output length");
    assert_eq!(out.len(), expected_len);

    // With a 1x1 identity kernel and stride 2, output picks every other element
    assert!((out[0] - 1.0).abs() < 1e-6); // input[0]
    assert!((out[1] - 3.0).abs() < 1e-6); // input[2]
    assert!((out[2] - 5.0).abs() < 1e-6); // input[4]
    assert!((out[3] - 7.0).abs() < 1e-6); // input[6]
}

// ---------------------------------------------------------------------------
// Multi-channel: each channel processed independently
// ---------------------------------------------------------------------------

#[test]
fn test_depthwise_conv_reference_multi_channel() {
    // 1 batch, 3 channels, 1x4 spatial, 1x1 kernel per channel.
    // Each channel has its own weight: ch0=2.0, ch1=0.5, ch2=-1.0
    let channels = 3;
    let h_in = 1;
    let w_in = 4;
    let input = vec![
        // ch 0
        1.0, 2.0, 3.0, 4.0, // ch 1
        10.0, 20.0, 30.0, 40.0, // ch 2
        100.0, 200.0, 300.0, 400.0,
    ];
    let weight = vec![2.0, 0.5, -1.0];
    let config = PtxDepthwiseConv2dConfig::new("dw_mc", channels, 1, 1);
    let out = depthwise_conv2d_reference(&input, &weight, None, &config, 1, h_in, w_in);

    assert_eq!(out.len(), channels * 4);

    // ch0: *2.0
    assert!((out[0] - 2.0).abs() < 1e-6);
    assert!((out[1] - 4.0).abs() < 1e-6);
    assert!((out[2] - 6.0).abs() < 1e-6);
    assert!((out[3] - 8.0).abs() < 1e-6);

    // ch1: *0.5
    assert!((out[4] - 5.0).abs() < 1e-6);
    assert!((out[5] - 10.0).abs() < 1e-6);
    assert!((out[6] - 15.0).abs() < 1e-6);
    assert!((out[7] - 20.0).abs() < 1e-6);

    // ch2: *-1.0
    assert!((out[8] - (-100.0)).abs() < 1e-6);
    assert!((out[9] - (-200.0)).abs() < 1e-6);
    assert!((out[10] - (-300.0)).abs() < 1e-6);
    assert!((out[11] - (-400.0)).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// PTX string validity
// ---------------------------------------------------------------------------

#[test]
fn test_depthwise_conv_ptx_validity() {
    let config = PtxDepthwiseConv2dConfig::new("dw_valid", 32, 3, 3)
        .with_stride(1, 1)
        .with_padding(1, 1)
        .with_bias(true);
    let ptx = generate_depthwise_conv2d_ptx(&config).expect("PTX generation must succeed");

    // Must be non-empty
    assert!(!ptx.is_empty(), "PTX output must not be empty");

    // Must contain the standard PTX module header
    assert!(ptx.contains(".version"), "missing PTX version directive");
    assert!(ptx.contains(".target"), "missing PTX target directive");
    assert!(
        ptx.contains(".address_size 64"),
        "missing 64-bit address size"
    );

    // Must contain the entry point declaration
    assert!(
        ptx.contains(".visible .entry dw_valid"),
        "missing entry point"
    );

    // Must declare register spaces
    assert!(
        ptx.contains(".reg .u32"),
        "missing u32 register declaration"
    );
    assert!(
        ptx.contains(".reg .f32"),
        "missing f32 register declaration"
    );
    assert!(
        ptx.contains(".reg .u64"),
        "missing u64 register declaration"
    );
    assert!(
        ptx.contains(".reg .pred"),
        "missing predicate register declaration"
    );

    // Must contain load/store instructions
    assert!(
        ptx.contains("ld.param.u64"),
        "missing parameter load instruction"
    );
    assert!(
        ptx.contains("ld.global.f32"),
        "missing global load instruction"
    );
    assert!(
        ptx.contains("st.global.f32"),
        "missing global store instruction"
    );

    // Must not contain CUDA C++ artifacts
    assert!(
        !ptx.contains("__global__"),
        "must not contain CUDA C++ __global__"
    );
    assert!(!ptx.contains("#include"), "must not contain C/C++ #include");
    assert!(!ptx.contains("extern \"C\""), "must not contain C++ extern");

    // Must terminate properly
    assert!(ptx.contains("ret;"), "missing return instruction");

    // Balanced braces
    let open_braces = ptx.matches('{').count();
    let close_braces = ptx.matches('}').count();
    assert_eq!(
        open_braces, close_braces,
        "unbalanced braces: {open_braces} open vs {close_braces} close"
    );
}
