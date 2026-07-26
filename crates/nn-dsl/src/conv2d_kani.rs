// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Conv2d builder (`build_conv2d_full`).
//!
//! Extracted from `conv2d.rs` for 500-line compliance.
//! Re: #1569.

use super::build_conv2d_full;

/// Proves `build_conv2d_full` never panics for any bounded parameter inputs.
#[kani::unwind(1)]
#[kani::proof]
fn conv2d_build_no_panic() {
    let in_h: usize = kani::any();
    let in_w: usize = kani::any();
    let kernel_h: usize = kani::any();
    let kernel_w: usize = kani::any();
    let stride_h: usize = kani::any();
    let stride_w: usize = kani::any();
    let padding_h: usize = kani::any();
    let padding_w: usize = kani::any();
    let dilation_h: usize = kani::any();
    let dilation_w: usize = kani::any();
    let in_channels: usize = kani::any();
    let out_channels: usize = kani::any();
    let groups: usize = kani::any();

    kani::assume(in_h >= 1 && in_h <= 256);
    kani::assume(in_w >= 1 && in_w <= 256);
    kani::assume(kernel_h <= 64);
    kani::assume(kernel_w <= 64);
    kani::assume(stride_h <= 64);
    kani::assume(stride_w <= 64);
    kani::assume(padding_h <= 64);
    kani::assume(padding_w <= 64);
    kani::assume(dilation_h <= 32);
    kani::assume(dilation_w <= 32);
    kani::assume(in_channels >= 1 && in_channels <= 64);
    kani::assume(out_channels >= 1 && out_channels <= 64);
    kani::assume(groups <= 64);

    let _ = build_conv2d_full(
        "kani_test",
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        in_h,
        in_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        false,
    );
}

/// Proves that when `build_conv2d_full` succeeds, output spatial dims >= 1.
#[kani::unwind(1)]
#[kani::proof]
fn conv2d_output_shape_positive() {
    let in_h: usize = kani::any();
    let in_w: usize = kani::any();
    let kernel_h: usize = kani::any();
    let kernel_w: usize = kani::any();
    let stride_h: usize = kani::any();
    let stride_w: usize = kani::any();
    let padding_h: usize = kani::any();
    let padding_w: usize = kani::any();
    let dilation_h: usize = kani::any();
    let dilation_w: usize = kani::any();

    kani::assume(in_h >= 1 && in_h <= 256);
    kani::assume(in_w >= 1 && in_w <= 256);
    kani::assume(kernel_h >= 1 && kernel_h <= 64);
    kani::assume(kernel_w >= 1 && kernel_w <= 64);
    kani::assume(stride_h >= 1 && stride_h <= 64);
    kani::assume(stride_w >= 1 && stride_w <= 64);
    kani::assume(padding_h <= 64);
    kani::assume(padding_w <= 64);
    kani::assume(dilation_h >= 1 && dilation_h <= 32);
    kani::assume(dilation_w >= 1 && dilation_w <= 32);

    if let Ok(def) = build_conv2d_full(
        "kani_test",
        4,
        2,
        kernel_h,
        kernel_w,
        in_h,
        in_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        1,
        false,
    ) {
        let output_node = &def.nodes[def.nodes.len() - 1];
        let out_h = output_node.shape[1];
        let out_w = output_node.shape[2];
        assert!(out_h >= 1, "Conv2d output height must be >= 1");
        assert!(out_w >= 1, "Conv2d output width must be >= 1");
    }
}

/// Proves that `out_channels` in the output shape matches the requested parameter.
#[kani::unwind(1)]
#[kani::proof]
fn conv2d_output_channels_preserved() {
    let in_channels: usize = kani::any();
    let out_channels: usize = kani::any();
    let kernel_h: usize = kani::any();
    let kernel_w: usize = kani::any();
    let in_h: usize = kani::any();
    let in_w: usize = kani::any();
    let stride_h: usize = kani::any();
    let stride_w: usize = kani::any();
    let padding_h: usize = kani::any();
    let padding_w: usize = kani::any();

    kani::assume(in_channels >= 1 && in_channels <= 64);
    kani::assume(out_channels >= 1 && out_channels <= 64);
    kani::assume(kernel_h >= 1 && kernel_h <= 32);
    kani::assume(kernel_w >= 1 && kernel_w <= 32);
    kani::assume(in_h >= 1 && in_h <= 256);
    kani::assume(in_w >= 1 && in_w <= 256);
    kani::assume(stride_h >= 1 && stride_h <= 16);
    kani::assume(stride_w >= 1 && stride_w <= 16);
    kani::assume(padding_h <= 32);
    kani::assume(padding_w <= 32);

    if let Ok(def) = build_conv2d_full(
        "kani_test",
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        in_h,
        in_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        1,
        1,
        1,
        false,
    ) {
        let output_node = &def.nodes[def.nodes.len() - 1];
        assert_eq!(
            output_node.shape[0], out_channels,
            "output channel dim must equal out_channels parameter"
        );
    }
}
