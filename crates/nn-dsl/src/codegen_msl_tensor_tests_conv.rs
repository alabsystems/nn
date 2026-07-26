// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d and ConvTranspose1d dispatch plan tests.
//!
//! Extracted from codegen_msl_tensor_tests.rs per 500-line file limit.

use crate::codegen_msl_tensor::{build_dispatch_plan, DispatchStep};
use crate::conv1d::build_conv1d;
use crate::conv_transpose_1d::build_conv_transpose_1d;
use crate::ir::ScalarType;

#[test]
fn test_dispatch_plan_conv1d_basic() {
    // Conv1d: in_ch=4, out_ch=2, kernel=3, in_len=8, stride=1, pad=0, no bias
    let def = build_conv1d("conv1d_dp", 4, 2, 3, 8, 1, 0, false).expect("build");
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("Conv1d dispatch");
    assert_eq!(
        plan.len(),
        1,
        "Conv1d should produce exactly 1 dispatch step"
    );
    match &plan[0] {
        DispatchStep::Conv1d(ref p) => {
            assert_eq!(p.stride, 1);
            assert_eq!(p.padding, 0);
            // out_len = (8 - 3) / 1 + 1 = 6; total = 2 * 6 = 12
            assert_eq!(p.total_elements, 12);
        }
        other => panic!("expected Conv1d step, got: {other:?}"),
    }
}

#[test]
fn test_dispatch_plan_conv1d_with_bias() {
    let def = build_conv1d("conv1d_b", 4, 2, 3, 8, 1, 0, true).expect("build");
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("Conv1d dispatch");
    assert_eq!(plan.len(), 1);
    match &plan[0] {
        DispatchStep::Conv1d(ref p) => {
            assert!(p.bias.is_some(), "bias node should be present");
        }
        other => panic!("expected Conv1d step, got: {other:?}"),
    }
}

#[test]
fn test_dispatch_plan_conv1d_stride_padding_dvoice() {
    // dvoice pattern: in_ch=1, out_ch=48, kernel=8, in_len=24000, stride=4, pad=2
    let def = build_conv1d("conv1d_dv", 1, 48, 8, 24000, 4, 2, false).expect("build");
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("Conv1d dispatch");
    assert_eq!(plan.len(), 1);
    match &plan[0] {
        DispatchStep::Conv1d(ref p) => {
            assert_eq!(p.stride, 4);
            assert_eq!(p.padding, 2);
            assert_eq!(p.in_channels, 1);
            assert_eq!(p.out_channels, 48);
            assert_eq!(p.kernel_size, 8);
            // out_len = (24000 + 2*2 - 8) / 4 + 1 = 5999
            let out_len = (24000 + 2 * 2 - 8) / 4 + 1;
            assert_eq!(p.total_elements, 48 * out_len);
        }
        other => panic!("expected Conv1d step, got: {other:?}"),
    }
}

#[test]
fn test_dispatch_plan_conv1d_kernel_name_unique() {
    let def = build_conv1d("test_k", 2, 4, 3, 8, 1, 0, false).expect("build");
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("Conv1d dispatch");
    match &plan[0] {
        DispatchStep::Conv1d(ref p) => {
            assert!(
                p.kernel_name.contains("test_k"),
                "kernel name should contain def name"
            );
            assert!(
                p.kernel_name.contains("conv1d"),
                "kernel name should contain 'conv1d'"
            );
        }
        other => panic!("expected Conv1d step, got: {other:?}"),
    }
}

#[test]
fn test_dispatch_plan_conv_transpose_1d_basic() {
    let def = build_conv_transpose_1d("ct1d_dp", 4, 2, 3, 8, 2, 1, 1, 1, false, 0).expect("build");
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("ConvTranspose1d dispatch");
    assert_eq!(plan.len(), 1);
    match &plan[0] {
        DispatchStep::ConvTranspose1d(ref p) => {
            assert_eq!(p.in_channels, 4);
            assert_eq!(p.out_channels, 2);
            assert_eq!(p.kernel_size, 3);
            assert_eq!(p.in_length, 8);
            assert_eq!(p.stride, 2);
            assert_eq!(p.padding, 1);
        }
        other => panic!("expected ConvTranspose1d step, got: {other:?}"),
    }
}

#[test]
fn test_dispatch_plan_conv_transpose_1d_with_bias() {
    let def = build_conv_transpose_1d("ct1d_b", 4, 2, 3, 8, 2, 1, 1, 1, true, 0).expect("build");
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("dispatch");
    match &plan[0] {
        DispatchStep::ConvTranspose1d(ref p) => {
            assert!(p.bias.is_some(), "bias should be present");
        }
        other => panic!("expected ConvTranspose1d step, got: {other:?}"),
    }
}
