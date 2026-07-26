// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `NativeOpKind::FusedUpsampleConv1d`.
//!
//! Verifies construction, parameter validation, Debug/Display formatting,
//! dispatch count estimation, serde round-trip, and peephole pattern
//! detection. Part of #4310.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::WeightRef;

use crate::trace_compile::{CompiledPlan, CompiledStep, NativeOpKind, PeepholeConfig};

// ===========================================================================
// 1. NativeOpKind::FusedUpsampleConv1d construction
// ===========================================================================

#[test]
fn test_fused_upsample_conv1d_basic_construction() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 4,
        out_channels: 8,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 4, 16],
    };
    assert_eq!(op.variant_name(), "FusedUpsampleConv1d");
}

#[test]
fn test_fused_upsample_conv1d_kokoro_f0_shapes() {
    // Kokoro f0_energy segment uses 6 upsample+conv1d pairs.
    // Typical shape: [1, 128, T] with factor=2, kernel=3.
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 128,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 128, 64],
    };
    assert_eq!(op.variant_name(), "FusedUpsampleConv1d");
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn test_fused_upsample_conv1d_large_upsample_factor() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 8,
        in_channels: 64,
        out_channels: 128,
        kernel_size: 7,
        stride: 1,
        padding: 3,
        input_shape: vec![1, 64, 32],
    };
    assert_eq!(op.variant_name(), "FusedUpsampleConv1d");
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

#[test]
fn test_fused_upsample_conv1d_different_in_out_channels() {
    // in_channels != out_channels (channel expansion during upsampling).
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 4,
        in_channels: 32,
        out_channels: 256,
        kernel_size: 5,
        stride: 1,
        padding: 2,
        input_shape: vec![1, 32, 128],
    };
    assert_eq!(op.variant_name(), "FusedUpsampleConv1d");
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

// ===========================================================================
// 2. Scale factor validation
// ===========================================================================

#[test]
fn test_fused_upsample_conv1d_factor_1_is_identity_upsample() {
    // factor=1 means no upsampling, effectively just a conv1d.
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 1,
        in_channels: 16,
        out_channels: 16,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 16, 64],
    };
    assert_eq!(op.variant_name(), "FusedUpsampleConv1d");
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

#[test]
fn test_fused_upsample_conv1d_various_factors() {
    for factor in [2, 3, 4, 8, 16] {
        let op = NativeOpKind::FusedUpsampleConv1d {
            upsample_factor: factor,
            in_channels: 64,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            input_shape: vec![1, 64, 32],
        };
        assert_eq!(op.variant_name(), "FusedUpsampleConv1d", "factor={factor}");
        assert_eq!(
            op.estimated_metal_dispatches(),
            1,
            "factor={factor} should still be single dispatch"
        );
    }
}

// ===========================================================================
// 3. Conv1d parameter validation within the fused op
// ===========================================================================

#[test]
fn test_fused_upsample_conv1d_stride_2() {
    // Stride > 1 with upsample: upsample by factor then downsample by stride.
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 4,
        in_channels: 64,
        out_channels: 128,
        kernel_size: 3,
        stride: 2,
        padding: 1,
        input_shape: vec![1, 64, 32],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

#[test]
fn test_fused_upsample_conv1d_no_padding() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 32,
        out_channels: 64,
        kernel_size: 3,
        stride: 1,
        padding: 0,
        input_shape: vec![1, 32, 16],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

#[test]
fn test_fused_upsample_conv1d_kernel_size_1() {
    // k=1 conv after upsample -- pointwise channel mixing post-upsample.
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 128,
        out_channels: 256,
        kernel_size: 1,
        stride: 1,
        padding: 0,
        input_shape: vec![1, 128, 64],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

#[test]
fn test_fused_upsample_conv1d_large_kernel() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 64,
        out_channels: 64,
        kernel_size: 15,
        stride: 1,
        padding: 7,
        input_shape: vec![1, 64, 128],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

#[test]
fn test_fused_upsample_conv1d_batch_size_4() {
    // Multi-batch input.
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 32,
        out_channels: 64,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![4, 32, 128],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

// ===========================================================================
// 4. Display/Debug formatting
// ===========================================================================

#[test]
fn test_fused_upsample_conv1d_debug_contains_variant_name() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 4,
        out_channels: 8,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 4, 16],
    };
    let dbg = format!("{op:?}");
    assert!(
        dbg.contains("FusedUpsampleConv1d"),
        "Debug must contain variant name: {dbg}"
    );
}

#[test]
fn test_fused_upsample_conv1d_debug_contains_parameters() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 4,
        in_channels: 128,
        out_channels: 256,
        kernel_size: 7,
        stride: 2,
        padding: 3,
        input_shape: vec![1, 128, 64],
    };
    let dbg = format!("{op:?}");
    assert!(
        dbg.contains("upsample_factor: 4"),
        "missing upsample_factor: {dbg}"
    );
    assert!(
        dbg.contains("in_channels: 128"),
        "missing in_channels: {dbg}"
    );
    assert!(
        dbg.contains("out_channels: 256"),
        "missing out_channels: {dbg}"
    );
    assert!(dbg.contains("kernel_size: 7"), "missing kernel_size: {dbg}");
    assert!(dbg.contains("stride: 2"), "missing stride: {dbg}");
    assert!(dbg.contains("padding: 3"), "missing padding: {dbg}");
}

#[test]
fn test_fused_upsample_conv1d_debug_contains_input_shape() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 64,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![2, 64, 256],
    };
    let dbg = format!("{op:?}");
    assert!(
        dbg.contains("[2, 64, 256]"),
        "Debug should contain input_shape: {dbg}"
    );
}

// ===========================================================================
// 5. Clone and serde round-trip
// ===========================================================================

#[test]
fn test_fused_upsample_conv1d_clone_identical() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 128,
        out_channels: 256,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 128, 64],
    };
    let cloned = op.clone();
    assert_eq!(format!("{op:?}"), format!("{cloned:?}"));
    assert_eq!(op.variant_name(), cloned.variant_name());
    assert_eq!(
        op.estimated_metal_dispatches(),
        cloned.estimated_metal_dispatches()
    );
}

#[test]
fn test_fused_upsample_conv1d_serde_round_trip() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 4,
        in_channels: 32,
        out_channels: 64,
        kernel_size: 5,
        stride: 1,
        padding: 2,
        input_shape: vec![1, 32, 128],
    };
    let json = serde_json::to_string(&op).expect("serialize");
    let deserialized: NativeOpKind = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.variant_name(), "FusedUpsampleConv1d");
    assert_eq!(deserialized.estimated_metal_dispatches(), 1);
    // Verify field values survived the round-trip.
    match &deserialized {
        NativeOpKind::FusedUpsampleConv1d {
            upsample_factor,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            input_shape,
        } => {
            assert_eq!(*upsample_factor, 4);
            assert_eq!(*in_channels, 32);
            assert_eq!(*out_channels, 64);
            assert_eq!(*kernel_size, 5);
            assert_eq!(*stride, 1);
            assert_eq!(*padding, 2);
            assert_eq!(input_shape, &[1, 32, 128]);
        }
        other => panic!("expected FusedUpsampleConv1d, got {other:?}"),
    }
}

// ===========================================================================
// 6. Dispatch count and encoding events
// ===========================================================================

#[test]
fn test_fused_upsample_conv1d_single_dispatch() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 128,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 128, 64],
    };
    assert_eq!(
        op.estimated_metal_dispatches(),
        1,
        "FusedUpsampleConv1d must be a single Metal dispatch"
    );
    assert_eq!(
        op.estimated_encoding_events(),
        1,
        "FusedUpsampleConv1d must be a single encoding event"
    );
}

// ===========================================================================
// 7. external_node_ids returns None (no external edges)
// ===========================================================================

#[test]
fn test_fused_upsample_conv1d_no_external_node_ids() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 64,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 64, 32],
    };
    assert!(
        op.external_node_ids().is_none(),
        "FusedUpsampleConv1d should have no external_node_ids"
    );
}

// ===========================================================================
// 8. collect_direct_step_deps returns empty (no direct deps)
// ===========================================================================

#[test]
fn test_fused_upsample_conv1d_no_direct_step_deps() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 64,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 64, 32],
    };
    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);
    assert!(
        deps.is_empty(),
        "FusedUpsampleConv1d should have no direct step deps"
    );
}

// ===========================================================================
// 9. Plan compilation with the fused op (NativeOp step in CompiledPlan)
// ===========================================================================

#[test]
fn test_fused_upsample_conv1d_in_compiled_plan() {
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(vec![0.0; 8 * 4 * 3], vec![8, 4, 3]).unwrap(),
    );
    weight_data.insert(
        "bias".to_string(),
        WeightRef::new(vec![0.0; 8], vec![8]).unwrap(),
    );

    let step = CompiledStep::NativeOp {
        op: NativeOpKind::FusedUpsampleConv1d {
            upsample_factor: 2,
            in_channels: 4,
            out_channels: 8,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            input_shape: vec![1, 4, 16],
        },
        weight_data,
    };

    let plan = CompiledPlan {
        steps: vec![CompiledStep::InputForward, step],
        input_shapes: vec![vec![1, 4, 16]],
        output_step: 1,
        weight_names: vec!["weight".to_string(), "bias".to_string()],
    };

    // Verify the plan has 1 NativeOp dispatch.
    let dispatch_count = crate::trace_compile::count_dispatches(&plan);
    assert_eq!(dispatch_count, 1);
}

#[test]
fn test_fused_upsample_conv1d_plan_weight_names() {
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(vec![0.0; 16 * 8 * 5], vec![16, 8, 5]).unwrap(),
    );
    weight_data.insert(
        "bias".to_string(),
        WeightRef::new(vec![0.0; 16], vec![16]).unwrap(),
    );

    let step = CompiledStep::NativeOp {
        op: NativeOpKind::FusedUpsampleConv1d {
            upsample_factor: 4,
            in_channels: 8,
            out_channels: 16,
            kernel_size: 5,
            stride: 1,
            padding: 2,
            input_shape: vec![1, 8, 32],
        },
        weight_data,
    };

    match &step {
        CompiledStep::NativeOp { weight_data, .. } => {
            assert!(weight_data.contains_key("weight"));
            assert!(weight_data.contains_key("bias"));
            assert_eq!(weight_data["weight"].shape(), &[16, 8, 5]);
            assert_eq!(weight_data["bias"].shape(), &[16]);
        }
        _ => panic!("expected NativeOp"),
    }
}

// ===========================================================================
// 10. PeepholeConfig fuse_upsample_conv1d field
// ===========================================================================

#[test]
fn test_peephole_config_upsample_conv1d_default_enabled() {
    let config = PeepholeConfig::default();
    assert!(
        config.fuse_upsample_conv1d,
        "fuse_upsample_conv1d must be enabled by default"
    );
}

#[test]
fn test_peephole_config_upsample_conv1d_disable() {
    let config = PeepholeConfig {
        fuse_upsample_conv1d: false,
        ..Default::default()
    };
    assert!(!config.fuse_upsample_conv1d);
    // Other passes should remain enabled.
    assert!(config.norm_activ_conv1d);
    assert!(config.fused_resblock);
    assert!(config.silu_mul);
}

// ===========================================================================
// 11. Multiple FusedUpsampleConv1d steps in a plan (f0_energy pattern)
// ===========================================================================

#[test]
fn test_multiple_fused_upsample_conv1d_in_plan() {
    // f0_energy has 6 upsample+conv1d pairs, each becoming a FusedUpsampleConv1d.
    let mut steps = vec![CompiledStep::InputForward];

    for i in 0..6 {
        let c_in = 128 >> i.min(3);
        let c_out = 128 >> (i + 1).min(3);
        let mut weight_data = HashMap::new();
        weight_data.insert(
            "weight".to_string(),
            WeightRef::new(vec![0.0; c_out * c_in * 3], vec![c_out, c_in, 3]).unwrap(),
        );
        weight_data.insert(
            "bias".to_string(),
            WeightRef::new(vec![0.0; c_out], vec![c_out]).unwrap(),
        );

        steps.push(CompiledStep::NativeOp {
            op: NativeOpKind::FusedUpsampleConv1d {
                upsample_factor: 2,
                in_channels: c_in,
                out_channels: c_out,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                input_shape: vec![1, c_in, 64 * (1 << i)],
            },
            weight_data,
        });
    }

    let plan = CompiledPlan {
        steps,
        input_shapes: vec![vec![1, 128, 64]],
        output_step: 6,
        weight_names: vec!["weight".to_string(), "bias".to_string()],
    };

    // All 6 FusedUpsampleConv1d steps count as dispatches.
    assert_eq!(crate::trace_compile::count_dispatches(&plan), 6);
}
