// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::WeightRef;

use super::*;
use crate::{NormActivConv1dParams, NormActivation, StyleProjectionParams};

/// Helper: make a FusedResBlock with absorbed style projection.
fn make_style_resblock(
    x_step: usize,
    style_step: usize,
    c1: usize,
    c2: usize,
    style_dim: usize,
) -> CompiledStep {
    let params = NormActivConv1dParams {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, c1, 100],
        output_channels: c1,
        kernel_size: 3,
    };
    let params2 = NormActivConv1dParams {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, c2, 100],
        output_channels: c2,
        kernel_size: 3,
    };

    let mut weight_data = HashMap::new();
    // style1: [2*c1, style_dim] weight + [2*c1] bias
    let w1 = WeightRef::new(vec![1.0; 2 * c1 * style_dim], vec![2 * c1, style_dim]).unwrap();
    let b1 = WeightRef::new(vec![0.1; 2 * c1], vec![2 * c1]).unwrap();
    weight_data.insert("style1_weight".to_string(), w1);
    weight_data.insert("style1_bias".to_string(), b1);
    // style2: [2*c2, style_dim] weight + [2*c2] bias
    let w2 = WeightRef::new(vec![2.0; 2 * c2 * style_dim], vec![2 * c2, style_dim]).unwrap();
    let b2 = WeightRef::new(vec![0.2; 2 * c2], vec![2 * c2]).unwrap();
    weight_data.insert("style2_weight".to_string(), w2);
    weight_data.insert("style2_bias".to_string(), b2);
    // Conv weights (should be preserved after batching).
    let conv_w = WeightRef::new(vec![0.5; c1 * c1 * 3], vec![c1, c1, 3]).unwrap();
    weight_data.insert("p1_conv_weight".to_string(), conv_w);

    CompiledStep::NativeOp {
        op: NativeOpKind::FusedResBlock {
            phase1: params,
            phase2: params2,
            input_steps: vec![x_step, style_step],
            residual_scale: 1.0,
            style_proj: Some(StyleProjectionParams {
                channels1: c1,
                channels2: c2,
                style_dim,
            }),
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: None,
        },
        weight_data,
    }
}

#[test]
fn test_batch_two_blocks_same_style() {
    // Layout: [0]=Passthrough(x), [1]=Passthrough(style), [2]=Identity,
    //         [3]=FusedResBlock(style_proj), [4]=FusedResBlock(style_proj)
    let mut steps = vec![
        CompiledStep::IdentityPassthrough, // 0: x placeholder
        CompiledStep::IdentityPassthrough, // 1: style placeholder
        CompiledStep::IdentityPassthrough, // 2: absorbed Linear slot
        make_style_resblock(0, 1, 4, 4, 2), // 3: block 0 (c1=4, c2=4, style_dim=2)
        make_style_resblock(0, 1, 8, 8, 2), // 4: block 1 (c1=8, c2=8, style_dim=2)
    ];

    batch_style_projections(&mut steps);

    // Slot 2 should now be BatchedStyleProjection.
    match &steps[2] {
        CompiledStep::NativeOp {
            op: NativeOpKind::BatchedStyleProjection { blocks, style_dim, total_out, style_step },
            weight_data,
        } => {
            assert_eq!(*style_dim, 2);
            // block0: 2*(4+4) = 16, block1: 2*(8+8) = 32. total = 48.
            assert_eq!(*total_out, 48);
            assert_eq!(*style_step, 1);
            assert_eq!(blocks.len(), 2);
            assert_eq!(blocks[0].offset, 0);
            assert_eq!(blocks[0].channels1, 4);
            assert_eq!(blocks[1].offset, 16);
            assert_eq!(blocks[1].channels1, 8);
            assert!(weight_data.contains_key("weight_t"));
            assert!(weight_data.contains_key("bias"));
            assert_eq!(weight_data["weight_t"].shape(), &[2, 48]);
            assert_eq!(weight_data["bias"].shape(), &[48]);
        }
        other => panic!("expected BatchedStyleProjection at slot 2, got {other:?}"),
    }

    // FusedResBlocks should now have style_batch_offset and no style_proj.
    for &rb_idx in &[3usize, 4] {
        match &steps[rb_idx] {
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedResBlock {
                    style_proj,
                    style_batch_offset: Some(sbo),
                    input_steps,
                    ..
                },
                weight_data,
            } => {
                assert!(style_proj.is_none(), "style_proj should be cleared");
                assert_eq!(input_steps, &[0, 2], "should point to [x, batch_slot]");
                // Style weights should be removed.
                assert!(!weight_data.contains_key("style1_weight"));
                assert!(!weight_data.contains_key("style2_weight"));
                // Conv weights should be preserved.
                assert!(weight_data.contains_key("p1_conv_weight"));
                // Check offsets.
                if rb_idx == 3 {
                    assert_eq!(sbo.offset, 0);
                    assert_eq!(sbo.channels1, 4);
                } else {
                    assert_eq!(sbo.offset, 16);
                    assert_eq!(sbo.channels1, 8);
                }
            }
            other => panic!("expected batched FusedResBlock at step {rb_idx}, got {other:?}"),
        }
    }
}

#[test]
fn test_single_block_not_batched() {
    // Only 1 FusedResBlock with style_proj — batching should NOT apply.
    let mut steps = vec![
        CompiledStep::IdentityPassthrough,
        CompiledStep::IdentityPassthrough,
        CompiledStep::IdentityPassthrough,
        make_style_resblock(0, 1, 4, 4, 2),
    ];

    batch_style_projections(&mut steps);

    // Step 3 should still have style_proj (unchanged).
    match &steps[3] {
        CompiledStep::NativeOp {
            op: NativeOpKind::FusedResBlock { style_proj: Some(_), .. },
            ..
        } => {} // OK — not batched.
        other => panic!("single block should not be batched, got {other:?}"),
    }
}

#[test]
fn test_no_slot_available_skips() {
    // No IdentityPassthrough between style_step (1) and first rb (2).
    let mut steps = vec![
        CompiledStep::IdentityPassthrough, // 0: x
        CompiledStep::IdentityPassthrough, // 1: style
        // No identity slot here — step 2 is already a FusedResBlock.
        make_style_resblock(0, 1, 4, 4, 2),
        make_style_resblock(0, 1, 4, 4, 2),
    ];

    batch_style_projections(&mut steps);

    // Should be unchanged — no slot available.
    assert!(matches!(
        &steps[2],
        CompiledStep::NativeOp {
            op: NativeOpKind::FusedResBlock { style_proj: Some(_), .. },
            ..
        }
    ));
}

#[test]
fn test_weight_concatenation_order() {
    // Verify that weight data from block 0 appears before block 1.
    let mut steps = vec![
        CompiledStep::IdentityPassthrough,
        CompiledStep::IdentityPassthrough,
        CompiledStep::IdentityPassthrough,
        make_style_resblock(0, 1, 2, 2, 1), // style1_w = [1.0; 4], style2_w = [2.0; 4]
        make_style_resblock(0, 1, 2, 2, 1), // same pattern
    ];

    batch_style_projections(&mut steps);

    let weight_t = match &steps[2] {
        CompiledStep::NativeOp {
            op: NativeOpKind::BatchedStyleProjection { .. },
            weight_data,
        } => weight_data.get("weight").expect("should have 'weight'"),
        _ => panic!("expected BatchedStyleProjection"),
    };

    // Block 0: style1_weight [4 × 1] = [1,1,1,1], style2_weight [4 × 1] = [2,2,2,2]
    // Block 1: same
    // Concatenated: [1,1,1,1, 2,2,2,2, 1,1,1,1, 2,2,2,2] (total_out=16, style_dim=1)
    let data = weight_t.data();
    assert_eq!(data.len(), 16);
    // Block 0 style1 (4 elements of 1.0) then style2 (4 elements of 2.0)
    assert_eq!(&data[0..4], &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(&data[4..8], &[2.0, 2.0, 2.0, 2.0]);
    // Block 1 same
    assert_eq!(&data[8..12], &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(&data[12..16], &[2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn test_dispatch_count_improvement() {
    let params = NormActivConv1dParams {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 4, 100],
        output_channels: 4,
        kernel_size: 3,
    };

    // Before: 2 blocks × 7 (3 base + 4 proj) = 14 dispatches.
    let before: usize = [3, 4].iter().map(|&idx| {
        match &vec![
            CompiledStep::IdentityPassthrough,
            CompiledStep::IdentityPassthrough,
            CompiledStep::IdentityPassthrough,
            make_style_resblock(0, 1, 4, 4, 2),
            make_style_resblock(0, 1, 4, 4, 2),
        ][idx] {
            CompiledStep::NativeOp { op, .. } => op.estimated_metal_dispatches(),
            _ => 0,
        }
    }).sum();
    assert_eq!(before, 14); // 2 × 7

    let mut steps = vec![
        CompiledStep::IdentityPassthrough,
        CompiledStep::IdentityPassthrough,
        CompiledStep::IdentityPassthrough,
        make_style_resblock(0, 1, 4, 4, 2),
        make_style_resblock(0, 1, 4, 4, 2),
    ];

    batch_style_projections(&mut steps);

    // After: 1 BatchedStyleProjection (2) + 2 × FusedResBlock(3 base) = 8.
    let after: usize = steps.iter().filter_map(|s| match s {
        CompiledStep::NativeOp { op, .. } => Some(op.estimated_metal_dispatches()),
        _ => None,
    }).sum();
    assert_eq!(after, 8); // 2 + 3 + 3
}
