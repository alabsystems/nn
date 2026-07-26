// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for conv1d dispatch batching optimizer.
//!
//! Verifies batch detection, compatibility criteria, dispatch savings
//! calculations, and edge cases.
//!
//! Part of #4264.

use std::collections::HashMap;

use nn_dsl::trace_compile::CompiledStep;
use nn_dsl::NativeOpKind;

use super::{ConvBatchOptimizer, PipelineConvBatchSummary};

/// Helper: create a Conv1dGemm NativeOp step with the given parameters.
fn make_conv1d_step(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    has_bias: bool,
    l_in: usize,
) -> CompiledStep {
    CompiledStep::NativeOp {
        op: NativeOpKind::Conv1dGemm {
            input_shape: vec![1, in_channels, l_in],
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            has_bias,
        },
        weight_data: HashMap::new(),
    }
}

/// Kokoro-typical conv1d: K=3, S=1, P=1, D=1, groups=1.
fn kokoro_conv1d(in_channels: usize, out_channels: usize, has_bias: bool) -> CompiledStep {
    make_conv1d_step(in_channels, out_channels, 3, 1, 1, 1, 1, has_bias, 256)
}

/// A passthrough step (zero-cost, should not break batch runs).
fn passthrough_step() -> CompiledStep {
    CompiledStep::Passthrough {
        op_name: "reshape".to_string(),
        output_shape: vec![1, 128, 256],
    }
}

/// A non-conv dispatch step (should break batch runs).
fn layer_norm_step() -> CompiledStep {
    CompiledStep::NativeOp {
        op: NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 256, 128],
            hidden_dim: 128,
        },
        weight_data: HashMap::new(),
    }
}

#[test]
fn test_no_conv1ds_no_batches() {
    let steps = vec![passthrough_step(), layer_norm_step(), passthrough_step()];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("test_segment", &steps);
    assert!(!analysis.has_opportunities());
    assert_eq!(analysis.groups.len(), 0);
    assert_eq!(analysis.total_saved, 0);
}

#[test]
fn test_single_conv1d_no_batch() {
    let steps = vec![kokoro_conv1d(128, 256, true)];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("test_segment", &steps);
    assert!(!analysis.has_opportunities());
    assert_eq!(analysis.groups.len(), 0);
    assert_eq!(analysis.total_saved, 0);
}

#[test]
fn test_two_compatible_conv1ds_form_batch() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        kokoro_conv1d(128, 128, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    assert!(analysis.has_opportunities());
    assert_eq!(analysis.groups.len(), 1);

    let group = &analysis.groups[0];
    assert_eq!(group.batch_size(), 2);
    assert_eq!(group.kernel_size, 3);
    assert_eq!(group.stride, 1);
    assert_eq!(group.input_channels, 128);
    assert_eq!(group.output_channels, vec![256, 128]);
    assert_eq!(group.total_output_channels, 384);
    assert!(group.all_have_bias);
    assert!(group.any_have_bias);
}

#[test]
fn test_three_compatible_conv1ds_form_batch() {
    let steps = vec![
        kokoro_conv1d(512, 512, true),
        kokoro_conv1d(512, 256, true),
        kokoro_conv1d(512, 128, false),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    assert_eq!(analysis.groups.len(), 1);

    let group = &analysis.groups[0];
    assert_eq!(group.batch_size(), 3);
    assert_eq!(group.output_channels, vec![512, 256, 128]);
    assert_eq!(group.total_output_channels, 896);
    assert!(!group.all_have_bias);
    assert!(group.any_have_bias);
}

#[test]
fn test_incompatible_kernel_size_splits_batches() {
    let steps = vec![
        make_conv1d_step(128, 256, 3, 1, 1, 1, 1, true, 256),
        make_conv1d_step(128, 128, 3, 1, 1, 1, 1, true, 256),
        make_conv1d_step(128, 256, 7, 3, 3, 1, 1, true, 256), // different K and S
        make_conv1d_step(128, 128, 7, 3, 3, 1, 1, true, 256),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    assert_eq!(analysis.groups.len(), 2);
    assert_eq!(analysis.groups[0].kernel_size, 3);
    assert_eq!(analysis.groups[1].kernel_size, 7);
}

#[test]
fn test_incompatible_input_channels_splits_batches() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        kokoro_conv1d(128, 128, true),
        kokoro_conv1d(256, 512, true), // different C_in
        kokoro_conv1d(256, 256, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    assert_eq!(analysis.groups.len(), 2);
    assert_eq!(analysis.groups[0].input_channels, 128);
    assert_eq!(analysis.groups[1].input_channels, 256);
}

#[test]
fn test_passthrough_does_not_break_batch() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        passthrough_step(),
        kokoro_conv1d(128, 128, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    assert_eq!(analysis.groups.len(), 1);
    assert_eq!(analysis.groups[0].batch_size(), 2);
}

#[test]
fn test_layer_norm_breaks_batch() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        layer_norm_step(),
        kokoro_conv1d(128, 128, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    // Each single conv1d is below min_batch_size=2.
    assert_eq!(analysis.groups.len(), 0);
}

#[test]
fn test_dispatch_savings_two_conv1ds_with_bias() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        kokoro_conv1d(128, 128, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    let group = &analysis.groups[0];

    // K=3 S=1 D=1 → direct path: 1 conv + 1 bias = 2 per conv.
    // Unbatched: 2 * 2 = 4.
    assert_eq!(group.unbatched_dispatches(), 4);
    // Batched: 1 batched conv + 1 batched bias = 2.
    assert_eq!(group.batched_dispatches(), 2);
    // Saved: 4 - 2 = 2.
    assert_eq!(group.dispatches_saved(), 2);
    assert_eq!(analysis.total_saved, 2);
}

#[test]
fn test_dispatch_savings_three_conv1ds_no_bias() {
    let steps = vec![
        kokoro_conv1d(128, 256, false),
        kokoro_conv1d(128, 128, false),
        kokoro_conv1d(128, 64, false),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    let group = &analysis.groups[0];

    // K=3 S=1 D=1 → direct path: 1 conv per conv, no bias.
    // Unbatched: 3 * 1 = 3.
    assert_eq!(group.unbatched_dispatches(), 3);
    // Batched: 1 batched conv = 1.
    assert_eq!(group.batched_dispatches(), 1);
    // Saved: 3 - 1 = 2.
    assert_eq!(group.dispatches_saved(), 2);
}

#[test]
fn test_empty_steps() {
    let steps: Vec<CompiledStep> = vec![];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("empty", &steps);
    assert!(!analysis.has_opportunities());
    assert_eq!(analysis.total_dispatches, 0);
    assert_eq!(analysis.conv1d_dispatches, 0);
    assert_eq!(analysis.total_saved, 0);
    assert_eq!(analysis.optimized_dispatches, 0);
}

#[test]
fn test_min_batch_size_three() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        kokoro_conv1d(128, 128, true),
    ];
    let optimizer = ConvBatchOptimizer::with_min_batch_size(3);
    let analysis = optimizer.analyze("generator", &steps);
    // Two conv1ds is below min_batch_size=3.
    assert!(!analysis.has_opportunities());
}

#[test]
fn test_different_input_length_splits() {
    let steps = vec![
        make_conv1d_step(128, 256, 3, 1, 1, 1, 1, true, 256),
        make_conv1d_step(128, 128, 3, 1, 1, 1, 1, true, 512), // different L_in
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    assert!(!analysis.has_opportunities());
}

#[test]
fn test_grouped_conv_not_batched_with_regular() {
    let steps = vec![
        make_conv1d_step(128, 256, 3, 1, 1, 1, 1, true, 256),   // groups=1
        make_conv1d_step(128, 128, 3, 1, 1, 1, 128, true, 256),  // groups=128 (depthwise)
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    assert!(!analysis.has_opportunities());
}

#[test]
fn test_multiple_batch_groups_in_segment() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        kokoro_conv1d(128, 128, true),
        layer_norm_step(),
        kokoro_conv1d(256, 512, true),
        kokoro_conv1d(256, 256, true),
        kokoro_conv1d(256, 128, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    assert_eq!(analysis.groups.len(), 2);
    assert_eq!(analysis.groups[0].batch_size(), 2);
    assert_eq!(analysis.groups[1].batch_size(), 3);
}

#[test]
fn test_pipeline_summary() {
    let seg1_steps = vec![
        kokoro_conv1d(128, 256, true),
        kokoro_conv1d(128, 128, true),
    ];
    let seg2_steps = vec![
        kokoro_conv1d(256, 512, true),
        kokoro_conv1d(256, 256, true),
        kokoro_conv1d(256, 128, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analyses = vec![
        optimizer.analyze("prosody", &seg1_steps),
        optimizer.analyze("generator", &seg2_steps),
    ];
    let summary = PipelineConvBatchSummary::from_analyses(analyses);

    assert!(summary.has_opportunities());
    assert_eq!(summary.total_groups, 2);
    assert!(summary.total_saved > 0);
    assert!(summary.optimized_dispatches() < summary.total_dispatches);
}

#[test]
fn test_reduction_pct_nonzero() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        kokoro_conv1d(128, 128, true),
        kokoro_conv1d(128, 64, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    // Reduction percentage is savings / conv1d_dispatches * 100.
    // 3 conv1ds with bias → 6 Metal dispatches (2 each, K=3 direct path).
    // Batched: 3 dispatches (shared im2col + GEMM + bias).
    // Saved: 3, reduction = 3/6 * 100 = 50%.
    assert!(analysis.reduction_pct() > 0.0);
    assert!(analysis.reduction_pct() <= 100.0);
    assert!(analysis.conv1d_dispatches > 0);
}

#[test]
fn test_reduction_pct_zero_no_opportunities() {
    let steps = vec![layer_norm_step()];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("test", &steps);
    assert_eq!(analysis.reduction_pct(), 0.0);
}

#[test]
fn test_display_formatting() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        kokoro_conv1d(128, 128, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    let display = format!("{analysis}");
    assert!(display.contains("generator"));
    assert!(display.contains("conv1ds"));
    assert!(display.contains("saves"));
}

#[test]
fn test_kokoro_generator_typical_shapes() {
    // Kokoro generator segment has conv1d layers with these typical shapes:
    // Multiple conv1d K=3 S=1 with different channel sizes in the ResBlocks.
    let steps = vec![
        // ResBlock 1: two conv1d layers with same C_in
        kokoro_conv1d(512, 512, true),
        kokoro_conv1d(512, 512, true),
        // Activation + norm between blocks (breaks batch)
        layer_norm_step(),
        // ResBlock 2: different C_in from upsample
        kokoro_conv1d(256, 256, true),
        kokoro_conv1d(256, 256, true),
        kokoro_conv1d(256, 256, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);

    // Should find 2 batch groups.
    assert_eq!(analysis.groups.len(), 2);
    assert_eq!(analysis.groups[0].batch_size(), 2);
    assert_eq!(analysis.groups[0].input_channels, 512);
    assert_eq!(analysis.groups[1].batch_size(), 3);
    assert_eq!(analysis.groups[1].input_channels, 256);

    // Total savings should be meaningful.
    // Group 1 (2 conv, K=3, bias): unbatched=4, batched=2, saved=2.
    // Group 2 (3 conv, K=3, bias): unbatched=6, batched=2, saved=4.
    // Total saved: 6.
    assert!(analysis.total_saved >= 4);
}

#[test]
fn test_narrow_view_does_not_break_batch() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        CompiledStep::NarrowView {
            byte_offset: 0,
            output_shape: vec![1, 128, 256],
            source_step: None,
        },
        kokoro_conv1d(128, 128, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    assert_eq!(analysis.groups.len(), 1);
    assert_eq!(analysis.groups[0].batch_size(), 2);
}

#[test]
fn test_identity_passthrough_does_not_break_batch() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        CompiledStep::IdentityPassthrough,
        kokoro_conv1d(128, 128, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    assert_eq!(analysis.groups.len(), 1);
    assert_eq!(analysis.groups[0].batch_size(), 2);
}

#[test]
fn test_constant_value_does_not_break_batch() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        CompiledStep::ConstantValue {
            value: 0.0,
            shape: vec![1, 128, 256],
        },
        kokoro_conv1d(128, 128, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("generator", &steps);
    assert_eq!(analysis.groups.len(), 1);
    assert_eq!(analysis.groups[0].batch_size(), 2);
}

#[test]
fn test_step_indices_correct() {
    let steps = vec![
        passthrough_step(),        // idx 0
        kokoro_conv1d(128, 256, true), // idx 1
        passthrough_step(),        // idx 2
        kokoro_conv1d(128, 128, true), // idx 3
        layer_norm_step(),         // idx 4
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("test", &steps);
    assert_eq!(analysis.groups.len(), 1);
    assert_eq!(analysis.groups[0].step_indices, vec![1, 3]);
}

#[test]
fn test_mixed_bias_batch() {
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        kokoro_conv1d(128, 128, false),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("test", &steps);
    assert_eq!(analysis.groups.len(), 1);
    let group = &analysis.groups[0];
    assert!(!group.all_have_bias);
    assert!(group.any_have_bias);
}

#[test]
fn test_conv1d_dispatch_count_tracking() {
    // Conv1dGemm K=3 S=1 D=1 with bias = 2 Metal dispatches each (direct path).
    let steps = vec![
        kokoro_conv1d(128, 256, true),
        kokoro_conv1d(128, 128, true),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("test", &steps);
    // 2 NativeOp steps counted as logical dispatches.
    assert_eq!(analysis.total_dispatches, 2);
    // Conv1d Metal dispatches: K=3 S=1 D=1 direct path = 2 each (conv + bias) = 4 total.
    assert_eq!(analysis.conv1d_dispatches, 4);
    // Savings: 4 - 2 = 2 Metal dispatches saved.
    assert_eq!(analysis.total_saved, 2);
}

#[test]
fn test_dispatch_savings_im2col_path() {
    // K=7 S=3 → im2col+GEMM path: 2 base + 1 bias = 3 per conv.
    let steps = vec![
        make_conv1d_step(128, 256, 7, 3, 3, 1, 1, true, 256),
        make_conv1d_step(128, 128, 7, 3, 3, 1, 1, true, 256),
        make_conv1d_step(128, 64, 7, 3, 3, 1, 1, true, 256),
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("test", &steps);
    let group = &analysis.groups[0];

    // Unbatched: 3 * (2 + 1) = 9 (im2col + GEMM + bias each).
    assert_eq!(group.unbatched_dispatches(), 9);
    // Batched: 2 + 1 = 3 (shared im2col + batched GEMM + batched bias).
    assert_eq!(group.batched_dispatches(), 3);
    // Saved: 9 - 3 = 6.
    assert_eq!(group.dispatches_saved(), 6);
}

#[test]
fn test_dilation_affects_compatibility() {
    let steps = vec![
        make_conv1d_step(128, 256, 3, 1, 1, 1, 1, true, 256),
        make_conv1d_step(128, 128, 3, 1, 2, 2, 1, true, 256), // dilation=2
    ];
    let optimizer = ConvBatchOptimizer::default();
    let analysis = optimizer.analyze("test", &steps);
    assert!(!analysis.has_opportunities());
}
