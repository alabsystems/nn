// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Metal command encoder batching.
//!
//! Tests grouping logic, reshape-only detection, CPU readback boundary
//! detection, and statistics tracking.

use nn_dsl::ir::{NodeId, ScalarType};
use nn_dsl::{DispatchStep, TensorNodeId};

use super::{has_cpu_readback_boundary, BatchStats, EncoderBatchPlanner, EncoderGroup};

/// Helper: create a dummy Elementwise step.
fn elementwise(input: usize, output: usize) -> DispatchStep {
    DispatchStep::Elementwise {
        kernel_name: format!("ew_{input}_{output}"),
        scalar_kernel: nn_dsl::ir::KernelDef::new(
            format!("ew_{input}_{output}"),
            vec![],
            ScalarType::F32,
            vec![],
            NodeId::new(0),
        ),
        inputs: vec![TensorNodeId::new(input)],
        output: TensorNodeId::new(output),
        total_elements: 1024,
    }
}

/// Helper: create a Reshape step.
fn reshape(input: usize, output: usize) -> DispatchStep {
    DispatchStep::Reshape {
        input: TensorNodeId::new(input),
        output: TensorNodeId::new(output),
    }
}

/// Helper: create a Reduce step.
fn reduce(input: usize, output: usize) -> DispatchStep {
    DispatchStep::Reduce {
        kernel_name: format!("reduce_{input}_{output}"),
        op: nn_dsl::tensor_ir::ReduceOp::Sum,
        dtype: ScalarType::F32,
        input: TensorNodeId::new(input),
        output: TensorNodeId::new(output),
        reduce_dim: 64,
        outer_size: 16,
        keepdim: false,
    }
}

/// Helper: create a Sigmoid step.
fn sigmoid(input: usize, output: usize) -> DispatchStep {
    DispatchStep::Sigmoid {
        kernel_name: format!("sigmoid_{input}_{output}"),
        dtype: ScalarType::F32,
        input: TensorNodeId::new(input),
        output: TensorNodeId::new(output),
        total_elements: 512,
    }
}

#[test]
fn test_empty_plan() {
    let groups = EncoderBatchPlanner::plan(&[]);
    assert!(groups.is_empty());
}

#[test]
fn test_single_compute_step() {
    let steps = vec![elementwise(0, 1)];
    let (groups, stats) = EncoderBatchPlanner::plan_with_stats(&steps);

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0], EncoderGroup { start: 0, end: 1, reshape_only: false });

    assert_eq!(stats.total_steps, 1);
    assert_eq!(stats.encoders_before, 1);
    assert_eq!(stats.encoders_after, 1);
    assert_eq!(stats.encoders_saved(), 0);
}

#[test]
fn test_single_reshape_step() {
    let steps = vec![reshape(0, 1)];
    let (groups, stats) = EncoderBatchPlanner::plan_with_stats(&steps);

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0], EncoderGroup { start: 0, end: 1, reshape_only: true });

    assert_eq!(stats.encoders_before, 0);
    assert_eq!(stats.encoders_after, 0);
    assert_eq!(stats.reshape_only_groups, 1);
}

#[test]
fn test_consecutive_compute_steps_grouped() {
    let steps = vec![
        elementwise(0, 1),
        reduce(1, 2),
        sigmoid(2, 3),
        elementwise(3, 4),
    ];
    let (groups, stats) = EncoderBatchPlanner::plan_with_stats(&steps);

    // All compute steps should be in one group.
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0], EncoderGroup { start: 0, end: 4, reshape_only: false });

    assert_eq!(stats.encoders_before, 4);
    assert_eq!(stats.encoders_after, 1);
    assert_eq!(stats.encoders_saved(), 3);
}

#[test]
fn test_reshape_between_compute_creates_three_groups() {
    let steps = vec![
        elementwise(0, 1),
        elementwise(1, 2),
        reshape(2, 3),
        reshape(3, 4),
        elementwise(4, 5),
    ];
    let (groups, stats) = EncoderBatchPlanner::plan_with_stats(&steps);

    assert_eq!(groups.len(), 3);
    assert_eq!(groups[0], EncoderGroup { start: 0, end: 2, reshape_only: false });
    assert_eq!(groups[1], EncoderGroup { start: 2, end: 4, reshape_only: true });
    assert_eq!(groups[2], EncoderGroup { start: 4, end: 5, reshape_only: false });

    assert_eq!(stats.encoders_before, 3);
    assert_eq!(stats.encoders_after, 2);
    assert_eq!(stats.reshape_only_groups, 1);
}

#[test]
fn test_interleaved_reshape_with_compute_stays_in_group() {
    // A single reshape between compute steps: because the reshape is sandwiched
    // between compute steps, the transition logic splits at boundaries.
    let steps = vec![
        elementwise(0, 1),
        reshape(1, 2),
        elementwise(2, 3),
    ];
    let (groups, _stats) = EncoderBatchPlanner::plan_with_stats(&steps);

    // Compute, then reshape starts a new group, then compute starts another.
    assert_eq!(groups.len(), 3);
    assert!(!groups[0].reshape_only);
    assert!(groups[1].reshape_only);
    assert!(!groups[2].reshape_only);
}

#[test]
fn test_kokoro_like_plan_reduces_encoders() {
    // Simulate a Kokoro-like dispatch sequence:
    // 6 compute + 1 reshape + 4 compute + 2 reshape + 3 compute.
    let mut steps = Vec::new();
    for i in 0..6 {
        steps.push(elementwise(i, i + 1));
    }
    steps.push(reshape(6, 7));
    for i in 7..11 {
        steps.push(elementwise(i, i + 1));
    }
    steps.push(reshape(11, 12));
    steps.push(reshape(12, 13));
    for i in 13..16 {
        steps.push(elementwise(i, i + 1));
    }

    let (groups, stats) = EncoderBatchPlanner::plan_with_stats(&steps);

    // 3 compute groups + 2 reshape-only groups.
    assert_eq!(groups.len(), 5);
    assert_eq!(stats.encoders_before, 13); // 13 non-reshape steps
    assert_eq!(stats.encoders_after, 3);   // 3 compute groups
    assert_eq!(stats.encoders_saved(), 10);
    assert_eq!(stats.reshape_only_groups, 2);
}

#[test]
fn test_all_reshapes() {
    let steps = vec![reshape(0, 1), reshape(1, 2), reshape(2, 3)];
    let (groups, stats) = EncoderBatchPlanner::plan_with_stats(&steps);

    assert_eq!(groups.len(), 1);
    assert!(groups[0].reshape_only);
    assert_eq!(stats.encoders_before, 0);
    assert_eq!(stats.encoders_after, 0);
}

#[test]
fn test_cpu_readback_boundary_false_for_compute_plans() {
    let steps = vec![
        elementwise(0, 1),
        reduce(1, 2),
        elementwise(2, 3),
    ];
    // Current dispatch plans have no inline CPU readback.
    assert!(!has_cpu_readback_boundary(&steps));
}

#[test]
fn test_cpu_readback_boundary_empty() {
    assert!(!has_cpu_readback_boundary(&[]));
}

#[test]
fn test_group_len_and_is_empty() {
    let group = EncoderGroup { start: 2, end: 7, reshape_only: false };
    assert_eq!(group.len(), 5);
    assert!(!group.is_empty());

    let empty = EncoderGroup { start: 3, end: 3, reshape_only: true };
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
}

#[test]
fn test_stats_avg_dispatches_per_encoder() {
    let stats = BatchStats {
        total_steps: 10,
        encoders_before: 8,
        encoders_after: 2,
        group_count: 3,
        reshape_only_groups: 1,
    };
    // (10 - 1) / 2 = 4.5
    assert!((stats.avg_dispatches_per_encoder() - 4.5).abs() < f64::EPSILON);
}

#[test]
fn test_stats_avg_dispatches_zero_encoders() {
    let stats = BatchStats {
        total_steps: 3,
        encoders_before: 0,
        encoders_after: 0,
        group_count: 1,
        reshape_only_groups: 1,
    };
    assert!((stats.avg_dispatches_per_encoder() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_leading_reshapes() {
    let steps = vec![
        reshape(0, 1),
        reshape(1, 2),
        elementwise(2, 3),
        elementwise(3, 4),
    ];
    let (groups, stats) = EncoderBatchPlanner::plan_with_stats(&steps);

    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0], EncoderGroup { start: 0, end: 2, reshape_only: true });
    assert_eq!(groups[1], EncoderGroup { start: 2, end: 4, reshape_only: false });
    assert_eq!(stats.encoders_after, 1);
}

#[test]
fn test_trailing_reshapes() {
    let steps = vec![
        elementwise(0, 1),
        elementwise(1, 2),
        reshape(2, 3),
        reshape(3, 4),
    ];
    let (groups, stats) = EncoderBatchPlanner::plan_with_stats(&steps);

    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0], EncoderGroup { start: 0, end: 2, reshape_only: false });
    assert_eq!(groups[1], EncoderGroup { start: 2, end: 4, reshape_only: true });
    assert_eq!(stats.encoders_after, 1);
}

#[test]
fn test_mixed_compute_types_same_group() {
    // Different compute step types should still be in the same group.
    let steps = vec![
        elementwise(0, 1),
        reduce(1, 2),
        sigmoid(2, 3),
    ];
    let groups = EncoderBatchPlanner::plan(&steps);

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].len(), 3);
    assert!(!groups[0].reshape_only);
}
