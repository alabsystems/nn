// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for compiled model memory report generation.
//!
//! Part of #3828.

use std::collections::HashMap;

use nn_dsl::buffer_planner::BufferPlan;
use nn_dsl::trace_compile::CompiledStep;

use crate::compiled_model::{CompiledModelDef, StepMeta};
use crate::compiled_model_memory_report::{
    bytes_to_human, format_memory_report, generate_memory_report, StepMemoryReport,
};

/// Build a minimal `CompiledModelDef` with no steps for empty-plan tests.
fn empty_def() -> CompiledModelDef {
    CompiledModelDef {
        steps: Vec::new(),
        step_metas: Vec::new(),
        weight_buffers: Vec::new(),
        constant_buffers: HashMap::new(),
        num_inputs: 0,
        input_specs: Vec::new(),
        output_step_indices: Vec::new(),
        output_metas: Vec::new(),
        buffer_plan: BufferPlan {
            total_bytes: 0,
            step_offsets: Vec::new(),
            step_sizes: Vec::new(),
            naive_total: 0,
            last_use: Vec::new(),
        },
        precision: None,
        input_name_cache: Vec::new(),
        release_at: Vec::new(),
        mixed_precision_active: false,
        autocast_policy: None,
        autocast_active: false,
        mixed_gemm_infos: Vec::new(),
        proof_certificate: None,
        shape_policy: crate::compiled_model::ShapePolicy::Fixed,
        icb_eligible: Vec::new(),
        icb_segments: Vec::new(),
        icb_segment_starts: HashMap::new(),
        concurrent_barriers: Vec::new(),
    }
}

/// Build a `CompiledModelDef` with the given steps and buffer sizes.
/// Each step is a `ConstantValue` for simplicity (no GPU required).
fn def_with_steps(
    step_sizes: &[usize],
    total_bytes: usize,
    edges: &[Vec<usize>],
) -> CompiledModelDef {
    let n = step_sizes.len();
    let steps: Vec<CompiledStep> = (0..n)
        .map(|_| CompiledStep::ConstantValue {
            value: 0.0,
            shape: vec![1],
        })
        .collect();
    let step_metas: Vec<StepMeta> = (0..n)
        .map(|i| StepMeta {
            edges: if i < edges.len() {
                edges[i].clone()
            } else {
                Vec::new()
            },
            scalar_type: nn_dsl::ir::ScalarType::F32,
            numel: step_sizes[i] / 4, // F32 = 4 bytes per element
        })
        .collect();
    let naive_total: usize = step_sizes.iter().sum();
    CompiledModelDef {
        steps,
        step_metas,
        weight_buffers: (0..n).map(|_| HashMap::new()).collect(),
        constant_buffers: HashMap::new(),
        num_inputs: 0,
        input_specs: Vec::new(),
        output_step_indices: if n > 0 { vec![n - 1] } else { vec![] },
        output_metas: Vec::new(),
        buffer_plan: BufferPlan {
            total_bytes,
            step_offsets: step_sizes
                .iter()
                .scan(0usize, |acc, &sz| {
                    let offset = *acc;
                    *acc += sz;
                    Some(if sz > 0 { Some(offset) } else { None })
                })
                .collect(),
            step_sizes: step_sizes.to_vec(),
            naive_total,
            last_use: (0..n).collect(),
        },
        precision: None,
        input_name_cache: Vec::new(),
        release_at: (0..n).map(|_| Vec::new()).collect(),
        mixed_precision_active: false,
        autocast_policy: None,
        autocast_active: false,
        mixed_gemm_infos: vec![None; n],
        proof_certificate: None,
        shape_policy: crate::compiled_model::ShapePolicy::Fixed,
        icb_eligible: vec![false; n],
        icb_segments: Vec::new(),
        icb_segment_starts: HashMap::new(),
        concurrent_barriers: vec![false; n],
    }
}

#[test]
fn test_memory_report_empty_plan_produces_zero_values() {
    let def = empty_def();
    let report = generate_memory_report(&def);

    assert_eq!(report.total_weight_bytes, 0);
    assert_eq!(report.total_intermediate_bytes, 0);
    assert_eq!(report.peak_intermediate_bytes, 0);
    assert_eq!(report.num_weight_buffers, 0);
    assert_eq!(report.num_intermediate_buffers, 0);
    assert!(report.per_step_breakdown.is_empty());
}

#[test]
fn test_memory_report_single_step_reports_correct_sizes() {
    // Single step with 4096 bytes output.
    let def = def_with_steps(&[4096], 4096, &[vec![]]);
    let report = generate_memory_report(&def);

    assert_eq!(report.total_intermediate_bytes, 4096);
    assert_eq!(report.peak_intermediate_bytes, 4096);
    assert_eq!(report.num_intermediate_buffers, 1);
    assert_eq!(report.per_step_breakdown.len(), 1);
    assert_eq!(report.per_step_breakdown[0].output_bytes, 4096);
    assert_eq!(report.per_step_breakdown[0].input_bytes, 0);
    assert_eq!(report.per_step_breakdown[0].weight_bytes, 0);
}

#[test]
fn test_memory_report_multi_step_computes_peak_correctly() {
    // Three steps: 1024, 2048, 512 bytes output.
    // With buffer reuse, peak could be less than naive total (3584).
    // We set total_bytes to 2048 (the peak from reuse).
    let edges = vec![
        vec![],    // step 0: no inputs
        vec![0],   // step 1: reads from step 0
        vec![1],   // step 2: reads from step 1
    ];
    let def = def_with_steps(&[1024, 2048, 512], 2048, &edges);
    let report = generate_memory_report(&def);

    assert_eq!(report.total_intermediate_bytes, 1024 + 2048 + 512);
    assert_eq!(report.peak_intermediate_bytes, 2048);
    assert_eq!(report.num_intermediate_buffers, 3);

    // Verify per-step input bytes are computed from edges.
    assert_eq!(report.per_step_breakdown[0].input_bytes, 0);
    assert_eq!(report.per_step_breakdown[1].input_bytes, 1024);
    assert_eq!(report.per_step_breakdown[2].input_bytes, 2048);
}

#[test]
fn test_bytes_to_human_formatting_bytes() {
    assert_eq!(bytes_to_human(0), "0 B");
    assert_eq!(bytes_to_human(1), "1 B");
    assert_eq!(bytes_to_human(512), "512 B");
    assert_eq!(bytes_to_human(1023), "1023 B");
}

#[test]
fn test_bytes_to_human_formatting_kb() {
    assert_eq!(bytes_to_human(1024), "1.0 KB");
    assert_eq!(bytes_to_human(1536), "1.5 KB");
    assert_eq!(bytes_to_human(256 * 1024), "256.0 KB");
}

#[test]
fn test_bytes_to_human_formatting_mb() {
    assert_eq!(bytes_to_human(1024 * 1024), "1.0 MB");
    assert_eq!(bytes_to_human(1024 * 1024 + 512 * 1024), "1.5 MB");
    assert_eq!(bytes_to_human(100 * 1024 * 1024), "100.0 MB");
}

#[test]
fn test_bytes_to_human_formatting_gb() {
    assert_eq!(bytes_to_human(1024 * 1024 * 1024), "1.0 GB");
    assert_eq!(bytes_to_human(2 * 1024 * 1024 * 1024), "2.0 GB");
}

#[test]
fn test_format_memory_report_produces_valid_output() {
    let edges = vec![vec![], vec![0]];
    let def = def_with_steps(&[4096, 2048], 4096, &edges);
    let report = generate_memory_report(&def);
    let formatted = format_memory_report(&report);

    assert!(formatted.contains("Compiled Model Memory Report"));
    assert!(formatted.contains("Weight memory:"));
    assert!(formatted.contains("Intermediate (naive):"));
    assert!(formatted.contains("Intermediate (peak):"));
    assert!(formatted.contains("Per-Step Breakdown"));
    assert!(formatted.contains("ConstantValue"));
}

#[test]
fn test_in_place_ops_reported_correctly() {
    // Step with zero output bytes should be marked in-place.
    let _n = 3;
    let step_sizes = vec![4096, 0, 2048];
    let edges = vec![vec![], vec![0], vec![1]];
    let mut def = def_with_steps(&step_sizes, 4096, &edges);

    // Replace step 1 with an IdentityPassthrough (zero allocation, in-place).
    def.steps[1] = CompiledStep::IdentityPassthrough;

    let report = generate_memory_report(&def);

    assert!(!report.per_step_breakdown[0].is_in_place);
    assert!(report.per_step_breakdown[1].is_in_place);
    assert!(!report.per_step_breakdown[2].is_in_place);
}

#[test]
fn test_input_forward_not_marked_in_place() {
    // InputForward with 0 output bytes should NOT be marked in-place.
    let mut def = def_with_steps(&[0], 0, &[vec![]]);
    def.steps[0] = CompiledStep::InputForward;

    let report = generate_memory_report(&def);
    assert!(!report.per_step_breakdown[0].is_in_place);
}

#[test]
fn test_display_impl_matches_format() {
    let def = def_with_steps(&[1024], 1024, &[vec![]]);
    let report = generate_memory_report(&def);

    let display_output = format!("{report}");
    let format_output = format_memory_report(&report);
    assert_eq!(display_output, format_output);
}

#[test]
fn test_step_memory_report_display() {
    let step = StepMemoryReport {
        step_index: 3,
        step_name: "Dispatch(matmul)".to_string(),
        input_bytes: 1024,
        output_bytes: 2048,
        weight_bytes: 4096,
        is_in_place: false,
    };
    let display = format!("{step}");
    assert!(display.contains("Step 3"));
    assert!(display.contains("matmul"));
    assert!(display.contains("1.0 KB"));
    assert!(display.contains("2.0 KB"));
    assert!(display.contains("4.0 KB"));
}

#[test]
fn test_passthrough_step_name() {
    let mut def = def_with_steps(&[0], 0, &[vec![]]);
    def.steps[0] = CompiledStep::Passthrough {
        op_name: "reshape".to_string(),
        output_shape: vec![2, 3],
    };

    let report = generate_memory_report(&def);
    assert_eq!(
        report.per_step_breakdown[0].step_name,
        "Passthrough(reshape)"
    );
}
