// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for dispatch plan profiler infrastructure.
//!
//! Part of #4264.

use super::*;

fn make_entry(idx: usize, name: &str, dur: f64, mem: usize) -> ProfileEntry {
    ProfileEntry {
        step_index: idx,
        step_name: name.to_string(),
        duration_us: dur,
        memory_bytes: mem,
    }
}

#[test]
fn test_top_n_returns_sorted_by_duration() {
    let profile = DispatchProfile::from_entries(vec![
        make_entry(0, "input", 1.0, 0),
        make_entry(1, "matmul_0", 500.0, 4096),
        make_entry(2, "relu", 10.0, 1024),
        make_entry(3, "matmul_1", 300.0, 4096),
        make_entry(4, "softmax", 50.0, 1024),
    ]);

    let top3 = profile.top_n(3);
    assert_eq!(top3.len(), 3);
    assert_eq!(top3[0].step_name, "matmul_0");
    assert_eq!(top3[1].step_name, "matmul_1");
    assert_eq!(top3[2].step_name, "softmax");
}

#[test]
fn test_top_n_with_n_larger_than_entries() {
    let profile = DispatchProfile::from_entries(vec![
        make_entry(0, "add", 10.0, 100),
        make_entry(1, "mul", 20.0, 200),
    ]);

    let top5 = profile.top_n(5);
    assert_eq!(top5.len(), 2);
    assert_eq!(top5[0].step_name, "mul");
    assert_eq!(top5[1].step_name, "add");
}

#[test]
fn test_top_n_empty() {
    let profile = DispatchProfile::from_entries(vec![]);
    let top = profile.top_n(5);
    assert!(top.is_empty());
}

#[test]
fn test_by_category_groups_correctly() {
    let profile = DispatchProfile::from_entries(vec![
        make_entry(0, "matmul_fused_add", 500.0, 4096),
        make_entry(1, "linear_proj", 300.0, 2048),
        make_entry(2, "conv1d_k8", 200.0, 1024),
        make_entry(3, "LayerNorm", 100.0, 512),
        make_entry(4, "relu_act", 50.0, 256),
        make_entry(5, "LstmSequence", 150.0, 2048),
        make_entry(6, "input", 1.0, 0),
        make_entry(7, "softmax_v2", 80.0, 512),
        make_entry(8, "embedding_lookup", 30.0, 1024),
    ]);

    let cats = profile.by_category();

    // matmul: 500 + 300 = 800
    assert!((cats["matmul"] - 800.0).abs() < 1e-6);
    // conv: 200
    assert!((cats["conv"] - 200.0).abs() < 1e-6);
    // normalization: 100
    assert!((cats["normalization"] - 100.0).abs() < 1e-6);
    // elementwise: 50 (relu_act)
    assert!((cats["elementwise"] - 50.0).abs() < 1e-6);
    // lstm: 150
    assert!((cats["lstm"] - 150.0).abs() < 1e-6);
    // passthrough: 1 (input)
    assert!((cats["passthrough"] - 1.0).abs() < 1e-6);
    // softmax: 80
    assert!((cats["softmax"] - 80.0).abs() < 1e-6);
    // embedding: 30
    assert!((cats["embedding"] - 30.0).abs() < 1e-6);
}

#[test]
fn test_by_category_unknown_goes_to_other() {
    let profile = DispatchProfile::from_entries(vec![
        make_entry(0, "exotic_op_xyz", 42.0, 100),
    ]);

    let cats = profile.by_category();
    assert!((cats["other"] - 42.0).abs() < 1e-6);
}

#[test]
fn test_summary_contains_expected_sections() {
    let profile = DispatchProfile::from_entries(vec![
        make_entry(0, "matmul_0", 500.0, 4096),
        make_entry(1, "relu", 10.0, 1024),
        make_entry(2, "softmax", 50.0, 512),
    ]);

    let summary = profile.summary();
    assert!(summary.contains("DispatchProfile:"));
    assert!(summary.contains("By category:"));
    assert!(summary.contains("Top 10 slowest steps:"));
    assert!(summary.contains("matmul_0"));
    assert!(summary.contains("matmul"));
    assert!(summary.contains("elementwise"));
}

#[test]
fn test_summary_display_impl_matches() {
    let profile = DispatchProfile::from_entries(vec![
        make_entry(0, "add", 10.0, 100),
    ]);

    let display = format!("{profile}");
    assert_eq!(display, profile.summary());
}

#[test]
fn test_total_memory_bytes() {
    let profile = DispatchProfile::from_entries(vec![
        make_entry(0, "a", 1.0, 100),
        make_entry(1, "b", 2.0, 200),
        make_entry(2, "c", 3.0, 300),
    ]);

    assert_eq!(profile.total_memory_bytes(), 600);
}

#[test]
fn test_total_us_computed_correctly() {
    let profile = DispatchProfile::from_entries(vec![
        make_entry(0, "a", 10.5, 0),
        make_entry(1, "b", 20.3, 0),
        make_entry(2, "c", 30.2, 0),
    ]);

    assert!((profile.total_us - 61.0).abs() < 1e-6);
}

#[test]
fn test_categorize_step_coverage() {
    assert_eq!(categorize_step("matmul_fused"), "matmul");
    assert_eq!(categorize_step("gemm_tile"), "matmul");
    assert_eq!(categorize_step("linear_proj"), "matmul");
    assert_eq!(categorize_step("conv1d"), "conv");
    assert_eq!(categorize_step("conv2d_k3"), "conv");
    assert_eq!(categorize_step("LstmSequence"), "lstm");
    assert_eq!(categorize_step("self_attention"), "attention");
    assert_eq!(categorize_step("sdpa_flash"), "attention");
    assert_eq!(categorize_step("LayerNorm"), "normalization");
    assert_eq!(categorize_step("rmsnorm"), "normalization");
    assert_eq!(categorize_step("instance_norm"), "normalization");
    assert_eq!(categorize_step("softmax_v2"), "softmax");
    assert_eq!(categorize_step("log_softmax"), "softmax");
    assert_eq!(categorize_step("embedding_table"), "embedding");
    assert_eq!(categorize_step("gather_idx"), "embedding");
    assert_eq!(categorize_step("input"), "passthrough");
    assert_eq!(categorize_step("identity"), "passthrough");
    assert_eq!(categorize_step("reshape_3d"), "passthrough");
    assert_eq!(categorize_step("narrow_view"), "passthrough");
    assert_eq!(categorize_step("constant"), "passthrough");
    assert_eq!(categorize_step("snake_act"), "elementwise");
    assert_eq!(categorize_step("relu_6"), "elementwise");
    assert_eq!(categorize_step("gelu_approx"), "elementwise");
    assert_eq!(categorize_step("silu"), "elementwise");
    assert_eq!(categorize_step("sigmoid"), "elementwise");
    assert_eq!(categorize_step("fused_add_mul_x3"), "elementwise");
    assert_eq!(categorize_step("some_weird_op"), "other");
}

#[test]
fn test_estimate_step_bytes() {
    assert_eq!(estimate_step_bytes(1024, 4), 4096);
    assert_eq!(estimate_step_bytes(1024, 2), 2048);
    assert_eq!(estimate_step_bytes(0, 4), 0);
    // Saturating behavior for overflow
    assert_eq!(estimate_step_bytes(usize::MAX, 2), usize::MAX);
}

#[test]
fn test_format_bytes() {
    assert_eq!(format_bytes(0), "0 B");
    assert_eq!(format_bytes(512), "512 B");
    assert_eq!(format_bytes(1024), "1.0 KB");
    assert_eq!(format_bytes(2560), "2.5 KB");
    assert_eq!(format_bytes(1_048_576), "1.0 MB");
    assert_eq!(format_bytes(5_242_880), "5.0 MB");
}

#[test]
fn test_profile_error_display() {
    let err = ProfileError::TimingCountMismatch {
        plan_steps: 10,
        timing_count: 5,
    };
    assert_eq!(
        err.to_string(),
        "timing count mismatch: plan has 10 steps but 5 timings provided"
    );
}
