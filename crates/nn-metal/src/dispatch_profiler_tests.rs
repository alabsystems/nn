// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`DispatchProfiler`] and related types.

use super::*;

fn make_entry(
    step_idx: usize,
    op_name: &str,
    dispatch_type: DispatchType,
    start_ns: u64,
    end_ns: u64,
    input_bytes: usize,
    output_bytes: usize,
) -> DispatchProfileEntry {
    DispatchProfileEntry {
        step_idx,
        op_name: op_name.to_string(),
        dispatch_type,
        start_ns,
        end_ns,
        input_bytes,
        output_bytes,
    }
}

#[test]
fn test_profiler_disabled_by_default() {
    let profiler = DispatchProfiler::new();
    assert!(!profiler.is_enabled());
    assert!(profiler.is_empty());
    assert_eq!(profiler.len(), 0);
}

#[test]
fn test_profiler_record_when_disabled_is_noop() {
    let mut profiler = DispatchProfiler::new();
    profiler.record(make_entry(
        0,
        "matmul",
        DispatchType::StandardOp("matmul".into()),
        0,
        1000,
        4096,
        2048,
    ));
    assert_eq!(profiler.len(), 0);
}

#[test]
fn test_profiler_record_when_enabled() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();
    assert!(profiler.is_enabled());

    profiler.record(make_entry(
        0,
        "matmul",
        DispatchType::StandardOp("matmul".into()),
        0,
        50_000,
        4096,
        2048,
    ));
    assert_eq!(profiler.len(), 1);

    profiler.record(make_entry(
        1,
        "relu",
        DispatchType::StandardOp("relu".into()),
        50_000,
        55_000,
        2048,
        2048,
    ));
    assert_eq!(profiler.len(), 2);
}

#[test]
fn test_profiler_enable_disable_toggle() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();
    profiler.record(make_entry(
        0,
        "op1",
        DispatchType::StandardOp("op1".into()),
        0,
        1000,
        100,
        100,
    ));
    assert_eq!(profiler.len(), 1);

    profiler.disable();
    profiler.record(make_entry(
        1,
        "op2",
        DispatchType::StandardOp("op2".into()),
        1000,
        2000,
        100,
        100,
    ));
    assert_eq!(profiler.len(), 1); // still 1, second not recorded
}

#[test]
fn test_entry_duration_ns() {
    let entry = make_entry(
        0,
        "test",
        DispatchType::StandardOp("test".into()),
        1000,
        5000,
        0,
        0,
    );
    assert_eq!(entry.duration_ns(), 4000);
}

#[test]
fn test_entry_duration_us() {
    let entry = make_entry(
        0,
        "test",
        DispatchType::StandardOp("test".into()),
        0,
        10_000,
        0,
        0,
    );
    assert!((entry.duration_us() - 10.0).abs() < 1e-9);
}

#[test]
fn test_entry_total_bytes() {
    let entry = make_entry(
        0,
        "test",
        DispatchType::StandardOp("test".into()),
        0,
        1000,
        4096,
        2048,
    );
    assert_eq!(entry.total_bytes(), 6144);
}

#[test]
fn test_entry_bandwidth_gbps() {
    // 1 GB in 1 second = 1 GB/s
    let entry = make_entry(
        0,
        "test",
        DispatchType::StandardOp("test".into()),
        0,
        1_000_000_000,  // 1 second
        500_000_000,     // 500 MB in
        500_000_000,     // 500 MB out = 1 GB total
    );
    let bw = entry.bandwidth_gbps();
    assert!((bw - 1.0).abs() < 0.01, "expected ~1.0 GB/s, got {bw}");
}

#[test]
fn test_entry_bandwidth_zero_duration() {
    let entry = make_entry(
        0,
        "test",
        DispatchType::StandardOp("test".into()),
        100,
        100,
        4096,
        2048,
    );
    assert_eq!(entry.bandwidth_gbps(), 0.0);
}

#[test]
fn test_total_dispatch_ns() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();
    profiler.record(make_entry(
        0,
        "a",
        DispatchType::StandardOp("a".into()),
        0,
        1000,
        0,
        0,
    ));
    profiler.record(make_entry(
        1,
        "b",
        DispatchType::NativeOp("b".into()),
        1000,
        4000,
        0,
        0,
    ));
    assert_eq!(profiler.total_dispatch_ns(), 4000);
}

#[test]
fn test_total_memory_bytes() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();
    profiler.record(make_entry(
        0,
        "a",
        DispatchType::StandardOp("a".into()),
        0,
        1000,
        1024,
        512,
    ));
    profiler.record(make_entry(
        1,
        "b",
        DispatchType::StandardOp("b".into()),
        1000,
        2000,
        256,
        128,
    ));
    // (1024+512) + (256+128) = 1920
    assert_eq!(profiler.total_memory_bytes(), 1920);
}

#[test]
fn test_memory_bandwidth_gbps() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();
    // 1 GB transferred in 1 second = 1 GB/s
    profiler.record(make_entry(
        0,
        "big",
        DispatchType::StandardOp("big".into()),
        0,
        1_000_000_000,
        1_073_741_824,
        0,
    ));
    let bw = profiler.memory_bandwidth_gbps();
    assert!((bw - 1.073_741_824).abs() < 0.01);
}

#[test]
fn test_memory_bandwidth_empty() {
    let profiler = DispatchProfiler::new();
    assert_eq!(profiler.memory_bandwidth_gbps(), 0.0);
}

#[test]
fn test_top_n_sorting() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();
    profiler.record(make_entry(
        0,
        "fast",
        DispatchType::StandardOp("fast".into()),
        0,
        100,
        0,
        0,
    ));
    profiler.record(make_entry(
        1,
        "slow",
        DispatchType::StandardOp("slow".into()),
        100,
        10_100,
        0,
        0,
    ));
    profiler.record(make_entry(
        2,
        "medium",
        DispatchType::StandardOp("medium".into()),
        10_100,
        15_100,
        0,
        0,
    ));

    let top = profiler.top_n(2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].op_name, "slow");
    assert_eq!(top[0].duration_ns(), 10_000);
    assert_eq!(top[1].op_name, "medium");
    assert_eq!(top[1].duration_ns(), 5000);
}

#[test]
fn test_top_n_exceeds_entries() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();
    profiler.record(make_entry(
        0,
        "only",
        DispatchType::StandardOp("only".into()),
        0,
        1000,
        0,
        0,
    ));

    let top = profiler.top_n(10);
    assert_eq!(top.len(), 1);
}

#[test]
fn test_report_by_type_breakdown() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();
    profiler.record(make_entry(
        0,
        "lstm_fused",
        DispatchType::NativeOp("lstm".into()),
        0,
        3000,
        1024,
        512,
    ));
    profiler.record(make_entry(
        1,
        "matmul",
        DispatchType::StandardOp("matmul".into()),
        3000,
        8000,
        2048,
        1024,
    ));
    profiler.record(make_entry(
        2,
        "chain_relu_add",
        DispatchType::FusedKernel("relu_add".into()),
        8000,
        9000,
        512,
        512,
    ));

    let report = profiler.report();
    assert_eq!(report.total_dispatches, 3);
    assert_eq!(report.total_ns, 9000);

    assert!(report.by_type.contains_key("native_op"));
    assert!(report.by_type.contains_key("standard_op"));
    assert!(report.by_type.contains_key("fused_kernel"));

    let native = &report.by_type["native_op"];
    assert_eq!(native.count, 1);
    assert_eq!(native.total_ns, 3000);

    let standard = &report.by_type["standard_op"];
    assert_eq!(standard.count, 1);
    assert_eq!(standard.total_ns, 5000);
}

#[test]
fn test_report_top_10() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();
    for i in 0..15 {
        profiler.record(make_entry(
            i,
            &format!("op_{i}"),
            DispatchType::StandardOp(format!("op_{i}")),
            i as u64 * 1000,
            i as u64 * 1000 + (i as u64 + 1) * 100,
            0,
            0,
        ));
    }
    let report = profiler.report();
    assert_eq!(report.top_10.len(), 10);
    // The slowest should be op_14 (duration = 1500ns)
    assert_eq!(report.top_10[0].step_idx, 14);
}

#[test]
fn test_fusion_opportunities() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();

    // Two consecutive element-wise ops with matching sizes -> fusion opportunity
    profiler.record(make_entry(
        0,
        "relu",
        DispatchType::StandardOp("relu".into()),
        0,
        1000,
        4096,
        4096,
    ));
    profiler.record(make_entry(
        1,
        "add",
        DispatchType::StandardOp("add".into()),
        1000,
        2000,
        4096,
        4096,
    ));
    // A matmul breaks the chain (not fusable)
    profiler.record(make_entry(
        2,
        "matmul",
        DispatchType::StandardOp("matmul".into()),
        2000,
        10_000,
        4096,
        2048,
    ));

    let report = profiler.report();
    assert_eq!(report.fusion_opportunities.len(), 1);
    let opp = &report.fusion_opportunities[0];
    assert_eq!(opp.first_idx, 0);
    assert_eq!(opp.second_idx, 1);
    assert_eq!(opp.first_op, "relu");
    assert_eq!(opp.second_op, "add");
    assert_eq!(opp.combined_ns, 2000);
    assert_eq!(opp.saved_bytes, 4096);
}

#[test]
fn test_no_fusion_when_sizes_differ() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();

    profiler.record(make_entry(
        0,
        "relu",
        DispatchType::StandardOp("relu".into()),
        0,
        1000,
        4096,
        4096,
    ));
    profiler.record(make_entry(
        1,
        "add",
        DispatchType::StandardOp("add".into()),
        1000,
        2000,
        2048, // different from previous output
        2048,
    ));

    let report = profiler.report();
    assert!(report.fusion_opportunities.is_empty());
}

#[test]
fn test_json_serialization() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();
    profiler.record(make_entry(
        0,
        "matmul",
        DispatchType::StandardOp("matmul".into()),
        0,
        50_000,
        4096,
        2048,
    ));

    let report = profiler.report();
    let json = report.to_json().expect("serialization should succeed");

    // Parse back to verify round-trip
    let parsed: DispatchProfileReport =
        serde_json::from_str(&json).expect("deserialization should succeed");
    assert_eq!(parsed.total_dispatches, 1);
    assert_eq!(parsed.total_ns, 50_000);
    assert_eq!(parsed.total_bytes, 6144);
}

#[test]
fn test_report_display_format() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();
    profiler.record(make_entry(
        0,
        "matmul",
        DispatchType::StandardOp("matmul".into()),
        0,
        100_000,
        8192,
        4096,
    ));

    let report = profiler.report();
    let display = format!("{report}");
    assert!(display.contains("Dispatch Profile Report"));
    assert!(display.contains("standard_op"));
    assert!(display.contains("matmul"));
}

#[test]
fn test_clear_entries() {
    let mut profiler = DispatchProfiler::new();
    profiler.enable();
    profiler.record(make_entry(
        0,
        "test",
        DispatchType::StandardOp("test".into()),
        0,
        1000,
        0,
        0,
    ));
    assert_eq!(profiler.len(), 1);

    profiler.clear();
    assert!(profiler.is_empty());
    assert_eq!(profiler.total_dispatch_ns(), 0);
}

#[test]
fn test_dispatch_type_category() {
    assert_eq!(DispatchType::NativeOp("x".into()).category(), "native_op");
    assert_eq!(
        DispatchType::FusedKernel("x".into()).category(),
        "fused_kernel"
    );
    assert_eq!(
        DispatchType::StandardOp("x".into()).category(),
        "standard_op"
    );
}

#[test]
fn test_dispatch_type_name() {
    assert_eq!(DispatchType::NativeOp("lstm".into()).name(), "lstm");
    assert_eq!(
        DispatchType::FusedKernel("relu_add".into()).name(),
        "relu_add"
    );
    assert_eq!(DispatchType::StandardOp("matmul".into()).name(), "matmul");
}

#[test]
fn test_dispatch_type_display() {
    let dt = DispatchType::NativeOp("lstm_seq".into());
    assert_eq!(format!("{dt}"), "NativeOp(lstm_seq)");
}

#[test]
fn test_empty_report() {
    let profiler = DispatchProfiler::new();
    let report = profiler.report();
    assert_eq!(report.total_dispatches, 0);
    assert_eq!(report.total_ns, 0);
    assert_eq!(report.total_bytes, 0);
    assert_eq!(report.bandwidth_gbps, 0.0);
    assert!(report.top_10.is_empty());
    assert!(report.fusion_opportunities.is_empty());
    assert!(report.by_type.is_empty());
}

#[test]
fn test_entry_saturating_sub_start_after_end() {
    // start > end should not panic, saturates to 0
    let entry = make_entry(
        0,
        "weird",
        DispatchType::StandardOp("weird".into()),
        5000,
        1000,
        0,
        0,
    );
    assert_eq!(entry.duration_ns(), 0);
    assert_eq!(entry.bandwidth_gbps(), 0.0);
}

#[test]
fn test_default_trait() {
    let profiler = DispatchProfiler::default();
    assert!(!profiler.is_enabled());
    assert!(profiler.is_empty());
}
