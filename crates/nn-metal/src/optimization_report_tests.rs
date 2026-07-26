// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Helper: create a minimal PerformanceReport for testing.
fn test_perf_report() -> nn_dsl::PerformanceReport {
    let seg = nn_dsl::SegmentPerformance::new("text_pipeline");
    nn_dsl::PerformanceReport::from_segments("kokoro", vec![seg])
}

#[test]
fn test_new_report_defaults() {
    let perf = test_perf_report();
    let report = OptimizationReport::new(0, "kokoro", &perf).expect("valid report");
    assert_eq!(report.version, OptimizationReport::CURRENT_VERSION);
    assert_eq!(report.iteration, 0);
    assert_eq!(report.model_name, "kokoro");
    assert!(report.bounds.is_none());
    assert!(report.parity.is_none());
    assert!(report.certificate.is_none());
    assert!(report.fusion_certificates.is_empty());
    assert!(report.recommendations.is_empty());
    assert!(report.contract_status.is_none());
}

#[test]
fn test_builder_methods() {
    let perf = test_perf_report();
    let bounds = serde_json::json!({"layer_count": 12});
    let parity = serde_json::json!({"max_error": 0.001});
    let cert = serde_json::json!({"quality": "pass"});
    let contract = ContractStatus::passing();

    let report = OptimizationReport::new(1, "kokoro", &perf)
        .expect("valid report")
        .with_bounds(&bounds)
        .with_parity(&parity)
        .with_certificate(&cert)
        .with_contract_status(contract);

    assert!(report.bounds.is_some());
    assert!(report.parity.is_some());
    assert!(report.certificate.is_some());
    assert!(report.contract_status.is_some());
    let cs = report.contract_status.as_ref().unwrap();
    assert!(cs.all_bounds_satisfied);
    assert!(cs.violations.is_empty());
}

#[test]
fn test_add_recommendation_and_fusion_cert() {
    let perf = test_perf_report();
    let mut report = OptimizationReport::new(0, "kokoro", &perf).expect("valid report");
    report.add_recommendation("reduce dispatches in text_pipeline");
    report.add_fusion_certificate(serde_json::json!({"pair": "a+b"}));
    assert_eq!(report.recommendations.len(), 1);
    assert_eq!(report.fusion_certificates.len(), 1);
}

#[test]
fn test_json_roundtrip() {
    let perf = test_perf_report();
    let report = OptimizationReport::new(2, "whisper", &perf).expect("valid report");
    let json = report.to_json().expect("serializes");
    let parsed: OptimizationReport = serde_json::from_str(&json).expect("deserializes");
    assert_eq!(parsed.model_name, "whisper");
    assert_eq!(parsed.iteration, 2);
}

#[test]
fn test_save_load_roundtrip() {
    let perf = test_perf_report();
    let report = OptimizationReport::new(3, "demucs", &perf)
        .expect("valid report")
        .with_contract_status(ContractStatus::passing());

    let dir = std::env::temp_dir().join(format!("opt_report_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("report.json");

    report.save(&path).expect("save succeeds");
    let loaded = OptimizationReport::load(&path).expect("load succeeds");
    assert_eq!(loaded.model_name, "demucs");
    assert_eq!(loaded.iteration, 3);
    assert!(loaded.contract_status.is_some());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_generate_bounds_recommendations_norm_chain() {
    let perf = test_perf_report();
    let bounds = serde_json::json!({
        "recommendations": [
            {"NormChainExplosion": {"chain_depth": 12, "total_expansion": 45.5}},
            {"PrecisionRisk": {"chained_norm_depth": 8, "precision_drift_ratio": 1.0023, "drift_per_layer": 0.000287}}
        ]
    });
    let mut report = OptimizationReport::new(0, "kokoro", &perf)
        .expect("valid report")
        .with_bounds(&bounds);

    report.generate_bounds_recommendations();
    assert_eq!(report.recommendations.len(), 2);
    assert!(report.recommendations[0].contains("NORM_CHAIN_EXPLOSION"));
    assert!(report.recommendations[0].contains("12-layer"));
    assert!(report.recommendations[1].contains("PRECISION_RISK"));
}

#[test]
fn test_generate_bounds_recommendations_no_bounds() {
    let perf = test_perf_report();
    let mut report = OptimizationReport::new(0, "kokoro", &perf).expect("valid report");
    report.generate_bounds_recommendations();
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_generate_flush_recommendations_within_budget() {
    let perf = test_perf_report();
    let mut report = OptimizationReport::new(0, "kokoro", &perf).expect("valid report");
    // Performance has 0 flushes/submits by default — should generate nothing.
    report.generate_flush_recommendations();
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_generate_flush_recommendations_exceeded() {
    let seg = nn_dsl::SegmentPerformance::new("text_pipeline");
    let perf = nn_dsl::PerformanceReport::from_segments("kokoro", vec![seg])
        .with_gpu_sync_stats(5, 2, 100);
    let mut report = OptimizationReport::new(0, "kokoro", &perf).expect("valid report");

    report.generate_flush_recommendations();
    assert_eq!(report.recommendations.len(), 2);
    assert!(report.recommendations[0].contains("FLUSH_BUDGET_EXCEEDED"));
    assert!(report.recommendations[0].contains("5"));
    assert!(report.recommendations[1].contains("SUBMIT_REGRESSION"));
    assert!(report.recommendations[1].contains("2"));
}

#[test]
fn test_generate_dispatch_recommendations_hotspot() {
    let mut hot = nn_dsl::SegmentPerformance::new("text_pipeline");
    hot.dispatches = 80;
    let mut cold = nn_dsl::SegmentPerformance::new("decoder");
    cold.dispatches = 20;
    let perf = nn_dsl::PerformanceReport::from_segments("kokoro", vec![hot, cold]);
    let mut report = OptimizationReport::new(0, "kokoro", &perf).expect("valid report");

    report.generate_dispatch_recommendations();
    assert_eq!(report.recommendations.len(), 1);
    assert!(report.recommendations[0].contains("text_pipeline"));
    assert!(report.recommendations[0].contains("80%"));
}

#[test]
fn test_generate_dispatch_recommendations_no_hotspot() {
    let mut a = nn_dsl::SegmentPerformance::new("a");
    a.dispatches = 30;
    let mut b = nn_dsl::SegmentPerformance::new("b");
    b.dispatches = 35;
    let mut c = nn_dsl::SegmentPerformance::new("c");
    c.dispatches = 35;
    let perf = nn_dsl::PerformanceReport::from_segments("kokoro", vec![a, b, c]);
    let mut report = OptimizationReport::new(0, "kokoro", &perf).expect("valid report");

    report.generate_dispatch_recommendations();
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_contract_status_passing() {
    let cs = ContractStatus::passing();
    assert!(cs.all_bounds_satisfied);
    assert!(cs.violations.is_empty());
    assert!(cs.tightened_bounds.is_empty());
}
