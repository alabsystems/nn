// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_benchmark_config_default_values() {
    let config = BenchmarkConfig::default();
    assert_eq!(config.warmup_iterations, 3);
    assert_eq!(config.measurement_iterations, 10);
    assert_eq!(config.image_width, 612);
    assert_eq!(config.image_height, 792);
    assert_eq!(config.regions_per_page, 20);
    assert_eq!(config.num_pages, 5);
}

#[test]
fn test_generate_random_regions_count() {
    let regions = generate_random_regions(10, 612, 792);
    assert_eq!(regions.len(), 10);
}

#[test]
fn test_generate_random_regions_zero() {
    let regions = generate_random_regions(0, 612, 792);
    assert!(regions.is_empty());
}

#[test]
fn test_generate_random_regions_class_coverage() {
    // With 10 regions, all 10 class types should be represented.
    let regions = generate_random_regions(10, 612, 792);
    let mut class_names: Vec<&str> = regions.iter().map(DocumentRegion::class_name).collect();
    class_names.sort_unstable();
    class_names.dedup();
    assert_eq!(class_names.len(), 10, "all 10 region classes should appear");
}

#[test]
fn test_generate_random_regions_bbox_within_bounds() {
    let w = 800;
    let h = 600;
    let regions = generate_random_regions(30, w, h);
    for region in &regions {
        let bbox = region.bbox();
        assert!(bbox[0] >= 0.0, "x1 must be non-negative");
        assert!(bbox[1] >= 0.0, "y1 must be non-negative");
        assert!(bbox[2] <= w as f32, "x2 must be within image width");
        assert!(bbox[3] <= h as f32, "y2 must be within image height");
        assert!(bbox[2] > bbox[0], "x2 must be greater than x1");
        assert!(bbox[3] > bbox[1], "y2 must be greater than y1");
    }
}

#[test]
fn test_generate_random_page_output_has_reading_order() {
    let page = generate_random_page_output(15);
    assert_eq!(page.regions.len(), 15);
    assert_eq!(page.reading_order.len(), 15);
    assert_eq!(page.width, 612);
    assert_eq!(page.height, 792);
}

#[test]
fn test_generate_random_document_structure() {
    let doc = generate_random_document(3, 10);
    assert_eq!(doc.pages.len(), 3);
    for page in &doc.pages {
        assert_eq!(page.regions.len(), 10);
    }
}

#[test]
fn test_bench_postprocess_returns_result() {
    let config = BenchmarkConfig {
        warmup_iterations: 1,
        measurement_iterations: 2,
        regions_per_page: 5,
        ..BenchmarkConfig::default()
    };
    let result = bench_postprocess(&config);
    assert_eq!(result.stage_name, "postprocess");
    assert!(result.duration_ms >= 0.0);
    assert_eq!(result.items_processed, 2 * 5);
    assert!(result.throughput >= 0.0);
}

#[test]
fn test_bench_export_json_returns_result() {
    let config = BenchmarkConfig {
        warmup_iterations: 1,
        measurement_iterations: 2,
        num_pages: 2,
        regions_per_page: 5,
        ..BenchmarkConfig::default()
    };
    let result = bench_export_json(&config).expect("export_json should succeed");
    assert_eq!(result.stage_name, "export_json");
    assert!(result.duration_ms >= 0.0);
    assert_eq!(result.items_processed, 2 * 2);
}

#[test]
fn test_bench_export_html_returns_result() {
    let config = BenchmarkConfig {
        warmup_iterations: 1,
        measurement_iterations: 2,
        num_pages: 2,
        regions_per_page: 5,
        ..BenchmarkConfig::default()
    };
    let result = bench_export_html(&config).expect("export_html should succeed");
    assert_eq!(result.stage_name, "export_html");
    assert!(result.duration_ms >= 0.0);
}

#[test]
fn test_bench_export_markdown_returns_result() {
    let config = BenchmarkConfig {
        warmup_iterations: 1,
        measurement_iterations: 2,
        num_pages: 2,
        regions_per_page: 5,
        ..BenchmarkConfig::default()
    };
    let result = bench_export_markdown(&config).expect("export_markdown should succeed");
    assert_eq!(result.stage_name, "export_markdown");
    assert!(result.duration_ms >= 0.0);
}

#[test]
fn test_bench_table_structure_returns_result() {
    let config = BenchmarkConfig {
        warmup_iterations: 1,
        measurement_iterations: 2,
        ..BenchmarkConfig::default()
    };
    let result = bench_table_structure(&config);
    assert_eq!(result.stage_name, "table_structure");
    assert!(result.duration_ms >= 0.0);
    assert_eq!(result.items_processed, 2);
}

#[test]
fn test_benchmark_summary_from_results() {
    let results = vec![
        BenchmarkResult {
            stage_name: "a".to_string(),
            duration_ms: 10.0,
            items_processed: 100,
            throughput: 10000.0,
        },
        BenchmarkResult {
            stage_name: "b".to_string(),
            duration_ms: 20.0,
            items_processed: 200,
            throughput: 10000.0,
        },
    ];
    let summary = BenchmarkSummary::from_results(results);
    assert_eq!(summary.results.len(), 2);
    assert!((summary.total_duration_ms - 30.0).abs() < f64::EPSILON);
}

#[test]
fn test_benchmark_summary_empty() {
    let summary = BenchmarkSummary::from_results(vec![]);
    assert!(summary.results.is_empty());
    assert!((summary.total_duration_ms - 0.0).abs() < f64::EPSILON);
    let report = summary.generate_report();
    assert!(report.contains("No benchmark results"));
}

#[test]
fn test_generate_report_contains_header_and_stats() {
    let results = vec![
        BenchmarkResult {
            stage_name: "postprocess".to_string(),
            duration_ms: 5.0,
            items_processed: 50,
            throughput: 10000.0,
        },
        BenchmarkResult {
            stage_name: "export_json".to_string(),
            duration_ms: 15.0,
            items_processed: 10,
            throughput: 666.7,
        },
    ];
    let summary = BenchmarkSummary::from_results(results);
    let report = summary.generate_report();

    assert!(report.contains("dpdf Pipeline Benchmark Report"));
    assert!(report.contains("postprocess"));
    assert!(report.contains("export_json"));
    assert!(report.contains("Min:"));
    assert!(report.contains("Max:"));
    assert!(report.contains("Mean:"));
    assert!(report.contains("P95:"));
    assert!(report.contains("Total:"));
}

#[test]
fn test_run_all_benchmarks_succeeds() {
    let config = BenchmarkConfig {
        warmup_iterations: 1,
        measurement_iterations: 1,
        regions_per_page: 3,
        num_pages: 1,
        ..BenchmarkConfig::default()
    };
    let summary = run_all_benchmarks(&config).expect("all benchmarks should succeed");
    assert_eq!(summary.results.len(), 5);
    let stage_names: Vec<&str> = summary
        .results
        .iter()
        .map(|r| r.stage_name.as_str())
        .collect();
    assert!(stage_names.contains(&"postprocess"));
    assert!(stage_names.contains(&"export_json"));
    assert!(stage_names.contains(&"export_html"));
    assert!(stage_names.contains(&"export_markdown"));
    assert!(stage_names.contains(&"table_structure"));
}

#[test]
fn test_compute_stats_single_value() {
    let stats = compute_stats(&[42.0]);
    assert!((stats.min - 42.0).abs() < f64::EPSILON);
    assert!((stats.max - 42.0).abs() < f64::EPSILON);
    assert!((stats.mean - 42.0).abs() < f64::EPSILON);
    assert!((stats.p95 - 42.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_stats_multiple_values() {
    let stats = compute_stats(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!((stats.min - 1.0).abs() < f64::EPSILON);
    assert!((stats.max - 5.0).abs() < f64::EPSILON);
    assert!((stats.mean - 3.0).abs() < f64::EPSILON);
    // P95 of 5 values: ceil(5 * 0.95) = 5, idx 4 (clamped to len-1=4) => 5.0
    assert!((stats.p95 - 5.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_stats_empty() {
    let stats = compute_stats(&[]);
    assert!((stats.min - 0.0).abs() < f64::EPSILON);
    assert!((stats.max - 0.0).abs() < f64::EPSILON);
    assert!((stats.mean - 0.0).abs() < f64::EPSILON);
    assert!((stats.p95 - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_generate_table_detections_structure() {
    let dets = generate_table_detections(3, 2);
    // 1 table + 3 rows + 2 cols = 6 detections.
    assert_eq!(dets.len(), 6);
    assert_eq!(dets[0].class_id, 0); // table
    assert_eq!(dets[1].class_id, 1); // row
    assert_eq!(dets[2].class_id, 1); // row
    assert_eq!(dets[3].class_id, 1); // row
    assert_eq!(dets[4].class_id, 2); // column
    assert_eq!(dets[5].class_id, 2); // column
}
