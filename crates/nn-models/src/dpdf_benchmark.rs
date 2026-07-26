// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-stage benchmark infrastructure for the dpdf document inference pipeline.
//!
//! Provides [`BenchmarkConfig`], [`BenchmarkResult`], and [`BenchmarkSummary`]
//! types for measuring throughput and latency of individual pipeline stages
//! (postprocess, export, table structure), plus synthetic data generators
//! for reproducible benchmarks without real document images.
//!
//! # Usage
//!
//! ```rust,no_run
//! use nn_models::dpdf_benchmark::{BenchmarkConfig, bench_postprocess, BenchmarkSummary};
//!
//! let config = BenchmarkConfig::default();
//! let results = vec![bench_postprocess(&config)];
//! let summary = BenchmarkSummary::from_results(results);
//! println!("{}", summary.generate_report());
//! ```

use std::time::Instant;

use crate::dpdf_export::{
    DocumentExporter, ExportError, HtmlExporter, JsonExporter, MarkdownExporter,
};
use crate::dpdf_pipeline::{DocumentOutput, DocumentRegion, DpdfPipeline, PageOutput};
use crate::dpdf_postprocess::{postprocess, PostProcessConfig};
use crate::table_structure::{self, TableStructureConfig};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration controlling benchmark execution parameters.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations before measurement (default 3).
    pub warmup_iterations: usize,
    /// Number of measurement iterations (default 10).
    pub measurement_iterations: usize,
    /// Synthetic image width in pixels (default 612).
    pub image_width: usize,
    /// Synthetic image height in pixels (default 792).
    pub image_height: usize,
    /// Number of regions per page for synthetic data (default 20).
    pub regions_per_page: usize,
    /// Number of pages for document-level benchmarks (default 5).
    pub num_pages: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            measurement_iterations: 10,
            image_width: 612,
            image_height: 792,
            regions_per_page: 20,
            num_pages: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Timing and throughput result for a single benchmark stage.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Human-readable stage name (e.g., "postprocess", "export_json").
    pub stage_name: String,
    /// Duration of the measured run in milliseconds.
    pub duration_ms: f64,
    /// Number of items processed in this measurement.
    pub items_processed: usize,
    /// Throughput in items per second.
    pub throughput: f64,
}

/// Aggregated benchmark summary across multiple stages.
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    /// Per-stage results.
    pub results: Vec<BenchmarkResult>,
    /// Total duration across all stages in milliseconds.
    pub total_duration_ms: f64,
}

impl BenchmarkSummary {
    /// Build a summary from a list of benchmark results.
    #[must_use]
    pub fn from_results(results: Vec<BenchmarkResult>) -> Self {
        let total_duration_ms = results.iter().map(|r| r.duration_ms).sum();
        Self {
            results,
            total_duration_ms,
        }
    }

    /// Generate a human-readable report with min/max/mean/p95 statistics.
    #[must_use]
    pub fn generate_report(&self) -> String {
        let mut lines = Vec::with_capacity(self.results.len() + 8);
        lines.push("=== dpdf Pipeline Benchmark Report ===".to_string());
        lines.push(String::new());

        if self.results.is_empty() {
            lines.push("No benchmark results.".to_string());
            return lines.join("\n");
        }

        // Per-stage summary.
        lines.push(format!(
            "{:<25} {:>12} {:>12} {:>14}",
            "Stage", "Duration(ms)", "Items", "Throughput(/s)"
        ));
        lines.push(format!("{:-<65}", ""));

        for r in &self.results {
            lines.push(format!(
                "{:<25} {:>12.3} {:>12} {:>14.1}",
                r.stage_name, r.duration_ms, r.items_processed, r.throughput,
            ));
        }

        lines.push(format!("{:-<65}", ""));

        // Aggregate statistics across all durations.
        let durations: Vec<f64> = self.results.iter().map(|r| r.duration_ms).collect();
        let stats = compute_stats(&durations);
        lines.push(format!("Min:    {:.3} ms", stats.min));
        lines.push(format!("Max:    {:.3} ms", stats.max));
        lines.push(format!("Mean:   {:.3} ms", stats.mean));
        lines.push(format!("P95:    {:.3} ms", stats.p95));
        lines.push(format!("Total:  {:.3} ms", self.total_duration_ms));

        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Synthetic data generators
// ---------------------------------------------------------------------------

/// Generate synthetic [`DocumentRegion`] values for benchmarking.
///
/// Distributes regions across the image area with deterministic positions
/// based on the index. Region types cycle through all 10 classes.
#[must_use]
pub fn generate_random_regions(
    count: usize,
    image_w: usize,
    image_h: usize,
) -> Vec<DocumentRegion> {
    let w = image_w as f32;
    let h = image_h as f32;
    let mut regions = Vec::with_capacity(count);

    for i in 0..count {
        let class_id = i % 10;
        // Deterministic pseudo-random placement based on index.
        let row = i / 3;
        let col = i % 3;
        let x1 = (col as f32) * (w / 3.0) + 5.0;
        let y1 = (row as f32) * 40.0 + 10.0;
        let x2 = (x1 + w / 4.0).min(w - 1.0);
        let y2 = (y1 + 30.0).min(h - 1.0);
        let bbox = [x1, y1, x2, y2];
        let confidence = 0.5 + (i as f32 % 5.0) * 0.1;

        let region = match class_id {
            0 => DocumentRegion::Caption {
                content: format!("Caption {i}"),
                bbox,
                confidence,
            },
            1 => DocumentRegion::Footnote {
                content: format!("Footnote {i}"),
                bbox,
                confidence,
            },
            2 => DocumentRegion::Formula {
                latex: Some(format!("x^{i}")),
                bbox,
                confidence,
            },
            3 => DocumentRegion::ListItem {
                content: format!("Item {i}"),
                bbox,
                confidence,
            },
            4 => DocumentRegion::PageFooter {
                content: format!("Footer {i}"),
                bbox,
                confidence,
            },
            5 => DocumentRegion::PageHeader {
                content: format!("Header {i}"),
                bbox,
                confidence,
            },
            6 => DocumentRegion::Figure {
                caption: Some(format!("Fig {i}")),
                bbox,
                confidence,
            },
            7 => DocumentRegion::SectionHeader {
                content: format!("Section {i}"),
                bbox,
                confidence,
            },
            8 => DocumentRegion::Table {
                cells: vec![
                    vec!["A".to_string(), "B".to_string()],
                    vec!["1".to_string(), "2".to_string()],
                ],
                bbox,
                confidence,
            },
            _ => DocumentRegion::Text {
                content: format!("Text block {i} with some content for benchmarking."),
                bbox,
                confidence,
            },
        };
        regions.push(region);
    }

    regions
}

/// Generate a synthetic [`PageOutput`] with the given number of regions.
#[must_use]
pub fn generate_random_page_output(num_regions: usize) -> PageOutput {
    let regions = generate_random_regions(num_regions, 612, 792);
    let reading_order = DpdfPipeline::compute_reading_order(&regions);
    PageOutput {
        regions,
        reading_order,
        width: 612,
        height: 792,
    }
}

/// Generate a synthetic [`DocumentOutput`] with the given page and region counts.
#[must_use]
pub fn generate_random_document(num_pages: usize, regions_per_page: usize) -> DocumentOutput {
    let pages = (0..num_pages)
        .map(|_| generate_random_page_output(regions_per_page))
        .collect();
    DocumentOutput { pages }
}

// ---------------------------------------------------------------------------
// Per-stage benchmark functions
// ---------------------------------------------------------------------------

/// Benchmark the postprocess stage (confidence filter, merge, dedup).
///
/// Runs `warmup_iterations` unmeasured, then `measurement_iterations` measured,
/// and returns the mean timing.
#[must_use]
pub fn bench_postprocess(config: &BenchmarkConfig) -> BenchmarkResult {
    let post_config = PostProcessConfig::default();
    let region_count = config.regions_per_page;

    // Warmup.
    for _ in 0..config.warmup_iterations {
        let mut regions =
            generate_random_regions(region_count, config.image_width, config.image_height);
        postprocess(&mut regions, &post_config);
    }

    // Measurement.
    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let mut regions =
            generate_random_regions(region_count, config.image_width, config.image_height);
        postprocess(&mut regions, &post_config);
    }
    let elapsed = start.elapsed();
    let duration_ms = elapsed.as_secs_f64() * 1000.0;
    let items = config.measurement_iterations * region_count;
    let throughput = if duration_ms > 0.0 {
        items as f64 / (duration_ms / 1000.0)
    } else {
        0.0
    };

    BenchmarkResult {
        stage_name: "postprocess".to_string(),
        duration_ms,
        items_processed: items,
        throughput,
    }
}

/// Benchmark JSON export of a synthetic document.
///
/// # Errors
///
/// Returns [`ExportError`] if JSON serialization fails.
pub fn bench_export_json(config: &BenchmarkConfig) -> Result<BenchmarkResult, ExportError> {
    let doc = generate_random_document(config.num_pages, config.regions_per_page);
    let exporter = JsonExporter::new();

    // Warmup.
    for _ in 0..config.warmup_iterations {
        let _ = exporter.export(&doc)?;
    }

    // Measurement.
    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let _ = exporter.export(&doc)?;
    }
    let elapsed = start.elapsed();
    let duration_ms = elapsed.as_secs_f64() * 1000.0;
    let items = config.measurement_iterations * config.num_pages;
    let throughput = if duration_ms > 0.0 {
        items as f64 / (duration_ms / 1000.0)
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        stage_name: "export_json".to_string(),
        duration_ms,
        items_processed: items,
        throughput,
    })
}

/// Benchmark HTML export of a synthetic document.
///
/// # Errors
///
/// Returns [`ExportError`] if HTML export fails.
pub fn bench_export_html(config: &BenchmarkConfig) -> Result<BenchmarkResult, ExportError> {
    let doc = generate_random_document(config.num_pages, config.regions_per_page);
    let exporter = HtmlExporter::new();

    for _ in 0..config.warmup_iterations {
        let _ = exporter.export(&doc)?;
    }

    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let _ = exporter.export(&doc)?;
    }
    let elapsed = start.elapsed();
    let duration_ms = elapsed.as_secs_f64() * 1000.0;
    let items = config.measurement_iterations * config.num_pages;
    let throughput = if duration_ms > 0.0 {
        items as f64 / (duration_ms / 1000.0)
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        stage_name: "export_html".to_string(),
        duration_ms,
        items_processed: items,
        throughput,
    })
}

/// Benchmark Markdown export of a synthetic document.
///
/// # Errors
///
/// Returns [`ExportError`] if Markdown export fails.
pub fn bench_export_markdown(config: &BenchmarkConfig) -> Result<BenchmarkResult, ExportError> {
    let doc = generate_random_document(config.num_pages, config.regions_per_page);
    let exporter = MarkdownExporter::new();

    for _ in 0..config.warmup_iterations {
        let _ = exporter.export(&doc)?;
    }

    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let _ = exporter.export(&doc)?;
    }
    let elapsed = start.elapsed();
    let duration_ms = elapsed.as_secs_f64() * 1000.0;
    let items = config.measurement_iterations * config.num_pages;
    let throughput = if duration_ms > 0.0 {
        items as f64 / (duration_ms / 1000.0)
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        stage_name: "export_markdown".to_string(),
        duration_ms,
        items_processed: items,
        throughput,
    })
}

/// Benchmark table structure parsing on synthetic detection data.
#[must_use]
pub fn bench_table_structure(config: &BenchmarkConfig) -> BenchmarkResult {
    let ts_config = TableStructureConfig::default();
    let detections = generate_table_detections(4, 3);

    // Warmup.
    for _ in 0..config.warmup_iterations {
        let _ = table_structure::parse_structure(&detections, &ts_config);
    }

    // Measurement.
    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let _ = table_structure::parse_structure(&detections, &ts_config);
    }
    let elapsed = start.elapsed();
    let duration_ms = elapsed.as_secs_f64() * 1000.0;
    let items = config.measurement_iterations;
    let throughput = if duration_ms > 0.0 {
        items as f64 / (duration_ms / 1000.0)
    } else {
        0.0
    };

    BenchmarkResult {
        stage_name: "table_structure".to_string(),
        duration_ms,
        items_processed: items,
        throughput,
    }
}

/// Run all per-stage benchmarks and return an aggregated summary.
///
/// # Errors
///
/// Returns [`ExportError`] if any export benchmark fails.
pub fn run_all_benchmarks(config: &BenchmarkConfig) -> Result<BenchmarkSummary, ExportError> {
    let results = vec![
        bench_postprocess(config),
        bench_export_json(config)?,
        bench_export_html(config)?,
        bench_export_markdown(config)?,
        bench_table_structure(config),
    ];
    Ok(BenchmarkSummary::from_results(results))
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

struct Stats {
    min: f64,
    max: f64,
    mean: f64,
    p95: f64,
}

fn compute_stats(values: &[f64]) -> Stats {
    if values.is_empty() {
        return Stats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            p95: 0.0,
        };
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;

    // P95: index at 95th percentile (ceiling).
    let p95_idx = ((sorted.len() as f64) * 0.95).ceil() as usize;
    let p95 = sorted[p95_idx.min(sorted.len() - 1)];

    Stats {
        min,
        max,
        mean,
        p95,
    }
}

// ---------------------------------------------------------------------------
// Synthetic table detection generator
// ---------------------------------------------------------------------------

/// Generate synthetic `Detection` objects for table structure benchmarking.
///
/// Creates `num_rows` row detections and `num_cols` column detections
/// evenly distributed across a unit-normalized bounding box.
fn generate_table_detections(
    num_rows: usize,
    num_cols: usize,
) -> Vec<nn_core::layers::vision::Detection> {
    let mut dets = Vec::with_capacity(num_rows + num_cols + 1);

    // Table bounding box (class 0).
    dets.push(nn_core::layers::vision::Detection {
        class_id: 0,
        confidence: 0.99,
        x1: 0.0,
        y1: 0.0,
        x2: 1.0,
        y2: 1.0,
    });

    let row_height = 1.0 / num_rows as f32;
    for r in 0..num_rows {
        dets.push(nn_core::layers::vision::Detection {
            class_id: 1, // row
            confidence: 0.95,
            x1: 0.0,
            y1: r as f32 * row_height,
            x2: 1.0,
            y2: (r + 1) as f32 * row_height,
        });
    }

    let col_width = 1.0 / num_cols as f32;
    for c in 0..num_cols {
        dets.push(nn_core::layers::vision::Detection {
            class_id: 2, // column
            confidence: 0.95,
            x1: c as f32 * col_width,
            y1: 0.0,
            x2: (c + 1) as f32 * col_width,
            y2: 1.0,
        });
    }

    dets
}

#[cfg(test)]
#[path = "dpdf_benchmark_tests.rs"]
mod tests;
