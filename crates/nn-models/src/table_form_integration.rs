// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration pipeline between UniTable (table extraction) and LayoutLMv3
//! (form entity labeling) for combined document understanding.
//!
//! This module orchestrates both models to produce a unified document
//! extraction result that includes:
//!
//! - Structured tables with cell-level spans and content.
//! - Form key-value fields with spatial associations.
//! - Region classification (table vs form vs mixed).
//!
//! # Pipeline
//!
//! 1. Run document layout detection to identify table and form regions.
//! 2. For table regions, run UniTable + cell post-processing + span recognition.
//! 3. For form regions, run LayoutLMv3 + BIO decoding + key-value pairing.
//! 4. Merge results, resolving overlapping table/form regions.

use crate::form_field_association::{FormAssociationConfig, FormExtractionResult};
use crate::table_span_recognition::SpanRecognitionConfig;
use crate::table_structure::StructuredTable;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the integrated table + form extraction pipeline.
#[derive(Debug, Clone)]
pub struct TableFormConfig {
    /// Configuration for table span recognition.
    pub span_config: SpanRecognitionConfig,
    /// Configuration for form field association.
    pub form_config: FormAssociationConfig,
    /// IoU threshold for merging overlapping table/form regions (default 0.3).
    pub region_merge_iou: f32,
    /// Minimum confidence for table regions (default 0.4).
    pub table_confidence_threshold: f32,
    /// Minimum confidence for form regions (default 0.4).
    pub form_confidence_threshold: f32,
}

impl Default for TableFormConfig {
    fn default() -> Self {
        Self {
            span_config: SpanRecognitionConfig::default(),
            form_config: FormAssociationConfig::default(),
            region_merge_iou: 0.3,
            table_confidence_threshold: 0.4,
            form_confidence_threshold: 0.4,
        }
    }
}

// ---------------------------------------------------------------------------
// Region classification
// ---------------------------------------------------------------------------

/// Classification of a document region for extraction routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// Pure table region (use UniTable).
    Table,
    /// Pure form region (use LayoutLMv3).
    Form,
    /// Mixed region containing both tabular and form elements.
    Mixed,
    /// Other region type (not processed by this pipeline).
    Other,
}

/// A detected document region with its classification.
#[derive(Debug, Clone)]
pub struct ClassifiedRegion {
    /// Bounding box `[x1, y1, x2, y2]` in pixel coordinates.
    pub bbox: [f32; 4],
    /// Detection confidence.
    pub confidence: f32,
    /// Region classification.
    pub kind: RegionKind,
    /// Source model identifier (for provenance tracking).
    pub source: &'static str,
}

// ---------------------------------------------------------------------------
// Extraction results
// ---------------------------------------------------------------------------

/// Extraction result for a single table region.
#[derive(Debug, Clone)]
pub struct TableExtractionResult {
    /// The structured table with spans.
    pub table: StructuredTable,
    /// Bounding box of the table region.
    pub bbox: [f32; 4],
    /// Confidence score.
    pub confidence: f32,
}

/// Combined extraction result for a document page.
#[derive(Debug, Clone)]
pub struct PageExtractionResult {
    /// Extracted tables.
    pub tables: Vec<TableExtractionResult>,
    /// Extracted form fields.
    pub form: FormExtractionResult,
    /// Regions that could not be classified as table or form.
    pub unclassified_regions: Vec<ClassifiedRegion>,
}

// ---------------------------------------------------------------------------
// Region classification
// ---------------------------------------------------------------------------

/// Classify layout detections into table, form, mixed, or other regions.
///
/// Uses class names from the layout detector. "table" maps to `Table`,
/// text-like regions near spatial form patterns map to `Form`, and
/// regions that overlap both get `Mixed`.
#[must_use]
pub fn classify_regions(
    detections: &[(String, [f32; 4], f32)],
    config: &TableFormConfig,
) -> Vec<ClassifiedRegion> {
    let mut regions = Vec::with_capacity(detections.len());

    for (class_name, bbox, confidence) in detections {
        let kind = match class_name.as_str() {
            "table" => {
                if *confidence >= config.table_confidence_threshold {
                    RegionKind::Table
                } else {
                    RegionKind::Other
                }
            }
            "text" | "list-item" => {
                if *confidence >= config.form_confidence_threshold {
                    RegionKind::Form
                } else {
                    RegionKind::Other
                }
            }
            _ => RegionKind::Other,
        };

        regions.push(ClassifiedRegion {
            bbox: *bbox,
            confidence: *confidence,
            kind,
            source: "layout_detector",
        });
    }

    // Second pass: detect mixed regions (table + form overlap).
    let table_regions: Vec<[f32; 4]> = regions
        .iter()
        .filter(|r| r.kind == RegionKind::Table)
        .map(|r| r.bbox)
        .collect();

    for region in &mut regions {
        if region.kind == RegionKind::Form {
            for table_bbox in &table_regions {
                if compute_iou(&region.bbox, table_bbox) > config.region_merge_iou {
                    region.kind = RegionKind::Mixed;
                    break;
                }
            }
        }
    }

    regions
}

// ---------------------------------------------------------------------------
// Result merging
// ---------------------------------------------------------------------------

/// Merge table and form extraction results into a unified page result.
///
/// Handles overlapping regions by preferring the extraction result with
/// higher confidence. Mixed regions get both table and form output.
#[must_use]
pub fn merge_results(
    tables: Vec<TableExtractionResult>,
    form: FormExtractionResult,
    classified: &[ClassifiedRegion],
) -> PageExtractionResult {
    let unclassified: Vec<ClassifiedRegion> = classified
        .iter()
        .filter(|r| r.kind == RegionKind::Other)
        .cloned()
        .collect();

    PageExtractionResult {
        tables,
        form,
        unclassified_regions: unclassified,
    }
}

/// Create an empty form extraction result.
#[must_use]
pub fn empty_form_result() -> FormExtractionResult {
    FormExtractionResult {
        fields: Vec::new(),
        headers: Vec::new(),
        orphan_values: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Summary statistics
// ---------------------------------------------------------------------------

/// Summary statistics for a page extraction result.
#[derive(Debug, Clone)]
pub struct ExtractionSummary {
    /// Number of tables extracted.
    pub num_tables: usize,
    /// Total number of table cells across all tables.
    pub total_cells: usize,
    /// Number of spanning cells across all tables.
    pub total_spanning_cells: usize,
    /// Number of form key-value pairs.
    pub num_form_fields: usize,
    /// Number of paired fields (with both key and value).
    pub num_paired_fields: usize,
    /// Number of form headers.
    pub num_headers: usize,
    /// Number of orphan values (not paired to any key).
    pub num_orphan_values: usize,
}

/// Compute summary statistics for a page extraction result.
#[must_use]
pub fn summarize(result: &PageExtractionResult) -> ExtractionSummary {
    let total_cells: usize = result
        .tables
        .iter()
        .map(|t| t.table.rows.iter().map(|r| r.cells.len()).sum::<usize>())
        .sum();

    let total_spanning_cells: usize = result
        .tables
        .iter()
        .map(|t| crate::table_span_recognition::count_spanning_cells(&t.table))
        .sum();

    let num_paired = result
        .form
        .fields
        .iter()
        .filter(|f| f.value.is_some())
        .count();

    ExtractionSummary {
        num_tables: result.tables.len(),
        total_cells,
        total_spanning_cells,
        num_form_fields: result.form.fields.len(),
        num_paired_fields: num_paired,
        num_headers: result.form.headers.len(),
        num_orphan_values: result.form.orphan_values.len(),
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute IoU between two `[x1, y1, x2, y2]` bounding boxes.
fn compute_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter_x1 = a[0].max(b[0]);
    let inter_y1 = a[1].max(b[1]);
    let inter_x2 = a[2].min(b[2]);
    let inter_y2 = a[3].min(b[3]);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        return 0.0;
    }
    inter_area / union_area
}

#[cfg(test)]
#[path = "table_form_integration_tests.rs"]
mod tests;
