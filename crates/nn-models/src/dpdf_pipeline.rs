// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end dpdf document inference pipeline.
//!
//! Orchestrates [`DocLayoutYolo`] (layout detection) and [`GraniteDocling`]
//! (OCR / document understanding) into a single pipeline that takes document
//! page images and produces structured [`DocumentOutput`] with classified
//! regions, reading order, and text/markdown export.
//!
//! # Classes (from DocLayout-YOLO)
//!
//! ```text
//! 0: caption     1: footnote      2: formula      3: list-item    4: page-footer
//! 5: page-header 6: picture       7: section-header 8: table      9: text
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use nn_models::dpdf_pipeline::{DpdfPipeline, PipelineConfig};
//!
//! let pipeline = DpdfPipeline::new(PipelineConfig::default());
//! let detections = vec![(9, 0.95, [10.0, 20.0, 300.0, 80.0])];
//! let regions = DpdfPipeline::detections_to_regions(&detections);
//! let page = pipeline.build_page(regions, 612, 792);
//! let md = DpdfPipeline::to_markdown(&page);
//! ```

use crate::doclayout_yolo::CLASS_NAMES;
use crate::dpdf_postprocess::{postprocess, PostProcessConfig};
use crate::table_structure::{self, TableStructureConfig};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Pipeline configuration controlling detection thresholds and OCR limits.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Minimum confidence for layout detections (default 0.25).
    pub layout_conf_threshold: f32,
    /// IoU threshold for NMS suppression (default 0.45).
    pub layout_iou_threshold: f32,
    /// Maximum number of tokens for OCR model (default 1024).
    pub ocr_max_tokens: usize,
    /// Whether to run table structure recognition (default true).
    pub enable_table_structure: bool,
    /// Post-processing configuration (merge, dedup, confidence filter).
    pub postprocess_config: PostProcessConfig,
    /// Table structure recognition configuration.
    pub table_structure_config: TableStructureConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            layout_conf_threshold: 0.25,
            layout_iou_threshold: 0.45,
            ocr_max_tokens: 1024,
            enable_table_structure: true,
            postprocess_config: PostProcessConfig::default(),
            table_structure_config: TableStructureConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Document region types
// ---------------------------------------------------------------------------

/// A classified region of a document page with bounding box and content.
///
/// Each variant maps to a DocLayout-YOLO class (see [`CLASS_NAMES`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DocumentRegion {
    /// Free-form text block (class 9: text).
    Text {
        content: String,
        bbox: [f32; 4],
        confidence: f32,
    },
    /// Table region with optional cell grid (class 8: table).
    Table {
        cells: Vec<Vec<String>>,
        bbox: [f32; 4],
        confidence: f32,
    },
    /// Figure/picture region with optional caption (class 6: picture).
    Figure {
        caption: Option<String>,
        bbox: [f32; 4],
        confidence: f32,
    },
    /// Mathematical formula with optional LaTeX (class 2: formula).
    Formula {
        latex: Option<String>,
        bbox: [f32; 4],
        confidence: f32,
    },
    /// Section heading (class 7: section-header).
    SectionHeader {
        content: String,
        bbox: [f32; 4],
        confidence: f32,
    },
    /// Page header (class 5: page-header).
    PageHeader {
        content: String,
        bbox: [f32; 4],
        confidence: f32,
    },
    /// Page footer (class 4: page-footer).
    PageFooter {
        content: String,
        bbox: [f32; 4],
        confidence: f32,
    },
    /// Caption for a figure or table (class 0: caption).
    Caption {
        content: String,
        bbox: [f32; 4],
        confidence: f32,
    },
    /// List item entry (class 3: list-item).
    ListItem {
        content: String,
        bbox: [f32; 4],
        confidence: f32,
    },
    /// Footnote text (class 1: footnote).
    Footnote {
        content: String,
        bbox: [f32; 4],
        confidence: f32,
    },
}

impl DocumentRegion {
    /// Return the bounding box `[x1, y1, x2, y2]` regardless of variant.
    #[must_use]
    pub fn bbox(&self) -> [f32; 4] {
        match self {
            Self::Text { bbox, .. }
            | Self::Table { bbox, .. }
            | Self::Figure { bbox, .. }
            | Self::Formula { bbox, .. }
            | Self::SectionHeader { bbox, .. }
            | Self::PageHeader { bbox, .. }
            | Self::PageFooter { bbox, .. }
            | Self::Caption { bbox, .. }
            | Self::ListItem { bbox, .. }
            | Self::Footnote { bbox, .. } => *bbox,
        }
    }

    /// Return the confidence score regardless of variant.
    #[must_use]
    pub fn confidence(&self) -> f32 {
        match self {
            Self::Text { confidence, .. }
            | Self::Table { confidence, .. }
            | Self::Figure { confidence, .. }
            | Self::Formula { confidence, .. }
            | Self::SectionHeader { confidence, .. }
            | Self::PageHeader { confidence, .. }
            | Self::PageFooter { confidence, .. }
            | Self::Caption { confidence, .. }
            | Self::ListItem { confidence, .. }
            | Self::Footnote { confidence, .. } => *confidence,
        }
    }

    /// Return the class name string for this region type.
    #[must_use]
    pub fn class_name(&self) -> &'static str {
        match self {
            Self::Caption { .. } => "caption",
            Self::Footnote { .. } => "footnote",
            Self::Formula { .. } => "formula",
            Self::ListItem { .. } => "list-item",
            Self::PageFooter { .. } => "page-footer",
            Self::PageHeader { .. } => "page-header",
            Self::Figure { .. } => "picture",
            Self::SectionHeader { .. } => "section-header",
            Self::Table { .. } => "table",
            Self::Text { .. } => "text",
        }
    }
}

// ---------------------------------------------------------------------------
// Page and document output
// ---------------------------------------------------------------------------

/// Structured output for a single document page.
#[derive(Debug, Clone)]
pub struct PageOutput {
    /// Classified regions on this page.
    pub regions: Vec<DocumentRegion>,
    /// Indices into `regions` in reading order (top-to-bottom, left-to-right).
    pub reading_order: Vec<usize>,
    /// Page width in pixels.
    pub width: usize,
    /// Page height in pixels.
    pub height: usize,
}

/// Structured output for an entire document.
#[derive(Debug, Clone)]
pub struct DocumentOutput {
    /// Per-page structured outputs.
    pub pages: Vec<PageOutput>,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Orchestrator for the dpdf end-to-end document inference pipeline.
///
/// Converts raw detection tuples `(class_id, confidence, bbox)` into
/// structured [`DocumentRegion`] objects, computes reading order, and
/// provides text and markdown export.
#[derive(Debug, Clone)]
pub struct DpdfPipeline {
    config: PipelineConfig,
}

impl DpdfPipeline {
    /// Create a new pipeline with the given configuration.
    #[must_use]
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Access the pipeline configuration.
    #[must_use]
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Build a [`PageOutput`] from regions, computing reading order.
    ///
    /// Applies post-processing (confidence filtering, merge, dedup) before
    /// computing reading order. When `enable_table_structure` is true,
    /// table regions are enriched with parsed cell grids from any available
    /// structure detections.
    #[must_use]
    pub fn build_page(
        &self,
        regions: Vec<DocumentRegion>,
        width: usize,
        height: usize,
    ) -> PageOutput {
        self.build_page_with_structure(regions, &[], width, height)
    }

    /// Build a [`PageOutput`] with optional table-structure detections.
    ///
    /// `table_dets` are raw [`Detection`] outputs from a Table Transformer
    /// structure-recognition pass. When non-empty and
    /// `enable_table_structure` is true, each `Table` region's cells are
    /// populated from the parsed structure.
    #[must_use]
    pub fn build_page_with_structure(
        &self,
        mut regions: Vec<DocumentRegion>,
        table_dets: &[nn_core::layers::vision::Detection],
        width: usize,
        height: usize,
    ) -> PageOutput {
        // 1. Post-processing: confidence filter, merge, dedup.
        postprocess(&mut regions, &self.config.postprocess_config);

        // 2. Table structure integration.
        if self.config.enable_table_structure && !table_dets.is_empty() {
            let structured =
                table_structure::parse_structure(table_dets, &self.config.table_structure_config);
            // Enrich each Table region with the parsed cell grid.
            enrich_table_regions(&mut regions, &structured);
        }

        // 3. Reading order.
        let reading_order = Self::compute_reading_order(&regions);
        PageOutput {
            regions,
            reading_order,
            width,
            height,
        }
    }

    /// Compute reading order for a set of regions.
    ///
    /// Sorts by vertical midpoint (top-to-bottom) with a horizontal
    /// tie-breaker (left-to-right). Page headers are placed first and
    /// page footers last regardless of position.
    #[must_use]
    pub fn compute_reading_order(regions: &[DocumentRegion]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..regions.len()).collect();
        indices.sort_by(|&a, &b| {
            let ra = &regions[a];
            let rb = &regions[b];

            // Page headers come first, page footers come last.
            let priority_a = region_sort_priority(ra);
            let priority_b = region_sort_priority(rb);
            priority_a
                .cmp(&priority_b)
                .then_with(|| {
                    let mid_y_a = f32::midpoint(ra.bbox()[1], ra.bbox()[3]);
                    let mid_y_b = f32::midpoint(rb.bbox()[1], rb.bbox()[3]);
                    mid_y_a
                        .partial_cmp(&mid_y_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    let mid_x_a = f32::midpoint(ra.bbox()[0], ra.bbox()[2]);
                    let mid_x_b = f32::midpoint(rb.bbox()[0], rb.bbox()[2]);
                    mid_x_a
                        .partial_cmp(&mid_x_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        indices
    }

    /// Classify a raw detection into the appropriate [`DocumentRegion`].
    ///
    /// `class_id` must be in `0..10` (see [`CLASS_NAMES`]). Out-of-range
    /// IDs default to `Text`.
    #[must_use]
    pub fn classify_detection(class_id: usize, bbox: [f32; 4], confidence: f32) -> DocumentRegion {
        match class_id {
            0 => DocumentRegion::Caption {
                content: String::new(),
                bbox,
                confidence,
            },
            1 => DocumentRegion::Footnote {
                content: String::new(),
                bbox,
                confidence,
            },
            2 => DocumentRegion::Formula {
                latex: None,
                bbox,
                confidence,
            },
            3 => DocumentRegion::ListItem {
                content: String::new(),
                bbox,
                confidence,
            },
            4 => DocumentRegion::PageFooter {
                content: String::new(),
                bbox,
                confidence,
            },
            5 => DocumentRegion::PageHeader {
                content: String::new(),
                bbox,
                confidence,
            },
            6 => DocumentRegion::Figure {
                caption: None,
                bbox,
                confidence,
            },
            7 => DocumentRegion::SectionHeader {
                content: String::new(),
                bbox,
                confidence,
            },
            8 => DocumentRegion::Table {
                cells: Vec::new(),
                bbox,
                confidence,
            },
            9 => DocumentRegion::Text {
                content: String::new(),
                bbox,
                confidence,
            },
            _ => DocumentRegion::Text {
                content: String::new(),
                bbox,
                confidence,
            },
        }
    }

    /// Convert raw detection tuples to classified [`DocumentRegion`] objects.
    ///
    /// Each tuple is `(class_id, confidence, [x1, y1, x2, y2])`.
    #[must_use]
    pub fn detections_to_regions(detections: &[(usize, f32, [f32; 4])]) -> Vec<DocumentRegion> {
        detections
            .iter()
            .map(|&(class_id, confidence, bbox)| {
                Self::classify_detection(class_id, bbox, confidence)
            })
            .collect()
    }

    /// Extract plain text from a page in reading order.
    ///
    /// Regions without textual content (e.g., figures) are represented
    /// by their class name in brackets.
    #[must_use]
    pub fn extract_text(page: &PageOutput) -> String {
        let mut parts = Vec::with_capacity(page.reading_order.len());
        for &idx in &page.reading_order {
            let region = &page.regions[idx];
            let text = region_text_content(region);
            if !text.is_empty() {
                parts.push(text);
            }
        }
        parts.join("\n")
    }

    /// Convert a page to Markdown format in reading order.
    ///
    /// - Section headers become `## heading`
    /// - List items become `- item`
    /// - Tables become pipe-delimited Markdown tables
    /// - Formulas become `$latex$` when LaTeX is available
    /// - Figures become `![Figure](caption)` placeholders
    #[must_use]
    pub fn to_markdown(page: &PageOutput) -> String {
        let mut lines = Vec::with_capacity(page.reading_order.len());
        for &idx in &page.reading_order {
            let region = &page.regions[idx];
            let md = region_to_markdown(region);
            if !md.is_empty() {
                lines.push(md);
            }
        }
        lines.join("\n\n")
    }

    /// Build a complete [`DocumentOutput`] from per-page detection lists.
    ///
    /// Each entry in `pages_detections` is a tuple of
    /// `(detections, page_width, page_height)`.
    ///
    /// Post-processing (confidence filtering, merge, dedup) is applied
    /// automatically to each page via [`build_page`].
    #[must_use]
    pub fn process_pages(
        &self,
        pages_detections: &[(&[(usize, f32, [f32; 4])], usize, usize)],
    ) -> DocumentOutput {
        let pages = pages_detections
            .iter()
            .map(|(dets, w, h)| {
                let regions = Self::detections_to_regions(dets);
                self.build_page(regions, *w, *h)
            })
            .collect();
        DocumentOutput { pages }
    }
}

// ---------------------------------------------------------------------------
// Helpers (private)
// ---------------------------------------------------------------------------

/// Sort priority bucket: headers first (0), normal content (1), footers last (2).
fn region_sort_priority(region: &DocumentRegion) -> u8 {
    match region {
        DocumentRegion::PageHeader { .. } => 0,
        DocumentRegion::PageFooter { .. } => 2,
        _ => 1,
    }
}

/// Extract the textual content from a region, or a bracketed placeholder.
fn region_text_content(region: &DocumentRegion) -> String {
    match region {
        DocumentRegion::Text { content, .. }
        | DocumentRegion::SectionHeader { content, .. }
        | DocumentRegion::PageHeader { content, .. }
        | DocumentRegion::PageFooter { content, .. }
        | DocumentRegion::Caption { content, .. }
        | DocumentRegion::ListItem { content, .. }
        | DocumentRegion::Footnote { content, .. } => {
            if content.is_empty() {
                format!("[{}]", region.class_name())
            } else {
                content.clone()
            }
        }
        DocumentRegion::Formula { latex, .. } => {
            latex.clone().unwrap_or_else(|| "[formula]".to_string())
        }
        DocumentRegion::Table { cells, .. } => {
            if cells.is_empty() {
                "[table]".to_string()
            } else {
                cells
                    .iter()
                    .map(|row| row.join("\t"))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        DocumentRegion::Figure { caption, .. } => {
            caption.clone().unwrap_or_else(|| "[picture]".to_string())
        }
    }
}

/// Convert a region to its Markdown representation.
fn region_to_markdown(region: &DocumentRegion) -> String {
    match region {
        DocumentRegion::SectionHeader { content, .. } => {
            if content.is_empty() {
                String::new()
            } else {
                format!("## {content}")
            }
        }
        DocumentRegion::ListItem { content, .. } => {
            if content.is_empty() {
                String::new()
            } else {
                format!("- {content}")
            }
        }
        DocumentRegion::Formula { latex, .. } => match latex {
            Some(l) => format!("${l}$"),
            None => "[formula]".to_string(),
        },
        DocumentRegion::Figure { caption, .. } => {
            let cap = caption.as_deref().unwrap_or("Figure");
            format!("![{cap}]()")
        }
        DocumentRegion::Table { cells, .. } => {
            if cells.is_empty() {
                "[table]".to_string()
            } else {
                table_to_markdown(cells)
            }
        }
        DocumentRegion::Text { content, .. }
        | DocumentRegion::PageHeader { content, .. }
        | DocumentRegion::PageFooter { content, .. }
        | DocumentRegion::Caption { content, .. }
        | DocumentRegion::Footnote { content, .. } => {
            if content.is_empty() {
                format!("[{}]", region.class_name())
            } else {
                content.clone()
            }
        }
    }
}

/// Render a cell grid as a Markdown pipe table.
fn table_to_markdown(cells: &[Vec<String>]) -> String {
    if cells.is_empty() {
        return String::new();
    }
    let mut lines = Vec::with_capacity(cells.len() + 1);
    // Header row
    let header = &cells[0];
    lines.push(format!("| {} |", header.join(" | ")));
    // Separator
    let sep: Vec<&str> = (0..header.len()).map(|_| "---").collect();
    lines.push(format!("| {} |", sep.join(" | ")));
    // Data rows
    for row in cells.iter().skip(1) {
        lines.push(format!("| {} |", row.join(" | ")));
    }
    lines.join("\n")
}

/// Enrich `Table` regions with cell data from a parsed [`StructuredTable`].
///
/// For each `Table` region, converts the structured table rows into the
/// `Vec<Vec<String>>` cell grid format used by [`DocumentRegion::Table`].
/// Cell labels use `(row,col)` notation. If the structured table has no
/// rows, the region is left unchanged.
fn enrich_table_regions(
    regions: &mut [DocumentRegion],
    structured: &table_structure::StructuredTable,
) {
    if structured.num_rows == 0 || structured.num_cols == 0 {
        return;
    }
    // Build a grid of cell labels from the structured table.
    let mut grid = vec![vec![String::new(); structured.num_cols]; structured.num_rows];
    for row in &structured.rows {
        for cell in &row.cells {
            if cell.row < structured.num_rows && cell.col < structured.num_cols {
                grid[cell.row][cell.col] = format!("({},{})", cell.row, cell.col);
            }
        }
    }

    for region in regions.iter_mut() {
        if let DocumentRegion::Table {
            ref mut cells,
            bbox: _,
            confidence: _,
        } = region
        {
            if cells.is_empty() {
                *cells = grid.clone();
            }
        }
    }
}

// Ensure CLASS_NAMES compatibility at compile time.
const _: () = {
    // DocLayout-YOLO has exactly 10 classes; classify_detection depends on this.
    assert!(CLASS_NAMES.len() == 10);
};

#[cfg(test)]
#[path = "dpdf_pipeline_tests.rs"]
mod tests;
