// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Table structure recognition: cell-level parsing from Table Transformer output.
//!
//! Given a set of [`Detection`] bounding boxes (rows, columns, spanning cells)
//! produced by the Table Transformer in structure-recognition mode, this module
//! reconstructs the logical table grid with cell positions, row/column spans,
//! and provides serialization to HTML, Markdown, and CSV.
//!
//! # Architecture
//!
//! 1. Filter detections by class (rows, columns, spanning cells).
//! 2. Sort rows/columns by spatial position.
//! 3. Assign each cell to a (row, col) grid position via IoU overlap.
//! 4. Detect spanning cells that cover multiple rows/columns.
//! 5. Emit [`StructuredTable`] with cell-level metadata.
//!
//! Reference: Smock et al. 2022, "PubTables-1M", CVPR 2022.

use nn_core::layers::vision::Detection;

// -- Table Transformer structure class indices (from STRUCTURE_CLASSES) -------
/// Class index for "table" in structure recognition output.
#[allow(dead_code)]
const CLASS_TABLE: u32 = 0;
/// Class index for "row".
const CLASS_ROW: u32 = 1;
/// Class index for "column".
const CLASS_COLUMN: u32 = 2;
/// Class index for "spanning-cell".
const CLASS_SPANNING_CELL: u32 = 3;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single cell within a structured table grid.
#[derive(Debug, Clone, PartialEq)]
pub struct TableCell {
    /// Zero-based row index.
    pub row: usize,
    /// Zero-based column index.
    pub col: usize,
    /// Number of rows this cell spans (>= 1).
    pub row_span: usize,
    /// Number of columns this cell spans (>= 1).
    pub col_span: usize,
    /// Bounding box `[x1, y1, x2, y2]` in normalized coordinates.
    pub bbox: [f32; 4],
    /// Detection confidence score.
    pub confidence: f32,
}

/// A single row of cells in a structured table.
#[derive(Debug, Clone, PartialEq)]
pub struct TableRow {
    /// Cells in this row, ordered by column index.
    pub cells: Vec<TableCell>,
}

/// A fully parsed table with grid structure and metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredTable {
    /// Rows of the table, ordered top-to-bottom.
    pub rows: Vec<TableRow>,
    /// Total number of logical rows.
    pub num_rows: usize,
    /// Total number of logical columns.
    pub num_cols: usize,
    /// Optional table caption (from class 0 detection if present).
    pub caption: Option<String>,
}

/// Configuration for table structure parsing.
#[derive(Debug, Clone)]
pub struct TableStructureConfig {
    /// IoU threshold for cell-to-row/column assignment (default 0.5).
    pub iou_threshold: f32,
    /// Y-overlap tolerance for row grouping (default 0.3).
    pub row_tolerance: f32,
    /// X-overlap tolerance for column grouping (default 0.3).
    pub col_tolerance: f32,
}

impl Default for TableStructureConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.5,
            row_tolerance: 0.3,
            col_tolerance: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Core parsing
// ---------------------------------------------------------------------------

/// Parse table structure from DETR detection output.
///
/// Accepts the filtered detections from Table Transformer structure recognition
/// (classes: table, row, column, spanning-cell, projected-row-header, no-object)
/// and reconstructs the logical cell grid.
///
/// # Algorithm
///
/// 1. Extract row and column detections, sort spatially.
/// 2. For each (row, column) intersection, create a cell with the overlapping bbox.
/// 3. Merge spanning-cell detections that cover multiple row/column intersections.
/// 4. Return the structured table with row/col span metadata.
#[must_use]
pub fn parse_structure(detections: &[Detection], config: &TableStructureConfig) -> StructuredTable {
    // Partition detections by class.
    let mut row_dets: Vec<&Detection> = Vec::new();
    let mut col_dets: Vec<&Detection> = Vec::new();
    let mut span_dets: Vec<&Detection> = Vec::new();

    for det in detections {
        match det.class_id {
            CLASS_ROW => row_dets.push(det),
            CLASS_COLUMN => col_dets.push(det),
            CLASS_SPANNING_CELL => span_dets.push(det),
            _ => {} // table bbox and no-object are not grid elements
        }
    }

    // Sort rows by vertical midpoint (top to bottom).
    row_dets.sort_by(|a, b| {
        let mid_a = f32::midpoint(a.y1, a.y2);
        let mid_b = f32::midpoint(b.y1, b.y2);
        mid_a
            .partial_cmp(&mid_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Sort columns by horizontal midpoint (left to right).
    col_dets.sort_by(|a, b| {
        let mid_a = f32::midpoint(a.x1, a.x2);
        let mid_b = f32::midpoint(b.x1, b.x2);
        mid_a
            .partial_cmp(&mid_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let num_rows = row_dets.len();
    let num_cols = col_dets.len();

    if num_rows == 0 || num_cols == 0 {
        return StructuredTable {
            rows: Vec::new(),
            num_rows: 0,
            num_cols: 0,
            caption: None,
        };
    }

    // Build a grid of cells from row/column intersections.
    // Track which cells are occupied by spanning cells.
    let mut occupied = vec![vec![false; num_cols]; num_rows];
    let mut table_rows: Vec<TableRow> = Vec::with_capacity(num_rows);

    // First pass: identify spanning cells and mark their grid coverage.
    let mut spanning_cells: Vec<TableCell> = Vec::new();
    for span_det in &span_dets {
        let span_bbox = [span_det.x1, span_det.y1, span_det.x2, span_det.y2];

        // Find which rows this spanning cell overlaps.
        let covered_rows =
            find_overlapping_indices(&row_dets, &span_bbox, config.row_tolerance, Axis::Row);
        // Find which columns this spanning cell overlaps.
        let covered_cols =
            find_overlapping_indices(&col_dets, &span_bbox, config.col_tolerance, Axis::Column);

        if covered_rows.is_empty() || covered_cols.is_empty() {
            continue;
        }

        let first_row = covered_rows[0];
        let first_col = covered_cols[0];
        let row_span = covered_rows.len();
        let col_span = covered_cols.len();

        // Mark covered cells as occupied.
        for &r in &covered_rows {
            for &c in &covered_cols {
                if r < num_rows && c < num_cols {
                    occupied[r][c] = true;
                }
            }
        }

        spanning_cells.push(TableCell {
            row: first_row,
            col: first_col,
            row_span,
            col_span,
            bbox: span_bbox,
            confidence: span_det.confidence,
        });
    }

    // Second pass: build regular cells for unoccupied intersections +
    // insert spanning cells at their anchor positions.
    for r in 0..num_rows {
        let mut cells: Vec<TableCell> = Vec::with_capacity(num_cols);
        for c in 0..num_cols {
            // Check if a spanning cell is anchored here.
            if let Some(span_cell) = spanning_cells.iter().find(|sc| sc.row == r && sc.col == c) {
                cells.push(span_cell.clone());
            } else if !occupied[r][c] {
                // Regular 1x1 cell: intersection of row and column bboxes.
                let row_det = row_dets[r];
                let col_det = col_dets[c];
                let cell_bbox = intersect_bbox(
                    &[row_det.x1, row_det.y1, row_det.x2, row_det.y2],
                    &[col_det.x1, col_det.y1, col_det.x2, col_det.y2],
                );
                let confidence = f32::midpoint(row_det.confidence, col_det.confidence);
                cells.push(TableCell {
                    row: r,
                    col: c,
                    row_span: 1,
                    col_span: 1,
                    bbox: cell_bbox,
                    confidence,
                });
            }
            // else: occupied by a spanning cell anchored elsewhere — skip.
        }
        table_rows.push(TableRow { cells });
    }

    StructuredTable {
        rows: table_rows,
        num_rows,
        num_cols,
        caption: None,
    }
}

// ---------------------------------------------------------------------------
// IoU computation
// ---------------------------------------------------------------------------

/// Compute Intersection over Union between two bounding boxes.
///
/// Both boxes are `[x1, y1, x2, y2]` format. Returns 0.0 for degenerate
/// or non-overlapping boxes.
#[must_use]
pub fn compute_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
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

// ---------------------------------------------------------------------------
// Serialization: HTML
// ---------------------------------------------------------------------------

/// Convert a [`StructuredTable`] to an HTML `<table>` string.
///
/// Emits `rowspan` and `colspan` attributes for spanning cells.
/// Empty cells get `&nbsp;` content.
#[must_use]
pub fn to_html(table: &StructuredTable) -> String {
    let mut html = String::with_capacity(256);
    html.push_str("<table>\n");

    if let Some(ref caption) = table.caption {
        html.push_str("  <caption>");
        push_html_escaped(&mut html, caption);
        html.push_str("</caption>\n");
    }

    // Track which cells are covered by a prior rowspan so we skip them.
    let mut covered = vec![vec![false; table.num_cols]; table.num_rows];

    for row in &table.rows {
        html.push_str("  <tr>\n");
        for cell in &row.cells {
            // Mark cells covered by this cell's span.
            for dr in 0..cell.row_span {
                for dc in 0..cell.col_span {
                    let r = cell.row + dr;
                    let c = cell.col + dc;
                    if r < table.num_rows && c < table.num_cols {
                        covered[r][c] = true;
                    }
                }
            }

            html.push_str("    <td");
            if cell.row_span > 1 {
                html.push_str(&format!(" rowspan=\"{}\"", cell.row_span));
            }
            if cell.col_span > 1 {
                html.push_str(&format!(" colspan=\"{}\"", cell.col_span));
            }
            html.push_str(">&nbsp;</td>\n");
        }
        html.push_str("  </tr>\n");
    }

    html.push_str("</table>");
    html
}

// ---------------------------------------------------------------------------
// Serialization: Markdown
// ---------------------------------------------------------------------------

/// Convert a [`StructuredTable`] to a Markdown pipe table.
///
/// Spanning cells are represented by repeating the cell placeholder across
/// the spanned columns. Row spans are approximated (Markdown has no native
/// rowspan support) by leaving spanned rows empty.
#[must_use]
pub fn to_markdown_table(table: &StructuredTable) -> String {
    if table.num_rows == 0 || table.num_cols == 0 {
        return String::new();
    }

    // Build a grid of cell labels.
    let mut grid = vec![vec![String::new(); table.num_cols]; table.num_rows];
    for row in &table.rows {
        for cell in &row.cells {
            let label = format!("({},{})", cell.row, cell.col);
            for dr in 0..cell.row_span {
                for dc in 0..cell.col_span {
                    let r = cell.row + dr;
                    let c = cell.col + dc;
                    if r < table.num_rows && c < table.num_cols {
                        if dr == 0 && dc == 0 {
                            grid[r][c] = label.clone();
                        }
                        // Spanned cells remain empty string (visual placeholder).
                    }
                }
            }
        }
    }

    let mut lines = Vec::with_capacity(table.num_rows + 1);

    // Header row.
    let header: Vec<&str> = grid[0].iter().map(String::as_str).collect();
    lines.push(format!("| {} |", header.join(" | ")));

    // Separator.
    let sep: Vec<&str> = (0..table.num_cols).map(|_| "---").collect();
    lines.push(format!("| {} |", sep.join(" | ")));

    // Data rows.
    for row_cells in grid.iter().skip(1) {
        let cols: Vec<&str> = row_cells.iter().map(String::as_str).collect();
        lines.push(format!("| {} |", cols.join(" | ")));
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Serialization: CSV
// ---------------------------------------------------------------------------

/// Convert a [`StructuredTable`] to CSV format.
///
/// Each cell is represented by its `(row,col)` label. Spanning cells appear
/// at their anchor position; spanned positions are empty.
#[must_use]
pub fn to_csv(table: &StructuredTable) -> String {
    if table.num_rows == 0 || table.num_cols == 0 {
        return String::new();
    }

    // Build a grid of cell labels (same logic as markdown).
    let mut grid = vec![vec![String::new(); table.num_cols]; table.num_rows];
    for row in &table.rows {
        for cell in &row.cells {
            let label = format!("({},{})", cell.row, cell.col);
            grid[cell.row][cell.col] = label;
        }
    }

    let mut lines = Vec::with_capacity(table.num_rows);
    for row_cells in &grid {
        let escaped: Vec<String> = row_cells.iter().map(|c| csv_escape(c)).collect();
        lines.push(escaped.join(","));
    }
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Axis selector for overlap computation.
#[derive(Debug, Clone, Copy)]
enum Axis {
    Row,
    Column,
}

/// Find which row or column detections overlap with `bbox` above a tolerance threshold.
///
/// For rows, we measure y-overlap ratio; for columns, x-overlap ratio.
/// Returns sorted indices of overlapping detections.
fn find_overlapping_indices(
    dets: &[&Detection],
    bbox: &[f32; 4],
    tolerance: f32,
    axis: Axis,
) -> Vec<usize> {
    let mut indices = Vec::new();
    for (i, det) in dets.iter().enumerate() {
        let overlap = match axis {
            Axis::Row => {
                // Y-axis overlap ratio.
                let det_min = det.y1;
                let det_max = det.y2;
                let bbox_min = bbox[1];
                let bbox_max = bbox[3];
                axis_overlap_ratio(det_min, det_max, bbox_min, bbox_max)
            }
            Axis::Column => {
                // X-axis overlap ratio.
                let det_min = det.x1;
                let det_max = det.x2;
                let bbox_min = bbox[0];
                let bbox_max = bbox[2];
                axis_overlap_ratio(det_min, det_max, bbox_min, bbox_max)
            }
        };
        if overlap > tolerance {
            indices.push(i);
        }
    }
    indices
}

/// Compute overlap ratio between two 1D intervals.
///
/// Returns the intersection length divided by the smaller interval's length.
/// Returns 0.0 for zero-length or non-overlapping intervals.
fn axis_overlap_ratio(a_min: f32, a_max: f32, b_min: f32, b_max: f32) -> f32 {
    let inter_min = a_min.max(b_min);
    let inter_max = a_max.min(b_max);
    let inter_len = (inter_max - inter_min).max(0.0);

    let a_len = (a_max - a_min).max(0.0);
    let b_len = (b_max - b_min).max(0.0);
    let min_len = a_len.min(b_len);

    if min_len <= 0.0 {
        return 0.0;
    }
    inter_len / min_len
}

/// Compute the intersection bbox of two `[x1, y1, x2, y2]` boxes.
///
/// If boxes do not overlap, returns a degenerate box where x2 <= x1 or y2 <= y1.
fn intersect_bbox(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

/// Escape a CSV field: wrap in quotes if it contains commas, quotes, or newlines.
fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// Push HTML-escaped text into a string buffer.
fn push_html_escaped(buf: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '<' => buf.push_str("&lt;"),
            '>' => buf.push_str("&gt;"),
            '&' => buf.push_str("&amp;"),
            '"' => buf.push_str("&quot;"),
            _ => buf.push(ch),
        }
    }
}

#[cfg(test)]
#[path = "table_structure_tests.rs"]
mod tests;
