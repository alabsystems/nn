// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Table structure recognition with row/column span inference.
//!
//! Extends [`super::table_structure`] with heuristic methods for detecting
//! row and column spans when explicit spanning-cell detections are missing
//! or unreliable. This is common with UniTable output which produces a flat
//! token sequence rather than explicit cell detections.
//!
//! # Approach
//!
//! 1. **Grid alignment**: Given a set of cell bounding boxes, infer the
//!    row/column grid by clustering vertical and horizontal positions.
//! 2. **Span detection**: Cells whose bounding boxes span multiple grid
//!    rows/columns are annotated with `row_span > 1` or `col_span > 1`.
//! 3. **Empty cell inference**: Grid positions not covered by any cell
//!    are marked as empty.

use crate::table_structure::{StructuredTable, TableCell, TableRow};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for span-aware table recognition.
#[derive(Debug, Clone)]
pub struct SpanRecognitionConfig {
    /// Tolerance for grouping y-coordinates into the same row (pixels).
    pub row_cluster_tolerance: f32,
    /// Tolerance for grouping x-coordinates into the same column (pixels).
    pub col_cluster_tolerance: f32,
    /// Minimum overlap ratio for a cell to be assigned to a grid position.
    pub min_overlap_ratio: f32,
}

impl Default for SpanRecognitionConfig {
    fn default() -> Self {
        Self {
            row_cluster_tolerance: 8.0,
            col_cluster_tolerance: 8.0,
            min_overlap_ratio: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Cell bbox input
// ---------------------------------------------------------------------------

/// A raw cell detection with bounding box and optional text.
#[derive(Debug, Clone)]
pub struct RawCellDetection {
    /// Bounding box `[x1, y1, x2, y2]` in pixel coordinates.
    pub bbox: [f32; 4],
    /// Detection confidence.
    pub confidence: f32,
    /// Optional cell text content.
    pub text: Option<String>,
}

// ---------------------------------------------------------------------------
// Grid inference
// ---------------------------------------------------------------------------

/// Inferred row/column grid boundaries.
#[derive(Debug, Clone)]
pub struct GridBoundaries {
    /// Sorted row boundaries: `row_boundaries[i]` is the `(y_min, y_max)` of row i.
    pub row_bounds: Vec<(f32, f32)>,
    /// Sorted column boundaries: `col_boundaries[i]` is the `(x_min, x_max)` of column i.
    pub col_bounds: Vec<(f32, f32)>,
}

/// Infer grid boundaries from a set of cell detections.
///
/// Clusters the vertical midpoints of cells into row bands and horizontal
/// midpoints into column bands.
#[must_use]
pub fn infer_grid(cells: &[RawCellDetection], config: &SpanRecognitionConfig) -> GridBoundaries {
    // Collect midpoints.
    let y_mids: Vec<f32> = cells
        .iter()
        .map(|c| (c.bbox[1] + c.bbox[3]) * 0.5)
        .collect();
    let x_mids: Vec<f32> = cells
        .iter()
        .map(|c| (c.bbox[0] + c.bbox[2]) * 0.5)
        .collect();

    let row_clusters = cluster_1d(&y_mids, config.row_cluster_tolerance);
    let col_clusters = cluster_1d(&x_mids, config.col_cluster_tolerance);

    // Convert clusters to bounds by taking the min y1 and max y2 of cells
    // assigned to each cluster.
    let row_bounds = clusters_to_bounds(cells, &row_clusters, Axis::Y);
    let col_bounds = clusters_to_bounds(cells, &col_clusters, Axis::X);

    GridBoundaries {
        row_bounds,
        col_bounds,
    }
}

/// Assign cells to grid positions and detect row/column spans.
///
/// Returns a [`StructuredTable`] with proper `row_span` and `col_span`
/// annotations for cells that cover multiple grid positions.
#[must_use]
pub fn recognize_spans(
    cells: &[RawCellDetection],
    config: &SpanRecognitionConfig,
) -> StructuredTable {
    let grid = infer_grid(cells, config);
    let num_rows = grid.row_bounds.len();
    let num_cols = grid.col_bounds.len();

    if num_rows == 0 || num_cols == 0 {
        return StructuredTable {
            rows: Vec::new(),
            num_rows: 0,
            num_cols: 0,
            caption: None,
        };
    }

    // Track which grid positions are occupied.
    let mut occupied = vec![vec![false; num_cols]; num_rows];
    let mut all_cells: Vec<TableCell> = Vec::new();

    for cell in cells {
        let covered_rows = find_covered_indices(
            &grid.row_bounds,
            cell.bbox[1],
            cell.bbox[3],
            config.min_overlap_ratio,
        );
        let covered_cols = find_covered_indices(
            &grid.col_bounds,
            cell.bbox[0],
            cell.bbox[2],
            config.min_overlap_ratio,
        );

        if covered_rows.is_empty() || covered_cols.is_empty() {
            continue;
        }

        let first_row = covered_rows[0];
        let first_col = covered_cols[0];
        let row_span = covered_rows.len();
        let col_span = covered_cols.len();

        // Mark grid positions as occupied.
        for &r in &covered_rows {
            for &c in &covered_cols {
                if r < num_rows && c < num_cols {
                    occupied[r][c] = true;
                }
            }
        }

        all_cells.push(TableCell {
            row: first_row,
            col: first_col,
            row_span,
            col_span,
            bbox: cell.bbox,
            confidence: cell.confidence,
        });
    }

    // Build table rows, including empty cells for unoccupied positions.
    let mut table_rows = Vec::with_capacity(num_rows);
    for r in 0..num_rows {
        let mut row_cells = Vec::new();
        for c in 0..num_cols {
            if let Some(tc) = all_cells.iter().find(|tc| tc.row == r && tc.col == c) {
                row_cells.push(tc.clone());
            } else if !occupied[r][c] {
                // Empty cell placeholder.
                let rb = &grid.row_bounds[r];
                let cb = &grid.col_bounds[c];
                row_cells.push(TableCell {
                    row: r,
                    col: c,
                    row_span: 1,
                    col_span: 1,
                    bbox: [cb.0, rb.0, cb.1, rb.1],
                    confidence: 0.0,
                });
            }
        }
        table_rows.push(TableRow { cells: row_cells });
    }

    StructuredTable {
        rows: table_rows,
        num_rows,
        num_cols,
        caption: None,
    }
}

// ---------------------------------------------------------------------------
// Span validation
// ---------------------------------------------------------------------------

/// Validate that a structured table has consistent span coverage.
///
/// Returns `true` if every grid position is covered exactly once
/// (either by a direct cell or by a spanning cell).
#[must_use]
pub fn validate_span_coverage(table: &StructuredTable) -> bool {
    if table.num_rows == 0 || table.num_cols == 0 {
        return true;
    }

    let mut coverage = vec![vec![0u32; table.num_cols]; table.num_rows];

    for row in &table.rows {
        for cell in &row.cells {
            for dr in 0..cell.row_span {
                for dc in 0..cell.col_span {
                    let r = cell.row + dr;
                    let c = cell.col + dc;
                    if r < table.num_rows && c < table.num_cols {
                        coverage[r][c] += 1;
                    }
                }
            }
        }
    }

    // Every position should be covered exactly once.
    coverage
        .iter()
        .all(|row| row.iter().all(|&count| count == 1))
}

/// Count the total number of spanning cells (cells where row_span > 1 or col_span > 1).
#[must_use]
pub fn count_spanning_cells(table: &StructuredTable) -> usize {
    table
        .rows
        .iter()
        .flat_map(|r| &r.cells)
        .filter(|c| c.row_span > 1 || c.col_span > 1)
        .count()
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Axis {
    X,
    Y,
}

/// Cluster 1D values with a tolerance threshold.
///
/// Returns cluster assignments: `result[i]` is the cluster ID for `values[i]`.
fn cluster_1d(values: &[f32], tolerance: f32) -> Vec<usize> {
    if values.is_empty() {
        return Vec::new();
    }

    // Sort indices by value.
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut assignments = vec![0usize; values.len()];
    let mut cluster_id = 0usize;
    let mut prev_val = indexed[0].1;
    assignments[indexed[0].0] = cluster_id;

    for &(idx, val) in indexed.iter().skip(1) {
        if (val - prev_val).abs() > tolerance {
            cluster_id += 1;
        }
        assignments[idx] = cluster_id;
        prev_val = val;
    }

    assignments
}

/// Convert cluster assignments to bounding ranges.
fn clusters_to_bounds(
    cells: &[RawCellDetection],
    assignments: &[usize],
    axis: Axis,
) -> Vec<(f32, f32)> {
    if assignments.is_empty() {
        return Vec::new();
    }

    let num_clusters = assignments.iter().copied().max().unwrap_or(0) + 1;
    let mut bounds = vec![(f32::INFINITY, f32::NEG_INFINITY); num_clusters];

    for (i, cell) in cells.iter().enumerate() {
        let cluster = assignments[i];
        let (lo, hi) = match axis {
            Axis::X => (cell.bbox[0], cell.bbox[2]),
            Axis::Y => (cell.bbox[1], cell.bbox[3]),
        };
        bounds[cluster].0 = bounds[cluster].0.min(lo);
        bounds[cluster].1 = bounds[cluster].1.max(hi);
    }

    // Filter out clusters that never got assigned (shouldn't happen but be safe).
    bounds
        .into_iter()
        .filter(|(lo, hi)| lo.is_finite() && hi.is_finite() && hi > lo)
        .collect()
}

/// Find which grid band indices a cell's range overlaps.
fn find_covered_indices(
    bounds: &[(f32, f32)],
    cell_min: f32,
    cell_max: f32,
    min_overlap_ratio: f32,
) -> Vec<usize> {
    let cell_len = (cell_max - cell_min).max(0.0);
    if cell_len <= 0.0 {
        return Vec::new();
    }

    bounds
        .iter()
        .enumerate()
        .filter(|(_i, (band_min, band_max))| {
            let band_len = (band_max - band_min).max(0.0);
            if band_len <= 0.0 {
                return false;
            }
            let overlap_min = cell_min.max(*band_min);
            let overlap_max = cell_max.min(*band_max);
            let overlap_len = (overlap_max - overlap_min).max(0.0);
            let overlap_ratio = overlap_len / band_len.min(cell_len);
            overlap_ratio >= min_overlap_ratio
        })
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
#[path = "table_span_recognition_tests.rs"]
mod tests;
