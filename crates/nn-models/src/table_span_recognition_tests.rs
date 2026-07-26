// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn make_cell(bbox: [f32; 4], confidence: f32) -> RawCellDetection {
    RawCellDetection {
        bbox,
        confidence,
        text: None,
    }
}

#[test]
fn test_infer_grid_basic() {
    let cells = vec![
        make_cell([0.0, 0.0, 50.0, 20.0], 0.9),
        make_cell([60.0, 0.0, 110.0, 20.0], 0.9),
        make_cell([0.0, 30.0, 50.0, 50.0], 0.9),
        make_cell([60.0, 30.0, 110.0, 50.0], 0.9),
    ];
    let config = SpanRecognitionConfig::default();
    let grid = infer_grid(&cells, &config);
    assert_eq!(grid.row_bounds.len(), 2);
    assert_eq!(grid.col_bounds.len(), 2);
}

#[test]
fn test_infer_grid_empty() {
    let config = SpanRecognitionConfig::default();
    let grid = infer_grid(&[], &config);
    assert!(grid.row_bounds.is_empty());
    assert!(grid.col_bounds.is_empty());
}

#[test]
fn test_recognize_spans_simple_2x2() {
    let cells = vec![
        make_cell([0.0, 0.0, 50.0, 20.0], 0.9),
        make_cell([60.0, 0.0, 110.0, 20.0], 0.8),
        make_cell([0.0, 30.0, 50.0, 50.0], 0.85),
        make_cell([60.0, 30.0, 110.0, 50.0], 0.7),
    ];
    let config = SpanRecognitionConfig::default();
    let table = recognize_spans(&cells, &config);
    assert_eq!(table.num_rows, 2);
    assert_eq!(table.num_cols, 2);
    assert_eq!(count_spanning_cells(&table), 0);
}

#[test]
fn test_recognize_spans_with_colspan() {
    // Row 1: two cells. Row 2: one wide cell spanning both columns.
    // Use midpoints that cluster together to get exactly 2 rows, 2 cols.
    let cells = vec![
        make_cell([0.0, 0.0, 50.0, 20.0], 0.9),
        make_cell([60.0, 0.0, 110.0, 20.0], 0.8),
        make_cell([0.0, 25.0, 110.0, 45.0], 0.85), // spans both columns, y-mid ~35
    ];
    let config = SpanRecognitionConfig {
        row_cluster_tolerance: 12.0,
        col_cluster_tolerance: 12.0,
        ..Default::default()
    };
    let table = recognize_spans(&cells, &config);
    assert_eq!(table.num_rows, 2);
    // The wide cell's x-midpoint (55) falls between the two column clusters,
    // so the grid may have 2 or 3 columns depending on tolerance. What matters
    // is that at least one cell spans multiple columns.
    assert!(table.num_cols >= 2);
    assert!(count_spanning_cells(&table) >= 1);
}

#[test]
fn test_recognize_spans_with_rowspan() {
    // Col 1: one tall cell spanning both rows. Col 2: two cells.
    // The tall cell's y-midpoint (25) is between the two row clusters (10, 40).
    let cells = vec![
        make_cell([0.0, 0.0, 50.0, 50.0], 0.9), // spans both rows
        make_cell([60.0, 0.0, 110.0, 20.0], 0.8),
        make_cell([60.0, 30.0, 110.0, 50.0], 0.85),
    ];
    let config = SpanRecognitionConfig {
        row_cluster_tolerance: 12.0,
        col_cluster_tolerance: 12.0,
        ..Default::default()
    };
    let table = recognize_spans(&cells, &config);
    // The tall cell's midpoint may form its own row cluster, giving 3 rows.
    // What matters is that at least one cell has row_span > 1.
    assert!(table.num_rows >= 2);
    assert_eq!(table.num_cols, 2);
    assert!(count_spanning_cells(&table) >= 1);
}

#[test]
fn test_recognize_spans_empty() {
    let config = SpanRecognitionConfig::default();
    let table = recognize_spans(&[], &config);
    assert_eq!(table.num_rows, 0);
    assert_eq!(table.num_cols, 0);
}

#[test]
fn test_validate_span_coverage_empty() {
    let table = StructuredTable {
        rows: Vec::new(),
        num_rows: 0,
        num_cols: 0,
        caption: None,
    };
    assert!(validate_span_coverage(&table));
}

#[test]
fn test_validate_span_coverage_simple() {
    let table = StructuredTable {
        rows: vec![
            TableRow {
                cells: vec![
                    TableCell {
                        row: 0,
                        col: 0,
                        row_span: 1,
                        col_span: 1,
                        bbox: [0.0; 4],
                        confidence: 1.0,
                    },
                    TableCell {
                        row: 0,
                        col: 1,
                        row_span: 1,
                        col_span: 1,
                        bbox: [0.0; 4],
                        confidence: 1.0,
                    },
                ],
            },
            TableRow {
                cells: vec![
                    TableCell {
                        row: 1,
                        col: 0,
                        row_span: 1,
                        col_span: 1,
                        bbox: [0.0; 4],
                        confidence: 1.0,
                    },
                    TableCell {
                        row: 1,
                        col: 1,
                        row_span: 1,
                        col_span: 1,
                        bbox: [0.0; 4],
                        confidence: 1.0,
                    },
                ],
            },
        ],
        num_rows: 2,
        num_cols: 2,
        caption: None,
    };
    assert!(validate_span_coverage(&table));
}

#[test]
fn test_validate_span_coverage_with_span() {
    let table = StructuredTable {
        rows: vec![
            TableRow {
                cells: vec![TableCell {
                    row: 0,
                    col: 0,
                    row_span: 1,
                    col_span: 2,
                    bbox: [0.0; 4],
                    confidence: 1.0,
                }],
            },
            TableRow {
                cells: vec![
                    TableCell {
                        row: 1,
                        col: 0,
                        row_span: 1,
                        col_span: 1,
                        bbox: [0.0; 4],
                        confidence: 1.0,
                    },
                    TableCell {
                        row: 1,
                        col: 1,
                        row_span: 1,
                        col_span: 1,
                        bbox: [0.0; 4],
                        confidence: 1.0,
                    },
                ],
            },
        ],
        num_rows: 2,
        num_cols: 2,
        caption: None,
    };
    assert!(validate_span_coverage(&table));
}

#[test]
fn test_cluster_1d_basic() {
    let values = vec![1.0, 1.5, 10.0, 10.3, 20.0];
    let assignments = cluster_1d(&values, 3.0);
    // 1.0 and 1.5 should be in the same cluster.
    assert_eq!(assignments[0], assignments[1]);
    // 10.0 and 10.3 should be in the same cluster.
    assert_eq!(assignments[2], assignments[3]);
    // Different cluster groups.
    assert_ne!(assignments[0], assignments[2]);
    assert_ne!(assignments[2], assignments[4]);
}

#[test]
fn test_cluster_1d_empty() {
    let assignments = cluster_1d(&[], 5.0);
    assert!(assignments.is_empty());
}

#[test]
fn test_count_spanning_cells() {
    let table = StructuredTable {
        rows: vec![TableRow {
            cells: vec![
                TableCell {
                    row: 0,
                    col: 0,
                    row_span: 2,
                    col_span: 1,
                    bbox: [0.0; 4],
                    confidence: 1.0,
                },
                TableCell {
                    row: 0,
                    col: 1,
                    row_span: 1,
                    col_span: 1,
                    bbox: [0.0; 4],
                    confidence: 1.0,
                },
            ],
        }],
        num_rows: 2,
        num_cols: 2,
        caption: None,
    };
    assert_eq!(count_spanning_cells(&table), 1);
}
