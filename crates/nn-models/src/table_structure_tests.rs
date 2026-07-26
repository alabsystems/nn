// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for table structure recognition: cell parsing, IoU, serialization.

use nn_core::layers::vision::Detection;

use super::*;

// ---------------------------------------------------------------------------
// Helper: build a Detection from bbox + class
// ---------------------------------------------------------------------------

fn det(x1: f32, y1: f32, x2: f32, y2: f32, confidence: f32, class_id: u32) -> Detection {
    Detection {
        x1,
        y1,
        x2,
        y2,
        confidence,
        class_id,
    }
}

// ---------------------------------------------------------------------------
// IoU tests
// ---------------------------------------------------------------------------

#[test]
fn test_compute_iou_identical_boxes() {
    let a = [0.0, 0.0, 1.0, 1.0];
    let b = [0.0, 0.0, 1.0, 1.0];
    let iou = compute_iou(&a, &b);
    assert!(
        (iou - 1.0).abs() < 1e-6,
        "identical boxes should have IoU=1.0, got {iou}"
    );
}

#[test]
fn test_compute_iou_no_overlap() {
    let a = [0.0, 0.0, 1.0, 1.0];
    let b = [2.0, 2.0, 3.0, 3.0];
    let iou = compute_iou(&a, &b);
    assert!(
        (iou - 0.0).abs() < 1e-6,
        "non-overlapping boxes should have IoU=0.0, got {iou}"
    );
}

#[test]
fn test_compute_iou_partial_overlap() {
    let a = [0.0, 0.0, 2.0, 2.0]; // area = 4
    let b = [1.0, 1.0, 3.0, 3.0]; // area = 4
                                  // intersection: [1,1,2,2] = area 1
                                  // union: 4 + 4 - 1 = 7
    let iou = compute_iou(&a, &b);
    let expected = 1.0 / 7.0;
    assert!(
        (iou - expected).abs() < 1e-6,
        "expected IoU={expected}, got {iou}"
    );
}

#[test]
fn test_compute_iou_contained_box() {
    let a = [0.0, 0.0, 4.0, 4.0]; // area = 16
    let b = [1.0, 1.0, 2.0, 2.0]; // area = 1, fully inside a
                                  // intersection = 1, union = 16
    let iou = compute_iou(&a, &b);
    let expected = 1.0 / 16.0;
    assert!(
        (iou - expected).abs() < 1e-6,
        "contained box IoU expected {expected}, got {iou}"
    );
}

#[test]
fn test_compute_iou_degenerate_box() {
    let a = [1.0, 1.0, 0.0, 0.0]; // degenerate: x2 < x1
    let b = [0.0, 0.0, 1.0, 1.0];
    let iou = compute_iou(&a, &b);
    assert!(
        iou.abs() < 1e-6,
        "degenerate box should have IoU=0.0, got {iou}"
    );
}

// ---------------------------------------------------------------------------
// parse_structure tests
// ---------------------------------------------------------------------------

#[test]
fn test_parse_structure_empty_detections() {
    let config = TableStructureConfig::default();
    let table = parse_structure(&[], &config);
    assert_eq!(table.num_rows, 0);
    assert_eq!(table.num_cols, 0);
    assert!(table.rows.is_empty());
}

#[test]
fn test_parse_structure_no_rows() {
    let config = TableStructureConfig::default();
    // Only columns, no rows.
    let dets = vec![
        det(0.0, 0.0, 0.3, 1.0, 0.9, CLASS_COLUMN),
        det(0.3, 0.0, 0.6, 1.0, 0.9, CLASS_COLUMN),
    ];
    let table = parse_structure(&dets, &config);
    assert_eq!(table.num_rows, 0);
    assert_eq!(table.num_cols, 0);
}

#[test]
fn test_parse_structure_simple_2x2() {
    let config = TableStructureConfig::default();
    let dets = vec![
        // 2 rows
        det(0.0, 0.0, 1.0, 0.4, 0.9, CLASS_ROW),
        det(0.0, 0.5, 1.0, 0.9, 0.8, CLASS_ROW),
        // 2 columns
        det(0.0, 0.0, 0.4, 1.0, 0.85, CLASS_COLUMN),
        det(0.5, 0.0, 0.9, 1.0, 0.85, CLASS_COLUMN),
    ];
    let table = parse_structure(&dets, &config);
    assert_eq!(table.num_rows, 2);
    assert_eq!(table.num_cols, 2);
    assert_eq!(table.rows.len(), 2);

    // Each row should have 2 cells, all 1x1.
    for (r, row) in table.rows.iter().enumerate() {
        assert_eq!(row.cells.len(), 2, "row {r} should have 2 cells");
        for cell in &row.cells {
            assert_eq!(cell.row_span, 1);
            assert_eq!(cell.col_span, 1);
        }
    }
}

#[test]
fn test_parse_structure_3x3_grid() {
    let config = TableStructureConfig::default();
    let dets = vec![
        // 3 rows
        det(0.0, 0.0, 1.0, 0.3, 0.9, CLASS_ROW),
        det(0.0, 0.35, 1.0, 0.65, 0.9, CLASS_ROW),
        det(0.0, 0.7, 1.0, 1.0, 0.9, CLASS_ROW),
        // 3 columns
        det(0.0, 0.0, 0.3, 1.0, 0.9, CLASS_COLUMN),
        det(0.35, 0.0, 0.65, 1.0, 0.9, CLASS_COLUMN),
        det(0.7, 0.0, 1.0, 1.0, 0.9, CLASS_COLUMN),
    ];
    let table = parse_structure(&dets, &config);
    assert_eq!(table.num_rows, 3);
    assert_eq!(table.num_cols, 3);
    assert_eq!(table.rows.len(), 3);

    // Total cells should be 9.
    let total_cells: usize = table.rows.iter().map(|r| r.cells.len()).sum();
    assert_eq!(total_cells, 9, "3x3 grid should have 9 cells");
}

#[test]
fn test_parse_structure_with_spanning_cell() {
    let config = TableStructureConfig {
        row_tolerance: 0.2,
        col_tolerance: 0.2,
        ..Default::default()
    };

    let dets = vec![
        // 2 rows
        det(0.0, 0.0, 1.0, 0.45, 0.9, CLASS_ROW),
        det(0.0, 0.55, 1.0, 1.0, 0.9, CLASS_ROW),
        // 2 columns
        det(0.0, 0.0, 0.45, 1.0, 0.9, CLASS_COLUMN),
        det(0.55, 0.0, 1.0, 1.0, 0.9, CLASS_COLUMN),
        // Spanning cell covering row 0, both columns
        det(0.0, 0.0, 1.0, 0.45, 0.95, CLASS_SPANNING_CELL),
    ];
    let table = parse_structure(&dets, &config);
    assert_eq!(table.num_rows, 2);
    assert_eq!(table.num_cols, 2);

    // Row 0 should have 1 cell (the spanning cell covering both columns).
    let row0_cells = &table.rows[0].cells;
    assert_eq!(
        row0_cells.len(),
        1,
        "spanning cell should be the only cell in row 0"
    );
    assert_eq!(row0_cells[0].col_span, 2);
    assert_eq!(row0_cells[0].row_span, 1);
}

#[test]
fn test_parse_structure_ignores_table_class() {
    let config = TableStructureConfig::default();
    // Only a table-level detection, no rows/columns.
    let dets = vec![det(0.0, 0.0, 1.0, 1.0, 0.99, CLASS_TABLE)];
    let table = parse_structure(&dets, &config);
    assert_eq!(table.num_rows, 0);
    assert_eq!(table.num_cols, 0);
}

// ---------------------------------------------------------------------------
// Serialization tests
// ---------------------------------------------------------------------------

#[test]
fn test_to_html_empty_table() {
    let table = StructuredTable {
        rows: vec![],
        num_rows: 0,
        num_cols: 0,
        caption: None,
    };
    let html = to_html(&table);
    assert!(html.contains("<table>"));
    assert!(html.contains("</table>"));
    assert!(!html.contains("<tr>"));
}

#[test]
fn test_to_html_with_caption() {
    let table = StructuredTable {
        rows: vec![],
        num_rows: 0,
        num_cols: 0,
        caption: Some("Test <Caption>".to_string()),
    };
    let html = to_html(&table);
    assert!(
        html.contains("Test &lt;Caption&gt;"),
        "HTML should escape angle brackets in caption"
    );
}

#[test]
fn test_to_html_with_cells() {
    let table = StructuredTable {
        rows: vec![TableRow {
            cells: vec![
                TableCell {
                    row: 0,
                    col: 0,
                    row_span: 1,
                    col_span: 1,
                    bbox: [0.0, 0.0, 0.5, 0.5],
                    confidence: 0.9,
                },
                TableCell {
                    row: 0,
                    col: 1,
                    row_span: 1,
                    col_span: 1,
                    bbox: [0.5, 0.0, 1.0, 0.5],
                    confidence: 0.8,
                },
            ],
        }],
        num_rows: 1,
        num_cols: 2,
        caption: None,
    };
    let html = to_html(&table);
    assert_eq!(html.matches("<td").count(), 2);
    assert!(
        !html.contains("rowspan"),
        "1x1 cells should not have rowspan"
    );
    assert!(
        !html.contains("colspan"),
        "1x1 cells should not have colspan"
    );
}

#[test]
fn test_to_html_spanning_attributes() {
    let table = StructuredTable {
        rows: vec![TableRow {
            cells: vec![TableCell {
                row: 0,
                col: 0,
                row_span: 2,
                col_span: 3,
                bbox: [0.0, 0.0, 1.0, 1.0],
                confidence: 0.9,
            }],
        }],
        num_rows: 2,
        num_cols: 3,
        caption: None,
    };
    let html = to_html(&table);
    assert!(html.contains("rowspan=\"2\""), "should have rowspan=2");
    assert!(html.contains("colspan=\"3\""), "should have colspan=3");
}

#[test]
fn test_to_markdown_table_empty() {
    let table = StructuredTable {
        rows: vec![],
        num_rows: 0,
        num_cols: 0,
        caption: None,
    };
    let md = to_markdown_table(&table);
    assert!(md.is_empty());
}

#[test]
fn test_to_markdown_table_2x2() {
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
                        confidence: 0.9,
                    },
                    TableCell {
                        row: 0,
                        col: 1,
                        row_span: 1,
                        col_span: 1,
                        bbox: [0.0; 4],
                        confidence: 0.9,
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
                        confidence: 0.9,
                    },
                    TableCell {
                        row: 1,
                        col: 1,
                        row_span: 1,
                        col_span: 1,
                        bbox: [0.0; 4],
                        confidence: 0.9,
                    },
                ],
            },
        ],
        num_rows: 2,
        num_cols: 2,
        caption: None,
    };
    let md = to_markdown_table(&table);
    let lines: Vec<&str> = md.lines().collect();
    assert_eq!(lines.len(), 3, "2x2 table: header + separator + 1 data row");
    assert!(lines[0].contains("(0,0)"));
    assert!(lines[0].contains("(0,1)"));
    assert!(lines[1].contains("---"));
    assert!(lines[2].contains("(1,0)"));
    assert!(lines[2].contains("(1,1)"));
}

#[test]
fn test_to_csv_empty() {
    let table = StructuredTable {
        rows: vec![],
        num_rows: 0,
        num_cols: 0,
        caption: None,
    };
    let csv = to_csv(&table);
    assert!(csv.is_empty());
}

#[test]
fn test_to_csv_2x2() {
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
                        confidence: 0.9,
                    },
                    TableCell {
                        row: 0,
                        col: 1,
                        row_span: 1,
                        col_span: 1,
                        bbox: [0.0; 4],
                        confidence: 0.9,
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
                        confidence: 0.9,
                    },
                    TableCell {
                        row: 1,
                        col: 1,
                        row_span: 1,
                        col_span: 1,
                        bbox: [0.0; 4],
                        confidence: 0.9,
                    },
                ],
            },
        ],
        num_rows: 2,
        num_cols: 2,
        caption: None,
    };
    let csv = to_csv(&table);
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains("(0,0)"));
    assert!(lines[0].contains("(0,1)"));
}

// ---------------------------------------------------------------------------
// Edge case tests
// ---------------------------------------------------------------------------

#[test]
fn test_compute_iou_zero_area_boxes() {
    // Line (zero height).
    let a = [0.0, 0.5, 1.0, 0.5];
    let b = [0.0, 0.0, 1.0, 1.0];
    let iou = compute_iou(&a, &b);
    assert!(
        iou.abs() < 1e-6,
        "zero-area box should have IoU=0.0, got {iou}"
    );
}

#[test]
fn test_parse_structure_single_row_single_col() {
    let config = TableStructureConfig::default();
    let dets = vec![
        det(0.0, 0.0, 1.0, 1.0, 0.9, CLASS_ROW),
        det(0.0, 0.0, 1.0, 1.0, 0.9, CLASS_COLUMN),
    ];
    let table = parse_structure(&dets, &config);
    assert_eq!(table.num_rows, 1);
    assert_eq!(table.num_cols, 1);
    assert_eq!(table.rows.len(), 1);
    assert_eq!(table.rows[0].cells.len(), 1);
    assert_eq!(table.rows[0].cells[0].row, 0);
    assert_eq!(table.rows[0].cells[0].col, 0);
}

#[test]
fn test_default_config_values() {
    let config = TableStructureConfig::default();
    assert!((config.iou_threshold - 0.5).abs() < 1e-6);
    assert!((config.row_tolerance - 0.3).abs() < 1e-6);
    assert!((config.col_tolerance - 0.3).abs() < 1e-6);
}

#[test]
fn test_csv_escape_with_comma() {
    let escaped = csv_escape("hello,world");
    assert_eq!(escaped, "\"hello,world\"");
}

#[test]
fn test_csv_escape_with_quotes() {
    let escaped = csv_escape("say \"hello\"");
    assert_eq!(escaped, "\"say \"\"hello\"\"\"");
}

#[test]
fn test_csv_escape_plain() {
    let escaped = csv_escape("plain");
    assert_eq!(escaped, "plain");
}
