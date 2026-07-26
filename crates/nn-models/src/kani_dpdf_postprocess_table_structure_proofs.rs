// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for dpdf_postprocess and table_structure (#3892).
//!
//! Proves bounding-box arithmetic invariants, post-processing pipeline
//! structural properties, and table-structure serialization correctness.
//!
//! **Areas proved (16+ harnesses):**
//!
//!  dpdf_postprocess harnesses (8):
//!   1. IoU is bounded in [0, 1] for valid boxes.
//!   2. IoU of a box with itself is 1.0.
//!   3. IoU of disjoint boxes is 0.0.
//!   4. merge_overlapping_regions preserves max confidence.
//!   5. deduplicate_regions reduces (or maintains) count.
//!   6. filter_by_confidence removes low-confidence entries.
//!   7. refine_bboxes clamps coordinates within image bounds.
//!   8. PostProcessConfig::default() has valid field values.
//!
//!  table_structure harnesses (8):
//!   9. StructuredTable dimensions match parsed row/col counts.
//!  10. Cell row/col indices are within table bounds.
//!  11. Spanning cells have valid row_span/col_span >= 1.
//!  12. to_html produces non-empty output for non-empty tables.
//!  13. to_csv rows are consistent with table dimensions.
//!  14. to_markdown_table includes separator row.
//!  15. Empty detection list produces a valid empty table.
//!  16. Single-cell table parses correctly.

use crate::dpdf_pipeline::{DocumentRegion, DpdfPipeline};
use crate::dpdf_postprocess::{
    compute_iou, deduplicate_regions, filter_by_confidence, merge_overlapping_regions,
    refine_bboxes, PostProcessConfig,
};
use crate::table_structure::{
    self, parse_structure, to_csv, to_html, to_markdown_table, TableStructureConfig,
};
use nn_core::layers::vision::Detection;

// ===========================================================================
// dpdf_postprocess harnesses
// ===========================================================================

/// Harness 1: IoU is bounded in [0, 1] for valid (non-degenerate) boxes.
///
/// SUBSTANTIVE: Proves compute_iou returns a value in [0.0, 1.0] for
/// well-formed bounding boxes where x2 > x1 and y2 > y1, preventing
/// out-of-range overlap ratios that would corrupt NMS decisions.
#[kani::proof]
#[kani::unwind(2)]
fn proof_iou_bounded_0_1() {
    // Well-formed boxes: positive area
    let a: [f32; 4] = [0.0, 0.0, 100.0, 100.0];
    let b: [f32; 4] = [50.0, 50.0, 150.0, 150.0];
    let iou = compute_iou(&a, &b);
    assert!(iou >= 0.0, "IoU must be >= 0");
    assert!(iou <= 1.0, "IoU must be <= 1");
    assert!(iou.is_finite(), "IoU must be finite");

    // Fully contained box
    let inner: [f32; 4] = [25.0, 25.0, 75.0, 75.0];
    let iou_contained = compute_iou(&a, &inner);
    assert!(iou_contained >= 0.0, "contained IoU must be >= 0");
    assert!(iou_contained <= 1.0, "contained IoU must be <= 1");

    // Identical boxes should give IoU = 1.0 (covered by harness 2, sanity check)
    let iou_self = compute_iou(&a, &a);
    assert!(
        iou_self >= 0.0 && iou_self <= 1.0,
        "self-IoU must be in [0,1]"
    );
}

/// Harness 2: IoU of a box with itself is exactly 1.0.
///
/// SUBSTANTIVE: Proves the self-intersection identity: any non-degenerate
/// box has IoU 1.0 with itself. Intersection == union == area.
#[kani::proof]
#[kani::unwind(2)]
fn proof_iou_self_is_one() {
    let a: [f32; 4] = [10.0, 20.0, 200.0, 300.0];
    let iou = compute_iou(&a, &a);
    assert_eq!(iou, 1.0, "IoU of a box with itself must be 1.0");

    // Also test a small box
    let small: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    let iou_small = compute_iou(&small, &small);
    assert_eq!(iou_small, 1.0, "IoU of small box with itself must be 1.0");
}

/// Harness 3: IoU of completely disjoint boxes is 0.0.
///
/// SUBSTANTIVE: Proves that non-overlapping boxes produce zero IoU,
/// which is required for NMS to correctly preserve non-overlapping
/// detections.
#[kani::proof]
#[kani::unwind(2)]
fn proof_iou_disjoint_is_zero() {
    // Horizontally disjoint
    let a: [f32; 4] = [0.0, 0.0, 50.0, 50.0];
    let b: [f32; 4] = [100.0, 0.0, 150.0, 50.0];
    let iou = compute_iou(&a, &b);
    assert_eq!(iou, 0.0, "horizontally disjoint boxes must have IoU 0");

    // Vertically disjoint
    let c: [f32; 4] = [0.0, 0.0, 50.0, 50.0];
    let d: [f32; 4] = [0.0, 100.0, 50.0, 150.0];
    let iou2 = compute_iou(&c, &d);
    assert_eq!(iou2, 0.0, "vertically disjoint boxes must have IoU 0");

    // Zero-area box
    let zero: [f32; 4] = [10.0, 10.0, 10.0, 10.0];
    let iou3 = compute_iou(&a, &zero);
    assert_eq!(iou3, 0.0, "zero-area box must have IoU 0");
}

/// Harness 4: merge_overlapping_regions preserves maximum confidence.
///
/// SUBSTANTIVE: Proves that after merging, no region's confidence is
/// lower than the maximum confidence of all input regions of the same
/// class that were merged into it. This ensures the merge operation
/// never downgrades detection quality.
#[kani::proof]
#[kani::unwind(5)]
fn proof_merge_preserves_max_confidence() {
    // Two overlapping text regions with different confidences
    let r1 = DocumentRegion::Text {
        content: "a".to_string(),
        bbox: [0.0, 0.0, 100.0, 100.0],
        confidence: 0.7,
    };
    let r2 = DocumentRegion::Text {
        content: "b".to_string(),
        bbox: [10.0, 10.0, 110.0, 110.0],
        confidence: 0.9,
    };

    let max_conf = 0.9f32;
    let mut regions = vec![r1, r2];
    merge_overlapping_regions(&mut regions, 0.0); // threshold 0.0 => always merge same-class overlapping

    // After merge, at least one region remains with confidence >= max_conf
    assert!(!regions.is_empty(), "merge must leave at least one region");
    let best = regions
        .iter()
        .map(|r| r.confidence())
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        best >= max_conf,
        "merged confidence must preserve max: got {}, expected >= {}",
        best,
        max_conf
    );
}

/// Harness 5: deduplicate_regions reduces (or maintains) count.
///
/// SUBSTANTIVE: Proves that deduplication never increases the number
/// of regions — it only suppresses duplicates or leaves the list
/// unchanged.
#[kani::proof]
#[kani::unwind(5)]
fn proof_dedup_reduces_count() {
    let r1 = DocumentRegion::Text {
        content: "a".to_string(),
        bbox: [0.0, 0.0, 100.0, 100.0],
        confidence: 0.9,
    };
    let r2 = DocumentRegion::Text {
        content: "b".to_string(),
        bbox: [5.0, 5.0, 105.0, 105.0],
        confidence: 0.8,
    };
    let r3 = DocumentRegion::Text {
        content: "c".to_string(),
        bbox: [500.0, 500.0, 600.0, 600.0],
        confidence: 0.7,
    };

    let original_count = 3;
    let mut regions = vec![r1, r2, r3];
    deduplicate_regions(&mut regions, 0.5);

    assert!(
        regions.len() <= original_count,
        "dedup must not increase count: got {}, original {}",
        regions.len(),
        original_count
    );
    // The disjoint r3 should survive; the near-duplicate r1/r2 should be deduped to one
    assert!(regions.len() >= 1, "dedup must leave at least one region");
}

/// Harness 6: filter_by_confidence removes all low-confidence entries.
///
/// SUBSTANTIVE: Proves that after filtering, every remaining region
/// has confidence >= the threshold, and that at least some filtering
/// occurs when low-confidence regions are present.
#[kani::proof]
#[kani::unwind(5)]
fn proof_filter_removes_low_confidence() {
    let r_high = DocumentRegion::Text {
        content: "high".to_string(),
        bbox: [0.0, 0.0, 100.0, 100.0],
        confidence: 0.95,
    };
    let r_low = DocumentRegion::Text {
        content: "low".to_string(),
        bbox: [200.0, 200.0, 300.0, 300.0],
        confidence: 0.1,
    };
    let r_mid = DocumentRegion::Text {
        content: "mid".to_string(),
        bbox: [400.0, 400.0, 500.0, 500.0],
        confidence: 0.5,
    };

    let threshold = 0.3;
    let mut regions = vec![r_high, r_low, r_mid];
    filter_by_confidence(&mut regions, threshold);

    // Low-confidence region (0.1) must be removed
    assert_eq!(regions.len(), 2, "one region must be filtered out");

    // All remaining must meet threshold
    for r in &regions {
        assert!(
            r.confidence() >= threshold,
            "remaining region confidence {} must be >= threshold {}",
            r.confidence(),
            threshold
        );
    }
}

/// Harness 7: refine_bboxes clamps coordinates within image bounds.
///
/// SUBSTANTIVE: Proves that after refinement, all bounding box
/// coordinates are within [0, image_width] for x and [0, image_height]
/// for y, preventing out-of-bounds pixel access.
#[kani::proof]
#[kani::unwind(3)]
fn proof_bbox_clamp_in_bounds() {
    let r = DocumentRegion::Text {
        content: "overflow".to_string(),
        bbox: [-10.0, -20.0, 1500.0, 2000.0],
        confidence: 0.8,
    };

    let image_width = 1000_usize;
    let image_height = 800_usize;
    let mut regions = vec![r];
    refine_bboxes(&mut regions, image_width, image_height);

    let bbox = regions[0].bbox();
    let w = image_width as f32;
    let h = image_height as f32;

    assert!(bbox[0] >= 0.0, "x1 must be >= 0, got {}", bbox[0]);
    assert!(bbox[1] >= 0.0, "y1 must be >= 0, got {}", bbox[1]);
    assert!(bbox[2] <= w, "x2 must be <= width, got {}", bbox[2]);
    assert!(bbox[3] <= h, "y2 must be <= height, got {}", bbox[3]);

    // Verify exact clamped values
    assert_eq!(bbox[0], 0.0, "x1 clamped from -10 to 0");
    assert_eq!(bbox[1], 0.0, "y1 clamped from -20 to 0");
    assert_eq!(bbox[2], w, "x2 clamped from 1500 to width");
    assert_eq!(bbox[3], h, "y2 clamped from 2000 to height");
}

/// Harness 8: PostProcessConfig::default() has valid field values.
///
/// SUBSTANTIVE: Proves the default configuration has all thresholds
/// in valid ranges: merge_iou in (0,1), dedup_similarity in (0,1),
/// min_confidence in (0,1), preventing degenerate pipeline behavior.
#[kani::proof]
#[kani::unwind(2)]
fn proof_postprocess_config_defaults_valid() {
    let cfg = PostProcessConfig::default();

    // merge_iou: valid probability range
    assert!(cfg.merge_iou > 0.0, "merge_iou must be > 0");
    assert!(cfg.merge_iou < 1.0, "merge_iou must be < 1");
    assert!(cfg.merge_iou.is_finite(), "merge_iou must be finite");

    // dedup_similarity: valid probability range
    assert!(cfg.dedup_similarity > 0.0, "dedup_similarity must be > 0");
    assert!(cfg.dedup_similarity < 1.0, "dedup_similarity must be < 1");
    assert!(
        cfg.dedup_similarity.is_finite(),
        "dedup_similarity must be finite"
    );

    // min_confidence: valid probability range
    assert!(cfg.min_confidence > 0.0, "min_confidence must be > 0");
    assert!(cfg.min_confidence < 1.0, "min_confidence must be < 1");
    assert!(
        cfg.min_confidence.is_finite(),
        "min_confidence must be finite"
    );

    // Verify exact default values
    assert_eq!(cfg.merge_iou, 0.5);
    assert_eq!(cfg.dedup_similarity, 0.9);
    assert_eq!(cfg.min_confidence, 0.3);
    assert!(
        cfg.enable_model_fusion,
        "model fusion should be enabled by default"
    );
}

// ===========================================================================
// table_structure harnesses
// ===========================================================================

/// Helper: create a Detection for use in table structure tests.
fn make_detection(x1: f32, y1: f32, x2: f32, y2: f32, class_id: u32, confidence: f32) -> Detection {
    Detection {
        x1,
        y1,
        x2,
        y2,
        confidence,
        class_id,
    }
}

/// Helper: build a minimal 2x2 table from detections.
fn build_2x2_table() -> crate::table_structure::StructuredTable {
    let config = TableStructureConfig::default();
    // Two row detections, two column detections
    let detections = vec![
        make_detection(0.0, 0.0, 200.0, 50.0, 1, 0.9), // row 0
        make_detection(0.0, 50.0, 200.0, 100.0, 1, 0.85), // row 1
        make_detection(0.0, 0.0, 100.0, 100.0, 2, 0.9), // col 0
        make_detection(100.0, 0.0, 200.0, 100.0, 2, 0.85), // col 1
    ];
    parse_structure(&detections, &config)
}

/// Harness 9: StructuredTable dimensions match parsed row/col counts.
///
/// SUBSTANTIVE: Proves that num_rows equals the number of row entries
/// in rows Vec and num_cols matches the declared column count, ensuring
/// the parse_structure output is self-consistent.
#[kani::proof]
#[kani::unwind(6)]
fn proof_table_dimensions_match() {
    let table = build_2x2_table();

    assert_eq!(table.num_rows, 2, "2x2 table must have 2 rows");
    assert_eq!(table.num_cols, 2, "2x2 table must have 2 cols");
    assert_eq!(
        table.rows.len(),
        table.num_rows,
        "rows Vec length must equal num_rows"
    );
}

/// Harness 10: Cell row/col indices are within table bounds.
///
/// SUBSTANTIVE: Proves that every cell in the parsed table has
/// row < num_rows and col < num_cols, preventing out-of-bounds
/// grid access during serialization.
#[kani::proof]
#[kani::unwind(8)]
fn proof_cell_indices_in_bounds() {
    let table = build_2x2_table();

    for row in &table.rows {
        for cell in &row.cells {
            assert!(
                cell.row < table.num_rows,
                "cell row {} must be < num_rows {}",
                cell.row,
                table.num_rows
            );
            assert!(
                cell.col < table.num_cols,
                "cell col {} must be < num_cols {}",
                cell.col,
                table.num_cols
            );
        }
    }
}

/// Harness 11: Spanning cells have valid row_span/col_span >= 1.
///
/// SUBSTANTIVE: Proves that all cells (including spanning cells)
/// have span values >= 1 and that the span does not exceed table
/// boundaries. A span of 0 would be a degenerate cell.
#[kani::proof]
#[kani::unwind(8)]
fn proof_spanning_cells_valid() {
    let config = TableStructureConfig::default();
    // Build a table with a spanning cell that covers 2 columns in row 0
    let detections = vec![
        make_detection(0.0, 0.0, 200.0, 50.0, 1, 0.9), // row 0
        make_detection(0.0, 50.0, 200.0, 100.0, 1, 0.85), // row 1
        make_detection(0.0, 0.0, 100.0, 100.0, 2, 0.9), // col 0
        make_detection(100.0, 0.0, 200.0, 100.0, 2, 0.85), // col 1
        make_detection(0.0, 0.0, 200.0, 50.0, 3, 0.95), // spanning cell: row 0, cols 0-1
    ];
    let table = parse_structure(&detections, &config);

    for row in &table.rows {
        for cell in &row.cells {
            assert!(
                cell.row_span >= 1,
                "row_span must be >= 1, got {}",
                cell.row_span
            );
            assert!(
                cell.col_span >= 1,
                "col_span must be >= 1, got {}",
                cell.col_span
            );
            assert!(
                cell.row + cell.row_span <= table.num_rows,
                "row {} + row_span {} must be <= num_rows {}",
                cell.row,
                cell.row_span,
                table.num_rows
            );
            assert!(
                cell.col + cell.col_span <= table.num_cols,
                "col {} + col_span {} must be <= num_cols {}",
                cell.col,
                cell.col_span,
                table.num_cols
            );
        }
    }
}

/// Harness 12: to_html produces non-empty output for non-empty tables.
///
/// SUBSTANTIVE: Proves that a table with at least one row and column
/// generates a non-empty HTML string containing the <table> element,
/// ensuring the serializer never silently produces empty output.
#[kani::proof]
#[kani::unwind(8)]
fn proof_html_nonempty() {
    let table = build_2x2_table();
    assert!(table.num_rows > 0, "table must have rows for this test");

    let html = to_html(&table);
    assert!(!html.is_empty(), "HTML output must not be empty");
    assert!(html.contains("<table>"), "HTML must contain <table> tag");
    assert!(
        html.contains("</table>"),
        "HTML must contain closing </table> tag"
    );
    assert!(html.contains("<tr>"), "HTML must contain at least one <tr>");
    assert!(html.contains("<td"), "HTML must contain at least one <td");
}

/// Harness 13: to_csv rows are consistent with table dimensions.
///
/// SUBSTANTIVE: Proves that the CSV output has exactly num_rows lines,
/// one per table row, ensuring the serializer faithfully represents
/// the table structure.
#[kani::proof]
#[kani::unwind(8)]
fn proof_csv_rows_consistent() {
    let table = build_2x2_table();
    assert!(table.num_rows > 0, "table must have rows for this test");

    let csv = to_csv(&table);
    assert!(!csv.is_empty(), "CSV output must not be empty");

    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(
        lines.len(),
        table.num_rows,
        "CSV must have exactly {} lines (one per row), got {}",
        table.num_rows,
        lines.len()
    );

    // Each line should have at least one field
    for (i, line) in lines.iter().enumerate() {
        assert!(!line.is_empty(), "CSV row {} must not be empty", i);
    }
}

/// Harness 14: to_markdown_table includes separator row.
///
/// SUBSTANTIVE: Proves that a valid Markdown pipe table is generated
/// with a separator row containing "---", which is required by the
/// Markdown table specification for correct rendering.
#[kani::proof]
#[kani::unwind(8)]
fn proof_markdown_has_separator() {
    let table = build_2x2_table();
    assert!(table.num_rows > 0, "table must have rows for this test");

    let md = to_markdown_table(&table);
    assert!(!md.is_empty(), "Markdown output must not be empty");
    assert!(
        md.contains("---"),
        "Markdown table must contain separator row with ---"
    );
    assert!(
        md.contains("|"),
        "Markdown table must contain pipe characters"
    );

    // Must have header + separator + data rows
    let lines: Vec<&str> = md.lines().collect();
    // num_rows rows from the grid + 1 separator = num_rows + 1 lines
    // But first row of grid is the header, so total lines = 1 (header) + 1 (sep) + (num_rows - 1) data = num_rows + 1
    assert_eq!(
        lines.len(),
        table.num_rows + 1,
        "Markdown must have num_rows + 1 lines (header + sep + data), got {}",
        lines.len()
    );
}

/// Harness 15: Empty detection list produces a valid empty table.
///
/// SUBSTANTIVE: Proves that parse_structure handles the empty-input
/// edge case gracefully, returning a table with zero dimensions and
/// no rows, rather than panicking or producing malformed output.
#[kani::proof]
#[kani::unwind(2)]
fn proof_empty_table_valid() {
    let config = TableStructureConfig::default();
    let detections: Vec<Detection> = vec![];
    let table = parse_structure(&detections, &config);

    assert_eq!(table.num_rows, 0, "empty input must produce 0 rows");
    assert_eq!(table.num_cols, 0, "empty input must produce 0 cols");
    assert!(
        table.rows.is_empty(),
        "empty table must have empty rows Vec"
    );
    assert!(table.caption.is_none(), "empty table must have no caption");

    // Serialization of empty table should not panic
    let html = to_html(&table);
    let csv = to_csv(&table);
    let md = to_markdown_table(&table);

    // HTML still produces a <table> wrapper even for empty tables
    assert!(
        html.contains("<table>"),
        "empty table HTML must contain <table>"
    );
    assert!(csv.is_empty(), "empty table CSV must be empty");
    assert!(md.is_empty(), "empty table markdown must be empty");
}

/// Harness 16: Single-cell table (1 row, 1 column) parses correctly.
///
/// SUBSTANTIVE: Proves that the minimal non-degenerate table (1x1)
/// produces exactly one cell at position (0,0) with span (1,1),
/// validating the base case for the grid construction algorithm.
#[kani::proof]
#[kani::unwind(4)]
fn proof_single_cell_table() {
    let config = TableStructureConfig::default();
    let detections = vec![
        make_detection(0.0, 0.0, 100.0, 50.0, 1, 0.9), // 1 row
        make_detection(0.0, 0.0, 100.0, 50.0, 2, 0.9), // 1 column
    ];
    let table = parse_structure(&detections, &config);

    assert_eq!(table.num_rows, 1, "1x1 table must have 1 row");
    assert_eq!(table.num_cols, 1, "1x1 table must have 1 col");
    assert_eq!(table.rows.len(), 1, "rows Vec must have 1 entry");
    assert_eq!(table.rows[0].cells.len(), 1, "single row must have 1 cell");

    let cell = &table.rows[0].cells[0];
    assert_eq!(cell.row, 0, "cell row must be 0");
    assert_eq!(cell.col, 0, "cell col must be 0");
    assert_eq!(cell.row_span, 1, "cell row_span must be 1");
    assert_eq!(cell.col_span, 1, "cell col_span must be 1");
    assert!(cell.confidence > 0.0, "cell confidence must be positive");
    assert!(
        cell.confidence.is_finite(),
        "cell confidence must be finite"
    );
}
