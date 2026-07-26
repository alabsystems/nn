// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration tests for the dpdf document inference pipeline.
//!
//! All tests use synthetic data (mock detections and regions) -- no external
//! weight files needed.
//!
//! Part of #3896.

use nn_core::layers::vision::Detection;
use nn_models::dpdf_pipeline::{DocumentRegion, DpdfPipeline, PipelineConfig};
use nn_models::dpdf_postprocess::{
    compute_iou, deduplicate_regions, filter_by_confidence, fuse_model_results,
    merge_overlapping_regions, postprocess, refine_bboxes, PostProcessConfig,
};
use nn_models::table_structure::{
    parse_structure, to_csv, to_html, to_markdown_table, TableStructureConfig,
};

// ============================================================================
// Helpers
// ============================================================================

/// Build a text region with content.
fn text_region(content: &str, bbox: [f32; 4], confidence: f32) -> DocumentRegion {
    DocumentRegion::Text {
        content: content.to_string(),
        bbox,
        confidence,
    }
}

/// Build a section header region.
fn section_header(content: &str, bbox: [f32; 4], confidence: f32) -> DocumentRegion {
    DocumentRegion::SectionHeader {
        content: content.to_string(),
        bbox,
        confidence,
    }
}

/// Build a table region with cell data.
fn table_region(cells: Vec<Vec<String>>, bbox: [f32; 4], confidence: f32) -> DocumentRegion {
    DocumentRegion::Table {
        cells,
        bbox,
        confidence,
    }
}

/// Build a figure region.
fn figure_region(caption: Option<&str>, bbox: [f32; 4], confidence: f32) -> DocumentRegion {
    DocumentRegion::Figure {
        caption: caption.map(ToString::to_string),
        bbox,
        confidence,
    }
}

// ============================================================================
// 1. Single page, single text region
// ============================================================================

#[test]
fn test_single_page_single_text_region() {
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let regions = vec![text_region("Hello world", [10.0, 20.0, 300.0, 60.0], 0.95)];
    let page = pipeline.build_page(regions, 612, 792);

    assert_eq!(page.regions.len(), 1);
    assert_eq!(page.reading_order.len(), 1);
    assert_eq!(page.reading_order[0], 0);
    assert_eq!(page.width, 612);
    assert_eq!(page.height, 792);

    let text = DpdfPipeline::extract_text(&page);
    assert_eq!(text, "Hello world");
}

// ============================================================================
// 2. Multi-region page (text, title, figure, table)
// ============================================================================

#[test]
fn test_multi_region_page_text_title_figure_table() {
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let regions = vec![
        section_header("Introduction", [10.0, 10.0, 300.0, 40.0], 0.9),
        text_region("Body text here.", [10.0, 50.0, 300.0, 120.0], 0.88),
        figure_region(Some("Figure 1"), [10.0, 130.0, 300.0, 250.0], 0.85),
        table_region(
            vec![
                vec!["Col A".into(), "Col B".into()],
                vec!["1".into(), "2".into()],
            ],
            [10.0, 260.0, 300.0, 350.0],
            0.92,
        ),
    ];
    let page = pipeline.build_page(regions, 612, 792);

    assert_eq!(page.regions.len(), 4);
    // All regions are in vertical order, so reading order should be 0,1,2,3.
    assert_eq!(page.reading_order, vec![0, 1, 2, 3]);

    let text = DpdfPipeline::extract_text(&page);
    assert!(text.contains("Introduction"));
    assert!(text.contains("Body text here."));
    assert!(text.contains("Figure 1"));
    assert!(text.contains("1\t2")); // table row
}

// ============================================================================
// 3. Table region triggers table structure parsing
// ============================================================================

#[test]
fn test_table_region_triggers_structure_parsing() {
    // Create Table Transformer-style detections for a 2x3 table.
    let detections = vec![
        // 2 rows
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 300.0,
            y2: 50.0,
            confidence: 0.9,
            class_id: 1,
        },
        Detection {
            x1: 0.0,
            y1: 50.0,
            x2: 300.0,
            y2: 100.0,
            confidence: 0.88,
            class_id: 1,
        },
        // 3 columns
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
            confidence: 0.85,
            class_id: 2,
        },
        Detection {
            x1: 100.0,
            y1: 0.0,
            x2: 200.0,
            y2: 100.0,
            confidence: 0.87,
            class_id: 2,
        },
        Detection {
            x1: 200.0,
            y1: 0.0,
            x2: 300.0,
            y2: 100.0,
            confidence: 0.86,
            class_id: 2,
        },
    ];

    let config = TableStructureConfig::default();
    let table = parse_structure(&detections, &config);

    assert_eq!(table.num_rows, 2);
    assert_eq!(table.num_cols, 3);
    assert_eq!(table.rows.len(), 2);

    // Each row should have 3 cells.
    for row in &table.rows {
        assert_eq!(row.cells.len(), 3);
    }
}

// ============================================================================
// 4. Post-processing removes overlapping duplicate regions
// ============================================================================

#[test]
fn test_postprocess_removes_overlapping_duplicates() {
    // Two nearly identical text regions (IoU > 0.9).
    let mut regions = vec![
        text_region("Hello", [10.0, 10.0, 200.0, 50.0], 0.90),
        text_region("Hello", [11.0, 11.0, 201.0, 51.0], 0.85),
    ];

    let config = PostProcessConfig {
        merge_iou: 0.5,
        dedup_similarity: 0.5,
        min_confidence: 0.3,
        enable_model_fusion: false,
    };
    postprocess(&mut regions, &config);

    // The duplicate should be removed (merged or deduplicated).
    assert_eq!(regions.len(), 1);
    // Higher confidence region (or merged) should survive.
    assert!(regions[0].confidence() >= 0.85);
}

// ============================================================================
// 5. Reading order correct for multi-column layout
// ============================================================================

#[test]
fn test_reading_order_multicolumn_layout() {
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    // Simulate a 2-column layout:
    // Top-left text, top-right text, bottom-left text, bottom-right text.
    let regions = vec![
        text_region("Bottom-right", [320.0, 400.0, 600.0, 500.0], 0.9), // idx 0
        text_region("Top-left", [10.0, 10.0, 290.0, 100.0], 0.9),       // idx 1
        text_region("Top-right", [320.0, 10.0, 600.0, 100.0], 0.9),     // idx 2
        text_region("Bottom-left", [10.0, 400.0, 290.0, 500.0], 0.9),   // idx 3
    ];
    let page = pipeline.build_page(regions, 612, 792);

    // Reading order: top-to-bottom then left-to-right.
    // Top row: idx 1 (left, mid_y=55) before idx 2 (right, mid_y=55).
    // Bottom row: idx 3 (left, mid_y=450) before idx 0 (right, mid_y=450).
    assert_eq!(page.reading_order, vec![1, 2, 3, 0]);

    let text = DpdfPipeline::extract_text(&page);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "Top-left");
    assert_eq!(lines[1], "Top-right");
    assert_eq!(lines[2], "Bottom-left");
    assert_eq!(lines[3], "Bottom-right");
}

// ============================================================================
// 6. Markdown export includes all region types
// ============================================================================

#[test]
fn test_markdown_export_all_region_types() {
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let regions = vec![
        section_header("Nn Section", [10.0, 10.0, 300.0, 40.0], 0.9),
        text_region("Some text.", [10.0, 50.0, 300.0, 90.0], 0.9),
        DocumentRegion::ListItem {
            content: "Item one".to_string(),
            bbox: [10.0, 100.0, 300.0, 120.0],
            confidence: 0.9,
        },
        DocumentRegion::Formula {
            latex: Some("E = mc^2".to_string()),
            bbox: [10.0, 130.0, 300.0, 160.0],
            confidence: 0.9,
        },
        figure_region(Some("Chart"), [10.0, 170.0, 300.0, 250.0], 0.9),
        table_region(
            vec![vec!["A".into(), "B".into()], vec!["1".into(), "2".into()]],
            [10.0, 260.0, 300.0, 340.0],
            0.9,
        ),
        DocumentRegion::Caption {
            content: "Table 1".to_string(),
            bbox: [10.0, 350.0, 300.0, 370.0],
            confidence: 0.9,
        },
        DocumentRegion::Footnote {
            content: "See appendix.".to_string(),
            bbox: [10.0, 700.0, 300.0, 720.0],
            confidence: 0.9,
        },
    ];
    let page = pipeline.build_page(regions, 612, 792);
    let md = DpdfPipeline::to_markdown(&page);

    assert!(md.contains("## Nn Section"), "should have markdown heading");
    assert!(md.contains("Some text."), "should have text content");
    assert!(md.contains("- Item one"), "should have list item");
    assert!(md.contains("$E = mc^2$"), "should have formula");
    assert!(md.contains("![Chart]()"), "should have figure");
    assert!(md.contains("| A | B |"), "should have table header");
    assert!(md.contains("| 1 | 2 |"), "should have table data");
    assert!(md.contains("Table 1"), "should have caption");
    assert!(md.contains("See appendix."), "should have footnote");
}

// ============================================================================
// 7. HTML table export with cell structure
// ============================================================================

#[test]
fn test_html_table_export_cell_structure() {
    // Build a 2x2 table with parse_structure.
    let detections = vec![
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 200.0,
            y2: 50.0,
            confidence: 0.9,
            class_id: 1,
        },
        Detection {
            x1: 0.0,
            y1: 50.0,
            x2: 200.0,
            y2: 100.0,
            confidence: 0.9,
            class_id: 1,
        },
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
            confidence: 0.9,
            class_id: 2,
        },
        Detection {
            x1: 100.0,
            y1: 0.0,
            x2: 200.0,
            y2: 100.0,
            confidence: 0.9,
            class_id: 2,
        },
    ];
    let table = parse_structure(&detections, &TableStructureConfig::default());

    let html = to_html(&table);
    assert!(html.starts_with("<table>"), "should start with <table>");
    assert!(html.ends_with("</table>"), "should end with </table>");
    assert!(html.contains("<tr>"), "should contain table rows");
    assert!(html.contains("<td>"), "should contain table cells");

    // Also test markdown and CSV serialization.
    let md = to_markdown_table(&table);
    assert!(
        md.contains("|"),
        "markdown table should have pipe separators"
    );
    assert!(
        md.contains("---"),
        "markdown table should have separator row"
    );

    let csv = to_csv(&table);
    assert!(!csv.is_empty(), "CSV output should not be empty");
}

// ============================================================================
// 8. Multiple pages processed
// ============================================================================

#[test]
fn test_multiple_pages_processed() {
    let pipeline = DpdfPipeline::new(PipelineConfig::default());

    let page1_dets: Vec<(usize, f32, [f32; 4])> = vec![
        (9, 0.95, [10.0, 10.0, 300.0, 50.0]), // text
        (7, 0.90, [10.0, 60.0, 300.0, 90.0]), // section-header
    ];
    let page2_dets: Vec<(usize, f32, [f32; 4])> = vec![
        (8, 0.88, [10.0, 10.0, 300.0, 200.0]), // table
    ];
    let page3_dets: Vec<(usize, f32, [f32; 4])> = vec![
        (6, 0.85, [10.0, 10.0, 400.0, 300.0]),  // picture
        (0, 0.80, [10.0, 310.0, 400.0, 340.0]), // caption
    ];

    let pages_input: Vec<(&[(usize, f32, [f32; 4])], usize, usize)> = vec![
        (&page1_dets, 612, 792),
        (&page2_dets, 612, 792),
        (&page3_dets, 800, 600),
    ];
    let doc = pipeline.process_pages(&pages_input);

    assert_eq!(doc.pages.len(), 3);
    assert_eq!(doc.pages[0].regions.len(), 2);
    assert_eq!(doc.pages[1].regions.len(), 1);
    assert_eq!(doc.pages[2].regions.len(), 2);

    // Page dimensions preserved.
    assert_eq!(doc.pages[2].width, 800);
    assert_eq!(doc.pages[2].height, 600);
}

// ============================================================================
// 9. Empty page produces empty output
// ============================================================================

#[test]
fn test_empty_page_empty_output() {
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let page = pipeline.build_page(vec![], 612, 792);

    assert!(page.regions.is_empty());
    assert!(page.reading_order.is_empty());

    let text = DpdfPipeline::extract_text(&page);
    assert!(text.is_empty());

    let md = DpdfPipeline::to_markdown(&page);
    assert!(md.is_empty());
}

// ============================================================================
// 10. All 10 DocumentRegion variants
// ============================================================================

#[test]
fn test_all_ten_document_region_variants() {
    // Class IDs 0..=9 map to 10 distinct DocumentRegion variants.
    let class_names_expected = [
        "caption",        // 0
        "footnote",       // 1
        "formula",        // 2
        "list-item",      // 3
        "page-footer",    // 4
        "page-header",    // 5
        "picture",        // 6
        "section-header", // 7
        "table",          // 8
        "text",           // 9
    ];

    for (class_id, expected_name) in class_names_expected.iter().enumerate() {
        let region = DpdfPipeline::classify_detection(class_id, [0.0, 0.0, 100.0, 50.0], 0.9);
        assert_eq!(
            region.class_name(),
            *expected_name,
            "class_id {class_id} should produce {expected_name}"
        );
        assert_eq!(region.confidence(), 0.9);
        assert_eq!(region.bbox(), [0.0, 0.0, 100.0, 50.0]);
    }

    // Verify that detections_to_regions handles all 10 at once.
    let detections: Vec<(usize, f32, [f32; 4])> = (0..10)
        .map(|id| {
            (
                id,
                0.8,
                [0.0, (id as f32) * 10.0, 100.0, (id as f32) * 10.0 + 9.0],
            )
        })
        .collect();
    let regions = DpdfPipeline::detections_to_regions(&detections);
    assert_eq!(regions.len(), 10);

    // Verify uniqueness of class names.
    let mut names: Vec<&str> = regions.iter().map(DocumentRegion::class_name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names.len(),
        10,
        "all 10 variants should have distinct class names"
    );
}

// ============================================================================
// 11. Confidence filtering removes low-quality detections
// ============================================================================

#[test]
fn test_confidence_filtering_removes_low_quality() {
    let mut regions = vec![
        text_region("High conf", [10.0, 10.0, 200.0, 50.0], 0.95),
        text_region("Medium conf", [10.0, 60.0, 200.0, 100.0], 0.50),
        text_region("Low conf", [10.0, 110.0, 200.0, 150.0], 0.10),
        text_region("Border conf", [10.0, 160.0, 200.0, 200.0], 0.30),
    ];

    filter_by_confidence(&mut regions, 0.30);

    // Only regions with confidence >= 0.30 survive.
    assert_eq!(regions.len(), 3);
    assert!(regions.iter().all(|r| r.confidence() >= 0.30));

    // Verify the low-confidence one was removed.
    let confs: Vec<f32> = regions.iter().map(DocumentRegion::confidence).collect();
    assert!(!confs.contains(&0.10));
}

// ============================================================================
// 12. NMS deduplication works end-to-end
// ============================================================================

#[test]
fn test_nms_deduplication_end_to_end() {
    // Three text regions: two are near-identical (high IoU), one is separate.
    let mut regions = vec![
        text_region("A", [10.0, 10.0, 200.0, 80.0], 0.92),
        text_region("A dup", [12.0, 12.0, 202.0, 82.0], 0.88), // overlaps heavily with first
        text_region("B", [10.0, 300.0, 200.0, 380.0], 0.85),   // separate region
    ];

    // Verify that the two overlapping boxes have high IoU.
    let iou = compute_iou(&[10.0, 10.0, 200.0, 80.0], &[12.0, 12.0, 202.0, 82.0]);
    assert!(
        iou > 0.8,
        "overlapping regions should have high IoU, got {iou}"
    );

    deduplicate_regions(&mut regions, 0.5);

    // After deduplication, one of the overlapping pair should be removed.
    assert_eq!(regions.len(), 2);
    // The higher-confidence region should survive.
    assert!(regions[0].confidence() >= 0.92 || regions[1].confidence() >= 0.92);
}

// ============================================================================
// 13. Pipeline config validation
// ============================================================================

#[test]
fn test_pipeline_config_default_values() {
    let config = PipelineConfig::default();
    assert!((config.layout_conf_threshold - 0.25).abs() < f32::EPSILON);
    assert!((config.layout_iou_threshold - 0.45).abs() < f32::EPSILON);
    assert_eq!(config.ocr_max_tokens, 1024);
    assert!(config.enable_table_structure);
}

#[test]
fn test_pipeline_config_custom() {
    let config = PipelineConfig {
        layout_conf_threshold: 0.5,
        layout_iou_threshold: 0.6,
        ocr_max_tokens: 512,
        enable_table_structure: false,
        postprocess_config: PostProcessConfig::default(),
        table_structure_config: TableStructureConfig::default(),
    };
    let pipeline = DpdfPipeline::new(config);
    assert!((pipeline.config().layout_conf_threshold - 0.5).abs() < f32::EPSILON);
    assert!(!pipeline.config().enable_table_structure);
}

#[test]
fn test_postprocess_config_default_values() {
    let config = PostProcessConfig::default();
    assert!((config.merge_iou - 0.5).abs() < f32::EPSILON);
    assert!((config.dedup_similarity - 0.9).abs() < f32::EPSILON);
    assert!((config.min_confidence - 0.3).abs() < f32::EPSILON);
    assert!(config.enable_model_fusion);
}

// ============================================================================
// 14. Different model combinations (multi-model fusion)
// ============================================================================

#[test]
fn test_different_model_fusion_combinations() {
    // DocLayout detects structural elements.
    let doclayout = vec![
        section_header("Title", [10.0, 10.0, 300.0, 40.0], 0.92),
        text_region("Body", [10.0, 50.0, 300.0, 120.0], 0.90),
    ];

    // Table Transformer detects a table in a non-overlapping region.
    let table_det = vec![table_region(vec![], [10.0, 200.0, 300.0, 350.0], 0.88)];

    // OCR detects text in an overlapping region (should be suppressed)
    // plus a non-overlapping text region.
    let ocr = vec![
        text_region("OCR body", [12.0, 52.0, 298.0, 118.0], 0.70), // overlaps with doclayout body
        text_region("Footer text", [10.0, 700.0, 300.0, 730.0], 0.75), // non-overlapping
    ];

    let fused = fuse_model_results(&doclayout, &table_det, &ocr);

    // Doclayout regions (2) + table (1, non-overlapping) + footer (1, non-overlapping) = 4
    // OCR body is suppressed because it overlaps with doclayout body.
    assert_eq!(fused.len(), 4);

    let class_names: Vec<&str> = fused.iter().map(DocumentRegion::class_name).collect();
    assert!(class_names.contains(&"section-header"));
    assert!(class_names.contains(&"table"));
}

#[test]
fn test_fusion_empty_sources() {
    let fused = fuse_model_results(&[], &[], &[]);
    assert!(fused.is_empty());

    // Only doclayout, no table or OCR.
    let doclayout = vec![text_region("A", [10.0, 10.0, 100.0, 50.0], 0.9)];
    let fused = fuse_model_results(&doclayout, &[], &[]);
    assert_eq!(fused.len(), 1);
}

// ============================================================================
// 15. Large synthetic page handling
// ============================================================================

#[test]
fn test_large_synthetic_page_handling() {
    let pipeline = DpdfPipeline::new(PipelineConfig::default());

    // Generate 200 synthetic text regions across a large page.
    let mut regions = Vec::with_capacity(200);
    for i in 0..200 {
        let y = (i as f32) * 20.0;
        regions.push(text_region(
            &format!("Region {i}"),
            [10.0, y, 300.0, y + 18.0],
            0.5 + (i as f32) * 0.002, // confidence: 0.5 to ~0.9
        ));
    }

    let page = pipeline.build_page(regions, 4000, 4000);
    assert_eq!(page.regions.len(), 200);
    assert_eq!(page.reading_order.len(), 200);

    // Reading order should be top-to-bottom (ascending y).
    for i in 1..page.reading_order.len() {
        let prev = &page.regions[page.reading_order[i - 1]];
        let curr = &page.regions[page.reading_order[i]];
        let prev_mid_y = f32::midpoint(prev.bbox()[1], prev.bbox()[3]);
        let curr_mid_y = f32::midpoint(curr.bbox()[1], curr.bbox()[3]);
        assert!(
            prev_mid_y <= curr_mid_y,
            "reading order should be top-to-bottom: region {i}"
        );
    }

    // Markdown export should succeed and include all regions.
    let md = DpdfPipeline::to_markdown(&page);
    assert!(md.contains("Region 0"));
    assert!(md.contains("Region 199"));
}

// ============================================================================
// Additional integration tests
// ============================================================================

#[test]
fn test_page_header_footer_reading_order_priority() {
    // Page headers should come first and footers last, regardless of bbox position.
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let regions = vec![
        text_region("Middle text", [10.0, 200.0, 300.0, 250.0], 0.9), // idx 0
        DocumentRegion::PageFooter {
            // idx 1
            content: "Page 1".to_string(),
            bbox: [10.0, 50.0, 300.0, 70.0], // positioned near top visually!
            confidence: 0.85,
        },
        DocumentRegion::PageHeader {
            // idx 2
            content: "Header".to_string(),
            bbox: [10.0, 700.0, 300.0, 720.0], // positioned near bottom visually!
            confidence: 0.88,
        },
    ];
    let page = pipeline.build_page(regions, 612, 792);

    // Header (idx 2) should come first, footer (idx 1) should come last.
    assert_eq!(page.reading_order[0], 2, "header should come first");
    assert_eq!(
        *page.reading_order.last().unwrap(),
        1,
        "footer should come last"
    );
}

#[test]
fn test_merge_overlapping_same_class_regions() {
    // Two text regions overlapping significantly.
    let mut regions = vec![
        text_region("A", [10.0, 10.0, 200.0, 100.0], 0.90),
        text_region("B", [50.0, 30.0, 220.0, 110.0], 0.85),
    ];

    let iou = compute_iou(&[10.0, 10.0, 200.0, 100.0], &[50.0, 30.0, 220.0, 110.0]);
    // Only merge if IoU > threshold.
    merge_overlapping_regions(&mut regions, 0.1); // Low threshold to force merge.

    if iou > 0.1 {
        assert_eq!(
            regions.len(),
            1,
            "overlapping same-class regions should merge"
        );
        // Merged bbox should enclose both.
        let bbox = regions[0].bbox();
        assert!((bbox[0] - 10.0).abs() < f32::EPSILON);
        assert!((bbox[1] - 10.0).abs() < f32::EPSILON);
        assert!((bbox[2] - 220.0).abs() < f32::EPSILON);
        assert!((bbox[3] - 110.0).abs() < f32::EPSILON);
        // Merged confidence should be the max.
        assert!((regions[0].confidence() - 0.90).abs() < f32::EPSILON);
    }
}

#[test]
fn test_merge_does_not_merge_different_classes() {
    // A text and a figure overlapping should NOT be merged.
    let mut regions = vec![
        text_region("Text", [10.0, 10.0, 200.0, 100.0], 0.90),
        figure_region(Some("Fig"), [10.0, 10.0, 200.0, 100.0], 0.85),
    ];

    merge_overlapping_regions(&mut regions, 0.1);
    assert_eq!(regions.len(), 2, "different classes should not merge");
}

#[test]
fn test_refine_bboxes_clamps_to_image() {
    let mut regions = vec![text_region("Overflowing", [-10.0, -5.0, 700.0, 800.0], 0.9)];

    refine_bboxes(&mut regions, 612, 792);

    let bbox = regions[0].bbox();
    assert!((bbox[0] - 0.0).abs() < f32::EPSILON, "x1 clamped to 0");
    assert!((bbox[1] - 0.0).abs() < f32::EPSILON, "y1 clamped to 0");
    assert!(
        (bbox[2] - 612.0).abs() < f32::EPSILON,
        "x2 clamped to width"
    );
    assert!(
        (bbox[3] - 792.0).abs() < f32::EPSILON,
        "y2 clamped to height"
    );
}

#[test]
fn test_classify_detection_unknown_class_defaults_to_text() {
    // Class IDs outside 0..9 should default to Text.
    let region = DpdfPipeline::classify_detection(99, [0.0, 0.0, 50.0, 50.0], 0.7);
    assert_eq!(region.class_name(), "text");
}

#[test]
fn test_full_pipeline_postprocess_then_page() {
    let pipeline = DpdfPipeline::new(PipelineConfig::default());

    // Start with raw detections including duplicates and low-confidence noise.
    let detections: Vec<(usize, f32, [f32; 4])> = vec![
        (7, 0.92, [10.0, 10.0, 300.0, 40.0]),   // section-header
        (9, 0.88, [10.0, 50.0, 300.0, 120.0]),  // text
        (9, 0.15, [10.0, 130.0, 300.0, 160.0]), // text, low confidence
        (9, 0.87, [11.0, 51.0, 301.0, 121.0]),  // near-duplicate of second text
        (8, 0.90, [10.0, 200.0, 300.0, 350.0]), // table
    ];

    let mut regions = DpdfPipeline::detections_to_regions(&detections);
    assert_eq!(regions.len(), 5);

    let config = PostProcessConfig::default();
    postprocess(&mut regions, &config);

    // Low-confidence region (0.15 < 0.3 min_confidence) should be filtered.
    assert!(regions.len() < 5);
    assert!(regions.iter().all(|r| r.confidence() >= 0.3));

    // Build page from postprocessed regions.
    let page = pipeline.build_page(regions, 612, 792);
    assert!(!page.regions.is_empty());
    assert!(!page.reading_order.is_empty());

    // Markdown should contain section header and table.
    let md = DpdfPipeline::to_markdown(&page);
    assert!(md.contains("[table]") || md.contains("|"));
}

#[test]
fn test_table_structure_spanning_cell() {
    // Table with a spanning cell that covers 2 columns in row 0.
    let detections = vec![
        // 2 rows
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 200.0,
            y2: 50.0,
            confidence: 0.9,
            class_id: 1,
        },
        Detection {
            x1: 0.0,
            y1: 50.0,
            x2: 200.0,
            y2: 100.0,
            confidence: 0.9,
            class_id: 1,
        },
        // 2 columns
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
            confidence: 0.9,
            class_id: 2,
        },
        Detection {
            x1: 100.0,
            y1: 0.0,
            x2: 200.0,
            y2: 100.0,
            confidence: 0.9,
            class_id: 2,
        },
        // Spanning cell: covers both columns in row 0.
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 200.0,
            y2: 50.0,
            confidence: 0.85,
            class_id: 3,
        },
    ];

    let table = parse_structure(&detections, &TableStructureConfig::default());

    assert_eq!(table.num_rows, 2);
    assert_eq!(table.num_cols, 2);

    // Row 0 should have a spanning cell.
    let row0 = &table.rows[0];
    let has_spanning = row0.cells.iter().any(|c| c.col_span > 1);
    assert!(has_spanning, "row 0 should have a spanning cell");
}

#[test]
fn test_table_structure_empty_detections() {
    let table = parse_structure(&[], &TableStructureConfig::default());
    assert_eq!(table.num_rows, 0);
    assert_eq!(table.num_cols, 0);
    assert!(table.rows.is_empty());

    // HTML/Markdown/CSV of empty table.
    let html = to_html(&table);
    assert!(html.contains("<table>"));

    let md = to_markdown_table(&table);
    assert!(md.is_empty());

    let csv = to_csv(&table);
    assert!(csv.is_empty());
}

#[test]
fn test_document_output_multi_page_pipeline() {
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let empty: Vec<(usize, f32, [f32; 4])> = vec![];
    let page1: Vec<(usize, f32, [f32; 4])> = vec![(9, 0.9, [10.0, 10.0, 200.0, 50.0])];
    let pages_input: Vec<(&[(usize, f32, [f32; 4])], usize, usize)> =
        vec![(&page1, 612, 792), (&empty, 612, 792), (&page1, 800, 600)];
    let doc = pipeline.process_pages(&pages_input);

    assert_eq!(doc.pages.len(), 3);
    assert_eq!(doc.pages[0].regions.len(), 1);
    assert!(doc.pages[1].regions.is_empty());
    assert_eq!(doc.pages[2].regions.len(), 1);
}

#[test]
fn test_iou_computation_edge_cases() {
    // Identical boxes: IoU = 1.0.
    let iou = compute_iou(&[0.0, 0.0, 100.0, 100.0], &[0.0, 0.0, 100.0, 100.0]);
    assert!((iou - 1.0).abs() < f32::EPSILON);

    // Non-overlapping boxes: IoU = 0.0.
    let iou = compute_iou(&[0.0, 0.0, 50.0, 50.0], &[100.0, 100.0, 200.0, 200.0]);
    assert!((iou - 0.0).abs() < f32::EPSILON);

    // Zero-area box: IoU = 0.0.
    let iou = compute_iou(&[10.0, 10.0, 10.0, 10.0], &[0.0, 0.0, 100.0, 100.0]);
    assert!((iou - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_extract_text_uses_brackets_for_empty_content() {
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    // Detections produce empty-content regions.
    let regions = DpdfPipeline::detections_to_regions(&[
        (9, 0.9, [10.0, 10.0, 100.0, 50.0]),
        (6, 0.8, [10.0, 60.0, 100.0, 120.0]),
    ]);
    let page = pipeline.build_page(regions, 612, 792);
    let text = DpdfPipeline::extract_text(&page);

    assert!(
        text.contains("[text]"),
        "empty text region should show [text]"
    );
    assert!(
        text.contains("[picture]"),
        "empty figure region should show [picture]"
    );
}
