// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the dpdf document inference pipeline.

use super::*;

// ---------------------------------------------------------------------------
// PipelineConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_config_default_thresholds() {
    let cfg = PipelineConfig::default();
    assert!((cfg.layout_conf_threshold - 0.25).abs() < f32::EPSILON);
    assert!((cfg.layout_iou_threshold - 0.45).abs() < f32::EPSILON);
    assert_eq!(cfg.ocr_max_tokens, 1024);
    assert!(cfg.enable_table_structure);
    // Postprocess config defaults
    assert!((cfg.postprocess_config.merge_iou - 0.5).abs() < f32::EPSILON);
    assert!((cfg.postprocess_config.dedup_similarity - 0.9).abs() < f32::EPSILON);
    assert!((cfg.postprocess_config.min_confidence - 0.3).abs() < f32::EPSILON);
    assert!(cfg.postprocess_config.enable_model_fusion);
    // Table structure config defaults
    assert!((cfg.table_structure_config.iou_threshold - 0.5).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// classify_detection — one test per class
// ---------------------------------------------------------------------------

#[test]
fn test_classify_detection_all_10_classes() {
    let class_map: [(usize, &str); 10] = [
        (0, "caption"),
        (1, "footnote"),
        (2, "formula"),
        (3, "list-item"),
        (4, "page-footer"),
        (5, "page-header"),
        (6, "picture"),
        (7, "section-header"),
        (8, "table"),
        (9, "text"),
    ];
    let bbox = [10.0, 20.0, 100.0, 50.0];
    for (id, expected_name) in &class_map {
        let region = DpdfPipeline::classify_detection(*id, bbox, 0.9);
        assert_eq!(
            region.class_name(),
            *expected_name,
            "class_id {id} should map to {expected_name}"
        );
        assert_eq!(region.bbox(), bbox);
        assert!((region.confidence() - 0.9).abs() < f32::EPSILON);
    }
}

#[test]
fn test_classify_detection_out_of_range_defaults_to_text() {
    let region = DpdfPipeline::classify_detection(99, [0.0; 4], 0.5);
    assert_eq!(region.class_name(), "text");
}

// ---------------------------------------------------------------------------
// detections_to_regions
// ---------------------------------------------------------------------------

#[test]
fn test_detections_to_regions_preserves_order() {
    let dets = vec![
        (7, 0.95, [0.0, 0.0, 100.0, 30.0]),   // section-header
        (9, 0.88, [0.0, 30.0, 100.0, 80.0]),  // text
        (1, 0.70, [0.0, 80.0, 100.0, 100.0]), // footnote
    ];
    let regions = DpdfPipeline::detections_to_regions(&dets);
    assert_eq!(regions.len(), 3);
    assert_eq!(regions[0].class_name(), "section-header");
    assert_eq!(regions[1].class_name(), "text");
    assert_eq!(regions[2].class_name(), "footnote");
}

#[test]
fn test_detections_to_regions_empty_input() {
    let regions = DpdfPipeline::detections_to_regions(&[]);
    assert!(regions.is_empty());
}

// ---------------------------------------------------------------------------
// compute_reading_order
// ---------------------------------------------------------------------------

#[test]
fn test_reading_order_top_to_bottom() {
    let regions = vec![
        DocumentRegion::Text {
            content: "bottom".into(),
            bbox: [0.0, 80.0, 100.0, 100.0],
            confidence: 0.9,
        },
        DocumentRegion::Text {
            content: "top".into(),
            bbox: [0.0, 0.0, 100.0, 20.0],
            confidence: 0.9,
        },
    ];
    let order = DpdfPipeline::compute_reading_order(&regions);
    assert_eq!(order, vec![1, 0], "top region (index 1) should come first");
}

#[test]
fn test_reading_order_header_first_footer_last() {
    let regions = vec![
        DocumentRegion::Text {
            content: "body".into(),
            bbox: [0.0, 50.0, 100.0, 70.0],
            confidence: 0.9,
        },
        DocumentRegion::PageFooter {
            content: "footer".into(),
            bbox: [0.0, 90.0, 100.0, 100.0],
            confidence: 0.8,
        },
        DocumentRegion::PageHeader {
            content: "header".into(),
            bbox: [0.0, 0.0, 100.0, 10.0],
            confidence: 0.8,
        },
    ];
    let order = DpdfPipeline::compute_reading_order(&regions);
    // header (idx 2) first, body (idx 0) middle, footer (idx 1) last
    assert_eq!(order[0], 2, "page header should be first");
    assert_eq!(order[2], 1, "page footer should be last");
}

#[test]
fn test_reading_order_left_to_right_tiebreak() {
    let regions = vec![
        DocumentRegion::Text {
            content: "right".into(),
            bbox: [200.0, 50.0, 300.0, 70.0],
            confidence: 0.9,
        },
        DocumentRegion::Text {
            content: "left".into(),
            bbox: [0.0, 50.0, 100.0, 70.0],
            confidence: 0.9,
        },
    ];
    let order = DpdfPipeline::compute_reading_order(&regions);
    assert_eq!(order, vec![1, 0], "left column should come before right");
}

#[test]
fn test_reading_order_empty() {
    let order = DpdfPipeline::compute_reading_order(&[]);
    assert!(order.is_empty());
}

// ---------------------------------------------------------------------------
// extract_text
// ---------------------------------------------------------------------------

#[test]
fn test_extract_text_uses_reading_order() {
    let regions = vec![
        DocumentRegion::Text {
            content: "Second".into(),
            bbox: [0.0, 50.0, 100.0, 70.0],
            confidence: 0.9,
        },
        DocumentRegion::SectionHeader {
            content: "First".into(),
            bbox: [0.0, 0.0, 100.0, 20.0],
            confidence: 0.95,
        },
    ];
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let page = pipeline.build_page(regions, 612, 792);
    let text = DpdfPipeline::extract_text(&page);
    assert!(
        text.starts_with("First"),
        "section header should come first in extracted text: {text}"
    );
}

#[test]
fn test_extract_text_placeholder_for_empty_content() {
    let regions = vec![DocumentRegion::Text {
        content: String::new(),
        bbox: [0.0, 0.0, 100.0, 50.0],
        confidence: 0.9,
    }];
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let page = pipeline.build_page(regions, 100, 100);
    let text = DpdfPipeline::extract_text(&page);
    assert_eq!(text, "[text]");
}

// ---------------------------------------------------------------------------
// to_markdown
// ---------------------------------------------------------------------------

#[test]
fn test_to_markdown_section_header() {
    let regions = vec![DocumentRegion::SectionHeader {
        content: "Introduction".into(),
        bbox: [0.0, 0.0, 200.0, 30.0],
        confidence: 0.95,
    }];
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let page = pipeline.build_page(regions, 612, 792);
    let md = DpdfPipeline::to_markdown(&page);
    assert_eq!(md, "## Introduction");
}

#[test]
fn test_to_markdown_list_item() {
    let regions = vec![DocumentRegion::ListItem {
        content: "Buy milk".into(),
        bbox: [0.0, 0.0, 200.0, 20.0],
        confidence: 0.9,
    }];
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let page = pipeline.build_page(regions, 612, 792);
    let md = DpdfPipeline::to_markdown(&page);
    assert_eq!(md, "- Buy milk");
}

#[test]
fn test_to_markdown_table() {
    let regions = vec![DocumentRegion::Table {
        cells: vec![
            vec!["Name".into(), "Age".into()],
            vec!["Alice".into(), "30".into()],
        ],
        bbox: [0.0, 0.0, 300.0, 100.0],
        confidence: 0.85,
    }];
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let page = pipeline.build_page(regions, 612, 792);
    let md = DpdfPipeline::to_markdown(&page);
    assert!(md.contains("| Name | Age |"), "markdown table header: {md}");
    assert!(md.contains("| --- | --- |"), "markdown separator: {md}");
    assert!(md.contains("| Alice | 30 |"), "markdown data row: {md}");
}

#[test]
fn test_to_markdown_formula_with_latex() {
    let regions = vec![DocumentRegion::Formula {
        latex: Some("E = mc^2".into()),
        bbox: [0.0, 0.0, 200.0, 30.0],
        confidence: 0.9,
    }];
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let page = pipeline.build_page(regions, 612, 792);
    let md = DpdfPipeline::to_markdown(&page);
    assert_eq!(md, "$E = mc^2$");
}

// ---------------------------------------------------------------------------
// process_pages (multi-page)
// ---------------------------------------------------------------------------

#[test]
fn test_process_pages_multi_page() {
    let pipeline = DpdfPipeline::new(PipelineConfig::default());
    let page1_dets: Vec<(usize, f32, [f32; 4])> = vec![(9, 0.9, [0.0, 0.0, 100.0, 50.0])];
    let page2_dets: Vec<(usize, f32, [f32; 4])> = vec![
        (7, 0.95, [0.0, 0.0, 200.0, 30.0]),
        (9, 0.88, [0.0, 30.0, 200.0, 80.0]),
    ];
    let pages_data: Vec<(&[(usize, f32, [f32; 4])], usize, usize)> = vec![
        (page1_dets.as_slice(), 612, 792),
        (page2_dets.as_slice(), 612, 792),
    ];
    let doc = pipeline.process_pages(&pages_data);
    assert_eq!(doc.pages.len(), 2);
    assert_eq!(doc.pages[0].regions.len(), 1);
    assert_eq!(doc.pages[1].regions.len(), 2);
    assert_eq!(doc.pages[0].width, 612);
    assert_eq!(doc.pages[1].height, 792);
}

// ---------------------------------------------------------------------------
// DocumentRegion accessors
// ---------------------------------------------------------------------------

#[test]
fn test_region_bbox_and_confidence() {
    let region = DocumentRegion::Figure {
        caption: Some("A chart".into()),
        bbox: [10.0, 20.0, 300.0, 400.0],
        confidence: 0.77,
    };
    assert_eq!(region.bbox(), [10.0, 20.0, 300.0, 400.0]);
    assert!((region.confidence() - 0.77).abs() < f32::EPSILON);
    assert_eq!(region.class_name(), "picture");
}
