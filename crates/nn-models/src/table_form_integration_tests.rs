// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::form_field_association::{EntitySpan, EntityType, FormField, FormHeader};
use crate::table_structure::{StructuredTable, TableCell, TableRow};

#[test]
fn test_classify_regions_table() {
    let config = TableFormConfig::default();
    let detections = vec![("table".to_string(), [0.0, 0.0, 100.0, 100.0], 0.9)];
    let regions = classify_regions(&detections, &config);
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].kind, RegionKind::Table);
}

#[test]
fn test_classify_regions_form() {
    let config = TableFormConfig::default();
    let detections = vec![("text".to_string(), [0.0, 0.0, 100.0, 100.0], 0.9)];
    let regions = classify_regions(&detections, &config);
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].kind, RegionKind::Form);
}

#[test]
fn test_classify_regions_low_confidence_table() {
    let config = TableFormConfig {
        table_confidence_threshold: 0.5,
        ..Default::default()
    };
    let detections = vec![("table".to_string(), [0.0, 0.0, 100.0, 100.0], 0.3)];
    let regions = classify_regions(&detections, &config);
    assert_eq!(regions[0].kind, RegionKind::Other);
}

#[test]
fn test_classify_regions_mixed() {
    let config = TableFormConfig {
        region_merge_iou: 0.2,
        ..Default::default()
    };
    let detections = vec![
        ("table".to_string(), [0.0, 0.0, 100.0, 100.0], 0.9),
        ("text".to_string(), [10.0, 10.0, 90.0, 90.0], 0.8),
    ];
    let regions = classify_regions(&detections, &config);
    // The text region overlaps the table region, so it becomes Mixed.
    let mixed_count = regions
        .iter()
        .filter(|r| r.kind == RegionKind::Mixed)
        .count();
    assert!(mixed_count >= 1);
}

#[test]
fn test_classify_regions_other() {
    let config = TableFormConfig::default();
    let detections = vec![("picture".to_string(), [0.0, 0.0, 100.0, 100.0], 0.9)];
    let regions = classify_regions(&detections, &config);
    assert_eq!(regions[0].kind, RegionKind::Other);
}

#[test]
fn test_merge_results_basic() {
    let tables = vec![TableExtractionResult {
        table: StructuredTable {
            rows: vec![TableRow {
                cells: vec![TableCell {
                    row: 0,
                    col: 0,
                    row_span: 1,
                    col_span: 1,
                    bbox: [0.0; 4],
                    confidence: 0.9,
                }],
            }],
            num_rows: 1,
            num_cols: 1,
            caption: None,
        },
        bbox: [0.0, 0.0, 100.0, 100.0],
        confidence: 0.9,
    }];

    let form = FormExtractionResult {
        fields: vec![FormField {
            key: EntitySpan {
                text: "Name:".to_string(),
                bbox: [200.0, 0.0, 250.0, 20.0],
                entity_type: EntityType::Question,
                confidence: 1.0,
            },
            value: Some(EntitySpan {
                text: "John".to_string(),
                bbox: [260.0, 0.0, 300.0, 20.0],
                entity_type: EntityType::Answer,
                confidence: 1.0,
            }),
        }],
        headers: Vec::new(),
        orphan_values: Vec::new(),
    };

    let classified = vec![
        ClassifiedRegion {
            bbox: [0.0, 0.0, 100.0, 100.0],
            confidence: 0.9,
            kind: RegionKind::Table,
            source: "layout_detector",
        },
        ClassifiedRegion {
            bbox: [200.0, 0.0, 400.0, 100.0],
            confidence: 0.8,
            kind: RegionKind::Form,
            source: "layout_detector",
        },
    ];

    let result = merge_results(tables, form, &classified);
    assert_eq!(result.tables.len(), 1);
    assert_eq!(result.form.fields.len(), 1);
    assert!(result.unclassified_regions.is_empty());
}

#[test]
fn test_summarize_empty() {
    let result = PageExtractionResult {
        tables: Vec::new(),
        form: empty_form_result(),
        unclassified_regions: Vec::new(),
    };
    let summary = summarize(&result);
    assert_eq!(summary.num_tables, 0);
    assert_eq!(summary.total_cells, 0);
    assert_eq!(summary.num_form_fields, 0);
    assert_eq!(summary.num_paired_fields, 0);
}

#[test]
fn test_summarize_with_data() {
    let tables = vec![TableExtractionResult {
        table: StructuredTable {
            rows: vec![
                TableRow {
                    cells: vec![TableCell {
                        row: 0,
                        col: 0,
                        row_span: 1,
                        col_span: 2,
                        bbox: [0.0; 4],
                        confidence: 0.9,
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
                            confidence: 0.9,
                        },
                        TableCell {
                            row: 1,
                            col: 1,
                            row_span: 1,
                            col_span: 1,
                            bbox: [0.0; 4],
                            confidence: 0.8,
                        },
                    ],
                },
            ],
            num_rows: 2,
            num_cols: 2,
            caption: None,
        },
        bbox: [0.0, 0.0, 100.0, 100.0],
        confidence: 0.9,
    }];

    let form = FormExtractionResult {
        fields: vec![
            FormField {
                key: EntitySpan {
                    text: "Name:".to_string(),
                    bbox: [0.0; 4],
                    entity_type: EntityType::Question,
                    confidence: 1.0,
                },
                value: Some(EntitySpan {
                    text: "John".to_string(),
                    bbox: [0.0; 4],
                    entity_type: EntityType::Answer,
                    confidence: 1.0,
                }),
            },
            FormField {
                key: EntitySpan {
                    text: "Age:".to_string(),
                    bbox: [0.0; 4],
                    entity_type: EntityType::Question,
                    confidence: 1.0,
                },
                value: None,
            },
        ],
        headers: vec![FormHeader {
            span: EntitySpan {
                text: "Personal Info".to_string(),
                bbox: [0.0; 4],
                entity_type: EntityType::Header,
                confidence: 1.0,
            },
        }],
        orphan_values: vec![EntitySpan {
            text: "orphan".to_string(),
            bbox: [0.0; 4],
            entity_type: EntityType::Answer,
            confidence: 1.0,
        }],
    };

    let result = PageExtractionResult {
        tables,
        form,
        unclassified_regions: Vec::new(),
    };
    let summary = summarize(&result);
    assert_eq!(summary.num_tables, 1);
    assert_eq!(summary.total_cells, 3);
    assert_eq!(summary.total_spanning_cells, 1);
    assert_eq!(summary.num_form_fields, 2);
    assert_eq!(summary.num_paired_fields, 1);
    assert_eq!(summary.num_headers, 1);
    assert_eq!(summary.num_orphan_values, 1);
}

#[test]
fn test_empty_form_result() {
    let result = empty_form_result();
    assert!(result.fields.is_empty());
    assert!(result.headers.is_empty());
    assert!(result.orphan_values.is_empty());
}

#[test]
fn test_region_kind_equality() {
    assert_eq!(RegionKind::Table, RegionKind::Table);
    assert_ne!(RegionKind::Table, RegionKind::Form);
    assert_ne!(RegionKind::Form, RegionKind::Mixed);
}
