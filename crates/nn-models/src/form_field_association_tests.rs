// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn make_token(text: &str, bbox: [f32; 4], tag: EntityTag) -> LabeledToken {
    LabeledToken {
        text: text.to_string(),
        bbox,
        tag,
    }
}

#[test]
fn test_entity_tag_from_label_id() {
    assert_eq!(EntityTag::from_label_id(0), EntityTag::Other);
    assert_eq!(EntityTag::from_label_id(1), EntityTag::BQuestion);
    assert_eq!(EntityTag::from_label_id(2), EntityTag::IQuestion);
    assert_eq!(EntityTag::from_label_id(3), EntityTag::BAnswer);
    assert_eq!(EntityTag::from_label_id(4), EntityTag::IAnswer);
    assert_eq!(EntityTag::from_label_id(5), EntityTag::BHeader);
    assert_eq!(EntityTag::from_label_id(6), EntityTag::IHeader);
    assert_eq!(EntityTag::from_label_id(99), EntityTag::Other);
}

#[test]
fn test_entity_tag_is_begin() {
    assert!(EntityTag::BQuestion.is_begin());
    assert!(EntityTag::BAnswer.is_begin());
    assert!(EntityTag::BHeader.is_begin());
    assert!(!EntityTag::IQuestion.is_begin());
    assert!(!EntityTag::IAnswer.is_begin());
    assert!(!EntityTag::IHeader.is_begin());
    assert!(!EntityTag::Other.is_begin());
}

#[test]
fn test_entity_tag_entity_type() {
    assert_eq!(
        EntityTag::BQuestion.entity_type(),
        Some(EntityType::Question)
    );
    assert_eq!(
        EntityTag::IQuestion.entity_type(),
        Some(EntityType::Question)
    );
    assert_eq!(EntityTag::BAnswer.entity_type(), Some(EntityType::Answer));
    assert_eq!(EntityTag::IAnswer.entity_type(), Some(EntityType::Answer));
    assert_eq!(EntityTag::BHeader.entity_type(), Some(EntityType::Header));
    assert_eq!(EntityTag::IHeader.entity_type(), Some(EntityType::Header));
    assert_eq!(EntityTag::Other.entity_type(), None);
}

#[test]
fn test_decode_bio_spans_simple() {
    let tokens = vec![
        make_token("Name", [0.0, 0.0, 50.0, 20.0], EntityTag::BQuestion),
        make_token(":", [50.0, 0.0, 55.0, 20.0], EntityTag::IQuestion),
        make_token("John", [60.0, 0.0, 100.0, 20.0], EntityTag::BAnswer),
        make_token("Doe", [100.0, 0.0, 140.0, 20.0], EntityTag::IAnswer),
    ];
    let spans = decode_bio_spans(&tokens);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].entity_type, EntityType::Question);
    assert_eq!(spans[0].text, "Name :");
    assert_eq!(spans[1].entity_type, EntityType::Answer);
    assert_eq!(spans[1].text, "John Doe");
}

#[test]
fn test_decode_bio_spans_with_other() {
    let tokens = vec![
        make_token("Name", [0.0, 0.0, 50.0, 20.0], EntityTag::BQuestion),
        make_token("padding", [55.0, 0.0, 80.0, 20.0], EntityTag::Other),
        make_token("John", [85.0, 0.0, 120.0, 20.0], EntityTag::BAnswer),
    ];
    let spans = decode_bio_spans(&tokens);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].text, "Name");
    assert_eq!(spans[1].text, "John");
}

#[test]
fn test_decode_bio_spans_consecutive_begins() {
    let tokens = vec![
        make_token("Q1", [0.0, 0.0, 30.0, 20.0], EntityTag::BQuestion),
        make_token("Q2", [35.0, 0.0, 65.0, 20.0], EntityTag::BQuestion),
    ];
    let spans = decode_bio_spans(&tokens);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].text, "Q1");
    assert_eq!(spans[1].text, "Q2");
}

#[test]
fn test_decode_bio_spans_empty() {
    let spans = decode_bio_spans(&[]);
    assert!(spans.is_empty());
}

#[test]
fn test_decode_bio_spans_header() {
    let tokens = vec![
        make_token("Section", [0.0, 0.0, 60.0, 20.0], EntityTag::BHeader),
        make_token("A", [60.0, 0.0, 75.0, 20.0], EntityTag::IHeader),
    ];
    let spans = decode_bio_spans(&tokens);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].entity_type, EntityType::Header);
    assert_eq!(spans[0].text, "Section A");
}

#[test]
fn test_pair_key_value_basic() {
    let spans = vec![
        EntitySpan {
            text: "Name:".to_string(),
            bbox: [0.0, 0.0, 50.0, 20.0],
            entity_type: EntityType::Question,
            confidence: 1.0,
        },
        EntitySpan {
            text: "John".to_string(),
            bbox: [60.0, 0.0, 100.0, 20.0],
            entity_type: EntityType::Answer,
            confidence: 1.0,
        },
    ];
    let config = FormAssociationConfig::default();
    let result = pair_key_value(&spans, &config);
    assert_eq!(result.fields.len(), 1);
    assert!(result.fields[0].value.is_some());
    assert_eq!(result.fields[0].value.as_ref().unwrap().text, "John");
    assert!(result.orphan_values.is_empty());
}

#[test]
fn test_pair_key_value_no_match() {
    let spans = vec![
        EntitySpan {
            text: "Name:".to_string(),
            bbox: [0.0, 0.0, 50.0, 20.0],
            entity_type: EntityType::Question,
            confidence: 1.0,
        },
        EntitySpan {
            text: "Far away".to_string(),
            bbox: [500.0, 500.0, 600.0, 520.0],
            entity_type: EntityType::Answer,
            confidence: 1.0,
        },
    ];
    let config = FormAssociationConfig {
        max_pair_distance: 100.0,
        ..Default::default()
    };
    let result = pair_key_value(&spans, &config);
    assert_eq!(result.fields.len(), 1);
    assert!(result.fields[0].value.is_none());
    assert_eq!(result.orphan_values.len(), 1);
}

#[test]
fn test_pair_key_value_multiple_pairs() {
    let spans = vec![
        EntitySpan {
            text: "Name:".to_string(),
            bbox: [0.0, 0.0, 50.0, 20.0],
            entity_type: EntityType::Question,
            confidence: 1.0,
        },
        EntitySpan {
            text: "John".to_string(),
            bbox: [60.0, 0.0, 100.0, 20.0],
            entity_type: EntityType::Answer,
            confidence: 1.0,
        },
        EntitySpan {
            text: "Age:".to_string(),
            bbox: [0.0, 30.0, 40.0, 50.0],
            entity_type: EntityType::Question,
            confidence: 1.0,
        },
        EntitySpan {
            text: "30".to_string(),
            bbox: [60.0, 30.0, 80.0, 50.0],
            entity_type: EntityType::Answer,
            confidence: 1.0,
        },
    ];
    let config = FormAssociationConfig::default();
    let result = pair_key_value(&spans, &config);
    assert_eq!(result.fields.len(), 2);
    assert!(result.fields[0].value.is_some());
    assert!(result.fields[1].value.is_some());
}

#[test]
fn test_extract_form_fields_end_to_end() {
    let tokens = vec![
        make_token("Name", [0.0, 0.0, 50.0, 20.0], EntityTag::BQuestion),
        make_token(":", [50.0, 0.0, 55.0, 20.0], EntityTag::IQuestion),
        make_token("John", [60.0, 0.0, 100.0, 20.0], EntityTag::BAnswer),
        make_token("Doe", [100.0, 0.0, 140.0, 20.0], EntityTag::IAnswer),
    ];
    let config = FormAssociationConfig::default();
    let result = extract_form_fields(&tokens, &config);
    assert_eq!(result.fields.len(), 1);
    assert_eq!(result.fields[0].key.text, "Name :");
    assert_eq!(result.fields[0].value.as_ref().unwrap().text, "John Doe");
}

#[test]
fn test_merge_bbox() {
    let merged = merge_bbox(&[10.0, 20.0, 30.0, 40.0], &[5.0, 25.0, 35.0, 38.0]);
    assert!((merged[0] - 5.0).abs() < 1e-6);
    assert!((merged[1] - 20.0).abs() < 1e-6);
    assert!((merged[2] - 35.0).abs() < 1e-6);
    assert!((merged[3] - 40.0).abs() < 1e-6);
}
