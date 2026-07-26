// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_default_pipeline_has_eight_models() {
    let registry = DpdfModelRegistry::default_pipeline();
    assert_eq!(registry.len(), 8);
}

#[test]
fn test_default_pipeline_not_empty() {
    let registry = DpdfModelRegistry::default_pipeline();
    assert!(!registry.is_empty());
}

#[test]
fn test_new_registry_is_empty() {
    let registry = DpdfModelRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
}

#[test]
fn test_get_granite_docling() {
    let registry = DpdfModelRegistry::default_pipeline();
    let entry = registry
        .get("granite_docling")
        .expect("granite_docling should exist");
    assert_eq!(entry.model_type, ModelType::VLM);
    assert_eq!(entry.parameter_count, 258_000_000);
}

#[test]
fn test_get_doclayout_yolo() {
    let registry = DpdfModelRegistry::default_pipeline();
    let entry = registry
        .get("doclayout_yolo")
        .expect("doclayout_yolo should exist");
    assert_eq!(entry.model_type, ModelType::LayoutDetection);
    assert_eq!(entry.parameter_count, 16_000_000);
}

#[test]
fn test_get_glm_ocr() {
    let registry = DpdfModelRegistry::default_pipeline();
    let entry = registry.get("glm_ocr").expect("glm_ocr should exist");
    assert_eq!(entry.model_type, ModelType::OCR);
    assert_eq!(entry.parameter_count, 900_000_000);
}

#[test]
fn test_get_table_transformer() {
    let registry = DpdfModelRegistry::default_pipeline();
    let entry = registry
        .get("table_transformer")
        .expect("table_transformer should exist");
    assert_eq!(entry.model_type, ModelType::TableStructure);
    assert_eq!(entry.parameter_count, 28_800_000);
}

#[test]
fn test_get_qwen3_vl() {
    let registry = DpdfModelRegistry::default_pipeline();
    let entry = registry.get("qwen3_vl").expect("qwen3_vl should exist");
    assert_eq!(entry.model_type, ModelType::VLM);
    assert_eq!(entry.parameter_count, 8_000_000_000);
}

#[test]
fn test_get_paddle_ocr() {
    let registry = DpdfModelRegistry::default_pipeline();
    let entry = registry.get("paddle_ocr").expect("paddle_ocr should exist");
    assert_eq!(entry.model_type, ModelType::VLM);
    assert_eq!(entry.parameter_count, 1_300_000_000);
}

#[test]
fn test_get_firered_ocr() {
    let registry = DpdfModelRegistry::default_pipeline();
    let entry = registry
        .get("firered_ocr")
        .expect("firered_ocr should exist");
    assert_eq!(entry.model_type, ModelType::OCR);
    assert_eq!(entry.parameter_count, 2_000_000_000);
}

#[test]
fn test_get_rt_detr_heron() {
    let registry = DpdfModelRegistry::default_pipeline();
    let entry = registry
        .get("rt_detr_heron")
        .expect("rt_detr_heron should exist");
    assert_eq!(entry.model_type, ModelType::LayoutDetection);
    assert_eq!(entry.parameter_count, 42_900_000);
}

#[test]
fn test_get_nonexistent_returns_none() {
    let registry = DpdfModelRegistry::default_pipeline();
    assert!(registry.get("nonexistent_model").is_none());
}

#[test]
fn test_list_by_type_ocr() {
    let registry = DpdfModelRegistry::default_pipeline();
    let ocr_models = registry.list_by_type(ModelType::OCR);
    assert_eq!(ocr_models.len(), 2); // glm_ocr, firered_ocr
    assert!(ocr_models.iter().all(|e| e.model_type == ModelType::OCR));
}

#[test]
fn test_list_by_type_vlm() {
    let registry = DpdfModelRegistry::default_pipeline();
    let vlm_models = registry.list_by_type(ModelType::VLM);
    assert_eq!(vlm_models.len(), 3); // granite_docling, qwen3_vl, paddle_ocr
    assert!(vlm_models.iter().all(|e| e.model_type == ModelType::VLM));
}

#[test]
fn test_list_by_type_layout_detection() {
    let registry = DpdfModelRegistry::default_pipeline();
    let layout_models = registry.list_by_type(ModelType::LayoutDetection);
    assert_eq!(layout_models.len(), 2); // doclayout_yolo, rt_detr_heron
    assert!(layout_models
        .iter()
        .all(|e| e.model_type == ModelType::LayoutDetection));
}

#[test]
fn test_list_by_type_table_structure() {
    let registry = DpdfModelRegistry::default_pipeline();
    let table_models = registry.list_by_type(ModelType::TableStructure);
    assert_eq!(table_models.len(), 1);
    assert_eq!(table_models[0].name, "table_transformer");
}

#[test]
fn test_register_custom_model() {
    let mut registry = DpdfModelRegistry::new();
    registry.register(ModelEntry {
        name: "custom_model".into(),
        model_type: ModelType::OCR,
        description: "A custom OCR model".into(),
        preprocess_config: DpdfPreprocessConfig::for_granite_docling(),
        parameter_count: 100_000,
    });
    assert_eq!(registry.len(), 1);
    let entry = registry
        .get("custom_model")
        .expect("custom_model should exist");
    assert_eq!(entry.model_type, ModelType::OCR);
    assert_eq!(entry.parameter_count, 100_000);
}

#[test]
fn test_register_overwrites_existing() {
    let mut registry = DpdfModelRegistry::default_pipeline();
    assert_eq!(
        registry.get("granite_docling").unwrap().parameter_count,
        258_000_000
    );

    registry.register(ModelEntry {
        name: "granite_docling".into(),
        model_type: ModelType::VLM,
        description: "Updated description".into(),
        preprocess_config: DpdfPreprocessConfig::for_granite_docling(),
        parameter_count: 999,
    });
    assert_eq!(registry.len(), 8); // still 8, not 9
    assert_eq!(
        registry.get("granite_docling").unwrap().parameter_count,
        999
    );
}

#[test]
fn test_models_iterator_count() {
    let registry = DpdfModelRegistry::default_pipeline();
    assert_eq!(registry.models().count(), 8);
}

#[test]
fn test_model_type_label() {
    assert_eq!(ModelType::LayoutDetection.label(), "Layout Detection");
    assert_eq!(ModelType::OCR.label(), "OCR");
    assert_eq!(ModelType::TableStructure.label(), "Table Structure");
    assert_eq!(ModelType::VLM.label(), "VLM");
}

#[test]
fn test_default_trait_creates_empty_registry() {
    let registry = DpdfModelRegistry::default();
    assert!(registry.is_empty());
}

#[test]
fn test_all_default_entries_have_nonempty_description() {
    let registry = DpdfModelRegistry::default_pipeline();
    for entry in registry.models() {
        assert!(
            !entry.description.is_empty(),
            "model '{}' has empty description",
            entry.name,
        );
    }
}

#[test]
fn test_all_default_entries_have_positive_param_count() {
    let registry = DpdfModelRegistry::default_pipeline();
    for entry in registry.models() {
        assert!(
            entry.parameter_count > 0,
            "model '{}' has zero parameter_count",
            entry.name,
        );
    }
}
