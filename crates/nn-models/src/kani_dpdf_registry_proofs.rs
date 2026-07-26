// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for dpdf_registry dispatch safety (#3939).
//!
//! Proves safety and correctness invariants for the model registry, including
//! dispatch lookup, model entry validation, and default pipeline completeness.
//!
//! **Areas proved (10 harnesses):**
//!
//!  1. All 4 ModelType variants can be constructed and labeled.
//!  2. `default_pipeline()` returns a registry with exactly 7 models.
//!  3. All default entries have non-empty name and positive parameter count.
//!  4. `get()` with an absent key returns `None` (never panics).
//!  5. `list_by_type()` returns only entries matching the requested type.
//!  6. Register-then-get roundtrip returns `Some` with matching name.
//!  7. Default registry has unique model names (no duplicates).
//!  8. All 4 ModelType variants are represented in the default pipeline.
//!  9. `register()` overwrites existing entries (len stays the same).
//! 10. `new()` creates an empty registry (`is_empty` is true, `len` is 0).

use crate::dpdf_image_preprocess::DpdfPreprocessConfig;
use crate::dpdf_registry::{DpdfModelRegistry, ModelEntry, ModelType};

// ===========================================================================
// Harness 1: ModelType exhaustiveness — all 4 variants constructible
// ===========================================================================

/// SUBSTANTIVE: Proves that all 4 ModelType variants can be constructed and
/// that `label()` returns a non-empty string for each. Catches dead variants
/// or missing match arms in `label()`.
#[kani::proof]
#[kani::unwind(2)]
fn proof_model_type_exhaustiveness() {
    let variants = [
        ModelType::LayoutDetection,
        ModelType::OCR,
        ModelType::TableStructure,
        ModelType::VLM,
    ];

    let mut i = 0;
    while i < variants.len() {
        let label = variants[i].label();
        assert!(!label.is_empty(), "ModelType::label() must be non-empty");
        i += 1;
    }

    // Verify distinctness of labels.
    assert_ne!(
        ModelType::LayoutDetection.label(),
        ModelType::OCR.label(),
        "distinct variants must have distinct labels"
    );
    assert_ne!(
        ModelType::TableStructure.label(),
        ModelType::VLM.label(),
        "distinct variants must have distinct labels"
    );
}

// ===========================================================================
// Harness 2: default_pipeline() completeness — exactly 7 models
// ===========================================================================

/// SUBSTANTIVE: Proves that `default_pipeline()` creates a registry with
/// exactly 7 entries and is not empty. Catches accidental omissions or
/// double-registrations that change the expected count.
#[kani::proof]
#[kani::unwind(2)]
fn proof_default_pipeline_has_seven_models() {
    let registry = DpdfModelRegistry::default_pipeline();
    assert_eq!(
        registry.len(),
        7,
        "default pipeline must have exactly 7 models"
    );
    assert!(!registry.is_empty(), "default pipeline must not be empty");
}

// ===========================================================================
// Harness 3: ModelEntry field invariants — name non-empty, params > 0
// ===========================================================================

/// SUBSTANTIVE: Proves that every entry in the default pipeline has a
/// non-empty name and a strictly positive parameter count. Catches entries
/// with placeholder/zero values.
#[kani::proof]
#[kani::unwind(10)]
fn proof_default_entries_field_invariants() {
    let registry = DpdfModelRegistry::default_pipeline();

    // Check each of the 7 known model names.
    let names = [
        "granite_docling",
        "doclayout_yolo",
        "glm_ocr",
        "table_transformer",
        "qwen3_vl",
        "paddle_ocr",
        "firered_ocr",
    ];

    let mut i = 0;
    while i < names.len() {
        let entry = registry.get(names[i]);
        assert!(
            entry.is_some(),
            "expected model must exist in default pipeline"
        );
        let entry = entry.unwrap();
        assert!(!entry.name.is_empty(), "model name must be non-empty");
        assert!(
            entry.parameter_count > 0,
            "model parameter_count must be positive"
        );
        assert!(
            !entry.description.is_empty(),
            "model description must be non-empty"
        );
        i += 1;
    }
}

// ===========================================================================
// Harness 4: get() safety — absent key returns None, never panics
// ===========================================================================

/// SUBSTANTIVE: Proves that `get()` with a key not present in the registry
/// returns `None` rather than panicking. Also proves that `get()` on an
/// empty registry returns `None`.
#[kani::proof]
#[kani::unwind(2)]
fn proof_get_absent_key_returns_none() {
    // Empty registry: any key must return None.
    let empty = DpdfModelRegistry::new();
    assert!(
        empty.get("nonexistent").is_none(),
        "empty registry get() must return None"
    );
    assert!(
        empty.get("").is_none(),
        "empty registry get(\"\") must return None"
    );

    // Populated registry: absent key must return None.
    let registry = DpdfModelRegistry::default_pipeline();
    assert!(
        registry.get("totally_fake_model").is_none(),
        "absent key in default pipeline must return None"
    );
    assert!(
        registry.get("").is_none(),
        "empty string key must return None"
    );
}

// ===========================================================================
// Harness 5: list_by_type() subset — returned entries match requested type
// ===========================================================================

/// SUBSTANTIVE: Proves that `list_by_type()` returns only entries whose
/// `model_type` matches the requested type. Covers all 4 variants on the
/// default pipeline.
#[kani::proof]
#[kani::unwind(10)]
fn proof_list_by_type_returns_correct_subset() {
    let registry = DpdfModelRegistry::default_pipeline();

    let types = [
        ModelType::LayoutDetection,
        ModelType::OCR,
        ModelType::TableStructure,
        ModelType::VLM,
    ];

    let mut t = 0;
    while t < types.len() {
        let entries = registry.list_by_type(types[t]);
        let mut i = 0;
        while i < entries.len() {
            assert_eq!(
                entries[i].model_type, types[t],
                "list_by_type must only return entries of the requested type"
            );
            i += 1;
        }
        t += 1;
    }

    // Verify expected counts for each type.
    assert_eq!(
        registry.list_by_type(ModelType::OCR).len(),
        3,
        "default pipeline must have 3 OCR models"
    );
    assert_eq!(
        registry.list_by_type(ModelType::VLM).len(),
        2,
        "default pipeline must have 2 VLM models"
    );
    assert_eq!(
        registry.list_by_type(ModelType::LayoutDetection).len(),
        1,
        "default pipeline must have 1 LayoutDetection model"
    );
    assert_eq!(
        registry.list_by_type(ModelType::TableStructure).len(),
        1,
        "default pipeline must have 1 TableStructure model"
    );
}

// ===========================================================================
// Harness 6: Register-get roundtrip — register then get returns Some
// ===========================================================================

/// SUBSTANTIVE: Proves that registering a model entry and then looking it up
/// by name returns `Some` with the correct name. Verifies the HashMap
/// insert/lookup contract.
#[kani::proof]
#[kani::unwind(2)]
fn proof_register_get_roundtrip() {
    let mut registry = DpdfModelRegistry::new();
    assert!(registry.is_empty());

    let entry = ModelEntry {
        name: "test_model".into(),
        model_type: ModelType::OCR,
        description: "A test model for proof".into(),
        preprocess_config: DpdfPreprocessConfig::for_granite_docling(),
        parameter_count: 42_000,
    };

    registry.register(entry);
    assert_eq!(
        registry.len(),
        1,
        "registry must have 1 entry after register"
    );

    let retrieved = registry.get("test_model");
    assert!(retrieved.is_some(), "registered model must be retrievable");

    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.name, "test_model", "retrieved name must match");
    assert_eq!(
        retrieved.model_type,
        ModelType::OCR,
        "retrieved type must match"
    );
    assert_eq!(
        retrieved.parameter_count, 42_000,
        "retrieved parameter_count must match"
    );
}

// ===========================================================================
// Harness 7: No duplicate names in default registry
// ===========================================================================

/// SUBSTANTIVE: Proves that the default pipeline has 7 unique model names.
/// Since `register()` uses the name as the HashMap key, duplicates would
/// silently overwrite. We verify all 7 expected names are present and
/// that `len() == 7` (which implies uniqueness).
#[kani::proof]
#[kani::unwind(10)]
fn proof_default_registry_unique_names() {
    let registry = DpdfModelRegistry::default_pipeline();

    let names = [
        "granite_docling",
        "doclayout_yolo",
        "glm_ocr",
        "table_transformer",
        "qwen3_vl",
        "paddle_ocr",
        "firered_ocr",
    ];

    // All 7 names must be present.
    let mut i = 0;
    while i < names.len() {
        assert!(
            registry.get(names[i]).is_some(),
            "expected model name must exist"
        );
        i += 1;
    }

    // len() == 7 with all 7 names present proves uniqueness: if any name
    // collided, len would be < 7 or a name would be missing.
    assert_eq!(registry.len(), 7, "all 7 names must be unique");
}

// ===========================================================================
// Harness 8: All ModelType variants represented in default pipeline
// ===========================================================================

/// SUBSTANTIVE: Proves that every `ModelType` variant has at least one
/// representative model in the default pipeline. Catches variant additions
/// that forget to add a default model.
#[kani::proof]
#[kani::unwind(10)]
fn proof_all_model_types_represented() {
    let registry = DpdfModelRegistry::default_pipeline();

    let types = [
        ModelType::LayoutDetection,
        ModelType::OCR,
        ModelType::TableStructure,
        ModelType::VLM,
    ];

    let mut t = 0;
    while t < types.len() {
        let entries = registry.list_by_type(types[t]);
        assert!(
            !entries.is_empty(),
            "every ModelType variant must have at least one model in default pipeline"
        );
        t += 1;
    }
}

// ===========================================================================
// Harness 9: register() overwrites existing entries
// ===========================================================================

/// SUBSTANTIVE: Proves that registering an entry with a name already in the
/// registry overwrites it (same key, updated value) and does not increase
/// the registry length.
#[kani::proof]
#[kani::unwind(2)]
fn proof_register_overwrites_existing() {
    let mut registry = DpdfModelRegistry::default_pipeline();
    assert_eq!(registry.len(), 7);

    // Original granite_docling has 258M params.
    let original = registry.get("granite_docling").unwrap();
    assert_eq!(original.parameter_count, 258_000_000);

    // Overwrite with different params.
    registry.register(ModelEntry {
        name: "granite_docling".into(),
        model_type: ModelType::VLM,
        description: "Overwritten".into(),
        preprocess_config: DpdfPreprocessConfig::for_granite_docling(),
        parameter_count: 999,
    });

    // Length must not increase.
    assert_eq!(registry.len(), 7, "overwrite must not increase len");

    // Value must be updated.
    let updated = registry.get("granite_docling").unwrap();
    assert_eq!(
        updated.parameter_count, 999,
        "overwrite must update parameter_count"
    );
}

// ===========================================================================
// Harness 10: new() creates an empty registry
// ===========================================================================

/// SUBSTANTIVE: Proves that `DpdfModelRegistry::new()` and
/// `DpdfModelRegistry::default()` both create empty registries with
/// `len() == 0` and `is_empty() == true`.
#[kani::proof]
#[kani::unwind(2)]
fn proof_new_creates_empty_registry() {
    let reg_new = DpdfModelRegistry::new();
    assert!(reg_new.is_empty(), "new() must create empty registry");
    assert_eq!(reg_new.len(), 0, "new() must have len 0");

    let reg_default = DpdfModelRegistry::default();
    assert!(
        reg_default.is_empty(),
        "default() must create empty registry"
    );
    assert_eq!(reg_default.len(), 0, "default() must have len 0");
}
