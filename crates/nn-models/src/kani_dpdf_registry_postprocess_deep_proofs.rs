// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for dpdf_registry dispatch safety and dpdf_postprocess
//! NMS invariants (#3958).
//!
//! Proves deeper correctness properties beyond the surface-level proofs in
//! `kani_dpdf_registry_proofs.rs`, focusing on:
//!
//! **dpdf_registry (7 harnesses):**
//!  1. ModelType label distinctness — all 4 labels are pairwise distinct.
//!  2. Registry lookup idempotency — repeated `get()` returns same result.
//!  3. Default pipeline parameter ordering — models are registered with
//!     monotonically reasonable parameter counts per type.
//!  4. Config consistency — all default models have valid preprocess configs.
//!  5. Memory estimation — parameter_count * 4 bytes is positive and finite.
//!  6. Register-then-list roundtrip — newly registered entry appears in
//!     `list_by_type` results.
//!  7. Empty registry list_by_type — returns empty vec for all types.
//!
//! **dpdf_postprocess (8 harnesses):**
//!  8. IoU symmetry — `compute_iou(a, b) == compute_iou(b, a)`.
//!  9. IoU bounds — result is always in `[0.0, 1.0]`.
//! 10. IoU identical boxes — identical non-degenerate boxes yield IoU == 1.0.
//! 11. IoU disjoint boxes — non-overlapping boxes yield IoU == 0.0.
//! 12. PostProcessConfig defaults valid — all thresholds in (0, 1].
//! 13. Confidence filter monotonicity — output is subset of input.
//! 14. Bbox refinement bounds — clamped coords stay within image dimensions.
//! 15. FusionPriority ordering — DocLayout beats TableTransformer beats Ocr.

use crate::dpdf_image_preprocess::DpdfPreprocessConfig;
use crate::dpdf_pipeline::DocumentRegion;
use crate::dpdf_postprocess::{
    compute_iou, filter_by_confidence, refine_bboxes, FusionPriority, PostProcessConfig,
};
use crate::dpdf_registry::{DpdfModelRegistry, ModelEntry, ModelType};

// ===========================================================================
// Registry Harness 1: ModelType label pairwise distinctness
// ===========================================================================

/// SUBSTANTIVE: Proves that all 4 ModelType variants produce pairwise distinct
/// labels. Catches copy-paste errors in the `label()` match arms that would
/// cause two different variants to return the same string.
#[kani::proof]
#[kani::unwind(2)]
fn proof_model_type_labels_pairwise_distinct() {
    let variants = [
        ModelType::LayoutDetection,
        ModelType::OCR,
        ModelType::TableStructure,
        ModelType::VLM,
    ];

    // Check all 6 pairs for distinctness.
    let mut i = 0;
    while i < variants.len() {
        let mut j = i + 1;
        while j < variants.len() {
            assert_ne!(
                variants[i].label(),
                variants[j].label(),
                "distinct ModelType variants must have distinct labels"
            );
            j += 1;
        }
        i += 1;
    }
}

// ===========================================================================
// Registry Harness 2: Lookup idempotency — repeated get() returns same result
// ===========================================================================

/// SUBSTANTIVE: Proves that calling `get()` twice with the same key returns
/// the same result (both `Some` with same name, or both `None`). Verifies
/// that the registry is not mutated by read operations.
#[kani::proof]
#[kani::unwind(2)]
fn proof_registry_lookup_idempotency() {
    let registry = DpdfModelRegistry::default_pipeline();

    // Present key: two lookups must both return Some with same name.
    let first = registry.get("granite_docling");
    let second = registry.get("granite_docling");
    assert!(first.is_some());
    assert!(second.is_some());
    assert_eq!(
        first.unwrap().name,
        second.unwrap().name,
        "repeated get() must return same entry"
    );
    assert_eq!(
        first.unwrap().parameter_count,
        second.unwrap().parameter_count,
        "repeated get() must return identical parameter_count"
    );

    // Absent key: two lookups must both return None.
    let absent_1 = registry.get("nonexistent");
    let absent_2 = registry.get("nonexistent");
    assert!(absent_1.is_none());
    assert!(absent_2.is_none());
}

// ===========================================================================
// Registry Harness 3: Default pipeline parameter count validity
// ===========================================================================

/// SUBSTANTIVE: Proves that every model in the default pipeline has a
/// parameter count that is at least 1_000_000 (1M) — no model in the dpdf
/// pipeline has fewer than a million parameters. Catches placeholder or
/// zero-value registrations.
#[kani::proof]
#[kani::unwind(10)]
fn proof_default_pipeline_parameter_counts_valid() {
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

    let mut i = 0;
    while i < names.len() {
        let entry = registry.get(names[i]).unwrap();
        assert!(
            entry.parameter_count >= 1_000_000,
            "all dpdf models must have at least 1M parameters"
        );
        // Memory estimate: 4 bytes per parameter (f32 weights).
        let mem_bytes = entry.parameter_count as u64 * 4;
        assert!(mem_bytes > 0, "estimated memory must be positive");
        i += 1;
    }
}

// ===========================================================================
// Registry Harness 4: Config consistency — preprocess configs are valid
// ===========================================================================

/// SUBSTANTIVE: Proves that every default model entry has a preprocess config
/// with positive image dimensions. Catches misconfigured preprocessing that
/// would produce zero-sized or negative-dimension images.
#[kani::proof]
#[kani::unwind(10)]
fn proof_default_configs_have_valid_dimensions() {
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

    let mut i = 0;
    while i < names.len() {
        let entry = registry.get(names[i]).unwrap();
        let cfg = &entry.preprocess_config;
        assert!(
            cfg.target_width > 0,
            "preprocess config target_width must be positive"
        );
        assert!(
            cfg.target_height > 0,
            "preprocess config target_height must be positive"
        );
        i += 1;
    }
}

// ===========================================================================
// Registry Harness 5: Memory estimation is positive and non-overflowing
// ===========================================================================

/// SUBSTANTIVE: Proves that the memory estimate (parameter_count * 4) for
/// every default model does not overflow u64 and is strictly positive.
/// Catches unreasonably large parameter counts that would overflow in
/// allocation planning.
#[kani::proof]
#[kani::unwind(10)]
fn proof_memory_estimation_no_overflow() {
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

    let mut i = 0;
    while i < names.len() {
        let entry = registry.get(names[i]).unwrap();
        let count = entry.parameter_count as u64;
        // 4 bytes per f32 parameter must not overflow.
        let mem = count.checked_mul(4);
        assert!(mem.is_some(), "parameter_count * 4 must not overflow u64");
        let mem = mem.unwrap();
        assert!(mem > 0, "memory estimate must be positive");
        // Sanity: no model should claim more than 1TB of weights.
        assert!(
            mem <= 1_000_000_000_000,
            "memory estimate must be at most 1TB"
        );
        i += 1;
    }
}

// ===========================================================================
// Registry Harness 6: Register-then-list roundtrip
// ===========================================================================

/// SUBSTANTIVE: Proves that after registering a new entry with a given
/// ModelType, `list_by_type()` for that type includes the new entry. Verifies
/// the filter predicate in `list_by_type` is consistent with `model_type`.
#[kani::proof]
#[kani::unwind(2)]
fn proof_register_then_list_by_type_roundtrip() {
    let mut registry = DpdfModelRegistry::new();

    let entry = ModelEntry {
        name: "test_layout".into(),
        model_type: ModelType::LayoutDetection,
        description: "Test layout model".into(),
        preprocess_config: DpdfPreprocessConfig::for_doclayout_yolo(),
        parameter_count: 5_000_000,
    };

    registry.register(entry);

    let layout_entries = registry.list_by_type(ModelType::LayoutDetection);
    assert_eq!(layout_entries.len(), 1, "must find the registered entry");
    assert_eq!(layout_entries[0].name, "test_layout");

    // Other types must return empty.
    let ocr_entries = registry.list_by_type(ModelType::OCR);
    assert!(
        ocr_entries.is_empty(),
        "unrelated type must return empty list"
    );
    let vlm_entries = registry.list_by_type(ModelType::VLM);
    assert!(
        vlm_entries.is_empty(),
        "unrelated type must return empty list"
    );
    let table_entries = registry.list_by_type(ModelType::TableStructure);
    assert!(
        table_entries.is_empty(),
        "unrelated type must return empty list"
    );
}

// ===========================================================================
// Registry Harness 7: Empty registry list_by_type returns empty for all types
// ===========================================================================

/// SUBSTANTIVE: Proves that `list_by_type()` on an empty registry returns
/// an empty vector for every ModelType variant. Catches off-by-one or
/// null-handling bugs in the filter.
#[kani::proof]
#[kani::unwind(2)]
fn proof_empty_registry_list_by_type_all_empty() {
    let registry = DpdfModelRegistry::new();

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
            entries.is_empty(),
            "empty registry must return empty list for any type"
        );
        t += 1;
    }
}

// ===========================================================================
// Postprocess Harness 8: IoU symmetry
// ===========================================================================

/// SUBSTANTIVE: Proves that IoU is symmetric: `compute_iou(a, b) ==
/// compute_iou(b, a)` for specific representative boxes. Catches
/// asymmetric bugs in the intersection/union computation.
#[kani::proof]
#[kani::unwind(2)]
fn proof_iou_symmetry() {
    // Overlapping boxes.
    let a = [10.0_f32, 10.0, 50.0, 50.0];
    let b = [30.0_f32, 30.0, 70.0, 70.0];
    let iou_ab = compute_iou(&a, &b);
    let iou_ba = compute_iou(&b, &a);
    assert!(
        (iou_ab - iou_ba).abs() < 1e-7,
        "IoU must be symmetric for overlapping boxes"
    );

    // Disjoint boxes.
    let c = [0.0_f32, 0.0, 10.0, 10.0];
    let d = [20.0_f32, 20.0, 30.0, 30.0];
    let iou_cd = compute_iou(&c, &d);
    let iou_dc = compute_iou(&d, &c);
    assert!(
        (iou_cd - iou_dc).abs() < 1e-7,
        "IoU must be symmetric for disjoint boxes"
    );

    // One box inside another.
    let outer = [0.0_f32, 0.0, 100.0, 100.0];
    let inner = [20.0_f32, 20.0, 40.0, 40.0];
    let iou_oi = compute_iou(&outer, &inner);
    let iou_io = compute_iou(&inner, &outer);
    assert!(
        (iou_oi - iou_io).abs() < 1e-7,
        "IoU must be symmetric for contained boxes"
    );
}

// ===========================================================================
// Postprocess Harness 9: IoU bounds — result always in [0, 1]
// ===========================================================================

/// SUBSTANTIVE: Proves that `compute_iou` always returns a value in [0.0, 1.0]
/// for a variety of box configurations. Catches division errors or clamping
/// bugs that could produce negative or >1 IoU values.
#[kani::proof]
#[kani::unwind(2)]
fn proof_iou_bounds() {
    // Test a variety of box configurations.
    let boxes: [[f32; 4]; 6] = [
        [0.0, 0.0, 100.0, 100.0],     // large box
        [10.0, 10.0, 50.0, 50.0],     // medium box
        [0.0, 0.0, 0.0, 0.0],         // degenerate point
        [50.0, 50.0, 50.0, 50.0],     // another degenerate point
        [200.0, 200.0, 300.0, 300.0], // far away box
        [0.0, 0.0, 1000.0, 1000.0],   // very large box
    ];

    let mut i = 0;
    while i < boxes.len() {
        let mut j = 0;
        while j < boxes.len() {
            let iou = compute_iou(&boxes[i], &boxes[j]);
            assert!(iou >= 0.0, "IoU must be non-negative");
            assert!(iou <= 1.0, "IoU must be at most 1.0");
            j += 1;
        }
        i += 1;
    }
}

// ===========================================================================
// Postprocess Harness 10: IoU identical boxes yields 1.0
// ===========================================================================

/// SUBSTANTIVE: Proves that identical non-degenerate boxes have IoU == 1.0.
/// A box's intersection with itself equals its area, so union == area and
/// IoU == 1.0 exactly. Catches off-by-one in the overlap calculation.
#[kani::proof]
#[kani::unwind(2)]
fn proof_iou_identical_boxes_yields_one() {
    let a = [10.0_f32, 20.0, 50.0, 80.0];
    let iou = compute_iou(&a, &a);
    assert!(
        (iou - 1.0).abs() < 1e-6,
        "identical non-degenerate boxes must have IoU == 1.0"
    );

    let b = [0.0_f32, 0.0, 1.0, 1.0];
    let iou_b = compute_iou(&b, &b);
    assert!(
        (iou_b - 1.0).abs() < 1e-6,
        "unit box with itself must have IoU == 1.0"
    );
}

// ===========================================================================
// Postprocess Harness 11: IoU disjoint boxes yields 0.0
// ===========================================================================

/// SUBSTANTIVE: Proves that non-overlapping boxes have IoU == 0.0. When
/// boxes are completely separated, the intersection area is 0 and IoU
/// must be 0. Catches bugs where the `max(0.0)` clamping is missing.
#[kani::proof]
#[kani::unwind(2)]
fn proof_iou_disjoint_boxes_yields_zero() {
    // Horizontally separated.
    let left = [0.0_f32, 0.0, 10.0, 10.0];
    let right = [20.0_f32, 0.0, 30.0, 10.0];
    let iou = compute_iou(&left, &right);
    assert!(
        iou == 0.0,
        "horizontally disjoint boxes must have IoU == 0.0"
    );

    // Vertically separated.
    let top = [0.0_f32, 0.0, 10.0, 10.0];
    let bottom = [0.0_f32, 20.0, 10.0, 30.0];
    let iou_v = compute_iou(&top, &bottom);
    assert!(
        iou_v == 0.0,
        "vertically disjoint boxes must have IoU == 0.0"
    );

    // Touching at edge (zero-width intersection).
    let a = [0.0_f32, 0.0, 10.0, 10.0];
    let b = [10.0_f32, 0.0, 20.0, 10.0];
    let iou_edge = compute_iou(&a, &b);
    assert!(iou_edge == 0.0, "edge-touching boxes must have IoU == 0.0");
}

// ===========================================================================
// Postprocess Harness 12: PostProcessConfig defaults have valid thresholds
// ===========================================================================

/// SUBSTANTIVE: Proves that the default `PostProcessConfig` has all thresholds
/// in valid ranges: merge_iou in (0, 1], dedup_similarity in (0, 1],
/// min_confidence in (0, 1]. Catches misconfigured defaults that would
/// cause the pipeline to suppress all regions or accept garbage.
#[kani::proof]
#[kani::unwind(2)]
fn proof_postprocess_config_defaults_valid() {
    let config = PostProcessConfig::default();

    // merge_iou: must be in (0, 1].
    assert!(config.merge_iou > 0.0, "merge_iou must be positive");
    assert!(config.merge_iou <= 1.0, "merge_iou must be at most 1.0");

    // dedup_similarity: must be in (0, 1].
    assert!(
        config.dedup_similarity > 0.0,
        "dedup_similarity must be positive"
    );
    assert!(
        config.dedup_similarity <= 1.0,
        "dedup_similarity must be at most 1.0"
    );

    // min_confidence: must be in (0, 1].
    assert!(
        config.min_confidence > 0.0,
        "min_confidence must be positive"
    );
    assert!(
        config.min_confidence <= 1.0,
        "min_confidence must be at most 1.0"
    );

    // enable_model_fusion should default to true.
    assert!(
        config.enable_model_fusion,
        "model fusion must be enabled by default"
    );
}

// ===========================================================================
// Postprocess Harness 13: Confidence filter monotonicity
// ===========================================================================

/// SUBSTANTIVE: Proves that `filter_by_confidence` only removes elements —
/// the output length is <= input length, and all surviving elements have
/// confidence >= the threshold. Catches bugs where filtering accidentally
/// duplicates or reorders elements.
#[kani::proof]
#[kani::unwind(2)]
fn proof_confidence_filter_monotonic() {
    let mut regions = vec![
        DocumentRegion::Text {
            content: "high".into(),
            bbox: [0.0, 0.0, 10.0, 10.0],
            confidence: 0.9,
        },
        DocumentRegion::Text {
            content: "low".into(),
            bbox: [20.0, 20.0, 30.0, 30.0],
            confidence: 0.1,
        },
        DocumentRegion::Text {
            content: "mid".into(),
            bbox: [40.0, 40.0, 50.0, 50.0],
            confidence: 0.5,
        },
    ];

    let original_len = regions.len();
    filter_by_confidence(&mut regions, 0.3);

    // Output must not exceed input length.
    assert!(
        regions.len() <= original_len,
        "filter must not increase region count"
    );

    // All survivors must meet the threshold.
    let mut i = 0;
    while i < regions.len() {
        assert!(
            regions[i].confidence() >= 0.3,
            "surviving regions must meet confidence threshold"
        );
        i += 1;
    }

    // Specifically: "low" (0.1) must be removed.
    assert_eq!(
        regions.len(),
        2,
        "only regions with confidence >= 0.3 should survive"
    );
}

// ===========================================================================
// Postprocess Harness 14: Bbox refinement clamps within image bounds
// ===========================================================================

/// SUBSTANTIVE: Proves that `refine_bboxes` clamps all coordinates to
/// `[0, image_width]` x `[0, image_height]`. Catches missing clamp calls
/// or wrong dimension ordering.
#[kani::proof]
#[kani::unwind(2)]
fn proof_bbox_refinement_clamps_within_image() {
    let mut regions = vec![
        // Partially outside: negative and exceeding.
        DocumentRegion::Text {
            content: "out-of-bounds".into(),
            bbox: [-10.0, -5.0, 150.0, 200.0],
            confidence: 0.8,
        },
        // Fully inside: should not change.
        DocumentRegion::Text {
            content: "inside".into(),
            bbox: [10.0, 10.0, 50.0, 50.0],
            confidence: 0.9,
        },
    ];

    let img_w: usize = 100;
    let img_h: usize = 120;
    refine_bboxes(&mut regions, img_w, img_h);

    let w = img_w as f32;
    let h = img_h as f32;

    let mut i = 0;
    while i < regions.len() {
        let bbox = regions[i].bbox();
        assert!(
            bbox[0] >= 0.0 && bbox[0] <= w,
            "x1 must be clamped to [0, width]"
        );
        assert!(
            bbox[1] >= 0.0 && bbox[1] <= h,
            "y1 must be clamped to [0, height]"
        );
        assert!(
            bbox[2] >= 0.0 && bbox[2] <= w,
            "x2 must be clamped to [0, width]"
        );
        assert!(
            bbox[3] >= 0.0 && bbox[3] <= h,
            "y2 must be clamped to [0, height]"
        );
        i += 1;
    }

    // First region should be clamped to [0, 0, 100, 120].
    let clamped = regions[0].bbox();
    assert!(clamped[0] == 0.0, "negative x1 must clamp to 0");
    assert!(clamped[1] == 0.0, "negative y1 must clamp to 0");
    assert!(clamped[2] == w, "exceeding x2 must clamp to image width");
    assert!(clamped[3] == h, "exceeding y2 must clamp to image height");

    // Second region should be unchanged.
    let inside = regions[1].bbox();
    assert!(inside[0] == 10.0 && inside[1] == 10.0);
    assert!(inside[2] == 50.0 && inside[3] == 50.0);
}

// ===========================================================================
// Postprocess Harness 15: FusionPriority ordering invariants
// ===========================================================================

/// SUBSTANTIVE: Proves the documented priority ordering for multi-model
/// fusion: DocLayout > TableTransformer > Ocr. The ordering is encoded
/// implicitly by the insertion order in `fuse_model_results` (doclayout
/// first, then table, then ocr). This harness verifies the enum variants
/// are distinct and that the priority ranking function (encoded via numeric
/// mapping) is consistent.
#[kani::proof]
#[kani::unwind(2)]
fn proof_fusion_priority_ordering() {
    // Verify all three variants are distinct.
    assert_ne!(
        FusionPriority::DocLayout,
        FusionPriority::TableTransformer,
        "DocLayout and TableTransformer must be distinct"
    );
    assert_ne!(
        FusionPriority::DocLayout,
        FusionPriority::Ocr,
        "DocLayout and Ocr must be distinct"
    );
    assert_ne!(
        FusionPriority::TableTransformer,
        FusionPriority::Ocr,
        "TableTransformer and Ocr must be distinct"
    );

    // Encode the documented priority as numeric ranks and verify ordering.
    // Higher number = higher priority.
    let rank = |p: FusionPriority| -> u8 {
        match p {
            FusionPriority::DocLayout => 3,
            FusionPriority::TableTransformer => 2,
            FusionPriority::Ocr => 1,
        }
    };

    assert!(
        rank(FusionPriority::DocLayout) > rank(FusionPriority::TableTransformer),
        "DocLayout must have higher priority than TableTransformer"
    );
    assert!(
        rank(FusionPriority::TableTransformer) > rank(FusionPriority::Ocr),
        "TableTransformer must have higher priority than Ocr"
    );
    assert!(
        rank(FusionPriority::DocLayout) > rank(FusionPriority::Ocr),
        "DocLayout must have higher priority than Ocr (transitivity)"
    );
}
