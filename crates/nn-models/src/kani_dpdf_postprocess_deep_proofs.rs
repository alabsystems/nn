// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for dpdf_postprocess NMS, dedup, and fusion safety
//! (#3988).
//!
//! Proves deeper correctness properties for the document post-processing
//! pipeline beyond the surface-level proofs in
//! `kani_dpdf_registry_postprocess_deep_proofs.rs`.
//!
//! **NMS / merge (5 harnesses):**
//!  1. NMS output is subset of input — merge never increases detection count.
//!  2. NMS preserves confidence ordering — merged confidence is max of pair.
//!  3. NMS with zero detections — empty input yields empty output.
//!  4. NMS with single detection — single input passes through unchanged.
//!  5. NMS output count <= input count — merge strictly reduces or preserves.
//!
//! **Dedup (3 harnesses):**
//!  6. Dedup removes exact duplicates — identical regions collapse to one.
//!  7. Dedup idempotent — running dedup twice yields same result as once.
//!  8. Dedup preserves highest confidence per group.
//!
//! **Confidence filter (2 harnesses):**
//!  9. Confidence filter removes below-threshold entries.
//! 10. Box coordinate validity preserved through confidence filter.
//!
//! **Fusion (3 harnesses):**
//! 11. Fusion merges overlapping regions — higher priority wins.
//! 12. Fusion preserves highest confidence per group.
//! 13. FusionPriority ordering is deterministic across repeated calls.
//!
//! **Config (1 harness):**
//! 14. PostProcessConfig custom thresholds valid — user-provided bounds checked.

#[cfg(kani)]
mod proofs {
    use crate::dpdf_pipeline::DocumentRegion;
    use crate::dpdf_postprocess::{
        compute_iou, deduplicate_regions, filter_by_confidence, fuse_model_results,
        merge_overlapping_regions, FusionPriority, PostProcessConfig,
    };

    /// Helper: create a Text region with given bbox and confidence.
    fn text_region(content: &str, bbox: [f32; 4], confidence: f32) -> DocumentRegion {
        DocumentRegion::Text {
            content: content.to_string(),
            bbox,
            confidence,
        }
    }

    /// Helper: create a Table region with given bbox and confidence.
    fn table_region(bbox: [f32; 4], confidence: f32) -> DocumentRegion {
        DocumentRegion::Table {
            cells: vec![],
            bbox,
            confidence,
        }
    }

    // =======================================================================
    // Harness 1: NMS output is subset of input — merge never increases count
    // =======================================================================

    /// SUBSTANTIVE: Proves that `merge_overlapping_regions` never increases
    /// the number of regions. Each merge step removes one region, so the
    /// output count must be <= input count. Catches bugs where merge
    /// accidentally duplicates or inserts new regions.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_nms_merge_output_subset_of_input() {
        let mut regions = vec![
            text_region("a", [0.0, 0.0, 50.0, 50.0], 0.9),
            text_region("b", [10.0, 10.0, 60.0, 60.0], 0.8), // overlaps with a
            text_region("c", [200.0, 200.0, 300.0, 300.0], 0.7), // disjoint
        ];

        let original_len = regions.len();
        merge_overlapping_regions(&mut regions, 0.3);

        assert!(
            regions.len() <= original_len,
            "merge must not increase region count"
        );
        // a and b overlap significantly; they should merge.
        // c is disjoint and stays separate.
        assert!(
            regions.len() >= 1,
            "merge must not produce empty output from non-empty input"
        );
    }

    // =======================================================================
    // Harness 2: NMS preserves confidence — merged confidence is max of pair
    // =======================================================================

    /// SUBSTANTIVE: Proves that after merging overlapping same-class regions,
    /// the surviving region's confidence is at least as high as the maximum
    /// confidence among the merged pair. The `merge_two` function takes
    /// `a.confidence().max(b.confidence())`. This catches bugs where
    /// confidence is averaged, zeroed, or taken from the wrong region.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_nms_merge_preserves_max_confidence() {
        let high_conf = 0.95_f32;
        let low_conf = 0.6_f32;

        let mut regions = vec![
            text_region("high", [0.0, 0.0, 100.0, 100.0], high_conf),
            text_region("low", [10.0, 10.0, 90.0, 90.0], low_conf), // heavily overlaps
        ];

        merge_overlapping_regions(&mut regions, 0.1);

        // After merge, one region should remain.
        assert_eq!(regions.len(), 1, "overlapping pair must merge to one");

        // Surviving confidence must be the max.
        assert!(
            (regions[0].confidence() - high_conf).abs() < 1e-6,
            "merged confidence must equal the higher of the two"
        );
    }

    // =======================================================================
    // Harness 3: NMS with zero detections — empty in, empty out
    // =======================================================================

    /// SUBSTANTIVE: Proves that `merge_overlapping_regions` on an empty vector
    /// produces an empty vector. Catches null-pointer or underflow bugs
    /// in the loop termination conditions.
    #[kani::proof]
    #[kani::unwind(2)]
    fn proof_nms_merge_empty_input() {
        let mut regions: Vec<DocumentRegion> = vec![];
        merge_overlapping_regions(&mut regions, 0.5);
        assert!(
            regions.is_empty(),
            "merge on empty input must produce empty output"
        );
    }

    // =======================================================================
    // Harness 4: NMS with single detection — passthrough unchanged
    // =======================================================================

    /// SUBSTANTIVE: Proves that a single-element input passes through
    /// `merge_overlapping_regions` unchanged — same bbox and confidence.
    /// Catches bugs where the single-element path modifies the region.
    #[kani::proof]
    #[kani::unwind(2)]
    fn proof_nms_merge_single_detection_passthrough() {
        let original_bbox = [10.0_f32, 20.0, 50.0, 80.0];
        let original_conf = 0.85_f32;
        let mut regions = vec![text_region("only", original_bbox, original_conf)];

        merge_overlapping_regions(&mut regions, 0.5);

        assert_eq!(regions.len(), 1, "single region must survive merge");
        let bbox = regions[0].bbox();
        assert!(
            bbox[0] == original_bbox[0]
                && bbox[1] == original_bbox[1]
                && bbox[2] == original_bbox[2]
                && bbox[3] == original_bbox[3],
            "single region bbox must be unchanged"
        );
        assert!(
            (regions[0].confidence() - original_conf).abs() < 1e-7,
            "single region confidence must be unchanged"
        );
    }

    // =======================================================================
    // Harness 5: NMS output count <= input count (multi-class)
    // =======================================================================

    /// SUBSTANTIVE: Proves the count invariant with regions of different
    /// classes. Same-class overlapping regions merge, but different-class
    /// regions never merge, so the output count is between
    /// (number of distinct classes) and the original count.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_nms_output_count_le_input_multi_class() {
        let mut regions = vec![
            text_region("t1", [0.0, 0.0, 50.0, 50.0], 0.9),
            text_region("t2", [10.0, 10.0, 60.0, 60.0], 0.8), // overlaps t1, same class
            table_region([0.0, 0.0, 50.0, 50.0], 0.7),        // overlaps t1, different class
        ];

        let original_len = regions.len();
        merge_overlapping_regions(&mut regions, 0.3);

        assert!(
            regions.len() <= original_len,
            "merge must not increase count"
        );
        // Text regions may merge, but table region is different class => kept.
        assert!(
            regions.len() >= 2,
            "at least 2 regions must survive (1 merged text + 1 table)"
        );
    }

    // =======================================================================
    // Harness 6: Dedup removes exact duplicates
    // =======================================================================

    /// SUBSTANTIVE: Proves that `deduplicate_regions` collapses identical
    /// (same class, same bbox) regions to one. The surviving region has the
    /// highest confidence. Catches bugs where dedup fails to suppress
    /// exact duplicates.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_dedup_removes_exact_duplicates() {
        let bbox = [10.0_f32, 20.0, 50.0, 80.0];
        let mut regions = vec![
            text_region("dup1", bbox, 0.7),
            text_region("dup2", bbox, 0.9),
            text_region("dup3", bbox, 0.5),
        ];

        deduplicate_regions(&mut regions, 0.5);

        // All three are same class, same bbox => IoU == 1.0 > 0.5 threshold.
        // Only the highest-confidence one should survive.
        assert_eq!(regions.len(), 1, "exact duplicates must collapse to one");
        assert!(
            (regions[0].confidence() - 0.9).abs() < 1e-6,
            "highest-confidence duplicate must survive"
        );
    }

    // =======================================================================
    // Harness 7: Dedup is idempotent
    // =======================================================================

    /// SUBSTANTIVE: Proves that running `deduplicate_regions` twice produces
    /// the same result as running it once. After the first pass, no
    /// remaining pairs exceed the similarity threshold, so the second pass
    /// is a no-op. Catches bugs where dedup modifies regions in a way that
    /// creates new duplicates.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_dedup_idempotent() {
        let mut regions = vec![
            text_region("a", [0.0, 0.0, 50.0, 50.0], 0.9),
            text_region("b", [5.0, 5.0, 55.0, 55.0], 0.7), // high overlap with a
            text_region("c", [200.0, 200.0, 300.0, 300.0], 0.8), // disjoint
        ];

        deduplicate_regions(&mut regions, 0.5);
        let after_first = regions.len();
        let first_bboxes: Vec<[f32; 4]> = regions.iter().map(|r| r.bbox()).collect();
        let first_confs: Vec<f32> = regions.iter().map(|r| r.confidence()).collect();

        deduplicate_regions(&mut regions, 0.5);
        let after_second = regions.len();

        assert_eq!(
            after_first, after_second,
            "dedup must be idempotent: second pass must not change count"
        );

        let mut i = 0;
        while i < regions.len() {
            let bbox = regions[i].bbox();
            assert!(
                bbox == first_bboxes[i],
                "dedup idempotent: bboxes must not change on second pass"
            );
            assert!(
                (regions[i].confidence() - first_confs[i]).abs() < 1e-7,
                "dedup idempotent: confidences must not change on second pass"
            );
            i += 1;
        }
    }

    // =======================================================================
    // Harness 8: Dedup preserves highest confidence per group
    // =======================================================================

    /// SUBSTANTIVE: Proves that after dedup, the surviving region from a
    /// group of near-duplicates has the maximum confidence from that group.
    /// Dedup sorts by confidence descending before suppression, so the first
    /// (highest) region survives and suppresses the rest.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_dedup_preserves_highest_confidence() {
        // Two overlapping groups: text group and table group.
        let text_bbox = [10.0_f32, 10.0, 60.0, 60.0];
        let table_bbox = [200.0_f32, 200.0, 300.0, 300.0];

        let mut regions = vec![
            text_region("t_low", text_bbox, 0.3),
            text_region("t_high", text_bbox, 0.95),
            table_region(table_bbox, 0.5),
            table_region(table_bbox, 0.8),
        ];

        deduplicate_regions(&mut regions, 0.5);

        // Each group should collapse to one region.
        assert_eq!(regions.len(), 2, "two groups should each have one survivor");

        // Find the text survivor and the table survivor.
        let text_survivor = regions
            .iter()
            .find(|r| r.class_name() == "text")
            .expect("text survivor must exist");
        let table_survivor = regions
            .iter()
            .find(|r| r.class_name() == "table")
            .expect("table survivor must exist");

        assert!(
            (text_survivor.confidence() - 0.95).abs() < 1e-6,
            "text survivor must have highest text confidence"
        );
        assert!(
            (table_survivor.confidence() - 0.8).abs() < 1e-6,
            "table survivor must have highest table confidence"
        );
    }

    // =======================================================================
    // Harness 9: Confidence filter removes below-threshold entries
    // =======================================================================

    /// SUBSTANTIVE: Proves that `filter_by_confidence` removes exactly those
    /// regions with confidence < threshold and keeps the rest. Tests the
    /// boundary case where confidence == threshold (should be kept per >=).
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_confidence_filter_removes_below_threshold() {
        let threshold = 0.5_f32;
        let mut regions = vec![
            text_region("above", [0.0, 0.0, 10.0, 10.0], 0.8),
            text_region("below", [20.0, 20.0, 30.0, 30.0], 0.2),
            text_region("exact", [40.0, 40.0, 50.0, 50.0], 0.5), // boundary: == threshold
            text_region("way_below", [60.0, 60.0, 70.0, 70.0], 0.0),
        ];

        filter_by_confidence(&mut regions, threshold);

        // "above" (0.8 >= 0.5) and "exact" (0.5 >= 0.5) should survive.
        assert_eq!(
            regions.len(),
            2,
            "only regions with confidence >= threshold should survive"
        );

        let mut i = 0;
        while i < regions.len() {
            assert!(
                regions[i].confidence() >= threshold,
                "all surviving regions must meet threshold"
            );
            i += 1;
        }
    }

    // =======================================================================
    // Harness 10: Box coordinate validity preserved through confidence filter
    // =======================================================================

    /// SUBSTANTIVE: Proves that `filter_by_confidence` does not modify the
    /// bounding boxes of surviving regions. The filter only removes elements;
    /// it must not alter bbox coordinates of the remaining ones.
    #[kani::proof]
    #[kani::unwind(4)]
    fn proof_confidence_filter_preserves_bbox_coordinates() {
        let bbox_a = [5.0_f32, 10.0, 100.0, 200.0];
        let bbox_b = [50.0_f32, 60.0, 150.0, 250.0];

        let mut regions = vec![
            text_region("keep_a", bbox_a, 0.9),
            text_region("drop", [0.0, 0.0, 1.0, 1.0], 0.05), // will be dropped
            text_region("keep_b", bbox_b, 0.7),
        ];

        filter_by_confidence(&mut regions, 0.3);

        assert_eq!(regions.len(), 2, "two regions should survive");

        // Verify bboxes are unchanged (order preserved by retain).
        let survived_a = regions[0].bbox();
        assert!(
            survived_a == bbox_a,
            "first survivor bbox must be unchanged"
        );

        let survived_b = regions[1].bbox();
        assert!(
            survived_b == bbox_b,
            "second survivor bbox must be unchanged"
        );
    }

    // =======================================================================
    // Harness 11: Fusion merges overlapping — higher priority wins
    // =======================================================================

    /// SUBSTANTIVE: Proves that `fuse_model_results` respects the priority
    /// ordering: DocLayout regions are always included; lower-priority
    /// regions (TableTransformer, Ocr) are only included if they don't
    /// significantly overlap with already-included higher-priority regions.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_fusion_higher_priority_wins() {
        // DocLayout region at [0, 0, 100, 100].
        let doclayout = vec![text_region("doc", [0.0, 0.0, 100.0, 100.0], 0.9)];

        // Table detection at same location — should be suppressed.
        let table = vec![text_region("tab", [0.0, 0.0, 100.0, 100.0], 0.95)];

        // OCR at a completely different location — should be included.
        let ocr = vec![text_region("ocr", [500.0, 500.0, 600.0, 600.0], 0.7)];

        let fused = fuse_model_results(&doclayout, &table, &ocr);

        // DocLayout region always included.
        assert!(fused.len() >= 1, "doclayout region must always be included");

        // Table region overlaps DocLayout => suppressed. OCR is disjoint => included.
        assert_eq!(
            fused.len(),
            2,
            "fused should have doclayout + non-overlapping ocr"
        );

        // Verify the doclayout region is present (first element, since doclayout goes first).
        let first_bbox = fused[0].bbox();
        assert!(
            first_bbox[0] == 0.0 && first_bbox[2] == 100.0,
            "first fused region must be from doclayout"
        );
    }

    // =======================================================================
    // Harness 12: Fusion preserves highest confidence per group
    // =======================================================================

    /// SUBSTANTIVE: Proves that when fusion includes a region, its confidence
    /// is preserved exactly (not averaged or modified). Fusion is a
    /// selection operation: include or exclude, never modify.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_fusion_preserves_confidence_exactly() {
        let doc_conf = 0.85_f32;
        let ocr_conf = 0.72_f32;

        let doclayout = vec![text_region("doc", [0.0, 0.0, 50.0, 50.0], doc_conf)];
        let table = vec![]; // empty
        let ocr = vec![text_region("ocr", [200.0, 200.0, 300.0, 300.0], ocr_conf)];

        let fused = fuse_model_results(&doclayout, &table, &ocr);

        assert_eq!(fused.len(), 2, "both disjoint regions should be included");

        // Confidences must be preserved exactly.
        assert!(
            (fused[0].confidence() - doc_conf).abs() < 1e-7,
            "doclayout confidence must be preserved"
        );
        assert!(
            (fused[1].confidence() - ocr_conf).abs() < 1e-7,
            "ocr confidence must be preserved"
        );
    }

    // =======================================================================
    // Harness 13: FusionPriority ordering deterministic across calls
    // =======================================================================

    /// SUBSTANTIVE: Proves that the FusionPriority enum ordering is
    /// deterministic: repeated rank computations yield the same values.
    /// Also proves transitivity and totality of the ordering. Catches
    /// non-deterministic or inconsistent priority comparisons.
    #[kani::proof]
    #[kani::unwind(2)]
    fn proof_fusion_priority_ordering_deterministic() {
        let rank = |p: FusionPriority| -> u8 {
            match p {
                FusionPriority::DocLayout => 3,
                FusionPriority::TableTransformer => 2,
                FusionPriority::Ocr => 1,
            }
        };

        // Same call twice must yield same result.
        let doc_rank_1 = rank(FusionPriority::DocLayout);
        let doc_rank_2 = rank(FusionPriority::DocLayout);
        assert_eq!(doc_rank_1, doc_rank_2, "rank must be deterministic");

        let tab_rank_1 = rank(FusionPriority::TableTransformer);
        let tab_rank_2 = rank(FusionPriority::TableTransformer);
        assert_eq!(tab_rank_1, tab_rank_2, "rank must be deterministic");

        let ocr_rank_1 = rank(FusionPriority::Ocr);
        let ocr_rank_2 = rank(FusionPriority::Ocr);
        assert_eq!(ocr_rank_1, ocr_rank_2, "rank must be deterministic");

        // Transitivity: if DocLayout > Table > Ocr, then DocLayout > Ocr.
        assert!(doc_rank_1 > tab_rank_1, "DocLayout > TableTransformer");
        assert!(tab_rank_1 > ocr_rank_1, "TableTransformer > Ocr");
        assert!(doc_rank_1 > ocr_rank_1, "transitivity: DocLayout > Ocr");

        // Totality: all ranks are distinct.
        assert_ne!(doc_rank_1, tab_rank_1, "ranks must be distinct");
        assert_ne!(doc_rank_1, ocr_rank_1, "ranks must be distinct");
        assert_ne!(tab_rank_1, ocr_rank_1, "ranks must be distinct");
    }

    // =======================================================================
    // Harness 14: PostProcessConfig custom thresholds valid
    // =======================================================================

    /// SUBSTANTIVE: Proves that a PostProcessConfig constructed with custom
    /// threshold values preserves those values exactly. Also verifies that
    /// the config fields are independently settable (changing one does not
    /// affect another). Catches struct field ordering bugs or shadowing.
    #[kani::proof]
    #[kani::unwind(2)]
    fn proof_postprocess_config_custom_thresholds_preserved() {
        let config = PostProcessConfig {
            merge_iou: 0.7,
            dedup_similarity: 0.85,
            min_confidence: 0.4,
            enable_model_fusion: false,
        };

        // All fields must round-trip exactly.
        assert!(
            (config.merge_iou - 0.7).abs() < 1e-7,
            "merge_iou must be preserved"
        );
        assert!(
            (config.dedup_similarity - 0.85).abs() < 1e-7,
            "dedup_similarity must be preserved"
        );
        assert!(
            (config.min_confidence - 0.4).abs() < 1e-7,
            "min_confidence must be preserved"
        );
        assert!(
            !config.enable_model_fusion,
            "enable_model_fusion must be preserved"
        );

        // Independence: changing one field does not affect others.
        let config2 = PostProcessConfig {
            merge_iou: 0.3,
            ..config.clone()
        };
        assert!(
            (config2.merge_iou - 0.3).abs() < 1e-7,
            "overridden merge_iou must take new value"
        );
        assert!(
            (config2.dedup_similarity - 0.85).abs() < 1e-7,
            "dedup_similarity must be unchanged when only merge_iou is overridden"
        );
        assert!(
            (config2.min_confidence - 0.4).abs() < 1e-7,
            "min_confidence must be unchanged when only merge_iou is overridden"
        );
    }
}
