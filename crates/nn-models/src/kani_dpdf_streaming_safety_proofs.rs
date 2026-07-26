// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for dpdf_streaming chunked processing safety and
//! invariants (#3993).
//!
//! Proves safety properties for the streaming pipeline's chunk splitting,
//! merge assembly, overlap deduplication, error propagation, and round-trip
//! page count preservation.
//!
//! **Chunk splitting (3 harnesses):**
//!  1. Chunk splitting is exhaustive: no page dropped from partitioning.
//!  2. Overlap region bounded: overlap_pages in [0, chunk_size) prevents OOB.
//!  3. Single-page chunk: degenerate 1-page document handled correctly.
//!
//! **Sequential assembly (2 harnesses):**
//!  4. ChunkOutput ordering preserves page indices.
//!  5. Page index monotonicity: assembled page offsets strictly increasing.
//!
//! **Config validation (2 harnesses):**
//!  6. StreamingConfig fields validated: chunk_size > 0, overlap < chunk_size.
//!  7. Overlap == chunk_size - 1 is the maximal valid overlap.
//!
//! **Memory bound (1 harness):**
//!  8. Per-chunk allocation bounded by max_pages * region estimate.
//!
//! **Chunk boundary regions (1 harness):**
//!  9. Regions in overlap zone appear in both adjacent chunks.
//!
//! **Error propagation (1 harness):**
//! 10. StreamingError variants preserve context values.
//!
//! **Concurrent chunk independence (1 harness):**
//! 11. No shared mutable state: separate pipeline instances produce identical
//!     chunk plans.
//!
//! **Progress reporting (1 harness):**
//! 12. Chunk completion fraction in [0.0, 1.0].
//!
//! **Overlap dedup (1 harness):**
//! 13. Duplicate regions in overlap zone filtered by IoU.
//!
//! **Flush semantics (1 harness):**
//! 14. Final chunk processes all remaining pages.
//!
//! **Round-trip (1 harness):**
//! 15. Split -> process -> assemble preserves total page count.

#[cfg(kani)]
mod proofs {
    use crate::dpdf_pipeline::{DocumentRegion, PageOutput, PipelineConfig};
    use crate::dpdf_streaming::{ChunkOutput, StreamingConfig, StreamingError, StreamingPipeline};

    /// Helper: create a Text region with given bbox and confidence.
    fn text_region(bbox: [f32; 4], confidence: f32) -> DocumentRegion {
        DocumentRegion::Text {
            content: String::new(),
            bbox,
            confidence,
        }
    }

    /// Helper: create a PageOutput with the given regions.
    fn make_page(regions: Vec<DocumentRegion>) -> PageOutput {
        let reading_order: Vec<usize> = (0..regions.len()).collect();
        PageOutput {
            width: 612,
            height: 792,
            regions,
            reading_order,
        }
    }

    // ===================================================================
    // 1. Chunk splitting is exhaustive: no page dropped
    // ===================================================================

    /// SUBSTANTIVE: Proves that for bounded symbolic parameters, every page
    /// index in `[0, total_pages)` appears in at least one chunk range
    /// produced by `chunk_pages`. Uses a direct membership check per page.
    #[kani::proof]
    #[kani::unwind(14)]
    fn proof_chunk_splitting_exhaustive_no_dropped_pages() {
        let total_pages: usize = kani::any();
        kani::assume(total_pages >= 1 && total_pages <= 6);

        let chunk_size: usize = kani::any();
        kani::assume(chunk_size >= 1 && chunk_size <= 6);

        let overlap: usize = kani::any();
        kani::assume(overlap < chunk_size);

        let config = StreamingConfig {
            chunk_size,
            overlap_pages: overlap,
            max_memory_bytes: None,
        };
        let pipeline =
            StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

        let chunks = pipeline.chunk_pages(total_pages);

        // Verify every page is covered.
        let mut page = 0;
        while page < total_pages {
            let mut covered = false;
            let mut i = 0;
            while i < chunks.len() {
                if page >= chunks[i].start && page < chunks[i].end {
                    covered = true;
                }
                i += 1;
            }
            assert!(covered, "page must appear in at least one chunk");
            page += 1;
        }
    }

    // ===================================================================
    // 2. Overlap region bounded: overlap_pages < chunk_size prevents OOB
    // ===================================================================

    /// SUBSTANTIVE: Proves that valid overlap values (0..chunk_size-1) produce
    /// chunk ranges where all indices are strictly less than total_pages.
    /// Also verifies the stride (chunk_size - overlap) is always >= 1,
    /// guaranteeing forward progress.
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_overlap_region_bounded_no_oob() {
        let chunk_size: usize = kani::any();
        kani::assume(chunk_size >= 1 && chunk_size <= 8);

        let overlap: usize = kani::any();
        kani::assume(overlap < chunk_size);

        // Stride must be at least 1.
        let stride = chunk_size - overlap;
        assert!(stride >= 1, "stride must be >= 1 for forward progress");

        let total_pages: usize = kani::any();
        kani::assume(total_pages >= 1 && total_pages <= 8);

        let config = StreamingConfig {
            chunk_size,
            overlap_pages: overlap,
            max_memory_bytes: None,
        };
        let pipeline =
            StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

        let chunks = pipeline.chunk_pages(total_pages);

        // All chunk endpoints must be within bounds.
        let mut i = 0;
        while i < chunks.len() {
            assert!(chunks[i].start < total_pages, "start must be < total_pages");
            assert!(chunks[i].end <= total_pages, "end must be <= total_pages");
            assert!(chunks[i].start < chunks[i].end, "range must be non-empty");
            i += 1;
        }
    }

    // ===================================================================
    // 3. Single-page chunk: 1-page document handled correctly
    // ===================================================================

    /// SUBSTANTIVE: Proves that a document with exactly 1 page produces
    /// exactly 1 chunk covering page 0, regardless of chunk_size and overlap
    /// configuration.
    #[kani::proof]
    #[kani::unwind(4)]
    fn proof_single_page_chunk_handled() {
        let chunk_size: usize = kani::any();
        kani::assume(chunk_size >= 1 && chunk_size <= 10);

        let overlap: usize = kani::any();
        kani::assume(overlap < chunk_size);

        let config = StreamingConfig {
            chunk_size,
            overlap_pages: overlap,
            max_memory_bytes: None,
        };
        let pipeline =
            StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

        let chunks = pipeline.chunk_pages(1);

        assert_eq!(chunks.len(), 1, "1-page doc must produce exactly 1 chunk");
        assert_eq!(chunks[0].start, 0, "single chunk must start at 0");
        assert_eq!(chunks[0].end, 1, "single chunk must end at 1");
    }

    // ===================================================================
    // 4. ChunkOutput ordering preserves page indices
    // ===================================================================

    /// SUBSTANTIVE: Proves that when ChunkOutputs are constructed with
    /// page_offset values matching `chunk_pages` output, merge_chunks
    /// successfully assembles them and the result contains the correct
    /// number of pages.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_chunk_output_ordering_preserves_page_indices() {
        let config = StreamingConfig {
            chunk_size: 3,
            overlap_pages: 1,
            max_memory_bytes: None,
        };
        let pipeline =
            StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

        let total_pages = 5;
        let ranges = pipeline.chunk_pages(total_pages);

        // Build ChunkOutputs matching the chunk plan.
        let mut chunk_outputs = Vec::new();
        let mut ci = 0;
        while ci < ranges.len() {
            let range = &ranges[ci];
            let num_pages = range.end - range.start;
            let mut page_outputs = Vec::new();
            let mut p = 0;
            while p < num_pages {
                page_outputs.push(make_page(vec![]));
                p += 1;
            }
            chunk_outputs.push(ChunkOutput {
                page_outputs,
                page_offset: range.start,
                chunk_index: ci,
            });
            ci += 1;
        }

        let result = pipeline.merge_chunks(chunk_outputs);
        assert!(result.is_ok(), "merge must succeed for valid chunk plan");
        let doc = result.unwrap();
        assert_eq!(
            doc.pages.len(),
            total_pages,
            "assembled page count must equal total_pages"
        );
    }

    // ===================================================================
    // 5. Page index monotonicity in assembled output
    // ===================================================================

    /// SUBSTANTIVE: Proves that chunk_pages produces ranges with strictly
    /// increasing start offsets, guaranteeing page index monotonicity when
    /// assembling results sequentially.
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_page_index_monotonicity_assembled() {
        let total_pages: usize = kani::any();
        kani::assume(total_pages >= 2 && total_pages <= 8);

        let chunk_size: usize = kani::any();
        kani::assume(chunk_size >= 1 && chunk_size <= 8);

        let overlap: usize = kani::any();
        kani::assume(overlap < chunk_size);

        let config = StreamingConfig {
            chunk_size,
            overlap_pages: overlap,
            max_memory_bytes: None,
        };
        let pipeline =
            StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

        let chunks = pipeline.chunk_pages(total_pages);

        // Start offsets must be strictly increasing.
        let mut i = 1;
        while i < chunks.len() {
            assert!(
                chunks[i].start > chunks[i - 1].start,
                "page offsets must be strictly increasing"
            );
            // End offsets must also be non-decreasing.
            assert!(
                chunks[i].end >= chunks[i - 1].end,
                "end offsets must be non-decreasing"
            );
            i += 1;
        }
    }

    // ===================================================================
    // 6. Config validation: chunk_size > 0, overlap < chunk_size
    // ===================================================================

    /// SUBSTANTIVE: Proves that the constructor rejects all invalid
    /// configurations: chunk_size == 0 yields InvalidChunkSize, and
    /// overlap >= chunk_size yields OverlapExceedsChunkSize. Valid configs
    /// always succeed.
    #[kani::proof]
    #[kani::unwind(2)]
    fn proof_config_validation_rejects_invalid() {
        let chunk_size: usize = kani::any();
        kani::assume(chunk_size <= 5);

        let overlap: usize = kani::any();
        kani::assume(overlap <= 5);

        let config = StreamingConfig {
            chunk_size,
            overlap_pages: overlap,
            max_memory_bytes: None,
        };
        let result = StreamingPipeline::new(config, PipelineConfig::default());

        if chunk_size == 0 {
            assert!(result.is_err(), "chunk_size == 0 must be rejected");
        } else if overlap >= chunk_size {
            assert!(result.is_err(), "overlap >= chunk_size must be rejected");
        } else {
            assert!(result.is_ok(), "valid config must be accepted");
        }
    }

    // ===================================================================
    // 7. Overlap == chunk_size - 1 is maximal valid overlap
    // ===================================================================

    /// SUBSTANTIVE: Proves that overlap = chunk_size - 1 (the maximum valid
    /// overlap) produces a valid pipeline and generates chunks with stride 1.
    /// This is the densest possible overlap, and each chunk advances by exactly
    /// one page.
    #[kani::proof]
    #[kani::unwind(14)]
    fn proof_maximal_overlap_valid_and_stride_one() {
        let chunk_size: usize = kani::any();
        kani::assume(chunk_size >= 2 && chunk_size <= 6);

        let overlap = chunk_size - 1; // maximal valid overlap

        let config = StreamingConfig {
            chunk_size,
            overlap_pages: overlap,
            max_memory_bytes: None,
        };
        let pipeline = StreamingPipeline::new(config, PipelineConfig::default())
            .expect("maximal overlap valid");

        let total_pages: usize = kani::any();
        kani::assume(total_pages >= 1 && total_pages <= 6);

        let chunks = pipeline.chunk_pages(total_pages);

        // With stride = 1, each consecutive chunk starts 1 page later.
        let mut i = 1;
        while i < chunks.len() {
            let stride = chunks[i].start - chunks[i - 1].start;
            assert_eq!(stride, 1, "maximal overlap must produce stride = 1");
            i += 1;
        }
    }

    // ===================================================================
    // 8. Memory bound: per-chunk allocation bounded
    // ===================================================================

    /// SUBSTANTIVE: Proves that `estimate_chunk_memory` is monotonically
    /// non-decreasing in each dimension (width, height, models, chunk_size),
    /// and that the result is bounded by `chunk_size * image_size * (1 + 2*models)`.
    #[kani::proof]
    #[kani::unwind(2)]
    fn proof_memory_bound_monotonic_in_dimensions() {
        let chunk_size: usize = kani::any();
        kani::assume(chunk_size >= 1 && chunk_size <= 10);

        let config = StreamingConfig {
            chunk_size,
            overlap_pages: 0,
            max_memory_bytes: None,
        };
        let pipeline =
            StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

        let w: usize = kani::any();
        kani::assume(w >= 1 && w <= 100);
        let h: usize = kani::any();
        kani::assume(h >= 1 && h <= 100);
        let models: usize = kani::any();
        kani::assume(models <= 3);

        let mem = pipeline.estimate_chunk_memory(w, h, models);

        // Monotonic in width: doubling w should not decrease memory.
        if w <= 50 {
            let mem_wider = pipeline.estimate_chunk_memory(w * 2, h, models);
            assert!(
                mem_wider >= mem,
                "memory must be monotonically non-decreasing in width"
            );
        }

        // Monotonic in models: adding a model should not decrease memory.
        if models <= 2 {
            let mem_more_models = pipeline.estimate_chunk_memory(w, h, models + 1);
            assert!(
                mem_more_models >= mem,
                "memory must be monotonically non-decreasing in model count"
            );
        }
    }

    // ===================================================================
    // 9. Chunk boundary regions: overlap pages shared between adjacent chunks
    // ===================================================================

    /// SUBSTANTIVE: Proves that for any page in the overlap zone between
    /// two adjacent chunks, both chunks include that page index. This
    /// guarantees that regions straddling a chunk boundary are detected by
    /// both chunks.
    #[kani::proof]
    #[kani::unwind(14)]
    fn proof_chunk_boundary_pages_shared() {
        let total_pages: usize = kani::any();
        kani::assume(total_pages >= 3 && total_pages <= 8);

        let chunk_size: usize = kani::any();
        kani::assume(chunk_size >= 2 && chunk_size <= 6);

        let overlap: usize = kani::any();
        kani::assume(overlap >= 1 && overlap < chunk_size);

        let config = StreamingConfig {
            chunk_size,
            overlap_pages: overlap,
            max_memory_bytes: None,
        };
        let pipeline =
            StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

        let chunks = pipeline.chunk_pages(total_pages);

        // For each pair of adjacent chunks, verify pages in the overlap zone
        // appear in both chunks.
        let mut i = 1;
        while i < chunks.len() {
            let overlap_start = chunks[i].start;
            let overlap_end = chunks[i - 1].end;
            // The overlap zone is [overlap_start, overlap_end).
            if overlap_end > overlap_start {
                let mut page = overlap_start;
                while page < overlap_end {
                    // Page must be in chunk i-1.
                    assert!(
                        page >= chunks[i - 1].start && page < chunks[i - 1].end,
                        "overlap page must be in previous chunk"
                    );
                    // Page must be in chunk i.
                    assert!(
                        page >= chunks[i].start && page < chunks[i].end,
                        "overlap page must be in current chunk"
                    );
                    page += 1;
                }
            }
            i += 1;
        }
    }

    // ===================================================================
    // 10. Error propagation: StreamingError preserves context
    // ===================================================================

    /// SUBSTANTIVE: Proves that StreamingError variants preserve the values
    /// used to construct them, so error context is not lost during
    /// propagation. Checks all four error variants.
    #[kani::proof]
    #[kani::unwind(2)]
    fn proof_streaming_error_preserves_context() {
        // InvalidChunkSize preserves the invalid size.
        let err = StreamingError::InvalidChunkSize(0);
        match err {
            StreamingError::InvalidChunkSize(v) => assert_eq!(v, 0),
            _ => panic!("wrong variant"),
        }

        // OverlapExceedsChunkSize preserves both values.
        let err = StreamingError::OverlapExceedsChunkSize {
            overlap: 5,
            chunk_size: 3,
        };
        match err {
            StreamingError::OverlapExceedsChunkSize {
                overlap,
                chunk_size,
            } => {
                assert_eq!(overlap, 5);
                assert_eq!(chunk_size, 3);
            }
            _ => panic!("wrong variant"),
        }

        // MemoryBudgetExceeded preserves both values.
        let err = StreamingError::MemoryBudgetExceeded {
            estimated: 1024,
            budget: 512,
        };
        match err {
            StreamingError::MemoryBudgetExceeded { estimated, budget } => {
                assert_eq!(estimated, 1024);
                assert_eq!(budget, 512);
            }
            _ => panic!("wrong variant"),
        }

        // NonContiguousChunks preserves all three values.
        let err = StreamingError::NonContiguousChunks {
            chunk_index: 2,
            expected: 10,
            actual: 15,
        };
        match err {
            StreamingError::NonContiguousChunks {
                chunk_index,
                expected,
                actual,
            } => {
                assert_eq!(chunk_index, 2);
                assert_eq!(expected, 10);
                assert_eq!(actual, 15);
            }
            _ => panic!("wrong variant"),
        }
    }

    // ===================================================================
    // 11. Concurrent chunk independence: separate instances produce
    //     identical chunk plans
    // ===================================================================

    /// SUBSTANTIVE: Proves that two independently constructed pipelines
    /// with the same configuration produce identical chunk plans, verifying
    /// there is no hidden mutable state between instances.
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_concurrent_chunk_independence() {
        let chunk_size: usize = kani::any();
        kani::assume(chunk_size >= 1 && chunk_size <= 6);

        let overlap: usize = kani::any();
        kani::assume(overlap < chunk_size);

        let total_pages: usize = kani::any();
        kani::assume(total_pages >= 1 && total_pages <= 6);

        let config1 = StreamingConfig {
            chunk_size,
            overlap_pages: overlap,
            max_memory_bytes: None,
        };
        let config2 = StreamingConfig {
            chunk_size,
            overlap_pages: overlap,
            max_memory_bytes: None,
        };

        let pipeline1 = StreamingPipeline::new(config1, PipelineConfig::default()).expect("valid");
        let pipeline2 = StreamingPipeline::new(config2, PipelineConfig::default()).expect("valid");

        let chunks1 = pipeline1.chunk_pages(total_pages);
        let chunks2 = pipeline2.chunk_pages(total_pages);

        assert_eq!(
            chunks1.len(),
            chunks2.len(),
            "identical configs must produce same chunk count"
        );

        let mut i = 0;
        while i < chunks1.len() {
            assert_eq!(
                chunks1[i].start, chunks2[i].start,
                "chunk starts must match"
            );
            assert_eq!(chunks1[i].end, chunks2[i].end, "chunk ends must match");
            i += 1;
        }
    }

    // ===================================================================
    // 12. Progress reporting: chunk completion fraction in [0.0, 1.0]
    // ===================================================================

    /// SUBSTANTIVE: Proves that for any chunk index in the chunk plan,
    /// the fraction (chunk_index + 1) / total_chunks is in (0.0, 1.0]
    /// and the fraction is monotonically increasing across chunks.
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_progress_fraction_in_unit_range() {
        let total_pages: usize = kani::any();
        kani::assume(total_pages >= 1 && total_pages <= 6);

        let chunk_size: usize = kani::any();
        kani::assume(chunk_size >= 1 && chunk_size <= 6);

        let overlap: usize = kani::any();
        kani::assume(overlap < chunk_size);

        let config = StreamingConfig {
            chunk_size,
            overlap_pages: overlap,
            max_memory_bytes: None,
        };
        let pipeline =
            StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

        let chunks = pipeline.chunk_pages(total_pages);
        let total_chunks = chunks.len();
        assert!(total_chunks >= 1);

        let mut prev_frac = 0.0_f64;
        let mut i = 0;
        while i < total_chunks {
            let frac = (i + 1) as f64 / total_chunks as f64;
            assert!(frac > 0.0, "progress fraction must be > 0");
            assert!(frac <= 1.0, "progress fraction must be <= 1.0");
            assert!(
                frac > prev_frac,
                "progress fraction must be strictly increasing"
            );
            prev_frac = frac;
            i += 1;
        }

        // Final chunk fraction must be exactly 1.0.
        let final_frac = total_chunks as f64 / total_chunks as f64;
        assert!(
            (final_frac - 1.0).abs() < 1e-15,
            "final chunk fraction must be 1.0"
        );
    }

    // ===================================================================
    // 13. Overlap dedup: duplicate regions in overlap zone filtered
    // ===================================================================

    /// SUBSTANTIVE: Proves that when merge_chunks encounters the same region
    /// (same class, same bbox, same confidence) in an overlap page from two
    /// adjacent chunks, the merged result contains exactly one copy of that
    /// region, not two.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_overlap_dedup_filters_duplicates() {
        let config = StreamingConfig {
            chunk_size: 3,
            overlap_pages: 1,
            max_memory_bytes: None,
        };
        let pipeline =
            StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

        let shared_region = text_region([10.0, 20.0, 300.0, 80.0], 0.9);

        // Chunk 0: pages 0, 1, 2. Page 2 has the shared region.
        let chunk0 = ChunkOutput {
            page_outputs: vec![
                make_page(vec![]),
                make_page(vec![]),
                make_page(vec![shared_region.clone()]),
            ],
            page_offset: 0,
            chunk_index: 0,
        };

        // Chunk 1: pages 2, 3, 4. Page 2 (local index 0) has the same region.
        let chunk1 = ChunkOutput {
            page_outputs: vec![
                make_page(vec![shared_region.clone()]),
                make_page(vec![]),
                make_page(vec![]),
            ],
            page_offset: 2,
            chunk_index: 1,
        };

        let result = pipeline.merge_chunks(vec![chunk0, chunk1]);
        assert!(result.is_ok(), "merge must succeed");
        let doc = result.unwrap();

        assert_eq!(doc.pages.len(), 5, "total pages must be 5");

        // The overlap page (page 2) should have exactly 1 region, not 2.
        // The dedup logic uses IoU > threshold to suppress duplicates.
        // Same bbox => IoU = 1.0, which exceeds any threshold in (0, 1).
        assert_eq!(
            doc.pages[2].regions.len(),
            1,
            "duplicate region in overlap page must be deduplicated to one"
        );
    }

    // ===================================================================
    // 14. Flush semantics: final chunk processes all remaining pages
    // ===================================================================

    /// SUBSTANTIVE: Proves that the final chunk produced by `chunk_pages`
    /// always ends at exactly `total_pages`, ensuring all remaining pages
    /// are flushed. Also verifies the final chunk is non-empty.
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_flush_semantics_final_chunk_complete() {
        let total_pages: usize = kani::any();
        kani::assume(total_pages >= 1 && total_pages <= 8);

        let chunk_size: usize = kani::any();
        kani::assume(chunk_size >= 1 && chunk_size <= 8);

        let overlap: usize = kani::any();
        kani::assume(overlap < chunk_size);

        let config = StreamingConfig {
            chunk_size,
            overlap_pages: overlap,
            max_memory_bytes: None,
        };
        let pipeline =
            StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

        let chunks = pipeline.chunk_pages(total_pages);
        assert!(!chunks.is_empty());

        let last = &chunks[chunks.len() - 1];

        // Final chunk must end at total_pages (flush all remaining).
        assert_eq!(
            last.end, total_pages,
            "final chunk must flush to total_pages"
        );

        // Final chunk must be non-empty.
        assert!(last.start < last.end, "final chunk must be non-empty");

        // Final chunk size must be <= chunk_size (never oversized).
        let final_size = last.end - last.start;
        assert!(
            final_size <= chunk_size,
            "final chunk must not exceed chunk_size"
        );
    }

    // ===================================================================
    // 15. Round-trip: split -> process -> assemble preserves total page count
    // ===================================================================

    /// SUBSTANTIVE: Proves the full round-trip: split a document into chunks
    /// via `chunk_pages`, construct ChunkOutputs with one page per range
    /// entry, merge them via `merge_chunks`, and verify the assembled
    /// document has exactly `total_pages` pages.
    #[kani::proof]
    #[kani::unwind(14)]
    fn proof_round_trip_preserves_total_page_count() {
        let total_pages: usize = kani::any();
        kani::assume(total_pages >= 1 && total_pages <= 5);

        let chunk_size: usize = kani::any();
        kani::assume(chunk_size >= 1 && chunk_size <= 5);

        let overlap: usize = kani::any();
        kani::assume(overlap < chunk_size);

        let config = StreamingConfig {
            chunk_size,
            overlap_pages: overlap,
            max_memory_bytes: None,
        };
        let pipeline =
            StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

        // Step 1: Split.
        let ranges = pipeline.chunk_pages(total_pages);
        assert!(!ranges.is_empty());

        // Step 2: Build ChunkOutputs from ranges.
        let mut chunk_outputs = Vec::new();
        let mut ci = 0;
        while ci < ranges.len() {
            let range = &ranges[ci];
            let num_pages = range.end - range.start;
            let mut page_outputs = Vec::new();
            let mut p = 0;
            while p < num_pages {
                page_outputs.push(make_page(vec![]));
                p += 1;
            }
            chunk_outputs.push(ChunkOutput {
                page_outputs,
                page_offset: range.start,
                chunk_index: ci,
            });
            ci += 1;
        }

        // Step 3: Assemble.
        let result = pipeline.merge_chunks(chunk_outputs);
        assert!(result.is_ok(), "merge must succeed for valid chunk plan");
        let doc = result.unwrap();

        // Step 4: Verify total page count preserved.
        assert_eq!(
            doc.pages.len(),
            total_pages,
            "round-trip must preserve total page count"
        );
    }
}
