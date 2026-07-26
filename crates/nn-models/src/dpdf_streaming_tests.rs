// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dpdf_pipeline::{DocumentRegion, PageOutput, PipelineConfig};

// ---------------------------------------------------------------------------
// StreamingConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn test_streaming_config_default_values() {
    let config = StreamingConfig::default();
    assert_eq!(config.chunk_size, 10);
    assert_eq!(config.overlap_pages, 1);
    assert!(config.max_memory_bytes.is_none());
}

// ---------------------------------------------------------------------------
// StreamingPipeline construction validation
// ---------------------------------------------------------------------------

#[test]
fn test_new_valid_config() {
    let result = StreamingPipeline::new(StreamingConfig::default(), PipelineConfig::default());
    assert!(result.is_ok());
}

#[test]
fn test_new_zero_chunk_size_returns_error() {
    let config = StreamingConfig {
        chunk_size: 0,
        ..StreamingConfig::default()
    };
    let result = StreamingPipeline::new(config, PipelineConfig::default());
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        StreamingError::InvalidChunkSize(0)
    ));
}

#[test]
fn test_new_overlap_equals_chunk_size_returns_error() {
    let config = StreamingConfig {
        chunk_size: 5,
        overlap_pages: 5,
        max_memory_bytes: None,
    };
    let result = StreamingPipeline::new(config, PipelineConfig::default());
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        StreamingError::OverlapExceedsChunkSize { .. }
    ));
}

#[test]
fn test_new_overlap_exceeds_chunk_size_returns_error() {
    let config = StreamingConfig {
        chunk_size: 3,
        overlap_pages: 4,
        max_memory_bytes: None,
    };
    let result = StreamingPipeline::new(config, PipelineConfig::default());
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// chunk_pages
// ---------------------------------------------------------------------------

#[test]
fn test_chunk_pages_zero_total() {
    let pipeline =
        StreamingPipeline::new(StreamingConfig::default(), PipelineConfig::default()).unwrap();
    let chunks = pipeline.chunk_pages(0);
    assert!(chunks.is_empty());
}

#[test]
fn test_chunk_pages_single_chunk() {
    let config = StreamingConfig {
        chunk_size: 10,
        overlap_pages: 1,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).unwrap();
    let chunks = pipeline.chunk_pages(5);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], 0..5);
}

#[test]
fn test_chunk_pages_exact_fit() {
    let config = StreamingConfig {
        chunk_size: 5,
        overlap_pages: 0,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).unwrap();
    let chunks = pipeline.chunk_pages(10);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], 0..5);
    assert_eq!(chunks[1], 5..10);
}

#[test]
fn test_chunk_pages_with_overlap() {
    let config = StreamingConfig {
        chunk_size: 10,
        overlap_pages: 2,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).unwrap();
    let chunks = pipeline.chunk_pages(25);
    // stride = 10 - 2 = 8
    // chunk 0: 0..10, chunk 1: 8..18, chunk 2: 16..25
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0], 0..10);
    assert_eq!(chunks[1], 8..18);
    assert_eq!(chunks[2], 16..25);
}

#[test]
fn test_chunk_pages_35_pages_default_config() {
    let config = StreamingConfig {
        chunk_size: 10,
        overlap_pages: 1,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).unwrap();
    let chunks = pipeline.chunk_pages(35);
    // stride = 10 - 1 = 9
    // chunk 0: 0..10, chunk 1: 9..19, chunk 2: 18..28, chunk 3: 27..35
    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks[0], 0..10);
    assert_eq!(chunks[1], 9..19);
    assert_eq!(chunks[2], 18..28);
    assert_eq!(chunks[3], 27..35);
}

#[test]
fn test_chunk_pages_single_page() {
    let config = StreamingConfig {
        chunk_size: 10,
        overlap_pages: 1,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).unwrap();
    let chunks = pipeline.chunk_pages(1);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], 0..1);
}

#[test]
fn test_chunk_pages_no_overlap() {
    let config = StreamingConfig {
        chunk_size: 5,
        overlap_pages: 0,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).unwrap();
    let chunks = pipeline.chunk_pages(12);
    // stride = 5, chunk 0: 0..5, chunk 1: 5..10, chunk 2: 10..12
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0], 0..5);
    assert_eq!(chunks[1], 5..10);
    assert_eq!(chunks[2], 10..12);
}

// ---------------------------------------------------------------------------
// merge_chunks
// ---------------------------------------------------------------------------

fn make_page(regions: Vec<DocumentRegion>, width: usize, height: usize) -> PageOutput {
    let reading_order = crate::dpdf_pipeline::DpdfPipeline::compute_reading_order(&regions);
    PageOutput {
        regions,
        reading_order,
        width,
        height,
    }
}

fn text_region(content: &str, bbox: [f32; 4], confidence: f32) -> DocumentRegion {
    DocumentRegion::Text {
        content: content.to_string(),
        bbox,
        confidence,
    }
}

#[test]
fn test_merge_chunks_empty() {
    let pipeline =
        StreamingPipeline::new(StreamingConfig::default(), PipelineConfig::default()).unwrap();
    let result = pipeline.merge_chunks(Vec::new()).unwrap();
    assert!(result.pages.is_empty());
}

#[test]
fn test_merge_chunks_single_chunk() {
    let pipeline =
        StreamingPipeline::new(StreamingConfig::default(), PipelineConfig::default()).unwrap();

    let page0 = make_page(
        vec![text_region("hello", [0.0, 0.0, 100.0, 50.0], 0.9)],
        612,
        792,
    );
    let page1 = make_page(
        vec![text_region("world", [0.0, 0.0, 100.0, 50.0], 0.8)],
        612,
        792,
    );
    let chunk = ChunkOutput {
        page_outputs: vec![page0, page1],
        page_offset: 0,
        chunk_index: 0,
    };

    let result = pipeline.merge_chunks(vec![chunk]).unwrap();
    assert_eq!(result.pages.len(), 2);
    assert_eq!(result.pages[0].regions.len(), 1);
    assert_eq!(result.pages[1].regions.len(), 1);
}

#[test]
fn test_merge_chunks_deduplicates_overlap() {
    let config = StreamingConfig {
        chunk_size: 3,
        overlap_pages: 1,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).unwrap();

    // Chunk 0: pages 0, 1, 2
    let chunk0 = ChunkOutput {
        page_outputs: vec![
            make_page(
                vec![text_region("p0", [0.0, 0.0, 100.0, 50.0], 0.9)],
                612,
                792,
            ),
            make_page(
                vec![text_region("p1", [0.0, 0.0, 100.0, 50.0], 0.8)],
                612,
                792,
            ),
            // Page 2 is the overlap page
            make_page(
                vec![text_region(
                    "p2-from-chunk0",
                    [10.0, 10.0, 200.0, 60.0],
                    0.7,
                )],
                612,
                792,
            ),
        ],
        page_offset: 0,
        chunk_index: 0,
    };

    // Chunk 1: pages 2, 3, 4. Page 2 overlaps.
    let chunk1 = ChunkOutput {
        page_outputs: vec![
            // Same region on page 2 with slightly different confidence.
            make_page(
                vec![text_region(
                    "p2-from-chunk1",
                    [10.0, 10.0, 200.0, 60.0],
                    0.85,
                )],
                612,
                792,
            ),
            make_page(
                vec![text_region("p3", [0.0, 0.0, 100.0, 50.0], 0.9)],
                612,
                792,
            ),
            make_page(
                vec![text_region("p4", [0.0, 0.0, 100.0, 50.0], 0.9)],
                612,
                792,
            ),
        ],
        page_offset: 2,
        chunk_index: 1,
    };

    let result = pipeline.merge_chunks(vec![chunk0, chunk1]).unwrap();
    assert_eq!(result.pages.len(), 5);

    // The overlap page (index 2) should have only 1 region (deduped).
    // The higher-confidence version (0.85 from chunk1) should win.
    assert_eq!(result.pages[2].regions.len(), 1);
    assert!(result.pages[2].regions[0].confidence() > 0.8);
}

#[test]
fn test_merge_chunks_non_overlapping_regions_preserved() {
    let config = StreamingConfig {
        chunk_size: 2,
        overlap_pages: 1,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).unwrap();

    // Chunk 0: pages 0, 1. Page 1 has a region on the left side.
    let chunk0 = ChunkOutput {
        page_outputs: vec![
            make_page(
                vec![text_region("p0", [0.0, 0.0, 100.0, 50.0], 0.9)],
                612,
                792,
            ),
            make_page(
                vec![text_region("left-region", [0.0, 0.0, 100.0, 50.0], 0.8)],
                612,
                792,
            ),
        ],
        page_offset: 0,
        chunk_index: 0,
    };

    // Chunk 1: pages 1, 2. Page 1 has a different region on the right side.
    let chunk1 = ChunkOutput {
        page_outputs: vec![
            make_page(
                vec![text_region("right-region", [400.0, 0.0, 600.0, 50.0], 0.85)],
                612,
                792,
            ),
            make_page(
                vec![text_region("p2", [0.0, 0.0, 100.0, 50.0], 0.9)],
                612,
                792,
            ),
        ],
        page_offset: 1,
        chunk_index: 1,
    };

    let result = pipeline.merge_chunks(vec![chunk0, chunk1]).unwrap();
    assert_eq!(result.pages.len(), 3);

    // Overlap page (index 1) should have both regions since they don't overlap.
    assert_eq!(result.pages[1].regions.len(), 2);
}

// ---------------------------------------------------------------------------
// estimate_chunk_memory
// ---------------------------------------------------------------------------

#[test]
fn test_estimate_chunk_memory_basic() {
    let config = StreamingConfig {
        chunk_size: 10,
        overlap_pages: 1,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).unwrap();

    let mem = pipeline.estimate_chunk_memory(1024, 1024, 2);
    // Per page: 1024 * 1024 * 12 = 12_582_912 bytes for raw image
    // With 2 models: per_page = 12_582_912 * (1 + 2*2) = 12_582_912 * 5 = 62_914_560
    // 10 pages: 629_145_600
    let expected_image_bytes: usize = 1024 * 1024 * 3 * 4;
    let expected_per_page = expected_image_bytes * (1 + 2 * 2);
    let expected_total = expected_per_page * 10;
    assert_eq!(mem, expected_total);
}

#[test]
fn test_estimate_chunk_memory_zero_dimensions() {
    let config = StreamingConfig {
        chunk_size: 5,
        overlap_pages: 0,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).unwrap();

    let mem = pipeline.estimate_chunk_memory(0, 0, 2);
    assert_eq!(mem, 0);
}

#[test]
fn test_estimate_chunk_memory_zero_models() {
    let config = StreamingConfig {
        chunk_size: 1,
        overlap_pages: 0,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).unwrap();

    let mem = pipeline.estimate_chunk_memory(100, 100, 0);
    // image_bytes = 100 * 100 * 12 = 120_000
    // per_page = 120_000 * (1 + 0) = 120_000
    assert_eq!(mem, 120_000);
}

// ---------------------------------------------------------------------------
// Accessor coverage
// ---------------------------------------------------------------------------

#[test]
fn test_accessors() {
    let streaming_config = StreamingConfig {
        chunk_size: 20,
        overlap_pages: 3,
        max_memory_bytes: Some(1_000_000),
    };
    let pipeline_config = PipelineConfig::default();
    let pipeline = StreamingPipeline::new(streaming_config, pipeline_config).unwrap();

    assert_eq!(pipeline.config().chunk_size, 20);
    assert_eq!(pipeline.config().overlap_pages, 3);
    assert_eq!(pipeline.config().max_memory_bytes, Some(1_000_000));
    // Pipeline config accessible.
    assert!(pipeline.pipeline_config().layout_conf_threshold > 0.0);
}
