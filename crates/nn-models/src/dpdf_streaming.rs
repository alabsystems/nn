// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Streaming (chunked) document processing for the dpdf pipeline.
//!
//! Large documents may exceed memory limits when processed page-by-page in a
//! single batch. [`StreamingPipeline`] splits a document into overlapping
//! chunks of pages, processes each chunk independently, and merges the
//! results into a single [`DocumentOutput`] with cross-boundary deduplication.
//!
//! # Overlap strategy
//!
//! Adjacent chunks share `overlap_pages` pages so that regions straddling a
//! chunk boundary are detected by both chunks. During merge, duplicate
//! regions in the overlap zone are deduplicated by IoU similarity.
//!
//! # Usage
//!
//! ```rust,no_run
//! use nn_models::dpdf_streaming::{StreamingConfig, StreamingPipeline};
//! use nn_models::dpdf_pipeline::PipelineConfig;
//!
//! let config = StreamingConfig::default();
//! let pipeline = StreamingPipeline::new(config, PipelineConfig::default());
//! let chunks = pipeline.chunk_pages(35);
//! assert_eq!(chunks.len(), 4); // 35 pages, chunk_size=10, overlap=1
//! ```

use std::ops::Range;

use crate::dpdf_pipeline::{DocumentOutput, DocumentRegion, PageOutput, PipelineConfig};
use crate::dpdf_postprocess::compute_iou;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for chunked document streaming.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Number of pages per chunk (default 10).
    pub chunk_size: usize,
    /// Pages shared between adjacent chunks for cross-boundary region merging
    /// (default 1).
    pub overlap_pages: usize,
    /// Optional memory budget in bytes. [`StreamingPipeline::estimate_chunk_memory`]
    /// can be used to check whether a chunk fits.
    pub max_memory_bytes: Option<usize>,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10,
            overlap_pages: 1,
            max_memory_bytes: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Chunk output
// ---------------------------------------------------------------------------

/// Output from processing a single chunk of pages.
#[derive(Debug, Clone)]
pub struct ChunkOutput {
    /// Per-page outputs within this chunk.
    pub page_outputs: Vec<PageOutput>,
    /// Starting page index of this chunk in the full document (0-based).
    pub page_offset: usize,
    /// Zero-based index of this chunk.
    pub chunk_index: usize,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from the streaming pipeline.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StreamingError {
    /// Chunk size must be at least 1.
    #[error("chunk_size must be >= 1, got {0}")]
    InvalidChunkSize(usize),

    /// Overlap must be strictly less than chunk size.
    #[error("overlap_pages ({overlap}) must be < chunk_size ({chunk_size})")]
    OverlapExceedsChunkSize { overlap: usize, chunk_size: usize },

    /// A chunk exceeds the configured memory budget.
    #[error("estimated chunk memory {estimated} bytes exceeds budget {budget} bytes")]
    MemoryBudgetExceeded { estimated: usize, budget: usize },

    /// Chunk indices are not contiguous or have incorrect offsets.
    #[error("non-contiguous chunk at index {chunk_index}: expected page_offset {expected}, got {actual}")]
    NonContiguousChunks {
        chunk_index: usize,
        expected: usize,
        actual: usize,
    },
}

// ---------------------------------------------------------------------------
// Streaming pipeline
// ---------------------------------------------------------------------------

/// Orchestrates chunked document processing with overlap-based merging.
///
/// Wraps a [`PipelineConfig`] and provides methods to split a document into
/// overlapping page chunks, merge chunk results, and estimate per-chunk
/// memory usage.
#[derive(Debug, Clone)]
pub struct StreamingPipeline {
    config: StreamingConfig,
    pipeline_config: PipelineConfig,
}

impl StreamingPipeline {
    /// Create a new streaming pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingError::InvalidChunkSize`] if `chunk_size` is zero,
    /// or [`StreamingError::OverlapExceedsChunkSize`] if `overlap_pages >= chunk_size`.
    pub fn new(
        config: StreamingConfig,
        pipeline_config: PipelineConfig,
    ) -> Result<Self, StreamingError> {
        if config.chunk_size == 0 {
            return Err(StreamingError::InvalidChunkSize(config.chunk_size));
        }
        if config.overlap_pages >= config.chunk_size {
            return Err(StreamingError::OverlapExceedsChunkSize {
                overlap: config.overlap_pages,
                chunk_size: config.chunk_size,
            });
        }
        Ok(Self {
            config,
            pipeline_config,
        })
    }

    /// Access the streaming configuration.
    #[must_use]
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Access the underlying pipeline configuration.
    #[must_use]
    pub fn pipeline_config(&self) -> &PipelineConfig {
        &self.pipeline_config
    }

    /// Compute chunk boundaries (as page ranges) for a document with
    /// `total_pages` pages.
    ///
    /// Each range represents a half-open interval `[start, end)` of page
    /// indices. Adjacent chunks overlap by `overlap_pages`.
    ///
    /// Returns an empty vec when `total_pages` is zero.
    #[must_use]
    pub fn chunk_pages(&self, total_pages: usize) -> Vec<Range<usize>> {
        if total_pages == 0 {
            return Vec::new();
        }

        let chunk_size = self.config.chunk_size;
        let overlap = self.config.overlap_pages;
        let stride = chunk_size - overlap;

        let mut chunks = Vec::new();
        let mut start = 0;
        while start < total_pages {
            let end = (start + chunk_size).min(total_pages);
            chunks.push(start..end);
            start += stride;
            // If the chunk we just pushed already reaches the end, stop.
            if end == total_pages {
                break;
            }
        }
        chunks
    }

    /// Merge processed chunks into a single [`DocumentOutput`].
    ///
    /// Pages from the overlap zone are deduplicated: for each page that
    /// appears in two adjacent chunks, regions from both copies are merged
    /// and duplicate regions (same class, IoU > dedup threshold from
    /// [`PostProcessConfig`]) are removed, keeping the higher-confidence
    /// detection.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingError::NonContiguousChunks`] if chunks have
    /// unexpected page offsets.
    pub fn merge_chunks(&self, chunks: Vec<ChunkOutput>) -> Result<DocumentOutput, StreamingError> {
        if chunks.is_empty() {
            return Ok(DocumentOutput { pages: Vec::new() });
        }

        let dedup_iou = self.pipeline_config.postprocess_config.dedup_similarity;

        // Determine total page count from the last chunk.
        let last = &chunks[chunks.len() - 1];
        let total_pages = last.page_offset + last.page_outputs.len();

        // Collect pages into document-global slots.
        let mut pages: Vec<Option<PageOutput>> = (0..total_pages).map(|_| None).collect();

        for (ci, chunk) in chunks.iter().enumerate() {
            // Validate contiguity for non-first chunks.
            if ci > 0 {
                let prev = &chunks[ci - 1];
                let expected_start = prev.page_offset + prev.page_outputs.len()
                    - self.config.overlap_pages.min(prev.page_outputs.len());
                // Allow the chunk to start anywhere in the valid overlap zone.
                if chunk.page_offset != expected_start {
                    return Err(StreamingError::NonContiguousChunks {
                        chunk_index: ci,
                        expected: expected_start,
                        actual: chunk.page_offset,
                    });
                }
            }

            for (local_idx, page) in chunk.page_outputs.iter().enumerate() {
                let global_idx = chunk.page_offset + local_idx;
                if global_idx >= total_pages {
                    break;
                }

                match pages[global_idx].take() {
                    None => {
                        pages[global_idx] = Some(page.clone());
                    }
                    Some(existing) => {
                        // Overlap page: merge regions from both chunks and dedup.
                        let merged = merge_overlap_pages(&existing, page, dedup_iou);
                        pages[global_idx] = Some(merged);
                    }
                }
            }
        }

        // Unwrap all pages (they should all be Some after processing).
        let pages = pages.into_iter().flatten().collect();
        Ok(DocumentOutput { pages })
    }

    /// Estimate memory usage (in bytes) for processing a single chunk.
    ///
    /// This is a rough estimate based on:
    /// - Image tensors: `image_width * image_height * 3 * 4` bytes per page
    ///   (3 channels, f32) times pages per chunk.
    /// - Model intermediate activations: approximated as 2x the input tensor
    ///   size per model.
    ///
    /// `num_models` is the number of model passes per page (e.g., layout +
    /// OCR = 2).
    #[must_use]
    pub fn estimate_chunk_memory(
        &self,
        image_width: usize,
        image_height: usize,
        num_models: usize,
    ) -> usize {
        let bytes_per_pixel: usize = 3 * 4; // 3 channels, f32
        let image_bytes = image_width
            .saturating_mul(image_height)
            .saturating_mul(bytes_per_pixel);
        // Input + ~2x activation overhead per model.
        let per_page = image_bytes.saturating_mul(1 + 2_usize.saturating_mul(num_models));
        per_page.saturating_mul(self.config.chunk_size)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Merge two [`PageOutput`]s representing the same page from adjacent chunks.
///
/// Combines regions from both pages, then deduplicates by IoU: among
/// same-class region pairs with IoU above `dedup_iou`, the lower-confidence
/// detection is dropped.
fn merge_overlap_pages(a: &PageOutput, b: &PageOutput, dedup_iou: f32) -> PageOutput {
    let mut regions: Vec<DocumentRegion> = a.regions.clone();

    // Add regions from `b` only if they are not duplicates of existing
    // regions from `a`.
    for region_b in &b.regions {
        let bbox_b = region_b.bbox();
        let is_dup = regions.iter().any(|region_a| {
            region_a.class_name() == region_b.class_name()
                && compute_iou(&region_a.bbox(), &bbox_b) > dedup_iou
        });
        if !is_dup {
            regions.push(region_b.clone());
        } else {
            // If there is a duplicate and b has higher confidence, replace.
            for existing in &mut regions {
                if existing.class_name() == region_b.class_name()
                    && compute_iou(&existing.bbox(), &bbox_b) > dedup_iou
                    && region_b.confidence() > existing.confidence()
                {
                    *existing = region_b.clone();
                    break;
                }
            }
        }
    }

    // Recompute reading order for the merged page.
    let reading_order = crate::dpdf_pipeline::DpdfPipeline::compute_reading_order(&regions);

    PageOutput {
        width: a.width,
        height: a.height,
        regions,
        reading_order,
    }
}

#[cfg(test)]
#[path = "dpdf_streaming_tests.rs"]
mod tests;
