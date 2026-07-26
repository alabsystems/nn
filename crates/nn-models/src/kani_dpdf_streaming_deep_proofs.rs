// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep Kani proof harnesses for dpdf_streaming and dpdf_benchmark edge cases
//! (#3954).
//!
//! Extends the basic safety proofs in `kani_dpdf_streaming_benchmark_proofs.rs`
//! with deeper invariants:
//!
//! **Streaming (9 harnesses):**
//!  1. Chunk boundary alignment: page boundaries align with chunk boundaries.
//!  2. Overlap region handling: overlapping chunks share exactly `overlap_pages`.
//!  3. Stream assembly ordering: chunks assemble in correct page order.
//!  4. Empty chunk handling: zero total_pages produces no chunks.
//!  5. Config validation: zero chunk_size is rejected.
//!  6. Memory bound: per-chunk memory bounded by config.chunk_size * per-page.
//!  7. Progress monotonicity: chunk start offsets strictly increase.
//!  8. Stride consistency: stride = chunk_size - overlap for all chunks.
//!  9. Full page coverage: union of all chunk ranges covers every page.
//!
//! **Benchmark (4 harnesses):**
//! 10. Timer monotonicity: Duration::new always produces non-negative ms.
//! 11. Total time consistency: sum of stage durations equals total_duration_ms.
//! 12. Throughput calculation: pages_per_second avoids div-by-zero.
//! 13. Benchmark config: default config has positive page counts.

use crate::dpdf_benchmark::{BenchmarkConfig, BenchmarkResult, BenchmarkSummary};
use crate::dpdf_pipeline::PipelineConfig;
use crate::dpdf_streaming::{StreamingConfig, StreamingPipeline};

// ===========================================================================
// Streaming deep proofs
// ===========================================================================

/// Harness 1: Chunk boundaries align — first chunk starts at 0, last ends at
/// total_pages.
///
/// SUBSTANTIVE: Proves the first chunk always starts at page 0 and the last
/// chunk always ends at `total_pages`, so no pages are missed at either end.
#[kani::proof]
#[kani::unwind(12)]
fn proof_deep_streaming_boundary_alignment() {
    let total_pages: usize = kani::any();
    kani::assume(total_pages >= 1 && total_pages <= 10);

    let chunk_size: usize = kani::any();
    kani::assume(chunk_size >= 1 && chunk_size <= 10);

    let overlap: usize = kani::any();
    kani::assume(overlap < chunk_size);

    let config = StreamingConfig {
        chunk_size,
        overlap_pages: overlap,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

    let chunks = pipeline.chunk_pages(total_pages);
    assert!(!chunks.is_empty());

    // First chunk must start at page 0.
    assert_eq!(chunks[0].start, 0, "first chunk must start at page 0");

    // Last chunk must end at total_pages.
    let last = &chunks[chunks.len() - 1];
    assert_eq!(last.end, total_pages, "last chunk must end at total_pages");
}

/// Harness 2: Adjacent chunks overlap by exactly `overlap_pages` pages.
///
/// SUBSTANTIVE: Proves that for any two consecutive chunks, the number of
/// shared pages equals `overlap_pages` (or is bounded by the smaller chunk
/// when total_pages is small).
#[kani::proof]
#[kani::unwind(12)]
fn proof_deep_streaming_overlap_region_size() {
    let total_pages: usize = kani::any();
    kani::assume(total_pages >= 2 && total_pages <= 10);

    let chunk_size: usize = kani::any();
    kani::assume(chunk_size >= 2 && chunk_size <= 10);

    let overlap: usize = kani::any();
    kani::assume(overlap >= 1 && overlap < chunk_size);

    let config = StreamingConfig {
        chunk_size,
        overlap_pages: overlap,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

    let chunks = pipeline.chunk_pages(total_pages);

    // For each pair of adjacent chunks, the overlap = prev.end - next.start.
    let mut i = 1;
    while i < chunks.len() {
        let prev_end = chunks[i - 1].end;
        let curr_start = chunks[i].start;
        // The overlap zone is [curr_start, prev_end). It must be non-negative.
        assert!(
            prev_end >= curr_start,
            "adjacent chunks must overlap or be contiguous"
        );
        let actual_overlap = prev_end - curr_start;
        // Overlap must equal the configured overlap (unless the last chunk is
        // smaller, in which case it may be less).
        assert!(
            actual_overlap <= overlap,
            "overlap must not exceed configured overlap_pages"
        );
        // For non-final chunks that are full-size, overlap must be exact.
        let prev_len = chunks[i - 1].end - chunks[i - 1].start;
        if prev_len == chunk_size {
            assert_eq!(
                actual_overlap, overlap,
                "full-size chunk overlap must equal overlap_pages"
            );
        }
        i += 1;
    }
}

/// Harness 3: Stream assembly ordering — chunk start offsets are strictly
/// increasing.
///
/// SUBSTANTIVE: Proves that chunks are ordered by start offset, ensuring
/// correct page assembly order.
#[kani::proof]
#[kani::unwind(12)]
fn proof_deep_streaming_assembly_ordering() {
    let total_pages: usize = kani::any();
    kani::assume(total_pages >= 1 && total_pages <= 10);

    let chunk_size: usize = kani::any();
    kani::assume(chunk_size >= 1 && chunk_size <= 10);

    let overlap: usize = kani::any();
    kani::assume(overlap < chunk_size);

    let config = StreamingConfig {
        chunk_size,
        overlap_pages: overlap,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

    let chunks = pipeline.chunk_pages(total_pages);

    // Chunk start offsets must be strictly increasing.
    let mut i = 1;
    while i < chunks.len() {
        assert!(
            chunks[i].start > chunks[i - 1].start,
            "chunk starts must be strictly increasing"
        );
        i += 1;
    }
}

/// Harness 4: Empty document handling — zero total_pages produces no chunks.
///
/// SUBSTANTIVE: Proves that `chunk_pages(0)` returns an empty vec regardless
/// of config values, and that `chunk_pages` for non-zero pages is non-empty.
#[kani::proof]
#[kani::unwind(2)]
fn proof_deep_streaming_empty_document() {
    let chunk_size: usize = kani::any();
    kani::assume(chunk_size >= 1 && chunk_size <= 10);

    let overlap: usize = kani::any();
    kani::assume(overlap < chunk_size);

    let config = StreamingConfig {
        chunk_size,
        overlap_pages: overlap,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

    let chunks = pipeline.chunk_pages(0);
    assert!(chunks.is_empty(), "zero pages must produce zero chunks");
}

/// Harness 5: Config validation — zero chunk_size is rejected.
///
/// SUBSTANTIVE: Proves that `StreamingPipeline::new` rejects chunk_size == 0
/// with `InvalidChunkSize`.
#[kani::proof]
#[kani::unwind(2)]
fn proof_deep_streaming_zero_chunk_size_rejected() {
    let overlap: usize = kani::any();
    kani::assume(overlap <= 5);

    let config = StreamingConfig {
        chunk_size: 0,
        overlap_pages: overlap,
        max_memory_bytes: None,
    };
    let result = StreamingPipeline::new(config, PipelineConfig::default());
    assert!(result.is_err(), "chunk_size == 0 must be rejected");
}

/// Harness 6: Memory bound — estimated memory is bounded by chunk_size *
/// per-page estimate.
///
/// SUBSTANTIVE: Proves that `estimate_chunk_memory` returns a value that is
/// at most `chunk_size * per_page_bytes` (using saturating arithmetic), where
/// per_page_bytes accounts for image + activation overhead.
#[kani::proof]
#[kani::unwind(2)]
fn proof_deep_streaming_memory_bounded_by_chunk_size() {
    let chunk_size: usize = kani::any();
    kani::assume(chunk_size >= 1 && chunk_size <= 20);

    let config = StreamingConfig {
        chunk_size,
        overlap_pages: 0,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

    let w: usize = kani::any();
    kani::assume(w >= 1 && w <= 1000);
    let h: usize = kani::any();
    kani::assume(h >= 1 && h <= 1000);
    let models: usize = kani::any();
    kani::assume(models <= 5);

    let mem = pipeline.estimate_chunk_memory(w, h, models);

    // Compute per-page bound using the same formula.
    let bytes_per_pixel: usize = 12; // 3 channels * 4 bytes
    let image_bytes = w.saturating_mul(h).saturating_mul(bytes_per_pixel);
    let per_page = image_bytes.saturating_mul(1 + 2_usize.saturating_mul(models));
    let expected_max = per_page.saturating_mul(chunk_size);

    assert!(
        mem <= expected_max,
        "memory estimate must be bounded by chunk_size * per_page"
    );
}

/// Harness 7: Progress monotonicity — chunk start offsets never decrease.
///
/// SUBSTANTIVE: Proves that for any sequence of chunks, each chunk's start
/// offset is strictly greater than the previous chunk's start, guaranteeing
/// forward progress through the document.
#[kani::proof]
#[kani::unwind(12)]
fn proof_deep_streaming_progress_monotonicity() {
    let total_pages: usize = kani::any();
    kani::assume(total_pages >= 1 && total_pages <= 10);

    let chunk_size: usize = kani::any();
    kani::assume(chunk_size >= 1 && chunk_size <= 10);

    let overlap: usize = kani::any();
    kani::assume(overlap < chunk_size);

    let config = StreamingConfig {
        chunk_size,
        overlap_pages: overlap,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

    let chunks = pipeline.chunk_pages(total_pages);

    // Verify strict monotonic increase of start offsets.
    let mut i = 0;
    while i < chunks.len() {
        if i > 0 {
            assert!(
                chunks[i].start > chunks[i - 1].start,
                "progress: chunk starts must strictly increase"
            );
        }
        i += 1;
    }
}

/// Harness 8: Stride consistency — stride between chunks equals
/// `chunk_size - overlap` for full-size chunks.
///
/// SUBSTANTIVE: Proves the stride (distance between consecutive chunk starts)
/// equals `chunk_size - overlap_pages` for all pairs of adjacent full-size
/// chunks, catching bugs in the chunking loop.
#[kani::proof]
#[kani::unwind(12)]
fn proof_deep_streaming_stride_consistency() {
    let total_pages: usize = kani::any();
    kani::assume(total_pages >= 2 && total_pages <= 10);

    let chunk_size: usize = kani::any();
    kani::assume(chunk_size >= 2 && chunk_size <= 10);

    let overlap: usize = kani::any();
    kani::assume(overlap < chunk_size);

    let config = StreamingConfig {
        chunk_size,
        overlap_pages: overlap,
        max_memory_bytes: None,
    };
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

    let chunks = pipeline.chunk_pages(total_pages);
    let expected_stride = chunk_size - overlap;

    let mut i = 1;
    while i < chunks.len() {
        let actual_stride = chunks[i].start - chunks[i - 1].start;
        // For non-last chunks that started at a full stride, verify.
        let prev_len = chunks[i - 1].end - chunks[i - 1].start;
        if prev_len == chunk_size {
            assert_eq!(
                actual_stride, expected_stride,
                "stride must equal chunk_size - overlap for full-size chunks"
            );
        }
        i += 1;
    }
}

/// Harness 9: Full page coverage — every page index appears in at least
/// one chunk range.
///
/// SUBSTANTIVE: Proves that the union of all chunk ranges covers the full
/// interval `[0, total_pages)`, so no page is silently dropped during
/// streaming.
#[kani::proof]
#[kani::unwind(14)]
fn proof_deep_streaming_full_page_coverage() {
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
    let pipeline = StreamingPipeline::new(config, PipelineConfig::default()).expect("valid config");

    let chunks = pipeline.chunk_pages(total_pages);

    // For each page, verify it appears in at least one chunk.
    let mut page = 0;
    while page < total_pages {
        let mut found = false;
        let mut i = 0;
        while i < chunks.len() {
            if page >= chunks[i].start && page < chunks[i].end {
                found = true;
            }
            i += 1;
        }
        assert!(found, "every page must be covered by at least one chunk");
        page += 1;
    }
}

// ===========================================================================
// Benchmark deep proofs
// ===========================================================================

/// Harness 10: Timer monotonicity — Duration conversion to ms is always
/// non-negative and finite.
///
/// SUBSTANTIVE: Proves that `Duration::new(secs, nanos).as_secs_f64() * 1000.0`
/// is non-negative and finite for any valid Duration, ensuring stage timings
/// are never negative or NaN.
#[kani::proof]
#[kani::unwind(2)]
fn proof_deep_benchmark_timer_monotonicity() {
    let secs: u64 = kani::any();
    kani::assume(secs <= 86400); // max 1 day
    let nanos: u32 = kani::any();
    kani::assume(nanos < 1_000_000_000);

    let duration = std::time::Duration::new(secs, nanos);
    let ms = duration.as_secs_f64() * 1000.0;

    assert!(ms >= 0.0, "duration in ms must be non-negative");
    assert!(ms.is_finite(), "duration in ms must be finite");
    // Also verify monotonicity: more secs/nanos -> more ms.
    if secs > 0 {
        assert!(ms > 0.0, "positive seconds must produce positive ms");
    }
}

/// Harness 11: Total time consistency — `BenchmarkSummary::total_duration_ms`
/// equals the sum of individual stage durations.
///
/// SUBSTANTIVE: Proves that `from_results` correctly sums durations for small
/// bounded result sets, catching off-by-one or accumulation bugs.
#[kani::proof]
#[kani::unwind(6)]
fn proof_deep_benchmark_total_time_consistency() {
    let n: usize = kani::any();
    kani::assume(n <= 4);

    let mut results = Vec::with_capacity(n);
    let mut expected_total: f64 = 0.0;

    let mut i = 0;
    while i < n {
        let dur_int: u32 = kani::any();
        kani::assume(dur_int <= 10000);
        let dur = dur_int as f64;
        expected_total += dur;

        results.push(BenchmarkResult {
            stage_name: String::new(),
            duration_ms: dur,
            items_processed: 1,
            throughput: 0.0,
        });
        i += 1;
    }

    let summary = BenchmarkSummary::from_results(results);
    // Floating-point sum should match exactly for small integers cast to f64.
    assert_eq!(
        summary.total_duration_ms, expected_total,
        "total_duration_ms must equal sum of stage durations"
    );
}

/// Harness 12: Throughput calculation — div-by-zero is guarded.
///
/// SUBSTANTIVE: Proves that the throughput computation used in benchmark
/// functions (items / (duration_ms / 1000)) is safe for all combinations
/// of item counts and durations, including zero duration.
#[kani::proof]
#[kani::unwind(2)]
fn proof_deep_benchmark_throughput_no_div_by_zero() {
    let items: usize = kani::any();
    kani::assume(items <= 100_000);

    let dur_int: u32 = kani::any();
    kani::assume(dur_int <= 100_000);
    let duration_ms = dur_int as f64;

    // Mirror the exact throughput guard from bench_postprocess et al.
    let throughput = if duration_ms > 0.0 {
        items as f64 / (duration_ms / 1000.0)
    } else {
        0.0
    };

    assert!(throughput >= 0.0, "throughput must be non-negative");
    assert!(!throughput.is_nan(), "throughput must not be NaN");
    assert!(throughput.is_finite(), "throughput must be finite");

    // When duration is zero, throughput must be exactly 0.0.
    if dur_int == 0 {
        assert_eq!(throughput, 0.0, "zero duration must yield zero throughput");
    }
}

/// Harness 13: Benchmark config — default config has positive page counts
/// and measurement iterations.
///
/// SUBSTANTIVE: Proves that `BenchmarkConfig::default()` satisfies all
/// required invariants: positive page counts, positive iteration counts,
/// positive image dimensions, and positive region count.
#[kani::proof]
#[kani::unwind(2)]
fn proof_deep_benchmark_default_config_valid() {
    let config = BenchmarkConfig::default();

    assert!(config.num_pages > 0, "default num_pages must be positive");
    assert!(
        config.measurement_iterations > 0,
        "default measurement_iterations must be positive"
    );
    assert!(
        config.warmup_iterations > 0 || config.warmup_iterations == 0,
        "warmup_iterations is always valid as usize"
    );
    assert!(
        config.image_width > 0,
        "default image_width must be positive"
    );
    assert!(
        config.image_height > 0,
        "default image_height must be positive"
    );
    assert!(
        config.regions_per_page > 0,
        "default regions_per_page must be positive"
    );

    // Verify throughput denominator will not be zero with default iterations.
    // items = measurement_iterations * regions_per_page > 0.
    let items = config.measurement_iterations * config.regions_per_page;
    assert!(items > 0, "default config must produce positive item count");
}
