// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Frame-bucket ICB compatibility for variable-length Kokoro synthesis.
//!
//! Kokoro synthesis handles variable-length inputs (different text lengths
//! produce different frame counts). Frame-bucket ICB pre-encodes ICBs for
//! common frame sizes so most synthesis calls can use ICB replay instead of
//! re-encoding dispatch commands each time.
//!
//! # Design
//!
//! - [`FrameBucketConfig`] defines the set of bucket sizes (e.g., powers of
//!   two, common audio frame lengths).
//! - [`FrameBucketSelector`] maps an arbitrary frame count to the smallest
//!   bucket that can contain it, with zero-padding for the remainder.
//! - [`FrameBucketIcbCache`] stores pre-encoded [`IcbHandle`]s keyed by
//!   bucket size. Immutable after construction for thread safety.
//! - [`pre_encode_buckets`] pre-encodes ICBs for all bucket sizes at model
//!   build time.
//! - [`try_replay_bucketed`] finds the matching bucket for a given frame
//!   count and returns the pre-encoded ICB handle.
//!
//! # Padding Invariant
//!
//! Padded frames (bucket_size > actual_frame_count) are zero-filled. The
//! model produces identical results for the valid prefix whether using ICB
//! replay or direct dispatch, because:
//! 1. Zero-padding does not contribute to output for supported ops (conv,
//!    elementwise, attention with masking).
//! 2. The caller truncates the output to the actual frame count after
//!    dispatch.
//!
//! Part of #3290.

use std::collections::HashMap;

/// Defines the set of frame-count bucket sizes for ICB pre-encoding.
///
/// Each bucket size represents a frame count for which an ICB can be
/// pre-encoded. At synthesis time, the input is padded to the smallest
/// bucket >= the actual frame count. The default configuration covers
/// common Kokoro synthesis lengths from 32 to 3072 frames.
#[derive(Debug, Clone)]
pub(crate) struct FrameBucketConfig {
    /// Sorted (ascending) bucket sizes. Each value is a frame count.
    /// Must be non-empty and strictly increasing.
    bucket_sizes: Vec<usize>,
}

impl FrameBucketConfig {
    /// Create a new config with the given bucket sizes.
    ///
    /// The sizes are sorted and deduplicated. Returns an error if the
    /// resulting set is empty.
    pub(crate) fn new(mut sizes: Vec<usize>) -> Result<Self, FrameBucketError> {
        sizes.sort_unstable();
        sizes.dedup();
        sizes.retain(|&s| s > 0);
        if sizes.is_empty() {
            return Err(FrameBucketError::EmptyConfig);
        }
        Ok(Self {
            bucket_sizes: sizes,
        })
    }

    /// The default Kokoro bucket configuration.
    ///
    /// Covers frame counts from 32 to 3072 with geometrically increasing
    /// spacing. Chosen to keep worst-case padding overhead below 50% for
    /// typical synthesis lengths (64-2048 frames).
    pub(crate) fn kokoro_default() -> Self {
        Self {
            bucket_sizes: vec![
                32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640,
                768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072,
            ],
        }
    }

    /// Returns the sorted bucket sizes.
    pub(crate) fn bucket_sizes(&self) -> &[usize] {
        &self.bucket_sizes
    }

    /// Number of configured buckets.
    pub(crate) fn num_buckets(&self) -> usize {
        self.bucket_sizes.len()
    }

    /// The largest bucket size. Frames exceeding this cannot use ICB replay.
    pub(crate) fn max_bucket(&self) -> usize {
        // bucket_sizes is non-empty and sorted ascending.
        self.bucket_sizes[self.bucket_sizes.len() - 1]
    }

    /// The smallest bucket size.
    pub(crate) fn min_bucket(&self) -> usize {
        self.bucket_sizes[0]
    }
}

impl Default for FrameBucketConfig {
    fn default() -> Self {
        Self::kokoro_default()
    }
}

/// Selects the optimal bucket for a given frame count.
///
/// Uses binary search over the sorted bucket sizes to find the smallest
/// bucket >= the requested frame count. Returns `None` if the frame count
/// exceeds the largest bucket.
#[derive(Debug, Clone)]
pub(crate) struct FrameBucketSelector {
    config: FrameBucketConfig,
}

impl FrameBucketSelector {
    /// Create a selector from a bucket configuration.
    pub(crate) fn new(config: FrameBucketConfig) -> Self {
        Self { config }
    }

    /// Select the smallest bucket >= `frame_count`.
    ///
    /// Returns `None` if `frame_count` exceeds the largest bucket or is 0.
    pub(crate) fn select(&self, frame_count: usize) -> Option<usize> {
        if frame_count == 0 {
            return None;
        }
        let sizes = &self.config.bucket_sizes;
        // Binary search: find the first size >= frame_count.
        match sizes.binary_search(&frame_count) {
            Ok(_) => Some(frame_count), // Exact match.
            Err(idx) => {
                if idx < sizes.len() {
                    Some(sizes[idx])
                } else {
                    None // Exceeds largest bucket.
                }
            }
        }
    }

    /// Returns the number of padding frames for a given frame count.
    ///
    /// `None` if no bucket can accommodate the frame count.
    pub(crate) fn padding_frames(&self, frame_count: usize) -> Option<usize> {
        self.select(frame_count)
            .map(|bucket| bucket - frame_count)
    }

    /// Returns the padding overhead ratio (0.0 to < 1.0).
    ///
    /// `None` if no bucket can accommodate the frame count.
    pub(crate) fn padding_ratio(&self, frame_count: usize) -> Option<f64> {
        self.select(frame_count).map(|bucket| {
            if bucket == 0 {
                0.0
            } else {
                (bucket - frame_count) as f64 / bucket as f64
            }
        })
    }

    /// Reference to the underlying config.
    pub(crate) fn config(&self) -> &FrameBucketConfig {
        &self.config
    }
}

/// A lightweight handle to a pre-encoded ICB for a specific bucket size.
///
/// Opaque type that stores the bucket size and an internal index into
/// the cache's ICB storage. Used by [`try_replay_bucketed`] to return
/// a reference to the correct pre-encoded ICB.
///
/// Immutable after creation. `Clone` and `Copy` for efficient passing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct IcbHandle {
    /// The frame-count bucket this ICB was encoded for.
    bucket_size: usize,
    /// Internal index into the cache's pre-encoded ICB list.
    cache_index: usize,
}

impl IcbHandle {
    /// The bucket size this handle was pre-encoded for.
    pub(crate) fn bucket_size(&self) -> usize {
        self.bucket_size
    }

    /// Internal cache index (for cache lookup).
    pub(crate) fn cache_index(&self) -> usize {
        self.cache_index
    }
}

/// Cache of pre-encoded ICBs keyed by bucket size.
///
/// Immutable after construction: all ICBs are pre-encoded at build time
/// via [`pre_encode_buckets`]. Thread-safe for concurrent read access
/// (no interior mutability).
///
/// The cache stores [`IcbHandle`]s in a `HashMap<usize, IcbHandle>` for
/// O(1) lookup by bucket size. The actual ICB encoding data is owned by
/// the handles.
#[derive(Debug)]
pub(crate) struct FrameBucketIcbCache {
    /// Pre-encoded ICB handles keyed by bucket size.
    handles: HashMap<usize, IcbHandle>,
    /// The bucket configuration used to build this cache.
    config: FrameBucketConfig,
    /// Number of successfully pre-encoded buckets.
    encoded_count: usize,
    /// Bucket sizes that failed pre-encoding (non-fatal).
    failed_buckets: Vec<usize>,
}

// SAFETY: FrameBucketIcbCache is immutable after construction. All fields
// are Send+Sync (HashMap, Vec, FrameBucketConfig are all Send+Sync, and
// IcbHandle is Copy). No interior mutability.
unsafe impl Sync for FrameBucketIcbCache {}

impl FrameBucketIcbCache {
    /// Look up the pre-encoded ICB handle for a specific bucket size.
    ///
    /// Returns `None` if no ICB was pre-encoded for this bucket size
    /// (either because it was not in the config or pre-encoding failed).
    pub(crate) fn get(&self, bucket_size: usize) -> Option<&IcbHandle> {
        self.handles.get(&bucket_size)
    }

    /// Number of successfully pre-encoded buckets.
    pub(crate) fn encoded_count(&self) -> usize {
        self.encoded_count
    }

    /// Total number of configured buckets (including failed ones).
    pub(crate) fn total_buckets(&self) -> usize {
        self.config.num_buckets()
    }

    /// Bucket sizes that failed pre-encoding.
    pub(crate) fn failed_buckets(&self) -> &[usize] {
        &self.failed_buckets
    }

    /// Whether all configured buckets were successfully pre-encoded.
    pub(crate) fn is_complete(&self) -> bool {
        self.failed_buckets.is_empty()
    }

    /// Reference to the underlying config.
    pub(crate) fn config(&self) -> &FrameBucketConfig {
        &self.config
    }

    /// Returns all successfully encoded bucket sizes, sorted ascending.
    pub(crate) fn encoded_bucket_sizes(&self) -> Vec<usize> {
        let mut sizes: Vec<usize> = self.handles.keys().copied().collect();
        sizes.sort_unstable();
        sizes
    }
}

/// Pre-encode ICBs for all bucket sizes in the config.
///
/// For each bucket size, invokes `encode_fn(bucket_size)` which should
/// return the cache index of the pre-encoded ICB (or an error if
/// pre-encoding fails for that size).
///
/// Failed buckets are recorded but do not prevent the cache from being
/// used — synthesis for those frame counts will fall back to direct
/// dispatch.
///
/// # Arguments
///
/// * `config` - The bucket configuration defining which sizes to pre-encode.
/// * `encode_fn` - A closure that pre-encodes an ICB for the given bucket
///   size and returns the cache index on success.
///
/// Part of #3290.
pub(crate) fn pre_encode_buckets<F>(
    config: &FrameBucketConfig,
    mut encode_fn: F,
) -> FrameBucketIcbCache
where
    F: FnMut(usize) -> Result<usize, String>,
{
    let mut handles = HashMap::with_capacity(config.num_buckets());
    let mut failed_buckets = Vec::new();

    for &bucket_size in config.bucket_sizes() {
        match encode_fn(bucket_size) {
            Ok(cache_index) => {
                handles.insert(
                    bucket_size,
                    IcbHandle {
                        bucket_size,
                        cache_index,
                    },
                );
            }
            Err(reason) => {
                eprintln!(
                    "[nn-metal] ICB frame-bucket pre-encode failed for size {bucket_size}: {reason}"
                );
                failed_buckets.push(bucket_size);
            }
        }
    }

    let encoded_count = handles.len();
    FrameBucketIcbCache {
        handles,
        config: config.clone(),
        encoded_count,
        failed_buckets,
    }
}

/// Find and return the ICB handle for the bucket matching `frame_count`.
///
/// Uses the selector to find the smallest bucket >= `frame_count`, then
/// looks up the pre-encoded ICB in the cache. Returns `None` if:
/// - `frame_count` exceeds the largest bucket
/// - No ICB was pre-encoded for the matching bucket
///
/// # Arguments
///
/// * `cache` - The pre-encoded frame-bucket ICB cache.
/// * `selector` - The bucket selector (wraps the same config as the cache).
/// * `frame_count` - The actual number of frames for this synthesis call.
///
/// Part of #3290.
pub(crate) fn try_replay_bucketed(
    cache: &FrameBucketIcbCache,
    selector: &FrameBucketSelector,
    frame_count: usize,
) -> Option<IcbHandle> {
    let bucket_size = selector.select(frame_count)?;
    cache.get(bucket_size).copied()
}

/// Errors from frame-bucket ICB operations.
#[derive(Debug, Clone, thiserror::Error)]
pub(crate) enum FrameBucketError {
    /// The bucket configuration has no valid sizes.
    #[error("frame bucket config is empty (no valid bucket sizes)")]
    EmptyConfig,
    /// A frame count exceeds the largest bucket size.
    #[error("frame count {frame_count} exceeds max bucket {max_bucket}")]
    ExceedsMaxBucket {
        frame_count: usize,
        max_bucket: usize,
    },
}

#[cfg(test)]
#[path = "compiled_model_icb_frame_bucket_tests.rs"]
mod tests;
