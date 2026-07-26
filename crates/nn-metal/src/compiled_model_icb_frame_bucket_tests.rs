// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for frame-bucket ICB compatibility.
//!
//! Tests the pure logic in `compiled_model_icb_frame_bucket.rs`:
//! bucket configuration, selection, caching, and replay lookup.
//!
//! Part of #3290.

use super::{
    pre_encode_buckets, try_replay_bucketed, FrameBucketConfig, FrameBucketError,
    FrameBucketIcbCache, FrameBucketSelector, IcbHandle,
};

// ── FrameBucketConfig ──────────────────────────────────────────────────

#[test]
fn config_kokoro_default_has_expected_sizes() {
    let config = FrameBucketConfig::kokoro_default();
    let sizes = config.bucket_sizes();

    assert_eq!(sizes.len(), 22);
    assert_eq!(sizes[0], 32);
    assert_eq!(sizes[sizes.len() - 1], 3072);

    // Verify sorted and strictly increasing.
    for window in sizes.windows(2) {
        assert!(
            window[0] < window[1],
            "bucket sizes must be strictly increasing: {} >= {}",
            window[0],
            window[1]
        );
    }
}

#[test]
fn config_custom_sizes_sorted_and_deduped() {
    let config = FrameBucketConfig::new(vec![256, 64, 128, 64, 256]).expect("valid config");
    assert_eq!(config.bucket_sizes(), &[64, 128, 256]);
}

#[test]
fn config_empty_sizes_error() {
    let result = FrameBucketConfig::new(vec![]);
    assert!(matches!(result, Err(FrameBucketError::EmptyConfig)));
}

#[test]
fn config_zero_sizes_filtered() {
    let result = FrameBucketConfig::new(vec![0, 0, 0]);
    assert!(matches!(result, Err(FrameBucketError::EmptyConfig)));
}

#[test]
fn config_mixed_zero_and_valid() {
    let config = FrameBucketConfig::new(vec![0, 64, 0, 128]).expect("valid config");
    assert_eq!(config.bucket_sizes(), &[64, 128]);
}

#[test]
fn config_max_and_min_bucket() {
    let config = FrameBucketConfig::new(vec![100, 50, 200]).expect("valid config");
    assert_eq!(config.min_bucket(), 50);
    assert_eq!(config.max_bucket(), 200);
}

#[test]
fn config_single_bucket() {
    let config = FrameBucketConfig::new(vec![256]).expect("valid config");
    assert_eq!(config.num_buckets(), 1);
    assert_eq!(config.min_bucket(), 256);
    assert_eq!(config.max_bucket(), 256);
}

#[test]
fn config_default_trait_matches_kokoro() {
    let default_config = FrameBucketConfig::default();
    let kokoro_config = FrameBucketConfig::kokoro_default();
    assert_eq!(default_config.bucket_sizes(), kokoro_config.bucket_sizes());
}

// ── FrameBucketSelector ────────────────────────────────────────────────

#[test]
fn selector_exact_match() {
    let config = FrameBucketConfig::new(vec![64, 128, 256]).expect("valid config");
    let selector = FrameBucketSelector::new(config);

    assert_eq!(selector.select(64), Some(64));
    assert_eq!(selector.select(128), Some(128));
    assert_eq!(selector.select(256), Some(256));
}

#[test]
fn selector_rounds_up_to_next_bucket() {
    let config = FrameBucketConfig::new(vec![64, 128, 256]).expect("valid config");
    let selector = FrameBucketSelector::new(config);

    assert_eq!(selector.select(1), Some(64));
    assert_eq!(selector.select(63), Some(64));
    assert_eq!(selector.select(65), Some(128));
    assert_eq!(selector.select(100), Some(128));
    assert_eq!(selector.select(129), Some(256));
    assert_eq!(selector.select(200), Some(256));
}

#[test]
fn selector_exceeds_max_returns_none() {
    let config = FrameBucketConfig::new(vec![64, 128, 256]).expect("valid config");
    let selector = FrameBucketSelector::new(config);

    assert_eq!(selector.select(257), None);
    assert_eq!(selector.select(1000), None);
}

#[test]
fn selector_zero_returns_none() {
    let config = FrameBucketConfig::new(vec![64, 128]).expect("valid config");
    let selector = FrameBucketSelector::new(config);

    assert_eq!(selector.select(0), None);
}

#[test]
fn selector_padding_frames_exact_is_zero() {
    let config = FrameBucketConfig::new(vec![64, 128, 256]).expect("valid config");
    let selector = FrameBucketSelector::new(config);

    assert_eq!(selector.padding_frames(64), Some(0));
    assert_eq!(selector.padding_frames(128), Some(0));
}

#[test]
fn selector_padding_frames_non_exact() {
    let config = FrameBucketConfig::new(vec![64, 128, 256]).expect("valid config");
    let selector = FrameBucketSelector::new(config);

    assert_eq!(selector.padding_frames(50), Some(14)); // 64 - 50
    assert_eq!(selector.padding_frames(100), Some(28)); // 128 - 100
    assert_eq!(selector.padding_frames(200), Some(56)); // 256 - 200
}

#[test]
fn selector_padding_frames_exceeds_max() {
    let config = FrameBucketConfig::new(vec![64, 128]).expect("valid config");
    let selector = FrameBucketSelector::new(config);

    assert_eq!(selector.padding_frames(200), None);
}

#[test]
fn selector_padding_ratio_exact_is_zero() {
    let config = FrameBucketConfig::new(vec![64, 128]).expect("valid config");
    let selector = FrameBucketSelector::new(config);

    let ratio = selector.padding_ratio(64).expect("should find bucket");
    assert!((ratio - 0.0).abs() < f64::EPSILON);
}

#[test]
fn selector_padding_ratio_half() {
    let config = FrameBucketConfig::new(vec![64, 128]).expect("valid config");
    let selector = FrameBucketSelector::new(config);

    // 32 frames → bucket 64 → 50% padding.
    let ratio = selector.padding_ratio(32).expect("should find bucket");
    assert!((ratio - 0.5).abs() < f64::EPSILON);
}

#[test]
fn selector_padding_ratio_exceeds_max_is_none() {
    let config = FrameBucketConfig::new(vec![64]).expect("valid config");
    let selector = FrameBucketSelector::new(config);

    assert!(selector.padding_ratio(100).is_none());
}

#[test]
fn selector_kokoro_typical_lengths() {
    // Verify Kokoro default config handles typical synthesis lengths.
    let config = FrameBucketConfig::kokoro_default();
    let selector = FrameBucketSelector::new(config);

    // Short utterance: ~50 frames → bucket 64.
    assert_eq!(selector.select(50), Some(64));
    // Medium utterance: ~180 frames → bucket 192.
    assert_eq!(selector.select(180), Some(192));
    // Long utterance: ~500 frames → bucket 512.
    assert_eq!(selector.select(500), Some(512));
    // Very long: ~2000 frames → bucket 2048.
    assert_eq!(selector.select(2000), Some(2048));
    // Max: 3072 frames → exact match.
    assert_eq!(selector.select(3072), Some(3072));
    // Beyond max: returns None.
    assert_eq!(selector.select(3073), None);
}

#[test]
fn selector_config_accessor() {
    let config = FrameBucketConfig::new(vec![64, 128]).expect("valid config");
    let selector = FrameBucketSelector::new(config);
    assert_eq!(selector.config().num_buckets(), 2);
}

// ── IcbHandle ──────────────────────────────────────────────────────────

#[test]
fn icb_handle_properties() {
    let handle = IcbHandle {
        bucket_size: 256,
        cache_index: 3,
    };
    assert_eq!(handle.bucket_size(), 256);
    assert_eq!(handle.cache_index(), 3);
}

#[test]
fn icb_handle_is_copy() {
    let handle = IcbHandle {
        bucket_size: 128,
        cache_index: 1,
    };
    let copy = handle;
    // Both are valid after copy (IcbHandle is Copy).
    assert_eq!(handle.bucket_size(), copy.bucket_size());
    assert_eq!(handle.cache_index(), copy.cache_index());
}

#[test]
fn icb_handle_equality() {
    let a = IcbHandle {
        bucket_size: 128,
        cache_index: 1,
    };
    let b = IcbHandle {
        bucket_size: 128,
        cache_index: 1,
    };
    let c = IcbHandle {
        bucket_size: 256,
        cache_index: 1,
    };
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// ── pre_encode_buckets ─────────────────────────────────────────────────

#[test]
fn pre_encode_all_succeed() {
    let config = FrameBucketConfig::new(vec![64, 128, 256]).expect("valid config");

    let cache = pre_encode_buckets(&config, |bucket_size| {
        // Simulate successful encoding: cache_index = bucket_size / 64.
        Ok(bucket_size / 64)
    });

    assert_eq!(cache.encoded_count(), 3);
    assert_eq!(cache.total_buckets(), 3);
    assert!(cache.is_complete());
    assert!(cache.failed_buckets().is_empty());

    let h64 = cache.get(64).expect("bucket 64 should exist");
    assert_eq!(h64.bucket_size(), 64);
    assert_eq!(h64.cache_index(), 1);

    let h128 = cache.get(128).expect("bucket 128 should exist");
    assert_eq!(h128.bucket_size(), 128);
    assert_eq!(h128.cache_index(), 2);

    let h256 = cache.get(256).expect("bucket 256 should exist");
    assert_eq!(h256.bucket_size(), 256);
    assert_eq!(h256.cache_index(), 4);
}

#[test]
fn pre_encode_partial_failure() {
    let config = FrameBucketConfig::new(vec![64, 128, 256]).expect("valid config");

    let cache = pre_encode_buckets(&config, |bucket_size| {
        if bucket_size == 128 {
            Err("unsupported size".into())
        } else {
            Ok(bucket_size / 64)
        }
    });

    assert_eq!(cache.encoded_count(), 2);
    assert!(!cache.is_complete());
    assert_eq!(cache.failed_buckets(), &[128]);

    assert!(cache.get(64).is_some());
    assert!(cache.get(128).is_none()); // Failed.
    assert!(cache.get(256).is_some());
}

#[test]
fn pre_encode_all_fail() {
    let config = FrameBucketConfig::new(vec![64, 128]).expect("valid config");

    let cache = pre_encode_buckets(&config, |_| Err("all fail".into()));

    assert_eq!(cache.encoded_count(), 0);
    assert!(!cache.is_complete());
    assert_eq!(cache.failed_buckets().len(), 2);
    assert!(cache.get(64).is_none());
    assert!(cache.get(128).is_none());
}

#[test]
fn pre_encode_encoded_bucket_sizes_sorted() {
    let config = FrameBucketConfig::new(vec![256, 64, 128]).expect("valid config");

    let cache = pre_encode_buckets(&config, Ok);

    let sizes = cache.encoded_bucket_sizes();
    assert_eq!(sizes, vec![64, 128, 256]);
}

// ── try_replay_bucketed ────────────────────────────────────────────────

#[test]
fn try_replay_exact_match() {
    let config = FrameBucketConfig::new(vec![64, 128, 256]).expect("valid config");
    let selector = FrameBucketSelector::new(config.clone());
    let cache = pre_encode_buckets(&config, Ok);

    let handle = try_replay_bucketed(&cache, &selector, 128);
    assert!(handle.is_some());
    let h = handle.unwrap();
    assert_eq!(h.bucket_size(), 128);
    assert_eq!(h.cache_index(), 128);
}

#[test]
fn try_replay_rounds_up() {
    let config = FrameBucketConfig::new(vec![64, 128, 256]).expect("valid config");
    let selector = FrameBucketSelector::new(config.clone());
    let cache = pre_encode_buckets(&config, Ok);

    // 100 frames → rounds up to bucket 128.
    let handle = try_replay_bucketed(&cache, &selector, 100);
    assert!(handle.is_some());
    assert_eq!(handle.unwrap().bucket_size(), 128);
}

#[test]
fn try_replay_exceeds_max_returns_none() {
    let config = FrameBucketConfig::new(vec![64, 128]).expect("valid config");
    let selector = FrameBucketSelector::new(config.clone());
    let cache = pre_encode_buckets(&config, Ok);

    let handle = try_replay_bucketed(&cache, &selector, 200);
    assert!(handle.is_none());
}

#[test]
fn try_replay_zero_returns_none() {
    let config = FrameBucketConfig::new(vec![64, 128]).expect("valid config");
    let selector = FrameBucketSelector::new(config.clone());
    let cache = pre_encode_buckets(&config, Ok);

    let handle = try_replay_bucketed(&cache, &selector, 0);
    assert!(handle.is_none());
}

#[test]
fn try_replay_failed_bucket_returns_none() {
    let config = FrameBucketConfig::new(vec![64, 128, 256]).expect("valid config");
    let selector = FrameBucketSelector::new(config.clone());

    // Bucket 128 fails to encode.
    let cache = pre_encode_buckets(&config, |size| {
        if size == 128 {
            Err("failed".into())
        } else {
            Ok(size)
        }
    });

    // 100 frames → rounds up to bucket 128, but it failed → None.
    let handle = try_replay_bucketed(&cache, &selector, 100);
    assert!(handle.is_none());

    // 64 frames → exact match, bucket 64 succeeded → Some.
    let handle = try_replay_bucketed(&cache, &selector, 64);
    assert!(handle.is_some());
}

// ── Thread safety ──────────────────────────────────────────────────────

#[test]
fn cache_is_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<FrameBucketIcbCache>();
    assert_sync::<FrameBucketIcbCache>();
    assert_send::<FrameBucketConfig>();
    assert_sync::<FrameBucketConfig>();
    assert_send::<FrameBucketSelector>();
    assert_sync::<FrameBucketSelector>();
    assert_send::<IcbHandle>();
    assert_sync::<IcbHandle>();
}

// ── Worst-case padding analysis ────────────────────────────────────────

#[test]
fn kokoro_default_worst_case_padding_under_50_percent() {
    // The kokoro_default config documents a <50% worst-case padding guarantee
    // for *typical synthesis lengths* (64-2048 frames). Below the smallest
    // bucket (32), e.g. frame_count=1 padded to 32, padding is inherently high
    // (31/32 = 0.969), so the guarantee is scoped to the typical range.
    let config = FrameBucketConfig::kokoro_default();
    let selector = FrameBucketSelector::new(config);

    let mut worst_ratio = 0.0_f64;
    let mut worst_frame_count = 0;

    for fc in 64..=2048 {
        if let Some(ratio) = selector.padding_ratio(fc) {
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst_frame_count = fc;
            }
        }
    }

    assert!(
        worst_ratio < 0.5,
        "worst-case padding ratio {worst_ratio:.3} at frame_count={worst_frame_count} exceeds 50%"
    );
}

#[test]
fn kokoro_default_typical_range_padding_under_25_percent() {
    // For typical Kokoro synthesis (64-2048 frames), padding should
    // generally be reasonable. Check that the average is under 25%.
    let config = FrameBucketConfig::kokoro_default();
    let selector = FrameBucketSelector::new(config);

    let mut total_ratio = 0.0_f64;
    let mut count = 0;

    for fc in 64..=2048 {
        if let Some(ratio) = selector.padding_ratio(fc) {
            total_ratio += ratio;
            count += 1;
        }
    }

    let avg_ratio = total_ratio / count as f64;
    assert!(
        avg_ratio < 0.25,
        "average padding ratio {avg_ratio:.3} in typical range exceeds 25%"
    );
}
