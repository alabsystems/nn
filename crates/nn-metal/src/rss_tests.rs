// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `rss.rs` — RSS and Metal GPU allocation measurement.

use super::*;

#[test]
fn test_rss_bytes_returns_some_on_macos() {
    let rss = rss_bytes();
    // On macOS, we should always get a value. On other platforms, None.
    if cfg!(target_os = "macos") {
        let bytes = rss.expect("mach_task_info should succeed");
        // RSS should be at least a few MB for any process.
        assert!(bytes > 1_000_000, "RSS too small: {bytes}");
        // And less than 100 GB (sanity check).
        assert!(bytes < 100_000_000_000, "RSS too large: {bytes}");
    }
}

#[test]
fn test_rss_mb_conversion() {
    let mb = rss_mb();
    if cfg!(target_os = "macos") {
        let val = mb.expect("should get RSS on macOS");
        assert!(val > 1.0, "RSS should be > 1 MB: {val}");
    }
}

#[test]
fn test_tracker_checkpoints() {
    let mut tracker = RssTracker::new();
    tracker.checkpoint("start");
    // Allocate some memory to see a delta.
    let _big: Vec<u8> = vec![0u8; 10_000_000];
    tracker.checkpoint("after_alloc");

    if cfg!(target_os = "macos") {
        assert_eq!(tracker.snapshots().len(), 2);
        assert_eq!(tracker.snapshots()[0].label, "start");
        assert_eq!(tracker.snapshots()[1].label, "after_alloc");
        assert!(tracker.peak_rss_bytes().is_some());
    }
}

#[test]
fn test_tracker_display() {
    let mut tracker = RssTracker::new();
    tracker.checkpoint("a");
    tracker.checkpoint("b");
    let s = format!("{tracker}");
    if cfg!(target_os = "macos") {
        assert!(s.contains("Memory Profile"));
        assert!(s.contains("a"));
        assert!(s.contains("b"));
    }
}

#[test]
fn test_delta_bytes() {
    let tracker = RssTracker {
        snapshots: vec![
            RssSnapshot {
                label: "a".into(),
                rss_bytes: 100_000_000,
                metal_bytes: Some(50_000_000),
            },
            RssSnapshot {
                label: "b".into(),
                rss_bytes: 150_000_000,
                metal_bytes: Some(80_000_000),
            },
        ],
    };
    assert_eq!(tracker.delta_bytes("a", "b"), Some(50_000_000));
    assert_eq!(tracker.delta_bytes("b", "a"), Some(-50_000_000));
    assert_eq!(tracker.delta_bytes("a", "nonexistent"), None);
    assert_eq!(tracker.peak_metal_bytes(), Some(80_000_000));
}
