// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Kokoro pipeline performance profiling infrastructure.
//!
//! Part of #4264.

use std::time::Duration;

use super::*;

/// Helper: create a segment profile with known values.
fn make_segment(
    name: &str,
    cpu_ms: f64,
    gpu_ms: Option<f64>,
    dispatches: usize,
) -> SegmentProfile {
    SegmentProfile::new(
        name,
        Duration::from_secs_f64(cpu_ms / 1000.0),
        gpu_ms.map(|g| Duration::from_secs_f64(g / 1000.0)),
        dispatches,
        dispatches,
    )
}

/// Helper: build a PipelineProfile from segment list.
fn make_profile(
    segments: Vec<SegmentProfile>,
    wall_ms: f64,
    sample_count: usize,
    flush_count: usize,
    blit_count: usize,
) -> PipelineProfile {
    let total_cpu: Duration = segments.iter().map(|s| s.cpu_time).sum();
    let total_gpu: Option<Duration> = {
        let gpu_times: Vec<Duration> = segments
            .iter()
            .filter_map(|s| s.gpu_time)
            .collect();
        if gpu_times.is_empty() {
            None
        } else {
            Some(gpu_times.into_iter().sum())
        }
    };
    let total_dispatches = segments.iter().map(|s| s.dispatch_count).sum();

    PipelineProfile {
        segments,
        total_wall_time: Duration::from_secs_f64(wall_ms / 1000.0),
        total_cpu_time: total_cpu,
        total_gpu_time: total_gpu,
        total_dispatches,
        total_metal_dispatches: total_dispatches,
        flush_count,
        submit_count: 1,
        blit_count,
        blits_eliminated: 0,
        sample_count,
        sample_rate: 24000,
        cache_misses: 0,
    }
}

#[test]
fn test_segment_profile_effective_time_uses_max() {
    let seg = make_segment("encode", 5.0, Some(10.0), 20);
    // GPU > CPU, so effective = GPU.
    assert_eq!(seg.effective_time(), Duration::from_secs_f64(10.0 / 1000.0));

    let seg2 = make_segment("prosody", 15.0, Some(3.0), 10);
    // CPU > GPU, so effective = CPU.
    assert_eq!(
        seg2.effective_time(),
        Duration::from_secs_f64(15.0 / 1000.0)
    );
}

#[test]
fn test_segment_profile_effective_time_fallback_no_gpu() {
    let seg = make_segment("regulate", 7.0, None, 5);
    assert_eq!(seg.effective_time(), Duration::from_secs_f64(7.0 / 1000.0));
}

#[test]
fn test_segment_profile_gpu_cpu_ratio() {
    let seg = make_segment("generate", 4.0, Some(12.0), 45);
    let ratio = seg.gpu_cpu_ratio().expect("should have ratio");
    assert!((ratio - 3.0).abs() < 1e-6, "expected ~3.0, got {ratio}");
}

#[test]
fn test_segment_profile_gpu_cpu_ratio_none_without_gpu() {
    let seg = make_segment("verify", 2.0, None, 0);
    assert!(seg.gpu_cpu_ratio().is_none());
}

#[test]
fn test_segment_profile_gpu_cpu_ratio_zero_cpu() {
    let seg = SegmentProfile::new("zero", Duration::ZERO, Some(Duration::from_millis(1)), 0, 0);
    assert!(seg.gpu_cpu_ratio().is_none());
}

#[test]
fn test_pipeline_profile_rtf() {
    // 24000 samples = 1 second of audio. 80ms wall time = RTF 0.08.
    let profile = make_profile(
        vec![make_segment("encode", 80.0, None, 100)],
        80.0,
        24000,
        1,
        0,
    );
    let rtf = profile.rtf().expect("should compute RTF");
    assert!((rtf - 0.08).abs() < 1e-6, "expected ~0.08, got {rtf}");
}

#[test]
fn test_pipeline_profile_rtf_zero_samples() {
    let profile = make_profile(
        vec![make_segment("empty", 10.0, None, 0)],
        10.0,
        0,
        1,
        0,
    );
    assert!(profile.rtf().is_none());
}

#[test]
fn test_pipeline_profile_audio_duration() {
    let profile = make_profile(
        vec![make_segment("test", 10.0, None, 10)],
        10.0,
        48000,
        1,
        0,
    );
    assert!((profile.audio_duration_secs() - 2.0).abs() < 1e-9);
}

#[test]
fn test_pipeline_profile_dispatch_gap() {
    let profile = make_profile(
        vec![
            make_segment("encode", 10.0, None, 80),
            make_segment("generate", 20.0, None, 60),
            make_segment("prosody", 5.0, None, 61),
        ],
        35.0,
        24000,
        1,
        0,
    );
    // total = 201, target = 60, gap = 141
    assert_eq!(profile.dispatch_gap(60), 141);
}

#[test]
fn test_pipeline_profile_dispatch_gap_below_target() {
    let profile = make_profile(
        vec![make_segment("small", 5.0, None, 20)],
        5.0,
        24000,
        1,
        0,
    );
    assert_eq!(profile.dispatch_gap(60), 0);
}

#[test]
fn test_pipeline_profile_rtf_gap() {
    let profile = make_profile(
        vec![make_segment("test", 80.0, None, 100)],
        80.0,
        24000,
        1,
        0,
    );
    let gap = profile.rtf_gap(0.03).expect("should compute RTF gap");
    assert!((gap - 0.05).abs() < 1e-6, "expected ~0.05, got {gap}");
}

#[test]
fn test_pipeline_profile_slowest_segments() {
    let profile = make_profile(
        vec![
            make_segment("fast", 2.0, None, 5),
            make_segment("slow", 50.0, None, 30),
            make_segment("medium", 10.0, None, 15),
        ],
        62.0,
        24000,
        1,
        0,
    );
    let slowest = profile.slowest_segments();
    assert_eq!(slowest[0].name, "slow");
    assert_eq!(slowest[1].name, "medium");
    assert_eq!(slowest[2].name, "fast");
}

#[test]
fn test_pipeline_profile_most_dispatches() {
    let profile = make_profile(
        vec![
            make_segment("few", 10.0, None, 5),
            make_segment("many", 10.0, None, 80),
            make_segment("some", 10.0, None, 25),
        ],
        30.0,
        24000,
        1,
        0,
    );
    let most = profile.most_dispatches();
    assert_eq!(most[0].name, "many");
    assert_eq!(most[1].name, "some");
    assert_eq!(most[2].name, "few");
}

#[test]
fn test_identify_bottleneck_dispatch_bound() {
    let profile = make_profile(
        vec![make_segment("heavy", 40.0, Some(40.0), 200)],
        40.0,
        24000,
        1,
        0,
    );
    assert_eq!(identify_bottleneck(&profile), BottleneckKind::DispatchBound);
}

#[test]
fn test_identify_bottleneck_gpu_bound() {
    // Few dispatches, GPU time >> CPU time.
    let profile = make_profile(
        vec![make_segment("gpu_heavy", 5.0, Some(30.0), 40)],
        30.0,
        24000,
        1,
        0,
    );
    assert_eq!(identify_bottleneck(&profile), BottleneckKind::GpuBound);
}

#[test]
fn test_identify_bottleneck_cpu_bound() {
    // Few dispatches, CPU time >> GPU time.
    let profile = make_profile(
        vec![make_segment("cpu_heavy", 30.0, Some(5.0), 40)],
        30.0,
        24000,
        1,
        0,
    );
    assert_eq!(identify_bottleneck(&profile), BottleneckKind::CpuBound);
}

#[test]
fn test_identify_bottleneck_memory_bound() {
    // Few dispatches, balanced GPU/CPU, but many blits relative to encodings.
    let mut profile = make_profile(
        vec![make_segment("balanced", 10.0, Some(10.0), 50)],
        10.0,
        24000,
        1,
        20, // 20 blits out of 70 total (50 + 20) = 28.6% > 10%
    );
    profile.total_metal_dispatches = 50;
    assert_eq!(identify_bottleneck(&profile), BottleneckKind::MemoryBound);
}

#[test]
fn test_identify_bottleneck_unknown_no_gpu_timing() {
    let profile = make_profile(
        vec![make_segment("no_gpu", 10.0, None, 40)],
        10.0,
        24000,
        1,
        0,
    );
    assert_eq!(identify_bottleneck(&profile), BottleneckKind::Unknown);
}

#[test]
fn test_format_profile_report_contains_key_sections() {
    let profile = make_profile(
        vec![
            make_segment("encode", 5.0, Some(8.0), 30),
            make_segment("generate", 20.0, Some(35.0), 80),
            make_segment("istft", 3.0, Some(2.0), 10),
        ],
        35.0,
        24000,
        2,
        3,
    );
    let report = format_profile_report(&profile);

    assert!(report.contains("Pipeline Performance Profile"), "missing header");
    assert!(report.contains("RTF:"), "missing RTF line");
    assert!(report.contains("Wall time:"), "missing wall time");
    assert!(report.contains("Dispatches:"), "missing dispatches");
    assert!(report.contains("Segment"), "missing table header");
    assert!(report.contains("encode"), "missing encode segment");
    assert!(report.contains("generate"), "missing generate segment");
    assert!(report.contains("istft"), "missing istft segment");
    assert!(report.contains("Bottleneck:"), "missing bottleneck");
    assert!(report.contains("Recommendations:"), "missing recommendations");
}

#[test]
fn test_format_profile_report_single_segment() {
    let profile = make_profile(
        vec![make_segment("only", 10.0, None, 20)],
        10.0,
        12000,
        1,
        0,
    );
    let report = format_profile_report(&profile);
    assert!(report.contains("only"));
    assert!(report.contains("Bottleneck:"));
}

#[test]
fn test_format_profile_report_zero_timing() {
    let profile = make_profile(
        vec![make_segment("zero", 0.0, Some(0.0), 0)],
        0.0,
        0,
        0,
        0,
    );
    let report = format_profile_report(&profile);
    // Should not panic; should produce a valid report.
    assert!(report.contains("Pipeline Performance Profile"));
}

#[test]
fn test_dispatches_per_ms() {
    let profile = make_profile(
        vec![make_segment("test", 100.0, None, 200)],
        100.0,
        24000,
        1,
        0,
    );
    // 200 dispatches / 100 ms = 2.0 dispatches/ms
    let rate = profile.dispatches_per_ms();
    assert!((rate - 2.0).abs() < 1e-6, "expected ~2.0, got {rate}");
}

#[test]
fn test_bottleneck_kind_display() {
    assert!(format!("{}", BottleneckKind::GpuBound).contains("GPU-bound"));
    assert!(format!("{}", BottleneckKind::CpuBound).contains("CPU-bound"));
    assert!(format!("{}", BottleneckKind::MemoryBound).contains("Memory-bound"));
    assert!(format!("{}", BottleneckKind::DispatchBound).contains("Dispatch-bound"));
    assert!(format!("{}", BottleneckKind::Unknown).contains("Unknown"));
}
