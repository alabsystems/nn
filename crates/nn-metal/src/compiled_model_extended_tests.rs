// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended infrastructure tests for compiled model subsystems.
//!
//! Tests cover builder patterns, cache config presets, cache statistics,
//! buffer pool stats, F16 autocast config, and precompile shapes -- all
//! without requiring a live Metal GPU context (except builder tests that
//! need a PipelineCache, which requires MetalContext on macOS).
//!
//! Part of #4186.

use nn_core::mixed_precision::MixedPrecisionPolicy;

use crate::buffer_pool_size_class::{
    SizeClassAllocator, SizeClassStats, NUM_SIZE_CLASSES, SIZE_CLASS_BOUNDARIES,
};
use crate::cache_stats::{CacheStats, CacheStatsSnapshot};
use crate::compiled_kokoro::precompile::PrecompileShapes;
use crate::segment_cache::{EvictionPolicy, SegmentCacheConfig, SegmentCacheStats, ShapeKeyedCache};
use crate::F16AutocastConfig;

// ═══════════════════════════════════════════════════════════════════════
// 1. CompiledModelBuilder -- builder pattern
// ═══════════════════════════════════════════════════════════════════════

/// Builder pattern: chaining with_peephole_config then build on an empty
/// graph succeeds (config is stored and used during compilation).
#[test]
fn builder_ext_with_peephole_config_builds_ok() {
    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![]);
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let config = nn_dsl::PeepholeConfig::default();
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .with_peephole_config(config)
        .build()
        .expect("build with peephole config should succeed");
    assert_eq!(model.num_steps(), 0, "empty graph produces 0-step model");
}

/// Builder pattern: chaining optimize then build on an empty graph succeeds.
#[test]
fn builder_ext_optimize_builds_ok() {
    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![]);
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let budget = std::time::Duration::from_secs(2);
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .optimize(budget)
        .build()
        .expect("build with optimize budget should succeed on empty graph");
    assert_eq!(model.num_steps(), 0);
}

/// Builder: building an empty graph with default config produces a valid
/// model with zero steps.
#[test]
fn builder_ext_build_empty_graph_default_config() {
    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![]);
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .build()
        .expect("empty graph build should succeed");
    assert_eq!(model.num_steps(), 0);
    assert_eq!(model.num_dispatches(), 0);
}

/// Builder: building an empty graph with a peephole config produces
/// a valid empty model.
#[test]
fn builder_ext_build_empty_graph_with_peephole() {
    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![]);
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let config = nn_dsl::PeepholeConfig::default();
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .with_peephole_config(config)
        .build()
        .expect("empty graph with peephole config should succeed");
    assert_eq!(model.num_steps(), 0);
}

/// Builder: force_dtype and autocast are mutually exclusive. When both
/// are set, build() should fail with an InvalidConfig error.
#[test]
fn builder_ext_force_dtype_and_autocast_mutually_exclusive() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let policy = MixedPrecisionPolicy::apple_silicon_default();

    use nn_core::dyn_tensor::DynTensor;
    use nn_core::{Device, DType};

    // Trace a simple graph with one op so the mutual-exclusion check runs.
    let (_, traced_graph) = nn_core::dyn_tensor::trace::trace_graph(|| {
        let input = DynTensor::zeros(&[1, 10], DType::F32, &Device::Cpu)?;
        let ones = DynTensor::ones(&[1, 10], DType::F32, &Device::Cpu)?;
        let output = input.add(&ones)?;
        Ok(output)
    })
    .expect("trace should succeed");

    let result = crate::compiled_model::CompiledModel::builder(&traced_graph, &cache)
        .force_dtype(DType::F16)
        .expect("F16 should be accepted")
        .autocast(policy)
        .build();
    assert!(result.is_err(), "force_dtype + autocast should conflict");
}

// ═══════════════════════════════════════════════════════════════════════
// 2. DispatchConfig validation -- PeepholeConfig defaults
// ═══════════════════════════════════════════════════════════════════════

/// Default PeepholeConfig has all passes enabled.
#[test]
fn peephole_config_ext_default_all_passes_enabled() {
    let config = nn_dsl::PeepholeConfig::default();
    assert!(config.auto_fuse_elementwise, "auto_fuse_elementwise default should be true");
    assert!(config.add_norm_linear, "add_norm_linear default should be true");
    assert!(config.fuse_adain_snake, "fuse_adain_snake default should be true");
}

/// PeepholeConfig can be modified via field assignment.
#[test]
fn peephole_config_ext_field_modification() {
    let config = nn_dsl::PeepholeConfig {
        auto_fuse_elementwise: false,
        ..Default::default()
    };
    assert!(!config.auto_fuse_elementwise);
    // Other fields remain at default.
    assert!(config.add_norm_linear);
}

// ═══════════════════════════════════════════════════════════════════════
// 3. PipelineCache stats -- CacheStats / CacheStatsSnapshot
// ═══════════════════════════════════════════════════════════════════════

/// Default CacheStatsSnapshot has all-zero fields.
#[test]
fn cache_stats_ext_snapshot_default_is_zero() {
    let snap = CacheStatsSnapshot::default();
    assert_eq!(snap.kernel_cache_hits, 0);
    assert_eq!(snap.kernel_cache_misses, 0);
    assert_eq!(snap.msl_cache_hits, 0);
    assert_eq!(snap.msl_cache_misses, 0);
    assert_eq!(snap.pipeline_cache_hits, 0);
    assert_eq!(snap.pipeline_cache_misses, 0);
    assert_eq!(snap.total_dispatches, 0);
    assert_eq!(snap.total_compile_time_us, 0);
}

/// Global CacheStats: recording hits/misses changes the snapshot.
/// Uses reset/snapshot around global singleton to avoid cross-test interference.
#[test]
fn cache_stats_ext_global_record_and_snapshot() {
    let stats = CacheStats::global();
    stats.reset();

    stats.record_kernel_hit();
    stats.record_kernel_hit();
    stats.record_kernel_miss();

    stats.record_msl_hit();
    stats.record_msl_miss();
    stats.record_msl_miss();

    stats.record_pipeline_hit();
    stats.record_pipeline_hit();
    stats.record_pipeline_hit();
    stats.record_pipeline_miss();

    stats.record_dispatch();
    stats.record_dispatch();

    stats.record_compile(500);
    stats.record_compile(300);

    let snap = stats.snapshot();
    // Use >= because the global singleton may have concurrent activity.
    assert!(snap.kernel_cache_hits >= 2);
    assert!(snap.kernel_cache_misses >= 1);
    assert!(snap.msl_cache_hits >= 1);
    assert!(snap.msl_cache_misses >= 2);
    assert!(snap.pipeline_cache_hits >= 3);
    assert!(snap.pipeline_cache_misses >= 1);
    assert!(snap.total_dispatches >= 2);
    assert!(snap.total_compile_time_us >= 800);
}

/// Hit rate computation from snapshot.
#[test]
fn cache_stats_ext_snapshot_hit_rate() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 8,
        kernel_cache_misses: 2,
        msl_cache_hits: 0,
        msl_cache_misses: 0,
        pipeline_cache_hits: 0,
        pipeline_cache_misses: 0,
        total_dispatches: 0,
        total_compile_time_us: 0,
    };
    let rate = snap.kernel_hit_rate();
    assert!((rate - 0.8).abs() < 1e-9, "kernel hit rate should be 0.8, got {rate}");
}

/// Hit rate is 0.0 when no lookups.
#[test]
fn cache_stats_ext_snapshot_hit_rate_zero_lookups() {
    let snap = CacheStatsSnapshot::default();
    assert_eq!(snap.hit_rate(), 0.0);
    assert_eq!(snap.kernel_hit_rate(), 0.0);
    assert_eq!(snap.msl_hit_rate(), 0.0);
    assert_eq!(snap.pipeline_hit_rate(), 0.0);
}

/// Average compile time computation.
#[test]
fn cache_stats_ext_avg_compile_time() {
    let snap = CacheStatsSnapshot {
        pipeline_cache_misses: 4,
        total_compile_time_us: 1000,
        ..Default::default()
    };
    assert!((snap.avg_compile_time_us() - 250.0).abs() < 1e-9);
}

/// Average compile time is 0 when no compilations.
#[test]
fn cache_stats_ext_avg_compile_time_zero() {
    let snap = CacheStatsSnapshot::default();
    assert_eq!(snap.avg_compile_time_us(), 0.0);
}

/// Reset on global singleton clears all counters.
#[test]
fn cache_stats_ext_global_reset() {
    let stats = CacheStats::global();
    stats.record_kernel_hit();
    stats.record_pipeline_miss();
    stats.record_dispatch();
    stats.reset();
    let snap = stats.snapshot();
    assert_eq!(snap.kernel_cache_hits, 0);
    assert_eq!(snap.pipeline_cache_misses, 0);
    assert_eq!(snap.total_dispatches, 0);
}

/// Summary string is non-empty and contains key terms.
#[test]
fn cache_stats_ext_snapshot_summary_format() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 5,
        kernel_cache_misses: 1,
        msl_cache_hits: 3,
        msl_cache_misses: 0,
        pipeline_cache_hits: 10,
        pipeline_cache_misses: 2,
        total_dispatches: 50,
        total_compile_time_us: 12000,
    };
    let summary = snap.summary();
    assert!(summary.contains("Kernel Def"), "summary should mention Kernel Def");
    assert!(summary.contains("MSL Codegen"), "summary should mention MSL Codegen");
    assert!(summary.contains("Pipeline"), "summary should mention Pipeline");
    assert!(summary.contains("Dispatches"), "summary should mention Dispatches");
}

// ═══════════════════════════════════════════════════════════════════════
// 4. BufferPool stats -- SizeClassAllocator
// ═══════════════════════════════════════════════════════════════════════

/// Fresh allocator has all-zero stats.
#[test]
fn buffer_pool_ext_fresh_stats_zero() {
    let alloc = SizeClassAllocator::new();
    let stats = alloc.stats();
    assert_eq!(stats.oversized_allocs, 0);
    assert_eq!(stats.total_free_bytes, 0);
    assert_eq!(stats.total_used_bytes, 0);
    assert_eq!(stats.hit_rate, 0.0);
    assert_eq!(stats.fragmentation_ratio, 0.0);
    for i in 0..NUM_SIZE_CLASSES {
        assert_eq!(stats.per_class[i].hits, 0);
        assert_eq!(stats.per_class[i].misses, 0);
    }
}

/// Allocation tracks misses (cold start) and updates in-use counts.
#[test]
fn buffer_pool_ext_allocation_miss_tracking() {
    let mut alloc = SizeClassAllocator::new();
    // Allocate 100 bytes -> class 0 (4 KB), should be a miss.
    let result = alloc.allocate(100).expect("should fit in a size class");
    assert_eq!(result.class, 0);
    assert_eq!(result.alloc_bytes, SIZE_CLASS_BOUNDARIES[0]);
    assert!(!result.reused, "first allocation should not be reused");

    let stats = alloc.stats();
    assert_eq!(stats.per_class[0].misses, 1);
    assert_eq!(stats.per_class[0].hits, 0);
    assert_eq!(stats.per_class[0].in_use_count, 1);
    assert_eq!(stats.per_class[0].peak_in_use, 1);
}

/// Deallocation + re-allocation produces a hit.
#[test]
fn buffer_pool_ext_reuse_tracking() {
    let mut alloc = SizeClassAllocator::new();
    let result = alloc.allocate(100).expect("class 0");
    assert!(!result.reused);

    // Deallocate, then allocate again -> should be a hit.
    assert!(alloc.deallocate(result.class));
    let result2 = alloc.allocate(50).expect("class 0 again");
    assert!(result2.reused, "second allocation should reuse freed buffer");
    assert_eq!(result2.class, 0);

    let stats = alloc.stats();
    assert_eq!(stats.per_class[0].hits, 1);
    assert_eq!(stats.per_class[0].misses, 1);
}

/// Oversized allocations are tracked.
#[test]
fn buffer_pool_ext_oversized_tracking() {
    let mut alloc = SizeClassAllocator::new();
    // 100 MB exceeds the 64 MB maximum class.
    let result = alloc.allocate(100 * 1024 * 1024);
    assert!(result.is_none(), "100 MB should be oversized");
    alloc.record_oversized();
    let stats = alloc.stats();
    assert_eq!(stats.oversized_allocs, 1);
}

/// Hit rate is correct after mixed operations.
#[test]
fn buffer_pool_ext_hit_rate_computation() {
    let mut alloc = SizeClassAllocator::new();
    // 3 misses then 1 dealloc + 1 reuse = 1 hit out of 4 total.
    alloc.allocate(100); // miss
    alloc.allocate(100); // miss
    let r = alloc.allocate(100).unwrap(); // miss
    alloc.deallocate(r.class);
    alloc.allocate(100); // hit

    let stats = alloc.stats();
    assert_eq!(stats.per_class[0].hits, 1);
    assert_eq!(stats.per_class[0].misses, 3);
    let expected = 1.0 / 4.0;
    assert!(
        (stats.per_class[0].hit_rate() - expected).abs() < 1e-9,
        "hit rate should be 0.25"
    );
}

/// SizeClassStats::total_allocs is hits + misses.
#[test]
fn size_class_stats_ext_total_allocs() {
    let stats = SizeClassStats {
        hits: 7,
        misses: 3,
        free_count: 0,
        in_use_count: 0,
        peak_in_use: 0,
        free_bytes: 0,
    };
    assert_eq!(stats.total_allocs(), 10);
}

/// Reset clears all allocator state.
#[test]
fn buffer_pool_ext_reset() {
    let mut alloc = SizeClassAllocator::new();
    alloc.allocate(100);
    alloc.record_oversized();
    alloc.reset();
    let stats = alloc.stats();
    assert_eq!(stats.oversized_allocs, 0);
    assert_eq!(stats.total_used_bytes, 0);
    for i in 0..NUM_SIZE_CLASSES {
        assert_eq!(stats.per_class[i].hits, 0);
        assert_eq!(stats.per_class[i].misses, 0);
        assert_eq!(stats.per_class[i].in_use_count, 0);
    }
}

/// Fragmentation ratio is computed correctly.
#[test]
fn buffer_pool_ext_fragmentation_ratio() {
    let mut alloc = SizeClassAllocator::new();
    // Allocate two buffers, deallocate one -> 50% free.
    let r1 = alloc.allocate(100).unwrap(); // class 0: 4 KB
    let _r2 = alloc.allocate(100).unwrap(); // class 0: 4 KB
    alloc.deallocate(r1.class);

    let stats = alloc.stats();
    // 1 free (4 KB) and 1 in-use (4 KB) -> fragmentation = 4096 / 8192 = 0.5.
    assert!(
        (stats.fragmentation_ratio - 0.5).abs() < 1e-9,
        "fragmentation should be 0.5, got {}",
        stats.fragmentation_ratio
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 5. SegmentCacheConfig presets
// ═══════════════════════════════════════════════════════════════════════

/// interactive() preset: 2 segments, 256 MB budget.
#[test]
fn segment_cache_config_ext_interactive_preset() {
    let config = SegmentCacheConfig::interactive();
    assert_eq!(config.max_segments_per_step, 2);
    assert_eq!(config.eviction, EvictionPolicy::Lru);
    assert_eq!(config.byte_budget, Some(256 * 1024 * 1024));
    assert!(config.shared_store.is_none());
}

/// batch() preset: 8 segments, 1024 MB budget.
#[test]
fn segment_cache_config_ext_batch_preset() {
    let config = SegmentCacheConfig::batch();
    assert_eq!(config.max_segments_per_step, 8);
    assert_eq!(config.byte_budget, Some(1024 * 1024 * 1024));
}

/// chorus() preset: 6 segments, 768 MB budget.
#[test]
fn segment_cache_config_ext_chorus_preset() {
    let config = SegmentCacheConfig::chorus();
    assert_eq!(config.max_segments_per_step, 6);
    assert_eq!(config.byte_budget, Some(768 * 1024 * 1024));
}

/// minimal() preset: 1 segment, 128 MB budget.
#[test]
fn segment_cache_config_ext_minimal_preset() {
    let config = SegmentCacheConfig::minimal();
    assert_eq!(config.max_segments_per_step, 1);
    assert_eq!(config.byte_budget, Some(128 * 1024 * 1024));
}

/// Presets have strictly increasing byte budgets from minimal to batch.
#[test]
fn segment_cache_config_ext_presets_ordered_budgets() {
    let budgets = [
        SegmentCacheConfig::minimal().byte_budget.unwrap(),
        SegmentCacheConfig::interactive().byte_budget.unwrap(),
        SegmentCacheConfig::chorus().byte_budget.unwrap(),
        SegmentCacheConfig::batch().byte_budget.unwrap(),
    ];
    for pair in budgets.windows(2) {
        assert!(
            pair[0] < pair[1],
            "budgets must be strictly increasing: {} < {}",
            pair[0],
            pair[1]
        );
    }
}

/// Presets have strictly increasing max_segments_per_step.
#[test]
fn segment_cache_config_ext_presets_ordered_entry_counts() {
    let counts = [
        SegmentCacheConfig::minimal().max_segments_per_step,
        SegmentCacheConfig::interactive().max_segments_per_step,
        SegmentCacheConfig::chorus().max_segments_per_step,
        SegmentCacheConfig::batch().max_segments_per_step,
    ];
    for pair in counts.windows(2) {
        assert!(
            pair[0] < pair[1],
            "entry counts must be strictly increasing: {} < {}",
            pair[0],
            pair[1]
        );
    }
}

/// All presets use LRU eviction.
#[test]
fn segment_cache_config_ext_presets_all_lru() {
    let configs = [
        SegmentCacheConfig::minimal(),
        SegmentCacheConfig::interactive(),
        SegmentCacheConfig::chorus(),
        SegmentCacheConfig::batch(),
    ];
    for config in &configs {
        assert_eq!(config.eviction, EvictionPolicy::Lru);
    }
}

/// Default config has no byte_budget (None).
#[test]
fn segment_cache_config_ext_default_no_byte_budget() {
    let config = SegmentCacheConfig::default();
    assert_eq!(config.byte_budget, None);
    assert_eq!(config.max_segments_per_step, 4);
}

// ═══════════════════════════════════════════════════════════════════════
// 6. SegmentCacheStats
// ═══════════════════════════════════════════════════════════════════════

/// Default stats are all zero.
#[test]
fn segment_cache_stats_ext_default_zero() {
    let stats = SegmentCacheStats::default();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.total_bytes, 0);
    assert_eq!(stats.lookups(), 0);
    assert_eq!(stats.hit_rate(), 0.0);
}

/// Hit rate computation with non-zero counts.
#[test]
fn segment_cache_stats_ext_hit_rate() {
    let stats = SegmentCacheStats {
        hits: 7,
        misses: 3,
        evictions: 1,
        total_bytes: 0,
    };
    let rate = stats.hit_rate();
    assert!((rate - 0.7).abs() < 1e-9, "hit rate should be 0.7, got {rate}");
    assert_eq!(stats.lookups(), 10);
}

/// ShapeKeyedCache accumulates stats correctly.
#[test]
fn shape_keyed_cache_ext_stats_tracking() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(2);

    // Miss (insert).
    assert_eq!(cache.get(&[1, 10]), None);
    cache.insert(vec![1, 10], 100);

    // Hit.
    assert_eq!(cache.get(&[1, 10]), Some(&100));

    // Miss (different shape).
    assert_eq!(cache.get(&[1, 20]), None);
    cache.insert(vec![1, 20], 200);

    // Eviction: capacity is 2, insert a third.
    cache.insert(vec![1, 30], 300);

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.evictions, 1, "one LRU eviction should have occurred");
}

/// reset_stats clears counters but preserves cache contents.
#[test]
fn shape_keyed_cache_ext_reset_stats() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
    cache.insert(vec![1], 10);
    cache.get(&[1]); // hit
    cache.get(&[2]); // miss

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);

    cache.reset_stats();
    let stats = cache.stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.evictions, 0);

    // Cache contents are still there.
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&[1]), Some(&10));
}

// ═══════════════════════════════════════════════════════════════════════
// 7. F16AutocastConfig
// ═══════════════════════════════════════════════════════════════════════

/// recommended() enables 6 segments and disables 2.
#[test]
fn f16_autocast_ext_recommended_segment_counts() {
    let config = F16AutocastConfig::recommended(MixedPrecisionPolicy::apple_silicon_default());
    assert_eq!(config.enabled_count(), 6);
    assert!(config.plbert);
    assert!(config.text);
    assert!(config.prosody);
    assert!(config.f0);
    assert!(config.generator);
    assert!(config.sinegen_post);
    assert!(!config.regulate);
    assert!(!config.sinegen_pre);
}

/// all() enables 8 segments.
#[test]
fn f16_autocast_ext_all_enables_all() {
    let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default());
    assert_eq!(config.enabled_count(), 8);
    assert!(config.any_enabled());
}

/// none() disables all segments.
#[test]
fn f16_autocast_ext_none_disables_all() {
    let config = F16AutocastConfig::none(MixedPrecisionPolicy::apple_silicon_default());
    assert_eq!(config.enabled_count(), 0);
    assert!(!config.any_enabled());
}

/// generator_only() enables exactly the generator.
#[test]
fn f16_autocast_ext_generator_only() {
    let config =
        F16AutocastConfig::generator_only(MixedPrecisionPolicy::apple_silicon_default());
    assert_eq!(config.enabled_count(), 1);
    assert!(config.generator);
    assert!(!config.plbert);
    assert!(!config.text);
}

/// policy_for_segment returns the base policy for enabled segments.
#[test]
fn f16_autocast_ext_policy_for_segment_enabled() {
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let config = F16AutocastConfig::recommended(policy.clone());
    let returned = config.policy_for_segment("generator").unwrap();
    assert_eq!(returned.compute_dtype, policy.compute_dtype);
}

/// policy_for_segment returns None for disabled segments.
#[test]
fn f16_autocast_ext_policy_for_segment_disabled() {
    let config = F16AutocastConfig::recommended(MixedPrecisionPolicy::apple_silicon_default());
    assert!(config.policy_for_segment("regulate").is_none());
    assert!(config.policy_for_segment("sinegen_pre").is_none());
}

/// policy_for_segment returns None for unknown segment names.
#[test]
fn f16_autocast_ext_policy_for_unknown_segment() {
    let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default());
    assert!(config.policy_for_segment("nonexistent").is_none());
}

/// Builder pattern toggles work correctly.
#[test]
fn f16_autocast_ext_builder_toggle() {
    let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default())
        .with_plbert(false)
        .with_regulate(false);
    assert_eq!(config.enabled_count(), 6);
    assert!(!config.plbert);
    assert!(!config.regulate);
    assert!(config.text);
    assert!(config.generator);
}

/// Building from none() and selectively enabling matches expected count.
#[test]
fn f16_autocast_ext_selective_enable_from_none() {
    let config = F16AutocastConfig::none(MixedPrecisionPolicy::apple_silicon_default())
        .with_generator(true)
        .with_prosody(true)
        .with_f0(true);
    assert_eq!(config.enabled_count(), 3);
}

// ═══════════════════════════════════════════════════════════════════════
// 8. PrecompileShapes presets
// ═══════════════════════════════════════════════════════════════════════

/// short() preset shape values.
#[test]
fn precompile_shapes_ext_short_preset() {
    let shapes = PrecompileShapes::short();
    assert_eq!(shapes.seq_lens, vec![10, 20, 40, 80]);
    assert_eq!(shapes.t_mels, vec![20, 40, 80, 160]);
}

/// long_form() preset shape values.
#[test]
fn precompile_shapes_ext_long_form_preset() {
    let shapes = PrecompileShapes::long_form();
    assert_eq!(shapes.seq_lens, vec![40, 80, 160, 256, 512]);
    assert_eq!(shapes.t_mels, vec![80, 160, 320, 640, 1024]);
}

/// chorus() preset shape values.
#[test]
fn precompile_shapes_ext_chorus_preset() {
    let shapes = PrecompileShapes::chorus();
    assert_eq!(shapes.seq_lens, vec![20, 40, 80, 128]);
    assert_eq!(shapes.t_mels, vec![40, 80, 160, 320]);
}

/// Default preset matches documented values.
#[test]
fn precompile_shapes_ext_default() {
    let shapes = PrecompileShapes::default();
    assert_eq!(shapes.seq_lens, vec![10, 20, 40, 80]);
    assert_eq!(shapes.t_mels, vec![20, 40, 80, 160, 320]);
}

/// t_frames() computes 2x t_mels.
#[test]
fn precompile_shapes_ext_t_frames_double() {
    let shapes = PrecompileShapes::short();
    let t_frames = shapes.t_frames();
    for (i, &t_mel) in shapes.t_mels.iter().enumerate() {
        assert_eq!(t_frames[i], 2 * t_mel, "t_frames[{i}] should be 2 * t_mels[{i}]");
    }
}

/// with_seq_lens overrides sequence lengths.
#[test]
fn precompile_shapes_ext_with_seq_lens() {
    let shapes = PrecompileShapes::short().with_seq_lens(vec![5, 15]);
    assert_eq!(shapes.seq_lens, vec![5, 15]);
    // t_mels are unchanged.
    assert_eq!(shapes.t_mels, vec![20, 40, 80, 160]);
}

/// with_t_mels overrides mel frame counts.
#[test]
fn precompile_shapes_ext_with_t_mels() {
    let shapes = PrecompileShapes::short().with_t_mels(vec![100, 200]);
    assert_eq!(shapes.t_mels, vec![100, 200]);
    // seq_lens unchanged.
    assert_eq!(shapes.seq_lens, vec![10, 20, 40, 80]);
}

/// from_token_lengths returns None for empty input.
#[test]
fn precompile_shapes_ext_from_empty_tokens() {
    let result = PrecompileShapes::from_token_lengths(&[]);
    assert!(result.is_none());
}

/// from_token_lengths produces deduplicated, sorted seq_lens.
#[test]
fn precompile_shapes_ext_from_token_lengths_dedup_sorted() {
    let shapes = PrecompileShapes::from_token_lengths(&[40, 20, 40, 10, 20])
        .expect("non-empty input should produce shapes");
    // seq_lens should be deduplicated and sorted.
    let mut prev = 0;
    for &len in &shapes.seq_lens {
        assert!(len > prev, "seq_lens must be strictly increasing");
        prev = len;
    }
    // All original values should appear.
    assert!(shapes.seq_lens.contains(&10));
    assert!(shapes.seq_lens.contains(&20));
    assert!(shapes.seq_lens.contains(&40));
}

/// long_form() has larger shapes than short().
#[test]
fn precompile_shapes_ext_long_form_larger_than_short() {
    let short = PrecompileShapes::short();
    let long = PrecompileShapes::long_form();
    assert!(
        *long.seq_lens.last().unwrap() > *short.seq_lens.last().unwrap(),
        "long_form max seq_len should exceed short max seq_len"
    );
    assert!(
        *long.t_mels.last().unwrap() > *short.t_mels.last().unwrap(),
        "long_form max t_mel should exceed short max t_mel"
    );
}
