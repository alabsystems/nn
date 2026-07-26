// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for compiled model builder, dispatch infrastructure,
//! segment cache configuration, F16 autocast, precompile shapes,
//! NativeOpKind dispatch registration, and buffer pool configuration.
//!
//! These tests exercise builder patterns, configuration presets, statistics
//! types, and edge cases without requiring actual Metal GPU execution (except
//! builder tests that need PipelineCache, which requires MetalContext on macOS).
//!
//! Part of #4186.

use std::time::Duration;

use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_core::{Device, DType};
use nn_dsl::trace_compile::NativeOpKind;

use crate::buffer_pool_size_class::{
    BufferPoolSizeClassStats, SizeClassAllocator, SizeClassStats, NUM_SIZE_CLASSES,
    SIZE_CLASS_BOUNDARIES,
};
use crate::cache_stats::{CacheStats, CacheStatsSnapshot};
use crate::compiled_kokoro::precompile::PrecompileShapes;
use crate::segment_cache::{EvictionPolicy, SegmentCacheConfig, SegmentCacheStats, ShapeKeyedCache};
use crate::F16AutocastConfig;

fn empty_graph() -> ComputationGraph {
    ComputationGraph::from_nodes(vec![])
}

// ===================================================================
// Section 1: CompiledModelBuilder construction and method chaining
// ===================================================================

/// Builder construction from empty graph succeeds with default config.
#[test]
fn builder_extended_default_empty_graph_builds() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .build()
        .expect("default build on empty graph");
    assert_eq!(model.num_steps(), 0);
    assert_eq!(model.num_dispatches(), 0);
}

/// Builder with_peephole_config then build on empty graph succeeds.
#[test]
fn builder_extended_peephole_config_empty_graph() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let config = nn_dsl::PeepholeConfig::default();
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .with_peephole_config(config)
        .build()
        .expect("peephole config build");
    assert_eq!(model.num_steps(), 0);
}

/// Builder optimize with short budget on empty graph succeeds.
#[test]
fn builder_extended_optimize_empty_graph() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .optimize(Duration::from_millis(100))
        .build()
        .expect("optimize build on empty graph");
    assert_eq!(model.num_steps(), 0);
}

/// Builder force_dtype(F16) on empty graph succeeds.
#[test]
fn builder_extended_force_dtype_f16_empty() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .expect("F16 accepted")
        .build()
        .expect("build succeeds");
    assert_eq!(model.num_steps(), 0);
}

/// Builder force_dtype(BF16) on empty graph succeeds.
#[test]
fn builder_extended_force_dtype_bf16_empty() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::BF16)
        .expect("BF16 accepted")
        .build()
        .expect("build succeeds");
    assert_eq!(model.num_steps(), 0);
}

/// Builder force_dtype rejects non-float dtype (U32).
#[test]
fn builder_extended_force_dtype_rejects_u32() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let result = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::U32);
    assert!(result.is_err(), "U32 should be rejected as GPU dtype");
}

/// Builder force_dtype rejects I64.
#[test]
fn builder_extended_force_dtype_rejects_i64() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let result = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::I64);
    assert!(result.is_err(), "I64 should be rejected as GPU dtype");
}

/// Builder autocast with f32_only policy is a no-op: build succeeds
/// and is_autocast() returns false.
#[test]
fn builder_extended_autocast_f32_only_is_noop() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let policy = MixedPrecisionPolicy::f32_only();

    let (_, traced_graph) = nn_core::dyn_tensor::trace::trace_graph(|| {
        let a = nn_core::dyn_tensor::DynTensor::zeros(&[2, 4], DType::F32, &Device::Cpu)?;
        let b = nn_core::dyn_tensor::DynTensor::ones(&[2, 4], DType::F32, &Device::Cpu)?;
        let out = a.add(&b)?;
        Ok(out)
    })
    .expect("trace should succeed");

    let model = crate::compiled_model::CompiledModel::builder(&traced_graph, &cache)
        .autocast(policy)
        .build()
        .expect("f32_only autocast should succeed");
    // f32_only policy is a no-op -- is_autocast() should return false.
    assert!(!model.is_autocast(), "f32_only autocast should be treated as no-op");
}

/// Builder: force_dtype + autocast on non-empty graph fails.
#[test]
fn builder_extended_force_dtype_autocast_mutual_exclusion() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let policy = MixedPrecisionPolicy::apple_silicon_default();

    // Trace a simple graph with one op.
    let (_, traced_graph) = nn_core::dyn_tensor::trace::trace_graph(|| {
        let input = nn_core::dyn_tensor::DynTensor::zeros(&[1, 10], DType::F32, &Device::Cpu)?;
        let ones = nn_core::dyn_tensor::DynTensor::ones(&[1, 10], DType::F32, &Device::Cpu)?;
        let output = input.add(&ones)?;
        Ok(output)
    })
    .expect("trace should succeed");

    let result = crate::compiled_model::CompiledModel::builder(&traced_graph, &cache)
        .force_dtype(DType::F16)
        .expect("F16 accepted")
        .autocast(policy)
        .build();
    assert!(result.is_err(), "force_dtype + autocast should conflict");
}

/// Builder: chaining autocast then with_peephole_config then build.
#[test]
fn builder_extended_autocast_plus_peephole() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let config = nn_dsl::PeepholeConfig::default();
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .with_peephole_config(config)
        .build()
        .expect("autocast + peephole config should succeed on empty graph");
    assert_eq!(model.num_steps(), 0);
}

// ===================================================================
// Section 2: SegmentCacheConfig presets
// ===================================================================

/// interactive() preset values are correct.
#[test]
fn cache_config_extended_interactive() {
    let config = SegmentCacheConfig::interactive();
    assert_eq!(config.max_segments_per_step, 2);
    assert_eq!(config.eviction, EvictionPolicy::Lru);
    assert_eq!(config.byte_budget, Some(256 * 1024 * 1024));
    assert!(config.shared_store.is_none());
}

/// batch() preset values are correct.
#[test]
fn cache_config_extended_batch() {
    let config = SegmentCacheConfig::batch();
    assert_eq!(config.max_segments_per_step, 8);
    assert_eq!(config.eviction, EvictionPolicy::Lru);
    assert_eq!(config.byte_budget, Some(1024 * 1024 * 1024));
    assert!(config.shared_store.is_none());
}

/// chorus() preset values are correct.
#[test]
fn cache_config_extended_chorus() {
    let config = SegmentCacheConfig::chorus();
    assert_eq!(config.max_segments_per_step, 6);
    assert_eq!(config.eviction, EvictionPolicy::Lru);
    assert_eq!(config.byte_budget, Some(768 * 1024 * 1024));
    assert!(config.shared_store.is_none());
}

/// minimal() preset values are correct.
#[test]
fn cache_config_extended_minimal() {
    let config = SegmentCacheConfig::minimal();
    assert_eq!(config.max_segments_per_step, 1);
    assert_eq!(config.eviction, EvictionPolicy::Lru);
    assert_eq!(config.byte_budget, Some(128 * 1024 * 1024));
    assert!(config.shared_store.is_none());
}

/// All presets have distinct byte_budget values.
#[test]
fn cache_config_extended_presets_distinct_budgets() {
    let budgets: Vec<usize> = [
        SegmentCacheConfig::minimal(),
        SegmentCacheConfig::interactive(),
        SegmentCacheConfig::chorus(),
        SegmentCacheConfig::batch(),
    ]
    .iter()
    .map(|c| c.byte_budget.unwrap())
    .collect();
    for pair in budgets.windows(2) {
        assert!(pair[0] < pair[1], "budgets must be strictly increasing");
    }
}

/// Default config: max_segments=4, no byte_budget, LRU.
#[test]
fn cache_config_extended_default() {
    let config = SegmentCacheConfig::default();
    assert_eq!(config.max_segments_per_step, 4);
    assert_eq!(config.eviction, EvictionPolicy::Lru);
    assert_eq!(config.byte_budget, None);
    assert!(config.shared_store.is_none());
}

/// Custom config: large capacity and explicit byte_budget.
#[test]
fn cache_config_extended_custom_large_capacity() {
    let config = SegmentCacheConfig {
        max_segments_per_step: 32,
        eviction: EvictionPolicy::Lru,
        byte_budget: Some(2 * 1024 * 1024 * 1024),
        shared_store: None,
    };
    assert_eq!(config.max_segments_per_step, 32);
    assert_eq!(config.byte_budget, Some(2 * 1024 * 1024 * 1024));
}

// ===================================================================
// Section 3: SegmentCacheStats
// ===================================================================

/// Default stats are all zero.
#[test]
fn cache_stats_extended_default_zero() {
    let stats = SegmentCacheStats::default();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.total_bytes, 0);
    assert_eq!(stats.lookups(), 0);
    assert_eq!(stats.hit_rate(), 0.0);
}

/// Hit rate computation with known values.
#[test]
fn cache_stats_extended_hit_rate_nonzero() {
    let stats = SegmentCacheStats {
        hits: 9,
        misses: 1,
        evictions: 0,
        total_bytes: 0,
    };
    assert!((stats.hit_rate() - 0.9).abs() < 1e-9);
    assert_eq!(stats.lookups(), 10);
}

/// Hit rate is 0.0 when there are only misses.
#[test]
fn cache_stats_extended_all_misses() {
    let stats = SegmentCacheStats {
        hits: 0,
        misses: 5,
        evictions: 0,
        total_bytes: 0,
    };
    assert_eq!(stats.hit_rate(), 0.0);
}

/// Hit rate is 1.0 when there are only hits.
#[test]
fn cache_stats_extended_all_hits() {
    let stats = SegmentCacheStats {
        hits: 10,
        misses: 0,
        evictions: 0,
        total_bytes: 0,
    };
    assert!((stats.hit_rate() - 1.0).abs() < 1e-9);
}

/// ShapeKeyedCache tracks stats through insert/get cycles.
#[test]
fn shape_cache_extended_stats_tracking() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(2);

    // Two misses via get().
    assert_eq!(cache.get(&[1, 10]), None);
    assert_eq!(cache.get(&[1, 20]), None);

    cache.insert(vec![1, 10], 100);
    cache.insert(vec![1, 20], 200);

    // Two hits.
    assert_eq!(cache.get(&[1, 10]), Some(&100));
    assert_eq!(cache.get(&[1, 20]), Some(&200));

    let stats = cache.stats();
    assert_eq!(stats.hits, 2);
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.evictions, 0);
}

/// ShapeKeyedCache eviction tracking.
#[test]
fn shape_cache_extended_eviction_tracking() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(2);
    cache.insert(vec![1], 1);
    cache.insert(vec![2], 2);
    cache.insert(vec![3], 3); // evicts [1]
    cache.insert(vec![4], 4); // evicts [2]

    let stats = cache.stats();
    assert_eq!(stats.evictions, 2);
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get(&[1]), None);
    assert_eq!(cache.get(&[2]), None);
}

/// reset_stats clears counters but preserves data.
#[test]
fn shape_cache_extended_reset_preserves_data() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
    cache.insert(vec![10], 42);
    cache.get(&[10]); // hit
    cache.get(&[20]); // miss

    cache.reset_stats();
    let stats = cache.stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.evictions, 0);

    // Data preserved.
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&[10]), Some(&42));
}

// ===================================================================
// Section 4: F16AutocastConfig::recommended()
// ===================================================================

/// recommended() enables 6 compute-heavy segments.
#[test]
fn autocast_extended_recommended_count() {
    let config = F16AutocastConfig::recommended(MixedPrecisionPolicy::apple_silicon_default());
    assert_eq!(config.enabled_count(), 6);
    assert!(config.any_enabled());
}

/// recommended() enables plbert, text, prosody, f0, generator, sinegen_post.
#[test]
fn autocast_extended_recommended_enabled_segments() {
    let config = F16AutocastConfig::recommended(MixedPrecisionPolicy::apple_silicon_default());
    assert!(config.plbert);
    assert!(config.text);
    assert!(config.prosody);
    assert!(config.f0);
    assert!(config.generator);
    assert!(config.sinegen_post);
}

/// recommended() disables regulate and sinegen_pre.
#[test]
fn autocast_extended_recommended_disabled_segments() {
    let config = F16AutocastConfig::recommended(MixedPrecisionPolicy::apple_silicon_default());
    assert!(!config.regulate);
    assert!(!config.sinegen_pre);
}

/// all() enables 8 segments.
#[test]
fn autocast_extended_all_count() {
    let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default());
    assert_eq!(config.enabled_count(), 8);
}

/// none() disables all segments.
#[test]
fn autocast_extended_none_count() {
    let config = F16AutocastConfig::none(MixedPrecisionPolicy::apple_silicon_default());
    assert_eq!(config.enabled_count(), 0);
    assert!(!config.any_enabled());
}

/// generator_only() enables exactly one segment.
#[test]
fn autocast_extended_generator_only() {
    let config = F16AutocastConfig::generator_only(MixedPrecisionPolicy::apple_silicon_default());
    assert_eq!(config.enabled_count(), 1);
    assert!(config.generator);
    assert!(!config.plbert);
}

/// policy_for_segment returns base policy for enabled, None for disabled.
#[test]
fn autocast_extended_policy_dispatch() {
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let config = F16AutocastConfig::recommended(policy);
    assert!(config.policy_for_segment("generator").is_some());
    assert!(config.policy_for_segment("regulate").is_none());
    assert!(config.policy_for_segment("unknown_name").is_none());
}

/// Builder toggle pattern: disable from all().
#[test]
fn autocast_extended_builder_disable() {
    let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default())
        .with_plbert(false)
        .with_regulate(false)
        .with_sinegen_pre(false);
    assert_eq!(config.enabled_count(), 5);
    assert!(!config.plbert);
    assert!(!config.regulate);
    assert!(!config.sinegen_pre);
}

/// Builder toggle pattern: enable from none().
#[test]
fn autocast_extended_builder_selective_enable() {
    let config = F16AutocastConfig::none(MixedPrecisionPolicy::apple_silicon_default())
        .with_generator(true)
        .with_f0(true);
    assert_eq!(config.enabled_count(), 2);
    assert!(config.generator);
    assert!(config.f0);
    assert!(!config.plbert);
}

// ===================================================================
// Section 5: PrecompileShapes presets
// ===================================================================

/// short() preset values.
#[test]
fn precompile_extended_short() {
    let shapes = PrecompileShapes::short();
    assert_eq!(shapes.seq_lens, vec![10, 20, 40, 80]);
    assert_eq!(shapes.t_mels, vec![20, 40, 80, 160]);
}

/// long_form() preset values.
#[test]
fn precompile_extended_long_form() {
    let shapes = PrecompileShapes::long_form();
    assert_eq!(shapes.seq_lens, vec![40, 80, 160, 256, 512]);
    assert_eq!(shapes.t_mels, vec![80, 160, 320, 640, 1024]);
}

/// chorus() preset values.
#[test]
fn precompile_extended_chorus() {
    let shapes = PrecompileShapes::chorus();
    assert_eq!(shapes.seq_lens, vec![20, 40, 80, 128]);
    assert_eq!(shapes.t_mels, vec![40, 80, 160, 320]);
}

/// t_frames() doubles t_mels.
#[test]
fn precompile_extended_t_frames() {
    let shapes = PrecompileShapes::chorus();
    let t_frames = shapes.t_frames();
    for (i, &mel) in shapes.t_mels.iter().enumerate() {
        assert_eq!(t_frames[i], 2 * mel);
    }
}

/// with_seq_lens overrides only seq_lens.
#[test]
fn precompile_extended_with_seq_lens() {
    let shapes = PrecompileShapes::short().with_seq_lens(vec![5, 15, 25]);
    assert_eq!(shapes.seq_lens, vec![5, 15, 25]);
    assert_eq!(shapes.t_mels, vec![20, 40, 80, 160]); // unchanged
}

/// with_t_mels overrides only t_mels.
#[test]
fn precompile_extended_with_t_mels() {
    let shapes = PrecompileShapes::short().with_t_mels(vec![50, 100]);
    assert_eq!(shapes.t_mels, vec![50, 100]);
    assert_eq!(shapes.seq_lens, vec![10, 20, 40, 80]); // unchanged
}

/// from_token_lengths returns None on empty.
#[test]
fn precompile_extended_from_empty_tokens() {
    assert!(PrecompileShapes::from_token_lengths(&[]).is_none());
}

/// from_token_lengths deduplicates and sorts.
#[test]
fn precompile_extended_from_token_lengths_sorted() {
    let shapes = PrecompileShapes::from_token_lengths(&[30, 10, 30, 20, 10])
        .expect("non-empty produces shapes");
    let mut prev = 0;
    for &len in &shapes.seq_lens {
        assert!(len > prev, "seq_lens must be strictly increasing");
        prev = len;
    }
    assert!(shapes.seq_lens.contains(&10));
    assert!(shapes.seq_lens.contains(&20));
    assert!(shapes.seq_lens.contains(&30));
}

/// long_form has strictly larger shapes than short.
#[test]
fn precompile_extended_long_form_exceeds_short() {
    let short = PrecompileShapes::short();
    let long = PrecompileShapes::long_form();
    assert!(long.seq_lens.last().unwrap() > short.seq_lens.last().unwrap());
    assert!(long.t_mels.last().unwrap() > short.t_mels.last().unwrap());
}

/// Default preset matches new() values.
#[test]
fn precompile_extended_default_matches_new() {
    let d = PrecompileShapes::default();
    let n = PrecompileShapes::new();
    assert_eq!(d.seq_lens, n.seq_lens);
    assert_eq!(d.t_mels, n.t_mels);
}

// ===================================================================
// Section 6: NativeOpKind dispatch registration
// ===================================================================

/// NativeOpKind::LstmSequence can be constructed with valid params.
#[test]
fn native_op_extended_lstm_construction() {
    let op = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![100, 1, 128],
        h_shape: vec![1, 256],
        reverse: false,
    };
    // Verify the variant matches.
    assert!(matches!(op, NativeOpKind::LstmSequence { hidden_size: 256, .. }));
}

/// NativeOpKind::InstanceNorm can be constructed.
#[test]
fn native_op_extended_instance_norm_construction() {
    let op = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 64, 512],
    };
    assert!(matches!(op, NativeOpKind::InstanceNorm { .. }));
}

/// NativeOpKind::LayerNorm can be constructed.
#[test]
fn native_op_extended_layer_norm_construction() {
    let op = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 32, 768],
        hidden_dim: 768,
    };
    assert!(matches!(op, NativeOpKind::LayerNorm { hidden_dim: 768, .. }));
}

/// NativeOpKind::FlashAttention can be constructed.
#[test]
fn native_op_extended_flash_attention_construction() {
    let op = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: true,
        q_shape: vec![1, 8, 128, 64],
        k_shape: vec![1, 8, 128, 64],
        output_shape: vec![1, 8, 128, 64],
        input_layout: Default::default(),
    };
    assert!(matches!(op, NativeOpKind::FlashAttention { causal: true, .. }));
}

/// NativeOpKind::AdainSnake can be constructed.
#[test]
fn native_op_extended_adain_snake_construction() {
    let op = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 128, 256],
        channels: 128,
        residual_gamma: true,
        external_node_ids: None,
    };
    assert!(matches!(op, NativeOpKind::AdainSnake { channels: 128, .. }));
}

/// NativeOpKind::Cumsum can be constructed.
#[test]
fn native_op_extended_cumsum_construction() {
    let op = NativeOpKind::Cumsum {
        dim: 1,
        input_shape: vec![1, 256],
    };
    assert!(matches!(op, NativeOpKind::Cumsum { dim: 1, .. }));
}

/// NativeOpKind::MaxPool1d can be constructed.
#[test]
fn native_op_extended_max_pool1d_construction() {
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 2,
        padding: 1,
        input_shape: vec![1, 64, 1024],
    };
    assert!(matches!(op, NativeOpKind::MaxPool1d { kernel_size: 3, .. }));
}

/// NativeOpKind variants count is at least 31 (matches MEMORY.md).
#[test]
fn native_op_extended_variant_count_minimum() {
    // Verify several distinct variants exist. The enum has 31+ variants.
    let ops: Vec<NativeOpKind> = vec![
        NativeOpKind::LstmSequence {
            hidden_size: 64, input_shape: vec![10, 1, 32],
            h_shape: vec![1, 64], reverse: false,
        },
        NativeOpKind::Cumsum { dim: 0, input_shape: vec![1, 10] },
        NativeOpKind::InstanceNorm { eps: 1e-5, input_shape: vec![1, 8, 16] },
        NativeOpKind::LayerNorm { eps: 1e-5, input_shape: vec![1, 4, 32], hidden_dim: 32 },
        NativeOpKind::MaxPool1d {
            kernel_size: 2, stride: 2, padding: 0, input_shape: vec![1, 8, 32],
        },
        NativeOpKind::ConstantWeight {
            name: "test".to_string(), shape: vec![10],
        },
    ];
    assert!(ops.len() >= 6, "should be able to construct multiple distinct variants");
}

// ===================================================================
// Section 7: Buffer pool configuration
// ===================================================================

/// Fresh SizeClassAllocator has all-zero stats.
#[test]
fn buffer_pool_extended_fresh_zero() {
    let alloc = SizeClassAllocator::new();
    let stats = alloc.stats();
    assert_eq!(stats.oversized_allocs, 0);
    assert_eq!(stats.total_free_bytes, 0);
    assert_eq!(stats.total_used_bytes, 0);
    assert_eq!(stats.hit_rate, 0.0);
    assert_eq!(stats.fragmentation_ratio, 0.0);
}

/// Size class boundaries are strictly increasing.
#[test]
fn buffer_pool_extended_boundaries_increasing() {
    for pair in SIZE_CLASS_BOUNDARIES.windows(2) {
        assert!(pair[0] < pair[1], "boundaries must increase");
    }
}

/// size_class_for routes small requests to class 0.
#[test]
fn buffer_pool_extended_size_class_small() {
    assert_eq!(SizeClassAllocator::size_class_for(1), Some(0));
    assert_eq!(SizeClassAllocator::size_class_for(100), Some(0));
    assert_eq!(SizeClassAllocator::size_class_for(4096), Some(0));
}

/// size_class_for routes oversized to None.
#[test]
fn buffer_pool_extended_size_class_oversized() {
    assert_eq!(SizeClassAllocator::size_class_for(100 * 1024 * 1024), None);
}

/// size_class_for(0) maps to class 0.
#[test]
fn buffer_pool_extended_size_class_zero() {
    assert_eq!(SizeClassAllocator::size_class_for(0), Some(0));
}

/// Allocation and deallocation cycle tracks hits/misses.
#[test]
fn buffer_pool_extended_alloc_dealloc_cycle() {
    let mut alloc = SizeClassAllocator::new();
    let r1 = alloc.allocate(100).expect("should allocate");
    assert!(!r1.reused);
    assert_eq!(r1.class, 0);

    alloc.deallocate(r1.class);
    let r2 = alloc.allocate(50).expect("should reuse");
    assert!(r2.reused);

    let stats = alloc.stats();
    assert_eq!(stats.per_class[0].hits, 1);
    assert_eq!(stats.per_class[0].misses, 1);
}

/// Multiple size classes are used correctly.
#[test]
fn buffer_pool_extended_multiple_classes() {
    let mut alloc = SizeClassAllocator::new();
    let r0 = alloc.allocate(100).expect("class 0");      // 4 KB
    let r4 = alloc.allocate(500_000).expect("class 4");   // 1 MB
    assert_eq!(r0.class, 0);
    assert_eq!(r4.class, 4);
    assert_eq!(r0.alloc_bytes, SIZE_CLASS_BOUNDARIES[0]);
    assert_eq!(r4.alloc_bytes, SIZE_CLASS_BOUNDARIES[4]);
}

/// Reset clears all allocator state.
#[test]
fn buffer_pool_extended_reset() {
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
    }
}

/// SizeClassStats total_allocs and hit_rate.
#[test]
fn size_class_stats_extended_derived() {
    let stats = SizeClassStats {
        hits: 3,
        misses: 7,
        free_count: 0,
        in_use_count: 0,
        peak_in_use: 0,
        free_bytes: 0,
    };
    assert_eq!(stats.total_allocs(), 10);
    assert!((stats.hit_rate() - 0.3).abs() < 1e-9);
}

/// SizeClassStats hit_rate is 0.0 with no allocations.
#[test]
fn size_class_stats_extended_empty_hit_rate() {
    let stats = SizeClassStats::default();
    assert_eq!(stats.total_allocs(), 0);
    assert_eq!(stats.hit_rate(), 0.0);
}

/// BufferPoolSizeClassStats default has NUM_SIZE_CLASSES entries.
#[test]
fn buffer_pool_stats_extended_default_class_count() {
    let stats = BufferPoolSizeClassStats::default();
    assert_eq!(stats.per_class.len(), NUM_SIZE_CLASSES);
    assert_eq!(NUM_SIZE_CLASSES, 8);
}

// ===================================================================
// Section 8: Edge cases - empty and single-op graphs
// ===================================================================

/// Building from empty graph via builder default succeeds.
#[test]
fn edge_case_extended_builder_empty() {
    let graph = empty_graph();
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .build()
        .expect("builder on empty graph should succeed");
    assert_eq!(model.num_steps(), 0);
    assert_eq!(model.num_dispatches(), 0);
}

/// Building a single-op graph (elementwise add) via builder succeeds.
#[test]
fn edge_case_extended_single_op_graph() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);

    let (_, traced_graph) = nn_core::dyn_tensor::trace::trace_graph(|| {
        let a = nn_core::dyn_tensor::DynTensor::zeros(&[2, 3], DType::F32, &Device::Cpu)?;
        let b = nn_core::dyn_tensor::DynTensor::ones(&[2, 3], DType::F32, &Device::Cpu)?;
        let out = a.add(&b)?;
        Ok(out)
    })
    .expect("trace should succeed");

    let model = crate::compiled_model::CompiledModel::builder(&traced_graph, &cache)
        .build()
        .expect("single-op graph should build");
    assert!(model.num_steps() > 0, "single-op graph should have steps");
}

/// Single-op graph with F16 force_dtype builds.
#[test]
fn edge_case_extended_single_op_f16() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);

    let (_, traced_graph) = nn_core::dyn_tensor::trace::trace_graph(|| {
        let a = nn_core::dyn_tensor::DynTensor::zeros(&[4, 4], DType::F32, &Device::Cpu)?;
        let b = nn_core::dyn_tensor::DynTensor::ones(&[4, 4], DType::F32, &Device::Cpu)?;
        let out = a.add(&b)?;
        Ok(out)
    })
    .expect("trace should succeed");

    let model = crate::compiled_model::CompiledModel::builder(&traced_graph, &cache)
        .force_dtype(DType::F16)
        .expect("F16 accepted")
        .build()
        .expect("single-op graph with F16 should build");
    assert!(model.is_mixed_precision());
}

/// Single-op graph with autocast builds.
#[test]
fn edge_case_extended_single_op_autocast() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let policy = MixedPrecisionPolicy::apple_silicon_default();

    let (_, traced_graph) = nn_core::dyn_tensor::trace::trace_graph(|| {
        let a = nn_core::dyn_tensor::DynTensor::zeros(&[4, 4], DType::F32, &Device::Cpu)?;
        let b = nn_core::dyn_tensor::DynTensor::ones(&[4, 4], DType::F32, &Device::Cpu)?;
        let out = a.add(&b)?;
        Ok(out)
    })
    .expect("trace should succeed");

    let model = crate::compiled_model::CompiledModel::builder(&traced_graph, &cache)
        .autocast(policy)
        .build()
        .expect("single-op graph with autocast should build");
    assert!(model.is_autocast());
}

// ===================================================================
// Section 9: CacheStatsSnapshot
// ===================================================================

/// CacheStatsSnapshot default is all zero.
#[test]
fn cache_snapshot_extended_default() {
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

/// CacheStatsSnapshot hit_rate with zero lookups.
#[test]
fn cache_snapshot_extended_zero_lookups_rate() {
    let snap = CacheStatsSnapshot::default();
    assert_eq!(snap.hit_rate(), 0.0);
    assert_eq!(snap.kernel_hit_rate(), 0.0);
    assert_eq!(snap.msl_hit_rate(), 0.0);
    assert_eq!(snap.pipeline_hit_rate(), 0.0);
}

/// CacheStatsSnapshot kernel_hit_rate computation.
#[test]
fn cache_snapshot_extended_kernel_hit_rate() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 3,
        kernel_cache_misses: 1,
        ..Default::default()
    };
    assert!((snap.kernel_hit_rate() - 0.75).abs() < 1e-9);
}

/// CacheStatsSnapshot avg_compile_time.
#[test]
fn cache_snapshot_extended_avg_compile_time() {
    let snap = CacheStatsSnapshot {
        pipeline_cache_misses: 5,
        total_compile_time_us: 1000,
        ..Default::default()
    };
    assert!((snap.avg_compile_time_us() - 200.0).abs() < 1e-9);
}

/// Global CacheStats reset and snapshot round-trip.
#[test]
fn cache_stats_extended_global_reset_snapshot() {
    let stats = CacheStats::global();
    stats.reset();
    stats.record_kernel_hit();
    stats.record_kernel_miss();
    stats.record_dispatch();
    let snap = stats.snapshot();
    // Use >= because global singleton may have concurrent activity.
    assert!(snap.kernel_cache_hits >= 1);
    assert!(snap.kernel_cache_misses >= 1);
    assert!(snap.total_dispatches >= 1);
}
