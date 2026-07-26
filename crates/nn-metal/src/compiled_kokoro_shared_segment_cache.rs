// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pre-warmed shared segment cache for multi-instance [`CompiledKokoro`].
//!
//! [`SharedSegmentCache`] holds `Arc<CompiledModelDef>` entries for each of
//! the 8 Kokoro pipeline segments, keyed by shape dimension. Multiple
//! [`CompiledKokoro`] instances created from the same cache share compiled
//! Metal pipelines and GPU weight buffers via `Arc`, eliminating per-instance
//! recompilation overhead.
//!
//! # Architecture
//!
//! ```text
//! SharedSegmentCache (Arc-wrapped, thread-safe)
//!   ├── shared_state: Arc<SharedKokoroState>   (model weights, verifier, iSTFT)
//!   ├── segment_defs[8]: HashMap<usize, Arc<CompiledModelDef>>
//!   │   ├── plbert[seq_len]      → Arc<CompiledModelDef>
//!   │   ├── text[seq_len]        → Arc<CompiledModelDef>
//!   │   ├── prosody[seq_len]     → Arc<CompiledModelDef>
//!   │   ├── f0[t_mel]            → Arc<CompiledModelDef>
//!   │   ├── generator[samples]   → Arc<CompiledModelDef>
//!   │   ├── regulate[seq_len]    → Arc<CompiledModelDef>
//!   │   ├── sinegen_pre[frames]  → Arc<CompiledModelDef>
//!   │   └── sinegen_post[frames] → Arc<CompiledModelDef>
//!   └── weight_aliases[8]: HashMap<(usize, String), MetalBuffer>
//!
//! CompiledKokoro instance (lightweight handle)
//!   ├── shared: Arc<SharedKokoroState>   (cloned from cache)
//!   └── seg_*: SegmentCache
//!       └── entries: CompiledModel::from_shared(Arc<CompiledModelDef>)
//!           (own execution caches, shared immutable definition)
//! ```
//!
//! Memory: N instances share one set of compiled pipelines and GPU weight
//! buffers. Per-instance overhead is only execution caches (planned buffer,
//! ICBs) — typically <1 MB per instance vs ~400 MB for full weights.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_metal::compiled_kokoro::SharedSegmentCache;
//!
//! // 1. Create and warm up a primary instance
//! let mut primary = CompiledKokoro::new(model)?;
//! let _ = primary.synthesize(&input_ids, &style, 1.0, &cache)?;
//!
//! // 2. Snapshot compiled state into a shared cache
//! let shared_cache = SharedSegmentCache::from_compiled(&primary);
//!
//! // 3. Create lightweight instances from the cache
//! let mut voice1 = shared_cache.create_instance();
//! let mut voice2 = shared_cache.create_instance();
//! // voice1 and voice2 share pipelines + weights via Arc
//! ```
//!
//! Part of #4104.

use std::collections::HashMap;
use std::sync::Arc;

use crate::buffer::MetalBuffer;
use crate::compiled_model::{CompiledModel, CompiledModelDef};
use crate::segment_cache::SegmentCacheConfig;

use super::segment_cache::SegmentCache;
use super::shared::SharedKokoroState;
use super::CompiledKokoro;

/// Index constants for the 8 pipeline segments.
const SEG_PLBERT: usize = 0;
const SEG_TEXT: usize = 1;
const SEG_PROSODY: usize = 2;
const SEG_F0: usize = 3;
const SEG_GENERATOR: usize = 4;
const SEG_REGULATE: usize = 5;
const SEG_SINEGEN_PRE: usize = 6;
const SEG_SINEGEN_POST: usize = 7;
const NUM_SEGMENTS: usize = 8;

/// Per-segment entry: shape key → shared model definition + weight aliases.
struct SegmentStore {
    /// Compiled model definitions keyed by shape dimension.
    /// Each `Arc<CompiledModelDef>` is immutable after compilation and can be
    /// shared across instances via `CompiledModel::from_shared()`.
    defs: HashMap<usize, Arc<CompiledModelDef>>,
    /// Shared GPU weight buffer aliases for this segment.
    /// Populated from the first compiled model's `weight_buffer_aliases()`.
    weight_aliases: Option<HashMap<(usize, String), MetalBuffer>>,
}

impl SegmentStore {
    /// Populate from an existing `SegmentCache` by extracting `Arc<CompiledModelDef>`
    /// from each cached entry and aliasing weight buffers.
    fn populate_from_cache(cache: &SegmentCache) -> Self {
        let mut defs = HashMap::new();

        // Extract Arc<CompiledModelDef> from the most recent entry.
        // SegmentCache stores (key, CompiledModel) pairs in LRU order.
        // We snapshot whatever is currently cached.
        if let Some(&(key, ref model)) = cache.most_recent() {
            defs.insert(key, model.share_def());
        }

        let weight_aliases = cache.shared_weights().map(|w| {
            w.iter()
                .map(|(k, buf)| (k.clone(), buf.alias()))
                .collect()
        });

        Self {
            defs,
            weight_aliases,
        }
    }

    /// Number of cached shape variants.
    fn len(&self) -> usize {
        self.defs.len()
    }

    /// Total GPU weight bytes in the shared aliases.
    fn weight_bytes(&self) -> usize {
        self.weight_aliases
            .as_ref()
            .map(|w| w.values().map(MetalBuffer::len).sum())
            .unwrap_or(0)
    }
}

/// Pre-warmed shared segment cache for multi-instance Kokoro dispatch.
///
/// Holds `Arc`-wrapped compiled model definitions and GPU weight buffer
/// aliases for all 8 pipeline segments. Thread-safe: wrap in `Arc` and
/// clone across threads (all inner data is `Arc`-shared or aliased).
///
/// Created from a warmed-up [`CompiledKokoro`] via [`from_compiled()`].
/// Use [`create_instance()`] to create lightweight dispatch handles.
///
/// Part of #4104.
pub struct SharedSegmentCache {
    /// Shared model state (weights, verifier, iSTFT basis).
    shared_state: Arc<SharedKokoroState>,
    /// Per-segment compiled model definitions and weight aliases.
    segments: [SegmentStore; NUM_SEGMENTS],
    /// Segment cache configuration propagated to new instances.
    segment_cache_config: SegmentCacheConfig,
    /// Per-segment peephole configs propagated to new instances.
    peephole_configs: HashMap<String, nn_dsl::PeepholeConfig>,
    /// Mixed-precision flag propagated to new instances.
    mixed_precision: bool,
    /// Autocast policy propagated to new instances.
    autocast_policy: Option<nn_core::mixed_precision::MixedPrecisionPolicy>,
    /// Per-segment autocast configuration propagated to new instances.
    /// Part of #4269.
    segment_autocast: Option<super::F16AutocastConfig>,
}

impl SharedSegmentCache {
    /// Create a shared cache by snapshotting the compiled state of a warmed-up
    /// [`CompiledKokoro`] instance.
    ///
    /// For best results, call `synthesize()` on the primary instance first to
    /// populate segment caches, then call this method. Only segments that have
    /// been compiled (cache non-empty) will be shared.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut primary = CompiledKokoro::new(model)?;
    /// // Warm up: compile all segments for a representative input
    /// let _ = primary.synthesize(&input_ids, &style, 1.0, &cache)?;
    /// // Snapshot compiled state
    /// let shared = SharedSegmentCache::from_compiled(&primary);
    /// // Create N lightweight instances
    /// let voices: Vec<_> = (0..8).map(|_| shared.create_instance()).collect();
    /// ```
    #[must_use]
    pub fn from_compiled(primary: &CompiledKokoro) -> Self {
        Self {
            shared_state: Arc::clone(&primary.shared),
            segments: [
                SegmentStore::populate_from_cache(&primary.seg_plbert),
                SegmentStore::populate_from_cache(&primary.seg_text),
                SegmentStore::populate_from_cache(&primary.seg_prosody),
                SegmentStore::populate_from_cache(&primary.seg_f0),
                SegmentStore::populate_from_cache(&primary.seg_generator),
                SegmentStore::populate_from_cache(&primary.seg_regulate),
                SegmentStore::populate_from_cache(&primary.seg_sinegen_pre),
                SegmentStore::populate_from_cache(&primary.seg_sinegen_post),
            ],
            segment_cache_config: primary.segment_cache_config.clone(),
            peephole_configs: primary.peephole_configs.clone(),
            mixed_precision: primary.mixed_precision,
            autocast_policy: primary.autocast_policy.clone(),
            segment_autocast: primary.segment_autocast.clone(),
        }
    }

    /// Create a lightweight [`CompiledKokoro`] instance from this shared cache.
    ///
    /// The new instance shares:
    /// - Model weights, verifier, and iSTFT basis via `Arc<SharedKokoroState>`
    /// - Compiled pipeline definitions via `Arc<CompiledModelDef>`
    /// - GPU weight buffers via `MetalBuffer::alias()` (zero-copy ARC bump)
    ///
    /// Per-instance state (planned buffers, ICBs, embedding cache) is unique.
    ///
    /// Memory overhead per instance: ~1-2 MB (execution caches only).
    /// Without sharing: ~400 MB (full weights + pipelines per instance).
    #[must_use]
    pub fn create_instance(&self) -> CompiledKokoro {
        CompiledKokoro {
            shared: Arc::clone(&self.shared_state),
            seg_plbert: self.build_segment_cache(SEG_PLBERT),
            seg_text: self.build_segment_cache(SEG_TEXT),
            seg_prosody: self.build_segment_cache(SEG_PROSODY),
            seg_f0: self.build_segment_cache(SEG_F0),
            seg_generator: self.build_segment_cache(SEG_GENERATOR),
            seg_regulate: self.build_segment_cache(SEG_REGULATE),
            seg_sinegen_pre: self.build_segment_cache(SEG_SINEGEN_PRE),
            seg_sinegen_post: self.build_segment_cache(SEG_SINEGEN_POST),
            plbert_emb_cache: HashMap::new(),
            regulate_total_cache: HashMap::new(),
            sinegen_last_cumphase: None,
            mixed_precision: self.mixed_precision,
            autocast_policy: self.autocast_policy.clone(),
            auto_release: false, // shared instances cannot auto-release
            segment_cache_config: self.segment_cache_config.clone(),
            peephole_configs: self.peephole_configs.clone(),
            segment_autocast: self.segment_autocast.clone(),
            crown_verification: false,
            crown_config: nn_tts_verify::CrownCertificateConfig::default(),
            pipeline_mode: super::PipelineMode::default(),
            shape_policy: crate::compiled_model::ShapePolicy::Fixed,
            // ICB replay enabled by default on cloned dispatch instances.
            // Part of #4264.
            icb_replay: crate::compiled_model::icb::replay::IcbReplayBuffer::new(
                crate::compiled_model::icb::replay::IcbReplayConfig::enabled(),
            ),
            #[cfg(feature = "plan-serde")]
            optimization_results: None,
        }
    }

    /// Build a `SegmentCache` for segment `idx`, pre-populated with shared
    /// `CompiledModel` instances from the `Arc<CompiledModelDef>` entries.
    fn build_segment_cache(&self, idx: usize) -> SegmentCache {
        let store = &self.segments[idx];
        let cfg = &self.segment_cache_config;

        // Create cache with weight aliases (if available) for future compilations
        // of new shape variants.
        let mut cache = match &store.weight_aliases {
            Some(w) => {
                let aliases = w.iter().map(|(k, b)| (k.clone(), b.alias())).collect();
                SegmentCache::with_config_and_shared_weights(cfg, aliases)
            }
            None => SegmentCache::with_config(cfg),
        };

        // Pre-populate with shared compiled models.
        for (&key, def) in &store.defs {
            let model = CompiledModel::from_shared(Arc::clone(def));
            cache.insert(key, model);
        }

        cache
    }

    /// Number of pre-warmed segments (segments with at least one compiled entry).
    #[must_use]
    pub fn warmed_segment_count(&self) -> usize {
        self.segments.iter().filter(|s| !s.defs.is_empty()).count()
    }

    /// Total number of compiled shape variants across all segments.
    #[must_use]
    pub fn total_compiled_entries(&self) -> usize {
        self.segments.iter().map(SegmentStore::len).sum()
    }

    /// Total GPU weight bytes across all segment weight aliases.
    ///
    /// This is the approximate GPU memory that is shared (not duplicated)
    /// across all instances created from this cache.
    #[must_use]
    pub fn shared_gpu_weight_bytes(&self) -> usize {
        self.segments.iter().map(SegmentStore::weight_bytes).sum()
    }

    /// Reference count of the shared model state.
    ///
    /// Returns the `Arc::strong_count()` of the `SharedKokoroState`. This
    /// includes the cache itself plus any instances created via
    /// [`create_instance()`].
    #[must_use]
    pub fn shared_state_refcount(&self) -> usize {
        Arc::strong_count(&self.shared_state)
    }

    /// Per-segment summary: `(name, compiled_entries, weight_bytes)`.
    #[must_use]
    pub fn segment_summary(&self) -> Vec<(&'static str, usize, usize)> {
        const NAMES: [&str; NUM_SEGMENTS] = [
            "plbert",
            "text",
            "prosody",
            "f0",
            "generator",
            "regulate",
            "sinegen_pre",
            "sinegen_post",
        ];
        NAMES
            .iter()
            .zip(self.segments.iter())
            .map(|(&name, store)| (name, store.len(), store.weight_bytes()))
            .collect()
    }
}

impl std::fmt::Debug for SharedSegmentCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedSegmentCache")
            .field("warmed_segments", &self.warmed_segment_count())
            .field("total_entries", &self.total_compiled_entries())
            .field("shared_gpu_bytes", &self.shared_gpu_weight_bytes())
            .field("shared_state_refcount", &self.shared_state_refcount())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify SegmentStore::populate_from_cache with empty cache produces empty store.
    #[test]
    fn test_segment_store_from_empty_cache_is_empty() {
        let cache = SegmentCache::new();
        let store = SegmentStore::populate_from_cache(&cache);
        assert_eq!(store.len(), 0);
        assert_eq!(store.weight_bytes(), 0);
        assert!(store.weight_aliases.is_none());
    }

    /// Verify SharedSegmentCache from an unwarmed CompiledKokoro has 0 entries.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_shared_cache_from_unwarmed_instance() {
        crate::test_common::init();
        let config = nn_models::kokoro_tts::KokoroConfig::default();
        let model = nn_models::kokoro_tts::KokoroModel::load(
            nn_core::VarBuilder::zeros(nn_core::DType::F32, &nn_core::Device::Cpu),
            &config,
        )
        .expect("model from zeros");
        let kokoro = CompiledKokoro::new(model).expect("CompiledKokoro::new");

        let shared = SharedSegmentCache::from_compiled(&kokoro);
        assert_eq!(
            shared.warmed_segment_count(),
            0,
            "unwarmed instance should have 0 warmed segments"
        );
        assert_eq!(shared.total_compiled_entries(), 0);
        assert_eq!(shared.shared_gpu_weight_bytes(), 0);
        // 2 refs: kokoro.shared + shared_cache.shared_state
        assert_eq!(shared.shared_state_refcount(), 2);
    }

    /// Verify create_instance produces a CompiledKokoro with shared state.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_create_instance_shares_state() {
        crate::test_common::init();
        let config = nn_models::kokoro_tts::KokoroConfig::default();
        let model = nn_models::kokoro_tts::KokoroModel::load(
            nn_core::VarBuilder::zeros(nn_core::DType::F32, &nn_core::Device::Cpu),
            &config,
        )
        .expect("model from zeros");
        let kokoro = CompiledKokoro::new(model).expect("CompiledKokoro::new");

        let shared = SharedSegmentCache::from_compiled(&kokoro);
        let instance1 = shared.create_instance();
        let instance2 = shared.create_instance();

        // All instances share the same SharedKokoroState
        // kokoro.shared + shared_cache.shared_state + instance1 + instance2 = 4
        assert_eq!(
            shared.shared_state_refcount(),
            4,
            "should have 4 refs (primary + cache + 2 instances)"
        );
        assert_eq!(instance1.config().d_en, 512);
        assert_eq!(instance2.config().d_en, 512);
    }

    /// Verify Debug formatting includes key metrics.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_shared_cache_debug_format() {
        crate::test_common::init();
        let config = nn_models::kokoro_tts::KokoroConfig::default();
        let model = nn_models::kokoro_tts::KokoroModel::load(
            nn_core::VarBuilder::zeros(nn_core::DType::F32, &nn_core::Device::Cpu),
            &config,
        )
        .expect("model from zeros");
        let kokoro = CompiledKokoro::new(model).expect("CompiledKokoro::new");

        let shared = SharedSegmentCache::from_compiled(&kokoro);
        let debug = format!("{shared:?}");
        assert!(
            debug.contains("SharedSegmentCache"),
            "should contain type name"
        );
        assert!(
            debug.contains("warmed_segments"),
            "should contain warmed_segments"
        );
        assert!(
            debug.contains("total_entries"),
            "should contain total_entries"
        );
    }

    /// Verify segment_summary returns all 8 segments.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_segment_summary_has_all_segments() {
        crate::test_common::init();
        let config = nn_models::kokoro_tts::KokoroConfig::default();
        let model = nn_models::kokoro_tts::KokoroModel::load(
            nn_core::VarBuilder::zeros(nn_core::DType::F32, &nn_core::Device::Cpu),
            &config,
        )
        .expect("model from zeros");
        let kokoro = CompiledKokoro::new(model).expect("CompiledKokoro::new");

        let shared = SharedSegmentCache::from_compiled(&kokoro);
        let summary = shared.segment_summary();
        assert_eq!(summary.len(), 8, "should have 8 segment entries");
        let names: Vec<&str> = summary.iter().map(|(n, _, _)| *n).collect();
        assert_eq!(
            names,
            vec![
                "plbert",
                "text",
                "prosody",
                "f0",
                "generator",
                "regulate",
                "sinegen_pre",
                "sinegen_post"
            ]
        );
    }

    /// Verify peephole configs are propagated to instances.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_peephole_configs_propagated() {
        crate::test_common::init();
        let config = nn_models::kokoro_tts::KokoroConfig::default();
        let model = nn_models::kokoro_tts::KokoroModel::load(
            nn_core::VarBuilder::zeros(nn_core::DType::F32, &nn_core::Device::Cpu),
            &config,
        )
        .expect("model from zeros");

        let mut peephole = HashMap::new();
        let gen_config = nn_dsl::PeepholeConfig {
            fused_resblock: false,
            ..Default::default()
        };
        peephole.insert("generator".to_string(), gen_config);

        let kokoro = CompiledKokoro::new(model)
            .expect("CompiledKokoro::new")
            .with_peephole_configs(peephole);

        let shared = SharedSegmentCache::from_compiled(&kokoro);
        let instance = shared.create_instance();
        assert_eq!(
            instance.peephole_configs().len(),
            1,
            "instance should inherit peephole configs"
        );
        let stored = instance.peephole_configs().get("generator").unwrap();
        assert!(
            !stored.fused_resblock,
            "fused_resblock should be disabled in instance"
        );
    }
}
