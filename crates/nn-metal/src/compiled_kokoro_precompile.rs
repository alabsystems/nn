// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pre-compilation and runtime warmup for Kokoro pipeline segments.
//!
//! Two modes:
//!
//! - **Build-time** ([`precompile_kokoro_msl`]): Traces segments, exports MSL
//!   to `.metal` files for `build.rs` to compile to `.metallib`.
//! - **Runtime** ([`CompiledKokoro::warmup`]): Traces and compiles segments
//!   directly into the LRU segment caches, eliminating first-call latency.
//!
//! Both use [`PrecompileShapes`] to specify representative input sizes.
//!
//! Part of #2218, #2918.

use std::collections::HashMap;
use std::path::Path;

use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;

use nn_core::dyn_tensor::DynTensor;
use nn_core::DType;

use crate::cache::PipelineCache;

use super::{generator_total_samples, model_device, CompiledKokoro, CompiledKokoroError};

/// All Kokoro segment kind names used as keys in per-segment config maps.
const SEGMENT_KINDS: [&str; 8] = [
    "plbert",
    "text",
    "prosody",
    "f0_energy",
    "generator",
    "regulate",
    "sinegen_pre",
    "sinegen_post",
];

/// Per-segment [`PeepholeConfig`](nn_dsl::PeepholeConfig) overrides for the
/// Kokoro pipeline.
///
/// Each field corresponds to one of the 8 Kokoro pipeline segments. When a
/// segment has `Some(config)`, that config is used during compilation instead
/// of the default (all passes enabled). When `None`, the default is used.
///
/// Use [`load_from_dir()`](Self::load_from_dir) to load configs from a
/// directory of `<segment>_config.json` files produced by
/// [`save_to_dir()`](Self::save_to_dir) or the optimizer search.
///
/// Part of #3828.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SegmentPeepholeConfigs {
    /// PlBert + bert_encoder (segment 0).
    pub plbert: Option<nn_dsl::PeepholeConfig>,
    /// TextEncoder (segment 1).
    pub text: Option<nn_dsl::PeepholeConfig>,
    /// ProsodyPredictor (segment 2).
    pub prosody: Option<nn_dsl::PeepholeConfig>,
    /// F0EnergyPredictor (segment 3).
    pub f0_energy: Option<nn_dsl::PeepholeConfig>,
    /// Generator (segment 4).
    pub generator: Option<nn_dsl::PeepholeConfig>,
    /// Regulate (segment 5).
    pub regulate: Option<nn_dsl::PeepholeConfig>,
    /// SineGen pre-cumsum (segment 5a).
    pub sinegen_pre: Option<nn_dsl::PeepholeConfig>,
    /// SineGen post-cumsum (segment 5b).
    pub sinegen_post: Option<nn_dsl::PeepholeConfig>,
}

impl SegmentPeepholeConfigs {
    /// Create a new empty config (all segments use defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up the config for a segment by kind name.
    ///
    /// Accepted names: `"plbert"`, `"text"`, `"prosody"`, `"f0_energy"`,
    /// `"f0"` (alias for f0_energy), `"generator"`, `"regulate"`,
    /// `"sinegen_pre"`, `"sinegen_post"`.
    #[must_use]
    pub fn for_segment(&self, kind: &str) -> Option<&nn_dsl::PeepholeConfig> {
        match kind {
            "plbert" => self.plbert.as_ref(),
            "text" => self.text.as_ref(),
            "prosody" => self.prosody.as_ref(),
            "f0_energy" | "f0" => self.f0_energy.as_ref(),
            "generator" => self.generator.as_ref(),
            "regulate" => self.regulate.as_ref(),
            "sinegen_pre" => self.sinegen_pre.as_ref(),
            "sinegen_post" => self.sinegen_post.as_ref(),
            _ => None,
        }
    }

    /// Convert to a `HashMap<String, PeepholeConfig>` compatible with
    /// [`CompiledKokoro::with_peephole_configs()`].
    ///
    /// Only segments with `Some` configs are included. The key names match
    /// the segment names used internally by `CompiledKokoro` (e.g., `"f0"`
    /// for the F0/energy segment).
    #[must_use]
    pub fn to_hashmap(&self) -> HashMap<String, nn_dsl::PeepholeConfig> {
        let mut map = HashMap::new();
        if let Some(ref c) = self.plbert {
            map.insert("plbert".to_string(), c.clone());
        }
        if let Some(ref c) = self.text {
            map.insert("text".to_string(), c.clone());
        }
        if let Some(ref c) = self.prosody {
            map.insert("prosody".to_string(), c.clone());
        }
        if let Some(ref c) = self.f0_energy {
            // Internal key is "f0" for backward compatibility with existing
            // peephole_configs HashMap usage.
            map.insert("f0".to_string(), c.clone());
        }
        if let Some(ref c) = self.generator {
            map.insert("generator".to_string(), c.clone());
        }
        if let Some(ref c) = self.regulate {
            map.insert("regulate".to_string(), c.clone());
        }
        if let Some(ref c) = self.sinegen_pre {
            map.insert("sinegen_pre".to_string(), c.clone());
        }
        if let Some(ref c) = self.sinegen_post {
            map.insert("sinegen_post".to_string(), c.clone());
        }
        map
    }

    /// Construct from a `HashMap<String, PeepholeConfig>` (e.g., from
    /// [`load_peephole_configs()`](super::load_peephole_configs)).
    ///
    /// Recognizes both `"f0_energy"` and `"f0"` as the F0/energy segment.
    #[must_use]
    pub fn from_hashmap(map: &HashMap<String, nn_dsl::PeepholeConfig>) -> Self {
        Self {
            plbert: map.get("plbert").cloned(),
            text: map.get("text").cloned(),
            prosody: map.get("prosody").cloned(),
            f0_energy: map.get("f0_energy").or_else(|| map.get("f0")).cloned(),
            generator: map.get("generator").cloned(),
            regulate: map.get("regulate").cloned(),
            sinegen_pre: map.get("sinegen_pre").cloned(),
            sinegen_post: map.get("sinegen_post").cloned(),
        }
    }

    /// Count how many segments have custom (non-default) configs.
    #[must_use]
    pub fn configured_count(&self) -> usize {
        [
            &self.plbert,
            &self.text,
            &self.prosody,
            &self.f0_energy,
            &self.generator,
            &self.regulate,
            &self.sinegen_pre,
            &self.sinegen_post,
        ]
        .iter()
        .filter(|c| c.is_some())
        .count()
    }

    /// Load per-segment configs from a directory of JSON files.
    ///
    /// Reads `<segment>_config.json` for each of the 8 segment kinds.
    /// Missing files are silently skipped (that segment uses defaults).
    ///
    /// Requires the `plan-serde` feature (for PeepholeConfig serde derives).
    ///
    /// # Errors
    ///
    /// Returns [`CompiledKokoroError::ConfigLoad`] if a file exists but
    /// contains invalid JSON.
    #[cfg(feature = "plan-serde")]
    pub fn load_from_dir(dir: impl AsRef<Path>) -> Result<Self, CompiledKokoroError> {
        let dir = dir.as_ref();
        let mut configs = Self::new();

        for kind in &SEGMENT_KINDS {
            let path = dir.join(format!("{kind}_config.json"));
            if !path.exists() {
                continue;
            }
            let data = std::fs::read_to_string(&path).map_err(|e| {
                CompiledKokoroError::ConfigLoad(format!("read {}: {e}", path.display()))
            })?;
            let config: nn_dsl::PeepholeConfig = serde_json::from_str(&data).map_err(|e| {
                CompiledKokoroError::ConfigLoad(format!("parse {}: {e}", path.display()))
            })?;
            match *kind {
                "plbert" => configs.plbert = Some(config),
                "text" => configs.text = Some(config),
                "prosody" => configs.prosody = Some(config),
                "f0_energy" => configs.f0_energy = Some(config),
                "generator" => configs.generator = Some(config),
                "regulate" => configs.regulate = Some(config),
                "sinegen_pre" => configs.sinegen_pre = Some(config),
                "sinegen_post" => configs.sinegen_post = Some(config),
                _ => {}
            }
        }

        Ok(configs)
    }

    /// Save per-segment configs to a directory of JSON files.
    ///
    /// Writes `<segment>_config.json` for each segment that has a
    /// `Some` config. Segments with `None` are skipped (no file written).
    /// Creates the directory if it does not exist.
    ///
    /// Uses pretty-printed JSON for debuggability and diffing.
    ///
    /// Requires the `plan-serde` feature (for PeepholeConfig serde derives).
    ///
    /// # Errors
    ///
    /// Returns [`CompiledKokoroError::ConfigLoad`] on I/O or serialization
    /// failure.
    #[cfg(feature = "plan-serde")]
    pub fn save_to_dir(&self, dir: impl AsRef<Path>) -> Result<(), CompiledKokoroError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(|e| {
            CompiledKokoroError::ConfigLoad(format!("create dir {}: {e}", dir.display()))
        })?;

        let segments: [(&str, &Option<nn_dsl::PeepholeConfig>); 8] = [
            ("plbert", &self.plbert),
            ("text", &self.text),
            ("prosody", &self.prosody),
            ("f0_energy", &self.f0_energy),
            ("generator", &self.generator),
            ("regulate", &self.regulate),
            ("sinegen_pre", &self.sinegen_pre),
            ("sinegen_post", &self.sinegen_post),
        ];

        for (kind, config_opt) in &segments {
            if let Some(config) = config_opt {
                let path = dir.join(format!("{kind}_config.json"));
                let json = serde_json::to_string_pretty(config).map_err(|e| {
                    CompiledKokoroError::ConfigLoad(format!("serialize {kind}: {e}"))
                })?;
                std::fs::write(&path, json).map_err(|e| {
                    CompiledKokoroError::ConfigLoad(format!("write {}: {e}", path.display()))
                })?;
            }
        }

        Ok(())
    }
}

// -- Cache invalidation for SegmentPeepholeConfigs (#3828 Phase 2B) -----------

/// Composite key for invalidating cached [`SegmentPeepholeConfigs`].
///
/// A cached config is valid only when the segment kind, input shapes,
/// and nn version all match. A version bump invalidates all cached
/// configs; a shape change invalidates per-segment configs.
///
/// Part of #3828 Phase 2B.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SegmentConfigCacheKey {
    /// Segment kind name (e.g., `"plbert"`, `"generator"`).
    pub segment_kind: String,
    /// Input tensor shapes for this segment.
    pub input_shapes: Vec<Vec<usize>>,
    /// nn crate version at the time the config was computed.
    pub nn_version: String,
}

impl SegmentConfigCacheKey {
    /// Create a new cache key.
    #[must_use]
    pub fn new(segment_kind: impl Into<String>, input_shapes: Vec<Vec<usize>>) -> Self {
        Self {
            segment_kind: segment_kind.into(),
            input_shapes,
            nn_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Create a cache key with an explicit version (for testing).
    #[must_use]
    pub fn with_version(
        segment_kind: impl Into<String>,
        input_shapes: Vec<Vec<usize>>,
        nn_version: impl Into<String>,
    ) -> Self {
        Self {
            segment_kind: segment_kind.into(),
            input_shapes,
            nn_version: nn_version.into(),
        }
    }
}

/// A cache wrapper around [`SegmentPeepholeConfigs`] with composite-key
/// invalidation.
///
/// Stores per-segment [`SegmentConfigCacheKey`]s alongside the configs.
/// On load, each segment's key is checked against the current key;
/// stale entries (different shape or nn version) are automatically
/// invalidated.
///
/// Part of #3828 Phase 2B.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SegmentConfigCache {
    /// Per-segment cache keys. Key = segment name (e.g., `"plbert"`).
    pub keys: HashMap<String, SegmentConfigCacheKey>,
    /// The cached peephole configs.
    pub configs: SegmentPeepholeConfigs,
}

impl SegmentConfigCache {
    /// Create a new empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a cache from existing configs and keys.
    #[must_use]
    pub fn from_parts(
        configs: SegmentPeepholeConfigs,
        keys: HashMap<String, SegmentConfigCacheKey>,
    ) -> Self {
        Self { keys, configs }
    }

    /// Check whether a segment's cached config is still valid.
    ///
    /// Returns `true` if the segment has a cached key that matches
    /// `current_key` exactly (segment kind, input shapes, and nn version).
    /// Returns `false` if no key is cached or if any component differs.
    #[must_use]
    pub fn is_valid(&self, segment: &str, current_key: &SegmentConfigCacheKey) -> bool {
        self.keys.get(segment).is_some_and(|k| k == current_key)
    }

    /// Invalidate the cached config for a single segment.
    ///
    /// Removes both the cache key and the peephole config for `segment`.
    pub fn invalidate(&mut self, segment: &str) {
        self.keys.remove(segment);
        self.set_segment_config(segment, None);
    }

    /// Invalidate all segments whose cached key does not match the
    /// corresponding entry in `current_keys`.
    ///
    /// Segments not present in `current_keys` are left untouched (they
    /// will be invalidated on the next per-segment validity check).
    ///
    /// Returns the number of segments invalidated.
    pub fn invalidate_stale(&mut self, current_keys: &HashMap<String, SegmentConfigCacheKey>) -> usize {
        let stale: Vec<String> = current_keys
            .iter()
            .filter(|(seg, key)| !self.is_valid(seg, key))
            .map(|(seg, _)| seg.clone())
            .collect();
        let count = stale.len();
        for seg in stale {
            self.invalidate(&seg);
        }
        count
    }

    /// Insert or update a segment's config and cache key.
    pub fn insert(
        &mut self,
        segment: &str,
        config: nn_dsl::PeepholeConfig,
        key: SegmentConfigCacheKey,
    ) {
        self.keys.insert(segment.to_string(), key);
        self.set_segment_config(segment, Some(config));
    }

    /// Save the cache (configs + keys) to a directory.
    ///
    /// Writes the configs via [`SegmentPeepholeConfigs::save_to_dir()`]
    /// and persists the keys as `_cache_keys.json` in the same directory.
    ///
    /// Requires the `plan-serde` feature.
    #[cfg(feature = "plan-serde")]
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<(), CompiledKokoroError> {
        let dir = dir.as_ref();
        self.configs.save_to_dir(dir)?;

        let keys_path = dir.join("_cache_keys.json");
        let json = serde_json::to_string_pretty(&self.keys).map_err(|e| {
            CompiledKokoroError::ConfigLoad(format!("serialize cache keys: {e}"))
        })?;
        std::fs::write(&keys_path, json).map_err(|e| {
            CompiledKokoroError::ConfigLoad(format!("write {}: {e}", keys_path.display()))
        })?;

        Ok(())
    }

    /// Load a cache (configs + keys) from a directory.
    ///
    /// Loads configs via [`SegmentPeepholeConfigs::load_from_dir()`]
    /// and reads keys from `_cache_keys.json`. If the keys file is
    /// missing, an empty key map is used (all segments will fail
    /// validity checks).
    ///
    /// Requires the `plan-serde` feature.
    #[cfg(feature = "plan-serde")]
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, CompiledKokoroError> {
        let dir = dir.as_ref();
        let configs = SegmentPeepholeConfigs::load_from_dir(dir)?;

        let keys_path = dir.join("_cache_keys.json");
        let keys = if keys_path.exists() {
            let data = std::fs::read_to_string(&keys_path).map_err(|e| {
                CompiledKokoroError::ConfigLoad(format!("read {}: {e}", keys_path.display()))
            })?;
            serde_json::from_str(&data).map_err(|e| {
                CompiledKokoroError::ConfigLoad(format!("parse {}: {e}", keys_path.display()))
            })?
        } else {
            HashMap::new()
        };

        Ok(Self { keys, configs })
    }

    /// Set the peephole config for a segment by name.
    fn set_segment_config(&mut self, segment: &str, config: Option<nn_dsl::PeepholeConfig>) {
        match segment {
            "plbert" => self.configs.plbert = config,
            "text" => self.configs.text = config,
            "prosody" => self.configs.prosody = config,
            "f0_energy" | "f0" => self.configs.f0_energy = config,
            "generator" => self.configs.generator = config,
            "regulate" => self.configs.regulate = config,
            "sinegen_pre" => self.configs.sinegen_pre = config,
            "sinegen_post" => self.configs.sinegen_post = config,
            _ => {}
        }
    }
}

/// Representative Kokoro input shapes for pre-compilation.
///
/// These cover the common text lengths and output durations. The
/// segment caches key by `seq_len` (segments 0-2) and `t_mel`
/// (segments 3-4), so pre-compiling these shapes avoids runtime
/// compilation for the most frequent inputs.
pub struct PrecompileShapes {
    /// Token sequence lengths (segments 0-2). Default: [10, 20, 40, 80].
    pub seq_lens: Vec<usize>,
    /// T_mel values for F0 and Generator (segments 3-4).
    /// Default: [20, 40, 80, 160, 320].
    pub t_mels: Vec<usize>,
}

impl PrecompileShapes {
    /// Create with default shapes.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Short texts only (chat messages, subtitles).
    ///
    /// Covers token sequences up to ~128 tokens and corresponding
    /// mel frames. Suitable for interactive TTS where inputs are
    /// typically single sentences.
    #[must_use]
    pub fn short() -> Self {
        Self {
            seq_lens: vec![10, 20, 40, 80],
            t_mels: vec![20, 40, 80, 160],
        }
    }

    /// Long-form synthesis (articles, books, narration).
    ///
    /// Covers larger token sequences and mel frame counts for
    /// paragraph-length inputs. Includes the full default range
    /// plus extended sizes for longer texts.
    #[must_use]
    pub fn long_form() -> Self {
        Self {
            seq_lens: vec![40, 80, 160, 256, 512],
            t_mels: vec![80, 160, 320, 640, 1024],
        }
    }

    /// Chorus synthesis (multiple short-medium texts).
    ///
    /// Optimized for multi-voice chorus where each voice synthesizes
    /// short-to-medium length texts. Covers the overlap region between
    /// short and default shapes for best cache hit rates across voices.
    #[must_use]
    pub fn chorus() -> Self {
        Self {
            seq_lens: vec![20, 40, 80, 128],
            t_mels: vec![40, 80, 160, 320],
        }
    }

    /// Override token sequence lengths (segments 0-2).
    #[must_use]
    pub fn with_seq_lens(mut self, lens: Vec<usize>) -> Self {
        self.seq_lens = lens;
        self
    }

    /// Override T_mel values (segments 3-4).
    #[must_use]
    pub fn with_t_mels(mut self, mels: Vec<usize>) -> Self {
        self.t_mels = mels;
        self
    }

    /// Derive T_frames values from T_mel (t_frames = 2 × t_mel).
    ///
    /// Used by sinegen segments (5a, 5b) which operate at double the mel rate.
    pub fn t_frames(&self) -> Vec<usize> {
        self.t_mels.iter().map(|&t| 2 * t).collect()
    }

    /// Create tailored precompile shapes from known token sequence lengths.
    ///
    /// Given the actual token lengths of chunks that will be synthesized,
    /// this generates a [`PrecompileShapes`] with:
    ///
    /// - `seq_lens`: the unique, sorted token lengths from the input.
    /// - `t_mels`: estimated mel frame counts derived from the token lengths.
    ///   Uses a heuristic of ~3x the token length (typical Kokoro average
    ///   duration per phoneme), covering the likely range.
    ///
    /// This is more efficient than the generic presets because it compiles
    /// only the shapes that will actually be used, avoiding wasted warmup
    /// time on shapes that never occur.
    ///
    /// Returns `None` if `token_lengths` is empty.
    #[must_use]
    pub fn from_token_lengths(token_lengths: &[usize]) -> Option<Self> {
        if token_lengths.is_empty() {
            return None;
        }

        // Deduplicate and sort seq_lens.
        let mut seq_lens: Vec<usize> = token_lengths.to_vec();
        seq_lens.sort_unstable();
        seq_lens.dedup();

        // Estimate t_mels from token lengths. Kokoro's duration predictor
        // typically yields 2-4 mel frames per phoneme token (average ~3).
        // We generate a range covering the likely output for each unique
        // token length, then deduplicate.
        let mut t_mels: Vec<usize> = Vec::new();
        for &seq_len in &seq_lens {
            // Conservative range: 2x to 4x the token length.
            let low = (seq_len * 2).max(1);
            let mid = seq_len * 3;
            let high = seq_len * 4;
            t_mels.push(low);
            t_mels.push(mid);
            t_mels.push(high);
        }
        t_mels.sort_unstable();
        t_mels.dedup();

        Some(Self { seq_lens, t_mels })
    }
}

impl Default for PrecompileShapes {
    fn default() -> Self {
        Self {
            seq_lens: vec![10, 20, 40, 80],
            t_mels: vec![20, 40, 80, 160, 320],
        }
    }
}

/// Result of a pre-compilation run.
#[derive(Debug)]
#[non_exhaustive]
pub struct PrecompileResult {
    /// Number of `.metal` files written.
    pub files_written: usize,
    /// Total MSL source bytes written.
    pub total_bytes: usize,
    /// Per-segment file counts.
    pub segment_counts: Vec<(&'static str, usize)>,
}

/// Pre-compile Kokoro pipeline segments to MSL files.
///
/// Traces each segment for the representative shapes in `shapes`,
/// compiles each trace to a [`CompiledPlan`], and exports MSL source
/// to `.metal` files in `output_dir`. The files are named
/// `{segment}_{shape}_{step}_{kernel}.metal`.
///
/// After running this, `cargo build` will pick up the `.metal` files
/// and compile them to a `.metallib` via `build.rs`.
///
/// # Errors
///
/// Returns an error if tracing or MSL codegen fails for any segment.
pub fn precompile_kokoro_msl(
    kokoro: &CompiledKokoro,
    output_dir: impl AsRef<Path>,
    shapes: &PrecompileShapes,
) -> Result<PrecompileResult, CompiledKokoroError> {
    let dir = output_dir.as_ref();
    std::fs::create_dir_all(dir)?;

    let mut total_files = 0;
    let mut total_bytes = 0;
    let mut segment_counts = Vec::new();

    // Segment 0: PlBert + bert_encoder (seq_len dependent)
    let seg0_count = precompile_segment_plbert(kokoro, dir, &shapes.seq_lens)?;
    total_files += seg0_count.0;
    total_bytes += seg0_count.1;
    segment_counts.push(("plbert", seg0_count.0));

    // Segment 1: TextEncoder (seq_len dependent)
    let seg1_count = precompile_segment_text(kokoro, dir, &shapes.seq_lens)?;
    total_files += seg1_count.0;
    total_bytes += seg1_count.1;
    segment_counts.push(("text", seg1_count.0));

    // Segment 2: ProsodyPredictor (seq_len dependent)
    let seg2_count = precompile_segment_prosody(kokoro, dir, &shapes.seq_lens)?;
    total_files += seg2_count.0;
    total_bytes += seg2_count.1;
    segment_counts.push(("prosody", seg2_count.0));

    // Segment 3: F0EnergyPredictor (t_mel dependent)
    let seg3_count = precompile_segment_f0(kokoro, dir, &shapes.t_mels)?;
    total_files += seg3_count.0;
    total_bytes += seg3_count.1;
    segment_counts.push(("f0", seg3_count.0));

    // Segment 4: Generator (t_mel dependent, cache key = 2 * t_mel * upsample_factor)
    let seg4_count = precompile_segment_generator(kokoro, dir, &shapes.t_mels)?;
    total_files += seg4_count.0;
    total_bytes += seg4_count.1;
    segment_counts.push(("generator", seg4_count.0));

    // Segment 5: Regulate (seq_len dependent, no model weights)
    let seg5_count = precompile_segment_regulate(kokoro, dir, &shapes.seq_lens)?;
    total_files += seg5_count.0;
    total_bytes += seg5_count.1;
    segment_counts.push(("regulate", seg5_count.0));

    // Segment 5a: SineGen pre-cumsum (t_frames dependent)
    let t_frames = shapes.t_frames();
    let seg5a_count = precompile_segment_sinegen_pre(kokoro, dir, &t_frames)?;
    total_files += seg5a_count.0;
    total_bytes += seg5a_count.1;
    segment_counts.push(("sinegen_pre", seg5a_count.0));

    // Segment 5b: SineGen post-cumsum (t_frames dependent, has SourceModule Linear)
    let seg5b_count = precompile_segment_sinegen_post(kokoro, dir, &t_frames)?;
    total_files += seg5b_count.0;
    total_bytes += seg5b_count.1;
    segment_counts.push(("sinegen_post", seg5b_count.0));

    Ok(PrecompileResult {
        files_written: total_files,
        total_bytes,
        segment_counts,
    })
}

// -- Runtime warmup: pre-compile segments into caches (#2918) ----------------

impl CompiledKokoro {
    /// Pre-compile pipeline segments into the segment caches at runtime,
    /// eliminating first-call compilation latency.
    ///
    /// Unlike [`precompile_kokoro_msl`] (build-time MSL file generation), this
    /// method traces and compiles segments directly into the LRU segment caches.
    /// Subsequent `synthesize()` calls that hit these shapes skip compilation.
    ///
    /// Warms up all 8 compiled segments: PlBert (0), TextEncoder (1),
    /// Prosody (2), F0 (3), Generator (4), Regulate (5), SineGen pre (5a),
    /// SineGen post (5b).
    ///
    /// Requires model weights — must be called before [`release_model_weights()`].
    ///
    /// # Arguments
    ///
    /// * `shapes` — Representative token lengths and mel frame counts.
    ///   [`PrecompileShapes::default()`] covers common input sizes.
    /// * `cache` — Metal pipeline cache.
    ///
    /// # Returns
    ///
    /// Number of segment compilations performed (skips already-cached shapes).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut kokoro = unsafe { CompiledKokoro::load("kokoro.safetensors")? };
    /// let cache = PipelineCache::new();
    /// let count = kokoro.warmup(&PrecompileShapes::default(), &cache)?;
    /// // First synthesize() with common input sizes is now fast.
    /// ```
    pub fn warmup(
        &mut self,
        shapes: &PrecompileShapes,
        cache: &PipelineCache,
    ) -> Result<usize, CompiledKokoroError> {
        if self.weights_released() {
            return Err(CompiledKokoroError::WeightsReleased);
        }

        let dev = model_device(self.shared.model.as_ref());
        let config = self.shared.config.clone();
        let d_en = config.d_en;
        let style_dim = config.style_dim;
        let max_dur = config.max_dur;
        let n_fft = config.n_fft;
        let prosody_dim = d_en + style_dim;
        let upsample_factor: usize = config.upsample_rates.iter().product();
        let hop_length = n_fft / 4;
        let n_bins = n_fft / 2 + 1;
        let source_upsample = upsample_factor * hop_length;
        let mut compiled = 0usize;

        // Segments keyed by seq_len: PlBert (0), TextEncoder (1), Prosody (2), Regulate (5).
        for &seq_len in &shapes.seq_lens {
            let input_ids = DynTensor::zeros(&[1, seq_len], DType::F32, &dev)?;
            if self.seg_plbert.get(seq_len).is_none() {
                self.ensure_seg_plbert(seq_len, &input_ids, cache)?;
                compiled += 1;
            }
            if self.seg_text.get(seq_len).is_none() {
                self.ensure_seg_text(seq_len, &input_ids, cache)?;
                compiled += 1;
            }
            if self.seg_prosody.get(seq_len).is_none() {
                let bert_features = DynTensor::zeros(&[1, d_en, seq_len], DType::F32, &dev)?;
                let style = DynTensor::zeros(&[1, style_dim], DType::F32, &dev)?;
                self.ensure_seg_prosody(seq_len, &bert_features, &style, cache)?;
                compiled += 1;
            }
            if self.seg_regulate.get(seq_len).is_none() {
                let dur_logits =
                    DynTensor::zeros(&[1, seq_len, max_dur], DType::F32, &dev)?;
                let speed_inv = DynTensor::full(&[1], 1.0, DType::F32, &dev)?;
                self.ensure_seg_regulate(seq_len, &dur_logits, &speed_inv, cache)?;
                compiled += 1;
            }
        }

        // Segments keyed by t_mel: F0 (3), Generator (4).
        for &t_mel in &shapes.t_mels {
            if self.seg_f0.get(t_mel).is_none() {
                let aligned = DynTensor::zeros(&[1, prosody_dim, t_mel], DType::F32, &dev)?;
                let style = DynTensor::zeros(&[1, style_dim], DType::F32, &dev)?;
                self.ensure_seg_f0(t_mel, &aligned, &style, cache)?;
                compiled += 1;
            }
            let total_samples = generator_total_samples(t_mel, upsample_factor)?;
            if self.seg_generator.get(total_samples).is_none() {
                let t_f0 = 2 * t_mel;
                let t_audio = t_f0 * source_upsample;
                let t_stft = t_audio / hop_length + 1;
                let regulated = DynTensor::zeros(&[1, d_en, t_mel], DType::F32, &dev)?;
                let f0 = DynTensor::zeros(&[1, 1, t_f0], DType::F32, &dev)?;
                let energy = DynTensor::zeros(&[1, 1, t_f0], DType::F32, &dev)?;
                let decoder_style = DynTensor::zeros(&[1, style_dim], DType::F32, &dev)?;
                let har_source =
                    DynTensor::zeros(&[1, 2 * n_bins, t_stft], DType::F32, &dev)?;
                self.ensure_seg_generator(
                    total_samples,
                    &regulated,
                    &f0,
                    &energy,
                    &decoder_style,
                    &har_source,
                    cache,
                )?;
                compiled += 1;
            }
        }

        // Segments keyed by t_frames (= 2 × t_mel): SineGen pre (5a), post (5b).
        // Extract Copy values from source_module before mutable calls.
        let sinegen_params = self.shared.source_module.as_ref().map(|sm| {
            (
                sm.linear().weight().device(),
                sm.sine_gen().n_channels(),
                f64::from(sm.sine_gen().voiced_threshold()),
            )
        });
        if let Some((sinegen_dev, n_ch, voiced_threshold)) = sinegen_params {
            for &t_mel in &shapes.t_mels {
                let t_frames = 2 * t_mel;
                if self.seg_sinegen_pre.get(t_frames).is_none() {
                    let f0_in =
                        DynTensor::zeros(&[1, t_frames, 1], DType::F32, &sinegen_dev)?;
                    self.ensure_seg_sinegen_pre(t_frames, &f0_in, source_upsample, cache)?;
                    compiled += 1;
                }
                if self.seg_sinegen_post.get(t_frames).is_none() {
                    let cum =
                        DynTensor::zeros(&[1, t_frames, n_ch], DType::F32, &sinegen_dev)?;
                    let f0_warmup =
                        DynTensor::zeros(&[1, t_frames, 1], DType::F32, &sinegen_dev)?;
                    self.ensure_seg_sinegen_post(
                        t_frames, &cum, &f0_warmup, source_upsample, voiced_threshold, cache,
                    )?;
                    compiled += 1;
                }
            }
        }

        Ok(compiled)
    }

    /// Pre-compile pipeline segments for configurable shapes, returning
    /// a structured result.
    ///
    /// This wraps [`warmup`](Self::warmup) to provide workload-specific
    /// shape presets and a structured [`WarmupShapesResult`].
    ///
    /// Use the convenience constructors on [`PrecompileShapes`] to target
    /// specific workloads:
    /// - [`PrecompileShapes::short()`] -- chat/subtitle latency-sensitive use
    /// - [`PrecompileShapes::long_form()`] -- narration/article synthesis
    /// - [`PrecompileShapes::chorus()`] -- multi-voice chorus pipelines
    ///
    /// # Arguments
    ///
    /// * `shapes` -- Token lengths and mel frame counts to precompile.
    /// * `cache` -- Metal pipeline cache.
    ///
    /// # Returns
    ///
    /// A [`WarmupShapesResult`] with the number of segment compilations
    /// performed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut kokoro = unsafe { CompiledKokoro::load("kokoro.safetensors")? };
    /// let cache = PipelineCache::new();
    ///
    /// // Warm up for short texts.
    /// let result = kokoro.warmup_shapes(&PrecompileShapes::short(), &cache)?;
    /// println!("Compiled {} segments", result.segments_compiled);
    /// ```
    ///
    /// Part of #3873.
    pub fn warmup_shapes(
        &mut self,
        shapes: &PrecompileShapes,
        cache: &PipelineCache,
    ) -> Result<WarmupShapesResult, CompiledKokoroError> {
        let segments_compiled = self.warmup(shapes, cache)?;
        Ok(WarmupShapesResult { segments_compiled })
    }

    /// Pre-compile pipeline segments using per-segment
    /// [`PeepholeConfig`](nn_dsl::PeepholeConfig) overrides.
    ///
    /// This combines [`SegmentPeepholeConfigs`] with the runtime warmup
    /// from [`warmup()`](Self::warmup). The per-segment configs are
    /// applied via [`with_peephole_configs()`](Self::with_peephole_configs),
    /// then all segments are compiled for the specified shapes.
    ///
    /// When a segment has a config in `segment_configs`, that config is
    /// used during compilation. Segments without a config use the default
    /// (all peephole passes enabled).
    ///
    /// # Arguments
    ///
    /// * `shapes` -- Representative token lengths and mel frame counts.
    /// * `cache` -- Metal pipeline cache.
    /// * `segment_configs` -- Per-segment peephole configurations. Pass
    ///   `None` to use defaults for all segments (equivalent to
    ///   [`warmup()`](Self::warmup)).
    ///
    /// # Returns
    ///
    /// Number of segment compilations performed (skips already-cached shapes).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut kokoro = unsafe { CompiledKokoro::load("kokoro.safetensors")? };
    /// let cache = PipelineCache::new();
    ///
    /// // Load per-segment configs from optimizer output.
    /// let configs = SegmentPeepholeConfigs::load_from_dir("./kokoro_configs/")?;
    /// let compiled = kokoro.precompile_segments_optimized(
    ///     &PrecompileShapes::default(), &cache, Some(&configs),
    /// )?;
    /// println!("Compiled {compiled} segments with custom peephole configs");
    /// ```
    pub fn precompile_segments_optimized(
        &mut self,
        shapes: &PrecompileShapes,
        cache: &PipelineCache,
        segment_configs: Option<&SegmentPeepholeConfigs>,
    ) -> Result<usize, CompiledKokoroError> {
        if let Some(configs) = segment_configs {
            // Apply the structured configs as a HashMap to the existing
            // peephole_configs field used by compile_seg_* methods.
            self.peephole_configs = configs.to_hashmap();

            // Clear segment caches so any entries compiled with the previous
            // (possibly default) configs are evicted. The warmup will
            // recompile with the new configs.
            self.clear_segment_caches();
        }

        self.warmup(shapes, cache)
    }
}

/// Result of a [`CompiledKokoro::warmup_shapes`] invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WarmupShapesResult {
    /// Number of segment compilations performed during warmup.
    /// Shapes already present in the cache are skipped.
    pub segments_compiled: usize,
}

// -- Optimizer-aware warmup (#3828) -------------------------------------------

/// Result of a [`CompiledKokoro::warmup_with_optimizer`] invocation.
#[cfg(feature = "plan-serde")]
#[derive(Debug)]
#[non_exhaustive]
pub struct OptimizerWarmupResult {
    /// Whether configs were loaded from a cached file (`true`) or
    /// computed fresh via optimizer search (`false`).
    pub loaded_from_cache: bool,
    /// Number of per-segment peephole configs applied.
    pub configs_applied: usize,
    /// Number of segment compilations performed during warmup.
    pub segments_compiled: usize,
}

#[cfg(feature = "plan-serde")]
impl CompiledKokoro {
    /// Pre-compile pipeline segments using optimal per-segment
    /// [`PeepholeConfig`](nn_dsl::PeepholeConfig)s.
    ///
    /// This combines the optimizer search from
    /// [`segment_optimizer_search`](CompiledKokoro::segment_optimizer_search)
    /// with the runtime warmup from [`warmup`](CompiledKokoro::warmup),
    /// adding a caching layer for the optimal configs.
    ///
    /// # Behavior
    ///
    /// 1. If `config_cache_path` is `Some` and the file exists, loads the
    ///    per-segment configs from JSON and applies them.
    /// 2. If `config_cache_path` is `Some` but the file does not exist,
    ///    runs the optimizer search with `per_segment_budget`, saves the
    ///    results to the file, and applies them.
    /// 3. If `config_cache_path` is `None`, runs the optimizer search
    ///    without caching.
    /// 4. In all cases, applies the configs via
    ///    [`with_peephole_configs`](CompiledKokoro::with_peephole_configs)
    ///    then runs [`warmup`](CompiledKokoro::warmup).
    ///
    /// # Arguments
    ///
    /// * `shapes` — Representative token lengths and mel frame counts.
    /// * `cache` — Metal pipeline cache.
    /// * `input_ids` — `[B, T]` token indices for the optimizer trace.
    /// * `style` — `[B, 2*style_dim]` voice embedding for the optimizer trace.
    /// * `speed` — Speaking rate multiplier for the optimizer trace.
    /// * `per_segment_budget` — Maximum time to spend optimizing each segment.
    /// * `config_cache_path` — Optional path to a JSON file for caching
    ///   optimal configs. When `None`, no caching is performed.
    ///
    /// # Returns
    ///
    /// An [`OptimizerWarmupResult`] with details about the operation.
    ///
    /// # Errors
    ///
    /// Returns [`CompiledKokoroError`] if loading, optimizing, saving,
    /// or warmup fails.
    ///
    /// Part of #3828.
    pub fn warmup_with_optimizer(
        &mut self,
        shapes: &PrecompileShapes,
        cache: &PipelineCache,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        per_segment_budget: std::time::Duration,
        config_cache_path: Option<&Path>,
    ) -> Result<OptimizerWarmupResult, CompiledKokoroError> {
        let (configs, loaded_from_cache, stored_results) = match config_cache_path {
            Some(path) if path.exists() => {
                let configs = super::load_peephole_configs(path)?;
                // When loading from cache, we don't have full OptimizationResult
                // data (the cached file only stores PeepholeConfig). Mark results
                // as None — the configs are still applied.
                (configs, true, None)
            }
            Some(path) => {
                let (configs, stored) = self.run_optimizer_and_save(
                    input_ids,
                    style,
                    speed,
                    cache,
                    per_segment_budget,
                    path,
                )?;
                (configs, false, Some(stored))
            }
            None => {
                let (configs, stored) = self.run_optimizer_to_configs(
                    input_ids,
                    style,
                    speed,
                    cache,
                    per_segment_budget,
                )?;
                (configs, false, Some(stored))
            }
        };

        let configs_applied = configs.len();
        self.peephole_configs = configs;
        #[cfg(feature = "plan-serde")]
        {
            self.optimization_results = stored_results;
            // Print optimization summary if results are available.
            if self.optimization_results.is_some() {
                eprintln!("{}", self.optimization_summary());
            }
        }

        // Clear all segment caches so entries compiled during the optimizer
        // search (which used default configs) are evicted. Without this,
        // warmup() would skip shapes already cached with sub-optimal configs.
        // Shared GPU weight buffers survive the clear (zero-copy aliasing).
        // Part of #3828.
        self.clear_segment_caches();

        let segments_compiled = self.warmup(shapes, cache)?;

        Ok(OptimizerWarmupResult {
            loaded_from_cache,
            configs_applied,
            segments_compiled,
        })
    }

    /// Run the optimizer search and return configs + stored results.
    fn run_optimizer_to_configs(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
        per_segment_budget: std::time::Duration,
    ) -> Result<
        (
            std::collections::HashMap<String, nn_dsl::PeepholeConfig>,
            Vec<(String, nn_dsl::OptimizationResult)>,
        ),
        CompiledKokoroError,
    > {
        let results =
            self.segment_optimizer_search(input_ids, style, speed, cache, per_segment_budget)?;
        let stored: Vec<(String, nn_dsl::OptimizationResult)> = results
            .iter()
            .map(|r| (r.segment_name.clone(), r.optimization.clone()))
            .collect();
        let configs = results
            .into_iter()
            .map(|r| (r.segment_name, r.optimization.config))
            .collect();
        Ok((configs, stored))
    }

    /// Run the optimizer search, convert to configs, and save to `path`.
    fn run_optimizer_and_save(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
        per_segment_budget: std::time::Duration,
        path: &Path,
    ) -> Result<
        (
            std::collections::HashMap<String, nn_dsl::PeepholeConfig>,
            Vec<(String, nn_dsl::OptimizationResult)>,
        ),
        CompiledKokoroError,
    > {
        let (configs, stored) =
            self.run_optimizer_to_configs(input_ids, style, speed, cache, per_segment_budget)?;
        super::save_peephole_configs(&configs, path)?;
        Ok((configs, stored))
    }
}

// -- Per-segment precompile functions extracted to compiled_kokoro_precompile_segments.rs --

#[path = "compiled_kokoro_precompile_segments.rs"]
mod segments;
use segments::{
    precompile_segment_f0, precompile_segment_generator, precompile_segment_plbert,
    precompile_segment_prosody, precompile_segment_regulate, precompile_segment_sinegen_post,
    precompile_segment_sinegen_pre, precompile_segment_text,
};

/// Trace + compile + export MSL for a single segment at one shape.
///
/// Returns `(files_written, bytes_written)`.
pub(super) fn export_plan_msl(
    graph: &ComputationGraph,
    segment_name: &'static str,
    shape_key: usize,
    output_dir: &Path,
) -> Result<(usize, usize), CompiledKokoroError> {
    let plan = compile_trace_to_plan_with_fusion(graph).map_err(|e| {
        CompiledKokoroError::PrecompileCompileFailed {
            segment: segment_name,
            shape_key,
            source: Box::new(e),
        }
    })?;

    let sources = plan.generate_msl(ScalarType::F32).map_err(|e| {
        CompiledKokoroError::PrecompileMslCodegenFailed {
            segment: segment_name,
            shape_key,
            source: Box::new(e),
        }
    })?;

    let mut files = 0;
    let mut bytes = 0;
    for src in &sources {
        let filename = format!(
            "{segment_name}_{shape_key}_{:03}_{}.metal",
            src.step_index, src.kernel_name
        );
        let path = output_dir.join(&filename);
        std::fs::write(&path, &src.msl)?;
        files += 1;
        bytes += src.msl.len();
    }

    Ok((files, bytes))
}

#[cfg(test)]
#[path = "compiled_kokoro_precompile_tests.rs"]
mod tests;
