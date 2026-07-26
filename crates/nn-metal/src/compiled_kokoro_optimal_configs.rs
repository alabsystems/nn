// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-segment optimal [`PeepholeConfig`] persistence for Kokoro.
//!
//! After the self-optimizing compiler finds optimal [`PeepholeConfig`] per
//! Kokoro segment, this module persists the results as JSON so optimized
//! configs survive across runs without re-running the exhaustive search.
//!
//! # Format
//!
//! ```json
//! {
//!   "version": "0.1.0",
//!   "peephole_field_count": 16,
//!   "segments": {
//!     "plbert": {
//!       "segment_name": "plbert",
//!       "peephole_config": { "norm_activ_conv1d": true, ... },
//!       "dispatch_count": 18,
//!       "estimated_cost_us": 102.3,
//!       "search_budget_ms": 5000,
//!       "configs_evaluated": 32768
//!     },
//!     ...
//!   }
//! }
//! ```
//!
//! Part of #3828 (Self-Optimizing ML Compiler).

use std::collections::HashMap;
use std::path::Path;

use nn_dsl::PeepholeConfig;

use super::CompiledKokoroError;

/// Count the number of boolean fields in [`PeepholeConfig`] by serializing
/// a default instance and counting JSON object keys.
///
/// This avoids depending on `PEEPHOLE_FIELD_COUNT` from nn-dsl (which is
/// `pub(crate)`). The count is used as a staleness signal: if the field
/// count changes between saves and loads, the search space has changed.
fn current_peephole_field_count() -> u32 {
    let config = PeepholeConfig::default();
    let value = serde_json::to_value(&config).expect("PeepholeConfig should serialize");
    value.as_object().map_or(0, |m| m.len() as u32)
}

/// Persisted per-segment optimal [`PeepholeConfig`] results for Kokoro.
///
/// Stores the best config found for each segment, along with metadata
/// for cache invalidation (version, field count) and debugging
/// (dispatch count, cost estimate, search budget).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KokoroOptimalConfigs {
    /// nn crate version at the time configs were saved.
    /// Used for cache invalidation: if the nn version changes,
    /// persisted configs may be stale (new optimizations, changed IR).
    pub version: String,
    /// Number of [`PeepholeConfig`] boolean fields when configs were saved.
    /// If this differs from the current field count, the search space has
    /// changed and configs should be re-optimized.
    pub peephole_field_count: u32,
    /// Per-segment optimal configuration. Keys are segment names:
    /// `"plbert"`, `"text"`, `"prosody"`, `"f0"`, `"generator"`,
    /// `"regulate"`, `"sinegen_pre"`, `"sinegen_post"`.
    pub segments: HashMap<String, SegmentOptimalConfig>,
}

/// Optimal configuration for a single Kokoro segment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SegmentOptimalConfig {
    /// Segment name (matches the key in [`KokoroOptimalConfigs::segments`]).
    pub segment_name: String,
    /// The [`PeepholeConfig`] that produced the lowest dispatch count.
    pub peephole_config: PeepholeConfig,
    /// Dispatch count achieved with this config.
    pub dispatch_count: usize,
    /// Estimated cost in microseconds from the cost model.
    pub estimated_cost_us: f64,
    /// Time budget in milliseconds used for the search.
    pub search_budget_ms: u64,
    /// Number of configs evaluated during the search.
    pub configs_evaluated: usize,
}

impl KokoroOptimalConfigs {
    /// Create a new empty configs container with the current version.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            peephole_field_count: current_peephole_field_count(),
            segments: HashMap::new(),
        }
    }

    /// Create a new configs container with a specific version string.
    ///
    /// Useful for testing version invalidation.
    #[must_use]
    pub fn with_version(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            peephole_field_count: current_peephole_field_count(),
            segments: HashMap::new(),
        }
    }

    /// Returns `true` if these configs are valid for the given nn version.
    ///
    /// Configs are valid when:
    /// 1. The version matches the current nn crate version.
    /// 2. The peephole field count matches the current field count.
    ///
    /// When invalid, callers should fall back to default configs and
    /// optionally re-run the optimizer.
    #[must_use]
    pub fn is_valid(&self, nn_version: &str) -> bool {
        self.version == nn_version
            && self.peephole_field_count == current_peephole_field_count()
    }

    /// Returns `true` if these configs are valid for the current build.
    #[must_use]
    pub fn is_current(&self) -> bool {
        self.is_valid(env!("CARGO_PKG_VERSION"))
    }

    /// Look up the optimal [`PeepholeConfig`] for a segment by name.
    ///
    /// Returns `None` if the segment has no persisted config.
    #[must_use]
    pub fn get_config(&self, segment_name: &str) -> Option<&PeepholeConfig> {
        self.segments.get(segment_name).map(|s| &s.peephole_config)
    }

    /// Look up the optimal [`PeepholeConfig`] for a segment, falling back
    /// to the default config if not found.
    #[must_use]
    pub fn get_config_or_default(&self, segment_name: &str) -> PeepholeConfig {
        self.segments
            .get(segment_name)
            .map(|s| s.peephole_config.clone())
            .unwrap_or_default()
    }

    /// Insert or update the optimal config for a segment.
    pub fn insert(&mut self, config: SegmentOptimalConfig) {
        self.segments.insert(config.segment_name.clone(), config);
    }

    /// Convert to a per-segment [`PeepholeConfig`] map suitable for
    /// [`CompiledKokoro::with_peephole_configs()`](super::CompiledKokoro::with_peephole_configs).
    #[must_use]
    pub fn to_peephole_map(&self) -> HashMap<String, PeepholeConfig> {
        self.segments
            .iter()
            .map(|(name, cfg)| (name.clone(), cfg.peephole_config.clone()))
            .collect()
    }

    /// Number of segments with persisted configs.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

impl Default for KokoroOptimalConfigs {
    fn default() -> Self {
        Self::new()
    }
}

/// Save [`KokoroOptimalConfigs`] to a JSON file.
///
/// Uses pretty-printed JSON for debuggability and diffing. Creates or
/// overwrites the file at `path`.
///
/// # Errors
///
/// Returns [`CompiledKokoroError::ConfigLoad`] on I/O or serialization failure.
pub fn save_optimal_configs(
    configs: &KokoroOptimalConfigs,
    path: &Path,
) -> Result<(), CompiledKokoroError> {
    let json = serde_json::to_string_pretty(configs)
        .map_err(|e| CompiledKokoroError::ConfigLoad(format!("serialize optimal configs: {e}")))?;
    std::fs::write(path, json).map_err(|e| {
        CompiledKokoroError::ConfigLoad(format!("write {}: {e}", path.display()))
    })?;
    Ok(())
}

/// Load [`KokoroOptimalConfigs`] from a JSON file.
///
/// # Errors
///
/// Returns [`CompiledKokoroError::ConfigLoad`] on I/O or deserialization failure.
pub fn load_optimal_configs(
    path: &Path,
) -> Result<KokoroOptimalConfigs, CompiledKokoroError> {
    let data = std::fs::read_to_string(path).map_err(|e| {
        CompiledKokoroError::ConfigLoad(format!("read {}: {e}", path.display()))
    })?;
    let configs: KokoroOptimalConfigs = serde_json::from_str(&data).map_err(|e| {
        CompiledKokoroError::ConfigLoad(format!("parse {}: {e}", path.display()))
    })?;
    Ok(configs)
}

/// Load [`KokoroOptimalConfigs`] from a JSON file, returning `None` if the
/// file does not exist (instead of an error).
///
/// Other errors (permission denied, invalid JSON) are still propagated.
///
/// # Errors
///
/// Returns [`CompiledKokoroError::ConfigLoad`] on non-`NotFound` I/O errors
/// or deserialization failure.
pub fn load_optimal_configs_if_exists(
    path: &Path,
) -> Result<Option<KokoroOptimalConfigs>, CompiledKokoroError> {
    match std::fs::read_to_string(path) {
        Ok(data) => {
            let configs: KokoroOptimalConfigs = serde_json::from_str(&data).map_err(|e| {
                CompiledKokoroError::ConfigLoad(format!("parse {}: {e}", path.display()))
            })?;
            Ok(Some(configs))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(CompiledKokoroError::ConfigLoad(format!(
            "read {}: {e}",
            path.display()
        ))),
    }
}
