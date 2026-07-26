// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Persistence helpers for [`PeepholeConfig`] and [`OptimizationResult`].
//!
//! Saves and loads optimal peephole configurations as JSON so the
//! self-optimizing compiler can skip the exhaustive search on subsequent
//! builds. Gated behind the `plan-serde` feature flag.
//!
//! Part of #3828 (Self-Optimizing ML Compiler).

use std::path::Path;

use crate::trace_compile::optimize_plan::PEEPHOLE_FIELD_COUNT;
use crate::trace_compile::{OptimizationResult, PeepholeConfig};

/// Errors from peephole config persistence operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PeepholeConfigPersistError {
    /// I/O error reading or writing the config file.
    #[error("peephole config I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization or deserialization error.
    #[error("peephole config JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Current version of the [`PeepholeConfig`] serialization format.
///
/// Bump this when the serialization format changes in a way that requires
/// migration beyond simple `#[serde(default)]` field additions.
pub const PEEPHOLE_CONFIG_VERSION: u32 = 1;

/// Versioned wrapper around [`PeepholeConfig`] for safe config migration.
///
/// Tracks the format version and field count at the time the config was
/// saved. When loading a persisted config, callers can check
/// [`is_current_version`](Self::is_current_version) and
/// [`needs_migration`](Self::needs_migration) to determine whether the
/// config was produced by a different (older or newer) version of the
/// optimizer.
///
/// The inner [`PeepholeConfig`] still uses `#[serde(default)]` for
/// backward compatibility -- missing fields default to `true`. The version
/// envelope adds *awareness* of staleness so callers can re-optimize
/// when new passes are available.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PeepholeConfigVersioned {
    /// Format version (see [`PEEPHOLE_CONFIG_VERSION`]).
    pub version: u32,
    /// Number of boolean fields in [`PeepholeConfig`] when this was saved.
    pub field_count: u32,
    /// The actual peephole configuration.
    pub config: PeepholeConfig,
}

impl PeepholeConfigVersioned {
    /// Wrap a [`PeepholeConfig`] with the current version and field count.
    #[must_use]
    pub fn new(config: PeepholeConfig) -> Self {
        Self {
            version: PEEPHOLE_CONFIG_VERSION,
            field_count: PEEPHOLE_FIELD_COUNT,
            config,
        }
    }

    /// Returns `true` if the version AND field count match the current build.
    #[must_use]
    pub fn is_current_version(&self) -> bool {
        self.version == PEEPHOLE_CONFIG_VERSION && self.field_count == PEEPHOLE_FIELD_COUNT
    }

    /// Returns `true` if the field count differs from the current build.
    ///
    /// When new peephole passes are added, old persisted configs will have
    /// a lower `field_count`. Callers should re-run the optimizer to explore
    /// the expanded search space.
    #[must_use]
    pub fn needs_migration(&self) -> bool {
        self.field_count != PEEPHOLE_FIELD_COUNT
    }
}

/// Save a [`PeepholeConfig`] to a JSON file at `path`.
///
/// Creates or overwrites the file. Uses pretty-printed JSON for
/// debuggability and diffing.
///
/// # Errors
///
/// Returns [`PeepholeConfigPersistError`] on I/O or serialization failure.
pub fn save_peephole_config(
    config: &PeepholeConfig,
    path: impl AsRef<Path>,
) -> Result<(), PeepholeConfigPersistError> {
    let json = serde_json::to_string_pretty(config)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load a [`PeepholeConfig`] from a JSON file at `path`.
///
/// # Errors
///
/// Returns [`PeepholeConfigPersistError`] on I/O or deserialization failure.
pub fn load_peephole_config(
    path: impl AsRef<Path>,
) -> Result<PeepholeConfig, PeepholeConfigPersistError> {
    let data = std::fs::read_to_string(path)?;
    let config: PeepholeConfig = serde_json::from_str(&data)?;
    Ok(config)
}

/// Summary of an [`OptimizationResult`] suitable for persistence.
///
/// Contains the optimal config, dispatch count, and search metadata
/// but NOT the full `CompiledPlan` (which is large and contains IR).
/// Use [`save_plan`](crate::compiled_plan_io::save_plan) to persist
/// the full plan separately.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizationResultSummary {
    /// The [`PeepholeConfig`] that produced the best plan.
    pub config: PeepholeConfig,
    /// Dispatch count of the best plan.
    pub dispatch_count: usize,
    /// Number of configurations explored during the search.
    pub configs_explored: usize,
    /// Baseline dispatch count (all passes enabled).
    pub baseline_dispatch_count: usize,
    /// Estimated cost of the best plan (nanoseconds).
    pub best_cost_ns: f64,
    /// Estimated cost of the baseline plan (nanoseconds).
    pub baseline_cost_ns: f64,
}

impl From<&OptimizationResult> for OptimizationResultSummary {
    fn from(result: &OptimizationResult) -> Self {
        Self {
            config: result.config.clone(),
            dispatch_count: result.dispatch_count,
            configs_explored: result.configs_explored,
            baseline_dispatch_count: result.baseline_dispatch_count,
            best_cost_ns: result.best_cost_ns,
            baseline_cost_ns: result.baseline_cost_ns,
        }
    }
}

/// Save a versioned [`PeepholeConfig`] to a JSON file at `path`.
///
/// Wraps the config in a [`PeepholeConfigVersioned`] envelope with the
/// current version and field count, then writes pretty-printed JSON.
///
/// # Errors
///
/// Returns [`PeepholeConfigPersistError`] on I/O or serialization failure.
pub fn save_versioned(
    config: &PeepholeConfig,
    path: impl AsRef<Path>,
) -> Result<(), PeepholeConfigPersistError> {
    let versioned = PeepholeConfigVersioned::new(config.clone());
    let json = serde_json::to_string_pretty(&versioned)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load a versioned [`PeepholeConfig`] from a JSON file at `path`.
///
/// Returns the full [`PeepholeConfigVersioned`] wrapper so the caller can
/// check [`is_current_version`](PeepholeConfigVersioned::is_current_version)
/// and [`needs_migration`](PeepholeConfigVersioned::needs_migration).
///
/// # Errors
///
/// Returns [`PeepholeConfigPersistError`] on I/O or deserialization failure.
pub fn load_versioned(
    path: impl AsRef<Path>,
) -> Result<PeepholeConfigVersioned, PeepholeConfigPersistError> {
    let data = std::fs::read_to_string(path)?;
    let versioned: PeepholeConfigVersioned = serde_json::from_str(&data)?;
    Ok(versioned)
}

/// Save an [`OptimizationResult`] summary to a JSON file at `path`.
///
/// Writes the optimal config, dispatch count, and search metadata.
/// Excludes the full `CompiledPlan` to keep the file small and
/// human-readable.
///
/// # Errors
///
/// Returns [`PeepholeConfigPersistError`] on I/O or serialization failure.
pub fn save_optimization_result_summary(
    result: &OptimizationResult,
    path: impl AsRef<Path>,
) -> Result<(), PeepholeConfigPersistError> {
    let summary = OptimizationResultSummary::from(result);
    let json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load an [`OptimizationResultSummary`] from a JSON file at `path`.
///
/// # Errors
///
/// Returns [`PeepholeConfigPersistError`] on I/O or deserialization failure.
pub fn load_optimization_result_summary(
    path: impl AsRef<Path>,
) -> Result<OptimizationResultSummary, PeepholeConfigPersistError> {
    let data = std::fs::read_to_string(path)?;
    let summary: OptimizationResultSummary = serde_json::from_str(&data)?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "nn_peephole_persist_test_{}_{name}",
            std::process::id()
        ))
    }

    #[test]
    fn test_round_trip_default_config() {
        let config = PeepholeConfig::default();
        let path = temp_path("default.json");

        save_peephole_config(&config, &path).expect("save should succeed");
        let restored = load_peephole_config(&path).expect("load should succeed");

        assert_eq!(config, restored);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_round_trip_all_true() {
        let config = PeepholeConfig {
            norm_activ_conv1d: true,
            fused_resblock: true,
            linear_activation: true,
            add_layer_norm: true,
            norm_linear: true,
            attention_transpose: true,
            flip_lstm: true,
            bilstm_cat: true,
            batched_linear_projection: true,
            channels_first_layer_norm: true,
            silu_mul: true,
            auto_fuse_elementwise: true,
            add_norm_linear: true,
            fuse_adain_snake: true,
            fuse_upsample_conv1d: true,
            fuse_instance_norm_mul_add: true,
            fuse_conv1d_activation: true,
            fuse_snake_instance_norm: true,
            fuse_conv1d_snake_norm: true,
            fuse_conv1d_snake_norm_resblock: true,
            fuse_add_instance_norm_conv1x1: true,
            fuse_conv_transpose1d_activation: true,
            norm_activ_conv_transpose1d: true,
            fuse_instance_norm_conv1d: true,
            fuse_conv1d_instance_norm: true,
            fuse_linear_layer_norm: true,
            fuse_resblock_chain: true,
            fuse_activation_conv1d: true,
        };
        let path = temp_path("all_true.json");

        save_peephole_config(&config, &path).expect("save should succeed");
        let restored = load_peephole_config(&path).expect("load should succeed");

        assert_eq!(config, restored);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_round_trip_all_false() {
        let config = PeepholeConfig {
            norm_activ_conv1d: false,
            fused_resblock: false,
            linear_activation: false,
            add_layer_norm: false,
            norm_linear: false,
            attention_transpose: false,
            flip_lstm: false,
            bilstm_cat: false,
            batched_linear_projection: false,
            channels_first_layer_norm: false,
            silu_mul: false,
            auto_fuse_elementwise: false,
            add_norm_linear: false,
            fuse_adain_snake: false,
            fuse_upsample_conv1d: false,
            fuse_instance_norm_mul_add: false,
            fuse_conv1d_activation: false,
            fuse_snake_instance_norm: false,
            fuse_conv1d_snake_norm: false,
            fuse_conv1d_snake_norm_resblock: false,
            fuse_add_instance_norm_conv1x1: false,
            fuse_conv_transpose1d_activation: false,
            norm_activ_conv_transpose1d: false,
            fuse_instance_norm_conv1d: false,
            fuse_conv1d_instance_norm: false,
            fuse_linear_layer_norm: false,
            fuse_resblock_chain: false,
            fuse_activation_conv1d: false,
        };
        let path = temp_path("all_false.json");

        save_peephole_config(&config, &path).expect("save should succeed");
        let restored = load_peephole_config(&path).expect("load should succeed");

        assert_eq!(config, restored);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_round_trip_mixed() {
        let config = PeepholeConfig {
            norm_activ_conv1d: true,
            fused_resblock: false,
            linear_activation: true,
            add_layer_norm: false,
            norm_linear: true,
            attention_transpose: false,
            flip_lstm: true,
            bilstm_cat: false,
            batched_linear_projection: true,
            channels_first_layer_norm: false,
            silu_mul: true,
            auto_fuse_elementwise: false,
            add_norm_linear: false,
            fuse_adain_snake: true,
            fuse_upsample_conv1d: true,
            fuse_instance_norm_mul_add: true,
            fuse_conv1d_activation: true,
            fuse_snake_instance_norm: true,
            fuse_conv1d_snake_norm: true,
            fuse_conv1d_snake_norm_resblock: true,
            fuse_add_instance_norm_conv1x1: true,
            fuse_conv_transpose1d_activation: true,
            norm_activ_conv_transpose1d: true,
            fuse_instance_norm_conv1d: true,
            fuse_conv1d_instance_norm: true,
            fuse_linear_layer_norm: true,
            fuse_resblock_chain: true,
            fuse_activation_conv1d: true,
        };
        let path = temp_path("mixed.json");

        save_peephole_config(&config, &path).expect("save should succeed");
        let restored = load_peephole_config(&path).expect("load should succeed");

        assert_eq!(config, restored);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_from_json_string() {
        let json = r#"{
            "norm_activ_conv1d": true,
            "fused_resblock": false,
            "linear_activation": true,
            "add_layer_norm": false,
            "norm_linear": true,
            "attention_transpose": false,
            "flip_lstm": true,
            "bilstm_cat": false,
            "batched_linear_projection": true,
            "channels_first_layer_norm": false,
            "silu_mul": true,
            "auto_fuse_elementwise": false,
            "add_norm_linear": true,
            "fuse_adain_snake": false
        }"#;

        let config: PeepholeConfig = serde_json::from_str(json).expect("should parse valid JSON");

        assert!(config.norm_activ_conv1d);
        assert!(!config.fused_resblock);
        assert!(config.linear_activation);
        assert!(!config.add_layer_norm);
        assert!(config.norm_linear);
        assert!(!config.attention_transpose);
        assert!(config.flip_lstm);
        assert!(!config.bilstm_cat);
        assert!(config.batched_linear_projection);
        assert!(!config.channels_first_layer_norm);
        assert!(config.silu_mul);
        assert!(!config.auto_fuse_elementwise);
        assert!(config.add_norm_linear);
        assert!(!config.fuse_adain_snake);
    }

    #[test]
    fn test_optimization_result_summary_round_trip() {
        let summary = OptimizationResultSummary {
            config: PeepholeConfig::default(),
            dispatch_count: 180,
            configs_explored: 4096,
            baseline_dispatch_count: 201,
            best_cost_ns: 9000.0,
            baseline_cost_ns: 10000.0,
        };
        let path = temp_path("summary.json");

        let json = serde_json::to_string_pretty(&summary).expect("serialize should succeed");
        std::fs::write(&path, &json).expect("write should succeed");

        let restored = load_optimization_result_summary(&path).expect("load should succeed");

        assert_eq!(restored.config, summary.config);
        assert_eq!(restored.dispatch_count, summary.dispatch_count);
        assert_eq!(restored.configs_explored, summary.configs_explored);
        assert_eq!(
            restored.baseline_dispatch_count,
            summary.baseline_dispatch_count
        );
        assert!((restored.best_cost_ns - summary.best_cost_ns).abs() < f64::EPSILON);
        assert!((restored.baseline_cost_ns - summary.baseline_cost_ns).abs() < f64::EPSILON);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_optimization_result_summary_writes_json() {
        use crate::trace_compile::{CompiledPlan, OptimizationResult};

        let result = OptimizationResult {
            plan: CompiledPlan {
                steps: vec![],
                input_shapes: vec![],
                output_step: 0,
                weight_names: vec![],
            },
            config: PeepholeConfig {
                norm_activ_conv1d: true,
                fused_resblock: false,
                linear_activation: true,
                add_layer_norm: true,
                norm_linear: false,
                attention_transpose: true,
                flip_lstm: false,
                bilstm_cat: true,
                batched_linear_projection: true,
                channels_first_layer_norm: false,
                silu_mul: true,
                auto_fuse_elementwise: true,
                add_norm_linear: true,
                fuse_adain_snake: true,
                fuse_upsample_conv1d: true,
                fuse_instance_norm_mul_add: true,
                fuse_conv1d_activation: true,
                fuse_snake_instance_norm: true,
                fuse_conv1d_snake_norm: true,
                fuse_conv1d_snake_norm_resblock: true,
                fuse_add_instance_norm_conv1x1: true,
                fuse_conv_transpose1d_activation: true,
                norm_activ_conv_transpose1d: true,
                fuse_instance_norm_conv1d: true,
                fuse_conv1d_instance_norm: true,
                fuse_linear_layer_norm: true,
                fuse_resblock_chain: true,
                fuse_activation_conv1d: true,
            },
            dispatch_count: 150,
            configs_explored: 2048,
            baseline_dispatch_count: 200,
            best_cost_ns: 8500.0,
            baseline_cost_ns: 11000.0,
        };
        let path = temp_path("opt_summary.json");

        save_optimization_result_summary(&result, &path).expect("save should succeed");

        // Verify the saved file can be parsed back.
        let restored = load_optimization_result_summary(&path).expect("load should succeed");
        assert_eq!(restored.dispatch_count, 150);
        assert_eq!(restored.configs_explored, 2048);
        assert_eq!(restored.baseline_dispatch_count, 200);
        assert!(restored.config.norm_activ_conv1d);
        assert!(!restored.config.fused_resblock);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_nonexistent_file_returns_error() {
        let path = temp_path("nonexistent.json");
        let result = load_peephole_config(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_invalid_json_returns_error() {
        let path = temp_path("invalid.json");
        std::fs::write(&path, "not valid json {{{").expect("write should succeed");

        let result = load_peephole_config(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    /// Comprehensive roundtrip that explicitly checks all 14 fields by name.
    #[test]
    fn test_roundtrip_all_15_fields_explicit() {
        let config = PeepholeConfig::default();
        let json = serde_json::to_string_pretty(&config).expect("serialize default");
        let restored: PeepholeConfig =
            serde_json::from_str(&json).expect("deserialize default roundtrip");

        // Verify every field individually.
        assert_eq!(config.norm_activ_conv1d, restored.norm_activ_conv1d);
        assert_eq!(config.fused_resblock, restored.fused_resblock);
        assert_eq!(config.linear_activation, restored.linear_activation);
        assert_eq!(config.add_layer_norm, restored.add_layer_norm);
        assert_eq!(config.norm_linear, restored.norm_linear);
        assert_eq!(config.attention_transpose, restored.attention_transpose);
        assert_eq!(config.flip_lstm, restored.flip_lstm);
        assert_eq!(
            config.batched_linear_projection,
            restored.batched_linear_projection
        );
        assert_eq!(
            config.channels_first_layer_norm,
            restored.channels_first_layer_norm
        );
        assert_eq!(config.silu_mul, restored.silu_mul);
        assert_eq!(config.auto_fuse_elementwise, restored.auto_fuse_elementwise);
        assert_eq!(config.bilstm_cat, restored.bilstm_cat);
        assert_eq!(config.add_norm_linear, restored.add_norm_linear);
        assert_eq!(config.fuse_adain_snake, restored.fuse_adain_snake);
        assert_eq!(config.fuse_upsample_conv1d, restored.fuse_upsample_conv1d);
        assert_eq!(
            config.fuse_instance_norm_mul_add,
            restored.fuse_instance_norm_mul_add
        );

        // Also verify all-false roundtrip field by field.
        let all_false = PeepholeConfig {
            norm_activ_conv1d: false,
            fused_resblock: false,
            linear_activation: false,
            add_layer_norm: false,
            norm_linear: false,
            attention_transpose: false,
            flip_lstm: false,
            bilstm_cat: false,
            batched_linear_projection: false,
            channels_first_layer_norm: false,
            silu_mul: false,
            auto_fuse_elementwise: false,
            add_norm_linear: false,
            fuse_adain_snake: false,
            fuse_upsample_conv1d: false,
            fuse_instance_norm_mul_add: false,
            fuse_conv1d_activation: false,
            fuse_snake_instance_norm: false,
            fuse_conv1d_snake_norm: false,
            fuse_conv1d_snake_norm_resblock: false,
            fuse_add_instance_norm_conv1x1: false,
            fuse_conv_transpose1d_activation: false,
            norm_activ_conv_transpose1d: false,
            fuse_instance_norm_conv1d: false,
            fuse_conv1d_instance_norm: false,
            fuse_linear_layer_norm: false,
            fuse_resblock_chain: false,
            fuse_activation_conv1d: false,
        };
        let json_false = serde_json::to_string_pretty(&all_false).expect("serialize all-false");
        let restored_false: PeepholeConfig =
            serde_json::from_str(&json_false).expect("deserialize all-false roundtrip");
        assert!(!restored_false.norm_activ_conv1d);
        assert!(!restored_false.fused_resblock);
        assert!(!restored_false.linear_activation);
        assert!(!restored_false.add_layer_norm);
        assert!(!restored_false.norm_linear);
        assert!(!restored_false.attention_transpose);
        assert!(!restored_false.flip_lstm);
        assert!(!restored_false.bilstm_cat);
        assert!(!restored_false.batched_linear_projection);
        assert!(!restored_false.channels_first_layer_norm);
        assert!(!restored_false.silu_mul);
        assert!(!restored_false.auto_fuse_elementwise);
        assert!(!restored_false.add_norm_linear);
        assert!(!restored_false.fuse_adain_snake);
        assert!(!restored_false.fuse_upsample_conv1d);
        assert!(!restored_false.fuse_instance_norm_mul_add);

        // Selective fields: only fuse_adain_snake and silu_mul enabled.
        let selective = PeepholeConfig {
            norm_activ_conv1d: false,
            fused_resblock: false,
            linear_activation: false,
            add_layer_norm: false,
            norm_linear: false,
            attention_transpose: false,
            flip_lstm: false,
            bilstm_cat: false,
            batched_linear_projection: false,
            channels_first_layer_norm: false,
            silu_mul: true,
            auto_fuse_elementwise: false,
            add_norm_linear: false,
            fuse_adain_snake: true,
            fuse_upsample_conv1d: true,
            fuse_instance_norm_mul_add: true,
            fuse_conv1d_activation: true,
            fuse_snake_instance_norm: true,
            fuse_conv1d_snake_norm: true,
            fuse_conv1d_snake_norm_resblock: true,
            fuse_add_instance_norm_conv1x1: true,
            fuse_conv_transpose1d_activation: true,
            norm_activ_conv_transpose1d: true,
            fuse_instance_norm_conv1d: true,
            fuse_conv1d_instance_norm: true,
            fuse_linear_layer_norm: true,
            fuse_resblock_chain: true,
            fuse_activation_conv1d: true,
        };
        let json_sel = serde_json::to_string_pretty(&selective).expect("serialize selective");
        let restored_sel: PeepholeConfig =
            serde_json::from_str(&json_sel).expect("deserialize selective roundtrip");
        assert!(!restored_sel.norm_activ_conv1d);
        assert!(!restored_sel.fused_resblock);
        assert!(restored_sel.silu_mul);
        assert!(restored_sel.fuse_adain_snake);
        assert_eq!(selective, restored_sel);
    }

    /// Backward compatibility: deserialize a 13-field JSON (missing `fuse_adain_snake`).
    ///
    /// Old configs saved before the `fuse_adain_snake` field was added should
    /// still load, with the missing field defaulting to `true` (from `Default`).
    #[test]
    fn test_backward_compat_13_field_json_missing_fuse_adain_snake() {
        let json_13 = r#"{
            "norm_activ_conv1d": true,
            "fused_resblock": false,
            "linear_activation": true,
            "add_layer_norm": false,
            "norm_linear": true,
            "attention_transpose": false,
            "flip_lstm": true,
            "bilstm_cat": false,
            "batched_linear_projection": true,
            "channels_first_layer_norm": false,
            "silu_mul": true,
            "auto_fuse_elementwise": false,
            "add_norm_linear": true
        }"#;

        let config: PeepholeConfig =
            serde_json::from_str(json_13).expect("should parse 13-field JSON with missing field");

        // Explicitly specified fields should match.
        assert!(config.norm_activ_conv1d);
        assert!(!config.fused_resblock);
        assert!(config.linear_activation);
        assert!(!config.add_layer_norm);
        assert!(config.norm_linear);
        assert!(!config.attention_transpose);
        assert!(config.flip_lstm);
        assert!(!config.bilstm_cat);
        assert!(config.batched_linear_projection);
        assert!(!config.channels_first_layer_norm);
        assert!(config.silu_mul);
        assert!(!config.auto_fuse_elementwise);
        assert!(config.add_norm_linear);

        // Missing field defaults to true (from Default impl).
        assert!(
            config.fuse_adain_snake,
            "missing fuse_adain_snake should default to true"
        );
    }

    /// Backward compatibility: deserialize a JSON with NO fields (empty object).
    ///
    /// All fields should default to `true` from the `Default` impl.
    #[test]
    fn test_backward_compat_empty_json_defaults_all_true() {
        let json_empty = "{}";

        let config: PeepholeConfig =
            serde_json::from_str(json_empty).expect("should parse empty JSON object");

        assert!(config.norm_activ_conv1d);
        assert!(config.fused_resblock);
        assert!(config.linear_activation);
        assert!(config.add_layer_norm);
        assert!(config.norm_linear);
        assert!(config.attention_transpose);
        assert!(config.flip_lstm);
        assert!(config.bilstm_cat);
        assert!(config.batched_linear_projection);
        assert!(config.channels_first_layer_norm);
        assert!(config.silu_mul);
        assert!(config.auto_fuse_elementwise);
        assert!(config.add_norm_linear);
        assert!(config.fuse_adain_snake);

        assert_eq!(config, PeepholeConfig::default());
    }

    /// Verify that the serialized JSON contains all 19 field names.
    #[test]
    fn test_serialized_json_contains_all_19_fields() {
        let config = PeepholeConfig::default();
        let json = serde_json::to_string_pretty(&config).expect("serialize");

        let expected_fields = [
            "norm_activ_conv1d",
            "fused_resblock",
            "linear_activation",
            "add_layer_norm",
            "norm_linear",
            "attention_transpose",
            "flip_lstm",
            "batched_linear_projection",
            "channels_first_layer_norm",
            "silu_mul",
            "auto_fuse_elementwise",
            "bilstm_cat",
            "add_norm_linear",
            "fuse_adain_snake",
            "fuse_upsample_conv1d",
            "fuse_instance_norm_mul_add",
            "fuse_conv1d_activation",
            "fuse_snake_instance_norm",
            "fuse_conv1d_snake_norm",
        ];
        for field in &expected_fields {
            assert!(
                json.contains(field),
                "serialized JSON should contain field '{field}'"
            );
        }
        assert_eq!(expected_fields.len(), 19, "should check all 19 fields");
    }

    // -- PeepholeConfigVersioned tests ----------------------------------------

    #[test]
    fn test_versioned_roundtrip() {
        let config = PeepholeConfig::default();
        let versioned = PeepholeConfigVersioned::new(config.clone());

        let json = serde_json::to_string_pretty(&versioned).expect("serialize versioned");
        let restored: PeepholeConfigVersioned =
            serde_json::from_str(&json).expect("deserialize versioned roundtrip");

        assert_eq!(restored.version, PEEPHOLE_CONFIG_VERSION);
        assert_eq!(restored.field_count, PEEPHOLE_FIELD_COUNT);
        assert_eq!(restored.config, config);
        assert_eq!(restored, versioned);
    }

    #[test]
    fn test_versioned_is_current_version() {
        let versioned = PeepholeConfigVersioned::new(PeepholeConfig::default());
        assert!(versioned.is_current_version());
        assert!(!versioned.needs_migration());
    }

    #[test]
    fn test_versioned_needs_migration_fewer_fields() {
        // Simulate loading a config saved when there were only 13 fields.
        let old_versioned = PeepholeConfigVersioned {
            version: 1,
            field_count: 13,
            config: PeepholeConfig::default(),
        };
        assert!(
            old_versioned.needs_migration(),
            "config with 13 fields should need migration when current is 15"
        );
        assert!(
            !old_versioned.is_current_version(),
            "config with 13 fields should not be current version"
        );
    }

    #[test]
    fn test_versioned_needs_migration_more_fields() {
        // Simulate loading a config from a future version with more fields.
        let future_versioned = PeepholeConfigVersioned {
            version: 1,
            field_count: 20,
            config: PeepholeConfig::default(),
        };
        assert!(
            future_versioned.needs_migration(),
            "config with 20 fields should need migration when current is 15"
        );
    }

    #[test]
    fn test_versioned_old_version_not_current() {
        // Even with matching field count, a different version is not current.
        let old_version = PeepholeConfigVersioned {
            version: 0,
            field_count: PEEPHOLE_FIELD_COUNT,
            config: PeepholeConfig::default(),
        };
        assert!(
            !old_version.is_current_version(),
            "version 0 should not be current even with matching field count"
        );
        // But field count matches, so no migration needed.
        assert!(
            !old_version.needs_migration(),
            "matching field count should not need migration"
        );
    }

    /// Backward compatibility: deserialize a versioned JSON with fewer fields.
    ///
    /// The inner `PeepholeConfig` uses `#[serde(default)]`, so missing
    /// config fields default to `true`. The version envelope correctly
    /// reports that migration is needed.
    #[test]
    fn test_versioned_backward_compat_fewer_config_fields() {
        // JSON with version envelope but only 13 config fields.
        let json = r#"{
            "version": 1,
            "field_count": 13,
            "config": {
                "norm_activ_conv1d": true,
                "fused_resblock": false,
                "linear_activation": true,
                "add_layer_norm": false,
                "norm_linear": true,
                "attention_transpose": false,
                "flip_lstm": true,
                "bilstm_cat": false,
                "batched_linear_projection": true,
                "channels_first_layer_norm": false,
                "silu_mul": true,
                "auto_fuse_elementwise": false,
                "add_norm_linear": true
            }
        }"#;

        let versioned: PeepholeConfigVersioned =
            serde_json::from_str(json).expect("should parse versioned JSON with 13-field config");

        assert_eq!(versioned.version, 1);
        assert_eq!(versioned.field_count, 13);
        assert!(versioned.needs_migration());
        assert!(!versioned.is_current_version());

        // Missing fields default to true.
        assert!(versioned.config.fuse_adain_snake);
        assert!(versioned.config.fuse_upsample_conv1d);
        assert!(versioned.config.fuse_instance_norm_mul_add);
        // Explicitly set fields are preserved.
        assert!(!versioned.config.fused_resblock);
        assert!(versioned.config.norm_activ_conv1d);
    }

    #[test]
    fn test_versioned_file_roundtrip() {
        let config = PeepholeConfig {
            norm_activ_conv1d: true,
            fused_resblock: false,
            linear_activation: true,
            add_layer_norm: false,
            norm_linear: true,
            attention_transpose: false,
            flip_lstm: true,
            bilstm_cat: false,
            batched_linear_projection: true,
            channels_first_layer_norm: false,
            silu_mul: true,
            auto_fuse_elementwise: false,
            add_norm_linear: true,
            fuse_adain_snake: true,
            fuse_upsample_conv1d: false,
            fuse_instance_norm_mul_add: false,
            fuse_conv1d_activation: false,
            fuse_snake_instance_norm: false,
            fuse_conv1d_snake_norm: false,
            fuse_conv1d_snake_norm_resblock: false,
            fuse_add_instance_norm_conv1x1: false,
            fuse_conv_transpose1d_activation: false,
            norm_activ_conv_transpose1d: false,
            fuse_instance_norm_conv1d: false,
            fuse_conv1d_instance_norm: false,
            fuse_linear_layer_norm: false,
            fuse_resblock_chain: false,
            fuse_activation_conv1d: false,
        };
        let versioned = PeepholeConfigVersioned::new(config);
        let path = temp_path("versioned.json");

        let json = serde_json::to_string_pretty(&versioned).expect("serialize versioned");
        std::fs::write(&path, &json).expect("write should succeed");

        let data = std::fs::read_to_string(&path).expect("read should succeed");
        let restored: PeepholeConfigVersioned =
            serde_json::from_str(&data).expect("deserialize from file");

        assert_eq!(restored, versioned);
        assert!(restored.is_current_version());
        assert!(!restored.needs_migration());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_versioned_new_sets_current_constants() {
        let versioned = PeepholeConfigVersioned::new(PeepholeConfig::default());
        assert_eq!(versioned.version, PEEPHOLE_CONFIG_VERSION);
        assert_eq!(versioned.field_count, PEEPHOLE_FIELD_COUNT);
        assert_eq!(versioned.field_count, 28);
    }

    // -- End-to-end persistence integration tests ------------------------------

    /// Full save_versioned → load_versioned roundtrip with default config.
    #[test]
    fn test_versioned_save_load_roundtrip_default() {
        let config = PeepholeConfig::default();
        let path = temp_path("versioned_rt_default.json");

        save_versioned(&config, &path).expect("save_versioned should succeed");
        let restored = load_versioned(&path).expect("load_versioned should succeed");

        assert_eq!(restored.config, config);
        assert_eq!(restored.version, PEEPHOLE_CONFIG_VERSION);
        assert_eq!(restored.field_count, PEEPHOLE_FIELD_COUNT);
        assert!(restored.is_current_version());
        assert!(!restored.needs_migration());

        let _ = std::fs::remove_file(&path);
    }

    /// save_versioned → load_versioned with non-default config (some passes disabled).
    #[test]
    fn test_versioned_save_load_roundtrip_non_default() {
        let config = PeepholeConfig {
            norm_activ_conv1d: true,
            fused_resblock: false,
            linear_activation: true,
            add_layer_norm: false,
            norm_linear: false,
            attention_transpose: true,
            flip_lstm: false,
            bilstm_cat: true,
            batched_linear_projection: false,
            channels_first_layer_norm: true,
            silu_mul: false,
            auto_fuse_elementwise: true,
            add_norm_linear: false,
            fuse_adain_snake: true,
            fuse_upsample_conv1d: false,
            fuse_instance_norm_mul_add: false,
            fuse_conv1d_activation: false,
            fuse_snake_instance_norm: false,
            fuse_conv1d_snake_norm: false,
            fuse_conv1d_snake_norm_resblock: false,
            fuse_add_instance_norm_conv1x1: false,
            fuse_conv_transpose1d_activation: false,
            norm_activ_conv_transpose1d: false,
            fuse_instance_norm_conv1d: false,
            fuse_conv1d_instance_norm: false,
            fuse_linear_layer_norm: false,
            fuse_resblock_chain: false,
            fuse_activation_conv1d: false,
        };
        let path = temp_path("versioned_rt_nondefault.json");

        save_versioned(&config, &path).expect("save_versioned should succeed");
        let restored = load_versioned(&path).expect("load_versioned should succeed");

        assert_eq!(restored.config, config);
        assert!(!restored.config.fused_resblock);
        assert!(restored.config.norm_activ_conv1d);
        assert!(!restored.config.norm_linear);
        assert!(!restored.config.silu_mul);
        assert!(restored.config.fuse_adain_snake);
        assert!(!restored.config.fuse_upsample_conv1d);
        assert!(!restored.config.fuse_instance_norm_mul_add);
        assert!(restored.is_current_version());

        let _ = std::fs::remove_file(&path);
    }

    /// Version mismatch detection — modify version after save, detect needs_migration.
    #[test]
    fn test_versioned_version_mismatch_detection() {
        let config = PeepholeConfig::default();
        let path = temp_path("versioned_mismatch.json");

        save_versioned(&config, &path).expect("save should succeed");

        let data = std::fs::read_to_string(&path).expect("read should succeed");
        let mut value: serde_json::Value = serde_json::from_str(&data).expect("parse JSON value");
        value["version"] = serde_json::Value::from(99u32);
        let modified_json = serde_json::to_string_pretty(&value).expect("re-serialize");
        std::fs::write(&path, &modified_json).expect("write modified JSON");

        let restored = load_versioned(&path).expect("load should succeed");
        assert_eq!(restored.version, 99);
        assert!(
            !restored.is_current_version(),
            "modified version should not be current"
        );
        assert!(!restored.needs_migration());
        assert_eq!(restored.config, config);

        let _ = std::fs::remove_file(&path);
    }

    /// Loading a config with fewer fields gracefully handles migration.
    #[test]
    fn test_versioned_fewer_fields_graceful_migration() {
        let path = temp_path("versioned_fewer_fields.json");

        let json = r#"{
            "version": 1,
            "field_count": 12,
            "config": {
                "norm_activ_conv1d": false,
                "fused_resblock": true,
                "linear_activation": false,
                "add_layer_norm": true,
                "norm_linear": false,
                "attention_transpose": true,
                "flip_lstm": false,
                "bilstm_cat": true,
                "batched_linear_projection": false,
                "channels_first_layer_norm": true,
                "silu_mul": false,
                "auto_fuse_elementwise": true
            }
        }"#;
        std::fs::write(&path, json).expect("write should succeed");

        let restored = load_versioned(&path).expect("load should succeed");

        assert_eq!(restored.version, 1);
        assert_eq!(restored.field_count, 12);
        assert!(
            restored.needs_migration(),
            "12 fields should need migration when current is 15"
        );
        assert!(!restored.is_current_version());

        assert!(!restored.config.norm_activ_conv1d);
        assert!(restored.config.fused_resblock);
        assert!(!restored.config.linear_activation);
        assert!(!restored.config.silu_mul);

        // Missing fields default to `true` from Default impl.
        assert!(
            restored.config.add_norm_linear,
            "missing add_norm_linear should default to true"
        );
        assert!(
            restored.config.fuse_adain_snake,
            "missing fuse_adain_snake should default to true"
        );
        assert!(
            restored.config.fuse_upsample_conv1d,
            "missing fuse_upsample_conv1d should default to true"
        );
        assert!(
            restored.config.fuse_instance_norm_mul_add,
            "missing fuse_instance_norm_mul_add should default to true"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// save/load with all 15 fields explicitly set to known values.
    #[test]
    fn test_versioned_all_15_fields_explicit_roundtrip() {
        let config = PeepholeConfig {
            norm_activ_conv1d: false,
            fused_resblock: true,
            linear_activation: false,
            add_layer_norm: true,
            norm_linear: false,
            attention_transpose: true,
            flip_lstm: false,
            bilstm_cat: true,
            batched_linear_projection: false,
            channels_first_layer_norm: true,
            silu_mul: false,
            auto_fuse_elementwise: true,
            add_norm_linear: false,
            fuse_adain_snake: true,
            fuse_upsample_conv1d: false,
            fuse_instance_norm_mul_add: false,
            fuse_conv1d_activation: false,
            fuse_snake_instance_norm: false,
            fuse_conv1d_snake_norm: false,
            fuse_conv1d_snake_norm_resblock: false,
            fuse_add_instance_norm_conv1x1: false,
            fuse_conv_transpose1d_activation: false,
            norm_activ_conv_transpose1d: false,
            fuse_instance_norm_conv1d: false,
            fuse_conv1d_instance_norm: false,
            fuse_linear_layer_norm: false,
            fuse_resblock_chain: false,
            fuse_activation_conv1d: false,
        };
        let path = temp_path("versioned_all15.json");

        save_versioned(&config, &path).expect("save should succeed");
        let restored = load_versioned(&path).expect("load should succeed");

        assert!(!restored.config.norm_activ_conv1d);
        assert!(restored.config.fused_resblock);
        assert!(!restored.config.linear_activation);
        assert!(restored.config.add_layer_norm);
        assert!(!restored.config.norm_linear);
        assert!(restored.config.attention_transpose);
        assert!(!restored.config.flip_lstm);
        assert!(restored.config.bilstm_cat);
        assert!(!restored.config.batched_linear_projection);
        assert!(restored.config.channels_first_layer_norm);
        assert!(!restored.config.silu_mul);
        assert!(restored.config.auto_fuse_elementwise);
        assert!(!restored.config.add_norm_linear);
        assert!(restored.config.fuse_adain_snake);
        assert!(!restored.config.fuse_upsample_conv1d);
        assert!(!restored.config.fuse_instance_norm_mul_add);

        assert_eq!(restored.config, config);
        assert!(restored.is_current_version());

        let _ = std::fs::remove_file(&path);
    }

    /// JSON format is human-readable (pretty-printed, field names visible).
    #[test]
    fn test_versioned_json_human_readable() {
        let config = PeepholeConfig::default();
        let path = temp_path("versioned_readable.json");

        save_versioned(&config, &path).expect("save should succeed");

        let data = std::fs::read_to_string(&path).expect("read should succeed");

        assert!(
            data.contains('\n'),
            "saved JSON should be pretty-printed with newlines"
        );
        assert!(
            data.contains("\"version\""),
            "JSON should contain version field name"
        );
        assert!(
            data.contains("\"field_count\""),
            "JSON should contain field_count field name"
        );
        assert!(
            data.contains("\"config\""),
            "JSON should contain config field name"
        );

        let expected_fields = [
            "norm_activ_conv1d",
            "fused_resblock",
            "linear_activation",
            "add_layer_norm",
            "norm_linear",
            "attention_transpose",
            "flip_lstm",
            "bilstm_cat",
            "batched_linear_projection",
            "channels_first_layer_norm",
            "silu_mul",
            "auto_fuse_elementwise",
            "add_norm_linear",
            "fuse_adain_snake",
            "fuse_upsample_conv1d",
            "fuse_instance_norm_mul_add",
            "fuse_conv1d_activation",
            "fuse_snake_instance_norm",
            "fuse_conv1d_snake_norm",
        ];
        for field in &expected_fields {
            assert!(
                data.contains(field),
                "JSON should contain field name '{field}'"
            );
        }
        assert_eq!(expected_fields.len(), 19, "should check all 19 fields");

        let _ = std::fs::remove_file(&path);
    }

    /// optimize_plan on empty graph → persist the result's config → reload and verify.
    #[test]
    fn test_versioned_optimize_persist_reload() {
        use crate::trace_compile::optimize_plan::optimize_plan;
        use nn_core::dyn_tensor::trace::ComputationGraph;
        use std::time::Duration;

        let graph = ComputationGraph::from_nodes(vec![]);
        let result = optimize_plan(&graph, Duration::from_secs(1))
            .expect("optimize_plan should succeed on empty graph");

        let path = temp_path("versioned_optimize_rt.json");
        save_versioned(&result.config, &path).expect("save_versioned should succeed");
        let restored = load_versioned(&path).expect("load_versioned should succeed");

        assert_eq!(restored.config, result.config);
        assert!(restored.is_current_version());
        assert!(!restored.needs_migration());

        let recompiled =
            crate::trace_compile::compile_trace_to_plan_configured(&graph, &restored.config)
                .expect("recompile should succeed");
        let recompiled_dispatches =
            crate::trace_compile::optimize_plan::count_dispatches(&recompiled);
        assert_eq!(
            recompiled_dispatches, result.dispatch_count,
            "recompiled dispatch count should match original"
        );

        let _ = std::fs::remove_file(&path);
    }
}
