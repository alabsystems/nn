// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`KokoroOptimalConfigs`] persistence.
//!
//! Part of #3828 (Self-Optimizing ML Compiler).

use std::path::Path;

use nn_dsl::PeepholeConfig;

use super::optimal_configs::{
    load_optimal_configs, load_optimal_configs_if_exists, save_optimal_configs,
    KokoroOptimalConfigs, SegmentOptimalConfig,
};

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "nn_optimal_configs_test_{}_{name}",
        std::process::id()
    ))
}

fn sample_segment(name: &str, dispatch_count: usize) -> SegmentOptimalConfig {
    SegmentOptimalConfig {
        segment_name: name.to_string(),
        peephole_config: PeepholeConfig::default(),
        dispatch_count,
        estimated_cost_us: dispatch_count as f64 * 5.5,
        search_budget_ms: 5000,
        configs_evaluated: 32768,
    }
}

fn sample_segment_custom(name: &str, config: PeepholeConfig) -> SegmentOptimalConfig {
    SegmentOptimalConfig {
        segment_name: name.to_string(),
        peephole_config: config,
        dispatch_count: 20,
        estimated_cost_us: 110.0,
        search_budget_ms: 3000,
        configs_evaluated: 16384,
    }
}

/// Expected peephole field count (28 boolean fields in PeepholeConfig).
/// Derived by serializing default config and counting keys.
fn expected_field_count() -> u32 {
    let config = PeepholeConfig::default();
    let value = serde_json::to_value(&config).expect("serialize");
    value.as_object().map_or(0, |m| m.len() as u32)
}

// ---- Serialize/deserialize roundtrip tests ----

#[test]
fn test_roundtrip_empty_configs() {
    let configs = KokoroOptimalConfigs::new();
    let path = temp_path("empty.json");

    save_optimal_configs(&configs, &path).expect("save should succeed");
    let restored = load_optimal_configs(&path).expect("load should succeed");

    assert_eq!(restored.version, configs.version);
    assert_eq!(restored.peephole_field_count, expected_field_count());
    assert!(restored.segments.is_empty());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_roundtrip_single_segment() {
    let mut configs = KokoroOptimalConfigs::new();
    configs.insert(sample_segment("plbert", 18));

    let path = temp_path("single.json");
    save_optimal_configs(&configs, &path).expect("save should succeed");
    let restored = load_optimal_configs(&path).expect("load should succeed");

    assert_eq!(restored.segment_count(), 1);
    let plbert = restored.segments.get("plbert").expect("plbert should exist");
    assert_eq!(plbert.segment_name, "plbert");
    assert_eq!(plbert.dispatch_count, 18);
    assert!((plbert.estimated_cost_us - 99.0).abs() < f64::EPSILON);
    assert_eq!(plbert.search_budget_ms, 5000);
    assert_eq!(plbert.configs_evaluated, 32768);
    assert_eq!(plbert.peephole_config, PeepholeConfig::default());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_roundtrip_all_segments() {
    let segment_names = [
        "plbert",
        "text",
        "prosody",
        "f0",
        "generator",
        "regulate",
        "sinegen_pre",
        "sinegen_post",
    ];

    let mut configs = KokoroOptimalConfigs::new();
    for (i, name) in segment_names.iter().enumerate() {
        configs.insert(sample_segment(name, 10 + i * 5));
    }

    let path = temp_path("all_segments.json");
    save_optimal_configs(&configs, &path).expect("save should succeed");
    let restored = load_optimal_configs(&path).expect("load should succeed");

    assert_eq!(restored.segment_count(), 8);
    for (i, name) in segment_names.iter().enumerate() {
        let seg = restored.segments.get(*name).expect("segment should exist");
        assert_eq!(seg.dispatch_count, 10 + i * 5);
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_roundtrip_custom_peephole_config() {
    let custom_config = PeepholeConfig {
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
        ..PeepholeConfig::default()
    };

    let mut configs = KokoroOptimalConfigs::new();
    configs.insert(sample_segment_custom("generator", custom_config.clone()));

    let path = temp_path("custom_config.json");
    save_optimal_configs(&configs, &path).expect("save should succeed");
    let restored = load_optimal_configs(&path).expect("load should succeed");

    let generator_cfg = restored
        .segments
        .get("generator")
        .expect("generator should exist");
    assert_eq!(generator_cfg.peephole_config, custom_config);
    assert!(!generator_cfg.peephole_config.norm_activ_conv1d);
    assert!(generator_cfg.peephole_config.fused_resblock);

    let _ = std::fs::remove_file(&path);
}

// ---- Version invalidation tests ----

#[test]
fn test_is_valid_current_version() {
    let configs = KokoroOptimalConfigs::new();
    assert!(configs.is_valid(env!("CARGO_PKG_VERSION")));
    assert!(configs.is_current());
}

#[test]
fn test_is_valid_wrong_version() {
    let configs = KokoroOptimalConfigs::with_version("99.0.0");
    assert!(!configs.is_valid(env!("CARGO_PKG_VERSION")));
    assert!(!configs.is_current());
}

#[test]
fn test_is_valid_wrong_field_count() {
    let mut configs = KokoroOptimalConfigs::new();
    configs.peephole_field_count = 12;
    assert!(!configs.is_valid(env!("CARGO_PKG_VERSION")));
    assert!(!configs.is_current());
}

#[test]
fn test_version_invalidation_after_roundtrip() {
    let configs = KokoroOptimalConfigs::with_version("0.0.1-old");

    let path = temp_path("old_version.json");
    save_optimal_configs(&configs, &path).expect("save should succeed");
    let restored = load_optimal_configs(&path).expect("load should succeed");

    assert!(!restored.is_current());
    assert!(restored.is_valid("0.0.1-old"));
    assert!(!restored.is_valid(env!("CARGO_PKG_VERSION")));

    let _ = std::fs::remove_file(&path);
}

// ---- Per-segment config lookup tests ----

#[test]
fn test_get_config_found() {
    let mut configs = KokoroOptimalConfigs::new();
    configs.insert(sample_segment("text", 12));

    let config = configs.get_config("text");
    assert!(config.is_some());
    assert_eq!(*config.unwrap(), PeepholeConfig::default());
}

#[test]
fn test_get_config_not_found() {
    let configs = KokoroOptimalConfigs::new();
    assert!(configs.get_config("nonexistent").is_none());
}

#[test]
fn test_get_config_or_default_found() {
    let custom = PeepholeConfig {
        norm_activ_conv1d: false,
        ..PeepholeConfig::default()
    };
    let mut configs = KokoroOptimalConfigs::new();
    configs.insert(sample_segment_custom("f0", custom.clone()));

    let result = configs.get_config_or_default("f0");
    assert_eq!(result, custom);
    assert!(!result.norm_activ_conv1d);
}

#[test]
fn test_get_config_or_default_missing_returns_default() {
    let configs = KokoroOptimalConfigs::new();
    let result = configs.get_config_or_default("missing_segment");
    assert_eq!(result, PeepholeConfig::default());
}

// ---- File not found handling ----

#[test]
fn test_load_nonexistent_file_returns_error() {
    let result = load_optimal_configs(Path::new("/nonexistent/optimal_configs.json"));
    assert!(result.is_err());
}

#[test]
fn test_load_if_exists_nonexistent_returns_none() {
    let result = load_optimal_configs_if_exists(Path::new("/nonexistent/optimal_configs.json"));
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn test_load_if_exists_existing_returns_some() {
    let mut configs = KokoroOptimalConfigs::new();
    configs.insert(sample_segment("plbert", 18));

    let path = temp_path("if_exists.json");
    save_optimal_configs(&configs, &path).expect("save should succeed");

    let result = load_optimal_configs_if_exists(&path);
    assert!(result.is_ok());
    let loaded = result.unwrap().expect("should be Some");
    assert_eq!(loaded.segment_count(), 1);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_load_invalid_json_returns_error() {
    let path = temp_path("invalid.json");
    std::fs::write(&path, "not valid json {{{").expect("write should succeed");

    let result = load_optimal_configs(&path);
    assert!(result.is_err());

    let result_if_exists = load_optimal_configs_if_exists(&path);
    assert!(result_if_exists.is_err());

    let _ = std::fs::remove_file(&path);
}

// ---- to_peephole_map tests ----

#[test]
fn test_to_peephole_map_empty() {
    let configs = KokoroOptimalConfigs::new();
    let map = configs.to_peephole_map();
    assert!(map.is_empty());
}

#[test]
fn test_to_peephole_map_preserves_configs() {
    let custom = PeepholeConfig {
        silu_mul: false,
        ..PeepholeConfig::default()
    };

    let mut configs = KokoroOptimalConfigs::new();
    configs.insert(sample_segment("plbert", 18));
    configs.insert(sample_segment_custom("generator", custom.clone()));

    let map = configs.to_peephole_map();
    assert_eq!(map.len(), 2);
    assert_eq!(map["plbert"], PeepholeConfig::default());
    assert_eq!(map["generator"], custom);
    assert!(!map["generator"].silu_mul);
}

// ---- insert/update tests ----

#[test]
fn test_insert_overwrites_existing() {
    let mut configs = KokoroOptimalConfigs::new();
    configs.insert(sample_segment("plbert", 18));
    assert_eq!(configs.segments["plbert"].dispatch_count, 18);

    configs.insert(sample_segment("plbert", 15));
    assert_eq!(configs.segments["plbert"].dispatch_count, 15);
    assert_eq!(configs.segment_count(), 1);
}

// ---- JSON format tests ----

#[test]
fn test_json_is_human_readable() {
    let mut configs = KokoroOptimalConfigs::new();
    configs.insert(sample_segment("plbert", 18));

    let path = temp_path("readable.json");
    save_optimal_configs(&configs, &path).expect("save should succeed");

    let data = std::fs::read_to_string(&path).expect("read should succeed");
    assert!(data.contains('\n'), "JSON should be pretty-printed");
    assert!(data.contains("\"version\""));
    assert!(data.contains("\"peephole_field_count\""));
    assert!(data.contains("\"segments\""));
    assert!(data.contains("\"plbert\""));
    assert!(data.contains("\"dispatch_count\""));
    assert!(data.contains("\"estimated_cost_us\""));
    assert!(data.contains("\"search_budget_ms\""));
    assert!(data.contains("\"configs_evaluated\""));
    assert!(data.contains("\"peephole_config\""));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_default_creates_current_version() {
    let configs = KokoroOptimalConfigs::default();
    assert_eq!(configs.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(configs.peephole_field_count, expected_field_count());
    assert!(configs.segments.is_empty());
    assert!(configs.is_current());
}

#[test]
fn test_field_count_is_28() {
    // Sanity check: PeepholeConfig currently has 28 boolean fields.
    assert_eq!(expected_field_count(), 28);
    let configs = KokoroOptimalConfigs::new();
    assert_eq!(configs.peephole_field_count, 28);
}

#[test]
fn test_with_optimal_configs_loads_valid_file() {
    let custom = PeepholeConfig {
        norm_activ_conv1d: false,
        fused_resblock: false,
        ..PeepholeConfig::default()
    };

    let mut optimal = KokoroOptimalConfigs::new();
    optimal.insert(SegmentOptimalConfig {
        segment_name: "plbert".to_string(),
        peephole_config: custom.clone(),
        dispatch_count: 15,
        estimated_cost_us: 80.0,
        search_budget_ms: 5000,
        configs_evaluated: 32768,
    });

    let path = temp_path("with_optimal.json");
    save_optimal_configs(&optimal, &path).expect("save should succeed");

    // Verify the map conversion includes the custom config.
    let loaded = load_optimal_configs(&path).expect("load should succeed");
    assert!(loaded.is_current());
    let map = loaded.to_peephole_map();
    assert_eq!(map.len(), 1);
    assert_eq!(map["plbert"], custom);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_with_optimal_configs_stale_version_not_applied() {
    let mut stale = KokoroOptimalConfigs::with_version("0.0.0-stale");
    stale.insert(sample_segment("plbert", 15));

    let path = temp_path("stale_optimal.json");
    save_optimal_configs(&stale, &path).expect("save should succeed");

    let loaded = load_optimal_configs(&path).expect("load should succeed");
    assert!(!loaded.is_current(), "stale configs should not be current");

    let _ = std::fs::remove_file(&path);
}
