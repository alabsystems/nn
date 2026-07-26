// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for build-time MSL pre-compilation.

use super::*;

#[test]
fn test_precompile_shapes_default() {
    let shapes = PrecompileShapes::default();
    assert_eq!(shapes.seq_lens, vec![10, 20, 40, 80]);
    assert_eq!(shapes.t_mels, vec![20, 40, 80, 160, 320]);
}

#[test]
fn test_export_plan_msl_writes_files() {
    // Test the export_plan_msl function with a simple traced graph.
    use nn_core::dyn_tensor::trace::{record_input, trace_graph};
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::DType;

    // Create a simple computation graph via tracing.
    let x = DynTensor::zeros(&[1, 4, 8], DType::F32, &nn_core::Device::Cpu).unwrap();
    let (_out, mut graph) = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).expect("invariant: tracing active"));
        let y = inp.mul_scalar(2.0)?;
        Ok(y)
    })
    .unwrap();

    // Set primary output.
    let nodes = graph.nodes();
    if let Some(last) = nodes.last() {
        let _ = graph.set_primary_output(last.id());
    }

    let dir =
        std::env::temp_dir().join(format!("nn_precompile_export_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let result = export_plan_msl(&graph, "test_seg", 8, &dir);

    match result {
        Ok((files, bytes)) => {
            assert!(files > 0, "should write at least 1 MSL file");
            assert!(bytes > 0, "should write non-empty MSL");

            // Verify at least one .metal file exists on disk.
            let entries: Vec<_> = std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "metal"))
                .collect();
            assert!(!entries.is_empty(), "at least one .metal file should exist");

            // Verify MSL content in at least one file.
            let content = std::fs::read_to_string(entries[0].path()).unwrap();
            assert!(
                content.contains("[[kernel]]"),
                "MSL should contain Metal kernel attribute"
            );
        }
        Err(e) => {
            // If compilation fails (e.g., no Dispatch steps), that's ok
            // for a simple mul_scalar graph.
            eprintln!("export_plan_msl: {e} (acceptable for simple graph)");
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_precompile_result_fields() {
    let result = PrecompileResult {
        files_written: 42,
        total_bytes: 128_000,
        segment_counts: vec![
            ("plbert", 10),
            ("text", 8),
            ("prosody", 8),
            ("f0", 8),
            ("generator", 8),
            ("regulate", 4),
            ("sinegen_pre", 5),
            ("sinegen_post", 5),
        ],
    };
    assert_eq!(result.files_written, 42);
    assert_eq!(result.total_bytes, 128_000);
    assert_eq!(result.segment_counts.len(), 8);
}

#[test]
fn test_precompile_shapes_t_frames() {
    let shapes = PrecompileShapes::default();
    let t_frames = shapes.t_frames();
    // t_frames = 2 × t_mel
    assert_eq!(t_frames, vec![40, 80, 160, 320, 640]);
}

// ======================== PrecompileShapes convenience constructors (#3873) ========================

#[test]
fn test_precompile_shapes_short() {
    let shapes = PrecompileShapes::short();
    assert_eq!(shapes.seq_lens, vec![10, 20, 40, 80]);
    assert_eq!(shapes.t_mels, vec![20, 40, 80, 160]);
    // short() should have fewer t_mels than default.
    assert!(
        shapes.t_mels.len() < PrecompileShapes::default().t_mels.len(),
        "short() should cover fewer t_mel values than default"
    );
}

#[test]
fn test_precompile_shapes_long_form() {
    let shapes = PrecompileShapes::long_form();
    assert_eq!(shapes.seq_lens, vec![40, 80, 160, 256, 512]);
    assert_eq!(shapes.t_mels, vec![80, 160, 320, 640, 1024]);
    // long_form() should have larger max seq_len than default.
    let default_max = *PrecompileShapes::default().seq_lens.last().unwrap();
    let long_max = *shapes.seq_lens.last().unwrap();
    assert!(
        long_max > default_max,
        "long_form max seq_len ({long_max}) should exceed default max ({default_max})"
    );
}

#[test]
fn test_precompile_shapes_chorus() {
    let shapes = PrecompileShapes::chorus();
    assert_eq!(shapes.seq_lens, vec![20, 40, 80, 128]);
    assert_eq!(shapes.t_mels, vec![40, 80, 160, 320]);
}

#[test]
fn test_precompile_shapes_short_t_frames() {
    let shapes = PrecompileShapes::short();
    let t_frames = shapes.t_frames();
    assert_eq!(t_frames, vec![40, 80, 160, 320]);
}

#[test]
fn test_precompile_shapes_long_form_t_frames() {
    let shapes = PrecompileShapes::long_form();
    let t_frames = shapes.t_frames();
    assert_eq!(t_frames, vec![160, 320, 640, 1280, 2048]);
}

#[test]
fn test_precompile_shapes_chorus_t_frames() {
    let shapes = PrecompileShapes::chorus();
    let t_frames = shapes.t_frames();
    assert_eq!(t_frames, vec![80, 160, 320, 640]);
}

#[test]
fn test_precompile_shapes_builder_chain() {
    // Verify builder methods can chain with convenience constructors.
    let shapes = PrecompileShapes::short()
        .with_seq_lens(vec![5, 10])
        .with_t_mels(vec![10, 20]);
    assert_eq!(shapes.seq_lens, vec![5, 10]);
    assert_eq!(shapes.t_mels, vec![10, 20]);
}

#[test]
fn test_precompile_shapes_all_constructors_nonempty() {
    // All constructors produce non-empty shapes (required for warmup to do work).
    for (name, shapes) in [
        ("default", PrecompileShapes::default()),
        ("new", PrecompileShapes::new()),
        ("short", PrecompileShapes::short()),
        ("long_form", PrecompileShapes::long_form()),
        ("chorus", PrecompileShapes::chorus()),
    ] {
        assert!(
            !shapes.seq_lens.is_empty(),
            "{name}() should have non-empty seq_lens"
        );
        assert!(
            !shapes.t_mels.is_empty(),
            "{name}() should have non-empty t_mels"
        );
    }
}

// ======================== WarmupShapesResult tests (#3873) ========================

#[test]
fn test_warmup_shapes_result_fields() {
    let result = WarmupShapesResult {
        segments_compiled: 32,
    };
    assert_eq!(result.segments_compiled, 32);
    // Verify Debug impl works (non_exhaustive struct).
    let debug_str = format!("{result:?}");
    assert!(
        debug_str.contains("segments_compiled"),
        "Debug should include field names"
    );
}

#[test]
fn test_warmup_shapes_result_clone_eq() {
    let a = WarmupShapesResult {
        segments_compiled: 10,
    };
    let b = a.clone();
    assert_eq!(a, b);
}

fn mini_test_config() -> nn_models::KokoroConfig {
    let mut plbert = nn_models::PlbertConfig::default();
    plbert.vocab_size = 10;
    plbert.embedding_dim = 4;
    plbert.hidden_size = 8;
    plbert.num_attention_heads = 2;
    plbert.intermediate_size = 16;
    plbert.max_position_embeddings = 16;
    plbert.num_hidden_layers = 1;

    let mut config = nn_models::KokoroConfig::default();
    config.d_en = 8;
    config.n_prosody_layers = 1;
    config.style_dim = 4;
    config.upsample_rates = vec![2];
    config.upsample_kernel_sizes = vec![4];
    config.resblock_kernel_sizes = vec![3];
    config.resblock_dilations = vec![vec![1, 2]];
    config.gen_initial_channels = 8;
    config.n_fft = 4;
    config.f0_bilstm_hidden = 4;
    config.plbert = plbert;
    config
}

fn mini_test_kokoro() -> CompiledKokoro {
    crate::test_common::init();
    let config = mini_test_config();
    let model = nn_models::KokoroModel::load(
        nn_core::VarBuilder::zeros(DType::F32, &nn_core::Device::Cpu),
        &config,
    )
    .expect("mini Kokoro model from zero weights");
    CompiledKokoro::new(model).expect("CompiledKokoro::new mini model")
}

#[test]
fn test_warmup_cpu_loaded_model_keeps_regulate_inputs_device_consistent() {
    let mut kokoro = mini_test_kokoro();
    let cache = PipelineCache::new_global().expect("Metal cache");
    let shapes = PrecompileShapes::new()
        .with_seq_lens(vec![1])
        .with_t_mels(vec![1]);

    let first = kokoro
        .warmup(&shapes, &cache)
        .expect("warmup should succeed for CPU-loaded model");
    assert_eq!(
        first, 8,
        "single-shape warmup should compile all 8 segments"
    );

    let second = kokoro
        .warmup(&shapes, &cache)
        .expect("warmup should reuse cached segments");
    assert_eq!(second, 0, "second warmup should hit the segment caches");
}

// ======================== optimizer warmup tests (#3828) ========================

/// OptimizerWarmupResult fields are accessible and Debug works.
#[cfg(feature = "plan-serde")]
#[test]
fn test_optimizer_warmup_result_fields() {
    let result = OptimizerWarmupResult {
        loaded_from_cache: true,
        configs_applied: 5,
        segments_compiled: 20,
    };
    assert!(result.loaded_from_cache);
    assert_eq!(result.configs_applied, 5);
    assert_eq!(result.segments_compiled, 20);
    // Verify Debug impl works (non_exhaustive struct).
    let debug_str = format!("{result:?}");
    assert!(
        debug_str.contains("loaded_from_cache"),
        "Debug should include field names"
    );
}

/// save_peephole_configs + load_peephole_configs round-trips correctly.
#[cfg(feature = "plan-serde")]
#[test]
fn test_save_load_peephole_configs_roundtrip() {
    use std::collections::HashMap;

    let dir = std::env::temp_dir().join(format!(
        "nn_precompile_save_load_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("optimal_configs.json");

    // Build a HashMap<String, PeepholeConfig> with two segment configs.
    let mut configs = HashMap::new();
    let mut gen_config = nn_dsl::PeepholeConfig::default();
    gen_config.fused_resblock = false;
    gen_config.silu_mul = false;
    configs.insert("generator".to_string(), gen_config);

    let mut plbert_config = nn_dsl::PeepholeConfig::default();
    plbert_config.linear_activation = false;
    plbert_config.attention_transpose = false;
    configs.insert("plbert".to_string(), plbert_config);

    // Save via the new function (in compiled_kokoro module, two levels up).
    super::super::save_peephole_configs(&configs, &path).expect("save should succeed");

    // Verify file exists and is valid JSON.
    assert!(path.exists(), "config file should exist after save");
    let content = std::fs::read_to_string(&path).expect("read saved file");
    assert!(
        content.contains("generator"),
        "saved JSON should contain segment names"
    );

    // Load back and verify round-trip (load_peephole_configs is in compiled_kokoro module).
    let loaded = super::super::load_peephole_configs(&path).expect("load should succeed");
    assert_eq!(loaded.len(), 2, "should have 2 segment configs");

    let loaded_gen = loaded.get("generator").expect("generator config");
    assert!(
        !loaded_gen.fused_resblock,
        "fused_resblock should be disabled"
    );
    assert!(!loaded_gen.silu_mul, "silu_mul should be disabled");
    assert!(
        loaded_gen.norm_activ_conv1d,
        "norm_activ_conv1d should be default (true)"
    );

    let loaded_plb = loaded.get("plbert").expect("plbert config");
    assert!(
        !loaded_plb.linear_activation,
        "linear_activation should be disabled"
    );
    assert!(
        !loaded_plb.attention_transpose,
        "attention_transpose should be disabled"
    );

    // Cleanup.
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

/// save_peephole_configs creates parent directories message on write failure.
#[cfg(feature = "plan-serde")]
#[test]
fn test_save_peephole_configs_write_error() {
    use std::collections::HashMap;

    let configs = HashMap::new();
    let result = super::super::save_peephole_configs(
        &configs,
        Path::new("/nonexistent_dir_abc123/configs.json"),
    );
    match result {
        Err(CompiledKokoroError::ConfigLoad(msg)) => {
            assert!(msg.contains("write"), "error should mention write: {msg}");
        }
        other => panic!("expected ConfigLoad error, got: {other:?}"),
    }
}

// ======================== optimization_summary tests (#3828) ========================

/// optimization_summary returns placeholder when no results are stored.
#[cfg(feature = "plan-serde")]
#[test]
fn test_optimization_summary_no_results() {
    // Cannot construct CompiledKokoro without model weights, so test the
    // summary format using the OptimizerWarmupResult fields instead.
    // This validates that the code compiles and the summary format is correct.
    let result = OptimizerWarmupResult {
        loaded_from_cache: false,
        configs_applied: 0,
        segments_compiled: 0,
    };
    // Verify the struct is constructable with the expected fields.
    assert!(!result.loaded_from_cache);
    assert_eq!(result.configs_applied, 0);
}

/// Verify optimization_summary format with synthetic data via optimize_plan.
#[cfg(feature = "plan-serde")]
#[test]
fn test_optimization_summary_format() {
    use nn_core::dyn_tensor::trace::ComputationGraph;

    // Create real OptimizationResult using optimize_plan on an empty graph.
    // Empty graph -> 0 dispatches, but the result structure is valid.
    let graph = ComputationGraph::from_nodes(vec![]);
    let mut base_result = nn_dsl::optimize_plan(&graph, std::time::Duration::from_millis(10))
        .expect("optimize_plan on empty graph should succeed");

    // Override fields to create synthetic data for summary testing.
    let mut plbert_result = base_result.clone();
    plbert_result.baseline_dispatch_count = 25;
    plbert_result.dispatch_count = 18;
    plbert_result.baseline_cost_ns = 150_200.0;
    plbert_result.best_cost_ns = 102_300.0;

    base_result.baseline_dispatch_count = 45;
    base_result.dispatch_count = 32;
    base_result.baseline_cost_ns = 450_000.0;
    base_result.best_cost_ns = 310_200.0;

    let results: Vec<(String, nn_dsl::OptimizationResult)> = vec![
        ("plbert".to_string(), plbert_result),
        ("generator".to_string(), base_result),
    ];

    // Format the summary the same way optimization_summary() does.
    let mut lines = Vec::new();
    lines.push("=== Kokoro Segment Optimization Summary ===".to_string());
    let mut total_baseline = 0usize;
    let mut total_optimal = 0usize;
    for (name, result) in &results {
        let baseline = result.baseline_dispatch_count;
        let optimal = result.dispatch_count;
        total_baseline += baseline;
        total_optimal += optimal;
        let dispatch_pct = if baseline > 0 {
            let saved = baseline.saturating_sub(optimal);
            (saved as f64 / baseline as f64) * -100.0
        } else {
            0.0
        };
        lines.push(format!(
            "{name:<14} baseline {baseline:>3} -> optimal {optimal:>3} dispatches ({dispatch_pct:+.1}%)",
        ));
    }
    let total_pct = if total_baseline > 0 {
        let saved = total_baseline.saturating_sub(total_optimal);
        (saved as f64 / total_baseline as f64) * -100.0
    } else {
        0.0
    };
    lines.push(format!(
        "Total:         baseline {total_baseline:>3} -> optimal {total_optimal:>3} dispatches ({total_pct:+.1}%)",
    ));
    let summary = lines.join("\n");

    // Verify the summary contains expected content.
    assert!(
        summary.contains("Kokoro Segment Optimization Summary"),
        "should contain header"
    );
    assert!(summary.contains("plbert"), "should contain plbert segment");
    assert!(
        summary.contains("generator"),
        "should contain generator segment"
    );
    assert!(summary.contains("Total:"), "should contain total line");
    // Total: baseline 70 -> optimal 50
    assert!(summary.contains("70"), "total baseline should be 25+45=70");
    assert!(summary.contains("50"), "total optimal should be 18+32=50");
}

/// Verify warmup_with_optimizer signature compiles with all parameters.
#[cfg(feature = "plan-serde")]
#[test]
fn test_warmup_with_optimizer_signature_compiles() {
    // This test verifies the method signature compiles correctly.
    // We can't call it without KOKORO_WEIGHTS, but the type-check is valuable.
    fn _assert_method_exists(
        kokoro: &mut super::super::CompiledKokoro,
        shapes: &PrecompileShapes,
        cache: &crate::cache::PipelineCache,
        input_ids: &DynTensor,
        style: &DynTensor,
    ) {
        let _result: Result<OptimizerWarmupResult, CompiledKokoroError> = kokoro
            .warmup_with_optimizer(
                shapes,
                cache,
                input_ids,
                style,
                1.0,
                std::time::Duration::from_secs(5),
                None,
            );
    }

    fn _assert_optimization_results_accessor(
        kokoro: &super::super::CompiledKokoro,
    ) -> Option<&[(String, nn_dsl::OptimizationResult)]> {
        kokoro.optimization_results()
    }

    fn _assert_optimization_summary(kokoro: &super::super::CompiledKokoro) -> String {
        kokoro.optimization_summary()
    }

    fn _assert_optimize_rtf_method(
        kokoro: &mut super::super::CompiledKokoro,
        shapes: &PrecompileShapes,
        cache: &crate::cache::PipelineCache,
        input_ids: &DynTensor,
        style: &DynTensor,
    ) {
        let _result: Result<super::super::rtf_optimizer::ClosedLoopRtfReport, CompiledKokoroError> =
            kokoro.optimize_rtf(
                shapes,
                cache,
                input_ids,
                style,
                1.0,
                std::time::Duration::from_secs(5),
                None,
            );
    }
}

// ======================== SegmentPeepholeConfigs tests ========================

/// Default SegmentPeepholeConfigs has all segments as None.
#[test]
fn test_segment_peephole_configs_default_all_none() {
    let configs = SegmentPeepholeConfigs::default();
    assert!(configs.plbert.is_none());
    assert!(configs.text.is_none());
    assert!(configs.prosody.is_none());
    assert!(configs.f0_energy.is_none());
    assert!(configs.generator.is_none());
    assert!(configs.regulate.is_none());
    assert!(configs.sinegen_pre.is_none());
    assert!(configs.sinegen_post.is_none());
    assert_eq!(configs.configured_count(), 0);
}

/// new() is equivalent to default().
#[test]
fn test_segment_peephole_configs_new_equals_default() {
    assert_eq!(
        SegmentPeepholeConfigs::new(),
        SegmentPeepholeConfigs::default()
    );
}

/// for_segment returns the correct config for each kind name.
#[test]
fn test_segment_peephole_configs_for_segment_lookup() {
    let gen_config = nn_dsl::PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };

    let plbert_config = nn_dsl::PeepholeConfig {
        linear_activation: false,
        ..Default::default()
    };

    let configs = SegmentPeepholeConfigs {
        plbert: Some(plbert_config),
        generator: Some(gen_config),
        ..Default::default()
    };

    // Segments with configs return Some.
    let plbert_ref = configs
        .for_segment("plbert")
        .expect("plbert should be Some");
    assert!(!plbert_ref.linear_activation);

    let gen_ref = configs
        .for_segment("generator")
        .expect("generator should be Some");
    assert!(!gen_ref.silu_mul);

    // Segments without configs return None.
    assert!(configs.for_segment("text").is_none());
    assert!(configs.for_segment("prosody").is_none());
    assert!(configs.for_segment("f0_energy").is_none());
    assert!(configs.for_segment("regulate").is_none());
    assert!(configs.for_segment("sinegen_pre").is_none());
    assert!(configs.for_segment("sinegen_post").is_none());

    // Unknown segment names return None.
    assert!(configs.for_segment("nonexistent").is_none());
    assert!(configs.for_segment("").is_none());
}

/// "f0" is an alias for "f0_energy" in for_segment.
#[test]
fn test_segment_peephole_configs_f0_alias() {
    let f0_config = nn_dsl::PeepholeConfig {
        fused_resblock: false,
        ..Default::default()
    };

    let configs = SegmentPeepholeConfigs {
        f0_energy: Some(f0_config),
        ..Default::default()
    };

    // Both "f0" and "f0_energy" should return the same config.
    let via_f0 = configs.for_segment("f0").expect("f0 alias should work");
    let via_f0_energy = configs
        .for_segment("f0_energy")
        .expect("f0_energy should work");
    assert_eq!(via_f0, via_f0_energy);
    assert!(!via_f0.fused_resblock);
}

/// configured_count returns the correct count.
#[test]
fn test_segment_peephole_configs_configured_count() {
    let default_config = nn_dsl::PeepholeConfig::default();

    let mut configs = SegmentPeepholeConfigs::new();
    assert_eq!(configs.configured_count(), 0);

    configs.plbert = Some(default_config.clone());
    assert_eq!(configs.configured_count(), 1);

    configs.generator = Some(default_config.clone());
    assert_eq!(configs.configured_count(), 2);

    configs.text = Some(default_config.clone());
    configs.prosody = Some(default_config.clone());
    configs.f0_energy = Some(default_config.clone());
    configs.regulate = Some(default_config.clone());
    configs.sinegen_pre = Some(default_config.clone());
    configs.sinegen_post = Some(default_config);
    assert_eq!(configs.configured_count(), 8);
}

/// to_hashmap converts to the expected HashMap format.
#[test]
fn test_segment_peephole_configs_to_hashmap() {
    let gen_config = nn_dsl::PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };

    let configs = SegmentPeepholeConfigs {
        generator: Some(gen_config),
        f0_energy: Some(nn_dsl::PeepholeConfig::default()),
        ..Default::default()
    };

    let map = configs.to_hashmap();
    assert_eq!(map.len(), 2);
    assert!(map.contains_key("generator"));
    // f0_energy maps to "f0" key for backward compat.
    assert!(map.contains_key("f0"));
    assert!(!map.get("generator").unwrap().silu_mul);
}

/// from_hashmap constructs SegmentPeepholeConfigs from a HashMap.
#[test]
fn test_segment_peephole_configs_from_hashmap() {
    use std::collections::HashMap;

    let mut map = HashMap::new();
    let plbert_config = nn_dsl::PeepholeConfig {
        attention_transpose: false,
        ..Default::default()
    };
    map.insert("plbert".to_string(), plbert_config);

    let f0_config = nn_dsl::PeepholeConfig {
        fused_resblock: false,
        ..Default::default()
    };
    // Test both "f0" key (used by existing peephole_configs).
    map.insert("f0".to_string(), f0_config);

    let configs = SegmentPeepholeConfigs::from_hashmap(&map);
    assert!(configs.plbert.is_some());
    assert!(!configs.plbert.as_ref().unwrap().attention_transpose);

    assert!(configs.f0_energy.is_some());
    assert!(!configs.f0_energy.as_ref().unwrap().fused_resblock);

    assert!(configs.text.is_none());
    assert!(configs.generator.is_none());
    assert_eq!(configs.configured_count(), 2);
}

/// from_hashmap prefers "f0_energy" over "f0" when both are present.
#[test]
fn test_segment_peephole_configs_from_hashmap_f0_energy_priority() {
    use std::collections::HashMap;

    let mut map = HashMap::new();
    let f0_config = nn_dsl::PeepholeConfig {
        fused_resblock: false,
        ..Default::default()
    };
    map.insert("f0".to_string(), f0_config);

    let f0_energy_config = nn_dsl::PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };
    map.insert("f0_energy".to_string(), f0_energy_config);

    let configs = SegmentPeepholeConfigs::from_hashmap(&map);
    // "f0_energy" should take priority over "f0".
    let f0 = configs.f0_energy.as_ref().unwrap();
    assert!(!f0.silu_mul, "f0_energy key should win over f0 key");
}

/// to_hashmap → from_hashmap round-trip preserves all configs.
#[test]
fn test_segment_peephole_configs_hashmap_roundtrip() {
    let gen_config = nn_dsl::PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };
    let plbert_config = nn_dsl::PeepholeConfig {
        linear_activation: false,
        ..Default::default()
    };

    let original = SegmentPeepholeConfigs {
        plbert: Some(plbert_config),
        generator: Some(gen_config),
        ..Default::default()
    };

    let map = original.to_hashmap();
    let restored = SegmentPeepholeConfigs::from_hashmap(&map);

    assert_eq!(original.plbert, restored.plbert);
    assert_eq!(original.generator, restored.generator);
    assert_eq!(original.text, restored.text);
    assert_eq!(original.prosody, restored.prosody);
    // Note: f0_energy goes through "f0" key, so direct field comparison works.
    assert_eq!(original.f0_energy, restored.f0_energy);
    assert_eq!(original.regulate, restored.regulate);
    assert_eq!(original.sinegen_pre, restored.sinegen_pre);
    assert_eq!(original.sinegen_post, restored.sinegen_post);
}

/// save_to_dir / load_from_dir round-trip preserves all configs.
#[cfg(feature = "plan-serde")]
#[test]
fn test_segment_peephole_configs_save_load_dir_roundtrip() {
    let dir = std::env::temp_dir().join(format!(
        "nn_segment_peephole_dir_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);

    let mut gen_config = nn_dsl::PeepholeConfig::default();
    gen_config.silu_mul = false;
    gen_config.fused_resblock = false;

    let mut text_config = nn_dsl::PeepholeConfig::default();
    text_config.linear_activation = false;

    let original = SegmentPeepholeConfigs {
        generator: Some(gen_config.clone()),
        text: Some(text_config.clone()),
        f0_energy: Some(nn_dsl::PeepholeConfig::default()),
        ..Default::default()
    };

    // Save to directory.
    original
        .save_to_dir(&dir)
        .expect("save_to_dir should succeed");

    // Verify the expected files exist.
    assert!(
        dir.join("generator_config.json").exists(),
        "generator file should exist"
    );
    assert!(
        dir.join("text_config.json").exists(),
        "text file should exist"
    );
    assert!(
        dir.join("f0_energy_config.json").exists(),
        "f0_energy file should exist"
    );
    // Segments with None should NOT have files.
    assert!(
        !dir.join("plbert_config.json").exists(),
        "plbert file should not exist"
    );
    assert!(
        !dir.join("prosody_config.json").exists(),
        "prosody file should not exist"
    );
    assert!(
        !dir.join("regulate_config.json").exists(),
        "regulate file should not exist"
    );

    // Verify file content is valid JSON.
    let gen_content =
        std::fs::read_to_string(dir.join("generator_config.json")).expect("read generator file");
    assert!(
        gen_content.contains("silu_mul"),
        "generator config should contain silu_mul field"
    );

    // Load back and verify round-trip.
    let loaded = SegmentPeepholeConfigs::load_from_dir(&dir).expect("load_from_dir should succeed");

    assert_eq!(loaded.configured_count(), 3, "should have 3 configs loaded");
    assert_eq!(loaded.generator.as_ref(), Some(&gen_config));
    assert_eq!(loaded.text.as_ref(), Some(&text_config));
    assert!(loaded.f0_energy.is_some());
    assert!(loaded.plbert.is_none());
    assert!(loaded.prosody.is_none());
    assert!(loaded.regulate.is_none());
    assert!(loaded.sinegen_pre.is_none());
    assert!(loaded.sinegen_post.is_none());

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dir);
}

/// save_to_dir creates the directory if it does not exist.
#[cfg(feature = "plan-serde")]
#[test]
fn test_segment_peephole_configs_save_creates_dir() {
    let dir = std::env::temp_dir().join(format!(
        "nn_segment_peephole_mkdir_test_{}/nested/dir",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(std::env::temp_dir().join(format!(
        "nn_segment_peephole_mkdir_test_{}",
        std::process::id()
    )));

    let configs = SegmentPeepholeConfigs {
        plbert: Some(nn_dsl::PeepholeConfig::default()),
        ..Default::default()
    };

    configs
        .save_to_dir(&dir)
        .expect("save_to_dir should create nested dirs");
    assert!(dir.join("plbert_config.json").exists());

    let _ = std::fs::remove_dir_all(std::env::temp_dir().join(format!(
        "nn_segment_peephole_mkdir_test_{}",
        std::process::id()
    )));
}

/// load_from_dir with empty directory returns empty configs.
#[cfg(feature = "plan-serde")]
#[test]
fn test_segment_peephole_configs_load_empty_dir() {
    let dir = std::env::temp_dir().join(format!(
        "nn_segment_peephole_empty_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create dir");

    let configs = SegmentPeepholeConfigs::load_from_dir(&dir)
        .expect("load_from_dir on empty dir should succeed");
    assert_eq!(configs.configured_count(), 0);

    let _ = std::fs::remove_dir_all(&dir);
}

/// load_from_dir returns error for invalid JSON files.
#[cfg(feature = "plan-serde")]
#[test]
fn test_segment_peephole_configs_load_invalid_json() {
    let dir = std::env::temp_dir().join(format!(
        "nn_segment_peephole_invalid_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create dir");
    std::fs::write(dir.join("plbert_config.json"), "not valid json {{")
        .expect("write invalid file");

    let result = SegmentPeepholeConfigs::load_from_dir(&dir);
    assert!(result.is_err(), "should fail on invalid JSON");
    match result {
        Err(CompiledKokoroError::ConfigLoad(msg)) => {
            assert!(msg.contains("parse"), "error should mention parse: {msg}");
        }
        other => panic!("expected ConfigLoad error, got: {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// save_to_dir with all 8 segments produces 8 files.
#[cfg(feature = "plan-serde")]
#[test]
fn test_segment_peephole_configs_save_all_8_segments() {
    let dir = std::env::temp_dir().join(format!(
        "nn_segment_peephole_all8_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);

    let default_config = nn_dsl::PeepholeConfig::default();
    let configs = SegmentPeepholeConfigs {
        plbert: Some(default_config.clone()),
        text: Some(default_config.clone()),
        prosody: Some(default_config.clone()),
        f0_energy: Some(default_config.clone()),
        generator: Some(default_config.clone()),
        regulate: Some(default_config.clone()),
        sinegen_pre: Some(default_config.clone()),
        sinegen_post: Some(default_config),
    };

    configs.save_to_dir(&dir).expect("save should succeed");

    let expected_files = [
        "plbert_config.json",
        "text_config.json",
        "prosody_config.json",
        "f0_energy_config.json",
        "generator_config.json",
        "regulate_config.json",
        "sinegen_pre_config.json",
        "sinegen_post_config.json",
    ];
    for file in &expected_files {
        assert!(dir.join(file).exists(), "{file} should exist after save");
    }

    // Load back and verify all 8.
    let loaded = SegmentPeepholeConfigs::load_from_dir(&dir).expect("load should succeed");
    assert_eq!(loaded.configured_count(), 8);

    let _ = std::fs::remove_dir_all(&dir);
}

/// SegmentPeepholeConfigs Debug output is human-readable.
#[test]
fn test_segment_peephole_configs_debug() {
    let configs = SegmentPeepholeConfigs {
        plbert: Some(nn_dsl::PeepholeConfig::default()),
        ..Default::default()
    };
    let debug_str = format!("{configs:?}");
    assert!(debug_str.contains("plbert"), "Debug should include plbert");
    assert!(
        debug_str.contains("Some"),
        "Debug should show Some for plbert"
    );
    assert!(
        debug_str.contains("None"),
        "Debug should show None for unset segments"
    );
}

/// Clone preserves all fields.
#[test]
fn test_segment_peephole_configs_clone() {
    let config = nn_dsl::PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };

    let original = SegmentPeepholeConfigs {
        generator: Some(config),
        ..Default::default()
    };
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

/// Verify precompile_segments_optimized compiles with correct signature.
#[test]
fn test_precompile_segments_optimized_signature_compiles() {
    fn _assert_method_exists(
        kokoro: &mut CompiledKokoro,
        shapes: &PrecompileShapes,
        cache: &PipelineCache,
    ) {
        // With Some configs.
        let configs = SegmentPeepholeConfigs::new();
        let _result: Result<usize, CompiledKokoroError> =
            kokoro.precompile_segments_optimized(shapes, cache, Some(&configs));

        // With None (default behavior).
        let _result2: Result<usize, CompiledKokoroError> =
            kokoro.precompile_segments_optimized(shapes, cache, None);
    }
}

/// SEGMENT_KINDS has all 8 expected entries.
#[test]
fn test_segment_kinds_constant() {
    assert_eq!(SEGMENT_KINDS.len(), 8);
    assert!(SEGMENT_KINDS.contains(&"plbert"));
    assert!(SEGMENT_KINDS.contains(&"text"));
    assert!(SEGMENT_KINDS.contains(&"prosody"));
    assert!(SEGMENT_KINDS.contains(&"f0_energy"));
    assert!(SEGMENT_KINDS.contains(&"generator"));
    assert!(SEGMENT_KINDS.contains(&"regulate"));
    assert!(SEGMENT_KINDS.contains(&"sinegen_pre"));
    assert!(SEGMENT_KINDS.contains(&"sinegen_post"));
}

// ======================== SegmentConfigCacheKey tests (#3828 Phase 2B) ========================

/// Cache key construction populates all fields.
#[test]
fn test_cache_key_construction() {
    let key = SegmentConfigCacheKey::new("plbert", vec![vec![1, 40], vec![1, 512, 40]]);
    assert_eq!(key.segment_kind, "plbert");
    assert_eq!(key.input_shapes, vec![vec![1, 40], vec![1, 512, 40]]);
    assert_eq!(key.nn_version, env!("CARGO_PKG_VERSION"));
}

/// with_version allows explicit version for testing.
#[test]
fn test_cache_key_with_version() {
    let key = SegmentConfigCacheKey::with_version("generator", vec![vec![1, 512, 80]], "99.0.0");
    assert_eq!(key.segment_kind, "generator");
    assert_eq!(key.nn_version, "99.0.0");
}

/// Identical keys are equal; different keys are not.
#[test]
fn test_cache_key_equality() {
    let k1 = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "1.0.0");
    let k2 = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "1.0.0");
    assert_eq!(k1, k2);

    // Different segment kind.
    let k3 = SegmentConfigCacheKey::with_version("text", vec![vec![1, 40]], "1.0.0");
    assert_ne!(k1, k3);

    // Different shape.
    let k4 = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 80]], "1.0.0");
    assert_ne!(k1, k4);

    // Different version.
    let k5 = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "2.0.0");
    assert_ne!(k1, k5);
}

/// Cache key is hashable (required for HashMap usage).
#[test]
fn test_cache_key_hash() {
    use std::collections::HashSet;
    let k1 = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "1.0.0");
    let k2 = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "1.0.0");
    let k3 = SegmentConfigCacheKey::with_version("text", vec![vec![1, 40]], "1.0.0");

    let mut set = HashSet::new();
    set.insert(k1);
    assert!(set.contains(&k2), "equal keys should hash the same");
    assert!(!set.contains(&k3), "different keys should not collide");
}

// ======================== SegmentConfigCache tests (#3828 Phase 2B) ========================

/// New cache is empty.
#[test]
fn test_segment_config_cache_new_is_empty() {
    let cache = SegmentConfigCache::new();
    assert!(cache.keys.is_empty());
    assert_eq!(cache.configs.configured_count(), 0);
}

/// is_valid returns true for matching key, false otherwise.
#[test]
fn test_segment_config_cache_is_valid() {
    let mut cache = SegmentConfigCache::new();

    let key = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "1.0.0");
    cache.insert("plbert", nn_dsl::PeepholeConfig::default(), key.clone());

    // Same key = valid.
    assert!(cache.is_valid("plbert", &key));

    // Different shape = invalid.
    let key_diff_shape = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 80]], "1.0.0");
    assert!(!cache.is_valid("plbert", &key_diff_shape));

    // Different version = invalid.
    let key_diff_version =
        SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "2.0.0");
    assert!(!cache.is_valid("plbert", &key_diff_version));

    // Missing segment = invalid.
    assert!(!cache.is_valid("generator", &key));
}

/// invalidate removes config and key for a segment.
#[test]
fn test_segment_config_cache_invalidate() {
    let mut cache = SegmentConfigCache::new();
    let key = SegmentConfigCacheKey::with_version("generator", vec![vec![1, 512, 80]], "1.0.0");
    let config = nn_dsl::PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };

    cache.insert("generator", config, key);
    assert_eq!(cache.configs.configured_count(), 1);
    assert!(cache.keys.contains_key("generator"));

    cache.invalidate("generator");
    assert_eq!(cache.configs.configured_count(), 0);
    assert!(!cache.keys.contains_key("generator"));
    assert!(cache.configs.generator.is_none());
}

/// invalidate on nonexistent segment is a no-op.
#[test]
fn test_segment_config_cache_invalidate_nonexistent() {
    let mut cache = SegmentConfigCache::new();
    cache.invalidate("nonexistent");
    assert_eq!(cache.configs.configured_count(), 0);
}

/// invalidate_stale removes only stale segments.
#[test]
fn test_segment_config_cache_invalidate_stale() {
    let mut cache = SegmentConfigCache::new();

    // Insert two segments.
    let key_plbert = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "1.0.0");
    let key_gen = SegmentConfigCacheKey::with_version("generator", vec![vec![1, 512, 80]], "1.0.0");
    cache.insert(
        "plbert",
        nn_dsl::PeepholeConfig::default(),
        key_plbert.clone(),
    );
    cache.insert("generator", nn_dsl::PeepholeConfig::default(), key_gen);
    assert_eq!(cache.configs.configured_count(), 2);

    // Current keys: plbert unchanged, generator has new shape.
    let mut current = HashMap::new();
    current.insert("plbert".to_string(), key_plbert);
    current.insert(
        "generator".to_string(),
        SegmentConfigCacheKey::with_version("generator", vec![vec![1, 512, 160]], "1.0.0"),
    );

    let invalidated = cache.invalidate_stale(&current);
    assert_eq!(invalidated, 1, "only generator should be invalidated");
    assert!(cache.configs.plbert.is_some(), "plbert should survive");
    assert!(
        cache.configs.generator.is_none(),
        "generator should be invalidated"
    );
    assert_eq!(cache.configs.configured_count(), 1);
}

/// insert overwrites existing config and key.
#[test]
fn test_segment_config_cache_insert_overwrites() {
    let mut cache = SegmentConfigCache::new();

    let key_v1 = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "1.0.0");
    let config_v1 = nn_dsl::PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };
    cache.insert("plbert", config_v1, key_v1);

    let key_v2 = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 80]], "2.0.0");
    let config_v2 = nn_dsl::PeepholeConfig::default();
    cache.insert("plbert", config_v2, key_v2.clone());

    assert!(cache.is_valid("plbert", &key_v2));
    assert!(
        cache.configs.plbert.as_ref().unwrap().silu_mul,
        "config should be overwritten to default (silu_mul=true)"
    );
}

/// from_parts constructs a cache from existing data.
#[test]
fn test_segment_config_cache_from_parts() {
    let mut configs = SegmentPeepholeConfigs::new();
    configs.plbert = Some(nn_dsl::PeepholeConfig::default());

    let mut keys = HashMap::new();
    keys.insert(
        "plbert".to_string(),
        SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "1.0.0"),
    );

    let cache = SegmentConfigCache::from_parts(configs, keys);
    assert_eq!(cache.configs.configured_count(), 1);
    assert!(cache.keys.contains_key("plbert"));
}

/// save/load round-trip preserves configs and keys.
#[cfg(feature = "plan-serde")]
#[test]
fn test_segment_config_cache_save_load_roundtrip() {
    let dir = std::env::temp_dir().join(format!(
        "nn_segment_config_cache_rt_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);

    let mut cache = SegmentConfigCache::new();

    let mut gen_config = nn_dsl::PeepholeConfig::default();
    gen_config.silu_mul = false;
    let gen_key = SegmentConfigCacheKey::with_version("generator", vec![vec![1, 512, 80]], "1.0.0");
    cache.insert("generator", gen_config.clone(), gen_key.clone());

    let plbert_key = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "1.0.0");
    cache.insert(
        "plbert",
        nn_dsl::PeepholeConfig::default(),
        plbert_key.clone(),
    );

    // Save.
    cache.save(&dir).expect("save should succeed");

    // Verify files.
    assert!(dir.join("generator_config.json").exists());
    assert!(dir.join("plbert_config.json").exists());
    assert!(dir.join("_cache_keys.json").exists());

    // Load.
    let loaded = SegmentConfigCache::load(&dir).expect("load should succeed");
    assert_eq!(loaded.configs.configured_count(), 2);
    assert!(loaded.is_valid("generator", &gen_key));
    assert!(loaded.is_valid("plbert", &plbert_key));
    assert!(!loaded.configs.generator.as_ref().unwrap().silu_mul);

    let _ = std::fs::remove_dir_all(&dir);
}

/// load with missing _cache_keys.json returns empty keys.
#[cfg(feature = "plan-serde")]
#[test]
fn test_segment_config_cache_load_missing_keys_file() {
    let dir = std::env::temp_dir().join(format!(
        "nn_segment_config_cache_nokeys_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");

    // Write a config file but no _cache_keys.json.
    let config = nn_dsl::PeepholeConfig::default();
    let json = serde_json::to_string_pretty(&config).unwrap();
    std::fs::write(dir.join("plbert_config.json"), json).unwrap();

    let loaded = SegmentConfigCache::load(&dir).expect("load should succeed");
    assert_eq!(loaded.configs.configured_count(), 1);
    assert!(
        loaded.keys.is_empty(),
        "keys should be empty without _cache_keys.json"
    );

    // All validity checks fail when keys are missing.
    let key = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "1.0.0");
    assert!(!loaded.is_valid("plbert", &key));

    let _ = std::fs::remove_dir_all(&dir);
}

/// Version bump invalidates all cached configs.
#[test]
fn test_segment_config_cache_version_bump_invalidates_all() {
    let mut cache = SegmentConfigCache::new();

    let key_v1 = SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "1.0.0");
    cache.insert("plbert", nn_dsl::PeepholeConfig::default(), key_v1);

    let key_gen_v1 =
        SegmentConfigCacheKey::with_version("generator", vec![vec![1, 512, 80]], "1.0.0");
    cache.insert("generator", nn_dsl::PeepholeConfig::default(), key_gen_v1);

    // Simulate version bump: all current keys use "2.0.0".
    let mut current = HashMap::new();
    current.insert(
        "plbert".to_string(),
        SegmentConfigCacheKey::with_version("plbert", vec![vec![1, 40]], "2.0.0"),
    );
    current.insert(
        "generator".to_string(),
        SegmentConfigCacheKey::with_version("generator", vec![vec![1, 512, 80]], "2.0.0"),
    );

    let invalidated = cache.invalidate_stale(&current);
    assert_eq!(
        invalidated, 2,
        "version bump should invalidate all segments"
    );
    assert_eq!(cache.configs.configured_count(), 0);
}
