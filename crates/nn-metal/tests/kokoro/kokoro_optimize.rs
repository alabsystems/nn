// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro optimizer integration tests.
//!
//! Tests that the self-optimizing compiler's PeepholeConfig search integrates
//! correctly with the Kokoro pipeline. Validates that:
//!
//! 1. `segment_gap_analysis()` produces baseline dispatch counts for all segments.
//! 2. `optimize_plan_with_cost()` runs successfully on empty and simple graphs.
//! 3. The optimizer never produces more dispatches than baseline.
//!
//! Part of Phase 2A (#3828).

use std::time::Duration;

use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_dsl::{optimize_plan_with_cost, CostModel};

/// Test that `optimize_plan_with_cost` runs successfully on an empty graph
/// and returns 0 dispatches with at least 1 config explored (baseline).
#[test]
fn test_optimize_empty_graph_returns_zero_dispatches() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let result = optimize_plan_with_cost(&graph, &CostModel::apple_m4(), Duration::from_secs(1))
        .expect("empty graph optimization should succeed");

    assert_eq!(result.dispatch_count, 0);
    assert_eq!(result.baseline_dispatch_count, 0);
    assert!(
        result.configs_explored >= 1,
        "should explore at least the baseline config"
    );
    eprintln!(
        "Empty graph: {} configs explored, {} dispatches",
        result.configs_explored, result.dispatch_count
    );
}

/// Test that `optimize_plan_with_cost` with zero budget returns the baseline.
#[test]
fn test_optimize_zero_budget_returns_baseline() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let result = optimize_plan_with_cost(&graph, &CostModel::apple_m4(), Duration::ZERO)
        .expect("zero-budget optimization should succeed");

    // With zero budget, only the baseline is compiled.
    assert_eq!(result.configs_explored, 1);
    assert_eq!(result.dispatch_count, 0);
    assert_eq!(result.baseline_dispatch_count, 0);
}

/// Test that the optimizer runs on a simple traced computation graph
/// (relu -> tanh chain) and never regresses dispatch count.
#[test]
fn test_optimize_simple_traced_graph_no_regression() {
    use nn_core::dyn_tensor::trace::trace_graph;

    let x = DynTensor::zeros(&[1, 8, 16], DType::F32, &Device::Cpu).unwrap();
    let (output, graph): (DynTensor, ComputationGraph) = trace_graph(|| {
        let y = x.relu()?;
        let z = y.tanh()?;
        Ok(z)
    })
    .unwrap();

    let cost_model = CostModel::apple_m4();
    let result = optimize_plan_with_cost(&graph, &cost_model, Duration::from_secs(5))
        .expect("simple graph optimization should succeed");

    // Optimizer should never make things worse.
    assert!(
        result.dispatch_count <= result.baseline_dispatch_count,
        "optimized dispatches ({}) should not exceed baseline ({})",
        result.dispatch_count,
        result.baseline_dispatch_count,
    );

    eprintln!("\n=== SIMPLE GRAPH OPTIMIZATION ===");
    eprintln!("{}", result.summarize());

    // Output tensor should be valid.
    assert_eq!(output.dtype(), DType::F32);
}

/// Test that `segment_gap_analysis()` produces baseline dispatch counts for
/// all traceable Kokoro segments, and that `theoretical_minimum <= dispatch_count`
/// for every segment.
///
/// This test builds a miniaturized Kokoro, synthesizes once to compile all
/// segments, then runs gap analysis. The results serve as the baseline that
/// the optimizer aims to improve.
#[test]
fn test_kokoro_segment_baseline_dispatch_counts() {
    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let (input_ids, style) = super::kokoro_gates::test_inputs();

    // Synthesize to compile all segments and populate caches.
    // Miniaturized weights produce near-zero audio that fails hard bounds
    // verification, so we ignore the synthesis error — the compiled segments
    // are populated as a side effect regardless.
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache);

    // Run gap analysis on all segments.
    let results = kokoro
        .segment_gap_analysis(&input_ids, &style, 1.0, &cache)
        .unwrap();

    eprintln!("\n=== KOKORO SEGMENT BASELINE (for optimizer comparison) ===");
    eprintln!(
        "{:<16} {:>10} {:>12} {:>10}",
        "Segment", "Dispatches", "Theo. Min", "Gap"
    );
    eprintln!("{}", "-".repeat(52));

    let mut total_dispatches = 0usize;
    let mut total_theoretical = 0usize;

    for seg in &results {
        let gap = seg.dispatch_count.saturating_sub(seg.theoretical_minimum);
        eprintln!(
            "{:<16} {:>10} {:>12} {:>10}",
            seg.segment_name, seg.dispatch_count, seg.theoretical_minimum, gap,
        );
        total_dispatches += seg.dispatch_count;
        total_theoretical += seg.theoretical_minimum;
    }

    let total_gap = total_dispatches.saturating_sub(total_theoretical);
    eprintln!("{}", "-".repeat(52));
    eprintln!(
        "{:<16} {:>10} {:>12} {:>10}",
        "TOTAL", total_dispatches, total_theoretical, total_gap,
    );
    eprintln!("Segments analyzed: {}", results.len());
    eprintln!("=============================================\n");

    // At least 5 segments should be analyzable.
    assert!(
        results.len() >= 5,
        "Expected at least 5 analyzed segments, got {}",
        results.len(),
    );

    // Sanity: theoretical_minimum <= dispatch_count for every segment.
    for seg in &results {
        assert!(
            seg.theoretical_minimum <= seg.dispatch_count,
            "Segment {}: theoretical_minimum ({}) > dispatch_count ({})",
            seg.segment_name,
            seg.theoretical_minimum,
            seg.dispatch_count,
        );
    }

    // Sanity: total dispatches should be > 0.
    assert!(
        total_dispatches > 0,
        "Total dispatches across all segments is 0 -- analysis may be broken",
    );
}

/// Test that `optimize_plan_with_cost` on a matmul-containing traced graph
/// produces valid results and never regresses.
#[test]
fn test_optimize_matmul_graph_no_regression() {
    use nn_core::dyn_tensor::trace::trace_graph;

    let a = DynTensor::ones(&[1, 4, 8], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::ones(&[1, 8, 4], DType::F32, &Device::Cpu).unwrap();

    let (_output, graph): (DynTensor, ComputationGraph) = trace_graph(|| {
        let c = a.matmul(&b)?;
        let d = c.relu()?;
        Ok(d)
    })
    .unwrap();

    let cost_model = CostModel::apple_m4();
    let result = optimize_plan_with_cost(&graph, &cost_model, Duration::from_secs(5))
        .expect("matmul graph optimization should succeed");

    assert!(
        result.dispatch_count <= result.baseline_dispatch_count,
        "optimized dispatches ({}) should not exceed baseline ({})",
        result.dispatch_count,
        result.baseline_dispatch_count,
    );

    eprintln!("\n=== MATMUL GRAPH OPTIMIZATION ===");
    eprintln!("{}", result.summarize());
}

/// Test that `OptimizationResult::summarize()` produces human-readable output.
#[test]
fn test_optimization_result_summarize_format() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let result = optimize_plan_with_cost(&graph, &CostModel::apple_m4(), Duration::from_secs(1))
        .expect("optimization should succeed");

    let summary = result.summarize();
    assert!(
        summary.contains("Optimization result"),
        "summary should contain header"
    );
    assert!(
        summary.contains("Configs explored"),
        "summary should mention configs explored"
    );
    eprintln!("\n{summary}");
}

/// Run PeepholeConfig optimizer search on all miniaturized Kokoro segments.
///
/// Uses `segment_optimizer_search()` to trace each segment and exhaustively
/// search 2048 PeepholeConfig combinations. Reports per-segment improvements
/// and verifies that the optimizer never regresses dispatch count.
///
/// Part of #3828 Phase 2C.
#[test]
fn test_kokoro_segment_optimizer_search() {
    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let (input_ids, style) = super::kokoro_gates::test_inputs();

    // Synthesize to compile all segments (ignore verification failure on mini weights).
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache);

    // Run optimizer search with 10s budget per segment.
    let results = kokoro
        .segment_optimizer_search(&input_ids, &style, 1.0, &cache, Duration::from_secs(10))
        .unwrap();

    eprintln!("\n=== KOKORO SEGMENT OPTIMIZER SEARCH (miniaturized) ===");
    eprintln!(
        "{:<16} {:>10} {:>10} {:>10} {:>8}",
        "Segment", "Baseline", "Optimized", "Saved", "Configs"
    );
    eprintln!("{}", "-".repeat(60));

    let mut total_baseline = 0usize;
    let mut total_optimized = 0usize;

    for seg in &results {
        let opt = &seg.optimization;
        let saved = opt
            .baseline_dispatch_count
            .saturating_sub(opt.dispatch_count);
        eprintln!(
            "{:<16} {:>10} {:>10} {:>10} {:>8}",
            seg.segment_name,
            opt.baseline_dispatch_count,
            opt.dispatch_count,
            saved,
            opt.configs_explored,
        );
        total_baseline += opt.baseline_dispatch_count;
        total_optimized += opt.dispatch_count;

        // Log the optimal config if different from default.
        if saved > 0 {
            eprintln!("  -> Best config: {:?}", opt.config);
        }
    }

    let total_saved = total_baseline.saturating_sub(total_optimized);
    eprintln!("{}", "-".repeat(60));
    eprintln!(
        "{:<16} {:>10} {:>10} {:>10}",
        "TOTAL", total_baseline, total_optimized, total_saved,
    );
    eprintln!("Segments analyzed: {}", results.len());
    eprintln!("====================================================\n");

    // At least 5 segments should be optimizable.
    assert!(
        results.len() >= 5,
        "Expected at least 5 optimized segments, got {}",
        results.len(),
    );

    // Optimizer must never regress: optimized <= baseline for every segment.
    for seg in &results {
        assert!(
            seg.optimization.dispatch_count <= seg.optimization.baseline_dispatch_count,
            "Segment {}: optimizer regressed! {} > {} dispatches",
            seg.segment_name,
            seg.optimization.dispatch_count,
            seg.optimization.baseline_dispatch_count,
        );
    }

    // Total optimized dispatches must not exceed total baseline.
    assert!(
        total_optimized <= total_baseline,
        "Total optimizer regression: {total_optimized} > {total_baseline}",
    );
}

/// Run `optimize_plan_with_cost` on miniaturized Kokoro segments via
/// `segment_optimizer_search`. For each of the 8 segments, prints baseline
/// vs optimized dispatch counts and the best PeepholeConfig found.
///
/// This test calls `optimize_plan_with_cost` indirectly through
/// `segment_optimizer_search`, which traces each segment and feeds the
/// resulting `ComputationGraph` directly to the optimizer. Direct graph
/// access is not possible from tests because segment trace functions are
/// `pub(super)` on `CompiledKokoro`.
///
/// Part of #3828.
#[test]
fn test_optimize_miniaturized_kokoro() {
    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let (input_ids, style) = super::kokoro_gates::test_inputs();

    // Synthesize once to populate segment caches.
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache);

    // Run optimizer search with 5-second budget per segment.
    let results = kokoro
        .segment_optimizer_search(&input_ids, &style, 1.0, &cache, Duration::from_secs(5))
        .unwrap();

    eprintln!("\n=== MINIATURIZED KOKORO OPTIMIZER (optimize_plan_with_cost) ===");
    eprintln!(
        "{:<16} {:>10} {:>10} {:>8} {:>12} {:>12}",
        "Segment", "Baseline", "Optimized", "Saved", "Base Cost", "Best Cost"
    );
    eprintln!("{}", "-".repeat(74));

    let mut total_baseline = 0usize;
    let mut total_optimized = 0usize;

    for seg in &results {
        let opt = &seg.optimization;
        let saved = opt
            .baseline_dispatch_count
            .saturating_sub(opt.dispatch_count);
        eprintln!(
            "{:<16} {:>10} {:>10} {:>8} {:>10.0}us {:>10.0}us",
            seg.segment_name,
            opt.baseline_dispatch_count,
            opt.dispatch_count,
            saved,
            opt.baseline_cost_ns / 1e3,
            opt.best_cost_ns / 1e3,
        );
        total_baseline += opt.baseline_dispatch_count;
        total_optimized += opt.dispatch_count;
    }

    let total_saved = total_baseline.saturating_sub(total_optimized);
    eprintln!("{}", "-".repeat(74));
    eprintln!(
        "{:<16} {:>10} {:>10} {:>8}",
        "TOTAL", total_baseline, total_optimized, total_saved,
    );

    // Print best PeepholeConfig for each segment that improved.
    eprintln!("\n--- Best PeepholeConfig per improved segment ---");
    for seg in &results {
        let opt = &seg.optimization;
        if opt.dispatch_count < opt.baseline_dispatch_count {
            eprintln!("  {}: {:?}", seg.segment_name, opt.config);
        }
    }
    eprintln!("=================================================================\n");

    // Verify: at least 5 segments optimizable (miniaturized model has 8).
    assert!(
        results.len() >= 5,
        "Expected at least 5 optimized segments, got {}",
        results.len(),
    );

    // Assert: optimized_count <= baseline_count for all segments.
    for seg in &results {
        assert!(
            seg.optimization.dispatch_count <= seg.optimization.baseline_dispatch_count,
            "Segment {}: optimizer regressed! optimized {} > baseline {}",
            seg.segment_name,
            seg.optimization.dispatch_count,
            seg.optimization.baseline_dispatch_count,
        );
    }

    // Assert: total optimized <= total baseline.
    assert!(
        total_optimized <= total_baseline,
        "Total optimizer regression: {total_optimized} > {total_baseline}",
    );
}

/// Test that Kokoro compiles successfully with all PeepholeConfig fields
/// disabled (all false). This is the worst-case config with no fusion passes
/// and should produce more dispatches than default, but must still be valid.
///
/// Part of #3828.
#[test]
fn test_all_disabled_peephole_config_compiles() {
    use std::collections::HashMap;

    let all_disabled = nn_dsl::PeepholeConfig {
        norm_activ_conv1d: false,
        fused_resblock: false,
        linear_activation: false,
        add_layer_norm: false,
        norm_linear: false,
        attention_transpose: false,
        flip_lstm: false,
        batched_linear_projection: false,
        channels_first_layer_norm: false,
        silu_mul: false,
        auto_fuse_elementwise: false,
        bilstm_cat: false,
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
        fuse_activation_conv1d: false,
        fuse_resblock_chain: false,
    };

    // Apply all-disabled config to every segment.
    let segment_names = [
        "plbert",
        "text",
        "prosody",
        "f0",
        "regulate",
        "generator",
        "sinegen_pre",
        "sinegen_post",
    ];
    let configs: HashMap<String, nn_dsl::PeepholeConfig> = segment_names
        .iter()
        .map(|name| (name.to_string(), all_disabled.clone()))
        .collect();

    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let kokoro_default_dispatches = {
        let (input_ids, style) = super::kokoro_gates::test_inputs();
        let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache);
        kokoro.total_dispatches()
    };

    // Rebuild with all-disabled config. Need a fresh instance to clear
    // compiled segment caches — peephole configs only affect new compilations.
    let (kokoro_base, cache) = super::kokoro_gates::build_kokoro();
    let mut kokoro_disabled = kokoro_base.with_peephole_configs(configs);

    let (input_ids, style) = super::kokoro_gates::test_inputs();
    // Synthesize with all-disabled config — must succeed (more dispatches, but valid).
    let result = kokoro_disabled.synthesize(&input_ids, &style, 1.0, &cache);
    assert!(
        result.is_ok(),
        "Kokoro with all-disabled PeepholeConfig should still compile and run, got: {:?}",
        result.err(),
    );

    let disabled_dispatches = kokoro_disabled.total_dispatches();

    eprintln!("\n=== ALL-DISABLED PEEPHOLE CONFIG ===");
    eprintln!("  Default config dispatches:      {kokoro_default_dispatches}");
    eprintln!("  All-disabled config dispatches:  {disabled_dispatches}");
    let diff = disabled_dispatches.saturating_sub(kokoro_default_dispatches);
    eprintln!("  Difference:                     +{diff} dispatches (fusion passes disabled)");
    eprintln!("=====================================\n");

    // All-disabled should produce >= default dispatches since fusion passes
    // are disabled. It may produce the same count if the miniaturized model
    // doesn't have fusible patterns that the disabled passes would catch.
    assert!(
        disabled_dispatches >= kokoro_default_dispatches,
        "All-disabled config ({disabled_dispatches}) should have >= dispatches \
         than default ({kokoro_default_dispatches})",
    );
}

/// Production Kokoro optimizer search using real weights from safetensors.
///
/// Gated behind KOKORO_WEIGHTS env var. Loads production Kokoro, traces all
/// segments, and runs PeepholeConfig optimizer search. Reports per-segment
/// improvements with production-scale graph structures.
///
/// All 8 segments should trace successfully. The optimizer converts
/// intermediate GPU tensors to the model device before tracing so that
/// forward passes inside `trace_graph` don't hit device-mismatch errors
/// (CPU model weights vs GPU intermediates). (#4250)
///
/// Part of #3828 Phase 2C.
#[test]
fn test_production_kokoro_optimizer_search() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "production optimizer search not run.",
    ) {
        Some(p) => p,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Use Warn policy: test tokens produce click artifacts with production
    // weights that fail the no_clicks hard bound. Part of #4262.
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("failed to load Kokoro weights")
    };

    // Synthesize to compile all segments. Ignore verification failure --
    // we only need compiled segments populated as a side effect.
    let input_ids = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 8],
        &Device::Cpu,
    )
    .unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &Device::Cpu).unwrap();
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache);

    // Run optimizer search with 10s budget per segment.
    let results = kokoro
        .segment_optimizer_search(&input_ids, &style, 1.0, &cache, Duration::from_secs(10))
        .unwrap();

    eprintln!("\n=== PRODUCTION KOKORO OPTIMIZER SEARCH ===");
    eprintln!(
        "{:<16} {:>10} {:>10} {:>10} {:>12} {:>12} {:>8}",
        "Segment", "Baseline", "Optimized", "Saved", "Base Cost", "Best Cost", "Configs"
    );
    eprintln!("{}", "-".repeat(84));

    let mut total_baseline = 0usize;
    let mut total_optimized = 0usize;
    let mut total_base_cost = 0.0_f64;
    let mut total_best_cost = 0.0_f64;

    for seg in &results {
        let opt = &seg.optimization;
        let saved = opt
            .baseline_dispatch_count
            .saturating_sub(opt.dispatch_count);
        eprintln!(
            "{:<16} {:>10} {:>10} {:>10} {:>10.0}us {:>10.0}us {:>8}",
            seg.segment_name,
            opt.baseline_dispatch_count,
            opt.dispatch_count,
            saved,
            opt.baseline_cost_ns / 1e3,
            opt.best_cost_ns / 1e3,
            opt.configs_explored,
        );
        total_baseline += opt.baseline_dispatch_count;
        total_optimized += opt.dispatch_count;
        total_base_cost += opt.baseline_cost_ns;
        total_best_cost += opt.best_cost_ns;

        if saved > 0 {
            eprintln!("  -> Best config: {:?}", opt.config);
        }
    }

    let total_saved = total_baseline.saturating_sub(total_optimized);
    let cost_reduction = if total_base_cost > 0.0 {
        (total_base_cost - total_best_cost) / total_base_cost * 100.0
    } else {
        0.0
    };
    eprintln!("{}", "-".repeat(84));
    eprintln!(
        "{:<16} {:>10} {:>10} {:>10} {:>10.0}us {:>10.0}us",
        "TOTAL",
        total_baseline,
        total_optimized,
        total_saved,
        total_base_cost / 1e3,
        total_best_cost / 1e3,
    );
    eprintln!(
        "Cost reduction: {cost_reduction:.1}%, Segments: {}",
        results.len()
    );
    eprintln!("=============================================\n");

    // All 8 segments should be optimizable now that trace inputs are
    // converted to the model device before tracing. (#4250)
    assert!(
        results.len() >= 8,
        "Expected all 8 optimized production segments, got {}",
        results.len(),
    );

    // Optimizer must never regress.
    for seg in &results {
        assert!(
            seg.optimization.dispatch_count <= seg.optimization.baseline_dispatch_count,
            "Production segment {}: optimizer regressed! {} > {} dispatches",
            seg.segment_name,
            seg.optimization.dispatch_count,
            seg.optimization.baseline_dispatch_count,
        );
    }
}
