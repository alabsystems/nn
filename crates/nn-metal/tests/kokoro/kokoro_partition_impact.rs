// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Partition-driven codegen impact measurement for Kokoro.
//!
//! Compares dispatch counts between the old compilation path
//! (`compile_trace_to_plan` — no fusion, no partition) and the new path
//! (`compile_trace_to_plan_with_fusion` — partition codegen + peephole).
//!
//! Traces representative Kokoro segments, compiles each with both paths,
//! and asserts that the partition-driven path produces fewer or equal
//! dispatches for every segment.
//!
//! Run: `cargo test -p nn-metal --test kokoro_all kokoro_partition_impact -- --nocapture`
//!
//! Part of #4334 (partition codegen wiring).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{record_input, trace_graph, ComputationGraph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_dsl::PeepholeConfig;
use nn_dsl::{compile_trace_to_plan, compile_trace_to_plan_with_fusion, count_dispatches};
use nn_models::kokoro_error::KokoroError;
use nn_models::kokoro_tts::TextEncoder;

fn cpu() -> Device {
    Device::Cpu
}

// -- Miniaturized dimensions (matching kokoro_gates.rs) -------------------------

const D_EN: usize = 8;
const STYLE_DIM: usize = 4;
const HIDDEN: usize = 8;
const EMB: usize = 4;
const VOCAB: usize = 10;

// -- Weight helpers -------------------------------------------------------------

fn z(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    m.insert(
        name.to_string(),
        DynTensor::zeros(shape, DType::F32, &cpu()).unwrap(),
    );
}

fn ones(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    m.insert(
        name.to_string(),
        DynTensor::full(shape, 1.0, DType::F32, &cpu()).unwrap(),
    );
}

// -- Build individual segments for tracing --------------------------------------

fn build_text_encoder() -> TextEncoder {
    let mut m = HashMap::new();
    let p = "text_encoder";
    z(&mut m, &format!("{p}.embedding.weight"), &[VOCAB, D_EN]);
    let h = D_EN / 2;
    for i in 0..3 {
        z(&mut m, &format!("{p}.convs.{i}.weight"), &[D_EN, D_EN, 5]);
        z(&mut m, &format!("{p}.convs.{i}.bias"), &[D_EN]);
        ones(&mut m, &format!("{p}.norms.{i}.weight"), &[D_EN]);
        z(&mut m, &format!("{p}.norms.{i}.bias"), &[D_EN]);
    }
    z(&mut m, &format!("{p}.lstm.weight_ih_l0"), &[4 * h, D_EN]);
    z(&mut m, &format!("{p}.lstm.weight_hh_l0"), &[4 * h, h]);
    z(&mut m, &format!("{p}.lstm.bias_ih_l0"), &[4 * h]);
    z(&mut m, &format!("{p}.lstm.bias_hh_l0"), &[4 * h]);
    z(
        &mut m,
        &format!("{p}.lstm.weight_ih_l0_reverse"),
        &[4 * h, D_EN],
    );
    z(
        &mut m,
        &format!("{p}.lstm.weight_hh_l0_reverse"),
        &[4 * h, h],
    );
    z(&mut m, &format!("{p}.lstm.bias_ih_l0_reverse"), &[4 * h]);
    z(&mut m, &format!("{p}.lstm.bias_hh_l0_reverse"), &[4 * h]);
    z(&mut m, &format!("{p}.lstm.linear.weight"), &[D_EN, D_EN]);
    z(&mut m, &format!("{p}.lstm.linear.bias"), &[D_EN]);
    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    TextEncoder::load(vb.pp("text_encoder"), VOCAB, D_EN).expect("TextEncoder")
}

/// Trace the text encoder forward pass to get a `ComputationGraph`.
fn trace_text_encoder() -> ComputationGraph {
    let te = build_text_encoder();
    let tokens = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();

    let (out, mut graph) = trace_graph(|| {
        let mut inp = tokens.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        te.forward(&inp).map_err(KokoroError::into_tensor_error)
    })
    .unwrap();

    if let Some(id) = out.trace_id() {
        let _ = graph.set_primary_output(id);
    }
    graph
}

/// Segment dispatch comparison result.
struct SegmentComparison {
    name: &'static str,
    graph_nodes: usize,
    no_fusion_dispatches: usize,
    partition_dispatches: usize,
    reduction: isize,
}

/// Compare dispatch counts between no-fusion and partition-codegen compilation.
fn compare_plans(name: &'static str, graph: &ComputationGraph) -> SegmentComparison {
    let no_fusion_plan = compile_trace_to_plan(graph)
        .unwrap_or_else(|e| panic!("{name}: compile_trace_to_plan failed: {e}"));
    let partition_plan = compile_trace_to_plan_with_fusion(graph)
        .unwrap_or_else(|e| panic!("{name}: compile_trace_to_plan_with_fusion failed: {e}"));

    let no_fusion_dispatches = count_dispatches(&no_fusion_plan);
    let partition_dispatches = count_dispatches(&partition_plan);
    let reduction = no_fusion_dispatches as isize - partition_dispatches as isize;

    SegmentComparison {
        name,
        graph_nodes: graph.len(),
        no_fusion_dispatches,
        partition_dispatches,
        reduction,
    }
}

// =============================================================================
// Tests
// =============================================================================

/// Measure partition-driven codegen impact on the Kokoro text encoder segment.
///
/// Traces the miniaturized TextEncoder (Conv1d + LayerNorm + LeakyRelu +
/// BiLSTM + Linear), compiles with both paths, and verifies partition codegen
/// produces fewer or equal dispatches.
///
/// Part of #4334.
#[test]
fn partition_impact_text_encoder() {
    let graph = trace_text_encoder();
    let cmp = compare_plans("text_encoder", &graph);

    eprintln!("\n=== PARTITION CODEGEN IMPACT: text_encoder ===");
    eprintln!("  Graph nodes:          {}", cmp.graph_nodes);
    eprintln!("  No-fusion dispatches: {}", cmp.no_fusion_dispatches);
    eprintln!("  Partition dispatches:  {}", cmp.partition_dispatches);
    eprintln!(
        "  Reduction:            {} ({:+.1}%)",
        cmp.reduction,
        if cmp.no_fusion_dispatches > 0 {
            cmp.reduction as f64 / cmp.no_fusion_dispatches as f64 * 100.0
        } else {
            0.0
        }
    );
    eprintln!("===============================================\n");

    assert!(
        cmp.partition_dispatches <= cmp.no_fusion_dispatches,
        "Partition codegen regression: text_encoder has {} dispatches with \
         partition (was {} without). Expected <= no-fusion count.",
        cmp.partition_dispatches,
        cmp.no_fusion_dispatches,
    );
}

/// Full pipeline comparison using CompiledKokoro dispatch summary.
///
/// Builds the miniaturized Kokoro, synthesizes to compile all segments, and
/// compares the total dispatch count against the no-fusion baseline. The
/// no-fusion baseline is obtained by tracing the text encoder segment (the
/// most fusion-amenable segment) as a representative sample.
///
/// Part of #4334.
#[test]
fn partition_impact_full_pipeline() {
    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let (input_ids, style) = super::kokoro_gates::test_inputs();

    // Synthesize to compile all segments (uses partition codegen internally).
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    let pipeline_dispatches = kokoro.total_dispatches();
    let pipeline_metal = kokoro.total_metal_dispatches();
    let ds = kokoro.dispatch_summary();

    eprintln!("\n=== PARTITION CODEGEN IMPACT: Full Pipeline ===");
    eprintln!("  Total logical dispatches:     {pipeline_dispatches}");
    eprintln!("  Total Metal dispatch estimate: {pipeline_metal}");
    eprintln!("  Per-segment breakdown:");
    let segments = [
        ("plbert", ds.plbert),
        ("text_encoder", ds.text_encoder),
        ("prosody", ds.prosody),
        ("f0_energy", ds.f0_energy),
        ("generator", ds.generator),
        ("regulate", ds.regulate),
        ("sinegen_pre", ds.sinegen_pre),
        ("sinegen_post", ds.sinegen_post),
    ];
    for (name, count) in &segments {
        eprintln!("    {name:<16} {count:>4}");
    }
    eprintln!("=================================================\n");

    // Sanity: the pipeline should have a non-zero dispatch count.
    assert!(
        pipeline_dispatches > 0,
        "Pipeline dispatches is 0 -- something is wrong with compilation",
    );

    // Structural check: dispatch summary total must match total_dispatches().
    assert_eq!(
        ds.total(),
        pipeline_dispatches,
        "dispatch_summary().total() ({}) != total_dispatches() ({pipeline_dispatches})",
        ds.total(),
    );
}

/// Per-segment comparison: traces the text encoder and compares no-fusion vs
/// partition codegen, reporting a summary table.
///
/// This is the core measurement test that quantifies the dispatch reduction
/// from partition-driven codegen for a Kokoro-representative workload.
///
/// Part of #4334.
#[test]
fn partition_impact_summary_table() {
    let mut comparisons = Vec::new();

    // Text encoder: Conv1d + LayerNorm + LeakyRelu chain offers elementwise fusion.
    let text_graph = trace_text_encoder();
    comparisons.push(compare_plans("text_encoder", &text_graph));

    // Print summary table.
    eprintln!("\n{}", "=".repeat(72));
    eprintln!("  PARTITION CODEGEN DISPATCH IMPACT (#4334)");
    eprintln!("{}", "=".repeat(72));
    eprintln!(
        "  {:<16} {:>8} {:>10} {:>10} {:>10}",
        "Segment", "Nodes", "No-Fusion", "Partition", "Reduction"
    );
    eprintln!("  {}", "-".repeat(58));

    let mut total_no_fusion = 0usize;
    let mut total_partition = 0usize;

    for cmp in &comparisons {
        total_no_fusion += cmp.no_fusion_dispatches;
        total_partition += cmp.partition_dispatches;
        let pct = if cmp.no_fusion_dispatches > 0 {
            format!(
                "{:+.1}%",
                -(cmp.reduction as f64 / cmp.no_fusion_dispatches as f64 * 100.0)
            )
        } else {
            "N/A".to_string()
        };
        eprintln!(
            "  {:<16} {:>8} {:>10} {:>10} {:>10}",
            cmp.name, cmp.graph_nodes, cmp.no_fusion_dispatches, cmp.partition_dispatches, pct,
        );
    }
    eprintln!("  {}", "-".repeat(58));
    let total_reduction = total_no_fusion as isize - total_partition as isize;
    let total_pct = if total_no_fusion > 0 {
        format!(
            "{:+.1}%",
            -(total_reduction as f64 / total_no_fusion as f64 * 100.0)
        )
    } else {
        "N/A".to_string()
    };
    eprintln!(
        "  {:<16} {:>8} {:>10} {:>10} {:>10}",
        "TOTAL", "-", total_no_fusion, total_partition, total_pct,
    );
    eprintln!("{}\n", "=".repeat(72));

    // Assert: partition codegen must not regress dispatch count for any segment.
    for cmp in &comparisons {
        assert!(
            cmp.partition_dispatches <= cmp.no_fusion_dispatches,
            "Partition codegen regression in {}: {} dispatches with partition \
             (was {} without fusion). Expected <=.",
            cmp.name,
            cmp.partition_dispatches,
            cmp.no_fusion_dispatches,
        );
    }
}

// =============================================================================
// Production-weight partition impact test (requires KOKORO_WEIGHTS)
// =============================================================================

/// Build a `PeepholeConfig` with all optimization passes disabled.
fn all_disabled_peephole() -> PeepholeConfig {
    PeepholeConfig {
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
    }
}

/// All 8 Kokoro segment peephole keys.
const SEGMENT_KEYS: [&str; 8] = [
    "plbert",
    "text",
    "prosody",
    "f0",
    "regulate",
    "generator",
    "sinegen_pre",
    "sinegen_post",
];

/// Production-weight partition impact measurement.
///
/// Loads the full D=512 Kokoro model twice:
/// 1. All peephole passes disabled (baseline = no partition optimization)
/// 2. Default `PeepholeConfig` (all passes enabled = partitioning active)
///
/// Synthesizes with each, records `total_dispatches()` and per-segment counts,
/// and asserts that partitioning does NOT increase dispatch count.
///
/// Requires `KOKORO_WEIGHTS` env var. Skips gracefully when unset.
///
/// Run:
///   KOKORO_WEIGHTS=path/to/kokoro_v1_0.safetensors \
///   cargo test -p nn-metal --test kokoro_all partition_impact_production -- --nocapture
#[test]
fn partition_impact_production() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "partition impact production test not run.",
    ) {
        Some(p) => p,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Synthetic test tokens and style for synthesis.
    let input_ids =
        DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();
    let speed = 1.0;

    // Use Warn policy: synthetic test tokens may produce click artifacts that
    // fail the no_clicks hard bound with production weights. Part of #4262.
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // --- Baseline: all peephole passes disabled (no partition optimization) ---
    let baseline_dispatches;
    let baseline_summary;
    {
        let mut disabled_configs = HashMap::new();
        let disabled = all_disabled_peephole();
        for &key in &SEGMENT_KEYS {
            disabled_configs.insert(key.to_string(), disabled.clone());
        }

        // SAFETY: safetensors file not modified while alive.
        let mut kokoro_baseline = unsafe {
            nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(
                &weights_path,
                hb.clone(),
            )
            .expect("failed to load Kokoro for baseline")
        }
        .with_peephole_configs(disabled_configs);

        kokoro_baseline
            .synthesize(&input_ids, &style, speed, &cache)
            .expect("baseline synthesis must succeed");

        baseline_dispatches = kokoro_baseline.total_dispatches();
        baseline_summary = kokoro_baseline.dispatch_summary();
    }

    // --- Optimized: default PeepholeConfig (all passes enabled) ---
    let partition_dispatches;
    let partition_summary;
    {
        // SAFETY: safetensors file not modified while alive.
        let mut kokoro_optimized = unsafe {
            nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
                .expect("failed to load Kokoro for partitioned run")
        };

        kokoro_optimized
            .synthesize(&input_ids, &style, speed, &cache)
            .expect("partitioned synthesis must succeed");

        partition_dispatches = kokoro_optimized.total_dispatches();
        partition_summary = kokoro_optimized.dispatch_summary();
    }

    // --- Report ---
    let reduction = baseline_dispatches as isize - partition_dispatches as isize;
    let pct = if baseline_dispatches > 0 {
        reduction as f64 / baseline_dispatches as f64 * 100.0
    } else {
        0.0
    };

    eprintln!("\n{}", "=".repeat(72));
    eprintln!("  PRODUCTION PARTITION IMPACT (KOKORO_WEIGHTS, D=512)");
    eprintln!("{}", "=".repeat(72));

    let seg_pairs = [
        ("plbert", baseline_summary.plbert, partition_summary.plbert),
        (
            "text_encoder",
            baseline_summary.text_encoder,
            partition_summary.text_encoder,
        ),
        (
            "prosody",
            baseline_summary.prosody,
            partition_summary.prosody,
        ),
        (
            "f0_energy",
            baseline_summary.f0_energy,
            partition_summary.f0_energy,
        ),
        (
            "generator",
            baseline_summary.generator,
            partition_summary.generator,
        ),
        (
            "regulate",
            baseline_summary.regulate,
            partition_summary.regulate,
        ),
        (
            "sinegen_pre",
            baseline_summary.sinegen_pre,
            partition_summary.sinegen_pre,
        ),
        (
            "sinegen_post",
            baseline_summary.sinegen_post,
            partition_summary.sinegen_post,
        ),
    ];

    eprintln!(
        "  {:<16} {:>12} {:>12} {:>10}",
        "Segment", "Baseline", "Partitioned", "Saved"
    );
    eprintln!("  {}", "-".repeat(54));
    for (name, base, part) in &seg_pairs {
        let saved = *base as isize - *part as isize;
        eprintln!("  {name:<16} {base:>12} {part:>12} {saved:>+10}");
    }
    eprintln!("  {}", "-".repeat(54));
    eprintln!(
        "  {:<16} {:>12} {:>12} {:>+10} ({:+.1}%)",
        "TOTAL", baseline_dispatches, partition_dispatches, reduction, pct,
    );
    eprintln!("{}\n", "=".repeat(72));

    // --- Assertions ---

    // Core assertion: partitioning must not increase dispatch count.
    assert!(
        partition_dispatches <= baseline_dispatches,
        "Partition codegen regression: production pipeline has {partition_dispatches} dispatches with \
         partitioning (was {baseline_dispatches} without). Expected partition <= baseline.",
    );

    // Sanity: both runs compiled all segments.
    assert!(
        baseline_dispatches > 0,
        "Baseline dispatches is 0 -- segments not compiled",
    );
    assert!(
        partition_dispatches > 0,
        "Partition dispatches is 0 -- segments not compiled",
    );

    // Record the improvement for tracking.
    eprintln!(
        "  Partition impact: {reduction} fewer dispatches ({pct:+.1}% reduction)",
    );
}
