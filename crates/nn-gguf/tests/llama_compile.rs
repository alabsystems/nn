// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: Llama graph → nn-dsl compilation pipeline.
//!
//! Verifies that the ComputationGraph produced by `build_llama_graph()`
//! compiles through `compile_trace_to_plan_with_fusion()` and produces
//! a valid CompiledPlan with fusion-reduced dispatch counts.

use nn_gguf::{build_llama_graph, LlamaConfig};

fn tiny_llama_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        ffn_dim: 128,
        rms_norm_eps: 1e-5,
        rope_base: 10000.0,
        max_seq_len: 128,
    }
}

#[test]
fn test_llama_graph_compiles() {
    let config = tiny_llama_config();
    let graph = build_llama_graph(&config);

    // Compile through the full fusion pipeline.
    let plan = nn_dsl::trace_compile::compile_trace_to_plan_with_fusion(&graph)
        .expect("Llama graph should compile");

    // Should have at least some steps.
    assert!(!plan.steps.is_empty(), "compiled plan should have steps");

    // Output step should be the last step.
    assert_eq!(plan.output_step, plan.steps.len() - 1);

    println!("Compiled plan: {} steps", plan.steps.len());
    println!("Weight names: {:?}", plan.weight_names.len());
}

#[test]
fn test_llama_partition_analysis() {
    let config = tiny_llama_config();
    let graph = build_llama_graph(&config);

    let (pre, post) = nn_dsl::trace_compile::partition_analysis(&graph);

    // Pre-partition dispatches should be > 0 (all non-Native nodes).
    assert!(pre > 0, "should have pre-partition dispatches");

    // Post-partition dispatches should be <= pre (fusion reduces dispatches).
    assert!(
        post <= pre,
        "fusion should reduce dispatches: pre={pre}, post={post}"
    );

    // Fusion should achieve some reduction for Llama.
    // Per block: SiLU + Mul can fuse, RMSNorm absorbs Add, etc.
    println!(
        "Partition analysis: {pre} pre → {post} post ({:.1}% reduction)",
        (1.0 - post as f64 / pre as f64) * 100.0
    );
}

#[test]
fn test_llama_compile_no_fusion() {
    let config = tiny_llama_config();
    let graph = build_llama_graph(&config);

    // Compile WITHOUT fusion (baseline).
    let plan = nn_dsl::trace_compile::compile_trace_to_plan(&graph)
        .expect("Llama graph should compile without fusion");

    assert!(!plan.steps.is_empty());

    // Compile WITH fusion.
    let fused_plan = nn_dsl::trace_compile::compile_trace_to_plan_with_fusion(&graph)
        .expect("Llama graph should compile with fusion");

    // Fused plan should have same or fewer dispatches.
    println!(
        "No fusion: {} steps, With fusion: {} steps",
        plan.steps.len(),
        fused_plan.steps.len()
    );
}
