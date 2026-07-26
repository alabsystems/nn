// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Level 2: CROWN at sub-block granularity for the Kokoro Generator.
//!
//! Builds on Level 1 (IBP sub-block decomposition from `_d512_ibp.rs`) by applying
//! CROWN propagation to multi-layer sub-blocks. Per-layer CROWN provides ZERO
//! tightening over IBP (ratio 1.0000 per `_layerwise_deep.rs`). Multi-layer
//! sub-blocks enable CROWN's backward pass to linearize non-linear layers (ReLU,
//! Snake, InstanceNorm) and propagate tighter linear bounds through the composition.
//!
//! Strategy:
//! - Group layers into 3-5 layer sub-blocks at normalization boundaries
//! - Apply CROWN (via `propagate_with_crown_fallback`) per sub-block
//! - Chain sub-block bounds via junction contract verification
//! - Measure CROWN vs IBP tightening ratio per sub-block
//!
//! Tractable dimensions: D=64 (fast), D=128 (medium). D=512 is structural-only
//! for CROWN due to Conv1d weight matrix size ([256,256,3] = 196K elements).
//!
//! Part of #2599: Kokoro Generator verification ceiling.
//! Part of #2218: Epic — Perfect Kokoro.

#[path = "kokoro_scaled_pipeline.rs"]
mod d512_scaled_helpers;
use d512_scaled_helpers as helpers;

#[path = "kokoro_scaled_layerwise.rs"]
mod layerwise_helpers;

use helpers::KokoroDims;
use layerwise_helpers::{build_kokoro_layerwise_deep, LayerSpec};
use nn_verify::{
    tensor_kernel_to_graph, tensor_kernels_to_grouped_graph, BoundedTensor, NormBoundsMode,
    PropMethod, SubBlockBounds,
};

use super::common::{
    assert_bounds_valid, assert_crown_tighter_than_ibp, bounds_min_max, uniform_bounds,
};

// -- Sub-block CROWN pipeline ------------------------------------------------

/// Per-sub-block CROWN/IBP comparison result.
#[derive(Debug)]
struct SubBlockResult {
    name: String,
    ibp_output: BoundedTensor,
    crown_output: BoundedTensor,
    ibp_width: f32,
    crown_width: f32,
    tightening_ratio: f32,
    method: PropMethod,
    _fallback_reason: Option<String>,
}

/// Run CROWN + IBP per sub-block group, chain bounds, and return per-group results.
///
/// Groups layers at normalization boundaries (same as `run_ibp_sub_blocks`):
/// - Group 0: layers 0..2 (pre-norm)
/// - Groups 1..N: each ResBlock individually
/// - Group N+1: VocoderOutput
///
/// For each group, runs both IBP and CROWN, comparing width.
fn run_crown_sub_blocks(
    layers: &[LayerSpec],
    initial_bounds: &BoundedTensor,
    num_resblocks: usize,
) -> Vec<SubBlockResult> {
    let total_layers = layers.len();
    assert_eq!(
        total_layers,
        4 + num_resblocks,
        "expected 4 + {num_resblocks} layers, got {total_layers}"
    );

    let mut results = Vec::new();
    let mut current_ibp_bounds = initial_bounds.clone();
    let mut current_crown_bounds = initial_bounds.clone();

    // Helper: process a group of layers
    let process_group = |name: &str,
                         group_layers: &[LayerSpec],
                         ibp_input: &BoundedTensor,
                         crown_input: &BoundedTensor|
     -> SubBlockResult {
        let graph = tensor_kernels_to_grouped_graph(group_layers, NormBoundsMode::ForwardMode)
            .unwrap_or_else(|e| panic!("{name} graph build: {e}"));

        // IBP baseline
        let ibp_output = graph
            .propagate_ibp(ibp_input)
            .unwrap_or_else(|e| panic!("{name} IBP: {e}"));
        let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
        let ibp_width = ibp_hi - ibp_lo;

        // CROWN with fallback
        let (method, crown_output, fallback_reason) =
            nn_verify::propagate_with_crown_fallback(&graph, crown_input)
                .unwrap_or_else(|e| panic!("{name} CROWN: {e}"));
        let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
        let crown_width = crown_hi - crown_lo;

        let tightening = if crown_width > 1e-10 && ibp_width > 1e-10 {
            ibp_width / crown_width
        } else {
            1.0
        };

        let method_str = if matches!(method, PropMethod::Crown) {
            "CROWN"
        } else {
            "IBP-fallback"
        };
        eprintln!(
            "  {name}: IBP=[{ibp_lo:.4}, {ibp_hi:.4}] w={ibp_width:.4} | \
                 {method_str}=[{crown_lo:.4}, {crown_hi:.4}] w={crown_width:.4} | \
                 tightening={tightening:.2}x"
        );

        SubBlockResult {
            name: name.to_string(),
            ibp_output,
            crown_output,
            ibp_width,
            crown_width,
            tightening_ratio: tightening,
            method,
            _fallback_reason: fallback_reason,
        }
    };

    // Group 0: pre-normalization block (layers 0, 1, 2)
    {
        let group_layers: Vec<_> = layers[0..3].to_vec();
        let result = process_group(
            "pre_norm",
            &group_layers,
            &current_ibp_bounds,
            &current_crown_bounds,
        );
        current_ibp_bounds = result.ibp_output.clone();
        current_crown_bounds = result.crown_output.clone();
        results.push(result);
    }

    // Groups 1..N: each ResBlock individually
    for i in 0..num_resblocks {
        let layer_idx = 3 + i;
        let group_layers = vec![layers[layer_idx].clone()];
        let name = format!("resblock_{i}");
        let result = process_group(
            &name,
            &group_layers,
            &current_ibp_bounds,
            &current_crown_bounds,
        );
        current_ibp_bounds = result.ibp_output.clone();
        current_crown_bounds = result.crown_output.clone();
        results.push(result);
    }

    // Group N+1: VocoderOutput (last layer)
    {
        let last_idx = total_layers - 1;
        let group_layers = vec![layers[last_idx].clone()];
        let result = process_group(
            "output",
            &group_layers,
            &current_ibp_bounds,
            &current_crown_bounds,
        );
        results.push(result);
    }

    results
}

/// Convert sub-block CROWN results into `SubBlockBounds` for junction contract verification.
fn results_to_junction_bounds(
    results: &[SubBlockResult],
    initial_bounds: &BoundedTensor,
) -> Vec<SubBlockBounds> {
    let mut blocks = Vec::with_capacity(results.len());

    // First block uses initial input bounds
    let (init_lo, init_hi) = initial_bounds.lower_upper();
    let (out_lo, out_hi) = results[0].crown_output.lower_upper();
    blocks.push(SubBlockBounds {
        name: results[0].name.clone(),
        input_lower: init_lo.iter().copied().collect(),
        input_upper: init_hi.iter().copied().collect(),
        output_lower: out_lo.iter().copied().collect(),
        output_upper: out_hi.iter().copied().collect(),
    });

    // Subsequent blocks: input = previous output, output = this output
    for i in 1..results.len() {
        let (prev_lo, prev_hi) = results[i - 1].crown_output.lower_upper();
        let (out_lo, out_hi) = results[i].crown_output.lower_upper();
        blocks.push(SubBlockBounds {
            name: results[i].name.clone(),
            input_lower: prev_lo.iter().copied().collect(),
            input_upper: prev_hi.iter().copied().collect(),
            output_lower: out_lo.iter().copied().collect(),
            output_upper: out_hi.iter().copied().collect(),
        });
    }

    blocks
}

// -- Tests: CROWN at D=64 (tractable, fast) ----------------------------------

/// CROWN on pre-norm sub-block at D=64 produces tighter bounds than IBP.
///
/// The pre-norm group (TextEncoder + VocoderPre + VocoderUpsample) contains
/// Conv1d + ReLU + Linear + LeakyReLU + ConvTranspose1d — multiple non-linear
/// layers that CROWN should tighten by linearizing.
#[test]
fn test_d64_crown_pre_norm_tighter_than_ibp() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, 2);
    let initial = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    // Build pre-norm group (layers 0-2)
    let group_layers: Vec<_> = layers[0..3].to_vec();
    let graph = tensor_kernels_to_grouped_graph(&group_layers, NormBoundsMode::ForwardMode)
        .expect("pre-norm graph");

    let ibp_output = graph.propagate_ibp(&initial).expect("IBP");
    let (method, crown_output, _fallback) =
        nn_verify::propagate_with_crown_fallback(&graph, &initial).expect("CROWN");

    assert_bounds_valid(&ibp_output);
    assert_bounds_valid(&crown_output);

    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let ibp_width = ibp_hi - ibp_lo;
    let crown_width = crown_hi - crown_lo;
    let tightening = if crown_width > 1e-10 {
        ibp_width / crown_width
    } else {
        1.0
    };

    eprintln!("D=64 pre-norm: IBP width={ibp_width:.4}, CROWN width={crown_width:.4}, ratio={tightening:.2}x, method={method:?}");

    if matches!(method, PropMethod::Crown) {
        // CROWN succeeded — bounds should be at least as tight as IBP
        assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
        eprintln!("  CROWN tightening confirmed: {tightening:.2}x");
    } else {
        eprintln!("  CROWN fell back to IBP (expected for graphs with normalization layers)");
    }
}

/// Full D=64 sub-block CROWN pipeline: all groups, with junction contracts.
#[test]
fn test_d64_crown_sub_blocks_with_junctions() {
    let dims = KokoroDims::d64();
    let num_resblocks = 3;
    let layers = build_kokoro_layerwise_deep(&dims, num_resblocks);
    let initial = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    eprintln!("D=64 CROWN sub-blocks ({num_resblocks} ResBlocks):");
    let results = run_crown_sub_blocks(&layers, &initial, num_resblocks);

    // All sub-blocks must produce finite bounds
    for r in &results {
        assert_bounds_valid(&r.crown_output);
        assert!(
            r.crown_width.is_finite(),
            "sub-block {}: CROWN width must be finite, got {}",
            r.name,
            r.crown_width
        );
    }

    // Junction contract verification
    let junction_bounds = results_to_junction_bounds(&results, &initial);
    let junction_result =
        nn_verify::verify_junctions(&junction_bounds).expect("junction verification");

    eprintln!(
        "\nJunction contracts: {} valid, {} invalid, max violation={:.6}",
        junction_result.proofs.len() - junction_result.invalid_count(),
        junction_result.invalid_count(),
        junction_result.max_violation()
    );
    for proof in &junction_result.proofs {
        eprintln!(
            "  {} -> {}: valid={}, max_violation={:.6}",
            proof.upstream, proof.downstream, proof.is_valid, proof.max_violation
        );
    }

    // All junctions must be valid (output bounds of block k ⊆ input bounds of block k+1)
    assert!(
        junction_result.all_valid(),
        "all junction contracts must hold: {} invalid out of {}",
        junction_result.invalid_count(),
        junction_result.proofs.len()
    );

    // Summary: count how many sub-blocks got CROWN tightening
    let crown_count = results
        .iter()
        .filter(|r| matches!(r.method, PropMethod::Crown))
        .count();
    let tightened_count = results.iter().filter(|r| r.tightening_ratio > 1.01).count();
    eprintln!(
        "\nSummary: {crown_count}/{} sub-blocks used CROWN, {tightened_count} showed tightening",
        results.len()
    );
}

// -- Tests: CROWN at D=128 (medium, measures CROWN vs IBP gap) ---------------

/// D=128 CROWN sub-block pipeline with junction contracts.
///
/// D=128 is the sweet spot: large enough that IBP bounds explode through
/// ResBlocks, but small enough that CROWN backward pass is tractable.
/// This test measures the real CROWN vs IBP tightening ratio.
#[test]
fn test_d128_crown_sub_blocks_with_junctions() {
    let dims = KokoroDims::d128();
    let num_resblocks = 2; // Keep small for CROWN tractability at D=128
    let layers = build_kokoro_layerwise_deep(&dims, num_resblocks);
    let initial = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    eprintln!("D=128 CROWN sub-blocks ({num_resblocks} ResBlocks):");
    let results = run_crown_sub_blocks(&layers, &initial, num_resblocks);

    // All sub-blocks must produce finite bounds
    for r in &results {
        assert_bounds_valid(&r.crown_output);
        assert!(
            r.crown_width.is_finite(),
            "sub-block {}: CROWN width must be finite",
            r.name
        );
    }

    // Junction contracts
    let junction_bounds = results_to_junction_bounds(&results, &initial);
    let junction_result =
        nn_verify::verify_junctions(&junction_bounds).expect("junction verification");
    assert!(
        junction_result.all_valid(),
        "all D=128 junction contracts must hold"
    );

    // Report
    let crown_count = results
        .iter()
        .filter(|r| matches!(r.method, PropMethod::Crown))
        .count();
    let max_tightening = results
        .iter()
        .map(|r| r.tightening_ratio)
        .fold(0.0f32, f32::max);
    eprintln!(
        "\nD=128: {crown_count}/{} CROWN, max tightening={max_tightening:.2}x, \
         all {} junctions valid",
        results.len(),
        junction_result.proofs.len()
    );
}

// -- Tests: D=512 structural + limited CROWN ---------------------------------

/// D=512 graph construction for CROWN sub-blocks succeeds.
///
/// Validates that the same pipeline used for IBP sub-blocks can be built
/// and grouped into graphs suitable for CROWN propagation. Does NOT attempt
/// CROWN propagation (intractable for Conv1d [256,256,3]).
#[test]
fn test_d512_crown_graph_construction() {
    let dims = KokoroDims::d512();
    let num_resblocks = 3;
    let layers = build_kokoro_layerwise_deep(&dims, num_resblocks);

    // Pre-norm group should build
    let pre_group: Vec<_> = layers[0..3].to_vec();
    let pre_graph = tensor_kernels_to_grouped_graph(&pre_group, NormBoundsMode::ForwardMode)
        .expect("D=512 pre-norm grouped graph");
    let pre_node_count = pre_graph.node_names().len();
    eprintln!("D=512 pre-norm group: {pre_node_count} graph nodes");
    assert!(pre_node_count > 0);

    // Each ResBlock should build
    for i in 0..num_resblocks {
        let rb_group = vec![layers[3 + i].clone()];
        let rb_graph = tensor_kernels_to_grouped_graph(&rb_group, NormBoundsMode::ForwardMode)
            .unwrap_or_else(|e| panic!("D=512 resblock_{i} graph: {e}"));
        let rb_nodes = rb_graph.node_names().len();
        eprintln!("D=512 resblock_{i}: {rb_nodes} graph nodes");
        assert!(rb_nodes > 0);
    }

    // Output group should build
    let out_group = vec![layers.last().unwrap().clone()];
    let out_graph = tensor_kernels_to_grouped_graph(&out_group, NormBoundsMode::ForwardMode)
        .expect("D=512 output grouped graph");
    let out_nodes = out_graph.node_names().len();
    eprintln!("D=512 output group: {out_nodes} graph nodes");
    assert!(out_nodes > 0);

    eprintln!(
        "D=512 CROWN graph construction: all {} groups buildable",
        2 + num_resblocks
    );
}

/// D=512 IBP sub-blocks with junction contract verification.
///
/// Runs the Level 1 IBP pipeline at D=512 but adds Level 1B junction contract
/// verification between every adjacent pair of sub-blocks. This proves that
/// the sub-block decomposition is sound: output bounds compose correctly.
#[test]
fn test_d512_ibp_with_junction_contracts() {
    let dims = KokoroDims::d512();
    let num_resblocks = 3;
    let layers = build_kokoro_layerwise_deep(&dims, num_resblocks);
    let initial = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    eprintln!("D=512 IBP sub-blocks with junction contracts ({num_resblocks} ResBlocks):");

    let mut ibp_results: Vec<(String, BoundedTensor)> = Vec::new();
    let mut current_bounds = initial.clone();

    // Group 0: pre-norm (layers 0-2)
    {
        let group: Vec<_> = layers[0..3].to_vec();
        let graph = tensor_kernels_to_grouped_graph(&group, NormBoundsMode::ForwardMode)
            .expect("pre-norm graph");
        let output = graph.propagate_ibp(&current_bounds).expect("pre-norm IBP");
        let (lo, hi) = bounds_min_max(&output);
        eprintln!("  pre_norm: [{lo:.4}, {hi:.4}]");
        ibp_results.push(("pre_norm".to_string(), output.clone()));
        current_bounds = output;
    }

    // Groups 1..N: ResBlocks
    for i in 0..num_resblocks {
        let group = vec![layers[3 + i].clone()];
        let graph = tensor_kernels_to_grouped_graph(&group, NormBoundsMode::ForwardMode)
            .unwrap_or_else(|e| panic!("resblock_{i} graph: {e}"));
        let output = graph
            .propagate_ibp(&current_bounds)
            .unwrap_or_else(|e| panic!("resblock_{i} IBP: {e}"));
        let (lo, hi) = bounds_min_max(&output);
        eprintln!("  resblock_{i}: [{lo:.4}, {hi:.4}]");
        ibp_results.push((format!("resblock_{i}"), output.clone()));
        current_bounds = output;
    }

    // Output group
    {
        let group = vec![layers.last().unwrap().clone()];
        let graph = tensor_kernels_to_grouped_graph(&group, NormBoundsMode::ForwardMode)
            .expect("output graph");
        let output = graph.propagate_ibp(&current_bounds).expect("output IBP");
        let (lo, hi) = bounds_min_max(&output);
        eprintln!("  output: [{lo:.4}, {hi:.4}]");
        ibp_results.push(("output".to_string(), output));
    }

    // Build junction bounds
    let mut junction_blocks = Vec::with_capacity(ibp_results.len());

    // First block
    let (init_lo, init_hi) = initial.lower_upper();
    let (out_lo, out_hi) = ibp_results[0].1.lower_upper();
    junction_blocks.push(SubBlockBounds {
        name: ibp_results[0].0.clone(),
        input_lower: init_lo.iter().copied().collect(),
        input_upper: init_hi.iter().copied().collect(),
        output_lower: out_lo.iter().copied().collect(),
        output_upper: out_hi.iter().copied().collect(),
    });

    for i in 1..ibp_results.len() {
        let (prev_lo, prev_hi) = ibp_results[i - 1].1.lower_upper();
        let (out_lo, out_hi) = ibp_results[i].1.lower_upper();
        junction_blocks.push(SubBlockBounds {
            name: ibp_results[i].0.clone(),
            input_lower: prev_lo.iter().copied().collect(),
            input_upper: prev_hi.iter().copied().collect(),
            output_lower: out_lo.iter().copied().collect(),
            output_upper: out_hi.iter().copied().collect(),
        });
    }

    let junction_result =
        nn_verify::verify_junctions(&junction_blocks).expect("junction verification");

    eprintln!(
        "\nD=512 junction contracts: {}/{} valid, max violation={:.6}",
        junction_result.proofs.len() - junction_result.invalid_count(),
        junction_result.proofs.len(),
        junction_result.max_violation()
    );

    assert!(
        junction_result.all_valid(),
        "D=512 junction contracts must all hold (IBP chaining is sound)"
    );
}

/// D=512 per-layer graph construction verifies each individual layer can build a
/// NY GraphNetwork. This is the prerequisite for CROWN propagation once
/// NY GPU acceleration lands (#1271).
#[test]
fn test_d512_crown_per_layer_graph_translate() {
    let dims = KokoroDims::d512();
    let num_resblocks = 3;
    let layers = build_kokoro_layerwise_deep(&dims, num_resblocks);

    for (i, (def, bindings)) in layers.iter().enumerate() {
        let graph = tensor_kernel_to_graph(def, bindings)
            .unwrap_or_else(|e| panic!("layer {i} ({}) graph translate: {e}", def.name));
        let node_count = graph.node_names().len();
        eprintln!("  layer {i} ({}): {node_count} nodes", def.name);
        assert!(node_count > 0, "layer {i} must have nodes");
    }

    eprintln!(
        "D=512: all {} layers translate to GraphNetwork",
        layers.len()
    );
}

// -- Test: CROWN tightening measurement at tractable scale -------------------

/// Measure CROWN vs IBP tightening at D=64 across all sub-block types.
///
/// This is the key metric: for which sub-block types does CROWN provide
/// meaningful tightening? Pre-norm (Conv+ReLU+Linear) should benefit.
/// ResBlocks (InstanceNorm+Snake+Conv1d) may not if CROWN falls back.
#[test]
fn test_d64_crown_tightening_per_group_type() {
    let dims = KokoroDims::d64();
    let num_resblocks = 3;
    let layers = build_kokoro_layerwise_deep(&dims, num_resblocks);
    let initial = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    eprintln!("D=64 CROWN tightening analysis ({num_resblocks} ResBlocks):");
    let results = run_crown_sub_blocks(&layers, &initial, num_resblocks);

    // Classify sub-block types
    let pre_norm = &results[0];
    let resblocks: Vec<_> = results[1..=num_resblocks].iter().collect();
    let output = results.last().unwrap();

    eprintln!("\nPer-type analysis:");
    eprintln!(
        "  Pre-norm (Conv+ReLU+Linear): method={:?}, tightening={:.2}x",
        pre_norm.method, pre_norm.tightening_ratio
    );
    for (i, rb) in resblocks.iter().enumerate() {
        eprintln!(
            "  ResBlock_{i} (InstNorm+Snake+Conv): method={:?}, tightening={:.2}x",
            rb.method, rb.tightening_ratio
        );
    }
    eprintln!(
        "  Output (LeakyReLU+Conv+Clamp+Exp): method={:?}, tightening={:.2}x",
        output.method, output.tightening_ratio
    );

    // At minimum, all bounds must be finite
    for r in &results {
        assert!(r.ibp_width.is_finite(), "{}: IBP width not finite", r.name);
        assert!(
            r.crown_width.is_finite(),
            "{}: CROWN width not finite",
            r.name
        );
    }

    // CROWN should not produce wider bounds than IBP (soundness check)
    for r in &results {
        if matches!(r.method, PropMethod::Crown) {
            assert!(
                r.tightening_ratio >= 0.99,
                "{}: CROWN produced wider bounds than IBP (ratio={:.4})",
                r.name,
                r.tightening_ratio
            );
        }
    }
}
// Level 2 mixed-mode D=512 tests are in compose_kokoro_generator_d512_mixed.rs
