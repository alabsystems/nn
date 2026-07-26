// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Test code uses `let (seq_len, d) = (..)` followed by `&[seq_len, d]` —
// this is intentional readability, not a tuple-to-array conversion.
#![allow(clippy::tuple_array_conversions)]

//! Layerwise attention verification — Phases 12–14.
//!
//! Phase 12: Basic layerwise decomposition verifying each attention sub-layer
//! independently with CROWN. Phase 13: D=64/D=128 multi-head decomposition,
//! empirical PE bounds, perturbation sweeps. Phase 14: D=256/D=512
//! production-scale verification with projected multi-head pipelines.
//!
//! Consolidated from `compose_attention_layerwise.rs` (Phase 12) and
//! `compose_attention_layerwise_phase13_14.rs` (Phases 13–14) per #1982.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phases 12–14.

pub(crate) use super::common;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_monotonicity.rs"]
pub(crate) mod helpers;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_layerwise_builders.rs"]
pub(crate) mod lw_builders;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_layerwise_runners.rs"]
mod runners;

use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};

// ===========================================================================
// Phase 12 tests: basic layerwise decomposition (D=4..32)
// ===========================================================================

/// Layerwise decomposition: verify each attention sub-layer with CROWN.
#[test]
fn test_attention_layerwise_decomposition() {
    let seq_len = 4;
    let d = 4;
    let (output, _, _, _) =
        runners::run_layerwise_pipeline_measured("lw12", seq_len, d, 0.5, 1.0, 0.1);
    common::assert_bounds_valid(&output);
    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[seq_len, d], "output shape");
}

/// D=16: beyond Phase 11 monolithic limit (CROWN fails at D>=12 monolithically).
#[test]
fn test_layerwise_scales_beyond_monolithic_limit() {
    let (seq_len, d) = (4, 16);
    let (output, _, _, _) =
        runners::run_layerwise_pipeline_measured("lw12", seq_len, d, 0.3, 1.0, 0.05);
    common::assert_bounds_valid(&output);
    eprintln!(
        "D=16 layerwise total width: {:.4}",
        lw_builders::measure_total_width(&output)
    );
}

/// D=32: next scaling milestone.
#[test]
fn test_layerwise_d32() {
    let (seq_len, d) = (4, 32);
    let (output, _, _, _) =
        runners::run_layerwise_pipeline_measured("lw12", seq_len, d, 0.2, 1.0, 0.05);
    common::assert_bounds_valid(&output);
    eprintln!(
        "D=32 layerwise total width: {:.4}",
        lw_builders::measure_total_width(&output)
    );
}

/// Diagonal dominance: score lower bounds exceed off-diagonal upper bounds.
#[test]
fn test_layerwise_score_diagonal_dominance() {
    let (seq_len, d) = (4, 8);
    let score_def = lw_builders::build_score_layer("diag_scores", seq_len, d);
    let pe = helpers::build_sinusoidal_pe(seq_len, d);
    let score_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe.clone()),
    ];
    let score_graph = tensor_kernel_to_graph(&score_def, &score_bindings).expect("score graph");
    // Perturbation 0.05: at d=8, PE diagonal margin ~0.15; ±0.05 preserves provability.
    let input = lw_builders::build_pe_centered_bounds(&pe, 0.05);
    let (method, output, _) =
        nn_verify::propagate_with_crown_fallback(&score_graph, &input).expect("score propagation");
    eprintln!("Diagonal dominance test: method={method:?}");
    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[seq_len, seq_len], "score output shape");
    let diag_dominant_rows = lw_builders::count_diagonal_dominant(&output, seq_len);
    eprintln!("Diagonal dominant rows: {diag_dominant_rows}/{seq_len}");
    assert!(
        diag_dominant_rows > 0,
        "expected >= 1 diag-dominant row out of {seq_len}, got {diag_dominant_rows}"
    );
}

/// Higher PE scale → easier diagonal dominance proof.
#[test]
fn test_layerwise_pe_scale_sweep() {
    let (seq_len, d) = (4, 8);
    let base_pe = helpers::build_sinusoidal_pe(seq_len, d);
    let pe_scales = [1.0_f32, 2.0, 4.0, 8.0];
    let mut dominant_counts = Vec::new();
    for &pe_scale in &pe_scales {
        let score_def = lw_builders::build_score_layer(&format!("pe_sweep_{pe_scale}"), seq_len, d);
        let mut pe = base_pe.clone();
        pe.mapv_inplace(|v| v * pe_scale);
        let score_bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(pe.clone()),
        ];
        let score_graph = tensor_kernel_to_graph(&score_def, &score_bindings).expect("score graph");
        let input = lw_builders::build_pe_centered_bounds(&pe, 0.1);
        let (_, output, _) =
            nn_verify::propagate_with_crown_fallback(&score_graph, &input).expect("propagation");
        let dominant = lw_builders::count_diagonal_dominant(&output, seq_len);
        eprintln!("PE scale={pe_scale}: {dominant}/{seq_len} rows diag-dominant");
        dominant_counts.push(dominant);
    }
    let last = *dominant_counts.last().expect("non-empty sweep");
    let first = dominant_counts[0];
    assert!(
        last >= first,
        "highest PE scale should have >= dominant rows: {last} vs {first}"
    );
}

/// Empirical bounds: PE-centered with realistic perturbation radii.
#[test]
fn test_layerwise_empirical_bounds() {
    let (seq_len, d) = (4, 8);
    let pe = helpers::build_sinusoidal_pe(seq_len, d);
    let score_def = lw_builders::build_score_layer("emp_scores", seq_len, d);
    let score_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe.clone()),
    ];
    let score_graph = tensor_kernel_to_graph(&score_def, &score_bindings).expect("graph");
    let input = lw_builders::build_pe_centered_bounds(&pe, 0.05);
    let (method, score_out, _) =
        nn_verify::propagate_with_crown_fallback(&score_graph, &input).expect("propagation");
    eprintln!("Empirical score layer: method={method:?}");
    common::assert_bounds_valid(&score_out);
    let dominant = lw_builders::count_diagonal_dominant(&score_out, seq_len);
    eprintln!("Empirical diagonal dominant rows: {dominant}/{seq_len}");
    assert!(
        dominant > 0,
        "expected >= 1 diag-dominant row, got {dominant}/{seq_len}"
    );
}

// ===========================================================================
// Phase 13 tests: D=64/D=128, multi-head, empirical bounds
// ===========================================================================

#[test]
fn test_layerwise_d64_uniform() {
    let seq_len = 4;
    let d = 64;
    let (output, score_w, sm_w, out_w) =
        runners::run_layerwise_pipeline_measured("lw13", seq_len, d, 0.1, 1.0, 0.02);
    common::assert_bounds_valid(&output);
    let avg = lw_builders::measure_avg_width(&output);
    eprintln!("D=64 layerwise: score_w={score_w:.4}, sm_w={sm_w:.4}, out_w={out_w:.4}");
    eprintln!("D=64 avg element width: {avg:.6}");
    let (lo, _hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[seq_len, d]);
}

#[test]
fn test_layerwise_d128_uniform() {
    let (seq_len, d) = (4, 128);
    let (output, score_w, sm_w, out_w) =
        runners::run_layerwise_pipeline_measured("lw13", seq_len, d, 0.05, 1.0, 0.01);
    common::assert_bounds_valid(&output);
    let avg = lw_builders::measure_avg_width(&output);
    eprintln!("D=128 layerwise: score_w={score_w:.4}, sm_w={sm_w:.4}, out_w={out_w:.4}");
    eprintln!("D=128 avg element width: {avg:.6}");
}

#[test]
fn test_width_scaling_analysis() {
    let seq_len = 4;
    let input_bound = 0.1;
    eprintln!("--- Width scaling analysis (T={seq_len}, ib={input_bound}) ---");
    eprintln!("  D      score_w    sm_w       out_w      avg_w/elem");
    let mut prev_avg = None;
    let mut scaling_ratios = Vec::new();
    for &d in &[8, 16, 32, 64, 128] {
        let v_scale = 0.1 / (d as f32).sqrt();
        let (output, score_w, sm_w, out_w) =
            runners::run_layerwise_pipeline_measured("lw13", seq_len, d, input_bound, 1.0, v_scale);
        let avg = lw_builders::measure_avg_width(&output);
        eprintln!("  {d:<5}  {score_w:>9.4}  {sm_w:>9.4}  {out_w:>9.4}  {avg:>10.6}");
        if let Some(pa) = prev_avg {
            let ratio = avg / pa;
            scaling_ratios.push(ratio);
            eprintln!("         ratio to prev: {ratio:.3}x");
        }
        prev_avg = Some(avg);
        common::assert_bounds_valid(&output);
    }
    for (i, &ratio) in scaling_ratios.iter().enumerate() {
        assert!(
            ratio < 10.0,
            "width scaling ratio {ratio:.2} at step {i} should be < 10x"
        );
    }
}

#[test]
fn test_layerwise_d64_empirical() {
    let (seq_len, d) = (4, 64);
    let (output, dominant) = runners::run_layerwise_empirical(seq_len, d, 0.05, 1.0);
    common::assert_bounds_valid(&output);
    let avg = lw_builders::measure_avg_width(&output);
    eprintln!("D=64 empirical: {dominant}/{seq_len} diag-dominant, avg_width={avg:.6}");
}

#[test]
fn test_layerwise_d128_empirical_pe2() {
    let (seq_len, d) = (4, 128);
    let (output, dominant) = runners::run_layerwise_empirical(seq_len, d, 0.05, 2.0);
    common::assert_bounds_valid(&output);
    let avg = lw_builders::measure_avg_width(&output);
    eprintln!(
        "D=128 empirical (pe_scale=2): {dominant}/{seq_len} diag-dominant, avg_width={avg:.6}"
    );
}

#[test]
fn test_multihead_4heads_d32() {
    let (seq_len, d_model, num_heads) = (4, 32, 4);
    let (dominants, avg_widths) =
        runners::run_multihead_layerwise(seq_len, d_model, num_heads, 0.05, 1.0);
    eprintln!(
        "--- Multi-head: H={num_heads}, D={d_model}, d_k={} ---",
        d_model / num_heads
    );
    for (h, (dom, width)) in dominants.iter().zip(avg_widths.iter()).enumerate() {
        eprintln!("  Head {h}: {dom}/{seq_len} diag-dominant, avg_width={width:.6}");
    }
    for &w in &avg_widths {
        assert!(w.is_finite(), "per-head width finite");
        // w >= 0.0 is structurally guaranteed by NY (lo <= hi),
        // so asserting non-negativity after is_finite() is tautological.
        // Instead, assert a meaningful upper bound for small-weight tests.
        assert!(w < 100.0, "per-head width should be bounded, got {w}");
    }
}

#[test]
fn test_multihead_8heads_d64() {
    let (seq_len, d_model, num_heads) = (4, 64, 8);
    let (dominants, avg_widths) =
        runners::run_multihead_layerwise(seq_len, d_model, num_heads, 0.05, 1.0);
    eprintln!(
        "--- Multi-head: H={num_heads}, D={d_model}, d_k={} ---",
        d_model / num_heads
    );
    let total_dominant: usize = dominants.iter().sum();
    eprintln!(
        "  Total diagonal dominant: {total_dominant}/{}",
        num_heads * seq_len
    );
    for &w in &avg_widths {
        assert!(
            w.is_finite() && w < 100.0,
            "per-head width should be finite and bounded, got {w}"
        );
    }
}

#[test]
fn test_multihead_8heads_d128() {
    let (seq_len, d_model, num_heads) = (4, 128, 8);
    let (dominants, avg_widths) =
        runners::run_multihead_layerwise(seq_len, d_model, num_heads, 0.05, 2.0);
    eprintln!(
        "--- Multi-head: H={num_heads}, D={d_model}, d_k={} ---",
        d_model / num_heads
    );
    let total_dominant: usize = dominants.iter().sum();
    eprintln!(
        "  Total diagonal dominant: {total_dominant}/{}",
        num_heads * seq_len
    );
    for &w in &avg_widths {
        assert!(
            w.is_finite() && w < 100.0,
            "per-head width should be finite and bounded, got {w}"
        );
    }
}

#[test]
fn test_perturbation_sweep_d64() {
    let (seq_len, d) = (4, 64);
    eprintln!("--- Perturbation sweep (D={d}, T={seq_len}) ---");
    eprintln!("  ε         dominant  avg_width");
    let mut dominant_at_eps = Vec::new();
    for &eps in &[0.2, 0.1, 0.05, 0.02, 0.01] {
        let (output, dominant) = runners::run_layerwise_empirical(seq_len, d, eps, 1.0);
        let avg = lw_builders::measure_avg_width(&output);
        dominant_at_eps.push((eps, dominant));
        eprintln!("  {eps:.3}      {dominant}/{seq_len}       {avg:.6}");
    }
    let dominants: Vec<usize> = dominant_at_eps.iter().map(|(_, d)| *d).collect();
    assert!(
        dominants.last().expect("non-empty") >= dominants.first().expect("non-empty"),
        "tighter perturbation should give >= diagonal dominance: {dominant_at_eps:?}"
    );
}

// ===========================================================================
// Phase 14 tests: D=256/D=512, projected pipelines
// ===========================================================================

#[test]
fn test_layerwise_d256_uniform() {
    let seq_len = 4;
    let d = 256;
    let (output, score_w, sm_w, out_w) =
        runners::run_layerwise_pipeline_measured("lw14", seq_len, d, 0.05, 1.0, 0.005);
    common::assert_bounds_valid(&output);
    let avg = lw_builders::measure_avg_width(&output);
    eprintln!(
        "D=256 layerwise: score_w={score_w:.4}, sm_w={sm_w:.4}, out_w={out_w:.4}, avg={avg:.6}"
    );
    let (lo, _hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[seq_len, d]);
}

#[test]
fn test_layerwise_d512_uniform() {
    let seq_len = 4;
    let d = 512;
    let (output, score_w, sm_w, out_w) =
        runners::run_layerwise_pipeline_measured("lw14", seq_len, d, 0.02, 1.0, 0.002);
    common::assert_bounds_valid(&output);
    let avg = lw_builders::measure_avg_width(&output);
    eprintln!(
        "D=512 layerwise: score_w={score_w:.4}, sm_w={sm_w:.4}, out_w={out_w:.4}, avg={avg:.6}"
    );
    let (lo, _hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[seq_len, d]);
}

#[test]
fn test_width_scaling_d128_to_d512() {
    let seq_len = 4;
    eprintln!("--- Width scaling D=128→512 (T={seq_len}) ---");
    eprintln!("  D      score_w     sm_w       out_w      avg_w/elem   ratio");
    let mut prev_avg = None;
    for &d in &[128, 256, 512] {
        let input_bound = 0.1 / (d as f32 / 128.0).sqrt();
        let v_scale = 0.1 / (d as f32).sqrt();
        let (output, score_w, sm_w, out_w) =
            runners::run_layerwise_pipeline_measured("lw14", seq_len, d, input_bound, 1.0, v_scale);
        let avg = lw_builders::measure_avg_width(&output);
        let ratio_str = match prev_avg {
            Some(pa) => format!("{:.3}x", avg / pa),
            None => "—".to_string(),
        };
        eprintln!("  {d:<5}  {score_w:>9.4}  {sm_w:>9.4}  {out_w:>9.4}  {avg:>10.6}   {ratio_str}");
        common::assert_bounds_valid(&output);
        prev_avg = Some(avg);
    }
}

#[test]
fn test_projected_d64_dk16() {
    let (seq_len, d_model, d_k) = (4, 64, 16);
    let (output, proj_w, score_w, sm_w, out_w) =
        runners::run_projected_pipeline(seq_len, d_model, d_k, 0.1, 1.0, 0.001, 1.0, 0.05);
    common::assert_bounds_valid(&output);
    let avg = lw_builders::measure_avg_width(&output);
    eprintln!(
        "Projected D={d_model}→d_k={d_k}: proj_w={proj_w:.4}, score_w={score_w:.4}, \
         sm_w={sm_w:.4}, out_w={out_w:.4}, avg={avg:.6}"
    );
}

#[test]
fn test_projected_d512_dk64() {
    let (seq_len, d_model, d_k) = (4, 512, 64);
    let (output, proj_w, score_w, sm_w, out_w) =
        runners::run_projected_pipeline(seq_len, d_model, d_k, 0.02, 1.0, 0.0001, 1.0, 0.01);
    common::assert_bounds_valid(&output);
    let avg = lw_builders::measure_avg_width(&output);
    eprintln!(
        "Projected D={d_model}→d_k={d_k}: proj_w={proj_w:.4}, score_w={score_w:.4}, \
         sm_w={sm_w:.4}, out_w={out_w:.4}, avg={avg:.6}"
    );
}

#[test]
fn test_projected_multihead_4h_d128() {
    let (seq_len, d_model, num_heads) = (4, 128, 4);
    let d_k = d_model / num_heads;
    let (avg_widths, dominants) =
        runners::run_projected_multihead(seq_len, d_model, num_heads, 0.05, 1.0, 0.001, 2.0, 0.05);
    eprintln!("--- Projected multi-head: H={num_heads}, D={d_model}, d_k={d_k} ---");
    let total_dom: usize = dominants.iter().sum();
    eprintln!(
        "  Total diagonal dominant: {total_dom}/{}",
        num_heads * seq_len
    );
    for &w in &avg_widths {
        assert!(
            w.is_finite() && w < 100.0,
            "per-head width should be finite and bounded, got {w}"
        );
    }
}

#[test]
fn test_projected_multihead_8h_d512() {
    let (seq_len, d_model, num_heads) = (4, 512, 8);
    let d_k = d_model / num_heads;
    let (avg_widths, dominants) =
        runners::run_projected_multihead(seq_len, d_model, num_heads, 0.02, 1.0, 0.0001, 2.0, 0.02);
    eprintln!("--- PRODUCTION: Projected multi-head H={num_heads}, D={d_model}, d_k={d_k} ---");
    let total_dom: usize = dominants.iter().sum();
    eprintln!(
        "  Total diagonal dominant: {total_dom}/{}",
        num_heads * seq_len
    );
    for &w in &avg_widths {
        assert!(
            w.is_finite() && w < 100.0,
            "per-head width should be finite and bounded, got {w}"
        );
    }
}

#[test]
fn test_projection_impact_d128() {
    let (seq_len, d) = (4, 128);
    let input_bound = 0.05;
    let k_scale = 1.0;
    let v_scale = 0.01;

    let (out_3layer, _, _, _) =
        runners::run_layerwise_pipeline_measured("lw14", seq_len, d, input_bound, k_scale, v_scale);
    let avg_3 = lw_builders::measure_avg_width(&out_3layer);

    let (out_4layer, _, _, _, _) =
        runners::run_projected_pipeline(seq_len, d, d, input_bound, 1.0, 0.0001, k_scale, v_scale);
    let avg_4 = lw_builders::measure_avg_width(&out_4layer);

    let overhead = avg_4 / avg_3;
    eprintln!(
        "Projection impact at D={d}: 3-layer avg={avg_3:.6}, 4-layer avg={avg_4:.6}, \
         overhead={overhead:.3}x"
    );
    assert!(
        overhead < 10.0,
        "projection overhead {overhead:.2}x should be < 10x"
    );
}
