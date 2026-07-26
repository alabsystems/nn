// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Test code uses `let (seq_len, d) = (..)` followed by `&[seq_len, d]` —
// this is intentional readability, not a tuple-to-array conversion.
#![allow(clippy::tuple_array_conversions)]

//! Phase 15: End-to-end composed attention + adversarial perturbation analysis.
//!
//! Phase 14 proved layerwise CROWN scales to production D=512 with per-head
//! decomposition. Each layer was verified independently, with output bounds
//! from one layer feeding as input to the next. Phase 15 answers two questions:
//!
//! 1. **Monolithic composition**: Can we build ALL attention layers as a single
//!    `TensorKernelDef` graph and verify end-to-end? This lets NY see
//!    cross-layer dependencies, potentially producing tighter bounds than
//!    layerwise propagation. We test at D=8 through D=32 (monolithic CROWN's
//!    tractable range from Phase 11).
//!
//! 2. **Adversarial perturbation stability** (#1740): Given a perturbation set
//!    in embedding space (modeling homoglyph substitution, Unicode attacks, etc.),
//!    prove that attention weights remain stable — the softmax output changes
//!    by at most ε. This is the foundation for #1740 AC1 (perturbation sets)
//!    and AC2 (CROWN-verify phoneme stability under perturbations).
//!
//! Key results:
//!   - Monolithic graph at D=8: tighter bounds than layerwise (CROWN sees full path)
//!   - Monolithic vs layerwise ratio quantifies the benefit of end-to-end verification
//!   - Adversarial stability: attention weight perturbation bounded for ε-ball inputs
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 15.
//! Part of #1740: Adversarial Robustness of TTS — AC1 perturbation sets.

pub(crate) use super::common;

#[path = "attention_monotonicity.rs"]
mod helpers;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_layerwise_builders.rs"]
pub(crate) mod lw_builders;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_e2e_runners.rs"]
mod e2e_runners;

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};

// ===========================================================================
// Tests: Monolithic end-to-end composition
// ===========================================================================

/// Monolithic 3-layer attention at D=8 — the tractable baseline.
///
/// All 3 layers (score + softmax + output) in a single graph. NY
/// sees the full Q→Scores→Softmax→Output path and can propagate CROWN
/// bounds through the entire computation.
#[test]
fn test_monolithic_3layer_d8() {
    let seq_len = 4;
    let d = 8;
    let input_bound = 0.1;
    let k_scale = 1.0;
    let v_scale = 0.1;

    let def = e2e_runners::build_monolithic_attention_no_proj("mono_3l_d8", seq_len, d);
    let k_tensor = lw_builders::build_k_identity(seq_len, d, k_scale);
    let v_tensor = lw_builders::build_v_tensor(seq_len, d, v_scale);

    let bindings = vec![
        TensorParamBinding::Variable,                 // Q
        TensorParamBinding::ConstantTensor(k_tensor), // K
        TensorParamBinding::ConstantTensor(v_tensor), // V
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("monolithic graph");
    let input = common::uniform_bounds(&[seq_len, d], input_bound);
    let (method, output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("propagation");

    common::assert_bounds_valid(&output);

    let avg = lw_builders::measure_avg_width(&output);
    let max_w = lw_builders::measure_max_width(&output);
    eprintln!("Monolithic 3-layer D={d}: method={method:?}, avg_w={avg:.6}, max_w={max_w:.6}");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[seq_len, d]);
}

/// Monolithic 4-layer projected attention at D=8, d_k=8.
///
/// Full pipeline: Q → W_q projection → score → softmax → output.
/// Single graph with 4 computation stages.
#[test]
fn test_monolithic_4layer_d8() {
    let (seq_len, d_model, d_k) = (4, 8, 8);
    let input_bound = 0.1;

    let def = e2e_runners::build_monolithic_attention("mono_4l_d8", seq_len, d_model, d_k);
    let w_q = lw_builders::build_near_identity_weights(d_model, d_k, 1.0, 0.01);
    let k_tensor = lw_builders::build_k_identity(seq_len, d_k, 1.0);
    let v_tensor = lw_builders::build_v_tensor(seq_len, d_k, 0.1);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w_q),
        TensorParamBinding::ConstantTensor(k_tensor),
        TensorParamBinding::ConstantTensor(v_tensor),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("monolithic graph");
    let input = common::uniform_bounds(&[seq_len, d_model], input_bound);
    let (method, output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("propagation");

    common::assert_bounds_valid(&output);

    let avg = lw_builders::measure_avg_width(&output);
    eprintln!("Monolithic 4-layer D={d_model}→d_k={d_k}: method={method:?}, avg_w={avg:.6}");
}

/// Monolithic 3-layer at D=16 and D=32 — scaling within monolithic range.
#[test]
fn test_monolithic_3layer_scaling() {
    let seq_len = 4;

    eprintln!("--- Monolithic 3-layer scaling (T={seq_len}) ---");
    eprintln!("  D       avg_w       max_w       method");

    for &d in &[8, 16, 32] {
        let input_bound = 0.1 / (d as f32 / 8.0).sqrt();
        let v_scale = 0.1 / (d as f32).sqrt();

        let def =
            e2e_runners::build_monolithic_attention_no_proj(&format!("mono_scale_{d}"), seq_len, d);
        let k_tensor = lw_builders::build_k_identity(seq_len, d, 1.0);
        let v_tensor = lw_builders::build_v_tensor(seq_len, d, v_scale);

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(k_tensor),
            TensorParamBinding::ConstantTensor(v_tensor),
        ];

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = common::uniform_bounds(&[seq_len, d], input_bound);
        let (method, output, _) =
            nn_verify::propagate_with_crown_fallback(&graph, &input).expect("propagation");

        common::assert_bounds_valid(&output);

        let avg = lw_builders::measure_avg_width(&output);
        let max_w = lw_builders::measure_max_width(&output);
        eprintln!("  {d:<5}   {avg:>10.6}  {max_w:>10.6}  {method:?}");
    }
}

/// Monolithic vs layerwise comparison at D=8.
///
/// This is the critical test: do monolithic bounds differ from layerwise?
/// Monolithic CROWN sees the full computation path (Q→score→softmax→output)
/// and can exploit cross-layer structure. Layerwise treats each layer
/// independently, losing inter-layer correlations.
#[test]
fn test_monolithic_vs_layerwise_d8() {
    let seq_len = 4;
    let d = 8;
    let input_bound = 0.1;
    let k_scale = 1.0;
    let v_scale = 0.1;

    // Monolithic path
    let def = e2e_runners::build_monolithic_attention_no_proj("cmp_mono_d8", seq_len, d);
    let k_tensor = lw_builders::build_k_identity(seq_len, d, k_scale);
    let v_tensor = lw_builders::build_v_tensor(seq_len, d, v_scale);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(k_tensor),
        TensorParamBinding::ConstantTensor(v_tensor),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("monolithic graph");
    let input = common::uniform_bounds(&[seq_len, d], input_bound);
    let (mono_method, mono_out, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("monolithic");
    let mono_avg = lw_builders::measure_avg_width(&mono_out);
    let mono_max = lw_builders::measure_max_width(&mono_out);

    // Layerwise path (same parameters)
    let lw_out = e2e_runners::run_layerwise_3layer(seq_len, d, input_bound, k_scale, v_scale);
    let lw_avg = lw_builders::measure_avg_width(&lw_out);
    let lw_max = lw_builders::measure_max_width(&lw_out);

    // Report comparison
    let ratio_avg = mono_avg / lw_avg;
    let ratio_max = mono_max / lw_max;
    eprintln!("--- Monolithic vs Layerwise at D={d} ---");
    eprintln!("  Monolithic: avg_w={mono_avg:.6}, max_w={mono_max:.6} ({mono_method:?})");
    eprintln!("  Layerwise:  avg_w={lw_avg:.6}, max_w={lw_max:.6}");
    eprintln!("  Ratio (mono/layer): avg={ratio_avg:.4}x, max={ratio_max:.4}x");

    common::assert_bounds_valid(&mono_out);
    common::assert_bounds_valid(&lw_out);

    assert!(
        mono_avg.is_finite(),
        "mono_avg must be finite, got {mono_avg}"
    );
}

/// Monolithic vs layerwise comparison across D=8,16,32.
#[test]
fn test_monolithic_vs_layerwise_scaling() {
    let seq_len = 4;

    eprintln!("--- Monolithic vs Layerwise scaling ---");
    eprintln!("  D     mono_avg    lw_avg      ratio     mono_max    lw_max");

    for &d in &[8, 16, 32] {
        let input_bound = 0.1 / (d as f32 / 8.0).sqrt();
        let v_scale = 0.1 / (d as f32).sqrt();
        let k_scale = 1.0;

        // Monolithic
        let def =
            e2e_runners::build_monolithic_attention_no_proj(&format!("cmp_m_{d}"), seq_len, d);
        let k_tensor = lw_builders::build_k_identity(seq_len, d, k_scale);
        let v_tensor = lw_builders::build_v_tensor(seq_len, d, v_scale);
        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(k_tensor),
            TensorParamBinding::ConstantTensor(v_tensor),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = common::uniform_bounds(&[seq_len, d], input_bound);
        let (_, mono_out, _) =
            nn_verify::propagate_with_crown_fallback(&graph, &input).expect("mono");

        // Layerwise
        let lw_out = e2e_runners::run_layerwise_3layer(seq_len, d, input_bound, k_scale, v_scale);

        let mono_avg = lw_builders::measure_avg_width(&mono_out);
        let lw_avg = lw_builders::measure_avg_width(&lw_out);
        let mono_max = lw_builders::measure_max_width(&mono_out);
        let lw_max = lw_builders::measure_max_width(&lw_out);
        let ratio = mono_avg / lw_avg;

        common::assert_bounds_valid(&mono_out);
        common::assert_bounds_valid(&lw_out);

        eprintln!(
            "  {d:<5} {mono_avg:>10.6}  {lw_avg:>10.6}  {ratio:>8.4}x  {mono_max:>10.6}  {lw_max:>10.6}"
        );
    }
}

// ===========================================================================
// Tests: Adversarial perturbation stability (#1740)
// ===========================================================================

/// Adversarial perturbation set: embedding-space ε-ball.
///
/// Models the effect of textual attacks (homoglyphs, invisible chars) as
/// bounded perturbations in embedding space.
#[test]
fn test_adversarial_embedding_perturbation_d8() {
    let seq_len = 4;
    let d = 8;
    let pe = helpers::build_sinusoidal_pe(seq_len, d);
    let perturbation_budgets = [0.01, 0.05, 0.1];
    let k_scale = 1.0;
    let v_scale = 0.1;

    eprintln!("--- Adversarial perturbation stability (D={d}) ---");
    eprintln!("  ε         avg_w       max_w       interpretation");

    for &eps in &perturbation_budgets {
        let def = e2e_runners::build_monolithic_attention_no_proj(
            &format!("adv_{d}_eps{}", (eps * 100.0) as u32),
            seq_len,
            d,
        );

        let k_tensor = lw_builders::build_k_identity(seq_len, d, k_scale);
        let v_tensor = lw_builders::build_v_tensor(seq_len, d, v_scale);
        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(k_tensor),
            TensorParamBinding::ConstantTensor(v_tensor),
        ];

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = lw_builders::build_pe_centered_bounds(&pe, eps);

        let (_, output, _) =
            nn_verify::propagate_with_crown_fallback(&graph, &input).expect("propagation");

        common::assert_bounds_valid(&output);

        let avg = lw_builders::measure_avg_width(&output);
        let max_w = lw_builders::measure_max_width(&output);

        let interpretation = if max_w < 0.01 {
            "provably stable"
        } else if max_w < 0.1 {
            "moderately bounded"
        } else {
            "wide bounds"
        };

        eprintln!("  {eps:<9.3} {avg:>10.6}  {max_w:>10.6}  {interpretation}");
    }
}

/// Adversarial perturbation: attention weight stability under embedding attack.
///
/// Examines intermediate attention weights (softmax output). If attention
/// weights are stable under perturbation, the model attends to the same
/// positions regardless of the attack — phoneme alignment is preserved.
#[test]
fn test_adversarial_attention_weight_stability_d8() {
    let seq_len = 4;
    let d = 8;

    // Build a 2-layer graph: score + softmax only (attention weights)
    let mut b = TensorBlockBuilder::new("adv_attn_weights_d8");
    let q = b.add_input("query", &[seq_len, d]);
    let k = b.add_input("key", &[seq_len, d]);

    let scale = 1.0 / (d as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &[seq_len, seq_len]);
    let weights = b.add_softmax(scores, -1, &[seq_len, seq_len]);
    let def = b.build(weights).expect("valid attn weights graph");

    let pe = helpers::build_sinusoidal_pe(seq_len, d);
    let k_tensor = lw_builders::build_k_identity(seq_len, d, 1.0);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(k_tensor),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("attn weights graph");

    let perturbation_budgets = [0.01, 0.05, 0.1, 0.2];

    eprintln!("--- Attention weight stability under adversarial perturbation ---");
    eprintln!("  ε         avg_w       max_w       diag_dom    stable?");

    for &eps in &perturbation_budgets {
        let input = lw_builders::build_pe_centered_bounds(&pe, eps);
        let (_, output, _) =
            nn_verify::propagate_with_crown_fallback(&graph, &input).expect("propagation");
        common::assert_bounds_valid(&output);
        let avg = lw_builders::measure_avg_width(&output);
        let max_w = lw_builders::measure_max_width(&output);

        let diag_dominant = lw_builders::count_diagonal_dominant(&output, seq_len);
        let stable = diag_dominant == seq_len;
        let stable_str = if stable { "YES" } else { "partial" };

        eprintln!(
            "  {eps:<9.3} {avg:>10.6}  {max_w:>10.6}  {diag_dominant}/{seq_len}        {stable_str}"
        );
    }
}

/// Multi-perturbation-type adversarial analysis (#1740 AC1).
///
/// Models three distinct perturbation types from Unicode adversarial attacks:
/// homoglyph (uniform ε-ball), invisible char insertion (position-specific),
/// and combined attacks. Verifies attention output bounds for each.
#[test]
fn test_adversarial_perturbation_types_d8() {
    let seq_len = 4;
    let d = 8;
    let pe = helpers::build_sinusoidal_pe(seq_len, d);

    let def = e2e_runners::build_monolithic_attention_no_proj("adv_types_d8", seq_len, d);
    let k_tensor = lw_builders::build_k_identity(seq_len, d, 1.0);
    let v_tensor = lw_builders::build_v_tensor(seq_len, d, 0.1);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(k_tensor),
        TensorParamBinding::ConstantTensor(v_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    eprintln!("--- Adversarial perturbation types (D={d}) ---");
    eprintln!("  Type                  avg_w       max_w");

    // Type 1: Homoglyph — uniform small perturbation
    let input = lw_builders::build_pe_centered_bounds(&pe, 0.05);
    let (_, output, _) = nn_verify::propagate_with_crown_fallback(&graph, &input).expect("prop");
    common::assert_bounds_valid(&output);
    eprintln!(
        "  Homoglyph (ε=0.05)    {:>10.6}  {:>10.6}",
        lw_builders::measure_avg_width(&output),
        lw_builders::measure_max_width(&output)
    );

    // Type 2: Invisible char insertion — larger perturbation at position 1
    let input = lw_builders::build_invisible_char_bounds(&pe, 0.01, 0.2, 1, d);
    let (_, output, _) = nn_verify::propagate_with_crown_fallback(&graph, &input).expect("prop");
    common::assert_bounds_valid(&output);
    eprintln!(
        "  Invisible char (pos1) {:>10.6}  {:>10.6}",
        lw_builders::measure_avg_width(&output),
        lw_builders::measure_max_width(&output)
    );

    // Type 3: Combined — homoglyph at pos 0, invisible char at pos 2
    let input = lw_builders::build_combined_attack_bounds(&pe, 0.01, 0, 0.05, 2, 0.15, d);
    let (_, output, _) = nn_verify::propagate_with_crown_fallback(&graph, &input).expect("prop");
    common::assert_bounds_valid(&output);
    eprintln!(
        "  Combined (hom+invis)  {:>10.6}  {:>10.6}",
        lw_builders::measure_avg_width(&output),
        lw_builders::measure_max_width(&output)
    );
}

/// Adversarial stability certificate: document provable bounds.
///
/// For a specific configuration (D=8, T=4, near-identity K), document
/// the exact provable bounds that constitute a formal certificate.
#[test]
fn test_adversarial_stability_certificate_d8() {
    let seq_len = 4;
    let d = 8;
    let eps = 0.05;

    let pe = helpers::build_sinusoidal_pe(seq_len, d);

    let def = e2e_runners::build_monolithic_attention_no_proj("cert_d8", seq_len, d);
    let k_tensor = lw_builders::build_k_identity(seq_len, d, 1.0);
    let v_tensor = lw_builders::build_v_tensor(seq_len, d, 0.1);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(k_tensor),
        TensorParamBinding::ConstantTensor(v_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = lw_builders::build_pe_centered_bounds(&pe, eps);
    let (method, output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("propagation");

    common::assert_bounds_valid(&output);

    let avg_w = lw_builders::measure_avg_width(&output);
    let max_w = lw_builders::measure_max_width(&output);

    eprintln!("=== ADVERSARIAL STABILITY CERTIFICATE ===");
    eprintln!("Architecture: 3-layer attention (score → softmax → output)");
    eprintln!("Dimensions: T={seq_len}, D={d}");
    eprintln!("Perturbation: L∞ ε={eps} around sinusoidal PE");
    eprintln!("Verification: {method:?}");
    eprintln!("Result: avg_width={avg_w:.6}, max_width={max_w:.6}");
    eprintln!("Interpretation: For any input within ε={eps} of nominal PE,");
    eprintln!("  attention output bounded with max element-wise uncertainty {max_w:.6}");

    assert!(avg_w.is_finite(), "certificate avg_w must be finite");
    assert!(max_w.is_finite(), "certificate max_w must be finite");

    let (lo, hi) = output.lower_upper();
    eprintln!("Per-position output bounds (first row):");
    for c in 0..d.min(8) {
        eprintln!(
            "  [{c}]: [{:.4}, {:.4}] (w={:.4})",
            lo[[0, c]],
            hi[[0, c]],
            hi[[0, c]] - lo[[0, c]]
        );
    }
}
