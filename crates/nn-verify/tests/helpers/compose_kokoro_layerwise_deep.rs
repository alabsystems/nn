// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Production-depth layerwise CROWN verification with 12+ normalization layers.
//!
//! The original 5-layer decomposition (`compose_kokoro_layerwise_d128.rs`) has
//! only 1 InstanceNorm layer. Production Kokoro has ~70 InstanceNorm/AdaIN
//! layers. Each normalization layer compounds interval width multiplicatively,
//! so bounds from the 5-layer model do not predict production behavior.
//!
//! This test uses `build_kokoro_layerwise_deep(dims, 12)` to chain 12 ResBlocks,
//! each containing 1 InstanceNorm + Snake + Conv1d + residual. This tests how
//! CROWN bounds compound through representative normalization depth.
//!
//! Part of #2573: Kokoro verification tests need production-representative depth.
//!
//! ## AC3 Calibration Result (D=64, 12 norms)
//!
//! Per-layer CROWN/IBP ratio = 1.0000 across all 16 layers. CROWN provides
//! ZERO tightening over IBP when each layer is a separate GraphNetwork. This
//! applies to ALL layer types (Conv1d, ReLU, InstanceNorm, Exp), not just
//! normalization layers. Per-layer decomposition eliminates all cross-layer
//! CROWN benefit. Multi-ResBlock sub-graphs (R1-212 plan) are the next step.

#[path = "kokoro_scaled_pipeline.rs"]
mod deep_scaled_helpers;
use deep_scaled_helpers as helpers;

#[path = "kokoro_scaled_layerwise.rs"]
mod deep_layerwise_helpers;

use super::common::{
    kokoro_recording::pipeline_tight_stage_count,
    kokoro_weights::{bt_max_width, uniform_bt},
};
use deep_layerwise_helpers::build_kokoro_layerwise_deep;
use deep_scaled_helpers::KokoroDims;
use nn_tts_verify::verify_layerwise;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, PropMethod};

/// Number of ResBlocks (each with 1 InstanceNorm) in the deep pipeline.
/// Representative of production depth (12 InstanceNorm ≈ 1 upsample stage).
const NUM_RESBLOCKS: usize = 12;

/// Log per-stage width growth through the pipeline certificate.
///
/// Reports the width expansion at each normalization-containing stage,
/// tracking how InstanceNorm compounds interval bounds.
fn log_per_stage_width(cert: &nn_tts_verify::PipelineCertificate, label: &str) {
    eprintln!(
        "{label}: per-stage width analysis ({} stages):",
        cert.stages.len()
    );
    let mut prev_width = None;
    for (i, stage) in cert.stages.iter().enumerate() {
        let lo_min = stage
            .output_lower
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let hi_max = stage
            .output_upper
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let width = hi_max - lo_min;
        let ratio = prev_width.map(|p: f64| if p > 0.0 { width / p } else { f64::NAN });
        let ratio_str = match ratio {
            Some(r) if r.is_finite() => format!(" (×{r:.3})"),
            _ => String::new(),
        };
        eprintln!(
            "  stage {i:2}: [{lo_min:.6}, {hi_max:.6}] width={width:.6}{ratio_str} method={}",
            stage.method
        );
        prev_width = Some(width);
    }
    // Summary: total expansion from first to last stage.
    let first_lo = cert
        .e2e_input_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let first_hi = cert
        .e2e_input_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let input_width = first_hi - first_lo;
    let output_lo = cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let output_hi = cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let output_width = output_hi - output_lo;
    let total_expansion = if input_width > 0.0 {
        output_width / input_width
    } else {
        f64::NAN
    };
    eprintln!(
        "{label}: total expansion: input_width={input_width:.6} → output_width={output_width:.6} \
         (×{total_expansion:.3})"
    );
}

// ===========================================================================
// D=64 deep layerwise (fast sanity check with 12 InstanceNorm layers)
// ===========================================================================

/// D=64, 12 ResBlocks: verify the deep pipeline constructs and propagates
/// without errors. Reports per-stage width growth.
#[test]
fn test_kokoro_layerwise_deep_d64() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    assert_eq!(
        layers.len(),
        NUM_RESBLOCKS + 4,
        "expected text_enc + vocoder_pre + upsample + {NUM_RESBLOCKS} resblocks + output"
    );

    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise(&layers, &initial).expect("D=64 deep layerwise");

    assert!(cert.is_valid, "D=64 deep pipeline must be valid");

    // All junctions must be compatible.
    for (i, j) in cert.junctions.iter().enumerate() {
        assert!(j.shape_compatible, "D=64 deep junction {i}: shape mismatch");
        assert!(
            j.bounds_contained,
            "D=64 deep junction {i}: bounds violation={:.6}",
            j.max_violation
        );
    }

    // P1: exp output must have positive lower bound (non-silence).
    let lo_min = cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        lo_min > 0.0,
        "D=64 deep P1: expected positive output, got {lo_min}"
    );

    // P2: output must be finite (bounded).
    let hi_max = cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        hi_max.is_finite(),
        "D=64 deep P2: expected finite output, got {hi_max}"
    );

    log_per_stage_width(&cert, "D=64 deep (12 norms)");
}

// ===========================================================================
// D=128 deep layerwise (production-representative normalization depth)
// ===========================================================================

/// D=128, 12 ResBlocks: the critical test for production-representative depth.
///
/// With 12 InstanceNorm layers, this reveals how CROWN bounds compound through
/// normalization depth. The 5-layer test (1 InstanceNorm) shows CROWN width
/// ~0.05; with 12 norms the width may be significantly larger.
///
/// Part of #2573 AC1: 10+ normalization layers in verification compose tests.
#[test]
fn test_kokoro_layerwise_deep_d128() {
    let dims = KokoroDims::d128();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise(&layers, &initial).expect("D=128 deep layerwise");

    assert!(cert.is_valid, "D=128 deep pipeline must be valid");

    // Junction validity.
    for (i, j) in cert.junctions.iter().enumerate() {
        assert!(
            j.shape_compatible,
            "D=128 deep junction {i}: shape mismatch"
        );
        assert!(
            j.bounds_contained,
            "D=128 deep junction {i}: bounds violation={:.6}",
            j.max_violation
        );
    }

    // P1 and P2: still must hold even with deep normalization.
    let lo_min = cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let hi_max = cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        lo_min > 0.0,
        "D=128 deep P1: expected positive, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "D=128 deep P2: expected finite, got {hi_max}"
    );

    // Count sound CROWN-family stages (CROWN/AlphaCrown/BetaCrown/Analytical).
    let crown_count = pipeline_tight_stage_count(&cert);
    let total = cert.stages.len();
    eprintln!(
        "D=128 deep: {crown_count}/{total} stages used sound CROWN-family propagation ({:.0}%)",
        crown_count as f64 / total as f64 * 100.0
    );

    // Per-stage width growth — the key diagnostic for #2573.
    log_per_stage_width(&cert, "D=128 deep (12 norms)");

    // Output width: report but do not assert a specific threshold yet.
    // The 5-layer test asserts < 10.0; with 12 norms it may be wider.
    // #2573 AC3 will calibrate the threshold based on this data.
    let output_width = hi_max - lo_min;
    eprintln!("D=128 deep output width: {output_width:.6}");
}

// ===========================================================================
// CROWN vs IBP per-layer comparison (AC3: vacuity threshold calibration)
// ===========================================================================

/// Per-layer CROWN vs IBP tightening ratio measurement.
struct LayerTighteningResult {
    layer_index: usize,
    ibp_width: f32,
    crown_width: f32,
    crown_method: PropMethod,
    /// ibp_width / crown_width. 1.0 = no tightening, >1.0 = CROWN tighter.
    ratio: f32,
    crown_fallback: Option<String>,
}

/// Run both IBP and CROWN independently on each layer, chaining output bounds.
///
/// Unlike `verify_layerwise` (which calls `propagate_with_crown_fallback`),
/// this always runs BOTH methods per layer to produce a direct comparison.
/// The CROWN/IBP ratio reveals whether per-layer CROWN provides meaningful
/// tightening beyond IBP through InstanceNorm layers.
///
/// Part of #2573 AC3: vacuity threshold calibration.
fn measure_per_layer_tightening(
    layers: &[(
        nn_dsl::tensor_ir::TensorKernelDef,
        Vec<nn_verify::TensorParamBinding>,
    )],
    initial_bounds: &BoundedTensor,
    label: &str,
) -> Vec<LayerTighteningResult> {
    let mut results = Vec::with_capacity(layers.len());
    let mut current_ibp_bounds = initial_bounds.clone();
    let mut current_crown_bounds = initial_bounds.clone();

    for (i, (layer, bindings)) in layers.iter().enumerate() {
        let graph = tensor_kernel_to_graph(layer, bindings)
            .unwrap_or_else(|e| panic!("{label} layer {i} graph build: {e}"));

        // IBP propagation (always succeeds for supported layer types).
        let ibp_output = graph
            .propagate_ibp(&current_ibp_bounds)
            .unwrap_or_else(|e| panic!("{label} layer {i} IBP: {e}"));
        let ibp_width = bt_max_width(&ibp_output);

        // CROWN propagation (may fall back to IBP on unsupported layers).
        let (method, crown_output, fallback) =
            nn_verify::propagate_with_crown_fallback(&graph, &current_crown_bounds)
                .unwrap_or_else(|e| panic!("{label} layer {i} CROWN: {e}"));
        let crown_width = bt_max_width(&crown_output);

        let ratio = if crown_width > 0.0 {
            ibp_width / crown_width
        } else {
            f32::INFINITY
        };

        results.push(LayerTighteningResult {
            layer_index: i,
            ibp_width,
            crown_width,
            crown_method: method,
            ratio,
            crown_fallback: fallback,
        });

        // Chain: use each method's own output as next layer's input.
        current_ibp_bounds = ibp_output;
        current_crown_bounds = crown_output;
    }

    results
}

/// Log per-layer CROWN/IBP comparison in a structured table.
fn log_tightening_table(results: &[LayerTighteningResult], label: &str) {
    eprintln!("\n=== {label}: CROWN vs IBP per-layer tightening ===");
    eprintln!(
        "{:<8} {:<10} {:<14} {:<14} {:<10} {:<10}",
        "Layer", "Method", "IBP Width", "CROWN Width", "Ratio", "Tighter?"
    );

    let mut crown_tighter_count = 0;
    let mut crown_equal_count = 0;

    for r in results {
        let tighter = if r.ratio > 1.01 {
            crown_tighter_count += 1;
            "YES"
        } else {
            crown_equal_count += 1;
            "~same"
        };
        eprintln!(
            "{:<8} {:<10} {:<14.6} {:<14.6} {:<10.4} {:<10}",
            r.layer_index,
            format!("{:?}", r.crown_method),
            r.ibp_width,
            r.crown_width,
            r.ratio,
            tighter,
        );
        if let Some(reason) = &r.crown_fallback {
            eprintln!("         fallback: {reason}");
        }
    }

    let total = results.len();
    eprintln!(
        "\n{label} summary: {crown_tighter_count}/{total} layers CROWN tighter, \
         {crown_equal_count}/{total} layers CROWN = IBP"
    );

    if let Some(last) = results.last() {
        eprintln!(
            "{label} e2e: IBP_width={:.6}, CROWN_width={:.6}, ratio={:.4}x",
            last.ibp_width, last.crown_width, last.ratio
        );
    }
}

/// D=64, 12 ResBlocks: per-layer CROWN vs IBP comparison.
///
/// Measures whether CROWN provides any tightening over IBP through
/// InstanceNorm layers. Result: CROWN = IBP (ratio 1.0000) for ALL layer
/// types, not just InstanceNorm. Per-layer decomposition eliminates all
/// cross-layer CROWN benefit.
///
/// Part of #2573 AC3: vacuity threshold calibration data.
#[test]
fn test_kokoro_layerwise_deep_d64_crown_vs_ibp() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);

    let results = measure_per_layer_tightening(&layers, &initial, "D=64 deep");
    log_tightening_table(&results, "D=64 deep (12 norms)");

    // Soundness: CROWN width must be <= IBP width (+ epsilon for fp).
    for r in &results {
        assert!(
            r.crown_width <= r.ibp_width + 1e-3,
            "Layer {}: CROWN width {} > IBP width {} (soundness)",
            r.layer_index,
            r.crown_width,
            r.ibp_width,
        );
    }

    // All bounds must be finite.
    for r in &results {
        assert!(
            r.ibp_width.is_finite(),
            "Layer {} IBP not finite",
            r.layer_index
        );
        assert!(
            r.crown_width.is_finite(),
            "Layer {} CROWN not finite",
            r.layer_index
        );
    }
}

// NOTE: D=128 CROWN vs IBP comparison test removed. The D=64 test (457s, 16 layers)
// proved CROWN/IBP ratio = 1.0000 across ALL layer types. This is a structural result:
// per-layer decomposition gives CROWN no multi-layer structure to exploit, regardless
// of dimension. D=128 would produce the same ratio = 1.0 but exceed the 10-min cargo
// test timeout (D=64 already took 7.6 min). The non-comparison D=128 deep test above
// (test_kokoro_layerwise_deep_d128) still verifies bound propagation at production scale.
