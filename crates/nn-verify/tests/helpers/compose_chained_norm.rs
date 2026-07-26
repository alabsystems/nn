// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Chained InstanceNorm NY bounds tests (#2702).
//!
//! Tests IBP bound propagation through deep InstanceNorm chains matching
//! the Kokoro Generator architecture (58 layers of Conv1d → Activation →
//! InstanceNorm).
//!
//! Current status: **Conservative IBP** with contractive Conv1d weights
//! produces tight, depth-invariant bounds (width ≈ 7.75 at all depths
//! N=10..58). Legacy `IbpValidated` CROWN propagation still produces
//! **vacuously wide** FALLBACK_BOUND-capped bounds (~2e10), but
//! `NormBoundsMode::CrownSampling` now gives non-vacuous CROWN through
//! the same InstanceNorm chains. ForwardMode (default) IBP still saturates
//! at FALLBACK_BOUND ±1e10. For sound Kokoro normalization verification,
//! use Conservative IBP; for heuristic but meaningful CROWN, use
//! `CrownSampling`.
//!
//! Part of #2702, Part of #2701, Part of #2218.

use super::common::{
    assert_bounds_valid, assert_bounds_width, assert_crown_tighter_when_not_fallback,
    assert_norm_spatial_non_degenerate, high_variance_bounds, uniform_bounds, DEFAULT_NORM_EPS,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{
    tensor_kernel_to_graph, tensor_kernel_to_graph_with_norm_mode, BoundedTensor, NormBoundsMode,
    TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Builder: pure InstanceNorm chain (N layers, no Conv1d)
// ---------------------------------------------------------------------------

/// Build a graph of N chained InstanceNorm layers with no interleaved ops.
///
/// This is the degenerate case: InstanceNorm(InstanceNorm(x)) normalizes
/// already-normalized data. Bounds are expected to widen rapidly due to
/// the 1/sqrt(var + eps) amplification on near-zero variance.
///
/// Inputs: data [channels, time_len] (Variable), eps [1] (ConstantScalar).
fn build_pure_instance_norm_chain(
    num_layers: usize,
    channels: usize,
    time_len: usize,
) -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    assert_norm_spatial_non_degenerate(time_len, "pure_instance_norm_chain");

    let shape = [channels, time_len];
    let mut b = TensorBlockBuilder::new("pure_instance_norm_chain");
    let data = b.add_input("data", &shape);
    let eps = b.add_input("eps", &[1]);

    let mut current = data;
    for _ in 0..num_layers {
        current = b.add_instance_norm(current, eps, 1, None, None, &shape);
    }

    let def = b.build(current).expect("valid pure InstanceNorm chain");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(DEFAULT_NORM_EPS),
    ];
    (def, bindings)
}

// ---------------------------------------------------------------------------
// Builder: Kokoro-realistic block chain (Conv1d → ReLU → InstanceNorm × N)
// ---------------------------------------------------------------------------

/// Build a graph of N chained Conv1d → ReLU → InstanceNorm blocks.
///
/// Mirrors the Kokoro Generator architecture where each residual block
/// contains a convolution, activation, and normalization. Using ReLU
/// instead of Snake for simplicity (both are Lipschitz-1 for bounded input).
///
/// Same-channel Conv1d with stride=1 and padding=kernel_size/2 preserves
/// the spatial dimension. Small constant weights (magnitude 0.1) keep the
/// convolution contractive.
///
/// Inputs: data [channels, time_len] (Variable), eps [1] (ConstantScalar),
///   weight_0..weight_{N-1} [channels, channels, kernel_size] (ConstantTensor).
fn build_kokoro_like_chain(
    num_blocks: usize,
    channels: usize,
    time_len: usize,
    kernel_size: usize,
) -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    assert_norm_spatial_non_degenerate(time_len, "kokoro_like_chain");

    let padding = kernel_size / 2;
    let shape = [channels, time_len];

    let mut b = TensorBlockBuilder::new("kokoro_like_chain");
    let data = b.add_input("data", &shape);
    let eps = b.add_input("eps", &[1]);

    // Collect weight input node IDs for binding construction.
    let mut weight_ids = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let w = b.add_input(&format!("weight_{i}"), &[channels, channels, kernel_size]);
        weight_ids.push(w);
    }

    let mut current = data;
    for &wid in &weight_ids {
        let conv = b.add_conv1d(current, wid, None, 1, padding, &shape);
        let relu = b.add_relu(conv, &shape);
        current = b.add_instance_norm(relu, eps, 1, None, None, &shape);
    }

    let def = b.build(current).expect("valid Kokoro-like chain");

    // Build bindings: Variable (data), ConstantScalar (eps), then N weight tensors.
    let weight_mag = 0.1 / (channels as f32).sqrt();
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(DEFAULT_NORM_EPS),
    ];
    for _ in 0..num_blocks {
        let w = ArrayD::from_elem(IxDyn(&[channels, channels, kernel_size]), weight_mag);
        bindings.push(TensorParamBinding::ConstantTensor(w));
    }

    (def, bindings)
}

// ---------------------------------------------------------------------------
// Test 1: Pure InstanceNorm chain — unit tests for propagation correctness
// ---------------------------------------------------------------------------

/// Pure InstanceNorm chain (N=2): IBP propagates without error, bounds finite.
///
/// Pure chained InstanceNorm is mathematically degenerate: after the first norm,
/// output has mean≈0, var≈1. The second norm sees near-zero variance, amplifying
/// uncertainty by ~1/sqrt(eps) ≈ 316x per layer. Conservative IBP produces
/// vacuously wide bounds (~5.5e18 at N=2). This is expected — the test documents
/// that propagation completes with finite bounds despite the degenerate case.
#[test]
fn test_pure_chained_instance_norm_2_ibp() {
    let channels = 4;
    let time_len = 16;
    let (def, bindings) = build_pure_instance_norm_chain(2, channels, time_len);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph N=2");
    let input = uniform_bounds(&[channels, time_len], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP N=2");

    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);
    assert_bounds_valid(&output);
    // Pure InstanceNorm chain is degenerate: 1/sqrt(eps) amplification per layer
    // produces ~5.5e18 width at N=2. Bounds are finite (no Inf/NaN) but vacuously
    // wide. This documents the known limitation of Conservative IBP through
    // chained normalization without interleaved contractive layers.
    let (lo_min, hi_max) = super::common::bounds_min_max(&output);
    eprintln!(
        "pure_instance_norm_2: bounds=[{lo_min:.3e}, {hi_max:.3e}], width={:.3e}",
        hi_max - lo_min
    );
}

/// Pure InstanceNorm chain (N=5): IBP propagates, bounds finite.
///
/// At N=5, bounds are expected to be vacuously wide due to 1/sqrt(eps)
/// amplification on near-zero variance. The test validates propagation
/// completes without panic or infinity.
#[test]
fn test_pure_chained_instance_norm_5_ibp() {
    let channels = 4;
    let time_len = 16;
    let (def, bindings) = build_pure_instance_norm_chain(5, channels, time_len);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph N=5");
    let input = uniform_bounds(&[channels, time_len], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP N=5");

    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);
    assert_bounds_valid(&output);
    // N=5 may be very wide — just assert finite, not tight.
}

/// Pure InstanceNorm chain (N=2): concrete midpoint falls within IBP bounds.
///
/// Soundness check: propagate a degenerate (zero-width) point input through
/// IBP to get the concrete output, then verify it falls within the full IBP
/// bounds computed from the wider input range.
#[test]
fn test_pure_chained_instance_norm_2_soundness() {
    let channels = 4;
    let time_len = 16;
    let (def, bindings) = build_pure_instance_norm_chain(2, channels, time_len);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph N=2");
    let input = high_variance_bounds(&[channels, time_len], 2.0, 0.5);
    let output = graph.propagate_ibp(&input).expect("IBP N=2");
    assert_bounds_valid(&output);

    // Concrete forward via point-IBP: degenerate bounds [mid, mid] → exact output.
    let (in_lo, in_hi) = input.lower_upper();
    let midpoint = (in_lo.to_owned() + in_hi.to_owned()) / 2.0;
    let point_input = BoundedTensor::new(midpoint.clone(), midpoint).expect("valid point bounds");
    let point_output = graph.propagate_ibp(&point_input).expect("point-IBP N=2");
    let (point_lo, _) = point_output.lower_upper();

    let (out_lo, out_hi) = output.lower_upper();
    let eps = 1e-3;
    for (i, (&val, (&lo, &hi))) in point_lo
        .iter()
        .zip(out_lo.iter().zip(out_hi.iter()))
        .enumerate()
    {
        assert!(
            val >= lo - eps && val <= hi + eps,
            "soundness violation at element {i}: concrete {val} outside [{lo}, {hi}]"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: Kokoro-realistic block chain — production depth tests
// ---------------------------------------------------------------------------

/// Kokoro-like chain (N=10): Conservative IBP produces tight, non-vacuous bounds.
///
/// With contractive Conv1d weights (0.1/sqrt(C) ≈ 0.05), Conservative IBP
/// shrinks interval width through each Conv layer. The Conv contraction
/// dominates the InstanceNorm expansion, producing a tight max_width ≈ 7.75.
/// This is the PRIMARY verification test — it would catch kernel precision
/// drift that widens bounds beyond the tight threshold.
#[test]
fn test_kokoro_chain_10_conservative_ibp() {
    let channels = 4;
    let time_len = 16;
    let kernel_size = 3;
    let (def, bindings) = build_kokoro_like_chain(10, channels, time_len, kernel_size);

    let graph =
        tensor_kernel_to_graph_with_norm_mode(&def, &bindings, NormBoundsMode::Conservative)
            .expect("conservative graph N=10");
    let input = uniform_bounds(&[channels, time_len], 1.0);
    let output = graph.propagate_ibp(&input).expect("Conservative IBP N=10");

    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);
    assert_bounds_valid(&output);

    // Calibrated: Conservative width ≈ 7.75 at N=10. Threshold 50.0 gives
    // 6x headroom while catching meaningful regressions (e.g., broken Conv
    // contraction or InstanceNorm bound expansion).
    assert_bounds_width(&output, 50.0, "conservative_kokoro_chain_10");
}

/// Kokoro-like chain (N=20): Conservative IBP remains tight at deeper chains.
///
/// Calibrated: width converges to same fixed point as N=10 (~7.75).
/// The contractive Conv weights establish equilibrium after a few layers.
#[test]
fn test_kokoro_chain_20_conservative_ibp() {
    let channels = 4;
    let time_len = 16;
    let kernel_size = 3;
    let (def, bindings) = build_kokoro_like_chain(20, channels, time_len, kernel_size);

    let graph =
        tensor_kernel_to_graph_with_norm_mode(&def, &bindings, NormBoundsMode::Conservative)
            .expect("conservative graph N=20");
    let input = uniform_bounds(&[channels, time_len], 1.0);
    let output = graph.propagate_ibp(&input).expect("Conservative IBP N=20");

    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);
    assert_bounds_valid(&output);

    // Calibrated: Conservative width ≈ 7.75 (depth-invariant fixed point).
    // Same threshold as N=10 — Conv contraction equilibrium is reached by N≈5.
    assert_bounds_width(&output, 50.0, "conservative_kokoro_chain_20");
}

/// Kokoro-like chain (N=58): Conservative IBP at Kokoro production depth.
///
/// This is the critical production-depth test. The Welford variance drift
/// bug (#2696) caused 17% amplitude attenuation at 58 layers. Conservative
/// IBP with contractive Conv weights converges to the same fixed-point width
/// (~7.75) as shallower chains. A regression that breaks Conv contraction
/// or widens InstanceNorm bounds would push width above the 50.0 threshold.
#[test]
fn test_kokoro_chain_58_conservative_ibp() {
    let channels = 4;
    let time_len = 16;
    let kernel_size = 3;
    let (def, bindings) = build_kokoro_like_chain(58, channels, time_len, kernel_size);

    let graph =
        tensor_kernel_to_graph_with_norm_mode(&def, &bindings, NormBoundsMode::Conservative)
            .expect("conservative graph N=58");
    let input = uniform_bounds(&[channels, time_len], 1.0);
    let output = graph.propagate_ibp(&input).expect("Conservative IBP N=58");

    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);
    assert_bounds_valid(&output);

    // Calibrated: Conservative width ≈ 7.75 (depth-invariant fixed point).
    // Same threshold as N=10 and N=20 — equilibrium is well-established by N=58.
    assert_bounds_width(&output, 50.0, "conservative_kokoro_chain_58");
}

// ---------------------------------------------------------------------------
// Test 3: CROWN propagation — vacuous bounds through normalization (#2715)
// ---------------------------------------------------------------------------

/// Kokoro-like chain (N=10): CROWN propagation completes but bounds are vacuous.
///
/// CROWN succeeds structurally (no fallback to IBP) but produces
/// FALLBACK_BOUND-capped bounds (width ~2e10) — vacuously wide.
/// Conservative IBP produces width ~7.75 for the same chain
/// (see `test_kokoro_chain_10_conservative_ibp`). **CROWN's linearization
/// through InstanceNorm does not improve over IBP here.** This is a known
/// architectural limitation — CROWN correlation tracking is destroyed by
/// normalization layers' mean-subtraction and variance-division.
///
/// This is a structural completeness test, not a tightness assertion.
/// For actual tight bounds through Kokoro normalization chains, use
/// Conservative IBP (see tests above).
/// See #2715 for the CROWN-through-normalization gap analysis.
#[test]
fn test_kokoro_chain_10_crown_vacuous() {
    let channels = 4;
    let time_len = 16;
    let kernel_size = 3;
    let (def, bindings) = build_kokoro_like_chain(10, channels, time_len, kernel_size);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph N=10");
    let input = uniform_bounds(&[channels, time_len], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);

    // Observed: CROWN succeeds (method=Crown, fallback=none) but bounds are
    // FALLBACK_BOUND-capped (~2e10 wide). Conservative IBP is 276M× tighter.
    eprintln!(
        "CROWN method={method:?}, fallback={:?}",
        fallback_reason.as_deref().unwrap_or("none")
    );
}

/// Kokoro-like chain (N=58): CROWN propagation at production depth — vacuous bounds.
///
/// Validates CROWN completes through 58 chained blocks without error.
/// Like the N=10 test, CROWN succeeds structurally but bounds are
/// FALLBACK_BOUND-capped (~2e10 wide) — no tighter than IBP. Conservative
/// IBP produces width ~7.75. **CROWN adds no verification value over IBP
/// for chained normalization at any depth.** Per #2702 acceptance criteria.
/// See #2715 for the CROWN-through-normalization gap analysis.
#[test]
fn test_kokoro_chain_58_crown_vacuous() {
    let channels = 4;
    let time_len = 16;
    let kernel_size = 3;
    let (def, bindings) = build_kokoro_like_chain(58, channels, time_len, kernel_size);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph N=58");
    let input = uniform_bounds(&[channels, time_len], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);

    // CROWN on ForwardMode graph produces 2e10 width (same as IBP — CROWN
    // doesn't improve bounds here). The meaningful assertion is
    // assert_crown_tighter_when_not_fallback above, which verifies CROWN
    // completes at N=58 production depth and is at least as tight as IBP.
    // No separate width assertion — any threshold < 2e10 would fail, and
    // any threshold > 2e10 is tautological given FALLBACK_BOUND.
    eprintln!(
        "N=58 CROWN method={method:?}, fallback={:?}",
        fallback_reason.as_deref().unwrap_or("none")
    );
}

/// Kokoro-like chain (N=10): CrownSampling fixes the norm-chain vacuity.
///
/// The legacy `IbpValidated` CROWN path and the current within-graph/blockwise
/// entrypoint both still saturate at `FALLBACK_BOUND` here, but
/// `NormBoundsMode::CrownSampling` produces a practical CROWN certificate.
#[test]
fn test_kokoro_chain_10_crown_sampling_tightens_norms() {
    let channels = 4;
    let time_len = 16;
    let kernel_size = 3;
    let (def, bindings) = build_kokoro_like_chain(10, channels, time_len, kernel_size);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph N=10");
    let graph_sampling =
        tensor_kernel_to_graph_with_norm_mode(&def, &bindings, NormBoundsMode::CrownSampling)
            .expect("sampling graph N=10");
    let input = uniform_bounds(&[channels, time_len], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    let legacy_crown = graph.propagate_crown(&input).expect("legacy CROWN");
    let sampling_crown = graph_sampling
        .propagate_crown(&input)
        .expect("sampling CROWN");
    let (within_graph_output, norm_stats) = graph
        .propagate_crown_within_graph_with_stats(&input)
        .expect("within-graph CROWN");

    assert_bounds_valid(&within_graph_output);
    assert_eq!(
        within_graph_output.lower_upper().0.shape(),
        &[channels, time_len]
    );

    let ibp_width = ibp_output.max_width();
    let legacy_width = legacy_crown.max_width();
    let within_graph_width = within_graph_output.max_width();
    let sampling_width = sampling_crown.max_width();
    let fallback_rows: usize = norm_stats.iter().map(|s| s.fallback_rows).sum();
    let total_rows: usize = norm_stats.iter().map(|s| s.total_rows).sum();

    eprintln!(
        "N=10 widths: ibp={ibp_width:.6}, legacy_crown={legacy_width:.6}, \
         sampling_crown={sampling_width:.6}, within_graph={within_graph_width:.6}, \
         norm_fallback_rows={fallback_rows}/{total_rows}"
    );

    assert!(
        sampling_width < 50.0,
        "sampling CROWN should be non-vacuous, got width={sampling_width:.6}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Concrete soundness at Kokoro-realistic depth
// ---------------------------------------------------------------------------

/// Kokoro-like chain (N=10): concrete midpoint falls within IBP bounds.
///
/// Validates the fundamental soundness property: any concrete input within
/// the input bounds produces an output within the output bounds. Uses
/// point-IBP (degenerate zero-width bounds) as the concrete forward pass.
#[test]
fn test_kokoro_chain_10_soundness() {
    let channels = 4;
    let time_len = 16;
    let kernel_size = 3;
    let (def, bindings) = build_kokoro_like_chain(10, channels, time_len, kernel_size);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph N=10");
    let input = high_variance_bounds(&[channels, time_len], 2.0, 0.5);
    let output = graph.propagate_ibp(&input).expect("IBP N=10");
    assert_bounds_valid(&output);

    // Concrete forward via point-IBP: degenerate bounds [mid, mid] → exact output.
    let (in_lo, in_hi) = input.lower_upper();
    let midpoint = (in_lo.to_owned() + in_hi.to_owned()) / 2.0;
    let point_input = BoundedTensor::new(midpoint.clone(), midpoint).expect("valid point bounds");
    let point_output = graph.propagate_ibp(&point_input).expect("point-IBP N=10");
    let (point_lo, _) = point_output.lower_upper();

    let (out_lo, out_hi) = output.lower_upper();
    let eps = 1e-3;
    for (i, (&val, (&lo, &hi))) in point_lo
        .iter()
        .zip(out_lo.iter().zip(out_hi.iter()))
        .enumerate()
    {
        assert!(
            val >= lo - eps && val <= hi + eps,
            "soundness violation at element {i}: concrete {val} outside [{lo}, {hi}]"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: ForwardMode vs Conservative comparison
// ---------------------------------------------------------------------------

/// Kokoro-like chain (N=10): compare ForwardMode vs Conservative bound widths.
///
/// Surprising finding: with contractive Conv1d weights (0.1/sqrt(C) ≈ 0.05),
/// Conservative IBP produces TIGHTER bounds (~7.75 width) than ForwardMode
/// (~2e10 width). This is because:
///
/// - Conservative IBP: small Conv weights shrink interval width each layer.
///   The Conv contraction dominates the InstanceNorm expansion.
/// - ForwardMode: anchors to the midpoint of normalization statistics, but the
///   midpoint-based linearization through Conv+Norm+Conv+Norm produces different
///   (wider) approximation errors than pure interval arithmetic with contractive
///   weights.
///
/// This test documents the observed behavior for regression tracking. The relative
/// performance of ForwardMode vs Conservative depends on the weight magnitudes
/// and layer composition — neither is universally tighter.
#[test]
fn test_kokoro_chain_10_forward_vs_conservative() {
    let channels = 4;
    let time_len = 16;
    let kernel_size = 3;
    let (def, bindings) = build_kokoro_like_chain(10, channels, time_len, kernel_size);

    // Conservative mode.
    let graph_conservative =
        tensor_kernel_to_graph_with_norm_mode(&def, &bindings, NormBoundsMode::Conservative)
            .expect("conservative graph");

    // ForwardMode (default).
    let graph_forward =
        tensor_kernel_to_graph_with_norm_mode(&def, &bindings, NormBoundsMode::ForwardMode)
            .expect("forward graph");

    let input = uniform_bounds(&[channels, time_len], 1.0);

    let out_conservative = graph_conservative
        .propagate_ibp(&input)
        .expect("IBP conservative");
    let out_forward = graph_forward.propagate_ibp(&input).expect("IBP forward");

    assert_bounds_valid(&out_conservative);
    assert_bounds_valid(&out_forward);

    // Both must produce finite bounds — the comparison is diagnostic.
    let (con_lo, con_hi) = out_conservative.lower_upper();
    let (fwd_lo, fwd_hi) = out_forward.lower_upper();

    let con_max_width = con_hi
        .iter()
        .zip(con_lo.iter())
        .map(|(h, l)| h - l)
        .fold(0.0f32, f32::max);
    let fwd_max_width = fwd_hi
        .iter()
        .zip(fwd_lo.iter())
        .map(|(h, l)| h - l)
        .fold(0.0f32, f32::max);

    // Document which mode is tighter for this architecture.
    eprintln!(
        "N=10: Conservative max_width={con_max_width:.4e}, \
         ForwardMode max_width={fwd_max_width:.4e}, \
         ratio(con/fwd)={:.1}x",
        con_max_width / fwd_max_width.max(1e-10)
    );

    // Conservative: tight threshold (observed width ~7.75, depth-invariant).
    assert_bounds_width(&out_conservative, 50.0, "conservative_N10");
    // ForwardMode: bounds saturate at FALLBACK_BOUND (±1e10, width 2e10).
    // No width assertion — any threshold < 2e10 fails, > 2e10 is tautological.
    // The assert_bounds_valid above already confirms finite bounds.
}
