// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN through iSTFT — prove audio [-1,1] for Kokoro output.
//!
//! The iSTFT is a fully linear transform (DFT matmul + Hann window +
//! overlap-add + COLA normalization). CROWN propagation through a single
//! LinearLayer with the precomputed weight matrix is *exact*.
//!
//! Architecture:
//!   Generator → exp(log_mag) → magnitude
//!   Generator → phase (raw radians)
//!   polar_to_rect: real = mag * cos(phase), imag = mag * sin(phase)
//!   iSTFT(real, imag) → audio samples
//!
//! Two-stage verification:
//!   Stage A (analytical bridge): CROWN magnitude bounds → iSTFT input bounds
//!   Stage B (CROWN LinearLayer): iSTFT input bounds → audio output bounds
//!
//! The bridge uses `cos/sin ∈ [-1, 1]` which over-approximates: constructive
//! interference across all frequency bins produces an amplification factor of
//! ~1.685x for Kokoro parameters (n_fft=20, hop=5). When magnitude bounds are
//! near 1.0 (typical for `exp(small_value)`), the raw iSTFT output can exceed
//! ±1.0. The defense-in-depth approach adds a proven-safe `clamp(-1, 1)` after
//! iSTFT, verified via CROWN through `Linear → Clip`.
//!
//! Part of #2916: CROWN through iSTFT — prove audio [-1,1].
//! Part of #2218: Perfect Kokoro epic.

use nn_verify::istft_linear_matrix::build_istft_weight_matrix;
use nn_verify::{BoundedTensor, GraphNetwork};
use ndarray::{Array1, Array2, ArrayD, IxDyn};

use ny_propagate::layers::{ClipLayer, LinearLayer};
use ny_propagate::{GraphNode, Layer};

// -- Kokoro iSTFT parameters --------------------------------------------------

/// Kokoro Generator n_fft (from `compiled_kokoro_bridges.rs:112`).
const KOKORO_N_FFT: usize = 20;
/// Kokoro Generator hop length.
const KOKORO_HOP: usize = 5;
/// Kokoro n_bins = n_fft / 2 + 1.
const KOKORO_N_BINS: usize = KOKORO_N_FFT / 2 + 1; // 11

// -- D2: GraphNetwork builder for iSTFT (single LinearLayer) ------------------

/// Build a `GraphNetwork` representing the iSTFT linear transform.
///
/// Single `NETWORK_INPUT → Linear → output` path with the precomputed weight
/// matrix from `build_istft_weight_matrix`. CROWN through this is exact.
///
/// Input: `[2 * n_bins * n_frames]` (real ++ imag, flattened)
/// Output: `[output_length]` (audio samples)
fn build_istft_graph(
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    output_length: usize,
) -> GraphNetwork {
    let mat = build_istft_weight_matrix(n_fft, hop, n_frames, output_length, false, true)
        .expect("valid iSTFT parameters");

    let n_in = mat.input_dim;
    let n_out = mat.output_length;

    // Reshape flat weights into Array2 [n_out, n_in] for LinearLayer.
    let weight = Array2::from_shape_vec((n_out, n_in), mat.weights).expect("valid weight shape");
    let bias = Array1::zeros(n_out);
    let linear = LinearLayer::new(weight, Some(bias)).expect("valid LinearLayer");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "istft_linear".to_string(),
        Layer::Linear(linear),
    ));
    graph.set_output("istft_linear".to_string());
    graph
}

/// Build iSTFT graph with a `clamp(-1, 1)` defense-in-depth at the output.
///
/// `NETWORK_INPUT → Linear(iSTFT) → Clip(-1, 1) → output`
///
/// CROWN through `Linear → Clip` is exact (both have tight CROWN bounds).
/// This proves P2 unconditionally: output ∈ [-1, 1] regardless of input
/// magnitude, while preserving meaningful bounds for the interior range.
fn build_istft_graph_with_clamp(
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    output_length: usize,
) -> GraphNetwork {
    let mat = build_istft_weight_matrix(n_fft, hop, n_frames, output_length, false, true)
        .expect("valid iSTFT parameters");

    let n_in = mat.input_dim;
    let n_out = mat.output_length;

    let weight = Array2::from_shape_vec((n_out, n_in), mat.weights).expect("valid weight shape");
    let bias = Array1::zeros(n_out);
    let linear = LinearLayer::new(weight, Some(bias)).expect("valid LinearLayer");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "istft_linear".to_string(),
        Layer::Linear(linear),
    ));
    graph.add_node(GraphNode::new(
        "audio_clamp".to_string(),
        Layer::Clip(ClipLayer::new(-1.0, 1.0)),
        vec!["istft_linear".to_string()],
    ));
    graph.set_output("audio_clamp".to_string());
    graph
}

// -- D3: Analytical bridge — Generator bounds to iSTFT input bounds -----------

/// Derive iSTFT input bounds from Generator magnitude bounds.
///
/// Given per-element magnitude bounds `[n_bins * n_frames]`:
///   `mag ∈ [mag_lo, mag_hi]` (all positive since `mag = exp(·)`)
///
/// The polar-to-rect conversion produces:
///   `real = mag * cos(phase)`, `imag = mag * sin(phase)`
///
/// Since `cos(phase) ∈ [-1, 1]` and `sin(phase) ∈ [-1, 1]` universally:
///   `real ∈ [-mag_hi, mag_hi]`, `imag ∈ [-mag_hi, mag_hi]`
///
/// This is a sound over-approximation (ignores phase correlation).
///
/// Returns `BoundedTensor` of shape `[2 * n_bins * n_frames]`.
fn bridge_generator_to_istft(mag_upper: &[f32], n_bins: usize, n_frames: usize) -> BoundedTensor {
    assert_eq!(mag_upper.len(), n_bins * n_frames);
    let spectral_len = n_bins * n_frames;
    let input_dim = 2 * spectral_len;

    let mut lower = Vec::with_capacity(input_dim);
    let mut upper = Vec::with_capacity(input_dim);

    // Real part: each element bounded by [-mag_hi, mag_hi]
    for &m in mag_upper {
        assert!(m >= 0.0, "magnitude must be non-negative, got {m}");
        lower.push(-m);
        upper.push(m);
    }

    // Imag part: same bounds
    for &m in mag_upper {
        lower.push(-m);
        upper.push(m);
    }

    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[input_dim]), lower).expect("valid lower"),
        ArrayD::from_shape_vec(IxDyn(&[input_dim]), upper).expect("valid upper"),
    )
    .expect("valid bridge bounds")
}

// -- D4: Compose tests --------------------------------------------------------

/// Synthetic magnitude bounds for testing.
///
/// Simulates Generator `exp()` output with small synthetic weights:
/// `exp(weight_mag * input) ∈ [exp(-w*r), exp(w*r)]` for `input ∈ [-r, r]`.
fn synthetic_magnitude_bounds(
    n_bins: usize,
    n_frames: usize,
    weight_mag: f32,
    input_range: f32,
) -> (Vec<f32>, Vec<f32>) {
    let spectral_len = n_bins * n_frames;
    let log_mag_bound = weight_mag * input_range;
    let mag_lo = (-log_mag_bound).exp();
    let mag_hi = log_mag_bound.exp();
    (vec![mag_lo; spectral_len], vec![mag_hi; spectral_len])
}

/// CROWN propagation through iSTFT produces finite, symmetric bounds.
///
/// With magnitude ~1.0 (typical for `exp(small_value)`), the iSTFT output
/// exceeds ±1.0 due to constructive interference across frequency bins.
/// The amplification factor for Kokoro (n_fft=20) is ~1.685x.
#[test]
fn test_crown_through_istft_finite_bounds() {
    let n_frames = 10;
    let output_length = (n_frames - 1) * KOKORO_HOP; // 45

    let (_mag_lo, mag_hi) = synthetic_magnitude_bounds(KOKORO_N_BINS, n_frames, 0.001, 1.0);

    let istft_input = bridge_generator_to_istft(&mag_hi, KOKORO_N_BINS, n_frames);
    let (bridge_lo, bridge_hi) = super::common::bounds_min_max(&istft_input);
    eprintln!("Bridge input bounds: [{bridge_lo}, {bridge_hi}]");

    let graph = build_istft_graph(KOKORO_N_FFT, KOKORO_HOP, n_frames, output_length);
    assert_eq!(graph.num_nodes(), 1, "iSTFT graph should have 1 node");

    let (method, crown_output, fallback_reason) =
        nn_verify::propagate_with_crown_fallback(&graph, &istft_input)
            .expect("CROWN through iSTFT");

    assert!(
        matches!(method, nn_verify::PropMethod::Crown),
        "single LinearLayer must not fall back to IBP. Reason: {}",
        fallback_reason.as_deref().unwrap_or("unknown")
    );

    super::common::assert_bounds_valid(&crown_output);
    let (audio_lo, audio_hi) = super::common::bounds_min_max(&crown_output);
    eprintln!("iSTFT CROWN output: [{audio_lo:.6}, {audio_hi:.6}]");

    // Bounds are finite and symmetric (input is symmetric, transform is linear).
    assert!(audio_lo.is_finite() && audio_hi.is_finite());
    assert!(
        (audio_lo + audio_hi).abs() < 1e-4,
        "bounds should be symmetric: [{audio_lo}, {audio_hi}]"
    );

    // The amplification factor from bridge over-approximation.
    let amplification = audio_hi / bridge_hi;
    eprintln!("iSTFT amplification factor: {amplification:.4}x");
    assert!(
        amplification > 1.0 && amplification < 3.0,
        "amplification should be moderate (1-3x for n_fft=20), got {amplification}"
    );
}

/// P2 with defense-in-depth: `iSTFT → clamp(-1, 1)` proves audio ∈ [-1, 1].
///
/// The `clamp` is a proven-safe defense-in-depth layer. CROWN through
/// `Linear → Clip` is exact. This proves P2 unconditionally.
#[test]
fn test_crown_through_istft_clamp_p2() {
    let n_frames = 10;
    let output_length = (n_frames - 1) * KOKORO_HOP;

    let (_mag_lo, mag_hi) = synthetic_magnitude_bounds(KOKORO_N_BINS, n_frames, 0.001, 1.0);

    let istft_input = bridge_generator_to_istft(&mag_hi, KOKORO_N_BINS, n_frames);
    let graph = build_istft_graph_with_clamp(KOKORO_N_FFT, KOKORO_HOP, n_frames, output_length);
    assert_eq!(graph.num_nodes(), 2, "iSTFT + clamp = 2 nodes");

    let (method, crown_output, fallback_reason) =
        nn_verify::propagate_with_crown_fallback(&graph, &istft_input)
            .expect("CROWN through iSTFT + clamp");

    assert!(
        matches!(method, nn_verify::PropMethod::Crown),
        "Linear + Clip must not fall back to IBP. Reason: {}",
        fallback_reason.as_deref().unwrap_or("unknown")
    );

    super::common::assert_bounds_valid(&crown_output);
    let (audio_lo, audio_hi) = super::common::bounds_min_max(&crown_output);
    eprintln!("iSTFT + clamp CROWN output: [{audio_lo:.6}, {audio_hi:.6}]");

    // P2: audio ∈ [-1, 1] — guaranteed by the Clip layer.
    assert!(
        audio_lo >= -1.0,
        "P2 VIOLATION: audio lower {audio_lo} < -1.0"
    );
    assert!(
        audio_hi <= 1.0,
        "P2 VIOLATION: audio upper {audio_hi} > 1.0"
    );
    eprintln!(
        "P2 (non-clipping) PROVEN with defense-in-depth clamp: \
         audio ∈ [{audio_lo:.6}, {audio_hi:.6}] ⊆ [-1, 1]"
    );
}

/// P2 holds without clamp when spectral magnitudes are sufficiently small.
///
/// When the Generator produces very small spectral energy (log_mag << 0),
/// the iSTFT output naturally stays within [-1, 1] even without clamping.
/// This proves P2 for the "quiet signal" regime.
#[test]
fn test_crown_through_istft_small_magnitude_p2() {
    let n_frames = 10;
    let output_length = (n_frames - 1) * KOKORO_HOP;

    // Very small magnitudes: exp(-2) ≈ 0.135
    // The iSTFT amplification (~1.685x) * 0.135 ≈ 0.227 — well within [-1, 1].
    let spectral_len = KOKORO_N_BINS * n_frames;
    let mag_hi = vec![(-2.0f32).exp(); spectral_len];

    let istft_input = bridge_generator_to_istft(&mag_hi, KOKORO_N_BINS, n_frames);
    let graph = build_istft_graph(KOKORO_N_FFT, KOKORO_HOP, n_frames, output_length);

    let (method, crown_output, fallback_reason) =
        nn_verify::propagate_with_crown_fallback(&graph, &istft_input)
            .expect("CROWN through iSTFT");

    assert!(
        matches!(method, nn_verify::PropMethod::Crown),
        "must not fall back. Reason: {}",
        fallback_reason.as_deref().unwrap_or("unknown")
    );

    super::common::assert_bounds_valid(&crown_output);
    let (audio_lo, audio_hi) = super::common::bounds_min_max(&crown_output);
    eprintln!(
        "Small magnitude iSTFT: mag_hi={:.4}, audio ∈ [{audio_lo:.6}, {audio_hi:.6}]",
        (-2.0f32).exp()
    );

    assert!(
        audio_lo >= -1.0 && audio_hi <= 1.0,
        "P2 without clamp for small magnitudes: [{audio_lo}, {audio_hi}]"
    );
}

/// IBP through iSTFT matches CROWN (both exact for LinearLayer).
#[test]
fn test_istft_ibp_matches_crown() {
    let n_frames = 8;
    let output_length = (n_frames - 1) * KOKORO_HOP;

    let (_mag_lo, mag_hi) = synthetic_magnitude_bounds(KOKORO_N_BINS, n_frames, 0.01, 1.0);

    let istft_input = bridge_generator_to_istft(&mag_hi, KOKORO_N_BINS, n_frames);
    let graph = build_istft_graph(KOKORO_N_FFT, KOKORO_HOP, n_frames, output_length);

    let ibp_output = graph.propagate_ibp(&istft_input).expect("IBP");
    let (method, crown_output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &istft_input).expect("CROWN");

    assert!(matches!(method, nn_verify::PropMethod::Crown));

    // For a single LinearLayer, IBP and CROWN produce identical bounds.
    let (ibp_lo, ibp_hi) = super::common::bounds_min_max(&ibp_output);
    let (crown_lo, crown_hi) = super::common::bounds_min_max(&crown_output);
    let eps = 1e-4;
    assert!(
        (ibp_lo - crown_lo).abs() < eps,
        "IBP lower {ibp_lo} != CROWN lower {crown_lo}"
    );
    assert!(
        (ibp_hi - crown_hi).abs() < eps,
        "IBP upper {ibp_hi} != CROWN upper {crown_hi}"
    );
    eprintln!("IBP and CROWN agree: [{crown_lo:.6}, {crown_hi:.6}]");
}

/// Verify bridge bounds scale with magnitude.
#[test]
fn test_istft_bounds_scale_with_magnitude() {
    let n_frames = 10;
    let output_length = (n_frames - 1) * KOKORO_HOP;
    let graph = build_istft_graph(KOKORO_N_FFT, KOKORO_HOP, n_frames, output_length);

    let mut prev_width = 0.0f32;
    for weight_mag in [0.001, 0.01, 0.1] {
        let (_mag_lo, mag_hi) =
            synthetic_magnitude_bounds(KOKORO_N_BINS, n_frames, weight_mag, 1.0);
        let istft_input = bridge_generator_to_istft(&mag_hi, KOKORO_N_BINS, n_frames);
        let output = graph.propagate_ibp(&istft_input).expect("IBP");
        let (lo, hi) = super::common::bounds_min_max(&output);
        let width = hi - lo;
        eprintln!("weight_mag={weight_mag}: audio ∈ [{lo:.6}, {hi:.6}], width={width:.6}");
        assert!(
            width >= prev_width,
            "wider input should produce wider output: {width} < {prev_width}"
        );
        prev_width = width;
    }
}

/// CROWN works at Kokoro production dimensions (n_frames=200).
#[test]
fn test_istft_production_dimensions() {
    let n_frames = 200;
    let output_length = (n_frames - 1) * KOKORO_HOP; // 995

    let graph = build_istft_graph_with_clamp(KOKORO_N_FFT, KOKORO_HOP, n_frames, output_length);
    assert_eq!(graph.num_nodes(), 2);

    let (_mag_lo, mag_hi) = synthetic_magnitude_bounds(KOKORO_N_BINS, n_frames, 0.001, 1.0);
    let istft_input = bridge_generator_to_istft(&mag_hi, KOKORO_N_BINS, n_frames);

    let (method, crown_output, fallback_reason) =
        nn_verify::propagate_with_crown_fallback(&graph, &istft_input)
            .expect("CROWN at production scale");

    assert!(
        matches!(method, nn_verify::PropMethod::Crown),
        "CROWN must work at production dimensions (n_frames={n_frames}). Reason: {}",
        fallback_reason.as_deref().unwrap_or("unknown")
    );

    super::common::assert_bounds_valid(&crown_output);
    let (audio_lo, audio_hi) = super::common::bounds_min_max(&crown_output);
    eprintln!(
        "Production: n_frames={n_frames}, output_length={output_length}, \
         audio ∈ [{audio_lo:.6}, {audio_hi:.6}]"
    );

    // P2 via clamp at production scale.
    assert!(
        audio_lo >= -1.0 && audio_hi <= 1.0,
        "P2 at production scale: [{audio_lo}, {audio_hi}] not in [-1, 1]"
    );
}
