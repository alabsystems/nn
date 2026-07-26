// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! IBP/CROWN through Kokoro forward-STFT — prove magnitude and phase bounds.
//!
//! The forward-STFT graph is built by NY (`ny_propagate::network::dsp`)
//! using Conv1d-based DFT decomposition:
//!   - Magnitude: Pad → Conv1d×2 (real/imag DFT) → Sqr×2 → Add → AddConstant(eps) → Sqrt
//!   - Phase: Pad → Conv1d×2 → Atan2
//!   - Full: both magnitude and phase concatenated
//!
//! Production Kokoro uses butterfly FFT (rustfft), not Conv1d DFT. These are
//! mathematically equivalent but numerically different — the phase wrapping
//! difference is documented in #2928 and is an inherent limitation of verifying
//! FFT through Conv1d decomposition.
//!
//! Part of #2993, Part of #2218.

use ny_propagate::network::dsp::{
    build_kokoro_forward_stft_full_graph, build_kokoro_forward_stft_magnitude_graph,
    build_kokoro_forward_stft_phase_graph,
};
use nn_verify::BoundedTensor;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// D3: STFT constant drift assertion — cross-repo sync
// ---------------------------------------------------------------------------

/// Assert nn's Kokoro STFT constants match NY's hardcoded values.
///
/// NY hardcodes these at:
///   `gamma-propagate/src/network/dsp/kokoro_forward_stft.rs:25-29`
///
/// The constants are `pub(crate)` in NY, so we hard-assert expected
/// values. If these fail, NY's verification graphs verify bounds on
/// a wrong STFT configuration.
#[test]
fn test_kokoro_stft_constants_cross_repo_sync() {
    let config = nn_models::KokoroConfig::default();
    assert_eq!(
        config.n_fft, 20,
        "n_fft must match NY KOKORO_STFT_N_FFT"
    );
    assert_eq!(
        config.n_fft / 4,
        5,
        "hop must match NY KOKORO_STFT_HOP"
    );
    assert_eq!(
        config.n_fft / 2 + 1,
        11,
        "freq_bins must match NY KOKORO_STFT_FREQ_BINS"
    );
    // mag_eps (1e-9) is internal to NY's graph construction,
    // not configurable from nn side. Documented, not enforced.
}

// ---------------------------------------------------------------------------
// D2: Forward STFT compose tests via NY graph import
// ---------------------------------------------------------------------------

/// Helper: compute expected frame count for Kokoro forward STFT.
///
/// `n_frames = (audio_len + 2*pad - n_fft) / hop + 1` where pad = n_fft/2.
fn expected_frame_count(audio_len: usize) -> usize {
    let n_fft = 20;
    let hop = 5;
    let pad = n_fft / 2;
    (audio_len + 2 * pad - n_fft) / hop + 1
}

/// IBP through forward-STFT magnitude graph produces finite, non-negative bounds.
///
/// Magnitude = sqrt(real² + imag² + eps), so lower bound must be >= 0.
#[test]
fn test_kokoro_forward_stft_magnitude_ibp() {
    let audio_len = 40;
    let graph = build_kokoro_forward_stft_magnitude_graph(audio_len).expect("magnitude graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, audio_len]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1, audio_len]), 1.0f32),
    )
    .expect("valid bounds");

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    super::common::assert_bounds_valid(&ibp_output);

    let n_frames = expected_frame_count(audio_len);
    let (lo, _hi) = ibp_output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[11, n_frames],
        "magnitude output shape [freq_bins, n_frames]"
    );

    let (lo_min, hi_max) = super::common::bounds_min_max(&ibp_output);
    eprintln!("Forward-STFT magnitude IBP: [{lo_min:.6}, {hi_max:.6}]");

    // Magnitude must be non-negative (sqrt of sum of squares + eps).
    assert!(
        lo_min >= 0.0,
        "magnitude lower bound must be >= 0, got {lo_min}"
    );
    // Bounds must be non-vacuous (width < 100 for [-1,1] input).
    let width = hi_max - lo_min;
    assert!(
        width < 100.0,
        "magnitude IBP bounds vacuously wide: width={width}"
    );
}

/// IBP through forward-STFT phase graph produces bounds within [-pi, pi].
///
/// Phase = atan2(imag, real), output domain is [-pi, pi].
#[test]
fn test_kokoro_forward_stft_phase_ibp() {
    let audio_len = 40;
    let graph = build_kokoro_forward_stft_phase_graph(audio_len).expect("phase graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, audio_len]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1, audio_len]), 1.0f32),
    )
    .expect("valid bounds");

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    super::common::assert_bounds_valid(&ibp_output);

    let n_frames = expected_frame_count(audio_len);
    let (lo, _hi) = ibp_output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[11, n_frames],
        "phase output shape [freq_bins, n_frames]"
    );

    let (lo_min, hi_max) = super::common::bounds_min_max(&ibp_output);
    eprintln!("Forward-STFT phase IBP: [{lo_min:.6}, {hi_max:.6}]");

    // Phase bounds must be within [-pi, pi] (Atan2 output domain).
    let pi = std::f32::consts::PI;
    assert!(lo_min >= -pi - 1e-5, "phase lower bound {lo_min} below -pi");
    assert!(hi_max <= pi + 1e-5, "phase upper bound {hi_max} above pi");
}

/// IBP through forward-STFT full graph (magnitude + phase concatenated).
///
/// Full graph output shape is [22, n_frames] = [11 magnitude + 11 phase, n_frames].
#[test]
fn test_kokoro_forward_stft_full_ibp() {
    let audio_len = 40;
    let graph = build_kokoro_forward_stft_full_graph(audio_len).expect("full graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, audio_len]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1, audio_len]), 1.0f32),
    )
    .expect("valid bounds");

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    super::common::assert_bounds_valid(&ibp_output);

    let n_frames = expected_frame_count(audio_len);
    let (lo, _hi) = ibp_output.lower_upper();
    // Full output: 11 freq_bins magnitude + 11 freq_bins phase = 22 rows.
    assert_eq!(
        lo.shape(),
        &[22, n_frames],
        "full output shape [2*freq_bins, n_frames]"
    );

    let (lo_min, hi_max) = super::common::bounds_min_max(&ibp_output);
    eprintln!("Forward-STFT full IBP: [{lo_min:.6}, {hi_max:.6}]");

    // Combined bounds: magnitude portion >= 0, phase portion in [-pi, pi].
    // Global min can be negative (phase), global max can exceed pi (magnitude).
    // Just assert finite and non-vacuous.
    assert!(lo_min.is_finite() && hi_max.is_finite());
    let width = hi_max - lo_min;
    assert!(
        width < 200.0,
        "full IBP bounds vacuously wide: width={width}"
    );
}
