// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Audio/DSP verification graph builders — consumes `ny_api::dsp`.
//!
//! NN is an audio stack (Kokoro vocoder, Demucs separation, Silero VAD, TTS
//! post-processing), and ny's `dsp` facade expresses the common DSP kernels as
//! [`ny_propagate::GraphNetwork`]s built purely from *existing* ny layer types
//! (`MulConstant`, `Tanh`, `Linear`, `Clip`, `Reciprocal`, `Conv1d`, …), so
//! sound bound propagation (IBP / CROWN / α-CROWN) works on them with no new
//! layer support. Until now NN only touched these builders from tests (the
//! Kokoro forward-STFT graphs); this adapter hands them to production NN code.
//!
//! Wrapped builders (the complete `ny_api::dsp` surface):
//!
//! - [`build_waveshaper_graph`] — tanh soft-clip `f(x) = tanh(d·x)/tanh(d)`.
//! - [`build_biquad_single_step_graph`] — one Direct Form II Transposed biquad
//!   IIR timestep as a single 3×3 linear map (IBP/CROWN bounds are **exact**),
//!   with [`BiquadCoefficients`].
//! - [`build_one_pole_step_graph`] — one envelope-smoother timestep
//!   `env' = c·rms + (1−c)·env` (linear, exact bounds).
//! - [`build_gain_reduction_graph`] — the compressor/ducking gain curve
//!   `gain(e) ∈ [max_atten, 1]`, with [`GainReductionParams`].
//! - [`build_envelope_follower_step_graph`] /
//!   [`verify_envelope_follower_conservative`] — combined smoother + gain
//!   reduction, and the sound attack/release-union bound, with
//!   [`EnvelopeFollowerParams`].
//! - [`build_kokoro_forward_stft_magnitude_graph`] /
//!   [`build_kokoro_forward_stft_phase_graph`] /
//!   [`build_kokoro_forward_stft_full_graph`] — the Kokoro forward-STFT
//!   frontend (`KOKORO_STFT_*` constants re-exported).
//! - [`build_qwen3_mel128_graph`] — the Qwen3 speaker-encoder mel-128
//!   spectrogram frontend.
//!
//! On top of the builders it provides the NN-facing verify convenience
//! ([`verify_dsp_graph`], [`verify_dsp_graph_with_config`],
//! [`default_dsp_verify_config`]): build a DSP graph, then bound-propagate a
//! [`ny_core::VerificationSpec`] through it and get a
//! [`ny_core::VerificationResult`] verdict — plus a flat-string reporting
//! helper ([`dsp_verification_summary`]).
//!
//! This adapter is additive — it does not touch NN's existing verify path. It
//! only consumes the `ny-api` facade; callers wanting the rich verdict read
//! [`ny_core::VerificationResult`] directly.
//!
//! ```rust,no_run
//! use nn_verify::ny_dsp::{
//!     build_waveshaper_graph, dsp_verification_summary, verify_dsp_graph,
//! };
//! # fn run() -> ny_core::Result<()> {
//! // Soft-clip stage at drive 2: |tanh(2x)/tanh(2)| ≤ 1 on x ∈ [-1, 1].
//! let graph = build_waveshaper_graph(2.0)?;
//! let spec = ny_core::VerificationSpec::new(
//!     vec![ny_core::Bound::new(-1.0, 1.0)],
//!     vec![ny_core::Bound::new(-1.001, 1.001)],
//! )?;
//! let result = verify_dsp_graph(&graph, &spec)?;
//! println!("{}", dsp_verification_summary(&result));
//! # Ok(())
//! # }
//! ```

use ny_api::verify::{PropagationConfig, PropagationMethod, Verifier};
use ny_core::{Result, VerificationResult, VerificationSpec};
use ny_propagate::GraphNetwork;

/// DSP parameter types consumed by the builders: biquad IIR coefficients
/// (Audio EQ Cookbook convention, `a0 = 1` pre-normalized), envelope-follower
/// smoothing + gain parameters, and the compressor gain-curve parameters.
pub use ny_api::dsp::{BiquadCoefficients, EnvelopeFollowerParams, GainReductionParams};

/// Fixed Kokoro forward-STFT configuration constants (FFT size, hop, reflect
/// pad, frequency-bin count, magnitude epsilon) used by the Kokoro builders.
pub use ny_api::dsp::{
    KOKORO_STFT_FREQ_BINS, KOKORO_STFT_HOP, KOKORO_STFT_MAG_EPS, KOKORO_STFT_N_FFT, KOKORO_STFT_PAD,
};

// ---------------------------------------------------------------------------
// Thin builders
// ---------------------------------------------------------------------------

/// Build a waveshaper (tanh soft-clip) verification graph for a fixed `drive`.
///
/// Implements `f(x) = tanh(drive·x) / tanh(drive)` as a 3-node pointwise chain
/// (`MulConstant → Tanh → MulConstant`), so the graph accepts any input shape.
/// For `drive < 1e-6` the function approaches identity and a single
/// `MulConstant(1.0)` node is returned. This is the soft-clip stage NN's audio
/// post chains use; `|f(x)| ≤ 1` for `|x| ≤ 1`, which [`verify_dsp_graph`] can
/// prove as an output-bound spec.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while assembling the graph
/// (defensive; the fixed three-node topology does not fail for finite drive).
pub fn build_waveshaper_graph(drive: f32) -> Result<GraphNetwork> {
    ny_api::dsp::build_waveshaper_graph(drive)
}

/// Build a single-timestep biquad IIR filter graph (Direct Form II Transposed).
///
/// The step is a purely linear 3×3 state-space map over `[x, z1_prev, z2_prev]`
/// producing `[y, z1_new, z2_new]`, so IBP and CROWN both give **exact** bounds
/// (zero relaxation gap). Coefficients follow the Audio EQ Cookbook convention
/// with `a0 = 1` (pre-normalized) — the form NN's EQ/filter code already uses.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] from constructing the underlying
/// `LinearLayer` (e.g. non-finite coefficients rejected by ny's layer checks).
pub fn build_biquad_single_step_graph(coeff: &BiquadCoefficients) -> Result<GraphNetwork> {
    ny_api::dsp::build_biquad_single_step_graph(coeff)
}

/// Build a single-timestep one-pole envelope smoother graph.
///
/// Implements `envelope' = c·rms + (1−c)·envelope_prev` as one 1×2 linear node
/// over `[rms, envelope_prev]`. For `0 < c < 1` this is a convex combination:
/// inputs in `[0, 1]` provably stay in `[0, 1]`, and the pole at `1−c` is
/// inside the unit circle (BIBO stable). Being linear, IBP/CROWN bounds are
/// exact. Use [`one_pole_coefficient`] to derive `c` from a sample rate and
/// time constant.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] from constructing the underlying
/// `LinearLayer`.
pub fn build_one_pole_step_graph(coefficient: f32) -> Result<GraphNetwork> {
    ny_api::dsp::build_one_pole_step_graph(coefficient)
}

/// Build the stateless compressor/ducking gain-reduction curve as a graph.
///
/// Maps an envelope value `e ∈ [0, 1]` to a gain multiplier via ny's
/// `Clip → Reciprocal → MulConstant → AddConstant → Clip → …` chain; the final
/// clamp guarantees `gain ∈ [params.max_atten, 1.0]`, which bound propagation
/// proves structurally. This is the gain curve NN's VAD-driven ducking uses.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while assembling the chain.
pub fn build_gain_reduction_graph(params: &GainReductionParams) -> Result<GraphNetwork> {
    ny_api::dsp::build_gain_reduction_graph(params)
}

/// Build a combined envelope-follower step graph: one-pole smoother followed by
/// the gain-reduction chain, over inputs `[rms, envelope_prev]` producing
/// `[gain]` with `gain ∈ [gain_params.max_atten, 1.0]`.
///
/// For a sound bound on the real attack/release-switching system, run this
/// once with the attack coefficient and once with the release coefficient and
/// union the bounds — or call [`verify_envelope_follower_conservative`], which
/// does exactly that.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while assembling the graph.
pub fn build_envelope_follower_step_graph(
    coefficient: f32,
    gain_params: &GainReductionParams,
) -> Result<GraphNetwork> {
    ny_api::dsp::build_envelope_follower_step_graph(coefficient, gain_params)
}

/// Compute sound conservative gain bounds for the attack/release-switching
/// envelope follower.
///
/// Runs the combined step graph with the attack coefficient (fast tracking,
/// envelope upper bound) and the release coefficient (slow tracking, envelope
/// lower bound) and unions the resulting gain bounds, which soundly bounds the
/// real system regardless of which coefficient it selects at each timestep.
/// `input` bounds the 2-element `[rms, envelope_prev]` state.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] from graph assembly or IBP propagation.
pub fn verify_envelope_follower_conservative(
    params: &EnvelopeFollowerParams,
    input: &ny_api::BoundedTensor,
) -> Result<ny_api::BoundedTensor> {
    ny_api::dsp::verify_envelope_follower_conservative(params, input)
}

/// Build the magnitude half of the Kokoro forward-STFT frontend.
///
/// Input shape `[1, audio_len]`, output shape
/// `[KOKORO_STFT_FREQ_BINS, n_frames]`. The DFT is expressed as fixed `Conv1d`
/// banks over a reflect-padded signal, magnitude as
/// `sqrt(real² + imag² + KOKORO_STFT_MAG_EPS)`.
///
/// # Errors
///
/// Returns [`ny_core::NyError::InvalidSpec`] when
/// `audio_len <= KOKORO_STFT_PAD` (reflect padding would be ill-formed) or the
/// padded length overflows; propagates any layer-construction error.
pub fn build_kokoro_forward_stft_magnitude_graph(audio_len: usize) -> Result<GraphNetwork> {
    ny_api::dsp::build_kokoro_forward_stft_magnitude_graph(audio_len)
}

/// Build the phase half of the Kokoro forward-STFT frontend
/// (`atan2(imag, real)`), input `[1, audio_len]`, output
/// `[KOKORO_STFT_FREQ_BINS, n_frames]`.
///
/// # Errors
///
/// Same conditions as [`build_kokoro_forward_stft_magnitude_graph`].
pub fn build_kokoro_forward_stft_phase_graph(audio_len: usize) -> Result<GraphNetwork> {
    ny_api::dsp::build_kokoro_forward_stft_phase_graph(audio_len)
}

/// Build the full Kokoro forward-STFT surface: magnitude and phase concatenated
/// on the frequency axis, input `[1, audio_len]`, output
/// `[2 · KOKORO_STFT_FREQ_BINS, n_frames]`.
///
/// # Errors
///
/// Same conditions as [`build_kokoro_forward_stft_magnitude_graph`].
pub fn build_kokoro_forward_stft_full_graph(audio_len: usize) -> Result<GraphNetwork> {
    ny_api::dsp::build_kokoro_forward_stft_full_graph(audio_len)
}

/// Build the exact Qwen3-TTS speaker-encoder mel-128 log-spectrogram frontend.
///
/// Input shape `[1, audio_len]` (24 kHz waveform), output shape
/// `[T_mel, 128]` (time-major log-mel), built from reflect pad, DFT `Conv1d`
/// banks, a Slaney mel filterbank `Conv1d`, log with floor, and a transpose.
///
/// # Errors
///
/// Returns [`ny_core::NyError::InvalidSpec`] when the audio is too short for
/// the reflect pad or a length computation overflows; propagates any
/// layer-construction error.
pub fn build_qwen3_mel128_graph(audio_len: usize) -> Result<GraphNetwork> {
    ny_api::dsp::build_qwen3_mel128_graph(audio_len)
}

// ---------------------------------------------------------------------------
// NN-facing parameter helpers
// ---------------------------------------------------------------------------

/// Derive a one-pole smoothing coefficient `c = 1 − exp(−1 / (sr·τ))` from a
/// sample rate (Hz) and time constant (seconds).
///
/// This is the standard envelope-follower coefficient mapping: larger `c`
/// tracks faster. Callers pass positive, finite values; the result is clamped
/// to `[0, 1]` so degenerate inputs cannot produce an unstable coefficient.
#[must_use]
pub fn one_pole_coefficient(sample_rate_hz: f32, time_constant_s: f32) -> f32 {
    (1.0 - (-1.0 / (sample_rate_hz * time_constant_s)).exp()).clamp(0.0, 1.0)
}

/// Default gain-reduction parameters for NN's VAD-driven ducking chain:
/// −30 dBFS threshold (`≈ 0.0316` linear), ratio `0.8`, and a −40 dB
/// attenuation floor (`0.01`).
///
/// Exposed as a named helper so NN call sites document their intent; build a
/// [`GainReductionParams`] directly for other compressor configurations.
#[must_use]
pub fn default_gain_reduction_params() -> GainReductionParams {
    GainReductionParams {
        threshold: 10.0_f32.powf(-30.0 / 20.0),
        ratio: 0.8,
        max_atten: 0.01,
    }
}

/// Default envelope-follower parameters at NN's 24 kHz audio rate (the Kokoro
/// vocoder output rate): 5 ms attack, 100 ms release, and
/// [`default_gain_reduction_params`] for the gain curve.
#[must_use]
pub fn default_envelope_follower_params() -> EnvelopeFollowerParams {
    EnvelopeFollowerParams {
        c_attack: one_pole_coefficient(24_000.0, 0.005),
        c_release: one_pole_coefficient(24_000.0, 0.1),
        gain: default_gain_reduction_params(),
    }
}

// ---------------------------------------------------------------------------
// Verify convenience: build, then bound-propagate a spec
// ---------------------------------------------------------------------------

/// Default propagation configuration for DSP graph verification: plain IBP.
///
/// The `ny_api::dsp` builders are designed for IBP soundness — the linear
/// steps (biquad, one-pole) are IBP-**exact**, and the waveshaper's single
/// nonlinearity gains nothing from CROWN — so IBP is the cheap, deterministic
/// default. Escalate through [`verify_dsp_graph_with_config`] (e.g.
/// `PropagationMethod::Crown` for the multi-nonlinearity gain-reduction chain)
/// when IBP leaves a spec unproven.
#[must_use]
pub fn default_dsp_verify_config() -> PropagationConfig {
    PropagationConfig {
        method: PropagationMethod::Ibp,
        ..PropagationConfig::default()
    }
}

/// Verify a DSP builder graph against a spec: bound-propagate the spec's input
/// box through `graph` and check the outputs against the spec's output bounds,
/// using [`default_dsp_verify_config`] (IBP).
///
/// Returns the standard [`ny_core::VerificationResult`] verdict (`Verified` /
/// `Unknown { BoundsTooLoose, .. }` / …). For a tighter method or a timeout,
/// use [`verify_dsp_graph_with_config`].
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] from propagation, and rejects
/// structurally invalid specs (empty output bounds, input-shape/bounds
/// mismatches) with [`ny_core::NyError::InvalidSpec`].
pub fn verify_dsp_graph(
    graph: &GraphNetwork,
    spec: &VerificationSpec,
) -> Result<VerificationResult> {
    verify_dsp_graph_with_config(graph, spec, &default_dsp_verify_config())
}

/// Verify a DSP builder graph against a spec with an explicit
/// [`PropagationConfig`] (method, iterations, tolerance).
///
/// See [`verify_dsp_graph`] for the default-config convenience wrapper and the
/// verdict semantics.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] from propagation or spec validation.
pub fn verify_dsp_graph_with_config(
    graph: &GraphNetwork,
    spec: &VerificationSpec,
    cfg: &PropagationConfig,
) -> Result<VerificationResult> {
    Verifier::new(cfg.clone()).verify_graph(graph, spec)
}

/// Render an NN-friendly one-line summary of a DSP verification verdict.
///
/// Reports the verdict, the propagation method that produced it, and a short
/// verdict-specific detail (bound count, counterexample dimension, or the
/// unknown-reason). This is intentionally a flat string for NN's reporting
/// style; callers needing structured data read the
/// [`ny_core::VerificationResult`] fields directly.
#[must_use]
pub fn dsp_verification_summary(result: &VerificationResult) -> String {
    let method = result.actual_method().unwrap_or("unreported");
    match result {
        VerificationResult::Verified { output_bounds, .. } => format!(
            "dsp: verified via {method} ({} output bound{})",
            output_bounds.len(),
            if output_bounds.len() == 1 { "" } else { "s" },
        ),
        VerificationResult::Violated { counterexample, .. } => format!(
            "dsp: violated via {method} ({}-d counterexample)",
            counterexample.len(),
        ),
        VerificationResult::Unknown { reason, .. } => {
            format!("dsp: unproven via {method} ({reason})")
        }
        VerificationResult::Timeout { .. } => format!("dsp: timeout via {method}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, ArrayD, IxDyn};
    use ny_api::BoundedTensor;
    use ny_core::{Bound, VerificationSpec};

    /// Stable biquad test coefficients: poles of `z² + a1·z + a2` satisfy the
    /// stability triangle (`|a2| < 1`, `|a1| < 1 + a2`).
    fn stable_biquad_coefficients() -> BiquadCoefficients {
        BiquadCoefficients {
            b0: 0.5,
            b1: 0.2,
            b2: 0.1,
            a1: -0.3,
            a2: 0.2,
        }
    }

    /// True Direct Form II Transposed single-step arithmetic, mirroring the
    /// state-space matrix the builder encodes.
    fn biquad_step_true(x: f32, z1: f32, z2: f32, c: &BiquadCoefficients) -> (f32, f32, f32) {
        let y = c.b0 * x + z1;
        let z1_new = (c.b1 - c.a1 * c.b0) * x - c.a1 * z1 + z2;
        let z2_new = (c.b2 - c.a2 * c.b0) * x - c.a2 * z1;
        (y, z1_new, z2_new)
    }

    #[test]
    fn ny_dsp_waveshaper_verifies_unit_amplitude_spec() {
        // |tanh(2x)/tanh(2)| ≤ 1 on x ∈ [-1, 1]; tanh is monotone, so IBP's
        // endpoint evaluation proves the unit-amplitude property outright.
        let graph = build_waveshaper_graph(2.0).expect("waveshaper graph");
        let spec =
            VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![Bound::new(-1.001, 1.001)])
                .expect("valid spec");

        let result = verify_dsp_graph(&graph, &spec).expect("verify should run");
        assert!(
            result.is_verified(),
            "unit-amplitude soft-clip spec must verify, got {result:?}"
        );

        let summary = dsp_verification_summary(&result);
        assert!(summary.contains("verified"), "summary: {summary}");
        assert!(summary.contains("Ibp"), "summary: {summary}");
    }

    #[test]
    fn ny_dsp_waveshaper_rejects_overtight_spec() {
        // f(±1) = ±1, so the output cannot be confined to [-0.5, 0.5]; the
        // sound verifier must NOT claim Verified.
        let graph = build_waveshaper_graph(2.0).expect("waveshaper graph");
        let spec = VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![Bound::new(-0.5, 0.5)])
            .expect("valid spec");

        let result = verify_dsp_graph(&graph, &spec).expect("verify should run");
        assert!(
            !result.is_verified(),
            "over-tight spec must not verify, got {result:?}"
        );
        assert!(
            matches!(result, VerificationResult::Unknown { .. }),
            "IBP leaves the over-tight box unproven (not violated), got {result:?}"
        );

        let summary = dsp_verification_summary(&result);
        assert!(summary.contains("unproven"), "summary: {summary}");
    }

    #[test]
    fn ny_dsp_biquad_step_verifies_reachable_state_space_box() {
        // The step is purely linear, so IBP is exact: for x ∈ [-1,1] and
        // z1, z2 ∈ [-1,1] the reachable box is
        //   y      ∈ ±(|b0| + 1)              = ±1.5
        //   z1_new ∈ ±(|b1−a1·b0| + |a1| + 1) = ±1.65
        //   z2_new ∈ ±(|b2−a2·b0| + |a2|)     = ±0.2
        let coeff = stable_biquad_coefficients();
        let graph = build_biquad_single_step_graph(&coeff).expect("biquad graph");

        let input_box = vec![Bound::new(-1.0, 1.0); 3];
        let spec = VerificationSpec::new(
            input_box.clone(),
            vec![
                Bound::new(-1.51, 1.51),
                Bound::new(-1.66, 1.66),
                Bound::new(-0.21, 0.21),
            ],
        )
        .expect("valid spec");
        let result = verify_dsp_graph(&graph, &spec).expect("verify should run");
        assert!(
            result.is_verified(),
            "exact linear box must verify, got {result:?}"
        );

        // Just inside the exact y-bound (1.5): must stay unproven — the exact
        // IBP interval is a witness that the tighter band is not implied.
        let tight_spec = VerificationSpec::new(
            input_box,
            vec![
                Bound::new(-1.4, 1.4),
                Bound::new(-1.66, 1.66),
                Bound::new(-0.21, 0.21),
            ],
        )
        .expect("valid spec");
        let tight = verify_dsp_graph(&graph, &tight_spec).expect("verify should run");
        assert!(
            !tight.is_verified(),
            "band tighter than the exact bound must not verify, got {tight:?}"
        );
    }

    #[test]
    fn ny_dsp_biquad_step_matches_direct_form_ii_transposed_arithmetic() {
        // Degenerate (point) input: IBP through the linear step must reproduce
        // the concrete Direct Form II Transposed arithmetic.
        let coeff = stable_biquad_coefficients();
        let graph = build_biquad_single_step_graph(&coeff).expect("biquad graph");

        let (x, z1, z2) = (0.5_f32, -0.25_f32, 0.125_f32);
        let point = arr1(&[x, z1, z2]).into_dyn();
        let input = BoundedTensor::new(point.clone(), point).expect("point tensor");

        let bounds = graph.propagate_ibp(&input).expect("ibp propagation");
        let (y, z1_new, z2_new) = biquad_step_true(x, z1, z2, &coeff);
        for (i, expected) in [y, z1_new, z2_new].into_iter().enumerate() {
            assert!(
                (bounds.lower()[[i]] - expected).abs() < 1e-5,
                "lower[{i}] = {} should equal concrete step output {expected}",
                bounds.lower()[[i]],
            );
            assert!(
                (bounds.upper()[[i]] - expected).abs() < 1e-5,
                "upper[{i}] = {} should equal concrete step output {expected}",
                bounds.upper()[[i]],
            );
        }
    }

    #[test]
    fn ny_dsp_one_pole_step_stays_in_unit_interval() {
        // Convex combination: rms, envelope_prev ∈ [0,1] ⇒ envelope' ∈ [0,1].
        let c = one_pole_coefficient(24_000.0, 0.005);
        assert!(c > 0.0 && c < 1.0, "coefficient must be in (0,1), got {c}");

        let graph = build_one_pole_step_graph(c).expect("one-pole graph");
        let spec = VerificationSpec::new(
            vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)],
            vec![Bound::new(-0.001, 1.001)],
        )
        .expect("valid spec");

        let ibp = verify_dsp_graph(&graph, &spec).expect("ibp verify");
        assert!(
            ibp.is_verified(),
            "IBP must prove [0,1]-invariance: {ibp:?}"
        );

        // The linear step is exact under CROWN too — exercise the explicit
        // config path with an escalated method.
        let crown_cfg = PropagationConfig {
            method: PropagationMethod::Crown,
            ..PropagationConfig::default()
        };
        let crown = verify_dsp_graph_with_config(&graph, &spec, &crown_cfg).expect("crown verify");
        assert!(
            crown.is_verified(),
            "CROWN must also prove the exact linear box: {crown:?}"
        );
    }

    #[test]
    fn ny_dsp_envelope_follower_conservative_gain_band() {
        let params = default_envelope_follower_params();
        // Attack (5 ms) must track faster than release (100 ms).
        assert!(
            0.0 < params.c_release && params.c_release < params.c_attack && params.c_attack < 1.0,
            "0 < c_release ({}) < c_attack ({}) < 1",
            params.c_release,
            params.c_attack,
        );

        // rms, envelope_prev ∈ [0, 1] ⇒ conservative gain ∈ [max_atten, 1].
        let input = BoundedTensor::new(
            arr1(&[0.0_f32, 0.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .expect("input tensor");
        let bounds = verify_envelope_follower_conservative(&params, &input)
            .expect("conservative envelope bounds");

        assert!(
            bounds.lower()[[0]] >= params.gain.max_atten - 1e-5,
            "gain lower {} must respect the attenuation floor {}",
            bounds.lower()[[0]],
            params.gain.max_atten,
        );
        assert!(
            bounds.upper()[[0]] <= 1.0 + 1e-5,
            "gain upper {} must not exceed unity",
            bounds.upper()[[0]],
        );
    }

    #[test]
    fn ny_dsp_kokoro_stft_graphs_have_expected_shapes_and_nonnegative_magnitude() {
        // audio_len = 25 > pad = 10; padded 45 ⇒ (45 − 20)/5 + 1 = 6 frames.
        let audio_len = 25;
        let frames = 6;
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, audio_len]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1, audio_len]), 1.0_f32),
        )
        .expect("waveform tensor");

        let magnitude = build_kokoro_forward_stft_magnitude_graph(audio_len)
            .expect("magnitude graph")
            .propagate_ibp(&input)
            .expect("magnitude ibp");
        assert_eq!(magnitude.lower().shape(), [KOKORO_STFT_FREQ_BINS, frames]);
        assert!(
            magnitude.lower().iter().all(|&v| v >= 0.0),
            "sqrt(power + eps) magnitude lower bounds must be non-negative"
        );

        let full = build_kokoro_forward_stft_full_graph(audio_len)
            .expect("full graph")
            .propagate_ibp(&input)
            .expect("full ibp");
        assert_eq!(full.lower().shape(), [2 * KOKORO_STFT_FREQ_BINS, frames]);

        let phase = build_kokoro_forward_stft_phase_graph(audio_len)
            .expect("phase graph")
            .propagate_ibp(&input)
            .expect("phase ibp");
        assert_eq!(phase.lower().shape(), [KOKORO_STFT_FREQ_BINS, frames]);
    }

    #[test]
    fn ny_dsp_kokoro_stft_rejects_audio_shorter_than_reflect_pad() {
        assert!(
            build_kokoro_forward_stft_magnitude_graph(KOKORO_STFT_PAD).is_err(),
            "audio_len <= pad must be rejected"
        );
    }

    #[test]
    fn ny_dsp_qwen3_mel128_bounds_time_major_output() {
        // audio_len = 400 > pad = 384; padded 1168 ⇒ (1168 − 1024)/256 + 1 = 1
        // mel frame; output is time-major [T_mel, 128].
        let audio_len = 400;
        let graph = build_qwen3_mel128_graph(audio_len).expect("mel128 graph");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, audio_len]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1, audio_len]), 1.0_f32),
        )
        .expect("waveform tensor");

        let bounds = graph.propagate_ibp(&input).expect("mel128 ibp");
        assert_eq!(bounds.lower().shape(), [1, 128]);
        assert!(
            bounds
                .lower()
                .iter()
                .chain(bounds.upper().iter())
                .all(|v| v.is_finite()),
            "log-mel bounds must be finite (log floor prevents -inf)"
        );
    }

    #[test]
    fn ny_dsp_default_config_and_coefficient_helpers() {
        let cfg = default_dsp_verify_config();
        assert_eq!(cfg.method, PropagationMethod::Ibp);

        let c = one_pole_coefficient(24_000.0, 0.005);
        let expected = 1.0 - (-1.0_f32 / (24_000.0 * 0.005)).exp();
        assert!(
            (c - expected).abs() < 1e-7,
            "closed form mismatch: {c} vs {expected}"
        );

        let gain = default_gain_reduction_params();
        assert!(
            (gain.threshold - 10.0_f32.powf(-1.5)).abs() < 1e-7,
            "-30 dBFS threshold"
        );
        assert!(gain.max_atten > 0.0 && gain.max_atten < 1.0);
    }
}
