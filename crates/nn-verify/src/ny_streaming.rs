// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Memory-efficient streaming CROWN verification adapter — consumes
//! [`ny_api::streaming`].
//!
//! NN's large transformer models (qwen3, glm5, gptoss, whisper) produce layer
//! stacks whose plain CROWN propagation retains bounds at every layer —
//! `O(L·N)` memory — which blows past any realistic budget at transformer
//! depth × activation width. ny's [`StreamingVerifier`] applies gradient
//! checkpointing to the CROWN pass: it stores checkpoints every `K` layers
//! (`O(L/K·N)` memory) and recomputes forward IBP segments on the backward
//! pass (`O(L·K)` compute), trading compute for memory. Optional f16
//! checkpoint compression roughly halves checkpoint bytes again, with sound
//! bound widening on decompression. (Advanced callers can drive the
//! checkpointed forward pass directly via
//! `StreamingVerifier::collect_checkpointed_bounds`, which yields
//! [`ny_api::streaming::CheckpointedBounds`].)
//!
//! The streaming verifier operates on ny's *sequential* network representation
//! ([`ny_propagate::Network`], re-exported by `ny_api::graph` as
//! `SequentialNetwork`), not the [`ny_propagate::GraphNetwork`] used by
//! [`crate::ny_ladder`]. It computes output bounds only; this adapter closes
//! the loop by checking those bounds against a [`ny_core::VerificationSpec`]
//! with the same containment semantics as ny's `Verifier::check_spec`
//! (`Verified` iff every computed output bound lies inside its required bound,
//! `Unknown`/`BoundsTooLoose` otherwise — never `Violated`, since streaming
//! CROWN performs no counterexample search).
//!
//! This adapter is additive — it does not touch NN's existing verify path. It
//! exposes thin NN-facing wrappers ([`verify_model_streaming`],
//! [`verify_model_streaming_with_config`]), a memory-savings estimator
//! ([`estimate_streaming_savings`]) for sizing the compute-for-memory trade
//! before committing to a run, and flat-string reporting helpers
//! ([`streaming_summary`], [`memory_savings_summary`]). Callers wanting the
//! rich verdict read the wrapped [`ny_core::VerificationResult`] directly.
//!
//! ```rust,no_run
//! use nn_verify::ny_streaming::{streaming_summary, verify_model_streaming};
//! # fn run(net: &ny_propagate::Network, spec: &ny_core::VerificationSpec)
//! #     -> ny_core::Result<()> {
//! let outcome = verify_model_streaming(net, spec)?;
//! println!("{}", streaming_summary(&outcome));
//! # Ok(())
//! # }
//! ```

use ny_api::streaming::{estimate_memory_savings, StreamingConfig, StreamingVerifier};
use ny_api::verify::{PropagationMethod, Verifier};
use ny_api::BoundedTensor;
use ny_core::{Bound, MethodUsed, NyError, SoundnessProvenance, UnknownReason, VerificationResult};
use ny_propagate::soundness_provenance_for_network;

/// Default streaming configuration for NN model verification.
///
/// Delegates to [`StreamingConfig::default`]: a checkpoint every 10 layers
/// (~90% activation-memory reduction at ~10× recompute on deep stacks) with
/// full-precision f32 checkpoint storage. For tighter memory budgets use
/// [`StreamingConfig::min_memory`] or [`StreamingConfig::with_f16_compression`]
/// with [`verify_model_streaming_with_config`]. Exposed as a named helper so NN
/// call sites document their intent rather than reaching for `Default` inline.
#[must_use]
pub fn default_streaming_config() -> StreamingConfig {
    StreamingConfig::default()
}

/// Outcome of a streaming (checkpointed CROWN) verification run.
///
/// Pairs the sound ny verdict with the checkpointing shape of the run so
/// reports can state *how* the memory budget was met. The verdict is either
/// `Verified` (every computed output bound inside its required bound) or
/// `Unknown` with [`UnknownReason::BoundsTooLoose`]; streaming CROWN performs
/// no counterexample search, so `Violated` is never produced.
#[derive(Debug, Clone)]
pub struct StreamingVerification {
    /// The sound verification verdict, with computed output bounds and
    /// soundness provenance attached.
    pub result: VerificationResult,
    /// Number of layers in the verified network.
    pub num_layers: usize,
    /// Effective checkpoint interval used (the configured interval clamped to
    /// at least 1, matching the verifier's own clamping).
    pub checkpoint_interval: usize,
    /// Whether checkpoints were stored compressed as f16 (with sound widening
    /// on decompression).
    pub f16_checkpoints: bool,
}

impl StreamingVerification {
    /// Whether the run proved the property (verdict is `Verified`).
    #[must_use]
    pub fn is_verified(&self) -> bool {
        self.result.is_verified()
    }
}

/// Estimated memory/compute trade for verifying a network with streaming CROWN.
///
/// Byte figures come from ny's [`estimate_memory_savings`] and assume f32
/// bound storage (`8` bytes per element for lower + upper) at a uniform
/// per-layer activation size; enabling f16 checkpoints roughly halves the
/// checkpoint portion of `streaming_bytes` again.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingMemoryEstimate {
    /// Number of layers in the network the estimate covers.
    pub num_layers: usize,
    /// Assumed per-layer activation element count (e.g. `seq_len × hidden`).
    pub tensor_elements: usize,
    /// Effective checkpoint interval `K` (clamped to at least 1).
    pub checkpoint_interval: usize,
    /// Whether the supplied configuration stores checkpoints as f16.
    pub f16_checkpoints: bool,
    /// Bytes plain CROWN would retain (bounds at every layer, `O(L·N)`).
    pub baseline_bytes: usize,
    /// Bytes streaming CROWN retains (checkpoints + input, `O(L/K·N)`).
    pub streaming_bytes: usize,
    /// Percentage of `baseline_bytes` saved by streaming.
    pub savings_percent: f32,
    /// Approximate compute multiplier paid for the savings: the backward pass
    /// recomputes forward segments of up to `K` layers, `O(L·K)` vs `O(L)`.
    pub recompute_factor: usize,
}

impl StreamingMemoryEstimate {
    /// Absolute bytes saved versus plain CROWN.
    #[must_use]
    pub fn bytes_saved(&self) -> usize {
        self.baseline_bytes.saturating_sub(self.streaming_bytes)
    }
}

/// Verify a spec on a sequential network via streaming (checkpointed) CROWN
/// using a sensible default configuration.
///
/// Builds the input tensor from the spec's input bounds (honoring
/// `input_shape`), runs ny's [`StreamingVerifier`] CROWN pass under the
/// default checkpointing budget, and checks the computed output bounds against
/// `spec.output_bounds()`. For control over the checkpoint interval or f16
/// checkpoint compression, use [`verify_model_streaming_with_config`].
///
/// Two spec fields are not enforced by the streaming path: `timeout_ms` (ny's
/// streaming verifier exposes no deadline hook — budget runs via the
/// checkpoint interval instead) and the additional P7 `output_constraints`
/// (only the box `output_bounds` are checked). Callers needing either use the
/// full [`ny_api::verify::Verifier`] or [`crate::ny_ladder`] paths.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while building the input tensor
/// (e.g. an `input_shape`/bounds mismatch) or propagating a layer, and returns
/// [`NyError::NumericalInstability`] if the propagated output bounds are
/// corrupt (NaN or inverted).
pub fn verify_model_streaming(
    net: &ny_propagate::Network,
    spec: &ny_core::VerificationSpec,
) -> ny_core::Result<StreamingVerification> {
    verify_model_streaming_with_config(net, spec, &default_streaming_config())
}

/// Verify a spec on a sequential network via streaming (checkpointed) CROWN
/// with an explicit [`StreamingConfig`].
///
/// See [`verify_model_streaming`] for the default-config convenience wrapper
/// and the verdict semantics, and [`estimate_streaming_savings`] to size the
/// memory/compute trade of a candidate `cfg` before running it.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while building the input tensor
/// or propagating a layer, and returns [`NyError::NumericalInstability`] if
/// the propagated output bounds are corrupt (NaN or inverted).
pub fn verify_model_streaming_with_config(
    net: &ny_propagate::Network,
    spec: &ny_core::VerificationSpec,
    cfg: &StreamingConfig,
) -> ny_core::Result<StreamingVerification> {
    let input = Verifier::bounds_to_tensor(spec.input_bounds(), spec.input_shape())?;
    let output = StreamingVerifier::new(cfg.clone()).propagate_crown_streaming(net, &input)?;
    // Streaming runs an IBP forward pass for checkpoints and a CROWN backward
    // pass; record provenance for the method that produced the final bounds.
    let provenance = soundness_provenance_for_network(net, &PropagationMethod::Crown);
    let result = check_output_against_spec(&output, spec.output_bounds(), provenance)?;
    Ok(StreamingVerification {
        result,
        num_layers: net.num_layers(),
        checkpoint_interval: cfg.checkpoint_interval.max(1),
        f16_checkpoints: cfg.use_f16_checkpoints,
    })
}

/// Estimate the compute-for-memory trade of verifying `net` with streaming
/// CROWN under `cfg`.
///
/// `tensor_elements` is the assumed per-layer activation element count (for a
/// transformer, typically `seq_len × hidden`); the layer count comes from the
/// network itself. Byte figures assume f32 bound storage — see
/// [`StreamingMemoryEstimate`] for the f16 caveat. Use this to pick a
/// checkpoint interval that fits the memory budget *before* paying for a run.
#[must_use]
pub fn estimate_streaming_savings(
    net: &ny_propagate::Network,
    tensor_elements: usize,
    cfg: &StreamingConfig,
) -> StreamingMemoryEstimate {
    let interval = cfg.checkpoint_interval.max(1);
    let (baseline_bytes, streaming_bytes, savings_percent) =
        estimate_memory_savings(net.num_layers(), tensor_elements, interval);
    StreamingMemoryEstimate {
        num_layers: net.num_layers(),
        tensor_elements,
        checkpoint_interval: interval,
        f16_checkpoints: cfg.use_f16_checkpoints,
        baseline_bytes,
        streaming_bytes,
        savings_percent,
        recompute_factor: interval,
    }
}

/// Render an NN-friendly one-line summary of a streaming verification run.
///
/// Reports the verdict, the layer count, and the checkpointing shape
/// (interval + storage precision). This is intentionally a flat string for
/// NN's reporting style; callers needing structured data read the
/// [`StreamingVerification`] fields directly.
#[must_use]
pub fn streaming_summary(outcome: &StreamingVerification) -> String {
    let verdict = if outcome.is_verified() {
        "verified"
    } else {
        "unproven"
    };
    let storage = if outcome.f16_checkpoints {
        "f16"
    } else {
        "f32"
    };
    format!(
        "streaming: {verdict} via checkpointed CROWN ({} layer{}, checkpoint interval {}, {storage} checkpoints)",
        outcome.num_layers,
        if outcome.num_layers == 1 { "" } else { "s" },
        outcome.checkpoint_interval,
    )
}

/// Render an NN-friendly one-line summary of a streaming memory estimate.
///
/// States both sides of the trade: bytes retained (versus plain CROWN) and the
/// approximate recompute multiplier paid for the savings.
#[must_use]
pub fn memory_savings_summary(estimate: &StreamingMemoryEstimate) -> String {
    let storage = if estimate.f16_checkpoints {
        "f16"
    } else {
        "f32"
    };
    format!(
        "streaming memory: {} -> {} bytes ({:.1}% saved) at ~{}x recompute ({} layers x {} elements, interval {}, {storage} checkpoints)",
        estimate.baseline_bytes,
        estimate.streaming_bytes,
        estimate.savings_percent,
        estimate.recompute_factor,
        estimate.num_layers,
        estimate.tensor_elements,
        estimate.checkpoint_interval,
    )
}

/// Check computed streaming-CROWN output bounds against the spec's required
/// bounds, mirroring ny's `Verifier::check_spec` containment semantics.
///
/// `Verified` iff every computed bound lies inside its required bound;
/// otherwise `Unknown` with [`UnknownReason::BoundsTooLoose`] carrying the
/// worst gap of the first violating output.
fn check_output_against_spec(
    output: &BoundedTensor,
    required: &[Bound],
    provenance: SoundnessProvenance,
) -> ny_core::Result<VerificationResult> {
    // Guard: an empty output spec would trivially verify any network. Spec
    // constructors already forbid this; kept to mirror ny's own guard.
    if required.is_empty() {
        return Err(NyError::InvalidSpec(
            "empty output_bounds in specification — nothing to verify".to_string(),
        ));
    }

    let mut output_bounds = Vec::with_capacity(output.lower().len());
    for (&l, &u) in output.lower().iter().zip(output.upper().iter()) {
        // concretize_sound() + forward tightening should never yield NaN or an
        // inverted interval; refuse (rather than panic in Bound's assert) if
        // corruption slips through.
        if l.is_nan() || u.is_nan() || l > u {
            return Err(NyError::NumericalInstability(format!(
                "streaming CROWN produced a corrupt output bound [{l}, {u}]"
            )));
        }
        output_bounds.push(Bound::new_allow_infinite(l, u));
    }

    // Guard: zip truncation below would silently skip spec requirements.
    if output_bounds.len() < required.len() {
        return Err(NyError::InvalidSpec(format!(
            "Network produced {} output bounds but spec requires {}",
            output_bounds.len(),
            required.len()
        )));
    }

    for (computed, req) in output_bounds.iter().zip(required.iter()) {
        if computed.lower() < req.lower() || computed.upper() > req.upper() {
            let lower_gap = if computed.lower() < req.lower() {
                req.lower() - computed.lower()
            } else {
                0.0
            };
            let upper_gap = if computed.upper() > req.upper() {
                computed.upper() - req.upper()
            } else {
                0.0
            };
            let gap = lower_gap.max(upper_gap);
            return Ok(VerificationResult::Unknown {
                provenance,
                bounds: output_bounds.clone(),
                reason: UnknownReason::BoundsTooLoose {
                    gap: if gap > 0.0 { Some(gap) } else { None },
                },
                actual_method: Some(MethodUsed::Crown),
            });
        }
    }

    Ok(VerificationResult::Verified {
        provenance,
        output_bounds,
        proof: None,
        actual_method: Some(MethodUsed::Crown),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_core::{Bound, VerificationSpec};
    use ny_propagate::layers::{Layer, LinearLayer, ReLULayer};
    use ny_propagate::Network;

    /// Tiny Linear → ReLU → Linear sequential network (2→2→2).
    ///
    /// Mirrors the graph used by `ny_ladder`'s tests, in ny's sequential
    /// representation. Both outputs compute
    /// `relu(x - y) + relu(y - x) = |x - y|`, so over the input box `[-1, 1]²`
    /// the true output range is exactly `[0, 2]` (IBP sees `[0, 4]`).
    fn linear_relu_linear_network() -> Network {
        let w1 =
            ndarray::Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).expect("2x2 w1");
        let linear1 = LinearLayer::new(w1, Some(ndarray::Array1::zeros(2))).expect("valid linear1");
        let w2 = ndarray::Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).expect("2x2 w2");
        let linear2 = LinearLayer::new(w2, Some(ndarray::Array1::zeros(2))).expect("valid linear2");

        let mut net = Network::new();
        net.add_layer(Layer::Linear(linear1));
        net.add_layer(Layer::ReLU(ReLULayer::new()));
        net.add_layer(Layer::Linear(linear2));
        net
    }

    /// Extract the computed output bounds from a `Verified` verdict.
    fn verified_bounds(outcome: &StreamingVerification) -> &[Bound] {
        match &outcome.result {
            VerificationResult::Verified { output_bounds, .. } => output_bounds,
            other => panic!("expected Verified, got {other:?}"),
        }
    }

    #[test]
    fn ny_streaming_verifies_wide_spec_with_sound_bounds() {
        let net = linear_relu_linear_network();
        // Input in [-1, 1]^2; true output range [0, 2] — a wide output spec is
        // comfortably provable by streaming CROWN.
        let spec = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            vec![Bound::new(-10.0, 10.0), Bound::new(-10.0, 10.0)],
        )
        .expect("valid spec");

        let outcome = verify_model_streaming(&net, &spec)
            .expect("streaming CROWN should run on linear→relu→linear network");

        assert!(outcome.is_verified(), "wide spec should verify");
        assert_eq!(outcome.num_layers, 3);
        assert_eq!(outcome.checkpoint_interval, 10, "default interval");
        assert!(!outcome.f16_checkpoints, "default stores f32 checkpoints");

        // Real bound behavior: soundness requires lower <= 0 and upper >= 2
        // (true range [0, 2]); the streaming path also tightens against the
        // forward IBP checkpoint, so lower >= 0 and upper <= 4.
        let bounds = verified_bounds(&outcome);
        assert_eq!(bounds.len(), 2);
        for (i, b) in bounds.iter().enumerate() {
            assert!(
                b.lower().abs() < 1e-5,
                "output {i}: lower must be 0 (sound <= 0, IBP-tightened >= 0), got {}",
                b.lower()
            );
            assert!(
                b.upper() >= 2.0 - 1e-4 && b.upper() <= 4.0 + 1e-4,
                "output {i}: upper must cover the true max 2 and be IBP-tightened <= 4, got {}",
                b.upper()
            );
        }

        let summary = streaming_summary(&outcome);
        assert!(summary.contains("verified"), "summary: {summary}");
        assert!(summary.contains("3 layers"), "summary: {summary}");
        assert!(summary.contains("f32"), "summary: {summary}");
    }

    #[test]
    fn ny_streaming_reports_unknown_with_gap_for_unsatisfiable_spec() {
        let net = linear_relu_linear_network();
        // True output range is [0, 2]; requiring [-0.5, 0.5] is falsifiable, so
        // no sound bound-propagation verdict can be Verified. Streaming CROWN
        // does no counterexample search, so the verdict must be Unknown with a
        // bounds-too-loose gap of at least 2 - 0.5 = 1.5.
        let spec = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            vec![Bound::new(-0.5, 0.5), Bound::new(-0.5, 0.5)],
        )
        .expect("valid spec");

        let outcome = verify_model_streaming(&net, &spec).expect("streaming CROWN should run");

        match &outcome.result {
            VerificationResult::Unknown { reason, bounds, .. } => {
                assert_eq!(bounds.len(), 2, "computed bounds must be reported");
                match reason {
                    UnknownReason::BoundsTooLoose { gap: Some(gap) } => {
                        assert!(
                            *gap >= 1.5 - 1e-4,
                            "upper >= 2 vs required 0.5 implies gap >= 1.5, got {gap}"
                        );
                    }
                    other => panic!("expected BoundsTooLoose with gap, got {other:?}"),
                }
            }
            other => panic!("expected Unknown, got {other:?}"),
        }

        let summary = streaming_summary(&outcome);
        assert!(summary.contains("unproven"), "summary: {summary}");
    }

    #[test]
    fn ny_streaming_with_f16_checkpoints_stays_sound() {
        let net = linear_relu_linear_network();
        let spec = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            vec![Bound::new(-10.0, 10.0), Bound::new(-10.0, 10.0)],
        )
        .expect("valid spec");

        // Checkpoint every layer, stored compressed: exercises the f16
        // decompress-and-widen path on every backward step.
        let cfg = StreamingConfig {
            checkpoint_interval: 1,
            use_f16_checkpoints: true,
            f16_widening_epsilon: 0.001,
        };

        let outcome = verify_model_streaming_with_config(&net, &spec, &cfg)
            .expect("f16 streaming CROWN should run");

        assert!(outcome.is_verified(), "wide spec should still verify");
        assert!(outcome.f16_checkpoints);
        assert_eq!(outcome.checkpoint_interval, 1);

        // Widened f16 checkpoints may loosen bounds slightly but must stay
        // sound: upper still covers the true max 2, lower stays near 0.
        let bounds = verified_bounds(&outcome);
        for (i, b) in bounds.iter().enumerate() {
            assert!(
                b.upper() >= 2.0 - 1e-3 && b.upper() <= 4.1,
                "output {i}: unsound/degenerate upper {}",
                b.upper()
            );
            assert!(
                b.lower() <= 1e-3 && b.lower() >= -0.1,
                "output {i}: unsound/degenerate lower {}",
                b.lower()
            );
        }

        let summary = streaming_summary(&outcome);
        assert!(summary.contains("f16"), "summary: {summary}");
    }

    #[test]
    fn ny_streaming_memory_estimate_reports_the_trade() {
        let net = linear_relu_linear_network();
        let estimate = estimate_streaming_savings(&net, 1000, &default_streaming_config());

        // 3 layers x 1000 elements x 8 bytes (f32 lower + upper) baseline;
        // streaming keeps ceil(3/10) = 1 checkpoint + the input.
        assert_eq!(estimate.num_layers, 3);
        assert_eq!(estimate.baseline_bytes, 3 * 1000 * 8);
        assert_eq!(estimate.streaming_bytes, 2 * 1000 * 8);
        assert!(
            estimate.savings_percent > 33.0 && estimate.savings_percent < 34.0,
            "expected ~33.3% savings, got {}",
            estimate.savings_percent
        );
        assert_eq!(estimate.bytes_saved(), 1000 * 8);
        assert_eq!(estimate.recompute_factor, 10);
        assert!(!estimate.f16_checkpoints);

        let summary = memory_savings_summary(&estimate);
        assert!(summary.contains("% saved"), "summary: {summary}");
        assert!(summary.contains("~10x recompute"), "summary: {summary}");
    }

    #[test]
    fn ny_streaming_memory_estimate_scales_at_transformer_depth() {
        // A 100-layer identity stack: at interval 10, streaming should reclaim
        // ~89% of the activation memory (11 checkpoints + input vs 100 layers).
        let mut net = Network::new();
        for _ in 0..100 {
            let w = ndarray::Array2::from_shape_vec((1, 1), vec![1.0]).expect("1x1 w");
            net.add_layer(Layer::Linear(
                LinearLayer::new(w, None).expect("valid 1x1 linear"),
            ));
        }

        let estimate = estimate_streaming_savings(&net, 1000, &default_streaming_config());
        assert!(
            estimate.savings_percent > 85.0 && estimate.savings_percent < 95.0,
            "expected ~89% savings at depth 100, got {}",
            estimate.savings_percent
        );
        assert!(estimate.streaming_bytes < estimate.baseline_bytes);
    }

    #[test]
    fn default_config_matches_streaming_default() {
        let cfg = default_streaming_config();
        assert_eq!(cfg.checkpoint_interval, 10);
        assert!(!cfg.use_f16_checkpoints);
        assert!((cfg.f16_widening_epsilon - 0.001).abs() < 1e-9);
    }
}
