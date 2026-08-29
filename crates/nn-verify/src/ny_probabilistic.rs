// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Probabilistic verification adapter — consumes `ny_api::probabilistic`.
//!
//! Deterministic CROWN bounds answer "can the output ever leave this box?";
//! the probabilistic surface answers the complementary question "where does the
//! output *typically* live, and with what confidence?". This module wires NN
//! model graphs into ny's Monte-Carlo + concentration workflow:
//!
//! 1. **Sample** the input region with [`ny_api::probabilistic::MonteCarloVerifier`]
//!    (uniform or Gaussian [`ny_api::probabilistic::InputDistribution`], seedable
//!    for bit-reproducible runs).
//! 2. **Evaluate** the [`ny_propagate::GraphNetwork`] on every sample
//!    ([`evaluate_graph_at`] — degenerate-interval IBP, i.e. an exact forward
//!    pass through the sound propagation engine).
//! 3. **Enclose** with a sound CROWN envelope over the whole region, recorded on
//!    the resulting [`ny_api::probabilistic::ProbabilisticBound`] so empirical
//!    statistics are always checked against a sound certificate.
//! 4. **Certify** the true output mean with non-asymptotic concentration
//!    inequalities: Hoeffding (range from CROWN) and optionally McDiarmid
//!    (Lipschitz from spectral norms), Bonferroni-corrected across output
//!    dimensions, via [`concentration_certificate`] and the
//!    `concentration_certificate_with_mcdiarmid[_optimistic]` variants.
//!
//! `nn-tts-verify` already consumes `ny_propagate::probabilistic` directly for
//! the Kokoro moonshot path; this adapter exposes the curated ny-api facade to
//! nn-verify proper, so probabilistic certificates are available for **all** NN
//! models, not just TTS. The heavier facade families (analytic distributional
//! propagation, CGF/Chernoff tails, higher moments, branch-and-bound
//! refinement) stay reachable at [`ny_api::probabilistic`] directly.
//!
//! This adapter is additive — it does not touch NN's existing verify path. It
//! exposes thin NN-facing wrappers ([`verify_model_monte_carlo`],
//! [`verify_model_monte_carlo_with_config`], the certificate constructors)
//! plus flat reporting helpers ([`probabilistic_summary`],
//! [`concentration_summary`]); callers that want the rich statistics read the
//! ny result types directly.
//!
//! ```rust,no_run
//! use nn_verify::ny_probabilistic::{probabilistic_summary, verify_model_monte_carlo};
//! # fn run(net: &ny_propagate::GraphNetwork, region: &ny_api::BoundedTensor)
//! #     -> ny_core::Result<()> {
//! let bound = verify_model_monte_carlo(net, region)?;
//! println!("{}", probabilistic_summary(&bound));
//! # Ok(())
//! # }
//! ```

use ndarray::ArrayD;
use ny_api::probabilistic::{
    estimate_lipschitz_from_network, ConcentrationCertificate, InputDistribution,
    MonteCarloVerifier, ProbabilisticBound,
};
use ny_api::BoundedTensor;

/// Default number of Monte-Carlo samples for NN model runs.
///
/// Enough for the Hoeffding half-width to be meaningfully below the CROWN
/// range (`epsilon ~ range * sqrt(ln(2/delta) / 2n)`) while a full graph
/// forward pass per sample stays cheap for unit-test-sized models.
pub const DEFAULT_MONTE_CARLO_SAMPLES: usize = 1024;

/// Default RNG seed for NN Monte-Carlo runs.
///
/// A fixed seed keeps NN verification runs bit-reproducible; callers that want
/// fresh randomness build their own [`MonteCarloVerifier`] without a seed.
pub const DEFAULT_MONTE_CARLO_SEED: u64 = 42;

/// Default Monte-Carlo configuration for NN model verification.
///
/// [`DEFAULT_MONTE_CARLO_SAMPLES`] uniform samples at the verifier's default
/// 95% confidence, seeded with [`DEFAULT_MONTE_CARLO_SEED`] so repeated runs
/// are bit-reproducible. The [`MonteCarloVerifier`] doubles as the config
/// object; exposed as a named helper so NN call sites document their intent
/// rather than reaching for `MonteCarloVerifier::new` inline.
#[must_use]
pub fn default_monte_carlo_config() -> MonteCarloVerifier {
    MonteCarloVerifier::new(DEFAULT_MONTE_CARLO_SAMPLES)
        .with_distribution(InputDistribution::Uniform)
        .with_seed(DEFAULT_MONTE_CARLO_SEED)
}

/// Evaluate a model graph at a single concrete input point.
///
/// Propagates the degenerate (zero-width) interval `[input, input]` through
/// the graph with IBP — an exact forward pass through the sound propagation
/// engine — and collapses any residual sound outward rounding at the interval
/// midpoint. Exposed so callers can produce their own output batches for
/// [`MonteCarloVerifier::compute_bounds`].
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while building the degenerate
/// input tensor or propagating it through a graph node.
pub fn evaluate_graph_at(
    net: &ny_propagate::GraphNetwork,
    input: &ArrayD<f32>,
) -> ny_core::Result<ArrayD<f32>> {
    let point = BoundedTensor::new(input.clone(), input.clone())?;
    let output = net.propagate_ibp(&point)?;
    Ok(output.center())
}

/// Run Monte-Carlo verification of a model graph over an input region using a
/// sensible default configuration.
///
/// Samples the region, evaluates the graph on every sample, and returns
/// empirical min/max/mean/std per output dimension together with a sound CROWN
/// envelope the empirical bounds must lie within
/// ([`ProbabilisticBound::is_consistent`]). For control over sample count,
/// input distribution, confidence, or seed, use
/// [`verify_model_monte_carlo_with_config`].
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while sampling, evaluating a
/// sample, or computing the CROWN envelope (including CROWN soundness
/// refusals, which this adapter deliberately does not degrade to IBP).
pub fn verify_model_monte_carlo(
    net: &ny_propagate::GraphNetwork,
    input_bounds: &BoundedTensor,
) -> ny_core::Result<ProbabilisticBound> {
    verify_model_monte_carlo_with_config(net, input_bounds, &default_monte_carlo_config())
}

/// Run Monte-Carlo verification of a model graph with an explicit
/// [`MonteCarloVerifier`] configuration.
///
/// See [`verify_model_monte_carlo`] for the default-config convenience wrapper
/// and the sampling/envelope semantics.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while sampling the input region,
/// evaluating a sample through the graph, or computing the sound CROWN
/// envelope.
pub fn verify_model_monte_carlo_with_config(
    net: &ny_propagate::GraphNetwork,
    input_bounds: &BoundedTensor,
    config: &MonteCarloVerifier,
) -> ny_core::Result<ProbabilisticBound> {
    let samples = config.sample_inputs(input_bounds)?;
    let outputs = samples
        .iter()
        .map(|sample| evaluate_graph_at(net, sample))
        .collect::<ny_core::Result<Vec<_>>>()?;
    let crown_bounds = net.propagate_crown_batched(input_bounds)?;
    config.compute_bounds(&outputs, Some(&crown_bounds))
}

/// Build a Hoeffding-only concentration certificate from a Monte-Carlo run.
///
/// Bounds the true output mean around the empirical mean with per-dimension
/// half-width `epsilon` derived from the CROWN range (distribution-free beyond
/// boundedness), Bonferroni-corrected so `confidence` is a valid simultaneous
/// guarantee across output dimensions. Hoeffding certificates are always sound
/// (`is_sound == true`).
///
/// # Errors
///
/// Returns [`ny_core::NyError::InvalidSpec`] when `bound` carries no CROWN
/// envelope (produced with `crown_bounds: None`), when `confidence` is outside
/// `(0, 1)`, or when `bound.num_samples` is zero.
pub fn concentration_certificate(
    bound: &ProbabilisticBound,
    confidence: f64,
) -> ny_core::Result<ConcentrationCertificate> {
    let crown = crown_envelope(bound)?;
    ConcentrationCertificate::compute(
        &bound.empirical_mean,
        &crown,
        bound.num_samples,
        confidence,
        true,
    )
}

/// Build a combined Hoeffding + McDiarmid concentration certificate from a
/// Monte-Carlo run, requiring a **sound** Lipschitz estimate.
///
/// The Lipschitz constant is estimated from `network` as the product of the
/// Linear layers' spectral norms; the McDiarmid interval is centered on the
/// Monte-Carlo empirical mean. Use [`tightest_epsilons`] to read the tighter
/// of the two inequalities per output dimension.
///
/// # Errors
///
/// Returns [`ny_core::NyError::InvalidSpec`] when `bound` carries no CROWN
/// envelope, when the Lipschitz estimate is not a sound upper bound (the
/// network contains layers outside the recognized 1-Lipschitz allow-list —
/// use [`concentration_certificate_with_mcdiarmid_optimistic`] to opt in),
/// or when `confidence` / `num_samples` are out of range. Propagates any
/// [`ny_core::NyError`] from Lipschitz estimation (e.g. an empty network).
pub fn concentration_certificate_with_mcdiarmid(
    bound: &ProbabilisticBound,
    network: &ny_propagate::Network,
    input_bounds: &BoundedTensor,
    confidence: f64,
) -> ny_core::Result<ConcentrationCertificate> {
    let crown = crown_envelope(bound)?;
    let lipschitz = estimate_lipschitz_from_network(network)?;
    ConcentrationCertificate::compute_with_mcdiarmid(
        &bound.empirical_mean,
        &crown,
        &bound.empirical_mean,
        input_bounds,
        &lipschitz,
        bound.num_samples,
        confidence,
        true,
    )
}

/// Build a combined Hoeffding + McDiarmid certificate accepting an optimistic
/// (possibly unsound) Lipschitz estimate.
///
/// Explicit opt-in variant of [`concentration_certificate_with_mcdiarmid`]:
/// layers outside the recognized allow-list are treated as 1-Lipschitz, and
/// the returned certificate's `is_sound` flag reflects the estimate's
/// soundness. Callers MUST check `is_sound` before using the certificate for
/// formal verification.
///
/// # Errors
///
/// Returns [`ny_core::NyError::InvalidSpec`] when `bound` carries no CROWN
/// envelope or when `confidence` / `num_samples` are out of range. Propagates
/// any [`ny_core::NyError`] from Lipschitz estimation.
pub fn concentration_certificate_with_mcdiarmid_optimistic(
    bound: &ProbabilisticBound,
    network: &ny_propagate::Network,
    input_bounds: &BoundedTensor,
    confidence: f64,
) -> ny_core::Result<ConcentrationCertificate> {
    let crown = crown_envelope(bound)?;
    let lipschitz = estimate_lipschitz_from_network(network)?;
    ConcentrationCertificate::compute_with_mcdiarmid_optimistic(
        &bound.empirical_mean,
        &crown,
        &bound.empirical_mean,
        input_bounds,
        &lipschitz,
        bound.num_samples,
        confidence,
        true,
    )
}

/// Tightest per-dimension confidence half-width from a concentration
/// certificate.
///
/// Takes `min(hoeffding.epsilon, mcdiarmid.epsilon)` per output dimension when
/// McDiarmid bounds are present (both inequalities hold simultaneously at the
/// certificate's per-dimension confidence, so the minimum is valid), otherwise
/// the Hoeffding epsilons directly.
#[must_use]
pub fn tightest_epsilons(certificate: &ConcentrationCertificate) -> Vec<f64> {
    match &certificate.mcdiarmid_bounds {
        Some(mcdiarmid) => certificate
            .hoeffding_bounds
            .iter()
            .zip(mcdiarmid.iter())
            .map(|(h, m)| h.epsilon.min(m.epsilon))
            .collect(),
        None => certificate
            .hoeffding_bounds
            .iter()
            .map(|h| h.epsilon)
            .collect(),
    }
}

/// Render an NN-friendly one-line summary of a Monte-Carlo run.
///
/// Reports sample count, confidence, output dimensionality, whether the
/// empirical bounds sit inside the sound CROWN envelope, and the widest
/// normal-approximation confidence interval on the mean. This is intentionally
/// a flat string for NN's reporting style; callers needing structured data
/// read the [`ProbabilisticBound`] fields directly.
#[must_use]
pub fn probabilistic_summary(bound: &ProbabilisticBound) -> String {
    let dims = bound.empirical_mean.len();
    let envelope = if bound.crown_lower.is_some() && bound.crown_upper.is_some() {
        if bound.is_consistent(1e-6) {
            "within CROWN envelope"
        } else {
            "OUTSIDE CROWN envelope"
        }
    } else {
        "no CROWN envelope"
    };
    let max_ci = bound
        .confidence_interval_width()
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    format!(
        "monte-carlo: {} sample{} @ {:.1}% confidence, {dims} output dim{}, {envelope}, \
         max mean-CI width {max_ci:.3e}",
        bound.num_samples,
        if bound.num_samples == 1 { "" } else { "s" },
        bound.confidence * 100.0,
        if dims == 1 { "" } else { "s" },
    )
}

/// Render an NN-friendly one-line summary of a concentration certificate.
///
/// Reports dimension count, overall confidence, the widest Hoeffding (and,
/// when present, McDiarmid) half-width, and whether every component of the
/// certificate is sound. Flat string by design; read the
/// [`ConcentrationCertificate`] fields for structured data.
#[must_use]
pub fn concentration_summary(certificate: &ConcentrationCertificate) -> String {
    let dims = certificate.hoeffding_bounds.len();
    let max_hoeffding = certificate
        .hoeffding_bounds
        .iter()
        .map(|b| b.epsilon)
        .fold(0.0_f64, f64::max);
    let mcdiarmid = match &certificate.mcdiarmid_bounds {
        Some(bounds) => {
            let max_eps = bounds.iter().map(|b| b.epsilon).fold(0.0_f64, f64::max);
            format!("mcdiarmid[max eps {max_eps:.3e}]")
        }
        None => "mcdiarmid[none]".to_string(),
    };
    format!(
        "concentration: {dims} dim{} @ {:.1}% confidence, \
         hoeffding[max eps {max_hoeffding:.3e}], {mcdiarmid}, {}",
        if dims == 1 { "" } else { "s" },
        certificate.overall_confidence * 100.0,
        if certificate.is_sound {
            "sound"
        } else {
            "OPTIMISTIC (not sound)"
        },
    )
}

/// Rebuild the sound CROWN envelope stored on a Monte-Carlo bound.
fn crown_envelope(bound: &ProbabilisticBound) -> ny_core::Result<BoundedTensor> {
    match (&bound.crown_lower, &bound.crown_upper) {
        (Some(lower), Some(upper)) => BoundedTensor::new(lower.clone(), upper.clone()),
        _ => Err(ny_core::NyError::InvalidSpec(
            "concentration certificate requires the Monte-Carlo bound to carry a sound CROWN \
             envelope (crown_lower/crown_upper); run verify_model_monte_carlo or pass \
             crown_bounds to MonteCarloVerifier::compute_bounds"
                .to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};
    use ny_propagate::layers::{Layer, LinearLayer, ReLULayer};
    use ny_propagate::{GraphNetwork, GraphNode, Network};

    /// Tiny Linear → ReLU → Linear graph (2→2→2) for end-to-end exercise.
    ///
    /// Mirrors the construction NN uses elsewhere (e.g. `ny_ladder` tests),
    /// keeping the network small and deterministic. Both output dimensions
    /// compute `|x1 - x2|`, so the reachable range over `[-1, 1]^2` is `[0, 2]`.
    fn linear_relu_linear_graph() -> GraphNetwork {
        let w1 =
            ndarray::Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).expect("2x2 w1");
        let linear1 = LinearLayer::new(w1, Some(ndarray::Array1::zeros(2))).expect("valid linear1");
        let w2 = ndarray::Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).expect("2x2 w2");
        let linear2 = LinearLayer::new(w2, Some(ndarray::Array1::zeros(2))).expect("valid linear2");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear2");
        graph
    }

    /// The same Linear → ReLU → Linear model as a sequential `Network`, for
    /// spectral-norm Lipschitz estimation (all layers on the sound allow-list).
    fn linear_relu_linear_network() -> Network {
        let w1 =
            ndarray::Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).expect("2x2 w1");
        let linear1 = LinearLayer::new(w1, Some(ndarray::Array1::zeros(2))).expect("valid linear1");
        let w2 = ndarray::Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).expect("2x2 w2");
        let linear2 = LinearLayer::new(w2, Some(ndarray::Array1::zeros(2))).expect("valid linear2");

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear1));
        network.add_layer(Layer::ReLU(ReLULayer::new()));
        network.add_layer(Layer::Linear(linear2));
        network
    }

    /// The input region `[-1, 1]^2`.
    fn unit_box() -> BoundedTensor {
        let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).expect("lower");
        let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).expect("upper");
        BoundedTensor::new(lower, upper).expect("valid box")
    }

    #[test]
    fn evaluate_graph_at_matches_hand_computation() {
        let graph = linear_relu_linear_graph();
        let x = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, -1.0]).expect("point");

        let y = evaluate_graph_at(&graph, &x).expect("forward evaluation");

        // W1·x = (2, -2); ReLU → (2, 0); W2·h = (2, 2).
        assert_eq!(y.shape(), &[2]);
        assert!((y[[0]] - 2.0).abs() < 1e-5, "y0 = {}", y[[0]]);
        assert!((y[[1]] - 2.0).abs() < 1e-5, "y1 = {}", y[[1]]);
    }

    #[test]
    fn monte_carlo_bound_stays_inside_crown_envelope() {
        let graph = linear_relu_linear_graph();
        let region = unit_box();
        let config = MonteCarloVerifier::new(256).with_seed(42);

        let bound = verify_model_monte_carlo_with_config(&graph, &region, &config)
            .expect("monte-carlo run on linear→relu→linear graph");

        // Sample-count accounting.
        assert_eq!(bound.num_samples, 256);
        assert_eq!(bound.empirical_mean.len(), 2);

        // Both output dims compute |x1 - x2| ∈ [0, 2] over the unit box.
        for j in 0..2 {
            assert!(
                bound.empirical_lower[[j]] >= -1e-5,
                "dim {j}: empirical lower {} below reachable minimum 0",
                bound.empirical_lower[[j]]
            );
            assert!(
                bound.empirical_upper[[j]] <= 2.0 + 1e-4,
                "dim {j}: empirical upper {} above reachable maximum 2",
                bound.empirical_upper[[j]]
            );
            assert!(
                bound.empirical_upper[[j]] > 0.5,
                "dim {j}: 256 uniform samples must explore the box, upper = {}",
                bound.empirical_upper[[j]]
            );
        }

        // The empirical statistics must sit inside the sound CROWN envelope…
        let crown_lower = bound.crown_lower.as_ref().expect("CROWN lower recorded");
        let crown_upper = bound.crown_upper.as_ref().expect("CROWN upper recorded");
        assert!(
            bound.is_consistent(1e-4),
            "empirical bounds must lie within the sound envelope"
        );
        // …and the envelope must contain the true reachable range [0, 2]
        // (0 attained at x1 == x2, 2 attained at (1, -1)).
        for j in 0..2 {
            assert!(
                crown_lower[[j]] <= 1e-5,
                "dim {j}: sound lower {} must not exceed attainable 0",
                crown_lower[[j]]
            );
            assert!(
                crown_upper[[j]] >= 2.0 - 1e-4,
                "dim {j}: sound upper {} must cover attainable 2",
                crown_upper[[j]]
            );
        }
    }

    #[test]
    fn seeded_monte_carlo_runs_are_bit_reproducible() {
        let graph = linear_relu_linear_graph();
        let region = unit_box();
        let config = MonteCarloVerifier::new(128).with_seed(7);

        let a = verify_model_monte_carlo_with_config(&graph, &region, &config).expect("first run");
        let b = verify_model_monte_carlo_with_config(&graph, &region, &config).expect("second run");

        for (x, y) in a.empirical_mean.iter().zip(b.empirical_mean.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "seeded means must reproduce");
        }
        for (x, y) in a.empirical_lower.iter().zip(b.empirical_lower.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "seeded lowers must reproduce");
        }
        for (x, y) in a.empirical_upper.iter().zip(b.empirical_upper.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "seeded uppers must reproduce");
        }
    }

    #[test]
    fn hoeffding_certificate_orders_by_sample_count() {
        let graph = linear_relu_linear_graph();
        let region = unit_box();

        let small = verify_model_monte_carlo_with_config(
            &graph,
            &region,
            &MonteCarloVerifier::new(256).with_seed(42),
        )
        .expect("256-sample run");
        let large = verify_model_monte_carlo_with_config(
            &graph,
            &region,
            &MonteCarloVerifier::new(1024).with_seed(42),
        )
        .expect("1024-sample run");

        let cert_small = concentration_certificate(&small, 0.99).expect("small certificate");
        let cert_large = concentration_certificate(&large, 0.99).expect("large certificate");

        assert!(cert_small.is_sound, "Hoeffding-only certificates are sound");
        assert!(cert_small.mcdiarmid_bounds.is_none());
        assert_eq!(cert_small.hoeffding_bounds.len(), 2);

        // Per-dimension accounting; Bonferroni over 2 dims at 99% confidence
        // means per-dim failure probability (1 - 0.99) / 2 = 0.005 by
        // construction of epsilon.
        for hb in &cert_small.hoeffding_bounds {
            assert_eq!(hb.num_samples, 256);
            assert!(hb.epsilon > 0.0 && hb.epsilon.is_finite());
            assert!(
                (hb.failure_probability - 0.005).abs() < 1e-9,
                "failure probability = {}",
                hb.failure_probability
            );
        }

        // Distribution-free ordering: same CROWN range, 4x the samples must
        // exactly halve the Hoeffding half-width (epsilon ∝ 1/√n).
        for (s, l) in cert_small
            .hoeffding_bounds
            .iter()
            .zip(cert_large.hoeffding_bounds.iter())
        {
            assert!(
                l.epsilon < s.epsilon,
                "more samples must tighten epsilon: {} !< {}",
                l.epsilon,
                s.epsilon
            );
            assert!(
                (l.epsilon / s.epsilon - 0.5).abs() < 1e-9,
                "epsilon ratio should be sqrt(256/1024) = 0.5, got {}",
                l.epsilon / s.epsilon
            );
        }
    }

    #[test]
    fn mcdiarmid_certificate_is_sound_for_linear_relu_linear() {
        let graph = linear_relu_linear_graph();
        let network = linear_relu_linear_network();
        let region = unit_box();
        let config = MonteCarloVerifier::new(256).with_seed(42);

        let bound = verify_model_monte_carlo_with_config(&graph, &region, &config)
            .expect("monte-carlo run");
        let combined = concentration_certificate_with_mcdiarmid(&bound, &network, &region, 0.99)
            .expect("Linear/ReLU are on the sound Lipschitz allow-list");

        assert!(combined.is_sound, "spectral-norm estimate must be sound");
        let mcdiarmid = combined
            .mcdiarmid_bounds
            .as_ref()
            .expect("McDiarmid component present");
        assert_eq!(mcdiarmid.len(), 2);
        for mb in mcdiarmid {
            assert!(mb.is_sound);
            assert!(mb.epsilon > 0.0 && mb.epsilon.is_finite());
            assert_eq!(mb.num_samples, 256);
        }

        // The combined view never loosens the Hoeffding half-width.
        let tightest = tightest_epsilons(&combined);
        assert_eq!(tightest.len(), 2);
        for (t, h) in tightest.iter().zip(combined.hoeffding_bounds.iter()) {
            assert!(
                *t <= h.epsilon,
                "tightest epsilon {t} must not exceed Hoeffding {}",
                h.epsilon
            );
        }
    }

    #[test]
    fn concentration_certificate_requires_crown_envelope() {
        let arr = |v: Vec<f32>| ArrayD::from_shape_vec(IxDyn(&[2]), v).expect("2-vector");
        let bound = ProbabilisticBound {
            empirical_lower: arr(vec![0.0, 0.0]),
            empirical_upper: arr(vec![1.0, 1.0]),
            empirical_mean: arr(vec![0.5, 0.5]),
            empirical_std: arr(vec![0.1, 0.1]),
            num_samples: 64,
            crown_lower: None,
            crown_upper: None,
            confidence: 0.95,
        };

        assert!(
            concentration_certificate(&bound, 0.95).is_err(),
            "a bound without a CROWN envelope must be refused"
        );
    }

    #[test]
    fn default_config_is_deterministic_uniform() {
        let config = default_monte_carlo_config();
        assert_eq!(config.num_samples, DEFAULT_MONTE_CARLO_SAMPLES);
        assert_eq!(config.seed, Some(DEFAULT_MONTE_CARLO_SEED));
        assert!(matches!(config.distribution, InputDistribution::Uniform));
        assert!((config.confidence - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn summaries_render_flat_reports() {
        let graph = linear_relu_linear_graph();
        let region = unit_box();
        let config = MonteCarloVerifier::new(256).with_seed(42);

        let bound = verify_model_monte_carlo_with_config(&graph, &region, &config)
            .expect("monte-carlo run");
        let summary = probabilistic_summary(&bound);
        assert!(summary.contains("256 samples"), "summary: {summary}");
        assert!(
            summary.contains("within CROWN envelope"),
            "summary: {summary}"
        );

        let certificate = concentration_certificate(&bound, 0.99).expect("certificate");
        let summary = concentration_summary(&certificate);
        assert!(summary.contains("hoeffding"), "summary: {summary}");
        assert!(summary.contains("sound"), "summary: {summary}");
        assert!(summary.contains("mcdiarmid[none]"), "summary: {summary}");
    }
}
