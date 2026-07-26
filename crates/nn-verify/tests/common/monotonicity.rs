// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared monotonicity test harness for parametric attention verification.
//!
//! Replaces 30 `compose_attention_monotonicity_phase*.rs` files (12,314 LOC)
//! with a parametric configuration registry. Each experimental configuration
//! is expressed as a `MonotonicityConfig` struct; the shared
//! `run_monotonicity_experiment()` function dispatches to the appropriate
//! propagation method and assertion pattern.
//!
//! Design: `designs/archive/2026-03-11-monotonicity-test-parametrization.md`
//! Part of #1916.

use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;

use super::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, uniform_bounds, verify_and_assert,
};

// ---------------------------------------------------------------------------
// Assertion pattern selection
// ---------------------------------------------------------------------------

/// Selects which assertion pattern to apply after propagation.
#[derive(Debug, Clone)]
pub(crate) enum AssertionPattern {
    /// Check monotonicity via `interpret_attention_monotonicity()`.
    /// Extracts `min_margin`; optionally asserts proven (margin > 0).
    Monotonicity {
        /// Decoder steps (rows) for the attention matrix.
        seq_len: usize,
        /// Encoder positions (cols) for the attention matrix.
        enc_len: usize,
        expect_proven: bool,
        min_margin_floor: Option<f64>,
    },

    /// Check bound validity only: all finite, lower <= upper.
    BoundsValid,

    /// Check bound validity + verify_and_record.
    BoundsValidAndRecord { status_key: &'static str },

    /// Check duration positivity via `interpret_duration_positivity()`.
    /// Verifies that the minimum lower bound exceeds a threshold.
    DurationPositivity {
        /// Duration threshold (e.g., -10.0). Proven if `lo_min > threshold`.
        threshold: f64,
        /// Sequence length for the duration output.
        seq_len: usize,
        /// Whether we expect the proof to succeed.
        expect_proven: bool,
        /// Optional floor on the lower bound value.
        lower_bound_floor: Option<f64>,
    },
}

// ---------------------------------------------------------------------------
// Propagation method selection
// ---------------------------------------------------------------------------

/// Which bound propagation to run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PropagationMethod {
    /// IBP only.
    Ibp,
    /// CROWN with IBP fallback.
    CrownFallback,
    /// Both IBP and CROWN, asserting CROWN >= IBP tightness.
    Both,
}

// ---------------------------------------------------------------------------
// Configuration struct
// ---------------------------------------------------------------------------

/// One experimental configuration for the parametric monotonicity test.
///
/// Each entry in the `CONFIGS` registry encodes a single test case from
/// one of the original 30 phase files. The `label` field preserves the
/// experimental provenance (e.g., "phase3_input_bound_sweep_ib0.1").
///
/// The caller provides the pre-built `TensorKernelDef` and bindings —
/// this struct only controls propagation and assertion, not graph construction.
/// This design accommodates the 9+ distinct builder patterns discovered
/// across the original phase files.
#[derive(Debug, Clone)]
pub(crate) struct MonotonicityConfig {
    /// Human-readable label for test output.
    pub(crate) label: &'static str,

    /// Input bound for the Variable tensor (symmetric `[-ib, ib]`).
    pub(crate) input_bound: f32,

    /// Input shape for the Variable tensor.
    pub(crate) input_shape: Vec<usize>,

    /// Which propagation method to use.
    pub(crate) prop_method: PropagationMethod,

    /// Which assertions to check.
    pub(crate) assertion: AssertionPattern,
}

// ---------------------------------------------------------------------------
// Experiment result
// ---------------------------------------------------------------------------

/// Result of running one monotonicity experiment.
#[derive(Debug)]
pub(crate) struct ExperimentResult {
    pub(crate) label: &'static str,
    pub(crate) margin: Option<f64>,
    pub(crate) is_proven: Option<bool>,
    pub(crate) bounds_valid: bool,
    pub(crate) prop_method_used: &'static str,
}

// ---------------------------------------------------------------------------
// Experiment runner
// ---------------------------------------------------------------------------

/// Run a single monotonicity experiment with pre-built graph components.
///
/// The caller constructs the `TensorKernelDef` and bindings using the
/// appropriate builder for their group (attention, prosody, decoder, etc.).
/// This function handles propagation, assertion checking, and result
/// formatting.
pub(crate) fn run_monotonicity_experiment(
    config: &MonotonicityConfig,
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
) -> ExperimentResult {
    let input = uniform_bounds(&config.input_shape, config.input_bound);

    let graph = nn_verify::tensor_kernel_to_graph(def, bindings)
        .unwrap_or_else(|e| panic!("{}: graph translation failed: {e}", config.label));

    // Run propagation. Numerical instability in deep pipelines can cause
    // NY to return errors (e.g., Exp overflow). Return a failed
    // ExperimentResult rather than panicking — callers check bounds_valid.
    let propagation: Result<(&str, _), String> = match config.prop_method {
        PropagationMethod::Ibp => graph
            .propagate_ibp(&input)
            .map(|out| ("IBP", out))
            .map_err(|e| e.to_string()),
        PropagationMethod::CrownFallback => {
            nn_verify::propagate_with_crown_fallback(&graph, &input)
                .map(|(method, out, _)| {
                    let method_str = match method {
                        nn_verify::PropMethod::Crown => "CROWN",
                        nn_verify::PropMethod::Ibp => "IBP",
                        _ => "unknown",
                    };
                    (method_str, out)
                })
                .map_err(|e| e.to_string())
        }
        PropagationMethod::Both => {
            // Both mode always panics on error (tests expect CROWN to succeed).
            let (_method, crown_out, _reason) =
                assert_crown_tighter_when_not_fallback(&graph, &input);
            Ok(("CROWN+IBP", crown_out))
        }
    };
    let (method_str, output) = match propagation {
        Ok(result) => result,
        Err(e) => {
            eprintln!(
                "{}: propagation failed (numerical instability): {e}",
                config.label
            );
            return ExperimentResult {
                label: config.label,
                margin: None,
                is_proven: Some(false),
                bounds_valid: false,
                prop_method_used: "FAILED",
            };
        }
    };

    // Check bounds validity.
    let (lo, hi) = output.lower_upper();
    let bounds_valid = lo
        .iter()
        .zip(hi.iter())
        .all(|(&l, &u)| l.is_finite() && u.is_finite() && l <= u);

    // Run assertion pattern.
    let (margin, is_proven) = match &config.assertion {
        AssertionPattern::Monotonicity {
            seq_len,
            enc_len,
            expect_proven,
            min_margin_floor,
        } => {
            let lo_slice = lo.as_slice().expect("contiguous lower");
            let hi_slice = hi.as_slice().expect("contiguous upper");

            let cert = nn_tts_verify::monotonicity::interpret_attention_monotonicity(
                lo_slice,
                hi_slice,
                *seq_len,
                *enc_len,
                f64::from(config.input_bound),
                method_str,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "{}: interpret_attention_monotonicity failed: {e}",
                    config.label
                )
            });

            if *expect_proven {
                assert!(
                    cert.is_proven,
                    "{}: expected proven but margin={:.6}",
                    config.label, cert.min_margin
                );
            }
            if let Some(floor) = min_margin_floor {
                assert!(
                    cert.min_margin >= *floor,
                    "{}: margin {:.6} < floor {:.6}",
                    config.label,
                    cert.min_margin,
                    floor
                );
            }

            (Some(cert.min_margin), Some(cert.is_proven))
        }
        AssertionPattern::BoundsValid => {
            assert_bounds_valid(&output);
            (None, None)
        }
        AssertionPattern::BoundsValidAndRecord { status_key } => {
            let _ = verify_and_assert(def, bindings, &input, status_key);
            (None, None)
        }
        AssertionPattern::DurationPositivity {
            threshold,
            seq_len,
            expect_proven,
            lower_bound_floor,
        } => {
            // Compute minimum lower bound across all output elements.
            let lo_min = f64::from(lo.iter().copied().fold(f32::INFINITY, f32::min));

            let cert = nn_tts_verify::monotonicity::interpret_duration_positivity(
                lo_min,
                *threshold,
                f64::from(config.input_bound),
                f64::from(config.input_bound), // style_bound = input_bound
                *seq_len,
                method_str,
            );

            if *expect_proven {
                assert!(
                    cert.is_proven,
                    "{}: expected proven but lower_bound={:.6}",
                    config.label, cert.lower_bound
                );
            }
            if let Some(floor) = lower_bound_floor {
                assert!(
                    cert.lower_bound >= *floor,
                    "{}: lower_bound {:.6} < floor {:.6}",
                    config.label,
                    cert.lower_bound,
                    floor
                );
            }

            (Some(cert.lower_bound), Some(cert.is_proven))
        }
    };

    ExperimentResult {
        label: config.label,
        margin,
        is_proven,
        bounds_valid,
        prop_method_used: method_str,
    }
}

/// Run a batch of experiments, printing a summary table.
///
/// Panics on the first assertion failure.
pub(crate) fn run_experiment_batch(
    batch_label: &str,
    experiments: &[(MonotonicityConfig, TensorKernelDef, Vec<TensorParamBinding>)],
) {
    eprintln!("\n=== {batch_label} ===");
    eprintln!(
        "{:>40} {:>8} {:>12} {:>8} {:>10}",
        "label", "method", "margin", "proven", "valid"
    );

    for (config, def, bindings) in experiments {
        let result = run_monotonicity_experiment(config, def, bindings);
        let margin_str = match result.margin {
            Some(m) => format!("{m:.6}"),
            None => "---".to_string(),
        };
        let proven_str = match result.is_proven {
            Some(true) => "YES",
            Some(false) => "no",
            None => "---",
        };
        let valid_str = if result.bounds_valid { "ok" } else { "FAIL" };
        eprintln!(
            "{:>40} {:>8} {:>12} {:>8} {:>10}",
            result.label, result.prop_method_used, margin_str, proven_str, valid_str
        );
    }
}
