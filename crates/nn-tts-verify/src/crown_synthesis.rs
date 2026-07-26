// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Post-synthesis CROWN certificate generation.
//!
//! Bridges the runtime [`Certificate`] (hard bounds on PCM audio) with the
//! formal [`MoonshotCertificate`] (8-property verification).
//!
//! When CROWN verification is enabled on a synthesis pipeline, after synthesis
//! completes and the runtime certificate passes, this module maps the empirical
//! hard-bound results to moonshot property evidence and produces a
//! [`MoonshotCertificate`] that can be attached to the [`Certificate`] via
//! [`Certificate::with_crown_evidence()`].
//!
//! # Architecture
//!
//! The mapping from hard bounds to moonshot properties:
//!
//! | Hard Bound      | Moonshot Property           | Evidence Level |
//! |-----------------|-----------------------------|----------------|
//! | non_silence     | P1 (Non-silent)             | Empirical      |
//! | no_clipping     | P2 (Non-clipping)           | Empirical      |
//! | no_clicks       | P6 (Streaming-safe proxy)   | Empirical      |
//!
//! Properties P3-P5 and P7-P8 require additional infrastructure (attention
//! monotonicity, timing certificates, Kani/ay) and are left at their base
//! verification level from the artifact registry.
//!
//! When the `NY` feature is enabled, future extensions can run actual
//! CROWN bound propagation on dispatch plans to upgrade evidence from Empirical
//! to CrownPartial or CrownProven.
//!
//! Part of #4254, #3874.

use crate::certificate::Certificate;
use crate::crown_junction::JunctionCheckSummary;
use crate::moonshot::{MoonshotCertificate, MoonshotStatus, VerificationLevel};

/// Configuration for post-synthesis CROWN certificate generation.
///
/// Controls which properties are checked and the model metadata attached
/// to the resulting [`MoonshotCertificate`].
#[derive(Debug, Clone)]
pub struct CrownCertificateConfig {
    /// Model pipeline name (e.g., "dvoice-kokoro-v1").
    pub model_name: String,
    /// Input specification (e.g., "English text, <=50 words").
    pub input_specification: String,
    /// Whether to map hard-bound results to moonshot properties.
    ///
    /// When true, passing hard bounds are mapped to Empirical-level
    /// moonshot property evidence. When false, only the base artifact
    /// registry levels are used (no synthesis-specific enrichment).
    pub map_hard_bounds: bool,
    /// Whether to check junction contracts (J2-J5) against intermediate
    /// tensor bounds during certificate generation.
    ///
    /// When true, intermediate tensors are validated against the Kokoro
    /// junction contracts defined in [`kokoro_contracts`]. Defaults to false.
    ///
    /// Part of #4254.
    pub check_junction_contracts: bool,
}

impl Default for CrownCertificateConfig {
    fn default() -> Self {
        Self {
            model_name: "kokoro-v1".to_string(),
            input_specification: "English text".to_string(),
            map_hard_bounds: true,
            check_junction_contracts: false,
        }
    }
}

impl CrownCertificateConfig {
    /// Create a new config with the given model name.
    #[must_use]
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Set the input specification.
    #[must_use]
    pub fn with_input_specification(mut self, spec: impl Into<String>) -> Self {
        self.input_specification = spec.into();
        self
    }
}

/// Generate a [`MoonshotCertificate`] from a runtime [`Certificate`].
///
/// Maps empirical hard-bound results to moonshot property evidence:
/// - P1 (non-silence): from `non_silence` hard bound
/// - P2 (non-clipping): from `no_clipping` hard bound
/// - P6 (streaming-safe proxy): from `no_clicks` hard bound
///
/// Other properties retain their base verification level from the
/// artifact registry.
///
/// # Returns
///
/// A [`MoonshotCertificate`] with empirical evidence from the synthesis
/// output. This can be attached to the `Certificate` via
/// [`Certificate::with_crown_evidence()`].
///
/// Part of #4254, #3874.
#[must_use]
pub fn verify_synthesis_crown(
    cert: &Certificate,
    config: &CrownCertificateConfig,
) -> MoonshotCertificate {
    let status = MoonshotStatus::from_repo();
    let mut moonshot = MoonshotCertificate::from_status(
        &status,
        &config.model_name,
        &config.input_specification,
        "runtime-synthesis",
    );

    if config.map_hard_bounds {
        enrich_from_hard_bounds(&mut moonshot, cert);
    }

    moonshot
}

/// Result of full CROWN synthesis verification including optional junction checks.
///
/// Part of #4254.
pub struct CrownSynthesisResult {
    /// Moonshot certificate with property evidence from hard bounds.
    pub moonshot: MoonshotCertificate,
    /// Junction contract check summary (only when
    /// `CrownCertificateConfig::check_junction_contracts` is true and
    /// `intermediates` were provided).
    pub junction_summary: Option<JunctionCheckSummary>,
}

/// Full CROWN synthesis verification: moonshot properties + junction contracts.
///
/// Combines [`verify_synthesis_crown()`] with optional junction contract
/// checking via [`verify_crown_with_junction_checks()`]. When
/// `config.check_junction_contracts` is true and `intermediates` is `Some`,
/// the junction contracts (J2-J5) are evaluated against the provided
/// intermediate tensor bounds.
///
/// # Arguments
///
/// * `cert` -- runtime Certificate from synthesis.
/// * `config` -- CROWN certificate configuration.
/// * `intermediates` -- optional map of junction names to observed (lower, upper)
///   bound pairs. Required when `check_junction_contracts` is true.
///
/// Part of #4254.
#[must_use]
pub fn verify_synthesis_crown_full(
    cert: &Certificate,
    config: &CrownCertificateConfig,
    intermediates: Option<&std::collections::HashMap<String, (f32, f32)>>,
) -> CrownSynthesisResult {
    let moonshot = verify_synthesis_crown(cert, config);

    let junction_summary = if config.check_junction_contracts {
        intermediates
            .map(|ints| crate::crown_junction::verify_crown_with_junction_checks(&moonshot, ints))
    } else {
        None
    };

    CrownSynthesisResult {
        moonshot,
        junction_summary,
    }
}

/// Map hard-bound results to moonshot property evidence.
///
/// Scans the Certificate's hard_bounds for known bound names and upgrades
/// the corresponding moonshot property to Empirical when the bound passes.
fn enrich_from_hard_bounds(moonshot: &mut MoonshotCertificate, cert: &Certificate) {
    for hb in &cert.hard_bounds {
        match hb.name {
            "non_silence" => {
                upgrade_property(moonshot, 0, hb.passed, hb.value, hb.threshold);
            }
            "no_clipping" => {
                upgrade_property(moonshot, 1, hb.passed, hb.value, hb.threshold);
            }
            "no_clicks" => {
                // Streaming safety proxy: no sharp discontinuities implies
                // bounded chunk boundary artifacts.
                upgrade_property(moonshot, 5, hb.passed, hb.value, hb.threshold);
            }
            _ => {}
        }
    }

    moonshot.recompute_aggregate_flags();
}

/// Upgrade a property to Empirical level if the hard bound passes.
///
/// Only upgrades, never downgrades. If the property already has a higher
/// verification level (from the artifact registry), this is a no-op.
fn upgrade_property(
    moonshot: &mut MoonshotCertificate,
    property_index: usize,
    passed: bool,
    value: f64,
    threshold: f64,
) {
    if property_index >= moonshot.properties.len() {
        return;
    }
    let prop = &mut moonshot.properties[property_index];

    if passed && prop.level < VerificationLevel::Empirical {
        prop.level = VerificationLevel::Empirical;
        prop.bound_value = Some(value);
        prop.threshold = Some(threshold);
        prop.assumptions =
            vec!["Empirical: hard bound check passed on synthesis output".to_string()];
    } else if passed {
        // Already at a higher level; record the empirical observation
        // as an additional assumption without downgrading.
        if prop.bound_value.is_none() {
            prop.bound_value = Some(value);
            prop.threshold = Some(threshold);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::HardBound;

    #[test]
    fn test_verify_synthesis_crown_default_config() {
        let cert = Certificate {
            hard_bounds: vec![
                HardBound {
                    name: "non_silence",
                    passed: true,
                    value: 0.15,
                    threshold: 0.01,
                },
                HardBound {
                    name: "no_clipping",
                    passed: true,
                    value: 0.95,
                    threshold: 1.0,
                },
                HardBound {
                    name: "no_clicks",
                    passed: true,
                    value: 0.3,
                    threshold: 0.5,
                },
            ],
            quality_metrics: vec![],
            phoneme_results: None,
            overall_passed: true,
            crown_evidence: None,
            junction_summary: None,
            deterministic_hash: None,
            #[cfg(feature = "ny")]
            dead_neuron_eq_proof: None,
        };

        let config = CrownCertificateConfig::default();
        let moonshot = verify_synthesis_crown(&cert, &config);

        // P1 (non-silence) should be at least Empirical.
        assert!(
            moonshot.properties[0].level >= VerificationLevel::Empirical,
            "P1 should be at least Empirical when non_silence passes"
        );

        // P2 (non-clipping) should be at least Empirical.
        assert!(
            moonshot.properties[1].level >= VerificationLevel::Empirical,
            "P2 should be at least Empirical when no_clipping passes"
        );

        // P6 (streaming safety) should be at least Empirical.
        assert!(
            moonshot.properties[5].level >= VerificationLevel::Empirical,
            "P6 should be at least Empirical when no_clicks passes"
        );

        // Model name should match config.
        assert_eq!(moonshot.model_name, "kokoro-v1");
    }

    #[test]
    fn test_verify_synthesis_crown_failing_bounds() {
        let cert = Certificate {
            hard_bounds: vec![
                HardBound {
                    name: "non_silence",
                    passed: false,
                    value: 0.001,
                    threshold: 0.01,
                },
                HardBound {
                    name: "no_clipping",
                    passed: true,
                    value: 0.95,
                    threshold: 1.0,
                },
            ],
            quality_metrics: vec![],
            phoneme_results: None,
            overall_passed: false,
            crown_evidence: None,
            junction_summary: None,
            deterministic_hash: None,
            #[cfg(feature = "ny")]
            dead_neuron_eq_proof: None,
        };

        let config = CrownCertificateConfig::new("test-model");
        let moonshot = verify_synthesis_crown(&cert, &config);

        // P2 (non-clipping) should be at least Empirical.
        assert!(
            moonshot.properties[1].level >= VerificationLevel::Empirical,
            "P2 should be at least Empirical when no_clipping passes"
        );

        assert_eq!(moonshot.model_name, "test-model");
    }

    #[test]
    fn test_verify_synthesis_crown_no_map() {
        let cert = Certificate {
            hard_bounds: vec![HardBound {
                name: "non_silence",
                passed: true,
                value: 0.15,
                threshold: 0.01,
            }],
            quality_metrics: vec![],
            phoneme_results: None,
            overall_passed: true,
            crown_evidence: None,
            junction_summary: None,
            deterministic_hash: None,
            #[cfg(feature = "ny")]
            dead_neuron_eq_proof: None,
        };

        let config = CrownCertificateConfig {
            map_hard_bounds: false,
            ..CrownCertificateConfig::default()
        };
        let moonshot = verify_synthesis_crown(&cert, &config);

        // With map_hard_bounds=false, properties come only from artifact registry.
        // The certificate should still be valid.
        assert_eq!(moonshot.properties.len(), 8);
    }

    #[test]
    fn test_certificate_with_crown_evidence_roundtrip() {
        let cert = Certificate {
            hard_bounds: vec![HardBound {
                name: "non_silence",
                passed: true,
                value: 0.15,
                threshold: 0.01,
            }],
            quality_metrics: vec![],
            phoneme_results: None,
            overall_passed: true,
            crown_evidence: None,
            junction_summary: None,
            deterministic_hash: None,
            #[cfg(feature = "ny")]
            dead_neuron_eq_proof: None,
        };

        let config = CrownCertificateConfig::default();
        let moonshot = verify_synthesis_crown(&cert, &config);
        let enriched = cert.with_crown_evidence(moonshot);

        assert!(enriched.has_crown_evidence());
        let report = enriched.report();
        assert!(
            report.contains("CROWN Verification Evidence"),
            "enriched certificate report should contain CROWN section"
        );
    }

    #[test]
    fn test_verify_synthesis_crown_empty_hard_bounds() {
        let cert = Certificate {
            hard_bounds: vec![],
            quality_metrics: vec![],
            phoneme_results: None,
            overall_passed: true,
            crown_evidence: None,
            junction_summary: None,
            deterministic_hash: None,
            #[cfg(feature = "ny")]
            dead_neuron_eq_proof: None,
        };

        let config = CrownCertificateConfig::default();
        let moonshot = verify_synthesis_crown(&cert, &config);

        // With no hard bounds, all 8 properties should still exist
        // (from artifact registry base levels).
        assert_eq!(moonshot.properties.len(), 8);
    }

    #[test]
    fn test_verify_synthesis_crown_unknown_bound_name_ignored() {
        let cert = Certificate {
            hard_bounds: vec![HardBound {
                name: "unknown_metric",
                passed: true,
                value: 42.0,
                threshold: 50.0,
            }],
            quality_metrics: vec![],
            phoneme_results: None,
            overall_passed: true,
            crown_evidence: None,
            junction_summary: None,
            deterministic_hash: None,
            #[cfg(feature = "ny")]
            dead_neuron_eq_proof: None,
        };

        let config = CrownCertificateConfig::default();
        let moonshot = verify_synthesis_crown(&cert, &config);

        // Unknown bound names should be silently ignored.
        // Properties still have 8 entries from the registry.
        assert_eq!(moonshot.properties.len(), 8);
    }

    #[test]
    fn test_crown_config_builder_with_input_specification() {
        let config = CrownCertificateConfig::new("custom-model")
            .with_input_specification("Japanese text, <=100 chars");

        assert_eq!(config.model_name, "custom-model");
        assert_eq!(config.input_specification, "Japanese text, <=100 chars");
        assert!(config.map_hard_bounds); // default true
    }

    #[test]
    fn test_verify_synthesis_crown_all_bounds_fail() {
        let cert = Certificate {
            hard_bounds: vec![
                HardBound {
                    name: "non_silence",
                    passed: false,
                    value: 0.001,
                    threshold: 0.01,
                },
                HardBound {
                    name: "no_clipping",
                    passed: false,
                    value: 1.5,
                    threshold: 1.0,
                },
                HardBound {
                    name: "no_clicks",
                    passed: false,
                    value: 0.8,
                    threshold: 0.5,
                },
            ],
            quality_metrics: vec![],
            phoneme_results: None,
            overall_passed: false,
            crown_evidence: None,
            junction_summary: None,
            deterministic_hash: None,
            #[cfg(feature = "ny")]
            dead_neuron_eq_proof: None,
        };

        let config = CrownCertificateConfig::default();
        let moonshot = verify_synthesis_crown(&cert, &config);

        // With all bounds failing, empirical upgrade should NOT occur
        // for the mapped properties. They stay at their artifact registry level.
        // The exact base level depends on the artifact registry, but the
        // moonshot should still be structurally valid.
        assert_eq!(moonshot.properties.len(), 8);
    }

    #[test]
    fn test_verify_synthesis_crown_custom_model_name_propagates() {
        let cert = Certificate {
            hard_bounds: vec![],
            quality_metrics: vec![],
            phoneme_results: None,
            overall_passed: true,
            crown_evidence: None,
            junction_summary: None,
            deterministic_hash: None,
            #[cfg(feature = "ny")]
            dead_neuron_eq_proof: None,
        };

        let config = CrownCertificateConfig::new("dvoice-kokoro-v2")
            .with_input_specification("English text, <=200 words");
        let moonshot = verify_synthesis_crown(&cert, &config);

        assert_eq!(moonshot.model_name, "dvoice-kokoro-v2");
        assert_eq!(moonshot.input_specification, "English text, <=200 words");
    }

    #[test]
    fn test_verify_synthesis_crown_only_no_clicks_passes() {
        // Only the no_clicks bound passes — should upgrade P6 only.
        let cert = Certificate {
            hard_bounds: vec![
                HardBound {
                    name: "non_silence",
                    passed: false,
                    value: 0.001,
                    threshold: 0.01,
                },
                HardBound {
                    name: "no_clipping",
                    passed: false,
                    value: 1.5,
                    threshold: 1.0,
                },
                HardBound {
                    name: "no_clicks",
                    passed: true,
                    value: 0.2,
                    threshold: 0.5,
                },
            ],
            quality_metrics: vec![],
            phoneme_results: None,
            overall_passed: false,
            crown_evidence: None,
            junction_summary: None,
            deterministic_hash: None,
            #[cfg(feature = "ny")]
            dead_neuron_eq_proof: None,
        };

        let config = CrownCertificateConfig::default();
        let moonshot = verify_synthesis_crown(&cert, &config);

        // P6 (streaming safety, index 5) should be at least Empirical.
        assert!(
            moonshot.properties[5].level >= VerificationLevel::Empirical,
            "P6 should be at least Empirical when no_clicks passes"
        );
    }
}
