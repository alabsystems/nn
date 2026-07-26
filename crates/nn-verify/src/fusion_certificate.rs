// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fusion equivalence certificates for verified kernel fusion.
//!
//! A `FusionEquivalenceCertificate` captures the evidence that a fused kernel
//! produces identical output (within epsilon) to its sequential components.
//! This is the artifact that ships with a compiled model, enabling offline
//! auditing of fusion correctness.
//!
//! Two proof sources are supported:
//! - **CROWN bounds**: NY diamond DAG diff propagation (may be loose)
//! - **Analytical ULP**: IEEE 754 floating-point error analysis (tight)
//!
//! When both exist, the certificate is valid if EITHER bound proves equivalence.
//!
//! # Dimension coverage
//!
//! Fusion equivalence is an element-wise property. The scalar diamond DAG proof
//! covers all tensor dimensions because the fused and sequential kernels compute
//! the same scalar function independently at each element position. There are no
//! cross-element interactions in the fusion diff graph.

use serde::{Deserialize, Serialize};

use crate::error::VerifyError;
use crate::fusion_spec::FusionVerification;
use crate::soundness_compat::VerificationSoundnessMode;
use crate::verify_types::PropMethod;

/// Current certificate format version.
pub const FUSION_CERTIFICATE_VERSION: u32 = 1;

/// Certificate proving a fused kernel is equivalent to its sequential components.
///
/// Ships with the compiled model for offline auditing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FusionEquivalenceCertificate {
    /// Certificate format version.
    pub version: u32,
    /// Name of the fused kernel.
    pub fused_kernel_name: String,
    /// Names of the sequential kernel pair: (first, second).
    pub sequential_names: (String, String),
    /// Production dimension this certificate covers.
    pub dimension: usize,
    /// Epsilon threshold used.
    pub epsilon: f32,
    /// CROWN-computed maximum absolute diff (may be loose).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub crown_bound: Option<f32>,
    /// Propagation method that produced the CROWN bound.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub crown_method: Option<PropMethod>,
    /// Analytical IEEE 754 ULP error bound (tight).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub analytical_bound: Option<AnalyticalFusionBound>,
    /// Soundness classification.
    pub soundness_mode: VerificationSoundnessMode,
    /// Per-variable input bounds used in verification.
    pub variable_bounds: Vec<(f32, f32)>,
    /// SHA-256 hash of the fused kernel source definition.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub fused_source_hash: Option<String>,
    /// ISO 8601 generation timestamp.
    pub generated_at: String,
    /// Justification for dimension coverage (element-wise independence).
    pub dimension_coverage_rationale: String,

    // --- Integrity fields (same as ProofCertificate v4) ---
    /// SHA-256 hex digest of the canonical certificate content (all fields
    /// except `content_hash` and `hmac_signature`).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub content_hash: Option<String>,
    /// HMAC-SHA256 hex digest over the `content_hash`, keyed with a shared
    /// secret. Prevents forgery of fusion equivalence proofs.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub hmac_signature: Option<String>,
}

impl FusionEquivalenceCertificate {
    /// Construct from a `FusionVerification` result and metadata.
    ///
    /// The CROWN bound is populated from the verification result. The
    /// analytical bound can be added separately via `with_analytical_bound()`.
    ///
    /// # Arguments
    ///
    /// * `verification` — NY fusion verification result
    /// * `first_name` — name of the first sequential kernel
    /// * `second_name` — name of the second sequential kernel
    /// * `production_dim` — production tensor dimension (e.g. 512)
    /// * `variable_bounds` — per-variable input bounds used in verification
    pub fn from_verification(
        verification: &FusionVerification,
        first_name: &str,
        second_name: &str,
        production_dim: usize,
        variable_bounds: &[(f32, f32)],
    ) -> Self {
        Self {
            version: FUSION_CERTIFICATE_VERSION,
            fused_kernel_name: verification.fused_kernel_name.clone(),
            sequential_names: (first_name.to_string(), second_name.to_string()),
            dimension: production_dim,
            epsilon: verification.epsilon,
            crown_bound: Some(verification.max_abs_diff),
            crown_method: Some(verification.method),
            analytical_bound: None,
            soundness_mode: verification.soundness_mode,
            variable_bounds: variable_bounds.to_vec(),
            fused_source_hash: None,
            generated_at: now_iso8601(),
            dimension_coverage_rationale: ELEMENTWISE_RATIONALE.to_string(),
            content_hash: None,
            hmac_signature: None,
        }
    }

    /// Add an analytical ULP error bound.
    #[must_use]
    pub fn with_analytical_bound(mut self, bound: AnalyticalFusionBound) -> Self {
        self.analytical_bound = Some(bound);
        self
    }

    /// Add a SHA-256 source hash for the fused kernel definition.
    #[must_use]
    pub fn with_source_hash(mut self, hash: String) -> Self {
        self.fused_source_hash = Some(hash);
        self
    }

    /// Whether the certificate proves equivalence within epsilon.
    ///
    /// Returns true if EITHER the CROWN bound or the analytical bound
    /// proves the diff is within epsilon.
    pub fn proves_equivalence(&self) -> bool {
        let crown_ok = self.crown_bound.map(|b| b <= self.epsilon).unwrap_or(false);
        let analytical_ok = self
            .analytical_bound
            .as_ref()
            .map(|b| b.proves_within_epsilon(self.epsilon))
            .unwrap_or(false);
        crown_ok || analytical_ok
    }

    /// The tightest proved bound across all proof sources.
    pub fn tightest_bound(&self) -> Option<f64> {
        let crown = self.crown_bound.map(f64::from);
        let analytical = self.analytical_bound.as_ref().map(|b| b.max_abs_diff);
        match (crown, analytical) {
            (Some(c), Some(a)) => Some(c.min(a)),
            (Some(c), None) => Some(c),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    /// Validate certificate internal consistency.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if the certificate is malformed.
    pub fn validate(&self) -> Result<(), VerifyError> {
        if self.version == 0 || self.version > FUSION_CERTIFICATE_VERSION {
            return Err(VerifyError::InvalidCertificate {
                reason: format!(
                    "unsupported version {} (current: {})",
                    self.version, FUSION_CERTIFICATE_VERSION
                ),
            });
        }
        if self.fused_kernel_name.is_empty() {
            return Err(VerifyError::InvalidCertificate {
                reason: "empty fused_kernel_name".to_string(),
            });
        }
        if !self.epsilon.is_finite() || self.epsilon < 0.0 {
            return Err(VerifyError::InvalidCertificate {
                reason: format!("invalid epsilon: {}", self.epsilon),
            });
        }
        if let Some(crown) = self.crown_bound {
            if !crown.is_finite() || crown < 0.0 {
                return Err(VerifyError::InvalidCertificate {
                    reason: format!("invalid crown_bound: {crown} (must be finite and >= 0)"),
                });
            }
        }
        if let Some(ref analytical) = self.analytical_bound {
            if !analytical.max_abs_diff.is_finite() || analytical.max_abs_diff < 0.0 {
                return Err(VerifyError::InvalidCertificate {
                    reason: format!(
                        "invalid analytical max_abs_diff: {} (must be finite and >= 0)",
                        analytical.max_abs_diff
                    ),
                });
            }
        }
        if let Some(ref hash) = self.fused_source_hash {
            if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(VerifyError::InvalidCertificate {
                    reason: format!("invalid SHA-256 hash format: {hash}"),
                });
            }
        }
        if !is_iso8601_utc(&self.generated_at) {
            return Err(VerifyError::InvalidCertificate {
                reason: format!(
                    "generated_at is not ISO 8601 UTC (YYYY-MM-DDTHH:MM:SSZ): {}",
                    self.generated_at
                ),
            });
        }
        for (i, &(lo, hi)) in self.variable_bounds.iter().enumerate() {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(VerifyError::InvalidCertificate {
                    reason: format!("non-finite variable_bounds[{i}]: ({lo}, {hi})"),
                });
            }
            if lo > hi {
                return Err(VerifyError::InvalidCertificate {
                    reason: format!("inverted variable_bounds[{i}]: {lo} > {hi}"),
                });
            }
        }
        Ok(())
    }

    /// Serialize to JSON.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if serialization fails.
    pub fn to_json(&self) -> Result<String, VerifyError> {
        serde_json::to_string_pretty(self).map_err(|e| VerifyError::InvalidCertificate {
            reason: format!("JSON serialization failed: {e}"),
        })
    }
}

/// Analytical IEEE 754 floating-point error bound for fusion equivalence.
///
/// Derived from counting the differing FP operations between fused and
/// sequential evaluation order, scaling by output magnitude and Lipschitz
/// amplification of downstream operations.
///
/// The bound is: `max_abs_diff = max_magnitude * differing_op_count * 2^-24 * lipschitz_factor`
///
/// where `2^-24 ≈ 5.96e-8` is the f32 machine epsilon (relative error per op).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AnalyticalFusionBound {
    /// Number of FP operations that differ between fused and sequential.
    pub differing_op_count: usize,
    /// Maximum intermediate value magnitude (from CROWN output bounds or analysis).
    pub max_magnitude: f64,
    /// Lipschitz amplification factor of downstream operations after the diff point.
    pub lipschitz_factor: f64,
    /// Derived maximum absolute diff.
    pub max_abs_diff: f64,
}

/// f32 machine epsilon: 2^-24.
const F32_MACHINE_EPS: f64 = 5.960_464_477_539_063e-8; // 2.0_f64.powi(-24)

impl AnalyticalFusionBound {
    /// Compute the analytical ULP error bound.
    ///
    /// # Arguments
    ///
    /// * `differing_op_count` — number of FP ops that differ between fused and sequential
    /// * `max_magnitude` — maximum intermediate value magnitude at the divergence point
    /// * `lipschitz_factor` — Lipschitz constant of operations downstream of divergence
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if any input is non-finite, or op count is zero.
    pub fn compute(
        differing_op_count: usize,
        max_magnitude: f64,
        lipschitz_factor: f64,
    ) -> Result<Self, VerifyError> {
        if differing_op_count == 0 {
            return Err(VerifyError::InvalidCertificate {
                reason: "differing_op_count must be > 0".to_string(),
            });
        }
        if !max_magnitude.is_finite() || max_magnitude < 0.0 {
            return Err(VerifyError::InvalidCertificate {
                reason: format!("invalid max_magnitude: {max_magnitude}"),
            });
        }
        if !lipschitz_factor.is_finite() || lipschitz_factor < 0.0 {
            return Err(VerifyError::InvalidCertificate {
                reason: format!("invalid lipschitz_factor: {lipschitz_factor}"),
            });
        }
        let max_abs_diff =
            max_magnitude * (differing_op_count as f64) * F32_MACHINE_EPS * lipschitz_factor;
        if !max_abs_diff.is_finite() {
            return Err(VerifyError::InvalidCertificate {
                reason: format!(
                    "computed max_abs_diff is non-finite: {max_abs_diff} \
                     (magnitude={max_magnitude}, ops={differing_op_count}, \
                     lipschitz={lipschitz_factor})"
                ),
            });
        }
        Ok(Self {
            differing_op_count,
            max_magnitude,
            lipschitz_factor,
            max_abs_diff,
        })
    }

    /// Whether this analytical bound proves the diff is within epsilon.
    pub fn proves_within_epsilon(&self, epsilon: f32) -> bool {
        self.max_abs_diff <= f64::from(epsilon)
    }
}

/// Pre-computed analytical bounds for known Kokoro fusion pairs.
///
/// Based on R1-151/R1-152 ULP analysis of fused vs sequential evaluation order.
pub mod known_bounds {
    use super::AnalyticalFusionBound;
    use crate::error::VerifyError;

    /// AdaIN+Snake: both fused and sequential use rsqrt() in the IR.
    /// Output magnitude ~64 (from CROWN bounds). Lipschitz factor ~2 (sin/cos <= 1, mul by 1/alpha ~2).
    pub fn adain_snake() -> Result<AnalyticalFusionBound, VerifyError> {
        AnalyticalFusionBound::compute(
            2,    // 2 differing compound ops (rsqrt vs sqrt.recip)
            64.0, // max output magnitude from CROWN
            2.0,  // downstream Lipschitz: sin<=1, mul by ~2
        )
    }

    /// LayerNorm+GELU: similar rsqrt divergence, smaller output magnitude.
    pub fn layer_norm_gelu() -> Result<AnalyticalFusionBound, VerifyError> {
        AnalyticalFusionBound::compute(
            2,    // ~2 differing ops
            10.0, // max output magnitude from CROWN
            1.2,  // GELU tanh-approx max derivative ~1.13 at x = sqrt(2)
        )
    }

    /// RMSNorm+SiLU+Mul: rsqrt divergence, larger output magnitude.
    pub fn rms_norm_silu_mul() -> Result<AnalyticalFusionBound, VerifyError> {
        AnalyticalFusionBound::compute(
            2,    // ~2 differing ops
            72.0, // max output magnitude from CROWN
            1.1,  // SiLU max derivative ~1.1 at x ~2.4
        )
    }

    /// AdaIN+LeakyReLU analytical fusion bound.
    ///
    /// Same rsqrt divergence as AdaIN+Snake, but LeakyReLU downstream has
    /// Lipschitz factor 1.0 (max derivative = max(1, slope) = 1 for
    /// slope < 1). Output magnitude 64.0 from CROWN (same AdaIN output).
    pub fn adain_leaky_relu() -> Result<AnalyticalFusionBound, VerifyError> {
        AnalyticalFusionBound::compute(
            2,    // 2 differing ops (rsqrt vs sqrt.recip)
            64.0, // max output magnitude from CROWN (same as adain_snake)
            1.0,  // LeakyReLU max derivative = 1.0 (positive branch)
        )
    }

    /// AdaLayerNorm: LayerNorm + adaptive affine (x * gamma + beta).
    /// Same rsqrt divergence as LayerNorm+GELU. Adaptive affine is linear,
    /// Lipschitz = max(|gamma|) ≈ 2.0 for prosody style parameters.
    pub fn ada_layer_norm() -> Result<AnalyticalFusionBound, VerifyError> {
        AnalyticalFusionBound::compute(
            2,    // 2 differing compound ops (rsqrt vs sqrt.recip)
            10.0, // max output magnitude from CROWN (same as layer_norm_gelu)
            2.0,  // adaptive affine Lipschitz: max(|gamma|) ≈ 2.0
        )
    }
}

const ELEMENTWISE_RATIONALE: &str =
    "Fusion equivalence is element-wise: the scalar proof covers all tensor dimensions \
     because fused and sequential kernels compute the same scalar function independently \
     at each element position with no cross-element interactions.";

use crate::certificate::now_iso8601;

/// Convert UNIX timestamp to ISO 8601 UTC string (`YYYY-MM-DDTHH:MM:SSZ`).
///
/// Uses the Howard Hinnant `civil_from_days` algorithm for calendar conversion.
/// Reference: <https://howardhinnant.github.io/date_algorithms.html>
///
/// Kept under `#[cfg(test)]` — production code uses [`now_iso8601`] from
/// `certificate_pipeline.rs`. Tests exercise this implementation directly.
#[cfg(any(test, kani))]
fn unix_secs_to_iso8601(secs: u64) -> String {
    let day_secs = 86_400_u64;
    let total_days = (secs / day_secs) as i64;
    let time_of_day = secs % day_secs;

    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // civil_from_days: convert days since 1970-01-01 to (year, month, day)
    let z = total_days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{y:04}-{m:02}-{d:02}T{hours:02}:{minutes:02}:{seconds:02}Z"
    )
}

/// Check if a string matches `YYYY-MM-DDTHH:MM:SSZ` ISO 8601 UTC format.
fn is_iso8601_utc(s: &str) -> bool {
    if s.len() != 20 {
        return false;
    }
    let b = s.as_bytes();
    b[4] == b'-'
        && b[7] == b'-'
        && b[10] == b'T'
        && b[13] == b':'
        && b[16] == b':'
        && b[19] == b'Z'
        && b[..4].iter().all(u8::is_ascii_digit)
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[8..10].iter().all(u8::is_ascii_digit)
        && b[11..13].iter().all(u8::is_ascii_digit)
        && b[14..16].iter().all(u8::is_ascii_digit)
        && b[17..19].iter().all(u8::is_ascii_digit)
}

#[cfg(test)]
#[path = "fusion_certificate_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "kani_fusion_certificate.rs"]
mod kani_proofs;
