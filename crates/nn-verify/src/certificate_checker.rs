// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Independent certificate checker for `.proof.json` files.
//!
//! Validates proof certificates without access to NY or the original
//! model. The checker verifies:
//!
//! 1. **Structural validity** — certificate passes `validate()`.
//! 2. **Layer trace consistency** — each layer's output bounds equal the next
//!    layer's input bounds (contiguous chain with no gaps).
//! 3. **Final output agreement** — the last layer's output bounds match the
//!    certificate's claimed `output_bounds`.
//! 4. **Hash verification** — weight and source hashes match files on disk.
//!
//! The checker does NOT re-derive bounds through layer computations (that
//! requires knowing layer types and weights). It verifies the *trace is
//! self-consistent*, which is sufficient to detect certificate tampering or
//! corruption without needing the verifier.

use std::path::Path;

use crate::certificate::integrity::{verify_signature, IntegrityError};
use crate::certificate::{CertificateBundle, ProofCertificate};
use crate::certificate_types::compute_file_hash;
use crate::error::VerifyError;
use crate::status::SmtProofVerdict;

// Types extracted to certificate_checker_types.rs (500-line limit).
#[path = "certificate_checker_types.rs"]
mod checker_types;
pub use checker_types::{CheckIssue, CheckResult, VacuityAssessment};

// Layer trace consistency checking extracted to certificate_checker_trace.rs.
#[path = "certificate_checker_trace.rs"]
mod trace;
use trace::{check_input_spec, check_layer_trace_consistency};

/// Default vacuity threshold: output intervals wider than this are considered
/// vacuously wide. Configurable per-model; this default is conservative.
pub const DEFAULT_VACUITY_THRESHOLD: f32 = 10.0;

// ---------------------------------------------------------------------------
// File hash cache — avoids re-hashing the same files N times per bundle
// ---------------------------------------------------------------------------

/// Pre-computed file hash results for bundle-level caching.
///
/// When checking multiple certificates in a bundle, file hashes are computed
/// once and reused. Without caching, `check_bundle` with N certificates reads
/// and hashes each file N times — O(N * file_size). With caching: O(file_size).
struct FileHashCache {
    weight: Option<Result<String, String>>,
    source: Option<Result<String, String>>,
}

impl FileHashCache {
    fn from_paths(weight_file: Option<&Path>, source_file: Option<&Path>) -> Self {
        Self {
            weight: weight_file.map(|p| compute_file_hash(p).map_err(|e| e.to_string())),
            source: source_file.map(|p| compute_file_hash(p).map_err(|e| e.to_string())),
        }
    }
}

/// Compare a certificate's expected hash against a pre-computed file hash.
fn verify_cached_hash(
    expected: &str,
    cached: &Result<String, String>,
    field: &str,
    issues: &mut Vec<CheckIssue>,
) {
    match cached {
        Ok(actual) if actual != expected => {
            let issue = match field {
                "weight_hash" => CheckIssue::WeightHashMismatch {
                    expected: expected.to_string(),
                    actual: actual.clone(),
                },
                _ => CheckIssue::SourceHashMismatch {
                    expected: expected.to_string(),
                    actual: actual.clone(),
                },
            };
            issues.push(issue);
        }
        Err(e) => {
            issues.push(CheckIssue::HashFileError {
                field: field.to_string(),
                error: e.clone(),
            });
        }
        Ok(_) => {} // Match
    }
}

// ---------------------------------------------------------------------------
// Core checking logic
// ---------------------------------------------------------------------------

/// Check a single proof certificate for self-consistency.
///
/// If `weight_file` or `source_file` is provided, the corresponding hash is
/// verified against the file on disk.
pub fn check_certificate(
    cert: &ProofCertificate,
    weight_file: Option<&Path>,
    source_file: Option<&Path>,
) -> CheckResult {
    check_certificate_with_key(cert, weight_file, source_file, None)
}

/// Check a single proof certificate, optionally verifying HMAC signature.
///
/// When `signing_key` is `Some`, the certificate's HMAC signature is verified
/// against the key. When `None`, signature checking is skipped (backward
/// compatible with pre-v4 unsigned certificates). (#3325)
pub fn check_certificate_with_key(
    cert: &ProofCertificate,
    weight_file: Option<&Path>,
    source_file: Option<&Path>,
    signing_key: Option<&[u8]>,
) -> CheckResult {
    let cache = FileHashCache::from_paths(weight_file, source_file);
    let mut result = check_certificate_core(cert, &cache);
    if let Some(key) = signing_key {
        verify_signature_into_issues(cert, key, &mut result.issues);
    }
    result
}

/// Core certificate checking with pre-computed file hashes.
fn check_certificate_core(cert: &ProofCertificate, file_hashes: &FileHashCache) -> CheckResult {
    let mut issues = Vec::new();

    // 1. Structural validation
    if let Err(e) = cert.validate() {
        issues.push(CheckIssue::StructuralError {
            message: e.to_string(),
        });
    }

    // 1b. Infeasible bounds check (#3153 F1): output_bounds.is_infeasible
    // means the proof failed and (0.0, 0.0) are sentinel values.
    if cert.output_bounds.is_infeasible {
        issues.push(CheckIssue::InfeasibleBounds);
    }

    // 1c. Input specification validation (#3153 F3): NaN, inverted, or empty
    // input bounds make the proof vacuously true (verified "for no inputs").
    check_input_spec(&cert.input_spec, &mut issues);

    // 2. Layer trace consistency
    match &cert.layer_bounds {
        Some(bounds) if !bounds.is_empty() => {
            // #3020: validate first layer's input_bounds against input_spec.
            check_first_layer_input_spec(&cert.input_spec, bounds, &mut issues);
            check_layer_trace_consistency(bounds, &mut issues);
            check_output_agreement(cert, bounds, &mut issues);
        }
        Some(_) => {
            // Empty bounds vector: same as missing bounds for checker purposes.
            issues.push(CheckIssue::NoLayerBounds);
        }
        None => {
            issues.push(CheckIssue::NoLayerBounds);
        }
    }

    // 3. Hash completeness — source_hash is expected on all certificates.
    // weight_hash is optional enrichment (v2 field): scalar kernels have no
    // weight files, and model-level verification with synthetic weights
    // legitimately has weight_hash: None. When a weight file is provided,
    // step 4 below verifies the hash matches — that's the correctness check.
    if cert.source_hash.is_none() {
        issues.push(CheckIssue::MissingHash {
            field: "source_hash".to_string(),
        });
    }

    // 4. Hash verification against pre-computed file hashes
    if let Some(ref expected) = cert.weight_hash {
        if let Some(ref cached) = file_hashes.weight {
            verify_cached_hash(expected, cached, "weight_hash", &mut issues);
        }
    }
    if let Some(ref expected) = cert.source_hash {
        if let Some(ref cached) = file_hashes.source {
            verify_cached_hash(expected, cached, "source_hash", &mut issues);
        }
    }

    // 4b. SMT proof consistency (#3095)
    check_smt_proof_consistency(cert, &mut issues);

    // 4c. Content hash integrity (#3222) — validate when present.
    check_content_hash_integrity(cert, &mut issues);

    // 5. Vacuity assessment — evaluate bound quality when layer bounds exist.
    let vacuity = match &cert.layer_bounds {
        Some(bounds) if !bounds.is_empty() => {
            // Count layers using tight bound methods (CROWN-family or analytical).
            // PropMethod::Ibp and MixedIbpCrown are not counted as tight.
            // AlphaCrown and BetaCrown are strictly tighter than Crown.
            // Analytical is closed-form (exact bounds).
            let crown_layers = bounds.iter().filter(|r| r.method.is_tight()).count();
            let total = bounds.len();
            let coverage = crown_layers as f32 / total as f32;
            let width = cert.output_width;
            let is_non_vacuous = coverage >= 0.5 && width < DEFAULT_VACUITY_THRESHOLD;
            if !is_non_vacuous {
                issues.push(CheckIssue::VacuousBounds {
                    crown_coverage: coverage,
                    output_width: width,
                });
            }
            Some(VacuityAssessment {
                crown_coverage: coverage,
                total_layers: total,
                crown_layers,
                ibp_layers: total - crown_layers,
                output_width: width,
                is_non_vacuous,
            })
        }
        _ => None,
    };

    CheckResult {
        kernel_name: cert.kernel_name.clone(),
        issues,
        vacuity,
    }
}

// ---------------------------------------------------------------------------
// SMT proof consistency (#3095)
// ---------------------------------------------------------------------------

/// Validate SMT proof artifacts are consistent with the claimed outcome.
///
/// - `smt_outcome == "Proven"` with no proof artifact → `SmtProofMissing`
/// - Proof artifact with `verdict == Invalid` → `SmtProofInvalid`
/// - All other combinations (no outcome, Unchecked, Verified, etc.) → no issue
fn check_smt_proof_consistency(cert: &ProofCertificate, issues: &mut Vec<CheckIssue>) {
    // Branch 1: claimed Proven but no proof artifact attached.
    if let Some(ref outcome) = cert.smt_outcome {
        if outcome == "Proven" && cert.smt_proof_alethe.is_none() {
            issues.push(CheckIssue::SmtProofMissing);
        }
    }

    // Branch 2: proof artifact present with Invalid verdict.
    if cert.smt_proof_alethe.is_some() {
        if cert.smt_proof_verdict == Some(SmtProofVerdict::Invalid) {
            issues.push(CheckIssue::SmtProofInvalid);
        }
    }
    // Branch 3 (Verified) and Branch 4 (no smt_outcome): no issues.
}

// ---------------------------------------------------------------------------
// Content hash integrity (#3222)
// ---------------------------------------------------------------------------

/// Validate content hash when present (v4+ certificates).
///
/// When `content_hash` is set, recomputes the hash and compares. This detects
/// both accidental corruption and intentional tampering (though HMAC signature
/// verification is needed for the latter — content hash alone does not require
/// a secret key, so a tamperer who also recomputes the hash would evade this).
fn check_content_hash_integrity(cert: &ProofCertificate, issues: &mut Vec<CheckIssue>) {
    if let Some(ref stored_hash) = cert.content_hash {
        match crate::certificate::integrity::compute_content_hash(cert) {
            Ok(computed) if computed != *stored_hash => {
                issues.push(CheckIssue::ContentHashMismatch {
                    expected: stored_hash.clone(),
                    actual: computed,
                });
            }
            Err(e) => {
                issues.push(CheckIssue::SignatureInvalid {
                    message: format!("failed to compute content hash: {e}"),
                });
            }
            Ok(_) => {} // Match — hash is valid.
        }
    }
}

// Output agreement checking extracted to certificate_checker_agreement.rs
#[path = "certificate_checker_agreement.rs"]
mod agreement;
use agreement::{check_first_layer_input_spec, check_output_agreement};

// ---------------------------------------------------------------------------
// HMAC signature verification bridge (#3325)
// ---------------------------------------------------------------------------

/// Verify HMAC signature on a certificate, pushing issues on failure.
///
/// Maps `IntegrityError` variants to `CheckIssue` variants so that keyed
/// verification integrates with the checker's issue-based result model.
fn verify_signature_into_issues(cert: &ProofCertificate, key: &[u8], issues: &mut Vec<CheckIssue>) {
    if let Err(e) = verify_signature(cert, key) {
        match e {
            IntegrityError::SignatureInvalid => {
                issues.push(CheckIssue::SignatureInvalid {
                    message: "HMAC verification failed".to_string(),
                });
            }
            IntegrityError::InvalidKeyLength => {
                issues.push(CheckIssue::SignatureKeyError {
                    message: "signing key length rejected by HMAC".to_string(),
                });
            }
            IntegrityError::MissingContentHash => {
                issues.push(CheckIssue::SignatureKeyError {
                    message: "certificate has no content_hash — cannot verify signature"
                        .to_string(),
                });
            }
            IntegrityError::MissingSignature => {
                issues.push(CheckIssue::SignatureKeyError {
                    message: "certificate has no hmac_signature field".to_string(),
                });
            }
            IntegrityError::ContentHashMismatch { expected, actual } => {
                issues.push(CheckIssue::ContentHashMismatch { expected, actual });
            }
            _ => {
                issues.push(CheckIssue::SignatureInvalid {
                    message: e.to_string(),
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bundle checking
// ---------------------------------------------------------------------------

/// Check all certificates in a bundle.
///
/// File hashes are computed once and reused across all certificates,
/// avoiding O(N) redundant file reads for N-certificate bundles.
pub fn check_bundle(
    bundle: &CertificateBundle,
    weight_file: Option<&Path>,
    source_file: Option<&Path>,
) -> Vec<CheckResult> {
    check_bundle_with_key(bundle, weight_file, source_file, None)
}

/// Check all certificates in a bundle, optionally verifying HMAC signatures.
///
/// When `signing_key` is `Some`, each certificate's HMAC signature is verified.
/// File hashes are computed once and reused across all certificates. (#3325)
pub fn check_bundle_with_key(
    bundle: &CertificateBundle,
    weight_file: Option<&Path>,
    source_file: Option<&Path>,
    signing_key: Option<&[u8]>,
) -> Vec<CheckResult> {
    let cache = FileHashCache::from_paths(weight_file, source_file);
    bundle
        .certificates
        .iter()
        .map(|cert| {
            let mut result = check_certificate_core(cert, &cache);
            if let Some(key) = signing_key {
                verify_signature_into_issues(cert, key, &mut result.issues);
            }
            result
        })
        .collect()
}

/// Load a bundle from a file and check all certificates.
pub fn check_bundle_file(
    proof_path: &Path,
    weight_file: Option<&Path>,
    source_file: Option<&Path>,
) -> Result<Vec<CheckResult>, VerifyError> {
    let bundle = CertificateBundle::load(proof_path)?;
    Ok(check_bundle(&bundle, weight_file, source_file))
}

/// Load a bundle from a file and check all certificates with keyed HMAC
/// verification. (#3325)
pub fn check_bundle_file_with_key(
    proof_path: &Path,
    weight_file: Option<&Path>,
    source_file: Option<&Path>,
    signing_key: Option<&[u8]>,
) -> Result<Vec<CheckResult>, VerifyError> {
    let bundle = CertificateBundle::load(proof_path)?;
    Ok(check_bundle_with_key(
        &bundle,
        weight_file,
        source_file,
        signing_key,
    ))
}

#[cfg(kani)]
#[path = "kani_certificate_checker.rs"]
mod kani_certificate_checker;

#[cfg(test)]
#[path = "certificate_checker_test_index.rs"]
mod test_index;
