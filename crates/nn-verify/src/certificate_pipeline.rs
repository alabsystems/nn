// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pipeline factory functions for generating proof certificates.
//!
//! Extracted from `certificate.rs` for the 500-line limit.

use crate::certificate_types::{compute_file_hash, LayerBoundRecord};
use crate::status::{InputBoundsRecord, ParamInputRecord, SmtStatusRecord};
use crate::verify_types::KernelVerification;

use super::ProofCertificate;

/// Optional enrichment data for proof certificates (v2 and v3 fields).
///
/// Pass to [`certificate_from_pipeline_enriched`] to populate enrichment fields
/// (source/weight hashes, Kani status, verifier version, SMT proofs) in one call.
/// All fields are optional — omit what you don't have.
#[derive(Debug, Default, Clone)]
pub struct CertificateEnrichment {
    /// Path to the kernel source file for SHA-256 fingerprinting.
    pub source_path: Option<std::path::PathBuf>,
    /// Path to the model weights file for SHA-256 fingerprinting.
    pub weight_path: Option<std::path::PathBuf>,
    /// Path to `kani_status.json` for Kani proof integration.
    pub kani_status_path: Option<std::path::PathBuf>,
    /// Per-layer bound traces from NY propagation.
    pub layer_bounds: Option<Vec<LayerBoundRecord>>,
    /// NY verifier version string.
    pub verifier_version: Option<String>,
    /// SMT status record with proof artifacts (v3).
    ///
    /// When provided and the record contains `proof_alethe`, the pipeline
    /// extracts the proof text and verdict into the certificate's v3 fields.
    pub smt_record: Option<SmtStatusRecord>,
}

/// Helper to generate a `ProofCertificate` from a pipeline result and input metadata.
pub fn certificate_from_pipeline(
    gamma_crown: &KernelVerification,
    variable_inputs: &[ParamInputRecord],
    constant_params: &[f32],
    smt_outcome: Option<&str>,
) -> ProofCertificate {
    certificate_from_pipeline_enriched(
        gamma_crown,
        variable_inputs,
        constant_params,
        smt_outcome,
        None,
    )
}

/// Generate a `ProofCertificate` with optional v2 enrichment.
///
/// Extends [`certificate_from_pipeline`] with support for source/weight hashes,
/// Kani proof records, layer bound traces, and verifier version. Pass `None` for
/// `enrichment` to get v1-equivalent behavior.
///
/// Hash computation and Kani status loading are best-effort: file I/O errors
/// are silently ignored (the certificate is still valid, just without those fields).
pub fn certificate_from_pipeline_enriched(
    gamma_crown: &KernelVerification,
    variable_inputs: &[ParamInputRecord],
    constant_params: &[f32],
    smt_outcome: Option<&str>,
    enrichment: Option<&CertificateEnrichment>,
) -> ProofCertificate {
    let input_spec = InputBoundsRecord {
        variable_inputs: variable_inputs.to_vec(),
        constant_params: constant_params.to_vec(),
        input_shape: if variable_inputs.is_empty() {
            None
        } else {
            Some(vec![variable_inputs.len()])
        },
        input_range: if variable_inputs.len() == 1 && variable_inputs[0].param_index == 0 {
            Some((variable_inputs[0].lower, variable_inputs[0].upper))
        } else {
            None
        },
    };
    let mut cert = ProofCertificate::from_verification(gamma_crown, input_spec);
    if let Some(outcome) = smt_outcome {
        cert = cert.with_smt_outcome(outcome);
    }

    if let Some(enrich) = enrichment {
        // Source hash — best-effort file read
        if let Some(ref path) = enrich.source_path {
            if let Ok(hash) = compute_file_hash(path) {
                cert = cert.with_source_hash(hash);
            }
        }
        // Weight hash — best-effort file read
        if let Some(ref path) = enrich.weight_path {
            if let Ok(hash) = compute_file_hash(path) {
                cert = cert.with_weight_hash(hash);
            }
        }
        // Kani status — best-effort load + lookup
        if let Some(ref path) = enrich.kani_status_path {
            if let Ok(Some(record)) =
                crate::kani_bridge::kani_record_from_file(path, &gamma_crown.kernel_name)
            {
                cert = cert.with_kani_status(record);
            }
        }
        // Layer bounds
        if let Some(ref bounds) = enrich.layer_bounds {
            cert = cert.with_layer_bounds(bounds.clone());
        }
        // Verifier version
        if let Some(ref version) = enrich.verifier_version {
            cert = cert.with_verifier_version(version.clone());
        }
        // SMT proof artifacts (v3)
        if let Some(ref smt) = enrich.smt_record {
            if let (Some(ref proof), Some(verdict)) = (&smt.proof_alethe, smt.proof_verdict) {
                cert = cert.with_smt_proof(proof.clone(), verdict);
            }
        }
    }

    cert
}

/// ISO 8601 UTC timestamp in `YYYY-MM-DDTHH:MM:SSZ` format.
///
/// Pure arithmetic from epoch seconds — no chrono/time dependency.
/// Uses Howard Hinnant's `civil_from_days` algorithm for leap-year correctness.
pub(crate) fn now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    epoch_secs_to_iso8601(secs)
}

/// Convert Unix epoch seconds to `YYYY-MM-DDTHH:MM:SSZ`.
fn epoch_secs_to_iso8601(epoch: u64) -> String {
    let secs_of_day = epoch % 86400;
    let hour = secs_of_day / 3600;
    let min = (secs_of_day % 3600) / 60;
    let sec = secs_of_day % 60;

    // Howard Hinnant's civil_from_days (public domain).
    // Input: days since 1970-01-01. Output: (year, month 1-12, day 1-31).
    let z = (epoch / 86400) + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

#[cfg(test)]
mod timestamp_tests {
    use super::*;

    #[test]
    fn test_epoch_to_iso8601_unix_epoch() {
        assert_eq!(epoch_secs_to_iso8601(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn test_epoch_to_iso8601_known_date() {
        // 2026-03-20 00:00:00 UTC = 1773964800
        assert_eq!(epoch_secs_to_iso8601(1773964800), "2026-03-20T00:00:00Z");
    }

    #[test]
    fn test_epoch_to_iso8601_leap_year() {
        // 2024-02-29 12:00:00 UTC = 1709208000
        assert_eq!(epoch_secs_to_iso8601(1709208000), "2024-02-29T12:00:00Z");
    }

    #[test]
    fn test_now_iso8601_format() {
        let ts = now_iso8601();
        assert!(ts.ends_with('Z'), "timestamp should end with Z: {ts}");
        assert_eq!(ts.len(), 20, "ISO 8601 should be 20 chars: {ts}");
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], "T");
        assert_eq!(&ts[13..14], ":");
        assert_eq!(&ts[16..17], ":");
    }
}
