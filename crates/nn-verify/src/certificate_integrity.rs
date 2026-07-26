// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cryptographic integrity for proof certificates.
//!
//! Provides HMAC-SHA256 signing and verification for any certificate type that
//! implements [`Signable`]: [`ProofCertificate`], [`EditCertificate`], and
//! [`FusionEquivalenceCertificate`]. A certificate's `content_hash` is a
//! SHA-256 digest of the canonical JSON (all fields except `content_hash` and
//! `hmac_signature`). The `hmac_signature` is an HMAC-SHA256 of that content
//! hash, keyed with a shared secret.
//!
//! ## Threat model
//!
//! Without a signature, anyone with write access to a `.proof.json` file can
//! modify bounds, soundness mode, or other fields while maintaining internal
//! consistency (the checker only validates self-consistency, not provenance).
//!
//! With HMAC-SHA256:
//! - `content_hash` alone detects accidental corruption (no key needed).
//! - `hmac_signature` detects intentional tampering (requires the signing key
//!   to forge).
//!
//! ## Key management
//!
//! The HMAC key is a shared secret between the build system (signer) and the
//! deployment system (verifier). Key provisioning is out of scope for this
//! module — callers pass the key as `&[u8]`. Typical sources: environment
//! variable, secrets manager, hardware security module.
//!
//! Part of #3222, #3020.

use hmac::{Hmac, KeyInit, Mac};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::certificate::{CertificateBundle, ProofCertificate};
use crate::edit_certificate::EditCertificate;
// `fusion_certificate` is an ny-gated module; this impl follows its gate so
// the crate still builds with --no-default-features.
#[cfg(feature = "ny")]
use crate::fusion_certificate::FusionEquivalenceCertificate;
use crate::signing_config::hex_decode;

type HmacSha256 = Hmac<Sha256>;

// ---------------------------------------------------------------------------
// Signable trait
// ---------------------------------------------------------------------------

/// Certificate types that support HMAC-SHA256 integrity signing.
///
/// Implementors must have `content_hash` and `hmac_signature` fields (both
/// `Option<String>`) and derive `Serialize`. The signing functions in this
/// module are generic over `Signable`, so all certificate types share the
/// same signing/verification logic.
pub trait Signable: Serialize {
    /// Read the stored content hash.
    fn content_hash(&self) -> Option<&str>;
    /// Read the stored HMAC signature.
    fn hmac_signature(&self) -> Option<&str>;
    /// Set the content hash field.
    fn set_content_hash(&mut self, hash: Option<String>);
    /// Set the HMAC signature field.
    fn set_hmac_signature(&mut self, sig: Option<String>);
}

impl Signable for ProofCertificate {
    fn content_hash(&self) -> Option<&str> {
        self.content_hash.as_deref()
    }
    fn hmac_signature(&self) -> Option<&str> {
        self.hmac_signature.as_deref()
    }
    fn set_content_hash(&mut self, hash: Option<String>) {
        self.content_hash = hash;
    }
    fn set_hmac_signature(&mut self, sig: Option<String>) {
        self.hmac_signature = sig;
    }
}

impl Signable for EditCertificate {
    fn content_hash(&self) -> Option<&str> {
        self.content_hash.as_deref()
    }
    fn hmac_signature(&self) -> Option<&str> {
        self.hmac_signature.as_deref()
    }
    fn set_content_hash(&mut self, hash: Option<String>) {
        self.content_hash = hash;
    }
    fn set_hmac_signature(&mut self, sig: Option<String>) {
        self.hmac_signature = sig;
    }
}

#[cfg(feature = "ny")]
impl Signable for FusionEquivalenceCertificate {
    fn content_hash(&self) -> Option<&str> {
        self.content_hash.as_deref()
    }
    fn hmac_signature(&self) -> Option<&str> {
        self.hmac_signature.as_deref()
    }
    fn set_content_hash(&mut self, hash: Option<String>) {
        self.content_hash = hash;
    }
    fn set_hmac_signature(&mut self, sig: Option<String>) {
        self.hmac_signature = sig;
    }
}

// ---------------------------------------------------------------------------
// Public signing/verification API
// ---------------------------------------------------------------------------

/// Compute the canonical content hash for any signable certificate.
///
/// Serializes the certificate with `content_hash` and `hmac_signature` set to
/// `None`, then returns the SHA-256 hex digest of the resulting JSON.
///
/// # Errors
///
/// Returns `IntegrityError::Serialization` if JSON serialization fails.
pub fn compute_content_hash(cert: &impl Signable) -> Result<String, IntegrityError> {
    let canonical = canonical_json(cert)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(crate::signing_config::hex_encode(&hasher.finalize()))
}

/// Sign a certificate: compute content hash and HMAC signature.
///
/// Works with any certificate type that implements [`Signable`]:
/// [`ProofCertificate`], [`EditCertificate`], [`FusionEquivalenceCertificate`].
///
/// Populates `content_hash` and `hmac_signature` on the certificate.
/// Any existing values in those fields are overwritten.
///
/// # Errors
///
/// Returns `IntegrityError::Serialization` if JSON serialization fails.
/// Returns `IntegrityError::InvalidKeyLength` if the key length is rejected
/// by HMAC (HMAC-SHA256 accepts any key length, so this is unlikely).
pub fn sign_certificate(cert: &mut impl Signable, key: &[u8]) -> Result<(), IntegrityError> {
    // Clear existing integrity fields before computing canonical form.
    cert.set_content_hash(None);
    cert.set_hmac_signature(None);
    let content_hash = compute_content_hash(cert)?;
    let signature = compute_hmac(&content_hash, key)?;
    cert.set_content_hash(Some(content_hash));
    cert.set_hmac_signature(Some(signature));
    Ok(())
}

/// Verify the content hash of a certificate (no key required).
///
/// Returns `Ok(())` if the stored `content_hash` matches the recomputed hash.
/// Returns `Err(IntegrityError::MissingContentHash)` if no hash is present.
/// Returns `Err(IntegrityError::ContentHashMismatch)` on mismatch.
pub fn verify_content_hash(cert: &impl Signable) -> Result<(), IntegrityError> {
    let stored = cert
        .content_hash()
        .ok_or(IntegrityError::MissingContentHash)?;
    let computed = compute_content_hash(cert)?;
    if stored != computed.as_str() {
        return Err(IntegrityError::ContentHashMismatch {
            expected: stored.to_string(),
            actual: computed,
        });
    }
    Ok(())
}

/// Verify the HMAC signature of a certificate.
///
/// Requires both `content_hash` and `hmac_signature` to be present.
/// First verifies the content hash, then verifies the HMAC using
/// constant-time comparison (`Mac::verify_slice`) to prevent timing
/// side-channel attacks.
///
/// # Errors
///
/// Returns `IntegrityError::MissingContentHash` or `MissingSignature` if
/// fields are absent. Returns `ContentHashMismatch` or `SignatureInvalid`
/// on verification failure.
pub fn verify_signature(cert: &impl Signable, key: &[u8]) -> Result<(), IntegrityError> {
    // First verify content hash integrity.
    verify_content_hash(cert)?;

    let content_hash = cert
        .content_hash()
        .expect("verify_content_hash guarantees content_hash is Some");
    let stored_sig = cert
        .hmac_signature()
        .ok_or(IntegrityError::MissingSignature)?;

    // Use constant-time HMAC verification (Mac::verify_slice) to prevent
    // timing side-channel attacks. Decode the stored hex signature back to
    // bytes for the comparison.
    let stored_bytes = hex_decode(stored_sig).map_err(|_| IntegrityError::SignatureInvalid)?;
    let mut mac = HmacSha256::new_from_slice(key).map_err(|_| IntegrityError::InvalidKeyLength)?;
    mac.update(content_hash.as_bytes());
    mac.verify_slice(&stored_bytes)
        .map_err(|_| IntegrityError::SignatureInvalid)?;
    Ok(())
}

/// Sign all certificates in a bundle.
///
/// Each certificate gets its own `content_hash` and `hmac_signature`.
pub fn sign_bundle(bundle: &mut CertificateBundle, key: &[u8]) -> Result<(), IntegrityError> {
    for cert in &mut bundle.certificates {
        sign_certificate(cert, key)?;
    }
    Ok(())
}

/// Verify signatures on all certificates in a bundle.
///
/// Unsigned certificates (no `content_hash` and no `hmac_signature`) are
/// silently skipped for backward compatibility with pre-v4 certificates.
/// Use [`verify_bundle_signatures_strict`] to reject unsigned certificates.
pub fn verify_bundle_signatures(
    bundle: &CertificateBundle,
    key: &[u8],
) -> Result<(), BundleIntegrityError> {
    verify_bundle_inner(bundle, key, false)
}

/// Strict bundle verification: all certificates must have signatures.
///
/// Unlike [`verify_bundle_signatures`], this rejects any certificate that
/// lacks integrity fields. This prevents signature-stripping attacks where
/// an attacker removes `content_hash` and `hmac_signature` to bypass
/// verification.
///
/// Use this in deployments where all certificates are expected to be signed.
pub fn verify_bundle_signatures_strict(
    bundle: &CertificateBundle,
    key: &[u8],
) -> Result<(), BundleIntegrityError> {
    verify_bundle_inner(bundle, key, true)
}

fn verify_bundle_inner(
    bundle: &CertificateBundle,
    key: &[u8],
    require_signatures: bool,
) -> Result<(), BundleIntegrityError> {
    for (i, cert) in bundle.certificates.iter().enumerate() {
        if cert.content_hash().is_none() && cert.hmac_signature().is_none() {
            if require_signatures {
                return Err(BundleIntegrityError {
                    certificate_index: i,
                    kernel_name: cert.kernel_name.clone(),
                    error: IntegrityError::MissingContentHash,
                });
            }
            continue; // Pre-v4 certificate — no integrity fields to check.
        }
        verify_signature(cert, key).map_err(|e| BundleIntegrityError {
            certificate_index: i,
            kernel_name: cert.kernel_name.clone(),
            error: e,
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Serialize any certificate to canonical JSON with integrity fields cleared.
///
/// Serializes field-by-field via `to_value` (no struct clone), then removes
/// `content_hash` and `hmac_signature` from the JSON map before stringifying.
/// Works with any `Serialize` type — the key removal is a no-op if those
/// fields don't exist.
fn canonical_json(cert: &impl Serialize) -> Result<String, IntegrityError> {
    let mut value = serde_json::to_value(cert).map_err(IntegrityError::Serialization)?;
    if let Some(obj) = value.as_object_mut() {
        obj.remove("content_hash");
        obj.remove("hmac_signature");
    }
    // Explicitly sort keys at all nesting levels. serde_json without
    // `preserve_order` uses BTreeMap (already sorted), but Cargo feature
    // unification could silently enable IndexMap via any transitive dep.
    // Sorting here makes the canonical form immune to Map implementation. #3297.
    sort_json_keys(&mut value);
    serde_json::to_string(&value).map_err(IntegrityError::Serialization)
}

/// Recursively sort all JSON object keys alphabetically.
///
/// Ensures `canonical_json` output is stable regardless of whether `serde_json`
/// uses `BTreeMap` (sorted, default) or `IndexMap` (insertion-ordered, with
/// `preserve_order` feature). Defense-in-depth for #3297.
fn sort_json_keys(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(String, serde_json::Value)> =
                std::mem::take(map).into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            for (_, v) in &mut entries {
                sort_json_keys(v);
            }
            *map = entries.into_iter().collect();
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                sort_json_keys(v);
            }
        }
        _ => {}
    }
}

/// Compute HMAC-SHA256 of `data` using `key`, returning a hex string.
fn compute_hmac(data: &str, key: &[u8]) -> Result<String, IntegrityError> {
    let mut mac = HmacSha256::new_from_slice(key).map_err(|_| IntegrityError::InvalidKeyLength)?;
    mac.update(data.as_bytes());
    let result = mac.finalize();
    Ok(crate::signing_config::hex_encode(&result.into_bytes()))
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors from certificate integrity operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IntegrityError {
    #[error("certificate has no content_hash field")]
    MissingContentHash,

    #[error("certificate has no hmac_signature field")]
    MissingSignature,

    #[error("content hash mismatch: stored {expected}, computed {actual}")]
    ContentHashMismatch { expected: String, actual: String },

    #[error("HMAC signature is invalid — certificate may have been tampered with")]
    SignatureInvalid,

    #[error("HMAC key length rejected")]
    InvalidKeyLength,

    #[error("JSON serialization failed: {0}")]
    Serialization(serde_json::Error),
}

/// Error from bundle-level integrity verification.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
#[error("certificate {certificate_index} ({kernel_name}): {error}")]
pub struct BundleIntegrityError {
    pub certificate_index: usize,
    pub kernel_name: String,
    pub error: IntegrityError,
}

#[cfg(test)]
#[path = "certificate_integrity_tests.rs"]
mod tests;
