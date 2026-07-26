// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deterministic synthesis certificate — hash-based regression detection.
//!
//! If synthesis is deterministic (seeded RNG), the SHA-256 hash of the PCM
//! output provides a zero-tolerance regression test. Any change to model weights,
//! code, or numerical backend that alters the output is immediately detected.

use std::fmt::Write as _;

use sha2::{Digest, Sha256};

/// Metadata for a deterministic synthesis certificate.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DeterministicMeta {
    /// Input text that was synthesized.
    pub input_text: Option<String>,
    /// Voice ID / speaker ID.
    pub voice_id: Option<String>,
    /// RNG seed used for synthesis.
    pub seed: Option<u64>,
}

/// Compute SHA-256 hash of PCM audio samples.
///
/// Hashes the raw f32 samples as little-endian bytes. Returns the hex digest.
/// The hash is deterministic for identical sample sequences regardless of
/// platform byte order (always uses LE encoding).
pub fn pcm_sha256(samples: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &s in samples {
        hasher.update(s.to_le_bytes());
    }
    hasher.finalize().iter().fold(String::new(), |mut acc, b| {
        let _ = write!(acc, "{b:02x}");
        acc
    })
}

/// A deterministic synthesis certificate entry.
///
/// Combines the PCM hash with optional metadata for full reproducibility tracking.
#[derive(Debug, Clone)]
pub struct DeterministicCert {
    /// SHA-256 hex digest of the PCM audio.
    pub pcm_hash: String,
    /// Optional metadata about the synthesis run.
    pub meta: DeterministicMeta,
}

impl DeterministicCert {
    /// Create a deterministic certificate from audio samples and metadata.
    pub fn from_audio(samples: &[f32], meta: DeterministicMeta) -> Self {
        Self {
            pcm_hash: pcm_sha256(samples),
            meta,
        }
    }

    /// Verify that a candidate audio matches the expected hash.
    pub fn verify(&self, candidate: &[f32]) -> bool {
        pcm_sha256(candidate) == self.pcm_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_audio_same_hash() {
        let audio = vec![0.1_f32, 0.2, 0.3, -0.5, 0.0];
        let h1 = pcm_sha256(&audio);
        let h2 = pcm_sha256(&audio);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_different_audio_different_hash() {
        let a = vec![0.1_f32, 0.2, 0.3];
        let b = vec![0.1_f32, 0.2, 0.4];
        assert_ne!(pcm_sha256(&a), pcm_sha256(&b));
    }

    #[test]
    fn test_empty_audio_hash() {
        let h = pcm_sha256(&[]);
        // SHA-256 of empty input is the standard empty hash.
        assert_eq!(h.len(), 64); // 256 bits = 64 hex chars.
    }

    #[test]
    fn test_deterministic_cert_verify() {
        let audio = vec![0.5_f32; 1000];
        let cert = DeterministicCert::from_audio(&audio, DeterministicMeta::default());
        assert!(cert.verify(&audio));
        assert!(!cert.verify(&vec![0.5_f32; 999]));
    }

    #[test]
    fn test_deterministic_cert_with_metadata() {
        let audio = vec![0.0_f32; 100];
        let meta = DeterministicMeta {
            input_text: Some("Hello world".to_string()),
            voice_id: Some("speaker_01".to_string()),
            seed: Some(42),
        };
        let cert = DeterministicCert::from_audio(&audio, meta);
        assert_eq!(cert.meta.seed, Some(42));
        assert!(cert.verify(&audio));
    }

    #[test]
    fn test_single_bit_change_detected() {
        let audio_a = vec![1.0_f32; 1000];
        let mut audio_b = audio_a.clone();
        // Flip a single bit in the last sample.
        let bits = audio_b[999].to_bits() ^ 1;
        audio_b[999] = f32::from_bits(bits);
        assert_ne!(pcm_sha256(&audio_a), pcm_sha256(&audio_b));
    }

    #[test]
    fn test_hash_is_hex_lowercase() {
        let h = pcm_sha256(&[1.0_f32]);
        assert!(h
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
}
