// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, unreachable_pub, clippy::duplicated_attributes)]

//! Shared certificate types and measurement helpers for Phase 16 compose tests.
//!
//! Contains `AttentionMonotonicityCertificate` and `PhonemeStabilityCertificate`
//! structs used by both attention certificate and phoneme certificate test files.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 16.
//! Part of #1740: Adversarial Robustness of TTS — AC2 phoneme stability.

use nn_verify::{BoundedTensor, PropMethod};

// ===========================================================================
// Certificate data structures
// ===========================================================================

/// A formal verification certificate for attention monotonicity.
///
/// Contains all information needed to independently verify the claim:
/// "For any input within the specified perturbation set, the attention
///  output/scores satisfy the documented bounds."
#[derive(Debug)]
pub struct AttentionMonotonicityCertificate {
    /// Human-readable architecture description.
    pub architecture: String,
    /// Sequence length (T).
    pub seq_len: usize,
    /// Model dimension (D).
    pub d_model: usize,
    /// Perturbation budget (L∞ ε).
    pub perturbation_eps: f32,
    /// Perturbation type (uniform, PE-centered, confusion-set).
    pub perturbation_type: String,
    /// Verification method used (CROWN or IBP).
    pub method: PropMethod,
    /// Average bound width across all output elements.
    pub avg_width: f32,
    /// Maximum bound width across all output elements.
    pub max_width: f32,
    /// Number of positions with provable diagonal dominance.
    pub diagonal_dominant_positions: usize,
    /// Total positions in the sequence.
    pub total_positions: usize,
    /// Whether ALL positions are diagonally dominant (monotonicity proved).
    pub monotonicity_proved: bool,
    /// Status key used for persistence in nn_verify_status.json.
    pub status_key: String,
}

impl AttentionMonotonicityCertificate {
    pub fn emit_report(&self) {
        eprintln!("=== ATTENTION MONOTONICITY CERTIFICATE ===");
        eprintln!("Architecture:     {}", self.architecture);
        eprintln!("Dimensions:       T={}, D={}", self.seq_len, self.d_model);
        eprintln!(
            "Perturbation:     {} (ε={})",
            self.perturbation_type, self.perturbation_eps
        );
        eprintln!("Method:           {:?}", self.method);
        eprintln!(
            "Bounds:           avg_w={:.6}, max_w={:.6}",
            self.avg_width, self.max_width
        );
        eprintln!(
            "Diagonal dom:     {}/{} positions",
            self.diagonal_dominant_positions, self.total_positions
        );
        eprintln!(
            "Monotonicity:     {}",
            if self.monotonicity_proved {
                "PROVED"
            } else {
                "NOT PROVED"
            }
        );
        eprintln!("Status key:       {}", self.status_key);
        eprintln!("==========================================");
    }
}

/// A formal verification certificate for phoneme encoder stability.
#[derive(Debug)]
pub struct PhonemeStabilityCertificate {
    /// Architecture description.
    pub architecture: String,
    /// Confusion set name that defines the perturbation.
    pub confusion_set_name: String,
    /// Number of tokens in the confusion set.
    pub confusion_set_size: usize,
    /// Confusion category (voicing, vowel proximity, etc.).
    pub confusion_category: String,
    /// Verification method used.
    pub method: PropMethod,
    /// Average output bound width.
    pub avg_width: f32,
    /// Maximum output bound width.
    pub max_width: f32,
    /// Status key for persistence.
    pub status_key: String,
}

impl PhonemeStabilityCertificate {
    pub fn emit_report(&self) {
        eprintln!("=== PHONEME STABILITY CERTIFICATE ===");
        eprintln!("Architecture:     {}", self.architecture);
        eprintln!(
            "Confusion set:    {} ({} tokens)",
            self.confusion_set_name, self.confusion_set_size
        );
        eprintln!("Category:         {}", self.confusion_category);
        eprintln!("Method:           {:?}", self.method);
        eprintln!(
            "Bounds:           avg_w={:.6}, max_w={:.6}",
            self.avg_width, self.max_width
        );
        eprintln!("Status key:       {}", self.status_key);
        eprintln!("=====================================");
    }
}

// ===========================================================================
// Measurement helpers
// ===========================================================================

pub fn measure_avg_width(bounds: &BoundedTensor) -> f32 {
    let (lo, hi) = bounds.lower_upper();
    let n = lo.len() as f32;
    let total: f32 = hi.iter().zip(lo.iter()).map(|(h, l)| h - l).sum();
    total / n
}

pub fn measure_max_width(bounds: &BoundedTensor) -> f32 {
    let (lo, hi) = bounds.lower_upper();
    hi.iter()
        .zip(lo.iter())
        .map(|(h, l)| h - l)
        .fold(0.0f32, f32::max)
}

/// Count positions with provable diagonal dominance.
///
/// For each row t of the score matrix [T, T]:
///   diag dominant if lower[t,t] > max_{j≠t} upper[t,j]
pub fn count_diagonal_dominant(bounds: &BoundedTensor, seq_len: usize) -> usize {
    let (lo, hi) = bounds.lower_upper();
    let mut count = 0;
    for t in 0..seq_len {
        let diag_lo = lo[[t, t]];
        let max_offdiag_hi = (0..seq_len)
            .filter(|&j| j != t)
            .map(|j| hi[[t, j]])
            .fold(f32::NEG_INFINITY, f32::max);
        if diag_lo > max_offdiag_hi {
            count += 1;
        }
    }
    count
}
