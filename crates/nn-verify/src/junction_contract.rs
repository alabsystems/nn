// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Junction contract verification for sub-block decomposition.
//!
//! After decomposing a network into sub-blocks (see [`subblock_decompose`]),
//! each sub-block is verified independently via IBP/CROWN. The junction
//! contract ensures that adjacent sub-blocks compose soundly: sub-block k's
//! proven output bounds must be **contained within** sub-block k+1's assumed
//! input bounds.
//!
//! # Soundness
//!
//! If every junction contract holds, the end-to-end bound is the composition
//! of all sub-block bounds — without the exponential blowup of monolithic
//! propagation through normalization layers.
//!
//! # Usage
//!
//! ```rust,ignore
//! let blocks = decompose(&graph)?;
//! let proofs = verify_junctions(&block_bounds)?;
//! assert!(proofs.all_valid());
//! ```
//!
//! Part of #2218: Epic — Perfect Kokoro.
//! Part of #2597: Generator [-inf, inf] bounds.

use crate::error::VerifyError;

/// Margin applied to junction contract checks to account for
/// SMT quantization and floating-point rounding.
///
/// Same value as `SMT_QUANTIZATION_MARGIN` in `prove.rs`.
pub const JUNCTION_MARGIN: f32 = 1e-4;

/// Proven bounds for a single sub-block.
#[derive(Debug, Clone)]
pub struct SubBlockBounds {
    /// Sub-block name (matches `SubBlock::name`).
    pub name: String,
    /// Assumed input lower bounds (per-element).
    pub input_lower: Vec<f32>,
    /// Assumed input upper bounds (per-element).
    pub input_upper: Vec<f32>,
    /// Proven output lower bounds (per-element, from IBP/CROWN).
    pub output_lower: Vec<f32>,
    /// Proven output upper bounds (per-element, from IBP/CROWN).
    pub output_upper: Vec<f32>,
}

/// Result of verifying one junction between adjacent sub-blocks.
#[derive(Debug, Clone)]
pub struct JunctionProof {
    /// Name of the upstream sub-block (output side).
    pub upstream: String,
    /// Name of the downstream sub-block (input side).
    pub downstream: String,
    /// Whether all output bounds are contained in input bounds.
    pub is_valid: bool,
    /// Maximum violation across all elements (0.0 if valid).
    pub max_violation: f32,
    /// Number of elements that violate the contract.
    pub violation_count: usize,
}

/// Result of verifying all junctions in a decomposed network.
#[derive(Debug, Clone)]
pub struct JunctionVerification {
    /// Proof for each junction (len = num_sub_blocks - 1).
    pub proofs: Vec<JunctionProof>,
}

impl JunctionVerification {
    /// Whether all junctions are valid (end-to-end composition is sound).
    #[must_use]
    pub fn all_valid(&self) -> bool {
        self.proofs.iter().all(|p| p.is_valid)
    }

    /// Maximum violation across all junctions.
    #[must_use]
    pub fn max_violation(&self) -> f32 {
        self.proofs
            .iter()
            .map(|p| p.max_violation)
            .fold(0.0f32, f32::max)
    }

    /// Number of junctions with violations.
    #[must_use]
    pub fn invalid_count(&self) -> usize {
        self.proofs.iter().filter(|p| !p.is_valid).count()
    }
}

/// Verify a single junction: does the upstream's output fit within the
/// downstream's input bounds?
///
/// For each element i:
///   upstream.output_lower[i] >= downstream.input_lower[i] - JUNCTION_MARGIN
///   upstream.output_upper[i] <= downstream.input_upper[i] + JUNCTION_MARGIN
///
/// # Errors
///
/// Returns `VerifyError::InvalidInput` if bounds dimensions don't match.
pub fn verify_junction(
    upstream: &SubBlockBounds,
    downstream: &SubBlockBounds,
) -> Result<JunctionProof, VerifyError> {
    let n = upstream.output_lower.len();

    if n != upstream.output_upper.len()
        || n != downstream.input_lower.len()
        || n != downstream.input_upper.len()
    {
        return Err(VerifyError::InvalidInput(format!(
            "junction bounds dimension mismatch: upstream output has {} elements, \
             downstream input has {}/{}",
            n,
            downstream.input_lower.len(),
            downstream.input_upper.len(),
        )));
    }

    let mut max_violation = 0.0f32;
    let mut violation_count = 0;

    for i in 0..n {
        let out_lo = upstream.output_lower[i];
        let out_hi = upstream.output_upper[i];
        let in_lo = downstream.input_lower[i];
        let in_hi = downstream.input_upper[i];

        // Check finiteness — non-finite bounds are always a violation.
        if !out_lo.is_finite() || !out_hi.is_finite() || !in_lo.is_finite() || !in_hi.is_finite() {
            violation_count += 1;
            max_violation = f32::INFINITY;
            continue;
        }

        // Lower bound check: output_lower must be >= input_lower - margin
        let lo_gap = in_lo - JUNCTION_MARGIN - out_lo;
        if lo_gap > 0.0 {
            violation_count += 1;
            max_violation = max_violation.max(lo_gap);
        }

        // Upper bound check: output_upper must be <= input_upper + margin
        let hi_gap = out_hi - in_hi - JUNCTION_MARGIN;
        if hi_gap > 0.0 {
            violation_count += 1;
            max_violation = max_violation.max(hi_gap);
        }
    }

    Ok(JunctionProof {
        upstream: upstream.name.clone(),
        downstream: downstream.name.clone(),
        is_valid: violation_count == 0,
        max_violation,
        violation_count,
    })
}

/// Verify all junctions in a sequence of sub-block bounds.
///
/// Checks that for each pair (blocks[k], blocks[k+1]), block k's output
/// bounds are contained within block k+1's input bounds.
///
/// # Errors
///
/// Returns `VerifyError::InvalidInput` if fewer than 2 blocks are provided,
/// or if any junction has mismatched dimensions.
pub fn verify_junctions(blocks: &[SubBlockBounds]) -> Result<JunctionVerification, VerifyError> {
    if blocks.len() < 2 {
        return Err(VerifyError::InvalidInput(
            "junction verification requires at least 2 sub-blocks".to_string(),
        ));
    }

    let mut proofs = Vec::with_capacity(blocks.len() - 1);
    for pair in blocks.windows(2) {
        let proof = verify_junction(&pair[0], &pair[1])?;
        proofs.push(proof);
    }

    Ok(JunctionVerification { proofs })
}

#[cfg(test)]
#[path = "junction_contract_tests.rs"]
mod tests;
