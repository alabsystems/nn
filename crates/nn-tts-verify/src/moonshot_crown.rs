// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Moonshot property verification via per-layer CROWN.
//!
//! Bridges the moonshot property tracker ([`MoonshotStatus`]) with actual
//! CROWN bounds verification at production dimensions via [`verify_layerwise`].
//!
//! # Architecture
//!
//! Each of the 8 moonshot properties maps to one or more CROWN-verifiable
//! conditions on model output bounds. For example:
//!
//! - **Property 1 (Non-silent):** output RMS > threshold → `output_lower` not
//!   all near-zero.
//! - **Property 2 (Non-clipping):** output ∈ [-1, 1] → `output_upper ≤ 1.0` and
//!   `output_lower ≥ -1.0`.
//!
//! The [`MoonshotPropertyResult`] captures per-property CROWN evidence with the
//! exact bounds that prove (or fail to prove) the property.

use crate::moonshot::{VerificationLevel, PROPERTY_NAMES};
use crate::pipeline::PipelineCertificate;
// TimingCertificate used by test submodules via super::TimingCertificate.
#[cfg(test)]
use crate::pipeline::TimingCertificate;

/// Result of verifying a single moonshot property via CROWN bounds.
#[derive(Debug, Clone)]
pub struct MoonshotPropertyResult {
    /// Property index (0-7).
    pub property_index: usize,
    /// Property name.
    pub property_name: &'static str,
    /// Whether the property is proven by the bounds.
    pub proven: bool,
    /// Verification level achieved.
    pub level: VerificationLevel,
    /// The bound value that proves (or fails to prove) the property.
    /// For non-clipping: max of output_upper (must be ≤ 1.0).
    /// For non-silence: min absolute bound (must be > threshold).
    pub bound_value: f64,
    /// The threshold the bound must meet.
    pub threshold: f64,
    /// Whether the underlying CROWN was sound (not IBP fallback).
    pub is_sound: bool,
    /// Human-readable explanation.
    pub explanation: String,
}

/// Bundle of moonshot property results from a single CROWN verification run.
#[derive(Debug, Clone)]
pub struct MoonshotCrownBundle {
    /// Per-property results (only properties checked in this run).
    pub results: Vec<MoonshotPropertyResult>,
    /// The pipeline certificate from per-layer CROWN.
    pub pipeline_cert: PipelineCertificate,
    /// Dimension used for verification.
    pub verification_dim: usize,
    /// Whether all checked properties are proven.
    pub all_proven: bool,
}

#[path = "moonshot_crown_properties.rs"]
mod properties;
pub use properties::{
    check_intelligibility_proxy, check_memory_boundedness, check_non_clipping, check_non_silence,
    check_streaming_safety, check_temporal_boundedness,
};

#[path = "moonshot_crown_speaker.rs"]
mod speaker;
pub use speaker::{check_speaker_consistency, SpeakerConsistencyEvidence};

#[path = "moonshot_crown_attention.rs"]
mod attention;
pub use attention::{
    check_intelligibility_with_monotonicity, check_intelligibility_with_weight_evidence,
    verify_all_crown_properties_with_attention, verify_all_crown_properties_with_evidence,
};

#[path = "moonshot_crown_pipeline.rs"]
mod pipeline_verify;
#[cfg(feature = "ny")]
pub use pipeline_verify::generate_crown_constructive_proofs;
pub use pipeline_verify::{
    verify_all_crown_properties, verify_moonshot_from_stages, verify_properties_from_pipeline,
    verify_properties_from_pipeline_with_streaming, verify_properties_with_timing,
    verify_properties_with_timing_and_memory, verify_properties_with_timing_and_streaming,
};

#[path = "moonshot_crown_implementation.rs"]
mod implementation;
pub use implementation::{
    analyze_dispatch_plan, check_implementation_correctness, is_metadata_only, ay_kernel_category,
    ay_proven_kernel_names, ImplementationCorrectnessEvidence,
};

#[cfg(feature = "ny")]
#[path = "moonshot_crown_probabilistic.rs"]
mod probabilistic;
#[cfg(feature = "ny")]
pub use probabilistic::{
    check_non_clipping_distributional, check_non_clipping_probabilistic,
    check_non_silence_probabilistic, verify_properties_probabilistic,
};

#[path = "moonshot_crown_display.rs"]
mod display;

#[cfg(test)]
#[path = "moonshot_crown_tests.rs"]
mod tests;
