// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Chorus pipeline stage enum and ordering validation.
//!
//! Defines [`ChorusPipelineStage`] — a typed representation of every
//! processing stage in the [`ChorusMasterPipeline`](super::ChorusMasterPipeline).
//! Each variant carries metadata (name, description, canonical order) and
//! the module provides validation to catch invalid stage orderings at
//! pipeline-construction time.
//!
//! Part of #4264.

use std::fmt;

// ---------------------------------------------------------------------------
// Stage enum
// ---------------------------------------------------------------------------

/// Every discrete processing stage in the chorus pipeline.
///
/// Variants are listed in canonical (default) processing order.
/// The ordering reflects signal-flow best practices:
///
/// 1. **Per-voice** stages (Alignment through Humanize) operate on
///    individual voice buffers before mixing.
/// 2. **Mix** stages (EnsembleBlend through SpatialDepth) combine
///    voices into a stereo bus.
/// 3. **Bus** stages (BusEq through Limiter) shape the mixed output.
///    Limiter is always last to prevent clipping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChorusPipelineStage {
    /// Cross-correlation time-alignment of voice onsets.
    Alignment,
    /// Allpass fractional-delay detuning (per-voice cents spread).
    Detune,
    /// Formant-preserving pitch shifting (PSOLA).
    FormantPitch,
    /// LFO-based F0 pitch modulation (vibrato).
    Vibrato,
    /// Breath insertion at detected pause regions.
    Breath,
    /// Micro-timing, amplitude envelope, and breathing humanization.
    Humanize,
    /// PSOLA ensemble blending for cohesive pitch.
    EnsembleBlend,
    /// Per-voice parametric EQ (biquad).
    PerVoiceEq,
    /// Per-voice sibilance reduction (de-esser).
    DeEss,
    /// Constant-power stereo pan-law mixing.
    StereoMix,
    /// 3D spatial positioning (early reflections, HRTF-lite).
    SpatialDepth,
    /// Bus-level parametric EQ.
    BusEq,
    /// Multiband dynamics compression.
    Dynamics,
    /// Harmonic saturation (tape/tube/console/warm).
    Saturation,
    /// Schroeder reverb (stereo, with optional early reflections).
    Reverb,
    /// Peak limiter to -0.1 dBFS (must be last).
    Limiter,
}

impl ChorusPipelineStage {
    /// Human-readable short name for logging and debug displays.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Alignment => "alignment",
            Self::Detune => "detune",
            Self::FormantPitch => "formant_pitch",
            Self::Vibrato => "vibrato",
            Self::Breath => "breath",
            Self::Humanize => "humanize",
            Self::EnsembleBlend => "ensemble_blend",
            Self::PerVoiceEq => "per_voice_eq",
            Self::DeEss => "de_ess",
            Self::StereoMix => "stereo_mix",
            Self::SpatialDepth => "spatial_depth",
            Self::BusEq => "bus_eq",
            Self::Dynamics => "dynamics",
            Self::Saturation => "saturation",
            Self::Reverb => "reverb",
            Self::Limiter => "limiter",
        }
    }

    /// One-line description of what the stage does.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Alignment => "Cross-correlation time-alignment of voice onsets",
            Self::Detune => "Allpass fractional-delay detuning (cents spread per voice)",
            Self::FormantPitch => "Formant-preserving pitch shifting via PSOLA",
            Self::Vibrato => "LFO-based F0 pitch modulation",
            Self::Breath => "Breath sound insertion at detected pause regions",
            Self::Humanize => "Micro-timing, amplitude envelope, and breathing humanization",
            Self::EnsembleBlend => "PSOLA ensemble blending for cohesive pitch",
            Self::PerVoiceEq => "Per-voice parametric biquad EQ",
            Self::DeEss => "Per-voice sibilance reduction (dynamic de-esser)",
            Self::StereoMix => "Constant-power stereo pan-law mixing",
            Self::SpatialDepth => "3D spatial positioning with early reflections",
            Self::BusEq => "Bus-level parametric EQ shaping",
            Self::Dynamics => "Multiband dynamics compression",
            Self::Saturation => "Harmonic saturation (tape/tube/console/warm)",
            Self::Reverb => "Schroeder reverb with optional early reflections",
            Self::Limiter => "Peak limiter to -0.1 dBFS (prevents clipping)",
        }
    }

    /// Canonical order index (0-based). Lower = earlier in the chain.
    #[must_use]
    pub fn order(&self) -> usize {
        match self {
            Self::Alignment => 0,
            Self::Detune => 1,
            Self::FormantPitch => 2,
            Self::Vibrato => 3,
            Self::Breath => 4,
            Self::Humanize => 5,
            Self::EnsembleBlend => 6,
            Self::PerVoiceEq => 7,
            Self::DeEss => 8,
            Self::StereoMix => 9,
            Self::SpatialDepth => 10,
            Self::BusEq => 11,
            Self::Dynamics => 12,
            Self::Saturation => 13,
            Self::Reverb => 14,
            Self::Limiter => 15,
        }
    }

    /// Whether this stage operates on individual voice buffers (per-voice).
    #[must_use]
    pub fn is_per_voice(&self) -> bool {
        self.order() <= Self::DeEss.order()
    }

    /// Whether this stage operates on the mixed stereo bus.
    #[must_use]
    pub fn is_bus(&self) -> bool {
        self.order() >= Self::BusEq.order()
    }

    /// Whether this stage is a mix/transition stage (blend + stereo + spatial).
    #[must_use]
    pub fn is_mix(&self) -> bool {
        matches!(
            self,
            Self::EnsembleBlend | Self::StereoMix | Self::SpatialDepth
        )
    }
}

impl fmt::Display for ChorusPipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// Default ordering
// ---------------------------------------------------------------------------

/// Returns the canonical (default) stage ordering for the full pipeline.
///
/// This is the recommended processing chain. Individual stages can be
/// omitted (the pipeline skips disabled stages), but the relative order
/// of enabled stages should match this sequence.
#[must_use]
pub fn default_order() -> Vec<ChorusPipelineStage> {
    vec![
        ChorusPipelineStage::Alignment,
        ChorusPipelineStage::Detune,
        ChorusPipelineStage::FormantPitch,
        ChorusPipelineStage::Vibrato,
        ChorusPipelineStage::Breath,
        ChorusPipelineStage::Humanize,
        ChorusPipelineStage::EnsembleBlend,
        ChorusPipelineStage::PerVoiceEq,
        ChorusPipelineStage::DeEss,
        ChorusPipelineStage::StereoMix,
        ChorusPipelineStage::SpatialDepth,
        ChorusPipelineStage::BusEq,
        ChorusPipelineStage::Dynamics,
        ChorusPipelineStage::Saturation,
        ChorusPipelineStage::Reverb,
        ChorusPipelineStage::Limiter,
    ]
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate a stage ordering, returning a list of constraint violations.
///
/// # Checked constraints
///
/// - **Limiter must be last** — nothing should follow the limiter.
/// - **Alignment should be first** — if present, alignment must precede
///   all other stages (reordering after alignment invalidates time offsets).
/// - **Per-voice stages before mix stages** — per-voice processing must
///   complete before voices are blended/panned.
/// - **Mix stages before bus stages** — mixing must happen before bus EQ,
///   dynamics, saturation, reverb, or limiting.
/// - **No duplicates** — each stage may appear at most once.
/// - **Monotonic canonical order** — stages should appear in non-decreasing
///   canonical order to maintain signal-flow correctness.
///
/// Returns `Ok(())` when all constraints are satisfied. Returns
/// `Err(violations)` with a human-readable message per violation.
pub fn validate_order(stages: &[ChorusPipelineStage]) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if stages.is_empty() {
        return Ok(());
    }

    // Duplicate check.
    let mut seen = std::collections::HashSet::new();
    for stage in stages {
        if !seen.insert(stage) {
            errors.push(format!("duplicate stage: {}", stage.name()));
        }
    }

    // Limiter must be last (if present).
    if let Some(pos) = stages
        .iter()
        .position(|s| *s == ChorusPipelineStage::Limiter)
    {
        if pos != stages.len() - 1 {
            errors.push(format!(
                "limiter must be the last stage (found at position {}, but pipeline has {} stages)",
                pos,
                stages.len()
            ));
        }
    }

    // Alignment must be first (if present).
    if let Some(pos) = stages
        .iter()
        .position(|s| *s == ChorusPipelineStage::Alignment)
    {
        if pos != 0 {
            errors.push(format!(
                "alignment should be the first stage (found at position {pos})"
            ));
        }
    }

    // Per-voice stages must precede mix stages.
    let last_per_voice = stages
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_per_voice() && !s.is_mix())
        .map(|(i, _)| i)
        .max();
    let first_mix = stages
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_mix())
        .map(|(i, _)| i)
        .min();
    // StereoMix is both mix and could be considered the boundary.
    let first_stereo = stages
        .iter()
        .enumerate()
        .find(|(_, s)| **s == ChorusPipelineStage::StereoMix)
        .map(|(i, _)| i);

    if let (Some(last_pv), Some(first_m)) = (last_per_voice, first_mix) {
        // EnsembleBlend is a mix stage but still operates on per-voice buffers,
        // so we only enforce per-voice < StereoMix.
        if let Some(first_s) = first_stereo {
            if last_pv > first_s {
                errors.push(format!(
                    "per-voice stage at position {last_pv} appears after stereo mix at position {first_s}"
                ));
            }
        } else if last_pv > first_m {
            errors.push(format!(
                "per-voice stage at position {last_pv} appears after mix stage at position {first_m}"
            ));
        }
    }

    // Mix stages must precede bus-only stages.
    let last_mix = stages
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_mix())
        .map(|(i, _)| i)
        .max();
    let first_bus = stages
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_bus() && !s.is_mix())
        .map(|(i, _)| i)
        .min();
    if let (Some(last_m), Some(first_b)) = (last_mix, first_bus) {
        if last_m > first_b {
            errors.push(format!(
                "mix stage at position {last_m} appears after bus stage at position {first_b}"
            ));
        }
    }

    // Monotonic canonical order check.
    let mut prev_order = 0;
    for (i, stage) in stages.iter().enumerate() {
        let ord = stage.order();
        if i > 0 && ord < prev_order {
            errors.push(format!(
                "stage '{}' (canonical order {}) appears after a stage with higher canonical order {} at position {}",
                stage.name(),
                ord,
                prev_order,
                i
            ));
        }
        prev_order = ord;
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

/// Format a pipeline stage list for debug/log output.
///
/// Produces a multi-line string showing stage index, name, and description:
///
/// ```text
/// Chorus Pipeline (6 stages):
///   [0] vibrato        — LFO-based F0 pitch modulation
///   [1] detune         — Allpass fractional-delay detuning (cents spread per voice)
///   [2] ensemble_blend — PSOLA ensemble blending for cohesive pitch
///   [3] stereo_mix     — Constant-power stereo pan-law mixing
///   [4] dynamics       — Multiband dynamics compression
///   [5] limiter        — Peak limiter to -0.1 dBFS (prevents clipping)
/// ```
#[must_use]
pub fn format_pipeline(stages: &[ChorusPipelineStage]) -> String {
    let mut out = format!("Chorus Pipeline ({} stages):\n", stages.len());
    for (i, stage) in stages.iter().enumerate() {
        out.push_str(&format!(
            "  [{i}] {name:<16} — {desc}\n",
            name = stage.name(),
            desc = stage.description(),
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_order_valid() {
        let order = default_order();
        validate_order(&order).expect("default order should be valid");
        assert_eq!(order.len(), 16);
    }

    #[test]
    fn test_default_order_is_monotonic() {
        let order = default_order();
        for (i, stage) in order.iter().enumerate() {
            assert_eq!(stage.order(), i, "stage {} has wrong order", stage.name());
        }
    }

    #[test]
    fn test_limiter_not_last_is_error() {
        let stages = vec![ChorusPipelineStage::Limiter, ChorusPipelineStage::Reverb];
        let errs = validate_order(&stages).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("limiter must be the last")));
    }

    #[test]
    fn test_alignment_not_first_is_error() {
        let stages = vec![ChorusPipelineStage::Vibrato, ChorusPipelineStage::Alignment];
        let errs = validate_order(&stages).unwrap_err();
        assert!(errs
            .iter()
            .any(|e| e.contains("alignment should be the first")));
    }

    #[test]
    fn test_duplicate_stage_is_error() {
        let stages = vec![ChorusPipelineStage::Vibrato, ChorusPipelineStage::Vibrato];
        let errs = validate_order(&stages).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("duplicate stage")));
    }

    #[test]
    fn test_subset_order_valid() {
        // A subset of stages in correct order should pass.
        let stages = vec![
            ChorusPipelineStage::Vibrato,
            ChorusPipelineStage::EnsembleBlend,
            ChorusPipelineStage::StereoMix,
            ChorusPipelineStage::Dynamics,
            ChorusPipelineStage::Limiter,
        ];
        validate_order(&stages).expect("subset in correct order should be valid");
    }

    #[test]
    fn test_empty_order_valid() {
        validate_order(&[]).expect("empty pipeline is valid");
    }

    #[test]
    fn test_stage_name_roundtrip() {
        for stage in default_order() {
            assert!(!stage.name().is_empty());
            assert!(!stage.description().is_empty());
        }
    }

    #[test]
    fn test_format_pipeline_output() {
        let stages = vec![
            ChorusPipelineStage::Vibrato,
            ChorusPipelineStage::StereoMix,
            ChorusPipelineStage::Limiter,
        ];
        let formatted = format_pipeline(&stages);
        assert!(formatted.contains("3 stages"));
        assert!(formatted.contains("vibrato"));
        assert!(formatted.contains("stereo_mix"));
        assert!(formatted.contains("limiter"));
    }

    #[test]
    fn test_is_per_voice_classification() {
        assert!(ChorusPipelineStage::Alignment.is_per_voice());
        assert!(ChorusPipelineStage::DeEss.is_per_voice());
        assert!(!ChorusPipelineStage::BusEq.is_per_voice());
        assert!(!ChorusPipelineStage::Limiter.is_per_voice());
    }

    #[test]
    fn test_is_bus_classification() {
        assert!(!ChorusPipelineStage::Vibrato.is_bus());
        assert!(ChorusPipelineStage::BusEq.is_bus());
        assert!(ChorusPipelineStage::Dynamics.is_bus());
        assert!(ChorusPipelineStage::Limiter.is_bus());
    }

    #[test]
    fn test_is_mix_classification() {
        assert!(ChorusPipelineStage::EnsembleBlend.is_mix());
        assert!(ChorusPipelineStage::StereoMix.is_mix());
        assert!(ChorusPipelineStage::SpatialDepth.is_mix());
        assert!(!ChorusPipelineStage::Vibrato.is_mix());
        assert!(!ChorusPipelineStage::Dynamics.is_mix());
    }

    #[test]
    fn test_display_trait() {
        assert_eq!(format!("{}", ChorusPipelineStage::Limiter), "limiter");
        assert_eq!(format!("{}", ChorusPipelineStage::Alignment), "alignment");
    }
}
