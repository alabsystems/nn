// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-segment F16 autocast configuration for [`CompiledKokoro`].
//!
//! [`F16AutocastConfig`] provides granular control over which Kokoro pipeline
//! segments use F16 autocast mixed precision. By default all segments are
//! enabled (backward-compatible with [`CompiledKokoro::with_autocast()`]).
//! Individual segments can be disabled for debugging or to avoid precision
//! issues in specific stages.
//!
//! # Example
//!
//! ```rust,ignore
//! use nn_metal::F16AutocastConfig;
//! use nn_core::mixed_precision::MixedPrecisionPolicy;
//!
//! // Enable autocast for all segments except regulate (pure elementwise).
//! let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default())
//!     .with_regulate(false);
//! let kokoro = CompiledKokoro::new(model)?
//!     .with_segment_autocast(config);
//! ```
//!
//! Part of #4269.

use nn_core::mixed_precision::MixedPrecisionPolicy;

/// Per-segment F16 autocast configuration.
///
/// Controls which of the 8 Kokoro pipeline segments use F16 autocast
/// mixed precision. Each field is a boolean: `true` = autocast enabled
/// for that segment, `false` = F32 (no autocast).
///
/// The `base_policy` field holds the [`MixedPrecisionPolicy`] used for
/// enabled segments (typically `apple_silicon_default()`).
///
/// Part of #4269.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F16AutocastConfig {
    /// Base mixed-precision policy applied to enabled segments.
    pub base_policy: MixedPrecisionPolicy,
    /// Segment 0: PlBert + bert_encoder.
    pub plbert: bool,
    /// Segment 1: TextEncoder.
    pub text: bool,
    /// Segment 2: ProsodyPredictor.
    pub prosody: bool,
    /// Segment 3: F0EnergyPredictor.
    pub f0: bool,
    /// Segment 4: Generator / FullDecoder.
    pub generator: bool,
    /// Segment 5: Regulate (elementwise chain).
    pub regulate: bool,
    /// Segment 5a: SineGen pre-cumsum.
    pub sinegen_pre: bool,
    /// Segment 5b: SineGen post-cumsum.
    pub sinegen_post: bool,
    /// Use fast half-precision accumulators in FusedResBlock conv kernels.
    ///
    /// When `true` AND `generator` is enabled (F16 autocast), the 24
    /// FusedResBlocks in the generator switch from float-accumulator F16
    /// (~1.36x throughput) to half-accumulator F16 (~2x throughput).
    /// Trades ~0.1-0.3 dB SNR for significant speed improvement.
    ///
    /// **Default: `false`** (safe, opt-in only). Enable after validating
    /// audio quality for your use case.
    pub use_fast_half_accumulator: bool,
}

impl F16AutocastConfig {
    /// Create a config with all segments enabled.
    ///
    /// This matches the behavior of [`CompiledKokoro::with_autocast()`] —
    /// all 8 segments use F16 autocast.
    #[must_use]
    pub fn all(base_policy: MixedPrecisionPolicy) -> Self {
        Self {
            base_policy,
            plbert: true,
            text: true,
            prosody: true,
            f0: true,
            generator: true,
            regulate: true,
            sinegen_pre: true,
            sinegen_post: true,
            use_fast_half_accumulator: false,
        }
    }

    /// Create a config with no segments enabled (all F32).
    ///
    /// Useful as a starting point when only a few segments should use F16.
    #[must_use]
    pub fn none(base_policy: MixedPrecisionPolicy) -> Self {
        Self {
            base_policy,
            plbert: false,
            text: false,
            prosody: false,
            f0: false,
            generator: false,
            regulate: false,
            sinegen_pre: false,
            sinegen_post: false,
            use_fast_half_accumulator: false,
        }
    }

    /// Create a recommended production config enabling F16 for all segments
    /// that benefit from reduced memory bandwidth.
    ///
    /// Enables autocast for:
    /// - **PlBert** (12-layer ALBERT transformer: embedding, attention, linear)
    /// - **TextEncoder** (embedding + linear + attention layers)
    /// - **ProsodyPredictor** (linear layers; LSTM stays F32 automatically)
    /// - **F0EnergyPredictor** (linear layers; LSTM stays F32 automatically)
    /// - **Generator** (35+ FusedResBlocks with Conv1d/ConvTranspose1d — heaviest segment)
    /// - **SineGen post-cumsum** (linear layer + elementwise ops)
    ///
    /// Disables autocast for:
    /// - **Regulate** (pure elementwise chain, minimal compute — F16 saves negligible bandwidth)
    /// - **SineGen pre-cumsum** (elementwise only, no weights — minimal benefit)
    ///
    /// The autocast system in [`compiled_model_builder`](crate::compiled_model_builder)
    /// automatically keeps accumulate ops (softmax, layer_norm, instance_norm,
    /// LSTM) in F32 even within F16-enabled segments, so all 6 enabled segments
    /// are numerically safe.
    ///
    /// Part of #4269.
    #[must_use]
    pub fn recommended(base_policy: MixedPrecisionPolicy) -> Self {
        Self {
            base_policy,
            plbert: true,
            text: true,
            prosody: true,
            f0: true,
            generator: true,
            regulate: false,
            sinegen_pre: false,
            sinegen_post: true,
            use_fast_half_accumulator: false,
        }
    }

    /// Create a CROWN-verified config by analyzing NY output bounds.
    ///
    /// Instead of manually deciding which segments can use F16, this method
    /// examines the CROWN-proven output bounds for each Kokoro pipeline segment
    /// and automatically enables F16 for segments where:
    /// - Output bounds are within F16 representable range (|bound| < 65504)
    /// - Output bounds are within the precision-adequate threshold (|bound| < 10000)
    /// - The segment has weight matmuls that benefit from F16 bandwidth reduction
    ///
    /// Returns the config and a detailed per-segment analysis via
    /// [`AutoPrecisionResult`](super::auto_precision::AutoPrecisionResult).
    ///
    /// # Current Results (from `nn_verify_status_kokoro.json`)
    ///
    /// - **F16**: plbert, text, prosody, generator, sinegen_post (5 segments)
    /// - **F32**: f0 (wide bounds: |max|=17683, ULP ~17.3), regulate (no weights),
    ///   sinegen_pre (no weights)
    ///
    /// Compared to [`recommended()`](Self::recommended), this disables f0
    /// because CROWN bounds reveal that F0 predictor outputs exceed the
    /// precision threshold where F16 ULP > 10.0, destroying sub-Hz pitch
    /// adjustments.
    ///
    /// Part of #4264.
    #[must_use]
    pub fn from_crown_bounds(
        segments: &[nn_tts_verify::kokoro_crown_verifier::SegmentBounds],
        base_policy: MixedPrecisionPolicy,
    ) -> super::auto_precision::AutoPrecisionResult {
        super::auto_precision::auto_precision_config(segments, base_policy)
    }

    /// Create a config enabling F16 for the generator segment only.
    ///
    /// The generator is the heaviest segment (~70% of total compute) with
    /// 35+ FusedResBlocks containing Conv1d and ConvTranspose1d layers.
    /// F16 autocast provides the largest bandwidth savings here.
    ///
    /// Use this as a conservative starting point. For maximum throughput,
    /// use [`recommended()`](Self::recommended) or [`all()`](Self::all).
    ///
    /// Part of #4269.
    #[must_use]
    pub fn generator_only(base_policy: MixedPrecisionPolicy) -> Self {
        Self::none(base_policy).with_generator(true)
    }

    /// Set PlBert autocast (builder pattern).
    #[must_use]
    pub fn with_plbert(mut self, enabled: bool) -> Self {
        self.plbert = enabled;
        self
    }

    /// Set TextEncoder autocast (builder pattern).
    #[must_use]
    pub fn with_text(mut self, enabled: bool) -> Self {
        self.text = enabled;
        self
    }

    /// Set ProsodyPredictor autocast (builder pattern).
    #[must_use]
    pub fn with_prosody(mut self, enabled: bool) -> Self {
        self.prosody = enabled;
        self
    }

    /// Set F0EnergyPredictor autocast (builder pattern).
    #[must_use]
    pub fn with_f0(mut self, enabled: bool) -> Self {
        self.f0 = enabled;
        self
    }

    /// Set Generator autocast (builder pattern).
    #[must_use]
    pub fn with_generator(mut self, enabled: bool) -> Self {
        self.generator = enabled;
        self
    }

    /// Set Regulate autocast (builder pattern).
    #[must_use]
    pub fn with_regulate(mut self, enabled: bool) -> Self {
        self.regulate = enabled;
        self
    }

    /// Set SineGen pre-cumsum autocast (builder pattern).
    #[must_use]
    pub fn with_sinegen_pre(mut self, enabled: bool) -> Self {
        self.sinegen_pre = enabled;
        self
    }

    /// Set SineGen post-cumsum autocast (builder pattern).
    #[must_use]
    pub fn with_sinegen_post(mut self, enabled: bool) -> Self {
        self.sinegen_post = enabled;
        self
    }

    /// Enable/disable fast half-precision accumulators in FusedResBlock conv
    /// kernels (builder pattern).
    ///
    /// When enabled AND the generator segment uses F16 autocast, the 24
    /// FusedResBlocks switch to half-accumulator kernels (~2x throughput vs
    /// ~1.36x for float-accumulator F16). Opt-in only; default `false`.
    #[must_use]
    pub fn with_fast_half_accumulator(mut self, enabled: bool) -> Self {
        self.use_fast_half_accumulator = enabled;
        self
    }

    /// Return the autocast policy for a named segment, or `None` if that
    /// segment is disabled.
    ///
    /// `segment_name` must be one of: `"plbert"`, `"text"`, `"prosody"`,
    /// `"f0"`, `"generator"`, `"regulate"`, `"sinegen_pre"`, `"sinegen_post"`.
    /// Unknown names return `None` (safe default: no autocast).
    #[must_use]
    pub fn policy_for_segment(&self, segment_name: &str) -> Option<&MixedPrecisionPolicy> {
        let enabled = match segment_name {
            "plbert" => self.plbert,
            "text" => self.text,
            "prosody" => self.prosody,
            "f0" => self.f0,
            "generator" => self.generator,
            "regulate" => self.regulate,
            "sinegen_pre" => self.sinegen_pre,
            "sinegen_post" => self.sinegen_post,
            _ => false,
        };
        if enabled {
            Some(&self.base_policy)
        } else {
            None
        }
    }

    /// Number of segments with autocast enabled.
    #[must_use]
    pub fn enabled_count(&self) -> usize {
        [
            self.plbert,
            self.text,
            self.prosody,
            self.f0,
            self.generator,
            self.regulate,
            self.sinegen_pre,
            self.sinegen_post,
        ]
        .iter()
        .filter(|&&b| b)
        .count()
    }

    /// Returns `true` if any segment has autocast enabled.
    #[must_use]
    pub fn any_enabled(&self) -> bool {
        self.enabled_count() > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_enables_all_segments() {
        let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default());
        assert_eq!(config.enabled_count(), 8);
        assert!(config.any_enabled());
    }

    #[test]
    fn test_none_disables_all_segments() {
        let config = F16AutocastConfig::none(MixedPrecisionPolicy::apple_silicon_default());
        assert_eq!(config.enabled_count(), 0);
        assert!(!config.any_enabled());
    }

    #[test]
    fn test_builder_toggles_individual_segments() {
        let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default())
            .with_regulate(false)
            .with_sinegen_pre(false);
        assert_eq!(config.enabled_count(), 6);
        assert!(config.policy_for_segment("plbert").is_some());
        assert!(config.policy_for_segment("regulate").is_none());
        assert!(config.policy_for_segment("sinegen_pre").is_none());
    }

    #[test]
    fn test_none_with_selective_enable() {
        let config = F16AutocastConfig::none(MixedPrecisionPolicy::apple_silicon_default())
            .with_generator(true)
            .with_f0(true);
        assert_eq!(config.enabled_count(), 2);
        assert!(config.policy_for_segment("generator").is_some());
        assert!(config.policy_for_segment("f0").is_some());
        assert!(config.policy_for_segment("plbert").is_none());
    }

    #[test]
    fn test_unknown_segment_returns_none() {
        let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default());
        assert!(config.policy_for_segment("nonexistent").is_none());
    }

    #[test]
    fn test_policy_for_segment_returns_base_policy() {
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let config = F16AutocastConfig::all(policy.clone());
        let returned = config.policy_for_segment("generator").unwrap();
        assert_eq!(returned.compute_dtype, policy.compute_dtype);
        assert_eq!(returned.weight_dtype, policy.weight_dtype);
        assert_eq!(returned.accumulate_dtype, policy.accumulate_dtype);
    }

    #[test]
    fn test_recommended_enables_compute_heavy_segments() {
        let config =
            F16AutocastConfig::recommended(MixedPrecisionPolicy::apple_silicon_default());
        // Compute-heavy segments enabled.
        assert!(config.plbert, "plbert should be enabled");
        assert!(config.text, "text should be enabled");
        assert!(config.prosody, "prosody should be enabled");
        assert!(config.f0, "f0 should be enabled");
        assert!(config.generator, "generator should be enabled");
        assert!(config.sinegen_post, "sinegen_post should be enabled");
        // Lightweight elementwise segments disabled.
        assert!(!config.regulate, "regulate should be disabled");
        assert!(!config.sinegen_pre, "sinegen_pre should be disabled");
        assert_eq!(config.enabled_count(), 6);
    }

    #[test]
    fn test_recommended_policy_for_segment() {
        let config =
            F16AutocastConfig::recommended(MixedPrecisionPolicy::apple_silicon_default());
        assert!(config.policy_for_segment("plbert").is_some());
        assert!(config.policy_for_segment("text").is_some());
        assert!(config.policy_for_segment("prosody").is_some());
        assert!(config.policy_for_segment("f0").is_some());
        assert!(config.policy_for_segment("generator").is_some());
        assert!(config.policy_for_segment("sinegen_post").is_some());
        assert!(config.policy_for_segment("regulate").is_none());
        assert!(config.policy_for_segment("sinegen_pre").is_none());
    }

    #[test]
    fn test_generator_only_enables_only_generator() {
        let config =
            F16AutocastConfig::generator_only(MixedPrecisionPolicy::apple_silicon_default());
        assert_eq!(config.enabled_count(), 1);
        assert!(config.generator, "generator should be enabled");
        assert!(!config.plbert, "plbert should be disabled");
        assert!(!config.text, "text should be disabled");
        assert!(!config.prosody, "prosody should be disabled");
        assert!(!config.f0, "f0 should be disabled");
        assert!(!config.regulate, "regulate should be disabled");
        assert!(!config.sinegen_pre, "sinegen_pre should be disabled");
        assert!(!config.sinegen_post, "sinegen_post should be disabled");
    }

    #[test]
    fn test_partialeq_identical_configs_are_equal() {
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let a = F16AutocastConfig::all(policy.clone());
        let b = F16AutocastConfig::all(policy);
        assert_eq!(a, b, "identical configs should be equal");
    }

    #[test]
    fn test_partialeq_different_configs_are_not_equal() {
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let a = F16AutocastConfig::all(policy.clone());
        let b = F16AutocastConfig::recommended(policy.clone());
        assert_ne!(a, b, "all vs recommended should differ");

        let c = F16AutocastConfig::none(policy);
        assert_ne!(a, c, "all vs none should differ");
    }

    #[test]
    fn test_partialeq_builder_toggle_changes_equality() {
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let a = F16AutocastConfig::all(policy.clone());
        let b = F16AutocastConfig::all(policy).with_regulate(false);
        assert_ne!(a, b, "toggling a segment should break equality");
    }
}
