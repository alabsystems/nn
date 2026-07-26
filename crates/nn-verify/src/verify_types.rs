// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verification result types and configuration.
//!
//! Pure data types used by the verification pipeline. Extracted from `verify.rs`
//! to provide a clear home for new result/config types without growing the core
//! verification logic file.

#[cfg(feature = "ny")]
use ny_api::BoundedTensor;
#[cfg(feature = "ny")]
use ny_core::MethodUsed;
#[cfg(feature = "ny")]
use ny_propagate::layers::LayerNormCrownMode;

use crate::error::VerifyError;
use crate::soundness_compat::{default_soundness_mode, VerificationSoundnessMode};
#[cfg(feature = "ny")]
use crate::util::finite_or;

/// Controls how normalization layers (InstanceNorm, RmsNorm, LayerNorm, AdaIN)
/// are configured for bound propagation.
///
/// **Default: [`ForwardMode`](Self::ForwardMode)** — uses input midpoint for
/// mean/variance statistics. Tighter than Conservative for isolated normalization
/// layers (~50x vs ~1e10x), but may produce vacuously wide bounds (~1e10) in
/// deep chains with contractive Conv weights (e.g., Kokoro 58-layer pipeline).
///
/// **[`Conservative`](Self::Conservative)** — standard IBP through normalization.
/// Sound and non-vacuous through deep contractive chains (width ~7.75 at N=10).
/// **Prefer Conservative for Kokoro normalization verification.**
///
/// **CROWN through norms is VACUOUS (R10#435):** CROWN linearization
/// through normalization layers produces vacuously wide bounds (~2e10 at
/// N=10 chained norms) because mean-subtraction and variance-division
/// destroy cross-layer correlations. IBP is 276M× tighter (~7.75).
/// All norms are block boundaries in trace_to_graph.rs; IBP within
/// blocks is provably tighter for all norm-containing architectures.
///
/// See dvoice #744 for context, nn #2701/#2715/#2948 for analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NormBoundsMode {
    /// Standard IBP — sound and non-vacuous through deep contractive chains.
    /// Produces tight bounds (width ~7.75 at N=10) when interleaved with
    /// contractive Conv layers. May be wider than ForwardMode for isolated
    /// normalization layers.
    /// `forward_mode: false`, `crown_mode: IbpValidated`.
    Conservative,
    /// Forward-mode IBP — uses input midpoint for mean/variance statistics.
    /// Tighter than Conservative for isolated normalization layers (~50x vs
    /// ~1e10x). However, in deep chains with contractive Conv weights, may
    /// hit FALLBACK_BOUND (~1e10) producing vacuously wide bounds. Sound for
    /// small perturbations; may not be perfectly sound for large perturbation radii.
    /// `forward_mode: true`, `crown_mode: IbpValidated`.
    ForwardMode,
    /// Forward-mode IBP + CROWN sampling linearization — tightest bounds.
    /// Enables CROWN to propagate through normalization layers via sampling-based
    /// linearization. Classified as `Heuristic` soundness.
    /// `forward_mode: true`, `crown_mode: Sampling`.
    CrownSampling,
}

impl NormBoundsMode {
    /// Whether this mode enables forward-mode IBP through normalization layers.
    #[must_use]
    pub fn forward_mode(self) -> bool {
        matches!(self, Self::ForwardMode | Self::CrownSampling)
    }

    /// The CROWN linearization mode for normalization layers.
    ///
    /// `Conservative` and `ForwardMode` use `IbpValidated` — Jacobian-based
    /// CROWN linearization with IBP-validated error margins. This is provably
    /// sound (the IBP bounds guarantee the linearization covers all outputs).
    /// `CrownSampling` uses heuristic sampling-based linearization.
    #[cfg(feature = "ny")]
    #[must_use]
    pub fn crown_mode(self) -> LayerNormCrownMode {
        match self {
            Self::Conservative | Self::ForwardMode => LayerNormCrownMode::IbpValidated,
            Self::CrownSampling => LayerNormCrownMode::Sampling,
        }
    }
}

/// Method used for bound propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "UPPERCASE")]
pub enum PropMethod {
    /// Interval Bound Propagation — fast, sound, may be loose.
    Ibp,
    /// CROWN linear relaxation — slower, tighter bounds.
    Crown,
    /// α-CROWN: optimized linear relaxation with learnable slopes.
    /// Tighter than CROWN for normalization layers (InstanceNorm, AdaIN).
    #[serde(alias = "ALPHACROWN")]
    AlphaCrown,
    /// β-CROWN: branch-and-bound with CROWN bounds.
    /// Most precise but most expensive. Uses α-CROWN for propagation
    /// with branching decisions on ReLU/activation splits.
    #[serde(alias = "BETACROWN")]
    BetaCrown,
    /// Analytical: closed-form bounds derived from mathematical analysis
    /// of the operation (e.g., linear transforms, known output ranges).
    Analytical,
    /// Mixed-mode: IBP for intractable layers, CROWN for tractable ones.
    /// Used when full CROWN is infeasible (e.g., D=512 Generator Stage 0).
    #[serde(rename = "mixed_IBP_CROWN")]
    MixedIbpCrown,
}

impl PropMethod {
    /// Whether this method produces tight (CROWN-quality) bounds.
    ///
    /// Returns `true` for Crown, AlphaCrown, BetaCrown, and Analytical.
    /// AlphaCrown/BetaCrown are strictly tighter than base CROWN;
    /// Analytical is exact (closed-form). IBP and MixedIbpCrown are loose.
    ///
    /// Use this instead of `== PropMethod::Crown` to avoid missing
    /// CROWN-family variants (#3344).
    #[must_use]
    pub fn is_tight(self) -> bool {
        matches!(
            self,
            Self::Crown | Self::AlphaCrown | Self::BetaCrown | Self::Analytical
        )
    }

    /// Convert NY's recorded verifier method tag into nn's public
    /// propagation method when the tag corresponds to a bound-propagation
    /// method we expose.
    #[cfg(feature = "ny")]
    #[must_use]
    pub(crate) fn from_method_used(method: &MethodUsed) -> Option<Self> {
        match method {
            MethodUsed::Ibp => Some(Self::Ibp),
            MethodUsed::Crown => Some(Self::Crown),
            MethodUsed::AlphaCrown => Some(Self::AlphaCrown),
            MethodUsed::BetaCrown => Some(Self::BetaCrown),
            _ => None,
        }
    }
}

/// Result of kernel bounds verification.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[must_use]
pub struct KernelVerification {
    /// Name of the kernel that was verified.
    pub kernel_name: String,
    /// Propagation method that produced these bounds.
    pub method: PropMethod,
    /// Lower bound on the kernel output (scalar summary: global min of lower bounds).
    pub output_lower: f32,
    /// Upper bound on the kernel output (scalar summary: global max of upper bounds).
    pub output_upper: f32,
    /// Width of the output interval (upper - lower).
    pub output_width: f32,
    /// Whether the output is provably finite (bounded).
    pub is_finite: bool,
    /// If CROWN was attempted but failed, the error reason is captured here.
    /// `None` means either CROWN was not attempted or CROWN succeeded.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub crown_fallback_reason: Option<String>,
    /// Soundness classification from NY provenance tracking.
    /// `Sound` means no heuristics were used; `Heuristic` means at least one
    /// approximation that weakens proof semantics was applied.
    #[serde(default = "default_soundness_mode")]
    pub soundness_mode: VerificationSoundnessMode,
    /// Full per-element output bounds from NY propagation.
    ///
    /// `None` for legacy scalar-only results; `Some` when tensor-level
    /// verification preserves the output shape. The scalar fields
    /// (`output_lower`/`output_upper`/`output_width`) remain as a global
    /// min/max summary for backward compatibility.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub output_tensor: Option<OutputTensorBounds>,
}

impl KernelVerification {
    /// Construct a new `KernelVerification` result.
    ///
    /// Required because the struct is `#[non_exhaustive]` — external crates
    /// cannot use struct literal syntax.
    #[must_use = "returns a new KernelVerification instance"]
    pub fn new(
        kernel_name: String,
        method: PropMethod,
        output_lower: f32,
        output_upper: f32,
        output_width: f32,
        is_finite: bool,
    ) -> Self {
        Self {
            kernel_name,
            method,
            output_lower,
            output_upper,
            output_width,
            is_finite,
            crown_fallback_reason: None,
            soundness_mode: VerificationSoundnessMode::Sound,
            output_tensor: None,
        }
    }

    /// Set the CROWN fallback reason.
    #[must_use = "returns the modified KernelVerification"]
    pub fn with_crown_fallback_reason(mut self, reason: Option<String>) -> Self {
        self.crown_fallback_reason = reason;
        self
    }

    /// Set the soundness mode.
    #[must_use = "returns the modified KernelVerification"]
    pub fn with_soundness_mode(mut self, mode: VerificationSoundnessMode) -> Self {
        self.soundness_mode = mode;
        self
    }
}

/// Full per-element output bounds from NY, stored as flat vectors
/// with shape metadata.
///
/// This is the serializable, `PartialEq`-implementing representation of the
/// NY `BoundedTensor` output. Use [`OutputTensorBounds::from_bounded_tensor`]
/// to construct from a `BoundedTensor`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct OutputTensorBounds {
    /// Per-element lower bounds (flattened in row-major order).
    pub lower: Vec<f32>,
    /// Per-element upper bounds (flattened in row-major order).
    pub upper: Vec<f32>,
    /// Shape of the output tensor.
    pub shape: Vec<usize>,
    /// Per-element finiteness: `true` where both lower and upper bounds were
    /// finite in the original `BoundedTensor`, `false` where at least one was
    /// replaced by `finite_or`. Allows consumers to distinguish "verified bound
    /// is 0.0" from "verification produced non-finite for this element."
    ///
    /// Empty when constructed directly (bypassing `from_bounded_tensor`) or
    /// deserialized from old JSON files without this field.
    #[serde(default)]
    pub finite_mask: Vec<bool>,
}

impl OutputTensorBounds {
    /// Construct output tensor bounds from pre-computed data.
    ///
    /// `finite_mask` marks which elements have finite bounds. Pass an empty
    /// vec if finiteness tracking is not needed.
    #[must_use]
    pub fn new(lower: Vec<f32>, upper: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            lower,
            upper,
            shape,
            finite_mask: Vec::new(),
        }
    }
}

#[cfg(feature = "ny")]
impl OutputTensorBounds {
    /// Construct from a NY `BoundedTensor`.
    ///
    /// Non-finite values (NaN, Infinity) are replaced with `0.0` so that the
    /// struct is always safe to serialize with `serde_json`.
    #[must_use]
    pub fn from_bounded_tensor(tensor: &BoundedTensor) -> Self {
        let (lower, upper) = tensor.lower_upper();
        let finite_mask: Vec<bool> = lower
            .iter()
            .zip(upper.iter())
            .map(|(&lo, &hi)| lo.is_finite() && hi.is_finite())
            .collect();
        Self {
            lower: lower.iter().map(|&v| finite_or(v, 0.0)).collect(),
            upper: upper.iter().map(|&v| finite_or(v, 0.0)).collect(),
            shape: tensor.shape().to_vec(),
            finite_mask,
        }
    }
}

/// Result of spec-based verification with propagation provenance.
///
/// Unlike raw `VerificationResult`, this wrapper tracks whether CROWN was
/// attempted and, on fallback, preserves the failure reason. This closes the
/// provenance gap where callers previously could not distinguish "CROWN was
/// never attempted" from "CROWN failed and we fell back to IBP."
#[cfg(feature = "ny")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[must_use]
pub struct SpecVerification {
    /// The underlying verification result (Verified / Violated / Unknown / Timeout).
    pub result: ny_api::VerificationResult,
    /// Propagation method that produced this result.
    pub method: PropMethod,
    /// If CROWN was attempted but failed, the error reason is captured here.
    /// `None` means either CROWN was not attempted or CROWN succeeded.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub crown_fallback_reason: Option<String>,
}

/// Default maximum output interval width before escalating from IBP to CROWN.
pub(crate) const DEFAULT_ESCALATION_THRESHOLD: f32 = 1e6;

/// Configuration for the verification pipeline.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VerifyConfig {
    escalation_threshold: f32,
    require_sound: bool,
    norm_mode: NormBoundsMode,
    collect_layer_bounds: bool,
}

impl VerifyConfig {
    /// Create a config with a custom escalation threshold.
    ///
    /// # Errors
    ///
    /// Returns [`VerifyError::InvalidThreshold`] if the threshold is negative,
    /// NaN, or infinite.
    #[must_use = "returns a Result that may contain an error"]
    pub fn with_threshold(escalation_threshold: f32) -> Result<Self, VerifyError> {
        if !escalation_threshold.is_finite() || escalation_threshold < 0.0 {
            return Err(VerifyError::InvalidThreshold {
                value: escalation_threshold,
            });
        }
        Ok(Self {
            escalation_threshold,
            require_sound: false,
            norm_mode: NormBoundsMode::ForwardMode,
            collect_layer_bounds: false,
        })
    }

    /// Set strict soundness checking (fails on heuristic approximations).
    #[must_use = "returns the modified config"]
    pub fn with_require_sound(mut self, require_sound: bool) -> Self {
        self.require_sound = require_sound;
        self
    }

    /// Set normalization layer bounds mode. Controls `forward_mode` and
    /// `crown_mode` on InstanceNorm, RmsNorm, LayerNorm, and AdaIN layers.
    ///
    /// Default: [`NormBoundsMode::ForwardMode`]. Use
    /// [`NormBoundsMode::Conservative`] for deep chains with contractive
    /// Conv weights (produces tighter bounds in that regime).
    #[must_use = "returns the modified config"]
    pub fn with_norm_mode(mut self, norm_mode: NormBoundsMode) -> Self {
        self.norm_mode = norm_mode;
        self
    }

    /// Enable per-layer bound trace collection for proof certificates (#802 AC3).
    ///
    /// When enabled, `TensorPipelineResult::layer_bounds` is populated with
    /// per-node bounds from CROWN-IBP propagation. Adds one extra NY
    /// pass (`collect_crown_ibp_bounds_dag_with_status`) after the main
    /// verification. Default: `false`.
    #[must_use = "returns the modified config"]
    pub fn with_collect_layer_bounds(mut self, collect: bool) -> Self {
        self.collect_layer_bounds = collect;
        self
    }

    /// Returns the escalation threshold.
    #[must_use]
    pub fn escalation_threshold(&self) -> f32 {
        self.escalation_threshold
    }
    /// Returns whether strict soundness is required.
    #[must_use]
    pub fn require_sound(&self) -> bool {
        self.require_sound
    }
    /// Returns the normalization layer bounds mode.
    #[must_use]
    pub fn norm_mode(&self) -> NormBoundsMode {
        self.norm_mode
    }
    /// Returns whether per-layer bound trace collection is enabled.
    #[must_use]
    pub fn collect_layer_bounds(&self) -> bool {
        self.collect_layer_bounds
    }
}

impl Default for VerifyConfig {
    fn default() -> Self {
        Self {
            escalation_threshold: DEFAULT_ESCALATION_THRESHOLD,
            require_sound: false,
            norm_mode: NormBoundsMode::ForwardMode,
            collect_layer_bounds: false,
        }
    }
}

#[cfg(all(test, feature = "ny"))]
#[path = "verify_types_tests.rs"]
mod tests;
