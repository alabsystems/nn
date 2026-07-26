// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Floating-point precision policy for Rust↔GPU equivalence.
//!
//! # Three-Tier Precision Model
//!
//! nn compiles the same Rust kernel to both a CPU reference and GPU code
//! (currently MSL; PTX and SPIR-V are planned for Phase 4). IEEE 754 floating-point arithmetic is not
//! associative, and GPU transcendental implementations (`sin`, `cos`, `exp`)
//! differ from Rust's `libm`. The precision contract defines how much
//! divergence is tolerable between Rust reference output and GPU output.
//!
//! ## Tiers
//!
//! | Tier | Metal compile flags | Math intrinsics | Use case |
//! |------|--------------------|--------------------|----------|
//! | **Strict** | `fast_math = false` | `metal::precise::*` | Verification-critical paths, exact reproducibility |
//! | **Normal** (default) | `fast_math = false` | `metal::precise::*` | Standard inference, balanced accuracy/performance |
//! | **Relaxed** | `fast_math = true` | `metal::*` (fast variants) | Performance-critical paths where wider tolerance is acceptable |
//!
//! All tiers use `metal::precise::*` for transcendentals except **Relaxed**,
//! which allows the Metal compiler to use faster, less accurate variants.
//! Only **Relaxed** enables Metal's `-ffast-math` flag, which permits
//! operation reordering and FMA contraction.
//!
//! ## Selecting a tier
//!
//! ```text
//! #[nn::kernel]                                  // Normal (default)
//! fn relu(x: f32) -> f32 { x.max(0.0) }
//!
//! #[nn::kernel(precision = "strict")]           // Strict
//! fn verified_snake(x: f32, alpha: f32) -> f32 { /* ... */ }
//!
//! #[nn::kernel(precision = "relaxed")]          // Relaxed
//! fn fast_gelu(x: f32) -> f32 { /* ... */ }
//! ```
//!
//! ## Differential tolerance formula
//!
//! Differential tests compare Rust reference output (`r`) against GPU
//! candidate output (`c`) using a combined absolute + relative tolerance:
//!
//! ```text
//! |r - c| ≤ abs_budget + rel_budget × |r|
//! ```
//!
//! The budgets are tier- and dtype-dependent. Phase A uses conservative
//! bootstrap constants (see [`bootstrap_budget`]). Phase B (future, issue #20)
//! will replace these with per-kernel budgets derived from NY bound
//! propagation, tying test tolerances to verification proofs.
//!
//! ### Phase A bootstrap budgets
//!
//! | dtype | Strict | Normal | Relaxed |
//! |-------|--------|--------|---------|
//! | f32 | 1e-6 | 1e-5 | 1e-4 |
//! | f16 | 1e-3 | 1e-2 | 1e-1 |
//!
//! ## IEEE 754 special-value handling
//!
//! [`within_differential_budget`] handles non-finite values explicitly:
//! - Both NaN → match (same domain error on both sides)
//! - Both same infinity → match
//! - Mixed finite/non-finite or opposite infinities → mismatch
//!
//! ## Pipeline
//!
//! The precision contract flows through the full pipeline:
//!
//! 1. `#[kernel(precision = "...")]` → parsed by proc-macro into [`PrecisionTier`]
//! 2. [`PrecisionContract::bootstrap`] → creates contract with tier-appropriate budgets
//! 3. MSL codegen ([`crate::emit_msl_with_contract`]) → selects `metal::precise::*`
//!    vs `metal::*` intrinsics based on contract tier
//! 4. Metal compile options → `set_fast_math_enabled(contract.fast_math)`
//! 5. Differential tests → compare using [`within_differential_budget`]
//! 6. (Future) NY → replaces bootstrap budgets with proved per-kernel bounds

use std::collections::HashMap;

use thiserror::Error;

use crate::ir::ScalarType;

/// User-selectable precision policy for Rust↔GPU equivalence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PrecisionTier {
    Strict,
    Normal,
    Relaxed,
}

impl PrecisionTier {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Normal => "normal",
            Self::Relaxed => "relaxed",
        }
    }

    #[must_use]
    pub const fn fast_math(self) -> bool {
        matches!(self, Self::Relaxed)
    }

    #[must_use = "returns a Result that may contain an error"]
    pub fn parse(value: &str) -> Result<Self, PrecisionParseError> {
        match value {
            "strict" => Ok(Self::Strict),
            "normal" => Ok(Self::Normal),
            "relaxed" => Ok(Self::Relaxed),
            other => Err(PrecisionParseError::Unsupported(other.to_string())),
        }
    }
}

/// Shared precision budget used by differential tests.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PrecisionContract {
    pub tier: PrecisionTier,
    pub fast_math: bool,
    pub differential_abs_budget: f32,
    pub differential_rel_budget: f32,
}

impl PrecisionContract {
    #[must_use]
    pub fn bootstrap(tier: PrecisionTier, dtype: ScalarType) -> Self {
        let (abs_budget, rel_budget) = bootstrap_budget(dtype, tier);
        Self {
            tier,
            fast_math: tier.fast_math(),
            differential_abs_budget: abs_budget,
            differential_rel_budget: rel_budget,
        }
    }
}

/// Errors from parsing a [`PrecisionTier`] string.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrecisionParseError {
    /// The input string is not one of `strict`, `normal`, or `relaxed`.
    #[error("unsupported precision tier `{0}`; expected strict|normal|relaxed")]
    Unsupported(String),
}

/// Per-parameter input bound for differential tests and Kani harnesses.
///
/// Parsed from `bounds(x = "-1e4..1e4")` in the `#[kernel]` attribute.
/// Defaults to `[-1e6, 1e6]` for f32 and `[-65504.0, 65504.0]` for f16
/// when not specified.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct InputBound {
    lo: f64,
    hi: f64,
}

impl InputBound {
    /// Default bound for f32 parameters: `[-1e6, 1e6]`.
    pub const DEFAULT_F32: Self = Self { lo: -1e6, hi: 1e6 };

    /// Default bound for f16 parameters: `[-65504.0, 65504.0]` (f16 max).
    pub const DEFAULT_F16: Self = Self {
        lo: -65504.0,
        hi: 65504.0,
    };

    /// Returns the lower bound.
    #[must_use]
    pub fn lo(&self) -> f64 {
        self.lo
    }

    /// Returns the upper bound.
    #[must_use]
    pub fn hi(&self) -> f64 {
        self.hi
    }

    /// Create a new input bound.
    ///
    /// # Errors
    ///
    /// Returns `InputBoundParseError` if `lo`/`hi` are non-finite or `lo > hi`.
    #[must_use = "returns a Result that may contain an error"]
    pub fn new(lo: f64, hi: f64) -> Result<Self, InputBoundParseError> {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(InputBoundParseError::NonFinite(format!("{lo}..{hi}")));
        }
        if lo > hi {
            return Err(InputBoundParseError::Inverted { lo, hi });
        }
        Ok(Self { lo, hi })
    }

    /// Parse a bound from a range string like `"-1e4..1e4"` or `"1e-8..1e3"`.
    #[must_use = "returns a Result that may contain an error"]
    pub fn parse(s: &str) -> Result<Self, InputBoundParseError> {
        let parts: Vec<&str> = s.split("..").collect();
        if parts.len() != 2 {
            return Err(InputBoundParseError::BadFormat(s.to_string()));
        }
        let lo: f64 = parts[0]
            .trim()
            .parse()
            .map_err(|_| InputBoundParseError::BadFloat(parts[0].trim().to_string()))?;
        let hi: f64 = parts[1]
            .trim()
            .parse()
            .map_err(|_| InputBoundParseError::BadFloat(parts[1].trim().to_string()))?;
        if !lo.is_finite() || !hi.is_finite() {
            return Err(InputBoundParseError::NonFinite(s.to_string()));
        }
        if lo > hi {
            return Err(InputBoundParseError::Inverted { lo, hi });
        }
        Ok(Self { lo, hi })
    }

    /// Default bound for a given scalar type.
    ///
    /// BF16 uses F16 defaults because Apple GPUs compute bf16 as f16.
    #[must_use]
    pub fn default_for(ty: ScalarType) -> Self {
        match ty {
            ScalarType::F32 => Self::DEFAULT_F32,
            ScalarType::F16 | ScalarType::BF16 => Self::DEFAULT_F16,
        }
    }
}

/// Per-kernel input bounds: a map from parameter name to bound.
///
/// Missing entries use the default bound for that parameter's type.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InputBounds {
    bounds: HashMap<String, InputBound>,
}

impl InputBounds {
    /// Create an empty bounds map (all params use defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a bound for a named parameter.
    pub fn insert(&mut self, name: impl Into<String>, bound: InputBound) {
        self.bounds.insert(name.into(), bound);
    }

    /// Look up the bound for a parameter, falling back to the type default.
    #[must_use]
    pub fn get(&self, name: &str, ty: ScalarType) -> InputBound {
        self.bounds
            .get(name)
            .copied()
            .unwrap_or_else(|| InputBound::default_for(ty))
    }

    /// Returns `true` if no explicit bounds were declared.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }
}

/// Errors from parsing an `InputBound` range string.
#[derive(Debug, Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum InputBoundParseError {
    #[error("expected `lo..hi` format, got `{0}`")]
    BadFormat(String),
    #[error("could not parse `{0}` as a float")]
    BadFloat(String),
    #[error("bound endpoints must be finite, got `{0}`")]
    NonFinite(String),
    #[error("inverted bound: lo={lo} > hi={hi}")]
    Inverted { lo: f64, hi: f64 },
}

/// Phase-A deterministic budgets while gamma-derived artifacts are not wired in.
///
/// BF16 uses F16 budgets because Apple GPUs compute bf16 as f16 (no native
/// bf16 hardware). The additional bf16→f16 conversion rounding is within
/// the existing F16 precision budget.
#[must_use]
pub fn bootstrap_budget(dtype: ScalarType, tier: PrecisionTier) -> (f32, f32) {
    let abs = match (dtype, tier) {
        (ScalarType::F32, PrecisionTier::Strict) => 1e-6,
        (ScalarType::F32, PrecisionTier::Normal) => 1e-5,
        // f16 strict is intentionally one order tighter than f16 normal.
        // BF16 uses f16 budgets — Apple GPUs convert bf16→f16 for compute.
        (ScalarType::F16 | ScalarType::BF16, PrecisionTier::Strict) => 1e-3,
        (ScalarType::F16 | ScalarType::BF16, PrecisionTier::Normal) => 1e-2,
        (ScalarType::F32, PrecisionTier::Relaxed) => 1e-4,
        (ScalarType::F16 | ScalarType::BF16, PrecisionTier::Relaxed) => 1e-1,
    };

    // Relative budget mirrors the absolute bootstrap budget until #20 artifacts
    // provide per-kernel abs/rel values.
    (abs, abs)
}

/// Compute the allowed tolerance for a differential test sample.
///
/// Returns `abs_budget + rel_budget * |reference|`.
#[must_use]
pub fn differential_tolerance(reference: f32, contract: PrecisionContract) -> f32 {
    contract.differential_abs_budget + contract.differential_rel_budget * reference.abs()
}

/// Check whether `candidate` is within the differential budget of `reference`.
///
/// Handles IEEE 754 special cases: matching NaNs pass, matching infinities
/// pass, mixed finite/non-finite values fail.
#[must_use]
pub fn within_differential_budget(
    reference: f32,
    candidate: f32,
    contract: PrecisionContract,
) -> bool {
    // IEEE 754: if both are NaN, treat as matching (both sides produced the
    // same domain error).  If both are the same infinity, match.  Mixed
    // finite/non-finite or opposite infinities are real mismatches.
    if reference.is_nan() && candidate.is_nan() {
        return true;
    }
    if reference.is_infinite() && candidate.is_infinite() {
        // +inf == +inf or -inf == -inf
        return reference == candidate;
    }
    if !reference.is_finite() || !candidate.is_finite() {
        return false;
    }
    (reference - candidate).abs() <= differential_tolerance(reference, contract)
}

#[cfg(test)]
#[path = "precision_tests.rs"]
mod tests;
