// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! [`VerifyRequest`] builder — composable entry point for kernel verification.
//!
//! # ny_api adapters wired here
//!
//! **Wired (opt-in, default off):** [`ny_ladder`](crate::ny_ladder).
//! [`VerifyRequest::ladder`] / [`VerifyRequest::ladder_default`] arm ny's
//! escalating verification ladder, and
//! [`VerifyRequest::verify_spec_laddered`] runs it — IBP → α-CROWN → CROWN →
//! β-CROWN, stopping at the first decisive rung — reporting which rung decided
//! in [`LadderedVerification::method_used`] plus the full per-rung trace.
//!
//! The ladder replaces nn's *fixed two-step* escalation
//! ([`verify_graph_against_spec_with_config`], IBP then one CROWN attempt) for
//! spec verification only: a ladder rung's decision is "did this method prove
//! the property", which needs a property. [`VerifyRequest::verify_bounds`]
//! computes an enclosure rather than deciding a property, so it keeps
//! [`run_escalation`] and is untouched by the ladder setting.
//!
//! `verify_spec` and `verify_bounds` behave exactly as before whether or not a
//! ladder config is set; only `verify_spec_laddered` consults it.

use nn_dsl::ir::KernelDef;
use ny_api::{Bound, BoundedTensor, VerificationResult, VerificationSpec};

use crate::error::VerifyError;
use crate::graph::{has_variable_comparison, kernel_to_graph, kernel_to_graph_multi, ParamBinding};
use crate::ny_ladder::{
    default_ladder_config, laddered_summary, verify_model_laddered_with_config,
};
use crate::verify::{
    run_escalation, single_variable_bindings, verify_graph_against_spec_with_config,
    KernelVerification, PropMethod, SpecVerification, VerifyConfig,
};
use crate::verify_input::{
    multi_scalar_input_bounds, validate_variable_bounds, verification_spec_from_tensors,
};

// Re-exported because they are part of `LadderedVerification`'s public shape:
// callers configure the ladder with `LadderConfig`, read the trace as
// `RungOutcome`s, and compare the deciding rung against `MethodUsed`.
pub use ny_api::ladder::{LadderConfig, RungOutcome};
pub use ny_core::MethodUsed;


/// Claim ny's CPU compute-engine seams once per process.
///
/// ny installs a blocked-SIMD CPU GEMM as a FLOOR under its engine-gated fast
/// paths, but only from `ny-cli`'s `main()` — so a library consumer like nn
/// ran without it. That is not merely a speed difference: where an engine
/// gates a REPRESENTATION choice, its absence changes the algorithm. The conv
/// Patches backward's batched-GEMM seam requires an installed engine, so
/// without one the composition falls back to a per-position scalar
/// `conv2d_transpose` loop — on every conv-heavy nn model, on every host.
///
/// Installing is `OnceLock`-guarded upstream and first-install-wins; nn
/// registers no accelerator on this seam, so claiming the floor here is safe.
/// It can move a bound's last ULP, never its validity (ny records
/// exact-rational oracle validation with zero violations).
fn init_ny_engines() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(ny_api::engines::init_process_engines);
}
/// Builder for kernel verification requests.
///
/// Replaces the combinatorial `verify_kernel_*` free functions with a single
/// composable API. Set optional parameters via chained setters, then call
/// [`verify_bounds`](Self::verify_bounds) or [`verify_spec`](Self::verify_spec).
#[derive(Debug)]
#[must_use = "builder does nothing until a terminal method is called"]
pub struct VerifyRequest<'a> {
    kernel: &'a KernelDef,
    config: VerifyConfig,
    bindings: Option<&'a [ParamBinding]>,
    constant_params: Option<&'a [f32]>,
    input_bounds: Option<&'a BoundedTensor>,
    variable_bounds: Option<&'a [(f32, f32)]>,
    required_output_bounds: Option<&'a [Bound]>,
    ladder: Option<LadderConfig>,
}

/// Result of a laddered spec verification run (see
/// [`VerifyRequest::verify_spec_laddered`]).
///
/// Reports which rung of ny's escalating ladder produced the verdict, plus the
/// full escalation trace, so a caller can audit exactly how much verification
/// effort the answer cost.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LadderedVerification {
    /// The decisive verification result the ladder returned. When no rung
    /// proved the property this is the *tightest* inconclusive result seen,
    /// carrying soundness provenance combined across every rung that ran.
    pub result: VerificationResult,
    /// The rung that decided — the ladder's own method tag.
    pub method_used: MethodUsed,
    /// `method_used` mapped into nn's [`PropMethod`], when it corresponds to a
    /// propagation method nn exposes. `None` for rungs outside that set (e.g.
    /// the complete MIP terminal).
    pub method: Option<PropMethod>,
    /// One entry per rung attempted, in escalation order.
    pub rungs: Vec<RungOutcome>,
    /// One-line NN-style summary from
    /// [`ny_ladder::laddered_summary`](crate::ny_ladder::laddered_summary).
    pub summary: String,
}

impl LadderedVerification {
    /// Whether the ladder proved the property.
    #[must_use]
    pub fn is_verified(&self) -> bool {
        self.result.is_verified()
    }

    /// Number of rungs the ladder actually ran (always ≥ 1: IBP runs first).
    #[must_use]
    pub fn rungs_run(&self) -> usize {
        self.rungs.len()
    }
}

impl<'a> VerifyRequest<'a> {
    /// Create a new verification request for the given kernel.
    pub fn new(kernel: &'a KernelDef) -> Self {
        Self {
            kernel,
            config: VerifyConfig::default(),
            bindings: None,
            constant_params: None,
            input_bounds: None,
            variable_bounds: None,
            required_output_bounds: None,
            ladder: None,
        }
    }

    /// Set the verification configuration (escalation threshold, soundness mode).
    #[must_use = "builder setters return the modified builder"]
    pub fn config(mut self, config: VerifyConfig) -> Self {
        self.config = config;
        self
    }

    /// Set explicit variable/constant parameter bindings for multi-variable kernels.
    #[must_use = "builder setters return the modified builder"]
    pub fn bindings(mut self, bindings: &'a [ParamBinding]) -> Self {
        self.bindings = Some(bindings);
        self
    }

    /// Set constant parameter values for single-variable kernels.
    #[must_use = "builder setters return the modified builder"]
    pub fn constant_params(mut self, params: &'a [f32]) -> Self {
        self.constant_params = Some(params);
        self
    }

    /// Set pre-built input bounds (a `BoundedTensor`).
    #[must_use = "builder setters return the modified builder"]
    pub fn input_bounds(mut self, bounds: &'a BoundedTensor) -> Self {
        self.input_bounds = Some(bounds);
        self
    }

    /// Set per-variable scalar bounds for multi-variable kernels.
    #[must_use = "builder setters return the modified builder"]
    pub fn variable_bounds(mut self, bounds: &'a [(f32, f32)]) -> Self {
        self.variable_bounds = Some(bounds);
        self
    }

    /// Set required output bounds for spec verification.
    #[must_use = "builder setters return the modified builder"]
    pub fn required_output_bounds(mut self, bounds: &'a [Bound]) -> Self {
        self.required_output_bounds = Some(bounds);
        self
    }

    /// Arm ny's escalating verification ladder with an explicit
    /// [`LadderConfig`], for use by [`verify_spec_laddered`](Self::verify_spec_laddered).
    ///
    /// Setting this changes nothing about [`verify_spec`](Self::verify_spec) or
    /// [`verify_bounds`](Self::verify_bounds) — the ladder is consulted only by
    /// `verify_spec_laddered`, which is the explicit opt-in.
    ///
    /// The escalation ceiling (`max_method`), the width threshold above which a
    /// tighter rung is worth attempting, the per-rung timeout, and the complete
    /// MIP terminal are all controlled here. Note that
    /// [`LadderConfig::escalation_width_threshold`] defaults to `1e6`, the same
    /// value as [`VerifyConfig`]'s: with the default the ladder stops at IBP
    /// whenever IBP's bounds are already narrow. Set it to `0.0` to make the
    /// ladder always climb while the property is unproven.
    #[must_use = "builder setters return the modified builder"]
    pub fn ladder(mut self, config: LadderConfig) -> Self {
        self.ladder = Some(config);
        self
    }

    /// Arm the ladder with [`ny_ladder::default_ladder_config`](crate::ny_ladder::default_ladder_config)
    /// (IBP → α-CROWN → CROWN → β-CROWN, no MIP terminal).
    #[must_use = "builder setters return the modified builder"]
    pub fn ladder_default(self) -> Self {
        self.ladder(default_ladder_config())
    }

    /// Run bounds verification, returning computed output bounds.
    ///
    /// For multi-variable kernels, set [`bindings`](Self::bindings) and
    /// [`variable_bounds`](Self::variable_bounds). For single-variable kernels,
    /// set [`constant_params`](Self::constant_params) and
    /// [`input_bounds`](Self::input_bounds).
    #[must_use = "returns a Result that may contain an error"]
    pub fn verify_bounds(self) -> Result<KernelVerification, VerifyError> {
        init_ny_engines();
        if let Some(bindings) = self.bindings {
            let vb = self.variable_bounds.ok_or_else(|| {
                VerifyError::InvalidInput("variable_bounds required with bindings".into())
            })?;
            validate_variable_bounds(bindings, vb)?;
            let ib = multi_scalar_input_bounds(vb)?;
            let uses_cmp = has_variable_comparison(self.kernel, bindings);
            let graph = kernel_to_graph_multi(self.kernel, bindings)?;
            let (verification, _) =
                run_escalation(&graph, &ib, &self.kernel.name, &self.config, uses_cmp)?;
            Ok(verification)
        } else {
            let cp = self.resolve_constant_params()?;
            let ib = self.input_bounds.ok_or_else(|| {
                VerifyError::InvalidInput("input_bounds required for single-variable path".into())
            })?;
            let bindings_vec = single_variable_bindings(cp);
            let uses_cmp = has_variable_comparison(self.kernel, &bindings_vec);
            let graph = kernel_to_graph(self.kernel, cp)?;
            let (verification, _) =
                run_escalation(&graph, ib, &self.kernel.name, &self.config, uses_cmp)?;
            Ok(verification)
        }
    }

    /// Run spec verification against required output bounds.
    ///
    /// Requires [`required_output_bounds`](Self::required_output_bounds) to be set.
    #[must_use = "returns a Result that may contain an error"]
    pub fn verify_spec(self) -> Result<SpecVerification, VerifyError> {
        init_ny_engines();
        let required = self.required_output_bounds.ok_or_else(|| {
            VerifyError::InvalidInput(
                "required_output_bounds required for spec verification".into(),
            )
        })?;
        if let Some(bindings) = self.bindings {
            let vb = self.variable_bounds.ok_or_else(|| {
                VerifyError::InvalidInput("variable_bounds required with bindings".into())
            })?;
            validate_variable_bounds(bindings, vb)?;
            let ib = multi_scalar_input_bounds(vb)?;
            let graph = kernel_to_graph_multi(self.kernel, bindings)?;
            verify_graph_against_spec_with_config(
                &graph,
                &ib,
                required,
                &self.config,
                &self.kernel.name,
            )
        } else {
            let cp = self.resolve_constant_params()?;
            let ib = self.input_bounds.ok_or_else(|| {
                VerifyError::InvalidInput("input_bounds required for single-variable path".into())
            })?;
            let graph = kernel_to_graph(self.kernel, cp)?;
            verify_graph_against_spec_with_config(
                &graph,
                ib,
                required,
                &self.config,
                &self.kernel.name,
            )
        }
    }

    /// Run spec verification through ny's **escalating ladder** instead of a
    /// single fixed method, reporting which rung decided.
    ///
    /// Where [`verify_spec`](Self::verify_spec) runs IBP and then at most one
    /// CROWN attempt, this climbs IBP → α-CROWN → CROWN → β-CROWN, stopping at
    /// the first rung that returns a decisive verdict. Escalation is
    /// demand-driven: a tighter rung is attempted only while the property is
    /// unproven *and* the previous rung's widest bound still exceeds
    /// [`LadderConfig::escalation_width_threshold`].
    ///
    /// # Soundness
    ///
    /// The ladder never reports `Verified` more strongly than the rung that
    /// proved it, and it combines [`SoundnessProvenance`](ny_core::SoundnessProvenance)
    /// across every rung that ran. Because a later rung is only *attempted* when
    /// an earlier one was inconclusive, and the returned inconclusive fallback
    /// is the tightest one observed, the laddered enclosure is never looser than
    /// the enclosure the same fixture's first rung produced — climbing can only
    /// tighten or decide.
    ///
    /// Uses the config set by [`ladder`](Self::ladder) /
    /// [`ladder_default`](Self::ladder_default); when neither was called it
    /// falls back to [`ny_ladder::default_ladder_config`](crate::ny_ladder::default_ladder_config),
    /// so calling this method is itself the opt-in.
    ///
    /// Requires [`required_output_bounds`](Self::required_output_bounds).
    ///
    /// # Errors
    ///
    /// Returns [`VerifyError::InvalidInput`] when `required_output_bounds` (or,
    /// on the multi-variable path, `variable_bounds`) is missing, and propagates
    /// translation errors plus any [`NyError`](ny_api::NyError) a rung raised.
    #[must_use = "returns a Result that may contain an error"]
    pub fn verify_spec_laddered(self) -> Result<LadderedVerification, VerifyError> {
        let cfg = self.ladder.clone().unwrap_or_else(default_ladder_config);
        let (graph, spec) = self.build_graph_and_spec()?;
        let laddered = verify_model_laddered_with_config(&graph, &spec, &cfg)?;

        Ok(LadderedVerification {
            method: PropMethod::from_method_used(&laddered.method_used),
            summary: laddered_summary(&laddered),
            method_used: laddered.method_used,
            rungs: laddered.rungs,
            result: laddered.result,
        })
    }

    /// Build the `(GraphNetwork, VerificationSpec)` pair a spec-level run needs.
    ///
    /// Shared by [`verify_spec_laddered`](Self::verify_spec_laddered) and kept
    /// structurally identical to [`verify_spec`](Self::verify_spec)'s
    /// translation so the two run against the same graph and the same spec —
    /// which is what makes their results directly comparable.
    fn build_graph_and_spec(
        &self,
    ) -> Result<(ny_propagate::GraphNetwork, VerificationSpec), VerifyError> {
        let required = self.required_output_bounds.ok_or_else(|| {
            VerifyError::InvalidInput(
                "required_output_bounds required for spec verification".into(),
            )
        })?;
        if let Some(bindings) = self.bindings {
            let vb = self.variable_bounds.ok_or_else(|| {
                VerifyError::InvalidInput("variable_bounds required with bindings".into())
            })?;
            validate_variable_bounds(bindings, vb)?;
            let ib = multi_scalar_input_bounds(vb)?;
            let graph = kernel_to_graph_multi(self.kernel, bindings)?;
            let spec = verification_spec_from_tensors(&ib, required)?;
            Ok((graph, spec))
        } else {
            let cp = self.resolve_constant_params()?;
            let ib = self.input_bounds.ok_or_else(|| {
                VerifyError::InvalidInput("input_bounds required for single-variable path".into())
            })?;
            let graph = kernel_to_graph(self.kernel, cp)?;
            let spec = verification_spec_from_tensors(ib, required)?;
            Ok((graph, spec))
        }
    }

    /// Resolve constant_params for the single-variable path.
    ///
    /// Single-param kernels don't need constants (the one param is the variable).
    /// Multi-param kernels require N-1 constant params. If `constant_params` is
    /// not set and the kernel has >1 params, return a clear builder-level error.
    fn resolve_constant_params(&self) -> Result<&[f32], VerifyError> {
        match self.constant_params {
            Some(cp) => Ok(cp),
            None => {
                if self.kernel.params.len() > 1 {
                    Err(VerifyError::InvalidInput(format!(
                        "kernel '{}' has {} parameters; use .constant_params() for \
                         single-variable mode (provide {} constants) or .bindings() \
                         for multi-variable mode",
                        self.kernel.name,
                        self.kernel.params.len(),
                        self.kernel.params.len() - 1,
                    )))
                } else {
                    Ok(&[])
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "verify_request_tests.rs"]
mod tests;
