// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! [`VerifyRequest`] builder — composable entry point for kernel verification.

use ny_api::{Bound, BoundedTensor};
use nn_dsl::ir::KernelDef;

use crate::error::VerifyError;
use crate::graph::{has_variable_comparison, kernel_to_graph, kernel_to_graph_multi, ParamBinding};
use crate::verify::{
    run_escalation, single_variable_bindings, verify_graph_against_spec_with_config,
    KernelVerification, SpecVerification, VerifyConfig,
};
use crate::verify_input::{multi_scalar_input_bounds, validate_variable_bounds};

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

    /// Run bounds verification, returning computed output bounds.
    ///
    /// For multi-variable kernels, set [`bindings`](Self::bindings) and
    /// [`variable_bounds`](Self::variable_bounds). For single-variable kernels,
    /// set [`constant_params`](Self::constant_params) and
    /// [`input_bounds`](Self::input_bounds).
    #[must_use = "returns a Result that may contain an error"]
    pub fn verify_bounds(self) -> Result<KernelVerification, VerifyError> {
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
