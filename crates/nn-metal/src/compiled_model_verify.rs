// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verification and certification constructors for [`CompiledModel`].
//!
//! Feature-gated behind `verify` — requires `nn-verify` as a dependency.
//! These are convenience wrappers around `builder().build()` + `nn_verify` APIs.
//!
//! Part of #3042 (Verified by default).

use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_core::{Result, TensorError};

use super::error::CompiledModelError;
use super::CompiledModel;
use crate::cache::PipelineCache;

impl CompiledModel {
    /// Best-effort auto-verification: translate graph, propagate bounds, serialize certificate.
    ///
    /// Returns `None` on any failure (untranslatable graph, verification error,
    /// serialization error). The compilation path is never affected.
    ///
    /// Part of #3042 (Verified by default).
    pub(crate) fn try_auto_verify(_graph: &ComputationGraph) -> Option<String> {
        // auto_verify module not yet committed — stub returns None (best-effort path)
        None
    }

    /// Compile a traced graph and auto-verify it via IBP bound propagation.
    ///
    /// Convenience wrapper that calls [`builder().build()`](Self::builder) then
    /// [`nn_verify::verify_trace`] on the same graph. Returns both the
    /// compiled model and the verification result.
    ///
    /// Requires the `verify` feature: `nn-metal = { ..., features = ["verify"] }`.
    ///
    /// # Errors
    ///
    /// Returns [`CompiledModelError::CompileFailed`] if trace compilation fails.
    /// Returns [`CompiledModelError::VerifyFailed`] if IBP propagation fails.
    ///
    /// Part of #3042.
    pub fn from_trace_verified(
        graph: &ComputationGraph,
        cache: &PipelineCache,
        bounds: &nn_verify::BoundedTensor,
    ) -> Result<(Self, nn_verify::VerifyTraceResult)> {
        let model = Self::builder(graph, cache).build()?;
        let result = nn_verify::verify_trace(graph, bounds)
            .map_err(|e| TensorError::from(CompiledModelError::VerifyFailed(e)))?;
        Ok((model, result))
    }

    /// Compile a traced graph and certify it with a proof certificate.
    ///
    /// Calls [`builder().build()`](Self::builder) then
    /// [`nn_verify::verify_compiled`] to produce a
    /// [`VerifiedModel<CompiledModel>`](nn_verify::VerifiedModel) pairing
    /// the compiled model with its certificate bundle.
    ///
    /// Use this when you need a machine-checkable proof certificate (e.g.,
    /// for deployment auditing). Use [`from_trace_verified`](Self::from_trace_verified)
    /// for quick IBP verification without certificate generation.
    ///
    /// # Errors
    ///
    /// Returns [`CompiledModelError::CompileFailed`] if trace compilation fails.
    /// Returns [`CompiledModelError::CertifyFailed`] if certification fails.
    ///
    /// Part of #3042.
    pub fn from_trace_certified(
        graph: &ComputationGraph,
        cache: &PipelineCache,
        bounds: &nn_verify::BoundedTensor,
        config: &nn_verify::CertifyConfig,
    ) -> Result<nn_verify::VerifiedModel<Self>> {
        let model = Self::builder(graph, cache).build()?;
        nn_verify::verify_compiled(model, graph, bounds, config)
            .map_err(|e| TensorError::from(CompiledModelError::CertifyFailed(e)))
    }
}
