// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Convenience constructors and certificate access for `CompiledModel`.
//!
//! Extracted from `compiled_model.rs` for 450-line compliance.

use nn_core::dyn_tensor::trace;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Result};

use super::CompiledModel;

impl CompiledModel {
    /// Trace a forward pass and compile it for repeated GPU execution.
    ///
    /// Combines `trace_graph()` + `builder().build()` into a single call with
    /// automatic input registration. Each input `DynTensor` is cloned,
    /// assigned a trace node ID, and passed to the closure. The closure
    /// runs eagerly (producing real outputs) while recording the graph.
    ///
    /// # Errors
    ///
    /// Returns an error if tracing fails, trace compilation fails, or
    /// weight upload fails.
    #[deprecated(
        note = "use `compile_forward_with(inputs, forward, cache, |b| b)` or trace + builder directly"
    )]
    pub fn compile_forward<F>(
        inputs: &[&DynTensor],
        forward: F,
        cache: &crate::cache::PipelineCache,
    ) -> Result<Self>
    where
        F: FnOnce(&[DynTensor]) -> Result<DynTensor>,
    {
        let input_meta: Vec<(Vec<usize>, DType)> = inputs
            .iter()
            .map(|t| (t.dims().to_vec(), t.dtype()))
            .collect();

        let (_output, graph) = trace::trace_graph(|| {
            let traced: Vec<DynTensor> = inputs
                .iter()
                .zip(input_meta.iter())
                .map(|(t, (shape, dtype))| {
                    let mut t = (*t).clone();
                    if let Some(id) = trace::record_input(shape, *dtype) {
                        t.set_trace_id(id);
                    }
                    t
                })
                .collect();
            forward(&traced)
        })?;

        Self::builder(&graph, cache).build()
    }

    /// Returns the proof certificate as JSON, if automatic verification succeeded.
    ///
    /// When the `verify` feature is enabled, `builder().build()` and `compile_forward()`
    /// automatically attempt NY verification. If successful, the certificate
    /// is stored here. Returns `None` if verification was disabled, unsupported for
    /// this model's ops, or failed.
    ///
    /// Part of #3042.
    pub fn proof_certificate_json(&self) -> Option<&str> {
        self.def.proof_certificate.as_deref()
    }

    /// Save the proof certificate to a file. Returns `Ok(false)` if no certificate.
    ///
    /// Part of #3042.
    pub fn save_proof_certificate(&self, path: impl AsRef<std::path::Path>) -> Result<bool> {
        let Some(json) = &self.def.proof_certificate else {
            return Ok(false);
        };
        std::fs::write(path, json)?;
        Ok(true)
    }
}
