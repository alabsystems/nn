// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verified compilation bridge — pairs certification with any compiled model.
//!
//! The caller compiles the model (using any backend) and passes the same
//! computation graph to certification. The result is a `VerifiedModel<M>`
//! that bundles the compiled model with its proof certificate.
//!
//! ```rust,ignore
//! use nn_verify::{verify_compiled, CertifyConfig};
//! use nn_core::dyn_tensor::trace::trace_graph;
//!
//! let (_, graph) = trace_graph(|| model.forward(&input))?;
//! let compiled = CompiledModel::builder(&graph, &cache).build()?;
//! let bounds = BoundedTensor::new(lower, upper)?;
//! let config = CertifyConfig::new("nn_model");
//! let verified = verify_compiled(compiled, &graph, &bounds, &config)?;
//! verified.save_certificate(Path::new("model.proof.json"))?;
//! ```
//!
//! Part of #3020 (Proof Certificates), #2218.

use std::path::Path;

use ny_api::BoundedTensor;
use nn_core::dyn_tensor::trace::ComputationGraph;

use crate::certify::{certify_model, CertifyConfig, CertifyError, CertifyResult};

/// A compiled model paired with its verification certificate.
///
/// Generic over `M` so any backend's compiled model type works
/// (e.g., `CompiledModel` from nn-metal, or a CPU-only model).
#[derive(Debug)]
#[non_exhaustive]
pub struct VerifiedModel<M> {
    /// The compiled model (backend-specific).
    pub model: M,
    /// The certification result (backend-agnostic).
    pub certificate: CertifyResult,
}

impl<M> VerifiedModel<M> {
    /// Save the certificate bundle to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns [`CertifyError::Verify`] if serialization or file I/O fails.
    pub fn save_certificate(&self, path: &Path) -> Result<(), CertifyError> {
        self.certificate.bundle.save(path)?;
        Ok(())
    }

    /// Whether the model's computation graph is fully compilable
    /// (no unverifiable learned ops).
    #[must_use]
    pub fn is_fully_verified(&self) -> bool {
        self.certificate.verifiability.is_fully_compilable()
    }
}

/// Certify a computation graph and pair the result with a pre-compiled model.
///
/// The caller compiles the model first (using any backend), then passes both
/// the compiled model and the original computation graph here. The graph is
/// translated to a NY `GraphNetwork`, bounds are propagated via
/// IBP/CROWN escalation, and a `CertificateBundle` is generated.
///
/// # Errors
///
/// Returns [`CertifyError::UnverifiableOps`] if the graph contains ops
/// classified as `UnverifiableLearned`.
/// Returns [`CertifyError::Verify`] for translation or propagation failures.
pub fn verify_compiled<M>(
    model: M,
    graph: &ComputationGraph,
    bounds: &BoundedTensor,
    config: &CertifyConfig,
) -> Result<VerifiedModel<M>, CertifyError> {
    let certificate = certify_model(graph, bounds, config)?;
    Ok(VerifiedModel { model, certificate })
}

/// Certify a compiled model with pre-computed peephole transform proofs (#4311).
///
/// This is the full certifying compiler path. The compilation pipeline:
/// 1. Applies peephole transforms (FusedResBlock, style absorption, batched style)
/// 2. Generates equivalence proofs via [`generate_kokoro_transform_bundle`]
/// 3. Passes the compiled model, graph, bounds, AND transform proofs here
///
/// The resulting `VerifiedModel` has both the NY bounds certificate
/// and the peephole transform equivalence proofs. For Milestone 1 (#4311),
/// this is the path that produces a `CertificateBundle` with zero unverified
/// transforms.
///
/// # Errors
///
/// Returns [`CertifyError::UnverifiableOps`] if the graph contains ops
/// classified as `UnverifiableLearned`.
/// Returns [`CertifyError::Verify`] for translation or propagation failures.
///
/// [`generate_kokoro_transform_bundle`]: crate::resblock_equivalence::generate_kokoro_transform_bundle
pub fn verify_compiled_with_transforms<M>(
    model: M,
    graph: &ComputationGraph,
    bounds: &BoundedTensor,
    config: CertifyConfig,
    transform_bundle: crate::certificate_types::TransformProofBundle,
) -> Result<VerifiedModel<M>, CertifyError> {
    let config = config.with_transform_proofs(transform_bundle);
    let certificate = certify_model(graph, bounds, &config)?;
    Ok(VerifiedModel { model, certificate })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn_core::dyn_tensor::trace::{record_input, trace_graph, TraceOp};
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::layers::{Linear, Module};
    use nn_core::Device;
    use ndarray::{ArrayD, IxDyn};

    /// Dummy compiled model for testing (stands in for CompiledModel).
    #[derive(Debug)]
    struct DummyCompiled {
        name: String,
    }

    #[test]
    fn test_verify_compiled_linear() {
        let weight = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
        let linear = Linear::new(weight, None).unwrap();
        let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).unwrap();

        let (_output, graph) = trace_graph(|| {
            let mut traced = input.clone();
            if let Some(id) = record_input(input.dims(), input.dtype()) {
                traced.set_trace_id(id);
            }
            let h = linear.forward(&traced)?;
            h.relu()
        })
        .unwrap();

        let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
        let input_bounds = BoundedTensor::new(lower, upper).unwrap();

        let compiled = DummyCompiled {
            name: "test_model".to_string(),
        };
        let config = CertifyConfig::new("test_linear");
        let verified = verify_compiled(compiled, &graph, &input_bounds, &config).unwrap();

        assert!(verified.is_fully_verified());
        assert!(!verified.certificate.bundle.certificates.is_empty());
        assert_eq!(verified.model.name, "test_model");
    }

    #[test]
    fn test_verify_compiled_unverifiable_rejected() {
        use nn_core::dyn_tensor::trace::TraceNode;
        use nn_core::DType;

        let input_node = TraceNode::new(
            0,
            "input".to_string(),
            TraceOp::Input,
            vec![],
            vec![1, 4],
            DType::F32,
        );
        let custom_node = TraceNode::new(
            1,
            "custom".to_string(),
            TraceOp::Custom {
                name: "unknown_op".to_string(),
            },
            vec![0],
            vec![1, 4],
            DType::F32,
        );
        let graph = ComputationGraph::from_nodes(vec![input_node, custom_node]);

        let lower = ArrayD::from_elem(IxDyn(&[1, 4]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[1, 4]), 1.0f32);
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let compiled = DummyCompiled {
            name: "bad_model".to_string(),
        };
        let config = CertifyConfig::new("test_unverifiable");
        let result = verify_compiled(compiled, &graph, &bounds, &config);

        match result {
            Err(CertifyError::UnverifiableOps { ops }) => {
                assert!(ops.contains(&"unknown_op".to_string()));
            }
            other => panic!("expected UnverifiableOps, got {other:?}"),
        }
    }
}
