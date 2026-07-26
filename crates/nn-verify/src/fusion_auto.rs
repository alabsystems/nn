// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Auto-generate fusion verification specs from detected elementwise chains.
//!
//! Takes [`FusionChainInfo`] from nn-dsl's trace fusion detection and
//! converts each pairwise entry into a NY fusion equivalence
//! proof via the existing [`verify_fusion_equivalence`] infrastructure.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_core::dyn_tensor::trace::trace_graph;
//! use nn_dsl::detect_fusion_chains;
//! use nn_verify::fusion_auto::{generate_fusion_specs, default_variable_bounds};
//! use nn_verify::verify_fusion_equivalence;
//!
//! let (output, graph) = trace_graph(|| model.forward(&input))?;
//! let chains = detect_fusion_chains(&graph)?;
//! let specs = generate_fusion_specs(&chains);
//! for spec in &specs {
//!     let bounds = default_variable_bounds(spec.num_shared_inputs());
//!     let result = verify_fusion_equivalence(&spec.as_fusion_spec(), &bounds, 1e-6)?;
//!     assert!(result.is_conclusive() && result.within_epsilon);
//! }
//! ```

use nn_dsl::ir::KernelDef;
use nn_dsl::FusionChainInfo;

use crate::fusion_spec::FusionSpec;

/// Owned fusion specification for automatic verification.
///
/// Like [`FusionSpec`] but owns its `KernelDef`s, allowing the spec to
/// outlive the chain detection that produced it.
#[derive(Debug, Clone)]
pub struct AutoFusionSpec {
    /// The 2-op fused kernel.
    pub fused: KernelDef,
    /// First individual kernel.
    pub first: KernelDef,
    /// Second individual kernel.
    pub second: KernelDef,
    /// Maps first's params to shared input indices.
    pub first_param_indices: Vec<usize>,
    /// Maps second's params to shared input indices.
    pub second_param_indices: Vec<usize>,
    /// Which param of second takes first's output.
    pub second_input_from_first: usize,
    /// Human-readable name for diagnostics and status tracking.
    pub chain_name: String,
}

impl AutoFusionSpec {
    /// Borrow as a [`FusionSpec`] for use with `verify_fusion_equivalence`.
    pub fn as_fusion_spec(&self) -> FusionSpec<'_> {
        FusionSpec {
            fused: &self.fused,
            first: &self.first,
            second: &self.second,
            num_shared_inputs: self.fused.params.len(),
            first_param_indices: &self.first_param_indices,
            second_param_indices: &self.second_param_indices,
            second_input_from_first: self.second_input_from_first,
        }
    }

    /// Number of shared input variables (= fused kernel's param count).
    pub fn num_shared_inputs(&self) -> usize {
        self.fused.params.len()
    }
}

/// Generate pairwise fusion verification specs from detected fusion chains.
///
/// For each chain of N ops, produces N-1 [`AutoFusionSpec`]s. Each spec
/// can be verified with [`verify_fusion_equivalence`](crate::verify_fusion_equivalence)
/// to prove that fusing two consecutive ops is mathematically equivalent
/// to running them sequentially.
pub fn generate_fusion_specs(chains: &[FusionChainInfo]) -> Vec<AutoFusionSpec> {
    let mut specs = Vec::new();
    for chain in chains {
        for (i, pair) in chain.pairs.iter().enumerate() {
            let name = if chain.pairs.len() == 1 {
                chain.chain_name.clone()
            } else {
                format!("{}_pair{i}", chain.chain_name)
            };
            specs.push(AutoFusionSpec {
                fused: pair.fused.clone(),
                first: pair.first.clone(),
                second: pair.second.clone(),
                first_param_indices: pair.first_param_indices.clone(),
                second_param_indices: pair.second_param_indices.clone(),
                second_input_from_first: pair.second_input_from_first,
                chain_name: name,
            });
        }
    }
    specs
}

/// Default variable bounds for auto-generated fusion verification.
///
/// Uses [-3.0, 3.0] for all variables, matching the range used by the
/// existing handwritten fusion tests. Callers can provide tighter
/// model-specific bounds for more precise verification.
pub fn default_variable_bounds(num_inputs: usize) -> Vec<(f32, f32)> {
    vec![(-3.0, 3.0); num_inputs]
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;
    use nn_dsl::detect_fusion_chains;

    fn input_node(id: u64, shape: &[usize]) -> TraceNode {
        TraceNode::new(
            id,
            format!("input_{id}"),
            TraceOp::Input,
            vec![],
            shape.to_vec(),
            DType::F32,
        )
    }

    fn op_node(id: u64, op: TraceOp, inputs: &[u64], shape: &[usize]) -> TraceNode {
        TraceNode::new(
            id,
            format!("{}_{id}", op.canonical_name()),
            op,
            inputs.to_vec(),
            shape.to_vec(),
            DType::F32,
        )
    }

    /// Build a simple graph: Input → Exp → Relu (2-op chain).
    fn build_exp_relu_graph() -> ComputationGraph {
        ComputationGraph::from_nodes(vec![
            input_node(0, &[1, 4, 8]),
            op_node(1, TraceOp::Exp, &[0], &[1, 4, 8]),
            op_node(2, TraceOp::Relu, &[1], &[1, 4, 8]),
        ])
    }

    /// Build a 3-op chain: Input → Exp → Mul(exp_out, input) → Relu.
    fn build_exp_mul_relu_graph() -> ComputationGraph {
        ComputationGraph::from_nodes(vec![
            input_node(0, &[1, 4, 8]),
            op_node(1, TraceOp::Exp, &[0], &[1, 4, 8]),
            op_node(2, TraceOp::Mul, &[1, 0], &[1, 4, 8]),
            op_node(3, TraceOp::Relu, &[2], &[1, 4, 8]),
        ])
    }

    #[test]
    fn test_detect_exp_relu_chain() {
        let graph = build_exp_relu_graph();
        let chains = detect_fusion_chains(&graph).expect("chain detection");
        assert_eq!(chains.len(), 1, "one chain detected");
        assert_eq!(chains[0].chain_len, 2);
        assert_eq!(chains[0].pairs.len(), 1, "one pair for 2-op chain");

        let pair = &chains[0].pairs[0];
        assert_eq!(pair.fused.params.len(), 1, "fused has 1 external input");
        assert_eq!(pair.first.params.len(), 1, "exp has 1 input");
        assert_eq!(pair.second.params.len(), 1, "relu has 1 input (from exp)");
        assert_eq!(pair.second_input_from_first, 0);
    }

    #[test]
    fn test_detect_three_op_chain() {
        let graph = build_exp_mul_relu_graph();
        let chains = detect_fusion_chains(&graph).expect("chain detection");
        assert_eq!(chains.len(), 1, "one chain detected");
        assert_eq!(chains[0].chain_len, 3);
        assert_eq!(chains[0].pairs.len(), 2, "two pairs for 3-op chain");
    }

    #[test]
    fn test_generate_specs_from_chain() {
        let graph = build_exp_relu_graph();
        let chains = detect_fusion_chains(&graph).expect("chain detection");
        let specs = generate_fusion_specs(&chains);
        assert_eq!(specs.len(), 1);

        let spec = &specs[0];
        assert_eq!(spec.num_shared_inputs(), 1);
        assert_eq!(spec.first_param_indices, vec![0]);
        assert_eq!(spec.second_input_from_first, 0);

        // Verify the FusionSpec borrows correctly.
        let _fs = spec.as_fusion_spec();
    }

    #[test]
    fn test_generate_specs_three_op_chain() {
        let graph = build_exp_mul_relu_graph();
        let chains = detect_fusion_chains(&graph).expect("chain detection");
        let specs = generate_fusion_specs(&chains);
        assert_eq!(specs.len(), 2, "two pairwise specs for 3-op chain");

        for spec in &specs {
            let _fs = spec.as_fusion_spec();
        }
    }

    #[test]
    fn test_default_variable_bounds() {
        let bounds = default_variable_bounds(3);
        assert_eq!(bounds.len(), 3);
        assert_eq!(bounds[0], (-3.0, 3.0));
    }
}
