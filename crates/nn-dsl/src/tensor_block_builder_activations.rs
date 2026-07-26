// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Activation and padding builder methods for `TensorBlockBuilder`.
//!
//! Extracted from `tensor_block_builder.rs` to stay under the 500-line limit.
//! Contains `add_sigmoid`, `add_gelu`, `add_relu`, `add_tanh`, `add_softplus`,
//! `add_exp`, and `add_zero_pad_1d`.

use crate::tensor_ir::{TensorNode, TensorNodeId, TensorOpKind};

use super::TensorBlockBuilder;

impl TensorBlockBuilder {
    /// Add a sigmoid activation op. Returns output node ID.
    pub fn add_sigmoid(&mut self, input: TensorNodeId, out_shape: &[usize]) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Sigmoid { input },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a fused SiLU / swish activation op: `silu(x) = x * sigmoid(x)`.
    ///
    /// Emits a single `TensorOpKind::Silu` node rather than the
    /// `Sigmoid` + `BinaryMul` decomposition. The fused form is what lets the
    /// verifier (ny `Layer::SiLU`) recognize a `MulBinary(SiLU(gate), up)`
    /// SwiGLU pattern and apply correlation-aware (zonotope) tightening of the
    /// gate*up product instead of decorrelating it. Returns output node ID.
    pub fn add_silu(&mut self, input: TensorNodeId, out_shape: &[usize]) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Silu { input },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a zero-pad-1d op (left and/or right padding on the last axis).
    ///
    /// Output shape: input shape with `shape[last] += pad_left + pad_right`.
    /// Used for causal Conv1d decomposition (`ZeroPad1d` + `Conv1d(padding=0)`).
    pub fn add_zero_pad_1d(
        &mut self,
        input: TensorNodeId,
        pad_left: usize,
        pad_right: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::ZeroPad1d {
                input,
                pad_left,
                pad_right,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a GELU activation op (tanh approximation). Returns output node ID.
    pub fn add_gelu(&mut self, input: TensorNodeId, out_shape: &[usize]) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Gelu { input },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a GELU activation op (exact erf). Returns output node ID.
    ///
    /// `0.5 * x * (1 + erf(x / sqrt(2)))`. More precise than the tanh
    /// approximation (`add_gelu`).
    pub fn add_gelu_erf(&mut self, input: TensorNodeId, out_shape: &[usize]) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::GeluErf { input },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a ReLU activation op (`max(x, 0)`). Returns output node ID.
    pub fn add_relu(&mut self, input: TensorNodeId, out_shape: &[usize]) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Relu { input },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a LeakyReLU activation op (`x if x >= 0, else negative_slope * x`).
    ///
    /// Used in Kokoro decoder (ISTFTNet vocoder): `LeakyReLU(0.1)` per upsample
    /// stage, `LeakyReLU(0.01)` before `conv_post`.
    pub fn add_leaky_relu(
        &mut self,
        input: TensorNodeId,
        negative_slope: f32,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::LeakyRelu {
                input,
                negative_slope,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add an ELU activation op (`x if x >= 0, else alpha * (exp(x) - 1)`).
    ///
    /// Alpha is baked into the single-dispatch MSL kernel as a compile-time
    /// constant. Used by Kokoro ISTFTNet decoder. Part of #3230 (Gap 3).
    pub fn add_elu(
        &mut self,
        input: TensorNodeId,
        alpha: f32,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Elu { input, alpha },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a tanh activation op. Returns output node ID.
    pub fn add_tanh(&mut self, input: TensorNodeId, out_shape: &[usize]) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Tanh { input },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a softplus activation op (`ln(1 + exp(x))`). Returns output node ID.
    ///
    /// Used in DeltaNet gate computation: `softplus(a_proj(x) + dt_bias)`.
    pub fn add_softplus(&mut self, input: TensorNodeId, out_shape: &[usize]) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Softplus { input },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add an exp activation op (`exp(x)`). Returns output node ID.
    ///
    /// Used in DeltaNet decay gate: `exp(g)` where `g < 0` produces decay in `(0, 1)`.
    pub fn add_exp(&mut self, input: TensorNodeId, out_shape: &[usize]) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Exp { input },
            out_shape.to_vec(),
        ));
        id
    }
}

// Kani proof harnesses for Softplus and Exp activation builders.
// Part of #834 AC10.
#[cfg(kani)]
#[path = "softplus_exp_kani_builder_tests.rs"]
mod softplus_exp_kani_builder;

/// Regular tests for node-count properties converted from tautological Kani
/// harnesses (removed in P1-67, 0d2c11a4). Node counts after explicit
/// construction are structurally guaranteed — model-checking adds no value,
/// but the assertions are useful as regression tests.
#[cfg(test)]
mod softplus_exp_builder_node_tests {
    use super::TensorBlockBuilder;

    #[test]
    fn softplus_builder_node_count() {
        let mut b = TensorBlockBuilder::new("test_softplus_count");
        let input = b.add_input("x", &[2, 3]);
        let out = b.add_softplus(input, &[2, 3]);
        let def = b.build(out).expect("valid graph");
        assert_eq!(
            def.nodes.len(),
            2,
            "softplus graph must have exactly 2 nodes (1 input + 1 Softplus)"
        );
    }

    #[test]
    fn exp_builder_node_count() {
        let mut b = TensorBlockBuilder::new("test_exp_count");
        let input = b.add_input("x", &[2, 3]);
        let out = b.add_exp(input, &[2, 3]);
        let def = b.build(out).expect("valid graph");
        assert_eq!(
            def.nodes.len(),
            2,
            "exp graph must have exactly 2 nodes (1 input + 1 Exp)"
        );
    }

    #[test]
    fn gate_subgraph_node_count() {
        let mut b = TensorBlockBuilder::new("test_gate_count");
        let input = b.add_input("a_proj_plus_bias", &[2, 4]);
        let sp = b.add_softplus(input, &[2, 4]);
        let gate = b.add_exp(sp, &[2, 4]);
        let def = b.build(gate).expect("valid graph");
        assert_eq!(
            def.nodes.len(),
            3,
            "gate sub-graph must have 3 nodes (1 input + Softplus + Exp)"
        );
    }
}
