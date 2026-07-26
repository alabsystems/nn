// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: `TensorOpKind::Embedding` → NY `AddConstant(midpoint)`.
//!
//! Part of #743: Embedding lookup op for token/phoneme embeddings.

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a minimal tensor kernel: indices input → embedding(weight) → output.
///
/// Two Input nodes: indices (Variable) and weight (ConstantTensor).
/// The bindings array must have 2 entries matching this order.
fn embedding_tensor_kernel(indices_shape: &[usize], weight_shape: &[usize]) -> TensorKernelDef {
    // Output shape: indices_shape ++ [embedding_dim]
    let embedding_dim = weight_shape[1];
    let mut out_shape = indices_shape.to_vec();
    out_shape.push(embedding_dim);

    TensorKernelDef::new(
        "embedding_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "indices".to_string(),
                    shape: indices_shape.to_vec(),
                },
                indices_shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "weight".to_string(),
                    shape: weight_shape.to_vec(),
                },
                weight_shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Embedding {
                    input: TensorNodeId::new(0),
                    weight: TensorNodeId::new(1),
                },
                out_shape,
            ),
        ],
        TensorNodeId::new(2),
    )
}

/// Create a weight table where each row r, dimension d has value `(r * dim + d) as f32`.
fn sequential_weights(num_embeddings: usize, embedding_dim: usize) -> ArrayD<f32> {
    let data: Vec<f32> = (0..num_embeddings * embedding_dim)
        .map(|i| i as f32)
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[num_embeddings, embedding_dim]), data)
        .expect("valid weight shape")
}

#[test]
fn test_embedding_tensor_builds_graph() {
    let def = embedding_tensor_kernel(&[4], &[10, 8]);
    let weights = sequential_weights(10, 8);
    let graph = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(weights),
        ],
    )
    .expect("embedding tensor graph should build");
    // The embedding lowers to an index-agnostic box subgraph (see
    // graph_tensor_embedding.rs): Reshape(flatten indices) -> Linear(zero weight,
    // midpoint bias) -> Qdq(widen to table spread) -> Reshape(restore shape) = 4
    // nodes for a non-degenerate weight table.
    assert_eq!(graph.num_nodes(), 4, "embedding graph should have 4 nodes");
}

#[test]
fn test_embedding_tensor_ibp_bounds_within_weight_extrema() {
    // Weight table: 4 embeddings, 3 dimensions.
    // Row 0: [0.0, 1.0, 2.0]
    // Row 1: [3.0, 4.0, 5.0]
    // Row 2: [6.0, 7.0, 8.0]
    // Row 3: [9.0, 10.0, 11.0]
    //
    // Per-dimension min/max (the tightest sound, index-agnostic box):
    //   dim 0: [0.0, 9.0]
    //   dim 1: [1.0, 10.0]
    //   dim 2: [2.0, 11.0]
    let def = embedding_tensor_kernel(&[2], &[4, 3]);
    let weights = sequential_weights(4, 3);
    let graph = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(weights),
        ],
    )
    .expect("embedding graph should build");

    // IBP input: the graph's NETWORK_INPUT is the INDEX tensor, shape [2]. The
    // emitted box is index-agnostic (zero-weight collapse), so the index interval
    // does not affect the output; any valid index bounds work.
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 3.0f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    // Output has the declared shape [*index_dims, D] = [2, 3].
    assert_eq!(lo.shape(), &[2, 3], "embedding output shape");

    // Per-dim table extrema. The emitted box must CONTAIN each dim's [lo, hi]
    // spread (sound over-approximation) for every output position, independent
    // of the index interval. The box is widened by the GLOBAL half-spread, so it
    // may be slightly wider than the per-dim spread but never narrower.
    let table_lo = [0.0f32, 1.0, 2.0];
    let table_hi = [9.0f32, 10.0, 11.0];
    for p in 0..2 {
        for d in 0..3 {
            assert!(
                lo[[p, d]] <= table_lo[d] + 1e-4,
                "pos {p} dim {d}: lower {} must be <= table min {}",
                lo[[p, d]],
                table_lo[d]
            );
            assert!(
                hi[[p, d]] >= table_hi[d] - 1e-4,
                "pos {p} dim {d}: upper {} must be >= table max {}",
                hi[[p, d]],
                table_hi[d]
            );
            assert!(
                lo[[p, d]].is_finite() && hi[[p, d]].is_finite(),
                "pos {p} dim {d}: bounds must be finite"
            );
        }
    }
}

#[test]
fn test_embedding_tensor_block_builder() {
    use nn_dsl::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new("embedding_block");
    let idx = b.add_input("indices", &[8]);
    let w = b.add_input("weight", &[256, 64]);
    let emb = b.add_embedding(idx, w, &[8, 64]);
    let def = b.build(emb).expect("valid graph");

    assert_eq!(def.name, "embedding_block");
    assert_eq!(def.nodes.len(), 3); // 2 inputs + 1 embedding
    assert_eq!(def.output, TensorNodeId::new(2));
    assert_eq!(def.nodes[2].shape, vec![8, 64]);
    assert!(matches!(
        &def.nodes[2].kind,
        TensorOpKind::Embedding { input, weight }
            if *input == TensorNodeId::new(0) && *weight == TensorNodeId::new(1)
    ));
}

#[test]
fn test_embedding_weight_variable_rejected() {
    // Both inputs as Variable — the weight must be ConstantTensor.
    let def = embedding_tensor_kernel(&[4], &[10, 8]);
    let result = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    );
    assert!(
        result.is_err(),
        "embedding with Variable weight should be rejected"
    );
}

#[test]
fn test_embedding_weight_scalar_rejected() {
    // Weight as ConstantScalar — should be rejected (needs 2-D table).
    let def = embedding_tensor_kernel(&[4], &[10, 8]);
    let result = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantScalar(1.0),
        ],
    );
    assert!(
        result.is_err(),
        "embedding with ConstantScalar weight should be rejected"
    );
}

#[test]
fn test_embedding_indices_weight_tensor_rejected() {
    // Indices as ConstantTensor — must be Variable.
    let def = embedding_tensor_kernel(&[4], &[10, 8]);
    let weights = sequential_weights(10, 8);
    let fake_indices = ArrayD::from_elem(IxDyn(&[4]), 0.0f32);
    let result = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::ConstantTensor(fake_indices),
            TensorParamBinding::ConstantTensor(weights),
        ],
    );
    assert!(
        result.is_err(),
        "embedding with ConstantTensor indices should be rejected"
    );
}

#[test]
fn test_embedding_nonfinite_weight_rejected() {
    // Weight table with NaN — should be rejected.
    let mut weights = sequential_weights(4, 3);
    weights[[1, 2]] = f32::NAN;
    let def = embedding_tensor_kernel(&[2], &[4, 3]);
    let result = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(weights),
        ],
    );
    assert!(
        result.is_err(),
        "embedding with NaN in weight table should be rejected"
    );
}

#[test]
fn test_embedding_dvoice_scale() {
    // dvoice phoneme embedding: 256 phonemes, 512 dims (Kokoro-style).
    let def = embedding_tensor_kernel(&[32], &[256, 512]);
    let weights = sequential_weights(256, 512);
    let graph = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(weights),
        ],
    )
    .expect("dvoice-scale embedding should build");
    // Reshape -> Linear -> Qdq -> Reshape (non-degenerate table); see
    // graph_tensor_embedding.rs for the index-agnostic box lowering.
    assert_eq!(graph.num_nodes(), 4);
}
