// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for graph construction and LayerSpec correctness (#4132).
//!
//! Proves properties of:
//! - `ComputationGraph::from_nodes`: id_to_index consistency, output node selection
//! - `ComputationGraph::validate_topology`: forward reference detection, acyclicity
//! - `ComputationGraph` node connectivity invariants
//! - `TraceNode` field accessor correctness and identity preservation
//! - `SubgraphSpec` / `extract_subgraph` input synthesis and node remapping
//! - `is_ay_compatible_op` coverage for all major activation/element-wise ops
//! - Graph dimension matching: output shape propagation through chains

#[cfg(kani)]
mod proofs {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
    use nn_core::DType;

    use crate::subgraph_extract::{
        extract_subgraph, is_ay_compatible_op, validate_subgraph, SubgraphSpec,
    };

    // ========================================================================
    // Helpers
    // ========================================================================

    fn make_input(id: u64, shape: Vec<usize>) -> TraceNode {
        TraceNode::new(
            id,
            format!("input_{id}"),
            TraceOp::Input,
            vec![],
            shape,
            DType::F32,
        )
    }

    fn make_relu(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
        TraceNode::new(
            id,
            format!("relu_{id}"),
            TraceOp::Relu,
            vec![input_id],
            shape,
            DType::F32,
        )
    }

    fn make_sigmoid(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
        TraceNode::new(
            id,
            format!("sigmoid_{id}"),
            TraceOp::Sigmoid,
            vec![input_id],
            shape,
            DType::F32,
        )
    }

    fn make_add(id: u64, lhs: u64, rhs: u64, shape: Vec<usize>) -> TraceNode {
        TraceNode::new(
            id,
            format!("add_{id}"),
            TraceOp::Add,
            vec![lhs, rhs],
            shape,
            DType::F32,
        )
    }

    fn make_neg(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
        TraceNode::new(
            id,
            format!("neg_{id}"),
            TraceOp::Neg,
            vec![input_id],
            shape,
            DType::F32,
        )
    }

    fn make_tanh(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
        TraceNode::new(
            id,
            format!("tanh_{id}"),
            TraceOp::Tanh,
            vec![input_id],
            shape,
            DType::F32,
        )
    }

    fn make_mul(id: u64, lhs: u64, rhs: u64, shape: Vec<usize>) -> TraceNode {
        TraceNode::new(
            id,
            format!("mul_{id}"),
            TraceOp::Mul,
            vec![lhs, rhs],
            shape,
            DType::F32,
        )
    }

    fn make_softmax(id: u64, input_id: u64, dim: usize, shape: Vec<usize>) -> TraceNode {
        TraceNode::new(
            id,
            format!("softmax_{id}"),
            TraceOp::Softmax { dim },
            vec![input_id],
            shape,
            DType::F32,
        )
    }

    // ========================================================================
    // 1. ComputationGraph::from_nodes — id_to_index consistency
    // ========================================================================

    /// Prove: from_nodes builds an id_to_index map where every node ID maps
    /// to its correct positional index.
    #[kani::unwind(8)]
    #[kani::proof]
    fn graph_from_nodes_id_to_index_consistent() {
        let nodes = vec![
            make_input(10, vec![4]),
            make_relu(20, 10, vec![4]),
            make_sigmoid(30, 20, vec![4]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        let all_nodes = graph.nodes();

        // Every node in the graph must be findable by its ID.
        for (i, node) in all_nodes.iter().enumerate() {
            let found = graph.node(node.id());
            assert!(found.is_some(), "node at index {i} must be findable by ID");
            assert_eq!(
                found.unwrap().id(),
                node.id(),
                "node lookup must return the same ID"
            );
        }
    }

    // ========================================================================
    // 2. ComputationGraph::from_nodes — output node is always the last node
    // ========================================================================

    /// Prove: from_nodes sets the output node to the last node in the vec.
    #[kani::unwind(8)]
    #[kani::proof]
    fn graph_from_nodes_output_is_last() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_relu(2, 1, vec![4]),
            make_sigmoid(3, 2, vec![4]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        let output = graph.output_node();
        assert!(output.is_some(), "non-empty graph must have output node");
        assert_eq!(output.unwrap().id(), 3, "output must be the last node");
    }

    // ========================================================================
    // 3. ComputationGraph::from_nodes — empty graph has no output
    // ========================================================================

    /// Prove: from_nodes on an empty vec produces is_empty() and no output_node.
    #[kani::unwind(8)]
    #[kani::proof]
    fn graph_from_nodes_empty_has_no_output() {
        let graph = ComputationGraph::from_nodes(vec![]);
        assert!(graph.is_empty(), "empty nodes must produce empty graph");
        assert!(
            graph.output_node().is_none(),
            "empty graph must have no output"
        );
        assert_eq!(graph.len(), 0);
    }

    // ========================================================================
    // 4. validate_topology — valid chain passes
    // ========================================================================

    /// Prove: a properly ordered chain (input -> relu -> sigmoid) passes
    /// topology validation.
    #[kani::unwind(8)]
    #[kani::proof]
    fn validate_topology_valid_chain_passes() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_relu(2, 1, vec![4]),
            make_sigmoid(3, 2, vec![4]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        assert!(
            graph.validate_topology().is_ok(),
            "valid topological order must pass"
        );
    }

    // ========================================================================
    // 5. validate_topology — forward reference fails
    // ========================================================================

    /// Prove: a graph where a node references a later node (forward reference)
    /// fails topology validation.
    #[kani::unwind(8)]
    #[kani::proof]
    fn validate_topology_forward_reference_fails() {
        // Node 1 references node 2, but node 2 comes after node 1.
        let nodes = vec![
            make_relu(1, 2, vec![4]), // references id 2 which hasn't been seen
            make_input(2, vec![4]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        assert!(
            graph.validate_topology().is_err(),
            "forward reference must fail topology validation"
        );
    }

    // ========================================================================
    // 6. validate_topology — dangling input reference fails
    // ========================================================================

    /// Prove: a node referencing a non-existent node ID fails validation.
    #[kani::unwind(8)]
    #[kani::proof]
    fn validate_topology_dangling_reference_fails() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_relu(2, 999, vec![4]), // references non-existent id 999
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        assert!(
            graph.validate_topology().is_err(),
            "dangling reference must fail topology validation"
        );
    }

    // ========================================================================
    // 7. Graph node count — from_nodes preserves all nodes
    // ========================================================================

    /// Prove: from_nodes preserves the exact count of nodes.
    #[kani::unwind(8)]
    #[kani::proof]
    fn graph_node_count_preserved() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_relu(2, 1, vec![4]),
            make_sigmoid(3, 2, vec![4]),
            make_neg(4, 3, vec![4]),
        ];
        let expected_count = nodes.len();
        let graph = ComputationGraph::from_nodes(nodes);
        assert_eq!(graph.len(), expected_count);
        assert_eq!(graph.nodes().len(), expected_count);
    }

    // ========================================================================
    // 8. input_nodes — correctly identifies Input ops
    // ========================================================================

    /// Prove: input_nodes returns exactly the nodes with TraceOp::Input.
    #[kani::unwind(8)]
    #[kani::proof]
    fn graph_input_nodes_correct() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_input(2, vec![8]),
            make_add(3, 1, 2, vec![4]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        let inputs = graph.input_nodes();
        assert_eq!(inputs.len(), 2, "must find exactly 2 input nodes");
        assert!(
            inputs.iter().all(|n| matches!(n.op(), TraceOp::Input)),
            "all returned nodes must be Input ops"
        );
    }

    // ========================================================================
    // 9. TraceNode — accessors match construction values
    // ========================================================================

    /// Prove: TraceNode field accessors return the values passed to the constructor.
    #[kani::unwind(1)]
    #[kani::proof]
    fn trace_node_accessors_match_construction() {
        let node = TraceNode::new(
            42,
            "test_node".to_string(),
            TraceOp::Relu,
            vec![10, 20],
            vec![2, 4],
            DType::F32,
        );
        assert_eq!(node.id(), 42);
        assert_eq!(node.name(), "test_node");
        assert!(matches!(node.op(), TraceOp::Relu));
        assert_eq!(node.inputs(), &[10, 20]);
        assert_eq!(node.output_shape(), &[2, 4]);
        assert_eq!(node.output_dtype(), DType::F32);
    }

    // ========================================================================
    // 10. mark_output — returns false for non-existent node
    // ========================================================================

    /// Prove: mark_output returns false when the node ID is not in the graph.
    #[kani::unwind(8)]
    #[kani::proof]
    fn mark_output_nonexistent_returns_false() {
        let nodes = vec![make_input(1, vec![4])];
        let mut graph = ComputationGraph::from_nodes(nodes);
        assert!(
            !graph.mark_output(999),
            "marking non-existent node must return false"
        );
    }

    // ========================================================================
    // 11. mark_output — existing node becomes additional output
    // ========================================================================

    /// Prove: mark_output on an existing node succeeds and adds it to outputs.
    #[kani::unwind(8)]
    #[kani::proof]
    fn mark_output_existing_node_succeeds() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_relu(2, 1, vec![4]),
            make_sigmoid(3, 2, vec![4]),
        ];
        let mut graph = ComputationGraph::from_nodes(nodes);
        // Default output is node 3 (last). Mark node 2 as additional output.
        assert!(graph.mark_output(2), "marking existing node must succeed");
        let outputs = graph.output_nodes();
        assert_eq!(outputs.len(), 2, "must have 2 output nodes");
    }

    // ========================================================================
    // 12. set_primary_output — replaces the output list
    // ========================================================================

    /// Prove: set_primary_output replaces the output list with a single node.
    #[kani::unwind(8)]
    #[kani::proof]
    fn set_primary_output_replaces_list() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_relu(2, 1, vec![4]),
            make_sigmoid(3, 2, vec![4]),
        ];
        let mut graph = ComputationGraph::from_nodes(nodes);
        // Default output is node 3. Replace with node 2.
        assert!(
            graph.set_primary_output(2),
            "setting primary output to existing node must succeed"
        );
        let output = graph.output_node();
        assert_eq!(output.unwrap().id(), 2, "output must now be node 2");
        let outputs = graph.output_nodes();
        assert_eq!(outputs.len(), 1, "output list must have exactly 1 entry");
    }

    // ========================================================================
    // 13. is_ay_compatible_op — activation ops are compatible
    // ========================================================================

    /// Prove: all standard activation ops are ay-compatible.
    #[kani::unwind(1)]
    #[kani::proof]
    fn ay_compatible_activation_ops() {
        assert!(is_ay_compatible_op(&TraceOp::Relu));
        assert!(is_ay_compatible_op(&TraceOp::Sigmoid));
        assert!(is_ay_compatible_op(&TraceOp::Tanh));
        assert!(is_ay_compatible_op(&TraceOp::Exp));
        assert!(is_ay_compatible_op(&TraceOp::Sin));
        assert!(is_ay_compatible_op(&TraceOp::Cos));
        assert!(is_ay_compatible_op(&TraceOp::Neg));
        assert!(is_ay_compatible_op(&TraceOp::Abs));
        assert!(is_ay_compatible_op(&TraceOp::Sqrt));
        assert!(is_ay_compatible_op(&TraceOp::Recip));
    }

    // ========================================================================
    // 14. is_ay_compatible_op — binary ops are compatible
    // ========================================================================

    /// Prove: binary arithmetic ops are ay-compatible.
    #[kani::unwind(1)]
    #[kani::proof]
    fn ay_compatible_binary_ops() {
        assert!(is_ay_compatible_op(&TraceOp::Add));
        assert!(is_ay_compatible_op(&TraceOp::Sub));
        assert!(is_ay_compatible_op(&TraceOp::Mul));
        assert!(is_ay_compatible_op(&TraceOp::Div));
    }

    // ========================================================================
    // 15. is_ay_compatible_op — composite ops are NOT compatible
    // ========================================================================

    /// Prove: composite ops (Conv, Linear, LSTM) are NOT ay-compatible.
    #[kani::unwind(1)]
    #[kani::proof]
    fn ay_incompatible_composite_ops() {
        let w = WeightRef::from_shape(&[4, 4]);
        assert!(!is_ay_compatible_op(&TraceOp::Linear {
            weight: w.clone(),
            bias: None,
        }));
        assert!(!is_ay_compatible_op(&TraceOp::Conv1d {
            weight: w,
            bias: None,
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }));
    }

    // ========================================================================
    // 16. extract_subgraph — preserves node count for full range
    // ========================================================================

    /// Prove: extracting a subgraph for the full node range preserves all
    /// non-input nodes and adds a synthetic input.
    #[kani::unwind(8)]
    #[kani::proof]
    fn extract_subgraph_full_range_preserves_structure() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_relu(2, 1, vec![4]),
            make_sigmoid(3, 2, vec![4]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        let spec = SubgraphSpec {
            start: 0,
            end: 3,
            include_ids: None,
        };
        let result = extract_subgraph(&graph, &spec);
        assert!(result.is_ok(), "full-range extraction must succeed");
        let sub = result.unwrap();
        // Subgraph contains a synthetic input plus the extracted nodes.
        assert!(
            sub.graph.len() >= 3,
            "subgraph must contain at least 3 nodes (input + relu + sigmoid)"
        );
    }

    // ========================================================================
    // 17. validate_subgraph — valid subgraph passes
    // ========================================================================

    /// Prove: a self-contained subgraph passes validation.
    #[kani::unwind(8)]
    #[kani::proof]
    fn validate_subgraph_self_contained_passes() {
        let nodes = vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])];
        let graph = ComputationGraph::from_nodes(nodes);
        let result = validate_subgraph(&graph);
        assert!(result.is_ok(), "self-contained graph must pass validation");
    }

    // ========================================================================
    // 18. Graph connectivity — binary op wires two inputs
    // ========================================================================

    /// Prove: a binary Add node in a graph correctly references its two inputs.
    #[kani::unwind(8)]
    #[kani::proof]
    fn graph_binary_op_has_two_inputs() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_input(2, vec![4]),
            make_add(3, 1, 2, vec![4]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        let add_node = graph.node(3).unwrap();
        assert_eq!(add_node.inputs().len(), 2, "Add must have exactly 2 inputs");
        assert_eq!(add_node.inputs()[0], 1);
        assert_eq!(add_node.inputs()[1], 2);
    }

    // ========================================================================
    // 19. Graph shape chain — output shapes propagate through unary chain
    // ========================================================================

    /// Prove: in a chain of unary ops (input -> relu -> tanh -> neg),
    /// all nodes report the same output shape.
    #[kani::unwind(8)]
    #[kani::proof]
    fn graph_unary_chain_shape_propagation() {
        let shape = vec![2, 4, 8];
        let nodes = vec![
            make_input(1, shape.clone()),
            make_relu(2, 1, shape.clone()),
            make_tanh(3, 2, shape.clone()),
            make_neg(4, 3, shape.clone()),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        for node in graph.nodes() {
            assert_eq!(
                node.output_shape(),
                &shape[..],
                "unary ops must preserve shape"
            );
        }
    }

    // ========================================================================
    // 20. Graph topology — diamond pattern validates
    // ========================================================================

    /// Prove: a diamond graph (input -> relu, input -> sigmoid, add(relu, sigmoid))
    /// passes topology validation — multiple consumers of one producer is valid.
    #[kani::unwind(8)]
    #[kani::proof]
    fn graph_diamond_topology_valid() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_relu(2, 1, vec![4]),
            make_sigmoid(3, 1, vec![4]),
            make_add(4, 2, 3, vec![4]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        assert!(
            graph.validate_topology().is_ok(),
            "diamond pattern must be valid topology"
        );
        // Verify the Add node has correct inputs.
        let add_node = graph.node(4).unwrap();
        assert_eq!(add_node.inputs(), &[2, 3]);
    }

    // ========================================================================
    // 21. Node ID uniqueness — from_nodes last-wins for duplicate IDs
    // ========================================================================

    /// Prove: when nodes have duplicate IDs, from_nodes maps the ID to the
    /// last occurrence (HashMap insert overwrites). The graph still has all
    /// nodes by position, but node() lookup returns the last one.
    #[kani::unwind(8)]
    #[kani::proof]
    fn graph_duplicate_ids_last_wins_lookup() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_relu(1, 0, vec![8]), // duplicate id=1
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        assert_eq!(graph.len(), 2, "both nodes are stored");
        // node() returns the second (last) mapping for id=1.
        let found = graph.node(1).unwrap();
        assert_eq!(
            found.output_shape(),
            &[8],
            "lookup must return the last node with id=1"
        );
    }

    // ========================================================================
    // 22. Softmax dimension — TraceOp stores dimension correctly
    // ========================================================================

    /// Prove: Softmax TraceOp preserves the dim field through node construction
    /// and graph storage.
    #[kani::unwind(8)]
    #[kani::proof]
    fn softmax_dim_preserved_in_graph() {
        let nodes = vec![make_input(1, vec![2, 4]), make_softmax(2, 1, 1, vec![2, 4])];
        let graph = ComputationGraph::from_nodes(nodes);
        let softmax_node = graph.node(2).unwrap();
        match softmax_node.op() {
            TraceOp::Softmax { dim } => {
                assert_eq!(*dim, 1, "Softmax dim must be preserved");
            }
            _ => panic!("node 2 must be Softmax"),
        }
    }

    // ========================================================================
    // 23. Multi-output graph — output_nodes returns all marked outputs
    // ========================================================================

    /// Prove: marking multiple outputs preserves all of them in order.
    #[kani::unwind(8)]
    #[kani::proof]
    fn multi_output_preserves_order() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_relu(2, 1, vec![4]),
            make_sigmoid(3, 1, vec![4]),
        ];
        let mut graph = ComputationGraph::from_nodes(nodes);
        // Default output is node 3. Mark node 2 as well.
        graph.mark_output(2);
        let outputs = graph.output_nodes();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].id(), 3, "first output must be node 3 (default)");
        assert_eq!(outputs[1].id(), 2, "second output must be node 2 (marked)");
    }

    // ========================================================================
    // 24. mark_output idempotent — duplicate marks don't add twice
    // ========================================================================

    /// Prove: marking the same node twice does not duplicate it in the output list.
    #[kani::unwind(8)]
    #[kani::proof]
    fn mark_output_idempotent() {
        let nodes = vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])];
        let mut graph = ComputationGraph::from_nodes(nodes);
        // Default output is node 2. Mark it again.
        graph.mark_output(2);
        graph.mark_output(2);
        let outputs = graph.output_nodes();
        assert_eq!(
            outputs.len(),
            1,
            "duplicate marks must not create duplicate outputs"
        );
    }

    // ========================================================================
    // 25. validate_topology — single input-only graph passes
    // ========================================================================

    /// Prove: a graph with only an Input node (no dependencies) passes validation.
    #[kani::unwind(8)]
    #[kani::proof]
    fn validate_topology_single_input_passes() {
        let graph = ComputationGraph::from_nodes(vec![make_input(1, vec![4])]);
        assert!(
            graph.validate_topology().is_ok(),
            "single input-only graph must be valid"
        );
    }

    // ========================================================================
    // 26. input_nodes — empty graph returns no inputs
    // ========================================================================

    /// Prove: an empty graph has no input nodes.
    #[kani::unwind(8)]
    #[kani::proof]
    fn input_nodes_empty_graph() {
        let graph = ComputationGraph::from_nodes(vec![]);
        assert!(graph.input_nodes().is_empty());
    }

    // ========================================================================
    // 27. Graph with MulAdd pattern — complex connectivity validates
    // ========================================================================

    /// Prove: a graph with mul-add pattern (x * y + z) validates correctly
    /// and all nodes are reachable.
    #[kani::unwind(8)]
    #[kani::proof]
    fn graph_mul_add_pattern_validates() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_input(2, vec![4]),
            make_input(3, vec![4]),
            make_mul(4, 1, 2, vec![4]),
            make_add(5, 4, 3, vec![4]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        assert!(graph.validate_topology().is_ok());
        assert_eq!(graph.len(), 5);
        // All nodes findable by ID.
        for id in 1..=5u64 {
            assert!(graph.node(id).is_some(), "node {id} must be findable");
        }
    }

    // ========================================================================
    // 28. Node dtype preservation — F32 and BF16 preserved
    // ========================================================================

    /// Prove: output_dtype is preserved for different DType values.
    #[kani::unwind(1)]
    #[kani::proof]
    fn node_dtype_preserved() {
        let f32_node = TraceNode::new(1, "f32".into(), TraceOp::Input, vec![], vec![4], DType::F32);
        assert_eq!(f32_node.output_dtype(), DType::F32);

        let bf16_node = TraceNode::new(
            2,
            "bf16".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::BF16,
        );
        assert_eq!(bf16_node.output_dtype(), DType::BF16);
    }

    // ========================================================================
    // 29. validate_topology — long chain validates
    // ========================================================================

    /// Prove: a 5-node sequential chain validates successfully.
    #[kani::unwind(8)]
    #[kani::proof]
    fn validate_topology_long_chain() {
        let nodes = vec![
            make_input(1, vec![4]),
            make_relu(2, 1, vec![4]),
            make_sigmoid(3, 2, vec![4]),
            make_tanh(4, 3, vec![4]),
            make_neg(5, 4, vec![4]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        assert!(graph.validate_topology().is_ok());
        assert_eq!(graph.output_node().unwrap().id(), 5);
    }

    // ========================================================================
    // 30. is_ay_compatible_op — shape ops are NOT compatible
    // ========================================================================

    /// Prove: shape-changing ops (Reshape, Transpose) are not ay-compatible.
    #[kani::unwind(1)]
    #[kani::proof]
    fn ay_incompatible_shape_ops() {
        assert!(!is_ay_compatible_op(&TraceOp::Reshape {
            target_shape: vec![8],
        }));
        assert!(!is_ay_compatible_op(&TraceOp::Transpose {
            dim0: 0,
            dim1: 1
        }));
        assert!(!is_ay_compatible_op(&TraceOp::Unsqueeze { dim: 0 }));
        assert!(!is_ay_compatible_op(&TraceOp::Squeeze { dim: 0 }));
    }
}
