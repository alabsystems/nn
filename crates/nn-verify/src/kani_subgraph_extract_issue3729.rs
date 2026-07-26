// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Issue-specific Kani harnesses for `subgraph_extract.rs` (#3729).

#[cfg(kani)]
mod proofs {
    use std::collections::{HashMap, HashSet};

    use kani::assume;
    use nn_core::dyn_tensor::trace::{ComputationGraph, NodeId, TraceNode, TraceOp};
    use nn_core::DType;

    use crate::subgraph_extract::{extract_subgraph, validate_subgraph, SubgraphSpec};

    fn input(id: NodeId) -> TraceNode {
        TraceNode::new(
            id,
            format!("input_{id}"),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        )
    }

    fn add(id: NodeId, lhs: NodeId, rhs: NodeId) -> TraceNode {
        TraceNode::new(
            id,
            format!("add_{id}"),
            TraceOp::Add,
            vec![lhs, rhs],
            vec![4],
            DType::F32,
        )
    }

    fn relu(id: NodeId, input_id: NodeId) -> TraceNode {
        TraceNode::new(
            id,
            format!("relu_{id}"),
            TraceOp::Relu,
            vec![input_id],
            vec![4],
            DType::F32,
        )
    }

    fn neg(id: NodeId, input_id: NodeId) -> TraceNode {
        TraceNode::new(
            id,
            format!("neg_{id}"),
            TraceOp::Neg,
            vec![input_id],
            vec![4],
            DType::F32,
        )
    }

    fn build_graph() -> ComputationGraph {
        ComputationGraph::from_nodes(vec![
            input(1),
            input(2),
            add(3, 1, 2),
            relu(4, 3),
            neg(5, 2),
            add(6, 4, 5),
        ])
    }

    fn selected_range() -> (usize, usize) {
        let choice: u8 = kani::any();
        assume(choice <= 1);
        if choice == 0 {
            (2, 5)
        } else {
            (3, 6)
        }
    }

    fn expected_external_deps(
        graph: &ComputationGraph,
        start: usize,
        end: usize,
    ) -> HashSet<NodeId> {
        let nodes = graph.nodes();
        let selected_ids: HashSet<NodeId> = (start..end).map(|index| nodes[index].id()).collect();
        let mut external = HashSet::new();
        for node in &nodes[start..end] {
            for &input_id in node.inputs() {
                if !selected_ids.contains(&input_id) {
                    external.insert(input_id);
                }
            }
        }
        external
    }

    #[kani::unwind(8)]
    #[kani::proof]
    fn extracted_node_ids_are_dense_and_valid() {
        let graph = build_graph();
        let (start, end) = selected_range();
        let result = extract_subgraph(&graph, &SubgraphSpec::IndexRange { start, end })
            .expect("valid extraction");
        let nodes = result.graph.nodes();

        assert!(validate_subgraph(&result.graph).is_ok());
        assert_eq!(
            nodes.len(),
            result.layer_count + result.synthetic_input_count
        );
        assert_eq!(result.id_map.len(), nodes.len());

        let mut seen = vec![false; nodes.len() + 1];
        for node in nodes {
            let id = node.id() as usize;
            assert!(id >= 1);
            assert!(id <= nodes.len());
            assert!(!seen[id], "remapped ids must be unique");
            seen[id] = true;
        }

        for id in 1..=nodes.len() {
            assert!(seen[id], "remapped ids must cover 1..=node_count");
        }
    }

    #[kani::unwind(8)]
    #[kani::proof]
    fn synthetic_inputs_match_external_dependencies() {
        let graph = build_graph();
        let (start, end) = selected_range();
        let expected_external = expected_external_deps(&graph, start, end);
        let result = extract_subgraph(&graph, &SubgraphSpec::IndexRange { start, end })
            .expect("valid extraction");

        assert_eq!(result.synthetic_input_count, expected_external.len());
        for external_id in expected_external {
            assert!(
                result.id_map.contains_key(&external_id),
                "every external dependency must have a synthetic input"
            );
        }
    }

    #[kani::unwind(8)]
    #[kani::proof]
    fn remapped_edges_preserve_original_edge_subset() {
        let graph = build_graph();
        let (start, end) = selected_range();
        let result = extract_subgraph(&graph, &SubgraphSpec::IndexRange { start, end })
            .expect("valid extraction");

        let inverse_map: HashMap<NodeId, NodeId> = result
            .id_map
            .iter()
            .map(|(&original, &remapped)| (remapped, original))
            .collect();

        for node in &result.graph.nodes()[..result.synthetic_input_count] {
            assert!(matches!(node.op(), TraceOp::Input));
            assert!(node.inputs().is_empty());
        }

        for node in &result.graph.nodes()[result.synthetic_input_count..] {
            let original_id = *inverse_map.get(&node.id()).expect("inverse id map");
            let original = graph.node(original_id).expect("original node");

            assert_eq!(node.inputs().len(), original.inputs().len());
            for &remapped_input in node.inputs() {
                let original_input = inverse_map
                    .get(&remapped_input)
                    .expect("all remapped inputs must resolve");
                assert!(
                    original.inputs().contains(original_input),
                    "every extracted edge must correspond to an original edge"
                );
            }
        }
    }
}
