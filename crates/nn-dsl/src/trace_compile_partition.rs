// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! DAG-aware graph partition algorithm for global fusion.
//!
//! Partitions a `ComputationGraph` into fusion groups where each group
//! becomes a single GPU dispatch. Uses reverse-topological consumer-driven
//! clustering: each node tries to merge into its producer's partition
//! if the fusion rules allow it and no cycle would be created.
//!
//! This replaces the sequential peephole approach for elementwise fusion,
//! enabling optimizations that peephole passes cannot find (non-adjacent
//! fusible ops, fan-out > 1 within a group, reduction absorption).
//!
//! References:
//! - `designs/2026-03-22-graph-partitioning-beyond-peephole.md`
//! - `designs/2026-03-25-tinygrad-study.md` Section 5.1

use std::collections::{HashMap, HashSet};

use nn_core::dyn_tensor::trace::{ComputationGraph, NodeId};

use super::trace_compile_classify::{can_fuse, fusion_category, FusionCategory};

/// A partition group: a set of graph nodes that will be compiled into
/// a single GPU dispatch.
#[derive(Debug)]
pub(crate) struct PartitionGroup {
    /// Indices into the `ComputationGraph` node list, in topological order.
    pub(crate) nodes: Vec<usize>,
    /// Dominant fusion category of this group (determines codegen strategy).
    pub(crate) category: FusionCategory,
}

/// Result of graph partitioning.
#[derive(Debug)]
pub(crate) struct PartitionResult {
    /// Partition groups in topological order (by first node in each group).
    pub(crate) groups: Vec<PartitionGroup>,
    /// Cross-partition edge DAG: `group_edge_map[i]` is the set of group
    /// indices that group `i` depends on (its input producers).
    pub(crate) group_edge_map: Vec<Vec<usize>>,
    /// For diagnostics: number of dispatches before partition (= node count
    /// minus inputs/constants/passthroughs).
    pub(crate) pre_partition_dispatches: usize,
    /// For diagnostics: number of dispatches after partition (= group count).
    pub(crate) post_partition_dispatches: usize,
}

/// Union-find (disjoint set) for partition merging.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path splitting
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }

    fn same(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }
}

/// Partition a `ComputationGraph` into fusion groups.
///
/// Algorithm: reverse-topological consumer-driven clustering.
/// 1. Classify each node by `FusionCategory`.
/// 2. Build adjacency (producer → consumer edges).
/// 3. Walk nodes in reverse topological order.
/// 4. For each node, try to merge with each producer's partition.
/// 5. Merge allowed if `can_fuse(producer_cat, consumer_cat)` AND
///    merging would not create a cycle in the partition DAG.
/// 6. Collect final groups in topological order.
pub(crate) fn partition_graph(graph: &ComputationGraph) -> PartitionResult {
    let n = graph.len();
    if n == 0 {
        return PartitionResult {
            groups: vec![],
            group_edge_map: vec![],
            pre_partition_dispatches: 0,
            post_partition_dispatches: 0,
        };
    }

    let nodes = graph.nodes();

    // Step 1: Classify each node.
    let categories: Vec<FusionCategory> = nodes.iter().map(|n| fusion_category(n.op())).collect();

    // Step 2: Build id → index map and producer → consumer edges.
    let id_to_idx: HashMap<NodeId, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id(), i)).collect();

    let mut producers_of: Vec<Vec<usize>> = vec![vec![]; n];
    for (idx, node) in nodes.iter().enumerate() {
        for &input_id in node.inputs() {
            if let Some(&input_idx) = id_to_idx.get(&input_id) {
                producers_of[idx].push(input_idx);
            }
        }
    }

    // Step 3: Initialize union-find (each node in its own partition).
    let mut uf = UnionFind::new(n);

    // Partition-level dependency DAG for efficient cycle detection.
    // `partition_deps[root]` = set of partition roots that `root` depends on.
    // `partition_rdeps[root]` = set of partition roots that depend on `root`.
    let mut partition_deps: Vec<HashSet<usize>> = (0..n).map(|_| HashSet::new()).collect();
    let mut partition_rdeps: Vec<HashSet<usize>> = (0..n).map(|_| HashSet::new()).collect();

    // Initialize partition-level edges from node-level edges.
    for (idx, node) in nodes.iter().enumerate() {
        for &input_id in node.inputs() {
            if let Some(&input_idx) = id_to_idx.get(&input_id) {
                if idx != input_idx {
                    partition_deps[idx].insert(input_idx);
                    partition_rdeps[input_idx].insert(idx);
                }
            }
        }
    }

    // Step 4: Walk in reverse topological order (nodes are already topo-sorted).
    for consumer_idx in (0..n).rev() {
        let consumer_cat = categories[consumer_idx];
        if consumer_cat == FusionCategory::Native {
            continue; // Native ops don't participate in fusion.
        }

        for &producer_idx in &producers_of[consumer_idx] {
            let producer_cat = categories[producer_idx];

            // Already in the same partition.
            if uf.same(producer_idx, consumer_idx) {
                continue;
            }

            // Check fusion rules.
            if !can_fuse(producer_cat, consumer_cat) {
                continue;
            }

            let producer_root = uf.find(producer_idx);
            let consumer_root = uf.find(consumer_idx);

            // Cycle check using partition-level DAG: merging creates a cycle
            // if the consumer's partition depends on a third partition that
            // depends on the producer's partition. Equivalently: any partition
            // reachable from consumer (excluding producer) that can also reach
            // producer would form a cycle. Conservative check: consumer depends
            // on third AND third depends on producer.
            let would_cycle = partition_deps[consumer_root].iter().any(|&dep_root| {
                dep_root != producer_root
                    && dep_root != consumer_root
                    && partition_deps[dep_root].contains(&producer_root)
            });

            if would_cycle {
                continue;
            }

            // Merge partitions in union-find.
            uf.union(producer_idx, consumer_idx);
            let new_root = uf.find(producer_idx);
            let old_root = if new_root == producer_root {
                consumer_root
            } else {
                producer_root
            };

            // Merge partition-level edges: new_root absorbs old_root's edges.
            if new_root != old_root {
                let old_deps: Vec<usize> = partition_deps[old_root].iter().copied().collect();
                let old_rdeps: Vec<usize> = partition_rdeps[old_root].iter().copied().collect();

                // Transfer old_root's dependencies to new_root.
                for dep in &old_deps {
                    if *dep != new_root && *dep != old_root {
                        partition_deps[new_root].insert(*dep);
                        partition_rdeps[*dep].remove(&old_root);
                        partition_rdeps[*dep].insert(new_root);
                    }
                }
                // Transfer old_root's reverse dependencies to new_root.
                for rdep in &old_rdeps {
                    if *rdep != new_root && *rdep != old_root {
                        partition_rdeps[new_root].insert(*rdep);
                        partition_deps[*rdep].remove(&old_root);
                        partition_deps[*rdep].insert(new_root);
                    }
                }
                // Remove self-edges.
                partition_deps[new_root].remove(&new_root);
                partition_deps[new_root].remove(&old_root);
                partition_rdeps[new_root].remove(&new_root);
                partition_rdeps[new_root].remove(&old_root);

                // Clear old_root.
                partition_deps[old_root].clear();
                partition_rdeps[old_root].clear();
            }
        }
    }

    // Step 5: Collect partition groups.
    let mut group_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = uf.find(i);
        group_map.entry(root).or_default().push(i);
    }

    // Sort groups by their first (lowest-index) node for topological order.
    let mut groups: Vec<PartitionGroup> = group_map
        .into_values()
        .map(|mut nodes| {
            nodes.sort_unstable();
            let dominant_cat = dominant_category(&nodes, &categories);
            PartitionGroup {
                nodes,
                category: dominant_cat,
            }
        })
        .collect();
    groups.sort_by_key(|g| g.nodes[0]);

    // Build group_edge_map: translate partition_deps (root-indexed) into
    // group-indexed form for downstream codegen. Part of #4283.
    let root_to_group: HashMap<usize, usize> = groups
        .iter()
        .enumerate()
        .map(|(gi, g)| (uf.find(g.nodes[0]), gi))
        .collect();
    let group_edge_map: Vec<Vec<usize>> = groups
        .iter()
        .map(|g| {
            let root = uf.find(g.nodes[0]);
            let mut deps: Vec<usize> = partition_deps[root]
                .iter()
                .filter_map(|&dep_root| root_to_group.get(&dep_root).copied())
                .collect();
            deps.sort_unstable();
            deps.dedup();
            deps
        })
        .collect();

    // Diagnostics.
    let pre_dispatch = categories
        .iter()
        .filter(|c| **c != FusionCategory::Native)
        .count();
    let post_dispatch = groups.len();

    PartitionResult {
        groups,
        group_edge_map,
        pre_partition_dispatches: pre_dispatch,
        post_partition_dispatches: post_dispatch,
    }
}

/// Determine the dominant category of a partition group.
///
/// Priority: Opaque > Reduction > Elementwise > Broadcast > Native.
/// The dominant category determines the codegen strategy for the group.
fn dominant_category(nodes: &[usize], categories: &[FusionCategory]) -> FusionCategory {
    let mut has_opaque = false;
    let mut has_reduction = false;
    let mut has_elementwise = false;

    for &idx in nodes {
        match categories[idx] {
            FusionCategory::Opaque => has_opaque = true,
            FusionCategory::Reduction => has_reduction = true,
            FusionCategory::Elementwise => has_elementwise = true,
            _ => {}
        }
    }

    if has_opaque {
        FusionCategory::Opaque
    } else if has_reduction {
        FusionCategory::Reduction
    } else if has_elementwise {
        FusionCategory::Elementwise
    } else {
        FusionCategory::Broadcast
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn_core::dyn_tensor::trace::{TraceNode, TraceOp, WeightRef};
    use nn_core::DType;

    fn build_graph(nodes: Vec<(TraceOp, Vec<usize>, Vec<usize>)>) -> ComputationGraph {
        let mut trace_nodes = Vec::new();
        for (i, (op, input_indices, shape)) in nodes.into_iter().enumerate() {
            let id = i as u64;
            let input_ids: Vec<u64> = input_indices
                .into_iter()
                .map(|idx: usize| idx as u64)
                .collect();
            let node = TraceNode::new(id, format!("node_{i}"), op, input_ids, shape, DType::F32);
            trace_nodes.push(node);
        }
        ComputationGraph::from_nodes(trace_nodes)
    }

    #[test]
    fn test_empty_graph() {
        let graph = ComputationGraph::from_nodes(vec![]);
        let result = partition_graph(&graph);
        assert_eq!(result.groups.len(), 0);
        assert!(result.group_edge_map.is_empty());
    }

    #[test]
    fn test_single_elementwise_chain() {
        // Input -> Relu -> Exp -> Sigmoid (should fuse Relu+Exp+Sigmoid)
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 64]),
            (TraceOp::Relu, vec![0], vec![1, 64]),
            (TraceOp::Exp, vec![1], vec![1, 64]),
            (TraceOp::Sigmoid, vec![2], vec![1, 64]),
        ]);
        let result = partition_graph(&graph);
        // Input is Native (own group), Relu+Exp+Sigmoid fuse into one.
        assert_eq!(result.post_partition_dispatches, 2);
        // group_edge_map: group 0 (Input) has no deps, group 1 (fused) depends on group 0.
        assert_eq!(result.group_edge_map.len(), 2);
        assert!(result.group_edge_map[0].is_empty());
        assert_eq!(result.group_edge_map[1], vec![0]);
    }

    #[test]
    fn test_opaque_with_epilogue() {
        // Input -> Linear -> Relu (Linear+Relu should fuse as opaque+epilogue)
        let w = WeightRef::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 2]),
            (
                TraceOp::Linear {
                    weight: w,
                    bias: None,
                },
                vec![0],
                vec![1, 2],
            ),
            (TraceOp::Relu, vec![1], vec![1, 2]),
        ]);
        let result = partition_graph(&graph);
        // Input (Native) + Linear+Relu (fused Opaque) = 2 groups.
        assert_eq!(result.post_partition_dispatches, 2);
    }

    #[test]
    fn test_opaque_barrier() {
        // Input -> Linear -> Linear (two Linears should NOT fuse)
        let w = WeightRef::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 2]),
            (
                TraceOp::Linear {
                    weight: w.clone(),
                    bias: None,
                },
                vec![0],
                vec![1, 2],
            ),
            (
                TraceOp::Linear {
                    weight: w,
                    bias: None,
                },
                vec![1],
                vec![1, 2],
            ),
        ]);
        let result = partition_graph(&graph);
        // Input + Linear + Linear = 3 groups (opaque barrier).
        assert_eq!(result.post_partition_dispatches, 3);
        // group_edge_map: group 0 (Input) no deps, group 1 (Linear) depends on 0,
        // group 2 (Linear) depends on 1.
        assert_eq!(result.group_edge_map.len(), 3);
        assert!(result.group_edge_map[0].is_empty());
        assert_eq!(result.group_edge_map[1], vec![0]);
        assert_eq!(result.group_edge_map[2], vec![1]);
    }

    #[test]
    fn test_reduction_absorbs_elementwise() {
        // Input -> Mul -> ReduceSum (Mul+ReduceSum should fuse)
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 64]),
            (TraceOp::Mul, vec![0, 0], vec![1, 64]),
            (
                TraceOp::ReduceSum {
                    dim: 1,
                    keepdim: false,
                },
                vec![1],
                vec![1],
            ),
        ]);
        let result = partition_graph(&graph);
        // Input (Native) + Mul+ReduceSum (fused) = 2 groups.
        assert_eq!(result.post_partition_dispatches, 2);
    }

    #[test]
    fn test_broadcast_absorbed() {
        // Input -> Reshape -> Relu (Reshape is zero-cost, should absorb)
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 64]),
            (
                TraceOp::Reshape {
                    target_shape: vec![64],
                },
                vec![0],
                vec![64],
            ),
            (TraceOp::Relu, vec![1], vec![64]),
        ]);
        let result = partition_graph(&graph);
        // Input (Native) + Reshape+Relu (fused) = 2 groups.
        assert_eq!(result.post_partition_dispatches, 2);
    }
}
