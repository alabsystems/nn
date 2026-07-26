// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! DAG-aware partition compiler for compiled execution plans.
//!
//! Operates on a [`CompiledPlan`] (post-compilation) to find global fusion
//! opportunities that the sequential peephole and chain-fusion passes miss.
//! Builds an explicit dependency DAG from `external_node_ids` and the
//! plan's edge map, then identifies [`FusionGroup`]s — sets of dispatch
//! steps that can be merged into a single GPU kernel launch.
//!
//! Primary fusion candidates: consecutive or non-adjacent elementwise
//! dispatches on same-shaped tensors with compatible data dependencies.
//!
//! Part of #4275 (Graph-global fusion — DAG-aware partition compiler).

use std::collections::BTreeSet;
use std::fmt;

use crate::trace_compile::{CompiledPlan, CompiledStep};

// ---------------------------------------------------------------------------
// Node classification
// ---------------------------------------------------------------------------

/// Classification of a compiled step for DAG fusion decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DagNodeKind {
    /// Elementwise dispatch: primary fusion candidate.
    Elementwise,
    /// Non-elementwise dispatch (matmul, conv, norm, etc.): fusion barrier.
    OpaqueDispatch,
    /// NativeOp (pre-fused): fusion barrier.
    NativeOp,
    /// Non-dispatch step (Passthrough, InputForward, etc.): not fusible.
    Metadata,
}

/// Returns `true` for kernel names that indicate a fusible elementwise op.
///
/// Heuristic: kernels named after known elementwise ops or already-fused
/// elementwise chains (`fused_*_xN`). Excludes matmul, conv, norm, linear,
/// softmax, etc.
fn is_elementwise_kernel_name(name: &str) -> bool {
    // Already-fused elementwise chains are still elementwise.
    if name.starts_with("fused_") {
        return true;
    }
    matches!(
        name,
        "relu"
            | "gelu"
            | "gelu_erf"
            | "sigmoid"
            | "silu"
            | "silu_mul"
            | "tanh"
            | "exp"
            | "log"
            | "sqrt"
            | "sqr"
            | "abs"
            | "neg"
            | "recip"
            | "sin"
            | "cos"
            | "floor"
            | "ceil"
            | "round"
            | "sign"
            | "fract"
            | "add"
            | "sub"
            | "mul"
            | "div"
            | "maximum"
            | "minimum"
            | "atan2"
            | "leaky_relu"
            | "elu"
            | "celu"
            | "prelu"
            | "clamp"
            | "powf"
            | "softplus"
            | "selu"
            | "mish"
            | "hard_sigmoid"
            | "hard_swish"
            | "softsign"
            | "where_cond"
            | "compare"
    )
}

/// Classify a `CompiledStep` for DAG fusion decisions.
fn classify_step(step: &CompiledStep) -> DagNodeKind {
    match step {
        CompiledStep::Dispatch { kernel, .. } => {
            if is_elementwise_kernel_name(kernel.name()) {
                DagNodeKind::Elementwise
            } else {
                DagNodeKind::OpaqueDispatch
            }
        }
        CompiledStep::NativeOp { .. } => DagNodeKind::NativeOp,
        CompiledStep::Passthrough { .. }
        | CompiledStep::NarrowView { .. }
        | CompiledStep::InputForward
        | CompiledStep::IdentityPassthrough
        | CompiledStep::ConstantValue { .. }
        | CompiledStep::RuntimeOp { .. } => DagNodeKind::Metadata,
    }
}

// ---------------------------------------------------------------------------
// DAG types
// ---------------------------------------------------------------------------

/// A node in the partition DAG, representing one step in the compiled plan.
#[derive(Debug, Clone)]
pub struct DagNode {
    /// Index into `CompiledPlan.steps`.
    pub step_index: usize,
    /// Classification for fusion decisions.
    pub kind: DagNodeKind,
    /// Output shape of this step (empty for non-dispatch steps).
    pub output_shape: Vec<usize>,
    /// Kernel name (empty string for non-dispatch steps).
    pub kernel_name: String,
}

/// Dependency DAG over compiled plan steps.
///
/// Each node corresponds to one `CompiledStep`. Edges encode data
/// dependencies: `predecessors[i]` lists the step indices that step `i`
/// reads from, and `successors[i]` lists steps that read step `i`'s output.
#[derive(Debug, Clone)]
pub struct PartitionDag {
    /// Per-step DAG nodes with metadata.
    pub nodes: Vec<DagNode>,
    /// `predecessors[i]` = set of step indices that step `i` depends on.
    pub predecessors: Vec<BTreeSet<usize>>,
    /// `successors[i]` = set of step indices that depend on step `i`.
    pub successors: Vec<BTreeSet<usize>>,
}

/// A fusion group: a set of dispatch steps that can be merged into a
/// single GPU kernel launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionGroup {
    /// Step indices in topological order.
    pub steps: Vec<usize>,
    /// Shared output shape of all steps in the group.
    pub shape: Vec<usize>,
    /// Number of GPU dispatches saved by fusing this group (steps.len() - 1).
    pub dispatches_saved: usize,
}

impl fmt::Display for FusionGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FusionGroup({} steps, shape {:?}, saves {} dispatches)",
            self.steps.len(),
            self.shape,
            self.dispatches_saved,
        )
    }
}

/// Summary of partition analysis results.
#[derive(Debug, Clone)]
pub struct PartitionSummary {
    /// Total dispatch steps in the plan.
    pub total_dispatches: usize,
    /// Number of elementwise dispatch steps.
    pub elementwise_dispatches: usize,
    /// Number of fusion groups found.
    pub fusion_groups: usize,
    /// Total dispatches that could be saved by fusing all groups.
    pub total_savings: usize,
    /// Theoretical minimum dispatches after fusion.
    pub theoretical_minimum: usize,
}

impl fmt::Display for PartitionSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PartitionSummary: {} dispatches ({} elementwise), {} fusion groups, \
             {} savings -> {} theoretical minimum",
            self.total_dispatches,
            self.elementwise_dispatches,
            self.fusion_groups,
            self.total_savings,
            self.theoretical_minimum,
        )
    }
}

// ---------------------------------------------------------------------------
// DAG construction
// ---------------------------------------------------------------------------

/// Build a dependency DAG from a compiled plan.
///
/// Uses `external_node_ids` on Dispatch and NativeOp steps to determine
/// data dependencies. For steps without `external_node_ids`, falls back
/// to positional heuristic (each step depends on the immediately preceding
/// dispatch step).
///
/// # Arguments
///
/// * `plan` - The compiled execution plan.
/// * `graph_edge_map` - Optional pre-computed edge map from `compute_edge_map()`.
///   When `Some`, used as the primary dependency source. When `None`,
///   dependencies are inferred from `external_node_ids` and positional
///   heuristic.
#[must_use]
pub fn partition_plan(plan: &CompiledPlan, graph_edge_map: Option<&[Vec<usize>]>) -> PartitionDag {
    let n = plan.steps.len();
    let mut nodes = Vec::with_capacity(n);
    let mut predecessors: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    let mut successors: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];

    // Build a NodeId -> step_index map for resolving external_node_ids.
    // external_node_ids are graph NodeIds; step indices correspond 1:1 with
    // graph node indices in the standard compilation path.
    // We also need to handle the case where external_node_ids reference
    // node IDs that map to earlier steps.

    // Phase 1: Build node metadata.
    for (i, step) in plan.steps.iter().enumerate() {
        let kind = classify_step(step);
        let (output_shape, kernel_name) = match step {
            CompiledStep::Dispatch { kernel, .. } => (
                kernel
                    .output_shape()
                    .map(<[usize]>::to_vec)
                    .unwrap_or_default(),
                kernel.name().to_string(),
            ),
            CompiledStep::Passthrough { output_shape, .. } => (output_shape.clone(), String::new()),
            CompiledStep::NarrowView { output_shape, .. } => (output_shape.clone(), String::new()),
            CompiledStep::ConstantValue { shape, .. } => (shape.clone(), String::new()),
            _ => (vec![], String::new()),
        };
        nodes.push(DagNode {
            step_index: i,
            kind,
            output_shape,
            kernel_name,
        });
    }

    // Phase 2: Build edges.
    if let Some(edge_map) = graph_edge_map {
        // Use pre-computed edge map directly.
        for (i, deps) in edge_map.iter().enumerate() {
            if i >= n {
                break;
            }
            for &dep in deps {
                if dep < n && dep != i {
                    predecessors[i].insert(dep);
                    successors[dep].insert(i);
                }
            }
        }
    } else {
        // Infer edges from external_node_ids and positional heuristic.
        for (i, step) in plan.steps.iter().enumerate() {
            let ext_ids: Option<&[u64]> = match step {
                CompiledStep::Dispatch {
                    external_node_ids: Some(ids),
                    ..
                } => Some(ids.as_slice()),
                CompiledStep::NativeOp { op, .. } => op.external_node_ids(),
                _ => None,
            };

            if let Some(ids) = ext_ids {
                // external_node_ids are graph NodeIds. In the standard
                // compilation path, step index == graph node index, so
                // NodeId N maps to step index N (if N < plan.steps.len()).
                for &node_id in ids {
                    let dep_idx = node_id as usize;
                    if dep_idx < n && dep_idx != i {
                        predecessors[i].insert(dep_idx);
                        successors[dep_idx].insert(i);
                    }
                }
            } else if i > 0 {
                // Positional heuristic: depend on the nearest preceding
                // dispatch/native step. This is conservative but captures
                // the common sequential pipeline pattern.
                for j in (0..i).rev() {
                    let prev_kind = classify_step(&plan.steps[j]);
                    if prev_kind == DagNodeKind::Elementwise
                        || prev_kind == DagNodeKind::OpaqueDispatch
                        || prev_kind == DagNodeKind::NativeOp
                    {
                        predecessors[i].insert(j);
                        successors[j].insert(i);
                        break;
                    }
                }
            }
        }
    }

    PartitionDag {
        nodes,
        predecessors,
        successors,
    }
}

// ---------------------------------------------------------------------------
// Fusion group detection
// ---------------------------------------------------------------------------

/// Find fusion groups in a partition DAG.
///
/// A fusion group is a maximal set of elementwise dispatch steps where:
/// 1. All steps have the same output shape.
/// 2. Dependencies within the group form a chain (no external consumers
///    between group members, except for the last step).
/// 3. Each intermediate step in the group has exactly one consumer that
///    is also in the group (fan-out = 1 within the fusion boundary).
///
/// Returns groups sorted by their first step index.
#[must_use]
pub fn find_fusion_groups(dag: &PartitionDag) -> Vec<FusionGroup> {
    let n = dag.nodes.len();
    let mut visited = vec![false; n];
    let mut groups = Vec::new();

    // Walk steps in topological order (they are already ordered by index).
    for i in 0..n {
        if visited[i] {
            continue;
        }
        if dag.nodes[i].kind != DagNodeKind::Elementwise {
            continue;
        }
        if dag.nodes[i].output_shape.is_empty() {
            continue;
        }

        // Try to grow a chain starting from this node.
        let mut chain = vec![i];
        visited[i] = true;
        let target_shape = &dag.nodes[i].output_shape;

        // Extend forward: find successors that are elementwise with same shape
        // and where the current tail has fan-out = 1 (only one consumer).
        let mut current = i;
        loop {
            // The current step must have exactly one successor that is:
            // (a) elementwise, (b) same shape, (c) not yet visited.
            let candidates: Vec<usize> = dag.successors[current]
                .iter()
                .copied()
                .filter(|&s| {
                    !visited[s]
                        && dag.nodes[s].kind == DagNodeKind::Elementwise
                        && dag.nodes[s].output_shape == *target_shape
                })
                .collect();

            // Fan-out check: if the current step has other consumers outside
            // the elementwise-same-shape set, we cannot fuse further because
            // the intermediate result is needed externally.
            let total_successors = dag.successors[current].len();
            if candidates.len() != 1 || total_successors > 1 {
                break;
            }

            let next = candidates[0];
            chain.push(next);
            visited[next] = true;
            current = next;
        }

        // Only report groups with 2+ steps (1 step = nothing to fuse).
        if chain.len() >= 2 {
            let savings = chain.len() - 1;
            groups.push(FusionGroup {
                steps: chain,
                shape: target_shape.clone(),
                dispatches_saved: savings,
            });
        }
    }

    groups.sort_by_key(|g| g.steps[0]);
    groups
}

/// Compute a summary of the partition analysis.
#[must_use]
pub fn partition_summary(dag: &PartitionDag, groups: &[FusionGroup]) -> PartitionSummary {
    let total_dispatches = dag
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                n.kind,
                DagNodeKind::Elementwise | DagNodeKind::OpaqueDispatch | DagNodeKind::NativeOp
            )
        })
        .count();

    let elementwise_dispatches = dag
        .nodes
        .iter()
        .filter(|n| n.kind == DagNodeKind::Elementwise)
        .count();

    let total_savings: usize = groups.iter().map(|g| g.dispatches_saved).sum();
    let theoretical_minimum = total_dispatches.saturating_sub(total_savings);

    PartitionSummary {
        total_dispatches,
        elementwise_dispatches,
        fusion_groups: groups.len(),
        total_savings,
        theoretical_minimum,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
    use crate::trace_compile::{CompiledKernel, CompiledPlan, CompiledStep};
    use std::collections::HashMap;

    /// Build a Dispatch step with a given kernel name and output shape.
    fn make_dispatch(name: &str, shape: &[usize]) -> CompiledStep {
        let output_id = TensorNodeId::new(0);
        let node = TensorNode::new(
            output_id,
            TensorOpKind::Input {
                name: "x".to_string(),
                shape: shape.to_vec(),
            },
            shape.to_vec(),
        );
        let def = TensorKernelDef {
            name: name.to_string(),
            nodes: vec![node],
            output: output_id,
        };
        CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: HashMap::new(),
            external_node_ids: None,
        }
    }

    /// Build a Dispatch step with external_node_ids.
    fn make_dispatch_with_deps(name: &str, shape: &[usize], ext_ids: Vec<u64>) -> CompiledStep {
        let output_id = TensorNodeId::new(0);
        let node = TensorNode::new(
            output_id,
            TensorOpKind::Input {
                name: "x".to_string(),
                shape: shape.to_vec(),
            },
            shape.to_vec(),
        );
        let def = TensorKernelDef {
            name: name.to_string(),
            nodes: vec![node],
            output: output_id,
        };
        CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: HashMap::new(),
            external_node_ids: Some(ext_ids),
        }
    }

    fn make_plan(steps: Vec<CompiledStep>) -> CompiledPlan {
        let output_step = if steps.is_empty() { 0 } else { steps.len() - 1 };
        CompiledPlan {
            steps,
            input_shapes: vec![],
            output_step,
            weight_names: vec![],
        }
    }

    // -- DAG construction tests -----------------------------------------------

    #[test]
    fn test_partition_empty_plan() {
        let plan = make_plan(vec![]);
        let dag = partition_plan(&plan, None);
        assert!(dag.nodes.is_empty());
        assert!(dag.predecessors.is_empty());
        assert!(dag.successors.is_empty());
    }

    #[test]
    fn test_partition_single_dispatch() {
        let plan = make_plan(vec![make_dispatch("relu", &[1, 64])]);
        let dag = partition_plan(&plan, None);
        assert_eq!(dag.nodes.len(), 1);
        assert_eq!(dag.nodes[0].kind, DagNodeKind::Elementwise);
        assert_eq!(dag.nodes[0].output_shape, vec![1, 64]);
        assert!(dag.predecessors[0].is_empty());
        assert!(dag.successors[0].is_empty());
    }

    #[test]
    fn test_partition_dag_from_external_node_ids() {
        // Step 0: InputForward
        // Step 1: relu (depends on step 0)
        // Step 2: exp (depends on step 1)
        // Step 3: sigmoid (depends on step 2)
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            make_dispatch_with_deps("relu", &[1, 64], vec![0]),
            make_dispatch_with_deps("exp", &[1, 64], vec![1]),
            make_dispatch_with_deps("sigmoid", &[1, 64], vec![2]),
        ]);
        let dag = partition_plan(&plan, None);

        assert_eq!(dag.nodes.len(), 4);
        // Step 0 (InputForward) has no predecessors.
        assert!(dag.predecessors[0].is_empty());
        // Step 1 depends on step 0.
        assert!(dag.predecessors[1].contains(&0));
        // Step 2 depends on step 1.
        assert!(dag.predecessors[2].contains(&1));
        // Step 3 depends on step 2.
        assert!(dag.predecessors[3].contains(&2));
        // Successor edges are symmetric.
        assert!(dag.successors[0].contains(&1));
        assert!(dag.successors[1].contains(&2));
        assert!(dag.successors[2].contains(&3));
    }

    #[test]
    fn test_partition_dag_from_edge_map() {
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            make_dispatch("relu", &[1, 64]),
            make_dispatch("exp", &[1, 64]),
            make_dispatch("sigmoid", &[1, 64]),
        ]);
        let edge_map = vec![
            vec![],  // step 0: no deps
            vec![0], // step 1: depends on step 0
            vec![1], // step 2: depends on step 1
            vec![2], // step 3: depends on step 2
        ];
        let dag = partition_plan(&plan, Some(&edge_map));

        assert!(dag.predecessors[1].contains(&0));
        assert!(dag.predecessors[2].contains(&1));
        assert!(dag.predecessors[3].contains(&2));
    }

    #[test]
    fn test_partition_dag_positional_fallback() {
        // No external_node_ids, no edge_map -> positional heuristic.
        let plan = make_plan(vec![
            make_dispatch("relu", &[1, 64]),
            make_dispatch("exp", &[1, 64]),
            make_dispatch("sigmoid", &[1, 64]),
        ]);
        let dag = partition_plan(&plan, None);

        // Positional heuristic: each step depends on the previous dispatch.
        assert!(dag.predecessors[1].contains(&0));
        assert!(dag.predecessors[2].contains(&1));
    }

    #[test]
    fn test_partition_node_classification() {
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            make_dispatch("relu", &[1, 64]),
            make_dispatch("linear", &[1, 64]),
            CompiledStep::Passthrough {
                op_name: "reshape".into(),
                output_shape: vec![64],
            },
        ]);
        let dag = partition_plan(&plan, None);

        assert_eq!(dag.nodes[0].kind, DagNodeKind::Metadata);
        assert_eq!(dag.nodes[1].kind, DagNodeKind::Elementwise);
        assert_eq!(dag.nodes[2].kind, DagNodeKind::OpaqueDispatch);
        assert_eq!(dag.nodes[3].kind, DagNodeKind::Metadata);
    }

    // -- Fusion group detection tests -----------------------------------------

    #[test]
    fn test_find_fusion_groups_elementwise_chain() {
        // Input -> relu -> exp -> sigmoid: all elementwise, same shape.
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            make_dispatch_with_deps("relu", &[1, 64], vec![0]),
            make_dispatch_with_deps("exp", &[1, 64], vec![1]),
            make_dispatch_with_deps("sigmoid", &[1, 64], vec![2]),
        ]);
        let dag = partition_plan(&plan, None);
        let groups = find_fusion_groups(&dag);

        assert_eq!(groups.len(), 1, "expected 1 fusion group");
        assert_eq!(groups[0].steps, vec![1, 2, 3]);
        assert_eq!(groups[0].shape, vec![1, 64]);
        assert_eq!(groups[0].dispatches_saved, 2);
    }

    #[test]
    fn test_find_fusion_groups_respects_shape_mismatch() {
        // relu [1, 64] -> matmul [1, 32] -> exp [1, 32]
        // relu and matmul have different shapes, so no fusion across them.
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            make_dispatch_with_deps("relu", &[1, 64], vec![0]),
            make_dispatch_with_deps("exp", &[1, 32], vec![1]),
        ]);
        let dag = partition_plan(&plan, None);
        let groups = find_fusion_groups(&dag);

        // Shape mismatch: no fusion group.
        assert!(
            groups.is_empty(),
            "expected no fusion groups due to shape mismatch"
        );
    }

    #[test]
    fn test_find_fusion_groups_respects_opaque_barrier() {
        // relu -> linear -> exp: linear is OpaqueDispatch, breaks the chain.
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            make_dispatch_with_deps("relu", &[1, 64], vec![0]),
            make_dispatch_with_deps("linear", &[1, 64], vec![1]),
            make_dispatch_with_deps("exp", &[1, 64], vec![2]),
        ]);
        let dag = partition_plan(&plan, None);
        let groups = find_fusion_groups(&dag);

        // linear is OpaqueDispatch, breaks the chain. No group of size >= 2.
        assert!(
            groups.is_empty(),
            "expected no fusion groups across opaque barrier"
        );
    }

    #[test]
    fn test_find_fusion_groups_respects_fan_out() {
        // relu -> (exp AND sigmoid): fan-out from relu blocks fusion.
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            make_dispatch_with_deps("relu", &[1, 64], vec![0]),
            make_dispatch_with_deps("exp", &[1, 64], vec![1]),
            make_dispatch_with_deps("sigmoid", &[1, 64], vec![1]), // also depends on relu
        ]);
        let dag = partition_plan(&plan, None);
        let groups = find_fusion_groups(&dag);

        // relu has fan-out = 2 (both exp and sigmoid depend on it).
        // Cannot fuse relu with either because the intermediate is needed.
        assert!(groups.is_empty(), "expected no fusion groups with fan-out");
    }

    #[test]
    fn test_find_fusion_groups_multiple_chains() {
        // Two independent elementwise chains.
        // Chain 1: steps 1->2 (relu -> exp)
        // Chain 2: steps 3->4 (sigmoid -> tanh)
        let edge_map = vec![
            vec![],  // step 0: input
            vec![0], // step 1: relu <- input
            vec![1], // step 2: exp <- relu
            vec![0], // step 3: sigmoid <- input
            vec![3], // step 4: tanh <- sigmoid
        ];
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            make_dispatch("relu", &[1, 64]),
            make_dispatch("exp", &[1, 64]),
            make_dispatch("sigmoid", &[1, 64]),
            make_dispatch("tanh", &[1, 64]),
        ]);
        let dag = partition_plan(&plan, Some(&edge_map));
        let groups = find_fusion_groups(&dag);

        assert_eq!(groups.len(), 2, "expected 2 independent fusion groups");
        assert_eq!(groups[0].steps, vec![1, 2]);
        assert_eq!(groups[1].steps, vec![3, 4]);
    }

    #[test]
    fn test_find_fusion_groups_no_dispatches() {
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            CompiledStep::Passthrough {
                op_name: "reshape".into(),
                output_shape: vec![1, 64],
            },
        ]);
        let dag = partition_plan(&plan, None);
        let groups = find_fusion_groups(&dag);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_find_fusion_groups_already_fused_chains() {
        // fused_relu_exp_x2 -> sigmoid: the fused kernel is still elementwise.
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            make_dispatch_with_deps("fused_relu_exp_x2", &[1, 64], vec![0]),
            make_dispatch_with_deps("sigmoid", &[1, 64], vec![1]),
        ]);
        let dag = partition_plan(&plan, None);
        let groups = find_fusion_groups(&dag);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].steps, vec![1, 2]);
    }

    // -- Summary tests --------------------------------------------------------

    #[test]
    fn test_partition_summary() {
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            make_dispatch_with_deps("relu", &[1, 64], vec![0]),
            make_dispatch_with_deps("exp", &[1, 64], vec![1]),
            make_dispatch_with_deps("sigmoid", &[1, 64], vec![2]),
        ]);
        let dag = partition_plan(&plan, None);
        let groups = find_fusion_groups(&dag);
        let summary = partition_summary(&dag, &groups);

        assert_eq!(summary.total_dispatches, 3);
        assert_eq!(summary.elementwise_dispatches, 3);
        assert_eq!(summary.fusion_groups, 1);
        assert_eq!(summary.total_savings, 2);
        assert_eq!(summary.theoretical_minimum, 1);
    }

    #[test]
    fn test_partition_summary_display() {
        let summary = PartitionSummary {
            total_dispatches: 10,
            elementwise_dispatches: 5,
            fusion_groups: 2,
            total_savings: 3,
            theoretical_minimum: 7,
        };
        let text = format!("{summary}");
        assert!(text.contains("10 dispatches"));
        assert!(text.contains("5 elementwise"));
        assert!(text.contains("2 fusion groups"));
        assert!(text.contains("3 savings"));
        assert!(text.contains("7 theoretical minimum"));
    }

    #[test]
    fn test_fusion_group_display() {
        let group = FusionGroup {
            steps: vec![1, 2, 3],
            shape: vec![1, 64],
            dispatches_saved: 2,
        };
        let text = format!("{group}");
        assert!(text.contains("3 steps"));
        assert!(text.contains("saves 2 dispatches"));
    }

    #[test]
    fn test_partition_dag_with_native_op_barrier() {
        // NativeOp steps are fusion barriers.
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            make_dispatch_with_deps("relu", &[1, 64], vec![0]),
            CompiledStep::NativeOp {
                op: crate::trace_compile::NativeOpKind::Cumsum {
                    dim: 0,
                    input_shape: vec![1, 64],
                },
                weight_data: HashMap::new(),
            },
            make_dispatch_with_deps("exp", &[1, 64], vec![2]),
        ]);
        let dag = partition_plan(&plan, None);
        let groups = find_fusion_groups(&dag);

        // NativeOp breaks the elementwise chain.
        assert!(
            groups.is_empty(),
            "expected no fusion groups across NativeOp barrier"
        );
    }

    #[test]
    fn test_partition_mixed_shapes_partial_fusion() {
        // relu [1,64] -> exp [1,64] -> linear [1,32] -> sigmoid [1,32] -> tanh [1,32]
        // First two can fuse (same shape), last two can fuse (same shape).
        let edge_map = vec![
            vec![],  // step 0: input
            vec![0], // step 1: relu
            vec![1], // step 2: exp
            vec![2], // step 3: linear (opaque)
            vec![3], // step 4: sigmoid
            vec![4], // step 5: tanh
        ];
        let plan = make_plan(vec![
            CompiledStep::InputForward,
            make_dispatch("relu", &[1, 64]),
            make_dispatch("exp", &[1, 64]),
            make_dispatch("linear", &[1, 32]),
            make_dispatch("sigmoid", &[1, 32]),
            make_dispatch("tanh", &[1, 32]),
        ]);
        let dag = partition_plan(&plan, Some(&edge_map));
        let groups = find_fusion_groups(&dag);

        // relu+exp can fuse; linear is opaque barrier; sigmoid+tanh can fuse.
        assert_eq!(groups.len(), 2, "expected 2 fusion groups");
        assert_eq!(groups[0].steps, vec![1, 2]);
        assert_eq!(groups[0].shape, vec![1, 64]);
        assert_eq!(groups[1].steps, vec![4, 5]);
        assert_eq!(groups[1].shape, vec![1, 32]);
    }
}
