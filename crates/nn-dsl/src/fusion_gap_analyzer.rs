// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fusion gap analyzer for compiled execution plans.
//!
//! Examines a [`CompiledPlan`] and reports WHY each pair of adjacent
//! dispatches was not fused. Produces a [`FusionGapAnalysis`] with
//! per-gap blocker classification and theoretical minimum dispatch count.
//!
//! Part of #3829 (Self-Optimizing ML Compiler, Phase 1).

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use nn_core::dyn_tensor::trace::{ComputationGraph, NodeId};

use super::fusion::is_fusible_elementwise;
use super::trace_compile_types::{CompiledPlan, CompiledStep};

/// Why a pair of adjacent dispatches was not fused.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum FusionBlocker {
    /// The first op's output feeds multiple consumers (fan-out > 1).
    FanOut,
    /// The two ops have incompatible output shapes.
    ShapeMismatch,
    /// At least one op is not a fusible elementwise operation.
    NonFusibleOp,
    /// One or both steps are not Dispatch steps (NativeOp, Passthrough, etc.).
    NotDispatch,
    /// The pair is already fused or part of a NativeOp.
    AlreadyOptimal,
    /// A peephole pass could match this pattern but did not fire.
    NoPeepholePattern,
    /// The two dispatch steps do not have a direct data dependency.
    NoDependency,
}

impl fmt::Display for FusionBlocker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FanOut => write!(f, "FanOut"),
            Self::ShapeMismatch => write!(f, "ShapeMismatch"),
            Self::NonFusibleOp => write!(f, "NonFusibleOp"),
            Self::NotDispatch => write!(f, "NotDispatch"),
            Self::AlreadyOptimal => write!(f, "AlreadyOptimal"),
            Self::NoPeepholePattern => write!(f, "NoPeepholePattern"),
            Self::NoDependency => write!(f, "NoDependency"),
        }
    }
}

/// A single fusion gap between two adjacent plan steps.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FusionGap {
    /// Index of the first step in the plan.
    pub step_a: usize,
    /// Index of the second step in the plan.
    pub step_b: usize,
    /// Kernel name of step A (if Dispatch).
    pub kernel_a: String,
    /// Kernel name of step B (if Dispatch).
    pub kernel_b: String,
    /// Why these two steps were not fused.
    pub reason: FusionBlocker,
    /// Estimated dispatch savings if this gap were closed (usually 1).
    pub savings: usize,
}

/// Result of analyzing all fusion gaps in a compiled plan.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FusionGapAnalysis {
    /// All identified fusion gaps.
    pub gaps: Vec<FusionGap>,
    /// Total dispatch steps in the plan (Dispatch + NativeOp).
    pub total_dispatches: usize,
    /// Theoretical minimum dispatches if all closable gaps were fused.
    pub theoretical_minimum: usize,
}

impl FusionGapAnalysis {
    /// Optimization opportunity as a percentage of total dispatches.
    ///
    /// Returns the fraction of dispatches that could theoretically be
    /// eliminated: `(total - theoretical_min) / total * 100`. Returns
    /// `0.0` when `total_dispatches` is zero to avoid division by zero.
    #[must_use]
    pub fn optimization_opportunity_pct(&self) -> f64 {
        if self.total_dispatches == 0 {
            return 0.0;
        }
        let reducible = self
            .total_dispatches
            .saturating_sub(self.theoretical_minimum);
        (reducible as f64 / self.total_dispatches as f64) * 100.0
    }

    /// Blocker distribution: how many gaps have each blocker type.
    ///
    /// Returns a `BTreeMap` for stable iteration / display ordering.
    #[must_use]
    pub fn blocker_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for gap in &self.gaps {
            *counts.entry(gap.reason.to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Human-readable summary string.
    ///
    /// Example output:
    /// ```text
    /// Fusion Gap Analysis: 201 dispatches (theoretical min: 78, 61.2% optimization opportunity)
    /// Top blockers: NonFusibleOp (45), FanOut (23), ShapeMismatch (12)
    /// ```
    #[must_use]
    pub fn summarize(&self) -> String {
        let pct = self.optimization_opportunity_pct();
        let mut summary = format!(
            "Fusion Gap Analysis: {} dispatches (theoretical min: {}, {:.1}% optimization opportunity)",
            self.total_dispatches, self.theoretical_minimum, pct,
        );

        let counts = self.blocker_counts();
        if !counts.is_empty() {
            // Sort descending by count for "top blockers" display.
            let mut sorted: Vec<_> = counts.into_iter().collect();
            sorted.sort_by_key(|x| std::cmp::Reverse(x.1));
            let blockers: Vec<String> = sorted
                .iter()
                .map(|(name, count)| format!("{name} ({count})"))
                .collect();
            summary.push_str("\nTop blockers: ");
            summary.push_str(&blockers.join(", "));
        }

        summary
    }
}

impl fmt::Display for FusionGapAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summarize())
    }
}

/// Build a use-count map: for each `NodeId`, how many downstream nodes
/// reference it as an input.
fn build_use_counts(graph: &ComputationGraph) -> HashMap<NodeId, usize> {
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    for node in graph.nodes() {
        for &input_id in node.inputs() {
            *counts.entry(input_id).or_insert(0) += 1;
        }
    }
    counts
}

/// Count dispatch steps (Dispatch + NativeOp) in a plan.
fn count_dispatches(plan: &CompiledPlan) -> usize {
    plan.steps
        .iter()
        .filter(|s| {
            matches!(
                s,
                CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
            )
        })
        .count()
}

/// Extract kernel name from a CompiledStep, if it is a Dispatch.
fn dispatch_kernel_name(step: &CompiledStep) -> Option<&str> {
    match step {
        CompiledStep::Dispatch { kernel, .. } => Some(kernel.name()),
        _ => None,
    }
}

/// Analyze fusion gaps in a compiled plan.
///
/// Walks adjacent pairs of [`CompiledStep`]s and classifies why each
/// pair of dispatch steps was not fused. Uses the computation graph to
/// check fan-out and shape compatibility.
///
/// # Arguments
///
/// * `plan` - The compiled execution plan to analyze.
/// * `graph` - The computation graph (for fan-out and shape lookups).
#[must_use]
pub fn analyze_fusion_gaps(plan: &CompiledPlan, graph: &ComputationGraph) -> FusionGapAnalysis {
    let use_counts = build_use_counts(graph);
    let nodes = graph.nodes();
    let total_dispatches = count_dispatches(plan);
    let mut gaps = Vec::new();

    for i in 0..plan.steps.len().saturating_sub(1) {
        let step_a = &plan.steps[i];
        let step_b = &plan.steps[i + 1];

        // Skip non-dispatch pairs -- only Dispatch-Dispatch gaps are fusible.
        let (name_a, name_b) = match (dispatch_kernel_name(step_a), dispatch_kernel_name(step_b)) {
            (Some(a), Some(b)) => (a.to_string(), b.to_string()),
            _ => {
                // If either is a NativeOp, it is already optimally fused.
                if matches!(step_a, CompiledStep::NativeOp { .. })
                    || matches!(step_b, CompiledStep::NativeOp { .. })
                {
                    continue;
                }
                // At least one side is not a dispatch -- record if mixed.
                if matches!(step_a, CompiledStep::Dispatch { .. })
                    || matches!(step_b, CompiledStep::Dispatch { .. })
                {
                    let ka = dispatch_kernel_name(step_a).unwrap_or("-").to_string();
                    let kb = dispatch_kernel_name(step_b).unwrap_or("-").to_string();
                    gaps.push(FusionGap {
                        step_a: i,
                        step_b: i + 1,
                        kernel_a: ka,
                        kernel_b: kb,
                        reason: FusionBlocker::NotDispatch,
                        savings: 0,
                    });
                }
                continue;
            }
        };

        // Already fused (kernel name starts with "fused_")?
        if name_a.starts_with("fused_") || name_b.starts_with("fused_") {
            gaps.push(FusionGap {
                step_a: i,
                step_b: i + 1,
                kernel_a: name_a,
                kernel_b: name_b,
                reason: FusionBlocker::AlreadyOptimal,
                savings: 0,
            });
            continue;
        }

        // Look up corresponding graph nodes (step index == node index).
        let (node_a, node_b) = match (nodes.get(i), nodes.get(i + 1)) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };

        // Check if both ops are fusible elementwise.
        let a_fusible = is_fusible_elementwise(node_a.op());
        let b_fusible = is_fusible_elementwise(node_b.op());
        if !a_fusible || !b_fusible {
            gaps.push(FusionGap {
                step_a: i,
                step_b: i + 1,
                kernel_a: name_a,
                kernel_b: name_b,
                reason: FusionBlocker::NonFusibleOp,
                savings: 0,
            });
            continue;
        }

        // Check data dependency: does step_b consume step_a's output?
        let a_id = node_a.id();
        if !node_b.inputs().contains(&a_id) {
            gaps.push(FusionGap {
                step_a: i,
                step_b: i + 1,
                kernel_a: name_a,
                kernel_b: name_b,
                reason: FusionBlocker::NoDependency,
                savings: 0,
            });
            continue;
        }

        // Check fan-out: step_a's output must have exactly 1 consumer.
        let use_count = use_counts.get(&a_id).copied().unwrap_or(0);
        if use_count > 1 {
            gaps.push(FusionGap {
                step_a: i,
                step_b: i + 1,
                kernel_a: name_a,
                kernel_b: name_b,
                reason: FusionBlocker::FanOut,
                savings: 1,
            });
            continue;
        }

        // Check shape compatibility.
        if node_a.output_shape() != node_b.output_shape() {
            gaps.push(FusionGap {
                step_a: i,
                step_b: i + 1,
                kernel_a: name_a,
                kernel_b: name_b,
                reason: FusionBlocker::ShapeMismatch,
                savings: 1,
            });
            continue;
        }

        // Both fusible, same shape, single consumer, data-dependent --
        // this should have been fused. Likely a peephole/chain detection gap.
        gaps.push(FusionGap {
            step_a: i,
            step_b: i + 1,
            kernel_a: name_a,
            kernel_b: name_b,
            reason: FusionBlocker::NoPeepholePattern,
            savings: 1,
        });
    }

    let closable_savings: usize = gaps.iter().map(|g| g.savings).sum();
    let theoretical_minimum = total_dispatches.saturating_sub(closable_savings);

    FusionGapAnalysis {
        gaps,
        total_dispatches,
        theoretical_minimum,
    }
}

/// Compute the theoretical minimum dispatch count from an analysis.
///
/// Convenience function -- equivalent to `analysis.theoretical_minimum`.
#[must_use]
pub fn theoretical_minimum_dispatches(analysis: &FusionGapAnalysis) -> usize {
    analysis.theoretical_minimum
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;

    use super::super::trace_compile_types::{CompiledKernel, CompiledPlan, CompiledStep};
    use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};

    /// Build a minimal CompiledStep::Dispatch with a given kernel name.
    fn make_dispatch(name: &str) -> CompiledStep {
        let node_id = TensorNodeId::new(0);
        let input_node = TensorNode::new(
            node_id,
            TensorOpKind::Input {
                name: "input_0".into(),
                shape: vec![1, 4],
            },
            vec![1, 4],
        );
        let def = TensorKernelDef::new(name, vec![input_node], node_id);
        CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: HashMap::new(),
            external_node_ids: None,
        }
    }

    #[test]
    fn test_empty_plan_no_gaps() {
        let plan = CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        };
        let graph = ComputationGraph::from_nodes(vec![]);
        let analysis = analyze_fusion_gaps(&plan, &graph);
        assert!(analysis.gaps.is_empty());
        assert_eq!(analysis.total_dispatches, 0);
        assert_eq!(analysis.theoretical_minimum, 0);
    }

    #[test]
    fn test_single_dispatch_no_gaps() {
        let node = TraceNode::new(
            0,
            "relu_0".into(),
            TraceOp::Relu,
            vec![],
            vec![1, 4],
            DType::F32,
        );
        let graph = ComputationGraph::from_nodes(vec![node]);

        let plan = CompiledPlan {
            steps: vec![make_dispatch("relu")],
            input_shapes: vec![vec![1, 4]],
            output_step: 0,
            weight_names: vec![],
        };
        let analysis = analyze_fusion_gaps(&plan, &graph);
        assert!(analysis.gaps.is_empty());
        assert_eq!(analysis.total_dispatches, 1);
    }

    #[test]
    fn test_non_fusible_op_detected() {
        // Two adjacent dispatch steps whose graph nodes are non-fusible ops.
        let n0 = TraceNode::new(
            0,
            "matmul".into(),
            TraceOp::MatMul,
            vec![],
            vec![1, 4],
            DType::F32,
        );
        let n1 = TraceNode::new(
            1,
            "softmax".into(),
            TraceOp::Softmax { dim: 1 },
            vec![0],
            vec![1, 4],
            DType::F32,
        );
        let graph = ComputationGraph::from_nodes(vec![n0, n1]);

        let plan = CompiledPlan {
            steps: vec![make_dispatch("matmul"), make_dispatch("softmax")],
            input_shapes: vec![],
            output_step: 1,
            weight_names: vec![],
        };
        let analysis = analyze_fusion_gaps(&plan, &graph);
        assert_eq!(analysis.gaps.len(), 1);
        assert_eq!(analysis.gaps[0].reason, FusionBlocker::NonFusibleOp);
        assert_eq!(analysis.gaps[0].savings, 0);
    }

    #[test]
    fn test_fan_out_detected() {
        // Node 0: Relu, Node 1: Exp consumes node 0.
        // Node 0 has 2 consumers (from a third node not in the plan but in the graph).
        let n0 = TraceNode::new(
            0,
            "relu".into(),
            TraceOp::Relu,
            vec![],
            vec![1, 4],
            DType::F32,
        );
        let n1 = TraceNode::new(
            1,
            "exp".into(),
            TraceOp::Exp,
            vec![0],
            vec![1, 4],
            DType::F32,
        );
        // Third node also consumes node 0, creating fan-out.
        let n2 = TraceNode::new(
            2,
            "neg".into(),
            TraceOp::Neg,
            vec![0],
            vec![1, 4],
            DType::F32,
        );
        let graph = ComputationGraph::from_nodes(vec![n0, n1, n2]);

        let plan = CompiledPlan {
            steps: vec![
                make_dispatch("relu"),
                make_dispatch("exp"),
                make_dispatch("neg"),
            ],
            input_shapes: vec![],
            output_step: 2,
            weight_names: vec![],
        };
        let analysis = analyze_fusion_gaps(&plan, &graph);
        // Gap between step 0 and step 1: fan-out (node 0 has 2 consumers).
        let fan_out_gaps: Vec<_> = analysis
            .gaps
            .iter()
            .filter(|g| g.reason == FusionBlocker::FanOut)
            .collect();
        assert_eq!(fan_out_gaps.len(), 1);
        assert_eq!(fan_out_gaps[0].savings, 1);
    }

    #[test]
    fn test_theoretical_minimum_convenience() {
        let analysis = FusionGapAnalysis {
            gaps: vec![FusionGap {
                step_a: 0,
                step_b: 1,
                kernel_a: "a".into(),
                kernel_b: "b".into(),
                reason: FusionBlocker::ShapeMismatch,
                savings: 1,
            }],
            total_dispatches: 5,
            theoretical_minimum: 4,
        };
        assert_eq!(theoretical_minimum_dispatches(&analysis), 4);
    }

    #[test]
    fn test_already_optimal_fused_kernel() {
        let n0 = TraceNode::new(
            0,
            "exp".into(),
            TraceOp::Exp,
            vec![],
            vec![1, 4],
            DType::F32,
        );
        let n1 = TraceNode::new(
            1,
            "relu".into(),
            TraceOp::Relu,
            vec![0],
            vec![1, 4],
            DType::F32,
        );
        let graph = ComputationGraph::from_nodes(vec![n0, n1]);

        let plan = CompiledPlan {
            steps: vec![make_dispatch("fused_exp_relu_x2"), make_dispatch("linear")],
            input_shapes: vec![],
            output_step: 1,
            weight_names: vec![],
        };
        let analysis = analyze_fusion_gaps(&plan, &graph);
        let optimal: Vec<_> = analysis
            .gaps
            .iter()
            .filter(|g| g.reason == FusionBlocker::AlreadyOptimal)
            .collect();
        assert_eq!(optimal.len(), 1);
        assert_eq!(optimal[0].savings, 0);
    }

    // -- Phase 2 method tests (#3830) ------------------------------------------

    #[test]
    fn test_optimization_opportunity_pct_zero_dispatches() {
        let analysis = FusionGapAnalysis {
            gaps: vec![],
            total_dispatches: 0,
            theoretical_minimum: 0,
        };
        assert!((analysis.optimization_opportunity_pct() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_optimization_opportunity_pct_no_savings() {
        let analysis = FusionGapAnalysis {
            gaps: vec![],
            total_dispatches: 10,
            theoretical_minimum: 10,
        };
        assert!((analysis.optimization_opportunity_pct() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_optimization_opportunity_pct_with_savings() {
        let analysis = FusionGapAnalysis {
            gaps: vec![FusionGap {
                step_a: 0,
                step_b: 1,
                kernel_a: "a".into(),
                kernel_b: "b".into(),
                reason: FusionBlocker::FanOut,
                savings: 1,
            }],
            total_dispatches: 200,
            theoretical_minimum: 80,
        };
        // (200 - 80) / 200 * 100 = 60.0
        assert!((analysis.optimization_opportunity_pct() - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_blocker_counts_empty() {
        let analysis = FusionGapAnalysis {
            gaps: vec![],
            total_dispatches: 5,
            theoretical_minimum: 5,
        };
        assert!(analysis.blocker_counts().is_empty());
    }

    #[test]
    fn test_blocker_counts_multiple_types() {
        let analysis = FusionGapAnalysis {
            gaps: vec![
                FusionGap {
                    step_a: 0,
                    step_b: 1,
                    kernel_a: "a".into(),
                    kernel_b: "b".into(),
                    reason: FusionBlocker::NonFusibleOp,
                    savings: 0,
                },
                FusionGap {
                    step_a: 1,
                    step_b: 2,
                    kernel_a: "b".into(),
                    kernel_b: "c".into(),
                    reason: FusionBlocker::FanOut,
                    savings: 1,
                },
                FusionGap {
                    step_a: 2,
                    step_b: 3,
                    kernel_a: "c".into(),
                    kernel_b: "d".into(),
                    reason: FusionBlocker::NonFusibleOp,
                    savings: 0,
                },
            ],
            total_dispatches: 10,
            theoretical_minimum: 9,
        };
        let counts = analysis.blocker_counts();
        assert_eq!(counts.get("NonFusibleOp"), Some(&2));
        assert_eq!(counts.get("FanOut"), Some(&1));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn test_summarize_format() {
        let analysis = FusionGapAnalysis {
            gaps: vec![
                FusionGap {
                    step_a: 0,
                    step_b: 1,
                    kernel_a: "a".into(),
                    kernel_b: "b".into(),
                    reason: FusionBlocker::NonFusibleOp,
                    savings: 0,
                },
                FusionGap {
                    step_a: 1,
                    step_b: 2,
                    kernel_a: "b".into(),
                    kernel_b: "c".into(),
                    reason: FusionBlocker::FanOut,
                    savings: 1,
                },
            ],
            total_dispatches: 10,
            theoretical_minimum: 9,
        };
        let summary = analysis.summarize();
        assert!(summary.contains("10 dispatches"));
        assert!(summary.contains("theoretical min: 9"));
        assert!(summary.contains("10.0% optimization opportunity"));
        assert!(summary.contains("Top blockers:"));
        assert!(summary.contains("NonFusibleOp (1)"));
        assert!(summary.contains("FanOut (1)"));
    }

    #[test]
    fn test_summarize_empty_gaps() {
        let analysis = FusionGapAnalysis {
            gaps: vec![],
            total_dispatches: 5,
            theoretical_minimum: 5,
        };
        let summary = analysis.summarize();
        assert!(summary.contains("5 dispatches"));
        assert!(summary.contains("0.0% optimization opportunity"));
        assert!(!summary.contains("Top blockers:"));
    }

    #[test]
    fn test_display_matches_summarize() {
        let analysis = FusionGapAnalysis {
            gaps: vec![FusionGap {
                step_a: 0,
                step_b: 1,
                kernel_a: "x".into(),
                kernel_b: "y".into(),
                reason: FusionBlocker::ShapeMismatch,
                savings: 1,
            }],
            total_dispatches: 4,
            theoretical_minimum: 3,
        };
        assert_eq!(format!("{analysis}"), analysis.summarize());
    }

    #[test]
    fn test_fusion_gap_analysis_serde_roundtrip() {
        let original = FusionGapAnalysis {
            gaps: vec![
                FusionGap {
                    step_a: 0,
                    step_b: 1,
                    kernel_a: "relu".into(),
                    kernel_b: "exp".into(),
                    reason: FusionBlocker::FanOut,
                    savings: 1,
                },
                FusionGap {
                    step_a: 3,
                    step_b: 4,
                    kernel_a: "matmul".into(),
                    kernel_b: "softmax".into(),
                    reason: FusionBlocker::NonFusibleOp,
                    savings: 0,
                },
                FusionGap {
                    step_a: 5,
                    step_b: 6,
                    kernel_a: "conv".into(),
                    kernel_b: "bn".into(),
                    reason: FusionBlocker::ShapeMismatch,
                    savings: 1,
                },
            ],
            total_dispatches: 20,
            theoretical_minimum: 18,
        };

        let json_str =
            serde_json::to_string_pretty(&original).expect("serialization should succeed");
        let restored: FusionGapAnalysis =
            serde_json::from_str(&json_str).expect("deserialization should succeed");

        assert_eq!(restored.total_dispatches, original.total_dispatches);
        assert_eq!(restored.theoretical_minimum, original.theoretical_minimum);
        assert_eq!(restored.gaps.len(), original.gaps.len());

        for (orig, rest) in original.gaps.iter().zip(restored.gaps.iter()) {
            assert_eq!(rest.step_a, orig.step_a);
            assert_eq!(rest.step_b, orig.step_b);
            assert_eq!(rest.kernel_a, orig.kernel_a);
            assert_eq!(rest.kernel_b, orig.kernel_b);
            assert_eq!(rest.reason, orig.reason);
            assert_eq!(rest.savings, orig.savings);
        }
    }

    #[test]
    fn test_fusion_blocker_display() {
        assert_eq!(format!("{}", FusionBlocker::FanOut), "FanOut");
        assert_eq!(format!("{}", FusionBlocker::ShapeMismatch), "ShapeMismatch");
        assert_eq!(format!("{}", FusionBlocker::NonFusibleOp), "NonFusibleOp");
        assert_eq!(format!("{}", FusionBlocker::NotDispatch), "NotDispatch");
        assert_eq!(
            format!("{}", FusionBlocker::AlreadyOptimal),
            "AlreadyOptimal"
        );
        assert_eq!(
            format!("{}", FusionBlocker::NoPeepholePattern),
            "NoPeepholePattern"
        );
        assert_eq!(format!("{}", FusionBlocker::NoDependency), "NoDependency");
    }
}
