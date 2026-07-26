// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the dispatch plan optimizer.
//!
//! Tests use synthetic dependency graphs (via `DepGraph::from_edge_map`)
//! and minimal `CompiledPlan` instances to verify reordering correctness.

use super::*;

// ---------------------------------------------------------------------------
// DepGraph construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_dep_graph_empty() {
    let edge_map: Vec<Vec<usize>> = Vec::new();
    let segments: Vec<Option<u32>> = Vec::new();
    let graph = DepGraph::from_edge_map(&edge_map, &segments);
    assert_eq!(graph.num_steps, 0);
}

#[test]
fn test_dep_graph_linear_chain() {
    // A -> B -> C -> D (linear chain, no reordering possible)
    // edge_map: 0=[], 1=[0], 2=[1], 3=[2]
    let edge_map = vec![vec![], vec![0], vec![1], vec![2]];
    let segments = vec![None; 4];
    let graph = DepGraph::from_edge_map(&edge_map, &segments);

    assert_eq!(graph.num_steps, 4);
    assert_eq!(graph.in_degree, vec![0, 1, 1, 1]);
    assert!(graph.successors[0].contains(&1));
    assert!(graph.successors[1].contains(&2));
    assert!(graph.successors[2].contains(&3));
}

#[test]
fn test_dep_graph_diamond() {
    // A -> B, A -> C, B+C -> D
    // edge_map: 0=[], 1=[0], 2=[0], 3=[1,2]
    let edge_map = vec![vec![], vec![0], vec![0], vec![1, 2]];
    let segments = vec![None; 4];
    let graph = DepGraph::from_edge_map(&edge_map, &segments);

    assert_eq!(graph.in_degree, vec![0, 1, 1, 2]);
    // A has two successors: B and C
    assert!(graph.successors[0].contains(&1));
    assert!(graph.successors[0].contains(&2));
    // D depends on B and C
    assert!(graph.predecessors[3].contains(&1));
    assert!(graph.predecessors[3].contains(&2));
}

#[test]
fn test_dep_graph_segment_ordering() {
    // Steps 0, 1, 2 are independent but 1 and 2 share segment 0.
    // Segment ordering adds edge: 1 -> 2.
    let edge_map = vec![vec![], vec![], vec![]];
    let segments = vec![None, Some(0), Some(0)];
    let graph = DepGraph::from_edge_map(&edge_map, &segments);

    assert_eq!(graph.in_degree[0], 0);
    assert_eq!(graph.in_degree[1], 0); // first in segment
    assert_eq!(graph.in_degree[2], 1); // depends on step 1 (segment order)
    assert!(graph.predecessors[2].contains(&1));
}

#[test]
fn test_dep_graph_multiple_segments() {
    // 6 steps: segment A = [0,2,4], segment B = [1,3,5]
    let edge_map = vec![vec![]; 6];
    let segments = vec![Some(0), Some(1), Some(0), Some(1), Some(0), Some(1)];
    let graph = DepGraph::from_edge_map(&edge_map, &segments);

    // Segment 0: 0->2->4
    assert!(graph.predecessors[2].contains(&0));
    assert!(graph.predecessors[4].contains(&2));
    // Segment 1: 1->3->5
    assert!(graph.predecessors[3].contains(&1));
    assert!(graph.predecessors[5].contains(&3));
    // No cross-segment deps
    assert!(!graph.predecessors[1].contains(&0));
    assert!(!graph.predecessors[2].contains(&1));
}

// ---------------------------------------------------------------------------
// Topological sort correctness tests
// ---------------------------------------------------------------------------

/// Helper: check that reordered indices respect all dependencies.
fn assert_valid_topological_order(order: &[StepId], edge_map: &[Vec<usize>]) {
    let n = edge_map.len();
    let mut position = vec![usize::MAX; n];
    for (pos, &step) in order.iter().enumerate() {
        assert!(step < n, "step index out of range: {step}");
        position[step] = pos;
    }

    for (step, deps) in edge_map.iter().enumerate() {
        for &dep in deps {
            if dep < n {
                assert!(
                    position[dep] < position[step],
                    "dependency violated: step {dep} (pos {}) must come before step {step} (pos {})",
                    position[dep],
                    position[step]
                );
            }
        }
    }
}

#[test]
fn test_linear_chain_no_reorder() {
    // Linear chain: 0 -> 1 -> 2 -> 3
    // No reordering possible — only one valid topological order.
    let edge_map = vec![vec![], vec![0], vec![1], vec![2]];
    let segments = vec![None; 4];
    let graph = DepGraph::from_edge_map(&edge_map, &segments);

    let step_bytes = vec![100, 200, 300, 400];
    let last_consumer = vec![1, 2, 3, 3];

    let mut in_degree = graph.in_degree.clone();
    let mut heap = BinaryHeap::new();
    let mut result = Vec::new();

    for step in 0..4 {
        if in_degree[step] == 0 {
            let freed = compute_freed_bytes(step, &edge_map, &step_bytes, &last_consumer);
            heap.push(PriorityEntry {
                step,
                freed_bytes: freed,
                original_index: step,
            });
        }
    }

    while let Some(entry) = heap.pop() {
        result.push(entry.step);
        for &succ in &graph.successors[entry.step] {
            in_degree[succ] = in_degree[succ].saturating_sub(1);
            if in_degree[succ] == 0 {
                let freed = compute_freed_bytes(succ, &edge_map, &step_bytes, &last_consumer);
                heap.push(PriorityEntry {
                    step: succ,
                    freed_bytes: freed,
                    original_index: succ,
                });
            }
        }
    }

    assert_eq!(result, vec![0, 1, 2, 3]);
    assert_valid_topological_order(&result, &edge_map);
}

#[test]
fn test_diamond_reorder_possible() {
    // Diamond: A(0) -> B(1), A(0) -> C(2), B+C -> D(3)
    // B and C are independent and can be reordered.
    let edge_map = vec![vec![], vec![0], vec![0], vec![1, 2]];
    let segments = vec![None; 4];
    let graph = DepGraph::from_edge_map(&edge_map, &segments);

    // Both B(1) and C(2) should be schedulable after A(0).
    // Verify the result is a valid topological order.
    let step_bytes = vec![1000, 500, 2000, 100];
    let last_consumer = vec![2, 3, 3, 3]; // A last used at C(2), B,C at D(3)

    let mut in_degree = graph.in_degree.clone();
    let mut heap = BinaryHeap::new();
    let mut result = Vec::new();

    for step in 0..4 {
        if in_degree[step] == 0 {
            let freed = compute_freed_bytes(step, &edge_map, &step_bytes, &last_consumer);
            heap.push(PriorityEntry {
                step,
                freed_bytes: freed,
                original_index: step,
            });
        }
    }

    while let Some(entry) = heap.pop() {
        result.push(entry.step);
        for &succ in &graph.successors[entry.step] {
            in_degree[succ] = in_degree[succ].saturating_sub(1);
            if in_degree[succ] == 0 {
                let freed = compute_freed_bytes(succ, &edge_map, &step_bytes, &last_consumer);
                heap.push(PriorityEntry {
                    step: succ,
                    freed_bytes: freed,
                    original_index: succ,
                });
            }
        }
    }

    assert_valid_topological_order(&result, &edge_map);
    // A must be first, D must be last.
    assert_eq!(result[0], 0);
    assert_eq!(result[3], 3);
    // B and C in some order (both valid).
    assert!(result[1] == 1 || result[1] == 2);
    assert!(result[2] == 1 || result[2] == 2);
    assert_ne!(result[1], result[2]);
}

#[test]
fn test_independent_branches_fully_reorderable() {
    // 4 independent steps: 0, 1, 2, 3 (no dependencies).
    // All orderings are valid.
    let edge_map = vec![vec![]; 4];
    let segments = vec![None; 4];
    let graph = DepGraph::from_edge_map(&edge_map, &segments);

    assert_eq!(graph.in_degree, vec![0, 0, 0, 0]);

    // All steps should be immediately schedulable.
    let step_bytes = vec![100, 200, 300, 400];
    let last_consumer = vec![0, 1, 2, 3]; // no consumers

    let mut in_degree = graph.in_degree.clone();
    let mut heap = BinaryHeap::new();
    let mut result = Vec::new();

    for step in 0..4 {
        if in_degree[step] == 0 {
            let freed = compute_freed_bytes(step, &edge_map, &step_bytes, &last_consumer);
            heap.push(PriorityEntry {
                step,
                freed_bytes: freed,
                original_index: step,
            });
        }
    }

    while let Some(entry) = heap.pop() {
        result.push(entry.step);
        for &succ in &graph.successors[entry.step] {
            in_degree[succ] = in_degree[succ].saturating_sub(1);
            if in_degree[succ] == 0 {
                let freed = compute_freed_bytes(succ, &edge_map, &step_bytes, &last_consumer);
                heap.push(PriorityEntry {
                    step: succ,
                    freed_bytes: freed,
                    original_index: succ,
                });
            }
        }
    }

    assert_eq!(result.len(), 4);
    assert_valid_topological_order(&result, &edge_map);
}

#[test]
fn test_segment_ordering_constraint() {
    // Steps 0 and 1 are independent, but both in segment 0.
    // Segment ordering forces 0 before 1.
    let edge_map = vec![vec![], vec![]];
    let segments = vec![Some(0), Some(0)];
    let graph = DepGraph::from_edge_map(&edge_map, &segments);

    assert_eq!(graph.in_degree[0], 0);
    assert_eq!(graph.in_degree[1], 1); // synthetic segment edge

    let step_bytes = vec![100, 100];
    let last_consumer = vec![1, 1];

    let mut in_degree = graph.in_degree.clone();
    let mut heap = BinaryHeap::new();
    let mut result = Vec::new();

    for step in 0..2 {
        if in_degree[step] == 0 {
            let freed = compute_freed_bytes(step, &edge_map, &step_bytes, &last_consumer);
            heap.push(PriorityEntry {
                step,
                freed_bytes: freed,
                original_index: step,
            });
        }
    }

    while let Some(entry) = heap.pop() {
        result.push(entry.step);
        for &succ in &graph.successors[entry.step] {
            in_degree[succ] = in_degree[succ].saturating_sub(1);
            if in_degree[succ] == 0 {
                let freed = compute_freed_bytes(succ, &edge_map, &step_bytes, &last_consumer);
                heap.push(PriorityEntry {
                    step: succ,
                    freed_bytes: freed,
                    original_index: succ,
                });
            }
        }
    }

    assert_eq!(result, vec![0, 1]);
}

#[test]
fn test_cross_segment_reordering() {
    // Steps: 0 (seg A), 1 (seg B), 2 (seg A), 3 (seg B)
    // Segment A chain: 0 -> 2. Segment B chain: 1 -> 3.
    // No data dependencies. Valid orders include interleaving.
    let edge_map = vec![vec![]; 4];
    let segments = vec![Some(0), Some(1), Some(0), Some(1)];
    let graph = DepGraph::from_edge_map(&edge_map, &segments);

    // Seg A: 0->2, Seg B: 1->3
    assert_eq!(graph.in_degree[0], 0);
    assert_eq!(graph.in_degree[1], 0);
    assert_eq!(graph.in_degree[2], 1); // depends on 0
    assert_eq!(graph.in_degree[3], 1); // depends on 1

    let step_bytes = vec![100, 200, 100, 200];
    let last_consumer = vec![2, 3, 2, 3];

    let mut in_degree = graph.in_degree.clone();
    let mut heap = BinaryHeap::new();
    let mut result = Vec::new();

    for step in 0..4 {
        if in_degree[step] == 0 {
            let freed = compute_freed_bytes(step, &edge_map, &step_bytes, &last_consumer);
            heap.push(PriorityEntry {
                step,
                freed_bytes: freed,
                original_index: step,
            });
        }
    }

    while let Some(entry) = heap.pop() {
        result.push(entry.step);
        for &succ in &graph.successors[entry.step] {
            in_degree[succ] = in_degree[succ].saturating_sub(1);
            if in_degree[succ] == 0 {
                let freed = compute_freed_bytes(succ, &edge_map, &step_bytes, &last_consumer);
                heap.push(PriorityEntry {
                    step: succ,
                    freed_bytes: freed,
                    original_index: succ,
                });
            }
        }
    }

    assert_eq!(result.len(), 4);

    // Verify segment ordering is preserved.
    let pos_0 = result.iter().position(|&s| s == 0).unwrap();
    let pos_2 = result.iter().position(|&s| s == 2).unwrap();
    let pos_1 = result.iter().position(|&s| s == 1).unwrap();
    let pos_3 = result.iter().position(|&s| s == 3).unwrap();
    assert!(pos_0 < pos_2, "segment A ordering: 0 before 2");
    assert!(pos_1 < pos_3, "segment B ordering: 1 before 3");
}

// ---------------------------------------------------------------------------
// OptimizedPlan statistics tests
// ---------------------------------------------------------------------------

#[test]
fn test_optimized_plan_display() {
    let plan = OptimizedPlan {
        reordered_indices: vec![0, 1, 2],
        reorder_count: 1,
        peak_memory_before: 1024 * 1024, // 1 MB
        peak_memory_after: 512 * 1024,   // 0.5 MB
        total_steps: 3,
    };
    let display = format!("{plan}");
    assert!(display.contains("steps: 3"));
    assert!(display.contains("reordered: 1"));
    assert!(display.contains("savings:"));
}

#[test]
fn test_optimized_plan_savings() {
    let plan = OptimizedPlan {
        reordered_indices: vec![],
        reorder_count: 0,
        peak_memory_before: 1000,
        peak_memory_after: 750,
        total_steps: 0,
    };
    assert_eq!(plan.memory_saved(), 250);
    assert!((plan.savings_pct() - 25.0).abs() < 0.01);
}

#[test]
fn test_optimized_plan_zero_memory() {
    let plan = OptimizedPlan {
        reordered_indices: vec![],
        reorder_count: 0,
        peak_memory_before: 0,
        peak_memory_after: 0,
        total_steps: 0,
    };
    assert_eq!(plan.memory_saved(), 0);
    assert_eq!(plan.savings_pct(), 0.0);
}

// ---------------------------------------------------------------------------
// estimate_step_bytes tests
// ---------------------------------------------------------------------------

#[test]
fn test_estimate_step_bytes_constant() {
    let step = CompiledStep::ConstantValue {
        value: 1.0,
        shape: vec![2, 3, 4],
    };
    // 2*3*4 = 24 elements * 4 bytes = 96
    assert_eq!(estimate_step_bytes(&step), 96);
}

#[test]
fn test_estimate_step_bytes_passthrough() {
    let step = CompiledStep::Passthrough {
        op_name: "reshape".to_string(),
        output_shape: vec![1, 2, 3],
    };
    assert_eq!(estimate_step_bytes(&step), 0);
}

#[test]
fn test_estimate_step_bytes_input_forward() {
    assert_eq!(estimate_step_bytes(&CompiledStep::InputForward), 0);
}

#[test]
fn test_estimate_step_bytes_identity() {
    assert_eq!(estimate_step_bytes(&CompiledStep::IdentityPassthrough), 0);
}

// ---------------------------------------------------------------------------
// Priority ordering tests
// ---------------------------------------------------------------------------

#[test]
fn test_priority_entry_ordering() {
    let a = PriorityEntry {
        step: 0,
        freed_bytes: 100,
        original_index: 5,
    };
    let b = PriorityEntry {
        step: 1,
        freed_bytes: 200,
        original_index: 3,
    };
    // b frees more bytes, so b > a.
    assert!(b > a);

    let c = PriorityEntry {
        step: 2,
        freed_bytes: 200,
        original_index: 1,
    };
    // Same freed_bytes. c has lower original_index, so c > b (stability).
    assert!(c > b);
}

// ---------------------------------------------------------------------------
// Peak memory computation tests
// ---------------------------------------------------------------------------

#[test]
fn test_compute_peak_memory_empty() {
    let steps: Vec<CompiledStep> = Vec::new();
    let edge_map: Vec<Vec<usize>> = Vec::new();
    let order: Vec<StepId> = Vec::new();
    assert_eq!(compute_peak_memory(&steps, &edge_map, &order), 0);
}

#[test]
fn test_compute_peak_memory_single() {
    let steps = vec![CompiledStep::ConstantValue {
        value: 0.0,
        shape: vec![10],
    }];
    let edge_map = vec![vec![]];
    let order = vec![0];
    // 10 elements * 4 bytes = 40
    assert_eq!(compute_peak_memory(&steps, &edge_map, &order), 40);
}

#[test]
fn test_compute_peak_memory_chain_with_reuse() {
    // Chain: 0 -> 1 -> 2
    // Step 0 produces 100 bytes, consumed only by step 1.
    // Step 1 produces 200 bytes, consumed only by step 2.
    // Step 2 produces 50 bytes.
    //
    // After step 0 executes: live = {0: 100}
    // After step 1 executes: live = {0: 100, 1: 200}; then free 0 -> {1: 200}
    // After step 2 executes: live = {1: 200, 2: 50}; then free 1 -> {2: 50}
    // Peak = 300 (at step 1, before freeing step 0)
    let steps = vec![
        CompiledStep::ConstantValue {
            value: 0.0,
            shape: vec![25],
        }, // 100 bytes
        CompiledStep::ConstantValue {
            value: 0.0,
            shape: vec![50],
        }, // 200 bytes
        CompiledStep::ConstantValue {
            value: 0.0,
            shape: vec![12],
        }, // 48 bytes (close to 50)
    ];
    let edge_map = vec![vec![], vec![0], vec![1]];
    let order = vec![0, 1, 2];
    let peak = compute_peak_memory(&steps, &edge_map, &order);
    // Step 0: alloc 100, peak=100; then no frees (last use of 0 is step 1)
    // Step 1: alloc 200, live=300, peak=300; free 0 (last use=1), live=200
    // Step 2: alloc 48, live=248, peak stays 300; free 1, live=48
    assert_eq!(peak, 300);
}
