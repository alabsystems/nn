// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for bipartite matching (Hungarian algorithm).

use super::*;

#[test]
fn test_hungarian_empty_matrix() {
    let result = hungarian_matching(Vec::new());
    assert!(result.is_empty());
}

#[test]
fn test_hungarian_single_element() {
    let cost = vec![vec![5.0]];
    let result = hungarian_matching(cost);
    assert_eq!(result, vec![(0, 0)]);
}

#[test]
fn test_hungarian_identity_assignment() {
    // Diagonal has lowest cost -> expect identity matching
    let cost = vec![
        vec![1.0, 100.0, 100.0],
        vec![100.0, 1.0, 100.0],
        vec![100.0, 100.0, 1.0],
    ];
    let result = hungarian_matching(cost);
    assert_eq!(result, vec![(0, 0), (1, 1), (2, 2)]);
}

#[test]
fn test_hungarian_swap_assignment() {
    // Off-diagonal is cheaper
    let cost = vec![vec![100.0, 1.0], vec![1.0, 100.0]];
    let result = hungarian_matching(cost);
    assert_eq!(result, vec![(0, 1), (1, 0)]);
}

#[test]
fn test_hungarian_rectangular_more_rows() {
    // 3 rows x 2 cols: only 2 assignments
    let cost = vec![vec![10.0, 1.0], vec![1.0, 10.0], vec![5.0, 5.0]];
    let result = hungarian_matching(cost.clone());
    assert_eq!(result.len(), 2);
    let total: f32 = result.iter().map(|&(r, c)| cost[r][c]).sum();
    assert!(total <= 2.0, "total cost {total} should be <= 2.0");
}

#[test]
fn test_hungarian_rectangular_more_cols() {
    // 2 rows x 3 cols: only 2 assignments
    let cost = vec![vec![10.0, 1.0, 5.0], vec![1.0, 10.0, 5.0]];
    let result = hungarian_matching(cost.clone());
    assert_eq!(result.len(), 2);
    let total: f32 = result.iter().map(|&(r, c)| cost[r][c]).sum();
    assert!(total <= 2.0, "total cost {total} should be <= 2.0");
}

#[test]
fn test_generalized_iou_perfect_overlap() {
    let b = [0.5, 0.5, 0.2, 0.2]; // cx, cy, w, h
    let giou = generalized_iou(b, b);
    assert!(
        (giou - 1.0).abs() < 1e-5,
        "perfect overlap should give GIoU=1.0, got {giou}"
    );
}

#[test]
fn test_generalized_iou_no_overlap() {
    let a = [0.1, 0.1, 0.1, 0.1];
    let b = [0.9, 0.9, 0.1, 0.1];
    let giou = generalized_iou(a, b);
    assert!(
        giou < 0.0,
        "non-overlapping boxes should have negative GIoU, got {giou}"
    );
}

#[test]
fn test_generalized_iou_partial_overlap() {
    let a = [0.3, 0.3, 0.4, 0.4]; // [0.1, 0.1, 0.5, 0.5]
    let b = [0.4, 0.4, 0.4, 0.4]; // [0.2, 0.2, 0.6, 0.6]
    let giou = generalized_iou(a, b);
    assert!(
        giou > 0.0 && giou < 1.0,
        "partial overlap GIoU should be in (0, 1), got {giou}"
    );
}

#[test]
fn test_compute_assignment_cost_basic() {
    let cost = compute_assignment_cost(0.9, [0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]);
    // Near-perfect prediction: class cost ~0.1, L1 cost ~0, GIoU cost ~0
    assert!(
        cost < 5.0,
        "near-perfect prediction should have low cost, got {cost}"
    );
}
