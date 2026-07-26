// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bipartite matching helpers for DETR-style assignment.

const CLASS_COST_WEIGHT: f32 = 1.0;
const L1_COST_WEIGHT: f32 = 5.0;
const GIOU_COST_WEIGHT: f32 = 2.0;

/// Solve minimum-cost bipartite matching with the Hungarian algorithm.
///
/// Returns `(row_idx, col_idx)` assignments for `min(rows, cols)` pairs.
#[must_use]
pub fn hungarian_matching(cost: Vec<Vec<f32>>) -> Vec<(usize, usize)> {
    let rows = cost.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = cost[0].len();
    if cols == 0 || cost.iter().any(|row| row.len() != cols) {
        return Vec::new();
    }

    if rows <= cols {
        solve_hungarian(&cost, false)
    } else {
        let mut transposed = vec![vec![0.0; rows]; cols];
        for (row_idx, row) in cost.iter().enumerate() {
            for (col_idx, value) in row.iter().enumerate() {
                transposed[col_idx][row_idx] = *value;
            }
        }
        solve_hungarian(&transposed, true)
    }
}

/// Compute generalized IoU for boxes in normalized `(cx, cy, w, h)` format.
#[must_use]
pub fn generalized_iou(box_a: [f32; 4], box_b: [f32; 4]) -> f32 {
    let a = cxcywh_to_xyxy(box_a);
    let b = cxcywh_to_xyxy(box_b);

    let inter_left = a[0].max(b[0]);
    let inter_top = a[1].max(b[1]);
    let inter_right = a[2].min(b[2]);
    let inter_bottom = a[3].min(b[3]);
    let inter_w = (inter_right - inter_left).max(0.0);
    let inter_h = (inter_bottom - inter_top).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = ((a[2] - a[0]).max(0.0)) * ((a[3] - a[1]).max(0.0));
    let area_b = ((b[2] - b[0]).max(0.0)) * ((b[3] - b[1]).max(0.0));
    let union = area_a + area_b - inter_area;
    if union <= 0.0 {
        return 0.0;
    }

    let iou = inter_area / union;
    let cover_left = a[0].min(b[0]);
    let cover_top = a[1].min(b[1]);
    let cover_right = a[2].max(b[2]);
    let cover_bottom = a[3].max(b[3]);
    let cover_area = ((cover_right - cover_left).max(0.0)) * ((cover_bottom - cover_top).max(0.0));
    if cover_area <= 0.0 {
        return iou;
    }

    iou - (cover_area - union) / cover_area
}

/// Compute DETR assignment cost for one prediction/target pair.
///
/// `class_probability` is the predicted probability of the target class.
#[must_use]
pub fn compute_assignment_cost(
    class_probability: f32,
    pred_box: [f32; 4],
    target_box: [f32; 4],
) -> f32 {
    let class_cost = 1.0 - class_probability.clamp(0.0, 1.0);
    let l1_cost = pred_box
        .iter()
        .zip(target_box.iter())
        .map(|(pred, target)| (pred - target).abs())
        .sum::<f32>();
    let giou_cost = 1.0 - generalized_iou(pred_box, target_box);

    CLASS_COST_WEIGHT * class_cost + L1_COST_WEIGHT * l1_cost + GIOU_COST_WEIGHT * giou_cost
}

fn solve_hungarian(cost: &[Vec<f32>], transposed: bool) -> Vec<(usize, usize)> {
    let rows = cost.len();
    let cols = cost[0].len();
    let mut u = vec![0.0f32; rows + 1];
    let mut v = vec![0.0f32; cols + 1];
    let mut p = vec![0usize; cols + 1];
    let mut way = vec![0usize; cols + 1];

    for row in 1..=rows {
        p[0] = row;
        let mut col0 = 0usize;
        let mut minv = vec![f32::INFINITY; cols + 1];
        let mut used = vec![false; cols + 1];

        loop {
            used[col0] = true;
            let row0 = p[col0];
            let mut delta = f32::INFINITY;
            let mut col1 = 0usize;

            for col in 1..=cols {
                if used[col] {
                    continue;
                }
                let cur = cost[row0 - 1][col - 1] - u[row0] - v[col];
                if cur < minv[col] {
                    minv[col] = cur;
                    way[col] = col0;
                }
                if minv[col] < delta {
                    delta = minv[col];
                    col1 = col;
                }
            }

            for col in 0..=cols {
                if used[col] {
                    u[p[col]] += delta;
                    v[col] -= delta;
                } else {
                    minv[col] -= delta;
                }
            }
            col0 = col1;

            if p[col0] == 0 {
                break;
            }
        }

        loop {
            let prev_col = way[col0];
            p[col0] = p[prev_col];
            col0 = prev_col;
            if col0 == 0 {
                break;
            }
        }
    }

    let mut assignments = Vec::with_capacity(rows.min(cols));
    for col in 1..=cols {
        if p[col] == 0 {
            continue;
        }
        let pair = if transposed {
            (col - 1, p[col] - 1)
        } else {
            (p[col] - 1, col - 1)
        };
        assignments.push(pair);
    }
    assignments.sort_unstable();
    assignments
}

fn cxcywh_to_xyxy(box_xywh: [f32; 4]) -> [f32; 4] {
    let half_w = box_xywh[2].max(0.0) * 0.5;
    let half_h = box_xywh[3].max(0.0) * 0.5;
    [
        box_xywh[0] - half_w,
        box_xywh[1] - half_h,
        box_xywh[0] + half_w,
        box_xywh[1] + half_h,
    ]
}

#[cfg(test)]
#[path = "bipartite_matching_tests.rs"]
mod tests;
