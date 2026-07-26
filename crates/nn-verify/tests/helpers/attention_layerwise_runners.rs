// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, unreachable_pub, clippy::duplicated_attributes)]

//! Pipeline runner functions for layerwise attention verification tests
//! (Phases 13–14).
//!
//! Extracted from the merged Phase 13–14 test file per #1978.

use super::common;
use super::helpers;
use super::lw_builders;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Run 3-layer attention pipeline and return output bounds + per-layer widths.
///
/// Returns `(output_bounds, score_width, softmax_width, output_width)`.
pub fn run_layerwise_pipeline_measured(
    prefix: &str,
    seq_len: usize,
    d: usize,
    input_bound: f32,
    k_scale: f32,
    v_scale: f32,
) -> (BoundedTensor, f32, f32, f32) {
    // Layer 1: Score computation
    let score_def = lw_builders::build_score_layer(&format!("{prefix}_scores_{d}"), seq_len, d);
    let k_tensor = lw_builders::build_k_identity(seq_len, d, k_scale);
    let score_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(k_tensor),
    ];
    let score_graph = tensor_kernel_to_graph(&score_def, &score_bindings).expect("score graph");
    let q_bounds = common::uniform_bounds(&[seq_len, d], input_bound);
    let (_, score_out, _) = nn_verify::propagate_with_crown_fallback(&score_graph, &q_bounds)
        .expect("score propagation");
    let score_width = lw_builders::measure_total_width(&score_out);

    // Layer 2: Softmax
    let sm_def = lw_builders::build_softmax_layer(&format!("{prefix}_softmax_{d}"), seq_len);
    let sm_bindings = vec![TensorParamBinding::Variable];
    let sm_graph = tensor_kernel_to_graph(&sm_def, &sm_bindings).expect("softmax graph");
    let (_, sm_out, _) = nn_verify::propagate_with_crown_fallback(&sm_graph, &score_out)
        .expect("softmax propagation");
    let sm_width = lw_builders::measure_total_width(&sm_out);

    // Layer 3: Output projection
    let out_def = lw_builders::build_output_layer(&format!("{prefix}_output_{d}"), seq_len, d);
    let v_data = vec![v_scale; seq_len * d];
    let v_tensor = ArrayD::from_shape_vec(IxDyn(&[seq_len, d]), v_data).expect("valid V");
    let out_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(v_tensor),
    ];
    let out_graph = tensor_kernel_to_graph(&out_def, &out_bindings).expect("output graph");
    let (_, out_out, _) =
        nn_verify::propagate_with_crown_fallback(&out_graph, &sm_out).expect("output propagation");
    let out_width = lw_builders::measure_total_width(&out_out);

    (out_out, score_width, sm_width, out_width)
}

/// Run layerwise pipeline with empirical (PE-centered) bounds.
///
/// Returns `(output_bounds, diagonal_dominant_count)`.
pub fn run_layerwise_empirical(
    seq_len: usize,
    d: usize,
    perturbation: f32,
    pe_scale: f32,
) -> (BoundedTensor, usize) {
    let score_def = lw_builders::build_score_layer(&format!("lw_emp_scores_{d}"), seq_len, d);
    let pe = helpers::build_sinusoidal_pe(seq_len, d);
    let mut pe_scaled = pe;
    pe_scaled.mapv_inplace(|v| v * pe_scale);

    let score_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_scaled.clone()),
    ];
    let score_graph = tensor_kernel_to_graph(&score_def, &score_bindings).expect("score graph");

    let input = lw_builders::build_pe_centered_bounds(&pe_scaled, perturbation);
    let (_, score_out, _) =
        nn_verify::propagate_with_crown_fallback(&score_graph, &input).expect("score propagation");
    let dominant = lw_builders::count_diagonal_dominant(&score_out, seq_len);

    // Continue through softmax + output
    let sm_def = lw_builders::build_softmax_layer(&format!("lw_emp_softmax_{d}"), seq_len);
    let sm_bindings = vec![TensorParamBinding::Variable];
    let sm_graph = tensor_kernel_to_graph(&sm_def, &sm_bindings).expect("softmax graph");
    let (_, sm_out, _) = nn_verify::propagate_with_crown_fallback(&sm_graph, &score_out)
        .expect("softmax propagation");

    let out_def = lw_builders::build_output_layer(&format!("lw_emp_output_{d}"), seq_len, d);
    let v_data: Vec<f32> = (0..seq_len * d)
        .map(|i| 0.1 * ((i % 7) as f32 - 3.0))
        .collect();
    let v_tensor = ArrayD::from_shape_vec(IxDyn(&[seq_len, d]), v_data).expect("valid V");
    let out_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(v_tensor),
    ];
    let out_graph = tensor_kernel_to_graph(&out_def, &out_bindings).expect("output graph");
    let (_, out_out, _) =
        nn_verify::propagate_with_crown_fallback(&out_graph, &sm_out).expect("output propagation");

    (out_out, dominant)
}

/// Run per-head layerwise verification for multi-head attention.
///
/// Returns `(per_head_dominant_counts, per_head_avg_widths)`.
pub fn run_multihead_layerwise(
    seq_len: usize,
    d_model: usize,
    num_heads: usize,
    perturbation: f32,
    pe_scale: f32,
) -> (Vec<usize>, Vec<f32>) {
    assert_eq!(
        d_model % num_heads,
        0,
        "d_model must be divisible by num_heads"
    );
    let head_dim = d_model / num_heads;

    let pe_full = helpers::build_sinusoidal_pe(seq_len, d_model);
    let mut pe_scaled = pe_full;
    pe_scaled.mapv_inplace(|v| v * pe_scale);

    let mut dominant_counts = Vec::new();
    let mut avg_widths = Vec::new();

    for h in 0..num_heads {
        let start_col = h * head_dim;
        let mut pe_head = vec![0.0f32; seq_len * head_dim];
        for t in 0..seq_len {
            for c in 0..head_dim {
                pe_head[t * head_dim + c] = pe_scaled[[t, start_col + c]];
            }
        }
        let pe_head_arr =
            ArrayD::from_shape_vec(IxDyn(&[seq_len, head_dim]), pe_head).expect("head PE");

        let score_def =
            lw_builders::build_score_layer(&format!("mh_h{h}_scores"), seq_len, head_dim);
        let score_bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(pe_head_arr.clone()),
        ];
        let score_graph = tensor_kernel_to_graph(&score_def, &score_bindings).expect("score graph");

        let input = lw_builders::build_pe_centered_bounds(&pe_head_arr, perturbation);
        let (_, score_out, _) = nn_verify::propagate_with_crown_fallback(&score_graph, &input)
            .expect("head score propagation");

        dominant_counts.push(lw_builders::count_diagonal_dominant(&score_out, seq_len));

        let sm_def = lw_builders::build_softmax_layer(&format!("mh_h{h}_softmax"), seq_len);
        let sm_bindings = vec![TensorParamBinding::Variable];
        let sm_graph = tensor_kernel_to_graph(&sm_def, &sm_bindings).expect("softmax graph");
        let (_, sm_out, _) = nn_verify::propagate_with_crown_fallback(&sm_graph, &score_out)
            .expect("softmax propagation");

        let out_def =
            lw_builders::build_output_layer(&format!("mh_h{h}_output"), seq_len, head_dim);
        let v_data: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| 0.1 * ((i % 5) as f32 - 2.0))
            .collect();
        let v_tensor =
            ArrayD::from_shape_vec(IxDyn(&[seq_len, head_dim]), v_data).expect("valid V");
        let out_bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(v_tensor),
        ];
        let out_graph = tensor_kernel_to_graph(&out_def, &out_bindings).expect("output graph");
        let (_, out_out, _) = nn_verify::propagate_with_crown_fallback(&out_graph, &sm_out)
            .expect("output propagation");

        avg_widths.push(lw_builders::measure_avg_width(&out_out));
    }

    (dominant_counts, avg_widths)
}

/// Run 4-layer projected attention pipeline.
///
/// Returns `(output_bounds, proj_width, score_width, sm_width, out_width)`.
pub fn run_projected_pipeline(
    seq_len: usize,
    d_model: usize,
    d_k: usize,
    input_bound: f32,
    w_diag: f32,
    w_offdiag: f32,
    k_scale: f32,
    v_scale: f32,
) -> (BoundedTensor, f32, f32, f32, f32) {
    // Layer 1: Linear projection Q → Q_proj
    let proj_def = lw_builders::build_projection_layer(
        &format!("lw_proj_{d_model}_{d_k}"),
        seq_len,
        d_model,
        d_k,
    );
    let w_q = lw_builders::build_near_identity_weights(d_model, d_k, w_diag, w_offdiag);
    let proj_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w_q),
    ];
    let proj_graph = tensor_kernel_to_graph(&proj_def, &proj_bindings).expect("projection graph");
    let q_bounds = common::uniform_bounds(&[seq_len, d_model], input_bound);
    let (_, proj_out, _) = nn_verify::propagate_with_crown_fallback(&proj_graph, &q_bounds)
        .expect("projection propagation");
    let proj_width = lw_builders::measure_total_width(&proj_out);

    // Layer 2: Score computation on projected Q
    let score_def =
        lw_builders::build_score_layer(&format!("lw_pscore_{d_model}_{d_k}"), seq_len, d_k);
    let k_tensor = lw_builders::build_k_identity(seq_len, d_k, k_scale);
    let score_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(k_tensor),
    ];
    let score_graph = tensor_kernel_to_graph(&score_def, &score_bindings).expect("score graph");
    let (_, score_out, _) = nn_verify::propagate_with_crown_fallback(&score_graph, &proj_out)
        .expect("score propagation");
    let score_width = lw_builders::measure_total_width(&score_out);

    // Layer 3: Softmax
    let sm_def = lw_builders::build_softmax_layer(&format!("lw_psm_{d_model}_{d_k}"), seq_len);
    let sm_bindings = vec![TensorParamBinding::Variable];
    let sm_graph = tensor_kernel_to_graph(&sm_def, &sm_bindings).expect("softmax graph");
    let (_, sm_out, _) = nn_verify::propagate_with_crown_fallback(&sm_graph, &score_out)
        .expect("softmax propagation");
    let sm_width = lw_builders::measure_total_width(&sm_out);

    // Layer 4: Output
    let out_def =
        lw_builders::build_output_layer(&format!("lw_pout_{d_model}_{d_k}"), seq_len, d_k);
    let v_data = vec![v_scale; seq_len * d_k];
    let v_tensor = ArrayD::from_shape_vec(IxDyn(&[seq_len, d_k]), v_data).expect("valid V");
    let out_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(v_tensor),
    ];
    let out_graph = tensor_kernel_to_graph(&out_def, &out_bindings).expect("output graph");
    let (_, out_out, _) =
        nn_verify::propagate_with_crown_fallback(&out_graph, &sm_out).expect("output propagation");
    let out_width = lw_builders::measure_total_width(&out_out);

    (out_out, proj_width, score_width, sm_width, out_width)
}

/// Run projected multi-head attention with per-head verification.
///
/// Returns `(per_head_avg_widths, per_head_dominant_counts)`.
pub fn run_projected_multihead(
    seq_len: usize,
    d_model: usize,
    num_heads: usize,
    _input_bound: f32,
    w_diag: f32,
    w_offdiag: f32,
    pe_scale: f32,
    perturbation: f32,
) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(d_model % num_heads, 0);
    let d_k = d_model / num_heads;

    let pe_full = helpers::build_sinusoidal_pe(seq_len, d_model);
    let mut pe_scaled = pe_full;
    pe_scaled.mapv_inplace(|v| v * pe_scale);

    let mut avg_widths = Vec::new();
    let mut dominant_counts = Vec::new();

    for h in 0..num_heads {
        let w_q_h = lw_builders::build_near_identity_weights(d_model, d_k, w_diag, w_offdiag);

        let proj_def =
            lw_builders::build_projection_layer(&format!("pmh_proj_h{h}"), seq_len, d_model, d_k);
        let proj_bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(w_q_h),
        ];
        let proj_graph = tensor_kernel_to_graph(&proj_def, &proj_bindings).expect("proj graph");

        let input = lw_builders::build_pe_centered_bounds(&pe_scaled, perturbation);
        let (_, proj_out, _) = nn_verify::propagate_with_crown_fallback(&proj_graph, &input)
            .expect("proj propagation");

        let start_col = h * d_k;
        let mut k_head = vec![0.0f32; seq_len * d_k];
        for t in 0..seq_len {
            for c in 0..d_k {
                k_head[t * d_k + c] = pe_scaled[[t, start_col + c]];
            }
        }
        let k_tensor = ArrayD::from_shape_vec(IxDyn(&[seq_len, d_k]), k_head).expect("K head");

        let score_def = lw_builders::build_score_layer(&format!("pmh_score_h{h}"), seq_len, d_k);
        let score_bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(k_tensor),
        ];
        let score_graph = tensor_kernel_to_graph(&score_def, &score_bindings).expect("score graph");
        let (_, score_out, _) = nn_verify::propagate_with_crown_fallback(&score_graph, &proj_out)
            .expect("score propagation");

        dominant_counts.push(lw_builders::count_diagonal_dominant(&score_out, seq_len));

        let sm_def = lw_builders::build_softmax_layer(&format!("pmh_sm_h{h}"), seq_len);
        let sm_bindings = vec![TensorParamBinding::Variable];
        let sm_graph = tensor_kernel_to_graph(&sm_def, &sm_bindings).expect("softmax graph");
        let (_, sm_out, _) = nn_verify::propagate_with_crown_fallback(&sm_graph, &score_out)
            .expect("softmax propagation");

        let out_def = lw_builders::build_output_layer(&format!("pmh_out_h{h}"), seq_len, d_k);
        let v_data: Vec<f32> = (0..seq_len * d_k)
            .map(|i| 0.1 * ((i % 5) as f32 - 2.0))
            .collect();
        let v_tensor = ArrayD::from_shape_vec(IxDyn(&[seq_len, d_k]), v_data).expect("V head");
        let out_bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(v_tensor),
        ];
        let out_graph = tensor_kernel_to_graph(&out_def, &out_bindings).expect("output graph");
        let (_, out_out, _) = nn_verify::propagate_with_crown_fallback(&out_graph, &sm_out)
            .expect("output propagation");

        avg_widths.push(lw_builders::measure_avg_width(&out_out));
    }

    (avg_widths, dominant_counts)
}
