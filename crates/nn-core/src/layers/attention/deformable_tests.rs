#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::Linear;
use crate::Device;

fn make_linear(in_dim: usize, out_dim: usize) -> Linear {
    let weight = DynTensor::from_vec(
        vec![0.1f32; in_dim * out_dim],
        &[out_dim, in_dim],
        &Device::Cpu,
    )
    .unwrap();
    Linear::new(weight, None).unwrap()
}

fn make_deformable(
    d_model: usize,
    num_heads: usize,
    num_points: usize,
    num_levels: usize,
) -> DeformableAttention {
    let cfg = if num_levels == 1 {
        DeformableAttentionConfig::single_scale(d_model, num_heads, num_points)
    } else {
        DeformableAttentionConfig::multi_scale(d_model, num_heads, num_points, num_levels)
    };
    let offset_dim = num_heads * num_levels * num_points * 2;
    let weight_dim = num_heads * num_levels * num_points;

    DeformableAttention::new(
        make_linear(d_model, d_model),
        make_linear(d_model, d_model),
        make_linear(d_model, offset_dim),
        make_linear(d_model, weight_dim),
        cfg,
    )
    .unwrap()
}

// -- Construction validation tests --

#[test]
fn test_config_single_scale() {
    let cfg = DeformableAttentionConfig::single_scale(256, 8, 4);
    assert_eq!(cfg.d_model, 256);
    assert_eq!(cfg.num_heads, 8);
    assert_eq!(cfg.num_points, 4);
    assert_eq!(cfg.num_levels, 1);
}

#[test]
fn test_config_multi_scale() {
    let cfg = DeformableAttentionConfig::multi_scale(256, 8, 4, 4);
    assert_eq!(cfg.num_levels, 4);
}

#[test]
fn test_new_rejects_zero_d_model() {
    let cfg = DeformableAttentionConfig::single_scale(0, 8, 4);
    let result = DeformableAttention::new(
        make_linear(1, 1),
        make_linear(1, 1),
        make_linear(1, 1),
        make_linear(1, 1),
        cfg,
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("d_model"));
}

#[test]
fn test_new_rejects_zero_heads() {
    let cfg = DeformableAttentionConfig::single_scale(256, 0, 4);
    let result = DeformableAttention::new(
        make_linear(256, 256),
        make_linear(256, 256),
        make_linear(256, 8),
        make_linear(256, 4),
        cfg,
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("num_heads"));
}

#[test]
fn test_new_rejects_zero_points() {
    let cfg = DeformableAttentionConfig::single_scale(256, 8, 0);
    let result = DeformableAttention::new(
        make_linear(256, 256),
        make_linear(256, 256),
        make_linear(256, 16),
        make_linear(256, 8),
        cfg,
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("num_points"));
}

#[test]
fn test_new_rejects_non_divisible() {
    let cfg = DeformableAttentionConfig::single_scale(255, 8, 4);
    let result = DeformableAttention::new(
        make_linear(255, 255),
        make_linear(255, 255),
        make_linear(255, 16),
        make_linear(255, 8),
        cfg,
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("divisible"));
}

// -- Single-scale forward pass tests --

#[test]
fn test_single_scale_output_shape() {
    let d_model = 32;
    let num_heads = 4;
    let num_points = 4;
    let layer = make_deformable(d_model, num_heads, num_points, 1);

    let batch = 2;
    let h = 8;
    let w = 8;
    let n_q = 10;

    let value = DynTensor::from_vec(
        vec![0.5f32; batch * d_model * h * w],
        &[batch, d_model, h, w],
        &Device::Cpu,
    )
    .unwrap();
    let query = DynTensor::from_vec(
        vec![0.1f32; batch * n_q * d_model],
        &[batch, n_q, d_model],
        &Device::Cpu,
    )
    .unwrap();
    // Reference points at center of feature map
    let ref_points = DynTensor::from_vec(
        vec![0.5f32; batch * n_q * 2],
        &[batch, n_q, 2],
        &Device::Cpu,
    )
    .unwrap();

    let output = layer
        .forward_single_scale(&value, &query, &ref_points)
        .unwrap();
    assert_eq!(output.dims(), &[batch, n_q, d_model]);
}

#[test]
fn test_single_scale_different_spatial_sizes() {
    let d = 16;
    let layer = make_deformable(d, 2, 2, 1);

    // Non-square feature map
    let value = DynTensor::from_vec(vec![1.0f32; d * 4 * 6], &[1, d, 4, 6], &Device::Cpu).unwrap();
    let query = DynTensor::from_vec(vec![0.1f32; 5 * d], &[1, 5, d], &Device::Cpu).unwrap();
    let ref_pts = DynTensor::from_vec(vec![0.5f32; 5 * 2], &[1, 5, 2], &Device::Cpu).unwrap();

    let out = layer
        .forward_single_scale(&value, &query, &ref_pts)
        .unwrap();
    assert_eq!(out.dims(), &[1, 5, d]);
}

#[test]
fn test_single_scale_boundary_reference_points() {
    let d = 16;
    let layer = make_deformable(d, 2, 2, 1);

    let value = DynTensor::from_vec(vec![1.0f32; d * 4 * 4], &[1, d, 4, 4], &Device::Cpu).unwrap();
    let query = DynTensor::from_vec(vec![0.1f32; 3 * d], &[1, 3, d], &Device::Cpu).unwrap();
    // Reference points at corners and center
    let ref_pts = DynTensor::from_vec(
        vec![
            0.0, 0.0, // top-left corner
            1.0, 1.0, // bottom-right corner
            0.5, 0.5, // center
        ],
        &[1, 3, 2],
        &Device::Cpu,
    )
    .unwrap();

    let out = layer
        .forward_single_scale(&value, &query, &ref_pts)
        .unwrap();
    assert_eq!(out.dims(), &[1, 3, d]);
    // Output should be finite
    let data = out.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}

// -- Multi-scale forward pass tests --

#[test]
fn test_multi_scale_output_shape() {
    let d = 32;
    let num_heads = 4;
    let num_points = 4;
    let num_levels = 3;
    let layer = make_deformable(d, num_heads, num_points, num_levels);

    let batch = 1;
    let n_q = 8;

    // 3 feature levels at different resolutions
    let v1 = DynTensor::from_vec(
        vec![0.5f32; batch * d * 8 * 8],
        &[batch, d, 8, 8],
        &Device::Cpu,
    )
    .unwrap();
    let v2 = DynTensor::from_vec(
        vec![0.5f32; batch * d * 4 * 4],
        &[batch, d, 4, 4],
        &Device::Cpu,
    )
    .unwrap();
    let v3 = DynTensor::from_vec(
        vec![0.5f32; batch * d * 2 * 2],
        &[batch, d, 2, 2],
        &Device::Cpu,
    )
    .unwrap();

    let query = DynTensor::from_vec(
        vec![0.1f32; batch * n_q * d],
        &[batch, n_q, d],
        &Device::Cpu,
    )
    .unwrap();
    let ref_pts = DynTensor::from_vec(
        vec![0.5f32; batch * n_q * 2],
        &[batch, n_q, 2],
        &Device::Cpu,
    )
    .unwrap();

    let out = layer
        .forward_multi_scale(
            &[&v1, &v2, &v3],
            &query,
            &ref_pts,
            &[(8, 8), (4, 4), (2, 2)],
        )
        .unwrap();
    assert_eq!(out.dims(), &[batch, n_q, d]);
}

#[test]
fn test_multi_scale_level_mismatch_error() {
    let d = 16;
    let layer = make_deformable(d, 2, 2, 2); // expects 2 levels

    let v1 = DynTensor::from_vec(vec![0.5f32; d * 4 * 4], &[1, d, 4, 4], &Device::Cpu).unwrap();
    let query = DynTensor::from_vec(vec![0.1f32; 4 * d], &[1, 4, d], &Device::Cpu).unwrap();
    let ref_pts = DynTensor::from_vec(vec![0.5f32; 4 * 2], &[1, 4, 2], &Device::Cpu).unwrap();

    // Only 1 level provided, expects 2
    let result = layer.forward_multi_scale(&[&v1], &query, &ref_pts, &[(4, 4)]);
    assert!(result.is_err());
    // DataLengthMismatch: expected 2, actual 1 — no "levels" substring in error.
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("expected 2") || msg.contains("length"),
        "error should mention expected count: {msg}"
    );
}

#[test]
fn test_reference_points_batch_mismatch() {
    let d = 16;
    let layer = make_deformable(d, 2, 2, 1);

    let value =
        DynTensor::from_vec(vec![0.5f32; 2 * d * 4 * 4], &[2, d, 4, 4], &Device::Cpu).unwrap();
    let query = DynTensor::from_vec(vec![0.1f32; 2 * 4 * d], &[2, 4, d], &Device::Cpu).unwrap();
    // Wrong batch size for reference points
    let ref_pts = DynTensor::from_vec(vec![0.5f32; 4 * 2], &[1, 4, 2], &Device::Cpu).unwrap();

    let result = layer.forward_single_scale(&value, &query, &ref_pts);
    assert!(result.is_err());
    // ShapeMismatch: expected [2,4,2], actual [1,4,2] — no "reference_points" in error.
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Shape mismatch") || msg.contains("expected"),
        "error should mention shape mismatch: {msg}"
    );
}

// grid_sample tests extracted to deformable_grid_sample_tests.rs.
#[path = "deformable_grid_sample_tests.rs"]
mod grid_sample_tests;
