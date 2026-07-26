// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error-path tests for DeformableAttention.
//!
//! Covers validation gaps identified by P1 proof_coverage audit:
//! - `num_levels = 0` rejection
//! - Value channel/batch mismatch in multi-scale forward
//! - Value/spatial_shapes per-level mismatch
//! - Module::forward error message

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

#[test]
fn test_new_rejects_zero_levels() {
    let cfg = DeformableAttentionConfig::multi_scale(256, 8, 4, 0);
    let result = DeformableAttention::new(
        make_linear(256, 256),
        make_linear(256, 256),
        make_linear(256, 0), // 8 * 0 * 4 * 2 = 0
        make_linear(256, 0), // 8 * 0 * 4 = 0
        cfg,
    );
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("num_levels"),
        "Expected error mentioning num_levels, got: {err_msg}"
    );
}

#[test]
fn test_multi_scale_value_channel_mismatch() {
    let d = 16;
    let num_heads = 2;
    let num_points = 2;
    let num_levels = 2;

    let offset_dim = num_heads * num_levels * num_points * 2; // 2*2*2*2 = 16
    let weight_dim = num_heads * num_levels * num_points; // 2*2*2 = 8
    let cfg = DeformableAttentionConfig::multi_scale(d, num_heads, num_points, num_levels);

    let layer = DeformableAttention::new(
        make_linear(d, d),
        make_linear(d, d),
        make_linear(d, offset_dim),
        make_linear(d, weight_dim),
        cfg,
    )
    .unwrap();

    let batch = 1;
    let n_q = 4;
    let wrong_d = 8; // d_model mismatch: layer expects d=16, value has 8 channels

    let v1 = DynTensor::from_vec(
        vec![0.5f32; batch * wrong_d * 4 * 4],
        &[batch, wrong_d, 4, 4],
        &Device::Cpu,
    )
    .unwrap();
    let v2 = DynTensor::from_vec(
        vec![0.5f32; batch * wrong_d * 2 * 2],
        &[batch, wrong_d, 2, 2],
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

    let result = layer.forward_multi_scale(&[&v1, &v2], &query, &ref_pts, &[(4, 4), (2, 2)]);
    assert!(result.is_err(), "Expected error for value channel mismatch");
}

#[test]
fn test_multi_scale_value_spatial_mismatch() {
    let d = 16;
    let num_heads = 2;
    let num_points = 2;
    let num_levels = 2;

    let offset_dim = num_heads * num_levels * num_points * 2;
    let weight_dim = num_heads * num_levels * num_points;
    let cfg = DeformableAttentionConfig::multi_scale(d, num_heads, num_points, num_levels);

    let layer = DeformableAttention::new(
        make_linear(d, d),
        make_linear(d, d),
        make_linear(d, offset_dim),
        make_linear(d, weight_dim),
        cfg,
    )
    .unwrap();

    let batch = 1;
    let n_q = 4;

    // v1 has spatial 4x4 but spatial_shapes says 8x8
    let v1 = DynTensor::from_vec(
        vec![0.5f32; batch * d * 4 * 4],
        &[batch, d, 4, 4],
        &Device::Cpu,
    )
    .unwrap();
    let v2 = DynTensor::from_vec(
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

    // spatial_shapes mismatch: v1 is 4x4 but we claim 8x8
    let result = layer.forward_multi_scale(&[&v1, &v2], &query, &ref_pts, &[(8, 8), (2, 2)]);
    assert!(result.is_err(), "Expected error for spatial shape mismatch");
}

#[test]
fn test_multi_scale_spatial_shapes_length_mismatch() {
    let d = 16;
    let num_heads = 2;
    let num_points = 2;
    let num_levels = 2;

    let offset_dim = num_heads * num_levels * num_points * 2;
    let weight_dim = num_heads * num_levels * num_points;
    let cfg = DeformableAttentionConfig::multi_scale(d, num_heads, num_points, num_levels);

    let layer = DeformableAttention::new(
        make_linear(d, d),
        make_linear(d, d),
        make_linear(d, offset_dim),
        make_linear(d, weight_dim),
        cfg,
    )
    .unwrap();

    let batch = 1;
    let n_q = 4;

    let v1 = DynTensor::from_vec(
        vec![0.5f32; batch * d * 4 * 4],
        &[batch, d, 4, 4],
        &Device::Cpu,
    )
    .unwrap();
    let v2 = DynTensor::from_vec(
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

    // Only 1 spatial_shapes entry for 2 levels
    let result = layer.forward_multi_scale(&[&v1, &v2], &query, &ref_pts, &[(4, 4)]);
    assert!(
        result.is_err(),
        "Expected error for spatial_shapes length != num_levels"
    );
}

#[test]
fn test_nan_reference_points_do_not_corrupt_output() {
    // When reference_points contain NaN, the sampling coordinates become NaN.
    // The NaN defense guard (continue on !is_finite) should skip those
    // contributions, producing 0.0 output for affected queries instead of NaN.
    let d = 8;
    let num_heads = 1;
    let num_points = 1;
    let cfg = DeformableAttentionConfig::single_scale(d, num_heads, num_points);
    let offset_dim = num_heads * num_points * 2; // H * K * 2 = 2
    let weight_dim = num_heads * num_points; // H * K = 1

    let layer = DeformableAttention::new(
        make_linear(d, d),
        make_linear(d, d),
        make_linear(d, offset_dim),
        make_linear(d, weight_dim),
        cfg,
    )
    .unwrap();

    let batch = 1;
    let n_q = 2;

    let value = DynTensor::from_vec(
        vec![1.0f32; batch * d * 4 * 4],
        &[batch, d, 4, 4],
        &Device::Cpu,
    )
    .unwrap();
    let query = DynTensor::from_vec(
        vec![0.1f32; batch * n_q * d],
        &[batch, n_q, d],
        &Device::Cpu,
    )
    .unwrap();
    // First query: valid ref point (0.5, 0.5). Second: NaN.
    let ref_pts = DynTensor::from_vec(
        vec![0.5, 0.5, f32::NAN, 0.5],
        &[batch, n_q, 2],
        &Device::Cpu,
    )
    .unwrap();

    let result = layer.forward_single_scale(&value, &query, &ref_pts);
    // Output should not contain NaN — the NaN ref point's contribution is
    // skipped (treated as zero-padding).
    assert!(
        result.is_ok(),
        "forward should not error: {:?}",
        result.err()
    );
    let out = result.unwrap();
    let out_data = out.as_cpu_f32().unwrap();
    for val in out_data.iter() {
        assert!(
            val.is_finite(),
            "output contains non-finite value {val} from NaN reference point"
        );
    }
}
