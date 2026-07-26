// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! dvoice kernel dispatch tests: instance_norm (K2), rms_norm (K5), RoPE (K6).
//!
//! Each test compiles hand-written MSL, dispatches via `DispatchMode`, and
//! compares results against a Rust reference implementation.

use nn_core::test_utils::assert_close_with_label as assert_close;
use nn_metal::{
    flush, BufferBinding, DispatchMode, KernelPipeline, MetalBackend, MetalContext, MetalError,
    PipelineCache,
};

// ===== Shared helpers =====

fn threadgroup_width_1d(total: u32) -> u32 {
    if total < 64 {
        total
    } else {
        64
    }
}

fn instance_norm_reference(
    x: &[f32],
    batch: usize,
    channels: usize,
    t_len: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for b in 0..batch {
        for c in 0..channels {
            let base = (b * channels + c) * t_len;
            let mut sum = 0.0;
            for t in 0..t_len {
                sum += x[base + t];
            }
            let mean = sum / t_len as f32;

            let mut var_sum = 0.0;
            for t in 0..t_len {
                let d = x[base + t] - mean;
                var_sum += d * d;
            }
            let inv_std = 1.0 / (var_sum / t_len as f32 + eps).sqrt();

            for t in 0..t_len {
                out[base + t] = gamma[c] * ((x[base + t] - mean) * inv_std) + beta[c];
            }
        }
    }
    out
}

fn rms_norm_reference(x: &[f32], rows: usize, hidden: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for r in 0..rows {
        let base = r * hidden;
        let mut ss = 0.0;
        for i in 0..hidden {
            let v = x[base + i];
            ss += v * v;
        }
        let scale = 1.0 / (ss / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            out[base + i] = x[base + i] * scale * weight[i];
        }
    }
    out
}

fn rope_reference(
    mut q: Vec<f32>,
    mut k: Vec<f32>,
    freqs: &[f32],
    batch: usize,
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    for hb in 0..(batch * n_heads) {
        for pos in 0..seq_len {
            for pair in 0..half_dim {
                let freq = freqs[pos * half_dim + pair];
                let cos_f = freq.cos();
                let sin_f = freq.sin();
                let base = hb * seq_len * head_dim + pos * head_dim + pair * 2;

                let q0 = q[base];
                let q1 = q[base + 1];
                q[base] = q0 * cos_f - q1 * sin_f;
                q[base + 1] = q0 * sin_f + q1 * cos_f;

                let k0 = k[base];
                let k1 = k[base + 1];
                k[base] = k0 * cos_f - k1 * sin_f;
                k[base + 1] = k0 * sin_f + k1 * cos_f;
            }
        }
    }
    (q, k)
}

// ===== K2: Instance Norm =====

const K2_INSTANCE_NORM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void instance_norm_1d_f32(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    constant uint& T_len [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint& has_affine [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint bc = tid;
    uint c = bc % C;
    uint base = bc * T_len;

    float sum = 0.0;
    for (uint t = 0; t < T_len; t++) {
        sum += x[base + t];
    }
    float mean = sum / float(T_len);

    float var_sum = 0.0;
    for (uint t = 0; t < T_len; t++) {
        float d = x[base + t] - mean;
        var_sum += d * d;
    }
    float inv_std = metal::rsqrt(var_sum / float(T_len) + eps);

    float g = has_affine ? gamma[c] : 1.0;
    float b = has_affine ? beta[c] : 0.0;
    for (uint t = 0; t < T_len; t++) {
        out[base + t] = g * ((x[base + t] - mean) * inv_std) + b;
    }
}
"#;

#[test]
fn test_k2_instance_norm_dispatch_matches_dvoice_reference() {
    let _ = MetalBackend::init();
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline = KernelPipeline::from_msl(
        &cache,
        K2_INSTANCE_NORM_MSL,
        "instance_norm_1d_f32",
        3,
        false,
    )
    .expect("compile K2 instance norm kernel");

    let batch = 2usize;
    let channels = 3usize;
    let t_len = 7usize;
    let batch_channels = batch * channels;
    let eps = 1e-5f32;

    let x: Vec<f32> = (0..(batch_channels * t_len))
        .map(|i| (i as f32 * 0.125) - 2.0)
        .collect();
    let gamma = vec![1.0f32, 0.75, 1.25];
    let beta = vec![0.2f32, -0.1, 0.05];
    let expected = instance_norm_reference(&x, batch, channels, t_len, &gamma, &beta, eps);

    let x_buf = ctx.create_buffer(&x).expect("x buffer");
    let gamma_buf = ctx.create_buffer(&gamma).expect("gamma buffer");
    let beta_buf = ctx.create_buffer(&beta).expect("beta buffer");
    let out_buf = ctx
        .create_buffer_zeroed(x.len() * size_of::<f32>())
        .expect("output buffer");

    let plan = DispatchMode::Elementwise {
        total: batch_channels as u32,
    }
    .plan()
    .expect("instance norm dispatch plan")
    .with_output_elems(x.len())
    .with_constants(vec![channels as u32, t_len as u32, eps.to_bits(), 1]);

    pipeline
        .dispatch_buffers(&ctx, &[&x_buf, &gamma_buf, &beta_buf], &out_buf, &plan)
        .expect("dispatch K2 instance norm");

    // Flush the lazy GPU batch before CPU readback (#2009).
    flush().expect("flush before readback");
    let actual = out_buf.contents::<f32>().expect("read output").to_vec();
    assert_close(&actual, &expected, 1e-5, "k2_instance_norm");
}

// ===== K5: RMS Norm =====

const K5_RMS_NORM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint base = tid * hidden;
    float ss = 0.0;
    for (uint i = 0; i < hidden; i++) {
        float v = x[base + i];
        ss += v * v;
    }
    float scale = metal::rsqrt(ss / float(hidden) + eps);
    for (uint i = 0; i < hidden; i++) {
        out[base + i] = x[base + i] * scale * weight[i];
    }
}
"#;

#[test]
fn test_k5_rms_norm_dispatch_matches_dvoice_reference() {
    let _ = MetalBackend::init();
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline = KernelPipeline::from_msl(&cache, K5_RMS_NORM_MSL, "rms_norm_f32", 2, false)
        .expect("compile K5 RMS norm kernel");

    let rows = 5usize;
    let hidden = 8usize;
    let eps = 1e-6f32;

    let x: Vec<f32> = (0..(rows * hidden))
        .map(|i| ((i as i32 - 15) as f32) * 0.05)
        .collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 0.8 + (i as f32) * 0.03).collect();
    let expected = rms_norm_reference(&x, rows, hidden, &weight, eps);

    let x_buf = ctx.create_buffer(&x).expect("x buffer");
    let weight_buf = ctx.create_buffer(&weight).expect("weight buffer");
    let out_buf = ctx
        .create_buffer_zeroed(x.len() * size_of::<f32>())
        .expect("output buffer");

    let plan = DispatchMode::Elementwise { total: rows as u32 }
        .plan()
        .expect("rms norm dispatch plan")
        .with_output_elems(x.len())
        .with_constants(vec![hidden as u32, eps.to_bits()]);

    pipeline
        .dispatch_buffers(&ctx, &[&x_buf, &weight_buf], &out_buf, &plan)
        .expect("dispatch K5 RMS norm");

    flush().expect("flush before readback");
    let actual = out_buf.contents::<f32>().expect("read output").to_vec();
    assert_close(&actual, &expected, 1e-5, "k5_rms_norm");
}

// ===== K6: RoPE =====

const K6_ROPE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rope_qk_f32(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const float* freqs [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint pos = tid.x;
    uint pair = tid.y;
    uint hb = tid.z;
    if (pos >= seq_len || pair >= head_dim / 2) return;

    float freq = freqs[pos * (head_dim / 2) + pair];
    float cos_f = metal::cos(freq);
    float sin_f = metal::sin(freq);
    uint base = hb * seq_len * head_dim + pos * head_dim + pair * 2;

    float q0 = q[base];
    float q1 = q[base + 1];
    q[base] = q0 * cos_f - q1 * sin_f;
    q[base + 1] = q0 * sin_f + q1 * cos_f;

    float k0 = k[base];
    float k1 = k[base + 1];
    k[base] = k0 * cos_f - k1 * sin_f;
    k[base + 1] = k0 * sin_f + k1 * cos_f;
}
"#;

#[test]
fn test_k6_rope_dispatch_matches_dvoice_reference() {
    let _ = MetalBackend::init();
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline = KernelPipeline::from_msl(&cache, K6_ROPE_MSL, "rope_qk_f32", 3, false)
        .expect("compile K6 RoPE kernel");

    let batch = 2usize;
    let n_heads = 2usize;
    let seq_len = 3usize;
    let head_dim = 6usize;
    let half_dim = head_dim / 2;

    let elems = batch * n_heads * seq_len * head_dim;
    let q: Vec<f32> = (0..elems).map(|i| i as f32 * 0.01 - 0.4).collect();
    let k: Vec<f32> = (0..elems).map(|i| i as f32 * -0.02 + 0.25).collect();
    let freqs: Vec<f32> = (0..(seq_len * half_dim))
        .map(|i| i as f32 * 0.1 - 0.2)
        .collect();

    let (expected_q, expected_k) = rope_reference(
        q.clone(),
        k.clone(),
        &freqs,
        batch,
        n_heads,
        seq_len,
        head_dim,
    );

    let q_buf = ctx.create_buffer(&q).expect("q buffer");
    let k_buf = ctx.create_buffer(&k).expect("k buffer");
    let freqs_buf = ctx.create_buffer(&freqs).expect("freqs buffer");

    let plan = DispatchMode::Grid3D {
        grid: [seq_len as u32, half_dim as u32, (n_heads * batch) as u32],
        threads: [threadgroup_width_1d(seq_len as u32), 1, 1],
    }
    .plan()
    .expect("rope dispatch plan")
    .with_constants(vec![n_heads as u32, seq_len as u32, head_dim as u32]);

    pipeline
        .dispatch_bindings(
            &ctx,
            &[
                BufferBinding::read_write(&q_buf),
                BufferBinding::read_write(&k_buf),
                BufferBinding::read_only(&freqs_buf),
            ],
            &plan,
        )
        .expect("dispatch K6 RoPE");

    flush().expect("flush before readback");
    let actual_q = q_buf.contents::<f32>().expect("read output").to_vec();
    let actual_k = k_buf.contents::<f32>().expect("read output").to_vec();
    assert_close(&actual_q, &expected_q, 1e-5, "k6_rope_q");
    assert_close(&actual_k, &expected_k, 1e-5, "k6_rope_k");
}

// ===== Dispatch binding role validation =====

#[test]
fn test_dispatch_bindings_rejects_all_read_only_roles() {
    let _ = MetalBackend::init();
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline = KernelPipeline::from_msl(&cache, K6_ROPE_MSL, "rope_qk_f32", 3, false)
        .expect("compile K6 RoPE kernel");

    let q = vec![0.0f32; 2];
    let k = vec![0.0f32; 2];
    let freqs = vec![0.0f32; 1];
    let q_buf = ctx.create_buffer(&q).expect("q buffer");
    let k_buf = ctx.create_buffer(&k).expect("k buffer");
    let freqs_buf = ctx.create_buffer(&freqs).expect("freqs buffer");
    let plan = DispatchMode::Grid3D {
        grid: [1, 1, 1],
        threads: [1, 1, 1],
    }
    .plan()
    .expect("role validation plan")
    .with_constants(vec![1, 1, 2]);

    let err = pipeline
        .dispatch_bindings(
            &ctx,
            &[
                BufferBinding::read_only(&q_buf),
                BufferBinding::read_only(&k_buf),
                BufferBinding::read_only(&freqs_buf),
            ],
            &plan,
        )
        .expect_err("read-only layout should be rejected");
    assert!(matches!(err, MetalError::InvalidDispatchBindings(_)));
}
