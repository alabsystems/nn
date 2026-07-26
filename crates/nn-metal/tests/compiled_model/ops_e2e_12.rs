// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests: Upsample1d, Upsample2d, PixelShuffle,
//! PixelUnshuffle.
//!
//! Continuation of `ops_e2e_11.rs` (tests 83+).
//! Fills proof coverage gaps: these vision/spatial ops all compile to
//! `CompiledStep::Dispatch` with reshape+broadcast+permute decompositions
//! but had zero GPU E2E tests.
//!
//! Part of #3020.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, TraceUpsampleMode};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- Test 83: Upsample1d factor=2 --------------------------------------------

/// Upsample1d: [1, 2, 4] → [1, 2, 8] with factor=2.
/// Nearest-neighbor: each time element is repeated `factor` times.
/// Compiles to reshape(insert 1) → broadcast → reshape(merge).
#[test]
fn test_compiled_upsample1d_factor2() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, ch, time) = (1, 2, 4);
    let factor = 2;
    let out_time = time * factor;
    let input_data = super::test_utils::rand_f32_vec(0xC010_0001, batch * ch * time, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, ch, time]),
        TraceNode::new(
            1,
            "upsample1d_0".into(),
            TraceOp::Upsample1d { factor },
            vec![0],
            vec![batch, ch, out_time],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * ch * out_time,
    );

    let expected = cpu_upsample1d(&input_data, batch, ch, time, factor);
    assert_close("upsample1d_factor2", &result, &expected, 0.0);
}

// -- Test 84: Upsample1d factor=3 --------------------------------------------

/// Upsample1d: [1, 1, 5] → [1, 1, 15] with factor=3.
/// Non-power-of-2 factor exercises different broadcast sizes.
#[test]
fn test_compiled_upsample1d_factor3() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, ch, time) = (1, 1, 5);
    let factor = 3;
    let out_time = time * factor;
    let input_data = super::test_utils::rand_f32_vec(0xC010_0002, batch * ch * time, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, ch, time]),
        TraceNode::new(
            1,
            "upsample1d_f3".into(),
            TraceOp::Upsample1d { factor },
            vec![0],
            vec![batch, ch, out_time],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * ch * out_time,
    );

    let expected = cpu_upsample1d(&input_data, batch, ch, time, factor);
    assert_close("upsample1d_factor3", &result, &expected, 0.0);
}

// -- Test 85: Upsample2d nearest 2x2 -----------------------------------------

/// Upsample2d nearest: [1, 1, 3, 4] → [1, 1, 6, 8] with scale_h=2, scale_w=2.
/// Decomposed as: reshape(insert 1 for H) → broadcast(sh) → reshape(merge H)
///              → reshape(insert 1 for W) → broadcast(sw) → reshape(merge W).
#[test]
fn test_compiled_upsample2d_nearest_2x2() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, ch, h, w) = (1, 1, 3, 4);
    let (sh, sw) = (2usize, 2usize);
    let (out_h, out_w) = (h * sh, w * sw);
    let input_data = super::test_utils::rand_f32_vec(0xC020_0001, batch * ch * h * w, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, ch, h, w]),
        TraceNode::new(
            1,
            "upsample2d_0".into(),
            TraceOp::Upsample2d {
                mode: TraceUpsampleMode::Nearest,
                scale_h: sh as f64,
                scale_w: sw as f64,
            },
            vec![0],
            vec![batch, ch, out_h, out_w],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * ch * out_h * out_w,
    );

    let expected = cpu_upsample2d_nearest(&input_data, batch, ch, h, w, sh, sw);
    assert_close("upsample2d_nearest_2x2", &result, &expected, 0.0);
}

// -- Test 86: Upsample2d nearest asymmetric 3x2 ------------------------------

/// Upsample2d nearest: [1, 2, 2, 3] → [1, 2, 6, 6] with scale_h=3, scale_w=2.
/// Asymmetric scales exercise different broadcast dimensions for H and W.
#[test]
fn test_compiled_upsample2d_nearest_3x2() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, ch, h, w) = (1, 2, 2, 3);
    let (sh, sw) = (3usize, 2usize);
    let (out_h, out_w) = (h * sh, w * sw);
    let input_data = super::test_utils::rand_f32_vec(0xC020_0002, batch * ch * h * w, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, ch, h, w]),
        TraceNode::new(
            1,
            "upsample2d_asym".into(),
            TraceOp::Upsample2d {
                mode: TraceUpsampleMode::Nearest,
                scale_h: sh as f64,
                scale_w: sw as f64,
            },
            vec![0],
            vec![batch, ch, out_h, out_w],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * ch * out_h * out_w,
    );

    let expected = cpu_upsample2d_nearest(&input_data, batch, ch, h, w, sh, sw);
    assert_close("upsample2d_nearest_3x2", &result, &expected, 0.0);
}

// -- Test 87: PixelShuffle r=2 -----------------------------------------------

/// PixelShuffle: [1, 8, 2, 3] → [1, 2, 4, 6] with upscale_factor=2.
/// C_in = C_out * r² = 2 * 4 = 8. Reshapes channels into spatial dims.
/// Decomposed as: reshape [B,C,r,r,H,W] → permute [0,1,4,2,5,3] → reshape.
#[test]
fn test_compiled_pixel_shuffle() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let r = 2;
    let (batch, c_out, h, w) = (1, 2, 2, 3);
    let c_in = c_out * r * r; // 8
    let (out_h, out_w) = (h * r, w * r); // 4, 6
    let input_data = super::test_utils::rand_f32_vec(0xC030_0001, batch * c_in * h * w, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, c_in, h, w]),
        TraceNode::new(
            1,
            "pixel_shuffle_0".into(),
            TraceOp::PixelShuffle { upscale_factor: r },
            vec![0],
            vec![batch, c_out, out_h, out_w],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * c_out * out_h * out_w,
    );

    let expected = cpu_pixel_shuffle(&input_data, batch, c_out, h, w, r);
    assert_close("pixel_shuffle_r2", &result, &expected, 0.0);
}

// -- Test 88: PixelUnshuffle r=2 ---------------------------------------------

/// PixelUnshuffle: [1, 2, 4, 6] → [1, 8, 2, 3] with downscale_factor=2.
/// Inverse of PixelShuffle. Packs spatial dims back into channels.
/// Decomposed as: reshape [B,C,H,r,W,r] → permute [0,1,3,5,2,4] → reshape.
#[test]
fn test_compiled_pixel_unshuffle() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let r = 2;
    let (batch, c_in, h_in, w_in) = (1, 2, 4, 6);
    let c_out = c_in * r * r; // 8
    let (h_out, w_out) = (h_in / r, w_in / r); // 2, 3
    let input_data =
        super::test_utils::rand_f32_vec(0xC040_0001, batch * c_in * h_in * w_in, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, c_in, h_in, w_in]),
        TraceNode::new(
            1,
            "pixel_unshuffle_0".into(),
            TraceOp::PixelUnshuffle {
                downscale_factor: r,
            },
            vec![0],
            vec![batch, c_out, h_out, w_out],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * c_out * h_out * w_out,
    );

    let expected = cpu_pixel_unshuffle(&input_data, batch, c_in, h_in, w_in, r);
    assert_close("pixel_unshuffle_r2", &result, &expected, 0.0);
}

// -- Test 89: PixelShuffle → PixelUnshuffle roundtrip -------------------------

/// Pipeline: PixelShuffle(r=2) → PixelUnshuffle(r=2) = identity.
/// [1, 8, 2, 3] → [1, 2, 4, 6] → [1, 8, 2, 3]. Output must match input.
#[test]
fn test_compiled_pixel_shuffle_unshuffle_roundtrip() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let r = 2;
    let (batch, c_out, h, w) = (1, 2, 2, 3);
    let c_in = c_out * r * r; // 8
    let (mid_h, mid_w) = (h * r, w * r); // 4, 6
    let n = batch * c_in * h * w;
    let input_data = super::test_utils::rand_f32_vec(0xC050_0001, n, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, c_in, h, w]),
        TraceNode::new(
            1,
            "pshuffle".into(),
            TraceOp::PixelShuffle { upscale_factor: r },
            vec![0],
            vec![batch, c_out, mid_h, mid_w],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "punshuffle".into(),
            TraceOp::PixelUnshuffle {
                downscale_factor: r,
            },
            vec![1],
            vec![batch, c_in, h, w],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        n,
    );

    assert_close("pixel_roundtrip", &result, &input_data, 0.0);
}

// -- CPU reference helpers ----------------------------------------------------

fn cpu_upsample1d(input: &[f32], batch: usize, ch: usize, time: usize, factor: usize) -> Vec<f32> {
    let out_time = time * factor;
    let mut out = vec![0.0f32; batch * ch * out_time];
    for b in 0..batch {
        for c in 0..ch {
            for t in 0..time {
                let val = input[b * ch * time + c * time + t];
                for f in 0..factor {
                    out[b * ch * out_time + c * out_time + t * factor + f] = val;
                }
            }
        }
    }
    out
}

fn cpu_upsample2d_nearest(
    input: &[f32],
    batch: usize,
    ch: usize,
    h: usize,
    w: usize,
    sh: usize,
    sw: usize,
) -> Vec<f32> {
    let (out_h, out_w) = (h * sh, w * sw);
    let mut out = vec![0.0f32; batch * ch * out_h * out_w];
    for b in 0..batch {
        for c in 0..ch {
            for iy in 0..h {
                for ix in 0..w {
                    let val = input[b * ch * h * w + c * h * w + iy * w + ix];
                    for dy in 0..sh {
                        for dx in 0..sw {
                            let oy = iy * sh + dy;
                            let ox = ix * sw + dx;
                            out[b * ch * out_h * out_w + c * out_h * out_w + oy * out_w + ox] = val;
                        }
                    }
                }
            }
        }
    }
    out
}

/// PixelShuffle CPU reference.
/// `[B, C*r², H, W] → [B, C, H*r, W*r]`
/// output[b, c, h*r+r1, w*r+r2] = input[b, c*r²+r1*r+r2, h, w]
fn cpu_pixel_shuffle(
    input: &[f32],
    batch: usize,
    c_out: usize,
    h: usize,
    w: usize,
    r: usize,
) -> Vec<f32> {
    let c_in = c_out * r * r;
    let (out_h, out_w) = (h * r, w * r);
    let mut out = vec![0.0f32; batch * c_out * out_h * out_w];
    for b in 0..batch {
        for c in 0..c_out {
            for iy in 0..h {
                for ix in 0..w {
                    for r1 in 0..r {
                        for r2 in 0..r {
                            let in_c = c * r * r + r1 * r + r2;
                            let in_idx = b * c_in * h * w + in_c * h * w + iy * w + ix;
                            let oy = iy * r + r1;
                            let ox = ix * r + r2;
                            let out_idx =
                                b * c_out * out_h * out_w + c * out_h * out_w + oy * out_w + ox;
                            out[out_idx] = input[in_idx];
                        }
                    }
                }
            }
        }
    }
    out
}

/// PixelUnshuffle CPU reference.
/// `[B, C, H*r, W*r] → [B, C*r², H, W]`
/// output[b, c*r²+r1*r+r2, h, w] = input[b, c, h*r+r1, w*r+r2]
fn cpu_pixel_unshuffle(
    input: &[f32],
    batch: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    r: usize,
) -> Vec<f32> {
    let c_out = c_in * r * r;
    let (h_out, w_out) = (h_in / r, w_in / r);
    let mut out = vec![0.0f32; batch * c_out * h_out * w_out];
    for b in 0..batch {
        for c in 0..c_in {
            for hy in 0..h_out {
                for wx in 0..w_out {
                    for r1 in 0..r {
                        for r2 in 0..r {
                            let iy = hy * r + r1;
                            let ix = wx * r + r2;
                            let in_idx = b * c_in * h_in * w_in + c * h_in * w_in + iy * w_in + ix;
                            let out_c = c * r * r + r1 * r + r2;
                            let out_idx =
                                b * c_out * h_out * w_out + out_c * h_out * w_out + hy * w_out + wx;
                            out[out_idx] = input[in_idx];
                        }
                    }
                }
            }
        }
    }
    out
}
