// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for primitive multi-mode dispatch (2D, 3D, reduction).
//!
//! Uses hand-written MSL fixtures to verify the dispatch infrastructure
//! binds buffers and configures grids correctly.
//!
//! dvoice kernel dispatch tests (K2, K5, K6) are in
//! `dispatch_modes_from_msl_kernels.rs`.

use nn_metal::{KernelPipeline, MetalBackend, MetalContext, MetalError, PipelineCache};

// ===== 2D Grid Test =====
//
// A simple 2D kernel: output[y * width + x] = input[y * width + x] + 1.0
// The kernel uses `thread_position_in_grid` to compute a 2D index.
const ADD_ONE_2D_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

float add_one_2d(float v) {
    return v + 1.0;
}

kernel void add_one_2d_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint& grid_w [[buffer(2)]],
    device const uint& grid_h [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint idx = pos.y * grid_w + pos.x;
    if (pos.x < grid_w && pos.y < grid_h) {
        output[idx] = add_one_2d(input[idx]);
    }
}
"#;

#[test]
fn test_2d_dispatch_add_one() {
    let _ = MetalBackend::init();
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());

    let pipeline = KernelPipeline::from_msl(&cache, ADD_ONE_2D_MSL, "add_one_2d_kernel", 1, false)
        .expect("compile 2d kernel");

    let width: u32 = 4;
    let height: u32 = 3;
    let total = (width * height) as usize;

    let input: Vec<f32> = (0..total).map(|i| i as f32).collect();

    let result = pipeline
        .dispatch_2d(&ctx, &[&input], [width, height], [4, 3])
        .expect("dispatch 2d");

    assert_eq!(result.len(), total);
    for (i, &v) in result.iter().enumerate() {
        let expected = i as f32 + 1.0;
        assert!(
            (v - expected).abs() < 1e-6,
            "result[{i}] = {v}, expected {expected}"
        );
    }
}

#[test]
fn test_2d_dispatch_wrong_arity_returns_param_count_mismatch() {
    let _ = MetalBackend::init();
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline = KernelPipeline::from_msl(&cache, ADD_ONE_2D_MSL, "add_one_2d_kernel", 1, false)
        .expect("compile 2d kernel");

    let width: u32 = 2;
    let height: u32 = 2;
    let input_a = vec![1.0f32; (width * height) as usize];
    let input_b = vec![2.0f32; (width * height) as usize];

    let err = pipeline
        .dispatch_2d(&ctx, &[&input_a, &input_b], [width, height], [2, 2])
        .expect_err("wrong arity must be rejected");
    assert!(matches!(
        err,
        MetalError::ParamCountMismatch {
            expected: 1,
            got: 2
        }
    ));
}

// ===== 3D Grid Test =====
//
// A 3D kernel: output[z * (W*H) + y * W + x] = input[...] * 2.0
const DOUBLE_3D_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

float double_val(float v) {
    return v * 2.0;
}

kernel void double_3d_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint& grid_x [[buffer(2)]],
    device const uint& grid_y [[buffer(3)]],
    device const uint& grid_z [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint idx = pos.z * grid_x * grid_y + pos.y * grid_x + pos.x;
    if (pos.x < grid_x && pos.y < grid_y && pos.z < grid_z) {
        output[idx] = double_val(input[idx]);
    }
}
"#;

#[test]
fn test_3d_dispatch_double() {
    let _ = MetalBackend::init();
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());

    let pipeline = KernelPipeline::from_msl(&cache, DOUBLE_3D_MSL, "double_3d_kernel", 1, false)
        .expect("compile 3d kernel");

    let dims: [u32; 3] = [3, 4, 2];
    let total = (dims[0] * dims[1] * dims[2]) as usize;

    let input: Vec<f32> = (0..total).map(|i| i as f32 * 0.5).collect();

    let result = pipeline
        .dispatch_3d(&ctx, &[&input], dims, [3, 4, 2])
        .expect("dispatch 3d");

    assert_eq!(result.len(), total);
    for (i, &v) in result.iter().enumerate() {
        let expected = i as f32 * 0.5 * 2.0;
        assert!(
            (v - expected).abs() < 1e-6,
            "result[{i}] = {v}, expected {expected}"
        );
    }
}

#[test]
fn test_3d_dispatch_wrong_arity_returns_param_count_mismatch() {
    let _ = MetalBackend::init();
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline = KernelPipeline::from_msl(&cache, DOUBLE_3D_MSL, "double_3d_kernel", 1, false)
        .expect("compile 3d kernel");

    let dims: [u32; 3] = [2, 2, 2];
    let total = (dims[0] * dims[1] * dims[2]) as usize;
    let input_a = vec![1.0f32; total];
    let input_b = vec![2.0f32; total];

    let err = pipeline
        .dispatch_3d(&ctx, &[&input_a, &input_b], dims, [2, 2, 2])
        .expect_err("wrong arity must be rejected");
    assert!(matches!(
        err,
        MetalError::ParamCountMismatch {
            expected: 1,
            got: 2
        }
    ));
}

// ===== Reduction Test =====
//
// Per-row sum reduction: for each row `r`, output[r] = sum(input[r * cols .. (r+1) * cols])
// Uses threadgroup shared memory for partial sums.
const ROW_SUM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void row_sum_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint& rows [[buffer(2)]],
    device const uint& cols [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Each threadgroup handles one row
    uint row = gid;
    if (row >= rows) return;

    // Each thread accumulates a strided portion of the row
    float sum = 0.0;
    for (uint i = tid; i < cols; i += tg_size) {
        sum += input[row * cols + i];
    }
    shared[tid] = sum;

    // Tree reduction in shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[row] = shared[0];
    }
}
"#;

#[test]
fn test_reduction_row_sum() {
    let _ = MetalBackend::init();
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());

    let pipeline = KernelPipeline::from_msl(&cache, ROW_SUM_MSL, "row_sum_kernel", 1, false)
        .expect("compile reduction kernel");

    let rows: u32 = 8;
    let cols: u32 = 64;
    let threads_per_group: u32 = 32;
    let shared_bytes = threads_per_group * 4; // f32 per thread

    // Input: row r, col c -> value = (r * cols + c) as f32
    let input: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();

    let result = pipeline
        .dispatch_reduction(&ctx, &[&input], rows, cols, threads_per_group, shared_bytes)
        .expect("dispatch reduction");

    assert_eq!(result.len(), rows as usize);
    for r in 0..rows {
        let expected: f32 = (0..cols).map(|c| (r * cols + c) as f32).sum();
        assert!(
            (result[r as usize] - expected).abs() < 1.0,
            "row {r}: result={}, expected={expected}",
            result[r as usize]
        );
    }
}

#[test]
fn test_reduction_dispatch_wrong_arity_returns_param_count_mismatch() {
    let _ = MetalBackend::init();
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline = KernelPipeline::from_msl(&cache, ROW_SUM_MSL, "row_sum_kernel", 1, false)
        .expect("compile reduction kernel");

    let rows: u32 = 2;
    let cols: u32 = 16;
    let threads_per_group: u32 = 8;
    let shared_bytes: u32 = threads_per_group * 4;
    let input_a = vec![1.0f32; (rows * cols) as usize];
    let input_b = vec![2.0f32; (rows * cols) as usize];

    let err = pipeline
        .dispatch_reduction(
            &ctx,
            &[&input_a, &input_b],
            rows,
            cols,
            threads_per_group,
            shared_bytes,
        )
        .expect_err("wrong arity must be rejected");
    assert!(matches!(
        err,
        MetalError::ParamCountMismatch {
            expected: 1,
            got: 2
        }
    ));
}

// ===== Buffer binding slot verification =====

#[test]
fn test_2d_dispatch_verifies_constants_at_correct_slots() {
    // Verify that grid_w is at buffer(2) and grid_h at buffer(3) for a 1-param kernel.
    // The kernel reads grid_w and grid_h and writes them to the output.
    let msl = r#"
#include <metal_stdlib>
using namespace metal;
float echo(float v) { return v; }
kernel void echo_grid_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint& grid_w [[buffer(2)]],
    device const uint& grid_h [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint idx = pos.y * grid_w + pos.x;
    if (pos.x < grid_w && pos.y < grid_h) {
        // Write grid dimensions to first two elements as proof they were bound correctly
        if (idx == 0) { output[0] = float(grid_w); }
        if (idx == 1) { output[1] = float(grid_h); }
        if (idx >= 2) { output[idx] = echo(input[idx]); }
    }
}
"#;

    let _ = MetalBackend::init();
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx.clone());
    let pipeline =
        KernelPipeline::from_msl(&cache, msl, "echo_grid_kernel", 1, false).expect("compile");

    let width: u32 = 8;
    let height: u32 = 4;
    let total = (width * height) as usize;
    let input: Vec<f32> = vec![0.0; total];

    let result = pipeline
        .dispatch_2d(&ctx, &[&input], [width, height], [8, 4])
        .expect("dispatch");

    // First two output elements should be the grid dimensions.
    assert!(
        (result[0] - width as f32).abs() < 1e-6,
        "output[0] should be grid_w={width}, got {}",
        result[0]
    );
    assert!(
        (result[1] - height as f32).abs() < 1e-6,
        "output[1] should be grid_h={height}, got {}",
        result[1]
    );
}
