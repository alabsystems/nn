// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source generation for cumulative sum (prefix scan) kernels.
//!
//! Extracted from `dyn_tensor_metal_cumsum_ops.rs` for the 500-line limit.
//! Contains the three MSL kernel sources for the Blelloch parallel prefix sum:
//! - Single-threadgroup scan (axis_size <= 256)
//! - Multi-pass block scan (pass 1)
//! - Multi-pass scan-block-sums (pass 2)
//! - Multi-pass propagate (pass 3)

/// MSL for single-threadgroup Blelloch inclusive prefix sum.
///
/// Kernel: `cumsum_f32`
/// Buffers: input(0), output(1), axis_size(2), inner_sz(3)
/// Threadgroups: one per slice, 256 threads each.
pub(super) fn single_pass_msl() -> String {
    let tg_size: usize = 256;
    let tg_size_half = tg_size / 2;
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void cumsum_f32(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    device const uint& axis_size [[buffer(2)]],
    device const uint& inner_sz  [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {{
    // gid = slice index (outer * inner), lid = position along axis
    uint outer_idx = gid / inner_sz;
    uint inner_idx = gid % inner_sz;
    uint base = outer_idx * (axis_size * inner_sz) + inner_idx;
    uint stride = inner_sz;

    threadgroup float shared[256];

    // Load into shared memory
    if (lid < axis_size) {{
        shared[lid] = input[base + lid * stride];
    }} else {{
        shared[lid] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inclusive prefix sum via up-sweep then down-sweep (Blelloch)
    // Up-sweep (reduce)
    for (uint s = 1; s < {tg_size}u; s *= 2) {{
        uint idx = (lid + 1) * s * 2 - 1;
        if (idx < {tg_size}u) {{
            shared[idx] += shared[idx - s];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Clear last element for exclusive scan, then add originals for inclusive
    if (lid == 0) {{
        shared[{tg_size}u - 1] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint s = {tg_size_half}u; s > 0; s /= 2) {{
        uint idx = (lid + 1) * s * 2 - 1;
        if (idx < {tg_size}u) {{
            float tmp = shared[idx];
            shared[idx] += shared[idx - s];
            shared[idx - s] = tmp;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Convert exclusive to inclusive by adding original value
    float orig = (lid < axis_size) ? input[base + lid * stride] : 0.0f;
    if (lid < axis_size) {{
        output[base + lid * stride] = shared[lid] + orig;
    }}
}}
"#,
    )
}

/// MSL for multi-pass block scan (pass 1).
///
/// Kernel: `cumsum_block_scan`
/// Each threadgroup scans a 256-element chunk, storing per-element inclusive
/// prefix sums and each chunk's total.
pub(super) fn block_scan_msl(bs: usize) -> String {
    let bs_half = bs / 2;
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void cumsum_block_scan(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    device float* block_sums        [[buffer(2)]],
    device const uint& axis_size    [[buffer(3)]],
    device const uint& inner_sz     [[buffer(4)]],
    device const uint& num_blocks_v [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]]
) {{
    // Linearized 1D grid: gid encodes (slice_idx, block_idx)
    uint slice_idx = gid / num_blocks_v;
    uint block_idx = gid % num_blocks_v;
    uint outer_idx = slice_idx / inner_sz;
    uint inner_idx = slice_idx % inner_sz;
    uint base = outer_idx * (axis_size * inner_sz) + inner_idx;
    uint stride = inner_sz;

    uint global_pos = block_idx * {bs}u + lid;

    threadgroup float shared[{bs}];

    if (global_pos < axis_size) {{
        shared[lid] = input[base + global_pos * stride];
    }} else {{
        shared[lid] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce)
    for (uint s = 1; s < {bs}u; s *= 2) {{
        uint idx = (lid + 1) * s * 2 - 1;
        if (idx < {bs}u) {{
            shared[idx] += shared[idx - s];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Store block total before clearing
    if (lid == 0) {{
        block_sums[slice_idx * num_blocks_v + block_idx] = shared[{bs}u - 1];
        shared[{bs}u - 1] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint s = {bs_half}u; s > 0; s /= 2) {{
        uint idx = (lid + 1) * s * 2 - 1;
        if (idx < {bs}u) {{
            float tmp = shared[idx];
            shared[idx] += shared[idx - s];
            shared[idx - s] = tmp;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write inclusive result (exclusive scan + original value)
    if (global_pos < axis_size) {{
        float orig = input[base + global_pos * stride];
        output[base + global_pos * stride] = shared[lid] + orig;
    }}
}}
"#,
    )
}

/// MSL for multi-pass scan of block sums (pass 2).
///
/// Kernel: `cumsum_scan_block_sums`
/// Single threadgroup scans the chunk totals from pass 1.
pub(super) fn scan_block_sums_msl(bs: usize) -> String {
    let bs_half = bs / 2;
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void cumsum_scan_block_sums(
    device float* block_sums        [[buffer(0)]],
    device float* scanned_sums      [[buffer(1)]],
    device const uint& num_blocks_v [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {{
    // gid = slice index
    uint base = gid * num_blocks_v;

    threadgroup float shared[{bs}];

    if (lid < num_blocks_v) {{
        shared[lid] = block_sums[base + lid];
    }} else {{
        shared[lid] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep
    for (uint s = 1; s < {bs}u; s *= 2) {{
        uint idx = (lid + 1) * s * 2 - 1;
        if (idx < {bs}u) {{
            shared[idx] += shared[idx - s];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (lid == 0) {{
        shared[{bs}u - 1] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint s = {bs_half}u; s > 0; s /= 2) {{
        uint idx = (lid + 1) * s * 2 - 1;
        if (idx < {bs}u) {{
            float tmp = shared[idx];
            shared[idx] += shared[idx - s];
            shared[idx - s] = tmp;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write inclusive result
    if (lid < num_blocks_v) {{
        float orig = block_sums[base + lid];
        scanned_sums[base + lid] = shared[lid] + orig;
    }}
}}
"#,
    )
}

/// MSL for multi-pass propagation (pass 3).
///
/// Kernel: `cumsum_propagate`
/// Each element adds its chunk's scanned prefix to get the global inclusive
/// prefix sum.
pub(super) const PROPAGATE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void cumsum_propagate(
    device float* output            [[buffer(0)]],
    device const float* scanned_sums [[buffer(1)]],
    device const uint& axis_size    [[buffer(2)]],
    device const uint& inner_sz     [[buffer(3)]],
    device const uint& num_blocks_v [[buffer(4)]],
    device const uint& block_size_v [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    // Decompose tid into (slice_idx, axis_pos)
    uint slice_idx = tid / axis_size;
    uint axis_pos = tid % axis_size;

    uint block_idx = axis_pos / block_size_v;
    if (block_idx == 0) return;  // First block needs no adjustment

    uint outer_idx = slice_idx / inner_sz;
    uint inner_idx = slice_idx % inner_sz;
    uint base = outer_idx * (axis_size * inner_sz) + inner_idx;
    uint stride = inner_sz;

    // Add the inclusive prefix sum of all PRECEDING blocks (block_idx - 1)
    float prefix = scanned_sums[slice_idx * num_blocks_v + block_idx - 1];
    output[base + axis_pos * stride] += prefix;
}
"#;
