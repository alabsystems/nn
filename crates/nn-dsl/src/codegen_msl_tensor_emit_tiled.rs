// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL emission for tiled shared-memory GEMM kernels.
//!
//! Middle-tier GEMM using 16×16 threadgroup tiles with shared memory.
//! Each threadgroup computes one output tile by iterating over K in
//! chunks of TILE_K, loading A and B tiles into threadgroup memory.
//!
//! Part of #3230 (Gap 1).

use crate::codegen_msl;
use crate::codegen_msl_structural;
use crate::codegen_msl_tensor::TensorMSLCodegenError;
use crate::ir::ScalarType;

/// Emit MSL source for a tiled linear (fully-connected) kernel.
///
/// Uses 16×16 threadgroup tiles with shared memory. Inner dimension
/// padded to 17 to avoid bank conflicts.
pub(crate) fn emit_tiled_linear_kernel(
    name: &str,
    dtype: ScalarType,
    in_features: usize,
    out_features: usize,
    batch_size: usize,
    has_bias: bool,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let needs_cast = t != acc;
    let in_feat = codegen_msl_structural::safe_msl_uint(in_features)?;
    let out_feat = codegen_msl_structural::safe_msl_uint(out_features)?;
    let m_val = codegen_msl_structural::safe_msl_uint(batch_size)?;

    let (bias_buf, out_buf) = if has_bias { ("2", "3") } else { ("", "2") };

    let mut params = format!(
        "    device const {t}* input  [[buffer(0)]],\n\
         \x20   device const {t}* weight [[buffer(1)]],\n"
    );
    if has_bias {
        params.push_str(&format!(
            "    device const {t}* bias   [[buffer({bias_buf})]],\n"
        ));
    }
    params.push_str(&format!(
        "    device {t}* output         [[buffer({out_buf})]],\n\
         \x20   uint2 tg_pos [[threadgroup_position_in_grid]],\n\
         \x20   uint2 local  [[thread_position_in_threadgroup]]"
    ));

    let load_a = if needs_cast {
        format!("{acc}(input[row * IN_FEATURES + k0 + local.x])")
    } else {
        "input[row * IN_FEATURES + k0 + local.x]".to_string()
    };
    let load_b = if needs_cast {
        format!("{acc}(weight[col * IN_FEATURES + k0 + local.y])")
    } else {
        "weight[col * IN_FEATURES + k0 + local.y]".to_string()
    };
    let store_expr = if needs_cast {
        format!("{t}(sum)")
    } else {
        "sum".to_string()
    };

    let bias_line = if has_bias {
        if needs_cast {
            format!("        sum += {acc}(bias[col]);\n")
        } else {
            "        sum += bias[col];\n".to_string()
        }
    } else {
        String::new()
    };

    Ok(format!(
        r#"[[kernel]] void {name}(
{params}
) {{
    const uint TILE = 16;
    const uint IN_FEATURES = {in_feat};
    const uint OUT_FEATURES = {out_feat};
    const uint M = {m_val};

    uint row = tg_pos.y * TILE + local.y;
    uint col = tg_pos.x * TILE + local.x;

    // Pad inner dim to 17 to avoid shared memory bank conflicts.
    threadgroup {acc} tile_a[16][17];
    threadgroup {acc} tile_b[16][17];

    {acc} sum = 0;
    for (uint k0 = 0; k0 < IN_FEATURES; k0 += TILE) {{
        // Cooperative load: each thread loads one element of each tile.
        if (row < M && (k0 + local.x) < IN_FEATURES)
            tile_a[local.y][local.x] = {load_a};
        else
            tile_a[local.y][local.x] = 0;

        if (col < OUT_FEATURES && (k0 + local.y) < IN_FEATURES)
            tile_b[local.y][local.x] = {load_b};
        else
            tile_b[local.y][local.x] = 0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE; kk++) {{
            sum += tile_a[local.y][kk] * tile_b[kk][local.x];
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (row < M && col < OUT_FEATURES) {{
{bias_line}        output[row * OUT_FEATURES + col] = {store_expr};
    }}
}}"#,
    ))
}

/// Emit MSL source for a tiled batched matmul kernel.
///
/// Uses 16×16 threadgroup tiles with shared memory. Supports transposed
/// right, broadcast right, and optional scale.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_tiled_matmul_kernel(
    name: &str,
    dtype: ScalarType,
    m: usize,
    k: usize,
    n: usize,
    _batch_size: usize,
    transpose_right: bool,
    broadcast_right: bool,
    scale: Option<f32>,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let needs_cast = t != acc;
    let m_val = codegen_msl_structural::safe_msl_uint(m)?;
    let k_val = codegen_msl_structural::safe_msl_uint(k)?;
    let n_val = codegen_msl_structural::safe_msl_uint(n)?;
    let mk = codegen_msl_structural::safe_msl_uint(m * k)?;

    let batch_offset_r_line = if broadcast_right {
        "    uint batch_offset_r = 0;".to_string()
    } else {
        let right_batch_stride = if transpose_right {
            codegen_msl_structural::safe_msl_uint(n * k)?
        } else {
            codegen_msl_structural::safe_msl_uint(k * n)?
        };
        format!("    uint batch_offset_r = batch_idx * ({right_batch_stride});")
    };

    // tile_b loading: right matrix index depends on transpose_right.
    // For tile_b[local.y][local.x], we load right[col_of_right, k0+local.y]
    let right_load_index = if transpose_right {
        // right is [*, N, K]: right[col * K + (k0 + local.y)]
        "batch_offset_r + col * K + k0 + local.y"
    } else {
        // right is [*, K, N]: right[(k0 + local.y) * N + col]
        "batch_offset_r + (k0 + local.y) * N + col"
    };

    let load_left = if needs_cast {
        format!("{acc}(left[batch_offset_l + row * K + k0 + local.x])")
    } else {
        "left[batch_offset_l + row * K + k0 + local.x]".to_string()
    };
    let load_right = if needs_cast {
        format!("{acc}(right[{right_load_index}])")
    } else {
        format!("right[{right_load_index}]")
    };
    let store_expr = if needs_cast {
        format!("{t}(sum)")
    } else {
        "sum".to_string()
    };
    let mn = codegen_msl_structural::safe_msl_uint(m * n)?;

    let scale_line = match scale {
        Some(s) => format!("        sum *= {acc}({s});\n"),
        None => String::new(),
    };

    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* left   [[buffer(0)]],
    device const {t}* right  [[buffer(1)]],
    device {t}* output       [[buffer(2)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 local  [[thread_position_in_threadgroup]]
) {{
    const uint TILE = 16;
    const uint M = {m_val};
    const uint K = {k_val};
    const uint N = {n_val};

    uint batch_idx = tg_pos.z;
    uint row = tg_pos.y * TILE + local.y;
    uint col = tg_pos.x * TILE + local.x;

    uint batch_offset_l = batch_idx * ({mk});
{batch_offset_r_line}

    threadgroup {acc} tile_a[16][17];
    threadgroup {acc} tile_b[16][17];

    {acc} sum = 0;
    for (uint k0 = 0; k0 < K; k0 += TILE) {{
        if (row < M && (k0 + local.x) < K)
            tile_a[local.y][local.x] = {load_left};
        else
            tile_a[local.y][local.x] = 0;

        if (col < N && (k0 + local.y) < K)
            tile_b[local.y][local.x] = {load_right};
        else
            tile_b[local.y][local.x] = 0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE; kk++) {{
            sum += tile_a[local.y][kk] * tile_b[kk][local.x];
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (row < M && col < N) {{
{scale_line}        output[batch_idx * ({mn}) + row * N + col] = {store_expr};
    }}
}}"#,
    ))
}
