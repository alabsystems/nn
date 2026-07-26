// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL code generation for spatial structural ops: Broadcast and Transpose.
//!
//! Extracted from `codegen_msl_structural.rs` (#1065 D3) for 500-line compliance.

use crate::codegen_msl;
use crate::codegen_msl_tensor::{TensorMSLCodegenError, TILED_TRANSPOSE_TILE_SIZE};
use crate::ir::ScalarType;
use crate::tensor_ir::BroadcastAlignment;

use super::{row_major_strides, safe_msl_uint};

/// Generate MSL statements that compute `in_idx` from `tid` using modular
/// indexing for broadcast-compatible shapes.
///
/// Output: a String of MSL lines that set `in_idx` = broadcast index of `tid`.
/// Used by both `emit_broadcast_kernel` and broadcast-aware binary op kernels.
pub(crate) fn build_broadcast_index_body(
    input_shape: &[usize],
    output_shape: &[usize],
    alignment: BroadcastAlignment,
) -> Result<String, TensorMSLCodegenError> {
    let rank = output_shape.len();
    let input_rank = input_shape.len();

    let mut body = String::new();
    body.push_str("    uint in_idx = 0;\n");
    body.push_str("    uint remainder = tid;\n");

    let out_strides = row_major_strides(output_shape)?;

    let offset = match alignment {
        BroadcastAlignment::Left => 0,
        BroadcastAlignment::Right => rank.saturating_sub(input_rank),
    };

    let in_strides = row_major_strides(input_shape)?;

    for (i, &stride) in out_strides.iter().enumerate() {
        let s = safe_msl_uint(stride)?;
        body.push_str(&format!(
            "    uint coord_{i} = remainder / {s};\n    remainder = remainder % {s};\n",
        ));
        let input_idx = if i >= offset { i - offset } else { continue };
        if input_idx < input_rank && input_shape[input_idx] > 1 {
            let in_s = safe_msl_uint(in_strides[input_idx])?;
            body.push_str(&format!("    in_idx += coord_{i} * {in_s};\n"));
        }
    }
    Ok(body)
}

/// Emit MSL for a broadcast kernel.
///
/// Maps each output element back to the corresponding input element using
/// modular indexing. Handles both left-aligned and right-aligned broadcast
/// patterns by computing the input index from the output index via strides.
///
/// Buffer layout:
/// - `buffer(0)`: input data
/// - `buffer(1)`: output data
/// - `buffer(2)`: total output elements (uint)
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn emit_broadcast_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    output_shape: &[usize],
    alignment: BroadcastAlignment,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let index_body = build_broadcast_index_body(input_shape, output_shape, alignment)?;

    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    constant uint& total    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total) return;
{index_body}    output[tid] = input[in_idx];
}}"#,
    ))
}

/// Emit a Metal transpose kernel that permutes axes via index remapping.
///
/// For each output thread index `tid`, decomposes into output multi-index
/// coordinates, then recomposes into input flat index using the permutation.
/// Output coordinate `c[i]` maps to input axis `axes[i]`.
///
/// For a 2D transpose `[1, 0]` of shape `[M, N]`:
///   `output[row * N + col] = input[col * M + row]`
///
/// Part of #809.
pub(crate) fn emit_transpose_kernel(
    name: &str,
    dtype: ScalarType,
    input_shape: &[usize],
    axes: &[usize],
    total_elements: usize,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);

    // Output shape = input_shape permuted by axes.
    let out_shape: Vec<usize> = axes.iter().map(|&a| input_shape[a]).collect();

    let out_strides = row_major_strides(&out_shape)?;
    let in_strides = row_major_strides(input_shape)?;

    let _ = total_elements; // total passed via Metal buffer, not embedded

    let mut body = String::new();
    body.push_str("    uint in_idx = 0;\n");
    body.push_str("    uint remainder = tid;\n");

    for (out_dim, &out_stride) in out_strides.iter().enumerate() {
        let os = safe_msl_uint(out_stride)?;
        // This output dimension corresponds to input axis axes[out_dim].
        let in_axis = axes[out_dim];
        let is = safe_msl_uint(in_strides[in_axis])?;
        body.push_str(&format!("    uint c{out_dim} = remainder / {os};\n"));
        body.push_str(&format!("    remainder = remainder % {os};\n"));
        body.push_str(&format!("    in_idx += c{out_dim} * {is};\n"));
    }

    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    constant uint& total    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total) return;
{body}    output[tid] = input[in_idx];
}}"#,
    ))
}

/// Emit MSL for a tiled shared-memory 2D transpose. Part of #3230 (Gap 4).
///
/// Uses a TILE×TILE threadgroup tile with +1 column padding to avoid shared
/// memory bank conflicts. Coalesced reads from global → shared, barrier,
/// then coalesced writes from shared → global in the transposed order.
/// 2-5× faster than naive per-element transpose for large matrices.
///
/// Handles batched transposes: `[B, M, N] → [B, N, M]` via grid z-dimension.
///
/// Buffer layout:
/// - `buffer(0)`: input data  `[batch * M * N]`
/// - `buffer(1)`: output data `[batch * N * M]`
/// - `buffer(2)`: M (uint, rows per batch matrix)
/// - `buffer(3)`: N (uint, cols per batch matrix)
///
/// Dispatch: `(ceil(N/TILE), ceil(M/TILE), batch)` threadgroups of `(TILE, TILE, 1)`.
pub(crate) fn emit_tiled_transpose_2d_kernel(name: &str, dtype: ScalarType) -> String {
    let t = codegen_msl::msl_type(dtype);
    let tile = TILED_TRANSPOSE_TILE_SIZE;
    let tile_pad = tile + 1; // +1 avoids shared memory bank conflicts

    format!(
        r#"[[kernel]] void {name}(
    device const {t}* input  [[buffer(0)]],
    device {t}* output       [[buffer(1)]],
    constant uint& M         [[buffer(2)]],
    constant uint& N         [[buffer(3)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid    [[thread_position_in_threadgroup]]
) {{
    threadgroup {t} tile[{tile}][{tile_pad}];
    uint batch_off = tg_pos.z * M * N;

    // Coalesced read: consecutive threads (tid.x) read consecutive columns.
    uint col = tg_pos.x * {tile} + tid.x;
    uint row = tg_pos.y * {tile} + tid.y;
    if (row < M && col < N) {{
        tile[tid.y][tid.x] = input[batch_off + row * N + col];
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Coalesced write: consecutive threads (tid.x) write consecutive output columns.
    // Output is [N, M]: element (out_row, out_col) at out_row * M + out_col.
    uint out_col = tg_pos.y * {tile} + tid.x;
    uint out_row = tg_pos.x * {tile} + tid.y;
    if (out_row < N && out_col < M) {{
        output[batch_off + out_row * M + out_col] = tile[tid.x][tid.y];
    }}
}}"#,
    )
}
