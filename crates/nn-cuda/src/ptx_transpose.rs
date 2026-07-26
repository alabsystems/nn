// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for 2D matrix transpose.
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 matrix
//! transpose using shared memory tiling for coalesced global memory access.
//!
//! ## Algorithm
//!
//! Tile-based transpose using shared memory to convert column reads into
//! coalesced writes:
//!
//! 1. Each thread block loads a `TILE x TILE` tile from the input matrix
//!    into shared memory (coalesced row reads).
//! 2. Synchronize via `bar.sync`.
//! 3. Each thread writes a transposed element from shared memory to global
//!    memory (coalesced column writes become coalesced row writes in output).
//!
//! Shared memory is padded by 1 element per row (`TILE + 1` stride) to avoid
//! bank conflicts during the transposed read.
//!
//! ## Kernel interface
//!
//! 2D transpose parameters:
//! - `param_input`  -- pointer to input matrix (f32), row-major `[rows, cols]`
//! - `param_output` -- pointer to output matrix (f32), row-major `[cols, rows]`
//! - `param_rows`   -- u32, number of rows in the input matrix
//! - `param_cols`   -- u32, number of columns in the input matrix
//!
//! Batched transpose adds:
//! - `param_batch`  -- u32, number of matrices in the batch
//!
//! ## Thread block configuration
//!
//! Block: `(TILE, TILE, 1)` = `(16, 16, 1)` = 256 threads.
//! Grid: `(ceil(cols / TILE), ceil(rows / TILE), 1)` for 2D.
//! Grid: `(ceil(cols / TILE), ceil(rows / TILE), batch)` for batched.

use crate::codegen_ptx::ptx_prelude;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Tile size for the transpose kernel (16x16 = 256 threads per block).
///
/// 16x16 is the sweet spot: fits in shared memory, provides good occupancy,
/// and enables coalesced access patterns on all NVIDIA architectures.
pub const TRANSPOSE_BLOCK_SIZE: u32 = 16;

/// SM target for transpose kernels.
const SM_TARGET: &str = "sm_70";

// ---------------------------------------------------------------------------
// 2D Transpose
// ---------------------------------------------------------------------------

/// Generate PTX for 2D matrix transpose using shared memory tiling.
///
/// Transposes a `[rows, cols]` row-major matrix to `[cols, rows]`.
/// Uses `TRANSPOSE_BLOCK_SIZE x TRANSPOSE_BLOCK_SIZE` tiles with shared
/// memory padding to avoid bank conflicts.
///
/// # Arguments
/// * `rows` -- number of rows in the input matrix
/// * `cols` -- number of columns in the input matrix
///
/// # Example
/// ```
/// use nn_cuda::ptx_transpose::generate_transpose_ptx;
/// let ptx = generate_transpose_ptx(64, 128);
/// assert!(ptx.contains(".entry ptx_transpose_f32"));
/// assert!(ptx.contains(".shared"));
/// ```
#[must_use]
pub fn generate_transpose_ptx(rows: u32, cols: u32) -> String {
    let tile = TRANSPOSE_BLOCK_SIZE;
    // Shared memory: TILE rows x (TILE+1) cols for bank-conflict-free access
    let smem_stride = tile + 1;

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// 2D Transpose f32: rows={rows}, cols={cols}, tile={tile}\n\
         // Shared memory stride={smem_stride} (padded to avoid bank conflicts)\n\n"
    ));

    // Shared memory declaration
    ptx.push_str(&format!(
        ".shared .align 4 .f32 tile_smem[{smem_total}];\n\n",
        smem_total = tile * smem_stride
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry ptx_transpose_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_rows,\n\
         \x20   .param .u32 param_cols\n\
         )\n\
         .reqntid {tile}, {tile}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<20>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<10>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_rows];\n\
         \x20   ld.param.u32  %r1,  [param_cols];\n\n",
    );

    // Thread indices within tile
    ptx.push_str(
        "\x20   // Thread indices\n\
         \x20   mov.u32       %r2, %tid.x;           // tx = threadIdx.x\n\
         \x20   mov.u32       %r3, %tid.y;           // ty = threadIdx.y\n\n",
    );

    // Global coordinates for the input read
    ptx.push_str(&format!(
        "\x20   // Input coordinates: (row, col) in input matrix\n\
         \x20   mov.u32       %r4, %ctaid.x;\n\
         \x20   mul.lo.u32    %r5, %r4, {tile};      // block_col_start\n\
         \x20   add.u32       %r6, %r5, %r2;         // col = block_col_start + tx\n\
         \x20   mov.u32       %r7, %ctaid.y;\n\
         \x20   mul.lo.u32    %r8, %r7, {tile};      // block_row_start\n\
         \x20   add.u32       %r9, %r8, %r3;         // row = block_row_start + ty\n\n"
    ));

    // Bounds check and load into shared memory
    ptx.push_str(&format!(
        "\x20   // Load input[row, col] into shared memory tile[ty][tx]\n\
         \x20   setp.lt.u32   %p0, %r9, %r0;         // row < rows?\n\
         \x20   setp.lt.u32   %p1, %r6, %r1;         // col < cols?\n\
         \x20   and.pred      %p2, %p0, %p1;          // in bounds?\n\
         \x20   @!%p2 bra     TR_SKIP_LOAD;\n\
         \x20   // input_offset = (row * cols + col) * 4\n\
         \x20   mad.lo.u32    %r10, %r9, %r1, %r6;   // row * cols + col\n\
         \x20   mul.wide.u32  %rd2, %r10, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\
         \x20   // smem_offset = (ty * smem_stride + tx) * 4\n\
         \x20   mad.lo.u32    %r11, %r3, {smem_stride}, %r2;\n\
         \x20   mul.wide.u32  %rd4, %r11, 4;\n\
         \x20   mov.u64       %rd5, tile_smem;\n\
         \x20   add.u64       %rd6, %rd5, %rd4;\n\
         \x20   st.shared.f32 [%rd6], %f0;\n\
         TR_SKIP_LOAD:\n\n"
    ));

    // Synchronize
    ptx.push_str("\x20   bar.sync      0;\n\n");

    // Output coordinates: transpose the tile indices
    ptx.push_str("\x20   // Output coordinates: (out_row, out_col) in output matrix\n\
         \x20   // out_row = block_col_start + ty (transposed)\n\
         \x20   // out_col = block_row_start + tx (transposed)\n\
         \x20   add.u32       %r12, %r5, %r3;        // out_row = block_col_start + ty\n\
         \x20   add.u32       %r13, %r8, %r2;        // out_col = block_row_start + tx\n\n");

    // Bounds check and store from shared memory
    ptx.push_str(&format!(
        "\x20   // Store tile[tx][ty] to output[out_row, out_col]\n\
         \x20   setp.lt.u32   %p0, %r12, %r1;        // out_row < cols?\n\
         \x20   setp.lt.u32   %p1, %r13, %r0;        // out_col < rows?\n\
         \x20   and.pred      %p2, %p0, %p1;\n\
         \x20   @!%p2 bra     TR_EXIT;\n\
         \x20   // Read from smem: tile[tx][ty] (transposed read)\n\
         \x20   mad.lo.u32    %r14, %r2, {smem_stride}, %r3;\n\
         \x20   mul.wide.u32  %rd4, %r14, 4;\n\
         \x20   mov.u64       %rd5, tile_smem;\n\
         \x20   add.u64       %rd6, %rd5, %rd4;\n\
         \x20   ld.shared.f32 %f1, [%rd6];\n\
         \x20   // output_offset = (out_row * rows + out_col) * 4\n\
         \x20   mad.lo.u32    %r15, %r12, %r0, %r13;\n\
         \x20   mul.wide.u32  %rd7, %r15, 4;\n\
         \x20   add.u64       %rd8, %rd1, %rd7;\n\
         \x20   st.global.f32 [%rd8], %f1;\n\n"
    ));

    ptx.push_str(
        "TR_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Batched Transpose
// ---------------------------------------------------------------------------

/// Generate PTX for batched 2D matrix transpose.
///
/// Transposes `batch` matrices, each `[rows, cols]` row-major, to
/// `[cols, rows]`. Uses `blockIdx.z` for the batch dimension.
///
/// # Arguments
/// * `batch` -- batch size (number of matrices)
/// * `rows`  -- number of rows per matrix
/// * `cols`  -- number of columns per matrix
///
/// # Example
/// ```
/// use nn_cuda::ptx_transpose::generate_batch_transpose_ptx;
/// let ptx = generate_batch_transpose_ptx(4, 32, 64);
/// assert!(ptx.contains(".entry ptx_batch_transpose_f32"));
/// assert!(ptx.contains("param_batch"));
/// ```
#[must_use]
pub fn generate_batch_transpose_ptx(batch: u32, rows: u32, cols: u32) -> String {
    let tile = TRANSPOSE_BLOCK_SIZE;
    let smem_stride = tile + 1;

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Batched Transpose f32: batch={batch}, rows={rows}, cols={cols}, tile={tile}\n\n"
    ));

    // Shared memory
    ptx.push_str(&format!(
        ".shared .align 4 .f32 tile_smem[{smem_total}];\n\n",
        smem_total = tile * smem_stride
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry ptx_batch_transpose_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_rows,\n\
         \x20   .param .u32 param_cols,\n\
         \x20   .param .u32 param_batch\n\
         )\n\
         .reqntid {tile}, {tile}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<24>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<12>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_rows];\n\
         \x20   ld.param.u32  %r1,  [param_cols];\n\
         \x20   ld.param.u32  %r16, [param_batch];\n\n",
    );

    // Thread and batch indices
    ptx.push_str("\x20   mov.u32       %r2, %tid.x;           // tx\n\
         \x20   mov.u32       %r3, %tid.y;           // ty\n\
         \x20   mov.u32       %r17, %ctaid.z;        // batch_idx = blockIdx.z\n\n\
         \x20   // Bounds check: batch_idx < batch\n\
         \x20   setp.ge.u32   %p3, %r17, %r16;\n\
         \x20   @%p3 bra      BTR_EXIT;\n\n");

    // Compute batch offset: batch_idx * rows * cols * 4
    ptx.push_str(
        "\x20   // Batch offset\n\
         \x20   mul.lo.u32    %r18, %r0, %r1;        // rows * cols (matrix_size)\n\
         \x20   mul.lo.u32    %r19, %r17, %r18;      // batch_idx * matrix_size\n\
         \x20   mul.wide.u32  %rd2, %r19, 4;         // byte offset\n\
         \x20   add.u64       %rd3, %rd0, %rd2;      // input base for this batch\n\
         \x20   add.u64       %rd4, %rd1, %rd2;      // output base for this batch\n\n",
    );

    // Global coordinates for input
    ptx.push_str(&format!(
        "\x20   mov.u32       %r4, %ctaid.x;\n\
         \x20   mul.lo.u32    %r5, %r4, {tile};      // block_col_start\n\
         \x20   add.u32       %r6, %r5, %r2;         // col\n\
         \x20   mov.u32       %r7, %ctaid.y;\n\
         \x20   mul.lo.u32    %r8, %r7, {tile};      // block_row_start\n\
         \x20   add.u32       %r9, %r8, %r3;         // row\n\n"
    ));

    // Load to shared memory
    ptx.push_str(&format!(
        "\x20   setp.lt.u32   %p0, %r9, %r0;\n\
         \x20   setp.lt.u32   %p1, %r6, %r1;\n\
         \x20   and.pred      %p2, %p0, %p1;\n\
         \x20   @!%p2 bra     BTR_SKIP_LOAD;\n\
         \x20   mad.lo.u32    %r10, %r9, %r1, %r6;\n\
         \x20   mul.wide.u32  %rd5, %r10, 4;\n\
         \x20   add.u64       %rd6, %rd3, %rd5;\n\
         \x20   ld.global.f32 %f0, [%rd6];\n\
         \x20   mad.lo.u32    %r11, %r3, {smem_stride}, %r2;\n\
         \x20   mul.wide.u32  %rd7, %r11, 4;\n\
         \x20   mov.u64       %rd8, tile_smem;\n\
         \x20   add.u64       %rd9, %rd8, %rd7;\n\
         \x20   st.shared.f32 [%rd9], %f0;\n\
         BTR_SKIP_LOAD:\n\n"
    ));

    // Synchronize
    ptx.push_str("\x20   bar.sync      0;\n\n");

    // Transposed store
    ptx.push_str(&format!(
        "\x20   add.u32       %r12, %r5, %r3;        // out_row = block_col_start + ty\n\
         \x20   add.u32       %r13, %r8, %r2;        // out_col = block_row_start + tx\n\
         \x20   setp.lt.u32   %p0, %r12, %r1;\n\
         \x20   setp.lt.u32   %p1, %r13, %r0;\n\
         \x20   and.pred      %p2, %p0, %p1;\n\
         \x20   @!%p2 bra     BTR_EXIT;\n\
         \x20   mad.lo.u32    %r14, %r2, {smem_stride}, %r3;\n\
         \x20   mul.wide.u32  %rd7, %r14, 4;\n\
         \x20   mov.u64       %rd8, tile_smem;\n\
         \x20   add.u64       %rd9, %rd8, %rd7;\n\
         \x20   ld.shared.f32 %f1, [%rd9];\n\
         \x20   mad.lo.u32    %r15, %r12, %r0, %r13;\n\
         \x20   mul.wide.u32  %rd10, %r15, 4;\n\
         \x20   add.u64       %rd11, %rd4, %rd10;\n\
         \x20   st.global.f32 [%rd11], %f1;\n\n"
    ));

    ptx.push_str(
        "BTR_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Compute grid and block dimensions for a 2D transpose kernel.
///
/// Grid: `(ceil(cols / TILE), ceil(rows / TILE), 1)`.
/// Block: `(TILE, TILE, 1)`.
///
/// # Returns
///
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_transpose_launch_config(rows: u32, cols: u32) -> ([u32; 3], [u32; 3]) {
    let tile = TRANSPOSE_BLOCK_SIZE;
    let grid_x = cols.div_ceil(tile);
    let grid_y = rows.div_ceil(tile);
    ([grid_x, grid_y, 1], [tile, tile, 1])
}

/// Compute grid and block dimensions for a batched transpose kernel.
///
/// Grid: `(ceil(cols / TILE), ceil(rows / TILE), batch)`.
/// Block: `(TILE, TILE, 1)`.
#[must_use]
pub fn ptx_batch_transpose_launch_config(batch: u32, rows: u32, cols: u32) -> ([u32; 3], [u32; 3]) {
    let tile = TRANSPOSE_BLOCK_SIZE;
    let grid_x = cols.div_ceil(tile);
    let grid_y = rows.div_ceil(tile);
    ([grid_x, grid_y, batch], [tile, tile, 1])
}

// ---------------------------------------------------------------------------
// Reference implementations
// ---------------------------------------------------------------------------

/// CPU reference for 2D matrix transpose.
///
/// Transposes a `[rows, cols]` row-major matrix to `[cols, rows]`.
#[must_use]
pub fn transpose_reference(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        data.len(),
        rows * cols,
        "data length {} != rows * cols = {}",
        data.len(),
        rows * cols
    );
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// CPU reference for batched 2D matrix transpose.
///
/// Transposes `batch` matrices, each `[rows, cols]` row-major, to `[cols, rows]`.
#[must_use]
pub fn batch_transpose_reference(data: &[f32], batch: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mat_size = rows * cols;
    assert_eq!(
        data.len(),
        batch * mat_size,
        "data length {} != batch * rows * cols = {}",
        data.len(),
        batch * mat_size
    );
    let mut out = vec![0.0f32; batch * mat_size];
    for b in 0..batch {
        let offset = b * mat_size;
        for r in 0..rows {
            for c in 0..cols {
                out[offset + c * rows + r] = data[offset + r * cols + c];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_transpose_tests.rs"]
mod ptx_transpose_tests;
