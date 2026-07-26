// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for tiled matrix multiplication.
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 matmul
//! with shared memory tiling. Unlike the CUDA C++ emission in [`ptx_emit`],
//! this module emits PTX assembly directly — no `nvcc` compilation step
//! needed. The PTX can be loaded via `cuModuleLoadData` (JIT) or assembled
//! to cubin via `ptxas`.
//!
//! ## Algorithm
//!
//! Standard shared-memory tiled GEMM: C[M,N] = A[M,K] * B[K,N].
//! Each thread block computes a TILE x TILE output tile. For each K-strip,
//! threads cooperatively load a TILE x TILE tile of A and B into shared
//! memory, synchronize, then each thread accumulates its dot product from
//! the shared tiles.
//!
//! ## PTX register usage
//!
//! - `%r0..%r15`: general-purpose 32-bit registers (indices, addresses, temps)
//! - `%f0..%f3`: 32-bit float registers (accumulator, loaded values, products)
//! - `%rd0..%rd7`: 64-bit registers (pointer arithmetic)
//! - `%p0..%p3`: predicate registers (bounds checks, loop conditions)
//!
//! ## Thread block configuration
//!
//! Default tile size: 16x16 (256 threads per block, good occupancy on sm_80).
//! Shared memory: 2 * TILE * TILE * 4 bytes (2 KiB for 16x16 tiles).
//! Grid: `(ceil(N/TILE), ceil(M/TILE))` — x-dim covers columns, y-dim covers rows.
//!
//! Parallel to Metal's `simd_gemm_f32` in `dyn_tensor_metal_matmul_simd_msl.rs`.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude, PtxCodegenError, DEFAULT_SM_TARGET};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for naive (non-tiled) matmul: 16 threads.
///
/// Each thread computes one output element in the naive kernel.
/// The tiled kernel uses `PTX_MATMUL_TILE_SIZE` instead.
pub const MATMUL_BLOCK_SIZE: u32 = 16;

/// Default tile dimension for PTX matmul (16x16 threads per block).
///
/// 16x16 = 256 threads — a good default for sm_80 (Ampere) occupancy.
/// Shared memory per block: 2 * 16 * 16 * 4 = 2,048 bytes.
pub const PTX_MATMUL_TILE_SIZE: usize = 16;

/// Maximum supported tile size for PTX matmul.
///
/// 32x32 = 1,024 threads per block (maximum for most NVIDIA GPUs).
/// Shared memory: 2 * 32 * 32 * 4 = 8,192 bytes.
pub const PTX_MATMUL_MAX_TILE: usize = 32;

/// Minimum supported tile size for PTX matmul.
pub const PTX_MATMUL_MIN_TILE: usize = 4;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for PTX matmul kernel generation.
#[derive(Debug, Clone)]
pub struct PtxMatmulConfig {
    /// Kernel function name in the PTX module.
    pub kernel_name: String,
    /// Tile dimension (both rows and columns). Must be in `[4, 32]`.
    pub tile_size: usize,
    /// SM target for the PTX prelude (e.g., "sm_80").
    pub sm_target: String,
}

impl PtxMatmulConfig {
    /// Create a config with default tile size (16) and sm_80 target.
    pub fn new(kernel_name: &str) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            tile_size: PTX_MATMUL_TILE_SIZE,
            sm_target: DEFAULT_SM_TARGET.to_string(),
        }
    }

    /// Set the tile size (must be in `[4, 32]`).
    #[must_use]
    pub fn with_tile_size(mut self, tile_size: usize) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Set the SM target (e.g., "sm_70", "sm_80", "sm_90").
    #[must_use]
    pub fn with_sm_target(mut self, target: &str) -> Self {
        self.sm_target = target.to_string();
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PtxCodegenError> {
        if self.tile_size < PTX_MATMUL_MIN_TILE || self.tile_size > PTX_MATMUL_MAX_TILE {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "tile_size must be {}..={}, got {}",
                PTX_MATMUL_MIN_TILE, PTX_MATMUL_MAX_TILE, self.tile_size
            )));
        }
        if self.kernel_name.is_empty() {
            return Err(PtxCodegenError::InvalidParameter(
                "kernel_name must not be empty".into(),
            ));
        }
        Ok(())
    }

    /// Shared memory bytes per block: 2 tiles of f32.
    #[must_use]
    pub fn shared_memory_bytes(&self) -> usize {
        2 * self.tile_size * self.tile_size * 4
    }

    /// Threads per block: tile_size * tile_size.
    #[must_use]
    pub fn threads_per_block(&self) -> usize {
        self.tile_size * self.tile_size
    }
}

impl Default for PtxMatmulConfig {
    fn default() -> Self {
        Self::new("ptx_matmul_f32")
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Emit a complete PTX module for tiled f32 matrix multiplication.
///
/// Generates raw PTX assembly (not CUDA C++) implementing:
///   `C[M, N] = A[M, K] * B[K, N]`
///
/// The kernel uses shared memory tiling with configurable tile size.
/// Parameters M, N, K are passed as kernel arguments (runtime-variable).
///
/// # Arguments
///
/// * `config` — Kernel configuration (name, tile size, SM target).
///
/// # Returns
///
/// Complete PTX module string ready for `cuModuleLoadData` or `ptxas`.
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_matmul::{emit_ptx_matmul, PtxMatmulConfig};
/// let config = PtxMatmulConfig::new("gemm_f32").with_tile_size(16);
/// let ptx = emit_ptx_matmul(&config).unwrap();
/// assert!(ptx.contains(".entry gemm_f32"));
/// assert!(ptx.contains(".shared .align 4"));
/// ```
pub fn emit_ptx_matmul(config: &PtxMatmulConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let tile = config.tile_size;
    let name = &config.kernel_name;
    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(8192);

    // -- Module header --
    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// Tiled f32 GEMM: C[M,N] = A[M,K] * B[K,N]\n\
         // Tile size: {tile}x{tile}, shared memory: {} bytes\n\n",
        config.shared_memory_bytes()
    ));

    // -- Shared memory declarations --
    // Two tiles: As[TILE*TILE] for A, Bs[TILE*TILE] for B.
    ptx.push_str(&format!(
        ".shared .align 4 .f32 As[{tile_sq}];\n\
         .shared .align 4 .f32 Bs[{tile_sq}];\n\n",
        tile_sq = tile * tile
    ));

    // -- Kernel entry point --
    // Parameters: A (ptr), B (ptr), C (ptr), M (u32), N (u32), K (u32)
    ptx.push_str(&format!(
        ".visible .entry {name}(\n\
         \x20   .param .u64 param_A,\n\
         \x20   .param .u64 param_B,\n\
         \x20   .param .u64 param_C,\n\
         \x20   .param .u32 param_M,\n\
         \x20   .param .u32 param_N,\n\
         \x20   .param .u32 param_K\n\
         )\n"
    ));

    // Shared memory requirement annotation.
    ptx.push_str(&format!(
        ".reqntid {tile}, {tile}\n\
         {{\n"
    ));

    // -- Register declarations --
    ptx.push_str(
        "\x20   // Register declarations\n\
         \x20   .reg .u32  %r<20>;\n\
         \x20   .reg .f32  %f<8>;\n\
         \x20   .reg .u64  %rd<12>;\n\
         \x20   .reg .pred %p<6>;\n\n",
    );

    // -- Load parameters --
    ptx.push_str(
        "\x20   // Load kernel parameters\n\
         \x20   ld.param.u64  %rd0, [param_A];\n\
         \x20   ld.param.u64  %rd1, [param_B];\n\
         \x20   ld.param.u64  %rd2, [param_C];\n\
         \x20   ld.param.u32  %r0,  [param_M];\n\
         \x20   ld.param.u32  %r1,  [param_N];\n\
         \x20   ld.param.u32  %r2,  [param_K];\n\n",
    );

    // -- Compute thread/block indices --
    // row = blockIdx.y * TILE + threadIdx.y
    // col = blockIdx.x * TILE + threadIdx.x
    // tx  = threadIdx.x
    // ty  = threadIdx.y
    ptx.push_str(&format!(
        "\x20   // Thread and block indices\n\
         \x20   mov.u32       %r3, %tid.x;          // tx = threadIdx.x\n\
         \x20   mov.u32       %r4, %tid.y;          // ty = threadIdx.y\n\
         \x20   mov.u32       %r5, %ctaid.x;        // blockIdx.x\n\
         \x20   mov.u32       %r6, %ctaid.y;        // blockIdx.y\n\
         \x20   mad.lo.u32    %r7, %r6, {tile}, %r4; // row = blockIdx.y * TILE + ty\n\
         \x20   mad.lo.u32    %r8, %r5, {tile}, %r3; // col = blockIdx.x * TILE + tx\n\n"
    ));

    // -- Initialize accumulator --
    ptx.push_str(&format!(
        "\x20   // Initialize accumulator to 0.0\n\
         \x20   mov.f32       %f0, {zero};          // acc = 0.0\n\n"
    ));

    // -- Compute number of K-tiles: num_tiles = (K + TILE - 1) / TILE --
    ptx.push_str(&format!(
        "\x20   // num_tiles = ceil(K / TILE)\n\
         \x20   add.u32       %r9, %r2, {tile_minus_1}; // K + TILE - 1\n\
         \x20   div.u32       %r9, %r9, {tile};         // / TILE\n\n",
        tile_minus_1 = tile - 1
    ));

    // -- K-tile loop --
    ptx.push_str(
        "\x20   // K-tile loop\n\
         \x20   mov.u32       %r10, 0;              // t = 0 (tile index)\n\
         TILE_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r10, %r9;      // t >= num_tiles?\n\
         \x20   @%p0 bra      TILE_DONE;\n\n",
    );

    // -- Load A tile: As[ty][tx] = (row < M && a_col < K) ? A[row*K + a_col] : 0.0 --
    // a_col = t * TILE + tx
    ptx.push_str(&format!(
        "\x20   // Load A tile into shared memory\n\
         \x20   mad.lo.u32    %r11, %r10, {tile}, %r3; // a_col = t * TILE + tx\n\
         \x20   setp.lt.u32   %p1, %r7, %r0;        // row < M?\n\
         \x20   setp.lt.u32   %p2, %r11, %r2;       // a_col < K?\n\
         \x20   and.pred       %p3, %p1, %p2;        // both in bounds?\n\
         \x20   mov.f32       %f1, {zero};           // default = 0.0\n\
         \x20   @!%p3 bra     SKIP_LOAD_A;\n\
         \x20   // A[row * K + a_col]\n\
         \x20   mad.lo.u32    %r12, %r7, %r2, %r11; // row * K + a_col\n\
         \x20   mul.wide.u32  %rd3, %r12, 4;         // byte offset\n\
         \x20   add.u64       %rd4, %rd0, %rd3;      // &A[row*K + a_col]\n\
         \x20   ld.global.f32 %f1, [%rd4];           // load A element\n\
         SKIP_LOAD_A:\n\
         \x20   // As[ty * TILE + tx] = f1\n\
         \x20   mad.lo.u32    %r12, %r4, {tile}, %r3; // ty * TILE + tx\n\
         \x20   mul.wide.u32  %rd3, %r12, 4;          // byte offset into As\n\
         \x20   mov.u64       %rd4, As;               // shared mem base\n\
         \x20   add.u64       %rd5, %rd4, %rd3;\n\
         \x20   st.shared.f32 [%rd5], %f1;\n\n"
    ));

    // -- Load B tile: Bs[ty][tx] = (b_row < K && col < N) ? B[b_row*N + col] : 0.0 --
    // b_row = t * TILE + ty
    ptx.push_str(&format!(
        "\x20   // Load B tile into shared memory\n\
         \x20   mad.lo.u32    %r13, %r10, {tile}, %r4; // b_row = t * TILE + ty\n\
         \x20   setp.lt.u32   %p1, %r13, %r2;       // b_row < K?\n\
         \x20   setp.lt.u32   %p2, %r8, %r1;        // col < N?\n\
         \x20   and.pred       %p3, %p1, %p2;        // both in bounds?\n\
         \x20   mov.f32       %f2, {zero};           // default = 0.0\n\
         \x20   @!%p3 bra     SKIP_LOAD_B;\n\
         \x20   // B[b_row * N + col]\n\
         \x20   mad.lo.u32    %r14, %r13, %r1, %r8; // b_row * N + col\n\
         \x20   mul.wide.u32  %rd3, %r14, 4;         // byte offset\n\
         \x20   add.u64       %rd4, %rd1, %rd3;      // &B[b_row*N + col]\n\
         \x20   ld.global.f32 %f2, [%rd4];           // load B element\n\
         SKIP_LOAD_B:\n\
         \x20   // Bs[ty * TILE + tx] = f2\n\
         \x20   mad.lo.u32    %r14, %r4, {tile}, %r3; // ty * TILE + tx\n\
         \x20   mul.wide.u32  %rd3, %r14, 4;          // byte offset into Bs\n\
         \x20   mov.u64       %rd4, Bs;               // shared mem base\n\
         \x20   add.u64       %rd5, %rd4, %rd3;\n\
         \x20   st.shared.f32 [%rd5], %f2;\n\n"
    ));

    // -- Barrier: wait for all threads to finish loading tiles --
    ptx.push_str(
        "\x20   // Synchronize after tile load\n\
         \x20   bar.sync      0;\n\n",
    );

    // -- Inner dot product loop: for i in 0..TILE: acc += As[ty][i] * Bs[i][tx] --
    ptx.push_str(&format!(
        "\x20   // Inner product loop over shared tile\n\
         \x20   mov.u32       %r15, 0;              // i = 0\n\
         DOT_LOOP:\n\
         \x20   setp.ge.u32   %p4, %r15, {tile};    // i >= TILE?\n\
         \x20   @%p4 bra      DOT_DONE;\n\
         \x20   // Load As[ty * TILE + i]\n\
         \x20   mad.lo.u32    %r16, %r4, {tile}, %r15; // ty * TILE + i\n\
         \x20   mul.wide.u32  %rd6, %r16, 4;\n\
         \x20   mov.u64       %rd7, As;\n\
         \x20   add.u64       %rd8, %rd7, %rd6;\n\
         \x20   ld.shared.f32 %f3, [%rd8];          // As[ty][i]\n\
         \x20   // Load Bs[i * TILE + tx]\n\
         \x20   mad.lo.u32    %r16, %r15, {tile}, %r3; // i * TILE + tx\n\
         \x20   mul.wide.u32  %rd6, %r16, 4;\n\
         \x20   mov.u64       %rd7, Bs;\n\
         \x20   add.u64       %rd8, %rd7, %rd6;\n\
         \x20   ld.shared.f32 %f4, [%rd8];          // Bs[i][tx]\n\
         \x20   // acc += As[ty][i] * Bs[i][tx]\n\
         \x20   fma.rn.f32    %f0, %f3, %f4, %f0;   // acc = a*b + acc\n\
         \x20   add.u32       %r15, %r15, 1;         // i++\n\
         \x20   bra           DOT_LOOP;\n\
         DOT_DONE:\n\n"
    ));

    // -- Barrier after dot product (before next tile load) --
    ptx.push_str(
        "\x20   // Synchronize before next tile iteration\n\
         \x20   bar.sync      0;\n\n",
    );

    // -- Advance tile index and loop back --
    ptx.push_str(
        "\x20   // Next tile\n\
         \x20   add.u32       %r10, %r10, 1;         // t++\n\
         \x20   bra           TILE_LOOP;\n\
         TILE_DONE:\n\n",
    );

    // -- Store result: if (row < M && col < N) C[row*N + col] = acc --
    ptx.push_str(
        "\x20   // Store result to C\n\
         \x20   setp.lt.u32   %p1, %r7, %r0;        // row < M?\n\
         \x20   setp.lt.u32   %p2, %r8, %r1;        // col < N?\n\
         \x20   and.pred       %p3, %p1, %p2;        // both in bounds?\n\
         \x20   @!%p3 bra     KERNEL_EXIT;\n\
         \x20   // C[row * N + col] = acc\n\
         \x20   mad.lo.u32    %r17, %r7, %r1, %r8;  // row * N + col\n\
         \x20   mul.wide.u32  %rd9, %r17, 4;         // byte offset\n\
         \x20   add.u64       %rd10, %rd2, %rd9;     // &C[row*N + col]\n\
         \x20   st.global.f32 [%rd10], %f0;          // store acc\n\
         KERNEL_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    Ok(ptx)
}

/// Compute the grid and block dimensions for a PTX matmul kernel.
///
/// Grid: `(ceil(N/tile), ceil(M/tile), 1)`.
/// Block: `(tile, tile, 1)`.
///
/// # Returns
///
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_matmul_launch_config(m: usize, n: usize, tile_size: usize) -> ([usize; 3], [usize; 3]) {
    let grid = [n.div_ceil(tile_size), m.div_ceil(tile_size), 1];
    let block = [tile_size, tile_size, 1];
    (grid, block)
}

/// Convenience: emit PTX matmul with default 16x16 tiles and sm_80.
///
/// Equivalent to `emit_ptx_matmul(&PtxMatmulConfig::new(name))`.
pub fn emit_ptx_matmul_default(name: &str) -> Result<String, PtxCodegenError> {
    emit_ptx_matmul(&PtxMatmulConfig::new(name))
}

// ---------------------------------------------------------------------------
// Naive (non-tiled) matmul
// ---------------------------------------------------------------------------

/// Generate naive SGEMM PTX: C[M,N] = A[M,K] * B[K,N].
///
/// Each thread computes one element of C by iterating over the entire K
/// dimension. No shared memory tiling. Suitable for small matrices or as
/// a correctness baseline for the tiled implementation.
///
/// Thread block: `(block, block, 1)` where `block` = [`MATMUL_BLOCK_SIZE`].
/// Grid: `(ceil(N/block), ceil(M/block), 1)`.
///
/// # Arguments
///
/// * `m` — number of rows in A / C (for PTX comments only; runtime param)
/// * `k` — shared dimension (A columns, B rows)
/// * `n` — number of columns in B / C
///
/// All dimensions are passed as runtime parameters in the kernel.
pub fn generate_matmul_ptx(m: u32, k: u32, n: u32) -> String {
    let block = MATMUL_BLOCK_SIZE;
    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&ptx_prelude("sm_70"));
    ptx.push_str(&format!(
        "// Naive f32 GEMM: C[{m},{n}] = A[{m},{k}] * B[{k},{n}]\n\
         // Block: {block}x{block}, no shared memory tiling\n\n"
    ));

    // Kernel entry
    ptx.push_str(
        ".visible .entry naive_matmul_f32(\n\
         \x20   .param .u64 param_A,\n\
         \x20   .param .u64 param_B,\n\
         \x20   .param .u64 param_C,\n\
         \x20   .param .u32 param_M,\n\
         \x20   .param .u32 param_N,\n\
         \x20   .param .u32 param_K\n\
         )\n",
    );

    ptx.push_str(&format!(".reqntid {block}, {block}\n{{\n"));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<16>;\n\
         \x20   .reg .f32  %f<6>;\n\
         \x20   .reg .u64  %rd<10>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_A];\n\
         \x20   ld.param.u64  %rd1, [param_B];\n\
         \x20   ld.param.u64  %rd2, [param_C];\n\
         \x20   ld.param.u32  %r0,  [param_M];\n\
         \x20   ld.param.u32  %r1,  [param_N];\n\
         \x20   ld.param.u32  %r2,  [param_K];\n\n",
    );

    // Compute row = blockIdx.y * blockDim.y + threadIdx.y
    //         col = blockIdx.x * blockDim.x + threadIdx.x
    ptx.push_str(&format!(
        "\x20   mov.u32       %r3, %tid.x;\n\
         \x20   mov.u32       %r4, %tid.y;\n\
         \x20   mov.u32       %r5, %ctaid.x;\n\
         \x20   mov.u32       %r6, %ctaid.y;\n\
         \x20   mad.lo.u32    %r7, %r6, {block}, %r4;  // row\n\
         \x20   mad.lo.u32    %r8, %r5, {block}, %r3;  // col\n\n"
    ));

    // Bounds check: row < M && col < N
    ptx.push_str(
        "\x20   setp.ge.u32   %p0, %r7, %r0;\n\
         \x20   setp.ge.u32   %p1, %r8, %r1;\n\
         \x20   or.pred        %p2, %p0, %p1;\n\
         \x20   @%p2 bra      NAIVE_EXIT;\n\n",
    );

    // acc = 0.0
    ptx.push_str(&format!(
        "\x20   mov.f32       %f0, {zero};  // acc = 0.0\n\
         \x20   mov.u32       %r9, 0;       // i = 0\n\n"
    ));

    // Loop: for i in 0..K: acc += A[row*K+i] * B[i*N+col]
    ptx.push_str(
        "NAIVE_LOOP:\n\
         \x20   setp.ge.u32   %p3, %r9, %r2;       // i >= K?\n\
         \x20   @%p3 bra      NAIVE_STORE;\n\
         \x20   // Load A[row*K + i]\n\
         \x20   mad.lo.u32    %r10, %r7, %r2, %r9;  // row*K + i\n\
         \x20   mul.wide.u32  %rd3, %r10, 4;\n\
         \x20   add.u64       %rd4, %rd0, %rd3;\n\
         \x20   ld.global.f32 %f1, [%rd4];\n\
         \x20   // Load B[i*N + col]\n\
         \x20   mad.lo.u32    %r11, %r9, %r1, %r8;  // i*N + col\n\
         \x20   mul.wide.u32  %rd5, %r11, 4;\n\
         \x20   add.u64       %rd6, %rd1, %rd5;\n\
         \x20   ld.global.f32 %f2, [%rd6];\n\
         \x20   // acc += a * b\n\
         \x20   fma.rn.f32    %f0, %f1, %f2, %f0;\n\
         \x20   add.u32       %r9, %r9, 1;\n\
         \x20   bra           NAIVE_LOOP;\n\n",
    );

    // Store C[row*N + col] = acc
    ptx.push_str(
        "NAIVE_STORE:\n\
         \x20   mad.lo.u32    %r12, %r7, %r1, %r8;  // row*N + col\n\
         \x20   mul.wide.u32  %rd7, %r12, 4;\n\
         \x20   add.u64       %rd8, %rd2, %rd7;\n\
         \x20   st.global.f32 [%rd8], %f0;\n\
         NAIVE_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

/// Generate tiled SGEMM PTX: C[M,N] = A[M,K] * B[K,N].
///
/// Wrapper around [`emit_ptx_matmul`] that accepts dimension parameters for
/// PTX comments and a tile size. The kernel itself accepts M, N, K at runtime.
///
/// # Arguments
///
/// * `m`, `k`, `n` — matrix dimensions (for documentation/comments in PTX)
/// * `tile` — tile size for shared memory blocking
pub fn generate_matmul_tiled_ptx(m: u32, k: u32, n: u32, tile: u32) -> String {
    let config = PtxMatmulConfig::new("tiled_matmul_f32")
        .with_tile_size(tile as usize)
        .with_sm_target("sm_70");

    let mut ptx = emit_ptx_matmul(&config)
        .expect("tiled matmul PTX generation should not fail for valid tile sizes");

    // Prepend a comment with the concrete dimensions
    let header = format!("// Generated tiled SGEMM for M={m}, K={k}, N={n}, tile={tile}\n");
    ptx.insert_str(0, &header);
    ptx
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

/// Compute C = A * B on CPU for reference/testing.
///
/// `A` is `[m, k]` row-major, `B` is `[k, n]` row-major, returns `C` `[m, n]`.
///
/// # Panics
///
/// Panics if `a.len() != m * k` or `b.len() != k * n`.
pub fn matmul_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(
        a.len(),
        m * k,
        "A must have m*k={} elements, got {}",
        m * k,
        a.len()
    );
    assert_eq!(
        b.len(),
        k * n,
        "B must have k*n={} elements, got {}",
        k * n,
        b.len()
    );

    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_matmul_tests.rs"]
mod ptx_matmul_tests;
