// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for softmax with warp-level reduction.
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 softmax
//! along the last dimension. Unlike the CUDA C++ emission in [`ptx_emit`],
//! this module emits PTX assembly directly — no `nvcc` compilation step
//! needed. The PTX can be loaded via `cuModuleLoadData` (JIT) or assembled
//! to cubin via `ptxas`.
//!
//! ## Algorithm
//!
//! Numerically stable softmax: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))`.
//!
//! Four phases:
//! 1. **Warp-level max reduction** — each thread reads a strided slice of the
//!    row, computes a local max, then reduces across the warp with `shfl.down.sync`.
//!    For dim > 32, cross-warp reduction uses shared memory.
//! 2. **Subtract max and exp** — each thread computes `exp(x_i - max)` using
//!    `ex2.approx.f32` (base-2 exponential) with a log2(e) prescale for speed.
//! 3. **Warp-level sum reduction** — same structure as phase 1 but for the
//!    exponential sum.
//! 4. **Divide by sum** — each thread normalizes its elements by the row sum.
//!
//! ## Warp-level vs shared memory reduction
//!
//! - **dim <= 32 (single warp):** Pure warp shuffle reduction (`shfl.down.sync`).
//!   No shared memory needed. One warp per row.
//! - **dim > 32 (multi-warp):** Each warp reduces locally via shuffles, then
//!   warp leaders write to shared memory. A final cross-warp reduction in
//!   shared memory produces the block-wide result.
//!
//! ## PTX register usage
//!
//! - `%r0..%r15`: general-purpose 32-bit registers (indices, temps)
//! - `%f0..%f7`: 32-bit float registers (accumulator, max, sum, exp vals)
//! - `%rd0..%rd7`: 64-bit registers (pointer arithmetic)
//! - `%p0..%p3`: predicate registers (bounds checks)
//!
//! ## Thread block configuration
//!
//! Block: `(block_size, 1, 1)` where `block_size = min(dim_rounded_up_to_warp, 256)`.
//! Grid: `(num_rows, 1, 1)` — one block per row.
//!
//! Parallel to Metal softmax in `dyn_tensor_metal_ops.rs`.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude, PtxCodegenError, DEFAULT_SM_TARGET};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// NVIDIA warp size (32 threads).
const WARP_SIZE: usize = 32;

/// Maximum block size for softmax (8 warps = 256 threads).
const MAX_BLOCK_SIZE: usize = 256;

/// Public softmax block size constant (256 threads = 8 warps).
///
/// Matches the maximum block size used by the PTX softmax kernel.
/// Useful for external launch configuration calculations.
pub const SOFTMAX_BLOCK_SIZE: u32 = 256;

/// log2(e) as f32 — prescale factor for `ex2.approx.f32`.
const LOG2_E: f32 = std::f32::consts::LOG2_E;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for PTX softmax kernel generation.
#[derive(Debug, Clone)]
pub struct PtxSoftmaxConfig {
    /// Kernel function name in the PTX module.
    pub kernel_name: String,
    /// Dimension of the softmax axis (last dimension size).
    pub dim: usize,
    /// SM target for the PTX prelude (e.g., "sm_80").
    pub sm_target: String,
    /// If true, generate log_softmax instead of softmax.
    ///
    /// log_softmax(x)_i = (x_i - max) - log(sum(exp(x - max)))
    /// More numerically stable than `log(softmax(x))` and avoids
    /// a redundant exp+log pair.
    pub log_mode: bool,
}

impl PtxSoftmaxConfig {
    /// Create a config with default sm_80 target (standard softmax).
    pub fn new(kernel_name: &str, dim: usize) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            dim,
            sm_target: DEFAULT_SM_TARGET.to_string(),
            log_mode: false,
        }
    }

    /// Create a log_softmax config with default sm_80 target.
    pub fn new_log(kernel_name: &str, dim: usize) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            dim,
            sm_target: DEFAULT_SM_TARGET.to_string(),
            log_mode: true,
        }
    }

    /// Set log_softmax mode.
    #[must_use]
    pub fn with_log_mode(mut self, log_mode: bool) -> Self {
        self.log_mode = log_mode;
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
        if self.dim == 0 {
            return Err(PtxCodegenError::InvalidParameter("dim must be > 0".into()));
        }
        if self.kernel_name.is_empty() {
            return Err(PtxCodegenError::InvalidParameter(
                "kernel_name must not be empty".into(),
            ));
        }
        Ok(())
    }

    /// Block size: number of threads per block.
    ///
    /// Round dim up to next multiple of WARP_SIZE, cap at MAX_BLOCK_SIZE.
    #[must_use]
    pub fn block_size(&self) -> usize {
        let rounded = self.dim.div_ceil(WARP_SIZE) * WARP_SIZE;
        rounded.min(MAX_BLOCK_SIZE)
    }

    /// Number of warps in the block.
    #[must_use]
    pub fn num_warps(&self) -> usize {
        self.block_size() / WARP_SIZE
    }

    /// Whether this config uses warp-only reduction (no shared memory).
    #[must_use]
    pub fn is_warp_only(&self) -> bool {
        self.num_warps() <= 1
    }

    /// Shared memory bytes needed (0 for warp-only, 4 * num_warps otherwise).
    #[must_use]
    pub fn shared_memory_bytes(&self) -> usize {
        if self.is_warp_only() {
            0
        } else {
            // One f32 per warp for cross-warp reduction.
            self.num_warps() * 4
        }
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Emit a complete PTX module for f32 softmax along the last dimension.
///
/// Generates raw PTX assembly implementing numerically stable softmax with
/// warp-level shuffle reduction. For dim <= 32, uses pure warp shuffles
/// (no shared memory). For dim > 32, uses shared memory for cross-warp
/// reduction.
///
/// # Arguments
///
/// * `config` — Kernel configuration (name, dim, SM target).
///
/// # Returns
///
/// Complete PTX module string ready for `cuModuleLoadData` or `ptxas`.
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_softmax::{emit_ptx_softmax, PtxSoftmaxConfig};
/// let config = PtxSoftmaxConfig::new("softmax_128", 128);
/// let ptx = emit_ptx_softmax(&config).unwrap();
/// assert!(ptx.contains(".entry softmax_128"));
/// assert!(ptx.contains("shfl.down.sync"));
/// ```
pub fn emit_ptx_softmax(config: &PtxSoftmaxConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let dim = config.dim;
    let name = &config.kernel_name;
    let block_size = config.block_size();
    let num_warps = config.num_warps();
    let warp_only = config.is_warp_only();

    let neg_inf = format_ptx_float(f32::NEG_INFINITY);
    let zero = format_ptx_float(0.0);
    let one = format_ptx_float(1.0);
    let log2e = format_ptx_float(LOG2_E);

    let log_mode = config.log_mode;
    let mode_label = if log_mode { "LogSoftmax" } else { "Softmax" };

    let mut ptx = String::with_capacity(8192);

    // -- Module header --
    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// {mode_label} f32: dim={dim}, block_size={block_size}, warps={num_warps}\n\
         // Reduction: {}\n\n",
        if warp_only {
            "warp-only (no shared memory)"
        } else {
            "warp shuffle + shared memory cross-warp"
        }
    ));

    // -- Shared memory for cross-warp reduction (only if multi-warp) --
    if !warp_only {
        ptx.push_str(&format!(
            ".shared .align 4 .f32 warp_scratch[{num_warps}];\n\n"
        ));
    }

    // -- Kernel entry point --
    // Parameters: input (ptr), output (ptr), row_size (u32), num_rows (u32)
    ptx.push_str(&format!(
        ".visible .entry {name}(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_row_size,\n\
         \x20   .param .u32 param_num_rows\n\
         )\n"
    ));

    ptx.push_str(&format!(".reqntid {block_size}\n{{\n"));

    // -- Register declarations --
    ptx.push_str(
        "\x20   // Register declarations\n\
         \x20   .reg .u32  %r<20>;\n\
         \x20   .reg .f32  %f<12>;\n\
         \x20   .reg .u64  %rd<10>;\n\
         \x20   .reg .pred %p<6>;\n\n",
    );

    // -- Load parameters --
    ptx.push_str(
        "\x20   // Load kernel parameters\n\
         \x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_row_size];\n\
         \x20   ld.param.u32  %r1,  [param_num_rows];\n\n",
    );

    // -- Compute thread/block indices --
    // row = blockIdx.x (one block per row)
    // tid = threadIdx.x
    ptx.push_str(
        "\x20   // Thread and block indices\n\
         \x20   mov.u32       %r2, %tid.x;           // tid = threadIdx.x\n\
         \x20   mov.u32       %r3, %ctaid.x;         // row = blockIdx.x\n\n",
    );

    // -- Bounds check: row < num_rows --
    ptx.push_str(
        "\x20   // Bounds check: row < num_rows\n\
         \x20   setp.ge.u32   %p0, %r3, %r1;         // row >= num_rows?\n\
         \x20   @%p0 bra      KERNEL_EXIT;\n\n",
    );

    // -- Compute row pointers --
    // row_in  = input  + row * row_size
    // row_out = output + row * row_size
    ptx.push_str(
        "\x20   // Compute row pointers\n\
         \x20   mul.lo.u32    %r4, %r3, %r0;         // row * row_size\n\
         \x20   mul.wide.u32  %rd2, %r4, 4;          // byte offset\n\
         \x20   add.u64       %rd3, %rd0, %rd2;      // &input[row * row_size]\n\
         \x20   add.u64       %rd4, %rd1, %rd2;      // &output[row * row_size]\n\n",
    );

    // -- Compute warp ID and lane ID --
    ptx.push_str("\x20   // Warp/lane decomposition\n\
         \x20   shr.u32       %r5, %r2, 5;           // warp_id = tid >> 5\n\
         \x20   and.b32       %r6, %r2, 31;          // lane_id = tid & 31\n\n");

    // =====================================================================
    // Phase 1: find max across the row
    // =====================================================================
    ptx.push_str(&format!(
        "\x20   // ---- Phase 1: find row max ----\n\
         \x20   mov.f32       %f0, {neg_inf};        // local_max = -inf\n\
         \x20   mov.u32       %r7, %r2;              // i = tid\n\
         PHASE1_LOOP:\n\
         \x20   setp.ge.u32   %p1, %r7, %r0;         // i >= row_size?\n\
         \x20   @%p1 bra      PHASE1_REDUCE;\n\
         \x20   // Load input[i]\n\
         \x20   mul.wide.u32  %rd5, %r7, 4;          // byte offset\n\
         \x20   add.u64       %rd6, %rd3, %rd5;      // &row_in[i]\n\
         \x20   ld.global.f32 %f1, [%rd6];           // val = row_in[i]\n\
         \x20   max.f32       %f0, %f0, %f1;         // local_max = max(local_max, val)\n\
         \x20   add.u32       %r7, %r7, {block_size}; // i += block_size\n\
         \x20   bra           PHASE1_LOOP;\n\
         PHASE1_REDUCE:\n\n"
    ));

    // Warp-level max reduction via shfl.down.sync
    emit_warp_reduce_max(&mut ptx, "%f0");

    // Cross-warp reduction if needed
    if !warp_only {
        emit_cross_warp_reduce_max(&mut ptx, num_warps);
    }

    // %f0 now holds the global row max for all threads
    ptx.push_str("\x20   // %f0 = row max (broadcast to all threads)\n\n");

    // =====================================================================
    // Phase 2: compute exp(x_i - max) and write to output
    // Phase 3: warp-level sum reduction (combined with phase 2 accumulation)
    // =====================================================================
    ptx.push_str(&format!(
        "\x20   // ---- Phase 2: exp(x - max) + Phase 3: sum ----\n\
         \x20   mov.f32       %f2, {zero};            // local_sum = 0.0\n\
         \x20   mov.u32       %r7, %r2;               // i = tid\n\
         PHASE2_LOOP:\n\
         \x20   setp.ge.u32   %p1, %r7, %r0;          // i >= row_size?\n\
         \x20   @%p1 bra      PHASE3_REDUCE;\n\
         \x20   // Load input[i] and compute exp(x - max)\n\
         \x20   mul.wide.u32  %rd5, %r7, 4;           // byte offset\n\
         \x20   add.u64       %rd6, %rd3, %rd5;       // &row_in[i]\n\
         \x20   ld.global.f32 %f3, [%rd6];            // val = row_in[i]\n\
         \x20   sub.f32       %f4, %f3, %f0;          // diff = val - max\n\
         \x20   mul.f32       %f5, %f4, {log2e};      // diff * log2(e)\n\
         \x20   ex2.approx.f32 %f6, %f5;              // exp(diff) = 2^(diff*log2e)\n\
         \x20   add.f32       %f2, %f2, %f6;          // local_sum += exp_val\n\
         \x20   // Store exp(x - max) to output[i]\n\
         \x20   add.u64       %rd7, %rd4, %rd5;       // &row_out[i]\n\
         \x20   st.global.f32 [%rd7], %f6;            // output[i] = exp_val\n\
         \x20   add.u32       %r7, %r7, {block_size}; // i += block_size\n\
         \x20   bra           PHASE2_LOOP;\n\
         PHASE3_REDUCE:\n\n"
    ));

    // Warp-level sum reduction via shfl.down.sync
    emit_warp_reduce_sum(&mut ptx, "%f2");

    // Cross-warp sum reduction if needed
    if !warp_only {
        emit_cross_warp_reduce_sum(&mut ptx, num_warps);
    }

    // %f2 now holds the global row sum for all threads

    // =====================================================================
    // Phase 4: normalize (softmax) or compute log (log_softmax)
    // =====================================================================
    if log_mode {
        // log_softmax: output[i] = (x_i - max) - log(sum)
        // log(sum) = lg2(sum) / lg2(e)  where lg2 is the PTX base-2 log
        let rcp_log2e = format_ptx_float(1.0 / LOG2_E); // 1/log2(e) = ln(2)
        ptx.push_str(&format!(
            "\x20   // ---- Phase 4: log_softmax normalize ----\n\
             \x20   // log_sum = lg2(sum) * (1/log2(e)) = lg2(sum) * ln(2)\n\
             \x20   lg2.approx.f32 %f7, %f2;              // lg2(sum)\n\
             \x20   mul.f32       %f7, %f7, {rcp_log2e};  // log_sum = lg2(sum) / log2(e)\n\
             \x20   mov.u32       %r7, %r2;               // i = tid\n\
             PHASE4_LOOP:\n\
             \x20   setp.ge.u32   %p1, %r7, %r0;          // i >= row_size?\n\
             \x20   @%p1 bra      KERNEL_EXIT;\n\
             \x20   // Load input[i], compute (x_i - max) - log_sum\n\
             \x20   mul.wide.u32  %rd5, %r7, 4;           // byte offset\n\
             \x20   add.u64       %rd6, %rd3, %rd5;       // &row_in[i]\n\
             \x20   ld.global.f32 %f8, [%rd6];            // x_i\n\
             \x20   sub.f32       %f9, %f8, %f0;          // x_i - max\n\
             \x20   sub.f32       %f9, %f9, %f7;          // (x_i - max) - log_sum\n\
             \x20   add.u64       %rd7, %rd4, %rd5;       // &row_out[i]\n\
             \x20   st.global.f32 [%rd7], %f9;            // output[i] = log_softmax\n\
             \x20   add.u32       %r7, %r7, {block_size}; // i += block_size\n\
             \x20   bra           PHASE4_LOOP;\n\n"
        ));
    } else {
        // Standard softmax: output[i] = exp(x_i - max) / sum
        ptx.push_str(&format!(
            "\x20   // ---- Phase 4: softmax normalize ----\n\
             \x20   div.approx.f32 %f7, {one}, %f2;      // inv_sum = 1.0 / sum\n\
             \x20   mov.u32       %r7, %r2;               // i = tid\n\
             PHASE4_LOOP:\n\
             \x20   setp.ge.u32   %p1, %r7, %r0;          // i >= row_size?\n\
             \x20   @%p1 bra      KERNEL_EXIT;\n\
             \x20   // Load output[i], multiply by inv_sum, store back\n\
             \x20   mul.wide.u32  %rd5, %r7, 4;           // byte offset\n\
             \x20   add.u64       %rd7, %rd4, %rd5;       // &row_out[i]\n\
             \x20   ld.global.f32 %f8, [%rd7];            // exp_val\n\
             \x20   mul.f32       %f9, %f8, %f7;          // exp_val * inv_sum\n\
             \x20   st.global.f32 [%rd7], %f9;            // output[i] = normalized\n\
             \x20   add.u32       %r7, %r7, {block_size}; // i += block_size\n\
             \x20   bra           PHASE4_LOOP;\n\n"
        ));
    }

    // -- Kernel exit --
    ptx.push_str(
        "KERNEL_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    Ok(ptx)
}

/// Emit warp-level max reduction on register `reg` using `shfl.down.sync`.
///
/// After this sequence, lane 0 of each warp holds the warp-wide max.
/// The result is then broadcast to all lanes via `shfl.idx.sync`.
fn emit_warp_reduce_max(ptx: &mut String, reg: &str) {
    ptx.push_str("\x20   // Warp-level max reduction (shfl.down.sync)\n");
    // Reduce within warp: offsets 16, 8, 4, 2, 1
    for offset in [16, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "\x20   shfl.down.sync.b32 %f10, {reg}, {offset}, 31, 0xFFFFFFFF;\n\
             \x20   max.f32       {reg}, {reg}, %f10;\n"
        ));
    }
    // Broadcast lane 0's result to all lanes in the warp
    ptx.push_str(&format!(
        "\x20   shfl.idx.sync.b32 {reg}, {reg}, 0, 31, 0xFFFFFFFF;\n\n"
    ));
}

/// Emit warp-level sum reduction on register `reg` using `shfl.down.sync`.
fn emit_warp_reduce_sum(ptx: &mut String, reg: &str) {
    ptx.push_str("\x20   // Warp-level sum reduction (shfl.down.sync)\n");
    for offset in [16, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "\x20   shfl.down.sync.b32 %f10, {reg}, {offset}, 31, 0xFFFFFFFF;\n\
             \x20   add.f32       {reg}, {reg}, %f10;\n"
        ));
    }
    ptx.push_str(&format!(
        "\x20   shfl.idx.sync.b32 {reg}, {reg}, 0, 31, 0xFFFFFFFF;\n\n"
    ));
}

/// Emit cross-warp max reduction using shared memory `warp_scratch`.
///
/// Each warp leader (lane 0) writes to shared memory, then a barrier
/// synchronizes all warps. Warp 0 reduces the scratch buffer, then
/// the result is broadcast via shared memory.
fn emit_cross_warp_reduce_max(ptx: &mut String, num_warps: usize) {
    ptx.push_str(&format!(
        "\x20   // Cross-warp max reduction via shared memory\n\
         \x20   // Warp leader (lane 0) writes local max to warp_scratch\n\
         \x20   setp.eq.u32   %p2, %r6, 0;            // lane_id == 0?\n\
         \x20   @!%p2 bra     CROSS_MAX_LOAD;\n\
         \x20   mul.wide.u32  %rd8, %r5, 4;           // warp_id * 4\n\
         \x20   mov.u64       %rd9, warp_scratch;\n\
         \x20   add.u64       %rd8, %rd9, %rd8;\n\
         \x20   st.shared.f32 [%rd8], %f0;\n\
         CROSS_MAX_LOAD:\n\
         \x20   bar.sync      0;\n\
         \x20   // All threads read warp 0's scratch and reduce\n\
         \x20   mov.f32       %f0, {neg_inf};\n",
        neg_inf = format_ptx_float(f32::NEG_INFINITY),
    ));
    // Each thread in warp 0 loads one scratch element (if lane_id < num_warps)
    ptx.push_str(&format!(
        "\x20   setp.lt.u32   %p3, %r2, {num_warps};  // tid < num_warps?\n\
         \x20   @!%p3 bra     CROSS_MAX_DONE;\n\
         \x20   mul.wide.u32  %rd8, %r2, 4;           // tid * 4\n\
         \x20   mov.u64       %rd9, warp_scratch;\n\
         \x20   add.u64       %rd8, %rd9, %rd8;\n\
         \x20   ld.shared.f32 %f0, [%rd8];\n\
         CROSS_MAX_DONE:\n"
    ));
    // Warp reduce the loaded values
    emit_warp_reduce_max(ptx, "%f0");
    // Broadcast result to all threads via shared memory
    ptx.push_str(
        "\x20   // Broadcast max to all threads via shared memory\n\
         \x20   setp.eq.u32   %p2, %r2, 0;            // tid == 0?\n\
         \x20   @!%p2 bra     BCAST_MAX_LOAD;\n\
         \x20   mov.u64       %rd9, warp_scratch;\n\
         \x20   st.shared.f32 [%rd9], %f0;\n\
         BCAST_MAX_LOAD:\n\
         \x20   bar.sync      0;\n\
         \x20   mov.u64       %rd9, warp_scratch;\n\
         \x20   ld.shared.f32 %f0, [%rd9];\n\n",
    );
}

/// Emit cross-warp sum reduction using shared memory `warp_scratch`.
fn emit_cross_warp_reduce_sum(ptx: &mut String, num_warps: usize) {
    ptx.push_str(&format!(
        "\x20   // Cross-warp sum reduction via shared memory\n\
         \x20   setp.eq.u32   %p2, %r6, 0;            // lane_id == 0?\n\
         \x20   @!%p2 bra     CROSS_SUM_LOAD;\n\
         \x20   mul.wide.u32  %rd8, %r5, 4;           // warp_id * 4\n\
         \x20   mov.u64       %rd9, warp_scratch;\n\
         \x20   add.u64       %rd8, %rd9, %rd8;\n\
         \x20   st.shared.f32 [%rd8], %f2;\n\
         CROSS_SUM_LOAD:\n\
         \x20   bar.sync      0;\n\
         \x20   mov.f32       %f2, {zero};\n",
        zero = format_ptx_float(0.0),
    ));
    ptx.push_str(&format!(
        "\x20   setp.lt.u32   %p3, %r2, {num_warps};  // tid < num_warps?\n\
         \x20   @!%p3 bra     CROSS_SUM_DONE;\n\
         \x20   mul.wide.u32  %rd8, %r2, 4;           // tid * 4\n\
         \x20   mov.u64       %rd9, warp_scratch;\n\
         \x20   add.u64       %rd8, %rd9, %rd8;\n\
         \x20   ld.shared.f32 %f2, [%rd8];\n\
         CROSS_SUM_DONE:\n"
    ));
    emit_warp_reduce_sum(ptx, "%f2");
    // Broadcast result to all threads
    ptx.push_str(
        "\x20   // Broadcast sum to all threads via shared memory\n\
         \x20   setp.eq.u32   %p2, %r2, 0;            // tid == 0?\n\
         \x20   @!%p2 bra     BCAST_SUM_LOAD;\n\
         \x20   mov.u64       %rd9, warp_scratch;\n\
         \x20   st.shared.f32 [%rd9], %f2;\n\
         BCAST_SUM_LOAD:\n\
         \x20   bar.sync      0;\n\
         \x20   mov.u64       %rd9, warp_scratch;\n\
         \x20   ld.shared.f32 %f2, [%rd9];\n\n",
    );
}

/// Compute softmax on CPU for reference/testing.
///
/// Numerically stable: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))`.
///
/// Operates on a single row.
pub fn softmax_reference(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Compute log-softmax on CPU for reference/testing.
///
/// Numerically stable: `log_softmax(x)_i = (x_i - max(x)) - log(sum(exp(x - max(x))))`.
///
/// Operates on a single row.
pub fn log_softmax_reference(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let log_sum = sum.ln();
    input.iter().map(|&x| (x - max_val) - log_sum).collect()
}

/// Convenience: emit PTX softmax with default sm_80 target.
pub fn emit_ptx_softmax_default(name: &str, dim: usize) -> Result<String, PtxCodegenError> {
    emit_ptx_softmax(&PtxSoftmaxConfig::new(name, dim))
}

/// Compute the grid and block dimensions for a PTX softmax kernel.
///
/// Grid: `(num_rows, 1, 1)` — one block per row.
/// Block: `(block_size, 1, 1)` — threads cooperate on one row.
///
/// # Returns
///
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_softmax_launch_config(num_rows: usize, dim: usize) -> ([usize; 3], [usize; 3]) {
    let config = PtxSoftmaxConfig::new("_", dim);
    let block_size = config.block_size();
    let grid = [num_rows, 1, 1];
    let block = [block_size, 1, 1];
    (grid, block)
}

// ---------------------------------------------------------------------------
// Convenience wrapper matching the task spec
// ---------------------------------------------------------------------------

/// Generate PTX for softmax or log_softmax along the last dimension.
///
/// This is a convenience wrapper around [`emit_ptx_softmax`] that uses
/// default settings (sm_80 target). When `log_mode` is true, generates
/// log_softmax instead of softmax.
///
/// # Arguments
///
/// * `log_mode` — If true, generate log_softmax; if false, generate softmax.
/// * `dim` — Size of the last dimension (the softmax axis).
pub fn generate_softmax_ptx(log_mode: bool, dim: usize) -> String {
    let name = if log_mode {
        "ptx_log_softmax_f32"
    } else {
        "ptx_softmax_f32"
    };
    let config = PtxSoftmaxConfig::new(name, dim).with_log_mode(log_mode);
    emit_ptx_softmax(&config).expect("softmax PTX generation failed")
}

/// Generate PTX for log-softmax along the last dimension.
///
/// Convenience wrapper around [`generate_softmax_ptx`] with `log_mode = true`.
/// Equivalent to `generate_softmax_ptx(true, dim as usize)` but with a
/// simpler signature matching the task spec.
///
/// # Arguments
///
/// * `n` — Size of the last dimension (the softmax axis).
pub fn generate_log_softmax_ptx(n: u32) -> String {
    generate_softmax_ptx(true, n as usize)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_softmax_tests.rs"]
mod ptx_softmax_tests;
