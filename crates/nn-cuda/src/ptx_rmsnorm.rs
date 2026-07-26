// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for RMS (Root Mean Square) layer normalization.
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 RMSNorm
//! along the last dimension. RMSNorm is the normalization layer used in
//! modern LLMs (Llama, Qwen3, GLM) and is simpler than full LayerNorm:
//! no mean subtraction, no beta (bias) parameter.
//!
//! ## Algorithm
//!
//! RMSNorm: `y_i = weight_i * x_i * rsqrt(mean(x^2) + eps)`
//!
//! Two phases:
//! 1. **Compute mean(x^2)** — each thread reads a strided slice of the
//!    normalization dimension, accumulates `x_i^2`, then reduces across the
//!    warp with `shfl.down.sync`. For dim > 32, cross-warp reduction uses
//!    shared memory. The sum is divided by `dim` to produce mean(x^2).
//! 2. **Normalize + scale** — each thread computes
//!    `y_i = weight_i * x_i * rsqrt(mean(x^2) + eps)`.
//!
//! ## Comparison with LayerNorm
//!
//! | Property         | LayerNorm               | RMSNorm                    |
//! |------------------|-------------------------|----------------------------|
//! | Formula          | gamma*(x-mean)/std+beta | weight * x * rsqrt(rms+e)  |
//! | Phases           | 4 (mean, var, norm, affine) | 2 (rms, norm+scale)    |
//! | Parameters       | gamma + beta            | weight only                |
//! | Mean subtraction | Yes                     | No                         |
//! | Used in          | BERT, GPT-2             | Llama, Qwen3, GLM, Gemma   |
//!
//! ## Warp-level vs shared memory reduction
//!
//! Same strategy as [`ptx_layernorm`]:
//! - **dim <= 32 (single warp):** Pure warp shuffle reduction.
//! - **dim > 32 (multi-warp):** Warp shuffles + shared memory cross-warp.
//!
//! ## Kernel interface
//!
//! Parameters (in generated kernel):
//! - `param_input`    — pointer to input tensor (f32)
//! - `param_output`   — pointer to output tensor (f32)
//! - `param_weight`   — pointer to weight tensor (f32, length = hidden_dim)
//! - `param_row_size` — u32, number of elements in the normalization dimension
//! - `param_num_rows` — u32, number of rows (outer dimensions product)
//!
//! ## Thread block configuration
//!
//! Block: `(block_size, 1, 1)` where `block_size = min(dim_rounded_up_to_warp, 256)`.
//! Grid: `(num_rows, 1, 1)` — one block per row.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude, PtxCodegenError, DEFAULT_SM_TARGET};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// NVIDIA warp size (32 threads).
const WARP_SIZE: usize = 32;

/// Maximum block size for RMSNorm (8 warps = 256 threads).
const MAX_BLOCK_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for PTX RMSNorm kernel generation.
///
/// RMSNorm normalizes by root mean square without mean subtraction:
/// `y_i = weight_i * x_i / sqrt(mean(x^2) + eps)`
///
/// Used in Llama, Qwen3, GLM, and other modern LLMs.
#[derive(Debug, Clone)]
pub struct PtxRmsNormConfig {
    /// Kernel function name in the PTX module.
    pub kernel_name: String,
    /// Hidden dimension size (the normalization dimension length).
    pub hidden_dim: usize,
    /// Epsilon for numerical stability in the rsqrt denominator.
    pub eps: f32,
    /// SM target for the PTX prelude (e.g., "sm_80").
    pub sm_target: String,
}

impl PtxRmsNormConfig {
    /// Create an RMSNorm config with default sm_80 target.
    pub fn new(kernel_name: &str, hidden_dim: usize, eps: f32) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            hidden_dim,
            eps,
            sm_target: DEFAULT_SM_TARGET.to_string(),
        }
    }

    /// Set the SM target (e.g., "sm_70", "sm_80", "sm_90").
    #[must_use]
    pub fn with_sm_target(mut self, target: &str) -> Self {
        self.sm_target = target.to_string();
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PtxCodegenError> {
        if self.hidden_dim == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "hidden_dim must be > 0".into(),
            ));
        }
        if self.kernel_name.is_empty() {
            return Err(PtxCodegenError::InvalidParameter(
                "kernel_name must not be empty".into(),
            ));
        }
        if !self.eps.is_finite() || self.eps < 0.0 {
            return Err(PtxCodegenError::InvalidParameter(
                "eps must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }

    /// Block size: number of threads per block.
    ///
    /// Round dim up to next multiple of WARP_SIZE, cap at MAX_BLOCK_SIZE.
    #[must_use]
    pub fn block_size(&self) -> usize {
        let rounded = self.hidden_dim.div_ceil(WARP_SIZE) * WARP_SIZE;
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

/// Emit a complete PTX module for f32 RMSNorm.
///
/// Generates raw PTX assembly implementing RMS normalization with
/// warp-level shuffle reduction. For dim <= 32, uses pure warp shuffles
/// (no shared memory). For dim > 32, uses shared memory for cross-warp
/// reduction.
///
/// # Parameters (in generated kernel)
///
/// * `param_input`    — pointer to input tensor (f32)
/// * `param_output`   — pointer to output tensor (f32)
/// * `param_weight`   — pointer to weight tensor (f32, length = hidden_dim)
/// * `param_row_size` — u32, number of elements in the normalization dimension
/// * `param_num_rows` — u32, number of rows (outer dimensions product)
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_rmsnorm::{emit_ptx_rmsnorm, PtxRmsNormConfig};
/// let config = PtxRmsNormConfig::new("rmsnorm_4096", 4096, 1e-5);
/// let ptx = emit_ptx_rmsnorm(&config).unwrap();
/// assert!(ptx.contains(".entry rmsnorm_4096"));
/// assert!(ptx.contains("rsqrt.approx.f32"));
/// ```
pub fn emit_ptx_rmsnorm(config: &PtxRmsNormConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let dim = config.hidden_dim;
    let name = &config.kernel_name;
    let block_size = config.block_size();
    let num_warps = config.num_warps();
    let warp_only = config.is_warp_only();
    let eps = config.eps;

    let zero = format_ptx_float(0.0);
    let eps_hex = format_ptx_float(eps);
    let rcp_dim = format_ptx_float(1.0 / dim as f32);

    let mut ptx = String::with_capacity(8192);

    // -- Module header --
    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// RMSNorm f32: hidden_dim={dim}, eps={eps}, \
         block_size={block_size}, warps={num_warps}\n\
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

    // -- Kernel entry point (no beta parameter) --
    ptx.push_str(&format!(
        ".visible .entry {name}(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u64 param_weight,\n\
         \x20   .param .u32 param_row_size,\n\
         \x20   .param .u32 param_num_rows\n\
         )\n"
    ));

    ptx.push_str(&format!(".reqntid {block_size}\n{{\n"));

    // -- Register declarations --
    ptx.push_str(
        "\x20   // Register declarations\n\
         \x20   .reg .u32  %r<20>;\n\
         \x20   .reg .f32  %f<16>;\n\
         \x20   .reg .u64  %rd<12>;\n\
         \x20   .reg .pred %p<6>;\n\n",
    );

    // -- Load parameters --
    ptx.push_str(
        "\x20   // Load kernel parameters\n\
         \x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u64  %rd2, [param_weight];\n\
         \x20   ld.param.u32  %r0,  [param_row_size];\n\
         \x20   ld.param.u32  %r1,  [param_num_rows];\n\n",
    );

    // -- Compute thread/block indices --
    ptx.push_str(
        "\x20   // Thread and block indices\n\
         \x20   mov.u32       %r2, %tid.x;           // tid = threadIdx.x\n\
         \x20   mov.u32       %r3, %ctaid.x;         // row = blockIdx.x\n\n",
    );

    // -- Bounds check: row < num_rows --
    ptx.push_str(
        "\x20   // Bounds check: row < num_rows\n\
         \x20   setp.ge.u32   %p0, %r3, %r1;         // row >= num_rows?\n\
         \x20   @%p0 bra      RMS_EXIT;\n\n",
    );

    // -- Compute row pointer for input and output --
    ptx.push_str(
        "\x20   // Compute row pointers\n\
         \x20   mul.lo.u32    %r4, %r3, %r0;         // row * row_size\n\
         \x20   mul.wide.u32  %rd4, %r4, 4;          // byte offset\n\
         \x20   add.u64       %rd5, %rd0, %rd4;      // &input[row * row_size]\n\
         \x20   add.u64       %rd6, %rd1, %rd4;      // &output[row * row_size]\n\n",
    );

    // -- Compute warp ID and lane ID --
    ptx.push_str(
        "\x20   // Warp/lane decomposition\n\
         \x20   shr.u32       %r5, %r2, 5;           // warp_id = tid >> 5\n\
         \x20   and.b32       %r6, %r2, 31;          // lane_id = tid & 31\n\n",
    );

    // =================================================================
    // Phase 1: compute sum of x^2
    // =================================================================
    ptx.push_str(&format!(
        "\x20   // ---- Phase 1: compute mean(x^2) ----\n\
         \x20   mov.f32       %f2, {zero};            // local_sq_sum = 0.0\n\
         \x20   mov.u32       %r7, %r2;               // i = tid\n\
         RMS_SQ_LOOP:\n\
         \x20   setp.ge.u32   %p1, %r7, %r0;          // i >= row_size?\n\
         \x20   @%p1 bra      RMS_SQ_REDUCE;\n\
         \x20   // Load input[i], accumulate x^2\n\
         \x20   mul.wide.u32  %rd7, %r7, 4;           // byte offset\n\
         \x20   add.u64       %rd8, %rd5, %rd7;       // &row_in[i]\n\
         \x20   ld.global.f32 %f3, [%rd8];            // val = row_in[i]\n\
         \x20   mul.f32       %f4, %f3, %f3;          // val^2\n\
         \x20   add.f32       %f2, %f2, %f4;          // local_sq_sum += val^2\n\
         \x20   add.u32       %r7, %r7, {block_size}; // i += block_size\n\
         \x20   bra           RMS_SQ_LOOP;\n\
         RMS_SQ_REDUCE:\n\n"
    ));

    // Warp-level sum reduction for x^2 sum
    emit_warp_reduce_sum(&mut ptx, "%f2");

    // Cross-warp reduction if needed
    if !warp_only {
        emit_cross_warp_reduce_sum(&mut ptx, num_warps);
    }

    // Compute mean_x2 = sq_sum / dim, then rsqrt(mean_x2 + eps)
    ptx.push_str(&format!(
        "\x20   // mean_x2 = sq_sum / dim\n\
         \x20   mul.f32       %f2, %f2, {rcp_dim};    // %f2 = mean(x^2)\n\
         \x20   // inv_rms = rsqrt(mean(x^2) + eps)\n\
         \x20   add.f32       %f6, %f2, {eps_hex};     // mean_x2 + eps\n\
         \x20   rsqrt.approx.f32 %f7, %f6;            // %f7 = 1/sqrt(mean_x2+eps)\n\n"
    ));

    // =================================================================
    // Phase 2: normalize with weight (no mean subtraction, no beta)
    // y_i = weight_i * x_i * inv_rms
    // =================================================================
    ptx.push_str(&format!(
        "\x20   // ---- Phase 2: normalize + scale ----\n\
         \x20   mov.u32       %r7, %r2;               // i = tid\n\
         RMS_NORM_LOOP:\n\
         \x20   setp.ge.u32   %p1, %r7, %r0;          // i >= row_size?\n\
         \x20   @%p1 bra      RMS_EXIT;\n\
         \x20   // Load input[i]\n\
         \x20   mul.wide.u32  %rd7, %r7, 4;           // byte offset for row elem\n\
         \x20   add.u64       %rd8, %rd5, %rd7;       // &row_in[i]\n\
         \x20   ld.global.f32 %f8, [%rd8];            // x_i\n\
         \x20   // normalized = x_i * inv_rms\n\
         \x20   mul.f32       %f10, %f8, %f7;         // x_i * inv_rms\n\
         \x20   // Load weight[i]\n\
         \x20   add.u64       %rd9, %rd2, %rd7;       // &weight[i]\n\
         \x20   ld.global.f32 %f11, [%rd9];           // weight_i\n\
         \x20   // y_i = weight_i * normalized\n\
         \x20   mul.f32       %f13, %f11, %f10;       // weight * (x * inv_rms)\n\
         \x20   // Store output[i]\n\
         \x20   add.u64       %rd11, %rd6, %rd7;      // &row_out[i]\n\
         \x20   st.global.f32 [%rd11], %f13;          // output[i] = y_i\n\
         \x20   add.u32       %r7, %r7, {block_size}; // i += block_size\n\
         \x20   bra           RMS_NORM_LOOP;\n\n"
    ));

    // -- Kernel exit --
    ptx.push_str(
        "RMS_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Warp reduction helpers
// ---------------------------------------------------------------------------

/// Emit warp-level sum reduction on register `reg` using `shfl.down.sync`.
///
/// After this sequence, lane 0 of each warp holds the warp-wide sum.
/// The result is then broadcast to all lanes via `shfl.idx.sync`.
fn emit_warp_reduce_sum(ptx: &mut String, reg: &str) {
    ptx.push_str("\x20   // Warp-level sum reduction (shfl.down.sync)\n");
    for offset in [16, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "\x20   shfl.down.sync.b32 %f14, {reg}, {offset}, 31, 0xFFFFFFFF;\n\
             \x20   add.f32       {reg}, {reg}, %f14;\n"
        ));
    }
    // Broadcast lane 0's result to all lanes in the warp
    ptx.push_str(&format!(
        "\x20   shfl.idx.sync.b32 {reg}, {reg}, 0, 31, 0xFFFFFFFF;\n\n"
    ));
}

/// Emit cross-warp sum reduction using shared memory `warp_scratch`.
///
/// Each warp leader (lane 0) writes to shared memory, then a barrier
/// synchronizes all warps. Warp 0 reduces the scratch buffer, then
/// the result is broadcast via shared memory.
fn emit_cross_warp_reduce_sum(ptx: &mut String, num_warps: usize) {
    let zero = format_ptx_float(0.0);
    ptx.push_str("\x20   // Cross-warp sum reduction via shared memory\n\
         \x20   // Warp leader (lane 0) writes local sum to warp_scratch\n\
         \x20   setp.eq.u32   %p2, %r6, 0;            // lane_id == 0?\n\
         \x20   @!%p2 bra     CROSS_RMS_LOAD;\n\
         \x20   mul.wide.u32  %rd7, %r5, 4;           // warp_id * 4\n\
         \x20   mov.u64       %rd8, warp_scratch;\n\
         \x20   add.u64       %rd7, %rd8, %rd7;\n");

    ptx.push_str(&format!(
        "\x20   st.shared.f32 [%rd7], %f2;\n\
         CROSS_RMS_LOAD:\n\
         \x20   bar.sync      0;\n\
         \x20   // All threads read scratch and reduce\n\
         \x20   mov.f32       %f2, {zero};\n"
    ));

    // Each thread in warp 0 loads one scratch element (if tid < num_warps)
    ptx.push_str(&format!(
        "\x20   setp.lt.u32   %p3, %r2, {num_warps};  // tid < num_warps?\n\
         \x20   @!%p3 bra     CROSS_RMS_DONE;\n\
         \x20   mul.wide.u32  %rd7, %r2, 4;           // tid * 4\n\
         \x20   mov.u64       %rd8, warp_scratch;\n\
         \x20   add.u64       %rd7, %rd8, %rd7;\n\
         \x20   ld.shared.f32 %f2, [%rd7];\n\
         CROSS_RMS_DONE:\n"
    ));

    // Warp reduce the loaded values
    emit_warp_reduce_sum(ptx, "%f2");

    // Broadcast result to all threads via shared memory
    ptx.push_str("\x20   // Broadcast RMS sum to all threads via shared memory\n\
         \x20   setp.eq.u32   %p2, %r2, 0;            // tid == 0?\n\
         \x20   @!%p2 bra     BCAST_RMS_LOAD;\n\
         \x20   mov.u64       %rd8, warp_scratch;\n\
         \x20   st.shared.f32 [%rd8], %f2;\n\
         BCAST_RMS_LOAD:\n\
         \x20   bar.sync      0;\n\
         \x20   mov.u64       %rd8, warp_scratch;\n\
         \x20   ld.shared.f32 %f2, [%rd8];\n\n");
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Convenience: emit PTX RMSNorm with default sm_80 target.
pub fn emit_ptx_rmsnorm_default(
    name: &str,
    hidden_dim: usize,
    eps: f32,
) -> Result<String, PtxCodegenError> {
    emit_ptx_rmsnorm(&PtxRmsNormConfig::new(name, hidden_dim, eps))
}

/// Compute the grid and block dimensions for a PTX RMSNorm kernel.
///
/// Grid: `(num_rows, 1, 1)` — one block per row.
/// Block: `(block_size, 1, 1)` — threads cooperate on one row.
///
/// # Returns
///
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_rmsnorm_launch_config(num_rows: usize, hidden_dim: usize) -> ([usize; 3], [usize; 3]) {
    let config = PtxRmsNormConfig::new("_", hidden_dim, 1e-5);
    let block_size = config.block_size();
    let grid = [num_rows, 1, 1];
    let block = [block_size, 1, 1];
    (grid, block)
}

/// Generate PTX for RMSNorm with default settings.
///
/// Convenience wrapper around [`emit_ptx_rmsnorm`] that uses
/// default settings (kernel name `"ptx_rmsnorm_f32"`, sm_80 target).
pub fn generate_rmsnorm_ptx(hidden_dim: usize, eps: f32) -> String {
    emit_ptx_rmsnorm_default("ptx_rmsnorm_f32", hidden_dim, eps)
        .expect("RMSNorm PTX generation failed")
}

/// Compute RMSNorm on CPU for reference/testing.
///
/// `y_i = weight_i * x_i / sqrt(mean(x^2) + eps)`
///
/// Operates on a single row of `hidden_dim` elements.
pub fn rmsnorm_reference(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(input.len(), weight.len());
    let dim = input.len() as f32;
    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / dim;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &w)| w * x * inv_rms)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_rmsnorm_tests.rs"]
mod ptx_rmsnorm_tests;
