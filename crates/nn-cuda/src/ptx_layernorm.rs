// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for Layer Normalization (LayerNorm).
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 LayerNorm
//! along the last dimension. LayerNorm is the normalization layer used in
//! transformer models (BERT, GPT-2, Whisper) with full mean subtraction,
//! variance computation, and affine transform (gamma/beta).
//!
//! ## Algorithm
//!
//! LayerNorm: `y_i = gamma_i * (x_i - mean) / sqrt(var + eps) + beta_i`
//!
//! Three phases:
//! 1. **Compute mean** -- each thread reads a strided slice of the
//!    normalization dimension, accumulates the sum, then reduces across the
//!    warp with `shfl.down.sync`. For dim > 32, cross-warp reduction uses
//!    shared memory. The sum is divided by `dim` to produce the mean.
//! 2. **Compute variance** -- each thread computes `(x_i - mean)^2` and
//!    accumulates, then reduces. Variance = sum / dim.
//! 3. **Normalize + affine** -- each thread computes
//!    `y_i = gamma_i * (x_i - mean) * rsqrt(var + eps) + beta_i`.
//!
//! ## Comparison with RMSNorm
//!
//! | Property         | LayerNorm                     | RMSNorm                    |
//! |------------------|-------------------------------|----------------------------|
//! | Formula          | gamma*(x-mean)/std+beta       | weight * x * rsqrt(rms+e)  |
//! | Phases           | 3 (mean, var, norm+affine)    | 2 (rms, norm+scale)        |
//! | Parameters       | gamma + beta                  | weight only                |
//! | Mean subtraction | Yes                           | No                         |
//! | Used in          | BERT, GPT-2, Whisper, Kokoro  | Llama, Qwen3, GLM, Gemma   |
//!
//! ## Warp-level vs shared memory reduction
//!
//! Same strategy as [`ptx_rmsnorm`]:
//! - **dim <= 32 (single warp):** Pure warp shuffle reduction.
//! - **dim > 32 (multi-warp):** Warp shuffles + shared memory cross-warp.
//!
//! ## Kernel interface
//!
//! Parameters (in generated kernel):
//! - `param_input`    -- pointer to input tensor (f32)
//! - `param_output`   -- pointer to output tensor (f32)
//! - `param_gamma`    -- pointer to gamma (scale) tensor (f32, length = normalized_shape)
//! - `param_beta`     -- pointer to beta (bias) tensor (f32, length = normalized_shape)
//! - `param_row_size` -- u32, number of elements in the normalization dimension
//! - `param_num_rows` -- u32, number of rows (outer dimensions product)
//!
//! ## Thread block configuration
//!
//! Block: `(block_size, 1, 1)` where `block_size = min(dim_rounded_up_to_warp, 256)`.
//! Grid: `(num_rows, 1, 1)` -- one block per row.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude, PtxCodegenError, DEFAULT_SM_TARGET};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// NVIDIA warp size (32 threads).
const WARP_SIZE: usize = 32;

/// Maximum block size for LayerNorm (8 warps = 256 threads).
const MAX_BLOCK_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for PTX LayerNorm kernel generation.
///
/// LayerNorm normalizes by mean subtraction and variance scaling with affine:
/// `y_i = gamma_i * (x_i - mean) / sqrt(var + eps) + beta_i`
///
/// Used in BERT, GPT-2, Whisper, Kokoro, and other transformer models.
#[derive(Debug, Clone)]
pub struct PtxLayerNormConfig {
    /// Kernel function name in the PTX module.
    pub kernel_name: String,
    /// Normalized shape size (the normalization dimension length).
    pub normalized_shape: usize,
    /// Epsilon for numerical stability in the rsqrt denominator.
    pub eps: f32,
    /// SM target for the PTX prelude (e.g., "sm_80").
    pub sm_target: String,
}

impl PtxLayerNormConfig {
    /// Create a LayerNorm config with default sm_80 target.
    pub fn new(kernel_name: &str, normalized_shape: usize, eps: f32) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            normalized_shape,
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
        if self.normalized_shape == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "normalized_shape must be > 0".into(),
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
        let rounded = self.normalized_shape.div_ceil(WARP_SIZE) * WARP_SIZE;
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
            self.num_warps() * 4
        }
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Emit a complete PTX module for f32 LayerNorm.
///
/// Generates raw PTX assembly implementing layer normalization with
/// warp-level shuffle reduction. For dim <= 32, uses pure warp shuffles
/// (no shared memory). For dim > 32, uses shared memory for cross-warp
/// reduction.
///
/// # Parameters (in generated kernel)
///
/// * `param_input`    -- pointer to input tensor (f32)
/// * `param_output`   -- pointer to output tensor (f32)
/// * `param_gamma`    -- pointer to gamma (scale) tensor (f32)
/// * `param_beta`     -- pointer to beta (bias) tensor (f32)
/// * `param_row_size` -- u32, normalized dimension size
/// * `param_num_rows` -- u32, number of rows (outer dims product)
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_layernorm::{emit_ptx_layernorm, PtxLayerNormConfig};
/// let config = PtxLayerNormConfig::new("layernorm_768", 768, 1e-5);
/// let ptx = emit_ptx_layernorm(&config).unwrap();
/// assert!(ptx.contains(".entry layernorm_768"));
/// assert!(ptx.contains("rsqrt.approx.f32"));
/// ```
pub fn emit_ptx_layernorm(config: &PtxLayerNormConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let dim = config.normalized_shape;
    let name = &config.kernel_name;
    let block_size = config.block_size();
    let num_warps = config.num_warps();
    let warp_only = config.is_warp_only();
    let eps = config.eps;

    let zero = format_ptx_float(0.0);
    let eps_hex = format_ptx_float(eps);
    let rcp_dim = format_ptx_float(1.0 / dim as f32);

    let mut ptx = String::with_capacity(12288);

    // -- Module header --
    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// LayerNorm f32: normalized_shape={dim}, eps={eps}, \
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

    // -- Kernel entry point --
    ptx.push_str(&format!(
        ".visible .entry {name}(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u64 param_gamma,\n\
         \x20   .param .u64 param_beta,\n\
         \x20   .param .u32 param_row_size,\n\
         \x20   .param .u32 param_num_rows\n\
         )\n"
    ));

    ptx.push_str(&format!(".reqntid {block_size}\n{{\n"));

    // -- Register declarations --
    ptx.push_str(
        "\x20   // Register declarations\n\
         \x20   .reg .u32  %r<20>;\n\
         \x20   .reg .f32  %f<20>;\n\
         \x20   .reg .u64  %rd<14>;\n\
         \x20   .reg .pred %p<6>;\n\n",
    );

    // -- Load parameters --
    ptx.push_str(
        "\x20   // Load kernel parameters\n\
         \x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u64  %rd2, [param_gamma];\n\
         \x20   ld.param.u64  %rd3, [param_beta];\n\
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
         \x20   @%p0 bra      LN_EXIT;\n\n",
    );

    // -- Compute row pointer for input and output --
    ptx.push_str(
        "\x20   // Compute row pointers\n\
         \x20   mul.lo.u32    %r4, %r3, %r0;         // row * row_size\n\
         \x20   mul.wide.u32  %rd5, %r4, 4;          // byte offset\n\
         \x20   add.u64       %rd6, %rd0, %rd5;      // &input[row * row_size]\n\
         \x20   add.u64       %rd7, %rd1, %rd5;      // &output[row * row_size]\n\n",
    );

    // -- Compute warp ID and lane ID --
    ptx.push_str(
        "\x20   // Warp/lane decomposition\n\
         \x20   shr.u32       %r5, %r2, 5;           // warp_id = tid >> 5\n\
         \x20   and.b32       %r6, %r2, 31;          // lane_id = tid & 31\n\n",
    );

    // =================================================================
    // Phase 1: compute mean
    // =================================================================
    ptx.push_str(&format!(
        "\x20   // ---- Phase 1: compute mean ----\n\
         \x20   mov.f32       %f0, {zero};            // local_sum = 0.0\n\
         \x20   mov.u32       %r7, %r2;               // i = tid\n\
         LN_MEAN_LOOP:\n\
         \x20   setp.ge.u32   %p1, %r7, %r0;          // i >= row_size?\n\
         \x20   @%p1 bra      LN_MEAN_REDUCE;\n\
         \x20   // Load input[i], accumulate sum\n\
         \x20   mul.wide.u32  %rd8, %r7, 4;           // byte offset\n\
         \x20   add.u64       %rd9, %rd6, %rd8;       // &row_in[i]\n\
         \x20   ld.global.f32 %f1, [%rd9];            // val = row_in[i]\n\
         \x20   add.f32       %f0, %f0, %f1;          // local_sum += val\n\
         \x20   add.u32       %r7, %r7, {block_size}; // i += block_size\n\
         \x20   bra           LN_MEAN_LOOP;\n\
         LN_MEAN_REDUCE:\n\n"
    ));

    // Warp-level sum reduction for mean
    emit_warp_reduce_sum(&mut ptx, "%f0");

    // Cross-warp reduction if needed
    if !warp_only {
        emit_cross_warp_reduce_sum(&mut ptx, num_warps, "CROSS_MEAN");
    }

    // Compute mean = sum / dim
    ptx.push_str(&format!(
        "\x20   // mean = sum / dim\n\
         \x20   mul.f32       %f0, %f0, {rcp_dim};    // %f0 = mean\n\n"
    ));

    // =================================================================
    // Phase 2: compute variance = mean((x - mean)^2)
    // =================================================================
    ptx.push_str(&format!(
        "\x20   // ---- Phase 2: compute variance ----\n\
         \x20   mov.f32       %f2, {zero};            // local_var_sum = 0.0\n\
         \x20   mov.u32       %r7, %r2;               // i = tid\n\
         LN_VAR_LOOP:\n\
         \x20   setp.ge.u32   %p1, %r7, %r0;          // i >= row_size?\n\
         \x20   @%p1 bra      LN_VAR_REDUCE;\n\
         \x20   // Load input[i], compute (x - mean)^2\n\
         \x20   mul.wide.u32  %rd8, %r7, 4;           // byte offset\n\
         \x20   add.u64       %rd9, %rd6, %rd8;       // &row_in[i]\n\
         \x20   ld.global.f32 %f3, [%rd9];            // val = row_in[i]\n\
         \x20   sub.f32       %f4, %f3, %f0;          // diff = val - mean\n\
         \x20   mul.f32       %f5, %f4, %f4;          // diff^2\n\
         \x20   add.f32       %f2, %f2, %f5;          // local_var_sum += diff^2\n\
         \x20   add.u32       %r7, %r7, {block_size}; // i += block_size\n\
         \x20   bra           LN_VAR_LOOP;\n\
         LN_VAR_REDUCE:\n\n"
    ));

    // Warp-level sum reduction for variance
    emit_warp_reduce_sum(&mut ptx, "%f2");

    // Cross-warp reduction if needed
    if !warp_only {
        emit_cross_warp_reduce_sum(&mut ptx, num_warps, "CROSS_VAR");
    }

    // Compute var = var_sum / dim, then rsqrt(var + eps)
    ptx.push_str(&format!(
        "\x20   // var = var_sum / dim\n\
         \x20   mul.f32       %f2, %f2, {rcp_dim};    // %f2 = variance\n\
         \x20   // inv_std = rsqrt(var + eps)\n\
         \x20   add.f32       %f6, %f2, {eps_hex};     // var + eps\n\
         \x20   rsqrt.approx.f32 %f7, %f6;            // %f7 = 1/sqrt(var+eps)\n\n"
    ));

    // =================================================================
    // Phase 3: normalize with gamma and beta
    // y_i = gamma_i * (x_i - mean) * inv_std + beta_i
    // =================================================================
    ptx.push_str(&format!(
        "\x20   // ---- Phase 3: normalize + affine ----\n\
         \x20   mov.u32       %r7, %r2;               // i = tid\n\
         LN_NORM_LOOP:\n\
         \x20   setp.ge.u32   %p1, %r7, %r0;          // i >= row_size?\n\
         \x20   @%p1 bra      LN_EXIT;\n\
         \x20   // Load input[i]\n\
         \x20   mul.wide.u32  %rd8, %r7, 4;           // byte offset for row elem\n\
         \x20   add.u64       %rd9, %rd6, %rd8;       // &row_in[i]\n\
         \x20   ld.global.f32 %f8, [%rd9];            // x_i\n\
         \x20   // normalized = (x_i - mean) * inv_std\n\
         \x20   sub.f32       %f9, %f8, %f0;          // x_i - mean\n\
         \x20   mul.f32       %f10, %f9, %f7;         // (x_i - mean) * inv_std\n\
         \x20   // Load gamma[i]\n\
         \x20   add.u64       %rd10, %rd2, %rd8;      // &gamma[i]\n\
         \x20   ld.global.f32 %f11, [%rd10];          // gamma_i\n\
         \x20   // Load beta[i]\n\
         \x20   add.u64       %rd11, %rd3, %rd8;      // &beta[i]\n\
         \x20   ld.global.f32 %f12, [%rd11];          // beta_i\n\
         \x20   // y_i = gamma_i * normalized + beta_i\n\
         \x20   fma.rn.f32    %f13, %f11, %f10, %f12; // gamma * norm + beta\n\
         \x20   // Store output[i]\n\
         \x20   add.u64       %rd12, %rd7, %rd8;      // &row_out[i]\n\
         \x20   st.global.f32 [%rd12], %f13;          // output[i] = y_i\n\
         \x20   add.u32       %r7, %r7, {block_size}; // i += block_size\n\
         \x20   bra           LN_NORM_LOOP;\n\n"
    ));

    // -- Kernel exit --
    ptx.push_str(
        "LN_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Warp reduction helpers
// ---------------------------------------------------------------------------

/// Emit warp-level sum reduction on register `reg` using `shfl.down.sync`.
fn emit_warp_reduce_sum(ptx: &mut String, reg: &str) {
    ptx.push_str("\x20   // Warp-level sum reduction (shfl.down.sync)\n");
    for offset in [16, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "\x20   shfl.down.sync.b32 %f18, {reg}, {offset}, 31, 0xFFFFFFFF;\n\
             \x20   add.f32       {reg}, {reg}, %f18;\n"
        ));
    }
    ptx.push_str(&format!(
        "\x20   shfl.idx.sync.b32 {reg}, {reg}, 0, 31, 0xFFFFFFFF;\n\n"
    ));
}

/// Emit cross-warp sum reduction using shared memory `warp_scratch`.
fn emit_cross_warp_reduce_sum(ptx: &mut String, num_warps: usize, label_prefix: &str) {
    let zero = format_ptx_float(0.0);
    // Determine which register based on label prefix
    let reg = if label_prefix.contains("MEAN") {
        "%f0"
    } else {
        "%f2"
    };

    ptx.push_str(&format!(
        "\x20   // Cross-warp sum reduction via shared memory ({label_prefix})\n\
         \x20   setp.eq.u32   %p2, %r6, 0;            // lane_id == 0?\n\
         \x20   @!%p2 bra     {label_prefix}_LOAD;\n\
         \x20   mul.wide.u32  %rd8, %r5, 4;           // warp_id * 4\n\
         \x20   mov.u64       %rd9, warp_scratch;\n\
         \x20   add.u64       %rd8, %rd9, %rd8;\n\
         \x20   st.shared.f32 [%rd8], {reg};\n\
         {label_prefix}_LOAD:\n\
         \x20   bar.sync      0;\n\
         \x20   mov.f32       {reg}, {zero};\n\
         \x20   setp.lt.u32   %p3, %r2, {num_warps};  // tid < num_warps?\n\
         \x20   @!%p3 bra     {label_prefix}_DONE;\n\
         \x20   mul.wide.u32  %rd8, %r2, 4;           // tid * 4\n\
         \x20   mov.u64       %rd9, warp_scratch;\n\
         \x20   add.u64       %rd8, %rd9, %rd8;\n\
         \x20   ld.shared.f32 {reg}, [%rd8];\n\
         {label_prefix}_DONE:\n"
    ));
    emit_warp_reduce_sum(ptx, reg);
    ptx.push_str(&format!(
        "\x20   // Broadcast to all threads via shared memory\n\
         \x20   setp.eq.u32   %p2, %r2, 0;            // tid == 0?\n\
         \x20   @!%p2 bra     BCAST_{label_prefix}_LOAD;\n\
         \x20   mov.u64       %rd9, warp_scratch;\n\
         \x20   st.shared.f32 [%rd9], {reg};\n\
         BCAST_{label_prefix}_LOAD:\n\
         \x20   bar.sync      0;\n\
         \x20   mov.u64       %rd9, warp_scratch;\n\
         \x20   ld.shared.f32 {reg}, [%rd9];\n\n"
    ));
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Convenience: emit PTX LayerNorm with default sm_80 target.
pub fn emit_ptx_layernorm_default(
    name: &str,
    normalized_shape: usize,
    eps: f32,
) -> Result<String, PtxCodegenError> {
    emit_ptx_layernorm(&PtxLayerNormConfig::new(name, normalized_shape, eps))
}

/// Compute the grid and block dimensions for a PTX LayerNorm kernel.
///
/// Grid: `(num_rows, 1, 1)` -- one block per row.
/// Block: `(block_size, 1, 1)` -- threads cooperate on one row.
///
/// # Returns
///
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_layernorm_launch_config(
    num_rows: usize,
    normalized_shape: usize,
) -> ([usize; 3], [usize; 3]) {
    let config = PtxLayerNormConfig::new("_", normalized_shape, 1e-5);
    let block_size = config.block_size();
    let grid = [num_rows, 1, 1];
    let block = [block_size, 1, 1];
    (grid, block)
}

/// Generate PTX for LayerNorm with default settings.
///
/// Convenience wrapper around [`emit_ptx_layernorm`] that uses
/// default settings (kernel name `"ptx_layernorm_f32"`, sm_80 target).
pub fn generate_layernorm_ptx(normalized_shape: usize) -> String {
    emit_ptx_layernorm_default("ptx_layernorm_f32", normalized_shape, 1e-5)
        .expect("LayerNorm PTX generation failed")
}

/// Compute LayerNorm on CPU for reference/testing.
///
/// `y_i = gamma_i * (x_i - mean) / sqrt(var + eps) + beta_i`
///
/// Operates on a single row of `normalized_shape` elements.
pub fn layernorm_reference(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(input.len(), gamma.len());
    assert_eq!(input.len(), beta.len());
    let dim = input.len() as f32;
    let mean: f32 = input.iter().sum::<f32>() / dim;
    let var: f32 = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / dim;
    let inv_std = 1.0 / (var + eps).sqrt();
    input
        .iter()
        .zip(gamma.iter())
        .zip(beta.iter())
        .map(|((&x, &g), &b)| g * (x - mean) * inv_std + b)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_layernorm_tests.rs"]
mod ptx_layernorm_tests;
