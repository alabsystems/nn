// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for Instance Normalization (InstanceNorm).
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 InstanceNorm.
//! Instance normalization normalizes each (batch, channel) pair independently
//! across the spatial dimensions H*W. Used in style transfer networks,
//! generative models (CycleGAN, pix2pix), and audio synthesis (Kokoro, WaveNet).
//!
//! ## Algorithm
//!
//! InstanceNorm: `y = gamma_c * (x - mean) / sqrt(var + eps) + beta_c`
//!
//! For each (batch, channel) pair:
//! 1. **Compute mean** -- average over H*W spatial elements.
//! 2. **Compute variance** -- mean((x - mean)^2) over spatial elements.
//! 3. **Normalize + affine** -- per-channel gamma and beta applied after
//!    normalization.
//!
//! ## Data format
//!
//! NCHW (batch, channels, height, width). Each thread block handles one
//! (batch, channel) pair. Threads cooperate to reduce over H*W spatial
//! elements.
//!
//! ## Comparison with other norms
//!
//! | Property         | InstanceNorm            | BatchNorm (inference) | GroupNorm               | LayerNorm               |
//! |------------------|------------------------|-----------------------|-------------------------|-------------------------|
//! | Norm axis        | Per-(N,C) over H*W     | Per-channel (C)       | Per-group (G channels)  | Last dim                |
//! | Stats source     | Computed per-instance   | Pre-computed          | Computed per-group      | Computed per-row        |
//! | Reduction needed | Yes (over spatial)      | No                    | Yes (within group)      | Yes (within row)        |
//! | Used in          | Style transfer, audio   | ResNet, VGG           | DETR, Stable Diffusion  | BERT, GPT-2, Whisper    |
//!
//! ## Kernel interface
//!
//! Parameters (in generated kernel):
//! - `param_input`    -- pointer to input tensor (f32), NCHW
//! - `param_output`   -- pointer to output tensor (f32), NCHW
//! - `param_gamma`    -- pointer to gamma/scale (f32, length = C)
//! - `param_beta`     -- pointer to beta/bias (f32, length = C)
//! - `param_spatial`  -- u32, H * W (spatial elements per channel)
//!
//! ## Thread block configuration
//!
//! Block: `(INSTANCENORM_BLOCK_SIZE, 1, 1)` = (256, 1, 1).
//! Grid: `(N * C, 1, 1)` -- one block per (batch, channel) pair.

use crate::codegen_ptx::format_ptx_float;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Block size for InstanceNorm kernel (256 threads).
pub const INSTANCENORM_BLOCK_SIZE: u32 = 256;

/// NVIDIA warp size (32 threads).
const WARP_SIZE: usize = 32;

/// Maximum block size for InstanceNorm (8 warps = 256 threads).
const MAX_BLOCK_SIZE: usize = INSTANCENORM_BLOCK_SIZE as usize;

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generate a complete PTX module for f32 Instance Normalization.
///
/// Produces PTX 7.0 / sm_70 assembly implementing instance normalization.
/// Each thread block handles one (batch, channel) pair, computing mean and
/// variance over H*W spatial dimensions via cooperative warp reduction,
/// then normalizing: `out = gamma * (x - mean) / sqrt(var + eps) + beta`.
///
/// # Arguments
///
/// * `channels` -- number of channels C (used in comment metadata)
/// * `height`   -- spatial height H (used in comment metadata)
/// * `width`    -- spatial width W (used in comment metadata)
/// * `eps`      -- numerical stability epsilon
///
/// # Returns
///
/// A `String` containing the complete PTX module.
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_instancenorm::generate_instancenorm_ptx;
/// let ptx = generate_instancenorm_ptx(64, 32, 32, 1e-5);
/// assert!(ptx.contains(".entry instancenorm_f32"));
/// assert!(ptx.contains("rsqrt.approx.f32"));
/// ```
pub fn generate_instancenorm_ptx(channels: u32, height: u32, width: u32, eps: f32) -> String {
    let spatial = u64::from(height) * u64::from(width);
    let block_size = MAX_BLOCK_SIZE;
    let num_warps = block_size / WARP_SIZE;

    let zero = format_ptx_float(0.0);
    let eps_hex = format_ptx_float(eps);

    let mut ptx = String::with_capacity(12288);

    // -- Module header (PTX 7.0 / sm_70) --
    ptx.push_str(".version 7.0\n.target sm_70\n.address_size 64\n\n");
    ptx.push_str(&format!(
        "// InstanceNorm f32: channels={channels}, height={height}, width={width}, \
         spatial={spatial}, eps={eps}, block_size={block_size}, warps={num_warps}\n\
         // Each block handles one (batch, channel) pair\n\n"
    ));

    // -- Shared memory for cross-warp reduction --
    if num_warps > 1 {
        ptx.push_str(&format!(
            ".shared .align 4 .f32 warp_scratch[{num_warps}];\n\n"
        ));
    }

    // -- Kernel entry point --
    ptx.push_str(
        ".visible .entry instancenorm_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u64 param_gamma,\n\
         \x20   .param .u64 param_beta,\n\
         \x20   .param .u32 param_spatial\n\
         )\n",
    );

    ptx.push_str(&format!(".reqntid {block_size}\n{{\n"));

    // -- Register declarations --
    ptx.push_str(
        "\x20   // Register declarations\n\
         \x20   .reg .u32  %r<24>;\n\
         \x20   .reg .f32  %f<20>;\n\
         \x20   .reg .u64  %rd<16>;\n\
         \x20   .reg .pred %p<8>;\n\n",
    );

    // -- Load parameters --
    ptx.push_str(
        "\x20   // Load kernel parameters\n\
         \x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u64  %rd2, [param_gamma];\n\
         \x20   ld.param.u64  %rd3, [param_beta];\n\
         \x20   ld.param.u32  %r0,  [param_spatial];\n\n",
    );

    // -- Compute thread/block indices --
    // blockIdx.x = n * C + c (one block per (batch, channel) pair)
    ptx.push_str(
        "\x20   // Thread and block indices\n\
         \x20   mov.u32       %r1, %tid.x;            // tid = threadIdx.x\n\
         \x20   mov.u32       %r2, %ctaid.x;          // block_id = n*C + c\n\n",
    );

    // -- Compute base offset for this (batch, channel) pair --
    // base_offset = block_id * spatial
    ptx.push_str(
        "\x20   // Compute base offset for this (batch, channel) in NCHW\n\
         \x20   mul.lo.u32    %r3, %r2, %r0;          // block_id * spatial\n\
         \x20   mul.wide.u32  %rd4, %r3, 4;           // byte offset\n\
         \x20   add.u64       %rd5, %rd0, %rd4;       // &input[base]\n\
         \x20   add.u64       %rd6, %rd1, %rd4;       // &output[base]\n\n",
    );

    // -- Compute channel index for gamma/beta lookup --
    // The caller sets blockIdx.x = n*C + c, so we don't know C at codegen
    // time. Instead, we pass the channel pointer offset from the caller.
    // Actually, for a simpler design: gamma and beta are per-channel,
    // and blockIdx.x encodes (n, c). We need to extract c.
    // Since we don't know C at PTX generation time (it's a runtime param),
    // we load gamma[block_id % channels] -- but channels is baked.
    // Better: just pass the channel as a param, or bake channels.
    // For simplicity, bake channels:
    ptx.push_str(&format!(
        "\x20   // Compute channel index: c = block_id % channels\n\
         \x20   rem.u32       %r4, %r2, {channels};    // c = block_id % C\n\
         \x20   mul.wide.u32  %rd7, %r4, 4;            // c * 4 bytes\n\
         \x20   add.u64       %rd8, %rd2, %rd7;        // &gamma[c]\n\
         \x20   ld.global.f32 %f15, [%rd8];            // gamma_c\n\
         \x20   add.u64       %rd9, %rd3, %rd7;        // &beta[c]\n\
         \x20   ld.global.f32 %f16, [%rd9];            // beta_c\n\n"
    ));

    // -- Warp/lane decomposition --
    ptx.push_str(
        "\x20   // Warp/lane decomposition\n\
         \x20   shr.u32       %r5, %r1, 5;            // warp_id = tid >> 5\n\
         \x20   and.b32       %r6, %r1, 31;           // lane_id = tid & 31\n\n",
    );

    // -- Reciprocal of spatial for mean --
    ptx.push_str(
        "\x20   // Reciprocal of spatial for mean computation\n\
         \x20   cvt.rn.f32.u32 %f17, %r0;             // (float)spatial\n\
         \x20   rcp.approx.f32 %f18, %f17;            // 1.0 / spatial\n\n",
    );

    // =================================================================
    // Phase 1: compute mean over spatial dimensions
    // =================================================================
    ptx.push_str(&format!(
        "\x20   // ---- Phase 1: compute mean over H*W ----\n\
         \x20   mov.f32       %f0, {zero};             // local_sum = 0.0\n\
         \x20   mov.u32       %r7, %r1;                // i = tid\n\
         IN_MEAN_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r7, %r0;           // i >= spatial?\n\
         \x20   @%p0 bra      IN_MEAN_REDUCE;\n\
         \x20   // Load input[base + i], accumulate sum\n\
         \x20   mul.wide.u32  %rd10, %r7, 4;           // byte offset\n\
         \x20   add.u64       %rd11, %rd5, %rd10;      // &input[base + i]\n\
         \x20   ld.global.f32 %f1, [%rd11];            // val\n\
         \x20   add.f32       %f0, %f0, %f1;           // local_sum += val\n\
         \x20   add.u32       %r7, %r7, {block_size};  // i += block_size\n\
         \x20   bra           IN_MEAN_LOOP;\n\
         IN_MEAN_REDUCE:\n\n"
    ));

    // Warp-level sum reduction for mean
    emit_warp_reduce_sum(&mut ptx, "%f0");

    // Cross-warp reduction if needed
    if num_warps > 1 {
        emit_cross_warp_reduce_sum(&mut ptx, num_warps, "CROSS_IN_MEAN");
    }

    // Compute mean = sum / spatial
    ptx.push_str(
        "\x20   // mean = sum / spatial\n\
         \x20   mul.f32       %f0, %f0, %f18;          // %f0 = mean\n\n",
    );

    // =================================================================
    // Phase 2: compute variance = mean((x - mean)^2) over spatial
    // =================================================================
    ptx.push_str(&format!(
        "\x20   // ---- Phase 2: compute variance over H*W ----\n\
         \x20   mov.f32       %f2, {zero};             // local_var_sum = 0.0\n\
         \x20   mov.u32       %r7, %r1;                // i = tid\n\
         IN_VAR_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r7, %r0;           // i >= spatial?\n\
         \x20   @%p0 bra      IN_VAR_REDUCE;\n\
         \x20   // Load input[base + i], compute (x - mean)^2\n\
         \x20   mul.wide.u32  %rd10, %r7, 4;           // byte offset\n\
         \x20   add.u64       %rd11, %rd5, %rd10;      // &input[base + i]\n\
         \x20   ld.global.f32 %f3, [%rd11];            // val\n\
         \x20   sub.f32       %f4, %f3, %f0;           // diff = val - mean\n\
         \x20   mul.f32       %f5, %f4, %f4;           // diff^2\n\
         \x20   add.f32       %f2, %f2, %f5;           // local_var_sum += diff^2\n\
         \x20   add.u32       %r7, %r7, {block_size};  // i += block_size\n\
         \x20   bra           IN_VAR_LOOP;\n\
         IN_VAR_REDUCE:\n\n"
    ));

    // Warp-level sum reduction for variance
    emit_warp_reduce_sum(&mut ptx, "%f2");

    // Cross-warp reduction if needed
    if num_warps > 1 {
        emit_cross_warp_reduce_sum(&mut ptx, num_warps, "CROSS_IN_VAR");
    }

    // Compute var = var_sum / spatial, then rsqrt(var + eps)
    ptx.push_str(&format!(
        "\x20   // var = var_sum / spatial\n\
         \x20   mul.f32       %f2, %f2, %f18;          // %f2 = variance\n\
         \x20   // inv_std = rsqrt(var + eps)\n\
         \x20   add.f32       %f6, %f2, {eps_hex};      // var + eps\n\
         \x20   rsqrt.approx.f32 %f7, %f6;             // %f7 = 1/sqrt(var+eps)\n\n"
    ));

    // =================================================================
    // Phase 3: normalize with per-channel gamma and beta
    // out = gamma * (x - mean) * inv_std + beta
    // =================================================================
    ptx.push_str(&format!(
        "\x20   // ---- Phase 3: normalize + affine ----\n\
         \x20   mov.u32       %r7, %r1;                // i = tid\n\
         IN_NORM_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r7, %r0;           // i >= spatial?\n\
         \x20   @%p0 bra      IN_EXIT;\n\
         \x20   // Load input[base + i]\n\
         \x20   mul.wide.u32  %rd10, %r7, 4;           // byte offset\n\
         \x20   add.u64       %rd11, %rd5, %rd10;      // &input[base + i]\n\
         \x20   ld.global.f32 %f8, [%rd11];            // x_i\n\
         \x20   // normalized = (x_i - mean) * inv_std\n\
         \x20   sub.f32       %f9, %f8, %f0;           // x_i - mean\n\
         \x20   mul.f32       %f10, %f9, %f7;          // (x_i - mean) * inv_std\n\
         \x20   // y_i = gamma * normalized + beta\n\
         \x20   fma.rn.f32    %f11, %f15, %f10, %f16;  // gamma * norm + beta\n\
         \x20   // Store output[base + i]\n\
         \x20   add.u64       %rd12, %rd6, %rd10;      // &output[base + i]\n\
         \x20   st.global.f32 [%rd12], %f11;           // output = y_i\n\
         \x20   add.u32       %r7, %r7, {block_size};  // i += block_size\n\
         \x20   bra           IN_NORM_LOOP;\n\n"
    ));

    // -- Kernel exit --
    ptx.push_str(
        "IN_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Warp reduction helpers
// ---------------------------------------------------------------------------

/// Emit warp-level sum reduction on register `reg` using `shfl.down.sync`.
fn emit_warp_reduce_sum(ptx: &mut String, reg: &str) {
    ptx.push_str("\x20   // Warp-level sum reduction (shfl.down.sync)\n");
    for offset in [16, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "\x20   shfl.down.sync.b32 %f19, {reg}, {offset}, 31, 0xFFFFFFFF;\n\
             \x20   add.f32       {reg}, {reg}, %f19;\n"
        ));
    }
    ptx.push_str(&format!(
        "\x20   shfl.idx.sync.b32 {reg}, {reg}, 0, 31, 0xFFFFFFFF;\n\n"
    ));
}

/// Emit cross-warp sum reduction using shared memory `warp_scratch`.
fn emit_cross_warp_reduce_sum(ptx: &mut String, num_warps: usize, label_prefix: &str) {
    let zero = format_ptx_float(0.0);
    let reg = if label_prefix.contains("MEAN") {
        "%f0"
    } else {
        "%f2"
    };

    ptx.push_str(&format!(
        "\x20   // Cross-warp sum reduction via shared memory ({label_prefix})\n\
         \x20   setp.eq.u32   %p2, %r6, 0;             // lane_id == 0?\n\
         \x20   @!%p2 bra     {label_prefix}_LOAD;\n\
         \x20   mul.wide.u32  %rd10, %r5, 4;           // warp_id * 4\n\
         \x20   mov.u64       %rd11, warp_scratch;\n\
         \x20   add.u64       %rd10, %rd11, %rd10;\n\
         \x20   st.shared.f32 [%rd10], {reg};\n\
         {label_prefix}_LOAD:\n\
         \x20   bar.sync      0;\n\
         \x20   mov.f32       {reg}, {zero};\n\
         \x20   setp.lt.u32   %p3, %r1, {num_warps};   // tid < num_warps?\n\
         \x20   @!%p3 bra     {label_prefix}_DONE;\n\
         \x20   mul.wide.u32  %rd10, %r1, 4;           // tid * 4\n\
         \x20   mov.u64       %rd11, warp_scratch;\n\
         \x20   add.u64       %rd10, %rd11, %rd10;\n\
         \x20   ld.shared.f32 {reg}, [%rd10];\n\
         {label_prefix}_DONE:\n"
    ));
    emit_warp_reduce_sum(ptx, reg);
    ptx.push_str(&format!(
        "\x20   // Broadcast to all threads via shared memory\n\
         \x20   setp.eq.u32   %p2, %r1, 0;             // tid == 0?\n\
         \x20   @!%p2 bra     BCAST_{label_prefix}_LOAD;\n\
         \x20   mov.u64       %rd11, warp_scratch;\n\
         \x20   st.shared.f32 [%rd11], {reg};\n\
         BCAST_{label_prefix}_LOAD:\n\
         \x20   bar.sync      0;\n\
         \x20   mov.u64       %rd11, warp_scratch;\n\
         \x20   ld.shared.f32 {reg}, [%rd11];\n\n"
    ));
}

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// Compute Instance Normalization on CPU for reference/testing.
///
/// `y = gamma_c * (x - mean) / sqrt(var + eps) + beta_c`
///
/// For each (batch, channel) pair, computes mean and variance over H*W
/// spatial elements, then normalizes with per-channel gamma and beta.
///
/// # Arguments
///
/// * `input` -- flattened NCHW input, length = N*C*H*W
/// * `gamma` -- per-channel scale, length = C
/// * `beta`  -- per-channel bias, length = C
/// * `n`     -- batch size
/// * `c`     -- number of channels
/// * `h`     -- spatial height
/// * `w`     -- spatial width
/// * `eps`   -- numerical stability epsilon
///
/// # Returns
///
/// Normalized output tensor, same shape as input.
pub fn instancenorm_reference(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    eps: f32,
) -> Vec<f32> {
    let spatial = h * w;
    let total = n * c * spatial;
    assert_eq!(input.len(), total, "input length must equal N*C*H*W");
    assert_eq!(gamma.len(), c, "gamma length must equal C");
    assert_eq!(beta.len(), c, "beta length must equal C");
    assert!(c > 0 && spatial > 0, "channels and spatial must be > 0");

    let mut output = vec![0.0f32; total];

    for batch in 0..n {
        for ch in 0..c {
            let base = batch * c * spatial + ch * spatial;

            // Compute mean over spatial
            let mut sum = 0.0f32;
            for i in 0..spatial {
                sum += input[base + i];
            }
            let mean = sum / spatial as f32;

            // Compute variance over spatial
            let mut var_sum = 0.0f32;
            for i in 0..spatial {
                let diff = input[base + i] - mean;
                var_sum += diff * diff;
            }
            let var = var_sum / spatial as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            // Normalize with per-channel gamma and beta
            for i in 0..spatial {
                let normalized = (input[base + i] - mean) * inv_std;
                output[base + i] = gamma[ch] * normalized + beta[ch];
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_instancenorm_tests.rs"]
mod ptx_instancenorm_tests;
