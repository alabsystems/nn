// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for fused residual operations.
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 fused
//! residual connection patterns common in transformer and ResNet architectures.
//! Unlike the CUDA C++ emission in [`ptx_emit`], this module emits PTX
//! assembly directly -- no `nvcc` compilation step needed.
//!
//! ## Supported Fusions
//!
//! | Pattern                  | Kernel               | Use case              |
//! |--------------------------|----------------------|-----------------------|
//! | `x + residual`           | Residual add         | Skip connections      |
//! | `ReLU(x + residual)`     | Residual add + ReLU  | ResNet blocks         |
//! | `LayerNorm(x + residual)`| Residual add + LN    | Transformer layers    |
//!
//! Fusing residual add with the subsequent activation or normalization
//! eliminates an intermediate global memory write-read round trip.
//!
//! ## Thread block configuration
//!
//! Residual add and residual add + ReLU use 1D grid-stride loops with
//! 256 threads per block. Residual add + LayerNorm uses one block per
//! row (same strategy as [`ptx_layernorm`]).

use crate::codegen_ptx::{format_ptx_float, ptx_prelude};
use crate::cuda_ffi::{CudaDim3, CudaLaunchConfig};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for residual elementwise kernels.
pub const RESIDUAL_BLOCK_SIZE: u32 = 256;

/// SM target for residual kernels (Volta+).
const SM_TARGET: &str = "sm_70";

/// NVIDIA warp size.
const WARP_SIZE: u32 = 32;

/// Maximum threads per block for LayerNorm fusion.
const MAX_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// Residual Add: out[i] = x[i] + residual[i]
// ---------------------------------------------------------------------------

/// Generate PTX for elementwise residual add: `out[i] = x[i] + residual[i]`.
///
/// # Arguments
///
/// * `n` - Total number of f32 elements.
///
/// # Kernel parameters (in generated PTX)
///
/// * `param_x`        -- pointer to x tensor (f32)
/// * `param_residual` -- pointer to residual tensor (f32)
/// * `param_output`   -- pointer to output tensor (f32)
/// * `param_n`        -- u32, total number of elements
pub fn generate_residual_add_ptx(n: u32) -> String {
    let block_size = RESIDUAL_BLOCK_SIZE;
    let mut ptx = String::with_capacity(2048);

    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// ResidualAdd f32: n={n}, block_size={block_size}\n\n"
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry ptx_residual_add_f32(\n\
         \x20   .param .u64 param_x,\n\
         \x20   .param .u64 param_residual,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_x];\n\
         \x20   ld.param.u64  %rd1, [param_residual];\n\
         \x20   ld.param.u64  %rd2, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    // Global index and grid stride
    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;   // global_idx\n\
         \x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;         // grid_stride\n\n",
    );

    // Grid-stride loop
    ptx.push_str(
        "RADD_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r0;\n\
         \x20   @%p0 bra      RADD_EXIT;\n\n",
    );

    // Load x[i] and residual[i]
    ptx.push_str(
        "\x20   mul.wide.u32  %rd3, %r4, 4;\n\
         \x20   add.u64       %rd4, %rd0, %rd3;\n\
         \x20   ld.global.f32 %f0, [%rd4];          // x[i]\n\
         \x20   add.u64       %rd5, %rd1, %rd3;\n\
         \x20   ld.global.f32 %f1, [%rd5];          // residual[i]\n\n",
    );

    // out[i] = x[i] + residual[i]
    ptx.push_str("\x20   add.f32       %f2, %f0, %f1;\n\n");

    // Store
    ptx.push_str(
        "\x20   add.u64       %rd6, %rd2, %rd3;\n\
         \x20   st.global.f32 [%rd6], %f2;\n\n",
    );

    // Advance
    ptx.push_str(
        "\x20   add.u32       %r4, %r4, %r6;       // idx += grid_stride\n\
         \x20   bra           RADD_LOOP;\n\n\
         RADD_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

/// CPU reference for residual add: `out[i] = x[i] + residual[i]`.
pub fn residual_add_reference(x: &[f32], residual: &[f32]) -> Vec<f32> {
    assert_eq!(
        x.len(),
        residual.len(),
        "x and residual must have same length"
    );
    x.iter()
        .zip(residual.iter())
        .map(|(&a, &b)| a + b)
        .collect()
}

/// Launch config for residual add kernel.
#[must_use]
pub fn residual_add_launch_config(n: usize) -> CudaLaunchConfig {
    CudaLaunchConfig::for_elementwise(n, RESIDUAL_BLOCK_SIZE)
}

// ---------------------------------------------------------------------------
// Residual Add + ReLU: out[i] = max(0, x[i] + residual[i])
// ---------------------------------------------------------------------------

/// Generate PTX for fused residual add + ReLU: `out[i] = max(0, x[i] + residual[i])`.
///
/// Saves one global memory round trip compared to separate add + ReLU.
///
/// # Arguments
///
/// * `n` - Total number of f32 elements.
///
/// # Kernel parameters (in generated PTX)
///
/// * `param_x`        -- pointer to x tensor (f32)
/// * `param_residual` -- pointer to residual tensor (f32)
/// * `param_output`   -- pointer to output tensor (f32)
/// * `param_n`        -- u32, total number of elements
pub fn generate_residual_add_relu_ptx(n: u32) -> String {
    let block_size = RESIDUAL_BLOCK_SIZE;
    let zero = format_ptx_float(0.0);
    let mut ptx = String::with_capacity(2048);

    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// ResidualAddReLU f32: n={n}, block_size={block_size}\n\n"
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry ptx_residual_add_relu_f32(\n\
         \x20   .param .u64 param_x,\n\
         \x20   .param .u64 param_residual,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_x];\n\
         \x20   ld.param.u64  %rd1, [param_residual];\n\
         \x20   ld.param.u64  %rd2, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    // Global index and grid stride
    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;   // global_idx\n\
         \x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;         // grid_stride\n\n",
    );

    // Grid-stride loop
    ptx.push_str(
        "RARELU_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r0;\n\
         \x20   @%p0 bra      RARELU_EXIT;\n\n",
    );

    // Load x[i] and residual[i]
    ptx.push_str(
        "\x20   mul.wide.u32  %rd3, %r4, 4;\n\
         \x20   add.u64       %rd4, %rd0, %rd3;\n\
         \x20   ld.global.f32 %f0, [%rd4];          // x[i]\n\
         \x20   add.u64       %rd5, %rd1, %rd3;\n\
         \x20   ld.global.f32 %f1, [%rd5];          // residual[i]\n\n",
    );

    // sum = x[i] + residual[i]; out = max(0, sum)
    ptx.push_str(&format!(
        "\x20   add.f32       %f2, %f0, %f1;        // sum = x + residual\n\
         \x20   max.f32       %f3, %f2, {zero};      // relu(sum)\n\n"
    ));

    // Store
    ptx.push_str(
        "\x20   add.u64       %rd6, %rd2, %rd3;\n\
         \x20   st.global.f32 [%rd6], %f3;\n\n",
    );

    // Advance
    ptx.push_str(
        "\x20   add.u32       %r4, %r4, %r6;       // idx += grid_stride\n\
         \x20   bra           RARELU_LOOP;\n\n\
         RARELU_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

/// CPU reference for residual add + ReLU: `out[i] = max(0, x[i] + residual[i])`.
pub fn residual_add_relu_reference(x: &[f32], residual: &[f32]) -> Vec<f32> {
    assert_eq!(
        x.len(),
        residual.len(),
        "x and residual must have same length"
    );
    x.iter()
        .zip(residual.iter())
        .map(|(&a, &b)| (a + b).max(0.0))
        .collect()
}

/// Launch config for residual add + ReLU kernel.
#[must_use]
pub fn residual_add_relu_launch_config(n: usize) -> CudaLaunchConfig {
    CudaLaunchConfig::for_elementwise(n, RESIDUAL_BLOCK_SIZE)
}

// ---------------------------------------------------------------------------
// Residual Add + LayerNorm: out = LayerNorm(x + residual)
// ---------------------------------------------------------------------------

/// Compute block size for LayerNorm-fused kernels.
///
/// Rounds `hidden` up to nearest multiple of warp size, capped at 256.
fn layernorm_block_size(hidden: u32) -> u32 {
    let rounded = hidden.div_ceil(WARP_SIZE) * WARP_SIZE;
    rounded.min(MAX_BLOCK_SIZE)
}

/// Number of warps in the LayerNorm fusion block.
fn num_warps(block_size: u32) -> u32 {
    block_size.div_ceil(WARP_SIZE)
}

/// Generate PTX for fused residual add + LayerNorm.
///
/// `out_i = gamma_i * ((x_i + residual_i) - mean) / sqrt(var + eps) + beta_i`
///
/// Three-phase warp-shuffle reduction (same strategy as `ptx_layernorm`):
/// 1. Compute sum of `(x + residual)` across the hidden dimension, derive mean.
/// 2. Compute variance of `(x + residual - mean)`.
/// 3. Normalize with gamma/beta affine transform.
///
/// Fusing the residual add into the LayerNorm avoids writing the intermediate
/// sum to global memory and reading it back.
///
/// # Arguments
///
/// * `n` - Total number of elements (num_rows * hidden).
/// * `hidden` - Size of the normalization dimension (innermost).
///
/// # Kernel parameters (in generated PTX)
///
/// * `param_x`        -- pointer to x tensor (f32)
/// * `param_residual` -- pointer to residual tensor (f32)
/// * `param_output`   -- pointer to output tensor (f32)
/// * `param_gamma`    -- pointer to gamma/scale (f32, length = hidden)
/// * `param_beta`     -- pointer to beta/bias (f32, length = hidden)
/// * `param_num_rows` -- u32, number of rows
/// * `param_hidden`   -- u32, hidden dimension size
pub fn generate_residual_add_layernorm_ptx(n: u32, hidden: u32) -> String {
    let block_size = layernorm_block_size(hidden);
    let nwarps = num_warps(block_size);
    let eps = format_ptx_float(1e-5);
    let zero = format_ptx_float(0.0);
    let mut ptx = String::with_capacity(8192);

    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// ResidualAddLayerNorm f32: n={n}, hidden={hidden}, \
         block_size={block_size}, warps={nwarps}\n\n"
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry ptx_residual_add_layernorm_f32(\n\
         \x20   .param .u64 param_x,\n\
         \x20   .param .u64 param_residual,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u64 param_gamma,\n\
         \x20   .param .u64 param_beta,\n\
         \x20   .param .u32 param_num_rows,\n\
         \x20   .param .u32 param_hidden\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Shared memory for cross-warp reduction (2 arrays: sum, sumsq)
    if nwarps > 1 {
        ptx.push_str(&format!(
            "\x20   .shared .align 4 .f32 smem_sum[{nwarps}];\n\
             \x20   .shared .align 4 .f32 smem_sumsq[{nwarps}];\n\n"
        ));
    }

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<24>;\n\
         \x20   .reg .f32  %f<16>;\n\
         \x20   .reg .u64  %rd<16>;\n\
         \x20   .reg .pred %p<8>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_x];\n\
         \x20   ld.param.u64  %rd1, [param_residual];\n\
         \x20   ld.param.u64  %rd2, [param_output];\n\
         \x20   ld.param.u64  %rd3, [param_gamma];\n\
         \x20   ld.param.u64  %rd4, [param_beta];\n\
         \x20   ld.param.u32  %r0,  [param_num_rows];\n\
         \x20   ld.param.u32  %r1,  [param_hidden];\n\n",
    );

    // Thread index, block index (one block per row)
    ptx.push_str(
        "\x20   mov.u32       %r2, %tid.x;           // tid\n\
         \x20   mov.u32       %r3, %ctaid.x;          // row index\n\n",
    );

    // Bounds check: row >= num_rows
    ptx.push_str(
        "\x20   setp.ge.u32   %p0, %r3, %r0;\n\
         \x20   @%p0 bra      RALN_EXIT;\n\n",
    );

    // Row base offset: row * hidden
    ptx.push_str("\x20   mul.lo.u32    %r4, %r3, %r1;         // row_base = row * hidden\n\n");

    // ---- Phase 1: Compute sum (for mean) ----
    ptx.push_str(&format!(
        "\x20   // Phase 1: sum for mean\n\
         \x20   mov.f32       %f0, {zero};            // thread_sum = 0\n\
         \x20   mov.u32       %r5, %r2;               // i = tid\n\
         RALN_SUM_LOOP:\n\
         \x20   setp.ge.u32   %p1, %r5, %r1;         // i >= hidden?\n\
         \x20   @%p1 bra      RALN_SUM_DONE;\n\
         \x20   add.u32       %r6, %r4, %r5;          // row_base + i\n\
         \x20   mul.wide.u32  %rd5, %r6, 4;\n\
         \x20   add.u64       %rd6, %rd0, %rd5;       // &x[row_base + i]\n\
         \x20   ld.global.f32 %f1, [%rd6];\n\
         \x20   add.u64       %rd7, %rd1, %rd5;       // &residual[row_base + i]\n\
         \x20   ld.global.f32 %f2, [%rd7];\n\
         \x20   add.f32       %f3, %f1, %f2;          // val = x + residual\n\
         \x20   add.f32       %f0, %f0, %f3;          // sum += val\n\
         \x20   add.u32       %r5, %r5, {block_size}; // i += block_size\n\
         \x20   bra           RALN_SUM_LOOP;\n\
         RALN_SUM_DONE:\n\n"
    ));

    // Warp shuffle reduction for sum
    emit_warp_reduce(&mut ptx, "%f0", "RALN_WSHUF_SUM");

    // Cross-warp reduction for sum (if needed)
    if nwarps > 1 {
        emit_cross_warp_reduce(&mut ptx, "%f0", "smem_sum", nwarps, "RALN_XWARP_SUM");
    }

    // mean = sum / hidden
    ptx.push_str(
        "\x20   cvt.rn.f32.u32 %f4, %r1;             // (float)hidden\n\
         \x20   div.approx.f32  %f5, %f0, %f4;        // mean = sum / hidden\n\n",
    );

    // ---- Phase 2: Compute variance ----
    ptx.push_str(&format!(
        "\x20   // Phase 2: variance\n\
         \x20   mov.f32       %f6, {zero};            // thread_sumsq = 0\n\
         \x20   mov.u32       %r5, %r2;               // i = tid\n\
         RALN_VAR_LOOP:\n\
         \x20   setp.ge.u32   %p2, %r5, %r1;         // i >= hidden?\n\
         \x20   @%p2 bra      RALN_VAR_DONE;\n\
         \x20   add.u32       %r6, %r4, %r5;          // row_base + i\n\
         \x20   mul.wide.u32  %rd5, %r6, 4;\n\
         \x20   add.u64       %rd6, %rd0, %rd5;\n\
         \x20   ld.global.f32 %f1, [%rd6];\n\
         \x20   add.u64       %rd7, %rd1, %rd5;\n\
         \x20   ld.global.f32 %f2, [%rd7];\n\
         \x20   add.f32       %f3, %f1, %f2;          // val = x + residual\n\
         \x20   sub.f32       %f7, %f3, %f5;          // diff = val - mean\n\
         \x20   fma.rn.f32    %f6, %f7, %f7, %f6;     // sumsq += diff*diff\n\
         \x20   add.u32       %r5, %r5, {block_size}; // i += block_size\n\
         \x20   bra           RALN_VAR_LOOP;\n\
         RALN_VAR_DONE:\n\n"
    ));

    // Warp shuffle reduction for sumsq
    emit_warp_reduce(&mut ptx, "%f6", "RALN_WSHUF_VAR");

    // Cross-warp reduction for sumsq (if needed)
    if nwarps > 1 {
        emit_cross_warp_reduce(&mut ptx, "%f6", "smem_sumsq", nwarps, "RALN_XWARP_VAR");
    }

    // inv_std = rsqrt(var / hidden + eps)
    ptx.push_str(&format!(
        "\x20   div.approx.f32  %f8, %f6, %f4;        // var = sumsq / hidden\n\
         \x20   add.f32         %f9, %f8, {eps};       // var + eps\n\
         \x20   rsqrt.approx.f32 %f10, %f9;            // inv_std = rsqrt(var+eps)\n\n"
    ));

    // ---- Phase 3: Normalize + affine ----
    ptx.push_str(&format!(
        "\x20   // Phase 3: normalize + affine\n\
         \x20   mov.u32       %r5, %r2;               // i = tid\n\
         RALN_NORM_LOOP:\n\
         \x20   setp.ge.u32   %p3, %r5, %r1;         // i >= hidden?\n\
         \x20   @%p3 bra      RALN_NORM_DONE;\n\
         \x20   add.u32       %r6, %r4, %r5;          // row_base + i\n\
         \x20   mul.wide.u32  %rd5, %r6, 4;\n\
         \x20   add.u64       %rd6, %rd0, %rd5;\n\
         \x20   ld.global.f32 %f1, [%rd6];\n\
         \x20   add.u64       %rd7, %rd1, %rd5;\n\
         \x20   ld.global.f32 %f2, [%rd7];\n\
         \x20   add.f32       %f3, %f1, %f2;          // val = x + residual\n\
         \x20   sub.f32       %f11, %f3, %f5;          // normed = val - mean\n\
         \x20   mul.f32       %f11, %f11, %f10;        // normed *= inv_std\n\
         \x20   // Load gamma[i] and beta[i]\n\
         \x20   mul.wide.u32  %rd8, %r5, 4;\n\
         \x20   add.u64       %rd9, %rd3, %rd8;       // &gamma[i]\n\
         \x20   ld.global.f32 %f12, [%rd9];\n\
         \x20   add.u64       %rd10, %rd4, %rd8;      // &beta[i]\n\
         \x20   ld.global.f32 %f13, [%rd10];\n\
         \x20   fma.rn.f32    %f14, %f12, %f11, %f13;  // gamma * normed + beta\n\
         \x20   // Store output[row_base + i]\n\
         \x20   add.u64       %rd11, %rd2, %rd5;\n\
         \x20   st.global.f32 [%rd11], %f14;\n\
         \x20   add.u32       %r5, %r5, {block_size}; // i += block_size\n\
         \x20   bra           RALN_NORM_LOOP;\n\
         RALN_NORM_DONE:\n\n"
    ));

    ptx.push_str(
        "RALN_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

/// Emit warp-level shuffle-down reduction for a float register.
///
/// Produces 5 iterations of `shfl.down.sync` with offsets 16, 8, 4, 2, 1.
fn emit_warp_reduce(ptx: &mut String, reg: &str, _label_prefix: &str) {
    ptx.push_str(&format!(
        "\x20   // Warp-level shuffle reduction for {reg}\n"
    ));
    for offset in [16, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "\x20   shfl.sync.down.b32 %f15, {reg}, {offset}, 31, 0xFFFFFFFF;\n\
             \x20   add.f32       {reg}, {reg}, %f15;\n"
        ));
    }
    ptx.push('\n');
}

/// Emit cross-warp reduction using shared memory.
///
/// After warp-level reduction, lane 0 of each warp writes to shared memory.
/// Thread 0 reads all warp results and reduces. Result broadcast via shared mem.
fn emit_cross_warp_reduce(
    ptx: &mut String,
    reg: &str,
    smem_name: &str,
    nwarps: u32,
    label_prefix: &str,
) {
    ptx.push_str(&format!(
        "\x20   // Cross-warp reduction: {smem_name}\n\
         \x20   mov.u32       %r7, %r2;              // tid\n\
         \x20   and.b32       %r8, %r7, 31;          // lane_id = tid & 31\n\
         \x20   shr.u32       %r9, %r7, 5;           // warp_id = tid >> 5\n\
         \x20   setp.ne.u32   %p4, %r8, 0;           // lane_id != 0?\n\
         \x20   @%p4 bra      {label_prefix}_SKIP_WRITE;\n\
         \x20   mul.lo.u32    %r10, %r9, 4;          // warp_id * sizeof(f32)\n\
         \x20   mov.u32       %r11, {smem_name};\n\
         \x20   add.u32       %r12, %r11, %r10;\n\
         \x20   st.shared.f32 [{smem_name} + %r10], {reg};\n\
         {label_prefix}_SKIP_WRITE:\n\
         \x20   bar.sync      0;\n"
    ));

    // Thread 0 reads all warp results
    ptx.push_str(&format!(
        "\x20   setp.ne.u32   %p5, %r7, 0;\n\
         \x20   @%p5 bra      {label_prefix}_BCAST;\n\
         \x20   mov.f32       {reg}, 0f00000000;     // reset\n"
    ));

    for w in 0..nwarps {
        let offset = w * 4;
        ptx.push_str(&format!(
            "\x20   ld.shared.f32 %f15, [{smem_name} + {offset}];\n\
             \x20   add.f32       {reg}, {reg}, %f15;\n"
        ));
    }

    // Broadcast result to all threads via shared memory
    ptx.push_str(&format!(
        "\x20   st.shared.f32 [{smem_name}], {reg};\n\
         {label_prefix}_BCAST:\n\
         \x20   bar.sync      0;\n\
         \x20   ld.shared.f32 {reg}, [{smem_name}];\n\n"
    ));
}

/// CPU reference for fused residual add + LayerNorm.
///
/// `out_i = gamma_i * ((x_i + residual_i) - mean) / sqrt(var + eps) + beta_i`
///
/// Processes `num_rows` rows of `hidden` elements each.
pub fn residual_add_layernorm_reference(
    x: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: &[f32],
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let n = x.len();
    assert_eq!(n, residual.len(), "x and residual must have same length");
    assert_eq!(gamma.len(), hidden, "gamma length must equal hidden");
    assert_eq!(beta.len(), hidden, "beta length must equal hidden");
    assert_eq!(n % hidden, 0, "total elements must be divisible by hidden");

    let num_rows = n / hidden;
    let mut output = vec![0.0f32; n];

    for row in 0..num_rows {
        let base = row * hidden;
        let row_slice_x = &x[base..base + hidden];
        let row_slice_r = &residual[base..base + hidden];

        // Compute sum (for mean)
        let sum: f32 = row_slice_x
            .iter()
            .zip(row_slice_r.iter())
            .map(|(&xi, &ri)| xi + ri)
            .sum();
        let mean = sum / hidden as f32;

        // Compute variance
        let var: f32 = row_slice_x
            .iter()
            .zip(row_slice_r.iter())
            .map(|(&xi, &ri)| {
                let val = xi + ri - mean;
                val * val
            })
            .sum::<f32>()
            / hidden as f32;

        let inv_std = 1.0 / (var + eps).sqrt();

        // Normalize + affine
        for i in 0..hidden {
            let val = row_slice_x[i] + row_slice_r[i];
            output[base + i] = gamma[i] * (val - mean) * inv_std + beta[i];
        }
    }

    output
}

/// Launch config for fused residual add + LayerNorm.
///
/// One block per row, block size = min(ceil_to_warp(hidden), 256).
#[must_use]
pub fn residual_add_layernorm_launch_config(num_rows: usize, hidden: usize) -> CudaLaunchConfig {
    let block_size = layernorm_block_size(hidden as u32);
    let grid_x = num_rows.min(u32::MAX as usize) as u32;
    CudaLaunchConfig {
        grid: CudaDim3::d1(grid_x),
        block: CudaDim3::d1(block_size),
        shared_mem_bytes: if num_warps(block_size) > 1 {
            // Two f32 arrays of nwarps each for sum and sumsq
            num_warps(block_size) * 4 * 2
        } else {
            0
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Residual Add ----

    #[test]
    fn test_residual_add_ptx_contains_version_and_target() {
        let ptx = generate_residual_add_ptx(1024);
        assert!(ptx.contains(".version 6.5"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".address_size 64"));
    }

    #[test]
    fn test_residual_add_ptx_contains_entry_point() {
        let ptx = generate_residual_add_ptx(1024);
        assert!(ptx.contains(".visible .entry ptx_residual_add_f32"));
    }

    #[test]
    fn test_residual_add_ptx_contains_params() {
        let ptx = generate_residual_add_ptx(1024);
        assert!(ptx.contains("param_x"));
        assert!(ptx.contains("param_residual"));
        assert!(ptx.contains("param_output"));
        assert!(ptx.contains("param_n"));
    }

    #[test]
    fn test_residual_add_ptx_contains_add_instruction() {
        let ptx = generate_residual_add_ptx(1024);
        assert!(ptx.contains("add.f32"));
    }

    #[test]
    fn test_residual_add_ptx_has_grid_stride_loop() {
        let ptx = generate_residual_add_ptx(1024);
        assert!(ptx.contains("RADD_LOOP:"));
        assert!(ptx.contains("RADD_EXIT:"));
        assert!(ptx.contains("bra"));
    }

    #[test]
    fn test_residual_add_ptx_is_pure_ptx() {
        let ptx = generate_residual_add_ptx(256);
        assert!(!ptx.contains("__global__"));
        assert!(!ptx.contains("#include"));
        assert!(!ptx.contains("__syncthreads"));
    }

    #[test]
    fn test_residual_add_reference_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let r = vec![0.5, 0.5, 0.5, 0.5];
        let out = residual_add_reference(&x, &r);
        assert_eq!(out, vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_residual_add_reference_zeros() {
        let x = vec![1.0, -1.0, 0.0];
        let r = vec![0.0, 0.0, 0.0];
        let out = residual_add_reference(&x, &r);
        assert_eq!(out, vec![1.0, -1.0, 0.0]);
    }

    #[test]
    fn test_residual_add_reference_negatives() {
        let x = vec![-1.0, -2.0];
        let r = vec![-3.0, 4.0];
        let out = residual_add_reference(&x, &r);
        assert_eq!(out, vec![-4.0, 2.0]);
    }

    #[test]
    fn test_residual_add_launch_config() {
        let cfg = residual_add_launch_config(1024);
        assert_eq!(cfg.block.x, 256);
        assert_eq!(cfg.grid.x, 4); // ceil(1024/256)
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    // ---- Residual Add + ReLU ----

    #[test]
    fn test_residual_add_relu_ptx_contains_entry_point() {
        let ptx = generate_residual_add_relu_ptx(1024);
        assert!(ptx.contains(".visible .entry ptx_residual_add_relu_f32"));
    }

    #[test]
    fn test_residual_add_relu_ptx_contains_max_instruction() {
        let ptx = generate_residual_add_relu_ptx(1024);
        assert!(ptx.contains("max.f32"), "fused ReLU must use max.f32");
    }

    #[test]
    fn test_residual_add_relu_ptx_contains_add() {
        let ptx = generate_residual_add_relu_ptx(1024);
        assert!(ptx.contains("add.f32"));
    }

    #[test]
    fn test_residual_add_relu_ptx_has_grid_stride_loop() {
        let ptx = generate_residual_add_relu_ptx(1024);
        assert!(ptx.contains("RARELU_LOOP:"));
        assert!(ptx.contains("RARELU_EXIT:"));
    }

    #[test]
    fn test_residual_add_relu_reference_basic() {
        let x = vec![1.0, -2.0, 3.0, -4.0];
        let r = vec![0.5, 0.5, -4.0, 5.0];
        let out = residual_add_relu_reference(&x, &r);
        assert_eq!(out, vec![1.5, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_residual_add_relu_reference_all_negative() {
        let x = vec![-10.0, -5.0];
        let r = vec![-1.0, -2.0];
        let out = residual_add_relu_reference(&x, &r);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn test_residual_add_relu_reference_all_positive() {
        let x = vec![1.0, 2.0, 3.0];
        let r = vec![1.0, 2.0, 3.0];
        let out = residual_add_relu_reference(&x, &r);
        assert_eq!(out, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_residual_add_relu_launch_config() {
        let cfg = residual_add_relu_launch_config(512);
        assert_eq!(cfg.block.x, 256);
        assert_eq!(cfg.grid.x, 2);
    }

    // ---- Residual Add + LayerNorm ----

    #[test]
    fn test_residual_add_layernorm_ptx_contains_entry_point() {
        let ptx = generate_residual_add_layernorm_ptx(512, 64);
        assert!(ptx.contains(".visible .entry ptx_residual_add_layernorm_f32"));
    }

    #[test]
    fn test_residual_add_layernorm_ptx_contains_phases() {
        let ptx = generate_residual_add_layernorm_ptx(1024, 128);
        assert!(ptx.contains("Phase 1"), "must have mean phase");
        assert!(ptx.contains("Phase 2"), "must have variance phase");
        assert!(ptx.contains("Phase 3"), "must have normalize phase");
    }

    #[test]
    fn test_residual_add_layernorm_ptx_contains_rsqrt() {
        let ptx = generate_residual_add_layernorm_ptx(512, 64);
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn test_residual_add_layernorm_ptx_contains_fma() {
        let ptx = generate_residual_add_layernorm_ptx(512, 64);
        assert!(ptx.contains("fma.rn.f32"));
    }

    #[test]
    fn test_residual_add_layernorm_ptx_contains_shuffle() {
        let ptx = generate_residual_add_layernorm_ptx(512, 64);
        assert!(ptx.contains("shfl.sync.down.b32"));
    }

    #[test]
    fn test_residual_add_layernorm_ptx_small_hidden_no_shared_mem() {
        // hidden=16 < 32, single warp: no shared memory needed
        let ptx = generate_residual_add_layernorm_ptx(256, 16);
        assert!(
            !ptx.contains("smem_sum"),
            "single warp should not use shared memory"
        );
    }

    #[test]
    fn test_residual_add_layernorm_ptx_large_hidden_uses_shared_mem() {
        // hidden=128 > 32, multi-warp: needs shared memory
        let ptx = generate_residual_add_layernorm_ptx(512, 128);
        assert!(
            ptx.contains("smem_sum"),
            "multi-warp must use shared memory"
        );
        assert!(
            ptx.contains("smem_sumsq"),
            "multi-warp must use shared memory for sumsq"
        );
        assert!(ptx.contains("bar.sync"), "multi-warp needs barriers");
    }

    #[test]
    fn test_residual_add_layernorm_ptx_contains_gamma_beta_loads() {
        let ptx = generate_residual_add_layernorm_ptx(512, 64);
        assert!(ptx.contains("param_gamma"));
        assert!(ptx.contains("param_beta"));
    }

    #[test]
    fn test_residual_add_layernorm_reference_identity_transform() {
        // gamma=1, beta=0 => pure normalization
        let hidden = 4;
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let r = vec![0.0, 0.0, 0.0, 0.0];
        let gamma = vec![1.0; hidden];
        let beta = vec![0.0; hidden];
        let out = residual_add_layernorm_reference(&x, &r, &gamma, &beta, hidden, 1e-5);

        // mean = 2.5, var = 1.25, inv_std = 1/sqrt(1.25 + 1e-5) ~ 0.89442
        let mean: f32 = 2.5;
        let var: f32 = 1.25;
        let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
        for i in 0..hidden {
            let expected = (x[i] - mean) * inv_std;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "element {i}: expected {expected}, got {}",
                out[i]
            );
        }
    }

    #[test]
    fn test_residual_add_layernorm_reference_with_residual() {
        let hidden = 4;
        let x = vec![1.0, 0.0, -1.0, 2.0];
        let r = vec![0.5, 0.5, 0.5, 0.5];
        let gamma = vec![2.0; hidden];
        let beta = vec![0.1; hidden];
        let out = residual_add_layernorm_reference(&x, &r, &gamma, &beta, hidden, 1e-5);

        // sum = (1.5 + 0.5 + -0.5 + 2.5) = 4.0, mean = 1.0
        let vals = [1.5_f32, 0.5, -0.5, 2.5];
        let mean: f32 = 1.0;
        let var: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
        for i in 0..hidden {
            let expected = 2.0 * (vals[i] - mean) * inv_std + 0.1;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "element {i}: expected {expected}, got {}",
                out[i]
            );
        }
    }

    #[test]
    fn test_residual_add_layernorm_reference_multi_row() {
        let hidden = 2;
        let x = vec![1.0, 3.0, 2.0, 4.0];
        let r = vec![0.0, 0.0, 0.0, 0.0];
        let gamma = vec![1.0; hidden];
        let beta = vec![0.0; hidden];
        let out = residual_add_layernorm_reference(&x, &r, &gamma, &beta, hidden, 1e-5);

        // Row 0: vals=[1,3], mean=2, var=1, inv_std=1/sqrt(1+eps)
        // Row 1: vals=[2,4], mean=3, var=1, inv_std=1/sqrt(1+eps)
        let inv_std = 1.0 / (1.0_f32 + 1e-5).sqrt();
        assert!((out[0] - -inv_std).abs() < 1e-5);
        assert!((out[1] - (1.0 * inv_std)).abs() < 1e-5);
        assert!((out[2] - -inv_std).abs() < 1e-5);
        assert!((out[3] - (1.0 * inv_std)).abs() < 1e-5);
    }

    #[test]
    fn test_residual_add_layernorm_launch_config_small() {
        let cfg = residual_add_layernorm_launch_config(8, 16);
        assert_eq!(cfg.grid.x, 8);
        // hidden=16 < 32, rounds up to 32
        assert_eq!(cfg.block.x, 32);
        // Single warp: no shared mem
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn test_residual_add_layernorm_launch_config_large() {
        let cfg = residual_add_layernorm_launch_config(32, 768);
        assert_eq!(cfg.grid.x, 32);
        // hidden=768: rounded to 768 (already multiple of 32), capped at 256
        assert_eq!(cfg.block.x, 256);
        // 256/32 = 8 warps, shared_mem = 8 * 4 * 2 = 64
        assert_eq!(cfg.shared_mem_bytes, 64);
    }

    // ---- Different n values produce different PTX comments ----

    #[test]
    fn test_residual_add_different_n_produces_different_ptx() {
        let ptx_a = generate_residual_add_ptx(256);
        let ptx_b = generate_residual_add_ptx(1024);
        assert_ne!(ptx_a, ptx_b);
    }

    #[test]
    fn test_residual_add_relu_different_n_produces_different_ptx() {
        let ptx_a = generate_residual_add_relu_ptx(256);
        let ptx_b = generate_residual_add_relu_ptx(1024);
        assert_ne!(ptx_a, ptx_b);
    }

    // ---- layernorm_block_size helper ----

    #[test]
    fn test_layernorm_block_size_small() {
        assert_eq!(layernorm_block_size(16), 32); // rounds up to warp
        assert_eq!(layernorm_block_size(32), 32); // exact warp
        assert_eq!(layernorm_block_size(33), 64); // rounds up
    }

    #[test]
    fn test_layernorm_block_size_capped() {
        assert_eq!(layernorm_block_size(256), 256);
        assert_eq!(layernorm_block_size(512), 256); // capped
        assert_eq!(layernorm_block_size(768), 256); // capped
    }
}
