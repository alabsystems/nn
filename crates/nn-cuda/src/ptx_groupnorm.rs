// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for Group Normalization (GroupNorm).
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 GroupNorm.
//! GroupNorm divides channels into groups and normalizes within each group
//! across channels-in-group and spatial dimensions. Used in object detection
//! (DETR), diffusion models (Stable Diffusion), and other architectures where
//! batch size is small or variable.
//!
//! ## Algorithm
//!
//! GroupNorm: `y_i = weight_c * (x_i - group_mean) / sqrt(group_var + eps) + bias_c`
//!
//! For each sample in the batch and each group:
//! 1. **Compute group mean** -- average over (channels_per_group * spatial) elements.
//! 2. **Compute group variance** -- mean((x - group_mean)^2).
//! 3. **Normalize + affine** -- per-channel weight and bias applied after normalization.
//!
//! ## Data format
//!
//! NCHW. Each thread block handles one (sample, group) pair. Threads cooperate
//! to reduce over channels_per_group * spatial_size elements.
//!
//! ## Comparison with other norms
//!
//! | Property         | GroupNorm                 | BatchNorm (inference) | LayerNorm               |
//! |------------------|--------------------------|----------------------|-------------------------|
//! | Norm axis        | Per-group (G channels)   | Per-channel (C)      | Last dim                |
//! | Stats source     | Computed per-group       | Pre-computed         | Computed per-row        |
//! | Reduction needed | Yes (within group)       | No                   | Yes (within row)        |
//! | Used in          | DETR, Stable Diffusion   | ResNet, VGG          | BERT, GPT-2, Whisper    |
//!
//! ## Kernel interface
//!
//! Parameters (in generated kernel):
//! - `param_input`    -- pointer to input tensor (f32), NCHW
//! - `param_output`   -- pointer to output tensor (f32), NCHW
//! - `param_weight`   -- pointer to weight (f32, length = C, per-channel)
//! - `param_bias`     -- pointer to bias (f32, length = C, per-channel)
//! - `param_group_size` -- u32, channels per group (C / num_groups)
//! - `param_spatial_size` -- u32, H * W
//! - `param_group_elems` -- u32, group_size * spatial_size (elements per group)
//!
//! ## Thread block configuration
//!
//! Block: `(block_size, 1, 1)` where block_size = min(group_elems rounded up, 256).
//! Grid: `(N * num_groups, 1, 1)` -- one block per (sample, group).

use crate::codegen_ptx::{format_ptx_float, ptx_prelude, PtxCodegenError, DEFAULT_SM_TARGET};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// NVIDIA warp size (32 threads).
const WARP_SIZE: usize = 32;

/// Maximum block size for GroupNorm (8 warps = 256 threads).
const MAX_BLOCK_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for PTX GroupNorm kernel generation.
///
/// GroupNorm normalizes within groups of channels:
/// `y_i = weight_c * (x_i - group_mean) / sqrt(group_var + eps) + bias_c`
///
/// Used in DETR, Stable Diffusion, and architectures with small batch sizes.
#[derive(Debug, Clone)]
pub struct PtxGroupNormConfig {
    /// Kernel function name in the PTX module.
    pub kernel_name: String,
    /// Number of groups to divide channels into.
    pub num_groups: usize,
    /// Number of channels (must be divisible by num_groups).
    pub num_channels: usize,
    /// Epsilon for numerical stability in the rsqrt denominator.
    pub eps: f32,
    /// SM target for the PTX prelude (e.g., "sm_80").
    pub sm_target: String,
}

impl PtxGroupNormConfig {
    /// Create a GroupNorm config with default sm_80 target.
    pub fn new(kernel_name: &str, num_groups: usize, num_channels: usize, eps: f32) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            num_groups,
            num_channels,
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

    /// Channels per group.
    #[must_use]
    pub fn channels_per_group(&self) -> usize {
        if self.num_groups == 0 {
            return 0;
        }
        self.num_channels / self.num_groups
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PtxCodegenError> {
        if self.num_groups == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "num_groups must be > 0".into(),
            ));
        }
        if self.num_channels == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "num_channels must be > 0".into(),
            ));
        }
        if !self.num_channels.is_multiple_of(self.num_groups) {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "num_channels ({}) must be divisible by num_groups ({})",
                self.num_channels, self.num_groups
            )));
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
    /// Based on channels_per_group * typical spatial size, but since spatial
    /// is runtime, we cap at MAX_BLOCK_SIZE. The kernel loops for larger groups.
    #[must_use]
    pub fn block_size(&self) -> usize {
        // Use channels_per_group as lower bound; real group_elems is runtime.
        let cpg = self.channels_per_group();
        let rounded = cpg.max(1).div_ceil(WARP_SIZE) * WARP_SIZE;
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

/// Emit a complete PTX module for f32 GroupNorm.
///
/// Generates raw PTX assembly implementing group normalization with
/// warp-level shuffle reduction. Each block processes one (sample, group)
/// pair, computing the group mean and variance via cooperative reduction,
/// then normalizing with per-channel weight and bias.
///
/// # Parameters (in generated kernel)
///
/// * `param_input`       -- pointer to input tensor (f32), NCHW
/// * `param_output`      -- pointer to output tensor (f32), NCHW
/// * `param_weight`      -- pointer to weight (f32, length = C)
/// * `param_bias`        -- pointer to bias (f32, length = C)
/// * `param_group_size`  -- u32, channels per group (C / num_groups)
/// * `param_spatial_size` -- u32, H * W
/// * `param_group_elems` -- u32, group_size * spatial_size
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_groupnorm::{emit_ptx_groupnorm, PtxGroupNormConfig};
/// let config = PtxGroupNormConfig::new("groupnorm_32_256", 32, 256, 1e-5);
/// let ptx = emit_ptx_groupnorm(&config).unwrap();
/// assert!(ptx.contains(".entry groupnorm_32_256"));
/// assert!(ptx.contains("rsqrt.approx.f32"));
/// ```
pub fn emit_ptx_groupnorm(config: &PtxGroupNormConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let name = &config.kernel_name;
    let num_groups = config.num_groups;
    let num_channels = config.num_channels;
    let cpg = config.channels_per_group();
    let eps = config.eps;
    let block_size = config.block_size();
    let num_warps = config.num_warps();
    let warp_only = config.is_warp_only();

    let zero = format_ptx_float(0.0);
    let eps_hex = format_ptx_float(eps);

    let mut ptx = String::with_capacity(12288);

    // -- Module header --
    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// GroupNorm f32: num_groups={num_groups}, num_channels={num_channels}, \
         channels_per_group={cpg}, eps={eps}, block_size={block_size}, warps={num_warps}\n\
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
         \x20   .param .u64 param_weight,\n\
         \x20   .param .u64 param_bias,\n\
         \x20   .param .u32 param_group_size,\n\
         \x20   .param .u32 param_spatial_size,\n\
         \x20   .param .u32 param_group_elems\n\
         )\n"
    ));

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
         \x20   ld.param.u64  %rd2, [param_weight];\n\
         \x20   ld.param.u64  %rd3, [param_bias];\n\
         \x20   ld.param.u32  %r0,  [param_group_size];\n\
         \x20   ld.param.u32  %r1,  [param_spatial_size];\n\
         \x20   ld.param.u32  %r2,  [param_group_elems];\n\n",
    );

    // -- Compute thread/block indices --
    // blockIdx.x = sample_idx * num_groups + group_idx
    ptx.push_str(
        "\x20   // Thread and block indices\n\
         \x20   mov.u32       %r3, %tid.x;            // tid = threadIdx.x\n\
         \x20   mov.u32       %r4, %ctaid.x;          // block_id = blockIdx.x\n\n",
    );

    // -- Compute base offset for this group --
    // base_offset = block_id * group_elems
    ptx.push_str(
        "\x20   // Compute base offset for this (sample, group) in NCHW\n\
         \x20   mul.lo.u32    %r5, %r4, %r2;          // block_id * group_elems\n\
         \x20   mul.wide.u32  %rd4, %r5, 4;           // byte offset\n\
         \x20   add.u64       %rd5, %rd0, %rd4;       // &input[base]\n\
         \x20   add.u64       %rd6, %rd1, %rd4;       // &output[base]\n\n",
    );

    // -- Warp/lane decomposition --
    ptx.push_str(
        "\x20   // Warp/lane decomposition\n\
         \x20   shr.u32       %r6, %r3, 5;            // warp_id = tid >> 5\n\
         \x20   and.b32       %r7, %r3, 31;           // lane_id = tid & 31\n\n",
    );

    // -- Compute reciprocal of group_elems for mean --
    ptx.push_str(
        "\x20   // Reciprocal of group_elems for mean computation\n\
         \x20   cvt.rn.f32.u32 %f16, %r2;             // (float)group_elems\n\
         \x20   rcp.approx.f32 %f17, %f16;            // 1.0 / group_elems\n\n",
    );

    // =================================================================
    // Phase 1: compute group mean
    // =================================================================
    ptx.push_str(&format!(
        "\x20   // ---- Phase 1: compute group mean ----\n\
         \x20   mov.f32       %f0, {zero};             // local_sum = 0.0\n\
         \x20   mov.u32       %r8, %r3;                // i = tid\n\
         GN_MEAN_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r8, %r2;           // i >= group_elems?\n\
         \x20   @%p0 bra      GN_MEAN_REDUCE;\n\
         \x20   // Load input[base + i], accumulate sum\n\
         \x20   mul.wide.u32  %rd7, %r8, 4;            // byte offset\n\
         \x20   add.u64       %rd8, %rd5, %rd7;        // &group_in[i]\n\
         \x20   ld.global.f32 %f1, [%rd8];             // val = group_in[i]\n\
         \x20   add.f32       %f0, %f0, %f1;           // local_sum += val\n\
         \x20   add.u32       %r8, %r8, {block_size};  // i += block_size\n\
         \x20   bra           GN_MEAN_LOOP;\n\
         GN_MEAN_REDUCE:\n\n"
    ));

    // Warp-level sum reduction for mean
    emit_warp_reduce_sum(&mut ptx, "%f0");

    // Cross-warp reduction if needed
    if !warp_only {
        emit_cross_warp_reduce_sum(&mut ptx, num_warps, "CROSS_GN_MEAN");
    }

    // Compute mean = sum / group_elems
    ptx.push_str(
        "\x20   // mean = sum / group_elems\n\
         \x20   mul.f32       %f0, %f0, %f17;          // %f0 = group_mean\n\n",
    );

    // =================================================================
    // Phase 2: compute group variance = mean((x - mean)^2)
    // =================================================================
    ptx.push_str(&format!(
        "\x20   // ---- Phase 2: compute group variance ----\n\
         \x20   mov.f32       %f2, {zero};             // local_var_sum = 0.0\n\
         \x20   mov.u32       %r8, %r3;                // i = tid\n\
         GN_VAR_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r8, %r2;           // i >= group_elems?\n\
         \x20   @%p0 bra      GN_VAR_REDUCE;\n\
         \x20   // Load input[base + i], compute (x - mean)^2\n\
         \x20   mul.wide.u32  %rd7, %r8, 4;            // byte offset\n\
         \x20   add.u64       %rd8, %rd5, %rd7;        // &group_in[i]\n\
         \x20   ld.global.f32 %f3, [%rd8];             // val = group_in[i]\n\
         \x20   sub.f32       %f4, %f3, %f0;           // diff = val - mean\n\
         \x20   mul.f32       %f5, %f4, %f4;           // diff^2\n\
         \x20   add.f32       %f2, %f2, %f5;           // local_var_sum += diff^2\n\
         \x20   add.u32       %r8, %r8, {block_size};  // i += block_size\n\
         \x20   bra           GN_VAR_LOOP;\n\
         GN_VAR_REDUCE:\n\n"
    ));

    // Warp-level sum reduction for variance
    emit_warp_reduce_sum(&mut ptx, "%f2");

    // Cross-warp reduction if needed
    if !warp_only {
        emit_cross_warp_reduce_sum(&mut ptx, num_warps, "CROSS_GN_VAR");
    }

    // Compute var = var_sum / group_elems, then rsqrt(var + eps)
    ptx.push_str(&format!(
        "\x20   // var = var_sum / group_elems\n\
         \x20   mul.f32       %f2, %f2, %f17;          // %f2 = group_variance\n\
         \x20   // inv_std = rsqrt(var + eps)\n\
         \x20   add.f32       %f6, %f2, {eps_hex};      // var + eps\n\
         \x20   rsqrt.approx.f32 %f7, %f6;             // %f7 = 1/sqrt(var+eps)\n\n"
    ));

    // =================================================================
    // Phase 3: normalize with per-channel weight and bias
    // y_i = weight_c * (x_i - mean) * inv_std + bias_c
    // =================================================================
    // The group_id is block_id % num_groups, but we pass group_size to
    // compute the channel index within the group:
    // local element i in group -> channel_in_group = i / spatial_size
    // global channel = group_id * group_size + channel_in_group
    // We compute group_id * group_size = (block_id % num_groups) * group_size
    // but it's easier to compute: channel = (base_elem / spatial) % total_channels
    // For NCHW: element at offset base+i has channel = (base+i)/spatial % C
    // We pass group_size and spatial_size to compute channel from local offset.
    ptx.push_str(&format!(
        "\x20   // ---- Phase 3: normalize + affine ----\n\
         \x20   // For channel lookup: global_channel = group_id * group_size + local_channel\n\
         \x20   // where local_channel = i / spatial_size, group_id is implicit from base\n\
         \x20   mov.u32       %r8, %r3;                // i = tid\n\
         GN_NORM_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r8, %r2;           // i >= group_elems?\n\
         \x20   @%p0 bra      GN_EXIT;\n\
         \x20   // Load input[base + i]\n\
         \x20   mul.wide.u32  %rd7, %r8, 4;            // byte offset\n\
         \x20   add.u64       %rd8, %rd5, %rd7;        // &group_in[i]\n\
         \x20   ld.global.f32 %f8, [%rd8];             // x_i\n\
         \x20   // Compute normalized = (x_i - mean) * inv_std\n\
         \x20   sub.f32       %f9, %f8, %f0;           // x_i - mean\n\
         \x20   mul.f32       %f10, %f9, %f7;          // (x_i - mean) * inv_std\n\
         \x20   // Compute channel index for weight/bias lookup\n\
         \x20   // global_elem = base + i, channel = (base+i) / spatial % total_channels\n\
         \x20   // But base = block_id * group_elems, so channel from local:\n\
         \x20   // local_channel = i / spatial_size\n\
         \x20   // global_channel = (block_id % num_groups_total) * group_size + local_channel\n\
         \x20   // Simpler: channel_in_group = i / spatial_size\n\
         \x20   div.u32       %r9, %r8, %r1;           // i / spatial_size = channel_in_group\n\
         \x20   // global element index = base + i\n\
         \x20   add.u32       %r10, %r5, %r8;          // global_idx = base + i\n\
         \x20   // global channel = global_idx / spatial_size\n\
         \x20   div.u32       %r11, %r10, %r1;         // global_channel\n\
         \x20   // Load weight[global_channel] and bias[global_channel]\n\
         \x20   mul.wide.u32  %rd9, %r11, 4;           // channel * 4 bytes\n\
         \x20   add.u64       %rd10, %rd2, %rd9;       // &weight[channel]\n\
         \x20   ld.global.f32 %f11, [%rd10];           // weight_c\n\
         \x20   add.u64       %rd11, %rd3, %rd9;       // &bias[channel]\n\
         \x20   ld.global.f32 %f12, [%rd11];           // bias_c\n\
         \x20   // y_i = weight_c * normalized + bias_c\n\
         \x20   fma.rn.f32    %f13, %f11, %f10, %f12;  // weight * norm + bias\n\
         \x20   // Store output[base + i]\n\
         \x20   add.u64       %rd12, %rd6, %rd7;       // &group_out[i]\n\
         \x20   st.global.f32 [%rd12], %f13;           // output = y_i\n\
         \x20   add.u32       %r8, %r8, {block_size};  // i += block_size\n\
         \x20   bra           GN_NORM_LOOP;\n\n"
    ));

    // -- Kernel exit --
    ptx.push_str(
        "GN_EXIT:\n\
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
         \x20   setp.eq.u32   %p2, %r7, 0;             // lane_id == 0?\n\
         \x20   @!%p2 bra     {label_prefix}_LOAD;\n\
         \x20   mul.wide.u32  %rd7, %r6, 4;            // warp_id * 4\n\
         \x20   mov.u64       %rd8, warp_scratch;\n\
         \x20   add.u64       %rd7, %rd8, %rd7;\n\
         \x20   st.shared.f32 [%rd7], {reg};\n\
         {label_prefix}_LOAD:\n\
         \x20   bar.sync      0;\n\
         \x20   mov.f32       {reg}, {zero};\n\
         \x20   setp.lt.u32   %p3, %r3, {num_warps};   // tid < num_warps?\n\
         \x20   @!%p3 bra     {label_prefix}_DONE;\n\
         \x20   mul.wide.u32  %rd7, %r3, 4;            // tid * 4\n\
         \x20   mov.u64       %rd8, warp_scratch;\n\
         \x20   add.u64       %rd7, %rd8, %rd7;\n\
         \x20   ld.shared.f32 {reg}, [%rd7];\n\
         {label_prefix}_DONE:\n"
    ));
    emit_warp_reduce_sum(ptx, reg);
    ptx.push_str(&format!(
        "\x20   // Broadcast to all threads via shared memory\n\
         \x20   setp.eq.u32   %p2, %r3, 0;             // tid == 0?\n\
         \x20   @!%p2 bra     BCAST_{label_prefix}_LOAD;\n\
         \x20   mov.u64       %rd8, warp_scratch;\n\
         \x20   st.shared.f32 [%rd8], {reg};\n\
         BCAST_{label_prefix}_LOAD:\n\
         \x20   bar.sync      0;\n\
         \x20   mov.u64       %rd8, warp_scratch;\n\
         \x20   ld.shared.f32 {reg}, [%rd8];\n\n"
    ));
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Convenience: emit PTX GroupNorm with default sm_80 target.
pub fn emit_ptx_groupnorm_default(
    name: &str,
    num_groups: usize,
    num_channels: usize,
    eps: f32,
) -> Result<String, PtxCodegenError> {
    emit_ptx_groupnorm(&PtxGroupNormConfig::new(
        name,
        num_groups,
        num_channels,
        eps,
    ))
}

/// Compute the grid and block dimensions for a PTX GroupNorm kernel.
///
/// Grid: `(batch_size * num_groups, 1, 1)` -- one block per (sample, group).
/// Block: `(block_size, 1, 1)` -- threads cooperate on one group.
///
/// # Returns
///
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_groupnorm_launch_config(
    batch_size: usize,
    num_groups: usize,
    num_channels: usize,
) -> ([usize; 3], [usize; 3]) {
    let config = PtxGroupNormConfig::new("_", num_groups, num_channels, 1e-5);
    let block_size = config.block_size();
    let grid = [batch_size * num_groups, 1, 1];
    let block = [block_size, 1, 1];
    (grid, block)
}

/// Generate PTX for GroupNorm with default settings.
///
/// Convenience wrapper around [`emit_ptx_groupnorm`] that uses
/// default settings (kernel name `"ptx_groupnorm_f32"`, sm_80 target).
pub fn generate_groupnorm_ptx(num_groups: usize, num_channels: usize) -> String {
    emit_ptx_groupnorm_default("ptx_groupnorm_f32", num_groups, num_channels, 1e-5)
        .expect("GroupNorm PTX generation failed")
}

/// Compute GroupNorm on CPU for reference/testing.
///
/// `y_i = weight_c * (x_i - group_mean) / sqrt(group_var + eps) + bias_c`
///
/// Operates on a full NCHW tensor flattened to a 1D slice.
///
/// # Arguments
///
/// * `input`   -- flattened NCHW input, length = N*C*H*W
/// * `output`  -- output buffer, same length as input
/// * `weight`  -- per-channel scale, length = C
/// * `bias`    -- per-channel bias, length = C
/// * `groups`  -- number of groups
/// * `channels` -- number of channels (must be divisible by groups)
/// * `spatial` -- spatial size H*W
/// * `eps`     -- numerical stability epsilon
pub fn groupnorm_reference(
    input: &[f32],
    output: &mut [f32],
    weight: &[f32],
    bias: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    assert_eq!(weight.len(), channels);
    assert_eq!(bias.len(), channels);
    assert_eq!(input.len(), output.len());
    assert!(channels > 0 && spatial > 0 && groups > 0);
    assert_eq!(channels % groups, 0);

    let cpg = channels / groups; // channels per group
    let group_elems = cpg * spatial;
    let batch_size = input.len() / (channels * spatial);

    for n in 0..batch_size {
        for g in 0..groups {
            // Compute group mean
            let base = n * channels * spatial + g * group_elems;
            let mut sum = 0.0f32;
            for i in 0..group_elems {
                sum += input[base + i];
            }
            let mean = sum / group_elems as f32;

            // Compute group variance
            let mut var_sum = 0.0f32;
            for i in 0..group_elems {
                let diff = input[base + i] - mean;
                var_sum += diff * diff;
            }
            let var = var_sum / group_elems as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            // Normalize with per-channel weight/bias
            for i in 0..group_elems {
                let channel_in_group = i / spatial;
                let global_channel = g * cpg + channel_in_group;
                let normalized = (input[base + i] - mean) * inv_std;
                output[base + i] = weight[global_channel] * normalized + bias[global_channel];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_groupnorm_tests.rs"]
mod ptx_groupnorm_tests;
