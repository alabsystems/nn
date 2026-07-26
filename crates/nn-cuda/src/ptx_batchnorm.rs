// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for Batch Normalization (BatchNorm).
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 BatchNorm
//! in inference mode using running statistics. BatchNorm is the normalization
//! layer used in CNNs (ResNet, VGG, EfficientNet) and many vision models.
//!
//! ## Algorithm (inference mode)
//!
//! BatchNorm inference: `y = weight * (x - running_mean) / sqrt(running_var + eps) + bias`
//!
//! Unlike LayerNorm, BatchNorm normalizes per-channel across the batch and
//! spatial dimensions. In inference mode, the running mean and variance are
//! pre-computed during training, so each element only needs the per-channel
//! affine transform -- no reduction is required.
//!
//! ## Data format
//!
//! NCHW (batch, channels, height, width). Each thread processes one element.
//! The channel index determines which running_mean, running_var, weight, and
//! bias values to use.
//!
//! ## Comparison with LayerNorm/RMSNorm
//!
//! | Property         | BatchNorm (inference)     | LayerNorm               | RMSNorm                    |
//! |------------------|--------------------------|-------------------------|----------------------------|
//! | Formula          | w*(x-mean)/std+b         | gamma*(x-mean)/std+beta | weight * x * rsqrt(rms+e)  |
//! | Stats source     | Pre-computed (running)   | Computed per-row        | Computed per-row           |
//! | Norm axis        | Per-channel (C)          | Last dim                | Last dim                   |
//! | Reduction needed | No (inference)           | Yes                     | Yes                        |
//! | Used in          | ResNet, VGG, EfficientNet| BERT, GPT-2, Whisper    | Llama, Qwen3, GLM          |
//!
//! ## Kernel interface
//!
//! Parameters (in generated kernel):
//! - `param_input`        -- pointer to input tensor (f32), NCHW
//! - `param_output`       -- pointer to output tensor (f32), NCHW
//! - `param_running_mean` -- pointer to running mean (f32, length = C)
//! - `param_running_var`  -- pointer to running variance (f32, length = C)
//! - `param_weight`       -- pointer to weight/gamma (f32, length = C)
//! - `param_bias`         -- pointer to bias/beta (f32, length = C)
//! - `param_num_channels` -- u32, number of channels (C)
//! - `param_spatial_size` -- u32, H*W (spatial elements per channel)
//! - `param_total`        -- u32, total elements (N*C*H*W)
//!
//! ## Thread block configuration
//!
//! Block: `(256, 1, 1)` -- standard elementwise block size.
//! Grid: `(ceil(total / 256), 1, 1)` -- one thread per element.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude, PtxCodegenError, DEFAULT_SM_TARGET};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Block size for BatchNorm elementwise kernel.
const BLOCK_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for PTX BatchNorm kernel generation.
///
/// BatchNorm (inference) normalizes per-channel using pre-computed running
/// statistics: `y = weight * (x - running_mean) / sqrt(running_var + eps) + bias`
///
/// Used in ResNet, VGG, EfficientNet, and other CNN architectures.
#[derive(Debug, Clone)]
pub struct PtxBatchNormConfig {
    /// Kernel function name in the PTX module.
    pub kernel_name: String,
    /// Number of channels (C dimension in NCHW).
    pub num_channels: usize,
    /// Epsilon for numerical stability in the rsqrt denominator.
    pub eps: f32,
    /// SM target for the PTX prelude (e.g., "sm_80").
    pub sm_target: String,
}

impl PtxBatchNormConfig {
    /// Create a BatchNorm config with default sm_80 target.
    pub fn new(kernel_name: &str, num_channels: usize, eps: f32) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
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

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PtxCodegenError> {
        if self.num_channels == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "num_channels must be > 0".into(),
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

    /// Block size for this kernel.
    #[must_use]
    pub fn block_size(&self) -> usize {
        BLOCK_SIZE
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Emit a complete PTX module for f32 BatchNorm (inference mode).
///
/// Generates raw PTX assembly implementing batch normalization with
/// pre-computed running statistics. Each thread normalizes one element
/// using the per-channel mean, variance, weight, and bias.
///
/// # Parameters (in generated kernel)
///
/// * `param_input`        -- pointer to input tensor (f32), NCHW
/// * `param_output`       -- pointer to output tensor (f32), NCHW
/// * `param_running_mean` -- pointer to running mean (f32, length = C)
/// * `param_running_var`  -- pointer to running variance (f32, length = C)
/// * `param_weight`       -- pointer to weight (f32, length = C)
/// * `param_bias`         -- pointer to bias (f32, length = C)
/// * `param_num_channels` -- u32, C
/// * `param_spatial_size` -- u32, H*W
/// * `param_total`        -- u32, total elements N*C*H*W
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_batchnorm::{emit_ptx_batchnorm, PtxBatchNormConfig};
/// let config = PtxBatchNormConfig::new("batchnorm_64", 64, 1e-5);
/// let ptx = emit_ptx_batchnorm(&config).unwrap();
/// assert!(ptx.contains(".entry batchnorm_64"));
/// assert!(ptx.contains("rsqrt.approx.f32"));
/// ```
pub fn emit_ptx_batchnorm(config: &PtxBatchNormConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let name = &config.kernel_name;
    let num_channels = config.num_channels;
    let eps = config.eps;
    let block_size = config.block_size();

    let eps_hex = format_ptx_float(eps);

    let mut ptx = String::with_capacity(8192);

    // -- Module header --
    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// BatchNorm f32 (inference): num_channels={num_channels}, eps={eps}, \
         block_size={block_size}\n\
         // NCHW format, elementwise with per-channel stats\n\n"
    ));

    // -- Kernel entry point --
    ptx.push_str(&format!(
        ".visible .entry {name}(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u64 param_running_mean,\n\
         \x20   .param .u64 param_running_var,\n\
         \x20   .param .u64 param_weight,\n\
         \x20   .param .u64 param_bias,\n\
         \x20   .param .u32 param_num_channels,\n\
         \x20   .param .u32 param_spatial_size,\n\
         \x20   .param .u32 param_total\n\
         )\n"
    ));

    ptx.push_str(&format!(".reqntid {block_size}\n{{\n"));

    // -- Register declarations --
    ptx.push_str(
        "\x20   // Register declarations\n\
         \x20   .reg .u32  %r<16>;\n\
         \x20   .reg .f32  %f<12>;\n\
         \x20   .reg .u64  %rd<10>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // -- Load parameters --
    ptx.push_str(
        "\x20   // Load kernel parameters\n\
         \x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u64  %rd2, [param_running_mean];\n\
         \x20   ld.param.u64  %rd3, [param_running_var];\n\
         \x20   ld.param.u64  %rd4, [param_weight];\n\
         \x20   ld.param.u64  %rd5, [param_bias];\n\
         \x20   ld.param.u32  %r0,  [param_num_channels];\n\
         \x20   ld.param.u32  %r1,  [param_spatial_size];\n\
         \x20   ld.param.u32  %r2,  [param_total];\n\n",
    );

    // -- Compute global thread index --
    ptx.push_str(&format!(
        "\x20   // Compute global index: idx = blockIdx.x * blockDim.x + threadIdx.x\n\
         \x20   mov.u32       %r3, %ctaid.x;          // blockIdx.x\n\
         \x20   mov.u32       %r4, %tid.x;            // threadIdx.x\n\
         \x20   mad.lo.u32    %r5, %r3, {block_size}, %r4; // idx = blockIdx.x * BLOCK + tid\n\n"
    ));

    // -- Bounds check: idx < total --
    ptx.push_str(
        "\x20   // Bounds check: idx < total\n\
         \x20   setp.ge.u32   %p0, %r5, %r2;          // idx >= total?\n\
         \x20   @%p0 bra      BN_EXIT;\n\n",
    );

    // -- Compute channel index: c = (idx / spatial_size) % num_channels --
    ptx.push_str(
        "\x20   // Compute channel index: c = (idx / spatial_size) % num_channels\n\
         \x20   div.u32       %r6, %r5, %r1;          // idx / spatial_size\n\
         \x20   rem.u32       %r7, %r6, %r0;          // (idx / spatial) % channels = c\n\n",
    );

    // -- Load per-channel parameters --
    ptx.push_str(
        "\x20   // Load per-channel running_mean[c], running_var[c], weight[c], bias[c]\n\
         \x20   mul.wide.u32  %rd6, %r7, 4;           // c * 4 (byte offset)\n\
         \x20   add.u64       %rd7, %rd2, %rd6;       // &running_mean[c]\n\
         \x20   ld.global.f32 %f0, [%rd7];            // mean_c\n\
         \x20   add.u64       %rd7, %rd3, %rd6;       // &running_var[c]\n\
         \x20   ld.global.f32 %f1, [%rd7];            // var_c\n\
         \x20   add.u64       %rd7, %rd4, %rd6;       // &weight[c]\n\
         \x20   ld.global.f32 %f2, [%rd7];            // weight_c\n\
         \x20   add.u64       %rd7, %rd5, %rd6;       // &bias[c]\n\
         \x20   ld.global.f32 %f3, [%rd7];            // bias_c\n\n",
    );

    // -- Load input[idx] --
    ptx.push_str(
        "\x20   // Load input[idx]\n\
         \x20   mul.wide.u32  %rd8, %r5, 4;           // idx * 4 (byte offset)\n\
         \x20   add.u64       %rd9, %rd0, %rd8;       // &input[idx]\n\
         \x20   ld.global.f32 %f4, [%rd9];            // x = input[idx]\n\n",
    );

    // -- Compute: y = weight * (x - mean) * rsqrt(var + eps) + bias --
    ptx.push_str(&format!(
        "\x20   // Normalize: y = weight * (x - mean) * rsqrt(var + eps) + bias\n\
         \x20   sub.f32       %f5, %f4, %f0;          // x - mean\n\
         \x20   add.f32       %f6, %f1, {eps_hex};     // var + eps\n\
         \x20   rsqrt.approx.f32 %f7, %f6;            // 1 / sqrt(var + eps)\n\
         \x20   mul.f32       %f8, %f5, %f7;          // (x - mean) * inv_std\n\
         \x20   fma.rn.f32    %f9, %f2, %f8, %f3;     // weight * normalized + bias\n\n"
    ));

    // -- Store output[idx] --
    ptx.push_str(
        "\x20   // Store output[idx]\n\
         \x20   add.u64       %rd9, %rd1, %rd8;       // &output[idx]\n\
         \x20   st.global.f32 [%rd9], %f9;            // output[idx] = y\n\n",
    );

    // -- Kernel exit --
    ptx.push_str(
        "BN_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Convenience: emit PTX BatchNorm with default sm_80 target.
pub fn emit_ptx_batchnorm_default(
    name: &str,
    num_channels: usize,
    eps: f32,
) -> Result<String, PtxCodegenError> {
    emit_ptx_batchnorm(&PtxBatchNormConfig::new(name, num_channels, eps))
}

/// Compute the grid and block dimensions for a PTX BatchNorm kernel.
///
/// Grid: `(ceil(total / BLOCK_SIZE), 1, 1)` -- one thread per element.
/// Block: `(BLOCK_SIZE, 1, 1)`.
///
/// # Returns
///
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_batchnorm_launch_config(total_elements: usize) -> ([usize; 3], [usize; 3]) {
    let grid_x = total_elements.div_ceil(BLOCK_SIZE);
    let grid = [grid_x, 1, 1];
    let block = [BLOCK_SIZE, 1, 1];
    (grid, block)
}

/// Generate PTX for BatchNorm with default settings.
///
/// Convenience wrapper around [`emit_ptx_batchnorm`] that uses
/// default settings (kernel name `"ptx_batchnorm_f32"`, sm_80 target).
pub fn generate_batchnorm_ptx(num_channels: usize) -> String {
    emit_ptx_batchnorm_default("ptx_batchnorm_f32", num_channels, 1e-5)
        .expect("BatchNorm PTX generation failed")
}

/// Compute BatchNorm on CPU for reference/testing (inference mode).
///
/// `y = weight * (x - running_mean) / sqrt(running_var + eps) + bias`
///
/// Operates on a full NCHW tensor flattened to a 1D slice.
///
/// # Arguments
///
/// * `input`        -- flattened NCHW input, length = N*C*H*W
/// * `output`       -- output buffer, same length as input
/// * `running_mean` -- per-channel running mean, length = C
/// * `running_var`  -- per-channel running variance, length = C
/// * `weight`       -- per-channel scale (gamma), length = C
/// * `bias`         -- per-channel bias (beta), length = C
/// * `channels`     -- number of channels C
/// * `spatial`      -- spatial size H*W
/// * `eps`          -- numerical stability epsilon
pub fn batchnorm_reference(
    input: &[f32],
    output: &mut [f32],
    running_mean: &[f32],
    running_var: &[f32],
    weight: &[f32],
    bias: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    assert_eq!(running_mean.len(), channels);
    assert_eq!(running_var.len(), channels);
    assert_eq!(weight.len(), channels);
    assert_eq!(bias.len(), channels);
    assert_eq!(input.len(), output.len());
    assert!(channels > 0 && spatial > 0);

    for (i, (x, y)) in input.iter().zip(output.iter_mut()).enumerate() {
        let c = (i / spatial) % channels;
        let inv_std = 1.0 / (running_var[c] + eps).sqrt();
        *y = weight[c] * (x - running_mean[c]) * inv_std + bias[c];
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_batchnorm_tests.rs"]
mod ptx_batchnorm_tests;
