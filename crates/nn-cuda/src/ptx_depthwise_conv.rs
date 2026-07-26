// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for depthwise 2D convolution (NCHW layout).
//!
//! Depthwise convolution applies a separate filter per input channel
//! (groups = channels). Each output channel depends only on the
//! corresponding input channel. This is the spatial filtering step
//! in depthwise-separable convolutions (MobileNet, EfficientNet).
//!
//! ## Layout
//!
//! - Input:  `[N, C, H_in, W_in]` (NCHW)
//! - Weight: `[C, 1, kH, kW]` (one filter per channel)
//! - Bias:   `[C]` (optional)
//! - Output: `[N, C, H_out, W_out]` (NCHW)
//!
//! ## Output size formula
//!
//! `out = (input_size + 2*padding - kernel_size) / stride + 1`
//!
//! ## Thread block configuration
//!
//! Default: 256 threads (1D block). Each thread produces one output element.
//! Grid: `(ceil(total_output_elements / block_size), 1, 1)`.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude, PtxCodegenError, DEFAULT_SM_TARGET};
use crate::cuda_ffi::{CudaDim3, CudaLaunchConfig};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for depthwise conv2d kernels.
pub const PTX_DEPTHWISE_CONV2D_BLOCK_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for PTX depthwise conv2d kernel generation.
#[derive(Debug, Clone)]
pub struct PtxDepthwiseConv2dConfig {
    /// Kernel function name in the PTX module.
    pub kernel_name: String,
    /// Number of channels (= groups for depthwise).
    pub channels: usize,
    /// Convolution kernel height.
    pub kernel_h: usize,
    /// Convolution kernel width.
    pub kernel_w: usize,
    /// Stride height.
    pub stride_h: usize,
    /// Stride width.
    pub stride_w: usize,
    /// Padding height (symmetric).
    pub padding_h: usize,
    /// Padding width (symmetric).
    pub padding_w: usize,
    /// Whether to add bias.
    pub use_bias: bool,
    /// Thread block size (default: 256).
    pub block_size: usize,
    /// SM target for the PTX prelude.
    pub sm_target: String,
}

impl PtxDepthwiseConv2dConfig {
    /// Create a config with default stride=1, pad=0, no bias.
    pub fn new(kernel_name: &str, channels: usize, kernel_h: usize, kernel_w: usize) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            channels,
            kernel_h,
            kernel_w,
            stride_h: 1,
            stride_w: 1,
            padding_h: 0,
            padding_w: 0,
            use_bias: false,
            block_size: PTX_DEPTHWISE_CONV2D_BLOCK_SIZE,
            sm_target: DEFAULT_SM_TARGET.to_string(),
        }
    }

    /// Set stride.
    #[must_use]
    pub fn with_stride(mut self, stride_h: usize, stride_w: usize) -> Self {
        self.stride_h = stride_h;
        self.stride_w = stride_w;
        self
    }

    /// Set padding.
    #[must_use]
    pub fn with_padding(mut self, padding_h: usize, padding_w: usize) -> Self {
        self.padding_h = padding_h;
        self.padding_w = padding_w;
        self
    }

    /// Enable bias.
    #[must_use]
    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Set block size.
    #[must_use]
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Set SM target.
    #[must_use]
    pub fn with_sm_target(mut self, target: &str) -> Self {
        self.sm_target = target.to_string();
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PtxCodegenError> {
        if self.kernel_name.is_empty() {
            return Err(PtxCodegenError::InvalidParameter(
                "kernel_name must not be empty".into(),
            ));
        }
        if self.channels == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "channels must be > 0".into(),
            ));
        }
        if self.kernel_h == 0 || self.kernel_w == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "kernel dimensions must be > 0".into(),
            ));
        }
        if self.stride_h == 0 || self.stride_w == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "stride dimensions must be > 0".into(),
            ));
        }
        if self.block_size == 0 || self.block_size > 1024 {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "block_size must be 1..=1024, got {}",
                self.block_size,
            )));
        }
        Ok(())
    }
}

impl Default for PtxDepthwiseConv2dConfig {
    fn default() -> Self {
        Self::new("ptx_depthwise_conv2d_f32", 32, 3, 3)
    }
}

// ---------------------------------------------------------------------------
// Output size
// ---------------------------------------------------------------------------

/// Compute the output size for a depthwise conv2d dimension.
///
/// Same formula as standard conv2d: `(in + 2*pad - kernel) / stride + 1`.
#[must_use]
pub fn depthwise_conv2d_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Option<usize> {
    let padded = input_size.checked_add(2 * padding)?;
    if padded < kernel_size {
        return None;
    }
    Some((padded - kernel_size) / stride + 1)
}

// ---------------------------------------------------------------------------
// Reference implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation for depthwise conv2d (NCHW layout).
///
/// Weight layout: `[C, 1, kH, kW]` — one kH*kW filter per channel.
pub fn depthwise_conv2d_reference(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &PtxDepthwiseConv2dConfig,
    batch_size: usize,
    h_in: usize,
    w_in: usize,
) -> Vec<f32> {
    let c = config.channels;
    let kh = config.kernel_h;
    let kw = config.kernel_w;
    let sh = config.stride_h;
    let sw = config.stride_w;
    let ph = config.padding_h;
    let pw = config.padding_w;

    let h_out = depthwise_conv2d_output_size(h_in, kh, sh, ph).expect("invalid output height");
    let w_out = depthwise_conv2d_output_size(w_in, kw, sw, pw).expect("invalid output width");

    assert_eq!(input.len(), batch_size * c * h_in * w_in);
    assert_eq!(weight.len(), c * kh * kw);
    if let Some(b) = bias {
        assert_eq!(b.len(), c);
    }

    let mut output = vec![0.0f32; batch_size * c * h_out * w_out];

    for n in 0..batch_size {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0.0f32;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = (oh * sh + ki) as isize - ph as isize;
                            let iw = (ow * sw + kj) as isize - pw as isize;
                            if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                let in_idx =
                                    ((n * c + ch) * h_in + ih as usize) * w_in + iw as usize;
                                let w_idx = (ch * kh + ki) * kw + kj;
                                acc += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    if let Some(b) = bias {
                        acc += b[ch];
                    }
                    let out_idx = ((n * c + ch) * h_out + oh) * w_out + ow;
                    output[out_idx] = acc;
                }
            }
        }
    }
    output
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generate PTX for f32 depthwise conv2d (NCHW layout).
///
/// # Kernel parameters
///
/// - `input`: `[N, C, H_in, W_in]` f32 pointer
/// - `weight`: `[C, 1, kH, kW]` f32 pointer (equivalently `[C, kH, kW]`)
/// - `bias`: `[C]` f32 pointer (only if `use_bias`)
/// - `output`: `[N, C, H_out, W_out]` f32 pointer
/// - `batch_size` (u32), `h_in` (u32), `w_in` (u32),
///   `h_out` (u32), `w_out` (u32)
///
/// Channels, kernel size, stride, and padding are baked as constants.
pub fn generate_depthwise_conv2d_ptx(
    config: &PtxDepthwiseConv2dConfig,
) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let name = &config.kernel_name;
    let block_size = config.block_size;
    let channels = config.channels;
    let kh = config.kernel_h;
    let kw = config.kernel_w;
    let sh = config.stride_h;
    let sw = config.stride_w;
    let ph = config.padding_h;
    let pw = config.padding_w;

    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(4096);

    // -- Module header --
    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// DepthwiseConv2d f32 (NCHW): channels={channels}, kernel={kh}x{kw}, \
         stride={sh}x{sw}, pad={ph}x{pw}, block={block_size}\n\n"
    ));

    // -- Kernel entry --
    ptx.push_str(&format!(
        ".visible .entry {name}(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_weight,\n"
    ));
    if config.use_bias {
        ptx.push_str("\x20   .param .u64 param_bias,\n");
    }
    ptx.push_str(
        "\x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_batch_size,\n\
         \x20   .param .u32 param_h_in,\n\
         \x20   .param .u32 param_w_in,\n\
         \x20   .param .u32 param_h_out,\n\
         \x20   .param .u32 param_w_out\n\
         )\n",
    );

    ptx.push_str(&format!(".reqntid {block_size}\n{{\n"));

    // -- Register declarations --
    ptx.push_str(
        "\x20   .reg .u32  %r<32>;\n\
         \x20   .reg .f32  %f<8>;\n\
         \x20   .reg .u64  %rd<12>;\n\
         \x20   .reg .pred %p<6>;\n\n",
    );

    // -- Load parameters --
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_weight];\n",
    );
    if config.use_bias {
        ptx.push_str("\x20   ld.param.u64  %rd2, [param_bias];\n");
    }
    ptx.push_str(
        "\x20   ld.param.u64  %rd3, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_batch_size];\n\
         \x20   ld.param.u32  %r1,  [param_h_in];\n\
         \x20   ld.param.u32  %r2,  [param_w_in];\n\
         \x20   ld.param.u32  %r3,  [param_h_out];\n\
         \x20   ld.param.u32  %r4,  [param_w_out];\n\n",
    );

    // -- Global index and stride --
    ptx.push_str(
        "\x20   mov.u32       %r5, %tid.x;\n\
         \x20   mov.u32       %r6, %ctaid.x;\n\
         \x20   mov.u32       %r7, %ntid.x;\n\
         \x20   mad.lo.u32    %r8, %r6, %r7, %r5;     // global_idx\n\
         \x20   mov.u32       %r9, %nctaid.x;\n\
         \x20   mul.lo.u32    %r10, %r9, %r7;          // grid_stride\n\n",
    );

    // Total output: batch * channels * h_out * w_out
    ptx.push_str(&format!(
        "\x20   mul.lo.u32    %r11, %r0, {channels};   // batch * channels\n\
         \x20   mul.lo.u32    %r11, %r11, %r3;         // * h_out\n\
         \x20   mul.lo.u32    %r11, %r11, %r4;         // * w_out = total\n\n"
    ));

    // -- Grid-stride loop --
    ptx.push_str(
        "DW_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r8, %r11;\n\
         \x20   @%p0 bra      DW_EXIT;\n\n",
    );

    // Decompose: ow, oh, c, n
    ptx.push_str(&format!(
        "\x20   rem.u32       %r12, %r8, %r4;          // ow = idx % w_out\n\
         \x20   div.u32       %r13, %r8, %r4;\n\
         \x20   rem.u32       %r14, %r13, %r3;         // oh = temp / h_out rem\n\
         \x20   div.u32       %r15, %r13, %r3;\n\
         \x20   rem.u32       %r16, %r15, {channels};  // ch = temp2 % channels\n\
         \x20   div.u32       %r17, %r15, {channels};  // n = temp2 / channels\n\n"
    ));

    // Initialize accumulator
    ptx.push_str(&format!(
        "\x20   mov.f32       %f0, {zero};             // acc = 0.0\n\n"
    ));

    // -- Kernel window loops --
    ptx.push_str(&format!(
        "\x20   mov.u32       %r18, 0;                // ki = 0\n\
         DW_KH_LOOP:\n\
         \x20   setp.ge.u32   %p1, %r18, {kh};\n\
         \x20   @%p1 bra      DW_KH_DONE;\n\n\
         \x20   mov.u32       %r19, 0;                // kj = 0\n\
         DW_KW_LOOP:\n\
         \x20   setp.ge.u32   %p2, %r19, {kw};\n\
         \x20   @%p2 bra      DW_KW_DONE;\n\n"
    ));

    // Compute input position
    ptx.push_str(&format!(
        "\x20   mul.lo.u32    %r20, %r14, {sh};       // oh * stride_h\n\
         \x20   add.u32       %r20, %r20, %r18;       // + ki\n\
         \x20   sub.u32       %r20, %r20, {ph};       // - pad_h\n\
         \x20   mul.lo.u32    %r21, %r12, {sw};       // ow * stride_w\n\
         \x20   add.u32       %r21, %r21, %r19;       // + kj\n\
         \x20   sub.u32       %r21, %r21, {pw};       // - pad_w\n\n"
    ));

    // Bounds check (unsigned arithmetic handles negative wrap)
    ptx.push_str(
        "\x20   setp.ge.u32   %p3, %r20, %r1;        // ih >= h_in?\n\
         \x20   @%p3 bra      DW_KW_NEXT;\n\
         \x20   setp.ge.u32   %p4, %r21, %r2;        // iw >= w_in?\n\
         \x20   @%p4 bra      DW_KW_NEXT;\n\n",
    );

    // Load input[n, ch, ih, iw]
    ptx.push_str(&format!(
        "\x20   mad.lo.u32    %r22, %r17, {channels}, %r16; // n * C + ch\n\
         \x20   mad.lo.u32    %r22, %r22, %r1, %r20;  // * h_in + ih\n\
         \x20   mad.lo.u32    %r22, %r22, %r2, %r21;  // * w_in + iw\n\
         \x20   mul.wide.u32  %rd4, %r22, 4;\n\
         \x20   add.u64       %rd5, %rd0, %rd4;\n\
         \x20   ld.global.f32 %f1, [%rd5];\n\n"
    ));

    // Load weight[ch, ki, kj]
    // Weight layout: [C, kH, kW] flattened
    ptx.push_str(&format!(
        "\x20   mad.lo.u32    %r23, %r16, {kh}, %r18; // ch * kH + ki\n\
         \x20   mad.lo.u32    %r23, %r23, {kw}, %r19; // * kW + kj\n\
         \x20   mul.wide.u32  %rd6, %r23, 4;\n\
         \x20   add.u64       %rd7, %rd1, %rd6;\n\
         \x20   ld.global.f32 %f2, [%rd7];\n\n"
    ));

    // FMA
    ptx.push_str("\x20   fma.rn.f32    %f0, %f1, %f2, %f0;    // acc += input * weight\n\n");

    // KW loop end
    ptx.push_str(
        "DW_KW_NEXT:\n\
         \x20   add.u32       %r19, %r19, 1;\n\
         \x20   bra           DW_KW_LOOP;\n\
         DW_KW_DONE:\n\n",
    );

    // KH loop end
    ptx.push_str(
        "\x20   add.u32       %r18, %r18, 1;\n\
         \x20   bra           DW_KH_LOOP;\n\
         DW_KH_DONE:\n\n",
    );

    // Add bias if enabled
    if config.use_bias {
        ptx.push_str(
            "\x20   // Add bias[ch]\n\
             \x20   mul.wide.u32  %rd8, %r16, 4;          // ch * sizeof(f32)\n\
             \x20   add.u64       %rd9, %rd2, %rd8;\n\
             \x20   ld.global.f32 %f3, [%rd9];\n\
             \x20   add.f32       %f0, %f0, %f3;\n\n",
        );
    }

    // Store output[n, ch, oh, ow]
    ptx.push_str(&format!(
        "\x20   mad.lo.u32    %r24, %r17, {channels}, %r16; // n * C + ch\n\
         \x20   mad.lo.u32    %r24, %r24, %r3, %r14;  // * h_out + oh\n\
         \x20   mad.lo.u32    %r24, %r24, %r4, %r12;  // * w_out + ow\n\
         \x20   mul.wide.u32  %rd10, %r24, 4;\n\
         \x20   add.u64       %rd11, %rd3, %rd10;\n\
         \x20   st.global.f32 [%rd11], %f0;\n\n"
    ));

    // Grid-stride advance
    ptx.push_str(
        "\x20   add.u32       %r8, %r8, %r10;         // global_idx += grid_stride\n\
         \x20   bra           DW_LOOP;\n\n\
         DW_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Launch config
// ---------------------------------------------------------------------------

/// Compute the CUDA launch config for a depthwise conv2d kernel.
#[must_use]
pub fn ptx_depthwise_conv2d_launch_config(
    batch: usize,
    channels: usize,
    output_h: usize,
    output_w: usize,
) -> CudaLaunchConfig {
    let total = batch * channels * output_h * output_w;
    let block_size = PTX_DEPTHWISE_CONV2D_BLOCK_SIZE as u32;
    let grid_x = (total as u64)
        .div_ceil(u64::from(block_size))
        .min(u64::from(u32::MAX)) as u32;
    CudaLaunchConfig {
        grid: CudaDim3::d1(grid_x),
        block: CudaDim3::d1(block_size),
        shared_mem_bytes: 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_depthwise_conv_tests.rs"]
mod ptx_depthwise_conv_tests;
