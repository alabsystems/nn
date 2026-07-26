// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for 2D pooling operations (NCHW layout).
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 pooling
//! kernels: max pooling, average pooling, and adaptive average pooling.
//! These are emitted as direct PTX — no `nvcc` compilation needed.
//!
//! ## Supported operations
//!
//! - **Max pool 2D**: Sliding window maximum over spatial dimensions
//! - **Avg pool 2D**: Sliding window mean over spatial dimensions
//! - **Adaptive avg pool 2D**: Output-size-driven average pooling
//!
//! ## Layout
//!
//! - Input:  `[N, C, H_in,  W_in ]` (NCHW)
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

/// Default block size for pooling kernels (256 threads).
pub const PTX_POOL2D_BLOCK_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for PTX max/avg pool 2D kernel generation.
#[derive(Debug, Clone)]
pub struct PtxPool2dConfig {
    /// Kernel function name in the PTX module.
    pub kernel_name: String,
    /// Pooling kernel height.
    pub kernel_h: usize,
    /// Pooling kernel width.
    pub kernel_w: usize,
    /// Stride height.
    pub stride_h: usize,
    /// Stride width.
    pub stride_w: usize,
    /// Padding height (symmetric).
    pub pad_h: usize,
    /// Padding width (symmetric).
    pub pad_w: usize,
    /// Thread block size (default: 256).
    pub block_size: usize,
    /// SM target for the PTX prelude (e.g., "sm_80").
    pub sm_target: String,
}

impl PtxPool2dConfig {
    /// Create a config with default stride equal to kernel size, pad=0.
    pub fn new(kernel_name: &str, kernel_h: usize, kernel_w: usize) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            kernel_h,
            kernel_w,
            stride_h: kernel_h,
            stride_w: kernel_w,
            pad_h: 0,
            pad_w: 0,
            block_size: PTX_POOL2D_BLOCK_SIZE,
            sm_target: DEFAULT_SM_TARGET.to_string(),
        }
    }

    /// Set stride height and width.
    #[must_use]
    pub fn with_stride(mut self, stride_h: usize, stride_w: usize) -> Self {
        self.stride_h = stride_h;
        self.stride_w = stride_w;
        self
    }

    /// Set padding height and width.
    #[must_use]
    pub fn with_padding(mut self, pad_h: usize, pad_w: usize) -> Self {
        self.pad_h = pad_h;
        self.pad_w = pad_w;
        self
    }

    /// Set the thread block size.
    #[must_use]
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Set the SM target.
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

impl Default for PtxPool2dConfig {
    fn default() -> Self {
        Self::new("ptx_pool2d_f32", 2, 2)
    }
}

/// Configuration for adaptive average pool 2D.
#[derive(Debug, Clone)]
pub struct PtxAdaptiveAvgPool2dConfig {
    /// Kernel function name in the PTX module.
    pub kernel_name: String,
    /// Desired output height.
    pub output_h: usize,
    /// Desired output width.
    pub output_w: usize,
    /// Thread block size (default: 256).
    pub block_size: usize,
    /// SM target for the PTX prelude.
    pub sm_target: String,
}

impl PtxAdaptiveAvgPool2dConfig {
    /// Create a config for adaptive avg pool with given output size.
    pub fn new(kernel_name: &str, output_h: usize, output_w: usize) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            output_h,
            output_w,
            block_size: PTX_POOL2D_BLOCK_SIZE,
            sm_target: DEFAULT_SM_TARGET.to_string(),
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PtxCodegenError> {
        if self.kernel_name.is_empty() {
            return Err(PtxCodegenError::InvalidParameter(
                "kernel_name must not be empty".into(),
            ));
        }
        if self.output_h == 0 || self.output_w == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "output dimensions must be > 0".into(),
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

// ---------------------------------------------------------------------------
// Output size calculation
// ---------------------------------------------------------------------------

/// Compute the output size for a pooling dimension.
///
/// `out = (input_size + 2*padding - kernel_size) / stride + 1`
///
/// Returns `None` if the parameters would produce a non-positive output.
#[must_use]
pub fn pool2d_output_size(
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
// Reference implementations
// ---------------------------------------------------------------------------

/// CPU reference implementation for max pool 2D (NCHW layout).
pub fn max_pool2d_reference(
    input: &[f32],
    batch: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    config: &PtxPool2dConfig,
) -> Vec<f32> {
    let h_out = pool2d_output_size(h_in, config.kernel_h, config.stride_h, config.pad_h)
        .expect("invalid output height");
    let w_out = pool2d_output_size(w_in, config.kernel_w, config.stride_w, config.pad_w)
        .expect("invalid output width");

    assert_eq!(input.len(), batch * channels * h_in * w_in);

    let mut output = vec![f32::NEG_INFINITY; batch * channels * h_out * w_out];

    for n in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;
                    for kh in 0..config.kernel_h {
                        for kw in 0..config.kernel_w {
                            let ih = oh * config.stride_h + kh;
                            let iw = ow * config.stride_w + kw;
                            let ih = ih as isize - config.pad_h as isize;
                            let iw = iw as isize - config.pad_w as isize;
                            if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                let idx =
                                    ((n * channels + c) * h_in + ih as usize) * w_in + iw as usize;
                                let val = input[idx];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    let out_idx = ((n * channels + c) * h_out + oh) * w_out + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }
    output
}

/// CPU reference implementation for avg pool 2D (NCHW layout).
pub fn avg_pool2d_reference(
    input: &[f32],
    batch: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    config: &PtxPool2dConfig,
) -> Vec<f32> {
    let h_out = pool2d_output_size(h_in, config.kernel_h, config.stride_h, config.pad_h)
        .expect("invalid output height");
    let w_out = pool2d_output_size(w_in, config.kernel_w, config.stride_w, config.pad_w)
        .expect("invalid output width");

    assert_eq!(input.len(), batch * channels * h_in * w_in);

    let mut output = vec![0.0f32; batch * channels * h_out * w_out];

    for n in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0f32;
                    let mut count = 0u32;
                    for kh in 0..config.kernel_h {
                        for kw in 0..config.kernel_w {
                            let ih = oh * config.stride_h + kh;
                            let iw = ow * config.stride_w + kw;
                            let ih = ih as isize - config.pad_h as isize;
                            let iw = iw as isize - config.pad_w as isize;
                            if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                let idx =
                                    ((n * channels + c) * h_in + ih as usize) * w_in + iw as usize;
                                sum += input[idx];
                                count += 1;
                            }
                        }
                    }
                    let out_idx = ((n * channels + c) * h_out + oh) * w_out + ow;
                    output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
    }
    output
}

/// CPU reference for adaptive avg pool 2D (NCHW layout).
pub fn adaptive_avg_pool2d_reference(
    input: &[f32],
    batch: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    output_h: usize,
    output_w: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), batch * channels * h_in * w_in);
    let mut output = vec![0.0f32; batch * channels * output_h * output_w];

    for n in 0..batch {
        for c in 0..channels {
            for oh in 0..output_h {
                for ow in 0..output_w {
                    let ih_start = (oh * h_in) / output_h;
                    let ih_end = ((oh + 1) * h_in) / output_h;
                    let iw_start = (ow * w_in) / output_w;
                    let iw_end = ((ow + 1) * w_in) / output_w;

                    let mut sum = 0.0f32;
                    let mut count = 0u32;
                    for ih in ih_start..ih_end {
                        for iw in iw_start..iw_end {
                            let idx = ((n * channels + c) * h_in + ih) * w_in + iw;
                            sum += input[idx];
                            count += 1;
                        }
                    }
                    let out_idx = ((n * channels + c) * output_h + oh) * output_w + ow;
                    output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
    }
    output
}

// ---------------------------------------------------------------------------
// PTX generation — max pool 2D
// ---------------------------------------------------------------------------

/// Generate PTX for f32 max pooling 2D (NCHW layout).
///
/// # Kernel parameters
///
/// - `input`: `[N, C, H_in, W_in]` f32 pointer
/// - `output`: `[N, C, H_out, W_out]` f32 pointer
/// - `batch_size` (u32), `channels` (u32), `h_in` (u32), `w_in` (u32),
///   `h_out` (u32), `w_out` (u32)
///
/// Pool kernel size, stride, and padding are baked as compile-time constants.
pub fn generate_max_pool2d_ptx(config: &PtxPool2dConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;
    generate_pool2d_ptx_inner(config, PoolKind::Max)
}

/// Generate PTX for f32 average pooling 2D (NCHW layout).
pub fn generate_avg_pool2d_ptx(config: &PtxPool2dConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;
    generate_pool2d_ptx_inner(config, PoolKind::Avg)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PoolKind {
    Max,
    Avg,
}

fn generate_pool2d_ptx_inner(
    config: &PtxPool2dConfig,
    kind: PoolKind,
) -> Result<String, PtxCodegenError> {
    let name = &config.kernel_name;
    let block_size = config.block_size;
    let kh = config.kernel_h;
    let kw = config.kernel_w;
    let sh = config.stride_h;
    let sw = config.stride_w;
    let ph = config.pad_h;
    let pw = config.pad_w;

    let kind_str = match kind {
        PoolKind::Max => "MaxPool2d",
        PoolKind::Avg => "AvgPool2d",
    };

    let neg_inf = format_ptx_float(f32::NEG_INFINITY);
    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(4096);

    // -- Module header --
    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// {kind_str} f32 (NCHW): kernel={kh}x{kw}, stride={sh}x{sw}, \
         pad={ph}x{pw}, block={block_size}\n\n"
    ));

    // -- Kernel entry --
    ptx.push_str(&format!(
        ".visible .entry {name}(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_batch_size,\n\
         \x20   .param .u32 param_channels,\n\
         \x20   .param .u32 param_h_in,\n\
         \x20   .param .u32 param_w_in,\n\
         \x20   .param .u32 param_h_out,\n\
         \x20   .param .u32 param_w_out\n\
         )\n"
    ));

    ptx.push_str(&format!(".reqntid {block_size}\n{{\n"));

    // -- Register declarations --
    ptx.push_str(
        "\x20   .reg .u32  %r<32>;\n\
         \x20   .reg .f32  %f<8>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<6>;\n\n",
    );

    // -- Load parameters --
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_batch_size];\n\
         \x20   ld.param.u32  %r1,  [param_channels];\n\
         \x20   ld.param.u32  %r2,  [param_h_in];\n\
         \x20   ld.param.u32  %r3,  [param_w_in];\n\
         \x20   ld.param.u32  %r4,  [param_h_out];\n\
         \x20   ld.param.u32  %r5,  [param_w_out];\n\n",
    );

    // -- Global thread index and grid stride --
    ptx.push_str(
        "\x20   mov.u32       %r6, %tid.x;\n\
         \x20   mov.u32       %r7, %ctaid.x;\n\
         \x20   mov.u32       %r8, %ntid.x;\n\
         \x20   mad.lo.u32    %r9, %r7, %r8, %r6;    // global_idx\n\
         \x20   mov.u32       %r10, %nctaid.x;\n\
         \x20   mul.lo.u32    %r11, %r10, %r8;        // grid_stride\n\n",
    );

    // -- Total output elements: batch * channels * h_out * w_out --
    ptx.push_str(
        "\x20   mul.lo.u32    %r12, %r0, %r1;         // batch * channels\n\
         \x20   mul.lo.u32    %r12, %r12, %r4;        // * h_out\n\
         \x20   mul.lo.u32    %r12, %r12, %r5;        // * w_out = total\n\n",
    );

    // -- Grid-stride loop --
    ptx.push_str(
        "POOL_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r9, %r12;\n\
         \x20   @%p0 bra      POOL_EXIT;\n\n",
    );

    // -- Decompose global_idx -> (n, c, oh, ow) --
    // ow = global_idx % w_out
    // temp1 = global_idx / w_out
    // oh = temp1 % h_out
    // temp2 = temp1 / h_out
    // c = temp2 % channels
    // n = temp2 / channels
    ptx.push_str(
        "\x20   rem.u32       %r13, %r9, %r5;         // ow = idx % w_out\n\
         \x20   div.u32       %r14, %r9, %r5;         // temp1 = idx / w_out\n\
         \x20   rem.u32       %r15, %r14, %r4;        // oh = temp1 % h_out\n\
         \x20   div.u32       %r16, %r14, %r4;        // temp2 = temp1 / h_out\n\
         \x20   rem.u32       %r17, %r16, %r1;        // c = temp2 % channels\n\
         \x20   div.u32       %r18, %r16, %r1;        // n = temp2 / channels\n\n",
    );

    // -- Initialize accumulator --
    match kind {
        PoolKind::Max => {
            ptx.push_str(&format!(
                "\x20   mov.f32       %f0, {neg_inf};        // max_val = -inf\n\n"
            ));
        }
        PoolKind::Avg => {
            ptx.push_str(&format!(
                "\x20   mov.f32       %f0, {zero};           // sum = 0.0\n\
                 \x20   mov.u32       %r30, 0;               // count = 0\n\n"
            ));
        }
    }

    // -- Loop over kernel window --
    ptx.push_str(&format!(
        "\x20   mov.u32       %r19, 0;               // kh_idx = 0\n\
         KH_LOOP:\n\
         \x20   setp.ge.u32   %p1, %r19, {kh};\n\
         \x20   @%p1 bra      KH_DONE;\n\n\
         \x20   mov.u32       %r20, 0;               // kw_idx = 0\n\
         KW_LOOP:\n\
         \x20   setp.ge.u32   %p2, %r20, {kw};\n\
         \x20   @%p2 bra      KW_DONE;\n\n"
    ));

    // -- Compute input position: ih = oh * stride_h + kh_idx - pad_h --
    ptx.push_str(&format!(
        "\x20   mul.lo.u32    %r21, %r15, {sh};       // oh * stride_h\n\
         \x20   add.u32       %r21, %r21, %r19;       // + kh_idx\n\
         \x20   sub.u32       %r21, %r21, {ph};       // - pad_h (unsigned; wraps if negative)\n\
         \x20   mul.lo.u32    %r22, %r13, {sw};       // ow * stride_w\n\
         \x20   add.u32       %r22, %r22, %r20;       // + kw_idx\n\
         \x20   sub.u32       %r22, %r22, {pw};       // - pad_w\n\n"
    ));

    // -- Bounds check: ih < h_in && iw < w_in (unsigned handles negatives) --
    ptx.push_str(
        "\x20   setp.ge.u32   %p3, %r21, %r2;        // ih >= h_in?\n\
         \x20   @%p3 bra      KW_NEXT;\n\
         \x20   setp.ge.u32   %p4, %r22, %r3;        // iw >= w_in?\n\
         \x20   @%p4 bra      KW_NEXT;\n\n",
    );

    // -- Load input[n, c, ih, iw] --
    // offset = ((n * channels + c) * h_in + ih) * w_in + iw
    ptx.push_str(
        "\x20   mad.lo.u32    %r23, %r18, %r1, %r17; // n * channels + c\n\
         \x20   mad.lo.u32    %r23, %r23, %r2, %r21; // * h_in + ih\n\
         \x20   mad.lo.u32    %r23, %r23, %r3, %r22; // * w_in + iw\n\
         \x20   mul.wide.u32  %rd2, %r23, 4;         // byte offset\n\
         \x20   add.u64       %rd3, %rd0, %rd2;      // &input[...]\n\
         \x20   ld.global.f32 %f1, [%rd3];           // val\n\n",
    );

    // -- Accumulate --
    match kind {
        PoolKind::Max => {
            ptx.push_str(
                "\x20   max.f32       %f0, %f0, %f1;        // max_val = max(max_val, val)\n\n",
            );
        }
        PoolKind::Avg => {
            ptx.push_str(
                "\x20   add.f32       %f0, %f0, %f1;        // sum += val\n\
                 \x20   add.u32       %r30, %r30, 1;        // count++\n\n",
            );
        }
    }

    // -- KW loop end --
    ptx.push_str(
        "KW_NEXT:\n\
         \x20   add.u32       %r20, %r20, 1;\n\
         \x20   bra           KW_LOOP;\n\
         KW_DONE:\n\n",
    );

    // -- KH loop end --
    ptx.push_str(
        "\x20   add.u32       %r19, %r19, 1;\n\
         \x20   bra           KH_LOOP;\n\
         KH_DONE:\n\n",
    );

    // -- For avg pool, divide by count --
    if kind == PoolKind::Avg {
        ptx.push_str(
            "\x20   // Divide sum by count for average\n\
             \x20   cvt.rn.f32.u32 %f2, %r30;           // count as float\n\
             \x20   div.rn.f32    %f0, %f0, %f2;        // avg = sum / count\n\n",
        );
    }

    // -- Store output[n, c, oh, ow] --
    ptx.push_str(
        "\x20   mad.lo.u32    %r24, %r18, %r1, %r17; // n * channels + c\n\
         \x20   mad.lo.u32    %r24, %r24, %r4, %r15; // * h_out + oh\n\
         \x20   mad.lo.u32    %r24, %r24, %r5, %r13; // * w_out + ow\n\
         \x20   mul.wide.u32  %rd4, %r24, 4;         // byte offset\n\
         \x20   add.u64       %rd5, %rd1, %rd4;      // &output[...]\n\
         \x20   st.global.f32 [%rd5], %f0;\n\n",
    );

    // -- Grid-stride advance --
    ptx.push_str(
        "\x20   add.u32       %r9, %r9, %r11;        // global_idx += grid_stride\n\
         \x20   bra           POOL_LOOP;\n\n",
    );

    ptx.push_str(
        "POOL_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// PTX generation — adaptive avg pool 2D
// ---------------------------------------------------------------------------

/// Generate PTX for f32 adaptive average pooling 2D (NCHW layout).
///
/// The window for each output element is computed as:
/// - `ih_start = oh * h_in / output_h`
/// - `ih_end = (oh + 1) * h_in / output_h`
///
/// (and similarly for width).
///
/// Output dimensions are baked as compile-time constants. Input H/W
/// are passed as runtime parameters.
pub fn generate_adaptive_avg_pool2d_ptx(
    config: &PtxAdaptiveAvgPool2dConfig,
) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let name = &config.kernel_name;
    let block_size = config.block_size;
    let out_h = config.output_h;
    let out_w = config.output_w;
    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// AdaptiveAvgPool2d f32 (NCHW): output={out_h}x{out_w}, block={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry {name}(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_batch_size,\n\
         \x20   .param .u32 param_channels,\n\
         \x20   .param .u32 param_h_in,\n\
         \x20   .param .u32 param_w_in\n\
         )\n"
    ));

    ptx.push_str(&format!(".reqntid {block_size}\n{{\n"));

    ptx.push_str(
        "\x20   .reg .u32  %r<32>;\n\
         \x20   .reg .f32  %f<8>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<6>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_batch_size];\n\
         \x20   ld.param.u32  %r1,  [param_channels];\n\
         \x20   ld.param.u32  %r2,  [param_h_in];\n\
         \x20   ld.param.u32  %r3,  [param_w_in];\n\n",
    );

    // Global thread index and stride
    ptx.push_str(
        "\x20   mov.u32       %r4, %tid.x;\n\
         \x20   mov.u32       %r5, %ctaid.x;\n\
         \x20   mov.u32       %r6, %ntid.x;\n\
         \x20   mad.lo.u32    %r7, %r5, %r6, %r4;    // global_idx\n\
         \x20   mov.u32       %r8, %nctaid.x;\n\
         \x20   mul.lo.u32    %r9, %r8, %r6;          // grid_stride\n\n",
    );

    // Total output elements: batch * channels * output_h * output_w
    ptx.push_str(&format!(
        "\x20   mul.lo.u32    %r10, %r0, %r1;\n\
         \x20   mul.lo.u32    %r10, %r10, {out_h};\n\
         \x20   mul.lo.u32    %r10, %r10, {out_w};    // total\n\n"
    ));

    ptx.push_str(
        "APOOL_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r7, %r10;\n\
         \x20   @%p0 bra      APOOL_EXIT;\n\n",
    );

    // Decompose: ow = idx % out_w, oh = (idx / out_w) % out_h, etc.
    ptx.push_str(&format!(
        "\x20   rem.u32       %r11, %r7, {out_w};     // ow\n\
         \x20   div.u32       %r12, %r7, {out_w};\n\
         \x20   rem.u32       %r13, %r12, {out_h};    // oh\n\
         \x20   div.u32       %r14, %r12, {out_h};\n\
         \x20   rem.u32       %r15, %r14, %r1;        // c\n\
         \x20   div.u32       %r16, %r14, %r1;        // n\n\n"
    ));

    // Compute adaptive window bounds:
    // ih_start = oh * h_in / output_h, ih_end = (oh+1) * h_in / output_h
    // iw_start = ow * w_in / output_w, iw_end = (ow+1) * w_in / output_w
    ptx.push_str(&format!(
        "\x20   mul.lo.u32    %r17, %r13, %r2;        // oh * h_in\n\
         \x20   div.u32       %r17, %r17, {out_h};    // ih_start\n\
         \x20   add.u32       %r18, %r13, 1;          // oh + 1\n\
         \x20   mul.lo.u32    %r18, %r18, %r2;        // (oh+1) * h_in\n\
         \x20   div.u32       %r18, %r18, {out_h};    // ih_end\n\
         \x20   mul.lo.u32    %r19, %r11, %r3;        // ow * w_in\n\
         \x20   div.u32       %r19, %r19, {out_w};    // iw_start\n\
         \x20   add.u32       %r20, %r11, 1;          // ow + 1\n\
         \x20   mul.lo.u32    %r20, %r20, %r3;        // (ow+1) * w_in\n\
         \x20   div.u32       %r20, %r20, {out_w};    // iw_end\n\n"
    ));

    // Initialize sum and count
    ptx.push_str(&format!(
        "\x20   mov.f32       %f0, {zero};            // sum = 0\n\
         \x20   mov.u32       %r30, 0;                // count = 0\n\n"
    ));

    // Loop ih_start..ih_end
    ptx.push_str(
        "\x20   mov.u32       %r21, %r17;             // ih = ih_start\n\
         AIH_LOOP:\n\
         \x20   setp.ge.u32   %p1, %r21, %r18;       // ih >= ih_end?\n\
         \x20   @%p1 bra      AIH_DONE;\n\n\
         \x20   mov.u32       %r22, %r19;             // iw = iw_start\n\
         AIW_LOOP:\n\
         \x20   setp.ge.u32   %p2, %r22, %r20;       // iw >= iw_end?\n\
         \x20   @%p2 bra      AIW_DONE;\n\n",
    );

    // Load input[n, c, ih, iw]
    ptx.push_str(
        "\x20   mad.lo.u32    %r23, %r16, %r1, %r15; // n * channels + c\n\
         \x20   mad.lo.u32    %r23, %r23, %r2, %r21; // * h_in + ih\n\
         \x20   mad.lo.u32    %r23, %r23, %r3, %r22; // * w_in + iw\n\
         \x20   mul.wide.u32  %rd2, %r23, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f1, [%rd3];\n\
         \x20   add.f32       %f0, %f0, %f1;          // sum += val\n\
         \x20   add.u32       %r30, %r30, 1;          // count++\n\n",
    );

    // iw loop end
    ptx.push_str(
        "\x20   add.u32       %r22, %r22, 1;\n\
         \x20   bra           AIW_LOOP;\n\
         AIW_DONE:\n\n\
         \x20   add.u32       %r21, %r21, 1;\n\
         \x20   bra           AIH_LOOP;\n\
         AIH_DONE:\n\n",
    );

    // Divide sum by count
    ptx.push_str(
        "\x20   cvt.rn.f32.u32 %f2, %r30;\n\
         \x20   div.rn.f32    %f0, %f0, %f2;          // avg = sum / count\n\n",
    );

    // Store output
    ptx.push_str(&format!(
        "\x20   mad.lo.u32    %r24, %r16, %r1, %r15;  // n * channels + c\n\
         \x20   mad.lo.u32    %r24, %r24, {out_h}, %r13; // * out_h + oh\n\
         \x20   mad.lo.u32    %r24, %r24, {out_w}, %r11; // * out_w + ow\n\
         \x20   mul.wide.u32  %rd4, %r24, 4;\n\
         \x20   add.u64       %rd5, %rd1, %rd4;\n\
         \x20   st.global.f32 [%rd5], %f0;\n\n"
    ));

    // Grid-stride advance
    ptx.push_str(
        "\x20   add.u32       %r7, %r7, %r9;\n\
         \x20   bra           APOOL_LOOP;\n\n\
         APOOL_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Launch config
// ---------------------------------------------------------------------------

/// Compute the CUDA launch config for a pool2d kernel.
#[must_use]
pub fn ptx_pool2d_launch_config(
    batch: usize,
    channels: usize,
    output_h: usize,
    output_w: usize,
) -> CudaLaunchConfig {
    let total = batch * channels * output_h * output_w;
    let block_size = PTX_POOL2D_BLOCK_SIZE as u32;
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
#[path = "ptx_pooling_tests.rs"]
mod ptx_pooling_tests;
