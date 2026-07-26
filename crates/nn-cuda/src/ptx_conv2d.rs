// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for 2D convolution (NCHW layout).
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 conv2d
//! with implicit im2col via shared memory. Unlike the CUDA C++ emission in
//! [`ptx_emit`], this module emits PTX assembly directly — no `nvcc`
//! compilation step needed. The PTX can be loaded via `cuModuleLoadData`
//! (JIT) or assembled to cubin via `ptxas`.
//!
//! ## Algorithm
//!
//! Each thread computes one output element `out[n, oc, oh, ow]`.
//! For standard kernels (kH > 1 or kW > 1), a shared memory tile caches
//! a patch of the input channel for the current output-channel group,
//! reducing redundant global loads across the kernel window.
//!
//! For 1x1 convolutions the shared memory tile is skipped — the kernel
//! degenerates to a direct dot product between the input pixel vector
//! and the filter weight vector, essentially a pointwise matmul.
//!
//! ## Layout
//!
//! - Input:  `[N, C_in,  H_in,  W_in ]` (NCHW)
//! - Weight: `[C_out, C_in/groups, kH, kW]`
//! - Bias:   `[C_out]` (optional)
//! - Output: `[N, C_out, H_out, W_out]` (NCHW)
//!
//! ## PTX register usage
//!
//! - `%r0..%r31`: general-purpose 32-bit registers (indices, loop vars, temps)
//! - `%f0..%f7`: 32-bit float registers (accumulator, loaded values, products)
//! - `%rd0..%rd11`: 64-bit registers (pointer arithmetic)
//! - `%p0..%p5`: predicate registers (bounds checks, loop conditions)
//!
//! ## Thread block configuration
//!
//! Default block: 16x16 (256 threads). Each thread produces one output pixel.
//! Grid: `(ceil(W_out/block_w), ceil(H_out/block_h), N * C_out)` — z-dim
//! covers batch and output-channel.
//!
//! Parallel to Metal conv2d in `dyn_tensor_metal_ops_conv.rs`.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude, PtxCodegenError, DEFAULT_SM_TARGET};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block width for conv2d output tile.
pub const PTX_CONV2D_BLOCK_W: usize = 16;

/// Default block height for conv2d output tile.
pub const PTX_CONV2D_BLOCK_H: usize = 16;

/// Minimum block dimension for conv2d.
pub const PTX_CONV2D_MIN_BLOCK: usize = 4;

/// Maximum block dimension for conv2d.
pub const PTX_CONV2D_MAX_BLOCK: usize = 32;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for PTX conv2d kernel generation.
#[derive(Debug, Clone)]
pub struct PtxConv2dConfig {
    /// Kernel function name in the PTX module.
    pub kernel_name: String,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Convolution kernel height.
    pub kernel_h: usize,
    /// Convolution kernel width.
    pub kernel_w: usize,
    /// Stride height.
    pub stride_h: usize,
    /// Stride width.
    pub stride_w: usize,
    /// Padding height (symmetric, applied to both top and bottom).
    pub pad_h: usize,
    /// Padding width (symmetric, applied to both left and right).
    pub pad_w: usize,
    /// Dilation height.
    pub dilation_h: usize,
    /// Dilation width.
    pub dilation_w: usize,
    /// Number of groups for grouped convolution.
    ///
    /// - `groups = 1`: standard convolution (all input channels contribute to all output channels)
    /// - `groups = in_channels = out_channels`: depthwise convolution
    /// - `1 < groups < in_channels`: grouped convolution
    pub groups: usize,
    /// Whether to add bias.
    pub use_bias: bool,
    /// Thread block width (output tile columns).
    pub block_w: usize,
    /// Thread block height (output tile rows).
    pub block_h: usize,
    /// SM target for the PTX prelude (e.g., "sm_80").
    pub sm_target: String,
}

impl PtxConv2dConfig {
    /// Create a config for a given kernel size with default stride=1,
    /// pad=0, dilation=1, groups=1, block 16x16, sm_80 target.
    ///
    /// Channel counts default to 0 (passed as runtime params to the kernel).
    /// Use [`with_channels`] for compile-time channel baking, or set
    /// `in_channels`/`out_channels` after construction.
    pub fn new(kernel_name: &str, kernel_h: usize, kernel_w: usize) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            in_channels: 0,
            out_channels: 0,
            kernel_h,
            kernel_w,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
            use_bias: false,
            block_w: PTX_CONV2D_BLOCK_W,
            block_h: PTX_CONV2D_BLOCK_H,
            sm_target: DEFAULT_SM_TARGET.to_string(),
        }
    }

    /// Create a config with explicit channel counts and kernel size.
    ///
    /// Default stride=1, pad=0, dilation=1, groups=1, block 16x16, sm_80.
    pub fn with_channels(
        kernel_name: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
            use_bias: false,
            block_w: PTX_CONV2D_BLOCK_W,
            block_h: PTX_CONV2D_BLOCK_H,
            sm_target: DEFAULT_SM_TARGET.to_string(),
        }
    }

    /// Set stride for both dimensions.
    #[must_use]
    pub fn with_stride(mut self, stride_h: usize, stride_w: usize) -> Self {
        self.stride_h = stride_h;
        self.stride_w = stride_w;
        self
    }

    /// Set symmetric padding for both dimensions.
    #[must_use]
    pub fn with_padding(mut self, pad_h: usize, pad_w: usize) -> Self {
        self.pad_h = pad_h;
        self.pad_w = pad_w;
        self
    }

    /// Set dilation for both dimensions.
    #[must_use]
    pub fn with_dilation(mut self, dilation_h: usize, dilation_w: usize) -> Self {
        self.dilation_h = dilation_h;
        self.dilation_w = dilation_w;
        self
    }

    /// Enable bias addition.
    #[must_use]
    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Set the thread block dimensions.
    #[must_use]
    pub fn with_block_size(mut self, block_h: usize, block_w: usize) -> Self {
        self.block_h = block_h;
        self.block_w = block_w;
        self
    }

    /// Set the SM target (e.g., "sm_70", "sm_80", "sm_90").
    #[must_use]
    pub fn with_sm_target(mut self, target: &str) -> Self {
        self.sm_target = target.to_string();
        self
    }

    /// Set number of groups for grouped convolution.
    #[must_use]
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
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
            return Err(PtxCodegenError::InvalidParameter(format!(
                "kernel size must be >= 1, got {}x{}",
                self.kernel_h, self.kernel_w
            )));
        }
        if self.stride_h == 0 || self.stride_w == 0 {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "stride must be >= 1, got {}x{}",
                self.stride_h, self.stride_w
            )));
        }
        if self.dilation_h == 0 || self.dilation_w == 0 {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "dilation must be >= 1, got {}x{}",
                self.dilation_h, self.dilation_w
            )));
        }
        if self.groups == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "groups must be > 0".into(),
            ));
        }
        // Validate channel/group divisibility when channels are specified
        if self.in_channels > 0 && !self.in_channels.is_multiple_of(self.groups) {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "in_channels ({}) must be divisible by groups ({})",
                self.in_channels, self.groups,
            )));
        }
        if self.out_channels > 0 && !self.out_channels.is_multiple_of(self.groups) {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "out_channels ({}) must be divisible by groups ({})",
                self.out_channels, self.groups,
            )));
        }
        if self.block_w < PTX_CONV2D_MIN_BLOCK
            || self.block_w > PTX_CONV2D_MAX_BLOCK
            || self.block_h < PTX_CONV2D_MIN_BLOCK
            || self.block_h > PTX_CONV2D_MAX_BLOCK
        {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "block dimensions must be {}..={}, got {}x{}",
                PTX_CONV2D_MIN_BLOCK, PTX_CONV2D_MAX_BLOCK, self.block_h, self.block_w
            )));
        }
        let threads = self.block_h * self.block_w;
        if threads > 1024 {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "threads per block must be <= 1024, got {} ({}x{})",
                threads, self.block_h, self.block_w
            )));
        }
        Ok(())
    }

    /// Whether this is a 1x1 convolution (pointwise — no shared memory needed).
    #[must_use]
    pub fn is_pointwise(&self) -> bool {
        self.kernel_h == 1 && self.kernel_w == 1
    }

    /// Effective kernel height accounting for dilation.
    #[must_use]
    pub fn effective_kernel_h(&self) -> usize {
        (self.kernel_h - 1) * self.dilation_h + 1
    }

    /// Effective kernel width accounting for dilation.
    #[must_use]
    pub fn effective_kernel_w(&self) -> usize {
        (self.kernel_w - 1) * self.dilation_w + 1
    }

    /// Shared memory bytes per block for the input tile cache.
    ///
    /// For non-pointwise convolutions: one tile of the input patch that
    /// the block needs, sized `(block_h * stride_h + effective_kH - stride_h)
    /// * (block_w * stride_w + effective_kW - stride_w)` floats.
    ///
    /// For 1x1 convolutions: 0 (no shared memory needed).
    #[must_use]
    pub fn shared_memory_bytes(&self) -> usize {
        if self.is_pointwise() {
            return 0;
        }
        let tile_h = self.input_tile_h();
        let tile_w = self.input_tile_w();
        tile_h * tile_w * 4 // f32 = 4 bytes
    }

    /// Threads per block.
    #[must_use]
    pub fn threads_per_block(&self) -> usize {
        self.block_h * self.block_w
    }

    /// Height of the input tile loaded into shared memory.
    #[must_use]
    pub fn input_tile_h(&self) -> usize {
        (self.block_h - 1) * self.stride_h + self.effective_kernel_h()
    }

    /// Width of the input tile loaded into shared memory.
    #[must_use]
    pub fn input_tile_w(&self) -> usize {
        (self.block_w - 1) * self.stride_w + self.effective_kernel_w()
    }

    /// Number of input channels per group.
    ///
    /// Returns 0 if `in_channels` is 0 (runtime channels mode).
    #[must_use]
    pub fn in_channels_per_group(&self) -> usize {
        if self.in_channels == 0 || self.groups == 0 {
            return 0;
        }
        self.in_channels / self.groups
    }

    /// Number of output channels per group.
    ///
    /// Returns 0 if `out_channels` is 0 (runtime channels mode).
    #[must_use]
    pub fn out_channels_per_group(&self) -> usize {
        if self.out_channels == 0 || self.groups == 0 {
            return 0;
        }
        self.out_channels / self.groups
    }

    /// Whether groups mode is active (groups > 1 with channels specified).
    #[must_use]
    pub fn has_groups(&self) -> bool {
        self.groups > 1
    }

    /// Whether this is a depthwise convolution
    /// (groups == in_channels == out_channels).
    #[must_use]
    pub fn is_depthwise(&self) -> bool {
        self.groups > 1
            && self.in_channels > 0
            && self.in_channels == self.out_channels
            && self.in_channels == self.groups
    }
}

impl Default for PtxConv2dConfig {
    fn default() -> Self {
        Self::new("ptx_conv2d_f32", 3, 3)
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Emit a complete PTX module for f32 2D convolution (NCHW layout).
///
/// Generates raw PTX assembly implementing:
///   `out[n, oc, oh, ow] = sum_{ic, kh, kw} input[n, ic, ih, iw] * weight[oc, ic, kh, kw] + bias[oc]`
///
/// where `ih = oh * stride_h - pad_h + kh * dilation_h` and similarly for `iw`.
///
/// # Parameters (kernel arguments)
///
/// - `input`: `[N, C_in, H_in, W_in]` f32 tensor pointer
/// - `weight`: `[C_out, C_in, kH, kW]` f32 tensor pointer
/// - `bias`: `[C_out]` f32 pointer (ignored if `use_bias` is false)
/// - `output`: `[N, C_out, H_out, W_out]` f32 tensor pointer
/// - `N, C_in, H_in, W_in, C_out, H_out, W_out`: dimension scalars (u32)
///
/// # Returns
///
/// Complete PTX module string ready for `cuModuleLoadData` or `ptxas`.
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_conv2d::{emit_ptx_conv2d, PtxConv2dConfig};
/// let config = PtxConv2dConfig::new("conv2d_3x3", 3, 3)
///     .with_padding(1, 1);
/// let ptx = emit_ptx_conv2d(&config).unwrap();
/// assert!(ptx.contains(".entry conv2d_3x3"));
/// assert!(ptx.contains(".shared .align 4"));
/// ```
pub fn emit_ptx_conv2d(config: &PtxConv2dConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    if config.is_pointwise() {
        emit_ptx_conv2d_pointwise(config)
    } else {
        emit_ptx_conv2d_general(config)
    }
}

/// Emit the optimized 1x1 pointwise convolution kernel (no shared memory).
fn emit_ptx_conv2d_pointwise(config: &PtxConv2dConfig) -> Result<String, PtxCodegenError> {
    let name = &config.kernel_name;
    let zero = format_ptx_float(0.0);
    let block_w = config.block_w;
    let block_h = config.block_h;

    let mut ptx = String::with_capacity(8192);

    // -- Module header --
    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// 1x1 pointwise conv2d (NCHW), no shared memory\n\
         // stride: {}x{}, pad: {}x{}\n\n",
        config.stride_h, config.stride_w, config.pad_h, config.pad_w
    ));

    // -- Kernel entry point --
    emit_kernel_signature(&mut ptx, name, config.use_bias);
    ptx.push_str(&format!(".reqntid {block_w}, {block_h}\n{{\n"));

    // -- Register declarations --
    ptx.push_str(
        "    // Register declarations\n\
         \x20   .reg .u32  %r<32>;\n\
         \x20   .reg .f32  %f<8>;\n\
         \x20   .reg .u64  %rd<12>;\n\
         \x20   .reg .pred %p<6>;\n\n",
    );

    // -- Load parameters --
    emit_load_params(&mut ptx, config.use_bias);

    // -- Compute output coordinates --
    // ow = blockIdx.x * block_w + threadIdx.x
    // oh = blockIdx.y * block_h + threadIdx.y
    // Linear z = blockIdx.z → n = z / C_out, oc = z % C_out
    ptx.push_str(&format!(
        "    // Output coordinates\n\
         \x20   mov.u32       %r10, %tid.x;\n\
         \x20   mov.u32       %r11, %tid.y;\n\
         \x20   mov.u32       %r12, %ctaid.x;\n\
         \x20   mov.u32       %r13, %ctaid.y;\n\
         \x20   mov.u32       %r14, %ctaid.z;        // z = n * C_out + oc\n\
         \x20   mad.lo.u32    %r15, %r12, {block_w}, %r10; // ow\n\
         \x20   mad.lo.u32    %r16, %r13, {block_h}, %r11; // oh\n\
         \x20   div.u32       %r17, %r14, %r4;       // n = z / C_out\n\
         \x20   rem.u32       %r18, %r14, %r4;       // oc = z % C_out\n\n"
    ));

    // -- Bounds check: oh < H_out && ow < W_out --
    ptx.push_str(
        "    // Bounds check\n\
         \x20   setp.lt.u32   %p0, %r16, %r5;        // oh < H_out?\n\
         \x20   setp.lt.u32   %p1, %r15, %r6;        // ow < W_out?\n\
         \x20   and.pred       %p2, %p0, %p1;\n\
         \x20   @!%p2 bra     KERNEL_EXIT;\n\n",
    );

    // -- Input coordinates for 1x1: ih = oh * stride_h - pad_h, iw = ow * stride_w - pad_w --
    // For 1x1 with pad=0, stride=1 this simplifies to ih=oh, iw=ow
    ptx.push_str(&format!(
        "    // Input coordinates (1x1 pointwise)\n\
         \x20   mul.lo.u32    %r19, %r16, {stride_h}; // oh * stride_h\n\
         \x20   sub.u32       %r19, %r19, {pad_h};    // ih = oh * stride_h - pad_h\n\
         \x20   mul.lo.u32    %r20, %r15, {stride_w}; // ow * stride_w\n\
         \x20   sub.u32       %r20, %r20, {pad_w};    // iw = ow * stride_w - pad_w\n\n",
        stride_h = config.stride_h,
        stride_w = config.stride_w,
        pad_h = config.pad_h,
        pad_w = config.pad_w,
    ));

    // -- Accumulator --
    ptx.push_str(&format!(
        "    // Initialize accumulator\n\
         \x20   mov.f32       %f0, {zero};\n\n"
    ));

    // -- Loop over input channels --
    ptx.push_str(
        "    // Loop over C_in\n\
         \x20   mov.u32       %r21, 0;               // ic = 0\n\
         IC_LOOP:\n\
         \x20   setp.ge.u32   %p3, %r21, %r1;       // ic >= C_in?\n\
         \x20   @%p3 bra      IC_DONE;\n\n",
    );

    // Bounds check ih, iw (may be out of range if pad > 0 with stride > 1)
    ptx.push_str(
        "    // Bounds check ih, iw\n\
         \x20   setp.lt.u32   %p4, %r19, %r2;        // ih < H_in? (unsigned: negative wraps)\n\
         \x20   setp.lt.u32   %p5, %r20, %r3;        // iw < W_in?\n\
         \x20   and.pred       %p4, %p4, %p5;\n\
         \x20   @!%p4 bra     IC_NEXT;               // skip if out of bounds\n\n",
    );

    // -- Load input[n, ic, ih, iw] --
    // offset = ((n * C_in + ic) * H_in + ih) * W_in + iw
    ptx.push_str(
        "    // input[n, ic, ih, iw]\n\
         \x20   mad.lo.u32    %r22, %r17, %r1, %r21; // n * C_in + ic\n\
         \x20   mad.lo.u32    %r22, %r22, %r2, %r19; // * H_in + ih\n\
         \x20   mad.lo.u32    %r22, %r22, %r3, %r20; // * W_in + iw\n\
         \x20   mul.wide.u32  %rd6, %r22, 4;\n\
         \x20   add.u64       %rd7, %rd0, %rd6;\n\
         \x20   ld.global.f32 %f1, [%rd7];\n\n",
    );

    // -- Load weight[oc, ic, 0, 0] --
    // offset = (oc * C_in + ic) * kH * kW = (oc * C_in + ic) for 1x1
    ptx.push_str(
        "    // weight[oc, ic, 0, 0]\n\
         \x20   mad.lo.u32    %r23, %r18, %r1, %r21; // oc * C_in + ic\n\
         \x20   mul.wide.u32  %rd6, %r23, 4;\n\
         \x20   add.u64       %rd7, %rd1, %rd6;\n\
         \x20   ld.global.f32 %f2, [%rd7];\n\n",
    );

    // -- FMA --
    ptx.push_str(
        "    // acc += input * weight\n\
         \x20   fma.rn.f32    %f0, %f1, %f2, %f0;\n\n",
    );

    // IC_NEXT / loop end
    ptx.push_str(
        "IC_NEXT:\n\
         \x20   add.u32       %r21, %r21, 1;\n\
         \x20   bra           IC_LOOP;\n\
         IC_DONE:\n\n",
    );

    // -- Add bias if enabled --
    if config.use_bias {
        emit_bias_add(&mut ptx);
    }

    // -- Store output --
    emit_store_output(&mut ptx);

    ptx.push_str("KERNEL_EXIT:\n    ret;\n}\n");

    Ok(ptx)
}

/// Emit the general conv2d kernel with shared memory input tile caching.
fn emit_ptx_conv2d_general(config: &PtxConv2dConfig) -> Result<String, PtxCodegenError> {
    let name = &config.kernel_name;
    let zero = format_ptx_float(0.0);
    let block_w = config.block_w;
    let block_h = config.block_h;
    let kh = config.kernel_h;
    let kw = config.kernel_w;
    let tile_h = config.input_tile_h();
    let tile_w = config.input_tile_w();
    let tile_size = tile_h * tile_w;

    let mut ptx = String::with_capacity(16384);

    // -- Module header --
    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// Conv2d (NCHW) with shared memory tile caching\n\
         // kernel: {kh}x{kw}, stride: {}x{}, pad: {}x{}, dilation: {}x{}\n\
         // Input tile: {tile_h}x{tile_w}, shared memory: {} bytes\n\n",
        config.stride_h,
        config.stride_w,
        config.pad_h,
        config.pad_w,
        config.dilation_h,
        config.dilation_w,
        config.shared_memory_bytes()
    ));

    // -- Shared memory declaration --
    ptx.push_str(&format!(
        ".shared .align 4 .f32 input_tile[{tile_size}];\n\n"
    ));

    // -- Kernel entry point --
    emit_kernel_signature(&mut ptx, name, config.use_bias);
    ptx.push_str(&format!(".reqntid {block_w}, {block_h}\n{{\n"));

    // -- Register declarations --
    ptx.push_str(
        "    // Register declarations\n\
         \x20   .reg .u32  %r<32>;\n\
         \x20   .reg .f32  %f<8>;\n\
         \x20   .reg .u64  %rd<12>;\n\
         \x20   .reg .pred %p<6>;\n\n",
    );

    // -- Load parameters --
    emit_load_params(&mut ptx, config.use_bias);

    // -- Compute output coordinates --
    ptx.push_str(&format!(
        "    // Output coordinates\n\
         \x20   mov.u32       %r10, %tid.x;           // tx\n\
         \x20   mov.u32       %r11, %tid.y;           // ty\n\
         \x20   mov.u32       %r12, %ctaid.x;\n\
         \x20   mov.u32       %r13, %ctaid.y;\n\
         \x20   mov.u32       %r14, %ctaid.z;         // z = n * C_out + oc\n\
         \x20   mad.lo.u32    %r15, %r12, {block_w}, %r10; // ow\n\
         \x20   mad.lo.u32    %r16, %r13, {block_h}, %r11; // oh\n\
         \x20   div.u32       %r17, %r14, %r4;        // n = z / C_out\n\
         \x20   rem.u32       %r18, %r14, %r4;        // oc = z % C_out\n\n"
    ));

    // -- Bounds check: oh < H_out && ow < W_out --
    // We still run the thread for shared memory loading, but skip the store.
    // Use a predicate to track if this thread has a valid output pixel.
    ptx.push_str(
        "    // Output bounds check (stored for later)\n\
         \x20   setp.lt.u32   %p0, %r16, %r5;        // oh < H_out?\n\
         \x20   setp.lt.u32   %p1, %r15, %r6;        // ow < W_out?\n\
         \x20   and.pred       %p2, %p0, %p1;\n\n",
    );

    // -- Accumulator --
    ptx.push_str(&format!(
        "    // Initialize accumulator\n\
         \x20   mov.f32       %f0, {zero};\n\n"
    ));

    // -- Loop over input channels --
    ptx.push_str(
        "    // Loop over C_in\n\
         \x20   mov.u32       %r21, 0;                // ic = 0\n\
         IC_LOOP:\n\
         \x20   setp.ge.u32   %p3, %r21, %r1;        // ic >= C_in?\n\
         \x20   @%p3 bra      IC_DONE;\n\n",
    );

    // -- Load input tile into shared memory --
    // Each thread loads one or more elements of the input tile.
    // Tile origin in input: (oh_base * stride_h - pad_h, ow_base * stride_w - pad_w)
    // where oh_base = blockIdx.y * block_h, ow_base = blockIdx.x * block_w
    ptx.push_str(&format!(
        "    // Load input tile into shared memory\n\
         \x20   // tile origin: ih_base = blockIdx.y * block_h * stride_h - pad_h\n\
         \x20   mul.lo.u32    %r22, %r13, {bh_times_sh}; // blockIdx.y * block_h * stride_h\n\
         \x20   sub.u32       %r22, %r22, {pad_h};       // ih_base\n\
         \x20   mul.lo.u32    %r23, %r12, {bw_times_sw}; // blockIdx.x * block_w * stride_w\n\
         \x20   sub.u32       %r23, %r23, {pad_w};       // iw_base\n\n",
        bh_times_sh = block_h * config.stride_h,
        bw_times_sw = block_w * config.stride_w,
        pad_h = config.pad_h,
        pad_w = config.pad_w,
    ));

    // -- Cooperative tile loading --
    // Thread linear index: tid = ty * block_w + tx
    // Each thread loads elements at stride = threads_per_block
    let threads = config.threads_per_block();
    ptx.push_str(&format!(
        "    // Cooperative tile load: {tile_size} elements, {threads} threads\n\
         \x20   mad.lo.u32    %r24, %r11, {block_w}, %r10; // tid = ty * block_w + tx\n\
         \x20   mov.u32       %r25, %r24;             // load_idx = tid\n\
         TILE_LOAD_LOOP:\n\
         \x20   setp.ge.u32   %p4, %r25, {tile_size}; // load_idx >= tile_size?\n\
         \x20   @%p4 bra      TILE_LOAD_DONE;\n\
         \x20   // Compute tile row/col from linear index\n\
         \x20   div.u32       %r26, %r25, {tile_w};   // tile_row\n\
         \x20   rem.u32       %r27, %r25, {tile_w};   // tile_col\n\
         \x20   // Map to input coordinates\n\
         \x20   add.u32       %r28, %r22, %r26;       // ih = ih_base + tile_row\n\
         \x20   add.u32       %r29, %r23, %r27;       // iw = iw_base + tile_col\n\
         \x20   // Bounds check (unsigned comparison handles negative via wrap)\n\
         \x20   setp.lt.u32   %p4, %r28, %r2;         // ih < H_in?\n\
         \x20   setp.lt.u32   %p5, %r29, %r3;         // iw < W_in?\n\
         \x20   and.pred       %p4, %p4, %p5;\n\
         \x20   mov.f32       %f1, {zero};             // default = 0 (padding)\n\
         \x20   @!%p4 bra     SKIP_TILE_LOAD;\n\
         \x20   // input[n, ic, ih, iw]\n\
         \x20   mad.lo.u32    %r30, %r17, %r1, %r21;  // n * C_in + ic\n\
         \x20   mad.lo.u32    %r30, %r30, %r2, %r28;  // * H_in + ih\n\
         \x20   mad.lo.u32    %r30, %r30, %r3, %r29;  // * W_in + iw\n\
         \x20   mul.wide.u32  %rd6, %r30, 4;\n\
         \x20   add.u64       %rd7, %rd0, %rd6;\n\
         \x20   ld.global.f32 %f1, [%rd7];\n\
         SKIP_TILE_LOAD:\n\
         \x20   // Store to shared memory: input_tile[load_idx]\n\
         \x20   mul.wide.u32  %rd8, %r25, 4;\n\
         \x20   mov.u64       %rd9, input_tile;\n\
         \x20   add.u64       %rd10, %rd9, %rd8;\n\
         \x20   st.shared.f32 [%rd10], %f1;\n\
         \x20   // Advance load index\n\
         \x20   add.u32       %r25, %r25, {threads};\n\
         \x20   bra           TILE_LOAD_LOOP;\n\
         TILE_LOAD_DONE:\n\n"
    ));

    // -- Barrier after tile load --
    ptx.push_str(
        "    // Synchronize after tile load\n\
         \x20   bar.sync      0;\n\n",
    );

    // -- Convolution: loop over kH, kW using shared memory tile --
    // For this thread's output pixel (oh, ow):
    //   local_row = ty * stride_h + kh * dilation_h
    //   local_col = tx * stride_w + kw * dilation_w
    //   val = input_tile[local_row * tile_w + local_col]
    ptx.push_str(&format!(
        "    // Convolution kernel loop (kH={kh}, kW={kw})\n\
         \x20   mov.u32       %r25, 0;                // kh_idx = 0\n\
         KH_LOOP:\n\
         \x20   setp.ge.u32   %p4, %r25, {kh};\n\
         \x20   @%p4 bra      KH_DONE;\n\
         \x20   mov.u32       %r26, 0;                // kw_idx = 0\n\
         KW_LOOP:\n\
         \x20   setp.ge.u32   %p4, %r26, {kw};\n\
         \x20   @%p4 bra      KW_DONE;\n\n"
    ));

    // Compute shared memory index for input value
    ptx.push_str(&format!(
        "    // Shared mem index: (ty * stride_h + kh * dilation_h) * tile_w + (tx * stride_w + kw * dilation_w)\n\
         \x20   mul.lo.u32    %r27, %r11, {stride_h}; // ty * stride_h\n\
         \x20   mul.lo.u32    %r28, %r25, {dilation_h}; // kh * dilation_h\n\
         \x20   add.u32       %r27, %r27, %r28;       // local_row\n\
         \x20   mul.lo.u32    %r28, %r10, {stride_w}; // tx * stride_w\n\
         \x20   mul.lo.u32    %r29, %r26, {dilation_w}; // kw * dilation_w\n\
         \x20   add.u32       %r28, %r28, %r29;       // local_col\n\
         \x20   mad.lo.u32    %r30, %r27, {tile_w}, %r28; // local_row * tile_w + local_col\n\
         \x20   mul.wide.u32  %rd6, %r30, 4;\n\
         \x20   mov.u64       %rd7, input_tile;\n\
         \x20   add.u64       %rd8, %rd7, %rd6;\n\
         \x20   ld.shared.f32 %f1, [%rd8];\n\n",
        stride_h = config.stride_h,
        stride_w = config.stride_w,
        dilation_h = config.dilation_h,
        dilation_w = config.dilation_w,
    ));

    // Load weight[oc, ic, kh, kw]
    // offset = ((oc * C_in + ic) * kH + kh) * kW + kw
    ptx.push_str(&format!(
        "    // weight[oc, ic, kh, kw]\n\
         \x20   mad.lo.u32    %r30, %r18, %r1, %r21;  // oc * C_in + ic\n\
         \x20   mad.lo.u32    %r30, %r30, {kh}, %r25;  // * kH + kh\n\
         \x20   mad.lo.u32    %r30, %r30, {kw}, %r26;  // * kW + kw\n\
         \x20   mul.wide.u32  %rd6, %r30, 4;\n\
         \x20   add.u64       %rd7, %rd1, %rd6;\n\
         \x20   ld.global.f32 %f2, [%rd7];\n\n"
    ));

    // FMA
    ptx.push_str(
        "    // acc += input_tile_val * weight\n\
         \x20   fma.rn.f32    %f0, %f1, %f2, %f0;\n\n",
    );

    // Close kw/kh loops
    ptx.push_str(
        "    // Next kw\n\
         \x20   add.u32       %r26, %r26, 1;\n\
         \x20   bra           KW_LOOP;\n\
         KW_DONE:\n\
         \x20   add.u32       %r25, %r25, 1;\n\
         \x20   bra           KH_LOOP;\n\
         KH_DONE:\n\n",
    );

    // -- Barrier before next channel iteration (protects shared memory) --
    ptx.push_str(
        "    // Synchronize before next channel tile\n\
         \x20   bar.sync      0;\n\n",
    );

    // Close ic loop
    ptx.push_str(
        "    // Next ic\n\
         \x20   add.u32       %r21, %r21, 1;\n\
         \x20   bra           IC_LOOP;\n\
         IC_DONE:\n\n",
    );

    // -- Bounds check for store --
    ptx.push_str(
        "    // Only store if output pixel is valid\n\
         \x20   @!%p2 bra     KERNEL_EXIT;\n\n",
    );

    // -- Bias --
    if config.use_bias {
        emit_bias_add(&mut ptx);
    }

    // -- Store output --
    emit_store_output(&mut ptx);

    ptx.push_str("KERNEL_EXIT:\n    ret;\n}\n");

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Emit the kernel signature (parameter list).
fn emit_kernel_signature(ptx: &mut String, name: &str, use_bias: bool) {
    ptx.push_str(&format!(".visible .entry {name}(\n"));
    ptx.push_str(
        "    .param .u64 param_input,\n\
         \x20   .param .u64 param_weight,\n",
    );
    if use_bias {
        ptx.push_str("    .param .u64 param_bias,\n");
    }
    ptx.push_str(
        "    .param .u64 param_output,\n\
         \x20   .param .u32 param_N,\n\
         \x20   .param .u32 param_C_in,\n\
         \x20   .param .u32 param_H_in,\n\
         \x20   .param .u32 param_W_in,\n\
         \x20   .param .u32 param_C_out,\n\
         \x20   .param .u32 param_H_out,\n\
         \x20   .param .u32 param_W_out\n\
         )\n",
    );
}

/// Emit parameter loads into registers.
///
/// Register allocation:
/// - `%rd0` = input ptr, `%rd1` = weight ptr, `%rd2` = bias ptr (if used), `%rd3` = output ptr
/// - `%r0` = N, `%r1` = C_in, `%r2` = H_in, `%r3` = W_in, `%r4` = C_out, `%r5` = H_out, `%r6` = W_out
fn emit_load_params(ptx: &mut String, use_bias: bool) {
    ptx.push_str(
        "    // Load kernel parameters\n\
         \x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_weight];\n",
    );
    if use_bias {
        ptx.push_str("    ld.param.u64  %rd2, [param_bias];\n");
    }
    ptx.push_str(
        "    ld.param.u64  %rd3, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_N];\n\
         \x20   ld.param.u32  %r1,  [param_C_in];\n\
         \x20   ld.param.u32  %r2,  [param_H_in];\n\
         \x20   ld.param.u32  %r3,  [param_W_in];\n\
         \x20   ld.param.u32  %r4,  [param_C_out];\n\
         \x20   ld.param.u32  %r5,  [param_H_out];\n\
         \x20   ld.param.u32  %r6,  [param_W_out];\n\n",
    );
}

/// Emit bias addition: `acc += bias[oc]`.
fn emit_bias_add(ptx: &mut String) {
    ptx.push_str(
        "    // Add bias[oc]\n\
         \x20   mul.wide.u32  %rd6, %r18, 4;          // oc * sizeof(f32)\n\
         \x20   add.u64       %rd7, %rd2, %rd6;       // &bias[oc]\n\
         \x20   ld.global.f32 %f3, [%rd7];\n\
         \x20   add.f32       %f0, %f0, %f3;\n\n",
    );
}

/// Emit output store: `output[n, oc, oh, ow] = acc`.
fn emit_store_output(ptx: &mut String) {
    // output offset = ((n * C_out + oc) * H_out + oh) * W_out + ow
    // %r17 = n, %r18 = oc, %r16 = oh, %r15 = ow
    // %r4 = C_out, %r5 = H_out, %r6 = W_out
    ptx.push_str(
        "    // Store output[n, oc, oh, ow]\n\
         \x20   mad.lo.u32    %r30, %r17, %r4, %r18;  // n * C_out + oc\n\
         \x20   mad.lo.u32    %r30, %r30, %r5, %r16;  // * H_out + oh\n\
         \x20   mad.lo.u32    %r30, %r30, %r6, %r15;  // * W_out + ow\n\
         \x20   mul.wide.u32  %rd6, %r30, 4;\n\
         \x20   add.u64       %rd7, %rd3, %rd6;\n\
         \x20   st.global.f32 [%rd7], %f0;\n\n",
    );
}

// ---------------------------------------------------------------------------
// Launch config
// ---------------------------------------------------------------------------

/// Compute the grid and block dimensions for a PTX conv2d kernel.
///
/// Grid: `(ceil(W_out/block_w), ceil(H_out/block_h), N * C_out)`.
/// Block: `(block_w, block_h, 1)`.
///
/// # Returns
///
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_conv2d_launch_config(
    h_out: usize,
    w_out: usize,
    batch_size: usize,
    c_out: usize,
    config: &PtxConv2dConfig,
) -> ([usize; 3], [usize; 3]) {
    let grid = [
        w_out.div_ceil(config.block_w),
        h_out.div_ceil(config.block_h),
        batch_size * c_out,
    ];
    let block = [config.block_w, config.block_h, 1];
    (grid, block)
}

/// Convenience: emit PTX conv2d with default 3x3 kernel, pad=1, sm_80.
pub fn emit_ptx_conv2d_default(name: &str) -> Result<String, PtxCodegenError> {
    emit_ptx_conv2d(&PtxConv2dConfig::new(name, 3, 3).with_padding(1, 1))
}

// ---------------------------------------------------------------------------
// Output size calculation
// ---------------------------------------------------------------------------

/// Compute the output height or width for a 2D convolution dimension.
///
/// `dim_out = (dim_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
///
/// Returns `None` if the parameters would produce a non-positive output.
#[must_use]
pub fn conv2d_output_size(
    dim_in: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    let effective_k = dilation
        .checked_mul(kernel_size.checked_sub(1)?)?
        .checked_add(1)?;
    let padded = dim_in.checked_add(2 * padding)?;
    if padded < effective_k {
        return None;
    }
    let numerator = padded - effective_k;
    Some(numerator / stride + 1)
}

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// Compute 2D convolution on CPU for reference/testing.
///
/// Implements the standard conv2d with NCHW layout, supporting grouped
/// convolution (including depthwise when groups == in_channels == out_channels).
///
/// # Arguments
///
/// * `input` — Flat f32 slice of shape `[N, C_in, H_in, W_in]` (NCHW)
/// * `weight` — Flat f32 slice of shape `[C_out, C_in/groups, kH, kW]`
/// * `bias` — Optional flat f32 slice of shape `[C_out]`
/// * `config` — Convolution parameters (channels, kernel, stride, padding, dilation, groups)
/// * `batch_size` — N
/// * `h_in` — Input height
/// * `w_in` — Input width
///
/// # Returns
///
/// Flat f32 Vec of shape `[N, C_out, H_out, W_out]`.
///
/// # Panics
///
/// Panics if the configuration is invalid or output dimensions are zero.
pub fn conv2d_reference(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &PtxConv2dConfig,
    batch_size: usize,
    h_in: usize,
    w_in: usize,
) -> Vec<f32> {
    let c_in = config.in_channels;
    let c_out = config.out_channels;
    let kh = config.kernel_h;
    let kw = config.kernel_w;
    let stride_h = config.stride_h;
    let stride_w = config.stride_w;
    let pad_h = config.pad_h;
    let pad_w = config.pad_w;
    let dilation_h = config.dilation_h;
    let dilation_w = config.dilation_w;
    let groups = config.groups;

    assert!(
        c_in > 0 && c_out > 0,
        "channels must be specified for reference"
    );
    assert!(groups > 0, "groups must be > 0");
    assert_eq!(c_in % groups, 0, "in_channels must be divisible by groups");
    assert_eq!(
        c_out % groups,
        0,
        "out_channels must be divisible by groups"
    );

    let ic_per_group = c_in / groups;
    let oc_per_group = c_out / groups;

    let h_out =
        conv2d_output_size(h_in, kh, stride_h, pad_h, dilation_h).expect("invalid output height");
    let w_out =
        conv2d_output_size(w_in, kw, stride_w, pad_w, dilation_w).expect("invalid output width");

    assert_eq!(input.len(), batch_size * c_in * h_in * w_in);
    assert_eq!(weight.len(), c_out * ic_per_group * kh * kw);
    if let Some(b) = bias {
        assert_eq!(b.len(), c_out);
    }

    let mut output = vec![0.0f32; batch_size * c_out * h_out * w_out];

    for n in 0..batch_size {
        for oc in 0..c_out {
            let group_idx = oc / oc_per_group;
            let ic_start = group_idx * ic_per_group;
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0.0f32;
                    for ic in 0..ic_per_group {
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * stride_h + khi * dilation_h;
                                let iw = ow * stride_w + kwi * dilation_w;
                                // Account for padding
                                let ih = ih as isize - pad_h as isize;
                                let iw = iw as isize - pad_w as isize;
                                if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let in_idx =
                                        ((n * c_in + ic_start + ic) * h_in + ih) * w_in + iw;
                                    let w_idx = ((oc * ic_per_group + ic) * kh + khi) * kw + kwi;
                                    acc += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    if let Some(b) = bias {
                        acc += b[oc];
                    }
                    let out_idx = ((n * c_out + oc) * h_out + oh) * w_out + ow;
                    output[out_idx] = acc;
                }
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_conv2d_tests.rs"]
mod ptx_conv2d_tests;

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config validation --

    #[test]
    fn test_config_default() {
        let c = PtxConv2dConfig::default();
        assert_eq!(c.kernel_h, 3);
        assert_eq!(c.kernel_w, 3);
        assert_eq!(c.stride_h, 1);
        assert_eq!(c.stride_w, 1);
        assert_eq!(c.pad_h, 0);
        assert_eq!(c.pad_w, 0);
        assert_eq!(c.dilation_h, 1);
        assert_eq!(c.dilation_w, 1);
        assert_eq!(c.block_h, 16);
        assert_eq!(c.block_w, 16);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_config_zero_kernel_size() {
        let c = PtxConv2dConfig::new("k", 0, 3);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_stride() {
        let c = PtxConv2dConfig::new("k", 3, 3).with_stride(0, 1);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_dilation() {
        let c = PtxConv2dConfig::new("k", 3, 3).with_dilation(0, 1);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_empty_name() {
        let c = PtxConv2dConfig::new("", 3, 3);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_block_too_small() {
        let c = PtxConv2dConfig::new("k", 3, 3).with_block_size(2, 16);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_block_too_large() {
        let c = PtxConv2dConfig::new("k", 3, 3).with_block_size(64, 16);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_threads_exceed_1024() {
        // 32 * 33 = 1056 > 1024. Use a valid block dim that still exceeds limit.
        // Since max block dim is 32, we need 32x32=1024 which is exactly OK.
        // So this test verifies that 1024 is accepted.
        let c = PtxConv2dConfig::new("k", 3, 3).with_block_size(32, 32);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_config_is_pointwise() {
        assert!(PtxConv2dConfig::new("k", 1, 1).is_pointwise());
        assert!(!PtxConv2dConfig::new("k", 3, 3).is_pointwise());
        assert!(!PtxConv2dConfig::new("k", 1, 3).is_pointwise());
    }

    #[test]
    fn test_config_effective_kernel_size() {
        let c = PtxConv2dConfig::new("k", 3, 3).with_dilation(2, 2);
        assert_eq!(c.effective_kernel_h(), 5); // (3-1)*2+1
        assert_eq!(c.effective_kernel_w(), 5);
    }

    #[test]
    fn test_config_shared_memory_bytes_3x3() {
        let c = PtxConv2dConfig::new("k", 3, 3);
        // input_tile_h = (16-1)*1 + 3 = 18
        // input_tile_w = (16-1)*1 + 3 = 18
        // bytes = 18 * 18 * 4 = 1296
        assert_eq!(c.input_tile_h(), 18);
        assert_eq!(c.input_tile_w(), 18);
        assert_eq!(c.shared_memory_bytes(), 18 * 18 * 4);
    }

    #[test]
    fn test_config_shared_memory_bytes_pointwise() {
        let c = PtxConv2dConfig::new("k", 1, 1);
        assert_eq!(c.shared_memory_bytes(), 0);
    }

    #[test]
    fn test_config_shared_memory_with_stride() {
        let c = PtxConv2dConfig::new("k", 3, 3).with_stride(2, 2);
        // input_tile_h = (16-1)*2 + 3 = 33
        // input_tile_w = (16-1)*2 + 3 = 33
        assert_eq!(c.input_tile_h(), 33);
        assert_eq!(c.input_tile_w(), 33);
        assert_eq!(c.shared_memory_bytes(), 33 * 33 * 4);
    }

    // -- PTX generation: structural checks for general (3x3) --

    #[test]
    fn test_ptx_conv2d_3x3_contains_version_and_target() {
        let ptx = emit_ptx_conv2d_default("conv2d_f32").unwrap();
        assert!(ptx.contains(".version 6.5"));
        assert!(ptx.contains(".target sm_80"));
        assert!(ptx.contains(".address_size 64"));
    }

    #[test]
    fn test_ptx_conv2d_3x3_contains_entry_point() {
        let ptx = emit_ptx_conv2d_default("nn_conv").unwrap();
        assert!(ptx.contains(".visible .entry nn_conv"));
    }

    #[test]
    fn test_ptx_conv2d_3x3_contains_params() {
        let ptx = emit_ptx_conv2d_default("conv").unwrap();
        assert!(ptx.contains("param_input"));
        assert!(ptx.contains("param_weight"));
        assert!(ptx.contains("param_output"));
        assert!(ptx.contains("param_N"));
        assert!(ptx.contains("param_C_in"));
        assert!(ptx.contains("param_H_in"));
        assert!(ptx.contains("param_W_in"));
        assert!(ptx.contains("param_C_out"));
        assert!(ptx.contains("param_H_out"));
        assert!(ptx.contains("param_W_out"));
    }

    #[test]
    fn test_ptx_conv2d_3x3_contains_shared_memory() {
        let ptx = emit_ptx_conv2d_default("conv").unwrap();
        assert!(
            ptx.contains(".shared .align 4 .f32 input_tile["),
            "must declare shared memory for input tile"
        );
    }

    #[test]
    fn test_ptx_conv2d_shared_memory_size_matches_tile() {
        // 3x3 with pad=1, block=16x16: tile_h=18, tile_w=18, size=324
        let ptx = emit_ptx_conv2d_default("conv").unwrap();
        assert!(
            ptx.contains("input_tile[324]"),
            "3x3 pad=1 block=16 should give input_tile[324]"
        );

        // 5x5 with pad=0, block=16x16: tile_h=20, tile_w=20, size=400
        let c5 = PtxConv2dConfig::new("conv5", 5, 5);
        let ptx5 = emit_ptx_conv2d(&c5).unwrap();
        assert!(
            ptx5.contains("input_tile[400]"),
            "5x5 pad=0 block=16 should give input_tile[400]"
        );
    }

    #[test]
    fn test_ptx_conv2d_3x3_contains_barrier() {
        let ptx = emit_ptx_conv2d_default("conv").unwrap();
        let bar_count = ptx.matches("bar.sync").count();
        assert!(
            bar_count >= 2,
            "must have at least 2 barriers (after tile load, before next channel), got {bar_count}"
        );
    }

    #[test]
    fn test_ptx_conv2d_3x3_contains_fma() {
        let ptx = emit_ptx_conv2d_default("conv").unwrap();
        assert!(ptx.contains("fma.rn.f32"));
    }

    #[test]
    fn test_ptx_conv2d_3x3_contains_global_loads_stores() {
        let ptx = emit_ptx_conv2d_default("conv").unwrap();
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
    }

    #[test]
    fn test_ptx_conv2d_3x3_contains_shared_loads_stores() {
        let ptx = emit_ptx_conv2d_default("conv").unwrap();
        assert!(ptx.contains("ld.shared.f32"));
        assert!(ptx.contains("st.shared.f32"));
    }

    #[test]
    fn test_ptx_conv2d_3x3_contains_reqntid() {
        let ptx = emit_ptx_conv2d_default("conv").unwrap();
        assert!(ptx.contains(".reqntid 16, 16"));
    }

    #[test]
    fn test_ptx_conv2d_ret_at_end() {
        let ptx = emit_ptx_conv2d_default("conv").unwrap();
        assert!(ptx.contains("ret;"));
    }

    // -- Different kernel sizes produce different PTX --

    #[test]
    fn test_different_kernel_sizes_produce_different_ptx() {
        let ptx_3x3 = emit_ptx_conv2d(&PtxConv2dConfig::new("conv", 3, 3)).unwrap();
        let ptx_5x5 = emit_ptx_conv2d(&PtxConv2dConfig::new("conv", 5, 5)).unwrap();
        let ptx_7x7 = emit_ptx_conv2d(&PtxConv2dConfig::new("conv", 7, 7)).unwrap();
        assert_ne!(ptx_3x3, ptx_5x5, "3x3 and 5x5 must differ");
        assert_ne!(ptx_3x3, ptx_7x7, "3x3 and 7x7 must differ");
        assert_ne!(ptx_5x5, ptx_7x7, "5x5 and 7x7 must differ");
    }

    #[test]
    fn test_different_strides_produce_different_ptx() {
        let ptx_s1 =
            emit_ptx_conv2d(&PtxConv2dConfig::new("conv", 3, 3).with_stride(1, 1)).unwrap();
        let ptx_s2 =
            emit_ptx_conv2d(&PtxConv2dConfig::new("conv", 3, 3).with_stride(2, 2)).unwrap();
        assert_ne!(ptx_s1, ptx_s2, "stride=1 and stride=2 must differ");
    }

    // -- 1x1 pointwise optimization --

    #[test]
    fn test_ptx_conv2d_1x1_no_shared_memory() {
        let c = PtxConv2dConfig::new("conv1x1", 1, 1);
        let ptx = emit_ptx_conv2d(&c).unwrap();
        assert!(
            !ptx.contains(".shared"),
            "1x1 conv must not declare shared memory"
        );
        assert!(
            !ptx.contains("input_tile"),
            "1x1 conv must not use input_tile"
        );
    }

    #[test]
    fn test_ptx_conv2d_1x1_still_valid_kernel() {
        let c = PtxConv2dConfig::new("conv1x1", 1, 1);
        let ptx = emit_ptx_conv2d(&c).unwrap();
        assert!(ptx.contains(".visible .entry conv1x1"));
        assert!(ptx.contains(".version 6.5"));
        assert!(ptx.contains("param_input"));
        assert!(ptx.contains("param_weight"));
        assert!(ptx.contains("param_output"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_ptx_conv2d_1x1_no_barrier() {
        let c = PtxConv2dConfig::new("conv1x1", 1, 1);
        let ptx = emit_ptx_conv2d(&c).unwrap();
        assert!(
            !ptx.contains("bar.sync"),
            "1x1 conv should not need barriers (no shared memory)"
        );
    }

    #[test]
    fn test_ptx_conv2d_1x1_differs_from_3x3() {
        let ptx_1x1 = emit_ptx_conv2d(&PtxConv2dConfig::new("conv", 1, 1)).unwrap();
        let ptx_3x3 = emit_ptx_conv2d(&PtxConv2dConfig::new("conv", 3, 3)).unwrap();
        assert_ne!(ptx_1x1, ptx_3x3);
    }

    // -- Bias --

    #[test]
    fn test_ptx_conv2d_with_bias() {
        let c = PtxConv2dConfig::new("conv_bias", 3, 3)
            .with_padding(1, 1)
            .with_bias(true);
        let ptx = emit_ptx_conv2d(&c).unwrap();
        assert!(ptx.contains("param_bias"), "must have bias parameter");
        assert!(
            ptx.contains("ld.param.u64  %rd2, [param_bias]"),
            "must load bias pointer"
        );
    }

    #[test]
    fn test_ptx_conv2d_without_bias() {
        let c = PtxConv2dConfig::new("conv_nobias", 3, 3).with_bias(false);
        let ptx = emit_ptx_conv2d(&c).unwrap();
        assert!(
            !ptx.contains("param_bias"),
            "must not have bias parameter when disabled"
        );
    }

    // -- Dilation --

    #[test]
    fn test_ptx_conv2d_with_dilation() {
        let c = PtxConv2dConfig::new("conv_dilated", 3, 3).with_dilation(2, 2);
        let ptx = emit_ptx_conv2d(&c).unwrap();
        // Effective kernel = 5x5, tile_h = (16-1)*1+5 = 20, tile_w = 20
        assert!(ptx.contains("input_tile[400]"));
        assert!(ptx.contains("dilation"));
    }

    // -- Custom SM target --

    #[test]
    fn test_ptx_conv2d_custom_sm_target() {
        let c = PtxConv2dConfig::new("conv_sm90", 3, 3).with_sm_target("sm_90");
        let ptx = emit_ptx_conv2d(&c).unwrap();
        assert!(ptx.contains(".target sm_90"));
    }

    // -- Custom block size --

    #[test]
    fn test_ptx_conv2d_custom_block_size() {
        let c = PtxConv2dConfig::new("conv_8x8", 3, 3).with_block_size(8, 8);
        let ptx = emit_ptx_conv2d(&c).unwrap();
        assert!(ptx.contains(".reqntid 8, 8"));
        // tile_h = (8-1)*1 + 3 = 10, tile_w = 10, size = 100
        assert!(ptx.contains("input_tile[100]"));
    }

    // -- Launch config --

    #[test]
    fn test_launch_config_basic() {
        let c = PtxConv2dConfig::new("conv", 3, 3);
        let (grid, block) = ptx_conv2d_launch_config(32, 32, 1, 64, &c);
        assert_eq!(grid, [2, 2, 64]); // ceil(32/16)=2, ceil(32/16)=2, 1*64
        assert_eq!(block, [16, 16, 1]);
    }

    #[test]
    fn test_launch_config_non_multiple() {
        let c = PtxConv2dConfig::new("conv", 3, 3);
        let (grid, block) = ptx_conv2d_launch_config(30, 50, 2, 32, &c);
        assert_eq!(grid, [4, 2, 64]); // ceil(50/16)=4, ceil(30/16)=2, 2*32=64
        assert_eq!(block, [16, 16, 1]);
    }

    #[test]
    fn test_launch_config_custom_block() {
        let c = PtxConv2dConfig::new("conv", 3, 3).with_block_size(8, 8);
        let (grid, block) = ptx_conv2d_launch_config(16, 16, 1, 3, &c);
        assert_eq!(grid, [2, 2, 3]); // ceil(16/8)=2, ceil(16/8)=2, 1*3
        assert_eq!(block, [8, 8, 1]);
    }

    // -- Instruction completeness --

    #[test]
    fn test_ptx_conv2d_instruction_set_coverage() {
        let ptx = emit_ptx_conv2d_default("conv").unwrap();
        let expected = [
            "ld.param",
            "mov.u32",
            "mad.lo.u32",
            "mul.wide.u32",
            "add.u64",
            "ld.global.f32",
            "st.global.f32",
            "ld.shared.f32",
            "st.shared.f32",
            "fma.rn.f32",
            "setp.lt.u32",
            "setp.ge.u32",
            "bar.sync",
            "bra",
            "ret",
        ];
        for instr in &expected {
            assert!(ptx.contains(instr), "PTX must contain instruction: {instr}");
        }
    }

    // -- Pure PTX, no CUDA C++ --

    #[test]
    fn test_ptx_conv2d_is_pure_ptx_not_cuda_cpp() {
        let ptx = emit_ptx_conv2d_default("conv").unwrap();
        assert!(!ptx.contains("__global__"));
        assert!(!ptx.contains("#include"));
        assert!(!ptx.contains("__shared__"));
        assert!(!ptx.contains("__syncthreads"));
    }

    // -- Convenience function --

    #[test]
    fn test_emit_ptx_conv2d_default_matches_config() {
        let default_ptx = emit_ptx_conv2d_default("test_conv").unwrap();
        let config_ptx =
            emit_ptx_conv2d(&PtxConv2dConfig::new("test_conv", 3, 3).with_padding(1, 1)).unwrap();
        assert_eq!(default_ptx, config_ptx);
    }
}
