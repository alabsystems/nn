// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for 1D convolution (NCL layout).
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 conv1d.
//! Unlike the CUDA C++ emission in [`ptx_emit`], this module emits PTX
//! assembly directly — no `nvcc` compilation step needed. The PTX can be
//! loaded via `cuModuleLoadData` (JIT) or assembled to cubin via `ptxas`.
//!
//! ## Algorithm
//!
//! Each thread computes one output element `out[n, oc, out_pos]`.
//! The kernel uses a grid-stride loop so a single launch can process
//! arbitrarily large output tensors. Grouped convolution is supported:
//! when `groups > 1`, each group of output channels only reads from its
//! corresponding group of input channels.
//!
//! ## Layout
//!
//! - Input:  `[N, C_in, L_in]` (NCL / channels-first 1D)
//! - Weight: `[C_out, C_in/groups, K]`
//! - Bias:   `[C_out]` (optional)
//! - Output: `[N, C_out, L_out]` (NCL)
//!
//! ## Output length formula
//!
//! `L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
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
//! Default: 256 threads (1D block). Each thread produces one output element.
//! Grid: `(ceil(total_output_elements / block_size), 1, 1)`.
//!
//! Parallel to Metal conv1d in `dyn_tensor_metal_ops_conv.rs` and
//! CPU SIMD conv1d in `nn-cpu`.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude, PtxCodegenError, DEFAULT_SM_TARGET};
use crate::cuda_ffi::{CudaDim3, CudaLaunchConfig};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for conv1d kernels (256 threads).
pub const PTX_CONV1D_BLOCK_SIZE: usize = 256;

/// Maximum supported kernel size for conv1d.
pub const PTX_CONV1D_MAX_KERNEL: usize = 1024;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for PTX conv1d kernel generation.
#[derive(Debug, Clone)]
pub struct PtxConv1dConfig {
    /// Kernel function name in the PTX module.
    pub kernel_name: String,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Convolution kernel size.
    pub kernel_size: usize,
    /// Stride along the length dimension.
    pub stride: usize,
    /// Symmetric padding applied to both ends.
    pub padding: usize,
    /// Dilation factor for the kernel.
    pub dilation: usize,
    /// Number of groups for grouped convolution.
    pub groups: usize,
    /// Whether to add bias after convolution.
    pub use_bias: bool,
    /// Thread block size (default: 256).
    pub block_size: usize,
    /// SM target for the PTX prelude (e.g., "sm_80").
    pub sm_target: String,
}

impl PtxConv1dConfig {
    /// Create a config with default stride=1, pad=0, dilation=1, groups=1,
    /// no bias, block_size=256, sm_80 target.
    pub fn new(
        kernel_name: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            use_bias: false,
            block_size: PTX_CONV1D_BLOCK_SIZE,
            sm_target: DEFAULT_SM_TARGET.to_string(),
        }
    }

    /// Set stride.
    #[must_use]
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set symmetric padding.
    #[must_use]
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Set dilation.
    #[must_use]
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set number of groups.
    #[must_use]
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// Enable bias addition.
    #[must_use]
    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Set the thread block size.
    #[must_use]
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
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
        if self.kernel_name.is_empty() {
            return Err(PtxCodegenError::InvalidParameter(
                "kernel_name must not be empty".into(),
            ));
        }
        if self.in_channels == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "in_channels must be > 0".into(),
            ));
        }
        if self.out_channels == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "out_channels must be > 0".into(),
            ));
        }
        if self.kernel_size == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "kernel_size must be > 0".into(),
            ));
        }
        if self.kernel_size > PTX_CONV1D_MAX_KERNEL {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "kernel_size {} exceeds maximum {}",
                self.kernel_size, PTX_CONV1D_MAX_KERNEL,
            )));
        }
        if self.stride == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "stride must be > 0".into(),
            ));
        }
        if self.dilation == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "dilation must be > 0".into(),
            ));
        }
        if self.groups == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "groups must be > 0".into(),
            ));
        }
        if !self.in_channels.is_multiple_of(self.groups) {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "in_channels ({}) must be divisible by groups ({})",
                self.in_channels, self.groups,
            )));
        }
        if !self.out_channels.is_multiple_of(self.groups) {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "out_channels ({}) must be divisible by groups ({})",
                self.out_channels, self.groups,
            )));
        }
        if self.block_size == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "block_size must be > 0".into(),
            ));
        }
        if self.block_size > 1024 {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "block_size must be <= 1024, got {}",
                self.block_size,
            )));
        }
        Ok(())
    }

    /// Effective kernel size accounting for dilation.
    #[must_use]
    pub fn effective_kernel_size(&self) -> usize {
        (self.kernel_size - 1) * self.dilation + 1
    }

    /// Number of input channels per group.
    #[must_use]
    pub fn in_channels_per_group(&self) -> usize {
        self.in_channels / self.groups
    }

    /// Number of output channels per group.
    #[must_use]
    pub fn out_channels_per_group(&self) -> usize {
        self.out_channels / self.groups
    }
}

impl Default for PtxConv1dConfig {
    fn default() -> Self {
        Self::new("ptx_conv1d_f32", 1, 1, 3)
    }
}

// ---------------------------------------------------------------------------
// Output length calculation
// ---------------------------------------------------------------------------

/// Compute the output length for a 1D convolution.
///
/// `L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
///
/// Returns `None` if the parameters would produce a non-positive output length.
#[must_use]
pub fn conv1d_output_length(
    length_in: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    let effective_k = dilation
        .checked_mul(kernel_size.checked_sub(1)?)?
        .checked_add(1)?;
    let padded = length_in.checked_add(2 * padding)?;
    if padded < effective_k {
        return None;
    }
    let numerator = padded - effective_k;
    Some(numerator / stride + 1)
}

// ---------------------------------------------------------------------------
// PTX generation — public API
// ---------------------------------------------------------------------------

/// Emit a complete PTX module for f32 1D convolution (NCL layout).
///
/// Generates raw PTX assembly implementing:
///   `out[n, oc, pos] = sum_{ic, k} input[n, g*C_g + ic, pos*stride - padding + k*dilation]
///                      * weight[oc, ic, k] + bias[oc]`
///
/// where `g = oc / (C_out / groups)` and `C_g = C_in / groups`.
///
/// # Parameters (kernel arguments)
///
/// - `input`: `[N, C_in, L_in]` f32 tensor pointer
/// - `weight`: `[C_out, C_in/groups, K]` f32 tensor pointer
/// - `bias`: `[C_out]` f32 pointer (ignored if `use_bias` is false)
/// - `output`: `[N, C_out, L_out]` f32 tensor pointer
/// - `N, L_in, L_out`: dimension scalars (u32)
///
/// The channel counts, kernel_size, stride, padding, dilation, and groups
/// are baked into the PTX as compile-time constants for maximum performance.
///
/// # Returns
///
/// Complete PTX module string ready for `cuModuleLoadData` or `ptxas`.
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_conv1d::{emit_ptx_conv1d, PtxConv1dConfig};
/// let config = PtxConv1dConfig::new("conv1d_k3", 64, 128, 3)
///     .with_padding(1);
/// let ptx = emit_ptx_conv1d(&config).unwrap();
/// assert!(ptx.contains(".entry conv1d_k3"));
/// ```
pub fn emit_ptx_conv1d(config: &PtxConv1dConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let name = &config.kernel_name;
    let block_size = config.block_size;
    let in_channels = config.in_channels;
    let out_channels = config.out_channels;
    let kernel_size = config.kernel_size;
    let stride = config.stride;
    let padding = config.padding;
    let dilation = config.dilation;
    let groups = config.groups;
    let ic_per_group = config.in_channels_per_group();

    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(8192);

    // -- Module header --
    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// Conv1d f32 (NCL): in_ch={in_channels}, out_ch={out_channels}, \
         kernel={kernel_size}, stride={stride}, pad={padding}, \
         dilation={dilation}, groups={groups}, block={block_size}\n\n"
    ));

    // -- Kernel entry point --
    // Parameters:
    //   input (ptr), weight (ptr), bias (ptr), output (ptr),
    //   batch_size (u32), length_in (u32), length_out (u32)
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
         \x20   .param .u32 param_length_in,\n\
         \x20   .param .u32 param_length_out\n\
         )\n",
    );

    ptx.push_str(&format!(".reqntid {block_size}\n{{\n"));

    // -- Register declarations --
    ptx.push_str(
        "\x20   // Register declarations\n\
         \x20   .reg .u32  %r<32>;\n\
         \x20   .reg .f32  %f<8>;\n\
         \x20   .reg .u64  %rd<12>;\n\
         \x20   .reg .pred %p<6>;\n\n",
    );

    // -- Load parameters --
    ptx.push_str(
        "\x20   // Load kernel parameters\n\
         \x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_weight];\n",
    );
    if config.use_bias {
        ptx.push_str("\x20   ld.param.u64  %rd2, [param_bias];\n");
    }
    ptx.push_str(
        "\x20   ld.param.u64  %rd3, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_batch_size];\n\
         \x20   ld.param.u32  %r1,  [param_length_in];\n\
         \x20   ld.param.u32  %r2,  [param_length_out];\n\n",
    );

    // -- Compute global thread index and stride --
    // global_idx = blockIdx.x * blockDim.x + threadIdx.x
    // grid_stride = gridDim.x * blockDim.x
    ptx.push_str(
        "\x20   // Compute global index and grid stride\n\
         \x20   mov.u32       %r3, %tid.x;           // threadIdx.x\n\
         \x20   mov.u32       %r4, %ctaid.x;         // blockIdx.x\n\
         \x20   mov.u32       %r5, %ntid.x;          // blockDim.x\n\
         \x20   mad.lo.u32    %r6, %r4, %r5, %r3;    // global_idx\n\
         \x20   mov.u32       %r7, %nctaid.x;        // gridDim.x\n\
         \x20   mul.lo.u32    %r8, %r7, %r5;         // grid_stride\n\n",
    );

    // -- Compute total output elements: batch_size * out_channels * length_out --
    ptx.push_str(&format!(
        "\x20   // Total output elements = batch * out_channels * length_out\n\
         \x20   mul.lo.u32    %r9, %r0, {out_channels};  // batch * out_channels\n\
         \x20   mul.lo.u32    %r9, %r9, %r2;             // * length_out\n\n"
    ));

    // -- Grid-stride loop --
    ptx.push_str(
        "CONV1D_LOOP:\n\
         \x20   // Bounds check: global_idx < total_elements\n\
         \x20   setp.ge.u32   %p0, %r6, %r9;\n\
         \x20   @%p0 bra      CONV1D_EXIT;\n\n",
    );

    // -- Decompose global_idx into (batch_idx, oc, out_pos) --
    // global_idx = batch_idx * (out_channels * length_out) + oc * length_out + out_pos
    // out_pos = global_idx % length_out
    // temp = global_idx / length_out
    // oc = temp % out_channels
    // batch_idx = temp / out_channels
    ptx.push_str(&format!(
        "\x20   // Decompose global_idx -> (batch_idx, oc, out_pos)\n\
         \x20   div.u32       %r10, %r6, %r2;        // temp = global_idx / length_out\n\
         \x20   rem.u32       %r11, %r6, %r2;        // out_pos = global_idx % length_out\n\
         \x20   rem.u32       %r12, %r10, {out_channels}; // oc = temp % out_channels\n\
         \x20   div.u32       %r13, %r10, {out_channels}; // batch_idx = temp / out_channels\n\n"
    ));

    // -- Compute group index and in-group channel offset --
    // group_idx = oc / out_channels_per_group
    // ic_start = group_idx * in_channels_per_group
    let oc_per_group = config.out_channels_per_group();
    ptx.push_str(&format!(
        "\x20   // Group computation\n\
         \x20   div.u32       %r14, %r12, {oc_per_group}; // group_idx = oc / oc_per_group\n\
         \x20   mul.lo.u32    %r15, %r14, {ic_per_group}; // ic_start = group_idx * ic_per_group\n\n"
    ));

    // -- Initialize accumulator --
    ptx.push_str(&format!(
        "\x20   // Initialize accumulator\n\
         \x20   mov.f32       %f0, {zero};            // acc = 0.0\n\n"
    ));

    // -- Compute base input position --
    // in_pos_base = out_pos * stride - padding
    // We store this as signed via u32 (negative values wrap, checked via bounds)
    ptx.push_str(&format!(
        "\x20   // Base input position: out_pos * stride - padding\n\
         \x20   mul.lo.u32    %r16, %r11, {stride};   // out_pos * stride\n\
         \x20   sub.u32       %r16, %r16, {padding};   // - padding (may underflow for padding > 0)\n\n"
    ));

    // -- Loop over input channels within group --
    ptx.push_str(&format!(
        "\x20   // Loop over input channels in group: ic = 0 .. ic_per_group\n\
         \x20   mov.u32       %r17, 0;                // ic = 0\n\
         IC_LOOP:\n\
         \x20   setp.ge.u32   %p1, %r17, {ic_per_group};\n\
         \x20   @%p1 bra      IC_DONE;\n\n"
    ));

    // -- Loop over kernel positions --
    ptx.push_str(&format!(
        "\x20   // Loop over kernel positions: k = 0 .. kernel_size\n\
         \x20   mov.u32       %r18, 0;                // k = 0\n\
         K_LOOP:\n\
         \x20   setp.ge.u32   %p2, %r18, {kernel_size};\n\
         \x20   @%p2 bra      K_DONE;\n\n"
    ));

    // -- Compute input position: in_pos = in_pos_base + k * dilation --
    ptx.push_str(&format!(
        "\x20   // in_pos = in_pos_base + k * dilation\n\
         \x20   mul.lo.u32    %r19, %r18, {dilation};  // k * dilation\n\
         \x20   add.u32       %r19, %r16, %r19;        // in_pos = base + k * dilation\n\n"
    ));

    // -- Bounds check: 0 <= in_pos < length_in --
    // Since we use unsigned arithmetic, negative in_pos wraps to large positive,
    // so a single `in_pos < length_in` check suffices.
    ptx.push_str(
        "\x20   // Bounds check: in_pos < length_in (unsigned handles negative wrap)\n\
         \x20   setp.ge.u32   %p3, %r19, %r1;        // in_pos >= length_in?\n\
         \x20   @%p3 bra      K_NEXT;                 // skip if out of bounds\n\n",
    );

    // -- Load input[batch_idx, ic_start + ic, in_pos] --
    // offset = ((batch_idx * in_channels + ic_start + ic) * length_in + in_pos) * 4
    ptx.push_str(&format!(
        "\x20   // Load input[batch_idx, ic_start + ic, in_pos]\n\
         \x20   mad.lo.u32    %r20, %r13, {in_channels}, %r15; // batch_idx * C_in + ic_start\n\
         \x20   add.u32       %r20, %r20, %r17;       // + ic\n\
         \x20   mad.lo.u32    %r20, %r20, %r1, %r19;  // * L_in + in_pos\n\
         \x20   mul.wide.u32  %rd4, %r20, 4;          // byte offset\n\
         \x20   add.u64       %rd5, %rd0, %rd4;       // &input[...]\n\
         \x20   ld.global.f32 %f1, [%rd5];            // val = input[...]\n\n"
    ));

    // -- Load weight[oc, ic, k] --
    // Weight layout: [out_channels, ic_per_group, kernel_size]
    // oc_in_group = oc % oc_per_group (but oc already gives global index)
    // offset = (oc * ic_per_group + ic) * kernel_size + k
    ptx.push_str(&format!(
        "\x20   // Load weight[oc, ic, k]\n\
         \x20   mad.lo.u32    %r21, %r12, {ic_per_group}, %r17; // oc * ic_per_group + ic\n\
         \x20   mad.lo.u32    %r21, %r21, {kernel_size}, %r18;  // * kernel_size + k\n\
         \x20   mul.wide.u32  %rd6, %r21, 4;          // byte offset\n\
         \x20   add.u64       %rd7, %rd1, %rd6;       // &weight[...]\n\
         \x20   ld.global.f32 %f2, [%rd7];            // w = weight[...]\n\n"
    ));

    // -- FMA: acc += input_val * weight_val --
    ptx.push_str(
        "\x20   // acc += input * weight\n\
         \x20   fma.rn.f32    %f0, %f1, %f2, %f0;\n\n",
    );

    // -- K loop end --
    ptx.push_str(
        "K_NEXT:\n\
         \x20   add.u32       %r18, %r18, 1;          // k++\n\
         \x20   bra           K_LOOP;\n\
         K_DONE:\n\n",
    );

    // -- IC loop end --
    ptx.push_str(
        "\x20   add.u32       %r17, %r17, 1;          // ic++\n\
         \x20   bra           IC_LOOP;\n\
         IC_DONE:\n\n",
    );

    // -- Add bias if enabled --
    if config.use_bias {
        ptx.push_str(
            "\x20   // Add bias[oc]\n\
             \x20   mul.wide.u32  %rd8, %r12, 4;          // oc * sizeof(f32)\n\
             \x20   add.u64       %rd9, %rd2, %rd8;       // &bias[oc]\n\
             \x20   ld.global.f32 %f3, [%rd9];            // bias_val\n\
             \x20   add.f32       %f0, %f0, %f3;          // acc += bias_val\n\n",
        );
    }

    // -- Store output[batch_idx, oc, out_pos] --
    // offset = ((batch_idx * out_channels + oc) * length_out + out_pos) * 4
    ptx.push_str(&format!(
        "\x20   // Store output[batch_idx, oc, out_pos]\n\
         \x20   mad.lo.u32    %r22, %r13, {out_channels}, %r12; // batch * C_out + oc\n\
         \x20   mad.lo.u32    %r22, %r22, %r2, %r11;  // * L_out + out_pos\n\
         \x20   mul.wide.u32  %rd10, %r22, 4;         // byte offset\n\
         \x20   add.u64       %rd11, %rd3, %rd10;     // &output[...]\n\
         \x20   st.global.f32 [%rd11], %f0;           // output[...] = acc\n\n"
    ));

    // -- Grid-stride advance --
    ptx.push_str(
        "\x20   // Grid-stride advance\n\
         \x20   add.u32       %r6, %r6, %r8;          // global_idx += grid_stride\n\
         \x20   bra           CONV1D_LOOP;\n\n",
    );

    // -- Kernel exit --
    ptx.push_str(
        "CONV1D_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Convenience API
// ---------------------------------------------------------------------------

/// Convenience: emit PTX conv1d with specified channel/kernel params and
/// default stride=1, pad=0, dilation=1, groups=1, sm_80.
pub fn emit_ptx_conv1d_default(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<String, PtxCodegenError> {
    let config = PtxConv1dConfig::new("ptx_conv1d_f32", in_channels, out_channels, kernel_size)
        .with_stride(stride)
        .with_padding(padding);
    emit_ptx_conv1d(&config)
}

/// Compute the CUDA launch configuration for a conv1d kernel.
///
/// Total output elements = `batch * out_channels * output_length`.
/// Grid: `(ceil(total / block_size), 1, 1)`.
/// Block: `(block_size, 1, 1)`.
#[must_use]
pub fn ptx_conv1d_launch_config(
    batch: usize,
    out_channels: usize,
    output_length: usize,
) -> CudaLaunchConfig {
    let total = batch * out_channels * output_length;
    let block_size = PTX_CONV1D_BLOCK_SIZE as u32;
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
#[path = "ptx_conv1d_tests.rs"]
mod ptx_conv1d_tests;
