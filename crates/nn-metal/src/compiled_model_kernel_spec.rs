// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Hardware-agnostic kernel parameter descriptor (`KernelSpec`).
//!
//! `KernelSpec` captures everything needed to encode a Metal compute dispatch
//! — kernel identity (name + MSL source), grid/threadgroup dimensions, buffer
//! binding layout, and constants — without depending on Metal-specific types
//! (no `ComputePipeline`, no `MetalBuffer`).
//!
//! Both the NativeEncoding path (direct dispatch) and the ICB path (indirect
//! command buffer) are produced from a single `KernelSpec`, eliminating the
//! parallel `plan_*_encoding` / `pre_compile_*` implementations per NativeOp.
//!
//! ## Conversion
//!
//! - [`KernelSpec::into_encoding`] compiles MSL → pipeline → `NativeEncoding`
//! - [`MultiKernelSpec::into_encoding_sequence`] → `Vec<NativeEncoding>`
//!
//! Part of #3503 (KernelSpec unification).

use crate::cache::PipelineCache;
use crate::kernel_dispatch::KernelPipeline;

use super::encoding::{
    AuxiliaryAlloc, NativeBindingSource, NativeDispatchMode, NativeEncoding,
};

// ---------------------------------------------------------------------------
// KernelSpec: pre-compilation kernel descriptor
// ---------------------------------------------------------------------------

/// Hardware-agnostic kernel parameter descriptor.
///
/// Captures everything needed to encode a Metal compute dispatch:
/// kernel identity (name + MSL), grid/threadgroup dimensions, buffer
/// binding layout, and constants. Both [`NativeEncoding`] and ICB codegen
/// are produced from this single description.
///
/// Constructed by per-NativeOp `spec_*()` functions (in
/// `compiled_model_kernel_spec_norm.rs` and
/// `compiled_model_kernel_spec_fused.rs`). Consumed by
/// [`into_encoding`](Self::into_encoding) at model build time.
#[derive(Debug)]
pub(crate) struct KernelSpec {
    /// Metal kernel function name (e.g., `"fused_instance_norm_float"`).
    pub kernel_name: String,
    /// MSL source code for the kernel.
    pub msl_source: String,
    /// Grid size: thread count (for `Threads` mode) or threadgroup count
    /// (for `Threadgroups` mode).
    pub grid: [u32; 3],
    /// Threadgroup size (threads per threadgroup per dimension).
    pub threadgroup: [u32; 3],
    /// Dispatch mode: threads vs threadgroups.
    pub dispatch_mode: SpecDispatchMode,
    /// Threadgroup memory in bytes (0 = none).
    pub threadgroup_memory_bytes: u64,
    /// Total output buffer size in bytes.
    pub output_bytes: usize,
    /// Buffer bindings in dispatch order: `(buffer_index, binding_source)`.
    pub bindings: Vec<(usize, KernelBinding)>,
    /// Number of kernel parameters (inputs + output + constants).
    /// Used by `KernelPipeline::from_msl`.
    pub param_count: usize,
    /// Whether to enable Metal fast-math for this kernel.
    pub fast_math: bool,
}

/// Dispatch mode for a [`KernelSpec`].
///
/// Maps 1:1 to [`NativeDispatchMode`] but lives in the pre-compilation layer
/// to avoid coupling `spec_*()` builders to the encoding types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpecDispatchMode {
    /// `dispatch_threads(grid, threadgroup)` — grid is total thread count.
    Threads,
    /// `dispatch_thread_groups(threadgroups, threads_per_group)` — grid is
    /// threadgroup count.
    Threadgroups,
}

/// Abstract binding descriptor for a [`KernelSpec`].
///
/// Maps to [`NativeBindingSource`] without carrying Metal-specific data.
/// Each variant matches the corresponding `NativeBindingSource` variant.
#[derive(Debug)]
pub(crate) enum KernelBinding {
    /// Input from the step's edge map (edge index).
    Edge(usize),
    /// Pre-uploaded weight buffer by name.
    Weight(String),
    /// Output buffer (allocated by the dispatch function).
    Output,
    /// Inline constant bytes (encoded via Metal `setBytes`).
    Constant(Vec<u8>),
    /// Output buffer from encoding[i] in a multi-dispatch sequence.
    Intermediate(usize),
    /// Auxiliary buffer from a prior encoding in a multi-dispatch sequence.
    IntermediateAuxiliary {
        encoding_idx: usize,
        auxiliary_idx: usize,
    },
    /// Pre-resolved `GpuSlice` passed by the caller.
    PreResolved(usize),
}

impl KernelBinding {
    /// Create a constant binding from a `u32` value.
    pub(crate) fn constant_u32(val: u32) -> Self {
        Self::Constant(bytemuck::bytes_of(&val).to_vec())
    }

    /// Create a constant binding from an `f32` value.
    pub(crate) fn constant_f32(val: f32) -> Self {
        Self::Constant(bytemuck::bytes_of(&val).to_vec())
    }
}

// ---------------------------------------------------------------------------
// MultiKernelSpec: multi-dispatch sequences (e.g., FusedResBlock)
// ---------------------------------------------------------------------------

/// Multi-dispatch kernel spec for sequences that require multiple Metal
/// dispatches with intermediate buffers (e.g., FusedResBlock 3-dispatch).
///
/// Each command is a [`KernelSpec`]; auxiliary allocs describe per-command
/// extra buffers (counters, partial sums, etc.).
#[derive(Debug)]
pub(crate) struct MultiKernelSpec {
    /// Individual dispatch commands in execution order.
    pub commands: Vec<KernelSpec>,
    /// Per-command auxiliary buffer allocations. Outer index = command index.
    pub auxiliary_allocs: Vec<Vec<AuxiliaryAllocSpec>>,
}

/// Auxiliary buffer allocation descriptor for a multi-dispatch command.
///
/// Pre-compilation equivalent of [`AuxiliaryAlloc`].
#[derive(Debug)]
pub(crate) struct AuxiliaryAllocSpec {
    /// Buffer size in bytes.
    pub bytes: usize,
    /// Metal buffer binding index for this allocation.
    pub binding_index: usize,
    /// If true, blit-fill with zeros before the compute dispatch.
    pub zero_fill: bool,
    /// If true, this buffer is exposed for subsequent encodings via
    /// `IntermediateAuxiliary` binding source.
    pub expose_as_intermediate: bool,
}

// ---------------------------------------------------------------------------
// Conversions: KernelSpec → NativeEncoding
// ---------------------------------------------------------------------------

impl KernelSpec {
    /// Compile MSL source into a Metal pipeline and build a [`NativeEncoding`].
    ///
    /// This is the primary conversion path: each `spec_*()` builder produces
    /// a `KernelSpec`, and the caller converts it to a `NativeEncoding` for
    /// runtime dispatch.
    ///
    /// # Errors
    ///
    /// Returns an error if MSL compilation fails or pipeline creation fails.
    pub(crate) fn into_encoding(
        self,
        cache: &PipelineCache,
    ) -> Result<NativeEncoding, String> {
        let pipeline = KernelPipeline::from_msl(
            cache,
            &self.msl_source,
            &self.kernel_name,
            self.param_count,
            self.fast_math,
        )
        .map_err(|e| format!("KernelSpec pipeline '{}': {e}", self.kernel_name))?;

        let bindings = self
            .bindings
            .into_iter()
            .map(|(idx, binding)| (idx, binding.into_native_binding()))
            .collect();

        Ok(NativeEncoding {
            pipeline,
            grid: self.grid,
            threadgroup: self.threadgroup,
            dispatch_mode: self.dispatch_mode.into_native(),
            threadgroup_memory_bytes: self.threadgroup_memory_bytes,
            output_bytes: self.output_bytes,
            bindings,
            auxiliary_allocs: Vec::new(),
        })
    }
}

impl MultiKernelSpec {
    /// Compile all commands into a `Vec<NativeEncoding>` sequence.
    ///
    /// Each command's MSL is compiled independently. Auxiliary allocations
    /// are attached to the corresponding encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if any command's MSL compilation fails.
    pub(crate) fn into_encoding_sequence(
        self,
        cache: &PipelineCache,
    ) -> Result<Vec<NativeEncoding>, String> {
        let mut encodings = Vec::with_capacity(self.commands.len());

        for (cmd_idx, (spec, aux_specs)) in self
            .commands
            .into_iter()
            .zip(self.auxiliary_allocs)
            .enumerate()
        {
            let mut encoding = spec.into_encoding(cache).map_err(|e| {
                format!("MultiKernelSpec command[{cmd_idx}]: {e}")
            })?;

            encoding.auxiliary_allocs = aux_specs
                .into_iter()
                .map(|a| AuxiliaryAlloc {
                    bytes: a.bytes,
                    binding_index: a.binding_index,
                    zero_fill: a.zero_fill,
                    expose_as_intermediate: a.expose_as_intermediate,
                })
                .collect();

            encodings.push(encoding);
        }

        Ok(encodings)
    }
}

// ---------------------------------------------------------------------------
// Internal conversions
// ---------------------------------------------------------------------------

impl SpecDispatchMode {
    /// Convert to the runtime [`NativeDispatchMode`].
    fn into_native(self) -> NativeDispatchMode {
        match self {
            Self::Threads => NativeDispatchMode::Threads,
            Self::Threadgroups => NativeDispatchMode::Threadgroups,
        }
    }
}

impl KernelBinding {
    /// Convert to the runtime [`NativeBindingSource`].
    fn into_native_binding(self) -> NativeBindingSource {
        match self {
            Self::Edge(idx) => NativeBindingSource::Edge(idx),
            Self::Weight(name) => NativeBindingSource::Weight(name),
            Self::Output => NativeBindingSource::Output,
            Self::Constant(bytes) => NativeBindingSource::Constant(bytes),
            Self::Intermediate(idx) => NativeBindingSource::Intermediate(idx),
            Self::IntermediateAuxiliary {
                encoding_idx,
                auxiliary_idx,
            } => NativeBindingSource::IntermediateAuxiliary {
                encoding_idx,
                auxiliary_idx,
            },
            Self::PreResolved(idx) => NativeBindingSource::PreResolved(idx),
        }
    }
}

// ---------------------------------------------------------------------------
// Spec builders: one function per NativeOp kind
// ---------------------------------------------------------------------------

/// Norm-family spec builders (InstanceNorm, LayerNorm, AddLayerNorm,
/// ChannelsFirstLayerNorm, AdaLayerNorm).
#[path = "compiled_model_kernel_spec_norm.rs"]
pub(crate) mod norm;

/// Fused-op spec builders (AdainSnake, AdainLeakyRelu).
#[path = "compiled_model_kernel_spec_fused.rs"]
mod fused;

/// Single-dispatch spec builders (GroupNorm, RmsNorm, Snake, FlashAttention).
#[path = "compiled_model_kernel_spec_ops.rs"]
mod ops;

/// GEMM-based spec builders (LinearActivation, NormLinear, Int8Gemm).
#[path = "compiled_model_kernel_spec_gemm.rs"]
mod gemm;

// Re-export spec builders so tests can use `super::*`.
#[cfg(test)]
pub(crate) use norm::{
    spec_instance_norm, spec_layer_norm, spec_add_layer_norm,
    spec_channels_first_layer_norm, spec_ada_layer_norm, NORM_TG_SIZE,
};
#[cfg(test)]
pub(crate) use fused::{spec_adain_snake, spec_adain_leaky_relu};
#[cfg(test)]
pub(crate) use ops::{spec_group_norm, spec_rms_norm, spec_snake, spec_flash_attention};
#[cfg(test)]
pub(crate) use gemm::{spec_linear_activation, spec_norm_linear};

#[cfg(test)]
#[path = "compiled_model_kernel_spec_tests.rs"]
mod tests;
