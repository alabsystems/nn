// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bridge between nn-dsl `KernelDef` IR and nn-metal GPU dispatch.
//!
//! [`KernelPipeline`] compiles a `KernelDef` to MSL, creates a Metal pipeline,
//! and dispatches compute work with proper buffer bindings.

use nn_dsl::emit_msl_with_contract;
use nn_dsl::ir::KernelDef;
use nn_dsl::KernelDescriptor;
use nn_dsl::{PrecisionContract, PrecisionTier};

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::error::MetalError;
use crate::kernel_source::KernelSource;
use crate::pipeline::ComputePipeline;

mod elementwise;
mod nd;

/// A compiled kernel ready for GPU dispatch.
///
/// Constructed from a `KernelDef` (nn-dsl IR). Holds the compiled Metal
/// pipeline and kernel metadata needed to bind buffers and dispatch.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct KernelPipeline {
    pipeline: ComputePipeline,
    param_count: usize,
    name: String,
    fast_math: bool,
}

/// Access mode for a bound kernel parameter buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferAccess {
    ReadOnly,
    ReadWrite,
    WriteOnly,
}

/// Buffer plus declared access role for low-level dispatch contracts.
#[derive(Clone, Copy, Debug)]
pub struct BufferBinding<'a> {
    buffer: &'a MetalBuffer,
    access: BufferAccess,
}

impl<'a> BufferBinding<'a> {
    #[must_use]
    pub fn read_only(buffer: &'a MetalBuffer) -> Self {
        Self {
            buffer,
            access: BufferAccess::ReadOnly,
        }
    }

    #[must_use]
    pub fn read_write(buffer: &'a MetalBuffer) -> Self {
        Self {
            buffer,
            access: BufferAccess::ReadWrite,
        }
    }

    #[must_use]
    pub fn write_only(buffer: &'a MetalBuffer) -> Self {
        Self {
            buffer,
            access: BufferAccess::WriteOnly,
        }
    }

    #[must_use]
    pub fn buffer(&self) -> &'a MetalBuffer {
        self.buffer
    }

    #[must_use]
    pub fn access(&self) -> BufferAccess {
        self.access
    }
}

impl KernelPipeline {
    /// Compile a `KernelDef` into a Metal compute pipeline using default
    /// precision (Normal tier, no fast-math).
    #[must_use = "returns a Result that may contain an error"]
    pub fn compile(cache: &PipelineCache, kernel: &KernelDef) -> Result<Self, MetalError> {
        let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, kernel.return_type);
        Self::compile_with_contract(cache, kernel, contract)
    }

    /// Compile a `KernelDef` with an explicit precision contract.
    #[must_use = "returns a Result that may contain an error"]
    pub fn compile_with_contract(
        cache: &PipelineCache,
        kernel: &KernelDef,
        contract: PrecisionContract,
    ) -> Result<Self, MetalError> {
        let msl = emit_msl_with_contract(kernel, contract)?;
        let entry = format!("{}_kernel", kernel.name);
        let source = KernelSource::new(&msl, &entry).with_fast_math(contract.fast_math);
        let pipeline = cache.get_or_compile(&source)?;
        Ok(Self {
            pipeline,
            param_count: kernel.params.len(),
            name: kernel.name.clone(),
            fast_math: contract.fast_math,
        })
    }

    /// Compile from a proc-macro-generated [`KernelDescriptor`].
    ///
    /// This is the preferred constructor for kernels defined with `#[kernel]`.
    /// The descriptor bundles MSL source, entry point, param count, and
    /// fast-math flag atomically, preventing mismatched metadata.
    ///
    /// ```text
    /// let pipeline = KernelPipeline::from_descriptor(&cache, &SNAKE_DESCRIPTOR)?;
    /// ```
    #[must_use = "returns a Result that may contain an error"]
    pub fn from_descriptor(
        cache: &PipelineCache,
        descriptor: &KernelDescriptor,
    ) -> Result<Self, MetalError> {
        Self::from_msl(
            cache,
            descriptor.msl_source,
            descriptor.entry_point,
            descriptor.param_count,
            descriptor.fast_math,
        )
    }

    /// Compile from pre-generated MSL source text.
    ///
    /// Use this for hand-written or externally generated MSL where a
    /// [`KernelDescriptor`] is not available. The caller is responsible for
    /// ensuring `param_count` matches the actual MSL kernel signature.
    ///
    /// For proc-macro-generated kernels, prefer [`from_descriptor`](Self::from_descriptor)
    /// which bundles all metadata atomically.
    #[must_use = "returns a Result that may contain an error"]
    pub fn from_msl(
        cache: &PipelineCache,
        msl_source: &str,
        entry_point: &str,
        param_count: usize,
        fast_math: bool,
    ) -> Result<Self, MetalError> {
        let source = KernelSource::new(msl_source, entry_point).with_fast_math(fast_math);
        let pipeline = cache.get_or_compile(&source)?;
        let name = entry_point
            .strip_suffix("_kernel")
            .unwrap_or(entry_point)
            .to_string();
        Ok(Self {
            pipeline,
            param_count,
            name,
            fast_math,
        })
    }

    /// Compile from MSL source with Metal function constant specialization.
    ///
    /// Like [`from_msl`](Self::from_msl) but produces a pipeline specialized
    /// for the given function constant values. The Metal compiler uses these
    /// values to unroll loops and eliminate dead code (#3449).
    ///
    /// `function_constants` is a slice of `(index, uint32_value)` pairs
    /// mapping to `[[function_constant(index)]]` declarations in the MSL.
    #[must_use = "returns a Result that may contain an error"]
    pub fn from_msl_specialized(
        cache: &PipelineCache,
        msl_source: &str,
        entry_point: &str,
        param_count: usize,
        fast_math: bool,
        function_constants: &[(u32, u32)],
    ) -> Result<Self, MetalError> {
        let mut source = KernelSource::new(msl_source, entry_point).with_fast_math(fast_math);
        for &(idx, val) in function_constants {
            source = source.with_function_constant(idx, val);
        }
        let pipeline = cache.get_or_compile(&source)?;
        let name = entry_point
            .strip_suffix("_kernel")
            .unwrap_or(entry_point)
            .to_string();
        Ok(Self {
            pipeline,
            param_count,
            name,
            fast_math,
        })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn param_count(&self) -> usize {
        self.param_count
    }

    #[must_use]
    pub fn fast_math(&self) -> bool {
        self.fast_math
    }

    /// Access the underlying pipeline (for custom encoding).
    #[must_use]
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }

    fn validate_inputs<T>(&self, inputs: &[&[T]]) -> Result<usize, MetalError> {
        if inputs.len() != self.param_count {
            return Err(MetalError::ParamCountMismatch {
                expected: self.param_count,
                got: inputs.len(),
            });
        }

        let total = inputs.first().map_or(0, |slice| slice.len());
        for (index, input) in inputs.iter().enumerate() {
            if input.len() != total {
                return Err(MetalError::InputLenMismatch {
                    expected: total,
                    got: input.len(),
                    index,
                });
            }
        }

        Ok(total)
    }

    fn total_u32(&self, total: usize) -> Result<u32, MetalError> {
        u32::try_from(total).map_err(|_| MetalError::DispatchSizeOverflow(total))
    }
}

/// Compute output buffer byte count with overflow checking.
///
/// Returns `MetalError::BufferByteOverflow` if the product exceeds `usize::MAX`.
fn checked_output_bytes(output_elems: usize, elem_size: usize) -> Result<usize, MetalError> {
    output_elems
        .checked_mul(elem_size)
        .ok_or(MetalError::BufferByteOverflow {
            elems: output_elems,
            elem_size,
        })
}

/// Pure dispatch-size conversion, matching [`KernelPipeline::total_u32`].
///
/// Extracted for Kani verification — the Metal-dependent method delegates
/// to `u32::try_from` with the same semantics.
#[cfg_attr(not(kani), allow(dead_code))]
#[inline]
pub(crate) fn check_dispatch_size(total: usize) -> Option<u32> {
    u32::try_from(total).ok()
}

#[cfg(test)]
mod tests {
    use super::check_dispatch_size;
    use crate::dispatch_plan::threadgroup_width_1d;

    #[test]
    fn test_threadgroup_width_small_total() {
        assert_eq!(threadgroup_width_1d(1), 1);
        assert_eq!(threadgroup_width_1d(32), 32);
        assert_eq!(threadgroup_width_1d(63), 63);
    }

    #[test]
    fn test_threadgroup_width_at_boundary() {
        assert_eq!(threadgroup_width_1d(64), 64);
    }

    #[test]
    fn test_threadgroup_width_above_boundary() {
        assert_eq!(threadgroup_width_1d(65), 64);
        assert_eq!(threadgroup_width_1d(1000), 64);
        assert_eq!(threadgroup_width_1d(u32::MAX), 64);
    }

    #[test]
    fn test_dispatch_size_zero() {
        assert_eq!(check_dispatch_size(0), Some(0));
    }

    #[test]
    fn test_dispatch_size_fits_u32() {
        assert_eq!(check_dispatch_size(1), Some(1));
        assert_eq!(check_dispatch_size(u32::MAX as usize), Some(u32::MAX));
    }

    #[test]
    fn test_dispatch_size_overflow() {
        assert_eq!(check_dispatch_size(u32::MAX as usize + 1), None);
        assert_eq!(check_dispatch_size(usize::MAX), None);
    }
}

#[cfg(kani)]
mod proofs {
    use super::check_dispatch_size;
    use crate::dispatch_plan::threadgroup_width_1d;

    /// Proves dispatch size conversion succeeds iff total fits in u32,
    /// and the converted value equals the original.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn dispatch_size_fits_u32_roundtrip() {
        let total: usize = kani::any();
        kani::assume(total <= u32::MAX as usize);
        let converted = check_dispatch_size(total).expect("should fit in u32");
        assert_eq!(converted as usize, total);
    }

    /// Proves dispatch size conversion fails for values above u32::MAX.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn dispatch_size_rejects_overflow() {
        let total: usize = kani::any();
        kani::assume(total > u32::MAX as usize);
        assert!(check_dispatch_size(total).is_none());
    }

    /// Proves threadgroup width is always in [1, 64] for non-zero totals,
    /// and never exceeds the total element count.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn threadgroup_width_bounded_and_nonzero() {
        let total: u32 = kani::any();
        kani::assume(total > 0);
        let width = threadgroup_width_1d(total);
        assert!(width >= 1);
        assert!(width <= 64);
        assert!(width <= total);
    }
}
