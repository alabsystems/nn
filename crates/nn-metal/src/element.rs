// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! [`MetalElement`] trait for type-generic Metal compute dispatch.
//!
//! Abstracts the buffer creation and readback patterns that differ between
//! element types (f32 vs [`half::f16`]). This allows a single generic dispatch
//! method to handle both types without code duplication.
//!
//! Uses fully-qualified [`half::f16`] paths (not the bare name `f16`) so
//! consumers compile on both stable and nightly toolchains where the compiler
//! may resolve bare `f16` to the unstable primitive `std::f16`.

use nn_dsl::ir::ScalarType;

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::gpu_scope;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
    impl Sealed for half::bf16 {}
}

/// Types that can be dispatched through Metal compute kernels.
///
/// Sealed: only `f32`, [`half::f16`], and [`half::bf16`] implement this trait.
/// Adding new element types requires adding a `Sealed` impl in this module.
///
/// [`half::bf16`] is converted to/from [`half::f16`] at the Metal boundary
/// because Apple GPUs do not have native bf16 compute — they operate on
/// `float` and `half` only.
pub trait MetalElement: sealed::Sealed + Copy + 'static {
    /// Create a Metal buffer from a slice of elements.
    fn create_buffer(ctx: &MetalContext, data: &[Self]) -> Result<MetalBuffer, MetalError>;

    /// Read elements back from a Metal buffer.
    fn read_buffer(buffer: &MetalBuffer) -> Result<Vec<Self>, MetalError>;

    /// Read `count` elements from a Metal buffer starting at `byte_offset`.
    ///
    /// Used by [`execute_tensor_dispatch`] to read the correct output region
    /// when arena allocation places the output at a non-zero offset within
    /// the arena buffer. Without this, readback would start from byte 0 of
    /// the full arena, returning stale intermediate data.
    fn read_buffer_at_offset(
        buffer: &MetalBuffer,
        byte_offset: usize,
        count: usize,
    ) -> Result<Vec<Self>, MetalError>;

    /// Size of one element in bytes (for output buffer allocation).
    fn element_size() -> usize;

    /// The MSL scalar type that corresponds to this element type.
    ///
    /// Used to verify that the `dtype` parameter passed to dispatch matches
    /// the runtime element type. bf16 maps to `F16` because Metal stores
    /// bf16 as f16.
    fn scalar_type() -> ScalarType;
}

impl MetalElement for f32 {
    fn create_buffer(ctx: &MetalContext, data: &[Self]) -> Result<MetalBuffer, MetalError> {
        ctx.create_buffer(data)
    }

    fn read_buffer(buffer: &MetalBuffer) -> Result<Vec<Self>, MetalError> {
        gpu_scope::flush()
            .map_err(|e| MetalError::DispatchFailed(format!("flush before readback: {e}")))?;
        Ok(buffer.contents::<Self>()?.to_vec())
    }

    fn read_buffer_at_offset(
        buffer: &MetalBuffer,
        byte_offset: usize,
        count: usize,
    ) -> Result<Vec<Self>, MetalError> {
        gpu_scope::flush()
            .map_err(|e| MetalError::DispatchFailed(format!("flush before readback: {e}")))?;
        Ok(buffer
            .contents_at_offset::<Self>(byte_offset, count)?
            .to_vec())
    }

    fn element_size() -> usize {
        size_of::<Self>()
    }

    fn scalar_type() -> ScalarType {
        ScalarType::F32
    }
}

impl MetalElement for half::f16 {
    fn create_buffer(ctx: &MetalContext, data: &[Self]) -> Result<MetalBuffer, MetalError> {
        let encoded: Vec<u16> = data.iter().map(|v| v.to_bits()).collect();
        ctx.create_buffer(&encoded)
    }

    fn read_buffer(buffer: &MetalBuffer) -> Result<Vec<Self>, MetalError> {
        gpu_scope::flush()
            .map_err(|e| MetalError::DispatchFailed(format!("flush before readback: {e}")))?;
        Ok(buffer
            .contents::<u16>()?
            .iter()
            .map(|v| Self::from_bits(*v))
            .collect())
    }

    fn read_buffer_at_offset(
        buffer: &MetalBuffer,
        byte_offset: usize,
        count: usize,
    ) -> Result<Vec<Self>, MetalError> {
        gpu_scope::flush()
            .map_err(|e| MetalError::DispatchFailed(format!("flush before readback: {e}")))?;
        Ok(buffer
            .contents_at_offset::<u16>(byte_offset, count)?
            .iter()
            .map(|v| Self::from_bits(*v))
            .collect())
    }

    fn element_size() -> usize {
        size_of::<u16>()
    }

    fn scalar_type() -> ScalarType {
        ScalarType::F16
    }
}

/// bf16 → f16 conversion at the Metal boundary. Metal GPUs have no native
/// bf16 compute, so bf16 data is converted to f16 for GPU dispatch and
/// converted back on readback. This loses the extra exponent range of bf16
/// but preserves the workflow for models stored as bf16 (common in LLMs).
impl MetalElement for half::bf16 {
    fn create_buffer(ctx: &MetalContext, data: &[Self]) -> Result<MetalBuffer, MetalError> {
        // Convert bf16 → f16 for Metal (bf16 has wider exponent range but
        // fewer mantissa bits; f16 has narrower range but same mantissa
        // precision as bf16 — values outside f16 range clamp to ±inf).
        let encoded: Vec<u16> = data
            .iter()
            .map(|v| half::f16::from_f32(v.to_f32()).to_bits())
            .collect();
        ctx.create_buffer(&encoded)
    }

    fn read_buffer(buffer: &MetalBuffer) -> Result<Vec<Self>, MetalError> {
        gpu_scope::flush()
            .map_err(|e| MetalError::DispatchFailed(format!("flush before readback: {e}")))?;
        // Convert f16 → bf16 on readback.
        Ok(buffer
            .contents::<u16>()?
            .iter()
            .map(|v| Self::from_f32(half::f16::from_bits(*v).to_f32()))
            .collect())
    }

    fn read_buffer_at_offset(
        buffer: &MetalBuffer,
        byte_offset: usize,
        count: usize,
    ) -> Result<Vec<Self>, MetalError> {
        gpu_scope::flush()
            .map_err(|e| MetalError::DispatchFailed(format!("flush before readback: {e}")))?;
        // Convert f16 → bf16 on readback.
        Ok(buffer
            .contents_at_offset::<u16>(byte_offset, count)?
            .iter()
            .map(|v| Self::from_f32(half::f16::from_bits(*v).to_f32()))
            .collect())
    }

    fn element_size() -> usize {
        // bf16 is stored as f16 on Metal, so element size matches f16.
        size_of::<u16>()
    }

    fn scalar_type() -> ScalarType {
        // bf16 is stored as f16 on Metal — MSL codegen uses "half".
        ScalarType::F16
    }
}
