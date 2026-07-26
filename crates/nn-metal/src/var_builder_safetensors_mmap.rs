// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Mmap convenience constructors and `MetalVarBuilderExt` trait for VarBuilder.
//!
//! Extracted from `var_builder_safetensors.rs` (#1572) to keep files under 500 lines.

use std::path::Path;
use std::sync::Arc;

use nn_core::{DType, Device, VarBuilder};

use crate::safetensors::{WeightError, WeightMap};

use super::{var_builder_from_weight_map, ShardedSafeTensorsBackend};

/// Load safetensors files and create a `VarBuilder` backed by mmap.
///
/// Matches candle's `VarBuilder::from_mmaped_safetensors(&[path], dtype, &device)`
/// signature. This is the primary weight-loading entry point for dvoice
/// production code (~25 call sites).
///
/// Supports both single-file and multi-file (sharded) safetensors. When
/// multiple paths are provided, each file gets its own mmap and Metal buffer;
/// tensor lookups search all shards transparently.
///
/// Uses the global Metal context ([`MetalBackend::init`] must be called first).
///
/// # Safety
///
/// The file(s) must not be modified or truncated while the returned
/// `VarBuilder` is alive. This is the standard mmap contract.
///
/// # Example
///
/// ```no_run
/// # use nn_core::{DType, Device};
/// # use nn_metal::from_mmaped_safetensors;
/// let vb = unsafe {
///     from_mmaped_safetensors(
///         &[std::path::Path::new("model.safetensors")],
///         DType::F32,
///         &Device::Cpu,
///     ).expect("load weights")
/// };
/// let weight = vb.pp("encoder").get(&[512, 256], "weight").expect("load weight");
/// ```
pub unsafe fn from_mmaped_safetensors(
    paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder, WeightError> {
    if paths.is_empty() {
        return Err(WeightError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "from_mmaped_safetensors: empty paths slice",
        )));
    }
    if paths.len() == 1 {
        // SAFETY: Caller of `from_mmaped_safetensors` guarantees files are not
        // modified while the returned VarBuilder is alive (mmap contract).
        let weight_map = unsafe { WeightMap::load_global(paths[0].as_ref())? };
        return Ok(var_builder_from_weight_map(weight_map, dtype, device));
    }
    // Multi-file (sharded) safetensors: load each shard into its own WeightMap.
    let mut shards = Vec::with_capacity(paths.len());
    for path in paths {
        // SAFETY: Same mmap contract as above — caller guarantees file stability.
        let wm = unsafe { WeightMap::load_global(path.as_ref())? };
        shards.push(wm);
    }
    Ok(VarBuilder::from_backend(
        Arc::new(ShardedSafeTensorsBackend { shards }),
        dtype,
        *device,
    ))
}

/// Load safetensors files with an explicit Metal context.
///
/// Same as [`from_mmaped_safetensors`] but takes an explicit [`MetalContext`]
/// instead of using the global context. Useful in tests and multi-context
/// scenarios.
///
/// # Safety
///
/// Same as [`from_mmaped_safetensors`]: files must not be modified while alive.
pub unsafe fn from_mmaped_safetensors_with_ctx(
    paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    ctx: &crate::context::MetalContext,
) -> Result<VarBuilder, WeightError> {
    if paths.is_empty() {
        return Err(WeightError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "from_mmaped_safetensors: empty paths slice",
        )));
    }
    if paths.len() == 1 {
        // SAFETY: Caller of `from_mmaped_safetensors_with_ctx` guarantees files
        // are not modified while the returned VarBuilder is alive (mmap contract).
        let weight_map = unsafe { WeightMap::load(paths[0].as_ref(), ctx)? };
        return Ok(var_builder_from_weight_map(weight_map, dtype, device));
    }
    let mut shards = Vec::with_capacity(paths.len());
    for path in paths {
        // SAFETY: Same mmap contract as above — caller guarantees file stability.
        let wm = unsafe { WeightMap::load(path.as_ref(), ctx)? };
        shards.push(wm);
    }
    Ok(VarBuilder::from_backend(
        Arc::new(ShardedSafeTensorsBackend { shards }),
        dtype,
        *device,
    ))
}

// -- Extension trait (candle API compat) --------------------------------------

/// Extension trait for `VarBuilder` that adds Metal-specific safetensors loading.
///
/// This trait exists because `VarBuilder` is defined in nn-core and Rust's
/// orphan rule prevents adding inherent methods from nn-metal. The pattern
/// matches [`MetalTensorExt`](crate::MetalTensorExt).
///
/// Consumers import the trait to use `VarBuilder::from_mmaped_safetensors(...)`:
///
/// ```no_run
/// # use nn_core::{DType, Device, VarBuilder};
/// # use nn_metal::MetalVarBuilderExt;
/// let vb = unsafe {
///     VarBuilder::from_mmaped_safetensors(
///         &[std::path::Path::new("model.safetensors")],
///         DType::F32,
///         &Device::Cpu,
///     ).expect("load weights")
/// };
/// ```
pub trait MetalVarBuilderExt {
    /// Load safetensors files and create a `VarBuilder` backed by mmap.
    ///
    /// Matches candle's `VarBuilder::from_mmaped_safetensors(&[path], dtype, &device)`
    /// associated function signature. Delegates to the free function
    /// [`from_mmaped_safetensors`].
    ///
    /// # Safety
    ///
    /// The file(s) must not be modified or truncated while the returned
    /// `VarBuilder` is alive. This is the standard mmap contract.
    unsafe fn from_mmaped_safetensors(
        paths: &[impl AsRef<Path>],
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder, WeightError>;

    /// Load safetensors files with an explicit Metal context.
    ///
    /// Same as [`MetalVarBuilderExt::from_mmaped_safetensors`] but takes an
    /// explicit [`MetalContext`](crate::context::MetalContext) instead of using
    /// the global context.
    ///
    /// # Safety
    ///
    /// Same as [`MetalVarBuilderExt::from_mmaped_safetensors`]: files must not
    /// be modified while alive.
    unsafe fn from_mmaped_safetensors_with_ctx(
        paths: &[impl AsRef<Path>],
        dtype: DType,
        device: &Device,
        ctx: &crate::context::MetalContext,
    ) -> Result<VarBuilder, WeightError>;
}

impl MetalVarBuilderExt for VarBuilder {
    unsafe fn from_mmaped_safetensors(
        paths: &[impl AsRef<Path>],
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder, WeightError> {
        // SAFETY: Delegates to `from_mmaped_safetensors` — caller upholds
        // the mmap contract (files not modified while VarBuilder is alive).
        unsafe { from_mmaped_safetensors(paths, dtype, device) }
    }

    unsafe fn from_mmaped_safetensors_with_ctx(
        paths: &[impl AsRef<Path>],
        dtype: DType,
        device: &Device,
        ctx: &crate::context::MetalContext,
    ) -> Result<VarBuilder, WeightError> {
        // SAFETY: Delegates to `from_mmaped_safetensors_with_ctx` — caller
        // upholds the mmap contract (files not modified while VarBuilder is alive).
        unsafe { from_mmaped_safetensors_with_ctx(paths, dtype, device, ctx) }
    }
}
