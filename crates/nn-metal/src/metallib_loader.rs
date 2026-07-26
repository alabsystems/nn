// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Load pre-compiled `.metallib` files and create compute pipelines.
//!
//! When `build.rs` compiles `.metal` sources to `.metallib` at build time,
//! this module loads the precompiled binary and creates `ComputePipeline`
//! objects without runtime MSL compilation.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_metal::metallib_loader::{load_metallib, pipelines_from_metallib};
//!
//! let library = load_metallib(&context, metallib_bytes)?;
//! let pipelines = pipelines_from_metallib(&context, &library, &["relu_kernel"])?;
//! ```

use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::ComputePipeline;

/// Load a pre-compiled `.metallib` from binary data.
///
/// Returns a Metal `Library` containing the precompiled functions.
///
/// # Errors
///
/// Returns `MetalError::MetallibLoad` if the data is not a valid metallib.
#[cfg(target_os = "macos")]
pub fn load_metallib(
    context: &MetalContext,
    metallib_bytes: &[u8],
) -> Result<metal::Library, MetalError> {
    context
        .device()
        .new_library_with_data(metallib_bytes)
        .map_err(MetalError::MetallibLoad)
}

/// Non-macOS stub.
#[cfg(not(target_os = "macos"))]
pub fn load_metallib(_context: &MetalContext, _metallib_bytes: &[u8]) -> Result<(), MetalError> {
    Err(MetalError::UnsupportedPlatform)
}

/// Create compute pipelines from a pre-compiled Metal library.
///
/// Looks up each function name in the library and creates a
/// `ComputePipeline` for it. Pipelines from metallib skip the MSL
/// compilation step entirely.
///
/// # Errors
///
/// Returns `MetalError::MissingEntryPoint` if a function name is not
/// found in the metallib.
#[cfg(target_os = "macos")]
pub fn pipelines_from_metallib(
    context: &MetalContext,
    library: &metal::Library,
    function_names: &[&str],
) -> Result<Vec<ComputePipeline>, MetalError> {
    let mut pipelines = Vec::with_capacity(function_names.len());

    for &name in function_names {
        let function = library
            .get_function(name, None)
            .map_err(|_| MetalError::MissingEntryPoint(name.to_owned()))?;

        let pipeline_state = context
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(MetalError::PipelineCreate)?;

        pipelines.push(ComputePipeline::from_raw(
            pipeline_state,
            name,
            false, // metallib pipelines don't use fast_math flag
        ));
    }

    Ok(pipelines)
}

/// Non-macOS stub.
#[cfg(not(target_os = "macos"))]
pub fn pipelines_from_metallib(
    _context: &MetalContext,
    _library: &(),
    _function_names: &[&str],
) -> Result<Vec<ComputePipeline>, MetalError> {
    Err(MetalError::UnsupportedPlatform)
}

/// Metallib bytes embedded into this binary at compile time.
///
/// `build.rs` always sets `NN_EMBEDDED_METALLIB`: it points at the compiled
/// `precompiled.metallib` when `.metal` sources were precompiled, or at an
/// empty placeholder file otherwise.
static EMBEDDED_METALLIB: &[u8] = include_bytes!(env!("NN_EMBEDDED_METALLIB"));

/// The precompiled `.metallib` embedded into this binary at compile time.
///
/// This is the proof-closed default shader-delivery path: the metallib bytes
/// are part of the compiled binary ("NNs compiled along with the program"),
/// so no runtime filesystem substitution is possible. Returns `None` when no
/// `.metal` sources were precompiled at build time (the common case during
/// development), in which case kernels are compiled at runtime from MSL
/// sources that are themselves embedded string constants — still no
/// filesystem involvement.
#[must_use]
pub fn embedded_metallib() -> Option<&'static [u8]> {
    (!EMBEDDED_METALLIB.is_empty()).then_some(EMBEDDED_METALLIB)
}

/// Path to the build-time precompiled metallib, if it was generated.
///
/// Returns `None` if no metallib was compiled at build time (the common
/// case during development). Returns `Some(path)` when `build.rs`
/// produced a `precompiled.metallib` and set `NN_PRECOMPILED_METALLIB`.
///
/// This path is **informational**. The default shader-delivery path is the
/// compile-time [`embedded_metallib`] bytes — the filesystem is never read
/// at runtime unless the caller explicitly opts in via
/// [`MetalInitOptions::allow_runtime_metallib`] **and** sets
/// `NN_ALLOW_RUNTIME_METALLIB=1` in the environment.
///
/// [`MetalInitOptions::allow_runtime_metallib`]: crate::MetalInitOptions::allow_runtime_metallib
#[must_use]
pub fn precompiled_metallib_path() -> Option<&'static str> {
    option_env!("NN_PRECOMPILED_METALLIB")
}
