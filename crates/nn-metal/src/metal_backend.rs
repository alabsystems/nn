// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal GPU backend for nn-core Tensor.
//!
//! Implements [`nn_core::Backend`] for Metal, bridging the typed tensor API
//! to GPU-allocated buffers. Uses `OnceLock` for the global Metal context
//! (single-device, matches dvoice M4 Max target).
//!
//! Direction 2 methods (`from_metal_buffer`, `to_cpu`, `metal_buffer`) operate
//! on `Tensor<D, T, MetalBackend>` via [`MetalTensorExt`] and require
//! `T: bytemuck::Pod` for safe GPU-CPU data transfer.

use std::sync::{Arc, Mutex, OnceLock};

use nn_core::backend::{Backend, CpuBackend};
use nn_core::{BackendDomain, BackendErrorKind, Device, Tensor, TensorElement, TensorError};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;

/// Global Metal context -- initialized once via [`MetalBackend::init`].
///
/// Using `OnceLock` rather than thread-local: simpler, works across threads
/// (important for data loading), single-device limitation matches dvoice.
///
/// `METAL_INIT` mutex guards the initialization path because
/// `OnceLock::get_or_try_init` is unstable.
static METAL_CTX: OnceLock<Arc<MetalContext>> = OnceLock::new();
static METAL_INIT: Mutex<()> = Mutex::new(());

/// Metal GPU tensor storage.
///
/// Wraps a `MetalBuffer` (shared-mode) with element count metadata.
/// `Clone` creates a new `Arc` reference, not a GPU buffer copy.
#[derive(Clone, Debug)]
pub struct MetalTensorStorage {
    buffer: Arc<MetalBuffer>,
    len_elements: usize,
}

impl MetalTensorStorage {
    /// Create a new `MetalTensorStorage` from a buffer and element count.
    pub(crate) fn new(buffer: Arc<MetalBuffer>, len_elements: usize) -> Self {
        Self {
            buffer,
            len_elements,
        }
    }

    /// Number of logical elements in this storage.
    #[must_use]
    pub fn len_elements(&self) -> usize {
        self.len_elements
    }

    /// Get a reference to the underlying Metal buffer.
    #[must_use]
    pub fn buffer(&self) -> &MetalBuffer {
        &self.buffer
    }
}

/// Metal GPU backend for nn-core Tensor.
///
/// Unlike `CpuBackend` (which is stateless), `MetalBackend` requires
/// a `MetalContext` (device + command queue) for GPU allocation. The context
/// is stored in a global `OnceLock` initialized via [`MetalBackend::init`].
#[derive(Clone, Debug)]
pub struct MetalBackend {
    context: Arc<MetalContext>,
}

/// Environment guard required — in addition to
/// [`MetalInitOptions::allow_runtime_metallib`] — before a `.metallib` may be
/// loaded from the filesystem at runtime. Must be set to `1`.
pub const RUNTIME_METALLIB_ENV_GUARD: &str = "NN_ALLOW_RUNTIME_METALLIB";

/// Options for [`MetalBackend::init_with`].
///
/// The default (used by [`MetalBackend::init`]) is the proof-closed
/// configuration: precompiled shader pipelines come only from the metallib
/// bytes embedded in the binary at compile time
/// ([`crate::metallib_loader::embedded_metallib`]); the filesystem is never
/// read at runtime.
#[derive(Debug, Clone, Default)]
pub struct MetalInitOptions {
    allow_runtime_metallib: bool,
}

impl MetalInitOptions {
    /// Create the default (proof-closed, embedded-only) options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Explicitly allow loading a `.metallib` from the filesystem at runtime.
    ///
    /// This is a deliberate escape hatch from the proof-closed doctrine
    /// ("weights and shaders embedded, no runtime substitution") and requires
    /// a second, environment-level opt-in: [`RUNTIME_METALLIB_ENV_GUARD`]
    /// set to `1` (`NN_ALLOW_RUNTIME_METALLIB=1`).
    /// With this flag set but the environment guard absent,
    /// [`MetalBackend::init_with`] returns
    /// [`MetalError::RuntimeMetallibDisabled`] instead of silently falling
    /// back. When the runtime load does happen it is logged loudly on
    /// stderr — never silently.
    #[must_use]
    pub fn allow_runtime_metallib(mut self, allow: bool) -> Self {
        self.allow_runtime_metallib = allow;
        self
    }

    /// Whether runtime filesystem metallib loading is requested.
    #[must_use]
    pub fn runtime_metallib_allowed(&self) -> bool {
        self.allow_runtime_metallib
    }
}

impl MetalBackend {
    /// Initialize the Metal backend with the system default device.
    ///
    /// Must be called before any `Tensor<_, _, MetalBackend>` allocation.
    /// Subsequent calls return the same context (idempotent via `OnceLock`).
    ///
    /// Uses [`MetalInitOptions::default`]: precompiled shader pipelines are
    /// sourced exclusively from the compile-time embedded metallib — the
    /// filesystem is never read. See [`MetalBackend::init_with`] for the
    /// explicit runtime-loading opt-in.
    #[must_use = "returns a Result that may contain an error"]
    pub fn init() -> Result<Self, MetalError> {
        Self::init_with(MetalInitOptions::default())
    }

    /// Initialize the Metal backend with explicit [`MetalInitOptions`].
    ///
    /// The global context and precompiled-pipeline store are initialized
    /// exactly once: the **first** successful call decides the shader
    /// source. Subsequent calls return the existing context; if `options`
    /// requests runtime metallib loading at that point, the request is
    /// loudly ignored (warning on stderr) because the decision has already
    /// been made. To guarantee the opt-in is honored, make this the first
    /// initialization call in the process.
    ///
    /// # Errors
    ///
    /// - [`MetalError::RuntimeMetallibDisabled`] if
    ///   [`MetalInitOptions::allow_runtime_metallib`] is set but the
    ///   [`RUNTIME_METALLIB_ENV_GUARD`] environment variable is not `1`.
    /// - [`MetalError::RuntimeMetallibUnavailable`] if runtime loading is
    ///   fully enabled but no build-time metallib path exists or the file
    ///   cannot be read. Explicit requests fail hard — no silent fallback.
    /// - [`MetalError::MetallibLoad`] if the selected metallib (embedded or
    ///   runtime) cannot be loaded.
    #[must_use = "returns a Result that may contain an error"]
    pub fn init_with(options: MetalInitOptions) -> Result<Self, MetalError> {
        // Fast path: already initialized.
        if let Some(ctx) = METAL_CTX.get() {
            warn_if_late_runtime_request(&options);
            return Ok(Self {
                context: ctx.clone(),
            });
        }
        // Slow path: initialize under lock to avoid double-init.
        let _lock = METAL_INIT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Re-check after acquiring lock (another thread may have initialized).
        if let Some(ctx) = METAL_CTX.get() {
            warn_if_late_runtime_request(&options);
            return Ok(Self {
                context: ctx.clone(),
            });
        }

        // Decide the precompiled-shader source *before* creating the
        // context so a misconfigured opt-in fails hard with no partial
        // global state.
        let source = resolve_metallib_source(
            &options,
            std::env::var(RUNTIME_METALLIB_ENV_GUARD).ok().as_deref(),
            crate::metallib_loader::precompiled_metallib_path(),
            crate::metallib_loader::embedded_metallib(),
        )?;

        let ctx = Arc::new(MetalContext::new()?);
        load_metallib_source(&ctx, source)?;
        let _ = METAL_CTX.set(ctx.clone());

        Ok(Self { context: ctx })
    }

    /// Get the underlying Metal context.
    #[must_use]
    pub fn context(&self) -> &MetalContext {
        &self.context
    }
}

/// Get the global Metal context, returning a user-facing `TensorError` if uninitialized.
fn get_metal_context() -> Result<Arc<MetalContext>, TensorError> {
    METAL_CTX.get().cloned().ok_or_else(|| {
        TensorError::backend_failure(
            BackendDomain::Metal,
            BackendErrorKind::Other,
            "MetalBackend not initialized -- call MetalBackend::init() first".into(),
        )
    })
}

/// Get a reference to the global Metal context.
///
/// Returns `MetalError::UninitializedBackend` if [`MetalBackend::init`] has
/// not been called. Used by convenience constructors that avoid requiring
/// consumers to thread `MetalContext` manually.
pub(crate) fn global_metal_context() -> Result<&'static MetalContext, MetalError> {
    METAL_CTX
        .get()
        .map(Arc::as_ref)
        .ok_or(MetalError::UninitializedBackend)
}

/// Where [`MetalBackend::init_with`] sources precompiled shader pipelines
/// from (#2467, proof-closed since the runtime-fallback removal).
#[derive(Debug, PartialEq, Eq)]
enum MetallibSource {
    /// No precompiled metallib: kernels compile at runtime from MSL sources
    /// embedded in the binary as string constants (no filesystem involved).
    None,
    /// Compile-time embedded metallib bytes — the proof-closed default.
    Embedded(&'static [u8]),
    /// Explicit, double-opt-in load from the filesystem at runtime.
    RuntimeFile(&'static str),
}

/// Decide the precompiled-shader source from options + environment.
///
/// Rules:
/// - Default: the compile-time embedded metallib (or nothing, if none was
///   embedded). The filesystem is **never** touched, even when a build-time
///   metallib path is known and even when the environment guard is set.
/// - [`MetalInitOptions::allow_runtime_metallib`] requests a runtime
///   filesystem load. It additionally requires
///   [`RUNTIME_METALLIB_ENV_GUARD`] set to `1`; otherwise this is a hard
///   error — an explicit request is never silently downgraded.
fn resolve_metallib_source(
    options: &MetalInitOptions,
    env_guard: Option<&str>,
    build_time_path: Option<&'static str>,
    embedded: Option<&'static [u8]>,
) -> Result<MetallibSource, MetalError> {
    if options.allow_runtime_metallib {
        if env_guard != Some("1") {
            return Err(MetalError::RuntimeMetallibDisabled);
        }
        let Some(path) = build_time_path else {
            return Err(MetalError::RuntimeMetallibUnavailable {
                path: "<no build-time metallib>".to_owned(),
                reason: "no .metallib was produced at build time \
                         (NN_PRECOMPILED_METALLIB unset)"
                    .to_owned(),
            });
        };
        return Ok(MetallibSource::RuntimeFile(path));
    }
    Ok(embedded.map_or(MetallibSource::None, MetallibSource::Embedded))
}

/// Load the resolved metallib source into the precompiled pipeline store.
///
/// Runtime filesystem loads are logged loudly on stderr — never silent.
/// Failures are hard errors; there is no fallthrough to another source.
fn load_metallib_source(ctx: &MetalContext, source: MetallibSource) -> Result<(), MetalError> {
    match source {
        MetallibSource::None => Ok(()),
        MetallibSource::Embedded(bytes) => load_precompiled_pipelines(ctx, bytes, "embedded"),
        MetallibSource::RuntimeFile(path) => {
            let bytes =
                std::fs::read(path).map_err(|e| MetalError::RuntimeMetallibUnavailable {
                    path: path.to_owned(),
                    reason: e.to_string(),
                })?;
            eprintln!("{}", runtime_metallib_warning(path));
            load_precompiled_pipelines(ctx, &bytes, "runtime")
        }
    }
}

/// The loud warning emitted when a `.metallib` is loaded from the filesystem
/// at runtime (explicit double opt-in only).
fn runtime_metallib_warning(path: &str) -> String {
    format!(
        "[nn-metal] WARNING: loading .metallib from the filesystem at runtime: {path} \
         (explicitly enabled via MetalInitOptions::allow_runtime_metallib(true) + \
         {RUNTIME_METALLIB_ENV_GUARD}=1; compile-time embedded shaders are bypassed)"
    )
}

/// Populate the precompiled pipeline store from metallib bytes.
///
/// A metallib that parses but contains none of the expected kernel entry
/// points is a hard error — a selected source that provides nothing must
/// not silently degrade to runtime MSL compilation.
fn load_precompiled_pipelines(
    ctx: &MetalContext,
    bytes: &[u8],
    origin: &str,
) -> Result<(), MetalError> {
    // Collect entry points from the precompile module.
    let sources = crate::precompile::collect_native_kernel_sources();
    let entry_points: Vec<&str> = sources.iter().map(|s| s.entry_point).collect();

    let count = crate::cache::load_precompiled_metallib(ctx, bytes, &entry_points)?;
    if count == 0 {
        return Err(MetalError::MetallibLoad(format!(
            "{origin} metallib contains none of the {} expected kernel entry points",
            entry_points.len()
        )));
    }
    eprintln!("[nn-metal] loaded {count} precompiled pipelines from {origin} metallib");
    Ok(())
}

/// Loudly note a runtime-metallib request that arrived after the global
/// backend was already initialized (the shader-source decision is made by
/// the first successful init and cannot be changed).
fn warn_if_late_runtime_request(options: &MetalInitOptions) {
    if options.allow_runtime_metallib {
        eprintln!(
            "[nn-metal] WARNING: allow_runtime_metallib(true) ignored — MetalBackend is \
             already initialized; the shader source was decided by the first init call"
        );
    }
}

/// Compute checked product of dimensions, returning `TensorError::DimensionOverflow` on overflow.
pub(crate) fn checked_dim_product(dims: &[usize]) -> Result<usize, TensorError> {
    dims.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: dims.to_vec(),
            })
    })
}

/// Map any displayable error to a `TensorError::BackendFailure` with Metal domain.
pub(crate) fn metal_err(e: impl std::fmt::Display) -> TensorError {
    TensorError::backend_failure(
        BackendDomain::Metal,
        BackendErrorKind::DispatchFailed,
        e.to_string(),
    )
}

impl Backend for MetalBackend {
    type TensorPrimitive<T: TensorElement> = MetalTensorStorage;

    fn device() -> Device {
        Device::metal()
    }

    fn zeros<const D: usize, T: TensorElement>(
        dims: [usize; D],
    ) -> nn_core::Result<Self::TensorPrimitive<T>> {
        let numel = checked_dim_product(&dims)?;
        let byte_len =
            numel
                .checked_mul(size_of::<T>())
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;
        let ctx = get_metal_context()?;
        let buffer = ctx.create_buffer_zeroed(byte_len).map_err(metal_err)?;
        Ok(MetalTensorStorage {
            buffer: Arc::new(buffer),
            len_elements: numel,
        })
    }

    fn ones<const D: usize, T: TensorElement>(
        dims: [usize; D],
    ) -> nn_core::Result<Self::TensorPrimitive<T>> {
        let numel = checked_dim_product(&dims)?;
        let type_size = size_of::<T>();
        let byte_len =
            numel
                .checked_mul(type_size)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;
        let ctx = get_metal_context()?;
        let buffer = ctx.create_buffer_zeroed(byte_len).map_err(metal_err)?;

        let one = T::one();
        // SAFETY: MetalBuffer was just created with `byte_len` bytes in shared
        // mode. The pointer is valid for `numel * type_size` bytes of writing.
        // T: Copy (from TensorElement) ensures no drop/init invariants.
        // All TensorElement impls are primitive types that are also
        // bytemuck::Pod, so copying their byte representation is well-defined.
        // The buffer is not yet submitted to the GPU, so there are no data
        // races.
        unsafe {
            let ptr = buffer.inner().contents().cast::<u8>();
            if ptr.is_null() {
                return Err(metal_err(MetalError::BufferReadback {
                    reason: "null buffer pointer in ones() init",
                    buf_len: byte_len,
                    type_size,
                }));
            }
            let one_ptr = (&raw const one).cast::<u8>();
            for i in 0..numel {
                std::ptr::copy_nonoverlapping(one_ptr, ptr.add(i * type_size), type_size);
            }
        }

        Ok(MetalTensorStorage {
            buffer: Arc::new(buffer),
            len_elements: numel,
        })
    }
}

// -- Metal-specific Tensor methods (Direction 2: #748 AC2+AC3) ----------------

/// Extension trait for Metal-backed tensors.
///
/// Provides GPU to CPU transfer and buffer access. Requires `T: bytemuck::Pod`
/// for safe reinterpretation of GPU buffer bytes.
pub trait MetalTensorExt<const D: usize, T: TensorElement + bytemuck::Pod> {
    /// Copy GPU buffer contents to a CPU tensor.
    #[must_use = "returns a Result that may contain an error"]
    fn to_cpu(&self) -> nn_core::Result<Tensor<D, T, CpuBackend>>;

    /// Get a reference to the underlying [`MetalBuffer`].
    #[must_use]
    fn metal_buffer(&self) -> &MetalBuffer;
}

impl<const D: usize, T: TensorElement + bytemuck::Pod> MetalTensorExt<D, T>
    for Tensor<D, T, MetalBackend>
{
    fn to_cpu(&self) -> nn_core::Result<Tensor<D, T, CpuBackend>> {
        // Flush the lazy GPU command batch before CPU readback.
        // Without this, GPU ops encoded into the lazy batch but not yet committed
        // would produce stale/zeroed data. Same pattern as gpu_to_cpu() in
        // dyn_tensor_metal_helpers.rs. See #1912, #1933 for prior stale-data bugs.
        crate::gpu_scope::flush()?;
        let data: &[T] = self.storage().buffer().contents::<T>().map_err(metal_err)?;
        let numel = self.numel();
        if data.len() < numel {
            return Err(TensorError::DataLengthMismatch {
                expected: numel,
                actual: data.len(),
            });
        }
        Tensor::<D, T, CpuBackend>::from_vec(*self.dims(), data[..numel].to_vec())
    }

    fn metal_buffer(&self) -> &MetalBuffer {
        self.storage().buffer()
    }
}

/// Wrap an existing [`MetalBuffer`] as a typed tensor.
///
/// The buffer must hold at least `dims.product() * size_of::<T>()` bytes.
/// Ownership of `buffer` is moved into an `Arc` inside `MetalTensorStorage`.
#[must_use = "returns a Result that may contain an error"]
pub fn from_metal_buffer<const D: usize, T: TensorElement + bytemuck::Pod>(
    dims: [usize; D],
    buffer: MetalBuffer,
) -> nn_core::Result<Tensor<D, T, MetalBackend>> {
    let numel = checked_dim_product(&dims)?;
    let expected_bytes =
        numel
            .checked_mul(size_of::<T>())
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: dims.to_vec(),
            })?;
    if buffer.len() < expected_bytes {
        return Err(TensorError::DataLengthMismatch {
            expected: numel,
            actual: buffer.len() / size_of::<T>(),
        });
    }
    let storage = MetalTensorStorage::new(Arc::new(buffer), numel);
    Tensor::from_storage(dims, storage)
}

#[cfg(test)]
#[path = "metal_backend_tests.rs"]
mod tests;
