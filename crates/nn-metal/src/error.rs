// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for the Metal backend.

use thiserror::Error;

/// Errors from the Metal compute backend.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MetalError {
    /// No Metal-capable GPU device found on this host.
    #[error("Metal is unavailable on this host")]
    NoDevice,
    /// MSL source failed to compile into a Metal library.
    #[error("failed to compile MSL source: {0}")]
    LibraryCompile(String),
    /// Compiled Metal library does not contain the expected kernel function.
    #[error("missing kernel entry point `{0}` in compiled library")]
    MissingEntryPoint(String),
    /// Metal pipeline state creation failed.
    #[error("failed to create compute pipeline: {0}")]
    PipelineCreate(String),
    /// Metal buffer allocation failed (e.g. out of GPU memory).
    #[error("failed to create buffer: size={0}")]
    BufferCreate(usize),
    /// GPU command buffer completed with a non-success status.
    #[error("Metal command buffer completed with error status: {0}")]
    DispatchFailed(String),
    /// Attempted to use Metal APIs on a non-macOS platform.
    #[error("nn-metal runtime helpers are only available on macOS")]
    UnsupportedPlatform,
    /// Kernel IR declares a different parameter count than was provided.
    #[error("kernel expects {expected} parameters but got {got}")]
    ParamCountMismatch { expected: usize, got: usize },
    /// Dispatch binding configuration is invalid.
    #[error("invalid dispatch bindings: {0}")]
    InvalidDispatchBindings(&'static str),
    /// Input data slice length does not match the expected element count.
    #[error("input slice {index} has length {got}, expected {expected}")]
    InputLenMismatch {
        expected: usize,
        got: usize,
        index: usize,
    },
    /// Total dispatch element count exceeds `u32::MAX`.
    #[error("dispatch element count {0} exceeds u32::MAX")]
    DispatchSizeOverflow(usize),
    /// Output buffer byte count overflows `usize`.
    #[error("output buffer byte count overflows: {elems} elements \u{d7} {elem_size} bytes")]
    BufferByteOverflow { elems: usize, elem_size: usize },
    /// A grid or threadgroup dimension is zero.
    #[error("invalid {dimension} dimension: {value} (must be non-zero)")]
    InvalidGridDimension { dimension: &'static str, value: u32 },
    /// Kernel IR validation failed before dispatch.
    #[error("kernel IR validation failed: {0}")]
    IRValidation(#[from] nn_dsl::ir::IRError),
    /// Global Metal backend not initialized — call [`MetalBackend::init`] first.
    #[error("Metal backend not initialized -- call MetalBackend::init() first")]
    UninitializedBackend,
    /// `mmap` returned a null pointer when loading a weight file.
    #[error("mmap returned null pointer for weight file")]
    NullMmapPointer,
    /// Command buffer is in an error/committed state; cannot create encoder.
    #[error("cannot create compute encoder: command buffer status is {0}")]
    EncoderCreate(String),
    /// Buffer backing pointer is not page-aligned for no-copy creation.
    #[error("buffer pointer not page-aligned: ptr=0x{ptr:x}, len={len}, page_size={page_size}")]
    BufferAlignment {
        ptr: usize,
        len: usize,
        page_size: usize,
    },
    /// Typed buffer readback failed (ZST, empty buffer, or alignment issue).
    #[error("buffer readback failed: {reason} (buf_len={buf_len}, type_size={type_size})")]
    BufferReadback {
        reason: &'static str,
        buf_len: usize,
        type_size: usize,
    },
    /// Arena sub-allocation exceeds remaining capacity.
    #[error(
        "arena overflow: requested {requested} bytes, {remaining} remaining of {capacity} total"
    )]
    ArenaOverflow {
        requested: usize,
        remaining: usize,
        capacity: usize,
    },
    /// Arena alignment value is not a power of two.
    #[error("arena alignment {alignment} is not a power of two")]
    InvalidArenaAlignment { alignment: usize },
    /// CPU-side buffer read attempted while lazy-batch GPU encodings are pending.
    ///
    /// Call `flush()` before reading buffer contents to ensure all pending GPU
    /// dispatches have committed. Without this, `contents()` returns stale data.
    /// See #1912 and #1933 for prior incidents.
    #[error(
        "flush() required before CPU buffer read — {pending_count} pending lazy-batch encodings"
    )]
    PendingFlushRequired { pending_count: usize },
    /// Pre-compiled `.metallib` data is invalid or corrupted.
    #[error("failed to load metallib: {0}")]
    MetallibLoad(String),
    /// Runtime (filesystem) metallib loading requested without the
    /// environment guard.
    ///
    /// The proof-closed default embeds shaders at compile time. Loading a
    /// `.metallib` from the filesystem at runtime requires the double
    /// opt-in: `MetalInitOptions::allow_runtime_metallib(true)` **and**
    /// `NN_ALLOW_RUNTIME_METALLIB=1` in the environment.
    #[error(
        "runtime .metallib loading is disabled: set NN_ALLOW_RUNTIME_METALLIB=1 in the \
         environment (in addition to MetalInitOptions::allow_runtime_metallib(true)) to \
         explicitly opt in — the default is compile-time embedded shaders only"
    )]
    RuntimeMetallibDisabled,
    /// Runtime metallib loading was explicitly enabled but the file cannot
    /// be used (missing build-time path, unreadable file, …).
    ///
    /// Explicitly requested runtime loading fails hard rather than silently
    /// falling back to another shader source.
    #[error("runtime .metallib unavailable: {path}: {reason}")]
    RuntimeMetallibUnavailable { path: String, reason: String },
    /// Blit copy source or destination offset+size exceeds buffer byte length.
    #[error(
        "blit_copy bounds exceeded: offset={offset} + size={size} > buffer_len={buffer_len} ({role})"
    )]
    BufferBoundsExceeded {
        buffer_len: usize,
        offset: usize,
        size: usize,
        role: &'static str,
    },
    /// CPU readback of an arena-backed tensor whose memory may have been
    /// overwritten by a subsequent arena generation.
    ///
    /// The tensor was allocated during arena generation `alloc_gen`, but the
    /// default arena is now at `current_gen` — meaning the arena was reset
    /// (and potentially re-allocated into) since this tensor was created.
    /// The buffer's ObjC ARC keeps the Metal allocation alive, but the
    /// *contents* may be from a different computation.
    ///
    /// See `designs/2026-03-14-arena-cross-thread-safety.md`.
    #[error("stale arena read: tensor from generation {alloc_gen}, arena now at {current_gen}")]
    StaleArenaRead { alloc_gen: u64, current_gen: u64 },
    /// GPU command buffer did not complete within the allowed timeout.
    ///
    /// Prevents a hung GPU shader or wedged Metal driver from blocking
    /// indefinitely and causing a macOS watchdog kernel panic. The timeout
    /// is set to [`GPU_TIMEOUT`](crate::dispatch::GPU_TIMEOUT).
    #[error("GPU command buffer timed out after {0:?} — possible GPU hang")]
    GpuTimeout(std::time::Duration),
    /// Arena checkpoint restore failed: saved offset is ahead of current offset.
    #[error("arena checkpoint restore failed: saved={saved} > current={current}")]
    ArenaCheckpoint { saved: usize, current: usize },
    /// Buffer byte offset exceeds buffer length.
    ///
    /// Returned when a byte offset used for GPU buffer binding exceeds the
    /// buffer's allocated byte length. This catches out-of-bounds Metal buffer
    /// accesses before they reach the GPU, preventing undefined behavior.
    /// Part of #4321.
    #[error(
        "buffer offset out of bounds: offset={offset} exceeds buffer_len={buffer_len} ({role})"
    )]
    BufferOffsetOutOfBounds {
        buffer_len: usize,
        offset: usize,
        role: &'static str,
    },
}

impl From<MetalError> for nn_core::TensorError {
    fn from(e: MetalError) -> Self {
        let kind = match &e {
            MetalError::BufferCreate(_) | MetalError::ArenaOverflow { .. } => {
                nn_core::BackendErrorKind::OutOfMemory
            }
            MetalError::LibraryCompile(_)
            | MetalError::MissingEntryPoint(_)
            | MetalError::PipelineCreate(_) => nn_core::BackendErrorKind::KernelCompile,
            MetalError::DispatchFailed(_)
            | MetalError::EncoderCreate(_)
            | MetalError::GpuTimeout(_) => nn_core::BackendErrorKind::DispatchFailed,
            _ => nn_core::BackendErrorKind::Other,
        };
        let msg = e.to_string();
        Self::backend_failure_with_source(
            nn_core::BackendDomain::Metal,
            kind,
            msg,
            e,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::MetalError;

    #[test]
    fn test_no_device_display() {
        let err = MetalError::NoDevice;
        assert_eq!(err.to_string(), "Metal is unavailable on this host");
    }

    #[test]
    fn test_library_compile_display() {
        let err = MetalError::LibraryCompile("syntax error at line 5".into());
        assert_eq!(
            err.to_string(),
            "failed to compile MSL source: syntax error at line 5"
        );
    }

    #[test]
    fn test_missing_entry_point_display() {
        let err = MetalError::MissingEntryPoint("nn_kernel".into());
        assert_eq!(
            err.to_string(),
            "missing kernel entry point `nn_kernel` in compiled library"
        );
    }

    #[test]
    fn test_pipeline_create_display() {
        let err = MetalError::PipelineCreate("internal error".into());
        assert_eq!(
            err.to_string(),
            "failed to create compute pipeline: internal error"
        );
    }

    #[test]
    fn test_buffer_create_display() {
        let err = MetalError::BufferCreate(0);
        assert_eq!(err.to_string(), "failed to create buffer: size=0");
    }

    #[test]
    fn test_unsupported_platform_display() {
        let err = MetalError::UnsupportedPlatform;
        assert_eq!(
            err.to_string(),
            "nn-metal runtime helpers are only available on macOS"
        );
    }

    #[test]
    fn test_param_count_mismatch_display() {
        let err = MetalError::ParamCountMismatch {
            expected: 3,
            got: 1,
        };
        assert_eq!(err.to_string(), "kernel expects 3 parameters but got 1");
    }

    #[test]
    fn test_input_len_mismatch_display() {
        let err = MetalError::InputLenMismatch {
            expected: 100,
            got: 50,
            index: 2,
        };
        assert_eq!(err.to_string(), "input slice 2 has length 50, expected 100");
    }

    #[test]
    fn test_invalid_dispatch_bindings_display() {
        let err =
            MetalError::InvalidDispatchBindings("at least one writable parameter role is required");
        assert_eq!(
            err.to_string(),
            "invalid dispatch bindings: at least one writable parameter role is required"
        );
    }

    #[test]
    fn test_dispatch_size_overflow_display() {
        let err = MetalError::DispatchSizeOverflow(5_000_000_000);
        assert_eq!(
            err.to_string(),
            "dispatch element count 5000000000 exceeds u32::MAX"
        );
    }

    #[test]
    fn test_invalid_grid_dimension_display() {
        let err = MetalError::InvalidGridDimension {
            dimension: "outer",
            value: 0,
        };
        assert_eq!(
            err.to_string(),
            "invalid outer dimension: 0 (must be non-zero)"
        );
    }

    #[test]
    fn test_null_mmap_pointer_display() {
        let err = MetalError::NullMmapPointer;
        assert_eq!(
            err.to_string(),
            "mmap returned null pointer for weight file"
        );
    }

    #[test]
    fn test_encoder_create_display() {
        let err = MetalError::EncoderCreate("Error".into());
        assert_eq!(
            err.to_string(),
            "cannot create compute encoder: command buffer status is Error"
        );
    }

    #[test]
    fn test_buffer_byte_overflow_display() {
        let err = MetalError::BufferByteOverflow {
            elems: 5_000_000_000,
            elem_size: 4,
        };
        assert_eq!(
            err.to_string(),
            "output buffer byte count overflows: 5000000000 elements \u{d7} 4 bytes"
        );
    }

    #[test]
    fn test_buffer_readback_display() {
        let err = MetalError::BufferReadback {
            reason: "zero-size type",
            buf_len: 1024,
            type_size: 0,
        };
        assert_eq!(
            err.to_string(),
            "buffer readback failed: zero-size type (buf_len=1024, type_size=0)"
        );
    }

    #[test]
    fn test_buffer_alignment_display() {
        let err = MetalError::BufferAlignment {
            ptr: 0x1001,
            len: 100,
            page_size: 4096,
        };
        assert_eq!(
            err.to_string(),
            "buffer pointer not page-aligned: ptr=0x1001, len=100, page_size=4096"
        );
    }

    #[test]
    fn test_pending_flush_required_display() {
        let err = MetalError::PendingFlushRequired { pending_count: 3 };
        assert_eq!(
            err.to_string(),
            "flush() required before CPU buffer read — 3 pending lazy-batch encodings"
        );
    }

    #[test]
    fn test_buffer_bounds_exceeded_display() {
        let err = MetalError::BufferBoundsExceeded {
            buffer_len: 1024,
            offset: 900,
            size: 200,
            role: "source",
        };
        assert_eq!(
            err.to_string(),
            "blit_copy bounds exceeded: offset=900 + size=200 > buffer_len=1024 (source)"
        );
    }

    #[test]
    fn test_stale_arena_read_display() {
        let err = MetalError::StaleArenaRead {
            alloc_gen: 3,
            current_gen: 7,
        };
        assert_eq!(
            err.to_string(),
            "stale arena read: tensor from generation 3, arena now at 7"
        );
    }

    #[test]
    fn test_gpu_timeout_display() {
        let err = MetalError::GpuTimeout(std::time::Duration::from_secs(60));
        assert!(err.to_string().contains("timed out after 60s"));
    }

    #[test]
    fn test_runtime_metallib_disabled_display_names_the_guard() {
        let err = MetalError::RuntimeMetallibDisabled;
        let msg = err.to_string();
        assert!(
            msg.contains("NN_ALLOW_RUNTIME_METALLIB"),
            "error must name the environment guard: {msg}"
        );
        assert!(
            msg.contains("allow_runtime_metallib"),
            "error must name the config flag: {msg}"
        );
    }

    #[test]
    fn test_runtime_metallib_unavailable_display() {
        let err = MetalError::RuntimeMetallibUnavailable {
            path: "/tmp/x.metallib".into(),
            reason: "No such file or directory".into(),
        };
        assert_eq!(
            err.to_string(),
            "runtime .metallib unavailable: /tmp/x.metallib: No such file or directory"
        );
    }

    #[test]
    fn test_buffer_offset_out_of_bounds_display() {
        let err = MetalError::BufferOffsetOutOfBounds {
            buffer_len: 1024,
            offset: 2048,
            role: "input",
        };
        assert_eq!(
            err.to_string(),
            "buffer offset out of bounds: offset=2048 exceeds buffer_len=1024 (input)"
        );
    }
}
