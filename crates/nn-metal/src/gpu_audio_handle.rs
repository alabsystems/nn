// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-resident audio buffer handle.
//!
//! [`GpuAudioHandle`] wraps a [`MetalBuffer`] containing PCM f32 audio data
//! on the GPU. Returned by `synthesize_gpu()` to defer the GPU-to-CPU transfer,
//! enabling zero-flush synthesis in the hot path.
//!
//! # Usage
//!
//! ```rust,no_run
//! # use nn_metal::GpuAudioHandle;
//! // let handle = kokoro.synthesize_gpu(&ids, &style, 1.0, &cache)?;
//! // ... do other work while audio is on GPU ...
//! // let pcm: Vec<f32> = handle.to_cpu()?;
//! ```

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::buffer::MetalBuffer;
use crate::dyn_tensor_metal::MetalTensorData;
use crate::error::MetalError;
use crate::gpu_fence::GpuFence;

/// GPU-resident audio buffer handle.
///
/// Wraps a [`MetalBuffer`] containing PCM f32 audio data on the GPU.
/// Returned by `synthesize_gpu()` to defer the GPU-to-CPU transfer.
///
/// # Zero-flush pattern
///
/// The standard `synthesize()` call returns `Vec<f32>`, requiring a `flush()`
/// to transfer GPU data to CPU. `GpuAudioHandle` defers that readback so
/// callers can pipeline other work before the transfer:
///
/// ```rust,no_run
/// # use nn_metal::GpuAudioHandle;
/// // let handle = kokoro.synthesize_gpu(&ids, &style, 1.0, &cache)?;
/// // ... queue more GPU work, prepare output buffers, etc. ...
/// // let pcm: Vec<f32> = handle.to_cpu()?;  // flush + readback happens here
/// ```
///
/// # Fence-backed pattern
///
/// When created with [`with_fence`](Self::with_fence), the handle carries a
/// [`GpuFence`] that tracks completion of the GPU work that produced the audio.
/// This avoids the global `flush()` call and enables independent tracking:
///
/// ```rust,no_run
/// # use nn_metal::GpuAudioHandle;
/// // let handle = kokoro.synthesize_gpu_fenced(&ids, &style, 1.0, &cache)?;
/// // ... CPU work while GPU runs ...
/// // assert!(handle.is_ready());  // non-blocking check
/// // let pcm: Vec<f32> = handle.to_cpu()?;  // waits for fence, then reads
/// ```
pub struct GpuAudioHandle {
    buffer: MetalBuffer,
    sample_count: usize,
    sample_rate: u32,
    /// Optional fence tracking completion of the GPU work that produced this
    /// audio buffer. When present, [`to_cpu`](Self::to_cpu) waits on the fence
    /// instead of calling the global `flush()`.
    fence: Option<GpuFence>,
}

// SAFETY: MetalBuffer wraps an Objective-C metal::Buffer allocated with
// StorageModeShared. After GPU work is committed (which `to_cpu()` ensures
// via `flush()`), the shared-mode buffer can be safely read from any thread.
// This is the same safety argument used by `MetalTensorData` and `WeightMap`.
unsafe impl Send for GpuAudioHandle {}

impl GpuAudioHandle {
    /// Create from a GPU buffer containing f32 PCM audio.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Metal buffer containing at least `sample_count * 4` bytes
    ///   of f32 PCM audio data.
    /// * `sample_count` - Number of f32 audio samples in the buffer.
    /// * `sample_rate` - Audio sample rate in Hz (e.g. 24000 for Kokoro).
    #[allow(dead_code)] // Prepared for chorus system synthesize_gpu()
    pub(crate) fn new(buffer: MetalBuffer, sample_count: usize, sample_rate: u32) -> Self {
        Self {
            buffer,
            sample_count,
            sample_rate,
            fence: None,
        }
    }

    /// Create from a GPU buffer with an attached fence for non-blocking submit.
    ///
    /// The fence tracks completion of the GPU work that produced this audio
    /// buffer. [`to_cpu`](Self::to_cpu) waits on the fence instead of calling
    /// the global `flush()`, enabling independent tracking of multiple
    /// outstanding GPU submissions.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Metal buffer containing at least `sample_count * 4` bytes
    ///   of f32 PCM audio data.
    /// * `sample_count` - Number of f32 audio samples in the buffer.
    /// * `sample_rate` - Audio sample rate in Hz (e.g. 24000 for Kokoro).
    /// * `fence` - Fence tracking completion of the GPU work.
    #[allow(dead_code)] // Prepared for pipelined chorus synthesize_gpu()
    pub(crate) fn with_fence(
        buffer: MetalBuffer,
        sample_count: usize,
        sample_rate: u32,
        fence: GpuFence,
    ) -> Self {
        Self {
            buffer,
            sample_count,
            sample_rate,
            fence: Some(fence),
        }
    }

    /// Transfer audio from GPU to CPU.
    ///
    /// If this handle has an attached [`GpuFence`], waits on the fence
    /// (targeted wait for the specific GPU submission). Otherwise, calls
    /// [`flush()`](crate::gpu_scope::flush) to ensure all pending GPU work
    /// is complete. Then reads the buffer contents into a `Vec<f32>`.
    pub fn to_cpu(&self) -> Result<Vec<f32>, MetalError> {
        // Ensure GPU work is complete before reading buffer contents.
        if let Some(fence) = &self.fence {
            // Fence-backed: wait only for the specific GPU submission.
            fence.wait_timeout(std::time::Duration::from_mins(1)).map_err(|e| {
                MetalError::DispatchFailed(format!(
                    "GpuAudioHandle::to_cpu fence wait failed: {e}"
                ))
            })?;
        } else {
            // No fence: flush all pending GPU work (legacy path).
            crate::gpu_scope::flush().map_err(|e| {
                MetalError::DispatchFailed(format!("GpuAudioHandle::to_cpu flush failed: {e}"))
            })?;
        }

        // SAFETY:
        // 1. GPU work is complete (fence wait or flush above).
        // 2. `sample_count` was set at construction by `pub(crate) fn new()`,
        //    which guarantees the caller allocated the buffer with at least
        //    `sample_count * size_of::<f32>()` bytes.
        // 3. The buffer was allocated with StorageModeShared, so CPU can read.
        let float_slice: &[f32] = self
            .buffer
            .contents_at_offset::<f32>(0, self.sample_count)?;
        Ok(float_slice.to_vec())
    }

    /// Check if the GPU work producing this audio is complete.
    ///
    /// Returns `true` if:
    /// - This handle has a fence and the GPU work has completed, OR
    /// - This handle has no fence (flush-based path; readiness is unknown,
    ///   conservatively returns `false`).
    ///
    /// Use this for non-blocking polling before calling [`to_cpu`](Self::to_cpu).
    #[must_use]
    pub fn is_ready(&self) -> bool {
        match &self.fence {
            Some(fence) => fence.is_completed(),
            None => false,
        }
    }

    /// Whether this handle has an attached fence.
    #[must_use]
    pub fn has_fence(&self) -> bool {
        self.fence.is_some()
    }

    /// Access the attached fence, if any.
    #[must_use]
    pub fn fence(&self) -> Option<&GpuFence> {
        self.fence.as_ref()
    }

    /// Number of audio samples in the buffer.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Audio sample rate in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f32 {
        self.sample_count as f32 / self.sample_rate as f32
    }

    /// Access the underlying GPU buffer (for advanced use like Metal audio
    /// playback or passing to another GPU pipeline without CPU readback).
    #[must_use]
    pub fn gpu_buffer(&self) -> &MetalBuffer {
        &self.buffer
    }

    /// Transfer audio from GPU to CPU as a `DynTensor`.
    ///
    /// Returns a CPU-resident `DynTensor` with shape `[1, 1, sample_count]`
    /// and dtype F32, matching the shape convention of `CompiledKokoro::synthesize()`.
    ///
    /// Calls [`flush()`](crate::gpu_scope::flush) to ensure all pending GPU
    /// work is complete, then reads the buffer into a CPU tensor.
    pub fn to_cpu_tensor(&self) -> Result<DynTensor, MetalError> {
        let pcm = self.to_cpu()?;
        DynTensor::from_vec(pcm, &[1, 1, self.sample_count], &Device::Cpu).map_err(|e| {
            MetalError::DispatchFailed(format!(
                "GpuAudioHandle::to_cpu_tensor tensor construction failed: {e}"
            ))
        })
    }

    /// Create a `GpuAudioHandle` from a GPU-resident `DynTensor`.
    ///
    /// Extracts the underlying `MetalBuffer` from the tensor's GPU storage.
    /// The tensor must be on a Metal GPU device and contain F32 data.
    ///
    /// # Arguments
    ///
    /// * `tensor` - GPU-resident DynTensor with shape `[1, 1, T_audio]`.
    /// * `sample_rate` - Audio sample rate in Hz (e.g. 24000 for Kokoro).
    pub(crate) fn from_dyn_tensor(
        tensor: &DynTensor,
        sample_rate: u32,
    ) -> Result<Self, MetalError> {
        let metal_data: &MetalTensorData = tensor.gpu_data().map_err(|e| {
            MetalError::DispatchFailed(format!(
                "GpuAudioHandle::from_dyn_tensor: not a GPU tensor: {e}"
            ))
        })?;
        let buffer = metal_data.buffer().alias();
        let sample_count = tensor.numel();
        Ok(Self::new(buffer, sample_count, sample_rate))
    }
}

#[cfg(test)]
#[path = "gpu_audio_handle_tests.rs"]
mod tests;
