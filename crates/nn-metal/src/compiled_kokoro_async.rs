// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Non-blocking synthesis for [`CompiledKokoro`].
//!
//! [`synthesize_async`](super::CompiledKokoro::synthesize_async) combines
//! [`synthesize_gpu`](super::CompiledKokoro::synthesize_gpu) with
//! [`GpuFence::submit_current`] to submit all GPU work and return a
//! caller-held fence. This enables pipeline-friendly usage where the caller
//! can do other work (prepare next utterance, mix audio, etc.) while the
//! GPU executes the synthesis pipeline.
//!
//! # Example
//!
//! ```rust,no_run
//! # use nn_metal::compiled_kokoro::CompiledKokoro;
//! # use nn_metal::PipelineCache;
//! // let mut kokoro = CompiledKokoro::new(model)?;
//! // let cache = PipelineCache::new();
//! //
//! // Submit synthesis — GPU starts immediately, CPU continues.
//! // let (fence, handle, cert) = kokoro.synthesize_async(&ids, &style, 1.0, &cache)?;
//! //
//! // ... do other work while GPU executes ...
//! //
//! // Wait for GPU completion, then read audio.
//! // if let Some(f) = fence { f.wait()?; }
//! // let pcm: Vec<f32> = handle.to_cpu()?;
//! ```
//!
//! Part of #4251.

use super::CompiledKokoro;
use super::CompiledKokoroError;
use crate::cache::PipelineCache;
use crate::gpu_fence::GpuFence;
use nn_core::dyn_tensor::DynTensor;
use nn_tts_verify::Certificate;

impl CompiledKokoro {
    /// Submit synthesis work to the GPU and return a fence for non-blocking wait.
    ///
    /// Combines [`synthesize_gpu()`](Self::synthesize_gpu) with
    /// [`GpuFence::submit_current()`] for pipeline-friendly async usage.
    /// The returned fence can be polled via [`GpuFence::is_completed()`] or
    /// blocked on via [`GpuFence::wait()`].
    ///
    /// # Arguments
    ///
    /// * `input_ids` - `[B, T]` token indices (U32 or F32).
    /// * `style` - `[B, 2*style_dim]` voice embedding (first half = decoder,
    ///   second half = prosody).
    /// * `speed` - Speaking rate multiplier (1.0 = normal).
    /// * `cache` - Metal pipeline cache.
    ///
    /// # Returns
    ///
    /// `(Option<GpuFence>, GpuAudioHandle, Certificate)` — the fence is `None`
    /// if no GPU work was pending at submit time (e.g., all work already
    /// committed by an internal sync point). The audio handle wraps the
    /// GPU-resident PCM buffer; call [`GpuAudioHandle::to_cpu()`] after
    /// waiting on the fence to read the audio.
    ///
    /// # Errors
    ///
    /// Returns [`CompiledKokoroError`] on synthesis failure (invalid inputs,
    /// GPU dispatch error, verification failure). The fence submit itself
    /// can fail with a [`TensorError`](nn_core::TensorError) wrapped in
    /// [`CompiledKokoroError::Tensor`].
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use nn_metal::compiled_kokoro::CompiledKokoro;
    /// # use nn_metal::PipelineCache;
    /// // let (fence, handle, cert) = kokoro.synthesize_async(&ids, &style, 1.0, &cache)?;
    /// // // ... prepare next utterance while GPU runs ...
    /// // if let Some(f) = fence { f.wait()?; }
    /// // let pcm = handle.to_cpu()?;
    /// ```
    ///
    /// Part of #4251.
    pub fn synthesize_async(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(Option<GpuFence>, crate::GpuAudioHandle, Certificate), CompiledKokoroError> {
        let (handle, certificate) = self.synthesize_gpu(input_ids, style, speed, cache)?;
        let fence = GpuFence::submit_current().map_err(|e| {
            CompiledKokoroError::Tensor(Box::new(e))
        })?;
        Ok((fence, handle, certificate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `synthesize_async` signature compiles and returns the expected
    /// tuple type. This is a compile-time type check — actual GPU execution
    /// requires weights and Metal hardware (tested in integration tests).
    #[test]
    fn test_synthesize_async_signature_compiles() {
        // Type assertion: synthesize_async returns the right tuple.
        fn _assert_return_type(
            kokoro: &mut CompiledKokoro,
            ids: &DynTensor,
            style: &DynTensor,
            cache: &PipelineCache,
        ) -> Result<(Option<GpuFence>, crate::GpuAudioHandle, Certificate), CompiledKokoroError>
        {
            kokoro.synthesize_async(ids, style, 1.0, cache)
        }
    }

    /// Verify that GpuFence methods are available on the returned fence type.
    #[test]
    fn test_gpu_fence_api_available() {
        // Compile-time check: GpuFence has is_completed() and wait().
        fn _assert_fence_api(fence: GpuFence) {
            let _completed: bool = fence.is_completed();
            let _result: Result<(), nn_core::TensorError> = fence.wait();
        }
    }
}
