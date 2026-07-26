// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Channel-based pull streaming for [`CompiledKokoro`].
//!
//! [`ChannelStreamingSession`] wraps the callback-driven
//! [`synthesize_streaming_callback`](super::CompiledKokoro::synthesize_streaming_callback)
//! API, feeding audio chunks into an [`mpsc::channel`](std::sync::mpsc::channel)
//! so that a consumer can pull chunks via a blocking or non-blocking interface.
//!
//! # Architecture
//!
//! Because `CompiledKokoro` is `!Send` (it contains `RefCell<Option<MetalBuffer>>`),
//! synthesis cannot happen on a background thread. Instead, the session runs
//! synthesis on the **calling thread** inside [`drive()`](ChannelStreamingSession::drive),
//! sending each chunk over an mpsc channel. A consumer can call
//! [`next_chunk()`](ChannelStreamingSession::next_chunk) (blocking) or
//! [`try_next_chunk()`](ChannelStreamingSession::try_next_chunk) (non-blocking)
//! to receive audio.
//!
//! For dvoice integration, the typical usage pattern is:
//!
//! ```text
//! // Thread A (synthesis):
//! let session = ChannelStreamingSession::new(chunks, &style, 1.0, config);
//! session.drive(&mut kokoro, &cache)?;  // blocks until all chunks are sent
//!
//! // Thread B (playback, via cloned receiver):
//! let rx = session.receiver();
//! while let Some(chunk) = rx.next_chunk() {
//!     match chunk {
//!         StreamChunk::Audio(pcm) => play(&pcm),
//!         StreamChunk::Done => break,
//!         StreamChunk::Error(msg) => handle_error(msg),
//!     }
//! }
//! ```
//!
//! Alternatively, for single-threaded usage, use the [`Iterator`] impl on
//! [`StreamReceiver`] which yields `Vec<f32>` chunks until synthesis is complete.
//!
//! Part of #4105.

use std::sync::mpsc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_models::kokoro_streaming::KokoroStreamConfig;

use super::{CompiledKokoro, CompiledKokoroError};
use crate::cache::PipelineCache;

/// A chunk delivered through the streaming channel.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// A chunk of PCM audio samples (f32, mono, 24kHz).
    Audio(Vec<f32>),
    /// Synthesis completed successfully. No more chunks will follow.
    Done,
    /// Synthesis encountered an error. No more chunks will follow.
    Error(String),
}

/// Channel-based pull streaming session for Kokoro TTS.
///
/// Wraps the callback-driven streaming API, sending audio chunks over an
/// mpsc channel. The consumer pulls chunks via [`StreamReceiver`].
///
/// # Lifecycle
///
/// 1. Create with [`new()`](Self::new).
/// 2. Obtain a [`StreamReceiver`] via [`receiver()`](Self::receiver).
/// 3. Call [`drive()`](Self::drive) to run synthesis (blocks until complete).
/// 4. The receiver yields chunks until `StreamChunk::Done` or `StreamChunk::Error`.
///
/// # Cancellation
///
/// Call [`cancel()`](Self::cancel) from any thread to signal early termination.
/// The synthesis loop will stop after the current chunk completes.
pub struct ChannelStreamingSession {
    /// Pre-chunked token ID tensors, each `[1, T_i]`.
    chunks: Vec<DynTensor>,
    /// Style embedding `[1, 2*style_dim]`.
    style: DynTensor,
    /// Speaking rate multiplier.
    speed: f32,
    /// Crossfade configuration.
    stream_config: KokoroStreamConfig,
    /// Sending end of the audio channel.
    sender: mpsc::Sender<StreamChunk>,
    /// Receiving end (wrapped for the consumer).
    receiver: Option<StreamReceiver>,
    /// Shared cancellation flag.
    cancel_flag: Arc<AtomicBool>,
    /// Shared counter of chunks delivered.
    chunks_delivered: Arc<AtomicUsize>,
    /// Total number of chunks.
    total_chunks: usize,
}

impl ChannelStreamingSession {
    /// Create a new channel-based streaming session.
    ///
    /// # Arguments
    ///
    /// * `chunks` - Pre-tokenized token ID tensors, each `[1, T_i]`.
    /// * `style` - Style embedding `[1, 2*style_dim]` shared across all chunks.
    /// * `speed` - Speaking rate multiplier (1.0 = normal).
    /// * `stream_config` - Crossfade configuration for chunk boundaries.
    #[must_use]
    pub fn new(
        chunks: Vec<DynTensor>,
        style: DynTensor,
        speed: f32,
        stream_config: KokoroStreamConfig,
    ) -> Self {
        let total_chunks = chunks.len();
        let (sender, rx) = mpsc::channel();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let chunks_delivered = Arc::new(AtomicUsize::new(0));

        let receiver = StreamReceiver {
            inner: rx,
            cancel_flag: Arc::clone(&cancel_flag),
            chunks_delivered: Arc::clone(&chunks_delivered),
            total_chunks,
            done: false,
        };

        Self {
            chunks,
            style,
            speed,
            stream_config,
            sender,
            receiver: Some(receiver),
            cancel_flag,
            chunks_delivered,
            total_chunks,
        }
    }

    /// Take the [`StreamReceiver`] for this session.
    ///
    /// Can only be called once. Returns `None` on subsequent calls.
    /// The receiver must be taken before calling [`drive()`](Self::drive).
    pub fn take_receiver(&mut self) -> Option<StreamReceiver> {
        self.receiver.take()
    }

    /// Run synthesis, sending chunks over the internal channel.
    ///
    /// This method blocks until all chunks are synthesized (or cancelled).
    /// Each synthesized chunk is sent as a `StreamChunk::Audio` message.
    /// After all chunks complete, a `StreamChunk::Done` is sent. On error,
    /// a `StreamChunk::Error` is sent.
    ///
    /// # Arguments
    ///
    /// * `kokoro` - Compiled Kokoro pipeline (mutable for segment cache updates).
    /// * `cache` - Metal pipeline cache.
    ///
    /// # Returns
    ///
    /// The number of chunks successfully synthesized.
    pub fn drive(
        &self,
        kokoro: &mut CompiledKokoro,
        cache: &PipelineCache,
    ) -> Result<usize, CompiledKokoroError> {
        if self.chunks.is_empty() {
            let _ = self.sender.send(StreamChunk::Done);
            return Ok(0);
        }

        let cancel = Arc::clone(&self.cancel_flag);
        let delivered = Arc::clone(&self.chunks_delivered);
        let sender = self.sender.clone();

        let result = kokoro.synthesize_streaming_callback(
            &self.chunks,
            &self.style,
            self.speed,
            &self.stream_config,
            cache,
            |chunk| {
                // Check cancellation before sending.
                if cancel.load(Ordering::Relaxed) {
                    return false;
                }

                let pcm = chunk.pcm.clone();
                delivered.fetch_add(1, Ordering::Relaxed);

                // If the receiver has been dropped, stop synthesis.
                if sender.send(StreamChunk::Audio(pcm)).is_err() {
                    return false;
                }
                true
            },
        );

        match result {
            Ok(count) => {
                let _ = self.sender.send(StreamChunk::Done);
                Ok(count)
            }
            Err(e) => {
                let msg = e.to_string();
                let _ = self.sender.send(StreamChunk::Error(msg));
                Err(e)
            }
        }
    }

    /// Signal cancellation.
    ///
    /// Sets a flag that the synthesis loop checks after each chunk.
    /// The current chunk will finish, but no subsequent chunks will be
    /// synthesized. A `StreamChunk::Done` is NOT automatically sent --
    /// the receiver will see the channel close.
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::Relaxed)
    }

    /// Total number of chunks in the session.
    #[must_use]
    pub fn total_chunks(&self) -> usize {
        self.total_chunks
    }

    /// Number of chunks delivered so far.
    #[must_use]
    pub fn delivered_count(&self) -> usize {
        self.chunks_delivered.load(Ordering::Relaxed)
    }

    /// Number of chunks remaining (estimated).
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.total_chunks
            .saturating_sub(self.chunks_delivered.load(Ordering::Relaxed))
    }
}

/// Receiving end of a [`ChannelStreamingSession`].
///
/// Provides blocking and non-blocking methods to pull audio chunks.
/// Implements [`Iterator`] yielding `Vec<f32>` audio chunks (skipping
/// `Done` and `Error` variants).
pub struct StreamReceiver {
    inner: mpsc::Receiver<StreamChunk>,
    cancel_flag: Arc<AtomicBool>,
    chunks_delivered: Arc<AtomicUsize>,
    total_chunks: usize,
    done: bool,
}

impl StreamReceiver {
    /// Pull the next chunk, blocking until one is available.
    ///
    /// Returns `None` when the channel is closed (synthesis complete or
    /// session dropped).
    pub fn next_chunk(&mut self) -> Option<StreamChunk> {
        if self.done {
            return None;
        }
        match self.inner.recv() {
            Ok(chunk) => {
                if matches!(chunk, StreamChunk::Done | StreamChunk::Error(_)) {
                    self.done = true;
                }
                Some(chunk)
            }
            Err(_) => {
                self.done = true;
                None
            }
        }
    }

    /// Try to pull the next chunk without blocking.
    ///
    /// Returns `None` if no chunk is ready yet. Returns
    /// `Some(StreamChunk::Done)` or `Some(StreamChunk::Error(_))` when
    /// synthesis is complete.
    pub fn try_next_chunk(&mut self) -> Option<StreamChunk> {
        if self.done {
            return None;
        }
        match self.inner.try_recv() {
            Ok(chunk) => {
                if matches!(chunk, StreamChunk::Done | StreamChunk::Error(_)) {
                    self.done = true;
                }
                Some(chunk)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.done = true;
                None
            }
        }
    }

    /// Signal cancellation to the producer.
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::Relaxed);
    }

    /// Total number of chunks in the session.
    #[must_use]
    pub fn total_chunks(&self) -> usize {
        self.total_chunks
    }

    /// Number of chunks delivered so far.
    #[must_use]
    pub fn delivered_count(&self) -> usize {
        self.chunks_delivered.load(Ordering::Relaxed)
    }

    /// Estimated number of chunks remaining.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.total_chunks
            .saturating_sub(self.chunks_delivered.load(Ordering::Relaxed))
    }

    /// Whether the session has completed (all chunks received or error).
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.done
    }
}

impl Iterator for StreamReceiver {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.next_chunk()? {
                StreamChunk::Audio(pcm) => return Some(pcm),
                StreamChunk::Done => return None,
                StreamChunk::Error(_) => return None,
            }
        }
    }
}

#[cfg(test)]
#[path = "compiled_kokoro_channel_streaming_tests.rs"]
mod tests;
