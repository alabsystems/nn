// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pull-based streaming long-form transcription.
//!
//! Processes audio in 30-second Whisper chunks on demand. Each call to
//! [`StreamingTranscriber::next_segment()`] decodes the next chunk and returns
//! the transcribed text, allowing the caller to process results incrementally
//! without waiting for the entire audio to be transcribed.
//!
//! # Usage
//!
//! ```no_run
//! use nn_whisper::{WhisperModel, WhisperConfig, WhisperTokenizer};
//! use nn_whisper::streaming::{StreamingTranscriber, StreamingConfig};
//! use nn_core::VarBuilder;
//!
//! let config = WhisperConfig::large_v3_turbo();
//! let vb = VarBuilder::zeros(nn_core::DType::F32, &nn_core::Device::Cpu);
//! let mut model = WhisperModel::load(&vb, config).expect("load model");
//! let tokenizer = WhisperTokenizer::from_vocab_str("{}").expect("tokenizer");
//! let audio = vec![0.0f32; 48_000]; // 3 seconds at 16 kHz
//!
//! let mut transcriber = StreamingTranscriber::new(
//!     &mut model,
//!     &audio,
//!     &tokenizer,
//!     StreamingConfig::default(),
//! );
//!
//! while let Some(segment) = transcriber.next_segment().expect("decode") {
//!     println!("[{:.2}s - {:.2}s] {}", segment.start_time, segment.end_time, segment.text);
//! }
//! ```

use crate::audio::whisper_mel_spectrogram_for_config;
use crate::config::{CHUNK_LENGTH, N_SAMPLES, SAMPLE_RATE};
use crate::decode::{temperature_fallback_decode, DecodeConfig, DEFAULT_TEMPERATURES};
use crate::tokenizer::{WhisperTokenizer, DEFAULT_NO_SPEECH_THRESHOLD, TIMESTAMP_BEGIN};
use crate::WhisperModel;
use nn_core::Result;

/// Configuration for streaming transcription.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StreamingConfig {
    /// Decode configuration for each chunk.
    pub decode_config: DecodeConfig,
    /// Temperature sequence for fallback decoding.
    pub temperatures: Vec<f64>,
    /// No-speech probability threshold. Chunks with `no_speech_prob` above
    /// this value are silently skipped. Default: 0.6.
    pub no_speech_threshold: f64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            decode_config: DecodeConfig::default(),
            temperatures: DEFAULT_TEMPERATURES.to_vec(),
            no_speech_threshold: DEFAULT_NO_SPEECH_THRESHOLD,
        }
    }
}

impl StreamingConfig {
    /// Set the decode configuration.
    #[must_use]
    pub fn with_decode_config(mut self, config: DecodeConfig) -> Self {
        self.decode_config = config;
        self
    }

    /// Set the temperature sequence for fallback decoding.
    #[must_use]
    pub fn with_temperatures(mut self, temps: Vec<f64>) -> Self {
        self.temperatures = temps;
        self
    }

    /// Set the no-speech probability threshold.
    #[must_use]
    pub fn with_no_speech_threshold(mut self, threshold: f64) -> Self {
        self.no_speech_threshold = threshold;
        self
    }
}

/// A single transcription segment produced by [`StreamingTranscriber`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StreamingSegment {
    /// Transcribed text for this segment.
    pub text: String,
    /// Decoded token IDs (excluding initial prompt tokens).
    pub tokens: Vec<usize>,
    /// Start time in seconds from the beginning of the audio.
    pub start_time: f32,
    /// End time in seconds from the beginning of the audio.
    pub end_time: f32,
    /// Average log-probability of the decoded tokens.
    pub avg_logprob: f32,
    /// Probability of no speech at the SOT position.
    pub no_speech_prob: f32,
}

impl StreamingSegment {
    /// Create a new streaming segment.
    #[must_use]
    fn new(
        text: String,
        tokens: Vec<usize>,
        start_time: f32,
        end_time: f32,
        avg_logprob: f32,
        no_speech_prob: f32,
    ) -> Self {
        Self {
            text,
            tokens,
            start_time,
            end_time,
            avg_logprob,
            no_speech_prob,
        }
    }
}

/// Pull-based streaming long-form transcription.
///
/// Processes audio in 30-second Whisper chunks. Each call to [`next_segment()`]
/// decodes the next chunk and returns the transcribed text. Returns `None`
/// when all audio has been processed.
///
/// The transcriber borrows the model and tokenizer for its lifetime, advancing
/// through the audio on each `next_segment()` call. Silent chunks (high
/// no-speech probability) are automatically skipped.
///
/// [`next_segment()`]: StreamingTranscriber::next_segment
pub struct StreamingTranscriber<'a> {
    model: &'a mut WhisperModel,
    tokenizer: &'a WhisperTokenizer,
    samples: Vec<f32>,
    config: StreamingConfig,
    /// Current position in samples.
    offset: usize,
    /// Chunk size in samples (30s * sample_rate = 480_000).
    chunk_samples: usize,
    /// Total audio duration in seconds.
    audio_duration: f32,
    /// Number of mel bins from model config.
    num_mel_bins: usize,
    /// Whether we have finished all chunks.
    done: bool,
}

impl<'a> StreamingTranscriber<'a> {
    /// Create a new streaming transcriber.
    ///
    /// # Arguments
    ///
    /// * `model` - The Whisper model (mutable for KV cache updates during decode).
    /// * `samples` - Raw PCM audio samples (f32, mono, 16 kHz).
    /// * `tokenizer` - Tokenizer for converting token IDs to text.
    /// * `config` - Streaming transcription configuration.
    pub fn new(
        model: &'a mut WhisperModel,
        samples: &[f32],
        tokenizer: &'a WhisperTokenizer,
        config: StreamingConfig,
    ) -> Self {
        let num_mel_bins = model.config().num_mel_bins;
        let audio_duration = samples.len() as f32 / SAMPLE_RATE as f32;
        Self {
            model,
            tokenizer,
            samples: samples.to_vec(),
            config,
            offset: 0,
            chunk_samples: N_SAMPLES,
            audio_duration,
            num_mel_bins,
            done: samples.is_empty(),
        }
    }

    /// Process the next 30-second chunk and return the transcribed segment.
    ///
    /// Returns `Ok(Some(segment))` with the transcription for the next chunk,
    /// `Ok(None)` when all audio has been processed, or `Err` on decode failure.
    ///
    /// Silent chunks (no-speech probability above threshold) are automatically
    /// skipped — the method advances past them and tries the next chunk.
    /// This means a single call may internally process multiple chunks before
    /// returning a non-silent segment or `None`.
    pub fn next_segment(&mut self) -> Result<Option<StreamingSegment>> {
        while !self.done && self.offset < self.samples.len() {
            let chunk_end = (self.offset + self.chunk_samples).min(self.samples.len());
            let chunk = &self.samples[self.offset..chunk_end];

            // whisper_mel_spectrogram_for_config pads/truncates to N_SAMPLES internally.
            let mel = whisper_mel_spectrogram_for_config(chunk, self.num_mel_bins)?;
            let encoder_output = self.model.encode(&mel)?;

            // Decode this chunk with temperature fallback.
            self.model.reset_kv_cache();
            let decode_result = temperature_fallback_decode(
                self.model,
                &encoder_output,
                &self.config.decode_config,
                &self.config.temperatures,
            )?;

            let chunk_offset_sec = self.offset as f32 / SAMPLE_RATE as f32;

            // Skip no-speech segments.
            if decode_result.no_speech_prob > self.config.no_speech_threshold {
                self.offset += self.chunk_samples;
                if self.offset >= self.samples.len() {
                    self.done = true;
                }
                continue;
            }

            // Extract timestamp-based seek advancement.
            let time_advance = extract_timestamp_advance(&decode_result.tokens);

            // Convert tokens to text.
            let text = self.tokenizer.decode(&decode_result.tokens)?;

            // Compute segment boundaries.
            let segment_start = chunk_offset_sec;
            let segment_end = if let Some(advance_sec) = time_advance {
                (chunk_offset_sec + advance_sec as f32).min(self.audio_duration)
            } else {
                (chunk_offset_sec + CHUNK_LENGTH as f32).min(self.audio_duration)
            };

            // Advance seek position.
            self.advance_offset(time_advance);

            // Skip empty text segments but still advance.
            if text.trim().is_empty() {
                continue;
            }

            return Ok(Some(StreamingSegment::new(
                text,
                decode_result.tokens,
                segment_start,
                segment_end,
                decode_result.avg_logprob as f32,
                decode_result.no_speech_prob as f32,
            )));
        }

        self.done = true;
        Ok(None)
    }

    /// Advance the offset based on timestamp tokens or a full chunk.
    fn advance_offset(&mut self, time_advance: Option<f64>) {
        use crate::config::HOP_LENGTH;

        if let Some(advance_sec) = time_advance {
            // Advance by the timestamp amount (in samples), clamped to one chunk.
            // Corrupted timestamp tokens could produce very large advance_sec values;
            // clamping to N_SAMPLES prevents offset from overshooting by more than
            // one chunk.
            let advance_samples =
                ((advance_sec * SAMPLE_RATE as f64) as usize).min(self.chunk_samples);
            self.offset += advance_samples.max(HOP_LENGTH); // Advance at least one hop.
        } else {
            // No timestamps: advance by full chunk.
            self.offset += self.chunk_samples;
        }

        if self.offset >= self.samples.len() {
            self.done = true;
        }
    }

    /// Number of remaining chunks to process (approximate).
    ///
    /// This is an estimate based on the current offset and chunk size.
    /// Timestamp-based seek advancement may cause the actual number of
    /// chunks to differ.
    #[must_use]
    pub fn remaining_chunks(&self) -> usize {
        if self.done {
            return 0;
        }
        let remaining_samples = self.samples.len().saturating_sub(self.offset);
        // Ceiling division: (remaining + chunk - 1) / chunk
        remaining_samples
            .checked_add(self.chunk_samples - 1)
            .map(|n| n / self.chunk_samples)
            .unwrap_or(0)
    }

    /// Total number of chunks in the audio (approximate).
    ///
    /// Based on the total audio length divided into 30-second chunks.
    #[must_use]
    pub fn total_chunks(&self) -> usize {
        if self.samples.is_empty() {
            return 0;
        }
        // Ceiling division.
        self.samples
            .len()
            .checked_add(self.chunk_samples - 1)
            .map(|n| n / self.chunk_samples)
            .unwrap_or(0)
    }

    /// Whether transcription is complete.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Current offset in samples.
    #[must_use]
    pub fn current_offset(&self) -> usize {
        self.offset
    }

    /// Current time position in seconds.
    #[must_use]
    pub fn current_time(&self) -> f32 {
        self.offset as f32 / SAMPLE_RATE as f32
    }
}

/// Extract the timestamp advance from decoded tokens.
///
/// Scans for the last timestamp token in the sequence and computes
/// how many seconds it represents from the start of the 30s chunk.
/// Returns `None` if no timestamp tokens are found.
fn extract_timestamp_advance(tokens: &[usize]) -> Option<f64> {
    let last_ts = tokens.iter().rev().find(|&&t| t >= TIMESTAMP_BEGIN)?;
    let ts_seconds = last_ts.saturating_sub(TIMESTAMP_BEGIN) as f64 * 0.02;
    Some(ts_seconds)
}

#[cfg(test)]
#[path = "streaming_tests.rs"]
mod tests;
