// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Long-form audio transcription: 30-second chunked decoding with seek.
//!
//! Implements the sliding-window transcription strategy from AI Provider Whisper's
//! `transcribe()`: segment audio into 30s chunks, decode each chunk, and
//! advance the seek position using timestamp tokens or fixed 30s hops.

use super::{temperature_fallback_decode, DecodeConfig, DecodingResult, DEFAULT_TEMPERATURES};
use crate::audio::whisper_mel_spectrogram_for_config;
use crate::config::{CHUNK_LENGTH, HOP_LENGTH, N_SAMPLES, SAMPLE_RATE};
use crate::tokenizer::{WhisperTokenizer, DEFAULT_NO_SPEECH_THRESHOLD, TIMESTAMP_BEGIN};
use crate::WhisperError;
use crate::WhisperModel;
use nn_core::Result;

/// Configuration for long-form transcription.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LongFormConfig {
    /// Decode configuration for each chunk.
    pub decode_config: DecodeConfig,
    /// Temperature sequence for fallback decoding.
    pub temperatures: Vec<f64>,
    /// No-speech probability threshold. Chunks with `no_speech_prob` above
    /// this value are silently skipped. Default: 0.6.
    pub no_speech_threshold: f64,
}

impl Default for LongFormConfig {
    fn default() -> Self {
        Self {
            decode_config: DecodeConfig::default(),
            temperatures: DEFAULT_TEMPERATURES.to_vec(),
            no_speech_threshold: DEFAULT_NO_SPEECH_THRESHOLD,
        }
    }
}

/// A transcription segment from long-form audio.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LongFormSegment {
    /// Transcribed text for this segment.
    pub text: String,
    /// Start time in seconds from the beginning of the audio.
    pub start: f64,
    /// End time in seconds from the beginning of the audio.
    pub end: f64,
    /// The underlying decoding result with token-level details.
    pub decode_result: DecodingResult,
}

/// Transcribe long-form audio by processing 30-second chunks.
///
/// Segments audio into 30s windows, decodes each with temperature fallback,
/// and advances the seek position using timestamp tokens. Chunks with high
/// no-speech probability are skipped.
///
/// # Arguments
///
/// * `model` - The Whisper model.
/// * `audio` - Raw PCM samples (f32, mono, 16 kHz).
/// * `tokenizer` - Tokenizer for converting token IDs to text.
/// * `config` - Long-form transcription configuration.
///
/// # Returns
///
/// A list of transcription segments with text and time offsets, plus an
/// overall `TranscriptionResult` combining all decoded text.
pub fn transcribe_long(
    model: &mut WhisperModel,
    audio: &[f32],
    tokenizer: &WhisperTokenizer,
    config: &LongFormConfig,
) -> Result<LongFormResult> {
    if audio.is_empty() {
        return Err(WhisperError::EmptyAudio {
            stage: "transcribe_long",
        }
        .into());
    }

    let num_mel_bins = model.config().num_mel_bins;
    let audio_duration = audio.len() as f64 / SAMPLE_RATE as f64;

    let mut segments = Vec::new();
    let mut seek: usize = 0; // current position in samples

    while seek < audio.len() {
        let chunk_end = (seek + N_SAMPLES).min(audio.len());
        let chunk = &audio[seek..chunk_end];

        // whisper_mel_spectrogram_for_config pads/truncates to N_SAMPLES internally.
        let mel = whisper_mel_spectrogram_for_config(chunk, num_mel_bins)?;
        let encoder_output = model.encode(&mel)?;

        // Decode this chunk with temperature fallback.
        model.reset_kv_cache();
        let decode_result = temperature_fallback_decode(
            model,
            &encoder_output,
            &config.decode_config,
            &config.temperatures,
        )?;

        // Skip no-speech segments.
        if decode_result.no_speech_prob > config.no_speech_threshold {
            seek += N_SAMPLES;
            continue;
        }

        let chunk_offset_sec = seek as f64 / SAMPLE_RATE as f64;

        // Extract timestamp-based seek advancement.
        // Look for the last timestamp token to determine how far to advance.
        let time_advance = extract_timestamp_advance(&decode_result.tokens);

        // Convert tokens to text.
        let text = tokenizer.decode(&decode_result.tokens)?;

        // Compute segment boundaries.
        let segment_start = chunk_offset_sec;
        let segment_end = if let Some(advance_sec) = time_advance {
            (chunk_offset_sec + advance_sec).min(audio_duration)
        } else {
            // No timestamp tokens — advance by full chunk length.
            (chunk_offset_sec + CHUNK_LENGTH as f64).min(audio_duration)
        };

        // Store segment (move text and decode_result — no clones needed).
        if !text.trim().is_empty() {
            segments.push(LongFormSegment {
                text,
                start: segment_start,
                end: segment_end,
                decode_result,
            });
        }

        // Advance seek position.
        if let Some(advance_sec) = time_advance {
            // Advance by the timestamp amount (in samples), clamped to one chunk.
            // Corrupted timestamp tokens could produce very large advance_sec values;
            // clamping to N_SAMPLES prevents seek from overshooting audio.len() by
            // more than one chunk.
            let advance_samples = ((advance_sec * SAMPLE_RATE as f64) as usize).min(N_SAMPLES);
            seek += advance_samples.max(HOP_LENGTH); // Advance at least one hop.
        } else {
            // No timestamps: advance by full chunk.
            seek += N_SAMPLES;
        }
    }

    // Combine all segment texts.
    let combined_text: String = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("");

    Ok(LongFormResult {
        text: combined_text,
        segments,
    })
}

/// Result of long-form transcription.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LongFormResult {
    /// Full transcribed text (concatenation of all segments).
    pub text: String,
    /// Individual transcription segments with timestamps.
    pub segments: Vec<LongFormSegment>,
}

/// Extract the timestamp advance from decoded tokens.
///
/// Scans for the last timestamp token in the sequence and computes
/// how many seconds it represents from the start of the 30s chunk.
/// Returns `None` if no timestamp tokens are found.
fn extract_timestamp_advance(tokens: &[usize]) -> Option<f64> {
    // Find the last timestamp token.
    let last_ts = tokens.iter().rev().find(|&&t| t >= TIMESTAMP_BEGIN)?;

    // Use saturating_sub for defense-in-depth: the `>= TIMESTAMP_BEGIN` filter
    // above guarantees this won't underflow, but saturating_sub is safer against
    // future refactoring that might change the filter condition.
    let ts_seconds = last_ts.saturating_sub(TIMESTAMP_BEGIN) as f64 * 0.02;
    Some(ts_seconds)
}

#[cfg(test)]
#[path = "decode_long_tests.rs"]
mod tests;
