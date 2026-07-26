// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Speech segment detection from per-chunk VAD probabilities.
//!
//! Implements the state machine from Python Silero `get_speech_timestamps`:
//! speech starts when probability exceeds threshold, ends after sustained
//! silence below a negative threshold. Ported from dvoice-vad segments.rs.

/// Detected speech segment with sample-level boundaries.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpeechSegment {
    /// First sample index of the segment (inclusive).
    pub start_sample: usize,
    /// Last sample index of the segment (exclusive).
    pub end_sample: usize,
    /// Start time in seconds.
    pub start_time: f32,
    /// End time in seconds.
    pub end_time: f32,
}

impl SpeechSegment {
    /// Create a speech segment with sample and time boundaries.
    pub fn new(start_sample: usize, end_sample: usize, start_time: f32, end_time: f32) -> Self {
        Self {
            start_sample,
            end_sample,
            start_time,
            end_time,
        }
    }

    /// Duration of the segment in seconds.
    #[inline]
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }
}

/// Configuration for speech segment detection.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
#[must_use]
pub struct SegmentConfig {
    /// Speech probability threshold to start a segment (default 0.5).
    pub threshold: f32,
    /// Minimum speech duration in milliseconds to keep a segment.
    pub min_speech_duration_ms: u32,
    /// Minimum silence duration in milliseconds to end a segment.
    pub min_silence_duration_ms: u32,
    /// Padding in milliseconds to add around each segment.
    pub speech_pad_ms: u32,
    /// Maximum speech segment duration in seconds (0 = no limit).
    pub max_speech_duration_s: f32,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 300,
            speech_pad_ms: 30,
            max_speech_duration_s: 0.0,
        }
    }
}

/// Build speech segments from per-chunk probabilities.
///
/// `probs` contains one speech probability per `chunk_size`-sample chunk.
/// `audio_len` is the total number of audio samples (for boundary clamping).
/// `sample_rate` is the audio sample rate in Hz.
pub(crate) fn segments_from_probs(
    probs: &[f32],
    audio_len: usize,
    config: &SegmentConfig,
    chunk_size: usize,
    sample_rate: u32,
) -> Vec<SpeechSegment> {
    if probs.is_empty() || chunk_size == 0 || sample_rate == 0 {
        return Vec::new();
    }

    let sr = sample_rate as f32;
    let chunk_dur = chunk_size as f32 / sr;
    // chunk_size > 0 and sample_rate > 0 guaranteed above, so chunk_dur > 0.

    // Finiteness guards: config values are u32, but after multiplication/division
    // the f32 result can overflow to infinity (e.g., u32::MAX * large sr).
    // Saturate to safe defaults rather than producing usize::MAX from `f32::INFINITY as usize`.
    let pad_f = config.speech_pad_ms as f32 * sr / 1000.0;
    let speech_pad_samples = if pad_f.is_finite() && pad_f >= 0.0 {
        pad_f as usize
    } else {
        0
    };
    let speech_f = (config.min_speech_duration_ms as f32 / 1000.0) / chunk_dur;
    let min_speech_chunks = if speech_f.is_finite() && speech_f >= 0.0 {
        speech_f.ceil() as usize
    } else {
        1
    };
    let silence_f = (config.min_silence_duration_ms as f32 / 1000.0) / chunk_dur;
    let min_silence_chunks = if silence_f.is_finite() && silence_f >= 0.0 {
        silence_f.ceil() as usize
    } else {
        1
    };

    // Negative threshold with precision tolerance matching Python Silero behavior.
    let neg_threshold = (config.threshold - 0.15 - 0.02f32).max(0.01);

    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut speech_start = 0usize;
    let mut temp_end = 0usize;
    let mut silence_start = 0usize;

    for (i, &prob) in probs.iter().enumerate() {
        let cur_sample = i * chunk_size;

        if !in_speech {
            if prob >= config.threshold {
                in_speech = true;
                speech_start = i;
                temp_end = 0;
                silence_start = 0;
            }
        } else if prob >= config.threshold {
            if temp_end != 0 {
                temp_end = 0;
            }
            silence_start = 0;
        } else {
            if silence_start == 0 {
                silence_start = i;
            }
            if prob < neg_threshold && temp_end == 0 {
                temp_end = cur_sample;
            }
            let silence_chunks = i - silence_start + 1;
            if silence_chunks >= min_silence_chunks && temp_end != 0 {
                let speech_len_chunks = silence_start - speech_start;
                if speech_len_chunks >= min_speech_chunks {
                    let start_sample = speech_start * chunk_size;
                    let end_sample = temp_end.min(audio_len);
                    segments.push(SpeechSegment {
                        start_sample,
                        end_sample,
                        start_time: start_sample as f32 / sr,
                        end_time: end_sample as f32 / sr,
                    });
                }
                in_speech = false;
                temp_end = 0;
                silence_start = 0;
            }
        }
    }

    // Handle final segment still in speech at end of audio.
    if in_speech {
        let speech_end = probs.len();
        if speech_end - speech_start >= min_speech_chunks {
            let start_sample = speech_start * chunk_size;
            let end_sample = (speech_end * chunk_size).min(audio_len);
            segments.push(SpeechSegment {
                start_sample,
                end_sample,
                start_time: start_sample as f32 / sr,
                end_time: end_sample as f32 / sr,
            });
        }
    }

    apply_padding(&mut segments, speech_pad_samples, audio_len, sr);

    if config.max_speech_duration_s > 0.0 {
        split_long_segments(&mut segments, config.max_speech_duration_s, sr);
    }

    segments
}

/// Pad segment boundaries, sharing gaps between adjacent segments.
fn apply_padding(segments: &mut [SpeechSegment], pad: usize, audio_len: usize, sr: f32) {
    let n = segments.len();
    for i in 0..n {
        if i == 0 {
            segments[i].start_sample = segments[i].start_sample.saturating_sub(pad);
        }
        if i + 1 < n {
            let gap = segments[i + 1]
                .start_sample
                .saturating_sub(segments[i].end_sample);
            if gap < 2 * pad {
                let half = gap / 2;
                segments[i].end_sample = (segments[i].end_sample + half).min(audio_len);
                segments[i + 1].start_sample =
                    segments[i + 1].start_sample.saturating_sub(gap - half);
            } else {
                segments[i].end_sample = (segments[i].end_sample + pad).min(audio_len);
                segments[i + 1].start_sample = segments[i + 1].start_sample.saturating_sub(pad);
            }
        } else {
            segments[i].end_sample = (segments[i].end_sample + pad).min(audio_len);
        }
        segments[i].start_time = segments[i].start_sample as f32 / sr;
        segments[i].end_time = segments[i].end_sample as f32 / sr;
    }
}

/// Split segments exceeding a maximum duration.
fn split_long_segments(segments: &mut Vec<SpeechSegment>, max_dur_s: f32, sr: f32) {
    let product = max_dur_s * sr;
    if !product.is_finite() || product <= 0.0 {
        return;
    }
    let max_samples = product as usize;
    let mut result = Vec::with_capacity(segments.len());
    for seg in segments.drain(..) {
        let len = seg.end_sample - seg.start_sample;
        if len <= max_samples {
            result.push(seg);
        } else {
            let mut start = seg.start_sample;
            while start < seg.end_sample {
                let end = (start + max_samples).min(seg.end_sample);
                result.push(SpeechSegment {
                    start_sample: start,
                    end_sample: end,
                    start_time: start as f32 / sr,
                    end_time: end as f32 / sr,
                });
                start = end;
            }
        }
    }
    *segments = result;
}

#[cfg(test)]
#[path = "silero_vad_segments_tests.rs"]
mod tests;
