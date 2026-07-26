// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Preset timeline builders for common mix automation patterns.

use super::{ms_to_samples, AutomationTimeline, SceneSnapshot};
use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

/// Build a timeline that swells from 1 voice to `n_voices` over `duration_ms`.
///
/// The first voice starts at full gain; additional voices fade in linearly
/// over the duration.
pub fn build_to_chorus(
    n_voices: usize,
    duration_ms: f32,
) -> Result<AutomationTimeline, KokoroError> {
    if n_voices == 0 {
        return Err(KokoroError::InvalidConfig {
            field: "n_voices",
            reason: "must be >= 1".into(),
        });
    }
    if !duration_ms.is_finite() || duration_ms <= 0.0 {
        return Err(KokoroError::InvalidConfig {
            field: "duration_ms",
            reason: format!("must be finite and > 0, got {duration_ms}"),
        });
    }
    let mut tl = AutomationTimeline::new();

    // Start: only voice 0 active.
    let mut start = SceneSnapshot::new(n_voices);
    for g in start.per_voice_gains.iter_mut().skip(1) {
        *g = 0.0;
    }
    tl.add_keyframe(0, start, 0.0);

    // End: all voices at equal gain.
    let end = SceneSnapshot::new(n_voices);
    let end_sample = ms_to_samples(duration_ms, KOKORO_SAMPLE_RATE);
    tl.add_keyframe(end_sample, end, duration_ms);

    Ok(tl)
}

/// Build a timeline that fades from `n_voices` down to 2 over `duration_ms`.
///
/// Voices beyond the first two fade to zero gain.
pub fn fade_to_intimate(
    n_voices: usize,
    duration_ms: f32,
) -> Result<AutomationTimeline, KokoroError> {
    if n_voices < 2 {
        return Err(KokoroError::InvalidConfig {
            field: "n_voices",
            reason: "must be >= 2 for fade_to_intimate".into(),
        });
    }
    if !duration_ms.is_finite() || duration_ms <= 0.0 {
        return Err(KokoroError::InvalidConfig {
            field: "duration_ms",
            reason: format!("must be finite and > 0, got {duration_ms}"),
        });
    }
    let mut tl = AutomationTimeline::new();

    // Start: all voices active.
    let start = SceneSnapshot::new(n_voices);
    tl.add_keyframe(0, start, 0.0);

    // End: only first two voices, narrower stereo, no reverb.
    let mut end = SceneSnapshot::new(n_voices);
    for g in end.per_voice_gains.iter_mut().skip(2) {
        *g = 0.0;
    }
    end.stereo_width = 0.4;
    end.reverb_mix = 0.0;
    end.dynamics_threshold = -12.0;
    let end_sample = ms_to_samples(duration_ms, KOKORO_SAMPLE_RATE);
    tl.add_keyframe(end_sample, end, duration_ms);

    Ok(tl)
}

/// Build a verse-chorus-verse dynamic swell timeline.
///
/// Three keyframes: quiet verse at 0, loud chorus at `peak_ms`,
/// quiet verse again at `total_ms`.
pub fn dynamic_swell(
    n_voices: usize,
    peak_ms: f32,
    total_ms: f32,
    transition_ms: f32,
) -> Result<AutomationTimeline, KokoroError> {
    if n_voices == 0 {
        return Err(KokoroError::InvalidConfig {
            field: "n_voices",
            reason: "must be >= 1".into(),
        });
    }
    if !peak_ms.is_finite() || peak_ms <= 0.0 {
        return Err(KokoroError::InvalidConfig {
            field: "peak_ms",
            reason: format!("must be finite and > 0, got {peak_ms}"),
        });
    }
    if !total_ms.is_finite() || total_ms <= peak_ms {
        return Err(KokoroError::InvalidConfig {
            field: "total_ms",
            reason: format!("must be finite and > peak_ms ({peak_ms}), got {total_ms}"),
        });
    }

    let mut tl = AutomationTimeline::new();

    // Verse: soft, narrow, dry.
    let mut verse = SceneSnapshot::new(n_voices);
    verse.master_gain = 0.5;
    verse.stereo_width = 0.4;
    verse.reverb_mix = 0.05;
    tl.add_keyframe(0, verse.clone(), 0.0);

    // Chorus: full, wide, wet.
    let mut chorus = SceneSnapshot::new(n_voices);
    chorus.master_gain = 1.0;
    chorus.stereo_width = 1.5;
    chorus.reverb_mix = 0.3;
    chorus.dynamics_threshold = -24.0;
    let peak_sample = ms_to_samples(peak_ms, KOKORO_SAMPLE_RATE);
    tl.add_keyframe(peak_sample, chorus, transition_ms);

    // Back to verse.
    let end_sample = ms_to_samples(total_ms, KOKORO_SAMPLE_RATE);
    tl.add_keyframe(end_sample, verse, transition_ms);

    Ok(tl)
}
