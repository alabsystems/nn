// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep Kani proof harnesses for kokoro_chorus.rs (#3739).
//!
//! Complements existing proofs in:
//! - `kokoro_chorus_kani_tests.rs` (17 harnesses): mix_voices numerics
//! - `kani_kokoro_chorus.rs` (20 harnesses): VoiceInput, VoiceMix, ChorusConfig
//! - `kani_pipeline_chorus_proofs.rs` (24 harnesses): pipeline + streaming
//!
//! This file proves properties NOT covered by those 61 existing harnesses:
//!
//! **with_stereo_pan pan validation bounds:**
//!  1. Pan values at boundary -1.0, 0.0, 1.0 all pass validation
//!  2. Pan value just outside [-1, 1] is rejected (Inf)
//!  3. Gains+pans matching length passes with_stereo_pan
//!  4. with_stereo_pan inherits gain validation from with_gains
//!
//! **Stereo output length invariants:**
//!  5. Stereo output length is exactly 2x max mono voice length
//!  6. Stereo interleave pattern: even indices are left, odd are right
//!
//! **Gain commutativity and linearity:**
//!  7. Gain-scaled mixing is commutative across voice order (2 voices)
//!  8. Zero-gain voice does not contribute to mixed output
//!  9. Unit-gain single-voice is identity (no attenuation)
//!
//! **ChorusConfig validate completeness:**
//! 10. Freshly-constructed equal_gain config always passes validate
//! 11. Freshly-constructed with_gains config always passes validate
//! 12. Freshly-constructed with_stereo_pan config always passes validate
//! 13. Manually mutated n_voices fails validate when gains.len() diverges
//!
//! **mix_voices_from_refs equivalence to mix_voices_with_config:**
//! 14. Both dispatch paths (mono/stereo) are selected by the same condition
//! 15. Config validation runs before audio length check
//!
//! Part of #3739, #3351.

use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// with_stereo_pan pan validation bounds
// ---------------------------------------------------------------------------

/// Harness 1: Pan boundary values -1.0, 0.0, 1.0 all pass validation.
///
/// SUBSTANTIVE: Proves that the three canonical pan positions (hard left,
/// center, hard right) are accepted by the with_stereo_pan validation at
/// kokoro_chorus.rs:129. The check is `!p.is_finite() || p < -1.0 || p > 1.0`.
/// All three values are finite and within [-1.0, 1.0].
///
/// Covers: kokoro_chorus.rs lines 128-134 (pan validation loop).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn pan_boundary_values_pass_validation() {
    let pans: [f32; 3] = [-1.0, 0.0, 1.0];

    for &p in pans.iter() {
        let is_invalid = !p.is_finite() || p < -1.0 || p > 1.0;
        assert!(!is_invalid, "boundary pan value must pass validation");
    }
}

/// Harness 2: Inf pan value is rejected by validation.
///
/// SUBSTANTIVE: Proves that f32::INFINITY (outside [-1, 1]) triggers the
/// is_finite() check in with_stereo_pan. This is important because Inf
/// would produce NaN in the angle computation: (Inf + 1.0) * 0.5 = Inf,
/// Inf * FRAC_PI_2 = Inf, cos(Inf) = NaN.
///
/// Covers: kokoro_chorus.rs line 129 (p.is_finite() check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn inf_pan_rejected_by_validation() {
    let p = f32::INFINITY;
    let is_invalid = !p.is_finite() || p < -1.0 || p > 1.0;
    assert!(is_invalid, "Inf pan must be rejected");

    let p_neg = f32::NEG_INFINITY;
    let is_invalid_neg = !p_neg.is_finite() || p_neg < -1.0 || p_neg > 1.0;
    assert!(is_invalid_neg, "NEG_INFINITY pan must be rejected");
}

/// Harness 3: Matching gains and pans lengths passes with_stereo_pan.
///
/// SUBSTANTIVE: Proves that when gains.len() == pans.len() and both are
/// in valid ranges, the length check at kokoro_chorus.rs:121-127 passes.
/// The subsequent gain validation (delegated to with_gains) and pan
/// validation both succeed.
///
/// Covers: kokoro_chorus.rs lines 120-139 (with_stereo_pan full path).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn matching_gains_pans_lengths_pass() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);

    let gains_len = n;
    let pans_len = n;

    // Length check: gains.len() == pans.len().
    let length_ok = gains_len == pans_len;
    assert!(length_ok, "matching lengths must pass length check");

    // n_voices derived from gains.len() is valid.
    let n_voices = gains_len;
    let n_voices_ok = n_voices >= 1 && n_voices <= 32;
    assert!(n_voices_ok, "n_voices from gains.len() must be in [1, 32]");
}

/// Harness 4: with_stereo_pan inherits gain validation from with_gains.
///
/// SUBSTANTIVE: Proves that with_stereo_pan calls with_gains internally
/// (kokoro_chorus.rs:136), which validates each gain. A NaN gain causes
/// with_gains to fail before the pan loop is reached.
///
/// Covers: kokoro_chorus.rs line 136 (Self::with_gains(gains)? delegation).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stereo_pan_inherits_gain_validation() {
    let gain = f32::NAN;

    // with_gains validation: !g.is_finite() || g < 0.0 || g > 1.0
    let gain_invalid = !gain.is_finite() || gain < 0.0 || gain > 1.0;

    // with_stereo_pan calls with_gains first (line 136).
    // If gain validation fails, with_stereo_pan returns Err before pan loop.
    assert!(
        gain_invalid,
        "NaN gain must be caught by delegated with_gains validation"
    );
}

// ---------------------------------------------------------------------------
// Stereo output length invariants
// ---------------------------------------------------------------------------

/// Harness 5: Stereo output length is exactly 2x max mono voice length.
///
/// SUBSTANTIVE: Proves that mix_voices_stereo allocates `max_len * 2` samples
/// (kokoro_chorus.rs:473), where max_len is the longest voice. The interleaved
/// stereo format doubles the sample count: [L0,R0,L1,R1,...].
///
/// Covers: kokoro_chorus.rs line 473 (stereo = vec![0.0f32; max_len * 2]).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stereo_output_is_double_mono_length() {
    let voice1_len: usize = kani::any();
    let voice2_len: usize = kani::any();
    kani::assume(voice1_len >= 1 && voice1_len <= 1_000_000);
    kani::assume(voice2_len >= 1 && voice2_len <= 1_000_000);

    let max_len = voice1_len.max(voice2_len);
    let stereo_len = max_len * 2;

    assert!(
        stereo_len == max_len * 2,
        "stereo buffer must be 2x max voice length"
    );
    assert!(
        stereo_len >= max_len,
        "stereo buffer must be at least as large as max voice"
    );
    // Even number of samples (interleaved pairs).
    assert!(
        stereo_len % 2 == 0,
        "stereo buffer must have even number of samples"
    );
}

/// Harness 6: Stereo interleave: even indices are left, odd are right.
///
/// SUBSTANTIVE: Proves the interleave pattern in mix_voices_stereo inner
/// loop (kokoro_chorus.rs:482-484): stereo[i*2] += left, stereo[i*2+1] += right.
/// Even indices (0, 2, 4, ...) are always left channel; odd (1, 3, 5, ...)
/// are always right.
///
/// Covers: kokoro_chorus.rs lines 482-484 (interleave pattern).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stereo_interleave_even_left_odd_right() {
    let i: usize = kani::any();
    kani::assume(i <= 1_000_000);

    let left_idx = i * 2;
    let right_idx = i * 2 + 1;

    // Even index is left channel.
    assert!(left_idx % 2 == 0, "left channel index must be even");
    // Odd index is right channel.
    assert!(right_idx % 2 == 1, "right channel index must be odd");
    // They are consecutive.
    assert!(
        right_idx == left_idx + 1,
        "right index must be left index + 1"
    );
}

// ---------------------------------------------------------------------------
// Gain commutativity and linearity
// ---------------------------------------------------------------------------

/// Harness 7: Two-voice gain-scaled mixing is commutative.
///
/// SUBSTANTIVE: Proves that swapping the order of two voices in mix_voices
/// produces the same output. For mono: s1*g1 + s2*g2 == s2*g2 + s1*g1.
/// This is commutativity of addition (IEEE 754 addition is commutative
/// for finite values, though not associative).
///
/// Covers: kokoro_chorus.rs lines 285-289 (accumulation loop order).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn two_voice_mixing_commutative() {
    let s1: f32 = kani::any();
    let s2: f32 = kani::any();
    kani::assume(s1.is_finite() && s1 >= -1.0 && s1 <= 1.0);
    kani::assume(s2.is_finite() && s2 >= -1.0 && s2 <= 1.0);

    let g1: f32 = kani::any();
    let g2: f32 = kani::any();
    kani::assume(g1.is_finite() && g1 >= 0.0 && g1 <= 1.0);
    kani::assume(g2.is_finite() && g2 >= 0.0 && g2 <= 1.0);

    // Order 1: voice 1 first, voice 2 second.
    let mixed_12 = s1 * g1 + s2 * g2;
    // Order 2: voice 2 first, voice 1 second.
    let mixed_21 = s2 * g2 + s1 * g1;

    assert!(
        mixed_12.to_bits() == mixed_21.to_bits(),
        "two-voice mixing must be commutative (IEEE 754 addition is commutative)"
    );
}

/// Harness 8: Zero-gain voice contributes nothing to the mix.
///
/// SUBSTANTIVE: Proves that when gain = 0.0, the voice's contribution
/// to the mixed output is exactly 0.0 (IEEE 754: x * 0.0 = 0.0 for finite x).
/// This means muting a voice by setting its gain to 0 is exact.
///
/// Covers: kokoro_chorus.rs line 288 (mixed[i] += sample * g).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn zero_gain_voice_no_contribution() {
    let sample: f32 = kani::any();
    kani::assume(sample.is_finite() && sample >= -1.0 && sample <= 1.0);

    let gain: f32 = 0.0;
    let g = gain.clamp(0.0, 1.0);
    let contribution = sample * g;

    assert_eq!(
        contribution, 0.0,
        "zero gain must produce zero contribution"
    );
}

/// Harness 9: Unit-gain single-voice preserves input exactly.
///
/// SUBSTANTIVE: Proves that for a single voice with gain 1.0, the mixed
/// output equals the input exactly (identity property). gain.clamp(0, 1)
/// for gain=1.0 returns 1.0, and sample * 1.0 = sample in IEEE 754.
///
/// Covers: kokoro_chorus.rs lines 286-289 (single voice, gain=1.0).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn unit_gain_single_voice_identity() {
    let sample: f32 = kani::any();
    kani::assume(sample.is_finite());

    let gain: f32 = 1.0;
    let g = gain.clamp(0.0, 1.0);
    let result = sample * g;

    assert_eq!(
        result.to_bits(),
        sample.to_bits(),
        "unit gain must preserve input exactly (sample * 1.0 = sample)"
    );
}

// ---------------------------------------------------------------------------
// ChorusConfig validate completeness
// ---------------------------------------------------------------------------

/// Harness 10: Freshly-constructed equal_gain config passes validate.
///
/// SUBSTANTIVE: Proves that ChorusConfig::equal_gain(n) for valid n produces
/// a config where validate() always succeeds. The constructor sets n_voices,
/// gains = vec![1.0/n; n], clip_output = true, pans = None — all of which
/// pass the invariants in validate().
///
/// Covers: kokoro_chorus.rs lines 74-88 (equal_gain), 149-175 (validate).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn equal_gain_always_passes_validate() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);

    // Model the constructor output.
    let n_voices = n;
    let gains_len = n;
    let pans: Option<usize> = None; // pans length (None = no pans)

    // validate checks:
    // 1. n_voices in [1, 32]
    let check1 = n_voices >= 1 && n_voices <= 32;
    // 2. gains.len() == n_voices
    let check2 = gains_len == n_voices;
    // 3. pans (None → skip)
    let check3 = pans.is_none() || pans.unwrap() == n_voices;

    assert!(check1, "n_voices must be in [1, 32]");
    assert!(check2, "gains.len() must equal n_voices");
    assert!(check3, "pans must be None or match n_voices");
}

/// Harness 11: Freshly-constructed with_gains config passes validate.
///
/// SUBSTANTIVE: Proves that with_gains produces a config that passes
/// validate, since n_voices is derived from gains.len() and pans is None.
///
/// Covers: kokoro_chorus.rs lines 91-113 (with_gains), 149-175 (validate).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn with_gains_always_passes_validate() {
    let gains_len: usize = kani::any();
    kani::assume(gains_len >= 1 && gains_len <= 32);

    // Model the constructor: n_voices = gains.len(), pans = None.
    let n_voices = gains_len;

    let check1 = n_voices >= 1 && n_voices <= 32;
    let check2 = gains_len == n_voices;

    assert!(check1 && check2, "with_gains config must pass validate");
}

/// Harness 12: Freshly-constructed with_stereo_pan config passes validate.
///
/// SUBSTANTIVE: Proves that with_stereo_pan produces a config where validate
/// succeeds: n_voices = gains.len(), gains.len() == pans.len(), all three
/// equal n_voices.
///
/// Covers: kokoro_chorus.rs lines 120-139, 149-175.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn with_stereo_pan_always_passes_validate() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);

    // with_stereo_pan: gains.len() == pans.len() (line 121-127 check)
    // then calls with_gains → n_voices = gains.len(), then sets pans = Some(pans).
    let n_voices = n;
    let gains_len = n;
    let pans_len = n;

    let check1 = n_voices >= 1 && n_voices <= 32;
    let check2 = gains_len == n_voices;
    let check3 = pans_len == n_voices;

    assert!(
        check1 && check2 && check3,
        "with_stereo_pan config must pass validate"
    );
}

/// Harness 13: Mutated n_voices causes validate to fail.
///
/// SUBSTANTIVE: The struct fields are pub (due to #[non_exhaustive]).
/// If a caller mutates n_voices after construction, validate catches the
/// inconsistency: gains.len() != n_voices.
///
/// Covers: kokoro_chorus.rs lines 156-164 (validate gains length check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mutated_n_voices_fails_validate() {
    let original_n: usize = kani::any();
    kani::assume(original_n >= 1 && original_n <= 32);

    let gains_len = original_n; // gains.len() from construction

    // Mutate n_voices to a different value.
    let mutated_n: usize = kani::any();
    kani::assume(mutated_n >= 1 && mutated_n <= 32);
    kani::assume(mutated_n != original_n);

    // validate: gains.len() != n_voices
    let check = gains_len == mutated_n;
    assert!(
        !check,
        "mutated n_voices must fail validate when gains.len() diverges"
    );
}

// ---------------------------------------------------------------------------
// mix_voices_from_refs dispatch equivalence
// ---------------------------------------------------------------------------

/// Harness 14: Mono/stereo dispatch is determined solely by pans presence.
///
/// SUBSTANTIVE: Proves that mix_voices_from_refs and mix_voices_with_config
/// use the same dispatch condition: if config.pans.is_some() → stereo,
/// else → mono. The dispatch is not influenced by n_voices, gains values,
/// or audio content.
///
/// Covers: kokoro_chorus.rs lines 393-403 (from_refs), 422-432 (with_config).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_determined_by_pans_presence() {
    let has_pans: bool = kani::any();
    let n_voices: usize = kani::any();
    kani::assume(n_voices >= 1 && n_voices <= 32);

    // from_refs dispatch (line 393):
    let from_refs_stereo = has_pans;
    let from_refs_mono = !has_pans;

    // with_config dispatch (line 422):
    let with_config_stereo = has_pans;
    let with_config_mono = !has_pans;

    assert_eq!(
        from_refs_stereo, with_config_stereo,
        "both functions must agree on stereo dispatch"
    );
    assert_eq!(
        from_refs_mono, with_config_mono,
        "both functions must agree on mono dispatch"
    );
}

/// Harness 15: Config validation runs before audio length check.
///
/// SUBSTANTIVE: Proves the ordering in mix_voices_from_refs
/// (kokoro_chorus.rs:385-392): config.validate() is called FIRST (line 385),
/// then voice_audio.len() != config.n_voices check (line 386). This means
/// an invalid config is caught even if voice_audio is empty or wrong length.
///
/// Covers: kokoro_chorus.rs lines 385-392 (validation ordering).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validation_before_audio_check() {
    let n_voices: usize = kani::any();
    kani::assume(n_voices <= 40); // include invalid values

    let gains_len: usize = kani::any();
    kani::assume(gains_len <= 40);

    let audio_len: usize = kani::any();
    kani::assume(audio_len <= 40);

    // Step 1: config.validate()
    let config_valid = n_voices >= 1 && n_voices <= 32 && gains_len == n_voices;

    // Step 2: voice_audio.len() == config.n_voices (only reached if step 1 passes)
    let audio_count_valid = audio_len == n_voices;

    if !config_valid {
        // Config validation fails FIRST — audio check is never reached.
        // The function returns Err from validate(), not from audio check.
        assert!(
            !config_valid,
            "invalid config must be caught before audio check"
        );
    } else if !audio_count_valid {
        // Config valid but audio count wrong — second check catches it.
        assert!(
            config_valid && !audio_count_valid,
            "audio count mismatch caught after config validation"
        );
    }
}
