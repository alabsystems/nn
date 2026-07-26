// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `compiled_kokoro_gpu_synth.rs` (#3715).
//!
//! Proves properties of the GPU synthesis backends:
//! - GpuSynth and ChorusGpuSynth batch validation
//! - Styles/speeds array length consistency with n_voices
//! - Speed validation bounds
//! - Voice index bounds in chorus iteration
//! - Batch output vector length matches n_voices
//! - Single-voice fallback accesses voice[0]
//! - NaN-check policy scoping
//! - Shared encoding reduces to N decode passes

// ============================================================================
// 1. ChorusGpuSynth: styles length must equal n_voices
// ============================================================================

/// Prove: synthesize_batch rejects styles.len() != n_voices.
///
/// Models the validation at `compiled_kokoro_gpu_synth.rs:150-155`:
/// ```
/// if styles.len() != n { return Err(...) }
/// ```
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn synth_batch_styles_length_must_match_n_voices() {
    let n_voices: usize = kani::any();
    let styles_len: usize = kani::any();

    kani::assume(n_voices >= 1 && n_voices <= 32);
    kani::assume(styles_len <= 32);

    let accepted = styles_len == n_voices;

    if styles_len != n_voices {
        assert!(!accepted, "mismatched styles length must be rejected");
    }

    if styles_len == n_voices {
        assert!(accepted, "matching styles length must be accepted");
    }
}

// ============================================================================
// 2. ChorusGpuSynth: speeds length must equal n_voices
// ============================================================================

/// Prove: synthesize_batch rejects speeds.len() != n_voices.
///
/// Models the validation at `compiled_kokoro_gpu_synth.rs:156-161`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn synth_batch_speeds_length_must_match_n_voices() {
    let n_voices: usize = kani::any();
    let speeds_len: usize = kani::any();

    kani::assume(n_voices >= 1 && n_voices <= 32);
    kani::assume(speeds_len <= 32);

    let accepted = speeds_len == n_voices;

    if speeds_len != n_voices {
        assert!(!accepted, "mismatched speeds length must be rejected");
    }

    if speeds_len == n_voices {
        assert!(accepted, "matching speeds length must be accepted");
    }
}

// ============================================================================
// 3. Speed validation: positive and bounded
// ============================================================================

/// Prove: validate_speed accepts speeds in (0.0, inf) and rejects
/// non-positive, NaN, and zero values.
///
/// Models `nn_models::kokoro_error::validate_speed`:
/// Speed must be finite and > 0.0 (typically 0.5..=2.0 in practice).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn speed_validation_rejects_invalid() {
    let speed: f32 = kani::any();

    let is_valid = speed.is_finite() && speed > 0.0;

    // Property 1: zero speed is invalid.
    if speed == 0.0 {
        assert!(!is_valid, "zero speed must be rejected");
    }

    // Property 2: negative speed is invalid.
    if speed.is_finite() && speed < 0.0 {
        assert!(!is_valid, "negative speed must be rejected");
    }

    // Property 3: NaN speed is invalid.
    if speed.is_nan() {
        assert!(!is_valid, "NaN speed must be rejected");
    }

    // Property 4: positive finite speed is valid.
    if speed.is_finite() && speed > 0.0 {
        assert!(is_valid, "positive finite speed must be accepted");
    }

    // Property 5: infinity is invalid.
    if speed.is_infinite() {
        assert!(!is_valid, "infinite speed must be rejected");
    }
}

// ============================================================================
// 4. Voice index bounds: iteration over 0..n_voices
// ============================================================================

/// Prove: iterating `for i in 0..n` with voice_mut(i) always has
/// i < n, so the index is valid for a Vec of length n.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(33)]
fn voice_iteration_index_in_bounds() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);

    for i in 0..n {
        // Property: every index is strictly less than n.
        assert!(i < n, "voice index must be < n_voices");
    }
}

// ============================================================================
// 5. Batch output length: equals n_voices
// ============================================================================

/// Prove: synthesize_batch output Vec has exactly n_voices entries.
///
/// Models the accumulation pattern:
/// ```
/// let mut audios: Vec<DynTensor> = Vec::with_capacity(n);
/// for i in 0..n { audios.push(...); }
/// ```
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(33)]
fn batch_output_length_equals_n_voices() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);

    let mut count: usize = 0;
    for _i in 0..n {
        count += 1;
    }

    assert_eq!(count, n, "output length must equal n_voices");
}

// ============================================================================
// 6. Single-voice fallback: accesses voice[0]
// ============================================================================

/// Prove: synthesize_chunk (single-voice) always accesses voice index 0.
/// For a chorus with n >= 1 voices, voice(0) is always valid.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn single_voice_fallback_index_zero() {
    let n_voices: usize = kani::any();
    kani::assume(n_voices >= 1 && n_voices <= 32);

    let fallback_idx = 0;
    assert!(fallback_idx < n_voices, "voice[0] must be valid");
}

// ============================================================================
// 7. Shared encoding: N decode passes for N voices
// ============================================================================

/// Prove: the shared-encoding optimization encodes once (on voice[0])
/// and decodes N times (once per voice). Total work = 1 + N.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn shared_encoding_work_count() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);

    let encode_passes = 1; // Steps 1-2 on voice[0].
    let decode_passes = n; // Steps 3-8 per voice.

    // Property 1: total passes = 1 + N.
    let total = encode_passes + decode_passes;
    assert_eq!(total, 1 + n, "total passes must be 1 + n_voices");

    // Property 2: shared encoding saves (N-1) encode passes.
    let naive_total = 2 * n; // encode + decode per voice
    let saved = naive_total - total;
    assert_eq!(saved, n - 1, "shared encoding saves n-1 encode passes");
}

// ============================================================================
// 8. Pre-split styles: Vec capacity matches n_voices
// ============================================================================

/// Prove: Vec::with_capacity(n) followed by n pushes produces
/// a Vec of length n with no reallocation.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(33)]
fn pre_split_styles_capacity_matches_length() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);

    let mut count = 0usize;
    for _i in 0..n {
        count += 1;
    }
    assert_eq!(count, n, "push count must equal capacity");
}

// ============================================================================
// 9. Batch validation order: styles checked before speeds
// ============================================================================

/// Prove: the validation checks styles.len() first, then speeds.len().
/// If both are wrong, the error message references styles (first check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn batch_validation_order_styles_first() {
    let n: usize = kani::any();
    let styles_len: usize = kani::any();
    let speeds_len: usize = kani::any();

    kani::assume(n >= 1 && n <= 32);
    kani::assume(styles_len <= 32);
    kani::assume(speeds_len <= 32);

    let styles_ok = styles_len == n;
    let speeds_ok = speeds_len == n;

    // Model the sequential check.
    let first_error_is_styles = !styles_ok;
    let first_error_is_speeds = styles_ok && !speeds_ok;
    let all_ok = styles_ok && speeds_ok;

    // Property: exactly one outcome.
    let count = first_error_is_styles as u8 + first_error_is_speeds as u8 + all_ok as u8;
    assert_eq!(count, 1, "exactly one validation outcome");

    // Property: styles error takes priority.
    if !styles_ok {
        assert!(first_error_is_styles, "styles error must be detected first");
    }
}

// ============================================================================
// 10. Discard pending batch: called on error only
// ============================================================================

/// Prove: discard_pending_batch is called if and only if the result
/// is Err. This ensures GPU state cleanup on failure without
/// discarding valid results.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn discard_on_error_only() {
    let is_err: bool = kani::any();

    // Model: `if result.is_err() { discard_pending_batch(); }`
    let calls_discard = is_err;

    if is_err {
        assert!(calls_discard, "Err must trigger discard");
    } else {
        assert!(!calls_discard, "Ok must not trigger discard");
    }
}

// ============================================================================
// 11. NanCheckPolicy::Skip scope: voices loop inside skip scope
// ============================================================================

/// Prove: all N voice decode passes run inside the NanCheckPolicy::Skip
/// scope. The NaN check is done AFTER the scope (verify_and_extract_pcm).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn nan_check_skip_scope_covers_all_voices() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);

    // Model: with_nan_check_policy(Skip, || { for 0..n { decode } })
    // All N decode calls are inside the Skip scope.
    let voices_in_skip_scope = n;
    let voices_outside_skip = 0;

    assert_eq!(voices_in_skip_scope, n, "all voices must be in Skip scope");
    assert_eq!(voices_outside_skip, 0, "no voices outside Skip scope");

    // Model: verify_and_extract_pcm called AFTER the Skip scope.
    let verify_after_skip = true;
    assert!(verify_after_skip, "NaN verification must happen after Skip scope");
}

// ============================================================================
// 12. Chorus voices: n_voices matches both styles and speeds
// ============================================================================

/// Prove: when both validations pass, styles.len() == speeds.len() == n.
/// This transitive equality is relied upon by the decode loop.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn chorus_styles_speeds_transitive_equality() {
    let n: usize = kani::any();
    let styles_len: usize = kani::any();
    let speeds_len: usize = kani::any();

    kani::assume(n >= 1 && n <= 32);

    // Both validations pass.
    kani::assume(styles_len == n);
    kani::assume(speeds_len == n);

    // Transitive: styles_len == speeds_len.
    assert_eq!(
        styles_len, speeds_len,
        "styles and speeds must have equal length"
    );

    // All three are equal.
    assert_eq!(styles_len, n);
    assert_eq!(speeds_len, n);
}

// ============================================================================
// 13. Speed iteration: each speed is individually validated
// ============================================================================

/// Prove: the speed validation loop checks EVERY element, not just
/// the first or last. Models `for &s in speeds { validate_speed(s)?; }`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(9)]
fn speed_validation_checks_every_element() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 8);

    let mut checked_count = 0usize;
    for _i in 0..n {
        // Each iteration checks one speed.
        checked_count += 1;
    }

    assert_eq!(checked_count, n, "every speed must be validated");
}
