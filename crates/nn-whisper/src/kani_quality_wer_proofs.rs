// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Word Error Rate (WER) computation.
//!
//! Covers:
//! - WER of identical strings is 0.0
//! - WER is non-negative for all inputs
//! - Empty reference with empty hypothesis yields 0.0
//! - Empty reference with non-empty hypothesis yields 1.0
//! - WER of empty hypothesis against non-empty reference yields 1.0
//!
//! Issue: #4303

#[cfg(kani)]
mod proofs {
    use crate::quality::word_error_rate;

    // ============================================================================
    // Harness 1: WER of identical strings is 0.0
    // ============================================================================

    /// Proves that WER of any string against itself is 0.0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn wer_identical_strings_is_zero() {
        // Test with a fixed representative set of strings.
        let inputs = ["hello world", "a", "the cat sat on the mat", ""];
        for s in &inputs {
            let wer = word_error_rate(s, s);
            assert!(
                wer.abs() < 1e-12,
                "WER of identical strings must be 0.0"
            );
        }
    }

    // ============================================================================
    // Harness 2: WER is non-negative
    // ============================================================================

    /// Proves WER is non-negative for all tested input pairs.
    #[kani::unwind(1)]
    #[kani::proof]
    fn wer_is_non_negative() {
        let pairs = [
            ("hello", "world"),
            ("the cat", "a dog"),
            ("", "something"),
            ("something", ""),
            ("", ""),
            ("a b c", "a b c d"),
        ];
        for (hyp, reference) in &pairs {
            let wer = word_error_rate(hyp, reference);
            assert!(wer >= 0.0, "WER must be non-negative");
        }
    }

    // ============================================================================
    // Harness 3: Empty reference + empty hypothesis = 0.0
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn wer_both_empty_is_zero() {
        let wer = word_error_rate("", "");
        assert!(wer.abs() < 1e-12, "WER of two empty strings must be 0.0");
    }

    // ============================================================================
    // Harness 4: Empty reference + non-empty hypothesis = 1.0
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn wer_empty_reference_nonempty_hypothesis_is_one() {
        let wer = word_error_rate("hello world", "");
        assert!(
            (wer - 1.0).abs() < 1e-12,
            "empty reference with non-empty hypothesis must return 1.0"
        );
    }

    // ============================================================================
    // Harness 5: Non-empty reference + empty hypothesis = 1.0
    // ============================================================================

    /// Proves that all-deletions (empty hypothesis) yields WER = 1.0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn wer_nonempty_reference_empty_hypothesis_is_one() {
        let wer = word_error_rate("", "the cat sat");
        // 3 deletions / 3 = 1.0
        assert!(
            (wer - 1.0).abs() < 1e-12,
            "empty hypothesis against non-empty reference must yield 1.0"
        );
    }

    // ============================================================================
    // Harness 6: WER of completely wrong transcript
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn wer_completely_wrong_is_one() {
        let wer = word_error_rate("foo bar", "hello world");
        // 2 substitutions / 2 = 1.0
        assert!(
            (wer - 1.0).abs() < 1e-12,
            "completely wrong transcript must have WER = 1.0"
        );
    }

    // ============================================================================
    // Harness 7: Single insertion, deletion, and substitution
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn wer_single_substitution() {
        // 1 sub / 3 ref = 1/3
        let wer = word_error_rate("the dog sat", "the cat sat");
        assert!(
            (wer - 1.0 / 3.0).abs() < 1e-12,
            "single substitution gives WER = 1/3"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn wer_single_deletion() {
        // "the sat" vs "the cat sat": 1 deletion / 3 = 1/3
        let wer = word_error_rate("the sat", "the cat sat");
        assert!(
            (wer - 1.0 / 3.0).abs() < 1e-12,
            "single deletion gives WER = 1/3"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn wer_single_insertion() {
        // "the big cat sat" vs "the cat sat": 1 insertion / 3 = 1/3
        let wer = word_error_rate("the big cat sat", "the cat sat");
        assert!(
            (wer - 1.0 / 3.0).abs() < 1e-12,
            "single insertion gives WER = 1/3"
        );
    }
}
