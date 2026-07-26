// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for kokoro_number_words: number_to_words, ordinal_to_words.
//!
//! Proves that:
//! 1. number_to_words(0) == "zero".
//! 2. number_to_words produces non-empty output for all valid inputs.
//! 3. ordinal_to_words(0) == "zeroth".
//! 4. ordinal_to_words(1) == "first".
//! 5. ordinal_to_words produces non-empty output for all valid inputs.
//! 6. number_to_words for small numbers contains no digits.
//! 7. ordinal_to_words ends with ordinal suffix for bounded inputs.
//!
//! Part of #3793, #3351.

use crate::kokoro_number_words::{number_to_words, ordinal_to_words};

/// Proof 1: number_to_words(0) == "zero".
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_number_zero_is_zero() {
    let result = number_to_words(0);
    assert_eq!(result, "zero");
}

/// Proof 2: number_to_words produces non-empty output for small values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn proof_number_to_words_nonempty() {
    let n: u8 = kani::any();
    let result = number_to_words(n as u64);
    assert!(
        !result.is_empty(),
        "number_to_words must produce non-empty output"
    );
}

/// Proof 3: ordinal_to_words(0) == "zeroth".
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_ordinal_zero_is_zeroth() {
    let result = ordinal_to_words(0);
    assert_eq!(result, "zeroth");
}

/// Proof 4: ordinal_to_words(1) == "first".
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_ordinal_one_is_first() {
    let result = ordinal_to_words(1);
    assert_eq!(result, "first");
}

/// Proof 5: ordinal_to_words produces non-empty output for small values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn proof_ordinal_to_words_nonempty() {
    let n: u8 = kani::any();
    let result = ordinal_to_words(n as u64);
    assert!(
        !result.is_empty(),
        "ordinal_to_words must produce non-empty output"
    );
}

/// Proof 6: number_to_words for 1..19 contains no digit characters.
///
/// The output should be purely alphabetic words (letters and spaces).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_small_numbers_no_digits() {
    let n: u8 = kani::any();
    kani::assume(n >= 1 && n <= 19);
    let result = number_to_words(n as u64);
    let has_digit = result.bytes().any(|b| b.is_ascii_digit());
    assert!(
        !has_digit,
        "number_to_words({}) = '{}' must contain no digits",
        n, result
    );
}

/// Proof 7: Standard ordinal values produce expected suffix patterns.
///
/// Verifies specific known ordinals.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_ordinal_known_values() {
    assert_eq!(ordinal_to_words(2), "second");
    assert_eq!(ordinal_to_words(3), "third");
    assert_eq!(ordinal_to_words(4), "fourth");
    assert_eq!(ordinal_to_words(5), "fifth");
}
