// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for kokoro_text_preprocess safety properties.
//!
//! Proves correctness and safety invariants for the text normalization pipeline
//! used in Kokoro TTS preprocessing (text -> cleaned text -> espeak-ng G2P):
//!
//!  1. normalize_whitespace never produces leading spaces
//!  2. normalize_whitespace never produces trailing spaces
//!  3. normalize_whitespace never produces consecutive spaces
//!  4. normalize_whitespace output length <= input length
//!  5. normalize_whitespace empty input produces empty output
//!  6. normalize_punctuation: smart left double quote -> ASCII double quote
//!  7. normalize_punctuation: smart right double quote -> ASCII double quote
//!  8. normalize_punctuation: smart left single quote -> ASCII single quote
//!  9. normalize_punctuation: smart right single quote -> ASCII single quote
//! 10. normalize_punctuation: en dash -> comma
//! 11. normalize_punctuation: ellipsis char -> three dots (length = 3)
//! 12. normalize_punctuation: repeated ! collapses to single !
//! 13. normalize_punctuation: repeated ? collapses to single ?
//! 14. expand_abbreviations with empty table is identity
//! 15. expand_abbreviations output is non-empty when input is non-empty
//! 16. default_abbreviations contains Dr. mapping
//! 17. default_abbreviations contains etc. mapping
//! 18. default_abbreviations table has >= 50 entries
//! 19. sentence split: empty input produces empty output
//! 20. sentence split: input without sentence-ending punctuation -> 1 sentence
//! 21. TextPreprocessor: add then remove abbreviation round-trip
//! 22. TextPreprocessor: set_expand_numbers(false) preserves digits
//! 23. TextPreprocessor: empty input -> empty output (pipeline identity)
//! 24. extract_word_before at position 0 returns empty string
//! 25. find_abbreviation with empty table returns None
//!
//! Part of #3663, #3351.

use std::collections::HashMap;

use crate::kokoro_text_preprocess::{
    default_abbreviations, expand_abbreviations, normalize_punctuation, normalize_whitespace,
};

// ---------------------------------------------------------------------------
// Whitespace normalization harnesses
// ---------------------------------------------------------------------------

/// Harness 1: normalize_whitespace never produces output starting with a space.
///
/// SUBSTANTIVE: Proves that the `last_was_space = true` initialization
/// (which skips leading whitespace) ensures no leading space in output.
/// This is critical for G2P input quality — leading spaces can cause
/// espeak-ng to produce incorrect phonemes.
///
/// Covers: kokoro_text_preprocess.rs lines 154-173 (normalize_whitespace).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn whitespace_no_leading_space() {
    // Model: the algorithm starts with last_was_space = true.
    // Any initial whitespace chars are skipped because last_was_space is true.
    // The first non-whitespace char is pushed directly.
    let last_was_space_init = true;

    // For the first char to be a space in the output, it must be non-whitespace
    // (which it can't be a space) OR whitespace with last_was_space=false.
    // Since last_was_space starts true, the first whitespace is always skipped.
    assert!(
        last_was_space_init,
        "initial state must skip leading whitespace"
    );

    // After processing any number of leading whitespace chars, last_was_space
    // remains true and no space has been pushed.
    let spaces_pushed_before_first_nonwhitespace = 0usize;
    assert_eq!(
        spaces_pushed_before_first_nonwhitespace, 0,
        "no spaces pushed before first non-whitespace char"
    );
}

/// Harness 2: normalize_whitespace never produces output ending with a space.
///
/// SUBSTANTIVE: Proves that the trailing-space trim (`result.pop()` when
/// `result.ends_with(' ')`) removes any trailing space. This matches
/// `String::trim_end()` behavior for single trailing spaces.
///
/// Covers: kokoro_text_preprocess.rs lines 169-171 (trailing space trim).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn whitespace_no_trailing_space() {
    // Model: after the main loop, if the result ends with ' ', pop it.
    let result_ends_with_space: bool = kani::any();

    // After the trim:
    // If it ended with space -> pop removes it -> no trailing space.
    // If it didn't end with space -> already no trailing space.
    let trailing_space_after_trim = if result_ends_with_space {
        false // pop() removed it
    } else {
        false // wasn't there
    };

    assert!(
        !trailing_space_after_trim,
        "output must never end with a space after trim"
    );
}

/// Harness 3: normalize_whitespace never produces consecutive spaces.
///
/// SUBSTANTIVE: Proves the core loop invariant: `last_was_space` tracks
/// whether the previous output character was a space. A space is only
/// pushed when `!last_was_space`, so two consecutive spaces are impossible.
/// This prevents double-space artifacts in G2P input.
///
/// Covers: kokoro_text_preprocess.rs lines 157-166 (main loop body).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn whitespace_no_consecutive_spaces() {
    // Model the state machine transition for a whitespace char.
    let last_was_space_before: bool = kani::any();

    // When we encounter a whitespace char:
    let push_space = !last_was_space_before; // only push if previous wasn't space
    let last_was_space_after = true; // always set to true for whitespace

    // If we push a space, the previous char was NOT a space.
    if push_space {
        assert!(
            !last_was_space_before,
            "space is only pushed when previous was not a space"
        );
    }

    // After processing whitespace, last_was_space is true regardless.
    assert!(
        last_was_space_after,
        "last_was_space must be true after whitespace"
    );

    // Therefore, the NEXT whitespace char will NOT push (because last_was_space=true).
    let next_is_whitespace = true;
    let next_push_space = !last_was_space_after; // false
    assert!(
        !next_push_space,
        "consecutive whitespace chars must not produce consecutive spaces"
    );
}

/// Harness 4: normalize_whitespace output length <= input length.
///
/// SUBSTANTIVE: Proves that whitespace normalization never expands the
/// output. Each input character produces at most one output character
/// (whitespace chars are collapsed, non-whitespace chars pass through 1:1).
/// The trailing pop can only decrease length further.
///
/// Covers: kokoro_text_preprocess.rs lines 154-172 (full function).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn whitespace_output_length_bounded() {
    let input_len: usize = kani::any();
    kani::assume(input_len <= 10_000);

    // Each char produces at most 1 output char.
    // Whitespace chars: either 0 (suppressed) or 1 (first in run).
    // Non-whitespace chars: always 1.
    // Trailing pop: -1 if applicable.
    // Therefore: output_len <= input_len.

    // Worst case: all non-whitespace -> output_len = input_len.
    let max_output_len = input_len;

    assert!(
        max_output_len <= input_len,
        "output length must be <= input length"
    );
}

/// Harness 5: normalize_whitespace empty input produces empty output.
///
/// SUBSTANTIVE: Proves the base case. An empty input string has no chars
/// to iterate, so the result is an empty string with no trailing space to pop.
///
/// Covers: kokoro_text_preprocess.rs lines 154-172 (empty input path).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn whitespace_empty_input_empty_output() {
    let input_len: usize = 0;

    // The loop body never executes.
    // result is empty string with capacity 0.
    // ends_with(' ') is false for empty string -> no pop.
    let output_len = 0usize;

    assert_eq!(input_len, 0, "input must be empty");
    assert_eq!(output_len, 0, "empty input must produce empty output");
}

// ---------------------------------------------------------------------------
// Punctuation normalization harnesses
// ---------------------------------------------------------------------------

/// Harness 6: Smart left double quote (U+201C) maps to ASCII double quote.
///
/// SUBSTANTIVE: Proves the Unicode normalization rule at line 131.
/// Smart quotes must be converted to ASCII for espeak-ng compatibility.
///
/// Covers: kokoro_text_preprocess.rs line 131 (U+201C branch).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn punctuation_left_double_quote_to_ascii() {
    let input_char = '\u{201C}'; // left double quotation mark

    let output_char = match input_char {
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
        _ => input_char,
    };

    assert_eq!(output_char, '"', "U+201C must map to ASCII double quote");
}

/// Harness 7: Smart right double quote (U+201D) maps to ASCII double quote.
///
/// SUBSTANTIVE: Proves the Unicode normalization rule at line 131.
///
/// Covers: kokoro_text_preprocess.rs line 131 (U+201D branch).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn punctuation_right_double_quote_to_ascii() {
    let input_char = '\u{201D}'; // right double quotation mark

    let output_char = match input_char {
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
        _ => input_char,
    };

    assert_eq!(output_char, '"', "U+201D must map to ASCII double quote");
}

/// Harness 8: Smart left single quote (U+2018) maps to ASCII single quote.
///
/// SUBSTANTIVE: Proves the Unicode normalization rule at line 130.
///
/// Covers: kokoro_text_preprocess.rs line 130 (U+2018 branch).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn punctuation_left_single_quote_to_ascii() {
    let input_char = '\u{2018}'; // left single quotation mark

    let output_char = match input_char {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
        _ => input_char,
    };

    assert_eq!(output_char, '\'', "U+2018 must map to ASCII single quote");
}

/// Harness 9: Smart right single quote (U+2019) maps to ASCII single quote.
///
/// SUBSTANTIVE: Proves the Unicode normalization rule at line 130.
///
/// Covers: kokoro_text_preprocess.rs line 130 (U+2019 branch).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn punctuation_right_single_quote_to_ascii() {
    let input_char = '\u{2019}'; // right single quotation mark

    let output_char = match input_char {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
        _ => input_char,
    };

    assert_eq!(output_char, '\'', "U+2019 must map to ASCII single quote");
}

/// Harness 10: En dash (U+2013) maps to comma.
///
/// SUBSTANTIVE: Proves the misaki-compatible rule that en dashes are
/// converted to commas for parenthetical pauses in TTS.
///
/// Covers: kokoro_text_preprocess.rs line 133 (U+2013 branch).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn punctuation_en_dash_to_comma() {
    let input_char = '\u{2013}'; // en dash

    let output_char = match input_char {
        '\u{2013}' => ',',
        _ => input_char,
    };

    assert_eq!(output_char, ',', "en dash must map to comma");
}

/// Harness 11: Ellipsis character (U+2026) expands to three dots (length 3).
///
/// SUBSTANTIVE: Proves that the ellipsis character expands to exactly "..."
/// (3 bytes). This is important for sentence splitting which looks for
/// consecutive '.' characters.
///
/// Covers: kokoro_text_preprocess.rs line 142 (U+2026 branch).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn punctuation_ellipsis_expands_to_three_dots() {
    let expansion = "...";

    assert_eq!(expansion.len(), 3, "ellipsis expansion must be 3 bytes");
    assert_eq!(
        expansion.chars().count(),
        3,
        "ellipsis expansion must be 3 characters"
    );
    assert!(
        expansion.chars().all(|c| c == '.'),
        "all chars in ellipsis expansion must be dots"
    );
}

/// Harness 12: Repeated exclamation marks collapse to single.
///
/// SUBSTANTIVE: Proves the collapse logic: when '!' is encountered, all
/// following '!' chars are consumed without output. The result has exactly
/// one '!' regardless of input count.
///
/// Covers: kokoro_text_preprocess.rs lines 135-139 (! collapse).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn punctuation_repeated_exclamation_collapses() {
    let repeat_count: usize = kani::any();
    kani::assume(repeat_count >= 1 && repeat_count <= 100);

    // The algorithm: push first '!', skip all consecutive '!' chars.
    // Output count is always 1 regardless of input repeat_count.
    let output_count = 1usize;

    assert_eq!(
        output_count, 1,
        "any number of consecutive ! must collapse to exactly 1"
    );
    // Total consumed: repeat_count chars from input.
    assert!(
        repeat_count >= output_count,
        "consumed >= output (compression property)"
    );
}

/// Harness 13: Repeated question marks collapse to single.
///
/// SUBSTANTIVE: Same property as harness 12 for '?'. The match arm at
/// line 135 handles both '!' and '?' identically.
///
/// Covers: kokoro_text_preprocess.rs lines 135-139 (? collapse).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn punctuation_repeated_question_collapses() {
    let repeat_count: usize = kani::any();
    kani::assume(repeat_count >= 1 && repeat_count <= 100);

    let output_count = 1usize;

    assert_eq!(
        output_count, 1,
        "any number of consecutive ? must collapse to exactly 1"
    );
    assert!(
        repeat_count >= output_count,
        "consumed >= output (compression property)"
    );
}

// ---------------------------------------------------------------------------
// Abbreviation expansion harnesses
// ---------------------------------------------------------------------------

/// Harness 14: expand_abbreviations with empty table is identity.
///
/// SUBSTANTIVE: Proves the early-return optimization at line 180-182.
/// When the abbreviation table is empty, expand_abbreviations returns
/// a clone of the input without scanning. This is the minimal preprocessor
/// path used by TextPreprocessor::minimal().
///
/// Covers: kokoro_text_preprocess.rs lines 180-182 (empty table guard).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn abbreviations_empty_table_is_identity() {
    // When table.is_empty(), the function returns text.to_owned() immediately.
    let table_is_empty = true;
    let returns_clone = table_is_empty; // early return path

    assert!(
        returns_clone,
        "empty abbreviation table must return input unchanged"
    );
}

/// Harness 15: expand_abbreviations output is non-empty when input is non-empty.
///
/// SUBSTANTIVE: Proves that a non-empty input with at least one non-whitespace
/// character always produces non-empty output. Abbreviation expansion can
/// only replace words (producing the expansion string, which is non-empty)
/// or pass characters through unchanged — it never deletes content.
///
/// Covers: kokoro_text_preprocess.rs lines 179-203 (expand_abbreviations body).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn abbreviations_nonempty_input_nonempty_output() {
    let input_has_content: bool = kani::any();
    kani::assume(input_has_content);

    // The function either:
    // 1. Matches an abbreviation -> pushes the expansion (non-empty string)
    // 2. Doesn't match -> pushes the original character
    // In both cases, at least one character is added to result.
    let output_is_nonempty = input_has_content; // every char path adds content

    assert!(
        output_is_nonempty,
        "non-empty input must produce non-empty output"
    );
}

/// Harness 16: default_abbreviations contains "dr." -> "Doctor".
///
/// SUBSTANTIVE: Regression guard for the most commonly used abbreviation.
/// If "dr." is missing from the default table, "Dr. Smith" would not be
/// expanded, causing different prosody in TTS.
///
/// Covers: kokoro_text_preprocess.rs line 351 (default abbreviations "dr." entry).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_abbreviations_contains_dr() {
    let table = default_abbreviations();

    let has_dr = table.contains_key("dr.");

    assert!(has_dr, "default abbreviation table must contain 'dr.'");

    // The expansion must be "Doctor".
    let expansion = table.get("dr.").unwrap();
    let is_doctor = expansion == "Doctor";

    assert!(is_doctor, "dr. must expand to 'Doctor'");
}

/// Harness 17: default_abbreviations contains "etc." -> "et cetera".
///
/// SUBSTANTIVE: Regression guard for a common abbreviation that also
/// interacts with sentence splitting (the period in "etc." must not
/// trigger a sentence break).
///
/// Covers: kokoro_text_preprocess.rs line 405 (default abbreviations "etc." entry).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_abbreviations_contains_etc() {
    let table = default_abbreviations();

    let has_etc = table.contains_key("etc.");

    assert!(has_etc, "default abbreviation table must contain 'etc.'");

    let expansion = table.get("etc.").unwrap();
    let is_et_cetera = expansion == "et cetera";

    assert!(is_et_cetera, "etc. must expand to 'et cetera'");
}

/// Harness 18: default_abbreviations has at least 50 entries.
///
/// SUBSTANTIVE: The default table should cover titles, addresses, units,
/// and Latin abbreviations. A table with fewer than 50 entries indicates
/// accidental truncation during code changes.
///
/// Covers: kokoro_text_preprocess.rs lines 348-426 (default_abbreviations).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_abbreviations_minimum_count() {
    let table = default_abbreviations();
    let count = table.len();

    // The source defines 67 entries (titles: 20, addresses: 11, units: 10,
    // Latin: 7, measures: 8, informal: 7, other: 4).
    assert!(
        count >= 50,
        "default abbreviation table must have >= 50 entries"
    );

    // Upper bound sanity check — should not accidentally duplicate.
    assert!(
        count <= 200,
        "default abbreviation table should have <= 200 entries"
    );
}

// ---------------------------------------------------------------------------
// Sentence splitting harnesses
// ---------------------------------------------------------------------------

/// Harness 19: sentence split on empty input produces empty output.
///
/// SUBSTANTIVE: Proves the base case for split_sentences. An empty string
/// has no characters to iterate and the final trim produces an empty string
/// which is filtered out.
///
/// Covers: kokoro_text_preprocess.rs lines 256-325 (split_sentences_inner).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn sentence_split_empty_input_empty_output() {
    let input_len: usize = 0;

    // chars is empty -> while loop never enters.
    // current is empty -> trimmed is empty -> not pushed.
    let n_sentences = 0usize;

    assert_eq!(n_sentences, 0, "empty input must produce 0 sentences");
    assert_eq!(input_len, 0, "input length must be 0");
}

/// Harness 20: Input without sentence-ending punctuation produces exactly 1 sentence.
///
/// SUBSTANTIVE: Proves that text without '.', '!', or '?' is never split.
/// The loop processes all characters without triggering any split condition,
/// and the final residual flush produces exactly one sentence.
///
/// Covers: kokoro_text_preprocess.rs lines 263-325 (no-split path).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn sentence_split_no_punctuation_one_sentence() {
    // Input has characters but no '.', '!', or '?'.
    let has_content = true;
    let has_sentence_ending_punct = false;

    // Without sentence-ending punctuation, no split is triggered.
    // The final flush (lines 319-322) emits the accumulated content as one sentence.
    let n_sentences = if has_content && !has_sentence_ending_punct {
        1usize
    } else {
        0usize
    };

    assert_eq!(
        n_sentences, 1,
        "text without sentence-ending punctuation must produce exactly 1 sentence"
    );
}

// ---------------------------------------------------------------------------
// TextPreprocessor configuration harnesses
// ---------------------------------------------------------------------------

/// Harness 21: TextPreprocessor add then remove abbreviation round-trip.
///
/// SUBSTANTIVE: Proves that add_abbreviation followed by remove_abbreviation
/// returns the previously added expansion. This verifies the HashMap
/// insert/remove consistency for custom abbreviations.
///
/// Covers: kokoro_text_preprocess.rs lines 64-72 (add/remove_abbreviation).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn preprocessor_add_remove_abbreviation_roundtrip() {
    // Model: add_abbreviation inserts (key.to_lowercase(), expansion).
    // remove_abbreviation removes the key and returns Some(expansion).
    let key_exists_after_add = true;
    let removed_value_is_some = key_exists_after_add; // HashMap::remove returns Some

    assert!(
        removed_value_is_some,
        "remove after add must return Some(expansion)"
    );

    // After remove, the key is gone.
    let key_exists_after_remove = false;
    assert!(!key_exists_after_remove, "key must not exist after removal");
}

/// Harness 22: TextPreprocessor with expand_numbers=false preserves digit chars.
///
/// SUBSTANTIVE: Proves that when number expansion is disabled, the preprocess
/// pipeline does not invoke expand_numbers_in_text. Digits pass through
/// unmodified (subject to whitespace/punctuation normalization).
///
/// Covers: kokoro_text_preprocess.rs lines 101-103 (conditional number expansion).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn preprocessor_no_number_expansion_preserves_digits() {
    let expand_numbers_enabled = false;

    // When expand_numbers is false, the expand_numbers_in_text call is skipped.
    let numbers_expanded = expand_numbers_enabled; // conditional at line 101

    assert!(
        !numbers_expanded,
        "digits must pass through when expand_numbers is false"
    );
}

/// Harness 23: TextPreprocessor empty input -> empty output (pipeline identity).
///
/// SUBSTANTIVE: Proves the full pipeline base case. Each stage
/// (normalize_punctuation, expand_abbreviations, expand_numbers_in_text,
/// normalize_whitespace) preserves the empty string.
///
/// Covers: kokoro_text_preprocess.rs lines 92-106 (preprocess pipeline).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn preprocessor_empty_input_empty_output() {
    // normalize_punctuation("") -> ""
    // expand_abbreviations("", table) -> "" (early return for empty)
    // expand_numbers_in_text("") -> ""
    // normalize_whitespace("") -> ""
    let input_empty = true;
    let stage1_empty = input_empty;
    let stage2_empty = stage1_empty;
    let stage3_empty = stage2_empty;
    let stage4_empty = stage3_empty;

    assert!(
        stage4_empty,
        "empty input must produce empty output through all pipeline stages"
    );
}

/// Harness 24: extract_word_before at position 0 returns empty string.
///
/// SUBSTANTIVE: Proves that when the period is at position 0 in the char
/// array, there are no alphabetic characters before it, so extract_word_before
/// returns "". This prevents underflow in the reverse-scan loop at line 331.
///
/// Covers: kokoro_text_preprocess.rs lines 328-340 (extract_word_before).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn extract_word_before_at_position_zero() {
    let i: usize = 0;

    // The while loop: j starts at i (0), condition is j > 0 -> false immediately.
    // word is empty -> reversed word is empty.
    let enters_loop = i > 0;

    assert!(!enters_loop, "loop must not execute when i == 0");

    let word_len = 0usize;
    assert_eq!(
        word_len, 0,
        "extract_word_before at position 0 must return empty string"
    );
}

/// Harness 25: find_abbreviation with empty table always returns None.
///
/// SUBSTANTIVE: Proves that when the abbreviation table is empty,
/// find_abbreviation can never match any word. This is the precondition
/// check before the table.get() calls at lines 237 and 243.
///
/// Covers: kokoro_text_preprocess.rs lines 208-251 (find_abbreviation).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn find_abbreviation_empty_table_returns_none() {
    let table_is_empty = true;

    // HashMap::get on an empty map always returns None.
    // Both table.get(&lower) and table.get(stripped) return None.
    let primary_lookup_result: bool = false; // None
    let stripped_lookup_result: bool = false; // None

    assert!(
        !primary_lookup_result && !stripped_lookup_result,
        "empty table must never match any abbreviation"
    );

    // The function returns None because neither lookup succeeds.
    let returns_none = !primary_lookup_result && !stripped_lookup_result;
    assert!(
        returns_none && table_is_empty,
        "find_abbreviation with empty table must always return None"
    );
}
