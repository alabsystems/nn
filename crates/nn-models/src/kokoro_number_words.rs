// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Number, currency, and ordinal expansion for text preprocessing.
//!
//! Converts numeric text to English words:
//! - Cardinal numbers: "42" → "forty two"
//! - Ordinal numbers: "1st" → "first"
//! - Currency: "$3.50" → "three dollars and fifty cents"
//! - Thousands separators: "1,000,000" → "one million"
//!
//! Used by [`TextPreprocessor`](super::kokoro_text_preprocess::TextPreprocessor)
//! as part of the G2P preprocessing pipeline.

// -- Number-to-words engine ---------------------------------------------------

const ONES: &[&str] = &[
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const TENS: &[&str] = &[
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// Convert a number (0..999_999_999_999) to English words.
#[must_use]
pub(crate) fn number_to_words(n: u64) -> String {
    if n == 0 {
        return "zero".to_owned();
    }

    let mut parts = Vec::new();

    let billions = n / 1_000_000_000;
    let millions = (n / 1_000_000) % 1_000;
    let thousands = (n / 1_000) % 1_000;
    let remainder = n % 1_000;

    if billions > 0 {
        parts.push(format!("{} billion", chunk_to_words(billions as u32)));
    }
    if millions > 0 {
        parts.push(format!("{} million", chunk_to_words(millions as u32)));
    }
    if thousands > 0 {
        parts.push(format!("{} thousand", chunk_to_words(thousands as u32)));
    }
    if remainder > 0 {
        parts.push(chunk_to_words(remainder as u32));
    }

    parts.join(" ")
}

/// Convert a 3-digit chunk (1..999) to words.
fn chunk_to_words(n: u32) -> String {
    debug_assert!(n <= 999);
    let mut parts = Vec::new();

    let hundreds = n / 100;
    let rest = n % 100;

    if hundreds > 0 {
        parts.push(format!("{} hundred", ONES[hundreds as usize]));
    }

    if rest > 0 {
        if rest < 20 {
            parts.push(ONES[rest as usize].to_owned());
        } else {
            let tens = rest / 10;
            let ones = rest % 10;
            if ones > 0 {
                parts.push(format!("{} {}", TENS[tens as usize], ONES[ones as usize]));
            } else {
                parts.push(TENS[tens as usize].to_owned());
            }
        }
    }

    parts.join(" ")
}

/// Convert a number to its ordinal word form.
#[must_use]
pub(crate) fn ordinal_to_words(n: u64) -> String {
    if n == 0 {
        return "zeroth".to_owned();
    }

    let cardinal = number_to_words(n);

    // Special cases for ordinal endings
    if cardinal.ends_with("one") {
        format!("{}first", &cardinal[..cardinal.len() - 3])
    } else if cardinal.ends_with("two") {
        format!("{}second", &cardinal[..cardinal.len() - 3])
    } else if cardinal.ends_with("three") {
        format!("{}third", &cardinal[..cardinal.len() - 5])
    } else if cardinal.ends_with("five") {
        format!("{}fifth", &cardinal[..cardinal.len() - 4])
    } else if cardinal.ends_with("eight") {
        format!("{}eighth", &cardinal[..cardinal.len() - 5])
    } else if cardinal.ends_with("nine") {
        format!("{}ninth", &cardinal[..cardinal.len() - 4])
    } else if cardinal.ends_with("twelve") {
        format!("{}twelfth", &cardinal[..cardinal.len() - 6])
    } else if cardinal.ends_with('y') {
        format!("{}ieth", &cardinal[..cardinal.len() - 1])
    } else {
        format!("{cardinal}th")
    }
}

// -- Number expansion in text -------------------------------------------------

struct Expansion {
    text: String,
    consumed_chars: usize,
}

/// Expand numbers, ordinals, and currency in text.
#[must_use]
pub(crate) fn expand_numbers_in_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len() * 2);
    let mut chars = text.char_indices();

    while let Some((i, ch)) = chars.next() {
        // Currency: $N.NN
        if ch == '$' {
            if let Some(expansion) = try_expand_currency(text, i + 1) {
                result.push_str(&expansion.text);
                // Skip the consumed characters
                for _ in 0..expansion.consumed_chars {
                    chars.next();
                }
                continue;
            }
        }

        // Number: digit sequence possibly followed by ordinal suffix
        if ch.is_ascii_digit() {
            if let Some(expansion) = try_expand_number(text, i) {
                result.push_str(&expansion.text);
                // Skip consumed chars (-1 because we already consumed the first digit)
                for _ in 0..expansion.consumed_chars.saturating_sub(1) {
                    chars.next();
                }
                continue;
            }
        }

        result.push(ch);
    }
    result
}

/// Try to expand a number starting at position `start` in text.
fn try_expand_number(text: &str, start: usize) -> Option<Expansion> {
    let remaining = &text[start..];

    // Collect digits (with optional commas for thousands)
    let mut num_str = String::new();
    let mut byte_len = 0;
    let mut char_count = 0;
    let mut has_commas = false;

    for ch in remaining.chars() {
        if ch.is_ascii_digit() {
            num_str.push(ch);
            byte_len += 1;
            char_count += 1;
        } else if ch == ',' && !num_str.is_empty() {
            // Check if comma is a thousands separator (followed by 3 digits)
            let after_comma = &remaining[byte_len + 1..];
            if after_comma.len() >= 3 && after_comma[..3].chars().all(|c| c.is_ascii_digit()) {
                has_commas = true;
                byte_len += 1;
                char_count += 1;
                // Don't add comma to num_str — strip it
            } else {
                break;
            }
        } else {
            break;
        }
    }

    if num_str.is_empty() {
        return None;
    }

    // Check for ordinal suffix (st, nd, rd, th)
    let after = &remaining[byte_len..];
    let ordinal_suffix = try_ordinal_suffix(after);
    if let Some(suffix_len) = ordinal_suffix {
        let n: u64 = num_str.parse().ok()?;
        if n <= 999_999_999_999 {
            return Some(Expansion {
                text: ordinal_to_words(n),
                consumed_chars: char_count + suffix_len,
            });
        }
    }

    // Plain number
    let n: u64 = num_str.parse().ok()?;
    // Only expand numbers up to a reasonable size
    if n <= 999_999_999_999 {
        let _ = has_commas; // thousands commas already stripped
        Some(Expansion {
            text: number_to_words(n),
            consumed_chars: char_count,
        })
    } else {
        None // Too large, leave as-is
    }
}

/// Check for ordinal suffix (st, nd, rd, th) at start of string.
fn try_ordinal_suffix(text: &str) -> Option<usize> {
    let lower: String = text.chars().take(2).flat_map(char::to_lowercase).collect();
    match lower.as_str() {
        "st" | "nd" | "rd" | "th" => {
            // Make sure suffix isn't part of a longer word
            let after = &text[2..];
            if after.is_empty() || after.chars().next().map_or(true, |c| !c.is_alphanumeric()) {
                Some(2)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Try to expand a dollar amount starting at position `start` (after the '$').
fn try_expand_currency(text: &str, start: usize) -> Option<Expansion> {
    let remaining = &text[start..];

    // Collect integer part
    let mut int_str = String::new();
    let mut byte_len = 0;
    let mut char_count = 0;

    for ch in remaining.chars() {
        if ch.is_ascii_digit() {
            int_str.push(ch);
            byte_len += 1;
            char_count += 1;
        } else if ch == ',' && !int_str.is_empty() {
            let after_comma = &remaining[byte_len + 1..];
            if after_comma.len() >= 3 && after_comma[..3].chars().all(|c| c.is_ascii_digit()) {
                byte_len += 1;
                char_count += 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    if int_str.is_empty() {
        return None;
    }

    let dollars: u64 = int_str.parse().ok()?;

    // Check for cents (.NN)
    let after = &remaining[byte_len..];
    let (cents, extra_chars) = if let Some(after_dot) = after.strip_prefix('.') {
        let cent_str: String = after_dot
            .chars()
            .take(2)
            .filter(char::is_ascii_digit)
            .collect();
        if cent_str.len() == 2 {
            let c: u32 = cent_str.parse().ok()?;
            (c, 3) // dot + 2 digits
        } else if cent_str.len() == 1 {
            let c: u32 = cent_str.parse().ok()?;
            (c * 10, 2) // dot + 1 digit
        } else {
            (0, 0)
        }
    } else {
        (0, 0)
    };

    if dollars > 999_999_999_999 {
        return None;
    }

    let mut words = String::new();

    if dollars > 0 {
        words.push_str(&number_to_words(dollars));
        if dollars == 1 {
            words.push_str(" dollar");
        } else {
            words.push_str(" dollars");
        }
    }

    if cents > 0 {
        if !words.is_empty() {
            words.push_str(" and ");
        }
        words.push_str(&number_to_words(u64::from(cents)));
        if cents == 1 {
            words.push_str(" cent");
        } else {
            words.push_str(" cents");
        }
    }

    if words.is_empty() {
        words.push_str("zero dollars");
    }

    Some(Expansion {
        text: words,
        consumed_chars: char_count + extra_chars,
    })
}

#[cfg(test)]
#[path = "kokoro_number_words_tests.rs"]
mod tests;
