// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

// -- Number expansion ---------------------------------------------------------

#[test]
fn test_number_to_words_zero() {
    assert_eq!(number_to_words(0), "zero");
}

#[test]
fn test_number_to_words_ones() {
    assert_eq!(number_to_words(1), "one");
    assert_eq!(number_to_words(9), "nine");
    assert_eq!(number_to_words(13), "thirteen");
    assert_eq!(number_to_words(19), "nineteen");
}

#[test]
fn test_number_to_words_tens() {
    assert_eq!(number_to_words(20), "twenty");
    assert_eq!(number_to_words(42), "forty two");
    assert_eq!(number_to_words(99), "ninety nine");
}

#[test]
fn test_number_to_words_hundreds() {
    assert_eq!(number_to_words(100), "one hundred");
    assert_eq!(number_to_words(123), "one hundred twenty three");
    assert_eq!(number_to_words(500), "five hundred");
    assert_eq!(number_to_words(999), "nine hundred ninety nine");
}

#[test]
fn test_number_to_words_thousands() {
    assert_eq!(number_to_words(1000), "one thousand");
    assert_eq!(number_to_words(1001), "one thousand one");
    assert_eq!(
        number_to_words(12345),
        "twelve thousand three hundred forty five"
    );
    assert_eq!(number_to_words(100000), "one hundred thousand");
}

#[test]
fn test_number_to_words_millions() {
    assert_eq!(number_to_words(1_000_000), "one million");
    assert_eq!(
        number_to_words(1_234_567),
        "one million two hundred thirty four thousand five hundred sixty seven"
    );
}

#[test]
fn test_number_to_words_billions() {
    assert_eq!(number_to_words(1_000_000_000), "one billion");
    assert_eq!(number_to_words(7_000_000_001), "seven billion one");
}

// -- Ordinal expansion --------------------------------------------------------

#[test]
fn test_ordinal_to_words_basic() {
    assert_eq!(ordinal_to_words(1), "first");
    assert_eq!(ordinal_to_words(2), "second");
    assert_eq!(ordinal_to_words(3), "third");
    assert_eq!(ordinal_to_words(4), "fourth");
    assert_eq!(ordinal_to_words(5), "fifth");
}

#[test]
fn test_ordinal_to_words_teens() {
    assert_eq!(ordinal_to_words(11), "eleventh");
    assert_eq!(ordinal_to_words(12), "twelfth");
    assert_eq!(ordinal_to_words(13), "thirteenth");
}

#[test]
fn test_ordinal_to_words_tens() {
    assert_eq!(ordinal_to_words(20), "twentieth");
    assert_eq!(ordinal_to_words(21), "twenty first");
    assert_eq!(ordinal_to_words(32), "thirty second");
    assert_eq!(ordinal_to_words(43), "forty third");
}

#[test]
fn test_ordinal_to_words_large() {
    assert_eq!(ordinal_to_words(100), "one hundredth");
    assert_eq!(ordinal_to_words(101), "one hundred first");
    assert_eq!(ordinal_to_words(1000), "one thousandth");
}

// -- Number expansion in text -------------------------------------------------

#[test]
fn test_expand_numbers_in_text_basic() {
    assert_eq!(
        expand_numbers_in_text("I have 42 cats"),
        "I have forty two cats"
    );
}

#[test]
fn test_expand_numbers_ordinals() {
    assert_eq!(expand_numbers_in_text("the 1st place"), "the first place");
    assert_eq!(expand_numbers_in_text("the 2nd item"), "the second item");
    assert_eq!(expand_numbers_in_text("the 3rd option"), "the third option");
    assert_eq!(
        expand_numbers_in_text("the 4th element"),
        "the fourth element"
    );
}

#[test]
fn test_expand_numbers_no_false_ordinal() {
    // "1stand" — "st" is followed by "and" (alphanumeric), so NOT an ordinal.
    // "1" is consumed as plain number → "one", "stand" remains.
    assert_eq!(expand_numbers_in_text("1stand"), "onestand");
    // True ordinal: "1st " — "st" followed by space (non-alphanumeric)
    assert_eq!(expand_numbers_in_text("1st "), "first ");
}

#[test]
fn test_expand_numbers_preserves_non_numbers() {
    assert_eq!(expand_numbers_in_text("hello world"), "hello world");
    assert_eq!(expand_numbers_in_text(""), "");
}

#[test]
fn test_expand_numbers_with_commas() {
    assert_eq!(
        expand_numbers_in_text("population is 1,000,000"),
        "population is one million"
    );
    assert_eq!(
        expand_numbers_in_text("earned 12,345 points"),
        "earned twelve thousand three hundred forty five points"
    );
}

// -- Currency expansion -------------------------------------------------------

#[test]
fn test_currency_basic() {
    let result = expand_numbers_in_text("costs $5");
    assert_eq!(result, "costs five dollars");
}

#[test]
fn test_currency_with_cents() {
    let result = expand_numbers_in_text("price is $3.50");
    assert_eq!(result, "price is three dollars and fifty cents");
}

#[test]
fn test_currency_cents_only() {
    let result = expand_numbers_in_text("only $0.99");
    assert_eq!(result, "only ninety nine cents");
}

#[test]
fn test_currency_one_dollar() {
    let result = expand_numbers_in_text("just $1");
    assert_eq!(result, "just one dollar");
}

#[test]
fn test_currency_one_cent() {
    let result = expand_numbers_in_text("$0.01 left");
    assert_eq!(result, "one cent left");
}

#[test]
fn test_currency_large() {
    let result = expand_numbers_in_text("worth $1,000,000");
    assert_eq!(result, "worth one million dollars");
}
