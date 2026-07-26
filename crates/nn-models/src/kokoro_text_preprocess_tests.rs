// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

// -- Punctuation normalization ------------------------------------------------

#[test]
fn test_normalize_smart_quotes() {
    assert_eq!(normalize_punctuation("\u{201C}hello\u{201D}"), "\"hello\"");
    assert_eq!(normalize_punctuation("\u{2018}hello\u{2019}"), "'hello'");
}

#[test]
fn test_normalize_repeated_punctuation() {
    assert_eq!(normalize_punctuation("wow!!!"), "wow!");
    assert_eq!(normalize_punctuation("really???"), "really?");
}

#[test]
fn test_normalize_ellipsis_char() {
    assert_eq!(normalize_punctuation("wait\u{2026}"), "wait...");
}

#[test]
fn test_normalize_en_dash() {
    assert_eq!(normalize_punctuation("foo \u{2013} bar"), "foo , bar");
}

// -- Whitespace normalization -------------------------------------------------

#[test]
fn test_normalize_whitespace_basic() {
    assert_eq!(normalize_whitespace("  hello   world  "), "hello world");
    assert_eq!(normalize_whitespace("a\t\nb"), "a b");
}

// -- Abbreviation expansion ---------------------------------------------------

#[test]
fn test_abbreviation_expansion() {
    let table = default_abbreviations();
    assert_eq!(
        expand_abbreviations("Dr. Smith went to St. Luke", &table),
        "Doctor Smith went to Street Luke"
    );
}

#[test]
fn test_abbreviation_case_insensitive() {
    let table = default_abbreviations();
    // Abbreviation matching is case-insensitive
    let result = expand_abbreviations("DR. JONES", &table);
    // DR. lowercases to dr. which matches
    assert!(result.contains("Doctor") || result.contains("Drive"));
}

#[test]
fn test_abbreviation_etc() {
    let table = default_abbreviations();
    assert_eq!(
        expand_abbreviations("cats, dogs, etc.", &table),
        "cats, dogs, et cetera"
    );
}

#[test]
fn test_abbreviation_no_partial_match() {
    let table = default_abbreviations();
    // "street" should NOT be expanded (only "st." with period)
    assert_eq!(
        expand_abbreviations("the street is long", &table),
        "the street is long"
    );
}

// -- Sentence splitting -------------------------------------------------------

#[test]
fn test_split_sentences_basic() {
    let pp = TextPreprocessor::english();
    let sentences = pp.split_sentences("Hello world. How are you? I am fine!");
    assert_eq!(sentences.len(), 3);
    assert_eq!(sentences[0], "Hello world.");
    assert_eq!(sentences[1], "How are you?");
    assert_eq!(sentences[2], "I am fine!");
}

#[test]
fn test_split_sentences_abbreviation() {
    let pp = TextPreprocessor::english();
    let sentences = pp.split_sentences("Dr. Smith is here. He is great.");
    // "Dr." should not cause a split
    assert_eq!(sentences.len(), 2);
    assert_eq!(sentences[0], "Dr. Smith is here.");
    assert_eq!(sentences[1], "He is great.");
}

#[test]
fn test_split_sentences_single() {
    let pp = TextPreprocessor::english();
    let sentences = pp.split_sentences("Just one sentence");
    assert_eq!(sentences.len(), 1);
    assert_eq!(sentences[0], "Just one sentence");
}

#[test]
fn test_split_sentences_ellipsis() {
    let pp = TextPreprocessor::english();
    let sentences = pp.split_sentences("Wait... Really? Yes!");
    assert_eq!(sentences.len(), 3);
}

// -- Full pipeline (TextPreprocessor) -----------------------------------------

#[test]
fn test_preprocessor_full_pipeline() {
    let pp = TextPreprocessor::english();
    let result = pp.preprocess("I have 42 cats!!!");
    assert_eq!(result, "I have forty two cats!");
}

#[test]
fn test_preprocessor_currency_and_punctuation() {
    let pp = TextPreprocessor::english();
    let result = pp.preprocess("It costs $3.50\u{2026}");
    assert_eq!(result, "It costs three dollars and fifty cents...");
}

#[test]
fn test_preprocessor_abbreviation_and_number() {
    let pp = TextPreprocessor::english();
    let result = pp.preprocess("Dr. Smith has 3 patients");
    assert_eq!(result, "Doctor Smith has three patients");
}

#[test]
fn test_preprocessor_empty() {
    let pp = TextPreprocessor::english();
    assert_eq!(pp.preprocess(""), "");
}

#[test]
fn test_preprocessor_passthrough() {
    let pp = TextPreprocessor::english();
    assert_eq!(pp.preprocess("hello world"), "hello world");
}

#[test]
fn test_preprocessor_minimal_no_abbreviations() {
    let pp = TextPreprocessor::minimal();
    let result = pp.preprocess("Dr. Smith has 42 cats");
    // No abbreviation expansion, but numbers still expand
    assert_eq!(result, "Dr. Smith has forty two cats");
}

#[test]
fn test_preprocessor_ordinal_in_context() {
    let pp = TextPreprocessor::english();
    let result = pp.preprocess("The 21st century is here.");
    assert_eq!(result, "The twenty first century is here.");
}
