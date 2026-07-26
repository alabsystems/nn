// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`EspeakRemapper`], [`RemapTable`], and [`PhonemeLexicon`].

use super::*;

// -- RemapTable tests --------------------------------------------------------

#[test]
fn test_remap_table_longest_match_first() {
    let table = RemapTable::new(vec![("aɪ".into(), "I".into()), ("a".into(), "X".into())]);
    // "aɪ" should match as diphthong "I", not "X" + "ɪ"
    assert_eq!(table.apply("aɪ"), "I");
    // Bare "a" should match as "X"
    assert_eq!(table.apply("a"), "X");
}

#[test]
fn test_remap_table_multiple_occurrences() {
    let table = RemapTable::new(vec![("dʒ".into(), "ʤ".into())]);
    assert_eq!(table.apply("dʒɛdʒ"), "ʤɛʤ");
}

#[test]
fn test_remap_table_empty() {
    let table = RemapTable::new(vec![]);
    assert_eq!(table.apply("hello"), "hello");
    assert!(table.is_empty());
}

#[test]
fn test_remap_table_insert_and_remove() {
    let mut table = RemapTable::new(vec![]);
    table.insert("foo", "bar");
    assert_eq!(table.len(), 1);
    assert_eq!(table.apply("foo"), "bar");
    assert_eq!(table.remove("foo"), Some("bar".into()));
    assert!(table.is_empty());
}

// -- EspeakRemapper tests ----------------------------------------------------

#[test]
fn test_english_us_diphthongs() {
    let r = EspeakRemapper::english_us();
    assert_eq!(r.remap("aɪ"), "I", "aɪ → I (eye)");
    assert_eq!(r.remap("aʊ"), "W", "aʊ → W (how)");
    assert_eq!(r.remap("eɪ"), "A", "eɪ → A (hey)");
    assert_eq!(r.remap("oʊ"), "O", "oʊ → O (go)");
    assert_eq!(r.remap("ɔɪ"), "Y", "ɔɪ → Y (boy)");
}

#[test]
fn test_english_us_affricates() {
    let r = EspeakRemapper::english_us();
    assert_eq!(r.remap("dʒ"), "ʤ", "dʒ → ʤ (jump)");
    assert_eq!(r.remap("tʃ"), "ʧ", "tʃ → ʧ (church)");
}

#[test]
fn test_english_us_r_coloring() {
    let r = EspeakRemapper::english_us();
    assert_eq!(r.remap("ɚ"), "əɹ", "ɚ → əɹ");
    assert_eq!(r.remap("ɜːɹ"), "ɜɹ", "ɜːɹ → ɜɹ (US nurse)");
}

#[test]
fn test_english_us_strips_tie_bar() {
    let r = EspeakRemapper::english_us();
    // espeak uses tie='^' for affricates: d^ʒ → strip ^ first, then dʒ → ʤ
    assert_eq!(r.remap("d^ʒ"), "ʤ");
    assert_eq!(r.remap("t^ʃ"), "ʧ");
}

#[test]
fn test_english_us_r_substitution() {
    let r = EspeakRemapper::english_us();
    assert_eq!(r.remap("r"), "ɹ", "r → ɹ");
}

#[test]
fn test_english_gb_diphthongs() {
    let r = EspeakRemapper::english_gb();
    assert_eq!(r.remap("əʊ"), "Q", "əʊ → Q (GB go)");
    assert_eq!(r.remap("eə"), "ɛː", "eə → ɛː (GB there)");
}

#[test]
fn test_multilingual_nasal_vowels() {
    let r = EspeakRemapper::multilingual();
    assert_eq!(r.remap("ɔ\u{0303}"), "C", "ɔ̃ → C (French bon)");
    assert_eq!(r.remap("ɑ\u{0303}"), "D", "ɑ̃ → D (French dans)");
    assert_eq!(r.remap("ɛ\u{0303}"), "E", "ɛ̃ → E (French vin)");
}

#[test]
fn test_multilingual_affricates() {
    let r = EspeakRemapper::multilingual();
    assert_eq!(r.remap("ts"), "ʦ");
    assert_eq!(r.remap("dz"), "ʣ");
}

#[test]
fn test_english_us_complex_word() {
    let r = EspeakRemapper::english_us();
    // "beautiful" espeak IPA: bjˈuːɾɪfəl
    // After remap: strips nothing unusual, ɾ stays (post-processing is separate)
    let result = r.remap("bjˈuːɾɪfəl");
    assert!(result.contains('ˈ'), "stress marker preserved");
    assert!(
        result.contains('ɾ'),
        "ɾ preserved (postprocess_v1 handles it separately)"
    );
}

// -- postprocess_v1 tests ----------------------------------------------------

#[test]
fn test_postprocess_v1_tap_and_glottal() {
    assert_eq!(postprocess_v1("ɾ"), "T", "ɾ → T");
    assert_eq!(postprocess_v1("ʔ"), "t", "ʔ → t");
    assert_eq!(postprocess_v1("bɾʌðɚ"), "bTʌðɚ");
    assert_eq!(postprocess_v1("bɛʔɚ"), "bɛtɚ");
}

#[test]
fn test_postprocess_v1_no_change_without_targets() {
    assert_eq!(postprocess_v1("hɛˈloʊ"), "hɛˈloʊ");
}

// -- PhonemeLexicon tests ----------------------------------------------------

#[test]
fn test_lexicon_from_tsv() {
    let tsv = "# Comment line\nhello\thɛˈloʊ\nworld\twɜːɹld\n\n";
    let lex = PhonemeLexicon::from_tsv(tsv);
    assert_eq!(lex.len(), 2);
    assert_eq!(lex.get("hello"), Some("hɛˈloʊ"));
    assert_eq!(lex.get("HELLO"), Some("hɛˈloʊ"), "case-insensitive lookup");
    assert_eq!(lex.get("world"), Some("wɜːɹld"));
    assert_eq!(lex.get("unknown"), None);
}

#[test]
fn test_lexicon_insert_and_remove() {
    let mut lex = PhonemeLexicon::new();
    assert!(lex.is_empty());
    lex.insert("test", "tɛst");
    assert_eq!(lex.get("test"), Some("tɛst"));
    lex.remove("test");
    assert!(lex.is_empty());
}

// -- EspeakRemapper extensibility tests --------------------------------------

#[test]
fn test_remapper_custom_table_extension() {
    let mut r = EspeakRemapper::english_us();
    // Add a custom mapping
    r.remap_table_mut().insert("ʁ", "ɹ");
    assert_eq!(r.remap("ʁ"), "ɹ", "custom mapping should work");
}

#[test]
fn test_remapper_custom_strip_chars() {
    let mut r = EspeakRemapper::english_us();
    r.strip_chars_mut().push('\u{02D1}'); // ˑ half-length mark
                                          // After stripping ˑ, "ɐ" (U+0250) remaps to "ə"
    assert_eq!(r.remap("ɐˑ"), "ə", "ˑ stripped, then ɐ→ə via table");
}
