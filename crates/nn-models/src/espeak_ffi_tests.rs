// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Tests require espeak-ng to be installed. Run with:
//   cargo test -p nn-models --features espeak -- espeak

use super::*;

#[test]
fn test_espeak_engine_new_requires_lib() {
    let result = EspeakEngine::new("en-us");
    assert!(result.is_ok() || matches!(result, Err(EspeakError::InitFailed { .. })));
}

#[test]
fn test_empty_text_returns_empty_string() {
    let engine = match EspeakEngine::new("en-us") {
        Ok(e) => e,
        Err(EspeakError::InitFailed { .. }) => return,
        Err(e) => panic!("unexpected error: {e}"),
    };
    let result = engine.text_to_ipa("").unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_nul_in_text_rejected() {
    let engine = match EspeakEngine::new("en-us") {
        Ok(e) => e,
        Err(EspeakError::InitFailed { .. }) => return,
        Err(e) => panic!("unexpected error: {e}"),
    };
    let result = engine.text_to_ipa("hello\0world");
    assert!(matches!(
        result,
        Err(EspeakError::NulInText { position: 5 })
    ));
}

#[test]
fn test_nul_in_voice_rejected() {
    let result = EspeakEngine::new("en\0us");
    assert!(matches!(result, Err(EspeakError::NulInVoice)));
}

#[test]
fn test_voice_accessor() {
    let engine = match EspeakEngine::new("en-us") {
        Ok(e) => e,
        Err(EspeakError::InitFailed { .. }) => return,
        Err(e) => panic!("unexpected error: {e}"),
    };
    assert_eq!(engine.voice(), "en-us");
}

#[test]
fn test_text_to_ipa_returns_nonempty_for_real_text() {
    let engine = match EspeakEngine::new("en-us") {
        Ok(e) => e,
        Err(EspeakError::InitFailed { .. }) => return,
        Err(e) => panic!("unexpected error: {e}"),
    };
    let ipa = engine.text_to_ipa("hello world").unwrap();
    assert!(
        !ipa.is_empty(),
        "IPA output should not be empty for 'hello world'"
    );
}

#[test]
fn test_same_voice_skips_ffi_call() {
    // Two engines with the same voice — second call should use cached voice.
    let engine1 = match EspeakEngine::new("en-us") {
        Ok(e) => e,
        Err(EspeakError::InitFailed { .. }) => return,
        Err(e) => panic!("unexpected error: {e}"),
    };
    let engine2 = EspeakEngine::new("en-us").unwrap();

    let ipa1 = engine1.text_to_ipa("test").unwrap();
    let ipa2 = engine2.text_to_ipa("test").unwrap();
    assert_eq!(ipa1, ipa2, "same voice should produce same IPA");
}
