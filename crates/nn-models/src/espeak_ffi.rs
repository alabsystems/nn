// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! FFI bindings to espeak-ng for text-to-IPA conversion.
//!
//! Provides a safe Rust wrapper around espeak-ng's `espeak_TextToPhonemes` C API.
//! Feature-gated behind `espeak` in nn-models Cargo.toml.
//!
//! # Usage
//!
//! ```ignore
//! use nn_models::espeak_ffi::EspeakEngine;
//!
//! let engine = EspeakEngine::new("en-us")?;
//! let ipa = engine.text_to_ipa("Hello world")?;
//! // ipa: "həlˈoʊ wˈɜːɹld" (espeak IPA output)
//! ```
//!
//! # Pipeline Integration
//!
//! ```text
//! Text → EspeakEngine::text_to_ipa() → IPA → EspeakRemapper::remap() → KokoroTokenizer::encode()
//! ```

use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Once;

/// espeak-ng C API constants.
const ESPEAKCCHARS_UTF8: i32 = 1;
/// IPA output mode (bit 1 set) with space separator (0x20 in bits 8-23).
const PHONEMEMODE_IPA: i32 = 0x02 | (b' ' as i32) << 8;

// espeak-ng C FFI declarations — link against libespeak-ng.
extern "C" {
    fn espeak_Initialize(
        output: i32,
        buflength: i32,
        path: *const std::ffi::c_char,
        options: i32,
    ) -> i32;

    fn espeak_SetVoiceByName(name: *const std::ffi::c_char) -> i32;

    fn espeak_TextToPhonemes(
        textptr: *mut *const std::ffi::c_void,
        textmode: i32,
        phonememode: i32,
    ) -> *const std::ffi::c_char;
}

/// Audio output mode: synchronous (no audio playback, phoneme-only).
const AUDIO_OUTPUT_SYNCHRONOUS: i32 = 2;
/// Initialize option: report phonemes as IPA.
const ESPEAK_INITIALIZE_PHONEME_IPA: i32 = 0x0008;

static ESPEAK_INIT: Once = Once::new();
static mut ESPEAK_INIT_RESULT: i32 = -1;

/// Errors from espeak-ng operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EspeakError {
    /// espeak_Initialize failed.
    #[error("espeak_Initialize failed with code {0}")]
    InitFailed(i32),

    /// espeak_SetVoiceByName failed.
    #[error("voice '{voice}' not found (espeak error {code})")]
    VoiceNotFound { voice: String, code: i32 },

    /// Input text contains interior NUL bytes.
    #[error("input text contains NUL byte")]
    NulInInput,

    /// espeak returned a null pointer.
    #[error("espeak_TextToPhonemes returned null")]
    NullResult,
}

/// Safe wrapper around espeak-ng for text-to-IPA conversion.
///
/// NOT Send/Sync — espeak-ng uses global state internally.
/// Create one per thread or protect with a mutex.
pub struct EspeakEngine {
    voice: String,
    _not_send: std::marker::PhantomData<*const ()>,
}

/// Find espeak-ng data directory. Checks `ESPEAK_DATA_PATH` env var,
/// then common homebrew paths on macOS, then returns `None` for system default.
fn find_espeak_data_path() -> Option<String> {
    if let Ok(path) = std::env::var("ESPEAK_DATA_PATH") {
        if std::path::Path::new(&path).exists() {
            return Some(path);
        }
    }
    // Homebrew on Apple Silicon / Intel.
    for candidate in [
        "/opt/homebrew/share/espeak-ng-data",
        "/usr/local/share/espeak-ng-data",
        "/usr/share/espeak-ng-data",
    ] {
        if std::path::Path::new(candidate).exists() {
            // espeak_Initialize expects the parent of espeak-ng-data.
            if let Some(parent) = std::path::Path::new(candidate).parent() {
                return Some(parent.to_string_lossy().into_owned());
            }
        }
    }
    None
}

impl EspeakEngine {
    /// Create a new engine with the given voice (e.g., "en-us", "en-gb", "fr").
    ///
    /// Initializes espeak-ng on first call (process-global, once).
    pub fn new(voice: &str) -> Result<Self, EspeakError> {
        // Initialize espeak-ng once per process.
        ESPEAK_INIT.call_once(|| {
            // Resolve data directory: env override, then homebrew paths, then NULL (system default).
            let data_path = find_espeak_data_path();
            let path_c = data_path.as_ref().map(|p| {
                CString::new(p.as_str()).expect("espeak data path should not contain NUL")
            });
            let path_ptr = path_c.as_ref().map_or(ptr::null(), |c| c.as_ptr());

            // SAFETY: espeak_Initialize is safe to call once with valid args.
            let result = unsafe {
                espeak_Initialize(
                    AUDIO_OUTPUT_SYNCHRONOUS,
                    0,
                    path_ptr,
                    ESPEAK_INITIALIZE_PHONEME_IPA,
                )
            };
            // SAFETY: write to static mut is safe inside Once::call_once.
            unsafe {
                ESPEAK_INIT_RESULT = result;
            }
        });

        // SAFETY: read after Once::call_once guarantees initialization is complete.
        let init_result = unsafe { ESPEAK_INIT_RESULT };
        if init_result == -1 {
            return Err(EspeakError::InitFailed(init_result));
        }

        // Set voice.
        let voice_c = CString::new(voice).map_err(|_| EspeakError::NulInInput)?;
        // SAFETY: voice_c is a valid C string, espeak_SetVoiceByName is safe after init.
        let rc = unsafe { espeak_SetVoiceByName(voice_c.as_ptr()) };
        if rc != 0 {
            return Err(EspeakError::VoiceNotFound {
                voice: voice.to_owned(),
                code: rc,
            });
        }

        Ok(Self {
            voice: voice.to_owned(),
            _not_send: std::marker::PhantomData,
        })
    }

    /// The voice name this engine was created with.
    #[must_use]
    pub fn voice(&self) -> &str {
        &self.voice
    }

    /// Convert text to IPA phonemes.
    ///
    /// Returns the full IPA transcription as a UTF-8 string.
    /// Espeak processes the text in segments (up to sentence/clause boundaries),
    /// so this function loops until the full text is consumed.
    pub fn text_to_ipa(&self, text: &str) -> Result<String, EspeakError> {
        let text_c = CString::new(text).map_err(|_| EspeakError::NulInInput)?;
        let mut result = String::new();

        // Re-set voice before each call (espeak global state may have been changed).
        let voice_c = CString::new(self.voice.as_str()).map_err(|_| EspeakError::NulInInput)?;
        // SAFETY: valid C string after init.
        let rc = unsafe { espeak_SetVoiceByName(voice_c.as_ptr()) };
        if rc != 0 {
            return Err(EspeakError::VoiceNotFound {
                voice: self.voice.clone(),
                code: rc,
            });
        }

        // textptr is advanced by espeak_TextToPhonemes on each call,
        // set to NULL when the entire text has been processed.
        let mut textptr: *const std::ffi::c_void = text_c.as_ptr() as *const std::ffi::c_void;

        loop {
            // SAFETY: textptr points into valid CString memory.
            // espeak_TextToPhonemes returns a pointer to an internal static buffer
            // that is valid until the next call.
            let phonemes_ptr = unsafe {
                espeak_TextToPhonemes(
                    &mut textptr as *mut *const std::ffi::c_void,
                    ESPEAKCCHARS_UTF8,
                    PHONEMEMODE_IPA,
                )
            };

            if phonemes_ptr.is_null() {
                break;
            }

            // SAFETY: espeak guarantees the returned pointer is a valid C string.
            let segment = unsafe { CStr::from_ptr(phonemes_ptr) };
            if let Ok(s) = segment.to_str() {
                if !result.is_empty() && !s.is_empty() {
                    result.push(' ');
                }
                result.push_str(s);
            }

            // textptr == NULL means all text consumed.
            if textptr.is_null() {
                break;
            }
        }

        Ok(result)
    }
}

/// Full text-to-Kokoro-tokens pipeline combining espeak FFI + remapper + tokenizer.
///
/// This is the zero-Python replacement for misaki + espeak-ng subprocess.
///
/// ```ignore
/// let tokens = text_to_kokoro_tokens("Hello world", "en-us")?;
/// ```
pub fn text_to_kokoro_tokens(text: &str, voice: &str) -> Result<Vec<u32>, TextToTokensError> {
    use crate::kokoro_g2p::EspeakRemapper;
    use crate::kokoro_tokenizer::{KokoroTokenizer, KokoroVocab};

    let engine = EspeakEngine::new(voice)?;
    let ipa = engine.text_to_ipa(text)?;

    // Select remapper based on voice.
    let remapper = if voice.starts_with("en") {
        if voice.contains("gb") {
            EspeakRemapper::english_gb()
        } else {
            EspeakRemapper::english_us()
        }
    } else {
        EspeakRemapper::multilingual()
    };

    let kokoro_phonemes = remapper.remap(&ipa);
    let tokenizer = KokoroTokenizer::new(KokoroVocab::kokoro_default());
    let tokens = tokenizer.encode(&kokoro_phonemes)?;

    Ok(tokens)
}

/// Errors from the full text-to-tokens pipeline.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TextToTokensError {
    /// espeak-ng error.
    #[error("espeak: {0}")]
    Espeak(#[from] EspeakError),

    /// Tokenizer error.
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] crate::kokoro_error::KokoroError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_espeak_engine_init_english() {
        let engine = EspeakEngine::new("en-us").expect("should init with en-us voice");
        assert_eq!(engine.voice(), "en-us");
    }

    #[test]
    fn test_espeak_text_to_ipa_hello() {
        let engine = EspeakEngine::new("en-us").expect("should init");
        let ipa = engine
            .text_to_ipa("Hello world")
            .expect("should produce IPA");
        assert!(!ipa.is_empty(), "IPA should not be empty");
        // espeak IPA for "hello" should contain 'h' and some vowel.
        assert!(ipa.contains('h'), "IPA should contain 'h': {ipa}");
    }

    #[test]
    fn test_espeak_text_to_ipa_empty() {
        let engine = EspeakEngine::new("en-us").expect("should init");
        let ipa = engine.text_to_ipa("").expect("empty text should not error");
        assert!(ipa.is_empty() || ipa.trim().is_empty());
    }

    #[test]
    fn test_espeak_text_to_ipa_multisentence() {
        let engine = EspeakEngine::new("en-us").expect("should init");
        let ipa = engine
            .text_to_ipa("Hello. How are you?")
            .expect("should handle multiple sentences");
        assert!(!ipa.is_empty());
    }

    #[test]
    fn test_espeak_invalid_voice() {
        let result = EspeakEngine::new("xx-nonexistent-zz");
        assert!(result.is_err(), "nonexistent voice should fail");
    }

    #[test]
    fn test_full_pipeline_text_to_tokens() {
        let tokens =
            text_to_kokoro_tokens("Hello world", "en-us").expect("full pipeline should work");
        assert!(!tokens.is_empty(), "should produce tokens");
        // Token 0 is PAD, tokens should be > 0 for actual phonemes.
        assert!(tokens.iter().any(|&t| t > 0), "should have non-pad tokens");
    }

    #[test]
    fn test_pipeline_produces_consistent_output() {
        let tokens1 = text_to_kokoro_tokens("Testing one two three", "en-us").expect("should work");
        let tokens2 = text_to_kokoro_tokens("Testing one two three", "en-us").expect("should work");
        assert_eq!(tokens1, tokens2, "same input should produce same tokens");
    }
}
