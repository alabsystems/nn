// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Key sourcing for HMAC-SHA256 certificate signing.
//!
//! Three key sources, checked in priority order:
//! 1. `NN_SIGNING_KEY` env var — hex-encoded key bytes
//! 2. `NN_SIGNING_KEY_FILE` env var — path to raw key bytes
//! 3. Programmatic — caller passes key to `CertifyConfig::with_signing_key()`
//!
//! No key configured = unsigned certificates (same as current behavior).
//!
//! Part of #3253, #3020.

/// Key source for HMAC-SHA256 certificate signing.
///
/// # Memory safety
///
/// - **Zeroization on drop:** The `Raw` variant zeros key bytes before
///   deallocation using volatile writes (not optimizable away).
/// - **Redacted debug output:** `Debug` prints key length, never key bytes.
///   Prevents accidental key leakage via logging, error messages, or panics.
#[derive(Clone, Default)]
pub enum SigningKey {
    /// Raw key bytes (from env or caller).
    Raw(Vec<u8>),
    /// No signing configured.
    #[default]
    None,
}

impl std::fmt::Debug for SigningKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Raw(bytes) => write!(f, "SigningKey::Raw([REDACTED; {} bytes])", bytes.len()),
            Self::None => write!(f, "SigningKey::None"),
        }
    }
}

impl Drop for SigningKey {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl SigningKey {
    /// Zero key material using volatile writes (not optimizable away).
    ///
    /// Called automatically by `Drop`. Extracted so tests can verify
    /// zeroing on a live object without reading freed memory (UB).
    fn zeroize(&mut self) {
        if let Self::Raw(ref mut bytes) = self {
            for byte in bytes.iter_mut() {
                // SAFETY: byte is a valid, aligned, dereferenceable pointer
                // to an initialized u8 within the Vec's allocation.
                unsafe {
                    std::ptr::write_volatile(std::ptr::from_mut::<u8>(byte), 0);
                }
            }
            // Compiler fence prevents reordering of the volatile writes
            // with respect to the Vec's deallocation in Drop.
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Resolve signing key from environment variables.
    ///
    /// Checks `NN_SIGNING_KEY` (hex-encoded) first, then `NN_SIGNING_KEY_FILE`
    /// (path to raw key bytes). Returns `SigningKey::None` if neither is set.
    ///
    /// Prints warnings to stderr on invalid hex or unreadable files, but does
    /// not fail — callers get `None` and certificates remain unsigned.
    #[must_use]
    pub fn from_env() -> Self {
        if let Ok(hex) = std::env::var("NN_SIGNING_KEY") {
            match hex_decode(&hex) {
                Ok(bytes) => {
                    if bytes.len() < 32 {
                        eprintln!(
                            "warning: NN_SIGNING_KEY is {} bytes (recommend >= 32)",
                            bytes.len()
                        );
                    }
                    return Self::Raw(bytes);
                }
                Err(e) => {
                    eprintln!("warning: NN_SIGNING_KEY invalid hex: {e}");
                    return Self::None;
                }
            }
        }
        if let Ok(path) = std::env::var("NN_SIGNING_KEY_FILE") {
            match std::fs::read(&path) {
                Ok(bytes) => {
                    if bytes.len() < 32 {
                        eprintln!(
                            "warning: NN_SIGNING_KEY_FILE key is {} bytes (recommend >= 32)",
                            bytes.len()
                        );
                    }
                    return Self::Raw(bytes);
                }
                Err(e) => {
                    eprintln!("warning: NN_SIGNING_KEY_FILE unreadable: {e}");
                    return Self::None;
                }
            }
        }
        Self::None
    }

    /// Returns the key bytes if configured.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Raw(bytes) => Some(bytes),
            Self::None => None,
        }
    }

    /// Returns `true` if no signing key is configured.
    #[must_use]
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Decode a hex string into bytes.
///
/// Rejects odd-length strings and non-hex characters.
pub(crate) fn hex_decode(hex: &str) -> Result<Vec<u8>, HexDecodeError> {
    let hex = hex.trim();
    if !hex.len().is_multiple_of(2) {
        return Err(HexDecodeError::OddLength);
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for chunk in hex.as_bytes().chunks(2) {
        let hi = hex_digit(chunk[0])?;
        let lo = hex_digit(chunk[1])?;
        bytes.push((hi << 4) | lo);
    }
    Ok(bytes)
}

/// Encode bytes as a lowercase hex string with no separators.
///
/// Matches the digest-0.10 `format!("{:x}", GenericArray)` output exactly, so
/// certificate hashes/signatures stay byte-compatible across the digest 0.11
/// upgrade (whose `Array` output type no longer implements `LowerHex`).
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn hex_digit(b: u8) -> Result<u8, HexDecodeError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(HexDecodeError::InvalidChar(b as char)),
    }
}

#[derive(Debug)]
pub(crate) enum HexDecodeError {
    OddLength,
    InvalidChar(char),
}

impl std::fmt::Display for HexDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OddLength => write!(f, "odd-length hex string"),
            Self::InvalidChar(c) => write!(f, "invalid hex character: '{c}'"),
        }
    }
}

#[cfg(kani)]
#[path = "signing_config_kani.rs"]
mod kani_proofs;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;

    /// Serializes tests that mutate `NN_SIGNING_KEY` / `NN_SIGNING_KEY_FILE`
    /// env vars. Without this, parallel test threads race on shared process
    /// environment state (env vars are process-global, not thread-local).
    ///
    /// Fix for #3318 item 5: env var race in signing_config.rs tests.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_hex_decode_valid() {
        assert_eq!(hex_decode("").unwrap(), Vec::<u8>::new());
        assert_eq!(hex_decode("00").unwrap(), vec![0u8]);
        assert_eq!(hex_decode("ff").unwrap(), vec![255]);
        assert_eq!(hex_decode("FF").unwrap(), vec![255]);
        assert_eq!(
            hex_decode("0123456789abcdef").unwrap(),
            vec![0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef]
        );
    }

    #[test]
    fn test_hex_decode_invalid() {
        assert!(hex_decode("0").is_err()); // odd length
        assert!(hex_decode("zz").is_err()); // invalid char
        assert!(hex_decode("0g").is_err()); // invalid char
    }

    #[test]
    fn test_hex_decode_trims_whitespace() {
        assert_eq!(hex_decode("  ff  ").unwrap(), vec![255]);
    }

    #[test]
    fn test_signing_key_none_default() {
        let key = SigningKey::default();
        assert!(key.is_none());
        assert!(key.as_bytes().is_none());
    }

    #[test]
    fn test_signing_key_raw() {
        let key = SigningKey::Raw(vec![1, 2, 3]);
        assert!(!key.is_none());
        assert_eq!(key.as_bytes().unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn test_signing_key_from_env_hex() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // 32-byte key as hex
        let hex_key = "ab".repeat(32); // 32-byte test key, generated so no literal looks like a credential
        std::env::set_var("NN_SIGNING_KEY", &hex_key);
        // Clear file var to prevent interference
        std::env::remove_var("NN_SIGNING_KEY_FILE");

        let key = SigningKey::from_env();
        assert!(!key.is_none());
        assert_eq!(key.as_bytes().unwrap().len(), 32);

        // Cleanup
        std::env::remove_var("NN_SIGNING_KEY");
    }

    #[test]
    fn test_signing_key_from_env_file() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("NN_SIGNING_KEY");

        let dir = std::env::temp_dir().join(format!("nn_signing_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let key_path = dir.join("test_key.bin");
        let key_bytes: Vec<u8> = (0..32).collect();
        let mut f = std::fs::File::create(&key_path).unwrap();
        f.write_all(&key_bytes).unwrap();

        std::env::set_var("NN_SIGNING_KEY_FILE", key_path.to_str().unwrap());

        let key = SigningKey::from_env();
        assert!(!key.is_none());
        assert_eq!(key.as_bytes().unwrap(), &key_bytes);

        // Cleanup
        std::env::remove_var("NN_SIGNING_KEY_FILE");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_signing_key_none_without_env() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("NN_SIGNING_KEY");
        std::env::remove_var("NN_SIGNING_KEY_FILE");

        let key = SigningKey::from_env();
        assert!(key.is_none());
    }

    // --- Memory safety tests ---

    #[test]
    fn test_signing_key_debug_redacts_key_material() {
        let key = SigningKey::Raw(vec![0xDE, 0xAD, 0xBE, 0xEF]);
        let debug_str = format!("{key:?}");

        // Must NOT contain actual key bytes in any representation.
        assert!(
            !debug_str.contains("222"), // 0xDE = 222
            "Debug output leaks key byte as decimal: {debug_str}"
        );
        // "REDACTED" contains "DE" so we can't naively check for "DE".
        // Instead, check no bracket-delimited byte lists appear.
        assert!(
            !debug_str.contains("[222") && !debug_str.contains("[0xDE"),
            "Debug output leaks key bytes as list: {debug_str}"
        );
        // Must contain REDACTED marker and correct length.
        assert!(
            debug_str.contains("REDACTED"),
            "Debug should contain REDACTED: {debug_str}"
        );
        assert!(
            debug_str.contains("4 bytes"),
            "Debug should show key length: {debug_str}"
        );
    }

    #[test]
    fn test_signing_key_debug_none_variant() {
        let key = SigningKey::None;
        let debug_str = format!("{key:?}");
        assert_eq!(debug_str, "SigningKey::None");
    }

    #[test]
    fn test_signing_key_zeroize_clears_bytes() {
        let mut key = SigningKey::Raw(vec![0xAA; 64]);

        // Verify bytes are non-zero before zeroization.
        assert!(
            key.as_bytes().unwrap().iter().all(|&b| b == 0xAA),
            "bytes should be 0xAA before zeroize"
        );

        // Call zeroize() directly — same code path as Drop::drop().
        // Object is still alive, so reading bytes is sound (no UB).
        key.zeroize();

        assert!(
            key.as_bytes().unwrap().iter().all(|&b| b == 0),
            "all bytes should be 0x00 after zeroize"
        );
    }

    #[test]
    fn test_signing_key_clone_is_independent() {
        let key1 = SigningKey::Raw(vec![1, 2, 3, 4]);
        let key2 = key1.clone();

        // Both should have the same bytes.
        assert_eq!(key1.as_bytes(), key2.as_bytes());

        // Dropping one should not affect the other.
        drop(key1);
        assert_eq!(key2.as_bytes().unwrap(), &[1, 2, 3, 4]);
    }
}
