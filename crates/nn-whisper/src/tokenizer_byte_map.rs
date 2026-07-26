// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPT-2 byte ↔ Unicode mapping tables.
//!
//! GPT-2 maps each byte value (0-255) to a specific Unicode codepoint to avoid
//! control characters in the vocabulary. This is a fixed, deterministic mapping.

use std::collections::HashMap;

/// Build the GPT-2 byte-to-unicode forward mapping.
///
/// Returns: byte value → unicode character.
pub(super) fn build_byte_encoder() -> HashMap<u8, char> {
    let mut mapping = HashMap::new();

    // Printable ASCII range: '!' (33) through '~' (126).
    for b in b'!'..=b'~' {
        mapping.insert(b, char::from(b));
    }
    // Latin-1 supplement: 0xA1 through 0xAC.
    for b in 0xA1u8..=0xACu8 {
        mapping.insert(b, char::from(b));
    }
    // Latin-1 supplement: 0xAE through 0xFF.
    for b in 0xAEu8..=0xFFu8 {
        mapping.insert(b, char::from(b));
    }

    // Remaining byte values map to Unicode codepoints starting at 256 (U+0100).
    let mut n = 256u32;
    for b in 0u8..=255u8 {
        if let std::collections::hash_map::Entry::Vacant(e) = mapping.entry(b) {
            if let Some(ch) = char::from_u32(n) {
                e.insert(ch);
                n += 1;
            }
        }
    }

    mapping
}

/// Build the reverse GPT-2 unicode-to-byte mapping.
///
/// Returns: unicode character → byte value.
pub(super) fn build_byte_decoder() -> HashMap<char, u8> {
    build_byte_encoder()
        .into_iter()
        .map(|(byte, ch)| (ch, byte))
        .collect()
}
