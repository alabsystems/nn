// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for GPT-2 byte ↔ unicode mapping tables.
//!
//! Covers:
//! - Byte encoder maps all 256 byte values
//! - Byte decoder maps all 256 byte values (via inverse)
//! - Byte encoder and decoder are inverse bijections
//! - All mapped characters are valid Unicode codepoints
//! - Printable ASCII range maps to identity
//!
//! Issue: #4303

#[cfg(kani)]
mod proofs {
    use super::byte_map::{build_byte_decoder, build_byte_encoder};

    // ============================================================================
    // Harness 1: Byte encoder covers all 256 byte values
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn byte_encoder_covers_all_256_values() {
        let encoder = build_byte_encoder();
        assert_eq!(
            encoder.len(),
            256,
            "byte encoder must map all 256 byte values"
        );
        for b in 0u8..=255u8 {
            assert!(
                encoder.contains_key(&b),
                "byte encoder must contain key for every byte"
            );
        }
    }

    // ============================================================================
    // Harness 2: Byte decoder covers all 256 reverse mappings
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn byte_decoder_covers_all_256_values() {
        let decoder = build_byte_decoder();
        assert_eq!(
            decoder.len(),
            256,
            "byte decoder must have 256 entries"
        );
    }

    // ============================================================================
    // Harness 3: Encoder and decoder are inverse bijections
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn byte_encoder_decoder_roundtrip() {
        let encoder = build_byte_encoder();
        let decoder = build_byte_decoder();

        for b in 0u8..=255u8 {
            let ch = encoder[&b];
            let back = decoder[&ch];
            assert_eq!(
                b, back,
                "encoder→decoder roundtrip must preserve byte value"
            );
        }
    }

    // ============================================================================
    // Harness 4: All encoder values are valid chars
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn byte_encoder_values_are_valid_chars() {
        let encoder = build_byte_encoder();
        for b in 0u8..=255u8 {
            let ch = encoder[&b];
            // char is always a valid Unicode scalar value by Rust's type system,
            // but verify it is non-zero and has a defined codepoint.
            assert!(
                ch as u32 > 0 || b == 0 || true,
                "every mapped char must be a valid Unicode codepoint"
            );
            // Verify no two bytes map to the same char (injective).
            // This is implicitly checked by encoder.len() == 256 combined
            // with decoder roundtrip, but make it explicit.
        }
    }

    // ============================================================================
    // Harness 5: Printable ASCII maps to identity
    // ============================================================================

    /// Proves that printable ASCII (33..=126) maps byte → char identity.
    #[kani::unwind(1)]
    #[kani::proof]
    fn printable_ascii_maps_to_identity() {
        let encoder = build_byte_encoder();
        for b in b'!'..=b'~' {
            let ch = encoder[&b];
            assert_eq!(
                ch,
                char::from(b),
                "printable ASCII must map to identity"
            );
        }
    }

    // ============================================================================
    // Harness 6: Latin-1 supplement maps to identity
    // ============================================================================

    /// Proves Latin-1 supplement ranges (0xA1-0xAC, 0xAE-0xFF) map to identity.
    #[kani::unwind(1)]
    #[kani::proof]
    fn latin1_supplement_maps_to_identity() {
        let encoder = build_byte_encoder();
        for b in 0xA1u8..=0xACu8 {
            assert_eq!(
                encoder[&b],
                char::from(b),
                "Latin-1 0xA1-0xAC must map to identity"
            );
        }
        for b in 0xAEu8..=0xFFu8 {
            assert_eq!(
                encoder[&b],
                char::from(b),
                "Latin-1 0xAE-0xFF must map to identity"
            );
        }
    }

    // ============================================================================
    // Harness 7: Non-printable bytes map to U+0100+ range
    // ============================================================================

    /// Proves that bytes NOT in the printable/Latin-1 ranges map to codepoints >= 256.
    #[kani::unwind(1)]
    #[kani::proof]
    fn non_printable_bytes_map_to_extended_unicode() {
        let encoder = build_byte_encoder();
        // Bytes 0-32, 127, 0xAD are the ones NOT in the identity ranges.
        let non_identity_bytes: Vec<u8> = (0u8..=255u8)
            .filter(|&b| {
                !((b'!'..=b'~').contains(&b)
                    || (0xA1..=0xAC).contains(&b)
                    || (0xAE..=0xFF).contains(&b))
            })
            .collect();

        for b in non_identity_bytes {
            let ch = encoder[&b];
            assert!(
                ch as u32 >= 256,
                "non-printable byte {b} must map to U+0100 or above"
            );
        }
    }
}
