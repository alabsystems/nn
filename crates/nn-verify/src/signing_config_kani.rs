// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for `SigningKey` and hex decoding.

use super::*;

/// Prove: `zeroize()` sets ALL bytes to zero for any key length 1..=64.
///
/// The volatile writes in `zeroize()` are the critical security path.
/// This harness proves the post-condition (all bytes == 0) for any
/// non-deterministic initial byte values and any key length in [1, 64].
///
/// Kani unwind 66 = max 64 bytes + 1 for loop termination + 1 for
/// the outer match.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(66)]
fn zeroize_clears_all_bytes() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 64);

    // Create a key with non-deterministic content.
    let mut bytes = Vec::with_capacity(len);
    for _ in 0..len {
        bytes.push(kani::any::<u8>());
    }
    let mut key = SigningKey::Raw(bytes);

    // Zeroize.
    key.zeroize();

    // Post-condition: every byte is 0.
    if let SigningKey::Raw(ref b) = key {
        for i in 0..b.len() {
            assert!(b[i] == 0, "byte at index {} must be 0 after zeroize", i);
        }
    } else {
        panic!("zeroize must not change the variant from Raw");
    }
}

/// Prove: `zeroize()` on `SigningKey::None` is a no-op (no panic).
#[kani::unwind(1)]
#[kani::proof]
fn zeroize_none_is_noop() {
    let mut key = SigningKey::None;
    key.zeroize();
    assert!(key.is_none(), "None variant must remain None after zeroize");
}

/// Prove: `hex_digit` returns correct value for all valid hex chars,
/// and errors for all other u8 values.
///
/// This proves the bit manipulation in hex_decode is correct:
/// - '0'-'9' → 0-9
/// - 'a'-'f' → 10-15
/// - 'A'-'F' → 10-15
/// - everything else → error
#[kani::unwind(1)]
#[kani::proof]
fn hex_digit_correct_for_all_bytes() {
    let b: u8 = kani::any();
    let result = hex_digit(b);
    match b {
        b'0'..=b'9' => {
            let v = result.unwrap();
            assert!(v == b - b'0');
            assert!(v <= 9);
        }
        b'a'..=b'f' => {
            let v = result.unwrap();
            assert!(v == b - b'a' + 10);
            assert!(v >= 10 && v <= 15);
        }
        b'A'..=b'F' => {
            let v = result.unwrap();
            assert!(v == b - b'A' + 10);
            assert!(v >= 10 && v <= 15);
        }
        _ => {
            assert!(result.is_err());
        }
    }
}
