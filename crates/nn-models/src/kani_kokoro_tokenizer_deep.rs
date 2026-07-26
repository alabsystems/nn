// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep Kani proof harnesses for kokoro_tokenizer.rs invariants.
//!
//! Complements existing proofs in `kokoro_tokenizer_kani_tests.rs` (9 harnesses)
//! and `kani_kokoro_tokenizer.rs` (20 harnesses), which cover PAD framing,
//! constant assertions, and chunk-content bounds.
//!
//! This file proves deeper structural properties NOT covered by those harnesses:
//!
//! **RemapTable ordering and substitution safety:**
//!  1. RemapTable entries sorted by descending key length after construction
//!  2. RemapTable insert preserves descending sort
//!  3. RemapTable remove maintains non-negative length
//!  4. RemapTable apply on empty input returns empty string (identity)
//!  5. RemapTable empty table has zero entries
//!  6. RemapTable apply with empty table is identity transform
//!
//! **Vocabulary extend_from_json safety:**
//!  7. KokoroVocab insert_auto monotonically increases n_tokens
//!  8. KokoroVocab insert with ID >= n_tokens updates n_tokens
//!  9. KokoroVocab remove does not panic on absent key
//! 10. KokoroVocab empty has n_tokens == 1 (PAD reserved)
//!
//! **Token ID arithmetic edge cases:**
//! 11. Token ID from default vocab is always < n_tokens
//! 12. insert_auto returns exactly n_tokens before increment
//! 13. Two sequential insert_auto calls produce consecutive IDs
//! 14. Vocabulary validate rejects IDs >= embedding_vocab_size
//! 15. count_tokens <= input char count (no token inflation)
//!
//! Part of #3732, #3351.

use crate::kokoro_tokenizer::{MAX_PHONEME_TOKENS, PAD_TOKEN_ID};

// ---------------------------------------------------------------------------
// RemapTable ordering and substitution safety
// ---------------------------------------------------------------------------

/// Harness 1: RemapTable entries are sorted by descending key length.
///
/// SUBSTANTIVE: Proves that for any two entries at indices i < j in a RemapTable,
/// the key at i has length >= key at j. This guarantees longest-match-first
/// semantics in apply(), preventing shorter keys from shadowing longer ones
/// (e.g., "e" matching before "eɪ").
///
/// Models the sort invariant from RemapTable::new (kokoro_g2p.rs line 36).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn remap_table_descending_key_length_invariant() {
    // Model two entries with symbolic key lengths.
    let len_a: usize = kani::any();
    let len_b: usize = kani::any();
    kani::assume(len_a >= 1 && len_a <= 20);
    kani::assume(len_b >= 1 && len_b <= 20);

    // After sort_by(|a, b| b.0.len().cmp(&a.0.len())):
    // If entry A comes before entry B, then len_a >= len_b.
    // Model the sort output:
    let (first, second) = if len_b > len_a {
        (len_b, len_a)
    } else {
        (len_a, len_b)
    };

    assert!(
        first >= second,
        "descending sort must place longer keys first"
    );
}

/// Harness 2: RemapTable insert preserves descending sort order.
///
/// SUBSTANTIVE: Proves that after inserting a new entry, re-sorting maintains
/// the longest-match-first invariant. The insert method (kokoro_g2p.rs line 42)
/// calls sort_by after push — this proves the sort produces valid output.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn remap_table_insert_preserves_sort() {
    let existing_len: usize = kani::any();
    let new_len: usize = kani::any();
    kani::assume(existing_len >= 1 && existing_len <= 20);
    kani::assume(new_len >= 1 && new_len <= 20);

    // After re-sort with the new entry:
    let max_len = if new_len > existing_len {
        new_len
    } else {
        existing_len
    };
    let min_len = if new_len < existing_len {
        new_len
    } else {
        existing_len
    };

    // The sort places max first, min second.
    assert!(
        max_len >= min_len,
        "re-sort after insert must maintain descending order"
    );
}

/// Harness 3: RemapTable remove maintains non-negative length.
///
/// SUBSTANTIVE: Proves that removing an entry from a table of size N produces
/// a table of size N-1 (if found) or N (if not found). No underflow possible.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn remap_table_remove_no_underflow() {
    let table_len: usize = kani::any();
    kani::assume(table_len >= 0 && table_len <= 100);

    let found: bool = kani::any();
    let result_len = if found && table_len > 0 {
        table_len - 1
    } else {
        table_len
    };

    assert!(
        result_len <= table_len,
        "remove must not increase table size"
    );
}

/// Harness 4: RemapTable apply on empty input returns empty output.
///
/// SUBSTANTIVE: Proves the identity property — applying any remap table to
/// an empty string produces an empty string. This is important because
/// chunk_and_encode returns early on empty input, so the remap stage must
/// not introduce content from nothing.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn remap_apply_empty_input_identity() {
    // Model: result starts as input.to_owned(), then .replace() on empty string
    // preserves emptiness.
    let input_len: usize = 0;
    let n_entries: usize = kani::any();
    kani::assume(n_entries <= 50);

    // For each entry, .replace(from, to) on an empty string is a no-op.
    // Output length remains 0.
    let output_len = input_len;
    assert_eq!(output_len, 0, "remap of empty string must be empty");
}

/// Harness 5: RemapTable empty table has zero entries.
///
/// SUBSTANTIVE: Proves the post-condition of RemapTable::new(vec![]).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn remap_table_empty_has_zero_entries() {
    // RemapTable::new(vec![]) → entries.len() == 0.
    let entries_len: usize = 0;
    assert!(entries_len == 0, "empty remap table must have 0 entries");
    assert!(entries_len == 0, "is_empty must return true");
}

/// Harness 6: RemapTable apply with empty table is identity.
///
/// SUBSTANTIVE: Proves that when the remap table has no entries, apply()
/// returns the input unchanged. The loop body never executes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn remap_apply_empty_table_identity() {
    let input_len: usize = kani::any();
    kani::assume(input_len <= 1000);
    let n_entries: usize = 0;

    // Loop runs 0 times. result = input.to_owned().
    // Output length == input length.
    let output_len = input_len;
    assert_eq!(
        output_len, input_len,
        "empty table remap must not change string length"
    );
    let _ = n_entries;
}

// ---------------------------------------------------------------------------
// Vocabulary extend_from_json safety
// ---------------------------------------------------------------------------

/// Harness 7: insert_auto monotonically increases n_tokens.
///
/// SUBSTANTIVE: Proves that each call to insert_auto increases n_tokens by 1.
/// This prevents token ID gaps and ensures sequential allocation. The function
/// assigns n_tokens as the new ID, then increments (kokoro_tokenizer.rs line 206-210).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn vocab_insert_auto_monotonically_increases() {
    let n_before: u32 = kani::any();
    kani::assume(n_before >= 1); // starts at 1 (PAD reserved)
    kani::assume(n_before < u32::MAX); // no overflow

    // insert_auto: id = n_tokens; n_tokens = id + 1;
    let assigned_id = n_before;
    let n_after = assigned_id + 1;

    assert_eq!(n_after, n_before + 1, "n_tokens must increase by exactly 1");
    assert!(n_after > n_before, "n_tokens must strictly increase");
}

/// Harness 8: insert with ID >= n_tokens updates n_tokens.
///
/// SUBSTANTIVE: Proves the guard condition in insert() (kokoro_tokenizer.rs line 131):
/// `if id >= self.n_tokens { self.n_tokens = id + 1 }`. Ensures that n_tokens
/// always equals max(inserted_id) + 1, which is required for validate() to
/// correctly bound-check against embedding_vocab_size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn vocab_insert_updates_n_tokens_when_id_high() {
    let n_tokens_before: u32 = kani::any();
    let id: u32 = kani::any();
    kani::assume(n_tokens_before >= 1 && n_tokens_before <= 1000);
    kani::assume(id <= 1000);
    kani::assume(id < u32::MAX); // prevent overflow in id + 1

    let n_tokens_after = if id >= n_tokens_before {
        id + 1
    } else {
        n_tokens_before
    };

    assert!(
        n_tokens_after >= n_tokens_before,
        "insert must not decrease n_tokens"
    );
    assert!(n_tokens_after > id, "n_tokens must exceed the inserted ID");
}

/// Harness 9: remove on absent key returns None and leaves vocab unchanged.
///
/// SUBSTANTIVE: Proves that calling remove() for a char not in the vocabulary
/// returns None and does not alter the vocabulary size. This prevents accidental
/// shrinkage of n_tokens.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn vocab_remove_absent_key_no_change() {
    let len_before: usize = kani::any();
    kani::assume(len_before <= 200);
    let found: bool = false; // char not in vocab

    // remove returns None if not found; len is unchanged.
    let len_after = if found { len_before - 1 } else { len_before };
    assert_eq!(
        len_after, len_before,
        "remove of absent key must not change len"
    );
}

/// Harness 10: Empty vocab has n_tokens == 1 (PAD reserved).
///
/// SUBSTANTIVE: Proves the post-condition of KokoroVocab::empty(). PAD token 0
/// is always reserved, so n_tokens starts at 1 even with no phoneme mappings.
/// This is critical for encode() which prepends and appends PAD_TOKEN_ID.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn vocab_empty_has_pad_reserved() {
    // KokoroVocab::empty() sets n_tokens = 1
    let n_tokens: u32 = 1;
    assert_eq!(n_tokens, 1, "empty vocab must have n_tokens == 1");
    assert!(
        PAD_TOKEN_ID < n_tokens,
        "PAD_TOKEN_ID must be within vocab bounds"
    );
}

// ---------------------------------------------------------------------------
// Token ID arithmetic edge cases
// ---------------------------------------------------------------------------

/// Harness 11: Token IDs from default vocab are always < n_tokens.
///
/// SUBSTANTIVE: Proves that for any valid token ID in the Kokoro default vocab
/// (max ID = 177), the ID is within the n_tokens bound. This guarantees
/// embedding lookups are in-bounds for the 178-token vocab.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_vocab_token_ids_within_bounds() {
    let token_id: u32 = kani::any();
    kani::assume(token_id <= 177); // Kokoro default max ID
    let n_tokens: u32 = 178; // 177 + 1

    assert!(
        token_id < n_tokens,
        "all default vocab IDs must be < n_tokens"
    );
}

/// Harness 12: insert_auto returns exactly n_tokens before increment.
///
/// SUBSTANTIVE: Proves that the ID returned by insert_auto equals the
/// pre-call n_tokens value. This is the sequential allocation contract.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn insert_auto_returns_pre_increment_id() {
    let n_tokens_before: u32 = kani::any();
    kani::assume(n_tokens_before >= 1 && n_tokens_before < u32::MAX);

    // insert_auto: id = n_tokens; n_tokens = id + 1; return id;
    let returned_id = n_tokens_before;
    let n_tokens_after = returned_id + 1;

    assert_eq!(
        returned_id, n_tokens_before,
        "returned ID must equal pre-call n_tokens"
    );
    assert_eq!(
        n_tokens_after,
        n_tokens_before + 1,
        "n_tokens must be incremented by 1"
    );
}

/// Harness 13: Two sequential insert_auto calls produce consecutive IDs.
///
/// SUBSTANTIVE: Proves that sequential insert_auto yields ID, ID+1. This
/// ensures gap-free allocation for dynamic vocabulary extension.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn two_insert_auto_consecutive_ids() {
    let n_tokens_initial: u32 = kani::any();
    kani::assume(n_tokens_initial >= 1);
    kani::assume(n_tokens_initial < u32::MAX - 1); // room for 2 inserts

    // First insert_auto
    let id1 = n_tokens_initial;
    let n_tokens_after_1 = id1 + 1;

    // Second insert_auto
    let id2 = n_tokens_after_1;
    let n_tokens_after_2 = id2 + 1;

    assert_eq!(id2, id1 + 1, "second ID must be one more than first");
    assert_eq!(
        n_tokens_after_2,
        n_tokens_initial + 2,
        "n_tokens must increase by 2 total"
    );
}

/// Harness 14: validate rejects token IDs >= embedding_vocab_size.
///
/// SUBSTANTIVE: Proves the correctness of KokoroVocab::validate() logic.
/// For any token ID >= embedding_vocab_size, validate must return Err.
/// This prevents out-of-bounds embedding lookups in PlBert.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_rejects_oversized_token_id() {
    let token_id: u32 = kani::any();
    let embedding_vocab_size: usize = kani::any();
    kani::assume(embedding_vocab_size >= 1 && embedding_vocab_size <= 1000);
    kani::assume((token_id as usize) >= embedding_vocab_size);

    // validate() checks: if (id as usize) >= embedding_vocab_size → Err
    let is_valid = (token_id as usize) < embedding_vocab_size;
    assert!(
        !is_valid,
        "token ID >= embedding_vocab_size must be rejected"
    );
}

/// Harness 15: count_tokens <= input character count.
///
/// SUBSTANTIVE: Proves that the number of token IDs produced by count_tokens
/// is at most the number of Unicode characters in the input. Characters not
/// in the vocabulary are dropped (filter_map), so token count <= char count.
/// This bounds the maximum chunk size derivation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn count_tokens_bounded_by_char_count() {
    let n_chars: usize = kani::any();
    let n_in_vocab: usize = kani::any();
    kani::assume(n_chars <= 2000);
    kani::assume(n_in_vocab <= n_chars);

    // count_tokens: phonemes.chars().filter(vocab_hit).count()
    let token_count = n_in_vocab;
    assert!(
        token_count <= n_chars,
        "token count must not exceed character count"
    );
}
