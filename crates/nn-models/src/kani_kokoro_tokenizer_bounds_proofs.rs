// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for kokoro_tokenizer token-bound and vocab-size checks.
//!
//! These proofs complement the existing tokenizer suites with direct checks on
//! the specific issue #3732 concerns:
//! - token IDs remain strictly below the embedding vocabulary size after
//!   validation
//! - `n_tokens()` stays one past the maximum inserted token ID
//! - the boundary case `id == embedding_vocab_size` is rejected
//! - validation is monotonic as the allowed embedding vocabulary grows

#[cfg(kani)]
mod proofs {
    use crate::kokoro_error::KokoroError;
    use crate::kokoro_tokenizer::{KokoroTokenizer, KokoroVocab, PAD_TOKEN_ID};

    /// After inserts, `n_tokens()` is always exactly one past the maximum
    /// inserted ID for non-padding token IDs.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn vocab_n_tokens_tracks_max_inserted_id() {
        let id_a: u8 = kani::any();
        let id_b: u8 = kani::any();
        kani::assume(id_a >= 1 && id_a <= 200);
        kani::assume(id_b >= 1 && id_b <= 200);

        let mut vocab = KokoroVocab::empty();
        vocab.insert('a', id_a as u32);
        vocab.insert('b', id_b as u32);

        let max_id = u32::from(id_a.max(id_b));
        assert_eq!(vocab.n_tokens(), max_id + 1);
        assert!(vocab.n_tokens() > max_id);
    }

    /// A validated vocabulary guarantees every encoded non-padding token stays
    /// below the embedding table bound.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn validated_vocab_keeps_encoded_ids_in_bounds() {
        let token_id: u8 = kani::any();
        let embedding_vocab_size: u16 = kani::any();
        kani::assume(token_id >= 1 && token_id <= 200);
        kani::assume(embedding_vocab_size >= 2 && embedding_vocab_size <= 256);
        kani::assume((token_id as usize) < embedding_vocab_size as usize);

        let mut vocab = KokoroVocab::empty();
        vocab.insert('a', token_id as u32);

        assert!(vocab.validate(embedding_vocab_size as usize).is_ok());

        let tokenizer =
            KokoroTokenizer::with_validated_vocab(vocab, embedding_vocab_size as usize).unwrap();
        let encoded = tokenizer.encode("a").unwrap();

        assert_eq!(encoded.len(), 3);
        assert_eq!(encoded[0], PAD_TOKEN_ID);
        assert_eq!(encoded[2], PAD_TOKEN_ID);
        assert_eq!(encoded[1], token_id as u32);
        assert!((encoded[1] as usize) < embedding_vocab_size as usize);
    }

    /// The boundary case `id == embedding_vocab_size` must be rejected because
    /// valid embedding indices are half-open: `[0, embedding_vocab_size)`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn validated_vocab_rejects_equal_boundary_id() {
        let embedding_vocab_size: u16 = kani::any();
        kani::assume(embedding_vocab_size >= 1 && embedding_vocab_size <= 256);

        let mut vocab = KokoroVocab::empty();
        vocab.insert('a', embedding_vocab_size as u32);

        let validate_result = vocab.validate(embedding_vocab_size as usize);
        assert!(matches!(
            validate_result,
            Err(KokoroError::InvalidConfig { field: "vocab", .. })
        ));

        let tokenizer_result =
            KokoroTokenizer::with_validated_vocab(vocab, embedding_vocab_size as usize);
        assert!(matches!(
            tokenizer_result,
            Err(KokoroError::InvalidConfig { field: "vocab", .. })
        ));
    }

    /// Once a vocabulary validates against a smaller embedding table, it also
    /// validates against any larger table.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn vocab_validation_is_monotonic_in_embedding_size() {
        let token_id: u8 = kani::any();
        let smaller_vocab_size: u16 = kani::any();
        let extra_capacity: u8 = kani::any();
        kani::assume(token_id >= 1 && token_id <= 200);
        kani::assume(smaller_vocab_size >= 2 && smaller_vocab_size <= 224);
        kani::assume((token_id as usize) < smaller_vocab_size as usize);
        kani::assume(extra_capacity <= 31);

        let larger_vocab_size = smaller_vocab_size as usize + extra_capacity as usize;

        let mut vocab = KokoroVocab::empty();
        vocab.insert('a', token_id as u32);

        assert!(vocab.validate(smaller_vocab_size as usize).is_ok());
        assert!(vocab.validate(larger_vocab_size).is_ok());
        assert!(larger_vocab_size >= smaller_vocab_size as usize);
    }
}
