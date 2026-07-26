// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for kokoro_pipeline step ordering and buffer-shape
//! propagation.
//!
//! These proofs complement the existing pipeline suites with issue #3732
//! checks that target:
//! - the ordered composition used by `text_to_tokens`
//! - propagation of padded token-buffer lengths into `[1, T]` synthesis tensors
//! - preservation of per-chunk shapes across multiple buffers

#[cfg(kani)]
mod proofs {
    use nn_core::DynTensor;

    use crate::kokoro_error::KokoroError;
    use crate::kokoro_g2p::EspeakRemapper;
    use crate::kokoro_pipeline::{chunks_to_tensors, KokoroSynth, KokoroTextPipeline};
    use crate::kokoro_text_preprocess::TextPreprocessor;
    use crate::kokoro_tokenizer::KokoroTokenizer;

    struct NoopSynth;

    impl KokoroSynth for NoopSynth {
        type Error = KokoroError;

        fn synthesize_chunk(
            &mut self,
            _input_ids: &DynTensor,
            _style: &DynTensor,
            _speed: f32,
        ) -> Result<Vec<f32>, Self::Error> {
            unreachable!("text_to_tokens proofs do not synthesize audio")
        }
    }

    /// `text_to_tokens` must behave like the explicit composition:
    /// preprocess -> phonemize -> remap -> tokenize.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn text_to_tokens_matches_manual_stage_composition() {
        let extra_whitespace: bool = kani::any();
        let ipa_len: u8 = kani::any();
        kani::assume(ipa_len >= 1 && ipa_len <= 4);

        let text = if extra_whitespace {
            "  hello world  "
        } else {
            "hello world"
        };
        let ipa = "a".repeat(ipa_len as usize);

        let pipeline = KokoroTextPipeline::new(
            TextPreprocessor::english(),
            EspeakRemapper::english_us(),
            KokoroTokenizer::kokoro_default(),
            NoopSynth,
        );

        let expected_cleaned = pipeline.preprocessor().preprocess(text);
        let expected_phonemes = pipeline.remapper().remap(&ipa);
        let expected_chunks = pipeline.tokenizer().chunk_and_encode(&expected_phonemes);

        let actual_chunks = pipeline
            .text_to_tokens(text, |cleaned| {
                assert_eq!(cleaned, expected_cleaned);
                Ok::<String, Box<dyn std::error::Error + Send + Sync>>(ipa.clone())
            })
            .unwrap();

        assert_eq!(actual_chunks, expected_chunks);
        assert!(!actual_chunks.is_empty());
    }

    /// A padded token buffer becomes a synthesis input tensor with shape
    /// `[1, encoded_len]`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn encoded_chunk_length_propagates_to_tensor_shape() {
        let content_len: u8 = kani::any();
        kani::assume(content_len >= 1 && content_len <= 16);

        let tokenizer = KokoroTokenizer::kokoro_default();
        let phonemes = "a".repeat(content_len as usize);
        let encoded = tokenizer.encode(&phonemes).unwrap();

        let tensors = chunks_to_tensors(&[(phonemes, encoded.clone())]).unwrap();
        let shape = tensors[0].shape();

        assert_eq!(tensors.len(), 1);
        assert_eq!(shape.dims(), &[1, encoded.len()]);
        assert_eq!(shape.dims()[1], content_len as usize + 2);
        assert_eq!(tensors[0].elem_count(), encoded.len());
    }

    /// Distinct chunk lengths must stay distinct after buffer-to-tensor
    /// conversion.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn per_chunk_buffer_shapes_propagate_independently() {
        let len_a: u8 = kani::any();
        let len_b: u8 = kani::any();
        kani::assume(len_a >= 1 && len_a <= 16);
        kani::assume(len_b >= 1 && len_b <= 16);
        kani::assume(len_a != len_b);

        let tokenizer = KokoroTokenizer::kokoro_default();
        let phonemes_a = "a".repeat(len_a as usize);
        let phonemes_b = "b".repeat(len_b as usize);
        let encoded_a = tokenizer.encode(&phonemes_a).unwrap();
        let encoded_b = tokenizer.encode(&phonemes_b).unwrap();

        let tensors = chunks_to_tensors(&[
            (phonemes_a, encoded_a.clone()),
            (phonemes_b, encoded_b.clone()),
        ])
        .unwrap();

        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].shape().dims(), &[1, encoded_a.len()]);
        assert_eq!(tensors[1].shape().dims(), &[1, encoded_b.len()]);
        assert_ne!(tensors[0].shape().dims()[1], tensors[1].shape().dims()[1]);
    }

    /// Across multiple chunks, the synthesis-input element counts add up to the
    /// sum of the source token-buffer lengths.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn total_tensor_elements_match_total_buffer_lengths() {
        let len_a: u8 = kani::any();
        let len_b: u8 = kani::any();
        kani::assume(len_a >= 1 && len_a <= 16);
        kani::assume(len_b >= 1 && len_b <= 16);

        let tokenizer = KokoroTokenizer::kokoro_default();
        let phonemes_a = "a".repeat(len_a as usize);
        let phonemes_b = "a".repeat(len_b as usize);
        let encoded_a = tokenizer.encode(&phonemes_a).unwrap();
        let encoded_b = tokenizer.encode(&phonemes_b).unwrap();
        let expected_total = encoded_a.len() + encoded_b.len();

        let tensors =
            chunks_to_tensors(&[(phonemes_a, encoded_a), (phonemes_b, encoded_b)]).unwrap();

        let total_tensor_elems: usize = tensors.iter().map(DynTensor::elem_count).sum();
        assert_eq!(total_tensor_elems, expected_total);
    }
}
