// Andrew Yates
// Copyright 2026 Andrew Yates Apache 2.0

use super::*;
use nn_core::{Device, DynTensor};

/// Mock synthesizer that returns a fixed PCM buffer (sine wave).
struct MockSynth {
    sample_rate: usize,
    call_count: usize,
}

impl MockSynth {
    fn new() -> Self {
        Self {
            sample_rate: 24000,
            call_count: 0,
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("mock synth error: {0}")]
struct MockSynthError(String);

impl KokoroSynth for MockSynth {
    type Error = MockSynthError;

    fn synthesize_chunk(
        &mut self,
        input_ids: &DynTensor,
        _style: &DynTensor,
        _speed: f32,
    ) -> Result<Vec<f32>, MockSynthError> {
        self.call_count += 1;
        // Generate ~0.1s of audio per token (proportional to input length)
        let seq_len = input_ids.dims()[1];
        let num_samples = seq_len * (self.sample_rate / 10);
        let pcm: Vec<f32> = (0..num_samples)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / self.sample_rate as f32).sin())
            .collect();
        Ok(pcm)
    }
}

fn mock_phonemize(text: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Simple mock: pass through text as "phonemes"
    Ok(text.to_string())
}

fn dummy_style() -> DynTensor {
    DynTensor::from_vec(vec![0.0f32; 256], &[1, 256], &Device::Cpu).unwrap()
}

#[test]
fn test_pipeline_construction_english() {
    let synth = MockSynth::new();
    let pipeline = KokoroTextPipeline::english(synth);
    assert_eq!(pipeline.tokenizer().max_tokens(), 510);
}

#[test]
fn test_pipeline_construction_custom() {
    let synth = MockSynth::new();
    let pipeline = KokoroTextPipeline::new(
        TextPreprocessor::minimal(),
        EspeakRemapper::english_us(),
        KokoroTokenizer::kokoro_default(),
        synth,
    );
    assert_eq!(pipeline.tokenizer().max_tokens(), 510);
}

#[test]
fn test_text_to_tokens_simple() {
    let synth = MockSynth::new();
    let pipeline = KokoroTextPipeline::english(synth);

    // Use a simple phonemize that returns Kokoro-compatible characters.
    let tokens = pipeline
        .text_to_tokens("hello", |_| Ok("hɛloʊ".to_string()))
        .unwrap();

    assert!(!tokens.is_empty());
    // Each chunk should have (phoneme_text, token_ids)
    for (text, ids) in &tokens {
        assert!(!text.is_empty());
        assert!(!ids.is_empty());
        // Token IDs should be padded: [0, ...ids, 0]
        assert_eq!(ids[0], 0, "first token should be PAD");
        assert_eq!(ids[ids.len() - 1], 0, "last token should be PAD");
    }
}

#[test]
fn test_text_to_audio_produces_chunks() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let style = dummy_style();
    let config = KokoroStreamConfig::default();

    let chunks = pipeline
        .text_to_audio("hello world", mock_phonemize, &style, 1.0, &config)
        .unwrap();

    assert!(!chunks.is_empty());
    // Each chunk should have PCM data
    for chunk in &chunks {
        assert!(!chunk.pcm.is_empty());
    }
    // Last chunk should be marked final
    assert!(chunks.last().unwrap().is_final);
}

#[test]
fn test_text_to_audio_calls_synth() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let style = dummy_style();
    let config = KokoroStreamConfig::default();

    let _chunks = pipeline
        .text_to_audio("hello", mock_phonemize, &style, 1.0, &config)
        .unwrap();

    // Synth should have been called at least once
    assert!(pipeline.synth().call_count > 0);
}

#[test]
fn test_phonemize_error_propagates() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let style = dummy_style();
    let config = KokoroStreamConfig::default();

    let result = pipeline.text_to_audio(
        "hello",
        |_| Err("espeak crashed".into()),
        &style,
        1.0,
        &config,
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, PipelineError::Phonemization(_)),
        "expected Phonemization error, got: {err}"
    );
}

#[test]
fn test_chunks_to_tensors_shape() {
    let chunks = vec![
        ("hello".to_string(), vec![0u32, 5, 10, 0]),
        ("world".to_string(), vec![0u32, 3, 7, 12, 0]),
    ];
    let tensors = chunks_to_tensors(&chunks).unwrap();
    assert_eq!(tensors.len(), 2);
    assert_eq!(tensors[0].dims(), &[1, 4]); // [1, 4] for 4 tokens
    assert_eq!(tensors[1].dims(), &[1, 5]); // [1, 5] for 5 tokens
}

#[test]
fn test_chunks_to_tensors_empty() {
    let chunks: Vec<(String, Vec<u32>)> = vec![];
    let tensors = chunks_to_tensors(&chunks).unwrap();
    assert!(tensors.is_empty());
}

#[test]
fn test_pipeline_accessor_methods() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);

    // Test immutable accessors
    let _ = pipeline.synth();
    let _ = pipeline.tokenizer();
    let _ = pipeline.preprocessor();
    let _ = pipeline.remapper();

    // Test mutable accessor
    let synth = pipeline.synth_mut();
    synth.call_count = 42;
    assert_eq!(pipeline.synth().call_count, 42);
}

// -- Chorus tests ---------------------------------------------------------

#[test]
fn test_text_to_chorus_two_voices() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let styles = vec![dummy_style(), dummy_style()];
    let speeds = vec![1.0, 1.05];
    let chorus_config = ChorusConfig::equal_gain(2).unwrap();
    let stream_config = KokoroStreamConfig::default();

    let chunks = pipeline
        .text_to_chorus(
            "hello world",
            mock_phonemize,
            &styles,
            &speeds,
            &chorus_config,
            &stream_config,
        )
        .unwrap();

    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert!(!chunk.pcm.is_empty());
    }
    assert!(chunks.last().unwrap().is_final);
    // 2 voices × (at least 1 chunk) = at least 2 synth calls
    assert!(pipeline.synth().call_count >= 2);
}

#[test]
fn test_text_to_chorus_call_count() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let n_voices = 4;
    let styles: Vec<DynTensor> = (0..n_voices).map(|_| dummy_style()).collect();
    let speeds: Vec<f32> = vec![1.0; n_voices];
    let chorus_config = ChorusConfig::equal_gain(n_voices).unwrap();
    let stream_config = KokoroStreamConfig::default();

    let _chunks = pipeline
        .text_to_chorus(
            "hello",
            mock_phonemize,
            &styles,
            &speeds,
            &chorus_config,
            &stream_config,
        )
        .unwrap();

    // Should be exactly n_voices × n_chunks calls.
    // "hello" → 1 token chunk → 4 voices × 1 chunk = 4 calls.
    assert_eq!(pipeline.synth().call_count, n_voices);
}

#[test]
fn test_text_to_chorus_style_count_mismatch() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let styles = vec![dummy_style()]; // 1 style but 3 voices
    let speeds = vec![1.0, 1.0, 1.0];
    let chorus_config = ChorusConfig::equal_gain(3).unwrap();
    let stream_config = KokoroStreamConfig::default();

    let result = pipeline.text_to_chorus(
        "hello",
        mock_phonemize,
        &styles,
        &speeds,
        &chorus_config,
        &stream_config,
    );

    assert!(result.is_err());
}

#[test]
fn test_text_to_chorus_speed_count_mismatch() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let styles = vec![dummy_style(), dummy_style()];
    let speeds = vec![1.0]; // 1 speed but 2 voices
    let chorus_config = ChorusConfig::equal_gain(2).unwrap();
    let stream_config = KokoroStreamConfig::default();

    let result = pipeline.text_to_chorus(
        "hello",
        mock_phonemize,
        &styles,
        &speeds,
        &chorus_config,
        &stream_config,
    );

    assert!(result.is_err());
}

// -- text_to_audio_streaming tests ----------------------------------------

#[test]
fn test_text_to_audio_streaming_callback_count() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let style = dummy_style();
    let config = KokoroStreamConfig::default();

    let mut callback_count = 0usize;
    let chunks = pipeline
        .text_to_audio_streaming(
            "hello world",
            mock_phonemize,
            &style,
            1.0,
            &config,
            |_chunk| {
                callback_count += 1;
            },
        )
        .unwrap();

    // Callback called once per chunk, matching returned Vec length.
    assert_eq!(callback_count, chunks.len());
    assert!(!chunks.is_empty());
}

#[test]
fn test_text_to_audio_streaming_matches_batch() {
    // Streaming and batch assembly should produce identical output.
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let style = dummy_style();
    let config = KokoroStreamConfig::default();

    let batch_chunks = pipeline
        .text_to_audio("hello world", mock_phonemize, &style, 1.0, &config)
        .unwrap();

    // Reset synth call count (via mutable access).
    pipeline.synth_mut().call_count = 0;

    let streaming_chunks = pipeline
        .text_to_audio_streaming("hello world", mock_phonemize, &style, 1.0, &config, |_| {})
        .unwrap();

    assert_eq!(batch_chunks.len(), streaming_chunks.len());
    for (b, s) in batch_chunks.iter().zip(streaming_chunks.iter()) {
        assert_eq!(b.pcm.len(), s.pcm.len(), "chunk PCM lengths should match");
        assert_eq!(b.is_final, s.is_final);
        assert_eq!(b.channels, s.channels);
        // PCM values should be identical (same mock synth, same crossfade).
        for (bv, sv) in b.pcm.iter().zip(s.pcm.iter()) {
            assert!(
                (bv - sv).abs() < 1e-6,
                "PCM mismatch: batch={bv}, streaming={sv}"
            );
        }
    }
}

#[test]
fn test_text_to_audio_streaming_last_chunk_is_final() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let style = dummy_style();
    let config = KokoroStreamConfig::default();

    let mut saw_final = false;
    let chunks = pipeline
        .text_to_audio_streaming("hello", mock_phonemize, &style, 1.0, &config, |chunk| {
            if chunk.is_final {
                saw_final = true;
            }
        })
        .unwrap();

    assert!(saw_final, "callback should see is_final on last chunk");
    assert!(chunks.last().unwrap().is_final);
}

#[test]
fn test_text_to_audio_streaming_phonemize_error() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let style = dummy_style();
    let config = KokoroStreamConfig::default();

    let result = pipeline.text_to_audio_streaming(
        "hello",
        |_| Err("espeak crashed".into()),
        &style,
        1.0,
        &config,
        |_| {},
    );

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        PipelineError::Phonemization(_)
    ));
}

// -- synthesize_batch tests ------------------------------------------------

/// Mock that tracks both individual and batch call patterns.
struct BatchTrackingSynth {
    chunk_calls: Vec<(usize, f32)>, // (seq_len, speed) per synthesize_chunk call
    sample_rate: usize,
}

impl BatchTrackingSynth {
    fn new() -> Self {
        Self {
            chunk_calls: Vec::new(),
            sample_rate: 24000,
        }
    }
}

impl KokoroSynth for BatchTrackingSynth {
    type Error = MockSynthError;

    fn synthesize_chunk(
        &mut self,
        input_ids: &DynTensor,
        _style: &DynTensor,
        speed: f32,
    ) -> Result<Vec<f32>, MockSynthError> {
        let seq_len = input_ids.dims()[1];
        self.chunk_calls.push((seq_len, speed));
        let num_samples = seq_len * (self.sample_rate / 10);
        Ok(vec![0.0f32; num_samples])
    }
}

/// Mock that overrides synthesize_batch to track batch calls.
struct BatchOverrideSynth {
    chunk_calls: usize,
    batch_calls: usize,
    sample_rate: usize,
}

impl BatchOverrideSynth {
    fn new() -> Self {
        Self {
            chunk_calls: 0,
            batch_calls: 0,
            sample_rate: 24000,
        }
    }
}

impl KokoroSynth for BatchOverrideSynth {
    type Error = MockSynthError;

    fn synthesize_chunk(
        &mut self,
        input_ids: &DynTensor,
        _style: &DynTensor,
        _speed: f32,
    ) -> Result<Vec<f32>, MockSynthError> {
        self.chunk_calls += 1;
        let seq_len = input_ids.dims()[1];
        Ok(vec![0.0f32; seq_len * (self.sample_rate / 10)])
    }

    fn synthesize_batch(
        &mut self,
        input_ids: &DynTensor,
        styles: &[DynTensor],
        _speeds: &[f32],
    ) -> Result<Vec<Vec<f32>>, MockSynthError> {
        self.batch_calls += 1;
        let seq_len = input_ids.dims()[1];
        let num_samples = seq_len * (self.sample_rate / 10);
        Ok((0..styles.len())
            .map(|_| vec![0.0f32; num_samples])
            .collect())
    }
}

#[test]
fn test_synthesize_batch_default_delegates_to_chunk() {
    let mut synth = BatchTrackingSynth::new();
    let input = DynTensor::from_vec_u32(vec![0, 5, 10, 0], &[1, 4], &Device::Cpu).unwrap();
    let styles = vec![dummy_style(), dummy_style(), dummy_style()];
    let speeds = vec![1.0, 1.1, 0.9];

    let result = synth.synthesize_batch(&input, &styles, &speeds).unwrap();

    // Default impl should call synthesize_chunk 3 times (one per voice).
    assert_eq!(synth.chunk_calls.len(), 3);
    assert_eq!(result.len(), 3);
    // Verify speeds were passed through correctly.
    assert!((synth.chunk_calls[0].1 - 1.0).abs() < 1e-6);
    assert!((synth.chunk_calls[1].1 - 1.1).abs() < 1e-6);
    assert!((synth.chunk_calls[2].1 - 0.9).abs() < 1e-6);
}

#[test]
fn test_chorus_uses_synthesize_batch_override() {
    let synth = BatchOverrideSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let styles = vec![dummy_style(), dummy_style()];
    let speeds = vec![1.0, 1.05];
    let chorus_config = ChorusConfig::equal_gain(2).unwrap();
    let stream_config = KokoroStreamConfig::default();

    let _chunks = pipeline
        .text_to_chorus(
            "hello",
            mock_phonemize,
            &styles,
            &speeds,
            &chorus_config,
            &stream_config,
        )
        .unwrap();

    // Chorus should call synthesize_batch (not synthesize_chunk).
    assert!(
        pipeline.synth().batch_calls > 0,
        "should use synthesize_batch"
    );
    assert_eq!(
        pipeline.synth().chunk_calls,
        0,
        "should not fall through to synthesize_chunk"
    );
}

#[test]
fn test_chorus_chunk_major_order() {
    // Verify chunk-major ordering: for N voices and C chunks, we get
    // C batch calls (one per chunk), not N×C individual calls.
    let synth = BatchOverrideSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let n_voices = 3;
    let styles: Vec<DynTensor> = (0..n_voices).map(|_| dummy_style()).collect();
    let speeds = vec![1.0; n_voices];
    let chorus_config = ChorusConfig::equal_gain(n_voices).unwrap();
    let stream_config = KokoroStreamConfig::default();

    let _chunks = pipeline
        .text_to_chorus(
            "hello",
            mock_phonemize,
            &styles,
            &speeds,
            &chorus_config,
            &stream_config,
        )
        .unwrap();

    // "hello" → 1 token chunk → exactly 1 batch call.
    assert_eq!(pipeline.synth().batch_calls, 1);
    assert_eq!(pipeline.synth().chunk_calls, 0);
}

// -- VoicePack convenience tests -------------------------------------------

use crate::kokoro_voice_pack::VoicePack;
use std::collections::HashMap;

fn make_voice_pack() -> VoicePack {
    let mut tensors = HashMap::new();
    tensors.insert(
        "af_heart".to_string(),
        DynTensor::from_vec(vec![0.1f32; 256], &[256], &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "am_adam".to_string(),
        DynTensor::from_vec(vec![0.2f32; 256], &[256], &Device::Cpu).unwrap(),
    );
    VoicePack::from_tensors(tensors, 128).unwrap()
}

#[test]
fn test_text_to_audio_named() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let pack = make_voice_pack();
    let config = KokoroStreamConfig::default();

    let chunks = pipeline
        .text_to_audio_named(
            "hello world",
            mock_phonemize,
            "af_heart",
            &pack,
            1.0,
            &config,
        )
        .unwrap();

    assert!(!chunks.is_empty());
    assert!(chunks.last().unwrap().is_final);
}

#[test]
fn test_text_to_audio_named_missing_voice() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let pack = make_voice_pack();
    let config = KokoroStreamConfig::default();

    let result =
        pipeline.text_to_audio_named("hello", mock_phonemize, "nonexistent", &pack, 1.0, &config);

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("nonexistent"),
        "error should name the voice: {msg}"
    );
}

#[test]
fn test_text_to_chorus_named() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let pack = make_voice_pack();
    let chorus_config = ChorusConfig::equal_gain(2).unwrap();
    let stream_config = KokoroStreamConfig::default();

    let chunks = pipeline
        .text_to_chorus_named(
            "hello world",
            mock_phonemize,
            &["af_heart", "am_adam"],
            &pack,
            &[1.0, 1.05],
            &chorus_config,
            &stream_config,
        )
        .unwrap();

    assert!(!chunks.is_empty());
    assert!(chunks.last().unwrap().is_final);
    assert!(pipeline.synth().call_count >= 2);
}

#[test]
fn test_text_to_chorus_named_missing_voice() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let pack = make_voice_pack();
    let chorus_config = ChorusConfig::equal_gain(2).unwrap();
    let stream_config = KokoroStreamConfig::default();

    let result = pipeline.text_to_chorus_named(
        "hello",
        mock_phonemize,
        &["af_heart", "missing_voice"],
        &pack,
        &[1.0, 1.0],
        &chorus_config,
        &stream_config,
    );

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("missing_voice"),
        "error should name the voice: {msg}"
    );
}

// -- text_to_chorus_streaming tests ----------------------------------------

#[test]
fn test_text_to_chorus_streaming_callback_count() {
    let synth = BatchOverrideSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let n_voices = 3;
    let styles: Vec<DynTensor> = (0..n_voices).map(|_| dummy_style()).collect();
    let speeds = vec![1.0; n_voices];
    let chorus_config = ChorusConfig::equal_gain(n_voices).unwrap();
    let stream_config = KokoroStreamConfig::default();

    let mut callback_count = 0usize;
    let chunks = pipeline
        .text_to_chorus_streaming(
            "hello world",
            mock_phonemize,
            &styles,
            &speeds,
            &chorus_config,
            &stream_config,
            |_chunk| {
                callback_count += 1;
            },
        )
        .unwrap();

    // Callback fires once per text chunk, not once per voice.
    assert_eq!(callback_count, chunks.len());
    assert!(!chunks.is_empty());
    // Should use synthesize_batch, not synthesize_chunk.
    assert!(pipeline.synth().batch_calls > 0);
    assert_eq!(pipeline.synth().chunk_calls, 0);
}

#[test]
fn test_text_to_chorus_streaming_style_count_mismatch() {
    let synth = MockSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let styles = vec![dummy_style()]; // 1 style but 3 voices
    let speeds = vec![1.0, 1.0, 1.0];
    let chorus_config = ChorusConfig::equal_gain(3).unwrap();
    let stream_config = KokoroStreamConfig::default();

    let result = pipeline.text_to_chorus_streaming(
        "hello",
        mock_phonemize,
        &styles,
        &speeds,
        &chorus_config,
        &stream_config,
        |_| {},
    );

    assert!(result.is_err());
}

#[test]
fn test_text_to_chorus_streaming_last_chunk_is_final() {
    let synth = BatchOverrideSynth::new();
    let mut pipeline = KokoroTextPipeline::english(synth);
    let styles = vec![dummy_style(), dummy_style()];
    let speeds = vec![1.0, 1.05];
    let chorus_config = ChorusConfig::equal_gain(2).unwrap();
    let stream_config = KokoroStreamConfig::default();

    let mut saw_final = false;
    let chunks = pipeline
        .text_to_chorus_streaming(
            "hello",
            mock_phonemize,
            &styles,
            &speeds,
            &chorus_config,
            &stream_config,
            |chunk| {
                if chunk.is_final {
                    saw_final = true;
                }
            },
        )
        .unwrap();

    assert!(saw_final, "callback should see is_final on last chunk");
    assert!(chunks.last().unwrap().is_final);
}
