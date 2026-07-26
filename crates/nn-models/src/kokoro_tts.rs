// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro-82M text-to-speech model.
//!
//! Speech synthesis pipeline (two parallel paths, dvoice v0.19 reference):
//! PlBert → bert_encoder → ProsodyPredictor → length_regulate → F0EnergyPredictor
//! TextEncoder(tokens) → length_regulate → FullDecoder(asr) + F0/energy → Generator → iSTFT → PCM.
//! See `designs/archive/2026-03-16-kokoro-architecture-correction.md` for architecture details.
//!
//! Submodules (extracted for 500-line compliance, #1342):
//! - [`KokoroConfig`] in `kokoro_config.rs`
//! - [`TextEncoder`] in `kokoro_text_encoder.rs`

use crate::kokoro_error::{check_tensor_finite, validate_speed, KokoroError};
use crate::kokoro_f0::F0EnergyPredictor;
use crate::kokoro_forward_stft::KokoroForwardStft;
use crate::kokoro_full_decoder::FullDecoder;
use crate::kokoro_source::SourceModule;
use crate::plbert::PlBert;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{check_output_finite, Linear, Module};
use nn_core::var_builder::VarBuilder;
use nn_core::{Device, Result, TensorError};

#[path = "kokoro_prosody.rs"]
mod prosody;
pub use prosody::{AdaLayerNorm, ProsodyPredictor};

#[path = "kokoro_config.rs"]
mod config;
pub use config::KokoroConfig;

#[path = "kokoro_text_encoder.rs"]
mod text_encoder;
pub use text_encoder::TextEncoder;

// -- TextPipelineResult -------------------------------------------------------

/// Named result from [`KokoroModel::forward_text()`].
///
/// Prevents silent parameter-swap bugs: all three fields are `DynTensor`
/// with similar shapes, so a tuple `(a, b, c)` is easy to destructure wrong.
#[derive(Debug)]
#[non_exhaustive]
pub struct TextPipelineResult {
    /// ProsodyPredictor features after length_regulate, `[B, d_model+style_dim, T_mel]`.
    /// Feeds into F0EnergyPredictor.
    pub aligned_dur: DynTensor,
    /// TextEncoder features after length_regulate, `[B, d_en, T_mel]`.
    /// Feeds into FullDecoder as ASR input.
    pub regulated: DynTensor,
    /// Raw duration logits from ProsodyPredictor, `[B, T, max_dur]`.
    pub dur_logits: DynTensor,
}

impl TextPipelineResult {
    /// Create a new text pipeline result.
    pub fn new(aligned_dur: DynTensor, regulated: DynTensor, dur_logits: DynTensor) -> Self {
        Self {
            aligned_dur,
            regulated,
            dur_logits,
        }
    }
}

// -- EncoderFeaturesResult ----------------------------------------------------

/// Result from [`KokoroModel::forward_encoder_features()`].
///
/// Contains the frozen encoder features needed for singing decoder LoRA
/// training data preparation (dvoice#2461). The `regulated` field is
/// the length-regulated TextEncoder output — the intermediate tensor
/// between the encoder and decoder in the Kokoro pipeline.
#[derive(Debug)]
#[non_exhaustive]
pub struct EncoderFeaturesResult {
    /// Length-regulated TextEncoder features, `[B, d_en, T_mel]`.
    ///
    /// For Kokoro-82M with default config: `[1, 512, T_mel]` where
    /// `T_mel = sum(round(durations))` depends on predicted phone durations.
    ///
    /// This is the encoder output that feeds into FullDecoder as ASR input.
    /// For singing training, this tensor is paired with F0 contours and
    /// target audio to train the decoder LoRA without re-running the encoder.
    pub regulated: DynTensor,

    /// Length-regulated ProsodyPredictor features, `[B, d_en+style_dim, T_mel]`.
    ///
    /// Feeds into F0EnergyPredictor. Included so callers can also extract
    /// F0/energy predictions from the frozen encoder if needed.
    pub aligned_dur: DynTensor,

    /// Predicted phone durations after sigmoid binning, `[B, T]`.
    ///
    /// Integer durations (as f32) used by length_regulate. Included for
    /// training data alignment (mapping phoneme indices to mel frames).
    pub durations: DynTensor,
}

impl EncoderFeaturesResult {
    /// Create a new encoder features result.
    pub fn new(regulated: DynTensor, aligned_dur: DynTensor, durations: DynTensor) -> Self {
        Self {
            regulated,
            aligned_dur,
            durations,
        }
    }
}

// -- length_regulate ----------------------------------------------------------

/// Expand features by predicted durations (like PyTorch's Kokoro length_regulate).
///
/// `features`: `[B, D, T]` — feature sequences.
/// `durations`: `[B, T]` — integer durations per frame (as f32).
/// Returns: `[B, D, T_mel]` where `T_mel = sum(durations)` per batch.
///
/// Uses traceable DynTensor ops so the operation can be captured by `trace_graph()`
/// and compiled via `CompiledStep::RuntimeOp` (Part of #2234). The repeat_interleave
/// with data-dependent counts emits `RuntimeOpKind::RepeatInterleave` at compile time.
///
/// Note: Currently handles B=1 only (matches dvoice inference usage).
pub fn length_regulate(
    features: &DynTensor,
    durations: &DynTensor,
) -> std::result::Result<DynTensor, KokoroError> {
    let dims = features.dims();
    if dims.len() != 3 {
        return Err(TensorError::RankMismatch {
            expected: 3,
            actual: dims.len(),
        }
        .into());
    }
    if durations.rank() != 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            actual: durations.rank(),
        }
        .into());
    }
    let batch = dims[0];
    if batch != 1 {
        return Err(TensorError::Unsupported(
            "length_regulate currently supports batch=1 only".into(),
        )
        .into());
    }
    // Round durations to nearest integer using banker's rounding (round-half-to-even),
    // matching PyTorch's torch.round() which uses IEEE 754 default rounding mode.
    // clamp_min(1.0) ensures every phoneme gets at least 1 frame — prevents
    // zero-length durations that would drop phonemes in repeat_interleave.
    // Matches PyTorch reference: torch.round(durations).clamp_min(1). Part of #2691.
    let counts = durations.squeeze(0)?.round()?.clamp_min(1.0)?;
    // features [1, D, T] → repeat_interleave on dim=2
    let mut result = features.repeat_interleave(2, &counts)?;
    // Mark segment boundary for verification (#2378): output shape is
    // data-dependent (T_mel = sum(durations)). The verify path splits
    // graphs here; the compile path ignores this marker.
    nn_core::dyn_tensor::trace::record_segment_boundary(
        &mut result,
        "length_regulate".to_string(),
        None,
    );
    Ok(result)
}

// -- Signal processing helpers (extracted to kokoro_signal.rs) ----------------

#[path = "kokoro_signal.rs"]
mod signal;
pub use signal::{
    build_har_from_source, build_har_source, harmonic_source, prepare_istft_input,
    KOKORO_HOP_LENGTH, KOKORO_N_BINS, KOKORO_N_FFT, KOKORO_SAMPLE_RATE,
};

/// Split a `[B, 2*style_dim]` voice embedding into decoder and prosody halves.
///
/// Returns `(decoder_style, prosody_style)`, each `[B, style_dim]`.
pub fn split_style_embedding(
    style: &DynTensor,
    style_dim: usize,
) -> Result<(DynTensor, DynTensor)> {
    let expected_dim1 = 2 * style_dim;
    if style.dims().len() < 2 || style.dims()[1] != expected_dim1 {
        return Err(TensorError::shape_mismatch(
            vec![0, expected_dim1],
            style.dims().to_vec(),
        ));
    }
    let decoder_style = style.narrow(1, 0, style_dim)?;
    let prosody_style = style.narrow(1, style_dim, style_dim)?;
    Ok((decoder_style, prosody_style))
}

// -- KokoroModel (top-level) --------------------------------------------------

/// Kokoro-82M text-to-speech model.
///
/// Full pipeline: PlBert → bert_encoder → TextEncoder → ProsodyPredictor →
///   length_regulate → F0/energy → SourceModule → FullDecoder → (magnitude, phase).
///
/// When SourceModule weights are present (`decoder.generator.m_source.*`), uses
/// the full 9-harmonic SineGen + Linear + tanh excitation. If those weights are
/// missing, `load()` leaves `source_module` as `None` and `forward()` returns
/// `KokoroError::MissingSourceModule` (there is no simplified fallback path).
///
/// Use `forward()` for spectrogram output, or `forward_audio()` (in `kokoro_audio`)
/// for end-to-end audio waveform via CPU iSTFT.
pub struct KokoroModel {
    plbert: PlBert,
    bert_encoder: Linear,
    text_encoder: TextEncoder,
    prosody_predictor: ProsodyPredictor,
    f0_predictor: F0EnergyPredictor,
    /// Multi-harmonic excitation source. If `None`, `forward()` returns
    /// `KokoroError::MissingSourceModule`.
    source_module: Option<SourceModule>,
    decoder: FullDecoder,
    config: KokoroConfig,
}

impl KokoroModel {
    /// Load model weights from a VarBuilder with config.
    pub fn load(vb: impl AsRef<VarBuilder>, config: &KokoroConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let plbert = PlBert::load(vb.pp("plbert"), &config.plbert)?;
        let bert_encoder = {
            let hidden = config.plbert.hidden_size;
            let w = vb.get(&[config.d_en, hidden], "bert_encoder.weight")?;
            let b = vb.get(&[config.d_en], "bert_encoder.bias")?;
            Linear::new(w, Some(b))?
        };
        let text_encoder =
            TextEncoder::load(vb.pp("text_encoder"), config.plbert.vocab_size, config.d_en)?;
        let prosody_predictor = ProsodyPredictor::load(
            vb.pp("prosody_predictor"),
            config.d_en,
            config.style_dim,
            config.n_prosody_layers,
            config.max_dur,
        )?;
        let f0_predictor = F0EnergyPredictor::load(
            vb.pp("predictor"),
            config.d_en,
            config.style_dim,
            config.f0_bilstm_hidden,
        )?;
        let decoder = FullDecoder::load(vb.pp("decoder"), config)?;
        // Try loading SourceModule (9-harmonic SineGen + Linear + tanh).
        // Only treat "weight not found" as absent; propagate structural
        // errors (shape mismatch, corrupt data, dtype) instead of silently
        // converting them to MissingSourceModule (#2699).
        let source_vb = vb.pp("decoder").pp("generator").pp("m_source");
        let source_module = if source_vb.contains_tensor("l_linear.weight") {
            Some(SourceModule::load(&source_vb)?)
        } else {
            None
        };
        Ok(Self {
            plbert,
            bert_encoder,
            text_encoder,
            prosody_predictor,
            f0_predictor,
            source_module,
            decoder,
            config: config.clone(),
        })
    }

    /// Access model configuration.
    #[must_use]
    pub fn config(&self) -> &KokoroConfig {
        &self.config
    }

    /// Access the PlBert encoder sub-module.
    #[must_use]
    pub fn plbert(&self) -> &PlBert {
        &self.plbert
    }

    /// Access the BERT output projection (Linear: hidden_size → d_en).
    #[must_use]
    pub fn bert_encoder(&self) -> &Linear {
        &self.bert_encoder
    }

    /// Access the TextEncoder sub-module (Embedding + Conv + LayerNorm + BiLSTM + Linear).
    #[must_use]
    pub fn text_encoder(&self) -> &TextEncoder {
        &self.text_encoder
    }

    /// Access the ProsodyPredictor sub-module.
    #[must_use]
    pub fn prosody_predictor(&self) -> &ProsodyPredictor {
        &self.prosody_predictor
    }

    /// Access the F0/energy predictor sub-module.
    #[must_use]
    pub fn f0_predictor(&self) -> &F0EnergyPredictor {
        &self.f0_predictor
    }

    /// Access the FullDecoder (Stage1 + Generator) sub-module.
    #[must_use]
    pub fn decoder(&self) -> &FullDecoder {
        &self.decoder
    }

    /// Access the SourceModule (if loaded). `None` means `forward()` will return
    /// `KokoroError::MissingSourceModule`.
    #[must_use]
    pub fn source_module(&self) -> Option<&SourceModule> {
        self.source_module.as_ref()
    }

    /// Reset the SineGen cumulative phase for a new utterance.
    ///
    /// Call at the start of each streaming session to ensure the first chunk
    /// begins with zero phase. Within a session, phase continuity across
    /// chunk boundaries is maintained automatically by SineGen.
    pub fn reset_sinegen_phase(&self) {
        if let Some(ref sm) = self.source_module {
            sm.reset_phase();
        }
    }

    /// Transfer SourceModule `l_linear` weights to the given device.
    ///
    /// No-op if source_module is `None` or already on the target device.
    /// Called by `CompiledKokoro` at init to ensure GPU-resident weights.
    pub fn ensure_source_device(&mut self, device: &Device) -> Result<()> {
        if let Some(ref sm) = self.source_module {
            self.source_module = Some(sm.to_device(device)?);
        }
        Ok(())
    }

    /// Forward: token IDs + pre-computed PlBert output → features + duration predictions.
    ///
    /// `input_ids`: `[B, T]` token indices (needed for TextEncoder embedding).
    /// `bert_output`: `[B, T, hidden_size]` pre-computed PlBert output.
    ///
    /// Returns `(aligned_dur, regulated, dur_logits)`:
    /// - `aligned_dur [B, d_model+style_dim, T_mel]`: ProsodyPredictor features after length_regulate
    ///   (includes style from DurationEncoder; feeds directly into F0EnergyPredictor).
    /// - `regulated [B, d_en, T_mel]`: TextEncoder features after length_regulate (for FullDecoder).
    /// - `dur_logits [B, T, max_dur]`: raw duration logits.
    ///
    /// Two parallel paths with two `length_regulate` calls match dvoice v0.19 reference:
    /// bert_encoder output → ProsodyPredictor → length_regulate → F0EnergyPredictor
    /// TextEncoder output → length_regulate → FullDecoder (asr input)
    ///
    /// # Errors
    /// Returns `KokoroError::InvalidSpeed` if speed is not positive and finite.
    /// Returns `KokoroError::NonFiniteIntermediate` if NaN/Inf detected at any stage.
    pub fn forward_text(
        &self,
        input_ids: &DynTensor,
        bert_output: &DynTensor,
        style: &DynTensor,
        speed: f32,
    ) -> std::result::Result<TextPipelineResult, KokoroError> {
        validate_speed(speed)?;
        // bert_output: [B, T, 768] → Linear → [B, T, d_en] → transpose → [B, d_en, T]
        let encoded = self.bert_encoder.forward(bert_output)?;
        check_tensor_finite(&encoded, "bert_encoder")?;
        let bert_features = encoded.transpose(1, 2)?;
        // TextEncoder: token IDs → [B, d_en, T]
        let text_features = self.text_encoder.forward(input_ids)?;
        check_tensor_finite(&text_features, "text_encoder")?;
        // ProsodyPredictor: bert_features (NOT text_features) — dvoice v0.19 reference.
        // Bug fix: #2511. ProsodyPredictor needs PlBert contextual ALBERT features.
        let (dur_logits, features) = self.prosody_predictor.forward(&bert_features, style)?;
        check_tensor_finite(&dur_logits, "prosody_dur_logits")?;
        check_tensor_finite(&features, "prosody_features")?;
        // Sigmoid-binned duration: sigmoid(logits).sum(last_dim) / speed
        // dur_logits: [B, T, max_dur] → sigmoid → sum(dim=2) → [B, T]
        // Clamp to [1.0, max_dur]: min=1 ensures every phoneme gets ≥1 frame
        // (prevents zero-length durations that drop phonemes in repeat_interleave).
        // Matches dvoice v0.19 reference: durations.clamp(1.0, 50.0).
        let max_dur = self.config.max_dur as f64;
        let durations = dur_logits
            .sigmoid()?
            .sum(2)?
            .mul_scalar(1.0 / f64::from(speed))?
            .clamp(1.0, max_dur)?;
        check_tensor_finite(&durations, "durations_after_sigmoid_sum")?;
        // Two parallel length_regulate calls (dvoice v0.19 reference):
        // 1. ProsodyPredictor features → aligned_dur (for F0EnergyPredictor)
        let aligned_dur = length_regulate(&features, &durations)?;
        check_tensor_finite(&aligned_dur, "length_regulate_dur")?;
        // 2. TextEncoder features → regulated (for FullDecoder asr input)
        let regulated = length_regulate(&text_features, &durations)?;
        check_tensor_finite(&regulated, "length_regulate_text")?;
        Ok(TextPipelineResult::new(aligned_dur, regulated, dur_logits))
    }

    /// Extract frozen encoder features for singing training data preparation.
    ///
    /// Runs the encoder portion only (PlBert → bert_encoder → TextEncoder →
    /// ProsodyPredictor → length_regulate) and returns the intermediate features
    /// without proceeding to the decoder or vocoder.
    ///
    /// This is the method dvoice needs for singing decoder LoRA training:
    /// the encoder is frozen, and its output features are pre-extracted and
    /// saved alongside F0 contours and target audio in training shards.
    ///
    /// # Arguments
    ///
    /// * `input_ids` — `[B, T]` phoneme token IDs (U32 DynTensor).
    /// * `style` — `[B, 2*style_dim]` voice embedding. The prosody half
    ///   (second `style_dim` elements) is used for duration prediction.
    /// * `speed` — Speaking rate multiplier (1.0 = normal).
    ///
    /// # Returns
    ///
    /// [`EncoderFeaturesResult`] containing:
    /// - `regulated`: `[B, d_en, T_mel]` — the primary encoder features
    ///   (TextEncoder output after length regulation). For Kokoro-82M
    ///   default config: `[1, 512, T_mel]`.
    /// - `aligned_dur`: `[B, d_en+style_dim, T_mel]` — prosody-aligned
    ///   features for optional F0/energy extraction.
    /// - `durations`: `[B, T]` — predicted phone durations for alignment.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidSpeed` if speed is not positive and finite.
    /// Returns `KokoroError::NonFiniteIntermediate` if NaN/Inf detected.
    pub fn forward_encoder_features(
        &self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
    ) -> std::result::Result<EncoderFeaturesResult, KokoroError> {
        validate_speed(speed)?;

        // Split style: second half → prosody style (for ProsodyPredictor)
        let style_dim = self.config.style_dim;
        let (_decoder_style, prosody_style) = split_style_embedding(style, style_dim)?;

        // PlBert: token IDs → contextual embeddings [B, T, hidden_size]
        let bert_output = self.plbert.forward(input_ids)?;
        check_tensor_finite(&bert_output, "plbert")?;

        // bert_encoder: [B, T, hidden_size] → [B, T, d_en] → transpose → [B, d_en, T]
        let encoded = self.bert_encoder.forward(&bert_output)?;
        check_tensor_finite(&encoded, "bert_encoder")?;
        let bert_features = encoded.transpose(1, 2)?;

        // TextEncoder: token IDs → [B, d_en, T]
        let text_features = self.text_encoder.forward(input_ids)?;
        check_tensor_finite(&text_features, "text_encoder")?;

        // ProsodyPredictor: bert_features → (dur_logits, features)
        let (dur_logits, features) = self
            .prosody_predictor
            .forward(&bert_features, &prosody_style)?;
        check_tensor_finite(&dur_logits, "prosody_dur_logits")?;
        check_tensor_finite(&features, "prosody_features")?;

        // Sigmoid-binned durations: sigmoid(logits).sum(last_dim) / speed
        let max_dur = self.config.max_dur as f64;
        let durations = dur_logits
            .sigmoid()?
            .sum(2)?
            .mul_scalar(1.0 / f64::from(speed))?
            .clamp(1.0, max_dur)?;
        check_tensor_finite(&durations, "durations_after_sigmoid_sum")?;

        // Length-regulate both feature paths
        let aligned_dur = length_regulate(&features, &durations)?;
        check_tensor_finite(&aligned_dur, "length_regulate_dur")?;
        let regulated = length_regulate(&text_features, &durations)?;
        check_tensor_finite(&regulated, "length_regulate_text")?;

        Ok(EncoderFeaturesResult::new(
            regulated,
            aligned_dur,
            durations,
        ))
    }

    /// Full forward: token IDs → (magnitude, phase) spectrogram.
    ///
    /// `input_ids`: `[B, T]` token indices (U32 DynTensor preferred; F32 legacy accepted).
    /// `style`: `[B, 2*style_dim]` voice embedding (first half → decoder, second half → prosody; default 256).
    /// `speed`: speaking rate multiplier (1.0 = normal).
    ///
    /// Returns `(magnitude [B, n_bins, T_out], phase [B, n_bins, T_out])`.
    /// Convert to audio waveform via iSTFT (see `prepare_istft_input`).
    ///
    /// # Errors
    /// Returns `KokoroError::InvalidSpeed` if speed is not positive and finite.
    /// Returns `KokoroError::MissingSourceModule` if SourceModule weights were not loaded.
    /// No fallback harmonic source is used in this case.
    /// Returns `KokoroError::NonFiniteIntermediate` if NaN/Inf detected at any stage.
    /// Returns `KokoroError::Tensor(NonFiniteData)` if final output contains NaN/Inf.
    pub fn forward(
        &self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
    ) -> std::result::Result<(DynTensor, DynTensor), KokoroError> {
        // validate_speed is called by forward_text below — no need to double-check.
        // Split style: first half → decoder style, second half → prosody style
        let style_dim = self.config.style_dim;
        let (decoder_style, prosody_style) = split_style_embedding(style, style_dim)?;

        // PlBert: token IDs → contextual embeddings [B, T, hidden_size]
        let bert_output = self.plbert.forward(input_ids)?;
        check_tensor_finite(&bert_output, "plbert")?;

        // Text pipeline: bert_encoder + TextEncoder + ProsodyPredictor + length_regulate
        let text_result = self.forward_text(input_ids, &bert_output, &prosody_style, speed)?;

        // F0/energy prediction: prosody-aligned features + style → (F0, energy) at 2x resolution
        let (f0, energy) = self
            .f0_predictor
            .forward(&text_result.aligned_dur, &prosody_style)?;
        check_tensor_finite(&f0, "f0_prediction")?;
        check_tensor_finite(&energy, "energy_prediction")?;

        // Build harmonic source from F0 and energy for Generator.
        let hop_length = self.config.n_fft / 4; // Kokoro default: 5
                                                // v1.0 convention: upsample_scale = product(upsample_rates) * hop_length.
                                                // SineGen generates audio at full sample rate (24kHz), matching
                                                // Generator.f0_upsamp = nn.Upsample(scale_factor=prod(rates)*hop).
        let source_upsample: usize =
            self.config.upsample_rates.iter().product::<usize>() * hop_length;
        let har_source = if let Some(ref sm) = self.source_module {
            // Full 9-harmonic SineGen + Linear + tanh excitation (#2507).
            // f0: [B, 1, 2T] → [B, 2T, 1] for SourceModule (channel-last).
            let f0_frames = f0.transpose(1, 2)?;
            let source = sm.forward(&f0_frames, source_upsample)?;
            check_tensor_finite(&source, "source_module")?;
            // [B, T_audio, 1] → [B, 1, T_audio] for forward STFT input.
            let source_ch = source.transpose(1, 2)?;
            // Forward STFT: time-domain → cat([magnitude, phase], dim=1) (#2645).
            // Matches PyTorch: stft.transform(source) → [B, 2*n_bins, T_stft].
            // Previous code used build_har_from_source() which expanded the time-domain
            // signal to frequency bins — wrong input format for Generator noise_convs.
            let dev = source_ch.device();
            let fwd_stft = KokoroForwardStft::new(self.config.n_fft, hop_length, &dev)?;
            // Use center padding to match torch.stft(center=True) default (#2651).
            let har = fwd_stft.forward_cat_center(&source_ch)?;
            check_tensor_finite(&har, "forward_stft")?;
            har
        } else {
            // SourceModule is required for frequency-domain harmonic source (#2667).
            // The old fallback (build_har_source) produced time-domain data where
            // Generator noise_convs expect STFT-domain [B, 2*n_bins, T_stft].
            return Err(KokoroError::MissingSourceModule);
        };

        // FullDecoder: regulated (TextEncoder asr) + f0 + energy + style + har_source → (magnitude, phase)
        let (magnitude, phase) = self.decoder.forward(
            &text_result.regulated,
            &f0,
            &energy,
            &decoder_style,
            &har_source,
        )?;
        // Final output finiteness check (GPU-native, no CPU round-trip).
        check_output_finite(&magnitude, "KokoroMag")?;
        check_output_finite(&phase, "KokoroPhase")?;
        Ok((magnitude, phase))
    }
}

#[cfg(kani)]
#[path = "kokoro_tts_kani_tests.rs"]
mod kani_proofs;

#[cfg(kani)]
#[path = "kani_kokoro_tts.rs"]
mod kani_model_proofs;

#[cfg(test)]
#[path = "kokoro_tts_tests.rs"]
mod tests;
