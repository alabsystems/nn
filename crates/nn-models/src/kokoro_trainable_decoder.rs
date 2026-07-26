// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LoRA-wrapped Kokoro decoder for singing voice fine-tuning.
//!
//! Provides [`TrainableKokoroDecoder`] which wraps the Stage 1 and Stage 2
//! conv layers of the Kokoro [`FullDecoder`] with LoRA adapters for
//! parameter-efficient fine-tuning on singing data.
//!
//! # Architecture
//!
//! The Kokoro decoder has two stages:
//! - **Stage 1** ([`FullDecoder`]): AdaIN residual blocks (encode + 4 decode)
//!   with Conv1d layers that control F0/energy conditioning.
//! - **Stage 2** ([`Generator`]): ISTFTNet upsampling with ResBlock Conv1d layers
//!   that shape the final waveform spectrum.
//!
//! Kokoro's ~100+ InstanceNorm layers normalize away F0 variation essential
//! for singing. By adding LoRA to the conv layers around each AdaIN block,
//! we learn to preserve F0 contour while keeping base weights frozen.
//!
//! # Fine-Tuning Stages
//!
//! - **Stage 1 (Singing):** LoRA on Stage 1 encode/decode conv layers only.
//!   Learns F0 contour preservation. Lower parameter count.
//! - **Stage 2 (Voice Character):** LoRA on Generator ResBlock conv layers.
//!   Learns voice timbre and spectral characteristics.
//! - **Both:** LoRA on all conv layers for full adaptation.
//!
//! # Example
//!
//! ```ignore
//! use nn_models::kokoro_trainable_decoder::{
//!     TrainableKokoroDecoder, SingingLoraConfig, SingingStage,
//! };
//!
//! // Stage 1: singing F0 adaptation
//! let config = SingingLoraConfig::stage1(16, 16.0);
//! let trainable = TrainableKokoroDecoder::from_full_decoder(&decoder, &config)?;
//!
//! // Stage 2: voice character adaptation
//! let config2 = SingingLoraConfig::stage2(8, 8.0);
//! let trainable2 = TrainableKokoroDecoder::from_full_decoder(&decoder, &config2)?;
//!
//! // Both stages
//! let config3 = SingingLoraConfig::both(16, 16.0, 8, 8.0);
//! let trainable3 = TrainableKokoroDecoder::from_full_decoder(&decoder, &config3)?;
//! ```
//!
//! Part of #4318.

use crate::kokoro_decoder::Generator;
use crate::kokoro_full_decoder::{FullDecoder, Stage1ResBlk};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Conv1d, LoraConfig};
use nn_core::{DType, Device, Result, TensorError};

// ---------------------------------------------------------------------------
// LoRA Conv1d adapter (pure DynTensor, no Var dependency)
// ---------------------------------------------------------------------------

/// LoRA adapter for a Conv1d weight tensor (pure DynTensor version).
///
/// Applies low-rank adaptation: `W_effective = W_frozen + scaling * reshape(B @ A)`.
/// The weight `[out_ch, in_ch_per_group, kernel_size]` is reshaped to
/// `[out_ch, in_ch_per_group * kernel_size]` for the rank decomposition.
#[derive(Debug, Clone)]
pub struct LoraConv1dAdapter {
    /// Frozen original weight, shape `[out_ch, in_ch_per_group, kernel_size]`.
    frozen_weight: DynTensor,
    /// Frozen original bias, shape `[out_ch]` (optional).
    frozen_bias: Option<DynTensor>,
    /// Low-rank A matrix, shape `[rank, in_ch_per_group * kernel_size]`.
    lora_a: DynTensor,
    /// Low-rank B matrix, shape `[out_ch, rank]`. Zero-initialized.
    lora_b: DynTensor,
    /// Scaling factor: `alpha / rank`.
    scaling: f64,
}

impl LoraConv1dAdapter {
    /// Create a LoRA adapter from a frozen Conv1d's weight and bias.
    pub fn from_conv_weight(
        frozen_weight: &DynTensor,
        frozen_bias: Option<&DynTensor>,
        config: &LoraConfig,
    ) -> Result<Self> {
        let shape = frozen_weight.shape().dims().to_vec();
        if shape.len() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: shape.len(),
            });
        }
        let out_ch = shape[0];
        let fan_in = shape[1] * shape[2]; // in_ch_per_group * kernel_size
        let rank = config.rank;

        if rank == 0 {
            return Err(TensorError::InvalidShape("LoRA rank must be > 0".into()));
        }

        // A: [rank, fan_in] -- random normal init
        let lora_a = DynTensor::randn(0.0, 1.0, &[rank, fan_in], &Device::Cpu)?;
        // B: [out_ch, rank] -- zero init (initial output == original)
        let lora_b = DynTensor::zeros(&[out_ch, rank], DType::F32, &Device::Cpu)?;

        let scaling = config.alpha as f64 / rank as f64;

        Ok(Self {
            frozen_weight: frozen_weight.clone(),
            frozen_bias: frozen_bias.cloned(),
            lora_a,
            lora_b,
            scaling,
        })
    }

    /// Create a LoRA adapter from a [`Conv1d`] layer.
    pub fn from_conv1d(conv: &Conv1d, config: &LoraConfig) -> Result<Self> {
        Self::from_conv_weight(conv.weight(), conv.bias(), config)
    }

    /// Trainable parameters: `[A, B]`.
    #[must_use]
    pub fn trainable_params(&self) -> Vec<&DynTensor> {
        vec![&self.lora_a, &self.lora_b]
    }

    /// Mutable trainable parameters: `[A, B]`.
    pub fn trainable_params_mut(&mut self) -> Vec<&mut DynTensor> {
        vec![&mut self.lora_a, &mut self.lora_b]
    }

    /// Merge LoRA into frozen weight: `W + scaling * reshape(B @ A)`.
    pub fn merged_weight(&self) -> Result<DynTensor> {
        let ba = self.lora_b.matmul(&self.lora_a)?;
        let scaled_ba = ba.mul_scalar(self.scaling)?;
        let shape = self.frozen_weight.shape().dims().to_vec();
        let delta = scaled_ba.reshape(shape)?;
        self.frozen_weight.add(&delta)
    }

    /// LoRA A matrix reference.
    #[must_use]
    pub fn lora_a(&self) -> &DynTensor {
        &self.lora_a
    }

    /// LoRA B matrix reference.
    #[must_use]
    pub fn lora_b(&self) -> &DynTensor {
        &self.lora_b
    }

    /// Set LoRA A matrix (for weight loading).
    pub fn set_lora_a(&mut self, a: DynTensor) -> Result<()> {
        if a.dims() != self.lora_a.dims() {
            return Err(TensorError::shape_mismatch(
                self.lora_a.dims().to_vec(),
                a.dims().to_vec(),
            ));
        }
        self.lora_a = a;
        Ok(())
    }

    /// Set LoRA B matrix (for weight loading).
    pub fn set_lora_b(&mut self, b: DynTensor) -> Result<()> {
        if b.dims() != self.lora_b.dims() {
            return Err(TensorError::shape_mismatch(
                self.lora_b.dims().to_vec(),
                b.dims().to_vec(),
            ));
        }
        self.lora_b = b;
        Ok(())
    }

    /// Frozen bias reference (if present).
    #[must_use]
    pub fn frozen_bias(&self) -> Option<&DynTensor> {
        self.frozen_bias.as_ref()
    }

    /// Scaling factor.
    #[must_use]
    pub fn scaling(&self) -> f64 {
        self.scaling
    }
}

// ---------------------------------------------------------------------------
// LoRA-wrapped Stage1ResBlk pair (conv1 + conv2)
// ---------------------------------------------------------------------------

/// LoRA adapters for both conv layers in a [`Stage1ResBlk`].
#[derive(Debug, Clone)]
pub struct LoraStage1Block {
    /// LoRA adapter for conv1.
    pub conv1_lora: LoraConv1dAdapter,
    /// LoRA adapter for conv2.
    pub conv2_lora: LoraConv1dAdapter,
}

impl LoraStage1Block {
    /// Create LoRA adapters from a Stage1ResBlk's conv layers.
    pub fn from_stage1_block(block: &Stage1ResBlk, config: &LoraConfig) -> Result<Self> {
        let conv1_lora = LoraConv1dAdapter::from_conv_weight(
            block.conv1().weight(),
            block.conv1().bias(),
            config,
        )?;
        let conv2_lora = LoraConv1dAdapter::from_conv_weight(
            block.conv2().weight(),
            block.conv2().bias(),
            config,
        )?;
        Ok(Self {
            conv1_lora,
            conv2_lora,
        })
    }

    /// All trainable parameters: A, B for conv1 + A, B for conv2.
    #[must_use]
    pub fn trainable_params(&self) -> Vec<&DynTensor> {
        let mut params = self.conv1_lora.trainable_params();
        params.extend(self.conv2_lora.trainable_params());
        params
    }

    /// Merge both conv LoRA adapters. Returns (conv1_merged, conv2_merged).
    pub fn merge(&self) -> Result<(DynTensor, DynTensor)> {
        Ok((
            self.conv1_lora.merged_weight()?,
            self.conv2_lora.merged_weight()?,
        ))
    }
}

// ---------------------------------------------------------------------------
// LoRA-wrapped ResBlock conv pair (conv1 + conv2 per dilation layer)
// ---------------------------------------------------------------------------

/// LoRA adapters for a (conv1, conv2) pair in a Generator ResBlock dilation layer.
#[derive(Debug, Clone)]
pub struct LoraResBlockPair {
    /// LoRA adapter for conv1 (dilated conv).
    pub conv1_lora: LoraConv1dAdapter,
    /// LoRA adapter for conv2 (non-dilated conv).
    pub conv2_lora: LoraConv1dAdapter,
}

impl LoraResBlockPair {
    /// Create LoRA adapters from a ResBlock dilation layer's conv pair.
    pub fn from_conv_pair(conv1: &Conv1d, conv2: &Conv1d, config: &LoraConfig) -> Result<Self> {
        let conv1_lora = LoraConv1dAdapter::from_conv1d(conv1, config)?;
        let conv2_lora = LoraConv1dAdapter::from_conv1d(conv2, config)?;
        Ok(Self {
            conv1_lora,
            conv2_lora,
        })
    }

    /// All trainable parameters: A, B for conv1 + A, B for conv2.
    #[must_use]
    pub fn trainable_params(&self) -> Vec<&DynTensor> {
        let mut params = self.conv1_lora.trainable_params();
        params.extend(self.conv2_lora.trainable_params());
        params
    }

    /// Merge both conv LoRA adapters. Returns (conv1_merged, conv2_merged).
    pub fn merge(&self) -> Result<(DynTensor, DynTensor)> {
        Ok((
            self.conv1_lora.merged_weight()?,
            self.conv2_lora.merged_weight()?,
        ))
    }
}

// ---------------------------------------------------------------------------
// LoRA-wrapped Generator (Stage 2)
// ---------------------------------------------------------------------------

/// LoRA adapters for the Generator's ResBlock conv layers (Stage 2).
///
/// Each ResBlock has multiple dilation layers, each with a (conv1, conv2) pair.
/// This wraps all such pairs with LoRA adapters for voice character fine-tuning.
#[derive(Debug, Clone)]
pub struct LoraGenerator {
    /// LoRA adapter pairs, one per dilation layer across all ResBlocks.
    resblock_loras: Vec<LoraResBlockPair>,
}

impl LoraGenerator {
    /// Create LoRA adapters from a [`Generator`]'s ResBlock conv layers.
    pub fn from_generator(generator: &Generator, config: &LoraConfig) -> Result<Self> {
        let mut resblock_loras = Vec::new();
        for resblock in generator.res_blocks() {
            for (conv1, conv2) in resblock.conv_pairs() {
                resblock_loras.push(LoraResBlockPair::from_conv_pair(conv1, conv2, config)?);
            }
        }
        Ok(Self { resblock_loras })
    }

    /// All trainable parameters across all ResBlock conv pairs.
    #[must_use]
    pub fn trainable_params(&self) -> Vec<&DynTensor> {
        let mut params = Vec::with_capacity(self.resblock_loras.len() * 4);
        for pair in &self.resblock_loras {
            params.extend(pair.trainable_params());
        }
        params
    }

    /// Number of LoRA conv pairs.
    #[must_use]
    pub fn num_lora_pairs(&self) -> usize {
        self.resblock_loras.len()
    }

    /// Merge all LoRA weights. Returns (conv1_merged, conv2_merged) per pair.
    pub fn merge_all(&self) -> Result<Vec<(DynTensor, DynTensor)>> {
        self.resblock_loras
            .iter()
            .map(|pair| pair.merge())
            .collect()
    }

    /// Access individual LoRA pairs.
    #[must_use]
    pub fn lora_pairs(&self) -> &[LoraResBlockPair] {
        &self.resblock_loras
    }

    /// Save LoRA weights with `generator.{i}.conv{1,2}.lora_{a,b}` naming.
    pub fn save_lora_weights(&self) -> Vec<(String, DynTensor)> {
        let mut weights = Vec::new();
        for (i, pair) in self.resblock_loras.iter().enumerate() {
            weights.push((
                format!("generator.{i}.conv1.lora_a"),
                pair.conv1_lora.lora_a().clone(),
            ));
            weights.push((
                format!("generator.{i}.conv1.lora_b"),
                pair.conv1_lora.lora_b().clone(),
            ));
            weights.push((
                format!("generator.{i}.conv2.lora_a"),
                pair.conv2_lora.lora_a().clone(),
            ));
            weights.push((
                format!("generator.{i}.conv2.lora_b"),
                pair.conv2_lora.lora_b().clone(),
            ));
        }
        weights
    }

    /// Load LoRA weights from a named tensor map.
    pub fn load_lora_weights(
        &mut self,
        map: &std::collections::HashMap<&str, &DynTensor>,
    ) -> Result<()> {
        for (i, pair) in self.resblock_loras.iter_mut().enumerate() {
            if let Some(t) = map.get(format!("generator.{i}.conv1.lora_a").as_str()) {
                pair.conv1_lora.set_lora_a((*t).clone())?;
            }
            if let Some(t) = map.get(format!("generator.{i}.conv1.lora_b").as_str()) {
                pair.conv1_lora.set_lora_b((*t).clone())?;
            }
            if let Some(t) = map.get(format!("generator.{i}.conv2.lora_a").as_str()) {
                pair.conv2_lora.set_lora_a((*t).clone())?;
            }
            if let Some(t) = map.get(format!("generator.{i}.conv2.lora_b").as_str()) {
                pair.conv2_lora.set_lora_b((*t).clone())?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Singing fine-tuning configuration
// ---------------------------------------------------------------------------

/// Which decoder stages to wrap with LoRA adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingingStage {
    /// Stage 1 only: encode + decode blocks (F0 contour preservation).
    Stage1,
    /// Stage 2 only: Generator ResBlock convs (voice character / timbre).
    Stage2,
    /// Both stages: full decoder adaptation.
    Both,
}

/// LoRA configuration for singing fine-tuning with stage-specific settings.
///
/// Allows different LoRA rank/alpha for Stage 1 vs Stage 2, since:
/// - Stage 1 needs higher rank to learn F0 conditioning around InstanceNorm
/// - Stage 2 may use lower rank for spectral shaping (fewer parameters)
#[derive(Debug, Clone, PartialEq)]
pub struct SingingLoraConfig {
    /// Which stages to apply LoRA to.
    pub stage: SingingStage,
    /// LoRA config for Stage 1 (encode/decode blocks). Ignored if stage is `Stage2`.
    pub stage1_config: LoraConfig,
    /// LoRA config for Stage 2 (Generator ResBlocks). Ignored if stage is `Stage1`.
    pub stage2_config: LoraConfig,
}

impl SingingLoraConfig {
    /// Stage 1 only: F0 contour preservation for singing.
    pub fn stage1(rank: usize, alpha: f32) -> Self {
        Self {
            stage: SingingStage::Stage1,
            stage1_config: LoraConfig::new(rank, alpha),
            stage2_config: LoraConfig::default(),
        }
    }

    /// Stage 2 only: voice character adaptation.
    pub fn stage2(rank: usize, alpha: f32) -> Self {
        Self {
            stage: SingingStage::Stage2,
            stage1_config: LoraConfig::default(),
            stage2_config: LoraConfig::new(rank, alpha),
        }
    }

    /// Both stages: full decoder adaptation.
    pub fn both(
        stage1_rank: usize,
        stage1_alpha: f32,
        stage2_rank: usize,
        stage2_alpha: f32,
    ) -> Self {
        Self {
            stage: SingingStage::Both,
            stage1_config: LoraConfig::new(stage1_rank, stage1_alpha),
            stage2_config: LoraConfig::new(stage2_rank, stage2_alpha),
        }
    }
}

impl Default for SingingLoraConfig {
    fn default() -> Self {
        Self::both(16, 16.0, 8, 8.0)
    }
}

// ---------------------------------------------------------------------------
// Merged weights
// ---------------------------------------------------------------------------

/// Merged weights from [`TrainableKokoroDecoder::merge_lora`].
pub struct MergedDecoderWeights {
    /// Merged encode block: (conv1, conv2). `None` if Stage 1 was not wrapped.
    pub encode: Option<(DynTensor, DynTensor)>,
    /// Merged decode blocks: Vec<(conv1, conv2)>. Empty if Stage 1 was not wrapped.
    pub decode: Vec<(DynTensor, DynTensor)>,
    /// Merged Generator ResBlock conv pairs. Empty if Stage 2 was not wrapped.
    pub generator: Vec<(DynTensor, DynTensor)>,
}

// ---------------------------------------------------------------------------
// TrainableKokoroDecoder
// ---------------------------------------------------------------------------

/// LoRA-wrapped Kokoro decoder for singing fine-tuning.
///
/// Wraps Stage 1 (encode + decode blocks) and/or Stage 2 (Generator ResBlock)
/// conv layers with LoRA adapters. The AdaIN layers, skip connections, upsample
/// ConvTranspose1d, noise convs, and output conv remain frozen.
///
/// # Frozen vs. Trainable
///
/// - **Frozen (by construction):** All original Conv1d weights, AdaIn layers,
///   ConvTranspose1d layers, and Generator structural layers. These are stored
///   as `DynTensor` values -- never modified.
/// - **Trainable:** LoRA A/B matrices for each targeted conv layer.
///   These are the only parameters returned by [`trainable_params()`].
///
/// # Usage
///
/// ```ignore
/// let config = SingingLoraConfig::stage1(16, 16.0);
/// let trainable = TrainableKokoroDecoder::from_full_decoder(&decoder, &config)?;
///
/// // Get all trainable parameters for optimizer
/// let params = trainable.trainable_params();
///
/// // After training: merge LoRA into frozen weights
/// let merged = trainable.merge_lora()?;
/// ```
pub struct TrainableKokoroDecoder {
    /// LoRA-wrapped encode block (Stage 1). `None` if Stage 2 only.
    encode_lora: Option<LoraStage1Block>,
    /// LoRA-wrapped decode blocks (Stage 1). Empty if Stage 2 only.
    decode_loras: Vec<LoraStage1Block>,
    /// LoRA-wrapped Generator ResBlock convs (Stage 2). `None` if Stage 1 only.
    generator_lora: Option<LoraGenerator>,
    /// Configuration used.
    singing_config: SingingLoraConfig,
}

impl TrainableKokoroDecoder {
    /// Create a trainable decoder from a pretrained [`FullDecoder`] with
    /// stage-specific LoRA configuration.
    ///
    /// This is the primary constructor for singing fine-tuning. It extracts
    /// conv weights from the `FullDecoder` and wraps them with LoRA adapters
    /// according to the [`SingingLoraConfig`].
    pub fn from_full_decoder(decoder: &FullDecoder, config: &SingingLoraConfig) -> Result<Self> {
        let do_stage1 = matches!(config.stage, SingingStage::Stage1 | SingingStage::Both);
        let do_stage2 = matches!(config.stage, SingingStage::Stage2 | SingingStage::Both);

        // Stage 1: wrap encode + decode blocks
        let encode_lora = if do_stage1 {
            Some(LoraStage1Block::from_stage1_block(
                decoder.encode_block(),
                &config.stage1_config,
            )?)
        } else {
            None
        };

        let decode_loras = if do_stage1 {
            decoder
                .decode_blocks()
                .iter()
                .map(|block| LoraStage1Block::from_stage1_block(block, &config.stage1_config))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        // Stage 2: wrap Generator ResBlock convs
        let generator_lora = if do_stage2 {
            Some(LoraGenerator::from_generator(
                decoder.generator(),
                &config.stage2_config,
            )?)
        } else {
            None
        };

        Ok(Self {
            encode_lora,
            decode_loras,
            generator_lora,
            singing_config: config.clone(),
        })
    }

    /// Create a trainable decoder wrapping Stage 1 only (backward-compatible).
    ///
    /// Uses a single [`LoraConfig`] for all Stage 1 layers. Generator stays frozen.
    pub fn from_pretrained(decoder: &FullDecoder, config: &LoraConfig) -> Result<Self> {
        let singing_config = SingingLoraConfig {
            stage: SingingStage::Stage1,
            stage1_config: config.clone(),
            stage2_config: LoraConfig::default(),
        };
        Self::from_full_decoder(decoder, &singing_config)
    }

    /// Returns all trainable LoRA parameters across the decoder.
    #[must_use]
    pub fn trainable_params(&self) -> Vec<&DynTensor> {
        let mut params = Vec::new();
        if let Some(encode) = &self.encode_lora {
            params.extend(encode.trainable_params());
        }
        for decode_lora in &self.decode_loras {
            params.extend(decode_lora.trainable_params());
        }
        if let Some(generator) = &self.generator_lora {
            params.extend(generator.trainable_params());
        }
        params
    }

    /// Total number of trainable LoRA parameters (scalar count).
    #[must_use]
    pub fn num_trainable_params(&self) -> usize {
        self.trainable_params()
            .iter()
            .map(|t| t.dims().iter().product::<usize>())
            .sum()
    }

    /// Number of LoRA adapter blocks (Stage 1 encode/decode blocks + Generator pairs).
    #[must_use]
    pub fn num_lora_blocks(&self) -> usize {
        let stage1 = if self.encode_lora.is_some() { 1 } else { 0 } + self.decode_loras.len();
        let stage2 = self
            .generator_lora
            .as_ref()
            .map_or(0, |g| g.num_lora_pairs());
        stage1 + stage2
    }

    /// Which singing stage this decoder is configured for.
    #[must_use]
    pub fn singing_stage(&self) -> SingingStage {
        self.singing_config.stage
    }

    /// Merge all LoRA weights into frozen weights for zero-overhead inference.
    pub fn merge_lora(&self) -> Result<MergedDecoderWeights> {
        let encode = self.encode_lora.as_ref().map(|e| e.merge()).transpose()?;

        let mut decode = Vec::with_capacity(self.decode_loras.len());
        for lora in &self.decode_loras {
            decode.push(lora.merge()?);
        }

        let generator = self
            .generator_lora
            .as_ref()
            .map(|g| g.merge_all())
            .transpose()?
            .unwrap_or_default();

        Ok(MergedDecoderWeights {
            encode,
            decode,
            generator,
        })
    }

    /// Save LoRA weights as named tensors (safetensors-compatible).
    ///
    /// Naming convention:
    /// - Stage 1: `encode.conv{1,2}.lora_{a,b}`, `decode.{i}.conv{1,2}.lora_{a,b}`
    /// - Stage 2: `generator.{i}.conv{1,2}.lora_{a,b}`
    pub fn save_lora_weights(&self) -> Vec<(String, DynTensor)> {
        let mut weights = Vec::new();

        // Stage 1: encode block
        if let Some(encode) = &self.encode_lora {
            weights.push((
                "encode.conv1.lora_a".into(),
                encode.conv1_lora.lora_a().clone(),
            ));
            weights.push((
                "encode.conv1.lora_b".into(),
                encode.conv1_lora.lora_b().clone(),
            ));
            weights.push((
                "encode.conv2.lora_a".into(),
                encode.conv2_lora.lora_a().clone(),
            ));
            weights.push((
                "encode.conv2.lora_b".into(),
                encode.conv2_lora.lora_b().clone(),
            ));
        }

        // Stage 1: decode blocks
        for (i, lora) in self.decode_loras.iter().enumerate() {
            weights.push((
                format!("decode.{i}.conv1.lora_a"),
                lora.conv1_lora.lora_a().clone(),
            ));
            weights.push((
                format!("decode.{i}.conv1.lora_b"),
                lora.conv1_lora.lora_b().clone(),
            ));
            weights.push((
                format!("decode.{i}.conv2.lora_a"),
                lora.conv2_lora.lora_a().clone(),
            ));
            weights.push((
                format!("decode.{i}.conv2.lora_b"),
                lora.conv2_lora.lora_b().clone(),
            ));
        }

        // Stage 2: generator ResBlock convs
        if let Some(generator) = &self.generator_lora {
            weights.extend(generator.save_lora_weights());
        }

        weights
    }

    /// Load LoRA weights from a named tensor map.
    ///
    /// Missing entries are silently skipped (those adapters keep their
    /// current values).
    pub fn load_lora_weights(&mut self, weight_map: &[(String, DynTensor)]) -> Result<()> {
        let map: std::collections::HashMap<&str, &DynTensor> =
            weight_map.iter().map(|(k, v)| (k.as_str(), v)).collect();

        // Stage 1: encode block
        if let Some(encode) = &mut self.encode_lora {
            if let Some(t) = map.get("encode.conv1.lora_a") {
                encode.conv1_lora.set_lora_a((*t).clone())?;
            }
            if let Some(t) = map.get("encode.conv1.lora_b") {
                encode.conv1_lora.set_lora_b((*t).clone())?;
            }
            if let Some(t) = map.get("encode.conv2.lora_a") {
                encode.conv2_lora.set_lora_a((*t).clone())?;
            }
            if let Some(t) = map.get("encode.conv2.lora_b") {
                encode.conv2_lora.set_lora_b((*t).clone())?;
            }
        }

        // Stage 1: decode blocks
        for (i, lora) in self.decode_loras.iter_mut().enumerate() {
            if let Some(t) = map.get(format!("decode.{i}.conv1.lora_a").as_str()) {
                lora.conv1_lora.set_lora_a((*t).clone())?;
            }
            if let Some(t) = map.get(format!("decode.{i}.conv1.lora_b").as_str()) {
                lora.conv1_lora.set_lora_b((*t).clone())?;
            }
            if let Some(t) = map.get(format!("decode.{i}.conv2.lora_a").as_str()) {
                lora.conv2_lora.set_lora_a((*t).clone())?;
            }
            if let Some(t) = map.get(format!("decode.{i}.conv2.lora_b").as_str()) {
                lora.conv2_lora.set_lora_b((*t).clone())?;
            }
        }

        // Stage 2: generator
        if let Some(generator) = &mut self.generator_lora {
            generator.load_lora_weights(&map)?;
        }

        Ok(())
    }

    /// Access the singing LoRA configuration.
    #[must_use]
    pub fn singing_config(&self) -> &SingingLoraConfig {
        &self.singing_config
    }

    /// Access the backward-compatible single LoRA config (Stage 1 config).
    #[must_use]
    pub fn config(&self) -> &LoraConfig {
        &self.singing_config.stage1_config
    }

    /// Access encode block LoRA adapters (Stage 1).
    #[must_use]
    pub fn encode_lora(&self) -> Option<&LoraStage1Block> {
        self.encode_lora.as_ref()
    }

    /// Access decode block LoRA adapters (Stage 1).
    #[must_use]
    pub fn decode_loras(&self) -> &[LoraStage1Block] {
        &self.decode_loras
    }

    /// Access Generator LoRA adapters (Stage 2).
    #[must_use]
    pub fn generator_lora(&self) -> Option<&LoraGenerator> {
        self.generator_lora.as_ref()
    }
}

#[cfg(test)]
#[path = "kokoro_trainable_decoder_tests.rs"]
mod tests;
