// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LoRA-wrapped Kokoro decoder layers for singing fine-tuning.
//!
//! Kokoro-82M's speech-trained decoder destroys singing F0 contour because
//! ~100+ InstanceNorm layers normalize away F0 variation. This module provides
//! LoRA wrappers for the decoder's conv layers so they can be fine-tuned on
//! singing data while keeping base weights frozen.
//!
//! # Architecture
//!
//! - [`LoraConv1d`]: LoRA adapter for a single Conv1d layer. Applies low-rank
//!   adaptation to the weight tensor by reshaping `[out_ch, in_ch_per_group, k]`
//!   to 2D for the `B @ A` matmul, then reshaping back.
//! - [`TrainableStage1ResBlk`]: LoRA-wrapped Stage 1 AdaIN residual block.
//!   Targets: `conv1`, `conv2` in each block (controls F0 mixing through AdaIN).
//! - [`TrainableGenerator`]: LoRA-wrapped ISTFTNet Generator.
//!   Targets: ResBlock conv layers in each upsampling stage.
//! - [`TrainableKokoroDecoder`]: Composite wrapper combining Stage 1 + Generator.
//!
//! # Example
//!
//! ```ignore
//! let decoder = TrainableKokoroDecoder::new(lora_rank, lora_alpha);
//! decoder.freeze_base();
//! decoder.unfreeze_lora();
//! // ... train on singing data ...
//! decoder.merge_lora()?; // fold LoRA into base for zero-overhead inference
//! ```
//!
//! Part of #4318.

use crate::error::{AutodiffError, Result};
use crate::var::Var;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// LoRA Conv1d adapter
// ---------------------------------------------------------------------------

/// LoRA adapter for a Conv1d weight tensor.
///
/// Applies low-rank adaptation: `W_effective = W_frozen + scaling * reshape(B @ A)`.
/// The weight `[out_ch, in_ch_per_group, kernel_size]` is reshaped to
/// `[out_ch, in_ch_per_group * kernel_size]` for the rank decomposition,
/// then reshaped back after merging.
///
/// Only `lora_a` and `lora_b` are trainable; the original weight is frozen.
#[derive(Debug)]
pub struct LoraConv1d {
    /// Frozen original weight, shape `[out_ch, in_ch_per_group, kernel_size]`.
    frozen_weight: DynTensor,
    /// Frozen original bias, shape `[out_ch]` (optional).
    frozen_bias: Option<DynTensor>,
    /// Low-rank A matrix, shape `[rank, in_ch_per_group * kernel_size]`.
    lora_a: Var,
    /// Low-rank B matrix, shape `[out_ch, rank]`. Zero-initialized.
    lora_b: Var,
    /// Scaling factor: `alpha / rank`.
    scaling: f64,
    /// Conv parameters from the frozen layer.
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
}

impl LoraConv1d {
    /// Create a LoRA adapter from frozen Conv1d weight/bias and convolution parameters.
    ///
    /// - `rank`: low-rank dimension (typical: 4, 8, 16).
    /// - `alpha`: scaling factor (typical: equal to rank).
    ///
    /// `A` is initialized with random normal values, `B` with zeros, so the
    /// initial output is identical to the original Conv1d.
    pub fn new(
        frozen_weight: DynTensor,
        frozen_bias: Option<DynTensor>,
        rank: usize,
        alpha: f64,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        if rank == 0 {
            return Err(AutodiffError::InvalidConfig {
                op: "LoraConv1d",
                reason: "LoRA rank must be > 0".into(),
            });
        }
        if !alpha.is_finite() {
            return Err(AutodiffError::InvalidConfig {
                op: "LoraConv1d",
                reason: format!("LoRA alpha must be finite, got {alpha}"),
            });
        }

        let shape = frozen_weight.shape().dims().to_vec();
        if shape.len() != 3 {
            return Err(AutodiffError::InvalidConfig {
                op: "LoraConv1d",
                reason: format!("expected 3D weight [out, in/g, k], got {}D", shape.len()),
            });
        }
        let out_ch = shape[0];
        let fan_in = shape[1] * shape[2]; // in_ch_per_group * kernel_size

        // A: [rank, fan_in] -- random normal init
        let a_data = DynTensor::randn(0.0, 1.0, &[rank, fan_in], &Device::Cpu)?;
        let lora_a = Var::new(a_data);

        // B: [out_ch, rank] -- zero init (initial output == original)
        let b_data = DynTensor::zeros(&[out_ch, rank], DType::F32, &Device::Cpu)?;
        let lora_b = Var::new(b_data);

        let scaling = alpha / rank as f64;
        if (scaling as f32).is_infinite() {
            return Err(AutodiffError::InvalidConfig {
                op: "LoraConv1d",
                reason: format!("LoRA scaling alpha/rank = {scaling} overflows f32"),
            });
        }

        Ok(Self {
            frozen_weight,
            frozen_bias,
            lora_a,
            lora_b,
            scaling,
            padding,
            stride,
            dilation,
            groups,
        })
    }

    /// Returns references to the trainable variables `[A, B]`.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<&Var> {
        vec![&self.lora_a, &self.lora_b]
    }

    /// Merge LoRA weights into the frozen weight for deployment.
    ///
    /// Returns `W_merged = W + scaling * reshape(B @ A, [out, in/g, k])`.
    pub fn merge(&self) -> Result<DynTensor> {
        let b = self.lora_b.data()?;
        let a = self.lora_a.data()?;

        // B @ A: [out_ch, rank] @ [rank, fan_in] -> [out_ch, fan_in]
        let ba = b.matmul(&a)?;
        let scaled_ba = ba.mul_scalar(self.scaling)?;

        // Reshape to match weight shape [out_ch, in_ch_per_group, kernel_size]
        let shape = self.frozen_weight.shape().dims().to_vec();
        let delta = scaled_ba.reshape(shape)?;

        let merged = self.frozen_weight.add(&delta)?;
        Ok(merged)
    }

    /// Effective weight: `W_frozen + scaling * reshape(B @ A)`.
    ///
    /// Used in the forward pass to compute the adapted convolution.
    pub fn effective_weight(&self) -> Result<DynTensor> {
        self.merge()
    }

    /// Forward pass: conv1d with effective (frozen + LoRA) weight.
    pub fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let w = self.effective_weight()?;
        let y = x.conv1d(&w, self.padding, self.stride, self.dilation, self.groups)?;
        match &self.frozen_bias {
            Some(bias) => {
                // Bias shape [out_channels] needs reshape to [1, out_channels, 1]
                // for broadcasting with [batch, out_channels, length].
                let out_ch = bias.dim(0)?;
                let bias_reshaped = bias.reshape([1, out_ch, 1])?;
                Ok(y.add(&bias_reshaped)?)
            }
            None => Ok(y),
        }
    }

    /// LoRA A matrix (trainable).
    #[must_use]
    pub fn lora_a(&self) -> &Var {
        &self.lora_a
    }

    /// LoRA B matrix (trainable).
    #[must_use]
    pub fn lora_b(&self) -> &Var {
        &self.lora_b
    }

    /// Scaling factor.
    #[must_use]
    pub fn scaling(&self) -> f64 {
        self.scaling
    }
}

// ---------------------------------------------------------------------------
// TrainableStage1ResBlk
// ---------------------------------------------------------------------------

/// LoRA-wrapped Stage 1 residual block for singing fine-tuning.
///
/// Targets `conv1` and `conv2` in each Stage1ResBlk with LoRA adapters.
/// These convolutions control how F0/energy conditioning mixes with the
/// encoder features through the AdaIN normalization path.
///
/// The AdaIN layers and skip connection (conv1x1, pool) remain frozen --
/// we only adapt the main residual path convolutions.
pub struct TrainableStage1ResBlk {
    /// LoRA adapters for conv1 in each block.
    conv1_lora: LoraConv1d,
    /// LoRA adapters for conv2 in each block.
    conv2_lora: LoraConv1d,
}

impl TrainableStage1ResBlk {
    /// Create LoRA adapters for a Stage1ResBlk's conv layers.
    ///
    /// - `conv1_weight`: frozen conv1 weight `[dim_out, dim_in, 3]`.
    /// - `conv1_bias`: frozen conv1 bias `[dim_out]`.
    /// - `conv2_weight`: frozen conv2 weight `[dim_out, dim_out, 3]`.
    /// - `conv2_bias`: frozen conv2 bias `[dim_out]`.
    /// - `rank`: LoRA rank.
    /// - `alpha`: LoRA scaling alpha.
    pub fn new(
        conv1_weight: DynTensor,
        conv1_bias: Option<DynTensor>,
        conv2_weight: DynTensor,
        conv2_bias: Option<DynTensor>,
        rank: usize,
        alpha: f64,
    ) -> Result<Self> {
        // Stage1ResBlk convs use padding=1, stride=1, dilation=1, groups=1
        let conv1_lora = LoraConv1d::new(conv1_weight, conv1_bias, rank, alpha, 1, 1, 1, 1)?;
        let conv2_lora = LoraConv1d::new(conv2_weight, conv2_bias, rank, alpha, 1, 1, 1, 1)?;
        Ok(Self {
            conv1_lora,
            conv2_lora,
        })
    }

    /// Returns all trainable LoRA variables (A and B for both conv1 and conv2).
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<&Var> {
        let mut vars = self.conv1_lora.trainable_vars();
        vars.extend(self.conv2_lora.trainable_vars());
        vars
    }

    /// Merge LoRA weights into frozen weights for both convolutions.
    pub fn merge(&self) -> Result<(DynTensor, DynTensor)> {
        Ok((self.conv1_lora.merge()?, self.conv2_lora.merge()?))
    }

    /// Access to conv1 LoRA adapter.
    #[must_use]
    pub fn conv1_lora(&self) -> &LoraConv1d {
        &self.conv1_lora
    }

    /// Access to conv2 LoRA adapter.
    #[must_use]
    pub fn conv2_lora(&self) -> &LoraConv1d {
        &self.conv2_lora
    }
}

// ---------------------------------------------------------------------------
// TrainableGenerator
// ---------------------------------------------------------------------------

/// LoRA-wrapped ISTFTNet Generator for singing fine-tuning.
///
/// Targets the ResBlock conv layers in each upsampling stage. Each ResBlock
/// has pairs of `(conv1, conv2)` per dilation, and we wrap all of them with
/// LoRA adapters.
///
/// The upsampling ConvTranspose1d, noise convs, and output conv remain frozen.
pub struct TrainableGenerator {
    /// LoRA adapters for all ResBlock convs: `(conv1_lora, conv2_lora)` per dilation per resblock.
    resblock_loras: Vec<(LoraConv1d, LoraConv1d)>,
}

impl TrainableGenerator {
    /// Create LoRA adapters for all ResBlock conv pairs in the Generator.
    ///
    /// `conv_weights` is a list of `(conv1_weight, conv1_bias, conv2_weight, conv2_bias)`
    /// for each dilation layer across all ResBlocks, in order.
    ///
    /// Each conv has the same kernel size and channels within a ResBlock dilation pair.
    /// Padding is `(kernel_size - 1) * dilation / 2` for conv1 and
    /// `(kernel_size - 1) / 2` for conv2. Since we merge LoRA into the weight
    /// (not into a separate matmul path), the padding/stride/dilation are
    /// only needed for the forward pass -- the merge just produces a new weight tensor.
    pub fn from_conv_weights(
        conv_weights: Vec<(DynTensor, Option<DynTensor>, DynTensor, Option<DynTensor>)>,
        paddings: Vec<(usize, usize)>,
        dilations: Vec<(usize, usize)>,
        rank: usize,
        alpha: f64,
    ) -> Result<Self> {
        if conv_weights.len() != paddings.len() || conv_weights.len() != dilations.len() {
            return Err(AutodiffError::InvalidConfig {
                op: "TrainableGenerator",
                reason: "conv_weights, paddings, and dilations must have same length".into(),
            });
        }
        let mut resblock_loras = Vec::with_capacity(conv_weights.len());
        for (i, ((w1, b1, w2, b2), ((p1, p2), (d1, d2)))) in conv_weights
            .into_iter()
            .zip(paddings.into_iter().zip(dilations))
            .enumerate()
        {
            let conv1_lora = LoraConv1d::new(w1, b1, rank, alpha, p1, 1, d1, 1).map_err(|e| {
                AutodiffError::InvalidConfig {
                    op: "TrainableGenerator",
                    reason: format!("conv1 LoRA init failed at index {i}: {e}"),
                }
            })?;
            let conv2_lora = LoraConv1d::new(w2, b2, rank, alpha, p2, 1, d2, 1).map_err(|e| {
                AutodiffError::InvalidConfig {
                    op: "TrainableGenerator",
                    reason: format!("conv2 LoRA init failed at index {i}: {e}"),
                }
            })?;
            resblock_loras.push((conv1_lora, conv2_lora));
        }
        Ok(Self { resblock_loras })
    }

    /// Returns all trainable LoRA variables across all ResBlock conv pairs.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<&Var> {
        let mut vars = Vec::with_capacity(self.resblock_loras.len() * 4);
        for (c1, c2) in &self.resblock_loras {
            vars.extend(c1.trainable_vars());
            vars.extend(c2.trainable_vars());
        }
        vars
    }

    /// Merge all LoRA weights into frozen weights.
    ///
    /// Returns merged `(conv1_weight, conv2_weight)` for each dilation pair.
    pub fn merge_all(&self) -> Result<Vec<(DynTensor, DynTensor)>> {
        self.resblock_loras
            .iter()
            .map(|(c1, c2)| Ok((c1.merge()?, c2.merge()?)))
            .collect()
    }

    /// Number of LoRA conv pairs.
    #[must_use]
    pub fn num_lora_pairs(&self) -> usize {
        self.resblock_loras.len()
    }

    /// Access individual LoRA pairs (for inspection/testing).
    #[must_use]
    pub fn lora_pairs(&self) -> &[(LoraConv1d, LoraConv1d)] {
        &self.resblock_loras
    }
}

// ---------------------------------------------------------------------------
// TrainableKokoroDecoder
// ---------------------------------------------------------------------------

/// LoRA configuration for the Kokoro decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct KokoroLoraConfig {
    /// LoRA rank for Stage 1 ResBlk convolutions.
    pub stage1_rank: usize,
    /// LoRA alpha for Stage 1 ResBlk convolutions.
    pub stage1_alpha: f64,
    /// LoRA rank for Generator ResBlock convolutions.
    pub generator_rank: usize,
    /// LoRA alpha for Generator ResBlock convolutions.
    pub generator_alpha: f64,
}

impl Default for KokoroLoraConfig {
    fn default() -> Self {
        Self {
            stage1_rank: 16,
            stage1_alpha: 16.0,
            generator_rank: 16,
            generator_alpha: 16.0,
        }
    }
}

/// Composite LoRA wrapper for the complete Kokoro decoder (Stage 1 + Generator).
///
/// Provides:
/// - `freeze_base()` / `unfreeze_lora()` API for training setup.
/// - `save_lora_weights()` / `load_lora_weights()` for safetensors-compatible
///   weight export/import.
/// - `merge_lora()` for zero-overhead inference after fine-tuning.
///
/// # Singing Fine-Tuning
///
/// The core problem: Kokoro's ~100+ InstanceNorm layers normalize away F0
/// variation that is essential for singing. By adding LoRA to the conv layers
/// that precede/follow each AdaIN block, we can learn to preserve F0 contour
/// while keeping the base speech model frozen.
pub struct TrainableKokoroDecoder {
    /// LoRA-wrapped Stage 1 blocks (encode + 4 decode blocks).
    stage1_blocks: Vec<TrainableStage1ResBlk>,
    /// LoRA-wrapped Generator ResBlock convs.
    generator: TrainableGenerator,
    /// F0 downsampling conv LoRA (optional, for F0_conv stride-2 projection).
    f0_conv_lora: Option<LoraConv1d>,
    /// Energy downsampling conv LoRA (optional, for N_conv stride-2 projection).
    n_conv_lora: Option<LoraConv1d>,
    /// Configuration used to create this decoder.
    config: KokoroLoraConfig,
}

impl TrainableKokoroDecoder {
    /// Create a trainable decoder from frozen weights and LoRA config.
    ///
    /// `stage1_convs`: list of `(conv1_w, conv1_b, conv2_w, conv2_b)` per Stage1ResBlk
    ///   (typically 5 blocks: 1 encode + 4 decode).
    /// `generator_convs`: list of `(conv1_w, conv1_b, conv2_w, conv2_b)` per ResBlock dilation.
    /// `generator_paddings`: `(conv1_padding, conv2_padding)` per dilation pair.
    /// `generator_dilations`: `(conv1_dilation, conv2_dilation)` per dilation pair.
    /// `f0_conv_weight`: optional F0_conv weight for LoRA wrapping.
    /// `n_conv_weight`: optional N_conv weight for LoRA wrapping.
    pub fn new(
        stage1_convs: Vec<(DynTensor, Option<DynTensor>, DynTensor, Option<DynTensor>)>,
        generator_convs: Vec<(DynTensor, Option<DynTensor>, DynTensor, Option<DynTensor>)>,
        generator_paddings: Vec<(usize, usize)>,
        generator_dilations: Vec<(usize, usize)>,
        f0_conv_weight: Option<(DynTensor, Option<DynTensor>)>,
        n_conv_weight: Option<(DynTensor, Option<DynTensor>)>,
        config: KokoroLoraConfig,
    ) -> Result<Self> {
        let mut stage1_blocks = Vec::with_capacity(stage1_convs.len());
        for (w1, b1, w2, b2) in stage1_convs {
            stage1_blocks.push(TrainableStage1ResBlk::new(
                w1,
                b1,
                w2,
                b2,
                config.stage1_rank,
                config.stage1_alpha,
            )?);
        }

        let generator = TrainableGenerator::from_conv_weights(
            generator_convs,
            generator_paddings,
            generator_dilations,
            config.generator_rank,
            config.generator_alpha,
        )?;

        // F0_conv: stride-2, padding=1, dilation=1, groups=1
        let f0_conv_lora = if let Some((w, b)) = f0_conv_weight {
            Some(LoraConv1d::new(
                w,
                b,
                config.stage1_rank,
                config.stage1_alpha,
                1,
                2,
                1,
                1,
            )?)
        } else {
            None
        };

        // N_conv: same parameters as F0_conv
        let n_conv_lora = if let Some((w, b)) = n_conv_weight {
            Some(LoraConv1d::new(
                w,
                b,
                config.stage1_rank,
                config.stage1_alpha,
                1,
                2,
                1,
                1,
            )?)
        } else {
            None
        };

        Ok(Self {
            stage1_blocks,
            generator,
            f0_conv_lora,
            n_conv_lora,
            config,
        })
    }

    /// Freeze the base model weights.
    ///
    /// This is a no-op by construction: base weights are stored as `DynTensor`
    /// (not `Var`), so they never participate in gradient computation. This
    /// method exists for API clarity and to match the standard LoRA workflow:
    ///
    /// ```ignore
    /// decoder.freeze_base();
    /// decoder.unfreeze_lora();
    /// // ... train ...
    /// ```
    pub fn freeze_base(&mut self) {
        // Base weights are DynTensor, not Var — already frozen by construction.
        // This method is an explicit semantic marker.
    }

    /// Unfreeze the LoRA adapter parameters for training.
    ///
    /// This is a no-op by construction: LoRA parameters (`lora_a`, `lora_b`)
    /// are stored as `Var`, which is always trainable. The optimizer receives
    /// them via [`trainable_vars()`]. This method exists for API clarity.
    pub fn unfreeze_lora(&mut self) {
        // LoRA params are Var — already trainable by construction.
        // This method is an explicit semantic marker.
    }

    /// Returns all trainable LoRA variables across the entire decoder.
    ///
    /// Used to register parameters with an optimizer.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<&Var> {
        let mut vars = Vec::new();
        for block in &self.stage1_blocks {
            vars.extend(block.trainable_vars());
        }
        vars.extend(self.generator.trainable_vars());
        if let Some(lora) = &self.f0_conv_lora {
            vars.extend(lora.trainable_vars());
        }
        if let Some(lora) = &self.n_conv_lora {
            vars.extend(lora.trainable_vars());
        }
        vars
    }

    /// Total number of trainable LoRA parameters.
    #[must_use]
    pub fn num_trainable_params(&self) -> usize {
        self.trainable_vars()
            .iter()
            .filter_map(|v| v.dims().ok())
            .map(|d| d.iter().product::<usize>())
            .sum()
    }

    /// Alias for [`num_trainable_params`] — total LoRA parameter count.
    #[must_use]
    pub fn lora_param_count(&self) -> usize {
        self.num_trainable_params()
    }

    /// Total number of frozen base parameters (not counting LoRA).
    /// This is an estimate based on the frozen weights stored in LoRA adapters.
    #[must_use]
    pub fn num_lora_pairs(&self) -> usize {
        self.stage1_blocks.len() * 2 // 2 convs per Stage1ResBlk
            + self.generator.num_lora_pairs() * 2 // 2 convs per ResBlock dilation
            + if self.f0_conv_lora.is_some() { 1 } else { 0 }
            + if self.n_conv_lora.is_some() { 1 } else { 0 }
    }

    /// Merge all LoRA weights into frozen weights for zero-overhead inference.
    ///
    /// Returns merged weights organized by component:
    /// - `stage1_merged`: `Vec<(conv1_merged, conv2_merged)>` per block.
    /// - `generator_merged`: `Vec<(conv1_merged, conv2_merged)>` per dilation pair.
    /// - `f0_conv_merged`: merged F0_conv weight (if LoRA was applied).
    /// - `n_conv_merged`: merged N_conv weight (if LoRA was applied).
    pub fn merge_lora(&self) -> Result<MergedKokoroWeights> {
        let mut stage1_merged = Vec::with_capacity(self.stage1_blocks.len());
        for block in &self.stage1_blocks {
            stage1_merged.push(block.merge()?);
        }

        let generator_merged = self.generator.merge_all()?;

        let f0_conv_merged = self.f0_conv_lora.as_ref().map(LoraConv1d::merge).transpose()?;
        let n_conv_merged = self.n_conv_lora.as_ref().map(LoraConv1d::merge).transpose()?;

        Ok(MergedKokoroWeights {
            stage1: stage1_merged,
            generator: generator_merged,
            f0_conv: f0_conv_merged,
            n_conv: n_conv_merged,
        })
    }

    /// Save LoRA weights as a flat map of named tensors.
    ///
    /// Weight names follow the pattern:
    /// - `stage1.{block_idx}.conv{1,2}.lora_{a,b}`
    /// - `generator.{pair_idx}.conv{1,2}.lora_{a,b}`
    /// - `f0_conv.lora_{a,b}`
    /// - `n_conv.lora_{a,b}`
    ///
    /// Compatible with safetensors format for serialization.
    pub fn save_lora_weights(&self) -> Result<Vec<(String, DynTensor)>> {
        let mut weights = Vec::new();

        for (i, block) in self.stage1_blocks.iter().enumerate() {
            weights.push((
                format!("stage1.{i}.conv1.lora_a"),
                block.conv1_lora().lora_a().data()?,
            ));
            weights.push((
                format!("stage1.{i}.conv1.lora_b"),
                block.conv1_lora().lora_b().data()?,
            ));
            weights.push((
                format!("stage1.{i}.conv2.lora_a"),
                block.conv2_lora().lora_a().data()?,
            ));
            weights.push((
                format!("stage1.{i}.conv2.lora_b"),
                block.conv2_lora().lora_b().data()?,
            ));
        }

        for (i, (c1, c2)) in self.generator.lora_pairs().iter().enumerate() {
            weights.push((format!("generator.{i}.conv1.lora_a"), c1.lora_a().data()?));
            weights.push((format!("generator.{i}.conv1.lora_b"), c1.lora_b().data()?));
            weights.push((format!("generator.{i}.conv2.lora_a"), c2.lora_a().data()?));
            weights.push((format!("generator.{i}.conv2.lora_b"), c2.lora_b().data()?));
        }

        if let Some(lora) = &self.f0_conv_lora {
            weights.push(("f0_conv.lora_a".into(), lora.lora_a().data()?));
            weights.push(("f0_conv.lora_b".into(), lora.lora_b().data()?));
        }
        if let Some(lora) = &self.n_conv_lora {
            weights.push(("n_conv.lora_a".into(), lora.lora_a().data()?));
            weights.push(("n_conv.lora_b".into(), lora.lora_b().data()?));
        }

        Ok(weights)
    }

    /// Load LoRA weights from a flat map of named tensors.
    ///
    /// The weight map should contain entries matching the names produced by
    /// [`save_lora_weights`]. Missing entries are skipped (those LoRA adapters
    /// retain their current values).
    pub fn load_lora_weights(&self, weight_map: &[(String, DynTensor)]) -> Result<()> {
        let map: std::collections::HashMap<&str, &DynTensor> =
            weight_map.iter().map(|(k, v)| (k.as_str(), v)).collect();

        for (i, block) in self.stage1_blocks.iter().enumerate() {
            if let Some(t) = map.get(format!("stage1.{i}.conv1.lora_a").as_str()) {
                block.conv1_lora().lora_a().set(t)?;
            }
            if let Some(t) = map.get(format!("stage1.{i}.conv1.lora_b").as_str()) {
                block.conv1_lora().lora_b().set(t)?;
            }
            if let Some(t) = map.get(format!("stage1.{i}.conv2.lora_a").as_str()) {
                block.conv2_lora().lora_a().set(t)?;
            }
            if let Some(t) = map.get(format!("stage1.{i}.conv2.lora_b").as_str()) {
                block.conv2_lora().lora_b().set(t)?;
            }
        }

        for (i, (c1, c2)) in self.generator.lora_pairs().iter().enumerate() {
            if let Some(t) = map.get(format!("generator.{i}.conv1.lora_a").as_str()) {
                c1.lora_a().set(t)?;
            }
            if let Some(t) = map.get(format!("generator.{i}.conv1.lora_b").as_str()) {
                c1.lora_b().set(t)?;
            }
            if let Some(t) = map.get(format!("generator.{i}.conv2.lora_a").as_str()) {
                c2.lora_a().set(t)?;
            }
            if let Some(t) = map.get(format!("generator.{i}.conv2.lora_b").as_str()) {
                c2.lora_b().set(t)?;
            }
        }

        if let Some(lora) = &self.f0_conv_lora {
            if let Some(t) = map.get("f0_conv.lora_a") {
                lora.lora_a().set(t)?;
            }
            if let Some(t) = map.get("f0_conv.lora_b") {
                lora.lora_b().set(t)?;
            }
        }
        if let Some(lora) = &self.n_conv_lora {
            if let Some(t) = map.get("n_conv.lora_a") {
                lora.lora_a().set(t)?;
            }
            if let Some(t) = map.get("n_conv.lora_b") {
                lora.lora_b().set(t)?;
            }
        }

        Ok(())
    }

    /// Access the LoRA configuration.
    #[must_use]
    pub fn config(&self) -> &KokoroLoraConfig {
        &self.config
    }

    /// Access Stage 1 LoRA blocks.
    #[must_use]
    pub fn stage1_blocks(&self) -> &[TrainableStage1ResBlk] {
        &self.stage1_blocks
    }

    /// Access Generator LoRA wrapper.
    #[must_use]
    pub fn generator(&self) -> &TrainableGenerator {
        &self.generator
    }

    /// Access F0 conv LoRA adapter (if present).
    #[must_use]
    pub fn f0_conv_lora(&self) -> Option<&LoraConv1d> {
        self.f0_conv_lora.as_ref()
    }

    /// Access N conv LoRA adapter (if present).
    #[must_use]
    pub fn n_conv_lora(&self) -> Option<&LoraConv1d> {
        self.n_conv_lora.as_ref()
    }
}

/// Merged weights from [`TrainableKokoroDecoder::merge_lora`].
pub struct MergedKokoroWeights {
    /// Merged Stage 1 block weights: `(conv1, conv2)` per block.
    pub stage1: Vec<(DynTensor, DynTensor)>,
    /// Merged Generator ResBlock weights: `(conv1, conv2)` per dilation pair.
    pub generator: Vec<(DynTensor, DynTensor)>,
    /// Merged F0_conv weight (if LoRA was applied).
    pub f0_conv: Option<DynTensor>,
    /// Merged N_conv weight (if LoRA was applied).
    pub n_conv: Option<DynTensor>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::{DType, Device};

    fn make_conv_weight(out_ch: usize, in_ch: usize, k: usize) -> DynTensor {
        DynTensor::randn(0.0, 0.1, &[out_ch, in_ch, k], &Device::Cpu).unwrap()
    }

    fn make_bias(ch: usize) -> DynTensor {
        DynTensor::zeros(&[ch], DType::F32, &Device::Cpu).unwrap()
    }

    // -- LoraConv1d tests --

    #[test]
    fn test_lora_conv1d_construction() {
        let w = make_conv_weight(16, 8, 3);
        let b = make_bias(16);
        let lora = LoraConv1d::new(w, Some(b), 4, 4.0, 1, 1, 1, 1).unwrap();
        assert_eq!(lora.trainable_vars().len(), 2);
        assert!((lora.scaling() - 1.0).abs() < 1e-10); // alpha/rank = 4/4 = 1.0
    }

    #[test]
    fn test_lora_conv1d_zero_rank_error() {
        let w = make_conv_weight(16, 8, 3);
        let result = LoraConv1d::new(w, None, 0, 4.0, 1, 1, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_conv1d_non_finite_alpha_error() {
        let w = make_conv_weight(16, 8, 3);
        let result = LoraConv1d::new(w, None, 4, f64::NAN, 1, 1, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_conv1d_wrong_dim_error() {
        // 2D weight should fail (expected 3D)
        let w = DynTensor::zeros(&[16, 8], DType::F32, &Device::Cpu).unwrap();
        let result = LoraConv1d::new(w, None, 4, 4.0, 1, 1, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_conv1d_initial_merge_equals_frozen() {
        // B is zero-initialized, so merge should equal frozen weight
        let w = make_conv_weight(16, 8, 3);
        let w_clone = w.clone();
        let lora = LoraConv1d::new(w, None, 4, 4.0, 1, 1, 1, 1).unwrap();
        let merged = lora.merge().unwrap();

        let diff = merged.sub(&w_clone).unwrap();
        let max_diff = diff
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            max_diff < 1e-6,
            "initial merge should equal frozen weight, got max diff {max_diff}"
        );
    }

    #[test]
    fn test_lora_conv1d_forward_shape() {
        let w = make_conv_weight(16, 8, 3);
        let b = make_bias(16);
        let lora = LoraConv1d::new(w, Some(b), 4, 4.0, 1, 1, 1, 1).unwrap();

        let x = DynTensor::randn(0.0, 1.0, &[1, 8, 32], &Device::Cpu).unwrap();
        let y = lora.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[1, 16, 32]);
    }

    // -- TrainableStage1ResBlk tests --

    #[test]
    fn test_trainable_stage1_resblk_construction() {
        let c1w = make_conv_weight(64, 32, 3);
        let c1b = make_bias(64);
        let c2w = make_conv_weight(64, 64, 3);
        let c2b = make_bias(64);
        let block = TrainableStage1ResBlk::new(c1w, Some(c1b), c2w, Some(c2b), 8, 8.0).unwrap();
        // 4 vars: A,B for conv1 + A,B for conv2
        assert_eq!(block.trainable_vars().len(), 4);
    }

    #[test]
    fn test_trainable_stage1_resblk_merge() {
        let c1w = make_conv_weight(64, 32, 3);
        let c2w = make_conv_weight(64, 64, 3);
        let block =
            TrainableStage1ResBlk::new(c1w.clone(), None, c2w.clone(), None, 4, 4.0).unwrap();
        let (m1, m2) = block.merge().unwrap();
        // Initial merge should equal frozen (B is zero-init)
        let d1 = m1
            .sub(&c1w)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let d2 = m2
            .sub(&c2w)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d1 < 1e-6, "conv1 merge drift: {d1}");
        assert!(d2 < 1e-6, "conv2 merge drift: {d2}");
    }

    // -- TrainableGenerator tests --

    #[test]
    fn test_trainable_generator_construction() {
        let convs = vec![
            (
                make_conv_weight(32, 32, 3),
                Some(make_bias(32)),
                make_conv_weight(32, 32, 3),
                Some(make_bias(32)),
            ),
            (
                make_conv_weight(32, 32, 3),
                Some(make_bias(32)),
                make_conv_weight(32, 32, 3),
                Some(make_bias(32)),
            ),
        ];
        let paddings = vec![(1, 1), (3, 1)];
        let dilations = vec![(1, 1), (3, 1)];
        let generator =
            TrainableGenerator::from_conv_weights(convs, paddings, dilations, 4, 4.0).unwrap();
        assert_eq!(generator.num_lora_pairs(), 2);
        // 4 vars per pair (A,B for conv1 + A,B for conv2) * 2 pairs = 8
        assert_eq!(generator.trainable_vars().len(), 8);
    }

    #[test]
    fn test_trainable_generator_merge_all() {
        let w1 = make_conv_weight(32, 32, 3);
        let w2 = make_conv_weight(32, 32, 3);
        let convs = vec![(w1.clone(), None, w2.clone(), None)];
        let paddings = vec![(1, 1)];
        let dilations = vec![(1, 1)];
        let generator =
            TrainableGenerator::from_conv_weights(convs, paddings, dilations, 4, 4.0).unwrap();
        let merged = generator.merge_all().unwrap();
        assert_eq!(merged.len(), 1);

        let d1 = merged[0]
            .0
            .sub(&w1)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let d2 = merged[0]
            .1
            .sub(&w2)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d1 < 1e-6, "generator conv1 merge drift: {d1}");
        assert!(d2 < 1e-6, "generator conv2 merge drift: {d2}");
    }

    // -- TrainableKokoroDecoder tests --

    #[test]
    fn test_trainable_kokoro_decoder_construction() {
        let config = KokoroLoraConfig::default();

        // 2 Stage1ResBlk blocks
        let stage1 = vec![
            (
                make_conv_weight(64, 32, 3),
                Some(make_bias(64)),
                make_conv_weight(64, 64, 3),
                Some(make_bias(64)),
            ),
            (
                make_conv_weight(64, 64, 3),
                Some(make_bias(64)),
                make_conv_weight(64, 64, 3),
                Some(make_bias(64)),
            ),
        ];

        // 1 Generator ResBlock dilation pair
        let gen_convs = vec![(
            make_conv_weight(32, 32, 3),
            Some(make_bias(32)),
            make_conv_weight(32, 32, 3),
            Some(make_bias(32)),
        )];
        let paddings = vec![(1, 1)];
        let dilations = vec![(1, 1)];

        let decoder = TrainableKokoroDecoder::new(
            stage1,
            gen_convs,
            paddings,
            dilations,
            Some((make_conv_weight(1, 1, 3), Some(make_bias(1)))),
            Some((make_conv_weight(1, 1, 3), Some(make_bias(1)))),
            config,
        )
        .unwrap();

        // 2 blocks * 4 vars + 1 pair * 4 vars + 2 F0/N * 2 vars = 16
        assert_eq!(decoder.trainable_vars().len(), 16);
        assert!(decoder.num_trainable_params() > 0);
    }

    #[test]
    fn test_trainable_kokoro_decoder_merge_roundtrip() {
        let config = KokoroLoraConfig {
            stage1_rank: 4,
            stage1_alpha: 4.0,
            generator_rank: 4,
            generator_alpha: 4.0,
        };

        let c1 = make_conv_weight(32, 16, 3);
        let c2 = make_conv_weight(32, 32, 3);
        let stage1 = vec![(c1.clone(), None, c2, None)];

        let gc1 = make_conv_weight(16, 16, 3);
        let gc2 = make_conv_weight(16, 16, 3);
        let gen_convs = vec![(gc1, None, gc2, None)];
        let paddings = vec![(1, 1)];
        let dilations = vec![(1, 1)];

        let decoder =
            TrainableKokoroDecoder::new(stage1, gen_convs, paddings, dilations, None, None, config)
                .unwrap();

        // B is zero-init, so merged should equal frozen
        let merged = decoder.merge_lora().unwrap();
        assert_eq!(merged.stage1.len(), 1);
        assert_eq!(merged.generator.len(), 1);
        assert!(merged.f0_conv.is_none());
        assert!(merged.n_conv.is_none());

        let d = merged.stage1[0]
            .0
            .sub(&c1)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d < 1e-6, "stage1 merge drift: {d}");
    }

    #[test]
    fn test_save_load_lora_weights_roundtrip() {
        let config = KokoroLoraConfig {
            stage1_rank: 4,
            stage1_alpha: 4.0,
            generator_rank: 4,
            generator_alpha: 4.0,
        };

        let stage1 = vec![(
            make_conv_weight(32, 16, 3),
            None,
            make_conv_weight(32, 32, 3),
            None,
        )];
        let gen_convs = vec![(
            make_conv_weight(16, 16, 3),
            None,
            make_conv_weight(16, 16, 3),
            None,
        )];

        let decoder = TrainableKokoroDecoder::new(
            stage1,
            gen_convs,
            vec![(1, 1)],
            vec![(1, 1)],
            None,
            None,
            config.clone(),
        )
        .unwrap();

        // Save weights
        let saved = decoder.save_lora_weights().unwrap();
        // 1 stage1 block * 4 + 1 gen pair * 4 = 8
        assert_eq!(saved.len(), 8);

        // Verify naming convention
        assert!(saved.iter().any(|(k, _)| k == "stage1.0.conv1.lora_a"));
        assert!(saved.iter().any(|(k, _)| k == "generator.0.conv2.lora_b"));

        // Create a fresh decoder and load the weights
        let stage1_2 = vec![(
            make_conv_weight(32, 16, 3),
            None,
            make_conv_weight(32, 32, 3),
            None,
        )];
        let gen_convs_2 = vec![(
            make_conv_weight(16, 16, 3),
            None,
            make_conv_weight(16, 16, 3),
            None,
        )];
        let decoder2 = TrainableKokoroDecoder::new(
            stage1_2,
            gen_convs_2,
            vec![(1, 1)],
            vec![(1, 1)],
            None,
            None,
            config,
        )
        .unwrap();

        // Load saved weights into the new decoder
        decoder2.load_lora_weights(&saved).unwrap();

        // Verify the loaded weights match
        let saved2 = decoder2.save_lora_weights().unwrap();
        for ((k1, v1), (k2, v2)) in saved.iter().zip(saved2.iter()) {
            assert_eq!(k1, k2);
            let diff = v1
                .sub(v2)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(diff < 1e-6, "weight {k1} mismatch: {diff}");
        }
    }

    #[test]
    fn test_kokoro_lora_config_default() {
        let config = KokoroLoraConfig::default();
        assert_eq!(config.stage1_rank, 16);
        assert!((config.stage1_alpha - 16.0).abs() < 1e-10);
        assert_eq!(config.generator_rank, 16);
        assert!((config.generator_alpha - 16.0).abs() < 1e-10);
    }
}
