// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder functions for Demucs spectral encoder `TensorKernelDef`s.
//!
//! Each encoder block is split into 3 sub-defs because the spectral branch
//! requires CPU-side axis-switch (permute+reshape) between stages:
//!
//! 1. **Main Conv1d + GELU**: Conv1d(k=8, s=4, p=2) → GELU. `[C_in, F]` per time step.
//! 2. **DConv**: 2 residual sub-layers. `[C_out, T]` per freq bin.
//! 3. **Rewrite + GLU**: Conv1d(k=1, C_out → C_out*2) → GLU. `[C_out, F']` per time step.
//!
//! Extracted from nn-metal as part of #860.

use std::borrow::Cow;
use std::collections::HashMap;

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorIRError, TensorKernelDef};

use crate::demucs_shared::{
    build_dconv_sublayer, channels_at_depth, conv1d_output_len, DConvSubLayerInputs,
    DCONV_COMPRESS, DCONV_DEPTH, GROUP_NORM_EPS, SPECTRAL_CONV_PADDING, SPECTRAL_FREQ_EMB_DIM,
    SPECTRAL_FREQ_EMB_FEATURES, SPECTRAL_INPUT_CHANNELS, SPECTRAL_KERNEL_SIZE, SPECTRAL_STRIDE,
};
use crate::demucs_spectral_weights::{DemucsSpectralEncoderWeights, SpectralEncoderBlockWeights};
use crate::DemucsBuilderError;

// Alias: these builders handle only the basic blocks (depths 0..3).
// Deep blocks (depths 4-5 with LSTM + attention) require separate builders.
use crate::demucs_shared::SPECTRAL_BASIC_DEPTH as DEPTH;

// ---------------------------------------------------------------------------
// Sub-def container type
// ---------------------------------------------------------------------------

/// Sub-definitions for a single spectral encoder block.
///
/// Each block is split into 3 sub-defs because CPU-side axis-switch
/// (permute+reshape) is needed between stages.
#[must_use]
pub struct SpectralEncoderBlockSubDefs {
    /// Conv1d(k=8, s=4) + GELU: input [C_in, F], output [C_out, F'].
    pub conv_gelu_def: TensorKernelDef,
    /// DConv(×2): input [C_out, T] (per freq bin), output [C_out, T].
    pub dconv_def: TensorKernelDef,
    /// Rewrite Conv1d(k=1) + GLU: input [C_out, F'], output [C_out, F'].
    pub rewrite_def: TensorKernelDef,
}

/// Named weight maps for the 3 sub-defs of one spectral encoder block.
pub struct SpectralEncoderBlockWeightMaps {
    pub conv_gelu: HashMap<String, Vec<f32>>,
    pub dconv: HashMap<String, Vec<f32>>,
    pub rewrite: HashMap<String, Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Conv1d output length for the spectral encoder's main convolution.
pub fn spectral_conv1d_out_len(
    in_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<usize, DemucsBuilderError> {
    conv1d_output_len(in_len, kernel_size, stride, padding)
}

// ---------------------------------------------------------------------------
// Weight validation helpers (backend-agnostic — typed errors)
// ---------------------------------------------------------------------------

/// Validate all weight tensors for the full spectral encoder.
pub fn validate_all_encoder_weights(
    weights: &DemucsSpectralEncoderWeights,
) -> Result<(), DemucsBuilderError> {
    if weights.blocks.len() != DEPTH {
        return Err(DemucsBuilderError::BlockCountMismatch {
            context: Cow::Borrowed("spectral encoder blocks"),
            expected: DEPTH,
            actual: weights.blocks.len(),
        });
    }

    for (block_idx, block) in weights.blocks.iter().enumerate() {
        let in_ch = if block_idx == 0 {
            SPECTRAL_INPUT_CHANNELS
        } else {
            channels_at_depth(block_idx - 1)
        };
        let out_ch = channels_at_depth(block_idx);
        let compressed = out_ch / DCONV_COMPRESS;
        let prefix = format!("block{block_idx}");

        // Main Conv1d: [out_ch, in_ch, kernel_size=8].
        crate::demucs_shared::validate_weight_size(
            &block.conv_weight,
            &format!("{prefix}.conv_weight"),
            out_ch * in_ch * SPECTRAL_KERNEL_SIZE,
        )?;
        crate::demucs_shared::validate_weight_size(
            &block.conv_bias,
            &format!("{prefix}.conv_bias"),
            out_ch,
        )?;

        // DConv sub-layers.
        if block.dconv.len() != DCONV_DEPTH {
            return Err(DemucsBuilderError::BlockCountMismatch {
                context: Cow::Owned(format!("{prefix}.dconv")),
                expected: DCONV_DEPTH,
                actual: block.dconv.len(),
            });
        }
        for (k, sub) in block.dconv.iter().enumerate() {
            let sp = format!("{prefix}.dc{k}");
            crate::demucs_shared::validate_weight_size(
                &sub.conv_compress_weight,
                &format!("{sp}_cw"),
                compressed * out_ch * crate::demucs_shared::DCONV_KERNEL,
            )?;
            crate::demucs_shared::validate_weight_size(
                &sub.conv_compress_bias,
                &format!("{sp}_cb"),
                compressed,
            )?;
            crate::demucs_shared::validate_weight_size(
                &sub.norm_compress_gamma,
                &format!("{sp}_ng"),
                compressed,
            )?;
            crate::demucs_shared::validate_weight_size(
                &sub.norm_compress_beta,
                &format!("{sp}_nb"),
                compressed,
            )?;
            crate::demucs_shared::validate_weight_size(
                &sub.conv_expand_weight,
                &format!("{sp}_ew"),
                out_ch * 2 * compressed,
            )?;
            crate::demucs_shared::validate_weight_size(
                &sub.conv_expand_bias,
                &format!("{sp}_eb"),
                out_ch * 2,
            )?;
            crate::demucs_shared::validate_weight_size(
                &sub.norm_expand_gamma,
                &format!("{sp}_eng"),
                out_ch * 2,
            )?;
            crate::demucs_shared::validate_weight_size(
                &sub.norm_expand_beta,
                &format!("{sp}_enb"),
                out_ch * 2,
            )?;
            crate::demucs_shared::validate_weight_size(
                &sub.layer_scale,
                &format!("{sp}_ls"),
                out_ch,
            )?;
        }

        // Rewrite Conv1d(k=1): [out_ch*2, out_ch, 1].
        crate::demucs_shared::validate_weight_size(
            &block.rewrite_weight,
            &format!("{prefix}.rw_weight"),
            out_ch * 2 * out_ch,
        )?;
        crate::demucs_shared::validate_weight_size(
            &block.rewrite_bias,
            &format!("{prefix}.rw_bias"),
            out_ch * 2,
        )?;
    }

    // Frequency embedding (optional — only if present).
    if let Some(ref emb) = weights.freq_emb_weight {
        crate::demucs_shared::validate_weight_size(
            emb,
            "freq_emb_weight",
            SPECTRAL_FREQ_EMB_FEATURES * SPECTRAL_FREQ_EMB_DIM,
        )?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Sub-def builders
// ---------------------------------------------------------------------------

/// Build the 3 sub-defs for a single spectral encoder block.
pub fn build_encoder_block_sub_defs(
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    f_in: usize,
    f_out: usize,
    t_len: usize,
) -> Result<SpectralEncoderBlockSubDefs, TensorIRError> {
    let conv_gelu_def = build_conv_gelu_def(block_idx, in_ch, out_ch, f_in, f_out)?;
    let dconv_def = build_dconv_def(block_idx, out_ch, t_len)?;
    let rewrite_def = build_rewrite_def(block_idx, out_ch, f_out)?;

    Ok(SpectralEncoderBlockSubDefs {
        conv_gelu_def,
        dconv_def,
        rewrite_def,
    })
}

/// Build the Conv1d + GELU sub-def (freq-axis downsampling).
///
/// Variable input: "data" [C_in, F].
/// Output: [C_out, F'].
fn build_conv_gelu_def(
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    f_in: usize,
    f_out: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    let name = format!("spec_enc_conv{block_idx}");

    let mut b = TensorBlockBuilder::new(&name);

    let data = b.add_input(nn_dsl::input_names::DATA, &[in_ch, f_in]);
    let conv_weight = b.add_input("conv_weight", &[out_ch, in_ch, SPECTRAL_KERNEL_SIZE]);
    let conv_bias = b.add_input("conv_bias", &[out_ch]);

    // Conv1d: downsample freq by stride.
    let conv_out = b.add_conv1d(
        data,
        conv_weight,
        Some(conv_bias),
        SPECTRAL_STRIDE,
        SPECTRAL_CONV_PADDING,
        &[out_ch, f_out],
    );

    // GELU activation.
    let gelu_out = b.add_gelu(conv_out, &[out_ch, f_out]);

    b.build(gelu_out)
}

/// Build the DConv sub-def: operates on [C_out, T] per frequency bin.
///
/// Identical to the spectral decoder's DConv def.
fn build_dconv_def(
    block_idx: usize,
    out_ch: usize,
    t_len: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    let name = format!("spec_enc_dconv{block_idx}");
    let compressed = out_ch / DCONV_COMPRESS;

    let mut b = TensorBlockBuilder::new(&name);

    let data = b.add_input(nn_dsl::input_names::DATA, &[out_ch, t_len]);

    let mut dconv_inputs = Vec::with_capacity(DCONV_DEPTH);
    for k in 0..DCONV_DEPTH {
        let di = DConvSubLayerInputs::add_to_builder(&mut b, k, out_ch, compressed);
        dconv_inputs.push(di);
    }

    let mut x = data;
    for di in &dconv_inputs {
        x = build_dconv_sublayer(&mut b, x, di, out_ch, compressed, t_len)?;
    }

    b.build(x)
}

/// Build the Rewrite + GLU sub-def.
///
/// Variable input: "data" [C_out, F'].
/// Conv1d(k=1, C_out → C_out*2) → GLU → [C_out, F'].
fn build_rewrite_def(
    block_idx: usize,
    out_ch: usize,
    f_out: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    let name = format!("spec_enc_rw{block_idx}");
    let doubled = out_ch * 2;

    let mut b = TensorBlockBuilder::new(&name);

    let data = b.add_input(nn_dsl::input_names::DATA, &[out_ch, f_out]);
    let rw_weight = b.add_input("rw_weight", &[doubled, out_ch, 1]);
    let rw_bias = b.add_input("rw_bias", &[doubled]);

    // Conv1d(k=1): [C_out, F'] → [C_out*2, F'].
    let rw_out = b.add_conv1d(data, rw_weight, Some(rw_bias), 1, 0, &[doubled, f_out]);

    // GLU: [C_out*2, F'] → [C_out, F'].
    let output = b.add_glu(rw_out, 0, &[doubled, f_out])?;

    b.build(output)
}

// ---------------------------------------------------------------------------
// Weight map builder
// ---------------------------------------------------------------------------

/// Build weight maps for all 3 sub-defs of one encoder block.
pub fn build_encoder_block_weight_maps(
    block: &SpectralEncoderBlockWeights,
) -> SpectralEncoderBlockWeightMaps {
    // Conv + GELU sub-def weights.
    let mut conv_gelu = HashMap::new();
    conv_gelu.insert("conv_weight".to_string(), block.conv_weight.clone());
    conv_gelu.insert("conv_bias".to_string(), block.conv_bias.clone());

    // DConv sub-def weights.
    let mut dconv = HashMap::new();
    for (k, sub) in block.dconv.iter().enumerate() {
        dconv.insert(format!("dc{k}_cw"), sub.conv_compress_weight.clone());
        dconv.insert(format!("dc{k}_cb"), sub.conv_compress_bias.clone());
        dconv.insert(format!("dc{k}_ng"), sub.norm_compress_gamma.clone());
        dconv.insert(format!("dc{k}_nb"), sub.norm_compress_beta.clone());
        dconv.insert(format!("dc{k}_ew"), sub.conv_expand_weight.clone());
        dconv.insert(format!("dc{k}_eb"), sub.conv_expand_bias.clone());
        dconv.insert(format!("dc{k}_eng"), sub.norm_expand_gamma.clone());
        dconv.insert(format!("dc{k}_enb"), sub.norm_expand_beta.clone());
        dconv.insert(format!("dc{k}_ls"), sub.layer_scale.clone());
        dconv.insert(format!("dc{k}_eps"), vec![GROUP_NORM_EPS]);
        dconv.insert(format!("dc{k}_eps2"), vec![GROUP_NORM_EPS]);
    }

    // Rewrite + GLU sub-def weights.
    let mut rewrite = HashMap::new();
    rewrite.insert("rw_weight".to_string(), block.rewrite_weight.clone());
    rewrite.insert("rw_bias".to_string(), block.rewrite_bias.clone());

    SpectralEncoderBlockWeightMaps {
        conv_gelu,
        dconv,
        rewrite,
    }
}

#[cfg(test)]
#[path = "demucs_spectral_encoder_builders_tests.rs"]
mod tests;
