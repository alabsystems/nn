// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder functions for Demucs spectral decoder `TensorKernelDef`s.
//!
//! Each decoder block is split into 3 sub-defs because the spectral branch
//! requires CPU-side axis-switch (permute+reshape) between stages:
//!
//! 1. **Rewrite**: skip_add → Conv2d(3×3) → GLU. Operates on `[C, F, T]`.
//! 2. **DConv**: 2 residual sub-layers. Operates on `[C, T]` per freq bin.
//! 3. **ConvTranspose1d**: upsample along freq → trim. `[C, F]` per time step.
//!
//! Extracted from nn-metal as part of #860.

use std::collections::HashMap;

use std::borrow::Cow;

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorIRError, TensorKernelDef};

use crate::demucs_shared::{
    build_dconv_sublayer, channels_at_depth, validate_weight_size, DConvSubLayerInputs,
    DCONV_COMPRESS, DCONV_DEPTH, GROUP_NORM_EPS, SPECTRAL_CONV_TR_PADDING, SPECTRAL_KERNEL_SIZE,
    SPECTRAL_OUTPUT_CHANNELS, SPECTRAL_REWRITE_KERNEL, SPECTRAL_REWRITE_PADDING, SPECTRAL_STRIDE,
};
use crate::demucs_spectral_weights::{DemucsSpectralDecoderWeights, SpectralDecoderBlockWeights};
use crate::DemucsBuilderError;

// Alias: these builders handle only the basic blocks (depths 0..3).
// Deep blocks (depths 4-5 with LSTM + attention) require separate builders.
use crate::demucs_shared::SPECTRAL_BASIC_DEPTH as DEPTH;

// ---------------------------------------------------------------------------
// Sub-def container type
// ---------------------------------------------------------------------------

/// Sub-definitions for a single spectral decoder block.
///
/// Each block is split into 3 sub-defs because CPU-side axis-switch
/// (permute+reshape) is needed between stages.
#[must_use]
pub struct SpectralDecoderBlockSubDefs {
    /// Conv2d(3×3) → GLU: input [C, F, T], output [C, F, T].
    pub rewrite_def: TensorKernelDef,
    /// DConv(×2): input [C, T] (per freq bin), output [C, T].
    pub dconv_def: TensorKernelDef,
    /// ConvTranspose1d: input [C, F] (per time step), output [C_out, F_out].
    pub conv_tr_def: TensorKernelDef,
}

/// Named weight maps for the 3 sub-defs of one spectral decoder block.
pub struct SpectralDecoderBlockWeightMaps {
    pub rewrite: HashMap<String, Vec<f32>>,
    pub dconv: HashMap<String, Vec<f32>>,
    pub conv_tr: HashMap<String, Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

/// Conv2d output length (same formula per axis).
///
/// Returns `Err` if `stride == 0` or `in_len + 2*padding < kernel_size`.
pub fn conv2d_output_len(
    in_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<usize, DemucsBuilderError> {
    if stride == 0 {
        return Err(DemucsBuilderError::InvalidConvDim {
            msg: Cow::Borrowed("conv2d_output_len: stride must be > 0"),
        });
    }
    let padded = in_len + 2 * padding;
    if padded < kernel_size {
        return Err(DemucsBuilderError::InvalidConvDim {
            msg: Cow::Owned(format!(
                "conv2d_output_len: padded {padded} < kernel_size {kernel_size}"
            )),
        });
    }
    Ok((padded - kernel_size) / stride + 1)
}

// ---------------------------------------------------------------------------
// Weight validation helpers (backend-agnostic — typed errors)
// ---------------------------------------------------------------------------

/// Validate all weight tensors for the full spectral decoder.
pub fn validate_all_decoder_weights(
    weights: &DemucsSpectralDecoderWeights,
) -> Result<(), DemucsBuilderError> {
    if weights.blocks.len() != DEPTH {
        return Err(DemucsBuilderError::BlockCountMismatch {
            context: Cow::Borrowed("spectral decoder blocks"),
            expected: DEPTH,
            actual: weights.blocks.len(),
        });
    }

    for (block_idx, block) in weights.blocks.iter().enumerate() {
        let encoder_depth = DEPTH - 1 - block_idx;
        let in_ch = channels_at_depth(encoder_depth);
        let out_ch = if encoder_depth == 0 {
            SPECTRAL_OUTPUT_CHANNELS
        } else {
            channels_at_depth(encoder_depth - 1)
        };
        let compressed = in_ch / DCONV_COMPRESS;
        let prefix = format!("block{block_idx}");

        // Rewrite Conv2d: [in_ch*2, in_ch, 3, 3].
        validate_weight_size(
            &block.rewrite_weight,
            &format!("{prefix}.rw_weight"),
            in_ch * 2 * in_ch * SPECTRAL_REWRITE_KERNEL * SPECTRAL_REWRITE_KERNEL,
        )?;
        validate_weight_size(&block.rewrite_bias, &format!("{prefix}.rw_bias"), in_ch * 2)?;

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
            validate_weight_size(
                &sub.conv_compress_weight,
                &format!("{sp}_cw"),
                compressed * in_ch * crate::demucs_shared::DCONV_KERNEL,
            )?;
            validate_weight_size(&sub.conv_compress_bias, &format!("{sp}_cb"), compressed)?;
            validate_weight_size(&sub.norm_compress_gamma, &format!("{sp}_ng"), compressed)?;
            validate_weight_size(&sub.norm_compress_beta, &format!("{sp}_nb"), compressed)?;
            validate_weight_size(
                &sub.conv_expand_weight,
                &format!("{sp}_ew"),
                in_ch * 2 * compressed,
            )?;
            validate_weight_size(&sub.conv_expand_bias, &format!("{sp}_eb"), in_ch * 2)?;
            validate_weight_size(&sub.norm_expand_gamma, &format!("{sp}_eng"), in_ch * 2)?;
            validate_weight_size(&sub.norm_expand_beta, &format!("{sp}_enb"), in_ch * 2)?;
            validate_weight_size(&sub.layer_scale, &format!("{sp}_ls"), in_ch)?;
        }

        // ConvTranspose1d (on freq axis).
        validate_weight_size(
            &block.conv_tr_weight,
            &format!("{prefix}.ct_weight"),
            in_ch * out_ch * SPECTRAL_KERNEL_SIZE,
        )?;
        validate_weight_size(&block.conv_tr_bias, &format!("{prefix}.ct_bias"), out_ch)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Sub-def builders
// ---------------------------------------------------------------------------

/// Build the 3 sub-defs for a single spectral decoder block.
#[allow(clippy::too_many_arguments)]
pub fn build_decoder_block_sub_defs(
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    f_in: usize,
    t_in: usize,
    rw_f_out: usize,
    rw_t_out: usize,
    target_f: usize,
    is_last: bool,
) -> Result<SpectralDecoderBlockSubDefs, TensorIRError> {
    let rewrite_def = build_rewrite_def(block_idx, in_ch, f_in, t_in, rw_f_out, rw_t_out)?;
    let dconv_def = build_dconv_def(block_idx, in_ch, rw_t_out)?;
    let conv_tr_def = build_conv_tr_def(block_idx, in_ch, out_ch, rw_f_out, target_f, is_last)?;

    Ok(SpectralDecoderBlockSubDefs {
        rewrite_def,
        dconv_def,
        conv_tr_def,
    })
}

/// Build the Rewrite sub-def: skip_add → Conv2d(3×3) → GLU.
///
/// Variable inputs: "data" [C, F*T], "skip" [C, F*T].
/// Operates on [C, F, T] via reshape within the def.
fn build_rewrite_def(
    block_idx: usize,
    in_ch: usize,
    f_in: usize,
    t_in: usize,
    rw_f_out: usize,
    rw_t_out: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    let name = format!("spec_dec_rw{block_idx}");
    let doubled = in_ch * 2;
    let ft = f_in * t_in;
    let rw_ft = rw_f_out * rw_t_out;

    let mut b = TensorBlockBuilder::new(&name);

    // Variable inputs as flat [C, F*T].
    let data = b.add_input(nn_dsl::input_names::DATA, &[in_ch, ft]);
    let skip = b.add_input(nn_dsl::input_names::SKIP, &[in_ch, ft]);

    // Rewrite Conv2d weights.
    let rw_weight = b.add_input(
        "rw_weight",
        &[
            doubled,
            in_ch,
            SPECTRAL_REWRITE_KERNEL,
            SPECTRAL_REWRITE_KERNEL,
        ],
    );
    let rw_bias = b.add_input("rw_bias", &[doubled]);

    // Skip add: [C, F*T].
    let x = b.add_binary_add(data, skip, &[in_ch, ft]);

    // Reshape to [C, F, T] for Conv2d.
    let x_3d = b.add_reshape(x, &[in_ch, f_in, t_in]);

    // Conv2d(in_ch → doubled, k=3×3, s=1, p=1): preserves F and T.
    let conv_out = b.add_conv2d(
        x_3d,
        rw_weight,
        Some(rw_bias),
        1,
        1,
        SPECTRAL_REWRITE_PADDING,
        SPECTRAL_REWRITE_PADDING,
        &[doubled, rw_f_out, rw_t_out],
    );

    // Reshape to [doubled, F*T] for GLU (operates on channel dim).
    let conv_flat = b.add_reshape(conv_out, &[doubled, rw_ft]);

    // GLU: [doubled, F*T] → [in_ch, F*T].
    let glu_out = b.add_glu(conv_flat, 0, &[doubled, rw_ft])?;

    // Output as [in_ch, rw_ft] flat — the forward() code handles layout.
    b.build(glu_out)
}

/// Build the DConv sub-def: operates on [C, T] per frequency bin.
///
/// Variable input: "data" [C, T].
fn build_dconv_def(
    block_idx: usize,
    in_ch: usize,
    t_len: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    let name = format!("spec_dec_dconv{block_idx}");
    let compressed = in_ch / DCONV_COMPRESS;

    let mut b = TensorBlockBuilder::new(&name);

    let data = b.add_input(nn_dsl::input_names::DATA, &[in_ch, t_len]);

    // DConv inputs (2 sub-layers).
    let mut dconv_inputs = Vec::with_capacity(DCONV_DEPTH);
    for k in 0..DCONV_DEPTH {
        let di = DConvSubLayerInputs::add_to_builder(&mut b, k, in_ch, compressed);
        dconv_inputs.push(di);
    }

    let mut x = data;
    for di in &dconv_inputs {
        x = build_dconv_sublayer(&mut b, x, di, in_ch, compressed, t_len)?;
    }

    b.build(x)
}

/// Build the ConvTranspose1d sub-def: upsample along freq axis + optional trim.
///
/// Variable input: "data" [C, F].
/// Output: [C_out, target_f] (or [C_out, ct_f_out] if no trim needed).
fn build_conv_tr_def(
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    f_in: usize,
    target_f: usize,
    is_last: bool,
) -> Result<TensorKernelDef, TensorIRError> {
    let name = format!("spec_dec_ct{block_idx}");

    let mut b = TensorBlockBuilder::new(&name);

    let data = b.add_input(nn_dsl::input_names::DATA, &[in_ch, f_in]);
    let ct_weight = b.add_input("ct_weight", &[in_ch, out_ch, SPECTRAL_KERNEL_SIZE]);
    let ct_bias = b.add_input("ct_bias", &[out_ch]);

    // ConvTranspose1d: upsample freq by SPECTRAL_STRIDE.
    let ct_f_out =
        (f_in - 1) * SPECTRAL_STRIDE + SPECTRAL_KERNEL_SIZE - 2 * SPECTRAL_CONV_TR_PADDING;
    let ct_out = b.add_conv_transpose_1d(
        data,
        ct_weight,
        Some(ct_bias),
        SPECTRAL_STRIDE,
        SPECTRAL_CONV_TR_PADDING,
        1, // dilation
        1, // groups
        0, // output_padding
        &[out_ch, ct_f_out],
    );

    // Trim to target frequency.
    let trimmed = if ct_f_out > target_f {
        b.add_narrow(ct_out, 1, 0, target_f, &[out_ch, target_f])
    } else {
        ct_out
    };

    // Note: GELU is applied CPU-side in forward() for simplicity.
    let _ = is_last;

    b.build(trimmed)
}

// ---------------------------------------------------------------------------
// Weight map builder
// ---------------------------------------------------------------------------

/// Build weight maps for all 3 sub-defs of one decoder block.
pub fn build_decoder_block_weight_maps(
    block: &SpectralDecoderBlockWeights,
) -> SpectralDecoderBlockWeightMaps {
    // Rewrite sub-def weights.
    let mut rewrite = HashMap::new();
    rewrite.insert("rw_weight".to_string(), block.rewrite_weight.clone());
    rewrite.insert("rw_bias".to_string(), block.rewrite_bias.clone());

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

    // ConvTranspose1d sub-def weights.
    let mut conv_tr = HashMap::new();
    conv_tr.insert("ct_weight".to_string(), block.conv_tr_weight.clone());
    conv_tr.insert("ct_bias".to_string(), block.conv_tr_bias.clone());

    SpectralDecoderBlockWeightMaps {
        rewrite,
        dconv,
        conv_tr,
    }
}

#[cfg(test)]
#[path = "demucs_spectral_decoder_builders_tests.rs"]
mod tests;
