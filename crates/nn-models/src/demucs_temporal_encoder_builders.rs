// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder functions for Demucs temporal encoder `TensorKernelDef`s.
//!
//! Each encoder block is built as a single `TensorKernelDef` containing all
//! operations inlined: Conv1d -> GELU -> DConv(x2) -> Rewrite(Conv1d k=1) -> GLU.
//! This minimizes CPU round-trips (4 per forward pass).
//!
//! Extracted from nn-metal as part of #860.

use std::collections::HashMap;

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;

use crate::demucs_shared::{
    build_dconv_sublayer, conv1d_output_len, DConvSubLayerInputs, DCONV_COMPRESS, DCONV_DEPTH,
    GROUP_NORM_EPS, TEMPORAL_CONV_PADDING, TEMPORAL_KERNEL_SIZE, TEMPORAL_STRIDE,
};
use crate::demucs_temporal_weights::EncoderBlockWeights;
use crate::DemucsBuilderError;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Conv1d output length for the encoder's main convolution.
pub fn conv1d_out_len(padded_t: usize) -> Result<usize, DemucsBuilderError> {
    conv1d_output_len(
        padded_t,
        TEMPORAL_KERNEL_SIZE,
        TEMPORAL_STRIDE,
        TEMPORAL_CONV_PADDING,
    )
}

// ---------------------------------------------------------------------------
// TensorKernelDef builder
// ---------------------------------------------------------------------------

/// Build a single encoder block as one `TensorKernelDef`.
///
/// Variable input: "data" [in_ch, padded_t].
/// Constant inputs: conv, dconv, and rewrite weight/bias tensors.
///
/// The stride padding is done on CPU before dispatch; `padded_t` is the
/// already-padded temporal dimension.
pub fn build_encoder_block_def(
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    padded_t: usize,
) -> Result<TensorKernelDef, DemucsBuilderError> {
    let name = format!("demucs_enc_block{block_idx}");
    let compressed = out_ch / DCONV_COMPRESS;
    let doubled = out_ch * 2;

    let mut b = TensorBlockBuilder::new(&name);

    // --- Variable input ---
    let data = b.add_input(nn_dsl::input_names::DATA, &[in_ch, padded_t]);

    // --- Conv1d inputs ---
    let conv_weight = b.add_input("conv_weight", &[out_ch, in_ch, TEMPORAL_KERNEL_SIZE]);
    let conv_bias = b.add_input("conv_bias", &[out_ch]);

    // --- DConv inputs (2 sub-layers) ---
    let mut dconv_inputs = Vec::with_capacity(DCONV_DEPTH);
    for k in 0..DCONV_DEPTH {
        let di = DConvSubLayerInputs::add_to_builder(&mut b, k, out_ch, compressed);
        dconv_inputs.push(di);
    }

    // --- Rewrite inputs ---
    let rw_weight = b.add_input("rw_weight", &[doubled, out_ch, 1]);
    let rw_bias = b.add_input("rw_bias", &[doubled]);

    // --- Step 1: Conv1d (downsample) ---
    let conv_t_out = conv1d_output_len(
        padded_t,
        TEMPORAL_KERNEL_SIZE,
        TEMPORAL_STRIDE,
        TEMPORAL_CONV_PADDING,
    )?;
    let conv_out = b.add_conv1d(
        data,
        conv_weight,
        Some(conv_bias),
        TEMPORAL_STRIDE,
        TEMPORAL_CONV_PADDING,
        &[out_ch, conv_t_out],
    );

    // --- Step 2: GELU (between conv and DConv, matching Python) ---
    let gelu_out = b.add_gelu(conv_out, &[out_ch, conv_t_out]);

    // --- Step 3: DConv (2 residual sub-layers) ---
    let mut dconv_out = gelu_out;
    for di in &dconv_inputs {
        dconv_out = build_dconv_sublayer(&mut b, dconv_out, di, out_ch, compressed, conv_t_out)?;
    }

    // --- Step 4: Rewrite Conv1d(k=1) -> GLU ---
    // Conv1d(out_ch -> doubled, k=1, s=1, p=0) preserves time dimension.
    let rw_out = b.add_conv1d(
        dconv_out,
        rw_weight,
        Some(rw_bias),
        1,
        0,
        &[doubled, conv_t_out],
    );
    // GLU halves channels: [doubled, T] -> [out_ch, T]
    let output = b.add_glu(rw_out, 0, &[doubled, conv_t_out])?;

    Ok(b.build(output)?)
}

// ---------------------------------------------------------------------------
// Weight map builder
// ---------------------------------------------------------------------------

/// Build a HashMap of named weight tensors matching the input names used in
/// `build_encoder_block_def`.
pub fn build_encoder_weight_map(block: &EncoderBlockWeights) -> HashMap<String, Vec<f32>> {
    let mut map = HashMap::new();

    map.insert("conv_weight".to_string(), block.conv_weight.clone());
    map.insert("conv_bias".to_string(), block.conv_bias.clone());

    for (k, sub) in block.dconv.iter().enumerate() {
        map.insert(format!("dc{k}_cw"), sub.conv_compress_weight.clone());
        map.insert(format!("dc{k}_cb"), sub.conv_compress_bias.clone());
        map.insert(format!("dc{k}_ng"), sub.norm_compress_gamma.clone());
        map.insert(format!("dc{k}_nb"), sub.norm_compress_beta.clone());
        map.insert(format!("dc{k}_ew"), sub.conv_expand_weight.clone());
        map.insert(format!("dc{k}_eb"), sub.conv_expand_bias.clone());
        map.insert(format!("dc{k}_eng"), sub.norm_expand_gamma.clone());
        map.insert(format!("dc{k}_enb"), sub.norm_expand_beta.clone());
        map.insert(format!("dc{k}_ls"), sub.layer_scale.clone());
        map.insert(format!("dc{k}_eps"), vec![GROUP_NORM_EPS]);
        map.insert(format!("dc{k}_eps2"), vec![GROUP_NORM_EPS]);
    }

    map.insert("rw_weight".to_string(), block.rewrite_weight.clone());
    map.insert("rw_bias".to_string(), block.rewrite_bias.clone());

    map
}

#[cfg(test)]
#[path = "demucs_temporal_encoder_builders_tests.rs"]
mod tests;
