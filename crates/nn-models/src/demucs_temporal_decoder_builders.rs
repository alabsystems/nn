// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder functions for Demucs temporal decoder `TensorKernelDef`s.
//!
//! Each decoder block is built as a single `TensorKernelDef` containing all
//! operations inlined: skip_add -> Rewrite -> GLU -> DConv(x2) -> ConvTranspose1d
//! -> trim -> [GELU]. This minimizes CPU round-trips (4 per forward pass).
//!
//! Extracted from nn-metal as part of #860.

use std::collections::HashMap;

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;

use crate::demucs_shared::{
    build_dconv_sublayer, conv1d_output_len, DConvSubLayerInputs, DCONV_COMPRESS, DCONV_DEPTH,
    DECODER_REWRITE_KERNEL, DECODER_REWRITE_PADDING, GROUP_NORM_EPS, TEMPORAL_CONV_TR_PADDING,
    TEMPORAL_KERNEL_SIZE, TEMPORAL_STRIDE,
};
use crate::demucs_temporal_weights::DecoderBlockWeights;
use crate::DemucsBuilderError;

// Re-export for callers that need conv1d_output_len.
pub use crate::demucs_shared::conv1d_output_len as decoder_conv1d_output_len;

// ---------------------------------------------------------------------------
// TensorKernelDef builder
// ---------------------------------------------------------------------------

/// Build a single decoder block as one `TensorKernelDef`.
///
/// Variable inputs: "data" [in_ch, t_in], "skip" [in_ch, t_in].
/// Constant inputs: rewrite, dconv, and conv_tr weight/bias tensors.
pub fn build_decoder_block_def(
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    t_in: usize,
    target_len: usize,
    is_last: bool,
) -> Result<TensorKernelDef, DemucsBuilderError> {
    let name = format!("demucs_dec_block{block_idx}");
    let compressed = in_ch / DCONV_COMPRESS;
    let doubled = in_ch * 2;

    let mut b = TensorBlockBuilder::new(&name);

    // --- Variable inputs ---
    let data = b.add_input(nn_dsl::input_names::DATA, &[in_ch, t_in]);
    let skip = b.add_input(nn_dsl::input_names::SKIP, &[in_ch, t_in]);

    // --- Rewrite inputs ---
    let rw_weight = b.add_input("rw_weight", &[doubled, in_ch, DECODER_REWRITE_KERNEL]);
    let rw_bias = b.add_input("rw_bias", &[doubled]);

    // --- DConv inputs (2 sub-layers) ---
    let mut dconv_inputs = Vec::with_capacity(DCONV_DEPTH);
    for k in 0..DCONV_DEPTH {
        let di = DConvSubLayerInputs::add_to_builder(&mut b, k, in_ch, compressed);
        dconv_inputs.push(di);
    }

    // --- ConvTranspose1d inputs ---
    let ct_weight = b.add_input("ct_weight", &[in_ch, out_ch, TEMPORAL_KERNEL_SIZE]);
    let ct_bias = b.add_input("ct_bias", &[out_ch]);

    // --- Step 1: Skip connection add ---
    let x = b.add_binary_add(data, skip, &[in_ch, t_in]);

    // --- Step 2: Rewrite Conv1d -> GLU ---
    // Conv1d(in_ch -> doubled, k=3, s=1, p=1) preserves time dimension.
    let rw_t_out = conv1d_output_len(t_in, DECODER_REWRITE_KERNEL, 1, DECODER_REWRITE_PADDING)?;
    let rw_out = b.add_conv1d(
        x,
        rw_weight,
        Some(rw_bias),
        1,
        DECODER_REWRITE_PADDING,
        &[doubled, rw_t_out],
    );
    // GLU halves channels: [doubled, T] -> [in_ch, T]
    let glu_out = b.add_glu(rw_out, 0, &[doubled, rw_t_out])?;

    // --- Step 3: DConv (2 residual sub-layers) ---
    let mut dconv_out = glu_out;
    for di in &dconv_inputs {
        dconv_out = build_dconv_sublayer(&mut b, dconv_out, di, in_ch, compressed, rw_t_out)?;
    }

    // --- Step 4: ConvTranspose1d (upsample) ---
    let ct_t_out =
        (rw_t_out - 1) * TEMPORAL_STRIDE + TEMPORAL_KERNEL_SIZE - 2 * TEMPORAL_CONV_TR_PADDING;
    let ct_out = b.add_conv_transpose_1d(
        dconv_out,
        ct_weight,
        Some(ct_bias),
        TEMPORAL_STRIDE,
        TEMPORAL_CONV_TR_PADDING,
        1, // dilation
        1, // groups
        0, // output_padding
        &[out_ch, ct_t_out],
    );

    // --- Step 5: Trim to target length ---
    let trimmed = if ct_t_out > target_len {
        b.add_narrow(ct_out, 1, 0, target_len, &[out_ch, target_len])
    } else {
        ct_out
    };

    // --- Step 6: GELU (non-last blocks only) ---
    let output = if is_last {
        trimmed
    } else {
        b.add_gelu(trimmed, &[out_ch, target_len])
    };

    Ok(b.build(output)?)
}

// ---------------------------------------------------------------------------
// Weight map builder
// ---------------------------------------------------------------------------

/// Build a HashMap of named weight tensors matching the input names used in
/// `build_decoder_block_def`.
pub fn build_decoder_weight_map(block: &DecoderBlockWeights) -> HashMap<String, Vec<f32>> {
    let mut map = HashMap::new();

    map.insert("rw_weight".to_string(), block.rewrite_weight.clone());
    map.insert("rw_bias".to_string(), block.rewrite_bias.clone());

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

    map.insert("ct_weight".to_string(), block.conv_tr_weight.clone());
    map.insert("ct_bias".to_string(), block.conv_tr_bias.clone());

    map
}

#[cfg(test)]
#[path = "demucs_temporal_decoder_builders_tests.rs"]
mod tests;
