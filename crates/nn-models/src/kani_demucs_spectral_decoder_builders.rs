// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Demucs spectral decoder builder dimensional safety.
//!
//! These harnesses focus on the decoder builder arithmetic and validation
//! contracts that protect block construction:
//! 1. Decoder channel flow matches the depth schedule.
//! 2. A zero-filled decoder with correct shapes passes validation.
//! 3. A channel-count mismatch in ConvTranspose bias is rejected.
//! 4. Rewrite Conv2d same-padding preserves spatial extent.
//! 5. ConvTranspose stride/padding expands frequency and trim honours target_f.

#[cfg(kani)]
mod proofs {
    use crate::demucs_shared::{
        channels_at_depth, DCONV_COMPRESS, DCONV_DEPTH, DCONV_KERNEL, SPECTRAL_BASIC_DEPTH,
        SPECTRAL_CONV_TR_PADDING, SPECTRAL_KERNEL_SIZE, SPECTRAL_OUTPUT_CHANNELS,
        SPECTRAL_REWRITE_KERNEL, SPECTRAL_REWRITE_PADDING, SPECTRAL_STRIDE,
    };
    use crate::demucs_spectral_decoder_builders::{
        build_decoder_block_sub_defs, conv2d_output_len, validate_all_decoder_weights,
    };
    use crate::demucs_spectral_weights::{
        DemucsSpectralDecoderWeights, SpectralDConvSubLayerWeights, SpectralDecoderBlockWeights,
    };
    use crate::DemucsBuilderError;

    fn zero_dconv_sub(channels: usize, compressed: usize) -> SpectralDConvSubLayerWeights {
        SpectralDConvSubLayerWeights {
            conv_compress_weight: vec![0.0; compressed * channels * DCONV_KERNEL],
            conv_compress_bias: vec![0.0; compressed],
            norm_compress_gamma: vec![0.0; compressed],
            norm_compress_beta: vec![0.0; compressed],
            conv_expand_weight: vec![0.0; channels * 2 * compressed],
            conv_expand_bias: vec![0.0; channels * 2],
            norm_expand_gamma: vec![0.0; channels * 2],
            norm_expand_beta: vec![0.0; channels * 2],
            layer_scale: vec![0.0; channels],
        }
    }

    fn zero_decoder_block(in_ch: usize, out_ch: usize) -> SpectralDecoderBlockWeights {
        let compressed = in_ch / DCONV_COMPRESS;
        let dconv_sub = zero_dconv_sub(in_ch, compressed);

        SpectralDecoderBlockWeights {
            rewrite_weight: vec![
                0.0;
                in_ch
                    * 2
                    * in_ch
                    * SPECTRAL_REWRITE_KERNEL
                    * SPECTRAL_REWRITE_KERNEL
            ],
            rewrite_bias: vec![0.0; in_ch * 2],
            dconv: vec![dconv_sub; DCONV_DEPTH],
            conv_tr_weight: vec![0.0; in_ch * out_ch * SPECTRAL_KERNEL_SIZE],
            conv_tr_bias: vec![0.0; out_ch],
        }
    }

    fn valid_decoder_weights() -> DemucsSpectralDecoderWeights {
        let mut blocks = Vec::with_capacity(SPECTRAL_BASIC_DEPTH);

        for block_idx in 0..SPECTRAL_BASIC_DEPTH {
            let encoder_depth = SPECTRAL_BASIC_DEPTH - 1 - block_idx;
            let in_ch = channels_at_depth(encoder_depth);
            let out_ch = if encoder_depth == 0 {
                SPECTRAL_OUTPUT_CHANNELS
            } else {
                channels_at_depth(encoder_depth - 1)
            };
            blocks.push(zero_decoder_block(in_ch, out_ch));
        }

        DemucsSpectralDecoderWeights { blocks }
    }

    /// Proves that the decoder channel schedule matches the reverse encoder
    /// depth schedule used by the production builders.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn decoder_channel_schedule_matches_depth() {
        let block_idx: u8 = kani::any();
        kani::assume((block_idx as usize) < SPECTRAL_BASIC_DEPTH);

        let encoder_depth = SPECTRAL_BASIC_DEPTH - 1 - block_idx as usize;
        let in_ch = channels_at_depth(encoder_depth);
        let out_ch = if encoder_depth == 0 {
            SPECTRAL_OUTPUT_CHANNELS
        } else {
            channels_at_depth(encoder_depth - 1)
        };

        assert!(in_ch > 0);
        assert!(out_ch > 0);

        if encoder_depth == 0 {
            assert_eq!(
                out_ch, SPECTRAL_OUTPUT_CHANNELS,
                "the last decoder block must project to spectral output channels"
            );
        } else {
            assert_eq!(
                in_ch,
                out_ch * 2,
                "non-terminal decoder stages must halve channels on the way out"
            );
        }
    }

    /// Proves that a shape-correct zero-filled decoder satisfies
    /// `validate_all_decoder_weights`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn valid_decoder_weights_pass_validation() {
        let weights = valid_decoder_weights();
        assert!(
            validate_all_decoder_weights(&weights).is_ok(),
            "shape-correct decoder weights must pass validation"
        );
    }

    /// Proves that a single-output-channel mismatch in ConvTranspose bias is
    /// rejected by the weight validator.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn conv_transpose_bias_length_mismatch_is_rejected() {
        let block_idx: u8 = kani::any();
        kani::assume((block_idx as usize) < SPECTRAL_BASIC_DEPTH);

        let mut weights = valid_decoder_weights();
        let block = &mut weights.blocks[block_idx as usize];
        let old_len = block.conv_tr_bias.len();
        let _ = block.conv_tr_bias.pop();

        let err = validate_all_decoder_weights(&weights).unwrap_err();
        match err {
            DemucsBuilderError::WeightSize {
                expected, actual, ..
            } => {
                assert_eq!(expected, old_len);
                assert_eq!(actual + 1, expected);
            }
            other => panic!("expected WeightSize error, got {other:?}"),
        }
    }

    /// Proves that the rewrite Conv2d uses same-padding (`k=3, s=1, p=1`) and
    /// therefore preserves frequency/time extent.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn rewrite_conv2d_same_padding_preserves_extent() {
        let in_len: u8 = kani::any();
        kani::assume(in_len >= 1 && in_len <= 32);

        let out = conv2d_output_len(
            in_len as usize,
            SPECTRAL_REWRITE_KERNEL,
            1,
            SPECTRAL_REWRITE_PADDING,
        )
        .unwrap();

        assert_eq!(
            out, in_len as usize,
            "same-padding rewrite Conv2d must preserve spatial extent"
        );
    }

    /// Proves that the ConvTranspose stage expands frequency according to the
    /// configured stride/padding arithmetic and that the builder trims to the
    /// requested target frequency.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn conv_transpose_stride_padding_and_trim_match_target_frequency() {
        let block_idx: u8 = kani::any();
        kani::assume((block_idx as usize) < SPECTRAL_BASIC_DEPTH);

        let encoder_depth = SPECTRAL_BASIC_DEPTH - 1 - block_idx as usize;
        let in_ch = channels_at_depth(encoder_depth);
        let out_ch = if encoder_depth == 0 {
            SPECTRAL_OUTPUT_CHANNELS
        } else {
            channels_at_depth(encoder_depth - 1)
        };

        let f_in: u8 = kani::any();
        let t_in: u8 = kani::any();
        kani::assume(f_in >= 1 && f_in <= 8);
        kani::assume(t_in >= 1 && t_in <= 8);

        let f_in = f_in as usize;
        let t_in = t_in as usize;
        let rw_f_out =
            conv2d_output_len(f_in, SPECTRAL_REWRITE_KERNEL, 1, SPECTRAL_REWRITE_PADDING).unwrap();
        let rw_t_out =
            conv2d_output_len(t_in, SPECTRAL_REWRITE_KERNEL, 1, SPECTRAL_REWRITE_PADDING).unwrap();

        let ct_f_out =
            (rw_f_out - 1) * SPECTRAL_STRIDE + SPECTRAL_KERNEL_SIZE - 2 * SPECTRAL_CONV_TR_PADDING;
        assert_eq!(
            ct_f_out,
            rw_f_out * SPECTRAL_STRIDE,
            "the configured ConvTranspose arithmetic must expand frequency by the stride"
        );

        let target_f: u8 = kani::any();
        kani::assume(target_f >= 1);
        kani::assume((target_f as usize) <= ct_f_out);
        let target_f = target_f as usize;

        let sub_defs = build_decoder_block_sub_defs(
            block_idx as usize,
            in_ch,
            out_ch,
            f_in,
            t_in,
            rw_f_out,
            rw_t_out,
            target_f,
            encoder_depth == 0,
        )
        .expect("bounded spectral decoder block should build");

        let rewrite_shape = &sub_defs.rewrite_def.nodes[sub_defs.rewrite_def.output.index()].shape;
        let dconv_shape = &sub_defs.dconv_def.nodes[sub_defs.dconv_def.output.index()].shape;
        let conv_tr_shape = &sub_defs.conv_tr_def.nodes[sub_defs.conv_tr_def.output.index()].shape;

        assert_eq!(rewrite_shape[0], in_ch);
        assert_eq!(rewrite_shape[1], rw_f_out * rw_t_out);
        assert_eq!(dconv_shape[0], in_ch);
        assert_eq!(dconv_shape[1], rw_t_out);
        assert_eq!(conv_tr_shape[0], out_ch);
        assert_eq!(conv_tr_shape[1], target_f);
    }
}
