// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Demucs spectral decoder builder shape arithmetic.

#[cfg(kani)]
mod proofs {
    use crate::demucs_shared::{
        channels_at_depth, SPECTRAL_BASIC_DEPTH, SPECTRAL_CONV_TR_PADDING, SPECTRAL_KERNEL_SIZE,
        SPECTRAL_OUTPUT_CHANNELS, SPECTRAL_REWRITE_KERNEL, SPECTRAL_REWRITE_PADDING,
        SPECTRAL_STRIDE,
    };
    use crate::demucs_spectral_decoder_builders::conv2d_output_len;

    /// The spectral rewrite Conv2d uses same-padding, so it preserves frequency
    /// and time dimensions for any bounded input shape.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn spectral_rewrite_same_padding_preserves_shape() {
        let f_in: usize = kani::any();
        let t_in: usize = kani::any();
        kani::assume(f_in >= 1 && f_in <= 128);
        kani::assume(t_in >= 1 && t_in <= 128);

        let f_out =
            conv2d_output_len(f_in, SPECTRAL_REWRITE_KERNEL, 1, SPECTRAL_REWRITE_PADDING).unwrap();
        let t_out =
            conv2d_output_len(t_in, SPECTRAL_REWRITE_KERNEL, 1, SPECTRAL_REWRITE_PADDING).unwrap();

        assert_eq!(f_out, f_in);
        assert_eq!(t_out, t_in);
    }

    /// The ConvTranspose1d frequency formula matches a 4x upsample before the
    /// optional trim to `target_f`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn spectral_conv_transpose_frequency_formula_matches_trim() {
        let f_in: usize = kani::any();
        kani::assume(f_in >= 1 && f_in <= 64);

        let ct_f_out =
            (f_in - 1) * SPECTRAL_STRIDE + SPECTRAL_KERNEL_SIZE - 2 * SPECTRAL_CONV_TR_PADDING;

        let target_f: usize = kani::any();
        kani::assume(target_f >= 1 && target_f <= ct_f_out + SPECTRAL_STRIDE);

        let final_f = if ct_f_out > target_f {
            target_f
        } else {
            ct_f_out
        };

        assert_eq!(ct_f_out, f_in * SPECTRAL_STRIDE);
        assert_eq!(final_f, target_f.min(ct_f_out));
    }

    /// Decoder block channel counts follow the encoder-depth schedule:
    /// deeper blocks halve channels until the last block emits the fixed
    /// spectral output width.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn spectral_decoder_channel_schedule_matches_depth() {
        let block_idx: usize = kani::any();
        kani::assume(block_idx < SPECTRAL_BASIC_DEPTH);

        let encoder_depth = SPECTRAL_BASIC_DEPTH - 1 - block_idx;
        let in_ch = channels_at_depth(encoder_depth);
        let out_ch = if encoder_depth == 0 {
            SPECTRAL_OUTPUT_CHANNELS
        } else {
            channels_at_depth(encoder_depth - 1)
        };

        assert!(in_ch >= out_ch);
        if encoder_depth == 0 {
            assert_eq!(out_ch, SPECTRAL_OUTPUT_CHANNELS);
        } else {
            assert_eq!(in_ch, out_ch * 2);
        }
    }
}
