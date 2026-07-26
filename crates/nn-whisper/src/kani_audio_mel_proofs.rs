// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Whisper mel-spectrogram validation.
//!
//! Covers:
//! - `pcm_to_mel()` rejects zero `n_fft`
//! - `pcm_to_mel()` rejects zero `hop_length`
//! - Mel filterbank length matches `n_mels * (n_fft / 2 + 1)`
//! - Bounded hop lengths produce the expected mel shape
//!
//! Issue: #3724

#[cfg(kani)]
mod proofs {
    use crate::audio::{mel_filterbank, pcm_to_mel};

    fn zero_audio(len: usize) -> Vec<f32> {
        vec![0.0f32; len]
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn pcm_to_mel_rejects_zero_fft_size() {
        let audio_len: usize = kani::any();
        let hop_length: usize = kani::any();
        let n_mels: usize = kani::any();

        kani::assume(audio_len >= 1 && audio_len <= 4);
        kani::assume(hop_length >= 1 && hop_length <= 4);
        kani::assume(n_mels >= 1 && n_mels <= 3);

        let audio = zero_audio(audio_len);
        let mel_filters = vec![0.0f32; n_mels];

        let result = pcm_to_mel(&audio, &mel_filters, 0, hop_length, n_mels);
        assert!(result.is_err(), "n_fft=0 must be rejected");
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn pcm_to_mel_rejects_zero_hop_length() {
        let audio_len: usize = kani::any();
        let n_mels: usize = kani::any();

        kani::assume(audio_len >= 1 && audio_len <= 4);
        kani::assume(n_mels >= 1 && n_mels <= 3);

        let n_fft = 4;
        let audio = zero_audio(audio_len);
        let mel_filters = mel_filterbank(n_mels, n_fft, 16_000);

        let result = pcm_to_mel(&audio, &mel_filters, n_fft, 0, n_mels);
        assert!(result.is_err(), "hop_length=0 must be rejected");
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn pcm_to_mel_rejects_wrong_frequency_bin_count() {
        let audio_len: usize = kani::any();
        let use_short_filterbank: bool = kani::any();

        kani::assume(audio_len >= 1 && audio_len <= 4);

        let n_fft = 6;
        let n_mels = 2;
        let expected_bins = n_mels * (n_fft / 2 + 1);
        let bad_len = if use_short_filterbank {
            expected_bins - 1
        } else {
            expected_bins + 1
        };

        let audio = zero_audio(audio_len);
        let mel_filters = vec![0.0f32; bad_len];

        let result = pcm_to_mel(&audio, &mel_filters, n_fft, 1, n_mels);
        assert!(
            result.is_err(),
            "mel_filters length must track n_fft/2+1 frequency bins"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(32)]
    fn bounded_hop_length_produces_expected_mel_shape() {
        let audio_len: usize = kani::any();
        let hop_length: usize = kani::any();
        let n_mels: usize = kani::any();

        kani::assume(audio_len >= 1 && audio_len <= 6);
        kani::assume(hop_length >= 1 && hop_length <= audio_len);
        kani::assume(n_mels == 1 || n_mels == 2);

        let n_fft = 4;
        let expected_freq_bins = n_fft / 2 + 1;
        let expected_frames = audio_len / hop_length + 1;
        let audio = zero_audio(audio_len);
        let mel_filters = mel_filterbank(n_mels, n_fft, 16_000);

        let mel = pcm_to_mel(&audio, &mel_filters, n_fft, hop_length, n_mels)
            .expect("bounded n_fft/hop_length should succeed");

        assert_eq!(
            mel_filters.len(),
            n_mels * expected_freq_bins,
            "mel filterbank must track FFT frequency bins"
        );
        assert_eq!(
            mel.dims(),
            &[1, n_mels, expected_frames],
            "mel tensor shape must follow [1, n_mels, frames]"
        );
        assert!(
            expected_frames >= 2,
            "valid bounded hops produce at least two frames"
        );
    }
}
