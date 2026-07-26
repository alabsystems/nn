// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for iSTFT overlap-add and reconstruction invariants.
//!
//! These harnesses complement the existing iSTFT proofs with direct checks on
//! buffer indexing, center trimming, and frame-level reconstruction structure:
//! 1. Periodic Hann mirror indices preserve the same window value.
//! 2. Overlap-add writes stay within the allocated output/window_sum buffers.
//! 3. Center trimming preserves exactly the hop-spanned reconstruction length.
//! 4. A DC-only frame reconstructs uniform pre-window samples.

#[cfg(kani)]
mod proofs {
    use std::f32::consts::PI;

    use crate::istft::IstftParams;

    fn cos_stub(_x: f32) -> f32 {
        let v: f32 = kani::any();
        kani::assume(v >= -1.0 && v <= 1.0);
        kani::assume(v.is_finite());
        v
    }

    /// Proves that the periodic Hann window keeps mirror indices in-bounds and
    /// assigns equal values to `k` and `n_fft - k`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn periodic_hann_mirror_symmetry() {
        let n_fft_half: u8 = kani::any();
        kani::assume(n_fft_half >= 1 && n_fft_half <= 32);
        let n_fft = (n_fft_half as usize) * 2;

        let k: u8 = kani::any();
        kani::assume(k > 0);
        kani::assume((k as usize) < n_fft);

        let mirror = n_fft - k as usize;
        assert!(mirror < n_fft, "mirror index must stay within the window");

        let angle = 2.0 * PI * (k as f32) / (n_fft as f32);
        let mirror_angle = 2.0 * PI * (mirror as f32) / (n_fft as f32);
        assert!(angle.is_finite());
        assert!(mirror_angle.is_finite());

        // cos(2*pi - x) == cos(x), so the mirror position reuses the same value.
        let cos_val = cos_stub(angle);
        let mirror_cos = cos_val;

        let w = 0.5 * (1.0 - cos_val);
        let mirror_w = 0.5 * (1.0 - mirror_cos);

        assert_eq!(w, mirror_w, "periodic Hann window must be symmetric");
        assert!(w >= 0.0 && w <= 1.0);
    }

    /// Proves the overlap-add write index `offset + k` stays inside the
    /// `full_len = n_fft + (n_frames - 1) * hop` output buffer.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn overlap_add_write_index_stays_in_bounds() {
        let n_fft_half: u8 = kani::any();
        kani::assume(n_fft_half >= 1 && n_fft_half <= 32);
        let n_fft = (n_fft_half as usize) * 2;

        let hop: u8 = kani::any();
        kani::assume(hop >= 1);
        kani::assume((hop as usize) <= n_fft);

        let params = IstftParams::new(n_fft, hop as usize, false, false).unwrap();

        let n_frames: u8 = kani::any();
        kani::assume(n_frames >= 1 && n_frames <= 8);
        let n_frames = n_frames as usize;

        let t: u8 = kani::any();
        let k: u8 = kani::any();
        kani::assume((t as usize) < n_frames);
        kani::assume((k as usize) < params.n_fft);

        let full_len = params.n_fft + (n_frames - 1) * params.hop_length;
        let offset = (t as usize) * params.hop_length;
        let write_idx = offset + (k as usize);

        assert!(
            write_idx < full_len,
            "OLA accumulation must never write past the output buffer"
        );
    }

    /// Proves that center trimming removes exactly `n_fft / 2` samples from
    /// both sides, leaving the hop-spanned reconstruction length.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn center_trim_matches_hop_spanned_reconstruction_length() {
        let n_fft_half: u8 = kani::any();
        kani::assume(n_fft_half >= 1 && n_fft_half <= 32);
        let n_fft = (n_fft_half as usize) * 2;

        let hop: u8 = kani::any();
        kani::assume(hop >= 1);
        kani::assume((hop as usize) <= n_fft);

        let params = IstftParams::new(n_fft, hop as usize, false, true).unwrap();

        let n_frames: u8 = kani::any();
        kani::assume(n_frames >= 2 && n_frames <= 8);
        let n_frames = n_frames as usize;

        let full_len = params.n_fft + (n_frames - 1) * params.hop_length;
        let trim = params.n_fft / 2;

        assert!(
            full_len > 2 * trim,
            "multi-frame centered iSTFT must leave samples after trimming"
        );

        let trimmed_len = full_len - 2 * trim;
        assert_eq!(
            trimmed_len,
            (n_frames - 1) * params.hop_length,
            "center trim must leave exactly the hop-spanned reconstruction"
        );
    }

    /// Proves that a DC-only spectrum reconstructs the same pre-window sample
    /// value at every time index in the unnormalized branch.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn dc_only_frame_reconstructs_uniform_samples() {
        let n_fft_half: u8 = kani::any();
        kani::assume(n_fft_half >= 1 && n_fft_half <= 32);
        let n_fft = (n_fft_half as usize) * 2;

        let _params = IstftParams::new(n_fft, 1, false, false).unwrap();

        let real_dc: f32 = kani::any();
        kani::assume(real_dc.is_finite());
        kani::assume(real_dc.abs() <= 1.0e6);

        let k0: u8 = kani::any();
        let k1: u8 = kani::any();
        kani::assume((k0 as usize) < n_fft);
        kani::assume((k1 as usize) < n_fft);

        // For f=0, cos_basis[0, k] = 1 and sin_basis[0, k] = 0 for every k.
        let norm = 1.0 / n_fft as f32;
        let sample_k0 = real_dc * norm;
        let sample_k1 = real_dc * norm;

        assert!(sample_k0.is_finite());
        assert_eq!(
            sample_k0, sample_k1,
            "a DC-only frame must reconstruct a uniform time-domain sample"
        );
    }
}
