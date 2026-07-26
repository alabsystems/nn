// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Module-aligned Kani proof harnesses for `lora.rs`.
//!
//! These proofs target the scalar algebra behind LoRA initialization, merge,
//! and forward evaluation.

#[cfg(kani)]
mod proofs {
    fn assume_bounded(x: f32, lo: f32, hi: f32) {
        kani::assume(x.is_finite());
        kani::assume(x >= lo && x <= hi);
    }

    fn assume_bounded_f64(x: f64, lo: f64, hi: f64) {
        kani::assume(x.is_finite());
        kani::assume(x >= lo && x <= hi);
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_lora_scaling_cast_safe() {
        let alpha: f64 = kani::any();
        let rank: u32 = kani::any();

        assume_bounded_f64(alpha, 1e-6, 1e6);
        kani::assume((1..=1024).contains(&rank));

        let scaling = alpha / rank as f64;
        let scaling_f32 = scaling as f32;

        assert!(scaling.is_finite() && scaling > 0.0);
        assert!(scaling_f32.is_finite() && scaling_f32 > 0.0);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_lora_zero_init_merge_is_identity() {
        let frozen_weight: f32 = kani::any();
        let lora_a: f32 = kani::any();
        let scaling: f32 = kani::any();

        assume_bounded(frozen_weight, -1e4, 1e4);
        assume_bounded(lora_a, -1e4, 1e4);
        assume_bounded(scaling, 1e-6, 1e3);

        let lora_b = 0.0f32;
        let merged = frozen_weight + scaling * lora_b * lora_a;

        assert!(merged.is_finite());
        assert!(merged == frozen_weight);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_lora_forward_matches_merged_rank1() {
        let x: f32 = kani::any();
        let frozen_weight: f32 = kani::any();
        let lora_a: f32 = kani::any();
        let lora_b: f32 = kani::any();
        let scaling: f32 = kani::any();

        assume_bounded(x, -10.0, 10.0);
        assume_bounded(frozen_weight, -10.0, 10.0);
        assume_bounded(lora_a, -10.0, 10.0);
        assume_bounded(lora_b, -10.0, 10.0);
        assume_bounded(scaling, 0.1, 10.0);

        let merged_out = x * (frozen_weight + scaling * lora_b * lora_a);
        let decomposed_out = x * frozen_weight + scaling * lora_b * (x * lora_a);

        assert!(merged_out.is_finite());
        assert!(decomposed_out.is_finite());
        assert!((merged_out - decomposed_out).abs() <= 0.01);
    }
}
