// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for LoRA (Low-Rank Adaptation) invariants.
//!
//! Proves scalar-level properties of LoRA's merge and forward computations:
//! - Scaling factor `alpha / rank` is finite for valid inputs
//! - Merge formula `w + scaling * dot(b_row, a_col)` produces finite output
//! - Two-matmul forward path is equivalent to merged weight path (scalar)
//! - Zero-initialized B produces exactly zero LoRA contribution
//!
//! Re: #13 (verified training epic).

#[cfg(kani)]
mod proofs {
    // ── Scalar LoRA formulas ─────────────────────────────────────────
    //
    // LoRA replaces W with W + (alpha/rank) * B @ A.
    // At the scalar level for a single output element:
    //
    //   merged_w_ij = w_ij + scaling * sum_k(b_ik * a_kj)
    //
    // The forward pass computes:
    //   base = sum_j(x_j * w_ji)           (one row of x @ W^T)
    //   lora = sum_j(x_j * sum_k(a_kj^T * b_ik^T)) * scaling
    //        = scaling * sum_k(b_ik * sum_j(x_j * a_kj))
    //
    // For rank=1 (scalar proof), this simplifies to:
    //   merged_w = w + scaling * b * a
    //   base_out = x * w
    //   lora_out = scaling * b * (x * a)

    fn assume_bounded(x: f32, lo: f32, hi: f32) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    fn assume_bounded_f64(x: f64, lo: f64, hi: f64) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    // ── Scaling factor proofs ────────────────────────────────────────

    /// LoRA scaling = alpha / rank is finite and positive for valid inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_lora_scaling_finite() {
        let alpha: f64 = kani::any();
        let rank: u32 = kani::any();
        assume_bounded_f64(alpha, 1e-6, 1e6);
        kani::assume(rank >= 1 && rank <= 1024);

        let scaling = alpha / rank as f64;
        assert!(scaling > 0.0, "scaling must be positive");
        assert!(!scaling.is_nan(), "scaling must not be NaN");
        assert!(!scaling.is_infinite(), "scaling must not be infinite");
    }

    /// LoRA scaling is bounded: scaling <= alpha (since rank >= 1).
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_lora_scaling_bounded() {
        let alpha: f64 = kani::any();
        let rank: u32 = kani::any();
        assume_bounded_f64(alpha, 1e-6, 1e6);
        kani::assume(rank >= 1 && rank <= 1024);

        let scaling = alpha / rank as f64;
        assert!(scaling <= alpha, "scaling must be <= alpha since rank >= 1");
    }

    /// Prove that validated scaling (within f32 range) survives the `as f32` cast finitely.
    /// This covers the gap where finite f64 scaling overflows f32.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_lora_scaling_f32_cast_safe() {
        let alpha: f64 = kani::any();
        let rank: u32 = kani::any();
        assume_bounded_f64(alpha, 1e-6, 1e6);
        kani::assume(rank >= 1 && rank <= 1024);

        let scaling = alpha / rank as f64;
        // After validation, the production code does `scaling as f32`
        let scaling_f32 = scaling as f32;
        assert!(
            scaling_f32.is_finite(),
            "scaling within validated range must survive f32 cast"
        );
    }

    // ── Merge formula proofs ─────────────────────────────────────────

    /// LoRA merge (rank=1 scalar): w + scaling * b * a is finite.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_lora_merge_scalar_finite() {
        let w: f32 = kani::any();
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        let scaling: f32 = kani::any();
        assume_bounded(w, -1e3, 1e3);
        assume_bounded(a, -1e3, 1e3);
        assume_bounded(b, -1e3, 1e3);
        assume_bounded(scaling, 1e-6, 1e3);

        let merged = w + scaling * b * a;
        assert!(!merged.is_nan(), "merged weight must not be NaN");
        assert!(!merged.is_infinite(), "merged weight must not be infinite");
    }

    /// LoRA merge (rank=2 dot product): w + scaling * (b0*a0 + b1*a1) is finite.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_lora_merge_rank2_finite() {
        let w: f32 = kani::any();
        let a0: f32 = kani::any();
        let a1: f32 = kani::any();
        let b0: f32 = kani::any();
        let b1: f32 = kani::any();
        let scaling: f32 = kani::any();
        assume_bounded(w, -1e3, 1e3);
        assume_bounded(a0, -1e3, 1e3);
        assume_bounded(a1, -1e3, 1e3);
        assume_bounded(b0, -1e3, 1e3);
        assume_bounded(b1, -1e3, 1e3);
        assume_bounded(scaling, 1e-6, 1e3);

        let dot = b0 * a0 + b1 * a1;
        let merged = w + scaling * dot;
        assert!(!merged.is_nan(), "merged weight must not be NaN");
        assert!(!merged.is_infinite(), "merged weight must not be infinite");
    }

    // ── Zero-init identity proof ─────────────────────────────────────

    /// When B is initialized to zeros, the LoRA contribution is exactly zero.
    /// This proves that a freshly-constructed LoRA layer is functionally
    /// identical to the original Linear layer.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_lora_zero_init_identity() {
        let x: f32 = kani::any();
        let w: f32 = kani::any();
        let a: f32 = kani::any();
        let scaling: f32 = kani::any();
        assume_bounded(x, -1e3, 1e3);
        assume_bounded(w, -1e3, 1e3);
        assume_bounded(a, -1e3, 1e3);
        assume_bounded(scaling, 1e-6, 1e3);

        let b: f32 = 0.0; // B is zero-initialized

        // Base output (frozen weight path)
        let base_out = x * w;
        // LoRA contribution
        let lora_out = scaling * b * (x * a);
        // Combined output
        let combined = base_out + lora_out;

        // lora_out must be zero since b=0 (IEEE 754: 0.0 * finite = ±0.0)
        assert!(
            lora_out == 0.0,
            "zero-init B must produce zero LoRA contribution"
        );
        // Combined must equal base in value (not bit-exact: ±0.0 have different
        // bits but adding ±0.0 to any value produces the same IEEE 754 value)
        assert!(
            combined == base_out,
            "zero-init LoRA must not change output"
        );
    }

    // ── Forward equivalence proof ────────────────────────────────────

    /// Two-matmul forward path produces same result as merged weight path.
    /// For rank=1 scalar case:
    ///   merged: y = x * (w + scaling * b * a)
    ///   decomposed: y = x * w + scaling * b * (x * a)
    /// These are algebraically identical; this proves it at the floating-point level.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_lora_forward_equivalence_scalar() {
        let x: f32 = kani::any();
        let w: f32 = kani::any();
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        let scaling: f32 = kani::any();
        // Use narrower range to avoid catastrophic cancellation
        assume_bounded(x, -10.0, 10.0);
        assume_bounded(w, -10.0, 10.0);
        assume_bounded(a, -10.0, 10.0);
        assume_bounded(b, -10.0, 10.0);
        assume_bounded(scaling, 0.1, 10.0);

        // Merged path: x * (w + scaling * b * a)
        let merged_w = w + scaling * b * a;
        let merged_out = x * merged_w;

        // Decomposed path: x * w + scaling * b * (x * a)
        let base = x * w;
        let lora = scaling * b * (x * a);
        let decomposed_out = base + lora;

        // Both must be finite
        assert!(!merged_out.is_nan() && !merged_out.is_infinite());
        assert!(!decomposed_out.is_nan() && !decomposed_out.is_infinite());

        // The two paths may differ by f32 rounding.
        // For |x|,|w|,|a|,|b| <= 10 and |scaling| <= 10:
        // Maximum product chain: 10 * (10 + 10 * 10 * 10) = 10 * 1010 = 10100
        // ULP at 10100 ~ 2^-10 ≈ 0.001, so 0.01 tolerance is conservative.
        let diff = (merged_out - decomposed_out).abs();
        assert!(
            diff <= 0.01,
            "merged and decomposed paths must agree within f32 rounding"
        );
    }

    // ── Rank-2 forward equivalence proof ─────────────────────────────

    /// Two-matmul forward equivalence for rank=2 with input dimension 2.
    ///
    /// For a single output element with in_features=2, rank=2:
    ///   merged: y = x0*(w0 + s*(b00*a00 + b01*a10)) + x1*(w1 + s*(b00*a01 + b01*a11))
    ///   decomposed: y = (x0*w0 + x1*w1) + s * (b00*(x0*a00 + x1*a01) + b01*(x0*a10 + x1*a11))
    ///
    /// These are algebraically identical; this proves it at the floating-point level.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_lora_forward_equivalence_rank2() {
        let x0: f32 = kani::any();
        let x1: f32 = kani::any();
        let w0: f32 = kani::any();
        let w1: f32 = kani::any();
        let a00: f32 = kani::any();
        let a01: f32 = kani::any();
        let a10: f32 = kani::any();
        let a11: f32 = kani::any();
        let b00: f32 = kani::any();
        let b01: f32 = kani::any();
        let scaling: f32 = kani::any();

        // Narrower range for rank-2: more multiplications accumulate error
        assume_bounded(x0, -5.0, 5.0);
        assume_bounded(x1, -5.0, 5.0);
        assume_bounded(w0, -5.0, 5.0);
        assume_bounded(w1, -5.0, 5.0);
        assume_bounded(a00, -5.0, 5.0);
        assume_bounded(a01, -5.0, 5.0);
        assume_bounded(a10, -5.0, 5.0);
        assume_bounded(a11, -5.0, 5.0);
        assume_bounded(b00, -5.0, 5.0);
        assume_bounded(b01, -5.0, 5.0);
        assume_bounded(scaling, 0.1, 5.0);

        // Merged path: y = sum_j x_j * (w_j + s * sum_k b_0k * a_kj)
        let merged_w0 = w0 + scaling * (b00 * a00 + b01 * a10);
        let merged_w1 = w1 + scaling * (b00 * a01 + b01 * a11);
        let merged_out = x0 * merged_w0 + x1 * merged_w1;

        // Decomposed path (two-matmul):
        // base = x0*w0 + x1*w1
        // lora_intermediate_k = sum_j x_j * a_kj  (x @ A^T, one per rank)
        // lora_out = s * sum_k b_0k * lora_intermediate_k
        let base = x0 * w0 + x1 * w1;
        let lora_int0 = x0 * a00 + x1 * a01; // rank-0 intermediate
        let lora_int1 = x0 * a10 + x1 * a11; // rank-1 intermediate
        let lora_out = scaling * (b00 * lora_int0 + b01 * lora_int1);
        let decomposed_out = base + lora_out;

        // Both must be finite
        assert!(!merged_out.is_nan() && !merged_out.is_infinite());
        assert!(!decomposed_out.is_nan() && !decomposed_out.is_infinite());

        // Tolerance: rank-2 with 2 inputs has deeper multiplication chains.
        // Max: 5*(5 + 5*(5*5 + 5*5)) = 5*255 = 1275, product ~6375
        // ULP at ~6000 ≈ 2^-10 ≈ 0.001. Multiple accumulations: 0.05 conservative.
        let diff = (merged_out - decomposed_out).abs();
        assert!(
            diff <= 0.05,
            "rank-2 merged and decomposed paths must agree within f32 rounding"
        );
    }

    // ── Negative alpha proof ─────────────────────────────────────────

    /// Negative alpha (used in concept unlearning) produces finite scaling and merge.
    ///
    /// LoRA with negative alpha reverses the learned direction: the adapter
    /// subtracts its contribution instead of adding it. This is valid for
    /// concept erasure and interference reduction.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_lora_negative_alpha_finite() {
        let alpha: f64 = kani::any();
        let rank: u32 = kani::any();
        assume_bounded_f64(alpha, -1e6, -1e-6); // negative alpha
        kani::assume(rank >= 1 && rank <= 1024);

        let scaling = alpha / rank as f64;
        assert!(
            scaling < 0.0,
            "negative alpha must produce negative scaling"
        );
        assert!(!scaling.is_nan(), "scaling must not be NaN");
        assert!(!scaling.is_infinite(), "scaling must not be infinite");

        // Verify merge is finite with negative scaling
        let w: f32 = kani::any();
        let b: f32 = kani::any();
        let a: f32 = kani::any();
        assume_bounded(w, -1e3, 1e3);
        assume_bounded(b, -1e3, 1e3);
        assume_bounded(a, -1e3, 1e3);

        let scaling_f32 = scaling as f32;
        kani::assume(scaling_f32.is_finite()); // validated at construction
        let merged = w + scaling_f32 * b * a;
        assert!(
            !merged.is_nan(),
            "merged weight with negative scaling must not be NaN"
        );
        assert!(
            !merged.is_infinite(),
            "merged weight with negative scaling must not be infinite"
        );
    }

    // ── Merge-then-freeze identity ───────────────────────────────────

    /// After merging LoRA into the base weight, applying a second LoRA with
    /// B=0 (freshly initialized) produces no additional change.
    ///
    /// This proves correctness of the merge-then-continue-training workflow:
    /// merge first adapter, initialize second adapter, output unchanged until
    /// second adapter is trained.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_lora_merge_then_reinit_identity() {
        let w: f32 = kani::any();
        let a1: f32 = kani::any();
        let b1: f32 = kani::any();
        let scaling1: f32 = kani::any();
        let a2: f32 = kani::any();
        let scaling2: f32 = kani::any();
        assume_bounded(w, -1e3, 1e3);
        assume_bounded(a1, -1e3, 1e3);
        assume_bounded(b1, -1e3, 1e3);
        assume_bounded(scaling1, 1e-6, 1e3);
        assume_bounded(a2, -1e3, 1e3);
        assume_bounded(scaling2, 1e-6, 1e3);

        let b2: f32 = 0.0; // second adapter zero-initialized

        // First merge: w_merged = w + scaling1 * b1 * a1
        let w_merged = w + scaling1 * b1 * a1;
        kani::assume(w_merged.is_finite()); // prior merge validated

        // Second LoRA contribution with zero B
        let second_lora = scaling2 * b2 * a2;

        // Second merge: w_final = w_merged + scaling2 * b2 * a2
        let w_final = w_merged + second_lora;

        assert!(
            second_lora == 0.0,
            "zero-init second LoRA must contribute nothing"
        );
        assert!(
            w_final.to_bits() == w_merged.to_bits(),
            "merge-then-reinit must not change merged weight"
        );
    }
}
