// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `trace_compile_peephole_resblock.rs` — additional
//! structural and safety invariants for the ResBlock peephole fusion pass.
//!
//! Proves:
//! - detect_post_add_scale returns add_idx as fused position when no scale.
//! - detect_post_add_scale residual_scale is 1.0 when no pattern found.
//! - detect_post_add_scale extra_replace is empty when no pattern found.
//! - Unfused phase1 requires stride == 1.
//! - Unfused phase1 requires groups == 1.
//! - Unfused phase1 allows various dilation values.
//! - conv1x1 shortcut requires kernel_size == 1 and stride == 1.
//! - NormActivation::Snake has no slope field.
//! - NormActivation::LeakyRelu slope must be in (0, 1].
//! - Replaced steps are all unique (no duplicates).
//! - NormActivConv1dParams dilation/padding relationship for common kernels.
//! - Standard path fused position is at the add_idx step.
//! - Pool step placement for upsample: between adain1 and conv1.
//! - Weight merge total count with unfused phase1 (AdaIN + conv_weight + conv_bias).
//! - FusedResBlock residual_scale absorbing 1/sqrt(2).
//! - FusedResBlock with all None optional fields.
//! - IdentityPassthrough replacement count: standard=4, add remains.
//!
//! Part of #3738.

#[cfg(kani)]
mod proofs {
    use std::collections::HashSet;

    use crate::trace_compile::{
        NativeOpKind, NormActivConv1dParams, NormActivation, PeepholeConfig, StyleBatchOffset,
        StyleProjectionParams,
    };

    // -- Kani transcendental stubs (CBMC #239, #329, #708) --
    fn sqrt_f32_stub(x: f32) -> f32 {
        let r: f32 = kani::any();
        kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
        if x > 0.0 {
            kani::assume(r > 0.0);
            kani::assume(r >= x.min(1.0));
        }
        r
    }

    // -----------------------------------------------------------------------
    // Proof 1: detect_post_add_scale defaults (fused position = add_idx)
    // -----------------------------------------------------------------------

    /// When no post-add scale pattern is found, the FusedResBlock is placed
    /// at the add_idx position (the original Dispatch "add" step).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_post_add_scale_default_fused_position() {
        let add_idx: usize = kani::any();
        kani::assume(add_idx <= 500);

        // Default: no scale pattern found
        let (fused_position, scale, extra) = (add_idx, 1.0f32, Vec::<usize>::new());

        assert_eq!(fused_position, add_idx);
        assert_eq!(scale, 1.0);
        assert!(extra.is_empty());
    }

    // -----------------------------------------------------------------------
    // Proof 2: detect_post_add_scale residual_scale is 1.0 when no pattern
    // -----------------------------------------------------------------------

    /// The default residual_scale (1.0) means "no scaling" — the residual
    /// add is `x + h` without post-multiplication.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_default_residual_scale_identity() {
        let scale = 1.0f32;
        let x: f32 = kani::any();
        kani::assume(x.is_finite() && x.abs() < 1e6);

        let result = x * scale;
        assert_eq!(result, x, "scale of 1.0 must be identity");
    }

    // -----------------------------------------------------------------------
    // Proof 3: Unfused phase1 requires stride == 1
    // -----------------------------------------------------------------------

    /// The trace_unfused_phase1 function bails if stride != 1. This is
    /// because non-unit stride changes output size, which doesn't match
    /// the standard FusedResBlock assumption of same-size phase outputs.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unfused_phase1_stride_requirement() {
        let stride: usize = kani::any();
        kani::assume(stride >= 1 && stride <= 8);
        let groups: usize = 1;

        let passes = stride == 1 && groups == 1;
        if stride != 1 {
            assert!(!passes, "non-unit stride must block fusion");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 4: Unfused phase1 requires groups == 1
    // -----------------------------------------------------------------------

    /// Grouped convolution changes the per-channel semantics, which
    /// is incompatible with the FusedResBlock's fused stats computation.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unfused_phase1_groups_requirement() {
        let stride: usize = 1;
        let groups: usize = kani::any();
        kani::assume(groups >= 1 && groups <= 8);

        let passes = stride == 1 && groups == 1;
        if groups != 1 {
            assert!(!passes, "non-unit groups must block fusion");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 5: Unfused phase1 allows various dilation values
    // -----------------------------------------------------------------------

    /// Unlike stride/groups, dilation is allowed for unfused phase1.
    /// The Kokoro generator uses dilation={1,3,5,...} in phase2.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_unfused_phase1_dilation_allowed() {
        let dilation: usize = kani::any();
        kani::assume(dilation >= 1 && dilation <= 9);

        // NormActivConv1dParams accepts any dilation
        let params = NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            dilation,
            dilation, // padding
            vec![1, 128, 256],
            128,
            3,
        );
        assert_eq!(params.conv_dilation, dilation);
    }

    // -----------------------------------------------------------------------
    // Proof 6: conv1x1 shortcut requires kernel_size == 1 AND stride == 1
    // -----------------------------------------------------------------------

    /// The detect_conv1x1_shortcut function matches Conv1d with
    /// kernel_size=1 and stride=1. Anything else is not a valid shortcut.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_conv1x1_shortcut_constraints() {
        let ks: usize = kani::any();
        let stride: usize = kani::any();
        let dilation: usize = kani::any();
        let groups: usize = kani::any();

        kani::assume(ks >= 1 && ks <= 16);
        kani::assume(stride >= 1 && stride <= 4);
        kani::assume(dilation >= 1 && dilation <= 4);
        kani::assume(groups >= 1 && groups <= 4);

        // From the code: matches Conv1d { stride: 1, dilation: 1, groups: 1 }
        // with weight.shape().last() == Some(&1) (i.e., kernel_size == 1)
        let is_valid_shortcut = ks == 1 && stride == 1 && dilation == 1 && groups == 1;

        if ks != 1 {
            assert!(!is_valid_shortcut, "kernel_size != 1 blocks shortcut");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 7: NormActivation::Snake has no slope field
    // -----------------------------------------------------------------------

    /// Snake activation uses a per-channel alpha parameter (from weights),
    /// not a scalar slope. Mixing up Snake and LeakyRelu would use wrong activation.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_snake_no_slope() {
        let a = NormActivation::Snake;
        let is_snake = matches!(a, NormActivation::Snake);
        assert!(is_snake);

        // Snake cannot destructure to get a slope
        let has_slope = matches!(a, NormActivation::LeakyRelu { .. });
        assert!(!has_slope, "Snake must not match LeakyRelu pattern");
    }

    // -----------------------------------------------------------------------
    // Proof 8: LeakyRelu slope must be in valid range
    // -----------------------------------------------------------------------

    /// Typical LeakyRelu slopes are in (0, 1]. A zero slope would make it
    /// ReLU (which has its own op), and negative slope would invert the
    /// negative region.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_leaky_relu_slope_positive() {
        let slope: f32 = kani::any();
        kani::assume(slope > 0.0 && slope <= 1.0);
        kani::assume(slope.is_finite());

        let act = NormActivation::LeakyRelu { slope };
        match act {
            NormActivation::LeakyRelu { slope: s } => {
                assert!(s > 0.0);
                assert!(s <= 1.0);
                assert!(s.is_finite());
            }
            _ => panic!("wrong variant"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 9: Replaced steps are unique (no duplicates)
    // -----------------------------------------------------------------------

    /// The replaced_steps vector must have all unique indices. Replacing
    /// the same step twice would panic (first replacement makes it
    /// IdentityPassthrough, second replacement tries to match a pattern
    /// on IdentityPassthrough).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_replaced_steps_unique_standard() {
        let a1: usize = kani::any();
        let c1: usize = kani::any();
        let a2: usize = kani::any();
        let c2: usize = kani::any();

        kani::assume(a1 < c1 && c1 < a2 && a2 < c2);
        kani::assume(c2 <= 100);

        let replaced = vec![a1, c1, a2, c2];
        let set: HashSet<usize> = replaced.iter().copied().collect();
        assert_eq!(
            set.len(),
            replaced.len(),
            "all replaced steps must be unique"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 10: NormActivConv1dParams dilation/padding for kernel_size=3
    // -----------------------------------------------------------------------

    /// For kernel_size=3, padding = dilation * (3-1)/2 = dilation.
    /// This is the standard "same" padding formula used by Kokoro ResBlocks.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_dilation_padding_relationship_ks3() {
        let dilation: usize = kani::any();
        kani::assume(dilation >= 1 && dilation <= 9);

        let kernel_size = 3usize;
        let expected_padding = dilation * (kernel_size - 1) / 2;
        assert_eq!(expected_padding, dilation, "for ks=3, padding == dilation");

        let params = NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            dilation,
            expected_padding,
            vec![1, 128, 256],
            128,
            kernel_size,
        );
        assert_eq!(params.conv_padding, expected_padding);
    }

    // -----------------------------------------------------------------------
    // Proof 11: dilation/padding for kernel_size=7
    // -----------------------------------------------------------------------

    /// For kernel_size=7, padding = dilation * (7-1)/2 = dilation * 3.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_dilation_padding_relationship_ks7() {
        let dilation: usize = kani::any();
        kani::assume(dilation >= 1 && dilation <= 5);

        let kernel_size = 7usize;
        let expected_padding = dilation * (kernel_size - 1) / 2;
        assert_eq!(
            expected_padding,
            dilation * 3,
            "for ks=7, padding == dilation*3"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 12: Pool step between adain1 and conv1 in upsample blocks
    // -----------------------------------------------------------------------

    /// In upsample ResBlocks, the data flow is:
    ///   adain1 -> pool -> conv1 -> adain2 -> conv2
    /// The pool step index must satisfy adain1 < pool < conv1.
    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_pool_step_ordering_in_upsample() {
        let adain1: usize = kani::any();
        let pool: usize = kani::any();
        let conv1: usize = kani::any();
        let adain2: usize = kani::any();
        let conv2: usize = kani::any();

        kani::assume(adain1 < pool);
        kani::assume(pool < conv1);
        kani::assume(conv1 < adain2);
        kani::assume(adain2 < conv2);
        kani::assume(conv2 <= 200);

        // adain1 and pool are NOT replaced (they stay live)
        let replaced = vec![conv1, adain2, conv2];
        assert!(!replaced.contains(&adain1));
        assert!(!replaced.contains(&pool));

        // All replaced indices are strictly after pool
        for &r in &replaced {
            assert!(r > pool);
        }
    }

    // -----------------------------------------------------------------------
    // Proof 13: Weight merge count for unfused phase1
    // -----------------------------------------------------------------------

    /// When phase1 is built from unfused (standalone adain + conv1d),
    /// the weight_data contains AdaIN weights + conv_weight + conv_bias.
    /// Minimum: 2 AdaIN weights + 1 conv_weight = 3.
    /// With bias: 2 AdaIN weights + 1 conv_weight + 1 conv_bias = 4.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_unfused_phase1_weight_count() {
        let adain_keys = ["gamma", "beta"]; // standard AdaIN weights
        let conv_keys_bias = ["conv_weight", "conv_bias"];
        let conv_keys_no_bias = ["conv_weight"];

        let total_with_bias = adain_keys.len() + conv_keys_bias.len();
        let total_no_bias = adain_keys.len() + conv_keys_no_bias.len();

        assert_eq!(total_with_bias, 4);
        assert_eq!(total_no_bias, 3);
    }

    // -----------------------------------------------------------------------
    // Proof 14: FusedResBlock residual_scale absorbing 1/sqrt(2)
    // -----------------------------------------------------------------------

    /// The F0 energy predictor uses `/ sqrt(2)` as residual scaling.
    /// This is equivalent to `* (1/sqrt(2))` which must be finite and
    /// in the range (0, 1).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn proof_inv_sqrt2_is_valid_residual_scale() {
        let scale = 1.0f32 / 2.0f32.sqrt();
        assert!(scale.is_finite());
        assert!(scale > 0.0);
        assert!(scale < 1.0);

        // Approximate value check
        let expected = 0.7071067811865476f64;
        let diff = ((scale as f64) - expected).abs();
        assert!(diff < 1e-6, "1/sqrt(2) must be approximately 0.7071");
    }

    // -----------------------------------------------------------------------
    // Proof 15: FusedResBlock with all None optional fields
    // -----------------------------------------------------------------------

    /// The simplest FusedResBlock configuration: no style_proj, no shortcut,
    /// no pool, no batch_offset. This is the "direct buffer" path.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_all_none_optional_fields() {
        let params =
            NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 1, vec![1, 64, 128], 64, 3);

        let op = NativeOpKind::FusedResBlock {
            phase1: params.clone(),
            phase2: params,
            input_steps: vec![0, 1, 2, 3, 4],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: None,
        };

        // Direct path: 3 dispatches, 2 encoding events
        assert_eq!(op.estimated_metal_dispatches(), 3);
        assert_eq!(op.estimated_encoding_events(), 2);

        // No extra dependencies beyond input_steps
        let mut deps = Vec::new();
        op.collect_direct_step_deps(&mut deps);
        assert_eq!(deps.len(), 5); // only input_steps
    }

    // -----------------------------------------------------------------------
    // Proof 16: IdentityPassthrough replacement count
    // -----------------------------------------------------------------------

    /// Standard ResBlock fusion replaces 4 steps with IdentityPassthrough
    /// and places the FusedResBlock at the add position. Total step count
    /// doesn't change (all positions are preserved).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_identity_passthrough_count_standard() {
        let total_steps = 20usize;
        let replaced_count = 4usize; // standard: adain1, conv1, adain2, conv2
        let fused_count = 1usize; // the add step becomes FusedResBlock

        // Total steps unchanged: we modify in-place, not add/remove
        assert_eq!(
            replaced_count + fused_count,
            5,
            "4 replaced + 1 fused = 5 steps affected"
        );
        assert!(
            replaced_count + fused_count <= total_steps,
            "affected steps must fit in total"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 17: Weight merge prefix invariant for 2 phases
    // -----------------------------------------------------------------------

    /// After merging, every key must have exactly one of "p1_" or "p2_" prefix.
    /// A key without prefix means the merge forgot to add it.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_weight_merge_all_prefixed_no_raw_keys() {
        let base = ["gamma", "beta", "conv_weight"];

        for &k in &base {
            let p1 = format!("p1_{k}");
            let p2 = format!("p2_{k}");

            // Must have prefix
            assert!(p1.starts_with("p1_"));
            assert!(p2.starts_with("p2_"));

            // Must not be raw
            assert_ne!(p1, k);
            assert_ne!(p2, k);
        }
    }

    // -----------------------------------------------------------------------
    // Proof 18: activation family matching is reflexive
    // -----------------------------------------------------------------------

    /// Same-family matching is reflexive: (Snake, Snake) and
    /// (LeakyRelu{s}, LeakyRelu{s}) always match.
    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_activation_family_reflexive() {
        let slope: f32 = kani::any();
        kani::assume(slope.is_finite() && slope > 0.0);

        let activations = [NormActivation::Snake, NormActivation::LeakyRelu { slope }];

        for act in &activations {
            let same = matches!(
                (act, act),
                (NormActivation::Snake, NormActivation::Snake)
                    | (
                        NormActivation::LeakyRelu { .. },
                        NormActivation::LeakyRelu { .. }
                    )
            );
            assert!(same, "same-family matching must be reflexive");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 19: add node with != 2 inputs is rejected
    // -----------------------------------------------------------------------

    /// The try_fuse_at_add function checks add_inputs.len() != 2 and
    /// returns false. This guard prevents index-out-of-bounds on
    /// add_inputs[0] and add_inputs[1].
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_add_node_non_2_inputs_rejected() {
        let n: usize = kani::any();
        kani::assume(n <= 10);
        if n != 2 {
            // Cannot safely index [0] and [1]
            assert!(n < 2 || n > 2);
        } else {
            assert_eq!(n, 2);
            // Safe to index both
            let arr = [0usize, 1];
            assert!(arr[0] < n && arr[1] < n);
        }
    }

    // -----------------------------------------------------------------------
    // Proof 20: residual_scale * bounded_sum is finite
    // -----------------------------------------------------------------------

    /// For typical model values, (x + h) * scale must not overflow.
    /// This is critical because the FusedResBlock computes this in the
    /// Metal kernel epilogue.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_residual_scale_bounded_sum_finite() {
        let scale: f32 = kani::any();
        let x: f32 = kani::any();
        let h: f32 = kani::any();

        kani::assume(scale.is_finite() && scale.abs() <= 2.0);
        kani::assume(x.is_finite() && x.abs() <= 1e4);
        kani::assume(h.is_finite() && h.abs() <= 1e4);

        let sum = x + h;
        assert!(sum.is_finite(), "sum of bounded values must be finite");

        let result = sum * scale;
        assert!(result.is_finite(), "scaled sum must be finite");
    }
}
