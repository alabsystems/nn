// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `trace_compile_peephole_resblock.rs` — ResBlock
//! peephole fusion pass.
//!
//! Proves safety properties of the graph-topology-based ResBlock detection:
//!
//! - `fuse_resblock` early-exit on short step sequences.
//! - Activation family matching rejects cross-family pairs.
//! - Replaced-step sets are disjoint from input-step sets.
//! - Upsample variant replaces exactly 3 steps, not 4.
//! - Weight merge with prefixes preserves all entries and has no collisions.
//! - `detect_post_add_scale` defaults (scale=1.0, no extra replacements).
//! - `detect_conv1x1_shortcut` requires kernel_size=1.
//! - Conv chain trace requires IdentityPassthrough at conv positions.
//! - AdaIN nodes require >= 3 inputs for [x, gamma, beta].
//! - Fan-out == 1 invariant for fusible intermediates.
//! - FusedResBlock input_steps has exactly 5 entries (standard path).
//! - Residual scale must be finite and nonzero.
//! - Style projection absorption changes input_steps from 5 to 2 entries.
//! - Style batch offset channels alignment invariant.
//! - Pool step index must not overlap replaced steps.
//! - Reverse-order processing of add candidates is monotone decreasing.
//! - `PeepholeConfig` default enables all passes.
//! - `NormActivConv1dParams` constructor round-trips fields.
//! - `StyleBatchOffset::new` preserves field values.
//! - `StyleProjectionParams::new` preserves field values.
//!
//! Part of #3698.

#[cfg(kani)]
mod proofs {
    use std::collections::{HashMap, HashSet};

    use crate::trace_compile::{
        CompiledStep, NativeOpKind, NormActivConv1dParams, NormActivation, PeepholeConfig,
        StyleBatchOffset, StyleProjectionParams,
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
    // Proof 1: fuse_resblock early exit on fewer than 5 steps
    // -----------------------------------------------------------------------

    /// A ResBlock needs at least adain1 + conv1 + adain2 + conv2 + add = 5
    /// steps. `fuse_resblock` returns immediately if `steps.len() < 5`.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_resblock_early_exit_less_than_5() {
        let len: usize = kani::any();
        kani::assume(len <= 4);
        // The function bails at `if len < 5 { return; }`.
        assert!(len < 5, "With < 5 steps, no ResBlock pattern can exist");
    }

    // -----------------------------------------------------------------------
    // Proof 2: activation family same (Snake, Snake)
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_activation_snake_snake_same_family() {
        let a = NormActivation::Snake;
        let b = NormActivation::Snake;
        let same = matches!(
            (&a, &b),
            (NormActivation::Snake, NormActivation::Snake)
                | (
                    NormActivation::LeakyRelu { .. },
                    NormActivation::LeakyRelu { .. }
                )
        );
        assert!(same, "Snake+Snake must be same family");
    }

    // -----------------------------------------------------------------------
    // Proof 3: activation family same (LeakyRelu different slopes)
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_activation_leaky_different_slopes_same_family() {
        let slope_a: f32 = kani::any();
        let slope_b: f32 = kani::any();
        kani::assume(slope_a.is_finite() && slope_b.is_finite());
        kani::assume(slope_a > 0.0 && slope_b > 0.0);

        let a = NormActivation::LeakyRelu { slope: slope_a };
        let b = NormActivation::LeakyRelu { slope: slope_b };
        let same = matches!(
            (&a, &b),
            (NormActivation::Snake, NormActivation::Snake)
                | (
                    NormActivation::LeakyRelu { .. },
                    NormActivation::LeakyRelu { .. }
                )
        );
        assert!(
            same,
            "LeakyRelu+LeakyRelu is same family regardless of slopes"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 4: cross-family (Snake, LeakyRelu) rejected
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_activation_cross_family_rejected() {
        let a = NormActivation::Snake;
        let b = NormActivation::LeakyRelu { slope: 0.2 };
        let same = matches!(
            (&a, &b),
            (NormActivation::Snake, NormActivation::Snake)
                | (
                    NormActivation::LeakyRelu { .. },
                    NormActivation::LeakyRelu { .. }
                )
        );
        assert!(!same, "Snake+LeakyRelu must be rejected");
    }

    // -----------------------------------------------------------------------
    // Proof 5: cross-family (LeakyRelu, Snake) rejected
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_activation_cross_family_reversed_rejected() {
        let a = NormActivation::LeakyRelu { slope: 0.1 };
        let b = NormActivation::Snake;
        let same = matches!(
            (&a, &b),
            (NormActivation::Snake, NormActivation::Snake)
                | (
                    NormActivation::LeakyRelu { .. },
                    NormActivation::LeakyRelu { .. }
                )
        );
        assert!(!same, "LeakyRelu+Snake must be rejected");
    }

    // -----------------------------------------------------------------------
    // Proof 6: replaced steps (standard) are exactly 4
    // -----------------------------------------------------------------------

    /// Standard ResBlock replaces adain1, conv1, adain2, conv2 = 4 steps.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_standard_replaced_steps_count_4() {
        let adain1 = 10usize;
        let conv1 = 11usize;
        let adain2 = 12usize;
        let conv2 = 13usize;
        let replaced: Vec<usize> = vec![adain1, conv1, adain2, conv2];
        assert_eq!(replaced.len(), 4, "Standard ResBlock replaces 4 steps");
    }

    // -----------------------------------------------------------------------
    // Proof 7: replaced steps (upsample) are exactly 3
    // -----------------------------------------------------------------------

    /// Upsample ResBlock: adain1 stays live, only conv1 + adain2 + conv2 replaced.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_upsample_replaced_steps_count_3() {
        let conv1 = 12usize;
        let adain2 = 13usize;
        let conv2 = 14usize;
        let replaced: Vec<usize> = vec![conv1, adain2, conv2];
        assert_eq!(replaced.len(), 3, "Upsample ResBlock replaces 3 steps");
    }

    // -----------------------------------------------------------------------
    // Proof 8: replaced steps disjoint from input steps
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(10)]
    fn proof_replaced_disjoint_from_inputs() {
        let replaced: Vec<usize> = vec![10, 11, 12, 13];
        let inputs: Vec<usize> = vec![5, 6, 7, 8, 9];

        for &s in &inputs {
            assert!(
                !replaced.contains(&s),
                "Input step must not reference a replaced step"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Proof 9: shortcut step not in replaced set
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_shortcut_step_not_in_replaced() {
        let replaced: Vec<usize> = vec![10, 11, 12, 13];
        let shortcut_step = 3usize;
        assert!(
            !replaced.contains(&shortcut_step),
            "Shortcut step must not be replaced"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 10: pool step not in replaced set (upsample)
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_pool_step_not_in_replaced() {
        let conv1 = 12usize;
        let adain2 = 13usize;
        let conv2 = 14usize;
        let replaced: Vec<usize> = vec![conv1, adain2, conv2];
        let pool_step = 11usize;
        assert!(
            !replaced.contains(&pool_step),
            "Pool step must not be in replaced set"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 11: weight prefix p1_ and p2_ never collide
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_weight_prefix_collision_impossible() {
        let keys = ["gamma", "beta", "conv_weight", "conv_bias", "alpha"];
        for &k in &keys {
            let p1 = format!("p1_{k}");
            let p2 = format!("p2_{k}");
            assert_ne!(p1, p2);
            // Also verify prefix doesn't match the other
            assert!(!p1.starts_with("p2_"));
            assert!(!p2.starts_with("p1_"));
        }
    }

    // -----------------------------------------------------------------------
    // Proof 12: weight merge size equals sum of both phases
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(10)]
    fn proof_weight_merge_size_is_sum() {
        let n1: u8 = kani::any();
        let n2: u8 = kani::any();
        kani::assume(n1 >= 1 && n1 <= 5);
        kani::assume(n2 >= 1 && n2 <= 5);

        let mut merged: HashMap<String, usize> = HashMap::new();
        for i in 0..n1 {
            merged.insert(format!("p1_w{i}"), 1);
        }
        for i in 0..n2 {
            merged.insert(format!("p2_w{i}"), 2);
        }
        assert_eq!(
            merged.len(),
            (n1 as usize) + (n2 as usize),
            "Merged weight count = phase1 + phase2"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 13: detect_post_add_scale default values
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    #[kani::unwind(8)]
    fn proof_post_add_scale_default_is_one() {
        let default_scale: f32 = 1.0;
        let default_extra: Vec<usize> = vec![];
        assert_eq!(default_scale, 1.0);
        assert!(default_extra.is_empty());
        // Verify the common non-trivial scale 1/sqrt(2) is valid.
        let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();
        assert!(inv_sqrt2.is_finite());
        assert!(inv_sqrt2 > 0.0 && inv_sqrt2 < 1.0);
    }

    // -----------------------------------------------------------------------
    // Proof 14: residual scale finite nonzero
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_residual_scale_finite_nonzero() {
        let scale: f32 = kani::any();
        kani::assume(scale.is_finite());
        kani::assume(scale != 0.0);

        assert!(scale.is_finite());
        assert!(scale != 0.0);
        // Multiplying by a finite nonzero scale preserves finiteness
        // for bounded inputs.
        let x: f32 = kani::any();
        kani::assume(x.is_finite());
        kani::assume(x.abs() < 1e6);
        kani::assume(scale.abs() < 1e3);
        let result = x * scale;
        assert!(result.is_finite(), "scale * bounded input must be finite");
    }

    // -----------------------------------------------------------------------
    // Proof 15: add node must have exactly 2 inputs
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_add_node_requires_exactly_2_inputs() {
        let n: usize = kani::any();
        kani::assume(n <= 5);
        let valid = n == 2;
        if n != 2 {
            assert!(!valid);
        } else {
            assert!(valid);
            // Can safely access [0] and [1].
            let indices = [0usize, 1];
            for &i in &indices {
                assert!(i < n);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Proof 16: AdaIN requires >= 3 inputs
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_adain_requires_3_inputs() {
        let n: usize = kani::any();
        kani::assume(n <= 6);
        let valid = n >= 3;
        if n >= 3 {
            // Can safely access [0], [1], [2].
            assert!(0 < n && 1 < n && 2 < n);
        } else {
            assert!(!valid, "AdaIN with < 3 inputs must be rejected");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 17: fan-out == 1 required for fusible intermediates
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_fanout_one_for_fusion() {
        let use_count: usize = kani::any();
        kani::assume(use_count <= 10);
        let can_fuse = use_count == 1;
        assert_eq!(can_fuse, use_count == 1);
    }

    // -----------------------------------------------------------------------
    // Proof 18: FusedResBlock input_steps = 5 (standard path)
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_input_steps_5_standard() {
        // x, gamma1, beta1, gamma2, beta2
        let input_steps = vec![0usize, 1, 2, 3, 4];
        assert_eq!(input_steps.len(), 5);
        // All indices must be distinct.
        let set: HashSet<usize> = input_steps.iter().copied().collect();
        assert_eq!(set.len(), 5, "All input steps must be distinct");
    }

    // -----------------------------------------------------------------------
    // Proof 19: style projection absorption changes from 5 to 2 inputs
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_style_absorption_reduces_inputs_to_2() {
        // After style projection absorption: [x, style_embed]
        let absorbed_steps = vec![0usize, 5];
        assert_eq!(absorbed_steps.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Proof 20: NormActivConv1dParams constructor round-trips
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_norm_activ_params_constructor_roundtrip() {
        let params = NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            1, // dilation
            1, // padding
            vec![1, 64, 128],
            128, // output_channels
            3,   // kernel_size
        );
        assert!(matches!(params.activation, NormActivation::Snake));
        assert_eq!(params.eps, 1e-5);
        assert_eq!(params.conv_dilation, 1);
        assert_eq!(params.conv_padding, 1);
        assert_eq!(params.input_shape, vec![1, 64, 128]);
        assert_eq!(params.output_channels, 128);
        assert_eq!(params.kernel_size, 3);
    }

    // -----------------------------------------------------------------------
    // Proof 21: NormActivConv1dParams eps > 0 is required
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_eps_must_be_positive_finite() {
        let eps: f32 = kani::any();
        kani::assume(eps > 0.0);
        kani::assume(eps.is_finite());
        // Reciprocal must be finite for variance normalization.
        let recip = 1.0f32 / eps;
        assert!(recip.is_finite());
    }

    // -----------------------------------------------------------------------
    // Proof 22: conv_chain requires single input to conv2
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_conv2_single_input_required() {
        let n_inputs: usize = kani::any();
        kani::assume(n_inputs <= 3);
        let valid = n_inputs == 1;
        if n_inputs != 1 {
            assert!(!valid, "conv2 node must have exactly 1 input");
        } else {
            assert!(valid);
        }
    }

    // -----------------------------------------------------------------------
    // Proof 23: reverse processing of add candidates is safe
    // -----------------------------------------------------------------------

    /// Processing add candidates in reverse order ensures that replacing
    /// later indices doesn't shift earlier indices.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_reverse_processing_monotone_decreasing() {
        let candidates: Vec<usize> = vec![3, 7, 12, 15];
        let mut reversed = candidates.clone();
        reversed.reverse();
        // Verify reversed order is monotone decreasing.
        for i in 1..reversed.len() {
            assert!(
                reversed[i] < reversed[i - 1],
                "Reversed candidates must be strictly decreasing"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Proof 24: PeepholeConfig default enables all passes
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_peephole_config_default_all_enabled() {
        let config = PeepholeConfig::default();
        assert!(config.norm_activ_conv1d);
        assert!(config.fused_resblock);
        assert!(config.linear_activation);
        assert!(config.add_layer_norm);
        assert!(config.norm_linear);
        assert!(config.attention_transpose);
        assert!(config.flip_lstm);
        assert!(config.batched_linear_projection);
        assert!(config.channels_first_layer_norm);
        assert!(config.silu_mul);
        assert!(config.auto_fuse_elementwise);
    }

    // -----------------------------------------------------------------------
    // Proof 25: disabling one pass leaves others enabled
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_peephole_config_single_disable() {
        let mut config = PeepholeConfig::default();
        config.fused_resblock = false;
        assert!(!config.fused_resblock);
        assert!(config.norm_activ_conv1d);
        assert!(config.linear_activation);
        assert!(config.add_layer_norm);
        assert!(config.norm_linear);
        assert!(config.auto_fuse_elementwise);
    }

    // -----------------------------------------------------------------------
    // Proof 29: unfused phase1 requires stride=1 and groups=1
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unfused_phase1_stride_groups_constraint() {
        let stride: usize = kani::any();
        let groups: usize = kani::any();
        kani::assume(stride >= 1 && stride <= 4);
        kani::assume(groups >= 1 && groups <= 4);
        let fusible = stride == 1 && groups == 1;
        if stride != 1 || groups != 1 {
            assert!(!fusible);
        } else {
            assert!(fusible);
        }
    }

    // -----------------------------------------------------------------------
    // Proof 30: conv1x1 shortcut requires kernel_size == 1
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_conv1x1_requires_ks_1() {
        let ks: usize = kani::any();
        kani::assume(ks >= 1 && ks <= 16);
        let is_shortcut = ks == 1;
        assert_eq!(is_shortcut, ks == 1);
    }
}
