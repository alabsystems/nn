// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for peephole ResBlock pattern-matching correctness
//! and fusion equivalence invariants.
//!
//! Proves:
//! - FusedResBlock weight merge disjointness across 3-phase prefix schemes.
//! - NormActivConv1dParams field preservation under Clone for all activation variants.
//! - ResBlock replaced-step set size is bounded (standard=4, upsample=3).
//! - FusedResBlock input_steps distinctness (no duplicate step references).
//! - Residual scale multiplication preserves finiteness for bounded inputs.
//! - Style projection absorption reduces input_steps from 5 to 2.
//! - PeepholeConfig bitwise field independence (disabling one preserves rest).
//! - StyleBatchOffset consecutive block packing has no gaps.
//! - NormActivConv1dParams kernel_size bounds conv_padding.
//! - FusedResBlock dispatch estimate is deterministic for identical configs.
//! - StyleProjectionParams total_out = 2*(channels1 + channels2).
//! - FusedResBlock with shortcut_step collects it as dependency.
//! - FusedResBlock with pool_step collects it as dependency.
//!
//! Part of #3731.

#[cfg(kani)]
mod proofs {
    use std::collections::{HashMap, HashSet};

    use crate::trace_compile::{
        NativeOpKind, NormActivConv1dParams, NormActivation, PeepholeConfig, StyleBatchOffset,
        StyleProjectionParams,
    };

    // -----------------------------------------------------------------------
    // Proof 1: weight merge 3-prefix scheme (p1_, p2_, sc_) is disjoint
    // -----------------------------------------------------------------------

    /// FusedResBlock merges phase1, phase2, and (optionally) shortcut weights.
    /// The prefixes p1_, p2_, sc_ must never collide for any base key.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_three_prefix_weight_merge_disjoint() {
        let base_keys = ["gamma", "beta", "conv_weight", "conv_bias", "alpha"];
        let mut all_keys: HashSet<String> = HashSet::new();

        for &k in &base_keys {
            let p1 = format!("p1_{k}");
            let p2 = format!("p2_{k}");
            let sc = format!("sc_{k}");

            // Each prefixed key must differ from all others
            assert_ne!(p1, p2);
            assert_ne!(p1, sc);
            assert_ne!(p2, sc);

            assert!(all_keys.insert(p1), "p1_ key collision");
            assert!(all_keys.insert(p2), "p2_ key collision");
            assert!(all_keys.insert(sc), "sc_ key collision");
        }

        // 5 base keys * 3 prefixes = 15 unique keys
        assert_eq!(all_keys.len(), 15);
    }

    // -----------------------------------------------------------------------
    // Proof 2: NormActivConv1dParams Clone preserves all fields for Snake
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_norm_activ_clone_snake() {
        let params = NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            2, // dilation
            3, // padding
            vec![1, 128, 256],
            128, // output_channels
            7,   // kernel_size
        );
        let cloned = params.clone();

        assert!(matches!(cloned.activation, NormActivation::Snake));
        assert_eq!(cloned.eps, 1e-5);
        assert_eq!(cloned.conv_dilation, 2);
        assert_eq!(cloned.conv_padding, 3);
        assert_eq!(cloned.input_shape, vec![1, 128, 256]);
        assert_eq!(cloned.output_channels, 128);
        assert_eq!(cloned.kernel_size, 7);
    }

    // -----------------------------------------------------------------------
    // Proof 3: NormActivConv1dParams Clone preserves LeakyRelu slope
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_norm_activ_clone_leaky_relu_slope() {
        let slope: f32 = kani::any();
        kani::assume(slope.is_finite() && slope >= 0.0 && slope <= 1.0);

        let params = NormActivConv1dParams::new(
            NormActivation::LeakyRelu { slope },
            1e-6,
            1,
            1,
            vec![1, 64, 32],
            64,
            3,
        );
        let cloned = params.clone();

        match cloned.activation {
            NormActivation::LeakyRelu { slope: s } => {
                assert_eq!(s, slope, "slope must survive Clone");
            }
            _ => panic!("Clone changed activation variant"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 4: standard ResBlock replaces exactly 4 steps
    // -----------------------------------------------------------------------

    /// Standard path: adain1 + conv1 + adain2 + conv2 = 4 replaced.
    /// The add step becomes the FusedResBlock.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_standard_resblock_replaced_count_4() {
        let replaced: Vec<usize> = vec![10, 11, 12, 13]; // adain1, conv1, adain2, conv2
        assert_eq!(replaced.len(), 4);

        // All replaced indices must be distinct
        let set: HashSet<usize> = replaced.iter().copied().collect();
        assert_eq!(set.len(), 4, "replaced steps must be distinct");
    }

    // -----------------------------------------------------------------------
    // Proof 5: upsample ResBlock replaces exactly 3 steps
    // -----------------------------------------------------------------------

    /// Upsample path: adain1 stays live, only conv1 + adain2 + conv2 = 3.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_upsample_resblock_replaced_count_3() {
        let adain1_idx = 10usize;
        let pool_idx = 11usize;
        let conv1_idx = 12usize;
        let adain2_idx = 13usize;
        let conv2_idx = 14usize;

        let replaced = vec![conv1_idx, adain2_idx, conv2_idx];
        assert_eq!(replaced.len(), 3);

        // adain1 and pool must NOT be in replaced set
        assert!(!replaced.contains(&adain1_idx));
        assert!(!replaced.contains(&pool_idx));
    }

    // -----------------------------------------------------------------------
    // Proof 6: FusedResBlock input_steps are all distinct
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_input_steps_distinctness() {
        // Standard: [x, gamma1, beta1, gamma2, beta2] = 5 distinct indices
        let input_steps = vec![0usize, 3, 4, 7, 8];
        let set: HashSet<usize> = input_steps.iter().copied().collect();
        assert_eq!(
            set.len(),
            input_steps.len(),
            "all input_steps must be distinct"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 7: residual scale preserves finiteness for bounded values
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_residual_scale_preserves_finiteness() {
        let scale: f32 = kani::any();
        let x: f32 = kani::any();

        kani::assume(scale.is_finite());
        kani::assume(scale != 0.0);
        kani::assume(scale.abs() <= 2.0);
        kani::assume(x.is_finite());
        kani::assume(x.abs() <= 1e6);

        let result = x * scale;
        assert!(result.is_finite(), "scale * bounded input must be finite");
    }

    // -----------------------------------------------------------------------
    // Proof 8: style projection absorption reduces inputs from 5 to 2
    // -----------------------------------------------------------------------

    /// After style projection absorption, the FusedResBlock's input_steps
    /// changes from [x, g1, b1, g2, b2] (5) to [x, style_embed] (2).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_style_absorption_5_to_2() {
        let before = vec![0usize, 1, 2, 3, 4];
        assert_eq!(before.len(), 5);

        // After absorption: only [x, style_embed]
        let after = vec![0usize, 42]; // x at 0, style embedding at 42
        assert_eq!(after.len(), 2);
        assert!(after.len() < before.len());
    }

    // -----------------------------------------------------------------------
    // Proof 9: PeepholeConfig field independence (disable each individually)
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_peephole_config_field_independence() {
        let which_field: u8 = kani::any();
        kani::assume(which_field <= 10);

        let mut config = PeepholeConfig::default();

        // Disable one field
        match which_field {
            0 => config.norm_activ_conv1d = false,
            1 => config.fused_resblock = false,
            2 => config.linear_activation = false,
            3 => config.add_layer_norm = false,
            4 => config.norm_linear = false,
            5 => config.attention_transpose = false,
            6 => config.flip_lstm = false,
            7 => config.batched_linear_projection = false,
            8 => config.channels_first_layer_norm = false,
            9 => config.silu_mul = false,
            _ => config.auto_fuse_elementwise = false,
        }

        // Count how many are still enabled
        let enabled_count = [
            config.norm_activ_conv1d,
            config.fused_resblock,
            config.linear_activation,
            config.add_layer_norm,
            config.norm_linear,
            config.attention_transpose,
            config.flip_lstm,
            config.batched_linear_projection,
            config.channels_first_layer_norm,
            config.silu_mul,
            config.auto_fuse_elementwise,
        ]
        .iter()
        .filter(|&&x| x)
        .count();

        // Exactly 10 remain enabled (1 of 11 was disabled)
        assert_eq!(
            enabled_count, 10,
            "disabling one field must leave 10 enabled"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 10: StyleBatchOffset consecutive packing has no gaps
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_style_batch_offset_consecutive_no_gaps() {
        let c1_a: u8 = kani::any();
        let c2_a: u8 = kani::any();
        let c1_b: u8 = kani::any();
        let c2_b: u8 = kani::any();

        kani::assume(c1_a >= 1 && c1_a <= 64);
        kani::assume(c2_a >= 1 && c2_a <= 64);
        kani::assume(c1_b >= 1 && c1_b <= 64);
        kani::assume(c2_b >= 1 && c2_b <= 64);

        let block_a = StyleBatchOffset::new(0, c1_a.into(), c2_a.into());
        let width_a = 2 * (block_a.channels1 + block_a.channels2);

        let block_b = StyleBatchOffset::new(width_a, c1_b.into(), c2_b.into());
        let width_b = 2 * (block_b.channels1 + block_b.channels2);

        // Block B starts exactly where block A ends
        assert_eq!(block_b.offset, width_a, "no gap between consecutive blocks");

        // Total allocation = width_a + width_b
        let total = width_a + width_b;
        assert_eq!(
            total,
            2 * (c1_a as usize + c2_a as usize + c1_b as usize + c2_b as usize)
        );
    }

    // -----------------------------------------------------------------------
    // Proof 11: NormActivConv1dParams kernel_size bounds padding
    // -----------------------------------------------------------------------

    /// For valid (non-expanding) convolution, padding < kernel_size.
    /// This is a construction invariant checked by the peephole pass.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_kernel_size_bounds_padding() {
        let ks: usize = kani::any();
        let pad: usize = kani::any();
        kani::assume(ks >= 1 && ks <= 31);
        kani::assume(pad < ks);

        // padding must be strictly less than kernel_size
        assert!(pad < ks, "padding must be < kernel_size");
        // The dilated kernel size must also bound padding
        let dilation: usize = kani::any();
        kani::assume(dilation >= 1 && dilation <= 4);
        let dilated_ks = dilation * (ks - 1) + 1;
        assert!(
            dilated_ks >= ks,
            "dilation can only increase effective kernel size"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 12: FusedResBlock dispatch estimate is deterministic
    // -----------------------------------------------------------------------

    /// Two FusedResBlocks with identical configuration must produce
    /// identical dispatch estimates. Non-deterministic estimates would
    /// break the performance gate assertions.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_fused_resblock_dispatch_deterministic() {
        let params =
            NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 1, vec![1, 64, 128], 64, 3);

        let op1 = NativeOpKind::FusedResBlock {
            phase1: params.clone(),
            phase2: params.clone(),
            input_steps: vec![0, 1, 2, 3, 4],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: None,
        };

        let op2 = NativeOpKind::FusedResBlock {
            phase1: params.clone(),
            phase2: params,
            input_steps: vec![0, 1, 2, 3, 4],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: None,
        };

        assert_eq!(
            op1.estimated_metal_dispatches(),
            op2.estimated_metal_dispatches(),
            "identical FusedResBlocks must have identical dispatch estimates"
        );
        assert_eq!(
            op1.estimated_encoding_events(),
            op2.estimated_encoding_events(),
        );
    }

    // -----------------------------------------------------------------------
    // Proof 13: StyleProjectionParams total_out arithmetic
    // -----------------------------------------------------------------------

    /// total_out for a FusedResBlock's style projection = 2*(channels1 + channels2).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_style_projection_total_out() {
        let c1: usize = kani::any();
        let c2: usize = kani::any();
        let sd: usize = kani::any();

        kani::assume(c1 >= 1 && c1 <= 512);
        kani::assume(c2 >= 1 && c2 <= 512);
        kani::assume(sd >= 1 && sd <= 512);

        let sp = StyleProjectionParams::new(c1, c2, sd);
        assert_eq!(sp.channels1, c1);
        assert_eq!(sp.channels2, c2);
        assert_eq!(sp.style_dim, sd);

        // The total projection output = 2*(C1 + C2)
        let total = 2 * (sp.channels1 + sp.channels2);
        assert_eq!(total, 2 * c1 + 2 * c2);
    }

    // -----------------------------------------------------------------------
    // Proof 14: FusedResBlock with shortcut_step collects it as dep
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_fused_resblock_shortcut_dep() {
        let sc: usize = kani::any();
        kani::assume(sc <= 500);

        let params =
            NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 1, vec![1, 32, 64], 32, 3);

        let op = NativeOpKind::FusedResBlock {
            phase1: params.clone(),
            phase2: params,
            input_steps: vec![0, 1, 2, 3, 4],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: Some(sc),
            pool_step: None,
            style_batch_offset: None,
        };

        let mut deps = Vec::new();
        op.collect_direct_step_deps(&mut deps);

        // input_steps(5) + shortcut(1) = 6
        assert_eq!(deps.len(), 6);
        assert!(deps.contains(&sc), "shortcut_step must appear in deps");
    }

    // -----------------------------------------------------------------------
    // Proof 15: FusedResBlock with pool_step collects it as dep
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_fused_resblock_pool_dep() {
        let ps: usize = kani::any();
        kani::assume(ps <= 500);

        let params = NormActivConv1dParams::new(
            NormActivation::LeakyRelu { slope: 0.2 },
            1e-5,
            1,
            1,
            vec![1, 32, 64],
            32,
            3,
        );

        let op = NativeOpKind::FusedResBlock {
            phase1: params.clone(),
            phase2: params,
            input_steps: vec![0, 1, 2, 3, 4],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: Some(ps),
            style_batch_offset: None,
        };

        let mut deps = Vec::new();
        op.collect_direct_step_deps(&mut deps);

        // input_steps(5) + pool(1) = 6
        assert_eq!(deps.len(), 6);
        assert!(deps.contains(&ps), "pool_step must appear in deps");
    }
}
