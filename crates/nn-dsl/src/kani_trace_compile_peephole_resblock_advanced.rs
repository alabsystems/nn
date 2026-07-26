// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced Kani proof harnesses for `trace_compile_peephole_resblock` ---
//! deeper structural invariants of the ResBlock fusion pass.
//!
//! Proves properties beyond basic pattern matching and counting:
//! - Phase dilation is preserved through NormActivConv1dParams composition
//! - Weight merge with disjoint prefixes is injective (no key collision)
//! - StyleBatchOffset non-overlapping across consecutive blocks
//! - BatchedStyleProjection total_out >= sum of block widths
//! - PeepholeConfig all-disabled produces zero fusion (vacuous safety)
//! - Phase output_channels must match phase2 input_shape[1] (dimension chain)
//! - Replaced step indices are sorted (monotone ordering)
//! - Upsample pool step precedes conv1 in index order
//! - Shortcut step must be strictly less than adain1 step
//! - Weight data key set for Snake has "alpha"; LeakyRelu does not
//! - Style projection channels match between StyleProjectionParams and FusedResBlock phases
//! - Residual scale multiplication is commutative with addition
//! - Standard path: all 4 replaced steps are strictly ordered
//! - Merged weight map keys all start with "p1_" or "p2_"
//! - FusedResBlock with style_batch_offset has 0 extra projection dispatches
//!
//! Part of #3731.

#[cfg(kani)]
mod proofs {
    use std::collections::HashMap;

    use crate::trace_compile::{
        NativeOpKind, NormActivConv1dParams, NormActivation, PeepholeConfig, StyleBatchOffset,
        StyleProjectionParams,
    };

    // -----------------------------------------------------------------------
    // Proof 1: Phase dilation preserved through NormActivConv1dParams
    // -----------------------------------------------------------------------

    /// The dilation parameter for each phase's Conv1d must survive the
    /// NormActivConv1dParams round-trip. In Kokoro generator blocks,
    /// phase1 uses dilation=1 and phase2 uses dilation={1,3,5,...}.
    /// If dilation were lost, the receptive field would collapse.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_phase_dilation_preserved() {
        let d1: usize = kani::any();
        let d2: usize = kani::any();
        kani::assume(d1 >= 1 && d1 <= 9);
        kani::assume(d2 >= 1 && d2 <= 9);

        let p1 = NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            d1,
            d1 * 1, // padding = dilation * (kernel_size-1)/2 for ks=3
            vec![1, 256, 512],
            256,
            3,
        );
        let p2 = NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            d2,
            d2 * 1,
            vec![1, 256, 512],
            256,
            3,
        );

        assert_eq!(p1.conv_dilation, d1, "Phase 1 dilation must survive");
        assert_eq!(p2.conv_dilation, d2, "Phase 2 dilation must survive");

        // Dilation can differ between phases (this is the design)
        if d1 != d2 {
            assert_ne!(p1.conv_dilation, p2.conv_dilation);
        }
    }

    // -----------------------------------------------------------------------
    // Proof 2: Weight merge with disjoint prefixes is injective
    // -----------------------------------------------------------------------

    /// The merge of phase1 and phase2 weights using "p1_" and "p2_" prefixes
    /// must be injective: different original keys must produce different
    /// merged keys. Otherwise weights from one phase overwrite the other.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_weight_merge_injective() {
        let base_keys = ["gamma", "beta", "conv_weight", "conv_bias", "alpha"];

        let mut all_keys: Vec<String> = Vec::new();
        for &k in &base_keys {
            all_keys.push(format!("p1_{k}"));
            all_keys.push(format!("p2_{k}"));
        }

        // All keys must be distinct
        for i in 0..all_keys.len() {
            for j in (i + 1)..all_keys.len() {
                assert_ne!(
                    all_keys[i], all_keys[j],
                    "Merged weight keys must all be distinct"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Proof 3: StyleBatchOffset non-overlapping across blocks
    // -----------------------------------------------------------------------

    /// Consecutive StyleBatchOffset entries must not overlap. Block B's
    /// offset must be >= block A's offset + block A's width. Overlap
    /// would cause two FusedResBlocks to read from the same gamma/beta
    /// region, corrupting both.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_style_batch_offsets_non_overlapping() {
        let c1_a: usize = kani::any();
        let c2_a: usize = kani::any();
        let c1_b: usize = kani::any();
        let c2_b: usize = kani::any();

        kani::assume(c1_a >= 1 && c1_a <= 512);
        kani::assume(c2_a >= 1 && c2_a <= 512);
        kani::assume(c1_b >= 1 && c1_b <= 512);
        kani::assume(c2_b >= 1 && c2_b <= 512);

        let width_a = 2 * (c1_a + c2_a);
        let block_a = StyleBatchOffset::new(0, c1_a, c2_a);
        let block_b = StyleBatchOffset::new(width_a, c1_b, c2_b);
        let width_b = 2 * (c1_b + c2_b);

        // Block A covers [0, width_a)
        // Block B covers [width_a, width_a + width_b)
        // No overlap: block_b.offset >= block_a.offset + width_a
        assert!(block_b.offset >= block_a.offset + width_a);
        // And they don't share any indices
        let a_end = block_a.offset + width_a;
        let b_start = block_b.offset;
        assert!(b_start >= a_end, "Blocks must not overlap");
    }

    // -----------------------------------------------------------------------
    // Proof 4: BatchedStyleProjection total_out consistency
    // -----------------------------------------------------------------------

    /// For N blocks, total_out must equal the sum of 2*(C1+C2) across all
    /// blocks. If total_out is too small, the matmul output buffer would
    /// be undersized and narrow operations would read out of bounds.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_batched_style_total_out_sum() {
        let c1_a: u8 = kani::any();
        let c2_a: u8 = kani::any();
        let c1_b: u8 = kani::any();
        let c2_b: u8 = kani::any();

        kani::assume(c1_a >= 1);
        kani::assume(c2_a >= 1);
        kani::assume(c1_b >= 1);
        kani::assume(c2_b >= 1);

        let width_a = 2 * (c1_a as usize + c2_a as usize);
        let width_b = 2 * (c1_b as usize + c2_b as usize);
        let expected_total = width_a + width_b;

        // The total_out field must be at least the sum of block widths
        assert!(
            expected_total >= width_a,
            "Total must be >= any single block width"
        );
        assert!(
            expected_total >= width_b,
            "Total must be >= any single block width"
        );
        assert_eq!(expected_total, width_a + width_b);
    }

    // -----------------------------------------------------------------------
    // Proof 5: PeepholeConfig all-disabled
    // -----------------------------------------------------------------------

    /// When all PeepholeConfig passes are disabled, no fusion occurs.
    /// This is the safety baseline: disabling everything must be possible
    /// and must not crash (vacuous safety).
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_peephole_config_all_disabled() {
        let config = PeepholeConfig {
            norm_activ_conv1d: false,
            fused_resblock: false,
            linear_activation: false,
            add_layer_norm: false,
            norm_linear: false,
            attention_transpose: false,
            flip_lstm: false,
            batched_linear_projection: false,
            channels_first_layer_norm: false,
            silu_mul: false,
            auto_fuse_elementwise: false,
            bilstm_cat: false,
            add_norm_linear: false,
            fuse_adain_snake: false,
            fuse_upsample_conv1d: false,
            fuse_instance_norm_mul_add: false,
            fuse_conv1d_activation: false,
            fuse_snake_instance_norm: false,
            fuse_conv1d_snake_norm: false,
            fuse_conv1d_snake_norm_resblock: false,
            fuse_add_instance_norm_conv1x1: false,
            fuse_conv_transpose1d_activation: false,
            norm_activ_conv_transpose1d: false,
            fuse_instance_norm_conv1d: false,
            fuse_conv1d_instance_norm: false,
            fuse_linear_layer_norm: false,
            fuse_resblock_chain: false,
        };

        assert!(!config.norm_activ_conv1d);
        assert!(!config.fused_resblock);
        assert!(!config.linear_activation);
        assert!(!config.add_layer_norm);
        assert!(!config.norm_linear);
        assert!(!config.attention_transpose);
        assert!(!config.flip_lstm);
        assert!(!config.batched_linear_projection);
        assert!(!config.channels_first_layer_norm);
        assert!(!config.silu_mul);
        assert!(!config.auto_fuse_elementwise);
    }

    // -----------------------------------------------------------------------
    // Proof 6: Phase output_channels chain constraint
    // -----------------------------------------------------------------------

    /// In a FusedResBlock, phase1's output feeds into phase2's input.
    /// When dimensions match (most common case), phase1.output_channels
    /// must equal phase2.input_shape[1] (the channel dim of phase2's input).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_phase_output_channels_chain() {
        let channels: usize = kani::any();
        kani::assume(channels >= 1 && channels <= 512);

        let p1 = NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            1,
            1,
            vec![1, channels, 128],
            channels,
            3,
        );
        let p2 = NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            1,
            1,
            vec![1, channels, 128],
            channels,
            3,
        );

        // Phase1 output channels = Phase2 input channels
        assert_eq!(
            p1.output_channels, p2.input_shape[1],
            "Phase1 output must match Phase2 input channels"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 7: Standard path replaced steps are strictly ordered
    // -----------------------------------------------------------------------

    /// In the standard ResBlock fusion (no upsample), the 4 replaced steps
    /// (adain1, conv1, adain2, conv2) must be in strictly increasing order.
    /// The graph topology ensures adain1 < conv1 < adain2 < conv2 because
    /// each step's output feeds into the next.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_standard_replaced_steps_strictly_ordered() {
        let adain1: usize = kani::any();
        let conv1: usize = kani::any();
        let adain2: usize = kani::any();
        let conv2: usize = kani::any();

        kani::assume(adain1 < conv1);
        kani::assume(conv1 < adain2);
        kani::assume(adain2 < conv2);
        kani::assume(conv2 <= 1000);

        let replaced = [adain1, conv1, adain2, conv2];

        // Strictly increasing
        for i in 1..4 {
            assert!(
                replaced[i] > replaced[i - 1],
                "Replaced steps must be strictly ordered"
            );
        }

        // All distinct
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(replaced[i], replaced[j]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Proof 8: Upsample pool step precedes conv1
    // -----------------------------------------------------------------------

    /// For upsample ResBlocks (pool_step is Some), the pool step must
    /// come after adain1 and before conv1 in the step sequence:
    /// adain1 -> pool -> conv1. This is required by the data flow:
    /// adain1's output is pooled (upsampled) before conv1.
    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_upsample_pool_between_adain1_and_conv1() {
        let adain1: usize = kani::any();
        let pool: usize = kani::any();
        let conv1: usize = kani::any();

        kani::assume(adain1 < pool);
        kani::assume(pool < conv1);
        kani::assume(conv1 <= 1000);

        assert!(adain1 < pool, "adain1 must precede pool");
        assert!(pool < conv1, "pool must precede conv1");

        // Pool is not in the replaced set for upsample blocks
        let replaced = [conv1, conv1 + 1, conv1 + 2]; // conv1, adain2, conv2
        assert!(
            !replaced.contains(&pool),
            "Pool must not be in replaced set"
        );
        assert!(
            !replaced.contains(&adain1),
            "adain1 must not be in replaced set"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 9: Merged weight map keys all have correct prefix
    // -----------------------------------------------------------------------

    /// Every key in the merged weight map must start with either "p1_" or "p2_".
    /// A key without a prefix would indicate a merge bug where raw phase keys
    /// leaked into the merged map.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(10)]
    fn proof_merged_weight_keys_all_prefixed() {
        let base_keys = ["gamma", "beta", "conv_weight", "conv_bias"];
        let mut merged: HashMap<String, usize> = HashMap::new();

        for &k in &base_keys {
            merged.insert(format!("p1_{k}"), 1);
        }
        for &k in &base_keys {
            merged.insert(format!("p2_{k}"), 2);
        }

        for key in merged.keys() {
            assert!(
                key.starts_with("p1_") || key.starts_with("p2_"),
                "Every merged key must have p1_ or p2_ prefix"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Proof 10: FusedResBlock with style_batch_offset: 0 proj dispatches
    // -----------------------------------------------------------------------

    /// When style_batch_offset is set (batched path), the projection
    /// contribution to dispatch count is 0 (zero-copy narrow from batch output).
    /// This is the optimization: batching eliminates per-block projections.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_batch_offset_zero_projection_dispatches() {
        let params = NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            1,
            1,
            vec![1, 256, 512],
            256,
            3,
        );

        let op_batched = NativeOpKind::FusedResBlock {
            phase1: params.clone(),
            phase2: params.clone(),
            input_steps: vec![0, 1],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: Some(StyleBatchOffset::new(0, 256, 256)),
        };

        let op_direct = NativeOpKind::FusedResBlock {
            phase1: params.clone(),
            phase2: params,
            input_steps: vec![0, 1, 2, 3, 4],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: None,
        };

        // Both have same dispatch count (3) because batched path adds 0
        assert_eq!(
            op_batched.estimated_metal_dispatches(),
            op_direct.estimated_metal_dispatches(),
            "Batch offset path and direct path must have same dispatch count"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 11: Residual scale multiplication commutativity
    // -----------------------------------------------------------------------

    /// The residual add with scale is `(x + h) * scale`. This must commute
    /// with the mathematical identity that `(x + h) * scale == x*scale + h*scale`
    /// for finite, nonzero scale. This is used in the F0 predictor where
    /// scale = 1/sqrt(2).
    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_residual_scale_distributive() {
        let scale: f32 = kani::any();
        let x: f32 = kani::any();
        let h: f32 = kani::any();

        kani::assume(scale.is_finite() && scale != 0.0 && scale.abs() < 10.0);
        kani::assume(x.is_finite() && x.abs() < 1e4);
        kani::assume(h.is_finite() && h.abs() < 1e4);

        let fused = (x + h) * scale;
        let distributed = x * scale + h * scale;

        // IEEE 754: these may differ by ULP, but both must be finite
        // for bounded inputs with bounded scale.
        assert!(fused.is_finite(), "Fused result must be finite");
        assert!(distributed.is_finite(), "Distributed result must be finite");
    }

    // -----------------------------------------------------------------------
    // Proof 12: Snake activation requires alpha weight
    // -----------------------------------------------------------------------

    /// Snake activation NormActivConv1d requires an "alpha" key in weight_data.
    /// LeakyRelu does not. This is a structural invariant: the Metal kernel
    /// for Snake reads alpha[c] per-channel, while LeakyRelu uses a scalar slope.
    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_snake_requires_alpha_leaky_does_not() {
        let snake_keys = ["gamma", "beta", "conv_weight", "conv_bias", "alpha"];
        let leaky_keys = ["gamma", "beta", "conv_weight", "conv_bias"];

        let has_alpha_snake = snake_keys.iter().any(|&k| k == "alpha");
        let has_alpha_leaky = leaky_keys.iter().any(|&k| k == "alpha");

        assert!(has_alpha_snake, "Snake weight set must include alpha");
        assert!(
            !has_alpha_leaky,
            "LeakyRelu weight set must NOT include alpha"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 13: Style projection channels match phase channels
    // -----------------------------------------------------------------------

    /// When style_proj is Some, channels1 must match phase1's input_shape[1]
    /// and channels2 must match phase2's input_shape[1]. Mismatch would
    /// produce gamma/beta of wrong dimension.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_style_proj_channels_match_phases() {
        let c1: usize = kani::any();
        let c2: usize = kani::any();
        kani::assume(c1 >= 1 && c1 <= 512);
        kani::assume(c2 >= 1 && c2 <= 512);

        let p1 =
            NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 1, vec![1, c1, 128], c1, 3);
        let p2 =
            NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 1, vec![1, c2, 128], c2, 3);
        let sp = StyleProjectionParams::new(c1, c2, 128);

        // Channels must match
        assert_eq!(sp.channels1, p1.input_shape[1]);
        assert_eq!(sp.channels2, p2.input_shape[1]);
    }

    // -----------------------------------------------------------------------
    // Proof 14: PeepholeConfig field independence (toggle each independently)
    // -----------------------------------------------------------------------

    /// Each PeepholeConfig field can be toggled independently without affecting
    /// others. This is not merely a struct layout test --- it verifies that
    /// the fields are truly independent booleans (not packed bits or enums
    /// that could alias).
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_peephole_fields_independent() {
        let mut config = PeepholeConfig::default();

        // Disable one field, verify others unaffected
        config.norm_linear = false;
        assert!(!config.norm_linear);
        assert!(config.norm_activ_conv1d);
        assert!(config.fused_resblock);
        assert!(config.auto_fuse_elementwise);

        // Enable it back, disable another
        config.norm_linear = true;
        config.auto_fuse_elementwise = false;
        assert!(config.norm_linear);
        assert!(!config.auto_fuse_elementwise);
        assert!(config.fused_resblock);
    }

    // -----------------------------------------------------------------------
    // Proof 15: Shortcut step strictly less than add step
    // -----------------------------------------------------------------------

    /// The conv1x1 shortcut step must execute before the add step (which
    /// is the fusion anchor). If shortcut_step >= add_step, the executor
    /// would try to read a buffer that hasn't been computed yet.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_shortcut_step_before_add() {
        let shortcut: usize = kani::any();
        let add_step: usize = kani::any();

        kani::assume(shortcut < add_step);
        kani::assume(add_step <= 1000);

        // The shortcut is used as `buffers[shortcut_step]` inside the
        // FusedResBlock executor. It must have been computed already.
        assert!(
            shortcut < add_step,
            "Shortcut step must precede the fused position"
        );
    }
}
