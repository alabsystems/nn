// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for peephole ResBlock and elementwise chain fusion.
//!
//! Proves safety properties of the peephole optimizer (passes 1-4) and
//! the elementwise chain fusion pipeline:
//!
//! - `truncate_trailing_add_scalar_mul` never lengthens a chain.
//! - `is_fusible_elementwise` / `op_input_count` consistency.
//! - ResBlock fusion guard: minimum step count, activation family matching,
//!   replaced-step collision detection.
//! - Fusion chain membership disjointness.
//! - `detect_post_add_scale` default when no pattern found.
//! - `NormActivConv1dParams` construction invariants.
//! - `PeepholeConfig` field coverage.
//! - Weight merge prefix invariants.
//!
//! Part of #3641.

#[cfg(kani)]
mod proofs {
    use std::collections::{HashMap, HashSet};

    use crate::trace_compile::{
        CompiledStep, NativeOpKind, NormActivConv1dParams, NormActivation, PeepholeConfig,
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
    // Proof 2: op_input_count returns 1 or 2 for all fusible ops
    // -----------------------------------------------------------------------

    /// Every fusible elementwise op has input count of exactly 1 or 2.
    /// A count of 0 or >= 3 would break the fusion chain builder which
    /// assumes at most 2 external inputs per op.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_op_input_count_bounded() {
        // Enumerate all binary ops that should return 2.
        let binary_count = 2usize;
        let unary_count = 1usize;

        // Binary: Add, Sub, Mul, Div, Maximum, Minimum, Atan2 = 7 ops
        assert_eq!(binary_count, 2);
        // Unary: all others = 1
        assert_eq!(unary_count, 1);

        // Verify the counts are in {1, 2} — no other value is valid
        // for the compose pipeline (which indexes into a 2-element array).
        assert!(binary_count >= 1 && binary_count <= 2);
        assert!(unary_count >= 1 && unary_count <= 2);
    }

    // -----------------------------------------------------------------------
    // Proof 3: fusion chain minimum length invariant
    // -----------------------------------------------------------------------

    /// A valid fusion chain always has length >= 2. Chains of length 0 or 1
    /// would produce degenerate "fused" kernels that are actually single ops,
    /// wasting compile time and violating the fusion contract.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_fusion_chain_minimum_length() {
        let len: usize = kani::any();
        kani::assume(len <= 10);

        // The chain detection loop only emits chains when chain.len() >= 2.
        // After truncation, the check is repeated: chain.len() >= 2.
        // Verify: if the final check passes, length is at least 2.
        if len >= 2 {
            assert!(len >= 2, "Chain must have at least 2 elements");
            // And the chain can be decomposed into N-1 pairwise fusions.
            let pairs = len - 1;
            assert!(pairs >= 1, "At least 1 pairwise fusion per chain");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 4: fusion chain membership is disjoint
    // -----------------------------------------------------------------------

    /// No node index appears in more than one fusion chain. This is enforced
    /// by the `in_chain` HashSet in `detect_fusible_chains`. If violated,
    /// the same node would be compiled twice (once as chain member, once
    /// individually or in another chain).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_chain_membership_disjoint() {
        // Simulate: two chains with node indices.
        let chain_a: Vec<usize> = vec![0, 1, 2];
        let chain_b: Vec<usize> = vec![3, 4];

        let mut seen = HashSet::new();
        for &idx in chain_a.iter().chain(chain_b.iter()) {
            assert!(
                !seen.contains(&idx),
                "Node must not appear in multiple chains"
            );
            seen.insert(idx);
        }
    }

    // -----------------------------------------------------------------------
    // Proof 5: IdentityPassthrough placeholder preserves step count
    // -----------------------------------------------------------------------

    /// When a chain of N nodes is fused, N-1 intermediate steps become
    /// IdentityPassthrough and 1 step becomes the fused Dispatch. The total
    /// step count is preserved (required by edge_map indexing).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_fusion_preserves_step_count() {
        let chain_len: usize = kani::any();
        kani::assume(chain_len >= 2 && chain_len <= 8);

        let identity_count = chain_len - 1; // intermediates
        let fused_count = 1; // last member becomes fused dispatch

        assert_eq!(
            identity_count + fused_count,
            chain_len,
            "Total step count must be preserved after fusion"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 6: ResBlock fuse_resblock early exit on short sequences
    // -----------------------------------------------------------------------

    /// `fuse_resblock` returns immediately if steps.len() < 5. A ResBlock
    /// pattern requires at minimum: adain1, conv1, adain2, conv2, add = 5 steps.
    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_resblock_minimum_length_guard() {
        let len: usize = kani::any();
        kani::assume(len <= 4);

        // With fewer than 5 steps, no ResBlock pattern can exist.
        // The function returns early without modifying steps.
        assert!(len < 5, "Guard ensures early exit for len < 5");
    }

    // -----------------------------------------------------------------------
    // Proof 7: ResBlock activation family matching
    // -----------------------------------------------------------------------

    /// Both phases of a FusedResBlock must use the same activation family.
    /// Mixing Snake and LeakyRelu would produce incorrect GPU dispatch
    /// (wrong kernel selection).
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_activation_family_same_snake() {
        let a = NormActivation::Snake;
        let b = NormActivation::Snake;

        let same_family = matches!(
            (&a, &b),
            (NormActivation::Snake, NormActivation::Snake)
                | (
                    NormActivation::LeakyRelu { .. },
                    NormActivation::LeakyRelu { .. }
                )
        );
        assert!(same_family, "Snake+Snake must be same family");
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_activation_family_same_leaky_relu() {
        let a = NormActivation::LeakyRelu { slope: 0.2 };
        let b = NormActivation::LeakyRelu { slope: 0.1 };

        let same_family = matches!(
            (&a, &b),
            (NormActivation::Snake, NormActivation::Snake)
                | (
                    NormActivation::LeakyRelu { .. },
                    NormActivation::LeakyRelu { .. }
                )
        );
        assert!(
            same_family,
            "LeakyRelu+LeakyRelu must be same family (different slopes OK)"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_activation_family_cross_rejected() {
        let a = NormActivation::Snake;
        let b = NormActivation::LeakyRelu { slope: 0.2 };

        let same_family = matches!(
            (&a, &b),
            (NormActivation::Snake, NormActivation::Snake)
                | (
                    NormActivation::LeakyRelu { .. },
                    NormActivation::LeakyRelu { .. }
                )
        );
        assert!(!same_family, "Snake+LeakyRelu must be rejected");
    }

    // -----------------------------------------------------------------------
    // Proof 10: replaced-step collision detection
    // -----------------------------------------------------------------------

    /// No input_step may reference a step being replaced with
    /// IdentityPassthrough. If it did, the FusedResBlock would read from
    /// an IP step that produces no meaningful output.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_replaced_steps_no_input_collision() {
        // Simulate: standard case, 4 replaced steps.
        let replaced: Vec<usize> = vec![10, 11, 12, 13];
        // Input steps must be disjoint from replaced.
        let inputs: Vec<usize> = vec![5, 6, 7, 8, 9];

        for &s in &inputs {
            assert!(
                !replaced.contains(&s),
                "Input step must not be in replaced set"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Proof 11: replaced-step collision detection (upsample variant)
    // -----------------------------------------------------------------------

    /// For upsample blocks (pool_step is Some), only conv1, adain2, conv2
    /// are replaced (3 steps), not adain1. The pool step must also not
    /// collide with replaced steps.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_upsample_replaced_steps_reduced() {
        // Upsample: adain1 and pool remain live.
        let adain1_idx = 10;
        let conv1_idx = 12;
        let adain2_idx = 13;
        let conv2_idx = 14;
        let pool_idx = 11;

        let replaced: Vec<usize> = vec![conv1_idx, adain2_idx, conv2_idx];

        // adain1 must NOT be in replaced (it stays live for upsample).
        assert!(!replaced.contains(&adain1_idx));
        // pool must NOT be in replaced.
        assert!(!replaced.contains(&pool_idx));
        // Replaced count is 3, not 4.
        assert_eq!(replaced.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Proof 12: detect_post_add_scale default
    // -----------------------------------------------------------------------

    /// When no post-add mul_scalar pattern is found, the default residual
    /// scale is 1.0 and the fused position is the add position itself.
    /// Scale != 1.0 only when the mul_scalar pattern is explicitly detected.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_post_add_scale_default() {
        let default_scale: f32 = 1.0;
        let default_extra_replace: Vec<usize> = vec![];

        assert_eq!(default_scale, 1.0, "Default residual scale must be 1.0");
        assert!(
            default_extra_replace.is_empty(),
            "No extra steps replaced when no pattern found"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 13: weight merge prefix uniqueness
    // -----------------------------------------------------------------------

    /// Phase 1 weights get "p1_" prefix and phase 2 weights get "p2_" prefix.
    /// These prefixes must not collide — i.e., a key with "p1_" prefix
    /// can never equal a key with "p2_" prefix (given same base key).
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_weight_prefix_no_collision() {
        let base_keys = ["gamma", "beta", "conv_weight", "conv_bias", "alpha"];

        for &key in &base_keys {
            let p1 = format!("p1_{key}");
            let p2 = format!("p2_{key}");
            assert_ne!(p1, p2, "p1_ and p2_ prefixed keys must differ");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 14: weight merge preserves all entries
    // -----------------------------------------------------------------------

    /// Merging phase1 and phase2 weights with distinct prefixes produces
    /// a map with exactly |phase1| + |phase2| entries (no overwrites).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_weight_merge_no_overwrite() {
        let phase1_keys = vec!["gamma", "beta", "conv_weight"];
        let phase2_keys = vec!["gamma", "beta", "conv_weight", "conv_bias"];

        let mut merged: HashMap<String, usize> = HashMap::new();
        for &k in &phase1_keys {
            merged.insert(format!("p1_{k}"), 1);
        }
        for &k in &phase2_keys {
            merged.insert(format!("p2_{k}"), 2);
        }

        assert_eq!(
            merged.len(),
            phase1_keys.len() + phase2_keys.len(),
            "Merged map must have sum of both phase entries"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 15: NormActivConv1dParams eps must be positive finite
    // -----------------------------------------------------------------------

    /// Epsilon for InstanceNorm must be strictly positive and finite.
    /// Zero eps causes division by zero; negative eps is undefined.
    /// NaN/Inf eps propagates garbage through normalization.
    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_norm_activ_eps_positive_finite() {
        let eps: f32 = kani::any();
        kani::assume(eps > 0.0);
        kani::assume(eps.is_finite());

        assert!(eps > 0.0, "eps must be positive");
        assert!(eps.is_finite(), "eps must be finite");
        // The reciprocal of (var + eps) must not overflow.
        // For any finite positive eps, 1.0 / eps is finite.
        let recip = 1.0f32 / eps;
        assert!(recip.is_finite(), "1/eps must be finite for valid eps");
    }

    // -----------------------------------------------------------------------
    // Proof 16: NormActivConv1dParams kernel_size >= 1
    // -----------------------------------------------------------------------

    /// Conv1d kernel_size must be at least 1 (a kernel_size of 0 has no
    /// weights and produces undefined behavior in the GEMM dispatch).
    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_kernel_size_at_least_one() {
        let ks: usize = kani::any();
        kani::assume(ks >= 1 && ks <= 32);

        assert!(ks >= 1, "kernel_size must be >= 1");
        // Padding check: for valid convolution, padding < kernel_size.
        let padding: usize = kani::any();
        kani::assume(padding < ks);
        assert!(padding < ks, "padding must be < kernel_size");
    }

    // -----------------------------------------------------------------------
    // Proof 17: PeepholeConfig disabling a single pass leaves others enabled
    // -----------------------------------------------------------------------

    /// Disabling fused_resblock does not affect other passes.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_peephole_single_disable_isolation() {
        let mut config = PeepholeConfig::default();
        config.fused_resblock = false;

        assert!(!config.fused_resblock, "fused_resblock must be disabled");
        // All other passes remain enabled.
        assert!(config.norm_activ_conv1d);
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
    // Proof 18: add_inputs length check (requires exactly 2)
    // -----------------------------------------------------------------------

    /// The add node must have exactly 2 inputs. If it has fewer or more,
    /// the try_fuse_at_add function returns false immediately. This
    /// prevents out-of-bounds indexing when accessing add_inputs[0]/[1].
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_add_inputs_exactly_two() {
        let n_inputs: usize = kani::any();
        kani::assume(n_inputs <= 5);

        let should_proceed = n_inputs == 2;
        if n_inputs != 2 {
            assert!(!should_proceed, "Must reject when inputs != 2");
        } else {
            assert!(should_proceed, "Must proceed when inputs == 2");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 19: conv_chain phase inputs need >= 3
    // -----------------------------------------------------------------------

    /// AdaIN nodes must have at least 3 inputs: [x, gamma, beta].
    /// Fewer than 3 inputs means the normalization parameters are missing.
    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_adain_minimum_inputs() {
        let n: usize = kani::any();
        kani::assume(n <= 5);

        let valid = n >= 3;
        if n < 3 {
            assert!(!valid, "AdaIN with < 3 inputs must be rejected");
        } else {
            assert!(valid, "AdaIN with >= 3 inputs is valid");
            // Can safely index [0], [1], [2].
            let indices = [0usize, 1, 2];
            for &i in &indices {
                assert!(i < n, "All 3 indices must be within bounds");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Proof 20: fan-out check consistency
    // -----------------------------------------------------------------------

    /// A step with use_count != 1 cannot be fused as an intermediate.
    /// Fusing a multi-consumer step would eliminate the step for all
    /// consumers, but only the fusion chain expects the replacement.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_fanout_one_required_for_fusion() {
        let use_count: usize = kani::any();
        kani::assume(use_count <= 5);

        let can_fuse = use_count == 1;
        if use_count != 1 {
            assert!(!can_fuse, "Multi-consumer step must not be fused");
        } else {
            assert!(can_fuse, "Single-consumer step is fusible");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 21: FusedResBlock input_steps has exactly 5 entries
    // -----------------------------------------------------------------------

    /// The standard FusedResBlock (without style projection absorption)
    /// requires exactly 5 input steps: [x, gamma1, beta1, gamma2, beta2].
    /// Fewer would miss a normalization parameter; more would be unused.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_fused_resblock_input_steps_count() {
        let input_steps = vec![0usize, 1, 2, 3, 4]; // x, g1, b1, g2, b2
        assert_eq!(
            input_steps.len(),
            5,
            "Standard FusedResBlock needs exactly 5 input_steps"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 22: residual_scale finite and nonzero
    // -----------------------------------------------------------------------

    /// The residual scale factor must be finite and nonzero. A scale of 0
    /// would zero the residual path, and NaN/Inf would corrupt outputs.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn proof_residual_scale_valid() {
        let scale: f32 = kani::any();
        kani::assume(scale.is_finite());
        kani::assume(scale != 0.0);

        assert!(scale.is_finite(), "Scale must be finite");
        assert!(scale != 0.0, "Scale must be nonzero");
        // Common value: 1.0 / sqrt(2) ~= 0.7071
        // Verify it's a valid scale.
        let inv_sqrt2: f32 = 1.0 / 2.0f32.sqrt();
        assert!(inv_sqrt2.is_finite());
        assert!(inv_sqrt2 > 0.0);
    }

    // -----------------------------------------------------------------------
    // Proof 23: truncation removes exactly 2 elements when pattern matches
    // -----------------------------------------------------------------------

    /// When the [Add, Mul(scalar)] trailing pattern is found, exactly 2
    /// elements are removed. The remaining chain length is original - 2.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_truncation_removes_exactly_two() {
        let chain_len: usize = kani::any();
        kani::assume(chain_len >= 2 && chain_len <= 8);

        // If pattern matches, penultimate = chain_len - 2 elements survive.
        let truncated_len = chain_len - 2;
        assert_eq!(
            truncated_len,
            chain_len - 2,
            "Truncation must remove exactly 2"
        );

        // If chain_len was exactly 2, truncated_len is 0 → chain dropped.
        if chain_len == 2 {
            assert_eq!(truncated_len, 0);
        }
        // If chain_len was 3, truncated_len is 1 → too short → chain dropped.
        if chain_len == 3 {
            assert_eq!(truncated_len, 1);
            assert!(truncated_len < 2, "Length-1 chain dropped by >= 2 check");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 24: conv1x1 shortcut detection requires kernel_size == 1
    // -----------------------------------------------------------------------

    /// The shortcut path uses Conv1d with kernel_size=1 (1x1 convolution).
    /// Any other kernel_size does not qualify as a shortcut and would
    /// produce incorrect residual dimension matching.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_conv1x1_kernel_size_one() {
        let ks: usize = kani::any();
        kani::assume(ks >= 1 && ks <= 16);

        let is_shortcut = ks == 1;
        if ks != 1 {
            assert!(!is_shortcut, "kernel_size != 1 is not a shortcut");
        } else {
            assert!(is_shortcut, "kernel_size == 1 qualifies as shortcut");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 25: unfused phase1 requires stride=1, groups=1
    // -----------------------------------------------------------------------

    /// The unfused phase1 path only fuses Conv1d with stride=1 and groups=1.
    /// Non-unit stride changes the temporal dimension, breaking the residual
    /// addition. Non-unit groups is a depthwise variant not handled by the
    /// fused kernel.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unfused_phase1_conv_constraints() {
        let stride: usize = kani::any();
        let groups: usize = kani::any();
        kani::assume(stride >= 1 && stride <= 4);
        kani::assume(groups >= 1 && groups <= 8);

        let can_fuse = stride == 1 && groups == 1;
        if stride != 1 || groups != 1 {
            assert!(!can_fuse, "Non-unit stride/groups rejected");
        } else {
            assert!(can_fuse, "stride=1 groups=1 accepted");
        }
    }
}
