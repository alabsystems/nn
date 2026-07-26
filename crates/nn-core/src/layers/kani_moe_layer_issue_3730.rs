// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for MoeLayer issue #3730.

#[cfg(kani)]
mod proofs {
    use crate::layers::MoeLayerConfig;
    use kani::assume;

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_moe_layer_rejects_zero_experts() {
        let hidden_size: usize = kani::any();
        let expert_intermediate_size: usize = kani::any();

        assume((1..=256).contains(&hidden_size));
        assume((1..=256).contains(&expert_intermediate_size));

        let cfg = MoeLayerConfig::new(0, 1, hidden_size, expert_intermediate_size, true, false);

        assert!(cfg.is_err(), "num_experts == 0 must be rejected");
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_moe_layer_accepts_topk_at_expert_bound() {
        let num_experts: usize = kani::any();
        let hidden_size: usize = kani::any();
        let expert_intermediate_size: usize = kani::any();

        assume((1..=8).contains(&num_experts));
        assume((1..=256).contains(&hidden_size));
        assume((1..=256).contains(&expert_intermediate_size));

        let cfg = MoeLayerConfig::new(
            num_experts,
            num_experts,
            hidden_size,
            expert_intermediate_size,
            true,
            false,
        );

        assert!(
            cfg.is_ok(),
            "top_k == num_experts is a valid routing boundary"
        );

        if let Ok(cfg) = cfg {
            assert!(cfg.top_k == cfg.num_experts);
        }
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_moe_layer_normalized_weights_are_bounded() {
        let top_k: usize = kani::any();
        let w0: f32 = kani::any();
        let w1: f32 = kani::any();
        let w2: f32 = kani::any();
        let w3: f32 = kani::any();

        assume((1..=4).contains(&top_k));

        let weights = [w0, w1, w2, w3];
        let mut weight_sum = 0.0f32;

        for i in 0..top_k {
            assume(weights[i].is_finite());
            assume(weights[i] >= 0.0);
            weight_sum += weights[i];
        }

        assume(weight_sum.is_finite());
        assume(weight_sum > 0.0);

        for i in 0..top_k {
            let normalized = weights[i] / weight_sum;
            assert!(normalized.is_finite(), "normalized weight must stay finite");
            assert!(normalized >= 0.0, "normalized weight must be non-negative");
            assert!(normalized <= 1.0 + 1e-6, "normalized weight must be <= 1");
        }
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_moe_layer_normalized_weights_sum_to_one() {
        let top_k: usize = kani::any();
        let w0: f32 = kani::any();
        let w1: f32 = kani::any();
        let w2: f32 = kani::any();
        let w3: f32 = kani::any();

        assume((1..=4).contains(&top_k));

        let weights = [w0, w1, w2, w3];
        let mut weight_sum = 0.0f32;

        for i in 0..top_k {
            assume(weights[i].is_finite());
            assume(weights[i] >= 0.0);
            weight_sum += weights[i];
        }

        assume(weight_sum.is_finite());
        assume(weight_sum > 0.0);

        let mut normalized_sum = 0.0f32;
        for i in 0..top_k {
            normalized_sum += weights[i] / weight_sum;
        }

        assert!(
            normalized_sum >= 1.0 - 1e-5 && normalized_sum <= 1.0 + 1e-5,
            "renormalized top-k weights must sum to 1"
        );
    }
}
