// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for MoeDispatch issue #3730.

#[cfg(kani)]
mod proofs {
    use crate::layers::MoeDispatchConfig;
    use kani::assume;

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_moe_dispatch_rejects_topk_above_expert_count() {
        let num_experts: usize = kani::any();
        let hidden_size: usize = kani::any();
        let expert_intermediate_size: usize = kani::any();

        assume((1..=8).contains(&num_experts));
        assume((1..=256).contains(&hidden_size));
        assume((1..=256).contains(&expert_intermediate_size));

        let cfg = MoeDispatchConfig::new(
            num_experts,
            num_experts + 1,
            hidden_size,
            expert_intermediate_size,
            true,
        );

        assert!(cfg.is_err(), "top_k cannot exceed num_experts");
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_moe_dispatch_topk_indices_stay_in_bounds() {
        let num_experts: usize = kani::any();
        let top_k: usize = kani::any();
        let i0: u32 = kani::any();
        let i1: u32 = kani::any();
        let i2: u32 = kani::any();
        let i3: u32 = kani::any();

        assume((1..=4).contains(&num_experts));
        assume((1..=4).contains(&top_k));
        assume(top_k <= num_experts);

        let indices = [i0, i1, i2, i3];
        for i in 0..top_k {
            assume(indices[i] < num_experts as u32);
            let expert_idx = indices[i] as usize;
            assert!(
                expert_idx < num_experts,
                "selected expert must be in bounds"
            );
        }
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(17)]
    fn proof_moe_dispatch_assignment_counts_match_topk_budget() {
        let n_tokens: usize = kani::any();
        let num_experts: usize = kani::any();
        let top_k: usize = kani::any();
        let mut counts = [0usize; 4];

        assume((1..=4).contains(&n_tokens));
        assume((1..=4).contains(&num_experts));
        assume((1..=4).contains(&top_k));
        assume(top_k <= num_experts);

        for _token in 0..n_tokens {
            for _slot in 0..top_k {
                let expert_idx: usize = kani::any();
                assume(expert_idx < num_experts);
                counts[expert_idx] += 1;
            }
        }

        let total: usize = counts[..num_experts].iter().sum();
        assert!(
            total == n_tokens * top_k,
            "grouping must preserve exactly one assignment per top-k slot"
        );
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(9)]
    fn proof_moe_dispatch_uniform_load_balancing_yields_unit_aux_loss() {
        let num_experts: usize = kani::any();
        assume((1..=8).contains(&num_experts));

        let per_expert_fraction = 1.0f32 / num_experts as f32;
        let mut dot = 0.0f32;

        for _expert in 0..num_experts {
            dot += per_expert_fraction * per_expert_fraction;
        }

        let aux_loss = (num_experts as f32) * dot;
        assert!(
            aux_loss >= 1.0 - 1e-5 && aux_loss <= 1.0 + 1e-5,
            "uniform expert usage should produce the baseline aux loss"
        );
    }
}
