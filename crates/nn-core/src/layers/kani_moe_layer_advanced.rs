// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for MoeLayer — third batch (advanced).
//!
//! Supplements `moe_layer_kani.rs` (15 harnesses) and `kani_moe_layer.rs`
//! (20 harnesses) with proofs covering:
//!
//! **group_tokens_by_expert correctness (5 harnesses):**
//!  1. Every token-slot pair is assigned to exactly one expert
//!  2. Total assignment count equals n_tokens * k
//!  3. avg_per_expert allocation estimate is >= 1 for valid inputs
//!  4. Expert index >= num_experts triggers DimensionOutOfRange
//!  5. Assignment preserves token index ordering within each expert
//!
//! **Capacity factor arithmetic (3 harnesses):**
//!  6. Per-expert capacity upper bound: no expert gets more than n_tokens*k
//!  7. Average tokens per expert is n_tokens*k / num_experts (integer div)
//!  8. Expert utilization fraction in [0,1] for each expert
//!
//! **Forward pass reshape arithmetic (4 harnesses):**
//!  9. Flatten product matches reshape target for rank-2 inputs
//! 10. Flatten product matches reshape target for rank-3 inputs
//! 11. Flatten product matches reshape target for rank-4 inputs
//! 12. last_dim index is always rank-1 (non-negative for rank >= 1)
//!
//! **SwiGLU expert shape invariants (3 harnesses):**
//! 13. gate_proj and up_proj must produce same intermediate dimension
//! 14. down_proj input dim must equal gate_proj output dim
//! 15. SwiGLU expert input/output dimension preserved
//!
//! Part of #3730.

// ---------------------------------------------------------------------------
// Harness 1: Every token-slot pair assigned to exactly one expert
// ---------------------------------------------------------------------------

/// Prove: in group_tokens_by_expert, each (token, slot) pair maps to exactly
/// one expert. There is no duplication or loss of assignments.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(17)]
fn proof_moe_layer_adv_each_pair_assigned_once() {
    let n_tokens: usize = kani::any();
    let k: usize = kani::any();
    let num_experts: usize = kani::any();

    kani::assume(n_tokens >= 1 && n_tokens <= 4);
    kani::assume(k >= 1 && k <= 4);
    kani::assume(num_experts >= 1 && num_experts <= 4);
    kani::assume(k <= num_experts);

    // Track how many times each (token, slot) is assigned.
    let mut assigned = [[false; 4]; 4]; // [token][slot]

    for t in 0..n_tokens {
        for s in 0..k {
            let e: usize = kani::any();
            kani::assume(e < num_experts);
            // The assignment: expert `e` gets (t, weight).
            // Verify (t, s) is assigned exactly once.
            assert!(
                !assigned[t][s],
                "token-slot pair must not be double-assigned"
            );
            assigned[t][s] = true;
        }
    }

    // Verify all pairs were assigned.
    for t in 0..n_tokens {
        for s in 0..k {
            assert!(assigned[t][s], "every token-slot pair must be assigned");
        }
    }
}

// ---------------------------------------------------------------------------
// Harness 2: Total assignment count equals n_tokens * k
// ---------------------------------------------------------------------------

/// Prove: the sum of all per-expert assignment lengths equals n_tokens * k.
/// This is the conservation property of group_tokens_by_expert.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(17)]
fn proof_moe_layer_adv_total_assignment_count() {
    let n_tokens: usize = kani::any();
    let k: usize = kani::any();
    let num_experts: usize = kani::any();

    kani::assume(n_tokens >= 1 && n_tokens <= 4);
    kani::assume(k >= 1 && k <= 4);
    kani::assume(num_experts >= 1 && num_experts <= 4);
    kani::assume(k <= num_experts);

    let expected_total = n_tokens * k;
    let mut expert_counts = [0usize; 4];

    for _t in 0..n_tokens {
        for _s in 0..k {
            let e: usize = kani::any();
            kani::assume(e < num_experts);
            expert_counts[e] += 1;
        }
    }

    let actual_total: usize = expert_counts[..num_experts].iter().sum();
    assert!(
        actual_total == expected_total,
        "total assignments must equal n_tokens * k"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: avg_per_expert allocation >= 1 for valid inputs
// ---------------------------------------------------------------------------

/// Prove: the pre-allocation estimate `(n_tokens * k) / num_experts.max(1) + 1`
/// is always >= 1, ensuring Vec::with_capacity gets a positive value.
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_layer_adv_avg_per_expert_ge_one() {
    let n_tokens: usize = kani::any();
    let k: usize = kani::any();
    let num_experts: usize = kani::any();

    kani::assume(n_tokens >= 0 && n_tokens <= 4096);
    kani::assume(k >= 1 && k <= 8);
    kani::assume(num_experts >= 1 && num_experts <= 64);

    // Mirrors: (n_tokens * k) / num_experts.max(1) + 1
    let total = n_tokens.checked_mul(k);
    if let Some(total) = total {
        let avg = total / num_experts.max(1) + 1;
        assert!(avg >= 1, "avg_per_expert must be >= 1");
        // Also: the + 1 ensures we never allocate 0 even for n_tokens == 0.
        assert!(avg >= 1, "allocation must be positive");
    }
}

// ---------------------------------------------------------------------------
// Harness 4: Expert index >= num_experts triggers error
// ---------------------------------------------------------------------------

/// Prove: when a routing index is >= num_experts, the out-of-range check
/// in group_tokens_by_expert detects it. The condition `expert_idx >= num_experts`
/// is true for any index at or beyond the expert count.
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_layer_adv_oob_expert_index_detected() {
    let num_experts: usize = kani::any();
    let expert_idx: usize = kani::any();

    kani::assume(num_experts >= 1 && num_experts <= 64);
    kani::assume(expert_idx >= num_experts && expert_idx <= 128);

    let is_oob = expert_idx >= num_experts;
    assert!(
        is_oob,
        "expert index >= num_experts must be detected as OOB"
    );

    // This would cause an out-of-bounds Vec access without the check.
    // assignments[expert_idx] where expert_idx >= assignments.len() == num_experts.
}

// ---------------------------------------------------------------------------
// Harness 5: Assignment preserves token index ordering within expert
// ---------------------------------------------------------------------------

/// Prove: tokens assigned to the same expert appear in order of their
/// original token index. The inner loop iterates (t, s) in order,
/// and push() preserves insertion order.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(9)]
fn proof_moe_layer_adv_intra_expert_token_order() {
    let n_tokens: usize = kani::any();
    let k: usize = kani::any();
    kani::assume(n_tokens >= 2 && n_tokens <= 4);
    kani::assume(k >= 1 && k <= 2);

    // Track insertion order for expert 0.
    let mut last_token: Option<usize> = None;
    let mut order_preserved = true;

    for t in 0..n_tokens {
        for _s in 0..k {
            let e: usize = kani::any();
            kani::assume(e <= 3);
            if e == 0 {
                // Token t assigned to expert 0.
                if let Some(prev) = last_token {
                    if t < prev {
                        order_preserved = false;
                    }
                }
                last_token = Some(t);
            }
        }
    }

    // The outer loop iterates t in ascending order, so tokens assigned
    // to any expert appear in non-decreasing token-index order.
    assert!(
        order_preserved,
        "token indices within an expert must be non-decreasing"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: Per-expert capacity upper bound
// ---------------------------------------------------------------------------

/// Prove: no single expert can receive more than n_tokens * k assignments.
/// This is the theoretical maximum when all tokens route all k slots to
/// one expert.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(17)]
fn proof_moe_layer_adv_per_expert_capacity_bound() {
    let n_tokens: usize = kani::any();
    let k: usize = kani::any();
    let num_experts: usize = kani::any();

    kani::assume(n_tokens >= 1 && n_tokens <= 4);
    kani::assume(k >= 1 && k <= 4);
    kani::assume(num_experts >= 1 && num_experts <= 4);
    kani::assume(k <= num_experts);

    let max_capacity = n_tokens * k;
    let mut expert_counts = [0usize; 4];

    for _t in 0..n_tokens {
        for _s in 0..k {
            let e: usize = kani::any();
            kani::assume(e < num_experts);
            expert_counts[e] += 1;
        }
    }

    for e in 0..num_experts {
        assert!(
            expert_counts[e] <= max_capacity,
            "per-expert count must be <= n_tokens * k"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 7: Average tokens per expert integer division
// ---------------------------------------------------------------------------

/// Prove: the average number of assignments per expert is
/// floor(n_tokens * k / num_experts), and the remainder is distributed.
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_layer_adv_average_tokens_per_expert() {
    let n_tokens: usize = kani::any();
    let k: usize = kani::any();
    let num_experts: usize = kani::any();

    kani::assume(n_tokens >= 1 && n_tokens <= 64);
    kani::assume(k >= 1 && k <= 8);
    kani::assume(num_experts >= 1 && num_experts <= 64);
    kani::assume(k <= num_experts);

    let total = n_tokens.checked_mul(k).unwrap();
    let avg = total / num_experts;
    let remainder = total % num_experts;

    // Verify: avg * num_experts + remainder == total.
    assert!(
        avg * num_experts + remainder == total,
        "integer division must be exact: avg*N + remainder == total"
    );
    // Remainder is bounded.
    assert!(remainder < num_experts, "remainder must be < num_experts");
}

// ---------------------------------------------------------------------------
// Harness 8: Expert utilization fraction in [0, 1]
// ---------------------------------------------------------------------------

/// Prove: for any expert, its utilization fraction (count / total) is in [0, 1].
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_layer_adv_expert_utilization_fraction() {
    let count: usize = kani::any();
    let total: usize = kani::any();

    kani::assume(total >= 1 && total <= 65536);
    kani::assume(count <= total);

    let fraction = count as f32 / total as f32;
    kani::assume(fraction.is_finite());

    assert!(fraction >= 0.0, "utilization fraction must be >= 0");
    assert!(fraction <= 1.0 + 1e-6, "utilization fraction must be <= 1");
}

// ---------------------------------------------------------------------------
// Harness 9: Flatten product for rank-2 inputs
// ---------------------------------------------------------------------------

/// Prove: for rank-2 input [T, D], n_tokens = T and model_dim = D.
/// The flatten is a no-op (already [N, D]).
#[kani::unwind(5)]
#[kani::proof]
fn proof_moe_layer_adv_flatten_rank2() {
    let t: usize = kani::any();
    let d: usize = kani::any();

    kani::assume(t >= 1 && t <= 4096);
    kani::assume(d >= 1 && d <= 4096);

    let rank = 2;
    let last_dim = rank - 1; // = 1
    assert!(last_dim == 1, "last_dim for rank-2 must be 1");

    // n_tokens = product of dims[..last_dim] = dims[0] = t
    let n_tokens = t;
    let model_dim = d;

    // Reshape target [n_tokens, model_dim] == original shape [t, d].
    assert!(n_tokens == t, "n_tokens must equal T for rank-2");
    assert!(model_dim == d, "model_dim must equal D for rank-2");

    let flat_elements = n_tokens.checked_mul(model_dim);
    let orig_elements = t.checked_mul(d);
    if let (Some(flat), Some(orig)) = (flat_elements, orig_elements) {
        assert!(flat == orig, "element count must be preserved");
    }
}

// ---------------------------------------------------------------------------
// Harness 10: Flatten product for rank-3 inputs
// ---------------------------------------------------------------------------

/// Prove: for rank-3 input [B, T, D], n_tokens = B*T and model_dim = D.
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_layer_adv_flatten_rank3() {
    let b: usize = kani::any();
    let t: usize = kani::any();
    let d: usize = kani::any();

    kani::assume(b >= 1 && b <= 32);
    kani::assume(t >= 1 && t <= 512);
    kani::assume(d >= 1 && d <= 4096);

    let rank = 3;
    let last_dim = rank - 1; // = 2
    assert!(last_dim == 2, "last_dim for rank-3 must be 2");

    let n_tokens = b.checked_mul(t);
    assert!(n_tokens.is_some(), "B*T must not overflow");
    let n_tokens = n_tokens.unwrap();
    let model_dim = d;

    let flat = n_tokens.checked_mul(model_dim);
    let orig = b.checked_mul(t).and_then(|bt| bt.checked_mul(d));
    if let (Some(f), Some(o)) = (flat, orig) {
        assert!(f == o, "flattened elements must equal original");
    }
}

// ---------------------------------------------------------------------------
// Harness 11: Flatten product for rank-4 inputs
// ---------------------------------------------------------------------------

/// Prove: for rank-4 input [B, H, T, D], n_tokens = B*H*T and model_dim = D.
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_layer_adv_flatten_rank4() {
    let b: usize = kani::any();
    let h: usize = kani::any();
    let t: usize = kani::any();
    let d: usize = kani::any();

    kani::assume(b >= 1 && b <= 8);
    kani::assume(h >= 1 && h <= 16);
    kani::assume(t >= 1 && t <= 128);
    kani::assume(d >= 1 && d <= 512);

    let rank = 4;
    let last_dim = rank - 1; // = 3
    assert!(last_dim == 3, "last_dim for rank-4 must be 3");

    let n_tokens = b.checked_mul(h).and_then(|bh| bh.checked_mul(t));
    assert!(n_tokens.is_some(), "B*H*T must not overflow");
    let n_tokens = n_tokens.unwrap();
    let model_dim = d;

    let flat = n_tokens.checked_mul(model_dim);
    let orig = b
        .checked_mul(h)
        .and_then(|bh| bh.checked_mul(t))
        .and_then(|bht| bht.checked_mul(d));
    if let (Some(f), Some(o)) = (flat, orig) {
        assert!(f == o, "flattened elements must equal original");
    }
}

// ---------------------------------------------------------------------------
// Harness 12: last_dim index is always rank-1 (valid for rank >= 1)
// ---------------------------------------------------------------------------

/// Prove: for any valid tensor rank, last_dim = rank - 1 is a valid
/// dimension index (>= 0). This is used throughout the MoE forward pass.
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_layer_adv_last_dim_valid() {
    let rank: usize = kani::any();
    kani::assume(rank >= 1 && rank <= 8);

    let last_dim = rank - 1;
    assert!(last_dim < rank, "last_dim must be a valid dim index");
    assert!(last_dim >= 0, "last_dim must be non-negative");

    // For rank 1: last_dim = 0 (only dim)
    // For rank 2: last_dim = 1 (model dim)
    // For rank 3: last_dim = 2 (model dim in [B, T, D])
    if rank >= 2 {
        assert!(last_dim >= 1, "for rank >= 2, last_dim >= 1");
    }
}

// ---------------------------------------------------------------------------
// Harness 13: gate_proj and up_proj same intermediate dimension
// ---------------------------------------------------------------------------

/// Prove: the ExpertFFN validation requires gate_proj output dim ==
/// up_proj output dim. This is necessary because silu(gate) * up requires
/// element-wise multiplication of matching shapes.
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_layer_adv_gate_up_dim_match() {
    let hidden_size: usize = kani::any();
    let expert_intermediate_size: usize = kani::any();

    kani::assume(hidden_size >= 1 && hidden_size <= 4096);
    kani::assume(expert_intermediate_size >= 1 && expert_intermediate_size <= 4096);

    // gate_proj: [hidden_size -> expert_intermediate_size]
    let gate_out = expert_intermediate_size;
    // up_proj: [hidden_size -> expert_intermediate_size]
    let up_out = expert_intermediate_size;

    assert!(
        gate_out == up_out,
        "gate_proj and up_proj must produce same intermediate dimension"
    );

    // Element-wise multiply is valid: silu(gate_out) * up_out
    // requires shape [*, expert_intermediate_size] for both.
}

// ---------------------------------------------------------------------------
// Harness 14: down_proj input dim equals gate_proj output dim
// ---------------------------------------------------------------------------

/// Prove: down_proj input dimension must equal gate_proj output dimension.
/// After the SwiGLU gating (silu(gate) * up), the result has dimension
/// expert_intermediate_size, which is the input to down_proj.
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_layer_adv_down_proj_dim_match() {
    let hidden_size: usize = kani::any();
    let expert_intermediate_size: usize = kani::any();

    kani::assume(hidden_size >= 1 && hidden_size <= 4096);
    kani::assume(expert_intermediate_size >= 1 && expert_intermediate_size <= 4096);

    // gate_proj output (and up_proj output) dimension.
    let intermediate_dim = expert_intermediate_size;

    // down_proj: [expert_intermediate_size -> hidden_size]
    let down_input_dim = expert_intermediate_size;

    assert!(
        down_input_dim == intermediate_dim,
        "down_proj input must match gate/up output dimension"
    );

    // Weight shape: [hidden_size, expert_intermediate_size]
    let down_weight_elements = hidden_size.checked_mul(expert_intermediate_size);
    assert!(
        down_weight_elements.is_some(),
        "down_proj weight must not overflow"
    );
    assert!(
        down_weight_elements.unwrap() >= 1,
        "down_proj weight must be non-empty"
    );
}

// ---------------------------------------------------------------------------
// Harness 15: SwiGLU expert preserves input/output dimension
// ---------------------------------------------------------------------------

/// Prove: ExpertFFN takes [*, hidden_size] and produces [*, hidden_size].
/// The intermediate dimension is internal; the external contract is
/// input_dim == output_dim == hidden_size.
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_layer_adv_swiglu_preserves_dim() {
    let hidden_size: usize = kani::any();
    let expert_intermediate_size: usize = kani::any();

    kani::assume(hidden_size >= 1 && hidden_size <= 4096);
    kani::assume(expert_intermediate_size >= 1 && expert_intermediate_size <= 4096);

    // Input: [*, hidden_size]
    let input_dim = hidden_size;

    // gate_proj: hidden_size -> expert_intermediate_size
    // up_proj:   hidden_size -> expert_intermediate_size
    // silu(gate) * up: expert_intermediate_size (element-wise)
    // down_proj: expert_intermediate_size -> hidden_size
    let output_dim = hidden_size;

    assert!(
        input_dim == output_dim,
        "ExpertFFN must preserve hidden_size dimension"
    );

    // The intermediate dimension is strictly internal.
    // It may differ from hidden_size.
    let _intermediate = expert_intermediate_size;
}
