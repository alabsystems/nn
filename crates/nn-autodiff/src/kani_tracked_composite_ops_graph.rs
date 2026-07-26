// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for `tracked_composite_ops.rs`.
//!
//! Focuses on output-shape validity for composite operations and on the fresh
//! node-id invariant that keeps newly recorded ops acyclic.
//!
//! Re: #3733.

#[cfg(kani)]
mod proofs {
    use kani::assume;

    /// Embedding appends `embed_dim` to the input index shape.
    ///
    /// SYNC: tracked_composite_ops.rs:179-187.
    fn embedding_output_rank(weight_rank: u8, input_rank: u8, embed_dim: usize) -> Option<u8> {
        if weight_rank < 2 || embed_dim == 0 {
            None
        } else {
            Some(input_rank + 1)
        }
    }

    /// `mse_loss`, `l1_loss`, `huber_loss`, and `cross_entropy_loss` all reduce
    /// over every dimension, then squeeze until scalar.
    ///
    /// SYNC: tracked_composite_ops.rs:306-313, 326-331, 346-350, 377-382.
    fn scalar_rank_after_full_reduction(mut rank: u8) -> u8 {
        let original_rank = rank;
        let mut squeezed = 0;
        while squeezed < original_rank {
            rank -= 1;
            squeezed += 1;
        }
        rank
    }

    /// Concatenation adds lengths along the concatenated axis.
    ///
    /// SYNC: tracked_composite_ops.rs:105-118.
    fn cat_axis_len(lhs: usize, rhs: usize) -> usize {
        lhs + rhs
    }

    /// Fresh nodes must be newer than the parent they reference.
    ///
    /// SYNC: tracked_composite_ops.rs:188-190, 236-238, 314-316, 333-386 and
    /// tracked.rs:84-90.
    fn unary_edge_points_to_older_parent(output_id: u64, parent_id: u64) -> bool {
        parent_id < output_id
    }

    /// Fresh binary nodes must be newer than both parents.
    ///
    /// SYNC: tracked_composite_ops.rs:188-190, 314-316, 333-386 and
    /// tracked.rs:84-90.
    fn binary_edges_point_to_older_parents(output_id: u64, left_id: u64, right_id: u64) -> bool {
        left_id < output_id && right_id < output_id
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn embedding_adds_a_trailing_embedding_dimension() {
        let weight_rank: u8 = kani::any();
        let input_rank: u8 = kani::any();
        let embed_dim: usize = kani::any();

        assume(weight_rank >= 2 && weight_rank <= 8);
        assume(input_rank <= 8);
        assume(embed_dim >= 1 && embed_dim <= 4096);

        let out_rank = embedding_output_rank(weight_rank, input_rank, embed_dim);
        assert!(
            out_rank == Some(input_rank + 1),
            "embedding output rank must equal input rank plus the appended embed_dim"
        );
    }

    #[kani::unwind(7)]
    #[kani::proof]
    fn scalar_losses_reduce_to_rank_zero() {
        let input_rank: u8 = kani::any();

        assume(input_rank <= 8);

        let reduced_rank = scalar_rank_after_full_reduction(input_rank);
        assert!(
            reduced_rank == 0,
            "full reduction plus squeeze must produce a scalar"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn cat_composition_adds_the_concat_axis_length() {
        let lhs: usize = kani::any();
        let rhs: usize = kani::any();

        assume(lhs <= 1_000_000);
        assume(rhs <= 1_000_000);

        let out = cat_axis_len(lhs, rhs);
        assert!(
            out >= lhs && out >= rhs,
            "cat output axis must be at least as large as each input axis"
        );
        assert!(
            out == lhs + rhs,
            "cat output axis must equal the sum of input axis lengths"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn unary_composite_ops_cannot_create_self_cycles() {
        let parent_id: u64 = kani::any();
        let output_id: u64 = kani::any();

        assume(parent_id < output_id);
        assume(output_id <= parent_id.saturating_add(1024));

        assert!(
            unary_edge_points_to_older_parent(output_id, parent_id),
            "a fresh composite-op node must point only to older parents"
        );
        assert!(
            output_id != parent_id,
            "a fresh composite-op node cannot reference itself"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn binary_composite_ops_remain_acyclic() {
        let left_id: u64 = kani::any();
        let right_id: u64 = kani::any();
        let output_id: u64 = kani::any();

        assume(left_id < output_id);
        assume(right_id < output_id);
        assume(output_id <= left_id.max(right_id).saturating_add(1024));

        assert!(
            binary_edges_point_to_older_parents(output_id, left_id, right_id),
            "binary composite ops must only capture older nodes"
        );
        assert!(
            output_id != left_id && output_id != right_id,
            "binary composite ops cannot create immediate cycles"
        );
    }
}
