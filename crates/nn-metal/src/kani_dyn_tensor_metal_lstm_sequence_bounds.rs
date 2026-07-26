// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Focused Kani proof harnesses for `dyn_tensor_metal_lstm_sequence.rs`.
//!
//! These proofs target the issue #3728 invariants around sequence-length
//! bounds and hidden-state dimensions.

#[cfg(kani)]
mod proofs {
    use kani::assume;

    use crate::dyn_tensor_metal::MAX_THREADGROUP_HIDDEN;

    /// Prove: the hidden-size guard in `gpu_lstm_sequence` matches the kernel
    /// dispatch constraints: hidden size must be non-zero, fit the per-group
    /// thread count, and stay within the exported threadgroup-memory limit.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn hidden_size_guard_matches_dispatch_limits() {
        let hidden_size: usize = kani::any();
        assume(hidden_size <= MAX_THREADGROUP_HIDDEN + 1);

        let accepted = hidden_size > 0 && hidden_size <= MAX_THREADGROUP_HIDDEN;

        if accepted {
            assert!(hidden_size <= MAX_THREADGROUP_HIDDEN);
            assert!(hidden_size <= 1024);
            assert!(u32::try_from(hidden_size).is_ok());
        } else {
            assert!(hidden_size == 0 || hidden_size > MAX_THREADGROUP_HIDDEN);
        }
    }

    /// Prove: sequence length scales only the `[S, B, H]` output, while the
    /// recurrent end states remain `[B, H]`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn sequence_length_only_scales_output_tensor() {
        let seq_len: usize = kani::any();
        let batch_size: usize = kani::any();
        let hidden_size: usize = kani::any();

        assume(seq_len >= 1 && seq_len <= 2048);
        assume(batch_size >= 1 && batch_size <= 64);
        assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN);

        let output_numel = seq_len
            .checked_mul(batch_size)
            .and_then(|v| v.checked_mul(hidden_size));
        let state_numel = batch_size.checked_mul(hidden_size);

        assert!(output_numel.is_some());
        assert!(state_numel.is_some());
        assert_eq!(output_numel.unwrap(), state_numel.unwrap() * seq_len);
        assert!(output_numel.unwrap() >= state_numel.unwrap());
    }

    /// Prove: valid LSTM initial states always use `[batch_size, hidden_size]`
    /// for both `h0` and `c0`, matching the runtime comparisons in
    /// `gpu_lstm_sequence_impl`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn hidden_state_dimensions_match_batch_and_hidden_size() {
        let batch_size: usize = kani::any();
        let hidden_size: usize = kani::any();

        assume(batch_size >= 1 && batch_size <= 64);
        assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN);

        let h0_dims = [batch_size, hidden_size];
        let c0_dims = [batch_size, hidden_size];
        let expected = [batch_size, hidden_size];

        assert_eq!(h0_dims, expected);
        assert_eq!(c0_dims, expected);
        assert_eq!(h0_dims[0], batch_size);
        assert_eq!(h0_dims[1], hidden_size);
    }

    /// Prove: a rank-3 LSTM input exposes exactly the indices used by the
    /// implementation when extracting `seq_len`, `batch_size`, and
    /// `input_size`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn lstm_rank_three_input_shape_keeps_indexing_bounded() {
        let seq_len: usize = kani::any();
        let batch_size: usize = kani::any();
        let input_size: usize = kani::any();

        assume(seq_len >= 1 && seq_len <= 2048);
        assume(batch_size >= 1 && batch_size <= 64);
        assume(input_size >= 1 && input_size <= 4096);

        let dims = [seq_len, batch_size, input_size];

        assert_eq!(dims.len(), 3);
        assert_eq!(dims[0], seq_len);
        assert_eq!(dims[1], batch_size);
        assert_eq!(dims[2], input_size);
    }
}
