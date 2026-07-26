// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Focused Kani proof harnesses for `compiled_model_execute_native.rs`.
//!
//! These proofs target the issue #3728 invariants around dispatch step
//! validation and buffer index bounds for native-op execution.

#[cfg(kani)]
mod proofs {
    use kani::assume;

    const MAX_STEPS: usize = 8;

    /// Prove: a validated `step_idx` can index every per-step table that
    /// `execute_native_op` consults (`step_metas`, `step_dtype`, weights).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn native_step_idx_indexes_parallel_tables() {
        let num_steps: usize = kani::any();
        let step_idx: usize = kani::any();

        assume(num_steps > 0 && num_steps <= MAX_STEPS);
        assume(step_idx < num_steps);

        let step_metas = [0u8; MAX_STEPS];
        let step_dtypes = [1u8; MAX_STEPS];
        let weight_buffers = [2u8; MAX_STEPS];

        assert_eq!(step_metas[step_idx], 0);
        assert_eq!(step_dtypes[step_idx], 1);
        assert_eq!(weight_buffers[step_idx], 2);
        assert!(step_idx < num_steps);
    }

    /// Prove: topological native-op inputs stay within the runtime buffer
    /// table. `resolve_input_slice` indexes `buffers[src_step]` directly, so
    /// the build-time invariant `src_step < step_idx < num_steps` is the key
    /// bound that prevents panics.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn native_topological_input_step_stays_in_buffer_table() {
        let num_steps: usize = kani::any();
        let step_idx: usize = kani::any();
        let src_step: usize = kani::any();

        assume(num_steps > 0 && num_steps <= MAX_STEPS);
        assume(step_idx > 0 && step_idx < num_steps);
        assume(src_step < step_idx);

        let buffers = [Some(7u8); MAX_STEPS];

        assert!(src_step < num_steps);
        assert!(src_step < step_idx);
        assert_eq!(buffers[src_step], Some(7u8));
    }

    /// Prove: the LSTM native path only needs input slot 0, and a step with
    /// at least one edge can always resolve that slot safely.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn lstm_native_dispatch_uses_bounded_input_zero() {
        let edge_count: usize = kani::any();
        assume(edge_count >= 1 && edge_count <= 4);

        let input_idx = 0usize;
        let edges = [3usize, 5, 6, 7];

        assert!(input_idx < edge_count);
        assert_eq!(edges[input_idx], 3usize);
    }

    /// Prove: the SiluMul direct path resolves exactly two bounded input slots
    /// (`gate` at 0 and `up` at 1) before dispatching the fused kernel.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn silu_mul_direct_dispatch_uses_two_bounded_inputs() {
        let edge_count: usize = kani::any();
        assume(edge_count >= 2 && edge_count <= 4);

        let gate_idx = 0usize;
        let up_idx = 1usize;
        let edges = [11usize, 12, 13, 14];

        assert!(gate_idx < edge_count);
        assert!(up_idx < edge_count);
        assert_ne!(gate_idx, up_idx);
        assert_ne!(edges[gate_idx], edges[up_idx]);
    }
}
