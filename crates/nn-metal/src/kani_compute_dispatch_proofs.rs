// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for [`ComputeDispatch`](super::dispatch::ComputeDispatch)
//! and [`CommandBatch`](super::dispatch::CommandBatch) batch lifecycle.
//!
//! `ComputeDispatch` is the core Metal GPU dispatch abstraction. Every GPU kernel
//! launch flows through it. These proofs verify the logical state-machine
//! properties of the dispatch lifecycle using pure-Rust models (no Metal GPU
//! hardware required).
//!
//! # Properties Proved (7 harnesses)
//!
//!  1. **Batch lifecycle**: begin_batch -> encode -> commit_and_wait is a
//!     complete cycle that transitions through all valid states.
//!  2. **Encoding counter**: Encoding count increments monotonically on each
//!     encode call, never skips or regresses.
//!  3. **No double-commit**: Committing an already-committed batch is safe
//!     (idempotent). The state machine rejects the transition.
//!  4. **Batch isolation**: Commands in batch N are independent of batch N+1.
//!     Encoding counts reset between batches.
//!  5. **Submit returns pending**: submit() produces a valid PendingBatch
//!     (state = Committed, not yet Completed).
//!  6. **Empty batch**: A batch with 0 encodings commits without error.
//!     The state machine allows Created -> Committed with no encodings.
//!  7. **Flush semantics**: flush() commits all pending work and resets state.
//!     Post-flush encoding count is 0 and no batch is active.

#[cfg(kani)]
mod proofs {
    // ========================================================================
    // State machine model for ComputeDispatch / CommandBatch lifecycle
    // ========================================================================
    //
    // The Metal dispatch lifecycle follows a strict state machine:
    //
    //   Created ──encode()──▸ Encoding ──commit()──▸ Committed ──wait()──▸ Completed
    //                 │                                   │
    //                 ▼                                   ▼
    //              (N encodings)                   (GPU executing)
    //
    // ComputeDispatch is single-encoder (one encode, then commit_and_wait).
    // CommandBatch supports multiple sequential encoders before commit.
    //
    // The model uses u8 state tags:
    //   0 = Created (command buffer allocated, no encoders yet)
    //   1 = Encoding (at least one encoder has been created/encoded)
    //   2 = Committed (commit() called, GPU work submitted)
    //   3 = Completed (wait succeeded, GPU work done)
    //   4 = Error (command buffer in error state)

    /// State tags for the command buffer lifecycle model.
    const STATE_CREATED: u8 = 0;
    const STATE_ENCODING: u8 = 1;
    const STATE_COMMITTED: u8 = 2;
    const STATE_COMPLETED: u8 = 3;
    const STATE_ERROR: u8 = 4;

    /// Maximum encodings per lazy batch before auto-flush (mirrors gpu_scope.rs).
    const MAX_LAZY_ENCODINGS: u32 = 1024;

    /// Model of the ComputeDispatch / CommandBatch state machine.
    ///
    /// Pure-Rust struct that tracks the logical state of a Metal command buffer
    /// dispatch lifecycle. All transitions mirror the real Metal API semantics.
    struct DispatchModel {
        state: u8,
        encoding_count: u32,
        ended: bool,
    }

    impl DispatchModel {
        /// Create a new dispatch in the Created state (mirrors `from_raw`).
        fn new() -> Self {
            Self {
                state: STATE_CREATED,
                encoding_count: 0,
                ended: false,
            }
        }

        /// Encode a compute dispatch (mirrors `ComputeDispatch::encode` or
        /// `BatchEncoder::encode`). Transitions Created -> Encoding on first
        /// call. Increments encoding count.
        ///
        /// Returns `true` if the encoding was accepted, `false` if the state
        /// does not permit encoding (already committed/completed/error).
        fn encode(&mut self) -> bool {
            match self.state {
                STATE_CREATED | STATE_ENCODING => {
                    self.state = STATE_ENCODING;
                    self.encoding_count += 1;
                    true
                }
                _ => false,
            }
        }

        /// Commit the command buffer (mirrors `commit_and_wait` or
        /// `commit_no_wait`). Only valid from Created or Encoding states.
        ///
        /// Returns `true` if commit succeeded, `false` if already committed
        /// or in an invalid state.
        fn commit(&mut self) -> bool {
            match self.state {
                STATE_CREATED | STATE_ENCODING => {
                    self.ended = true;
                    self.state = STATE_COMMITTED;
                    true
                }
                _ => false,
            }
        }

        /// Wait for completion (mirrors `wait_with_timeout`). Only valid
        /// from Committed state.
        ///
        /// Returns `true` if wait succeeded (Completed), `false` otherwise.
        fn wait(&mut self) -> bool {
            if self.state == STATE_COMMITTED {
                self.state = STATE_COMPLETED;
                true
            } else {
                false
            }
        }

        /// Combined commit-and-wait (mirrors `ComputeDispatch::commit_and_wait`
        /// and `CommandBatch::commit_and_wait`).
        fn commit_and_wait(&mut self) -> bool {
            if self.commit() {
                self.wait()
            } else {
                false
            }
        }
    }

    /// Model of the lazy batch system (mirrors `gpu_scope.rs` thread-local state).
    struct LazyBatchModel {
        active: bool,
        encoding_count: u32,
        pending: bool,
    }

    impl LazyBatchModel {
        fn new() -> Self {
            Self {
                active: false,
                encoding_count: 0,
                pending: false,
            }
        }

        /// Create or reuse the lazy batch (mirrors `get_or_create_batch`).
        fn ensure_batch(&mut self) {
            if !self.active {
                self.active = true;
                self.encoding_count = 0;
            }
        }

        /// Record an encoding (mirrors the encoding_count increment in
        /// `ensure_batch`).
        fn record_encoding(&mut self) {
            self.ensure_batch();
            self.encoding_count += 1;
        }

        /// Submit the batch without waiting (mirrors `submit()`).
        fn submit(&mut self) {
            if self.active {
                self.active = false;
                self.pending = true;
                self.encoding_count = 0;
            }
        }

        /// Flush: commit and wait for all pending work (mirrors `flush()`).
        fn flush(&mut self) {
            // First sync any prior pending batch.
            self.pending = false;
            // Then commit current batch.
            if self.active {
                self.active = false;
                self.encoding_count = 0;
            }
        }

        /// Sync: wait for the most recently submitted batch (mirrors `sync()`).
        fn sync(&mut self) {
            self.pending = false;
        }
    }

    // ========================================================================
    // 1. Batch lifecycle: begin_batch -> encode -> commit_and_wait
    // ========================================================================

    /// Proves that the complete dispatch lifecycle (Created -> Encoding ->
    /// Committed -> Completed) transitions through all valid states in order.
    ///
    /// This models `ComputeDispatch::from_raw()` -> `encode()` ->
    /// `commit_and_wait()` — the pattern used by every single-kernel GPU
    /// dispatch in nn-metal.
    #[kani::proof]
    #[kani::unwind(4)]
    fn proof_batch_lifecycle_complete() {
        let mut dispatch = DispatchModel::new();

        // Initial state: Created
        assert_eq!(dispatch.state, STATE_CREATED, "must start in Created state");
        assert_eq!(dispatch.encoding_count, 0, "must start with 0 encodings");
        assert!(!dispatch.ended, "must start with ended=false");

        // Encode at least one dispatch.
        let n_encodes: u8 = kani::any();
        kani::assume(n_encodes >= 1 && n_encodes <= 4);
        let mut i: u8 = 0;
        while i < n_encodes {
            let ok = dispatch.encode();
            assert!(ok, "encode must succeed in Created/Encoding state");
            i += 1;
        }

        // After encoding: state is Encoding.
        assert_eq!(
            dispatch.state, STATE_ENCODING,
            "must be in Encoding state after encode"
        );
        assert_eq!(
            dispatch.encoding_count, n_encodes as u32,
            "encoding count must match number of encode calls"
        );

        // commit_and_wait: transitions through Committed -> Completed.
        let ok = dispatch.commit_and_wait();
        assert!(ok, "commit_and_wait must succeed from Encoding state");
        assert_eq!(
            dispatch.state, STATE_COMPLETED,
            "must reach Completed state"
        );
        assert!(dispatch.ended, "ended flag must be set after commit");
    }

    // ========================================================================
    // 2. Encoding counter monotonically increments
    // ========================================================================

    /// Proves that the encoding counter increments by exactly 1 on each encode
    /// call and never decreases.
    ///
    /// This models the `ENCODING_COUNT` thread-local in `gpu_scope.rs` which
    /// tracks how many dispatches have been recorded in the current lazy batch.
    /// Monotonicity ensures the auto-flush threshold is reached correctly.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_encoding_counter_monotonic_increment() {
        let mut dispatch = DispatchModel::new();

        let n: u8 = kani::any();
        kani::assume(n >= 1 && n <= 6);

        let mut prev_count: u32 = 0;
        let mut i: u8 = 0;
        while i < n {
            let before = dispatch.encoding_count;
            assert_eq!(before, prev_count, "count must equal tracked value");

            let ok = dispatch.encode();
            assert!(ok, "encode must succeed");

            let after = dispatch.encoding_count;
            assert_eq!(
                after,
                before + 1,
                "encoding count must increment by exactly 1"
            );
            assert!(
                after > before,
                "encoding count must strictly increase"
            );

            prev_count = after;
            i += 1;
        }

        assert_eq!(
            dispatch.encoding_count, n as u32,
            "final encoding count must equal number of encode calls"
        );
    }

    // ========================================================================
    // 3. No double-commit: committing an already-committed batch is safe
    // ========================================================================

    /// Proves that attempting to commit a batch that has already been committed
    /// is idempotent — the second commit is rejected (returns false) and the
    /// state remains unchanged.
    ///
    /// In the real Metal API, `CommandBatch::new_encoder()` guards against
    /// creating encoders on committed/completed/error command buffers by checking
    /// `command_buffer.status()`. This proof verifies the logical model.
    #[kani::proof]
    #[kani::unwind(4)]
    fn proof_no_double_commit() {
        let mut dispatch = DispatchModel::new();

        // Optionally encode some work.
        let do_encode: bool = kani::any();
        if do_encode {
            dispatch.encode();
        }

        // First commit: must succeed.
        let first_ok = dispatch.commit();
        assert!(first_ok, "first commit must succeed");
        assert_eq!(
            dispatch.state, STATE_COMMITTED,
            "state must be Committed after first commit"
        );

        // Save state before second commit attempt.
        let state_before = dispatch.state;
        let count_before = dispatch.encoding_count;

        // Second commit: must be rejected.
        let second_ok = dispatch.commit();
        assert!(
            !second_ok,
            "second commit must fail (idempotent rejection)"
        );

        // State must be unchanged after rejected commit.
        assert_eq!(
            dispatch.state, state_before,
            "state must not change on rejected commit"
        );
        assert_eq!(
            dispatch.encoding_count, count_before,
            "encoding count must not change on rejected commit"
        );

        // Encoding after commit must also be rejected.
        let encode_ok = dispatch.encode();
        assert!(
            !encode_ok,
            "encode must fail after commit"
        );
    }

    // ========================================================================
    // 4. Batch isolation: batch N is independent of batch N+1
    // ========================================================================

    /// Proves that completing one batch and starting a new one produces
    /// independent state. The second batch starts fresh with 0 encodings
    /// and in Created state, regardless of what the first batch did.
    ///
    /// This models the real behavior where each `CommandBatch` is a separate
    /// Metal command buffer with its own lifecycle. In `gpu_scope.rs`, after
    /// `flush()` the `ENCODING_COUNT` resets to 0 and a fresh batch is created.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_batch_isolation() {
        // --- Batch N ---
        let mut batch_n = DispatchModel::new();
        let n_encodes: u8 = kani::any();
        kani::assume(n_encodes >= 1 && n_encodes <= 4);

        let mut i: u8 = 0;
        while i < n_encodes {
            batch_n.encode();
            i += 1;
        }
        let batch_n_ok = batch_n.commit_and_wait();
        assert!(batch_n_ok, "batch N must complete successfully");
        let batch_n_final_count = batch_n.encoding_count;

        // --- Batch N+1 ---
        let mut batch_n1 = DispatchModel::new();

        // Batch N+1 must start completely fresh.
        assert_eq!(
            batch_n1.state, STATE_CREATED,
            "batch N+1 must start in Created state"
        );
        assert_eq!(
            batch_n1.encoding_count, 0,
            "batch N+1 must start with 0 encodings"
        );
        assert!(!batch_n1.ended, "batch N+1 must start with ended=false");

        // Batch N+1 encoding count is independent of batch N.
        let m_encodes: u8 = kani::any();
        kani::assume(m_encodes >= 1 && m_encodes <= 4);

        let mut j: u8 = 0;
        while j < m_encodes {
            batch_n1.encode();
            j += 1;
        }

        assert_eq!(
            batch_n1.encoding_count, m_encodes as u32,
            "batch N+1 count must be independent of batch N"
        );
        // The two counts are independent (they may or may not be equal by coincidence,
        // but there is no causal relationship).
        assert_eq!(
            batch_n_final_count, n_encodes as u32,
            "batch N count must be preserved independently"
        );

        let batch_n1_ok = batch_n1.commit_and_wait();
        assert!(batch_n1_ok, "batch N+1 must complete successfully");
    }

    // ========================================================================
    // 5. Submit returns pending: submit() produces a valid PendingBatch
    // ========================================================================

    /// Proves that `submit()` transitions a batch from Created/Encoding to
    /// Committed (not Completed), producing a "pending" state. The pending
    /// batch is not yet complete and requires an explicit `wait()` / `sync()`.
    ///
    /// Models `CommandBatch::commit_no_wait()` which returns a `PendingBatch`
    /// with `is_completed() == false` initially.
    #[kani::proof]
    #[kani::unwind(4)]
    fn proof_submit_returns_pending() {
        let mut dispatch = DispatchModel::new();

        // Optionally encode some work.
        let n_encodes: u8 = kani::any();
        kani::assume(n_encodes <= 4);
        let mut i: u8 = 0;
        while i < n_encodes {
            dispatch.encode();
            i += 1;
        }

        // Submit (commit without waiting).
        let commit_ok = dispatch.commit();
        assert!(commit_ok, "commit must succeed from Created/Encoding");

        // After commit, state is Committed — NOT Completed.
        assert_eq!(
            dispatch.state, STATE_COMMITTED,
            "after submit, state must be Committed (pending)"
        );
        assert_ne!(
            dispatch.state, STATE_COMPLETED,
            "after submit, state must NOT be Completed"
        );

        // The pending batch can then be waited on.
        let wait_ok = dispatch.wait();
        assert!(wait_ok, "wait on pending batch must succeed");
        assert_eq!(
            dispatch.state, STATE_COMPLETED,
            "after wait, state must be Completed"
        );
    }

    // ========================================================================
    // 6. Empty batch: 0 encodings commits without error
    // ========================================================================

    /// Proves that a batch with zero encodings can still be committed and
    /// completed without error. This matches Metal's behavior where an empty
    /// command buffer is valid — it's a no-op on the GPU.
    ///
    /// In `gpu_scope.rs`, `flush()` may be called when the encoding count is
    /// 0 (e.g., immediately after a prior flush). The empty batch commits
    /// as a no-op.
    #[kani::proof]
    #[kani::unwind(4)]
    fn proof_empty_batch_commits_without_error() {
        let mut dispatch = DispatchModel::new();

        // No encode calls — 0 encodings.
        assert_eq!(dispatch.encoding_count, 0, "must have 0 encodings");
        assert_eq!(dispatch.state, STATE_CREATED, "must be in Created state");

        // commit_and_wait with 0 encodings.
        let ok = dispatch.commit_and_wait();
        assert!(ok, "empty batch commit_and_wait must succeed");
        assert_eq!(
            dispatch.state, STATE_COMPLETED,
            "empty batch must reach Completed state"
        );
        assert_eq!(
            dispatch.encoding_count, 0,
            "encoding count must remain 0"
        );
        assert!(dispatch.ended, "ended flag must be set");
    }

    // ========================================================================
    // 7. Flush semantics: flush() commits all pending work and resets state
    // ========================================================================

    /// Proves that `flush()` commits all pending work (current batch + prior
    /// submitted batch) and resets the lazy batch state. After flush:
    /// - No active batch
    /// - No pending batch
    /// - Encoding count is 0
    ///
    /// Models the full `flush()` path from `gpu_scope.rs` which calls `sync()`
    /// then commits the current batch.
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_flush_commits_all_pending_work() {
        let mut model = LazyBatchModel::new();

        // Start with no state.
        assert!(!model.active, "must start with no active batch");
        assert!(!model.pending, "must start with no pending batch");
        assert_eq!(model.encoding_count, 0, "must start with 0 encodings");

        // Record some encodings (creates a batch lazily).
        let n_encodes: u8 = kani::any();
        kani::assume(n_encodes >= 1 && n_encodes <= 6);
        let mut i: u8 = 0;
        while i < n_encodes {
            model.record_encoding();
            i += 1;
        }

        assert!(model.active, "batch must be active after encoding");
        assert_eq!(
            model.encoding_count, n_encodes as u32,
            "encoding count must match"
        );

        // Optionally submit first (creates a pending batch).
        let do_submit: bool = kani::any();
        if do_submit {
            model.submit();
            assert!(!model.active, "batch must not be active after submit");
            assert!(model.pending, "must have pending batch after submit");
            assert_eq!(
                model.encoding_count, 0,
                "encoding count must reset after submit"
            );

            // Optionally start a second batch before flushing.
            let m_encodes: u8 = kani::any();
            kani::assume(m_encodes <= 4);
            let mut j: u8 = 0;
            while j < m_encodes {
                model.record_encoding();
                j += 1;
            }
        }

        // Flush: commits everything.
        model.flush();

        // Post-flush invariants.
        assert!(
            !model.active,
            "no active batch after flush"
        );
        assert!(
            !model.pending,
            "no pending batch after flush"
        );
        assert_eq!(
            model.encoding_count, 0,
            "encoding count must be 0 after flush"
        );
    }
}
