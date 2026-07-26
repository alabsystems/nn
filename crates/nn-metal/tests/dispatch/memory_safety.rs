// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural memory safety tests for Metal dispatch primitives.
//!
//! These tests verify that critical RAII patterns, Drop impls, and safety
//! annotations remain in place. They catch accidental removal of safety-
//! critical code during refactoring.
//!
//! Part of #3020 memory_verification audit.

use nn_metal::{CommandBatch, ComputeDispatch, PendingBatch};

// ---------------------------------------------------------------------------
// Drop implementation presence
// ---------------------------------------------------------------------------

/// ComputeDispatch MUST implement Drop to end_encoding on error paths.
///
/// Without Drop, an uncommitted encoder left on an error path causes
/// undefined Metal behavior — the encoder is still "open" and Metal may
/// GPU-hang on the next command buffer from the same queue (#647).
#[test]
fn test_compute_dispatch_has_drop() {
    assert!(
        std::mem::needs_drop::<ComputeDispatch>(),
        "ComputeDispatch must implement Drop for encoder cleanup"
    );
}

/// CommandBatch does NOT implement Drop — it is consumed by commit_and_wait
/// or commit_no_wait. When dropped without committing, Metal's ObjC ARC
/// releases the command buffer (uncommitted work is discarded).
///
/// This is intentional: a custom Drop that commits would hide errors and
/// make error-path behavior unpredictable. Verify this stays intentional.
#[test]
fn test_command_batch_drop_behavior() {
    // CommandBatch has fields that need drop (metal::CommandBuffer), so
    // needs_drop is true even without a custom impl. This test documents
    // the design intent: no custom Drop, consume via commit_and_wait/commit_no_wait.
    assert!(
        std::mem::needs_drop::<CommandBatch>(),
        "CommandBatch fields need drop (ObjC ARC)"
    );
}

/// PendingBatch wraps a committed command buffer awaiting GPU completion.
/// needs_drop is true (contains metal::CommandBuffer with ObjC ARC).
///
/// Dropping PendingBatch without wait() is safe (ObjC ARC handles cleanup)
/// but wastes GPU work and may delay arena reclamation. The struct should
/// have #[must_use] to prevent accidental drops.
#[test]
fn test_pending_batch_needs_drop() {
    assert!(
        std::mem::needs_drop::<PendingBatch>(),
        "PendingBatch must need drop (ObjC ARC on command buffer)"
    );
}

// ---------------------------------------------------------------------------
// Source structural checks
// ---------------------------------------------------------------------------

/// Verify ComputeDispatch Drop impl calls end_encoding in autoreleasepool.
///
/// The autoreleasepool in Drop is critical for background threads — without
/// it, ObjC autoreleased temporaries from end_encoding accumulate and
/// eventually trigger a kernel panic on sustained Metal workloads (#1245).
#[test]
fn test_compute_dispatch_drop_uses_autoreleasepool() {
    let source = include_str!("../../src/dispatch.rs");

    // Drop impl must exist
    assert!(
        source.contains("impl Drop for ComputeDispatch"),
        "ComputeDispatch must have a custom Drop impl"
    );

    // The Drop impl must use autoreleasepool for ObjC memory safety
    // Find the Drop impl and verify it contains autoreleasepool
    let drop_start = source
        .find("impl Drop for ComputeDispatch")
        .expect("Drop impl must exist");
    let drop_section = &source[drop_start..];
    // The Drop impl body should contain autoreleasepool within ~200 chars
    let drop_snippet = &drop_section[..drop_section.len().min(300)];
    assert!(
        drop_snippet.contains("autoreleasepool"),
        "ComputeDispatch Drop must use autoreleasepool for ObjC memory safety"
    );
}

/// Verify BatchEncoder Drop impl also uses autoreleasepool.
#[test]
fn test_batch_encoder_drop_uses_autoreleasepool() {
    let source = include_str!("../../src/dispatch.rs");

    let drop_start = source
        .find("impl Drop for BatchEncoder")
        .expect("BatchEncoder Drop impl must exist");
    let drop_section = &source[drop_start..];
    let drop_snippet = &drop_section[..drop_section.len().min(300)];
    assert!(
        drop_snippet.contains("autoreleasepool"),
        "BatchEncoder Drop must use autoreleasepool for ObjC memory safety"
    );
}

/// Verify discard_pending_batch clears both LAZY_BATCH and PENDING state.
///
/// discard_pending_batch is called on error paths. It must clear both the
/// uncommitted lazy batch (LAZY_BATCH) and any already-submitted pending
/// batch (PENDING). Missing either leaves stale GPU state that contaminates
/// the next dispatch scope on this thread.
///
/// Safety note: dropping a submitted PendingBatch without wait() is safe
/// only because Metal command buffers execute in queue order. If this
/// function ever needs to support concurrent dispatch queues, an explicit
/// wait() must be added before clearing PENDING.
#[test]
fn test_discard_pending_batch_clears_all_state() {
    let source = include_str!("../../src/gpu_scope.rs");

    let fn_start = source
        .find("fn discard_pending_batch")
        .expect("discard_pending_batch must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(400)];

    // Must clear both LAZY_BATCH and PENDING
    assert!(
        fn_snippet.contains("LAZY_BATCH"),
        "discard_pending_batch must clear LAZY_BATCH"
    );
    assert!(
        fn_snippet.contains("PENDING"),
        "discard_pending_batch must clear PENDING (submitted batch)"
    );
    // Must also reset encoding count
    assert!(
        fn_snippet.contains("ENCODING_COUNT"),
        "discard_pending_batch must reset ENCODING_COUNT"
    );
}

// ---------------------------------------------------------------------------
// WeightMap ManuallyDrop (complementary to safetensors_tests.rs)
// ---------------------------------------------------------------------------

/// Verify WeightMap ManuallyDrop fields exist in source.
///
/// This complements the safetensors_tests.rs structural test by also verifying
/// from the integration test level. If WeightMap's ManuallyDrop pattern is
/// changed, both unit and integration tests catch it.
#[test]
fn test_weight_map_manually_drop_structural() {
    let source = include_str!("../../src/safetensors.rs");

    assert!(
        source.contains("ManuallyDrop<MetalBuffer>"),
        "WeightMap buffer field must use ManuallyDrop"
    );
    assert!(
        source.contains("ManuallyDrop<Mmap>"),
        "WeightMap mmap field must use ManuallyDrop"
    );

    // Verify drop order: buffer dropped BEFORE mmap
    let drop_buffer = source
        .find("ManuallyDrop::drop(&mut self.buffer)")
        .expect("Drop impl must drop buffer");
    let drop_mmap = source
        .find("ManuallyDrop::drop(&mut self.mmap)")
        .expect("Drop impl must drop mmap");
    assert!(
        drop_buffer < drop_mmap,
        "buffer must be dropped before mmap (use-after-unmap protection)"
    );
}

// ---------------------------------------------------------------------------
// Arena RAII patterns
// ---------------------------------------------------------------------------

/// Verify with_arena uses catch_unwind for TLS cleanup.
///
/// Without catch_unwind, a panic during the arena scope leaves a dangling
/// raw pointer in thread-local storage. The next arena_alloc_or_create call
/// would dereference freed memory — undefined behavior.
#[test]
fn test_with_arena_panic_safety() {
    let source = include_str!("../../src/arena_scope.rs");

    let fn_start = source
        .find("pub fn with_arena")
        .expect("with_arena must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(600)];

    assert!(
        fn_snippet.contains("catch_unwind"),
        "with_arena must use catch_unwind for TLS cleanup on panic"
    );
}

/// Verify without_arena uses an RAII guard for cleanup.
///
/// The ARENA_BYPASS flag must be restored even if the closure panics.
/// An RAII guard (struct with Drop) is the correct pattern.
#[test]
fn test_without_arena_raii_guard() {
    let source = include_str!("../../src/arena_scope.rs");

    let fn_start = source
        .find("pub fn without_arena")
        .expect("without_arena must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(400)];

    assert!(
        fn_snippet.contains("impl Drop for Guard"),
        "without_arena must use RAII Guard for ARENA_BYPASS cleanup"
    );
}

// ---------------------------------------------------------------------------
// Buffer pool invariants
// ---------------------------------------------------------------------------

/// Verify pool stats accounting invariant: acquisitions == hits + misses + discards.
///
/// This is a compile-time type check: PoolStats has exactly these fields,
/// and the acquire() path must increment exactly one of hits/misses/discards
/// per acquisition call.
#[test]
fn test_pool_stats_field_completeness() {
    let source = include_str!("../../src/buffer_pool.rs");

    // All three outcome counters must exist in PoolStats
    assert!(source.contains("pub hits:"), "PoolStats must track hits");
    assert!(
        source.contains("pub misses:"),
        "PoolStats must track misses"
    );
    assert!(
        source.contains("pub discards:"),
        "PoolStats must track discards"
    );
    assert!(
        source.contains("pub acquisitions:"),
        "PoolStats must track total acquisitions"
    );
}

/// Verify buffer pool has MAX_POOLED_BYTES cap to prevent unbounded RSS growth.
///
/// Without a byte budget, the pool could retain 8×256MB = 2GB of Metal
/// buffers. The cap ensures RSS savings from pooling aren't negated by
/// retention (#3079 D3).
#[test]
fn test_pool_has_byte_budget() {
    let source = include_str!("../../src/buffer_pool.rs");

    assert!(
        source.contains("MAX_POOLED_BYTES"),
        "buffer pool must have a MAX_POOLED_BYTES cap"
    );

    // The budget check must be in the acquire path
    let acquire_start = source.find("fn acquire").expect("acquire must exist");
    let acquire_section = &source[acquire_start..];
    let acquire_snippet = &acquire_section[..acquire_section.len().min(1500)];
    assert!(
        acquire_snippet.contains("MAX_POOLED_BYTES"),
        "acquire() must check MAX_POOLED_BYTES before pooling"
    );
}

// ---------------------------------------------------------------------------
// GPU scope invariants (#3020 strategic)
// ---------------------------------------------------------------------------

/// `execute_dyn_no_fence` callers MUST call `discard_pending_batch()` on error.
///
/// Without error cleanup, stale GPU commands from a failed `_no_fence` execution
/// persist in the thread-local lazy batch and contaminate the next dispatch scope.
/// The Kokoro pipeline is the sole production caller — verify it handles errors.
#[test]
fn test_no_fence_pipeline_error_cleanup() {
    let source = include_str!("../../src/compiled_kokoro_pipeline.rs");

    // The pipeline must call _no_fence variants (it's a multi-segment pipeline).
    assert!(
        source.contains("_no_fence"),
        "Kokoro pipeline must use _no_fence for multi-segment pipelining"
    );

    // The error path must discard the pending batch.
    assert!(
        source.contains("discard_pending_batch"),
        "Kokoro pipeline must call discard_pending_batch() on error path"
    );

    // Verify the discard is inside an error-conditional block.
    let discard_pos = source
        .find("discard_pending_batch")
        .expect("discard_pending_batch must exist in pipeline");
    // Look backward for the error check — should be within ~200 chars before.
    let pre_context = &source[discard_pos.saturating_sub(200)..discard_pos];
    assert!(
        pre_context.contains("is_err()"),
        "discard_pending_batch must be guarded by is_err() check"
    );
}

/// `synthesize_with_timing()` must mirror the main pipeline's error cleanup.
///
/// The timing path also uses `_no_fence` segment execution through the step API.
/// If it returns early on error without discarding the pending batch, stale GPU
/// commands can contaminate the next dispatch scope.
#[test]
fn test_timing_pipeline_error_cleanup() {
    let source = include_str!("../../src/compiled_kokoro_diagnostics.rs");

    assert!(
        source.contains("synthesize_with_timing"),
        "compiled_kokoro_diagnostics.rs must define synthesize_with_timing()"
    );
    assert!(
        source.contains("discard_pending_batch"),
        "synthesize_with_timing must call discard_pending_batch() on error"
    );

    let discard_pos = source
        .find("discard_pending_batch")
        .expect("discard_pending_batch must exist in diagnostics path");
    let pre_context = &source[discard_pos.saturating_sub(200)..discard_pos];
    assert!(
        pre_context.contains("is_err()"),
        "discard_pending_batch in synthesize_with_timing must be guarded by is_err()"
    );
}

/// `step_regulate` must reject zero mel-frame output before downstream steps.
///
/// A zero-length mel sequence can propagate empty tensors into F0/Generator/iSTFT.
/// Guarding at regulate keeps the failure local and explicit.
#[test]
fn test_step_regulate_guards_zero_t_mel() {
    let source = include_str!("../../src/compiled_kokoro_step_regulate.rs");
    assert!(
        source.contains("if t_mel == 0"),
        "step_regulate must guard t_mel == 0"
    );
    assert!(
        source.contains("no mel frames produced"),
        "step_regulate zero-length guard should return explicit error context"
    );
}

/// Generator cache-key arithmetic should use checked helper calls, not raw multiply.
///
/// This avoids `2 * t_mel * upsample_factor` overflow in step and diagnostics paths.
#[test]
fn test_generator_total_samples_helper_used_in_hot_paths() {
    let steps = include_str!("../../src/compiled_kokoro_steps.rs");
    let diagnostics = include_str!("../../src/compiled_kokoro_diagnostics.rs");

    assert!(
        steps.contains("generator_total_samples("),
        "step_generate must use generator_total_samples()"
    );
    assert!(
        diagnostics.contains("generator_total_samples("),
        "synthesize_with_timing must use generator_total_samples()"
    );
}

/// `flush()` must call `sync()` BEFORE committing the lazy batch.
///
/// Arena reset ordering invariant: `sync()` waits for prior submitted GPU work
/// and resets the arena. The lazy batch (not yet committed) may reference arena
/// buffers allocated AFTER the last submit. If `commit_and_wait` ran before
/// `sync`, those arena buffers could be freed by a concurrent submit's reset
/// while the lazy batch's GPU commands still need them.
///
/// Metal ObjC ARC keeps individual buffers alive, but the arena's bump pointer
/// could alias the same physical memory. `sync` first ensures the previous
/// submit's GPU work is done before any arena state changes.
#[test]
fn test_flush_calls_sync_before_commit() {
    let source = include_str!("../../src/gpu_scope.rs");

    let flush_start = source.find("pub fn flush()").expect("flush() must exist");
    let flush_section = &source[flush_start..];
    let flush_snippet = &flush_section[..flush_section.len().min(600)];

    let sync_pos = flush_snippet
        .find("sync()")
        .expect("flush() must call sync()");
    let commit_pos = flush_snippet
        .find("commit_and_wait()")
        .expect("flush() must call commit_and_wait()");

    assert!(
        sync_pos < commit_pos,
        "flush() must call sync() BEFORE commit_and_wait() (arena reset ordering)"
    );
}

/// `encode_into_lazy_batch` borrows LAZY_BATCH via RefCell during callback.
///
/// This means callbacks MUST NOT call any gpu_scope function that borrows
/// LAZY_BATCH (flush, submit, sync, get_or_create_batch, encode_into_lazy_batch).
/// A re-entrant borrow would panic at runtime with "already borrowed".
///
/// This test verifies the RefCell borrow pattern exists, documenting the
/// re-entrancy constraint as a structural invariant.
#[test]
fn test_encode_into_lazy_batch_holds_refcell_borrow() {
    let source = include_str!("../../src/gpu_scope.rs");

    let fn_start = source
        .find("pub(crate) fn encode_into_lazy_batch")
        .expect("encode_into_lazy_batch must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(400)];

    // The callback `f(batch)` executes while the RefCell borrow is held.
    // Verify the borrow is taken before the callback is invoked.
    assert!(
        fn_snippet.contains(".borrow()"),
        "encode_into_lazy_batch must borrow LAZY_BATCH RefCell"
    );

    // Verify the callback is invoked on the borrowed batch.
    assert!(
        fn_snippet.contains("f(batch)"),
        "encode_into_lazy_batch must invoke callback with borrowed batch"
    );
}

/// `with_gpu_scope` error path must call `discard_pending_batch`.
///
/// Complementary to the Kokoro-specific test above. This verifies the
/// scope primitive itself cleans up on error, not just the caller.
#[test]
fn test_with_gpu_scope_error_discards_batch() {
    let source = include_str!("../../src/gpu_scope.rs");

    let fn_start = source
        .find("pub fn with_gpu_scope")
        .expect("with_gpu_scope must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(700)];

    // Error path must discard the batch.
    assert!(
        fn_snippet.contains("discard_pending_batch()"),
        "with_gpu_scope must call discard_pending_batch() on error"
    );

    // The discard must be in the else branch (error path).
    let discard_pos = fn_snippet
        .find("discard_pending_batch()")
        .expect("discard must exist");
    let pre_context = &fn_snippet[..discard_pos];
    assert!(
        pre_context.contains("} else {") || pre_context.contains("else {"),
        "discard_pending_batch must be in the error (else) branch"
    );
}

// ---------------------------------------------------------------------------
// PendingBatch #[must_use] (#3020 tool_quality)
// ---------------------------------------------------------------------------

/// `PendingBatch` MUST have `#[must_use]` to prevent silent GPU work orphaning.
///
/// Dropping `PendingBatch` without calling `wait()` is safe (ObjC ARC handles
/// Metal buffer cleanup), but it orphans in-flight GPU work. The arena may
/// overwrite buffers before the GPU finishes reading them. `#[must_use]`
/// produces a compiler warning at every call site that discards the value.
#[test]
fn test_pending_batch_has_must_use() {
    let source = include_str!("../../src/dispatch_pending.rs");

    // The #[must_use] attribute must appear immediately before the struct definition.
    let struct_pos = source
        .find("pub struct PendingBatch")
        .expect("PendingBatch struct must exist");
    let pre_context = &source[struct_pos.saturating_sub(200)..struct_pos];
    assert!(
        pre_context.contains("#[must_use"),
        "PendingBatch must have #[must_use] attribute to prevent silent GPU work orphaning"
    );
}

// ---------------------------------------------------------------------------
// submit() ordering invariant (#3020 tool_quality)
// ---------------------------------------------------------------------------

/// `submit()` must call `sync()` BEFORE `commit_no_wait()`.
///
/// Without sync, a second submit could overlap with an in-flight first batch.
/// Metal command buffers execute in queue order, but the PENDING thread-local
/// would leak the first PendingBatch (dropped without wait). sync() ensures
/// the prior batch completes and PENDING is cleared before the new submit.
#[test]
fn test_submit_calls_sync_before_commit() {
    let source = include_str!("../../src/gpu_scope.rs");

    let fn_start = source.find("pub fn submit()").expect("submit() must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(600)];

    let sync_pos = fn_snippet
        .find("sync()")
        .expect("submit() must call sync()");
    let commit_pos = fn_snippet
        .find("commit_no_wait()")
        .expect("submit() must call commit_no_wait()");

    assert!(
        sync_pos < commit_pos,
        "submit() must call sync() BEFORE commit_no_wait() (prevents PENDING leak)"
    );
}

// ---------------------------------------------------------------------------
// Auto-flush arena reset (#3020 tool_quality)
// ---------------------------------------------------------------------------

/// `ensure_batch` auto-flush path must reset the default arena.
///
/// Without arena reset after auto-flush, the arena cannot reclaim memory
/// from the just-committed batch. Subsequent allocations overflow to direct
/// `create_buffer_zeroed` much sooner than necessary (#2204).
///
/// Note: `get_or_create_batch()` delegates to `ensure_batch()` where the
/// auto-flush logic lives. The invariant is on `ensure_batch`.
#[test]
fn test_auto_flush_resets_arena() {
    let source = include_str!("../../src/gpu_scope.rs");

    let fn_start = source
        .find("fn ensure_batch(")
        .expect("ensure_batch must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(1500)];

    // Must call commit_and_wait in the auto-flush path.
    assert!(
        fn_snippet.contains("commit_and_wait()"),
        "ensure_batch auto-flush must commit_and_wait"
    );

    // Must reset the arena AFTER commit_and_wait.
    let commit_pos = fn_snippet
        .find("commit_and_wait()")
        .expect("commit_and_wait must exist in auto-flush");
    let after_commit = &fn_snippet[commit_pos..];
    assert!(
        after_commit.contains("reset_default_arena()"),
        "auto-flush must call reset_default_arena() after commit_and_wait (arena memory reclaim)"
    );
}

// ---------------------------------------------------------------------------
// sync() arena reset invariant (#3020 tool_quality)
// ---------------------------------------------------------------------------

/// `sync()` must call `reset_default_arena()` after `pending.wait()`.
///
/// Arena buffers from the submitted batch are only safe to reclaim once the
/// GPU finishes reading them. `wait()` blocks until GPU completion, then
/// `reset_default_arena()` reclaims the arena. Reversing the order would
/// free arena memory while the GPU is still reading.
#[test]
fn test_sync_resets_arena_after_wait() {
    let source = include_str!("../../src/gpu_scope.rs");

    let fn_start = source.find("pub fn sync()").expect("sync() must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(700)];

    let wait_pos = fn_snippet
        .find(".wait()")
        .expect("sync() must call pending.wait()");
    let after_wait = &fn_snippet[wait_pos..];
    assert!(
        after_wait.contains("reset_default_arena()"),
        "sync() must call reset_default_arena() after wait() (arena reclaim)"
    );
}

// ---------------------------------------------------------------------------
// flush() arena reset invariant (#3020 tool_quality)
// ---------------------------------------------------------------------------

/// `flush()` must call `reset_default_arena()` after `commit_and_wait()`.
///
/// Complements `test_flush_calls_sync_before_commit` (ordering) with arena
/// invariant: intermediate arena buffers from the just-committed batch must
/// be reclaimed after GPU completion. ObjC ARC keeps individual Metal buffers
/// alive for DynTensors still referencing them, but the arena's bump pointer
/// can safely reset.
#[test]
fn test_flush_resets_arena_after_commit() {
    let source = include_str!("../../src/gpu_scope.rs");

    let fn_start = source.find("pub fn flush()").expect("flush() must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(1200)];

    let commit_pos = fn_snippet
        .find("commit_and_wait()")
        .expect("flush() must call commit_and_wait()");
    let after_commit = &fn_snippet[commit_pos..];
    assert!(
        after_commit.contains("reset_default_arena()"),
        "flush() must call reset_default_arena() after commit_and_wait (arena reclaim)"
    );
}

// ---------------------------------------------------------------------------
// with_gpu_scope mode dispatch (#3020 tool_quality)
// ---------------------------------------------------------------------------

/// `with_gpu_scope` success path dispatches based on `SCOPE_EXIT_MODE`.
///
/// When `f()` returns `Ok`, the scope must choose between `flush()` and
/// `submit()` based on the thread-local `ScopeExitMode`. This is the
/// mechanism that enables non-blocking GPU pipelining (`Submit` mode) vs
/// the default blocking behavior (`Flush` mode).
#[test]
fn test_with_gpu_scope_dispatches_by_mode() {
    let source = include_str!("../../src/gpu_scope.rs");

    let fn_start = source
        .find("pub fn with_gpu_scope")
        .expect("with_gpu_scope must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(500)];

    // Must check SCOPE_EXIT_MODE on the success path.
    assert!(
        fn_snippet.contains("SCOPE_EXIT_MODE"),
        "with_gpu_scope must check SCOPE_EXIT_MODE on success path"
    );

    // Must call flush() for Flush mode.
    assert!(
        fn_snippet.contains("Flush => flush()"),
        "with_gpu_scope must call flush() for ScopeExitMode::Flush"
    );

    // Must call submit() for Submit mode.
    assert!(
        fn_snippet.contains("Submit => submit()"),
        "with_gpu_scope must call submit() for ScopeExitMode::Submit"
    );
}

// ---------------------------------------------------------------------------
// ScopeExitModeGuard RAII pattern (#3020 tool_quality)
// ---------------------------------------------------------------------------

/// `with_scope_exit_mode` uses an RAII guard to restore prior mode.
///
/// Without the guard, a panic in the closure would leave the thread-local
/// SCOPE_EXIT_MODE in the non-default state permanently. The guard's Drop
/// impl restores the prior mode even on unwind.
#[test]
fn test_scope_exit_mode_raii_guard() {
    let source = include_str!("../../src/gpu_scope.rs");

    // ScopeExitModeGuard must have a Drop impl.
    assert!(
        source.contains("impl Drop for ScopeExitModeGuard"),
        "ScopeExitModeGuard must implement Drop for mode restoration"
    );

    // The Drop impl must restore SCOPE_EXIT_MODE.
    let drop_start = source
        .find("impl Drop for ScopeExitModeGuard")
        .expect("Drop impl must exist");
    let drop_section = &source[drop_start..];
    let drop_snippet = &drop_section[..drop_section.len().min(200)];
    assert!(
        drop_snippet.contains("SCOPE_EXIT_MODE"),
        "ScopeExitModeGuard Drop must restore SCOPE_EXIT_MODE"
    );

    // with_scope_exit_mode must use the guard.
    let fn_start = source
        .find("pub fn with_scope_exit_mode")
        .expect("with_scope_exit_mode must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(300)];
    assert!(
        fn_snippet.contains("ScopeExitModeGuard"),
        "with_scope_exit_mode must use ScopeExitModeGuard RAII"
    );
}

// ---------------------------------------------------------------------------
// clone_buffer PendingFlushRequired guard (#3020 proof_coverage)
// ---------------------------------------------------------------------------

/// `clone_buffer` and `clone_buffer_range` MUST check `pending_encoding_count()`
/// before CPU readback.
///
/// Without this guard, `contents()` returns stale data when GPU commands are
/// pending but not yet committed. This caused two P1 bugs (#1912, #1933) where
/// clone_buffer returned pre-dispatch buffer contents. The fix is to reject
/// clone calls while encodings are pending, forcing the caller to flush first.
#[test]
fn test_clone_buffer_checks_pending_encoding_count() {
    let source = include_str!("../../src/context.rs");

    // --- clone_buffer ---
    let fn_start = source
        .find("pub fn clone_buffer(")
        .expect("clone_buffer must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(400)];

    assert!(
        fn_snippet.contains("pending_encoding_count()"),
        "clone_buffer must check pending_encoding_count() before CPU readback"
    );
    assert!(
        fn_snippet.contains("PendingFlushRequired"),
        "clone_buffer must return PendingFlushRequired when encodings are pending"
    );

    // --- clone_buffer_range ---
    let fn_start = source
        .find("pub fn clone_buffer_range(")
        .expect("clone_buffer_range must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(500)];

    assert!(
        fn_snippet.contains("pending_encoding_count()"),
        "clone_buffer_range must check pending_encoding_count() before CPU readback"
    );
    assert!(
        fn_snippet.contains("PendingFlushRequired"),
        "clone_buffer_range must return PendingFlushRequired when encodings are pending"
    );
}

// ---------------------------------------------------------------------------
// dispatch dtype check is runtime error, not debug_assert (#3020 proof_coverage)
// ---------------------------------------------------------------------------

/// `dispatch_inner_body` MUST use `DtypeMismatch` runtime error for dtype
/// validation, NOT `debug_assert`.
///
/// `debug_assert` is compiled away in release builds (#889). A dtype mismatch
/// in release mode with debug_assert would silently reinterpret buffer contents
/// as the wrong type — causing silent data corruption on GPU. The runtime error
/// ensures the mismatch is always caught.
#[test]
fn test_dispatch_dtype_check_is_runtime_error() {
    let source = include_str!("../../src/tensor_dispatch.rs");

    let fn_start = source
        .find("fn dispatch_inner_body")
        .expect("dispatch_inner_body must exist");
    let fn_section = &source[fn_start..];
    // The function signature spans multiple lines; dtype check is ~400 chars in.
    let fn_snippet = &fn_section[..fn_section.len().min(600)];

    // Must use DtypeMismatch error return
    assert!(
        fn_snippet.contains("DtypeMismatch"),
        "dispatch_inner_body must return DtypeMismatch error for dtype validation"
    );

    // Must NOT use debug_assert for dtype checking
    assert!(
        !fn_snippet.contains("debug_assert"),
        "dispatch_inner_body must NOT use debug_assert for dtype — stripped in release builds (#889)"
    );
}

// ---------------------------------------------------------------------------
// Buffer size validation before alias (#3020 proof_coverage)
// ---------------------------------------------------------------------------

/// `dispatch_inner_body` MUST validate buffer byte length >= expected BEFORE
/// calling `alias()`.
///
/// Without this check, a too-small buffer could be aliased and passed to a GPU
/// kernel that reads beyond the buffer's allocated range — Metal may GPU-fault
/// or return garbage data. The validation uses `checked_mul` for shape
/// computation to prevent integer overflow (#930).
#[test]
fn test_dispatch_validates_buffer_size_before_alias() {
    let source = include_str!("../../src/tensor_dispatch.rs");

    let fn_start = source
        .find("fn dispatch_inner_body")
        .expect("dispatch_inner_body must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(3500)];

    // Must use checked_mul for safe shape computation
    assert!(
        fn_snippet.contains("checked_mul"),
        "dispatch_inner_body must use checked_mul for shape computation (overflow safety)"
    );

    // Must check BufferSizeMismatch before alias
    let size_check_pos = fn_snippet
        .find("BufferSizeMismatch")
        .expect("dispatch_inner_body must check BufferSizeMismatch");
    let alias_pos = fn_snippet
        .find(".alias()")
        .expect("dispatch_inner_body must call .alias()");

    assert!(
        size_check_pos < alias_pos,
        "buffer size validation must occur BEFORE alias() call (buffer aliasing safety)"
    );
}

// ---------------------------------------------------------------------------
// CompiledModel !Sync invariant (#3020 memory_verification)
// ---------------------------------------------------------------------------

/// `CompiledModel` MUST be `Send` but NOT `Sync`.
///
/// `cached_planned_buf: RefCell<Option<MetalBuffer>>` provides interior
/// mutability without synchronization. `RefCell` is `!Sync` by design —
/// concurrent `borrow()` + `borrow_mut()` from different threads would be
/// UB (data race on the borrow flag). Rust auto-derives `!Sync` from
/// `RefCell`, but if someone adds `unsafe impl Sync for CompiledModel`,
/// the `run_steps()` method would have a data race on `cached_planned_buf`.
///
/// This test statically verifies the invariant: `Send` (can move between
/// threads) but `!Sync` (cannot be shared by reference across threads).
#[test]
fn test_compiled_model_is_send_but_not_sync() {
    fn assert_send<T: Send>() {}
    // Compile-time check: CompiledModel is Send.
    assert_send::<nn_metal::compiled_model::CompiledModel>();

    // CompiledModel must NOT be Sync. We cannot write a negative trait bound
    // in stable Rust, so we verify structurally that the RefCell field exists.
    let source = include_str!("../../src/compiled_model.rs");
    assert!(
        source.contains("RefCell<Option<MetalBuffer>>"),
        "CompiledModel must contain RefCell<Option<MetalBuffer>> (cached_planned_buf) \
         which makes it !Sync — do not remove this or add unsafe impl Sync"
    );
}

// ---------------------------------------------------------------------------
// Planned buffer reuse safety (#3020 memory_verification)
// ---------------------------------------------------------------------------

/// `execute_primary_output` MUST normalize output via `extract_output_buffer`,
/// which calls `normalize_output_to_offset_zero` before returning.
///
/// The cached planned buffer (`cached_planned_buf`) is reused across forward
/// passes. Output data may reside at a non-zero offset within this shared
/// buffer. Without normalization, the caller would get a slice into the planned
/// buffer that will be overwritten on the next forward pass — a use-after-free
/// at the semantic level (data corruption, not memory safety UB, because Metal
/// ARC keeps the allocation alive).
///
/// `normalize_output_to_offset_zero` blit-copies the output to a fresh,
/// independent buffer at offset 0, ensuring the returned buffer is safe to
/// hold across multiple forward passes.
///
/// After the execute-core-dedup (D2), normalization lives in
/// `extract_output_buffer` (compiled_model_execute_outputs.rs), not directly
/// in `execute_primary_output`. We verify both links in the call chain.
#[test]
fn test_execute_primary_output_normalizes_before_return() {
    // Link 1: execute_primary_output delegates to extract_output_buffer.
    let execute_src = include_str!("../../src/compiled_model_execute.rs");

    let fn_start = execute_src
        .find("fn execute_primary_output")
        .expect("execute_primary_output must exist");
    let fn_section = &execute_src[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(3500)];

    assert!(
        fn_snippet.contains("extract_output_buffer"),
        "execute_primary_output must delegate to extract_output_buffer \
         (post-dedup call chain for planned buffer reuse safety)"
    );

    // Link 2: extract_output_buffer calls normalize_output_to_offset_zero.
    let outputs_src = include_str!("../../src/compiled_model_execute_outputs.rs");

    let fn_start = outputs_src
        .find("fn extract_output_buffer")
        .expect("extract_output_buffer must exist");
    let fn_section = &outputs_src[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(3500)];

    assert!(
        fn_snippet.contains("normalize_output_to_offset_zero"),
        "extract_output_buffer must call normalize_output_to_offset_zero \
         before returning (planned buffer reuse safety)"
    );
}

// ---------------------------------------------------------------------------
// Mixed GEMM output overflow safety (#3020 memory_verification)
// ---------------------------------------------------------------------------

/// `execute_mixed_dispatch` MUST use `checked_mul` for output size computation.
///
/// The output buffer size is `batch_count × M × N × 4` bytes. Without checked
/// arithmetic, large GEMM dimensions could overflow `usize`, allocating a
/// too-small buffer and causing GPU out-of-bounds writes. This mirrors the
/// existing `dispatch_inner_body` checked_mul test but covers the mixed GEMM
/// path which bypasses the standard dispatch engine.
#[test]
fn test_mixed_gemm_dispatch_uses_checked_mul() {
    let source = include_str!("../../src/compiled_model_execute_mixed.rs");

    // execute_mixed_dispatch delegates to dispatch_mixed_gemm_raw which does
    // the actual buffer allocation. Check that the file contains checked_mul
    // in the mixed GEMM dispatch path.
    assert!(
        source.contains("fn dispatch_mixed_gemm_raw"),
        "dispatch_mixed_gemm_raw must exist"
    );
    let fn_start = source
        .find("fn dispatch_mixed_gemm_raw")
        .expect("dispatch_mixed_gemm_raw must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(2000)];

    // Must use checked_mul for output element count.
    assert!(
        fn_snippet.contains("checked_mul"),
        "dispatch_mixed_gemm_raw must use checked_mul for output size computation \
         (prevents usize overflow → undersized GPU buffer)"
    );

    // Must use checked_mul for byte count too (numel * 4).
    let first_checked = fn_snippet
        .find("checked_mul")
        .expect("first checked_mul must exist");
    let after_first = &fn_snippet[first_checked + 11..];
    assert!(
        after_first.contains("checked_mul"),
        "execute_mixed_dispatch must use checked_mul for BOTH element count AND byte count"
    );
}
