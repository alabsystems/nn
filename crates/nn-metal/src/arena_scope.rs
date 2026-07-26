// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Thread-local arena scope for GPU buffer allocation.
//!
//! Provides [`with_arena`], [`without_arena`], and [`arena_alloc_or_create`] —
//! the allocation routing layer that dispatches to explicit arenas, the
//! default always-on arena, or standalone `create_buffer_zeroed` fallback.
//!
//! Extracted from `arena.rs` to keep both files under 450 lines (#2218).

use std::cell::{Cell, RefCell};

use super::ActivationArena;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;

/// Default arena capacity (64 MB). With auto-grow enabled (#4289), the arena
/// doubles in size on overflow instead of falling back to standalone allocation.
pub(crate) const DEFAULT_ARENA_CAPACITY: usize = 64 * 1024 * 1024;

thread_local! {
    /// Active arena for the current thread (from `with_arena` scope).
    static ARENA: RefCell<Option<*mut ActivationArena>> = const { RefCell::new(None) };

    /// Always-on default arena. Auto-created on first GPU allocation. Reset
    /// on each `gpu_scope::flush()` boundary, matching the lazy batch model.
    pub(super) static DEFAULT_ARENA: RefCell<Option<ActivationArena>> = const { RefCell::new(None) };

    /// Cumulative arena hit count since last `reset_arena_stats()`.
    pub(super) static ARENA_HIT_COUNT: Cell<usize> = const { Cell::new(0) };

    /// Cumulative arena miss (overflow → fresh alloc) count.
    pub(super) static ARENA_MISS_COUNT: Cell<usize> = const { Cell::new(0) };

    /// Arena generation from the most recent `arena_alloc_or_create` call.
    ///
    /// Set to `Some(gen)` on arena hit, `None` on fallback (fresh alloc).
    /// Read by `MetalTensorData` constructors at the dispatch boundary to
    /// stamp arena-backed tensors with generation info for stale-read
    /// detection (#2328).
    pub(super) static LAST_ALLOC_GEN: Cell<Option<u64>> = const { Cell::new(None) };

    /// When `true`, `arena_alloc_or_create` routes ALL allocations to
    /// standalone `create_buffer_zeroed` buffers, bypassing both explicit
    /// and default arenas. Set by [`without_arena`] scope. (#2372)
    static ARENA_BYPASS: Cell<bool> = const { Cell::new(false) };

    /// Decode scope: arena generation at which the current autoregressive
    /// decode scope started. When `Some(gen)`, any arena-backed tensor with
    /// `alloc_gen >= gen` is considered non-stale, even if the arena has
    /// advanced multiple generations (due to flush/reset cycles within the
    /// decode loop). Set by [`with_decode_scope`], cleared on exit. (#3359)
    static DECODE_SCOPE_GEN: Cell<Option<u64>> = const { Cell::new(None) };

    /// Planned-buffer redirect for NativeOp steps (#3448).
    ///
    /// When armed, `arena_alloc_or_create` returns the planned buffer region
    /// instead of arena allocation when the requested byte count matches
    /// `expected_bytes`. Single-use: consumed on the first matching allocation.
    ///
    /// For single-dispatch NativeOps (InstanceNorm, LayerNorm, etc.), the
    /// redirect targets the output allocation. For multi-dispatch NativeOps,
    /// the redirect may be consumed by an intermediate of the same size — the
    /// blit-skip detection in `run_steps_inner` handles this safely by
    /// checking pointer identity after execution.
    static PLANNED_REDIRECT: RefCell<Option<PlannedRedirect>> = const { RefCell::new(None) };
}

/// Armed redirect state: planned buffer region + expected allocation size.
struct PlannedRedirect {
    buffer: MetalBuffer,
    offset: usize,
    expected_bytes: usize,
}

/// Execute a closure with all GPU buffer allocations routed through `arena`.
///
/// Intermediate buffers created by dispatch helpers during `f` are
/// sub-allocated from `arena` instead of hitting the Metal allocator.
/// The arena is NOT automatically reset after `f` — the caller controls
/// reset timing to ensure output tensors remain valid.
///
/// # Nesting
///
/// Nested `with_arena` calls reuse the outer arena. The inner call is a no-op.
///
/// # Safety contract
///
/// The arena reference is stored as a raw pointer in thread-local storage
/// for the duration of `f`. This is safe because:
/// - The `&mut arena` borrow is exclusive for the duration of `f`.
/// - The thread-local is cleared before `with_arena` returns.
/// - The arena cannot be accessed from other threads (thread-local).
pub fn with_arena<F, T>(arena: &mut ActivationArena, f: F) -> T
where
    F: FnOnce() -> T,
{
    let already_active = ARENA.with(|cell| cell.borrow().is_some());
    if already_active {
        return f();
    }

    let arena_ptr: *mut ActivationArena = arena;
    ARENA.with(|cell| *cell.borrow_mut() = Some(arena_ptr));

    // Use catch_unwind to guarantee TLS cleanup even if f() panics.
    // Without this, a panic would skip the cleanup below, leaving a
    // dangling raw pointer in TLS that arena_alloc_or_create would
    // dereference on the next call — undefined behavior.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

    ARENA.with(|cell| *cell.borrow_mut() = None);

    match result {
        Ok(val) => val,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Route all GPU allocations to standalone buffers within this scope.
///
/// Tensors allocated inside `without_arena` are never recycled by the arena
/// and remain valid until explicitly dropped. Use this when a model forward
/// pass produces intermediate tensors that must outlive the arena generation
/// (e.g., dvoice Kokoro decoder with ~100+ ops). (#2372, #2371, #2373)
///
/// # Nesting
///
/// Bypass is Priority 0 in `arena_alloc_or_create` — it always wins.
/// `without_arena` inside `with_arena` → bypass wins (standalone buffer).
/// `with_arena` inside `without_arena` → bypass still wins. This is
/// intentional: if the caller requested no arena, honor it unconditionally.
pub fn without_arena<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    struct Guard {
        prev: bool,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            ARENA_BYPASS.with(|c| c.set(self.prev));
        }
    }
    let _guard = Guard {
        prev: ARENA_BYPASS.with(Cell::get),
    };
    ARENA_BYPASS.with(|c| c.set(true));
    f()
}

/// Returns `true` if arena bypass is active on the current thread.
pub(crate) fn is_arena_bypassed() -> bool {
    ARENA_BYPASS.with(Cell::get)
}

/// Returns `true` if an arena scope is active on the current thread.
pub(crate) fn is_arena_active() -> bool {
    ARENA.with(|cell| cell.borrow().is_some())
}

/// Execute a closure within a decode scope that prevents stale arena reads.
///
/// Autoregressive decode loops (Qwen3-TTS, GPT-style generation) call
/// `model_fn` repeatedly, with `flush()` → arena reset between steps.
/// Tensors allocated in earlier decode steps (e.g., KV cache updates,
/// intermediate results) would be flagged as stale by the generation check
/// in `gpu_to_cpu`, because the arena generation advances on each flush.
///
/// `with_decode_scope` records the arena generation at entry. Any tensor
/// allocated at or after that generation is considered non-stale for the
/// duration of the scope, regardless of how many arena resets occur.
///
/// # Safety contract
///
/// This is safe because ObjC ARC keeps the underlying Metal buffer alive
/// as long as any `DynTensor` holds a reference. The arena bump pointer
/// reset does NOT deallocate — it only reclaims logical space. The stale
/// check is a defense-in-depth diagnostic, not a memory safety guard.
/// Suppressing it within a decode scope is correct when the caller knows
/// tensors will be consumed before the scope exits.
///
/// # Nesting
///
/// Nested `with_decode_scope` calls preserve the outer scope's generation
/// (the outermost scope wins, since it has the earliest generation).
///
/// # Kokoro usage
///
/// Kokoro is non-autoregressive (single forward pass), but uses decode scope
/// to suppress false stale-read errors caused by mid-pipeline arena resets
/// from sync() (prefix-sum readback) and auto-flush (128-encoding threshold).
/// Production D=512 exceeds the auto-flush limit, triggering multiple arena
/// generation advances within a single synthesis call. Part of #4264.
///
/// See #3359.
pub fn with_decode_scope<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let already_active = DECODE_SCOPE_GEN.with(Cell::get).is_some();
    if already_active {
        // Nested: reuse the outer scope's generation.
        return f();
    }

    // Record the current default arena generation as the scope baseline.
    // If no arena exists yet, generation 0 is the correct baseline — any
    // tensor allocated after this point will have gen >= 0.
    let scope_gen = super::stats::default_arena_generation().unwrap_or(0);
    DECODE_SCOPE_GEN.with(|c| c.set(Some(scope_gen)));

    // Use catch_unwind for cleanup safety, matching with_arena pattern.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

    DECODE_SCOPE_GEN.with(|c| c.set(None));

    match result {
        Ok(val) => val,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Returns the decode scope generation if a decode scope is active.
///
/// Used by `gpu_to_cpu` stale-read detection: if `alloc_gen >= decode_scope_gen`,
/// the tensor is within the decode scope and should not be considered stale.
pub(crate) fn decode_scope_generation() -> Option<u64> {
    DECODE_SCOPE_GEN.with(Cell::get)
}

/// Arm the planned-buffer redirect for the next matching allocation (#3448).
///
/// When armed, `arena_alloc_or_create` will return a view into the planned
/// buffer (at `offset`) instead of an arena allocation, provided the
/// requested byte count matches `expected_bytes`. The redirect is single-use:
/// consumed on the first matching allocation.
///
/// Call [`clear_planned_redirect`] after the NativeOp execution to disarm
/// any unconsumed redirect.
pub(crate) fn set_planned_redirect(buffer: &MetalBuffer, offset: usize, expected_bytes: usize) {
    PLANNED_REDIRECT.with(|cell| {
        *cell.borrow_mut() = Some(PlannedRedirect {
            buffer: buffer.alias(),
            offset,
            expected_bytes,
        });
    });
}

/// Disarm the planned-buffer redirect, discarding any unconsumed state.
///
/// Must be called after NativeOp execution to prevent a stale redirect
/// from being consumed by a subsequent step's allocation. Prefer using
/// [`PlannedRedirectGuard`] for automatic cleanup on error paths.
pub(crate) fn clear_planned_redirect() {
    PLANNED_REDIRECT.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// RAII guard that clears the planned-buffer redirect on drop (#3448).
///
/// Ensures cleanup even when `execute_native_op` returns `Err` and the
/// `?` operator bypasses the explicit `clear_planned_redirect()` call.
/// The guard is created by [`arm_planned_redirect_guard`].
pub(crate) struct PlannedRedirectGuard(());

impl Drop for PlannedRedirectGuard {
    fn drop(&mut self) {
        clear_planned_redirect();
    }
}

/// Arm the planned-buffer redirect and return an RAII guard that clears
/// it on drop. This is the preferred API — the guard ensures cleanup on
/// both success and error paths.
pub(crate) fn arm_planned_redirect_guard(
    buffer: &MetalBuffer,
    offset: usize,
    expected_bytes: usize,
) -> PlannedRedirectGuard {
    set_planned_redirect(buffer, offset, expected_bytes);
    PlannedRedirectGuard(())
}

/// Try to consume the planned-buffer redirect.
///
/// Returns `Some((buffer, offset))` if the redirect is armed and `bytes`
/// matches the expected allocation size. The redirect is consumed (disarmed)
/// on match. Returns `None` if not armed or size doesn't match.
fn take_planned_redirect(bytes: usize) -> Option<(MetalBuffer, usize)> {
    PLANNED_REDIRECT.with(|cell| {
        let guard = cell.borrow();
        if let Some(ref redirect) = *guard {
            if redirect.expected_bytes == bytes {
                drop(guard);
                let r = cell.borrow_mut().take().unwrap();
                return Some((r.buffer, r.offset));
            }
        }
        None
    })
}

/// Try allocating from `arena`; on overflow, fall back to standalone buffer.
///
/// Shared by Priority 1 (explicit `with_arena`) and Priority 2 (default arena).
/// On success, updates hit count and last alloc generation. On `ArenaOverflow`,
/// logs a diagnostic and creates a fresh buffer (#2914).
fn try_arena_alloc(
    arena: &mut ActivationArena,
    ctx: &MetalContext,
    bytes: usize,
) -> Result<(MetalBuffer, usize), MetalError> {
    match arena.alloc(bytes) {
        Ok(td) => {
            ARENA_HIT_COUNT.with(|c| c.set(c.get() + 1));
            LAST_ALLOC_GEN.with(|c| c.set(td.arena_generation()));
            Ok((td.buffer, td.byte_offset))
        }
        Err(MetalError::ArenaOverflow {
            requested,
            remaining,
            capacity,
        }) => {
            eprintln!(
                "[nn-metal] arena overflow: {requested}B req, {remaining}B free, {capacity}B cap"
            );
            ARENA_MISS_COUNT.with(|c| c.set(c.get() + 1));
            LAST_ALLOC_GEN.with(|c| c.set(None));
            super::pool::pool_acquire(ctx, bytes)
        }
        Err(e) => Err(e),
    }
}

/// Allocate a buffer from the active arena, or fall back to `create_buffer_zeroed`.
///
/// Returns `(buffer, byte_offset)`. When allocated from the arena, the
/// byte offset is the sub-allocation offset within the shared arena buffer.
/// When falling back to a fresh allocation, the offset is always 0.
///
/// # Allocation priority
///
/// 0. `without_arena` bypass (if active) — standalone buffer (#2372)
///    0.5. Planned-buffer redirect (if armed and size matches) — #3448
/// 1. Explicit `with_arena` scope (if active)
/// 2. Always-on default arena (auto-created on first call)
/// 3. Fresh `create_buffer_zeroed` (on arena overflow)
///
/// Callers MUST propagate the returned offset when binding the buffer for
/// GPU dispatch — discarding it causes writes to offset 0 of the shared
/// arena buffer, silently corrupting earlier sub-allocations.
pub(crate) fn arena_alloc_or_create(
    ctx: &MetalContext,
    bytes: usize,
) -> Result<(MetalBuffer, usize), MetalError> {
    // Priority 0: Bypass — route to buffer pool (#2372, #3079 D3).
    if is_arena_bypassed() {
        ARENA_MISS_COUNT.with(|c| c.set(c.get() + 1));
        LAST_ALLOC_GEN.with(|c| c.set(None));
        return super::pool::pool_acquire(ctx, bytes);
    }

    // Priority 0.5: Planned-buffer redirect for NativeOp steps (#3448).
    // Returns the pre-allocated planned buffer region when the requested
    // byte count matches, eliminating the post-execution blit relocation.
    if let Some((buf, offset)) = take_planned_redirect(bytes) {
        LAST_ALLOC_GEN.with(|c| c.set(None));
        return Ok((buf, offset));
    }

    // Priority 1: Explicit with_arena scope (graceful overflow via try_arena_alloc).
    if is_arena_active() {
        return ARENA.with(|cell| {
            let guard = cell.borrow();
            match guard.as_ref() {
                Some(&arena_ptr) => {
                    // SAFETY: raw pointer valid for duration of with_arena (catch_unwind cleanup).
                    let arena = unsafe { &mut *arena_ptr };
                    try_arena_alloc(arena, ctx, bytes)
                }
                None => {
                    LAST_ALLOC_GEN.with(|c| c.set(None));
                    super::pool::pool_acquire(ctx, bytes)
                }
            }
        });
    }

    // Priority 2: Always-on default arena with auto-grow (#4289).
    DEFAULT_ARENA.with(|cell| {
        let mut guard = cell.borrow_mut();
        if guard.is_none() {
            match ActivationArena::new(ctx, DEFAULT_ARENA_CAPACITY) {
                Ok(mut arena) => {
                    arena.set_auto_grow(ctx);
                    *guard = Some(arena);
                }
                Err(_) => {
                    ARENA_MISS_COUNT.with(|c| c.set(c.get() + 1));
                    LAST_ALLOC_GEN.with(|c| c.set(None));
                    return super::pool::pool_acquire(ctx, bytes);
                }
            }
        }
        let arena = guard.as_mut().expect("invariant: guard is Some");
        try_arena_alloc(arena, ctx, bytes)
    })
}

/// Reset the default arena's bump pointer.
///
/// Called by `gpu_scope::flush()` after `commit_and_wait` — all GPU work
/// is complete so intermediate arena buffers are no longer referenced by
/// pending command encoders. ObjC ARC keeps the underlying Metal buffer
/// alive for any `DynTensor` that still holds a reference.
pub(crate) fn reset_default_arena() {
    DEFAULT_ARENA.with(|cell| {
        if let Some(arena) = cell.borrow_mut().as_mut() {
            arena.reset();
        }
    });
}

/// Save the default arena's bump offset AND generation for later restore.
///
/// Returns `Option<(offset, generation)>`. The generation is checked at
/// restore time — if it doesn't match, the arena was reset (by flush or
/// auto-flush) between checkpoint and restore, making the checkpoint stale.
/// Skipping restore is safe because the arena is already clean after reset.
///
/// This structural guard eliminates the need for per-path `needs_cast` checks
/// that previously tried to skip checkpoint/restore around known flush sites.
/// See #2913 (original feature), #3133 (panic fix).
pub(crate) fn checkpoint_default_arena() -> Option<(usize, u64)> {
    DEFAULT_ARENA.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|a| (a.checkpoint(), a.generation()))
    })
}

/// Restore the default arena's bump offset to a saved checkpoint.
///
/// **Generation-guarded:** If the arena's current generation differs from
/// the saved generation, the arena was reset (flush/auto-flush) since the
/// checkpoint was taken. In that case, restore is skipped — the arena is
/// already at offset 0 from the reset.
///
/// See #3133: without this guard, `restore_checkpoint` panics when
/// `saved_offset > current_offset` (which happens after arena reset).
pub(crate) fn restore_default_arena(saved: Option<(usize, u64)>) {
    if let Some((offset, saved_gen)) = saved {
        DEFAULT_ARENA.with(|cell| {
            if let Some(arena) = cell.borrow_mut().as_mut() {
                if arena.generation() == saved_gen {
                    // Generation matches, so offset is valid — unwrap is safe.
                    let _ = arena.restore_checkpoint(offset);
                }
                // Generation mismatch: arena was reset between checkpoint
                // and restore. The checkpoint is stale — skip restore.
            }
        });
    }
}

/// Attempt to reset the thread-local arena. Returns `true` if a reset was performed.
///
/// Priority 1: explicit `with_arena` scope. Priority 2: default arena.
pub fn try_reset_active_arena() -> bool {
    if is_arena_bypassed() {
        return false;
    }

    // Priority 1: Explicit with_arena scope.
    if is_arena_active() {
        let did_reset = ARENA.with(|cell| {
            let guard = cell.borrow();
            match guard.as_ref() {
                Some(&arena_ptr) => {
                    // SAFETY: raw pointer valid for duration of with_arena scope.
                    let arena = unsafe { &mut *arena_ptr };
                    arena.reset();
                    true
                }
                None => false,
            }
        });
        if did_reset {
            return true;
        }
    }

    // Priority 2: Default arena.
    DEFAULT_ARENA.with(|cell| {
        if let Some(arena) = cell.borrow_mut().as_mut() {
            arena.reset();
            true
        } else {
            false
        }
    })
}

/// Ensure the default arena can hold at least `min_bytes` without growing.
///
/// Pre-sizes the arena before a heavy workload to avoid growth during the
/// hot path. If the default arena does not exist yet, it is created with
/// `max(DEFAULT_ARENA_CAPACITY, min_bytes)`. Part of #4289.
pub fn ensure_default_arena_capacity(
    ctx: &MetalContext,
    min_bytes: usize,
) -> Result<(), MetalError> {
    DEFAULT_ARENA.with(|cell| {
        let mut guard = cell.borrow_mut();
        if guard.is_none() {
            let cap = DEFAULT_ARENA_CAPACITY.max(min_bytes);
            let mut arena = ActivationArena::new(ctx, cap)?;
            arena.set_auto_grow(ctx);
            *guard = Some(arena);
            return Ok(());
        }
        let arena = guard.as_mut().expect("invariant: guard is Some");
        arena.ensure_capacity(ctx, min_bytes)
    })
}

/// Number of growth events in the current generation of the default arena.
///
/// Returns 0 if the default arena has not been initialized.
pub(crate) fn default_arena_growth_count() -> usize {
    DEFAULT_ARENA.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or(0, ActivationArena::growth_count)
    })
}

/// Total number of growth events since the default arena was created.
///
/// Returns 0 if the default arena has not been initialized.
/// Useful for diagnosing whether the initial arena capacity is sufficient
/// across an entire session. Part of #4289.
pub fn default_arena_total_growth_count() -> usize {
    DEFAULT_ARENA.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or(0, ActivationArena::total_growth_count)
    })
}

/// Number of overflow events in the current generation of the default arena.
///
/// Returns 0 if the default arena has not been initialized. Part of #4289.
pub(crate) fn default_arena_overflow_count() -> usize {
    DEFAULT_ARENA.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or(0, ActivationArena::overflow_count)
    })
}

/// Total overflow events since the default arena was created.
///
/// Returns 0 if the default arena has not been initialized. Part of #4289.
pub(crate) fn default_arena_total_overflow_count() -> usize {
    DEFAULT_ARENA.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or(0, ActivationArena::total_overflow_count)
    })
}

/// Cumulative overflow bytes in the current generation of the default arena.
///
/// Returns 0 if the default arena has not been initialized. Part of #4289.
pub(crate) fn default_arena_overflow_bytes() -> usize {
    DEFAULT_ARENA.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or(0, ActivationArena::overflow_bytes)
    })
}

/// Total overflow bytes since the default arena was created.
///
/// Returns 0 if the default arena has not been initialized. Part of #4289.
pub(crate) fn default_arena_total_overflow_bytes() -> usize {
    DEFAULT_ARENA.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or(0, ActivationArena::total_overflow_bytes)
    })
}
