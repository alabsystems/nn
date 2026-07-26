// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! # Blit Copy Analysis for Kokoro TTS Pipeline
//!
//! Investigative analysis of the ~152 GPU blit copies (buffer-to-buffer copies)
//! that account for ~45% of total GPU encodings (compute + blits = ~303) in
//! the Kokoro TTS pipeline.
//!
//! ## Blit Copy Sources
//!
//! There are **5 production code paths** that produce GPU blit copies:
//!
//! ### Source 1: `relocate_to_planned_buffer` (ALL 152 counted blits)
//!
//! **File:** `compiled_model_execute_helpers.rs:80-139`
//! **Counter:** Increments `TOTAL_BLITS` via `ensure_batch_for_blit()`
//!
//! This is the ONLY source of blits counted in `stats.blits`. Every compiled
//! step with a non-zero `step_sizes[i]` in the `BufferPlan` triggers a blit
//! to relocate the step's output from its arena allocation into the planned
//! contiguous buffer. The blit is SKIPPED when the step wrote directly into
//! the planned buffer (detected via `is_same_allocation` pointer check).
//!
//! **How it works in `run_steps_inner` (compiled_model_execute_steps.rs:363-381):**
//! ```text
//! if step has planned_offset AND step_sizes > 0:
//!     if output buffer IS the planned buffer (pointer identity):
//!         skip blit (already_in_planned = true)
//!     else:
//!         blit_copy(output -> planned_buffer[offset..offset+size])  // +1 TOTAL_BLITS
//! ```
//!
//! ### Source 2: `dispatch_execute_plan` output normalization (UNCOUNTED)
//!
//! **File:** `tensor_dispatch.rs:357-372`
//! **Counter:** NOT counted in `TOTAL_BLITS` (uses `encode_into_lazy_batch`)
//!
//! When an IR Dispatch step's output buffer has `out_offset != 0` or
//! `out_buf.len() != out_bytes`, the output is blit-copied to a fresh
//! zero-offset buffer. This happens when the arena sub-allocates at a
//! non-zero offset within the shared arena buffer.
//!
//! Since this blit is encoded inside `encode_into_lazy_batch` without calling
//! `ensure_batch_for_blit()`, it does NOT increment `TOTAL_BLITS`. It is
//! counted as part of the Dispatch step's single `TOTAL_ENCODINGS` increment
//! from `get_or_create_batch()`.
//!
//! This means the actual number of Metal blit encoder commands is HIGHER than
//! `stats.blits` reports. The uncounted blits from Source 2 are hidden inside
//! the compute encoding count.
//!
//! ### Source 3: `normalize_output_to_offset_zero` (UNCOUNTED)
//!
//! **File:** `compiled_model_execute_helpers.rs:39-72`
//! **Counter:** Uses `encode_into_lazy_batch` (not `ensure_batch_for_blit`)
//!
//! Called when extracting the model's final output (`extract_output_buffer` in
//! `compiled_model_execute_outputs.rs:118`). If the output slice has a non-zero
//! byte offset or oversized buffer, a blit normalizes it to offset 0.
//!
//! For models using the buffer planner, this fires when the output step was
//! already relocated to the planned buffer (which has offset > 0 for all but
//! the first step). So each compiled segment execution ends with 1 extra
//! uncounted blit per output to normalize offset → 0.
//!
//! ### Source 4: Packed Stack/Concat blits (tensor_dispatch_packed.rs)
//!
//! When an operation has >28 inputs, individual input buffers are blit-copied
//! into a single packed buffer. Not significant for Kokoro's pipeline shape.
//!
//! ### Source 5: `blit_fill` for atomic counters (norm_conv_stats)
//!
//! Zero-fills atomic counter buffers before multi-threadgroup reduction
//! dispatches. Minor — only fires for multi-TG norm+conv fusion.
//!
//! ## Why IR Dispatch Steps Always Blit (The Main Optimization Target)
//!
//! **This is the key finding.** There are two interacting mechanisms that
//! prevent IR Dispatch steps from skipping relocation blits:
//!
//! ### Mechanism 1: No planned-buffer redirect armed
//!
//! The `arm_native_op_redirect` mechanism (which allows a step to write
//! directly into the planned buffer, skipping the relocation blit) is
//! armed for `NativeOp` steps and `mixed GEMM` steps, but NOT for plain
//! IR Dispatch steps:
//!
//! ```text
//! compiled_model_execute_steps.rs:
//!   CompiledStep::NativeOp => {
//!       let _redirect_guard = helpers::arm_native_op_redirect(...);
//!       let output = self.execute_native_op(...);  // arena sees redirect
//!   }
//!
//!   CompiledStep::Dispatch (mixed GEMM) => {
//!       let _redirect_guard = helpers::arm_native_op_redirect(...);
//!       let output = self.execute_mixed_dispatch(...);  // returns GpuSlice
//!   }
//!
//!   CompiledStep::Dispatch (IR) => {
//!       // NO redirect armed — and arming one would NOT help (see below)
//!       let output = self.execute_dispatch(...);
//!       // relocate_to_planned_buffer ALWAYS fires
//!   }
//! ```
//!
//! ### Mechanism 2: `dispatch_execute_plan` output normalization (THE BLOCKER)
//!
//! Even if a planned-buffer redirect WERE armed for IR Dispatch steps, it
//! would not eliminate blits. Here is why:
//!
//! `dispatch_execute_plan` (tensor_dispatch.rs:357-372) normalizes its
//! output before returning. If the output buffer has `out_offset != 0` or
//! `out_buf.len() != out_bytes`, it creates a FRESH zero-offset buffer and
//! blit-copies the output into it. When the planned-buffer redirect routes
//! output to the planned buffer, `out_offset` is the planned region offset
//! (non-zero for all but the first step). The normalization fires, creating
//! a fresh buffer, and the `is_same_allocation` check in
//! `relocate_to_planned_buffer` FAILS — resulting in a DOUBLE blit:
//!
//! ```text
//! With naive redirect for IR Dispatch:
//!   1. Arena redirect → allocates in planned buffer at non-zero offset
//!   2. dispatch_execute_plan normalizes → blit to FRESH buffer (uncounted)
//!   3. relocate_to_planned_buffer → blit from fresh buffer BACK to planned
//! Result: 2 blits instead of 1 — WORSE than no redirect
//! ```
//!
//! This normalization does NOT happen for:
//! - **NativeOp steps**: They return `GpuSlice` directly without normalization.
//! - **Mixed GEMM steps**: `dispatch_mixed_gemm_raw` returns `GpuSlice::new(out_buf, out_offset)` — no normalization.
//!
//! For NativeOp steps, `arena_alloc_or_create` in `arena_scope.rs:366-372`
//! checks the planned redirect first. If the requested byte count matches
//! `expected_bytes`, it returns the planned buffer region directly. The step
//! writes into the planned buffer, the `is_same_allocation` check succeeds,
//! and the blit is skipped.
//!
//! For IR Dispatch steps, even with a redirect, `dispatch_execute_plan`
//! destroys the planned-buffer identity via normalization. The output ends
//! up in a fresh zero-offset buffer, and `relocate_to_planned_buffer` must
//! blit it into the planned region. This is the single largest source of
//! avoidable blits, but eliminating it requires changing `dispatch_execute_plan`
//! to skip normalization when the caller handles non-zero offsets (see R1+R2).
//!
//! ## Reduction Opportunities
//!
//! ### R0: Planned-buffer redirect for mixed GEMM steps (IMPLEMENTED)
//!
//! **Savings:** Variable (depends on how many steps use mixed GEMM path)
//!
//! Mixed GEMM steps (`dispatch_mixed_gemm_raw`) return `GpuSlice` with
//! offset directly — they do NOT go through `dispatch_execute_plan`'s
//! normalization. This means the planned-buffer redirect works correctly
//! for mixed GEMM steps: the redirect arms the planned buffer, the GEMM
//! allocates into it, and `is_same_allocation` detects the match and
//! skips the relocation blit.
//!
//! **Status:** DONE. Redirect guards added for both the Dispatch mixed GEMM
//! branch (autocast path) and the NativeOp mixed GEMM branch in
//! `compiled_model_execute_steps.rs`.
//!
//! ### R1+R2: Skip normalization + arm redirect for IR Dispatch (HIGH IMPACT)
//!
//! **Estimated savings:** 60-80 blits (all IR Dispatch steps with planned offsets)
//!
//! **These two changes are coupled — R1 without R2 causes DOUBLE blits.**
//!
//! **R2 (prerequisite):** Modify `dispatch_execute_plan` to skip its
//! output normalization blit when the caller handles non-zero offsets.
//! Currently, `dispatch_execute_plan` (tensor_dispatch.rs:357-372) checks
//! `out_offset == 0 && out_buf.len() == out_bytes` and blits to a fresh
//! zero-offset buffer if either fails. For planned-buffer allocations,
//! `out_offset != 0`, so it ALWAYS blits.
//!
//! **Implementation options for R2:**
//! - Thread-local flag (like `arm_planned_redirect`): compiled model
//!   execution sets `skip_dispatch_normalization = true` before calling
//!   `execute_dispatch`, and `dispatch_execute_plan` checks it before
//!   normalizing. Minimal blast radius — other callers are unaffected.
//! - New parameter: `dispatch_execute_plan(..., normalize_output: bool)`.
//!   More explicit but requires changing the function signature.
//!
//! **R1 (after R2):** Arm `arm_native_op_redirect` for IR Dispatch steps
//! in `run_steps_inner`, same pattern as NativeOp. With normalization
//! skipped, the output stays in the planned buffer, `is_same_allocation`
//! succeeds, and the relocation blit is skipped.
//!
//! **Why R1 alone fails:** If the redirect routes the output allocation to
//! the planned buffer (non-zero offset), `dispatch_execute_plan` normalizes
//! it to a fresh zero-offset buffer. The planned-buffer identity is
//! destroyed. Then `relocate_to_planned_buffer` blits from the fresh buffer
//! back to the planned region. Result: 2 blits instead of 1.
//!
//! **Complication:** IR Dispatch steps may have multi-step dispatch plans
//! (e.g., norm decomposition with intermediate allocations). The redirect
//! is single-use and consumed on the first matching allocation. If an
//! intermediate matches `expected_bytes`, the redirect is consumed early.
//! The `is_same_allocation` fallback blit is safe in all cases.
//!
//! **Risk:** MEDIUM. R2 changes `dispatch_execute_plan` behavior. Needs
//! analysis of all callers. A thread-local opt-in flag minimizes blast
//! radius — only compiled model execution opts in.
//!
//! ### R3: Reduce buffer planner allocations (LOW IMPACT)
//!
//! **Estimated savings:** 5-10 blits
//!
//! Some steps that produce small outputs (e.g., ConstantValue for scalar
//! parameters) still get planned offsets and trigger relocation blits. The
//! buffer planner could skip allocation for steps below a size threshold
//! (e.g., < 256 bytes), avoiding the blit overhead for tiny buffers.
//!
//! ### R4: Fuse consecutive blit + compute patterns (PARTIALLY DONE)
//!
//! Metal command buffers serialize encoder execution. A blit followed by a
//! compute dispatch that reads the blit destination is serialized by the GPU
//! hardware. However, each blit encoder creation has CPU overhead (ObjC
//! `new_blit_command_encoder` call + autoreleasepool). Batching multiple blits
//! into a single blit encoder reduces this overhead.
//!
//! **Status:** `CommandBatch::blit_copy_batch` API added (#4264). Encodes
//! N copy operations into a single blit encoder, amortizing the ObjC
//! encoder creation/destruction cost. Available for future use in segment
//! boundaries and packed dispatch paths.
//!
//! ### R5: Increase MAX_LAZY_ENCODINGS (DONE)
//!
//! Kokoro's 239 total events (181 compute + 58 blits) triggered 1
//! mid-pipeline auto-flush at MAX_LAZY_ENCODINGS=128. Increased to 256
//! (#4264). Eliminates 1 `commit_and_wait` stall per synthesis call.
//! Safe after StaleArenaRead fix (arena reset removed from auto-flush).
//!
//! ## Per-Segment Blit Breakdown (Estimated)
//!
//! Based on the pipeline structure (8 compiled segments, each with Dispatch
//! and NativeOp steps that have planned offsets):
//!
//! | Segment       | Steps | Dispatch Steps | NativeOp Steps | Est. Blits |
//! |---------------|-------|----------------|----------------|------------|
//! | plbert        | ~25   | ~14            | ~0             | ~14        |
//! | text          | ~20   | ~13            | ~2             | ~13-15     |
//! | prosody       | ~25   | ~15            | ~4             | ~15-19     |
//! | regulate      | ~10   | ~5             | ~0             | ~5         |
//! | f0            | ~40   | ~30            | ~2             | ~30-32     |
//! | generator     | ~60   | ~46            | ~10            | ~46-56     |
//! | sinegen_pre   | ~10   | ~6             | ~0             | ~6         |
//! | sinegen_post  | ~10   | ~7             | ~0             | ~7         |
//! | **Total**     | ~200  | ~136           | ~18            | ~136-152   |
//!
//! The ~136 Dispatch-step blits are the primary target. NativeOp blits are
//! partially mitigated by the existing planned-buffer redirect, but some
//! NativeOps with multi-dispatch patterns (FusedResBlock) may still trigger
//! blits when the redirect is consumed by an intermediate allocation.
//!
//! ## Summary of Findings
//!
//! 1. **All 152 counted blits** come from `relocate_to_planned_buffer` in the
//!    compiled model execution loop.
//!
//! 2. **IR Dispatch steps never skip blits** due to TWO interacting mechanisms:
//!    (a) no planned-buffer redirect armed, AND (b) `dispatch_execute_plan`
//!    normalizes output to zero-offset, destroying planned-buffer identity.
//!    Simply arming the redirect (R1 alone) causes DOUBLE blits — the
//!    normalization must be skipped first (R2).
//!
//! 3. **Mixed GEMM steps CAN skip blits** because `dispatch_mixed_gemm_raw`
//!    returns `GpuSlice` with offset (no normalization). Redirect guards
//!    have been added for both Dispatch and NativeOp mixed GEMM branches.
//!
//! 4. **Additional uncounted blits** exist in `dispatch_execute_plan` (Source 2)
//!    and `normalize_output_to_offset_zero` (Source 3). These are not tracked
//!    by `TOTAL_BLITS` and inflate the actual Metal encoder count beyond what
//!    `stats.blits` reports.
//!
//! 5. **The buffer planner works correctly** — it minimizes peak GPU memory
//!    by aliasing non-overlapping lifetimes. The blits are the mechanism for
//!    making this memory reuse work. The goal is not to eliminate the planner,
//!    but to eliminate the *relocation* blits by having steps write directly
//!    into the planned regions.
//!
//! 6. **No `.to_device()` blits in the hot path.** `DynTensor::to_device()`
//!    returns `Ok(self.clone())` when already on the correct device. The
//!    `.to_device(&gpu())` calls in `compiled_kokoro_steps.rs` are no-ops
//!    for tensors already on Metal.
//!
//! ## Recommended Implementation Order
//!
//! 1. **R0** (mixed GEMM redirect) — DONE. Low risk, safe for steps that
//!    bypass `dispatch_execute_plan`.
//!
//! 2. **R2** (skip `dispatch_execute_plan` normalization) — DONE (#4264).
//!    Thread-local `SKIP_DISPATCH_NORMALIZATION` flag + RAII guard. Only
//!    compiled model execution opts in.
//!
//! 3. **R1** (planned redirect for IR Dispatch steps) — DONE (#4264). Arms
//!    redirect for all Dispatch steps. High impact: ~60-80 fewer blits.
//!
//! 4. **R3** (skip planner for tiny buffers) — minor gains, low risk.
//!
//! 5. **R4** (multi-copy blit batching) — API added (`blit_copy_batch`),
//!    awaiting integration into step loop for segment-boundary batching.
//!
//! 6. **R5** (increase MAX_LAZY_ENCODINGS) — DONE (#4264). 128 → 256.
//!    Eliminates 1 mid-pipeline auto-flush for Kokoro.
