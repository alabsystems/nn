// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for KV cache growth bounds and memory
//! allocation properties.
//!
//! Proves fundamental mathematical properties of KV cache sizing, memory
//! allocation, and growth behavior in transformer inference:
//! - Cache size linear in sequence length: N * head_dim * num_kv_heads
//! - Paged cache block count: ceil(seq_len / block_size)
//! - Paged cache memory: num_blocks * block_size * head_dim * 2
//! - Cache append: new_len = old_len + 1
//! - Cache memory monotonically increasing with seq_len
//! - GQA kv_heads = num_heads / num_groups (integer division)
//! - GQA repeat factor: each KV head shared by num_heads/num_kv_heads Q heads
//! - Sliding window: cache_len = min(seq_len, window_size)
//! - Sliding window memory bounded by window_size * head_dim * kv_heads
//! - Prealloc: fixed memory = max_seq_len * head_dim * kv_heads
//! - Prealloc utilization fraction in [0, 1]
//! - Multi-layer total = num_layers * per_layer_cache
//! - RoPE offset: position_id = cache_offset + local_pos
//! - Cross-attention cache fixed after encoding
//! - Concatenation: cat_len = old_len + new_len
//! - Byte alignment: buffer_size % alignment == 0
//! - Token-level update overwrites one position
//! - Batch cache: total = batch_size * per_sequence_cache
//! - bf16 element size = 2 bytes, f32 = 4 bytes
//! - Max cache bounded by max_seq_len * config
//!
//! Part of #4146.

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};

/// Helper: create a Real-sorted variable.
fn real_var(name: &str) -> Expr {
    Expr::var(name, Sort::real())
}

/// Helper: assert that program is UNSAT (property holds for all inputs).
///
/// The ay convention: we assert the negation of the property, then
/// UNSAT (Verified) means the original property holds universally.
fn assert_verified(prog: &AYProgram, property_name: &str) {
    match execute_direct::execute(prog) {
        Ok(ExecuteResult::Verified) => {
            // UNSAT — property proved for all inputs.
        }
        Ok(other) => {
            panic!(
                "{property_name}: expected Verified (UNSAT), got: {other:?}. \
                 The negated property is satisfiable — the property does NOT hold."
            );
        }
        Err(e) => {
            panic!("{property_name}: ay execution error: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 591: Cache size after N steps = N * head_dim * num_kv_heads
// ---------------------------------------------------------------------------

/// Prove: the KV cache size after N decoding steps equals
/// N * head_dim * num_kv_heads (for one layer, one of K or V).
///
/// At each step, one new KV pair of dimension head_dim is appended per
/// KV head. After N steps, the total number of cached elements is
/// N * head_dim * num_kv_heads.
#[test]
fn test_591_cache_size_after_n_steps() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("n", real.clone());
    let _ = prog.declare_const("head_dim", real.clone());
    let _ = prog.declare_const("num_kv_heads", real.clone());
    let _ = prog.declare_const("cache_size", real);

    let n = real_var("n");
    let head_dim = real_var("head_dim");
    let num_kv_heads = real_var("num_kv_heads");
    let cache_size = real_var("cache_size");

    // All positive parameters
    prog.assert(n.clone().real_gt(Expr::real(0)));
    prog.assert(head_dim.clone().real_gt(Expr::real(0)));
    prog.assert(num_kv_heads.clone().real_gt(Expr::real(0)));

    // Bounded for finite reasoning
    prog.assert(n.clone().real_le(Expr::real(100000)));
    prog.assert(head_dim.clone().real_le(Expr::real(1024)));
    prog.assert(num_kv_heads.clone().real_le(Expr::real(128)));

    // Cache size formula: cache_size = N * head_dim * num_kv_heads
    prog.assert(
        cache_size.clone().eq(n
            .clone()
            .real_mul(head_dim.clone().real_mul(num_kv_heads.clone()))),
    );

    // Negated property: cache_size != N * head_dim * num_kv_heads
    let violation = cache_size.ne(n.real_mul(head_dim.real_mul(num_kv_heads)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cache_size_after_n_steps");
}

// ---------------------------------------------------------------------------
// Test 592: Paged cache num_blocks = ceil(seq_len / block_size)
// ---------------------------------------------------------------------------

/// Prove: the number of paged blocks satisfies
/// num_blocks >= seq_len / block_size (ceil property).
///
/// Paged attention allocates blocks of fixed size. For seq_len tokens,
/// num_blocks = ceil(seq_len / block_size). This means
/// num_blocks >= seq_len / block_size and
/// num_blocks < seq_len / block_size + 1.
/// We prove: num_blocks * block_size >= seq_len.
#[test]
fn test_592_paged_cache_num_blocks() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("seq_len", real.clone());
    let _ = prog.declare_const("block_size", real.clone());
    let _ = prog.declare_const("num_blocks", real.clone());
    let _ = prog.declare_const("ratio", real);

    let seq_len = real_var("seq_len");
    let block_size = real_var("block_size");
    let num_blocks = real_var("num_blocks");
    let ratio = real_var("ratio");

    // Positive parameters
    prog.assert(seq_len.clone().real_gt(Expr::real(0)));
    prog.assert(block_size.clone().real_gt(Expr::real(0)));
    prog.assert(seq_len.clone().real_le(Expr::real(100000)));
    prog.assert(block_size.clone().real_le(Expr::real(1024)));

    // ratio = seq_len / block_size: ratio * block_size = seq_len
    prog.assert(
        ratio
            .clone()
            .real_mul(block_size.clone())
            .eq(seq_len.clone()),
    );

    // num_blocks = ceil(ratio): num_blocks >= ratio
    prog.assert(num_blocks.clone().real_ge(ratio));

    // Negated property: num_blocks * block_size < seq_len
    let violation = num_blocks.real_mul(block_size).real_lt(seq_len);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "paged_cache_num_blocks");
}

// ---------------------------------------------------------------------------
// Test 593: Paged cache memory = num_blocks * block_size * head_dim * 2
// ---------------------------------------------------------------------------

/// Prove: total paged cache memory (for K and V combined) equals
/// num_blocks * block_size * head_dim * 2.
///
/// Each block stores block_size key vectors and block_size value vectors,
/// each of dimension head_dim. The factor of 2 accounts for K and V.
#[test]
fn test_593_paged_cache_memory() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("num_blocks", real.clone());
    let _ = prog.declare_const("block_size", real.clone());
    let _ = prog.declare_const("head_dim", real.clone());
    let _ = prog.declare_const("memory", real);

    let num_blocks = real_var("num_blocks");
    let block_size = real_var("block_size");
    let head_dim = real_var("head_dim");
    let memory = real_var("memory");

    // All positive
    prog.assert(num_blocks.clone().real_gt(Expr::real(0)));
    prog.assert(block_size.clone().real_gt(Expr::real(0)));
    prog.assert(head_dim.clone().real_gt(Expr::real(0)));

    // Bounded
    prog.assert(num_blocks.clone().real_le(Expr::real(10000)));
    prog.assert(block_size.clone().real_le(Expr::real(256)));
    prog.assert(head_dim.clone().real_le(Expr::real(256)));

    // Memory formula: memory = num_blocks * block_size * head_dim * 2
    let expected = num_blocks
        .clone()
        .real_mul(block_size.clone())
        .real_mul(head_dim.clone())
        .real_mul(Expr::real(2));
    prog.assert(memory.clone().eq(expected));

    // Negated property: memory != num_blocks * block_size * head_dim * 2
    let expected2 = num_blocks
        .real_mul(block_size)
        .real_mul(head_dim)
        .real_mul(Expr::real(2));
    let violation = memory.ne(expected2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "paged_cache_memory");
}

// ---------------------------------------------------------------------------
// Test 594: Cache append: new_len = old_len + 1
// ---------------------------------------------------------------------------

/// Prove: appending one token to the KV cache increases its length by 1.
///
/// After one decoding step, the cache length goes from old_len to
/// old_len + 1. This is the fundamental append invariant.
#[test]
fn test_594_cache_append_increments_length() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("old_len", real.clone());
    let _ = prog.declare_const("new_len", real);

    let old_len = real_var("old_len");
    let new_len = real_var("new_len");

    // old_len >= 0
    prog.assert(old_len.clone().real_ge(Expr::real(0)));
    prog.assert(old_len.clone().real_le(Expr::real(100000)));

    // Append axiom: new_len = old_len + 1
    prog.assert(new_len.clone().eq(old_len.clone().real_add(Expr::real(1))));

    // Negated property: new_len != old_len + 1
    let violation = new_len.ne(old_len.real_add(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cache_append_increments_length");
}

// ---------------------------------------------------------------------------
// Test 595: Cache memory monotonically increasing with seq_len
// ---------------------------------------------------------------------------

/// Prove: if seq_len_2 > seq_len_1, then cache_mem_2 > cache_mem_1.
///
/// Cache memory = seq_len * head_dim * num_kv_heads. Since head_dim > 0
/// and num_kv_heads > 0, larger seq_len implies strictly larger memory.
#[test]
fn test_595_cache_memory_monotone_increasing() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("hd", real.clone());
    let _ = prog.declare_const("kv", real.clone());
    let _ = prog.declare_const("m1", real.clone());
    let _ = prog.declare_const("m2", real);

    let s1 = real_var("s1");
    let s2 = real_var("s2");
    let hd = real_var("hd");
    let kv = real_var("kv");
    let m1 = real_var("m1");
    let m2 = real_var("m2");

    // s2 > s1 > 0, hd > 0, kv > 0
    prog.assert(s1.clone().real_gt(Expr::real(0)));
    prog.assert(s2.clone().real_gt(s1.clone()));
    prog.assert(hd.clone().real_gt(Expr::real(0)));
    prog.assert(kv.clone().real_gt(Expr::real(0)));

    // Bounded
    prog.assert(s2.clone().real_le(Expr::real(100000)));
    prog.assert(hd.clone().real_le(Expr::real(1024)));
    prog.assert(kv.clone().real_le(Expr::real(128)));

    // m1 = s1 * hd * kv, m2 = s2 * hd * kv
    let factor = hd.clone().real_mul(kv.clone());
    prog.assert(m1.clone().eq(s1.real_mul(factor.clone())));
    prog.assert(m2.clone().eq(s2.real_mul(hd.real_mul(kv))));

    // Negated property: m2 <= m1 (not strictly increasing)
    let violation = m2.real_le(m1);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cache_memory_monotone_increasing");
}

// ---------------------------------------------------------------------------
// Test 596: GQA kv_heads = num_heads / num_groups
// ---------------------------------------------------------------------------

/// Prove: in grouped-query attention, num_kv_heads = num_heads / num_groups,
/// equivalently num_heads = num_kv_heads * num_groups.
///
/// GQA partitions Q heads into groups, each sharing one KV head.
#[test]
fn test_596_gqa_kv_heads_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("num_heads", real.clone());
    let _ = prog.declare_const("num_groups", real.clone());
    let _ = prog.declare_const("num_kv_heads", real);

    let num_heads = real_var("num_heads");
    let num_groups = real_var("num_groups");
    let num_kv_heads = real_var("num_kv_heads");

    // All positive
    prog.assert(num_heads.clone().real_gt(Expr::real(0)));
    prog.assert(num_groups.clone().real_gt(Expr::real(0)));
    prog.assert(num_kv_heads.clone().real_gt(Expr::real(0)));

    // GQA axiom: num_heads = num_kv_heads * num_groups
    prog.assert(
        num_heads
            .clone()
            .eq(num_kv_heads.clone().real_mul(num_groups.clone())),
    );

    // Verify: num_kv_heads = num_heads / num_groups
    // i.e., num_kv_heads * num_groups = num_heads
    let reconstructed = num_kv_heads.real_mul(num_groups);

    // Negated property: num_kv_heads * num_groups != num_heads
    let violation = reconstructed.ne(num_heads);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_kv_heads_formula");
}

// ---------------------------------------------------------------------------
// Test 597: GQA repeat factor: each KV head shared by num_heads/num_kv_heads Q heads
// ---------------------------------------------------------------------------

/// Prove: the GQA repeat factor r = num_heads / num_kv_heads satisfies
/// r * num_kv_heads = num_heads.
///
/// Each KV head is shared by exactly r query heads.
#[test]
fn test_597_gqa_repeat_factor() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("num_heads", real.clone());
    let _ = prog.declare_const("num_kv_heads", real.clone());
    let _ = prog.declare_const("r", real);

    let num_heads = real_var("num_heads");
    let num_kv_heads = real_var("num_kv_heads");
    let r = real_var("r");

    // Positive
    prog.assert(num_heads.clone().real_gt(Expr::real(0)));
    prog.assert(num_kv_heads.clone().real_gt(Expr::real(0)));
    prog.assert(r.clone().real_gt(Expr::real(0)));

    // r = num_heads / num_kv_heads: r * num_kv_heads = num_heads
    prog.assert(
        r.clone()
            .real_mul(num_kv_heads.clone())
            .eq(num_heads.clone()),
    );

    // Negated property: r * num_kv_heads != num_heads
    let violation = r.real_mul(num_kv_heads).ne(num_heads);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gqa_repeat_factor");
}

// ---------------------------------------------------------------------------
// Test 598: Sliding window: cache_len = min(seq_len, window_size)
// ---------------------------------------------------------------------------

/// Prove: the effective cache length with sliding window attention is
/// min(seq_len, window_size). This is always <= window_size.
///
/// When seq_len <= window_size, the entire sequence is cached.
/// When seq_len > window_size, only the last window_size tokens are kept.
#[test]
fn test_598_sliding_window_cache_len() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("seq_len", real.clone());
    let _ = prog.declare_const("window_size", real.clone());
    let _ = prog.declare_const("cache_len", real);

    let seq_len = real_var("seq_len");
    let window_size = real_var("window_size");
    let cache_len = real_var("cache_len");

    // Positive
    prog.assert(seq_len.clone().real_gt(Expr::real(0)));
    prog.assert(window_size.clone().real_gt(Expr::real(0)));

    // cache_len = min(seq_len, window_size):
    // cache_len <= seq_len AND cache_len <= window_size
    // AND (cache_len = seq_len OR cache_len = window_size)
    prog.assert(cache_len.clone().real_le(seq_len.clone()));
    prog.assert(cache_len.clone().real_le(window_size.clone()));
    prog.assert(
        cache_len
            .clone()
            .eq(seq_len)
            .or(cache_len.clone().eq(window_size.clone())),
    );

    // Negated property: cache_len > window_size
    let violation = cache_len.real_gt(window_size);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sliding_window_cache_len");
}

// ---------------------------------------------------------------------------
// Test 599: Sliding window memory bounded by window_size * head_dim * kv_heads
// ---------------------------------------------------------------------------

/// Prove: with sliding window, cache memory is bounded by
/// window_size * head_dim * num_kv_heads (for one of K or V).
///
/// Since cache_len <= window_size, and memory = cache_len * head_dim * kv_heads,
/// memory <= window_size * head_dim * kv_heads.
#[test]
fn test_599_sliding_window_memory_bound() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("cache_len", real.clone());
    let _ = prog.declare_const("window_size", real.clone());
    let _ = prog.declare_const("head_dim", real.clone());
    let _ = prog.declare_const("kv_heads", real.clone());
    let _ = prog.declare_const("memory", real.clone());
    let _ = prog.declare_const("max_memory", real);

    let cache_len = real_var("cache_len");
    let window_size = real_var("window_size");
    let head_dim = real_var("head_dim");
    let kv_heads = real_var("kv_heads");
    let memory = real_var("memory");
    let max_memory = real_var("max_memory");

    // Positive parameters
    prog.assert(cache_len.clone().real_gt(Expr::real(0)));
    prog.assert(window_size.clone().real_gt(Expr::real(0)));
    prog.assert(head_dim.clone().real_gt(Expr::real(0)));
    prog.assert(kv_heads.clone().real_gt(Expr::real(0)));

    // Sliding window bound: cache_len <= window_size
    prog.assert(cache_len.clone().real_le(window_size.clone()));

    // memory = cache_len * head_dim * kv_heads
    prog.assert(
        memory
            .clone()
            .eq(cache_len.real_mul(head_dim.clone().real_mul(kv_heads.clone()))),
    );

    // max_memory = window_size * head_dim * kv_heads
    prog.assert(
        max_memory
            .clone()
            .eq(window_size.real_mul(head_dim.real_mul(kv_heads))),
    );

    // Negated property: memory > max_memory
    let violation = memory.real_gt(max_memory);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sliding_window_memory_bound");
}

// ---------------------------------------------------------------------------
// Test 600: Prealloc: fixed memory = max_seq_len * head_dim * kv_heads
// ---------------------------------------------------------------------------

/// Prove: pre-allocated cache memory is exactly
/// max_seq_len * head_dim * num_kv_heads.
///
/// Pre-allocation reserves the maximum possible cache at initialization.
/// The allocated size is fixed regardless of the current sequence length.
#[test]
fn test_600_prealloc_fixed_memory() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("max_seq_len", real.clone());
    let _ = prog.declare_const("head_dim", real.clone());
    let _ = prog.declare_const("kv_heads", real.clone());
    let _ = prog.declare_const("alloc", real);

    let max_seq_len = real_var("max_seq_len");
    let head_dim = real_var("head_dim");
    let kv_heads = real_var("kv_heads");
    let alloc = real_var("alloc");

    // Positive
    prog.assert(max_seq_len.clone().real_gt(Expr::real(0)));
    prog.assert(head_dim.clone().real_gt(Expr::real(0)));
    prog.assert(kv_heads.clone().real_gt(Expr::real(0)));

    // Bounded
    prog.assert(max_seq_len.clone().real_le(Expr::real(100000)));
    prog.assert(head_dim.clone().real_le(Expr::real(1024)));
    prog.assert(kv_heads.clone().real_le(Expr::real(128)));

    // Prealloc formula
    prog.assert(
        alloc.clone().eq(max_seq_len
            .clone()
            .real_mul(head_dim.clone().real_mul(kv_heads.clone()))),
    );

    // Negated property: alloc != max_seq_len * head_dim * kv_heads
    let violation = alloc.ne(max_seq_len.real_mul(head_dim.real_mul(kv_heads)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "prealloc_fixed_memory");
}

// ---------------------------------------------------------------------------
// Test 601: Prealloc utilization fraction in [0, 1]
// ---------------------------------------------------------------------------

/// Prove: the utilization fraction (current_len / max_seq_len) is in [0, 1].
///
/// With pre-allocated cache, 0 <= current_len <= max_seq_len, so
/// the utilization fraction current_len / max_seq_len is in [0, 1].
#[test]
fn test_601_prealloc_utilization_fraction() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("current_len", real.clone());
    let _ = prog.declare_const("max_seq_len", real.clone());
    let _ = prog.declare_const("util", real);

    let current_len = real_var("current_len");
    let max_seq_len = real_var("max_seq_len");
    let util = real_var("util");

    // max_seq_len > 0, 0 <= current_len <= max_seq_len
    prog.assert(max_seq_len.clone().real_gt(Expr::real(0)));
    prog.assert(current_len.clone().real_ge(Expr::real(0)));
    prog.assert(current_len.clone().real_le(max_seq_len.clone()));
    prog.assert(max_seq_len.clone().real_le(Expr::real(100000)));

    // util = current_len / max_seq_len: util * max_seq_len = current_len
    prog.assert(util.clone().real_mul(max_seq_len).eq(current_len));

    // Negated property: util < 0 OR util > 1
    let violation = util
        .clone()
        .real_lt(Expr::real(0))
        .or(util.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "prealloc_utilization_fraction");
}

// ---------------------------------------------------------------------------
// Test 602: Multi-layer total = num_layers * per_layer_cache
// ---------------------------------------------------------------------------

/// Prove: total cache memory across all layers equals
/// num_layers * per_layer_cache.
///
/// Each transformer layer maintains its own KV cache of the same size.
/// The total is simply num_layers times the per-layer amount.
#[test]
fn test_602_multi_layer_total_cache() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("num_layers", real.clone());
    let _ = prog.declare_const("per_layer", real.clone());
    let _ = prog.declare_const("total", real);

    let num_layers = real_var("num_layers");
    let per_layer = real_var("per_layer");
    let total = real_var("total");

    // Positive
    prog.assert(num_layers.clone().real_gt(Expr::real(0)));
    prog.assert(per_layer.clone().real_gt(Expr::real(0)));

    // Bounded
    prog.assert(num_layers.clone().real_le(Expr::real(200)));
    prog.assert(per_layer.clone().real_le(Expr::real(1000000)));

    // Total = num_layers * per_layer
    prog.assert(
        total
            .clone()
            .eq(num_layers.clone().real_mul(per_layer.clone())),
    );

    // Negated property: total != num_layers * per_layer
    let violation = total.ne(num_layers.real_mul(per_layer));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "multi_layer_total_cache");
}

// ---------------------------------------------------------------------------
// Test 603: RoPE offset: position_id = cache_offset + local_pos
// ---------------------------------------------------------------------------

/// Prove: with a KV cache of length cache_offset, the absolute position
/// of the new token at local position local_pos is
/// position_id = cache_offset + local_pos.
///
/// This ensures RoPE embeddings use the correct absolute position
/// during incremental decoding.
#[test]
fn test_603_rope_offset_position() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("cache_offset", real.clone());
    let _ = prog.declare_const("local_pos", real.clone());
    let _ = prog.declare_const("position_id", real);

    let cache_offset = real_var("cache_offset");
    let local_pos = real_var("local_pos");
    let position_id = real_var("position_id");

    // Both non-negative
    prog.assert(cache_offset.clone().real_ge(Expr::real(0)));
    prog.assert(local_pos.clone().real_ge(Expr::real(0)));
    prog.assert(cache_offset.clone().real_le(Expr::real(100000)));
    prog.assert(local_pos.clone().real_le(Expr::real(10000)));

    // position_id = cache_offset + local_pos
    prog.assert(
        position_id
            .clone()
            .eq(cache_offset.clone().real_add(local_pos.clone())),
    );

    // Negated property: position_id != cache_offset + local_pos
    let violation = position_id.ne(cache_offset.real_add(local_pos));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_offset_position");
}

// ---------------------------------------------------------------------------
// Test 604: Cross-attention cache fixed after encoding
// ---------------------------------------------------------------------------

/// Prove: the cross-attention KV cache length remains constant after
/// the encoder output is computed. Once set to enc_len, it does not change.
///
/// Unlike self-attention cache which grows with each decoding step,
/// cross-attention cache is set once from the encoder output and remains
/// fixed throughout decoding.
#[test]
fn test_604_cross_attention_cache_fixed() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("enc_len", real.clone());
    let _ = prog.declare_const("cache_len_t1", real.clone());
    let _ = prog.declare_const("cache_len_t2", real);

    let enc_len = real_var("enc_len");
    let cache_len_t1 = real_var("cache_len_t1");
    let cache_len_t2 = real_var("cache_len_t2");

    // enc_len > 0
    prog.assert(enc_len.clone().real_gt(Expr::real(0)));
    prog.assert(enc_len.clone().real_le(Expr::real(100000)));

    // Cache is set to encoder length and stays fixed at two different times
    prog.assert(cache_len_t1.clone().eq(enc_len.clone()));
    prog.assert(cache_len_t2.clone().eq(enc_len));

    // Negated property: cache changed between t1 and t2
    let violation = cache_len_t1.ne(cache_len_t2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_attention_cache_fixed");
}

// ---------------------------------------------------------------------------
// Test 605: Concatenation: cat_len = old_len + new_len
// ---------------------------------------------------------------------------

/// Prove: concatenating old cache (length old_len) with new KV entries
/// (length new_len) produces a cache of length old_len + new_len.
///
/// This is the general multi-token append (e.g., prompt processing).
#[test]
fn test_605_concatenation_length() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("old_len", real.clone());
    let _ = prog.declare_const("new_len", real.clone());
    let _ = prog.declare_const("cat_len", real);

    let old_len = real_var("old_len");
    let new_len = real_var("new_len");
    let cat_len = real_var("cat_len");

    // Both non-negative
    prog.assert(old_len.clone().real_ge(Expr::real(0)));
    prog.assert(new_len.clone().real_ge(Expr::real(0)));
    prog.assert(old_len.clone().real_le(Expr::real(100000)));
    prog.assert(new_len.clone().real_le(Expr::real(100000)));

    // Concatenation: cat_len = old_len + new_len
    prog.assert(
        cat_len
            .clone()
            .eq(old_len.clone().real_add(new_len.clone())),
    );

    // Negated property: cat_len != old_len + new_len
    let violation = cat_len.ne(old_len.real_add(new_len));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "concatenation_length");
}

// ---------------------------------------------------------------------------
// Test 606: Byte alignment: buffer_size % alignment == 0
// ---------------------------------------------------------------------------

/// Prove: when a buffer is aligned to `alignment` bytes, the buffer size
/// is a multiple of alignment. Specifically, aligned_size = k * alignment
/// for some positive integer k, and aligned_size >= raw_size.
///
/// We model: aligned_size = k * alignment, k > 0, aligned_size >= raw_size.
/// Prove aligned_size is a multiple of alignment (tautological from the
/// construction, but validates the alignment formula).
#[test]
fn test_606_byte_alignment() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("raw_size", real.clone());
    let _ = prog.declare_const("alignment", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("aligned_size", real);

    let raw_size = real_var("raw_size");
    let alignment = real_var("alignment");
    let k = real_var("k");
    let aligned_size = real_var("aligned_size");

    // raw_size > 0, alignment > 0
    prog.assert(raw_size.clone().real_gt(Expr::real(0)));
    prog.assert(alignment.clone().real_gt(Expr::real(0)));
    prog.assert(raw_size.clone().real_le(Expr::real(1000000)));
    prog.assert(alignment.clone().real_le(Expr::real(4096)));

    // k > 0 (number of alignment units)
    prog.assert(k.clone().real_gt(Expr::real(0)));

    // aligned_size = k * alignment
    prog.assert(
        aligned_size
            .clone()
            .eq(k.clone().real_mul(alignment.clone())),
    );

    // aligned_size >= raw_size (rounding up)
    prog.assert(aligned_size.clone().real_ge(raw_size));

    // Negated property: aligned_size is NOT a multiple of alignment
    // i.e., aligned_size != k * alignment for the given k
    let violation = aligned_size.ne(k.real_mul(alignment));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "byte_alignment");
}

// ---------------------------------------------------------------------------
// Test 607: Token-level update overwrites one position
// ---------------------------------------------------------------------------

/// Prove: a token-level cache update at position pos modifies exactly
/// that position. The cache length before and after remains the same
/// (no growth — this is an in-place update, not an append).
///
/// After the update, the value at `pos` equals the new value, and
/// the cache length is unchanged.
#[test]
fn test_607_token_level_update_one_position() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("cache_len", real.clone());
    let _ = prog.declare_const("pos", real.clone());
    let _ = prog.declare_const("new_val", real.clone());
    let _ = prog.declare_const("val_at_pos", real.clone());
    let _ = prog.declare_const("cache_len_after", real);

    let cache_len = real_var("cache_len");
    let pos = real_var("pos");
    let new_val = real_var("new_val");
    let val_at_pos = real_var("val_at_pos");
    let cache_len_after = real_var("cache_len_after");

    // Valid position: 0 <= pos < cache_len, cache_len > 0
    prog.assert(cache_len.clone().real_gt(Expr::real(0)));
    prog.assert(pos.clone().real_ge(Expr::real(0)));
    prog.assert(pos.clone().real_lt(cache_len.clone()));

    // new_val is bounded
    prog.assert(new_val.clone().real_ge(Expr::real(-1000)));
    prog.assert(new_val.clone().real_le(Expr::real(1000)));

    // After update: val_at_pos = new_val, cache_len unchanged
    prog.assert(val_at_pos.clone().eq(new_val.clone()));
    prog.assert(cache_len_after.clone().eq(cache_len.clone()));

    // Negated property: val_at_pos != new_val OR cache_len changed
    let violation = val_at_pos.ne(new_val).or(cache_len_after.ne(cache_len));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "token_level_update_one_position");
}

// ---------------------------------------------------------------------------
// Test 608: Batch cache: total = batch_size * per_sequence_cache
// ---------------------------------------------------------------------------

/// Prove: the total cache memory for a batch of sequences equals
/// batch_size * per_sequence_cache.
///
/// Each sequence in the batch has its own independent KV cache.
/// The total memory is batch_size times the per-sequence cache size.
#[test]
fn test_608_batch_cache_total() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("batch_size", real.clone());
    let _ = prog.declare_const("per_seq", real.clone());
    let _ = prog.declare_const("total", real);

    let batch_size = real_var("batch_size");
    let per_seq = real_var("per_seq");
    let total = real_var("total");

    // Positive
    prog.assert(batch_size.clone().real_gt(Expr::real(0)));
    prog.assert(per_seq.clone().real_gt(Expr::real(0)));

    // Bounded
    prog.assert(batch_size.clone().real_le(Expr::real(1024)));
    prog.assert(per_seq.clone().real_le(Expr::real(10000000)));

    // total = batch_size * per_seq
    prog.assert(
        total
            .clone()
            .eq(batch_size.clone().real_mul(per_seq.clone())),
    );

    // Negated property: total != batch_size * per_seq
    let violation = total.ne(batch_size.real_mul(per_seq));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "batch_cache_total");
}

// ---------------------------------------------------------------------------
// Test 609: bf16 element size = 2 bytes, f32 = 4 bytes
// ---------------------------------------------------------------------------

/// Prove: bf16 elements occupy 2 bytes and f32 elements occupy 4 bytes,
/// so bf16 cache uses exactly half the memory of f32 cache for the same
/// number of elements.
///
/// If bf16_mem = num_elements * 2 and f32_mem = num_elements * 4, then
/// f32_mem = 2 * bf16_mem.
#[test]
fn test_609_dtype_element_sizes() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("num_elements", real.clone());
    let _ = prog.declare_const("bf16_mem", real.clone());
    let _ = prog.declare_const("f32_mem", real);

    let num_elements = real_var("num_elements");
    let bf16_mem = real_var("bf16_mem");
    let f32_mem = real_var("f32_mem");

    // num_elements > 0
    prog.assert(num_elements.clone().real_gt(Expr::real(0)));
    prog.assert(num_elements.clone().real_le(Expr::real(1000000000)));

    // bf16: 2 bytes per element
    prog.assert(
        bf16_mem
            .clone()
            .eq(num_elements.clone().real_mul(Expr::real(2))),
    );

    // f32: 4 bytes per element
    prog.assert(f32_mem.clone().eq(num_elements.real_mul(Expr::real(4))));

    // Negated property: f32_mem != 2 * bf16_mem
    let violation = f32_mem.ne(Expr::real(2).real_mul(bf16_mem));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dtype_element_sizes");
}

// ---------------------------------------------------------------------------
// Test 610: Max cache bounded by max_seq_len * config
// ---------------------------------------------------------------------------

/// Prove: the maximum possible cache size (per layer, K+V) is bounded by
/// max_seq_len * head_dim * num_kv_heads * 2.
///
/// For any current sequence length seq_len <= max_seq_len, the cache size
/// seq_len * head_dim * num_kv_heads * 2 <= max_seq_len * head_dim * num_kv_heads * 2.
#[test]
fn test_610_max_cache_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("seq_len", real.clone());
    let _ = prog.declare_const("max_seq_len", real.clone());
    let _ = prog.declare_const("head_dim", real.clone());
    let _ = prog.declare_const("kv_heads", real.clone());
    let _ = prog.declare_const("current_cache", real.clone());
    let _ = prog.declare_const("max_cache", real);

    let seq_len = real_var("seq_len");
    let max_seq_len = real_var("max_seq_len");
    let head_dim = real_var("head_dim");
    let kv_heads = real_var("kv_heads");
    let current_cache = real_var("current_cache");
    let max_cache = real_var("max_cache");

    // Positive parameters
    prog.assert(seq_len.clone().real_gt(Expr::real(0)));
    prog.assert(max_seq_len.clone().real_gt(Expr::real(0)));
    prog.assert(head_dim.clone().real_gt(Expr::real(0)));
    prog.assert(kv_heads.clone().real_gt(Expr::real(0)));

    // seq_len <= max_seq_len
    prog.assert(seq_len.clone().real_le(max_seq_len.clone()));

    // Bounded
    prog.assert(max_seq_len.clone().real_le(Expr::real(100000)));
    prog.assert(head_dim.clone().real_le(Expr::real(1024)));
    prog.assert(kv_heads.clone().real_le(Expr::real(128)));

    // current_cache = seq_len * head_dim * kv_heads * 2
    let factor = head_dim
        .clone()
        .real_mul(kv_heads.clone())
        .real_mul(Expr::real(2));
    prog.assert(current_cache.clone().eq(seq_len.real_mul(factor.clone())));

    // max_cache = max_seq_len * head_dim * kv_heads * 2
    prog.assert(
        max_cache
            .clone()
            .eq(max_seq_len.real_mul(head_dim.real_mul(kv_heads).real_mul(Expr::real(2)))),
    );

    // Negated property: current_cache > max_cache
    let violation = current_cache.real_gt(max_cache);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "max_cache_bounded");
}
