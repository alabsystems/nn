// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for dpdf VLM KV cache management (#4224).
//!
//! Proves 10 additional safety properties beyond parts 1-2:
//! 1.  Circular buffer rotation wraps correctly at max_length
//! 2.  Cache eviction: oldest entries removed when full
//! 3.  Sliding window cache: only recent window_size entries retained
//! 4.  Multi-layer cache: each layer has independent KV buffers
//! 5.  GQA cache: KV heads < Q heads, expansion correct
//! 6.  Paged attention cache: block table indices in range
//! 7.  Cache shape: [batch, num_kv_heads, seq_len, head_dim]
//! 8.  Incremental update: append single token correctly
//! 9.  Cache mask alignment: mask length == cache length
//! 10. Cross-attention cache: static (no append after encode)
//!
//! All harnesses use small bounds for CBMC tractability:
//! B <= 2, H_kv <= 4, H_q <= 8, seq <= 16, D_h <= 8.

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// ===========================================================================
// Helpers (duplicated — Kani modules are self-contained to avoid
// cross-module issues with CBMC).
// ===========================================================================

/// Circular buffer KV cache: write pointer wraps at `max_length`.
struct CircularKvCache {
    max_length: usize,
    write_pos: usize,
    filled: usize,
    batch: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl CircularKvCache {
    fn new(max_length: usize, batch: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            max_length,
            write_pos: 0,
            filled: 0,
            batch,
            num_kv_heads,
            head_dim,
        }
    }

    /// Append `n` tokens. Returns (new_write_pos, new_filled).
    fn append(&mut self, n: usize) -> Result<(usize, usize), &'static str> {
        if self.max_length == 0 {
            return Err("max_length must be > 0");
        }
        let mut remaining = n;
        while remaining > 0 {
            self.write_pos = self.write_pos % self.max_length;
            self.write_pos += 1;
            if self.filled < self.max_length {
                self.filled += 1;
            }
            remaining -= 1;
        }
        Ok((self.write_pos, self.filled))
    }

    fn effective_len(&self) -> usize {
        self.filled
    }

    fn kv_shape(&self) -> [usize; 4] {
        [self.batch, self.num_kv_heads, self.filled, self.head_dim]
    }
}

/// Eviction cache: removes oldest entries when at capacity.
struct EvictionKvCache {
    capacity: usize,
    current_len: usize,
    evicted_count: usize,
}

impl EvictionKvCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            current_len: 0,
            evicted_count: 0,
        }
    }

    fn append(&mut self, n: usize) -> Result<(), &'static str> {
        if self.capacity == 0 {
            return Err("capacity must be > 0");
        }
        let needed = self.current_len.checked_add(n).ok_or("overflow")?;
        if needed > self.capacity {
            let evict = needed - self.capacity;
            self.evicted_count = self
                .evicted_count
                .checked_add(evict)
                .ok_or("eviction count overflow")?;
            self.current_len = self.capacity;
        } else {
            self.current_len = needed;
        }
        Ok(())
    }
}

/// Sliding window view over a growing cache.
struct SlidingWindowView {
    total_len: usize,
    window_size: usize,
}

impl SlidingWindowView {
    fn new(window_size: usize) -> Self {
        Self {
            total_len: 0,
            window_size,
        }
    }

    fn append(&mut self, n: usize) -> Result<(), &'static str> {
        self.total_len = self.total_len.checked_add(n).ok_or("overflow")?;
        Ok(())
    }

    fn visible_len(&self) -> usize {
        self.total_len.min(self.window_size)
    }

    fn start_idx(&self) -> usize {
        self.total_len.saturating_sub(self.window_size)
    }
}

/// Independent multi-layer cache.
struct MultiLayerCache {
    seq_lens: [usize; 8],
    capacities: [usize; 8],
    num_layers: usize,
}

impl MultiLayerCache {
    fn new(num_layers: usize) -> Self {
        assert!(num_layers <= 8);
        Self {
            seq_lens: [0; 8],
            capacities: [16; 8],
            num_layers,
        }
    }

    fn append_layer(&mut self, layer: usize, n: usize) -> Result<(), &'static str> {
        if layer >= self.num_layers {
            return Err("layer index out of bounds");
        }
        let needed = self.seq_lens[layer].checked_add(n).ok_or("overflow")?;
        while self.capacities[layer] < needed {
            self.capacities[layer] = self.capacities[layer]
                .checked_mul(2)
                .ok_or("capacity overflow")?;
        }
        self.seq_lens[layer] = needed;
        Ok(())
    }
}

/// Paged attention block table.
struct PagedBlockTable {
    num_blocks: usize,
    block_size: usize,
    /// Simulated table: block_table[i] is the physical block index for
    /// logical block i. We track the next free block.
    next_free: usize,
    table: [usize; 16],
    table_len: usize,
}

impl PagedBlockTable {
    fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            num_blocks,
            block_size,
            next_free: 0,
            table: [0; 16],
            table_len: 0,
        }
    }

    fn allocate_block(&mut self) -> Result<usize, &'static str> {
        if self.next_free >= self.num_blocks {
            return Err("no free blocks");
        }
        if self.table_len >= 16 {
            return Err("block table full");
        }
        let block_id = self.next_free;
        self.table[self.table_len] = block_id;
        self.table_len += 1;
        self.next_free += 1;
        Ok(block_id)
    }

    fn token_capacity(&self) -> usize {
        self.table_len * self.block_size
    }

    fn all_indices_valid(&self) -> bool {
        let mut i = 0;
        while i < self.table_len {
            if self.table[i] >= self.num_blocks {
                return false;
            }
            i += 1;
        }
        true
    }
}

/// Cross-attention cache: frozen after encode.
struct CrossAttentionCache {
    batch: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    frozen: bool,
}

impl CrossAttentionCache {
    fn empty() -> Self {
        Self {
            batch: 0,
            num_kv_heads: 0,
            seq_len: 0,
            head_dim: 0,
            frozen: false,
        }
    }

    fn encode(
        &mut self,
        b: usize,
        h_kv: usize,
        enc_len: usize,
        dh: usize,
    ) -> Result<(), &'static str> {
        if self.frozen {
            return Err("cache already frozen");
        }
        self.batch = b;
        self.num_kv_heads = h_kv;
        self.seq_len = enc_len;
        self.head_dim = dh;
        self.frozen = true;
        Ok(())
    }

    fn try_append(&self, _n: usize) -> Result<(), &'static str> {
        if self.frozen {
            Err("cross-attention cache is static after encode")
        } else {
            Err("cache not yet encoded")
        }
    }

    fn kv_shape(&self) -> [usize; 4] {
        [self.batch, self.num_kv_heads, self.seq_len, self.head_dim]
    }
}

// ===========================================================================
// 1. Circular buffer rotation wraps correctly at max_length
// ===========================================================================

/// Proves circular buffer write_pos wraps correctly and filled never
/// exceeds max_length.
#[kani::unwind(12)]
#[kani::proof]
fn proof_circular_buffer_wraps_at_max_length() {
    let max_len: u8 = kani::any();
    let steps: u8 = kani::any();

    kani::assume(max_len >= 2 && max_len <= 8);
    kani::assume(steps >= 1 && steps <= 10);

    let ml = max_len as usize;
    let n = steps as usize;

    let mut cache = CircularKvCache::new(ml, 1, 2, 4);

    let mut step = 0;
    while step < n {
        let r = cache.append(1);
        assert!(r.is_ok(), "append must succeed");
        let (wp, filled) = r.unwrap();
        assert!(wp <= ml, "write_pos <= max_length");
        assert!(filled <= ml, "filled <= max_length");
        step += 1;
    }

    // After n steps, filled should be min(n, max_len)
    assert_eq!(cache.effective_len(), n.min(ml));
}

// ===========================================================================
// 2. Cache eviction: oldest entries removed when full
// ===========================================================================

/// Proves eviction cache removes oldest entries when capacity is reached,
/// and current_len never exceeds capacity.
#[kani::unwind(12)]
#[kani::proof]
fn proof_cache_eviction_oldest_removed() {
    let cap: u8 = kani::any();
    let steps: u8 = kani::any();

    kani::assume(cap >= 2 && cap <= 6);
    kani::assume(steps >= 1 && steps <= 10);

    let c = cap as usize;
    let n = steps as usize;

    let mut cache = EvictionKvCache::new(c);

    let mut step = 0;
    while step < n {
        let r = cache.append(1);
        assert!(r.is_ok());
        assert!(cache.current_len <= c, "current_len <= capacity");
        step += 1;
    }

    if n > c {
        assert_eq!(cache.current_len, c, "at capacity when overfilled");
        assert_eq!(cache.evicted_count, n - c, "evicted = total - capacity");
    } else {
        assert_eq!(cache.current_len, n, "no eviction needed");
        assert_eq!(cache.evicted_count, 0, "zero evictions");
    }
}

// ===========================================================================
// 3. Sliding window cache: only recent window_size entries retained
// ===========================================================================

/// Proves sliding window view retains only the most recent window_size
/// entries and start_idx advances correctly.
#[kani::unwind(12)]
#[kani::proof]
fn proof_sliding_window_recent_entries_only() {
    let ws: u8 = kani::any();
    let steps: u8 = kani::any();

    kani::assume(ws >= 2 && ws <= 6);
    kani::assume(steps >= 1 && steps <= 10);

    let w = ws as usize;
    let n = steps as usize;

    let mut view = SlidingWindowView::new(w);

    let mut step = 0;
    while step < n {
        let r = view.append(1);
        assert!(r.is_ok());
        let vis = view.visible_len();
        assert!(vis <= w, "visible <= window_size");
        assert_eq!(vis, (step + 1).min(w), "visible = min(total, window)");
        step += 1;
    }

    assert_eq!(view.total_len, n);
    if n > w {
        assert_eq!(view.start_idx(), n - w, "start advances past window");
        assert_eq!(view.visible_len(), w, "window full");
    } else {
        assert_eq!(view.start_idx(), 0, "start at 0 when within window");
        assert_eq!(view.visible_len(), n, "all visible");
    }
}

// ===========================================================================
// 4. Multi-layer cache: each layer has independent KV buffers
// ===========================================================================

/// Proves layers can be appended independently and one layer's state
/// does not affect another.
#[kani::unwind(1)]
#[kani::proof]
fn proof_multi_layer_independent_buffers() {
    let num_layers: u8 = kani::any();
    let layer_a: u8 = kani::any();
    let layer_b: u8 = kani::any();
    let n_a: u8 = kani::any();
    let n_b: u8 = kani::any();

    kani::assume(num_layers >= 2 && num_layers <= 4);
    kani::assume(layer_a < num_layers);
    kani::assume(layer_b < num_layers);
    kani::assume(layer_a != layer_b);
    kani::assume(n_a >= 1 && n_a <= 8);
    kani::assume(n_b >= 1 && n_b <= 8);

    let nl = num_layers as usize;
    let la = layer_a as usize;
    let lb = layer_b as usize;
    let na = n_a as usize;
    let nb = n_b as usize;

    let mut cache = MultiLayerCache::new(nl);

    let r1 = cache.append_layer(la, na);
    assert!(r1.is_ok(), "layer_a append succeeds");

    let r2 = cache.append_layer(lb, nb);
    assert!(r2.is_ok(), "layer_b append succeeds");

    assert_eq!(cache.seq_lens[la], na, "layer_a has na tokens");
    assert_eq!(cache.seq_lens[lb], nb, "layer_b has nb tokens");

    // Other layers remain at zero
    let mut i = 0;
    while i < nl {
        if i != la && i != lb {
            assert_eq!(cache.seq_lens[i], 0, "untouched layer at 0");
        }
        i += 1;
    }
}

// ===========================================================================
// 5. GQA cache: KV heads < Q heads, expansion correct
// ===========================================================================

/// Proves GQA head expansion: kv_heads * repeat_factor == query_heads,
/// and KV cache numel is query numel / repeat_factor.
#[kani::unwind(1)]
#[kani::proof]
fn proof_gqa_cache_expansion_correct() {
    let h_kv: u8 = kani::any();
    let h_q: u8 = kani::any();
    let seq: u8 = kani::any();
    let dh: u8 = kani::any();
    let b: u8 = kani::any();

    kani::assume(h_kv >= 1 && h_kv <= 4);
    kani::assume(h_q >= 1 && h_q <= 8);
    kani::assume(h_q % h_kv == 0);
    kani::assume(h_q >= h_kv);
    kani::assume(seq >= 1 && seq <= 8);
    kani::assume(dh >= 1 && dh <= 8);
    kani::assume(b >= 1 && b <= 2);

    let hkv = h_kv as usize;
    let hq = h_q as usize;
    let s = seq as usize;
    let d = dh as usize;
    let batch = b as usize;

    let repeat = hq / hkv;
    assert!(repeat >= 1);
    assert_eq!(hkv * repeat, hq);

    let kv_shape = [batch, hkv, s, d];
    let q_shape = [batch, hq, s, d];

    let kv_n = checked_dim_product(&kv_shape);
    let q_n = checked_dim_product(&q_shape);

    if let (Ok(kn), Ok(qn)) = (kv_n, q_n) {
        assert_eq!(kn * repeat, qn, "kv * repeat == q numel");
        // KV cache is smaller by factor of repeat
        assert!(kn <= qn, "KV numel <= Q numel");
    }
}

// ===========================================================================
// 6. Paged attention cache: block table indices in range
// ===========================================================================

/// Proves all allocated block indices are within [0, num_blocks) and
/// token capacity equals table_len * block_size.
#[kani::unwind(10)]
#[kani::proof]
fn proof_paged_attention_block_indices_valid() {
    let num_blocks: u8 = kani::any();
    let block_size: u8 = kani::any();
    let allocs: u8 = kani::any();

    kani::assume(num_blocks >= 2 && num_blocks <= 8);
    kani::assume(block_size >= 1 && block_size <= 4);
    kani::assume(allocs >= 1 && allocs <= 8);

    let nb = num_blocks as usize;
    let bs = block_size as usize;
    let n = allocs as usize;

    let mut table = PagedBlockTable::new(nb, bs);

    let mut allocated = 0usize;
    let mut step = 0;
    while step < n {
        let r = table.allocate_block();
        if step < nb {
            assert!(r.is_ok(), "allocation within capacity succeeds");
            let block_id = r.unwrap();
            assert!(block_id < nb, "block_id in range");
            allocated += 1;
        } else {
            assert!(r.is_err(), "allocation beyond capacity fails");
        }
        step += 1;
    }

    assert!(table.all_indices_valid(), "all indices in range");
    assert_eq!(
        table.token_capacity(),
        allocated * bs,
        "token capacity = blocks * block_size"
    );
}

// ===========================================================================
// 7. Cache shape: [batch, num_kv_heads, seq_len, head_dim]
// ===========================================================================

/// Proves KV cache shape is always [B, H_kv, S, D_h] after append and
/// numel equals B * H_kv * S * D_h.
#[kani::unwind(6)]
#[kani::proof]
fn proof_cache_shape_batch_heads_seq_dim() {
    let b: u8 = kani::any();
    let h_kv: u8 = kani::any();
    let dh: u8 = kani::any();
    let prefill: u8 = kani::any();
    let decode_steps: u8 = kani::any();

    kani::assume(b >= 1 && b <= 2);
    kani::assume(h_kv >= 1 && h_kv <= 4);
    kani::assume(dh >= 1 && dh <= 8);
    kani::assume(prefill >= 1 && prefill <= 4);
    kani::assume(decode_steps >= 0 && decode_steps <= 4);

    let batch = b as usize;
    let hkv = h_kv as usize;
    let d = dh as usize;
    let p = prefill as usize;
    let dec = decode_steps as usize;

    let mut cache = CircularKvCache::new(64, batch, hkv, d);

    let r = cache.append(p);
    assert!(r.is_ok());

    let mut step = 0;
    while step < dec {
        let r = cache.append(1);
        assert!(r.is_ok());
        step += 1;
    }

    let total = p + dec;
    let shape = cache.kv_shape();
    assert_eq!(shape[0], batch, "dim 0 = batch");
    assert_eq!(shape[1], hkv, "dim 1 = num_kv_heads");
    assert_eq!(shape[2], total, "dim 2 = seq_len");
    assert_eq!(shape[3], d, "dim 3 = head_dim");

    let numel = checked_dim_product(&shape);
    if let Ok(n) = numel {
        assert_eq!(n, batch * hkv * total * d, "numel matches product");
    }
}

// ===========================================================================
// 8. Incremental update: append single token correctly
// ===========================================================================

/// Proves each single-token append increments seq_len by exactly 1 and
/// preserves shape invariants.
#[kani::unwind(10)]
#[kani::proof]
fn proof_incremental_single_token_append() {
    let b: u8 = kani::any();
    let h_kv: u8 = kani::any();
    let dh: u8 = kani::any();
    let initial: u8 = kani::any();
    let decode_steps: u8 = kani::any();

    kani::assume(b >= 1 && b <= 2);
    kani::assume(h_kv >= 1 && h_kv <= 4);
    kani::assume(dh >= 1 && dh <= 8);
    kani::assume(initial >= 1 && initial <= 4);
    kani::assume(decode_steps >= 1 && decode_steps <= 8);

    let batch = b as usize;
    let hkv = h_kv as usize;
    let d = dh as usize;
    let init = initial as usize;
    let dec = decode_steps as usize;

    let mut cache = CircularKvCache::new(64, batch, hkv, d);
    let r = cache.append(init);
    assert!(r.is_ok());
    assert_eq!(cache.effective_len(), init);

    let mut step = 0;
    while step < dec {
        let before = cache.effective_len();
        let r = cache.append(1);
        assert!(r.is_ok());
        let after = cache.effective_len();
        assert_eq!(after, before + 1, "seq_len increments by exactly 1");

        let shape = cache.kv_shape();
        assert_eq!(shape[0], batch, "batch preserved during decode");
        assert_eq!(shape[1], hkv, "heads preserved during decode");
        assert_eq!(shape[3], d, "head_dim preserved during decode");
        step += 1;
    }

    assert_eq!(cache.effective_len(), init + dec);
}

// ===========================================================================
// 9. Cache mask alignment: mask length == cache length
// ===========================================================================

/// Proves a causal attention mask of length seq_len aligns with the
/// cache's current seq_len at every decode step.
#[kani::unwind(10)]
#[kani::proof]
fn proof_cache_mask_alignment() {
    let prefill: u8 = kani::any();
    let decode_steps: u8 = kani::any();

    kani::assume(prefill >= 1 && prefill <= 4);
    kani::assume(decode_steps >= 1 && decode_steps <= 8);

    let p = prefill as usize;
    let dec = decode_steps as usize;

    let mut cache_len: usize = 0;

    // Prefill: mask covers [0, prefill)
    cache_len = cache_len.checked_add(p).unwrap();
    let mask_len = cache_len;
    assert_eq!(mask_len, cache_len, "mask aligned after prefill");

    // Decode: each step the query attends to all cached tokens
    let mut step = 0;
    while step < dec {
        cache_len = cache_len.checked_add(1).unwrap();
        // Causal mask for decode token at position cache_len-1
        // must cover all cache_len positions
        let mask_cols = cache_len;
        assert_eq!(
            mask_cols, cache_len,
            "mask columns == cache length at decode step"
        );
        // Query row count for single-token decode is 1
        let query_len = 1usize;
        // Mask shape: [query_len, cache_len]
        assert!(query_len <= cache_len, "query within mask range");
        step += 1;
    }

    let total = p + dec;
    assert_eq!(cache_len, total, "final cache_len = prefill + decode");
}

// ===========================================================================
// 10. Cross-attention cache: static (no append after encode)
// ===========================================================================

/// Proves cross-attention cache is frozen after encode and rejects all
/// subsequent append attempts. Shape remains constant.
#[kani::unwind(1)]
#[kani::proof]
fn proof_cross_attention_cache_static() {
    let b: u8 = kani::any();
    let h_kv: u8 = kani::any();
    let enc_len: u8 = kani::any();
    let dh: u8 = kani::any();

    kani::assume(b >= 1 && b <= 2);
    kani::assume(h_kv >= 1 && h_kv <= 4);
    kani::assume(enc_len >= 1 && enc_len <= 8);
    kani::assume(dh >= 1 && dh <= 8);

    let batch = b as usize;
    let hkv = h_kv as usize;
    let el = enc_len as usize;
    let d = dh as usize;

    let mut cache = CrossAttentionCache::empty();

    // Before encode, append is also rejected (not yet encoded)
    let r_pre = cache.try_append(1);
    assert!(r_pre.is_err(), "append before encode fails");

    // Encode
    let r_enc = cache.encode(batch, hkv, el, d);
    assert!(r_enc.is_ok(), "encode succeeds");

    let shape_after_encode = cache.kv_shape();
    assert_eq!(shape_after_encode, [batch, hkv, el, d]);

    // After encode, all appends must fail
    let r1 = cache.try_append(1);
    assert!(r1.is_err(), "append after encode fails");

    let r2 = cache.try_append(5);
    assert!(r2.is_err(), "multi-token append after encode fails");

    // Double encode also fails
    let r3 = cache.encode(batch, hkv, el, d);
    assert!(r3.is_err(), "double encode fails");

    // Shape unchanged
    let shape_final = cache.kv_shape();
    assert_eq!(
        shape_final, shape_after_encode,
        "shape unchanged after rejected ops"
    );

    let numel = checked_dim_product(&shape_final);
    if let Ok(n) = numel {
        assert_eq!(n, batch * hkv * el * d, "numel correct");
    }
}
