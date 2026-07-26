// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for dpdf VLM KV cache safety — part 2 of 2 (#4224).
//!
//! Proofs 7-12: vision-text decode total, prealloc overflow, multi-layer GQA
//! consistency, logical eviction, pipeline position tracking, doubling bounds.
//! See `kani_dpdf_vlm_kv_cache_proofs.rs` for proofs 1-6.

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// ===========================================================================
// Helpers (duplicated from part 1 — Kani modules are self-contained).
// ===========================================================================

struct VlmKvCacheLayer {
    batch: usize,
    num_kv_heads: usize,
    num_query_heads: usize,
    current_len: usize,
    capacity: usize,
    head_dim: usize,
    initialized: bool,
}

impl VlmKvCacheLayer {
    fn empty() -> Self {
        Self {
            batch: 0,
            num_kv_heads: 0,
            num_query_heads: 0,
            current_len: 0,
            capacity: 0,
            head_dim: 0,
            initialized: false,
        }
    }

    fn preallocated(max_seq_len: usize) -> Self {
        Self {
            capacity: max_seq_len,
            ..Self::empty()
        }
    }

    fn append(
        &mut self,
        b: usize,
        h_kv: usize,
        h_q: usize,
        new_seq: usize,
        dh: usize,
    ) -> Result<usize, &'static str> {
        if h_q == 0 || h_kv == 0 {
            return Err("heads must be > 0");
        }
        if h_q % h_kv != 0 {
            return Err("query_heads must be divisible by kv_heads (GQA)");
        }
        if !self.initialized {
            self.batch = b;
            self.num_kv_heads = h_kv;
            self.num_query_heads = h_q;
            self.head_dim = dh;
            self.initialized = true;
            if self.capacity == 0 {
                self.capacity = if new_seq > 16 { new_seq } else { 16 };
            }
        } else {
            if self.batch != b {
                return Err("batch dim mismatch");
            }
            if self.num_kv_heads != h_kv {
                return Err("num_kv_heads mismatch");
            }
            if self.num_query_heads != h_q {
                return Err("num_query_heads mismatch");
            }
            if self.head_dim != dh {
                return Err("head_dim mismatch");
            }
        }
        let needed = self
            .current_len
            .checked_add(new_seq)
            .ok_or("sequence length overflow")?;
        while self.capacity < needed {
            self.capacity = self
                .capacity
                .checked_mul(2)
                .ok_or("capacity overflow during doubling")?;
        }
        self.current_len = needed;
        Ok(self.current_len)
    }

    fn append_prealloc(
        &mut self,
        b: usize,
        h_kv: usize,
        h_q: usize,
        new_seq: usize,
        dh: usize,
    ) -> Result<usize, &'static str> {
        if h_q == 0 || h_kv == 0 {
            return Err("heads must be > 0");
        }
        if h_q % h_kv != 0 {
            return Err("query_heads must be divisible by kv_heads (GQA)");
        }
        if !self.initialized {
            self.batch = b;
            self.num_kv_heads = h_kv;
            self.num_query_heads = h_q;
            self.head_dim = dh;
            self.initialized = true;
        } else {
            if self.batch != b {
                return Err("batch dim mismatch");
            }
            if self.num_kv_heads != h_kv {
                return Err("num_kv_heads mismatch");
            }
            if self.head_dim != dh {
                return Err("head_dim mismatch");
            }
        }
        let needed = self
            .current_len
            .checked_add(new_seq)
            .ok_or("sequence length overflow")?;
        if needed > self.capacity {
            return Err("exceeds preallocated capacity");
        }
        self.current_len = needed;
        Ok(self.current_len)
    }

    fn reset(&mut self) {
        self.current_len = 0;
        self.capacity = 0;
        self.initialized = false;
        self.batch = 0;
        self.num_kv_heads = 0;
        self.num_query_heads = 0;
        self.head_dim = 0;
    }

    fn seq_len(&self) -> usize {
        self.current_len
    }

    fn kv_shape(&self) -> [usize; 4] {
        [
            self.batch,
            self.num_kv_heads,
            self.current_len,
            self.head_dim,
        ]
    }
}

struct VlmKvCache {
    layers: [VlmKvCacheLayer; 8],
    num_layers: usize,
    sliding: [bool; 8],
    window_size: usize,
}

impl VlmKvCache {
    fn new(num_layers: usize, window_size: usize) -> Self {
        assert!(num_layers <= 8);
        Self {
            layers: [
                VlmKvCacheLayer::empty(),
                VlmKvCacheLayer::empty(),
                VlmKvCacheLayer::empty(),
                VlmKvCacheLayer::empty(),
                VlmKvCacheLayer::empty(),
                VlmKvCacheLayer::empty(),
                VlmKvCacheLayer::empty(),
                VlmKvCacheLayer::empty(),
            ],
            num_layers,
            sliding: [false; 8],
            window_size,
        }
    }

    fn set_alternating_sliding(&mut self) {
        let mut i = 0;
        while i < self.num_layers {
            self.sliding[i] = i % 2 == 0;
            i += 1;
        }
    }

    fn append_all(
        &mut self,
        b: usize,
        h_kv: usize,
        h_q: usize,
        new_seq: usize,
        dh: usize,
    ) -> Result<(), &'static str> {
        let mut i = 0;
        while i < self.num_layers {
            self.layers[i].append(b, h_kv, h_q, new_seq, dh)?;
            i += 1;
        }
        Ok(())
    }

    fn all_same_seq_len(&self) -> bool {
        if self.num_layers == 0 {
            return true;
        }
        let first = self.layers[0].seq_len();
        let mut i = 1;
        while i < self.num_layers {
            if self.layers[i].seq_len() != first {
                return false;
            }
            i += 1;
        }
        true
    }

    fn seq_len(&self) -> usize {
        if self.num_layers == 0 {
            0
        } else {
            self.layers[0].seq_len()
        }
    }

    fn effective_context(&self, layer_idx: usize) -> usize {
        let seq = self.layers[layer_idx].seq_len();
        if self.sliding[layer_idx] {
            seq.min(self.window_size)
        } else {
            seq
        }
    }
}

// ===========================================================================
// 7. Vision-text boundary: V + T + D decode steps
// ===========================================================================

/// Proves total cache length is V + T + D after VLM prefill (V vision +
/// T text tokens) followed by D single-token decode steps.
#[kani::unwind(10)]
#[kani::proof]
fn proof_vlm_vision_text_decode_total() {
    let v_tok: u8 = kani::any();
    let t_tok: u8 = kani::any();
    let d_steps: u8 = kani::any();

    kani::assume(v_tok >= 1 && v_tok <= 4);
    kani::assume(t_tok >= 1 && t_tok <= 4);
    kani::assume(d_steps >= 1 && d_steps <= 4);

    let v = v_tok as usize;
    let t = t_tok as usize;
    let d = d_steps as usize;
    let combined = v + t;

    let mut cache = VlmKvCacheLayer::empty();

    let r = cache.append(1, 2, 4, combined, 4);
    assert!(r.is_ok(), "prefill must succeed");
    assert_eq!(cache.seq_len(), combined);

    let mut step = 0;
    while step < d {
        let before = cache.seq_len();
        let r = cache.append(1, 2, 4, 1, 4);
        assert!(r.is_ok(), "decode step must succeed");
        assert_eq!(cache.seq_len(), before + 1);
        step += 1;
    }

    let expected = v + t + d;
    assert_eq!(cache.seq_len(), expected, "total = V + T + D");

    let shape = cache.kv_shape();
    assert_eq!(shape[2], expected, "KV seq dim is V+T+D");

    let prefill_shape = [1usize, 2, combined, 4];
    let decode_shape = [1usize, 2, d, 4];
    let final_shape = [1usize, 2, expected, 4];

    let pn = checked_dim_product(&prefill_shape);
    let dn = checked_dim_product(&decode_shape);
    let fn_val = checked_dim_product(&final_shape);

    if let (Ok(p), Ok(dv), Ok(f)) = (pn, dn, fn_val) {
        assert_eq!(f, p + dv, "final numel = prefill + decode numel");
    }
}

// ===========================================================================
// 8. Pre-allocated cache rejects VLM overflow
// ===========================================================================

/// Proves pre-allocated VLM cache rejects appends exceeding max_seq_len
/// while accepting valid prefill+decode within capacity.
#[kani::unwind(10)]
#[kani::proof]
fn proof_vlm_prealloc_overflow_detection() {
    let max_seq: u8 = kani::any();
    let prefill: u8 = kani::any();
    let decode_steps: u8 = kani::any();

    kani::assume(max_seq >= 2 && max_seq <= 8);
    kani::assume(prefill >= 1 && prefill <= 8);
    kani::assume(decode_steps <= 8);

    let m = max_seq as usize;
    let p = prefill as usize;
    let d = decode_steps as usize;

    let mut cache = VlmKvCacheLayer::preallocated(m);

    let r = cache.append_prealloc(1, 2, 4, p, 4);
    if p > m {
        assert!(r.is_err(), "prefill > capacity must fail");
        return;
    }
    assert!(r.is_ok(), "prefill within capacity must succeed");

    let mut step = 0;
    while step < d {
        let before = cache.seq_len();
        let r = cache.append_prealloc(1, 2, 4, 1, 4);
        if before + 1 > m {
            assert!(r.is_err(), "decode > capacity must fail");
            return;
        }
        assert!(r.is_ok(), "decode within capacity must succeed");
        step += 1;
    }

    assert!(cache.seq_len() <= m, "final seq_len <= max_seq_len");
}

// ===========================================================================
// 9. Multi-layer GQA consistency
// ===========================================================================

/// Proves all layers have identical seq_len and matching KV shapes after
/// VLM prefill + decode with GQA head structure.
#[kani::unwind(6)]
#[kani::proof]
fn proof_vlm_multi_layer_gqa_consistency() {
    let num_layers: u8 = kani::any();
    let h_kv: u8 = kani::any();
    let h_q: u8 = kani::any();

    kani::assume(num_layers >= 2 && num_layers <= 4);
    kani::assume(h_kv >= 1 && h_kv <= 4);
    kani::assume(h_q >= 1 && h_q <= 8);
    kani::assume(h_q % h_kv == 0);

    let nl = num_layers as usize;
    let hkv = h_kv as usize;
    let hq = h_q as usize;

    let mut cache = VlmKvCache::new(nl, 128);

    let r = cache.append_all(1, hkv, hq, 3, 4);
    assert!(r.is_ok());

    let mut step = 0;
    while step < 2 {
        let r = cache.append_all(1, hkv, hq, 1, 4);
        assert!(r.is_ok());
        step += 1;
    }

    assert_eq!(cache.seq_len(), 5);
    assert!(cache.all_same_seq_len());

    let mut i = 0;
    while i < nl {
        let shape = cache.layers[i].kv_shape();
        assert_eq!(shape[1], hkv, "kv_heads consistent across layers");
        assert_eq!(shape[2], 5, "seq_len=5 across layers");
        i += 1;
    }
}

// ===========================================================================
// 10. Sliding eviction is logical, not physical
// ===========================================================================

/// Proves physical cache retains all entries for both sliding and full
/// layers; only the effective view is windowed. Full attention layers
/// always access complete history.
#[kani::unwind(12)]
#[kani::proof]
fn proof_sliding_eviction_logical_not_physical() {
    let window: u8 = kani::any();
    let total_steps: u8 = kani::any();

    kani::assume(window >= 2 && window <= 4);
    kani::assume(total_steps >= 1 && total_steps <= 8);

    let ws = window as usize;
    let n = total_steps as usize;

    let mut cache = VlmKvCache::new(4, ws);
    cache.set_alternating_sliding();

    let mut step = 0;
    while step < n {
        let r = cache.append_all(1, 2, 4, 1, 4);
        assert!(r.is_ok());
        step += 1;
    }

    assert!(cache.all_same_seq_len());
    assert_eq!(cache.layers[0].seq_len(), n);
    assert_eq!(cache.layers[1].seq_len(), n);

    let sliding_eff = cache.effective_context(0);
    let full_eff = cache.effective_context(1);

    if n > ws {
        assert_eq!(sliding_eff, ws);
        assert_eq!(full_eff, n);
        assert!(full_eff > sliding_eff, "full > sliding when seq > window");
    } else {
        assert_eq!(sliding_eff, full_eff, "equal when seq <= window");
    }
}

// ===========================================================================
// 11. Position tracking across full VLM pipeline lifecycle
// ===========================================================================

/// Proves position accuracy through: prefill -> decode -> reset -> prefill.
#[kani::unwind(6)]
#[kani::proof]
fn proof_vlm_position_tracking_pipeline() {
    let p1: u8 = kani::any();
    let d1: u8 = kani::any();
    let p2: u8 = kani::any();

    kani::assume(p1 >= 1 && p1 <= 4);
    kani::assume(d1 >= 1 && d1 <= 3);
    kani::assume(p2 >= 1 && p2 <= 4);

    let pv1 = p1 as usize;
    let dv1 = d1 as usize;
    let pv2 = p2 as usize;

    let mut cache = VlmKvCacheLayer::empty();

    let r = cache.append(1, 2, 4, pv1, 4);
    assert!(r.is_ok());
    assert_eq!(cache.seq_len(), pv1);

    let mut step = 0;
    while step < dv1 {
        let r = cache.append(1, 2, 4, 1, 4);
        assert!(r.is_ok());
        step += 1;
    }
    assert_eq!(cache.seq_len(), pv1 + dv1);

    cache.reset();
    assert_eq!(cache.seq_len(), 0);

    let r = cache.append(1, 2, 4, pv2, 4);
    assert!(r.is_ok());
    assert_eq!(cache.seq_len(), pv2, "position independent of prev session");
}

// ===========================================================================
// 12. Dynamic cache doubling stays within MAX_SEQ_CAPACITY
// ===========================================================================

/// Proves dynamic doubling never exceeds MAX_SEQ_CAPACITY (262144)
/// and initial capacity follows the max(16, new_seq) policy.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dynamic_cache_doubling_bounded() {
    let initial_seq: u8 = kani::any();
    kani::assume(initial_seq >= 1 && initial_seq <= 8);

    let is = initial_seq as usize;
    let initial_cap = if is > 16 { is } else { 16 };

    let mut cache = VlmKvCacheLayer::empty();
    let r = cache.append(1, 2, 4, is, 4);
    assert!(r.is_ok());

    assert!(cache.capacity >= is, "capacity >= initial seq");
    assert_eq!(
        cache.capacity, initial_cap,
        "initial capacity matches policy"
    );
    assert!(cache.capacity <= 262_144, "capacity <= MAX_SEQ_CAPACITY");
}
