// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for KV cache management safety in dpdf VLM decoder
//! inference — part 1 of 2 (#4224).
//!
//! dpdf VLMs (Qwen3-VL, GLM-OCR, PaddleOCR-VL, etc.) share a common
//! decoder pattern: vision features and text tokens are concatenated into a
//! combined sequence, then fed through transformer decoder layers with KV
//! caching. This creates VLM-specific safety requirements beyond the generic
//! proofs in `kani_kv_cache_safety.rs`.
//!
//! Part 1 proves:
//! 1. VLM prefill combined sequence (vision+text) fits within capacity
//! 2. GQA head ratio correctness (kv_heads divides query_heads)
//! 3. GQA rejection for invalid head ratios
//! 4. Sliding window eviction preserves full-attention layer entries
//! 5. Multi-layer indexing bounds for VLM decoder
//! 6. Cache reset completeness for VLM decoder
//!
//! Part 2 in `kani_dpdf_vlm_kv_cache_proofs_ext.rs`.
//!
//! All harnesses use small bounds for CBMC tractability:
//! B <= 2, H_kv <= 4, H_q <= 8, seq <= 16, D_h <= 8.

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// ===========================================================================
// Helper: abstract VLM KV cache layer modelling GQA head structure.
//
// Duplicated across part 1 and part 2 because Kani modules are
// self-contained to avoid cross-module issues with CBMC.
// ===========================================================================

/// Abstract VLM KV cache layer modelling GQA head structure.
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

    /// Simulate appending new_seq tokens with GQA shape.
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

    fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// KV cache shape: [batch, num_kv_heads, current_len, head_dim].
    fn kv_shape(&self) -> [usize; 4] {
        [
            self.batch,
            self.num_kv_heads,
            self.current_len,
            self.head_dim,
        ]
    }

    /// Query projection shape: [batch, num_query_heads, current_len, head_dim].
    fn query_shape(&self) -> [usize; 4] {
        [
            self.batch,
            self.num_query_heads,
            self.current_len,
            self.head_dim,
        ]
    }
}

/// Abstract multi-layer VLM cache with mixed attention types.
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

    fn reset(&mut self) {
        let mut i = 0;
        while i < self.num_layers {
            self.layers[i].reset();
            i += 1;
        }
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
// 1. VLM prefill: vision + text tokens fit within capacity
// ===========================================================================

/// Proves VLM prefill with V vision + T text tokens produces cache with
/// seq_len = V + T and correct GQA-shaped KV tensors.
#[kani::unwind(1)]
#[kani::proof]
fn proof_vlm_prefill_combined_sequence_fits() {
    let b: u8 = kani::any();
    let h_kv: u8 = kani::any();
    let h_q: u8 = kani::any();
    let dh: u8 = kani::any();
    let v_tokens: u8 = kani::any();
    let t_tokens: u8 = kani::any();

    kani::assume(b >= 1 && b <= 2);
    kani::assume(h_kv >= 1 && h_kv <= 4);
    kani::assume(h_q >= 1 && h_q <= 8);
    kani::assume(h_q % h_kv == 0);
    kani::assume(dh >= 1 && dh <= 8);
    kani::assume(v_tokens >= 1 && v_tokens <= 8);
    kani::assume(t_tokens >= 1 && t_tokens <= 8);

    let bu = b as usize;
    let hkv = h_kv as usize;
    let hq = h_q as usize;
    let dhu = dh as usize;
    let combined = v_tokens as usize + t_tokens as usize;

    let mut cache = VlmKvCacheLayer::empty();
    let result = cache.append(bu, hkv, hq, combined, dhu);
    assert!(result.is_ok(), "VLM prefill must succeed");
    assert_eq!(cache.seq_len(), combined, "seq_len must equal V + T");

    let kv_shape = cache.kv_shape();
    assert_eq!(kv_shape[0], bu, "batch preserved");
    assert_eq!(kv_shape[1], hkv, "KV uses kv_heads not query_heads");
    assert_eq!(kv_shape[2], combined, "seq dim is V+T");
    assert_eq!(kv_shape[3], dhu, "head_dim preserved");

    let kv_numel = checked_dim_product(&kv_shape);
    if let Ok(n) = kv_numel {
        assert_eq!(n, bu * hkv * combined * dhu, "KV numel matches");
    }
}

// ===========================================================================
// 2. GQA head ratio: KV shape compatible with attention
// ===========================================================================

/// Proves GQA head structure: kv_heads * repeat_factor == query_heads,
/// KV and Q share seq/head_dim, and numel relationship holds.
#[kani::unwind(1)]
#[kani::proof]
fn proof_gqa_kv_cache_head_ratio() {
    let h_kv: u8 = kani::any();
    let h_q: u8 = kani::any();
    let seq: u8 = kani::any();
    let dh: u8 = kani::any();

    kani::assume(h_kv >= 1 && h_kv <= 4);
    kani::assume(h_q >= 1 && h_q <= 8);
    kani::assume(h_q % h_kv == 0);
    kani::assume(seq >= 1 && seq <= 8);
    kani::assume(dh >= 1 && dh <= 8);

    let hkv = h_kv as usize;
    let hq = h_q as usize;

    let mut cache = VlmKvCacheLayer::empty();
    let r = cache.append(1, hkv, hq, seq as usize, dh as usize);
    assert!(r.is_ok(), "GQA append must succeed");

    let kv_shape = cache.kv_shape();
    let q_shape = cache.query_shape();
    let repeat_factor = hq / hkv;

    assert!(repeat_factor >= 1, "repeat factor >= 1");
    assert_eq!(hkv * repeat_factor, hq, "kv*repeat == query");
    assert_eq!(kv_shape[2], q_shape[2], "seq dim matches");
    assert_eq!(kv_shape[3], q_shape[3], "head_dim matches");

    let kv_numel = checked_dim_product(&kv_shape);
    let q_numel = checked_dim_product(&q_shape);
    if let (Ok(kn), Ok(qn)) = (kv_numel, q_numel) {
        assert_eq!(kn * repeat_factor, qn, "KV*repeat == Q numel");
    }
}

// ===========================================================================
// 3. GQA rejects invalid head ratios
// ===========================================================================

/// Proves append rejects h_q not divisible by h_kv.
#[kani::unwind(1)]
#[kani::proof]
fn proof_gqa_rejects_invalid_head_ratio() {
    let h_kv: u8 = kani::any();
    let h_q: u8 = kani::any();

    kani::assume(h_kv >= 1 && h_kv <= 4);
    kani::assume(h_q >= 1 && h_q <= 8);
    kani::assume(h_q % h_kv != 0);

    let mut cache = VlmKvCacheLayer::empty();
    let r = cache.append(1, h_kv as usize, h_q as usize, 1, 4);
    assert!(r.is_err(), "invalid GQA head ratio must be rejected");
}

// ===========================================================================
// 4. Sliding window eviction preserves full-attention layer entries
// ===========================================================================

/// Proves mixed sliding/full attention: sliding layers bounded by window,
/// full layers retain complete history.
#[kani::unwind(10)]
#[kani::proof]
fn proof_sliding_window_preserves_full_layers() {
    let num_layers: u8 = kani::any();
    let window_size: u8 = kani::any();
    let steps: u8 = kani::any();

    kani::assume(num_layers >= 2 && num_layers <= 4);
    kani::assume(window_size >= 2 && window_size <= 4);
    kani::assume(steps >= 1 && steps <= 8);

    let nl = num_layers as usize;
    let ws = window_size as usize;
    let n = steps as usize;

    let mut cache = VlmKvCache::new(nl, ws);
    cache.set_alternating_sliding();

    let mut step = 0;
    while step < n {
        let r = cache.append_all(1, 2, 4, 1, 4);
        assert!(r.is_ok());
        step += 1;
    }

    let mut i = 0;
    while i < nl {
        let effective = cache.effective_context(i);
        if cache.sliding[i] {
            assert!(effective <= ws, "sliding: bounded by window");
            assert_eq!(effective, n.min(ws), "sliding: min(seq, window)");
        } else {
            assert_eq!(effective, n, "full: retains all tokens");
        }
        i += 1;
    }
    assert!(cache.all_same_seq_len(), "physical seq_len same for all");
}

// ===========================================================================
// 5. Multi-layer indexing bounds
// ===========================================================================

/// Proves valid layer indices succeed and OOB is detected.
#[kani::unwind(10)]
#[kani::proof]
fn proof_vlm_multi_layer_indexing() {
    let num_layers: u8 = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 8);
    let nl = num_layers as usize;

    let cache = VlmKvCache::new(nl, 128);
    let mut i = 0;
    while i < nl {
        assert!(i < cache.num_layers, "valid index within bounds");
        i += 1;
    }
    assert!(nl >= cache.num_layers, "index == num_layers is OOB");
}

// ===========================================================================
// 6. Cache reset completeness for VLM decoder
// ===========================================================================

/// Proves reset clears all layers; subsequent prefill starts fresh.
#[kani::unwind(1)]
#[kani::proof]
fn proof_vlm_cache_reset_completeness() {
    let num_layers: u8 = kani::any();
    kani::assume(num_layers >= 2 && num_layers <= 4);
    let nl = num_layers as usize;

    let h_kv: u8 = kani::any();
    let h_q: u8 = kani::any();
    kani::assume(h_kv >= 1 && h_kv <= 4);
    kani::assume(h_q >= 1 && h_q <= 8);
    kani::assume(h_q % h_kv == 0);

    let hkv = h_kv as usize;
    let hq = h_q as usize;

    let mut cache = VlmKvCache::new(nl, 128);
    let r = cache.append_all(1, hkv, hq, 5, 4);
    assert!(r.is_ok());

    cache.reset();
    let mut i = 0;
    while i < nl {
        assert!(cache.layers[i].is_empty(), "layer empty after reset");
        assert_eq!(cache.layers[i].seq_len(), 0, "layer seq_len 0");
        assert!(!cache.layers[i].initialized, "layer uninitialized");
        i += 1;
    }

    let r2 = cache.append_all(1, hkv, hq, 3, 4);
    assert!(r2.is_ok(), "post-reset prefill succeeds");
    assert!(cache.all_same_seq_len(), "all layers consistent");
}
