// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL kernel source for Flash Attention 2 (Tri Dao, arXiv:2307.08691).
//!
//! Tiled attention with online softmax — O(S_q * head_dim) memory instead of
//! O(S_q * S_kv). Each threadgroup processes Br=32 Q rows for one (batch, head)
//! pair, iterating over K/V in Bc=32-row blocks.
//!
//! Supports:
//! - **GQA**: Grouped-query attention via `group_size` constant. Query heads
//!   within a group share K/V heads: `kv_head = query_head / group_size`.
//! - **Causal masking**: Block-level tile skipping + per-element masking.
//!   Skips ~50% of K/V tiles for autoregressive decoding.
//!
//! Issue: #2434

/// Flash Attention 2 kernel — f32 variant with GQA and causal masking.
///
/// Buffer layout:
/// - buffer(0): Q  `[B*H_q, S_q, D]` (row-major, contiguous per head)
/// - buffer(1): K  `[B*H_kv, S_kv, D]`
/// - buffer(2): V  `[B*H_kv, S_kv, D]`
/// - buffer(3): O  `[B*H_q, S_q, D]` (output)
/// - buffer(4): S_q (uint constant)
/// - buffer(5): S_kv (uint constant)
/// - buffer(6): D (uint constant, head_dim)
/// - buffer(7): scale_bits (uint constant, bit-cast to float)
/// - buffer(8): H_q (uint constant, number of query heads)
/// - buffer(9): group_size (uint constant, H_q / H_kv; 1 for MHA)
/// - buffer(10): causal (uint constant, 0 or 1)
///
/// Grid: `[ceil(S_q / Br), B * H_q, 1]` threadgroups
/// Threads per threadgroup: `[Br, 1, 1]` (one thread per Q row)
///
/// Threadgroup memory: `Bc * MAX_D * sizeof(float)` bytes for K/V tile.
/// At D=128: 32 * 128 * 4 = 16,384 bytes.
pub(super) const FLASH_ATTN_F32_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Br = 32;        // Q block size (rows per threadgroup)
constant uint Bc = 32;        // K/V block size
constant uint MAX_D = 128;    // max head_dim for thread-local arrays

kernel void flash_attn_f32(
    device const float* Q            [[buffer(0)]],
    device const float* K            [[buffer(1)]],
    device const float* V            [[buffer(2)]],
    device float*       O            [[buffer(3)]],
    device const uint&  S_q_val      [[buffer(4)]],
    device const uint&  S_kv_val     [[buffer(5)]],
    device const uint&  D_val        [[buffer(6)]],
    device const uint&  scale_bits   [[buffer(7)]],
    device const uint&  H_q_val      [[buffer(8)]],
    device const uint&  group_size_val [[buffer(9)]],
    device const uint&  causal_val   [[buffer(10)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]]
) {
    uint bh_idx = tgid.y;       // flattened (batch, query_head) index
    uint q_block = tgid.x;      // which Q block
    uint S_q = S_q_val;
    uint S_kv = S_kv_val;
    uint D = D_val;
    // Pre-multiply scale by log2(e) so we can use exp2 (single HW instruction
    // on Apple GPUs) instead of exp in the online softmax loop. #1815 I1.
    float scale = as_type<float>(scale_bits) * M_LOG2E_F;
    uint H_q = H_q_val;
    uint group_size = group_size_val;
    uint causal = causal_val;

    // GQA: map query head to K/V head.
    // bh_idx = batch_idx * H_q + head_idx
    // kv_head = head_idx / group_size
    // kv_bh_idx = batch_idx * H_kv + kv_head
    uint head_idx = bh_idx % H_q;
    uint batch_idx = bh_idx / H_q;
    uint kv_head = head_idx / group_size;
    uint H_kv = H_q / group_size;
    uint kv_bh_idx = batch_idx * H_kv + kv_head;

    uint q_row = q_block * Br + tid;
    // IMPORTANT: Do NOT early-return for invalid Q rows. All threads must
    // participate in cooperative K/V tile loads and threadgroup barriers.
    // Use a validity flag instead.
    bool q_valid = (q_row < S_q);

    // Base offsets for this (batch, head) pair.
    // Q is addressed by bh_idx (query head), K/V by kv_bh_idx (KV head).
    uint safe_q_row = q_valid ? q_row : 0;
    uint q_base = bh_idx * S_q * D + safe_q_row * D;
    uint kv_base = kv_bh_idx * S_kv * D;

    // Thread-local: Q row, O accumulator, and attention scores.
    float q_local[MAX_D];
    float o_local[MAX_D];
    for (uint d = 0; d < D; d++) {
        q_local[d] = q_valid ? Q[q_base + d] : 0.0f;
        o_local[d] = 0.0f;
    }

    float m_val = -INFINITY;     // running row max
    float l_val = 0.0f;          // running softmax denominator

    // Shared memory for K or V tile: [Bc, D].
    threadgroup float kv_tile[Bc * MAX_D];

    uint num_kv_blocks = (S_kv + Bc - 1) / Bc;

    // Causal: last K/V block worth processing for this Q block.
    // For causal attention, position (q_row, k_col) is masked when k_col > q_row.
    // Block-level: skip K/V blocks where kv_start > q_block_last_row.
    uint q_block_last = q_block * Br + Br - 1;
    if (q_block_last >= S_q) q_block_last = S_q - 1;
    uint causal_kv_limit = causal ? (q_block_last / Bc + 1) : num_kv_blocks;
    uint effective_kv_blocks = min(num_kv_blocks, causal_kv_limit);

    for (uint kb = 0; kb < effective_kv_blocks; kb++) {
        uint kv_start = kb * Bc;

        // Attention weights (softmax output) — declared at outer scope so
        // they survive across the V tile loading barrier.
        float s_local[Bc];

        // ------- Phase 1: Load K block into shared memory -------
        // ALL threads participate in cooperative loading (including those
        // with invalid Q rows) to ensure the full tile is populated.
        uint k_row = kv_start + tid;
        if (k_row < S_kv) {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = K[kv_base + k_row * D + d];
            }
        } else {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Only threads with valid Q rows compute scores and softmax.
        if (q_valid) {
            // ------- Phase 2: Compute attention scores S = q @ K^T * scale -------
            for (uint j = 0; j < Bc; j++) {
                float dot = 0.0f;
                for (uint d = 0; d < D; d++) {
                    dot += q_local[d] * kv_tile[j * D + d];
                }
                s_local[j] = dot * scale;
            }

            // Mask invalid K positions (past S_kv boundary).
            uint valid_kv = min(Bc, S_kv - kv_start);
            for (uint j = valid_kv; j < Bc; j++) {
                s_local[j] = -INFINITY;
            }

            // Causal masking: mask positions where k_col > q_row.
            if (causal) {
                for (uint j = 0; j < Bc; j++) {
                    uint k_col = kv_start + j;
                    if (k_col > q_row) {
                        s_local[j] = -INFINITY;
                    }
                }
            }

            // ------- Phase 3: Online softmax -------
            float m_block = -INFINITY;
            for (uint j = 0; j < Bc; j++) {
                m_block = max(m_block, s_local[j]);
            }

            float m_new = max(m_val, m_block);
            float correction = exp2(m_val - m_new);

            float l_block = 0.0f;
            for (uint j = 0; j < Bc; j++) {
                s_local[j] = exp2(s_local[j] - m_new);
                l_block += s_local[j];
            }

            float l_new = correction * l_val + l_block;

            // Rescale previous O accumulator by correction factor.
            for (uint d = 0; d < D; d++) {
                o_local[d] *= correction;
            }

            m_val = m_new;
            l_val = l_new;
        }

        // ------- Phase 4: Load V block (ALL threads participate) -------
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint v_row = kv_start + tid;
        if (v_row < S_kv) {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = V[kv_base + v_row * D + d];
            }
        } else {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ------- Phase 5: Accumulate P @ V (only valid Q threads) -------
        if (q_valid) {
            for (uint j = 0; j < Bc; j++) {
                float w = s_local[j];
                for (uint d = 0; d < D; d++) {
                    o_local[d] += w * kv_tile[j * D + d];
                }
            }
        }
    }

    // ------- Final normalization: O = O_unnorm / l -------
    if (q_valid) {
        uint o_base = bh_idx * S_q * D + q_row * D;
        if (l_val > 0.0f) {
            float inv_l = 1.0f / l_val;
            for (uint d = 0; d < D; d++) {
                O[o_base + d] = o_local[d] * inv_l;
            }
        } else {
            // All keys masked for this query row — write deterministic zeros
            // instead of leaving uninitialized GPU memory. Part of #2218 F12.
            for (uint d = 0; d < D; d++) {
                O[o_base + d] = 0.0f;
            }
        }
    }
}
"#;

/// Flash Attention 2 kernel — f16 variant (half inputs, float accumulators).
///
/// Same algorithm as `flash_attn_f32` but reads `half` Q/K/V and writes `half`
/// output. Accumulation is in `float` for numerical stability during online
/// softmax. Threadgroup shared memory uses `half` — halves memory vs f32.
///
/// Buffer layout identical to f32 variant except buffers 0-3 are `half*`.
///
/// Supports BF16 inputs — BF16 is transparently stored as F16 on Metal
/// (see `MetalElement` impl for `half::bf16` in `element.rs`).
pub(super) const FLASH_ATTN_F16_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Br = 32;
constant uint Bc = 32;
constant uint MAX_D = 128;

kernel void flash_attn_f16(
    device const half*  Q            [[buffer(0)]],
    device const half*  K            [[buffer(1)]],
    device const half*  V            [[buffer(2)]],
    device half*        O            [[buffer(3)]],
    device const uint&  S_q_val      [[buffer(4)]],
    device const uint&  S_kv_val     [[buffer(5)]],
    device const uint&  D_val        [[buffer(6)]],
    device const uint&  scale_bits   [[buffer(7)]],
    device const uint&  H_q_val      [[buffer(8)]],
    device const uint&  group_size_val [[buffer(9)]],
    device const uint&  causal_val   [[buffer(10)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]]
) {
    uint bh_idx = tgid.y;
    uint q_block = tgid.x;
    uint S_q = S_q_val;
    uint S_kv = S_kv_val;
    uint D = D_val;
    // Pre-multiply scale by log2(e) for exp2 optimization. #1815 I1.
    float scale = as_type<float>(scale_bits) * M_LOG2E_F;
    uint H_q = H_q_val;
    uint group_size = group_size_val;
    uint causal = causal_val;

    // GQA head mapping (identical to f32 variant).
    uint head_idx = bh_idx % H_q;
    uint batch_idx = bh_idx / H_q;
    uint kv_head = head_idx / group_size;
    uint H_kv = H_q / group_size;
    uint kv_bh_idx = batch_idx * H_kv + kv_head;

    uint q_row = q_block * Br + tid;
    bool q_valid = (q_row < S_q);

    uint safe_q_row = q_valid ? q_row : 0;
    uint q_base = bh_idx * S_q * D + safe_q_row * D;
    uint kv_base = kv_bh_idx * S_kv * D;

    // Thread-local Q and O in float for accumulation precision.
    float q_local[MAX_D];
    float o_local[MAX_D];
    for (uint d = 0; d < D; d++) {
        q_local[d] = q_valid ? float(Q[q_base + d]) : 0.0f;
        o_local[d] = 0.0f;
    }

    float m_val = -INFINITY;
    float l_val = 0.0f;

    // Shared memory in half — halves threadgroup memory vs f32 variant.
    threadgroup half kv_tile[Bc * MAX_D];

    uint num_kv_blocks = (S_kv + Bc - 1) / Bc;

    uint q_block_last = q_block * Br + Br - 1;
    if (q_block_last >= S_q) q_block_last = S_q - 1;
    uint causal_kv_limit = causal ? (q_block_last / Bc + 1) : num_kv_blocks;
    uint effective_kv_blocks = min(num_kv_blocks, causal_kv_limit);

    for (uint kb = 0; kb < effective_kv_blocks; kb++) {
        uint kv_start = kb * Bc;
        float s_local[Bc];

        // Phase 1: Load K block into shared memory (half).
        uint k_row = kv_start + tid;
        if (k_row < S_kv) {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = K[kv_base + k_row * D + d];
            }
        } else {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (q_valid) {
            // Phase 2: Compute scores in float (upcast from half shared memory).
            for (uint j = 0; j < Bc; j++) {
                float dot = 0.0f;
                for (uint d = 0; d < D; d++) {
                    dot += q_local[d] * float(kv_tile[j * D + d]);
                }
                s_local[j] = dot * scale;
            }

            uint valid_kv = min(Bc, S_kv - kv_start);
            for (uint j = valid_kv; j < Bc; j++) {
                s_local[j] = -INFINITY;
            }

            if (causal) {
                for (uint j = 0; j < Bc; j++) {
                    uint k_col = kv_start + j;
                    if (k_col > q_row) {
                        s_local[j] = -INFINITY;
                    }
                }
            }

            // Phase 3: Online softmax (in float, using exp2 — #1815 I1).
            float m_block = -INFINITY;
            for (uint j = 0; j < Bc; j++) {
                m_block = max(m_block, s_local[j]);
            }

            float m_new = max(m_val, m_block);
            float correction = exp2(m_val - m_new);

            float l_block = 0.0f;
            for (uint j = 0; j < Bc; j++) {
                s_local[j] = exp2(s_local[j] - m_new);
                l_block += s_local[j];
            }

            float l_new = correction * l_val + l_block;

            for (uint d = 0; d < D; d++) {
                o_local[d] *= correction;
            }

            m_val = m_new;
            l_val = l_new;
        }

        // Phase 4: Load V block (half shared memory).
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint v_row = kv_start + tid;
        if (v_row < S_kv) {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = V[kv_base + v_row * D + d];
            }
        } else {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 5: Accumulate P @ V in float (upcast from half V tile).
        if (q_valid) {
            for (uint j = 0; j < Bc; j++) {
                float w = s_local[j];
                for (uint d = 0; d < D; d++) {
                    o_local[d] += w * float(kv_tile[j * D + d]);
                }
            }
        }
    }

    // Final normalization: write half output.
    if (q_valid) {
        uint o_base = bh_idx * S_q * D + q_row * D;
        if (l_val > 0.0f) {
            float inv_l = 1.0f / l_val;
            for (uint d = 0; d < D; d++) {
                O[o_base + d] = half(o_local[d] * inv_l);
            }
        } else {
            // All keys masked — write deterministic zeros. Part of #2218 F12.
            for (uint d = 0; d < D; d++) {
                O[o_base + d] = half(0.0f);
            }
        }
    }
}
"#;
