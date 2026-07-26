// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL kernel source for Flash Attention 2 — sequence-first layout variant.
//!
//! Input layout: `[B, S, H, D]` (sequence-first) instead of `[B, H, S, D]`.
//! Eliminates 4 Transpose dispatches per attention block by reading Q/K/V
//! directly in their natural layout via stride-based addressing.
//!
//! Same Flash Attention 2 algorithm (Tri Dao, arXiv:2307.08691) and buffer
//! bindings as the heads-first variant. Only the global memory addressing
//! changes — threadgroup shared memory tiles remain contiguous `[Bc, D]`.
//!
//! Part of #1815 Tier 5 D1 and #3088 (attention transpose elimination).

/// Flash Attention 2 — f32 SeqFirst variant.
///
/// Buffer layout (same bindings as heads-first):
/// - buffer(0): Q  `[B, S_q, H_q, D]` (row-major, sequence-first)
/// - buffer(1): K  `[B, S_kv, H_kv, D]`
/// - buffer(2): V  `[B, S_kv, H_kv, D]`
/// - buffer(3): O  `[B, S_q, H_q, D]` (output)
/// - buffer(4..10): constants (S_q, S_kv, D, scale_bits, H_q, group_size, causal)
///
/// Grid: `[ceil(S_q / Br), B * H_q, 1]` threadgroups
///
/// Stride difference from heads-first:
/// - Heads-first `[B*H, S, D]`: row stride = D (contiguous per head)
/// - SeqFirst `[B, S, H, D]`: row stride = H*D (interleaved heads)
pub(super) const FLASH_ATTN_F32_SEQ_FIRST_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Br = 32;        // Q block size (rows per threadgroup)
constant uint Bc = 32;        // K/V block size
constant uint MAX_D = 128;    // max head_dim for thread-local arrays

kernel void flash_attn_f32_seq_first(
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
    // Pre-multiply scale by log2(e) for exp2 optimization. #1815 I1.
    float scale = as_type<float>(scale_bits) * M_LOG2E_F;
    uint H_q = H_q_val;
    uint group_size = group_size_val;
    uint causal = causal_val;

    // GQA: map query head to K/V head.
    uint head_idx = bh_idx % H_q;
    uint batch_idx = bh_idx / H_q;
    uint kv_head = head_idx / group_size;
    uint H_kv = H_q / group_size;

    // SeqFirst strides: row stride = H * D (skip all heads at one position).
    uint q_row_stride = H_q * D;
    uint kv_row_stride = H_kv * D;

    uint q_row = q_block * Br + tid;
    bool q_valid = (q_row < S_q);

    // SeqFirst base: Q[batch, seq, head, d] = batch*S_q*H_q*D + seq*H_q*D + head*D + d
    uint safe_q_row = q_valid ? q_row : 0;
    uint q_batch_head = batch_idx * S_q * q_row_stride + head_idx * D;
    uint q_base = q_batch_head + safe_q_row * q_row_stride;
    uint kv_batch_head = batch_idx * S_kv * kv_row_stride + kv_head * D;

    // Thread-local: Q row, O accumulator.
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
    uint q_block_last = q_block * Br + Br - 1;
    if (q_block_last >= S_q) q_block_last = S_q - 1;
    uint causal_kv_limit = causal ? (q_block_last / Bc + 1) : num_kv_blocks;
    uint effective_kv_blocks = min(num_kv_blocks, causal_kv_limit);

    for (uint kb = 0; kb < effective_kv_blocks; kb++) {
        uint kv_start = kb * Bc;

        float s_local[Bc];

        // ------- Phase 1: Load K block into shared memory -------
        // Strided read: K[batch, kv_start+tid, kv_head, d]
        uint k_row = kv_start + tid;
        if (k_row < S_kv) {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = K[kv_batch_head + k_row * kv_row_stride + d];
            }
        } else {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (q_valid) {
            // ------- Phase 2: Compute attention scores S = q @ K^T * scale -------
            for (uint j = 0; j < Bc; j++) {
                float dot = 0.0f;
                for (uint d = 0; d < D; d++) {
                    dot += q_local[d] * kv_tile[j * D + d];
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

            // ------- Phase 3: Online softmax (using exp2 — #1815 I1) -------
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

        // ------- Phase 4: Load V block (strided read) -------
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint v_row = kv_start + tid;
        if (v_row < S_kv) {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = V[kv_batch_head + v_row * kv_row_stride + d];
            }
        } else {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ------- Phase 5: Accumulate P @ V -------
        if (q_valid) {
            for (uint j = 0; j < Bc; j++) {
                float w = s_local[j];
                for (uint d = 0; d < D; d++) {
                    o_local[d] += w * kv_tile[j * D + d];
                }
            }
        }
    }

    // ------- Final normalization: strided write to O -------
    if (q_valid) {
        uint o_base = q_batch_head + q_row * q_row_stride;
        if (l_val > 0.0f) {
            float inv_l = 1.0f / l_val;
            for (uint d = 0; d < D; d++) {
                O[o_base + d] = o_local[d] * inv_l;
            }
        } else {
            for (uint d = 0; d < D; d++) {
                O[o_base + d] = 0.0f;
            }
        }
    }
}
"#;

/// Flash Attention 2 — f16 SeqFirst variant (half inputs, float accumulators).
///
/// Same stride-based addressing as f32 SeqFirst. Reads `half` Q/K/V,
/// accumulates in `float`, writes `half` output. Threadgroup memory is `half`.
pub(super) const FLASH_ATTN_F16_SEQ_FIRST_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Br = 32;
constant uint Bc = 32;
constant uint MAX_D = 128;

kernel void flash_attn_f16_seq_first(
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

    uint head_idx = bh_idx % H_q;
    uint batch_idx = bh_idx / H_q;
    uint kv_head = head_idx / group_size;
    uint H_kv = H_q / group_size;

    // SeqFirst strides.
    uint q_row_stride = H_q * D;
    uint kv_row_stride = H_kv * D;

    uint q_row = q_block * Br + tid;
    bool q_valid = (q_row < S_q);

    uint safe_q_row = q_valid ? q_row : 0;
    uint q_batch_head = batch_idx * S_q * q_row_stride + head_idx * D;
    uint q_base = q_batch_head + safe_q_row * q_row_stride;
    uint kv_batch_head = batch_idx * S_kv * kv_row_stride + kv_head * D;

    // Thread-local Q and O in float for accumulation precision.
    float q_local[MAX_D];
    float o_local[MAX_D];
    for (uint d = 0; d < D; d++) {
        q_local[d] = q_valid ? float(Q[q_base + d]) : 0.0f;
        o_local[d] = 0.0f;
    }

    float m_val = -INFINITY;
    float l_val = 0.0f;

    threadgroup half kv_tile[Bc * MAX_D];

    uint num_kv_blocks = (S_kv + Bc - 1) / Bc;

    uint q_block_last = q_block * Br + Br - 1;
    if (q_block_last >= S_q) q_block_last = S_q - 1;
    uint causal_kv_limit = causal ? (q_block_last / Bc + 1) : num_kv_blocks;
    uint effective_kv_blocks = min(num_kv_blocks, causal_kv_limit);

    for (uint kb = 0; kb < effective_kv_blocks; kb++) {
        uint kv_start = kb * Bc;
        float s_local[Bc];

        // Phase 1: Load K block (strided, half).
        uint k_row = kv_start + tid;
        if (k_row < S_kv) {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = K[kv_batch_head + k_row * kv_row_stride + d];
            }
        } else {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (q_valid) {
            // Phase 2: Scores (upcast from half shared memory).
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

            // Phase 3: Online softmax (float, using exp2 — #1815 I1).
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

        // Phase 4: Load V block (strided, half).
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint v_row = kv_start + tid;
        if (v_row < S_kv) {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = V[kv_batch_head + v_row * kv_row_stride + d];
            }
        } else {
            for (uint d = 0; d < D; d++) {
                kv_tile[tid * D + d] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 5: Accumulate P @ V (upcast from half V tile).
        if (q_valid) {
            for (uint j = 0; j < Bc; j++) {
                float w = s_local[j];
                for (uint d = 0; d < D; d++) {
                    o_local[d] += w * float(kv_tile[j * D + d]);
                }
            }
        }
    }

    // Final normalization: write half output (strided).
    if (q_valid) {
        uint o_base = q_batch_head + q_row * q_row_stride;
        if (l_val > 0.0f) {
            float inv_l = 1.0f / l_val;
            for (uint d = 0; d < D; d++) {
                O[o_base + d] = half(o_local[d] * inv_l);
            }
        } else {
            for (uint d = 0; d < D; d++) {
                O[o_base + d] = half(0.0f);
            }
        }
    }
}
"#;
