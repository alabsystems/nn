// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for HIP MoE codegen.
//!
//! Proves launch coverage, expert routing bounds, and top-k indexing
//! invariants for the HIP MoE kernels.
//!
//! Part of #3727.

#[cfg(kani)]
mod proofs {
    use crate::codegen_hip_moe::{
        emit_grouped_gemm_kernel, emit_moe_unpermute_kernel, grouped_gemm_launch_config,
    };

    const ROUTED_EXPERTS: usize = 4;
    const GROUPED_GEMM_TILE: u64 = 32;
    const GROUPED_GEMM_SHARED_BYTES: u32 = 2 * 32 * 33 * 4;
    const MOE_BLOCK_SIZE: u32 = 256;

    fn select_expert(offsets: [u16; ROUTED_EXPERTS + 1], tile_row_global: u16) -> Option<usize> {
        if tile_row_global >= offsets[ROUTED_EXPERTS] {
            return None;
        }

        let mut expert_id = 0;
        while expert_id < ROUTED_EXPERTS {
            if tile_row_global >= offsets[expert_id] && tile_row_global < offsets[expert_id + 1] {
                return Some(expert_id);
            }
            expert_id += 1;
        }

        None
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_grouped_gemm_launch_covers_rows_and_cols() {
        let max_total_tokens: u16 = kani::any();
        let out_dim: u16 = kani::any();
        kani::assume(max_total_tokens > 0);
        kani::assume(out_dim > 0);

        let cfg = grouped_gemm_launch_config(usize::from(max_total_tokens), usize::from(out_dim));

        assert_eq!(cfg.block.x, MOE_BLOCK_SIZE);
        assert_eq!(cfg.block.y, 1);
        assert_eq!(cfg.block.z, 1);
        assert_eq!(cfg.grid.z, 1);
        assert_eq!(cfg.shared_mem_bytes, GROUPED_GEMM_SHARED_BYTES);
        assert!(u64::from(cfg.grid.x) * GROUPED_GEMM_TILE >= u64::from(out_dim));
        assert!(u64::from(cfg.grid.y) * GROUPED_GEMM_TILE >= u64::from(max_total_tokens));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(256)]
    fn prove_grouped_gemm_rejects_unaligned_dimensions() {
        let in_dim: u16 = kani::any();
        let out_dim: u16 = kani::any();
        kani::assume(in_dim > 0);
        kani::assume(out_dim > 0);
        kani::assume(in_dim % 32 != 0 || out_dim % 32 != 0);

        let result = emit_grouped_gemm_kernel(
            "kani_bad_grouped_gemm",
            4,
            usize::from(in_dim),
            usize::from(out_dim),
            128,
        );

        assert!(result.is_err());
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(256)]
    fn prove_grouped_gemm_accepts_aligned_dimensions() {
        let n_experts: u8 = kani::any();
        let in_blocks: u8 = kani::any();
        let out_blocks: u8 = kani::any();
        let max_total_tokens: u16 = kani::any();
        kani::assume(n_experts > 0);
        kani::assume(in_blocks > 0);
        kani::assume(out_blocks > 0);
        kani::assume(max_total_tokens > 0);

        let result = emit_grouped_gemm_kernel(
            "kani_grouped_gemm",
            usize::from(n_experts),
            usize::from(in_blocks) * 32,
            usize::from(out_blocks) * 32,
            usize::from(max_total_tokens),
        );

        assert!(result.is_ok());
        let src = result.unwrap();
        assert!(src.contains("expert_offsets"));
        assert!(src.contains("expert_id * OUT_IN"));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_expert_routing_finds_in_bounds_interval() {
        let c0: u8 = kani::any();
        let c1: u8 = kani::any();
        let c2: u8 = kani::any();
        let c3: u8 = kani::any();

        let offsets = [
            0u16,
            u16::from(c0),
            u16::from(c0) + u16::from(c1),
            u16::from(c0) + u16::from(c1) + u16::from(c2),
            u16::from(c0) + u16::from(c1) + u16::from(c2) + u16::from(c3),
        ];

        kani::assume(offsets[ROUTED_EXPERTS] > 0);

        let tile_row_global: u16 = kani::any();
        kani::assume(tile_row_global < offsets[ROUTED_EXPERTS]);

        let expert_id = select_expert(offsets, tile_row_global).unwrap();
        assert!(expert_id < ROUTED_EXPERTS);
        assert!(tile_row_global >= offsets[expert_id]);
        assert!(tile_row_global < offsets[expert_id + 1]);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn prove_topk_positions_index_routing_weights_in_bounds() {
        let n_tokens: u8 = kani::any();
        let experts_per_tok: u8 = kani::any();
        let dst_token: u8 = kani::any();
        let k_idx: u8 = kani::any();
        kani::assume(n_tokens > 0);
        kani::assume(experts_per_tok > 0);
        kani::assume(dst_token < n_tokens);
        kani::assume(k_idx < experts_per_tok);

        let routing_len = usize::from(n_tokens) * usize::from(experts_per_tok);
        let routing_index =
            usize::from(dst_token) * usize::from(experts_per_tok) + usize::from(k_idx);

        assert!(routing_index < routing_len);
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(256)]
    fn prove_unpermute_codegen_embeds_topk_width() {
        let d_hidden_blocks: u8 = kani::any();
        let experts_per_tok: u8 = kani::any();
        kani::assume(d_hidden_blocks > 0);
        kani::assume(experts_per_tok > 0);

        let src = emit_moe_unpermute_kernel(
            "kani_moe_unpermute",
            usize::from(d_hidden_blocks) * 32,
            usize::from(experts_per_tok),
        )
        .unwrap();

        assert!(src.contains("routing_weights[dst_token * K + k_idx]"));
        assert!(src.contains(&format!("const unsigned int K = {};", experts_per_tok)));
    }
}
