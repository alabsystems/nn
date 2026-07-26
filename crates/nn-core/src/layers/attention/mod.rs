// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Attention mechanisms and positional encodings.
//!
//! - [`MultiHeadAttention`] — standard multi-head attention with optional GQA, KV cache, RoPE
//! - [`SlidingWindowAttention`] — banded sliding window attention (Mistral, LongNet, BigBird)
//! - [`JointAttention`] — cross-modal attention (DiT models)
//! - [`DeformableAttention`] — deformable DETR attention with learned sampling offsets
//! - [`RotaryEmbedding`] / [`HalfRotaryEmbedding`] — rotary position embedding (RoPE)
//! - [`RotaryEmbedding2d`] — 2D rotary embedding for vision transformers
//! - [`alibi_bias`] / [`alibi_slopes`] — ALiBi position bias (Press et al. 2021)

// -- SDPA utilities (scaled dot-product attention, causal masks, repeat_kv) ---
mod sdpa;
pub use sdpa::{
    causal_mask, causal_mask_dtype, causal_mask_with_offset, repeat_kv, sdpa, sdpa_causal,
};

// -- Multi-Head Attention (standard transformer attention) --------------------
mod multi_head;
pub use multi_head::MultiHeadAttention;

// -- Joint Attention (DiT cross-modal attention) ------------------------------
mod joint;
pub use joint::JointAttention;

// -- Deformable Attention (Zhu et al., 2021 — deformable DETR) ----------------
mod deformable;
pub use deformable::{DeformableAttention, DeformableAttentionConfig};

// -- Rotary Position Embedding ------------------------------------------------
mod rope;
pub use rope::{rope, HalfRotaryEmbedding, RotaryEmbedding, YarnScaling};

// -- 2D Rotary Position Embedding + Sinusoidal 2D ----------------------------
mod rope_2d;
pub use rope_2d::{sinusoidal_2d, RotaryEmbedding2d};

// -- Multimodal RoPE (M-ROPE: 3-component temporal/height/width) -------------
mod rope_multimodal;
pub use rope_multimodal::MultimodalRoPE;

// -- Window attention utilities (partition/unpartition for local attention) ----
mod window;
pub use window::{
    window_partition, window_unpartition, AttentionMode, WindowAttentionConfig,
    WindowMultiHeadAttention,
};

// -- Interleaved Multimodal RoPE (Qwen3-VL, #3857) ----------------------------
mod interleaved_mrope;
pub use interleaved_mrope::{InterleavedMRoPE, InterleavedMRoPEConfig};

// -- ALiBi Position Bias (Press et al. 2021) ----------------------------------
mod alibi;
pub use alibi::{alibi_bias, alibi_bias_scaled, alibi_slopes};

// -- Sliding Window Attention (Mistral, LongNet, BigBird) ---------------------
mod sliding_window;
pub use sliding_window::{sliding_window_mask, SlidingWindowAttention};

// -- Gated DeltaNet (Yang et al., 2024 — linear attention for Qwen3.5) -------
mod gated_delta_net;
pub use gated_delta_net::{GatedDeltaNet, GatedDeltaNetConfig};

// -- SageAttention (Zhang et al., 2024 — INT8 quantized attention, #3862) -----
mod sage_attention;
pub use sage_attention::{SageAttention, SageAttentionConfig};

// -- Kani proof harnesses for SDPA + RoPE correctness (#3608) -----------------
#[cfg(kani)]
#[path = "kani_sdpa_rope_proofs.rs"]
mod kani_sdpa_rope_proofs;

// -- Kani proof harnesses for sliding_window + window + rope_2d (#3672) --------
#[cfg(kani)]
#[path = "kani_sliding_window_proofs.rs"]
mod kani_sliding_window_proofs;

#[cfg(kani)]
#[path = "kani_window_proofs.rs"]
mod kani_window_proofs;

#[cfg(kani)]
#[path = "kani_rope_2d_proofs.rs"]
mod kani_rope_2d_proofs;

// -- Advanced Kani proof harnesses for SDPA + RoPE + ALiBi (#3671) --------------
#[cfg(kani)]
#[path = "kani_sdpa_advanced.rs"]
mod kani_sdpa_advanced;

#[cfg(kani)]
#[path = "kani_rope_advanced.rs"]
mod kani_rope_advanced;

#[cfg(kani)]
#[path = "kani_alibi_advanced.rs"]
mod kani_alibi_advanced;

// -- Kani proof harnesses for Gated DeltaNet attention (#3699) ------------------
#[cfg(kani)]
#[path = "kani_gated_delta_net.rs"]
mod kani_gated_delta_net;

// -- Extended Kani proof harnesses for Gated DeltaNet attention (#3744) ---------
#[cfg(kani)]
#[path = "kani_gated_delta_net_extended.rs"]
mod kani_gated_delta_net_extended;

// -- Kani proof harnesses for Interleaved M-ROPE (#3867) -----------------------
#[cfg(kani)]
#[path = "kani_interleaved_mrope_proofs.rs"]
mod kani_interleaved_mrope_proofs;

// -- Kani proof harnesses for SageAttention + DeformableAttention (#4074) ------
#[cfg(kani)]
#[path = "kani_sage_deformable_proofs.rs"]
mod kani_sage_deformable_proofs;

// -- Advanced Kani proof harnesses for MLA + MultimodalRoPE (#4096) -----------
#[cfg(kani)]
#[path = "kani_mla_mrope_advanced_proofs.rs"]
mod kani_mla_mrope_advanced_proofs;

// -- Extended Kani proof harnesses for RoPE safety (#4191) --------------------
#[cfg(kani)]
#[path = "kani_rope_extended_proofs.rs"]
mod kani_rope_extended_proofs;
