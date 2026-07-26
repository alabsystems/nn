// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Autoregressive generation, KV caching, beam search, and CTC decoding.
//!
//! - [`KvCache`] / [`KvCacheLayer`] — key-value caching for transformer inference
//! - [`generate`] — greedy and top-k autoregressive decoding
//! - [`beam_search`] — beam search decoding with length penalty
//! - [`ctc_greedy_decode`] / [`ctc_beam_decode`] — CTC decoding for ASR

// -- KV Cache -----------------------------------------------------------------
pub mod kv_cache;
pub use kv_cache::{KvCache, KvCacheBackend, KvCacheLayer, KvCacheLayerBackend};

// -- Pre-allocated KV Cache (GPU-resident compiled decoder inference) ----------
pub mod prealloc_kv_cache;
pub use prealloc_kv_cache::PreallocKvCacheLayer;

#[path = "prealloc_kv_cache_multi.rs"]
pub mod prealloc_kv_cache_multi;
pub use prealloc_kv_cache_multi::PreallocKvCache;

// -- Paged KV Cache (memory-efficient batch serving) --------------------------
pub mod paged_kv_cache;
pub use paged_kv_cache::PagedKvCache;

// -- Autoregressive generation ------------------------------------------------
pub mod autoregressive;
pub use autoregressive::{generate, GenerationConfig, GenerationOutput};

// -- Beam search decoding -----------------------------------------------------
pub mod beam_search;
pub use beam_search::{beam_search, BeamHypothesis, BeamSearchConfig, BeamSearchOutput};

// -- CTC decoding (greedy + beam search) --------------------------------------
mod ctc;
pub use ctc::{ctc_beam_decode, ctc_greedy_decode, CtcBeamHypothesis, CtcConfig};

// -- KV cache decode loop (GPU-resident compiled transformer inference) -------
pub mod decode_loop;
pub use decode_loop::{decode_generate, decode_step, prefill, DecodeContext};

// -- Multi-Token Prediction head ----------------------------------------------
mod mtp_head;
pub use mtp_head::{MtpHead, MtpHeadConfig};

// -- Speculative decoding with MTP head ---------------------------------------
mod mtp_speculative;
pub use mtp_speculative::{greedy_decode_with_verification, SpeculativeConfig, SpeculativeOutput};

// -- Kani proofs: MTP speculative decoding + multi-layer KV cache -------------
#[cfg(kani)]
#[path = "kani_mtp_kvcache_proofs.rs"]
mod kani_mtp_kvcache_proofs;

// -- Kani proofs: Extended paged KV cache safety invariants -------------------
#[cfg(kani)]
#[path = "kani_paged_kv_cache_extended.rs"]
mod kani_paged_kv_cache_extended;

// -- Kani proofs: Beam search and greedy decoding safety (#4239) --------------
#[cfg(kani)]
#[path = "kani_generation_safety.rs"]
mod kani_generation_safety;

// -- Kani proofs: dpdf VLM beam search and greedy decoding safety (#4239) -----
#[cfg(kani)]
#[path = "kani_dpdf_vlm_beam_greedy_proofs.rs"]
mod kani_dpdf_vlm_beam_greedy_proofs;

#[cfg(kani)]
#[path = "kani_dpdf_vlm_beam_greedy_proofs2.rs"]
mod kani_dpdf_vlm_beam_greedy_proofs2;

// -- Kani proofs: Extended beam search dpdf VLM safety (#4239) ---------------
#[cfg(kani)]
#[path = "kani_beam_dpdf_extended.rs"]
mod kani_beam_dpdf_extended;
