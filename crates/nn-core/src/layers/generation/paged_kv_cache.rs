// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Paged KV cache for memory-efficient batch serving.
//!
//! [`PagedKvCache`] uses a page-table design inspired by vLLM to manage
//! KV cache memory across multiple concurrent sequences. Instead of
//! pre-allocating a contiguous buffer per sequence, physical pages of
//! fixed size are allocated from a shared pool and mapped to sequences
//! via per-sequence page tables.
//!
//! # Why paging?
//!
//! In batch serving, sequences have different lengths and lifetimes.
//! Contiguous allocation wastes memory on padding and suffers from
//! fragmentation when sequences complete at different times. Paged
//! allocation solves both:
//!
//! - **No fragmentation:** Pages are uniform-sized and interchangeable.
//! - **No padding waste:** Each sequence uses only as many pages as it needs.
//! - **Efficient reclamation:** When a sequence completes, its pages return
//!   to the free pool immediately — no compaction needed.
//!
//! # Layout
//!
//! Each physical page stores `page_size` token positions for one layer,
//! for all heads. Page data is stored as flat `Vec<f32>` of length
//! `num_heads * head_dim * page_size`.
//!
//! The page pool is shared across all layers — `page_pool_k[page_idx]`
//! and `page_pool_v[page_idx]` hold K and V data for one page.
//! Each page is assigned to exactly one `(seq_id, layer)` pair.
//!
//! # Example
//!
//! ```ignore
//! let mut cache = PagedKvCache::new(16, 1024, 32, 8, 64)?;
//! cache.allocate_sequence(0)?;
//! cache.append_kv(0, 0, &k_data, &v_data)?;
//! let (k, v) = cache.get_kv(0, 0)?;
//! cache.free_sequence(0);
//! ```

use std::collections::HashMap;

use crate::{Result, TensorError};

/// Fixed-size page storing KV data for `page_size` token positions.
///
/// Data layout: `[page_size, num_heads, head_dim]` flattened to `Vec<f32>`.
/// Each page belongs to exactly one `(seq_id, layer)` logical slot.
#[derive(Debug, Clone)]
struct Page {
    data: Vec<f32>,
}

impl Page {
    /// Create a zero-initialized page.
    fn new(page_size: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            data: vec![0.0; page_size * num_heads * head_dim],
        }
    }
}

/// Per-sequence, per-layer page mapping.
///
/// Tracks which physical pages are assigned to this sequence for each layer,
/// and how many tokens have been written per layer.
#[derive(Debug, Clone)]
struct SequencePageTable {
    /// `layer_pages[layer]` = list of physical page indices for this layer.
    layer_pages: Vec<Vec<usize>>,
    /// Per-layer token count. `layer_token_counts[layer]` is the number of
    /// tokens written to that layer. In normal usage all layers have the same
    /// count, but the per-layer tracking avoids coupling between layers during
    /// `append_kv`.
    layer_token_counts: Vec<usize>,
}

/// Paged KV cache for memory-efficient batch serving.
///
/// Manages a shared pool of fixed-size pages. Sequences are allocated
/// pages on demand as tokens are appended. When a sequence completes,
/// its pages are returned to the free pool for reuse.
///
/// This is a CPU-side data structure. GPU paged attention kernels would
/// read the page tables to gather/scatter KV data from physical pages.
#[derive(Debug, Clone)]
pub struct PagedKvCache {
    /// Number of token positions per page.
    page_size: usize,
    /// Total number of physical pages in the pool.
    num_pages: usize,
    /// Number of transformer layers.
    num_layers: usize,
    /// Number of KV heads per layer.
    num_heads: usize,
    /// Dimension of each attention head.
    head_dim: usize,
    /// Physical K page pool. `page_pool_k[page_idx]` holds key data.
    page_pool_k: Vec<Page>,
    /// Physical V page pool. `page_pool_v[page_idx]` holds value data.
    page_pool_v: Vec<Page>,
    /// Per-sequence page tables. `page_tables[seq_id]` maps layers to pages.
    page_tables: HashMap<usize, SequencePageTable>,
    /// Stack of free (unallocated) page indices.
    free_pages: Vec<usize>,
}

impl PagedKvCache {
    /// Create a new paged KV cache.
    ///
    /// # Arguments
    ///
    /// * `page_size` — tokens per page (e.g., 16). Must be > 0.
    /// * `num_pages` — total physical pages in the pool. Must be > 0.
    /// * `num_layers` — number of transformer layers. Must be > 0.
    /// * `num_heads` — number of KV heads per layer. Must be > 0.
    /// * `head_dim` — dimension of each attention head. Must be > 0.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ValueOutOfRange` if any dimension is zero.
    pub fn new(
        page_size: usize,
        num_pages: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if page_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "PagedKvCache: page_size must be > 0",
            });
        }
        if num_pages == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "PagedKvCache: num_pages must be > 0",
            });
        }
        if num_layers == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "PagedKvCache: num_layers must be > 0",
            });
        }
        if num_heads == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "PagedKvCache: num_heads must be > 0",
            });
        }
        if head_dim == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "PagedKvCache: head_dim must be > 0",
            });
        }

        let page_pool_k = (0..num_pages)
            .map(|_| Page::new(page_size, num_heads, head_dim))
            .collect();
        let page_pool_v = (0..num_pages)
            .map(|_| Page::new(page_size, num_heads, head_dim))
            .collect();

        // All pages start free. Use reverse order so pop() yields lowest index first.
        let free_pages = (0..num_pages).rev().collect();

        Ok(Self {
            page_size,
            num_pages,
            num_layers,
            num_heads,
            head_dim,
            page_pool_k,
            page_pool_v,
            page_tables: HashMap::new(),
            free_pages,
        })
    }

    /// Allocate a new sequence in the cache.
    ///
    /// Assigns one initial page per layer for the sequence. The sequence
    /// starts with zero tokens — use [`append_kv`](Self::append_kv) to add data.
    ///
    /// # Errors
    ///
    /// - Returns an error if `seq_id` is already allocated.
    /// - Returns an error if there are not enough free pages (`num_layers` needed).
    pub fn allocate_sequence(&mut self, seq_id: usize) -> Result<()> {
        if self.page_tables.contains_key(&seq_id) {
            return Err(TensorError::ValueOutOfRange {
                description: "PagedKvCache: sequence already allocated",
            });
        }
        if self.free_pages.len() < self.num_layers {
            return Err(TensorError::ValueOutOfRange {
                description: "PagedKvCache: not enough free pages to allocate sequence",
            });
        }

        let mut layer_pages = Vec::with_capacity(self.num_layers);
        for _layer in 0..self.num_layers {
            let page_idx = self.free_pages.pop().ok_or(TensorError::ValueOutOfRange {
                description: "PagedKvCache: free page pool exhausted during allocation",
            })?;
            layer_pages.push(vec![page_idx]);
        }

        self.page_tables.insert(
            seq_id,
            SequencePageTable {
                layer_pages,
                layer_token_counts: vec![0; self.num_layers],
            },
        );
        Ok(())
    }

    /// Number of elements per token position: `num_heads * head_dim`.
    fn elements_per_token(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Append K/V data for a single token position to a specific layer.
    ///
    /// `k_data` and `v_data` must each have length `num_heads * head_dim`.
    ///
    /// If the current last page for this layer is full, a new page is allocated
    /// from the free pool.
    ///
    /// # Errors
    ///
    /// - Returns an error if `seq_id` is not allocated.
    /// - Returns an error if `layer` is out of range.
    /// - Returns an error if `k_data` or `v_data` has wrong length.
    /// - Returns an error if the free pool is exhausted when a new page is needed.
    pub fn append_kv(
        &mut self,
        seq_id: usize,
        layer: usize,
        k_data: &[f32],
        v_data: &[f32],
    ) -> Result<()> {
        let elems = self.elements_per_token();
        if k_data.len() != elems {
            return Err(TensorError::InvalidShape(format!(
                "PagedKvCache::append_kv: k_data length {} != expected {}",
                k_data.len(),
                elems,
            )));
        }
        if v_data.len() != elems {
            return Err(TensorError::InvalidShape(format!(
                "PagedKvCache::append_kv: v_data length {} != expected {}",
                v_data.len(),
                elems,
            )));
        }
        if layer >= self.num_layers {
            return Err(TensorError::DimensionOutOfRange {
                dim: layer,
                rank: self.num_layers,
            });
        }

        let seq = self
            .page_tables
            .get_mut(&seq_id)
            .ok_or(TensorError::ValueOutOfRange {
                description: "PagedKvCache::append_kv: sequence not allocated",
            })?;

        let layer_count = seq.layer_token_counts[layer];
        // Position within the current page for this layer.
        let offset_in_page = layer_count % self.page_size;

        // If the current page is full, allocate a new one.
        if offset_in_page == 0 && layer_count > 0 {
            let new_page = self.free_pages.pop().ok_or(TensorError::ValueOutOfRange {
                description: "PagedKvCache: no free pages for expansion",
            })?;
            seq.layer_pages[layer].push(new_page);
        }

        let page_idx = *seq.layer_pages[layer]
            .last()
            .ok_or(TensorError::ValueOutOfRange {
                description: "PagedKvCache: empty page list for layer",
            })?;

        let start = offset_in_page * elems;
        let end = start + elems;

        self.page_pool_k[page_idx].data[start..end].copy_from_slice(k_data);
        self.page_pool_v[page_idx].data[start..end].copy_from_slice(v_data);

        seq.layer_token_counts[layer] += 1;

        Ok(())
    }

    /// Read all cached K/V data for a sequence at a given layer.
    ///
    /// Returns `(k_data, v_data)` where each is a `Vec<f32>` of length
    /// `token_count * num_heads * head_dim`, laid out as
    /// `[token_count, num_heads, head_dim]`.
    ///
    /// # Errors
    ///
    /// - Returns an error if `seq_id` is not allocated.
    /// - Returns an error if `layer` is out of range.
    pub fn get_kv(&self, seq_id: usize, layer: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        if layer >= self.num_layers {
            return Err(TensorError::DimensionOutOfRange {
                dim: layer,
                rank: self.num_layers,
            });
        }

        let seq = self
            .page_tables
            .get(&seq_id)
            .ok_or(TensorError::ValueOutOfRange {
                description: "PagedKvCache::get_kv: sequence not allocated",
            })?;

        let layer_count = seq.layer_token_counts[layer];
        let elems = self.elements_per_token();
        let total_elems = layer_count * elems;
        let mut k_out = Vec::with_capacity(total_elems);
        let mut v_out = Vec::with_capacity(total_elems);

        let mut remaining = layer_count;
        for &page_idx in &seq.layer_pages[layer] {
            let tokens_in_page = remaining.min(self.page_size);
            let byte_len = tokens_in_page * elems;
            k_out.extend_from_slice(&self.page_pool_k[page_idx].data[..byte_len]);
            v_out.extend_from_slice(&self.page_pool_v[page_idx].data[..byte_len]);
            remaining -= tokens_in_page;
        }

        Ok((k_out, v_out))
    }

    /// Free all pages allocated to a sequence, returning them to the pool.
    ///
    /// After this call, `seq_id` is no longer valid. Pages are zeroed
    /// on next allocation (via `Page::new`), not on free — this is the
    /// common pattern for pool allocators.
    pub fn free_sequence(&mut self, seq_id: usize) {
        if let Some(seq) = self.page_tables.remove(&seq_id) {
            for layer_pages in &seq.layer_pages {
                for &page_idx in layer_pages {
                    self.free_pages.push(page_idx);
                }
            }
        }
    }

    /// Number of currently free (unallocated) pages.
    #[must_use]
    pub fn num_free_pages(&self) -> usize {
        self.free_pages.len()
    }

    /// Total number of physical pages in the pool.
    #[must_use]
    pub fn num_pages(&self) -> usize {
        self.num_pages
    }

    /// Number of tokens per page.
    #[must_use]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Number of active sequences.
    #[must_use]
    pub fn num_active_sequences(&self) -> usize {
        self.page_tables.len()
    }

    /// Token count for a given sequence (from layer 0), or `None` if not allocated.
    #[must_use]
    pub fn sequence_token_count(&self, seq_id: usize) -> Option<usize> {
        self.page_tables
            .get(&seq_id)
            .map(|s| s.layer_token_counts[0])
    }
}

#[cfg(kani)]
#[path = "kani_paged_kv_cache.rs"]
mod kani_paged_kv_cache;

#[cfg(test)]
#[path = "paged_kv_cache_tests.rs"]
mod tests;
