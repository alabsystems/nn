// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! BPE encoding methods for WhisperTokenizer.
//!
//! Extracted from `tokenizer.rs` (#1667) to keep files under 400 lines.
//! Contains `WhisperTokenizer::encode()` and the internal `bpe()` method.

use std::collections::BinaryHeap;

use super::{bpe_pair_key, pre_tokenize, WhisperError, WhisperTokenizer};
use nn_core::{Result, TensorError};

/// Priority queue entry for BPE merge selection. Lower rank = higher priority.
#[derive(Eq, PartialEq)]
struct BpeMerge {
    rank: usize,
    pos: usize,
}

impl Ord for BpeMerge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering: smallest rank = highest priority (min-heap via BinaryHeap).
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.pos.cmp(&self.pos))
    }
}

impl PartialOrd for BpeMerge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl WhisperTokenizer {
    /// Encode text to token IDs using GPT-2 byte-level BPE.
    ///
    /// Requires BPE merges loaded via [`from_vocab_and_merges`] or [`from_files`].
    /// Returns error if merges are not loaded or a token is not in the vocabulary.
    ///
    /// The input text is split into words at byte boundaries matching the GPT-2
    /// pre-tokenization pattern, then each word is encoded independently via BPE.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        if self.bpe_ranks.is_empty() {
            return Err(WhisperError::MissingMerges.into());
        }

        let mut ids = Vec::new();
        for word in pre_tokenize(text) {
            // Convert each byte to its GPT-2 unicode character.
            let encoded: String = word
                .as_bytes()
                .iter()
                .map(|&b| self.byte_encoder[&b])
                .collect();

            // Apply BPE merges.
            let tokens = self.bpe(&encoded);

            // Look up each BPE token in the vocabulary.
            for tok in &tokens {
                let id = self.token_to_id.get(tok).copied().ok_or_else(|| {
                    TensorError::from(WhisperError::TokenNotInVocab { token: tok.clone() })
                })?;
                ids.push(id);
            }
        }

        Ok(ids)
    }

    /// Apply BPE merges to a GPT-2 byte-encoded string.
    ///
    /// Uses a priority queue for O(n log n) merge selection instead of O(n²)
    /// linear scan. Each character starts as a node in a doubly-linked list.
    /// Merges consume two adjacent nodes by concatenating their strings and
    /// removing the right node. Stale heap entries (referencing removed nodes)
    /// are detected via alive flags and pair re-verification, then skipped.
    pub(crate) fn bpe(&self, token: &str) -> Vec<String> {
        let chars: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        let n = chars.len();
        if n <= 1 {
            return chars;
        }

        // Doubly-linked list via parallel arrays (usize::MAX = sentinel).
        let mut text: Vec<String> = chars;
        let mut prev: Vec<usize> = (0..n)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let mut next: Vec<usize> = (0..n)
            .map(|i| if i + 1 < n { i + 1 } else { usize::MAX })
            .collect();
        let mut alive: Vec<bool> = vec![true; n];

        // Seed the heap with all initial adjacent pairs.
        // key_buf is reused across all lookups to avoid per-lookup String allocation.
        let mut heap = BinaryHeap::with_capacity(n);
        let mut key_buf = String::new();
        let mut i = 0;
        while next[i] != usize::MAX {
            let j = next[i];
            bpe_pair_key(&mut key_buf, &text[i], &text[j]);
            if let Some(&rank) = self.bpe_ranks.get(&key_buf) {
                heap.push(BpeMerge { rank, pos: i });
            }
            i = j;
        }

        self.bpe_merge_loop(
            &mut heap,
            &mut text,
            &mut prev,
            &mut next,
            &mut alive,
            &mut key_buf,
        );
        collect_linked_list(&mut text, &next, &alive, n)
    }

    /// Execute the BPE merge loop, consuming entries from the priority queue.
    fn bpe_merge_loop(
        &self,
        heap: &mut BinaryHeap<BpeMerge>,
        text: &mut [String],
        prev: &mut [usize],
        next: &mut [usize],
        alive: &mut [bool],
        key_buf: &mut String,
    ) {
        while let Some(BpeMerge { rank: _, pos }) = heap.pop() {
            if !alive[pos] {
                continue;
            }
            let right = next[pos];
            if right == usize::MAX || !alive[right] {
                continue;
            }
            // Re-verify the pair still exists (stale entry detection).
            bpe_pair_key(key_buf, &text[pos], &text[right]);
            if !self.bpe_ranks.contains_key(key_buf.as_str()) {
                continue;
            }

            // Merge: concatenate text[pos] += text[right], remove right node.
            let right_text = std::mem::take(&mut text[right]);
            text[pos].push_str(&right_text);
            alive[right] = false;

            // Relink: pos.next = right.next; right.next.prev = pos.
            let right_next = next[right];
            next[pos] = right_next;
            if right_next != usize::MAX {
                prev[right_next] = pos;
            }

            // Re-insert new neighbor pairs into the heap.
            if prev[pos] != usize::MAX {
                let lp = prev[pos];
                bpe_pair_key(key_buf, &text[lp], &text[pos]);
                if let Some(&rank) = self.bpe_ranks.get(key_buf.as_str()) {
                    heap.push(BpeMerge { rank, pos: lp });
                }
            }
            if next[pos] != usize::MAX {
                let rp = next[pos];
                bpe_pair_key(key_buf, &text[pos], &text[rp]);
                if let Some(&rank) = self.bpe_ranks.get(key_buf.as_str()) {
                    heap.push(BpeMerge { rank, pos });
                }
            }
        }
    }
}

/// Collect surviving nodes by traversing the linked list.
fn collect_linked_list(
    text: &mut [String],
    next: &[usize],
    alive: &[bool],
    n: usize,
) -> Vec<String> {
    let mut result = Vec::new();
    let mut cur = 0;
    while cur < n && !alive[cur] {
        cur += 1;
    }
    while cur < n && alive[cur] {
        result.push(std::mem::take(&mut text[cur]));
        if next[cur] == usize::MAX {
            break;
        }
        cur = next[cur];
    }
    result
}
