// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CTC (Connectionist Temporal Classification) decoding.
//!
//! Greedy and beam-search decoding for CTC-based OCR and speech models.
//! Input: logits tensor `[T, vocab_size]` (one time step per row).
//! Output: decoded token sequence with repeats collapsed and blanks removed.

use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};

/// Configuration for CTC decoding.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CtcConfig {
    /// Token ID representing the CTC blank symbol (default: 0).
    pub blank_id: u32,
}

impl CtcConfig {
    /// Create config with a custom blank token ID.
    #[must_use]
    pub fn new(blank_id: u32) -> Self {
        Self { blank_id }
    }
}

/// CTC greedy decoding: argmax per time step, collapse repeats, remove blanks.
///
/// Input `logits`: shape `[T, vocab_size]` — raw logits (pre-softmax).
/// Returns the decoded token sequence.
///
/// Algorithm:
/// 1. Take argmax at each time step → raw token sequence
/// 2. Collapse consecutive duplicate tokens
/// 3. Remove blank tokens
pub fn ctc_greedy_decode(logits: &DynTensor, config: &CtcConfig) -> Result<Vec<u32>> {
    if logits.rank() != 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            actual: logits.rank(),
        });
    }
    let t = logits.dim(0)?;
    let vocab = logits.dim(1)?;
    if t == 0 || vocab == 0 {
        return Ok(Vec::new());
    }
    if (config.blank_id as usize) >= vocab {
        return Err(TensorError::DimensionOutOfRange {
            dim: config.blank_id as usize,
            rank: vocab,
        });
    }
    let data = logits.to_f32_array()?;
    let flat: Vec<f32> = data.iter().copied().collect();
    // Step 1: argmax per timestep.
    let mut raw_tokens: Vec<u32> = Vec::with_capacity(t);
    for step in 0..t {
        let row = &flat[step * vocab..(step + 1) * vocab];
        let best = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        raw_tokens.push(best);
    }
    // Step 2: collapse consecutive duplicates.
    let mut collapsed: Vec<u32> = Vec::with_capacity(t);
    let mut prev: Option<u32> = None;
    for &tok in &raw_tokens {
        if prev != Some(tok) {
            collapsed.push(tok);
            prev = Some(tok);
        }
    }
    // Step 3: remove blanks.
    let result: Vec<u32> = collapsed
        .into_iter()
        .filter(|&tok| tok != config.blank_id)
        .collect();
    Ok(result)
}

/// A single beam hypothesis for CTC beam decoding.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CtcBeamHypothesis {
    /// Decoded token sequence (without blanks or collapsed repeats).
    pub tokens: Vec<u32>,
    /// Log probability score.
    pub log_prob: f64,
}

impl CtcBeamHypothesis {
    /// Create a CTC beam hypothesis.
    pub fn new(tokens: Vec<u32>, log_prob: f64) -> Self {
        Self { tokens, log_prob }
    }
}

/// CTC prefix beam search decoding.
///
/// Input `logits`: shape `[T, vocab_size]` — raw logits (pre-softmax).
/// Returns up to `beam_width` hypotheses sorted by score (highest first).
///
/// Uses the prefix beam search algorithm: maintains beam_width hypotheses,
/// each tracking probability of ending in blank vs non-blank separately.
pub fn ctc_beam_decode(
    logits: &DynTensor,
    config: &CtcConfig,
    beam_width: usize,
) -> Result<Vec<CtcBeamHypothesis>> {
    if logits.rank() != 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            actual: logits.rank(),
        });
    }
    if beam_width == 0 {
        return Err(TensorError::ValueOutOfRange {
            description: "ctc_beam_decode: beam_width must be > 0",
        });
    }
    let t = logits.dim(0)?;
    let vocab = logits.dim(1)?;
    if t == 0 || vocab == 0 {
        return Ok(vec![CtcBeamHypothesis {
            tokens: Vec::new(),
            log_prob: 0.0,
        }]);
    }
    if (config.blank_id as usize) >= vocab {
        return Err(TensorError::DimensionOutOfRange {
            dim: config.blank_id as usize,
            rank: vocab,
        });
    }
    let data = logits.to_f32_array()?;
    let flat: Vec<f32> = data.iter().copied().collect();
    // Compute log-softmax per time step.
    let mut log_probs = vec![0.0f64; t * vocab];
    for step in 0..t {
        let row = &flat[step * vocab..(step + 1) * vocab];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // Guard: if max_val is NEG_INFINITY (all inputs are -inf), the subtraction
        // `v - max_val` produces NaN (inf - inf). Fill with -inf instead.
        // Matches the CPU softmax NaN guard (#1310) and beam_search_helpers.rs.
        if max_val == f32::NEG_INFINITY {
            for v in 0..vocab {
                log_probs[step * vocab + v] = f64::NEG_INFINITY;
            }
            continue;
        }
        let sum_exp: f64 = row.iter().map(|&v| f64::from(v - max_val).exp()).sum();
        let log_sum = sum_exp.ln() + f64::from(max_val);
        for v in 0..vocab {
            log_probs[step * vocab + v] = f64::from(row[v]) - log_sum;
        }
    }
    // Prefix trie: nodes store (token, parent) pairs. Extending a prefix
    // allocates one node — O(1). HashMap keys are TrieNodeId (usize) — O(1)
    // hash/compare. Reconstruction walks parent pointers — O(T) but only once
    // at the end. This reduces per-timestep cost from O(T × B × V) to O(B × V).
    let mut trie = PrefixTrie::new();
    let root = trie.root();

    // Each beam: (trie_node_id, prob_blank, prob_non_blank).
    type Beam = (TrieNodeId, f64, f64);
    let mut beams: Vec<Beam> = vec![(root, 0.0, f64::NEG_INFINITY)];

    for step in 0..t {
        let lp = &log_probs[step * vocab..(step + 1) * vocab];
        let mut next_beams: std::collections::HashMap<TrieNodeId, (f64, f64)> =
            std::collections::HashMap::new();
        for &(node_id, pb, pnb) in &beams {
            let p_total = log_add(pb, pnb);
            let last_token = trie.last_token(node_id);
            // Extend with blank — reuses same node (prefix unchanged).
            if let Some(entry) = next_beams.get_mut(&node_id) {
                entry.0 = log_add(entry.0, p_total + lp[config.blank_id as usize]);
            } else {
                next_beams.insert(
                    node_id,
                    (p_total + lp[config.blank_id as usize], f64::NEG_INFINITY),
                );
            }
            // Extend with each non-blank token.
            for (c, &lp_c) in lp.iter().enumerate() {
                let c_u32 = c as u32;
                if c_u32 == config.blank_id {
                    continue;
                }
                if last_token == Some(c_u32) {
                    // Same as last character: only extend from blank path.
                    let ext_id = trie.extend(node_id, c_u32);
                    if let Some(entry) = next_beams.get_mut(&ext_id) {
                        entry.1 = log_add(entry.1, pb + lp_c);
                    } else {
                        next_beams.insert(ext_id, (f64::NEG_INFINITY, pb + lp_c));
                    }
                    // Also keep the prefix without extending (from non-blank path).
                    if let Some(entry) = next_beams.get_mut(&node_id) {
                        entry.1 = log_add(entry.1, pnb + lp_c);
                    } else {
                        next_beams.insert(node_id, (f64::NEG_INFINITY, pnb + lp_c));
                    }
                } else {
                    // Different character: extend from total path.
                    let ext_id = trie.extend(node_id, c_u32);
                    if let Some(entry) = next_beams.get_mut(&ext_id) {
                        entry.1 = log_add(entry.1, p_total + lp_c);
                    } else {
                        next_beams.insert(ext_id, (f64::NEG_INFINITY, p_total + lp_c));
                    }
                }
            }
        }
        // Prune to beam_width.
        let mut candidates: Vec<Beam> = next_beams
            .into_iter()
            .map(|(node_id, (pb, pnb))| (node_id, pb, pnb))
            .collect();
        candidates.sort_by(|a, b| {
            let sa = log_add(a.1, a.2);
            let sb = log_add(b.1, b.2);
            sb.total_cmp(&sa)
        });
        candidates.truncate(beam_width);
        beams = candidates;
    }
    // Collect results — reconstruct prefix sequences from trie (O(T) per beam).
    let mut results: Vec<CtcBeamHypothesis> = beams
        .into_iter()
        .map(|(node_id, pb, pnb)| CtcBeamHypothesis {
            tokens: trie.reconstruct(node_id),
            log_prob: log_add(pb, pnb),
        })
        .collect();
    results.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
    Ok(results)
}

/// Log-domain addition: log(exp(a) + exp(b)), numerically stable.
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// Node ID in the prefix trie. 0 is the root (empty prefix).
type TrieNodeId = usize;

/// Arena-based prefix trie for CTC beam search.
///
/// Each node stores a token and its parent index. Extending a prefix is O(1)
/// (allocate a new node). Looking up or comparing prefixes uses integer IDs —
/// O(1) hash and equality. Reconstructing the full token sequence walks parent
/// pointers in O(T).
///
/// Deduplication: `extend()` reuses an existing child node if the same
/// `(parent, token)` pair was already created, via a HashMap lookup. This
/// ensures that two beams arriving at the same prefix share the same node ID
/// and can be merged in the beam HashMap.
struct PrefixTrie {
    /// nodes[i] = (token, parent_id). nodes[0] is the root sentinel.
    nodes: Vec<(u32, TrieNodeId)>,
    /// Maps (parent_id, token) → child node ID for deduplication.
    children: std::collections::HashMap<(TrieNodeId, u32), TrieNodeId>,
}

impl PrefixTrie {
    fn new() -> Self {
        Self {
            // Root node: token is unused (sentinel), parent points to self.
            nodes: vec![(u32::MAX, 0)],
            children: std::collections::HashMap::new(),
        }
    }

    fn root(&self) -> TrieNodeId {
        0
    }

    /// Get the last token of the prefix ending at `node_id`, or None for root.
    fn last_token(&self, node_id: TrieNodeId) -> Option<u32> {
        if node_id == 0 {
            None
        } else {
            Some(self.nodes[node_id].0)
        }
    }

    /// Extend a prefix by one token. Returns the child node ID.
    /// Deduplicates: reuses existing child if `(parent, token)` already exists.
    fn extend(&mut self, parent: TrieNodeId, token: u32) -> TrieNodeId {
        let key = (parent, token);
        if let Some(&child_id) = self.children.get(&key) {
            return child_id;
        }
        let child_id = self.nodes.len();
        self.nodes.push((token, parent));
        self.children.insert(key, child_id);
        child_id
    }

    /// Reconstruct the full token sequence by walking parent pointers.
    fn reconstruct(&self, mut node_id: TrieNodeId) -> Vec<u32> {
        let mut tokens = Vec::new();
        while node_id != 0 {
            let (token, parent) = self.nodes[node_id];
            tokens.push(token);
            node_id = parent;
        }
        tokens.reverse();
        tokens
    }
}

#[cfg(test)]
#[path = "ctc_tests.rs"]
mod tests;
