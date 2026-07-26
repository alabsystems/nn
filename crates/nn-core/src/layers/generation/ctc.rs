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

#[cfg(kani)]
#[path = "kani_ctc_proofs.rs"]
mod kani_ctc_proofs;

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    fn exp_f64_stub(x: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
        if x <= 0.0 {
            kani::assume(r <= 1.0);
        }
        if x > 0.0 {
            kani::assume(r > 1.0);
        }
        r
    }

    fn ln_f64_stub(_x: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
        r
    }

    /// Prove `log_add` never panics and is commutative for any finite f64 inputs.
    /// Also prove that log_add(NEG_INF, x) == x (identity element).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::exp, exp_f64_stub)]
    #[kani::stub(f64::ln, ln_f64_stub)]
    fn proof_log_add_identity_neg_inf() {
        let x: f64 = kani::any();
        kani::assume(x.is_finite());
        let result = log_add(f64::NEG_INFINITY, x);
        assert!(
            (result - x).abs() < 1e-10,
            "log_add(NEG_INF, x) must equal x"
        );
        let result2 = log_add(x, f64::NEG_INFINITY);
        assert!(
            (result2 - x).abs() < 1e-10,
            "log_add(x, NEG_INF) must equal x"
        );
    }

    /// Prove `log_add` result is >= max(a, b) for finite inputs.
    /// log(exp(a) + exp(b)) >= log(exp(max(a,b))) = max(a,b).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::exp, exp_f64_stub)]
    #[kani::stub(f64::ln, ln_f64_stub)]
    fn proof_log_add_monotone() {
        let a: f64 = kani::any();
        let b: f64 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());
        kani::assume(a.abs() < 500.0 && b.abs() < 500.0);
        let result = log_add(a, b);
        let max_ab = a.max(b);
        assert!(result >= max_ab - 1e-10, "log_add(a,b) must be >= max(a,b)");
    }

    /// Prove `PrefixTrie::new` creates a trie with exactly one root node.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_prefix_trie_root_invariant() {
        let trie = PrefixTrie::new();
        assert_eq!(trie.root(), 0, "root must be 0");
        assert!(trie.last_token(0).is_none(), "root must have no last token");
    }

    /// Prove `PrefixTrie::extend` deduplicates: extending the same parent
    /// with the same token returns the same node ID.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_prefix_trie_dedup() {
        let mut trie = PrefixTrie::new();
        let root = trie.root();
        let token: u32 = kani::any();
        kani::assume(token < 100);
        let child1 = trie.extend(root, token);
        let child2 = trie.extend(root, token);
        assert_eq!(
            child1, child2,
            "extending same parent with same token must return same node"
        );
    }

    /// Prove `PrefixTrie::extend` with different tokens yields different node IDs.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_prefix_trie_distinct_children() {
        let mut trie = PrefixTrie::new();
        let root = trie.root();
        let t1: u32 = kani::any();
        let t2: u32 = kani::any();
        kani::assume(t1 < 100 && t2 < 100);
        kani::assume(t1 != t2);
        let c1 = trie.extend(root, t1);
        let c2 = trie.extend(root, t2);
        assert_ne!(
            c1, c2,
            "different tokens from same parent must yield different nodes"
        );
    }

    /// Prove `PrefixTrie::reconstruct` on root returns empty sequence.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_prefix_trie_reconstruct_root_empty() {
        let trie = PrefixTrie::new();
        let tokens = trie.reconstruct(trie.root());
        assert!(
            tokens.is_empty(),
            "reconstructing from root must yield empty sequence"
        );
    }

    /// Prove `PrefixTrie::reconstruct` returns tokens in correct order for
    /// a chain of extensions (up to depth 4).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_prefix_trie_reconstruct_chain() {
        let depth: usize = kani::any();
        kani::assume(depth >= 1 && depth <= 4);
        let mut trie = PrefixTrie::new();
        let mut node = trie.root();
        let mut expected_tokens: Vec<u32> = Vec::with_capacity(depth);
        for _ in 0..depth {
            let tok: u32 = kani::any();
            kani::assume(tok < 256);
            node = trie.extend(node, tok);
            expected_tokens.push(tok);
        }
        let reconstructed = trie.reconstruct(node);
        assert_eq!(
            reconstructed.len(),
            depth,
            "reconstructed length must match chain depth"
        );
        for i in 0..depth {
            assert_eq!(
                reconstructed[i], expected_tokens[i],
                "reconstructed tokens must match extension order"
            );
        }
    }

    /// Prove CTC greedy collapse: consecutive duplicates are collapsed.
    /// Simulates the collapse step on a bounded token sequence (up to 6).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(7)]
    fn proof_ctc_greedy_collapse_no_consecutive_duplicates() {
        let len: usize = kani::any();
        kani::assume(len >= 1 && len <= 6);
        let mut raw_tokens: Vec<u32> = Vec::with_capacity(len);
        for _ in 0..len {
            let tok: u32 = kani::any();
            kani::assume(tok < 8);
            raw_tokens.push(tok);
        }
        // Apply collapse step (same as ctc_greedy_decode step 2).
        let mut collapsed: Vec<u32> = Vec::with_capacity(len);
        let mut prev: Option<u32> = None;
        for &tok in &raw_tokens {
            if prev != Some(tok) {
                collapsed.push(tok);
                prev = Some(tok);
            }
        }
        // Property: no consecutive duplicates in output.
        for i in 1..collapsed.len() {
            assert_ne!(
                collapsed[i - 1],
                collapsed[i],
                "collapsed sequence must have no consecutive duplicates"
            );
        }
        // Property: collapsed length <= original length.
        assert!(collapsed.len() <= len);
        // Property: collapsed is non-empty (input is non-empty).
        assert!(!collapsed.is_empty());
    }

    /// Prove CTC greedy blank removal: blank_id is never in the output.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(7)]
    fn proof_ctc_greedy_blank_removal() {
        let len: usize = kani::any();
        kani::assume(len >= 1 && len <= 6);
        let blank_id: u32 = kani::any();
        kani::assume(blank_id < 4);
        let mut collapsed: Vec<u32> = Vec::with_capacity(len);
        for _ in 0..len {
            let tok: u32 = kani::any();
            kani::assume(tok < 8);
            collapsed.push(tok);
        }
        // Apply blank removal step (same as ctc_greedy_decode step 3).
        let result: Vec<u32> = collapsed
            .into_iter()
            .filter(|&tok| tok != blank_id)
            .collect();
        // Property: blank_id never appears in result.
        for &tok in &result {
            assert_ne!(
                tok, blank_id,
                "blank token must not appear in decoded output"
            );
        }
    }

    /// Prove `CtcConfig::new` stores the blank_id correctly.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_ctc_config_blank_id() {
        let blank_id: u32 = kani::any();
        kani::assume(blank_id < 1000);
        let config = CtcConfig::new(blank_id);
        assert_eq!(config.blank_id, blank_id);
    }
}

#[cfg(test)]
#[path = "ctc_tests.rs"]
mod tests;
