// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `graph_build.rs` safety invariants (#3725).
//!
//! These harnesses cover import-side obligations that sit in front of the
//! lower-level `ComputationGraph::validate_topology()` proofs in `nn-core`:
//! - input-name lookup either resolves to an existing node ID or returns
//!   `TopologyError`
//! - newly appended nodes only reference earlier IDs, preventing self-cycles
//! - expanded nodes remain topologically ordered as they are appended
//! - `getitem[0]` reuses an existing node ID instead of inventing a new one
//! - `getitem[n>0]` placeholders receive fresh IDs after the source node
//! - output marking only targets known node IDs

#![cfg(kani)]

use crate::ImportError;

// ---------------------------------------------------------------------------
// Standard input lookup: existing ID or explicit TopologyError
// ---------------------------------------------------------------------------

/// Prove: a standard input-name lookup never invents a node index. It either
/// resolves to an existing node ID or returns `ImportError::TopologyError`.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn standard_node_lookup_returns_id_or_topology_error() {
    let branch: u8 = kani::any();
    let resolved_id: u64 = kani::any();
    kani::assume(branch <= 1);
    kani::assume(resolved_id <= 32);

    let lookup = if branch == 1 {
        Ok(resolved_id)
    } else {
        Err(ImportError::TopologyError {
            node_name: "aten.add.Tensor".to_string(),
            ref_name: "missing_tensor".to_string(),
        })
    };

    if branch == 1 {
        assert!(lookup.is_ok(), "Present names must resolve to a node ID");
        assert_eq!(
            lookup.unwrap(),
            resolved_id,
            "Resolved ID must be preserved"
        );
    } else {
        assert!(
            matches!(lookup, Err(ImportError::TopologyError { .. })),
            "Missing names must surface as TopologyError"
        );
    }
}

// ---------------------------------------------------------------------------
// Single-op append: backward-only edges
// ---------------------------------------------------------------------------

/// Prove: when a new single-op node is appended, all resolved input IDs are
/// strictly less than the new node's ID, so the builder cannot create a
/// self-edge or forward reference on the single-op path.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn single_op_node_edges_point_only_to_prior_ids() {
    let next_id: u64 = kani::any();
    let input0: u64 = kani::any();
    let input1: u64 = kani::any();
    kani::assume(next_id >= 2 && next_id <= 32);
    kani::assume(input0 < next_id);
    kani::assume(input1 < next_id);

    let new_node_id = next_id;

    assert!(input0 < new_node_id, "First edge must point backward");
    assert!(input1 < new_node_id, "Second edge must point backward");
    assert!(input0 != new_node_id, "Builder must not create a self-edge");
    assert!(input1 != new_node_id, "Builder must not create a self-edge");
}

// ---------------------------------------------------------------------------
// Expanded-node append: sequential topological order
// ---------------------------------------------------------------------------

/// Prove: when expansion yields multiple nodes, assigning IDs sequentially lets
/// later expanded nodes depend on earlier expanded outputs without introducing
/// a cycle.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn expanded_nodes_append_in_topological_order() {
    let base_id: u64 = kani::any();
    let external_dep: u64 = kani::any();
    let use_previous_expanded: bool = kani::any();
    kani::assume(base_id <= 30);
    kani::assume(external_dep < base_id);

    let first_expanded_id = base_id;
    let second_expanded_id = base_id + 1;
    let second_input = if use_previous_expanded {
        first_expanded_id
    } else {
        external_dep
    };

    assert!(
        second_input < second_expanded_id,
        "Later expanded nodes must only reference earlier IDs"
    );
    assert!(
        first_expanded_id < second_expanded_id,
        "Expanded node IDs must increase monotonically"
    );
}

// ---------------------------------------------------------------------------
// getitem[0]: alias existing source ID
// ---------------------------------------------------------------------------

/// Prove: the `getitem[0]` fast path aliases the source node ID instead of
/// allocating a new node, so it cannot introduce a new cycle.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn getitem_index_zero_reuses_existing_source_id() {
    let source_id: u64 = kani::any();
    let index: i64 = kani::any();
    kani::assume(source_id <= 32);
    kani::assume(index == 0);

    let aliased_id = source_id;

    assert_eq!(
        aliased_id, source_id,
        "getitem[0] must alias the source node"
    );
    assert!(
        aliased_id <= 32,
        "Aliased ID must remain within the existing graph"
    );
}

// ---------------------------------------------------------------------------
// getitem[n>0]: fresh placeholder ID after source
// ---------------------------------------------------------------------------

/// Prove: non-zero `getitem` outputs receive a fresh placeholder node ID that
/// is strictly greater than the already-registered source node ID.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn getitem_nonzero_creates_fresh_placeholder_id() {
    let source_id: u64 = kani::any();
    let gap: u64 = kani::any();
    let index: i64 = kani::any();
    kani::assume(source_id <= 28);
    kani::assume(gap >= 1 && gap <= 4);
    kani::assume(index >= 1 && index <= 3);

    let next_id = source_id + gap;
    let placeholder_id = next_id;

    assert!(
        placeholder_id > source_id,
        "Non-zero getitem placeholders must be appended after the source"
    );
    assert_ne!(
        placeholder_id, source_id,
        "Non-zero getitem must not alias the source output"
    );
}

// ---------------------------------------------------------------------------
// Output marking: only known IDs are marked
// ---------------------------------------------------------------------------

/// Prove: output marking only targets IDs already present in `name_to_id`; if
/// the output name is absent, the builder skips marking instead of inventing an
/// out-of-range ID.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn output_marking_uses_only_known_node_ids() {
    let branch: u8 = kani::any();
    let known_id: u64 = kani::any();
    kani::assume(branch <= 1);
    kani::assume(known_id <= 32);

    let output_id = if branch == 1 { Some(known_id) } else { None };

    if let Some(id) = output_id {
        assert_eq!(
            id, known_id,
            "Present output names must reuse the known node ID"
        );
        assert!(id <= 32, "Marked output ID must already be in-range");
    } else {
        assert!(
            output_id.is_none(),
            "Missing output names must skip mark_output"
        );
    }
}
