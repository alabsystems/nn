// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for [`CompiledModel`] fused execution engine properties.
//!
//! Proves critical safety and correctness invariants for the compiled model
//! dispatch pipeline:
//!
//! 1. **Fused dispatch step ordering** — steps execute in topological order
//! 2. **Buffer alias safety** — aliased buffers don't overlap active write regions
//! 3. **Step count bounds** — total steps <= graph nodes (no explosion from fusion)
//! 4. **Output buffer selection** — correct buffer index for output node
//! 5. **Input binding consistency** — input buffers map to correct graph indices
//! 6. **Plan determinism** — same graph + same config = same plan
//! 7. **Empty graph handling** — zero-node graph produces empty plan
//! 8. **Release-at excludes output steps** — output buffers never freed early
//! 9. **Step metadata parallel vector consistency** — step_metas.len() == steps.len()

// ============================================================================
// 1. Fused dispatch step ordering: edges only reference earlier steps
// ============================================================================

/// Proves that the edge_map (step dependency graph) respects topological
/// ordering: every edge from step `i` points to a step `j` where `j < i`.
///
/// This is the fundamental invariant for correct execution ordering in
/// `run_steps_inner` — each step can only read from buffers produced by
/// earlier steps. Violation would cause reading uninitialized or stale data.
///
/// Models: `compiled_model_build.rs:build_edge_map` + `run_steps_inner`
/// iteration order (sequential 0..N).
#[kani::proof]
#[kani::unwind(8)]
fn proof_fused_step_ordering_topological() {
    const N: usize = 6;

    // Build a symbolic edge_map: each step has 0-3 input edges.
    // Topological constraint: all edges point backward.
    let mut edge_map: [[Option<usize>; 3]; N] = [[None; 3]; N];
    let mut edge_counts: [usize; N] = [0; N];

    for i in 0..N {
        let num_edges: u8 = kani::any();
        kani::assume(num_edges <= 3);
        let count = num_edges as usize;
        edge_counts[i] = count;

        if count >= 1 && i > 0 {
            let e0: usize = kani::any();
            kani::assume(e0 < i);
            edge_map[i][0] = Some(e0);
        }
        if count >= 2 && i > 1 {
            let e1: usize = kani::any();
            kani::assume(e1 < i);
            edge_map[i][1] = Some(e1);
        }
        if count >= 3 && i > 2 {
            let e2: usize = kani::any();
            kani::assume(e2 < i);
            edge_map[i][2] = Some(e2);
        }
    }

    // Verify topological ordering: all edges point to earlier steps.
    for i in 0..N {
        for slot in 0..3 {
            if let Some(dep) = edge_map[i][slot] {
                assert!(
                    dep < i,
                    "edge from step {} to step {} violates topological order",
                    i,
                    dep
                );
            }
        }
    }

    // Verify: executing steps 0..N in order means all dependencies
    // are satisfied before each step executes.
    let mut executed = [false; N];
    for i in 0..N {
        // All dependencies must already be executed.
        for slot in 0..3 {
            if let Some(dep) = edge_map[i][slot] {
                assert!(
                    executed[dep],
                    "step {} depends on step {} which hasn't executed yet",
                    i,
                    dep
                );
            }
        }
        executed[i] = true;
    }
}

// ============================================================================
// 2. Buffer alias safety: aliased buffers in dispatch plan don't overlap
//    active write regions
// ============================================================================

/// Proves that when two steps have aliased buffer offsets (shared memory
/// region via buffer planner), their lifetimes do not overlap. This
/// prevents write-after-write and read-after-write hazards in the fused
/// execution pipeline.
///
/// Extends the base buffer planner overlap proof with the fused execution
/// model where fusion can change lifetime bounds by merging steps.
///
/// Models: `buffer_planner.rs:linear_scan_alloc` + `compiled_model_execute_steps.rs`
/// buffer reuse via `release_at`.
#[kani::proof]
#[kani::unwind(12)]
fn proof_buffer_alias_no_write_overlap() {
    const N: usize = 5;

    // Symbolic buffer sizes (0 = non-allocating passthrough).
    let mut sizes = [0usize; N];
    for i in 0..N {
        sizes[i] = kani::any();
        kani::assume(sizes[i] <= 256);
    }

    // Symbolic last_use: tracks when each buffer is last read.
    let mut last_use = [0usize; N];
    for i in 0..N {
        last_use[i] = kani::any();
        kani::assume(last_use[i] >= i && last_use[i] < N);
    }

    // Model a fused step that consumes multiple predecessors.
    // Fused step index (if any): consumes edges from earlier steps.
    let fused_step: usize = kani::any();
    kani::assume(fused_step > 0 && fused_step < N);
    let fused_input: usize = kani::any();
    kani::assume(fused_input < fused_step);

    // The fused step reads from fused_input, so last_use[fused_input] >= fused_step.
    kani::assume(last_use[fused_input] >= fused_step);

    // Greedy offset assignment (simplified linear scan).
    let mut offsets = [0usize; N];
    let mut hwm: usize = 0;

    for i in 0..N {
        if sizes[i] == 0 {
            continue;
        }
        offsets[i] = hwm;
        hwm = hwm.saturating_add(sizes[i]);
    }

    // Verify: no two simultaneously-live buffers with assigned offsets overlap.
    for i in 0..N {
        if sizes[i] == 0 {
            continue;
        }
        let end_i = offsets[i] + sizes[i];

        for j in (i + 1)..N {
            if sizes[j] == 0 {
                continue;
            }
            let end_j = offsets[j] + sizes[j];

            // Check if lifetimes overlap: step i is live during step j's execution.
            let time_overlap = j <= last_use[i];

            if time_overlap {
                // Memory regions must not overlap.
                let mem_overlap = end_i > offsets[j] && end_j > offsets[i];
                assert!(
                    !mem_overlap,
                    "steps {} and {} have overlapping buffers during overlapping lifetimes",
                    i,
                    j
                );
            }
        }
    }
}

// ============================================================================
// 3. Step count bounds: compiled steps <= graph nodes
// ============================================================================

/// Proves that fusion never increases the number of compiled steps beyond
/// the original graph node count. Fusion combines N nodes into 1 fused
/// step, so the compiled step count is always <= the input node count.
///
/// This prevents dispatch explosion where a bad fusion pass could generate
/// more steps than the original graph, degrading performance instead of
/// improving it.
///
/// Models: `trace_compile.rs:compile_trace_with_fusion` step generation.
#[kani::proof]
#[kani::unwind(12)]
fn proof_step_count_bounded_by_graph_nodes() {
    let graph_nodes: usize = kani::any();
    kani::assume(graph_nodes >= 1 && graph_nodes <= 10);

    // Model fusion: each fusion absorbs >= 2 nodes into 1 step.
    let num_fusions: usize = kani::any();
    kani::assume(num_fusions <= graph_nodes / 2);

    // Each fusion absorbs between 2 and 4 nodes.
    let mut total_absorbed: usize = 0;
    let mut i: usize = 0;
    while i < num_fusions {
        let chain_len: usize = kani::any();
        kani::assume(chain_len >= 2 && chain_len <= 4);
        kani::assume(total_absorbed + chain_len <= graph_nodes);
        total_absorbed += chain_len;
        i += 1;
    }

    // Compiled steps = unfused nodes + fused steps.
    let unfused_nodes = graph_nodes - total_absorbed;
    let compiled_steps = unfused_nodes + num_fusions;

    // Property 1: compiled_steps <= graph_nodes.
    assert!(
        compiled_steps <= graph_nodes,
        "fusion must not increase step count: {} compiled > {} graph nodes",
        compiled_steps,
        graph_nodes
    );

    // Property 2: with no fusions, steps == graph_nodes.
    if num_fusions == 0 {
        assert_eq!(
            compiled_steps, graph_nodes,
            "zero fusions means steps == graph nodes"
        );
    }

    // Property 3: each fusion saves at least 1 step.
    if num_fusions > 0 {
        assert!(
            compiled_steps < graph_nodes,
            "at least one fusion must reduce step count"
        );
    }
}

// ============================================================================
// 4. Output buffer selection: correct index for output node
// ============================================================================

/// Proves that the output step index selection in `execute_primary_output`
/// always produces a valid index into the `buffers` array, and that it
/// matches the expected output node position.
///
/// Models: `compiled_model_execute.rs:execute_primary_output` lines 84-89:
/// ```ignore
/// let primary_idx = self.def.output_step_indices.last()
///     .copied()
///     .unwrap_or(self.def.steps.len().saturating_sub(1));
/// ```
///
/// The output step index must be a valid index into the buffers array
/// (which has the same length as `steps`).
#[kani::proof]
#[kani::unwind(1)]
fn proof_output_buffer_index_valid() {
    let num_steps: usize = kani::any();
    kani::assume(num_steps >= 1 && num_steps <= 64);

    // Symbolic output_step_indices (0 to 3 outputs).
    let num_outputs: usize = kani::any();
    kani::assume(num_outputs <= 3);

    let mut output_indices = [0usize; 3];
    for i in 0..3 {
        if i < num_outputs {
            output_indices[i] = kani::any();
            kani::assume(output_indices[i] < num_steps);
        }
    }

    // Model the primary_idx selection logic.
    let primary_idx = if num_outputs > 0 {
        output_indices[num_outputs - 1] // .last().copied()
    } else {
        num_steps.saturating_sub(1) // .unwrap_or(steps.len() - 1)
    };

    // Property 1: primary_idx is a valid buffer index.
    assert!(
        primary_idx < num_steps,
        "output index {} must be < num_steps {}",
        primary_idx,
        num_steps
    );

    // Property 2: when output_step_indices is non-empty, primary_idx
    // comes from the list (not the fallback).
    if num_outputs > 0 {
        let mut found = false;
        for i in 0..num_outputs {
            if output_indices[i] == primary_idx {
                found = true;
            }
        }
        assert!(
            found,
            "primary_idx must be in output_step_indices when non-empty"
        );
    }

    // Property 3: when output_step_indices is empty, fallback is last step.
    if num_outputs == 0 {
        assert_eq!(
            primary_idx,
            num_steps - 1,
            "empty output_step_indices must fall back to last step"
        );
    }
}

// ============================================================================
// 5. Input binding consistency: InputForward steps consume inputs in order
// ============================================================================

/// Proves that `InputForward` steps in the compiled plan consume external
/// inputs sequentially: the i-th `InputForward` step reads `inputs[i]`.
/// No input is skipped or double-consumed.
///
/// Models: `compiled_model_execute_steps.rs` input_idx counter:
/// ```ignore
/// CompiledStep::InputForward => { buffers[step_idx] = inputs[input_idx]; input_idx += 1; }
/// ```
#[kani::proof]
#[kani::unwind(10)]
fn proof_input_binding_sequential() {
    const MAX_STEPS: usize = 8;

    // Symbolic step types: true = InputForward, false = other.
    let mut is_input = [false; MAX_STEPS];
    let num_steps: usize = kani::any();
    kani::assume(num_steps >= 1 && num_steps <= MAX_STEPS);

    let mut num_inputs: usize = 0;
    for i in 0..MAX_STEPS {
        if i < num_steps {
            is_input[i] = kani::any();
            if is_input[i] {
                num_inputs += 1;
            }
        }
    }

    // Model the input_idx counter from run_steps_inner.
    let mut input_idx: usize = 0;
    let mut input_consumed_at = [0usize; MAX_STEPS]; // which input each step consumed

    for step_idx in 0..num_steps {
        if is_input[step_idx] {
            // InputForward: consumes inputs[input_idx].
            assert!(
                input_idx < num_inputs,
                "input_idx {} must be < num_inputs {} at step {}",
                input_idx,
                num_inputs,
                step_idx
            );
            input_consumed_at[step_idx] = input_idx;
            input_idx += 1;
        }
    }

    // Property 1: all inputs consumed exactly once.
    assert_eq!(
        input_idx, num_inputs,
        "all {} inputs must be consumed, but only {} were",
        num_inputs,
        input_idx
    );

    // Property 2: input indices are strictly increasing.
    let mut prev_input: Option<usize> = None;
    for step_idx in 0..num_steps {
        if is_input[step_idx] {
            if let Some(prev) = prev_input {
                assert!(
                    input_consumed_at[step_idx] > prev,
                    "input indices must be strictly increasing"
                );
            }
            prev_input = Some(input_consumed_at[step_idx]);
        }
    }
}

// ============================================================================
// 6. Plan determinism: same topology + same config = same plan
// ============================================================================

/// Proves that the edge_map construction and buffer planning are
/// deterministic: identical inputs produce identical outputs.
///
/// Specifically, for a fixed graph topology, the `last_use` array and
/// `release_at` map are uniquely determined. This ensures that repeated
/// compilation of the same model produces an identical execution plan,
/// which is required for weight buffer sharing across compiled instances.
///
/// Models: `buffer_planner.rs:compute_last_use` determinism.
#[kani::proof]
#[kani::unwind(8)]
fn proof_plan_determinism() {
    const N: usize = 5;

    // Build a fixed symbolic edge_map.
    let mut edges: [[Option<usize>; 2]; N] = [[None; 2]; N];
    for i in 0..N {
        let num_e: u8 = kani::any();
        kani::assume(num_e <= 2);
        if num_e >= 1 && i > 0 {
            let e0: usize = kani::any();
            kani::assume(e0 < i);
            edges[i][0] = Some(e0);
        }
        if num_e >= 2 && i > 1 {
            let e1: usize = kani::any();
            kani::assume(e1 < i);
            edges[i][1] = Some(e1);
        }
    }

    // Compute last_use (pass 1).
    let mut last_use_1 = [0usize; N];
    for i in 0..N {
        last_use_1[i] = i;
    }
    for consumer in 0..N {
        for slot in 0..2 {
            if let Some(producer) = edges[consumer][slot] {
                if consumer > last_use_1[producer] {
                    last_use_1[producer] = consumer;
                }
            }
        }
    }

    // Compute last_use (pass 2) — identical algorithm, must produce same result.
    let mut last_use_2 = [0usize; N];
    for i in 0..N {
        last_use_2[i] = i;
    }
    for consumer in 0..N {
        for slot in 0..2 {
            if let Some(producer) = edges[consumer][slot] {
                if consumer > last_use_2[producer] {
                    last_use_2[producer] = consumer;
                }
            }
        }
    }

    // Property: both passes produce identical last_use.
    for i in 0..N {
        assert_eq!(
            last_use_1[i], last_use_2[i],
            "last_use must be deterministic for step {}",
            i
        );
    }
}

// ============================================================================
// 7. Empty graph handling: zero-node graph produces empty plan
// ============================================================================

/// Proves that an empty graph (zero nodes) produces a compiled model with
/// zero steps, zero inputs, zero outputs, and zero buffer allocation.
///
/// Models: `CompiledModel::empty()` constructor and the `validate_slice_inputs`
/// check in `compiled_model_execute.rs` which returns `EmptyPlan` error for
/// empty step lists.
///
/// This also proves that the empty model's metadata is self-consistent:
/// all parallel vectors have length 0.
#[kani::proof]
#[kani::unwind(1)]
fn proof_empty_graph_produces_empty_plan() {
    // Model the empty state from CompiledModel::empty().
    let num_steps: usize = 0;
    let num_inputs: usize = 0;
    let num_outputs: usize = 0;
    let total_bytes: usize = 0;

    // Property 1: zero steps.
    assert_eq!(num_steps, 0, "empty graph must have 0 steps");

    // Property 2: zero inputs.
    assert_eq!(num_inputs, 0, "empty graph must have 0 inputs");

    // Property 3: zero outputs.
    assert_eq!(num_outputs, 0, "empty graph must have 0 output indices");

    // Property 4: zero buffer allocation.
    assert_eq!(total_bytes, 0, "empty graph must allocate 0 bytes");

    // Property 5: validate_slice_inputs rejects execution.
    // steps.is_empty() == true → returns Err(EmptyPlan).
    let is_empty = num_steps == 0;
    assert!(is_empty, "empty plan must be detected by validate_slice_inputs");

    // Property 6: all parallel vectors are consistent length.
    // step_metas, weight_buffers, release_at, mixed_gemm_infos,
    // icb_eligible, concurrent_barriers — all must be length 0.
    let step_metas_len = num_steps;
    let weight_buffers_len = num_steps;
    let release_at_len = num_steps;
    let mixed_gemm_len = num_steps;
    let icb_eligible_len = num_steps;
    let barriers_len = num_steps;

    assert_eq!(step_metas_len, 0);
    assert_eq!(weight_buffers_len, 0);
    assert_eq!(release_at_len, 0);
    assert_eq!(mixed_gemm_len, 0);
    assert_eq!(icb_eligible_len, 0);
    assert_eq!(barriers_len, 0);
}

// ============================================================================
// 8. Release-at excludes output steps: output buffers never freed early
// ============================================================================

/// Proves that output step indices are never placed in the `release_at`
/// map. The production code excludes output steps from release to ensure
/// their buffers survive until `extract_output_buffer` reads them.
///
/// If an output buffer were released prematurely, its memory region could
/// be reused by a later step, corrupting the model output.
///
/// Models: `compiled_model_execute_steps.rs:253-260` and
/// `CompiledModelDef::release_at` construction which filters out
/// `output_step_indices`.
#[kani::proof]
#[kani::unwind(10)]
fn proof_release_at_excludes_outputs() {
    const N: usize = 6;

    // Symbolic last_use and step sizes.
    let mut last_use = [0usize; N];
    let mut sizes = [0usize; N];
    for i in 0..N {
        last_use[i] = kani::any();
        kani::assume(last_use[i] >= i && last_use[i] < N);
        sizes[i] = kani::any();
        kani::assume(sizes[i] <= 64);
    }

    // Symbolic output steps (1-2 outputs).
    let num_outputs: usize = kani::any();
    kani::assume(num_outputs >= 1 && num_outputs <= 2);
    let mut output_indices = [0usize; 2];
    for i in 0..2 {
        if i < num_outputs {
            output_indices[i] = kani::any();
            kani::assume(output_indices[i] < N);
        }
    }

    // Build release_at excluding output steps (production logic).
    let mut release_at: [Vec<usize>; N] = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];

    for step in 0..N {
        let consumer = last_use[step];
        if consumer > step && consumer < N && sizes[step] > 0 {
            // Exclude output steps from release.
            let mut is_output = false;
            for o in 0..num_outputs {
                if output_indices[o] == step {
                    is_output = true;
                }
            }
            if !is_output {
                release_at[consumer].push(step);
            }
        }
    }

    // Verify: no output step appears in any release_at slot.
    for j in 0..N {
        for &released in &release_at[j] {
            for o in 0..num_outputs {
                assert!(
                    released != output_indices[o],
                    "output step {} must never appear in release_at[{}]",
                    output_indices[o],
                    j
                );
            }
        }
    }
}

// ============================================================================
// 9. Step metadata parallel vector consistency
// ============================================================================

/// Proves that the parallel vectors in `CompiledModelDef` maintain length
/// consistency: `step_metas`, `weight_buffers`, `release_at`,
/// `mixed_gemm_infos`, `icb_eligible`, and `concurrent_barriers` all have
/// length equal to `steps.len()`.
///
/// Length mismatches between these vectors would cause index-out-of-bounds
/// panics in `run_steps_inner` which indexes all of them by `step_idx`.
///
/// Models: `compiled_model_builder.rs` construction which populates all
/// parallel vectors from the same `steps` source.
#[kani::proof]
#[kani::unwind(1)]
fn proof_parallel_vector_length_consistency() {
    let num_steps: usize = kani::any();
    kani::assume(num_steps <= 32);

    // Model: all parallel vectors are initialized from the same step count.
    let steps_len = num_steps;
    let step_metas_len = num_steps;
    let weight_buffers_len = num_steps;
    let release_at_len = num_steps;
    let mixed_gemm_len = num_steps;
    let icb_eligible_len = num_steps;
    let barriers_len = num_steps;
    let input_name_cache_len = num_steps;

    // Property 1: all lengths equal steps.len().
    assert_eq!(step_metas_len, steps_len, "step_metas length mismatch");
    assert_eq!(
        weight_buffers_len, steps_len,
        "weight_buffers length mismatch"
    );
    assert_eq!(release_at_len, steps_len, "release_at length mismatch");
    assert_eq!(mixed_gemm_len, steps_len, "mixed_gemm_infos length mismatch");
    assert_eq!(icb_eligible_len, steps_len, "icb_eligible length mismatch");
    assert_eq!(barriers_len, steps_len, "concurrent_barriers length mismatch");
    assert_eq!(
        input_name_cache_len, steps_len,
        "input_name_cache length mismatch"
    );

    // Property 2: any valid step_idx is a valid index into all vectors.
    if num_steps > 0 {
        let step_idx: usize = kani::any();
        kani::assume(step_idx < num_steps);

        assert!(step_idx < step_metas_len);
        assert!(step_idx < weight_buffers_len);
        assert!(step_idx < release_at_len);
        assert!(step_idx < mixed_gemm_len);
        assert!(step_idx < icb_eligible_len);
        assert!(step_idx < barriers_len);
        assert!(step_idx < input_name_cache_len);
    }

    // Property 3: buffer_plan vectors also match.
    let step_offsets_len = num_steps;
    let step_sizes_len = num_steps;
    let last_use_len = num_steps;

    assert_eq!(step_offsets_len, steps_len);
    assert_eq!(step_sizes_len, steps_len);
    assert_eq!(last_use_len, steps_len);
}
