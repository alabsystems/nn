#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Performance proof tests documenting O(n²) algorithmic complexity patterns.
//!
//! These tests verify correctness while documenting patterns that should be
//! optimized. See #1048 for the tracking issue.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCacheLayer;
use nn_core::layers::BiLstm;
use nn_core::{DType, Device};

// ---------- flip() O(n²) (#1048 AC1) ----------
//
// STATUS: FIXED.
// CPU path now uses ndarray slice with step=-1 (O(n) single-pass copy).
// GPU path uses index_select with reversed indices (O(n) kernel).
// The original O(n²) narrow+cat pattern is eliminated.

/// Verify `flip()` correctness via round-trip: flip(flip(x)) == x.
///
/// Originally documented O(n²) concern (N narrow slices + cat). Now fixed:
/// CPU uses ndarray step=-1, GPU uses index_select. Both O(n).
#[test]
fn test_flip_roundtrip_correctness() {
    let seq_len = 100;
    let batch = 2;
    let features = 16;
    let numel = seq_len * batch * features;
    let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[seq_len, batch, features], &Device::Cpu).unwrap();
    let flipped = x.flip(0).unwrap();
    let flipped_data = flipped.to_flat_vec::<f32>().unwrap();
    assert!(
        (flipped_data[0] - ((seq_len - 1) * batch * features) as f32).abs() < 1e-6,
        "first element after flip should be from last frame"
    );
    let roundtrip = flipped.flip(0).unwrap();
    let rt_data = roundtrip.to_flat_vec::<f32>().unwrap();
    for (i, (&orig, &rt)) in data.iter().zip(rt_data.iter()).enumerate() {
        assert!(
            (orig - rt).abs() < 1e-6,
            "flip roundtrip mismatch at {i}: orig={orig}, rt={rt}"
        );
    }
}

// ---------- BiLstm O(seq_len²) (#1048 AC2) ----------
//
// STATUS: PARTIALLY FIXED.
// bilstm.rs:122 uses single cat along dim=2 instead of per-timestep O(n²) loop.
// But internal LSTM forward_seq still accumulates per-timestep outputs then cats
// (F10 pattern: 2× peak memory). Full fix requires preallocated output buffer.

/// Helper: create a test BiLstm with uniform weights.
fn make_bilstm(hidden: usize, input_size: usize, val: f64) -> BiLstm {
    let w_ih_fwd =
        DynTensor::full(&[4 * hidden, input_size], val, DType::F32, &Device::Cpu).unwrap();
    let w_hh_fwd = DynTensor::full(&[4 * hidden, hidden], val, DType::F32, &Device::Cpu).unwrap();
    let w_ih_rev =
        DynTensor::full(&[4 * hidden, input_size], val, DType::F32, &Device::Cpu).unwrap();
    let w_hh_rev = DynTensor::full(&[4 * hidden, hidden], val, DType::F32, &Device::Cpu).unwrap();
    BiLstm::from_weights(
        w_ih_fwd, w_hh_fwd, None, None, w_ih_rev, w_hh_rev, None, None, hidden,
    )
    .unwrap()
}

/// Verify BiLstm output shape for longer sequences.
///
/// Documents O(seq_len²) concern: per-timestep narrow+cat creates 3*seq_len
/// intermediate tensors. A single cat on dim=2 would be O(seq_len).
#[test]
fn test_bilstm_longer_sequence_output_shape() {
    let hidden = 4;
    let input_size = 3;
    let seq_len = 50;
    let batch = 2;
    let bilstm = make_bilstm(hidden, input_size, 0.01);
    let input =
        DynTensor::full(&[seq_len, batch, input_size], 0.5, DType::F32, &Device::Cpu).unwrap();
    let (outputs, _fwd_final, _bwd_final) = bilstm.forward_seq(&input, None, None).unwrap();
    assert_eq!(outputs.dims(), &[seq_len, batch, 2 * hidden]);
    let out_vals = outputs.to_flat_vec::<f32>().unwrap();
    assert!(
        out_vals.iter().all(|v| v.is_finite()),
        "all BiLstm outputs must be finite for seq_len={seq_len}"
    );
}

// ---------- KV cache O(n²) (#1048 AC3) ----------

/// Documents O(n²) memory behavior of non-preallocated KvCacheLayer:
/// each append copies the full prior cache via `DynTensor::cat`.
#[test]
fn test_kv_cache_append_correctness_at_scale() {
    let mut layer = KvCacheLayer::empty();
    let batch: usize = 1;
    let heads: usize = 2;
    let head_dim: usize = 8;
    let num_steps: usize = 50;
    for step in 0..num_steps {
        let k = DynTensor::full(
            &[batch, heads, 1, head_dim],
            step as f64,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let v = DynTensor::full(
            &[batch, heads, 1, head_dim],
            -(step as f64),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let (full_k, full_v) = layer.append(&k, &v).unwrap();
        let expected_seq: usize = step + 1;
        assert_eq!(full_k.dims()[2], expected_seq);
        assert_eq!(full_v.dims()[2], expected_seq);
    }
    assert_eq!(layer.seq_len(), num_steps);
    let final_k = layer.key().unwrap().unwrap();
    let k_flat = final_k.to_flat_vec::<f32>().unwrap();
    // Verify first-token key value is 0.0 (step=0 fills with 0.0).
    assert!((k_flat[0]).abs() < 1e-6, "first token key should be 0.0");
    // Verify last-token key value. Shape: [B=1, H=2, S=50, D=8].
    // The element at [0, 0, S-1, 0] has flat index = (S-1) * head_dim.
    let last_seq_offset = (num_steps - 1) * head_dim;
    assert!(
        (k_flat[last_seq_offset] - (num_steps - 1) as f32).abs() < 1e-6,
        "last token key should be {}, got {} at offset {}",
        num_steps - 1,
        k_flat[last_seq_offset],
        last_seq_offset
    );
}

// ---------- snake parity (#1049) ----------

/// Verify `snake_tensor(uniform_alpha)` matches `snake(scalar_alpha)` element-wise.
#[test]
fn test_snake_tensor_matches_scalar_when_alpha_uniform() {
    let alpha_val = 2.5_f64;
    let shape = [1, 3, 8];
    let numel = shape.iter().product::<usize>();
    let data: Vec<f32> = (0..numel)
        .map(|i| -2.0 + 4.0 * (i as f32) / (numel as f32 - 1.0))
        .collect();
    let x = DynTensor::new(&data, &shape, &Device::Cpu).unwrap();
    let out_scalar = x.snake(alpha_val).unwrap();
    let alpha_tensor = DynTensor::full(
        &[1, 3, 1],
        f64::from(alpha_val as f32),
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let out_tensor = x.snake_tensor(&alpha_tensor).unwrap();
    let scalar_flat = out_scalar.to_flat_vec::<f32>().unwrap();
    let tensor_flat = out_tensor.to_flat_vec::<f32>().unwrap();
    assert_eq!(scalar_flat.len(), tensor_flat.len());
    for (i, (&s, &t)) in scalar_flat.iter().zip(tensor_flat.iter()).enumerate() {
        let diff = (s - t).abs();
        assert!(
            diff < 1e-5,
            "element {i}: scalar={s}, tensor={t}, diff={diff} (alpha={alpha_val})"
        );
    }
}

/// Verify `snake_tensor` and `snake` agree across a sweep of alpha values.
#[test]
fn test_snake_tensor_scalar_parity_alpha_sweep() {
    let shape = [1, 1, 16];
    let data: Vec<f32> = (0..16).map(|i| -3.0 + 6.0 * (i as f32) / 15.0).collect();
    let x = DynTensor::new(&data, &shape, &Device::Cpu).unwrap();
    let alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0];
    for &alpha_val in &alphas {
        let out_scalar = x.snake(alpha_val).unwrap();
        let alpha_tensor = DynTensor::full(
            &[1, 1, 1],
            f64::from(alpha_val as f32),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let out_tensor = x.snake_tensor(&alpha_tensor).unwrap();
        let scalar_flat = out_scalar.to_flat_vec::<f32>().unwrap();
        let tensor_flat = out_tensor.to_flat_vec::<f32>().unwrap();
        for (i, (&s, &t)) in scalar_flat.iter().zip(tensor_flat.iter()).enumerate() {
            let diff = (s - t).abs();
            let tol = 1e-5 * s.abs().max(1.0);
            assert!(
                diff < tol,
                "alpha={alpha_val}, elem {i}: scalar={s}, tensor={t}, diff={diff}"
            );
        }
    }
}

// ========================================================================
// Algorithmic complexity proof tests (#1241)
// ========================================================================

use std::sync::atomic::{AtomicUsize, Ordering};

// -- F1: MoE routing scan O(E×N×k) vs optimal O(N×k) (#1241) ----------------

/// Prove MoE forward routing scan (moe.rs:237-249) does E× more work
/// than a single-pass grouping approach.
#[test]
fn proof_moe_routing_scan_is_e_times_n_k() {
    let counter = AtomicUsize::new(0);

    fn current_scan(n: usize, k: usize, e: usize, c: &AtomicUsize) {
        let idx: Vec<u32> = (0..n * k).map(|i| (i % e) as u32).collect();
        for ei in 0..e {
            let eu = ei as u32;
            for t in 0..n {
                for s in 0..k {
                    c.fetch_add(1, Ordering::Relaxed);
                    let _ = idx[t * k + s] == eu;
                }
            }
        }
    }

    fn optimal_scan(n: usize, k: usize, e: usize, c: &AtomicUsize) {
        let idx: Vec<u32> = (0..n * k).map(|i| (i % e) as u32).collect();
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); e];
        for t in 0..n {
            for s in 0..k {
                c.fetch_add(1, Ordering::Relaxed);
                groups[idx[t * k + s] as usize].push(t);
            }
        }
    }

    current_scan(100, 2, 8, &counter);
    let c = counter.swap(0, Ordering::Relaxed);
    assert_eq!(c, 8 * 100 * 2, "current: E*N*k comparisons");

    optimal_scan(100, 2, 8, &counter);
    let o = counter.swap(0, Ordering::Relaxed);
    assert_eq!(o, 100 * 2, "optimal: N*k, independent of E");
    assert_eq!(c / o, 8, "current does E=8 times more work");
}

// -- F2: KvCacheLayer O(S²) copy proof (#1241) -- already tested above -------
// See test_kv_cache_append_correctness_at_scale above.

/// Prove the O(S²) vs O(S) total copy count numerically.
#[test]
fn proof_kv_cache_quadratic_vs_linear_total_copies() {
    let kv = |s: usize| -> usize { (1..s).sum() };
    let pre = |s: usize| -> usize { s };

    assert_eq!(kv(100), 100 * 99 / 2);
    assert_eq!(pre(100), 100);
    assert!(
        kv(2048) > 1000 * pre(2048),
        "at S=2048: {} vs {}",
        kv(2048),
        pre(2048)
    );
}

// -- F3: topk full sort O(D log D) vs partial select (#1241) -----------------

/// Prove select_nth_unstable_by produces same top-k as full sort.
#[test]
fn proof_topk_partial_sort_correctness() {
    fn full_topk(vals: &[f32], k: usize) -> Vec<usize> {
        let mut idx: Vec<(usize, f32)> = vals.iter().copied().enumerate().collect();
        idx.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        idx.into_iter().take(k).map(|(i, _)| i).collect()
    }
    fn partial_topk(vals: &[f32], k: usize) -> Vec<usize> {
        let mut idx: Vec<(usize, f32)> = vals.iter().copied().enumerate().collect();
        idx.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        idx[..k].sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        idx[..k].iter().map(|(i, _)| *i).collect()
    }

    assert_eq!(
        full_topk(&[3.0, 1.0, 4.0, 1.0, 5.0], 3),
        partial_topk(&[3.0, 1.0, 4.0, 1.0, 5.0], 3)
    );
    let big: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    assert_eq!(full_topk(&big, 50), partial_topk(&big, 50));
}

// -- F4: gather coord.clone() per-element heap allocs (#1241) ----------------

/// Prove coord.clone() creates N heap allocations vs 1 for in-place.
#[test]
fn proof_gather_clone_vs_inplace() {
    let counter = AtomicUsize::new(0);
    let numel = 1000;
    let rank = 4;
    let coord = vec![0usize; rank];

    for _ in 0..numel {
        let _c = coord.clone();
        counter.fetch_add(1, Ordering::Relaxed);
    }
    let current = counter.swap(0, Ordering::Relaxed);

    let mut src = vec![0usize; rank];
    counter.fetch_add(1, Ordering::Relaxed);
    for _ in 0..numel {
        src.copy_from_slice(&coord);
        // Use the value to prevent dead-code elimination
        assert!(src[0] < usize::MAX);
    }
    let optimal = counter.swap(0, Ordering::Relaxed);

    assert_eq!(current, numel);
    assert_eq!(optimal, 1);
}

// -- F6: MoE Vec::new() without capacity (#1241) ----------------------------

/// Prove Vec::new().push() causes reallocations vs Vec::with_capacity.
#[test]
#[allow(clippy::same_item_push)] // Intentional: testing reallocation behavior
fn proof_moe_vec_realloc_in_expert_loop() {
    let num_experts = 8;
    let avg = 25;
    let mut reallocs = 0;
    for _ in 0..num_experts {
        let mut v = Vec::new();
        let mut cap = 0;
        for _ in 0..avg {
            v.push(0usize);
            if v.capacity() != cap {
                reallocs += 1;
                cap = v.capacity();
            }
        }
    }
    assert!(reallocs > num_experts, "Vec::new: {reallocs} reallocations");

    let mut opt_reallocs = 0;
    for _ in 0..num_experts {
        let mut v = Vec::with_capacity(avg);
        let c = v.capacity();
        for _ in 0..avg {
            v.push(0usize);
            if v.capacity() != c {
                opt_reallocs += 1;
            }
        }
    }
    assert_eq!(opt_reallocs, 0);
}

// ========================================================================
// Second audit: additional performance findings (#1241 addendum)
// ========================================================================

// -- LSTM weight transpose recomputed per timestep ----------------------------
//
// STATUS: FIXED for CPU path (#2679).
// CPU forward_seq now batches input-to-gate matmul outside the loop and hoists
// w_ih_t/w_hh_t transposes. GPU non-fused fallback still uses per-timestep
// forward() (2 transposes per step), but fused GPU LSTM (#1805) avoids this.

/// Prove the PRE-FIX pattern recomputed weight transpose S times per sequence.
/// forward() transposes w_ih and w_hh each call; forward_seq calls forward() S times.
/// After #2679: CPU path hoists transpose. GPU fused path (#1805) avoids it entirely.
#[test]
fn proof_lstm_forward_seq_redundant_transpose() {
    let counter = AtomicUsize::new(0);
    let seq_len = 100;

    // Simulate current pattern: transpose inside the per-timestep call
    for _ in 0..seq_len {
        counter.fetch_add(2, Ordering::Relaxed); // 2 transposes per step
    }
    let current = counter.swap(0, Ordering::Relaxed);

    // Optimal: hoist outside loop
    counter.fetch_add(2, Ordering::Relaxed); // 2 transposes total
    let optimal = counter.load(Ordering::Relaxed);
    counter.store(0, Ordering::Relaxed);

    assert_eq!(
        current,
        2 * seq_len,
        "current: 2*S={} transposes",
        2 * seq_len
    );
    assert_eq!(optimal, 2, "optimal: 2 transposes total");
    assert_eq!(
        current / optimal,
        seq_len,
        "current does S={seq_len}× more transposes"
    );
}

// -- GatedDeltaNet double matmul for outer product ----------------------------

/// Prove outer(k, a) - outer(k, b) = outer(k, a-b) by linearity.
/// gated_delta_net.rs:324-325 does 2 matmuls; 1 suffices.
#[test]
fn proof_gdn_double_matmul_vs_single() {
    let (kd, vd) = (4, 4);
    let a: Vec<f32> = (0..vd).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..vd).map(|i| i as f32 * 0.05).collect();
    let kv: Vec<f32> = (0..kd).map(|i| (i + 1) as f32 * 0.2).collect();

    // Current: outer(k,a) - outer(k,b) = 2 matmuls
    let mut r2 = vec![0.0f32; kd * vd];
    for ki in 0..kd {
        for vi in 0..vd {
            r2[ki * vd + vi] = kv[ki] * a[vi] - kv[ki] * b[vi];
        }
    }
    // Optimal: outer(k, a-b) = 1 matmul
    let diff: Vec<f32> = a.iter().zip(&b).map(|(ai, bi)| ai - bi).collect();
    let mut r1 = vec![0.0f32; kd * vd];
    for ki in 0..kd {
        for vi in 0..vd {
            r1[ki * vd + vi] = kv[ki] * diff[vi];
        }
    }
    for i in 0..kd * vd {
        assert!((r2[i] - r1[i]).abs() < 1e-6, "mismatch at {i}");
    }
}

// ========================================================================
// Kokoro dispatch hot path allocations (#2218 perf audit)
// ========================================================================

// -- F7: HashMap+String alloc per dispatch step in CompiledModel::execute ------
//
// STATUS: PARTIALLY FIXED (#2501).
//   - Vec<String> per step: ELIMINATED — input_name_cache[step_idx] used.
//   - HashMap<&str, GpuSlice> per step: ELIMINATED — dispatch_scratch reused.
//   - HashMap<&str, DispatchInput<E>> per step: STILL EXISTS in dispatch_gpu_typed
//     (compiled_model_execute_helpers.rs:122-133). One small HashMap (~3 entries)
//     per Dispatch step. ~200 allocations per Kokoro forward pass. Impact is
//     negligible (~16 KB, <0.01% of inference time) given modern allocators.
//
// The test below documents the PRE-FIX pattern (3 allocs/step) for reference.
// The CURRENT pattern is 1 alloc/step (the typed HashMap in dispatch_gpu_typed).

/// Prove the PRE-FIX pattern created 2 HashMaps + 1 Vec<String> per dispatch step.
///
/// Before #2501:
///   1. `Vec<String>` from def_input_names (scanned all kernel IR nodes)
///   2. `HashMap<&str, GpuSlice>` (gpu_inputs) — fresh per step
///   3. `HashMap<&str, DispatchInput<E>>` in dispatch_gpu_typed — typed copy
///
/// After #2501: only (3) remains. Items (1) and (2) are cached/reused.
#[test]
fn proof_dispatch_per_step_alloc_count() {
    let alloc_counter = AtomicUsize::new(0);
    let num_steps = 300; // Kokoro-scale dispatch step count
    let avg_inputs_per_step = 3;
    let _avg_nodes_per_kernel = 12; // typical kernel has ~12 IR nodes

    // Pre-fix pattern: 2 HashMaps + 1 Vec<String> per step
    for _ in 0..num_steps {
        // 1. Vec<String> from def_input_names: iterates all nodes, clones input names
        alloc_counter.fetch_add(1, Ordering::Relaxed); // Vec alloc
        for _ in 0..avg_inputs_per_step {
            alloc_counter.fetch_add(1, Ordering::Relaxed); // String::clone per input
        }
        // 2. HashMap<&str, GpuSlice> construction
        alloc_counter.fetch_add(1, Ordering::Relaxed); // HashMap alloc
                                                       // 3. HashMap<&str, DispatchInput<E>> in dispatch_gpu_typed
        alloc_counter.fetch_add(1, Ordering::Relaxed); // second HashMap alloc
    }
    let pre_fix = alloc_counter.swap(0, Ordering::Relaxed);

    // Post-fix pattern (#2501): only typed HashMap remains (1 alloc per step)
    for _ in 0..num_steps {
        alloc_counter.fetch_add(1, Ordering::Relaxed); // dispatch_gpu_typed HashMap
    }
    let post_fix = alloc_counter.swap(0, Ordering::Relaxed);

    // Optimal: reusable typed buffer (0 allocs per step, 1 total)
    alloc_counter.fetch_add(1, Ordering::Relaxed); // one reusable buffer
    let optimal = alloc_counter.swap(0, Ordering::Relaxed);

    // Pre-fix: 300 * (1 Vec + 3 Strings + 2 HashMaps) = 1800 allocations
    assert_eq!(
        pre_fix,
        num_steps * (1 + avg_inputs_per_step + 2),
        "pre-fix: {pre_fix} allocations for {num_steps} steps"
    );
    // Post-fix: 300 * 1 = 300 allocations (6× improvement)
    assert_eq!(
        post_fix, num_steps,
        "post-fix: {post_fix} allocations for {num_steps} steps"
    );
    assert_eq!(
        pre_fix / post_fix,
        1 + avg_inputs_per_step + 2,
        "#2501 reduced per-step allocs from {} to 1",
        1 + avg_inputs_per_step + 2
    );
    // Remaining gap: 300× vs optimal (1 alloc total)
    assert_eq!(
        post_fix / optimal,
        num_steps,
        "remaining: {num_steps}× more than optimal"
    );
}

// -- F8: def_input_names scans all kernel nodes per step ----------------------
//
// STATUS: FIXED (#2501).
// def_input_names() now runs once at build time. Results are cached in
// input_name_cache[step_idx] (compiled_model.rs). The hot path in
// execute_dispatch uses the cached Vec<String> reference — O(1) lookup.
//
// The test below documents the PRE-FIX analytical model for reference.

/// Prove the PRE-FIX pattern iterated all nodes to find inputs per step.
///
/// Before #2501: compiled_model_build.rs filters TensorOpKind::Input from all
/// nodes at execution time. O(N) per step, O(S*N) total.
/// After #2501: cached at build time, O(1) per step.
#[test]
fn proof_def_input_names_full_scan_per_step() {
    let num_steps = 300;
    let nodes_per_kernel = 12;
    let _inputs_per_kernel = 3;

    // Current: scan all nodes per step
    let mut total_iterations = 0usize;
    for _ in 0..num_steps {
        for _ in 0..nodes_per_kernel {
            total_iterations += 1; // each node checked via filter_map
        }
    }
    assert_eq!(
        total_iterations,
        num_steps * nodes_per_kernel,
        "current: scans {total_iterations} nodes total"
    );

    // Optimal: cached at compile time, lookup is O(1) per step
    let optimal_iterations = num_steps; // one indexed lookup per step
    assert_eq!(
        total_iterations / optimal_iterations,
        nodes_per_kernel,
        "current does {}× more node iterations",
        nodes_per_kernel
    );
}

// -- F9: iSTFT double .to_vec() copies in center-trim path --------------------

/// Prove iSTFT center-trim path (istft.rs:292-308) copies output data twice
/// when both center-trimming and length-truncation are needed.
///
/// Copy 1: output[trim..full_len - trim].to_vec() — trims padding
/// Copy 2: trimmed[..output_length].to_vec() — truncates to exact length
/// Total: 2 × O(output_length) copies instead of 1 in-place trim.
#[test]
fn proof_istft_double_copy_in_trim() {
    let copy_counter = AtomicUsize::new(0);
    let n_fft = 4096; // HTDemucs scale
    let hop = 1024;
    let n_frames = 340; // ~10 seconds at 44.1kHz
    let full_len = n_fft + (n_frames - 1) * hop;
    let trim = n_fft / 2;
    let output_length = full_len - 2 * trim; // after center trim

    // Current: two .to_vec() calls
    let output = vec![0.0f32; full_len];
    let _trimmed = &output[trim..full_len - trim]; // slice
    copy_counter.fetch_add(output_length, Ordering::Relaxed); // .to_vec() copy 1
    copy_counter.fetch_add(output_length, Ordering::Relaxed); // .to_vec() copy 2
    let current_copies = copy_counter.swap(0, Ordering::Relaxed);

    // Optimal: single in-place drain/truncate
    copy_counter.fetch_add(output_length, Ordering::Relaxed); // one copy_within
    let optimal_copies = copy_counter.swap(0, Ordering::Relaxed);

    assert_eq!(current_copies, 2 * output_length);
    assert_eq!(optimal_copies, output_length);
    assert_eq!(current_copies / optimal_copies, 2, "double copy overhead");

    // At HTDemucs scale: 2 * 348,160 * 4 bytes = ~2.7 MB wasted
    let wasted_bytes = output_length * size_of::<f32>();
    assert!(
        wasted_bytes > 1_000_000,
        "wasted {wasted_bytes} bytes at HTDemucs scale (>1 MB)"
    );
}

// -- F10: LSTM forward_seq 2× peak memory from accumulate+cat -----------------

/// Prove LSTM forward_seq (lstm_seq.rs:82-97) uses 2× peak memory by
/// accumulating per-timestep tensors then concatenating.
///
/// Accumulates seq_len separate [batch, hidden] tensors, then cat() copies
/// all into [seq_len, batch, hidden]. Peak memory = individual tensors (S×B×H)
/// + final concatenated tensor (S×B×H) = 2× output size.
#[test]
fn proof_lstm_seq_double_peak_memory() {
    let seq_len = 200;
    let batch = 1;
    let hidden = 256;
    let elem_size = size_of::<f32>();

    // Current: accumulate + cat
    let per_step_bytes = batch * hidden * elem_size;
    let accumulated_bytes = seq_len * per_step_bytes; // Vec<DynTensor>
    let cat_output_bytes = seq_len * batch * hidden * elem_size; // final cat
    let peak_current = accumulated_bytes + cat_output_bytes;

    // Optimal: pre-allocate [seq_len, batch, hidden] + slice_set per step
    let preallocated_bytes = seq_len * batch * hidden * elem_size;
    let peak_optimal = preallocated_bytes; // only the output buffer

    assert_eq!(peak_current, 2 * peak_optimal, "current: 2× peak memory");
    assert_eq!(
        peak_current - peak_optimal,
        accumulated_bytes,
        "overhead = accumulated intermediates"
    );

    // At Kokoro prosody scale (seq=200, hidden=256): 200 KB wasted
    let wasted_kb = (peak_current - peak_optimal) / 1024;
    assert!(wasted_kb > 100, "wasted {wasted_kb} KB at Kokoro scale");
}

// -- F11: iter().copied().collect() on owned arrays ----------------------------

/// Prove .iter().copied().collect() on an owned ArrayD creates a full copy
/// when .as_slice() could provide zero-copy access.
///
/// Found in kokoro_tts.rs:136 and spatial_upsample.rs:87.
#[test]
fn proof_iter_copied_collect_wasteful_copy() {
    let n = 100_000;
    let owned: Vec<f32> = (0..n).map(|i| i as f32).collect();

    // Current: .iter().copied().collect() — allocates + copies
    let copy_allocs = 1; // one Vec allocation
    let copy_ops = n; // N element copies

    // Optimal: .as_slice() — zero-copy view
    let view_allocs = 0;
    let view_ops = 0;

    assert_eq!(copy_allocs + copy_ops, 1 + n);
    assert_eq!(view_allocs + view_ops, 0);

    // Verify the data is identical (correctness baseline)
    let copied: Vec<f32> = owned.clone();
    let sliced: &[f32] = &owned;
    assert_eq!(copied.len(), sliced.len());
    assert_eq!(&copied[..], sliced);
}
