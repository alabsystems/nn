// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RSS measurement tests for Kokoro model inference.
//!
//! Measures process RSS at key pipeline stages and reports results as JSON
//! to stdout for programmatic collection by `scripts/measure_rss.py`.
//!
//! - `test_small_matmul_rss`: baseline CPU matmul RSS (no weights needed).
//! - `test_kokoro_mini_rss_measurement`: synthetic weight map allocation
//!    sized to match Kokoro's layer structure (no production weights).
//! - `test_kokoro_production_rss_measurement`: full production model load +
//!    synthesis (requires `KOKORO_WEIGHTS` env var; skips gracefully without it).
//!
//! Run:
//! ```bash
//! cargo test -p nn-metal --test kokoro_all -- rss_measurement --nocapture
//! ```
//!
//! Part of #3211.

use std::collections::HashMap;
use std::time::Instant;

use nn_core::dyn_tensor::DynTensor;
use nn_core::memory_stats::MemorySnapshot;
use nn_core::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

/// Emit a JSON measurement line to stdout for collection.
fn emit_json(framework: &str, model_name: &str, peak_rss_mb: f64, inference_time_ms: f64) {
    println!(
        r#"{{"framework":"{}","model_name":"{}","peak_rss_mb":{:.1},"inference_time_ms":{:.2}}}"#,
        framework, model_name, peak_rss_mb, inference_time_ms,
    );
}

// -- Tests: small matmul RSS baseline ------------------------------------------

#[test]
fn test_small_matmul_rss() {
    let snap_before = MemorySnapshot::capture().expect("memory snapshot");

    // Allocate two moderate matrices and multiply on CPU.
    let a = DynTensor::full(&[512, 512], 0.01, DType::F32, &cpu()).unwrap();
    let b = DynTensor::full(&[512, 512], 0.01, DType::F32, &cpu()).unwrap();

    let t0 = Instant::now();
    let _c = a.matmul(&b).unwrap();
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let snap_after = MemorySnapshot::capture().expect("memory snapshot");

    let rss_delta_mb =
        (snap_after.current_rss as f64 - snap_before.current_rss as f64) / (1024.0 * 1024.0);

    eprintln!(
        "matmul 512x512: RSS delta {:.1} MB, peak {:.1} MB, time {:.2} ms",
        rss_delta_mb,
        snap_after.peak_rss_mb(),
        elapsed_ms,
    );

    emit_json(
        "nn_cpu",
        "matmul_512x512",
        snap_after.peak_rss_mb(),
        elapsed_ms,
    );

    // Sanity: a 512x512 f32 matmul should not use more than 100 MB of RSS growth.
    assert!(
        rss_delta_mb < 100.0,
        "matmul RSS growth too large: {rss_delta_mb:.1} MB"
    );
}

// -- Tests: mini Kokoro weight loading RSS -------------------------------------

#[test]
fn test_kokoro_mini_rss_measurement() {
    // Allocate a weight map sized similarly to Kokoro's layer structure.
    // This measures RSS of DynTensor weight allocation without requiring
    // the full model construction (which needs hundreds of matching weights).
    //
    // Kokoro-82M has ~82M parameters at f32 = ~328 MB raw weight data.
    // This mini test allocates ~200 tensors totaling ~10 MB to measure
    // the per-tensor overhead and HashMap bookkeeping.

    let snap_before = MemorySnapshot::capture().expect("memory snapshot");

    let t0 = Instant::now();
    let mut weights: HashMap<String, DynTensor> = HashMap::new();

    // Simulate encoder layers (attention + FFN weights).
    for layer_idx in 0..4 {
        let prefix = format!("encoder.layers.{layer_idx}");
        for suffix in [
            "attention.self.query.weight",
            "attention.self.query.bias",
            "attention.self.key.weight",
            "attention.self.key.bias",
            "attention.self.value.weight",
            "attention.self.value.bias",
            "attention.output.dense.weight",
            "attention.output.dense.bias",
            "intermediate.dense.weight",
            "intermediate.dense.bias",
            "output.dense.weight",
            "output.dense.bias",
        ] {
            let name = format!("{prefix}.{suffix}");
            let shape = if suffix.ends_with("bias") {
                vec![128]
            } else {
                vec![128, 128]
            };
            weights.insert(
                name,
                DynTensor::zeros(&shape, DType::F32, &cpu()).unwrap(),
            );
        }
    }

    // Simulate decoder conv layers.
    for ch_idx in 0..8 {
        let name = format!("decoder.conv.{ch_idx}.weight");
        weights.insert(
            name,
            DynTensor::zeros(&[256, 128, 3], DType::F32, &cpu()).unwrap(),
        );
        let name = format!("decoder.conv.{ch_idx}.bias");
        weights.insert(
            name,
            DynTensor::zeros(&[256], DType::F32, &cpu()).unwrap(),
        );
    }

    // Simulate LSTM weights (largest single tensors in Kokoro).
    for direction in ["forward", "reverse"] {
        for gate in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
            let name = format!("lstm.{direction}.{gate}");
            let shape = if gate.starts_with("weight") {
                vec![4 * 256, 256]
            } else {
                vec![4 * 256]
            };
            weights.insert(
                name,
                DynTensor::zeros(&shape, DType::F32, &cpu()).unwrap(),
            );
        }
    }

    let n_tensors = weights.len();
    let total_bytes: usize = weights.values().map(|t| t.numel() * 4).sum();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let snap_after = MemorySnapshot::capture().expect("memory snapshot");

    let rss_delta_mb =
        (snap_after.current_rss as f64 - snap_before.current_rss as f64) / (1024.0 * 1024.0);

    eprintln!(
        "kokoro mini weights: {n_tensors} tensors, {:.1} MB raw, RSS delta {:.1} MB, \
         peak {:.1} MB, time {:.2} ms",
        total_bytes as f64 / (1024.0 * 1024.0),
        rss_delta_mb,
        snap_after.peak_rss_mb(),
        load_ms,
    );

    emit_json(
        "nn_cpu",
        "kokoro_mini_weights",
        snap_after.peak_rss_mb(),
        load_ms,
    );

    // Mini weight set should be < 50 MB RSS growth.
    assert!(
        rss_delta_mb < 50.0,
        "kokoro mini RSS growth too large: {rss_delta_mb:.1} MB"
    );
}

// -- Tests: production Kokoro RSS (requires KOKORO_WEIGHTS) --------------------

#[test]
fn test_kokoro_production_rss_measurement() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "production RSS measurement needs weights",
    ) {
        Some(p) => p,
        None => return,
    };

    let snap_start = MemorySnapshot::capture().expect("memory snapshot");

    // Initialize Metal backend.
    let _backend = nn_metal::MetalBackend::init().expect("Metal init");
    let cache = nn_metal::PipelineCache::new_global().expect("PipelineCache");

    let snap_metal = MemorySnapshot::capture().expect("memory snapshot");

    // Use Warn policy: test tokens [0..7] produce click artifacts with
    // production weights that fail the no_clicks hard bound. Part of #4262.
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // Load production model.
    let mut kokoro = unsafe {
        nn_metal::CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("CompiledKokoro::load")
    }
    .with_auto_release_weights();

    let snap_loaded = MemorySnapshot::capture().expect("memory snapshot");

    // Synthetic input for a short synthesis.
    let input_ids =
        DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu())
            .expect("input_ids");
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).expect("style");

    // First synthesis (cold).
    let t0 = Instant::now();
    let (audio, _cert, _diag) = kokoro
        .synthesize_with_memory(&input_ids, &style, 1.0, &cache)
        .expect("synthesis");
    let synth_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let snap_synth = MemorySnapshot::capture().expect("memory snapshot");

    let audio_samples = audio.numel();
    let metal_alloc = nn_metal::rss::metal_allocated_bytes();

    eprintln!("=== Production Kokoro RSS Measurement ===");
    eprintln!(
        "  process start:    RSS {:.1} MB",
        snap_start.current_rss_mb()
    );
    eprintln!(
        "  after Metal init: RSS {:.1} MB (+{:.1})",
        snap_metal.current_rss_mb(),
        (snap_metal.current_rss as f64 - snap_start.current_rss as f64) / (1024.0 * 1024.0),
    );
    eprintln!(
        "  after model load: RSS {:.1} MB (+{:.1})",
        snap_loaded.current_rss_mb(),
        (snap_loaded.current_rss as f64 - snap_metal.current_rss as f64) / (1024.0 * 1024.0),
    );
    eprintln!(
        "  after synthesis:  RSS {:.1} MB (+{:.1})",
        snap_synth.current_rss_mb(),
        (snap_synth.current_rss as f64 - snap_loaded.current_rss as f64) / (1024.0 * 1024.0),
    );
    eprintln!(
        "  peak RSS:         {:.1} MB",
        snap_synth.peak_rss_mb()
    );
    if let Some(metal) = metal_alloc {
        eprintln!(
            "  Metal allocated:  {:.1} MB",
            metal as f64 / (1024.0 * 1024.0)
        );
    }
    eprintln!("  audio samples:    {audio_samples}");
    eprintln!("  synthesis time:   {synth_ms:.2} ms");

    emit_json(
        "nn_metal",
        "kokoro_production",
        snap_synth.peak_rss_mb(),
        synth_ms,
    );
}
