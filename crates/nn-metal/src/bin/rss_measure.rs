// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RSS measurement binary for validating #3079 memory optimizations.
//!
//! Loads Kokoro from safetensors weights, runs `synthesize_with_memory()`,
//! and prints per-stage RSS checkpoints with deltas.
//!
//! Compiles with `--no-default-features` to bypass the NY dependency.
//!
//! # Usage
//!
//! ```bash
//! KOKORO_WEIGHTS=~/path/to/kokoro_v1_0.safetensors \
//!   cargo run -p nn-metal --no-default-features --bin rss_measure
//! ```
//!
//! Part of #3079.

use std::process;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

fn main() {
    let weights_path = match std::env::var("KOKORO_WEIGHTS") {
        Ok(p) if !p.is_empty() && std::path::Path::new(&p).exists() => p,
        Ok(p) if !p.is_empty() => {
            eprintln!("KOKORO_WEIGHTS path does not exist: {p}");
            process::exit(1);
        }
        _ => {
            eprintln!("Set KOKORO_WEIGHTS to the path of kokoro_v1_0.safetensors");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  KOKORO_WEIGHTS=~/kokoro_v1_0.safetensors \\");
            eprintln!("    cargo run -p nn-metal --no-default-features --bin rss_measure");
            process::exit(1);
        }
    };

    // RSS before anything.
    let mut pre_rss = nn_metal::rss::RssTracker::new();
    pre_rss.checkpoint("process_start");

    // Initialize Metal backend.
    let _backend = nn_metal::MetalBackend::init().unwrap_or_else(|e| {
        eprintln!("Metal init failed: {e}");
        process::exit(1);
    });
    pre_rss.checkpoint("after_metal_init");

    let cache = nn_metal::PipelineCache::new_global().unwrap_or_else(|e| {
        eprintln!("PipelineCache::new_global failed: {e}");
        process::exit(1);
    });
    pre_rss.checkpoint("after_cache_init");

    // Load model.
    // SAFETY: safetensors file is not modified while alive.
    let mut kokoro = unsafe {
        nn_metal::CompiledKokoro::load(&weights_path).unwrap_or_else(|e| {
            eprintln!("CompiledKokoro::load failed: {e}");
            process::exit(1);
        })
    }
    .with_auto_release_weights();
    pre_rss.checkpoint("after_model_load");

    eprintln!("=== Pre-synthesis Memory ===");
    eprintln!("{pre_rss}");
    if let Some(budget) = nn_metal::rss::metal_budget_bytes() {
        eprintln!(
            "  Metal budget: {:.1} MB",
            budget as f64 / (1024.0 * 1024.0)
        );
    }
    eprintln!();

    // Synthetic input: 8 tokens, standard style vector.
    let cpu = Device::Cpu;
    let input_ids = DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu)
        .expect("input_ids");
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu).expect("style");
    let speed = 1.0;

    // First synthesis (cold — triggers compilation + weight upload).
    eprintln!("--- Synthesis 1 (cold) ---");
    let (audio, cert, diag) = kokoro
        .synthesize_with_memory(&input_ids, &style, speed, &cache)
        .unwrap_or_else(|e| {
            eprintln!("synthesize_with_memory failed: {e}");
            process::exit(1);
        });

    let audio_len = audio.numel();

    eprintln!("  audio samples: {audio_len}");
    eprintln!(
        "  certificate:   {}",
        if cert.overall_passed { "PASS" } else { "FAIL" }
    );
    eprintln!();
    eprintln!("{diag}");
    if let Some(rss) = &diag.rss {
        eprintln!("{rss}");
    }

    // Second synthesis (warm — cached segments, pool reuse).
    eprintln!();
    eprintln!("--- Synthesis 2 (warm) ---");
    let (_audio2, _cert2, diag2) = kokoro
        .synthesize_with_memory(&input_ids, &style, speed, &cache)
        .unwrap_or_else(|e| {
            eprintln!("synthesize_with_memory (warm) failed: {e}");
            process::exit(1);
        });

    eprintln!("{diag2}");
    if let Some(rss) = &diag2.rss {
        eprintln!("{rss}");
    }

    // Third synthesis (steady-state confirmation).
    eprintln!();
    eprintln!("--- Synthesis 3 (steady-state) ---");
    let (_audio3, _cert3, diag3) = kokoro
        .synthesize_with_memory(&input_ids, &style, speed, &cache)
        .unwrap_or_else(|e| {
            eprintln!("synthesize_with_memory (steady) failed: {e}");
            process::exit(1);
        });

    eprintln!("{diag3}");
    if let Some(rss) = &diag3.rss {
        eprintln!("{rss}");
    }

    // Final memory summary with per-domain breakdown (#3079 D7).
    let breakdown = kokoro.memory_breakdown();
    eprintln!();
    eprintln!("=== Final Memory Summary ===");
    eprintln!("{breakdown}");
    if let Some(budget) = nn_metal::rss::metal_budget_bytes() {
        eprintln!(
            "  Metal budget: {:>8.1} MB",
            budget as f64 / (1024.0 * 1024.0)
        );
    }
    eprintln!();
    eprintln!("Target: < 1900 MB RSS (baseline: 3728→1885 MB nn, 1639 MB PyTorch)");
}
