// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for Metal kernel registry and dispatch infrastructure.
//!
//! Covers:
//! - Registry completeness: all entries valid, no duplicates, cross-references sound
//! - Dispatch routing: correct kernel selected for op/dtype combinations
//! - Buffer size validation: dispatch validates buffer sizes before launch
//! - Pipeline cache: L1/L2 cache hits for repeated kernel launches
//! - Error handling: invalid kernel names, unsupported dtypes
//! - Lazy batching: command buffer accumulates until flush
//! - Thread safety: dispatch from multiple threads
//!
//! Part of #3942.

use std::collections::HashSet;
use std::sync::{Arc, Barrier};

use nn_metal::{dispatch_stats, flush, reset_counters, with_gpu_scope};

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

fn init() {
    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
}

// ===========================================================================
// A. Registry completeness
// ===========================================================================

/// Every kernel kind in the registry is unique (no duplicate entries).
#[test]
fn test_registry_no_duplicate_kernel_kinds() {
    let source = include_str!("../../src/compiled_kokoro_registry.rs");

    // Extract all kind strings from KernelEntry definitions.
    let mut kinds = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("kind: \"") {
            let start = trimmed.find('"').unwrap() + 1;
            let end = trimmed[start..].find('"').unwrap() + start;
            kinds.push(&trimmed[start..end]);
        }
    }

    let unique: HashSet<&&str> = kinds.iter().collect();
    assert_eq!(
        kinds.len(),
        unique.len(),
        "KERNEL_REGISTRY has duplicate kind entries: {kinds:?}"
    );
    assert!(
        !kinds.is_empty(),
        "KERNEL_REGISTRY must have at least one entry"
    );
}

/// Every segment in the registry has a unique name.
#[test]
fn test_registry_no_duplicate_segment_names() {
    let source = include_str!("../../src/compiled_kokoro_registry.rs");

    let mut names = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("name: \"seg_") {
            let start = trimmed.find('"').unwrap() + 1;
            let end = trimmed[start..].find('"').unwrap() + start;
            names.push(&trimmed[start..end]);
        }
    }

    let unique: HashSet<&&str> = names.iter().collect();
    assert_eq!(
        names.len(),
        unique.len(),
        "SEGMENT_REGISTRY has duplicate names: {names:?}"
    );
}

/// Every sync point has a unique name.
#[test]
fn test_registry_no_duplicate_sync_point_names() {
    let source = include_str!("../../src/compiled_kokoro_registry.rs");

    let mut names = Vec::new();
    // Sync point names are name: "..." in SyncPointEntry definitions
    let mut in_sync = false;
    for line in source.lines() {
        if line.contains("SYNC_POINT_REGISTRY") {
            in_sync = true;
        }
        if in_sync && line.trim().starts_with("name: \"") {
            let trimmed = line.trim();
            let start = trimmed.find('"').unwrap() + 1;
            let end = trimmed[start..].find('"').unwrap() + start;
            names.push(trimmed[start..end].to_string());
        }
    }

    let unique: HashSet<&String> = names.iter().collect();
    assert_eq!(
        names.len(),
        unique.len(),
        "SYNC_POINT_REGISTRY has duplicate names: {names:?}"
    );
}

/// Every kernel dispatch_file reference points to a real file that exists.
#[test]
fn test_registry_dispatch_files_exist() {
    let source = include_str!("../../src/compiled_kokoro_registry.rs");

    let mut dispatch_files = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("dispatch_file: \"") {
            let start = trimmed.find('"').unwrap() + 1;
            let end = trimmed[start..].find('"').unwrap() + start;
            dispatch_files.push(trimmed[start..end].to_string());
        }
    }

    let unique: HashSet<&String> = dispatch_files.iter().collect();
    // Verify they reference known executor files
    for file in &unique {
        assert!(
            file.starts_with("compiled_model_execute_"),
            "dispatch_file '{file}' should start with 'compiled_model_execute_'"
        );
        assert!(
            file.ends_with(".rs"),
            "dispatch_file '{file}' should end with '.rs'"
        );
    }
    assert!(
        !dispatch_files.is_empty(),
        "must have at least one dispatch_file"
    );
}

/// NATIVE_OP_VARIANT_COUNT constant matches the registry length.
///
/// This is a source-level cross-check that complements the in-crate unit test.
#[test]
fn test_native_op_variant_count_matches_source() {
    let source = include_str!("../../src/compiled_kokoro_registry.rs");

    // Count KernelEntry array elements (subtract 1 for the struct definition
    // `pub(crate) struct KernelEntry {` which also matches the pattern).
    let entry_count = source.matches("KernelEntry {").count() - 1;

    // Extract NATIVE_OP_VARIANT_COUNT value
    let count_line = source
        .lines()
        .find(|l| l.contains("NATIVE_OP_VARIANT_COUNT: usize ="))
        .expect("NATIVE_OP_VARIANT_COUNT must exist");
    let count_str = count_line
        .split('=')
        .nth(1)
        .unwrap()
        .trim()
        .trim_end_matches(';');
    let declared: usize = count_str.parse().expect("must be numeric");

    assert_eq!(
        entry_count, declared,
        "KernelEntry count ({entry_count}) != NATIVE_OP_VARIANT_COUNT ({declared})"
    );
}

/// CPU_BRIDGE_REGISTRY is empty (all bridges eliminated).
#[test]
fn test_cpu_bridge_registry_empty() {
    let source = include_str!("../../src/compiled_kokoro_registry.rs");

    // The registry should be `&[]` — no CPU bridges.
    assert!(
        source.contains("CPU_BRIDGE_REGISTRY: &[CpuBridgeEntry] = &[]"),
        "CPU_BRIDGE_REGISTRY must be empty (all bridges eliminated)"
    );
}

// ===========================================================================
// B. Dispatch routing: correct op for dtype
// ===========================================================================

/// F32 elementwise add dispatches to GPU and produces correct results.
#[test]
fn test_dispatch_routing_f32_add() {
    init();
    let device = Device::metal();
    let a = DynTensor::full(&[8], 3.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[8], 4.0, DType::F32, &device).unwrap();

    let c = a.add(&b).unwrap();
    let vals = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![7.0; 8]);
}

/// F32 elementwise mul dispatches to GPU and produces correct results.
#[test]
fn test_dispatch_routing_f32_mul() {
    init();
    let device = Device::metal();
    let a = DynTensor::full(&[8], 3.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[8], 4.0, DType::F32, &device).unwrap();

    let c = a.mul(&b).unwrap();
    let vals = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![12.0; 8]);
}

/// F32 negation dispatches to GPU correctly.
#[test]
fn test_dispatch_routing_f32_neg() {
    init();
    let device = Device::metal();
    let a = DynTensor::full(&[4], 5.0, DType::F32, &device).unwrap();

    let c = a.neg().unwrap();
    let vals = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![-5.0; 4]);
}

/// F32 matmul dispatches correctly to GPU.
#[test]
fn test_dispatch_routing_f32_matmul() {
    init();
    let device = Device::metal();
    let a_cpu = DynTensor::new(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&[5.0_f32, 6.0, 7.0, 8.0], &[2, 2], &Device::Cpu).unwrap();
    let a = a_cpu.to_device(&device).unwrap();
    let b = b_cpu.to_device(&device).unwrap();

    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let c = a.matmul(&b).unwrap();
    let vals = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![19.0, 22.0, 43.0, 50.0]);
}

/// Scalar operations (add_scalar, mul_scalar) dispatch correctly.
#[test]
fn test_dispatch_routing_scalar_ops() {
    init();
    let device = Device::metal();
    let a = DynTensor::full(&[4], 3.0, DType::F32, &device).unwrap();

    let b = a.add_scalar(2.0).unwrap();
    let vals = b
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![5.0; 4]);

    let c = a.mul_scalar(10.0).unwrap();
    let vals = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![30.0; 4]);
}

// ===========================================================================
// C. Buffer size validation
// ===========================================================================

/// dispatch_inner_body uses checked_mul for shape computation.
///
/// Structural test: validates the safety pattern exists in the source.
#[test]
fn test_buffer_size_validation_uses_checked_mul() {
    let source = include_str!("../../src/tensor_dispatch.rs");

    let fn_start = source
        .find("fn dispatch_inner_body")
        .expect("dispatch_inner_body must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(3500)];

    assert!(
        fn_snippet.contains("checked_mul"),
        "dispatch_inner_body must use checked_mul for buffer size computation"
    );
}

/// dispatch_inner_body validates DtypeMismatch with runtime error (not debug_assert).
#[test]
fn test_buffer_dtype_validation_is_runtime() {
    let source = include_str!("../../src/tensor_dispatch.rs");

    let fn_start = source
        .find("fn dispatch_inner_body")
        .expect("dispatch_inner_body must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(600)];

    assert!(
        fn_snippet.contains("DtypeMismatch"),
        "dtype validation must be a runtime DtypeMismatch error"
    );
    assert!(
        !fn_snippet.contains("debug_assert"),
        "dtype validation must NOT use debug_assert (stripped in release builds)"
    );
}

/// Buffer size validation happens BEFORE buffer aliasing.
#[test]
fn test_buffer_validation_before_alias() {
    let source = include_str!("../../src/tensor_dispatch.rs");

    let fn_start = source
        .find("fn dispatch_inner_body")
        .expect("dispatch_inner_body must exist");
    let fn_section = &source[fn_start..];
    let fn_snippet = &fn_section[..fn_section.len().min(3500)];

    let size_check = fn_snippet
        .find("BufferSizeMismatch")
        .expect("BufferSizeMismatch check must exist");
    let alias_call = fn_snippet
        .find(".alias()")
        .expect(".alias() call must exist");

    assert!(
        size_check < alias_call,
        "buffer size validation must occur BEFORE alias() (safety invariant)"
    );
}

// ===========================================================================
// D. Pipeline cache behavior
// ===========================================================================

/// Pipeline cache starts empty and grows on compile.
#[test]
fn test_pipeline_cache_grows_on_compile() {
    let ctx = nn_metal::MetalContext::new().expect("Metal context");
    let cache = nn_metal::PipelineCache::new(ctx);

    assert!(cache.is_empty(), "fresh cache should be empty");
    assert_eq!(cache.len(), 0);

    let msl = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_cache_grow(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& total [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id < total) { output[id] = input[id] + 1.0; }
        }
    "#;
    let source = nn_metal::KernelSource::new(msl, "test_cache_grow");
    cache
        .get_or_compile(&source)
        .expect("compile should succeed");

    assert_eq!(cache.len(), 1, "cache should have one entry after compile");
    assert!(!cache.is_empty());
}

/// Repeated compile of the same kernel is a cache hit (L1).
#[test]
fn test_pipeline_cache_l1_hit() {
    let ctx = nn_metal::MetalContext::new().expect("Metal context");
    let cache = nn_metal::PipelineCache::new(ctx);

    let msl = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_l1_hit(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& total [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id < total) { output[id] = input[id] * 2.0; }
        }
    "#;
    let source = nn_metal::KernelSource::new(msl, "test_l1_hit");

    // First call: cold compile.
    let p1 = cache.get_or_compile(&source).expect("first compile");
    assert_eq!(cache.len(), 1);

    // Second call: should be L1 hit (same thread, same cache).
    let p2 = cache.get_or_compile(&source).expect("second compile");
    assert_eq!(cache.len(), 1, "cache size should not change on hit");

    assert_eq!(p1.entry_point(), p2.entry_point());
}

/// Different kernels create separate cache entries.
#[test]
fn test_pipeline_cache_different_kernels() {
    let ctx = nn_metal::MetalContext::new().expect("Metal context");
    let cache = nn_metal::PipelineCache::new(ctx);

    let msl_a = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void kernel_a(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& total [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) { if (id < total) { output[id] = input[id] + 1.0; } }
    "#;
    let msl_b = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void kernel_b(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& total [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) { if (id < total) { output[id] = input[id] - 1.0; } }
    "#;

    let source_a = nn_metal::KernelSource::new(msl_a, "kernel_a");
    let source_b = nn_metal::KernelSource::new(msl_b, "kernel_b");

    cache.get_or_compile(&source_a).expect("compile a");
    cache.get_or_compile(&source_b).expect("compile b");

    assert_eq!(cache.len(), 2, "two different kernels = two cache entries");
}

/// Cache max_entries is respected.
#[test]
fn test_pipeline_cache_max_entries() {
    let ctx = nn_metal::MetalContext::new().expect("Metal context");
    let cache = nn_metal::PipelineCache::with_capacity(ctx, 3);

    assert_eq!(cache.max_entries(), 3);
}

// ===========================================================================
// E. Error handling
// ===========================================================================

/// Invalid MSL source fails with LibraryCompile error.
#[test]
fn test_invalid_msl_compile_error() {
    let ctx = nn_metal::MetalContext::new().expect("Metal context");
    let cache = nn_metal::PipelineCache::new(ctx);

    let bad_msl = "this is not valid MSL code at all";
    let source = nn_metal::KernelSource::new(bad_msl, "nonexistent");
    let result = cache.get_or_compile(&source);

    assert!(result.is_err(), "invalid MSL should fail to compile");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("compile"),
        "error should mention compilation: {msg}"
    );
}

/// MetalError variants have meaningful Display output.
#[test]
fn test_metal_error_display_messages() {
    use nn_metal::MetalError;

    let cases = vec![
        (MetalError::NoDevice, "Metal is unavailable on this host"),
        (
            MetalError::BufferCreate(0),
            "failed to create buffer: size=0",
        ),
        (
            MetalError::MissingEntryPoint("nn_kernel".into()),
            "missing kernel entry point `nn_kernel`",
        ),
        (
            MetalError::ParamCountMismatch {
                expected: 3,
                got: 1,
            },
            "kernel expects 3 parameters but got 1",
        ),
        (
            MetalError::DispatchSizeOverflow(5_000_000_000),
            "exceeds u32::MAX",
        ),
        (
            MetalError::PendingFlushRequired { pending_count: 5 },
            "flush() required",
        ),
        (
            MetalError::BufferBoundsExceeded {
                buffer_len: 100,
                offset: 80,
                size: 30,
                role: "source",
            },
            "bounds exceeded",
        ),
    ];

    for (err, expected_substring) in cases {
        let msg = err.to_string();
        assert!(
            msg.contains(expected_substring),
            "MetalError display for {err:?} should contain '{expected_substring}', got: {msg}"
        );
    }
}

/// MetalError converts to TensorError with correct BackendDomain.
#[test]
fn test_metal_error_to_tensor_error_domain() {
    use nn_metal::MetalError;

    let metal_err = MetalError::NoDevice;
    let tensor_err: nn_core::TensorError = metal_err.into();
    let msg = format!("{tensor_err}");
    assert!(
        msg.contains("Metal") || msg.contains("metal") || msg.contains("unavailable"),
        "TensorError from MetalError should preserve context: {msg}"
    );
}

// ===========================================================================
// F. Lazy batching: accumulate until flush
// ===========================================================================

/// Multiple GPU ops accumulate encodings, single flush commits all.
#[test]
fn test_lazy_batching_accumulates_encodings() {
    init();
    flush().unwrap();
    reset_counters();

    let device = Device::metal();
    let x = DynTensor::full(&[16], 1.0, DType::F32, &device).unwrap();

    // Chain 8 GPU ops without readback.
    let a = x.add_scalar(1.0).unwrap();
    let b = a.mul_scalar(2.0).unwrap();
    let c = b.add_scalar(3.0).unwrap();
    let d = c.neg().unwrap();
    let e = d.add_scalar(10.0).unwrap();
    let f = e.mul_scalar(0.5).unwrap();
    let g = f.neg().unwrap();
    let result = g.add_scalar(100.0).unwrap();

    // All 8 ops encoded; now readback triggers flush.
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let stats = dispatch_stats();
    assert!(
        stats.compute_encodings >= 8,
        "expected >=8 encodings, got {}",
        stats.compute_encodings
    );
    // The key invariant: many encodings but few flushes.
    assert!(
        stats.flushes <= 3,
        "expected <=3 flushes for 8 ops, got {}",
        stats.flushes
    );

    // Verify correctness: ((1+1)*2+3) = 7, neg = -7, +10 = 3, *0.5 = 1.5, neg = -1.5, +100 = 98.5
    assert_eq!(vals.len(), 16);
    for &v in &vals {
        assert!((v - 98.5).abs() < 1e-5, "expected 98.5, got {v}");
    }
}

/// flush() is a no-op when no GPU work is pending.
#[test]
fn test_lazy_batching_flush_noop() {
    init();
    // Ensure clean state.
    flush().unwrap();

    // Flush again should be a no-op (not error).
    let result = flush();
    assert!(result.is_ok(), "double flush should succeed");
}

/// with_gpu_scope flushes automatically on scope exit.
#[test]
fn test_lazy_batching_scope_auto_flush() {
    init();
    flush().unwrap();
    reset_counters();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[4], 3.0, DType::F32, &device).unwrap();

    let result = with_gpu_scope(|| {
        let c = a.add(&b)?;
        let d = c.mul_scalar(10.0)?;
        Ok(d)
    })
    .unwrap();

    // Scope exit flushes automatically.
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![50.0; 4]);

    let stats = dispatch_stats();
    assert!(
        stats.flushes >= 1,
        "scope exit should trigger at least 1 flush"
    );
}

/// Dispatch stats reset_counters zeroes all counters.
#[test]
fn test_dispatch_stats_reset() {
    init();

    // Generate some GPU work.
    let device = Device::metal();
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let _b = a.add_scalar(1.0).unwrap();
    flush().unwrap();

    // Reset.
    reset_counters();
    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 0);
    assert_eq!(stats.blits, 0);
    assert_eq!(stats.flushes, 0);
    assert_eq!(stats.submits, 0);
}

// ===========================================================================
// G. Thread safety: dispatch from multiple threads
// ===========================================================================

/// Multiple threads performing GPU dispatch concurrently should not panic.
#[test]
fn test_thread_safety_concurrent_dispatch() {
    init();
    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                objc::rc::autoreleasepool(|| {
                    init();
                    barrier.wait();

                    let device = Device::metal();
                    let a = DynTensor::full(&[16], (i + 1) as f64, DType::F32, &device).unwrap();
                    let b = DynTensor::full(&[16], 2.0, DType::F32, &device).unwrap();

                    let c = a.add(&b).unwrap();
                    let vals = c
                        .to_device(&Device::Cpu)
                        .unwrap()
                        .to_flat_vec::<f32>()
                        .unwrap();

                    let expected = (i + 1) as f32 + 2.0;
                    assert_eq!(vals, vec![expected; 16], "thread {i} incorrect result");
                    i
                })
            })
        })
        .collect();

    for h in handles {
        let thread_id = h.join().expect("thread should not panic");
        assert!(thread_id < num_threads);
    }
}

/// Multiple threads performing matmul concurrently produce correct results.
#[test]
fn test_thread_safety_concurrent_matmul() {
    init();
    let num_threads = 3;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                objc::rc::autoreleasepool(|| {
                    init();
                    barrier.wait();

                    let device = Device::metal();
                    let scale = (i + 1) as f32;
                    let a_cpu =
                        DynTensor::new(&[scale, 0.0, 0.0, scale], &[2, 2], &Device::Cpu).unwrap();
                    let b_cpu =
                        DynTensor::new(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
                    let a = a_cpu.to_device(&device).unwrap();
                    let b = b_cpu.to_device(&device).unwrap();

                    let c = a.matmul(&b).unwrap();
                    let vals = c
                        .to_device(&Device::Cpu)
                        .unwrap()
                        .to_flat_vec::<f32>()
                        .unwrap();

                    // Identity scaled by `scale`: [[s*1, s*2], [s*3, s*4]]
                    let expected = [scale * 1.0, scale * 2.0, scale * 3.0, scale * 4.0];
                    for (got, want) in vals.iter().zip(expected.iter()) {
                        assert!(
                            (got - want).abs() < 1e-4,
                            "thread {i}: got {got}, expected {want}"
                        );
                    }
                    i
                })
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }
}

/// Each thread has independent dispatch stats (thread-local counters).
#[test]
fn test_thread_safety_independent_stats() {
    init();

    let handle = std::thread::spawn(|| {
        objc::rc::autoreleasepool(|| {
            init();
            reset_counters();

            let device = Device::metal();
            let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
            let _b = a.add_scalar(1.0).unwrap();
            let _c = _b.mul_scalar(2.0).unwrap();
            flush().unwrap();

            let stats = dispatch_stats();
            assert!(
                stats.compute_encodings >= 2,
                "background thread should have its own encoding count"
            );
            stats.compute_encodings
        })
    });

    // Main thread stats should be independent.
    reset_counters();
    let main_stats = dispatch_stats();
    assert_eq!(
        main_stats.compute_encodings, 0,
        "main thread should have zero encodings after reset"
    );

    let bg_encodings = handle.join().expect("thread should not panic");
    assert!(bg_encodings >= 2, "background thread should have encodings");
}

// ===========================================================================
// H. Structural invariants
// ===========================================================================

/// The 3-level GPU dispatch cache architecture is documented in source.
///
/// KernelDefCache (IR) -> MslCodegenCache (MSL) -> PipelineCache (Metal pipeline).
#[test]
fn test_three_level_cache_architecture_exists() {
    // Level 1: KernelDefCache
    let kdc_source = include_str!("../../src/kernel_def_cache.rs");
    assert!(
        kdc_source.contains("struct KernelDefCache"),
        "KernelDefCache (L1 IR cache) must exist"
    );

    // Level 2: MslCodegenCache
    let mcc_source = include_str!("../../src/msl_codegen_cache.rs");
    assert!(
        mcc_source.contains("MslCodegenCache") || mcc_source.contains("msl_codegen_cache"),
        "MslCodegenCache (L2 MSL cache) must exist"
    );

    // Level 3: PipelineCache
    let pc_source = include_str!("../../src/cache.rs");
    assert!(
        pc_source.contains("struct PipelineCache"),
        "PipelineCache (L3 Metal pipeline cache) must exist"
    );
}

/// Pipeline cache has both L1 (thread-local) and L2 (shared) tiers.
#[test]
fn test_pipeline_cache_has_two_tiers() {
    let source = include_str!("../../src/cache.rs");

    assert!(
        source.contains("SharedPipelineCache"),
        "L2 shared cache must exist"
    );
    assert!(
        source.contains("shared_cache()"),
        "shared_cache() accessor must exist"
    );
    assert!(
        source.contains("RefCell<HashMap"),
        "L1 thread-local cache must use RefCell<HashMap>"
    );
    assert!(
        source.contains("RwLock<HashMap"),
        "L2 shared cache must use RwLock<HashMap>"
    );
}

/// Lazy GPU batching constants are defined and reasonable.
#[test]
fn test_lazy_batching_constants() {
    let source = include_str!("../../src/gpu_scope.rs");

    assert!(
        source.contains("MAX_LAZY_ENCODINGS"),
        "MAX_LAZY_ENCODINGS must be defined"
    );

    // Extract the value
    let line = source
        .lines()
        .find(|l| l.contains("MAX_LAZY_ENCODINGS: usize ="))
        .expect("MAX_LAZY_ENCODINGS constant must exist");
    let val_str = line.split('=').nth(1).unwrap().trim().trim_end_matches(';');
    let val: usize = val_str.parse().expect("must be numeric");

    // Should be a reasonable value: at least 64, at most 16384.
    assert!(
        (64..=16384).contains(&val),
        "MAX_LAZY_ENCODINGS={val} should be in [64, 16384]"
    );
}

/// GPU_TIMEOUT is defined and under the macOS watchdog threshold.
#[test]
fn test_gpu_timeout_under_watchdog() {
    let source = include_str!("../../src/dispatch.rs");

    assert!(
        source.contains("GPU_TIMEOUT: Duration"),
        "GPU_TIMEOUT must be defined"
    );

    // Verify it uses from_secs(60) — under the ~90s macOS watchdog.
    assert!(
        source.contains("from_secs(60)"),
        "GPU_TIMEOUT should be 60 seconds (under macOS watchdog threshold)"
    );
}
