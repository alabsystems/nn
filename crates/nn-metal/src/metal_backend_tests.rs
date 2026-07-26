// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`MetalBackend`], [`MetalTensorStorage`], and Direction 2 APIs.

use std::sync::Arc;

use nn_core::backend::Backend;
use nn_core::Tensor;

use super::*;

#[test]
fn test_metal_backend_init() {
    let _backend = MetalBackend::init().expect("Metal init");
}

#[test]
fn test_metal_backend_device() {
    assert_eq!(MetalBackend::device(), Device::metal());
}

#[test]
fn test_metal_backend_init_idempotent() {
    let b1 = MetalBackend::init().expect("first init");
    let b2 = MetalBackend::init().expect("second init");
    assert!(Arc::ptr_eq(&b1.context, &b2.context));
}

#[test]
fn test_metal_backend_zeros_f32() {
    let _backend = MetalBackend::init().expect("Metal init");
    let storage = MetalBackend::zeros::<2, f32>([3, 4]).expect("zeros allocation");
    assert_eq!(storage.len_elements(), 12);
    let data: &[f32] = storage.buffer().contents().expect("readback");
    assert_eq!(data.len(), 12);
    assert!(
        data.iter().all(|&v| v == 0.0),
        "all elements should be zero"
    );
}

#[test]
fn test_metal_backend_ones_f32() {
    let _backend = MetalBackend::init().expect("Metal init");
    let storage = MetalBackend::ones::<2, f32>([3, 4]).expect("ones allocation");
    assert_eq!(storage.len_elements(), 12);
    let data: &[f32] = storage.buffer().contents().expect("readback");
    assert_eq!(data.len(), 12);
    assert!(data.iter().all(|&v| v == 1.0), "all elements should be one");
}

#[test]
fn test_metal_backend_zeros_i32() {
    let _backend = MetalBackend::init().expect("Metal init");
    let storage = MetalBackend::zeros::<1, i32>([5]).expect("zeros allocation");
    assert_eq!(storage.len_elements(), 5);
    let data: &[i32] = storage.buffer().contents().expect("readback");
    assert_eq!(data.len(), 5);
    assert!(data.iter().all(|&v| v == 0), "all elements should be zero");
}

#[test]
fn test_metal_backend_scalar() {
    let _backend = MetalBackend::init().expect("Metal init");
    let storage = MetalBackend::zeros::<0, f32>([]).expect("scalar zeros");
    assert_eq!(storage.len_elements(), 1);
}

#[test]
fn test_metal_backend_ones_scalar() {
    let _backend = MetalBackend::init().expect("Metal init");
    let storage = MetalBackend::ones::<0, f32>([]).expect("scalar ones");
    assert_eq!(storage.len_elements(), 1);
    let data: &[f32] = storage.buffer().contents().expect("readback");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0], 1.0, "scalar ones element should be 1.0");
}

#[test]
fn test_metal_backend_overflow_returns_error() {
    let _backend = MetalBackend::init().expect("Metal init");
    let result = MetalBackend::zeros::<2, f32>([usize::MAX, 2]);
    assert!(result.is_err(), "overflowing dims should return error");
}

#[test]
fn test_metal_backend_ones_overflow_returns_error() {
    let _backend = MetalBackend::init().expect("Metal init");
    let result = MetalBackend::ones::<2, f32>([usize::MAX, 2]);
    assert!(result.is_err(), "overflowing dims should return error");
}

#[test]
fn test_metal_tensor_storage_clone_is_shared() {
    let _backend = MetalBackend::init().expect("Metal init");
    let storage = MetalBackend::zeros::<1, f32>([4]).expect("zeros allocation");
    let cloned = storage.clone();
    assert!(Arc::ptr_eq(&storage.buffer, &cloned.buffer));
}

// -- Proof-closed metallib source resolution (#2467 hardening) -------------
//
// Doctrine: shaders are embedded at compile time; loading a `.metallib`
// from the filesystem at runtime requires an explicit double opt-in
// (config flag + environment guard) and is loud, never silent.

#[test]
fn test_default_options_do_not_allow_runtime_metallib() {
    assert!(!MetalInitOptions::default().runtime_metallib_allowed());
    assert!(!MetalInitOptions::new().runtime_metallib_allowed());
}

#[test]
fn test_resolve_default_never_reads_filesystem() {
    // Even with a build-time metallib path known AND the env guard set,
    // the default options must not select a filesystem source.
    let src = resolve_metallib_source(
        &MetalInitOptions::default(),
        Some("1"),
        Some("/tmp/nn-test.metallib"),
        None,
    )
    .expect("default resolution must succeed");
    assert_eq!(src, MetallibSource::None);
}

#[test]
fn test_resolve_default_prefers_embedded_bytes() {
    static BYTES: &[u8] = b"MTLB-test";
    let src = resolve_metallib_source(
        &MetalInitOptions::default(),
        None,
        Some("/tmp/nn-test.metallib"),
        Some(BYTES),
    )
    .expect("default resolution must succeed");
    assert_eq!(src, MetallibSource::Embedded(BYTES));
}

#[test]
fn test_resolve_runtime_flag_without_env_guard_is_hard_error() {
    let opts = MetalInitOptions::new().allow_runtime_metallib(true);
    let err = resolve_metallib_source(&opts, None, Some("/tmp/nn-test.metallib"), None)
        .expect_err("opt-in without env guard must fail hard");
    assert!(matches!(err, MetalError::RuntimeMetallibDisabled));
    assert!(
        err.to_string().contains("NN_ALLOW_RUNTIME_METALLIB"),
        "error must name the guard: {err}"
    );
}

#[test]
fn test_resolve_runtime_flag_with_wrong_guard_value_is_hard_error() {
    let opts = MetalInitOptions::new().allow_runtime_metallib(true);
    for wrong in ["0", "true", "yes", ""] {
        let err = resolve_metallib_source(&opts, Some(wrong), Some("/tmp/nn-test.metallib"), None)
            .expect_err("guard values other than \"1\" must not enable runtime loading");
        assert!(matches!(err, MetalError::RuntimeMetallibDisabled));
    }
}

#[test]
fn test_resolve_env_guard_alone_does_not_enable_runtime() {
    // The environment variable alone (no config flag) must not enable
    // filesystem loading — the config flag is required at minimum.
    static BYTES: &[u8] = b"MTLB-test";
    let src = resolve_metallib_source(
        &MetalInitOptions::default(),
        Some("1"),
        Some("/tmp/nn-test.metallib"),
        Some(BYTES),
    )
    .expect("default resolution must succeed");
    assert_eq!(src, MetallibSource::Embedded(BYTES));
}

#[test]
fn test_resolve_runtime_opt_in_selects_build_time_path() {
    let opts = MetalInitOptions::new().allow_runtime_metallib(true);
    let src = resolve_metallib_source(&opts, Some("1"), Some("/tmp/nn-test.metallib"), None)
        .expect("fully enabled opt-in must resolve");
    assert_eq!(src, MetallibSource::RuntimeFile("/tmp/nn-test.metallib"));
}

#[test]
fn test_resolve_runtime_opt_in_without_build_path_is_hard_error() {
    let opts = MetalInitOptions::new().allow_runtime_metallib(true);
    let err = resolve_metallib_source(&opts, Some("1"), None, None)
        .expect_err("opt-in with no build-time metallib must fail hard");
    assert!(matches!(err, MetalError::RuntimeMetallibUnavailable { .. }));
}

#[test]
fn test_runtime_metallib_warning_is_loud() {
    let msg = runtime_metallib_warning("/tmp/nn-test.metallib");
    assert!(msg.contains("WARNING"), "warning must be loud: {msg}");
    assert!(msg.contains("/tmp/nn-test.metallib"), "must name the path: {msg}");
    assert!(
        msg.contains("NN_ALLOW_RUNTIME_METALLIB"),
        "must name the guard: {msg}"
    );
}

#[test]
fn test_runtime_metallib_missing_file_is_hard_error() {
    let backend = MetalBackend::init().expect("Metal init");
    let err = load_metallib_source(
        backend.context(),
        MetallibSource::RuntimeFile("/nonexistent/nn-test.metallib"),
    )
    .expect_err("missing runtime metallib must fail hard, not fall back");
    assert!(matches!(err, MetalError::RuntimeMetallibUnavailable { .. }));
}

// -- Direction 2 tests (from_metal_buffer, to_cpu, metal_buffer) ----------

#[test]
fn test_from_metal_buffer_f32() {
    let backend = MetalBackend::init().expect("Metal init");
    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let buffer = backend
        .context()
        .create_buffer(&data)
        .expect("create buffer");
    let tensor: Tensor<2, f32, MetalBackend> =
        from_metal_buffer([2, 3], buffer).expect("from_metal_buffer");
    assert_eq!(tensor.dims(), &[2, 3]);
    assert_eq!(tensor.numel(), 6);
}

#[test]
fn test_from_metal_buffer_undersized() {
    let backend = MetalBackend::init().expect("Metal init");
    let data: [f32; 2] = [1.0, 2.0];
    let buffer = backend
        .context()
        .create_buffer(&data)
        .expect("create buffer");
    let result: nn_core::Result<Tensor<2, f32, MetalBackend>> = from_metal_buffer([2, 3], buffer);
    assert!(result.is_err(), "undersized buffer should return error");
}

#[test]
fn test_to_cpu_roundtrip() {
    let backend = MetalBackend::init().expect("Metal init");
    let data: [f32; 4] = [1.5, 2.5, 3.5, 4.5];
    let buffer = backend
        .context()
        .create_buffer(&data)
        .expect("create buffer");
    let gpu: Tensor<1, f32, MetalBackend> =
        from_metal_buffer([4], buffer).expect("from_metal_buffer");
    let cpu = gpu.to_cpu().expect("to_cpu");
    assert_eq!(cpu.dims(), &[4]);
    assert_eq!(cpu.as_ndarray().as_slice().unwrap(), &[1.5, 2.5, 3.5, 4.5]);
}

#[test]
fn test_zeros_to_cpu() {
    let _backend = MetalBackend::init().expect("Metal init");
    let gpu: Tensor<2, f32, MetalBackend> = Tensor::zeros([3, 2]).expect("GPU zeros");
    let cpu = gpu.to_cpu().expect("to_cpu");
    assert_eq!(cpu.dims(), &[3, 2]);
    assert!(cpu.as_ndarray().iter().all(|&v| v == 0.0));
}

#[test]
fn test_ones_to_cpu() {
    let _backend = MetalBackend::init().expect("Metal init");
    let gpu: Tensor<1, f32, MetalBackend> = Tensor::ones([5]).expect("GPU ones");
    let cpu = gpu.to_cpu().expect("to_cpu");
    assert!(cpu.as_ndarray().iter().all(|&v| v == 1.0));
}

#[test]
fn test_metal_buffer_accessor() {
    let backend = MetalBackend::init().expect("Metal init");
    let data: [f32; 3] = [10.0, 20.0, 30.0];
    let buffer = backend
        .context()
        .create_buffer(&data)
        .expect("create buffer");
    let byte_len = buffer.len();
    let tensor: Tensor<1, f32, MetalBackend> =
        from_metal_buffer([3], buffer).expect("from_metal_buffer");
    assert_eq!(tensor.metal_buffer().len(), byte_len);
}

#[test]
fn test_from_metal_buffer_i32_roundtrip() {
    let backend = MetalBackend::init().expect("Metal init");
    let data: [i32; 4] = [10, 20, 30, 40];
    let buffer = backend
        .context()
        .create_buffer(&data)
        .expect("create buffer");
    let gpu: Tensor<1, i32, MetalBackend> =
        from_metal_buffer([4], buffer).expect("from_metal_buffer");
    let cpu = gpu.to_cpu().expect("to_cpu");
    assert_eq!(cpu.as_ndarray().as_slice().unwrap(), &[10, 20, 30, 40]);
}
