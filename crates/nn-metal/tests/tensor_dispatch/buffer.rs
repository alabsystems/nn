// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for buffer-to-buffer dispatch (#895).
//!
//! Validates that `execute_tensor_dispatch_to_buffer` produces GPU output
//! matching `execute_tensor_dispatch`, and that `DispatchInput::Gpu` correctly
//! chains two dispatch calls without CPU round-trips.

use super::test_utils::{assert_within_budget, metal_setup, rand_f32_vec};
use nn_dsl::{tensor_block_builder::TensorBlockBuilder, ScalarType};
use nn_metal::{
    execute_tensor_dispatch, execute_tensor_dispatch_to_buffer, DispatchInput, GpuSlice,
    MetalBackend, MetalElement,
};
use std::collections::HashMap;

// ===========================================================================
// Dispatch-to-buffer returns same output as dispatch-to-vec
// ===========================================================================

/// `execute_tensor_dispatch_to_buffer` → manual readback must equal
/// `execute_tensor_dispatch` → `Vec<f32>`.
#[test]
fn test_dispatch_to_buffer_matches_dispatch_to_vec() {
    let _ = MetalBackend::init();
    let cache = metal_setup();

    let shape = [4_usize, 16];
    let total = shape[0] * shape[1];

    // Build a simple sigmoid kernel.
    let mut b = TensorBlockBuilder::new("buf_sigmoid");
    let x = b.add_input("x", &shape);
    let sig = b.add_sigmoid(x, &shape);
    let kernel = b.build(sig).expect("valid graph");

    let x_data = rand_f32_vec(0xBF00_0001, total, -5.0, 5.0);

    // Path 1: existing dispatch → Vec<f32>
    let mut cpu_inputs: HashMap<&str, Vec<f32>> = HashMap::new();
    cpu_inputs.insert("x", x_data.clone());
    let vec_out =
        execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &cpu_inputs).expect("vec path");

    // Path 2: buffer dispatch → MetalBuffer → manual readback
    let mut buf_inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
    buf_inputs.insert("x", DispatchInput::Cpu(&x_data));
    let out_buf = execute_tensor_dispatch_to_buffer(&cache, &kernel, ScalarType::F32, &buf_inputs)
        .expect("buffer path");
    let buf_out: Vec<f32> = f32::read_buffer(out_buf.buffer()).expect("readback");

    assert_eq!(vec_out.len(), buf_out.len(), "output length mismatch");
    // Exact equality — same GPU dispatch, same commit_and_wait, same data.
    assert_eq!(
        vec_out, buf_out,
        "buffer path must produce identical output"
    );
}

// ===========================================================================
// Chain two dispatches via DispatchInput::Gpu (no CPU round-trip)
// ===========================================================================

/// Chain sigmoid → relu via GPU buffer: sigmoid output feeds relu input
/// without CPU readback. Compare against sequential CPU-path dispatches.
#[test]
fn test_chained_buffer_dispatch_no_cpu_roundtrip() {
    let _ = MetalBackend::init();
    let cache = metal_setup();

    let shape = [4_usize, 16];
    let total = shape[0] * shape[1];

    // Kernel 1: sigmoid
    let mut b1 = TensorBlockBuilder::new("chain_sigmoid");
    let x1 = b1.add_input("x", &shape);
    let sig = b1.add_sigmoid(x1, &shape);
    let k1 = b1.build(sig).expect("sigmoid kernel");

    // Kernel 2: relu
    let mut b2 = TensorBlockBuilder::new("chain_relu");
    let x2 = b2.add_input("x", &shape);
    let relu = b2.add_relu(x2, &shape);
    let k2 = b2.build(relu).expect("relu kernel");

    let x_data = rand_f32_vec(0xCF00_0002, total, -5.0, 5.0);

    // CPU reference: sigmoid then relu
    let cpu_ref: Vec<f32> = x_data
        .iter()
        .map(|&v| {
            let s = 1.0 / (1.0 + (-v).exp());
            s.max(0.0) // relu
        })
        .collect();

    // Buffer-chained path: sigmoid → buffer → relu (no CPU round-trip)
    let mut sig_inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
    sig_inputs.insert("x", DispatchInput::Cpu(&x_data));
    let sig_buf = execute_tensor_dispatch_to_buffer(&cache, &k1, ScalarType::F32, &sig_inputs)
        .expect("sigmoid buffer dispatch");

    let mut relu_inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
    relu_inputs.insert("x", DispatchInput::Gpu(sig_buf));
    let relu_buf = execute_tensor_dispatch_to_buffer(&cache, &k2, ScalarType::F32, &relu_inputs)
        .expect("relu buffer dispatch");

    let gpu_out: Vec<f32> = f32::read_buffer(relu_buf.buffer()).expect("final readback");

    assert_eq!(gpu_out.len(), total, "chained output length");
    assert_within_budget("chained_sigmoid_relu", &gpu_out, &cpu_ref);
}

// ===========================================================================
// GPU buffer size validation (#930)
// ===========================================================================

/// AC3: undersized GPU buffer must return BufferSizeMismatch error.
#[test]
fn test_undersized_gpu_buffer_returns_error() {
    let cache = metal_setup();

    let shape = [4_usize, 16]; // 64 elements × 4 bytes = 256 bytes expected

    let mut b = TensorBlockBuilder::new("size_check_sigmoid");
    let x = b.add_input("x", &shape);
    let sig = b.add_sigmoid(x, &shape);
    let kernel = b.build(sig).expect("valid graph");

    // Create an undersized buffer: 32 bytes instead of the expected 256.
    let undersized = cache
        .context()
        .create_buffer_zeroed(32)
        .expect("create undersized buffer");

    let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
    inputs.insert("x", DispatchInput::Gpu(GpuSlice::from_ref(&undersized, 0)));

    let result = execute_tensor_dispatch_to_buffer(&cache, &kernel, ScalarType::F32, &inputs);
    let err = result.expect_err("should fail with undersized buffer");
    let msg = err.to_string();
    assert!(
        msg.contains("buffer size mismatch"),
        "expected BufferSizeMismatch error, got: {msg}"
    );
}

/// AC4: correctly-sized GPU buffer must dispatch successfully.
#[test]
fn test_correct_sized_gpu_buffer_succeeds() {
    let _ = MetalBackend::init();
    let cache = metal_setup();

    let shape = [4_usize, 16]; // 64 elements
    let total = shape[0] * shape[1];

    let mut b = TensorBlockBuilder::new("size_ok_sigmoid");
    let x = b.add_input("x", &shape);
    let sig = b.add_sigmoid(x, &shape);
    let kernel = b.build(sig).expect("valid graph");

    // Create a correctly-sized buffer via CPU data upload.
    let x_data = rand_f32_vec(0xDF00_0003, total, -3.0, 3.0);
    let correct_buf = <f32 as MetalElement>::create_buffer(cache.context(), &x_data)
        .expect("create correct buffer");

    let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
    inputs.insert("x", DispatchInput::Gpu(GpuSlice::from_ref(&correct_buf, 0)));

    let result = execute_tensor_dispatch_to_buffer(&cache, &kernel, ScalarType::F32, &inputs);
    assert!(result.is_ok(), "correctly-sized GPU buffer should succeed");

    // Verify output matches CPU dispatch path.
    let mut cpu_inputs: HashMap<&str, Vec<f32>> = HashMap::new();
    cpu_inputs.insert("x", x_data);
    let cpu_out =
        execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &cpu_inputs).expect("cpu path");
    let gpu_out: Vec<f32> = f32::read_buffer(result.unwrap().buffer()).expect("readback");
    assert_eq!(cpu_out, gpu_out, "GPU buffer path must match CPU path");
}

// ===========================================================================
// PrecisionContract dispatch path (#2083)
// ===========================================================================

/// `execute_tensor_dispatch_to_buffer_with_contract` with `PrecisionTier::Strict`
/// must produce output matching the default-contract path within budget.
#[test]
fn test_dispatch_to_buffer_with_strict_contract() {
    use nn_dsl::{PrecisionContract, PrecisionTier};
    use nn_metal::execute_tensor_dispatch_to_buffer_with_contract;

    let _ = MetalBackend::init();
    let cache = metal_setup();

    let shape = [4_usize, 16];
    let total = shape[0] * shape[1];

    let mut b = TensorBlockBuilder::new("contract_sigmoid");
    let x = b.add_input("x", &shape);
    let sig = b.add_sigmoid(x, &shape);
    let kernel = b.build(sig).expect("valid graph");

    let x_data = rand_f32_vec(0xEF00_0001, total, -5.0, 5.0);

    // Path 1: default contract (via execute_tensor_dispatch_to_buffer)
    let mut default_inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
    default_inputs.insert("x", DispatchInput::Cpu(&x_data));
    let default_buf =
        execute_tensor_dispatch_to_buffer(&cache, &kernel, ScalarType::F32, &default_inputs)
            .expect("default contract dispatch");
    let default_out: Vec<f32> = f32::read_buffer(default_buf.buffer()).expect("readback default");

    // Path 2: explicit Strict contract
    let strict_contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    let mut strict_inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
    strict_inputs.insert("x", DispatchInput::Cpu(&x_data));
    let strict_buf = execute_tensor_dispatch_to_buffer_with_contract(
        &cache,
        &kernel,
        ScalarType::F32,
        &strict_inputs,
        strict_contract,
    )
    .expect("strict contract dispatch");
    let strict_out: Vec<f32> = f32::read_buffer(strict_buf.buffer()).expect("readback strict");

    assert_eq!(
        default_out.len(),
        strict_out.len(),
        "output length mismatch"
    );
    assert_within_budget("strict_vs_default_contract", &strict_out, &default_out);
}

// ===========================================================================
// DtypeMismatch error path (#889)
// ===========================================================================

/// Passing `ScalarType::F16` with `f32` element type must return DtypeMismatch.
#[test]
fn test_dtype_mismatch_returns_error() {
    let cache = metal_setup();

    let shape = [2_usize, 4];
    let total = shape[0] * shape[1];

    let mut b = TensorBlockBuilder::new("dtype_mismatch_test");
    let x = b.add_input("x", &shape);
    let sig = b.add_sigmoid(x, &shape);
    let kernel = b.build(sig).expect("valid graph");

    let x_data = rand_f32_vec(0xFF00_0001, total, -1.0, 1.0);

    let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
    inputs.insert("x", DispatchInput::Cpu(&x_data));

    // f32 element type with F16 ScalarType → DtypeMismatch
    let result = execute_tensor_dispatch_to_buffer(&cache, &kernel, ScalarType::F16, &inputs);
    let err = result.expect_err("should fail with dtype mismatch");
    let msg = err.to_string();
    assert!(
        msg.contains("dtype mismatch"),
        "expected DtypeMismatch error, got: {msg}"
    );
}
