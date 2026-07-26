// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal differential tests for Snake1d: Rust reference vs **generated** MSL on GPU.
//!
//! These tests compile MSL produced by `emit_msl()` on the snake `KernelDef`,
//! dispatch on Metal, and compare against the Rust reference implementation
//! within tolerance. This validates the codegen pipeline end-to-end (issue #78).
//!
//! Both the generated MSL and the Rust reference `snake_scalar` clamp alpha
//! to `SNAKE_MIN_ALPHA`, so the two paths agree for all alpha values including
//! zero and sub-threshold values.

#![cfg(target_os = "macos")]

use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use nn_dsl::{
    build_snake_scalar_kernel, differential_tolerance, emit_msl, snake_ref_f16, snake_ref_f32,
    within_differential_budget, KernelDef, PrecisionContract, PrecisionTier, ScalarType,
};
use objc::rc::autoreleasepool;
use std::mem::{size_of, size_of_val};

/// Build an f16 variant of a `KernelDef` by retyping all params and the return type.
///
/// The IR nodes are type-agnostic (types are inferred from params + return_type),
/// so this produces a valid f16 kernel from any f32 kernel.
fn retype_kernel_f16(kernel: &KernelDef) -> KernelDef {
    let mut k = kernel.clone();
    k.return_type = ScalarType::F16;
    for param in &mut k.params {
        param.ty = ScalarType::F16;
    }
    k
}

/// Pre-expand per-channel alpha to per-element for the generated kernel's
/// buffer layout (where `alpha[tid]` is read directly, not indexed by channel).
fn expand_alpha<T: Copy>(
    alpha_per_channel: &[T],
    channels: usize,
    length: usize,
    total: usize,
) -> Result<Vec<T>, String> {
    if channels == 0 {
        return Err("channels must be non-zero".to_string());
    }
    if length == 0 {
        return Err("length must be non-zero".to_string());
    }
    if alpha_per_channel.len() != channels {
        return Err(format!(
            "alpha_per_channel length {} must match channels {channels}",
            alpha_per_channel.len()
        ));
    }
    let frame_size = channels
        .checked_mul(length)
        .ok_or_else(|| "channels * length overflow in expand_alpha".to_string())?;
    if !total.is_multiple_of(frame_size) {
        return Err("total must be divisible by channels * length".to_string());
    }

    Ok((0..total)
        .map(|tid| {
            let channel = (tid / length) % channels;
            alpha_per_channel[channel]
        })
        .collect())
}

fn compile_pipeline(
    msl_source: &str,
    entry_point: &str,
    contract: PrecisionContract,
) -> Option<(Device, metal::ComputePipelineState)> {
    let device = Device::system_default()?;
    // Drain autoreleased NSString temporaries from metal-rs string
    // conversion inside new_library_with_source/get_function (dvoice#1245).
    let pipeline = autoreleasepool(|| {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(contract.fast_math);
        let library = device
            .new_library_with_source(msl_source, &options)
            .expect("generated MSL should compile");
        let function = library
            .get_function(entry_point, None)
            .expect("entry point should exist");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline should build")
    });
    Some((device, pipeline))
}

/// Dispatch the generated scalar snake kernel on Metal and return the output.
///
/// The generated kernel has buffer layout:
/// - buffer(0): y (input, per-element)
/// - buffer(1): alpha (input, per-element, pre-expanded from per-channel)
/// - buffer(2): out
/// - buffer(3): total
fn dispatch_snake_generated<T: Copy>(
    msl_source: &str,
    entry_point: &str,
    x: &[T],
    alpha_expanded: &[T],
    contract: PrecisionContract,
) -> Vec<T> {
    let total = x.len();
    assert_eq!(alpha_expanded.len(), total);

    let (device, pipeline) = compile_pipeline(msl_source, entry_point, contract)
        .expect("Metal device is required for snake differential tests");

    let x_buffer = device.new_buffer_with_data(
        x.as_ptr().cast(),
        size_of_val(x) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let alpha_buffer = device.new_buffer_with_data(
        alpha_expanded.as_ptr().cast(),
        size_of_val(alpha_expanded) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buffer =
        device.new_buffer(size_of_val(x) as u64, MTLResourceOptions::StorageModeShared);

    let total_u32 = u32::try_from(total).expect("total should fit u32");

    let queue = device.new_command_queue();
    // Drain autoreleased ObjC objects (commandBuffer, computeCommandEncoder)
    // to prevent kernel zone exhaustion on test threads (dvoice#1245).
    autoreleasepool(|| {
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&x_buffer), 0);
        encoder.set_buffer(1, Some(&alpha_buffer), 0);
        encoder.set_buffer(2, Some(&out_buffer), 0);
        encoder.set_bytes(3, size_of::<u32>() as u64, (&raw const total_u32).cast());

        let threads_per_group = MTLSize::new(pipeline.thread_execution_width().max(1), 1, 1);
        let threads_per_grid = MTLSize::new(total as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_group);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let status = command_buffer.status();
        assert_eq!(
            status,
            MTLCommandBufferStatus::Completed,
            "Metal command buffer failed during snake differential dispatch: {status:?}"
        );
    });

    let out_ptr = out_buffer.contents() as *const T;
    // SAFETY:
    // - out_buffer.contents() is page-aligned (Metal guarantee), sufficient for T alignment
    // - wait_until_completed() above ensures all `total` elements have been written by the GPU
    // - No mutable references to out_buffer exist during this read
    // - out_buffer remains alive through this scope (owned by this function)
    let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, total) };
    out_slice.to_vec()
}

#[test]
fn test_expand_alpha_layout_mapping() {
    let alpha = vec![0.1f32, 0.5f32];
    let expanded = expand_alpha(&alpha, 2, 3, 12).expect("valid expand_alpha call");
    let expected = vec![0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5];
    assert_eq!(expanded, expected);
}

#[test]
fn test_expand_alpha_rejects_zero_channels() {
    let alpha = vec![1.0f32];
    let err = expand_alpha(&alpha, 0, 4, 4).unwrap_err();
    assert!(
        err.contains("channels must be non-zero"),
        "expected zero-channels error, got: {err}"
    );
}

#[test]
fn test_expand_alpha_rejects_alpha_size_mismatch() {
    let alpha = vec![1.0f32];
    let err = expand_alpha(&alpha, 2, 4, 8).unwrap_err();
    assert!(
        err.contains("must match channels"),
        "expected size-mismatch error, got: {err}"
    );
}

#[test]
fn test_generated_msl_compiles_f32() {
    let kernel = build_snake_scalar_kernel().expect("snake kernel should build");
    let msl = emit_msl(&kernel).expect("MSL generation should succeed");
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    compile_pipeline(&msl, "snake_kernel", contract)
        .expect("f32 generated pipeline should compile");
}

#[test]
fn test_generated_msl_compiles_f16() {
    let kernel = build_snake_scalar_kernel().expect("snake kernel should build");
    let f16_kernel = retype_kernel_f16(&kernel);
    let msl = emit_msl(&f16_kernel).expect("f16 MSL generation should succeed");
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F16);
    compile_pipeline(&msl, "snake_kernel", contract)
        .expect("f16 generated pipeline should compile");
}

#[test]
fn test_rust_vs_generated_metal_differential_f32() {
    let kernel = build_snake_scalar_kernel().expect("snake kernel should build");
    let msl = emit_msl(&kernel).expect("MSL generation should succeed");
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);

    let batch = 2usize;
    let channels = 4usize;
    let length = 33usize;
    let total = batch * channels * length;

    let x: Vec<f32> = (0..total)
        .map(|index| ((index as f32) * 0.173).sin() * 9.7)
        .collect();
    let alpha = vec![0.0f32, 1e-10, 1.0, 3.0];
    let alpha_expanded = expand_alpha(&alpha, channels, length, total).expect("valid alpha layout");

    let rust_out = snake_ref_f32(&x, &alpha, channels, length).expect("valid layout");
    let metal_out = dispatch_snake_generated(&msl, "snake_kernel", &x, &alpha_expanded, contract);

    assert_eq!(
        rust_out.len(),
        metal_out.len(),
        "Rust and Metal output lengths must match"
    );
    for (index, (lhs, rhs)) in rust_out.iter().zip(metal_out.iter()).enumerate() {
        let delta = (lhs - rhs).abs();
        let tolerance = differential_tolerance(*lhs, contract);
        assert!(
            within_differential_budget(*lhs, *rhs, contract),
            "f32 differential mismatch at {index}: rust={lhs}, metal={rhs}, delta={delta}, tolerance={tolerance}"
        );
    }
}

#[test]
fn test_rust_vs_generated_metal_differential_f16() {
    let kernel = build_snake_scalar_kernel().expect("snake kernel should build");
    let f16_kernel = retype_kernel_f16(&kernel);
    let msl = emit_msl(&f16_kernel).expect("f16 MSL generation should succeed");
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F16);

    let batch = 2usize;
    let channels = 3usize;
    let length = 17usize;
    let total = batch * channels * length;

    let x: Vec<half::f16> = (0..total)
        .map(|index| half::f16::from_f32(((index as f32) * 0.087).cos() * 7.5))
        .collect();
    let alpha = vec![
        half::f16::from_f32(0.0),
        half::f16::from_f32(1.0),
        half::f16::from_f32(4.0),
    ];
    let alpha_expanded = expand_alpha(&alpha, channels, length, total).expect("valid alpha layout");

    let rust_out = snake_ref_f16(&x, &alpha, channels, length).expect("valid layout");
    let metal_out = dispatch_snake_generated(&msl, "snake_kernel", &x, &alpha_expanded, contract);

    assert_eq!(
        rust_out.len(),
        metal_out.len(),
        "Rust and Metal output lengths must match"
    );
    for (index, (lhs, rhs)) in rust_out.iter().zip(metal_out.iter()).enumerate() {
        let lhs_f = lhs.to_f32();
        let rhs_f = rhs.to_f32();
        let delta = (lhs_f - rhs_f).abs();
        let tolerance = differential_tolerance(lhs_f, contract);
        assert!(
            within_differential_budget(lhs_f, rhs_f, contract),
            "f16 differential mismatch at {index}: rust={lhs_f}, metal={rhs_f}, delta={delta}, tolerance={tolerance}"
        );
    }
}
