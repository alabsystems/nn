// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end HIP dispatch: codegen → compile → load → launch.
//!
//! Ties together the full pipeline from a `TensorKernelDef` IR graph
//! to executed GPU kernels. Each step is independently usable, but this
//! module provides the convenience API for the common case.
//!
//! # Pipeline
//!
//! ```text
//! TensorKernelDef (nn-dsl IR)
//!   → emit_tensor_hip()          [codegen_hip_tensor.rs]
//!   → compile_hip_source()       [compile_hip.rs]
//!   → HipRuntime::load_kernel()  [hip_runtime.rs]
//!   → HipRuntime::launch()       [hip_runtime.rs]
//! ```

use std::collections::HashMap;

use nn_dsl::{DispatchStep, ScalarType, TensorKernelDef};

use crate::compile_hip::HipModule;
use crate::hip_cache::HipCache;
use crate::hip_ffi::LaunchConfig;
use crate::hip_runtime::{HipBuffer, HipKernel, HipRuntime, HipRuntimeError, HipStream};
use crate::HipCodegenError;

/// Errors from the end-to-end dispatch pipeline.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HipDispatchError {
    #[error("codegen: {0}")]
    Codegen(#[from] HipCodegenError),

    #[error("compilation: {0}")]
    Compile(#[from] crate::compile_hip::HipCompileError),

    #[error("runtime: {0}")]
    Runtime(#[from] HipRuntimeError),
}

/// A compiled and loaded HIP kernel ready for repeated dispatch.
///
/// Holds references to the runtime, loaded kernel, and a reusable stream.
/// Created by [`HipDispatcher::prepare`].
pub struct PreparedKernel {
    kernel: HipKernel,
    stream: HipStream,
    config: LaunchConfig,
}

impl PreparedKernel {
    /// The kernel function name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.kernel.name()
    }

    /// The launch configuration.
    #[must_use]
    pub fn config(&self) -> &LaunchConfig {
        &self.config
    }

    /// Launch the kernel with the given buffers.
    pub fn launch(&self, rt: &HipRuntime, buffers: &[&HipBuffer]) -> Result<(), HipRuntimeError> {
        rt.launch(&self.kernel, &self.stream, self.config, buffers)
    }

    /// Synchronize the stream (wait for kernel completion).
    pub fn synchronize(&self) -> Result<(), HipRuntimeError> {
        self.stream.synchronize()
    }
}

/// High-level dispatcher: codegen → compile → load → launch.
///
/// Caches compiled modules for kernel reuse across multiple dispatches.
pub struct HipDispatcher {
    rt: HipRuntime,
    cache: HipCache,
    target_arch: String,
    /// Module cache: source hash → compiled module.
    modules: HashMap<String, HipModule>,
}

impl HipDispatcher {
    /// Create a new dispatcher for the given device and target architecture.
    pub fn new(device_ordinal: i32, target_arch: &str) -> Result<Self, HipDispatchError> {
        let rt = HipRuntime::init(device_ordinal)?;
        let cache = HipCache::default_location().map_err(HipRuntimeError::Io)?;
        Ok(Self {
            rt,
            cache,
            target_arch: target_arch.to_owned(),
            modules: HashMap::new(),
        })
    }

    /// Access the underlying runtime for buffer operations.
    #[must_use]
    pub fn runtime(&self) -> &HipRuntime {
        &self.rt
    }

    /// The target GPU architecture.
    #[must_use]
    pub fn target_arch(&self) -> &str {
        &self.target_arch
    }

    /// Compile and load a kernel from a `TensorKernelDef` IR graph.
    ///
    /// Returns a [`PreparedKernel`] ready for repeated dispatch.
    /// The compiled module is cached — subsequent calls with the same
    /// IR skip recompilation.
    pub fn prepare(
        &mut self,
        kernel_def: &TensorKernelDef,
        dtype: ScalarType,
        kernel_name: &str,
        config: LaunchConfig,
    ) -> Result<PreparedKernel, HipDispatchError> {
        let source = crate::emit_tensor_hip(kernel_def, dtype)?;
        self.prepare_from_source(&source, kernel_name, config)
    }

    /// Compile and load a kernel from raw HIP C++ source.
    pub fn prepare_from_source(
        &mut self,
        source: &str,
        kernel_name: &str,
        config: LaunchConfig,
    ) -> Result<PreparedKernel, HipDispatchError> {
        let hash = HipCache::content_hash(source, &self.target_arch);
        let module = if let Some(m) = self.modules.get(&hash) {
            m.clone()
        } else {
            let m = crate::compile_hip_source(source, &self.target_arch, Some(&self.cache))?;
            self.modules.insert(hash, m.clone());
            m
        };

        let kernel = self.rt.load_kernel(&module, kernel_name)?;
        let stream = self.rt.create_stream()?;

        Ok(PreparedKernel {
            kernel,
            stream,
            config,
        })
    }
}

/// Compute the [`LaunchConfig`] for a given dispatch step.
///
/// This is the bridge between the backend-agnostic dispatch plan and
/// HIP-specific grid/block dimensions. Returns `Ok(None)` for no-op steps
/// (e.g., `Reshape`), `Ok(Some(config))` for compute steps, or
/// `Err(UnsupportedStep)` for unknown variants.
pub fn launch_config_for_step(
    step: &DispatchStep,
) -> Result<Option<LaunchConfig>, HipCodegenError> {
    use crate::codegen_hip::HIP_BLOCK_SIZE;
    let bs = HIP_BLOCK_SIZE as u32;

    match step {
        // No-op: zero-copy reshape.
        DispatchStep::Reshape { .. } => Ok(None),

        // Elementwise unary activations.
        DispatchStep::Sigmoid { total_elements, .. }
        | DispatchStep::Gelu { total_elements, .. }
        | DispatchStep::GeluErf { total_elements, .. }
        | DispatchStep::Relu { total_elements, .. }
        | DispatchStep::Tanh { total_elements, .. } => {
            Ok(Some(LaunchConfig::for_elementwise(*total_elements, bs)))
        }

        // Elementwise binary ops.
        DispatchStep::BinaryAdd { total_elements, .. }
        | DispatchStep::BinaryMul { total_elements, .. } => {
            Ok(Some(LaunchConfig::for_elementwise(*total_elements, bs)))
        }

        // Composed elementwise (KernelDef IR).
        DispatchStep::Elementwise { total_elements, .. } => {
            Ok(Some(LaunchConfig::for_elementwise(*total_elements, bs)))
        }

        // Matrix multiplication.
        DispatchStep::MatMul {
            m,
            n,
            total_elements,
            ..
        } => {
            if *m >= 16 && *n >= 16 {
                Ok(Some(LaunchConfig::for_matmul(*m, *n, 16, 16)))
            } else {
                Ok(Some(LaunchConfig::for_elementwise(*total_elements, bs)))
            }
        }

        // Linear layer.
        DispatchStep::Linear {
            total_elements,
            batch_size,
            out_features,
            ..
        } => {
            if *batch_size >= 16 && *out_features >= 16 {
                Ok(Some(LaunchConfig::for_matmul(
                    *batch_size,
                    *out_features,
                    16,
                    16,
                )))
            } else {
                Ok(Some(LaunchConfig::for_elementwise(*total_elements, bs)))
            }
        }

        // Softmax: one threadgroup per reduction slice.
        DispatchStep::Softmax { outer_size, .. } => {
            Ok(Some(LaunchConfig::for_reduction(*outer_size, bs)))
        }

        // Embedding lookup.
        DispatchStep::Embedding { total_elements, .. } => {
            Ok(Some(LaunchConfig::for_elementwise(*total_elements, bs)))
        }

        // Reduce (Sum, Mean, Max, Min): one threadgroup per outer slice.
        DispatchStep::Reduce { outer_size, .. } => {
            Ok(Some(LaunchConfig::for_reduction(*outer_size, bs)))
        }

        // Broadcast.
        DispatchStep::Broadcast { total_elements, .. } => {
            Ok(Some(LaunchConfig::for_elementwise(*total_elements, bs)))
        }

        // Narrow: compute total from shape, replacing axis dim with length.
        DispatchStep::Narrow {
            input_shape,
            length,
            axis,
            ..
        } => {
            let total: usize = input_shape
                .iter()
                .enumerate()
                .map(|(i, &d)| if i == *axis { *length } else { d })
                .product();
            Ok(Some(LaunchConfig::for_elementwise(total, bs)))
        }

        // Transpose.
        DispatchStep::Transpose { total_elements, .. } => {
            Ok(Some(LaunchConfig::for_elementwise(*total_elements, bs)))
        }

        // Concat: total = product of output shape.
        DispatchStep::Concat {
            first_input_shape,
            input_axis_sizes,
            axis,
            ..
        } => {
            let total_axis: usize = input_axis_sizes.iter().sum();
            let total: usize = first_input_shape
                .iter()
                .enumerate()
                .map(|(i, &d)| if i == *axis { total_axis } else { d })
                .product();
            Ok(Some(LaunchConfig::for_elementwise(total, bs)))
        }

        // Convolutions (tuple struct variants).
        DispatchStep::Conv1d(p) => Ok(Some(LaunchConfig::for_elementwise(p.total_elements, bs))),
        DispatchStep::Conv2d(p) => Ok(Some(LaunchConfig::for_elementwise(p.total_elements, bs))),
        DispatchStep::ConvTranspose1d(p) => {
            Ok(Some(LaunchConfig::for_elementwise(p.total_elements, bs)))
        }

        // AxisSelect: total = product of output shape (axis dim removed).
        DispatchStep::AxisSelect {
            input_shape, axis, ..
        } => {
            let total: usize = input_shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != *axis)
                .map(|(_, &d)| d)
                .product();
            Ok(Some(LaunchConfig::for_elementwise(total.max(1), bs)))
        }

        // Stack: total = n_inputs * product(input_shape).
        DispatchStep::Stack {
            inputs,
            input_shape,
            ..
        } => {
            let per_input: usize = input_shape.iter().product();
            let total = inputs.len() * per_input;
            Ok(Some(LaunchConfig::for_elementwise(total, bs)))
        }

        // ZeroPad1d.
        DispatchStep::ZeroPad1d {
            channels,
            out_length,
            ..
        } => Ok(Some(LaunchConfig::for_elementwise(
            *channels * *out_length,
            bs,
        ))),

        // IndexSelect.
        DispatchStep::IndexSelect { total_elements, .. } => {
            Ok(Some(LaunchConfig::for_elementwise(*total_elements, bs)))
        }

        // Gather.
        DispatchStep::Gather { total_elements, .. } => {
            Ok(Some(LaunchConfig::for_elementwise(*total_elements, bs)))
        }

        // Simdgroup variants: rocWMMA tiled GEMM when dimensions are aligned,
        // else fall back to naive matmul/linear grid.
        DispatchStep::SimdgroupLinear(ref p) => {
            if crate::codegen_hip_tensor_emit_gemm::should_use_rocwmma(
                p.batch_size,
                p.in_features,
                p.out_features,
            ) {
                Ok(Some(LaunchConfig::for_rocwmma(
                    p.batch_size,
                    p.out_features,
                    1,
                )))
            } else if p.batch_size >= 16 && p.out_features >= 16 {
                Ok(Some(LaunchConfig::for_matmul(
                    p.batch_size,
                    p.out_features,
                    16,
                    16,
                )))
            } else {
                let total = p.batch_size * p.out_features;
                Ok(Some(LaunchConfig::for_elementwise(total, bs)))
            }
        }

        DispatchStep::SimdgroupMatMul(ref p) => {
            if crate::codegen_hip_tensor_emit_gemm::should_use_rocwmma(p.m, p.k, p.n) {
                Ok(Some(LaunchConfig::for_rocwmma(p.m, p.n, p.batch_size)))
            } else if p.m >= 16 && p.n >= 16 {
                Ok(Some(LaunchConfig::for_matmul(p.m, p.n, 16, 16)))
            } else {
                let total = p.m * p.n * p.batch_size;
                Ok(Some(LaunchConfig::for_elementwise(total, bs)))
            }
        }

        // Unknown variant — must not silently skip computation.
        _ => Err(HipCodegenError::UnsupportedStep {
            step_name: "unknown",
        }),
    }
}

#[cfg(test)]
#[path = "hip_dispatch_tests.rs"]
mod tests;
