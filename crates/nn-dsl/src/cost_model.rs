// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Roofline-based cost model for compiled execution plans.
//!
//! Estimates wall-clock time for a [`CompiledPlan`] as
//! `max(compute_time, memory_time) + launch_overhead` per dispatch step.
//! Accounts for SIMD occupancy loss when tensor dimensions are not multiples
//! of the SIMD width.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use crate::trace_compile::{CompiledPlan, CompiledStep};

/// Roofline-based cost model for compiled execution plans.
///
/// Estimates execution time as `max(compute_time, memory_time) + launch_overhead`
/// per dispatch step. Accounts for SIMD occupancy loss when tensor dimensions
/// are not multiples of the SIMD width.
#[derive(Clone, Debug)]
pub struct CostModel {
    /// Fixed kernel launch overhead in nanoseconds.
    pub launch_overhead_ns: f64,
    /// Compute throughput per op type (FLOP/s).
    pub op_throughput: HashMap<String, f64>,
    /// Device memory bandwidth in bytes/second.
    pub bandwidth_bytes_per_sec: f64,
    /// SIMD group width (32 for Apple GPU, 32 for NVIDIA).
    pub simd_width: usize,
}

/// Default FLOP/s throughput used when no op-specific entry exists.
const DEFAULT_THROUGHPUT_FLOPS: f64 = 1e12; // 1 TFLOP/s

/// Cost estimate for a compiled execution plan.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CostEstimate {
    /// Total estimated time in nanoseconds.
    pub total_ns: f64,
    /// Per-step breakdown (step index, estimated ns).
    pub per_step_ns: Vec<(usize, f64)>,
    /// Number of dispatch steps costed.
    pub dispatch_count: usize,
}

impl CostModel {
    /// Create a cost model with reasonable defaults for Apple M4 (base).
    ///
    /// - Bandwidth: ~400 GB/s (unified memory)
    /// - Launch overhead: ~2000 ns
    /// - SIMD width: 32 (Apple GPU simdgroup size)
    /// - Default throughput: 1 TFLOP/s (conservative for mixed workloads)
    #[must_use]
    pub fn apple_m4() -> Self {
        Self {
            launch_overhead_ns: 2000.0,
            op_throughput: HashMap::new(),
            bandwidth_bytes_per_sec: 400e9,
            simd_width: 32,
        }
    }

    /// Create a cost model calibrated for Apple M4 Max.
    ///
    /// M4 Max has 40 GPU cores (vs 10 on base M4), delivering significantly
    /// higher compute throughput while sharing the same memory bandwidth:
    ///
    /// - Bandwidth: ~400 GB/s (unified memory, same as base M4)
    /// - Launch overhead: ~1500 ns (lower due to larger GPU scheduler)
    /// - SIMD width: 32 (Apple GPU simdgroup size)
    /// - F32 throughput: ~35 TFLOP/s
    /// - F16 throughput: ~70 TFLOP/s
    /// - Op-specific throughputs for matmul, conv, softmax, elementwise
    #[must_use]
    pub fn apple_m4_max() -> Self {
        let mut op_throughput = HashMap::new();
        // MatMul: near peak F32 utilization on large matrices.
        op_throughput.insert("matmul".to_string(), 30e12);
        // Conv1d/Conv2d: slightly below peak due to memory access patterns.
        op_throughput.insert("conv1d".to_string(), 20e12);
        op_throughput.insert("conv2d".to_string(), 20e12);
        // Softmax: memory-bound in practice, moderate compute.
        op_throughput.insert("softmax".to_string(), 8e12);
        // Elementwise ops (gelu, relu, snake, silu): bandwidth-limited.
        op_throughput.insert("gelu".to_string(), 10e12);
        op_throughput.insert("relu".to_string(), 12e12);
        op_throughput.insert("snake".to_string(), 8e12);
        op_throughput.insert("silu".to_string(), 10e12);
        // Layer/instance norm: moderate compute, sequential reduction.
        op_throughput.insert("layer_norm".to_string(), 6e12);
        op_throughput.insert("instance_norm".to_string(), 6e12);

        Self {
            launch_overhead_ns: 1500.0,
            op_throughput,
            bandwidth_bytes_per_sec: 400e9,
            simd_width: 32,
        }
    }

    /// Create a cost model calibrated for Apple M1 GPU.
    ///
    /// M1 has 128 execution units (~8 GPU cores), delivering ~2.6 TFLOPS F32:
    ///
    /// - Bandwidth: ~68.25 GB/s (unified memory, LPDDR4X)
    /// - Launch overhead: ~3000 ns (first-generation Apple Silicon GPU scheduler)
    /// - SIMD width: 32 (Apple GPU simdgroup size)
    /// - F32 throughput: ~2.6 TFLOP/s
    /// - Op-specific throughputs for matmul, conv, softmax, elementwise
    #[must_use]
    pub fn apple_m1() -> Self {
        let mut op_throughput = HashMap::new();
        op_throughput.insert("matmul".to_string(), 2.0e12);
        op_throughput.insert("conv1d".to_string(), 1.5e12);
        op_throughput.insert("conv2d".to_string(), 1.5e12);
        op_throughput.insert("softmax".to_string(), 0.8e12);
        op_throughput.insert("gelu".to_string(), 1.0e12);
        op_throughput.insert("relu".to_string(), 1.2e12);
        op_throughput.insert("snake".to_string(), 0.8e12);
        op_throughput.insert("silu".to_string(), 1.0e12);
        op_throughput.insert("layer_norm".to_string(), 0.6e12);
        op_throughput.insert("instance_norm".to_string(), 0.6e12);

        Self {
            launch_overhead_ns: 3000.0,
            op_throughput,
            bandwidth_bytes_per_sec: 68.25e9,
            simd_width: 32,
        }
    }

    /// Create a cost model calibrated for Apple M2 GPU.
    ///
    /// M2 has 10 GPU cores, delivering ~3.6 TFLOPS F32:
    ///
    /// - Bandwidth: ~100 GB/s (unified memory, LPDDR5)
    /// - Launch overhead: ~2500 ns (improved over M1 GPU scheduler)
    /// - SIMD width: 32 (Apple GPU simdgroup size)
    /// - F32 throughput: ~3.6 TFLOP/s
    /// - Op-specific throughputs for matmul, conv, softmax, elementwise
    #[must_use]
    pub fn apple_m2() -> Self {
        let mut op_throughput = HashMap::new();
        op_throughput.insert("matmul".to_string(), 2.8e12);
        op_throughput.insert("conv1d".to_string(), 2.0e12);
        op_throughput.insert("conv2d".to_string(), 2.0e12);
        op_throughput.insert("softmax".to_string(), 1.0e12);
        op_throughput.insert("gelu".to_string(), 1.4e12);
        op_throughput.insert("relu".to_string(), 1.6e12);
        op_throughput.insert("snake".to_string(), 1.0e12);
        op_throughput.insert("silu".to_string(), 1.4e12);
        op_throughput.insert("layer_norm".to_string(), 0.8e12);
        op_throughput.insert("instance_norm".to_string(), 0.8e12);

        Self {
            launch_overhead_ns: 2500.0,
            op_throughput,
            bandwidth_bytes_per_sec: 100e9,
            simd_width: 32,
        }
    }

    /// Create a cost model calibrated for Apple M3 GPU.
    ///
    /// M3 has 10 GPU cores with dynamic caching, delivering ~4.1 TFLOPS F32:
    ///
    /// - Bandwidth: ~100 GB/s (unified memory, LPDDR5)
    /// - Launch overhead: ~2000 ns (dynamic caching reduces scheduler overhead)
    /// - SIMD width: 32 (Apple GPU simdgroup size)
    /// - F32 throughput: ~4.1 TFLOP/s
    /// - Op-specific throughputs for matmul, conv, softmax, elementwise
    #[must_use]
    pub fn apple_m3() -> Self {
        let mut op_throughput = HashMap::new();
        op_throughput.insert("matmul".to_string(), 3.2e12);
        op_throughput.insert("conv1d".to_string(), 2.4e12);
        op_throughput.insert("conv2d".to_string(), 2.4e12);
        op_throughput.insert("softmax".to_string(), 1.2e12);
        op_throughput.insert("gelu".to_string(), 1.6e12);
        op_throughput.insert("relu".to_string(), 1.8e12);
        op_throughput.insert("snake".to_string(), 1.2e12);
        op_throughput.insert("silu".to_string(), 1.6e12);
        op_throughput.insert("layer_norm".to_string(), 1.0e12);
        op_throughput.insert("instance_norm".to_string(), 1.0e12);

        Self {
            launch_overhead_ns: 2000.0,
            op_throughput,
            bandwidth_bytes_per_sec: 100e9,
            simd_width: 32,
        }
    }

    /// Create a cost model calibrated for Apple M4 Pro GPU.
    ///
    /// M4 Pro has 20 GPU cores, delivering ~8.2 TFLOPS F32:
    ///
    /// - Bandwidth: ~273 GB/s (unified memory, LPDDR5X)
    /// - Launch overhead: ~1800 ns (improved scheduler with more GPU cores)
    /// - SIMD width: 32 (Apple GPU simdgroup size)
    /// - F32 throughput: ~8.2 TFLOP/s
    /// - Op-specific throughputs for matmul, conv, softmax, elementwise
    #[must_use]
    pub fn apple_m4_pro() -> Self {
        let mut op_throughput = HashMap::new();
        op_throughput.insert("matmul".to_string(), 6.5e12);
        op_throughput.insert("conv1d".to_string(), 4.5e12);
        op_throughput.insert("conv2d".to_string(), 4.5e12);
        op_throughput.insert("softmax".to_string(), 2.5e12);
        op_throughput.insert("gelu".to_string(), 3.2e12);
        op_throughput.insert("relu".to_string(), 3.6e12);
        op_throughput.insert("snake".to_string(), 2.5e12);
        op_throughput.insert("silu".to_string(), 3.2e12);
        op_throughput.insert("layer_norm".to_string(), 2.0e12);
        op_throughput.insert("instance_norm".to_string(), 2.0e12);

        Self {
            launch_overhead_ns: 1800.0,
            op_throughput,
            bandwidth_bytes_per_sec: 273e9,
            simd_width: 32,
        }
    }

    /// Create a cost model calibrated for NVIDIA A100 (80 GB SXM).
    ///
    /// A100 delivers 19.5 TFLOPS F32, 312 TFLOPS TF32, 624 GB/s HBM2e:
    ///
    /// - Bandwidth: ~2039 GB/s (HBM2e, 80 GB variant)
    /// - Launch overhead: ~5000 ns (PCIe/NVLink kernel launch latency)
    /// - SIMD width: 32 (NVIDIA warp size)
    /// - F32 throughput: ~19.5 TFLOP/s
    /// - Higher matmul efficiency due to Tensor Cores (TF32)
    /// - Op-specific throughputs for matmul, conv, softmax, elementwise
    #[must_use]
    pub fn nvidia_a100() -> Self {
        let mut op_throughput = HashMap::new();
        // MatMul benefits from Tensor Cores (TF32 mode ~156 TFLOP/s),
        // but practical F32 throughput is lower. Use ~17 TFLOP/s effective.
        op_throughput.insert("matmul".to_string(), 17.0e12);
        op_throughput.insert("conv1d".to_string(), 12.0e12);
        op_throughput.insert("conv2d".to_string(), 12.0e12);
        op_throughput.insert("softmax".to_string(), 5.0e12);
        op_throughput.insert("gelu".to_string(), 8.0e12);
        op_throughput.insert("relu".to_string(), 10.0e12);
        op_throughput.insert("snake".to_string(), 6.0e12);
        op_throughput.insert("silu".to_string(), 8.0e12);
        op_throughput.insert("layer_norm".to_string(), 4.0e12);
        op_throughput.insert("instance_norm".to_string(), 4.0e12);

        Self {
            launch_overhead_ns: 5000.0,
            op_throughput,
            bandwidth_bytes_per_sec: 2039e9,
            simd_width: 32,
        }
    }

    /// Create a cost model calibrated for NVIDIA RTX 4090.
    ///
    /// RTX 4090 delivers 82.6 TFLOPS F32, 1.01 TB/s GDDR6X bandwidth:
    ///
    /// - Bandwidth: ~1008 GB/s (GDDR6X)
    /// - Launch overhead: ~7000 ns (consumer PCIe, higher driver overhead)
    /// - SIMD width: 32 (NVIDIA warp size)
    /// - F32 throughput: ~82.6 TFLOP/s (Ada Lovelace, 16384 CUDA cores)
    /// - High matmul efficiency for large matrices
    /// - Op-specific throughputs for matmul, conv, softmax, elementwise
    #[must_use]
    pub fn nvidia_rtx_4090() -> Self {
        let mut op_throughput = HashMap::new();
        // Practical matmul throughput ~70% of peak for large matrices.
        op_throughput.insert("matmul".to_string(), 60.0e12);
        op_throughput.insert("conv1d".to_string(), 40.0e12);
        op_throughput.insert("conv2d".to_string(), 40.0e12);
        op_throughput.insert("softmax".to_string(), 15.0e12);
        op_throughput.insert("gelu".to_string(), 25.0e12);
        op_throughput.insert("relu".to_string(), 30.0e12);
        op_throughput.insert("snake".to_string(), 18.0e12);
        op_throughput.insert("silu".to_string(), 25.0e12);
        op_throughput.insert("layer_norm".to_string(), 12.0e12);
        op_throughput.insert("instance_norm".to_string(), 12.0e12);

        Self {
            launch_overhead_ns: 7000.0,
            op_throughput,
            bandwidth_bytes_per_sec: 1008e9,
            simd_width: 32,
        }
    }

    /// Estimate the wall-clock cost of executing a compiled plan.
    ///
    /// Only `Dispatch` and `NativeOp` steps incur GPU cost. Passthrough,
    /// InputForward, IdentityPassthrough, ConstantValue, NarrowView, and
    /// RuntimeOp steps are treated as zero-cost metadata operations.
    #[must_use]
    pub fn estimate(&self, plan: &CompiledPlan) -> CostEstimate {
        let mut total_ns = 0.0;
        let mut per_step_ns = Vec::new();
        let mut dispatch_count = 0usize;

        for (idx, step) in plan.steps.iter().enumerate() {
            let step_ns = match step {
                CompiledStep::Dispatch { kernel, .. } => {
                    dispatch_count += 1;
                    let elements = kernel
                        .output_shape()
                        .map(|s| s.iter().product::<usize>())
                        .unwrap_or(0);
                    self.step_cost(kernel.name(), elements)
                }
                CompiledStep::NativeOp { op, .. } => {
                    let metal_dispatches = op.estimated_metal_dispatches();
                    dispatch_count += metal_dispatches;
                    // Native ops are pre-fused; estimate as N launches with
                    // a nominal element count (throughput-dominated).
                    metal_dispatches as f64 * self.launch_overhead_ns
                }
                // Non-dispatch steps: zero cost.
                CompiledStep::Passthrough { .. }
                | CompiledStep::NarrowView { .. }
                | CompiledStep::InputForward
                | CompiledStep::IdentityPassthrough
                | CompiledStep::ConstantValue { .. }
                | CompiledStep::RuntimeOp { .. } => 0.0,
            };

            if step_ns > 0.0 {
                per_step_ns.push((idx, step_ns));
            }
            total_ns += step_ns;
        }

        CostEstimate {
            total_ns,
            per_step_ns,
            dispatch_count,
        }
    }

    /// Build a calibration plan for a compiled execution plan.
    ///
    /// Returns one [`CalibrationRecord`] per dispatch or NativeOp step,
    /// pre-filled with the roofline-estimated cost. Non-dispatch steps
    /// (Passthrough, InputForward, etc.) are excluded — they are zero-cost
    /// metadata operations.
    ///
    /// After GPU profiling, fill in `actual_ns` on each record and call
    /// [`CalibrationReport::from_records`] to compute aggregate statistics.
    #[must_use]
    pub fn calibration_plan(&self, plan: &CompiledPlan) -> Vec<CalibrationRecord> {
        let mut records = Vec::new();

        for (idx, step) in plan.steps.iter().enumerate() {
            match step {
                CompiledStep::Dispatch { kernel, .. } => {
                    let name = kernel.name().to_string();
                    let elements = kernel
                        .output_shape()
                        .map(|s| s.iter().product::<usize>())
                        .unwrap_or(0);
                    let estimated_ns = self.step_cost(&name, elements);
                    let is_memory_bound = self.is_memory_bound(&name, elements);
                    records.push(CalibrationRecord {
                        step_index: idx,
                        estimated_ns,
                        actual_ns: None,
                        op_name: name,
                        is_memory_bound,
                    });
                }
                CompiledStep::NativeOp { op, .. } => {
                    let name = op.variant_name().to_string();
                    let dispatches = op.estimated_metal_dispatches();
                    let estimated_ns = dispatches as f64 * self.launch_overhead_ns;
                    // NativeOps are pre-fused and launch-overhead dominated;
                    // treat them as memory-bound (bandwidth-limited) since
                    // the fused kernel typically streams through data once.
                    records.push(CalibrationRecord {
                        step_index: idx,
                        estimated_ns,
                        actual_ns: None,
                        op_name: name,
                        is_memory_bound: true,
                    });
                }
                // Non-dispatch steps: zero cost, no calibration needed.
                CompiledStep::Passthrough { .. }
                | CompiledStep::NarrowView { .. }
                | CompiledStep::InputForward
                | CompiledStep::IdentityPassthrough
                | CompiledStep::ConstantValue { .. }
                | CompiledStep::RuntimeOp { .. } => {}
            }
        }

        records
    }

    /// Determine whether a dispatch step is memory-bound based on arithmetic
    /// intensity (FLOP / byte transferred).
    ///
    /// A step is memory-bound when the memory transfer time equals or exceeds
    /// the compute time under the roofline model.
    fn is_memory_bound(&self, op_name: &str, elements: usize) -> bool {
        if elements == 0 {
            return true;
        }
        let throughput = self
            .op_throughput
            .get(op_name)
            .copied()
            .unwrap_or(DEFAULT_THROUGHPUT_FLOPS);

        let compute_ns = (elements as f64 / throughput) * 1e9;
        let bytes_transferred = (elements as f64) * 4.0 * 2.0;
        let memory_ns = (bytes_transferred / self.bandwidth_bytes_per_sec) * 1e9;

        memory_ns >= compute_ns
    }

    /// Estimate cost for a single dispatch step given op name and element count.
    fn step_cost(&self, op_name: &str, elements: usize) -> f64 {
        if elements == 0 {
            return self.launch_overhead_ns;
        }

        let throughput = self
            .op_throughput
            .get(op_name)
            .copied()
            .unwrap_or(DEFAULT_THROUGHPUT_FLOPS);

        // Compute time: elements / throughput (seconds) → nanoseconds.
        let compute_ns = (elements as f64 / throughput) * 1e9;

        // Memory time: bytes transferred / bandwidth → nanoseconds.
        // Assume f32 (4 bytes per element), read + write = 2x.
        let bytes_transferred = (elements as f64) * 4.0 * 2.0;
        let memory_ns = (bytes_transferred / self.bandwidth_bytes_per_sec) * 1e9;

        // Occupancy penalty: partial SIMD groups waste lanes.
        let occupancy = if self.simd_width == 0 || elements.is_multiple_of(self.simd_width) {
            1.0
        } else {
            let remainder = elements % self.simd_width;
            // At least 10% occupancy to avoid division explosion.
            f64::max(0.1, remainder as f64 / self.simd_width as f64)
        };

        self.launch_overhead_ns + f64::max(compute_ns, memory_ns) / occupancy
    }
}

/// A single calibration record for one dispatch step.
///
/// Pre-filled with estimated cost from the roofline model. The `actual_ns`
/// field starts as `None` and is populated after GPU profiling measures the
/// real dispatch time.
#[derive(Clone, Debug)]
pub struct CalibrationRecord {
    /// Index into `CompiledPlan::steps`.
    pub step_index: usize,
    /// Roofline-estimated wall-clock time in nanoseconds.
    pub estimated_ns: f64,
    /// Measured wall-clock time in nanoseconds (`None` until profiled).
    pub actual_ns: Option<f64>,
    /// Human-readable operation name (kernel name or NativeOp variant).
    pub op_name: String,
    /// Whether the roofline model classifies this step as memory-bound
    /// (memory transfer time >= compute time).
    pub is_memory_bound: bool,
}

/// Aggregate statistics comparing estimated vs actual dispatch times.
///
/// Computed from a set of [`CalibrationRecord`]s that have been profiled
/// (i.e., `actual_ns` is `Some`), or from paired prediction/actual slices
/// via [`CostModel::calibrate`]. Use [`CalibrationReport::from_records`]
/// or [`CostModel::calibrate`] to construct.
#[derive(Clone, Debug)]
pub struct CalibrationReport {
    /// Mean absolute error between estimated and actual times (nanoseconds).
    pub mean_absolute_error_ns: f64,
    /// Largest overestimate: max(estimated - actual) across all records.
    /// Zero if no step was overestimated.
    pub max_overestimate_ns: f64,
    /// Largest underestimate: max(actual - estimated) across all records.
    /// Zero if no step was underestimated.
    pub max_underestimate_ns: f64,
    /// Pearson correlation coefficient between estimated and actual times.
    /// NaN if fewer than 2 profiled records or zero variance.
    pub correlation: f64,
    /// Mean ratio of predicted to actual time across all matched entries.
    ///
    /// A value of 1.0 means perfect calibration on average. Values > 1.0
    /// indicate systematic overestimation; < 1.0 indicates underestimation.
    /// NaN if no entries have positive actual times.
    pub mean_error_ratio: f64,
    /// Maximum ratio of predicted to actual time across all matched entries.
    ///
    /// Identifies the worst-calibrated step. NaN if no entries have positive
    /// actual times.
    pub max_error_ratio: f64,
    /// Per-entry calibration data (only populated by [`CostModel::calibrate`];
    /// empty when constructed via [`CalibrationReport::from_records`]).
    pub entries: Vec<CalibrationData>,
}

/// A single calibration data point pairing a predicted cost with an actual
/// measured cost for a named dispatch step category.
#[derive(Clone, Debug)]
pub struct CalibrationData {
    /// Human-readable name identifying the dispatch step category
    /// (e.g., "matmul", "softmax", "conv1d").
    pub step_name: String,
    /// Predicted (estimated) wall-clock time in nanoseconds.
    pub predicted_ns: f64,
    /// Actual (measured) wall-clock time in nanoseconds.
    pub actual_ns: f64,
}

/// Error type for calibration operations.
#[derive(Clone, Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CalibrationError {
    /// No matching step names found between predictions and actuals.
    #[error("no matching step names between predictions and actuals")]
    NoMatchingSteps,
    /// A matched step has a non-positive actual time, making ratio
    /// computation undefined.
    #[error("step '{name}' has non-positive actual_ns ({actual_ns}); ratio undefined")]
    NonPositiveActual {
        /// Name of the step with the invalid actual time.
        name: String,
        /// The non-positive actual time value.
        actual_ns: f64,
    },
}

impl CalibrationReport {
    /// Compute a calibration report from a slice of records.
    ///
    /// Only records where `actual_ns` is `Some` are included. Returns a
    /// report with all-zero fields (and NaN correlation/ratios) if no
    /// records have actual measurements.
    #[must_use]
    pub fn from_records(records: &[CalibrationRecord]) -> Self {
        let paired: Vec<(f64, f64)> = records
            .iter()
            .filter_map(|r| r.actual_ns.map(|a| (r.estimated_ns, a)))
            .collect();

        if paired.is_empty() {
            return Self {
                mean_absolute_error_ns: 0.0,
                max_overestimate_ns: 0.0,
                max_underestimate_ns: 0.0,
                correlation: f64::NAN,
                mean_error_ratio: f64::NAN,
                max_error_ratio: f64::NAN,
                entries: Vec::new(),
            };
        }

        let n = paired.len() as f64;
        let mut sum_abs_err = 0.0;
        let mut max_over = 0.0_f64;
        let mut max_under = 0.0_f64;
        let mut sum_ratio = 0.0_f64;
        let mut max_ratio = 0.0_f64;
        let mut ratio_count = 0usize;

        for &(est, act) in &paired {
            let err = est - act;
            sum_abs_err += err.abs();
            max_over = max_over.max(err); // positive = overestimate
            max_under = max_under.max(-err); // positive = underestimate
            if act > 0.0 {
                let ratio = est / act;
                sum_ratio += ratio;
                max_ratio = max_ratio.max(ratio);
                ratio_count += 1;
            }
        }

        // Clamp to zero: if all errors are in one direction, the other
        // extreme should be zero rather than negative.
        max_over = max_over.max(0.0);
        max_under = max_under.max(0.0);

        let correlation = if paired.len() < 2 {
            f64::NAN
        } else {
            pearson_r(&paired)
        };

        let (mean_error_ratio, max_error_ratio) = if ratio_count > 0 {
            (sum_ratio / ratio_count as f64, max_ratio)
        } else {
            (f64::NAN, f64::NAN)
        };

        Self {
            mean_absolute_error_ns: sum_abs_err / n,
            max_overestimate_ns: max_over,
            max_underestimate_ns: max_under,
            correlation,
            mean_error_ratio,
            max_error_ratio,
            entries: Vec::new(),
        }
    }

    /// Human-readable summary of the calibration report.
    ///
    /// Includes aggregate statistics and per-entry details when available.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut out = String::with_capacity(256);
        out.push_str("Calibration Report\n");
        out.push_str(&format!(
            "  Mean absolute error:  {:.1} ns\n",
            self.mean_absolute_error_ns
        ));
        out.push_str(&format!(
            "  Max overestimate:     {:.1} ns\n",
            self.max_overestimate_ns
        ));
        out.push_str(&format!(
            "  Max underestimate:    {:.1} ns\n",
            self.max_underestimate_ns
        ));
        if self.mean_error_ratio.is_finite() {
            out.push_str(&format!(
                "  Mean error ratio:     {:.4}x\n",
                self.mean_error_ratio
            ));
        } else {
            out.push_str("  Mean error ratio:     N/A\n");
        }
        if self.max_error_ratio.is_finite() {
            out.push_str(&format!(
                "  Max error ratio:      {:.4}x\n",
                self.max_error_ratio
            ));
        } else {
            out.push_str("  Max error ratio:      N/A\n");
        }
        if self.correlation.is_finite() {
            out.push_str(&format!(
                "  Correlation:          {:.4}\n",
                self.correlation
            ));
        } else {
            out.push_str("  Correlation:          N/A\n");
        }

        if !self.entries.is_empty() {
            out.push_str(&format!("  Entries: {}\n", self.entries.len()));
            for entry in &self.entries {
                let ratio = if entry.actual_ns > 0.0 {
                    format!("{:.4}x", entry.predicted_ns / entry.actual_ns)
                } else {
                    "N/A".to_string()
                };
                out.push_str(&format!(
                    "    {}: predicted={:.1} ns, actual={:.1} ns, ratio={}\n",
                    entry.step_name, entry.predicted_ns, entry.actual_ns, ratio
                ));
            }
        }

        out
    }

    /// Compute per-category correction factors from the calibration entries.
    ///
    /// Each factor is `actual_ns / predicted_ns` for that step name. To
    /// improve future predictions, multiply the cost model's estimate by
    /// the corresponding factor. Returns an empty map if `entries` is empty
    /// or all entries have non-positive predicted times.
    #[must_use]
    pub fn adjustment_factors(&self) -> BTreeMap<String, f64> {
        let mut factors = BTreeMap::new();
        for entry in &self.entries {
            if entry.predicted_ns > 0.0 && entry.actual_ns > 0.0 {
                factors.insert(
                    entry.step_name.clone(),
                    entry.actual_ns / entry.predicted_ns,
                );
            }
        }
        factors
    }
}

impl CostModel {
    /// Apply per-category adjustment factors to the cost model's op throughput.
    ///
    /// Each factor is `actual_ns / predicted_ns` for a given op category (from
    /// [`CalibrationReport::adjustment_factors()`]). Multiplying the roofline
    /// estimate by this factor corrects the model to match real GPU timings.
    ///
    /// This works by adjusting `op_throughput` in the *inverse* direction:
    /// if the model overestimates (predicted > actual, factor < 1.0), we
    /// *increase* throughput so the cost estimate comes down. If the model
    /// underestimates (predicted < actual, factor > 1.0), we *decrease*
    /// throughput so the cost estimate goes up.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let report = CostModel::calibrate(&predictions, &actuals)?;
    /// let factors = report.adjustment_factors();
    /// cost_model.apply_adjustment_factors(&factors);
    /// // Subsequent estimate() calls use calibrated throughputs.
    /// ```
    ///
    /// Factors for ops not already in `op_throughput` are inserted using
    /// `DEFAULT_THROUGHPUT_FLOPS` (1 TFLOP/s) as the baseline. Factors
    /// that are non-positive or non-finite are skipped.
    ///
    /// Part of #4264.
    pub fn apply_adjustment_factors(&mut self, factors: &BTreeMap<String, f64>) {
        for (op_name, &factor) in factors {
            if !factor.is_finite() || factor <= 0.0 {
                continue;
            }
            // factor = actual / predicted. If factor > 1.0, the model
            // underestimates cost (actual > predicted), meaning throughput
            // is too high. Divide throughput by factor to slow the estimate.
            // If factor < 1.0, model overestimates, so throughput needs to
            // increase (divide by factor < 1.0 = multiply by 1/factor).
            let current = self
                .op_throughput
                .get(op_name)
                .copied()
                .unwrap_or(DEFAULT_THROUGHPUT_FLOPS);
            self.op_throughput.insert(op_name.clone(), current / factor);
        }
    }

    /// Apply adjustment factors from a [`CalibrationReport`] to this cost model.
    ///
    /// Convenience method that chains [`CalibrationReport::adjustment_factors()`]
    /// with [`apply_adjustment_factors()`](Self::apply_adjustment_factors).
    ///
    /// Part of #4264.
    pub fn apply_calibration_report(&mut self, report: &CalibrationReport) {
        let factors = report.adjustment_factors();
        self.apply_adjustment_factors(&factors);
    }

    /// Calibrate the cost model by comparing predicted and actual timings.
    ///
    /// Matches predictions to actuals by step name. Only steps present in
    /// both slices contribute to the report. Returns an error if no names
    /// match or if any matched step has a non-positive actual time.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Pairs of `(step_name, predicted_ns)` from the cost
    ///   model's estimates.
    /// * `actuals` - Pairs of `(step_name, actual_ns)` from GPU profiling.
    ///
    /// # Errors
    ///
    /// Returns [`CalibrationError::NoMatchingSteps`] if no step names are
    /// shared between `predictions` and `actuals`.
    ///
    /// Returns [`CalibrationError::NonPositiveActual`] if a matched step
    /// has `actual_ns <= 0.0`.
    pub fn calibrate(
        predictions: &[(String, f64)],
        actuals: &[(String, f64)],
    ) -> Result<CalibrationReport, CalibrationError> {
        // Index actuals by name for O(n) matching.
        let actual_map: HashMap<&str, f64> = actuals
            .iter()
            .map(|(name, ns)| (name.as_str(), *ns))
            .collect();

        let mut entries = Vec::new();
        for (name, pred_ns) in predictions {
            if let Some(&act_ns) = actual_map.get(name.as_str()) {
                entries.push(CalibrationData {
                    step_name: name.clone(),
                    predicted_ns: *pred_ns,
                    actual_ns: act_ns,
                });
            }
        }

        if entries.is_empty() {
            return Err(CalibrationError::NoMatchingSteps);
        }

        // Validate all actual values are positive before computing ratios.
        for entry in &entries {
            if !entry.actual_ns.is_finite() || entry.actual_ns <= 0.0 {
                return Err(CalibrationError::NonPositiveActual {
                    name: entry.step_name.clone(),
                    actual_ns: entry.actual_ns,
                });
            }
        }

        let paired: Vec<(f64, f64)> = entries
            .iter()
            .map(|e| (e.predicted_ns, e.actual_ns))
            .collect();

        let n = paired.len() as f64;
        let mut sum_abs_err = 0.0;
        let mut max_over = 0.0_f64;
        let mut max_under = 0.0_f64;
        let mut sum_ratio = 0.0_f64;
        let mut max_ratio = 0.0_f64;

        for &(pred, act) in &paired {
            let err = pred - act;
            sum_abs_err += err.abs();
            max_over = max_over.max(err);
            max_under = max_under.max(-err);
            let ratio = pred / act;
            sum_ratio += ratio;
            max_ratio = max_ratio.max(ratio);
        }

        max_over = max_over.max(0.0);
        max_under = max_under.max(0.0);

        let correlation = if paired.len() < 2 {
            f64::NAN
        } else {
            pearson_r(&paired)
        };

        Ok(CalibrationReport {
            mean_absolute_error_ns: sum_abs_err / n,
            max_overestimate_ns: max_over,
            max_underestimate_ns: max_under,
            correlation,
            mean_error_ratio: sum_ratio / n,
            max_error_ratio: max_ratio,
            entries,
        })
    }
}

/// Pearson correlation coefficient for (x, y) pairs.
///
/// Returns NaN if either variable has zero variance.
fn pearson_r(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    let sum_x: f64 = pairs.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = pairs.iter().map(|(_, y)| y).sum();
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for &(x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < f64::EPSILON {
        f64::NAN
    } else {
        cov / denom
    }
}

impl CostEstimate {
    /// Returns the N most expensive steps, sorted by cost descending.
    ///
    /// Each entry is `(step_index, cost_ns)`. If the plan has fewer than
    /// `n` costed steps, returns all of them.
    #[must_use]
    pub fn top_expensive_steps(&self, n: usize) -> Vec<(usize, f64)> {
        let mut sorted: Vec<_> = self.per_step_ns.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    /// Human-readable summary of the cost estimate.
    #[must_use]
    pub fn summarize(&self) -> String {
        let total_us = self.total_ns / 1e3;
        let total_ms = self.total_ns / 1e6;

        let mut summary = format!(
            "CostEstimate: {:.1} us ({:.3} ms), {} dispatches",
            total_us, total_ms, self.dispatch_count
        );

        if !self.per_step_ns.is_empty() {
            summary.push_str(&format!(", {} costed steps", self.per_step_ns.len()));

            // Show top-3 most expensive steps.
            let mut sorted: Vec<_> = self.per_step_ns.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top = sorted.iter().take(3);
            for (i, (step_idx, ns)) in top.enumerate() {
                summary.push_str(&format!(
                    "\n  #{}: step {} = {:.1} us",
                    i + 1,
                    step_idx,
                    ns / 1e3
                ));
            }
        }

        summary
    }
}

impl fmt::Display for CostEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summarize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
    use crate::trace_compile::{CompiledKernel, CompiledPlan, CompiledStep};

    /// Build a minimal `CompiledPlan` with no steps.
    fn empty_plan() -> CompiledPlan {
        CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        }
    }

    /// Build a `CompiledKernel` with a given name and output shape.
    fn make_kernel(name: &str, output_shape: &[usize]) -> CompiledKernel {
        let output_id = TensorNodeId::new(1);
        let node = TensorNode::new(
            output_id,
            TensorOpKind::Input {
                name: "x".to_string(),
                shape: output_shape.to_vec(),
            },
            output_shape.to_vec(),
        );
        let def = TensorKernelDef {
            name: name.to_string(),
            nodes: vec![node],
            output: output_id,
        };
        CompiledKernel::new(def)
    }

    #[test]
    fn test_cost_model_empty_plan_zero_cost() {
        let model = CostModel::apple_m4();
        let plan = empty_plan();
        let est = model.estimate(&plan);
        assert_eq!(est.total_ns, 0.0);
        assert_eq!(est.dispatch_count, 0);
        assert!(est.per_step_ns.is_empty());
    }

    #[test]
    fn test_cost_model_single_dispatch_includes_launch_overhead() {
        let model = CostModel::apple_m4();
        let kernel = make_kernel("gelu", &[1, 256]);
        let plan = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![1, 256]],
            output_step: 0,
            weight_names: vec![],
        };

        let est = model.estimate(&plan);
        assert_eq!(est.dispatch_count, 1);
        assert!(
            est.total_ns >= model.launch_overhead_ns,
            "total_ns ({}) should be >= launch_overhead_ns ({})",
            est.total_ns,
            model.launch_overhead_ns
        );
        assert_eq!(est.per_step_ns.len(), 1);
    }

    #[test]
    fn test_cost_model_passthrough_is_free() {
        let model = CostModel::apple_m4();
        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::Passthrough {
                    op_name: "reshape".to_string(),
                    output_shape: vec![1, 256],
                },
                CompiledStep::IdentityPassthrough,
            ],
            input_shapes: vec![vec![1, 256]],
            output_step: 1,
            weight_names: vec![],
        };

        let est = model.estimate(&plan);
        assert_eq!(est.total_ns, 0.0);
        assert_eq!(est.dispatch_count, 0);
    }

    #[test]
    fn test_cost_model_m4_produces_reasonable_estimates() {
        let model = CostModel::apple_m4();
        // Simulate a 1M-element kernel (~4 MB at f32).
        let kernel = make_kernel("matmul", &[1024, 1024]);
        let plan = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![1024, 1024]],
            output_step: 0,
            weight_names: vec![],
        };

        let est = model.estimate(&plan);
        // 1M elements * 4 bytes * 2 (read+write) = 8 MB.
        // At 400 GB/s: 8e6 / 400e9 * 1e9 = 20 ns memory time.
        // 1M elements / 1e12 FLOP/s * 1e9 = 1.048576 ns compute time.
        // Memory-bound => ~20 ns + 2000 ns launch = ~2020 ns total.
        assert!(est.total_ns > 2000.0, "should include launch overhead");
        assert!(
            est.total_ns < 100_000.0,
            "1M elements should cost < 100 us on M4"
        );
    }

    #[test]
    fn test_cost_model_occupancy_penalty_non_aligned() {
        let model = CostModel::apple_m4();
        // 33 elements: 1 remainder in a 32-wide SIMD group.
        let kernel_aligned = make_kernel("relu", &[32]);
        let kernel_unaligned = make_kernel("relu", &[33]);

        let plan_aligned = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel: kernel_aligned,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![32]],
            output_step: 0,
            weight_names: vec![],
        };
        let plan_unaligned = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel: kernel_unaligned,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![33]],
            output_step: 0,
            weight_names: vec![],
        };

        let est_aligned = model.estimate(&plan_aligned);
        let est_unaligned = model.estimate(&plan_unaligned);
        // Unaligned should have a higher cost due to occupancy penalty,
        // even though it has only 1 more element.
        assert!(
            est_unaligned.total_ns > est_aligned.total_ns,
            "unaligned ({}) should cost more than aligned ({})",
            est_unaligned.total_ns,
            est_aligned.total_ns
        );
    }

    #[test]
    fn test_cost_estimate_summarize_not_empty() {
        let model = CostModel::apple_m4();
        let kernel = make_kernel("softmax", &[1, 512]);
        let plan = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![1, 512]],
            output_step: 0,
            weight_names: vec![],
        };

        let est = model.estimate(&plan);
        let summary = est.summarize();
        assert!(summary.contains("CostEstimate:"));
        assert!(summary.contains("1 dispatches"));
    }

    #[test]
    fn test_cost_estimate_display_matches_summarize() {
        let est = CostEstimate {
            total_ns: 5000.0,
            per_step_ns: vec![(0, 5000.0)],
            dispatch_count: 1,
        };
        assert_eq!(format!("{est}"), est.summarize());
    }

    #[test]
    fn test_top_expensive_steps_sorted_descending() {
        let est = CostEstimate {
            total_ns: 10000.0,
            per_step_ns: vec![(0, 1000.0), (1, 5000.0), (2, 3000.0), (3, 1000.0)],
            dispatch_count: 4,
        };
        let top2 = est.top_expensive_steps(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0], (1, 5000.0));
        assert_eq!(top2[1], (2, 3000.0));
    }

    #[test]
    fn test_top_expensive_steps_n_exceeds_count() {
        let est = CostEstimate {
            total_ns: 2000.0,
            per_step_ns: vec![(0, 2000.0)],
            dispatch_count: 1,
        };
        let top5 = est.top_expensive_steps(5);
        assert_eq!(top5.len(), 1, "should return all when n > available");
    }

    #[test]
    fn test_top_expensive_steps_empty() {
        let est = CostEstimate {
            total_ns: 0.0,
            per_step_ns: vec![],
            dispatch_count: 0,
        };
        let top3 = est.top_expensive_steps(3);
        assert!(top3.is_empty());
    }

    // ---- M4 Max preset and calibration tests ----

    #[test]
    fn test_cost_model_m4_max_preset_fields() {
        let model = CostModel::apple_m4_max();
        assert_eq!(model.launch_overhead_ns, 1500.0);
        assert_eq!(model.bandwidth_bytes_per_sec, 400e9);
        assert_eq!(model.simd_width, 32);
        // M4 Max should have op-specific throughputs populated.
        assert!(
            !model.op_throughput.is_empty(),
            "apple_m4_max should have op-specific throughputs"
        );
        assert!(
            model.op_throughput.contains_key("matmul"),
            "should have matmul throughput"
        );
        assert!(
            model.op_throughput.contains_key("softmax"),
            "should have softmax throughput"
        );
    }

    #[test]
    fn test_cost_monotonic_with_dispatch_count() {
        // Cost estimates must be monotonically increasing as dispatch count grows.
        let model = CostModel::apple_m4_max();
        let kernel = make_kernel("gelu", &[1, 1024]);

        let mut costs = Vec::new();
        for n in 1..=5 {
            let steps: Vec<CompiledStep> = (0..n)
                .map(|_| CompiledStep::Dispatch {
                    kernel: kernel.clone(),
                    weight_data: Default::default(),
                    external_node_ids: None,
                })
                .collect();
            let plan = CompiledPlan {
                steps,
                input_shapes: vec![vec![1, 1024]],
                output_step: 0,
                weight_names: vec![],
            };
            let est = model.estimate(&plan);
            costs.push(est.total_ns);
        }

        for i in 1..costs.len() {
            assert!(
                costs[i] > costs[i - 1],
                "cost should increase with dispatch count: {} dispatches ({:.1} ns) \
                 <= {} dispatches ({:.1} ns)",
                i + 1,
                costs[i],
                i,
                costs[i - 1]
            );
        }
    }

    #[test]
    fn test_cost_larger_matrices_cost_more() {
        // Larger tensors must produce higher cost estimates than smaller ones.
        let model = CostModel::apple_m4_max();

        let sizes: &[&[usize]] = &[
            &[32, 32],     // 1K elements
            &[128, 128],   // 16K elements
            &[512, 512],   // 256K elements
            &[1024, 1024], // 1M elements
        ];

        let mut costs = Vec::new();
        for shape in sizes {
            let kernel = make_kernel("matmul", shape);
            let plan = CompiledPlan {
                steps: vec![CompiledStep::Dispatch {
                    kernel,
                    weight_data: Default::default(),
                    external_node_ids: None,
                }],
                input_shapes: vec![shape.to_vec()],
                output_step: 0,
                weight_names: vec![],
            };
            let est = model.estimate(&plan);
            costs.push(est.total_ns);
        }

        for i in 1..costs.len() {
            assert!(
                costs[i] > costs[i - 1],
                "larger matrix should cost more: shape {:?} ({:.1} ns) \
                 <= shape {:?} ({:.1} ns)",
                sizes[i],
                costs[i],
                sizes[i - 1],
                costs[i - 1]
            );
        }
    }

    #[test]
    fn test_cost_memory_bound_vs_compute_bound() {
        // For M4 Max, softmax has lower throughput (8 TFLOP/s) while matmul
        // has higher throughput (30 TFLOP/s). For the same element count the
        // memory transfer time is identical, but compute_ns differs.
        // The lower-throughput op should cost at least as much as the
        // higher-throughput op, since max(compute, memory) can only be
        // larger when compute_ns is larger.
        let model = CostModel::apple_m4_max();

        let elements = &[10000, 10000]; // 100M elements
        let kernel_matmul = make_kernel("matmul", elements);
        let kernel_softmax = make_kernel("softmax", elements);

        let plan_matmul = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel: kernel_matmul,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![elements.to_vec()],
            output_step: 0,
            weight_names: vec![],
        };
        let plan_softmax = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel: kernel_softmax,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![elements.to_vec()],
            output_step: 0,
            weight_names: vec![],
        };

        let est_matmul = model.estimate(&plan_matmul);
        let est_softmax = model.estimate(&plan_softmax);

        assert!(est_matmul.total_ns > 0.0);
        assert!(est_softmax.total_ns > 0.0);
        assert!(
            est_softmax.total_ns >= est_matmul.total_ns,
            "lower-throughput op (softmax: {:.1} ns) should cost >= \
             higher-throughput op (matmul: {:.1} ns)",
            est_softmax.total_ns,
            est_matmul.total_ns
        );
    }

    #[test]
    fn test_cost_m4_max_faster_than_m4_for_compute_ops() {
        // M4 Max has op-specific throughputs much higher than the base M4's
        // default 1 TFLOP/s, plus lower launch overhead (1500 vs 2000 ns).
        // For compute-heavy ops, M4 Max should produce lower cost estimates.
        let m4 = CostModel::apple_m4();
        let m4_max = CostModel::apple_m4_max();

        let kernel = make_kernel("matmul", &[1024, 1024]);
        let plan = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![1024, 1024]],
            output_step: 0,
            weight_names: vec![],
        };

        let est_m4 = m4.estimate(&plan);
        let est_m4_max = m4_max.estimate(&plan);

        assert!(
            est_m4_max.total_ns < est_m4.total_ns,
            "M4 Max ({:.1} ns) should be faster than base M4 ({:.1} ns) for matmul",
            est_m4_max.total_ns,
            est_m4.total_ns
        );

        let speedup = est_m4.total_ns / est_m4_max.total_ns;
        assert!(
            speedup > 1.0,
            "speedup ratio should be > 1.0, got {speedup:.3}"
        );
    }

    #[test]
    fn test_cost_m4_max_unknown_op_uses_default_throughput() {
        // An op name not in the M4 Max throughput table should fall back
        // to DEFAULT_THROUGHPUT_FLOPS (1 TFLOP/s), same as base M4.
        let m4 = CostModel::apple_m4();
        let m4_max = CostModel::apple_m4_max();

        let kernel = make_kernel("exotic_custom_op", &[1024]);
        let plan = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![1024]],
            output_step: 0,
            weight_names: vec![],
        };

        let est_m4 = m4.estimate(&plan);
        let est_m4_max = m4_max.estimate(&plan);

        // Both use the same default throughput and same bandwidth, so the
        // only difference is launch_overhead_ns (2000 vs 1500).
        let overhead_diff = m4.launch_overhead_ns - m4_max.launch_overhead_ns;
        let cost_diff = est_m4.total_ns - est_m4_max.total_ns;

        assert!(
            (cost_diff - overhead_diff).abs() < 1e-6,
            "cost difference ({cost_diff:.6}) should equal launch overhead difference ({overhead_diff:.6}) \
             for unknown ops"
        );
    }

    #[test]
    fn test_cost_m4_max_roofline_manual_check() {
        // Verify the roofline model against manual calculation for M4 Max.
        //   elements = 1024 * 1024 = 1_048_576
        //   matmul throughput = 30 TFLOP/s
        //   compute_ns = 1_048_576 / 30e12 * 1e9
        //   memory_ns  = 1_048_576 * 8 / 400e9 * 1e9
        //   occupancy  = 1.0 (1_048_576 is a multiple of 32)
        //   total = 1500 + max(compute_ns, memory_ns)
        let model = CostModel::apple_m4_max();
        let kernel = make_kernel("matmul", &[1024, 1024]);
        let plan = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![1024, 1024]],
            output_step: 0,
            weight_names: vec![],
        };

        let est = model.estimate(&plan);
        let elements = 1024.0 * 1024.0;
        let compute_ns = (elements / 30e12) * 1e9;
        let memory_ns = (elements * 4.0 * 2.0 / 400e9) * 1e9;
        let expected = 1500.0 + f64::max(compute_ns, memory_ns);

        assert!(
            (est.total_ns - expected).abs() < 1e-3,
            "roofline estimate ({:.6} ns) should match manual calculation ({:.6} ns)",
            est.total_ns,
            expected
        );
    }

    // ---- Calibration plan and report tests ----

    #[test]
    fn test_calibration_plan_empty_plan_produces_empty() {
        let model = CostModel::apple_m4();
        let plan = empty_plan();
        let records = model.calibration_plan(&plan);
        assert!(
            records.is_empty(),
            "empty plan should produce no calibration records"
        );
    }

    #[test]
    fn test_calibration_plan_matches_dispatch_step_count() {
        let model = CostModel::apple_m4_max();
        let k1 = make_kernel("matmul", &[512, 512]);
        let k2 = make_kernel("gelu", &[1, 1024]);
        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::Dispatch {
                    kernel: k1,
                    weight_data: Default::default(),
                    external_node_ids: None,
                },
                CompiledStep::Passthrough {
                    op_name: "reshape".to_string(),
                    output_shape: vec![1, 512, 512],
                },
                CompiledStep::Dispatch {
                    kernel: k2,
                    weight_data: Default::default(),
                    external_node_ids: None,
                },
                CompiledStep::IdentityPassthrough,
            ],
            input_shapes: vec![vec![512, 512]],
            output_step: 3,
            weight_names: vec![],
        };

        let records = model.calibration_plan(&plan);
        // Only 2 Dispatch steps should produce records.
        assert_eq!(records.len(), 2, "should have one record per dispatch step");
        assert_eq!(records[0].step_index, 1);
        assert_eq!(records[0].op_name, "matmul");
        assert_eq!(records[1].step_index, 3);
        assert_eq!(records[1].op_name, "gelu");
        // All actual_ns should be None (not yet profiled).
        for r in &records {
            assert!(
                r.actual_ns.is_none(),
                "actual_ns should be None before profiling"
            );
            assert!(r.estimated_ns > 0.0, "estimated_ns should be positive");
        }
    }

    #[test]
    fn test_calibration_plan_estimated_ns_matches_estimate() {
        // Verify that calibration_plan estimated_ns values match estimate().
        let model = CostModel::apple_m4_max();
        let kernel = make_kernel("softmax", &[1, 2048]);
        let plan = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![1, 2048]],
            output_step: 0,
            weight_names: vec![],
        };

        let est = model.estimate(&plan);
        let records = model.calibration_plan(&plan);
        assert_eq!(records.len(), 1);
        assert!(
            (records[0].estimated_ns - est.total_ns).abs() < 1e-6,
            "calibration estimated_ns ({:.6}) should match estimate total_ns ({:.6})",
            records[0].estimated_ns,
            est.total_ns,
        );
    }

    #[test]
    fn test_calibration_plan_memory_bound_classification() {
        // With base M4 (default 1 TFLOP/s, 400 GB/s):
        // arithmetic intensity = 1 FLOP / 8 bytes = 0.125 FLOP/byte
        // machine balance = 1e12 / 400e9 = 2.5 FLOP/byte
        // 0.125 < 2.5 => memory-bound.
        let model = CostModel::apple_m4();
        let kernel = make_kernel("relu", &[1024]);
        let plan = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![1024]],
            output_step: 0,
            weight_names: vec![],
        };

        let records = model.calibration_plan(&plan);
        assert_eq!(records.len(), 1);
        assert!(
            records[0].is_memory_bound,
            "small elementwise op should be memory-bound on base M4"
        );
    }

    #[test]
    fn test_calibration_report_no_actuals_all_zero() {
        let records = vec![CalibrationRecord {
            step_index: 0,
            estimated_ns: 5000.0,
            actual_ns: None,
            op_name: "matmul".to_string(),
            is_memory_bound: false,
        }];
        let report = CalibrationReport::from_records(&records);
        assert_eq!(report.mean_absolute_error_ns, 0.0);
        assert_eq!(report.max_overestimate_ns, 0.0);
        assert_eq!(report.max_underestimate_ns, 0.0);
        assert!(
            report.correlation.is_nan(),
            "correlation should be NaN with no actuals"
        );
    }

    #[test]
    fn test_calibration_report_empty_records() {
        let report = CalibrationReport::from_records(&[]);
        assert_eq!(report.mean_absolute_error_ns, 0.0);
        assert!(report.correlation.is_nan());
    }

    #[test]
    fn test_calibration_report_perfect_prediction() {
        // When estimated == actual for all records, MAE = 0 and r = 1.0.
        let records = vec![
            CalibrationRecord {
                step_index: 0,
                estimated_ns: 1000.0,
                actual_ns: Some(1000.0),
                op_name: "a".to_string(),
                is_memory_bound: true,
            },
            CalibrationRecord {
                step_index: 1,
                estimated_ns: 3000.0,
                actual_ns: Some(3000.0),
                op_name: "b".to_string(),
                is_memory_bound: false,
            },
            CalibrationRecord {
                step_index: 2,
                estimated_ns: 5000.0,
                actual_ns: Some(5000.0),
                op_name: "c".to_string(),
                is_memory_bound: false,
            },
        ];
        let report = CalibrationReport::from_records(&records);
        assert!(
            report.mean_absolute_error_ns.abs() < 1e-10,
            "MAE should be 0 for perfect prediction, got {}",
            report.mean_absolute_error_ns
        );
        assert!(
            report.max_overestimate_ns.abs() < 1e-10,
            "max overestimate should be 0"
        );
        assert!(
            report.max_underestimate_ns.abs() < 1e-10,
            "max underestimate should be 0"
        );
        assert!(
            (report.correlation - 1.0).abs() < 1e-10,
            "correlation should be 1.0, got {}",
            report.correlation
        );
    }

    #[test]
    fn test_calibration_report_overestimate_and_underestimate() {
        let records = vec![
            CalibrationRecord {
                step_index: 0,
                estimated_ns: 5000.0,
                actual_ns: Some(3000.0), // overestimate by 2000
                op_name: "a".to_string(),
                is_memory_bound: true,
            },
            CalibrationRecord {
                step_index: 1,
                estimated_ns: 1000.0,
                actual_ns: Some(4000.0), // underestimate by 3000
                op_name: "b".to_string(),
                is_memory_bound: false,
            },
        ];
        let report = CalibrationReport::from_records(&records);

        // MAE = (2000 + 3000) / 2 = 2500
        assert!(
            (report.mean_absolute_error_ns - 2500.0).abs() < 1e-6,
            "MAE should be 2500, got {}",
            report.mean_absolute_error_ns
        );
        // max_overestimate = 2000 (step 0: 5000 - 3000)
        assert!(
            (report.max_overestimate_ns - 2000.0).abs() < 1e-6,
            "max overestimate should be 2000, got {}",
            report.max_overestimate_ns
        );
        // max_underestimate = 3000 (step 1: 4000 - 1000)
        assert!(
            (report.max_underestimate_ns - 3000.0).abs() < 1e-6,
            "max underestimate should be 3000, got {}",
            report.max_underestimate_ns
        );
        // With 2 points, correlation is well-defined (negative in this case).
        assert!(
            report.correlation.is_finite(),
            "correlation should be finite with 2 points"
        );
        // est=[5000,1000] vs act=[3000,4000] => negative correlation.
        assert!(
            report.correlation < 0.0,
            "correlation should be negative (inverted), got {}",
            report.correlation
        );
    }

    #[test]
    fn test_calibration_report_single_actual_nan_correlation() {
        // With a single profiled record, correlation is undefined (NaN).
        let records = vec![CalibrationRecord {
            step_index: 0,
            estimated_ns: 2000.0,
            actual_ns: Some(2500.0),
            op_name: "a".to_string(),
            is_memory_bound: true,
        }];
        let report = CalibrationReport::from_records(&records);
        assert!(
            (report.mean_absolute_error_ns - 500.0).abs() < 1e-6,
            "MAE should be 500, got {}",
            report.mean_absolute_error_ns
        );
        assert!(
            report.correlation.is_nan(),
            "correlation should be NaN with only 1 profiled record"
        );
    }

    #[test]
    fn test_calibration_report_mixed_profiled_and_unprofiled() {
        // Only records with actual_ns contribute to the report.
        let records = vec![
            CalibrationRecord {
                step_index: 0,
                estimated_ns: 1000.0,
                actual_ns: Some(1200.0),
                op_name: "a".to_string(),
                is_memory_bound: true,
            },
            CalibrationRecord {
                step_index: 1,
                estimated_ns: 5000.0,
                actual_ns: None, // not yet profiled
                op_name: "b".to_string(),
                is_memory_bound: false,
            },
            CalibrationRecord {
                step_index: 2,
                estimated_ns: 3000.0,
                actual_ns: Some(2800.0),
                op_name: "c".to_string(),
                is_memory_bound: false,
            },
        ];
        let report = CalibrationReport::from_records(&records);
        // Only 2 profiled records: errors are |1000-1200|=200, |3000-2800|=200.
        assert!(
            (report.mean_absolute_error_ns - 200.0).abs() < 1e-6,
            "MAE should be 200 (only profiled records), got {}",
            report.mean_absolute_error_ns
        );
    }

    // ---- Hardware preset tests ----

    /// Helper: verify common constraints across all presets.
    fn assert_preset_reasonable(model: &CostModel, name: &str) {
        assert!(
            model.launch_overhead_ns > 0.0,
            "{name}: launch_overhead_ns must be positive"
        );
        assert!(
            model.bandwidth_bytes_per_sec > 0.0,
            "{name}: bandwidth_bytes_per_sec must be positive"
        );
        assert!(
            model.simd_width == 32,
            "{name}: simd_width should be 32 (GPU warp/simdgroup size)"
        );
        assert!(
            !model.op_throughput.is_empty(),
            "{name}: should have op-specific throughputs"
        );
        // All throughput values must be positive.
        for (op, &tflops) in &model.op_throughput {
            assert!(
                tflops > 0.0,
                "{name}: throughput for '{op}' must be positive, got {tflops}"
            );
        }
    }

    #[test]
    fn test_apple_m1_preset_fields() {
        let model = CostModel::apple_m1();
        assert_preset_reasonable(&model, "apple_m1");
        assert_eq!(model.launch_overhead_ns, 3000.0);
        assert_eq!(model.bandwidth_bytes_per_sec, 68.25e9);
        assert!(model.op_throughput.contains_key("matmul"));
        assert!(model.op_throughput.contains_key("softmax"));
    }

    #[test]
    fn test_apple_m2_preset_fields() {
        let model = CostModel::apple_m2();
        assert_preset_reasonable(&model, "apple_m2");
        assert_eq!(model.launch_overhead_ns, 2500.0);
        assert_eq!(model.bandwidth_bytes_per_sec, 100e9);
        assert!(model.op_throughput.contains_key("matmul"));
        assert!(model.op_throughput.contains_key("conv1d"));
    }

    #[test]
    fn test_apple_m3_preset_fields() {
        let model = CostModel::apple_m3();
        assert_preset_reasonable(&model, "apple_m3");
        assert_eq!(model.launch_overhead_ns, 2000.0);
        assert_eq!(model.bandwidth_bytes_per_sec, 100e9);
        assert!(model.op_throughput.contains_key("matmul"));
        assert!(model.op_throughput.contains_key("layer_norm"));
    }

    #[test]
    fn test_apple_m4_pro_preset_fields() {
        let model = CostModel::apple_m4_pro();
        assert_preset_reasonable(&model, "apple_m4_pro");
        assert_eq!(model.launch_overhead_ns, 1800.0);
        assert_eq!(model.bandwidth_bytes_per_sec, 273e9);
        assert!(model.op_throughput.contains_key("matmul"));
        assert!(model.op_throughput.contains_key("instance_norm"));
    }

    #[test]
    fn test_nvidia_a100_preset_fields() {
        let model = CostModel::nvidia_a100();
        assert_preset_reasonable(&model, "nvidia_a100");
        assert_eq!(model.launch_overhead_ns, 5000.0);
        assert_eq!(model.bandwidth_bytes_per_sec, 2039e9);
        assert!(model.op_throughput.contains_key("matmul"));
        assert!(model.op_throughput.contains_key("gelu"));
    }

    #[test]
    fn test_nvidia_rtx_4090_preset_fields() {
        let model = CostModel::nvidia_rtx_4090();
        assert_preset_reasonable(&model, "nvidia_rtx_4090");
        assert_eq!(model.launch_overhead_ns, 7000.0);
        assert_eq!(model.bandwidth_bytes_per_sec, 1008e9);
        assert!(model.op_throughput.contains_key("matmul"));
        assert!(model.op_throughput.contains_key("silu"));
    }

    #[test]
    fn test_apple_silicon_generation_ordering() {
        // M1 < M2 < M3 < M4 < M4 Pro < M4 Max in matmul throughput.
        let m1 = CostModel::apple_m1();
        let m2 = CostModel::apple_m2();
        let m3 = CostModel::apple_m3();
        let m4 = CostModel::apple_m4(); // base M4 uses default 1 TFLOP/s
        let m4_pro = CostModel::apple_m4_pro();
        let m4_max = CostModel::apple_m4_max();

        let t_m1 = m1.op_throughput["matmul"];
        let t_m2 = m2.op_throughput["matmul"];
        let t_m3 = m3.op_throughput["matmul"];
        let t_m4_pro = m4_pro.op_throughput["matmul"];
        let t_m4_max = m4_max.op_throughput["matmul"];

        assert!(t_m1 < t_m2, "M1 matmul < M2 matmul");
        assert!(t_m2 < t_m3, "M2 matmul < M3 matmul");
        assert!(t_m3 < t_m4_pro, "M3 matmul < M4 Pro matmul");
        assert!(t_m4_pro < t_m4_max, "M4 Pro matmul < M4 Max matmul");

        // Base M4 has no op_throughput entries — it uses the default.
        // The default (1 TFLOP/s) is lower than M1's explicit matmul (2 TFLOP/s).
        assert!(
            m4.op_throughput.is_empty(),
            "base M4 should have no op-specific throughputs"
        );
    }

    #[test]
    fn test_apple_silicon_bandwidth_ordering() {
        // Bandwidth should increase with newer generations / higher-end chips.
        let m1 = CostModel::apple_m1();
        let m2 = CostModel::apple_m2();
        let m4_pro = CostModel::apple_m4_pro();
        let m4_max = CostModel::apple_m4_max();

        assert!(
            m1.bandwidth_bytes_per_sec < m2.bandwidth_bytes_per_sec,
            "M1 bandwidth < M2 bandwidth"
        );
        assert!(
            m2.bandwidth_bytes_per_sec < m4_pro.bandwidth_bytes_per_sec,
            "M2 bandwidth < M4 Pro bandwidth"
        );
        assert!(
            m4_pro.bandwidth_bytes_per_sec < m4_max.bandwidth_bytes_per_sec,
            "M4 Pro bandwidth < M4 Max bandwidth"
        );
    }

    #[test]
    fn test_nvidia_higher_bandwidth_than_apple() {
        // Datacenter GPUs (A100) and high-end consumer (RTX 4090) both
        // have higher memory bandwidth than Apple Silicon.
        let m4_max = CostModel::apple_m4_max();
        let a100 = CostModel::nvidia_a100();
        let rtx4090 = CostModel::nvidia_rtx_4090();

        assert!(
            a100.bandwidth_bytes_per_sec > m4_max.bandwidth_bytes_per_sec,
            "A100 bandwidth ({:.0} GB/s) > M4 Max bandwidth ({:.0} GB/s)",
            a100.bandwidth_bytes_per_sec / 1e9,
            m4_max.bandwidth_bytes_per_sec / 1e9
        );
        assert!(
            rtx4090.bandwidth_bytes_per_sec > m4_max.bandwidth_bytes_per_sec,
            "RTX 4090 bandwidth ({:.0} GB/s) > M4 Max bandwidth ({:.0} GB/s)",
            rtx4090.bandwidth_bytes_per_sec / 1e9,
            m4_max.bandwidth_bytes_per_sec / 1e9
        );
    }

    #[test]
    fn test_nvidia_higher_dispatch_overhead_than_apple() {
        // NVIDIA GPUs have higher dispatch overhead than Apple Silicon
        // due to PCIe/driver overhead.
        let m4_max = CostModel::apple_m4_max();
        let a100 = CostModel::nvidia_a100();
        let rtx4090 = CostModel::nvidia_rtx_4090();

        assert!(
            a100.launch_overhead_ns > m4_max.launch_overhead_ns,
            "A100 launch overhead > M4 Max launch overhead"
        );
        assert!(
            rtx4090.launch_overhead_ns > m4_max.launch_overhead_ns,
            "RTX 4090 launch overhead > M4 Max launch overhead"
        );
    }

    #[test]
    fn test_rtx_4090_higher_matmul_throughput_than_a100() {
        // RTX 4090 has higher F32 matmul throughput than A100 (60 vs 17 TFLOP/s
        // effective in the cost model). Verify this directly.
        let a100 = CostModel::nvidia_a100();
        let rtx4090 = CostModel::nvidia_rtx_4090();

        let a100_matmul = a100.op_throughput["matmul"];
        let rtx4090_matmul = rtx4090.op_throughput["matmul"];

        assert!(
            rtx4090_matmul > a100_matmul,
            "RTX 4090 matmul throughput ({:.0} TFLOP/s) should exceed \
             A100 matmul throughput ({:.0} TFLOP/s)",
            rtx4090_matmul / 1e12,
            a100_matmul / 1e12
        );
    }

    #[test]
    fn test_a100_higher_bandwidth_than_rtx_4090() {
        // A100 HBM2e (2039 GB/s) has ~2x the bandwidth of RTX 4090 GDDR6X
        // (1008 GB/s). For memory-bound workloads, A100 wins.
        let a100 = CostModel::nvidia_a100();
        let rtx4090 = CostModel::nvidia_rtx_4090();

        assert!(
            a100.bandwidth_bytes_per_sec > rtx4090.bandwidth_bytes_per_sec,
            "A100 bandwidth ({:.0} GB/s) should exceed RTX 4090 ({:.0} GB/s)",
            a100.bandwidth_bytes_per_sec / 1e9,
            rtx4090.bandwidth_bytes_per_sec / 1e9
        );

        // Verify A100 is faster on a memory-bound workload (the roofline
        // model treats elementwise ops as 1 FLOP/element, making most
        // dispatches memory-bound).
        let kernel = make_kernel("relu", &[1024, 1024]);
        let plan = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![1024, 1024]],
            output_step: 0,
            weight_names: vec![],
        };

        let est_a100 = a100.estimate(&plan);
        let est_rtx4090 = rtx4090.estimate(&plan);

        assert!(
            est_a100.total_ns < est_rtx4090.total_ns,
            "A100 ({:.1} ns) should be faster than RTX 4090 ({:.1} ns) \
             on memory-bound workload due to higher bandwidth",
            est_a100.total_ns,
            est_rtx4090.total_ns
        );
    }

    #[test]
    fn test_all_presets_produce_positive_estimates() {
        // Every preset should produce positive cost estimates for a
        // non-trivial dispatch plan.
        let presets: Vec<(&str, CostModel)> = vec![
            ("apple_m1", CostModel::apple_m1()),
            ("apple_m2", CostModel::apple_m2()),
            ("apple_m3", CostModel::apple_m3()),
            ("apple_m4", CostModel::apple_m4()),
            ("apple_m4_pro", CostModel::apple_m4_pro()),
            ("apple_m4_max", CostModel::apple_m4_max()),
            ("nvidia_a100", CostModel::nvidia_a100()),
            ("nvidia_rtx_4090", CostModel::nvidia_rtx_4090()),
        ];

        let kernel = make_kernel("matmul", &[512, 512]);
        let plan = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![512, 512]],
            output_step: 0,
            weight_names: vec![],
        };

        for (name, model) in &presets {
            let est = model.estimate(&plan);
            assert!(
                est.total_ns > 0.0,
                "{name}: total_ns should be positive, got {:.6}",
                est.total_ns
            );
            assert_eq!(est.dispatch_count, 1, "{name}: should have 1 dispatch");
        }
    }

    #[test]
    fn test_newer_apple_silicon_faster_on_matmul() {
        // For the same workload, newer Apple Silicon should be faster
        // due to higher throughput and lower launch overhead.
        let m1 = CostModel::apple_m1();
        let m2 = CostModel::apple_m2();
        let m3 = CostModel::apple_m3();

        let kernel = make_kernel("matmul", &[1024, 1024]);
        let plan = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![1024, 1024]],
            output_step: 0,
            weight_names: vec![],
        };

        let est_m1 = m1.estimate(&plan);
        let est_m2 = m2.estimate(&plan);
        let est_m3 = m3.estimate(&plan);

        assert!(
            est_m2.total_ns < est_m1.total_ns,
            "M2 ({:.1} ns) should be faster than M1 ({:.1} ns)",
            est_m2.total_ns,
            est_m1.total_ns
        );
        assert!(
            est_m3.total_ns < est_m2.total_ns,
            "M3 ({:.1} ns) should be faster than M2 ({:.1} ns)",
            est_m3.total_ns,
            est_m2.total_ns
        );
    }

    #[test]
    fn test_dispatch_overhead_ranges() {
        // Apple Silicon: 1-3 us, NVIDIA: 5-10 us.
        let apple_models = vec![
            ("M1", CostModel::apple_m1()),
            ("M2", CostModel::apple_m2()),
            ("M3", CostModel::apple_m3()),
            ("M4", CostModel::apple_m4()),
            ("M4 Pro", CostModel::apple_m4_pro()),
            ("M4 Max", CostModel::apple_m4_max()),
        ];
        for (name, model) in &apple_models {
            assert!(
                model.launch_overhead_ns >= 1000.0 && model.launch_overhead_ns <= 3000.0,
                "{name}: Apple launch overhead ({:.0} ns) should be 1-3 us",
                model.launch_overhead_ns
            );
        }

        let nvidia_models = vec![
            ("A100", CostModel::nvidia_a100()),
            ("RTX 4090", CostModel::nvidia_rtx_4090()),
        ];
        for (name, model) in &nvidia_models {
            assert!(
                model.launch_overhead_ns >= 5000.0 && model.launch_overhead_ns <= 10000.0,
                "{name}: NVIDIA launch overhead ({:.0} ns) should be 5-10 us",
                model.launch_overhead_ns
            );
        }
    }
}

#[cfg(test)]
#[path = "cost_model_tests.rs"]
mod cost_model_tests;

#[cfg(test)]
#[path = "cost_model_calibration_tests.rs"]
mod cost_model_calibration_tests;
