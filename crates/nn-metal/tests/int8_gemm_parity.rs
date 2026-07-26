// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! INT8 GEMM numerical parity tests: GPU (Metal simdgroup kernel) vs CPU reference.
//!
//! Validates the `int8_matmul_dequant` MSL kernel by:
//! 1. Quantizing random F32 weights to INT8 (symmetric/asymmetric per-channel)
//! 2. Computing CPU reference: dequantize INT8 weights to F32, then matmul
//! 3. Dispatching the INT8 GEMM Metal kernel directly (on-the-fly dequant in GPU)
//! 4. Comparing GPU output against CPU reference within tolerance
//!
//! The GPU kernel dequantizes INT8->half during tile load and accumulates in F32,
//! so additional half-precision rounding error is expected vs the CPU F32 path.
//!
//! Error budget analysis:
//! - INT8 quantization error per weight: ~scale/2 (rounding to nearest integer)
//! - Half-precision rounding per element: ~2^-10 * |value| (relative)
//! - Accumulation over K steps: error grows as ~sqrt(K) * per-step error
//! - For K=8:  combined atol ~ 1e-2 (small)
//! - For K=256: combined atol ~ 5e-2 (medium)
//! - For K=512: combined atol ~ 8e-2 (large)
//!
//! Part of #3529.

#![cfg(target_os = "macos")]

mod test_utils;

use nn_core::layers::quantized::{quantize_per_channel, Int8Mode, Int8QuantParams};
use nn_core::test_prng::rand_f32_vec;
use nn_metal::{DispatchMode, KernelPipeline, MetalBuffer, PipelineCache};

use test_utils::metal_setup;

// ---------------------------------------------------------------------------
// MSL generation (inlined from int8_gemm_msl.rs to avoid pub(crate) dependency)
// ---------------------------------------------------------------------------

/// Generate INT8 dequantizing GEMM MSL with compile-time M, K, N constants.
///
/// Mirrors `generate_int8_gemm_msl()` from `crates/nn-metal/src/int8_gemm_msl.rs`.
/// Buffer layout: A(f32), W(u8/i8), scale(f32), zp(i32), [bias(f32)], C(f32).
fn generate_int8_gemm_msl(m: usize, k: usize, n: usize, has_bias: bool) -> String {
    let bias_param = if has_bias {
        "    device const float* bias           [[buffer(4)]],\n"
    } else {
        ""
    };
    let out_buf_idx = if has_bias { 5 } else { 4 };
    let bias_add = if has_bias {
        "            val += bias[gc];\n"
    } else {
        ""
    };

    format!(
        r#"#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint TILE = 32;
constant uint SIMD_SIZE = 32;
constant uint PADDED = TILE + 1;
constant uint M_DIM = {m}u;
constant uint K_DIM = {k}u;
constant uint N_DIM = {n}u;

kernel void int8_matmul_dequant(
    device const float*  A              [[buffer(0)]],
    device const uchar*  W              [[buffer(1)]],
    device const float*  scale          [[buffer(2)]],
    device const int*    zero_point     [[buffer(3)]],
{bias_param}    device float*        C              [[buffer({out_buf_idx})]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]]
) {{
    uint tile_row = tgid.y * TILE;
    uint tile_col = tgid.x * TILE;

    threadgroup half  As[TILE * PADDED];
    threadgroup half  Ws[TILE * PADDED];
    threadgroup float tile_out[TILE * PADDED];

    uint sg_col_start = sg_id * 8;

    simdgroup_matrix<float, 8, 8> acc[4];
    for (uint i = 0; i < 4; i++) {{
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);
    }}

    uint tid_linear = sg_id * SIMD_SIZE + lane_id;
    uint num_k_tiles = (K_DIM + TILE - 1) / TILE;

    for (uint kt = 0; kt < num_k_tiles; kt++) {{
        uint k_start = kt * TILE;

        for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {{
            uint row = idx / TILE;
            uint col = idx % TILE;
            uint gr = tile_row + row;
            uint gc = k_start + col;
            float val = (gr < M_DIM && gc < K_DIM) ? A[gr * K_DIM + gc] : 0.0f;
            As[row * PADDED + col] = half(val);
        }}

        for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {{
            uint row = idx / TILE;
            uint col = idx % TILE;
            uint gk = k_start + row;
            uint gn = tile_col + col;
            half w_half = half(0.0h);
            if (gk < K_DIM && gn < N_DIM) {{
                uchar w_raw = W[gn * K_DIM + gk];
                int w_i8 = int(as_type<char>(w_raw));
                float w_f32 = float(w_i8 - zero_point[gn]) * scale[gn];
                w_half = half(w_f32);
            }}
            Ws[row * PADDED + col] = w_half;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE; kk += 8) {{
            simdgroup_matrix<half, 8, 8> Bmat;
            simdgroup_load(Bmat, &Ws[kk * PADDED + sg_col_start], PADDED);
            for (uint ri = 0; ri < 4; ri++) {{
                simdgroup_matrix<half, 8, 8> Amat;
                simdgroup_load(Amat, &As[(ri * 8) * PADDED + kk], PADDED);
                simdgroup_multiply_accumulate(acc[ri], Amat, Bmat, acc[ri]);
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (uint ri = 0; ri < 4; ri++) {{
        simdgroup_store(acc[ri], &tile_out[(ri * 8) * PADDED + sg_col_start], PADDED);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {{
        uint r = idx / TILE;
        uint c = idx % TILE;
        uint gr = tile_row + r;
        uint gc = tile_col + c;
        if (gr < M_DIM && gc < N_DIM) {{
            float val = tile_out[r * PADDED + c];
{bias_add}            C[gr * N_DIM + gc] = val;
        }}
    }}
}}"#
    )
}

/// Threadgroup memory bytes: As(half) + Ws(half) + tile_out(float) = 8448.
const THREADGROUP_BYTES: u64 = 2 * 32 * 33 * 2 + 32 * 33 * 4;

// ---------------------------------------------------------------------------
// CPU reference: dequantize INT8 weights to F32, then matmul
// ---------------------------------------------------------------------------

/// CPU reference for INT8 dequantizing matmul: `C = A @ dequant(W)^T + bias`.
///
/// - `act`: F32 activations `[M, K]` row-major
/// - `weight_u8`: INT8 weights `[N, K]` stored as u8 (reinterpret as i8)
/// - `params`: per-channel scale and zero_point (length N)
/// - `bias`: optional F32 bias `[N]`
///
/// Returns F32 output `[M, N]` row-major.
fn cpu_int8_matmul_ref(
    act: &[f32],
    weight_u8: &[u8],
    params: &Int8QuantParams,
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    // Dequantize all weights to F32 for the reference path.
    let mut w_f32 = vec![0.0_f32; n * k];
    for row in 0..n {
        let scale = params.scale[row];
        let zp = params.zero_point[row];
        for col in 0..k {
            let q_u8 = weight_u8[row * k + col];
            let q_i8 = q_u8 as i8;
            w_f32[row * k + col] = (f32::from(q_i8) - zp as f32) * scale;
        }
    }

    // Matmul: C[i][j] = sum_kk A[i][kk] * W[j][kk]  (W transposed layout)
    let mut output = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for kk in 0..k {
                sum += act[i * k + kk] * w_f32[j * k + kk];
            }
            if let Some(b) = bias {
                sum += b[j];
            }
            output[i * n + j] = sum;
        }
    }
    output
}

// ---------------------------------------------------------------------------
// GPU dispatch helper
// ---------------------------------------------------------------------------

/// Dispatch the INT8 GEMM kernel on Metal and return the F32 output.
///
/// Creates Metal buffers for activations, INT8 weights, scale, zero_point,
/// optional bias, and output. Compiles the generated MSL, dispatches via
/// simdgroup 3D grid, flushes, and reads back the result.
fn gpu_int8_matmul(
    cache: &PipelineCache,
    act: &[f32],
    weight_u8: &[u8],
    scale: &[f32],
    zero_point: &[i32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let has_bias = bias.is_some();
    let ctx = cache.context();

    // Create Metal buffers.
    let act_buf = ctx.create_buffer(act).expect("create activation buffer");
    let weight_buf = ctx.create_buffer(weight_u8).expect("create weight buffer");
    let scale_buf = ctx.create_buffer(scale).expect("create scale buffer");
    let zp_buf = ctx
        .create_buffer(zero_point)
        .expect("create zero_point buffer");
    let bias_buf = bias.map(|b| ctx.create_buffer(b).expect("create bias buffer"));
    let total_output = m * n;
    let out_buf = ctx
        .create_buffer_zeroed(total_output * 4)
        .expect("create output buffer");

    // Generate MSL and compile pipeline.
    let msl_source = generate_int8_gemm_msl(m, k, n, has_bias);
    let param_count = if has_bias { 5 } else { 4 };
    let pipeline = KernelPipeline::from_msl(
        cache,
        &msl_source,
        "int8_matmul_dequant",
        param_count,
        false,
    )
    .expect("compile INT8 GEMM pipeline");

    // Build dispatch plan: simdgroup 32x32 tiles.
    let m_u32 = u32::try_from(m).expect("m fits u32");
    let n_u32 = u32::try_from(n).expect("n fits u32");
    let plan = DispatchMode::Grid3D {
        grid: [n_u32.div_ceil(32), m_u32.div_ceil(32), 1],
        threads: [32, 4, 1],
    }
    .plan()
    .expect("create dispatch plan")
    .with_output_elems(total_output)
    .with_constants(vec![])
    .with_use_threadgroups(true)
    .with_threadgroup_memory_bytes(Some(THREADGROUP_BYTES));

    // Bind buffers: A, W, scale, zero_point, [bias].
    let mut inputs: Vec<&MetalBuffer> = Vec::with_capacity(param_count);
    inputs.push(&act_buf);
    inputs.push(&weight_buf);
    inputs.push(&scale_buf);
    inputs.push(&zp_buf);
    if let Some(ref b) = bias_buf {
        inputs.push(b);
    }
    let offsets = vec![0usize; param_count];

    // Dispatch.
    pipeline
        .dispatch_buffers_with_all_offsets(ctx, &inputs, &offsets, &out_buf, 0, &plan)
        .expect("dispatch INT8 GEMM");

    // Flush GPU and read back result.
    nn_metal::flush().expect("flush GPU");
    let result: &[f32] = out_buf.contents().expect("read output buffer");
    result[..total_output].to_vec()
}

// ---------------------------------------------------------------------------
// Parity comparison
// ---------------------------------------------------------------------------

/// Result of a parity comparison between CPU and GPU outputs.
struct ParityResult {
    max_abs_diff: f32,
    max_abs_diff_idx: usize,
    mean_abs_diff: f32,
    num_nan_gpu: usize,
    num_inf_gpu: usize,
}

/// Compare CPU and GPU output vectors element-wise.
///
/// Returns detailed metrics including max/mean absolute difference and
/// counts of non-finite GPU values (NaN, Inf).
fn compare_outputs(cpu_out: &[f32], gpu_out: &[f32]) -> ParityResult {
    assert_eq!(cpu_out.len(), gpu_out.len(), "output length mismatch");
    let mut max_abs_diff = 0.0_f32;
    let mut max_abs_diff_idx = 0;
    let mut sum_abs_diff = 0.0_f64;
    let mut num_nan = 0_usize;
    let mut num_inf = 0_usize;

    for (i, (&c, &g)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
        if g.is_nan() {
            num_nan += 1;
            continue;
        }
        if g.is_infinite() {
            num_inf += 1;
            continue;
        }
        let diff = (c - g).abs();
        sum_abs_diff += f64::from(diff);
        if diff > max_abs_diff {
            max_abs_diff = diff;
            max_abs_diff_idx = i;
        }
    }

    let finite_count = cpu_out.len() - num_nan - num_inf;
    let mean_abs_diff = if finite_count > 0 {
        (sum_abs_diff / finite_count as f64) as f32
    } else {
        0.0
    };

    ParityResult {
        max_abs_diff,
        max_abs_diff_idx,
        mean_abs_diff,
        num_nan_gpu: num_nan,
        num_inf_gpu: num_inf,
    }
}

/// Run a single parity test case: quantize weights, compute CPU and GPU, compare.
fn run_parity_test(
    label: &str,
    m: usize,
    k: usize,
    n: usize,
    mode: Int8Mode,
    with_bias: bool,
    atol: f32,
) {
    let cache = metal_setup();

    // Deterministic random activations in [-1, 1].
    let seed_act = (m as u64) * 1000 + (k as u64) * 100 + (n as u64);
    let act = rand_f32_vec(seed_act, m * k, -1.0, 1.0);

    // Deterministic random weights in [-0.5, 0.5] (typical trained weight range).
    let seed_w = seed_act.wrapping_add(0x1234_5678);
    let w_f32 = rand_f32_vec(seed_w, n * k, -0.5, 0.5);

    // Quantize weights to INT8.
    let w_tensor =
        nn_core::dyn_tensor::DynTensor::from_vec(w_f32, &[n, k], &nn_core::Device::Cpu)
            .expect("create weight tensor");
    let (q_tensor, params) = quantize_per_channel(&w_tensor, mode).expect("quantize");
    let q_data = q_tensor.to_flat_vec::<u8>().expect("get quantized data");

    // Optional bias.
    let bias_data = if with_bias {
        let seed_b = seed_act.wrapping_add(0xABCD_EF01);
        Some(rand_f32_vec(seed_b, n, -0.2, 0.2))
    } else {
        None
    };

    // CPU reference.
    let cpu_out = cpu_int8_matmul_ref(&act, &q_data, &params, bias_data.as_deref(), m, k, n);

    // GPU result.
    let gpu_out = gpu_int8_matmul(
        &cache,
        &act,
        &q_data,
        &params.scale,
        &params.zero_point,
        bias_data.as_deref(),
        m,
        k,
        n,
    );

    // Compare with detailed diagnostics.
    let parity = compare_outputs(&cpu_out, &gpu_out);

    // GPU must not produce NaN or Inf.
    assert_eq!(
        parity.num_nan_gpu,
        0,
        "{label}: GPU produced {0} NaN values (out of {1} elements)",
        parity.num_nan_gpu,
        gpu_out.len(),
    );
    assert_eq!(
        parity.num_inf_gpu,
        0,
        "{label}: GPU produced {0} Inf values (out of {1} elements)",
        parity.num_inf_gpu,
        gpu_out.len(),
    );

    // Check absolute tolerance.
    assert!(
        parity.max_abs_diff <= atol,
        "{label}: max diff {:.6e} at [{idx}] exceeds atol {atol:.6e}\n  \
         cpu={:.6} gpu={:.6} (mean_diff={:.6e})",
        parity.max_abs_diff,
        cpu_out[parity.max_abs_diff_idx],
        gpu_out[parity.max_abs_diff_idx],
        parity.mean_abs_diff,
        idx = parity.max_abs_diff_idx,
    );

    eprintln!(
        "{label}: PASS (max_diff={:.6e}, mean_diff={:.6e}, atol={atol})",
        parity.max_abs_diff, parity.mean_abs_diff,
    );
}

// ===========================================================================
// Small: M=4, K=8, N=4 (easy to debug, single K-tile)
// ===========================================================================

/// Tolerance note: The GPU kernel dequantizes to half before simdgroup MAC.
/// For small K (8), the half-precision rounding error per accumulation step
/// is ~ 2^-10 ~ 1e-3, and over K=8 steps the accumulated error is small.
/// The INT8 quantization error itself is ~ scale/2 per element.
/// Combined tolerance: 1e-2 is generous for K=8.
const ATOL_SMALL: f32 = 1e-2;

/// Medium tolerance: K=256 accumulates more half-precision rounding.
/// Per-step error ~ 1e-3 * sqrt(256) ~ 1.6e-2 (pessimistic bound).
const ATOL_MEDIUM: f32 = 5e-2;

/// Large tolerance: K=512 with larger M and N. The accumulated half-precision
/// error grows as ~sqrt(K). For K=512: ~1e-3 * sqrt(512) ~ 2.3e-2.
/// Using 8e-2 to leave margin for pathological quantization rounding.
const ATOL_LARGE: f32 = 8e-2;

#[test]
fn test_int8_gemm_parity_small_symmetric_no_bias() {
    run_parity_test(
        "small_sym_nobias",
        4,
        8,
        4,
        Int8Mode::Symmetric,
        false,
        ATOL_SMALL,
    );
}

#[test]
fn test_int8_gemm_parity_small_symmetric_with_bias() {
    run_parity_test(
        "small_sym_bias",
        4,
        8,
        4,
        Int8Mode::Symmetric,
        true,
        ATOL_SMALL,
    );
}

#[test]
fn test_int8_gemm_parity_small_asymmetric_no_bias() {
    run_parity_test(
        "small_asym_nobias",
        4,
        8,
        4,
        Int8Mode::Asymmetric,
        false,
        ATOL_SMALL,
    );
}

#[test]
fn test_int8_gemm_parity_small_asymmetric_with_bias() {
    run_parity_test(
        "small_asym_bias",
        4,
        8,
        4,
        Int8Mode::Asymmetric,
        true,
        ATOL_SMALL,
    );
}

// ===========================================================================
// Medium: M=32, K=256, N=64 (multiple K-tiles, realistic hidden dim)
// ===========================================================================

#[test]
fn test_int8_gemm_parity_medium_symmetric_no_bias() {
    run_parity_test(
        "med_sym_nobias",
        32,
        256,
        64,
        Int8Mode::Symmetric,
        false,
        ATOL_MEDIUM,
    );
}

#[test]
fn test_int8_gemm_parity_medium_symmetric_with_bias() {
    run_parity_test(
        "med_sym_bias",
        32,
        256,
        64,
        Int8Mode::Symmetric,
        true,
        ATOL_MEDIUM,
    );
}

#[test]
fn test_int8_gemm_parity_medium_asymmetric_no_bias() {
    run_parity_test(
        "med_asym_nobias",
        32,
        256,
        64,
        Int8Mode::Asymmetric,
        false,
        ATOL_MEDIUM,
    );
}

#[test]
fn test_int8_gemm_parity_medium_asymmetric_with_bias() {
    run_parity_test(
        "med_asym_bias",
        32,
        256,
        64,
        Int8Mode::Asymmetric,
        true,
        ATOL_MEDIUM,
    );
}

// ===========================================================================
// Large: M=64, K=512, N=128 (multi-tile in all dimensions, LLM-scale)
// ===========================================================================

#[test]
fn test_int8_gemm_parity_large_symmetric_no_bias() {
    run_parity_test(
        "large_sym_nobias",
        64,
        512,
        128,
        Int8Mode::Symmetric,
        false,
        ATOL_LARGE,
    );
}

#[test]
fn test_int8_gemm_parity_large_symmetric_with_bias() {
    run_parity_test(
        "large_sym_bias",
        64,
        512,
        128,
        Int8Mode::Symmetric,
        true,
        ATOL_LARGE,
    );
}

#[test]
fn test_int8_gemm_parity_large_asymmetric_with_bias() {
    run_parity_test(
        "large_asym_bias",
        64,
        512,
        128,
        Int8Mode::Asymmetric,
        true,
        ATOL_LARGE,
    );
}

// ===========================================================================
// Non-tile-aligned dimensions (boundary handling)
// ===========================================================================

/// K=40 is not divisible by TILE=32. The kernel must handle the partial
/// last K-tile correctly (reading zero for out-of-bounds indices).
#[test]
fn test_int8_gemm_parity_k_not_tile_aligned() {
    run_parity_test(
        "k_not_aligned",
        4,
        40,
        4,
        Int8Mode::Symmetric,
        false,
        ATOL_SMALL,
    );
}

/// K=17: smaller than one tile, non-power-of-2.
#[test]
fn test_int8_gemm_parity_k_small_non_aligned() {
    run_parity_test("k17", 4, 17, 4, Int8Mode::Symmetric, false, ATOL_SMALL);
}

/// N not divisible by 32: partial column tile.
#[test]
fn test_int8_gemm_parity_n_not_tile_aligned() {
    run_parity_test(
        "n_not_aligned",
        4,
        32,
        13,
        Int8Mode::Symmetric,
        false,
        ATOL_SMALL,
    );
}

/// M not divisible by 32: partial row tile.
#[test]
fn test_int8_gemm_parity_m_not_tile_aligned() {
    run_parity_test(
        "m_not_aligned",
        7,
        32,
        4,
        Int8Mode::Symmetric,
        false,
        ATOL_SMALL,
    );
}

/// All dimensions non-tile-aligned with symmetric quantization.
#[test]
fn test_int8_gemm_parity_all_non_aligned_symmetric() {
    run_parity_test(
        "all_non_aligned_sym",
        5,
        19,
        11,
        Int8Mode::Symmetric,
        true,
        ATOL_SMALL,
    );
}

/// All dimensions non-tile-aligned with asymmetric quantization.
/// Exercises the partial-tile boundary paths with non-zero zero_points.
#[test]
fn test_int8_gemm_parity_all_non_aligned_asymmetric() {
    run_parity_test(
        "all_non_aligned_asym",
        5,
        19,
        11,
        Int8Mode::Asymmetric,
        true,
        ATOL_SMALL,
    );
}

// ===========================================================================
// Single-row M=1 (common for inference / autoregressive decoding)
// ===========================================================================

#[test]
fn test_int8_gemm_parity_single_row() {
    run_parity_test(
        "single_row",
        1,
        64,
        32,
        Int8Mode::Symmetric,
        true,
        ATOL_SMALL,
    );
}

/// Single row with large K (token inference through a big linear layer).
#[test]
fn test_int8_gemm_parity_single_row_large_k() {
    run_parity_test(
        "single_row_large_k",
        1,
        512,
        128,
        Int8Mode::Symmetric,
        true,
        ATOL_LARGE,
    );
}

// ===========================================================================
// Edge case: zero-weight channel (scale == 0.0 path in quantization)
// ===========================================================================

/// Verifies correct handling when one or more output channels have all-zero
/// weights, producing scale=0.0 in the quantization params. The GPU kernel
/// must produce zero output for those channels (0 * 0.0 = 0.0).
#[test]
fn test_int8_gemm_parity_zero_weight_channel() {
    let cache = metal_setup();

    let m = 4;
    let k = 16;
    let n = 4;

    // Create weights where channel 1 is all zeros.
    let seed = 0xDEAD_BEEFu64;
    let mut w_f32 = rand_f32_vec(seed, n * k, -0.5, 0.5);
    // Zero out channel 1 (row 1 of [N, K]).
    for col in 0..k {
        w_f32[k + col] = 0.0;
    }

    let act = rand_f32_vec(seed.wrapping_add(1), m * k, -1.0, 1.0);

    let w_tensor =
        nn_core::dyn_tensor::DynTensor::from_vec(w_f32, &[n, k], &nn_core::Device::Cpu)
            .expect("create weight tensor");
    let (q_tensor, params) =
        quantize_per_channel(&w_tensor, Int8Mode::Symmetric).expect("quantize");
    let q_data = q_tensor.to_flat_vec::<u8>().expect("get quantized data");

    // Verify scale is 0.0 for the zeroed channel.
    assert_eq!(
        params.scale[1], 0.0,
        "channel 1 scale should be 0.0 for all-zero weights"
    );

    let cpu_out = cpu_int8_matmul_ref(&act, &q_data, &params, None, m, k, n);
    let gpu_out = gpu_int8_matmul(
        &cache,
        &act,
        &q_data,
        &params.scale,
        &params.zero_point,
        None,
        m,
        k,
        n,
    );

    let parity = compare_outputs(&cpu_out, &gpu_out);
    assert_eq!(parity.num_nan_gpu, 0, "zero-weight: GPU NaN");
    assert_eq!(parity.num_inf_gpu, 0, "zero-weight: GPU Inf");

    // Channel 1 output should be exactly zero on both CPU and GPU.
    for row in 0..m {
        let idx = row * n + 1;
        assert_eq!(
            cpu_out[idx], 0.0,
            "CPU output for zero-weight channel should be 0.0"
        );
        assert_eq!(
            gpu_out[idx], 0.0,
            "GPU output for zero-weight channel should be 0.0"
        );
    }

    assert!(
        parity.max_abs_diff <= ATOL_SMALL,
        "zero_weight: max diff {:.6e} exceeds atol {ATOL_SMALL:.6e}",
        parity.max_abs_diff,
    );
    eprintln!(
        "zero_weight_channel: PASS (max_diff={:.6e}, mean_diff={:.6e})",
        parity.max_abs_diff, parity.mean_abs_diff,
    );
}

// ===========================================================================
// K=1 edge case (single element dot product per output)
// ===========================================================================

/// K=1 means each output element is just `activation * dequant(weight) + bias`.
/// No accumulation, so the only error source is half-precision rounding of
/// the single dequantized weight and the single activation value.
#[test]
fn test_int8_gemm_parity_k_equals_1() {
    run_parity_test("k_equals_1", 4, 1, 4, Int8Mode::Symmetric, true, ATOL_SMALL);
}

// ===========================================================================
// Exact tile boundary: M=32, K=32, N=32 (exactly one tile in each dimension)
// ===========================================================================

/// Single complete tile in each dimension. No partial-tile handling needed.
/// This is the "happy path" for the simdgroup kernel.
#[test]
fn test_int8_gemm_parity_exact_tile() {
    run_parity_test(
        "exact_tile",
        32,
        32,
        32,
        Int8Mode::Symmetric,
        false,
        ATOL_SMALL,
    );
}
