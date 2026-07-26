// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL codegen for the fused LSTM sequence kernel.
//!
//! Generates a Metal Shading Language compute kernel that processes
//! the entire `[seq_len, batch, input_size]` sequence in one dispatch.
//! Each thread handles one (batch, hidden_unit) pair across all timesteps.
//! Threadgroup memory shares the h vector for the `w_hh @ h` dot product.

/// Generate MSL source for the LSTM sequence kernel.
///
/// The kernel processes the full sequence on-GPU with a for-loop over
/// timesteps. Each thread handles one (batch, hidden_unit) pair.
/// Threadgroup memory is used for h-state sharing across threads
/// within a batch element for the w_hh @ h inner product.
///
/// When `reverse` is true, the kernel reads input in reverse timestep order
/// and writes output in reverse order. This eliminates the need for external
/// `flip(dim=0)` dispatches in BiLSTM backward direction, saving ~192 Metal
/// dispatches in Kokoro (45% of total). Part of #1815.
pub(super) fn lstm_sequence_msl(hidden_size: usize) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

// Sigmoid activation.
inline float sigmoid(float x) {{
    return 1.0f / (1.0f + exp(-x));
}}

kernel void lstm_forward_sequence(
    device const float* input      [[buffer(0)]],   // [seq_len, batch, input_size]
    device const float* w_ih       [[buffer(1)]],   // [4*hidden, input_size]
    device const float* w_hh       [[buffer(2)]],   // [4*hidden, hidden_size]
    device const float* bias       [[buffer(3)]],   // [4*hidden] or empty
    device const float* h0         [[buffer(4)]],   // [batch, hidden_size]
    device const float* c0         [[buffer(5)]],   // [batch, hidden_size]
    device float* output           [[buffer(6)]],   // [seq_len, batch, hidden_size]
    device float* h_n              [[buffer(7)]],   // [batch, hidden_size]
    device float* c_n              [[buffer(8)]],   // [batch, hidden_size]
    constant uint& seq_len         [[buffer(9)]],
    constant uint& batch_size      [[buffer(10)]],
    constant uint& input_size      [[buffer(11)]],
    constant uint& hidden_size_val [[buffer(12)]],
    constant uint& has_bias        [[buffer(13)]],
    constant uint& reverse         [[buffer(14)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {{
    uint b = gid.x;   // batch index (one threadgroup per batch element)
    uint h = tid.x;   // hidden unit index (thread within threadgroup)

    if (b >= batch_size || h >= hidden_size_val) return;

    // Threadgroup shared memory for h vector.
    threadgroup float shared_h[{hidden_size}];

    // Initialize from h0, c0.
    float h_val = h0[b * hidden_size_val + h];
    float c_val = c0[b * hidden_size_val + h];

    for (uint t = 0; t < seq_len; t++) {{
        // Compute the actual timestep index. In reverse mode, read/write
        // from the end of the sequence. This eliminates external flip()
        // dispatches for BiLSTM backward direction (#1815).
        uint ts = reverse ? (seq_len - 1 - t) : t;

        // Write current h to shared memory so all threads can read it.
        shared_h[h] = h_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute 4 gates: i, f, g, o
        // gates[g] = sum_k(w_ih[g*H + h, k] * x[t,b,k])
        //          + sum_j(w_hh[g*H + h, j] * shared_h[j])
        //          + bias[g*H + h]
        //
        // Uses Kahan compensated summation to match CPU BLAS precision
        // for large input_size (e.g., 640 for Kokoro). Without compensation,
        // serial f32 accumulation over 640+ terms can diverge from the CPU
        // matmul path enough to saturate LSTM gates. (#2083)
        float gates[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
        float comp[4]  = {{0.0f, 0.0f, 0.0f, 0.0f}};  // Kahan compensation

        // Input contribution: w_ih @ x_ts (Kahan-compensated)
        uint x_base = (ts * batch_size + b) * input_size;
        for (uint k = 0; k < input_size; k++) {{
            float x_val = input[x_base + k];
            for (uint g = 0; g < 4; g++) {{
                float prod = w_ih[(g * hidden_size_val + h) * input_size + k] * x_val - comp[g];
                float new_sum = gates[g] + prod;
                comp[g] = (new_sum - gates[g]) - prod;
                gates[g] = new_sum;
            }}
        }}

        // Hidden contribution: w_hh @ h (Kahan-compensated, continues accumulation)
        for (uint j = 0; j < hidden_size_val; j++) {{
            float hj = shared_h[j];
            for (uint g = 0; g < 4; g++) {{
                float prod = w_hh[(g * hidden_size_val + h) * hidden_size_val + j] * hj - comp[g];
                float new_sum = gates[g] + prod;
                comp[g] = (new_sum - gates[g]) - prod;
                gates[g] = new_sum;
            }}
        }}

        // Add bias if present.
        if (has_bias) {{
            for (uint g = 0; g < 4; g++) {{
                gates[g] += bias[g * hidden_size_val + h];
            }}
        }}

        // Apply gate activations (PyTorch gate order: i, f, g, o).
        float i_gate = sigmoid(gates[0]);
        float f_gate = sigmoid(gates[1]);
        float g_gate = tanh(gates[2]);
        float o_gate = sigmoid(gates[3]);

        // Update cell and hidden state.
        c_val = f_gate * c_val + i_gate * g_gate;
        h_val = o_gate * tanh(c_val);

        // Write output for this timestep (at ts, not t, for reverse mode).
        output[(ts * batch_size + b) * hidden_size_val + h] = h_val;

        // Barrier before next timestep to ensure all threads have updated h_val.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write final states.
    h_n[b * hidden_size_val + h] = h_val;
    c_n[b * hidden_size_val + h] = c_val;
}}
"#
    )
}

/// Generate MSL for the precomputed-input LSTM sequence kernel (F32).
///
/// Takes `input_proj` (`[seq_len, batch, 4*hidden_size]`) instead of raw input +
/// w_ih. The input projection `X @ W_ih.T + bias` is computed externally via
/// simdgroup matmul (parallel across all timesteps), so this kernel only does
/// the sequential `w_hh @ h` recurrence. Reduces per-timestep arithmetic from
/// `input_size + hidden_size` to just `hidden_size` iterations.
///
/// Part of #2981 (LSTM input GEMM pre-computation), restored in #3491.
pub(super) fn lstm_sequence_precomputed_msl(hidden_size: usize) -> String {
    lstm_sequence_precomputed_msl_impl(hidden_size, false)
}

/// Mixed-precision variant of the precomputed LSTM kernel.
///
/// `w_hh`, `h0`, `c0` are `half*` (F16). `input_proj` and outputs are `float*`.
/// Part of #2981, restored in #3491.
pub(super) fn lstm_sequence_precomputed_mixed_msl(hidden_size: usize) -> String {
    lstm_sequence_precomputed_msl_impl(hidden_size, true)
}

fn lstm_sequence_precomputed_msl_impl(hidden_size: usize, mixed: bool) -> String {
    let (w_type, w_cast_open, w_cast_close) = if mixed {
        ("half", "float(", ")")
    } else {
        ("float", "", "")
    };
    let kernel_name = if mixed {
        "lstm_forward_sequence_precomputed_mixed"
    } else {
        "lstm_forward_sequence_precomputed"
    };
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

inline float sigmoid(float x) {{
    return 1.0f / (1.0f + exp(-x));
}}

kernel void {kernel_name}(
    device const float* input_proj   [[buffer(0)]],   // [seq_len, batch, 4*hidden_size]
    device const {w_type}* w_hh      [[buffer(1)]],   // [4*hidden, hidden_size]
    device const {w_type}* h0        [[buffer(2)]],   // [batch, hidden_size]
    device const {w_type}* c0        [[buffer(3)]],   // [batch, hidden_size]
    device float* output             [[buffer(4)]],   // [seq_len, batch, hidden_size]
    device float* h_n                [[buffer(5)]],   // [batch, hidden_size]
    device float* c_n                [[buffer(6)]],   // [batch, hidden_size]
    constant uint& seq_len           [[buffer(7)]],
    constant uint& batch_size        [[buffer(8)]],
    constant uint& hidden_size_val   [[buffer(9)]],
    constant uint& reverse_mode      [[buffer(10)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {{
    uint b = gid.x;
    uint h = tid.x;

    if (b >= batch_size || h >= hidden_size_val) return;

    threadgroup float shared_h[{hidden_size}];

    float h_val = {w_cast_open}h0[b * hidden_size_val + h]{w_cast_close};
    float c_val = {w_cast_open}c0[b * hidden_size_val + h]{w_cast_close};

    for (uint t = 0; t < seq_len; t++) {{
        uint ts = reverse_mode ? (seq_len - 1 - t) : t;

        shared_h[h] = h_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load pre-computed input projection (4 reads, no input loop).
        float gates[4];
        uint proj_base = (ts * batch_size + b) * (4 * hidden_size_val);
        for (uint g = 0; g < 4; g++) {{
            gates[g] = input_proj[proj_base + g * hidden_size_val + h];
        }}

        // Hidden contribution: w_hh @ h (Kahan-compensated).
        float comp[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
        for (uint j = 0; j < hidden_size_val; j++) {{
            float hj = shared_h[j];
            for (uint g = 0; g < 4; g++) {{
                float prod = {w_cast_open}w_hh[(g * hidden_size_val + h) * hidden_size_val + j]{w_cast_close} * hj - comp[g];
                float new_sum = gates[g] + prod;
                comp[g] = (new_sum - gates[g]) - prod;
                gates[g] = new_sum;
            }}
        }}

        float i_gate = sigmoid(gates[0]);
        float f_gate = sigmoid(gates[1]);
        float g_gate = tanh(gates[2]);
        float o_gate = sigmoid(gates[3]);

        c_val = f_gate * c_val + i_gate * g_gate;
        h_val = o_gate * tanh(c_val);

        output[(ts * batch_size + b) * hidden_size_val + h] = h_val;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    h_n[b * hidden_size_val + h] = h_val;
    c_n[b * hidden_size_val + h] = c_val;
}}
"#
    )
}
