// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU neural network operation sub-trait for [`GpuBackend`](super::GpuBackend) decomposition.
//!
//! Contains 27 fused NN methods extracted from the monolithic `GpuBackend` trait:
//! softmax, log_softmax, conv1d, conv2d, conv3d, conv_transpose1d, layer_norm, group_norm,
//! rms_norm, instance_norm, snake_tensor, adain_snake, adain_leaky_relu, rope,
//! lstm_cell, lstm_sequence, clamp, clamp_min, clamp_max, sdpa, sdpa_causal,
//! resize_bilinear, scalar_binary_op, moe_scatter_gather,
//! max_pool2d, avg_pool2d, adaptive_avg_pool2d.
//! All methods are optional (default `None` → CPU/decomposed fallback).

use super::DynTensor;
use crate::Result;

/// GPU neural network operations: fused norm, conv, softmax, LSTM, RoPE.
///
/// All methods return `Option<Result<...>>` — `None` triggers CPU/decomposed
/// fallback, `Some(Ok(..))` returns the GPU result, `Some(Err(e))` propagates.
pub trait GpuNnOps: Send + Sync {
    /// Softmax along a dimension on GPU.
    fn softmax(&self, _x: &DynTensor, _dim: usize) -> Option<Result<DynTensor>> {
        None
    }

    /// Log-softmax along a dimension on GPU.
    fn log_softmax(&self, _x: &DynTensor, _dim: usize) -> Option<Result<DynTensor>> {
        None
    }

    /// 1-D convolution on GPU.
    fn conv1d(
        &self,
        _input: &DynTensor,
        _kernel: &DynTensor,
        _bias: Option<&DynTensor>,
        _padding: usize,
        _stride: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// 2-D convolution on GPU.
    fn conv2d(
        &self,
        _input: &DynTensor,
        _kernel: &DynTensor,
        _bias: Option<&DynTensor>,
        _padding: usize,
        _stride: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// 3-D convolution on GPU.
    ///
    /// Input: `[B, C_in, D, H, W]`, Kernel: `[C_out, C_in/groups, kD, kH, kW]`
    /// Output: `[B, C_out, out_D, out_H, out_W]`
    ///
    /// Needed for 3D patch embeddings (Qwen3-VL vision encoder).
    fn conv3d(
        &self,
        _input: &DynTensor,
        _kernel: &DynTensor,
        _bias: Option<&DynTensor>,
        _padding: [usize; 3],
        _stride: [usize; 3],
        _dilation: [usize; 3],
        _groups: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// 1-D transposed convolution (deconvolution) on GPU.
    fn conv_transpose1d(
        &self,
        _input: &DynTensor,
        _kernel: &DynTensor,
        _bias: Option<&DynTensor>,
        _padding: usize,
        _output_padding: usize,
        _stride: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused LayerNorm on GPU: `(x - mean) / sqrt(var + eps) * weight + bias`.
    fn layer_norm(
        &self,
        _x: &DynTensor,
        _weight: &DynTensor,
        _bias: &DynTensor,
        _eps: f64,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused GroupNorm on GPU.
    fn group_norm(
        &self,
        _x: &DynTensor,
        _num_groups: usize,
        _weight: &DynTensor,
        _bias: &DynTensor,
        _eps: f64,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused RmsNorm on GPU: `x / rms(x) * weight`.
    fn rms_norm(
        &self,
        _x: &DynTensor,
        _weight: &DynTensor,
        _eps: f64,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused InstanceNorm on GPU: `(x - mean) / sqrt(var + eps)`.
    ///
    /// Parameter-free normalization — no weight or bias. Normalizes each
    /// channel independently per sample over spatial dimensions.
    ///
    /// - `x`: input tensor `[B, C, *spatial]` (rank ≥ 3)
    /// - `eps`: numerical stability epsilon
    ///
    /// Returns `None` to fall back to the decomposed path.
    fn instance_norm(&self, _x: &DynTensor, _eps: f64) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused RoPE (Rotary Position Embedding) on GPU.
    ///
    /// Applies the rotation `y[2i] = x[2i]*cos - x[2i+1]*sin` and
    /// `y[2i+1] = x[2i]*sin + x[2i+1]*cos` in a single dispatch, replacing
    /// the 11-dispatch decomposed path (4 narrow + 4 broadcast_mul +
    /// 1 broadcast_sub + 1 broadcast_add + 1 cat).
    ///
    /// - `x`: input tensor `[..., seq_len, head_dim]` (head_dim must be even)
    /// - `cos`: precomputed cosines `[seq_len, head_dim/2]`
    /// - `sin`: precomputed sines `[seq_len, head_dim/2]`
    ///
    /// Returns `None` to fall back to the decomposed path.
    fn rope(
        &self,
        _x: &DynTensor,
        _cos: &DynTensor,
        _sin: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused LSTM cell — single dispatch for all gate operations.
    ///
    /// Computes one LSTM timestep in a single GPU dispatch graph:
    /// ```text
    /// gates = input @ w_ih^T + hidden @ w_hh^T + bias
    /// i, f, g, o = split(gates, 4)
    /// c_new = sigmoid(f) * cell + sigmoid(i) * tanh(g)
    /// h_new = sigmoid(o) * tanh(c_new)
    /// ```
    ///
    /// Returns `(h_new, c_new)` each `[batch, hidden_size]`.
    ///
    /// Returns `None` to fall back to the decomposed path.
    fn lstm_cell(
        &self,
        _input: &DynTensor,
        _hidden: &DynTensor,
        _cell: &DynTensor,
        _w_ih: &DynTensor,
        _w_hh: &DynTensor,
        _bias: Option<&DynTensor>,
        _hidden_size: usize,
    ) -> Option<Result<(DynTensor, DynTensor)>> {
        None
    }

    /// Fused LSTM sequence — single GPU dispatch for full sequence.
    ///
    /// Processes the entire `[seq_len, batch, input_size]` input in one Metal
    /// kernel launch, eliminating per-timestep `commit_and_wait()` barriers.
    ///
    /// Returns `(output, h_n, c_n)`:
    /// - `output`: `[seq_len, batch, hidden_size]`
    /// - `h_n`: `[batch, hidden_size]` (final hidden state)
    /// - `c_n`: `[batch, hidden_size]` (final cell state)
    ///
    /// Returns `None` to fall back to the per-timestep loop.
    fn lstm_sequence(
        &self,
        _input: &DynTensor,
        _w_ih: &DynTensor,
        _w_hh: &DynTensor,
        _bias: Option<&DynTensor>,
        _h0: &DynTensor,
        _c0: &DynTensor,
        _hidden_size: usize,
    ) -> Option<Result<(DynTensor, DynTensor, DynTensor)>> {
        None
    }

    /// Fused per-channel Snake activation on GPU: `x + (1/alpha) * sin²(alpha * x)`.
    ///
    /// Single dispatch graph replaces 6 separate GPU dispatches in the
    /// decomposed path (clamp + broadcast_mul + sin + sqr + recip + mul + add).
    /// Used by Kokoro TTS ISTFTNet decoder (36 invocations per forward pass).
    ///
    /// - `x`: input tensor (any rank)
    /// - `alpha`: per-channel parameter (broadcasts left-aligned over `x`)
    ///
    /// Returns `None` to fall back to the decomposed path.
    fn snake_tensor(&self, _x: &DynTensor, _alpha: &DynTensor) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused AdaIN+Snake on GPU: InstanceNorm → affine(gamma, beta) → Snake(alpha).
    ///
    /// Combines the full AdaIN normalization with Snake activation into a single
    /// dispatch graph, eliminating intermediate buffers between the two operations.
    /// Used by Kokoro TTS ResBlocks (36 invocations per forward pass).
    ///
    /// - `x`: input tensor `[B, C, T]` (rank 3)
    /// - `gamma`: style-projected scale `[B, C, 1]`
    /// - `beta`: style-projected bias `[B, C, 1]`
    /// - `alpha`: per-channel Snake parameter `[1, C, 1]` (must match input rank)
    /// - `eps`: numerical stability epsilon for InstanceNorm
    ///
    /// Returns `None` to fall back to the decomposed path.
    fn adain_snake(
        &self,
        _x: &DynTensor,
        _gamma: &DynTensor,
        _beta: &DynTensor,
        _alpha: &DynTensor,
        _eps: f64,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused AdaIN+LeakyRelu on GPU: InstanceNorm → affine(gamma, beta) → LeakyRelu(slope).
    ///
    /// Combines the full AdaIN normalization with LeakyRelu activation into a single
    /// dispatch, eliminating intermediate buffers. Used by Kokoro F0EnergyPredictor
    /// (12 invocations per forward pass).
    ///
    /// - `x`: input tensor `[B, C, T]` (rank 3)
    /// - `gamma`: style-projected scale `[B, C, 1]`
    /// - `beta`: style-projected bias `[B, C, 1]`
    /// - `eps`: numerical stability epsilon for InstanceNorm
    /// - `slope`: LeakyRelu negative slope (e.g. 0.2)
    ///
    /// Returns `None` to fall back to the decomposed path.
    fn adain_leaky_relu(
        &self,
        _x: &DynTensor,
        _gamma: &DynTensor,
        _beta: &DynTensor,
        _eps: f64,
        _slope: f64,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused clamp on GPU: `max(lo, min(hi, x))` in a single dispatch.
    ///
    /// Replaces the 8-encoding relu decomposition: `clamp_min(3) + clamp_max(5)`.
    /// Returns `None` to fall back to the decomposed path.
    fn clamp(&self, _x: &DynTensor, _min: f64, _max: f64) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused clamp_min on GPU: `max(lo, x)` in a single dispatch.
    ///
    /// Replaces the 3-encoding relu decomposition: `sub_scalar + relu + add_scalar`.
    /// Returns `None` to fall back to the decomposed path.
    fn clamp_min(&self, _x: &DynTensor, _min: f64) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused clamp_max on GPU: `min(hi, x)` in a single dispatch.
    ///
    /// Replaces the 5-encoding relu decomposition: `neg + add_scalar + relu + neg + add_scalar`.
    /// Returns `None` to fall back to the decomposed path.
    fn clamp_max(&self, _x: &DynTensor, _max: f64) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused scaled dot-product attention on GPU (Flash Attention).
    ///
    /// Fuses the entire SDPA pipeline (`Q@K^T*scale [+mask] → softmax → @V`)
    /// into a single GPU dispatch, avoiding O(S²) intermediate allocation.
    ///
    /// - `q`: `[B, H_q, S_q, head_dim]`
    /// - `k`: `[B, H_kv, S_kv, head_dim]` (H_kv ≤ H_q for GQA)
    /// - `v`: `[B, H_kv, S_kv, head_dim]`
    /// - `mask`: optional additive mask `[*, *, S_q, S_kv]`
    /// - `scale`: typically `1/sqrt(head_dim)`
    ///
    /// Returns `None` to fall back to the decomposed path.
    ///
    /// Issue: #2434
    fn sdpa(
        &self,
        _q: &DynTensor,
        _k: &DynTensor,
        _v: &DynTensor,
        _mask: Option<&DynTensor>,
        _scale: f64,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused scaled dot-product attention with causal masking on GPU.
    ///
    /// Same as [`sdpa`](Self::sdpa) but with built-in causal masking (upper
    /// triangle masked to `-inf`). Uses block-level tile skipping for ~50%
    /// compute savings on autoregressive attention.
    ///
    /// - `q`: `[B, H_q, S, head_dim]`
    /// - `k`: `[B, H_kv, S, head_dim]` (S_q must equal S_kv)
    /// - `v`: `[B, H_kv, S, head_dim]`
    /// - `scale`: typically `1/sqrt(head_dim)`
    ///
    /// Returns `None` to fall back to the decomposed path with explicit mask.
    ///
    /// Issue: #2434
    fn sdpa_causal(
        &self,
        _q: &DynTensor,
        _k: &DynTensor,
        _v: &DynTensor,
        _scale: f64,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Bilinear interpolation resize to absolute target dimensions on GPU.
    ///
    /// Input: `[N, C, H_in, W_in]` or `[C, H_in, W_in]` (rank 3 or 4).
    /// Output: same batch/channel dims with `[target_h, target_w]`.
    ///
    /// Coordinate mapping: `src = (dst + 0.5) * (in_size / out_size) - 0.5`,
    /// clamped to `[0, in_size - 1]`. Matches PyTorch `F.interpolate(mode='bilinear',
    /// align_corners=False)`.
    ///
    /// Returns `None` to fall back to the CPU round-trip path.
    ///
    /// Issue: #3535
    fn resize_bilinear(
        &self,
        _x: &DynTensor,
        _target_h: usize,
        _target_w: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Scalar binary op: `x op scalar` in a single GPU dispatch.
    ///
    /// Eliminates the `scalar_like()` CPU alloc + GPU transfer + broadcast
    /// overhead for `add_scalar`, `mul_scalar`, `sub_scalar`, `div_scalar`.
    /// The scalar value is baked into the MSL kernel as an inline constant
    /// (same pattern as `clamp`/`clamp_min`/`clamp_max`).
    ///
    /// Returns `None` to fall back to `scalar_like()` + `broadcast_binary_op`.
    ///
    /// Issue: #3230 (Gap 2)
    fn scalar_binary_op(
        &self,
        _op: super::BinaryOp,
        _x: &DynTensor,
        _scalar: f64,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// 2-D max pooling on GPU.
    ///
    /// Input: `[B, C, H, W]`, Output: `[B, C, out_H, out_W]`
    /// Sliding window max with stride/padding.
    ///
    /// Returns `None` to fall back to the CPU round-trip path.
    ///
    /// Issue: #4323
    fn max_pool2d(
        &self,
        _x: &DynTensor,
        _kernel_size: usize,
        _stride: usize,
        _padding: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// 2-D average pooling on GPU (count_include_pad=false).
    ///
    /// Input: `[B, C, H, W]`, Output: `[B, C, out_H, out_W]`
    /// Sliding window mean with stride/padding.
    ///
    /// Returns `None` to fall back to the CPU round-trip path.
    ///
    /// Issue: #4323
    fn avg_pool2d(
        &self,
        _x: &DynTensor,
        _kernel_size: usize,
        _stride: usize,
        _padding: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Adaptive 2-D average pooling on GPU.
    ///
    /// Input: `[B, C, H, W]`, Output: `[B, C, out_h, out_w]`
    /// Automatically computes window sizes from input/output dimensions.
    ///
    /// Returns `None` to fall back to the CPU round-trip path.
    ///
    /// Issue: #4323
    fn adaptive_avg_pool2d(
        &self,
        _x: &DynTensor,
        _out_h: usize,
        _out_w: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused BatchNorm on GPU: `(x - running_mean) / sqrt(running_var + eps) * weight + bias`.
    ///
    /// Uses precomputed running statistics (inference mode). No reduction needed --
    /// purely per-element with per-channel parameters. Input: `[N, C, *spatial]`,
    /// running_mean/running_var: `[C]`, weight/bias: `[C]` (optional).
    ///
    /// Single dispatch replaces ~6 separate GPU dispatches in the decomposed path
    /// (reshape + broadcast_sub + add_scalar + sqrt + recip + broadcast_mul + broadcast_add).
    ///
    /// Returns `None` to fall back to the decomposed CPU/GPU path.
    ///
    /// Issue: #4324
    fn batch_norm(
        &self,
        _x: &DynTensor,
        _running_mean: &DynTensor,
        _running_var: &DynTensor,
        _weight: Option<&DynTensor>,
        _bias: Option<&DynTensor>,
        _eps: f64,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Fused MoE scatter-gather dispatch on GPU.
    ///
    /// Performs the entire MoE token-to-expert routing, expert execution, and
    /// weighted combination in a single GPU dispatch, eliminating per-expert
    /// loop overhead and CPU readback of routing indices.
    ///
    /// - `hidden`: `[N, D]` flattened token embeddings (F32)
    /// - `indices`: `[N, K]` expert indices from top-k routing (U32)
    /// - `weights`: `[N, K]` routing weights (F32, normalized)
    /// - `expert_gate_weights`: list of `[intermediate, D]` gate_proj weights per expert
    /// - `expert_up_weights`: list of `[intermediate, D]` up_proj weights per expert
    /// - `expert_down_weights`: list of `[D, intermediate]` down_proj weights per expert
    /// - `num_experts`: total number of experts
    ///
    /// Returns `[N, D]` with the weighted combination of expert outputs.
    ///
    /// Returns `None` to fall back to the per-expert loop dispatch.
    ///
    /// Issue: #3547
    fn moe_scatter_gather(
        &self,
        _hidden: &DynTensor,
        _indices: &DynTensor,
        _weights: &DynTensor,
        _expert_gate_weights: &[DynTensor],
        _expert_up_weights: &[DynTensor],
        _expert_down_weights: &[DynTensor],
        _num_experts: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }
}
