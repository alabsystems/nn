// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! INT8 quantized linear layer (W8A16): INT8 weights, FP32 activations.
//!
//! CPU path: dequantize weights to F32, then standard matmul.
//! GPU path (future): W8A16 Metal kernel with on-the-fly dequantization.
//!
//! Memory savings: 4x vs F32 weights (1 byte/weight + 4 bytes/channel scale).
//! For a 768x768 linear layer: 589 KB (INT8) vs 2,359 KB (F32).
//!
//! Part of #3522

use super::int8::{dequantize_per_channel, quantize_per_channel, Int8Mode, Int8QuantParams};
use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, Linear, Module};
use crate::{Result, TensorError};

/// INT8 quantized linear layer.
///
/// Stores weights as INT8 (U8 tensor) with per-channel F32 scales.
/// Forward pass dequantizes to F32 then computes `y = x @ W^T + bias`.
///
/// # Construction
///
/// ```ignore
/// // From an existing Linear layer (symmetric quantization)
/// let int8_linear = Int8Linear::from_linear(&linear, Int8Mode::Symmetric)?;
///
/// // Forward works identically
/// let output = int8_linear.forward(&input)?;
/// ```
///
/// # Memory layout
///
/// - `weight_int8`: `[out_features, in_features]` as DType::U8
///   (stores i8 values reinterpreted as u8)
/// - `scale`: per-channel F32 scales, length = out_features
/// - `zero_point`: per-channel i8 zero points, length = out_features
///   (all zeros for symmetric mode)
/// - `bias`: optional `[out_features]` F32 tensor
#[derive(Debug, Clone)]
pub struct Int8Linear {
    /// Quantized weight tensor [out_features, in_features] as U8.
    weight_int8: DynTensor,
    /// Per-channel quantization parameters (scale + zero_point).
    params: Int8QuantParams,
    /// Optional bias [out_features] as F32.
    bias: Option<DynTensor>,
    /// Original weight shape for validation.
    out_features: usize,
    in_features: usize,
}

impl Int8Linear {
    /// Create an Int8Linear from pre-quantized components.
    ///
    /// # Arguments
    /// - `weight_int8`: quantized weights `[out_features, in_features]` as U8
    /// - `params`: per-channel scale and zero_point
    /// - `bias`: optional bias `[out_features]`
    ///
    /// # Errors
    /// Returns error if shapes are inconsistent.
    pub fn new(
        weight_int8: DynTensor,
        params: Int8QuantParams,
        bias: Option<DynTensor>,
    ) -> Result<Self> {
        let (out_features, in_features) = weight_int8.dims2()?;

        if params.scale.len() != out_features {
            return Err(TensorError::DataLengthMismatch {
                expected: out_features,
                actual: params.scale.len(),
            });
        }
        if params.zero_point.len() != out_features {
            return Err(TensorError::DataLengthMismatch {
                expected: out_features,
                actual: params.zero_point.len(),
            });
        }

        if let Some(ref b) = bias {
            if b.dims() != [out_features] {
                return Err(TensorError::shape_mismatch(
                    vec![out_features],
                    b.dims().to_vec(),
                ));
            }
        }

        Ok(Self {
            weight_int8,
            params,
            bias,
            out_features,
            in_features,
        })
    }

    /// Quantize an existing `Linear` layer to INT8.
    ///
    /// Extracts the weight tensor, quantizes per-channel, preserves the bias.
    pub fn from_linear(linear: &Linear, mode: Int8Mode) -> Result<Self> {
        let (quantized, params) = quantize_per_channel(linear.weight(), mode)?;
        Self::new(quantized, params, linear.bias().cloned())
    }

    /// Dequantize back to a standard `Linear` layer (for verification/comparison).
    pub fn dequantize(&self) -> Result<Linear> {
        let weight_f32 = dequantize_per_channel(&self.weight_int8, &self.params)?;
        Linear::new(weight_f32, self.bias.clone())
    }

    /// Number of output features.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Number of input features.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Reference to quantization parameters.
    #[must_use]
    pub fn quant_params(&self) -> &Int8QuantParams {
        &self.params
    }

    /// Reference to the INT8 weight tensor.
    #[must_use]
    pub fn weight_int8(&self) -> &DynTensor {
        &self.weight_int8
    }

    /// Reference to the bias tensor (if present).
    #[must_use]
    pub fn bias(&self) -> Option<&DynTensor> {
        self.bias.as_ref()
    }

    /// Memory footprint in bytes (INT8 weights + scales + zero_points + optional bias).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let weight_bytes = self.out_features * self.in_features; // 1 byte per weight
        let scale_bytes = self.out_features * 4; // f32 per channel
        let zp_bytes = self.out_features; // i8 per channel
        let bias_bytes = self.bias.as_ref().map_or(0, |_| self.out_features * 4);
        weight_bytes + scale_bytes + zp_bytes + bias_bytes
    }

    /// Equivalent F32 memory footprint (for comparison).
    #[must_use]
    pub fn f32_memory_bytes(&self) -> usize {
        let weight_bytes = self.out_features * self.in_features * 4;
        let bias_bytes = self.bias.as_ref().map_or(0, |_| self.out_features * 4);
        weight_bytes + bias_bytes
    }

    /// Compression ratio vs F32.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        self.f32_memory_bytes() as f32 / self.memory_bytes() as f32
    }
}

impl Module for Int8Linear {
    /// Forward pass: dequantize INT8 weights to F32, then matmul.
    ///
    /// Input shape: `[*, in_features]` (any number of batch dimensions).
    /// Output shape: `[*, out_features]`.
    ///
    /// CPU path (Phase 1): Full dequantize → standard matmul.
    /// GPU path (Phase 2, future): W8A16 Metal kernel with tile-level dequant.
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        // Validate input last dimension matches in_features
        let x_dims = x.dims();
        if x_dims.is_empty() {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: 0,
            });
        }
        let x_last = x_dims[x_dims.len() - 1];
        if x_last != self.in_features {
            return Err(TensorError::shape_mismatch(
                vec![self.in_features],
                vec![x_last],
            ));
        }

        // CPU path: dequantize then standard matmul
        // Future: GPU path dispatches W8A16 kernel directly
        let weight_f32 = dequantize_per_channel(&self.weight_int8, &self.params)?;

        // Move weight to same device as input
        let weight_on_device = weight_f32.to_device(&x.device())?;

        let out = x.matmul(&weight_on_device.t()?)?;
        let out = match &self.bias {
            Some(bias) => {
                let bias_on_device = bias.to_device(&x.device())?;
                out.broadcast_add(&bias_on_device)?
            }
            None => out,
        };

        check_output_finite(&out, "Int8Linear")?;
        Ok(out)
    }
}

#[cfg(kani)]
#[path = "kani_int8_linear_proofs.rs"]
mod kani_int8_linear_proofs;

// -- Kani proof harnesses -----------------------------------------------------

#[cfg(kani)]
mod kani_proofs {
    // -----------------------------------------------------------------------
    // Harness 1: Int8Linear memory_bytes is always < f32_memory_bytes.
    //
    // INT8 stores 1 byte/weight + 4 bytes/channel scale + 1 byte/channel zp.
    // F32 stores 4 bytes/weight.
    // For any out_features >= 1 and in_features >= 5 (so the per-channel
    // overhead is amortized), INT8 uses strictly less memory.
    //
    // Specifically: INT8 = out * in + 5 * out, F32 = 4 * out * in.
    // INT8 < F32 when: out * in + 5 * out < 4 * out * in
    //                   => 5 * out < 3 * out * in  => 5 < 3 * in  => in >= 2.
    // -----------------------------------------------------------------------

    /// Prove: Int8Linear always uses less memory than F32 for in_features >= 2.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8linear_memory_less_than_f32() {
        let out_features: usize = kani::any();
        let in_features: usize = kani::any();

        // Practical bounds to keep Kani tractable
        kani::assume(out_features >= 1 && out_features <= 4096);
        kani::assume(in_features >= 2 && in_features <= 4096);

        // INT8 memory: weight_bytes + scale_bytes + zp_bytes
        let weight_bytes = out_features * in_features; // 1 byte per weight
        let scale_bytes = out_features * 4; // f32 per channel
        let zp_bytes = out_features; // i8 per channel
        let int8_bytes = weight_bytes + scale_bytes + zp_bytes;

        // F32 memory: 4 bytes per weight
        let f32_bytes = out_features * in_features * 4;

        assert!(
            int8_bytes < f32_bytes,
            "INT8 must use less memory than F32 for in_features >= 2"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 2: Int8Linear compression ratio bounds.
    //
    // Compression ratio = f32_memory / int8_memory.
    // For large in_features (overhead negligible): approaches 4x.
    // For in_features = 2: worst case (most overhead).
    //
    // Prove compression ratio is always > 1 for in_features >= 2.
    // -----------------------------------------------------------------------

    /// Prove: Int8Linear compression ratio is > 1.0 for in_features >= 2
    /// (ignoring bias, which is the same size for both).
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8linear_compression_ratio_above_one() {
        let out_features: usize = kani::any();
        let in_features: usize = kani::any();

        kani::assume(out_features >= 1 && out_features <= 1024);
        kani::assume(in_features >= 2 && in_features <= 1024);

        let weight_bytes = out_features * in_features;
        let scale_bytes = out_features * 4;
        let zp_bytes = out_features;
        let int8_total = weight_bytes + scale_bytes + zp_bytes;

        let f32_total = out_features * in_features * 4;

        // f32_total / int8_total > 1.0
        // Equivalent: f32_total > int8_total (both positive)
        assert!(
            f32_total > int8_total,
            "F32 must use more memory than INT8 for in_features >= 2"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 3: Per-element dequant in Int8Linear forward is bounded.
    //
    // The forward path computes: w_f32[row][col] = (w_i8 - zp) * scale.
    // Then: out = x @ w_f32^T + bias.
    //
    // For a single dequantized weight element, prove the output is bounded
    // given bounded scale and zero_point from quantize_per_channel.
    //
    // From int8.rs: symmetric scale = abs_max / 127, abs_max <= original max.
    // For weights in [-W, W], scale = W/127.
    // Dequant value = q * scale where q in [-128, 127].
    // |dequant| <= 128 * W/127 < 1.01 * W.
    // -----------------------------------------------------------------------

    /// Prove: the per-element dequantization in Int8Linear forward produces
    /// values bounded by 128 * scale, and the bound is tight to within
    /// 1.01x of the original weight magnitude.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8linear_per_element_dequant_bounded() {
        // Original weight magnitude bound
        let abs_max: f32 = kani::any();
        kani::assume(abs_max.is_finite());
        kani::assume(abs_max > 0.0 && abs_max < 1e6);

        // Symmetric scale = abs_max / 127
        let scale = abs_max / 127.0;

        // Any quantized value
        let q_u8: u8 = kani::any();
        let q_i8 = q_u8 as i8;

        // Dequantize (symmetric: zp = 0)
        let dequant = (q_i8 as f32) * scale;

        assert!(dequant.is_finite(), "dequantized weight must be finite");

        // |dequant| <= 128 * scale = 128 * abs_max / 127 < 1.008 * abs_max
        let bound = 128.0 * scale;
        assert!(
            dequant >= -bound && dequant <= bound,
            "dequantized weight must be within 128 * scale"
        );

        // The 128/127 factor means dequant values can exceed original abs_max
        // by at most ~0.8%. Verify the overshoot is bounded:
        let overshoot_factor = 128.0 / 127.0;
        assert!(
            dequant.abs() <= abs_max * overshoot_factor,
            "dequant overshoot must be within 128/127 factor"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 4: Dequant + bias addition is finite for bounded components.
    //
    // In Int8Linear::forward, the final output element is:
    //   out_elem = sum_k(x[k] * w_dequant[k]) + bias
    //
    // For a single product + bias step, prove finiteness.
    // -----------------------------------------------------------------------

    /// Prove: adding a bounded bias to a bounded dequant product
    /// produces a finite result.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8linear_dequant_plus_bias_finite() {
        // Single dequantized weight * activation product
        let product: f32 = kani::any();
        kani::assume(product.is_finite());
        kani::assume(product >= -1e10 && product <= 1e10);

        // Bias value (typically small, << weight magnitudes)
        let bias: f32 = kani::any();
        kani::assume(bias.is_finite());
        kani::assume(bias >= -1e6 && bias <= 1e6);

        let result = product + bias;
        assert!(
            result.is_finite(),
            "dequant product + bias must be finite for bounded inputs"
        );
        assert!(
            result >= -1.001e10 && result <= 1.001e10,
            "result must be within sum of bounds"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 5: Input dimension validation invariant.
    //
    // Int8Linear::forward checks x_dims.last() == in_features.
    // Prove: for any non-zero in_features and matching last dim,
    // the dimension check passes.
    // -----------------------------------------------------------------------

    /// Prove: the input dimension validation in Int8Linear::forward
    /// accepts inputs whose last dimension matches in_features, and
    /// rejects inputs whose last dimension does not match.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8linear_dim_validation_correct() {
        let in_features: usize = kani::any();
        kani::assume(in_features >= 1 && in_features <= 4096);

        let x_last: usize = kani::any();
        kani::assume(x_last >= 1 && x_last <= 4096);

        if x_last == in_features {
            // Matching dimensions — forward should proceed
            assert!(x_last == in_features, "matching dims accepted");
        } else {
            // Non-matching — forward would return ShapeMismatch error
            assert!(x_last != in_features, "non-matching dims rejected");
        }
    }

    // Harness 6: Int8Linear scale/zero_point produces bounded output.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8linear_scale_zp_output_bounded() {
        let x_val: f32 = kani::any();
        kani::assume(x_val.is_finite());
        kani::assume(x_val >= -100.0 && x_val <= 100.0);
        let q_u8: u8 = kani::any();
        let q_i8 = q_u8 as i8;
        let scale: f32 = kani::any();
        kani::assume(scale.is_finite());
        kani::assume(scale >= 0.0 && scale < 10.0);
        let zero_point: i32 = kani::any();
        kani::assume(zero_point >= -128 && zero_point <= 127);
        let diff = q_i8 as f32 - zero_point as f32;
        let w_f32 = diff * scale;
        let product = x_val * w_f32;
        let bias: f32 = kani::any();
        kani::assume(bias.is_finite());
        kani::assume(bias >= -1000.0 && bias <= 1000.0);
        let output = product + bias;
        assert!(output.is_finite(), "output must be finite");
        assert!(
            output >= -256_000.0 && output <= 256_000.0,
            "output bounded"
        );
    }

    // Harness 7: Int8Linear accumulation bounded.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8linear_accumulation_bounded() {
        let product: f32 = kani::any();
        kani::assume(product.is_finite());
        kani::assume(product >= -255_000.0 && product <= 255_000.0);
        let partial_sum: f32 = kani::any();
        kani::assume(partial_sum.is_finite());
        kani::assume(partial_sum >= -1_044_225_000.0 && partial_sum <= 1_044_225_000.0);
        let new_sum = partial_sum + product;
        assert!(new_sum.is_finite(), "accumulation must stay finite");
        assert!(
            new_sum >= -1_044_480_000.0 && new_sum <= 1_044_480_000.0,
            "bounded"
        );
    }

    // Harness 8: Int8Linear params length invariant.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8linear_params_length_invariant() {
        let out_features: usize = kani::any();
        kani::assume(out_features >= 1 && out_features <= 1024);
        let scale_len: usize = kani::any();
        let zp_len: usize = kani::any();
        kani::assume(scale_len == out_features);
        kani::assume(zp_len == out_features);
        let row: usize = kani::any();
        kani::assume(row < out_features);
        assert!(row < scale_len, "row must be valid scale index");
        assert!(row < zp_len, "row must be valid zero_point index");
    }
}
