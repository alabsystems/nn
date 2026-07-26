// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL emission helpers for individual tensor op types.
//!
//! Extracted from `codegen_msl_tensor_emit.rs` to keep both files under the
//! 500-line limit.

use crate::codegen_msl;
use crate::codegen_msl_structural;
use crate::codegen_msl_tensor::{BinaryBroadcastInfo, BroadcastSide, TensorMSLCodegenError};
use crate::ir::ScalarType;

/// Emit MSL source for a binary addition kernel: `out[tid] = left[tid] + right[tid]`.
///
/// When `broadcast` is `Some`, one operand uses modular indexing instead of
/// flat `tid` access (fused Broadcast+BinaryAdd).
pub(super) fn emit_binary_add_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
    broadcast: Option<&BinaryBroadcastInfo>,
) -> Result<String, TensorMSLCodegenError> {
    if let Some(bcast) = broadcast {
        return emit_binary_broadcast_kernel(name, dtype, total_elements, bcast, "+");
    }
    let t = codegen_msl::msl_type(dtype);
    let n = codegen_msl_structural::safe_msl_uint(total_elements)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* left  [[buffer(0)]],
    device const {t}* right [[buffer(1)]],
    device {t}* output      [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
    output[tid] = left[tid] + right[tid];
}}
"#
    ))
}

/// Emit MSL source for a binary multiplication kernel: `out[tid] = left[tid] * right[tid]`.
///
/// When `broadcast` is `Some`, one operand uses modular indexing instead of
/// flat `tid` access (fused Broadcast+BinaryMul).
pub(super) fn emit_binary_mul_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
    broadcast: Option<&BinaryBroadcastInfo>,
) -> Result<String, TensorMSLCodegenError> {
    if let Some(bcast) = broadcast {
        return emit_binary_broadcast_kernel(name, dtype, total_elements, bcast, "*");
    }
    let t = codegen_msl::msl_type(dtype);
    let n = codegen_msl_structural::safe_msl_uint(total_elements)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* left  [[buffer(0)]],
    device const {t}* right [[buffer(1)]],
    device {t}* output      [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
    output[tid] = left[tid] * right[tid];
}}
"#
    ))
}

/// Emit MSL for a broadcast-aware binary op kernel.
///
/// One operand is flat (same size as output, uses `tid`). The other is smaller
/// and uses modular indexing (`in_idx`) to replicate across the output shape.
fn emit_binary_broadcast_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
    bcast: &BinaryBroadcastInfo,
    op: &str,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let n = codegen_msl_structural::safe_msl_uint(total_elements)?;
    let index_body = codegen_msl_structural::build_broadcast_index_body(
        &bcast.input_shape,
        &bcast.output_shape,
        bcast.alignment,
    )?;
    let (left_idx, right_idx) = match bcast.side {
        BroadcastSide::Left => ("in_idx", "tid"),
        BroadcastSide::Right => ("tid", "in_idx"),
    };
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* left  [[buffer(0)]],
    device const {t}* right [[buffer(1)]],
    device {t}* output      [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
{index_body}    output[tid] = left[{left_idx}] {op} right[{right_idx}];
}}
"#
    ))
}

/// Emit MSL source for a sigmoid activation kernel: `out[tid] = 1 / (1 + exp(-in[tid]))`.
///
/// F16/BF16 intermediates are promoted to float to avoid precision loss (#3250).
pub(super) fn emit_sigmoid_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let n = codegen_msl_structural::safe_msl_uint(total_elements)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
    {acc} x = {acc}(input[tid]);
    output[tid] = {t}({acc}(1.0) / ({acc}(1.0) + metal::precise::exp(-x)));
}}
"#
    ))
}

/// Emit MSL source for a GELU activation kernel (tanh approximation via exp).
///
/// Uses the exp-based form `0.5 * x * (2.0 - 2.0 / (exp(2*inner) + 1.0))`
/// which is mathematically equivalent to `0.5 * x * (1 + tanh(inner))` but
/// matches the scalar reference in `gelu.rs` exactly. This ensures NY
/// bounds verified against the scalar form also cover the GPU code path.
///
/// See #679: tanh vs exp forms differ by up to ~5e-7 on f32 (1 ULP).
pub(super) fn emit_gelu_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let n = codegen_msl_structural::safe_msl_uint(total_elements)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
    {acc} x = {acc}(input[tid]);
    {acc} inner = {acc}(0.7978845608028654) * (x + {acc}(0.044715) * x * x * x);
    {acc} e2 = metal::precise::exp({acc}(2.0) * inner);
    output[tid] = {t}({acc}(0.5) * x * ({acc}(2.0) - {acc}(2.0) / (e2 + {acc}(1.0))));
}}
"#
    ))
}

/// Emit MSL source for a GELU activation kernel (exact erf).
///
/// Uses the erf form `0.5 * x * (1 + erf(x / sqrt(2)))` which matches
/// `DynTensor::gelu_erf()`. More precise than the tanh approximation
/// (`emit_gelu_kernel`).
///
/// The erf is computed via the Abramowitz & Stegun formula 7.1.26 polynomial
/// approximation, since MSL does not provide a native `erf()` — neither
/// `metal::precise::erf` nor `metal::fast::erf` exist. This matches the
/// CPU reference implementation in `dyn_tensor/ops/math.rs` and the IR
/// graph in `dyn_tensor_metal_kernels_complex.rs:build_erf_graph()`.
pub(super) fn emit_gelu_erf_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let n = codegen_msl_structural::safe_msl_uint(total_elements)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
    {acc} x = {acc}(input[tid]);
    // erf(x / sqrt(2)) via Abramowitz & Stegun 7.1.26
    {acc} u = x * {acc}(0.7071067811865476);
    {acc} ax = abs(u);
    {acc} et = {acc}(1.0) / ({acc}(1.0) + {acc}(0.3275911) * ax);
    {acc} poly = (((({acc}(1.0614054) * et + {acc}(-1.453152)) * et + {acc}(1.4214138)) * et + {acc}(-0.28449674)) * et + {acc}(0.2548296)) * et;
    {acc} sign_u = (u >= {acc}(0.0)) ? {acc}(1.0) : {acc}(-1.0);
    {acc} erf_val = sign_u * ({acc}(1.0) - poly * metal::precise::exp(-(u * u)));
    output[tid] = {t}({acc}(0.5) * x * ({acc}(1.0) + erf_val));
}}
"#
    ))
}

/// Emit MSL source for a ReLU activation kernel: `out[tid] = max(in[tid], 0)`.
pub(super) fn emit_relu_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let n = codegen_msl_structural::safe_msl_uint(total_elements)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
    {t} x = input[tid];
    output[tid] = max(x, {t}(0.0));
}}
"#
    ))
}

/// Emit MSL source for a tanh activation kernel: `out[tid] = tanh(in[tid])`.
pub(super) fn emit_tanh_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let n = codegen_msl_structural::safe_msl_uint(total_elements)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
    output[tid] = metal::precise::tanh(input[tid]);
}}
"#
    ))
}

/// Emit MSL source for a LeakyReLU activation kernel:
/// `out[tid] = select(x, slope * x, x < 0)`.
///
/// The slope is baked as a compile-time MSL constant — no buffer binding needed.
/// Same 2-buffer signature (input + output) as Relu/Sigmoid/etc.
/// Part of #3230 (Gap 3).
pub(super) fn emit_leaky_relu_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
    negative_slope: f32,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let n = codegen_msl_structural::safe_msl_uint(total_elements)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
    {t} x = input[tid];
    output[tid] = select(x, {t}({negative_slope}) * x, x < {t}(0.0));
}}
"#
    ))
}

/// Emit MSL source for an ELU activation kernel:
/// `out[tid] = select(x, alpha * (exp(x) - 1), x < 0)`.
///
/// Alpha is baked as a compile-time MSL constant — no buffer binding needed.
/// Same 2-buffer signature as LeakyRelu. Part of #3230 (Gap 3).
pub(super) fn emit_elu_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
    alpha: f32,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let n = codegen_msl_structural::safe_msl_uint(total_elements)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
    {acc} x = {acc}(input[tid]);
    output[tid] = {t}(select(x, {acc}({alpha}) * (exp(x) - {acc}(1.0)), x < {acc}(0.0)));
}}
"#
    ))
}

/// Emit MSL source for an exp activation kernel: `out[tid] = exp(in[tid])`.
pub(super) fn emit_exp_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let n = codegen_msl_structural::safe_msl_uint(total_elements)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
    output[tid] = metal::precise::exp(input[tid]);
}}
"#
    ))
}

/// Emit MSL source for a softplus activation kernel: `out[tid] = log(1 + exp(in[tid]))`.
///
/// F16/BF16 intermediates are promoted to float to avoid precision loss (#3250).
pub(super) fn emit_softplus_kernel(
    name: &str,
    dtype: ScalarType,
    total_elements: usize,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let n = codegen_msl_structural::safe_msl_uint(total_elements)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
    {acc} x = {acc}(input[tid]);
    output[tid] = {t}(metal::precise::log({acc}(1.0) + metal::precise::exp(x)));
}}
"#
    ))
}

/// Emit MSL source for a zero-pad-1d kernel: copy input elements with left/right zero padding.
///
/// Layout: row-major `[channels, length]`. Each thread handles one output element.
/// Output `[c, t]` reads `input[c * in_length + (t - pad_left)]` if `t` falls in
/// the input range, otherwise writes `0.0`.
///
/// Buffer layout:
/// - `buffer(0)`: input data
/// - `buffer(1)`: output data
pub(super) fn emit_zero_pad_1d_kernel(
    name: &str,
    dtype: ScalarType,
    channels: usize,
    in_length: usize,
    pad_left: usize,
    out_length: usize,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let total = channels.checked_mul(out_length).ok_or_else(|| {
        TensorMSLCodegenError::ShapeProductOverflow {
            shape: vec![channels, out_length],
        }
    })?;
    let n = codegen_msl_structural::safe_msl_uint(total)?;
    let ol = codegen_msl_structural::safe_msl_uint(out_length)?;
    let il = codegen_msl_structural::safe_msl_uint(in_length)?;
    let pl = codegen_msl_structural::safe_msl_uint(pad_left)?;
    Ok(format!(
        r#"[[kernel]] void {name}(
    device const {t}* input [[buffer(0)]],
    device {t}* output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= {n}u) return;
    uint c = tid / {ol}u;
    uint ot = tid % {ol}u;
    if (ot >= {pl}u && ot < {pl}u + {il}u) {{
        output[tid] = input[c * {il}u + (ot - {pl}u)];
    }} else {{
        output[tid] = {t}(0.0);
    }}
}}
"#
    ))
}

// Complex MSL emission functions (linear, matmul, softmax, embedding) extracted
// to `codegen_msl_tensor_emit_complex.rs` to keep this file under 300 lines.
