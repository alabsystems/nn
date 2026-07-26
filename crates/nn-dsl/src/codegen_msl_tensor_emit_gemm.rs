// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL codegen for fused GEMM + activation kernels.
//!
//! Extracted from `codegen_msl_tensor_emit_complex.rs` to keep files under
//! 450 lines. Part of #2256 (Linear→Activation GEMM epilogue fusion).

use crate::codegen_msl;
use crate::codegen_msl_structural;
use crate::codegen_msl_tensor::TensorMSLCodegenError;
use crate::ir::ScalarType;
use crate::trace_compile::GemmActivation;

/// Emit MSL source for a linear (fully-connected) layer kernel with optional
/// fused activation.
///
/// When `activation` is `Some`, the activation function is applied to the
/// f32 accumulator *before* casting to storage type. This is more precise
/// than the 2-dispatch path (which casts to storage, reads back, then
/// activates). Part of #2256 (GEMM fusion D3).
///
/// When `include_prelude` is true, the output includes `#include <metal_stdlib>`
/// / `using namespace metal;` so it can be compiled standalone by
/// `KernelPipeline::from_msl`. When false, the caller is expected to provide
/// the prelude (e.g., `emit_tensor_msl_with_plan`).
pub fn emit_linear_activation_kernel(
    name: &str,
    dtype: ScalarType,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    activation: Option<&GemmActivation>,
    include_prelude: bool,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let needs_cast = t != acc;

    let (bias_buf, out_buf, total_buf) = if has_bias {
        ("2", "3", "4")
    } else {
        ("", "2", "3")
    };

    let mut params = format!(
        "    device const {t}* input  [[buffer(0)]],\n\
         \x20   device const {t}* weight [[buffer(1)]],\n"
    );
    if has_bias {
        params.push_str(&format!(
            "    device const {t}* bias   [[buffer({bias_buf})]],\n"
        ));
    }
    params.push_str(&format!(
        "    device {t}* output         [[buffer({out_buf})]],\n\
         \x20   constant uint& total      [[buffer({total_buf})]],\n\
         \x20   uint tid [[thread_position_in_grid]]"
    ));

    let in_feat = codegen_msl_structural::safe_msl_uint(in_features)?;
    let out_feat = codegen_msl_structural::safe_msl_uint(out_features)?;

    let bias_line = if has_bias {
        if needs_cast {
            format!("    sum += {acc}(bias[col]);\n")
        } else {
            "    sum += bias[col];\n".to_string()
        }
    } else {
        String::new()
    };

    // Activation applied to accumulator before casting to storage type.
    let activation_line = match activation {
        Some(act) => gemm_activation_msl_line(act, acc),
        None => String::new(),
    };

    let store_expr = if needs_cast {
        format!("{t}(sum)")
    } else {
        "sum".to_string()
    };

    let load_input = if needs_cast {
        format!("{acc}(input[row * IN_FEATURES + k])")
    } else {
        "input[row * IN_FEATURES + k]".to_string()
    };
    let load_weight = if needs_cast {
        format!("{acc}(weight[col * IN_FEATURES + k])")
    } else {
        "weight[col * IN_FEATURES + k]".to_string()
    };

    let prelude = if include_prelude {
        "#include <metal_stdlib>\nusing namespace metal;\n\n"
    } else {
        ""
    };

    Ok(format!(
        r#"{prelude}[[kernel]] void {name}(
{params}
) {{
    if (tid >= total) return;
    const uint IN_FEATURES = {in_feat};
    const uint OUT_FEATURES = {out_feat};

    uint row = tid / OUT_FEATURES;
    uint col = tid % OUT_FEATURES;

    {acc} sum = 0;
    for (uint k = 0; k < IN_FEATURES; k++) {{
        sum += {load_input} * {load_weight};
    }}
{bias_line}{activation_line}    output[tid] = {store_expr};
}}"#,
    ))
}

/// Generate MSL code applying a [`GemmActivation`] to a named variable.
///
/// `var` is the MSL variable name to read and overwrite (e.g., `"sum"` for the
/// naive kernel, `"val"` for the simdgroup write-back).  `acc` is the
/// accumulator type string (e.g., `"float"`).  `indent` is the leading
/// whitespace for each emitted line.
///
/// Uses `metal::precise::exp`/`metal::precise::tanh` for GeluErf, Sigmoid,
/// Silu, and Tanh to avoid FTZ issues with denormals.  Part of #2256 D3+D4.
pub fn gemm_activation_msl_var(
    activation: &GemmActivation,
    acc: &str,
    var: &str,
    indent: &str,
) -> String {
    match activation {
        GemmActivation::Relu => {
            format!("{indent}{var} = max({var}, {acc}(0));\n")
        }
        GemmActivation::Gelu => {
            format!(
                "{indent}{var} = {acc}(0.5) * {var} * ({acc}(1) + metal::precise::tanh(\
                 {acc}(0.7978845608) * ({var} + {acc}(0.044715) * {var} * {var} * {var})));\n"
            )
        }
        GemmActivation::GeluErf => {
            format!(
                "{indent}{{ {acc} u = {var} * {acc}(0.7071067811865476);\n\
                 {indent}  {acc} ax = abs(u);\n\
                 {indent}  {acc} et = {acc}(1.0) / ({acc}(1.0) + {acc}(0.3275911) * ax);\n\
                 {indent}  {acc} poly = (((({acc}(1.0614054) * et + {acc}(-1.453152)) * et \
                 + {acc}(1.4214138)) * et + {acc}(-0.28449674)) * et + {acc}(0.2548296)) * et;\n\
                 {indent}  {acc} sign_u = (u >= {acc}(0.0)) ? {acc}(1.0) : {acc}(-1.0);\n\
                 {indent}  {acc} erf_val = sign_u * ({acc}(1.0) - poly * metal::precise::exp(-(u * u)));\n\
                 {indent}  {var} = {acc}(0.5) * {var} * ({acc}(1.0) + erf_val); }}\n"
            )
        }
        GemmActivation::Sigmoid => {
            format!("{indent}{var} = {acc}(1) / ({acc}(1) + metal::precise::exp(-{var}));\n")
        }
        GemmActivation::Silu => {
            format!("{indent}{var} = {var} / ({acc}(1) + metal::precise::exp(-{var}));\n")
        }
        GemmActivation::Tanh => {
            format!("{indent}{var} = metal::precise::tanh({var});\n")
        }
    }
}

/// Generate MSL applying activation to the `sum` accumulator (naive kernel).
fn gemm_activation_msl_line(activation: &GemmActivation, acc: &str) -> String {
    gemm_activation_msl_var(activation, acc, "sum", "    ")
}
