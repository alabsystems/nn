// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL kernel wrapper emission (direct-binding and packed-buffer variants).
//!
//! Extracted from `codegen_msl.rs` for the 500-line limit.

use crate::ir::KernelDef;

use super::{msl_type, wrapper_out_buffer_index, wrapper_total_buffer_index};

/// `scalar_fn_name` is the name of the scalar helper function to call from the
/// wrapper. In the combined `emit_msl` output this is the prefixed name
/// (`_nn_{name}`) to avoid MSL built-in collisions; the kernel entry point
/// itself remains `{kernel.name}_kernel`.
pub(super) fn emit_kernel_wrapper(kernel: &KernelDef, scalar_fn_name: &str) -> String {
    let ret = msl_type(kernel.return_type);

    let mut buffer_params = Vec::new();
    for (i, param) in kernel.params.iter().enumerate() {
        buffer_params.push(format!(
            "    device const {ty}* {name} [[buffer({i})]]",
            ty = msl_type(param.ty),
            name = param.name,
        ));
    }
    let out_idx = wrapper_out_buffer_index(kernel.params.len());
    buffer_params.push(format!("    device {ret}* out [[buffer({out_idx})]]"));
    buffer_params.push(format!(
        "    constant uint& total [[buffer({})]]",
        wrapper_total_buffer_index(kernel.params.len())
    ));
    buffer_params.push("    uint tid [[thread_position_in_grid]]".to_string());

    let call_args: Vec<String> = kernel
        .params
        .iter()
        .map(|p| format!("{}[tid]", p.name))
        .collect();

    format!(
        "[[kernel]] void {name}_kernel(\n{buffer_params}\n) {{\n    if (tid >= total) return;\n    out[tid] = {scalar_fn_name}({call_args});\n}}",
        name = kernel.name,
        buffer_params = buffer_params.join(",\n"),
        call_args = call_args.join(", "),
    )
}

/// Emit a packed `[[kernel]]` wrapper that reads all parameters from a single
/// contiguous buffer via an offsets array.
///
/// The kernel function is named `{kernel.name}_packed_kernel` (distinct from
/// the direct-binding `{kernel.name}_kernel`).
///
/// Part of #1649.
pub(super) fn emit_packed_kernel_wrapper(kernel: &KernelDef, scalar_fn_name: &str) -> String {
    let ret = msl_type(kernel.return_type);
    let n = kernel.params.len();

    // Build call arguments: read each param from packed_inputs[offsets[i] + tid].
    let call_args: Vec<String> = (0..n)
        .map(|i| format!("packed_inputs[offsets[{i}] + tid]"))
        .collect();

    format!(
        "[[kernel]] void {name}_packed_kernel(\n\
         \x20   device const {ret}* packed_inputs [[buffer(0)]],\n\
         \x20   constant uint* offsets [[buffer(1)]],\n\
         \x20   device {ret}* out [[buffer(2)]],\n\
         \x20   constant uint& total [[buffer(3)]],\n\
         \x20   uint tid [[thread_position_in_grid]]\n\
         ) {{\n\
         \x20   if (tid >= total) return;\n\
         \x20   out[tid] = {scalar_fn_name}({call_args});\n\
         }}",
        name = kernel.name,
        ret = ret,
        call_args = call_args.join(", "),
    )
}
