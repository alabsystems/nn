// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code, unreachable_pub)]

//! Consolidated tensor IR tests: validation (core, axis_select, broadcast,
//! conv1d, instance_norm, stack, structural), codegen ops, emit, and GLU.

#[path = "tensor_ir/tensor_codegen_structural_ops.rs"]
mod tensor_codegen_structural_ops;
#[path = "tensor_ir/tensor_emit_glu.rs"]
mod tensor_emit_glu;
#[path = "tensor_ir/tensor_emit_sigmoid.rs"]
mod tensor_emit_sigmoid;
#[path = "tensor_ir/tensor_glu_validation.rs"]
mod tensor_glu_validation;
#[path = "tensor_ir/tensor_ir_validation_axis_select.rs"]
mod tensor_ir_validation_axis_select;
#[path = "tensor_ir/tensor_ir_validation_broadcast.rs"]
mod tensor_ir_validation_broadcast;
#[path = "tensor_ir/tensor_ir_validation_conv1d.rs"]
mod tensor_ir_validation_conv1d;
#[path = "tensor_ir/tensor_ir_validation_core.rs"]
mod tensor_ir_validation_core;
#[path = "tensor_ir/tensor_ir_validation_instance_norm.rs"]
mod tensor_ir_validation_instance_norm;
#[path = "tensor_ir/tensor_ir_validation_stack.rs"]
mod tensor_ir_validation_stack;
#[path = "tensor_ir/tensor_ir_validation_structural.rs"]
mod tensor_ir_validation_structural;
