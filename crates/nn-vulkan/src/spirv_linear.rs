// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for fused linear (matmul + bias) compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules for the linear transformation:
//!
//! ```text
//! output[row, col] = sum_k(input[row, k] * weight[col, k]) + bias[col]
//! ```
//!
//! Weight is stored in `[out_features, in_features]` layout (transposed
//! relative to the input), which is standard for `nn.Linear`. Each thread
//! computes one output element at position `(row, col)`.
//!
//! # Buffer layout
//!
//! - Binding 0: Input `[batch_size, in_features]` — `float[]` (row-major)
//! - Binding 1: Weight `[out_features, in_features]` — `float[]` (row-major)
//! - Binding 2: Bias `[out_features]` — `float[]` (only for biased variant)
//! - Binding 2/3: Output `[batch_size, out_features]` — `float[]` (row-major)
//!
//! # Push constants
//!
//! - `uint batch_size` at offset 0
//! - `uint in_features` at offset 4
//! - `uint out_features` at offset 8

use crate::spirv_emit::SPIRV_MAGIC;

/// Workgroup size for linear kernels (1D dispatch).
pub const LINEAR_WORKGROUP_SIZE: u32 = 64;

// ---- SPIR-V constants (duplicated to keep module independent) ----

const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;

const fn op(word_count: u16, opcode: u16) -> u32 {
    (word_count as u32) << 16 | opcode as u32
}

// Opcodes.
const OP_CAPABILITY: u16 = 17;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LOOP_MERGE: u16 = 246;
const OP_PHI: u16 = 245;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_FADD: u16 = 129;
const OP_FMUL: u16 = 133;
const OP_U_LESS_THAN: u16 = 176;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_IMUL: u16 = 132;
const OP_IADD: u16 = 128;
const OP_UDIV: u16 = 134;
const OP_UMOD: u16 = 137;

// Decorations.
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BLOCK: u32 = 2;

// Built-ins.
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// Storage classes.
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// Execution model / mode.
const EXECUTION_MODEL_GL_COMPUTE: u32 = 5;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

// Capability.
const CAPABILITY_SHADER: u32 = 1;

// Memory model.
const ADDRESSING_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;

// Function control.
const FUNCTION_CONTROL_NONE: u32 = 0;

/// Encode a string as SPIR-V literal words (null-terminated, padded to 4-byte boundary).
fn encode_string(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    let word_count = (bytes.len() + 1).div_ceil(4);
    let mut words = vec![0u32; word_count];
    for (i, &b) in bytes.iter().enumerate() {
        let word_idx = i / 4;
        let byte_idx = i % 4;
        words[word_idx] |= u32::from(b) << (byte_idx * 8);
    }
    words
}

/// SPIR-V module builder (local to this module).
struct SpirVBuilder {
    bound: u32,
    capabilities: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    execution_modes: Vec<u32>,
    annotations: Vec<u32>,
    type_declarations: Vec<u32>,
    functions: Vec<u32>,
}

impl SpirVBuilder {
    fn new() -> Self {
        Self {
            bound: 1,
            capabilities: Vec::new(),
            memory_model: Vec::new(),
            entry_points: Vec::new(),
            execution_modes: Vec::new(),
            annotations: Vec::new(),
            type_declarations: Vec::new(),
            functions: Vec::new(),
        }
    }

    fn id(&mut self) -> u32 {
        let id = self.bound;
        self.bound += 1;
        id
    }

    fn capability(&mut self, cap: u32) {
        self.capabilities.push(op(2, OP_CAPABILITY));
        self.capabilities.push(cap);
    }

    fn memory_model(&mut self, addressing: u32, model: u32) {
        self.memory_model.push(op(3, OP_MEMORY_MODEL));
        self.memory_model.push(addressing);
        self.memory_model.push(model);
    }

    fn entry_point_compute(&mut self, func_id: u32, name: &str, interface_ids: &[u32]) {
        let name_words = encode_string(name);
        let wc = 3 + name_words.len() as u16 + interface_ids.len() as u16;
        self.entry_points.push(op(wc, OP_ENTRY_POINT));
        self.entry_points.push(EXECUTION_MODEL_GL_COMPUTE);
        self.entry_points.push(func_id);
        self.entry_points.extend_from_slice(&name_words);
        self.entry_points.extend_from_slice(interface_ids);
    }

    fn execution_mode_local_size(&mut self, func_id: u32, x: u32, y: u32, z: u32) {
        self.execution_modes.push(op(6, OP_EXECUTION_MODE));
        self.execution_modes.push(func_id);
        self.execution_modes.push(EXECUTION_MODE_LOCAL_SIZE);
        self.execution_modes.push(x);
        self.execution_modes.push(y);
        self.execution_modes.push(z);
    }

    fn decorate(&mut self, target: u32, decoration: u32, operands: &[u32]) {
        let wc = 3 + operands.len() as u16;
        self.annotations.push(op(wc, OP_DECORATE));
        self.annotations.push(target);
        self.annotations.push(decoration);
        self.annotations.extend_from_slice(operands);
    }

    fn member_decorate(
        &mut self,
        struct_type: u32,
        member: u32,
        decoration: u32,
        operands: &[u32],
    ) {
        let wc = 4 + operands.len() as u16;
        self.annotations.push(op(wc, OP_MEMBER_DECORATE));
        self.annotations.push(struct_type);
        self.annotations.push(member);
        self.annotations.push(decoration);
        self.annotations.extend_from_slice(operands);
    }

    fn type_void(&mut self) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(2, OP_TYPE_VOID));
        self.type_declarations.push(result);
        result
    }

    fn type_bool(&mut self) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(2, OP_TYPE_BOOL));
        self.type_declarations.push(result);
        result
    }

    fn type_int(&mut self, width: u32, signedness: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_TYPE_INT));
        self.type_declarations.push(result);
        self.type_declarations.push(width);
        self.type_declarations.push(signedness);
        result
    }

    fn type_float(&mut self, width: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(3, OP_TYPE_FLOAT));
        self.type_declarations.push(result);
        self.type_declarations.push(width);
        result
    }

    fn type_vector(&mut self, component_type: u32, count: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_TYPE_VECTOR));
        self.type_declarations.push(result);
        self.type_declarations.push(component_type);
        self.type_declarations.push(count);
        result
    }

    fn type_runtime_array(&mut self, element_type: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(3, OP_TYPE_RUNTIME_ARRAY));
        self.type_declarations.push(result);
        self.type_declarations.push(element_type);
        result
    }

    fn type_struct(&mut self, member_types: &[u32]) -> u32 {
        let result = self.id();
        let wc = 2 + member_types.len() as u16;
        self.type_declarations.push(op(wc, OP_TYPE_STRUCT));
        self.type_declarations.push(result);
        self.type_declarations.extend_from_slice(member_types);
        result
    }

    fn type_pointer(&mut self, storage_class: u32, pointee_type: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_TYPE_POINTER));
        self.type_declarations.push(result);
        self.type_declarations.push(storage_class);
        self.type_declarations.push(pointee_type);
        result
    }

    fn type_function(&mut self, return_type: u32, param_types: &[u32]) -> u32 {
        let result = self.id();
        let wc = 3 + param_types.len() as u16;
        self.type_declarations.push(op(wc, OP_TYPE_FUNCTION));
        self.type_declarations.push(result);
        self.type_declarations.push(return_type);
        self.type_declarations.extend_from_slice(param_types);
        result
    }

    fn constant_u32(&mut self, type_id: u32, value: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_CONSTANT));
        self.type_declarations.push(type_id);
        self.type_declarations.push(result);
        self.type_declarations.push(value);
        result
    }

    fn constant_f32(&mut self, type_id: u32, value: f32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_CONSTANT));
        self.type_declarations.push(type_id);
        self.type_declarations.push(result);
        self.type_declarations.push(value.to_bits());
        result
    }

    fn variable_global(&mut self, ptr_type: u32, storage_class: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_VARIABLE));
        self.type_declarations.push(ptr_type);
        self.type_declarations.push(result);
        self.type_declarations.push(storage_class);
        result
    }

    fn func_begin(&mut self, result_type: u32, func_id: u32, control: u32, func_type: u32) {
        self.functions.push(op(5, OP_FUNCTION));
        self.functions.push(result_type);
        self.functions.push(func_id);
        self.functions.push(control);
        self.functions.push(func_type);
    }

    fn func_end(&mut self) {
        self.functions.push(op(1, OP_FUNCTION_END));
    }

    fn label(&mut self) -> u32 {
        let result = self.id();
        self.functions.push(op(2, OP_LABEL));
        self.functions.push(result);
        result
    }

    fn label_with_id(&mut self, id: u32) {
        self.functions.push(op(2, OP_LABEL));
        self.functions.push(id);
    }

    fn op_return(&mut self) {
        self.functions.push(op(1, OP_RETURN));
    }

    fn branch(&mut self, target_label: u32) {
        self.functions.push(op(2, OP_BRANCH));
        self.functions.push(target_label);
    }

    fn branch_conditional(&mut self, condition: u32, true_label: u32, false_label: u32) {
        self.functions.push(op(4, OP_BRANCH_CONDITIONAL));
        self.functions.push(condition);
        self.functions.push(true_label);
        self.functions.push(false_label);
    }

    fn selection_merge(&mut self, merge_label: u32) {
        self.functions.push(op(3, OP_SELECTION_MERGE));
        self.functions.push(merge_label);
        self.functions.push(0);
    }

    fn loop_merge(&mut self, merge_label: u32, continue_label: u32) {
        self.functions.push(op(4, OP_LOOP_MERGE));
        self.functions.push(merge_label);
        self.functions.push(continue_label);
        self.functions.push(0);
    }

    fn phi(&mut self, result_type: u32, operands: &[(u32, u32)]) -> u32 {
        let result = self.id();
        let wc = 3 + (operands.len() as u16) * 2;
        self.functions.push(op(wc, OP_PHI));
        self.functions.push(result_type);
        self.functions.push(result);
        for &(val, parent) in operands {
            self.functions.push(val);
            self.functions.push(parent);
        }
        result
    }

    fn load(&mut self, result_type: u32, pointer: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(4, OP_LOAD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(pointer);
        result
    }

    fn store(&mut self, pointer: u32, object: u32) {
        self.functions.push(op(3, OP_STORE));
        self.functions.push(pointer);
        self.functions.push(object);
    }

    fn access_chain(&mut self, result_type: u32, base: u32, indices: &[u32]) -> u32 {
        let result = self.id();
        let wc = 4 + indices.len() as u16;
        self.functions.push(op(wc, OP_ACCESS_CHAIN));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(base);
        self.functions.extend_from_slice(indices);
        result
    }

    fn composite_extract(&mut self, result_type: u32, composite: u32, index: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_COMPOSITE_EXTRACT));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(composite);
        self.functions.push(index);
        result
    }

    fn fadd(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FADD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn fmul(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FMUL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn u_less_than(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_LESS_THAN));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn u_greater_than_equal(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_GREATER_THAN_EQUAL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn imul(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_IMUL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn iadd(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_IADD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn udiv(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_UDIV));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn umod(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_UMOD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn build(self) -> Vec<u32> {
        let mut module = Vec::with_capacity(512);
        module.push(SPIRV_MAGIC);
        module.push(SPIRV_VERSION_1_0);
        module.push(GENERATOR_MAGIC);
        module.push(self.bound);
        module.push(0);
        module.extend_from_slice(&self.capabilities);
        module.extend_from_slice(&self.memory_model);
        module.extend_from_slice(&self.entry_points);
        module.extend_from_slice(&self.execution_modes);
        module.extend_from_slice(&self.annotations);
        module.extend_from_slice(&self.type_declarations);
        module.extend_from_slice(&self.functions);
        module
    }
}

/// Fixup a phi instruction to add an additional (value, parent) operand.
fn fixup_phi(functions: &mut Vec<u32>, phi_id: u32, value: u32, parent: u32) {
    let mut pos = 0;
    while pos < functions.len() {
        let word = functions[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 {
            break;
        }
        if opcode == OP_PHI && pos + 2 < functions.len() && functions[pos + 2] == phi_id {
            let insert_pos = pos + word_count;
            functions.insert(insert_pos, parent);
            functions.insert(insert_pos, value);
            let new_wc = word_count + 2;
            functions[pos] = op(new_wc as u16, OP_PHI);
            return;
        }
        pos += word_count;
    }
}

/// Generate a SPIR-V 1.0 binary for fused linear (matmul + bias).
///
/// Each thread computes one output element: `output[row, col] = dot(input[row, :], weight[col, :]) + bias[col]`.
///
/// # Arguments
///
/// * `in_features` - Input dimension (compile-time hint; actual from push constants).
/// * `out_features` - Output dimension (compile-time hint; actual from push constants).
///
/// # Buffers
///
/// - Binding 0: Input `[batch_size, in_features]`
/// - Binding 1: Weight `[out_features, in_features]`
/// - Binding 2: Bias `[out_features]`
/// - Binding 3: Output `[batch_size, out_features]`
///
/// # Returns
///
/// SPIR-V binary as `Vec<u32>`.
pub fn generate_linear_spirv(in_features: u32, out_features: u32) -> Vec<u32> {
    let _ = in_features;
    let _ = out_features;
    generate_linear_spirv_inner(true)
}

/// Generate a SPIR-V 1.0 binary for linear without bias.
///
/// Each thread computes: `output[row, col] = dot(input[row, :], weight[col, :])`.
///
/// # Arguments
///
/// * `in_features` - Input dimension (compile-time hint; actual from push constants).
/// * `out_features` - Output dimension (compile-time hint; actual from push constants).
///
/// # Buffers
///
/// - Binding 0: Input `[batch_size, in_features]`
/// - Binding 1: Weight `[out_features, in_features]`
/// - Binding 2: Output `[batch_size, out_features]`
///
/// # Returns
///
/// SPIR-V binary as `Vec<u32>`.
pub fn generate_linear_no_bias_spirv(in_features: u32, out_features: u32) -> Vec<u32> {
    let _ = in_features;
    let _ = out_features;
    generate_linear_spirv_inner(false)
}

/// Internal implementation for both biased and non-biased linear.
///
/// # Algorithm
///
/// ```text
/// gid = gl_GlobalInvocationID.x
/// total = batch_size * out_features
/// if gid >= total: return
///
/// col = gid % out_features
/// row = gid / out_features
///
/// acc = 0.0
/// for k in 0..in_features:
///     acc += input[row * in_features + k] * weight[col * in_features + k]
///
/// if has_bias:
///     acc += bias[col]
///
/// output[row * out_features + col] = acc
/// ```
fn generate_linear_spirv_inner(has_bias: bool) -> Vec<u32> {
    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // ---- Types ----
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Input buffer struct (binding 0).
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    // Weight buffer struct (binding 1).
    let ty_struct_weight = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_weight, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_weight, 0, DECORATION_OFFSET, &[0]);

    // Bias buffer struct (binding 2, only for biased variant).
    let ty_struct_bias = if has_bias {
        let s = b.type_struct(&[ty_rtarr_float]);
        b.decorate(s, DECORATION_BLOCK, &[]);
        b.member_decorate(s, 0, DECORATION_OFFSET, &[0]);
        Some(s)
    } else {
        None
    };

    // Output buffer struct.
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint batch_size, uint in_features, uint out_features }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);
    b.member_decorate(ty_struct_pc, 2, DECORATION_OFFSET, &[8]);

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_weight = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_weight);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    let ptr_sb_bias = if has_bias {
        Some(b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_bias.unwrap()))
    } else {
        None
    };

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_2u = b.constant_u32(ty_uint, 2);
    let const_f0 = b.constant_f32(ty_float, 0.0);

    // Global variables.
    let var_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_input, DECORATION_BINDING, &[0]);

    let var_weight = b.variable_global(ptr_sb_weight, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_weight, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_weight, DECORATION_BINDING, &[1]);

    let var_bias = if has_bias {
        let v = b.variable_global(ptr_sb_bias.unwrap(), STORAGE_CLASS_STORAGE_BUFFER);
        b.decorate(v, DECORATION_DESCRIPTOR_SET, &[0]);
        b.decorate(v, DECORATION_BINDING, &[2]);
        Some(v)
    } else {
        None
    };

    let output_binding = if has_bias { 3 } else { 2 };
    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[output_binding]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, LINEAR_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    // Load gl_GlobalInvocationID.x
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load push constants: batch_size, in_features, out_features.
    let pc_batch_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let batch_size = b.load(ty_uint, pc_batch_ptr);
    let pc_in_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let in_features = b.load(ty_uint, pc_in_ptr);
    let pc_out_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_2u]);
    let out_features = b.load(ty_uint, pc_out_ptr);

    // total = batch_size * out_features
    let total = b.imul(ty_uint, batch_size, out_features);

    // Bounds check: gid >= total -> return.
    let cmp_oob = b.u_greater_than_equal(ty_bool, gid, total);
    let return_label = b.id();
    let body_label = b.id();
    b.selection_merge(body_label);
    b.branch_conditional(cmp_oob, return_label, body_label);

    b.label_with_id(return_label);
    b.op_return();

    b.label_with_id(body_label);

    // col = gid % out_features
    let col = b.umod(ty_uint, gid, out_features);
    // row = gid / out_features
    let row = b.udiv(ty_uint, gid, out_features);

    // Precompute row_offset = row * in_features, weight_offset = col * in_features.
    let row_offset = b.imul(ty_uint, row, in_features);
    let weight_offset = b.imul(ty_uint, col, in_features);

    // ---- Dot product loop ----
    let loop_header = b.id();
    let loop_body = b.id();
    let loop_continue = b.id();
    let loop_merge = b.id();

    b.branch(loop_header);

    b.label_with_id(loop_header);
    b.loop_merge(loop_merge, loop_continue);
    let phi_k = b.phi(ty_uint, &[(const_0u, body_label)]);
    let phi_acc = b.phi(ty_float, &[(const_f0, body_label)]);
    let cmp_k = b.u_less_than(ty_bool, phi_k, in_features);
    b.branch_conditional(cmp_k, loop_body, loop_merge);

    // Loop body: acc += input[row_offset + k] * weight[weight_offset + k]
    b.label_with_id(loop_body);

    let input_idx = b.iadd(ty_uint, row_offset, phi_k);
    let input_ptr = b.access_chain(ptr_sb_float, var_input, &[const_0u, input_idx]);
    let input_val = b.load(ty_float, input_ptr);

    let weight_idx = b.iadd(ty_uint, weight_offset, phi_k);
    let weight_ptr = b.access_chain(ptr_sb_float, var_weight, &[const_0u, weight_idx]);
    let weight_val = b.load(ty_float, weight_ptr);

    let prod = b.fmul(ty_float, input_val, weight_val);
    let new_acc = b.fadd(ty_float, phi_acc, prod);

    b.branch(loop_continue);
    b.label_with_id(loop_continue);
    let next_k = b.iadd(ty_uint, phi_k, const_1u);
    fixup_phi(&mut b.functions, phi_k, next_k, loop_continue);
    fixup_phi(&mut b.functions, phi_acc, new_acc, loop_continue);
    b.branch(loop_header);

    b.label_with_id(loop_merge);

    // Add bias if present.
    let final_val = if has_bias {
        let bias_ptr = b.access_chain(ptr_sb_float, var_bias.unwrap(), &[const_0u, col]);
        let bias_val = b.load(ty_float, bias_ptr);
        b.fadd(ty_float, phi_acc, bias_val)
    } else {
        phi_acc
    };

    // Store output[row * out_features + col].
    let row_times_out = b.imul(ty_uint, row, out_features);
    let out_idx = b.iadd(ty_uint, row_times_out, col);
    let out_ptr = b.access_chain(ptr_sb_float, var_output, &[const_0u, out_idx]);
    b.store(out_ptr, final_val);

    b.op_return();
    b.func_end();

    b.build()
}

/// CPU reference implementation of linear transformation.
///
/// Computes `output[row, col] = dot(input[row, :], weight[col, :]) + bias[col]`.
///
/// # Arguments
///
/// * `input` - Input tensor, flattened as `[batch_size * in_features]`.
/// * `weight` - Weight matrix, flattened as `[out_features * in_features]`.
/// * `bias` - Optional bias vector of length `out_features`.
/// * `in_features` - Input dimension.
/// * `out_features` - Output dimension.
///
/// # Returns
///
/// Output tensor, flattened as `[batch_size * out_features]`.
pub fn linear_reference(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    assert_eq!(weight.len(), out_features * in_features);
    let batch_size = input.len() / in_features;
    assert_eq!(
        batch_size * in_features,
        input.len(),
        "input length must be divisible by in_features"
    );
    if let Some(b) = bias {
        assert_eq!(b.len(), out_features, "bias length must equal out_features");
    }

    let mut output = vec![0.0f32; batch_size * out_features];
    for row in 0..batch_size {
        for col in 0..out_features {
            let mut acc = 0.0f32;
            for k in 0..in_features {
                acc += input[row * in_features + k] * weight[col * in_features + k];
            }
            if let Some(b) = bias {
                acc += b[col];
            }
            output[row * out_features + col] = acc;
        }
    }
    output
}

#[cfg(test)]
#[path = "spirv_linear_tests.rs"]
mod spirv_linear_tests;
