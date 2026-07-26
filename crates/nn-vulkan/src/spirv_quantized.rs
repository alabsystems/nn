// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Direct SPIR-V binary generation for quantized operations.
//!
//! Generates SPIR-V compute shaders for INT8 quantize/dequantize without
//! external compilers. These kernels enable efficient inference with
//! quantized model weights on Vulkan-capable hardware.
//!
//! # Supported operations
//!
//! - [`generate_dequantize_int8_spirv`]: INT8 dequantize: `output[i] = scale * (int8[i] - zero_point)`
//! - [`generate_quantize_f32_to_int8_spirv`]: F32 quantize: `output[i] = clamp(round(input[i] / scale) + zero_point, -128, 127)`
//!
//! # CPU reference implementations
//!
//! - [`dequantize_reference`]: CPU dequantize for differential testing
//! - [`quantize_reference`]: CPU quantize for differential testing
//!
//! # Buffer layout
//!
//! Dequantize shader:
//! - Binding 0: Input buffer (int[]) — packed i8 values stored as int32
//! - Binding 1: Output buffer (float[])
//! - Push constants: { uint n, float scale, int zero_point }
//!
//! Quantize shader:
//! - Binding 0: Input buffer (float[])
//! - Binding 1: Output buffer (int[]) — packed i8 values stored as int32
//! - Push constants: { uint n, float scale, int zero_point }
//!
//! All shaders use workgroup size of 256 threads (1D dispatch).

/// Workgroup size for quantized operation shaders.
pub const QUANTIZED_WORKGROUP_SIZE: u32 = 256;

// ---- SPIR-V opcode helpers (local to this module) ----

/// Encode a SPIR-V instruction word: (word_count << 16) | opcode.
const fn op(word_count: u16, opcode: u16) -> u32 {
    (word_count as u32) << 16 | opcode as u32
}

// SPIR-V opcodes used in this module.
const OP_CAPABILITY: u16 = 17;
const OP_EXT_INST_IMPORT: u16 = 11;
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
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_FADD: u16 = 129;
const OP_FMUL: u16 = 133;
#[allow(dead_code)]
const OP_FSUB: u16 = 131;
const OP_FDIV: u16 = 136;
const OP_ISUB: u16 = 130;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_CONVERT_S_TO_F: u16 = 111;
const OP_CONVERT_F_TO_S: u16 = 110;
const OP_EXT_INST: u16 = 12;
const OP_S_LESS_THAN: u16 = 177;
const OP_S_GREATER_THAN: u16 = 175;
const OP_SELECT: u16 = 169;

// Decoration constants.
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BLOCK: u32 = 2;

// Built-in constants.
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// Storage class constants.
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

// GLSL.std.450 extended instruction set opcodes.
const GLSL_STD_450_ROUND: u32 = 1;

/// SPIR-V version 1.0.
const SPIRV_VERSION_1_0: u32 = 0x0001_0000;

/// Generator magic (nn-vulkan = 0x4E4E0000, "NN\0").
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;

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

/// SPIR-V module builder for quantized ops.
struct SpirVBuilder {
    bound: u32,
    capabilities: Vec<u32>,
    extensions: Vec<u32>,
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
            extensions: Vec::new(),
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

    fn ext_inst_import(&mut self, name: &str) -> u32 {
        let result = self.id();
        let name_words = encode_string(name);
        let wc = 2 + name_words.len() as u16;
        self.extensions.push(op(wc, OP_EXT_INST_IMPORT));
        self.extensions.push(result);
        self.extensions.extend_from_slice(&name_words);
        result
    }

    fn memory_model_decl(&mut self, addressing: u32, model: u32) {
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

    fn constant_i32(&mut self, type_id: u32, value: i32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_CONSTANT));
        self.type_declarations.push(type_id);
        self.type_declarations.push(result);
        self.type_declarations.push(value as u32);
        result
    }

    #[allow(dead_code)]
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

    // ---- Function body instructions ----

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
        self.functions.push(0); // SelectionControl: None
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

    fn fmul(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FMUL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    #[allow(dead_code)]
    fn fadd(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FADD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    #[allow(dead_code)]
    fn fsub(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FSUB));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn fdiv(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FDIV));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn isub(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_ISUB));
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

    fn convert_s_to_f(&mut self, result_type: u32, value: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(4, OP_CONVERT_S_TO_F));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(value);
        result
    }

    fn convert_f_to_s(&mut self, result_type: u32, value: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(4, OP_CONVERT_F_TO_S));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(value);
        result
    }

    fn ext_inst(
        &mut self,
        result_type: u32,
        ext_set: u32,
        instruction: u32,
        operands: &[u32],
    ) -> u32 {
        let result = self.id();
        let wc = 5 + operands.len() as u16;
        self.functions.push(op(wc, OP_EXT_INST));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(ext_set);
        self.functions.push(instruction);
        self.functions.extend_from_slice(operands);
        result
    }

    fn s_less_than(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_S_LESS_THAN));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn s_greater_than(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_S_GREATER_THAN));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn select(&mut self, result_type: u32, condition: u32, true_val: u32, false_val: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(6, OP_SELECT));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(condition);
        self.functions.push(true_val);
        self.functions.push(false_val);
        result
    }

    fn build(self) -> Vec<u32> {
        let mut module = Vec::with_capacity(256);

        // Header.
        module.push(crate::spirv_emit::SPIRV_MAGIC);
        module.push(SPIRV_VERSION_1_0);
        module.push(GENERATOR_MAGIC);
        module.push(self.bound);
        module.push(0); // Reserved schema.

        // Sections in required order.
        module.extend_from_slice(&self.capabilities);
        module.extend_from_slice(&self.extensions);
        module.extend_from_slice(&self.memory_model);
        module.extend_from_slice(&self.entry_points);
        module.extend_from_slice(&self.execution_modes);
        module.extend_from_slice(&self.annotations);
        module.extend_from_slice(&self.type_declarations);
        module.extend_from_slice(&self.functions);

        module
    }
}

/// Generate a SPIR-V binary for INT8 dequantization.
///
/// Computes: `output[i] = scale * (int_input[i] - zero_point)`
///
/// The input buffer stores int8 values as int32 (sign-extended). Each thread
/// processes one element. The `n` parameter is passed via push constants.
///
/// # Buffers
///
/// - Binding 0: Input buffer (int[]) — i8 values stored as i32
/// - Binding 1: Output buffer (float[])
///
/// # Push constants
///
/// - `uint n` at offset 0 — number of elements
/// - `float scale` at offset 4
/// - `int zero_point` at offset 8
///
/// # Arguments
///
/// * `_n` - Number of elements (for future use in specialization; currently
///   passed via push constants at dispatch time)
#[must_use]
pub fn generate_dequantize_int8_spirv(_n: u32) -> Vec<u32> {
    let mut b = SpirVBuilder::new();

    let func_id = b.id();

    // Capability + extensions.
    b.capability(CAPABILITY_SHADER);
    let glsl_ext = b.ext_inst_import("GLSL.std.450");
    // Suppress unused variable warning — glsl_ext reserved for future GLSL.std.450 use.
    let _ = glsl_ext;

    // Memory model.
    b.memory_model_decl(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // --- Types ---
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0); // unsigned 32-bit
    let ty_int = b.type_int(32, 1); // signed 32-bit
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime array of int (input: i8 stored as i32).
    let ty_rtarr_int = b.type_runtime_array(ty_int);
    b.decorate(ty_rtarr_int, DECORATION_ARRAY_STRIDE, &[4]);

    // Runtime array of float (output).
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Struct for input buffer: { int data[]; }
    let ty_struct_in = b.type_struct(&[ty_rtarr_int]);
    b.decorate(ty_struct_in, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_in, 0, DECORATION_OFFSET, &[0]);

    // Struct for output buffer: { float data[]; }
    let ty_struct_out = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_out, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_out, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n; float scale; int zero_point; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_float, ty_int]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);
    b.member_decorate(ty_struct_pc, 2, DECORATION_OFFSET, &[8]);

    // Pointer types.
    let ptr_sb_in = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_in);
    let ptr_sb_out = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_out);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_int = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_int);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);
    let ptr_pc_float = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_float);
    let ptr_pc_int = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_int);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_2u = b.constant_u32(ty_uint, 2);

    // Variables.
    let var_in = b.variable_global(ptr_sb_in, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_in, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_in, DECORATION_BINDING, &[0]);

    let var_out = b.variable_global(ptr_sb_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_out, DECORATION_BINDING, &[1]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, QUANTIZED_WORKGROUP_SIZE, 1, 1);

    // --- Function body ---
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry = b.label();

    // Load gl_GlobalInvocationID.x
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let idx = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load n from push constants.
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let n = b.load(ty_uint, pc_n_ptr);

    // Bounds check: if (idx >= n) return.
    let cmp = b.u_greater_than_equal(ty_bool, idx, n);
    let merge_label = b.id();
    let then_label = b.id();
    b.selection_merge(merge_label);
    b.branch_conditional(cmp, merge_label, then_label);

    // Then block (idx < n).
    b.label_with_id(then_label);

    // Load scale from push constants.
    let pc_scale_ptr = b.access_chain(ptr_pc_float, var_pc, &[const_1u]);
    let scale = b.load(ty_float, pc_scale_ptr);

    // Load zero_point from push constants.
    let pc_zp_ptr = b.access_chain(ptr_pc_int, var_pc, &[const_2u]);
    let zero_point = b.load(ty_int, pc_zp_ptr);

    // Load input[idx] (int32 containing i8 value).
    let ptr_data_in = b.access_chain(ptr_sb_int, var_in, &[const_0u, idx]);
    let val_int = b.load(ty_int, ptr_data_in);

    // Compute: (val_int - zero_point)
    let diff = b.isub(ty_int, val_int, zero_point);

    // Convert to float.
    let diff_f = b.convert_s_to_f(ty_float, diff);

    // Multiply by scale: output = scale * (val - zero_point)
    let result = b.fmul(ty_float, scale, diff_f);

    // Store to output[idx].
    let ptr_data_out = b.access_chain(ptr_sb_float, var_out, &[const_0u, idx]);
    b.store(ptr_data_out, result);

    // Branch to merge.
    b.branch(merge_label);

    // Merge block.
    b.label_with_id(merge_label);
    b.op_return();
    b.func_end();

    b.build()
}

/// Generate a SPIR-V binary for F32-to-INT8 quantization.
///
/// Computes: `output[i] = clamp(round(input[i] / scale) + zero_point, -128, 127)`
///
/// The output buffer stores int8 values as int32 (sign-extended). Each thread
/// processes one element. The `n` parameter is passed via push constants.
///
/// # Buffers
///
/// - Binding 0: Input buffer (float[])
/// - Binding 1: Output buffer (int[]) — i8 values stored as i32
///
/// # Push constants
///
/// - `uint n` at offset 0 — number of elements
/// - `float scale` at offset 4
/// - `int zero_point` at offset 8
///
/// # Arguments
///
/// * `_n` - Number of elements (for future use in specialization; currently
///   passed via push constants at dispatch time)
#[must_use]
pub fn generate_quantize_f32_to_int8_spirv(_n: u32) -> Vec<u32> {
    let mut b = SpirVBuilder::new();

    let func_id = b.id();

    // Capability + extensions.
    b.capability(CAPABILITY_SHADER);
    let glsl_ext = b.ext_inst_import("GLSL.std.450");

    // Memory model.
    b.memory_model_decl(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // --- Types ---
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_int = b.type_int(32, 1);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime array of float (input).
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Runtime array of int (output: i8 stored as i32).
    let ty_rtarr_int = b.type_runtime_array(ty_int);
    b.decorate(ty_rtarr_int, DECORATION_ARRAY_STRIDE, &[4]);

    // Struct for input buffer.
    let ty_struct_in = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_in, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_in, 0, DECORATION_OFFSET, &[0]);

    // Struct for output buffer.
    let ty_struct_out = b.type_struct(&[ty_rtarr_int]);
    b.decorate(ty_struct_out, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_out, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n; float scale; int zero_point; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_float, ty_int]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);
    b.member_decorate(ty_struct_pc, 2, DECORATION_OFFSET, &[8]);

    // Pointer types.
    let ptr_sb_in = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_in);
    let ptr_sb_out = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_out);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_sb_int = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_int);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);
    let ptr_pc_float = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_float);
    let ptr_pc_int = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_int);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_2u = b.constant_u32(ty_uint, 2);
    let const_min_i8 = b.constant_i32(ty_int, -128);
    let const_max_i8 = b.constant_i32(ty_int, 127);

    // Variables.
    let var_in = b.variable_global(ptr_sb_in, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_in, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_in, DECORATION_BINDING, &[0]);

    let var_out = b.variable_global(ptr_sb_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_out, DECORATION_BINDING, &[1]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, QUANTIZED_WORKGROUP_SIZE, 1, 1);

    // --- Function body ---
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry = b.label();

    // Load gl_GlobalInvocationID.x
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let idx = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load n from push constants.
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let n = b.load(ty_uint, pc_n_ptr);

    // Bounds check: if (idx >= n) return.
    let cmp = b.u_greater_than_equal(ty_bool, idx, n);
    let merge_label = b.id();
    let then_label = b.id();
    b.selection_merge(merge_label);
    b.branch_conditional(cmp, merge_label, then_label);

    // Then block (idx < n).
    b.label_with_id(then_label);

    // Load scale from push constants.
    let pc_scale_ptr = b.access_chain(ptr_pc_float, var_pc, &[const_1u]);
    let scale = b.load(ty_float, pc_scale_ptr);

    // Load zero_point from push constants.
    let pc_zp_ptr = b.access_chain(ptr_pc_int, var_pc, &[const_2u]);
    let zero_point = b.load(ty_int, pc_zp_ptr);

    // Load input[idx].
    let ptr_data_in = b.access_chain(ptr_sb_float, var_in, &[const_0u, idx]);
    let val_f = b.load(ty_float, ptr_data_in);

    // Compute: input / scale
    let divided = b.fdiv(ty_float, val_f, scale);

    // Round to nearest integer (GLSL.std.450 Round).
    let rounded = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_ROUND, &[divided]);

    // Convert float to signed int.
    let as_int = b.convert_f_to_s(ty_int, rounded);

    // Add zero_point: quantized = round(input / scale) + zero_point
    // Using SPIR-V OpIAdd (opcode 128).
    let with_zp = {
        let result = b.id();
        b.functions.push(op(5, 128)); // OpIAdd
        b.functions.push(ty_int);
        b.functions.push(result);
        b.functions.push(as_int);
        b.functions.push(zero_point);
        result
    };

    // Clamp to [-128, 127] using OpSelect.
    // if (with_zp < -128) { -128 } else if (with_zp > 127) { 127 } else { with_zp }
    let cmp_lo = b.s_less_than(ty_bool, with_zp, const_min_i8);
    let clamped_lo = b.select(ty_int, cmp_lo, const_min_i8, with_zp);
    let cmp_hi = b.s_greater_than(ty_bool, clamped_lo, const_max_i8);
    let clamped = b.select(ty_int, cmp_hi, const_max_i8, clamped_lo);

    // Store to output[idx].
    let ptr_data_out = b.access_chain(ptr_sb_int, var_out, &[const_0u, idx]);
    b.store(ptr_data_out, clamped);

    // Branch to merge.
    b.branch(merge_label);

    // Merge block.
    b.label_with_id(merge_label);
    b.op_return();
    b.func_end();

    b.build()
}

/// CPU reference implementation for INT8 dequantization.
///
/// Computes: `output[i] = scale * (data[i] - zero_point)` for each element.
///
/// This is the exact same computation as the SPIR-V shader, used for
/// differential testing.
#[must_use]
pub fn dequantize_reference(data: &[i8], scale: f32, zero_point: i8) -> Vec<f32> {
    data.iter()
        .map(|&val| scale * (f32::from(val) - f32::from(zero_point)))
        .collect()
}

/// CPU reference implementation for F32-to-INT8 quantization.
///
/// Computes: `output[i] = clamp(round(input[i] / scale) + zero_point, -128, 127)`.
///
/// This is the exact same computation as the SPIR-V shader, used for
/// differential testing.
#[must_use]
pub fn quantize_reference(data: &[f32], scale: f32, zero_point: i8) -> Vec<i8> {
    data.iter()
        .map(|&val| {
            let quantized = (val / scale).round() as i32 + i32::from(zero_point);
            quantized.clamp(-128, 127) as i8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
    use crate::spirv_emit::SPIRV_MAGIC;

    // ---- Header validation ----

    fn assert_valid_header(spirv: &[u32], label: &str) {
        assert!(spirv.len() >= 5, "{label}: module too short");
        assert_eq!(spirv[0], SPIRV_MAGIC, "{label}: wrong magic number");
        assert_eq!(spirv[1], SPIRV_VERSION_1_0, "{label}: wrong SPIR-V version");
        assert_eq!(spirv[2], GENERATOR_MAGIC, "{label}: wrong generator magic");
        assert!(spirv[3] > 0, "{label}: bound must be > 0");
        assert_eq!(spirv[4], 0, "{label}: schema must be 0");
    }

    fn assert_entry_point_main(spirv: &[u32], label: &str) {
        let name =
            find_entry_point_name(spirv).unwrap_or_else(|| panic!("{label}: no entry point found"));
        assert_eq!(name, "main", "{label}: entry point name must be 'main'");
    }

    fn assert_workgroup_size(spirv: &[u32], label: &str) {
        let wg = find_workgroup_size(spirv)
            .unwrap_or_else(|| panic!("{label}: no workgroup size found"));
        assert_eq!(
            wg,
            [QUANTIZED_WORKGROUP_SIZE, 1, 1],
            "{label}: wrong workgroup size"
        );
    }

    fn has_opcode(spirv: &[u32], target_opcode: u16) -> bool {
        let mut pos = 5;
        while pos < spirv.len() {
            let word = spirv[pos];
            let word_count = (word >> 16) as usize;
            let opcode = (word & 0xFFFF) as u16;
            if word_count == 0 || pos + word_count > spirv.len() {
                break;
            }
            if opcode == target_opcode {
                return true;
            }
            pos += word_count;
        }
        false
    }

    // ---- generate_dequantize_int8_spirv ----

    #[test]
    fn test_dequantize_spirv_valid_header() {
        let spirv = generate_dequantize_int8_spirv(1024);
        assert_valid_header(&spirv, "dequantize_int8");
    }

    #[test]
    fn test_dequantize_spirv_entry_point() {
        let spirv = generate_dequantize_int8_spirv(1024);
        assert_entry_point_main(&spirv, "dequantize_int8");
    }

    #[test]
    fn test_dequantize_spirv_workgroup_size() {
        let spirv = generate_dequantize_int8_spirv(1024);
        assert_workgroup_size(&spirv, "dequantize_int8");
    }

    #[test]
    fn test_dequantize_spirv_contains_convert_s_to_f() {
        let spirv = generate_dequantize_int8_spirv(1024);
        assert!(
            has_opcode(&spirv, OP_CONVERT_S_TO_F),
            "dequantize shader must contain OpConvertSToF for int-to-float conversion"
        );
    }

    #[test]
    fn test_dequantize_spirv_contains_fmul() {
        let spirv = generate_dequantize_int8_spirv(1024);
        assert!(
            has_opcode(&spirv, OP_FMUL),
            "dequantize shader must contain OpFMul for scale multiplication"
        );
    }

    #[test]
    fn test_dequantize_spirv_contains_isub() {
        let spirv = generate_dequantize_int8_spirv(1024);
        assert!(
            has_opcode(&spirv, OP_ISUB),
            "dequantize shader must contain OpISub for zero_point subtraction"
        );
    }

    // ---- generate_quantize_f32_to_int8_spirv ----

    #[test]
    fn test_quantize_spirv_valid_header() {
        let spirv = generate_quantize_f32_to_int8_spirv(1024);
        assert_valid_header(&spirv, "quantize_f32_to_int8");
    }

    #[test]
    fn test_quantize_spirv_entry_point() {
        let spirv = generate_quantize_f32_to_int8_spirv(1024);
        assert_entry_point_main(&spirv, "quantize_f32_to_int8");
    }

    #[test]
    fn test_quantize_spirv_workgroup_size() {
        let spirv = generate_quantize_f32_to_int8_spirv(1024);
        assert_workgroup_size(&spirv, "quantize_f32_to_int8");
    }

    #[test]
    fn test_quantize_spirv_contains_convert_f_to_s() {
        let spirv = generate_quantize_f32_to_int8_spirv(1024);
        assert!(
            has_opcode(&spirv, OP_CONVERT_F_TO_S),
            "quantize shader must contain OpConvertFToS for float-to-int conversion"
        );
    }

    #[test]
    fn test_quantize_spirv_contains_fdiv() {
        let spirv = generate_quantize_f32_to_int8_spirv(1024);
        assert!(
            has_opcode(&spirv, OP_FDIV),
            "quantize shader must contain OpFDiv for scale division"
        );
    }

    #[test]
    fn test_quantize_spirv_contains_select() {
        let spirv = generate_quantize_f32_to_int8_spirv(1024);
        assert!(
            has_opcode(&spirv, OP_SELECT),
            "quantize shader must contain OpSelect for clamping"
        );
    }

    #[test]
    fn test_quantize_spirv_contains_ext_inst_round() {
        let spirv = generate_quantize_f32_to_int8_spirv(1024);
        assert!(
            has_opcode(&spirv, OP_EXT_INST),
            "quantize shader must use GLSL.std.450 Round"
        );
    }

    // ---- CPU reference: dequantize ----

    #[test]
    fn test_dequantize_reference_basic() {
        let data: Vec<i8> = vec![10, 20, 30, -10, -20];
        let scale = 0.5;
        let zero_point = 0i8;
        let result = dequantize_reference(&data, scale, zero_point);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 5.0).abs() < 1e-6);
        assert!((result[1] - 10.0).abs() < 1e-6);
        assert!((result[2] - 15.0).abs() < 1e-6);
        assert!((result[3] - (-5.0)).abs() < 1e-6);
        assert!((result[4] - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_reference_with_zero_point() {
        let data: Vec<i8> = vec![10, 20, 0];
        let scale = 1.0;
        let zero_point = 5i8;
        let result = dequantize_reference(&data, scale, zero_point);
        // output = scale * (val - zero_point)
        assert!((result[0] - 5.0).abs() < 1e-6); // 1.0 * (10 - 5)
        assert!((result[1] - 15.0).abs() < 1e-6); // 1.0 * (20 - 5)
        assert!((result[2] - (-5.0)).abs() < 1e-6); // 1.0 * (0 - 5)
    }

    #[test]
    fn test_dequantize_reference_edge_values() {
        let data: Vec<i8> = vec![i8::MIN, i8::MAX, 0];
        let scale = 0.1;
        let zero_point = 0i8;
        let result = dequantize_reference(&data, scale, zero_point);
        assert!((result[0] - (-12.8)).abs() < 1e-5); // 0.1 * -128
        assert!((result[1] - 12.7).abs() < 1e-5); // 0.1 * 127
        assert!((result[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_reference_empty() {
        let result = dequantize_reference(&[], 1.0, 0);
        assert!(result.is_empty());
    }

    // ---- CPU reference: quantize ----

    #[test]
    fn test_quantize_reference_basic() {
        let data = vec![5.0f32, 10.0, 15.0, -5.0, -10.0];
        let scale = 0.5;
        let zero_point = 0i8;
        let result = quantize_reference(&data, scale, zero_point);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 10); // round(5.0 / 0.5) + 0
        assert_eq!(result[1], 20); // round(10.0 / 0.5) + 0
        assert_eq!(result[2], 30); // round(15.0 / 0.5) + 0
        assert_eq!(result[3], -10); // round(-5.0 / 0.5) + 0
        assert_eq!(result[4], -20); // round(-10.0 / 0.5) + 0
    }

    #[test]
    fn test_quantize_reference_with_zero_point() {
        let data = vec![5.0f32, 15.0, -5.0];
        let scale = 1.0;
        let zero_point = 5i8;
        let result = quantize_reference(&data, scale, zero_point);
        assert_eq!(result[0], 10); // round(5.0 / 1.0) + 5
        assert_eq!(result[1], 20); // round(15.0 / 1.0) + 5
        assert_eq!(result[2], 0); // round(-5.0 / 1.0) + 5
    }

    #[test]
    fn test_quantize_reference_clamping_overflow() {
        // Value that would exceed 127 after quantization.
        let data = vec![1000.0f32];
        let scale = 1.0;
        let zero_point = 0i8;
        let result = quantize_reference(&data, scale, zero_point);
        assert_eq!(result[0], 127); // Clamped to max i8
    }

    #[test]
    fn test_quantize_reference_clamping_underflow() {
        // Value that would go below -128 after quantization.
        let data = vec![-1000.0f32];
        let scale = 1.0;
        let zero_point = 0i8;
        let result = quantize_reference(&data, scale, zero_point);
        assert_eq!(result[0], -128); // Clamped to min i8
    }

    #[test]
    fn test_quantize_reference_empty() {
        let result = quantize_reference(&[], 1.0, 0);
        assert!(result.is_empty());
    }

    // ---- Roundtrip: quantize then dequantize ----

    #[test]
    fn test_roundtrip_quantize_dequantize() {
        let scale = 0.1;
        let zero_point = 0i8;
        // Values within representable range: [-12.8, 12.7] for scale=0.1
        let original = vec![0.0f32, 1.0, -1.0, 5.0, -5.0, 10.0, -10.0];
        let quantized = quantize_reference(&original, scale, zero_point);
        let dequantized = dequantize_reference(&quantized, scale, zero_point);

        for (i, (&orig, &deq)) in original.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            assert!(
                error <= scale / 2.0 + 1e-6,
                "Roundtrip error at index {i}: orig={orig}, deq={deq}, error={error}, max_allowed={}",
                scale / 2.0
            );
        }
    }

    #[test]
    fn test_roundtrip_with_zero_point() {
        let scale = 0.5;
        let zero_point = 10i8;
        let original = vec![0.0f32, 5.0, -5.0, 20.0, -20.0];
        let quantized = quantize_reference(&original, scale, zero_point);
        let dequantized = dequantize_reference(&quantized, scale, zero_point);

        for (i, (&orig, &deq)) in original.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            assert!(
                error <= scale / 2.0 + 1e-6,
                "Roundtrip error at index {i}: orig={orig}, deq={deq}, error={error}",
            );
        }
    }

    // ---- Cross-cutting structural tests ----

    #[test]
    fn test_all_quantized_shaders_have_capability() {
        for (name, spirv) in [
            ("dequantize", generate_dequantize_int8_spirv(256)),
            ("quantize", generate_quantize_f32_to_int8_spirv(256)),
        ] {
            assert!(
                has_opcode(&spirv, OP_CAPABILITY),
                "{name}: must have OpCapability"
            );
        }
    }

    #[test]
    fn test_all_quantized_shaders_have_memory_model() {
        for (name, spirv) in [
            ("dequantize", generate_dequantize_int8_spirv(256)),
            ("quantize", generate_quantize_f32_to_int8_spirv(256)),
        ] {
            assert!(
                has_opcode(&spirv, OP_MEMORY_MODEL),
                "{name}: must have OpMemoryModel"
            );
        }
    }

    #[test]
    fn test_all_quantized_shaders_have_function_structure() {
        for (name, spirv) in [
            ("dequantize", generate_dequantize_int8_spirv(256)),
            ("quantize", generate_quantize_f32_to_int8_spirv(256)),
        ] {
            assert!(
                has_opcode(&spirv, OP_FUNCTION),
                "{name}: must have OpFunction"
            );
            assert!(
                has_opcode(&spirv, OP_FUNCTION_END),
                "{name}: must have OpFunctionEnd"
            );
            assert!(has_opcode(&spirv, OP_LABEL), "{name}: must have OpLabel");
            assert!(has_opcode(&spirv, OP_RETURN), "{name}: must have OpReturn");
        }
    }

    #[test]
    fn test_all_quantized_shaders_have_bounds_check() {
        for (name, spirv) in [
            ("dequantize", generate_dequantize_int8_spirv(256)),
            ("quantize", generate_quantize_f32_to_int8_spirv(256)),
        ] {
            assert!(
                has_opcode(&spirv, OP_U_GREATER_THAN_EQUAL),
                "{name}: must have bounds check (OpUGreaterThanEqual)"
            );
            assert!(
                has_opcode(&spirv, OP_BRANCH_CONDITIONAL),
                "{name}: must have conditional branch for bounds check"
            );
        }
    }

    #[test]
    fn test_dequantize_different_n_values_produce_valid_spirv() {
        for n in [1, 64, 256, 1024, 65536] {
            let spirv = generate_dequantize_int8_spirv(n);
            assert_valid_header(&spirv, &format!("dequantize_n={n}"));
            assert_entry_point_main(&spirv, &format!("dequantize_n={n}"));
        }
    }

    #[test]
    fn test_quantize_different_n_values_produce_valid_spirv() {
        for n in [1, 64, 256, 1024, 65536] {
            let spirv = generate_quantize_f32_to_int8_spirv(n);
            assert_valid_header(&spirv, &format!("quantize_n={n}"));
            assert_entry_point_main(&spirv, &format!("quantize_n={n}"));
        }
    }
}
