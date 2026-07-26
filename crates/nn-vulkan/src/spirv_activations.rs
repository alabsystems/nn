// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for activation function compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules for element-wise activation functions:
//!
//! - [`generate_gelu_spirv`]: GELU activation (tanh approximation).
//! - [`generate_silu_spirv`]: SiLU/Swish activation: `x * sigmoid(x)`.
//! - [`generate_snake_spirv`]: Snake activation: `x + (1/alpha) * sin(alpha*x)^2`.
//! - [`generate_fused_adain_snake_spirv`]: Fused AdaIN + Snake activation.
//!
//! All shaders use:
//! - Configurable workgroup size (1D dispatch)
//! - Push constants for tensor length
//! - `StorageBuffer` storage class with `std430` layout
//! - SPIR-V 1.0 for maximum Vulkan compatibility
//! - Separate input and output buffers (binding 0 = input, binding 1 = output)
//!
//! GLSL.std.450 extended instruction set is used for transcendental operations
//! (Tanh, Exp, Sin).

use crate::spirv_emit::SPIRV_MAGIC;

/// Default workgroup size for activation kernels (1D dispatch).
pub const ACTIVATION_WORKGROUP_SIZE: u32 = 256;

// ---- SPIR-V constants ----

const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;

const fn op(word_count: u16, opcode: u16) -> u32 {
    (word_count as u32) << 16 | opcode as u32
}

// Opcodes.
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
const OP_FSUB: u16 = 131;
const OP_FDIV: u16 = 136;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_IADD: u16 = 128;
const OP_IMUL: u16 = 132;
const OP_UMOD: u16 = 137;
const OP_EXT_INST: u16 = 12;

// Decorations.
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BLOCK: u32 = 2;
const DECORATION_NON_WRITABLE: u32 = 24;

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

// GLSL.std.450 extended instruction set opcodes.
const GLSL_STD_450_TANH: u32 = 21;
const GLSL_STD_450_EXP: u32 = 27;
const GLSL_STD_450_SIN: u32 = 13;
const GLSL_STD_450_SQRT: u32 = 31;

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

/// Convert a `Vec<u32>` SPIR-V module to `Vec<u8>` (little-endian).
fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// SPIR-V module builder.
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

    // ---- Type declarations ----

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

    fn u_gte(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_GREATER_THAN_EQUAL));
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

    fn imul(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_IMUL));
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

    fn build(self) -> Vec<u32> {
        let mut module = Vec::with_capacity(256);
        module.push(SPIRV_MAGIC);
        module.push(SPIRV_VERSION_1_0);
        module.push(GENERATOR_MAGIC);
        module.push(self.bound);
        module.push(0);
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

/// Set up common elementwise kernel infrastructure.
///
/// Returns (builder, func_id, glsl_ext, ty_float, ty_uint, ty_bool,
///          ptr_sb_float, var_input, var_output, gid_x, const_n, const_0u)
///
/// Layout:
/// - Binding 0 (set 0): Input buffer float[] (readonly)
/// - Binding 1 (set 0): Output buffer float[]
/// - Push constants: { uint n; }
#[allow(clippy::type_complexity)]
fn setup_elementwise_kernel(
    workgroup_size: u32,
    entry_name: &str,
) -> (
    SpirVBuilder,
    u32, // func_id
    u32, // glsl_ext
    u32, // ty_float
    u32, // ty_uint
    u32, // ty_bool
    u32, // ptr_sb_float
    u32, // var_input
    u32, // var_output
    u32, // gid_x
    u32, // const_0u
) {
    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model_decl(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // Types.
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Input buffer struct (readonly).
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    // Output buffer struct.
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n; }
    let ty_struct_pc = b.type_struct(&[ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);

    // Global variables.
    let var_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_input, DECORATION_BINDING, &[0]);
    b.decorate(var_input, DECORATION_NON_WRITABLE, &[]);

    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[1]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, entry_name, &[var_gid]);
    b.execution_mode_local_size(func_id, workgroup_size, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    // Load global invocation ID.
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load n from push constants.
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let n_val = b.load(ty_uint, pc_n_ptr);

    // Bounds check: if (gid_x >= n) return;
    let cmp_oob = b.u_gte(ty_bool, gid_x, n_val);
    let label_body = b.id();
    let label_exit = b.id();
    b.selection_merge(label_exit);
    b.branch_conditional(cmp_oob, label_exit, label_body);

    // Body label — caller emits the computation here.
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_body);

    (
        b,
        func_id,
        glsl_ext,
        ty_float,
        ty_uint,
        ty_bool,
        ptr_sb_float,
        var_input,
        var_output,
        gid_x,
        const_0u,
    )
}

/// Finish an elementwise kernel after the body has been emitted.
fn finish_elementwise_kernel(b: &mut SpirVBuilder, label_exit: u32) {
    b.branch(label_exit);

    // Exit label.
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_exit);

    b.op_return();
    b.func_end();
}

// ============================================================
// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================

/// Generate a SPIR-V 1.0 binary module for element-wise GELU activation.
///
/// Uses the tanh approximation:
///   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// # Layout
///
/// - **Binding 0** (set 0): Input buffer `float[]` (readonly).
/// - **Binding 1** (set 0): Output buffer `float[]`.
/// - **Push constants**: `{ uint n; }` — number of elements.
///
/// # Arguments
///
/// * `workgroup_size` — Number of threads per workgroup (typically 256).
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_activations::generate_gelu_spirv;
/// let spirv = generate_gelu_spirv(256);
/// assert_eq!(spirv.len() % 4, 0);
/// let words: Vec<u32> = spirv.chunks_exact(4)
///     .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
///     .collect();
/// assert_eq!(words[0], 0x07230203); // SPIR-V magic
/// ```
pub fn generate_gelu_spirv(workgroup_size: u32) -> Vec<u8> {
    let (
        mut b,
        _func_id,
        glsl_ext,
        ty_float,
        _ty_uint,
        _ty_bool,
        ptr_sb_float,
        var_input,
        var_output,
        gid_x,
        const_0u,
    ) = setup_elementwise_kernel(workgroup_size, "main");

    // Constants for GELU tanh approximation.
    let const_half = b.constant_f32(ty_float, 0.5);
    let const_one = b.constant_f32(ty_float, 1.0);
    let const_coeff = b.constant_f32(ty_float, 0.044715);
    // sqrt(2/pi) = 0.7978845608...
    let const_sqrt2pi = b.constant_f32(ty_float, 0.797_884_6_f32);

    // Load x = input[gid_x].
    let ptr_in = b.access_chain(ptr_sb_float, var_input, &[const_0u, gid_x]);
    let x = b.load(ty_float, ptr_in);

    // x3 = x * x * x
    let x2 = b.fmul(ty_float, x, x);
    let x3 = b.fmul(ty_float, x2, x);

    // inner = x + 0.044715 * x^3
    let coeff_x3 = b.fmul(ty_float, const_coeff, x3);
    let inner = b.fadd(ty_float, x, coeff_x3);

    // tanh_arg = sqrt(2/pi) * inner
    let tanh_arg = b.fmul(ty_float, const_sqrt2pi, inner);

    // tanh_val = tanh(tanh_arg)
    let tanh_val = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_TANH, &[tanh_arg]);

    // result = 0.5 * x * (1 + tanh_val)
    let one_plus_tanh = b.fadd(ty_float, const_one, tanh_val);
    let half_x = b.fmul(ty_float, const_half, x);
    let result = b.fmul(ty_float, half_x, one_plus_tanh);

    // Store output[gid_x] = result.
    let ptr_out = b.access_chain(ptr_sb_float, var_output, &[const_0u, gid_x]);
    b.store(ptr_out, result);

    // Find the exit label (it was pre-allocated in setup, need to extract it).
    // The exit label is the one right after the selection_merge — we need to look at
    // the last selection_merge's target. Since setup_elementwise_kernel creates it,
    // we can reconstruct: the label_exit was the one before the body label.
    // Actually, let's look at the builder state to find label_exit.
    // The branch_conditional was: cmp_oob -> label_exit, label_body.
    // label_body was emitted. label_exit needs to be emitted now.
    // We need to find label_exit. It's the target of branch_conditional when cmp_oob is true.
    // Let's scan backwards to find it from the OpBranchConditional.
    let label_exit = find_exit_label(&b.functions);

    finish_elementwise_kernel(&mut b, label_exit);

    let words = b.build();
    words_to_bytes(&words)
}

/// Find the exit label from the OpBranchConditional in the function body.
/// The pattern is: OpBranchConditional cond true_label false_label
/// where true_label is the exit (out-of-bounds early return).
fn find_exit_label(functions: &[u32]) -> u32 {
    let mut pos = 0;
    while pos < functions.len() {
        let word = functions[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 {
            break;
        }
        if opcode == OP_BRANCH_CONDITIONAL && word_count >= 4 {
            // OpBranchConditional %cond %true_label %false_label
            return functions[pos + 2]; // true_label = exit
        }
        pos += word_count;
    }
    panic!("could not find exit label in function body");
}

// ============================================================
// SiLU/Swish activation: x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================

/// Generate a SPIR-V 1.0 binary module for element-wise SiLU (Swish) activation.
///
/// Computes: `SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))`
///
/// # Layout
///
/// - **Binding 0** (set 0): Input buffer `float[]` (readonly).
/// - **Binding 1** (set 0): Output buffer `float[]`.
/// - **Push constants**: `{ uint n; }` — number of elements.
///
/// # Arguments
///
/// * `workgroup_size` — Number of threads per workgroup (typically 256).
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_activations::generate_silu_spirv;
/// let spirv = generate_silu_spirv(256);
/// assert_eq!(spirv.len() % 4, 0);
/// ```
pub fn generate_silu_spirv(workgroup_size: u32) -> Vec<u8> {
    let (
        mut b,
        _func_id,
        glsl_ext,
        ty_float,
        _ty_uint,
        _ty_bool,
        ptr_sb_float,
        var_input,
        var_output,
        gid_x,
        const_0u,
    ) = setup_elementwise_kernel(workgroup_size, "main");

    // Constants.
    let const_one = b.constant_f32(ty_float, 1.0);
    let const_neg_one = b.constant_f32(ty_float, -1.0);

    // Load x = input[gid_x].
    let ptr_in = b.access_chain(ptr_sb_float, var_input, &[const_0u, gid_x]);
    let x = b.load(ty_float, ptr_in);

    // neg_x = -x (we compute -1.0 * x)
    let neg_x = b.fmul(ty_float, const_neg_one, x);

    // exp_neg_x = exp(-x)
    let exp_neg_x = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_EXP, &[neg_x]);

    // denom = 1 + exp(-x)
    let denom = b.fadd(ty_float, const_one, exp_neg_x);

    // result = x / (1 + exp(-x))
    let result = b.fdiv(ty_float, x, denom);

    // Store output[gid_x] = result.
    let ptr_out = b.access_chain(ptr_sb_float, var_output, &[const_0u, gid_x]);
    b.store(ptr_out, result);

    let label_exit = find_exit_label(&b.functions);
    finish_elementwise_kernel(&mut b, label_exit);

    let words = b.build();
    words_to_bytes(&words)
}

// ============================================================
// Snake activation: x + (1/alpha) * sin(alpha * x)^2
// ============================================================

/// Generate a SPIR-V 1.0 binary module for element-wise Snake activation.
///
/// Computes: `Snake(x, alpha) = x + (1/alpha) * sin(alpha * x)^2`
///
/// The alpha parameter is baked into the SPIR-V as push constants.
///
/// # Layout
///
/// - **Binding 0** (set 0): Input buffer `float[]` (readonly).
/// - **Binding 1** (set 0): Output buffer `float[]`.
/// - **Binding 2** (set 0): Alpha buffer `float[]` (readonly, per-channel).
/// - **Push constants**: `{ uint n; uint channels; }` — total elements and channel count.
///
/// Each element at index `i` uses `alpha[i % channels]`.
///
/// # Arguments
///
/// * `workgroup_size` — Number of threads per workgroup (typically 256).
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_activations::generate_snake_spirv;
/// let spirv = generate_snake_spirv(256);
/// assert_eq!(spirv.len() % 4, 0);
/// ```
pub fn generate_snake_spirv(workgroup_size: u32) -> Vec<u8> {
    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model_decl(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // Types.
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Input buffer struct (readonly).
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    // Output buffer struct.
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Alpha buffer struct (readonly).
    let ty_struct_alpha = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_alpha, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_alpha, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n; uint channels; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_sb_alpha = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_alpha);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_one_f = b.constant_f32(ty_float, 1.0);

    // Global variables.
    let var_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_input, DECORATION_BINDING, &[0]);
    b.decorate(var_input, DECORATION_NON_WRITABLE, &[]);

    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[1]);

    let var_alpha = b.variable_global(ptr_sb_alpha, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_alpha, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_alpha, DECORATION_BINDING, &[2]);
    b.decorate(var_alpha, DECORATION_NON_WRITABLE, &[]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, workgroup_size, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    // Load global invocation ID.
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load n and channels from push constants.
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let n_val = b.load(ty_uint, pc_n_ptr);
    let pc_ch_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let channels = b.load(ty_uint, pc_ch_ptr);

    // Bounds check.
    let cmp_oob = b.u_gte(ty_bool, gid_x, n_val);
    let label_body = b.id();
    let label_exit = b.id();
    b.selection_merge(label_exit);
    b.branch_conditional(cmp_oob, label_exit, label_body);

    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_body);

    // Load x = input[gid_x].
    let ptr_in = b.access_chain(ptr_sb_float, var_input, &[const_0u, gid_x]);
    let x = b.load(ty_float, ptr_in);

    // channel_idx = gid_x % channels.
    let ch_idx = b.umod(ty_uint, gid_x, channels);

    // Load alpha = alpha_buf[channel_idx].
    let ptr_alpha = b.access_chain(ptr_sb_float, var_alpha, &[const_0u, ch_idx]);
    let alpha = b.load(ty_float, ptr_alpha);

    // inv_alpha = 1.0 / alpha
    let inv_alpha = b.fdiv(ty_float, const_one_f, alpha);

    // alpha_x = alpha * x
    let alpha_x = b.fmul(ty_float, alpha, x);

    // sin_val = sin(alpha * x)
    let sin_val = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_SIN, &[alpha_x]);

    // sin2 = sin_val * sin_val
    let sin2 = b.fmul(ty_float, sin_val, sin_val);

    // snake = x + inv_alpha * sin2
    let scaled = b.fmul(ty_float, inv_alpha, sin2);
    let result = b.fadd(ty_float, x, scaled);

    // Store output[gid_x] = result.
    let ptr_out = b.access_chain(ptr_sb_float, var_output, &[const_0u, gid_x]);
    b.store(ptr_out, result);

    finish_elementwise_kernel(&mut b, label_exit);

    let words = b.build();
    words_to_bytes(&words)
}

// ============================================================
// Fused AdaIN + Snake activation
// ============================================================

/// Generate a SPIR-V 1.0 binary module for fused AdaIN + Snake activation.
///
/// Computes for each element at position `(batch, channel, time)`:
///   1. AdaIN: `y = scale[ch] * (x - mean[ch]) / sqrt(var[ch] + eps) + bias[ch]`
///   2. Snake: `z = y + (1/alpha[ch]) * sin(alpha[ch] * y)^2`
///
/// # Layout
///
/// - **Binding 0** (set 0): Input buffer `float[n]` (readonly).
/// - **Binding 1** (set 0): Output buffer `float[n]`.
/// - **Binding 2** (set 0): Params buffer `float[channels * 5]` (readonly).
///   Layout: `[mean[C], var[C], scale[C], bias[C], alpha[C]]` concatenated.
/// - **Push constants**: `{ uint n; uint channels; }`.
///
/// # Arguments
///
/// * `workgroup_size` — Number of threads per workgroup.
/// * `channels` — Number of channels (hint; actual from push constants).
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_activations::generate_fused_adain_snake_spirv;
/// let spirv = generate_fused_adain_snake_spirv(256, 64);
/// assert_eq!(spirv.len() % 4, 0);
/// ```
pub fn generate_fused_adain_snake_spirv(workgroup_size: u32, _channels: u32) -> Vec<u8> {
    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model_decl(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // Types.
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Input buffer struct (readonly).
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    // Output buffer struct.
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Params buffer struct (readonly): [mean, var, scale, bias, alpha] interleaved by channel.
    let ty_struct_params = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_params, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_params, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n; uint channels; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_sb_params = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_params);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_2u = b.constant_u32(ty_uint, 2);
    let const_3u = b.constant_u32(ty_uint, 3);
    let const_4u = b.constant_u32(ty_uint, 4);
    let const_5u = b.constant_u32(ty_uint, 5);
    let const_one_f = b.constant_f32(ty_float, 1.0);
    let const_eps = b.constant_f32(ty_float, 1e-5);

    // Global variables.
    let var_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_input, DECORATION_BINDING, &[0]);
    b.decorate(var_input, DECORATION_NON_WRITABLE, &[]);

    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[1]);

    let var_params = b.variable_global(ptr_sb_params, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_params, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_params, DECORATION_BINDING, &[2]);
    b.decorate(var_params, DECORATION_NON_WRITABLE, &[]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, workgroup_size, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    // Load global invocation ID.
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load n and channels from push constants.
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let n_val = b.load(ty_uint, pc_n_ptr);
    let pc_ch_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let channels_val = b.load(ty_uint, pc_ch_ptr);

    // Bounds check.
    let cmp_oob = b.u_gte(ty_bool, gid_x, n_val);
    let label_body = b.id();
    let label_exit = b.id();
    b.selection_merge(label_exit);
    b.branch_conditional(cmp_oob, label_exit, label_body);

    b.functions.push(op(2, OP_LABEL));
    b.functions.push(label_body);

    // Load x = input[gid_x].
    let ptr_in = b.access_chain(ptr_sb_float, var_input, &[const_0u, gid_x]);
    let x = b.load(ty_float, ptr_in);

    // channel_idx = gid_x % channels.
    let ch_idx = b.umod(ty_uint, gid_x, channels_val);

    // Params layout: [mean_0..mean_C-1, var_0..var_C-1, scale_0..scale_C-1, bias_0..bias_C-1, alpha_0..alpha_C-1]
    // mean offset = ch_idx
    // var offset = channels + ch_idx
    // scale offset = 2*channels + ch_idx
    // bias offset = 3*channels + ch_idx
    // alpha offset = 4*channels + ch_idx

    // mean_idx = 0 * channels + ch_idx = ch_idx
    let ptr_mean = b.access_chain(ptr_sb_float, var_params, &[const_0u, ch_idx]);
    let mean_val = b.load(ty_float, ptr_mean);

    // var_idx = 1 * channels + ch_idx
    let var_offset = b.imul(ty_uint, const_1u, channels_val);
    let var_idx = b.iadd(ty_uint, var_offset, ch_idx);
    let ptr_var = b.access_chain(ptr_sb_float, var_params, &[const_0u, var_idx]);
    let var_val = b.load(ty_float, ptr_var);

    // scale_idx = 2 * channels + ch_idx
    let scale_offset = b.imul(ty_uint, const_2u, channels_val);
    let scale_idx = b.iadd(ty_uint, scale_offset, ch_idx);
    let ptr_scale = b.access_chain(ptr_sb_float, var_params, &[const_0u, scale_idx]);
    let scale_val = b.load(ty_float, ptr_scale);

    // bias_idx = 3 * channels + ch_idx
    let bias_offset = b.imul(ty_uint, const_3u, channels_val);
    let bias_idx = b.iadd(ty_uint, bias_offset, ch_idx);
    let ptr_bias = b.access_chain(ptr_sb_float, var_params, &[const_0u, bias_idx]);
    let bias_val = b.load(ty_float, ptr_bias);

    // alpha_idx = 4 * channels + ch_idx
    let alpha_offset = b.imul(ty_uint, const_4u, channels_val);
    let alpha_idx = b.iadd(ty_uint, alpha_offset, ch_idx);
    let ptr_alpha = b.access_chain(ptr_sb_float, var_params, &[const_0u, alpha_idx]);
    let alpha_val = b.load(ty_float, ptr_alpha);

    // ---- AdaIN: y = scale * (x - mean) / sqrt(var + eps) + bias ----
    let x_minus_mean = b.fsub(ty_float, x, mean_val);
    let var_plus_eps = b.fadd(ty_float, var_val, const_eps);
    let sqrt_var = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_SQRT, &[var_plus_eps]);
    let normalized = b.fdiv(ty_float, x_minus_mean, sqrt_var);
    let scaled = b.fmul(ty_float, scale_val, normalized);
    let y = b.fadd(ty_float, scaled, bias_val);

    // ---- Snake: z = y + (1/alpha) * sin(alpha * y)^2 ----
    let inv_alpha = b.fdiv(ty_float, const_one_f, alpha_val);
    let alpha_y = b.fmul(ty_float, alpha_val, y);
    let sin_val = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_SIN, &[alpha_y]);
    let sin2 = b.fmul(ty_float, sin_val, sin_val);
    let snake_term = b.fmul(ty_float, inv_alpha, sin2);
    let z = b.fadd(ty_float, y, snake_term);

    // Store output[gid_x] = z.
    let ptr_out = b.access_chain(ptr_sb_float, var_output, &[const_0u, gid_x]);
    b.store(ptr_out, z);

    finish_elementwise_kernel(&mut b, label_exit);

    // Suppress unused warnings for constants allocated during setup.
    let _ = (const_5u,);

    let words = b.build();
    words_to_bytes(&words)
}

// ============================================================
// Reference CPU implementations
// ============================================================

/// Reference GELU activation on CPU (tanh approximation).
///
/// `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
pub fn gelu_reference(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = x + 0.044715 * x * x * x;
    0.5 * x * (1.0 + (sqrt_2_over_pi * inner).tanh())
}

/// Reference SiLU/Swish activation on CPU.
///
/// `SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))`
pub fn silu_reference(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Reference Snake activation on CPU.
///
/// `Snake(x, alpha) = x + (1/alpha) * sin(alpha * x)^2`
pub fn snake_reference(x: f32, alpha: f32) -> f32 {
    let sin_val = (alpha * x).sin();
    x + (1.0 / alpha) * sin_val * sin_val
}

/// Reference fused AdaIN + Snake activation on CPU.
///
/// 1. AdaIN: `y = scale * (x - mean) / sqrt(var + eps) + bias`
/// 2. Snake: `z = y + (1/alpha) * sin(alpha * y)^2`
pub fn fused_adain_snake_reference(
    x: f32,
    mean: f32,
    var: f32,
    scale: f32,
    bias: f32,
    alpha: f32,
) -> f32 {
    let eps = 1e-5_f32;
    let y = scale * (x - mean) / (var + eps).sqrt() + bias;
    snake_reference(y, alpha)
}

#[cfg(test)]
#[path = "spirv_activations_tests.rs"]
mod spirv_activations_tests;
