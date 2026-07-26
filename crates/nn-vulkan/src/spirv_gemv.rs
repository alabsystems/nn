// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for GEMV, dot product, and outer product compute shaders.
//!
//! Three operations are provided:
//!
//! - [`generate_gemv_spirv`]: Matrix-vector multiply y = A @ x where A is [M, N] and x is [N].
//! - [`generate_dot_spirv`]: Dot product of two vectors, producing a scalar.
//! - [`generate_outer_spirv`]: Outer product of two vectors a[M] and b[N], producing C[M, N].
//!
//! All variants use:
//! - Storage buffers with `std430` layout
//! - Push constants for dimensions
//! - Bounds checking for non-power-of-2 sizes
//! - SPIR-V 1.0 for maximum Vulkan compatibility

use crate::spirv_emit::SPIRV_MAGIC;

/// Default workgroup size for GEMV and vector operations.
pub const GEMV_WORKGROUP_SIZE: u32 = 256;

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

/// Generate a SPIR-V 1.0 binary for GEMV: y = A @ x.
///
/// Each thread computes one output element: `y[row] = sum_k(A[row, k] * x[k])`.
///
/// # Arguments
///
/// * `m` - Number of rows in A (compile-time hint; actual from push constants).
/// * `n` - Number of columns in A / length of x (compile-time hint; actual from push constants).
///
/// # Buffers
///
/// - Binding 0: A `[M, N]` (row-major) -- `float[]`
/// - Binding 1: x `[N]` -- `float[]`
/// - Binding 2: y `[M]` -- `float[]` (output)
///
/// # Push constants
///
/// - `uint m` at offset 0
/// - `uint n` at offset 4
///
/// # Returns
///
/// SPIR-V binary as `Vec<u32>`.
pub fn generate_gemv_spirv(m: u32, n: u32) -> Vec<u32> {
    let _ = m;
    let _ = n;

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

    // Buffer structs.
    let ty_struct_a = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_a, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_a, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_x = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_x, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_x, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_y = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_y, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_y, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint m, uint n }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);

    // Pointer types.
    let ptr_sb_a = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_a);
    let ptr_sb_x = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_x);
    let ptr_sb_y = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_y);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_f0 = b.constant_f32(ty_float, 0.0);

    // Global variables.
    let var_a = b.variable_global(ptr_sb_a, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_a, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_a, DECORATION_BINDING, &[0]);

    let var_x = b.variable_global(ptr_sb_x, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_x, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_x, DECORATION_BINDING, &[1]);

    let var_y = b.variable_global(ptr_sb_y, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_y, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_y, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, GEMV_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    // Load gl_GlobalInvocationID.x
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load push constants: m, n.
    let pc_m_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let pc_m = b.load(ty_uint, pc_m_ptr);
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let pc_n = b.load(ty_uint, pc_n_ptr);

    // Bounds check: gid >= m -> return.
    let cmp_oob = b.u_greater_than_equal(ty_bool, gid, pc_m);
    let return_label = b.id();
    let body_label = b.id();
    b.selection_merge(body_label);
    b.branch_conditional(cmp_oob, return_label, body_label);

    b.label_with_id(return_label);
    b.op_return();

    b.label_with_id(body_label);

    // row_offset = gid * n
    let row_offset = b.imul(ty_uint, gid, pc_n);

    // ---- Dot product loop: acc = sum_k(A[gid*n + k] * x[k]) ----
    let loop_header = b.id();
    let loop_body = b.id();
    let loop_continue = b.id();
    let loop_merge = b.id();

    b.branch(loop_header);

    b.label_with_id(loop_header);
    b.loop_merge(loop_merge, loop_continue);
    let phi_k = b.phi(ty_uint, &[(const_0u, body_label)]);
    let phi_acc = b.phi(ty_float, &[(const_f0, body_label)]);
    let cmp_k = b.u_less_than(ty_bool, phi_k, pc_n);
    b.branch_conditional(cmp_k, loop_body, loop_merge);

    // Loop body: acc += A[row_offset + k] * x[k]
    b.label_with_id(loop_body);

    let a_idx = b.iadd(ty_uint, row_offset, phi_k);
    let a_ptr = b.access_chain(ptr_sb_float, var_a, &[const_0u, a_idx]);
    let a_val = b.load(ty_float, a_ptr);

    let x_ptr = b.access_chain(ptr_sb_float, var_x, &[const_0u, phi_k]);
    let x_val = b.load(ty_float, x_ptr);

    let prod = b.fmul(ty_float, a_val, x_val);
    let new_acc = b.fadd(ty_float, phi_acc, prod);

    b.branch(loop_continue);
    b.label_with_id(loop_continue);
    let next_k = b.iadd(ty_uint, phi_k, const_1u);
    fixup_phi(&mut b.functions, phi_k, next_k, loop_continue);
    fixup_phi(&mut b.functions, phi_acc, new_acc, loop_continue);
    b.branch(loop_header);

    b.label_with_id(loop_merge);

    // Store y[gid] = acc.
    let y_ptr = b.access_chain(ptr_sb_float, var_y, &[const_0u, gid]);
    b.store(y_ptr, phi_acc);

    b.op_return();
    b.func_end();

    b.build()
}

/// CPU reference implementation of GEMV: y = A @ x.
///
/// # Arguments
///
/// * `a` - Matrix A, flattened as `[M * N]` (row-major).
/// * `x` - Vector x of length N.
/// * `m` - Number of rows in A.
/// * `n` - Number of columns in A.
///
/// # Returns
///
/// Output vector y of length M.
pub fn gemv_reference(a: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * n, "A length must be m * n");
    assert_eq!(x.len(), n, "x length must be n");

    let mut y = vec![0.0f32; m];
    for row in 0..m {
        let mut acc = 0.0f32;
        for k in 0..n {
            acc += a[row * n + k] * x[k];
        }
        y[row] = acc;
    }
    y
}

/// Generate a SPIR-V 1.0 binary for dot product: result = sum_k(a[k] * b[k]).
///
/// Uses a parallel reduction pattern: each thread computes a partial sum over
/// a strided range, then writes it to the output. The output buffer has one
/// element per workgroup; the host must sum them for the final scalar.
///
/// # Arguments
///
/// * `n` - Length of vectors (compile-time hint; actual from push constants).
///
/// # Buffers
///
/// - Binding 0: a `[N]` -- `float[]`
/// - Binding 1: b `[N]` -- `float[]`
/// - Binding 2: output `[num_workgroups]` -- `float[]` (partial sums)
///
/// # Push constants
///
/// - `uint n` at offset 0
///
/// # Returns
///
/// SPIR-V binary as `Vec<u32>`.
pub fn generate_dot_spirv(n: u32) -> Vec<u32> {
    let _ = n;

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

    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Buffer structs.
    let ty_struct_a = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_a, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_a, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_b = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_b, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_b, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_out = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_out, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_out, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint n }
    let ty_struct_pc = b.type_struct(&[ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);

    // Pointer types.
    let ptr_sb_a = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_a);
    let ptr_sb_b = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_b);
    let ptr_sb_out = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_out);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);

    // Global variables.
    let var_a = b.variable_global(ptr_sb_a, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_a, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_a, DECORATION_BINDING, &[0]);

    let var_b = b.variable_global(ptr_sb_b, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_b, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_b, DECORATION_BINDING, &[1]);

    let var_out = b.variable_global(ptr_sb_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_out, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, GEMV_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    // Load gl_GlobalInvocationID.x
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load push constants: n.
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let pc_n = b.load(ty_uint, pc_n_ptr);

    // Bounds check: gid >= n -> return.
    let cmp_oob = b.u_greater_than_equal(ty_bool, gid, pc_n);
    let return_label = b.id();
    let body_label = b.id();
    b.selection_merge(body_label);
    b.branch_conditional(cmp_oob, return_label, body_label);

    b.label_with_id(return_label);
    b.op_return();

    b.label_with_id(body_label);

    // Each thread computes a[gid] * b[gid] and stores to output[gid].
    // Host-side reduction sums the output buffer for the final dot product.
    let a_ptr = b.access_chain(ptr_sb_float, var_a, &[const_0u, gid]);
    let a_val = b.load(ty_float, a_ptr);

    let b_ptr = b.access_chain(ptr_sb_float, var_b, &[const_0u, gid]);
    let b_val = b.load(ty_float, b_ptr);

    let prod = b.fmul(ty_float, a_val, b_val);

    let out_ptr = b.access_chain(ptr_sb_float, var_out, &[const_0u, gid]);
    b.store(out_ptr, prod);

    b.op_return();
    b.func_end();

    b.build()
}

/// CPU reference implementation of dot product.
///
/// # Arguments
///
/// * `a` - First vector.
/// * `b` - Second vector (must be same length as `a`).
///
/// # Returns
///
/// Scalar dot product.
pub fn dot_reference(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Generate a SPIR-V 1.0 binary for outer product: C[i, j] = a[i] * b[j].
///
/// Each thread computes one element of the output matrix C.
///
/// # Arguments
///
/// * `m` - Length of vector a (compile-time hint; actual from push constants).
/// * `n` - Length of vector b (compile-time hint; actual from push constants).
///
/// # Buffers
///
/// - Binding 0: a `[M]` -- `float[]`
/// - Binding 1: b `[N]` -- `float[]`
/// - Binding 2: C `[M * N]` -- `float[]` (output, row-major)
///
/// # Push constants
///
/// - `uint m` at offset 0
/// - `uint n` at offset 4
///
/// # Returns
///
/// SPIR-V binary as `Vec<u32>`.
pub fn generate_outer_spirv(m: u32, n: u32) -> Vec<u32> {
    let _ = m;
    let _ = n;

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

    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Buffer structs.
    let ty_struct_a = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_a, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_a, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_b = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_b, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_b, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_c = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_c, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_c, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint m, uint n }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);

    // Pointer types.
    let ptr_sb_a = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_a);
    let ptr_sb_b = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_b);
    let ptr_sb_c = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_c);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // We need UDiv and UMod for index computation.
    const OP_UDIV: u16 = 134;
    const OP_UMOD: u16 = 137;

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);

    // Global variables.
    let var_a = b.variable_global(ptr_sb_a, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_a, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_a, DECORATION_BINDING, &[0]);

    let var_b = b.variable_global(ptr_sb_b, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_b, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_b, DECORATION_BINDING, &[1]);

    let var_c = b.variable_global(ptr_sb_c, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_c, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_c, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, GEMV_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    // Load gl_GlobalInvocationID.x
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load push constants: m, n.
    let pc_m_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let pc_m = b.load(ty_uint, pc_m_ptr);
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let pc_n = b.load(ty_uint, pc_n_ptr);

    // total = m * n
    let total = b.imul(ty_uint, pc_m, pc_n);

    // Bounds check: gid >= total -> return.
    let cmp_oob = b.u_greater_than_equal(ty_bool, gid, total);
    let return_label = b.id();
    let body_label = b.id();
    b.selection_merge(body_label);
    b.branch_conditional(cmp_oob, return_label, body_label);

    b.label_with_id(return_label);
    b.op_return();

    b.label_with_id(body_label);

    // row = gid / n, col = gid % n
    // Inline UDiv and UMod since they're not in the builder.
    let row = {
        let result = b.id();
        b.functions.push(op(5, OP_UDIV));
        b.functions.push(ty_uint);
        b.functions.push(result);
        b.functions.push(gid);
        b.functions.push(pc_n);
        result
    };
    let col = {
        let result = b.id();
        b.functions.push(op(5, OP_UMOD));
        b.functions.push(ty_uint);
        b.functions.push(result);
        b.functions.push(gid);
        b.functions.push(pc_n);
        result
    };

    // C[gid] = a[row] * b[col]
    let a_ptr = b.access_chain(ptr_sb_float, var_a, &[const_0u, row]);
    let a_val = b.load(ty_float, a_ptr);

    let b_ptr = b.access_chain(ptr_sb_float, var_b, &[const_0u, col]);
    let b_val = b.load(ty_float, b_ptr);

    let prod = b.fmul(ty_float, a_val, b_val);

    let c_ptr = b.access_chain(ptr_sb_float, var_c, &[const_0u, gid]);
    b.store(c_ptr, prod);

    b.op_return();
    b.func_end();

    b.build()
}

/// CPU reference implementation of outer product: C[i, j] = a[i] * b[j].
///
/// # Arguments
///
/// * `a` - First vector of length M.
/// * `b` - Second vector of length N.
///
/// # Returns
///
/// Output matrix C, flattened as `[M * N]` (row-major).
pub fn outer_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    let m = a.len();
    let n = b.len();
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = a[i] * b[j];
        }
    }
    c
}

#[cfg(test)]
#[path = "spirv_gemv_tests.rs"]
mod spirv_gemv_tests;
