// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for dtype cast operations (F32/F16/BF16).
//!
//! Generates SPIR-V 1.0 binary modules for floating-point type conversions:
//!
//! - [`generate_f32_to_f16_spirv`]: F32 to F16 via OpFConvert (Float16 capability)
//! - [`generate_f16_to_f32_spirv`]: F16 to F32 via OpFConvert (Float16 capability)
//! - [`generate_f32_to_bf16_spirv`]: F32 to BF16 via bitwise truncation (uint16 storage)
//! - [`generate_bf16_to_f32_spirv`]: BF16 to F32 via bitwise shift (uint16 storage)
//!
//! All shaders use:
//! - Workgroup size of 256 threads (1D dispatch)
//! - Push constants for element count
//! - `StorageBuffer` storage class with `std430` layout
//!
//! BF16 does not have native SPIR-V support. It is emulated using uint16 storage
//! with bitwise conversion: BF16 is the upper 16 bits of an IEEE 754 float32.
//! F32 to BF16: reinterpret as uint32, shift right by 16, store as uint16.
//! BF16 to F32: load uint16, shift left by 16, reinterpret as float32.

use crate::spirv_emit::SPIRV_MAGIC;

/// Workgroup size for cast operations.
pub const CAST_WORKGROUP_SIZE: u32 = 256;

// ---- SPIR-V constants (duplicated to keep modules independent) ----

const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;

const fn op(word_count: u16, opcode: u16) -> u32 {
    (word_count as u32) << 16 | opcode as u32
}

// Opcodes.
const OP_CAPABILITY: u16 = 17;
const OP_EXTENSION: u16 = 10;
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
const OP_U_LESS_THAN: u16 = 176;
const OP_FCONVERT: u16 = 115;
const OP_BITCAST: u16 = 124;
const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
const OP_UCONVERT: u16 = 113;

// Capabilities.
const CAPABILITY_SHADER: u32 = 1;
const CAPABILITY_FLOAT16: u32 = 9;
const CAPABILITY_INT16: u32 = 22;
const CAPABILITY_STORAGE_BUFFER_16BIT: u32 = 4433;

// Decorations.
const DECORATION_BLOCK: u32 = 2;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_NON_WRITABLE: u32 = 24;

// Builtins.
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// Storage classes.
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// Execution model and mode.
const EXECUTION_MODEL_GLCOMPUTE: u32 = 5;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

// Addressing / memory model.
const ADDRESSING_MODEL_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;

// Function control.
const FUNCTION_CONTROL_NONE: u32 = 0;

// ---- SpirVBuilder ----

struct SpirVBuilder {
    next_id: u32,
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
            next_id: 1,
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

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn capability(&mut self, cap: u32) {
        self.capabilities.push(op(2, OP_CAPABILITY));
        self.capabilities.push(cap);
    }

    fn extension(&mut self, name: &str) {
        let name_words = string_to_words(name);
        let wc = 1 + name_words.len() as u16;
        self.extensions.push(op(wc, OP_EXTENSION));
        self.extensions.extend_from_slice(&name_words);
    }

    fn memory_model(&mut self, addressing: u32, model: u32) {
        self.memory_model.push(op(3, OP_MEMORY_MODEL));
        self.memory_model.push(addressing);
        self.memory_model.push(model);
    }

    fn entry_point(&mut self, model: u32, func_id: u32, name: &str, interfaces: &[u32]) {
        let name_words = string_to_words(name);
        let wc = (3 + name_words.len() + interfaces.len()) as u16;
        self.entry_points.push(op(wc, OP_ENTRY_POINT));
        self.entry_points.push(model);
        self.entry_points.push(func_id);
        self.entry_points.extend_from_slice(&name_words);
        self.entry_points.extend_from_slice(interfaces);
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
        let wc = (3 + operands.len()) as u16;
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
        let wc = (4 + operands.len()) as u16;
        self.annotations.push(op(wc, OP_MEMBER_DECORATE));
        self.annotations.push(struct_type);
        self.annotations.push(member);
        self.annotations.push(decoration);
        self.annotations.extend_from_slice(operands);
    }

    fn type_void(&mut self) -> u32 {
        let id = self.alloc_id();
        self.type_declarations.push(op(2, OP_TYPE_VOID));
        self.type_declarations.push(id);
        id
    }

    fn type_int(&mut self, width: u32, signedness: u32) -> u32 {
        let id = self.alloc_id();
        self.type_declarations.push(op(4, OP_TYPE_INT));
        self.type_declarations.push(id);
        self.type_declarations.push(width);
        self.type_declarations.push(signedness);
        id
    }

    fn type_float(&mut self, width: u32) -> u32 {
        let id = self.alloc_id();
        self.type_declarations.push(op(3, OP_TYPE_FLOAT));
        self.type_declarations.push(id);
        self.type_declarations.push(width);
        id
    }

    fn type_vector(&mut self, component: u32, count: u32) -> u32 {
        let id = self.alloc_id();
        self.type_declarations.push(op(4, OP_TYPE_VECTOR));
        self.type_declarations.push(id);
        self.type_declarations.push(component);
        self.type_declarations.push(count);
        id
    }

    fn type_runtime_array(&mut self, element: u32) -> u32 {
        let id = self.alloc_id();
        self.type_declarations.push(op(3, OP_TYPE_RUNTIME_ARRAY));
        self.type_declarations.push(id);
        self.type_declarations.push(element);
        id
    }

    fn type_struct(&mut self, members: &[u32]) -> u32 {
        let id = self.alloc_id();
        let wc = (2 + members.len()) as u16;
        self.type_declarations.push(op(wc, OP_TYPE_STRUCT));
        self.type_declarations.push(id);
        self.type_declarations.extend_from_slice(members);
        id
    }

    fn type_pointer(&mut self, storage_class: u32, pointee: u32) -> u32 {
        let id = self.alloc_id();
        self.type_declarations.push(op(4, OP_TYPE_POINTER));
        self.type_declarations.push(id);
        self.type_declarations.push(storage_class);
        self.type_declarations.push(pointee);
        id
    }

    fn type_function(&mut self, return_type: u32, params: &[u32]) -> u32 {
        let id = self.alloc_id();
        let wc = (3 + params.len()) as u16;
        self.type_declarations.push(op(wc, OP_TYPE_FUNCTION));
        self.type_declarations.push(id);
        self.type_declarations.push(return_type);
        self.type_declarations.extend_from_slice(params);
        id
    }

    fn constant_u32(&mut self, ty: u32, value: u32) -> u32 {
        let id = self.alloc_id();
        self.type_declarations.push(op(4, OP_CONSTANT));
        self.type_declarations.push(ty);
        self.type_declarations.push(id);
        self.type_declarations.push(value);
        id
    }

    fn variable(&mut self, ptr_type: u32, storage_class: u32) -> u32 {
        let id = self.alloc_id();
        self.type_declarations.push(op(4, OP_VARIABLE));
        self.type_declarations.push(ptr_type);
        self.type_declarations.push(id);
        self.type_declarations.push(storage_class);
        id
    }

    // ---- Function body instructions ----

    fn func_begin(&mut self, result_type: u32, func_type: u32) -> u32 {
        let id = self.alloc_id();
        self.functions.push(op(5, OP_FUNCTION));
        self.functions.push(result_type);
        self.functions.push(id);
        self.functions.push(FUNCTION_CONTROL_NONE);
        self.functions.push(func_type);
        id
    }

    fn func_end(&mut self) {
        self.functions.push(op(1, OP_FUNCTION_END));
    }

    fn label(&mut self) -> u32 {
        let id = self.alloc_id();
        self.functions.push(op(2, OP_LABEL));
        self.functions.push(id);
        id
    }

    fn ret(&mut self) {
        self.functions.push(op(1, OP_RETURN));
    }

    fn branch(&mut self, target: u32) {
        self.functions.push(op(2, OP_BRANCH));
        self.functions.push(target);
    }

    fn branch_conditional(&mut self, cond: u32, true_label: u32, false_label: u32) {
        self.functions.push(op(4, OP_BRANCH_CONDITIONAL));
        self.functions.push(cond);
        self.functions.push(true_label);
        self.functions.push(false_label);
    }

    fn selection_merge(&mut self, merge_label: u32) {
        self.functions.push(op(3, OP_SELECTION_MERGE));
        self.functions.push(merge_label);
        self.functions.push(0); // None selection control
    }

    fn load(&mut self, result_type: u32, pointer: u32) -> u32 {
        let id = self.alloc_id();
        self.functions.push(op(4, OP_LOAD));
        self.functions.push(result_type);
        self.functions.push(id);
        self.functions.push(pointer);
        id
    }

    fn store(&mut self, pointer: u32, value: u32) {
        self.functions.push(op(3, OP_STORE));
        self.functions.push(pointer);
        self.functions.push(value);
    }

    fn access_chain(&mut self, result_type: u32, base: u32, indices: &[u32]) -> u32 {
        let id = self.alloc_id();
        let wc = (4 + indices.len()) as u16;
        self.functions.push(op(wc, OP_ACCESS_CHAIN));
        self.functions.push(result_type);
        self.functions.push(id);
        self.functions.push(base);
        self.functions.extend_from_slice(indices);
        id
    }

    fn composite_extract(&mut self, result_type: u32, composite: u32, index: u32) -> u32 {
        let id = self.alloc_id();
        self.functions.push(op(5, OP_COMPOSITE_EXTRACT));
        self.functions.push(result_type);
        self.functions.push(id);
        self.functions.push(composite);
        self.functions.push(index);
        id
    }

    fn u_less_than(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let id = self.alloc_id();
        self.functions.push(op(5, OP_U_LESS_THAN));
        self.functions.push(result_type);
        self.functions.push(id);
        self.functions.push(a);
        self.functions.push(b);
        id
    }

    fn fconvert(&mut self, result_type: u32, value: u32) -> u32 {
        let id = self.alloc_id();
        self.functions.push(op(4, OP_FCONVERT));
        self.functions.push(result_type);
        self.functions.push(id);
        self.functions.push(value);
        id
    }

    fn bitcast(&mut self, result_type: u32, value: u32) -> u32 {
        let id = self.alloc_id();
        self.functions.push(op(4, OP_BITCAST));
        self.functions.push(result_type);
        self.functions.push(id);
        self.functions.push(value);
        id
    }

    fn shift_right_logical(&mut self, result_type: u32, base: u32, shift: u32) -> u32 {
        let id = self.alloc_id();
        self.functions.push(op(5, OP_SHIFT_RIGHT_LOGICAL));
        self.functions.push(result_type);
        self.functions.push(id);
        self.functions.push(base);
        self.functions.push(shift);
        id
    }

    fn shift_left_logical(&mut self, result_type: u32, base: u32, shift: u32) -> u32 {
        let id = self.alloc_id();
        self.functions.push(op(5, OP_SHIFT_LEFT_LOGICAL));
        self.functions.push(result_type);
        self.functions.push(id);
        self.functions.push(base);
        self.functions.push(shift);
        id
    }

    fn uconvert(&mut self, result_type: u32, value: u32) -> u32 {
        let id = self.alloc_id();
        self.functions.push(op(4, OP_UCONVERT));
        self.functions.push(result_type);
        self.functions.push(id);
        self.functions.push(value);
        id
    }

    // ---- Build ----

    fn build(self) -> Vec<u32> {
        let body_len = self.capabilities.len()
            + self.extensions.len()
            + self.memory_model.len()
            + self.entry_points.len()
            + self.execution_modes.len()
            + self.annotations.len()
            + self.type_declarations.len()
            + self.functions.len();
        let mut words = Vec::with_capacity(5 + body_len);
        words.push(SPIRV_MAGIC);
        words.push(SPIRV_VERSION_1_0);
        words.push(GENERATOR_MAGIC);
        words.push(self.next_id); // bound
        words.push(0); // reserved
        words.extend_from_slice(&self.capabilities);
        words.extend_from_slice(&self.extensions);
        words.extend_from_slice(&self.memory_model);
        words.extend_from_slice(&self.entry_points);
        words.extend_from_slice(&self.execution_modes);
        words.extend_from_slice(&self.annotations);
        words.extend_from_slice(&self.type_declarations);
        words.extend_from_slice(&self.functions);
        words
    }
}

// ---- String helper ----

fn string_to_words(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    let padded_len = bytes.len() + 1; // null terminator
    let word_count = padded_len.div_ceil(4);
    let mut words = vec![0u32; word_count];
    for (i, &b) in bytes.iter().enumerate() {
        let word_idx = i / 4;
        let byte_idx = i % 4;
        words[word_idx] |= u32::from(b) << (byte_idx * 8);
    }
    words
}

// ---- Common setup for cast shaders ----

/// Setup information returned by `setup_cast`.
///
/// Some fields are kept for completeness even though not all callers use them.
/// They hold SPIR-V IDs that were allocated during module construction.
#[allow(dead_code)]
struct CastSetup {
    b: SpirVBuilder,
    func_id: u32,
    gid: u32,
    n_val: u32,
    in_bounds: u32,
    merge_label: u32,
    body_label: u32,
    // Types needed by callers.
    uint_type: u32,
    bool_type: u32,
    // Input/output buffer element pointer types and base variables.
    in_elem_ptr_type: u32,
    out_elem_ptr_type: u32,
    in_buf_var: u32,
    out_buf_var: u32,
}

/// Set up a cast shader skeleton.
///
/// `make_in_out_types` receives `&mut SpirVBuilder` and `uint_type`, and returns
/// `(input_element_type, output_element_type)`. The caller defines whatever
/// extra float/int types it needs for input and output.
fn setup_cast<F>(make_in_out_types: F) -> CastSetup
where
    F: FnOnce(&mut SpirVBuilder, u32) -> (u32, u32),
{
    let mut b = SpirVBuilder::new();

    // Capabilities (Shader is always needed; callers add Float16/Int16 after).
    b.capability(CAPABILITY_SHADER);

    // Memory model.
    b.memory_model(ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450);

    // Basic types.
    let void_type = b.type_void();
    let uint_type = b.type_int(32, 0);
    let uvec3_type = b.type_vector(uint_type, 3);
    // OpULessThan returns OpTypeBool — we need a proper bool type.
    let bool_type_id = b.alloc_id();
    b.type_declarations.push(op(2, OP_TYPE_BOOL));
    b.type_declarations.push(bool_type_id);

    // Let the caller create the input/output element types.
    let (in_elem_type, out_elem_type) = make_in_out_types(&mut b, uint_type);

    // Push constant block (n: uint).
    let pc_struct = b.type_struct(&[uint_type]);
    let pc_ptr_type = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, pc_struct);
    let pc_var = b.variable(pc_ptr_type, STORAGE_CLASS_PUSH_CONSTANT);

    // Input buffer.
    let in_rt_array = b.type_runtime_array(in_elem_type);
    let in_struct = b.type_struct(&[in_rt_array]);
    let in_ptr_type = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, in_struct);
    let in_buf_var = b.variable(in_ptr_type, STORAGE_CLASS_STORAGE_BUFFER);

    // Output buffer.
    let out_rt_array = b.type_runtime_array(out_elem_type);
    let out_struct = b.type_struct(&[out_rt_array]);
    let out_ptr_type = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, out_struct);
    let out_buf_var = b.variable(out_ptr_type, STORAGE_CLASS_STORAGE_BUFFER);

    // Global invocation ID.
    let uvec3_input_ptr = b.type_pointer(STORAGE_CLASS_INPUT, uvec3_type);
    let gl_global_id = b.variable(uvec3_input_ptr, STORAGE_CLASS_INPUT);

    // Element pointer types.
    let in_elem_ptr_type = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, in_elem_type);
    let out_elem_ptr_type = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, out_elem_type);
    let _uint_sb_ptr = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, uint_type);
    let uint_pc_ptr = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, uint_type);

    // Constants.
    let const_0 = b.constant_u32(uint_type, 0);

    // Function type.
    let func_type = b.type_function(void_type, &[]);

    // Decorations: push constant block.
    b.decorate(pc_struct, DECORATION_BLOCK, &[]);
    b.member_decorate(pc_struct, 0, DECORATION_OFFSET, &[0]);

    // Decorations: input buffer.
    b.decorate(in_struct, DECORATION_BLOCK, &[]);
    b.member_decorate(in_struct, 0, DECORATION_OFFSET, &[0]);
    let in_elem_stride = element_stride(in_elem_type, &b);
    b.decorate(in_rt_array, DECORATION_ARRAY_STRIDE, &[in_elem_stride]);
    b.decorate(in_buf_var, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(in_buf_var, DECORATION_BINDING, &[0]);
    b.decorate(in_buf_var, DECORATION_NON_WRITABLE, &[]);

    // Decorations: output buffer.
    b.decorate(out_struct, DECORATION_BLOCK, &[]);
    b.member_decorate(out_struct, 0, DECORATION_OFFSET, &[0]);
    let out_elem_stride = element_stride(out_elem_type, &b);
    b.decorate(out_rt_array, DECORATION_ARRAY_STRIDE, &[out_elem_stride]);
    b.decorate(out_buf_var, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(out_buf_var, DECORATION_BINDING, &[1]);

    // Decorations: global invocation ID.
    b.decorate(
        gl_global_id,
        DECORATION_BUILTIN,
        &[BUILTIN_GLOBAL_INVOCATION_ID],
    );

    // Entry point (declared later, after func_id is known).
    // We'll do it after func_begin.

    // Begin function.
    let func_id = b.func_begin(void_type, func_type);

    // Entry point and execution mode.
    b.entry_point(EXECUTION_MODEL_GLCOMPUTE, func_id, "main", &[gl_global_id]);
    b.execution_mode_local_size(func_id, CAST_WORKGROUP_SIZE, 1, 1);

    let _entry_label = b.label();

    // Load global invocation ID.x.
    let gid_vec = b.load(uvec3_type, gl_global_id);
    let gid = b.composite_extract(uint_type, gid_vec, 0);

    // Load n from push constants.
    let n_ptr = b.access_chain(uint_pc_ptr, pc_var, &[const_0]);
    let n_val = b.load(uint_type, n_ptr);

    // Bounds check: if gid < n.
    let in_bounds = b.u_less_than(bool_type_id, gid, n_val);

    // Pre-allocate merge and body labels.
    let merge_label = b.alloc_id();
    let body_label = b.alloc_id();

    b.selection_merge(merge_label);
    b.branch_conditional(in_bounds, body_label, merge_label);

    // Body label.
    b.functions.push(op(2, OP_LABEL));
    b.functions.push(body_label);

    CastSetup {
        b,
        func_id,
        gid,
        n_val,
        in_bounds,
        merge_label,
        body_label,
        uint_type,
        bool_type: bool_type_id,
        in_elem_ptr_type,
        out_elem_ptr_type,
        in_buf_var,
        out_buf_var,
    }
}

/// Finish a cast shader (branch to merge, return, function end, build).
fn finish_cast(mut setup: CastSetup) -> Vec<u32> {
    // Branch to merge.
    setup.b.branch(setup.merge_label);

    // Merge label.
    setup.b.functions.push(op(2, OP_LABEL));
    setup.b.functions.push(setup.merge_label);

    // Return and end function.
    setup.b.ret();
    setup.b.func_end();

    setup.b.build()
}

/// Determine the byte stride of an element type for ArrayStride decoration.
/// We inspect the type_declarations buffer to find the type's width.
fn element_stride(type_id: u32, builder: &SpirVBuilder) -> u32 {
    let decls = &builder.type_declarations;
    let mut i = 0;
    while i < decls.len() {
        let word = decls[i];
        let wc = (word >> 16) as usize;
        let opc = (word & 0xFFFF) as u16;
        if wc >= 3 && i + 2 < decls.len() && decls[i + 1] == type_id {
            match opc {
                OP_TYPE_FLOAT => return decls[i + 2] / 8, // width in bits → bytes
                OP_TYPE_INT => return decls[i + 2] / 8,
                _ => {}
            }
        }
        i += wc.max(1);
    }
    4 // default to 4 bytes
}

// ---- F32 <-> F16 ----

/// Generate SPIR-V for F32 to F16 conversion.
///
/// Uses `Float16` capability and `OpFConvert` for native conversion.
/// Input: binding 0 (F32 runtime array). Output: binding 1 (F16 runtime array).
/// Push constant: `n` (element count).
pub fn generate_f32_to_f16_spirv(_n: u32) -> Vec<u32> {
    let mut setup = setup_cast(|b, _uint_type| {
        let f32_type = b.type_float(32);
        let f16_type = b.type_float(16);
        (f32_type, f16_type)
    });

    // Add Float16 capability.
    setup.b.capability(CAPABILITY_FLOAT16);

    let const_0 = setup.b.constant_u32(setup.uint_type, 0);

    // Load input[gid].
    let in_ptr = setup.b.access_chain(
        setup.in_elem_ptr_type,
        setup.in_buf_var,
        &[const_0, setup.gid],
    );

    // Need f32 type for load — look it up from the input element pointer.
    // The input element type is f32 (first type created in the closure).
    // We need result type for load. Let's find the f32 type ID.
    // Actually the load's result type must match the element type.
    // We don't have it directly, so let's search the type declarations.
    let f32_type = find_float_type(&setup.b, 32).expect("f32 type must exist");
    let f16_type = find_float_type(&setup.b, 16).expect("f16 type must exist");

    let val = setup.b.load(f32_type, in_ptr);

    // Convert.
    let converted = setup.b.fconvert(f16_type, val);

    // Store output[gid].
    let out_ptr = setup.b.access_chain(
        setup.out_elem_ptr_type,
        setup.out_buf_var,
        &[const_0, setup.gid],
    );
    setup.b.store(out_ptr, converted);

    finish_cast(setup)
}

/// Generate SPIR-V for F16 to F32 conversion.
///
/// Uses `Float16` capability and `OpFConvert` for native conversion.
/// Input: binding 0 (F16 runtime array). Output: binding 1 (F32 runtime array).
/// Push constant: `n` (element count).
pub fn generate_f16_to_f32_spirv(_n: u32) -> Vec<u32> {
    let mut setup = setup_cast(|b, _uint_type| {
        let f16_type = b.type_float(16);
        let f32_type = b.type_float(32);
        (f16_type, f32_type)
    });

    // Add Float16 capability.
    setup.b.capability(CAPABILITY_FLOAT16);

    let const_0 = setup.b.constant_u32(setup.uint_type, 0);

    let f16_type = find_float_type(&setup.b, 16).expect("f16 type must exist");
    let f32_type = find_float_type(&setup.b, 32).expect("f32 type must exist");

    // Load input[gid] (f16).
    let in_ptr = setup.b.access_chain(
        setup.in_elem_ptr_type,
        setup.in_buf_var,
        &[const_0, setup.gid],
    );
    let val = setup.b.load(f16_type, in_ptr);

    // Convert to f32.
    let converted = setup.b.fconvert(f32_type, val);

    // Store output[gid].
    let out_ptr = setup.b.access_chain(
        setup.out_elem_ptr_type,
        setup.out_buf_var,
        &[const_0, setup.gid],
    );
    setup.b.store(out_ptr, converted);

    finish_cast(setup)
}

// ---- F32 <-> BF16 ----

/// Generate SPIR-V for F32 to BF16 conversion.
///
/// BF16 has no native SPIR-V type. It is stored as uint16.
/// Conversion: reinterpret F32 as uint32 via OpBitcast, shift right 16, convert to uint16.
/// Input: binding 0 (F32 runtime array). Output: binding 1 (uint16 runtime array).
/// Push constant: `n` (element count).
pub fn generate_f32_to_bf16_spirv(_n: u32) -> Vec<u32> {
    let mut setup = setup_cast(|b, _uint_type| {
        let f32_type = b.type_float(32);
        let u16_type = b.type_int(16, 0);
        (f32_type, u16_type)
    });

    // Add Int16 capability and 16-bit storage extension.
    setup.b.capability(CAPABILITY_INT16);
    setup.b.capability(CAPABILITY_STORAGE_BUFFER_16BIT);
    setup.b.extension("SPV_KHR_16bit_storage");

    let const_0 = setup.b.constant_u32(setup.uint_type, 0);
    let const_16 = setup.b.constant_u32(setup.uint_type, 16);

    let f32_type = find_float_type(&setup.b, 32).expect("f32 type must exist");
    let u16_type = find_int_type(&setup.b, 16, 0).expect("u16 type must exist");

    // Load input[gid] (f32).
    let in_ptr = setup.b.access_chain(
        setup.in_elem_ptr_type,
        setup.in_buf_var,
        &[const_0, setup.gid],
    );
    let val = setup.b.load(f32_type, in_ptr);

    // Bitcast f32 → uint32.
    let as_u32 = setup.b.bitcast(setup.uint_type, val);

    // Shift right 16 → upper 16 bits (bf16 value) now in lower 16 bits of uint32.
    let shifted = setup
        .b
        .shift_right_logical(setup.uint_type, as_u32, const_16);

    // Convert uint32 → uint16 via OpUConvert.
    let as_u16 = setup.b.uconvert(u16_type, shifted);

    // Store output[gid].
    let out_ptr = setup.b.access_chain(
        setup.out_elem_ptr_type,
        setup.out_buf_var,
        &[const_0, setup.gid],
    );
    setup.b.store(out_ptr, as_u16);

    finish_cast(setup)
}

/// Generate SPIR-V for BF16 to F32 conversion.
///
/// BF16 is stored as uint16. Conversion: load uint16, convert to uint32, shift left 16,
/// bitcast to f32.
/// Input: binding 0 (uint16 runtime array). Output: binding 1 (F32 runtime array).
/// Push constant: `n` (element count).
pub fn generate_bf16_to_f32_spirv(_n: u32) -> Vec<u32> {
    let mut setup = setup_cast(|b, _uint_type| {
        let u16_type = b.type_int(16, 0);
        let f32_type = b.type_float(32);
        (u16_type, f32_type)
    });

    // Add Int16 capability and 16-bit storage extension.
    setup.b.capability(CAPABILITY_INT16);
    setup.b.capability(CAPABILITY_STORAGE_BUFFER_16BIT);
    setup.b.extension("SPV_KHR_16bit_storage");

    let const_0 = setup.b.constant_u32(setup.uint_type, 0);
    let const_16 = setup.b.constant_u32(setup.uint_type, 16);

    let u16_type = find_int_type(&setup.b, 16, 0).expect("u16 type must exist");
    let f32_type = find_float_type(&setup.b, 32).expect("f32 type must exist");

    // Load input[gid] (uint16 = bf16 bits).
    let in_ptr = setup.b.access_chain(
        setup.in_elem_ptr_type,
        setup.in_buf_var,
        &[const_0, setup.gid],
    );
    let val_u16 = setup.b.load(u16_type, in_ptr);

    // Convert uint16 → uint32 via OpUConvert.
    let val_u32 = setup.b.uconvert(setup.uint_type, val_u16);

    // Shift left 16 → bf16 bits now in upper 16 bits of uint32.
    let shifted = setup
        .b
        .shift_left_logical(setup.uint_type, val_u32, const_16);

    // Bitcast uint32 → f32.
    let as_f32 = setup.b.bitcast(f32_type, shifted);

    // Store output[gid].
    let out_ptr = setup.b.access_chain(
        setup.out_elem_ptr_type,
        setup.out_buf_var,
        &[const_0, setup.gid],
    );
    setup.b.store(out_ptr, as_f32);

    finish_cast(setup)
}

// ---- Type lookup helpers ----

/// Find a float type with the given width in the builder's type declarations.
fn find_float_type(builder: &SpirVBuilder, width: u32) -> Option<u32> {
    let decls = &builder.type_declarations;
    let mut i = 0;
    while i < decls.len() {
        let word = decls[i];
        let wc = (word >> 16) as usize;
        let opc = (word & 0xFFFF) as u16;
        if opc == OP_TYPE_FLOAT && wc == 3 && i + 2 < decls.len() && decls[i + 2] == width {
            return Some(decls[i + 1]);
        }
        i += wc.max(1);
    }
    None
}

/// Find an integer type with the given width and signedness.
fn find_int_type(builder: &SpirVBuilder, width: u32, signedness: u32) -> Option<u32> {
    let decls = &builder.type_declarations;
    let mut i = 0;
    while i < decls.len() {
        let word = decls[i];
        let wc = (word >> 16) as usize;
        let opc = (word & 0xFFFF) as u16;
        if opc == OP_TYPE_INT
            && wc == 4
            && i + 3 < decls.len()
            && decls[i + 2] == width
            && decls[i + 3] == signedness
        {
            return Some(decls[i + 1]);
        }
        i += wc.max(1);
    }
    None
}

#[cfg(test)]
#[path = "spirv_cast_tests.rs"]
mod spirv_cast_tests;
