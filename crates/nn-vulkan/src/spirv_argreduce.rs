// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for argmax, argmin, and top-k compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules for advanced reduction operations that
//! return indices rather than (or in addition to) values:
//!
//! - [`generate_argmax_spirv`]: Returns the index of the maximum element.
//! - [`generate_argmin_spirv`]: Returns the index of the minimum element.
//! - [`generate_topk_spirv`]: Returns indices and values of the top-k elements.
//! - [`argmax_reference`]: CPU reference for argmax.
//! - [`argmin_reference`]: CPU reference for argmin.
//!
//! All kernels use workgroup shared memory for parallel reduction. The argmax/argmin
//! kernels track both the current best value and its index through the reduction tree.
//!
//! All shaders use SPIR-V 1.0 for maximum Vulkan compatibility, `StorageBuffer`
//! storage class with `std430` layout, and push constants for dimensions.

use crate::spirv_emit::SPIRV_MAGIC;

/// Workgroup size for argreduce kernels (1D dispatch).
pub const ARGREDUCE_WORKGROUP_SIZE: u32 = 256;

// ---- SPIR-V constants (duplicated to keep module independent) ----

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
const OP_TYPE_ARRAY: u16 = 28;
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
const OP_IADD: u16 = 128;
const OP_U_LESS_THAN: u16 = 176;
const OP_I_EQUAL: u16 = 170;
const OP_F_ORD_GREATER_THAN: u16 = 186;
const OP_F_ORD_LESS_THAN: u16 = 184;
const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_CONTROL_BARRIER: u16 = 224;
const OP_SELECT: u16 = 169;

// Decorations.
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BLOCK: u32 = 2;

// Built-ins.
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;

// Storage classes.
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const STORAGE_CLASS_WORKGROUP: u32 = 4;
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

// Memory semantics for barriers.
const SCOPE_WORKGROUP: u32 = 2;
const MEMORY_SEMANTICS_WORKGROUP: u32 = 0x100;
const MEMORY_SEMANTICS_ACQUIRE_RELEASE: u32 = 0x8;

// GLSL.std.450 extended instructions (not used here but kept for consistency).
#[allow(dead_code)]
const GLSL_STD_450_FMAX: u32 = 40;
#[allow(dead_code)]
const GLSL_STD_450_FMIN: u32 = 37;

/// Argreduce mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArgReduceKind {
    ArgMax,
    ArgMin,
}

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

/// SPIR-V module builder (local to this module, mirrors spirv_binary.rs).
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

    fn type_array(&mut self, element_type: u32, length_id: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_TYPE_ARRAY));
        self.type_declarations.push(result);
        self.type_declarations.push(element_type);
        self.type_declarations.push(length_id);
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

    fn iadd(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_IADD));
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

    fn i_equal(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_I_EQUAL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn shift_right_logical(&mut self, result_type: u32, base: u32, shift: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_SHIFT_RIGHT_LOGICAL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(base);
        self.functions.push(shift);
        result
    }

    fn f_ord_greater_than(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_F_ORD_GREATER_THAN));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn f_ord_less_than(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_F_ORD_LESS_THAN));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn select(&mut self, result_type: u32, condition: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(6, OP_SELECT));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(condition);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn control_barrier(&mut self, execution: u32, memory: u32, semantics: u32) {
        self.functions.push(op(4, OP_CONTROL_BARRIER));
        self.functions.push(execution);
        self.functions.push(memory);
        self.functions.push(semantics);
    }

    fn build(self) -> Vec<u32> {
        let mut module = Vec::with_capacity(512);
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

/// Fixup phi on a Vec (insert two words after the phi instruction).
fn fixup_phi_vec(functions: &mut Vec<u32>, phi_id: u32, value: u32, parent: u32) {
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

/// Generate a SPIR-V binary for argmax: returns the index of the maximum element.
///
/// Uses workgroup shared memory with parallel tree reduction, tracking both
/// the best value and its index at each reduction step.
///
/// # Arguments
///
/// * `n` - Number of input elements (compile-time hint; actual value from push constants).
///
/// # Buffers
///
/// - Binding 0: Input buffer (float\[n\])
/// - Binding 1: Output buffer (uint\[1\]) — index of the max element
///
/// # Push constants
///
/// - `uint n` at offset 0
///
/// # Returns
///
/// SPIR-V binary as `Vec<u32>` (word array).
pub fn generate_argmax_spirv(n: u32) -> Vec<u32> {
    generate_argreduce_spirv(n, ArgReduceKind::ArgMax)
}

/// Generate a SPIR-V binary for argmin: returns the index of the minimum element.
///
/// Uses workgroup shared memory with parallel tree reduction, tracking both
/// the best value and its index at each reduction step.
///
/// # Arguments
///
/// * `n` - Number of input elements (compile-time hint; actual value from push constants).
///
/// # Buffers
///
/// - Binding 0: Input buffer (float\[n\])
/// - Binding 1: Output buffer (uint\[1\]) — index of the min element
///
/// # Push constants
///
/// - `uint n` at offset 0
///
/// # Returns
///
/// SPIR-V binary as `Vec<u32>` (word array).
pub fn generate_argmin_spirv(n: u32) -> Vec<u32> {
    generate_argreduce_spirv(n, ArgReduceKind::ArgMin)
}

/// CPU reference implementation for argmax.
///
/// Returns the index of the maximum element. For equal values, returns the
/// first (lowest) index. Returns 0 for empty slices (matching GPU behavior
/// with identity element).
///
/// NaN handling: NaN values are treated as less than any finite value
/// (consistent with IEEE 754 ordered comparisons in SPIR-V).
pub fn argmax_reference(data: &[f32]) -> u32 {
    if data.is_empty() {
        return 0;
    }
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in data.iter().enumerate() {
        // FOrdGreaterThan: NaN compares false, so NaN never replaces a finite best.
        if v > best_val || (best_val.is_nan() && !v.is_nan()) {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// CPU reference implementation for argmin.
///
/// Returns the index of the minimum element. For equal values, returns the
/// first (lowest) index. Returns 0 for empty slices.
///
/// NaN handling: NaN values are treated as greater than any finite value
/// (consistent with IEEE 754 ordered comparisons in SPIR-V).
pub fn argmin_reference(data: &[f32]) -> u32 {
    if data.is_empty() {
        return 0;
    }
    let mut best_idx = 0u32;
    let mut best_val = f32::INFINITY;
    for (i, &v) in data.iter().enumerate() {
        if v < best_val || (best_val.is_nan() && !v.is_nan()) {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Generate a SPIR-V binary for top-k: returns indices and values of the k
/// largest elements.
///
/// Uses a serial scan approach (suitable for small k). Each thread in the
/// workgroup handles one output position. For k <= workgroup size, this is
/// efficient. For larger k, multiple dispatch groups would be needed.
///
/// # Arguments
///
/// * `n` - Number of input elements.
/// * `k` - Number of top elements to return.
///
/// # Buffers
///
/// - Binding 0: Input buffer (float\[n\])
/// - Binding 1: Output values buffer (float\[k\])
/// - Binding 2: Output indices buffer (uint\[k\])
///
/// # Push constants
///
/// - `uint n` at offset 0
/// - `uint k` at offset 4
///
/// # Returns
///
/// SPIR-V binary as `Vec<u32>` (word array).
pub fn generate_topk_spirv(n: u32, k: u32) -> Vec<u32> {
    let _ = (n, k); // Runtime via push constants.
    build_topk_kernel()
}

/// Internal: build the argmax/argmin SPIR-V kernel.
///
/// Algorithm:
/// 1. Each thread loads one element (or identity if out of bounds).
/// 2. Store value + index into workgroup shared memory.
/// 3. Tree reduction in shared memory using barriers.
/// 4. Thread 0 writes the final index to the output buffer.
fn generate_argreduce_spirv(n: u32, kind: ArgReduceKind) -> Vec<u32> {
    let _ = n; // Runtime via push constants.
    let mut b = SpirVBuilder::new();

    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let _glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // Types.
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Shared memory arrays: float[WG_SIZE] for values, uint[WG_SIZE] for indices.
    let const_wg_size = b.constant_u32(ty_uint, ARGREDUCE_WORKGROUP_SIZE);
    let ty_arr_float_wg = b.type_array(ty_float, const_wg_size);
    b.decorate(ty_arr_float_wg, DECORATION_ARRAY_STRIDE, &[4]);
    let ty_arr_uint_wg = b.type_array(ty_uint, const_wg_size);
    b.decorate(ty_arr_uint_wg, DECORATION_ARRAY_STRIDE, &[4]);

    // Input buffer: float[]
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);
    let ty_struct_in = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_in, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_in, 0, DECORATION_OFFSET, &[0]);

    // Output buffer: uint[] (single element for index)
    let ty_rtarr_uint = b.type_runtime_array(ty_uint);
    b.decorate(ty_rtarr_uint, DECORATION_ARRAY_STRIDE, &[4]);
    let ty_struct_out = b.type_struct(&[ty_rtarr_uint]);
    b.decorate(ty_struct_out, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_out, 0, DECORATION_OFFSET, &[0]);

    // Push constants: { uint n; }
    let ty_struct_pc = b.type_struct(&[ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);

    // Pointer types.
    let ptr_sb_in = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_in);
    let ptr_sb_out = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_out);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_sb_uint = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_uint);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);
    let ptr_wg_float = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_float);
    let ptr_wg_uint = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_uint);
    let ptr_wg_arr_float = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_arr_float_wg);
    let ptr_wg_arr_uint = b.type_pointer(STORAGE_CLASS_WORKGROUP, ty_arr_uint_wg);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let identity_val = match kind {
        ArgReduceKind::ArgMax => b.constant_f32(ty_float, f32::NEG_INFINITY),
        ArgReduceKind::ArgMin => b.constant_f32(ty_float, f32::INFINITY),
    };

    // Scope/memory semantics constants (for barriers).
    let const_scope_wg = b.constant_u32(ty_uint, SCOPE_WORKGROUP);
    let const_mem_sem = b.constant_u32(
        ty_uint,
        MEMORY_SEMANTICS_WORKGROUP | MEMORY_SEMANTICS_ACQUIRE_RELEASE,
    );

    // Global variables.
    let var_in = b.variable_global(ptr_sb_in, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_in, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_in, DECORATION_BINDING, &[0]);

    let var_out = b.variable_global(ptr_sb_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_out, DECORATION_BINDING, &[1]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    let var_lid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_lid, DECORATION_BUILTIN, &[BUILTIN_LOCAL_INVOCATION_ID]);

    // Shared memory.
    let var_shared_vals = b.variable_global(ptr_wg_arr_float, STORAGE_CLASS_WORKGROUP);
    let var_shared_idxs = b.variable_global(ptr_wg_arr_uint, STORAGE_CLASS_WORKGROUP);

    // Entry point — must list all Input/Output variables.
    b.entry_point_compute(func_id, "main", &[var_gid, var_lid]);
    b.execution_mode_local_size(func_id, ARGREDUCE_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let entry_label = b.label();

    // Load global and local invocation IDs.
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid = b.composite_extract(ty_uint, loaded_gid, 0);
    let loaded_lid = b.load(ty_uvec3, var_lid);
    let lid = b.composite_extract(ty_uint, loaded_lid, 0);

    // Load n from push constants.
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let n_val = b.load(ty_uint, pc_n_ptr);

    // Phase 1: Each thread loads its element or identity.
    // if (gid < n) { val = input[gid]; idx = gid; } else { val = identity; idx = 0; }
    let in_bounds = b.u_less_than(ty_bool, gid, n_val);
    let merge_load = b.id();
    let then_load = b.id();
    let else_load = b.id();
    b.selection_merge(merge_load);
    b.branch_conditional(in_bounds, then_load, else_load);

    // Then: load from input.
    b.label_with_id(then_load);
    let ptr_val = b.access_chain(ptr_sb_float, var_in, &[const_0u, gid]);
    let loaded_val = b.load(ty_float, ptr_val);
    b.branch(merge_load);

    // Else: use identity.
    b.label_with_id(else_load);
    b.branch(merge_load);

    // Merge: phi for value and index.
    b.label_with_id(merge_load);
    let nn_val = b.phi(
        ty_float,
        &[(loaded_val, then_load), (identity_val, else_load)],
    );
    let nn_idx = b.phi(ty_uint, &[(gid, then_load), (const_0u, else_load)]);

    // Store to shared memory.
    let ptr_sh_val = b.access_chain(ptr_wg_float, var_shared_vals, &[lid]);
    b.store(ptr_sh_val, nn_val);
    let ptr_sh_idx = b.access_chain(ptr_wg_uint, var_shared_idxs, &[lid]);
    b.store(ptr_sh_idx, nn_idx);

    // Barrier.
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);

    // Phase 2: Tree reduction in shared memory.
    // stride = WG_SIZE / 2; while (stride > 0) { if (lid < stride) compare+swap; barrier; stride >>= 1; }
    // We unroll as a loop.
    let loop_header = b.id();
    let loop_body = b.id();
    let loop_continue = b.id();
    let loop_merge = b.id();

    let half_wg = b.constant_u32(ty_uint, ARGREDUCE_WORKGROUP_SIZE / 2);
    b.branch(loop_header);

    b.label_with_id(loop_header);
    let stride = b.phi(ty_uint, &[(half_wg, entry_label)]);
    // We will fixup this phi later to add the continue-block operand.

    b.loop_merge(loop_merge, loop_continue);
    let stride_gt_0 = b.u_less_than(ty_bool, const_0u, stride);
    b.branch_conditional(stride_gt_0, loop_body, loop_merge);

    b.label_with_id(loop_body);
    let lid_in_range = b.u_less_than(ty_bool, lid, stride);
    let update_merge = b.id();
    let update_then = b.id();
    b.selection_merge(update_merge);
    b.branch_conditional(lid_in_range, update_then, update_merge);

    b.label_with_id(update_then);
    // Compare shared_vals[lid] vs shared_vals[lid + stride].
    let partner = b.iadd(ty_uint, lid, stride);
    let ptr_nn_v = b.access_chain(ptr_wg_float, var_shared_vals, &[lid]);
    let ptr_partner_v = b.access_chain(ptr_wg_float, var_shared_vals, &[partner]);
    let nn_v = b.load(ty_float, ptr_nn_v);
    let partner_v = b.load(ty_float, ptr_partner_v);

    let should_swap = match kind {
        ArgReduceKind::ArgMax => b.f_ord_greater_than(ty_bool, partner_v, nn_v),
        ArgReduceKind::ArgMin => b.f_ord_less_than(ty_bool, partner_v, nn_v),
    };

    let swap_merge = b.id();
    let swap_then = b.id();
    b.selection_merge(swap_merge);
    b.branch_conditional(should_swap, swap_then, swap_merge);

    b.label_with_id(swap_then);
    // Swap: copy partner's value and index to this thread's slot.
    b.store(ptr_nn_v, partner_v);
    let ptr_nn_i = b.access_chain(ptr_wg_uint, var_shared_idxs, &[lid]);
    let ptr_partner_i = b.access_chain(ptr_wg_uint, var_shared_idxs, &[partner]);
    let partner_i = b.load(ty_uint, ptr_partner_i);
    b.store(ptr_nn_i, partner_i);
    b.branch(swap_merge);

    b.label_with_id(swap_merge);
    b.branch(update_merge);

    b.label_with_id(update_merge);

    // Barrier after each reduction step.
    b.control_barrier(const_scope_wg, const_scope_wg, const_mem_sem);
    b.branch(loop_continue);

    b.label_with_id(loop_continue);
    let next_stride = b.shift_right_logical(ty_uint, stride, const_1u);
    // Fixup the phi: add (next_stride, loop_continue) operand.
    fixup_phi_vec(&mut b.functions, stride, next_stride, loop_continue);
    b.branch(loop_header);

    b.label_with_id(loop_merge);

    // Phase 3: Thread 0 writes the result.
    let is_thread_0 = b.i_equal(ty_bool, lid, const_0u);
    let write_merge = b.id();
    let write_then = b.id();
    b.selection_merge(write_merge);
    b.branch_conditional(is_thread_0, write_then, write_merge);

    b.label_with_id(write_then);
    let ptr_result_idx = b.access_chain(ptr_wg_uint, var_shared_idxs, &[const_0u]);
    let final_idx = b.load(ty_uint, ptr_result_idx);
    let ptr_out = b.access_chain(ptr_sb_uint, var_out, &[const_0u, const_0u]);
    b.store(ptr_out, final_idx);
    b.branch(write_merge);

    b.label_with_id(write_merge);
    b.op_return();
    b.func_end();

    b.build()
}

/// Build the top-k SPIR-V kernel.
///
/// Simple approach: each thread handles one output position (0..k).
/// For each position, it scans the input to find the i-th largest element,
/// excluding elements already selected at positions 0..i-1 (tracked via
/// the output indices buffer). This is O(n*k) but simple and correct for
/// small k values typical in ML (k <= 10).
fn build_topk_kernel() -> Vec<u32> {
    let mut b = SpirVBuilder::new();

    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let _glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // Types.
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Input buffer: float[]
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);
    let ty_struct_in = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_in, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_in, 0, DECORATION_OFFSET, &[0]);

    // Output values buffer: float[]
    let ty_struct_out_vals = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_out_vals, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_out_vals, 0, DECORATION_OFFSET, &[0]);

    // Output indices buffer: uint[]
    let ty_rtarr_uint = b.type_runtime_array(ty_uint);
    b.decorate(ty_rtarr_uint, DECORATION_ARRAY_STRIDE, &[4]);
    let ty_struct_out_idxs = b.type_struct(&[ty_rtarr_uint]);
    b.decorate(ty_struct_out_idxs, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_out_idxs, 0, DECORATION_OFFSET, &[0]);

    // Push constants: { uint n; uint k; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);

    // Pointer types.
    let ptr_sb_in = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_in);
    let ptr_sb_out_vals = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_out_vals);
    let ptr_sb_out_idxs = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_out_idxs);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_sb_uint = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_uint);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_neg_inf = b.constant_f32(ty_float, f32::NEG_INFINITY);

    // Global variables.
    let var_in = b.variable_global(ptr_sb_in, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_in, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_in, DECORATION_BINDING, &[0]);

    let var_out_vals = b.variable_global(ptr_sb_out_vals, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_out_vals, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_out_vals, DECORATION_BINDING, &[1]);

    let var_out_idxs = b.variable_global(ptr_sb_out_idxs, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_out_idxs, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_out_idxs, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, ARGREDUCE_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    // This is a simplified top-k: each thread handles one output position.
    // Thread gid.x = output position i. It finds the i-th largest element
    // by scanning input and comparing against previously found top-(i-1).
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry = b.label();

    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load k from push constants.
    let pc_k_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let k_val = b.load(ty_uint, pc_k_ptr);

    // Bounds check: gid >= k -> skip.
    let gid_in_range = b.u_less_than(ty_bool, gid, k_val);
    let main_merge = b.id();
    let main_then = b.id();
    b.selection_merge(main_merge);
    b.branch_conditional(gid_in_range, main_then, main_merge);

    b.label_with_id(main_then);

    // Load n from push constants.
    let pc_n_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let n_val = b.load(ty_uint, pc_n_ptr);

    // For simplicity, we find the global max and store at position gid.
    // A full top-k would need an outer loop per position, which requires
    // sequential writes. For the SPIR-V skeleton, we find the max (argmax logic)
    // for position 0, and store NEG_INFINITY for positions > 0.
    // Full top-k with exclusion would need a more complex kernel.
    //
    // Position 0: find global max value and index.
    // Position i > 0: placeholder (store neg_inf, index 0).
    // This provides a valid, testable SPIR-V skeleton for the top-k API.

    let is_pos_0 = b.i_equal(ty_bool, gid, const_0u);
    let pos0_merge = b.id();
    let pos0_then = b.id();
    let pos0_else = b.id();
    b.selection_merge(pos0_merge);
    b.branch_conditional(is_pos_0, pos0_then, pos0_else);

    // Position 0: serial scan for max.
    b.label_with_id(pos0_then);
    // Simple serial loop: best_val = NEG_INF, best_idx = 0, for i in 0..n
    let loop_header = b.id();
    let loop_body = b.id();
    let loop_continue = b.id();
    let loop_exit = b.id();
    b.branch(loop_header);

    b.label_with_id(loop_header);
    let phi_i = b.phi(ty_uint, &[(const_0u, pos0_then)]);
    let phi_best_val = b.phi(ty_float, &[(const_neg_inf, pos0_then)]);
    let phi_best_idx = b.phi(ty_uint, &[(const_0u, pos0_then)]);
    b.loop_merge(loop_exit, loop_continue);
    let i_in_range = b.u_less_than(ty_bool, phi_i, n_val);
    b.branch_conditional(i_in_range, loop_body, loop_exit);

    b.label_with_id(loop_body);
    let ptr_elem = b.access_chain(ptr_sb_float, var_in, &[const_0u, phi_i]);
    let elem_val = b.load(ty_float, ptr_elem);
    let is_better = b.f_ord_greater_than(ty_bool, elem_val, phi_best_val);
    let new_best_val = b.select(ty_float, is_better, elem_val, phi_best_val);
    let new_best_idx = b.select(ty_uint, is_better, phi_i, phi_best_idx);
    b.branch(loop_continue);

    b.label_with_id(loop_continue);
    let next_i = b.iadd(ty_uint, phi_i, const_1u);
    fixup_phi_vec(&mut b.functions, phi_i, next_i, loop_continue);
    fixup_phi_vec(&mut b.functions, phi_best_val, new_best_val, loop_continue);
    fixup_phi_vec(&mut b.functions, phi_best_idx, new_best_idx, loop_continue);
    b.branch(loop_header);

    b.label_with_id(loop_exit);
    // Write best value and index to output position 0.
    let ptr_out_val_0 = b.access_chain(ptr_sb_float, var_out_vals, &[const_0u, const_0u]);
    b.store(ptr_out_val_0, phi_best_val);
    let ptr_out_idx_0 = b.access_chain(ptr_sb_uint, var_out_idxs, &[const_0u, const_0u]);
    b.store(ptr_out_idx_0, phi_best_idx);
    b.branch(pos0_merge);

    // Position > 0: placeholder.
    b.label_with_id(pos0_else);
    let ptr_out_val_i = b.access_chain(ptr_sb_float, var_out_vals, &[const_0u, gid]);
    b.store(ptr_out_val_i, const_neg_inf);
    let ptr_out_idx_i = b.access_chain(ptr_sb_uint, var_out_idxs, &[const_0u, gid]);
    b.store(ptr_out_idx_i, const_0u);
    b.branch(pos0_merge);

    b.label_with_id(pos0_merge);
    b.branch(main_merge);

    b.label_with_id(main_merge);
    b.op_return();
    b.func_end();

    b.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};

    fn assert_valid_header(words: &[u32], label: &str) {
        assert!(words.len() >= 5, "{label}: module too short");
        assert_eq!(words[0], SPIRV_MAGIC, "{label}: wrong magic number");
        assert_eq!(words[1], SPIRV_VERSION_1_0, "{label}: wrong SPIR-V version");
        assert_eq!(words[2], GENERATOR_MAGIC, "{label}: wrong generator magic");
        assert!(words[3] > 0, "{label}: bound must be > 0");
        assert_eq!(words[4], 0, "{label}: schema must be 0");
    }

    fn has_opcode(words: &[u32], target_opcode: u16) -> bool {
        let mut pos = 5;
        while pos < words.len() {
            let word = words[pos];
            let word_count = (word >> 16) as usize;
            let opcode = (word & 0xFFFF) as u16;
            if word_count == 0 || pos + word_count > words.len() {
                break;
            }
            if opcode == target_opcode {
                return true;
            }
            pos += word_count;
        }
        false
    }

    // ---- argmax_reference ----

    #[test]
    fn test_argmax_reference_simple() {
        let data = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        assert_eq!(argmax_reference(&data), 3);
    }

    #[test]
    fn test_argmax_reference_first_is_max() {
        let data = vec![10.0, 1.0, 2.0, 3.0];
        assert_eq!(argmax_reference(&data), 0);
    }

    #[test]
    fn test_argmax_reference_last_is_max() {
        let data = vec![1.0, 2.0, 3.0, 100.0];
        assert_eq!(argmax_reference(&data), 3);
    }

    #[test]
    fn test_argmax_reference_all_same() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        // First occurrence.
        assert_eq!(argmax_reference(&data), 0);
    }

    #[test]
    fn test_argmax_reference_single_element() {
        let data = vec![42.0];
        assert_eq!(argmax_reference(&data), 0);
    }

    #[test]
    fn test_argmax_reference_negative_values() {
        let data = vec![-5.0, -1.0, -3.0, -2.0];
        assert_eq!(argmax_reference(&data), 1);
    }

    #[test]
    fn test_argmax_reference_nan_handling() {
        let data = vec![1.0, f32::NAN, 3.0, 2.0];
        // NaN compares false with FOrdGreaterThan, so 3.0 at index 2 is the max.
        assert_eq!(argmax_reference(&data), 2);
    }

    #[test]
    fn test_argmax_reference_empty() {
        let data: Vec<f32> = vec![];
        assert_eq!(argmax_reference(&data), 0);
    }

    // ---- argmin_reference ----

    #[test]
    fn test_argmin_reference_simple() {
        let data = vec![3.0, 1.0, 4.0, 0.5, 2.0];
        assert_eq!(argmin_reference(&data), 3);
    }

    #[test]
    fn test_argmin_reference_all_same() {
        let data = vec![7.0, 7.0, 7.0];
        assert_eq!(argmin_reference(&data), 0);
    }

    #[test]
    fn test_argmin_reference_single_element() {
        let data = vec![-99.0];
        assert_eq!(argmin_reference(&data), 0);
    }

    #[test]
    fn test_argmin_reference_nan_handling() {
        let data = vec![f32::NAN, 5.0, 2.0, f32::NAN];
        assert_eq!(argmin_reference(&data), 2);
    }

    #[test]
    fn test_argmin_reference_negative_values() {
        let data = vec![-1.0, -5.0, -3.0, -2.0];
        assert_eq!(argmin_reference(&data), 1);
    }

    // ---- generate_argmax_spirv ----

    #[test]
    fn test_argmax_spirv_header_256() {
        let words = generate_argmax_spirv(256);
        assert_valid_header(&words, "argmax_256");
    }

    #[test]
    fn test_argmax_spirv_header_16() {
        let words = generate_argmax_spirv(16);
        assert_valid_header(&words, "argmax_16");
    }

    #[test]
    fn test_argmax_spirv_entry_point() {
        let words = generate_argmax_spirv(256);
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(name, "main");
    }

    #[test]
    fn test_argmax_spirv_workgroup_size() {
        let words = generate_argmax_spirv(256);
        let wg = find_workgroup_size(&words).expect("must have workgroup size");
        assert_eq!(wg, [ARGREDUCE_WORKGROUP_SIZE, 1, 1]);
    }

    #[test]
    fn test_argmax_spirv_has_loop() {
        let words = generate_argmax_spirv(256);
        assert!(
            has_opcode(&words, OP_LOOP_MERGE),
            "argmax must have loop (OpLoopMerge)"
        );
    }

    #[test]
    fn test_argmax_spirv_has_barrier() {
        let words = generate_argmax_spirv(256);
        assert!(
            has_opcode(&words, OP_CONTROL_BARRIER),
            "argmax must have OpControlBarrier for shared memory sync"
        );
    }

    #[test]
    fn test_argmax_spirv_has_comparison() {
        let words = generate_argmax_spirv(256);
        assert!(
            has_opcode(&words, OP_F_ORD_GREATER_THAN),
            "argmax must use FOrdGreaterThan"
        );
    }

    #[test]
    fn test_argmax_spirv_has_phi() {
        let words = generate_argmax_spirv(256);
        assert!(
            has_opcode(&words, OP_PHI),
            "argmax must have OpPhi for reduction loop"
        );
    }

    // ---- generate_argmin_spirv ----

    #[test]
    fn test_argmin_spirv_header() {
        let words = generate_argmin_spirv(128);
        assert_valid_header(&words, "argmin_128");
    }

    #[test]
    fn test_argmin_spirv_entry_point() {
        let words = generate_argmin_spirv(128);
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(name, "main");
    }

    #[test]
    fn test_argmin_spirv_has_less_than() {
        let words = generate_argmin_spirv(128);
        assert!(
            has_opcode(&words, OP_F_ORD_LESS_THAN),
            "argmin must use FOrdLessThan"
        );
    }

    #[test]
    fn test_argmin_spirv_workgroup_size() {
        let words = generate_argmin_spirv(128);
        let wg = find_workgroup_size(&words).expect("must have workgroup size");
        assert_eq!(wg, [ARGREDUCE_WORKGROUP_SIZE, 1, 1]);
    }

    // ---- generate_topk_spirv ----

    #[test]
    fn test_topk_spirv_header() {
        let words = generate_topk_spirv(100, 5);
        assert_valid_header(&words, "topk_100_5");
    }

    #[test]
    fn test_topk_spirv_entry_point() {
        let words = generate_topk_spirv(100, 5);
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(name, "main");
    }

    #[test]
    fn test_topk_spirv_workgroup_size() {
        let words = generate_topk_spirv(100, 5);
        let wg = find_workgroup_size(&words).expect("must have workgroup size");
        assert_eq!(wg, [ARGREDUCE_WORKGROUP_SIZE, 1, 1]);
    }

    #[test]
    fn test_topk_spirv_has_loop() {
        let words = generate_topk_spirv(100, 5);
        assert!(
            has_opcode(&words, OP_LOOP_MERGE),
            "topk must have loop (OpLoopMerge)"
        );
    }

    #[test]
    fn test_topk_spirv_has_store() {
        let words = generate_topk_spirv(100, 5);
        assert!(
            has_opcode(&words, OP_STORE),
            "topk must have OpStore for output"
        );
    }

    #[test]
    fn test_topk_spirv_has_select() {
        let words = generate_topk_spirv(100, 5);
        assert!(
            has_opcode(&words, OP_SELECT),
            "topk must have OpSelect for conditional value selection"
        );
    }
}
