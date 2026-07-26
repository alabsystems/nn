// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code, unreachable_pub)]

//! Consolidated IR validation tests: KernelDef::validate(), IR structure,
//! advanced validation, properties, ref safety, and topology.

#[path = "ir_validate/ir_structure.rs"]
mod ir_structure;
#[path = "ir_validate/ir_validate_advanced.rs"]
mod ir_validate_advanced;
#[path = "ir_validate/ir_validate_properties.rs"]
mod ir_validate_properties;
#[path = "ir_validate/ir_validate_refs.rs"]
mod ir_validate_refs;
#[path = "ir_validate/ir_validate_topology.rs"]
mod ir_validate_topology;
#[path = "ir_validate/ir_validation.rs"]
mod ir_validation;
