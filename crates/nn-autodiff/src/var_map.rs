// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Named variable store for model parameters.
//!
//! [`VarMap`] provides get-or-create semantics: retrieving a variable by name
//! creates it on first access with zero-initialized data, and subsequent
//! accesses return the same variable.

use std::collections::HashMap;
use std::path::Path;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::error::Result;
use crate::var::Var;

/// Named variable store.
///
/// Maps string names to [`Var`] instances, supporting get-or-create access
/// and bulk iteration for optimizer `step()` calls.
pub struct VarMap {
    vars: HashMap<String, Var>,
}

impl VarMap {
    /// Create an empty variable map.
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    /// Get or create a variable by name.
    ///
    /// If the name exists, validates that shape and dtype match the original,
    /// then returns the existing variable. Otherwise creates a zero-initialized
    /// variable and stores it.
    pub fn get(
        &mut self,
        name: &str,
        dims: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Var> {
        if let Some(var) = self.vars.get(name) {
            if var.dims()? != dims {
                return Err(crate::error::AutodiffError::ShapeMismatch {
                    expected: var.dims()?,
                    got: dims.to_vec(),
                });
            }
            if var.dtype()? != dtype {
                return Err(crate::error::AutodiffError::DTypeMismatch {
                    name: name.to_string(),
                    expected: var.dtype()?,
                    got: dtype,
                });
            }
            let _ = device; // device checked at creation, not on retrieval
            return Ok(var.clone());
        }
        let var = Var::zeros(dims, dtype, device)?;
        self.vars.insert(name.to_string(), var.clone());
        Ok(var)
    }

    /// All variables in the map (for optimizer iteration).
    pub fn all_vars(&self) -> Vec<Var> {
        self.vars.values().cloned().collect()
    }

    /// Number of named variables.
    pub fn len(&self) -> usize {
        self.vars.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    /// Save all named variables to a safetensors file.
    ///
    /// Each variable is exported as a named tensor using the key it was
    /// registered with. All tensors are converted to CPU f32 before writing.
    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<()> {
        let tensors: HashMap<String, DynTensor> = self
            .vars
            .iter()
            .map(|(name, var)| Ok((name.clone(), var.data()?)))
            .collect::<Result<_>>()?;
        nn_core::dyn_tensor::save_safetensors(&tensors, path)?;
        Ok(())
    }

    /// Load named variables from a safetensors file, updating existing Vars
    /// in-place.
    ///
    /// Variables in the file whose name matches a variable in the map are
    /// loaded (with shape validation). Variables in the file but not in the
    /// map are ignored (allows partial loads for transfer learning).
    /// Variables in the map but not in the file are left unchanged.
    pub fn load_safetensors(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let loaded = nn_core::dyn_tensor::load_safetensors(path)?;
        for (name, tensor) in &loaded {
            if let Some(var) = self.vars.get(name) {
                // Validate finiteness before writing into Var — a corrupted
                // safetensors file with NaN/Inf would silently poison model
                // parameters, producing garbage output on all future forwards.
                // Fast path: any_non_finite() uses GPU-native reduction when
                // available, avoiding a full GPU→CPU transfer on the happy path.
                if tensor.any_non_finite()? {
                    // Only transfer to CPU on the error path for the count.
                    let view = tensor.to_f32_array()?;
                    let count = view.iter().filter(|v| !v.is_finite()).count();
                    return Err(crate::error::AutodiffError::NonFiniteCheckpoint {
                        name: name.clone(),
                        count,
                    });
                }
                var.set(tensor)?;
            }
        }
        Ok(())
    }

    /// Export all named variables as a `HashMap<String, DynTensor>`.
    ///
    /// Useful for custom serialization or checkpointing outside safetensors.
    pub fn to_tensors(&self) -> Result<HashMap<String, DynTensor>> {
        self.vars
            .iter()
            .map(|(name, var)| Ok((name.clone(), var.data()?)))
            .collect()
    }
}

impl Default for VarMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "var_map_tests.rs"]
mod tests;
