// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Hierarchical scoped weight loader for model porting.
//!
//! [`VarBuilder`] provides candle-compatible hierarchical weight loading via
//! `.pp(prefix)` scoping and `.get(dims, name)` retrieval, returning
//! [`DynTensor`]. Thread-safe (`Arc<dyn TensorBackend>`), cheap to clone.
//!
//! Built-in backends: [`ZerosBackend`] (tests), [`TensorMapBackend`] (manual).
//! Production backend: `SafeTensorsBackend` in nn-metal (wraps `WeightMap`).

use crate::dyn_tensor::DynTensor;
use crate::mixed_precision::MixedPrecisionPolicy;
use crate::{DType, Device, Result};
use std::collections::HashMap;
use std::sync::Arc;

// -- TensorBackend trait ------------------------------------------------------

/// Backend trait for VarBuilder — provides tensor retrieval by name.
///
/// Implementations handle actual storage (safetensors, in-memory, zeros).
/// nn-core provides [`ZerosBackend`] and [`TensorMapBackend`].
/// nn-metal provides `SafeTensorsBackend` (wraps `WeightMap`).
pub trait TensorBackend: Send + Sync {
    /// Load a tensor by name, validating shape matches expected dims.
    fn get(&self, dims: &[usize], name: &str, dtype: DType, device: &Device) -> Result<DynTensor>;

    /// Load a tensor by name without shape validation.
    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> Result<DynTensor>;

    /// Check if a tensor with this name exists.
    fn contains_tensor(&self, name: &str) -> bool;

    /// List all tensor names available in this backend.
    ///
    /// Used for weight discovery — e.g., building rename maps from HuggingFace
    /// safetensors to NN model conventions. Returns empty for backends with
    /// infinite key spaces (like [`ZerosBackend`]).
    fn tensor_names(&self) -> Vec<String> {
        Vec::new()
    }
}

// -- VarBuilder struct --------------------------------------------------------

/// Hierarchical scoped weight loader.
///
/// Matches candle's `VarBuilder` API. Thread-safe (`Arc<dyn TensorBackend>`).
/// `.pp()` is a cheap clone (Arc + Vec push).
///
/// # Example
///
/// ```no_run
/// use nn_core::var_builder::VarBuilder;
/// use nn_core::{DType, Device};
///
/// let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
/// let encoder_vb = vb.pp("encoder");
/// let weight = encoder_vb.get(&[512, 256], "weight").expect("load weight");
/// ```
/// Name mapping function type for weight key transformation.
///
/// Applied to the fully-resolved key (after `pp()` prefix + tensor name)
/// before backend lookup. Enables loading weights from checkpoints that
/// use different naming conventions.
pub type NameMapFn = Arc<dyn Fn(&str) -> String + Send + Sync>;

#[derive(Clone)]
pub struct VarBuilder {
    backend: Arc<dyn TensorBackend>,
    path: Vec<String>,
    dtype: DType,
    device: Device,
    precision_policy: Option<MixedPrecisionPolicy>,
    name_map: Option<NameMapFn>,
}

impl VarBuilder {
    /// Create from a backend with explicit dtype and device.
    pub fn from_backend(backend: Arc<dyn TensorBackend>, dtype: DType, device: Device) -> Self {
        Self {
            backend,
            path: Vec::new(),
            dtype,
            device,
            precision_policy: None,
            name_map: None,
        }
    }

    /// Push a prefix segment. Returns a new VarBuilder sharing the same backend.
    ///
    /// ```no_run
    /// # use nn_core::var_builder::VarBuilder;
    /// # use nn_core::{DType, Device};
    /// # let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    /// let encoder_vb = vb.pp("encoder");
    /// let weight = encoder_vb.get(&[512, 256], "weight").expect("load weight");
    /// // resolves key: "encoder.weight"
    /// ```
    #[must_use]
    pub fn pp<S: ToString>(&self, prefix: S) -> Self {
        let prefix_str = prefix.to_string();
        let mut path = self.path.clone();
        // Skip empty prefixes to avoid ".weight" or "encoder..weight" keys.
        if !prefix_str.is_empty() {
            path.push(prefix_str);
        }
        Self {
            backend: Arc::clone(&self.backend),
            path,
            dtype: self.dtype,
            device: self.device,
            precision_policy: self.precision_policy.clone(),
            name_map: self.name_map.clone(),
        }
    }

    /// Load a tensor by name, prefixed by the current path.
    ///
    /// Shape is validated: if the loaded tensor's shape doesn't match `dims`,
    /// returns `TensorError::ShapeMismatch`.
    pub fn get(&self, dims: &[usize], name: &str) -> Result<DynTensor> {
        let full_name = self.resolve_name(name);
        self.backend.get(dims, &full_name, self.dtype, &self.device)
    }

    /// Load a tensor by name without shape validation.
    pub fn get_unchecked(&self, name: &str) -> Result<DynTensor> {
        let full_name = self.resolve_name(name);
        self.backend
            .get_unchecked(&full_name, self.dtype, &self.device)
    }

    /// Check if a tensor exists under the current prefix.
    pub fn contains_tensor(&self, name: &str) -> bool {
        let full_name = self.resolve_name(name);
        self.backend.contains_tensor(&full_name)
    }

    /// Attach a name mapping function for weight key transformation.
    ///
    /// The function receives the fully-resolved key (after prefix + tensor name
    /// concatenation) and returns the key to look up in the backend. Propagates
    /// through `.pp()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use nn_core::var_builder::VarBuilder;
    /// # use nn_core::{DType, Device};
    /// let vb = VarBuilder::zeros(DType::F32, &Device::Cpu)
    ///     .with_name_mapping(|name| name.replace("layernorm_before", "layer_norm1"));
    /// ```
    #[must_use]
    pub fn with_name_mapping<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        self.name_map = Some(Arc::new(f));
        self
    }

    /// Attach a [`WeightNameMapper`] for structured weight key transformation.
    ///
    /// This is the trait-based alternative to [`with_name_mapping`]. The mapper
    /// receives the fully-resolved NN key and returns the checkpoint key.
    /// Propagates through `.pp()`.
    ///
    /// Use [`HfToNnMapper`] for composable rule-based mapping with prefix,
    /// segment, and suffix rules. See the [`weight_name_mapper`] module for details.
    ///
    /// [`with_name_mapping`]: Self::with_name_mapping
    /// [`HfToNnMapper`]: crate::var_builder::HfToNnMapper
    /// [`weight_name_mapper`]: crate::var_builder::weight_name_mapper
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use nn_core::var_builder::{VarBuilder, HfToNnMapper};
    /// # use nn_core::{DType, Device};
    /// let mapper = HfToNnMapper::new()
    ///     .with_prefix_rule("model.layers", "encoder.layer")
    ///     .with_segment_rule("self_attn", "attention");
    /// let vb = VarBuilder::zeros(DType::F32, &Device::Cpu)
    ///     .with_weight_name_mapper(mapper);
    /// ```
    #[must_use]
    pub fn with_weight_name_mapper(self, mapper: impl WeightNameMapper + 'static) -> Self {
        self.with_name_mapping(move |name| mapper.map_name(name))
    }

    /// Attach prefix-based name mapping for weight key transformation.
    ///
    /// Each pair `(from, to)` replaces a matching prefix. First match wins.
    /// Propagates through `.pp()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use nn_core::var_builder::VarBuilder;
    /// # use nn_core::{DType, Device};
    /// let vb = VarBuilder::zeros(DType::F32, &Device::Cpu)
    ///     .with_prefix_mapping(&[
    ///         ("encoder.layer", "vision_model.encoder.layers"),
    ///     ]);
    /// ```
    #[must_use]
    pub fn with_prefix_mapping(self, prefix_pairs: &[(&str, &str)]) -> Self {
        let pairs: Vec<(String, String)> = prefix_pairs
            .iter()
            .map(|(from, to)| (from.to_string(), to.to_string()))
            .collect();
        self.with_name_mapping(move |name| {
            for (from, to) in &pairs {
                if let Some(rest) = name.strip_prefix(from.as_str()) {
                    return format!("{to}{rest}");
                }
            }
            name.to_string()
        })
    }

    /// Attach an exact-key rename map for weight key transformation.
    ///
    /// Keys in `rename_map` are the NN model names (what the model code
    /// requests via `pp().get()`); values are the checkpoint names (what the
    /// safetensors file contains). Keys not in the map pass through unchanged.
    ///
    /// This is the recommended approach for loading HuggingFace weights into
    /// NN models where naming conventions differ.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use nn_core::var_builder::VarBuilder;
    /// # use nn_core::{DType, Device};
    /// # use std::collections::HashMap;
    /// let rename = HashMap::from([
    ///     ("encoder.attn.q.weight".into(), "model.encoder.layers.0.self_attn.q_proj.weight".into()),
    /// ]);
    /// let vb = VarBuilder::zeros(DType::F32, &Device::Cpu)
    ///     .with_rename_map(rename);
    /// ```
    #[must_use]
    pub fn with_rename_map(self, rename_map: HashMap<String, String>) -> Self {
        self.with_name_mapping(move |name| {
            rename_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.to_string())
        })
    }

    /// List all tensor names available in the backend.
    ///
    /// Returns raw backend keys (before any name mapping). When a rename map
    /// is applied via [`with_rename_map`] or [`with_name_mapping`], the returned
    /// names are the checkpoint names, not the NN model names. Use this for
    /// weight discovery — e.g., enumerate what keys a safetensors file contains
    /// to build a rename map. Returns empty for infinite-key backends like
    /// [`ZerosBackend`].
    ///
    /// [`with_rename_map`]: Self::with_rename_map
    /// [`with_name_mapping`]: Self::with_name_mapping
    #[must_use]
    pub fn tensor_names(&self) -> Vec<String> {
        self.backend.tensor_names()
    }

    /// Whether a name mapping is attached.
    #[must_use]
    pub fn has_name_mapping(&self) -> bool {
        self.name_map.is_some()
    }

    /// Current dtype.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Current device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Return a new VarBuilder with a different dtype.
    #[must_use]
    pub fn to_dtype(&self, dtype: DType) -> Self {
        Self {
            dtype,
            ..self.clone()
        }
    }

    /// Return a new VarBuilder with a different device.
    #[must_use]
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            device,
            ..self.clone()
        }
    }

    /// Current prefix as dot-separated string.
    #[must_use]
    pub fn prefix(&self) -> String {
        self.path.join(".")
    }

    /// Attach a mixed-precision policy. The policy propagates through `.pp()`
    /// so all sub-builders inherit it.
    #[must_use]
    pub fn with_precision_policy(mut self, policy: MixedPrecisionPolicy) -> Self {
        self.precision_policy = Some(policy);
        self
    }

    /// Get the effective weight dtype for loading.
    ///
    /// When a precision policy is set, returns the policy's `weight_dtype`.
    /// Without a policy, falls back to the VarBuilder's dtype.
    #[must_use]
    pub fn effective_weight_dtype(&self) -> DType {
        self.precision_policy
            .as_ref()
            .map(|p| p.weight_dtype)
            .unwrap_or(self.dtype)
    }

    /// Get the attached precision policy, if any.
    #[must_use]
    pub fn precision_policy(&self) -> Option<&MixedPrecisionPolicy> {
        self.precision_policy.as_ref()
    }

    // -- Private --------------------------------------------------------------

    fn resolve_name(&self, tensor_name: &str) -> String {
        let name = if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            format!("{}.{}", self.path.join("."), tensor_name)
        };
        match &self.name_map {
            Some(f) => f(&name),
            None => name,
        }
    }
}

impl AsRef<Self> for VarBuilder {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl std::fmt::Debug for VarBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarBuilder")
            .field("prefix", &self.prefix())
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

// -- Built-in backends --------------------------------------------------------

// Convenience constructors and built-in backends.
mod backends;

pub use backends::{TensorMapBackend, ZerosBackend};

// -- Weight name mapper -------------------------------------------------------

// Pluggable weight name translation for HF-to-NN import.
mod weight_name_mapper;

pub use weight_name_mapper::{verify_mapper_coverage, HfToNnMapper, WeightNameMapper};

#[cfg(kani)]
mod kani_proofs;

#[cfg(test)]
mod tests;
