// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Hooked module wrapper for activation capture during forward passes.
//!
//! [`HookedModule`] wraps any [`TrainableModule`] to capture activations
//! after each forward pass. This enables causal tracing workflows:
//!
//! 1. Run a "clean" forward pass, capture activations at each layer
//! 2. Run a "corrupted" forward pass with patched activations
//! 3. Compare outputs to identify which layers influence the target behavior
//!
//! # Example
//!
//! ```no_run
//! use nn_autodiff::hooked::HookedModule;
//! use nn_autodiff::trainable::{TrainableLinear, TrainableModule};
//! use nn_autodiff::TrackedTensor;
//! use nn_core::{DynTensor, Device};
//! use std::sync::Arc;
//!
//! # fn main() -> std::result::Result<(), nn_autodiff::AutodiffError> {
//! let layer = TrainableLinear::new(4, 3, true)?;
//! let mut hooked = HookedModule::new(layer, "linear_0".to_string());
//!
//! let handle = hooked.activate_hooks();
//! let x = Arc::new(TrackedTensor::from_tensor(
//!     DynTensor::from_vec(vec![1.0; 8], &[2, 4], &Device::Cpu)?,
//! ));
//! let y = hooked.forward(&x)?;
//!
//! assert_eq!(hooked.capture_count(), 1);
//! drop(handle); // deactivates hooks
//! # Ok(())
//! # }
//! ```

use crate::error::Result;
use crate::hooks::{ActivationCapture, HookHandle, HookableModule};
use crate::tracked::TrackedTensor;
use crate::trainable::TrainableModule;
use crate::var::Var;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A wrapper around any [`TrainableModule`] that captures activations.
///
/// When hooks are active (via [`activate_hooks`](Self::activate_hooks)),
/// each call to [`forward`](TrainableModule::forward) records the output
/// activation alongside the layer name.
///
/// The inner module is accessible via [`inner`](Self::inner) and
/// [`inner_mut`](Self::inner_mut) for direct access when needed.
///
/// Uses `RefCell` for interior mutability on captures during `forward()`,
/// since `TrainableModule::forward` takes `&self`.
pub struct HookedModule<M: TrainableModule> {
    inner: M,
    layer_name: String,
    hooks_active: Arc<AtomicBool>,
    captures: RefCell<Vec<ActivationCapture>>,
}

impl<M: TrainableModule> HookedModule<M> {
    /// Wrap a trainable module with activation capture.
    ///
    /// `layer_name` identifies this module in attribution results.
    pub fn new(inner: M, layer_name: String) -> Self {
        Self {
            inner,
            layer_name,
            hooks_active: Arc::new(AtomicBool::new(false)),
            captures: RefCell::new(Vec::new()),
        }
    }

    /// Activate hooks and return a handle.
    ///
    /// Dropping the returned [`HookHandle`] deactivates hooks.
    pub fn activate_hooks(&mut self) -> HookHandle {
        self.hooks_active.store(true, Ordering::Relaxed);
        HookHandle::new(Arc::clone(&self.hooks_active))
    }

    /// Reference to the inner module.
    #[must_use]
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Mutable reference to the inner module.
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }

    /// The name assigned to this hooked module.
    #[must_use]
    pub fn layer_name(&self) -> &str {
        &self.layer_name
    }

    /// Number of captures recorded so far.
    #[must_use]
    pub fn capture_count(&self) -> usize {
        self.captures.borrow().len()
    }
}

impl<M: TrainableModule> TrainableModule for HookedModule<M> {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let output = self.inner.forward(x)?;

        if self.hooks_active.load(Ordering::Relaxed) {
            self.captures.borrow_mut().push(ActivationCapture {
                layer_name: self.layer_name.clone(),
                activation: output.tensor().clone(),
            });
        }

        Ok(output)
    }

    fn vars(&self) -> Vec<&Var> {
        self.inner.vars()
    }
}

impl<M: TrainableModule> HookableModule for HookedModule<M> {
    fn register_forward_hook(&mut self) -> HookHandle {
        self.activate_hooks()
    }

    fn clone_captures(&self) -> Vec<ActivationCapture> {
        self.captures.borrow().clone()
    }

    fn capture_count(&self) -> usize {
        self.captures.borrow().len()
    }

    fn clear_captures(&mut self) {
        self.captures.borrow_mut().clear();
    }
}

impl<M: TrainableModule> HookedModule<M> {
    /// Access captured activations via a closure (RefCell-safe).
    ///
    /// This is the preferred API for reading captures, since
    /// `captured_activations()` cannot return a borrow through RefCell.
    pub fn with_captures<R>(&self, f: impl FnOnce(&[ActivationCapture]) -> R) -> R {
        let guard = self.captures.borrow();
        f(&guard)
    }

    /// Clone all captured activations out of the RefCell.
    ///
    /// Convenience method when you need owned data.
    pub fn clone_captures(&self) -> Vec<ActivationCapture> {
        self.captures.borrow().clone()
    }
}

impl<M: TrainableModule + std::fmt::Debug> std::fmt::Debug for HookedModule<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookedModule")
            .field("inner", &self.inner)
            .field("layer_name", &self.layer_name)
            .field("hooks_active", &self.hooks_active.load(Ordering::Relaxed))
            .field("captures_count", &self.captures.borrow().len())
            .finish()
    }
}

#[cfg(test)]
#[path = "hooked_tests.rs"]
mod tests;
