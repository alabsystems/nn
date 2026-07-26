// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Activation capture hooks for causal tracing and weight attribution.
//!
//! Provides the infrastructure for recording intermediate activations during
//! forward passes, enabling causal tracing workflows (activation patching,
//! gradient-based attribution) for verified weight surgery.
//!
//! # Architecture
//!
//! - [`ActivationCapture`]: A recorded activation from a named layer.
//! - [`HookHandle`]: RAII guard that deactivates hooks on drop.
//! - [`HookableModule`]: Trait for modules that support activation capture.

use nn_core::dyn_tensor::DynTensor;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Unique identifier for a hook registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HookId(u64);

static NEXT_HOOK_ID: AtomicU64 = AtomicU64::new(0);

impl HookId {
    fn next() -> Self {
        Self(NEXT_HOOK_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// A captured activation from a named layer during a forward pass.
///
/// Stores both the layer name (for identification in attribution results)
/// and the output tensor value at that layer.
#[derive(Debug, Clone)]
pub struct ActivationCapture {
    /// Name identifying which layer produced this activation.
    pub layer_name: String,
    /// The output tensor captured from the layer.
    pub activation: DynTensor,
}

/// RAII guard that controls hook lifetime.
///
/// When dropped, sets the associated `active` flag to `false`, stopping
/// further activation capture. This ensures hooks don't leak if the
/// caller forgets to explicitly deactivate them.
///
/// # Example
///
/// ```no_run
/// # use nn_autodiff::hooks::HookHandle;
/// # use std::sync::Arc;
/// # use std::sync::atomic::AtomicBool;
/// let active = Arc::new(AtomicBool::new(true));
/// let handle = HookHandle::new(active.clone());
/// assert!(active.load(std::sync::atomic::Ordering::Relaxed));
/// drop(handle);
/// assert!(!active.load(std::sync::atomic::Ordering::Relaxed));
/// ```
pub struct HookHandle {
    id: HookId,
    active: Arc<AtomicBool>,
}

impl HookHandle {
    /// Create a new hook handle tied to the given active flag.
    pub fn new(active: Arc<AtomicBool>) -> Self {
        Self {
            id: HookId::next(),
            active,
        }
    }

    /// Unique identifier for this hook registration.
    #[must_use]
    pub fn id(&self) -> HookId {
        self.id
    }

    /// Whether this hook is currently active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Manually deactivate this hook (also happens on drop).
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::Relaxed);
    }
}

impl Drop for HookHandle {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for HookHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookHandle")
            .field("id", &self.id)
            .field("active", &self.is_active())
            .finish()
    }
}

/// Trait for modules that support activation capture hooks.
///
/// Implementing types can register forward hooks to capture intermediate
/// activations during forward passes, enabling causal tracing and
/// gradient-based weight attribution.
pub trait HookableModule {
    /// Register a forward hook that captures activations.
    ///
    /// Returns a [`HookHandle`] — drop the handle to deactivate the hook.
    fn register_forward_hook(&mut self) -> HookHandle;

    /// Clone all activations captured since the last call to
    /// [`clear_captures`](Self::clear_captures).
    ///
    /// Returns owned data because implementations may use interior mutability
    /// (`RefCell`) that prevents returning borrowed slices.
    fn clone_captures(&self) -> Vec<ActivationCapture>;

    /// Number of captured activations.
    fn capture_count(&self) -> usize;

    /// Clear all captured activations.
    ///
    /// Call between forward passes to avoid unbounded memory growth.
    fn clear_captures(&mut self);
}

#[cfg(test)]
#[path = "hooks_tests.rs"]
mod tests;
