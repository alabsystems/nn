// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Gradient storage and backward pass.
//!
//! [`GradStore`] accumulates gradients for both variables (by [`VarId`]) and
//! intermediate nodes (by [`NodeId`]). [`backward()`] performs reverse-mode
//! automatic differentiation via topological sort of the computation graph.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;

use crate::backward_rules::backward_op;
use crate::error::{AutodiffError, Result};
use crate::tracked::{NodeId, TrackedTensor};
use crate::var::{Var, VarId};

/// Accumulated gradients from backward pass.
///
/// After `backward()`, retrieve gradients for trainable variables via
/// [`GradStore::get()`].
#[derive(Debug)]
pub struct GradStore {
    var_grads: HashMap<VarId, DynTensor>,
    node_grads: HashMap<NodeId, DynTensor>,
}

impl GradStore {
    /// Create an empty gradient store.
    pub fn new() -> Self {
        Self {
            var_grads: HashMap::new(),
            node_grads: HashMap::new(),
        }
    }

    /// Get gradient for a variable.
    pub fn get(&self, var: &Var) -> Option<&DynTensor> {
        self.var_grads.get(&var.id())
    }

    /// Get gradient by VarId.
    pub fn get_id(&self, id: &VarId) -> Option<&DynTensor> {
        self.var_grads.get(id)
    }

    /// Iterate over all variable gradients.
    pub fn var_grads(&self) -> impl Iterator<Item = (&VarId, &DynTensor)> {
        self.var_grads.iter()
    }

    /// Mutably iterate over all variable gradients (e.g., for unscaling).
    pub fn var_grads_mut(&mut self) -> impl Iterator<Item = (&VarId, &mut DynTensor)> {
        self.var_grads.iter_mut()
    }

    /// Accumulate gradient for a variable.
    ///
    /// Validates that the gradient shape matches any existing gradient for this
    /// variable. A shape mismatch indicates a backward rule bug (producing
    /// gradients with wrong shape that would silently broadcast during add).
    ///
    /// Uses `add_assign` for in-place accumulation when the existing gradient
    /// buffer is uniquely owned (common case: no shared views). This avoids
    /// allocating a new tensor on every fan-in add — critical for residual
    /// connections and skip connections where multiple backward paths
    /// accumulate into the same variable.
    pub(crate) fn accumulate_var(&mut self, id: VarId, grad: &DynTensor) -> Result<()> {
        if let Some(existing) = self.var_grads.get_mut(&id) {
            if existing.dims() != grad.dims() {
                return Err(AutodiffError::ShapeMismatch {
                    expected: existing.dims().to_vec(),
                    got: grad.dims().to_vec(),
                });
            }
            existing.add_assign(grad)?;
        } else {
            self.var_grads.insert(id, grad.clone());
        }
        Ok(())
    }

    /// Accumulate gradient for an intermediate node.
    ///
    /// Validates shape consistency on accumulation (same as `accumulate_var`).
    ///
    /// Uses `add_assign` for in-place accumulation when possible (see
    /// [`accumulate_var`](Self::accumulate_var) for details).
    pub(crate) fn accumulate_node(&mut self, id: NodeId, grad: &DynTensor) -> Result<()> {
        if let Some(existing) = self.node_grads.get_mut(&id) {
            if existing.dims() != grad.dims() {
                return Err(AutodiffError::ShapeMismatch {
                    expected: existing.dims().to_vec(),
                    got: grad.dims().to_vec(),
                });
            }
            existing.add_assign(grad)?;
        } else {
            self.node_grads.insert(id, grad.clone());
        }
        Ok(())
    }

    /// Get gradient for an intermediate node.
    pub(crate) fn get_node(&self, id: &NodeId) -> Option<&DynTensor> {
        self.node_grads.get(id)
    }

    /// Retain only gradients for the specified variables, dropping all others.
    ///
    /// Useful for selective attribution: run full backward, then filter to
    /// only the weight matrices you care about. Reduces memory when the
    /// computation graph has many variables but you only need gradients for
    /// a few target weights.
    pub fn retain_only(&mut self, target_vars: &[&Var]) {
        let target_ids: HashSet<VarId> = target_vars.iter().map(|v| v.id()).collect();
        self.var_grads.retain(|id, _| target_ids.contains(id));
    }

    /// Number of variable gradients stored.
    #[must_use]
    pub fn var_count(&self) -> usize {
        self.var_grads.len()
    }
}

impl Default for GradStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Reverse-mode automatic differentiation.
///
/// Computes gradients of `loss` with respect to all trainable variables
/// reachable in the computation graph. `loss` must be a scalar tensor.
///
/// # Example
/// ```no_run
/// use nn_autodiff::{Var, TrackedTensor, backward};
/// use nn_core::dyn_tensor::DynTensor;
/// use nn_core::Device;
/// use std::sync::Arc;
///
/// # fn main() -> std::result::Result<(), nn_autodiff::AutodiffError> {
/// let x = Var::new(DynTensor::from_vec(vec![3.0], &[1], &Device::Cpu)?);
/// let t = Arc::new(TrackedTensor::from_var(&x)?);
/// let y = t.sqr()?; // y = x^2
/// let grads = backward(&y)?;
/// // dy/dx = 2x = 6.0
/// let grad = grads.get(&x).unwrap();
/// # Ok(())
/// # }
/// ```
#[must_use = "backward() returns the gradient store; discarding it silently loses all computed gradients"]
pub fn backward(loss: &Arc<TrackedTensor>) -> Result<GradStore> {
    // Validate: loss must be a scalar
    if loss.tensor().numel() != 1 {
        return Err(AutodiffError::NonScalarLoss {
            shape: loss.tensor().dims().to_vec(),
        });
    }

    // Validate: loss value must be finite (not NaN or Inf).
    // A non-finite loss produces garbage gradients for all variables.
    // Catching it here avoids wasting the full backward pass computation.
    if loss.tensor().any_non_finite()? {
        return Err(AutodiffError::NonFiniteLoss);
    }

    // Topological sort
    let sorted = topological_sort(loss);

    // Initialize gradient store with d(loss)/d(loss) = 1
    let mut grads = GradStore::new();
    let ones = DynTensor::ones(
        loss.tensor().dims(),
        loss.tensor().dtype(),
        &loss.tensor().device(),
    )?;
    grads.accumulate_node(loss.node_id(), &ones)?;

    // Reverse pass: iterate from loss toward leaves (reverse of DFS post-order)
    for node in sorted.iter().rev() {
        let grad = match grads.get_node(&node.node_id()) {
            Some(g) => g.clone(),
            None => continue, // no gradient flows to this node
        };

        // Accumulate into var_grads if this is a Var
        if let Some(var_id) = node.var_id() {
            grads.accumulate_var(var_id, &grad)?;
        }

        // Propagate gradients through the operation
        if let Some(op) = node.op() {
            backward_op(op, &grad, &mut grads)?;
        }
    }

    Ok(grads)
}

/// Selective backward: compute gradients only for specified variables.
///
/// Runs full backward pass then filters the gradient store to retain only
/// the target variables. This is a convenience wrapper — the computational
/// cost is the same as `backward()` since the full graph must be traversed.
///
/// Used for gradient-based weight attribution: identify which weight matrices
/// most influence a target metric by examining their gradient magnitudes.
///
/// # Example
///
/// ```no_run
/// use nn_autodiff::{Var, TrackedTensor, backward_for_vars};
/// use nn_core::{DynTensor, Device};
/// use std::sync::Arc;
///
/// # fn main() -> std::result::Result<(), nn_autodiff::AutodiffError> {
/// let w1 = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu)?);
/// let w2 = Var::new(DynTensor::from_vec(vec![0.5, 1.5, 2.5, 3.5], &[2, 2], &Device::Cpu)?);
/// let x = Arc::new(TrackedTensor::from_tensor(
///     DynTensor::from_vec(vec![1.0, 1.0], &[1, 2], &Device::Cpu)?,
/// ));
///
/// let t1 = Arc::new(TrackedTensor::from_var(&w1)?);
/// let t2 = Arc::new(TrackedTensor::from_var(&w2)?);
/// let h = x.matmul(&t1.transpose(0, 1)?)?;
/// let y = h.matmul(&t2.transpose(0, 1)?)?;
/// let loss = y.sum_keepdim(1)?.sum_keepdim(0)?;
///
/// // Only get gradient for w1
/// let grads = backward_for_vars(&loss, &[&w1])?;
/// assert!(grads.get(&w1).is_some());
/// assert!(grads.get(&w2).is_none());
/// # Ok(())
/// # }
/// ```
#[must_use = "backward_for_vars() returns the gradient store; discarding it silently loses all computed gradients"]
pub fn backward_for_vars(loss: &Arc<TrackedTensor>, target_vars: &[&Var]) -> Result<GradStore> {
    let mut grads = backward(loss)?;
    grads.retain_only(target_vars);
    Ok(grads)
}

/// Topological sort via iterative DFS (returns nodes in post-order).
///
/// Uses an explicit stack instead of recursion to avoid stack overflow on deep
/// computation graphs (e.g., 1000+ sequential operations in RNNs or long
/// autoregressive decode loops).
fn topological_sort(root: &Arc<TrackedTensor>) -> Vec<Arc<TrackedTensor>> {
    let mut visited = HashSet::new();
    let mut sorted = Vec::new();

    // Each stack frame: (node, children_pushed).
    // On first visit (children_pushed=false), push children then revisit with true.
    // On second visit (children_pushed=true), emit the node to sorted.
    let mut stack: Vec<(Arc<TrackedTensor>, bool)> = vec![(Arc::clone(root), false)];

    while let Some((node, children_pushed)) = stack.pop() {
        let id = node.node_id().as_u64();

        if children_pushed {
            // Second visit: all children processed, emit this node
            sorted.push(node);
            continue;
        }

        if !visited.insert(id) {
            continue;
        }

        // First visit: push self again (for post-order emit), then push children
        stack.push((Arc::clone(&node), true));

        if let Some(op) = node.op() {
            // Push children in reverse order so they are processed left-to-right
            let inputs = op_inputs(op);
            for input in inputs.into_iter().rev() {
                if !visited.contains(&input.node_id().as_u64()) {
                    stack.push((input, false));
                }
            }
        }
    }

    sorted
}

#[path = "grad_op_inputs.rs"]
mod op_inputs_mod;
use op_inputs_mod::op_inputs;

#[cfg(test)]
#[path = "grad_test_helpers.rs"]
mod test_helpers;

#[cfg(test)]
#[path = "grad_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "grad_tests_broadcast.rs"]
mod broadcast_tests;

#[cfg(test)]
#[path = "grad_tests_cross_entropy.rs"]
mod cross_entropy_tests;

#[cfg(test)]
#[path = "grad_tests_new_ops.rs"]
mod new_ops_tests;

#[cfg(test)]
#[path = "loss_grad_target_fd_tests.rs"]
mod loss_grad_target_fd_tests;

#[cfg(test)]
#[path = "grad_tests_new_ops_fd.rs"]
mod new_ops_fd_tests;

#[cfg(test)]
#[path = "grad_tests_pool_fd.rs"]
mod pool_fd_tests;

#[cfg(test)]
#[path = "grad_tests_stress.rs"]
mod stress_tests;

#[cfg(test)]
#[path = "grad_tests_key_ops_fd.rs"]
mod key_ops_fd_tests;

#[cfg(test)]
#[path = "autodiff_wave11_tests.rs"]
mod wave11_tests;

#[cfg(test)]
#[path = "grad_tests_gelu_erf.rs"]
mod gelu_erf_tests;

#[cfg(test)]
#[path = "grad_tests_higher_order.rs"]
mod higher_order_tests;
