// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::hooks::HookableModule;
use crate::trainable::TrainableLinear;
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

fn cpu() -> Device {
    Device::Cpu
}

#[test]
fn test_hooked_forward_captures_activation() {
    let layer = TrainableLinear::new(4, 3, true).unwrap();
    let mut hooked = HookedModule::new(layer, "linear_0".to_string());

    let handle = hooked.activate_hooks();
    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0; 8], &[2, 4], &cpu()).unwrap(),
    ));
    let y = hooked.forward(&x).unwrap();

    assert_eq!(hooked.capture_count(), 1);
    let captures = hooked.clone_captures();
    assert_eq!(captures[0].layer_name, "linear_0");
    assert_eq!(captures[0].activation.dims(), y.tensor().dims());

    drop(handle);
}

#[test]
fn test_hooked_no_capture_without_activation() {
    let layer = TrainableLinear::new(4, 3, true).unwrap();
    let hooked = HookedModule::new(layer, "linear_0".to_string());

    // Hooks not activated
    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0; 8], &[2, 4], &cpu()).unwrap(),
    ));
    let _y = hooked.forward(&x).unwrap();

    assert_eq!(hooked.capture_count(), 0);
}

#[test]
fn test_hooked_multiple_forward_passes() {
    let layer = TrainableLinear::new(4, 3, false).unwrap();
    let mut hooked = HookedModule::new(layer, "fc".to_string());

    let _handle = hooked.activate_hooks();
    let x1 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0; 4], &[1, 4], &cpu()).unwrap(),
    ));
    let x2 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![2.0; 4], &[1, 4], &cpu()).unwrap(),
    ));

    let _y1 = hooked.forward(&x1).unwrap();
    let _y2 = hooked.forward(&x2).unwrap();

    assert_eq!(hooked.capture_count(), 2);
}

#[test]
fn test_hooked_clear_captures() {
    let layer = TrainableLinear::new(4, 3, true).unwrap();
    let mut hooked = HookedModule::new(layer, "fc".to_string());

    let _handle = hooked.activate_hooks();
    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0; 4], &[1, 4], &cpu()).unwrap(),
    ));
    let _y = hooked.forward(&x).unwrap();
    assert_eq!(hooked.capture_count(), 1);

    hooked.clear_captures();
    assert_eq!(hooked.capture_count(), 0);
}

#[test]
fn test_hooked_handle_deactivates_hooks() {
    let layer = TrainableLinear::new(4, 3, true).unwrap();
    let mut hooked = HookedModule::new(layer, "fc".to_string());

    let handle = hooked.activate_hooks();
    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0; 4], &[1, 4], &cpu()).unwrap(),
    ));
    let _y = hooked.forward(&x).unwrap();
    assert_eq!(hooked.capture_count(), 1);

    // Drop handle deactivates hooks
    drop(handle);

    // Further forward passes should not capture
    let _y2 = hooked.forward(&x).unwrap();
    assert_eq!(hooked.capture_count(), 1); // still 1, not 2
}

#[test]
fn test_hooked_vars_delegates() {
    let layer = TrainableLinear::new(4, 3, true).unwrap();
    let expected_vars = layer.vars().len();
    let hooked = HookedModule::new(layer, "fc".to_string());

    assert_eq!(hooked.vars().len(), expected_vars);
}

#[test]
fn test_hooked_inner_access() {
    let layer = TrainableLinear::new(4, 3, true).unwrap();
    let hooked = HookedModule::new(layer, "fc".to_string());

    assert_eq!(hooked.layer_name(), "fc");
    // inner() returns the wrapped TrainableLinear
    let _inner = hooked.inner();
}

#[test]
fn test_hooked_with_captures_closure() {
    let layer = TrainableLinear::new(4, 3, true).unwrap();
    let mut hooked = HookedModule::new(layer, "fc".to_string());

    let _handle = hooked.activate_hooks();
    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0; 4], &[1, 4], &cpu()).unwrap(),
    ));
    let _y = hooked.forward(&x).unwrap();

    let name = hooked.with_captures(|caps| caps[0].layer_name.clone());
    assert_eq!(name, "fc");
}

#[test]
fn test_hooked_hookable_module_trait() {
    let layer = TrainableLinear::new(4, 3, true).unwrap();
    let mut hooked = HookedModule::new(layer, "fc".to_string());

    // Use HookableModule trait methods
    let _handle = hooked.register_forward_hook();
    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0; 4], &[1, 4], &cpu()).unwrap(),
    ));
    let _y = hooked.forward(&x).unwrap();

    assert_eq!(HookableModule::capture_count(&hooked), 1);
    let captures = HookableModule::clone_captures(&hooked);
    assert_eq!(captures.len(), 1);

    hooked.clear_captures();
    assert_eq!(HookableModule::capture_count(&hooked), 0);
}

#[test]
fn test_backward_for_vars_selective() {
    use crate::grad::backward_for_vars;

    let w1 = Var::new(DynTensor::from_vec(vec![1.0, 0.5, 0.5, 1.0], &[2, 2], &cpu()).unwrap());
    let w2 = Var::new(DynTensor::from_vec(vec![0.5, 1.5, 2.5, 3.5], &[2, 2], &cpu()).unwrap());

    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0, 1.0], &[1, 2], &cpu()).unwrap(),
    ));

    let t1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let t2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let wt1 = t1.transpose(0, 1).unwrap();
    let h = x.matmul(&wt1).unwrap();
    let wt2 = t2.transpose(0, 1).unwrap();
    let y = h.matmul(&wt2).unwrap();
    let loss = y.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();

    // Only get gradient for w1
    let grads = backward_for_vars(&loss, &[&w1]).unwrap();
    assert!(grads.get(&w1).is_some());
    assert!(grads.get(&w2).is_none());
    assert_eq!(grads.var_count(), 1);
}

#[test]
fn test_grad_store_retain_only() {
    use crate::grad::backward;

    let w1 = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap());
    let w2 = Var::new(DynTensor::from_vec(vec![3.0, 4.0], &[1, 2], &cpu()).unwrap());

    let t1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let t2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let sum = t1.add(&t2).unwrap();
    let loss = sum.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();

    let mut grads = backward(&loss).unwrap();
    assert_eq!(grads.var_count(), 2);

    grads.retain_only(&[&w2]);
    assert_eq!(grads.var_count(), 1);
    assert!(grads.get(&w1).is_none());
    assert!(grads.get(&w2).is_some());
}
