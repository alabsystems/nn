// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_hook_handle_deactivates_on_drop() {
    let active = Arc::new(AtomicBool::new(true));
    let handle = HookHandle::new(active.clone());
    assert!(handle.is_active());
    assert!(active.load(Ordering::Relaxed));

    drop(handle);
    assert!(!active.load(Ordering::Relaxed));
}

#[test]
fn test_hook_handle_manual_deactivate() {
    let active = Arc::new(AtomicBool::new(true));
    let handle = HookHandle::new(active.clone());
    assert!(handle.is_active());

    handle.deactivate();
    assert!(!handle.is_active());
    assert!(!active.load(Ordering::Relaxed));
}

#[test]
fn test_hook_ids_are_unique() {
    let a1 = Arc::new(AtomicBool::new(true));
    let a2 = Arc::new(AtomicBool::new(true));
    let h1 = HookHandle::new(a1);
    let h2 = HookHandle::new(a2);
    assert_ne!(h1.id(), h2.id());
}

#[test]
fn test_activation_capture_fields() {
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::Device;

    let tensor = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let capture = ActivationCapture {
        layer_name: "layer_0".to_string(),
        activation: tensor,
    };
    assert_eq!(capture.layer_name, "layer_0");
    assert_eq!(capture.activation.dims(), &[3]);
}
