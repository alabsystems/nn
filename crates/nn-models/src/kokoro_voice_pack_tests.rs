// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nn_core::{Device, DynTensor};

fn style_1d(val: f32) -> DynTensor {
    DynTensor::from_vec(vec![val; 256], &[256], &Device::Cpu).unwrap()
}

fn style_2d(val: f32) -> DynTensor {
    DynTensor::from_vec(vec![val; 256], &[1, 256], &Device::Cpu).unwrap()
}

#[test]
fn test_from_tensors_1d() {
    let mut tensors = HashMap::new();
    tensors.insert("af_heart".to_string(), style_1d(0.1));
    tensors.insert("am_adam".to_string(), style_1d(0.2));

    let pack = VoicePack::from_tensors(tensors, 128).unwrap();
    assert_eq!(pack.len(), 2);

    let heart = pack.get("af_heart").unwrap();
    assert_eq!(heart.dims(), &[1, 256]);

    let adam = pack.get("am_adam").unwrap();
    assert_eq!(adam.dims(), &[1, 256]);
}

#[test]
fn test_from_tensors_2d() {
    let mut tensors = HashMap::new();
    tensors.insert("af_bella".to_string(), style_2d(0.3));

    let pack = VoicePack::from_tensors(tensors, 128).unwrap();
    let bella = pack.get("af_bella").unwrap();
    assert_eq!(bella.dims(), &[1, 256]);
}

#[test]
fn test_wrong_shape_rejects() {
    let bad = DynTensor::from_vec(vec![0.0; 100], &[100], &Device::Cpu).unwrap();
    let mut tensors = HashMap::new();
    tensors.insert("bad_voice".to_string(), bad);

    let result = VoicePack::from_tensors(tensors, 128);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("bad_voice"),
        "error should name the voice: {msg}"
    );
}

#[test]
fn test_wrong_2d_shape_rejects() {
    let bad = DynTensor::from_vec(vec![0.0; 512], &[2, 256], &Device::Cpu).unwrap();
    let mut tensors = HashMap::new();
    tensors.insert("bad_batch".to_string(), bad);

    let result = VoicePack::from_tensors(tensors, 128);
    assert!(result.is_err());
}

#[test]
fn test_3d_tensor_rejects() {
    let bad = DynTensor::from_vec(vec![0.0; 256], &[1, 1, 256], &Device::Cpu).unwrap();
    let mut tensors = HashMap::new();
    tensors.insert("bad_rank".to_string(), bad);

    let result = VoicePack::from_tensors(tensors, 128);
    assert!(result.is_err());
}

#[test]
fn test_empty_pack() {
    let pack = VoicePack::empty(128);
    assert!(pack.is_empty());
    assert_eq!(pack.len(), 0);
    assert!(pack.get("nonexistent").is_none());
}

#[test]
fn test_get_or_err_missing() {
    let pack = VoicePack::empty(128);
    let result = pack.get_or_err("missing");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("missing"),
        "error should name the voice: {msg}"
    );
}

#[test]
fn test_add_voice() {
    let mut pack = VoicePack::empty(128);
    pack.add_voice("af_heart", &style_1d(0.1)).unwrap();
    assert_eq!(pack.len(), 1);
    assert!(pack.get("af_heart").is_some());
}

#[test]
fn test_add_voice_wrong_shape() {
    let mut pack = VoicePack::empty(128);
    let bad = DynTensor::from_vec(vec![0.0; 100], &[100], &Device::Cpu).unwrap();
    let result = pack.add_voice("bad", &bad);
    assert!(result.is_err());
}

#[test]
fn test_sorted_voice_names() {
    let mut tensors = HashMap::new();
    tensors.insert("cm_zoe".to_string(), style_1d(0.1));
    tensors.insert("af_heart".to_string(), style_1d(0.2));
    tensors.insert("am_adam".to_string(), style_1d(0.3));

    let pack = VoicePack::from_tensors(tensors, 128).unwrap();
    assert_eq!(
        pack.sorted_voice_names(),
        vec!["af_heart", "am_adam", "cm_zoe"]
    );
}

#[test]
fn test_style_dim_accessor() {
    let pack = VoicePack::empty(64);
    assert_eq!(pack.style_dim(), 64);
}

#[test]
fn test_custom_style_dim() {
    // style_dim=4 → expected tensor length = 8
    let small = DynTensor::from_vec(vec![0.1; 8], &[8], &Device::Cpu).unwrap();
    let mut tensors = HashMap::new();
    tensors.insert("test_voice".to_string(), small);

    let pack = VoicePack::from_tensors(tensors, 4).unwrap();
    let voice = pack.get("test_voice").unwrap();
    assert_eq!(voice.dims(), &[1, 8]);
}

#[test]
fn test_safetensors_roundtrip() {
    // Build a voice pack, save to bytes, reload
    let mut tensors = HashMap::new();
    tensors.insert("af_heart".to_string(), style_1d(0.5));
    tensors.insert("am_adam".to_string(), style_1d(-0.3));

    // Save to safetensors bytes
    let pack = VoicePack::from_tensors(tensors, 128).unwrap();
    let bytes = nn_core::dyn_tensor::tensors_to_safetensors_bytes(&pack.voices).unwrap();

    // Reload
    let pack2 = VoicePack::load_from_bytes(&bytes, 128).unwrap();
    assert_eq!(pack2.len(), 2);
    assert!(pack2.get("af_heart").is_some());
    assert!(pack2.get("am_adam").is_some());

    // Verify values survive roundtrip
    let heart = pack2.get("af_heart").unwrap();
    let vals = heart.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 0.5).abs() < 1e-6);
}
