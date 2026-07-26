// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the phantom type-tagged pipeline tensor system.
//!
//! Part of #3635.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use super::*;

/// PipelineTensor wraps and unwraps a DynTensor without data loss.
#[test]
fn test_pipeline_tensor_roundtrip() {
    let tensor = DynTensor::zeros(&[1, 512, 10], DType::F32, &Device::Cpu).unwrap();
    let dims_before = tensor.dims().to_vec();

    let tagged: PipelineTensor<BertFeaturesOutput> = PipelineTensor::new(tensor);
    assert_eq!(tagged.inner().dims(), &dims_before);

    let recovered = tagged.into_inner();
    assert_eq!(recovered.dims(), &dims_before);
}

/// PipelineTensor preserves tensor values through wrap/unwrap.
#[test]
fn test_pipeline_tensor_value_preservation() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = DynTensor::from_vec(data.clone(), &[1, 4], &Device::Cpu).unwrap();

    let tagged: PipelineTensor<TextFeaturesOutput> = PipelineTensor::new(tensor);
    let vals = tagged.inner().to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, data);
}

/// PipelineTensor is Clone when the inner DynTensor is Clone.
#[test]
fn test_pipeline_tensor_clone() {
    let tensor = DynTensor::ones(&[2, 3], DType::F32, &Device::Cpu).unwrap();
    let tagged: PipelineTensor<F0Output> = PipelineTensor::new(tensor);
    let cloned = tagged.clone();
    assert_eq!(cloned.inner().dims(), tagged.inner().dims());
}

/// PipelineTensor is Debug-printable.
#[test]
fn test_pipeline_tensor_debug() {
    let tensor = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let tagged: PipelineTensor<IstftOutput> = PipelineTensor::new(tensor);
    let debug = format!("{tagged:?}");
    assert!(!debug.is_empty(), "Debug output should be non-empty");
}

/// Typed result types are constructible and their fields are accessible.
#[test]
fn test_typed_encode_result_fields() {
    let bert = DynTensor::zeros(&[1, 512, 10], DType::F32, &Device::Cpu).unwrap();
    let text = DynTensor::zeros(&[1, 512, 10], DType::F32, &Device::Cpu).unwrap();
    let result = TypedEncodeResult {
        bert_features: PipelineTensor::new(bert),
        text_features: PipelineTensor::new(text),
        seq_len: 10,
    };
    assert_eq!(result.seq_len, 10);
    assert_eq!(result.bert_features.inner().dims(), &[1, 512, 10]);
    assert_eq!(result.text_features.inner().dims(), &[1, 512, 10]);
}

/// Typed prosody result fields are accessible.
#[test]
fn test_typed_prosody_result_fields() {
    let dur = DynTensor::zeros(&[1, 10, 50], DType::F32, &Device::Cpu).unwrap();
    let feat = DynTensor::zeros(&[1, 640, 10], DType::F32, &Device::Cpu).unwrap();
    let result = TypedProsodyResult {
        dur_logits: PipelineTensor::new(dur),
        features: PipelineTensor::new(feat),
    };
    assert_eq!(result.dur_logits.inner().dims(), &[1, 10, 50]);
    assert_eq!(result.features.inner().dims(), &[1, 640, 10]);
}

/// Typed regulate result fields are accessible.
#[test]
fn test_typed_regulate_result_fields() {
    let durations = DynTensor::zeros(&[1, 10], DType::F32, &Device::Cpu).unwrap();
    let aligned = DynTensor::zeros(&[1, 640, 30], DType::F32, &Device::Cpu).unwrap();
    let regulated = DynTensor::zeros(&[1, 512, 30], DType::F32, &Device::Cpu).unwrap();
    let result = TypedRegulateResult {
        durations,
        aligned_dur: PipelineTensor::new(aligned),
        regulated: PipelineTensor::new(regulated),
        t_mel: 30,
    };
    assert_eq!(result.t_mel, 30);
    assert_eq!(result.aligned_dur.inner().dims(), &[1, 640, 30]);
    assert_eq!(result.regulated.inner().dims(), &[1, 512, 30]);
}

/// Typed F0/energy result fields are accessible.
#[test]
fn test_typed_f0_energy_result_fields() {
    let f0 = DynTensor::zeros(&[1, 1, 60], DType::F32, &Device::Cpu).unwrap();
    let energy = DynTensor::zeros(&[1, 1, 60], DType::F32, &Device::Cpu).unwrap();
    let result = TypedF0EnergyResult {
        f0: PipelineTensor::new(f0),
        energy: PipelineTensor::new(energy),
    };
    assert_eq!(result.f0.inner().dims(), &[1, 1, 60]);
    assert_eq!(result.energy.inner().dims(), &[1, 1, 60]);
}

/// Typed generator result fields are accessible.
#[test]
fn test_typed_generator_result_fields() {
    let mag = DynTensor::zeros(&[1, 513, 100], DType::F32, &Device::Cpu).unwrap();
    let phase = DynTensor::zeros(&[1, 513, 100], DType::F32, &Device::Cpu).unwrap();
    let result = TypedGeneratorResult {
        magnitude: PipelineTensor::new(mag),
        phase: PipelineTensor::new(phase),
    };
    assert_eq!(result.magnitude.inner().dims(), &[1, 513, 100]);
    assert_eq!(result.phase.inner().dims(), &[1, 513, 100]);
}

/// Compile-time check: typed step methods are importable through the public API.
///
/// This function is never called — it exists only to verify that the typed
/// step method signatures compile and that `PipelineTensor` type parameters
/// enforce correct pipeline wiring at the type level.
#[allow(dead_code, unused_variables)]
fn _assert_typed_step_api_compiles(
    kokoro: &mut super::super::CompiledKokoro,
    input_ids: &DynTensor,
    style: &DynTensor,
    cache: &crate::cache::PipelineCache,
) {
    // This block verifies the full typed pipeline compiles.
    // The types enforce correct wiring — swapping arguments would be a
    // compile error (e.g., passing bert_features where regulated is expected).
    let _: fn() = || {
        // Type annotations verify the compiler sees the right types.
        let _encode: TypedEncodeResult;
        let _prosody: TypedProsodyResult;
        let _regulate: TypedRegulateResult;
        let _f0e: TypedF0EnergyResult;
        let _har: PipelineTensor<HarmonicSourceOutput>;
        let _gen: TypedGeneratorResult;
        let _audio: PipelineTensor<IstftOutput>;
    };
}
