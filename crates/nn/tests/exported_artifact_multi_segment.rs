// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(feature = "import-metal", target_os = "macos"))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use nn::import::convert_multi_segment as import_convert_multi_segment;
use nn::metal::{CompiledModel, MetalBackend, PipelineCache};
use nn::{compile_multi_segment, convert_multi_segment, convert_multi_segment_to_metal};
use nn::{save_safetensors, Device, DynTensor};

struct ExportFixtureDir {
    path: PathBuf,
}

impl ExportFixtureDir {
    fn new(name: &str) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "nn_{name}_{}_{}_{}",
            std::process::id(),
            unique,
            std::thread::current().name().unwrap_or("unnamed"),
        ));
        std::fs::create_dir_all(&path).expect("create export fixture dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ExportFixtureDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn mlp_graph_json() -> serde_json::Value {
    serde_json::from_str(include_str!("../../nn-import/test_data/e2e_mlp.json"))
        .expect("parse shared MLP fixture")
}

fn shared_weight_graph_json() -> serde_json::Value {
    serde_json::from_str(
        r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    {"as_tensor": {"name": "p_fc1_weight"}},
                    {"as_tensor": {"name": "p_fc1_bias"}},
                    {"as_tensor": {"name": "z"}}
                ],
                "outputs": [{"as_tensor": {"name": "head_out"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.linear.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "z"}}, "kind": 1},
                            {"name": "weight", "arg": {"as_tensor": {"name": "p_fc1_weight"}}, "kind": 1},
                            {"name": "bias", "arg": {"as_tensor": {"name": "p_fc1_bias"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "head_out"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "z": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_fc1_weight": {"dtype": 7, "sizes": [{"as_int": 8}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_fc1_bias": {"dtype": 7, "sizes": [{"as_int": 8}], "requires_grad": true, "strides": [{"as_int": 1}]},
                    "head_out": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 8}], "requires_grad": false, "strides": [{"as_int": 8}, {"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_fc1_weight"}, "parameter_name": "fc1.weight"}},
                    {"parameter": {"arg": {"name": "p_fc1_bias"}, "parameter_name": "fc1.bias"}},
                    {"user_input": {"arg": {"as_tensor": {"name": "z"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "head_out"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "opset_version": {"aten": 10},
        "range_constraints": {}
    }"#,
    )
    .expect("parse shared-weight segment fixture")
}

fn write_multi_segment_fixture() -> (ExportFixtureDir, Vec<(String, serde_json::Value)>, PathBuf) {
    let dir = ExportFixtureDir::new("exported_artifact_multi_segment");

    let mut tensors = HashMap::new();
    tensors.insert(
        "fc1.weight".to_string(),
        DynTensor::from_vec(
            (0..32u32).map(|i| (i as f32) * 0.01).collect(),
            &[8, 4],
            &Device::Cpu,
        )
        .expect("fc1.weight tensor"),
    );
    tensors.insert(
        "fc1.bias".to_string(),
        DynTensor::from_vec(vec![0.0f32; 8], &[8], &Device::Cpu).expect("fc1.bias tensor"),
    );
    tensors.insert(
        "fc2.weight".to_string(),
        DynTensor::from_vec(
            (0..24u32).map(|i| (i as f32) * 0.01).collect(),
            &[3, 8],
            &Device::Cpu,
        )
        .expect("fc2.weight tensor"),
    );
    tensors.insert(
        "fc2.bias".to_string(),
        DynTensor::from_vec(vec![0.0f32; 3], &[3], &Device::Cpu).expect("fc2.bias tensor"),
    );

    let weights_path = dir.path().join("shared_weights.safetensors");
    save_safetensors(&tensors, &weights_path).expect("write shared weights");

    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("head".to_string(), shared_weight_graph_json()),
    ];

    (dir, graphs, weights_path)
}

fn metal_cache() -> Option<PipelineCache> {
    let _backend = MetalBackend::init().ok()?;
    nn::metal::register_metal_dyn_backend();
    PipelineCache::new_global().ok()
}

fn execute_segment(
    model: &CompiledModel,
    cache: &PipelineCache,
    input_cpu: &DynTensor,
) -> Vec<f32> {
    let input_gpu = input_cpu
        .to_device(&Device::metal())
        .expect("fixture input -> metal");
    let output = model
        .execute_dyn(cache, &[&input_gpu])
        .expect("segment execution");
    output
        .to_device(&Device::Cpu)
        .expect("segment output -> cpu")
        .to_flat_vec::<f32>()
        .expect("flatten segment output")
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: output length mismatch"
    );
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "{label}[{idx}]: actual={actual}, expected={expected}, diff={diff}"
        );
    }
}

#[test]
fn root_multi_segment_surfaces_match_manual_compile_and_execution() {
    let Some(cache) = metal_cache() else {
        return;
    };

    let (_dir, graphs, weights_path) = write_multi_segment_fixture();

    let root_imported = convert_multi_segment(&graphs, &weights_path)
        .expect("root multi-segment import should succeed");
    assert_eq!(root_imported.segment_order, vec!["encoder", "head"]);
    assert_eq!(
        root_imported.shared_weights,
        vec!["fc1.bias".to_string(), "fc1.weight".to_string()]
    );

    let root_staged = compile_multi_segment(&root_imported, &cache)
        .expect("root staged multi-segment compile should succeed");
    let root_direct = convert_multi_segment_to_metal(&graphs, &weights_path, &cache)
        .expect("root direct multi-segment compile should succeed");

    assert_eq!(root_staged.segment_order, root_imported.segment_order);
    assert_eq!(root_direct.segment_order, root_imported.segment_order);
    assert_eq!(root_staged.shared_weights, root_imported.shared_weights);
    assert_eq!(root_direct.shared_weights, root_imported.shared_weights);

    let manual_imported = import_convert_multi_segment(&graphs, &weights_path)
        .expect("manual import path should succeed");
    let manual_encoder = CompiledModel::builder(
        &manual_imported
            .get_segment("encoder")
            .expect("manual encoder segment")
            .graph,
        &cache,
    )
    .build()
    .expect("manual encoder compile should succeed");
    let manual_head = CompiledModel::builder(
        &manual_imported
            .get_segment("head")
            .expect("manual head segment")
            .graph,
        &cache,
    )
    .build()
    .expect("manual head compile should succeed");

    let encoder_input = DynTensor::from_vec(vec![0.25, -0.5, 0.75, 1.25], &[1, 4], &Device::Cpu)
        .expect("encoder input");
    let head_input = DynTensor::from_vec(vec![-1.0, 0.5, 1.5, -0.25], &[1, 4], &Device::Cpu)
        .expect("head input");

    let root_staged_encoder = execute_segment(
        root_staged
            .get_segment("encoder")
            .expect("root staged encoder"),
        &cache,
        &encoder_input,
    );
    let root_direct_encoder = execute_segment(
        root_direct
            .get_segment("encoder")
            .expect("root direct encoder"),
        &cache,
        &encoder_input,
    );
    let manual_encoder_output = execute_segment(&manual_encoder, &cache, &encoder_input);

    let root_staged_head = execute_segment(
        root_staged.get_segment("head").expect("root staged head"),
        &cache,
        &head_input,
    );
    let root_direct_head = execute_segment(
        root_direct.get_segment("head").expect("root direct head"),
        &cache,
        &head_input,
    );
    let manual_head_output = execute_segment(&manual_head, &cache, &head_input);

    assert_close(
        "root staged encoder vs manual",
        &root_staged_encoder,
        &manual_encoder_output,
        1e-6,
    );
    assert_close(
        "root direct encoder vs manual",
        &root_direct_encoder,
        &manual_encoder_output,
        1e-6,
    );
    assert_close(
        "root staged head vs manual",
        &root_staged_head,
        &manual_head_output,
        1e-6,
    );
    assert_close(
        "root direct head vs manual",
        &root_direct_head,
        &manual_head_output,
        1e-6,
    );
}
