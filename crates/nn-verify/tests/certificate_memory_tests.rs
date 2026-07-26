// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Memory safety and layout tests for certificate pipeline types.
//!
//! Verifies:
//! 1. Struct/enum sizes stay bounded as fields are added.
//! 2. Clone produces fully independent copies (no shared interior mutability).
//! 3. CertificateBundle::save cleans up .tmp files on success.
//! 4. ProofCertificate with large layer_bounds doesn't exhibit stack issues.
//! 5. Send+Sync for concurrent safety.
//!
//! Part of #3020 (certificate pipeline verification), memory_verification phase.

use nn_verify::status::{InputBoundsRecord, ParamInputRecord};
use nn_verify::{
    CertificateBundle, CheckIssue, CheckResult, KernelVerification, LayerBoundRecord,
    OutputTensorBounds, ProofCertificate, PropMethod, VacuityAssessment, VerificationSoundnessMode,
};

// ---------------------------------------------------------------------------
// Type size assertions — prevent silent struct/enum bloat
// ---------------------------------------------------------------------------

/// ProofCertificate has 21+ fields. Each new Option<T> adds ~24 bytes
/// (discriminant + padding + T). This assertion catches silent growth.
///
/// Current expected size: ~400-600 bytes on 64-bit (varies by alignment).
/// Threshold is generous (1024) to avoid false positives from alignment
/// changes, but tight enough to catch accidental Vec/HashMap additions
/// to the struct body (which would push it over 1024).
#[test]
fn test_proof_certificate_struct_size_bounded() {
    let size = size_of::<ProofCertificate>();
    assert!(
        size <= 1024,
        "ProofCertificate is {size} bytes — expected <= 1024. \
         Adding large inline fields (Vec, HashMap) to the struct body \
         instead of Option<Box<T>> causes stack bloat."
    );
    // Sanity: it should be non-trivially sized (has String + Vec fields).
    assert!(
        size >= 128,
        "ProofCertificate is only {size} bytes — suspiciously small. \
         Did the struct lose fields?"
    );
}

/// CheckIssue has 20+ variants. The largest (LayerTraceGap) contains two
/// Vec<(f32,f32)> = 2×24 bytes inline + discriminant. Enum size equals
/// the largest variant, so all CheckIssue values pay that cost.
#[test]
fn test_check_issue_enum_size_bounded() {
    let size = size_of::<CheckIssue>();
    assert!(
        size <= 128,
        "CheckIssue is {size} bytes — expected <= 128. \
         Variants with large inline data inflate ALL variant sizes. \
         Consider boxing large payloads: Box<Vec<(f32,f32)>>."
    );
}

/// LayerBoundRecord contains two Vec<(f32,f32)> plus metadata. The struct
/// itself should be small (all heap-allocated via Vec).
#[test]
fn test_layer_bound_record_size_bounded() {
    let size = size_of::<LayerBoundRecord>();
    assert!(
        size <= 256,
        "LayerBoundRecord is {size} bytes — expected <= 256."
    );
}

/// CertificateBundle's stack footprint should be small (heap-allocated certs).
#[test]
fn test_certificate_bundle_size_bounded() {
    let size = size_of::<CertificateBundle>();
    assert!(
        size <= 128,
        "CertificateBundle is {size} bytes — expected <= 128."
    );
}

// ---------------------------------------------------------------------------
// Clone independence — mutating a clone must not affect the original
// ---------------------------------------------------------------------------

/// ProofCertificate clone produces a fully independent copy.
/// Mutating fields on the clone (including heap-allocated Vecs) must not
/// affect the original. This catches accidental Rc/Arc sharing.
#[test]
fn test_proof_certificate_clone_independence() {
    let mut tensor = OutputTensorBounds::new(vec![-5.0; 10], vec![5.0; 10], vec![10]);
    tensor.finite_mask = vec![true; 10];

    let mut result = KernelVerification::new(
        "clone_test".to_string(),
        PropMethod::Crown,
        -5.0,
        5.0,
        10.0,
        true,
    )
    .with_soundness_mode(VerificationSoundnessMode::Sound);
    result.output_tensor = Some(tensor);

    let spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -10.0, 10.0)], &[1.0]);

    let layer_bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-10.0, 10.0); 10],
        output_bounds: vec![(-5.0, 5.0); 10],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: Some(vec![]),
    }];

    let original = ProofCertificate::from_verification(&result, spec)
        .with_layer_bounds(layer_bounds)
        .with_source_hash("a".repeat(64));

    let mut cloned = original.clone();

    // Mutate the clone: change kernel_name, modify layer_bounds, add smt_outcome.
    cloned.kernel_name = "mutated".to_string();
    if let Some(ref mut bounds) = cloned.layer_bounds {
        bounds[0].output_bounds[0] = (999.0, 999.0);
    }
    cloned.smt_outcome = Some("Mutated".to_string());

    // Original must be unaffected.
    assert_eq!(original.kernel_name, "clone_test");
    assert_eq!(
        original.layer_bounds.as_ref().unwrap()[0].output_bounds[0],
        (-5.0, 5.0)
    );
    assert!(original.smt_outcome.is_none());
}

/// CertificateBundle clone produces an independent copy of its certificates.
#[test]
fn test_certificate_bundle_clone_independence() {
    let result = KernelVerification::new(
        "bundle_clone_test".to_string(),
        PropMethod::Ibp,
        -1.0,
        1.0,
        2.0,
        true,
    )
    .with_soundness_mode(VerificationSoundnessMode::Sound);

    let spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);

    let original = CertificateBundle::new("test_model")
        .with_certificate(ProofCertificate::from_verification(&result, spec));

    let mut cloned = original.clone();
    cloned.certificates[0].kernel_name = "mutated_bundle".to_string();
    cloned.model_name = "mutated_model".to_string();

    assert_eq!(original.model_name, "test_model");
    assert_eq!(original.certificates[0].kernel_name, "bundle_clone_test");
}

// ---------------------------------------------------------------------------
// File I/O RAII — .tmp file cleanup
// ---------------------------------------------------------------------------

/// CertificateBundle::save removes the .tmp file on successful save.
///
/// The atomic write pattern (write .tmp → rename to target) must not
/// leave orphaned .tmp files after successful completion.
#[test]
fn test_bundle_save_cleans_up_tmp_file() {
    let dir = std::env::temp_dir().join(format!("nn_cert_mem_save_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let target_path = dir.join("test_bundle.proof.json");

    let bundle = CertificateBundle::new("tmp_cleanup_test");
    bundle.save(&target_path).expect("save should succeed");

    // The .tmp file should NOT exist after successful save.
    let tmp_path = {
        let mut s = target_path.as_os_str().to_owned();
        s.push(".tmp");
        std::path::PathBuf::from(s)
    };
    assert!(
        !tmp_path.exists(),
        ".tmp file should be removed after successful save: {}",
        tmp_path.display()
    );
    assert!(target_path.exists(), "target file should exist");

    // Clean up.
    let _ = std::fs::remove_file(&target_path);
    let _ = std::fs::remove_dir(&dir);
}

/// CertificateBundle save/load roundtrip preserves all fields — no data loss.
#[test]
fn test_bundle_save_load_roundtrip_preserves_data() {
    let dir = std::env::temp_dir().join(format!("nn_cert_mem_roundtrip_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("roundtrip.proof.json");

    let result = KernelVerification::new(
        "roundtrip_kernel".to_string(),
        PropMethod::Crown,
        -3.0,
        7.0,
        10.0,
        true,
    )
    .with_soundness_mode(VerificationSoundnessMode::Sound);

    let spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -5.0, 5.0)], &[2.5]);

    let original = CertificateBundle::new("roundtrip_model")
        .with_certificate(ProofCertificate::from_verification(&result, spec));

    original.save(&path).expect("save");
    let loaded = CertificateBundle::load(&path).expect("load");

    assert_eq!(loaded.model_name, original.model_name);
    assert_eq!(loaded.certificates.len(), 1);
    assert_eq!(
        loaded.certificates[0].kernel_name,
        original.certificates[0].kernel_name
    );
    assert_eq!(
        loaded.certificates[0].output_bounds,
        original.certificates[0].output_bounds
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

// ---------------------------------------------------------------------------
// Large certificate memory behavior
// ---------------------------------------------------------------------------

/// ProofCertificate with many layer bounds (deep model) doesn't cause
/// stack overflow on creation, clone, or validation.
///
/// 10000 layers × 100 elements = 10M bounds pairs. All heap-allocated.
/// This tests that the type system doesn't accidentally put bounds on stack.
#[test]
fn test_deep_certificate_no_stack_overflow() {
    let n_layers = 10_000;
    let n_elements = 100;

    let bounds: Vec<LayerBoundRecord> = (0..n_layers)
        .map(|i| LayerBoundRecord {
            layer_index: i,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0); n_elements],
            output_bounds: vec![(-0.5, 0.5); n_elements],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(if i == 0 { vec![] } else { vec![i - 1] }),
        })
        .collect();

    let result = KernelVerification::new(
        "deep_model".to_string(),
        PropMethod::Crown,
        -0.5,
        0.5,
        1.0,
        true,
    )
    .with_soundness_mode(VerificationSoundnessMode::Sound);

    let spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);

    let cert = ProofCertificate::from_verification(&result, spec).with_layer_bounds(bounds);

    // Validate: should succeed (all layers consistent).
    cert.validate().expect("deep cert should validate");

    // Clone: should not stack-overflow (all data is heap-allocated).
    let cloned = cert.clone();
    assert_eq!(cloned.layer_bounds.as_ref().unwrap().len(), n_layers,);

    // Verify total heap allocation is proportional to n_layers * n_elements.
    let layer_data_size: usize = cert
        .layer_bounds
        .as_ref()
        .unwrap()
        .iter()
        .map(|r| {
            r.input_bounds.len() * size_of::<(f32, f32)>()
                + r.output_bounds.len() * size_of::<(f32, f32)>()
        })
        .sum();
    // Each layer: 100 elements × 2 vecs × 8 bytes = 1600 bytes.
    // 10000 layers = 16 MB of bounds data alone.
    let expected_min = n_layers * n_elements * 2 * size_of::<(f32, f32)>();
    assert_eq!(layer_data_size, expected_min);
}

// ---------------------------------------------------------------------------
// Send/Sync for certificate types
// ---------------------------------------------------------------------------

/// ProofCertificate must be Send + Sync for safe concurrent access.
/// This is a compile-time assertion.
#[test]
fn test_proof_certificate_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ProofCertificate>();
    assert_send_sync::<CertificateBundle>();
    assert_send_sync::<CheckResult>();
    assert_send_sync::<CheckIssue>();
    assert_send_sync::<VacuityAssessment>();
}
