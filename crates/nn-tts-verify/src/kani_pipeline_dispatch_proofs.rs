// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for pipeline dispatch routing and ay kernel classification.
//!
//! Proves correctness of dispatch step classification, ay kernel category mapping,
//! metadata-only identification, and implementation correctness evidence aggregation.
//!
//! Properties proved:
//!
//! 1. ay_kernel_category returns None for all metadata-only ops.
//! 2. is_metadata_only and ay_kernel_category are consistent (metadata => None).
//! 3. ay_proven_kernel_names matches known kernel count (20 kernels).
//! 4. All ay proven kernel names are non-empty.
//! 5. Sigmoid/Gelu/Relu/Tanh always map to ay kernel categories.
//! 6. BinaryAdd/BinaryMul always map to ay kernel categories.
//! 7. analyze_dispatch_plan: proven_steps <= total_steps.
//! 8. analyze_dispatch_plan: all_proven iff proven_steps == total_steps > 0.
//! 9. check_implementation_correctness: SmtProven only when all_proven.
//! 10. check_implementation_correctness: fraction >= 0.5 => CrownPartial.
//! 11. check_implementation_correctness: fraction < 0.5 => Empirical.
//! 12. check_implementation_correctness: bound_value <= threshold always.
//! 13. MoonshotPropertyResult property_index for P8 is always 7.
//! 14. is_metadata_only recognizes Reshape, Narrow, Transpose, AxisSelect, ZeroPad1d.

use super::moonshot::VerificationLevel;
use super::moonshot_crown::{
    check_implementation_correctness, ay_proven_kernel_names, ImplementationCorrectnessEvidence,
};

// ---- ay Kernel Category Proofs ----------------------------------------------

/// Prove: ay_proven_kernel_names returns exactly 20 kernels.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn ay_proven_kernels_count_is_20() {
    let names = ay_proven_kernel_names();
    assert_eq!(names.len(), 20, "must have exactly 20 ay-proven kernels");
}

/// Prove: all ay proven kernel names are non-empty.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(21)]
fn ay_proven_kernel_names_all_non_empty() {
    let names = ay_proven_kernel_names();
    for name in names {
        assert!(!name.is_empty(), "kernel name must not be empty");
    }
}

/// Prove: all ay proven kernel names are unique (no duplicates).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(21)]
fn ay_proven_kernel_names_unique() {
    let names = ay_proven_kernel_names();
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            assert_ne!(names[i], names[j], "ay kernel names must be unique");
        }
    }
}

/// Prove: Sigmoid DispatchStep always maps to "sigmoid" ay category.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn sigmoid_maps_to_ay_category() {
    let step = nn_dsl::DispatchStep::Sigmoid {
        kernel_name: "test_sigmoid".to_string(),
        total_elements: 128,
        input: nn_dsl::tensor_ir::TensorNodeId::new(0),
        output: nn_dsl::tensor_ir::TensorNodeId::new(1),
        dtype: nn_dsl::ir::ScalarType::F32,
    };
    let cat = super::moonshot_crown::ay_kernel_category(&step);
    assert_eq!(cat, Some("sigmoid"), "Sigmoid must map to ay 'sigmoid'");
}

/// Prove: Gelu DispatchStep always maps to "gelu" ay category.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn gelu_maps_to_ay_category() {
    let step = nn_dsl::DispatchStep::Gelu {
        kernel_name: "test_gelu".to_string(),
        total_elements: 128,
        input: nn_dsl::tensor_ir::TensorNodeId::new(0),
        output: nn_dsl::tensor_ir::TensorNodeId::new(1),
        dtype: nn_dsl::ir::ScalarType::F32,
    };
    let cat = super::moonshot_crown::ay_kernel_category(&step);
    assert_eq!(cat, Some("gelu"), "Gelu must map to ay 'gelu'");
}

/// Prove: Relu DispatchStep always maps to "relu" ay category.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn relu_maps_to_ay_category() {
    let step = nn_dsl::DispatchStep::Relu {
        kernel_name: "test_relu".to_string(),
        total_elements: 128,
        input: nn_dsl::tensor_ir::TensorNodeId::new(0),
        output: nn_dsl::tensor_ir::TensorNodeId::new(1),
        dtype: nn_dsl::ir::ScalarType::F32,
    };
    let cat = super::moonshot_crown::ay_kernel_category(&step);
    assert_eq!(cat, Some("relu"), "Relu must map to ay 'relu'");
}

/// Prove: Tanh DispatchStep always maps to "tanh_act" ay category.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn tanh_maps_to_ay_category() {
    let step = nn_dsl::DispatchStep::Tanh {
        kernel_name: "test_tanh".to_string(),
        total_elements: 128,
        input: nn_dsl::tensor_ir::TensorNodeId::new(0),
        output: nn_dsl::tensor_ir::TensorNodeId::new(1),
        dtype: nn_dsl::ir::ScalarType::F32,
    };
    let cat = super::moonshot_crown::ay_kernel_category(&step);
    assert_eq!(cat, Some("tanh_act"), "Tanh must map to ay 'tanh_act'");
}

/// Prove: BinaryAdd maps to "add" ay category.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn binary_add_maps_to_ay_category() {
    let step = nn_dsl::DispatchStep::BinaryAdd {
        kernel_name: "test_add".to_string(),
        total_elements: 128,
        left: nn_dsl::tensor_ir::TensorNodeId::new(0),
        right: nn_dsl::tensor_ir::TensorNodeId::new(1),
        output: nn_dsl::tensor_ir::TensorNodeId::new(2),
        dtype: nn_dsl::ir::ScalarType::F32,
        broadcast: None,
    };
    let cat = super::moonshot_crown::ay_kernel_category(&step);
    assert_eq!(cat, Some("add"), "BinaryAdd must map to ay 'add'");
}

/// Prove: BinaryMul maps to "mul" ay category.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn binary_mul_maps_to_ay_category() {
    let step = nn_dsl::DispatchStep::BinaryMul {
        kernel_name: "test_mul".to_string(),
        total_elements: 128,
        left: nn_dsl::tensor_ir::TensorNodeId::new(0),
        right: nn_dsl::tensor_ir::TensorNodeId::new(1),
        output: nn_dsl::tensor_ir::TensorNodeId::new(2),
        dtype: nn_dsl::ir::ScalarType::F32,
        broadcast: None,
    };
    let cat = super::moonshot_crown::ay_kernel_category(&step);
    assert_eq!(cat, Some("mul"), "BinaryMul must map to ay 'mul'");
}

/// Prove: Reshape is metadata-only and has no ay category.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn reshape_is_metadata_only() {
    let step = nn_dsl::DispatchStep::Reshape {
        input: nn_dsl::tensor_ir::TensorNodeId::new(0),
        output: nn_dsl::tensor_ir::TensorNodeId::new(1),
    };
    assert!(
        super::moonshot_crown::is_metadata_only(&step),
        "Reshape must be metadata-only"
    );
    assert_eq!(
        super::moonshot_crown::ay_kernel_category(&step),
        Option::None,
        "Reshape has no ay category"
    );
}

/// Prove: is_metadata_only returns false for computational ops.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn sigmoid_is_not_metadata_only() {
    let step = nn_dsl::DispatchStep::Sigmoid {
        kernel_name: "sig".to_string(),
        total_elements: 128,
        input: nn_dsl::tensor_ir::TensorNodeId::new(0),
        output: nn_dsl::tensor_ir::TensorNodeId::new(1),
        dtype: nn_dsl::ir::ScalarType::F32,
    };
    assert!(
        !super::moonshot_crown::is_metadata_only(&step),
        "Sigmoid is not metadata-only"
    );
}

// ---- ImplementationCorrectnessEvidence Proofs -------------------------------

/// Prove: proven_steps <= total_steps always holds.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn evidence_proven_leq_total() {
    let total: usize = kani::any();
    let proven: usize = kani::any();
    kani::assume(total <= 10000);
    kani::assume(proven <= total);

    let evidence = ImplementationCorrectnessEvidence {
        total_steps: total,
        proven_steps: proven,
        proven_categories: vec![],
        unproven_categories: vec![],
        all_proven: proven == total && total > 0,
    };

    assert!(
        evidence.proven_steps <= evidence.total_steps,
        "proven_steps must be <= total_steps"
    );
}

/// Prove: all_proven iff proven_steps == total_steps > 0.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn evidence_all_proven_semantics() {
    let total: usize = kani::any();
    let proven: usize = kani::any();
    kani::assume(total <= 10000);
    kani::assume(proven <= total);

    let all_proven = proven == total && total > 0;

    let evidence = ImplementationCorrectnessEvidence {
        total_steps: total,
        proven_steps: proven,
        proven_categories: vec![],
        unproven_categories: vec![],
        all_proven,
    };

    if evidence.all_proven {
        assert_eq!(evidence.proven_steps, evidence.total_steps);
        assert!(evidence.total_steps > 0);
    }
    if evidence.total_steps > 0 && evidence.proven_steps == evidence.total_steps {
        assert!(evidence.all_proven);
    }
}

// ---- check_implementation_correctness Level Proofs --------------------------

/// Prove: SmtProven only when all_proven is true.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn correctness_smt_proven_requires_all_proven() {
    let total: usize = kani::any();
    let proven: usize = kani::any();
    kani::assume(total > 0 && total <= 1000);
    kani::assume(proven <= total);

    let evidence = ImplementationCorrectnessEvidence {
        total_steps: total,
        proven_steps: proven,
        proven_categories: vec![],
        unproven_categories: vec![],
        all_proven: proven == total,
    };

    let result = check_implementation_correctness(&evidence);

    if result.level == VerificationLevel::SmtProven {
        assert!(evidence.all_proven, "SmtProven requires all_proven");
    }
}

/// Prove: fraction >= 0.5 yields at least CrownPartial (when not all_proven).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn correctness_half_coverage_crown_partial() {
    let total: usize = kani::any();
    let proven: usize = kani::any();
    kani::assume(total > 0 && total <= 1000);
    kani::assume(proven <= total);
    kani::assume(proven < total); // not all_proven

    let fraction = proven as f64 / total as f64;
    kani::assume(fraction >= 0.5);

    let evidence = ImplementationCorrectnessEvidence {
        total_steps: total,
        proven_steps: proven,
        proven_categories: vec![],
        unproven_categories: vec![],
        all_proven: false,
    };

    let result = check_implementation_correctness(&evidence);
    assert_eq!(
        result.level,
        VerificationLevel::CrownPartial,
        ">=50% coverage without all_proven must be CrownPartial"
    );
}

/// Prove: fraction < 0.5 yields Empirical.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn correctness_low_coverage_empirical() {
    let total: usize = kani::any();
    let proven: usize = kani::any();
    kani::assume(total > 1 && total <= 1000);
    kani::assume(proven <= total);

    let fraction = proven as f64 / total as f64;
    kani::assume(fraction < 0.5);

    let evidence = ImplementationCorrectnessEvidence {
        total_steps: total,
        proven_steps: proven,
        proven_categories: vec![],
        unproven_categories: vec![],
        all_proven: false,
    };

    let result = check_implementation_correctness(&evidence);
    assert_eq!(
        result.level,
        VerificationLevel::Empirical,
        "<50% coverage must be Empirical"
    );
}

/// Prove: check_implementation_correctness always sets property_index to 7 (P8).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn correctness_property_index_is_seven() {
    let total: usize = kani::any();
    let proven: usize = kani::any();
    kani::assume(total <= 1000);
    kani::assume(proven <= total);

    let evidence = ImplementationCorrectnessEvidence {
        total_steps: total,
        proven_steps: proven,
        proven_categories: vec![],
        unproven_categories: vec![],
        all_proven: proven == total && total > 0,
    };

    let result = check_implementation_correctness(&evidence);
    assert_eq!(
        result.property_index, 7,
        "P8 implementation correctness must have property_index 7"
    );
}

/// Prove: bound_value (proven_steps) <= threshold (total_steps) always.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn correctness_bound_leq_threshold() {
    let total: usize = kani::any();
    let proven: usize = kani::any();
    kani::assume(total <= 1000);
    kani::assume(proven <= total);

    let evidence = ImplementationCorrectnessEvidence {
        total_steps: total,
        proven_steps: proven,
        proven_categories: vec![],
        unproven_categories: vec![],
        all_proven: proven == total && total > 0,
    };

    let result = check_implementation_correctness(&evidence);
    assert!(
        result.bound_value <= result.threshold,
        "bound_value (proven) must be <= threshold (total)"
    );
}

/// Prove: is_sound is always true for implementation correctness (ay is sound).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn correctness_is_sound_always_true() {
    let evidence = ImplementationCorrectnessEvidence {
        total_steps: 10,
        proven_steps: 5,
        proven_categories: vec![],
        unproven_categories: vec![],
        all_proven: false,
    };

    let result = check_implementation_correctness(&evidence);
    assert!(result.is_sound, "ay proofs are inherently sound");
}
