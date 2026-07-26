// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `verify_model!` declarative macro — generate verification tests from model specs.
//!
//! Eliminates ~30 lines of manual wiring per model (trace → verify → certify).
//!
//! # Basic usage (IBP verification)
//!
//! ```rust,ignore
//! nn_verify::verify_model! {
//!     name: nn_linear_relu,
//!     model: { Linear::new(weight, None).unwrap() },
//!     forward: |m, x| m.forward(&x)?.relu(),
//!     input: DynTensor::randn(&[1, 784], DType::F32, &Device::Cpu),
//!     bounds: nn_verify::uniform_bounds(&[1, 784], 1.0).unwrap(),
//! }
//! ```
//!
//! # With full certification
//!
//! ```rust,ignore
//! nn_verify::verify_model! {
//!     name: nn_classifier,
//!     model: { NnModel::new(&vb, config).unwrap() },
//!     forward: |m, x| m.forward(&x),
//!     input: DynTensor::randn(&[1, 784], DType::F32, &Device::Cpu),
//!     bounds: nn_verify::uniform_bounds(&[1, 784], 1.0).unwrap(),
//!     certify: true,
//! }
//! ```
//!
//! Part of #3020, #3051, #2218.

/// Generate verification tests from a model specification.
///
/// Creates a test module named `$name` containing:
/// - `fn verify()` — traces the model, runs IBP bound propagation, asserts non-vacuous bounds
/// - `fn certify()` (when `certify: true`) — additionally produces a `CertificateBundle`,
///   saves it to `$CARGO_MANIFEST_DIR/proofs/$name.proof.json`, and validates all checks pass
///
/// # Parameters
///
/// - `name`: identifier used as the test module name
/// - `model`: expression that constructs the model (wrapped in braces for complex exprs)
/// - `forward`: closure `|model_ident, input_ident| expr` for the forward pass.
///   `model_ident` receives `&model`, `input_ident` receives the traced input `DynTensor`.
///   The closure body must return `Result<DynTensor>`.
/// - `input`: expression that creates the input `DynTensor`
/// - `bounds`: expression that creates the `BoundedTensor` for verification
/// - `certify` (optional): `true` to also generate a certification test
#[macro_export]
macro_rules! verify_model {
    // Arm 1: verify only
    (
        name: $name:ident,
        model: $model:expr,
        forward: |$m:ident, $x:ident| $body:expr,
        input: $input:expr,
        bounds: $bounds:expr $(,)?
    ) => {
        #[cfg(test)]
        mod $name {
            use super::*;

            #[test]
            fn verify() {
                let model = $model;
                let input = $input;
                let (_output, graph) = $crate::__macro_internals::trace_graph(|| {
                    let mut traced = input.clone();
                    if let Some(id) =
                        $crate::__macro_internals::record_input(traced.dims(), traced.dtype())
                    {
                        traced.set_trace_id(id);
                    }
                    let $m = &model;
                    let $x = traced;
                    $body
                })
                .expect("model tracing should succeed");

                let bounds = $bounds;
                let result =
                    $crate::verify_trace(&graph, &bounds).expect("IBP verification should succeed");
                assert!(
                    result.ibp_width < $crate::DEFAULT_VACUITY_THRESHOLD,
                    "IBP bounds are vacuously wide (width: {}, threshold: {})",
                    result.ibp_width,
                    $crate::DEFAULT_VACUITY_THRESHOLD,
                );
            }
        }
    };

    // Arm 2: verify + certify
    (
        name: $name:ident,
        model: $model:expr,
        forward: |$m:ident, $x:ident| $body:expr,
        input: $input:expr,
        bounds: $bounds:expr,
        certify: true $(,)?
    ) => {
        #[cfg(test)]
        mod $name {
            use super::*;

            fn trace_model() -> (
                $crate::__macro_internals::DynTensor,
                $crate::__macro_internals::ComputationGraph,
                $crate::BoundedTensor,
            ) {
                let model = $model;
                let input = $input;
                let (output, graph) = $crate::__macro_internals::trace_graph(|| {
                    let mut traced = input.clone();
                    if let Some(id) =
                        $crate::__macro_internals::record_input(traced.dims(), traced.dtype())
                    {
                        traced.set_trace_id(id);
                    }
                    let $m = &model;
                    let $x = traced;
                    $body
                })
                .expect("model tracing should succeed");
                let bounds = $bounds;
                (output, graph, bounds)
            }

            #[test]
            fn verify() {
                let (_output, graph, bounds) = trace_model();
                let result =
                    $crate::verify_trace(&graph, &bounds).expect("IBP verification should succeed");
                assert!(
                    result.ibp_width < $crate::DEFAULT_VACUITY_THRESHOLD,
                    "IBP bounds are vacuously wide (width: {}, threshold: {})",
                    result.ibp_width,
                    $crate::DEFAULT_VACUITY_THRESHOLD,
                );
            }

            #[test]
            fn certify() {
                let (_output, graph, bounds) = trace_model();
                let config = $crate::CertifyConfig::new(stringify!($name));
                let result = $crate::certify_model(&graph, &bounds, &config)
                    .expect("certification should succeed");

                // Save certificate
                let proof_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("proofs");
                std::fs::create_dir_all(&proof_dir).ok();
                let cert_path = proof_dir.join(concat!(stringify!($name), ".proof.json"));
                result
                    .bundle
                    .save(&cert_path)
                    .expect("certificate save should succeed");

                // Validate certificate — filter out MissingHash (no enrichment
                // in test context). VacuousBounds is NOT filtered: vacuously
                // wide certificates have zero verification value (#3200 F1).
                let checks = $crate::check_bundle(&result.bundle, None, None);
                let fatal: Vec<_> = checks
                    .iter()
                    .flat_map(|c| &c.issues)
                    .filter(|i| !matches!(i, $crate::CheckIssue::MissingHash { .. }))
                    .collect();
                assert!(
                    fatal.is_empty(),
                    "certificate has fatal issues: {:?}",
                    fatal,
                );
            }
        }
    };
}
