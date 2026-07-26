// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "convert-model")]

#[test]
fn import_surface_stays_available_without_import_metal() {
    use nn::import::{
        build_graph, build_weight_map, parse_exported_program, ConvertArtifactKind,
        ConvertIntakePath, ConvertReport, ExportedProgram, ImportedGraph, InputSpec, OutputSpec,
    };

    fn check_type<T>() {}

    let _ = build_graph;
    let _ = build_weight_map;
    let _ = parse_exported_program;

    check_type::<ExportedProgram>();
    check_type::<ImportedGraph>();
    check_type::<InputSpec>();
    check_type::<OutputSpec>();
    check_type::<ConvertReport>();
    check_type::<ConvertIntakePath>();
    check_type::<ConvertArtifactKind>();
}

#[test]
fn root_report_provenance_helpers_are_available_without_import_metal() {
    use nn::{ConvertArtifactKind, ConvertIntakePath, ConvertReport};

    let _: fn(&ConvertReport) -> String = ConvertReport::provenance_summary;
    let _: fn(&ConvertReport) -> &'static str = ConvertReport::artifact_readiness_note;
    let _: fn(ConvertIntakePath) -> &'static str = ConvertIntakePath::label;
    let _: fn(ConvertArtifactKind) -> &'static str = ConvertArtifactKind::label;

    assert_eq!(
        ConvertIntakePath::ExportedArtifacts.label(),
        "exported artifacts"
    );
    assert_eq!(
        ConvertIntakePath::CliExportedPytorch.label(),
        "CLI-exported PyTorch"
    );
    assert_eq!(
        ConvertArtifactKind::BackendAgnosticConvertedGraph.label(),
        "backend-agnostic converted graph"
    );
    assert_eq!(
        ConvertArtifactKind::CompiledMetalArtifact.label(),
        "compiled Metal artifact"
    );
}

#[test]
fn root_and_import_report_types_match() {
    fn accept_root_report(_: nn::ConvertReport) {}
    fn accept_root_intake(_: nn::ConvertIntakePath) {}
    fn accept_root_artifact(_: nn::ConvertArtifactKind) {}

    fn produce_import_report() -> Option<nn::import::ConvertReport> {
        None
    }

    fn produce_import_intake() -> Option<nn::import::ConvertIntakePath> {
        None
    }

    fn produce_import_artifact() -> Option<nn::import::ConvertArtifactKind> {
        None
    }

    if let Some(report) = produce_import_report() {
        accept_root_report(report);
    }
    if let Some(intake) = produce_import_intake() {
        accept_root_intake(intake);
    }
    if let Some(artifact) = produce_import_artifact() {
        accept_root_artifact(artifact);
    }
}
