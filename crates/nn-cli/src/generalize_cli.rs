// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `nn generalize` certificate generation and replay.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use nn_verify::{
    EvidenceArtifactPaths, EvidenceArtifacts, GeneralizationCertificate,
    GeneralizationCertificateInputs, HypothesisEncoding,
};

const MINIMUM_HMAC_KEY_BYTES: usize = 32;

#[derive(Debug, Subcommand)]
pub(crate) enum GeneralizeCommand {
    /// Calculate and save a content-sealed population-reward certificate.
    Calculate {
        /// Reward transcript JSON following schema version 1.
        transcript: PathBuf,

        #[command(flatten)]
        artifacts: ArtifactArguments,

        /// Stable ID for the fixed, hypothesis-independent prefix-code scheme.
        #[arg(long)]
        encoding_scheme: String,

        /// Charged bit length C(h) of the encoded hypothesis.
        ///
        /// Every byte in the encoded file is charged (8 bits per byte). Do not
        /// include the theorem's 2 ln C overhead here.
        #[arg(long)]
        description_bits: u64,

        /// Failure probability delta; reported confidence is 1-delta.
        #[arg(long, default_value = "0.05")]
        delta: f64,

        /// Output certificate JSON path.
        #[arg(short, long)]
        output: PathBuf,

        /// Raw HMAC key file used to authenticate the certificate.
        #[arg(long)]
        hmac_key_file: Option<PathBuf>,

        /// Print the complete certificate JSON to stdout.
        #[arg(long)]
        json: bool,
    },

    /// Replay artifact identities, transcript aggregation, and bound arithmetic.
    Check {
        /// Generalization certificate JSON produced by `calculate`.
        certificate: PathBuf,

        /// Exact reward transcript used to create the certificate.
        transcript: PathBuf,

        #[command(flatten)]
        artifacts: ArtifactArguments,

        /// Raw HMAC key file required when the certificate is signed.
        #[arg(long)]
        hmac_key_file: Option<PathBuf>,

        /// Print a structured replay report as JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, Args)]
pub(crate) struct ArtifactArguments {
    /// Frozen base-model artifact or content manifest.
    #[arg(long)]
    base_model: PathBuf,

    /// Actual packed prefix-code bitstream for the final hypothesis.
    #[arg(long)]
    encoded_hypothesis: PathBuf,

    /// Fixed reconstruction-decoder specification for the hypothesis code.
    #[arg(long)]
    hypothesis_decoder_spec: PathBuf,

    /// Tokenizer, sampler, stopping rules, and environment specification.
    #[arg(long)]
    generation_spec: PathBuf,

    /// Deterministic bounded reward/verifier specification or manifest.
    #[arg(long)]
    reward_spec: PathBuf,

    /// Manifest defining the claimed deployment prompt distribution.
    #[arg(long)]
    prompt_distribution_manifest: PathBuf,

    /// Ordered training-corpus manifest addressed by transcript indices.
    #[arg(long)]
    training_corpus_manifest: PathBuf,
}

impl ArtifactArguments {
    fn paths(&self) -> EvidenceArtifactPaths {
        EvidenceArtifactPaths::new(
            self.base_model.clone(),
            self.encoded_hypothesis.clone(),
            self.hypothesis_decoder_spec.clone(),
            self.generation_spec.clone(),
            self.reward_spec.clone(),
            self.prompt_distribution_manifest.clone(),
            self.training_corpus_manifest.clone(),
        )
    }
}

pub(crate) fn run(command: GeneralizeCommand) -> Result<()> {
    match command {
        GeneralizeCommand::Calculate {
            transcript,
            artifacts,
            encoding_scheme,
            description_bits,
            delta,
            output,
            hmac_key_file,
            json,
        } => calculate(
            &transcript,
            &artifacts,
            &encoding_scheme,
            description_bits,
            delta,
            &output,
            hmac_key_file.as_deref(),
            json,
        ),
        GeneralizeCommand::Check {
            certificate,
            transcript,
            artifacts,
            hmac_key_file,
            json,
        } => check(
            &certificate,
            &transcript,
            &artifacts,
            hmac_key_file.as_deref(),
            json,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn calculate(
    transcript_path: &Path,
    artifact_arguments: &ArtifactArguments,
    encoding_scheme: &str,
    description_bits: u64,
    delta: f64,
    output: &Path,
    hmac_key_file: Option<&Path>,
    json: bool,
) -> Result<()> {
    ensure_parent_dir(output)?;
    ensure_output_does_not_alias_input(output, transcript_path, artifact_arguments, hmac_key_file)?;
    let transcript = std::fs::read(transcript_path)
        .with_context(|| format!("failed to read transcript {}", transcript_path.display()))?;
    let artifact_paths = artifact_arguments.paths();
    let training_corpus_manifest = std::fs::read(&artifact_paths.training_corpus_manifest)
        .with_context(|| {
            format!(
                "failed to read training-corpus manifest {}",
                artifact_paths.training_corpus_manifest.display()
            )
        })?;
    let artifacts = EvidenceArtifacts::from_paths(&artifact_paths)
        .context("failed to hash generalization evidence artifacts")?;
    let inputs = GeneralizationCertificateInputs::new(
        artifacts,
        HypothesisEncoding::new(encoding_scheme, description_bits),
        delta,
    );
    let mut certificate =
        GeneralizationCertificate::calculate(inputs, &transcript, &training_corpus_manifest)
            .context("generalization certificate calculation failed")?;
    if let Some(key_path) = hmac_key_file {
        let key = read_hmac_key(key_path)?;
        nn_verify::sign_certificate(&mut certificate, &key)
            .context("failed to authenticate generalization certificate")?;
        certificate.validate()?;
    }

    certificate
        .save(output)
        .with_context(|| format!("failed to save certificate {}", output.display()))?;

    if json {
        println!("{}", certificate.to_json()?);
    } else {
        println!("Generalization certificate saved to {}", output.display());
        println!("  theorem:     {}", certificate.paper_reference);
        println!(
            "  empirical:   {:.9}",
            certificate.bound_inputs.empirical_reward
        );
        println!("  raw lower:   {:.9}", certificate.bound.raw_lower_bound);
        println!(
            "  certified:   {:.9} at {:.2}% confidence",
            certificate.bound.certified_lower_bound,
            certificate.bound.confidence * 100.0
        );
        println!(
            "  assessment:  {}",
            if certificate.is_non_vacuous() {
                "NON-VACUOUS"
            } else {
                "VACUOUS (no improvement over the declared reward floor)"
            }
        );
    }
    Ok(())
}

fn ensure_output_does_not_alias_input(
    output: &Path,
    transcript: &Path,
    artifact_arguments: &ArtifactArguments,
    hmac_key_file: Option<&Path>,
) -> Result<()> {
    let output_parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let output_name = output
        .file_name()
        .context("certificate output path must name a file")?;
    let normalized_output = if output.exists() {
        std::fs::canonicalize(output)?
    } else {
        std::fs::canonicalize(output_parent)?.join(output_name)
    };
    let artifact_paths = artifact_arguments.paths();
    let mut inputs = vec![
        transcript,
        artifact_paths.base_model.as_path(),
        artifact_paths.encoded_hypothesis.as_path(),
        artifact_paths.hypothesis_decoder_spec.as_path(),
        artifact_paths.generation_spec.as_path(),
        artifact_paths.reward_spec.as_path(),
        artifact_paths.prompt_distribution_manifest.as_path(),
        artifact_paths.training_corpus_manifest.as_path(),
    ];
    if let Some(key_path) = hmac_key_file {
        inputs.push(key_path);
    }
    for input in inputs {
        let same_canonical_path = std::fs::canonicalize(input)? == normalized_output;
        #[cfg(unix)]
        let same_file_identity = if output.exists() {
            use std::os::unix::fs::MetadataExt;
            let output_metadata = std::fs::metadata(output)?;
            let input_metadata = std::fs::metadata(input)?;
            output_metadata.dev() == input_metadata.dev()
                && output_metadata.ino() == input_metadata.ino()
        } else {
            false
        };
        #[cfg(not(unix))]
        let same_file_identity = false;
        if same_canonical_path || same_file_identity {
            bail!(
                "certificate output {} aliases input evidence {}",
                output.display(),
                input.display()
            );
        }
    }
    Ok(())
}

fn check(
    certificate_path: &Path,
    transcript_path: &Path,
    artifact_arguments: &ArtifactArguments,
    hmac_key_file: Option<&Path>,
    json: bool,
) -> Result<()> {
    let certificate = GeneralizationCertificate::load(certificate_path)
        .with_context(|| format!("failed to load certificate {}", certificate_path.display()))?;
    let hmac_signature_verified = match (certificate.hmac_signature.is_some(), hmac_key_file) {
        (true, Some(key_path)) => {
            let key = read_hmac_key(key_path)?;
            nn_verify::verify_signature(&certificate, &key)
                .context("generalization certificate HMAC verification failed")?;
            true
        }
        (true, None) => bail!(
            "certificate contains an HMAC signature; pass --hmac-key-file to verify provenance"
        ),
        (false, Some(_)) => bail!("an HMAC key was supplied but the certificate is unsigned"),
        (false, None) => false,
    };
    let transcript = std::fs::read(transcript_path)
        .with_context(|| format!("failed to read transcript {}", transcript_path.display()))?;
    let mut report = certificate
        .replay(&transcript, &artifact_arguments.paths())
        .context("generalization certificate replay failed")?;
    report.hmac_signature_verified = hmac_signature_verified;

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("Generalization certificate replay: PASS");
        println!(
            "  identities:  {} artifacts + transcript",
            report.evidence_artifacts_verified
        );
        println!("  empirical:   {:.9}", report.empirical_reward);
        println!("  certified:   {:.9}", report.certified_lower_bound);
        println!("  confidence:  {:.2}%", report.confidence * 100.0);
        println!(
            "  assessment:  {}",
            if report.non_vacuous {
                "NON-VACUOUS"
            } else {
                "VACUOUS"
            }
        );
        if report.hmac_signature_verified {
            println!("  hmac:        VERIFIED");
        }
    }
    Ok(())
}

fn read_hmac_key(path: &Path) -> Result<Vec<u8>> {
    let key = std::fs::read(path)
        .with_context(|| format!("failed to read HMAC key {}", path.display()))?;
    if key.len() < MINIMUM_HMAC_KEY_BYTES {
        bail!(
            "HMAC key {} is {} bytes; at least {} bytes are required",
            path.display(),
            key.len(),
            MINIMUM_HMAC_KEY_BYTES
        );
    }
    Ok(key)
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;
    use crate::{Cli, Commands};

    fn artifact_flags() -> Vec<&'static str> {
        vec![
            "--base-model",
            "base.bin",
            "--encoded-hypothesis",
            "adapter.bits",
            "--hypothesis-decoder-spec",
            "encoding.json",
            "--generation-spec",
            "generation.json",
            "--reward-spec",
            "reward.json",
            "--prompt-distribution-manifest",
            "distribution.json",
            "--training-corpus-manifest",
            "corpus.json",
        ]
    }

    #[test]
    fn test_generalize_calculate_parses() {
        let mut args = vec!["nn", "generalize", "calculate", "rewards.json"];
        args.extend(artifact_flags());
        args.extend([
            "--encoding-scheme",
            "tiny-lora-kt-v1",
            "--description-bits",
            "7726",
            "--output",
            "reward.cert.json",
        ]);
        let cli = Cli::try_parse_from(args).unwrap();
        assert!(matches!(
            cli.command,
            Commands::Generalize {
                command: GeneralizeCommand::Calculate {
                    description_bits: 7726,
                    delta: 0.05,
                    ..
                }
            }
        ));
    }

    #[test]
    fn test_generalize_check_requires_all_artifacts() {
        let result = Cli::try_parse_from([
            "nn",
            "generalize",
            "check",
            "reward.cert.json",
            "rewards.json",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_generalization_certificate_roundtrips_in_cli_build() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let license = workspace.join("LICENSE");
        let training_manifest = workspace.join("examples/generalization-corpus.json");
        let paths = EvidenceArtifactPaths::new(
            license.clone(),
            license.clone(),
            license.clone(),
            license.clone(),
            license.clone(),
            license.clone(),
            training_manifest.clone(),
        );
        let artifacts = EvidenceArtifacts::from_paths(&paths).unwrap();
        let description_bits = artifacts.encoded_hypothesis.byte_len * 8;
        let inputs = GeneralizationCertificateInputs::new(
            artifacts,
            HypothesisEncoding::new("cli-roundtrip-test", description_bits),
            0.05,
        );
        let transcript = include_bytes!("../../../examples/generalization-transcript.json");
        let manifest = std::fs::read(training_manifest).unwrap();
        let certificate =
            GeneralizationCertificate::calculate(inputs, transcript, &manifest).unwrap();
        let json = certificate.to_json().unwrap();
        let decoded = GeneralizationCertificate::from_json(&json).unwrap();
        assert_eq!(decoded, certificate);
        decoded.validate().unwrap();
    }

    #[test]
    fn test_generalize_calculate_then_check_end_to_end() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let license = workspace.join("LICENSE");
        let training_manifest = workspace.join("examples/generalization-corpus.json");
        let transcript = workspace.join("examples/generalization-transcript.json");
        let artifacts = ArtifactArguments {
            base_model: license.clone(),
            encoded_hypothesis: license.clone(),
            hypothesis_decoder_spec: license.clone(),
            generation_spec: license.clone(),
            reward_spec: license.clone(),
            prompt_distribution_manifest: license.clone(),
            training_corpus_manifest: training_manifest,
        };
        let description_bits = std::fs::metadata(&license).unwrap().len() * 8;
        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("nested/model.generalization.json");
        let key_path = temporary.path().join("generalization.hmac-key");
        std::fs::write(&key_path, b"cli-e2e-generalization-key-32bytes").unwrap();

        calculate(
            &transcript,
            &artifacts,
            "cli-e2e-prefix-v1",
            description_bits,
            0.05,
            &output,
            Some(&key_path),
            false,
        )
        .unwrap();
        assert!(output.is_file());
        assert!(check(&output, &transcript, &artifacts, None, false).is_err());
        let wrong_key_path = temporary.path().join("wrong.hmac-key");
        std::fs::write(&wrong_key_path, b"wrong-generalization-key-at-least-32").unwrap();
        assert!(check(
            &output,
            &transcript,
            &artifacts,
            Some(&wrong_key_path),
            false,
        )
        .is_err());
        check(&output, &transcript, &artifacts, Some(&key_path), false).unwrap();
    }

    #[test]
    fn test_generalize_rejects_short_hmac_key() {
        let temporary = tempfile::tempdir().unwrap();
        let key_path = temporary.path().join("short.hmac-key");
        std::fs::write(&key_path, b"too-short").unwrap();
        assert!(read_hmac_key(&key_path).is_err());
    }

    #[test]
    fn test_generalize_output_cannot_alias_evidence() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let license = workspace.join("LICENSE");
        let artifacts = ArtifactArguments {
            base_model: license.clone(),
            encoded_hypothesis: license.clone(),
            hypothesis_decoder_spec: license.clone(),
            generation_spec: license.clone(),
            reward_spec: license.clone(),
            prompt_distribution_manifest: license.clone(),
            training_corpus_manifest: workspace.join("examples/generalization-corpus.json"),
        };
        assert!(ensure_output_does_not_alias_input(&license, &license, &artifacts, None).is_err());

        let bare_output = PathBuf::from(format!(
            ".nn-generalization-bare-output-test-{}.json",
            std::process::id()
        ));
        assert!(!bare_output.exists());
        assert!(
            ensure_output_does_not_alias_input(&bare_output, &license, &artifacts, None).is_ok()
        );
    }
}
