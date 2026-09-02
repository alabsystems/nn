# NN — Verified ML Framework

Apache-2.0 · [github.com/alabsystems/nn](https://github.com/alabsystems/nn)

A Rust-native ML framework where model code, GPU kernels, and proof tooling live together.

- **Kani-backed kernel verification** — write a GPU kernel in Rust; `#[nn::kernel]` emits the Metal shader, a proof harness, and a differential test.
- **Metal as the production backend** — compiled `CompiledModel` dispatch with op fusion and weight uploads.
- **Exported-artifact import bridge** — `torch.export` JSON + `safetensors` → optimized Metal model with a structured `ConvertReport`.
- **Bound propagation** — `ny` IBP/CROWN over composed networks; `ay` SMT for small subgraphs.
- **Population-reward certificates** — replayable compression-based PAC-Bayes
  lower bounds for frozen, prefix-coded models and sealed reward transcripts.

## Install

The `nn` CLI ships as a signed package through `atpkg` (the
[aterm](https://github.com/alabsystems/aterm) package manager):

```sh
atpkg install nn
```

As a library, depend on the public snapshot — its manifest pins the published
`ny`/`ay` releases, so it resolves standalone:

```toml
nn = { git = "https://github.com/alabsystems/nn", features = ["metal"] }
```

## CLI

```sh
nn device                                   # device + pipeline-cache report; no input files
nn convert  graph.json weights.safetensors  # compile + print a ConvertReport
nn compile  graph.json weights.safetensors -o model.nnc
nn run      graph.json weights.safetensors --input in.safetensors [--compiled model.nnc]
nn optimize model.nnc --graph graph.json --weights weights.safetensors
```

Every subcommand but `nn device` needs a `graph.json` + `weights.safetensors` pair;
`nn convert --from-pytorch --model-spec module:Class --input-shape …` shells out to
`nn_export.py` to produce them. This is not raw ONNX intake.

## Quick start

Build a model from scratch:

```rust
use nn::{DType, Device, DynTensor, Module, Result, VarBuilder};
use nn::{linear, embedding, layer_norm};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let emb = embedding(1000, 128, &vb.pp("embed"))?;
    let ln = layer_norm(128, Default::default(), &vb.pp("ln"))?;
    let proj = linear(128, 32, &vb.pp("proj"))?;

    let ids = DynTensor::from_vec_u32(vec![1, 42, 7], &[3], &device)?;
    let x = emb.forward(&ids)?;
    let x = ln.forward(&x)?;
    let out = proj.forward(&x)?;

    assert_eq!(out.dims(), &[3, 32]);
    Ok(())
}
```

Compile exported PyTorch artifacts to Metal (needs `import-metal`):

```rust
use std::path::Path;
use nn::{convert, metal::PipelineCache, OptLevel};

let cache = PipelineCache::new_global()?;
let result = convert(
    Path::new("model_graph.json"),    // from torch.export
    Path::new("weights.safetensors"),
    &cache,
)
.optimize(OptLevel::Full)             // the default
.build()?;

let output = result.result.model.execute_dyn(&cache, &[&input])?;
```

## Features

| Feature | Purpose |
|---------|---------|
| `metal` | Apple Metal GPU backend (macOS only) |
| `dsl` | `#[nn::kernel]` / `#[nn::model]` proc-macros |
| `training` | Reverse-mode autodiff + optimizers (AdamW, SGD, AdaFactor, LoRA) |
| `import` | `torch.export` JSON parsing and graph building |
| `import-metal` | `nn::convert()` — full compiled bridge to Metal |
| `models` | Kokoro TTS, HTDemucs, ECAPA-TDNN, Silero VAD, PLBert |
| `whisper` / `qwen3` / `glm5` | Individual STT and LLM model crates |
| `gguf` | GGUF weight-format parser |
| `verify` | `ny` bound propagation over compiled models (implies `metal`) |
| `reftest` | Reference-tensor comparison tooling |

## Layout

`crates/nn` is the umbrella facade over a workspace split by concern:

- **Core** — `nn-core` (tensors, `DynTensor`, ops, modules); `nn-dsl` / `nn-macros` (eDSL, proc-macros)
- **Backends** — `nn-metal` (production), `nn-cuda`, `nn-vulkan`, `nn-cpu`
- **Import** — `nn-import` (`torch.export` → traced graph → compiled model), `nn-gguf`
- **Models** — `nn-models`, `nn-whisper`, `nn-qwen3`, `nn-glm5`, `nn-gptoss`
- **Training** — `nn-autodiff`, `nn-optim`
- **Verification** — `nn-verify`, `nn-tts-verify`, `nn-reftest`
- **Tooling** — `nn-cli`, the `nn` binary

## Verification

NN combines GPU kernel formal verification (Kani / CBMC) with neural-network
bound propagation (`ny`, `ay`). Proofs operate on Rust source.

| Level | Method | What it catches |
|-------|--------|----------------|
| 0 | Type system | Rank mismatches at compile time |
| 1 | Static analysis | Dimension mismatches at runtime |
| 2 | IBP (`ny`) | Unbounded outputs, NaN/Inf |
| 3 | CROWN (`ny`) | Tighter-than-IBP correlations where wired |
| 4 | SMT (`ay`) | Complete verification of small subgraphs |

Levels 2–3 are what `verify` turns on. Level 4 emits SMT-LIB2 with
uninterpreted-function approximation for transcendentals by default; in-process
solving is the opt-in `nn-verify/ay-smt` feature. Kani harnesses are `#[cfg(kani)]`
items, not a Cargo feature — build under Kani to run them.

`silero_vad.proof.json` is a worked certificate bundle — read
`silero_vad.proof.README.md` first: it is schema and pipeline evidence over a
synthetic-weight graph, **not** deployment evidence.
`designs/architecture-certified-pipeline.md` sketches the intended end state.

Statistical generalization is a separate assurance dimension from Levels 0–4.
`nn generalize calculate` implements the population-reward bounds from
arXiv:2607.14506v1 and `nn generalize check` replays artifact identities,
corpus/transcript linkage, subsample selection, aggregation, and bound
arithmetic. It does not turn an average-case distributional claim into a
universal network-property proof.

## Status

Pre-1.0. Metal is the production path, and it is macOS-only. The workspace also
builds on x86_64 Linux, where `nn device` and `nn optimize` work and the GPU
subcommands decline up front — no CPU inference path is wired into the CLI.
CUDA/Vulkan/CPU backends compile but have surface gaps. Verification coverage is
partial: useful, not yet end-to-end proof-powered.

`v0.3.0` is the current published tag. This snapshot is source-only and carries
no lockfile, so it is a build target, not a reproducible one.

Development builds against the Trust verifying compiler; this snapshot pins
stock upstream Rust instead, because the `trust` channel is a locally linked
toolchain no outside clone can resolve. Trust's in-compilation verification is
switched off repo-wide even in the development build, so it contributes nothing
to the levels above today — those are `nn-verify`'s own bound-propagation and
SMT lanes plus the Kani harnesses, run as tests.

Turning it back on is written and pushed but not merged: the development branch
`trust/sanctioned-verify-split` verifies first-party crates while leaving build
scripts, proc-macros and third-party dependencies alone, and it is proven to
issue exactly those flags. It is blocked on compiler throughput, not on this
code — verifying `nn-core` did not finish in over an hour of CPU, in an
unmemoized per-statement recompute inside the verifier's VC generator. Every
other crate depends on `nn-core`, so no crate in this workspace has yet been
measured under verification.

## License

Apache-2.0 © Andrew Yates
