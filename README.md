# NN — Verified ML Framework

**v0.1.0** · Apache-2.0 · [github.com/alabsystems/nn](https://github.com/alabsystems/nn)

A Rust-native ML framework where model code, GPU kernels, and proof tooling live together.

- **Kani-backed kernel verification** — write a GPU kernel in Rust, get a proof harness and differential test for free.
- **Metal as the production backend** — compiled `CompiledModel` dispatch with op fusion and weight uploads.
- **Exported-artifact import bridge** — `torch.export` JSON + `safetensors` → optimized Metal model with a structured `ConvertReport`.
- **Bound propagation** — partial NY / ay integration for verifying composed networks.

## Install

The `nn` CLI ships as a signed package through `atpkg` (the
[aterm](https://github.com/alabsystems/aterm) package manager):

```sh
atpkg install nn
```

To use nn as a library, depend on the public snapshot — its manifest pins the
published `ny`/`ay` releases, so it resolves standalone:

```toml
nn = { git = "https://github.com/alabsystems/nn", features = ["metal"] }
```

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

Compile exported PyTorch artifacts to Metal:

```rust
use std::path::Path;
use nn::{convert, metal::PipelineCache, OptLevel};

let cache = PipelineCache::new()?;
let result = convert(
    Path::new("model_graph.json"),    // from torch.export
    Path::new("weights.safetensors"),
    &cache,
)
.optimize(OptLevel::Full)
.build()?;

let output = result.result.model.execute_dyn(&cache, &[&input])?;
```

## Features

| Feature | Purpose |
|---------|---------|
| `metal` | Apple Metal GPU backend |
| `dsl` | `#[nn::kernel]` / `#[nn::model]` proc-macros |
| `training` | Reverse-mode autodiff + optimizers (AdamW, SGD, LoRA) |
| `import` | `torch.export` JSON parsing and graph building |
| `import-metal` | `nn::convert()` — full compiled bridge to Metal |
| `models` | Kokoro TTS, Whisper, Qwen3, GLM-5, HTDemucs, ECAPA-TDNN, Silero VAD |
| `verify` | Kani / NY / ay verification hooks |
| `reftest` | Reference-tensor comparison tooling |

## Layout

`crates/nn` is the umbrella facade. The workspace is split by concern:

- `nn-core` — tensor types, `DynTensor`, op kernels, nn modules
- `nn-metal` — Metal backend, compiled-model pipeline, fusion passes
- `nn-cuda` / `nn-vulkan` / `nn-cpu` — additional backend targets
- `nn-import` — `torch.export` → traced graph → compiled model
- `nn-dsl` / `nn-macros` — eDSL and proc-macros for kernels and models
- `nn-models` — model implementations (Kokoro, Whisper, etc.)
- `nn-whisper` / `nn-qwen3` / `nn-glm5` / `nn-gptoss` — model-specific crates (STT, Qwen3, GLM, GPT-OSS MoE)
- `nn-gguf` — GGUF weight-format parser
- `nn-autodiff` / `nn-optim` — training infrastructure
- `nn-verify` / `nn-tts-verify` — verification surfaces (NY bound propagation + ay SMT)
- `nn-cli` / `nn-optimize` — `nn` command-line tool and dispatch-plan optimizer

## Verification

NN combines GPU kernel formal verification (Kani / CBMC) with neural-network bound propagation (NY, ay). Proofs operate on Rust source; Metal is the production backend today, with broader model-property coverage still in progress.

| Level | Method | What it catches |
|-------|--------|----------------|
| 0 | Type system | Rank mismatches at compile time |
| 1 | Static analysis | Dimension mismatches at runtime |
| 2 | IBP (NY) | Unbounded outputs, NaN/Inf |
| 3 | CROWN (NY) | Tighter-than-IBP correlations where wired |
| 4 | SMT (ay) | Complete verification of small subgraphs |

## Status

Pre-1.0. Metal is the production path. CUDA/Vulkan/CPU backends compile but have surface gaps. Verification coverage is partial — useful, not yet end-to-end proof-powered.

## License

Apache-2.0 © Andrew Yates
