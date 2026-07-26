# Architecture: `CertifiedPipeline` + ay-constraint synthesis

**Date:** 2026-05-18 · **Author:** Andrew Yates · **Status:** Design proposal

NN's verification and synthesis surface centers on one typed object — `CertifiedPipeline` — that owns stages, region contracts, transformation/refinement certificates, backend lowering witnesses, and a serialized replayable proof bundle. Verification, elimination, lowering, and synthesis are all *passes* over this object.

## The three moves

1. **One central abstraction: `CertifiedPipeline`.** A single typed object covers the surface area that would otherwise sprawl across many verification-side crates.
2. **Synthesis as a first-class operation.** Verification answers "does this network satisfy this property?". Synthesis answers "given this property, synthesize the network." Verification is then the degenerate case where the network is pre-specified.
3. **ay as a backend, not just a checker.** ay already supports CHC (Constrained Horn Clauses) and SMT. NN encodes network-with-constraints problems for ay to solve.

## The core abstraction: `CertifiedPipeline`

A `CertifiedPipeline` is a typed sequence of stages, where each stage has:

- **Input region contract** — `BoundedTensor` with shape, dtype, and provenance
- **Computation** — either a concrete `Block` (ResNet block, transformer block, etc.) or a `Hole` (to be synthesized)
- **Output region contract** — bounds derived (verified) or asserted (synthesized)
- **Soundness witness** — produced at this joint, checkable downstream
- **Transformation certificate** (if stage is a transformation) — proof that this stage refines the previous

```rust
pub struct CertifiedPipeline {
    pub stages: Vec<Stage>,
    pub source_identity: ContentHash,    // hash of original spec
    pub backend_target: BackendTarget,   // Cpu | Metal | Ane | Gpu
    pub soundness_mode: SoundnessMode,   // Strict | Best-effort | Unchecked
    pub serialized_proof: ProofBundle,   // replayable certificate
}

pub enum Stage {
    Block { spec: BlockSpec, bound_cert: BoundCertificate },
    Transform { from: BlockId, to: BlockId, refinement_cert: RefinementCertificate },
    Hole { constraints: Vec<Constraint>, target: HoleTarget },
}
```

Everything in NN is a **pass** over this object:

| Operation | Pass | Input | Output |
|---|---|---|---|
| Verify network on input region | `verify` | `CertifiedPipeline` with concrete stages | `CertifiedPipeline` with bound certs |
| Eliminate dead neurons | `eliminate` | verified pipeline | new pipeline with `Transform` stages |
| Prove pair-equivalence | `verify_pair` | two pipelines + ε | equivalence cert linking them |
| Lower to Metal | `lower::metal` | pipeline | pipeline + backend witnesses |
| Streaming-verify (e.g. Whisper) | `verify_streaming` | pipeline | pipeline w/ memory-bounded proof |
| **Synthesize from constraints** | `synthesize::ay` | pipeline with `Hole` stages + constraints | pipeline with concrete blocks + synthesis witness |
| Replay a proof bundle | `replay` | `CertifiedPipeline` + frozen `ProofBundle` | bool: certificate verifies |

One cohesive object family with one notion of "certificate."

## AY-constraint synthesis

Specify the network as a set of constraints and have ay solve for it.

### API sketch

```rust
let problem = ConstraintProblem {
    architecture_template: ArchTemplate::transformer_decoder(layers: 6, dim: 512),
    input_region: BoundedTensor::box(low: -1.0, high: 1.0, shape: [1, 128]),
    output_constraints: vec![
        OutputConstraint::Bounded { axis: -1, lower: 0.0, upper: 1.0 },
        OutputConstraint::AtPoint { input: example_input_0, output: target_output_0 },
        OutputConstraint::RobustOn { region: robustness_region, class: 7 },
        OutputConstraint::EquivalentTo { reference: ref_pipeline, eps: 0.01 },
        OutputConstraint::ZeroActivation { input_region: bad_region, neurons: vec![23, 47] },
    ],
    objective: SynthesisObjective::MinimizeSize {
        target_params: 100_000,
        weight: 0.7,
    },
    backend: SolverBackend::AY { timeout: Duration::from_hours(24) },
    soundness_mode: SoundnessMode::Strict,
};

let pipeline = CertifiedPipeline::from_hole(problem);
match pipeline.synthesize_ay() {
    SynthesisResult::Found(certified_pipeline) => {
        // carries concrete weights, ay synthesis witness (the CHC derivation),
        // bound certificates per stage, ready to lower.
    }
    SynthesisResult::Unsat { core } => {
        // ay returned UNSAT-core: which constraints were jointly infeasible.
    }
    SynthesisResult::Unknown { partial, reason } => {
        // Hit timeout; partial assignment available for warm-start.
    }
}
```

### How ay solves it

NN compiles the `ConstraintProblem` to a CHC system:

1. **Network as CHC predicates.** Each layer becomes a relation `Layer_i(input, output) ↔ output = activation(W_i × input + b_i)`. Weights and biases are existentially quantified.
2. **Input-region constraints as bound predicates.** `InRegion(x) ↔ low ≤ x ≤ high`.
3. **Output constraints as CHC implications.**
   - `Bounded`: `InRegion(x) → Output(x, y) → low ≤ y ≤ high`
   - `AtPoint`: `Output(input_0, y_0)` (concrete)
   - `RobustOn`: `InRegion(robust_region, x) → argmax(Output(x, y)) = class`
   - `EquivalentTo`: `Output(x, y_1) → Reference(x, y_2) → |y_1 - y_2| ≤ ε`
   - `ZeroActivation`: `InRegion(bad, x) → Neuron_i(x) ≤ 0` for each target i
4. **Objective as MaxSMT.** Minimize_size becomes a soft constraint over weight non-zeros.
5. **ay solves.** Returns concrete weight values or UNSAT-core.

This is program synthesis from specifications, with proof.

### Why ay

- CHC-COMP-tier CHC solver
- Tier-0 scalar safety extraction
- Native integration with NY bound propagation (already in tree)

NN's existing `gamma_propagate::ay` integration is the synthesis backend's foundation. No new dependency.

### Practical scope per problem class

| Problem | Feasibility |
|---|---|
| Synthesize 1-layer linear network from ≤ 20 input/output examples | minutes |
| Synthesize 2-layer ReLU network satisfying a robustness region | hours; often UNSAT or UNKNOWN |
| Synthesize a 6-layer transformer matching N examples | impractical without architecture priors |
| Synthesize pruning mask: given network N, find S ⊆ neurons to keep s.t. equivalence holds within ε | tractable (dual of `eliminate_and_verify`) |
| Synthesize quantization schedule: assign bit-widths per layer to maximize accuracy | tractable per-block |
| Synthesize a small student network ≡ a large teacher, within ε on training distribution | research-grade hard |

The first two and the dual-of-elimination are the immediate wins.

## Design contracts

- **One core abstraction (`CertifiedPipeline`), N passes** — no per-feature crate sprawl.
- **Dtype is part of region contract; the contract is checkable, not assumed.**
- **Neuron classification is a pluggable strategy** documented per `Layer` type; `eliminate` operates on bounded-tensor abstraction, not raw bounds.
- **Performance contract per pass; benchmark targets per pass; failure modes documented per backend.**
- **`BackendTarget` is explicit per pipeline; the Metal lane has a separate witness + parity test pass.**
- **Streaming summaries selectable: interval, zonotope, polytope** — richer abstractions where needed, with an explicit correlation-budget knob.
- **Synthesis is ay-CHC-based.** PGD is an optional `Falsifier` strategy with a budget, not the primary loop.
- **`ProofBundle` is first-class:** versioned, hash-keyed, replayable, machine-readable JSON+CBOR.
- **`BackendTarget` lowering passes produce a `LoweringWitness`** linking source pipeline to compiled binary.

## Workspace shape

```
nn-core
  ├── pipeline/           CertifiedPipeline, Stage, Hole, ContentHash
  ├── certificate/        BoundCertificate, RefinementCertificate, ProofBundle, replay
  ├── region/             BoundedTensor variants, dtype contracts, region operations
  ├── soundness/          SoundnessMode, SoundnessWitness, joint-checking
  ├── backend/            BackendTarget, LoweringWitness (trait)
  └── passes/             traits for Verify, Eliminate, Lower, Synthesize, Replay

nn-passes-verify          verify pass (uses NY bound propagation)
nn-passes-eliminate       eliminate-via-equivalence pass (composition-based)
nn-passes-synthesize-ay   synthesize pass (ay-CHC backend)
nn-passes-lower-metal     lower pass to Metal kernels (existing nn-metal)
nn-passes-stream          streaming verify with richer summaries

nn-dsl                    eDSL for model spec (produces `CertifiedPipeline`)
nn-models                 model catalog
```

Each pass crate is one cohesive concept.

## Build order

1. **`CertifiedPipeline` core** in `nn-core`. Port existing `nn-verify` functionality to a `Verify` pass.
2. **Passes:** implement `Eliminate`, `Lower::Metal`, `Stream`.
3. **`synthesize_ay`** — the ambitious bet. First milestones:
   - 1-layer linear network from N examples (sanity)
   - "Find pruning mask such that equivalence holds" (high-value; dual of elimination)
   - "Synthesize robust 2-layer network on a small region" (research milestone)

## Open questions

- **`Hole` semantics for nested architectures.** How do you specify "a transformer block of arbitrary attention head count, to be chosen by ay"? Likely template-instantiation + finite search, not pure ay.
- **Synthesis scalability.** Can ay actually solve 1M-param network synthesis? Probably not directly — need architecture priors, decomposition into block-wise synthesis, and warm-starting from a candidate.
- **Proof bundle interop.** Does the `ProofBundle` format align with the NY audit pipeline? Likely yes; needs explicit alignment.
