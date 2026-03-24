# Capability Cartography Layer

**Capability Cartography Layer** is a standalone research framework for turning transparent educational model implementations into a cumulative empirical instrument for studying capability formation, failure geometry, compressibility structure, retrieval dependence, and phase-like behavioral transitions in language models.

This repository is designed to sit on top of three companion resources:

- `pageman/sutskever-30-implementations` as the experimental substrate
- `pageman/Sutskever-Agent` as the orchestration and explanatory layer
- `pageman/gpt1-from-sutskever30` as the first controlled transformer wind tunnel

These links are preserved explicitly in code and artifacts. The adapter layer records canonical repository URLs and configured local roots so every cartography export can retain its provenance back to those three repos rather than treating them as anonymous backends.

The central claim of this project is simple: benchmark folklore is not enough. If a model is “surprisingly strong,” “brittle,” “emergent,” or “lost in the middle,” those labels should resolve into measurable regions, descriptor profiles, compressibility signatures, and threshold estimates. This repo exists to provide the scaffolding for exactly that.

## Why This Repository Exists

Most open educational model repositories are optimized for understanding how a model is built. That is useful, but incomplete. The deeper scientific question is not merely how a mechanism is implemented, but under what conditions a capability appears, stabilizes, degrades, or collapses.

That question becomes especially important when dealing with:

- retrieval-augmented models that succeed only under favorable context geometry
- reasoning benchmarks that conflate memorization with abstraction
- compact demo models whose transparency makes them ideal for causal intervention
- claims of “emergence” that are really unmeasured threshold behavior
- claims of “brittleness” that are really undocumented interactions between task structure, distractors, and objective choice

The Capability Cartography Layer reframes those issues as a mapping problem. Instead of relying on single scalar benchmark scores, it creates structured artifacts that describe:

- what kind of task the model was given
- what the intervention profile was
- how compressible the task and model behavior looked
- where capability seemed to turn on or turn off
- whether failure was gradual, abrupt, or masked by misleadingly favorable conditions

## Conceptual Model

The repository is organized around five tightly linked subsystems.

### 1. Standardized Experiment System

The system treats each experiment as a capability probe rather than a one-off training script. An experiment has:

- a substrate
- a task
- a benchmark label
- a realism level
- an intervention profile
- one or more measured trajectories

This creates a common contract across:

- educational notebook assets
- compact hand-built transformer code
- agent-generated narratives

The goal is to allow cumulative measurement rather than isolated demos.

### 2. Task Descriptor System

Every task or evaluation instance is described in structured form rather than by benchmark name alone. The descriptor system captures several families of signal:

- surface statistics
- latent structure proxies
- retrieval geometry
- perturbation profile
- cognitive-operation hints
- structural complexity proxies

That matters because a benchmark label such as “QA,” “math,” or “reasoning” is too coarse to predict success or failure. Two tasks can share a label while differing sharply in distractor density, relational depth, temporal complexity, or answer position bias.

### 3. Compressibility Stack

The compressibility layer estimates three different kinds of compression:

- surface compression using standard codecs
- predictive compression using loss-style proxies
- structural compression using description-length-like approximations

The objective is not to claim perfect MDL or Kolmogorov measurements. It is to provide practical, comparable proxies that reveal mismatch. If a task looks statistically easy at the surface level but still demands a structurally costly representation, that gap is informative. If predictive loss is low but structural compression remains poor, that is also informative.

### 4. Intervention System

Capabilities should not be described only after they appear. They should be stress-tested and moved. The intervention system therefore exposes knobs over:

- architecture
- objective
- data regime
- retrieval
- context geometry
- interpretability-related settings

This is where the GPT-1 companion project becomes especially useful. A compact, inspectable transformer is the right place to run one-factor-at-a-time sweeps before moving to larger or less transparent systems.

### 5. Boundary Analysis System

The end product of the framework is not just a log file. It is a map. The boundary layer looks for:

- changepoints
- threshold estimates
- competence regimes
- phase-like transitions

This makes it possible to say more than “the model got better.” Instead, one can say:

- capability score crossed from collapse into partial competence at a particular step
- the median threshold for a sweep sits at a specific value
- retrieval dependence remains high even after apparent capability gains
- a success region is narrow and bordered by rapid collapse under perturbation

## What Is In This Repository

### `capability_cartography/`

The main Python package.

- [`capability_cartography/schemas.py`](./capability_cartography/schemas.py)
  Defines the shared dataclasses for descriptors, trajectories, interventions, artifacts, and boundary summaries.

- [`capability_cartography/descriptors.py`](./capability_cartography/descriptors.py)
  Extracts task descriptors from text or arrays.

- [`capability_cartography/compressibility.py`](./capability_cartography/compressibility.py)
  Computes surface, predictive, and structural compression proxies plus gap measures.

- [`capability_cartography/boundary.py`](./capability_cartography/boundary.py)
  Detects abrupt regime shifts and fits lightweight threshold summaries.

- [`capability_cartography/adapters.py`](./capability_cartography/adapters.py)
  Integrates with external repos through configurable roots rather than hardcoded local assumptions.

- [`capability_cartography/runner.py`](./capability_cartography/runner.py)
  Provides the shared runner that ties descriptors, compressibility, interventions, wind-tunnel probing, and export together.

- [`capability_cartography/demo.py`](./capability_cartography/demo.py)
  Demonstrates the full pipeline and writes JSON artifacts to `./artifacts/`.

- [`capability_cartography/sweeps.py`](./capability_cartography/sweeps.py)
  Runs scale/data/task-family grids and builds a sweep registry suitable for onset-surface analysis.

- [`capability_cartography/surfaces.py`](./capability_cartography/surfaces.py)
  Fits lightweight predictive surfaces and onset thresholds across sweep records.

- [`capability_cartography/storage.py`](./capability_cartography/storage.py)
  Persists per-run artifacts and tabular sweep summaries.

- [`capability_cartography/metrics.py`](./capability_cartography/metrics.py)
  Computes aggregate metrics, calibration-style error, and capability proxies for measured synthetic trajectories.

### `tests/`

Basic unit tests using the Python standard library’s `unittest` module.

## Repository Positioning Relative to the Companion Repos

This repository is intentionally separate from the educational substrate. That separation is deliberate for several reasons:

- the measurement layer evolves on different timescales than the underlying notebooks
- the cartography layer should be reusable across multiple substrates, not only one
- it should be publishable and versioned independently
- it should be able to depend on external local paths or clones without requiring those repos to absorb measurement-specific logic

In practical terms:

- `sutskever-30-implementations` remains the transparent experimental substrate
- `gpt1-from-sutskever30` remains the compact wind tunnel
- `Sutskever-Agent` remains the orchestration and explanatory layer
- this repository becomes the measurement spine that links them

## Installation

### Minimal installation

```bash
git clone https://github.com/pageman/Capability-Cartography-Layer
cd Capability-Cartography-Layer
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the package itself with its lightweight dependencies:

- `numpy`
- `PyYAML`

### Optional companion repositories

If you want live integration with the three companion repos rather than fallback behavior, clone them somewhere on disk and then point this repository at them via environment variables:

```bash
export SUTSKEVER30_ROOT=/path/to/sutskever-30-implementations
export GPT1_WIND_TUNNEL_ROOT=/path/to/gpt1-from-sutskever30
export SUTSKEVER_AGENT_ROOT=/path/to/Sutskever-Agent/sutskever-agent
```

The adapters are designed so that:

- if the substrate repo is configured, notebook metadata can be discovered directly
- if the GPT-1 repo is configured, the runner can import and probe the actual GPT-1 implementation
- if the agent repo is configured, skill metadata can be read from the agent manifest
- if some repos are missing, the framework still works in reduced mode

They also try a small set of likely local clone locations automatically. On a machine with the repos cloned in common places under `~` or `~/Downloads`, linked mode can work without exporting any environment variables.

Every exported artifact now includes `linked_repositories` metadata referencing:

- `https://github.com/pageman/Sutskever-30-implementations`
- `https://github.com/pageman/Sutskever-Agent`
- `https://github.com/pageman/gpt1-from-Sutskever30`

## Quick Start

Run the demo:

```bash
python3 -m capability_cartography.demo
```

This will:

1. define a small intervention profile
2. construct a descriptor-bearing text experiment
3. compute compressibility proxies
4. detect boundary events
5. generate a narrative summary
6. export artifacts to `./artifacts/`
7. run a GPT-1 wind tunnel profile using the configured adapter or a fallback dry-run approximation
8. run a multi-axis sweep over scale, data volume, and task family
9. export sweep records and surface summaries to `./artifacts/sweeps/`
10. run a measured study using actual tiny GPT-1 training loops on task families derived from the linked substrate
11. export held-out validation, bootstrap intervals, and falsifiable law statements to `./artifacts/measured/`

## Linked Mode

If the companion repositories already exist in standard local locations, this is enough:

```bash
cd Capability-Cartography-Layer
python3 -m capability_cartography.demo
```

If you want to force specific linked roots, use:

```bash
SUTSKEVER30_ROOT=/path/to/sutskever-30-implementations \
SUTSKEVER_AGENT_ROOT=/path/to/Sutskever-Agent/sutskever-agent \
GPT1_WIND_TUNNEL_ROOT=/path/to/gpt1-from-Sutskever30 \
python3 -m capability_cartography.demo
```

When linked mode is active, the exported artifacts will show `available: true` in `linked_repositories` and will record the configured local roots for:

- `pageman/Sutskever-30-implementations`
- `pageman/Sutskever-Agent`
- `pageman/gpt1-from-Sutskever30`

## Example Output Artifacts

The exported JSON artifacts are designed to be:

- machine-readable
- easy to diff
- easy for agents to narrate
- stable enough to serve as research audit trail records

An artifact contains:

- the experiment spec
- the full capability trajectory
- descriptor payloads
- compressibility summaries
- boundary events
- fitted boundary statistics
- aggregate series metrics
- linked companion-repository metadata
- an optional narrative layer

The measured study outputs also contain:

- repeated-seed records
- holdout validation metrics
- bootstrap coefficient intervals
- provenance with linked repository commits
- falsifiable law statements with explicit validation error

## Configuration Model

The principal control object is `InterventionConfig`. It groups settings into six domains:

- `architecture`
- `objective`
- `data_regime`
- `retrieval`
- `context_geometry`
- `interpretability`

This makes it possible to specify sweeps and runs in a way that is legible both to humans and to downstream orchestration systems.

## Design Philosophy

### Separate measurement from pedagogy

Educational code is excellent for transparency but often weak as instrumentation. This repository does not replace pedagogical implementations. It upgrades them into a measurement surface.

### Prefer transparent proxies over ornate black boxes

Many of the estimates here are proxies rather than mathematically exact quantities. That is acceptable if the proxies are:

- explicit
- reproducible
- comparable across runs
- useful for ordering and diagnosis

### Keep the coupling loose

The repository does not require invasive edits to companion repos. It consumes them through adapters and environment-variable configuration.

### Build toward cumulative science

A useful research layer should not only produce one interesting notebook screenshot. It should produce artifacts that can accumulate over time, compare across runs, and support explanation.

## Current Capabilities

At the current version, the repository provides:

- a shared experiment contract
- text and array descriptor extraction
- multi-view compressibility estimation
- lightweight boundary and regime analysis
- a GPT-1 wind tunnel bridge
- a Sutskever-Agent narration bridge
- a sweep runner over scale, data, and task family
- persistent storage for sweep registries
- lightweight response-surface fitting
- measured tiny-run execution against the linked GPT-1 implementation
- repeated-seed measured studies across differentiated task families
- held-out predictive validation and bootstrap uncertainty intervals
- provenance capture with companion-repo commits and dirty state
- falsifiable law statements exported as machine-readable artifacts
- JSON artifact export
- a runnable demo
- basic tests

## Current Limitations

This repository is an early measurement spine, not a complete mature research platform yet.

Important limitations:

- boundary fitting is intentionally lightweight and heuristic
- the descriptor schema currently uses proxies rather than fully learned latent estimators
- notebook execution wrapping is metadata-oriented today rather than full notebook runtime orchestration
- the GPT-1 integration currently emphasizes probing and instrumentation rather than large-scale sweep automation
- plotting and interactive atlas visualization are not yet included in this standalone repo
- the current response-surface fitting is still lightweight and should be replaced by stronger uncertainty-aware models
- notebook execution is still linked to the Sutskever substrate primarily through metadata and provenance, not full execution wrapping
- the measured studies are still small-scale and should be expanded before making strong external claims

These are appropriate next targets, not hidden problems.

## Suggested Next Steps

If you are extending this repository, the highest-value next additions are:

1. full sweep orchestration over intervention grids
2. explicit failure-atlas classifiers trained on exported artifacts
3. visualization modules for onset surfaces and phase regions
4. richer notebook wrapping for direct execution of substrate experiments
5. stronger integration with the Sutskever-Agent workflow layer
6. richer compressibility estimators using actual model weights from live runs

## Development

Run the tests with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Run the demo with:

```bash
python3 -m capability_cartography.demo
```

## Suggested GitHub Description

If you are publishing this as `github.com/pageman/Capability-Cartography-Layer`, a strong short description would be:

> A standalone measurement framework for mapping capability formation, failure geometry, compressibility, retrieval dependence, and phase boundaries across the Sutskever research substrate and compact GPT-style wind tunnels.

## Citation

```bibtex
@misc{capability-cartography-layer-2026,
  author    = {Paul "The Pageman" Pajo, pageman@gmail.com},
  title     = {Capability-Cartography-Layer: a standalone empirical framework for mapping capability formation, failure geometry, compressibility, retrieval dependence, and phase boundaries},
  year      = {2026},
  url       = {https://github.com/pageman/Capability-Cartography-Layer},
  note      = {Research instrumentation layer designed to integrate pageman/sutskever-30-implementations,
               pageman/Sutskever-Agent, and pageman/gpt1-from-sutskever30 into a unified capability-mapping workflow.}
}
```

## License

This repository is released under the MIT License. See [`LICENSE`](./LICENSE).

## Final Framing

Capability Cartography Layer is meant to convert vague claims into map-like artifacts.

Instead of:

- “emergence”
- “brittleness”
- “surprisingly strong”
- “lost in the middle”
- “probably memorizing”

the repository pushes toward:

- onset surfaces
- threshold estimates
- descriptor-conditioned failure prediction
- compressibility gap analysis
- intervention-sensitive region shifts

That is the shift from anecdote to cartography, and from cartography toward an empirical science of model behavior.
