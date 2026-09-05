# Post-training quantisation of world models: LeWM vs DINOv3-WM

A research repository comparing the **planning-cycle latency** and **Push-T task performance** of two
self-supervised world models under reduced-precision inference, on a single NVIDIA L40S.

- **LeWM** — the reference implementation from [`stable-worldmodel`](https://github.com/galilai-group/stable-worldmodel),
  used unmodified: encoder co-trained with the predictor, a single CLS-token latent.
- **DINOv3-WM** — this project's variant of DINO-WM: the platform's `prejepa` predictor with the
  reference DINOv2 backbone swapped for a frozen **DINOv3**, predicting over the full 196-patch grid.

Both are exported PyTorch → ONNX → TensorRT and evaluated closed-loop against a CEM planner at
**FP32, FP16, FP8 (E4M3) and INT8**, with the two 8-bit precisions calibrated by both **`max`** and
**`entropy`**. Every speed number is reported with the success rate measured on the same solves.

Training, the Push-T environment, the CEM solver and the evaluation loop come from
`stable-worldmodel` and are used as they are. The code in this repository starts at a trained
checkpoint: export, calibration, quantisation, benchmarking and the statistical analysis.

## Repository map

| Path | What it is |
| --- | --- |
| `SPEC.md` | The contract: requirements, invariants, scope, ownership boundaries |
| `docs/architecture.md` | Non-obvious design rationale — silent-failure traps, platform quirks, library deviations |
| `docs/platform_api.md` | The `stable-worldmodel` API as read from the pinned installed source |
| `src/` | The owned code: adapter, export, calibration, engine runtime, shims, benchmark, statistics, report |
| `scripts/` | Vendored platform entrypoints (`train/`, `plan/`), byte-unmodified apart from importing the encode-path override |
| `conf/experiment/` | Hydra overlays layered on the vendored configs |
| `tests/` | The owned code's contract tests (CPU-only; the GPU legs are exercised on the pod) |
| `reports/figs/` | The one committed display figure, re-copied from a render and never hand-edited |

## Setup

```bash
cp .env.example .env    # then fill in STABLEWM_HOME, WANDB_API_KEY, HF_TOKEN
```

`.env` is the single source for runtime configuration — never `export` these into a shell. It is
gitignored and read once on `import src`. `STABLEWM_HOME` must point at persistent storage (on
RunPod, the mounted network volume): datasets, checkpoints, TensorRT engines and every report land
under it, and it is **required** — leaving it unset raises rather than silently writing to the
ephemeral container filesystem.

```bash
./setup.sh
```

Installs uv, syncs the locked dependencies (torch cu124), then installs TensorRT, the NVIDIA
TensorRT Model Optimizer and a CUDA-12 `onnxruntime-gpu` **outside** the lock — see
`docs/architecture.md` for why that split is load-bearing. It also fetches the Push-T expert
dataset. A later bare `uv sync` prunes the out-of-lock installs; re-run `setup.sh` to restore them.

## Reproducing the study

The whole sequence, as one orchestrated run:

```bash
uv run python -m src.pipeline
```

It executes the stages below in order, appending to `pipeline_manifest.json`. Add
`stages=<a,b,c>` to resume a subset (always run in canonical order), `tracks=lewm` for one model,
`diagnostics=true` to include the gates and smoke checks, or `dry_run=true` to print the plan.

| Stage | Command it runs | Produces |
| --- | --- | --- |
| `archive` | in-process | Snapshots the previous run's artifacts before they are superseded |
| `export` | `src.export model= precision= [calibration_method=]` | TensorRT engines under `$STABLEWM_HOME/engines/` |
| `sr_eval` | `src.sr_eval precision= calibration_method=` | `sr.json` — success rate + the raw per-cycle latency vectors |
| `isolation` | `src.sr_eval encoder_precision= predictor_precision=` | Component-isolation SR, under composite `enc-<A>+pred-<B>` keys |
| `benchmark` | `src.study track= precision= calibration_method=` | `results.<track>.json` + `latencies.<track>.json` (raw engine-step samples) |
| `stats` | `src.stats from= out=` | `stats.json` — the confidence intervals and independence tests |
| `report` | `src.report from= calibration_method=` | The reported tables and figure |
| `gpu_telemetry` | `src.gpu_telemetry from=` | Throttle diagnostics from the logged `nvidia-smi dmon` telemetry |
| `figs` | in-process | Refreshes the committed `reports/figs/` copy |

Diagnostic stages (`pytest`, `verify_encode`, `smoke`, `fidelity`, `sr_shim`, `precision_match`,
`probe_ranges`) are off by default. `fidelity`, `sr_shim` and `precision_match` are **owner
sign-off surfaces**: they print a drift table rather than passing or failing, because the judgement
they support is not a tolerance (`SPEC.md` §Implementation Boundaries).

Only the export, evaluation and benchmark stages need the L40S. Everything downstream reads the
stored samples, so a render or a re-analysis runs anywhere:

```bash
uv run python -m src.report
```

It resolves `$STABLEWM_HOME/reports/phase5` itself, from `.env`. Do not write that path into the
command — your shell expands `$STABLEWM_HOME` before Python reads `.env`, so it would collapse to
`/reports/phase5`. Pass `from=` only to read a *different* directory, as a literal path:

```bash
uv run python -m src.report from=/mnt/archive/2026-08-07
```

## Artifacts

Everything durable lives under `$STABLEWM_HOME/reports/phase5/`, never in git. The **canonical**
artifacts are the raw ones — `results.<track>.json`, `latencies.<track>.json` and `sr.json`, which
carry the samples every statistic is re-derived from. `stats.json` and the rendered outputs are
regenerable views of them, and a render never rewrites its own inputs.

| Artifact | Contents |
| --- | --- |
| `speed_table.<method>.txt` | Per-cycle p50/p95, sample size, and success rate, each absolute value with its 95% interval |
| `latency_means_table.txt` | Mean encode-step and predictor-step latency per engine call, and the per-cycle component total, cycle and residual overhead, with bootstrap intervals |
| `isolation_table.<method>.txt` | Success rate per component-isolation configuration |
| `speed_vs_sr.png` | Success rate against median per-cycle latency, faceted per model |
| `gpu_logs/` | Per-run `nvidia-smi dmon` telemetry and its throttle diagnostics |

Tables carrying an 8-bit row are **method-scoped by filename** and name their calibration method in
the body: an INT8 or FP8 engine is a per-method build, so one method's numbers are never printed
under the other's label. `latency_means_table.txt` spans both, because its configuration column
names the method on every row.

## Running the tests

```bash
uv run python -m pytest -q
```

CPU-only and hardware-free — they pin the contracts of the owned code (call-count weighting,
equal-n truncation, method scoping, artefact immutability, the statistical constructions) against
synthetic fixtures. The engine legs are exercised on the pod through the diagnostic stages.

## Licence

MIT — see `LICENSE`.
