# Reproducibility with Docker

This directory contains everything needed for the one-command, CPU-only
reproduction of the experiments:

- `Dockerfile` — Python 3.12, the pinned packages from `requirements.txt`, and
  acados/leap-c built from the commits pinned in `external/leap-c`.
- `compose.yaml` — the `reproduce` service. It mounts `../output` and
  `../results` on the host and forwards the runtime configuration.
- `Makefile` — the entry points documented below.
- `scripts/` — the container entry point and the analysis/RL runners.

The repository root only keeps `.dockerignore`, because Docker requires that
file next to the build context (the repository root).

## Quick start

From a fresh clone with Docker and Docker Compose installed, run:

```bash
make -C docker
```

This initializes the pinned submodules, builds the CPU-only image, runs the
test suite, and runs all sensitivity analyses. No host Python, acados, or CMake
installation is needed. On an 8-core CPU the default target takes about 10
minutes after the image is built (the first build takes a few minutes more).
`cd docker && make` works too.

Results are written to `results/<UTC timestamp>/` (with `results/latest`
pointing at the most recent run). Each analysis has its own isolated working
directory, log, and `output/` folder, so cached `.npy`/`.npz` files from earlier
runs are never reused. `results/host-git-info.txt` records the host commit and
submodule revisions, and `provenance.txt` records the container environment.

## Available modes

```bash
make -C docker test                 # pytest only
make -C docker                      # tests + all sensitivity analyses
make -C docker rl                   # main SAC-ZOP suite (long)
make -C docker rl RL_SUITE=transfer # default -> large transfer experiments (long)
make -C docker rl RL_SUITE=baseline # tracking-controller baselines
make -C docker rl RL_SUITE=all      # baseline + main + transfer (longest)
make -C docker full                 # analyses + every RL experiment
make -C docker shell                # interactive shell in the container
```

The default target intentionally excludes the RL training suites: the main
suite is 4 car/formulation combinations × 5 seeds, and each run takes roughly
1–3 hours on a CPU. Use `RL_SEEDS`, `RL_JOBS`, `RL_MAIN_STEPS`,
`RL_TRANSFER_DEFAULT_STEPS`, and `RL_TRANSFER_LARGE_STEPS` to control them, for
example:

```bash
make -C docker rl RL_SUITE=main RL_SEEDS="0" RL_MAIN_STEPS=2000 RL_JOBS=1
```

To run only part of the sensitivity pipeline:

```bash
docker compose -f docker/compose.yaml run --rm \
  -e ANALYSIS_STEPS=dynamic-analytical,compare-gradients \
  reproduce analysis
```

RL runs keep the original layout under `output/` and write one log per job to
`output/logs/<timestamp>/`. The container image is intended for x86_64 Linux
(the Tera renderer used by acados is distributed as a Linux amd64 binary).

## Layout

```text
docker/
├── Dockerfile
├── Makefile
├── README.md
├── compose.yaml
└── scripts/
    ├── docker_entrypoint.sh
    ├── run_analysis.sh
    └── run_rl.sh
```
