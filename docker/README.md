# Reproducibility with Docker

This directory contains the CPU-only reproduction environment for the
experiments. The repository root provides the two one-command entry points:

- `reproduce.sh` — Linux and WSL.
- `reproduce.ps1` — Windows PowerShell.

Both build the pinned environment and run the complete pipeline: tests, all
sensitivity analyses, and every RL experiment.

## One-command reproduction

Linux or WSL, from the repository root:

```bash
./reproduce.sh
```

Windows PowerShell, from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproduce.ps1
```

If Docker is already installed and running, these commands are
non-interactive. The scripts initialize the pinned submodules, build the image
(using the Docker layer cache on later runs), and then run the container. No
host Python, acados, or CMake installation is needed.

The first image build takes a few minutes. The analyses take about 10 minutes
after that. The RL suites are the long part: the main suite is 4
car/formulation combinations × 5 seeds, and each run takes roughly 1–3 hours on
a CPU. Use the configuration variables below to shorten a trial run.

## Docker setup behavior

The scripts try to get from a fresh machine to a running container with as
little interaction as possible.

**Linux.** If Docker is missing, the script installs Docker Engine and the
Compose plugin using Docker's official convenience script. This may ask for
your `sudo` password once. The user is added to the `docker` group, and the
current run uses `sudo`; subsequent runs should not need it.

**Windows.** The script checks WSL2 and installs Docker Desktop with the
official installer, accepting the license and selecting the WSL2 backend. It
then starts Docker Desktop, switches it to Linux containers if necessary, and
waits for the daemon. This may require one UAC/admin approval.

**WSL.** Docker Desktop is expected. The script can start Docker Desktop from
WSL. If the `docker` CLI is not visible, enable this distribution under
*Docker Desktop → Settings → Resources → WSL Integration* once, or run
`reproduce.ps1` from Windows instead.

### When interaction is unavoidable

A completely blank machine can still require one human step:

- Linux: entering the `sudo` password if Docker must be installed or started.
- Windows: approving the Docker Desktop/WSL2 installation via UAC.
- Windows: rebooting if WSL2 was just enabled, and enabling hardware
  virtualization (VT-x/AMD-V) in BIOS/UEFI if it is disabled.
- Any platform: a manually installed Docker Desktop that has never completed
  its first-run onboarding may need to be opened once.

After those one-time steps, the same command runs to completion without
further prompts.

## Outputs

Analysis results are written to `results/<run-id>/`, with `results/latest`
pointing at the most recent run. On filesystems without symbolic-link support,
the run ID is written to `results/latest.txt` instead.

Each analysis has its own isolated working directory, log, and `output/`
folder, so cached `.npy`/`.npz` files from earlier runs are never reused.
`results/host-git-info.txt` records the host commit and submodule revisions,
and `provenance.txt` records the container environment.

RL runs keep the original layout under `output/` and write one log per job to
`output/logs/<run-id>/`. The bootstrap scripts use one run ID for both the
analysis and RL outputs.

## Modes and configuration

The scripts default to `full`. A shorter mode can be passed as the first
argument on Linux/WSL, or with `-Mode` on Windows:

```bash
./reproduce.sh analysis    # tests + all sensitivity analyses
./reproduce.sh test        # pytest only
./reproduce.sh rl          # main SAC-ZOP suite only
```

```powershell
.\reproduce.ps1 -Mode analysis
.\reproduce.ps1 -Mode test
.\reproduce.ps1 -Mode rl
```

Runtime configuration is passed through the environment, for example:

```bash
RL_SEEDS="0 1" RL_JOBS=2 ./reproduce.sh rl
RL_MAIN_STEPS=2000 RL_SEEDS="0" ./reproduce.sh rl
ANALYSIS_STEPS=dynamic-analytical,compare-gradients ./reproduce.sh analysis
RL_SUITE=transfer ./reproduce.sh rl
```

On Windows, set the same variables in PowerShell before running the script:

```powershell
$env:RL_SEEDS = "0 1"
$env:RL_JOBS = 2
.\reproduce.ps1 -Mode rl
```

The RL jobs are memory-hungry. The default concurrency inside the container is
its CPU count; on Docker Desktop that is often 1. Raise `RL_JOBS` only if the
container has enough memory.

## Make targets

The scripts do not require `make`. If you already have Docker and prefer the
original make entry points, they remain available:

```bash
make -C docker test                 # pytest only
make -C docker                      # tests + all sensitivity analyses
make -C docker rl                   # main SAC-ZOP suite (long)
make -C docker rl RL_SUITE=transfer # default -> large transfer experiments
make -C docker rl RL_SUITE=baseline # tracking-controller baselines
make -C docker rl RL_SUITE=all      # baseline + main + transfer
make -C docker full                 # analyses + every RL experiment
make -C docker shell                # interactive shell in the container
```

`make -C docker full` and the default script command run the same pipeline.

## Layout

```text
reproduce.sh          # Linux/WSL one-command entry point
reproduce.ps1         # Windows one-command entry point
.gitattributes        # keeps scripts and container files at LF
.dockerignore         # must stay at the build-context root
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

`.gitattributes` keeps the shell scripts, Makefile, and Docker files at LF line
endings, so a Windows checkout with `core.autocrlf=true` cannot break the
container entrypoint.

## Platform notes

The image is intended for x86_64 Linux. The Tera renderer used by acados is
distributed as a Linux amd64 binary, so ARM64 hosts are not supported. On
Windows, Docker Desktop must be in Linux-containers mode with the WSL2 backend.

For the best Windows performance, clone the repository inside the WSL2
filesystem and run `./reproduce.sh` from WSL. The native PowerShell script also
works, but Windows bind mounts are slower than WSL2-native files.
