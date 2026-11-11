#!/usr/bin/env bash
# One-command reproduction for Linux and WSL.
#
# If Docker is already installed and running this script is non-interactive.
# If it is missing, it attempts to install/start it and may ask for sudo once.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"

log() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

MODE="${1:-full}"
case "$MODE" in
    full|analysis|test|rl) ;;
    *)
        die "Usage: $0 [full|analysis|test|rl]"
        ;;
esac
if [[ $# -gt 1 ]]; then
    die "Usage: $0 [full|analysis|test|rl]"
fi

[[ "$(uname -s)" == "Linux" ]] || \
    die "This script is for Linux/WSL. On Windows, run: powershell -ExecutionPolicy Bypass -File .\\docker\\reproduce.ps1"

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64) ;;
    *)
        die "Unsupported architecture: $ARCH. The acados Tera renderer requires x86_64 Linux."
        ;;
esac

IS_WSL=false
if [[ -n "${WSL_DISTRO_NAME:-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=true
fi

command -v git >/dev/null 2>&1 || die "git is required to initialize the submodules."

SUDO=()
if [[ "${EUID}" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
fi

can_root() {
    [[ "${EUID}" -eq 0 || ${#SUDO[@]} -gt 0 ]]
}

as_root() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    elif [[ ${#SUDO[@]} -gt 0 ]]; then
        "${SUDO[@]}" "$@"
    else
        die "This step needs root privileges, but sudo is not available."
    fi
}

DOCKER=(docker)
docker_ok() { "${DOCKER[@]}" info >/dev/null 2>&1; }
compose_ok() { "${DOCKER[@]}" compose version >/dev/null 2>&1; }

wait_for_docker() {
    local timeout="${1:-120}" elapsed=0
    while (( elapsed < timeout )); do
        if docker_ok; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

try_start_docker_desktop_from_wsl() {
    [[ "$IS_WSL" == true ]] || return 1
    local desktop="/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe"
    [[ -e "$desktop" ]] || return 1

    log "Starting Docker Desktop..."
    if command -v cmd.exe >/dev/null 2>&1; then
        ( cd /mnt/c 2>/dev/null && cmd.exe /c start "" "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe" >/dev/null 2>&1 ) || true
    elif command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command "Start-Process 'C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe'" >/dev/null 2>&1 || true
    else
        return 1
    fi
    return 0
}

try_start_docker() {
    if [[ "$IS_WSL" == true ]]; then
        try_start_docker_desktop_from_wsl
        return
    fi

    can_root || return 1
    if command -v systemctl >/dev/null 2>&1; then
        as_root systemctl start docker >/dev/null 2>&1 || true
    elif command -v service >/dev/null 2>&1; then
        as_root service docker start >/dev/null 2>&1 || true
    fi
}

install_docker_engine() {
    log "Docker is not installed. Installing Docker Engine and the Compose plugin..."
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        die "curl or wget is required to install Docker automatically."
    fi

    if command -v curl >/dev/null 2>&1; then
        if ! curl -fsSL https://get.docker.com | as_root sh; then
            die "Automatic Docker installation failed. Install it manually: https://docs.docker.com/engine/install/"
        fi
    else
        if ! wget -qO- https://get.docker.com | as_root sh; then
            die "Automatic Docker installation failed. Install it manually: https://docs.docker.com/engine/install/"
        fi
    fi

    if command -v systemctl >/dev/null 2>&1; then
        as_root systemctl enable --now docker >/dev/null 2>&1 || true
    elif command -v service >/dev/null 2>&1; then
        as_root service docker start >/dev/null 2>&1 || true
    fi

    if [[ "${EUID}" -ne 0 && ${#SUDO[@]} -gt 0 ]]; then
        as_root usermod -aG docker "$(id -un)" >/dev/null 2>&1 || true
        # The current session does not have the new group yet. Use sudo for
        # this run; the next login will be able to use docker directly.
        DOCKER=("${SUDO[@]}" docker)
    fi
}

ensure_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        if [[ "$IS_WSL" == true ]]; then
            try_start_docker_desktop_from_wsl || die \
                "Docker is not available in this WSL distribution. Install Docker Desktop on Windows (with WSL integration), or run .\\docker\\reproduce.ps1 from Windows, then rerun."
            hash -r 2>/dev/null || true
            command -v docker >/dev/null 2>&1 || die \
                "Docker Desktop was found, but the docker CLI is not available in WSL. Enable this distribution under Docker Desktop > Settings > Resources > WSL Integration, then rerun."
        else
            install_docker_engine
        fi
    fi

    if docker_ok; then
        return 0
    fi

    # Docker may be installed but only accessible through sudo.
    if [[ "${EUID}" -ne 0 && ${#SUDO[@]} -gt 0 ]] && "${SUDO[@]}" docker info >/dev/null 2>&1; then
        DOCKER=("${SUDO[@]}" docker)
        return 0
    fi

    log "Docker is not running; trying to start it..."
    if try_start_docker; then
        if wait_for_docker 180; then
            return 0
        fi
    fi

    if [[ "${EUID}" -ne 0 && ${#SUDO[@]} -gt 0 ]] && "${SUDO[@]}" docker info >/dev/null 2>&1; then
        DOCKER=("${SUDO[@]}" docker)
        return 0
    fi

    die "Docker is installed but not reachable. Start Docker and rerun this script."
}

ensure_linux_containers() {
    local os_type
    os_type="$("${DOCKER[@]}" info --format '{{.OSType}}' 2>/dev/null || true)"
    if [[ "$os_type" != "linux" ]]; then
        die "Docker is running Windows containers. Switch Docker Desktop to Linux containers and rerun."
    fi
}

ensure_docker
compose_ok || die "Docker Compose v2 is required. Install the Docker Compose plugin and rerun."
ensure_linux_containers

log "Initializing submodules..."
git -C "$ROOT" submodule update --init --recursive

GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]]; then
    GIT_DIRTY=dirty
else
    GIT_DIRTY=clean
fi
export GIT_COMMIT GIT_DIRTY
# The RL scripts parse this as %Y%m%d_%H%M%S, so keep underscores.
export REPRO_RUN_ID="${REPRO_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "$ROOT/results"
{
    echo "# Host repository state captured by docker/reproduce.sh"
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "## commit"
    echo "$GIT_COMMIT"
    echo
    echo "## status"
    git -C "$ROOT" status --short 2>/dev/null || true
    echo
    echo "## submodules"
    git -C "$ROOT" submodule status --recursive 2>/dev/null || true
} > "$ROOT/results/host-git-info.txt"

ENV_ARGS=()
for var in ANALYSIS_STEPS REPRO_RUN_ID RL_SEEDS RL_JOBS RL_MAIN_STEPS RL_TRANSFER_DEFAULT_STEPS RL_TRANSFER_LARGE_STEPS; do
    if [[ -n "${!var:-}" ]]; then
        ENV_ARGS+=(-e "$var=${!var}")
    fi
done

RUN_ARGS=("$MODE")
if [[ "$MODE" == "rl" ]]; then
    RUN_ARGS=(rl "${RL_SUITE:-main}")
fi

COMPOSE=("${DOCKER[@]}" compose -f "$SCRIPT_DIR/compose.yaml")

log "Building the reproduction image..."
"${COMPOSE[@]}" build \
    --build-arg "GIT_COMMIT=$GIT_COMMIT" \
    --build-arg "GIT_DIRTY=$GIT_DIRTY"

log "Running reproduction mode: $MODE (run id: $REPRO_RUN_ID)"
set +e
"${COMPOSE[@]}" run --rm "${ENV_ARGS[@]}" reproduce "${RUN_ARGS[@]}"
status=$?
set -e

if [[ "${DOCKER[0]:-}" == "sudo" ]]; then
    as_root chown -R "$(id -u):$(id -g)" "$ROOT/results" "$ROOT/output" 2>/dev/null || true
fi

if [[ $status -eq 0 ]]; then
    log "Reproduction complete."
    log "Analysis results: $ROOT/results"
    log "RL output:        $ROOT/output"
else
    warn "Reproduction failed with exit code $status."
fi
exit "$status"
