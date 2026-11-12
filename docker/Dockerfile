# Reproducible CPU environment for dimensionless-mpcrl-racing.
#
# The image contains the *pinned* acados and leap-c sources from the git
# submodules, built from source, plus the fully pinned Python environment from
# requirements.txt.  Nothing is installed from the host machine.

FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libgomp1 \
        ninja-build \
        wget \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /workspace

# Install Python dependencies first so source edits do not invalidate this layer.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Build the pinned acados version from the leap-c submodule.
COPY external/leap-c ./external/leap-c
RUN set -eux; \
    cd external/leap-c/external/acados; \
    cmake -S . -B build \
        -DACADOS_WITH_OPENMP=ON \
        -DACADOS_NUM_THREADS=1 \
        -G Ninja; \
    cmake --build build --target install -- -j"$(nproc)"; \
    bash .github/linux/install_tera.sh

ENV ACADOS_SOURCE_DIR=/workspace/external/leap-c/external/acados \
    ACADOS_INSTALL_DIR=/workspace/external/leap-c/external/acados \
    LD_LIBRARY_PATH=/workspace/external/leap-c/external/acados/lib \
    PATH=/workspace/external/leap-c/external/acados/bin:$PATH \
    PYTHONPATH=/workspace

# Install the local Python interfaces from the pinned sources.  --no-deps keeps
# requirements.txt authoritative.
RUN set -eux; \
    python -m pip install --no-deps -e external/leap-c/external/acados/interfaces/acados_template; \
    python -m pip install --no-deps -e external/leap-c

# Copy the project code last; this is the layer that changes most often.
COPY race_car ./race_car
COPY sensitivity_analysis ./sensitivity_analysis
COPY tests ./tests
COPY docker/scripts ./scripts
COPY pytest.ini README.md SETUP.md ./

RUN chmod +x /workspace/scripts/*.sh

# Provenance supplied by `make`; kept in the final layer so changing a commit
# does not invalidate the expensive dependency and acados build layers.
ARG GIT_COMMIT=unknown
ARG GIT_DIRTY=unknown
ENV GIT_COMMIT=${GIT_COMMIT} \
    GIT_DIRTY=${GIT_DIRTY}

# Generated acados code, experiment output, and reproducibility results are
# written below /workspace/output. docker-compose mounts ./output from the host
# over this path.
RUN mkdir -p /workspace/output

ENTRYPOINT ["/workspace/scripts/docker_entrypoint.sh"]
CMD ["analysis"]
