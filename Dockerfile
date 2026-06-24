# NGC PyTorch image: bundles a CUDA-matched TensorRT plus a tuned torch build.
# Pick a tag whose CUDA <= the L40S host driver (run `nvidia-smi` on the L40S).
# Override at build time, e.g. `docker build --build-arg NGC_TAG=25.01-py3 .`
ARG NGC_TAG=24.10-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

# uv for fast, reproducible dependency installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install ONLY the owned dependency layer into the NGC system Python.
# Torch and TensorRT are already installed in the base image; uv's pip resolver
# treats the transitive `torch` requirement as already satisfied and does NOT
# reinstall it, and `tensorrt` is never requested (it is absent from pyproject).
# A separate virtualenv is deliberately avoided -- it would shadow the NGC
# torch/TRT with PyPI builds. If a dependency tries to upgrade the NGC torch,
# pin it via a uv constraint rather than letting it win.
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml \
 && uv pip install --system pytest

# Source is bind-mounted over /workspace at runtime (see docker-compose.yml);
# this copy is only a fallback for standalone `docker run`.
COPY . .

CMD ["bash"]
