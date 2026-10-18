# Runtime image for the Naturalness package and the paper's reproduction scripts.
#
#   docker build -t naturalness:cpu .                                  # CPU only
#   docker build -t naturalness:cu124 --build-arg TORCH_VARIANT=cu124 .  # CUDA 12.4
#
# The Python stack is that of the paper's runs: Python 3.11, torch 2.5.1,
# requirements.txt and BLearner at commit 734fd76. Data and results are not
# baked in; README.md shows how to mount them.
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="naturalness" \
      org.opencontainers.image.description="Naturalness grounding and the reproduction scripts of its paper"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_RETRIES=10 \
    PIP_RESUME_RETRIES=20 \
    PIP_DEFAULT_TIMEOUT=120 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    HF_HOME=/tmp/huggingface \
    B_LEARNER_REPOSITORY=/opt/BLearner \
    HOME=/tmp \
    PYTORCH_NVML_BASED_CUDA_CHECK=1 \
    PYTHONPATH=/workspace/naturalness

# git fetches BLearner, curl the Twins data; procps lets the paper scripts stop
# their jobs on Ctrl-C; libgomp1 is the OpenMP runtime of torch and xgboost.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl git libgomp1 procps \
    && rm -rf /var/lib/apt/lists/*

# safe.directory lets any --user read the commit of these root-owned checkouts.
ARG B_LEARNER_COMMIT=734fd7639da54554e41d944bfc2187a24e75cf81
RUN git clone https://github.com/CausalML/BLearner.git /opt/BLearner \
    && git -C /opt/BLearner checkout "${B_LEARNER_COMMIT}" \
    && test "$(git -C /opt/BLearner rev-parse HEAD)" = "${B_LEARNER_COMMIT}" \
    && git config --system --add safe.directory /opt/BLearner \
    && git config --system --add safe.directory /workspace/naturalness

WORKDIR /workspace/naturalness

COPY --chmod=0755 docker/retry.sh /usr/local/bin/retry
COPY requirements.txt requirements-dev.txt ./

ARG TORCH_VARIANT=cpu
ARG PIP_INSTALL_ATTEMPTS=5
RUN case "${TORCH_VARIANT}" in \
        cpu)   torch_version="2.5.1+cpu"; torch_index="https://download.pytorch.org/whl/cpu" ;; \
        cu124) torch_version="2.5.1";     torch_index="https://download.pytorch.org/whl/cu124" ;; \
        *) echo "Unsupported TORCH_VARIANT: ${TORCH_VARIANT}" >&2; exit 2 ;; \
    esac \
    && python -m pip install --upgrade pip \
    && retry "${PIP_INSTALL_ATTEMPTS}" \
        python -m pip install --index-url "${torch_index}" "torch==${torch_version}"
RUN retry "${PIP_INSTALL_ATTEMPTS}" \
    python -m pip install -r requirements-dev.txt

COPY . .

RUN mkdir -p results /tmp/matplotlib /tmp/numba-cache /tmp/huggingface \
    && chmod 0777 results \
    && chmod 1777 /tmp/matplotlib /tmp/numba-cache /tmp/huggingface \
    && python docker/image_smoke.py --torch-variant "${TORCH_VARIANT}" \
    && chmod -R a+rwX /tmp/numba-cache /tmp/huggingface

CMD ["python", "examples/toy_naturalness.py"]
