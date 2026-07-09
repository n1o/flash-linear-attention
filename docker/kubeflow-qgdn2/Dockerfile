ARG BASE_IMAGE=nvidia/cuda:13.1.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=100
ARG CODEX_NPM_PACKAGE=@openai/codex@0.143.0
ARG OPENCODE_NPM_PACKAGE=opencode-ai@1.17.16

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/${NB_USER}/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    JUPYTER_ROOT_DIR=/home/${NB_USER} \
    FLA_IMAGE_SOURCE=/opt/flash-linear-attention \
    FLA_WORKDIR=/home/${NB_USER}/flash-linear-attention \
    VIRTUAL_ENV=/opt/python \
    CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;10.3" \
    PATH=/opt/python/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        htop \
        less \
        libgl1 \
        libglib2.0-0 \
        libnuma1 \
        ninja-build \
        nvtop \
        openssh-client \
        pciutils \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        tini \
        tmux \
        unzip \
        vim \
        wget && \
    python3 -m venv /opt/python && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g "${CODEX_NPM_PACKAGE}" "${OPENCODE_NPM_PACKAGE}" && \
    npm cache clean --force && \
    codex --version && \
    opencode --version

RUN python -m pip install --upgrade pip setuptools wheel packaging && \
    python -m pip install \
        --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu130 && \
    python -m pip install \
        accelerate \
        cuda-python \
        'datasets>=3.3.0' \
        einops \
        hf_transfer \
        huggingface_hub \
        ipykernel \
        ipywidgets \
        jupyterlab \
        jupyterlab-git \
        jupytext \
        matplotlib \
        numpy \
        nvidia-ml-py \
        pandas \
        plotly \
        pybind11 \
        rich \
        seaborn \
        tensorboard \
        tilelang \
        tqdm \
        triton \
        wandb

WORKDIR /opt/flash-linear-attention
COPY . /opt/flash-linear-attention

RUN python -m pip install -e '.[cuda,test,benchmark,tilelang]' && \
    python -m ipykernel install --sys-prefix --name fla-qgdn2 --display-name "Python (FLA QGDN2)" && \
    python - <<'PY'
import torch
import triton
import tilelang

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("triton", triton.__version__)
print("tilelang", getattr(tilelang, "__version__", "unknown"))
PY

RUN python -m pip check

COPY docker/kubeflow-qgdn2/certs/eset_root_ca.crt /usr/local/share/ca-certificates/eset_root_ca.crt
RUN update-ca-certificates

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt

RUN set -eux; \
    if ! getent group "${NB_GID}" >/dev/null; then groupadd --gid "${NB_GID}" "${NB_USER}"; fi; \
    if getent passwd "${NB_UID}" >/dev/null; then \
        EXISTING_USER="$(getent passwd "${NB_UID}" | cut -d: -f1)"; \
        if [ "${EXISTING_USER}" != "${NB_USER}" ]; then \
            usermod --login "${NB_USER}" --home "/home/${NB_USER}" --move-home "${EXISTING_USER}"; \
        fi; \
        usermod --gid "${NB_GID}" "${NB_USER}"; \
    else \
        useradd --uid "${NB_UID}" --gid "${NB_GID}" --create-home --shell /bin/bash "${NB_USER}"; \
    fi; \
    mkdir -p /home/${NB_USER}/.cache /workspace && \
    chown -R "${NB_UID}:${NB_GID}" /home/${NB_USER} /workspace /opt/flash-linear-attention

COPY docker/kubeflow-qgdn2/opencode-config/ /opt/opencode-config/
RUN cd /opt/opencode-config && \
    npm ci --omit=dev && \
    mkdir -p /home/${NB_USER}/.config && \
    cp -a /opt/opencode-config /home/${NB_USER}/.config/opencode && \
    chown -R "${NB_UID}:${NB_GID}" /opt/opencode-config /home/${NB_USER}/.config/opencode

COPY docker/kubeflow-qgdn2/start-notebook.sh /usr/local/bin/start-notebook
RUN chmod +x /usr/local/bin/start-notebook

USER ${NB_UID}
WORKDIR /home/${NB_USER}

ENV PATH=/home/${NB_USER}/.local/bin:/usr/local/bin:${PATH} \
    PYTHONPATH=/home/${NB_USER}/flash-linear-attention:/opt/flash-linear-attention

EXPOSE 8888

ENTRYPOINT ["tini", "--"]
CMD ["/usr/local/bin/start-notebook"]
