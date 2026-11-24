FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
# no cache: otherwise uv's download cache doubles the torch/CUDA layer
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1

# deps first: this layer stays cached across code changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --group saliency --group vlm --no-install-project

COPY src ./src
COPY resources/best_inceptionv3_model_multiclass.h5 ./resources/
RUN uv sync --frozen --no-default-groups --group saliency --group vlm

# Qwen weights are not baked in: mount a HF cache at /root/.cache/huggingface (see README)
ENV PATH=/app/.venv/bin:$PATH
WORKDIR /work
ENTRYPOINT ["vlm-explain"]
