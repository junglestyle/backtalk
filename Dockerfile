# CUDA 13 with cuDNN 9: the libraries onnxruntime-gpu's build loads (docs/standards.md says why this isn't a
# python:*-slim base).
FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04@sha256:14f6d08d1cd4a96effbfe3101d0b56326f552c199d05e4979ee0bd616df5811b
COPY --from=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 /uv /uvx /bin/
# The speak worker's runtime: Node's binary alone, which needs nothing the base doesn't have.
COPY --from=node:26.13.1-trixie-slim@sha256:e6bedef217677b276c591795e169840edd9096dbf13f29df8ef8d1856b8820bc /usr/local/bin/node /usr/local/bin/node
# uv installs the Python .python-version names, inside the image.
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY .python-version pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
# The venv's Python under a name of its own, which GPU tools such as nvidia-smi list the server as, so a GPU
# worker sharing the host can ignore it by name.
RUN ln -s python /app/.venv/bin/backtalk-server
# A fixed uid, so the volume mounted at /var/cache/backtalk can be owned by it.
RUN groupadd --system --gid 8765 backtalk \
    && useradd --system --uid 8765 --gid backtalk --no-create-home backtalk \
    && mkdir -p /var/cache/backtalk \
    && chown backtalk:backtalk /var/cache/backtalk
# The models, and what huggingface_hub keeps while it downloads them, go to the volume, not the image.
ENV BACKTALK_CACHE=/var/cache/backtalk HF_HOME=/var/cache/backtalk/huggingface
USER backtalk
EXPOSE 8765
# Python itself, not `uv run`, so the server is PID 1 and gets the SIGTERM a stop sends.
ENTRYPOINT ["/app/.venv/bin/backtalk-server", "-m", "backtalk"]
