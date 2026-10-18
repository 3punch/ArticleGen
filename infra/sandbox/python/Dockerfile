FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Keep the runner offline at execution time; dependencies are baked into this image.
RUN python -m pip install --no-cache-dir pytest==8.3.5 \
    && mkdir -p /workspace \
    && chmod 1777 /workspace /tmp

WORKDIR /workspace
USER 65532:65532
