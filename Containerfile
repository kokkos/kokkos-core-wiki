# Build environment for the Kokkos documentation site.
# Build locally with:
#   docker build -t kokkos-docs -f Containerfile .
# or:
#   podman build -t kokkos-docs -f Containerfile .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        graphviz \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY build_requirements.txt /tmp/build_requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/build_requirements.txt

WORKDIR /workspace/docs
CMD ["make", "html"]
