# AArch64 build/test environment for the dbt translator.
#
# The emitted ARM64 code cannot run on an x86_64 host, so this image provides a
# real aarch64 userland (via QEMU/binfmt under Docker Desktop) where the full
# test suite -- including the host-gated execution tests -- can be run.
#
#   docker build --platform linux/arm64 -t dbt-arm64 -f docker/arm64.Dockerfile .
#   docker run --rm --platform linux/arm64 -v "$PWD":/src dbt-arm64 \
#       bash -lc "cmake -S /src -B /tmp/b -G Ninja && cmake --build /tmp/b && ctest --test-dir /tmp/b --output-on-failure"
#
# The build directory is kept outside the bind mount so the emulated build does
# not collide with the host's build/ tree.

FROM --platform=linux/arm64 ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

CMD ["bash"]
