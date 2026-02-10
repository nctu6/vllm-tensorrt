#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: install_tensorrt_deps.sh [--no-mpi] [--with-mpi] [--no-ucx] [--with-ucx] [--with-lfs] [--with-fmha-cubin] [--no-fmha-cubin]

Installs TensorRT runtime deps. Optional flags add packages required for
TensorRT-LLM multi-device (MPI), UCX/NIXL features, Git LFS assets, or
generate FMHA v2 cubins for SM80 (enabled by default). MPI and UCX are
installed by default to match TensorRT-LLM defaults.

Environment:
  TENSORRT_TARBALL_URL     Optional URL to TensorRT tarball (TensorRT-*.tar.gz).
  TENSORRT_TARBALL_SHA256  Optional SHA256 checksum for the tarball.
  TENSORRT_INSTALL_DEBS    Install NVIDIA TensorRT debs if libs are missing (default: 1).
  TENSORRT_DEB_VERSION     TensorRT deb version (default: 10.9.0.34-1+cuda12.8).
EOF
}

WITH_MPI=1
WITH_UCX=1
WITH_LFS=0
WITH_FMHA_CUBIN=1
TENSORRT_PIP_SPEC="${TENSORRT_PIP_SPEC:-10.13.*}"
USE_LOCAL_TENSORRT="${USE_LOCAL_TENSORRT:-1}"
ALLOW_PIP_TENSORRT="${ALLOW_PIP_TENSORRT:-0}"
TENSORRT_INSTALL_DEBS="${TENSORRT_INSTALL_DEBS:-1}"
TENSORRT_DEB_VERSION="${TENSORRT_DEB_VERSION:-10.9.0.34-1+cuda12.8}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-mpi)
      WITH_MPI=1
      ;;
    --no-mpi)
      WITH_MPI=0
      ;;
    --with-ucx)
      WITH_UCX=1
      ;;
    --no-ucx)
      WITH_UCX=0
      ;;
    --with-lfs)
      WITH_LFS=1
      ;;
    --with-fmha-cubin)
      WITH_FMHA_CUBIN=1
      ;;
    --no-fmha-cubin)
      WITH_FMHA_CUBIN=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown аргумент: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

# Default-disable UCX/NIXL and multi-device unless explicitly enabled.
case "${VLLM_TLLM_ENABLE_UCX:-ON}" in
  1|[Oo][Nn]|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]) VLLM_TLLM_ENABLE_UCX=ON ;;
  *) VLLM_TLLM_ENABLE_UCX=OFF ;;
esac
case "${VLLM_TLLM_ENABLE_MULTI_DEVICE:-ON}" in
  1|[Oo][Nn]|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]) VLLM_TLLM_ENABLE_MULTI_DEVICE=ON ;;
  *) VLLM_TLLM_ENABLE_MULTI_DEVICE=OFF ;;
esac
case "${VLLM_TLLM_ENABLE_NVSHMEM:-OFF}" in
  1|[Oo][Nn]|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]) VLLM_TLLM_ENABLE_NVSHMEM=ON ;;
  *) VLLM_TLLM_ENABLE_NVSHMEM=OFF ;;
esac
if [[ ${WITH_UCX} -eq 1 ]]; then
  VLLM_TLLM_ENABLE_UCX=ON
fi
export VLLM_TLLM_ENABLE_UCX
export VLLM_TLLM_ENABLE_MULTI_DEVICE
export VLLM_TLLM_ENABLE_NVSHMEM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRT_VENDOR_ROOT="${VLLM_ROOT}/third_party/tensorrt"

ensure_trt_tarball() {
  local existing=""
  existing="$(find "${TRT_VENDOR_ROOT}" -maxdepth 2 -type f -name 'TensorRT-*.tar*' -print -quit 2>/dev/null || true)"
  if [[ -n "${existing}" ]]; then
    return 0
  fi
  if [[ -z "${TENSORRT_TARBALL_URL:-}" ]]; then
    return 0
  fi
  local tarball="${TRT_VENDOR_ROOT}/$(basename "${TENSORRT_TARBALL_URL}")"
  echo "Downloading TensorRT tarball to ${tarball}..."
  curl -L --fail --retry 3 --retry-delay 2 -o "${tarball}" "${TENSORRT_TARBALL_URL}"
  if [[ -n "${TENSORRT_TARBALL_SHA256:-}" ]]; then
    echo "${TENSORRT_TARBALL_SHA256}  ${tarball}" | sha256sum -c -
  fi
}

ensure_trt_extract() {
  local archive=""
  archive="$(find "${TRT_VENDOR_ROOT}" -maxdepth 2 -type f -name 'TensorRT-*.tar*' -print -quit 2>/dev/null || true)"
  if [[ -z "${archive}" ]]; then
    return 0
  fi
  local install_root="${TRT_VENDOR_ROOT}/install"
  if [[ -d "${install_root}/include" ]]; then
    return 0
  fi
  echo "Extracting TensorRT package from ${archive}..."
  mkdir -p "${install_root}"
  tar -xf "${archive}" -C "${install_root}"
  local extracted
  extracted="$(find "${install_root}" -maxdepth 1 -type d -name 'TensorRT*' -print -quit 2>/dev/null || true)"
  if [[ -n "${extracted}" && -d "${extracted}/include" ]]; then
    rsync -a "${extracted}/" "${install_root}/"
  fi
}

ensure_trt_tarball
if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found. This script is for Debian/Ubuntu."
  exit 1
fi

apt-get update

# Ensure NCCL provides window APIs (ncclWindow_t, ncclCommWindowRegister).
apt-mark unhold libnccl2 libnccl-dev || true
apt-get install -y \
  libnccl2=2.27.7-1+cuda12.9 \
  libnccl-dev=2.27.7-1+cuda12.9

pkgs=(
  ca-certificates
  curl
  libnvinfer10
  libnvinfer-dev
  libnvonnxparsers10
  libnvonnxparsers-dev
  libnvinfer-plugin10
  libnuma-dev
  libzmq3-dev
  ripgrep
  pkg-config
  autoconf
  automake
  libtool
  make
  g++
)

if [[ ${WITH_MPI} -eq 1 ]]; then
  pkgs+=(
    openmpi-bin
    libopenmpi-dev
  )
fi

if [[ ${WITH_UCX} -eq 1 ]]; then
  pkgs+=(
    libucx-dev
    ucx-utils
  )
fi

if [[ ${WITH_LFS} -eq 1 ]]; then
  pkgs+=(
    git-lfs
  )
fi

apt-get install -y "${pkgs[@]}"

ensure_trt_tarball
ensure_trt_extract

has_trt_libs() {
  if find "${TRT_VENDOR_ROOT}" -type f -name 'libnvinfer.so*' -print -quit 2>/dev/null | grep -q .; then
    return 0
  fi
  if ldconfig -p 2>/dev/null | awk '{print $1}' | grep -q '^libnvinfer\.so'; then
    return 0
  fi
  if find /usr/lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu -type f -name 'libnvinfer.so*' -print -quit 2>/dev/null | grep -q .; then
    return 0
  fi
  return 1
}

ensure_trt_debs() {
  if [[ "${TENSORRT_INSTALL_DEBS}" != "1" ]]; then
    return 0
  fi
  local py_trt_installed=0
  if dpkg -s python3-libnvinfer >/dev/null 2>&1; then
    py_trt_installed=1
  fi
  if [[ -z "${TENSORRT_DEB_VERSION}" ]]; then
    cuda_ver=""
    if [[ -f "/usr/local/cuda/version.json" ]]; then
      cuda_ver="$(python - <<'PY'
import json
try:
    with open("/usr/local/cuda/version.json", "r", encoding="utf-8") as f:
        print(json.load(f).get("cuda", {}).get("version", ""))
except Exception:
    pass
PY
)"
    elif command -v nvcc >/dev/null 2>&1; then
      cuda_ver="$(nvcc --version | awk -F'release ' '/release/ {print $2}' | awk '{print $1}')"
    fi
    if [[ "${cuda_ver}" == 12.8* ]]; then
      TENSORRT_DEB_VERSION="10.9.0.34-1+cuda12.8"
    fi
  fi
  if [[ -z "${TENSORRT_DEB_VERSION}" && ${py_trt_installed} -eq 1 && has_trt_libs ]]; then
    return 0
  fi
  local candidate=""
  candidate="$(apt-cache policy libnvinfer10 2>/dev/null | awk '/Candidate:/ {print $2}')"
  if [[ -z "${candidate}" || "${candidate}" == "(none)" ]]; then
    echo "Installing NVIDIA CUDA keyring for TensorRT debs..."
    curl -L --fail --retry 3 --retry-delay 2 -o /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update
  fi
  if [[ -n "${TENSORRT_DEB_VERSION}" ]]; then
    echo "Installing TensorRT debs pinned to ${TENSORRT_DEB_VERSION}..."
    apt-mark unhold \
      libnvinfer10 libnvinfer-dev libnvinfer-headers-dev \
      libnvinfer-plugin10 libnvinfer-plugin-dev libnvinfer-headers-plugin-dev \
      libnvinfer-lean10 libnvinfer-lean-dev \
      libnvinfer-dispatch10 libnvinfer-dispatch-dev \
      libnvinfer-vc-plugin10 libnvinfer-vc-plugin-dev \
      libnvonnxparsers10 libnvonnxparsers-dev \
      python3-libnvinfer python3-libnvinfer-dev \
      python3-libnvinfer-lean python3-libnvinfer-dispatch || true
    apt-get install -y --allow-downgrades --allow-change-held-packages \
      "libnvinfer10=${TENSORRT_DEB_VERSION}" \
      "libnvinfer-dev=${TENSORRT_DEB_VERSION}" \
      "libnvinfer-headers-dev=${TENSORRT_DEB_VERSION}" \
      "libnvinfer-plugin10=${TENSORRT_DEB_VERSION}" \
      "libnvinfer-plugin-dev=${TENSORRT_DEB_VERSION}" \
      "libnvinfer-headers-plugin-dev=${TENSORRT_DEB_VERSION}" \
      "libnvinfer-lean10=${TENSORRT_DEB_VERSION}" \
      "libnvinfer-lean-dev=${TENSORRT_DEB_VERSION}" \
      "libnvinfer-dispatch10=${TENSORRT_DEB_VERSION}" \
      "libnvinfer-dispatch-dev=${TENSORRT_DEB_VERSION}" \
      "libnvinfer-vc-plugin10=${TENSORRT_DEB_VERSION}" \
      "libnvinfer-vc-plugin-dev=${TENSORRT_DEB_VERSION}" \
      "libnvonnxparsers10=${TENSORRT_DEB_VERSION}" \
      "libnvonnxparsers-dev=${TENSORRT_DEB_VERSION}" \
      "python3-libnvinfer=${TENSORRT_DEB_VERSION}" \
      "python3-libnvinfer-lean=${TENSORRT_DEB_VERSION}" \
      "python3-libnvinfer-dispatch=${TENSORRT_DEB_VERSION}" \
      "python3-libnvinfer-dev=${TENSORRT_DEB_VERSION}"
  else
    apt-get install -y \
      libnvinfer10 \
      libnvinfer-dev \
      libnvinfer-plugin10 \
      libnvinfer-headers-plugin-dev \
      libnvonnxparsers10 \
      libnvonnxparsers-dev \
      python3-libnvinfer \
      python3-libnvinfer-dev
  fi
}

ensure_trt_debs

TRT_ROOT_CANDIDATE=""
if [[ -d "${TRT_VENDOR_ROOT}/install/include" ]]; then
  TRT_ROOT_CANDIDATE="${TRT_VENDOR_ROOT}/install"
elif [[ -d "${TRT_VENDOR_ROOT}/include" ]]; then
  TRT_ROOT_CANDIDATE="${TRT_VENDOR_ROOT}"
elif [[ -f "/usr/include/x86_64-linux-gnu/NvInfer.h" ]] || [[ -f "/usr/include/NvInfer.h" ]]; then
  TRT_ROOT_CANDIDATE="/usr"
fi
if ! has_trt_libs; then
  echo "Error: TensorRT libraries (libnvinfer.so*) not found after install."
  echo "Provide a TensorRT tarball in ${TRT_VENDOR_ROOT} or set TENSORRT_TARBALL_URL,"
  echo "or ensure NVIDIA TensorRT debs are available in your apt repos."
  exit 1
fi
if [[ -n "${TRT_ROOT_CANDIDATE}" ]]; then
  export VLLM_TENSORRT_ROOT="${VLLM_TENSORRT_ROOT:-${TRT_ROOT_CANDIDATE}}"
  export TENSORRT_ROOT="${TENSORRT_ROOT:-${VLLM_TENSORRT_ROOT}}"
  export TensorRT_ROOT="${TensorRT_ROOT:-${VLLM_TENSORRT_ROOT}}"
  export TRT_ROOT="${TRT_ROOT:-${VLLM_TENSORRT_ROOT}}"
  if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    if [[ ":${CMAKE_PREFIX_PATH}:" != *":${VLLM_TENSORRT_ROOT}:"* ]]; then
      export CMAKE_PREFIX_PATH="${VLLM_TENSORRT_ROOT}:${CMAKE_PREFIX_PATH}"
    fi
  else
    export CMAKE_PREFIX_PATH="${VLLM_TENSORRT_ROOT}"
  fi
  for libdir in "${VLLM_TENSORRT_ROOT}/lib" "${VLLM_TENSORRT_ROOT}/lib64" \
                "${VLLM_TENSORRT_ROOT}/lib/x86_64-linux-gnu" \
                "${VLLM_TENSORRT_ROOT}/targets/x86_64-linux/lib"; do
    if [[ -d "${libdir}" ]]; then
      export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
      export LIBRARY_PATH="${libdir}:${LIBRARY_PATH:-}"
    fi
  done
  echo "Using TensorRT root: ${VLLM_TENSORRT_ROOT}"
fi

ensure_trt_pythonpath() {
  if [[ ! -d "/usr/lib/python3/dist-packages" && ! -d "/usr/lib/python3.10/dist-packages" ]]; then
    return 0
  fi
  python - <<'PY'
import os
import site

dist_paths = ["/usr/lib/python3/dist-packages", "/usr/lib/python3.10/dist-packages"]
dist_paths = [p for p in dist_paths if os.path.isdir(p)]
if not dist_paths:
    raise SystemExit(0)

targets = []
for base in site.getsitepackages():
    targets.append(base)
user_site = site.getusersitepackages()
if user_site:
    targets.append(user_site)

for base in targets:
    os.makedirs(base, exist_ok=True)
    pth = os.path.join(base, "tensorrt_system.pth")
    existing = set()
    if os.path.exists(pth):
        with open(pth, "r", encoding="utf-8") as f:
            existing = {line.strip() for line in f if line.strip()}
    with open(pth, "w", encoding="utf-8") as f:
        for p in dist_paths:
            existing.add(p)
        for p in sorted(existing):
            f.write(p + "\n")
PY
}

ensure_trt_pythonpath

ensure_nvtx_headers() {
  if [[ -f "/usr/local/cuda/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/targets/x86_64-linux/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/include/nvtx3/nvtx3.hpp" ]]; then
    return 0
  fi

  echo "NVTX headers not found. Installing NVTX package..."
  apt-get install -y cuda-nvtx-12-8 || true
  apt-get install -y cuda-nvtx-dev-12-8 || true
  apt-get install -y libnvtx3-dev || true
  apt-get install -y libnvtx-dev || true
  apt-get install -y cuda-toolkit-12-8 || true

  if [[ -f "/usr/local/cuda/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/targets/x86_64-linux/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/include/nvtx3/nvtx3.hpp" ]]; then
    return 0
  fi

  echo "NVTX headers still missing after apt. Installing NVTX headers from GitHub..."
  NVTX_REPO="https://github.com/NVIDIA/NVTX.git"
  NVTX_TAG="v3.1.0"
  TMP_NVTX_DIR="/tmp/nvtx-src"
  rm -rf "${TMP_NVTX_DIR}"
  git clone --depth 1 -b "${NVTX_TAG}" "${NVTX_REPO}" "${TMP_NVTX_DIR}"
  if [[ -d "${TMP_NVTX_DIR}/c/include/nvtx3" ]]; then
    mkdir -p "/usr/local/include/nvtx3"
    cp -a "${TMP_NVTX_DIR}/c/include/nvtx3/." "/usr/local/include/nvtx3/"
  fi
  rm -rf "${TMP_NVTX_DIR}"

  if [[ -d "/usr/local/cuda/include" ]]; then
    mkdir -p "/usr/local/cuda/include/nvtx3"
    cp -f "/usr/local/include/nvtx3/nvtx3.hpp" "/usr/local/cuda/include/nvtx3/nvtx3.hpp"
  fi
  if [[ -d "/usr/local/cuda-12.8/include" ]]; then
    mkdir -p "/usr/local/cuda-12.8/include/nvtx3"
    cp -f "/usr/local/include/nvtx3/nvtx3.hpp" "/usr/local/cuda-12.8/include/nvtx3/nvtx3.hpp"
  fi

  if [[ -f "/usr/local/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/include/nvtx3/nvtx3.hpp" ]]; then
    return 0
  fi

  if [[ -f "/usr/local/cuda/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/targets/x86_64-linux/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/include/nvtx3/nvtx3.hpp" ]]; then
    return 0
  fi

  echo "NVTX headers still missing after apt. Installing NVTX headers from source..."
  python - <<'PY'
import os
import urllib.request

url = "https://raw.githubusercontent.com/NVIDIA/NVTX/v3.1.0/c/include/nvtx3/nvtx3.hpp"
dst_dir = "/usr/local/include/nvtx3"
os.makedirs(dst_dir, exist_ok=True)
dst = os.path.join(dst_dir, "nvtx3.hpp")
urllib.request.urlretrieve(url, dst)
PY

  if [[ -d "/usr/local/cuda/include" ]]; then
    mkdir -p "/usr/local/cuda/include/nvtx3"
    cp -f "/usr/local/include/nvtx3/nvtx3.hpp" "/usr/local/cuda/include/nvtx3/nvtx3.hpp"
  fi
  if [[ -d "/usr/local/cuda-12.8/include" ]]; then
    mkdir -p "/usr/local/cuda-12.8/include/nvtx3"
    cp -f "/usr/local/include/nvtx3/nvtx3.hpp" "/usr/local/cuda-12.8/include/nvtx3/nvtx3.hpp"
  fi

  if [[ -f "/usr/local/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/include/nvtx3/nvtx3.hpp" ]]; then
    return 0
  fi

  echo "Error: NVTX headers still missing. Ensure nvtx3.hpp is available in your CUDA install."
  exit 1
}

ensure_nvtx_headers

version_ge() {
  local ver_a="$1"
  local ver_b="$2"
  if [[ -z "${ver_a}" ]]; then
    return 1
  fi
  if [[ "${ver_a}" == "${ver_b}" ]]; then
    return 0
  fi
  local sorted
  sorted="$(printf '%s\n' "${ver_a}" "${ver_b}" | sort -V | head -n1)"
  [[ "${sorted}" == "${ver_b}" ]]
}

if [[ ${WITH_UCX} -eq 1 ]]; then
  UCX_REQUIRED_VERSION="1.20.0"
  UCX_VERSION_INSTALLED=""
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists ucx; then
    UCX_VERSION_INSTALLED="$(pkg-config --modversion ucx || true)"
  fi

  if ! version_ge "${UCX_VERSION_INSTALLED}" "${UCX_REQUIRED_VERSION}"; then
    echo "Building UCX from source (required >= ${UCX_REQUIRED_VERSION}, found '${UCX_VERSION_INSTALLED:-none}')"
    UCX_REPO="https://github.com/openucx/ucx.git"
    UCX_BRANCH="v1.20.x"
    UCX_COMMIT="f656dbdf93e72e60b5d6ca78b9e3d9e744e789bd"
    UCX_INSTALL_PATH="/usr/local/ucx"
    CUDA_PATH="/usr/local/cuda"
    TMP_UCX_DIR="/tmp/ucx-src"
    rm -rf "${TMP_UCX_DIR}"
    git clone -b "${UCX_BRANCH}" "${UCX_REPO}" "${TMP_UCX_DIR}"
    (cd "${TMP_UCX_DIR}" && git checkout "${UCX_COMMIT}")
    (cd "${TMP_UCX_DIR}" && ./autogen.sh)
    (cd "${TMP_UCX_DIR}" && ./contrib/configure-release \
      --prefix="${UCX_INSTALL_PATH}" \
      --enable-shared \
      --disable-static \
      --disable-doxygen-doc \
      --enable-optimizations \
      --enable-cma \
      --enable-devel-headers \
      --with-cuda="${CUDA_PATH}" \
      --with-verbs \
      --with-dm \
      --enable-mt)
    (cd "${TMP_UCX_DIR}" && make -j"$(nproc)" install)
    rm -rf "${TMP_UCX_DIR}"
  fi

  if [[ -d "/usr/local/ucx" ]]; then
    export UCX_ROOT="/usr/local/ucx"
    for libdir in "${UCX_ROOT}/lib" "${UCX_ROOT}/lib64"; do
      if [[ -d "${libdir}" ]]; then
        export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="${libdir}:${LIBRARY_PATH:-}"
        export PKG_CONFIG_PATH="${libdir}/pkgconfig:${PKG_CONFIG_PATH:-}"
      fi
    done
  fi
fi
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 --no-input --exists-action i
python -m pip install -U pip packaging setuptools wheel setuptools-scm --no-input --exists-action i
if [[ ${WITH_MPI} -eq 1 ]]; then
  python -m pip install -U mpi4py --no-input --exists-action i
fi

TRT_PYTHON_DIR="${TRT_VENDOR_ROOT}/python"
TRT_LOCAL_ROOT="${VLLM_TENSORRT_ROOT:-}"
if [[ -n "${TRT_LOCAL_ROOT}" && -d "${TRT_LOCAL_ROOT}/python" ]]; then
  TRT_PYTHON_DIR="${TRT_LOCAL_ROOT}/python"
fi
TRT_WHEEL_GLOB="${TRT_PYTHON_DIR}/build/bindings_wheel/dist/tensorrt-*.whl"

python -m pip uninstall -y tensorrt tensorrt_bindings tensorrt_libs tensorrt_dispatch_libs tensorrt_lean_libs nvidia-tensorrt \
  tensorrt_cu12 tensorrt_cu12_bindings tensorrt_cu12_libs \
  tensorrt_cu13 tensorrt_cu13_bindings tensorrt_cu13_libs \
  || true
python - <<'PY'
import site
import shutil
import os

paths = []
for base in site.getsitepackages():
    paths.append(base)
user_site = site.getusersitepackages()
if user_site:
    paths.append(user_site)

for base in paths:
    for name in (
        "tensorrt",
        "tensorrt_bindings",
        "tensorrt_libs",
        "tensorrt_dispatch_libs",
        "tensorrt_lean_libs",
        "tensorrt_cu12",
        "tensorrt_cu12_bindings",
        "tensorrt_cu12_libs",
        "tensorrt_cu13",
        "tensorrt_cu13_bindings",
        "tensorrt_cu13_libs",
    ):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
PY

if [[ ${USE_LOCAL_TENSORRT} -eq 1 && -n "${TRT_LOCAL_ROOT}" && -d "${TRT_PYTHON_DIR}" && -f "${TRT_LOCAL_ROOT}/include/NvInfer.h" ]]; then
  if ! ls ${TRT_WHEEL_GLOB} >/dev/null 2>&1; then
    PYVER=$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
    PYTHON_MAJOR_VERSION="${PYVER%%.*}"
    PYTHON_MINOR_VERSION="${PYVER##*.}"
    TRT_LIBPATH=""
    for libdir in "${TRT_LOCAL_ROOT}/lib" "${TRT_LOCAL_ROOT}/lib64" "${TRT_LOCAL_ROOT}/targets/x86_64-linux/lib"; do
      if [[ -d "${libdir}" ]]; then
        TRT_LIBPATH="${libdir}"
        break
      fi
    done
    if [[ -z "${TRT_LIBPATH}" ]]; then
      libnvinfer_path="$(find "${TRT_LOCAL_ROOT}" -type f -name 'libnvinfer.so*' -print -quit 2>/dev/null || true)"
      if [[ -n "${libnvinfer_path}" ]]; then
        TRT_LIBPATH="$(dirname "${libnvinfer_path}")"
      fi
    fi
    if [[ -z "${TRT_LIBPATH}" ]]; then
      sys_libnvinfer="$(ldconfig -p 2>/dev/null | awk '/libnvinfer\.so/ {print $NF; exit}')"
      if [[ -z "${sys_libnvinfer}" ]]; then
        sys_libnvinfer="$(find /usr/lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu -type f -name 'libnvinfer.so*' -print -quit 2>/dev/null || true)"
      fi
      if [[ -n "${sys_libnvinfer}" ]]; then
        TRT_LIBPATH="$(dirname "${sys_libnvinfer}")"
        echo "Using system TensorRT libraries from ${TRT_LIBPATH}"
      fi
    fi
    if [[ -z "${TRT_LIBPATH}" ]]; then
      echo "Error: TensorRT libraries not found."
      echo "Expected libnvinfer.so* under ${TRT_LOCAL_ROOT} or system paths."
      echo "Place an NVIDIA TensorRT tarball (TensorRT-*.tar.gz) under ${TRT_VENDOR_ROOT} or install TensorRT dev packages."
      exit 1
    fi
    echo "Building TensorRT Python wheel from ${TRT_LOCAL_ROOT}..."
    (cd "${TRT_PYTHON_DIR}" && \
      TRT_OSSPATH="${TRT_LOCAL_ROOT}" \
      TRT_LIBPATH="${TRT_LIBPATH}" \
      PYTHON_MAJOR_VERSION="${PYTHON_MAJOR_VERSION}" \
      PYTHON_MINOR_VERSION="${PYTHON_MINOR_VERSION}" \
      bash build.sh)
  fi

  if ls ${TRT_WHEEL_GLOB} >/dev/null 2>&1; then
    python -m pip install -U ${TRT_WHEEL_GLOB} --no-input --exists-action i
  else
    echo "Error: TensorRT wheel not found at ${TRT_WHEEL_GLOB}."
    exit 1
  fi
fi

if ! python - <<'PY' >/dev/null 2>&1
import tensorrt  # noqa: F401
PY
then
  if [[ ${ALLOW_PIP_TENSORRT} -eq 1 ]]; then
    echo "Installing TensorRT from NVIDIA PyPI (${TENSORRT_PIP_SPEC})..."
    python -m pip install --extra-index-url https://pypi.nvidia.com "tensorrt==${TENSORRT_PIP_SPEC}" \
      --no-input --exists-action i \
      || python -m pip install --extra-index-url https://pypi.nvidia.com "nvidia-tensorrt==${TENSORRT_PIP_SPEC}" \
        --no-input --exists-action i
  else
    echo "Error: TensorRT import failed and pip fallback is disabled (ALLOW_PIP_TENSORRT=0)."
    exit 1
  fi
fi

if [[ ${WITH_LFS} -eq 1 ]]; then
  if command -v git >/dev/null 2>&1; then
    TLLM_DIR="${SCRIPT_DIR}/../third_party/tensorrt_llm"
    if [[ -d "${TLLM_DIR}/.git" ]]; then
      git -C "${TLLM_DIR}" lfs install
      git -C "${TLLM_DIR}" lfs pull
    else
      echo "Warning: ${TLLM_DIR} is not a git repo; skipping git lfs pull."
    fi
  else
    echo "Warning: git not found; skipping git lfs pull."
  fi
fi

if [[ ${WITH_FMHA_CUBIN} -eq 1 ]]; then
  # SCRIPT_DIR and VLLM_ROOT already set above.
  TLLM_DIR="${SCRIPT_DIR}/../third_party/tensorrt_llm"
  FMHA_DIR="${TLLM_DIR}/cpp/kernels/fmha_v2"
  CUBIN_DST="${TLLM_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin"
  FMHA_V2_CU_DST="${TLLM_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmha_v2_cu"

  if [[ ! -d "${FMHA_DIR}" ]]; then
    echo "Error: ${FMHA_DIR} not found; cannot generate FMHA cubins."
    exit 1
  fi

  if [[ ! -d "${CUBIN_DST}" ]]; then
    mkdir -p "${CUBIN_DST}"
  fi

  # Ensure TRTLLM-specific header is reachable for generated kernels.
  TRTLLM_FMHA_HEADER="${TLLM_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
  if [[ -f "${TRTLLM_FMHA_HEADER}" ]]; then
    cp -f "${TRTLLM_FMHA_HEADER}" "${FMHA_DIR}/"
  fi

  # Ensure TRTLLM headers are visible to fmha_v2 build.
  export CXXFLAGS="${CXXFLAGS:-} -I${TLLM_DIR}/cpp -I${TLLM_DIR}/cpp/include -I${TLLM_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention"
  export CUDAFLAGS="${CUDAFLAGS:-} -I${TLLM_DIR}/cpp -I${TLLM_DIR}/cpp/include -I${TLLM_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention"

  export VLLM_FMHA_CUDA_ARCH_LIST="${VLLM_FMHA_CUDA_ARCH_LIST:-8.0}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${VLLM_FMHA_CUDA_ARCH_LIST}}"
  export CUDA_ARCHS="${CUDA_ARCHS:-${TORCH_CUDA_ARCH_LIST}}"
  export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-${CUDA_ARCHS}}"
  export ENABLE_HMMA_FP32=1
  export GENERATE_CUBIN=1
  export GENERATE_CU_TRTLLM=true
  export VLLM_FMHA_USE_CUBIN_HEADER_SM80=1
  TMP_DIR="${FMHA_DIR}/temp"
  mkdir -p "${TMP_DIR}"
  mkdir -p "${FMHA_DIR}/cubin"
  mkdir -p "${FMHA_DIR}/generated"

  (cd "${FMHA_DIR}" && python3 setup.py && make -j cubin TMP_DIR="${TMP_DIR}" 2>&1 | tee ${SCRIPT_DIR}/../install.log)

  # If generated/cubin landed in repo root, sync them back to fmha_v2.
  ROOT_GENERATED="${VLLM_ROOT}/generated"
  ROOT_CUBIN="${VLLM_ROOT}/cubin"
  if [[ -d "${ROOT_GENERATED}" ]]; then
    mkdir -p "${FMHA_DIR}/generated"
    cp -a "${ROOT_GENERATED}/." "${FMHA_DIR}/generated/"
  fi
  if [[ -d "${ROOT_CUBIN}" ]]; then
    mkdir -p "${FMHA_DIR}/cubin"
    cp -a "${ROOT_CUBIN}/." "${FMHA_DIR}/cubin/"
  fi

  # Copy generated *.cu kernels into the location CMake expects.
  if [[ -d "${FMHA_DIR}/generated" ]]; then
    mkdir -p "${FMHA_V2_CU_DST}"
    if compgen -G "${FMHA_DIR}/generated/*_sm*.cu" > /dev/null; then
      cp -f "${FMHA_DIR}/generated/"*_sm*.cu "${FMHA_V2_CU_DST}/"
    fi
  fi

  # Copy generated fmha_cubin.* and cubin blobs into TensorRT-LLM cubin dir.
  if [[ -f "${FMHA_DIR}/generated/fmha_cubin.h" ]]; then
    cp -f "${FMHA_DIR}/generated/fmha_cubin.h" "${CUBIN_DST}/"
  fi
  if [[ -f "${FMHA_DIR}/generated/fmha_cubin.cpp" ]]; then
    cp -f "${FMHA_DIR}/generated/fmha_cubin.cpp" "${CUBIN_DST}/"
  fi
  if compgen -G "${FMHA_DIR}/cubin/*.cubin.cpp" > /dev/null; then
    cp -f "${FMHA_DIR}/cubin/"*.cubin.cpp "${CUBIN_DST}/"
  fi
  if compgen -G "${FMHA_DIR}/cubin/*.cu.cubin" > /dev/null; then
    cp -f "${FMHA_DIR}/cubin/"*.cu.cubin "${CUBIN_DST}/"
  fi

  # Ensure cubin data symbols live under tensorrt_llm::kernels (no ABI inline ns).
  if compgen -G "${CUBIN_DST}/*.cubin.cpp" > /dev/null; then
    for f in "${CUBIN_DST}/"*.cubin.cpp; do
      if grep -q "TRTLLM_NAMESPACE_BEGIN" "${f}"; then
        tmp="${f}.tmp"
        sed -e 's/TRTLLM_NAMESPACE_BEGIN/namespace tensorrt_llm {/' \
            -e 's/TRTLLM_NAMESPACE_END/}/' \
            "${f}" > "${tmp}"
        mv -f "${tmp}" "${f}"
        continue
      fi
      if ! grep -q "namespace tensorrt_llm" "${f}"; then
        tmp="${f}.tmp"
        {
          echo 'namespace tensorrt_llm {'
          echo 'namespace kernels {'
          cat "${f}"
          echo '} // namespace kernels'
          echo '} // namespace tensorrt_llm'
        } > "${tmp}"
        mv -f "${tmp}" "${f}"
      fi
    done
  fi
fi

cat <<'EOF'

Notes:
- If you do NOT install MPI/UCX, disable the features during build:
  export VLLM_TLLM_ENABLE_MULTI_DEVICE=OFF
  export VLLM_TLLM_ENABLE_UCX=OFF
- If you need LFS assets in third_party/tensorrt_llm, run with --with-lfs.
- To generate SM80 FMHA cubins, run with --with-fmha-cubin
  (optionally set VLLM_FMHA_CUDA_ARCH_LIST=8.0).
EOF
