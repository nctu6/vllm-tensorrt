#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: install_tensorrt_deps.sh [--with-mpi] [--with-ucx] [--with-lfs] [--with-fmha-cubin] [--no-fmha-cubin]

Installs TensorRT runtime deps. Optional flags add packages required for
TensorRT-LLM multi-device (MPI), UCX/NIXL features, Git LFS assets, or
generate FMHA v2 cubins for SM80 (enabled by default).
EOF
}

WITH_MPI=0
WITH_UCX=0
WITH_LFS=0
WITH_FMHA_CUBIN=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-mpi)
      WITH_MPI=1
      ;;
    --with-ucx)
      WITH_UCX=1
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

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found. This script is for Debian/Ubuntu."
  exit 1
fi

apt-get update

pkgs=(
  libnvinfer10
  libnvinfer-dev
  libnvonnxparsers10
  libnvinfer-plugin10
  libnuma-dev
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
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 --no-input --exists-action i
python -m pip install -U pip packaging setuptools wheel setuptools-scm --no-input --exists-action i

if [[ ${WITH_LFS} -eq 1 ]]; then
  if command -v git >/dev/null 2>&1; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
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
