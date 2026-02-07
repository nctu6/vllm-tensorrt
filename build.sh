#!/usr/bin/env bash
set -euo pipefail

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-80}"
export CUDA_ARCHS="${CUDA_ARCHS:-${TORCH_CUDA_ARCH_LIST}}"
export GENERATE_CU_TRTLLM=true
export VLLM_FMHA_USE_CUBIN_HEADER_SM80=1

# If fmha_v2 generation ran from repo root, sync outputs into TLLM paths.
VLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_GENERATED="${VLLM_ROOT}/generated"
ROOT_CUBIN="${VLLM_ROOT}/cubin"
TLLM_DIR="${VLLM_ROOT}/third_party/tensorrt_llm"
FMHA_DIR="${TLLM_DIR}/cpp/kernels/fmha_v2"
FMHA_V2_CU_DST="${TLLM_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmha_v2_cu"
CUBIN_DST="${TLLM_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin"
FMHA_RECONFIGURE=0

# Ensure TRTLLM FMHA header is reachable for generated kernels.
TRTLLM_FMHA_HEADER="${TLLM_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
if [[ -f "${TRTLLM_FMHA_HEADER}" ]]; then
  cp -f "${TRTLLM_FMHA_HEADER}" "${FMHA_DIR}/"
fi

# Make TRTLLM headers visible during fmha_v2 generation.
export CXXFLAGS="${CXXFLAGS:-} -I${TLLM_DIR}/cpp -I${TLLM_DIR}/cpp/include -I${TLLM_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention"
export CUDAFLAGS="${CUDAFLAGS:-} -I${TLLM_DIR}/cpp -I${TLLM_DIR}/cpp/include -I${TLLM_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention"

if [[ -d "${FMHA_DIR}/generated" ]]; then
  mkdir -p "${FMHA_V2_CU_DST}" "${CUBIN_DST}"
  if compgen -G "${FMHA_DIR}/generated/*_sm*.cu" > /dev/null; then
    cp -f "${FMHA_DIR}/generated/"*_sm*.cu "${FMHA_V2_CU_DST}/"
  fi
  if [[ -f "${FMHA_DIR}/generated/fmha_cubin.h" ]]; then
    cp -f "${FMHA_DIR}/generated/fmha_cubin.h" "${CUBIN_DST}/"
  fi
  if [[ -f "${FMHA_DIR}/generated/fmha_cubin.cpp" ]]; then
    cp -f "${FMHA_DIR}/generated/fmha_cubin.cpp" "${CUBIN_DST}/"
  fi
  FMHA_RECONFIGURE=1
fi

if [[ -d "${FMHA_DIR}/cubin" ]]; then
  mkdir -p "${CUBIN_DST}"
  if compgen -G "${FMHA_DIR}/cubin/*.cubin.cpp" > /dev/null; then
    cp -f "${FMHA_DIR}/cubin/"*.cubin.cpp "${CUBIN_DST}/"
  fi
  if compgen -G "${FMHA_DIR}/cubin/*.cu.cubin" > /dev/null; then
    cp -f "${FMHA_DIR}/cubin/"*.cu.cubin "${CUBIN_DST}/"
  fi
  FMHA_RECONFIGURE=1
fi

if [[ -d "${ROOT_GENERATED}" ]]; then
  mkdir -p "${FMHA_DIR}/generated" "${FMHA_V2_CU_DST}"
  cp -a "${ROOT_GENERATED}/." "${FMHA_DIR}/generated/"
  if compgen -G "${ROOT_GENERATED}/*_sm*.cu" > /dev/null; then
    cp -f "${ROOT_GENERATED}/"*_sm*.cu "${FMHA_V2_CU_DST}/"
  fi
  if [[ -f "${ROOT_GENERATED}/fmha_cubin.h" ]]; then
    mkdir -p "${CUBIN_DST}"
    cp -f "${ROOT_GENERATED}/fmha_cubin.h" "${CUBIN_DST}/"
  fi
  if [[ -f "${ROOT_GENERATED}/fmha_cubin.cpp" ]]; then
    mkdir -p "${CUBIN_DST}"
    cp -f "${ROOT_GENERATED}/fmha_cubin.cpp" "${CUBIN_DST}/"
  fi
  FMHA_RECONFIGURE=1
fi

if [[ -d "${ROOT_CUBIN}" ]]; then
  mkdir -p "${FMHA_DIR}/cubin" "${CUBIN_DST}"
  cp -a "${ROOT_CUBIN}/." "${FMHA_DIR}/cubin/"
  if compgen -G "${ROOT_CUBIN}/*.cubin.cpp" > /dev/null; then
    cp -f "${ROOT_CUBIN}/"*.cubin.cpp "${CUBIN_DST}/"
  fi
  if compgen -G "${ROOT_CUBIN}/*.cu.cubin" > /dev/null; then
    cp -f "${ROOT_CUBIN}/"*.cu.cubin "${CUBIN_DST}/"
  fi
  FMHA_RECONFIGURE=1
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

CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: ./build.sh [--clean]
  --clean   Remove build artifacts before building
EOF
      exit 0
      ;;
  esac
done

if [[ "$CLEAN" -eq 1 || "$FMHA_RECONFIGURE" -eq 1 ]]; then
  echo "[build.sh] --clean specified, removing build artifacts..."
  rm -rf build/temp/
  rm -rf third_party/tensorrt_llm/cpp/build_RelWithDebInfo/
fi

pip install --no-build-isolation -e . 2>&1 | tee build.log
