#!/usr/bin/env bash
set -euo pipefail

# Repo root.
VLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer vendored TensorRT; fall back to any user-provided env.
TRT_VENDOR_ROOT="${VLLM_ROOT}/third_party/tensorrt"
TRT_ROOT_CANDIDATE=""
if [[ -d "${TRT_VENDOR_ROOT}/install/include" ]]; then
  TRT_ROOT_CANDIDATE="${TRT_VENDOR_ROOT}/install"
elif [[ -d "${TRT_VENDOR_ROOT}/include" ]]; then
  TRT_ROOT_CANDIDATE="${TRT_VENDOR_ROOT}"
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
                "${VLLM_TENSORRT_ROOT}/targets/x86_64-linux/lib"; do
    if [[ -d "${libdir}" ]]; then
      export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
      export LIBRARY_PATH="${libdir}:${LIBRARY_PATH:-}"
    fi
  done
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
# Default to SM80 only unless overridden via --cuda_architectures.
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-80}"
export CUDA_ARCHS="${CUDA_ARCHS:-${TORCH_CUDA_ARCH_LIST}}"
export GENERATE_CU_TRTLLM=true
export VLLM_FMHA_USE_CUBIN_HEADER_SM80=1

# Match TensorRT-LLM defaults: UCX and multi-device are ON unless explicitly set.
normalize_bool() {
  case "${1:-}" in
    1|[Oo][Nn]|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]) echo "ON" ;;
    *) echo "OFF" ;;
  esac
}
export VLLM_TLLM_ENABLE_UCX
export VLLM_TLLM_ENABLE_MULTI_DEVICE
export VLLM_TLLM_ENABLE_NVSHMEM
VLLM_TLLM_ENABLE_UCX="$(normalize_bool "${VLLM_TLLM_ENABLE_UCX:-ON}")"
VLLM_TLLM_ENABLE_MULTI_DEVICE="$(normalize_bool "${VLLM_TLLM_ENABLE_MULTI_DEVICE:-ON}")"
VLLM_TLLM_ENABLE_NVSHMEM="$(normalize_bool "${VLLM_TLLM_ENABLE_NVSHMEM:-OFF}")"
export VLLM_TENSORRT_LLM_ENABLE_UCX="${VLLM_TLLM_ENABLE_UCX}"
TLLM_ENABLE_UCX="${VLLM_TLLM_ENABLE_UCX}"
TLLM_ENABLE_MULTI_DEVICE="${VLLM_TLLM_ENABLE_MULTI_DEVICE}"
TLLM_ENABLE_NVSHMEM="${VLLM_TLLM_ENABLE_NVSHMEM}"

# Prefer UCX from /usr/local/ucx if present.
if [[ -d "/usr/local/ucx" ]]; then
  export UCX_ROOT="${UCX_ROOT:-/usr/local/ucx}"
  for libdir in "${UCX_ROOT}/lib" "${UCX_ROOT}/lib64"; do
    if [[ -d "${libdir}" ]]; then
      export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
      export LIBRARY_PATH="${libdir}:${LIBRARY_PATH:-}"
      export PKG_CONFIG_PATH="${libdir}/pkgconfig:${PKG_CONFIG_PATH:-}"
    fi
  done
fi

# If fmha_v2 generation ran from repo root, sync outputs into TLLM paths.
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

CUDA_ARCH_OVERRIDE=""
CLEAN=0
TLLM_RECONFIGURE=0
TLLM_CONFIGURE=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --cuda_architectures=*)
      CUDA_ARCH_OVERRIDE="${arg#*=}"
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./build.sh [--clean] [--cuda_architectures=LIST]
  --clean   Remove build artifacts before building
  --cuda_architectures  Override CMAKE_CUDA_ARCHITECTURES (e.g. "80" or "80;90-real")
EOF
      exit 0
      ;;
  esac
done

if [[ -n "${CUDA_ARCH_OVERRIDE}" ]]; then
  export CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH_OVERRIDE}"
fi

# If the TensorRT root changed, force a TLLM rebuild.
TLLM_BUILD_DIR="${TLLM_DIR}/cpp/build_RelWithDebInfo"
# Force Unix Makefiles by default to avoid generator mismatches.
TLLM_GENERATOR="Unix Makefiles"
CMAKE_GENERATOR_VALUE="${TLLM_GENERATOR}"

# Clean stale FetchContent subbuilds if their generator differs.
DEPS_DIR="${VLLM_ROOT}/.deps"
if [[ -d "${DEPS_DIR}" ]]; then
  for subbuild in "${DEPS_DIR}"/*-subbuild; do
    [[ -d "${subbuild}" ]] || continue
    cache="${subbuild}/CMakeCache.txt"
    if [[ -f "${cache}" ]]; then
      cached_generator="$(grep -m1 '^CMAKE_GENERATOR:INTERNAL=' "${cache}" | sed 's/.*=//')"
      if [[ -n "${cached_generator}" && "${cached_generator}" != "${CMAKE_GENERATOR_VALUE}" ]]; then
        echo "[build.sh] Cleaning stale subbuild cache ${subbuild} (${cached_generator} -> ${CMAKE_GENERATOR_VALUE})"
        rm -rf "${subbuild}"
      fi
    fi
  done
fi
if [[ -f "${TLLM_BUILD_DIR}/CMakeCache.txt" && -n "${VLLM_TENSORRT_ROOT:-}" ]]; then
  cached_trt_root="$(grep -m1 '^TensorRT_ROOT:.*=' "${TLLM_BUILD_DIR}/CMakeCache.txt" | sed 's/.*=//')"
  if [[ -n "${cached_trt_root}" && "${cached_trt_root}" != "${VLLM_TENSORRT_ROOT}" ]]; then
    TLLM_RECONFIGURE=1
  fi
fi
if [[ -f "${TLLM_BUILD_DIR}/CMakeCache.txt" ]]; then
  cached_ucx="$(grep -m1 '^ENABLE_UCX:BOOL=' "${TLLM_BUILD_DIR}/CMakeCache.txt" | sed 's/.*=//')"
  cached_multi_device="$(grep -m1 '^ENABLE_MULTI_DEVICE:BOOL=' "${TLLM_BUILD_DIR}/CMakeCache.txt" | sed 's/.*=//')"
  cached_nvshmem="$(grep -m1 '^ENABLE_NVSHMEM:BOOL=' "${TLLM_BUILD_DIR}/CMakeCache.txt" | sed 's/.*=//')"
  if [[ -n "${cached_ucx}" && "${cached_ucx}" != "${TLLM_ENABLE_UCX}" ]]; then
    TLLM_RECONFIGURE=1
  fi
  if [[ -n "${cached_multi_device}" && "${cached_multi_device}" != "${TLLM_ENABLE_MULTI_DEVICE}" ]]; then
    TLLM_RECONFIGURE=1
  fi
  if [[ -n "${cached_nvshmem}" && "${cached_nvshmem}" != "${TLLM_ENABLE_NVSHMEM}" ]]; then
    TLLM_RECONFIGURE=1
  fi
fi
if [[ -f "${TLLM_BUILD_DIR}/CMakeCache.txt" ]]; then
  cached_generator="$(grep -m1 '^CMAKE_GENERATOR:INTERNAL=' "${TLLM_BUILD_DIR}/CMakeCache.txt" | sed 's/.*=//')"
  if [[ -n "${cached_generator}" && "${cached_generator}" != "${TLLM_GENERATOR}" ]]; then
    echo "[build.sh] TensorRT-LLM generator mismatch (${cached_generator} -> ${TLLM_GENERATOR}); cleaning ${TLLM_BUILD_DIR}"
    rm -rf "${TLLM_BUILD_DIR}"
    TLLM_RECONFIGURE=1
  fi
fi

# Clean stale vendored TensorRT build cache if generator differs.
TRT_BUILD_DIR="${TRT_VENDOR_ROOT}/build_RelWithDebInfo"
if [[ -f "${TRT_BUILD_DIR}/CMakeCache.txt" ]]; then
  cached_trt_generator="$(grep -m1 '^CMAKE_GENERATOR:INTERNAL=' "${TRT_BUILD_DIR}/CMakeCache.txt" | sed 's/.*=//')"
  if [[ -n "${cached_trt_generator}" && "${cached_trt_generator}" != "${CMAKE_GENERATOR_VALUE}" ]]; then
    echo "[build.sh] TensorRT generator mismatch (${cached_trt_generator} -> ${CMAKE_GENERATOR_VALUE}); cleaning ${TRT_BUILD_DIR}"
    rm -rf "${TRT_BUILD_DIR}"
  fi
fi

if [[ "$FMHA_RECONFIGURE" -eq 1 || "$TLLM_RECONFIGURE" -eq 1 ]]; then
  TLLM_CONFIGURE=1
fi

if [[ "$CLEAN" -eq 1 ]]; then
  echo "[build.sh] --clean specified, removing build artifacts..."
  rm -rf build/temp/
  rm -rf third_party/tensorrt_llm/cpp/build_RelWithDebInfo/
fi

# Ensure TensorRT-LLM Python bindings are built before installing vLLM.
TLLM_BINDINGS_DST="${VLLM_ROOT}/tensorrt_llm"
mkdir -p "${TLLM_BINDINGS_DST}"

if ! compgen -G "${TLLM_BINDINGS_DST}/bindings*.so" > /dev/null && \
   ! compgen -G "${TLLM_BINDINGS_DST}/_tensorrt_llm*.so" > /dev/null; then
  if [[ -d "${TLLM_BUILD_DIR}" ]]; then
    echo "[build.sh] Building TensorRT-LLM bindings target (bindings)..."
    set +e
    cmake --build "${TLLM_BUILD_DIR}" --target bindings
    cmake_status=$?
    set -e
    if [[ ${cmake_status} -ne 0 ]]; then
      echo "[build.sh] bindings target not available or failed; falling back to build_wheel.py."
    fi
  fi

  if ! compgen -G "${TLLM_BUILD_DIR}/tensorrt_llm/bindings*.so" > /dev/null && \
     ! compgen -G "${TLLM_BUILD_DIR}/tensorrt_llm/bindings/bindings*.so" > /dev/null && \
     ! compgen -G "${TLLM_BUILD_DIR}/tensorrt_llm/bindings/_tensorrt_llm*.so" > /dev/null && \
     ! compgen -G "${TLLM_BUILD_DIR}/tensorrt_llm/nanobind/bindings*.so" > /dev/null; then
    echo "[build.sh] Building TensorRT-LLM bindings via build_wheel.py..."
    TLLM_CONFIGURE=1
    python "${TLLM_DIR}/scripts/build_wheel.py" \
      --no-venv \
      --build_type RelWithDebInfo \
      --build_dir "${TLLM_BUILD_DIR}" \
      --generator "${TLLM_GENERATOR}" \
      --cuda_architectures "${CMAKE_CUDA_ARCHITECTURES}" \
      --skip_building_wheel \
      ${TLLM_CONFIGURE:+--configure_cmake} \
      --extra-cmake-vars "ENABLE_UCX=${TLLM_ENABLE_UCX}" \
      --extra-cmake-vars "ENABLE_MULTI_DEVICE=${TLLM_ENABLE_MULTI_DEVICE}" \
      --extra-cmake-vars "ENABLE_NVSHMEM=${TLLM_ENABLE_NVSHMEM}" \
      ${VLLM_TENSORRT_ROOT:+--trt_root "${VLLM_TENSORRT_ROOT}"}
  fi
fi

# If TensorRT-LLM bindings were built, copy them into the vendored package.
TLLM_BINDINGS_SRC=""
TLLM_BINDINGS_FROM_DST=0
TLLM_EP_BUILD_DIR="${VLLM_ROOT}/build/temp/tensorrt_llm_build-prefix/src/tensorrt_llm_build-build"
for cand in "${TLLM_DIR}"/cpp/build_*/tensorrt_llm/bindings*.so \
            "${TLLM_DIR}"/cpp/build_*/tensorrt_llm/bindings/bindings*.so \
            "${TLLM_DIR}"/cpp/build_*/tensorrt_llm/bindings/_tensorrt_llm*.so \
            "${TLLM_DIR}"/cpp/build_*/tensorrt_llm/nanobind/bindings*.so \
            "${TLLM_EP_BUILD_DIR}"/tensorrt_llm/nanobind/bindings*.so; do
  if [[ -f "${cand}" ]]; then
    TLLM_BINDINGS_SRC="${cand}"
  fi
done
if [[ -z "${TLLM_BINDINGS_SRC}" ]]; then
  mapfile -t _tllm_binding_candidates < <(find "${TLLM_DIR}/cpp" "${TLLM_EP_BUILD_DIR}" \
    \( -name "bindings*.so" -o -name "_tensorrt_llm*.so" \) \
    -path "*/build_*/*" -type f 2>/dev/null)
  for cand in "${_tllm_binding_candidates[@]}"; do
    TLLM_BINDINGS_SRC="${cand}"
    break
  done
fi
if [[ -z "${TLLM_BINDINGS_SRC}" ]]; then
  for cand in "${TLLM_BINDINGS_DST}"/bindings*.so \
              "${TLLM_BINDINGS_DST}"/_tensorrt_llm*.so; do
    if [[ -f "${cand}" ]]; then
      TLLM_BINDINGS_SRC="${cand}"
      TLLM_BINDINGS_FROM_DST=1
      break
    fi
  done
fi
if [[ -n "${TLLM_BINDINGS_SRC}" ]]; then
  if [[ "${TLLM_BINDINGS_FROM_DST}" -eq 1 ]]; then
    echo "[build.sh] Reusing existing TensorRT-LLM binding: ${TLLM_BINDINGS_SRC}"
  else
    cp -f "${TLLM_BINDINGS_SRC}" "${TLLM_BINDINGS_DST}/"
    echo "[build.sh] Installed TensorRT-LLM binding: ${TLLM_BINDINGS_SRC}"
  fi
else
  echo "[build.sh] Error: TensorRT-LLM bindings not found after build."
  echo "[build.sh] Expected bindings*.so (or _tensorrt_llm*.so) under ${TLLM_DIR}/cpp/build_*/"
  exit 1
fi

# Ensure kv_cache_manager_v2 rawref extension exists in the vendored package.
RAWREF_DIR="${TLLM_BINDINGS_DST}/runtime/kv_cache_manager_v2/rawref"
if [[ -d "${RAWREF_DIR}" ]]; then
  if ! compgen -G "${RAWREF_DIR}/_rawref*.so" > /dev/null; then
    echo "[build.sh] Building TensorRT-LLM rawref extension..."
    if [[ -x "${VLLM_ROOT}/.venv/bin/python" ]]; then
      PYTHON_BIN="${VLLM_ROOT}/.venv/bin/python"
    else
      PYTHON_BIN="$(command -v python || command -v python3)"
    fi
    if [[ -z "${PYTHON_BIN}" ]]; then
      echo "[build.sh] Error: python not found for rawref build."
      exit 1
    fi
    pushd "${RAWREF_DIR}" >/dev/null
    "${PYTHON_BIN}" setup.py build_ext --inplace
    popd >/dev/null
  fi
fi

# Ensure TensorRT-LLM deep_gemm module is available.
DEEP_GEMM_DIR="${TLLM_BINDINGS_DST}/deep_gemm"
DEEP_GEMM_SO="${TLLM_BINDINGS_DST}/deep_gemm_cpp_tllm"
if ! compgen -G "${DEEP_GEMM_SO}"*.so > /dev/null || [[ ! -d "${DEEP_GEMM_DIR}" ]]; then
  if [[ -d "${TLLM_BUILD_DIR}" ]]; then
    echo "[build.sh] Building TensorRT-LLM deep_gemm target..."
    set +e
    cmake --build "${TLLM_BUILD_DIR}" --target deep_gemm
    deep_gemm_status=$?
    set -e
    if [[ ${deep_gemm_status} -ne 0 ]]; then
      echo "[build.sh] Warning: deep_gemm target failed; attempting to use existing artifacts if available."
    fi
  fi
  deep_gemm_so_src=""
  for cand in "${TLLM_BUILD_DIR}/tensorrt_llm/deep_gemm/"deep_gemm_cpp_tllm*.so; do
    if [[ -f "${cand}" ]]; then
      deep_gemm_so_src="${cand}"
      break
    fi
  done
  deep_gemm_py_src="${TLLM_BUILD_DIR}/tensorrt_llm/deep_gemm/python/deep_gemm"
  if [[ -n "${deep_gemm_so_src}" ]]; then
    cp -f "${deep_gemm_so_src}" "${TLLM_BINDINGS_DST}/"
  fi
  if [[ -d "${deep_gemm_py_src}" ]]; then
    rm -rf "${DEEP_GEMM_DIR}"
    mkdir -p "${DEEP_GEMM_DIR}"
    cp -a "${deep_gemm_py_src}/." "${DEEP_GEMM_DIR}/"
  fi
fi
if ! compgen -G "${DEEP_GEMM_SO}"*.so > /dev/null || [[ ! -d "${DEEP_GEMM_DIR}" ]]; then
  echo "[build.sh] Error: TensorRT-LLM deep_gemm artifacts not found after build."
  echo "[build.sh] Expected ${DEEP_GEMM_SO}*.so and ${DEEP_GEMM_DIR}/"
  exit 1
fi

pip install --no-build-isolation -e . 2>&1 | tee pip_install.log

# If TensorRT-LLM bindings were built, copy them into the vendored package.
TLLM_BINDINGS_DST="${VLLM_ROOT}/tensorrt_llm"
mkdir -p "${TLLM_BINDINGS_DST}"
TLLM_BINDINGS_SRC=""
TLLM_BINDINGS_FROM_DST=0
TLLM_EP_BUILD_DIR="${VLLM_ROOT}/build/temp/tensorrt_llm_build-prefix/src/tensorrt_llm_build-build"
for cand in "${TLLM_DIR}"/cpp/build_*/tensorrt_llm/bindings*.so \
            "${TLLM_DIR}"/cpp/build_*/tensorrt_llm/bindings/bindings*.so \
            "${TLLM_DIR}"/cpp/build_*/tensorrt_llm/bindings/_tensorrt_llm*.so \
            "${TLLM_DIR}"/cpp/build_*/tensorrt_llm/nanobind/bindings*.so \
            "${TLLM_EP_BUILD_DIR}"/tensorrt_llm/nanobind/bindings*.so; do
  if [[ -f "${cand}" ]]; then
    TLLM_BINDINGS_SRC="${cand}"
  fi
done
if [[ -z "${TLLM_BINDINGS_SRC}" ]]; then
  mapfile -t _tllm_binding_candidates < <(find "${TLLM_DIR}/cpp" "${TLLM_EP_BUILD_DIR}" \
    \( -name "bindings*.so" -o -name "_tensorrt_llm*.so" \) \
    -path "*/build_*/*" -type f 2>/dev/null)
  for cand in "${_tllm_binding_candidates[@]}"; do
    TLLM_BINDINGS_SRC="${cand}"
    break
  done
fi
if [[ -z "${TLLM_BINDINGS_SRC}" ]]; then
  for cand in "${TLLM_BINDINGS_DST}"/bindings*.so \
              "${TLLM_BINDINGS_DST}"/_tensorrt_llm*.so; do
    if [[ -f "${cand}" ]]; then
      TLLM_BINDINGS_SRC="${cand}"
      TLLM_BINDINGS_FROM_DST=1
      break
    fi
  done
fi
if [[ -n "${TLLM_BINDINGS_SRC}" ]]; then
  if [[ "${TLLM_BINDINGS_FROM_DST}" -eq 1 ]]; then
    echo "[build.sh] Reusing existing TensorRT-LLM binding: ${TLLM_BINDINGS_SRC}"
  else
    cp -f "${TLLM_BINDINGS_SRC}" "${TLLM_BINDINGS_DST}/"
    echo "[build.sh] Installed TensorRT-LLM binding: ${TLLM_BINDINGS_SRC}"
  fi
else
  echo "[build.sh] Warning: TensorRT-LLM bindings not found under ${TLLM_DIR}/cpp/build_*/"
fi
