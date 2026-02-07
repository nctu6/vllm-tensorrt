# Build TensorRT-LLM (vendored) for CUDA builds.

include(ExternalProject)

if(DEFINED ENV{VLLM_TENSORRT_LLM_SRC_DIR})
  set(VLLM_TENSORRT_LLM_SRC_DIR $ENV{VLLM_TENSORRT_LLM_SRC_DIR})
endif()

if(NOT VLLM_TENSORRT_LLM_SRC_DIR)
  set(VLLM_TENSORRT_LLM_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/../../third_party/tensorrt_llm")
endif()

set(TLLM_CPP_DIR "${VLLM_TENSORRT_LLM_SRC_DIR}/cpp")

if(NOT EXISTS "${TLLM_CPP_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "TensorRT-LLM source not found at ${TLLM_CPP_DIR}. "
    "Set VLLM_TENSORRT_LLM_SRC_DIR to the vendored TensorRT-LLM source tree.")
endif()

# Match TensorRT-LLM build_wheel.py build dir convention.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(TLLM_BUILD_DIR "${TLLM_CPP_DIR}/build")
else()
  set(TLLM_BUILD_DIR "${TLLM_CPP_DIR}/build_${CMAKE_BUILD_TYPE}")
endif()

set(TLLM_BUILD_PYT "ON")
set(TLLM_BUILD_DEEP_EP "ON")
set(TLLM_BUILD_DEEP_GEMM "ON")
set(TLLM_BUILD_FLASH_MLA "ON")
set(TLLM_ENABLE_MULTI_DEVICE "OFF")
set(TLLM_ENABLE_UCX "OFF")

# Ensure CUDA_ARCHS is populated from env when available so TLLM doesn't fall
# back to "native" (which filters out all sm* sources).
if(NOT DEFINED CUDA_ARCHS OR CUDA_ARCHS STREQUAL "")
  if(DEFINED ENV{CUDA_ARCHS} AND NOT "$ENV{CUDA_ARCHS}" STREQUAL "")
    set(CUDA_ARCHS "$ENV{CUDA_ARCHS}")
  elseif(DEFINED ENV{TORCH_CUDA_ARCH_LIST} AND NOT "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
    set(CUDA_ARCHS "$ENV{TORCH_CUDA_ARCH_LIST}")
  elseif(DEFINED ENV{VLLM_FMHA_CUDA_ARCH_LIST} AND NOT "$ENV{VLLM_FMHA_CUDA_ARCH_LIST}" STREQUAL "")
    set(CUDA_ARCHS "$ENV{VLLM_FMHA_CUDA_ARCH_LIST}")
  endif()
endif()

# If TensorRT-LLM has been configured before, reuse its cached options.
set(TLLM_CACHE_FILE "${TLLM_BUILD_DIR}/CMakeCache.txt")
if(EXISTS "${TLLM_CACHE_FILE}")
  file(STRINGS "${TLLM_CACHE_FILE}" _tllm_cache_lines REGEX "^BUILD_.*:BOOL=")
  foreach(_line ${_tllm_cache_lines})
    if(_line MATCHES "^BUILD_PYT:BOOL=(.*)$")
      set(TLLM_BUILD_PYT "${CMAKE_MATCH_1}")
    elseif(_line MATCHES "^BUILD_DEEP_EP:BOOL=(.*)$")
      set(TLLM_BUILD_DEEP_EP "${CMAKE_MATCH_1}")
    elseif(_line MATCHES "^BUILD_DEEP_GEMM:BOOL=(.*)$")
      set(TLLM_BUILD_DEEP_GEMM "${CMAKE_MATCH_1}")
    elseif(_line MATCHES "^BUILD_FLASH_MLA:BOOL=(.*)$")
      set(TLLM_BUILD_FLASH_MLA "${CMAKE_MATCH_1}")
    elseif(_line MATCHES "^ENABLE_MULTI_DEVICE:BOOL=(.*)$")
      set(TLLM_ENABLE_MULTI_DEVICE "${CMAKE_MATCH_1}")
    elseif(_line MATCHES "^ENABLE_UCX:BOOL=(.*)$")
      set(TLLM_ENABLE_UCX "${CMAKE_MATCH_1}")
    endif()
  endforeach()
endif()

if(DEFINED ENV{VLLM_TLLM_ENABLE_MULTI_DEVICE})
  set(TLLM_ENABLE_MULTI_DEVICE $ENV{VLLM_TLLM_ENABLE_MULTI_DEVICE})
elseif(DEFINED ENV{VLLM_TENSORRT_LLM_ENABLE_MULTI_DEVICE})
  set(TLLM_ENABLE_MULTI_DEVICE $ENV{VLLM_TENSORRT_LLM_ENABLE_MULTI_DEVICE})
endif()
if(DEFINED ENV{VLLM_TLLM_ENABLE_UCX})
  set(TLLM_ENABLE_UCX $ENV{VLLM_TLLM_ENABLE_UCX})
elseif(DEFINED ENV{VLLM_TENSORRT_LLM_ENABLE_UCX})
  set(TLLM_ENABLE_UCX $ENV{VLLM_TENSORRT_LLM_ENABLE_UCX})
endif()

set(TLLM_CMAKE_ARGS
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DBUILD_PYT=${TLLM_BUILD_PYT}
  -DBUILD_DEEP_EP=${TLLM_BUILD_DEEP_EP}
  -DBUILD_DEEP_GEMM=${TLLM_BUILD_DEEP_GEMM}
  -DBUILD_FLASH_MLA=${TLLM_BUILD_FLASH_MLA}
  -DENABLE_MULTI_DEVICE=${TLLM_ENABLE_MULTI_DEVICE}
  -DENABLE_UCX=${TLLM_ENABLE_UCX}
  -DBUILD_TESTS=OFF
  -DBUILD_BENCHMARKS=OFF
  -DBUILD_MICRO_BENCHMARKS=OFF
  -DNVTX_DISABLE=OFF
)

# Prefer vendored TensorRT if available; allow override via env.
if(DEFINED ENV{VLLM_TENSORRT_ROOT})
  set(VLLM_TENSORRT_ROOT $ENV{VLLM_TENSORRT_ROOT})
endif()
if(DEFINED ENV{TENSORRT_ROOT} AND NOT VLLM_TENSORRT_ROOT)
  set(VLLM_TENSORRT_ROOT $ENV{TENSORRT_ROOT})
endif()
set(_tllm_trt_source "${CMAKE_CURRENT_LIST_DIR}/../../third_party/tensorrt")
set(_tllm_trt_install "${_tllm_trt_source}/install")
if(NOT VLLM_TENSORRT_ROOT)
  if(EXISTS "${_tllm_trt_install}/include")
    set(VLLM_TENSORRT_ROOT "${_tllm_trt_install}")
  else()
    set(VLLM_TENSORRT_ROOT "${_tllm_trt_source}")
  endif()
else()
  # Prefer the install prefix when it exists so libnvonnxparser is discoverable.
  if(VLLM_TENSORRT_ROOT STREQUAL "${_tllm_trt_source}" AND EXISTS "${_tllm_trt_install}/include")
    set(VLLM_TENSORRT_ROOT "${_tllm_trt_install}")
  endif()
endif()
if(EXISTS "${VLLM_TENSORRT_ROOT}/include")
  set(_tllm_prefix_path "${VLLM_TENSORRT_ROOT}")
  # Keep the install prefix in the search path so libs (e.g. nvonnxparser) are found.
  if(EXISTS "${_tllm_trt_install}"
     AND NOT VLLM_TENSORRT_ROOT STREQUAL "${_tllm_trt_install}")
    set(_tllm_prefix_path "${_tllm_prefix_path};${_tllm_trt_install}")
  endif()
  if(DEFINED ENV{CMAKE_PREFIX_PATH} AND NOT "$ENV{CMAKE_PREFIX_PATH}" STREQUAL "")
    set(_tllm_prefix_path "${_tllm_prefix_path};$ENV{CMAKE_PREFIX_PATH}")
  endif()
  list(APPEND TLLM_CMAKE_ARGS
    -DTENSORRT_ROOT=${VLLM_TENSORRT_ROOT}
    -DTensorRT_ROOT=${VLLM_TENSORRT_ROOT}
    -DCMAKE_PREFIX_PATH=${_tllm_prefix_path}
  )
endif()

# TensorRT-LLM does not accept CMAKE_CUDA_ARCHITECTURES=OFF. Use CUDA_ARCHS
# from vLLM when available; otherwise fall back to "native". TensorRT-LLM
# expects integer archs (e.g. 80) instead of dotted (e.g. 8.0).
if(DEFINED CUDA_ARCHS AND NOT CUDA_ARCHS STREQUAL "")
  set(_tllm_archs_in "${CUDA_ARCHS}")
  string(REPLACE "," ";" _tllm_arch_list "${_tllm_archs_in}")
  set(_tllm_arch_out "")
  foreach(_arch ${_tllm_arch_list})
    string(STRIP "${_arch}" _arch)
    if(_arch MATCHES "^[0-9]+\\.[0-9]+$")
      string(REPLACE "." "" _arch "${_arch}")
    endif()
    if(_arch MATCHES "^[0-9]+$")
      list(APPEND _tllm_arch_out "${_arch}")
    endif()
  endforeach()
  if(_tllm_arch_out)
    list(JOIN _tllm_arch_out ";" TLLM_CUDA_ARCHS)
  else()
    set(TLLM_CUDA_ARCHS "native")
  endif()
else()
  set(TLLM_CUDA_ARCHS "native")
endif()
list(APPEND TLLM_CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=${TLLM_CUDA_ARCHS})

set(TLLM_BUILD_TARGETS tensorrt_llm nvinfer_plugin_tensorrt_llm)
if(TLLM_BUILD_PYT STREQUAL "ON")
  list(APPEND TLLM_BUILD_TARGETS th_common bindings pg_utils)
  if(TLLM_BUILD_DEEP_EP STREQUAL "ON")
    list(APPEND TLLM_BUILD_TARGETS deep_ep)
  endif()
  if(TLLM_BUILD_DEEP_GEMM STREQUAL "ON")
    list(APPEND TLLM_BUILD_TARGETS deep_gemm)
  endif()
  if(TLLM_BUILD_FLASH_MLA STREQUAL "ON")
    list(APPEND TLLM_BUILD_TARGETS flash_mla)
  endif()
endif()
if(NOT WIN32)
  list(APPEND TLLM_BUILD_TARGETS executorWorker)
endif()

set(TLLM_BUILD_TARGET_ARGS "")
foreach(_tgt ${TLLM_BUILD_TARGETS})
  list(APPEND TLLM_BUILD_TARGET_ARGS --target ${_tgt})
endforeach()

set(TLLM_DEPENDS "")
if(TARGET tensorrt)
  list(APPEND TLLM_DEPENDS tensorrt)
endif()

ExternalProject_Add(
  tensorrt_llm_build
  SOURCE_DIR ${TLLM_CPP_DIR}
  BINARY_DIR ${TLLM_BUILD_DIR}
  CMAKE_ARGS ${TLLM_CMAKE_ARGS}
  DEPENDS ${TLLM_DEPENDS}
  BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE} ${TLLM_BUILD_TARGET_ARGS}
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS
    ${TLLM_BUILD_DIR}/tensorrt_llm/libtensorrt_llm.so
    ${TLLM_BUILD_DIR}/tensorrt_llm/thop/libth_common.so
    ${TLLM_BUILD_DIR}/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so
)

add_custom_target(tensorrt_llm DEPENDS tensorrt_llm_build)

# Install/copy artifacts into the vllm package.
install(CODE "
  file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm\")
  file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm/libs\")
  file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm/bin\")

  file(GLOB _tllm_bindings \"${TLLM_BUILD_DIR}/tensorrt_llm/bindings*.so\")
  if(_tllm_bindings)
    file(COPY \${_tllm_bindings} DESTINATION \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm\")
  endif()

  if(EXISTS \"${TLLM_BUILD_DIR}/tensorrt_llm/libtensorrt_llm.so\")
    file(COPY \"${TLLM_BUILD_DIR}/tensorrt_llm/libtensorrt_llm.so\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm/libs\")
  endif()
  if(EXISTS \"${TLLM_BUILD_DIR}/tensorrt_llm/thop/libth_common.so\")
    file(COPY \"${TLLM_BUILD_DIR}/tensorrt_llm/thop/libth_common.so\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm/libs\")
  endif()
  if(EXISTS \"${TLLM_BUILD_DIR}/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so\")
    file(COPY \"${TLLM_BUILD_DIR}/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm/libs\")
  endif()
  if(EXISTS \"${TLLM_BUILD_DIR}/tensorrt_llm/runtime/utils/libpg_utils.so\")
    file(COPY \"${TLLM_BUILD_DIR}/tensorrt_llm/runtime/utils/libpg_utils.so\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm/libs\")
  endif()

  file(GLOB _agent_bindings \"${TLLM_BUILD_DIR}/tensorrt_llm/**/tensorrt_llm_transfer_agent_binding*.so\")
  if(_agent_bindings)
    file(COPY \${_agent_bindings} DESTINATION \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm\")
  endif()

  if(EXISTS \"${TLLM_BUILD_DIR}/tensorrt_llm/executor_worker/executorWorker\")
    file(COPY \"${TLLM_BUILD_DIR}/tensorrt_llm/executor_worker/executorWorker\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm/bin\")
  endif()

  if(EXISTS \"${TLLM_BUILD_DIR}/tensorrt_llm/runtime/kv_cache_manager_v2\")
    file(COPY \"${TLLM_BUILD_DIR}/tensorrt_llm/runtime/kv_cache_manager_v2\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm/runtime\")
  endif()

  if(EXISTS \"${TLLM_BUILD_DIR}/tensorrt_llm/deep_ep/python/tensorrt_llm/deep_ep\")
    file(COPY \"${TLLM_BUILD_DIR}/tensorrt_llm/deep_ep/python/tensorrt_llm/deep_ep\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm\")
  endif()
  if(EXISTS \"${TLLM_BUILD_DIR}/tensorrt_llm/deep_gemm/python/tensorrt_llm/deep_gemm\")
    file(COPY \"${TLLM_BUILD_DIR}/tensorrt_llm/deep_gemm/python/tensorrt_llm/deep_gemm\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm\")
  endif()
  if(EXISTS \"${TLLM_BUILD_DIR}/tensorrt_llm/flash_mla/python/tensorrt_llm/flash_mla\")
    file(COPY \"${TLLM_BUILD_DIR}/tensorrt_llm/flash_mla/python/tensorrt_llm/flash_mla\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/tensorrt_llm\")
  endif()
" COMPONENT tensorrt_llm)
