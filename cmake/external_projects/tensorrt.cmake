# Build vendored TensorRT from source to provide headers/libs for TensorRT-LLM.

include(ExternalProject)

if(DEFINED ENV{VLLM_TENSORRT_SRC_DIR})
  set(VLLM_TENSORRT_SRC_DIR $ENV{VLLM_TENSORRT_SRC_DIR})
endif()

if(NOT VLLM_TENSORRT_SRC_DIR)
  set(VLLM_TENSORRT_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/../../third_party/tensorrt")
endif()

if(NOT EXISTS "${VLLM_TENSORRT_SRC_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "TensorRT source not found at ${VLLM_TENSORRT_SRC_DIR}. "
    "Set VLLM_TENSORRT_SRC_DIR to the TensorRT source tree.")
endif()

set(TENSORRT_INSTALL_DIR "${VLLM_TENSORRT_SRC_DIR}/install")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(TENSORRT_BUILD_DIR "${VLLM_TENSORRT_SRC_DIR}/build")
else()
  set(TENSORRT_BUILD_DIR "${VLLM_TENSORRT_SRC_DIR}/build_${CMAKE_BUILD_TYPE}")
endif()

set(TENSORRT_CMAKE_ARGS
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX=${TENSORRT_INSTALL_DIR}
  -DBUILD_PLUGINS=ON
  -DBUILD_PARSERS=ON
  -DBUILD_SAMPLES=OFF
  -DBUILD_SAFE_SAMPLES=OFF
  -DTRT_SAFETY_INFERENCE_ONLY=OFF
)

# TensorRT OSS needs the proprietary nvinfer libs present. Look for a bundled
# binary distribution under the vendored TensorRT root and pass TRT_LIB_DIR.
set(_tensorrt_lib_candidates
  "${VLLM_TENSORRT_SRC_DIR}/lib"
  "${VLLM_TENSORRT_SRC_DIR}/lib64"
  "${VLLM_TENSORRT_SRC_DIR}/targets/x86_64-linux-gnu/lib"
  "${VLLM_TENSORRT_SRC_DIR}/install/lib"
)
set(TENSORRT_LIB_DIR "")
foreach(_cand ${_tensorrt_lib_candidates})
  if(EXISTS "${_cand}/libnvinfer.so" OR EXISTS "${_cand}/libnvinfer.so.10")
    set(TENSORRT_LIB_DIR "${_cand}")
    break()
  endif()
endforeach()
if(NOT TENSORRT_LIB_DIR)
  message(FATAL_ERROR
    "TensorRT OSS build requires prebuilt libnvinfer.so. "
    "Please unpack the NVIDIA TensorRT binary distribution into "
    "${VLLM_TENSORRT_SRC_DIR} so libnvinfer.so is under lib/ or lib64/.")
endif()
list(APPEND TENSORRT_CMAKE_ARGS -DTRT_LIB_DIR=${TENSORRT_LIB_DIR})

# Align TensorRT GPU_ARCHS with vLLM CUDA_ARCHS when provided.
if(DEFINED CUDA_ARCHS AND NOT CUDA_ARCHS STREQUAL "")
  set(_trt_archs_in "${CUDA_ARCHS}")
  string(REPLACE "," ";" _trt_arch_list "${_trt_archs_in}")
  set(_trt_arch_out "")
  foreach(_arch ${_trt_arch_list})
    string(STRIP "${_arch}" _arch)
    if(_arch MATCHES "^[0-9]+\\.[0-9]+$")
      string(REPLACE "." "" _arch "${_arch}")
    endif()
    if(_arch MATCHES "^[0-9]+$")
      list(APPEND _trt_arch_out "${_arch}")
    endif()
  endforeach()
  if(_trt_arch_out)
    list(JOIN _trt_arch_out ";" TENSORRT_GPU_ARCHS)
    list(APPEND TENSORRT_CMAKE_ARGS -DGPU_ARCHS=${TENSORRT_GPU_ARCHS})
  endif()
endif()

ExternalProject_Add(
  tensorrt_build
  SOURCE_DIR ${VLLM_TENSORRT_SRC_DIR}
  BINARY_DIR ${TENSORRT_BUILD_DIR}
  CMAKE_ARGS ${TENSORRT_CMAKE_ARGS}
  BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE}
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install --config ${CMAKE_BUILD_TYPE}
  BUILD_BYPRODUCTS
    ${TENSORRT_INSTALL_DIR}/lib/libnvinfer.so
    ${TENSORRT_INSTALL_DIR}/lib/libnvonnxparser.so
    ${TENSORRT_INSTALL_DIR}/lib/libnvinfer_plugin.so
)

add_custom_target(tensorrt DEPENDS tensorrt_build)
