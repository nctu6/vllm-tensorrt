# Build TurboMind (vendored from lmdeploy) for CUDA builds.

# Allow overriding the TurboMind source directory.
if(DEFINED ENV{VLLM_TURBOMIND_SRC_DIR})
  set(VLLM_TURBOMIND_SRC_DIR $ENV{VLLM_TURBOMIND_SRC_DIR})
endif()

if(NOT VLLM_TURBOMIND_SRC_DIR)
  set(VLLM_TURBOMIND_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/../../third_party/turbomind")
endif()

if(NOT EXISTS "${VLLM_TURBOMIND_SRC_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "TurboMind source not found at ${VLLM_TURBOMIND_SRC_DIR}. "
    "Set VLLM_TURBOMIND_SRC_DIR to the vendored lmdeploy source tree.")
endif()

# Configure TurboMind build options.
set(BUILD_PY_FFI ON CACHE BOOL "Build TurboMind Python FFI" FORCE)
set(BUILD_TEST OFF CACHE BOOL "Disable TurboMind tests" FORCE)
set(CALL_FROM_SETUP_PY ON CACHE BOOL "Install TurboMind into the CMake prefix" FORCE)
set(BUILD_MULTI_GPU ON CACHE BOOL "Enable TurboMind multi-GPU support" FORCE)
set(SPARSITY_SUPPORT OFF CACHE BOOL "Disable TurboMind sparsity support" FORCE)
set(BUILD_FAST_MATH ON CACHE BOOL "Enable TurboMind fast math" FORCE)

add_subdirectory(${VLLM_TURBOMIND_SRC_DIR} ${CMAKE_BINARY_DIR}/turbomind)
