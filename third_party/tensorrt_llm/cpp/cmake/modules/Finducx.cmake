#
# Minimal UCX finder for TensorRT-LLM/NIXL.
#
# Uses pkg-config (preferred) and common UCX roots.
#

include(FindPackageHandleStandardArgs)
find_package(PkgConfig REQUIRED)

set(_UCX_ROOT_HINTS "")
if(DEFINED UCX_ROOT)
  list(APPEND _UCX_ROOT_HINTS "${UCX_ROOT}")
endif()
if(DEFINED ENV{UCX_ROOT})
  list(APPEND _UCX_ROOT_HINTS "$ENV{UCX_ROOT}")
endif()
list(APPEND _UCX_ROOT_HINTS "/usr/local/ucx" "/usr")

set(_UCX_PKG_PATH "")
foreach(_root IN LISTS _UCX_ROOT_HINTS)
  foreach(_libdir IN ITEMS lib lib64 "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
    if(EXISTS "${_root}/${_libdir}/pkgconfig/ucx.pc")
      list(APPEND _UCX_PKG_PATH "${_root}/${_libdir}/pkgconfig")
    endif()
  endforeach()
endforeach()

if(_UCX_PKG_PATH)
  list(REMOVE_DUPLICATES _UCX_PKG_PATH)
  string(JOIN ":" _ucx_pkg_path ${_UCX_PKG_PATH})
  if(DEFINED ENV{PKG_CONFIG_PATH} AND NOT "$ENV{PKG_CONFIG_PATH}" STREQUAL "")
    set(ENV{PKG_CONFIG_PATH} "${_ucx_pkg_path}:$ENV{PKG_CONFIG_PATH}")
  else()
    set(ENV{PKG_CONFIG_PATH} "${_ucx_pkg_path}")
  endif()
endif()

pkg_check_modules(UCX REQUIRED ucx)

set(ucx_FOUND "${UCX_FOUND}")
set(ucx_INCLUDE_DIRS "${UCX_INCLUDE_DIRS}")
set(ucx_LIBRARIES "${UCX_LIBRARIES}")

if(ucx_FOUND AND NOT TARGET ucx::ucx)
  add_library(ucx::ucx INTERFACE IMPORTED)
  set_target_properties(
    ucx::ucx
    PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${ucx_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${ucx_LIBRARIES}"
  )
endif()

foreach(_ucx_component IN ITEMS ucp ucs uct ucm)
  if(ucx_FOUND AND NOT TARGET "ucx::${_ucx_component}")
    add_library("ucx::${_ucx_component}" INTERFACE IMPORTED)
    set_target_properties(
      "ucx::${_ucx_component}"
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ucx_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${ucx_LIBRARIES}"
    )
  endif()
endforeach()

find_package_handle_standard_args(
  ucx
  FOUND_VAR ucx_FOUND
  REQUIRED_VARS ucx_INCLUDE_DIRS ucx_LIBRARIES
)
