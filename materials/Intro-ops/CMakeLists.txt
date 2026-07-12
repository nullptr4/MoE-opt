cmake_minimum_required(VERSION 3.22)

project(operator_runtime_training LANGUAGES CXX)

option(CAMP_ENABLE_NVIDIA "Build NVIDIA CUDA backend" ON)
option(CAMP_ENABLE_TILELANG "Enable TileLang Python backend metadata" ON)
option(CAMP_ENABLE_METAX "Build MetaX backend" OFF)
set(CAMP_ENABLE_CUTE "AUTO" CACHE STRING "Enable optional CuTe/CUTLASS headers for NVIDIA custom operators: AUTO, ON, or OFF")
set_property(CACHE CAMP_ENABLE_CUTE PROPERTY STRINGS AUTO ON OFF)

set(CAMP_CUTLASS_ROOT "" CACHE PATH "Optional CUTLASS checkout root for CuTe-based custom NVIDIA operators")
set(CAMP_CUTE_INCLUDE_DIRS "" CACHE STRING "Optional semicolon-separated CuTe/CUTLASS include directories")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(CAMP_ENABLE_NVIDIA AND CAMP_ENABLE_METAX)
  message(FATAL_ERROR "CAMP_ENABLE_NVIDIA and CAMP_ENABLE_METAX cannot both be ON; build separate variants")
endif()

string(TOUPPER "${CAMP_ENABLE_CUTE}" CAMP_ENABLE_CUTE_MODE)
if(NOT CAMP_ENABLE_CUTE_MODE MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "CAMP_ENABLE_CUTE must be AUTO, ON, or OFF")
endif()

function(camp_append_cute_root root)
  if(NOT root)
    return()
  endif()
  list(APPEND CAMP_CUTE_CANDIDATE_INCLUDE_DIRS
    "${root}"
    "${root}/include"
    "${root}/tools/util/include"
  )
  set(CAMP_CUTE_CANDIDATE_INCLUDE_DIRS "${CAMP_CUTE_CANDIDATE_INCLUDE_DIRS}" PARENT_SCOPE)
endfunction()

set(CAMP_CUTE_TARGET_INCLUDE_DIRS "")
set(CAMP_USE_CUTE OFF)
if(NOT CAMP_ENABLE_CUTE_MODE STREQUAL "OFF")
  if(NOT CAMP_ENABLE_NVIDIA)
    if(CAMP_ENABLE_CUTE_MODE STREQUAL "ON")
      message(FATAL_ERROR "CAMP_ENABLE_CUTE=ON requires CAMP_ENABLE_NVIDIA=ON")
    endif()
  else()
    set(CAMP_CUTE_CANDIDATE_INCLUDE_DIRS "")

    camp_append_cute_root("${CAMP_CUTLASS_ROOT}")
    camp_append_cute_root("${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")
    foreach(_camp_cute_env_var CUTLASS_ROOT CUTLASS_HOME CUTLASS_PATH)
      if(DEFINED ENV{${_camp_cute_env_var}} AND NOT "$ENV{${_camp_cute_env_var}}" STREQUAL "")
        camp_append_cute_root("$ENV{${_camp_cute_env_var}}")
      endif()
    endforeach()

    if(CAMP_CUTE_INCLUDE_DIRS)
      list(APPEND CAMP_CUTE_CANDIDATE_INCLUDE_DIRS ${CAMP_CUTE_INCLUDE_DIRS})
    endif()

    if(CAMP_CUTE_CANDIDATE_INCLUDE_DIRS)
      list(REMOVE_DUPLICATES CAMP_CUTE_CANDIDATE_INCLUDE_DIRS)
    endif()

    foreach(_camp_cute_include_dir IN LISTS CAMP_CUTE_CANDIDATE_INCLUDE_DIRS)
      if(EXISTS "${_camp_cute_include_dir}/cute/tensor.hpp")
        list(APPEND CAMP_CUTE_TARGET_INCLUDE_DIRS "${_camp_cute_include_dir}")
      endif()
    endforeach()

    if(CAMP_CUTE_TARGET_INCLUDE_DIRS)
      list(REMOVE_DUPLICATES CAMP_CUTE_TARGET_INCLUDE_DIRS)
      set(CAMP_USE_CUTE ON)
      message(STATUS "CuTe/CUTLASS support enabled: ${CAMP_CUTE_TARGET_INCLUDE_DIRS}")
    elseif(CAMP_ENABLE_CUTE_MODE STREQUAL "ON")
      message(FATAL_ERROR "CAMP_ENABLE_CUTE=ON requires cute/tensor.hpp via CAMP_CUTLASS_ROOT, CAMP_CUTE_INCLUDE_DIRS, or CUTLASS_ROOT/CUTLASS_HOME/CUTLASS_PATH")
    else()
      message(STATUS "CuTe/CUTLASS support disabled: cute/tensor.hpp was not found")
    endif()
  endif()
endif()

if(CAMP_ENABLE_METAX)
  find_package(MetaX REQUIRED)
  set(ENV{CUCC_PATH} "${MetaX_CUBRIDGE_ROOT}")
  set(ENV{CUDA_PATH} "${MetaX_CUBRIDGE_ROOT}")
  list(APPEND CMAKE_MODULE_PATH "${MetaX_CMAKE_MODULE_DIR}")
  set(CMAKE_MACA_SOURCE_FILE_EXTENSIONS maca;cu;cc;cpp;c)
endif()

if(CAMP_ENABLE_NVIDIA OR CAMP_ENABLE_METAX)
  if(CAMP_ENABLE_METAX)
    enable_language(MACA)
  else()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    include(cmake/cuda_helpers.cmake)
  endif()
endif()

if(CAMP_ENABLE_METAX)
  set(CAMP_METAX_COMPILE_OPTIONS
    -D__MACA_NO_HALF_OPERATORS__
    -D__FAST_BLOCK_RED__
  )
endif()

add_subdirectory(ops)
