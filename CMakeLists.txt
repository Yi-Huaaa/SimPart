cmake_minimum_required(VERSION 3.24)

project(SimPart
  VERSION 1.0.0
  DESCRIPTION "Replication-aided partitioning for GPU logic simulation"
  LANGUAGES CXX CUDA)

option(SIMPART_BUILD_REPCUT "Build the RepCut baseline (requires Mt-KaHyPar)" ON)
option(SIMPART_BUILD_TESTS "Build the unit tests" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Conditional CUDA graph nodes require CUDA 12.3 or newer.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.3)
  message(FATAL_ERROR "SimPart requires CUDA 12.3 or newer (found ${CMAKE_CUDA_COMPILER_VERSION})")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Mt-KaHyPar (C interface) for the RepCut baseline.
set(SIMPART_WITH_REPCUT OFF)
if(SIMPART_BUILD_REPCUT)
  find_path(MTKAHYPAR_INCLUDE_DIR libmtkahypar.h)
  find_library(MTKAHYPAR_LIBRARY mtkahypar)
  if(MTKAHYPAR_INCLUDE_DIR AND MTKAHYPAR_LIBRARY)
    set(SIMPART_WITH_REPCUT ON)
    message(STATUS "Mt-KaHyPar: ${MTKAHYPAR_LIBRARY}")
  else()
    message(WARNING "Mt-KaHyPar not found: building without the RepCut baseline "
                    "(set CMAKE_PREFIX_PATH or -DSIMPART_BUILD_REPCUT=OFF)")
  endif()
endif()

add_subdirectory(simpart)
add_subdirectory(app)

if(SIMPART_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
