# -------------------------------------------------------------------
# fsim CMake 
#
# To enable thread sanitizer: 
# -DCMAKE_CXX_FLAGS="-fsanitize=thread -g"
#
# To enable address and leak sanitizers
# -DCMAKE_CXX_FLAGS="-fsanitize=address  -fsanitize=leak -g"
#
# To enable different c++ standards
# -DCMAKE_CXX_STANDARD=17
# -DCMAKE_CXX_STANDARD=20
# -DCMAKE_CXX_STANDARD=23
#
# To switch to different build types:
# -DCMAKE_BUILD_TYPE=DEBUG
# -DCMAKE_BUILD_TYPE=RELEASE
# -------------------------------------------------------------------

# CMake version
cmake_minimum_required(VERSION 3.20)

# Project name
project(
  fsim
  VERSION 1.0
  LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Add OpenMP
# 20240126 add, ref: https://stackoverflow.com/questions/49937059/findopenmp-requires-c-or-cxx-language-to-be-enabled-error-on-ubuntu-16-04-and-c
# -----------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# build options
option(FSIM_BUILD_TESTS "Enables builds of tests" ON)
option(FSIM_BUILD_CUDA "Enables builds of cuda code" ON)

# project-specific variables
set(FSIM_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)
message(STATUS "FSIM_3RD_PARTY_DIR: " ${FSIM_3RD_PARTY_DIR})
message(STATUS "CMAKE_CUDA_COMPILER: " ${CMAKE_CUDA_COMPILER})
message(STATUS "CMAKE_CUDA_COMPILER_ID: " ${CMAKE_CUDA_COMPILER_ID})
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: " ${CMAKE_CUDA_COMPILER_VERSION})
message(STATUS "CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})
message(STATUS "FSIM_BUILD_CUDA: " ${FSIM_BUILD_CUDA})
set(FSIM_CMAKE_DIR ${PROJECT_SOURCE_DIR}/cmake)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)


# -----------------------------------------------------------------------------
# find nvcc
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
# -----------------------------------------------------------------------------
if(FSIM_BUILD_CUDA)
  message(STATUS "Configuring CUDA ...")
  enable_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "\nNo CUDA compiler found")
  endif()
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11") # need 12.2 (or higher)
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(FATAL_ERROR "\nfsim requires CUDA at least v12 (Driver: version 550 or higher)")
  endif()

  # configure cuBLAS
  message(STATUS "Configuring CUDA BLAS ...")
  list(APPEND CMAKE_MODULE_PATH ${FSIM_CMAKE_DIR}/cuda_modules/)
  find_package(cuBLAS)
  message(STATUS "CUBLAS_FOUND: ${CUBLAS_FOUND}")
  message(STATUS "CUBLAS_INCLUDE_DIRS: ${CUBLAS_INCLUDE_DIRS}")
  message(STATUS "CUBLAS_LIBRARIES: ${CUBLAS_LIBRARIES}")

endif()

# # Add cuBLAS library if necessary
# if (FSIM_BUILD_CUDA)
#     target_link_libraries(${PROJECT_NAME} INTERFACE ${CUBLAS_LIBRARIES})
#     target_include_directories(${PROJECT_NAME} INTERFACE ${CUBLAS_INCLUDE_DIRS})
# endif()

# # Add other settings
# add_library(${PROJECT_NAME} INTERFACE)
# target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
# target_link_libraries(${PROJECT_NAME} INTERFACE Threads::Threads)
# target_include_directories(${PROJECT_NAME} INTERFACE
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#   $<INSTALL_INTERFACE:include/> 
# )



# include the default directory
include_directories(${PROJECT_SOURCE_DIR})

add_library(default_settings INTERFACE)
add_library(fsim::default_settings ALIAS default_settings)


# -----------------------------------------------------------------------------
# fsim::default_setting
# -----------------------------------------------------------------------------

# error setting
add_library(error_settings INTERFACE)

# cuda compiler options
if(FSIM_BUILD_CUDA)
  message(STATUS "Configuring CUDA compiler options ...")
  target_compile_options(
    error_settings
    BEFORE
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-Wall,-Wextra,-Wfatal-errors,-Wshadow>
  )
endif(FSIM_BUILD_CUDA)


# -----------------------------------------------------------------------------
# start build
# -----------------------------------------------------------------------------
add_subdirectory(fsim)
add_subdirectory(main)


# -----------------------------------------------------------------------------
# Unittest
# -----------------------------------------------------------------------------
if(FSIM_BUILD_TESTS)
  add_subdirectory(unittests)
endif(FSIM_BUILD_TESTS)


# -----------------------------------------------------------------------------
# make test
# instruction for running the unittests automatically 
# -----------------------------------------------------------------------------
# Enable test
include(CTest)

