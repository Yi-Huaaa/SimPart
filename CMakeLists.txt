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

# build options
option(FSIM_BUILD_TESTS "Enables builds of tests" ON)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# include the default directory
include_directories(${PROJECT_SOURCE_DIR})

add_library(default_settings INTERFACE)
add_library(fsim::default_settings ALIAS default_settings)

# -----------------------------------------------------------------------------
# start build
# -----------------------------------------------------------------------------


# project-specific variables
set(FSIM_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)
message(STATUS "FSIM_3RD_PARTY_DIR: " ${FSIM_3RD_PARTY_DIR})


add_subdirectory(fsim)
add_subdirectory(main)

# -----------------------------------------------------------------------------
# Unittest
# -----------------------------------------------------------------------------
if(FSIM_BUILD_TESTS)
  add_subdirectory(unittests)
endif(FSIM_BUILD_TESTS)

