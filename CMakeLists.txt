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

cmake_minimum_required(VERSION 3.20)

project(
  fsim
  VERSION 1.0
  LANGUAGES CXX
)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# include the default directory
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(fsim)
add_subdirectory(main)
