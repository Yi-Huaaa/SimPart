# CMakeLists.txt in the main directory
add_executable(simple simple.cpp)
add_executable(simple_cuda simple_cuda.cu)
target_link_libraries(simple PRIVATE fsim)
target_link_libraries(simple_cuda PRIVATE fsim_cuda)

# Set compiler flags for the "main" target
target_compile_options(simple PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
)

# Set compiler flags for the "main" target
target_compile_options(simple_cuda PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
)
