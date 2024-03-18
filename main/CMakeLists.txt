# CMakeLists.txt in the main directory
add_executable(simple simple.cpp)
add_executable(simple_cuda simple_cuda.cu)
add_executable(simple_cuda_graph simple_cuda_graph.cu)

# Link fsim libraries
target_link_libraries(simple PRIVATE fsim)
target_link_libraries(simple_cuda PRIVATE fsim_cuda)
target_link_libraries(simple_cuda_graph PRIVATE fsim_cuda_graph)

# Link cuBLAS library
target_link_libraries(simple_cuda_graph PRIVATE ${CUBLAS_LIBRARIES})

# Include cuBLAS include directories
target_include_directories(simple_cuda_graph PRIVATE ${CUBLAS_INCLUDE_DIRS})

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
  -g
  -G
)


# Set compiler flags for the "main" target
target_compile_options(simple_cuda_graph PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
)
