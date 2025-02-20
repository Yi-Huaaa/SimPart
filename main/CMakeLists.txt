# CMakeLists.txt in the main directory
add_executable(simple simple.cpp)
add_executable(simple_cuda simple_cuda.cu)
add_executable(simple_cuda_graph simple_cuda_graph.cu)
add_executable(simple_cuda_partition simple_partition.cu)
add_executable(MA_cuda_partition MA_partition.cu)

# Link fsim libraries
target_link_libraries(simple PRIVATE fsim)
target_link_libraries(simple_cuda PRIVATE fsim_cuda)
target_link_libraries(simple_cuda_graph PRIVATE fsim_cuda_graph)
target_link_libraries(simple_cuda_partition PRIVATE fsim_cuda_partition)
target_link_libraries(MA_cuda_partition PRIVATE fsim_MA_cuda_partition)

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
  -O3
)

# Set compiler flags for the "main" target
target_compile_options(simple_cuda PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -lineinfo
)


# Set compiler flags for the "main" target
target_compile_options(simple_cuda_graph PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -O3
)


# Set compiler flags for the "main" target
target_compile_options(simple_cuda_partition PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -lineinfo
  -O3
)

# Set compiler flags for the "main" target
target_compile_options(MA_cuda_partition PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -lineinfo
  -O3
)
