# CMakeLists.txt in the fsim directory
# Fault simulation - CPU version
add_library(fsim 
  fault.cpp fault.hpp
  pattern.cpp pattern.hpp
  base_ppg.cpp base_ppg.hpp
  pi.cpp pi.hpp
  po.cpp po.hpp
  gate.cpp gate.hpp
  simulation.cpp simulation.hpp
)

# Fault simulation - GPU version
add_library(fsim_cuda 
  fault.cpp fault.hpp
  pattern.cpp pattern.hpp
  base_ppg.cpp base_ppg.hpp
  pi.cpp pi.hpp
  po.cpp po.hpp
  gate.cpp gate.hpp
  simulation.cpp simulation.hpp
  ./cuda/simulation.cu ./cuda/simulation.cuh
)

# GPU - CUDA Graph Simulator 
add_library(fsim_cuda_graph 
  ./cuda_graph/simulation.cu ./cuda_graph/simulation.cuh
)

# CPU and GPU paritioner 
add_library(fsim_cuda_partition
  ./cuda_partition/cpu_partition.cu  ./cuda_partition/cpu_partition.cuh
  ./cuda_partition/cuda_partition.cu ./cuda_partition/cuda_partition.cuh
  # CPU: 
  fault.cpp fault.hpp
  pattern.cpp pattern.hpp
  base_ppg.cpp base_ppg.hpp
  pi.cpp pi.hpp
  po.cpp po.hpp
  gate.cpp gate.hpp
  simulation.cpp simulation.hpp
  ./cpu_simulator/cpu_simulation.cuh 
  ./cuda_simulator/gpu_int_simulation.cu
  ./cuda_simulator/gpu_int_simulation.cuh  
  ./cuda_simulator/gpu_sizet_simulation.cu
  ./cuda_simulator/gpu_sizet_simulation.cuh
)

# Simulation:
add_library(galps_simulator
  ./cpu_simulator/cpu_simulation.cuh 
  ./cuda_simulator/gpu_int_simulation.cuh 
  ./cuda_simulator/gpu_int_simulation.cu
  ./cuda_simulator/gpu_sizet_simulation.cu
  ./cuda_simulator/gpu_sizet_simulation.cuh  
)

add_library(fsim_MA_cuda_partition
  ./cuda_MA_partition/cuda_MA_partition.cu ./cuda_MA_partition/cuda_MA_partition.cuh
  ./cuda_MA_partition/cuda_MA_simulation/gpu_simulation.cu ./cuda_MA_partition/cuda_MA_simulation/gpu_simulation.cuh
  # CPU: 
  fault.cpp fault.hpp
  pattern.cpp pattern.hpp
  base_ppg.cpp base_ppg.hpp
  pi.cpp pi.hpp
  po.cpp po.hpp
  gate.cpp gate.hpp
  simulation.cpp simulation.hpp
)



# -----------------------------------------------------------------------------
# Link OpenMP for CUDA partition (needed for OpenMP support in CUDA code)
# -----------------------------------------------------------------------------
target_link_libraries(fsim_cuda_partition PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(galps_simulator PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(fsim_MA_cuda_partition PRIVATE OpenMP::OpenMP_CXX)


# -----------------------------------------------------------------------------
# Add cuBLAS library if necessary
# -----------------------------------------------------------------------------
if (FSIM_BUILD_CUDA)
  target_link_libraries(${PROJECT_NAME} INTERFACE ${CUBLAS_LIBRARIES})
  target_include_directories(${PROJECT_NAME} INTERFACE ${CUBLAS_INCLUDE_DIRS})
endif()


# -----------------------------------------------------------------------------
# KAHYPAR_LIB 
# -----------------------------------------------------------------------------
find_library(KAHYPAR_LIB kahypar PATHS /usr/local/lib)

# Add include directories for libkahypar headers
include_directories(/usr/local/include)

# Link the library with the target fsim_cuda_partition
target_link_libraries(fsim_cuda_partition PRIVATE ${KAHYPAR_LIB})
message(STATUS "KAHYPAR_LIB: ${KAHYPAR_LIB}")

# Link the library with the target fsim_MA_cuda_partition
target_link_libraries(fsim_MA_cuda_partition PRIVATE ${KAHYPAR_LIB})
message(STATUS "KAHYPAR_LIB: ${KAHYPAR_LIB}")


# -----------------------------------------------------------------------------
# MT-KAHYPAR_LIB 
# -----------------------------------------------------------------------------
find_library(MT_KAHYPAR_LIB mtkahypar PATHS /usr/local/lib)

# Add include directories for libmtkahypar headers
include_directories(/usr/local/include)

# Link the library with the target fsim_cuda_partition
target_link_libraries(fsim_cuda_partition PRIVATE ${MT_KAHYPAR_LIB})
message(STATUS "MT_KAHYPAR_LIB: ${MT_KAHYPAR_LIB}")

# Link the library with the target fsim_MA_cuda_partition
target_link_libraries(fsim_MA_cuda_partition PRIVATE ${MT_KAHYPAR_LIB})
message(STATUS "MT_KAHYPAR_LIB: ${MT_KAHYPAR_LIB}")




# Set compiler flags for the "fsim" library
target_compile_options(fsim PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -O3  
)

target_compile_options(fsim_cuda PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -lineinfo
  -O3  
)

target_compile_options(fsim_cuda_graph PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -O3  
)

target_compile_options(fsim_cuda_partition PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -lineinfo
  -O3
  -DNDEBUG
)

target_compile_options(galps_simulator PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -O3  
)

target_compile_options(fsim_MA_cuda_partition PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -lineinfo
  -O3
  -DNDEBUG
)