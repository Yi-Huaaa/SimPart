add_library(simpart
  circuit.cpp
  patterns.cpp
  levelize.cpp
  partition.cpp
  cpu_simulator.cpp
  simpart.cu
  gpu_simulator.cu)

target_include_directories(simpart PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(simpart PUBLIC CUDA::cudart OpenMP::OpenMP_CXX)
target_compile_options(simpart PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wshadow>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wshadow>)

if(SIMPART_WITH_REPCUT)
  target_sources(simpart PRIVATE repcut.cpp)
  target_include_directories(simpart PRIVATE ${MTKAHYPAR_INCLUDE_DIR})
  target_link_libraries(simpart PRIVATE ${MTKAHYPAR_LIBRARY})
  target_compile_definitions(simpart PUBLIC SIMPART_WITH_REPCUT)
endif()
