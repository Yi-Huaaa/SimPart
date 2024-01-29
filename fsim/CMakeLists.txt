# CMakeLists.txt in the fsim directory
add_library(fsim 
  fault.cpp fault.hpp
  pattern.cpp pattern.hpp
  base_ppg.cpp base_ppg.hpp
  pi.cpp pi.hpp
  po.cpp po.hpp
  gate.cpp gate.hpp
  simulation.cpp simulation.hpp
)

# Check if the -Wall flag is present in the compiler flags
if(NOT "${CMAKE_CXX_FLAGS}" MATCHES "-Wall")
  message(WARNING "Warning: Compiler flags do not include '-Wall'")
endif()

# Set compiler flags for the "fsim" library
target_compile_options(fsim PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
)

