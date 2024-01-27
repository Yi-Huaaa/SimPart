# CMakeLists.txt in the fsim directory
add_library(fsim 
  fault.cpp fault.hpp
  pattern.cpp pattern.hpp
  gate.cpp gate.hpp
  simulation.cpp simulation.hpp
)

# Set compiler flags for the "fsim" library
target_compile_options(fsim PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
)
