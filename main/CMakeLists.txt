# CMakeLists.txt in the main directory
add_executable(simple simple.cpp)
target_link_libraries(simple PRIVATE fsim)

# Set compiler flags for the "main" target
target_compile_options(simple PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
)
