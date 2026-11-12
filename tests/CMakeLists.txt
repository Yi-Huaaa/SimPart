add_executable(simpart-tests test_simpart.cpp)
target_include_directories(simpart-tests PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party)
target_link_libraries(simpart-tests PRIVATE simpart)
target_compile_definitions(simpart-tests PRIVATE
  SIMPART_BENCHMARK_DIR="${PROJECT_SOURCE_DIR}/benchmark")
add_test(NAME simpart-tests COMMAND simpart-tests)
