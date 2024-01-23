enable_testing()

include(${FSIM_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND FSIM_UNITTESTS 
  test_basics
)

# string(FIND '${CMAKE_CXX_FLAGS}' "-fsanitize" sanitize)
# if(${sanitize} EQUAL -1)
#   message(STATUS "Add test_exception to unit tests")
#   list(APPEND FSIM_UNITTESTS test_exceptions)
# endif()


foreach(unittest IN LISTS FSIM_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  target_link_libraries(${unittest} ${PROJECT_NAME} fsim::default_settings)
  target_include_directories(${unittest} PRIVATE ${FSIM_3RD_PARTY_DIR}/doctest)
  doctest_discover_tests(${unittest})
endforeach()
