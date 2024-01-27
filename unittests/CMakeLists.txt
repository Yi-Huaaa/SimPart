enable_testing()

include(${FSIM_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND FSIM_UNITTESTS 
  test_basics
)

# Add unittests for fsim 
foreach(unittest IN LISTS FSIM_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  
  # Set compiler flags for the "unittests" target
  target_compile_options(${unittest} PRIVATE
    -Wall
    -Wextra
    -Wfatal-errors
    -Wshadow
  )

  target_link_libraries(${unittest} ${PROJECT_NAME} fsim::default_settings)
  target_include_directories(${unittest} PRIVATE ${FSIM_3RD_PARTY_DIR}/doctest)
  doctest_discover_tests(${unittest})

endforeach()


# add_test(NAME ${unittest} COMMAND ${unittest})

# Add the test to CTest
# Add the unit tests to the test target
foreach(unittest IN LISTS FSIM_UNITTESTS)
  add_test(NAME ${unittest} COMMAND ${unittest})
endforeach()