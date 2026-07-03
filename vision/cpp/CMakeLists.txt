cmake_minimum_required(VERSION 3.16)
project(wproc_kerf_inspect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Core image-processing library (no external dependencies).
add_library(wproc STATIC
  src/io.cpp
  src/filters.cpp
  src/threshold.cpp
  src/morphology.cpp
  src/connected.cpp
  src/draw.cpp
  src/kerf_inspect.cpp
)
target_include_directories(wproc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Applications.
add_executable(kerf_inspect apps/kerf_inspect_main.cpp)
target_link_libraries(kerf_inspect PRIVATE wproc)

add_executable(gen_synthetic apps/gen_synthetic.cpp)
target_link_libraries(gen_synthetic PRIVATE wproc)

# Tests.
enable_testing()
add_executable(test_imgproc tests/test_imgproc.cpp)
target_link_libraries(test_imgproc PRIVATE wproc)
add_test(NAME imgproc COMMAND test_imgproc)
