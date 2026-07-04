cmake_minimum_required(VERSION 3.16)
project(optomech_lens_sim LANGUAGES CXX)

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

# Optomechanical simulation library (no external dependencies).
add_library(optm STATIC
  src/optics.cpp
  src/tolerance.cpp
  src/cam.cpp
  src/fea.cpp
  src/ois.cpp
  src/ppm.cpp
  src/talloc.cpp
  src/aa.cpp
  src/housing.cpp
  src/athermal.cpp
)
target_include_directories(optm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Demo application (emits JSON metrics + figures).
add_executable(optm_demo apps/optm_demo.cpp)
target_link_libraries(optm_demo PRIVATE optm)

# Tests.
enable_testing()
add_executable(test_tolerance tests/test_tolerance.cpp)
target_link_libraries(test_tolerance PRIVATE optm)
add_test(NAME tolerance COMMAND test_tolerance)

add_executable(test_cam tests/test_cam.cpp)
target_link_libraries(test_cam PRIVATE optm)
add_test(NAME cam COMMAND test_cam)

add_executable(test_fea tests/test_fea.cpp)
target_link_libraries(test_fea PRIVATE optm)
add_test(NAME fea COMMAND test_fea)

add_executable(test_ois tests/test_ois.cpp)
target_link_libraries(test_ois PRIVATE optm)
add_test(NAME ois COMMAND test_ois)

add_executable(test_talloc tests/test_talloc.cpp)
target_link_libraries(test_talloc PRIVATE optm)
add_test(NAME talloc COMMAND test_talloc)

add_executable(test_aa tests/test_aa.cpp)
target_link_libraries(test_aa PRIVATE optm)
add_test(NAME aa COMMAND test_aa)

add_executable(test_housing tests/test_housing.cpp)
target_link_libraries(test_housing PRIVATE optm)
add_test(NAME housing COMMAND test_housing)

add_executable(test_athermal tests/test_athermal.cpp)
target_link_libraries(test_athermal PRIVATE optm)
add_test(NAME athermal COMMAND test_athermal)
