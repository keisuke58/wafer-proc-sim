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
  src/simd_filters.cpp
  src/metrology.cpp
  src/spc.cpp
  src/blade_health.cpp
  src/apc.cpp
  src/stealth_ir.cpp
  src/wafer_map.cpp
  src/grind_metrology.cpp
  src/grind_surface.cpp
  src/grind_control.cpp
)
target_include_directories(wproc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Threads REQUIRED)
target_link_libraries(wproc PUBLIC Threads::Threads)

# Enable AVX2/FMA only for the SIMD translation unit (portable elsewhere).
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" WPROC_HAS_MAVX2)
if(WPROC_HAS_MAVX2)
  set_source_files_properties(src/simd_filters.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# Applications.
add_executable(kerf_inspect apps/kerf_inspect_main.cpp)
target_link_libraries(kerf_inspect PRIVATE wproc)

add_executable(gen_synthetic apps/gen_synthetic.cpp)
target_link_libraries(gen_synthetic PRIVATE wproc)

add_executable(bench apps/bench.cpp)
target_link_libraries(bench PRIVATE wproc Threads::Threads)

add_executable(grind_demo apps/grind_demo.cpp)
target_link_libraries(grind_demo PRIVATE wproc)

# Tests.
enable_testing()
add_executable(test_imgproc tests/test_imgproc.cpp)
target_link_libraries(test_imgproc PRIVATE wproc)
add_test(NAME imgproc COMMAND test_imgproc)

add_executable(test_simd_parity tests/test_simd_parity.cpp)
target_link_libraries(test_simd_parity PRIVATE wproc Threads::Threads)
add_test(NAME simd_parity COMMAND test_simd_parity)

add_executable(test_metrology tests/test_metrology.cpp)
target_link_libraries(test_metrology PRIVATE wproc)
add_test(NAME metrology COMMAND test_metrology)

add_executable(test_blade_health tests/test_blade_health.cpp)
target_link_libraries(test_blade_health PRIVATE wproc)
add_test(NAME blade_health COMMAND test_blade_health)

add_executable(test_apc tests/test_apc.cpp)
target_link_libraries(test_apc PRIVATE wproc)
add_test(NAME apc COMMAND test_apc)

add_executable(test_stealth tests/test_stealth.cpp)
target_link_libraries(test_stealth PRIVATE wproc)
add_test(NAME stealth COMMAND test_stealth)

add_executable(test_wafer_map tests/test_wafer_map.cpp)
target_link_libraries(test_wafer_map PRIVATE wproc)
add_test(NAME wafer_map COMMAND test_wafer_map)

add_executable(test_grind_metrology tests/test_grind_metrology.cpp)
target_link_libraries(test_grind_metrology PRIVATE wproc)
add_test(NAME grind_metrology COMMAND test_grind_metrology)

add_executable(test_grind_surface tests/test_grind_surface.cpp)
target_link_libraries(test_grind_surface PRIVATE wproc)
add_test(NAME grind_surface COMMAND test_grind_surface)

add_executable(test_grind_control tests/test_grind_control.cpp)
target_link_libraries(test_grind_control PRIVATE wproc)
add_test(NAME grind_control COMMAND test_grind_control)

# ── Optional backends ────────────────────────────────────────────────────────

# OpenCV backend + parity test (cross-validates the from-scratch kernels).
option(WPROC_WITH_OPENCV "Build the OpenCV backend and parity test" ON)
if(WPROC_WITH_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core imgproc)
  if(OpenCV_FOUND)
    add_library(wproc_cv STATIC src/cv_backend.cpp src/street_detect.cpp)
    target_link_libraries(wproc_cv PUBLIC wproc ${OpenCV_LIBS})
    target_include_directories(wproc_cv PRIVATE ${OpenCV_INCLUDE_DIRS})

    add_executable(test_cv_parity tests/test_cv_parity.cpp)
    target_link_libraries(test_cv_parity PRIVATE wproc_cv)
    add_test(NAME cv_parity COMMAND test_cv_parity)

    add_executable(test_street tests/test_street.cpp)
    target_link_libraries(test_street PRIVATE wproc_cv ${OpenCV_LIBS})
    target_include_directories(test_street PRIVATE ${OpenCV_INCLUDE_DIRS})
    add_test(NAME street COMMAND test_street)
    message(STATUS "wproc: OpenCV ${OpenCV_VERSION} backend enabled")
  else()
    message(STATUS "wproc: OpenCV not found — backend skipped")
  endif()
endif()

# Qt6 GUI viewer (headless self-test via QT_QPA_PLATFORM=offscreen).
option(WPROC_BUILD_GUI "Build the Qt6 kerf_viewer GUI" ON)
if(WPROC_BUILD_GUI)
  find_package(Qt6 QUIET COMPONENTS Widgets)
  if(Qt6_FOUND)
    add_executable(kerf_viewer gui/kerf_viewer.cpp)
    target_link_libraries(kerf_viewer PRIVATE wproc Qt6::Widgets)
    message(STATUS "wproc: Qt6 kerf_viewer GUI enabled")
  else()
    message(STATUS "wproc: Qt6 not found — GUI skipped")
  endif()
endif()

# CUDA kernels (built only when a CUDA toolkit is present).
option(WPROC_WITH_CUDA "Build CUDA filter kernels if a toolkit is found" ON)
if(WPROC_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_library(wproc_cuda STATIC cuda/wproc_cuda.cu)
    target_link_libraries(wproc_cuda PUBLIC wproc)
    target_include_directories(wproc_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(wproc_cuda PROPERTIES CUDA_STANDARD 17)
    message(STATUS "wproc: CUDA backend enabled")
  else()
    message(STATUS "wproc: CUDA toolkit not found — GPU kernels skipped")
  endif()
endif()
