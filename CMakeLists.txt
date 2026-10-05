cmake_minimum_required(VERSION 3.18)
project(PsiSimLSMRuntime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compatibility with the original applications/psisimulation build command.
option(LSM_BUILD_PYTHON_BINDINGS "Build vendored LSM Python bindings" ON)

if(LSM_BUILD_PYTHON_BINDINGS)
  include(FetchContent)
  find_package(Threads REQUIRED)
  find_package(OpenMP REQUIRED COMPONENTS CXX)

  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()

  # nlohmann_json is header-only: use an installed copy if there is one,
  # otherwise the 110 KB release archive (headers + CMake target) instead of
  # cloning the whole repository (~300 MB).
  find_package(nlohmann_json 3.11 QUIET)
  if(nlohmann_json_FOUND)
    message(STATUS "Using installed nlohmann_json ${nlohmann_json_VERSION}")
  else()
    FetchContent_Declare(
      nlohmann_json
      URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
      URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
    FetchContent_MakeAvailable(nlohmann_json)
  endif()

  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(pybind11)

  function(configure_psisim_binding target)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(
      ${target}
      PRIVATE
      nlohmann_json::nlohmann_json
      Threads::Threads
      OpenMP::OpenMP_CXX)
    set_target_properties(
      ${target}
      PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
  endfunction()

  set(PSISIM_RUNTIME_SOURCES
    src/SourceMaps/SourceMapStore.cpp
    src/PredictDistribution/PredictDistribution.cpp
    src/Tree/Tree.cpp
    src/ShardMap/ShardMap.cpp)

  pybind11_add_module(
    predict_distribution
    bindings/predict_distribution_py.cpp
    ${PSISIM_RUNTIME_SOURCES})
  configure_psisim_binding(predict_distribution)

  pybind11_add_module(
    qdistance
    bindings/qdistance_py.cpp
    ${PSISIM_RUNTIME_SOURCES}
    src/QDistance/qdistance.cpp)
  configure_psisim_binding(qdistance)

  # PsiSim-owned (not synced from LSM): learned dependency graph and the
  # propagation-only centered dynamics used by the webapp.
  pybind11_add_module(
    psisim_dynamics
    bindings/psisim_dynamics_py.cpp
    ${PSISIM_RUNTIME_SOURCES})
  configure_psisim_binding(psisim_dynamics)
endif()
