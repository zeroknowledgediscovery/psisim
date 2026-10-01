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

  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
  FetchContent_MakeAvailable(nlohmann_json)

  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0)
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
endif()
