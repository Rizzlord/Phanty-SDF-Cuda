cmake_minimum_required(VERSION 3.18)
project(contouring LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(DC_ENABLE_CUDA "Enable CUDA backend" ON)
option(DC_ENABLE_VIEWER "Enable Polyscope viewer" ON)
option(DC_ENABLE_NANOVDB "Enable NanoVDB support" OFF)

include(libigl)

if(DC_ENABLE_VIEWER)
    igl_include(glfw)
    add_subdirectory(external/polyscope)
endif()

set(CORE_SRC
    src/cpp/Cell.cpp
    src/cpp/contouring.cpp
    src/cpp/hermite_update.cpp
    src/cpp/vertex_refinement.cpp
    src/cpp/sdf.cpp
    src/cpp/Sphere.cpp
    src/cpp/sample_fun.cpp
    src/cpp/dc_backend.cpp
)

add_library(dc_core STATIC ${CORE_SRC})
target_include_directories(dc_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(dc_core PUBLIC
    igl::core
)

if(DC_ENABLE_VIEWER)
    target_compile_definitions(dc_core PUBLIC DC_ENABLE_VIEWER)
endif()
if(DC_ENABLE_NANOVDB)
    target_compile_definitions(dc_core PUBLIC DC_ENABLE_NANOVDB)
endif()

set(DC_CUDA_FOUND OFF)
if(DC_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(DC_CUDA_FOUND ON)
    endif()
endif()

if(DC_CUDA_FOUND)
    add_library(dc_cuda STATIC src/cuda/dc_cuda.cu src/cuda/dc_cuda_sparse.cu src/cuda/mesh_to_sdf_cuda.cu)
    target_include_directories(dc_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(dc_cuda PUBLIC
        dc_core
    )
    target_compile_definitions(dc_core PUBLIC DC_ENABLE_CUDA)
    target_compile_definitions(dc_cuda PUBLIC DC_ENABLE_CUDA)
    if(DC_ENABLE_NANOVDB)
        target_compile_definitions(dc_cuda PUBLIC DC_ENABLE_NANOVDB)
    endif()
endif()

if(DC_ENABLE_VIEWER)
    add_library(dc_viewer STATIC src/cpp/ui.cpp)
    target_include_directories(dc_viewer PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(dc_viewer PUBLIC
        dc_core
        igl::glfw
        polyscope
    )
    target_compile_definitions(dc_viewer PUBLIC DC_ENABLE_VIEWER)
    if(DC_CUDA_FOUND)
        target_compile_definitions(dc_viewer PUBLIC DC_ENABLE_CUDA)
    endif()
endif()

add_executable(dc_cli src/cpp/dc_cli.cpp)
target_link_libraries(dc_cli PRIVATE dc_core)
if(DC_CUDA_FOUND)
    target_link_libraries(dc_cli PRIVATE dc_cuda)
    target_compile_definitions(dc_cli PRIVATE DC_ENABLE_CUDA)
endif()

add_executable(dc_tests src/cpp/dc_tests.cpp)
target_link_libraries(dc_tests PRIVATE dc_core)
if(DC_CUDA_FOUND)
    target_link_libraries(dc_tests PRIVATE dc_cuda)
    target_compile_definitions(dc_tests PRIVATE DC_ENABLE_CUDA)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)

if(pybind11_FOUND)
    pybind11_add_module(contouring_py src/cpp/bindings.cpp)
    set_target_properties(contouring_py PROPERTIES 
        OUTPUT_NAME "_contouring_cpp_module"
    )
    set_target_properties(contouring_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/python/contouring"
    )
    target_include_directories(contouring_py PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(contouring_py PRIVATE dc_core pybind11::module)
    if(DC_ENABLE_VIEWER)
        target_link_libraries(contouring_py PRIVATE dc_viewer)
        target_compile_definitions(contouring_py PRIVATE DC_ENABLE_VIEWER)
    endif()
    if(DC_CUDA_FOUND)
        target_link_libraries(contouring_py PRIVATE dc_cuda)
        target_compile_definitions(contouring_py PRIVATE DC_ENABLE_CUDA)
    endif()
endif()