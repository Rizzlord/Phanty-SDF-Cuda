cmake_minimum_required(VERSION 3.16)
project(contouring)

# Default to Debug build if no build type is specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the build type" FORCE)
endif()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)
igl_include(glfw)

# Polyscope (as a submodule under external/polyscope)
add_subdirectory(external/polyscope)

# ============================================
# Library: contouring  (src/*.cpp + include/)
# ============================================
file(GLOB CONTOURING_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp
)
# Remove bindings.cpp from the library source files (bindings are built as a pybind11 module)
list(REMOVE_ITEM CONTOURING_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/bindings.cpp")

add_library(contouring ${CONTOURING_SRC_FILES})

target_include_directories(contouring PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# contouring depends on libigl + polyscope
target_link_libraries(contouring PUBLIC
  igl::glfw
  polyscope
)

# ============================================
# Python bindings
# ============================================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)

if(pybind11_FOUND)
  # Use the 'contouring_py' name to avoid conflict with your static lib
  pybind11_add_module(contouring_py src/cpp/bindings.cpp)

  set_target_properties(contouring_py PROPERTIES 
    OUTPUT_NAME "_contouring_cpp_module"
  )

  # Place the compiled Python extension inside the Python package so tests
  # and `import _contouring_cpp_module` work when `SRC/python` is on sys.path.
  set_target_properties(contouring_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/SRC/python/contouring"
  )

    # Link the library and the headers
    target_link_libraries(contouring_py PRIVATE contouring pybind11::module)
    
    target_include_directories(contouring_py PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()