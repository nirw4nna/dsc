cmake_minimum_required(VERSION 3.18)
project(dsc_bindings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Building Python bindings.")
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)

if (NOT DEFINED DSC_LIB OR NOT DEFINED DSC_INCLUDE_DIR)
    message(FATAL_ERROR "Set DSC_LIB to your libdsc.a and DSC_INCLUDE_DIR to the headers.")
else ()
    message(STATUS "DSC_LIB: ${DSC_LIB}")
    message(STATUS "DSC_INCLUDE_DIR: ${DSC_INCLUDE_DIR}")
    # Get the dirname of DSC_LIB so that I can output the compiled module there
    get_filename_component(DSC_PYTHON_BINDINGS_PATH ${DSC_LIB} DIRECTORY)
endif ()

# Link against DSC static lib
add_library(dsc STATIC IMPORTED GLOBAL)
set_target_properties(dsc PROPERTIES
        IMPORTED_LOCATION "${DSC_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${DSC_INCLUDE_DIR}"
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
)

nanobind_add_module(
        core
        NB_STATIC
        STABLE_ABI
        LTO
        NOMINSIZE
        NB_DOMAIN dsc
        ${CMAKE_CURRENT_SOURCE_DIR}/dsc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
)

# Output the bindings alongside the static lib
set_target_properties(core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${DSC_PYTHON_BINDINGS_PATH}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${DSC_PYTHON_BINDINGS_PATH}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${DSC_PYTHON_BINDINGS_PATH})

target_link_libraries(core PRIVATE dsc)

