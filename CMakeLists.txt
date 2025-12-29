cmake_minimum_required(VERSION 3.20)

project(OpenMP_Examples C)

# -------------------- Language / defaults --------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Put *all* runtime outputs under build/bin/...
set(_BIN_ROOT "${CMAKE_BINARY_DIR}/bin")

# -------------------- Source discovery -----------------------
file(GLOB_RECURSE _EXAMPLE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.c"
)

# Optional: nicer IDE view (Visual Studio)
source_group(TREE "${CMAKE_SOURCE_DIR}/src" PREFIX "src" FILES ${_EXAMPLE_SOURCES})

# -------------------- OpenMP (optional but typical) -----------
# If you want to build even non-OpenMP files without requiring OpenMP,
# keep QUIET and only link when found.
find_package(OpenMP QUIET)

# -------------------- One executable per C file ---------------
foreach(_SRC IN LISTS _EXAMPLE_SOURCES)

    # Relative path within src/
    file(RELATIVE_PATH _REL_FROM_SRC "${CMAKE_SOURCE_DIR}/src" "${_SRC}")
    get_filename_component(_REL_DIR  "${_REL_FROM_SRC}" DIRECTORY)  # e.g. 01_basics
    get_filename_component(_BASE     "${_REL_FROM_SRC}" NAME_WE)    # e.g. omp_hello

    # Unique internal target name (must not contain '/'; also stable across platforms)
    set(_TGT "${_REL_FROM_SRC}")
    string(REPLACE "/"  "__" _TGT "${_TGT}")
    string(REPLACE "\\" "__" _TGT "${_TGT}")
    string(REPLACE "."  "_"  _TGT "${_TGT}")

    add_executable("${_TGT}" "${_SRC}")

    # Clean user-facing binary name (what appears in bin/<dir>/...)
    # Example: omp_hello.c -> omp_hello_c.exe
    set_target_properties("${_TGT}" PROPERTIES
        OUTPUT_NAME "${_BASE}_c"
    )

    # Output directory mirrors src/ structure: bin/<relative_path>/
    if(_REL_DIR STREQUAL "")
        set(_OUT_DIR "${_BIN_ROOT}")
    else()
        set(_OUT_DIR "${_BIN_ROOT}/${_REL_DIR}")
    endif()

    # Single-config generators (Ninja, Makefiles)
    set_target_properties("${_TGT}" PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${_OUT_DIR}"
    )

    # Multi-config generators (Visual Studio, Xcode)
    foreach(_CFG IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${_CFG}" _CFG_UP)
        set_target_properties("${_TGT}" PROPERTIES
            "RUNTIME_OUTPUT_DIRECTORY_${_CFG_UP}" "${_OUT_DIR}"
        )
    endforeach()

    # Warnings
    if(MSVC)
        target_compile_options("${_TGT}" PRIVATE /W4)
    else()
        target_compile_options("${_TGT}" PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # Enable OpenMP if available
    if(OpenMP_C_FOUND)
        target_link_libraries("${_TGT}" PRIVATE OpenMP::OpenMP_C)
    elseif(MSVC)
        # Fallback for older MSVC configurations
        target_compile_options("${_TGT}" PRIVATE /openmp)
    endif()

    # Reduce clutter in bin/: keep .exe in bin/, move .pdb elsewhere, and disable .ilk
    if(MSVC)
        # Disable incremental linking -> no .ilk
        target_link_options("${_TGT}" PRIVATE /INCREMENTAL:NO)

        # Put PDBs under build/pdb/<mirrored subdirs> (instead of next to .exe)
        set(_PDB_DIR "${CMAKE_BINARY_DIR}/pdb")
        if(NOT _REL_DIR STREQUAL "")
            set(_PDB_DIR "${_PDB_DIR}/${_REL_DIR}")
        endif()

        set_target_properties("${_TGT}" PROPERTIES
            PDB_OUTPUT_DIRECTORY "${_PDB_DIR}"
        )

        foreach(_CFG IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
            string(TOUPPER "${_CFG}" _CFG_UP)
            set_target_properties("${_TGT}" PROPERTIES
                "PDB_OUTPUT_DIRECTORY_${_CFG_UP}" "${_PDB_DIR}"
            )
        endforeach()
    endif()

endforeach()

message(STATUS "Discovered ${_EXAMPLE_SOURCES} -> ${_BIN_ROOT}/<mirrored subdirs>")
