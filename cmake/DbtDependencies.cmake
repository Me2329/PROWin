include(FetchContent)

# Zydis and GoogleTest declare an older cmake_minimum_required than CMake 4.x
# accepts. Raising the floor for the subprojects keeps them configurable without
# patching upstream sources.
set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "" FORCE)

# --- Zydis: the x86/x86-64 decoder wrapped by the frontend -----------------
# Only the decoder is needed, so the formatter tools, examples and docs are off.
set(ZYDIS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_DOXYGEN OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)

FetchContent_Declare(zydis
    GIT_REPOSITORY https://github.com/zyantific/zydis.git
    GIT_TAG v4.1.1
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    SYSTEM)

# --- GoogleTest ------------------------------------------------------------
if(DBT_BUILD_TESTS)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        SYSTEM)

    FetchContent_MakeAvailable(zydis googletest)
else()
    FetchContent_MakeAvailable(zydis)
endif()
