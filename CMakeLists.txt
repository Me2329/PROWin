cmake_minimum_required(VERSION 3.25)

project(dbt
    VERSION 0.1.0
    DESCRIPTION "Minimal modular x86_64 to ARM64 binary translation library"
    LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

option(DBT_BUILD_TESTS "Build the dbt unit test suite" ${PROJECT_IS_TOP_LEVEL})
option(DBT_BUILD_EXAMPLES "Build the example programs" ${PROJECT_IS_TOP_LEVEL})
option(DBT_WERROR "Treat compiler warnings as errors" ON)
option(DBT_ENABLE_ASAN "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
# Hardening is on by default: this library writes and executes generated code,
# so the cheap runtime checks are worth their cost even in release builds.
option(DBT_HARDENED "Enable libstdc++ bounds assertions and the stack protector" ON)

# The ARM64 code this library emits can only be *executed* on an ARM64 host.
# Everywhere else the backend is still fully testable through golden-encoding
# tests, so only the execution suite is gated on this.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^([Aa][Aa][Rr][Cc][Hh]64|[Aa][Rr][Mm]64)$")
    set(DBT_HOST_IS_ARM64 ON)
else()
    set(DBT_HOST_IS_ARM64 OFF)
endif()

include(DbtWarnings)
include(DbtSanitizers)
include(DbtDependencies)

add_subdirectory(src)

if(DBT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(DBT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

message(STATUS "dbt ${PROJECT_VERSION}")
message(STATUS "  compiler       : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  host processor : ${CMAKE_SYSTEM_PROCESSOR} (arm64 execution tests: ${DBT_HOST_IS_ARM64})")
message(STATUS "  tests          : ${DBT_BUILD_TESTS}")
message(STATUS "  warnings-as-err: ${DBT_WERROR}")
message(STATUS "  sanitizers     : ${DBT_ENABLE_ASAN}")
