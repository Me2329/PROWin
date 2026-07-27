# An INTERFACE target carrying the project warning policy.
#
# This is deliberately *not* applied through CMAKE_CXX_FLAGS: third-party code
# pulled in by FetchContent (Zydis, GoogleTest) must not be compiled with
# -Werror, or an upstream warning breaks our build.

add_library(dbt_warnings INTERFACE)
add_library(dbt::warnings ALIAS dbt_warnings)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dbt_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wcast-qual
        -Wold-style-cast
        -Wnon-virtual-dtor
        -Woverloaded-virtual
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(dbt_warnings INTERFACE
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
            -Wuseless-cast)
    endif()
elseif(MSVC)
    target_compile_options(dbt_warnings INTERFACE /W4 /permissive-)
endif()

if(DBT_WERROR)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(dbt_warnings INTERFACE -Werror)
    elseif(MSVC)
        target_compile_options(dbt_warnings INTERFACE /WX)
    endif()
endif()
