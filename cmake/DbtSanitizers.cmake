# An INTERFACE target carrying sanitizer flags.
#
# Sanitizer flags must reach both the compile and link lines, and must propagate
# to anything linking the library, so this target is linked PUBLIC.
#
# AddressSanitizer is not shipped with the MinGW-w64 (w64devkit) GCC toolchain.
# Enabling DBT_ENABLE_ASAN there fails at configure time with a clear message
# rather than producing a build that links against missing runtime symbols.

add_library(dbt_sanitizers INTERFACE)
add_library(dbt::sanitizers ALIAS dbt_sanitizers)

# Hardening that works everywhere, including the MinGW toolchain where
# AddressSanitizer is unavailable.
#
# _GLIBCXX_ASSERTIONS turns operator[] on vector/array/span into a checked
# access, which is exactly the class of bug that matters when the indices come
# from decoded instruction fields.
if(DBT_HARDENED)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # libstdc++ and libc++ spell their bounds-checking modes differently.
        # Defining only _GLIBCXX_ASSERTIONS leaves macOS (libc++) unhardened
        # while still reporting hardening as enabled.
        target_compile_definitions(dbt_sanitizers INTERFACE _GLIBCXX_ASSERTIONS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_definitions(dbt_sanitizers INTERFACE
                _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE)
        endif()
        target_compile_options(dbt_sanitizers INTERFACE -fstack-protector-strong)
    elseif(MSVC)
        target_compile_options(dbt_sanitizers INTERFACE /GS /sdl)
    endif()
endif()

if(DBT_ENABLE_ASAN)
    if(MSVC)
        target_compile_options(dbt_sanitizers INTERFACE /fsanitize=address)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        if(MINGW)
            message(FATAL_ERROR
                "DBT_ENABLE_ASAN is ON but the MinGW-w64 GCC toolchain does not ship "
                "AddressSanitizer. Use clang-cl, MSVC, or a Linux/macOS toolchain for "
                "sanitizer builds, or configure with -DDBT_ENABLE_ASAN=OFF.")
        endif()

        set(_dbt_san_flags
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
            -fno-sanitize-recover=all)
        target_compile_options(dbt_sanitizers INTERFACE ${_dbt_san_flags})
        target_link_options(dbt_sanitizers INTERFACE ${_dbt_san_flags})
        unset(_dbt_san_flags)
    endif()
endif()
