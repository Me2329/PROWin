add_library(dbt
    common/version.cpp
    decoder/decoder.cpp
    ir/ir.cpp
    frontend/lifter.cpp
    backend/jit_memory.cpp
    backend/compiler.cpp
    runtime/dispatcher.cpp
)
add_library(dbt::dbt ALIAS dbt)

target_include_directories(dbt
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(dbt
    PRIVATE dbt::warnings Zydis
    PUBLIC dbt::sanitizers)

target_compile_features(dbt PUBLIC cxx_std_20)

set_target_properties(dbt PROPERTIES
    VERSION ${PROJECT_VERSION}
    POSITION_INDEPENDENT_CODE ON)
