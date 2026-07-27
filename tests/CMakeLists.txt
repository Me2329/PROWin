include(GoogleTest)

# Declares one test executable and registers its cases with CTest.
#   dbt_add_test(<name> <source>...)
function(dbt_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name}
        PRIVATE
            dbt::dbt
            dbt::warnings
            GTest::gtest_main)
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    gtest_discover_tests(${name}
        DISCOVERY_TIMEOUT 30
        PROPERTIES LABELS unit)
endfunction()

dbt_add_test(smoke_test smoke_test.cpp)
dbt_add_test(decoder_test decoder_test.cpp)
dbt_add_test(ir_test ir_test.cpp)
dbt_add_test(lifter_test lifter_test.cpp)
dbt_add_test(encoder_test encoder_test.cpp)
dbt_add_test(jit_memory_test jit_memory_test.cpp)
dbt_add_test(compiler_test compiler_test.cpp)
dbt_add_test(runtime_test runtime_test.cpp)
