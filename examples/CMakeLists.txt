add_executable(dbt_translate translate.cpp)

target_link_libraries(dbt_translate
    PRIVATE
        dbt::dbt
        dbt::warnings)
