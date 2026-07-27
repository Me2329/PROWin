#include <gtest/gtest.h>

#include "dbt/common/types.hpp"
#include "dbt/version.hpp"

namespace {

TEST(Smoke, FixedWidthTypesHaveExpectedSizes) {
    static_assert(sizeof(dbt::u8) == 1);
    static_assert(sizeof(dbt::u16) == 2);
    static_assert(sizeof(dbt::u32) == 4);
    static_assert(sizeof(dbt::u64) == 8);
    static_assert(sizeof(dbt::GuestAddr) == 8);
    static_assert(sizeof(dbt::Arm64Word) == dbt::kArm64InstSize);
    SUCCEED();
}

TEST(Smoke, VersionStringMatchesVersionConstants) {
    EXPECT_EQ(dbt::version_string(), "0.1.0");
    EXPECT_EQ(dbt::kVersionMajor, 0u);
    EXPECT_EQ(dbt::kVersionMinor, 1u);
    EXPECT_EQ(dbt::kVersionPatch, 0u);
}

TEST(Smoke, HostExecutionSupportMatchesBuildArchitecture) {
#if defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_TRUE(dbt::host_can_execute_arm64());
#else
    EXPECT_FALSE(dbt::host_can_execute_arm64());
#endif
}

}  // namespace
