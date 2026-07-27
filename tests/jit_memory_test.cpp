#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "dbt/backend/arm64_encoder.hpp"
#include "dbt/backend/jit_memory.hpp"
#include "dbt/version.hpp"

namespace {

namespace a64 = dbt::backend::a64;
using dbt::Arm64Word;
using dbt::usize;
using dbt::backend::JitError;
using dbt::backend::JitMemory;

JitMemory allocate_or_fail(usize bytes) {
    JitError error = JitError::None;
    JitMemory memory = JitMemory::allocate(bytes, error);
    EXPECT_EQ(error, JitError::None) << dbt::backend::to_string(error);
    EXPECT_TRUE(memory.valid());
    return memory;
}

// --- Allocation ------------------------------------------------------------

TEST(JitMemory, DefaultConstructedIsEmpty) {
    const JitMemory memory;
    EXPECT_FALSE(memory.valid());
    EXPECT_EQ(memory.size(), 0u);
    EXPECT_FALSE(memory.is_executable());
    EXPECT_EQ(memory.code(), nullptr);
}

TEST(JitMemory, ZeroSizeIsRejected) {
    JitError error = JitError::None;
    const JitMemory memory = JitMemory::allocate(0, error);
    EXPECT_EQ(error, JitError::InvalidSize);
    EXPECT_FALSE(memory.valid());
}

TEST(JitMemory, CapacityIsRoundedUpToAPage) {
    const JitMemory memory = allocate_or_fail(16);
    EXPECT_EQ(memory.size(), 16u);
    EXPECT_EQ(memory.capacity(), JitMemory::page_size());
    EXPECT_GE(memory.capacity(), memory.size());
    EXPECT_EQ(memory.capacity() % JitMemory::page_size(), 0u);
}

TEST(JitMemory, LargeAllocationSpansMultiplePages) {
    const usize page = JitMemory::page_size();
    const JitMemory memory = allocate_or_fail(page + 1);
    EXPECT_EQ(memory.capacity(), page * 2);
}

TEST(JitMemory, StartsWritableAndNotExecutable) {
    const JitMemory memory = allocate_or_fail(64);
    EXPECT_FALSE(memory.is_executable());
    EXPECT_NE(memory.code(), nullptr);
}

// --- Writing ---------------------------------------------------------------

TEST(JitMemory, WriteThenReadBackRoundTrips) {
    JitMemory memory = allocate_or_fail(64);
    const std::array<Arm64Word, 3> words{a64::movz(a64::Reg::X0, 42), a64::nop(),
                                         a64::ret()};

    ASSERT_EQ(memory.write(0, words), JitError::None);

    std::array<Arm64Word, 3> read_back{};
    std::memcpy(read_back.data(), memory.code(), sizeof(read_back));
    EXPECT_EQ(read_back, words);
}

TEST(JitMemory, WriteAtAnOffsetLandsAtTheRightPlace) {
    JitMemory memory = allocate_or_fail(64);
    const std::array<Arm64Word, 1> first{a64::nop()};
    const std::array<Arm64Word, 1> second{a64::ret()};

    ASSERT_EQ(memory.write(0, first), JitError::None);
    ASSERT_EQ(memory.write(1, second), JitError::None);

    std::array<Arm64Word, 2> read_back{};
    std::memcpy(read_back.data(), memory.code(), sizeof(read_back));
    EXPECT_EQ(read_back[0], a64::nop());
    EXPECT_EQ(read_back[1], a64::ret());
}

TEST(JitMemory, EmptyWriteIsANoOp) {
    JitMemory memory = allocate_or_fail(64);
    EXPECT_EQ(memory.write(0, std::span<const Arm64Word>{}), JitError::None);
}

TEST(JitMemory, WritePastTheEndIsRejected) {
    JitMemory memory = allocate_or_fail(64);
    const usize capacity_words = memory.capacity() / dbt::kArm64InstSize;
    const std::array<Arm64Word, 1> word{a64::nop()};

    // One word past the last valid slot.
    EXPECT_EQ(memory.write(capacity_words, word), JitError::WriteOutOfRange);
    // The final valid slot must still be accepted.
    EXPECT_EQ(memory.write(capacity_words - 1, word), JitError::None);
}

TEST(JitMemory, OversizedWriteIsRejected) {
    JitMemory memory = allocate_or_fail(64);
    const usize capacity_words = memory.capacity() / dbt::kArm64InstSize;
    const std::vector<Arm64Word> words(capacity_words + 1, a64::nop());

    EXPECT_EQ(memory.write(0, words), JitError::WriteOutOfRange);
}

TEST(JitMemory, OffsetOverflowIsRejectedRatherThanWrapping) {
    JitMemory memory = allocate_or_fail(64);
    const std::array<Arm64Word, 2> words{a64::nop(), a64::ret()};

    // An offset near the top of the address space must not wrap when the
    // length is added to it.
    EXPECT_EQ(memory.write(~usize{0} - 1, words), JitError::WriteOutOfRange);
    EXPECT_EQ(memory.write(~usize{0}, words), JitError::WriteOutOfRange);
}

TEST(JitMemory, WritingToAnEmptyBufferFails) {
    JitMemory memory;
    const std::array<Arm64Word, 1> word{a64::nop()};
    EXPECT_EQ(memory.write(0, word), JitError::NotWritable);
}

// --- W^X transition --------------------------------------------------------

TEST(JitMemory, MakeExecutableFlipsTheFlag) {
    JitMemory memory = allocate_or_fail(64);
    const std::array<Arm64Word, 1> word{a64::ret()};
    ASSERT_EQ(memory.write(0, word), JitError::None);

    ASSERT_EQ(memory.make_executable(), JitError::None);
    EXPECT_TRUE(memory.is_executable());
}

TEST(JitMemory, WritesAreRefusedOnceExecutable) {
    JitMemory memory = allocate_or_fail(64);
    ASSERT_EQ(memory.make_executable(), JitError::None);

    const std::array<Arm64Word, 1> word{a64::nop()};
    // The whole point of W^X: no path back to writable.
    EXPECT_EQ(memory.write(0, word), JitError::NotWritable);
}

TEST(JitMemory, SealingTwiceIsReported) {
    JitMemory memory = allocate_or_fail(64);
    ASSERT_EQ(memory.make_executable(), JitError::None);
    EXPECT_EQ(memory.make_executable(), JitError::AlreadyExecutable);
}

TEST(JitMemory, ContentSurvivesTheProtectionChange) {
    JitMemory memory = allocate_or_fail(64);
    const std::array<Arm64Word, 2> words{a64::movz(a64::Reg::X0, 7), a64::ret()};
    ASSERT_EQ(memory.write(0, words), JitError::None);
    ASSERT_EQ(memory.make_executable(), JitError::None);

    std::array<Arm64Word, 2> read_back{};
    std::memcpy(read_back.data(), memory.code(), sizeof(read_back));
    EXPECT_EQ(read_back, words);
}

// --- Lifetime --------------------------------------------------------------

TEST(JitMemory, MoveConstructionTransfersOwnership) {
    JitMemory source = allocate_or_fail(64);
    const void* address = source.code();
    const usize size = source.size();

    const JitMemory moved(std::move(source));
    EXPECT_EQ(moved.code(), address);
    EXPECT_EQ(moved.size(), size);
    EXPECT_TRUE(moved.valid());

    EXPECT_FALSE(source.valid());
    EXPECT_EQ(source.code(), nullptr);
    EXPECT_EQ(source.size(), 0u);
}

TEST(JitMemory, MoveAssignmentReleasesThePreviousMapping) {
    JitMemory first = allocate_or_fail(64);
    JitMemory second = allocate_or_fail(128);
    const void* second_address = second.code();

    first = std::move(second);
    EXPECT_EQ(first.code(), second_address);
    EXPECT_FALSE(second.valid());
}

TEST(JitMemory, SelfMoveAssignmentIsSafe) {
    JitMemory memory = allocate_or_fail(64);
    const void* address = memory.code();

    JitMemory& alias = memory;
    memory = std::move(alias);

    EXPECT_TRUE(memory.valid());
    EXPECT_EQ(memory.code(), address);
}

TEST(JitMemory, ResetReleasesTheMapping) {
    JitMemory memory = allocate_or_fail(64);
    memory.reset();
    EXPECT_FALSE(memory.valid());
    EXPECT_EQ(memory.size(), 0u);
    EXPECT_FALSE(memory.is_executable());
}

TEST(JitMemory, ErrorNames) {
    EXPECT_EQ(dbt::backend::to_string(JitError::None), "none");
    EXPECT_EQ(dbt::backend::to_string(JitError::WriteOutOfRange), "write-out-of-range");
    EXPECT_EQ(dbt::backend::to_string(JitError::NotWritable), "not-writable");
}

// --- Execution (ARM64 hosts only) ------------------------------------------

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(JitMemoryExec, EmittedCodeActuallyRuns) {
    JitMemory memory = allocate_or_fail(64);
    // movz x0, #42 ; ret   -- returns 42 in the result register.
    const std::array<Arm64Word, 2> words{a64::movz(a64::Reg::X0, 42), a64::ret()};
    ASSERT_EQ(memory.write(0, words), JitError::None);
    ASSERT_EQ(memory.make_executable(), JitError::None);

    using Fn = dbt::u64 (*)();
    const Fn fn = memory.entry<Fn>();
    ASSERT_NE(fn, nullptr);
    EXPECT_EQ(fn(), 42u);
}

TEST(JitMemoryExec, ArithmeticSequenceRuns) {
    JitMemory memory = allocate_or_fail(64);
    // movz x0,#10 ; movz x1,#32 ; add x0,x0,x1 ; ret   -> 42
    const std::array<Arm64Word, 4> words{
        a64::movz(a64::Reg::X0, 10), a64::movz(a64::Reg::X1, 32),
        a64::add_reg(a64::Reg::X0, a64::Reg::X0, a64::Reg::X1), a64::ret()};
    ASSERT_EQ(memory.write(0, words), JitError::None);
    ASSERT_EQ(memory.make_executable(), JitError::None);

    using Fn = dbt::u64 (*)();
    EXPECT_EQ(memory.entry<Fn>()(), 42u);
}
#else
TEST(JitMemoryExec, SkippedOnNonArm64Hosts) {
    // The encoders are still fully verified by the golden-word tests; only
    // execution requires an AArch64 host.
    EXPECT_FALSE(dbt::host_can_execute_arm64());
    GTEST_SKIP() << "emitted ARM64 code cannot be executed on this host";
}
#endif

}  // namespace
