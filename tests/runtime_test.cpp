#include <gtest/gtest.h>

#include <array>
#include <span>

#include "dbt/common/types.hpp"
#include "dbt/runtime/cpu_state.hpp"
#include "dbt/runtime/dispatcher.hpp"
#include "dbt/runtime/translation_cache.hpp"
#include "dbt/version.hpp"

namespace {

using dbt::GuestAddr;
using dbt::u8;
using dbt::decoder::X86Reg;
using dbt::runtime::CpuState;
using dbt::runtime::Dispatcher;
using dbt::runtime::ExecStatus;
using dbt::runtime::TranslateStatus;

constexpr GuestAddr kBase = 0x1000;

/// mov rax, rbx ; ret
constexpr std::array<u8, 4> kMovRet{0x48, 0x89, 0xD8, 0xC3};

constexpr dbt::usize reg_index(X86Reg reg) {
    return static_cast<dbt::usize>(static_cast<u8>(reg));
}

// --- CpuState --------------------------------------------------------------

TEST(CpuStateLayout, MatchesTheOffsetsEmittedCodeUses) {
    // Compiled blocks address these by hard-coded offset, so drift here is a
    // silent memory-corruption bug rather than a compile error elsewhere.
    static_assert(dbt::runtime::gpr_offset(X86Reg::Rax) == 0);
    static_assert(dbt::runtime::gpr_offset(X86Reg::Rbx) == 24);
    static_assert(dbt::runtime::gpr_offset(X86Reg::R15) == 120);
    static_assert(dbt::runtime::kRipOffset == 128);
    static_assert(dbt::runtime::kLinkTableOffset == 144);
    static_assert(sizeof(CpuState) == 152);

    const CpuState state;
    EXPECT_EQ(state.rip, 0u);
    EXPECT_EQ(state.gpr[0], 0u);
    EXPECT_EQ(state.gpr.size(), 16u);
}

// --- Translation cache -----------------------------------------------------

TEST(TranslationCache, StartsEmpty) {
    const dbt::runtime::TranslationCache cache;
    EXPECT_TRUE(cache.empty());
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_EQ(cache.find(kBase), nullptr);
    EXPECT_EQ(cache.misses(), 1u);
}

TEST(TranslationCache, InsertThenFindReturnsTheSameBlock) {
    dbt::runtime::TranslationCache cache;
    const auto* inserted = cache.insert(kBase, dbt::runtime::TranslatedBlock{});
    ASSERT_NE(inserted, nullptr);
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_EQ(cache.find(kBase), inserted);
    EXPECT_EQ(cache.hits(), 1u);
}

TEST(TranslationCache, PointersSurviveGrowth) {
    // The dispatcher holds block pointers across insertions, so a rehash must
    // not invalidate them.
    dbt::runtime::TranslationCache cache;
    const auto* first = cache.insert(kBase, dbt::runtime::TranslatedBlock{});
    for (GuestAddr addr = kBase + 1; addr < kBase + 512; ++addr) {
        static_cast<void>(cache.insert(addr, dbt::runtime::TranslatedBlock{}));
    }
    EXPECT_EQ(cache.find(kBase), first);
}

TEST(TranslationCache, InvalidateAndClear) {
    dbt::runtime::TranslationCache cache;
    static_cast<void>(cache.insert(kBase, dbt::runtime::TranslatedBlock{}));
    static_cast<void>(cache.insert(kBase + 8, dbt::runtime::TranslatedBlock{}));

    EXPECT_TRUE(cache.invalidate(kBase));
    EXPECT_FALSE(cache.invalidate(kBase));  // already gone
    EXPECT_EQ(cache.size(), 1u);

    cache.clear();
    EXPECT_TRUE(cache.empty());
}

// --- Translation -----------------------------------------------------------

TEST(Dispatcher, TranslatesABlockAndSealsItExecutable) {
    Dispatcher dispatcher(kMovRet, kBase);
    const auto result = dispatcher.translate(kBase);

    ASSERT_TRUE(result.ok()) << dbt::runtime::to_string(result.status);
    ASSERT_NE(result.block, nullptr);
    EXPECT_FALSE(result.from_cache);
    EXPECT_EQ(result.block->guest_addr(), kBase);
    EXPECT_GT(result.block->word_count(), 0u);
    // W^X: the buffer is sealed before it is ever handed out.
    EXPECT_TRUE(result.block->executable());
    EXPECT_NE(result.block->code(), nullptr);
}

TEST(Dispatcher, SecondTranslationIsServedFromTheCache) {
    Dispatcher dispatcher(kMovRet, kBase);
    const auto first = dispatcher.translate(kBase);
    ASSERT_TRUE(first.ok());

    const auto second = dispatcher.translate(kBase);
    ASSERT_TRUE(second.ok());
    EXPECT_TRUE(second.from_cache);
    EXPECT_EQ(second.block, first.block) << "the block must not be recompiled";
    EXPECT_EQ(dispatcher.cache().size(), 1u);
}

TEST(Dispatcher, InvalidatingForcesRecompilation) {
    Dispatcher dispatcher(kMovRet, kBase);
    const auto first = dispatcher.translate(kBase);
    ASSERT_TRUE(first.ok());

    EXPECT_TRUE(dispatcher.cache().invalidate(kBase));
    const auto second = dispatcher.translate(kBase);
    ASSERT_TRUE(second.ok());
    EXPECT_FALSE(second.from_cache);
}

TEST(Dispatcher, ContainsTracksTheMappedRegion) {
    const Dispatcher dispatcher(kMovRet, kBase);
    EXPECT_TRUE(dispatcher.contains(kBase));
    EXPECT_TRUE(dispatcher.contains(kBase + 3));
    EXPECT_FALSE(dispatcher.contains(kBase + 4));  // one past the end
    EXPECT_FALSE(dispatcher.contains(kBase - 1));
    EXPECT_FALSE(dispatcher.contains(0));
}

TEST(Dispatcher, AddressOutsideTheRegionIsRejected) {
    Dispatcher dispatcher(kMovRet, kBase);

    const auto below = dispatcher.translate(kBase - 1);
    EXPECT_FALSE(below.ok());
    EXPECT_EQ(below.status, TranslateStatus::OutOfBounds);

    const auto above = dispatcher.translate(kBase + 4096);
    EXPECT_FALSE(above.ok());
    EXPECT_EQ(above.status, TranslateStatus::OutOfBounds);
}

TEST(Dispatcher, LiftFailurePropagates) {
    // 0x06 (PUSH ES) is not encodable in 64-bit mode.
    const std::array<u8, 2> bad{0x06, 0x00};
    Dispatcher dispatcher(bad, kBase);

    const auto result = dispatcher.translate(kBase);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status, TranslateStatus::LiftFailed);
    EXPECT_EQ(result.lift_error, dbt::frontend::LiftError::DecodeFailed);
    EXPECT_EQ(dispatcher.cache().size(), 0u) << "failed blocks must not be cached";
}

TEST(Dispatcher, CompileFailurePropagates) {
    // 66 89 D8  mov ax, bx -- lifts fine, but a 16-bit write merges into the
    // existing register rather than zero-extending, so the backend refuses it
    // instead of translating it with the wrong semantics.
    const std::array<u8, 4> narrow{0x66, 0x89, 0xD8, 0xC3};
    Dispatcher dispatcher(narrow, kBase);

    const auto result = dispatcher.translate(kBase);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status, TranslateStatus::CompileFailed);
    EXPECT_EQ(result.compile_error, dbt::backend::CompileError::UnsupportedWidth);
}

TEST(Dispatcher, DistinctAddressesGetDistinctBlocks) {
    // Two RETs back to back.
    const std::array<u8, 2> code{0xC3, 0xC3};
    Dispatcher dispatcher(code, kBase);

    const auto first = dispatcher.translate(kBase);
    const auto second = dispatcher.translate(kBase + 1);
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());
    EXPECT_NE(first.block, second.block);
    EXPECT_EQ(dispatcher.cache().size(), 2u);
}

// --- Block chaining --------------------------------------------------------

/// EB 00  jmp +0   (targets the RET two bytes on)
/// C3     ret
constexpr std::array<u8, 3> kJumpThenRet{0xEB, 0x00, 0xC3};

TEST(Dispatcher, ExitsLinkOnceTheirTargetIsTranslated) {
    Dispatcher dispatcher(kJumpThenRet, kBase);

    // The jump's exit has nowhere to go yet, so it stays unlinked.
    ASSERT_TRUE(dispatcher.translate(kBase).ok());
    EXPECT_EQ(dispatcher.linked_exits(), 0u);

    // Translating the target satisfies the waiting exit.
    ASSERT_TRUE(dispatcher.translate(kBase + 2).ok());
    EXPECT_EQ(dispatcher.linked_exits(), 1u);
}

TEST(Dispatcher, ExitLinksImmediatelyWhenTheTargetAlreadyExists) {
    Dispatcher dispatcher(kJumpThenRet, kBase);

    // Translate the target first; it is a RET, so it contributes no exits.
    ASSERT_TRUE(dispatcher.translate(kBase + 2).ok());
    EXPECT_EQ(dispatcher.linked_exits(), 0u);

    // Now the jump can be linked without ever going on the pending list.
    ASSERT_TRUE(dispatcher.translate(kBase).ok());
    EXPECT_EQ(dispatcher.linked_exits(), 1u);
}

TEST(Dispatcher, ChainedEntrySkipsThePrologue) {
    // The whole point of chaining: the predecessor leaves the guest registers
    // live, so the successor must not reload them.
    Dispatcher dispatcher(kJumpThenRet, kBase);
    const auto res = dispatcher.translate(kBase);
    ASSERT_TRUE(res.ok());
    ASSERT_NE(res.block, nullptr);

    const auto* entry = static_cast<const dbt::u8*>(res.block->code());
    const auto* chained = static_cast<const dbt::u8*>(res.block->chained_entry());
    EXPECT_EQ(chained - entry,
              static_cast<std::ptrdiff_t>(dbt::backend::kPrologueWords *
                                          dbt::kArm64InstSize));
}

TEST(Dispatcher, StatusNames) {
    EXPECT_EQ(dbt::runtime::to_string(TranslateStatus::OutOfBounds), "out-of-bounds");
    EXPECT_EQ(dbt::runtime::to_string(TranslateStatus::CompileFailed),
              "compile-failed");
    EXPECT_EQ(dbt::runtime::to_string(ExecStatus::Halted), "halted");
    EXPECT_EQ(dbt::runtime::to_string(ExecStatus::HostCannotExecute),
              "host-cannot-execute");
}

// --- Execution -------------------------------------------------------------

TEST(Dispatcher, RunRefusesOnHostsThatCannotExecuteArm64) {
    // Only meaningful where the refusal path is reachable. On an ARM64 host
    // run() really executes, and this fixture has no stack: CpuState starts
    // zeroed, so guest RSP is 0 and the RET lowering pops from a null pointer.
    // With no memory sandbox that is a genuine host segfault rather than a
    // translator bug -- see DispatcherExec below, which supplies a real stack.
    if (dbt::host_can_execute_arm64()) {
        GTEST_SKIP() << "host can execute ARM64; the refusal path is unreachable";
    }

    Dispatcher dispatcher(kMovRet, kBase);
    CpuState state;
    state.rip = kBase;

    // The critical safety property: never jump into foreign machine code.
    const auto result = dispatcher.run(state);
    EXPECT_EQ(result.status, ExecStatus::HostCannotExecute);
    EXPECT_EQ(result.steps, 0u);
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(DispatcherExec, MovBetweenRegistersRuns) {
    Dispatcher dispatcher(kMovRet, kBase);
    CpuState state;
    state.rip = kBase;
    state.gpr[reg_index(X86Reg::Rbx)] = 0xDEADBEEF;
    // RET pops the return address, so park one outside the code region to halt.
    std::array<dbt::u64, 1> stack{0xF000};
    state.gpr[reg_index(X86Reg::Rsp)] = reinterpret_cast<dbt::u64>(stack.data());

    const auto result = dispatcher.run(state);
    EXPECT_TRUE(result.ok()) << dbt::runtime::to_string(result.status);
    EXPECT_EQ(state.gpr[reg_index(X86Reg::Rax)], 0xDEADBEEFu);
    EXPECT_EQ(state.rip, 0xF000u);
}

TEST(DispatcherExec, AddUpdatesTheDestinationRegister) {
    // 48 01 D8  add rax, rbx | C3 ret
    const std::array<u8, 4> code{0x48, 0x01, 0xD8, 0xC3};
    Dispatcher dispatcher(code, kBase);

    CpuState state;
    state.rip = kBase;
    state.gpr[reg_index(X86Reg::Rax)] = 10;
    state.gpr[reg_index(X86Reg::Rbx)] = 32;
    std::array<dbt::u64, 1> stack{0xF000};
    state.gpr[reg_index(X86Reg::Rsp)] = reinterpret_cast<dbt::u64>(stack.data());

    const auto result = dispatcher.run(state);
    EXPECT_TRUE(result.ok()) << dbt::runtime::to_string(result.status);
    EXPECT_EQ(state.gpr[reg_index(X86Reg::Rax)], 42u);
}

TEST(DispatcherExec, ConditionalBranchPicksTheRightEdge) {
    // cmp rax, rbx ; je +1 ; ret  -- taken when the two registers are equal.
    const std::array<u8, 6> code{0x48, 0x39, 0xD8,  // cmp rax, rbx
                                 0x74, 0x01,        // je +1
                                 0xC3};             // ret
    Dispatcher dispatcher(code, kBase);

    CpuState state;
    state.rip = kBase;
    state.gpr[reg_index(X86Reg::Rax)] = 7;
    state.gpr[reg_index(X86Reg::Rbx)] = 7;

    const auto result = dispatcher.run(state);
    EXPECT_TRUE(result.ok()) << dbt::runtime::to_string(result.status);
    // Equal, so the taken edge wins: rip == kBase + 5 + 1.
    EXPECT_EQ(state.rip, kBase + 6);
}

TEST(DispatcherExec, BlocksAreReusedAcrossSteps) {
    Dispatcher dispatcher(kMovRet, kBase);
    CpuState state;
    state.rip = kBase;
    std::array<dbt::u64, 1> stack{0xF000};
    state.gpr[reg_index(X86Reg::Rsp)] = reinterpret_cast<dbt::u64>(stack.data());

    static_cast<void>(dispatcher.run(state));
    const dbt::usize compiled = dispatcher.cache().size();

    state.rip = kBase;
    state.gpr[reg_index(X86Reg::Rsp)] = reinterpret_cast<dbt::u64>(stack.data());
    static_cast<void>(dispatcher.run(state));
    EXPECT_EQ(dispatcher.cache().size(), compiled) << "no block should recompile";
}

TEST(DispatcherExec, StepBudgetStopsAnInfiniteLoop) {
    // EB FE  jmp -2 -- branches to itself forever.
    const std::array<u8, 2> code{0xEB, 0xFE};
    Dispatcher dispatcher(code, kBase, Dispatcher::Options{.max_steps = 8});

    CpuState state;
    state.rip = kBase;
    const auto result = dispatcher.run(state);
    EXPECT_EQ(result.status, ExecStatus::StepLimitReached);
    EXPECT_EQ(result.steps, 8u);
}
#endif

}  // namespace
