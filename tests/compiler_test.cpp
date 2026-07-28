#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <span>
#include <vector>

#include "dbt/backend/arm64_encoder.hpp"
#include "dbt/backend/compiler.hpp"
#include "dbt/backend/register_map.hpp"
#include "dbt/frontend/lifter.hpp"
#include "dbt/ir/ir.hpp"
#include "dbt/runtime/cpu_state.hpp"

namespace {

namespace a64 = dbt::backend::a64;
using a64::Reg;
using dbt::Arm64Word;
using dbt::u8;
using dbt::backend::CompileError;
using dbt::backend::Compiler;
using dbt::backend::CompileResult;
using dbt::backend::kPrologueWords;
using dbt::decoder::Cond;
using dbt::decoder::X86Reg;
using dbt::ir::BlockId;
using dbt::ir::Function;
using dbt::ir::InstId;
using dbt::ir::IRBuilder;

/// The instruction stream with the fixed prologue stripped off.
std::span<const Arm64Word> body(const CompileResult& result) {
    if (result.words.size() < kPrologueWords) {
        return {};
    }
    return std::span<const Arm64Word>(result.words).subspan(kPrologueWords);
}

bool contains(const CompileResult& result, Arm64Word word) {
    return std::find(result.words.begin(), result.words.end(), word) !=
           result.words.end();
}

CompileResult compile_ok(const Function& func) {
    const Compiler compiler;
    CompileResult result = compiler.compile(func);
    EXPECT_TRUE(result.ok()) << dbt::backend::to_string(result.error);
    return result;
}

/// Compiles guest bytes end to end: decode -> lift -> ARM64.
CompileResult compile_x86(std::span<const u8> code) {
    const dbt::frontend::Lifter lifter;
    const auto lifted = lifter.lift_block(code, 0x1000);
    EXPECT_TRUE(lifted.ok()) << dbt::frontend::to_string(lifted.error);
    return compile_ok(lifted.function);
}

/// Lifts guest bytes and reports the compiler's verdict without asserting it,
/// so refusals can be checked.
CompileError compile_x86_verdict(std::span<const u8> code) {
    const dbt::frontend::Lifter lifter;
    const auto lifted = lifter.lift_block(code, 0x1000);
    EXPECT_TRUE(lifted.ok()) << dbt::frontend::to_string(lifted.error);
    const Compiler compiler;
    return compiler.compile(lifted.function).error;
}

// --- Prologue and epilogue -------------------------------------------------

TEST(Compiler, PrologueSavesFrameAndLoadsGuestRegisters) {
    Function func;
    IRBuilder b(func);
    static_cast<void>(b.create_block(0x1000));
    b.ret();

    const auto result = compile_ok(func);
    ASSERT_GE(result.words.size(), kPrologueWords);

    EXPECT_EQ(result.words[0],
              a64::stp_pre(a64::kFramePointer, a64::kLinkReg, a64::kStackPointer, -48));
    EXPECT_EQ(result.words[1],
              a64::stp_off(Reg::X19, Reg::X20, a64::kStackPointer, 16));
    EXPECT_EQ(result.words[2], a64::str_imm(Reg::X28, a64::kStackPointer, 32));
    EXPECT_EQ(result.words[3], a64::mov_reg(a64::kFramePointer, a64::kStackPointer));
    // The CpuState pointer must be parked before x0 becomes guest RAX.
    EXPECT_EQ(result.words[4], a64::mov_reg(Reg::X28, Reg::X0));
    EXPECT_EQ(result.words[5], a64::ldr_imm(Reg::X0, Reg::X28, 0));
    EXPECT_EQ(result.words[6], a64::ldr_imm(Reg::X1, Reg::X28, 8));
    EXPECT_EQ(result.words[kPrologueWords - 1], a64::ldr_imm(Reg::X15, Reg::X28, 120));
}

TEST(Compiler, EpilogueSpillsGuestRegistersAndRestoresFrame) {
    Function func;
    IRBuilder b(func);
    static_cast<void>(b.create_block(0x1000));
    b.ret();

    const auto result = compile_ok(func);
    const auto tail = body(result);
    ASSERT_EQ(tail.size(), 20u);  // 16 spills + 3 restores + ret

    EXPECT_EQ(tail[0], a64::str_imm(Reg::X0, Reg::X28, 0));
    EXPECT_EQ(tail[15], a64::str_imm(Reg::X15, Reg::X28, 120));
    EXPECT_EQ(tail[16], a64::ldr_imm(Reg::X28, a64::kStackPointer, 32));
    EXPECT_EQ(tail[17], a64::ldp_off(Reg::X19, Reg::X20, a64::kStackPointer, 16));
    EXPECT_EQ(tail[18],
              a64::ldp_post(a64::kFramePointer, a64::kLinkReg, a64::kStackPointer, 48));
    EXPECT_EQ(tail[19], a64::ret());
}

TEST(Compiler, GuestRegisterOffsetsMatchCpuStateLayout) {
    // The emitted code hard-codes these offsets, so they must track the struct.
    static_assert(dbt::runtime::gpr_offset(X86Reg::Rax) == 0);
    static_assert(dbt::runtime::gpr_offset(X86Reg::R15) == 120);
    static_assert(dbt::runtime::kRipOffset == 128);
    SUCCEED();
}

// --- Data movement ---------------------------------------------------------

TEST(Compiler, LoadGuestRegisterEmitsNothing) {
    // The whole point of pinning guest registers: reading one is free.
    Function func;
    IRBuilder b(func);
    static_cast<void>(b.create_block());
    const InstId value = b.load_guest_reg(X86Reg::Rbx);
    b.store_guest_reg(X86Reg::Rax, value);
    b.ret();

    const auto result = compile_ok(func);
    const auto tail = body(result);
    ASSERT_FALSE(tail.empty());
    // A single register move, then the epilogue.
    EXPECT_EQ(tail[0], a64::mov_reg(Reg::X0, Reg::X3));  // rax <- rbx
}

TEST(Compiler, StoringARegisterToItselfEmitsNoMove) {
    Function func;
    IRBuilder b(func);
    static_cast<void>(b.create_block());
    const InstId value = b.load_guest_reg(X86Reg::Rax);
    b.store_guest_reg(X86Reg::Rax, value);
    b.ret();

    const auto result = compile_ok(func);
    const auto tail = body(result);
    ASSERT_FALSE(tail.empty());
    // Straight into the epilogue: no redundant `mov x0, x0`.
    EXPECT_EQ(tail[0], a64::str_imm(Reg::X0, Reg::X28, 0));
}

TEST(Compiler, SmallConstantUsesSingleMovz) {
    Function func;
    IRBuilder b(func);
    static_cast<void>(b.create_block());
    const InstId value = b.const_int(42);
    b.store_guest_reg(X86Reg::Rax, value);
    b.ret();

    const auto result = compile_ok(func);
    const auto tail = body(result);
    ASSERT_GE(tail.size(), 2u);
    EXPECT_EQ(tail[0], a64::movz(Reg::X16, 42));
    EXPECT_EQ(tail[1], a64::mov_reg(Reg::X0, Reg::X16));
}

TEST(Compiler, SmallNegativeConstantUsesMovn) {
    Function func;
    IRBuilder b(func);
    static_cast<void>(b.create_block());
    const InstId value = b.const_int(-1);
    b.store_guest_reg(X86Reg::Rax, value);
    b.ret();

    const auto result = compile_ok(func);
    const auto tail = body(result);
    ASSERT_FALSE(tail.empty());
    EXPECT_EQ(tail[0], a64::movn(Reg::X16, 0));  // ~0 == -1
}

TEST(Compiler, WideConstantUsesMovzThenMovk) {
    Function func;
    IRBuilder b(func);
    static_cast<void>(b.create_block());
    const InstId value = b.const_int(0x12345678);
    b.store_guest_reg(X86Reg::Rax, value);
    b.ret();

    const auto result = compile_ok(func);
    const auto tail = body(result);
    ASSERT_GE(tail.size(), 2u);
    EXPECT_EQ(tail[0], a64::movz(Reg::X16, 0x5678, 0));
    EXPECT_EQ(tail[1], a64::movk(Reg::X16, 0x1234, 16));
}

TEST(Compiler, MemoryAccessUsesLdrAndStr) {
    Function func;
    IRBuilder b(func);
    static_cast<void>(b.create_block());
    const InstId addr = b.load_guest_reg(X86Reg::Rbx);
    const InstId loaded = b.load_mem(addr);
    b.store_guest_reg(X86Reg::Rax, loaded);
    b.ret();

    const auto result = compile_ok(func);
    const auto tail = body(result);
    ASSERT_GE(tail.size(), 2u);
    EXPECT_EQ(tail[0], a64::ldr_imm(Reg::X16, Reg::X3, 0));
    EXPECT_EQ(tail[1], a64::mov_reg(Reg::X0, Reg::X16));
}

// --- Arithmetic and flags --------------------------------------------------

TEST(Compiler, AddUsesTheFlagSettingForm) {
    Function func;
    IRBuilder b(func);
    static_cast<void>(b.create_block());
    const InstId lhs = b.load_guest_reg(X86Reg::Rax);
    const InstId rhs = b.load_guest_reg(X86Reg::Rbx);
    const InstId sum = b.add(lhs, rhs);
    b.store_guest_reg(X86Reg::Rax, sum);
    b.ret();

    const auto result = compile_ok(func);
    const auto tail = body(result);
    ASSERT_GE(tail.size(), 2u);
    // ADDS, not ADD: x86 ADD always updates EFLAGS.
    EXPECT_EQ(tail[0], a64::add_reg(Reg::X16, Reg::X0, Reg::X3, true));
    EXPECT_EQ(tail[1], a64::mov_reg(Reg::X0, Reg::X16));
}

TEST(Compiler, AddressArithmeticUsesTheNonFlagSettingForm) {
    Function func;
    IRBuilder b(func);
    static_cast<void>(b.create_block());
    const InstId base = b.load_guest_reg(X86Reg::Rbx);
    const InstId index = b.load_guest_reg(X86Reg::Rdx);
    const InstId scaled = b.addr_shl(index, 3);
    const InstId addr = b.addr_add(base, scaled);
    const InstId loaded = b.load_mem(addr);
    b.store_guest_reg(X86Reg::Rax, loaded);
    b.ret();

    const auto result = compile_ok(func);
    const auto tail = body(result);
    ASSERT_GE(tail.size(), 2u);
    EXPECT_EQ(tail[0], a64::lsl_imm(Reg::X16, Reg::X2, 3));
    // Plain ADD -- the flag-setting form would clobber a pending comparison.
    EXPECT_EQ(tail[1], a64::add_reg(Reg::X17, Reg::X3, Reg::X16, false));
    EXPECT_NE(tail[1], a64::add_reg(Reg::X17, Reg::X3, Reg::X16, true));
}

TEST(Compiler, CmpEmitsSubsToTheZeroRegister) {
    Function func;
    IRBuilder b(func);
    const BlockId head = b.create_block();
    const BlockId exit = func.create_block();

    b.set_insert_point(head);
    const InstId lhs = b.load_guest_reg(X86Reg::Rax);
    const InstId rhs = b.load_guest_reg(X86Reg::Rbx);
    const InstId flags = b.cmp(lhs, rhs);
    b.branch(flags, Cond::Equal, exit, exit);
    b.set_insert_point(exit);
    b.ret();

    const auto result = compile_ok(func);
    EXPECT_TRUE(contains(result, a64::cmp_reg(Reg::X0, Reg::X3)));
}

// --- Control flow ----------------------------------------------------------

TEST(Compiler, UnconditionalJumpIsPatchedToTheTargetBlock) {
    Function func;
    IRBuilder b(func);
    const BlockId head = b.create_block();
    const BlockId target = func.create_block();

    b.set_insert_point(head);
    b.jump(target);
    b.set_insert_point(target);
    b.ret();

    const auto result = compile_ok(func);
    // The jump sits immediately after the prologue and the target starts on the
    // very next word, so the displacement is exactly one instruction.
    EXPECT_EQ(result.words[kPrologueWords], a64::b(4));
}

TEST(Compiler, ConditionalBranchEmitsBothEdges) {
    Function func;
    IRBuilder b(func);
    const BlockId head = b.create_block();
    const BlockId taken = func.create_block();
    const BlockId fall = func.create_block();

    b.set_insert_point(head);
    const InstId lhs = b.load_guest_reg(X86Reg::Rax);
    const InstId rhs = b.load_guest_reg(X86Reg::Rbx);
    const InstId flags = b.cmp(lhs, rhs);
    b.branch(flags, Cond::Equal, taken, fall);
    b.set_insert_point(taken);
    b.ret();
    b.set_insert_point(fall);
    b.ret();

    const auto result = compile_ok(func);
    const auto tail = body(result);
    ASSERT_GE(tail.size(), 3u);
    EXPECT_EQ(tail[0], a64::cmp_reg(Reg::X0, Reg::X3));
    // b.eq to the taken block, one instruction past the following b.
    EXPECT_EQ(tail[1], a64::b_cond(a64::Cond::EQ, 8));
    // The fallthrough block starts after the taken block's 20-word epilogue.
    EXPECT_EQ(tail[2], a64::b(4 + 20 * 4));
}

TEST(Compiler, BlockExitWritesNextRipIntoCpuState) {
    // RIP has no pinned register, so it must be stored straight to CpuState --
    // this is how the dispatcher learns where to resume.
    // EB 05  jmp +5
    const std::array<u8, 2> code{0xEB, 0x05};
    const auto result = compile_x86(code);

    EXPECT_TRUE(contains(result, a64::movz(Reg::X16, 0x1007)));
    EXPECT_TRUE(contains(result,
                         a64::str_imm(Reg::X16, Reg::X28,
                                      static_cast<dbt::u32>(dbt::runtime::kRipOffset))))
        << "the exit block must write CpuState.rip";
}

// --- The carry inversion, end to end ---------------------------------------

TEST(Compiler, JbAfterCmpBecomesCarryClear) {
    // cmp rax, rbx ; jb -- x86 CF==1 means borrow, which AArch64 spells C==0.
    Function func;
    IRBuilder b(func);
    const BlockId head = b.create_block();
    const BlockId exit = func.create_block();

    b.set_insert_point(head);
    const InstId lhs = b.load_guest_reg(X86Reg::Rax);
    const InstId rhs = b.load_guest_reg(X86Reg::Rbx);
    const InstId flags = b.cmp(lhs, rhs);
    b.branch(flags, Cond::Below, exit, exit);
    b.set_insert_point(exit);
    b.ret();

    const auto result = compile_ok(func);
    const auto tail = body(result);
    ASSERT_GE(tail.size(), 2u);
    EXPECT_EQ(tail[1], a64::b_cond(a64::Cond::CC, 8))
        << "JB after a comparison must use CC/LO";
    EXPECT_NE(tail[1], a64::b_cond(a64::Cond::CS, 8));
}

TEST(Compiler, JbAfterAddBecomesCarrySet) {
    // add rax, rbx ; jb -- after an addition both architectures agree that
    // carry means carry-out, so the inversion does not apply.
    Function func;
    IRBuilder b(func);
    const BlockId head = b.create_block();
    const BlockId exit = func.create_block();

    b.set_insert_point(head);
    const InstId lhs = b.load_guest_reg(X86Reg::Rax);
    const InstId rhs = b.load_guest_reg(X86Reg::Rbx);
    const InstId sum = b.add(lhs, rhs);
    b.store_guest_reg(X86Reg::Rax, sum);
    b.branch(sum, Cond::Below, exit, exit);
    b.set_insert_point(exit);
    b.ret();

    const auto result = compile_ok(func);
    EXPECT_TRUE(contains(result, a64::b_cond(a64::Cond::CS, 8)))
        << "JB after an addition must use CS, not CC";
    EXPECT_FALSE(contains(result, a64::b_cond(a64::Cond::CC, 8)));
}

// --- Refusals --------------------------------------------------------------

TEST(Compiler, ParityConditionsAreRefused) {
    Function func;
    IRBuilder b(func);
    const BlockId head = b.create_block();
    const BlockId exit = func.create_block();

    b.set_insert_point(head);
    const InstId lhs = b.load_guest_reg(X86Reg::Rax);
    const InstId rhs = b.load_guest_reg(X86Reg::Rbx);
    const InstId flags = b.cmp(lhs, rhs);
    b.branch(flags, Cond::Parity, exit, exit);
    b.set_insert_point(exit);
    b.ret();

    const Compiler compiler;
    const auto result = compiler.compile(func);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error, CompileError::UnsupportedCondition);
}

TEST(Compiler, NarrowWidthsAreRefusedRatherThanMistranslated) {
    Function func;
    IRBuilder b(func);
    static_cast<void>(b.create_block());
    // A 16-bit operation. Unlike a 32-bit write, which zero-extends and so maps
    // straight onto the W registers, a 16-bit write merges into the existing
    // register contents -- semantics the backend does not model.
    const InstId value = b.load_guest_reg(X86Reg::Rax, dbt::ir::Type::I16);
    b.store_guest_reg(X86Reg::Rbx, value);
    b.ret();

    const Compiler compiler;
    const auto result = compiler.compile(func);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error, CompileError::UnsupportedWidth);
}

TEST(Compiler, MalformedIrIsRejected) {
    const Function func;  // no blocks at all
    const Compiler compiler;
    const auto result = compiler.compile(func);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error, CompileError::InvalidIr);
}

TEST(Compiler, CodeBudgetIsEnforced) {
    Function func;
    IRBuilder b(func);
    static_cast<void>(b.create_block());
    b.ret();

    const Compiler bounded(Compiler::Options{.max_words = 4});
    const auto result = bounded.compile(func);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error, CompileError::CodeTooLarge);
}

TEST(Compiler, ErrorNames) {
    EXPECT_EQ(dbt::backend::to_string(CompileError::None), "none");
    EXPECT_EQ(dbt::backend::to_string(CompileError::UnsupportedCondition),
              "unsupported-condition");
    EXPECT_EQ(dbt::backend::to_string(CompileError::UnsupportedWidth),
              "unsupported-width");
}

// --- End to end: x86 bytes straight through to ARM64 -----------------------

TEST(Compiler, MovRegToRegEndToEnd) {
    // 48 89 D8  mov rax, rbx  |  C3  ret
    const std::array<u8, 4> code{0x48, 0x89, 0xD8, 0xC3};
    const auto result = compile_x86(code);
    EXPECT_TRUE(contains(result, a64::mov_reg(Reg::X0, Reg::X3)));
}

TEST(Compiler, AddRegToRegEndToEnd) {
    // 48 01 D8  add rax, rbx  |  C3
    const std::array<u8, 4> code{0x48, 0x01, 0xD8, 0xC3};
    const auto result = compile_x86(code);
    EXPECT_TRUE(contains(result, a64::add_reg(Reg::X16, Reg::X0, Reg::X3, true)));
}

TEST(Compiler, SubRegFromRegEndToEnd) {
    // 48 29 D8  sub rax, rbx  |  C3
    const std::array<u8, 4> code{0x48, 0x29, 0xD8, 0xC3};
    const auto result = compile_x86(code);
    EXPECT_TRUE(contains(result, a64::sub_reg(Reg::X16, Reg::X0, Reg::X3, true)));
}

TEST(Compiler, CmpThenJumpBelowEndToEnd) {
    // 48 39 D8  cmp rax, rbx  |  72 02  jb +2
    const std::array<u8, 5> code{0x48, 0x39, 0xD8, 0x72, 0x02};
    const auto result = compile_x86(code);

    EXPECT_TRUE(contains(result, a64::cmp_reg(Reg::X0, Reg::X3)));
    // The full pipeline must preserve the carry inversion: find any b.cc.
    const bool has_cc =
        std::any_of(result.words.begin(), result.words.end(), [](Arm64Word word) {
            return (word & 0xFF00001Fu) ==
                   (0x54000000u | static_cast<dbt::u32>(a64::Cond::CC));
        });
    EXPECT_TRUE(has_cc) << "jb must lower to b.cc after a comparison";
}

// --- Expanded instruction set, end to end ----------------------------------

TEST(Compiler, AndUsesTheFlagSettingLogicalForm) {
    // 48 21 D8  and rax, rbx  |  C3
    const std::array<u8, 4> code{0x48, 0x21, 0xD8, 0xC3};
    const auto result = compile_x86(code);
    // ANDS exists on AArch64, so a single instruction suffices.
    EXPECT_TRUE(contains(result, a64::and_reg(Reg::X16, Reg::X0, Reg::X3, true)));
}

TEST(Compiler, OrAndXorPublishFlagsWithAFollowingTst) {
    // 48 09 D8  or rax, rbx  |  C3
    const std::array<u8, 4> or_code{0x48, 0x09, 0xD8, 0xC3};
    const auto or_result = compile_x86(or_code);
    EXPECT_TRUE(contains(or_result, a64::orr_reg(Reg::X16, Reg::X0, Reg::X3)));
    // AArch64 has no ORRS, so the flags come from TST on the result.
    EXPECT_TRUE(contains(or_result, a64::tst_reg(Reg::X16, Reg::X16)));

    // 48 31 D8  xor rax, rbx  |  C3
    const std::array<u8, 4> xor_code{0x48, 0x31, 0xD8, 0xC3};
    const auto xor_result = compile_x86(xor_code);
    EXPECT_TRUE(contains(xor_result, a64::eor_reg(Reg::X16, Reg::X0, Reg::X3)));
    EXPECT_TRUE(contains(xor_result, a64::tst_reg(Reg::X16, Reg::X16)));
}

TEST(Compiler, TestLowersToTstWithNoDestination) {
    // 48 85 D8  test rax, rbx  |  C3
    const std::array<u8, 4> code{0x48, 0x85, 0xD8, 0xC3};
    const auto result = compile_x86(code);
    EXPECT_TRUE(contains(result, a64::tst_reg(Reg::X0, Reg::X3)));
}

TEST(Compiler, NotLowersToMvnAndSetsNoFlags) {
    // 48 F7 D0  not rax  |  C3
    const std::array<u8, 4> code{0x48, 0xF7, 0xD0, 0xC3};
    const auto result = compile_x86(code);
    EXPECT_TRUE(contains(result, a64::mvn_reg(Reg::X16, Reg::X0)));
    // MVN has no flag-setting form, which is exactly x86 NOT's behaviour.
    EXPECT_FALSE(contains(result, a64::tst_reg(Reg::X16, Reg::X16)));
}

TEST(Compiler, NegLowersToFlagSettingNegs) {
    // 48 F7 D8  neg rax  |  C3
    const std::array<u8, 4> code{0x48, 0xF7, 0xD8, 0xC3};
    const auto result = compile_x86(code);
    EXPECT_TRUE(contains(result, a64::neg_reg(Reg::X16, Reg::X0, true)));
}

TEST(Compiler, JbAfterNegUsesSubtractionCarrySemantics) {
    // neg rax ; jb +2 -- NEG subtracts from zero, so x86 CF is a borrow and
    // the ARM64 carry sense inverts, exactly as it does after SUB/CMP.
    const std::array<u8, 5> code{0x48, 0xF7, 0xD8, 0x72, 0x02};
    const auto result = compile_x86(code);

    const bool has_cc =
        std::any_of(result.words.begin(), result.words.end(), [](Arm64Word word) {
            return (word & 0xFF00001Fu) ==
                   (0x54000000u | static_cast<dbt::u32>(a64::Cond::CC));
        });
    EXPECT_TRUE(has_cc) << "jb after neg must lower to b.cc";
}

TEST(Compiler, PushLowersToAddressArithmeticAndAStore) {
    // 55  push rbp  |  C3 -- the guest stack pointer is x4, not the native SP.
    const std::array<u8, 2> code{0x55, 0xC3};
    const auto result = compile_x86(code);

    EXPECT_TRUE(contains(result, a64::movn(Reg::X16, 7)));  // -8
    EXPECT_TRUE(contains(result, a64::add_reg(Reg::X17, Reg::X4, Reg::X16, false)));
    EXPECT_TRUE(contains(result, a64::str_imm(Reg::X5, Reg::X17, 0)));
}

TEST(Compiler, CompilesAClangCompiledFunctionEndToEnd) {
    // long add(long a, long b) { return a + b; }  -- clang -O0, System V.
    // The milestone: a genuine compiled function, prologue through RET,
    // translated all the way to ARM64.
    const std::array<u8, 22> code{
        0x55,                    // push rbp
        0x48, 0x89, 0xE5,        // mov rbp, rsp
        0x48, 0x89, 0x7D, 0xF8,  // mov [rbp-8], rdi
        0x48, 0x89, 0x75, 0xF0,  // mov [rbp-16], rsi
        0x48, 0x8B, 0x45, 0xF8,  // mov rax, [rbp-8]
        0x48, 0x03, 0x45, 0xF0,  // add rax, [rbp-16]
        0x5D,                    // pop rbp
        0xC3                     // ret
    };

    const auto result = compile_x86(code);
    EXPECT_GT(result.words.size(), kPrologueWords);
    EXPECT_EQ(result.size_bytes() % dbt::kArm64InstSize, 0u);
}

// --- 32-bit widths ---------------------------------------------------------

TEST(Compiler, ThirtyTwoBitMoveUsesTheWRegisterForm) {
    // 89 D8  mov eax, ebx  |  C3 -- writing a W register zero-extends, which is
    // exactly x86's rule for a 32-bit destination.
    const std::array<u8, 3> code{0x89, 0xD8, 0xC3};
    const auto result = compile_x86(code);
    EXPECT_TRUE(contains(
        result, a64::with_width(a64::mov_reg(Reg::X0, Reg::X3), a64::Width::W32)));
}

TEST(Compiler, ThirtyTwoBitSelfMoveIsNotElided) {
    // 89 C0  mov eax, eax exists precisely to clear the upper half of RAX, so
    // unlike a 64-bit self-move it must still emit an instruction.
    const std::array<u8, 3> code{0x89, 0xC0, 0xC3};
    const auto result = compile_x86(code);
    EXPECT_TRUE(contains(
        result, a64::with_width(a64::mov_reg(Reg::X0, Reg::X0), a64::Width::W32)));
}

TEST(Compiler, ThirtyTwoBitArithmeticAndCompareUseWForms) {
    // 01 D8  add eax, ebx
    const std::array<u8, 3> add_code{0x01, 0xD8, 0xC3};
    const auto added = compile_x86(add_code);
    EXPECT_TRUE(contains(added,
                         a64::with_width(a64::add_reg(Reg::X16, Reg::X0, Reg::X3, true),
                                         a64::Width::W32)));

    // 39 D8  cmp eax, ebx -- comparing 64-bit here would let garbage in the
    // upper half change the result.
    const std::array<u8, 3> cmp_code{0x39, 0xD8, 0xC3};
    const auto compared = compile_x86(cmp_code);
    EXPECT_TRUE(contains(
        compared, a64::with_width(a64::cmp_reg(Reg::X0, Reg::X3), a64::Width::W32)));
}

TEST(Compiler, CompilesAnIntReturningClangFunction) {
    // int add(int a, int b) { return a + b; }  -- clang -O0, System V.
    // The 32-bit counterpart of the long/long milestone: ordinary C `int` code.
    const std::array<u8, 18> code{
        0x55,              // push rbp
        0x48, 0x89, 0xE5,  // mov rbp, rsp
        0x89, 0x7D, 0xFC,  // mov [rbp-4], edi
        0x89, 0x75, 0xF8,  // mov [rbp-8], esi
        0x8B, 0x45, 0xFC,  // mov eax, [rbp-4]
        0x03, 0x45, 0xF8,  // add eax, [rbp-8]
        0x5D,              // pop rbp
        0xC3               // ret
    };

    const auto result = compile_x86(code);
    EXPECT_GT(result.words.size(), kPrologueWords);

    // The 32-bit spills and reloads must use the W-register load/store forms.
    const bool has_w_load =
        std::any_of(result.words.begin(), result.words.end(), [](Arm64Word word) {
            return (word & 0xFFC00000u) == 0xB9400000u;
        });
    const bool has_w_store =
        std::any_of(result.words.begin(), result.words.end(), [](Arm64Word word) {
            return (word & 0xFFC00000u) == 0xB9000000u;
        });
    EXPECT_TRUE(has_w_load) << "32-bit reloads must use LDR Wt";
    EXPECT_TRUE(has_w_store) << "32-bit spills must use STR Wt";
}

// --- Shifts, steps, and the flag-validity gate -----------------------------

TEST(Compiler, ImmediateShiftLowersToShiftPlusTst) {
    // 48 C1 E0 03  shl rax, 3  |  C3
    const std::array<u8, 5> code{0x48, 0xC1, 0xE0, 0x03, 0xC3};
    const auto result = compile_x86(code);
    EXPECT_TRUE(contains(result, a64::lsl_imm(Reg::X16, Reg::X0, 3)));
    // AArch64 shifts set no flags, so ZF/SF are published separately.
    EXPECT_TRUE(contains(result, a64::tst_reg(Reg::X16, Reg::X16)));
}

TEST(Compiler, RightShiftsPickLogicalOrArithmetic) {
    const std::array<u8, 5> shr_code{0x48, 0xC1, 0xE8, 0x03, 0xC3};  // shr rax,3
    const std::array<u8, 5> sar_code{0x48, 0xC1, 0xF8, 0x03, 0xC3};  // sar rax,3
    EXPECT_TRUE(contains(compile_x86(shr_code), a64::lsr_imm(Reg::X16, Reg::X0, 3)));
    EXPECT_TRUE(contains(compile_x86(sar_code), a64::asr_imm(Reg::X16, Reg::X0, 3)));
}

TEST(Compiler, IncAndDecLowerToAddSubImmediateOne) {
    const std::array<u8, 4> inc_code{0x48, 0xFF, 0xC0, 0xC3};  // inc rax | ret
    const std::array<u8, 4> dec_code{0x48, 0xFF, 0xC8, 0xC3};  // dec rax | ret
    EXPECT_TRUE(
        contains(compile_x86(inc_code), a64::add_imm(Reg::X16, Reg::X0, 1, true)));
    EXPECT_TRUE(
        contains(compile_x86(dec_code), a64::sub_imm(Reg::X16, Reg::X0, 1, true)));
}

TEST(Compiler, CarryTestAfterAShiftIsRefused) {
    // shr rax, 1 ; jb +2
    // x86 sets CF from the bit shifted out. Our LSR + TST lowering cannot
    // reproduce that, so branching on carry here must be refused rather than
    // reading whatever NZCV happens to hold -- a silent wrong branch otherwise.
    const std::array<u8, 6> code{0x48, 0xC1, 0xE8, 0x01, 0x72, 0x02};
    EXPECT_EQ(compile_x86_verdict(code), CompileError::UnsupportedCondition);
}

TEST(Compiler, ZeroTestAfterAShiftIsAllowed) {
    // shr rax, 1 ; je +2 -- ZF genuinely survives the lowering, so this is
    // translated rather than refused.
    const std::array<u8, 6> code{0x48, 0xC1, 0xE8, 0x01, 0x74, 0x02};
    EXPECT_EQ(compile_x86_verdict(code), CompileError::None);
}

TEST(Compiler, CarryTestAfterIncIsRefused) {
    // inc rax ; jb +2 -- x86 preserves CF across INC on purpose; ADDS
    // overwrites it, so the old carry is gone and the branch is refused.
    const std::array<u8, 5> code{0x48, 0xFF, 0xC0, 0x72, 0x02};
    EXPECT_EQ(compile_x86_verdict(code), CompileError::UnsupportedCondition);

    // The same INC followed by a zero test is fine.
    const std::array<u8, 5> zero_code{0x48, 0xFF, 0xC0, 0x74, 0x02};
    EXPECT_EQ(compile_x86_verdict(zero_code), CompileError::None);
}

TEST(Compiler, CarryTestAfterACompareIsStillAllowed) {
    // The gate must not over-refuse: CMP defines every modelled flag.
    const std::array<u8, 5> code{0x48, 0x39, 0xD8, 0x72, 0x02};  // cmp rax,rbx; jb
    EXPECT_EQ(compile_x86_verdict(code), CompileError::None);
}

// --- Extending moves -------------------------------------------------------

TEST(Compiler, MovzxLowersToUnsignedExtension) {
    // 0F B6 C3  movzx eax, bl  |  C3 -- bl is the low byte of rbx (x3).
    const std::array<u8, 4> byte_code{0x0F, 0xB6, 0xC3, 0xC3};
    EXPECT_TRUE(contains(compile_x86(byte_code), a64::uxtb(Reg::X16, Reg::X3)));

    // 0F B7 C3  movzx eax, bx
    const std::array<u8, 4> word_code{0x0F, 0xB7, 0xC3, 0xC3};
    EXPECT_TRUE(contains(compile_x86(word_code), a64::uxth(Reg::X16, Reg::X3)));
}

TEST(Compiler, MovsxLowersToSignedExtensionAtTheDestinationWidth) {
    // 48 0F BE C3  movsx rax, bl -- 64-bit destination.
    const std::array<u8, 5> wide{0x48, 0x0F, 0xBE, 0xC3, 0xC3};
    EXPECT_TRUE(contains(compile_x86(wide),
                         a64::sxtb(Reg::X16, Reg::X3, a64::Width::X64)));

    // 0F BE C3  movsx eax, bl -- a 32-bit destination uses the W form.
    const std::array<u8, 4> narrow{0x0F, 0xBE, 0xC3, 0xC3};
    EXPECT_TRUE(contains(compile_x86(narrow),
                         a64::sxtb(Reg::X16, Reg::X3, a64::Width::W32)));
}

TEST(Compiler, MovsxdLowersToSxtw) {
    // 48 63 C3  movsxd rax, ebx  |  C3
    const std::array<u8, 4> code{0x48, 0x63, 0xC3, 0xC3};
    EXPECT_TRUE(contains(compile_x86(code), a64::sxtw(Reg::X16, Reg::X3)));
}

TEST(Compiler, ByteMemorySourceUsesLdrb) {
    // 0F B6 03  movzx eax, byte [rbx]  |  C3
    const std::array<u8, 4> code{0x0F, 0xB6, 0x03, 0xC3};
    EXPECT_TRUE(contains(compile_x86(code), a64::ldrb(Reg::X16, Reg::X3)));
}

TEST(Compiler, NarrowRegisterWritesAreStillRefused) {
    // 88 D8  mov al, bl -- an 8-bit register write merges into the existing
    // register rather than zero-extending, which is not modelled.
    const std::array<u8, 3> code{0x88, 0xD8, 0xC3};
    EXPECT_EQ(compile_x86_verdict(code), CompileError::UnsupportedWidth);
}

// --- Indirect branches -----------------------------------------------------

TEST(Compiler, IndirectJumpWritesTheTargetStraightIntoCpuStateRip) {
    // FF E0  jmp rax -- rax is pinned to x0, so the target needs no move.
    const std::array<u8, 2> code{0xFF, 0xE0};
    const auto result = compile_x86(code);
    EXPECT_TRUE(contains(
        result, a64::str_imm(Reg::X0, Reg::X28,
                             static_cast<dbt::u32>(dbt::runtime::kRipOffset))));
}

TEST(Compiler, IndirectCallCompilesToPushPlusRipPublication) {
    // FF D0  call rax
    const std::array<u8, 2> code{0xFF, 0xD0};
    const auto result = compile_x86(code);
    EXPECT_GT(result.words.size(), kPrologueWords);
    EXPECT_TRUE(contains(
        result, a64::str_imm(Reg::X0, Reg::X28,
                             static_cast<dbt::u32>(dbt::runtime::kRipOffset))));
}

TEST(Compiler, EveryEmittedWordIsFourBytes) {
    const std::array<u8, 4> code{0x48, 0x01, 0xD8, 0xC3};
    const auto result = compile_x86(code);
    EXPECT_EQ(result.size_bytes(), result.words.size() * 4);
    EXPECT_EQ(result.size_bytes() % dbt::kArm64InstSize, 0u);
}

}  // namespace
