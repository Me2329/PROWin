#include <gtest/gtest.h>

#include <array>
#include <random>
#include <span>
#include <vector>

#include "dbt/common/types.hpp"
#include "dbt/decoder/decoder.hpp"

namespace {

using dbt::GuestAddr;
using dbt::u8;
using dbt::decoder::Cond;
using dbt::decoder::DecodeError;
using dbt::decoder::Decoder;
using dbt::decoder::Mnemonic;
using dbt::decoder::OperandKind;
using dbt::decoder::X86Reg;

/// Arbitrary but non-zero guest address, so branch-target arithmetic that
/// forgets to add the instruction address shows up as a mismatch.
constexpr GuestAddr kBase = 0x1000;

class DecoderTest : public ::testing::Test {
protected:
    Decoder dec;
};

// --- MOV -------------------------------------------------------------------

TEST_F(DecoderTest, MovRegToReg) {
    // 48 89 D8  mov rax, rbx
    const std::array<u8, 3> code{0x48, 0x89, 0xD8};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Mov);
    EXPECT_EQ(res.inst.length, 3);
    EXPECT_EQ(res.inst.address, kBase);
    EXPECT_EQ(res.inst.next_address(), kBase + 3);
    ASSERT_EQ(res.inst.operand_count, 2);

    EXPECT_EQ(res.inst.op(0).kind, OperandKind::Register);
    EXPECT_EQ(res.inst.op(0).reg, X86Reg::Rax);
    EXPECT_EQ(res.inst.op(0).size_bits, 64);

    EXPECT_EQ(res.inst.op(1).kind, OperandKind::Register);
    EXPECT_EQ(res.inst.op(1).reg, X86Reg::Rbx);
    EXPECT_EQ(res.inst.op(1).size_bits, 64);

    EXPECT_FALSE(res.inst.is_terminator());
}

TEST_F(DecoderTest, MovImm32ToReg) {
    // 48 C7 C0 2A 00 00 00  mov rax, 42
    const std::array<u8, 7> code{0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Mov);
    EXPECT_EQ(res.inst.length, 7);
    ASSERT_EQ(res.inst.operand_count, 2);
    EXPECT_EQ(res.inst.op(0).reg, X86Reg::Rax);
    EXPECT_EQ(res.inst.op(1).kind, OperandKind::Immediate);
    EXPECT_EQ(res.inst.op(1).imm, 42);
}

TEST_F(DecoderTest, MovImm64ToRegSignExtendsCorrectly) {
    // 48 B8 FF FF FF FF FF FF FF FF  movabs rax, -1
    const std::array<u8, 10> code{0x48, 0xB8, 0xFF, 0xFF, 0xFF,
                                  0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Mov);
    EXPECT_EQ(res.inst.length, 10);
    ASSERT_EQ(res.inst.operand_count, 2);
    EXPECT_EQ(res.inst.op(1).kind, OperandKind::Immediate);
    EXPECT_EQ(res.inst.op(1).imm, -1);
}

TEST_F(DecoderTest, MovLoadFromMemory) {
    // 48 8B 03  mov rax, [rbx]
    const std::array<u8, 3> code{0x48, 0x8B, 0x03};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Mov);
    ASSERT_EQ(res.inst.operand_count, 2);
    EXPECT_EQ(res.inst.op(0).kind, OperandKind::Register);
    EXPECT_EQ(res.inst.op(0).reg, X86Reg::Rax);

    ASSERT_EQ(res.inst.op(1).kind, OperandKind::Memory);
    EXPECT_EQ(res.inst.op(1).mem.base, X86Reg::Rbx);
    EXPECT_EQ(res.inst.op(1).mem.index, X86Reg::None);
    EXPECT_EQ(res.inst.op(1).mem.disp, 0);
    EXPECT_FALSE(res.inst.op(1).mem.rip_relative);
    EXPECT_EQ(res.inst.op(1).size_bits, 64);
}

TEST_F(DecoderTest, MovStoreToMemoryWithDisplacement) {
    // 48 89 43 08  mov [rbx+8], rax
    const std::array<u8, 4> code{0x48, 0x89, 0x43, 0x08};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Mov);
    EXPECT_EQ(res.inst.length, 4);
    ASSERT_EQ(res.inst.operand_count, 2);

    ASSERT_EQ(res.inst.op(0).kind, OperandKind::Memory);
    EXPECT_EQ(res.inst.op(0).mem.base, X86Reg::Rbx);
    EXPECT_EQ(res.inst.op(0).mem.disp, 8);

    EXPECT_EQ(res.inst.op(1).kind, OperandKind::Register);
    EXPECT_EQ(res.inst.op(1).reg, X86Reg::Rax);
}

TEST_F(DecoderTest, MovWithScaledIndex) {
    // 48 8B 04 D3  mov rax, [rbx + rdx*8]
    const std::array<u8, 4> code{0x48, 0x8B, 0x04, 0xD3};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    ASSERT_EQ(res.inst.operand_count, 2);
    ASSERT_EQ(res.inst.op(1).kind, OperandKind::Memory);
    EXPECT_EQ(res.inst.op(1).mem.base, X86Reg::Rbx);
    EXPECT_EQ(res.inst.op(1).mem.index, X86Reg::Rdx);
    EXPECT_EQ(res.inst.op(1).mem.scale, 8);
}

TEST_F(DecoderTest, MovRipRelativeResolvesAbsoluteTarget) {
    // 48 8B 05 10 00 00 00  mov rax, [rip+0x10]
    const std::array<u8, 7> code{0x48, 0x8B, 0x05, 0x10, 0x00, 0x00, 0x00};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.length, 7);
    ASSERT_EQ(res.inst.operand_count, 2);
    ASSERT_EQ(res.inst.op(1).kind, OperandKind::Memory);
    EXPECT_TRUE(res.inst.op(1).mem.rip_relative);
    EXPECT_EQ(res.inst.op(1).mem.base, X86Reg::Rip);
    EXPECT_EQ(res.inst.op(1).mem.disp, 0x10);

    // RIP is the address of the *next* instruction.
    EXPECT_TRUE(res.inst.has_rip_target);
    EXPECT_EQ(res.inst.rip_target, kBase + 7 + 0x10);
}

TEST_F(DecoderTest, MovExtendedRegistersDecodeThroughRexBits) {
    // 4D 89 C1  mov r9, r8
    const std::array<u8, 3> code{0x4D, 0x89, 0xC1};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    ASSERT_EQ(res.inst.operand_count, 2);
    EXPECT_EQ(res.inst.op(0).reg, X86Reg::R9);
    EXPECT_EQ(res.inst.op(1).reg, X86Reg::R8);
}

TEST_F(DecoderTest, MovDword32BitOperandSize) {
    // 89 D8  mov eax, ebx
    const std::array<u8, 2> code{0x89, 0xD8};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.length, 2);
    ASSERT_EQ(res.inst.operand_count, 2);
    EXPECT_EQ(res.inst.op(0).size_bits, 32);
    EXPECT_EQ(res.inst.op(0).reg, X86Reg::Rax);
}

// --- ADD / SUB / CMP -------------------------------------------------------

TEST_F(DecoderTest, AddRegToReg) {
    // 48 01 D8  add rax, rbx
    const std::array<u8, 3> code{0x48, 0x01, 0xD8};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Add);
    ASSERT_EQ(res.inst.operand_count, 2);
    EXPECT_EQ(res.inst.op(0).reg, X86Reg::Rax);
    EXPECT_EQ(res.inst.op(1).reg, X86Reg::Rbx);
    EXPECT_FALSE(res.inst.is_terminator());
}

TEST_F(DecoderTest, AddImm8ToRegSignExtends) {
    // 48 83 C0 FB  add rax, -5
    const std::array<u8, 4> code{0x48, 0x83, 0xC0, 0xFB};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Add);
    EXPECT_EQ(res.inst.length, 4);
    ASSERT_EQ(res.inst.operand_count, 2);
    EXPECT_EQ(res.inst.op(0).reg, X86Reg::Rax);
    EXPECT_EQ(res.inst.op(1).kind, OperandKind::Immediate);
    EXPECT_EQ(res.inst.op(1).imm, -5);
}

TEST_F(DecoderTest, SubRegFromReg) {
    // 48 29 D8  sub rax, rbx
    const std::array<u8, 3> code{0x48, 0x29, 0xD8};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Sub);
    ASSERT_EQ(res.inst.operand_count, 2);
    EXPECT_EQ(res.inst.op(0).reg, X86Reg::Rax);
    EXPECT_EQ(res.inst.op(1).reg, X86Reg::Rbx);
}

TEST_F(DecoderTest, CmpRegWithReg) {
    // 48 39 D8  cmp rax, rbx
    const std::array<u8, 3> code{0x48, 0x39, 0xD8};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Cmp);
    ASSERT_EQ(res.inst.operand_count, 2);
    EXPECT_EQ(res.inst.op(0).reg, X86Reg::Rax);
    EXPECT_EQ(res.inst.op(1).reg, X86Reg::Rbx);
    EXPECT_FALSE(res.inst.is_terminator());
}

TEST_F(DecoderTest, CmpRegWithImmediate) {
    // 48 83 F8 0A  cmp rax, 10
    const std::array<u8, 4> code{0x48, 0x83, 0xF8, 0x0A};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Cmp);
    ASSERT_EQ(res.inst.operand_count, 2);
    EXPECT_EQ(res.inst.op(1).kind, OperandKind::Immediate);
    EXPECT_EQ(res.inst.op(1).imm, 10);
}

// --- Control flow ----------------------------------------------------------

TEST_F(DecoderTest, JmpRel8ComputesTargetFromNextAddress) {
    // EB 05  jmp +5
    const std::array<u8, 2> code{0xEB, 0x05};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Jmp);
    EXPECT_EQ(res.inst.length, 2);
    EXPECT_TRUE(res.inst.has_branch_target);
    EXPECT_EQ(res.inst.branch_target, kBase + 2 + 5);
    EXPECT_TRUE(res.inst.is_terminator());
}

TEST_F(DecoderTest, JmpRel8NegativeDisplacement) {
    // EB FE  jmp -2  (branch to self)
    const std::array<u8, 2> code{0xEB, 0xFE};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Jmp);
    EXPECT_TRUE(res.inst.has_branch_target);
    EXPECT_EQ(res.inst.branch_target, kBase);
}

TEST_F(DecoderTest, JmpRel32) {
    // E9 00 01 00 00  jmp +0x100
    const std::array<u8, 5> code{0xE9, 0x00, 0x01, 0x00, 0x00};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Jmp);
    EXPECT_EQ(res.inst.length, 5);
    EXPECT_EQ(res.inst.branch_target, kBase + 5 + 0x100);
}

TEST_F(DecoderTest, JeRel8CarriesEqualCondition) {
    // 74 10  je +0x10
    const std::array<u8, 2> code{0x74, 0x10};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Jcc);
    EXPECT_EQ(res.inst.cond, Cond::Equal);
    EXPECT_TRUE(res.inst.has_branch_target);
    EXPECT_EQ(res.inst.branch_target, kBase + 2 + 0x10);
    EXPECT_TRUE(res.inst.is_terminator());
}

TEST_F(DecoderTest, JneRel32CarriesNotEqualCondition) {
    // 0F 85 20 00 00 00  jne +0x20
    const std::array<u8, 6> code{0x0F, 0x85, 0x20, 0x00, 0x00, 0x00};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Jcc);
    EXPECT_EQ(res.inst.cond, Cond::NotEqual);
    EXPECT_EQ(res.inst.length, 6);
    EXPECT_EQ(res.inst.branch_target, kBase + 6 + 0x20);
}

TEST_F(DecoderTest, SignedAndUnsignedConditionsAreDistinct) {
    // 7C 02  jl +2   (signed less)
    const std::array<u8, 2> jl{0x7C, 0x02};
    const auto res_jl = dec.decode(jl, kBase);
    ASSERT_TRUE(res_jl.ok()) << res_jl.error;
    EXPECT_EQ(res_jl.inst.cond, Cond::Less);

    // 72 02  jb +2   (unsigned below)
    const std::array<u8, 2> jb{0x72, 0x02};
    const auto res_jb = dec.decode(jb, kBase);
    ASSERT_TRUE(res_jb.ok()) << res_jb.error;
    EXPECT_EQ(res_jb.inst.cond, Cond::Below);
}

TEST_F(DecoderTest, Ret) {
    const std::array<u8, 1> code{0xC3};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Ret);
    EXPECT_EQ(res.inst.length, 1);
    EXPECT_TRUE(res.inst.is_terminator());
    EXPECT_FALSE(res.inst.has_branch_target);
}

// --- Bounds and error handling ---------------------------------------------

TEST_F(DecoderTest, EmptyBufferReportsEmptyBuffer) {
    const auto res = dec.decode(std::span<const u8>{}, kBase);
    EXPECT_FALSE(res.ok());
    EXPECT_EQ(res.error, DecodeError::EmptyBuffer);
}

TEST_F(DecoderTest, TruncatedInstructionDoesNotReadPastTheSpan) {
    // `48 8B` needs a ModRM byte that the buffer does not contain.
    const std::array<u8, 2> code{0x48, 0x8B};
    const auto res = dec.decode(code, kBase);
    EXPECT_FALSE(res.ok());
    EXPECT_EQ(res.error, DecodeError::TruncatedInstruction);
}

TEST_F(DecoderTest, TruncatedImmediateReportsTruncation) {
    // `mov rax, imm32` needs 7 bytes; only 5 are supplied.
    const std::array<u8, 5> code{0x48, 0xC7, 0xC0, 0x2A, 0x00};
    const auto res = dec.decode(code, kBase);
    EXPECT_FALSE(res.ok());
    EXPECT_EQ(res.error, DecodeError::TruncatedInstruction);
}

TEST_F(DecoderTest, IllegalOpcodeReportsInvalidInstruction) {
    // 0x06 (PUSH ES) is not encodable in 64-bit mode.
    const std::array<u8, 4> code{0x06, 0x00, 0x00, 0x00};
    const auto res = dec.decode(code, kBase);
    EXPECT_FALSE(res.ok());
    EXPECT_EQ(res.error, DecodeError::InvalidInstruction);
}

TEST_F(DecoderTest, LegalButOutOfScopeInstructionReportsUnsupported) {
    // 90  nop -- valid, but outside the translated subset.
    const std::array<u8, 1> code{0x90};
    const auto res = dec.decode(code, kBase);
    EXPECT_FALSE(res.ok());
    EXPECT_EQ(res.error, DecodeError::UnsupportedInstruction);
}

TEST_F(DecoderTest, DecodeStopsAtInstructionBoundaryNotBufferEnd) {
    // A 3-byte MOV followed by unrelated trailing bytes.
    const std::array<u8, 8> code{0x48, 0x89, 0xD8, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Mov);
    EXPECT_EQ(res.inst.length, 3);
}

TEST_F(DecoderTest, DecodedLengthNeverExceedsArchitecturalMaximum) {
    // 15 prefix bytes then an opcode: longer than any legal instruction.
    std::array<u8, 16> code{};
    code.fill(0x66);
    code[15] = 0x90;
    const auto res = dec.decode(code, kBase);

    if (res.ok()) {
        EXPECT_LE(res.inst.length, dbt::kMaxX86InstLength);
    } else {
        EXPECT_NE(res.error, DecodeError::None);
    }
}

TEST_F(DecoderTest, ArbitraryBytesNeverProduceOutOfBoundsOrOverlongDecodes) {
    // Deterministic pseudo-random sweep: the decoder must always terminate with
    // either a well-formed result whose length fits inside the supplied buffer,
    // or an explicit error. Run under ASan/UBSan this also proves no read runs
    // past the end of the span.
    std::mt19937 rng(0xDB7u);
    std::uniform_int_distribution<int> byte_dist(0, 255);

    for (int iteration = 0; iteration < 20000; ++iteration) {
        const std::size_t len = 1 + static_cast<std::size_t>(iteration % 15);
        std::vector<u8> buffer(len);
        for (u8& b : buffer) {
            b = static_cast<u8>(byte_dist(rng));
        }

        const auto res = dec.decode(buffer, kBase);
        if (res.ok()) {
            EXPECT_GE(res.inst.length, 1);
            EXPECT_LE(static_cast<std::size_t>(res.inst.length), buffer.size());
            EXPECT_LE(res.inst.length, dbt::kMaxX86InstLength);
            EXPECT_LE(res.inst.operand_count, dbt::decoder::kMaxOperands);
        }
    }
}

// --- Diagnostics -----------------------------------------------------------

TEST_F(DecoderTest, ToStringCoversTheSupportedSubset) {
    EXPECT_EQ(dbt::decoder::to_string(Mnemonic::Mov), "mov");
    EXPECT_EQ(dbt::decoder::to_string(Mnemonic::Add), "add");
    EXPECT_EQ(dbt::decoder::to_string(Mnemonic::Sub), "sub");
    EXPECT_EQ(dbt::decoder::to_string(Mnemonic::Cmp), "cmp");
    EXPECT_EQ(dbt::decoder::to_string(Mnemonic::Jmp), "jmp");
    EXPECT_EQ(dbt::decoder::to_string(Mnemonic::Ret), "ret");
    EXPECT_EQ(dbt::decoder::to_string(X86Reg::Rax), "rax");
    EXPECT_EQ(dbt::decoder::to_string(X86Reg::R15), "r15");
    EXPECT_EQ(dbt::decoder::to_string(Cond::Equal), "e");
}

// --- Expanded instruction set ----------------------------------------------

TEST_F(DecoderTest, PushAndPopDecode) {
    const std::array<u8, 1> push_rbp{0x55};
    const auto pushed = dec.decode(push_rbp, kBase);
    ASSERT_TRUE(pushed.ok()) << pushed.error;
    EXPECT_EQ(pushed.inst.mnemonic, Mnemonic::Push);
    ASSERT_EQ(pushed.inst.operand_count, 1);
    EXPECT_EQ(pushed.inst.op(0).reg, X86Reg::Rbp);
    EXPECT_FALSE(pushed.inst.is_terminator());

    const std::array<u8, 1> pop_rbp{0x5D};
    const auto popped = dec.decode(pop_rbp, kBase);
    ASSERT_TRUE(popped.ok()) << popped.error;
    EXPECT_EQ(popped.inst.mnemonic, Mnemonic::Pop);
    EXPECT_EQ(popped.inst.op(0).reg, X86Reg::Rbp);
}

TEST_F(DecoderTest, CallEndsABlockAndCarriesItsTarget) {
    // E8 10 00 00 00  call +0x10
    const std::array<u8, 5> code{0xE8, 0x10, 0x00, 0x00, 0x00};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Call);
    EXPECT_TRUE(res.inst.is_terminator());
    EXPECT_TRUE(res.inst.has_branch_target);
    EXPECT_EQ(res.inst.branch_target, kBase + 5 + 0x10);
}

TEST_F(DecoderTest, LeaKeepsItsMemoryOperandUndereferenced) {
    // 48 8D 44 1A 08  lea rax, [rdx + rbx*1 + 8]
    const std::array<u8, 5> code{0x48, 0x8D, 0x44, 0x1A, 0x08};
    const auto res = dec.decode(code, kBase);

    ASSERT_TRUE(res.ok()) << res.error;
    EXPECT_EQ(res.inst.mnemonic, Mnemonic::Lea);
    ASSERT_EQ(res.inst.operand_count, 2);
    EXPECT_EQ(res.inst.op(0).reg, X86Reg::Rax);
    ASSERT_EQ(res.inst.op(1).kind, OperandKind::Memory);
    EXPECT_EQ(res.inst.op(1).mem.base, X86Reg::Rdx);
    EXPECT_EQ(res.inst.op(1).mem.index, X86Reg::Rbx);
    EXPECT_EQ(res.inst.op(1).mem.disp, 8);
}

TEST_F(DecoderTest, LogicalInstructionsDecode) {
    const std::array<u8, 3> and_rax{0x48, 0x21, 0xD8};
    const std::array<u8, 3> or_rax{0x48, 0x09, 0xD8};
    const std::array<u8, 3> xor_rax{0x48, 0x31, 0xD8};
    const std::array<u8, 3> test_rax{0x48, 0x85, 0xD8};
    const std::array<u8, 3> not_rax{0x48, 0xF7, 0xD0};
    const std::array<u8, 3> neg_rax{0x48, 0xF7, 0xD8};

    EXPECT_EQ(dec.decode(and_rax, kBase).inst.mnemonic, Mnemonic::And);
    EXPECT_EQ(dec.decode(or_rax, kBase).inst.mnemonic, Mnemonic::Or);
    EXPECT_EQ(dec.decode(xor_rax, kBase).inst.mnemonic, Mnemonic::Xor);
    EXPECT_EQ(dec.decode(test_rax, kBase).inst.mnemonic, Mnemonic::Test);
    EXPECT_EQ(dec.decode(not_rax, kBase).inst.mnemonic, Mnemonic::Not);
    EXPECT_EQ(dec.decode(neg_rax, kBase).inst.mnemonic, Mnemonic::Neg);

    // NOT and NEG are single-operand forms.
    const auto negated = dec.decode(neg_rax, kBase);
    ASSERT_TRUE(negated.ok()) << negated.error;
    ASSERT_EQ(negated.inst.operand_count, 1);
    EXPECT_EQ(negated.inst.op(0).reg, X86Reg::Rax);
}

TEST_F(DecoderTest, ShiftsAndStepsDecode) {
    // 48 C1 E0 03  shl rax, 3   (C1 /4 ib)
    const std::array<u8, 4> shl{0x48, 0xC1, 0xE0, 0x03};
    const auto shifted = dec.decode(shl, kBase);
    ASSERT_TRUE(shifted.ok()) << shifted.error;
    EXPECT_EQ(shifted.inst.mnemonic, Mnemonic::Shl);
    ASSERT_EQ(shifted.inst.operand_count, 2);
    EXPECT_EQ(shifted.inst.op(0).reg, X86Reg::Rax);
    EXPECT_EQ(shifted.inst.op(1).kind, OperandKind::Immediate);
    EXPECT_EQ(shifted.inst.op(1).imm, 3);

    const std::array<u8, 4> shr{0x48, 0xC1, 0xE8, 0x03};  // shr rax, 3
    const std::array<u8, 4> sar{0x48, 0xC1, 0xF8, 0x03};  // sar rax, 3
    EXPECT_EQ(dec.decode(shr, kBase).inst.mnemonic, Mnemonic::Shr);
    EXPECT_EQ(dec.decode(sar, kBase).inst.mnemonic, Mnemonic::Sar);

    const std::array<u8, 3> inc_rax{0x48, 0xFF, 0xC0};  // inc rax
    const std::array<u8, 3> dec_rax{0x48, 0xFF, 0xC8};  // dec rax
    EXPECT_EQ(dec.decode(inc_rax, kBase).inst.mnemonic, Mnemonic::Inc);
    EXPECT_EQ(dec.decode(dec_rax, kBase).inst.mnemonic, Mnemonic::Dec);
}

TEST_F(DecoderTest, ExtendingMovesDecodeWithTheirSourceWidth) {
    // 0F B6 C3  movzx eax, bl
    const std::array<u8, 3> zx_byte{0x0F, 0xB6, 0xC3};
    const auto zx = dec.decode(zx_byte, kBase);
    ASSERT_TRUE(zx.ok()) << zx.error;
    EXPECT_EQ(zx.inst.mnemonic, Mnemonic::Movzx);
    ASSERT_EQ(zx.inst.operand_count, 2);
    EXPECT_EQ(zx.inst.op(0).size_bits, 32);
    EXPECT_EQ(zx.inst.op(1).size_bits, 8);  // the width the extend starts from
    EXPECT_EQ(zx.inst.op(1).reg, X86Reg::Rbx);

    // 0F B7 C3  movzx eax, bx
    const std::array<u8, 3> zx_word{0x0F, 0xB7, 0xC3};
    EXPECT_EQ(dec.decode(zx_word, kBase).inst.op(1).size_bits, 16);

    // 48 0F BE C3  movsx rax, bl
    const std::array<u8, 4> sx_byte{0x48, 0x0F, 0xBE, 0xC3};
    const auto sx = dec.decode(sx_byte, kBase);
    ASSERT_TRUE(sx.ok()) << sx.error;
    EXPECT_EQ(sx.inst.mnemonic, Mnemonic::Movsx);
    EXPECT_EQ(sx.inst.op(0).size_bits, 64);
    EXPECT_EQ(sx.inst.op(1).size_bits, 8);

    // 48 63 C3  movsxd rax, ebx -- folded onto Movsx; the widths distinguish it.
    const std::array<u8, 3> sxd{0x48, 0x63, 0xC3};
    const auto wide = dec.decode(sxd, kBase);
    ASSERT_TRUE(wide.ok()) << wide.error;
    EXPECT_EQ(wide.inst.mnemonic, Mnemonic::Movsx);
    EXPECT_EQ(wide.inst.op(1).size_bits, 32);
}

TEST_F(DecoderTest, RegisterEnumeratorsMatchHardwareEncoding) {
    static_assert(static_cast<dbt::u8>(X86Reg::Rax) == 0);
    static_assert(static_cast<dbt::u8>(X86Reg::Rcx) == 1);
    static_assert(static_cast<dbt::u8>(X86Reg::Rdx) == 2);
    static_assert(static_cast<dbt::u8>(X86Reg::Rbx) == 3);
    static_assert(static_cast<dbt::u8>(X86Reg::Rsp) == 4);
    static_assert(static_cast<dbt::u8>(X86Reg::Rbp) == 5);
    static_assert(static_cast<dbt::u8>(X86Reg::Rsi) == 6);
    static_assert(static_cast<dbt::u8>(X86Reg::Rdi) == 7);
    static_assert(static_cast<dbt::u8>(X86Reg::R15) == 15);
    static_assert(dbt::decoder::is_gpr(X86Reg::R15));
    static_assert(!dbt::decoder::is_gpr(X86Reg::Rip));
    static_assert(!dbt::decoder::is_gpr(X86Reg::None));
    SUCCEED();
}

}  // namespace
