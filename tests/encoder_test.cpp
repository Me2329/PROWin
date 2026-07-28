#include <gtest/gtest.h>

#include "dbt/backend/arm64_encoder.hpp"
#include "dbt/backend/register_map.hpp"
#include "dbt/decoder/decoder.hpp"

namespace {

namespace a64 = dbt::backend::a64;
using a64::Reg;
using dbt::backend::map_gpr;
using dbt::backend::to_arm64_condition;
using dbt::decoder::X86Reg;

// Golden words below come from the ARM Architecture Reference Manual encodings
// and match standard assembler output. They are asserted at compile time
// wherever possible, so a regression breaks the build, not just the test run.

// --- Move wide immediate ---------------------------------------------------

TEST(Arm64Encoder, MoveWideImmediate) {
    static_assert(a64::movz(Reg::X0, 0) == 0xD2800000u);           // movz x0, #0
    static_assert(a64::movz(Reg::X1, 0xFFFF) == 0xD29FFFE1u);      // movz x1, #65535
    static_assert(a64::movz(Reg::X0, 0x1234, 16) == 0xD2A24680u);
    static_assert(a64::movk(Reg::X0, 0x1234, 16) == 0xF2A24680u);
    static_assert(a64::movn(Reg::X0, 0) == 0x92800000u);           // movn x0, #0
    static_assert(a64::movz(Reg::X15, 0) == 0xD280000Fu);
    SUCCEED();
}

TEST(Arm64Encoder, MoveWideShiftSelectsCorrectHalfword) {
    static_assert(a64::movk(Reg::X0, 1, 0) == 0xF2800020u);
    static_assert(a64::movk(Reg::X0, 1, 16) == 0xF2A00020u);
    static_assert(a64::movk(Reg::X0, 1, 32) == 0xF2C00020u);
    static_assert(a64::movk(Reg::X0, 1, 48) == 0xF2E00020u);
    SUCCEED();
}

// --- Arithmetic ------------------------------------------------------------

TEST(Arm64Encoder, AddSubShiftedRegister) {
    static_assert(a64::add_reg(Reg::X0, Reg::X1, Reg::X2) == 0x8B020020u);
    static_assert(a64::sub_reg(Reg::X0, Reg::X1, Reg::X2) == 0xCB020020u);
    // The flag-setting forms differ only in bit 29.
    static_assert(a64::add_reg(Reg::X0, Reg::X1, Reg::X2, true) == 0xAB020020u);
    static_assert(a64::sub_reg(Reg::X0, Reg::X1, Reg::X2, true) == 0xEB020020u);
    static_assert(a64::add_reg(Reg::X0, Reg::X1, Reg::X2, true) ==
                  (a64::add_reg(Reg::X0, Reg::X1, Reg::X2) | (1u << 29)));
    SUCCEED();
}

TEST(Arm64Encoder, CmpIsSubsToZeroRegister) {
    static_assert(a64::cmp_reg(Reg::X1, Reg::X2) == 0xEB02003Fu);
    static_assert(a64::cmp_reg(Reg::X1, Reg::X2) ==
                  a64::sub_reg(a64::kZeroReg, Reg::X1, Reg::X2, true));
    static_assert(a64::cmp_imm(Reg::X1, 1) == 0xF100043Fu);
    SUCCEED();
}

TEST(Arm64Encoder, AddSubImmediate) {
    static_assert(a64::add_imm(Reg::X0, Reg::X1, 1) == 0x91000420u);
    static_assert(a64::sub_imm(Reg::X0, Reg::X1, 1) == 0xD1000420u);
    static_assert(a64::add_imm(Reg::X0, Reg::X0, 0xFFF) == 0x913FFC00u);
    SUCCEED();
}

TEST(Arm64Encoder, MovRegisterIsOrrWithZeroRegister) {
    static_assert(a64::mov_reg(Reg::X0, Reg::X1) == 0xAA0103E0u);
    static_assert(a64::mov_reg(Reg::X15, Reg::X28) == 0xAA1C03EFu);
    SUCCEED();
}

TEST(Arm64Encoder, LogicalShiftLeft) {
    static_assert(a64::lsl_imm(Reg::X0, Reg::X1, 3) == 0xD37DF020u);
    static_assert(a64::lsl_imm(Reg::X0, Reg::X1, 1) == 0xD37FF820u);
    SUCCEED();
}

TEST(Arm64Encoder, RightShiftsLogicalAndArithmetic) {
    // lsr x0, x1, #3 / asr x0, x1, #3
    static_assert(a64::lsr_imm(Reg::X0, Reg::X1, 3) == 0xD343FC20u);
    static_assert(a64::asr_imm(Reg::X0, Reg::X1, 3) == 0x9343FC20u);
    // ASR is SBFM where LSR is UBFM, so they are genuinely different encodings.
    static_assert(a64::lsr_imm(Reg::X0, Reg::X1, 3) !=
                  a64::asr_imm(Reg::X0, Reg::X1, 3));
    SUCCEED();
}

TEST(Arm64Encoder, ThirtyTwoBitShiftsUseDifferentFieldWidths) {
    using a64::Width;
    // lsl w0, w1, #3 / lsr w0, w1, #3 / asr w0, w1, #3
    static_assert(a64::lsl_imm(Reg::X0, Reg::X1, 3, Width::W32) == 0x531D7020u);
    static_assert(a64::lsr_imm(Reg::X0, Reg::X1, 3, Width::W32) == 0x53037C20u);
    static_assert(a64::asr_imm(Reg::X0, Reg::X1, 3, Width::W32) == 0x13037C20u);

    // The 32-bit forms are NOT reachable by clearing sf: the N field and the
    // immr/imms widths change too. That is why the shifts take an explicit
    // width instead of going through with_width.
    static_assert(a64::lsl_imm(Reg::X0, Reg::X1, 3, Width::W32) !=
                  a64::with_width(a64::lsl_imm(Reg::X0, Reg::X1, 3), Width::W32));
    SUCCEED();
}

// --- Loads and stores ------------------------------------------------------

TEST(Arm64Encoder, LoadStoreUnsignedOffset) {
    static_assert(a64::ldr_imm(Reg::X0, Reg::X1) == 0xF9400020u);
    static_assert(a64::str_imm(Reg::X0, Reg::X1) == 0xF9000020u);
    // The immediate is scaled by the access size.
    static_assert(a64::ldr_imm(Reg::X0, Reg::X1, 8) == 0xF9400420u);
    static_assert(a64::ldr_imm(Reg::X0, Reg::X28, 120) == 0xF9403F80u);
    SUCCEED();
}

TEST(Arm64Encoder, PairLoadStoreWithWriteBack) {
    // The standard prologue/epilogue pair.
    static_assert(a64::stp_pre(Reg::X29, Reg::X30, a64::kStackPointer, -16) ==
                  0xA9BF7BFDu);
    static_assert(a64::ldp_post(Reg::X29, Reg::X30, a64::kStackPointer, 16) ==
                  0xA8C17BFDu);
    SUCCEED();
}

// --- Branches --------------------------------------------------------------

TEST(Arm64Encoder, Branches) {
    static_assert(a64::b(4) == 0x14000001u);
    static_assert(a64::b(0) == 0x14000000u);
    static_assert(a64::b(-4) == 0x17FFFFFFu);  // sign-extended imm26
    static_assert(a64::br(Reg::X0) == 0xD61F0000u);
    static_assert(a64::ret() == 0xD65F03C0u);
    static_assert(a64::ret(Reg::X16) == 0xD65F0200u);
    static_assert(a64::nop() == 0xD503201Fu);
    SUCCEED();
}

TEST(Arm64Encoder, ConditionalBranches) {
    static_assert(a64::b_cond(a64::Cond::EQ, 4) == 0x54000020u);
    static_assert(a64::b_cond(a64::Cond::NE, 4) == 0x54000021u);
    static_assert(a64::b_cond(a64::Cond::EQ, 0) == 0x54000000u);
    static_assert(a64::b_cond(a64::Cond::LT, 8) == 0x5400004Bu);
    SUCCEED();
}

// --- Range predicates ------------------------------------------------------

TEST(Arm64Encoder, RangePredicates) {
    static_assert(a64::fits_imm12(0));
    static_assert(a64::fits_imm12(4095));
    static_assert(!a64::fits_imm12(4096));

    static_assert(a64::fits_ldst_offset64(0));
    static_assert(a64::fits_ldst_offset64(8));
    static_assert(!a64::fits_ldst_offset64(4));  // not 8-byte aligned
    static_assert(a64::fits_ldst_offset64(32760));
    static_assert(!a64::fits_ldst_offset64(32768));

    static_assert(a64::fits_pair_offset(-16));
    static_assert(a64::fits_pair_offset(504));
    static_assert(!a64::fits_pair_offset(512));
    static_assert(!a64::fits_pair_offset(4));  // not 8-byte aligned

    static_assert(a64::fits_branch26(4));
    static_assert(!a64::fits_branch26(3));  // not instruction aligned
    // B.cond spans +/-1 MiB, but the largest representable offset is
    // (2^18 - 1) * 4 == 1048572 -- four bytes short of 1 MiB exactly.
    static_assert(a64::fits_branch19((1 << 20) - 4));
    static_assert(!a64::fits_branch19(1 << 20));
    static_assert(a64::fits_branch19(-(1 << 20)));  // the negative end reaches further
    static_assert(!a64::fits_branch19(1 << 21));
    SUCCEED();
}

// --- 32-bit widths ---------------------------------------------------------

TEST(Arm64Encoder, ThirtyTwoBitFormsClearTheSfBit) {
    using a64::Width;
    // add w0, w1, w2 / sub w0, w1, w2
    static_assert(a64::with_width(a64::add_reg(Reg::X0, Reg::X1, Reg::X2),
                                  Width::W32) == 0x0B020020u);
    static_assert(a64::with_width(a64::sub_reg(Reg::X0, Reg::X1, Reg::X2),
                                  Width::W32) == 0x4B020020u);
    // mov w0, w1
    static_assert(a64::with_width(a64::mov_reg(Reg::X0, Reg::X1), Width::W32) ==
                  0x2A0103E0u);
    // movz w0, #42
    static_assert(a64::with_width(a64::movz(Reg::X0, 42), Width::W32) == 0x52800540u);
    // ands w0, w1, w2 / cmp w1, w2
    static_assert(a64::with_width(a64::and_reg(Reg::X0, Reg::X1, Reg::X2, true),
                                  Width::W32) == 0x6A020020u);
    static_assert(a64::with_width(a64::cmp_reg(Reg::X1, Reg::X2), Width::W32) ==
                  0x6B02003Fu);
    // X64 leaves the encoding untouched.
    static_assert(a64::with_width(a64::add_reg(Reg::X0, Reg::X1, Reg::X2),
                                  Width::X64) == 0x8B020020u);
    SUCCEED();
}

TEST(Arm64Encoder, ThirtyTwoBitLoadStoreScalesByFour) {
    static_assert(a64::ldr_imm32(Reg::X0, Reg::X1) == 0xB9400020u);
    static_assert(a64::str_imm32(Reg::X0, Reg::X1) == 0xB9000020u);
    static_assert(a64::ldr_imm32(Reg::X0, Reg::X1, 4) == 0xB9400420u);
    static_assert(a64::fits_ldst_offset32(4));
    static_assert(!a64::fits_ldst_offset32(2));  // not 4-byte aligned
    SUCCEED();
}

// --- Extension and narrow load/store ---------------------------------------

TEST(Arm64Encoder, ExtensionForms) {
    using a64::Width;
    // uxtb w0, w1 / uxth w0, w1 -- writing a W register zero-extends, so these
    // cover MOVZX at either destination width.
    static_assert(a64::uxtb(Reg::X0, Reg::X1) == 0x53001C20u);
    static_assert(a64::uxth(Reg::X0, Reg::X1) == 0x53003C20u);
    // sxtb x0, w1 / sxth x0, w1 / sxtw x0, w1
    static_assert(a64::sxtb(Reg::X0, Reg::X1) == 0x93401C20u);
    static_assert(a64::sxth(Reg::X0, Reg::X1) == 0x93403C20u);
    static_assert(a64::sxtw(Reg::X0, Reg::X1) == 0x93407C20u);
    // The signed forms depend on the destination width; the unsigned ones do not.
    static_assert(a64::sxtb(Reg::X0, Reg::X1, Width::W32) == 0x13001C20u);
    static_assert(a64::sxtb(Reg::X0, Reg::X1, Width::W32) !=
                  a64::sxtb(Reg::X0, Reg::X1, Width::X64));
    SUCCEED();
}

TEST(Arm64Encoder, ByteAndHalfwordLoadStore) {
    static_assert(a64::ldrb(Reg::X0, Reg::X1) == 0x39400020u);
    static_assert(a64::ldrh(Reg::X0, Reg::X1) == 0x79400020u);
    static_assert(a64::strb(Reg::X0, Reg::X1) == 0x39000020u);
    static_assert(a64::strh(Reg::X0, Reg::X1) == 0x79000020u);
    // LDRB is unscaled; LDRH scales its offset by two.
    static_assert(a64::ldrb(Reg::X0, Reg::X1, 1) == 0x39400420u);
    static_assert(a64::ldrh(Reg::X0, Reg::X1, 2) == 0x79400420u);
    SUCCEED();
}

// --- Register mapping ------------------------------------------------------

TEST(RegisterMap, X86GprsMapOntoX0ThroughX15) {
    static_assert(map_gpr(X86Reg::Rax) == Reg::X0);
    static_assert(map_gpr(X86Reg::Rcx) == Reg::X1);
    static_assert(map_gpr(X86Reg::Rdx) == Reg::X2);
    static_assert(map_gpr(X86Reg::Rbx) == Reg::X3);
    static_assert(map_gpr(X86Reg::Rsp) == Reg::X4);
    static_assert(map_gpr(X86Reg::Rbp) == Reg::X5);
    static_assert(map_gpr(X86Reg::Rsi) == Reg::X6);
    static_assert(map_gpr(X86Reg::Rdi) == Reg::X7);
    static_assert(map_gpr(X86Reg::R8) == Reg::X8);
    static_assert(map_gpr(X86Reg::R15) == Reg::X15);
    SUCCEED();
}

TEST(RegisterMap, ReservedRegistersDoNotOverlapGuestState) {
    // Guest state occupies x0-x15; the backend's own registers must sit above.
    static_assert(static_cast<dbt::u8>(dbt::backend::kScratch0) > 15);
    static_assert(static_cast<dbt::u8>(dbt::backend::kScratch1) > 15);
    static_assert(static_cast<dbt::u8>(dbt::backend::kCpuStateBase) > 15);
    static_assert(dbt::backend::kScratch0 != dbt::backend::kScratch1);
    static_assert(dbt::backend::kCpuStateBase != dbt::backend::kScratch0);
    static_assert(dbt::backend::kCpuStateBase != dbt::backend::kScratch1);
    // ...and must not collide with the PCS frame pointer or link register.
    static_assert(dbt::backend::kCpuStateBase != a64::kFramePointer);
    static_assert(dbt::backend::kCpuStateBase != a64::kLinkReg);
    SUCCEED();
}

TEST(RegisterMap, OnlyGeneralPurposeRegistersAreMappable) {
    static_assert(dbt::backend::is_mappable(X86Reg::Rax));
    static_assert(dbt::backend::is_mappable(X86Reg::R15));
    static_assert(!dbt::backend::is_mappable(X86Reg::Rip));
    static_assert(!dbt::backend::is_mappable(X86Reg::None));
    SUCCEED();
}

// --- Condition mapping -----------------------------------------------------

TEST(ConditionMap, SignedAndEqualityConditionsMapDirectly) {
    a64::Cond out{};
    using dbt::decoder::Cond;

    EXPECT_TRUE(to_arm64_condition(Cond::Equal, true, out));
    EXPECT_EQ(out, a64::Cond::EQ);
    EXPECT_TRUE(to_arm64_condition(Cond::NotEqual, true, out));
    EXPECT_EQ(out, a64::Cond::NE);
    EXPECT_TRUE(to_arm64_condition(Cond::Less, true, out));
    EXPECT_EQ(out, a64::Cond::LT);
    EXPECT_TRUE(to_arm64_condition(Cond::GreaterEqual, true, out));
    EXPECT_EQ(out, a64::Cond::GE);
    EXPECT_TRUE(to_arm64_condition(Cond::LessEqual, true, out));
    EXPECT_EQ(out, a64::Cond::LE);
    EXPECT_TRUE(to_arm64_condition(Cond::Greater, true, out));
    EXPECT_EQ(out, a64::Cond::GT);
    EXPECT_TRUE(to_arm64_condition(Cond::Sign, true, out));
    EXPECT_EQ(out, a64::Cond::MI);
    EXPECT_TRUE(to_arm64_condition(Cond::Overflow, true, out));
    EXPECT_EQ(out, a64::Cond::VS);
}

TEST(ConditionMap, CarryIsInvertedAfterSubtraction) {
    a64::Cond out{};
    using dbt::decoder::Cond;

    // x86 CF==1 after SUB means "borrow"; AArch64 C==0 means the same thing.
    EXPECT_TRUE(to_arm64_condition(Cond::Below, true, out));
    EXPECT_EQ(out, a64::Cond::CC) << "JB after SUB must become CC/LO, not CS";

    EXPECT_TRUE(to_arm64_condition(Cond::AboveEqual, true, out));
    EXPECT_EQ(out, a64::Cond::CS) << "JAE after SUB must become CS/HS, not CC";

    EXPECT_TRUE(to_arm64_condition(Cond::BelowEqual, true, out));
    EXPECT_EQ(out, a64::Cond::LS);
    EXPECT_TRUE(to_arm64_condition(Cond::Above, true, out));
    EXPECT_EQ(out, a64::Cond::HI);
}

TEST(ConditionMap, CarryIsNotInvertedAfterAddition) {
    a64::Cond out{};
    using dbt::decoder::Cond;

    // After ADD both architectures agree: carry means carry-out.
    EXPECT_TRUE(to_arm64_condition(Cond::Below, false, out));
    EXPECT_EQ(out, a64::Cond::CS) << "JB after ADD must become CS, not CC";

    EXPECT_TRUE(to_arm64_condition(Cond::AboveEqual, false, out));
    EXPECT_EQ(out, a64::Cond::CC);
    EXPECT_TRUE(to_arm64_condition(Cond::BelowEqual, false, out));
    EXPECT_EQ(out, a64::Cond::HI);
    EXPECT_TRUE(to_arm64_condition(Cond::Above, false, out));
    EXPECT_EQ(out, a64::Cond::LS);
}

TEST(ConditionMap, EqualityIsUnaffectedByTheFlagSource) {
    a64::Cond from_sub{};
    a64::Cond from_add{};
    using dbt::decoder::Cond;

    ASSERT_TRUE(to_arm64_condition(Cond::Equal, true, from_sub));
    ASSERT_TRUE(to_arm64_condition(Cond::Equal, false, from_add));
    EXPECT_EQ(from_sub, from_add);
}

TEST(ConditionMap, ParityHasNoArm64Equivalent) {
    a64::Cond out{};
    using dbt::decoder::Cond;

    // AArch64 has no parity flag, so these must be rejected rather than
    // silently mapped to something that merely looks close.
    EXPECT_FALSE(to_arm64_condition(Cond::Parity, true, out));
    EXPECT_FALSE(to_arm64_condition(Cond::NotParity, true, out));
    EXPECT_FALSE(to_arm64_condition(Cond::None, true, out));
}

}  // namespace
