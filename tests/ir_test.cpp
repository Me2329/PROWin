#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "dbt/decoder/decoder.hpp"
#include "dbt/ir/ir.hpp"

namespace {

using dbt::decoder::Cond;
using dbt::decoder::X86Reg;
using dbt::ir::BlockId;
using dbt::ir::Function;
using dbt::ir::Inst;
using dbt::ir::InstId;
using dbt::ir::IRBuilder;
using dbt::ir::Opcode;
using dbt::ir::Type;

/// Builds `rax = rax + rbx; return`.
Function build_straight_line() {
    Function func;
    IRBuilder b(func);
    b.create_block(0x1000);
    const InstId lhs = b.load_guest_reg(X86Reg::Rax);
    const InstId rhs = b.load_guest_reg(X86Reg::Rbx);
    const InstId sum = b.add(lhs, rhs);
    b.store_guest_reg(X86Reg::Rax, sum);
    b.ret();
    return func;
}

// --- Opcode traits ---------------------------------------------------------

TEST(IrTraits, TerminatorClassification) {
    static_assert(dbt::ir::is_terminator(Opcode::Jump));
    static_assert(dbt::ir::is_terminator(Opcode::Branch));
    static_assert(dbt::ir::is_terminator(Opcode::Return));
    static_assert(!dbt::ir::is_terminator(Opcode::Add));
    static_assert(!dbt::ir::is_terminator(Opcode::StoreMem));
    SUCCEED();
}

TEST(IrTraits, ValueAndFlagDefinition) {
    static_assert(dbt::ir::defines_value(Opcode::Add));
    static_assert(dbt::ir::defines_value(Opcode::LoadGuestReg));
    static_assert(!dbt::ir::defines_value(Opcode::StoreGuestReg));
    static_assert(!dbt::ir::defines_value(Opcode::Return));

    // ADD and SUB produce both a result and flags; CMP produces only flags.
    static_assert(dbt::ir::defines_flags(Opcode::Add));
    static_assert(dbt::ir::defines_flags(Opcode::Sub));
    static_assert(dbt::ir::defines_flags(Opcode::Cmp));
    static_assert(!dbt::ir::defines_flags(Opcode::Const));
    SUCCEED();
}

// --- Flag validity model ---------------------------------------------------

TEST(IrFlags, ArithmeticDefinesEveryModelledFlag) {
    namespace flag = dbt::ir::flag;
    static_assert(dbt::ir::defined_flags(Opcode::Add) == flag::kAll);
    static_assert(dbt::ir::defined_flags(Opcode::Sub) == flag::kAll);
    static_assert(dbt::ir::defined_flags(Opcode::Cmp) == flag::kAll);
    static_assert(dbt::ir::defined_flags(Opcode::Neg) == flag::kAll);
    // x86 clears CF/OF for logical ops and ARM64 TST clears C/V, so all four
    // agree and every flag is trustworthy.
    static_assert(dbt::ir::defined_flags(Opcode::And) == flag::kAll);
    static_assert(dbt::ir::defined_flags(Opcode::Test) == flag::kAll);
    SUCCEED();
}

TEST(IrFlags, ShiftsDoNotDefineCarryOrOverflow) {
    namespace flag = dbt::ir::flag;
    // x86 sets CF from the last bit shifted out; LSL/LSR + TST cannot
    // reproduce that, so CF must be reported as unavailable.
    static_assert((dbt::ir::defined_flags(Opcode::Shl) & flag::kCarry) == 0);
    static_assert((dbt::ir::defined_flags(Opcode::Shr) & flag::kCarry) == 0);
    static_assert((dbt::ir::defined_flags(Opcode::Sar) & flag::kOverflow) == 0);
    // ZF and SF do survive.
    static_assert((dbt::ir::defined_flags(Opcode::Shl) & flag::kZero) != 0);
    static_assert((dbt::ir::defined_flags(Opcode::Shl) & flag::kSign) != 0);
    SUCCEED();
}

TEST(IrFlags, IncAndDecDoNotDefineCarry) {
    namespace flag = dbt::ir::flag;
    // x86 preserves CF across INC/DEC deliberately; the ARM64 ADDS/SUBS that
    // implements them overwrites it, so CF is not trustworthy afterwards.
    static_assert((dbt::ir::defined_flags(Opcode::Inc) & flag::kCarry) == 0);
    static_assert((dbt::ir::defined_flags(Opcode::Dec) & flag::kCarry) == 0);
    static_assert((dbt::ir::defined_flags(Opcode::Inc) & flag::kOverflow) != 0);
    SUCCEED();
}

TEST(IrFlags, NonFlagOpcodesDefineNothing) {
    namespace flag = dbt::ir::flag;
    static_assert(dbt::ir::defined_flags(Opcode::Not) == flag::kNone);
    static_assert(dbt::ir::defined_flags(Opcode::Const) == flag::kNone);
    static_assert(dbt::ir::defined_flags(Opcode::AddrAdd) == flag::kNone);
    // defines_flags is derived from the mask, so the two always agree.
    static_assert(!dbt::ir::defines_flags(Opcode::Not));
    static_assert(dbt::ir::defines_flags(Opcode::Shl));
    SUCCEED();
}

TEST(IrFlags, ConditionsDeclareTheFlagsTheyRead) {
    namespace flag = dbt::ir::flag;
    using dbt::decoder::Cond;
    static_assert(dbt::ir::required_flags(Cond::Equal) == flag::kZero);
    static_assert(dbt::ir::required_flags(Cond::Below) == flag::kCarry);
    static_assert(dbt::ir::required_flags(Cond::Above) ==
                  (flag::kCarry | flag::kZero));
    static_assert(dbt::ir::required_flags(Cond::Less) ==
                  (flag::kSign | flag::kOverflow));
    static_assert(dbt::ir::required_flags(Cond::Greater) ==
                  (flag::kSign | flag::kOverflow | flag::kZero));
    // Parity demands everything, guaranteeing it is always refused.
    static_assert(dbt::ir::required_flags(Cond::Parity) == flag::kAll);
    SUCCEED();
}

TEST(IrFlags, TheRefusalRuleCatchesCarryAfterAShift) {
    // This is exactly the check the backend performs: required & ~defined.
    using dbt::decoder::Cond;
    constexpr auto unmet = [](Opcode producer, Cond cond) {
        return dbt::ir::required_flags(cond) & ~dbt::ir::defined_flags(producer);
    };

    static_assert(unmet(Opcode::Shr, Cond::Below) != 0);  // refused
    static_assert(unmet(Opcode::Shr, Cond::Equal) == 0);  // allowed
    static_assert(unmet(Opcode::Inc, Cond::Below) != 0);  // refused
    static_assert(unmet(Opcode::Inc, Cond::Equal) == 0);  // allowed
    static_assert(unmet(Opcode::Cmp, Cond::Below) == 0);  // allowed
    SUCCEED();
}

TEST(IrTraits, ArityMatchesOpcode) {
    static_assert(dbt::ir::operand_arity(Opcode::Const) == 0);
    static_assert(dbt::ir::operand_arity(Opcode::LoadMem) == 1);
    static_assert(dbt::ir::operand_arity(Opcode::StoreMem) == 2);
    static_assert(dbt::ir::operand_arity(Opcode::Branch) == 1);
    static_assert(dbt::ir::operand_arity(Opcode::Add) == 2);
    SUCCEED();
}

TEST(IrTraits, BitWidths) {
    static_assert(dbt::ir::bit_width(Type::I8) == 8);
    static_assert(dbt::ir::bit_width(Type::I32) == 32);
    static_assert(dbt::ir::bit_width(Type::I64) == 64);
    SUCCEED();
}

// --- Construction ----------------------------------------------------------

TEST(IrFunction, FirstCreatedBlockBecomesEntry) {
    Function func;
    const BlockId first = func.create_block(0x1000);
    const BlockId second = func.create_block(0x2000);

    EXPECT_EQ(func.entry(), first);
    EXPECT_NE(first, second);
    EXPECT_EQ(func.block_count(), 2u);
    EXPECT_EQ(func.block(second).guest_addr, 0x2000u);
}

TEST(IrFunction, EachInstructionIsNamedByItsOwnId) {
    const Function func = build_straight_line();

    // SSA: ids are dense and increasing, and each value-defining instruction is
    // referenced by the id it was assigned on creation.
    ASSERT_EQ(func.inst_count(), 5u);
    EXPECT_EQ(func.inst(0).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(func.inst(1).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(func.inst(2).opcode, Opcode::Add);
    EXPECT_EQ(func.inst(2).operands[0], 0u);
    EXPECT_EQ(func.inst(2).operands[1], 1u);
    EXPECT_EQ(func.inst(3).opcode, Opcode::StoreGuestReg);
    EXPECT_EQ(func.inst(3).operands[0], 2u);
    EXPECT_EQ(func.inst(4).opcode, Opcode::Return);
}

TEST(IrFunction, GuestAddressStampPropagatesToInstructions) {
    Function func;
    IRBuilder b(func);
    b.create_block(0x1000);
    b.set_guest_addr(0x1000);
    const InstId a = b.load_guest_reg(X86Reg::Rax);
    b.set_guest_addr(0x1003);
    const InstId c = b.const_int(7);
    b.store_guest_reg(X86Reg::Rax, c);
    b.ret();

    EXPECT_EQ(func.inst(a).guest_addr, 0x1000u);
    EXPECT_EQ(func.inst(c).guest_addr, 0x1003u);
}

TEST(IrFunction, CmpProducesFlagsTypedValue) {
    Function func;
    IRBuilder b(func);
    b.create_block();
    const InstId lhs = b.load_guest_reg(X86Reg::Rax);
    const InstId rhs = b.load_guest_reg(X86Reg::Rbx);
    const InstId flags = b.cmp(lhs, rhs);
    b.ret();

    EXPECT_EQ(func.inst(flags).opcode, Opcode::Cmp);
    // The type field records the operand width; being a flag producer is a
    // property of the opcode, not of the type.
    EXPECT_EQ(func.inst(flags).type, Type::I64);
    EXPECT_TRUE(dbt::ir::defines_flags(func.inst(flags).opcode));
    EXPECT_TRUE(func.inst(flags).defines_value());
}

TEST(IrFunction, OutOfRangeAccessThrows) {
    const Function func = build_straight_line();
    EXPECT_THROW(static_cast<void>(func.inst(99)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(func.block(99)), std::out_of_range);

    Function empty;
    EXPECT_THROW(static_cast<void>(empty.append(0, Inst{})), std::out_of_range);
}

// --- Verifier: accepting well-formed IR ------------------------------------

TEST(IrVerify, AcceptsStraightLineBlock) {
    const Function func = build_straight_line();
    std::string error;
    EXPECT_TRUE(func.verify(&error)) << error;
}

TEST(IrVerify, AcceptsConditionalControlFlow) {
    // if (rax == rbx) goto taken; else goto fallthrough;
    Function func;
    IRBuilder b(func);
    const BlockId head = b.create_block(0x1000);
    const BlockId taken = func.create_block(0x1100);
    const BlockId fall = func.create_block(0x1010);

    b.set_insert_point(head);
    const InstId lhs = b.load_guest_reg(X86Reg::Rax);
    const InstId rhs = b.load_guest_reg(X86Reg::Rbx);
    const InstId flags = b.cmp(lhs, rhs);
    b.branch(flags, Cond::Equal, taken, fall);

    b.set_insert_point(taken);
    b.ret();
    b.set_insert_point(fall);
    b.ret();

    std::string error;
    EXPECT_TRUE(func.verify(&error)) << error;
    EXPECT_TRUE(func.is_sealed(head));
    EXPECT_TRUE(func.is_sealed(taken));
    EXPECT_TRUE(func.is_sealed(fall));
}

TEST(IrVerify, AcceptsBackEdge) {
    Function func;
    IRBuilder b(func);
    const BlockId loop = b.create_block(0x1000);
    b.jump(loop);

    std::string error;
    EXPECT_TRUE(func.verify(&error)) << error;
}

// --- Verifier: rejecting malformed IR --------------------------------------

TEST(IrVerify, RejectsFunctionWithNoBlocks) {
    const Function func;
    std::string error;
    EXPECT_FALSE(func.verify(&error));
    EXPECT_NE(error.find("no blocks"), std::string::npos) << error;
}

TEST(IrVerify, RejectsEmptyBlock) {
    Function func;
    static_cast<void>(func.create_block(0x1000));
    std::string error;
    EXPECT_FALSE(func.verify(&error));
    EXPECT_NE(error.find("empty"), std::string::npos) << error;
}

TEST(IrVerify, RejectsBlockWithoutTerminator) {
    Function func;
    IRBuilder b(func);
    b.create_block(0x1000);
    b.load_guest_reg(X86Reg::Rax);  // no terminator follows

    std::string error;
    EXPECT_FALSE(func.verify(&error));
    EXPECT_NE(error.find("terminator"), std::string::npos) << error;
}

TEST(IrVerify, RejectsInstructionAfterTerminator) {
    Function func;
    IRBuilder b(func);
    b.create_block(0x1000);
    b.ret();
    b.load_guest_reg(X86Reg::Rax);  // unreachable, and illegal

    std::string error;
    EXPECT_FALSE(func.verify(&error));
    EXPECT_NE(error.find("last"), std::string::npos) << error;
}

TEST(IrVerify, RejectsUseBeforeDefinition) {
    Function func;
    const BlockId blk = func.create_block(0x1000);

    Inst load;
    load.opcode = Opcode::LoadGuestReg;
    load.guest_reg = X86Reg::Rax;
    const InstId value = func.append(blk, load);

    // Reference an instruction id that has not been emitted yet.
    Inst store;
    store.opcode = Opcode::StoreGuestReg;
    store.guest_reg = X86Reg::Rbx;
    store.operands[0] = value + 10;
    store.operand_count = 1;
    func.append(blk, store);

    Inst term;
    term.opcode = Opcode::Return;
    func.append(blk, term);

    std::string error;
    EXPECT_FALSE(func.verify(&error));
    EXPECT_NE(error.find("unknown value"), std::string::npos) << error;
}

TEST(IrVerify, RejectsOperandThatDefinesNoValue) {
    Function func;
    const BlockId blk = func.create_block(0x1000);

    Inst load;
    load.opcode = Opcode::LoadGuestReg;
    load.guest_reg = X86Reg::Rax;
    const InstId loaded = func.append(blk, load);

    Inst store;
    store.opcode = Opcode::StoreGuestReg;
    store.guest_reg = X86Reg::Rbx;
    store.operands[0] = loaded;
    store.operand_count = 1;
    const InstId stored = func.append(blk, store);

    // StoreGuestReg defines nothing, so consuming its id is invalid.
    Inst bad;
    bad.opcode = Opcode::StoreGuestReg;
    bad.guest_reg = X86Reg::Rcx;
    bad.operands[0] = stored;
    bad.operand_count = 1;
    func.append(blk, bad);

    Inst term;
    term.opcode = Opcode::Return;
    func.append(blk, term);

    std::string error;
    EXPECT_FALSE(func.verify(&error));
    EXPECT_NE(error.find("defines no value"), std::string::npos) << error;
}

TEST(IrVerify, RejectsArityMismatch) {
    Function func;
    const BlockId blk = func.create_block(0x1000);

    Inst load;
    load.opcode = Opcode::LoadGuestReg;
    load.guest_reg = X86Reg::Rax;
    const InstId value = func.append(blk, load);

    // Add requires two operands.
    Inst add;
    add.opcode = Opcode::Add;
    add.operands[0] = value;
    add.operand_count = 1;
    func.append(blk, add);

    Inst term;
    term.opcode = Opcode::Return;
    func.append(blk, term);

    std::string error;
    EXPECT_FALSE(func.verify(&error));
    EXPECT_NE(error.find("expected"), std::string::npos) << error;
}

TEST(IrVerify, RejectsOutOfRangeBranchTarget) {
    Function func;
    IRBuilder b(func);
    b.create_block(0x1000);
    b.jump(42);  // no such block

    std::string error;
    EXPECT_FALSE(func.verify(&error));
    EXPECT_NE(error.find("out-of-range"), std::string::npos) << error;
}

TEST(IrVerify, RejectsBranchOnNonFlagsValue) {
    Function func;
    IRBuilder b(func);
    const BlockId head = b.create_block(0x1000);
    const BlockId other = func.create_block(0x1100);

    b.set_insert_point(head);
    const InstId plain = b.load_guest_reg(X86Reg::Rax);  // i64, not flags
    b.branch(plain, Cond::Equal, other, other);
    b.set_insert_point(other);
    b.ret();

    std::string error;
    EXPECT_FALSE(func.verify(&error));
    EXPECT_NE(error.find("non-flags"), std::string::npos) << error;
}

TEST(IrVerify, RejectsBranchWithoutCondition) {
    Function func;
    IRBuilder b(func);
    const BlockId head = b.create_block(0x1000);
    const BlockId other = func.create_block(0x1100);

    b.set_insert_point(head);
    const InstId lhs = b.load_guest_reg(X86Reg::Rax);
    const InstId rhs = b.load_guest_reg(X86Reg::Rbx);
    const InstId flags = b.cmp(lhs, rhs);
    b.branch(flags, Cond::None, other, other);
    b.set_insert_point(other);
    b.ret();

    std::string error;
    EXPECT_FALSE(func.verify(&error));
    EXPECT_NE(error.find("condition"), std::string::npos) << error;
}

// --- Diagnostics -----------------------------------------------------------

TEST(IrDump, RendersReadableText) {
    const Function func = build_straight_line();
    const std::string text = func.to_string();

    EXPECT_NE(text.find("block0 (entry):"), std::string::npos) << text;
    EXPECT_NE(text.find("load_guest_reg rax"), std::string::npos) << text;
    EXPECT_NE(text.find("%2 = add %0, %1"), std::string::npos) << text;
    EXPECT_NE(text.find("return"), std::string::npos) << text;
}

TEST(IrDump, OpcodeNames) {
    EXPECT_EQ(dbt::ir::to_string(Opcode::Add), "add");
    EXPECT_EQ(dbt::ir::to_string(Opcode::LoadGuestReg), "load_guest_reg");
    EXPECT_EQ(dbt::ir::to_string(Opcode::Branch), "branch");
    EXPECT_EQ(dbt::ir::to_string(Type::I64), "i64");
    EXPECT_EQ(dbt::ir::to_string(Type::Flags), "flags");
}

}  // namespace
