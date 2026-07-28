#include <gtest/gtest.h>

#include <array>
#include <span>
#include <string>

#include "dbt/common/types.hpp"
#include "dbt/decoder/decoder.hpp"
#include "dbt/frontend/lifter.hpp"
#include "dbt/ir/ir.hpp"

namespace {

using dbt::GuestAddr;
using dbt::u8;
using dbt::usize;
using dbt::decoder::Cond;
using dbt::decoder::X86Reg;
using dbt::frontend::Lifter;
using dbt::frontend::LiftError;
using dbt::frontend::LiftResult;
using dbt::ir::Function;
using dbt::ir::Inst;
using dbt::ir::Opcode;

constexpr GuestAddr kBase = 0x1000;

class LifterTest : public ::testing::Test {
protected:
    Lifter lifter;

    /// Lifts `code` and asserts the resulting IR satisfies every invariant.
    LiftResult lift_ok(std::span<const u8> code) {
        LiftResult res = lifter.lift_block(code, kBase);
        EXPECT_TRUE(res.ok()) << dbt::frontend::to_string(res.error);
        std::string error;
        EXPECT_TRUE(res.function.verify(&error))
            << error << "\n"
            << res.function.to_string();
        return res;
    }
};

/// Counts instructions with a given opcode across the whole function.
usize count_opcode(const Function& func, Opcode op) {
    usize n = 0;
    for (const Inst& in : func.insts()) {
        if (in.opcode == op) {
            ++n;
        }
    }
    return n;
}

/// Returns the first instruction with `op`, or nullptr.
const Inst* find_opcode(const Function& func, Opcode op) {
    for (const Inst& in : func.insts()) {
        if (in.opcode == op) {
            return &in;
        }
    }
    return nullptr;
}

// --- MOV -------------------------------------------------------------------

TEST_F(LifterTest, MovRegToRegBecomesLoadThenStore) {
    // 48 89 D8  mov rax, rbx   |  C3  ret
    const std::array<u8, 4> code{0x48, 0x89, 0xD8, 0xC3};
    const auto res = lift_ok(code);
    EXPECT_EQ(res.guest_inst_count, 2u);

    const Function& f = res.function;
    // The MOV lifts to exactly: load rbx, store rax.
    ASSERT_GE(f.inst_count(), 2u);
    EXPECT_EQ(f.inst(0).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(f.inst(0).guest_reg, X86Reg::Rbx);
    EXPECT_EQ(f.inst(1).opcode, Opcode::StoreGuestReg);
    EXPECT_EQ(f.inst(1).guest_reg, X86Reg::Rax);
    EXPECT_EQ(f.inst(1).operands[0], 0u);

    // Provenance is preserved.
    EXPECT_EQ(f.inst(0).guest_addr, kBase);
}

TEST_F(LifterTest, MovImmediateBecomesConstThenStore) {
    // 48 C7 C0 2A 00 00 00  mov rax, 42  |  C3
    const std::array<u8, 8> code{0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    EXPECT_EQ(f.inst(0).opcode, Opcode::Const);
    EXPECT_EQ(f.inst(0).imm, 42);
    EXPECT_EQ(f.inst(1).opcode, Opcode::StoreGuestReg);
    EXPECT_EQ(f.inst(1).guest_reg, X86Reg::Rax);
}

TEST_F(LifterTest, MovLoadFromMemoryComputesAddress) {
    // 48 8B 43 08  mov rax, [rbx+8]  |  C3
    const std::array<u8, 5> code{0x48, 0x8B, 0x43, 0x08, 0xC3};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    // base + disp, using flag-free addressing arithmetic.
    EXPECT_EQ(f.inst(0).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(f.inst(0).guest_reg, X86Reg::Rbx);
    EXPECT_EQ(f.inst(1).opcode, Opcode::Const);
    EXPECT_EQ(f.inst(1).imm, 8);
    EXPECT_EQ(f.inst(2).opcode, Opcode::AddrAdd);
    EXPECT_EQ(f.inst(3).opcode, Opcode::LoadMem);
    EXPECT_EQ(f.inst(3).operands[0], 2u);
    EXPECT_EQ(f.inst(4).opcode, Opcode::StoreGuestReg);
    EXPECT_EQ(f.inst(4).guest_reg, X86Reg::Rax);
}

TEST_F(LifterTest, MovStoreToMemory) {
    // 48 89 03  mov [rbx], rax  |  C3
    const std::array<u8, 4> code{0x48, 0x89, 0x03, 0xC3};
    const auto res = lift_ok(code);

    EXPECT_EQ(count_opcode(res.function, Opcode::StoreMem), 1u);
    const Inst* store = find_opcode(res.function, Opcode::StoreMem);
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->operand_count, 2);
}

TEST_F(LifterTest, ScaledIndexLowersToShiftNotMultiply) {
    // 48 8B 04 D3  mov rax, [rbx + rdx*8]  |  C3
    const std::array<u8, 5> code{0x48, 0x8B, 0x04, 0xD3, 0xC3};
    const auto res = lift_ok(code);

    const Function& f = res.function;

    // Check the address expression directly rather than counting opcodes: the
    // trailing RET also emits an addr_add for its stack unwind.
    const Inst* load = find_opcode(f, Opcode::LoadMem);
    ASSERT_NE(load, nullptr) << f.to_string();

    const Inst& addr = f.inst(load->operands[0]);
    ASSERT_EQ(addr.opcode, Opcode::AddrAdd) << f.to_string();

    const Inst& base = f.inst(addr.operands[0]);
    EXPECT_EQ(base.opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(base.guest_reg, X86Reg::Rbx);

    const Inst& scaled = f.inst(addr.operands[1]);
    ASSERT_EQ(scaled.opcode, Opcode::AddrShl) << f.to_string();
    EXPECT_EQ(scaled.imm, 3);  // log2(8): a shift, never a multiply
    EXPECT_EQ(f.inst(scaled.operands[0]).guest_reg, X86Reg::Rdx);
}

TEST_F(LifterTest, RipRelativeOperandFoldsToAbsoluteConstant) {
    // 48 8B 05 10 00 00 00  mov rax, [rip+0x10]  |  C3
    const std::array<u8, 8> code{0x48, 0x8B, 0x05, 0x10, 0x00, 0x00, 0x00, 0xC3};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    // The address is a single constant; no register load is needed.
    EXPECT_EQ(f.inst(0).opcode, Opcode::Const);
    EXPECT_EQ(f.inst(0).imm, static_cast<dbt::i64>(kBase + 7 + 0x10));
    EXPECT_EQ(f.inst(1).opcode, Opcode::LoadMem);
}

// --- Arithmetic and flags --------------------------------------------------

TEST_F(LifterTest, AddReadsBothOperandsWritesBackAndDefinesFlags) {
    // 48 01 D8  add rax, rbx  |  C3
    const std::array<u8, 4> code{0x48, 0x01, 0xD8, 0xC3};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    EXPECT_EQ(f.inst(0).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(f.inst(0).guest_reg, X86Reg::Rax);
    EXPECT_EQ(f.inst(1).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(f.inst(1).guest_reg, X86Reg::Rbx);
    EXPECT_EQ(f.inst(2).opcode, Opcode::Add);
    EXPECT_EQ(f.inst(3).opcode, Opcode::StoreGuestReg);
    EXPECT_EQ(f.inst(3).guest_reg, X86Reg::Rax);

    static_assert(dbt::ir::defines_flags(Opcode::Add));
}

TEST_F(LifterTest, SubLowersToSubOpcode) {
    // 48 29 D8  sub rax, rbx  |  C3
    const std::array<u8, 4> code{0x48, 0x29, 0xD8, 0xC3};
    const auto res = lift_ok(code);
    EXPECT_EQ(count_opcode(res.function, Opcode::Sub), 1u);
}

TEST_F(LifterTest, CmpDefinesFlagsWithoutWritingBack) {
    // 48 39 D8  cmp rax, rbx  |  C3
    const std::array<u8, 4> code{0x48, 0x39, 0xD8, 0xC3};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    EXPECT_EQ(count_opcode(f, Opcode::Cmp), 1u);
    const Inst* cmp = find_opcode(f, Opcode::Cmp);
    ASSERT_NE(cmp, nullptr);
    EXPECT_TRUE(dbt::ir::defines_flags(cmp->opcode));
}

TEST_F(LifterTest, AddressArithmeticDoesNotClobberFlags) {
    // cmp rax, rbx  |  mov rcx, [rdx+16]  |  je +4
    // The MOV's address computation must not disturb the flags the JE consumes.
    const std::array<u8, 9> code{0x48, 0x39, 0xD8,        // cmp rax, rbx
                                 0x48, 0x8B, 0x4A, 0x10,  // mov rcx, [rdx+16]
                                 0x74, 0x04};             // je +4
    const auto res = lift_ok(code);

    const Function& f = res.function;
    const Inst* branch = find_opcode(f, Opcode::Branch);
    ASSERT_NE(branch, nullptr) << f.to_string();

    // The branch must consume the CMP, not anything the address math produced.
    const Inst& producer = f.inst(branch->operands[0]);
    EXPECT_EQ(producer.opcode, Opcode::Cmp) << f.to_string();
    EXPECT_TRUE(dbt::ir::defines_flags(producer.opcode));

    // Addressing used the flag-free opcodes.
    EXPECT_GE(count_opcode(f, Opcode::AddrAdd), 1u);
}

// --- Control flow ----------------------------------------------------------

TEST_F(LifterTest, JmpProducesExitBlockCarryingTargetRip) {
    // EB 05  jmp +5
    const std::array<u8, 2> code{0xEB, 0x05};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    ASSERT_EQ(f.block_count(), 2u);
    const Inst* jump = find_opcode(f, Opcode::Jump);
    ASSERT_NE(jump, nullptr);

    // The exit block writes the branch target into RIP and returns.
    const dbt::ir::BasicBlock& exit = f.block(jump->true_block);
    EXPECT_EQ(exit.guest_addr, kBase + 2 + 5);
    ASSERT_EQ(exit.insts.size(), 3u);
    const Inst& konst = f.inst(exit.insts[0]);
    EXPECT_EQ(konst.opcode, Opcode::Const);
    EXPECT_EQ(konst.imm, static_cast<dbt::i64>(kBase + 2 + 5));
    const Inst& store = f.inst(exit.insts[1]);
    EXPECT_EQ(store.opcode, Opcode::StoreGuestReg);
    EXPECT_EQ(store.guest_reg, X86Reg::Rip);
    EXPECT_EQ(f.inst(exit.insts[2]).opcode, Opcode::Return);
}

TEST_F(LifterTest, JccProducesTakenAndFallthroughExits) {
    // 48 39 D8  cmp rax, rbx  |  74 10  je +0x10
    const std::array<u8, 5> code{0x48, 0x39, 0xD8, 0x74, 0x10};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    ASSERT_EQ(f.block_count(), 3u);  // body + two exits

    const Inst* branch = find_opcode(f, Opcode::Branch);
    ASSERT_NE(branch, nullptr);
    EXPECT_EQ(branch->cond, Cond::Equal);
    EXPECT_NE(branch->true_block, branch->false_block);

    EXPECT_EQ(f.block(branch->true_block).guest_addr, kBase + 5 + 0x10);
    EXPECT_EQ(f.block(branch->false_block).guest_addr, kBase + 5);
}

TEST_F(LifterTest, RetPopsReturnAddressAndUnwindsStack) {
    const std::array<u8, 1> code{0xC3};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    // rsp load, [rsp] load, store rip, const 8, addr_add, store rsp, return
    EXPECT_EQ(f.inst(0).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(f.inst(0).guest_reg, X86Reg::Rsp);
    EXPECT_EQ(f.inst(1).opcode, Opcode::LoadMem);
    EXPECT_EQ(f.inst(2).opcode, Opcode::StoreGuestReg);
    EXPECT_EQ(f.inst(2).guest_reg, X86Reg::Rip);
    EXPECT_EQ(f.inst(3).opcode, Opcode::Const);
    EXPECT_EQ(f.inst(3).imm, 8);
    EXPECT_EQ(f.inst(4).opcode, Opcode::AddrAdd);
    EXPECT_EQ(f.inst(5).opcode, Opcode::StoreGuestReg);
    EXPECT_EQ(f.inst(5).guest_reg, X86Reg::Rsp);
    EXPECT_EQ(f.inst(6).opcode, Opcode::Return);
}

TEST_F(LifterTest, LiftingStopsAtTheFirstTerminator) {
    // ret, then bytes that must never be lifted.
    const std::array<u8, 4> code{0xC3, 0x48, 0x89, 0xD8};
    const auto res = lift_ok(code);
    EXPECT_EQ(res.guest_inst_count, 1u);
    EXPECT_EQ(count_opcode(res.function, Opcode::StoreGuestReg), 2u);  // rip, rsp
}

TEST_F(LifterTest, RunningOffTheEndStillProducesVerifiableIr) {
    // A MOV with no terminator after it.
    const std::array<u8, 3> code{0x48, 0x89, 0xD8};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    const Inst* jump = find_opcode(f, Opcode::Jump);
    ASSERT_NE(jump, nullptr);
    EXPECT_EQ(f.block(jump->true_block).guest_addr, kBase + 3);
}

// --- Error handling --------------------------------------------------------

TEST_F(LifterTest, DecodeFailurePropagatesWithAddress) {
    // mov rax, rbx, then an illegal opcode.
    const std::array<u8, 5> code{0x48, 0x89, 0xD8, 0x06, 0x00};
    const auto res = lifter.lift_block(code, kBase);

    EXPECT_FALSE(res.ok());
    EXPECT_EQ(res.error, LiftError::DecodeFailed);
    EXPECT_EQ(res.decode_error, dbt::decoder::DecodeError::InvalidInstruction);
    EXPECT_EQ(res.error_addr, kBase + 3);
    EXPECT_EQ(res.guest_inst_count, 1u);
}

TEST_F(LifterTest, UnsupportedInstructionIsReported) {
    // 90  nop -- decodes, but is outside the subset.
    const std::array<u8, 1> code{0x90};
    const auto res = lifter.lift_block(code, kBase);

    EXPECT_FALSE(res.ok());
    EXPECT_EQ(res.error, LiftError::DecodeFailed);
    EXPECT_EQ(res.decode_error, dbt::decoder::DecodeError::UnsupportedInstruction);
}

TEST_F(LifterTest, JccWithoutPrecedingFlagsIsRejected) {
    // 74 10  je +0x10, with nothing defining flags first.
    const std::array<u8, 2> code{0x74, 0x10};
    const auto res = lifter.lift_block(code, kBase);

    EXPECT_FALSE(res.ok());
    EXPECT_EQ(res.error, LiftError::FlagsUnavailable);
    EXPECT_EQ(res.error_addr, kBase);
}

TEST_F(LifterTest, IndirectJumpPublishesItsTargetAsTheNextRip) {
    // FF E0  jmp rax -- the target is only known at run time, so it is handed
    // to the dispatcher through CpuState.rip rather than becoming a branch.
    const std::array<u8, 2> code{0xFF, 0xE0};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    EXPECT_EQ(f.inst(0).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(f.inst(0).guest_reg, X86Reg::Rax);
    EXPECT_EQ(f.inst(1).opcode, Opcode::StoreGuestReg);
    EXPECT_EQ(f.inst(1).guest_reg, X86Reg::Rip);
    EXPECT_EQ(f.inst(2).opcode, Opcode::Return);
    // No exit block is built, because there is no constant target to name.
    EXPECT_EQ(f.block_count(), 1u);
}

TEST_F(LifterTest, IndirectJumpThroughMemoryLoadsItsTarget) {
    // FF 23  jmp [rbx]
    const std::array<u8, 2> code{0xFF, 0x23};
    const auto res = lift_ok(code);
    EXPECT_EQ(count_opcode(res.function, Opcode::LoadMem), 1u);
    EXPECT_EQ(count_opcode(res.function, Opcode::Return), 1u);
}

TEST_F(LifterTest, IndirectCallReadsTheTargetBeforePushing) {
    // FF D0  call rax -- x86 evaluates the operand against the pre-call RSP,
    // so the target read must precede the push.
    const std::array<u8, 2> code{0xFF, 0xD0};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    EXPECT_EQ(f.inst(0).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(f.inst(0).guest_reg, X86Reg::Rax);
    // One store for the pushed return address.
    EXPECT_EQ(count_opcode(f, Opcode::StoreMem), 1u);
    EXPECT_EQ(f.inst(static_cast<dbt::ir::InstId>(f.inst_count() - 1)).opcode,
              Opcode::Return);
}

TEST_F(LifterTest, InstructionBudgetIsEnforced) {
    const Lifter bounded(Lifter::Options{.max_guest_insts = 2});
    // Three MOVs, no terminator.
    const std::array<u8, 9> code{0x48, 0x89, 0xD8, 0x48, 0x89, 0xD8, 0x48, 0x89, 0xD8};
    const auto res = bounded.lift_block(code, kBase);

    EXPECT_FALSE(res.ok());
    EXPECT_EQ(res.error, LiftError::BlockTooLong);
    EXPECT_EQ(res.guest_inst_count, 2u);

    // Even when cut short, the emitted IR must still be well-formed.
    std::string error;
    EXPECT_TRUE(res.function.verify(&error)) << error;
}

TEST_F(LifterTest, EmptyInputProducesAnImmediateExit) {
    const auto res = lift_ok(std::span<const u8>{});
    EXPECT_EQ(res.guest_inst_count, 0u);
    EXPECT_EQ(res.function.block_count(), 2u);
}

// --- Multi-instruction blocks ----------------------------------------------

TEST_F(LifterTest, StraightLineSequenceLiftsInOrder) {
    // mov rax, 1 | add rax, rbx | cmp rax, rcx | jne +2
    const std::array<u8, 15> code{
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00,  // mov rax, 1
        0x48, 0x01, 0xD8,                          // add rax, rbx
        0x48, 0x39, 0xC8,                          // cmp rax, rcx
        0x75, 0x02                                 // jne +2
    };

    const auto res = lift_ok(code);
    EXPECT_EQ(res.guest_inst_count, 4u);

    const Function& f = res.function;
    EXPECT_EQ(count_opcode(f, Opcode::Add), 1u);
    EXPECT_EQ(count_opcode(f, Opcode::Cmp), 1u);

    const Inst* branch = find_opcode(f, Opcode::Branch);
    ASSERT_NE(branch, nullptr);
    EXPECT_EQ(branch->cond, Cond::NotEqual);
    // The branch consumes the CMP, which is the latest flag definition.
    EXPECT_EQ(f.inst(branch->operands[0]).opcode, Opcode::Cmp);
}

TEST_F(LifterTest, LatestFlagDefinitionWins) {
    // cmp rax, rbx | add rcx, rdx | je +2
    // The JE must consume the ADD's flags, not the earlier CMP's.
    const std::array<u8, 8> code{0x48, 0x39, 0xD8,  // cmp rax, rbx
                                 0x48, 0x01, 0xD1,  // add rcx, rdx
                                 0x74, 0x02};       // je +2
    const auto res = lift_ok(code);

    const Function& f = res.function;
    const Inst* branch = find_opcode(f, Opcode::Branch);
    ASSERT_NE(branch, nullptr);
    EXPECT_EQ(f.inst(branch->operands[0]).opcode, Opcode::Add) << f.to_string();
}

// --- Stack and calls -------------------------------------------------------

TEST_F(LifterTest, PushDecrementsRspThenStores) {
    // 55  push rbp  |  C3  ret
    const std::array<u8, 2> code{0x55, 0xC3};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    // The source is read before RSP moves: load rbp, load rsp, const -8,
    // addr_add, store rsp, store_mem.
    EXPECT_EQ(f.inst(0).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(f.inst(0).guest_reg, X86Reg::Rbp);
    EXPECT_EQ(f.inst(1).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(f.inst(1).guest_reg, X86Reg::Rsp);
    EXPECT_EQ(f.inst(2).opcode, Opcode::Const);
    EXPECT_EQ(f.inst(2).imm, -8);
    EXPECT_EQ(f.inst(3).opcode, Opcode::AddrAdd);
    EXPECT_EQ(f.inst(4).opcode, Opcode::StoreGuestReg);
    EXPECT_EQ(f.inst(4).guest_reg, X86Reg::Rsp);
    EXPECT_EQ(f.inst(5).opcode, Opcode::StoreMem);
    // The store goes to the *new* stack top.
    EXPECT_EQ(f.inst(5).operands[0], 3u);
}

TEST_F(LifterTest, PopLoadsThenIncrementsRsp) {
    // 5D  pop rbp  |  C3
    const std::array<u8, 2> code{0x5D, 0xC3};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    EXPECT_EQ(f.inst(0).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(f.inst(0).guest_reg, X86Reg::Rsp);
    EXPECT_EQ(f.inst(1).opcode, Opcode::LoadMem);
    EXPECT_EQ(f.inst(2).opcode, Opcode::Const);
    EXPECT_EQ(f.inst(2).imm, 8);
    EXPECT_EQ(f.inst(3).opcode, Opcode::AddrAdd);
    EXPECT_EQ(f.inst(4).guest_reg, X86Reg::Rsp);
    // The destination write comes last, so `pop rsp` would keep the load.
    EXPECT_EQ(f.inst(5).opcode, Opcode::StoreGuestReg);
    EXPECT_EQ(f.inst(5).guest_reg, X86Reg::Rbp);
}

TEST_F(LifterTest, CallPushesTheReturnAddressAndJumps) {
    // E8 10 00 00 00  call +0x10
    const std::array<u8, 5> code{0xE8, 0x10, 0x00, 0x00, 0x00};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    // The return address is the instruction after the call.
    EXPECT_EQ(f.inst(0).opcode, Opcode::Const);
    EXPECT_EQ(f.inst(0).imm, static_cast<dbt::i64>(kBase + 5));
    EXPECT_EQ(count_opcode(f, Opcode::StoreMem), 1u);

    const Inst* jump = find_opcode(f, Opcode::Jump);
    ASSERT_NE(jump, nullptr) << f.to_string();
    EXPECT_EQ(f.block(jump->true_block).guest_addr, kBase + 5 + 0x10);
}

TEST_F(LifterTest, LeaComputesAnAddressWithoutLoadingFromIt) {
    // 48 8D 44 1A 08  lea rax, [rdx + rbx + 8]  |  C3
    const std::array<u8, 6> code{0x48, 0x8D, 0x44, 0x1A, 0x08, 0xC3};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    EXPECT_GE(count_opcode(f, Opcode::AddrAdd), 2u);
    // The RET lowering loads [rsp]; LEA itself must not add a second load.
    EXPECT_EQ(count_opcode(f, Opcode::LoadMem), 1u) << f.to_string();
}

// --- Logical operations ----------------------------------------------------

TEST_F(LifterTest, LogicalOpsLowerAndDefineFlags) {
    const std::array<u8, 4> and_code{0x48, 0x21, 0xD8, 0xC3};
    const std::array<u8, 4> or_code{0x48, 0x09, 0xD8, 0xC3};
    const std::array<u8, 4> xor_code{0x48, 0x31, 0xD8, 0xC3};

    EXPECT_EQ(count_opcode(lift_ok(and_code).function, Opcode::And), 1u);
    EXPECT_EQ(count_opcode(lift_ok(or_code).function, Opcode::Or), 1u);
    EXPECT_EQ(count_opcode(lift_ok(xor_code).function, Opcode::Xor), 1u);

    static_assert(dbt::ir::defines_flags(Opcode::And));
    static_assert(dbt::ir::defines_flags(Opcode::Or));
    static_assert(dbt::ir::defines_flags(Opcode::Xor));
}

TEST_F(LifterTest, TestDefinesFlagsWithoutWritingBack) {
    // 48 85 D8  test rax, rbx  |  74 02  je +2
    const std::array<u8, 5> code{0x48, 0x85, 0xD8, 0x74, 0x02};
    const auto res = lift_ok(code);

    const Function& f = res.function;
    const Inst* branch = find_opcode(f, Opcode::Branch);
    ASSERT_NE(branch, nullptr);
    EXPECT_EQ(f.inst(branch->operands[0]).opcode, Opcode::Test);
    EXPECT_TRUE(dbt::ir::defines_flags(f.inst(branch->operands[0]).opcode));
}

TEST_F(LifterTest, NotPreservesTheFlagsAnEarlierCompareSet) {
    // cmp rax, rbx | not rcx | je +2
    // x86 NOT does not touch EFLAGS, so the JE must still see the CMP.
    const std::array<u8, 8> code{0x48, 0x39, 0xD8,  // cmp rax, rbx
                                 0x48, 0xF7, 0xD1,  // not rcx
                                 0x74, 0x02};       // je +2
    const auto res = lift_ok(code);

    const Function& f = res.function;
    const Inst* branch = find_opcode(f, Opcode::Branch);
    ASSERT_NE(branch, nullptr) << f.to_string();
    EXPECT_EQ(f.inst(branch->operands[0]).opcode, Opcode::Cmp) << f.to_string();
    static_assert(!dbt::ir::defines_flags(Opcode::Not));
}

TEST_F(LifterTest, NegDefinesFlagsWithSubtractionSemantics) {
    // 48 F7 D8  neg rax  |  C3
    const std::array<u8, 4> code{0x48, 0xF7, 0xD8, 0xC3};
    const auto res = lift_ok(code);

    EXPECT_EQ(count_opcode(res.function, Opcode::Neg), 1u);
    // NEG is a subtraction from zero, so its carry follows the x86 borrow
    // convention and must be inverted for ARM64.
    static_assert(dbt::ir::flags_from_subtraction(Opcode::Neg));
    static_assert(!dbt::ir::flags_from_subtraction(Opcode::Add));
}

// --- A real compiled function ----------------------------------------------

TEST_F(LifterTest, LiftsAClangCompiledFunctionProlgueAndBody) {
    // long add(long a, long b) { return a + b; }  -- clang -O0, System V:
    //   push rbp
    //   mov  rbp, rsp
    //   mov  [rbp-8], rdi
    //   mov  [rbp-16], rsi
    //   mov  rax, [rbp-8]
    //   add  rax, [rbp-16]
    //   pop  rbp
    //   ret
    const std::array<u8, 22> code{
        0x55,                                // push rbp
        0x48, 0x89, 0xE5,                    // mov rbp, rsp
        0x48, 0x89, 0x7D, 0xF8,              // mov [rbp-8], rdi
        0x48, 0x89, 0x75, 0xF0,              // mov [rbp-16], rsi
        0x48, 0x8B, 0x45, 0xF8,              // mov rax, [rbp-8]
        0x48, 0x03, 0x45, 0xF0,              // add rax, [rbp-16]
        0x5D,                                // pop rbp
        0xC3                                 // ret
    };

    const auto res = lift_ok(code);
    EXPECT_EQ(res.guest_inst_count, 8u);
    EXPECT_EQ(count_opcode(res.function, Opcode::Add), 1u);
}

// --- Shifts and steps ------------------------------------------------------

TEST_F(LifterTest, ImmediateShiftsLowerWithTheirCount) {
    // 48 C1 E0 03  shl rax, 3  |  C3
    const std::array<u8, 5> code{0x48, 0xC1, 0xE0, 0x03, 0xC3};
    const auto res = lift_ok(code);

    const Inst* shifted = find_opcode(res.function, Opcode::Shl);
    ASSERT_NE(shifted, nullptr) << res.function.to_string();
    EXPECT_EQ(shifted->imm, 3);
    EXPECT_EQ(shifted->operand_count, 1);
}

TEST_F(LifterTest, VariableShiftsAreRefused) {
    // 48 D3 E0  shl rax, cl -- the count is a register, which has no lowering.
    const std::array<u8, 3> code{0x48, 0xD3, 0xE0};
    const auto res = lifter.lift_block(code, kBase);
    EXPECT_FALSE(res.ok());
    EXPECT_EQ(res.error, LiftError::UnsupportedOperand);
}

TEST_F(LifterTest, IncAndDecLowerToTheirOwnOpcodes) {
    const std::array<u8, 4> inc_code{0x48, 0xFF, 0xC0, 0xC3};  // inc rax | ret
    const std::array<u8, 4> dec_code{0x48, 0xFF, 0xC8, 0xC3};  // dec rax | ret

    EXPECT_EQ(count_opcode(lift_ok(inc_code).function, Opcode::Inc), 1u);
    EXPECT_EQ(count_opcode(lift_ok(dec_code).function, Opcode::Dec), 1u);
    // They stay distinct from Add/Sub precisely because their flag semantics
    // differ: x86 preserves CF across INC and DEC.
    EXPECT_EQ(count_opcode(lift_ok(inc_code).function, Opcode::Add), 0u);
}

// --- Extending moves -------------------------------------------------------

TEST_F(LifterTest, MovzxAndMovsxCarryTheirSourceWidth) {
    // 0F B6 C3  movzx eax, bl  |  C3
    const std::array<u8, 4> zx{0x0F, 0xB6, 0xC3, 0xC3};
    const auto zx_res = lift_ok(zx);
    const Inst* zext = find_opcode(zx_res.function, Opcode::ZeroExtend);
    ASSERT_NE(zext, nullptr) << zx_res.function.to_string();
    EXPECT_EQ(zext->imm, 8);

    // 48 0F BE C3  movsx rax, bl  |  C3
    const std::array<u8, 5> sx{0x48, 0x0F, 0xBE, 0xC3, 0xC3};
    const auto sx_res = lift_ok(sx);
    const Inst* sext = find_opcode(sx_res.function, Opcode::SignExtend);
    ASSERT_NE(sext, nullptr) << sx_res.function.to_string();
    EXPECT_EQ(sext->imm, 8);
}

TEST_F(LifterTest, RegisterSourcesAreReadAtFullWidth) {
    // The byte lives in the low bits of the whole guest register, so the load
    // is 64-bit and the extend does the narrowing -- no I8 register read.
    const std::array<u8, 4> code{0x0F, 0xB6, 0xC3, 0xC3};  // movzx eax, bl
    const auto res = lift_ok(code);
    EXPECT_EQ(res.function.inst(0).opcode, Opcode::LoadGuestReg);
    EXPECT_EQ(res.function.inst(0).guest_reg, X86Reg::Rbx);
    EXPECT_EQ(res.function.inst(0).type, dbt::ir::Type::I64);
}

TEST_F(LifterTest, MovzxFromMemoryEmitsANarrowLoad) {
    // 0F B6 03  movzx eax, byte [rbx]  |  C3
    const std::array<u8, 4> code{0x0F, 0xB6, 0x03, 0xC3};
    const auto res = lift_ok(code);

    const Inst* load = find_opcode(res.function, Opcode::LoadMem);
    ASSERT_NE(load, nullptr) << res.function.to_string();
    EXPECT_EQ(load->type, dbt::ir::Type::I8);
}

TEST_F(LifterTest, ExtendingMovesPreserveFlags) {
    // cmp rax, rbx | movzx ecx, bl | je +2
    // x86 MOVZX does not touch EFLAGS, so the JE must still see the CMP.
    const std::array<u8, 8> code{0x48, 0x39, 0xD8,  // cmp rax, rbx
                                 0x0F, 0xB6, 0xCB,  // movzx ecx, bl
                                 0x74, 0x02};       // je +2
    const auto res = lift_ok(code);

    const Inst* branch = find_opcode(res.function, Opcode::Branch);
    ASSERT_NE(branch, nullptr) << res.function.to_string();
    EXPECT_EQ(res.function.inst(branch->operands[0]).opcode, Opcode::Cmp);
    static_assert(!dbt::ir::defines_flags(Opcode::ZeroExtend));
    static_assert(!dbt::ir::defines_flags(Opcode::SignExtend));
}

TEST_F(LifterTest, ErrorNames) {
    EXPECT_EQ(dbt::frontend::to_string(LiftError::None), "none");
    EXPECT_EQ(dbt::frontend::to_string(LiftError::DecodeFailed), "decode-failed");
    EXPECT_EQ(dbt::frontend::to_string(LiftError::FlagsUnavailable),
              "flags-unavailable");
    EXPECT_EQ(dbt::frontend::to_string(LiftError::IndirectBranch), "indirect-branch");
}

}  // namespace
