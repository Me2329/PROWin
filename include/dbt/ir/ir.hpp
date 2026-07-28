#pragma once

#include <array>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "dbt/common/types.hpp"
#include "dbt/decoder/decoder.hpp"

namespace dbt::ir {

using InstId = u32;
using BlockId = u32;

inline constexpr InstId kNoInst = std::numeric_limits<InstId>::max();
inline constexpr BlockId kNoBlock = std::numeric_limits<BlockId>::max();

/// Maximum SSA operands any opcode takes. StoreMem (address, value) is the
/// widest in the supported subset.
inline constexpr usize kMaxIrOperands = 2;

/// Width of a value. The translator models integer widths only.
enum class Type : u8 {
    I8 = 0,
    I16,
    I32,
    I64,
    /// Produced by flag-defining instructions; consumed only by Branch.
    Flags,
};

[[nodiscard]] constexpr u16 bit_width(Type type) noexcept {
    switch (type) {
    case Type::I8:
        return 8;
    case Type::I16:
        return 16;
    case Type::I32:
        return 32;
    case Type::I64:
        return 64;
    case Type::Flags:
        break;
    }
    return 0;
}

enum class Opcode : u8 {
    /// imm -> value
    Const = 0,
    /// guest_reg -> value
    LoadGuestReg,
    /// (value) -> guest_reg
    StoreGuestReg,
    /// (address) -> value
    LoadMem,
    /// (address, value)
    StoreMem,
    /// (lhs, rhs) -> value, also defines flags
    Add,
    /// (lhs, rhs) -> value, also defines flags
    Sub,
    /// (lhs, rhs) -> flags only
    Cmp,
    /// (lhs, rhs) -> value, WITHOUT touching flags.
    /// Used for effective-address arithmetic, which must not clobber EFLAGS.
    AddrAdd,
    /// (value) -> value shifted left by `imm`, without touching flags.
    AddrShl,
    /// (lhs, rhs) -> value, also defines flags (CF and OF cleared, as x86 does)
    And,
    Or,
    Xor,
    /// (lhs, rhs) -> flags only; the AND-based counterpart of Cmp.
    Test,
    /// (value) -> ~value. x86 NOT leaves the flags untouched.
    Not,
    /// (value) -> -value, also defines flags with subtraction semantics.
    Neg,
    /// (value) shifted by `imm`. Only ZF and SF survive the lowering.
    Shl,
    Shr,
    Sar,
    /// (value) +/- 1. x86 deliberately preserves CF, which ARM64 cannot.
    Inc,
    Dec,
    /// (value) widened from its low `imm` bits. x86 MOVZX/MOVSX leave EFLAGS
    /// untouched, so neither defines any flag.
    ZeroExtend,
    SignExtend,
    /// unconditional transfer to true_block
    Jump,
    /// (flags) -> true_block if cond holds, else false_block
    Branch,
    /// leaves the translated region
    Return,
};

[[nodiscard]] constexpr bool is_terminator(Opcode op) noexcept {
    return op == Opcode::Jump || op == Opcode::Branch || op == Opcode::Return;
}

/// True when the opcode produces an SSA value other instructions may reference.
[[nodiscard]] constexpr bool defines_value(Opcode op) noexcept {
    switch (op) {
    case Opcode::Const:
    case Opcode::LoadGuestReg:
    case Opcode::LoadMem:
    case Opcode::Add:
    case Opcode::Sub:
    case Opcode::Cmp:
    case Opcode::AddrAdd:
    case Opcode::AddrShl:
    case Opcode::And:
    case Opcode::Or:
    case Opcode::Xor:
    case Opcode::Test:
    case Opcode::Not:
    case Opcode::Neg:
    case Opcode::Shl:
    case Opcode::Shr:
    case Opcode::Sar:
    case Opcode::Inc:
    case Opcode::Dec:
    case Opcode::ZeroExtend:
    case Opcode::SignExtend:
        return true;
    case Opcode::StoreGuestReg:
    case Opcode::StoreMem:
    case Opcode::Jump:
    case Opcode::Branch:
    case Opcode::Return:
        break;
    }
    return false;
}

/// The x86 status flags the translator models.
///
/// PF and AF are absent by design: AArch64 has no parity flag and no equivalent
/// of the adjust flag, so conditions depending on them are refused outright
/// rather than approximated.
namespace flag {
inline constexpr u8 kNone = 0;
inline constexpr u8 kCarry = 1u << 0;
inline constexpr u8 kZero = 1u << 1;
inline constexpr u8 kSign = 1u << 2;
inline constexpr u8 kOverflow = 1u << 3;
inline constexpr u8 kAll = kCarry | kZero | kSign | kOverflow;
}  // namespace flag

/// Which flags the opcode leaves in a *trustworthy* state after lowering.
///
/// This is deliberately narrower than "which flags x86 writes". A shift writes
/// CF from the last bit shifted out, but our LSL/LSR/ASR + TST lowering cannot
/// reproduce that, so CF is reported as not defined and any branch needing it
/// is refused. INC/DEC are the mirror image: x86 preserves CF on purpose, while
/// the ARM64 add/subtract that implements them overwrites it.
[[nodiscard]] constexpr u8 defined_flags(Opcode op) noexcept {
    switch (op) {
    case Opcode::Add:
    case Opcode::Sub:
    case Opcode::Cmp:
    case Opcode::Neg:
        return flag::kAll;
    case Opcode::And:
    case Opcode::Or:
    case Opcode::Xor:
    case Opcode::Test:
        // x86 clears CF and OF for logical ops; ARM64 TST clears C and V, so
        // all four agree.
        return flag::kAll;
    case Opcode::Shl:
    case Opcode::Shr:
    case Opcode::Sar:
        return flag::kZero | flag::kSign;
    case Opcode::Inc:
    case Opcode::Dec:
        return flag::kZero | flag::kSign | flag::kOverflow;
    default:
        return flag::kNone;
    }
}

/// Which x86 flags a condition actually reads.
///
/// Paired with defined_flags(), this lets the backend refuse a branch whose
/// condition depends on a flag the producing instruction could not model --
/// for instance a carry test after a shift -- instead of branching on a value
/// that happens to be sitting in NZCV.
[[nodiscard]] constexpr u8 required_flags(decoder::Cond cond) noexcept {
    switch (cond) {
    case decoder::Cond::Equal:
    case decoder::Cond::NotEqual:
        return flag::kZero;
    case decoder::Cond::Sign:
    case decoder::Cond::NotSign:
        return flag::kSign;
    case decoder::Cond::Overflow:
    case decoder::Cond::NotOverflow:
        return flag::kOverflow;
    case decoder::Cond::Below:
    case decoder::Cond::AboveEqual:
        return flag::kCarry;
    case decoder::Cond::BelowEqual:
    case decoder::Cond::Above:
        return flag::kCarry | flag::kZero;
    case decoder::Cond::Less:
    case decoder::Cond::GreaterEqual:
        return flag::kSign | flag::kOverflow;
    case decoder::Cond::LessEqual:
    case decoder::Cond::Greater:
        return flag::kSign | flag::kOverflow | flag::kZero;
    case decoder::Cond::Parity:
    case decoder::Cond::NotParity:
    case decoder::Cond::None:
        break;
    }
    // Parity has no ARM64 equivalent at all; demanding every flag guarantees
    // the check below rejects it.
    return flag::kAll;
}

/// True when the opcode updates the emulated EFLAGS state.
[[nodiscard]] constexpr bool defines_flags(Opcode op) noexcept {
    // Opcode::Not is deliberately absent: x86 NOT preserves EFLAGS.
    return defined_flags(op) != flag::kNone;
}

/// True when the opcode's flags follow subtraction semantics, where x86 CF
/// means "borrow" and the AArch64 carry sense is inverted.
[[nodiscard]] constexpr bool flags_from_subtraction(Opcode op) noexcept {
    return op == Opcode::Sub || op == Opcode::Cmp || op == Opcode::Neg;
}

[[nodiscard]] constexpr usize operand_arity(Opcode op) noexcept {
    switch (op) {
    case Opcode::Const:
    case Opcode::LoadGuestReg:
    case Opcode::Jump:
    case Opcode::Return:
        return 0;
    case Opcode::StoreGuestReg:
    case Opcode::LoadMem:
    case Opcode::Branch:
    case Opcode::AddrShl:
    case Opcode::Not:
    case Opcode::Neg:
    case Opcode::Shl:
    case Opcode::Shr:
    case Opcode::Sar:
    case Opcode::Inc:
    case Opcode::Dec:
    case Opcode::ZeroExtend:
    case Opcode::SignExtend:
        return 1;
    case Opcode::StoreMem:
    case Opcode::Add:
    case Opcode::Sub:
    case Opcode::Cmp:
    case Opcode::AddrAdd:
    case Opcode::And:
    case Opcode::Or:
    case Opcode::Xor:
    case Opcode::Test:
        return 2;
    }
    return 0;
}

/// A single SSA instruction.
///
/// Every instruction defines at most one value, and that value is named by the
/// instruction's own InstId, so no separate value table is needed.
struct Inst {
    Opcode opcode = Opcode::Const;
    Type type = Type::I64;
    u8 operand_count = 0;
    std::array<InstId, kMaxIrOperands> operands{kNoInst, kNoInst};

    /// Const payload.
    i64 imm = 0;
    /// LoadGuestReg / StoreGuestReg target.
    decoder::X86Reg guest_reg = decoder::X86Reg::None;
    /// Branch predicate.
    decoder::Cond cond = decoder::Cond::None;
    /// Jump target, and the taken edge of a Branch.
    BlockId true_block = kNoBlock;
    /// The not-taken edge of a Branch.
    BlockId false_block = kNoBlock;
    /// Guest address this instruction was lifted from, for diagnostics.
    GuestAddr guest_addr = 0;

    [[nodiscard]] constexpr bool is_terminator() const noexcept {
        return ir::is_terminator(opcode);
    }
    [[nodiscard]] constexpr bool defines_value() const noexcept {
        return ir::defines_value(opcode);
    }
};

struct BasicBlock {
    BlockId id = kNoBlock;
    /// Guest address this block starts at.
    GuestAddr guest_addr = 0;
    std::vector<InstId> insts;
};

/// A translated region: a flat instruction pool plus the blocks indexing it.
class Function {
public:
    [[nodiscard]] BlockId create_block(GuestAddr guest_addr = 0);

    /// Appends `inst` to `block` and returns its id.
    /// Throws std::out_of_range if `block` is not a valid block id.
    InstId append(BlockId block, const Inst& inst);

    [[nodiscard]] const Inst& inst(InstId id) const;
    [[nodiscard]] const BasicBlock& block(BlockId id) const;

    [[nodiscard]] usize inst_count() const noexcept { return insts_.size(); }
    [[nodiscard]] usize block_count() const noexcept { return blocks_.size(); }

    [[nodiscard]] const std::vector<Inst>& insts() const noexcept { return insts_; }
    [[nodiscard]] const std::vector<BasicBlock>& blocks() const noexcept {
        return blocks_;
    }

    [[nodiscard]] BlockId entry() const noexcept { return entry_; }
    void set_entry(BlockId id) noexcept { entry_ = id; }

    /// True when `block` ends in a terminator.
    [[nodiscard]] bool is_sealed(BlockId block) const;

    /// Checks the structural invariants: one terminator per block and only in
    /// final position, operands referencing value-defining instructions defined
    /// earlier, arity matching the opcode, and branch targets in range.
    ///
    /// On failure `error`, when non-null, receives a human-readable reason.
    [[nodiscard]] bool verify(std::string* error = nullptr) const;

    /// Multi-line textual dump, for test diagnostics.
    [[nodiscard]] std::string to_string() const;

private:
    std::vector<Inst> insts_;
    std::vector<BasicBlock> blocks_;
    BlockId entry_ = kNoBlock;
};

/// Convenience layer for emitting into a Function.
class IRBuilder {
public:
    explicit IRBuilder(Function& function) noexcept : func_(&function) {}

    /// Creates a block and makes it current. The first block created also
    /// becomes the function entry.
    BlockId create_block(GuestAddr guest_addr = 0);
    void set_insert_point(BlockId block) noexcept { current_ = block; }
    [[nodiscard]] BlockId insert_point() const noexcept { return current_; }

    InstId const_int(i64 value, Type type = Type::I64);
    InstId load_guest_reg(decoder::X86Reg reg, Type type = Type::I64);
    /// `type` is the width of the x86 write. A 32-bit store zero-extends into
    /// the full guest register, so the backend must know which it is.
    InstId store_guest_reg(decoder::X86Reg reg, InstId value,
                           Type type = Type::I64);
    InstId load_mem(InstId address, Type type = Type::I64);
    InstId store_mem(InstId address, InstId value, Type type = Type::I64);
    InstId add(InstId lhs, InstId rhs, Type type = Type::I64);
    InstId sub(InstId lhs, InstId rhs, Type type = Type::I64);
    InstId cmp(InstId lhs, InstId rhs, Type type = Type::I64);
    InstId addr_add(InstId lhs, InstId rhs);
    InstId addr_shl(InstId value, i64 shift);
    // `and`, `or`, `xor` and `not` are alternative operator tokens in C++, so
    // these carry a bit_ prefix.
    InstId bit_and(InstId lhs, InstId rhs, Type type = Type::I64);
    InstId bit_or(InstId lhs, InstId rhs, Type type = Type::I64);
    InstId bit_xor(InstId lhs, InstId rhs, Type type = Type::I64);
    InstId test(InstId lhs, InstId rhs, Type type = Type::I64);
    InstId bit_not(InstId value, Type type = Type::I64);
    InstId neg(InstId value, Type type = Type::I64);
    /// `opcode` must be Shl, Shr or Sar.
    InstId shift(Opcode opcode, InstId value, i64 amount, Type type = Type::I64);
    /// `opcode` must be Inc or Dec.
    InstId step(Opcode opcode, InstId value, Type type = Type::I64);
    /// `opcode` must be ZeroExtend or SignExtend; `source_bits` is 8, 16 or 32.
    InstId extend(Opcode opcode, InstId value, i64 source_bits, Type type);
    InstId jump(BlockId target);
    InstId branch(InstId flags, decoder::Cond cond, BlockId taken, BlockId fallthrough);
    InstId ret();

    /// Stamps subsequent instructions with this guest address.
    void set_guest_addr(GuestAddr addr) noexcept { guest_addr_ = addr; }

private:
    InstId emit(Inst inst);

    Function* func_;
    BlockId current_ = kNoBlock;
    GuestAddr guest_addr_ = 0;
};

[[nodiscard]] std::string_view to_string(Opcode value) noexcept;
[[nodiscard]] std::string_view to_string(Type value) noexcept;

}  // namespace dbt::ir
