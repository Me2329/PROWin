#include "dbt/frontend/lifter.hpp"

namespace dbt::frontend {
namespace {

using decoder::DecodedInst;
using decoder::Mnemonic;
using decoder::Operand;
using decoder::OperandKind;
using decoder::X86Reg;

ir::Type type_for(u16 size_bits) noexcept {
    switch (size_bits) {
    case 8:
        return ir::Type::I8;
    case 16:
        return ir::Type::I16;
    case 32:
        return ir::Type::I32;
    default:
        return ir::Type::I64;
    }
}

/// log2 of an x86 SIB scale, which is always 1, 2, 4 or 8.
i64 scale_shift(u8 scale) noexcept {
    switch (scale) {
    case 2:
        return 1;
    case 4:
        return 2;
    case 8:
        return 3;
    default:
        return 0;
    }
}

/// Carries the per-block lifting state.
///
/// `last_flags_` names the most recent flag-defining instruction in the block,
/// which is what a following Jcc consumes.
class BlockLifter {
public:
    BlockLifter(ir::Function& function, GuestAddr base)
        : func_(function), builder_(function) {
        body_ = builder_.create_block(base);
    }

    [[nodiscard]] ir::BlockId body() const noexcept { return body_; }

    void begin_inst(GuestAddr addr) { builder_.set_guest_addr(addr); }

    /// Lowers one decoded instruction. Returns LiftError::None on success.
    LiftError lower(const DecodedInst& inst);

    /// Emits a terminator for a block that ran off the end without one.
    void terminate_fallthrough(GuestAddr next_addr) {
        builder_.set_insert_point(body_);
        const ir::BlockId exit = make_exit(next_addr);
        builder_.set_insert_point(body_);
        static_cast<void>(builder_.jump(exit));
    }

private:
    /// Builds a block that records `target` as the next guest RIP and returns
    /// to the dispatcher.
    ir::BlockId make_exit(GuestAddr target) {
        const ir::BlockId saved = builder_.insert_point();
        const ir::BlockId exit = func_.create_block(target);
        builder_.set_insert_point(exit);
        const ir::InstId addr = builder_.const_int(static_cast<i64>(target));
        static_cast<void>(builder_.store_guest_reg(X86Reg::Rip, addr));
        static_cast<void>(builder_.ret());
        builder_.set_insert_point(saved);
        return exit;
    }

    /// rsp -= 8, then store `value` at the new top of stack.
    ///
    /// The guest stack pointer is an ordinary mapped register, so PUSH is just
    /// address arithmetic plus a store -- no native stack is involved.
    void push_value(ir::InstId value) {
        const ir::InstId sp = builder_.load_guest_reg(X86Reg::Rsp);
        const ir::InstId slot = builder_.const_int(-8);
        const ir::InstId new_sp = builder_.addr_add(sp, slot);
        static_cast<void>(builder_.store_guest_reg(X86Reg::Rsp, new_sp));
        static_cast<void>(builder_.store_mem(new_sp, value));
    }

    /// Materialises `base + index*scale + disp` without disturbing flags.
    ir::InstId effective_address(const DecodedInst& inst, const Operand& op) {
        const decoder::MemRef& mem = op.mem;

        // The decoder already folded RIP + displacement into an absolute
        // address, so a RIP-relative operand is just a constant here.
        if (mem.rip_relative) {
            return builder_.const_int(static_cast<i64>(inst.rip_target));
        }

        ir::InstId addr = ir::kNoInst;
        if (mem.base != X86Reg::None) {
            addr = builder_.load_guest_reg(mem.base);
        }
        if (mem.index != X86Reg::None) {
            ir::InstId index = builder_.load_guest_reg(mem.index);
            const i64 shift = scale_shift(mem.scale);
            if (shift != 0) {
                index = builder_.addr_shl(index, shift);
            }
            addr = (addr == ir::kNoInst) ? index : builder_.addr_add(addr, index);
        }
        if (mem.disp != 0 || addr == ir::kNoInst) {
            const ir::InstId disp = builder_.const_int(mem.disp);
            addr = (addr == ir::kNoInst) ? disp : builder_.addr_add(addr, disp);
        }
        return addr;
    }

    /// Produces the value of a source operand.
    ir::InstId read(const DecodedInst& inst, const Operand& op, bool& ok) {
        ok = true;
        const ir::Type type = type_for(op.size_bits);
        switch (op.kind) {
        case OperandKind::Register:
            return builder_.load_guest_reg(op.reg, type);
        case OperandKind::Immediate:
            return builder_.const_int(op.imm, type);
        case OperandKind::Memory:
            return builder_.load_mem(effective_address(inst, op), type);
        case OperandKind::None:
            break;
        }
        ok = false;
        return ir::kNoInst;
    }

    /// Commits `value` to a destination operand.
    bool write(const DecodedInst& inst, const Operand& op, ir::InstId value) {
        switch (op.kind) {
        case OperandKind::Register:
            static_cast<void>(
                builder_.store_guest_reg(op.reg, value, type_for(op.size_bits)));
            return true;
        case OperandKind::Memory:
            static_cast<void>(builder_.store_mem(effective_address(inst, op), value,
                                                 type_for(op.size_bits)));
            return true;
        case OperandKind::Immediate:
        case OperandKind::None:
            break;
        }
        return false;  // an immediate is not a legal destination
    }

    ir::Function& func_;
    ir::IRBuilder builder_;
    ir::BlockId body_ = ir::kNoBlock;
    ir::InstId last_flags_ = ir::kNoInst;
};

LiftError BlockLifter::lower(const DecodedInst& inst) {
    switch (inst.mnemonic) {
    case Mnemonic::Mov: {
        if (inst.operand_count != 2) {
            return LiftError::UnsupportedOperand;
        }
        bool ok = false;
        const ir::InstId value = read(inst, inst.op(1), ok);
        if (!ok) {
            return LiftError::UnsupportedOperand;
        }
        return write(inst, inst.op(0), value) ? LiftError::None
                                              : LiftError::UnsupportedOperand;
    }

    case Mnemonic::Add:
    case Mnemonic::Sub: {
        if (inst.operand_count != 2) {
            return LiftError::UnsupportedOperand;
        }
        bool ok_lhs = false;
        bool ok_rhs = false;
        const ir::Type type = type_for(inst.op(0).size_bits);
        const ir::InstId lhs = read(inst, inst.op(0), ok_lhs);
        const ir::InstId rhs = read(inst, inst.op(1), ok_rhs);
        if (!ok_lhs || !ok_rhs) {
            return LiftError::UnsupportedOperand;
        }
        const ir::InstId result = (inst.mnemonic == Mnemonic::Add)
                                      ? builder_.add(lhs, rhs, type)
                                      : builder_.sub(lhs, rhs, type);
        // ADD/SUB both write back and define flags.
        last_flags_ = result;
        return write(inst, inst.op(0), result) ? LiftError::None
                                               : LiftError::UnsupportedOperand;
    }

    case Mnemonic::Cmp: {
        if (inst.operand_count != 2) {
            return LiftError::UnsupportedOperand;
        }
        bool ok_lhs = false;
        bool ok_rhs = false;
        const ir::InstId lhs = read(inst, inst.op(0), ok_lhs);
        const ir::InstId rhs = read(inst, inst.op(1), ok_rhs);
        if (!ok_lhs || !ok_rhs) {
            return LiftError::UnsupportedOperand;
        }
        // CMP discards the difference and keeps only the flags.
        last_flags_ = builder_.cmp(lhs, rhs, type_for(inst.op(0).size_bits));
        return LiftError::None;
    }

    case Mnemonic::Jmp: {
        if (!inst.has_branch_target) {
            return LiftError::IndirectBranch;
        }
        const ir::BlockId exit = make_exit(inst.branch_target);
        static_cast<void>(builder_.jump(exit));
        return LiftError::None;
    }

    case Mnemonic::Jcc: {
        if (last_flags_ == ir::kNoInst) {
            return LiftError::FlagsUnavailable;
        }
        if (!inst.has_branch_target) {
            return LiftError::IndirectBranch;
        }
        const ir::BlockId taken = make_exit(inst.branch_target);
        const ir::BlockId fallthrough = make_exit(inst.next_address());
        static_cast<void>(builder_.branch(last_flags_, inst.cond, taken, fallthrough));
        return LiftError::None;
    }

    case Mnemonic::Ret: {
        // RET pops the return address into RIP and unwinds the stack slot.
        const ir::InstId sp = builder_.load_guest_reg(X86Reg::Rsp);
        const ir::InstId return_addr = builder_.load_mem(sp);
        static_cast<void>(builder_.store_guest_reg(X86Reg::Rip, return_addr));
        const ir::InstId slot = builder_.const_int(8);
        const ir::InstId new_sp = builder_.addr_add(sp, slot);
        static_cast<void>(builder_.store_guest_reg(X86Reg::Rsp, new_sp));
        static_cast<void>(builder_.ret());
        return LiftError::None;
    }

    case Mnemonic::Push: {
        if (inst.operand_count != 1) {
            return LiftError::UnsupportedOperand;
        }
        bool ok = false;
        // The source is read before RSP moves, so `push rsp` stores the old
        // value, as x86 requires.
        const ir::InstId value = read(inst, inst.op(0), ok);
        if (!ok) {
            return LiftError::UnsupportedOperand;
        }
        push_value(value);
        return LiftError::None;
    }

    case Mnemonic::Pop: {
        if (inst.operand_count != 1) {
            return LiftError::UnsupportedOperand;
        }
        const ir::InstId sp = builder_.load_guest_reg(X86Reg::Rsp);
        const ir::InstId value = builder_.load_mem(sp);
        const ir::InstId slot = builder_.const_int(8);
        const ir::InstId new_sp = builder_.addr_add(sp, slot);
        static_cast<void>(builder_.store_guest_reg(X86Reg::Rsp, new_sp));
        // The destination write comes last so that `pop rsp` ends up holding
        // the loaded value rather than the adjusted pointer.
        return write(inst, inst.op(0), value) ? LiftError::None
                                              : LiftError::UnsupportedOperand;
    }

    case Mnemonic::Call: {
        if (!inst.has_branch_target) {
            return LiftError::IndirectBranch;
        }
        const ir::InstId return_addr =
            builder_.const_int(static_cast<i64>(inst.next_address()));
        push_value(return_addr);
        const ir::BlockId exit = make_exit(inst.branch_target);
        static_cast<void>(builder_.jump(exit));
        return LiftError::None;
    }

    case Mnemonic::Lea: {
        if (inst.operand_count != 2 || inst.op(1).kind != OperandKind::Memory) {
            return LiftError::UnsupportedOperand;
        }
        // LEA computes the address without dereferencing it, and leaves the
        // flags alone.
        const ir::InstId addr = effective_address(inst, inst.op(1));
        return write(inst, inst.op(0), addr) ? LiftError::None
                                             : LiftError::UnsupportedOperand;
    }

    case Mnemonic::And:
    case Mnemonic::Or:
    case Mnemonic::Xor: {
        if (inst.operand_count != 2) {
            return LiftError::UnsupportedOperand;
        }
        bool ok_lhs = false;
        bool ok_rhs = false;
        const ir::Type type = type_for(inst.op(0).size_bits);
        const ir::InstId lhs = read(inst, inst.op(0), ok_lhs);
        const ir::InstId rhs = read(inst, inst.op(1), ok_rhs);
        if (!ok_lhs || !ok_rhs) {
            return LiftError::UnsupportedOperand;
        }
        ir::InstId result = ir::kNoInst;
        if (inst.mnemonic == Mnemonic::And) {
            result = builder_.bit_and(lhs, rhs, type);
        } else if (inst.mnemonic == Mnemonic::Or) {
            result = builder_.bit_or(lhs, rhs, type);
        } else {
            result = builder_.bit_xor(lhs, rhs, type);
        }
        last_flags_ = result;
        return write(inst, inst.op(0), result) ? LiftError::None
                                               : LiftError::UnsupportedOperand;
    }

    case Mnemonic::Test: {
        if (inst.operand_count != 2) {
            return LiftError::UnsupportedOperand;
        }
        bool ok_lhs = false;
        bool ok_rhs = false;
        const ir::InstId lhs = read(inst, inst.op(0), ok_lhs);
        const ir::InstId rhs = read(inst, inst.op(1), ok_rhs);
        if (!ok_lhs || !ok_rhs) {
            return LiftError::UnsupportedOperand;
        }
        last_flags_ = builder_.test(lhs, rhs, type_for(inst.op(0).size_bits));
        return LiftError::None;
    }

    case Mnemonic::Not: {
        if (inst.operand_count != 1) {
            return LiftError::UnsupportedOperand;
        }
        bool ok = false;
        const ir::InstId value = read(inst, inst.op(0), ok);
        if (!ok) {
            return LiftError::UnsupportedOperand;
        }
        // x86 NOT preserves EFLAGS, so last_flags_ is deliberately untouched.
        const ir::InstId result =
            builder_.bit_not(value, type_for(inst.op(0).size_bits));
        return write(inst, inst.op(0), result) ? LiftError::None
                                               : LiftError::UnsupportedOperand;
    }

    case Mnemonic::Neg: {
        if (inst.operand_count != 1) {
            return LiftError::UnsupportedOperand;
        }
        bool ok = false;
        const ir::InstId value = read(inst, inst.op(0), ok);
        if (!ok) {
            return LiftError::UnsupportedOperand;
        }
        const ir::InstId result = builder_.neg(value, type_for(inst.op(0).size_bits));
        last_flags_ = result;
        return write(inst, inst.op(0), result) ? LiftError::None
                                               : LiftError::UnsupportedOperand;
    }

    case Mnemonic::Shl:
    case Mnemonic::Shr:
    case Mnemonic::Sar: {
        // Only the immediate-count forms lower; `shl reg, cl` needs a
        // variable-shift path that does not exist yet.
        if (inst.operand_count != 2 || inst.op(1).kind != OperandKind::Immediate) {
            return LiftError::UnsupportedOperand;
        }
        bool ok = false;
        const ir::InstId value = read(inst, inst.op(0), ok);
        if (!ok) {
            return LiftError::UnsupportedOperand;
        }
        const ir::Opcode opcode = (inst.mnemonic == Mnemonic::Shl) ? ir::Opcode::Shl
                                  : (inst.mnemonic == Mnemonic::Shr)
                                      ? ir::Opcode::Shr
                                      : ir::Opcode::Sar;
        const ir::InstId result = builder_.shift(opcode, value, inst.op(1).imm,
                                                 type_for(inst.op(0).size_bits));
        last_flags_ = result;
        return write(inst, inst.op(0), result) ? LiftError::None
                                               : LiftError::UnsupportedOperand;
    }

    case Mnemonic::Inc:
    case Mnemonic::Dec: {
        if (inst.operand_count != 1) {
            return LiftError::UnsupportedOperand;
        }
        bool ok = false;
        const ir::InstId value = read(inst, inst.op(0), ok);
        if (!ok) {
            return LiftError::UnsupportedOperand;
        }
        const ir::InstId result = builder_.step(
            inst.mnemonic == Mnemonic::Inc ? ir::Opcode::Inc : ir::Opcode::Dec,
            value, type_for(inst.op(0).size_bits));
        last_flags_ = result;
        return write(inst, inst.op(0), result) ? LiftError::None
                                               : LiftError::UnsupportedOperand;
    }

    case Mnemonic::Movzx:
    case Mnemonic::Movsx: {
        if (inst.operand_count != 2) {
            return LiftError::UnsupportedOperand;
        }
        const Operand& source = inst.op(1);

        ir::InstId value = ir::kNoInst;
        if (source.kind == OperandKind::Register) {
            // The narrow value occupies the low bits of the full guest
            // register, so read the whole register and let the extend select.
            value = builder_.load_guest_reg(source.reg, ir::Type::I64);
        } else if (source.kind == OperandKind::Memory) {
            value = builder_.load_mem(effective_address(inst, source),
                                      type_for(source.size_bits));
        } else {
            return LiftError::UnsupportedOperand;
        }

        // MOVZX/MOVSX leave EFLAGS untouched, so last_flags_ stays as it was.
        const ir::InstId result = builder_.extend(
            inst.mnemonic == Mnemonic::Movzx ? ir::Opcode::ZeroExtend
                                             : ir::Opcode::SignExtend,
            value, source.size_bits, type_for(inst.op(0).size_bits));
        return write(inst, inst.op(0), result) ? LiftError::None
                                               : LiftError::UnsupportedOperand;
    }

    case Mnemonic::Invalid:
        break;
    }
    return LiftError::UnsupportedOperand;
}

}  // namespace

LiftResult Lifter::lift_block(std::span<const u8> code, GuestAddr base) const {
    LiftResult result;
    BlockLifter lifter(result.function, base);

    usize offset = 0;
    for (usize count = 0; count < options_.max_guest_insts; ++count) {
        if (offset >= code.size()) {
            lifter.terminate_fallthrough(base + offset);
            return result;
        }

        const GuestAddr address = base + offset;
        const auto decoded = decoder_.decode(code.subspan(offset), address);
        if (!decoded.ok()) {
            result.error = LiftError::DecodeFailed;
            result.decode_error = decoded.error;
            result.error_addr = address;
            return result;
        }

        lifter.begin_inst(address);
        const LiftError error = lifter.lower(decoded.inst);
        if (error != LiftError::None) {
            result.error = error;
            result.error_addr = address;
            return result;
        }

        ++result.guest_inst_count;
        offset += decoded.inst.length;

        if (decoded.inst.is_terminator()) {
            return result;
        }
    }

    // Budget exhausted without reaching a terminator: close the block so the IR
    // stays well-formed, and report why it was cut short.
    lifter.terminate_fallthrough(base + offset);
    result.error = LiftError::BlockTooLong;
    result.error_addr = base + offset;
    return result;
}

std::string_view to_string(LiftError value) noexcept {
    switch (value) {
    case LiftError::None:
        return "none";
    case LiftError::DecodeFailed:
        return "decode-failed";
    case LiftError::UnsupportedOperand:
        return "unsupported-operand";
    case LiftError::FlagsUnavailable:
        return "flags-unavailable";
    case LiftError::IndirectBranch:
        return "indirect-branch";
    case LiftError::BlockTooLong:
        return "block-too-long";
    }
    return "unknown";
}

}  // namespace dbt::frontend
