#include "dbt/ir/ir.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace dbt::ir {
namespace {

std::string describe(InstId id) {
    return (id == kNoInst) ? std::string("<none>") : ("%" + std::to_string(id));
}

}  // namespace

// --- Function --------------------------------------------------------------

BlockId Function::create_block(GuestAddr guest_addr) {
    const auto id = static_cast<BlockId>(blocks_.size());
    BasicBlock blk;
    blk.id = id;
    blk.guest_addr = guest_addr;
    blocks_.push_back(std::move(blk));
    if (entry_ == kNoBlock) {
        entry_ = id;
    }
    return id;
}

InstId Function::append(BlockId block, const Inst& inst) {
    if (block >= blocks_.size()) {
        throw std::out_of_range("ir::Function::append: invalid block id");
    }
    const auto id = static_cast<InstId>(insts_.size());
    insts_.push_back(inst);
    blocks_[block].insts.push_back(id);
    return id;
}

const Inst& Function::inst(InstId id) const {
    if (id >= insts_.size()) {
        throw std::out_of_range("ir::Function::inst: invalid instruction id");
    }
    return insts_[id];
}

const BasicBlock& Function::block(BlockId id) const {
    if (id >= blocks_.size()) {
        throw std::out_of_range("ir::Function::block: invalid block id");
    }
    return blocks_[id];
}

bool Function::is_sealed(BlockId block) const {
    if (block >= blocks_.size()) {
        return false;
    }
    const BasicBlock& blk = blocks_[block];
    return !blk.insts.empty() && insts_[blk.insts.back()].is_terminator();
}

bool Function::verify(std::string* error) const {
    const auto fail = [error](std::string reason) {
        if (error != nullptr) {
            *error = std::move(reason);
        }
        return false;
    };

    if (blocks_.empty()) {
        return fail("function has no blocks");
    }
    if (entry_ >= blocks_.size()) {
        return fail("entry block id is out of range");
    }

    for (const BasicBlock& blk : blocks_) {
        if (blk.insts.empty()) {
            return fail("block " + std::to_string(blk.id) + " is empty");
        }

        for (usize i = 0; i < blk.insts.size(); ++i) {
            const InstId id = blk.insts[i];
            if (id >= insts_.size()) {
                return fail("block " + std::to_string(blk.id) +
                            " references unknown instruction " + describe(id));
            }
            const Inst& in = insts_[id];
            const bool last = (i + 1 == blk.insts.size());

            if (in.is_terminator() && !last) {
                return fail("terminator " + describe(id) +
                            " is not the last instruction of block " +
                            std::to_string(blk.id));
            }
            if (!in.is_terminator() && last) {
                return fail("block " + std::to_string(blk.id) +
                            " does not end in a terminator");
            }

            if (in.operand_count != operand_arity(in.opcode)) {
                return fail("instruction " + describe(id) + " (" +
                            std::string(ir::to_string(in.opcode)) + ") has " +
                            std::to_string(in.operand_count) +
                            " operands, expected " +
                            std::to_string(operand_arity(in.opcode)));
            }

            for (u8 operand = 0; operand < in.operand_count; ++operand) {
                const InstId use = in.operands[operand];
                if (use >= insts_.size()) {
                    return fail("instruction " + describe(id) +
                                " references unknown value " + describe(use));
                }
                if (use >= id) {
                    return fail("instruction " + describe(id) + " uses value " +
                                describe(use) + " that is not defined before it");
                }
                if (!insts_[use].defines_value()) {
                    return fail("instruction " + describe(id) + " uses " +
                                describe(use) + ", which defines no value");
                }
            }

            if (in.opcode == Opcode::Branch) {
                // ADD and SUB define flags alongside an integer result, so the
                // test is whether the producer defines flags at all -- not
                // whether its result type is Flags.
                if (!ir::defines_flags(insts_[in.operands[0]].opcode)) {
                    return fail("branch " + describe(id) +
                                " consumes a non-flags value");
                }
                if (in.cond == decoder::Cond::None) {
                    return fail("branch " + describe(id) + " has no condition");
                }
                if (in.false_block >= blocks_.size()) {
                    return fail("branch " + describe(id) +
                                " has an out-of-range fallthrough target");
                }
            }
            if (in.opcode == Opcode::Jump || in.opcode == Opcode::Branch) {
                if (in.true_block >= blocks_.size()) {
                    return fail("terminator " + describe(id) +
                                " has an out-of-range target");
                }
            }
        }
    }

    return true;
}

std::string Function::to_string() const {
    std::ostringstream out;
    for (const BasicBlock& blk : blocks_) {
        out << "block" << blk.id;
        if (blk.id == entry_) {
            out << " (entry)";
        }
        out << ":\n";
        for (const InstId id : blk.insts) {
            const Inst& in = insts_[id];
            out << "  ";
            if (in.defines_value()) {
                out << "%" << id << " = ";
            }
            out << ir::to_string(in.opcode);
            switch (in.opcode) {
            case Opcode::Const:
                out << " " << in.imm;
                break;
            case Opcode::LoadGuestReg:
            case Opcode::StoreGuestReg:
                out << " " << decoder::to_string(in.guest_reg);
                break;
            case Opcode::Jump:
                out << " block" << in.true_block;
                break;
            case Opcode::Branch:
                out << "." << decoder::to_string(in.cond) << " block"
                    << in.true_block << ", block" << in.false_block;
                break;
            default:
                break;
            }
            for (u8 operand = 0; operand < in.operand_count; ++operand) {
                out << (operand == 0 ? " " : ", ") << describe(in.operands[operand]);
            }
            out << "  ; " << ir::to_string(in.type) << "\n";
        }
    }
    return out.str();
}

// --- IRBuilder -------------------------------------------------------------

BlockId IRBuilder::create_block(GuestAddr guest_addr) {
    current_ = func_->create_block(guest_addr);
    return current_;
}

InstId IRBuilder::emit(Inst inst) {
    inst.guest_addr = guest_addr_;
    return func_->append(current_, inst);
}

InstId IRBuilder::const_int(i64 value, Type type) {
    Inst in;
    in.opcode = Opcode::Const;
    in.type = type;
    in.imm = value;
    return emit(in);
}

InstId IRBuilder::load_guest_reg(decoder::X86Reg reg, Type type) {
    Inst in;
    in.opcode = Opcode::LoadGuestReg;
    in.type = type;
    in.guest_reg = reg;
    return emit(in);
}

InstId IRBuilder::store_guest_reg(decoder::X86Reg reg, InstId value, Type type) {
    Inst in;
    in.opcode = Opcode::StoreGuestReg;
    in.type = type;
    in.guest_reg = reg;
    in.operands[0] = value;
    in.operand_count = 1;
    return emit(in);
}

InstId IRBuilder::load_mem(InstId address, Type type) {
    Inst in;
    in.opcode = Opcode::LoadMem;
    in.type = type;
    in.operands[0] = address;
    in.operand_count = 1;
    return emit(in);
}

InstId IRBuilder::store_mem(InstId address, InstId value, Type type) {
    Inst in;
    in.opcode = Opcode::StoreMem;
    in.type = type;
    in.operands[0] = address;
    in.operands[1] = value;
    in.operand_count = 2;
    return emit(in);
}

InstId IRBuilder::add(InstId lhs, InstId rhs, Type type) {
    Inst in;
    in.opcode = Opcode::Add;
    in.type = type;
    in.operands[0] = lhs;
    in.operands[1] = rhs;
    in.operand_count = 2;
    return emit(in);
}

InstId IRBuilder::sub(InstId lhs, InstId rhs, Type type) {
    Inst in;
    in.opcode = Opcode::Sub;
    in.type = type;
    in.operands[0] = lhs;
    in.operands[1] = rhs;
    in.operand_count = 2;
    return emit(in);
}

InstId IRBuilder::cmp(InstId lhs, InstId rhs, Type type) {
    // CMP discards the difference and keeps only the flags. `type` records the
    // operand width, which the backend needs so that a 32-bit compare really
    // compares 32 bits instead of being polluted by the upper half.
    Inst in;
    in.opcode = Opcode::Cmp;
    in.type = type;
    in.operands[0] = lhs;
    in.operands[1] = rhs;
    in.operand_count = 2;
    return emit(in);
}

InstId IRBuilder::addr_add(InstId lhs, InstId rhs) {
    Inst in;
    in.opcode = Opcode::AddrAdd;
    in.type = Type::I64;
    in.operands[0] = lhs;
    in.operands[1] = rhs;
    in.operand_count = 2;
    return emit(in);
}

InstId IRBuilder::addr_shl(InstId value, i64 shift) {
    Inst in;
    in.opcode = Opcode::AddrShl;
    in.type = Type::I64;
    in.operands[0] = value;
    in.operand_count = 1;
    in.imm = shift;
    return emit(in);
}

namespace {

/// Shared shape for the two-operand logical opcodes.
Inst make_binary(Opcode opcode, InstId lhs, InstId rhs, Type type) {
    Inst in;
    in.opcode = opcode;
    in.type = type;
    in.operands[0] = lhs;
    in.operands[1] = rhs;
    in.operand_count = 2;
    return in;
}

}  // namespace

InstId IRBuilder::bit_and(InstId lhs, InstId rhs, Type type) {
    return emit(make_binary(Opcode::And, lhs, rhs, type));
}

InstId IRBuilder::bit_or(InstId lhs, InstId rhs, Type type) {
    return emit(make_binary(Opcode::Or, lhs, rhs, type));
}

InstId IRBuilder::bit_xor(InstId lhs, InstId rhs, Type type) {
    return emit(make_binary(Opcode::Xor, lhs, rhs, type));
}

InstId IRBuilder::test(InstId lhs, InstId rhs, Type type) {
    // Like Cmp, TEST keeps only the flags; `type` records the operand width.
    return emit(make_binary(Opcode::Test, lhs, rhs, type));
}

InstId IRBuilder::bit_not(InstId value, Type type) {
    Inst in;
    in.opcode = Opcode::Not;
    in.type = type;
    in.operands[0] = value;
    in.operand_count = 1;
    return emit(in);
}

InstId IRBuilder::neg(InstId value, Type type) {
    Inst in;
    in.opcode = Opcode::Neg;
    in.type = type;
    in.operands[0] = value;
    in.operand_count = 1;
    return emit(in);
}

InstId IRBuilder::shift(Opcode opcode, InstId value, i64 amount, Type type) {
    Inst in;
    in.opcode = opcode;
    in.type = type;
    in.operands[0] = value;
    in.operand_count = 1;
    in.imm = amount;
    return emit(in);
}

InstId IRBuilder::step(Opcode opcode, InstId value, Type type) {
    Inst in;
    in.opcode = opcode;
    in.type = type;
    in.operands[0] = value;
    in.operand_count = 1;
    return emit(in);
}

InstId IRBuilder::extend(Opcode opcode, InstId value, i64 source_bits, Type type) {
    Inst in;
    in.opcode = opcode;
    in.type = type;
    in.operands[0] = value;
    in.operand_count = 1;
    in.imm = source_bits;
    return emit(in);
}

InstId IRBuilder::jump(BlockId target) {
    Inst in;
    in.opcode = Opcode::Jump;
    in.type = Type::I64;
    in.true_block = target;
    return emit(in);
}

InstId IRBuilder::branch(InstId flags, decoder::Cond cond, BlockId taken,
                         BlockId fallthrough) {
    Inst in;
    in.opcode = Opcode::Branch;
    in.type = Type::I64;
    in.operands[0] = flags;
    in.operand_count = 1;
    in.cond = cond;
    in.true_block = taken;
    in.false_block = fallthrough;
    return emit(in);
}

InstId IRBuilder::ret() {
    Inst in;
    in.opcode = Opcode::Return;
    in.type = Type::I64;
    return emit(in);
}

// --- Diagnostics -----------------------------------------------------------

std::string_view to_string(Opcode value) noexcept {
    switch (value) {
    case Opcode::Const:
        return "const";
    case Opcode::LoadGuestReg:
        return "load_guest_reg";
    case Opcode::StoreGuestReg:
        return "store_guest_reg";
    case Opcode::LoadMem:
        return "load_mem";
    case Opcode::StoreMem:
        return "store_mem";
    case Opcode::Add:
        return "add";
    case Opcode::Sub:
        return "sub";
    case Opcode::Cmp:
        return "cmp";
    case Opcode::AddrAdd:
        return "addr_add";
    case Opcode::AddrShl:
        return "addr_shl";
    case Opcode::And:
        return "and";
    case Opcode::Or:
        return "or";
    case Opcode::Xor:
        return "xor";
    case Opcode::Test:
        return "test";
    case Opcode::Not:
        return "not";
    case Opcode::Neg:
        return "neg";
    case Opcode::Shl:
        return "shl";
    case Opcode::Shr:
        return "shr";
    case Opcode::Sar:
        return "sar";
    case Opcode::Inc:
        return "inc";
    case Opcode::Dec:
        return "dec";
    case Opcode::ZeroExtend:
        return "zext";
    case Opcode::SignExtend:
        return "sext";
    case Opcode::Jump:
        return "jump";
    case Opcode::Branch:
        return "branch";
    case Opcode::Return:
        return "return";
    }
    return "unknown";
}

std::string_view to_string(Type value) noexcept {
    switch (value) {
    case Type::I8:
        return "i8";
    case Type::I16:
        return "i16";
    case Type::I32:
        return "i32";
    case Type::I64:
        return "i64";
    case Type::Flags:
        return "flags";
    }
    return "unknown";
}

}  // namespace dbt::ir
