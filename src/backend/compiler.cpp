#include "dbt/backend/compiler.hpp"

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "dbt/backend/register_map.hpp"
#include "dbt/runtime/cpu_state.hpp"

namespace dbt::backend {
namespace {

using a64::Reg;

/// Temporaries available for IR values that are not already pinned to a guest
/// register. x16/x17 are the architecture's scratch registers; x19/x20 are
/// callee-saved and are preserved by the prologue.
constexpr std::array<Reg, 4> kTempPool{Reg::X16, Reg::X17, Reg::X19, Reg::X20};

/// Stack frame: x29/x30 at 0, x19/x20 at 16, x28 at 32. Sixteen-byte aligned.
constexpr i32 kFrameSize = 48;
constexpr i32 kSavedPairOffset = 16;
constexpr u32 kSavedBaseOffset = 32;

/// A branch whose displacement is only known once every block has been laid out.
struct Fixup {
    usize word_index = 0;
    ir::BlockId target = ir::kNoBlock;
    bool conditional = false;
    a64::Cond cond = a64::Cond::AL;
};

class FunctionCompiler {
public:
    FunctionCompiler(const ir::Function& function, usize max_words)
        : func_(function), max_words_(max_words) {}

    CompileResult run();

private:
    void emit(Arm64Word word) {
        if (words_.size() < max_words_) {
            words_.push_back(word);
        } else {
            fail(CompileError::CodeTooLarge, ir::kNoInst);
        }
    }

    void fail(CompileError error, ir::InstId inst) {
        if (error_ == CompileError::None) {
            error_ = error;
            error_inst_ = inst;
        }
    }

    [[nodiscard]] bool failed() const noexcept { return error_ != CompileError::None; }

    [[nodiscard]] Reg acquire_temp(ir::InstId owner);
    [[nodiscard]] Reg reg_of(ir::InstId id) const { return value_reg_[id]; }

    void emit_prologue();
    void emit_epilogue();
    void emit_const(Reg dst, i64 value, a64::Width width);
    void lower(ir::InstId id, usize position);
    void compute_layout_and_liveness();
    void free_dead_values(usize position);
    void protect_pinned_register(Reg pinned, ir::InstId keep, usize position);

    const ir::Function& func_;
    usize max_words_ = 0;

    std::vector<Arm64Word> words_;
    std::vector<Fixup> fixups_;
    std::vector<usize> block_start_;

    /// Emission order: flattened instruction ids, and each id's position in it.
    std::vector<ir::InstId> order_;
    std::vector<usize> position_of_;
    std::vector<usize> last_use_;
    std::vector<Reg> value_reg_;
    /// Which pool entries are taken, and by which value.
    std::array<ir::InstId, kTempPool.size()> temp_owner_{};

    CompileError error_ = CompileError::None;
    ir::InstId error_inst_ = ir::kNoInst;
};

void FunctionCompiler::compute_layout_and_liveness() {
    const usize count = func_.inst_count();
    position_of_.assign(count, 0);
    last_use_.assign(count, 0);
    value_reg_.assign(count, Reg::ZrSp);

    // Blocks are emitted in id order, which puts the entry block first.
    for (const ir::BasicBlock& blk : func_.blocks()) {
        for (const ir::InstId id : blk.insts) {
            position_of_[id] = order_.size();
            order_.push_back(id);
        }
    }

    for (usize position = 0; position < order_.size(); ++position) {
        const ir::Inst& in = func_.inst(order_[position]);
        for (u8 k = 0; k < in.operand_count; ++k) {
            const ir::InstId use = in.operands[k];
            if (use < count && position > last_use_[use]) {
                last_use_[use] = position;
            }
        }
    }
}

Reg FunctionCompiler::acquire_temp(ir::InstId owner) {
    for (usize i = 0; i < kTempPool.size(); ++i) {
        if (temp_owner_[i] == ir::kNoInst) {
            temp_owner_[i] = owner;
            return kTempPool[i];
        }
    }
    fail(CompileError::OutOfRegisters, owner);
    return Reg::X16;
}

void FunctionCompiler::free_dead_values(usize position) {
    for (usize i = 0; i < kTempPool.size(); ++i) {
        const ir::InstId owner = temp_owner_[i];
        if (owner != ir::kNoInst && last_use_[owner] <= position) {
            temp_owner_[i] = ir::kNoInst;
        }
    }
}

void FunctionCompiler::protect_pinned_register(Reg pinned, ir::InstId keep,
                                               usize position) {
    // A LoadGuestReg value is not copied -- it just names the pinned register.
    // If some still-live value names the register we are about to overwrite,
    // move it to a temporary first so the store cannot clobber it.
    for (const ir::InstId id : order_) {
        if (id == keep || value_reg_[id] != pinned) {
            continue;
        }
        if (!func_.inst(id).defines_value() || position_of_[id] > position) {
            continue;
        }
        if (last_use_[id] <= position) {
            continue;  // already dead
        }
        const Reg temp = acquire_temp(id);
        if (failed()) {
            return;
        }
        emit(a64::mov_reg(temp, pinned));
        value_reg_[id] = temp;
    }
}

void FunctionCompiler::emit_const(Reg dst, i64 value, a64::Width width) {
    const bool is32 = (width == a64::Width::W32);
    auto bits = static_cast<u64>(value);
    if (is32) {
        // A 32-bit MOVZ writes a W register, which zero-extends; only the low
        // half of the constant is meaningful.
        bits &= 0xFFFFFFFFu;
    }

    if (bits <= 0xFFFFu) {
        emit(a64::with_width(a64::movz(dst, static_cast<u16>(bits)), width));
        return;
    }
    // A small negative fits MOVN, which writes the inverted immediate.
    const u64 inverted = is32 ? (~bits & 0xFFFFFFFFu) : ~bits;
    if (inverted <= 0xFFFFu) {
        emit(a64::with_width(a64::movn(dst, static_cast<u16>(inverted)), width));
        return;
    }

    bool first = true;
    const u8 bit_count = is32 ? u8{32} : u8{64};
    for (u8 shift = 0; shift < bit_count; shift = static_cast<u8>(shift + 16)) {
        const auto halfword = static_cast<u16>((bits >> shift) & 0xFFFFu);
        if (halfword == 0) {
            continue;
        }
        emit(a64::with_width(first ? a64::movz(dst, halfword, shift)
                                   : a64::movk(dst, halfword, shift),
                             width));
        first = false;
    }
    if (first) {
        emit(a64::with_width(a64::movz(dst, 0), width));
    }
}

void FunctionCompiler::emit_prologue() {
    emit(a64::stp_pre(a64::kFramePointer, a64::kLinkReg, a64::kStackPointer,
                      -kFrameSize));
    emit(a64::stp_off(Reg::X19, Reg::X20, a64::kStackPointer, kSavedPairOffset));
    emit(a64::str_imm(kCpuStateBase, a64::kStackPointer, kSavedBaseOffset));
    emit(a64::mov_reg(a64::kFramePointer, a64::kStackPointer));
    // The CpuState pointer arrives in x0, which is about to become guest RAX,
    // so it must be parked before any guest register is loaded.
    emit(a64::mov_reg(kCpuStateBase, Reg::X0));

    for (u8 i = 0; i < decoder::kNumGpr; ++i) {
        const auto guest = static_cast<decoder::X86Reg>(i);
        emit(a64::ldr_imm(map_gpr(guest), kCpuStateBase, runtime::gpr_offset(guest)));
    }
}

void FunctionCompiler::emit_epilogue() {
    for (u8 i = 0; i < decoder::kNumGpr; ++i) {
        const auto guest = static_cast<decoder::X86Reg>(i);
        emit(a64::str_imm(map_gpr(guest), kCpuStateBase, runtime::gpr_offset(guest)));
    }
    emit(a64::ldr_imm(kCpuStateBase, a64::kStackPointer, kSavedBaseOffset));
    emit(a64::ldp_off(Reg::X19, Reg::X20, a64::kStackPointer, kSavedPairOffset));
    emit(a64::ldp_post(a64::kFramePointer, a64::kLinkReg, a64::kStackPointer,
                       kFrameSize));
    emit(a64::ret());
}

void FunctionCompiler::lower(ir::InstId id, usize position) {
    const ir::Inst& in = func_.inst(id);

    // 64- and 32-bit operations lower directly. A 32-bit write zero-extends
    // into the full register on both architectures, so the W-register forms
    // carry x86's semantics exactly. Narrower widths still have merge
    // semantics that are not modelled, so they stay refused.
    // Narrow widths are meaningful only on memory accesses, where LDRB/LDRH
    // zero-extend cleanly. A narrow *register* write merges into the existing
    // value, which is not modelled, so that stays refused.
    const bool narrow = (in.type == ir::Type::I8) || (in.type == ir::Type::I16);
    const bool narrow_ok = narrow && (in.opcode == ir::Opcode::LoadMem ||
                                      in.opcode == ir::Opcode::StoreMem);
    const bool width_ok = (in.type == ir::Type::I64) || (in.type == ir::Type::I32) ||
                          (in.type == ir::Type::Flags) || narrow_ok;
    if (!width_ok) {
        fail(CompileError::UnsupportedWidth, id);
        return;
    }
    const a64::Width width =
        (in.type == ir::Type::I32) ? a64::Width::W32 : a64::Width::X64;

    switch (in.opcode) {
    case ir::Opcode::LoadGuestReg: {
        // RIP has no pinned register: it lives only in CpuState.
        if (in.guest_reg == decoder::X86Reg::Rip) {
            const Reg dst = acquire_temp(id);
            if (failed()) {
                return;
            }
            value_reg_[id] = dst;
            emit(a64::ldr_imm(dst, kCpuStateBase,
                              static_cast<u32>(runtime::kRipOffset)));
            break;
        }
        if (!is_mappable(in.guest_reg)) {
            fail(CompileError::UnsupportedOpcode, id);
            return;
        }
        // Free: the value already lives in its pinned register.
        value_reg_[id] = map_gpr(in.guest_reg);
        break;
    }

    case ir::Opcode::StoreGuestReg: {
        const ir::InstId store_source = in.operands[0];
        // Every block exit writes the next guest RIP straight into CpuState so
        // the dispatcher learns where execution should resume.
        if (in.guest_reg == decoder::X86Reg::Rip) {
            emit(a64::str_imm(reg_of(store_source), kCpuStateBase,
                              static_cast<u32>(runtime::kRipOffset)));
            break;
        }
        if (!is_mappable(in.guest_reg)) {
            fail(CompileError::UnsupportedOpcode, id);
            return;
        }
        const Reg dst = map_gpr(in.guest_reg);
        const ir::InstId source = store_source;
        protect_pinned_register(dst, source, position);
        if (failed()) {
            return;
        }
        const Reg src = reg_of(source);
        // A 32-bit move is never redundant, even when src == dst: `mov eax, eax`
        // exists precisely to clear the upper half of RAX.
        if (src != dst || width == a64::Width::W32) {
            emit(a64::with_width(a64::mov_reg(dst, src), width));
        }
        break;
    }

    case ir::Opcode::Const: {
        const Reg dst = acquire_temp(id);
        if (failed()) {
            return;
        }
        value_reg_[id] = dst;
        emit_const(dst, in.imm, width);
        break;
    }

    case ir::Opcode::LoadMem: {
        const Reg addr = reg_of(in.operands[0]);
        const Reg dst = acquire_temp(id);
        if (failed()) {
            return;
        }
        value_reg_[id] = dst;
        switch (in.type) {
        case ir::Type::I8:
            emit(a64::ldrb(dst, addr, 0));
            break;
        case ir::Type::I16:
            emit(a64::ldrh(dst, addr, 0));
            break;
        case ir::Type::I32:
            emit(a64::ldr_imm32(dst, addr, 0));
            break;
        default:
            emit(a64::ldr_imm(dst, addr, 0));
            break;
        }
        break;
    }

    case ir::Opcode::StoreMem: {
        const Reg addr = reg_of(in.operands[0]);
        const Reg value = reg_of(in.operands[1]);
        switch (in.type) {
        case ir::Type::I8:
            emit(a64::strb(value, addr, 0));
            break;
        case ir::Type::I16:
            emit(a64::strh(value, addr, 0));
            break;
        case ir::Type::I32:
            emit(a64::str_imm32(value, addr, 0));
            break;
        default:
            emit(a64::str_imm(value, addr, 0));
            break;
        }
        break;
    }

    case ir::Opcode::Add:
    case ir::Opcode::Sub: {
        const Reg lhs = reg_of(in.operands[0]);
        const Reg rhs = reg_of(in.operands[1]);
        const Reg dst = acquire_temp(id);
        if (failed()) {
            return;
        }
        value_reg_[id] = dst;
        // The flag-setting forms: x86 ADD/SUB always update EFLAGS.
        emit(a64::with_width(in.opcode == ir::Opcode::Add
                                 ? a64::add_reg(dst, lhs, rhs, true)
                                 : a64::sub_reg(dst, lhs, rhs, true),
                             width));
        break;
    }

    case ir::Opcode::Cmp: {
        // CMP keeps only the flags, so it needs no destination register.
        emit(a64::with_width(a64::cmp_reg(reg_of(in.operands[0]),
                                          reg_of(in.operands[1])),
                             width));
        break;
    }

    case ir::Opcode::Test: {
        // Same shape as CMP, but AND-based.
        emit(a64::with_width(a64::tst_reg(reg_of(in.operands[0]),
                                          reg_of(in.operands[1])),
                             width));
        break;
    }

    case ir::Opcode::And:
    case ir::Opcode::Or:
    case ir::Opcode::Xor: {
        const Reg lhs = reg_of(in.operands[0]);
        const Reg rhs = reg_of(in.operands[1]);
        const Reg dst = acquire_temp(id);
        if (failed()) {
            return;
        }
        value_reg_[id] = dst;
        if (in.opcode == ir::Opcode::And) {
            emit(a64::with_width(a64::and_reg(dst, lhs, rhs, true), width));
        } else {
            // AArch64 has ANDS but no ORRS or EORS, so the flags for OR and XOR
            // have to be published by a following TST.
            emit(a64::with_width(in.opcode == ir::Opcode::Or
                                     ? a64::orr_reg(dst, lhs, rhs)
                                     : a64::eor_reg(dst, lhs, rhs),
                                 width));
            emit(a64::with_width(a64::tst_reg(dst, dst), width));
        }
        break;
    }

    case ir::Opcode::Not: {
        const Reg src = reg_of(in.operands[0]);
        const Reg dst = acquire_temp(id);
        if (failed()) {
            return;
        }
        value_reg_[id] = dst;
        // MVN sets no flags, matching x86 NOT.
        emit(a64::with_width(a64::mvn_reg(dst, src), width));
        break;
    }

    case ir::Opcode::Neg: {
        const Reg src = reg_of(in.operands[0]);
        const Reg dst = acquire_temp(id);
        if (failed()) {
            return;
        }
        value_reg_[id] = dst;
        emit(a64::with_width(a64::neg_reg(dst, src, true), width));
        break;
    }

    case ir::Opcode::ZeroExtend:
    case ir::Opcode::SignExtend: {
        const Reg src = reg_of(in.operands[0]);
        const Reg dst = acquire_temp(id);
        if (failed()) {
            return;
        }
        value_reg_[id] = dst;

        const bool sign = (in.opcode == ir::Opcode::SignExtend);
        switch (in.imm) {
        case 8:
            emit(sign ? a64::sxtb(dst, src, width) : a64::uxtb(dst, src));
            break;
        case 16:
            emit(sign ? a64::sxth(dst, src, width) : a64::uxth(dst, src));
            break;
        case 32:
            // Writing a W register already zero-extends, so only the signed
            // form needs a dedicated instruction here.
            emit(sign ? a64::sxtw(dst, src)
                      : a64::with_width(a64::mov_reg(dst, src), a64::Width::W32));
            break;
        default:
            fail(CompileError::UnsupportedWidth, id);
            return;
        }
        break;
    }

    case ir::Opcode::Shl:
    case ir::Opcode::Shr:
    case ir::Opcode::Sar: {
        const Reg src = reg_of(in.operands[0]);
        const Reg dst = acquire_temp(id);
        if (failed()) {
            return;
        }
        value_reg_[id] = dst;
        const auto amount = static_cast<u8>(in.imm & 0x3F);
        emit(in.opcode == ir::Opcode::Shl   ? a64::lsl_imm(dst, src, amount, width)
             : in.opcode == ir::Opcode::Shr ? a64::lsr_imm(dst, src, amount, width)
                                            : a64::asr_imm(dst, src, amount, width));
        // AArch64 shifts set no flags, so ZF and SF are published with a TST.
        // CF and OF stay unavailable, which ir::defined_flags records.
        emit(a64::with_width(a64::tst_reg(dst, dst), width));
        break;
    }

    case ir::Opcode::Inc:
    case ir::Opcode::Dec: {
        const Reg src = reg_of(in.operands[0]);
        const Reg dst = acquire_temp(id);
        if (failed()) {
            return;
        }
        value_reg_[id] = dst;
        // ADDS/SUBS also write C, which x86 INC/DEC deliberately preserve; the
        // flag model marks CF undefined for these so no branch can read it.
        emit(a64::with_width(in.opcode == ir::Opcode::Inc
                                 ? a64::add_imm(dst, src, 1, true)
                                 : a64::sub_imm(dst, src, 1, true),
                             width));
        break;
    }

    case ir::Opcode::AddrAdd: {
        const Reg lhs = reg_of(in.operands[0]);
        const Reg rhs = reg_of(in.operands[1]);
        const Reg dst = acquire_temp(id);
        if (failed()) {
            return;
        }
        value_reg_[id] = dst;
        // Deliberately the non-flag-setting form: addressing must not disturb
        // the flags a later conditional branch depends on.
        emit(a64::add_reg(dst, lhs, rhs, false));
        break;
    }

    case ir::Opcode::AddrShl: {
        const Reg src = reg_of(in.operands[0]);
        const Reg dst = acquire_temp(id);
        if (failed()) {
            return;
        }
        value_reg_[id] = dst;
        emit(a64::lsl_imm(dst, src, static_cast<u8>(in.imm & 0x3F)));
        break;
    }

    case ir::Opcode::Jump: {
        fixups_.push_back(Fixup{words_.size(), in.true_block, false, a64::Cond::AL});
        emit(a64::b(0));
        break;
    }

    case ir::Opcode::Branch: {
        // Which ARM64 condition to use depends on whether the flags came from a
        // subtraction, because x86 and AArch64 disagree on the sense of carry
        // after a subtract but agree after an add.
        const ir::Opcode producer = func_.inst(in.operands[0]).opcode;
        const bool from_subtraction = ir::flags_from_subtraction(producer);

        // Refuse a condition that reads a flag this producer could not model --
        // a carry test after a shift, say -- rather than branching on whatever
        // happens to be left in NZCV.
        if ((ir::required_flags(in.cond) & ~ir::defined_flags(producer)) != 0) {
            fail(CompileError::UnsupportedCondition, id);
            return;
        }

        a64::Cond cond = a64::Cond::AL;
        if (!to_arm64_condition(in.cond, from_subtraction, cond)) {
            fail(CompileError::UnsupportedCondition, id);
            return;
        }

        fixups_.push_back(Fixup{words_.size(), in.true_block, true, cond});
        emit(a64::b_cond(cond, 0));
        fixups_.push_back(Fixup{words_.size(), in.false_block, false, a64::Cond::AL});
        emit(a64::b(0));
        break;
    }

    case ir::Opcode::Return:
        emit_epilogue();
        break;
    }
}

CompileResult FunctionCompiler::run() {
    CompileResult result;

    std::string verify_error;
    if (!func_.verify(&verify_error)) {
        result.error = CompileError::InvalidIr;
        return result;
    }

    compute_layout_and_liveness();
    temp_owner_.fill(ir::kNoInst);

    emit_prologue();

    block_start_.assign(func_.block_count(), 0);
    usize position = 0;
    for (const ir::BasicBlock& blk : func_.blocks()) {
        block_start_[blk.id] = words_.size();
        for (const ir::InstId id : blk.insts) {
            lower(id, position);
            if (failed()) {
                result.error = error_;
                result.error_inst = error_inst_;
                return result;
            }
            free_dead_values(position);
            ++position;
        }
    }

    // Patch every branch now that block addresses are known.
    for (const Fixup& fixup : fixups_) {
        const auto target = static_cast<i64>(block_start_[fixup.target]);
        const auto from = static_cast<i64>(fixup.word_index);
        const i64 byte_offset = (target - from) * static_cast<i64>(kArm64InstSize);

        const bool in_range = fixup.conditional ? a64::fits_branch19(byte_offset)
                                                : a64::fits_branch26(byte_offset);
        if (!in_range) {
            result.error = CompileError::BranchOutOfRange;
            return result;
        }

        words_[fixup.word_index] =
            fixup.conditional
                ? a64::b_cond(fixup.cond, static_cast<i32>(byte_offset))
                : a64::b(static_cast<i32>(byte_offset));
    }

    result.words = std::move(words_);
    return result;
}

}  // namespace

CompileResult Compiler::compile(const ir::Function& function) const {
    FunctionCompiler compiler(function, options_.max_words);
    return compiler.run();
}

std::string_view to_string(CompileError value) noexcept {
    switch (value) {
    case CompileError::None:
        return "none";
    case CompileError::InvalidIr:
        return "invalid-ir";
    case CompileError::UnsupportedOpcode:
        return "unsupported-opcode";
    case CompileError::UnsupportedCondition:
        return "unsupported-condition";
    case CompileError::UnsupportedWidth:
        return "unsupported-width";
    case CompileError::BranchOutOfRange:
        return "branch-out-of-range";
    case CompileError::OutOfRegisters:
        return "out-of-registers";
    case CompileError::CodeTooLarge:
        return "code-too-large";
    }
    return "unknown";
}

}  // namespace dbt::backend
