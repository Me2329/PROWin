#include "dbt/decoder/decoder.hpp"

#include <Zydis/Zydis.h>

#include <algorithm>
#include <array>
#include <ostream>

namespace dbt::decoder {
namespace {

/// Translates a Zydis register into the hardware-encoded X86Reg index.
///
/// `ok` is cleared for anything the translator does not model, which includes
/// the legacy high-byte registers: Zydis places AH/CH/DH/BH inside the GPR8
/// class, so class-relative ids there do not line up with the ModRM encoding
/// and must be remapped rather than used directly.
X86Reg map_register(ZydisRegister reg, bool& ok) noexcept {
    ok = true;
    if (reg == ZYDIS_REGISTER_NONE) {
        return X86Reg::None;
    }
    if (reg == ZYDIS_REGISTER_RIP || reg == ZYDIS_REGISTER_EIP) {
        return X86Reg::Rip;
    }

    switch (ZydisRegisterGetClass(reg)) {
    case ZYDIS_REGCLASS_GPR64:
    case ZYDIS_REGCLASS_GPR32:
    case ZYDIS_REGCLASS_GPR16: {
        // These classes are ordered exactly as the hardware encodes them.
        const ZyanI8 id = ZydisRegisterGetId(reg);
        if (id >= 0 && static_cast<usize>(id) < kNumGpr) {
            return static_cast<X86Reg>(static_cast<u8>(id));
        }
        break;
    }
    case ZYDIS_REGCLASS_GPR8: {
        if (reg >= ZYDIS_REGISTER_AH && reg <= ZYDIS_REGISTER_BH) {
            break;  // No direct analogue in the register mapping.
        }
        const ZyanI8 id = ZydisRegisterGetId(reg);
        if (id >= 0) {
            // AL..BL occupy 0..3; everything from SPL onward is shifted by the
            // four high-byte entries.
            const int encoded = (id < 4) ? int{id} : int{id} - 4;
            if (static_cast<usize>(encoded) < kNumGpr) {
                return static_cast<X86Reg>(static_cast<u8>(encoded));
            }
        }
        break;
    }
    default:
        break;
    }

    ok = false;
    return X86Reg::None;
}

/// Maps a Zydis Jcc mnemonic to its condition, or Cond::None if it is not a
/// conditional branch.
Cond map_condition(ZydisMnemonic mnemonic) noexcept {
    switch (mnemonic) {
    case ZYDIS_MNEMONIC_JO:
        return Cond::Overflow;
    case ZYDIS_MNEMONIC_JNO:
        return Cond::NotOverflow;
    case ZYDIS_MNEMONIC_JB:
        return Cond::Below;
    case ZYDIS_MNEMONIC_JNB:
        return Cond::AboveEqual;
    case ZYDIS_MNEMONIC_JZ:
        return Cond::Equal;
    case ZYDIS_MNEMONIC_JNZ:
        return Cond::NotEqual;
    case ZYDIS_MNEMONIC_JBE:
        return Cond::BelowEqual;
    case ZYDIS_MNEMONIC_JNBE:
        return Cond::Above;
    case ZYDIS_MNEMONIC_JS:
        return Cond::Sign;
    case ZYDIS_MNEMONIC_JNS:
        return Cond::NotSign;
    case ZYDIS_MNEMONIC_JP:
        return Cond::Parity;
    case ZYDIS_MNEMONIC_JNP:
        return Cond::NotParity;
    case ZYDIS_MNEMONIC_JL:
        return Cond::Less;
    case ZYDIS_MNEMONIC_JNL:
        return Cond::GreaterEqual;
    case ZYDIS_MNEMONIC_JLE:
        return Cond::LessEqual;
    case ZYDIS_MNEMONIC_JNLE:
        return Cond::Greater;
    default:
        return Cond::None;
    }
}

DecodeError classify_failure(ZyanStatus status) noexcept {
    if (status == ZYDIS_STATUS_NO_MORE_DATA) {
        return DecodeError::TruncatedInstruction;
    }
    return DecodeError::InvalidInstruction;
}

}  // namespace

struct Decoder::Impl {
    ZydisDecoder zydis{};
};

Decoder::Decoder() : impl_(std::make_unique<Impl>()) {
    ZydisDecoderInit(&impl_->zydis, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
}

Decoder::~Decoder() = default;
Decoder::Decoder(Decoder&&) noexcept = default;
Decoder& Decoder::operator=(Decoder&&) noexcept = default;

DecodeResult Decoder::decode(std::span<const u8> code, GuestAddr address) const {
    DecodeResult result;
    result.inst.address = address;

    if (code.empty()) {
        result.error = DecodeError::EmptyBuffer;
        return result;
    }

    // No legal instruction is longer than kMaxX86InstLength, so the decoder is
    // never handed more than that many bytes. Combined with the span this
    // bounds every read to the caller's buffer.
    const usize limit = std::min(code.size(), kMaxX86InstLength);

    ZydisDecodedInstruction zinst{};
    std::array<ZydisDecodedOperand, ZYDIS_MAX_OPERAND_COUNT> zops{};

    const ZyanStatus status = ZydisDecoderDecodeFull(&impl_->zydis, code.data(), limit,
                                                     &zinst, zops.data());
    if (!ZYAN_SUCCESS(status)) {
        result.error = classify_failure(status);
        return result;
    }

    switch (zinst.mnemonic) {
    case ZYDIS_MNEMONIC_MOV:
        result.inst.mnemonic = Mnemonic::Mov;
        break;
    case ZYDIS_MNEMONIC_ADD:
        result.inst.mnemonic = Mnemonic::Add;
        break;
    case ZYDIS_MNEMONIC_SUB:
        result.inst.mnemonic = Mnemonic::Sub;
        break;
    case ZYDIS_MNEMONIC_CMP:
        result.inst.mnemonic = Mnemonic::Cmp;
        break;
    case ZYDIS_MNEMONIC_JMP:
        result.inst.mnemonic = Mnemonic::Jmp;
        break;
    case ZYDIS_MNEMONIC_RET:
        result.inst.mnemonic = Mnemonic::Ret;
        break;
    case ZYDIS_MNEMONIC_PUSH:
        result.inst.mnemonic = Mnemonic::Push;
        break;
    case ZYDIS_MNEMONIC_POP:
        result.inst.mnemonic = Mnemonic::Pop;
        break;
    case ZYDIS_MNEMONIC_CALL:
        result.inst.mnemonic = Mnemonic::Call;
        break;
    case ZYDIS_MNEMONIC_LEA:
        result.inst.mnemonic = Mnemonic::Lea;
        break;
    case ZYDIS_MNEMONIC_AND:
        result.inst.mnemonic = Mnemonic::And;
        break;
    case ZYDIS_MNEMONIC_OR:
        result.inst.mnemonic = Mnemonic::Or;
        break;
    case ZYDIS_MNEMONIC_XOR:
        result.inst.mnemonic = Mnemonic::Xor;
        break;
    case ZYDIS_MNEMONIC_TEST:
        result.inst.mnemonic = Mnemonic::Test;
        break;
    case ZYDIS_MNEMONIC_NOT:
        result.inst.mnemonic = Mnemonic::Not;
        break;
    case ZYDIS_MNEMONIC_NEG:
        result.inst.mnemonic = Mnemonic::Neg;
        break;
    case ZYDIS_MNEMONIC_SHL:
        result.inst.mnemonic = Mnemonic::Shl;
        break;
    case ZYDIS_MNEMONIC_SHR:
        result.inst.mnemonic = Mnemonic::Shr;
        break;
    case ZYDIS_MNEMONIC_SAR:
        result.inst.mnemonic = Mnemonic::Sar;
        break;
    case ZYDIS_MNEMONIC_INC:
        result.inst.mnemonic = Mnemonic::Inc;
        break;
    case ZYDIS_MNEMONIC_DEC:
        result.inst.mnemonic = Mnemonic::Dec;
        break;
    default: {
        const Cond cond = map_condition(zinst.mnemonic);
        if (cond == Cond::None) {
            result.error = DecodeError::UnsupportedInstruction;
            return result;
        }
        result.inst.mnemonic = Mnemonic::Jcc;
        result.inst.cond = cond;
        break;
    }
    }

    result.inst.length = zinst.length;

    const usize visible = std::min(static_cast<usize>(zinst.operand_count_visible),
                                   kMaxOperands);
    for (usize i = 0; i < visible; ++i) {
        const ZydisDecodedOperand& zop = zops[i];
        Operand op;
        op.size_bits = zop.size;

        switch (zop.type) {
        case ZYDIS_OPERAND_TYPE_REGISTER: {
            bool ok = false;
            op.kind = OperandKind::Register;
            op.reg = map_register(zop.reg.value, ok);
            if (!ok) {
                result.error = DecodeError::UnsupportedInstruction;
                return result;
            }
            break;
        }
        case ZYDIS_OPERAND_TYPE_IMMEDIATE:
            op.kind = OperandKind::Immediate;
            op.imm = zop.imm.is_signed ? zop.imm.value.s
                                       : static_cast<i64>(zop.imm.value.u);
            break;
        case ZYDIS_OPERAND_TYPE_MEMORY: {
            op.kind = OperandKind::Memory;
            op.mem.scale = (zop.mem.scale == 0) ? u8{1} : zop.mem.scale;
            op.mem.disp = zop.mem.disp.has_displacement ? zop.mem.disp.value : 0;

            bool base_ok = false;
            op.mem.base = map_register(zop.mem.base, base_ok);
            bool index_ok = false;
            op.mem.index = map_register(zop.mem.index, index_ok);
            if (!base_ok || !index_ok) {
                result.error = DecodeError::UnsupportedInstruction;
                return result;
            }
            op.mem.rip_relative = (op.mem.base == X86Reg::Rip);

            if (op.mem.rip_relative) {
                ZyanU64 absolute = 0;
                if (ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&zinst, &zop, address,
                                                          &absolute))) {
                    result.inst.has_rip_target = true;
                    result.inst.rip_target = absolute;
                }
            }
            break;
        }
        default:
            result.error = DecodeError::UnsupportedInstruction;
            return result;
        }

        result.inst.operands[i] = op;
    }
    result.inst.operand_count = static_cast<u8>(visible);

    // Relative branches carry their displacement as an immediate; resolve it to
    // an absolute guest address once, here, so no caller repeats the arithmetic.
    if (result.inst.mnemonic == Mnemonic::Jmp || result.inst.mnemonic == Mnemonic::Jcc ||
        result.inst.mnemonic == Mnemonic::Call) {
        for (usize i = 0; i < visible; ++i) {
            if (zops[i].type == ZYDIS_OPERAND_TYPE_IMMEDIATE && zops[i].imm.is_relative) {
                ZyanU64 absolute = 0;
                if (ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&zinst, &zops[i], address,
                                                          &absolute))) {
                    result.inst.has_branch_target = true;
                    result.inst.branch_target = absolute;
                }
                break;
            }
        }
    }

    return result;
}

std::string_view to_string(Mnemonic value) noexcept {
    switch (value) {
    case Mnemonic::Mov:
        return "mov";
    case Mnemonic::Add:
        return "add";
    case Mnemonic::Sub:
        return "sub";
    case Mnemonic::Cmp:
        return "cmp";
    case Mnemonic::Jmp:
        return "jmp";
    case Mnemonic::Jcc:
        return "jcc";
    case Mnemonic::Ret:
        return "ret";
    case Mnemonic::Push:
        return "push";
    case Mnemonic::Pop:
        return "pop";
    case Mnemonic::Call:
        return "call";
    case Mnemonic::Lea:
        return "lea";
    case Mnemonic::And:
        return "and";
    case Mnemonic::Or:
        return "or";
    case Mnemonic::Xor:
        return "xor";
    case Mnemonic::Test:
        return "test";
    case Mnemonic::Not:
        return "not";
    case Mnemonic::Neg:
        return "neg";
    case Mnemonic::Shl:
        return "shl";
    case Mnemonic::Shr:
        return "shr";
    case Mnemonic::Sar:
        return "sar";
    case Mnemonic::Inc:
        return "inc";
    case Mnemonic::Dec:
        return "dec";
    case Mnemonic::Invalid:
        break;
    }
    return "invalid";
}

std::string_view to_string(X86Reg value) noexcept {
    switch (value) {
    case X86Reg::Rax:
        return "rax";
    case X86Reg::Rcx:
        return "rcx";
    case X86Reg::Rdx:
        return "rdx";
    case X86Reg::Rbx:
        return "rbx";
    case X86Reg::Rsp:
        return "rsp";
    case X86Reg::Rbp:
        return "rbp";
    case X86Reg::Rsi:
        return "rsi";
    case X86Reg::Rdi:
        return "rdi";
    case X86Reg::R8:
        return "r8";
    case X86Reg::R9:
        return "r9";
    case X86Reg::R10:
        return "r10";
    case X86Reg::R11:
        return "r11";
    case X86Reg::R12:
        return "r12";
    case X86Reg::R13:
        return "r13";
    case X86Reg::R14:
        return "r14";
    case X86Reg::R15:
        return "r15";
    case X86Reg::Rip:
        return "rip";
    case X86Reg::None:
        break;
    }
    return "none";
}

std::string_view to_string(Cond value) noexcept {
    switch (value) {
    case Cond::Overflow:
        return "o";
    case Cond::NotOverflow:
        return "no";
    case Cond::Below:
        return "b";
    case Cond::AboveEqual:
        return "ae";
    case Cond::Equal:
        return "e";
    case Cond::NotEqual:
        return "ne";
    case Cond::BelowEqual:
        return "be";
    case Cond::Above:
        return "a";
    case Cond::Sign:
        return "s";
    case Cond::NotSign:
        return "ns";
    case Cond::Parity:
        return "p";
    case Cond::NotParity:
        return "np";
    case Cond::Less:
        return "l";
    case Cond::GreaterEqual:
        return "ge";
    case Cond::LessEqual:
        return "le";
    case Cond::Greater:
        return "g";
    case Cond::None:
        break;
    }
    return "none";
}

std::string_view to_string(OperandKind value) noexcept {
    switch (value) {
    case OperandKind::Register:
        return "register";
    case OperandKind::Immediate:
        return "immediate";
    case OperandKind::Memory:
        return "memory";
    case OperandKind::None:
        break;
    }
    return "none";
}

std::string_view to_string(DecodeError value) noexcept {
    switch (value) {
    case DecodeError::None:
        return "none";
    case DecodeError::EmptyBuffer:
        return "empty-buffer";
    case DecodeError::TruncatedInstruction:
        return "truncated-instruction";
    case DecodeError::InvalidInstruction:
        return "invalid-instruction";
    case DecodeError::UnsupportedInstruction:
        return "unsupported-instruction";
    }
    return "unknown";
}

std::ostream& operator<<(std::ostream& os, Mnemonic value) { return os << to_string(value); }
std::ostream& operator<<(std::ostream& os, X86Reg value) { return os << to_string(value); }
std::ostream& operator<<(std::ostream& os, Cond value) { return os << to_string(value); }
std::ostream& operator<<(std::ostream& os, OperandKind value) {
    return os << to_string(value);
}
std::ostream& operator<<(std::ostream& os, DecodeError value) {
    return os << to_string(value);
}

}  // namespace dbt::decoder
