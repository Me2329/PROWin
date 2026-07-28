#pragma once

#include <array>
#include <cassert>
#include <iosfwd>
#include <memory>
#include <span>
#include <string_view>

#include "dbt/common/types.hpp"

namespace dbt::decoder {

/// x86-64 general-purpose registers, numbered by their hardware encoding so the
/// enumerator value doubles as the ModRM/REX register index.
enum class X86Reg : u8 {
    Rax = 0,
    Rcx,
    Rdx,
    Rbx,
    Rsp,
    Rbp,
    Rsi,
    Rdi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    Rip,
    None,
};

/// Number of general-purpose registers the translator models.
inline constexpr usize kNumGpr = 16;

[[nodiscard]] constexpr bool is_gpr(X86Reg reg) noexcept {
    return static_cast<u8>(reg) < kNumGpr;
}

/// The instruction subset this translator understands. Anything outside this
/// set is reported as DecodeError::UnsupportedInstruction rather than being
/// silently mistranslated.
enum class Mnemonic : u8 {
    Invalid = 0,
    Mov,
    Add,
    Sub,
    Cmp,
    Jmp,
    Jcc,
    Ret,
    Push,
    Pop,
    Call,
    Lea,
    And,
    Or,
    Xor,
    Test,
    Not,
    Neg,
    Shl,
    Shr,
    Sar,
    Inc,
    Dec,
    Movzx,
    Movsx,
};

/// x86 condition codes, ordered by their `tttn` encoding.
enum class Cond : u8 {
    Overflow = 0,
    NotOverflow,
    Below,
    AboveEqual,
    Equal,
    NotEqual,
    BelowEqual,
    Above,
    Sign,
    NotSign,
    Parity,
    NotParity,
    Less,
    GreaterEqual,
    LessEqual,
    Greater,
    None,
};

enum class OperandKind : u8 {
    None = 0,
    Register,
    Immediate,
    Memory,
};

/// A resolved `[base + index*scale + disp]` reference.
///
/// For RIP-relative operands `base` is X86Reg::Rip, `rip_relative` is true, and
/// `disp` still holds the raw displacement; the absolute address is exposed
/// separately on DecodedInst so callers never have to redo the arithmetic.
struct MemRef {
    X86Reg base = X86Reg::None;
    X86Reg index = X86Reg::None;
    u8 scale = 1;
    i64 disp = 0;
    bool rip_relative = false;
};

struct Operand {
    OperandKind kind = OperandKind::None;
    /// Access width in bits (8, 16, 32 or 64).
    u16 size_bits = 0;
    /// Valid when kind == Register.
    X86Reg reg = X86Reg::None;
    /// Valid when kind == Immediate; already sign-extended to 64 bits.
    i64 imm = 0;
    /// Valid when kind == Memory.
    MemRef mem{};
};

/// Upper bound on visible operands for the supported subset.
inline constexpr usize kMaxOperands = 4;

struct DecodedInst {
    Mnemonic mnemonic = Mnemonic::Invalid;
    /// Only meaningful when mnemonic == Mnemonic::Jcc.
    Cond cond = Cond::None;
    /// Guest address this instruction was decoded at.
    GuestAddr address = 0;
    /// Encoded length in bytes; always in [1, kMaxX86InstLength].
    u8 length = 0;
    u8 operand_count = 0;
    std::array<Operand, kMaxOperands> operands{};
    /// Set for JMP/Jcc with a relative displacement.
    bool has_branch_target = false;
    GuestAddr branch_target = 0;
    /// Absolute target of a RIP-relative memory operand, when one is present.
    bool has_rip_target = false;
    GuestAddr rip_target = 0;

    /// Address of the instruction following this one.
    [[nodiscard]] constexpr GuestAddr next_address() const noexcept {
        return address + length;
    }

    /// True when this instruction ends a basic block.
    [[nodiscard]] constexpr bool is_terminator() const noexcept {
        return mnemonic == Mnemonic::Jmp || mnemonic == Mnemonic::Jcc ||
               mnemonic == Mnemonic::Ret || mnemonic == Mnemonic::Call;
    }

    /// Precondition: index < operand_count.
    [[nodiscard]] const Operand& op(usize index) const noexcept {
        assert(index < operand_count && "operand index out of range");
        return operands[index];
    }
};

enum class DecodeError : u8 {
    None = 0,
    /// No bytes were supplied.
    EmptyBuffer,
    /// The buffer ends part-way through an otherwise valid instruction.
    TruncatedInstruction,
    /// The bytes do not form a legal x86-64 instruction.
    InvalidInstruction,
    /// A legal instruction outside the translator's supported subset.
    UnsupportedInstruction,
};

struct DecodeResult {
    DecodedInst inst{};
    DecodeError error = DecodeError::None;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return error == DecodeError::None;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok(); }
};

/// Wraps the Zydis decoder and lowers its output into the types above.
///
/// The Zydis headers are confined to the implementation file so nothing outside
/// src/decoder depends on them.
class Decoder {
public:
    Decoder();
    ~Decoder();

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;

    /// Decodes the single instruction at the start of `code`.
    ///
    /// `code` bounds every read: no more than min(code.size(),
    /// kMaxX86InstLength) bytes are ever examined, so a truncated or malformed
    /// buffer yields an error rather than reading past the end.
    [[nodiscard]] DecodeResult decode(std::span<const u8> code,
                                      GuestAddr address = 0) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string_view to_string(Mnemonic value) noexcept;
[[nodiscard]] std::string_view to_string(X86Reg value) noexcept;
[[nodiscard]] std::string_view to_string(Cond value) noexcept;
[[nodiscard]] std::string_view to_string(OperandKind value) noexcept;
[[nodiscard]] std::string_view to_string(DecodeError value) noexcept;

// Streaming operators so GoogleTest reports enumerator names rather than
// integers on failure.
std::ostream& operator<<(std::ostream& os, Mnemonic value);
std::ostream& operator<<(std::ostream& os, X86Reg value);
std::ostream& operator<<(std::ostream& os, Cond value);
std::ostream& operator<<(std::ostream& os, OperandKind value);
std::ostream& operator<<(std::ostream& os, DecodeError value);

}  // namespace dbt::decoder
