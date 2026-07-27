#pragma once

#include "dbt/backend/arm64_encoder.hpp"
#include "dbt/common/types.hpp"
#include "dbt/decoder/decoder.hpp"

namespace dbt::backend {

/// Direct mapping of the sixteen x86-64 general-purpose registers onto ARM64
/// x0-x15. The guest RSP lives in x4 like any other guest register -- the
/// native stack pointer stays separate, so translated code can still use the
/// host stack for its own frame.
///
/// The registers above x15 are reserved by the backend:
///   x16, x17  scratch (the architecture's IP0/IP1, free between calls)
///   x28       pointer to the CpuState the block operates on
///   x29, x30  frame pointer and link register, per the AArch64 PCS
[[nodiscard]] constexpr a64::Reg map_gpr(decoder::X86Reg reg) noexcept {
    return static_cast<a64::Reg>(static_cast<u8>(reg) & 0x0Fu);
}

inline constexpr a64::Reg kCpuStateBase = a64::Reg::X28;
inline constexpr a64::Reg kScratch0 = a64::Reg::X16;
inline constexpr a64::Reg kScratch1 = a64::Reg::X17;

/// Highest ARM64 register used to hold guest state.
inline constexpr a64::Reg kLastMappedGpr = a64::Reg::X15;

/// True when `reg` is a general-purpose register the mapping covers.
[[nodiscard]] constexpr bool is_mappable(decoder::X86Reg reg) noexcept {
    return decoder::is_gpr(reg);
}

/// Translates an x86 condition into its ARM64 equivalent.
///
/// The subtlety is the carry flag. After a subtraction x86 sets CF to mean
/// "a borrow occurred", whereas AArch64 sets C to mean "no borrow" -- the two
/// are inverses. So JB (CF==1) becomes CC/LO and JAE (CF==0) becomes CS/HS.
/// After an addition both architectures agree that carry means carry-out, and
/// the mapping flips back.
///
/// `flags_from_subtraction` must say which kind of operation produced the flags
/// being tested. Getting it wrong silently inverts unsigned comparisons.
///
/// Returns false for conditions with no ARM64 equivalent: AArch64 has no parity
/// flag, so JP/JNP cannot be expressed.
[[nodiscard]] constexpr bool to_arm64_condition(decoder::Cond cond,
                                                bool flags_from_subtraction,
                                                a64::Cond& out) noexcept {
    using decoder::Cond;
    switch (cond) {
    case Cond::Equal:
        out = a64::Cond::EQ;
        return true;
    case Cond::NotEqual:
        out = a64::Cond::NE;
        return true;
    case Cond::Below:  // CF == 1
        out = flags_from_subtraction ? a64::Cond::CC : a64::Cond::CS;
        return true;
    case Cond::AboveEqual:  // CF == 0
        out = flags_from_subtraction ? a64::Cond::CS : a64::Cond::CC;
        return true;
    case Cond::BelowEqual:  // CF == 1 || ZF == 1
        out = flags_from_subtraction ? a64::Cond::LS : a64::Cond::HI;
        return true;
    case Cond::Above:  // CF == 0 && ZF == 0
        out = flags_from_subtraction ? a64::Cond::HI : a64::Cond::LS;
        return true;
    case Cond::Sign:
        out = a64::Cond::MI;
        return true;
    case Cond::NotSign:
        out = a64::Cond::PL;
        return true;
    case Cond::Overflow:
        out = a64::Cond::VS;
        return true;
    case Cond::NotOverflow:
        out = a64::Cond::VC;
        return true;
    case Cond::Less:
        out = a64::Cond::LT;
        return true;
    case Cond::GreaterEqual:
        out = a64::Cond::GE;
        return true;
    case Cond::LessEqual:
        out = a64::Cond::LE;
        return true;
    case Cond::Greater:
        out = a64::Cond::GT;
        return true;
    case Cond::Parity:
    case Cond::NotParity:
        // AArch64 has no parity flag; emulating it would need an explicit
        // popcount sequence, which is outside the supported subset.
        break;
    case Cond::None:
        break;
    }
    return false;
}

}  // namespace dbt::backend
