#pragma once

#include "dbt/common/types.hpp"

/// Direct AArch64 instruction encoders.
///
/// Every A64 instruction is a single 32-bit word, so each encoder is a pure
/// constexpr function returning that word. This keeps the backend dependency
/// free and lets the golden-encoding tests run as static_asserts on any host,
/// including the x86_64 machines where the emitted code cannot be executed.
///
/// Encoders assume their inputs are in range; use the fits_* predicates first.
/// Out-of-range fields are masked rather than silently overflowing into
/// neighbouring bits.
namespace dbt::backend::a64 {

/// General-purpose register operand. Encoding 31 means XZR for most
/// instructions and SP for the load/store and add/sub-immediate forms.
enum class Reg : u8 {
    X0 = 0, X1, X2, X3, X4, X5, X6, X7,
    X8, X9, X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X19, X20, X21, X22, X23,
    X24, X25, X26, X27, X28, X29, X30,
    /// Encoding 31: XZR in data-processing, SP in load/store and add/sub imm.
    ZrSp = 31,
};

inline constexpr Reg kZeroReg = Reg::ZrSp;
inline constexpr Reg kStackPointer = Reg::ZrSp;
inline constexpr Reg kLinkReg = Reg::X30;
inline constexpr Reg kFramePointer = Reg::X29;

/// AArch64 condition codes, in encoding order.
enum class Cond : u8 {
    EQ = 0,  ///< equal (Z==1)
    NE,      ///< not equal
    CS,      ///< carry set / unsigned higher or same (HS)
    CC,      ///< carry clear / unsigned lower (LO)
    MI,      ///< negative
    PL,      ///< positive or zero
    VS,      ///< overflow set
    VC,      ///< overflow clear
    HI,      ///< unsigned higher
    LS,      ///< unsigned lower or same
    GE,      ///< signed greater or equal
    LT,      ///< signed less than
    GT,      ///< signed greater than
    LE,      ///< signed less or equal
    AL,      ///< always
    NV,      ///< always (reserved encoding)
};

namespace detail {

[[nodiscard]] constexpr u32 r(Reg reg) noexcept {
    return static_cast<u32>(reg) & 0x1Fu;
}

/// Truncates `value` to `width` bits.
[[nodiscard]] constexpr u32 mask(u32 value, u32 width) noexcept {
    return (width >= 32u) ? value : (value & ((1u << width) - 1u));
}

}  // namespace detail

/// Operand width.
///
/// AArch64 selects it with the `sf` bit (31) on data-processing instructions:
/// 1 picks the 64-bit X registers, 0 the 32-bit W registers. Writing a W
/// register zero-extends into the full X register, which is precisely x86's
/// rule for a 32-bit destination -- so the two architectures line up and no
/// explicit masking is needed.
enum class Width : u8 {
    W32,
    X64,
};

/// Re-encodes a data-processing word for `width`.
///
/// Valid only for instructions whose 64-bit form sets sf, which covers every
/// arithmetic, logical and move-wide encoder below. The load/store forms encode
/// their size in a different field and have dedicated 32-bit helpers instead.
[[nodiscard]] constexpr Arm64Word with_width(Arm64Word word, Width width) noexcept {
    return (width == Width::X64) ? word : (word & ~0x80000000u);
}

// --- Range predicates ------------------------------------------------------

/// ADD/SUB immediate forms carry an unsigned 12-bit immediate.
[[nodiscard]] constexpr bool fits_imm12(u64 value) noexcept {
    return value <= 0xFFFu;
}

/// LDR/STR unsigned-offset forms scale the offset by the access size, so a
/// 64-bit access needs an 8-byte-aligned offset below 32768.
[[nodiscard]] constexpr bool fits_ldst_offset64(u64 byte_offset) noexcept {
    return (byte_offset % 8u == 0u) && ((byte_offset / 8u) <= 0xFFFu);
}

/// LDP/STP carry a signed 7-bit immediate scaled by 8.
[[nodiscard]] constexpr bool fits_pair_offset(i64 byte_offset) noexcept {
    return (byte_offset % 8 == 0) && (byte_offset / 8 >= -64) &&
           (byte_offset / 8 <= 63);
}

/// Unconditional B reaches +/-128 MiB.
[[nodiscard]] constexpr bool fits_branch26(i64 byte_offset) noexcept {
    return (byte_offset % 4 == 0) && (byte_offset / 4 >= -(1 << 25)) &&
           (byte_offset / 4 <= (1 << 25) - 1);
}

/// B.cond reaches +/-1 MiB.
[[nodiscard]] constexpr bool fits_branch19(i64 byte_offset) noexcept {
    return (byte_offset % 4 == 0) && (byte_offset / 4 >= -(1 << 18)) &&
           (byte_offset / 4 <= (1 << 18) - 1);
}

// --- Move wide immediate ---------------------------------------------------

/// MOVZ Xd, #imm16, LSL #shift   (shift is 0, 16, 32 or 48)
[[nodiscard]] constexpr Arm64Word movz(Reg rd, u16 imm16, u8 shift = 0) noexcept {
    const u32 hw = (shift / 16u) & 0x3u;
    return 0xD2800000u | (hw << 21) | (static_cast<u32>(imm16) << 5) | detail::r(rd);
}

/// MOVK Xd, #imm16, LSL #shift -- keeps the other bits of Xd.
[[nodiscard]] constexpr Arm64Word movk(Reg rd, u16 imm16, u8 shift = 0) noexcept {
    const u32 hw = (shift / 16u) & 0x3u;
    return 0xF2800000u | (hw << 21) | (static_cast<u32>(imm16) << 5) | detail::r(rd);
}

/// MOVN Xd, #imm16, LSL #shift -- writes ~(imm16 << shift).
[[nodiscard]] constexpr Arm64Word movn(Reg rd, u16 imm16, u8 shift = 0) noexcept {
    const u32 hw = (shift / 16u) & 0x3u;
    return 0x92800000u | (hw << 21) | (static_cast<u32>(imm16) << 5) | detail::r(rd);
}

// --- Data processing, shifted register -------------------------------------

/// ADD/ADDS Xd, Xn, Xm
[[nodiscard]] constexpr Arm64Word add_reg(Reg rd, Reg rn, Reg rm,
                                          bool set_flags = false) noexcept {
    const u32 s = set_flags ? (1u << 29) : 0u;
    return 0x8B000000u | s | (detail::r(rm) << 16) | (detail::r(rn) << 5) |
           detail::r(rd);
}

/// SUB/SUBS Xd, Xn, Xm
[[nodiscard]] constexpr Arm64Word sub_reg(Reg rd, Reg rn, Reg rm,
                                          bool set_flags = false) noexcept {
    const u32 s = set_flags ? (1u << 29) : 0u;
    return 0xCB000000u | s | (detail::r(rm) << 16) | (detail::r(rn) << 5) |
           detail::r(rd);
}

/// CMP Xn, Xm -- an alias for SUBS XZR, Xn, Xm.
[[nodiscard]] constexpr Arm64Word cmp_reg(Reg rn, Reg rm) noexcept {
    return sub_reg(kZeroReg, rn, rm, true);
}

/// MOV Xd, Xm -- an alias for ORR Xd, XZR, Xm.
[[nodiscard]] constexpr Arm64Word mov_reg(Reg rd, Reg rm) noexcept {
    return 0xAA0003E0u | (detail::r(rm) << 16) | detail::r(rd);
}

/// NEG/NEGS Xd, Xm -- an alias for SUB/SUBS Xd, XZR, Xm.
[[nodiscard]] constexpr Arm64Word neg_reg(Reg rd, Reg rm,
                                          bool set_flags = false) noexcept {
    return sub_reg(rd, kZeroReg, rm, set_flags);
}

// --- Logical, shifted register ---------------------------------------------
//
// AArch64 provides a flag-setting AND (ANDS) but no ORRS or EORS, so OR and XOR
// need a following TST to publish their flags.

/// AND/ANDS Xd, Xn, Xm
[[nodiscard]] constexpr Arm64Word and_reg(Reg rd, Reg rn, Reg rm,
                                          bool set_flags = false) noexcept {
    const u32 base = set_flags ? 0xEA000000u : 0x8A000000u;
    return base | (detail::r(rm) << 16) | (detail::r(rn) << 5) | detail::r(rd);
}

/// TST Xn, Xm -- an alias for ANDS XZR, Xn, Xm.
///
/// Sets N and Z from the result and clears C and V, which is exactly what x86
/// does to SF/ZF/CF/OF for its logical instructions.
[[nodiscard]] constexpr Arm64Word tst_reg(Reg rn, Reg rm) noexcept {
    return and_reg(kZeroReg, rn, rm, true);
}

/// ORR Xd, Xn, Xm
[[nodiscard]] constexpr Arm64Word orr_reg(Reg rd, Reg rn, Reg rm) noexcept {
    return 0xAA000000u | (detail::r(rm) << 16) | (detail::r(rn) << 5) | detail::r(rd);
}

/// EOR Xd, Xn, Xm
[[nodiscard]] constexpr Arm64Word eor_reg(Reg rd, Reg rn, Reg rm) noexcept {
    return 0xCA000000u | (detail::r(rm) << 16) | (detail::r(rn) << 5) | detail::r(rd);
}

/// MVN Xd, Xm -- an alias for ORN Xd, XZR, Xm.
[[nodiscard]] constexpr Arm64Word mvn_reg(Reg rd, Reg rm) noexcept {
    return 0xAA2003E0u | (detail::r(rm) << 16) | detail::r(rd);
}

// --- Data processing, immediate --------------------------------------------

/// ADD/ADDS Xd, Xn, #imm12
[[nodiscard]] constexpr Arm64Word add_imm(Reg rd, Reg rn, u16 imm12,
                                          bool set_flags = false) noexcept {
    const u32 s = set_flags ? (1u << 29) : 0u;
    return 0x91000000u | s | (detail::mask(imm12, 12) << 10) | (detail::r(rn) << 5) |
           detail::r(rd);
}

/// SUB/SUBS Xd, Xn, #imm12
[[nodiscard]] constexpr Arm64Word sub_imm(Reg rd, Reg rn, u16 imm12,
                                          bool set_flags = false) noexcept {
    const u32 s = set_flags ? (1u << 29) : 0u;
    return 0xD1000000u | s | (detail::mask(imm12, 12) << 10) | (detail::r(rn) << 5) |
           detail::r(rd);
}

/// CMP Xn, #imm12 -- an alias for SUBS XZR, Xn, #imm12.
[[nodiscard]] constexpr Arm64Word cmp_imm(Reg rn, u16 imm12) noexcept {
    return sub_imm(kZeroReg, rn, imm12, true);
}

// The shift aliases take an explicit width rather than going through
// with_width: their 32-bit forms change the N field and the immr/imms widths as
// well as sf, so clearing bit 31 alone would not produce a valid encoding.

/// LSL Xd, Xn, #shift -- an alias for UBFM with the canonical field values.
[[nodiscard]] constexpr Arm64Word lsl_imm(Reg rd, Reg rn, u8 shift,
                                          Width width = Width::X64) noexcept {
    if (width == Width::X64) {
        const u32 s = shift & 0x3Fu;
        return 0xD3400000u | (((64u - s) & 0x3Fu) << 16) | ((63u - s) << 10) |
               (detail::r(rn) << 5) | detail::r(rd);
    }
    const u32 s = shift & 0x1Fu;
    return 0x53000000u | (((32u - s) & 0x1Fu) << 16) | ((31u - s) << 10) |
           (detail::r(rn) << 5) | detail::r(rd);
}

/// LSR Xd, Xn, #shift -- an alias for UBFM Xd, Xn, #shift, #63.
[[nodiscard]] constexpr Arm64Word lsr_imm(Reg rd, Reg rn, u8 shift,
                                          Width width = Width::X64) noexcept {
    if (width == Width::X64) {
        return 0xD3400000u | ((shift & 0x3Fu) << 16) | (63u << 10) |
               (detail::r(rn) << 5) | detail::r(rd);
    }
    return 0x53000000u | ((shift & 0x1Fu) << 16) | (31u << 10) |
           (detail::r(rn) << 5) | detail::r(rd);
}

/// ASR Xd, Xn, #shift -- an alias for SBFM Xd, Xn, #shift, #63.
[[nodiscard]] constexpr Arm64Word asr_imm(Reg rd, Reg rn, u8 shift,
                                          Width width = Width::X64) noexcept {
    if (width == Width::X64) {
        return 0x93400000u | ((shift & 0x3Fu) << 16) | (63u << 10) |
               (detail::r(rn) << 5) | detail::r(rd);
    }
    return 0x13000000u | ((shift & 0x1Fu) << 16) | (31u << 10) |
           (detail::r(rn) << 5) | detail::r(rd);
}

// --- Load and store, unsigned offset ---------------------------------------

/// LDR Xt, [Xn, #byte_offset] -- offset must satisfy fits_ldst_offset64.
[[nodiscard]] constexpr Arm64Word ldr_imm(Reg rt, Reg rn, u32 byte_offset = 0) noexcept {
    const u32 scaled = detail::mask(byte_offset / 8u, 12);
    return 0xF9400000u | (scaled << 10) | (detail::r(rn) << 5) | detail::r(rt);
}

/// STR Xt, [Xn, #byte_offset]
[[nodiscard]] constexpr Arm64Word str_imm(Reg rt, Reg rn, u32 byte_offset = 0) noexcept {
    const u32 scaled = detail::mask(byte_offset / 8u, 12);
    return 0xF9000000u | (scaled << 10) | (detail::r(rn) << 5) | detail::r(rt);
}

/// A 32-bit load/store scales its offset by 4 rather than 8.
[[nodiscard]] constexpr bool fits_ldst_offset32(u64 byte_offset) noexcept {
    return (byte_offset % 4u == 0u) && ((byte_offset / 4u) <= 0xFFFu);
}

/// LDR Wt, [Xn, #byte_offset] -- zero-extends into the full X register.
[[nodiscard]] constexpr Arm64Word ldr_imm32(Reg rt, Reg rn,
                                            u32 byte_offset = 0) noexcept {
    const u32 scaled = detail::mask(byte_offset / 4u, 12);
    return 0xB9400000u | (scaled << 10) | (detail::r(rn) << 5) | detail::r(rt);
}

/// STR Wt, [Xn, #byte_offset]
[[nodiscard]] constexpr Arm64Word str_imm32(Reg rt, Reg rn,
                                            u32 byte_offset = 0) noexcept {
    const u32 scaled = detail::mask(byte_offset / 4u, 12);
    return 0xB9000000u | (scaled << 10) | (detail::r(rn) << 5) | detail::r(rt);
}

/// LDRB Wt, [Xn, #byte_offset] -- zero-extending byte load, offset unscaled.
[[nodiscard]] constexpr Arm64Word ldrb(Reg rt, Reg rn, u32 byte_offset = 0) noexcept {
    return 0x39400000u | (detail::mask(byte_offset, 12) << 10) |
           (detail::r(rn) << 5) | detail::r(rt);
}

/// LDRH Wt, [Xn, #byte_offset] -- zero-extending halfword load, offset /2.
[[nodiscard]] constexpr Arm64Word ldrh(Reg rt, Reg rn, u32 byte_offset = 0) noexcept {
    return 0x79400000u | (detail::mask(byte_offset / 2u, 12) << 10) |
           (detail::r(rn) << 5) | detail::r(rt);
}

/// STRB Wt, [Xn, #byte_offset]
[[nodiscard]] constexpr Arm64Word strb(Reg rt, Reg rn, u32 byte_offset = 0) noexcept {
    return 0x39000000u | (detail::mask(byte_offset, 12) << 10) |
           (detail::r(rn) << 5) | detail::r(rt);
}

/// STRH Wt, [Xn, #byte_offset]
[[nodiscard]] constexpr Arm64Word strh(Reg rt, Reg rn, u32 byte_offset = 0) noexcept {
    return 0x79000000u | (detail::mask(byte_offset / 2u, 12) << 10) |
           (detail::r(rn) << 5) | detail::r(rt);
}

// --- Extension -------------------------------------------------------------
//
// UXTB/UXTH write a W register, which zero-extends into the full X register,
// so they cover x86 MOVZX at either destination width. The sign-extending
// forms are SBFM and do depend on the destination width.

/// UXTB Wd, Wn -- zero-extend the low byte.
[[nodiscard]] constexpr Arm64Word uxtb(Reg rd, Reg rn) noexcept {
    return 0x53001C00u | (detail::r(rn) << 5) | detail::r(rd);
}

/// UXTH Wd, Wn -- zero-extend the low halfword.
[[nodiscard]] constexpr Arm64Word uxth(Reg rd, Reg rn) noexcept {
    return 0x53003C00u | (detail::r(rn) << 5) | detail::r(rd);
}

/// SXTB Xd/Wd, Wn -- sign-extend the low byte.
[[nodiscard]] constexpr Arm64Word sxtb(Reg rd, Reg rn,
                                       Width width = Width::X64) noexcept {
    const u32 base = (width == Width::X64) ? 0x93401C00u : 0x13001C00u;
    return base | (detail::r(rn) << 5) | detail::r(rd);
}

/// SXTH Xd/Wd, Wn -- sign-extend the low halfword.
[[nodiscard]] constexpr Arm64Word sxth(Reg rd, Reg rn,
                                       Width width = Width::X64) noexcept {
    const u32 base = (width == Width::X64) ? 0x93403C00u : 0x13003C00u;
    return base | (detail::r(rn) << 5) | detail::r(rd);
}

/// SXTW Xd, Wn -- sign-extend a 32-bit value to 64 bits.
[[nodiscard]] constexpr Arm64Word sxtw(Reg rd, Reg rn) noexcept {
    return 0x93407C00u | (detail::r(rn) << 5) | detail::r(rd);
}

/// STP Xt, Xt2, [Xn, #offset]!  (pre-index, writes back to Xn)
[[nodiscard]] constexpr Arm64Word stp_pre(Reg rt, Reg rt2, Reg rn,
                                          i32 byte_offset) noexcept {
    const u32 imm7 = detail::mask(static_cast<u32>(byte_offset / 8), 7);
    return 0xA9800000u | (imm7 << 15) | (detail::r(rt2) << 10) | (detail::r(rn) << 5) |
           detail::r(rt);
}

/// LDP Xt, Xt2, [Xn], #offset   (post-index, writes back to Xn)
[[nodiscard]] constexpr Arm64Word ldp_post(Reg rt, Reg rt2, Reg rn,
                                           i32 byte_offset) noexcept {
    const u32 imm7 = detail::mask(static_cast<u32>(byte_offset / 8), 7);
    return 0xA8C00000u | (imm7 << 15) | (detail::r(rt2) << 10) | (detail::r(rn) << 5) |
           detail::r(rt);
}

/// STP Xt, Xt2, [Xn, #offset]   (signed offset, no write-back)
[[nodiscard]] constexpr Arm64Word stp_off(Reg rt, Reg rt2, Reg rn,
                                          i32 byte_offset) noexcept {
    const u32 imm7 = detail::mask(static_cast<u32>(byte_offset / 8), 7);
    return 0xA9000000u | (imm7 << 15) | (detail::r(rt2) << 10) | (detail::r(rn) << 5) |
           detail::r(rt);
}

/// LDP Xt, Xt2, [Xn, #offset]   (signed offset, no write-back)
[[nodiscard]] constexpr Arm64Word ldp_off(Reg rt, Reg rt2, Reg rn,
                                          i32 byte_offset) noexcept {
    const u32 imm7 = detail::mask(static_cast<u32>(byte_offset / 8), 7);
    return 0xA9400000u | (imm7 << 15) | (detail::r(rt2) << 10) | (detail::r(rn) << 5) |
           detail::r(rt);
}

// --- Branches --------------------------------------------------------------

/// B #byte_offset -- offset is relative to this instruction.
[[nodiscard]] constexpr Arm64Word b(i32 byte_offset) noexcept {
    const u32 imm26 = detail::mask(static_cast<u32>(byte_offset / 4), 26);
    return 0x14000000u | imm26;
}

/// B.<cond> #byte_offset
[[nodiscard]] constexpr Arm64Word b_cond(Cond cond, i32 byte_offset) noexcept {
    const u32 imm19 = detail::mask(static_cast<u32>(byte_offset / 4), 19);
    return 0x54000000u | (imm19 << 5) | (static_cast<u32>(cond) & 0xFu);
}

/// BR Xn -- indirect branch.
[[nodiscard]] constexpr Arm64Word br(Reg rn) noexcept {
    return 0xD61F0000u | (detail::r(rn) << 5);
}

/// RET Xn -- returns to the address in Xn, X30 by default.
[[nodiscard]] constexpr Arm64Word ret(Reg rn = kLinkReg) noexcept {
    return 0xD65F0000u | (detail::r(rn) << 5);
}

/// NOP
[[nodiscard]] constexpr Arm64Word nop() noexcept {
    return 0xD503201Fu;
}

}  // namespace dbt::backend::a64
