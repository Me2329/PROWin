#pragma once

#include <array>
#include <cstddef>

#include "dbt/common/types.hpp"
#include "dbt/decoder/decoder.hpp"

namespace dbt::runtime {

/// The emulated x86-64 machine state a translated block operates on.
///
/// Translated code addresses these fields by fixed byte offsets held in a base
/// register, so the layout is part of the ABI between the compiler and the
/// runtime. The static_asserts below pin it: changing a field order or type
/// without updating the emitted offsets is a build error, not a crash at run
/// time.
struct CpuState {
    /// General-purpose registers, indexed by the x86 hardware encoding
    /// (RAX == 0 ... R15 == 15).
    std::array<u64, decoder::kNumGpr> gpr{};
    /// Next guest instruction to execute. Written by every block exit.
    u64 rip = 0;
    /// Emulated EFLAGS.
    ///
    /// Not yet written by compiled code: the IR resolves every conditional
    /// branch against a flag definition inside the same block, so live flags
    /// never cross a block boundary. The field exists for the runtime and for
    /// future cross-block flag support.
    u64 rflags = 0;
    /// Base of the dispatcher's block-link table, or null when chaining is off.
    ///
    /// A chainable block exit loads its successor's host address from
    /// `link_table[slot]` and branches straight to it, skipping the successor's
    /// prologue. A null entry means "not linked yet" and falls back to the
    /// dispatcher. The table lives in ordinary heap memory, so linking is a
    /// plain data write -- no executable page ever becomes writable.
    void** link_table = nullptr;
};

inline constexpr usize kGprOffset = 0;
inline constexpr usize kRipOffset = 128;
inline constexpr usize kRflagsOffset = 136;
inline constexpr usize kLinkTableOffset = 144;

static_assert(offsetof(CpuState, gpr) == kGprOffset);
static_assert(offsetof(CpuState, rip) == kRipOffset);
static_assert(offsetof(CpuState, rflags) == kRflagsOffset);
static_assert(offsetof(CpuState, link_table) == kLinkTableOffset);
static_assert(sizeof(CpuState) == 152);
static_assert(alignof(CpuState) == 8);

/// Byte offset of a guest register within CpuState.
[[nodiscard]] constexpr u32 gpr_offset(decoder::X86Reg reg) noexcept {
    // The mask promotes to unsigned int, which is already u32.
    return (static_cast<u8>(reg) & 0x0Fu) * 8u;
}

/// Signature every compiled block exposes: it is handed the state to run on.
using BlockFn = void (*)(CpuState*);

}  // namespace dbt::runtime
