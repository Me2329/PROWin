#pragma once

#include <string_view>
#include <vector>

#include "dbt/backend/arm64_encoder.hpp"
#include "dbt/common/types.hpp"
#include "dbt/ir/ir.hpp"

namespace dbt::backend {

enum class CompileError : u8 {
    None = 0,
    /// The IR failed ir::Function::verify().
    InvalidIr,
    /// An opcode the backend does not lower.
    UnsupportedOpcode,
    /// A condition with no ARM64 equivalent (JP/JNP -- there is no parity flag).
    UnsupportedCondition,
    /// An operation on a width narrower than 64 bits. x86 sub-register writes
    /// have merge/zero-extend semantics the backend does not model yet, so they
    /// are refused rather than translated incorrectly.
    UnsupportedWidth,
    /// A branch displacement outside the reach of B or B.cond.
    BranchOutOfRange,
    /// The block needs more simultaneous temporaries than the pool provides.
    OutOfRegisters,
    /// The emitted code exceeded the configured budget.
    CodeTooLarge,
};

[[nodiscard]] std::string_view to_string(CompileError value) noexcept;

/// One chainable exit in a compiled block.
///
/// The emitted code reads `CpuState::link_table[slot]`; once the runtime has
/// translated `target`, writing that block's chained entry point into the slot
/// turns the exit into a direct branch.
struct LinkSite {
    /// Index into the link table, assigned by the compiler.
    usize slot = 0;
    /// Guest address this exit leads to.
    GuestAddr target = 0;
};

struct CompileResult {
    /// The emitted instruction stream, one word per AArch64 instruction.
    std::vector<Arm64Word> words;
    /// Chainable exits, in slot order.
    std::vector<LinkSite> link_sites;
    CompileError error = CompileError::None;
    /// Instruction that could not be lowered, when one is to blame.
    ir::InstId error_inst = ir::kNoInst;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return error == CompileError::None;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok(); }
    [[nodiscard]] usize size_bytes() const noexcept {
        return words.size() * kArm64InstSize;
    }
};

/// Lowers verified SSA IR into an AArch64 instruction stream.
///
/// The produced code is a function of type `void(CpuState*)`. Its prologue
/// saves the frame and the callee-saved registers it uses, parks the CpuState
/// pointer in x28, and loads the sixteen guest registers into x0-x15. Each
/// Return spills them back and restores the frame.
///
/// Because guest registers are pinned to x0-x15, an IR LoadGuestReg costs no
/// instruction at all -- it simply names the register the value already lives
/// in. Temporaries come from a small pool (x16, x17, x19, x20).
class Compiler {
public:
    struct Options {
        /// Upper bound on emitted instructions, so a pathological function
        /// cannot exhaust memory.
        usize max_words = 4096;
    };

    Compiler() = default;
    explicit Compiler(Options options) : options_(options) {}

    /// `first_link_slot` is the base index this block's chainable exits are
    /// numbered from, so slots stay unique across the whole link table. The
    /// offsets are baked into the emitted loads, hence they must be final at
    /// compile time.
    [[nodiscard]] CompileResult compile(const ir::Function& function,
                                        usize first_link_slot = 0) const;

    [[nodiscard]] const Options& options() const noexcept { return options_; }

private:
    Options options_;
};

/// Instructions the prologue occupies: frame save (3), frame pointer, CpuState
/// pointer, then one load per guest register. Exposed so tests can index past
/// it into the block body.
///
/// A block entered through a chained branch starts here instead, skipping the
/// sixteen register reloads because the predecessor left them live.
inline constexpr usize kPrologueWords = 5 + 16;

/// Instructions a chainable exit adds before its epilogue: load the table,
/// test it, load the slot, test it, branch. Both loads are guarded because a
/// CpuState with no link table must not be dereferenced.
inline constexpr usize kChainWords = 5;

}  // namespace dbt::backend
