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

struct CompileResult {
    /// The emitted instruction stream, one word per AArch64 instruction.
    std::vector<Arm64Word> words;
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

    [[nodiscard]] CompileResult compile(const ir::Function& function) const;

    [[nodiscard]] const Options& options() const noexcept { return options_; }

private:
    Options options_;
};

/// Instructions the prologue occupies: frame save (3), frame pointer, CpuState
/// pointer, then one load per guest register. Exposed so tests can index past
/// it into the block body.
inline constexpr usize kPrologueWords = 5 + 16;

}  // namespace dbt::backend
