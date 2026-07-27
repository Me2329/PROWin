#pragma once

#include <span>
#include <string_view>

#include "dbt/common/types.hpp"
#include "dbt/decoder/decoder.hpp"
#include "dbt/ir/ir.hpp"

namespace dbt::frontend {

enum class LiftError : u8 {
    None = 0,
    /// The decoder rejected the bytes; see LiftResult::decode_error.
    DecodeFailed,
    /// A legal operand form the lifter cannot lower (e.g. writing an immediate).
    UnsupportedOperand,
    /// A Jcc with no flag-defining instruction earlier in the same block.
    /// Cross-block flag dependencies are not modelled.
    FlagsUnavailable,
    /// JMP through a register or memory operand.
    IndirectBranch,
    /// The instruction budget ran out before a terminator was reached.
    BlockTooLong,
};

struct LiftResult {
    ir::Function function;
    LiftError error = LiftError::None;
    /// Populated when error == LiftError::DecodeFailed.
    decoder::DecodeError decode_error = decoder::DecodeError::None;
    /// Guest address the failure occurred at.
    GuestAddr error_addr = 0;
    /// Number of guest instructions successfully lifted.
    usize guest_inst_count = 0;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return error == LiftError::None;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok(); }
};

/// Lifts one guest basic block into SSA IR.
///
/// Lifting stops at the first terminator. Every exit from the block -- taken
/// branch, fallthrough, or running out of budget -- becomes its own small exit
/// block that writes the next guest RIP and returns to the dispatcher, so the
/// runtime always learns where execution should resume.
class Lifter {
public:
    struct Options {
        /// Upper bound on guest instructions per block, so a run of
        /// non-terminating bytes cannot expand without limit.
        usize max_guest_insts = 64;
    };

    Lifter() = default;
    explicit Lifter(Options options) : options_(options) {}

    /// `code` holds the guest bytes beginning at `base`.
    [[nodiscard]] LiftResult lift_block(std::span<const u8> code, GuestAddr base) const;

    [[nodiscard]] const Options& options() const noexcept { return options_; }

private:
    Options options_;
    decoder::Decoder decoder_;
};

[[nodiscard]] std::string_view to_string(LiftError value) noexcept;

}  // namespace dbt::frontend
