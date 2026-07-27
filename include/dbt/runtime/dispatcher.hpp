#pragma once

#include <span>
#include <string_view>

#include "dbt/backend/compiler.hpp"
#include "dbt/backend/jit_memory.hpp"
#include "dbt/common/types.hpp"
#include "dbt/frontend/lifter.hpp"
#include "dbt/runtime/cpu_state.hpp"
#include "dbt/runtime/translation_cache.hpp"

namespace dbt::runtime {

enum class TranslateStatus : u8 {
    Ok = 0,
    /// The requested guest address lies outside the mapped code region.
    OutOfBounds,
    /// The frontend could not lift the block.
    LiftFailed,
    /// The backend could not compile the lifted IR.
    CompileFailed,
    /// Executable memory could not be allocated, written, or sealed.
    MemoryFailed,
};

struct TranslateResult {
    /// Owned by the cache; valid until the cache is cleared or the entry is
    /// invalidated.
    const TranslatedBlock* block = nullptr;
    TranslateStatus status = TranslateStatus::Ok;
    frontend::LiftError lift_error = frontend::LiftError::None;
    backend::CompileError compile_error = backend::CompileError::None;
    backend::JitError jit_error = backend::JitError::None;
    /// True when the block was already cached and nothing was recompiled.
    bool from_cache = false;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return status == TranslateStatus::Ok;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok(); }
};

enum class ExecStatus : u8 {
    /// Execution left the mapped code region: nothing further to run.
    Halted = 0,
    /// The step budget was exhausted while still inside the region.
    StepLimitReached,
    /// A block could not be translated; see RunResult::translate_status.
    TranslationFailed,
    /// This host cannot execute the AArch64 code the backend emits.
    HostCannotExecute,
};

struct RunResult {
    ExecStatus status = ExecStatus::Halted;
    /// Blocks executed.
    usize steps = 0;
    /// Guest address execution stopped at.
    GuestAddr stopped_at = 0;
    TranslateStatus translate_status = TranslateStatus::Ok;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return status == ExecStatus::Halted || status == ExecStatus::StepLimitReached;
    }
};

[[nodiscard]] std::string_view to_string(TranslateStatus value) noexcept;
[[nodiscard]] std::string_view to_string(ExecStatus value) noexcept;

/// Drives translation and execution of a guest program.
///
/// Guest code is supplied as a flat byte span mapped at `base`. Translation
/// works on any host and is fully testable; only `run` requires an ARM64 host,
/// and it refuses rather than jumping into foreign machine code.
class Dispatcher {
public:
    struct Options {
        /// Maximum blocks executed by one run() call, so a guest infinite loop
        /// cannot hang the process.
        usize max_steps = 1024;
        frontend::Lifter::Options lifter{};
        backend::Compiler::Options compiler{};
    };

    // Declared as two overloads rather than one with a defaulted argument:
    // inside the class body Options is not yet complete enough to brace
    // initialise as a default argument.
    Dispatcher(std::span<const u8> guest_code, GuestAddr base);
    Dispatcher(std::span<const u8> guest_code, GuestAddr base, Options options);

    /// Returns the block for `addr`, compiling it on first use.
    [[nodiscard]] TranslateResult translate(GuestAddr addr);

    /// Executes from `state.rip` until the code region is left, the step budget
    /// runs out, or translation fails.
    [[nodiscard]] RunResult run(CpuState& state);

    /// True when `addr` falls inside the mapped guest code region.
    [[nodiscard]] bool contains(GuestAddr addr) const noexcept {
        return addr >= base_ && (addr - base_) < guest_code_.size();
    }

    [[nodiscard]] const TranslationCache& cache() const noexcept { return cache_; }
    [[nodiscard]] TranslationCache& cache() noexcept { return cache_; }
    [[nodiscard]] GuestAddr base() const noexcept { return base_; }
    [[nodiscard]] const Options& options() const noexcept { return options_; }

private:
    std::span<const u8> guest_code_;
    GuestAddr base_ = 0;
    Options options_;
    TranslationCache cache_;
    frontend::Lifter lifter_;
    backend::Compiler compiler_;
};

}  // namespace dbt::runtime
