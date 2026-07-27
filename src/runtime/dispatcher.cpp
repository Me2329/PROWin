#include "dbt/runtime/dispatcher.hpp"

#include <utility>

#include "dbt/version.hpp"

namespace dbt::runtime {

Dispatcher::Dispatcher(std::span<const u8> guest_code, GuestAddr base)
    : Dispatcher(guest_code, base, Options{}) {}

Dispatcher::Dispatcher(std::span<const u8> guest_code, GuestAddr base, Options options)
    : guest_code_(guest_code),
      base_(base),
      options_(options),
      lifter_(options.lifter),
      compiler_(options.compiler) {}

TranslateResult Dispatcher::translate(GuestAddr addr) {
    TranslateResult result;

    if (const TranslatedBlock* cached = cache_.find(addr); cached != nullptr) {
        result.block = cached;
        result.from_cache = true;
        return result;
    }

    if (!contains(addr)) {
        result.status = TranslateStatus::OutOfBounds;
        return result;
    }

    // The subspan bounds every byte the frontend can reach.
    const usize offset = addr - base_;
    const auto lifted = lifter_.lift_block(guest_code_.subspan(offset), addr);
    if (!lifted.ok()) {
        result.status = TranslateStatus::LiftFailed;
        result.lift_error = lifted.error;
        return result;
    }

    const auto compiled = compiler_.compile(lifted.function);
    if (!compiled.ok()) {
        result.status = TranslateStatus::CompileFailed;
        result.compile_error = compiled.error;
        return result;
    }

    backend::JitError jit_error = backend::JitError::None;
    backend::JitMemory memory =
        backend::JitMemory::allocate(compiled.size_bytes(), jit_error);
    if (jit_error != backend::JitError::None) {
        result.status = TranslateStatus::MemoryFailed;
        result.jit_error = jit_error;
        return result;
    }

    jit_error = memory.write(0, compiled.words);
    if (jit_error != backend::JitError::None) {
        result.status = TranslateStatus::MemoryFailed;
        result.jit_error = jit_error;
        return result;
    }

    // W^X: writable up to this point, executable from here on, never both.
    jit_error = memory.make_executable();
    if (jit_error != backend::JitError::None) {
        result.status = TranslateStatus::MemoryFailed;
        result.jit_error = jit_error;
        return result;
    }

    result.block = cache_.insert(
        addr, TranslatedBlock(addr, std::move(memory), compiled.words.size()));
    return result;
}

RunResult Dispatcher::run(CpuState& state) {
    RunResult result;
    result.stopped_at = state.rip;

    // Refuse rather than calling into instructions this CPU cannot decode.
    if (!host_can_execute_arm64()) {
        result.status = ExecStatus::HostCannotExecute;
        return result;
    }

    while (result.steps < options_.max_steps) {
        result.stopped_at = state.rip;

        if (!contains(state.rip)) {
            result.status = ExecStatus::Halted;
            return result;
        }

        const TranslateResult translated = translate(state.rip);
        if (!translated.ok()) {
            result.status = ExecStatus::TranslationFailed;
            result.translate_status = translated.status;
            return result;
        }

        // Each block writes the next guest RIP into the state before returning,
        // which is what advances this loop.
        translated.block->entry()(&state);
        ++result.steps;
    }

    result.stopped_at = state.rip;
    result.status = ExecStatus::StepLimitReached;
    return result;
}

std::string_view to_string(TranslateStatus value) noexcept {
    switch (value) {
    case TranslateStatus::Ok:
        return "ok";
    case TranslateStatus::OutOfBounds:
        return "out-of-bounds";
    case TranslateStatus::LiftFailed:
        return "lift-failed";
    case TranslateStatus::CompileFailed:
        return "compile-failed";
    case TranslateStatus::MemoryFailed:
        return "memory-failed";
    }
    return "unknown";
}

std::string_view to_string(ExecStatus value) noexcept {
    switch (value) {
    case ExecStatus::Halted:
        return "halted";
    case ExecStatus::StepLimitReached:
        return "step-limit-reached";
    case ExecStatus::TranslationFailed:
        return "translation-failed";
    case ExecStatus::HostCannotExecute:
        return "host-cannot-execute";
    }
    return "unknown";
}

}  // namespace dbt::runtime
