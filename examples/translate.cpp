// A guided tour of the translation pipeline.
//
// Takes one real compiled function, walks it through decode -> lift -> compile,
// prints what each stage produced, and -- on an ARM64 host -- runs the result.
//
// This builds and runs everywhere. Only the final execution step needs ARM64;
// everywhere else the emitted instruction words are still printed, which is how
// the backend is developed on x86 machines.

#include <array>
#include <cstdio>
#include <iostream>
#include <string>

#include "dbt/backend/compiler.hpp"
#include "dbt/frontend/lifter.hpp"
#include "dbt/runtime/cpu_state.hpp"
#include "dbt/runtime/dispatcher.hpp"
#include "dbt/version.hpp"

namespace {

constexpr dbt::GuestAddr kGuestBase = 0x1000;

/// long add(long a, long b) { return a + b; }   -- clang -O0, System V ABI.
constexpr std::array<dbt::u8, 22> kProgram{
    0x55,                    // push rbp
    0x48, 0x89, 0xE5,        // mov  rbp, rsp
    0x48, 0x89, 0x7D, 0xF8,  // mov  [rbp-8], rdi     (a)
    0x48, 0x89, 0x75, 0xF0,  // mov  [rbp-16], rsi    (b)
    0x48, 0x8B, 0x45, 0xF8,  // mov  rax, [rbp-8]
    0x48, 0x03, 0x45, 0xF0,  // add  rax, [rbp-16]
    0x5D,                    // pop  rbp
    0xC3                     // ret
};

void rule(const char* title) {
    std::cout << "\n== " << title << " ==\n\n";
}

}  // namespace

int main() {
    std::cout << "dbt " << dbt::version_string() << "  --  x86-64 to ARM64\n"
              << "host can execute ARM64: "
              << (dbt::host_can_execute_arm64() ? "yes" : "no") << '\n';

    // --- Stage 1: decode and lift into SSA IR ------------------------------
    const dbt::frontend::Lifter lifter;
    const auto lifted = lifter.lift_block(kProgram, kGuestBase);
    if (!lifted.ok()) {
        std::cerr << "lift failed: " << dbt::frontend::to_string(lifted.error)
                  << " at " << lifted.error_addr << '\n';
        return 1;
    }

    rule("SSA IR");
    std::cout << lifted.function.to_string() << lifted.guest_inst_count
              << " guest instructions -> " << lifted.function.inst_count()
              << " IR instructions across " << lifted.function.block_count()
              << " blocks\n";

    // The verifier is what makes the IR trustworthy: every operand defined
    // before use, one terminator per block, branch targets in range.
    std::string error;
    std::cout << "verify: " << (lifted.function.verify(&error) ? "ok" : error)
              << '\n';

    // --- Stage 2: compile to ARM64 -----------------------------------------
    const dbt::backend::Compiler compiler;
    const auto compiled = compiler.compile(lifted.function);
    if (!compiled.ok()) {
        std::cerr << "compile failed: " << dbt::backend::to_string(compiled.error)
                  << '\n';
        return 1;
    }

    rule("ARM64");
    for (dbt::usize i = 0; i < compiled.words.size(); ++i) {
        std::printf("%4u  %08x%s\n", static_cast<unsigned>(i),
                    static_cast<unsigned>(compiled.words[i]),
                    (i + 1 == dbt::backend::kPrologueWords)
                        ? "   <- prologue ends; chained branches enter here"
                        : "");
    }
    std::cout << compiled.words.size() << " instructions, "
              << compiled.size_bytes() << " bytes, " << compiled.link_sites.size()
              << " chainable exits\n";

    // --- Stage 3: execute ---------------------------------------------------
    rule("execution");
    if (!dbt::host_can_execute_arm64()) {
        std::cout << "skipped: this host cannot execute ARM64.\n"
                     "Every encoding above is still verified at compile time by\n"
                     "static_assert, which is how the backend is developed on "
                     "x86.\n";
        return 0;
    }

    dbt::runtime::Dispatcher dispatcher(kProgram, kGuestBase);
    dbt::runtime::CpuState state;
    state.rip = kGuestBase;
    state.gpr[7] = 10;  // RDI -- first argument
    state.gpr[6] = 32;  // RSI -- second argument

    // RET pops its return address off the guest stack, so give it one that
    // lands outside the code region; the dispatcher then halts.
    std::array<dbt::u64, 64> stack{};
    stack[32] = 0xDEAD0000;                                 // sentinel return
    state.gpr[4] = reinterpret_cast<dbt::u64>(&stack[32]);  // RSP
    state.gpr[5] = reinterpret_cast<dbt::u64>(&stack[48]);  // RBP

    const auto result = dispatcher.run(state);
    std::cout << "status : " << dbt::runtime::to_string(result.status) << '\n'
              << "blocks : " << result.steps << '\n'
              << "linked : " << dispatcher.linked_exits() << '\n'
              << "rax    : " << state.gpr[0] << "   (expected 42)\n";

    return (state.gpr[0] == 42) ? 0 : 1;
}
