# dbt

A minimal, heavily-tested **x86-64 → ARM64 binary translator** in modern C++20.

A working reference implementation of the architecture behind Rosetta 2, FEX-Emu
and QEMU: decoder → SSA IR → verifier → register allocator → ARM64 JIT →
dispatcher. Built to be read and extended.

```
x86-64 bytes ──▶ Decoder ──▶ SSA IR ──▶ Verifier ──▶ Compiler ──▶ ARM64 words
   (Zydis)                  (typed,                (pinned regs,    (W^X JIT
                            flag-aware)             flag model)      memory)
```

## Scope

**This translates code you hand it. It does not run applications.**

22 instruction forms at 32- and 64-bit widths, with full addressing modes. No
SSE/AVX, no x87, no syscall translation, no executable loader, no memory
sandbox. It is a foundation and a reference, not an emulator you can point at a
binary.

If you want to *run* x86 software on ARM today, use
[FEX-Emu](https://github.com/FEX-Emu/FEX), [box64](https://github.com/ptitSeb/box64)
or [QEMU](https://www.qemu.org/). This project exists to make the technique
legible.

## Build

Requires CMake 3.25+, Ninja, and a C++20 compiler. Zydis and GoogleTest are
fetched automatically.

```bash
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
```

On macOS use the `mac-debug` preset (Apple Clang, and AddressSanitizer is
available there via `mac-asan`):

```bash
cmake --preset mac-debug && cmake --build --preset mac-debug && ctest --preset mac-debug
```

## Using it

```cpp
#include "dbt/runtime/cpu_state.hpp"
#include "dbt/runtime/dispatcher.hpp"

// 48 01 D8  add rax, rbx
// C3        ret
constexpr std::array<dbt::u8, 4> code{0x48, 0x01, 0xD8, 0xC3};
dbt::runtime::Dispatcher dispatcher(code, 0x1000);

dbt::runtime::CpuState state;
state.rip = 0x1000;
state.gpr[0] = 10;   // RAX
state.gpr[3] = 32;   // RBX  (indices are the x86 hardware encoding)

const auto result = dispatcher.run(state);   // state.gpr[0] == 42
```

`run()` requires an ARM64 host and returns `HostCannotExecute` elsewhere.
`translate()` works everywhere and hands back the emitted instruction words, so
the whole pipeline is inspectable on an x86 development machine.

## Design notes

**Instruction encodings are verified at compile time.** Every ARM64 encoder is a
`constexpr` function checked against golden words with `static_assert`, so an
encoding regression breaks the build rather than a test run. This also means the
backend is fully verifiable on hosts that cannot execute what it emits.

**Guest registers are pinned.** RAX–R15 map onto x0–x15, so an IR
`LoadGuestReg` costs zero instructions — it just names the register the value
already lives in. x28 holds the `CpuState` pointer; x16/x17/x19/x20 are
temporaries.

**Flags are modelled per-flag, not per-instruction.** `defined_flags(Opcode)`
records which flags survive lowering and `required_flags(Cond)` which a branch
reads; the backend refuses any branch where `required & ~defined` is non-empty.
So:

| Sequence | Result |
|---|---|
| `cmp rax, rbx ; jb` | translated |
| `shr rax, 1 ; jb` | **refused** — x86 CF is the shifted-out bit, which `LSR`+`TST` cannot reproduce |
| `shr rax, 1 ; je` | translated — ZF genuinely survives |
| `inc rax ; jb` | **refused** — x86 preserves CF on purpose; `ADDS` overwrites it |

**Carry inverts on subtraction.** After a subtract, x86 CF means *borrow* while
AArch64 C means *no borrow*, so `JB` becomes `B.CC`. After an addition both agree
and the mapping flips back — the compiler inspects the flag-producing opcode to
decide.

**Failures are typed, never silent.** `UnsupportedCondition`,
`UnsupportedWidth`, `FlagsUnavailable`, `IndirectBranch`, `BranchOutOfRange`.
The translator refuses rather than emitting plausible-but-wrong code.

**JIT memory is W^X.** Mapped read/write, filled, then flipped to read/execute —
never both, with no path back to writable. Apple Silicon uses `MAP_JIT` with
`pthread_jit_write_protect_np`, since the kernel refuses `mprotect(PROT_EXEC)`
on ordinary anonymous memory there.

## Supported instructions

`MOV` `ADD` `SUB` `CMP` `AND` `OR` `XOR` `TEST` `NOT` `NEG` `LEA` `PUSH` `POP`
`CALL` `SHL` `SHR` `SAR` `INC` `DEC` `JMP` `Jcc` `RET`

Addressing: register, immediate, `[base + index*scale + disp]`, RIP-relative.

## Known limitations

- **No memory sandbox.** Guest addresses are dereferenced as host pointers. Do
  not feed this untrusted input.
- **No self-modifying-code detection.** `TranslationCache::invalidate()` exists
  but nothing calls it automatically.
- **No memory-ordering model.** x86 is TSO, ARM is weakly ordered. Multithreaded
  guests would need barriers, or Apple Silicon's hardware TSO mode.
- 8- and 16-bit widths are refused (they merge rather than zero-extend).
- Variable-count shifts (`shl reg, cl`) and indirect `JMP`/`CALL` are refused.
- `JP`/`JNP` are refused — AArch64 has no parity flag.

## Roadmap

`MOVZX`/`MOVSX` → indirect branches → block chaining (patch direct branches
between translated blocks instead of returning to the dispatcher) → lazy flags
(materialise NZCV only when a condition consumes it) → SSE2.

## License

MIT. Zydis is MIT, GoogleTest is BSD-3-Clause.
