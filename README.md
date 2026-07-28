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
or [QEMU](https://www.qemu.org/).

If you want to **build** something like them — or understand how they work —
start here. See [Building an emulator on this](#building-an-emulator-on-this).

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

## Try it

```bash
cmake --preset debug && cmake --build --preset debug && ./build/debug/examples/dbt_translate
```

Takes one clang-compiled function, prints the SSA IR, prints the emitted ARM64
words, and — on an ARM64 host — runs it. The first two stages work on any
machine, which is how the backend is developed on x86.

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

**Blocks chain through a link table, not patched branches.** An exit with a
statically known target loads its successor's address from
`CpuState::link_table[slot]` and branches straight into its body, skipping the
prologue because the predecessor left the guest registers live. That replaces a
~35-instruction dispatcher round trip — sixteen spills, frame teardown, hash
lookup, re-entry, sixteen reloads — with five instructions.

Patching direct branches would have been faster still, but it needs a code arena
(so blocks land within ±128 MB) and unsealing executable pages to RW for each
patch. Reading from heap memory costs one extra load and keeps executable pages
permanently non-writable. Both loads are guarded: a null table means chaining is
off, a null slot means the successor is not translated yet, and either falls
back to the dispatcher.

**JIT memory is W^X.** Mapped read/write, filled, then flipped to read/execute —
never both, with no path back to writable. Apple Silicon uses `MAP_JIT` with
`pthread_jit_write_protect_np`, since the kernel refuses `mprotect(PROT_EXEC)`
on ordinary anonymous memory there.

## Supported instructions

`MOV` `MOVZX` `MOVSX` `ADD` `SUB` `CMP` `AND` `OR` `XOR` `TEST` `NOT` `NEG`
`LEA` `PUSH` `POP` `CALL` `SHL` `SHR` `SAR` `INC` `DEC` `JMP` `Jcc` `RET`

Addressing: register, immediate, `[base + index*scale + disp]`, RIP-relative.
Branches may be direct or register/memory-indirect, so function pointers and
virtual dispatch translate.

## Known limitations

- **No memory sandbox.** Guest addresses are dereferenced as host pointers. Do
  not feed this untrusted input.
- **No self-modifying-code detection.** `TranslationCache::invalidate()` exists
  but nothing calls it automatically.
- **No memory-ordering model.** x86 is TSO, ARM is weakly ordered. Multithreaded
  guests would need barriers, or Apple Silicon's hardware TSO mode.
- 8- and 16-bit *register writes* are refused (they merge rather than
  zero-extend); byte and halfword memory access is supported.
- Variable-count shifts (`shl reg, cl`) are refused.
- `JP`/`JNP` are refused — AArch64 has no parity flag.

## Building an emulator on this

This is the same pipeline FEX-Emu, box64 and QEMU are built on, with the
structural work already in place:

- **A typed SSA IR with a verifier**, so a malformed translation fails loudly
  instead of producing wrong code.
- **A register allocator** with liveness tracking and pinned guest registers.
- **Compile-time-verified ARM64 encodings**, so the backend can grow without
  silent encoding bugs.
- **A per-flag validity model** — the part most hobby translators get wrong,
  and the reason this one refuses rather than mistranslates.
- **W^X JIT memory** working on Windows, Linux and Apple Silicon.
- **A translation cache and dispatch loop** at block granularity.

Adding an instruction touches four places — decoder mnemonic, IR opcode, lifter
case, backend lowering — plus a test at each layer. The instructions already
implemented are worked examples of every operand form you will need.

What stands between this and running real programs, roughly in order:

| Step | Why it matters |
|---|---|
| `MOVZX`/`MOVSX`, indirect `JMP`/`CALL` | 8/16-bit data, function pointers, vtables |
| Block chaining | patch direct branches between blocks instead of returning to the dispatcher — typically 5–10× on hot loops |
| Lazy flags | materialise NZCV only when a condition consumes it; also how PF and AF become expressible |
| SSE2 | the mandatory baseline for x86-64 — even `memcpy` uses it |
| Loader + syscall translation | ELF/Mach-O/PE, guest `mmap`, signals, TLS |
| Memory ordering | x86 is TSO, ARM is weakly ordered; Apple Silicon exposes a hardware TSO bit for exactly this |

That is a large body of work — FEX is on the order of 500k lines and years of
effort — but none of it requires rethinking the architecture here. Take the
roadmap below and extend outward.

## Roadmap

Done: `MOVZX`/`MOVSX`, register- and memory-indirect branches, block chaining.

Next, in order:

1. **Lazy flags** — materialise NZCV only when a condition consumes it. The
   per-flag validity model is already in place; deferring the computation is
   what turns it into a performance win, and it is also how PF and AF become
   expressible instead of refused.
2. **SSE2** — the mandatory baseline for x86-64.
3. **Loader and syscall translation** — ELF/Mach-O/PE, guest `mmap`, signals.
4. **Memory ordering** — x86 TSO on a weakly-ordered CPU.

## License

MIT. Zydis is MIT, GoogleTest is BSD-3-Clause.
