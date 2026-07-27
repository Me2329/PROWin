#pragma once

#include <cstring>
#include <span>
#include <string_view>
#include <type_traits>

#include "dbt/common/types.hpp"

namespace dbt::backend {

enum class JitError : u8 {
    None = 0,
    /// A zero-byte allocation was requested.
    InvalidSize,
    /// The operating system refused the mapping.
    AllocationFailed,
    /// Writes are only allowed before the buffer is made executable.
    NotWritable,
    /// The write would fall outside the allocation.
    WriteOutOfRange,
    /// The permission change was rejected.
    ProtectFailed,
    /// The buffer has already been sealed.
    AlreadyExecutable,
};

[[nodiscard]] std::string_view to_string(JitError value) noexcept;

/// A page-aligned buffer for JIT-compiled ARM64 code.
///
/// The buffer follows a strict W^X discipline: it is mapped read/write, filled
/// in, then flipped to read/execute. It is never simultaneously writable and
/// executable, and once sealed it cannot be made writable again -- a fresh
/// allocation is required. `make_executable` also flushes the instruction
/// cache, which is mandatory on AArch64 because its instruction and data caches
/// are not coherent.
///
/// Apple silicon is the exception. There the kernel refuses to add PROT_EXEC to
/// an ordinary anonymous mapping, so the buffer is created with MAP_JIT and the
/// write/execute split is enforced per thread by pthread_jit_write_protect_np
/// instead of per page. The write window is held open only around each copy,
/// but two consequences are worth knowing: the pages are execute-capable from
/// allocation, so is_executable() reports this class's contract rather than a
/// hardware guarantee, and the toggle is thread-global across every MAP_JIT
/// mapping in the process.
///
/// Move-only; the mapping is released in the destructor.
class JitMemory {
public:
    JitMemory() = default;
    ~JitMemory();

    JitMemory(const JitMemory&) = delete;
    JitMemory& operator=(const JitMemory&) = delete;
    JitMemory(JitMemory&& other) noexcept;
    JitMemory& operator=(JitMemory&& other) noexcept;

    /// Reserves at least `bytes` of read/write memory, rounded up to a page.
    /// On failure the returned object is invalid and `error` says why.
    [[nodiscard]] static JitMemory allocate(usize bytes, JitError& error);

    [[nodiscard]] bool valid() const noexcept { return base_ != nullptr; }
    /// Bytes the caller asked for.
    [[nodiscard]] usize size() const noexcept { return size_; }
    /// Bytes actually mapped, rounded up to the page size.
    [[nodiscard]] usize capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool is_executable() const noexcept { return executable_; }

    /// Copies `words` in at `word_offset`. Every write is bounds-checked
    /// against the allocation, so an offset or length past the end is rejected
    /// rather than corrupting adjacent memory.
    [[nodiscard]] JitError write(usize word_offset, std::span<const Arm64Word> words);

    /// Transitions the mapping from read/write to read/execute and flushes the
    /// instruction cache. Idempotent calls are reported as AlreadyExecutable.
    [[nodiscard]] JitError make_executable();

    /// Base address. Only safe to call through once is_executable() is true.
    [[nodiscard]] const void* code() const noexcept { return base_; }

    /// Base address as a function pointer of type `Fn`.
    ///
    /// Casting an object pointer to a function pointer is only conditionally
    /// supported by the standard, so this copies the bits rather than using
    /// reinterpret_cast, which keeps the build clean under -Wpedantic.
    template <typename Fn>
    [[nodiscard]] Fn entry() const noexcept {
        static_assert(std::is_pointer_v<Fn>, "entry() requires a pointer type");
        static_assert(sizeof(Fn) == sizeof(void*));
        Fn fn = nullptr;
        std::memcpy(&fn, &base_, sizeof(fn));
        return fn;
    }

    /// Releases the mapping and returns the object to the empty state.
    void reset() noexcept;

    /// Page size of the host, in bytes.
    [[nodiscard]] static usize page_size() noexcept;

private:
    void* base_ = nullptr;
    usize size_ = 0;
    usize capacity_ = 0;
    bool executable_ = false;
};

}  // namespace dbt::backend
