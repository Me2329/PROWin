#include "dbt/backend/jit_memory.hpp"

#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

#if defined(__APPLE__)
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#endif

// Apple silicon refuses a later mprotect to PROT_EXEC on ordinary anonymous
// memory, so generated code has to go through MAP_JIT. Everywhere else the
// stricter map-RW-then-protect-RX transition is available.
#if defined(__APPLE__) && defined(__aarch64__)
#define DBT_APPLE_JIT 1
#else
#define DBT_APPLE_JIT 0
#endif

namespace dbt::backend {
namespace {

/// Rounds `value` up to the next multiple of `granularity`.
usize round_up(usize value, usize granularity) noexcept {
    if (granularity == 0) {
        return value;
    }
    const usize remainder = value % granularity;
    return (remainder == 0) ? value : value + (granularity - remainder);
}

}  // namespace

usize JitMemory::page_size() noexcept {
#if defined(_WIN32)
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return static_cast<usize>(info.dwPageSize);
#else
    const long size = sysconf(_SC_PAGESIZE);
    return (size > 0) ? static_cast<usize>(size) : usize{4096};
#endif
}

JitMemory JitMemory::allocate(usize bytes, JitError& error) {
    JitMemory memory;
    error = JitError::None;

    if (bytes == 0) {
        error = JitError::InvalidSize;
        return memory;
    }

    const usize capacity = round_up(bytes, page_size());

    // Mapped writable but NOT executable: the W^X transition happens later, in
    // make_executable(). The pages are never both at once.
#if defined(_WIN32)
    void* base = VirtualAlloc(nullptr, capacity, MEM_COMMIT | MEM_RESERVE,
                              PAGE_READWRITE);
#else
    int protection = PROT_READ | PROT_WRITE;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#if DBT_APPLE_JIT
    // MAP_JIT mappings carry RWX; the hardware enforces W^X per thread rather
    // than per page, toggled through pthread_jit_write_protect_np.
    protection |= PROT_EXEC;
    flags |= MAP_JIT;
#endif
    void* base = mmap(nullptr, capacity, protection, flags, -1, 0);
    if (base == MAP_FAILED) {
        base = nullptr;
    }
#endif

    if (base == nullptr) {
        error = JitError::AllocationFailed;
        return memory;
    }

    memory.base_ = base;
    memory.size_ = bytes;
    memory.capacity_ = capacity;
    memory.executable_ = false;
    return memory;
}

JitMemory::~JitMemory() {
    reset();
}

JitMemory::JitMemory(JitMemory&& other) noexcept
    : base_(other.base_),
      size_(other.size_),
      capacity_(other.capacity_),
      executable_(other.executable_) {
    other.base_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    other.executable_ = false;
}

JitMemory& JitMemory::operator=(JitMemory&& other) noexcept {
    if (this != &other) {
        reset();
        base_ = other.base_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        executable_ = other.executable_;
        other.base_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
        other.executable_ = false;
    }
    return *this;
}

void JitMemory::reset() noexcept {
    if (base_ != nullptr) {
#if defined(_WIN32)
        VirtualFree(base_, 0, MEM_RELEASE);
#else
        munmap(base_, capacity_);
#endif
    }
    base_ = nullptr;
    size_ = 0;
    capacity_ = 0;
    executable_ = false;
}

JitError JitMemory::write(usize word_offset, std::span<const Arm64Word> words) {
    if (base_ == nullptr) {
        return JitError::NotWritable;
    }
    if (executable_) {
        return JitError::NotWritable;
    }
    if (words.empty()) {
        return JitError::None;
    }

    // Bounds check in word units. The subtraction form avoids overflowing when
    // word_offset is near the top of the address space, which a plain
    // `offset + size > capacity` comparison would wrap through.
    const usize capacity_words = capacity_ / kArm64InstSize;
    if (word_offset > capacity_words || words.size() > capacity_words - word_offset) {
        return JitError::WriteOutOfRange;
    }

    auto* dest = static_cast<Arm64Word*>(base_) + word_offset;
#if DBT_APPLE_JIT
    // Open the write window only around the copy. The toggle is thread-global
    // across every MAP_JIT mapping, so leaving it open would make already
    // sealed blocks writable as well.
    pthread_jit_write_protect_np(0);
    std::memcpy(dest, words.data(), words.size() * sizeof(Arm64Word));
    pthread_jit_write_protect_np(1);
#else
    std::memcpy(dest, words.data(), words.size() * sizeof(Arm64Word));
#endif
    return JitError::None;
}

JitError JitMemory::make_executable() {
    if (base_ == nullptr) {
        return JitError::ProtectFailed;
    }
    if (executable_) {
        return JitError::AlreadyExecutable;
    }

#if defined(_WIN32)
    DWORD previous = 0;
    if (VirtualProtect(base_, capacity_, PAGE_EXECUTE_READ, &previous) == 0) {
        return JitError::ProtectFailed;
    }
    // AArch64 instruction and data caches are not coherent; without this the
    // core may execute stale bytes.
    FlushInstructionCache(GetCurrentProcess(), base_, capacity_);
#elif DBT_APPLE_JIT
    // The mapping is already execute-capable and writes were closed after each
    // copy, so all that remains is publishing the new instructions to the
    // instruction cache.
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(base_, capacity_);
#else
    if (mprotect(base_, capacity_, PROT_READ | PROT_EXEC) != 0) {
        return JitError::ProtectFailed;
    }
#if defined(__GNUC__) || defined(__clang__)
    auto* start = static_cast<char*>(base_);
    __builtin___clear_cache(start, start + capacity_);
#endif
#endif

    executable_ = true;
    return JitError::None;
}

std::string_view to_string(JitError value) noexcept {
    switch (value) {
    case JitError::None:
        return "none";
    case JitError::InvalidSize:
        return "invalid-size";
    case JitError::AllocationFailed:
        return "allocation-failed";
    case JitError::NotWritable:
        return "not-writable";
    case JitError::WriteOutOfRange:
        return "write-out-of-range";
    case JitError::ProtectFailed:
        return "protect-failed";
    case JitError::AlreadyExecutable:
        return "already-executable";
    }
    return "unknown";
}

}  // namespace dbt::backend
