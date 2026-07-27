#pragma once

#include <memory>
#include <unordered_map>
#include <utility>

#include "dbt/backend/jit_memory.hpp"
#include "dbt/common/types.hpp"
#include "dbt/runtime/cpu_state.hpp"

namespace dbt::runtime {

/// One compiled guest block together with the executable memory holding it.
///
/// Move-only, because it owns a JitMemory mapping.
class TranslatedBlock {
public:
    TranslatedBlock() = default;
    TranslatedBlock(GuestAddr guest_addr, backend::JitMemory memory, usize word_count)
        : memory_(std::move(memory)),
          guest_addr_(guest_addr),
          word_count_(word_count) {}

    TranslatedBlock(const TranslatedBlock&) = delete;
    TranslatedBlock& operator=(const TranslatedBlock&) = delete;
    TranslatedBlock(TranslatedBlock&&) noexcept = default;
    TranslatedBlock& operator=(TranslatedBlock&&) noexcept = default;

    [[nodiscard]] GuestAddr guest_addr() const noexcept { return guest_addr_; }
    [[nodiscard]] usize word_count() const noexcept { return word_count_; }
    [[nodiscard]] const void* code() const noexcept { return memory_.code(); }
    [[nodiscard]] bool executable() const noexcept { return memory_.is_executable(); }

    /// The block's entry point. Only safe to call on an ARM64 host, and only
    /// once executable() is true.
    [[nodiscard]] BlockFn entry() const noexcept { return memory_.entry<BlockFn>(); }

private:
    backend::JitMemory memory_;
    GuestAddr guest_addr_ = 0;
    usize word_count_ = 0;
};

/// Maps a guest address to the block compiled for it.
///
/// Blocks are held behind unique_ptr so that a rehash never invalidates a
/// pointer the dispatcher is still holding.
class TranslationCache {
public:
    [[nodiscard]] const TranslatedBlock* find(GuestAddr addr) const {
        const auto it = blocks_.find(addr);
        if (it == blocks_.end()) {
            ++misses_;
            return nullptr;
        }
        ++hits_;
        return it->second.get();
    }

    /// Stores `block`, replacing any previous entry for the same address.
    const TranslatedBlock* insert(GuestAddr addr, TranslatedBlock block) {
        auto owned = std::make_unique<TranslatedBlock>(std::move(block));
        TranslatedBlock* raw = owned.get();
        blocks_[addr] = std::move(owned);
        return raw;
    }

    /// Drops the block for `addr`, if any. Returns true when one was removed.
    bool invalidate(GuestAddr addr) { return blocks_.erase(addr) != 0; }

    void clear() noexcept { blocks_.clear(); }

    [[nodiscard]] usize size() const noexcept { return blocks_.size(); }
    [[nodiscard]] bool empty() const noexcept { return blocks_.empty(); }
    [[nodiscard]] usize hits() const noexcept { return hits_; }
    [[nodiscard]] usize misses() const noexcept { return misses_; }
    void reset_stats() noexcept {
        hits_ = 0;
        misses_ = 0;
    }

private:
    std::unordered_map<GuestAddr, std::unique_ptr<TranslatedBlock>> blocks_;
    mutable usize hits_ = 0;
    mutable usize misses_ = 0;
};

}  // namespace dbt::runtime
