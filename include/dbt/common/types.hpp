#pragma once

#include <cstddef>
#include <cstdint>

namespace dbt {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using usize = std::size_t;

/// An address in the guest (x86-64) address space.
using GuestAddr = u64;

/// A single AArch64 instruction word.
using Arm64Word = u32;

/// Every A64 instruction is exactly four bytes; the backend relies on this when
/// computing branch displacements.
inline constexpr usize kArm64InstSize = 4;

/// The longest legal x86-64 instruction. Anything longer is malformed, and the
/// decoder uses this to bound how far it will ever read past a start offset.
inline constexpr usize kMaxX86InstLength = 15;

}  // namespace dbt
