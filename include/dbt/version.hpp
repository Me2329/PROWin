#pragma once

#include <string_view>

#include "dbt/common/types.hpp"

namespace dbt {

inline constexpr u32 kVersionMajor = 0;
inline constexpr u32 kVersionMinor = 1;
inline constexpr u32 kVersionPatch = 0;

/// Human-readable "major.minor.patch" identifier for this build.
[[nodiscard]] std::string_view version_string() noexcept;

/// True when the running host can execute the AArch64 code this library emits.
/// On other hosts the backend still compiles and is verified by encoding tests,
/// but the produced buffers must not be called.
[[nodiscard]] constexpr bool host_can_execute_arm64() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    return true;
#else
    return false;
#endif
}

}  // namespace dbt
