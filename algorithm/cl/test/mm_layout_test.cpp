#include "algorithm/cl/mm_layout.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace cooboc {
namespace algorithm {
namespace cl {
namespace {


TEST(mm_layout, smoke_test) {
    // Setup

    constexpr const std::size_t kWidth {8U};
    constexpr const std::size_t kHeight {2U};
    constexpr const std::size_t kYSize {kWidth * kHeight};        // Y
    constexpr const std::size_t kUSize {kWidth * kHeight / 4};    // U
    constexpr const std::size_t kVSize {kWidth * kHeight / 4};    // V

    const std::vector<std::uint8_t> inputBuffer {
      1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,    // Y
      2U, 2U, 2U, 2U, 2U, 2U, 2U, 2U,                                    // U
      3U, 3U, 3U, 3U, 3U, 3U, 3U, 3U                                     // V
    };

    std::vector<std::uint8_t> outputBuffer {};
    outputBuffer.reserve(kYSize + kUSize + kVSize);

    ASSERT_EQ(inputBuffer.size(), outputBuffer.size());

    // Run
    reorderImageLayout(inputBuffer.data(), kWidth, kHeight, outputBuffer.data());

    // Check
}

}    // namespace
}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc