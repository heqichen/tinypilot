#include "algorithm/cl/mm_layout.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <vector>


namespace cooboc {
namespace algorithm {
namespace cl {
namespace {


TEST(mm_layout, smoke_test) {
    // Setup
    constexpr const std::size_t kWidth {16U};
    constexpr const std::size_t kHeight {2U};
    constexpr const std::size_t kYSize {kWidth * kHeight};        // Y
    constexpr const std::size_t kUSize {kWidth * kHeight / 4};    // U
    constexpr const std::size_t kVSize {kWidth * kHeight / 4};    // V

    // 0 2
    // 1 3
    const std::vector<std::uint8_t> inputBuffer {
      1U,  2U,  1U,  2U,  1U,  2U,  1U,  2U, 3U, 4U, 3U, 4U, 3U, 4U, 3U, 4U,    // Y
      5U,  6U,  5U,  6U,  5U,  6U,  5U,  6U, 7U, 8U, 7U, 8U, 7U, 8U, 7U, 8U,    // Y
      9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,                                    // U
      10U, 10U, 10U, 10U, 10U, 10U, 10U, 10U                                    // V
    };

    const std::vector<std::uint8_t> expectedOutputBuffer {
      1U,  1U,  1U,  1U,  3U,  3U,  3U,  3U,    // top left
      5U,  5U,  5U,  5U,  7U,  7U,  7U,  7U,    // top right
      2U,  2U,  2U,  2U,  4U,  4U,  4U,  4U,    // bottom left
      6U,  6U,  6U,  6U,  8U,  8U,  8U,  8U,    // bottom right
      9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,    // U
      10U, 10U, 10U, 10U, 10U, 10U, 10U, 10U    // V
    };

    std::vector<std::uint8_t> outputBuffer(kYSize + kUSize + kVSize);

    // Make sure test are correct
    ASSERT_EQ(kWidth % 16, 0);    // make sure the width is multiple of 16, so that cl can use command to sppedup
    ASSERT_EQ(kHeight % 2, 0);    // make sure the height is even number
    ASSERT_EQ(inputBuffer.size(), outputBuffer.size());
    ASSERT_EQ(inputBuffer.size(), expectedOutputBuffer.size());

    // Run
    reorderImageLayout(inputBuffer.data(), kWidth, kHeight, outputBuffer.data());

    // Check
    EXPECT_EQ(outputBuffer, expectedOutputBuffer);
}

}    // namespace
}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc
