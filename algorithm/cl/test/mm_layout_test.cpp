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

    const std::vector<std::uint8_t> inputBuffer {
      1U, 2U, 1U, 2U, 1U, 2U, 1U, 2U, 1U, 2U, 1U, 2U, 1U, 2U, 1U, 2U,    // Y
      5U, 6U, 5U, 6U, 5U, 6U, 5U, 6U, 5U, 6U, 5U, 6U, 5U, 6U, 5U, 6U,    // Y
      7U, 7U, 7U, 7U, 7U, 7U, 7U, 7U,                                    // U
      8U, 8U, 8U, 8U, 8U, 8U, 8U, 8U                                     // V
    };

    const std::vector<std::uint8_t> expectedOutputBuffer {
      1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,    // top left
      2U, 2U, 2U, 2U, 2U, 2U, 2U, 2U,    // top right
      5U, 5U, 5U, 5U, 5U, 5U, 5U, 5U,    // bottom left
      6U, 6U, 6U, 6U, 6U, 6U, 6U, 6U,    // bottom right
      7U, 7U, 7U, 7U, 7U, 7U, 7U, 7U,    // U
      8U, 8U, 8U, 8U, 8U, 8U, 8U, 8U     // V
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

    for (const auto i : outputBuffer) {
        std::printf("%u ", i);
    }
    std::printf("\r\n");
}

}    // namespace
}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc
