#include "perception/vision/test/prepare_test.h"
#include <gtest/gtest.h>
#include "perception/vision/prepare.h"

namespace cooboc {
namespace perception {
namespace vision {
namespace {

TEST(PrepareTest, smoke_test) {
    // Setup
    constexpr const std::size_t kInputWdith {640U};
    constexpr const std::size_t kInputHeight {480U};
    constexpr const std::size_t kInputSize {kInputWdith * kInputHeight * 3 / 2};    // NV12

    constexpr const std::size_t kOutputWidth {512U};
    constexpr const std::size_t kOutputHeight {256U};
    constexpr const std::size_t kOutputYSize {kOutputWidth * kOutputHeight};
    constexpr const std::size_t kOutputUVSize {kOutputWidth * kOutputHeight / 4U};
    constexpr const std::size_t kOutputSize {kOutputYSize + 2 * kOutputUVSize};    // NV12

    std::array<std::uint8_t, kInputSize> inputBuffer;
    for (std::size_t x = 0; x < kInputWdith; ++x) {
        for (std::size_t y = 0; y < kInputHeight; ++y) {
            // Y
            inputBuffer[y * kInputWdith + x] = (x % 16) + 100U;
            // U
            if ((x % 2 == 0) && (y % 2 == 0)) {
                inputBuffer[kInputWdith * kInputHeight + (y / 2) * (kInputWdith / 2) + (x / 2)] = (x % 4) + 128U;
            }
            // V
            if ((x % 2 == 0) && (y % 2 == 0)) {
                inputBuffer[kInputWdith * kInputHeight + (kInputWdith * kInputHeight / 4U) +
                            (y / 2) * (kInputWdith / 2) + (x / 2)] = (x % 4) + 132U;
            }
        }
    }

    std::array<std::uint8_t, kOutputSize> outputBuffer {};
    // Run
    ASSERT_NO_THROW(prepare(inputBuffer.data(), kInputWdith, kInputHeight, outputBuffer.data()));
}

}    // namespace
}    // namespace vision
}    // namespace perception
}    // namespace cooboc