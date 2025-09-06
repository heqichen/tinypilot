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
    for (std::size_t i {0U}; i < kInputSize; ++i) {
        inputBuffer[i] = 0U;
    }

    std::array<std::uint8_t, kOutputSize> outputBuffer {};
    // Run
    ASSERT_NO_THROW(prepare(inputBuffer.data(), kInputWdith, kInputHeight, outputBuffer.data()));
}

TEST(PrepareTest, givienInputImageShouldOutputCorrectly) {
    // Setup
    constexpr const std::size_t kInputWdith = 8U;
    constexpr const std::size_t kInputHeight = 2U;
    const std::array<std::uint8_t, (kInputWdith * kInputHeight) * 3U / 2U> inputFrame    //
      {101U, 103U, 105U, 107U, 109U, 111U, 113U, 115U,                                   // Y
       102U, 104U, 106U, 108U, 110U, 112U, 114U, 116U,                                   // Y
       128U, 129U, 130U, 131U,                                                           // U
       132U, 133U, 134U, 135U};                                                          // V

    constexpr const std::size_t kOutputSize = 512U * 256U * 3U / 2U;
    std::array<std::uint8_t, kOutputSize> outputBuffer {};

    constexpr const std::size_t kOutputWidth {512U};
    constexpr const std::size_t kOutputHeight {256U};
    constexpr const std::size_t kOutputYSize = kOutputWidth * kOutputHeight;
    constexpr const std::size_t kOutputUVSize = (kOutputWidth / 2) * (kOutputHeight / 2);
    constexpr const std::size_t kOutputFrameSize = kOutputYSize + (kOutputUVSize * 2);

    // Run
    prepare(inputFrame.data(), kInputWdith, kInputHeight, outputBuffer.data());

    // Check
    std::uint8_t lastValue {0U};

    auto check = [&outputBuffer](std::size_t channel, std::uint8_t minv, std::uint8_t maxv) {
        bool hasEven {false};
        bool hasOdd {false};
        const std::size_t w = kOutputWidth / 2;
        const std::size_t h = kOutputHeight / 2;
        const std::size_t offset = channel * w * h;
        for (std::size_t y {0U}; y < kOutputHeight / 2; ++y) {
            for (std::size_t x {0U}; x < kOutputWidth / 2; ++x) {
                const std::uint8_t& value = outputBuffer[offset + y * w + x];
                ASSERT_GT(value, minv);
                ASSERT_LT(value, maxv);
                if (value % 2 == 0) {
                    hasOdd = true;
                } else {
                    hasEven = true;
                }
            }
        }
        EXPECT_TRUE(hasOdd && hasEven);
    };

    check(0U, 101U, 116U);
    check(1U, 101U, 116U);
    check(2U, 101U, 116U);
    check(3U, 101U, 116U);
    check(4U, 128U, 131U);
    check(5U, 132U, 135U);

    // for (std::size_t y {0U}; y < (kOutputHeight * 3 / 2); ++y) {
    //     for (std::size_t x {0U}; x < kOutputWidth; ++x) {
    //         std::printf("%3u ", outputBuffer[y * kOutputWidth + x]);
    //     }
    //     std::printf("\r\n");
    // }
}

}    // namespace
}    // namespace vision
}    // namespace perception
}    // namespace cooboc