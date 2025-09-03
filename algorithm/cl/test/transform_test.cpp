#include "algorithm/cl/transform.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace cooboc {
namespace algorithm {
namespace cl {
namespace {

TEST(TransformParameterTest, givenInputShouldCalculateCorrectly) {
    {
        constexpr const TransformParameter transParam {makeTransformParameter(640U, 480U, 512U, 256U)};
        EXPECT_FLOAT_EQ(transParam.scale, 1.25F);
        EXPECT_FLOAT_EQ(transParam.offsetX, 0.0F);
        EXPECT_FLOAT_EQ(transParam.offsetY, 80.0F);
    }

    {
        constexpr const TransformParameter transParam {makeTransformParameter(1920U, 1080U, 512U, 256U)};
        EXPECT_FLOAT_EQ(transParam.scale, 3.75F);
        EXPECT_FLOAT_EQ(transParam.offsetX, 0.0F);
        EXPECT_FLOAT_EQ(transParam.offsetY, 60.0F);
    }
}

TEST(transform, smoke_test) {
    // Setup
    constexpr const std::size_t kInputWdith {640U};
    constexpr const std::size_t kInputHeight {480U};
    constexpr const std::size_t kInputSize {kInputWdith * kInputHeight * 3 / 2};    // NV12

    constexpr const std::size_t kOutputWidth {512U};
    constexpr const std::size_t kOutputHeight {256U};
    constexpr const std::size_t kOutputSize {kOutputWidth * kOutputHeight * 3 / 2};    // NV12


    constexpr const TransformParameter transParam {
      makeTransformParameter(kInputWdith, kInputHeight, kOutputWidth, kOutputHeight)};

    std::array<std::uint8_t, kInputSize> inputBuffer;
    for (std::size_t x = 0; x < kInputWdith; ++x) {
        for (std::size_t y = 0; y < kInputHeight; ++y) {
            // inputBuffer[y * kInputWdith + x] = ((x + y) % 256);
            inputBuffer[y * kInputWdith + x] = x % 4 + y % 2;
        }
    }

    std::array<std::uint8_t, kOutputSize> outputBuffer {};
    // Run
    transform(inputBuffer.data(), kInputWdith, kInputHeight, transParam, outputBuffer.data());

    // Check
    const std::vector<std::uint8_t> expectedOutput10Row {1, 2, 3, 4, 2, 3, 4, 1, 3, 4};
    std::vector<std::uint8_t> actualOutput;
    for (std::size_t x = 0; x < 10U; ++x) {
        actualOutput.push_back(outputBuffer[10 * kInputWdith + x]);
    }
    EXPECT_EQ(actualOutput, expectedOutput10Row);


    // for (auto i : actualOutput) {
    //     std::printf("%d ", i);
    // }
    // std::printf("\r\n");
    // for (std::size_t y = 0; y < 10; ++y) {
    //     for (std::size_t x = 0; x < 10; ++x) {
    //         std::printf("%d ", outputBuffer[y * kOutputWidth + x]);
    //     }
    //     std::printf("\r\n");
    // }
}

}    // namespace
}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc