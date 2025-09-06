#include "algorithm/cl/test/transform_test.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "algorithm/cl/transform.h"

namespace cooboc {
namespace algorithm {
namespace cl {
namespace {

TEST(TransformParameterTest, givenInputShouldCalculateCorrectly) {
    {
        const TransformParameter transParam {makeTransformParameter(640U, 480U, 512U, 256U)};
        EXPECT_FLOAT_EQ(transParam.scale, 1.25F);
        EXPECT_FLOAT_EQ(transParam.offsetX, 0.0F);
        EXPECT_FLOAT_EQ(transParam.offsetY, 80.0F);
    }

    {
        const TransformParameter transParam {makeTransformParameter(1920U, 1080U, 512U, 256U)};
        EXPECT_FLOAT_EQ(transParam.scale, 3.75F);
        EXPECT_FLOAT_EQ(transParam.offsetX, 0.0F);
        EXPECT_FLOAT_EQ(transParam.offsetY, 60.0F);
    }
}

TEST(transformTest, smoke_test) {
    // Setup
    constexpr const std::size_t kInputWdith {640U};
    constexpr const std::size_t kInputHeight {480U};
    constexpr const std::size_t kInputSize {kInputWdith * kInputHeight * 3 / 2};    // NV12

    constexpr const std::size_t kOutputWidth {512U};
    constexpr const std::size_t kOutputHeight {256U};
    constexpr const std::size_t kOutputSize {kOutputWidth * kOutputHeight * 3 / 2};    // NV12


    std::array<std::uint8_t, kInputSize> inputBuffer;
    for (std::size_t x = 0; x < kInputWdith; ++x) {
        for (std::size_t y = 0; y < kInputHeight; ++y) {
            // inputBuffer[y * kInputWdith + x] = ((x + y) % 256);
            inputBuffer[y * kInputWdith + x] = x % 4 + y % 2;
        }
    }

    std::array<std::uint8_t, kOutputSize> outputBuffer {};
    // Run
    ASSERT_NO_THROW(transform(inputBuffer.data(), kInputWdith, kInputHeight, outputBuffer.data()));

    // Check
    const std::vector<std::uint8_t> expectedOutput10Row {1, 2, 3, 4, 2, 3, 4, 1, 3, 4};
    std::vector<std::uint8_t> actualOutput;
    for (std::size_t x = 0; x < 10U; ++x) {
        actualOutput.push_back(outputBuffer[10 * kInputWdith + x]);
    }
    EXPECT_EQ(actualOutput, expectedOutput10Row);
}

TEST(transformTest, givenFrameShouldTransformCorrectly) {
    // Setup
    constexpr const std::size_t kInputWdith = 8U;
    constexpr const std::size_t kInputHeight = 2U;
    const std::array<std::uint8_t, (kInputWdith * kInputHeight) * 3U / 2U> inputFrame    //
      {100U, 101U, 102U, 103U, 104U, 105U, 106U, 107U,                                   // Y
       108U, 109U, 110U, 111U, 112U, 113U, 114U, 115U,                                   // Y
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
    transform(inputFrame.data(), kInputWdith, kInputHeight, outputBuffer.data());

    // Check
    std::uint8_t lastValue {0U};

    // Evaluate Y-Plate
    for (std::size_t y {0U}; y < kOutputHeight; ++y) {
        lastValue = 0U;
        for (std::size_t x {0U}; x < kOutputWidth; ++x) {
            const std::size_t index = y * kOutputWidth + x;
            // Y-Plate range from 100 to 115
            EXPECT_GE(outputBuffer[index], 100U);
            EXPECT_LE(outputBuffer[index], 115U);
            // In increase
            EXPECT_GE(outputBuffer[index], lastValue);
            lastValue = outputBuffer[index];
        }
    }

    // Evaluate U-Plate
    for (std::size_t y {0U}; y < (kOutputHeight / 2); ++y) {
        lastValue = 0U;
        for (std::size_t x {0U}; x < (kOutputWidth / 2); ++x) {
            const std::size_t index = kOutputYSize + y * (kOutputWidth / 2) + x;
            // U-Plate range from 128 to 131
            EXPECT_GE(outputBuffer[index], 128U);
            EXPECT_LE(outputBuffer[index], 131U);
            // In increase
            EXPECT_GE(outputBuffer[index], lastValue);
            lastValue = outputBuffer[index];
        }
    }

    // Evaluate V-Plate
    for (std::size_t y {0U}; y < (kOutputHeight / 2); ++y) {
        lastValue = 0U;
        for (std::size_t x {0U}; x < (kOutputWidth / 2); ++x) {
            const std::size_t index = kOutputYSize + kOutputUVSize + y * (kOutputWidth / 2) + x;
            // V-Plate range from 132 to 135
            EXPECT_GE(outputBuffer[index], 132U);
            EXPECT_LE(outputBuffer[index], 135U);
            // In increase
            EXPECT_GE(outputBuffer[index], lastValue);
            lastValue = outputBuffer[index];
        }
    }


    // for (std::size_t y {0U}; y < (kOutputHeight * 3 / 2); ++y) {
    //     for (std::size_t x {0U}; x < kOutputWidth; ++x) {
    //         std::printf("%3u ", outputBuffer[y * kOutputWidth + x]);
    //     }
    //     std::printf("\r\n");
    // }
}


}    // namespace
}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc