#include "system/test/module/vision_smoke_test.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "perception/vision/prepare.h"


namespace cooboc {
namespace module_test {
namespace {

TEST(ALGO_ML_TEST, smoke_test) {
    // Setup
    // Make a video frame
    constexpr const std::size_t kVideoWidth {1920U};
    constexpr const std::size_t kVideoHeight {1080U};
    constexpr const std::size_t kVideoYSize {kVideoWidth * kVideoHeight};            // Y
    constexpr const std::size_t kVideoUVSize {kVideoWidth * kVideoHeight / 4U};      // U, V
    constexpr const std::size_t kVideoFrameSize {kVideoYSize + kVideoUVSize * 2};    // NV12

    std::array<std::uint8_t, kVideoFrameSize> videoFrame {};

    // Y plane
    for (std::size_t y {0U}; y < kVideoHeight; ++y) {
        for (std::size_t x {0U}; x < kVideoWidth; ++x) {
            videoFrame[(y * kVideoWidth) + x] = (x + y) % 128U;
        }
    }
    for (std::size_t y {0U}; y < kVideoHeight / 2U; ++y) {
        for (std::size_t x {0U}; x < kVideoWidth / 2U; ++x) {
            // U plane
            videoFrame[kVideoYSize + (y * kVideoWidth / 2) + x] = ((x + y) % 64U) + 128U;                   // U
            videoFrame[kVideoYSize + kVideoUVSize + (y * kVideoWidth / 2) + x] = ((x + y) % 64U) + 192U;    // V
        }
    }

    // RUN
    // Prepare
    // TODO move to cl later
    std::array<std::uint8_t, 128 * 256 * 6> imageData {};
    perception::vision::prepare(videoFrame.data(), kVideoWidth, kVideoHeight, imageData.data());
}

}    // namespace
}    // namespace module_test
}    // namespace cooboc
