
#include "system/test/module/video_to_nn_test.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "perception/vision/ml/model_runner.h"
#include "perception/vision/prepare.h"
#include "sensor/vision/input/video_input.h"

namespace cooboc {
namespace module_test {
namespace {


TEST(MODULE_TEST, GivenVideoShouldOutputCorrectVisionResult) {
    const sensor::vision::input::VideoInput::OnVideoFrameCallback onVideoFrameCallback =
      [](const std::uint8_t *frameData, std::size_t width, std::size_t height) {
          std::printf("got frame %lu x %lu\r\n", width, height);


          std::array<std::uint8_t, 128 * 256 * 6> imageData {};
          perception::vision::prepare(frameData, width, height, imageData.data());

          std::array<std::uint8_t, 12 * 128 * 256> teleImages;
          std::array<std::uint8_t, 12 * 128 * 256> wideImages;

          memcpy(teleImages.data(), imageData.data(), 6 * 128 * 256);
          memcpy(teleImages.data() + 6 * 128 * 256, imageData.data(), 6 * 128 * 256);
          memcpy(wideImages.data(), imageData.data(), 6 * 128 * 256);
          memcpy(wideImages.data() + 6 * 128 * 256, imageData.data(), 6 * 128 * 256);

          perception::vision::ml::run(teleImages, wideImages);
      };

    // main procedure

    // Setup
    // Register callback

    sensor::vision::input::VideoInput videoInput;
    videoInput.registerCallback(onVideoFrameCallback);


    // Run

    while (!videoInput.end()) {
        videoInput.tick();
    }
}
}    // namespace
}    // namespace module_test
}    // namespace cooboc
