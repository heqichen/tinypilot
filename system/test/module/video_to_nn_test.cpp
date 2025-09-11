
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


TEST(ALGO_ML_TEST, GivenVideoShouldOutputCorrectVisionResult) {
    const sensor::vision::input::VideoInput::OnVideoFrameCallback onVideoFrameCallback =
      [](const std::uint8_t *data, std::size_t width, std::size_t height) {
          std::printf("got frame %lu x %lu\r\n", width, height);
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
