
#include "system/test/module/video_to_nn_test.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "perception/vision/ml/model_runner.h"
#include "perception/vision/prepare.h"
#include "sensor/vision/input/video_input.h"


namespace cooboc {
namespace module_test {
namespace {

static std::size_t count {0U};

void writeAllToFile(const std::array<std::uint8_t, 12 * 128 * 256> &imgs,
                    const std::array<std::uint8_t, 12 * 128 * 256> &bigImgs,
                    const std::array<float, 632U> &output) {
    std::string inputImgsFilename = "output/imgs" + std::to_string(count) + ".csv";
    std::string inputBigImgsFilename = "output/bigImgs" + std::to_string(count) + ".csv";
    std::string outputFilename = "output/output" + std::to_string(count) + ".csv";

    std::ofstream inputImgsFile(inputImgsFilename);
    std::ofstream inputBigImgsFile(inputBigImgsFilename);
    std::ofstream outputFile(outputFilename);

    for (std::size_t i {0U}; i < 12U * 128U * 256U; ++i) {
        inputImgsFile << static_cast<unsigned int>(imgs[i]) << std::endl;
    }

    for (std::size_t i {0U}; i < 12U * 128U * 256U; ++i) {
        inputBigImgsFile << static_cast<unsigned int>(bigImgs[i]) << std::endl;
    }

    for (std::size_t i {0U}; i < 632U; ++i) {
        outputFile << output[i] << std::endl;
    }
    count++;
}

TEST(MODULE_TEST, GivenVideoShouldOutputCorrectVisionResult) {
    constexpr const char *kVideoPath = "assets/driving_clip.mp4";
    const sensor::vision::input::VideoInput::OnVideoFrameCallback onVideoFrameCallback =
      [](const std::uint8_t *frameData, std::size_t width, std::size_t height) {
          std::printf("got frame %lu x %lu\r\n", width, height);
          std::array<float, 632U> outputBuffer {};


          std::array<std::uint8_t, 128 * 256 * 6> imageData {};
          perception::vision::prepare(frameData, width, height, imageData.data());

          std::array<std::uint8_t, 12 * 128 * 256> teleImages;
          std::array<std::uint8_t, 12 * 128 * 256> wideImages;

          memcpy(teleImages.data(), imageData.data(), 6 * 128 * 256);
          memcpy(teleImages.data() + 6 * 128 * 256, imageData.data(), 6 * 128 * 256);
          memcpy(wideImages.data(), imageData.data(), 6 * 128 * 256);
          memcpy(wideImages.data() + 6 * 128 * 256, imageData.data(), 6 * 128 * 256);

          perception::vision::ml::run(teleImages, wideImages, outputBuffer);

          for (int i = 0; i < 632; ++i) {
              std::printf("%f ", outputBuffer[i]);
          }
          std::printf("\r\n");

          writeAllToFile(teleImages, wideImages, outputBuffer);
      };

    // main procedure

    // Setup
    // Register callback

    sensor::vision::input::VideoInput videoInput(kVideoPath);
    videoInput.registerCallback(onVideoFrameCallback);


    // Run

    while (!videoInput.end()) {
        videoInput.tick();
    }
}
}    // namespace
}    // namespace module_test
}    // namespace cooboc
