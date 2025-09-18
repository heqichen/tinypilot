#include "ml/vision_runner.h"
#include <gtest/gtest.h>
#include <cstdint>
#include "datadefs/ml/vision_output.h"
#include "perception/vision/test/prepare_test.h"

#ifndef VISION_TFLITE_FILE_PATH
#define VISION_TFLITE_FILE_PATH "assets/driving_vision_fp32_float32.tflite"
#endif

namespace cooboc {
namespace ml {
namespace {
TEST(ModelRunnerTest, smoke_test) {
    // Setup

    VisionRunner visionRunner(VISION_TFLITE_FILE_PATH);

    std::array<std::uint8_t, 12 * 128 * 256> images = {};
    std::array<std::uint8_t, 12 * 128 * 256> bigImages = {};
    images.fill(0);
    bigImages.fill(0);

    datadef::ml::VisionOutput output = {};


    // Run
    ASSERT_NO_THROW(visionRunner.run(images, bigImages, output));


    EXPECT_NEAR(output.meta[0], -5.4686141014099121F, 1e-2F);
    EXPECT_NEAR(output.desire_pred[0], 1.9410521984100342F, 1e-2F);
    EXPECT_NEAR(output.pose[0], 01.9842386245727539, 1e-2F);
    EXPECT_NEAR(output.wide_from_device_euler[0], -0.00044107018038630486F, 1e-2F);
    EXPECT_NEAR(output.road_transform[0], -4.5066699385643005e-05F, 1e-2F);
    EXPECT_NEAR(output.hidden_state[0], -0.0062733613885939121F, 1e-2F);
    EXPECT_NEAR(output.pad[0], 0.0F, 1e-2F);


    // EXPECT_NEAR(output.meta[0], 1.0f, 1e-5f);
    // for (std::size_t i {0U}; i < 632U; ++i) {
    //     std::printf("%f ", ((float *)(&output))[i]);
    // }
    // std::printf("\r\n");
}

}    // namespace
}    // namespace ml
}    // namespace cooboc