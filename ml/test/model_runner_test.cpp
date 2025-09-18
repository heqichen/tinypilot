#include "ml/model_runner.h"
#include <gtest/gtest.h>
#include "perception/vision/test/prepare_test.h"

namespace cooboc {
namespace ml {
namespace {
TEST(ModelRunnerTest, smoke_test) {
    // Setup
    const char* tfliteFilepath = "/home/heqichen/workspace/tinypilot/mlutils/modelgen/models_tflite/"
                                 "test_model_pb/driving_vision_fp32_float32.tflite";
    // Run
    ModelRunner runner(tfliteFilepath);
    // ASSERT_NO_THROW();
}

}    // namespace
}    // namespace ml
}    // namespace cooboc