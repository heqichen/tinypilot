#include "ml/policy_runner.h"
#include <gtest/gtest.h>
#include <cstdint>
#include "datadefs/ml/policy_output.h"
#include "perception/vision/test/prepare_test.h"

#ifndef POLICY_TFLITE_FILE_PATH
#define POLICY_TFLITE_FILE_PATH "assets/driving_policy_fp32_float32.tflite"
#endif

namespace cooboc {
namespace ml {
namespace {
TEST(PolicyRunnerTest, smoke_test) {
    // Setup

    PolicyRunner policyRunner(POLICY_TFLITE_FILE_PATH);
    datadef::ml::PolicyInput input = {};
    memset(&input, 0, sizeof(input));
    datadef::ml::PolicyOutput output = {};

    ASSERT_EQ(sizeof(datadef::ml::PolicyOutput), 5884 * sizeof(float));


    // Run
    ASSERT_NO_THROW(policyRunner.run(input, output));

    // CHECK
    EXPECT_NEAR(output.plan[0], 0.0F, 1e-2F);
    EXPECT_NEAR(output.laneLines[0], 0.0F, 1e-2F);
    EXPECT_NEAR(output.laneLinesProb[0], 0.0F, 1e-2F);
    EXPECT_NEAR(output.roadEdges[0], 0.0F, 1e-2F);
    EXPECT_NEAR(output.lead[0], 0.0F, 1e-2F);
    EXPECT_NEAR(output.leadProb[0], 0.0F, 1e-2F);
    EXPECT_NEAR(output.desireState[0], 0.0F, 1e-2F);
    EXPECT_NEAR(output.simPose[0], 0.0F, 1e-2F);
    EXPECT_NEAR(output.desiredCurvature[0], 0.0F, 1e-2F);
    EXPECT_NEAR(output.pad[0], 0.0F, 1e-2F);
}

}    // namespace
}    // namespace ml
}    // namespace cooboc