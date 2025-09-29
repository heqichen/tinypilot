#include "ml/policy_runner.h"
#include <gtest/gtest.h>
#include <armnn/IRuntime.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <cstdint>

namespace cooboc {
namespace ml {
namespace {
// TEST(PolicyRunnerTest, smoke_test) {
//     // Setup

//     PolicyRunner policyRunner(POLICY_TFLITE_FILE_PATH);
//     datadef::ml::PolicyInput input = {};
//     memset(&input, 0, sizeof(input));
//     datadef::ml::PolicyOutput output = {};

//     ASSERT_EQ(sizeof(datadef::ml::PolicyOutput), 5884 * sizeof(float));


//     // Run
//     ASSERT_NO_THROW(policyRunner.run(input, output));

//     // CHECK
//     EXPECT_NEAR(output.plan[0], 0.0F, 1e-2F);
//     EXPECT_NEAR(output.laneLines[0], 0.0F, 1e-2F);
//     EXPECT_NEAR(output.laneLinesProb[0], 0.0F, 1e-2F);
//     EXPECT_NEAR(output.roadEdges[0], 0.0F, 1e-2F);
//     EXPECT_NEAR(output.lead[0], 0.0F, 1e-2F);
//     EXPECT_NEAR(output.leadProb[0], 0.0F, 1e-2F);
//     EXPECT_NEAR(output.desireState[0], 0.0F, 1e-2F);
//     EXPECT_NEAR(output.simPose[0], 0.0F, 1e-2F);
//     EXPECT_NEAR(output.desiredCurvature[0], 0.0F, 1e-2F);
//     EXPECT_NEAR(output.pad[0], 0.0F, 1e-2F);
// }

TEST(WWW, WWW) {
  armnn::ConfigureLogging(true, true, armnn::LogSeverity::Trace);
    armnnTfLiteParser::ITfLiteParser::TfLiteParserOptions parserOption;
    parserOption.m_StandInLayerForUnsupported = true;
    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create(parserOption);
    armnn::INetworkPtr network =
      // parser->CreateNetworkFromBinaryFile("/home/heqichen/workspace/tinypilot/mlutils/modelgen/models_py/saved_model/2_float32.tflite");
      parser->CreateNetworkFromBinaryFile(
        "/home/heqichen/workspace/tinypilot/mlutils/modelgen/models_py/output.tflite/output_float32.tflite");


    // Create ArmNN runtime
    armnn::IRuntime::CreationOptions options;    // default options
    // options.m_EnableGpuProfiling = false;
    // options.m_ProfilingOptions.m_EnableProfiling = false;
    // options.m_ProfilingOptions.m_TimelineEnabled = false;
    // options.m_ProfilingOptions.m_OutgoingCaptureFile = std::string("armnn_out.bin");
    // options.m_ProfilingOptions.m_IncomingCaptureFile = std::string("armnn_in.bin");
    // options.m_BackendOptions.emplace_back(
    //          armnn::BackendOptions{"CpuRef",
    //            {
    //              {"TuningLevel", 3},
    //              {"TuningFile", "output.test.prof"},

    //            }
    //          });


    armnn::IRuntimePtr run = armnn::IRuntime::Create(options);

    std::vector<std::string> inputNames = parser->GetSubgraphInputTensorNames(0);
    std::vector<std::string> outputNames = parser->GetSubgraphOutputTensorNames(0);

    std::size_t subgraphCount = parser->GetSubgraphCount();
    std::printf("Subgraph count: %zu\n", subgraphCount);
    std::printf("Input names:\r\n");

    for (const auto& name : inputNames) {
        std::printf("%s: ", name.c_str());
        armnn::BindingPointInfo inputBinding = parser->GetNetworkInputBindingInfo(0, name);
        printf("id=%d, ", inputBinding.first);
        armnn::TensorShape shape = inputBinding.second.GetShape();
        unsigned int dimNum = shape.GetNumDimensions();
        for (unsigned int i = 0; i < dimNum; ++i) {
            std::printf("%u ", shape[i]);
        }
        printf("\r\n");
    }

    std::printf("\nOutput names:\r\n");
    for (const auto& name : outputNames) {
        std::printf("%s: ", name.c_str());
        armnn::BindingPointInfo outputBinding = parser->GetNetworkOutputBindingInfo(0, name);
        printf("id=%d, ", outputBinding.first);
        armnn::TensorShape shape = outputBinding.second.GetShape();
        unsigned int dimNum = shape.GetNumDimensions();
        for (unsigned int i = 0; i < dimNum; ++i) {
            std::printf("%u ", shape[i]);
        }
        printf("\r\n");
    }

    // Optimise ArmNN network
    // std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef, armnn::Compute::CpuAcc,
    // armnn::Compute::GpuAcc};
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    armnn::IOptimizedNetworkPtr optNet = Optimize(*network, backends, run->GetDeviceSpec());
    if (!optNet) {
        // This shouldn't happen for this simple sample, with reference backend.
        // But in general usage Optimize could fail if the hardware at runtime cannot
        // support the model that has been provided.
        std::cerr << "Error: Failed to optimise the input network." << std::endl;
        throw 1;
    }

    // Load graph into runtime
    armnn::NetworkId networkIdentifier;
    run->LoadNetwork(networkIdentifier, std::move(optNet));
}

}    // namespace
}    // namespace ml
}    // namespace cooboc