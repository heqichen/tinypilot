#include "ml/policy_runner.h"
#include <armnn/IRuntime.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <array>
#include <cstdio>
#include "datadefs/ml/policy_input.h"
#include "datadefs/ml/policy_output.h"
#include "ml/model_runtime.h"
#include "model_runtime.h"

namespace cooboc {
namespace ml {

PolicyRunner::PolicyRunner(const char *tfliteFilepath) {
  printf("load model: [%s]\r\n", tfliteFilepath);
    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(tfliteFilepath);

    const std::vector<armnn::BackendId>& backendPreferences {armnn::Compute::CpuRef};

    // Optimise ArmNN network
    armnn::IOptimizedNetworkPtr optNet =
      Optimize(*network, backendPreferences, ModelRuntime::getRuntime()->GetDeviceSpec());

    std::printf(
      "WARNING: Use CPU now, please change to GPU in production. %s:%d [%s()]\r\n", __FILE__, __LINE__, __FUNCTION__);
    if (!optNet) {
        // This shouldn't happen for this simple sample, with reference backend.
        // But in general usage Optimize could fail if the hardware at runtime cannot
        // support the model that has been provided.
        std::cerr << "Error: Failed to optimise the input network." << std::endl;
        throw 1;
    }
    std::printf("op netwrok \r\n");

    // Load graph into runtime
    ModelRuntime::getRuntime()->LoadNetwork(networkIdentifier_, std::move(optNet));

    // Load graph input and output binding
    desireBinding_ = parser->GetNetworkInputBindingInfo(0, "desire");
    trafficConventionBinding_ = parser->GetNetworkInputBindingInfo(0, "traffic_convention");
    lateralControlParamsBinding_ = parser->GetNetworkInputBindingInfo(0, "lateral_control_params");
    prevDesiredCurvBinding_ = parser->GetNetworkInputBindingInfo(0, "prev_desired_curv");
    featuresBufferBinding_ = parser->GetNetworkInputBindingInfo(0, "features_buffer");

    outputBinding_ = parser->GetNetworkOutputBindingInfo(0, "Identity");

    assert(sizeof(datadef::ml::PolicyOutput) == 5884 * sizeof(float));
}

void PolicyRunner::run(const datadef::ml::PolicyInput &input, datadef::ml::PolicyOutput &output) {
    // Put data into tensor
    const armnn::InputTensors inputTensors {
      {desireBinding_.first, armnn::ConstTensor(desireBinding_.second, input.desire)},
      {trafficConventionBinding_.first, armnn::ConstTensor(trafficConventionBinding_.second, input.trafficConvention)},
      {lateralControlParamsBinding_.first,
       armnn::ConstTensor(lateralControlParamsBinding_.second, input.lateralControlParams)},
      {prevDesiredCurvBinding_.first, armnn::ConstTensor(prevDesiredCurvBinding_.second, input.prevDesiredCurv)},
      {featuresBufferBinding_.first, armnn::ConstTensor(featuresBufferBinding_.second, input.featuresBuffer)}};

    armnn::OutputTensors outputTensors {
      {outputBinding_.first, armnn::Tensor(outputBinding_.second, &output)},
    };

    // Execute network
    ModelRuntime::getRuntime()->EnqueueWorkload(networkIdentifier_, inputTensors, outputTensors);
}

}    // namespace ml
}    // namespace cooboc