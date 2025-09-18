#include "ml/vision_runner.h"
#include <armnn/IRuntime.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <array>
#include <cstdio>
#include "datadefs/ml/vision_output.h"
#include "ml/model_runtime.h"
#include "model_runtime.h"

namespace cooboc {
namespace ml {

VisionRunner::VisionRunner(const char *tfliteFilepath) {
    // Load model
    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(tfliteFilepath);

    // Optimise ArmNN network
    armnn::IOptimizedNetworkPtr optNet =
      Optimize(*network, {armnn::Compute::CpuRef}, ModelRuntime::getRuntime()->GetDeviceSpec());
    std::printf(
      "WARNING: Use CPU now, please change to GPU in production. %s:%d [%s()]\r\n", __FILE__, __LINE__, __FUNCTION__);
    if (!optNet) {
        // This shouldn't happen for this simple sample, with reference backend.
        // But in general usage Optimize could fail if the hardware at runtime cannot
        // support the model that has been provided.
        std::cerr << "Error: Failed to optimise the input network." << std::endl;
        throw 1;
    }

    // Load graph into runtime
    ModelRuntime::getRuntime()->LoadNetwork(networkIdentifier_, std::move(optNet));

    // Load graph input and output binding
    inputImgsBinding_ = parser->GetNetworkInputBindingInfo(0, "input_imgs");
    bigInputImgsBinding_ = parser->GetNetworkInputBindingInfo(0, "big_input_imgs");
    outputBinding_ = parser->GetNetworkOutputBindingInfo(0, "Identity");

    assert(sizeof(datadef::ml::VisionOutput) == 632 * sizeof(float));
}

void VisionRunner::run(const std::array<std::uint8_t, 12 * 128 * 256> &images,
                       const std::array<std::uint8_t, 12 * 128 * 256> &bigImages,
                       datadef::ml::VisionOutput &output) {
    // Put data into tensor
    const armnn::InputTensors inputTensors {
      {inputImgsBinding_.first, armnn::ConstTensor(inputImgsBinding_.second, images.data())},
      {bigInputImgsBinding_.first, armnn::ConstTensor(bigInputImgsBinding_.second, images.data())},
    };

    // float outputBuffer[1U * 632U];
    armnn::OutputTensors outputTensors {
      {outputBinding_.first, armnn::Tensor(outputBinding_.second, &output)},
    };

    // Execute network
    ModelRuntime::getRuntime()->EnqueueWorkload(networkIdentifier_, inputTensors, outputTensors);
}

}    // namespace ml
}    // namespace cooboc