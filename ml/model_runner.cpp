#include "ml/model_runner.h"
#include <armnn/IRuntime.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>

namespace cooboc {
namespace ml {

ModelRunner::ModelRunner(const char *tfliteFilepath) {
    // Load model
    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(tfliteFilepath);

    // Create ArmNN runtime
    armnn::IRuntime::CreationOptions options;    // default options
    armnn::IRuntimePtr run = armnn::IRuntime::Create(options);

    // Optimise ArmNN network
    armnn::IOptimizedNetworkPtr optNet = Optimize(*network, {armnn::Compute::GpuAcc}, run->GetDeviceSpec());
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

}    // namespace ml
}    // namespace cooboc