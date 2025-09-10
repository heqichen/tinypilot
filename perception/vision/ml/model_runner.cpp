#include "perception/vision/ml/model_runner.h"
#include <armnn/IRuntime.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <array>
#include <cstdint>
#include <cstdio>

namespace cooboc {
namespace perception {
namespace vision {
namespace ml {

void run(std::array<std::uint8_t, 12 * 128 * 256> images, std::array<std::uint8_t, 12 * 128 * 256> bigImages) {
    // Load model
    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile("models/onnx2py_fp32_float32.tflite");

    // Create ArmNN runtime
    armnn::IRuntime::CreationOptions options;    // default options
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
    armnn::IOptimizedNetworkPtr optNet = Optimize(*network, {armnn::Compute::CpuRef}, run->GetDeviceSpec());
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

    // Creates structures for inputs and outputs.
    // Set the tensors in the network.
    armnn::TensorInfo inputTensorInfo(armnn::TensorShape({1, 128, 256, 12}), armnn::DataType::Float32);

    // Load data
    std::uint8_t inputImgsBuffer[1U * 128U * 256U * 12U];
    std::uint8_t bigInputImgsBuffer[1U * 128U * 256U * 12U];
    std::memset(inputImgsBuffer, 0U, sizeof(inputImgsBuffer));
    std::memset(bigInputImgsBuffer, 0U, sizeof(bigInputImgsBuffer));

    memcpy(inputImgsBuffer, images.data(), 128U * 256U * 12U);
    memcpy(bigInputImgsBuffer, bigImages.data(), 128U * 256U * 12U);


    // Put data into tensor
    armnn::BindingPointInfo inputImgsBinding = parser->GetNetworkInputBindingInfo(0, "input_imgs");
    armnn::BindingPointInfo bigInputImgsBinding = parser->GetNetworkInputBindingInfo(0, "big_input_imgs");
    armnn::InputTensors inputTensors {
      {inputImgsBinding.first, armnn::ConstTensor(inputImgsBinding.second, inputImgsBuffer)},
      {bigInputImgsBinding.first, armnn::ConstTensor(bigInputImgsBinding.second, inputImgsBuffer)},
    };

    float outputBuffer[1U * 632U];
    armnn::BindingPointInfo outputBinding = parser->GetNetworkOutputBindingInfo(0, "Identity");
    armnn::OutputTensors outputTensors {
      {outputBinding.first, armnn::Tensor(outputBinding.second, outputBuffer)},
    };


    // Execute network
    run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    // auto start = std::chrono::high_resolution_clock::now();

    // for (int i=0; i<1000; ++i) {
    //     // Execute network
    //     run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
    // }
    // // Record end time
    // auto end = std::chrono::high_resolution_clock::now();

    // // Calculate duration
    // std::chrono::duration<double, std::milli> duration = end - start;

    // // Print the duration
    // std::cout << "Function execution time: " << duration.count() << " milliseconds" << std::endl;


    for (int i = 0; i < 632; ++i) {
        std::printf("%f ", outputBuffer[i]);
    }
    std::printf("\r\n");
}

}    // namespace ml
}    // namespace vision
}    // namespace perception
}    // namespace cooboc
