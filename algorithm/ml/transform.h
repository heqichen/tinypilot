#ifndef __ALGORIGHTM_CL_TRANSFORM_H__
#define __ALGORIGHTM_CL_TRANSFORM_H__

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdint>

namespace cooboc {
namespace algorithm {
namespace ml {

struct TransformParameter {
    float scale;
    float offsetX;
    float offsetY;
};

constexpr TransformParameter makeTransformParameter(std::size_t inputWdith,
                                                    std::size_t inputHeight,
                                                    std::size_t outputWidth,
                                                    std::size_t outputHeight) {
    float inputAspect = float(inputWdith) / float(inputHeight);
    float outputAspect = float(outputWidth) / float(outputHeight);
    float scale = (inputAspect > outputAspect) ? (float(inputHeight) / float(outputHeight))
                                               : (float(inputWdith) / float(outputWidth));
    float offsetX = (inputWdith - outputWidth * scale) / 2.0F;
    float offsetY = (inputHeight - outputHeight * scale) / 2.0F;
    return {scale, offsetX, offsetY};
}


void transform(const std::uint8_t *videoFrame,
               const std::size_t width,
               const std::size_t height,
               const TransformParameter &transformParameter,
               std::uint8_t *transformedFrame);
}    // namespace ml
}    // namespace algorithm
}    // namespace cooboc


#endif