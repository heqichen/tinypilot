#ifndef __ALGORIGHTM_CL_TRANSFORM_H__
#define __ALGORIGHTM_CL_TRANSFORM_H__

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdint>

namespace cooboc {
namespace algorithm {
namespace cl {

struct TransformParameter {
    float scale;
    float offsetX;
    float offsetY;
};

TransformParameter makeTransformParameter(std::size_t inputWdith,
                                          std::size_t inputHeight,
                                          std::size_t outputWidth,
                                          std::size_t outputHeight);


void transform(const std::uint8_t *videoFrame,
               const std::size_t width,
               const std::size_t height,
               std::uint8_t *transformedFrame);
}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc


#endif