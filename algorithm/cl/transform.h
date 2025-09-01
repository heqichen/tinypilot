#ifndef __ALGORIGHTM_CL_TRANSFORM_H__
#define __ALGORIGHTM_CL_TRANSFORM_H__

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdint>

namespace cooboc {
namespace algorithm {
namespace cl {
void transform(const std::uint8_t *videoFrame,
               const std::size_t width,
               const std::size_t height,
               std::uint8_t *transformedFrame);
}
}    // namespace algorithm
}    // namespace cooboc


#endif