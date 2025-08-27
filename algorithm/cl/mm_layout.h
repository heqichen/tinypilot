#ifndef __ALGORITHM_CL_MM_LAYOUT_H__
#define __ALGORITHM_CL_MM_LAYOUT_H__

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdint>

namespace cooboc {
namespace algorithm {
namespace cl {

/**
 * from YUV image to SuperCombo
 */
void reorderImageLayout(const std::uint8_t* src, std::uint8_t* dst, int width, int height);

}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc

#endif