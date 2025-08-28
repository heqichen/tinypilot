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
void reorderImageLayout(const std::uint8_t* src, const std::size_t width, const std::size_t height, std::uint8_t* dst);

}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc

#endif