#ifndef __DATADEFS_VISION_FRAME_H__
#define __DATADEFS_VISION_FRAME_H__

#include <cstdint>

namespace cooboc {
namespace datadefs {
namespace vision {

struct Frame {
    int width;
    int height;
    // int stride;  // in bytes
    std::uint8_t* data;
};

}    // namespace vision
}    // namespace datadefs
}    // namespace cooboc

#endif