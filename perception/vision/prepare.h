#ifndef __PERCEPTION_VISION_PREPARE_H__
#define __PERCEPTION_VISION_PREPARE_H__

#include <cstdint>

namespace cooboc {
namespace perception {
namespace vision {

void prepare(std::uint8_t const* const video_frame,
             std::size_t const width,
             std::size_t const height,
             std::uint8_t* const image_data);

}    // namespace vision
}    // namespace perception
}    // namespace cooboc
#endif
