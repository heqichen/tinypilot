#ifndef __PERCEPTION_VISION_ML_MODEL_RUNNER_H__
#define __PERCEPTION_VISION_ML_MODEL_RUNNER_H__

#include <array>
#include <cstdint>


namespace cooboc {
namespace perception {
namespace vision {
namespace ml {

void run(std::array<std::uint8_t, 12 * 128 * 256> images, std::array<std::uint8_t, 12 * 128 * 256> bigImages);

}    // namespace ml
}    // namespace vision
}    // namespace perception
}    // namespace cooboc

#endif