#include "perception/vision/prepare.h"
#include <array>
#include <cstdint>
#include "algorithm/cl/transform.h"

namespace cooboc {
namespace perception {
namespace vision {

void prepare(std::uint8_t const* const video_frame,
             std::size_t const width,
             std::size_t const height,
             std::uint8_t* const image_data) {
    // Transfrom big image into small
    const algorithm::cl::TransformParameter transParam {
      algorithm::cl::makeTransformParameter(width, height, 512U, 256U)};
    std::array<std::uint8_t, 512U * 256U * 3U / 2U> resizedFrame {};
    algorithm::cl::transform(video_frame, width, height, transParam, resizedFrame.data());

    //
}
}    // namespace vision
}    // namespace perception
}    // namespace cooboc